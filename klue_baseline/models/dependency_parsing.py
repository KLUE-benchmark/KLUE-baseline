import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

from klue_baseline.data.klue_dp import get_dep_labels, get_pos_labels
from klue_baseline.models import BaseTransformer, Mode

logger = logging.getLogger(__name__)


class DPResult:
    """Result object for DataParallel"""

    def __init__(self, heads: torch.Tensor, types: torch.Tensor) -> None:
        self.heads = heads
        self.types = types


class DPTransformer(BaseTransformer):

    mode = Mode.DependencyParsing

    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace], metrics: dict = {}) -> None:
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)

        super().__init__(
            hparams,
            num_labels=None,
            mode=self.mode,
            model_type=AutoModel,
            metrics=metrics,
        )

        self.hidden_size = hparams.hidden_size
        self.input_size = self.model.config.hidden_size
        self.arc_space = hparams.arc_space
        self.type_space = hparams.type_space

        self.n_pos_labels = len(get_pos_labels())
        self.n_dp_labels = len(get_dep_labels())

        if hparams.no_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.n_pos_labels + 1, hparams.pos_dim)

        enc_dim = self.input_size * 2
        if self.pos_embedding is not None:
            enc_dim += hparams.pos_dim

        self.encoder = nn.LSTM(
            enc_dim,
            self.hidden_size,
            hparams.encoder_layers,
            batch_first=True,
            dropout=0.33,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            self.hidden_size, self.hidden_size, hparams.decoder_layers, batch_first=True, dropout=0.33
        )

        self.dropout = nn.Dropout2d(p=0.33)

        self.src_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.hx_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.arc_c = nn.Linear(self.hidden_size * 2, self.arc_space)
        self.type_c = nn.Linear(self.hidden_size * 2, self.type_space)
        self.arc_h = nn.Linear(self.hidden_size, self.arc_space)
        self.type_h = nn.Linear(self.hidden_size, self.type_space)

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)
        self.bilinear = BiLinear(self.type_space, self.type_space, self.n_dp_labels)

    @overrides
    def forward(
        self,
        bpe_head_mask: torch.Tensor,
        bpe_tail_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        head_ids: torch.Tensor,
        max_word_length: int,
        mask_e: torch.Tensor,
        mask_d: torch.Tensor,
        batch_index: torch.Tensor,
        is_training: bool = True,
        **inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.model(**inputs)
        outputs = outputs[0]
        outputs, sent_len = self.resize_outputs(outputs, bpe_head_mask, bpe_tail_mask, max_word_length)

        if self.pos_embedding is not None:
            pos_outputs = self.pos_embedding(pos_ids)
            pos_outputs = self.dropout(pos_outputs)
            outputs = torch.cat([outputs, pos_outputs], dim=2)

        # encoder
        packed_outputs = pack_padded_sequence(outputs, sent_len, batch_first=True, enforce_sorted=False)
        encoder_outputs, hn = self.encoder(packed_outputs)
        encoder_outputs, outputs_len = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer
        hn = self._transform_decoder_init_state(hn)

        # decoder
        src_encoding = F.elu(self.src_dense(encoder_outputs[:, 1:]))
        sent_len = [i - 1 for i in sent_len]
        packed_outputs = pack_padded_sequence(src_encoding, sent_len, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.decoder(packed_outputs, hn)
        decoder_outputs, outputs_len = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_outputs = self.dropout(decoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer

        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))
        type_c = F.elu(self.type_c(encoder_outputs))

        arc_h = F.elu(self.arc_h(decoder_outputs))
        type_h = F.elu(self.type_h(decoder_outputs))

        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)

        # use predicted head_ids when validation step
        if not is_training:
            head_ids = torch.argmax(out_arc, dim=2)

        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type

    @overrides
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        input_ids, masks, ids, max_word_length = batch
        attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = masks
        head_ids, type_ids, pos_ids = ids
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_size = head_ids.size()[0]
        batch_index = torch.arange(0, int(batch_size)).long()
        head_index = (
            torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
        )

        # forward
        out_arc, out_type = self(
            bpe_head_mask, bpe_tail_mask, pos_ids, head_ids, max_word_length, mask_e, mask_d, batch_index, **inputs
        )

        # compute loss
        minus_inf = -1e8
        minus_mask_d = (1 - mask_d) * minus_inf
        minus_mask_e = (1 - mask_e) * minus_inf
        out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=2)
        loss_type = F.log_softmax(out_type, dim=2)

        loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
        loss_type = loss_type * mask_d.unsqueeze(2)
        num = mask_d.sum()

        loss_arc = loss_arc[batch_index, head_index, head_ids.data.t()].transpose(0, 1)
        loss_type = loss_type[batch_index, head_index, type_ids.data.t()].transpose(0, 1)
        loss_arc = -loss_arc.sum() / num
        loss_type = -loss_type.sum() / num
        loss = loss_arc + loss_type

        self.log("train/loss_arc", loss_arc)
        self.log("train/loss_type", loss_type)
        self.log("train/loss", loss)

        return {"loss": loss}

    @overrides
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid") -> dict:
        input_ids, masks, ids, max_word_length = batch
        attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = masks
        head_ids, type_ids, pos_ids = ids
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_index = torch.arange(0, head_ids.size()[0]).long()

        out_arc, out_type = self(
            bpe_head_mask,
            bpe_tail_mask,
            pos_ids,
            head_ids,
            max_word_length,
            mask_e,
            mask_d,
            batch_index,
            is_training=False,
            **inputs,
        )

        # predict arc and its type
        heads = torch.argmax(out_arc, dim=2)
        types = torch.argmax(out_type, dim=2)

        preds = DPResult(heads, types)
        labels = DPResult(head_ids, type_ids)

        return {"preds": preds, "labels": labels}

    @overrides
    def validation_epoch_end(
        self, outputs: List[Dict[str, DPResult]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        all_preds = []
        all_labels = []
        for output in zip(outputs):
            all_preds.append(output[0]["preds"])
            all_labels.append(output[0]["labels"])

        if write_predictions is True:
            self.write_prediction_file(all_preds, all_labels)

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(all_preds, all_labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def write_prediction_file(self, prs: List[DPResult], gts: List[DPResult]) -> None:
        """Write head, head type predictions and corresponding labels to json file. Each line indicates a word."""
        head_preds, type_preds, head_labels, type_labels = self._flatten_prediction_and_labels(prs, gts)
        save_path = self.output_dir.joinpath("transformers/pred")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"pred-{self.step_count}.json"), "w", encoding="utf-8") as f:
            for h, t, hl, tl in zip(head_preds, type_preds, head_labels, type_labels):
                f.write(" ".join([str(h), str(t), str(hl), str(tl)]) + "\n")

    def _flatten_prediction_and_labels(
        self, preds: List[DPResult], labels: List[DPResult]
    ) -> Tuple[List, List, List, List]:
        """Convert prediction and labels to np.array and remove -1s."""
        head_pred_list = list()
        head_label_list = list()
        type_pred_list = list()
        type_label_list = list()
        for pred, label in zip(preds, labels):
            head_pred_list += pred.heads.cpu().flatten().tolist()
            head_label_list += label.heads.cpu().flatten().tolist()
            type_pred_list += pred.types.cpu().flatten().tolist()
            type_label_list += label.types.cpu().flatten().tolist()
        head_preds = np.array(head_pred_list)
        head_labels = np.array(head_label_list)
        type_preds = np.array(type_pred_list)
        type_labels = np.array(type_label_list)

        index = [i for i, label in enumerate(head_labels) if label == -1]
        head_preds = np.delete(head_preds, index)
        head_labels = np.delete(head_labels, index)
        index = [i for i, label in enumerate(type_labels) if label == -1]
        type_preds = np.delete(type_preds, index)
        type_labels = np.delete(type_labels, index)

        return (
            head_preds.tolist(),
            type_preds.tolist(),
            head_labels.tolist(),
            type_labels.tolist(),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("transformers")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.config.save_step = self.step_count
        torch.save(self.state_dict(), save_path.joinpath("dp-model.bin"))
        self.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        BaseTransformer.add_specific_args(parser, root_dir)
        parser.add_argument("--encoder_layers", default=1, type=int, help="Number of layers of encoder")
        parser.add_argument("--decoder_layers", default=1, type=int, help="Number of layers of decoder")
        parser.add_argument("--hidden_size", default=768, type=int, help="Number of hidden units in LSTM")
        parser.add_argument("--arc_space", default=512, type=int, help="Dimension of tag space")
        parser.add_argument("--type_space", default=256, type=int, help="Dimension of tag space")
        parser.add_argument("--no_pos", action="store_true", help="Do not use pos feature in head layers")
        parser.add_argument("--pos_dim", default=256, type=int, help="Dimension of pos embedding")
        args = parser.parse_args()
        if not args.no_pos and args.pos_dim <= 0:
            parser.error("--pos_dim should be a positive integer when --no_pos is False.")
        return parser

    def resize_outputs(
        self, outputs: torch.Tensor, bpe_head_mask: torch.Tensor, bpe_tail_mask: torch.Tensor, max_word_length: int
    ) -> Tuple[torch.Tensor, List]:
        """Resize output of pre-trained transformers (bsz, max_token_length, hidden_dim) to word-level outputs (bsz, max_word_length, hidden_dim*2). """
        batch_size, input_size, hidden_size = outputs.size()
        word_outputs = torch.zeros(batch_size, max_word_length + 1, hidden_size * 2).to(outputs.device)
        sent_len = list()

        for batch_id in range(batch_size):
            head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]
            tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]
            assert len(head_ids) == len(tail_ids)

            word_outputs[batch_id][0] = torch.cat(
                (outputs[batch_id][0], outputs[batch_id][0])
            )  # replace root with [CLS]
            for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
                word_outputs[batch_id][i + 1] = torch.cat((outputs[batch_id][head], outputs[batch_id][tail]))
            sent_len.append(i + 2)

        return word_outputs, sent_len

    def _transform_decoder_init_state(self, hn: torch.Tensor) -> torch.Tensor:
        hn, cn = hn
        cn = cn[-2:]  # take the last layer
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)
        cn = self.hx_dense(cn)
        if self.decoder.num_layers > 1:
            cn = torch.cat(
                [
                    cn,
                    torch.autograd.Variable(cn.data.new(self.decoder.num_layers - 1, batch_size, hidden_size).zero_()),
                ],
                dim=0,
            )
        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn


class BiAttention(nn.Module):
    def __init__(  # type: ignore[no-untyped-def]
        self, input_size_encoder: int, input_size_decoder: int, num_labels: int, biaffine: bool = True, **kwargs
    ) -> None:
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(
        self,
        input_d: torch.Tensor,
        input_e: torch.Tensor,
        mask_d: Optional[torch.Tensor] = None,
        mask_e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class BiLinear(nn.Module):
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))
