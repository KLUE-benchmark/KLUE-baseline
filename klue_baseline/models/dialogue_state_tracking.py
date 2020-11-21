import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from transformers import AutoModel

from klue_baseline.models import BaseTransformer, Mode

logger = logging.getLogger(__name__)


class DSTResult:
    """Result object for DataParallel"""

    def __init__(self, prs: List[List[str]], gts: List[List[str]], guids: List[str]) -> None:
        self.prs = prs
        self.gts = gts
        self.guids = guids


class DSTTransformer(BaseTransformer):

    mode = Mode.DialogueStateTracking
    REQUIRE_ADDITIONAL_POOLER_LAYER_BY_TYPE = ["electra"]

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
        self.processor = hparams.processor
        hparams.processor = None

        self.teacher_forcing = self.hparams.teacher_forcing
        self.parallel_decoding = self.hparams.parallel_decoding

        self.slot_meta = self.processor.slot_meta
        self.slot_vocab = [
            self.processor.tokenizer.encode(slot.replace("-", " "), add_special_tokens=False) for slot in self.slot_meta
        ]
        # refer the vars (encoder_config, encoder) in super class (BaseTransformer)
        self.encoder_config = self.config
        self.encoder = self.model

        if self._is_require_pooler_layer():
            from transformers.modeling_bert import BertPooler

            self.encoder_pooler_layer = BertPooler(self.encoder_config)

        self.decoder = SlotGenerator(
            self.encoder_config.vocab_size,
            self.encoder_config.hidden_size,
            self.encoder_config.hidden_dropout_prob,
            self.slot_meta,
            self.processor.gating2id,
            self.processor.tokenizer.pad_token_id,
            self.parallel_decoding,
        )

        self.decoder.set_slot_idx(self.slot_vocab)
        self.tie_weight()

        self.loss_gen = masked_cross_entropy_for_value
        self.loss_gate = nn.CrossEntropyLoss()

        self.metrics = nn.ModuleDict(metrics)

    def tie_weight(self) -> None:
        """Share the embedding layer for both encoder and decoder"""
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: int = 10,
        teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # TODO: Need to be refactored before code release
        outputs_dict = self.encoder(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_outputs = outputs_dict["last_hidden_state"]
        if "pooler_output" in outputs_dict.keys():
            pooler_output = outputs_dict["pooler_output"]
        else:
            pooler_output = self.encoder_pooler_layer(encoder_outputs)

        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids, encoder_outputs, pooler_output.unsqueeze(0), attention_mask, max_len, teacher
        )

        return all_point_outputs, all_gate_outputs

    @overrides
    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids, segment_ids, input_masks, gating_ids, target_ids, _, _ = batch

        if self.teacher_forcing > 0.0 and random.random() < self.teacher_forcing:
            tf = target_ids
        else:
            tf = None

        all_point_outputs, all_gate_outputs = self(input_ids, segment_ids, input_masks, target_ids.size(-1), tf)
        loss_gen = self.loss_gen(
            all_point_outputs.contiguous(), target_ids.contiguous().view(-1), self.tokenizer.pad_token_id
        )
        loss_gate = self.loss_gate(
            all_gate_outputs.contiguous().view(-1, len(self.processor.gating2id.keys())),
            gating_ids.contiguous().view(-1),
        )
        loss = loss_gen + loss_gate

        self.log("train/loss", loss)
        self.log("train/loss_gen", loss_gen)
        self.log("train/loss_gate", loss_gate)

        return {"loss": loss}

    @overrides
    def validation_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int, data_type: str = "valid", write_predictions: bool = False
    ) -> Dict[str, Any]:
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids, labels = batch
        all_point_outputs, all_gate_outputs = self(input_ids, segment_ids, input_masks)

        _, generated_ids = all_point_outputs.max(-1)
        _, gated_ids = all_gate_outputs.max(-1)

        prs = [self.processor.recover_state(gate, gen) for gate, gen in zip(gated_ids.tolist(), generated_ids.tolist())]

        dst_result = DSTResult(prs=prs, gts=labels, guids=guids)
        # val_output_dict = {"prs": prs, "gts": gts}
        # return val_output_dict
        return {"results": dst_result}

    @overrides
    def validation_epoch_end(
        self, outputs: List[Dict[str, List[str]]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        # prs = [output["prs"] for output in outputs]  # B * steps
        # gts = [output["gts"] for output in outputs]
        prs = []
        gts = []
        guids = []
        for output in outputs:
            if type(output["results"]) == list:
                for result in output["results"]:
                    prs += result.prs
                    gts += result.gts
                    guids += result.guids
            else:
                prs += output["results"].prs
                gts += output["results"].gts
                guids += output["results"].guids

        if write_predictions:
            self.write_prediction_file(prs, gts, guids)

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(prs, gts)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def write_prediction_file(
        self, prs: Sequence[Sequence[str]], gts: Sequence[Sequence[str]], guids: Sequence[str]
    ) -> None:
        pred_dict = []
        for pr, gt, guid in zip(prs, gts, guids):
            item = {"guid": guid, "pr": pr, "gt": gt}
            pred_dict.append(item)
        # raw text save
        save_path = self.output_dir.joinpath("transformers/pred")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"pred-{self.step_count}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(pred_dict, indent=4, ensure_ascii=False))

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("transformers")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.config.save_step = self.step_count
        torch.save(self.state_dict(), save_path.joinpath(f"trade-{self.step_count}.bin"))
        self.tokenizer.save_pretrained(save_path)
        self.config.save_pretrained(save_path)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        BaseTransformer.add_specific_args(parser, root_dir)
        parser.add_argument("--teacher_forcing", default=0.5, type=float, help="teacher_forcing")
        parser.add_argument(
            "--parallel_decoding",
            action="store_true",
            help="Decode all slot-values in parallel manner.",
        )
        return parser

    def _is_require_pooler_layer(self) -> bool:
        if self.encoder_config.model_type in set(self.REQUIRE_ADDITIONAL_POOLER_LAYER_BY_TYPE):
            return True
        else:
            return False


class SlotGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float,
        slot_meta: List[str],
        gating2id: Dict[str, int],
        pad_idx: int = 0,
        parallel_decoding: bool = True,
    ) -> None:
        super(SlotGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.slot_meta = slot_meta
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, self.hidden_size, padding_idx=pad_idx)  # shared with encoder

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True)

        # receive gate info from processor
        self.gating2id = gating2id  # {"none": 0, "dontcare": 1, "ptr": 2, "yes":3, "no": 4}
        self.num_gates = len(self.gating2id.keys())

        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, self.num_gates)

        self.slot_embed_idx: List[List[int]] = []
        self.parallel_decoding = parallel_decoding

    def set_slot_idx(self, slot_vocab_idx: List[List[int]]) -> None:
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        hidden: torch.Tensor,
        input_masks: torch.Tensor,
        max_len: int,
        teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)
        # slot_embedding
        slot_e = torch.sum(self.embed(slot), 1)  # J, d
        J = slot_e.size(0)

        if self.parallel_decoding:
            all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(batch_size, J, self.num_gates).to(input_ids.device)

            w = slot_e.repeat(batch_size, 1).unsqueeze(1)
            hidden = hidden.repeat_interleave(J, dim=1)
            encoder_output = encoder_output.repeat_interleave(J, dim=0)
            input_ids = input_ids.repeat_interleave(J, dim=0)
            input_masks = input_masks.repeat_interleave(J, dim=0)
            num_decoding = 1

        else:
            # Seperate Decoding
            all_point_outputs = torch.zeros(J, batch_size, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(J, batch_size, self.num_gates).to(input_ids.device)
            num_decoding = J

        for j in range(num_decoding):

            if not self.parallel_decoding:
                w = slot_e[j].expand(batch_size, 1, self.hidden_size)

            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D

                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
                attn_history = F.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = F.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
                p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)

                if teacher is not None:
                    if self.parallel_decoding:
                        w = self.embed(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                    else:
                        w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D

                if k == 0:
                    gated_logit = self.w_gate(context.squeeze(1))  # B,3
                    if self.parallel_decoding:
                        all_gate_outputs = gated_logit.view(batch_size, J, self.num_gates)
                    else:
                        _, gated = gated_logit.max(1)  # maybe `-1` would be more clear
                        all_gate_outputs[j] = gated_logit

                if self.parallel_decoding:
                    all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)
                else:
                    all_point_outputs[j, :, k, :] = p_final

        if not self.parallel_decoding:
            all_point_outputs = all_point_outputs.transpose(0, 1)
            all_gate_outputs = all_gate_outputs.transpose(0, 1)

        return all_point_outputs, all_gate_outputs


def masked_cross_entropy_for_value(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss
