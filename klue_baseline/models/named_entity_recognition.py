import argparse
import logging
from typing import Any, Dict, List, Union

import torch
from overrides import overrides
from transformers import AutoModelForTokenClassification

from klue_baseline.data import check_tokenizer_type
from klue_baseline.models import BaseTransformer, Mode

logger = logging.getLogger(__name__)


class NERTransformer(BaseTransformer):

    mode = Mode.NamedEntityRecognition

    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace], metrics: dict = {}) -> None:
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)

        self.tokenizer = hparams.tokenizer
        self.tokenizer_type = check_tokenizer_type(self.tokenizer)  # ["xlm-sp", "bert-wp", "other']
        # When unk, representing subword, is expanded to represent multiple
        # characters to align with character-level labels, this special
        # representation is used to represent characters from the second.
        # (e.g., 찝찝이 [UNK] --> 찝 [UNK] / 찝 [+UNK] / 이 [+UNK])
        self.in_unk_token = "[+UNK]"

        super().__init__(
            hparams,
            num_labels=hparams.num_labels,
            mode=self.mode,
            model_type=AutoModelForTokenClassification,
            metrics=metrics,
        )

    @overrides
    def forward(self, **inputs: torch.Tensor) -> Any:
        return self.model(**inputs)

    @overrides
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]
        outputs = self(**inputs)
        loss = outputs[0]

        self.log("train/loss", loss)
        return {"loss": loss}

    @overrides
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid") -> dict:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]

        outputs = self(**inputs)
        loss, logits = outputs[:2]

        self.log(f"{data_type}/loss", loss, on_step=False, on_epoch=True, logger=True)

        return {"logits": logits, "labels": inputs["labels"]}

    @overrides
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        """When validation step ends, either token- or character-level predicted
        labels are aligned with the original character-level labels and then
        evaluated.
        """
        list_of_subword_preds = self._convert_outputs_to_preds(outputs)
        if self.tokenizer_type == "xlm-sp":
            strip_char = "▁"
        elif self.tokenizer_type == "bert-wp":
            strip_char = "##"
        else:
            raise ValueError("This code only supports XLMRobertaTokenizer & BertWordpieceTokenizer")

        original_examples = self.hparams.data[data_type]["original_examples"]
        list_of_character_preds = []
        list_of_originals = []
        label_list = self.hparams.label_list

        for i, (subword_preds, example) in enumerate(zip(list_of_subword_preds, original_examples)):
            original_sentence = example["original_sentence"]  # 안녕 하세요 ^^
            character_preds = [subword_preds[0].tolist()]  # [CLS]
            character_preds_idx = 1
            for word in original_sentence.split(" "):  # ['안녕', '하세요', '^^']
                if character_preds_idx >= self.hparams.max_seq_length - 1:
                    break
                subwords = self.tokenizer.tokenize(word)  # 안녕 -> [안, ##녕] / 하세요 -> [하, ##세요] / ^^ -> [UNK]
                if self.tokenizer.unk_token in subwords:  # 뻥튀기가 필요한 case!
                    unk_aligned_subwords = self.tokenizer_out_aligner(
                        word, subwords, strip_char
                    )  # [UNK] -> [UNK, +UNK]
                    unk_flag = False
                    for subword in unk_aligned_subwords:
                        if character_preds_idx >= self.hparams.max_seq_length - 1:
                            break
                        subword_pred = subword_preds[character_preds_idx].tolist()
                        subword_pred_label = label_list[subword_pred]
                        if subword == self.tokenizer.unk_token:
                            unk_flag = True
                            character_preds.append(subword_pred)
                            continue
                        elif subword == self.in_unk_token:
                            if subword_pred_label == "O":
                                character_preds.append(subword_pred)
                            else:
                                _, entity_category = subword_pred_label.split("-")
                                character_pred_label = "I-" + entity_category
                                character_pred = label_list.index(character_pred_label)
                                character_preds.append(character_pred)
                            continue
                        else:
                            if unk_flag:
                                character_preds_idx += 1
                                subword_pred = subword_preds[character_preds_idx].tolist()
                                character_preds.append(subword_pred)
                                unk_flag = False
                            else:
                                character_preds.append(subword_pred)
                                character_preds_idx += 1  # `+UNK`가 끝나는 시점에서도 += 1 을 해줘야 다음 label로 넘어감
                else:
                    for subword in subwords:
                        if character_preds_idx >= self.hparams.max_seq_length - 1:
                            break
                        subword = subword.replace(strip_char, "")  # xlm roberta: "▁" / others "##"
                        subword_pred = subword_preds[character_preds_idx].tolist()
                        subword_pred_label = label_list[subword_pred]
                        for i in range(0, len(subword)):  # 안, 녕
                            if i == 0:
                                character_preds.append(subword_pred)
                            else:
                                if subword_pred_label == "O":
                                    character_preds.append(subword_pred)
                                else:
                                    _, entity_category = subword_pred_label.split("-")
                                    character_pred_label = "I-" + entity_category
                                    character_pred = label_list.index(character_pred_label)
                                    character_preds.append(character_pred)
                        character_preds_idx += 1
            character_preds.append(subword_preds[-1].tolist())  # [SEP] label
            list_of_character_preds.extend(character_preds)
            original_labels = ["O"] + example["original_clean_labels"][: len(character_preds) - 2] + ["O"]
            originals = []
            for label in original_labels:
                originals.append(label_list.index(label))
            assert len(character_preds) == len(originals)
            list_of_originals.extend(originals)

        self._set_metrics_device()

        if write_predictions is True:
            self.predictions = list_of_character_preds

        for k, metric in self.metrics.items():
            metric(list_of_character_preds, list_of_originals, label_list)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def tokenizer_out_aligner(self, t_in: str, t_out: List[str], strip_char: str = "##") -> List[str]:
        """Aligns with character-level labels after tokenization.

        Example:
            >>> t_in = "베쏭이,제5원소"
            >>> t_out = ['[UNK]', ',', '제', '##5', '##원', '##소']
            >>> tokenizer_out_aligner(t_in, t_out, strip_char="##")
            ['[UNK]', '[+UNK]', '[+UNK]', ',', '제', '##5', '##원', '##소']

            >>> t_in = "미나藤井美菜27가"
            >>> t_out = ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
            >>> tokenizer_out_aligner(t_in, t_out, strip_char="##")
            ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
        """
        t_out_new = []
        i, j = 0, 0
        UNK_flag = False
        while True:
            if i == len(t_in) and j == len(t_out) - 1:
                break
            step_t_out = len(t_out[j].replace(strip_char, "")) if t_out[j] != self.tokenizer.unk_token else 1
            if UNK_flag:
                t_out_new.append(self.in_unk_token)
            else:
                t_out_new.append(t_out[j])
            if j < len(t_out) - 1 and t_out[j] == self.tokenizer.unk_token and t_out[j + 1] != self.tokenizer.unk_token:
                i += step_t_out
                UNK_flag = True
                if t_in[i] == t_out[j + 1][0]:
                    j += 1
                    UNK_flag = False
            else:
                i += step_t_out
                j += 1
                UNK_flag = False
            if j == len(t_out):
                UNK_flag = True
                j -= 1
        return t_out_new

    @overrides
    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, axis=2)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        BaseTransformer.add_specific_args(parser, root_dir)
        return parser
