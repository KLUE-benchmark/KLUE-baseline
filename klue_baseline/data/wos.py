import argparse
import dataclasses
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import DataProcessor, KlueDataModule

logger = logging.getLogger(__name__)


@dataclass
class WoSInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""

        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass
class WoSInputFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]
    label: Optional[List[str]] = None


class WoSDataset(Dataset):
    def __init__(self, features: List[Union[torch.Tensor, str, List[str]]]) -> None:
        self.features = features
        self.length = len(self.features)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, str, List[str]]:
        return self.features[idx]


class WoSDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        super().__init__()
        self.hparams = args
        self.processor = processor

    def prepare_dataset(self, dataset_type: str) -> Any:
        """Called to initialize data. Use the call to construct features and dataset."""

        logger.info("Creating features from dataset file at %s", self.hparams.data_dir)

        if dataset_type == "train":
            dataset = self.processor.get_train_dataset(self.hparams.data_dir, self.hparams.train_file_name)
        elif dataset_type == "dev":
            dataset = self.processor.get_dev_dataset(self.hparams.data_dir, self.hparams.dev_file_name)
        elif dataset_type == "test":
            dataset = self.processor.get_test_dataset(self.hparams.data_dir, self.hparams.test_file_name)
        else:
            raise ValueError(f"{dataset_type} do not support. [train|dev|test]")
        logger.info(f"Prepare {dataset_type} dataset (Count: {len(dataset)}) ")
        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.prepare_dataset(dataset_type),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=self.processor.collate_fn,
        )

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, rood_dir: str) -> argparse.ArgumentParser:
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            help="The input data dir",
        )
        parser.add_argument(
            "--train_file_name",
            default=None,
            type=str,
            help="Name of the train file",
        )
        parser.add_argument(
            "--dev_file_name",
            default=None,
            type=str,
            help="Name of the dev file",
        )
        parser.add_argument(
            "--test_file_name",
            default=None,
            type=str,
            help="Name of the test file",
        )
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        return parser


class WoSProcessor(DataProcessor):

    origin_train_file_name = "wos-v1.1_train.json"
    origin_dev_file_name = "wos-v1.1_dev.json"
    origin_test_file_name = "wos-v1.1_test.json"

    datamodule_type = WoSDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

        self.tokenizer = tokenizer
        self.slot_meta: List[str] = []
        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes": 3, "no": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: str = None) -> Dataset:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: str = None) -> Dataset:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "dev")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: str = None) -> Dataset:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        if not os.path.exists(file_path):
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> None:
        pass

    def _create_dataset(self, file_path: str, dataset_type: str) -> Dataset:

        # Read ontology file if exists and store the slots
        if self.hparams.ontology_path:
            _, self.slot_meta = self.build_slot_from_ontology(self.hparams.ontology_path)

        # Extract slots from a given dialogue and merge with ontology slots
        with open(file_path, "r", encoding="utf-8") as dial_file:
            dials = json.load(dial_file)
        slot_from_dials = self.build_slot_meta(dials)
        self.slot_meta = self.merge_slot_meta(slot_from_dials)

        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples, dataset_type)

        """
        input_ids = torch.LongTensor(self.pad_ids([f.input_id for f in features], self.tokenizer.pad_token_id))
        segment_ids = torch.LongTensor(self.pad_ids([f.segment_id for f in features], self.tokenizer.pad_token_id))
        input_masks = input_ids.ne(self.tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([f.gating_id for f in features])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(f.target_ids) for f in features], self.tokenizer.pad_token_id
        )
        return TensorDataset(input_ids, segment_ids, input_masks, gating_ids, target_ids, guids)
        """

        return WoSDataset(features)

    def _create_examples(self, file_path: str, dataset_type: str) -> List[WoSInputExample]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for dialogue in data:
                dialogue_examples = self.get_examples_from_dialogue(dialogue)
                examples.extend(dialogue_examples)
        return examples

    def _convert_features(self, examples: List[WoSInputExample], dataset_type: str) -> List[WoSInputFeature]:
        features = []
        for example in examples:
            feature = self._convert_example_to_feature(example, dataset_type)
            if feature:
                features.append(feature)
        return features

    def _convert_example_to_feature(self, example: WoSInputExample, dataset_type: str) -> WoSInputFeature:
        dialogue_context = example.context_turns + [self.tokenizer.sep_token] + example.current_turn

        input_id = self.tokenizer.convert_tokens_to_ids(dialogue_context)
        len_input_id = len(input_id)
        if len_input_id > self.hparams.max_seq_length - 2:
            if dataset_type == "train" and not self.hparams.truncate:
                """Skip this training data which is longer than max_seq_length"""
                logger.info(
                    f"[{dataset_type}] Skip the context [{example.guid}] "
                    f"since the length of dialogue exceeds {self.hparams.max_seq_length - 2} < {len_input_id}"
                )
                return None  # type: ignore[return-value]
            else:
                input_id = input_id[len_input_id - (self.hparams.max_seq_length - 2) :]
                logger.info(
                    f"[{dataset_type}] Truncate the context [{example.guid}] "
                    f"since the length of dialogue exceeds {self.hparams.max_seq_length - 2} < {len_input_id}"
                )
        input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id]
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        state = self.convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.tokenizer.encode(value, add_special_tokens=False)
            len_target_id = len(target_id)
            if len_target_id > self.hparams.max_seq_length - 1:
                if dataset_type == "train" and not self.hparams.truncate:
                    """Skip this training data which is longer than max_seq_length"""
                    logger.info(
                        f"[{dataset_type}] Skip the slot [{value}] "
                        f"since the length of slot exceeds {self.hparams.max_seq_length - 1} < {len_target_id}"
                    )
                    return None  # type: ignore[return-value]
                else:
                    target_id = target_id[len_target_id - (self.hparams.max_seq_length - 1) :]
                    logger.info(
                        f"[{dataset_type}] Truncate the slot [{value}] "
                        f"since the length of slot exceeds {self.hparams.max_seq_length - 1} < {len_target_id}"
                    )
            target_id = target_id + [self.tokenizer.sep_token_id]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.tokenizer.pad_token_id)

        return WoSInputFeature(example.guid, input_id, segment_id, gating_id, target_ids, example.label)

    @staticmethod
    def pad_ids(arrays: List[List[int]], pad_idx: int, max_length: int = -1) -> List[List[int]]:
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    @staticmethod
    def pad_id_of_matrix(arrays: torch.Tensor, pad_idx: int, max_length: int = -1, left: bool = False) -> torch.Tensor:
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, length = array.size()
            pad = torch.zeros(n, (max_length - length))
            pad[
                :,
                :,
            ] = pad_idx
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def get_examples_from_dialogue(self, dialogue: Dict[str, List[Dict]]) -> List[WoSInputExample]:
        dialogue_id = dialogue["guid"]
        examples = []
        history: List[str] = []
        d_idx = 0
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            if idx:
                sys_utter = dialogue["dialogue"][idx - 1]["text"]
            else:
                sys_utter = ""

            sys_utter = self.tokenizer.tokenize(sys_utter)
            user_utter = self.tokenizer.tokenize(turn["text"])
            state = turn["state"]
            context = deepcopy(history)
            examples.append(
                WoSInputExample(
                    guid=f"{dialogue_id}-{d_idx}",
                    context_turns=context,
                    current_turn=sys_utter + [self.tokenizer.sep_token] + user_utter,
                    label=state,
                )
            )
            if history:
                history.append(self.tokenizer.sep_token)
            history.extend(sys_utter)
            history.append(self.tokenizer.sep_token)
            history.extend(user_utter)
            d_idx += 1
        return examples

    def merge_slot_meta(self, slot_from_dial: List[str]) -> List[str]:
        exist_slot_set = set(self.slot_meta)
        for slot in slot_from_dial:
            exist_slot_set.add(slot)
        return sorted(list(exist_slot_set))

    @staticmethod
    def build_slot_from_ontology(file_path: str) -> Tuple[List[str], List[str]]:
        """Read ontology file: expected format is `DOMAIN-SLOT`"""

        domains = []
        slots = []
        with open(file_path, "r", encoding="utf-8") as ontology_file:
            for line in ontology_file:
                domain_slot = line.split("-")
                assert len(domain_slot) == 2
                domains.append(domain_slot[0])
                slots.append(line)
        return domains, slots

    def build_slot_meta(self, data: List[Dict[str, List[dict]]]) -> List[str]:
        slot_meta = []
        for dialog in data:
            for turn in dialog["dialogue"]:
                if not turn.get("state"):
                    continue
                for dom_slot_value in turn["state"]:
                    domain_slot, _ = self.split_slot(dom_slot_value, get_domain_slot=True)
                    if domain_slot not in slot_meta:
                        slot_meta.append(domain_slot)
        return sorted(slot_meta)

    @staticmethod
    def split_slot(dom_slot_value: str, get_domain_slot: bool = False) -> Tuple[str, ...]:
        try:
            dom, slot, value = dom_slot_value.split("-")
        except ValueError:
            tempo = dom_slot_value.split("-")
            if len(tempo) < 2:
                return dom_slot_value, dom_slot_value, dom_slot_value
            dom, slot = tempo[0], tempo[1]
            value = dom_slot_value.replace("%s-%s-" % (dom, slot), "").strip()

        if get_domain_slot:
            return "%s-%s" % (dom, slot), value
        return dom, slot, value

    def recover_state(self, gate_list: List[int], gen_list: List[List[int]]) -> List[str]:
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue
            elif self.id2gating[gate] == "dontcare":
                recovered.append("%s-%s" % (slot, "dontcare"))
                continue
            elif self.id2gating[gate] == "yes":
                recovered.append("%s-%s" % (slot, "yes"))
                continue
            elif self.id2gating[gate] == "no":
                recovered.append("%s-%s" % (slot, "no"))
                continue
            elif self.id2gating[gate] == "ptr":
                # Append a token until special tokens appear
                token_id_list = []
                for id_ in value:
                    if id_ in self.tokenizer.all_special_ids:
                        break
                    token_id_list.append(id_)
                value = self.tokenizer.decode(token_id_list, skip_special_tokens=True)
                # This is a basic post-processing for generative DST models based on wordpiece (using punctuation split)
                value = value.replace(" : ", ":").replace(" , ", ", ").replace("##", "")
            else:
                raise ValueError(f"{self.id2gating[gate]} do not support. [none|dontcare|ptr|yes|no]")

            if value == "none":  # type: ignore[comparison-overlap]
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def convert_state_dict(self, state: Sequence[str]) -> Dict[str, str]:
        dic = {}
        for slot in state:
            s, v = self.split_slot(slot, get_domain_slot=True)
            dic[s] = v
        return dic

    def collate_fn(
        self, batch: List[WoSInputFeature]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[Optional[List[str]]]
    ]:
        input_ids = torch.LongTensor(self.pad_ids([b.input_id for b in batch], self.tokenizer.pad_token_id))
        segment_ids = torch.LongTensor(self.pad_ids([b.segment_id for b in batch], self.tokenizer.pad_token_type_id))
        input_masks = input_ids.ne(self.tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix([torch.LongTensor(b.target_ids) for b in batch], self.tokenizer.pad_token_id)
        guids = [b.guid for b in batch]
        labels = [b.label for b in batch]
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids, labels

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--ontology_path",
            default="",
            type=str,
        )
        parser.add_argument(
            "--truncate",
            action="store_true",
            help="truncate left-side of context in training data rather than skipping the context",
        )
        return parser
