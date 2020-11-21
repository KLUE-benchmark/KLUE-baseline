import argparse
import logging
import os
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import DataProcessor, KlueDataModule

logger = logging.getLogger(__name__)


class KlueDPInputExample:
    """A single training/test example for Dependency Parsing in .conllu format

    Args:
        guid : Unique id for the example
        text : string. the original form of sentence
        token_id : token id
        token : 어절
        pos : POS tag(s)
        head : dependency head
        dep : dependency relation
    """

    def __init__(
        self, guid: str, text: str, sent_id: int, token_id: int, token: str, pos: str, head: str, dep: str
    ) -> None:
        self.guid = guid
        self.text = text
        self.sent_id = sent_id
        self.token_id = token_id
        self.token = token
        self.pos = pos
        self.head = head
        self.dep = dep


class KlueDPInputFeatures:
    """A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        bpe_head_mask : Mask to mark the head token of bpe in aejeol
        head_ids : head ids for each aejeols on head token index
        dep_ids : dependecy relations for each aejeols on head token index
        pos_ids : pos tag for each aejeols on head token index
    """

    def __init__(
        self,
        guid: str,
        ids: List[int],
        mask: List[int],
        bpe_head_mask: List[int],
        bpe_tail_mask: List[int],
        head_ids: List[int],
        dep_ids: List[int],
        pos_ids: List[int],
    ) -> None:
        self.guid = guid
        self.input_ids = ids
        self.attention_mask = mask
        self.bpe_head_mask = bpe_head_mask
        self.bpe_tail_mask = bpe_tail_mask
        self.head_ids = head_ids
        self.dep_ids = dep_ids
        self.pos_ids = pos_ids


class KlueDPDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        super().__init__()
        self.hparams = args
        self.processor = processor

    def prepare_dataset(self, dataset_type: str) -> Any:
        "Called to initialize data. Use the call to construct features and dataset"
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
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser.add_argument("--data_dir", default=None, type=str, help="The input data dir", required=True)
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


class KlueDPProcessor(DataProcessor):

    origin_train_file_name = "klue-dp-v1.1_train.tsv"
    origin_dev_file_name = "klue-dp-v1.1_dev.tsv"
    origin_test_file_name = "klue-dp-v1.1_test.tsv"

    datamodule_type = KlueDPDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

    def _create_examples(self, file_path: str, dataset_type: str) -> List[KlueDPInputExample]:
        sent_id = -1
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "" or line == "\n" or line == "\t":
                    continue
                if line.startswith("#"):
                    parsed = line.strip().split("\t")
                    if len(parsed) != 2:  # metadata line about dataset
                        continue
                    else:
                        sent_id += 1
                        text = parsed[1].strip()
                        guid = parsed[0].replace("##", "").strip()
                else:
                    token_list = [token.replace("\n", "") for token in line.split("\t")] + ["-", "-"]
                    examples.append(
                        KlueDPInputExample(
                            guid=guid,
                            text=text,
                            sent_id=sent_id,
                            token_id=int(token_list[0]),
                            token=token_list[1],
                            pos=token_list[3],
                            head=token_list[4],
                            dep=token_list[5],
                        )
                    )
        return examples

    def convert_examples_to_features(
        self,
        examples: List[KlueDPInputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        pos_label_list: List[str],
        dep_label_list: List[str],
    ) -> List[KlueDPInputFeatures]:

        pos_label_map = {label: i for i, label in enumerate(pos_label_list)}
        dep_label_map = {label: i for i, label in enumerate(dep_label_list)}

        SENT_ID = 0

        token_list: List[str] = []
        pos_list: List[str] = []
        head_list: List[int] = []
        dep_list: List[str] = []

        features = []
        for example in examples:
            if SENT_ID != example.sent_id:
                SENT_ID = example.sent_id
                encoded = tokenizer.encode_plus(
                    " ".join(token_list),
                    None,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )

                ids, mask = encoded["input_ids"], encoded["attention_mask"]

                bpe_head_mask = [0]
                bpe_tail_mask = [0]
                head_ids = [-1]
                dep_ids = [-1]
                pos_ids = [-1]  # --> CLS token

                for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
                    bpe_len = len(tokenizer.tokenize(token))
                    head_token_mask = [1] + [0] * (bpe_len - 1)
                    tail_token_mask = [0] * (bpe_len - 1) + [1]
                    bpe_head_mask.extend(head_token_mask)
                    bpe_tail_mask.extend(tail_token_mask)

                    head_mask = [head] + [-1] * (bpe_len - 1)
                    head_ids.extend(head_mask)
                    dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
                    dep_ids.extend(dep_mask)
                    pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
                    pos_ids.extend(pos_mask)

                bpe_head_mask.append(0)
                bpe_tail_mask.append(0)
                head_ids.append(-1)
                dep_ids.append(-1)
                pos_ids.append(-1)  # END token
                if len(bpe_head_mask) > max_length:
                    bpe_head_mask = bpe_head_mask[:max_length]
                    bpe_tail_mask = bpe_tail_mask[:max_length]
                    head_ids = head_ids[:max_length]
                    dep_ids = dep_ids[:max_length]
                    pos_ids = pos_ids[:max_length]

                else:
                    bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
                    bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
                    head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
                    dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
                    pos_ids.extend([-1] * (max_length - len(pos_ids)))

                feature = KlueDPInputFeatures(
                    guid=example.guid,
                    ids=ids,
                    mask=mask,
                    bpe_head_mask=bpe_head_mask,
                    bpe_tail_mask=bpe_tail_mask,
                    head_ids=head_ids,
                    dep_ids=dep_ids,
                    pos_ids=pos_ids,
                )
                features.append(feature)

                token_list = []
                pos_list = []
                head_list = []
                dep_list = []

            token_list.append(example.token)
            pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
            head_list.append(int(example.head))
            dep_list.append(example.dep)

        encoded = tokenizer.encode_plus(
            " ".join(token_list),
            None,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        ids, mask = encoded["input_ids"], encoded["attention_mask"]

        bpe_head_mask = [0]
        bpe_tail_mask = [0]
        head_ids = [-1]
        dep_ids = [-1]
        pos_ids = [-1]  # --> CLS token

        for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
            bpe_len = len(tokenizer.tokenize(token))
            head_token_mask = [1] + [0] * (bpe_len - 1)
            tail_token_mask = [0] * (bpe_len - 1) + [1]
            bpe_head_mask.extend(head_token_mask)
            bpe_tail_mask.extend(tail_token_mask)

            head_mask = [head] + [-1] * (bpe_len - 1)
            head_ids.extend(head_mask)
            dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
            dep_ids.extend(dep_mask)
            pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
            pos_ids.extend(pos_mask)

        bpe_head_mask.append(0)
        bpe_tail_mask.append(0)
        head_ids.append(-1)
        dep_ids.append(-1)  # END token
        bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
        bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
        head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
        dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
        pos_ids.extend([-1] * (max_length - len(pos_ids)))

        feature = KlueDPInputFeatures(
            guid=example.guid,
            ids=ids,
            mask=mask,
            bpe_head_mask=bpe_head_mask,
            bpe_tail_mask=bpe_tail_mask,
            head_ids=head_ids,
            dep_ids=dep_ids,
            pos_ids=pos_ids,
        )
        features.append(feature)

        for feature in features[:3]:
            logger.info("*** Example ***")
            logger.info("input_ids: %s" % feature.input_ids)
            logger.info("attention_mask: %s" % feature.attention_mask)
            logger.info("bpe_head_mask: %s" % feature.bpe_head_mask)
            logger.info("bpe_tail_mask: %s" % feature.bpe_tail_mask)
            logger.info("head_id: %s" % feature.head_ids)
            logger.info("dep_ids: %s" % feature.dep_ids)
            logger.info("pos_ids: %s" % feature.pos_ids)

        return features

    def _convert_features(self, examples: List[KlueDPInputExample]) -> List[KlueDPInputFeatures]:
        return self.convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.hparams.max_seq_length,
            dep_label_list=get_dep_labels(),
            pos_label_list=get_pos_labels(),
        )

    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_bpe_head_mask = torch.tensor([f.bpe_head_mask for f in features], dtype=torch.long)
        all_bpe_tail_mask = torch.tensor([f.bpe_tail_mask for f in features], dtype=torch.long)
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_dep_ids = torch.tensor([f.dep_ids for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)

        return TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_bpe_head_mask,
            all_bpe_tail_mask,
            all_head_ids,
            all_dep_ids,
            all_pos_ids,
        )

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "dev")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        if not os.path.exists(file_path):
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> List[str]:
        return get_dep_labels()

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, Any, Any, Any]:
        # 1. set args
        batch_size = len(batch)
        pos_padding_idx = None if self.hparams.no_pos else len(get_pos_labels())
        # 2. build inputs : input_ids, attention_mask, bpe_head_mask, bpe_tail_mask
        batch_input_ids = []
        batch_attention_masks = []
        batch_bpe_head_masks = []
        batch_bpe_tail_masks = []
        for batch_id in range(batch_size):
            (
                input_id,
                attention_mask,
                bpe_head_mask,
                bpe_tail_mask,
                _,
                _,
                _,
            ) = batch[batch_id]
            batch_input_ids.append(input_id)
            batch_attention_masks.append(attention_mask)
            batch_bpe_head_masks.append(bpe_head_mask)
            batch_bpe_tail_masks.append(bpe_tail_mask)
        # 2. build inputs : packing tensors
        # 나는 밥을 먹는다. => [CLS] 나 ##는 밥 ##을 먹 ##는 ##다 . [SEP]
        # input_id : [2, 717, 2259, 1127, 2069, 1059, 2259, 2062, 18, 3, 0, 0, ...]
        # bpe_head_mask : [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, ...] (indicate word start (head) idx)
        input_ids = torch.stack(batch_input_ids)
        attention_masks = torch.stack(batch_attention_masks)
        bpe_head_masks = torch.stack(batch_bpe_head_masks)
        bpe_tail_masks = torch.stack(batch_bpe_tail_masks)
        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()
        # 3. token_to_words : placeholders
        head_ids = torch.zeros(batch_size, max_word_length).long()
        type_ids = torch.zeros(batch_size, max_word_length).long()
        pos_ids = torch.zeros(batch_size, max_word_length + 1).long()
        mask_e = torch.zeros(batch_size, max_word_length + 1).long()
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            (
                _,
                _,
                bpe_head_mask,
                _,
                token_head_ids,
                token_type_ids,
                token_pos_ids,
            ) = batch[batch_id]
            # head_id : [1, 3, 5] (prediction candidates)
            # token_head_ids : [-1, 3, -1, 3, -1, 0, -1, -1, -1, .-1, ...] (ground truth head ids)
            head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]
            word_length = len(head_id)
            head_id.extend([0] * (max_word_length - word_length))
            head_ids[batch_id] = token_head_ids[head_id]
            type_ids[batch_id] = token_type_ids[head_id]
            if not self.hparams.no_pos:
                pos_ids[batch_id][0] = torch.tensor(pos_padding_idx)
                pos_ids[batch_id][1:] = token_pos_ids[head_id]
                pos_ids[batch_id][int(torch.sum(bpe_head_mask)) + 1 :] = torch.tensor(pos_padding_idx)
            mask_e[batch_id] = torch.LongTensor([1] * (word_length + 1) + [0] * (max_word_length - word_length))
        mask_d = mask_e[:, 1:]
        # 4. pack everything
        masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
        ids = (head_ids, type_ids, pos_ids)

        return input_ids, masks, ids, max_word_length

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        return parser


def get_dep_labels() -> List[str]:
    """
    label for dependency relations format:

    {structure}_(optional){function}

    """
    dep_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
    ]
    return dep_labels


def get_pos_labels() -> List[str]:
    """label for part-of-speech tags"""

    return [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
    ]
