import argparse
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import torch
from overrides import overrides
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import DataProcessor, InputExample, InputFeatures, KlueDataModule
from klue_baseline.data.utils import check_tokenizer_type

logger = logging.getLogger(__name__)


class KlueREProcessor(DataProcessor):

    origin_train_file_name = "klue-re-v1.1_train.json"
    origin_dev_file_name = "klue-re-v1.1_dev.json"
    origin_test_file_name = "klue-re-v1.1_test.json"
    origin_relation_file_name = "relation_list.json"

    datamodule_type = KlueDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

        # special tokens to mark the subject/object entity boundaries
        self.subject_start_marker = "<subj>"
        self.subject_end_marker = "</subj>"
        self.object_start_marker = "<obj>"
        self.object_end_marker = "</obj>"

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.subject_start_marker,
                    self.subject_end_marker,
                    self.object_start_marker,
                    self.object_end_marker,
                ]
            }
        )

        # Load relation class
        relation_class_file_path = os.path.join(args.data_dir, args.relation_filename or self.origin_relation_file_name)

        with open(relation_class_file_path, "r", encoding="utf-8") as f:
            self.relation_class = json.load(f)["relations"]

        # Check type of tokenizer
        self.tokenizer_type = check_tokenizer_type(tokenizer)

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
    def get_labels(self) -> Any:
        return self.relation_class

    def _create_examples(self, file_path: str, dataset_type: str) -> List[InputExample]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            data_lst = json.load(f)

        for data in data_lst:
            guid = data["guid"]
            text = data["sentence"]
            subject_entity = data["subject_entity"]
            object_entity = data["object_entity"]
            label = data["label"]

            text = self._mark_entity_spans(
                text=text,
                subject_range=(int(subject_entity["start_idx"]), int(subject_entity["end_idx"])),
                object_range=(int(object_entity["start_idx"]), int(object_entity["end_idx"])),
            )
            examples.append(InputExample(guid=guid, text_a=text, label=label))

        return examples

    def _mark_entity_spans(
        self,
        text: str,
        subject_range: Tuple[int, int],
        object_range: Tuple[int, int],
    ) -> str:
        """Adds entity markers to the text to identify the subject/object entities.

        Args:
            text: Original sentence
            subject_range: Pair of start and end indices of subject entity
            object_range: Pair of start and end indices of object entity

        Returns:
            A string of text with subject/object entity markers
        """
        if subject_range < object_range:
            segments = [
                text[: subject_range[0]],
                self.subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                self.subject_end_marker,
                text[subject_range[1] + 1 : object_range[0]],
                self.object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                self.object_end_marker,
                text[object_range[1] + 1 :],
            ]
        elif subject_range > object_range:
            segments = [
                text[: object_range[0]],
                self.object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                self.object_end_marker,
                text[object_range[1] + 1 : subject_range[0]],
                self.subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                self.subject_end_marker,
                text[subject_range[1] + 1 :],
            ]
        else:
            raise ValueError("Entity boundaries overlap.")

        marked_text = "".join(segments)

        return marked_text

    def _convert_example_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        max_length = self.hparams.max_seq_length
        if max_length is None:
            max_length = self.tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.get_labels())}
        labels = [label_map[example.label] for example in examples]

        def fix_tokenization_error(text: str, tokenizer_type: str) -> Any:
            """Fix the tokenization due to the `obj` and `subj` marker inserted
            in the middle of a word.

            Example:
                >>> text = "<obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가"
                >>> tokens = ['<obj>', '조지', '해리', '##슨', '</obj>', '이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '가']
                >>> fix_tokenization_error(text, tokenizer_type="bert-wp")
                ['<obj>', '조지', '해리', '##슨', '</obj>', '##이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '##가']
            """
            tokens = self.tokenizer.tokenize(text)
            # subject
            if text[text.find(self.subject_end_marker) + len(self.subject_end_marker)] != " ":
                space_idx = tokens.index(self.subject_end_marker) + 1
                if tokenizer_type == "xlm-sp":
                    if tokens[space_idx] == "▁":
                        tokens.pop(space_idx)
                    elif tokens[space_idx].startswith("▁"):
                        tokens[space_idx] = tokens[space_idx][1:]
                elif tokenizer_type == "bert-wp":
                    if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                        tokens[space_idx] = "##" + tokens[space_idx]

            # object
            if text[text.find(self.object_end_marker) + len(self.object_end_marker)] != " ":
                space_idx = tokens.index(self.object_end_marker) + 1
                if tokenizer_type == "xlm-sp":
                    if tokens[space_idx] == "▁":
                        tokens.pop(space_idx)
                    elif tokens[space_idx].startswith("▁"):
                        tokens[space_idx] = tokens[space_idx][1:]
                elif tokenizer_type == "bert-wp":
                    if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                        tokens[space_idx] = "##" + tokens[space_idx]

            return tokens

        tokenized_examples = [fix_tokenization_error(example.text_a, self.tokenizer_type) for example in examples]
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(self.tokenizer.convert_tokens_to_ids(tokens), None) for tokens in tokenized_examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i in range(5):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (examples[i].guid))
            logger.info("origin example: %s" % examples[i])
            logger.info("origin tokens: %s" % self.tokenizer.tokenize(examples[i].text_a))
            logger.info("fixed tokens: %s" % tokenized_examples[i])
            logger.info("features: %s" % features[i])

        return features

    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_example_to_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueREProcessor.datamodule_type.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--relation_filename",
            default="relation_list.json",
            type=str,
            help="File name of list of relation classes",
        )
        return parser
