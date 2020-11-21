# type: ignore

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """A single example of data.utils

    This is for YNAT, KLUE-NLI, KLUE-NER, and KLUE-RE.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> None:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """A single set of features of data to feed into the pretrained models.

    This is for YNAT, KLUE-STS, KLUE-NLI, KLUE-NER, and KLUE-RE. Property names
    are same with the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self) -> None:
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DataProcessor:
    """Base class for data converters of klue data sets."""

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        self.hparams = args
        self.tokenizer = tokenizer

    def get_train_dataset(self, data_dir: str, file_name: str) -> Dataset:
        """Gets a :class:`Dataset` for the train set."""
        raise NotImplementedError()

    def get_dev_dataset(self, data_dir: str, file_name: str) -> Dataset:
        """Gets a :class:`Dataset` for the dev set."""
        raise NotImplementedError()

    def get_test_dataset(self, data_dir: str, file_name: str) -> Dataset:
        """Gets a :class:`Dataset` for the test set."""
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def add_specific_args(self, parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        raise NotImplementedError()


class KlueDataModule(pl.LightningDataModule):
    """Constructs a basic datamodule for dataset and dataloader."""

    def __init__(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        super().__init__()
        self.hparams = args
        self.processor = processor

    def prepare_dataset(self, dataset_type: str) -> Dataset:
        """Initializes data. Uses to construct features and dataset."""

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
