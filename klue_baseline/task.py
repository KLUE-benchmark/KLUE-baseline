import argparse
import json
import logging
import os

from transformers import AutoTokenizer

from klue_baseline.data import DataProcessor, KlueDataModule
from klue_baseline.models import BaseTransformer, Mode
from klue_baseline.utils import Command

logger = logging.getLogger(__name__)


class KlueTask:
    """Combines task-specific data processor, model and metrics.

    Args:
        processor_type: Task-specific DataProcessor.
        model_type: Task-specific model type.
        metrics: Task-specific metrics.
    """

    def __init__(self, processor_type: DataProcessor, model_type: BaseTransformer, metrics: dict) -> None:
        self.processor_type = processor_type
        self.model_type = model_type
        self.metrics = metrics

    def setup(self, args: argparse.Namespace, command: str) -> None:
        """Setup data, tokenizer, and model."""
        self.set_filename(args)

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )
        processor = self.processor_type(args, tokenizer)
        datamodule = self.processor_type.datamodule_type(args, processor)

        self.set_command_specifics(args, command, datamodule)
        self.set_mode_specifics(args, processor)
        self.model = self.model_type(args, metrics=self.metrics)
        if self.model.model.config.vocab_size < len(tokenizer):
            self.model.model.resize_token_embeddings(len(tokenizer))

    def set_filename(self, args: argparse.Namespace) -> None:
        """Set filename from file_map.json if it exists in data_dir."""
        if os.path.exists(os.path.join(args.data_dir, "file_map.json")):
            with open(os.path.join(args.data_dir, "file_map.json"), "r") as f:
                file_map = json.load(f)
            args.train_file_name = file_map.get("train", None)
            args.dev_file_name = file_map.get("dev", None)
            args.test_file_name = file_map.get("test", None)
            logger.info(f"Set train_file_name of '{args.task}' as '{args.train_file_name}' from file_map.json")
            logger.info(f"Set dev_file_name of '{args.task}' as '{args.dev_file_name}' from file_map.json")
            logger.info(f"Set test_file_name of '{args.task}' as '{args.test_file_name}' from file_map.json")

    def set_command_specifics(self, args: argparse.Namespace, command: str, datamodule: KlueDataModule) -> None:
        """Set command specific variables and arguments."""
        if command == Command.Train:
            self.train_loader = datamodule.train_dataloader()
            self.val_loader = datamodule.val_dataloader()
            try:
                self.test_loader = datamodule.test_dataloader()
            except NotImplementedError:
                self.test_loader = self.val_loader

            args.dataset_size = len(self.train_loader.dataset)  # total_steps

        elif command == Command.Evaluate:
            self.val_loader = datamodule.val_dataloader()
        elif command == Command.Test:
            self.test_loader = datamodule.test_dataloader()

    def set_mode_specifics(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        """Set mode specific arguments."""
        if self.model_type.mode in [Mode.SequenceClassification, Mode.NamedEntityRecognition, Mode.RelationExtraction]:  # type: ignore[comparison-overlap]
            args.num_labels = len(processor.get_labels())
            args.label_list = processor.get_labels()
        elif self.model_type.mode == Mode.SemanticTextualSimilarity:
            args.num_labels = 1
        elif self.model_type.mode == Mode.DialogueStateTracking:
            args.num_labels = None
            args.processor = processor

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "train_dataloader": self.train_loader,
            "val_dataloaders": self.val_loader,
        }
