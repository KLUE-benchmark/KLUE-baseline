import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pkg_resources
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

try:
    pkg = "pytorch_lightning"
    min_ver = "1.0.4"
    pkg_resources.require(f"{pkg}>={min_ver}")
except pkg_resources.VersionConflict:
    logger.warning(
        f"{pkg}>={min_ver} is required for a normal functioning of this module, but found {pkg}=={pkg_resources.get_distribution(pkg).version}. Try pip install -r examples/requirements.txt"
    )


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    """Initializes a model, tokenizer and config for the task."""

    USE_TOKEN_TYPE_MODELS = ["bert", "xlnet", "electra"]

    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels: Optional[int] = None,
        mode: str = "base",
        config: Optional[PretrainedConfig] = None,
        model_type: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        metrics: Dict[str, Any] = {},
        **config_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        data = getattr(hparams, "data", None)
        if data is not None:
            delattr(hparams, "data")
        self.save_hyperparameters(hparams)
        self.hparams.data = data

        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.predictions: List[int] = []

        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config  # type: ignore[no-redef]

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer = tokenizer
        self.model = model_type.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        self.metrics = nn.ModuleDict(metrics)
        self.eval_dataset_type = "valid"

    def is_use_token_type(self) -> bool:
        if self.config.model_type in set(self.USE_TOKEN_TYPE_MODELS):
            return True
        else:
            return False

    def get_lr_scheduler(self) -> Any:
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]

        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.num_warmup_steps(), num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def training_step(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step_end(self, training_step_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # For DataParallel
        return {"loss": training_step_outputs["loss"].mean()}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, data_type: str) -> Dict[str, torch.Tensor]:
        # return Format: (e.g. dictionary {"logits": logits, "labels": labels})
        raise NotImplementedError

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        preds = self._convert_outputs_to_preds(outputs)
        labels = torch.cat([output["labels"] for output in outputs], dim=0)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(preds, labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        assert self.eval_dataset_type in {"valid", "test"}
        return self.validation_step(batch, batch_idx, data_type=self.eval_dataset_type)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        assert self.eval_dataset_type in {"valid", "test"}
        return self.validation_epoch_end(outputs, data_type=self.eval_dataset_type, write_predictions=True)

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> Any:
        # outputs is output (dict, return object from validation_step) of list
        raise NotImplementedError

    def _set_metrics_device(self) -> None:
        device = next(self.parameters()).device
        for _, metric in self.metrics.items():
            if metric.device is None:
                metric.device = device

    def num_warmup_steps(self) -> Any:
        num_warmup_steps = self.hparams.warmup_steps
        if num_warmup_steps is None and self.hparams.warmup_ratio is not None:
            num_warmup_steps = self.total_steps() * self.hparams.warmup_ratio
            num_warmup_steps = math.ceil(num_warmup_steps)

        if num_warmup_steps is None:
            num_warmup_steps = 0
        return num_warmup_steps

    def total_steps(self) -> Any:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.num_gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.hparams.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("transformers")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=None, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--warmup_ratio", default=None, type=float, help="Linear warmup over warmup_step ratio.")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--verbose_step_count", default=100, type=int)
        return parser
