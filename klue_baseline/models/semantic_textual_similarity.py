import argparse
import logging
from typing import Dict, List

import torch
from overrides import overrides

from .mode import Mode
from .sequence_classification import SCTransformer

logger = logging.getLogger(__name__)


class STSTransformer(SCTransformer):

    mode: str = Mode.SemanticTextualSimilarity

    def __init__(self, hparams: argparse.Namespace, metrics: dict = {}) -> None:
        super().__init__(hparams, metrics=metrics)

    @overrides
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        preds = self._convert_outputs_to_preds(outputs)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(preds, labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    @overrides
    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return logits.squeeze()
