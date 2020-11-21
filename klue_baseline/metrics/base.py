from typing import Any, Callable, Optional

import torch
from overrides import overrides
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities import rank_zero_warn


class BaseMetric(Metric):
    """Base class for metrics."""

    def __init__(
        self,
        metric_fn: Callable,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            "MetricBase will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.metric_fn = metric_fn
        self.device = device

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates state with predictions and targets.

        Args:
            preds: Predictions from model
            targets: Ground truth values
        """

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> Any:
        """Computes metric value over state."""

        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets)
        score = torch.tensor(score).to(self.device)
        return score


class LabelRequiredMetric(BaseMetric):
    """Metrics requiring label information."""

    def __init__(
        self,
        metric_fn: Callable,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            metric_fn=metric_fn,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            device=device,
        )
        self.label_info = None

    @overrides
    def update(self, preds: torch.Tensor, targets: torch.Tensor, label_info: Optional[Any] = None) -> None:
        """Updates state with predictions and targets.

        Args:
            preds: Predictions from model
            targets: Ground truth values
            label_info: Additional label information to compute the metric
        """

        self.preds.append(preds)
        self.targets.append(targets)
        if self.label_info is None:
            self.label_info = label_info

    @overrides
    def compute(self) -> Any:
        """Computes metric value over state."""

        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets, self.label_info)
        score = torch.tensor(score).to(self.device)
        return score
