# type: ignore

import csv
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):

    SKIP_KEYS = set(["log", "progress_bar"])

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Train batch loss
        global_step = pl_module.global_step
        verbose_step_count = pl_module.hparams.verbose_step_count

        if global_step != 0 and global_step % verbose_step_count == 0:
            batch_loss = trainer.logged_metrics["train/loss"]
            rank_zero_info(f"Step: {global_step} - Loss: {batch_loss}")

        # LR Scheduler
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        global_step = pl_module.global_step
        if global_step == 0:
            return

        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics

        for k, v in metrics.items():
            if k in self.SKIP_KEYS:
                continue
            rank_zero_info(f"{k} = {v}")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        rank_zero_info("***** Test results *****")

        # Write Predictions
        try:
            self._write_predictions(trainer.test_dataloaders, pl_module)
        except BaseException:
            pass

        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for k, v in metrics.items():
                if k in self.SKIP_KEYS:
                    continue
                rank_zero_info(f"{k} = {v}")
                writer.write(f"{k} = {v}\n")

    def _write_predictions(self, dataloaders: DataLoader, pl_module: pl.LightningModule) -> None:
        index = 0
        output_test_pred_file = os.path.join(pl_module.hparams.output_dir, "test_predictions.tsv")
        with open(output_test_pred_file, "w", newline="\n") as csvfile:
            one_example = dataloaders[0].dataset.examples[0]
            fieldnames = list(one_example.to_dict().keys()) + ["prediction"]

            writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()

            for dataloader in dataloaders:
                for example in dataloader.dataset.examples:
                    row = example.to_dict()
                    row["prediction"] = pl_module.predictions[index].item()

                    writer.writerow(row)
                    index += 1
