import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from overrides import overrides
from transformers import AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult

from klue_baseline.models import BaseTransformer, Mode

logger = logging.getLogger(__name__)


# for multi gpu environment
@dataclass
class QAResults:
    results: List[SquadResult]


class MRCTransformer(BaseTransformer):

    mode: str = Mode.MachineReadingComprehension

    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace], metrics: dict = {}) -> None:
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        super().__init__(
            hparams,
            mode=self.mode,
            model_type=AutoModelForQuestionAnswering,
            metrics=metrics,
        )

    @overrides
    def forward(self, **inputs: torch.Tensor) -> Any:
        return self.model(**inputs)

    @overrides
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_features = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        if self.is_use_token_type():
            input_features["token_type_ids"] = batch[2]

        loss = self(**input_features)[0]
        self.log("train/loss", loss)
        return {"loss": loss}

    @overrides
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid"
    ) -> Dict[str, QAResults]:
        with torch.no_grad():
            input_features = {"input_ids": batch[0], "attention_mask": batch[1]}
            if self.is_use_token_type():
                input_features["token_type_ids"] = batch[2]

            start_logits, end_logits = self(**input_features)

        results = []
        feature_indices = batch[3].tolist()
        total_features = self.hparams.data[data_type]["features"]

        for i, feature_index in enumerate(feature_indices):
            unique_id = int(total_features[feature_index].unique_id)
            single_example_start_logits = start_logits[i].tolist()
            single_example_end_logits = end_logits[i].tolist()
            results.append(SquadResult(unique_id, single_example_start_logits, single_example_end_logits))

        return {"results": QAResults(results)}

    @overrides
    def validation_epoch_end(
        self, outputs: List[dict], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        qa_results = []
        for output in outputs:
            if isinstance(output["results"], list):
                for device_outputs in output["results"]:
                    qa_results.extend(device_outputs.results)
            else:
                qa_results.extend(output["results"].results)

        examples = self.hparams.data[data_type]["examples"]
        features = self.hparams.data[data_type]["features"]
        do_lower_case = getattr(self.tokenizer, "do_lower_case", False)

        # Require part of the data for validation sanity check
        if len(qa_results) == (self.hparams.eval_batch_size * self.hparams.num_sanity_val_steps):
            features = features[: len(qa_results)]
            feature_qas_ids = set([feature.qas_id for feature in features])
            examples = [example for example in examples if example.qas_id in feature_qas_ids]

        preds = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=qa_results,
            n_best_size=self.hparams.n_best_size,
            max_answer_length=self.hparams.max_answer_length,
            do_lower_case=do_lower_case,
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=self.hparams.verbose_logging,
            version_2_with_negative=True,  # for unanswerable question
            null_score_diff_threshold=0,
            tokenizer=self.tokenizer,
        )

        for k, metric in self.metrics.items():
            metric(preds, examples)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        BaseTransformer.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )
        parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.",
        )
        parser.add_argument(
            "--verbose_logging",
            action="store_true",
            help="If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation.",
        )
        return parser
