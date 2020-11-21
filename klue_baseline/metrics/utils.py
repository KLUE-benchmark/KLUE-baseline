import re
import string
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

NORMALIZE_CHAR_PATTERN = re.compile(r"[\'\"《》<>〈〉]\(\)\‘\’")
PUNCTUATION_SET = set(string.punctuation)
KLUE_MRC_NUM_QUESTION_TYPE = 3


def normalize_answer_for_klue_mrc(answer: str) -> str:
    """Excludes useless characters in answer string.

    Args:
        answer: The raw text of answer.

    Returns:
        The normalized answer.
    """
    answer = NORMALIZE_CHAR_PATTERN.sub(" ", answer.lower())
    answer = "".join(c for c in answer if c not in PUNCTUATION_SET)
    answer = " ".join(answer.split())
    return answer


def rouge_w_score_for_klue_mrc(pred: str, label: str, beta: int = 1) -> float:
    """Calculates character level ROUGE-W score https://en.wikipedia.org/wiki/ROUGE_(metric)"""
    if label == "":
        return float(pred == label)

    matcher = SequenceMatcher(None, pred, label)
    longest_common_consecutive_sequence_length = matcher.find_longest_match(0, len(pred), 0, len(label)).size

    precision = longest_common_consecutive_sequence_length / len(pred) if len(pred) else 0.0
    recall = longest_common_consecutive_sequence_length / len(label) if len(label) else 0.0

    if precision + recall == 0.0:
        return 0.0

    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def compute_em_and_rouge_w_score_for_klue_mrc(pred: str, labels: List[str]) -> Tuple[float, float]:
    """Calculates Exact Match(EM) and ROUGE-W scores for single example.
    The maximum EM and ROUGE-W scores will be returned among the multiple labels.

    Args:
        pred: The predicted answer of single example.
        label: The ground truth answer of single example.

    Returns:
        Exact Match(EM), ROUGE-W score
    """
    em_scores, rouge_scores = [0.0], [0.0]

    for label in labels:
        em_scores.append(float(pred == label))
        rouge_scores.append(rouge_w_score_for_klue_mrc(pred, label))

    return max(em_scores), max(rouge_scores)


def evaluate_for_klue_mrc(labels: List[Dict[str, Any]], predictions: Dict[Any, str]) -> Dict[str, float]:
    """Calculate average EM and ROUGE-W scores of total evaluation examples.

    Args:
        labels: Dictionary of guid, question_type, ground_truth.
        predictions: Dictionary of question_type, prediction.

    Returns:
        Average Exact Match(EM), ROUGE-W score
    """
    exact_match_scores, rouge_scores = [], []

    for label in labels:
        if label["qid"] not in predictions:
            continue

        pred_answer = normalize_answer_for_klue_mrc(predictions[label["qid"]])
        ground_truths = (
            [normalize_answer_for_klue_mrc(answer) for answer in label["ground_truth"]]
            if label["ground_truth"]
            else [""]
        )

        em, rouge = compute_em_and_rouge_w_score_for_klue_mrc(pred_answer, ground_truths)
        exact_match_scores.append(em)
        rouge_scores.append(rouge)

    return {"exact_match": np.mean(exact_match_scores), "rouge": np.mean(rouge_scores)}


def extract_labels_from_dataset_for_klue_mrc(datas: List[Any]) -> List[Dict[str, Any]]:
    """Extracts the labels from evaluation dataset examples."""
    labels = []
    for example in datas:
        for paragraph in example["paragraphs"]:
            for qa in paragraph["qas"]:
                label = {
                    "qid": qa["guid"],
                    "qtype": qa["question_type"],
                    "ground_truth": [answer["text"] for answer in qa["answers"]],
                }
                labels.append(label)
    return labels


def compute_prf_for_wos(gold: Sequence[str], pred: Sequence[str]) -> Tuple[float, float, float, float]:
    """Most of the below code is from https://github.com/jasonwu0731/trade-dst"""
    tp, fp, fn = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                tp += 1
            else:
                fn += 1
        for p in pred:
            if p not in gold:
                fp += 1
        precision = tp / float(tp + fp) if (tp + fp) != 0 else 0
        recall = tp / float(tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, f1, count = 1, 1, 1, 1
        else:
            precision, recall, f1, count = 0, 0, 0, 1
    return f1, recall, precision, count
