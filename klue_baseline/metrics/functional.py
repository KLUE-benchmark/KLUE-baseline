import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import sklearn
from scipy.stats import pearsonr
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.scheme import IOB2

from klue_baseline.data.klue_mrc import KlueMRCExample
from klue_baseline.models.dependency_parsing import DPResult

from .utils import (
    KLUE_MRC_NUM_QUESTION_TYPE,
    compute_em_and_rouge_w_score_for_klue_mrc,
    compute_prf_for_wos,
    normalize_answer_for_klue_mrc,
)

logger = logging.getLogger(__name__)


def ynat_macro_f1(preds: np.ndarray, targets: np.ndarray) -> Any:
    return sklearn.metrics.f1_score(targets, preds, average="macro") * 100.0


def klue_nli_acc(preds: np.ndarray, targets: np.ndarray) -> Any:
    return (preds == targets).mean() * 100.0


def klue_sts_pearsonr(preds: np.ndarray, labels: np.ndarray) -> Any:
    return pearsonr(preds, labels)[0] * 100.0


def klue_sts_f1(preds: np.ndarray, labels: np.ndarray) -> Any:
    threshold = 3
    preds = np.where(preds >= threshold, 1, 0)
    labels = np.where(labels >= threshold, 1, 0)
    return sklearn.metrics.f1_score(labels, preds, average="binary") * 100.0


def klue_ner_entity_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
    """KLUE-NER entity-level macro F1 (except O tag)"""
    preds = np.array(preds).flatten().tolist()
    labels = np.array(labels).flatten().tolist()
    preds_label = []
    labels_label = []

    for pred in preds:
        preds_label.append(label_list[pred])
    for label in labels:
        labels_label.append(label_list[label])

    entity_macro_f1 = ner_f1_score([labels_label], [preds_label], average="macro", mode="strict", scheme=IOB2)
    return entity_macro_f1 * 100.0


def klue_ner_char_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
    """KLUE-NER character level macro f1 (except O tag)"""
    label_indices = list(range(len(label_list)))
    preds = np.array(preds).flatten().tolist()
    trues = np.array(labels).flatten().tolist()
    return sklearn.metrics.f1_score(trues, preds, labels=label_indices, average="macro", zero_division=True) * 100.0


def klue_re_micro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
    """KLUE-RE micro f1 (except no_relation)"""
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs: np.ndarray, labels: np.ndarray) -> Any:
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def klue_dp_uas_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    return sklearn.metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="macro") * 100.0


def klue_dp_uas_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    return sklearn.metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="micro") * 100.0


def klue_dp_las_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    # UAS : head correct / LAS : head + type correct
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return sklearn.metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="macro") * 100.0


def klue_dp_las_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return sklearn.metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="micro") * 100.0


def klue_mrc_em(preds: List[Dict[str, str]], examples: List[List[KlueMRCExample]]) -> Any:
    """KLUE-MRC Exact Match (EM)"""
    preds, examples = preds[0], examples[0]

    em_scores_per_question_type: List[List[float]] = [[], [], []]
    for example in examples:
        prediction = normalize_answer_for_klue_mrc(preds[example.qas_id])
        ground_truths = [normalize_answer_for_klue_mrc(answer["text"]) for answer in example.answers]
        # For unanswerable questions, only correct answer is empty string
        if not ground_truths:
            ground_truths = [""]

        em_score, _ = compute_em_and_rouge_w_score_for_klue_mrc(prediction, ground_truths)
        em_scores_per_question_type[example.question_type - 1].append(em_score)

    logger.info("** Exact Match(EM) scores by type **")
    for question_type in range(KLUE_MRC_NUM_QUESTION_TYPE):
        question_type_em_scores = em_scores_per_question_type[question_type]
        avg_em_score = np.mean(question_type_em_scores) * 100.0
        logger.info(f"type{question_type+1} ({len(question_type_em_scores)}): {avg_em_score:.4f}")

    total_em_scores = [score for scores in em_scores_per_question_type for score in scores]
    return np.mean(total_em_scores) * 100.0


def klue_mrc_rouge_w(preds: List[Dict[str, str]], examples: List[List[KlueMRCExample]]) -> Any:
    """KLUE-MRC ROUGE-W"""
    preds, examples = preds[0], examples[0]
    rouge_scores_per_question_type: List[List[float]] = [[], [], []]
    for example in examples:
        prediction = normalize_answer_for_klue_mrc(preds[example.qas_id])
        ground_truths = [normalize_answer_for_klue_mrc(answer["text"]) for answer in example.answers]
        # For unanswerable questions, only correct answer is empty string
        if not ground_truths:
            ground_truths = [""]

        _, rouge_score = compute_em_and_rouge_w_score_for_klue_mrc(prediction, ground_truths)
        rouge_scores_per_question_type[example.question_type - 1].append(rouge_score)

    logger.info("** ROUGE-W scores by type **")
    for question_type in range(KLUE_MRC_NUM_QUESTION_TYPE):
        question_type_rouge_scores = rouge_scores_per_question_type[question_type]
        avg_rouge_score = np.mean(question_type_rouge_scores) * 100.0
        logger.info(f"type{question_type+1} ({len(question_type_rouge_scores)}): {avg_rouge_score:.4f}")

    total_rouge_scores = [score for scores in rouge_scores_per_question_type for score in scores]
    return np.mean(total_rouge_scores) * 100.0


def wos_jga(pred_steps: Sequence[Sequence[str]], trgt_steps: Sequence[Sequence[str]]) -> Any:
    total, joint_goal_acc = 0, 0
    for (pred_batch, trgt_batch) in zip(pred_steps, trgt_steps):
        for (pred, trgt) in zip(pred_batch, trgt_batch):
            if set(pred) == set(trgt):
                joint_goal_acc += 1
            total += 1

    joint_goal_acc_score = joint_goal_acc / float(total) if total != 0 else 0
    return joint_goal_acc_score * 100.0


def wos_slot_micro_f1(pred_steps: Sequence[Sequence[str]], trgt_steps: Sequence[Sequence[str]]) -> Any:
    count, f1 = 0, 0
    for (pred_batch, trgt_batch) in zip(pred_steps, trgt_steps):
        for (pred, trgt) in zip(pred_batch, trgt_batch):
            curr_f1, _, _, curr_count = compute_prf_for_wos(gold=trgt, pred=pred)
            f1 += curr_f1
            count += curr_count

    f1_score = f1 / float(count) if count != 0 else 0
    return f1_score * 100.0
