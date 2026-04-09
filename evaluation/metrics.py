"""
Shared evaluation metrics for all groups.

Metrics:
  - contains_match : case-insensitive substring match (Groups 1, 2 HaluEval, Group 3)
  - token_f1       : token-level F1 score (Group 2 NarrativeQA)
  - accuracy       : exact match on categorical labels (Group 2 QuALITY, Group 3 FRAMES)
  - action_f1      : token F1 on action representations (Group 3 Mind2Web)
"""

from collections import Counter
from typing import List


def contains_match(prediction: str, ground_truth: str) -> int:
    """
    Returns 1 if ground_truth appears (case-insensitive) anywhere in prediction, else 0.
    Skips evaluation if ground_truth is empty (e.g. LoCoMo cat-5 adversarial questions).
    """
    if not ground_truth:
        return None  # skip
    return int(ground_truth.strip().lower() in prediction.strip().lower())


def token_f1(prediction: str, references: List[str]) -> float:
    """
    Computes token-level F1 between prediction and a list of reference answers.
    Returns the best F1 across all references.
    Used for NarrativeQA.
    """
    def tokenize(s: str) -> List[str]:
        return s.lower().split()

    pred_tokens = tokenize(prediction)
    best = 0.0
    for ref in references:
        ref_tokens = tokenize(ref)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def accuracy(prediction, gold_label) -> int:
    """
    Returns 1 if prediction matches gold_label exactly, else 0.
    Handles both letter ('A') and numeric (1) gold labels.
    Used for QuALITY MCQ and FRAMES.
    """
    LETTER_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    pred = str(prediction).strip().upper()
    pred_num = LETTER_MAP.get(pred, None)
    gold = int(gold_label)
    if pred_num is not None:
        return int(pred_num == gold)
    try:
        return int(int(pred) == gold)
    except ValueError:
        return 0


def action_f1(predicted_repr: str, gold_repr: str) -> float:
    """
    Token-level F1 on action representation strings.
    Used for Mind2Web action evaluation.
    """
    return token_f1(predicted_repr, [gold_repr])


def score_group1(results: list) -> dict:
    """
    Compute Group 1 contains-match score from a list of result dicts.
    Skips entries where ground_truth is empty (cat-5 adversarial).
    Expects fields: pred (str), gt (str)
    """
    valid = [r for r in results if r.get("gt", "")]
    if not valid:
        return {"correct": 0, "total": 0, "score": 0.0}
    correct = sum(contains_match(r["pred"], r["gt"]) or 0 for r in valid)
    return {"correct": correct, "total": len(valid), "score": 100 * correct / len(valid)}


def score_narrativeqa(results: list) -> dict:
    """
    Compute NarrativeQA token F1 score.
    Expects fields: model_answer, reference_answer1, reference_answer2
    """
    f1s = [
        token_f1(r["model_answer"], [r["reference_answer1"], r["reference_answer2"]])
        for r in results
    ]
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"avg_f1": 100 * avg_f1, "total": len(results)}


def score_quality(results: list) -> dict:
    """
    Compute QuALITY accuracy score.
    Expects fields: model_letter or model_choice (int), gold_label (int)
    """
    LETTER_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    correct = 0
    for r in results:
        pred = str(r.get("model_letter", r.get("model_choice", ""))).strip().upper()
        gold = int(r["gold_label"])
        pred_num = LETTER_MAP.get(pred, None)
        if pred_num is None:
            try:
                pred_num = int(pred)
            except ValueError:
                pred_num = 0
        if pred_num == gold:
            correct += 1
    return {"correct": correct, "total": len(results), "score": 100 * correct / len(results)}


def score_halueval(results: list) -> dict:
    """
    Compute HaluEval contains-match score.
    Expects fields: model_answer, right_answer
    """
    correct = sum(
        1 for r in results
        if r.get("right_answer", "").lower() in r.get("model_answer", "").lower()
    )
    return {"correct": correct, "total": len(results), "score": 100 * correct / len(results)}
