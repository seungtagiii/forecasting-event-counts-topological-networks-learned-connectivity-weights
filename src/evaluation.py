"""Metrics do not tune thresholds using the simulated graph.

No lambda truth metric is computed: privileged marked intensity is not the
conditional intensity under the common observation filtration.
"""
from __future__ import annotations

import numpy as np


def average_precision(truth: np.ndarray, scores: np.ndarray) -> float | None:
    truth = truth.astype(bool).ravel()
    scores = scores.ravel()
    positive = int(truth.sum())
    if positive == 0:
        return None
    # Threshold groups, not arbitrary ordering within tied scores.
    order = np.argsort(-scores, kind="stable")
    labels, ranked_scores = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    precision = tp / (ends + 1)
    recall = tp / positive
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def graph_metrics(truth_source_target: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    p = len(truth_source_target)
    mask = ~np.eye(p, dtype=bool)
    truth = truth_source_target.astype(bool)
    prediction = score >= threshold
    np.fill_diagonal(prediction, False)
    tp = int(np.sum(prediction[mask] & truth[mask]))
    fp = int(np.sum(prediction[mask] & ~truth[mask]))
    fn = int(np.sum(~prediction[mask] & truth[mask]))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    # Count reversal as one edit, not two; additions/deletions as one each.
    shd = 0
    for i in range(p):
        for j in range(i + 1, p):
            a = (bool(truth[i, j]), bool(truth[j, i]))
            b = (bool(prediction[i, j]), bool(prediction[j, i]))
            shd += 0 if a == b else 1 if a in {(True, False), (False, True)} and b == a[::-1] else sum(x != y for x, y in zip(a, b))
    true_scores = score[mask & truth]
    false_scores = score[mask & ~truth]
    return {"average_precision": average_precision(truth[mask], score[mask]),
            "random_ranking_reference_prevalence": float(truth[mask].mean()),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "shd_reversal_counts_as_one": int(shd), "tp": tp, "fp": fp, "fn": fn,
            "threshold_fixed_before_test": threshold,
            "mean_true_edge_score": float(true_scores.mean()) if len(true_scores) else None,
            "mean_nonedge_score": float(false_scores.mean()) if len(false_scores) else None,
            "sid": None, "sid_reason": "not_implemented_and_predictions_not_guaranteed_DAG"}


def forecast_metrics(y: np.ndarray, intensity: np.ndarray, log_mass: np.ndarray, delta: float) -> dict:
    mean = intensity * delta
    error = mean - y
    return {"poisson_count_nll": float(-log_mass.mean()),
            "count_mae": float(np.abs(error).mean()), "count_rmse": float(np.sqrt(np.mean(error**2))),
            "mean_predicted_intensity": float(intensity.mean()),
            "mean_observed_count": float(y.mean()), "evaluated_counts": int(y.size),
            "lambda_rmse": None, "lambda_mae": None,
            "lambda_truth_status": "unavailable_for_common_observed_history_not_replaced_by_count_over_delta"}
