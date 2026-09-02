"""Evaluation metrics (spec §9), numpy-only (no sklearn dependency).

- ``average_precision`` matches sklearn's ``average_precision_score`` (step
  interpolation: AP = Σ_n (R_n − R_{n−1}) · P_n).
- ``macro_map`` averages per-class AP over classes that have at least one
  positive — the early-stop metric for training.
- ``auroc`` via the rank statistic (Mann-Whitney U) with tie-averaged ranks.
"""

from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary average precision. Returns nan if there are no positives."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")  # stable, descending
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # AP = Σ (recall_n - recall_{n-1}) * precision_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def macro_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean AP over classes with >=1 positive. y_*: (N, C)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aps = [
        average_precision(y_true[:, c], y_score[:, c])
        for c in range(y_true.shape[1])
    ]
    aps = [ap for ap in aps if not np.isnan(ap)]
    if not aps:
        return float("nan")
    return float(np.mean(aps))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUROC via the rank statistic. Returns nan if only one class present."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Average ranks within tied score groups.
    _assign_tie_averaged_ranks(y_score, ranks)

    sum_ranks_pos = ranks[y_true].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Recall (TPR) at the operating point with FPR <= target_fpr (spec §9).

    Returns the highest recall achievable while keeping FPR at or below the
    target. nan if there are no positives or no negatives.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")  # stable, descending
    s_sorted = y_score[order]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    # Only score-group boundaries are achievable operating points: a real
    # threshold can't split tied scores, so evaluate FPR/TPR at the last index of
    # each tied run (roc_curve semantics). Without this, ties inflate recall.
    boundary = np.ones(len(s_sorted), dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]

    fpr = fp[boundary] / n_neg
    tpr = tp[boundary] / n_pos
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean AUROC over classes with both a positive and a negative. y_*: (N, C)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    scores = [
        auroc(y_true[:, c], y_score[:, c])
        for c in range(y_true.shape[1])
    ]
    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _assign_tie_averaged_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    """In place: give tied values their average rank (stable AUROC under ties)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
