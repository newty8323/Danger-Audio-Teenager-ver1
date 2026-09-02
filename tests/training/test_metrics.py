import numpy as np

from training.metrics import (
    auroc,
    average_precision,
    macro_auroc,
    macro_map,
    recall_at_fpr,
)


def test_average_precision_known_value():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    # positives at ranks 1 and 3: AP = 0.5*1 + 0.5*(2/3) = 0.8333...
    assert abs(average_precision(y_true, y_score) - 0.8333333) < 1e-6


def test_average_precision_perfect_is_one():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    assert abs(average_precision(y_true, y_score) - 1.0) < 1e-9


def test_average_precision_no_positive_is_nan():
    assert np.isnan(average_precision(np.array([0, 0]), np.array([0.5, 0.1])))


def test_macro_map_skips_positive_free_classes():
    y_true = np.array([[1, 0], [0, 0]])  # class 1 has no positive
    y_score = np.array([[0.9, 0.2], [0.1, 0.8]])
    # only class 0 counts; its AP is 1.0
    assert abs(macro_map(y_true, y_score) - 1.0) < 1e-9


def test_auroc_perfect_and_reversed():
    y_true = np.array([1, 1, 0, 0])
    assert auroc(y_true, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auroc(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_auroc_all_ties_is_half():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(auroc(y_true, y_score) - 0.5) < 1e-9


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.5, 0.9])))


def test_macro_auroc_averages_valid_classes():
    y_true = np.array([[1, 1], [0, 1]])  # class 1 single-class -> skipped
    y_score = np.array([[0.9, 0.3], [0.1, 0.7]])
    assert abs(macro_auroc(y_true, y_score) - 1.0) < 1e-9


def test_recall_at_fpr_perfect_separation():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.15, 0.1, 0.05])  # positives on top
    assert recall_at_fpr(y_true, y_score, target_fpr=0.01) == 1.0


def test_recall_at_fpr_reversed_is_zero():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.05, 0.9, 0.8, 0.7, 0.6])  # positives at bottom
    assert recall_at_fpr(y_true, y_score, target_fpr=0.01) == 0.0


def test_recall_at_fpr_respects_budget():
    # 1 positive above all negatives, 1 positive buried; 10 negatives -> target 0.1
    # allows FPR up to 0.1 (1 negative). Only the top positive clears FPR=0.
    y_true = np.array([1] + [0] * 10 + [1])
    y_score = np.array([1.0] + list(np.linspace(0.9, 0.1, 10)) + [0.0])
    assert recall_at_fpr(y_true, y_score, target_fpr=0.05) == 0.5  # 1 of 2 positives


def test_recall_at_fpr_no_positive_is_nan():
    assert np.isnan(recall_at_fpr(np.array([0, 0, 0]), np.array([0.5, 0.2, 0.1])))


def test_recall_at_fpr_ties_not_inflated():
    # A positive and a negative with identical scores cannot be separated by any
    # threshold -> recall at FPR 1% must be 0, not an optimistic 1.
    assert recall_at_fpr(np.array([1, 0]), np.array([0.5, 0.5]), target_fpr=0.01) == 0.0
