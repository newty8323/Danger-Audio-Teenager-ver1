import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.taxonomy import load_taxonomy
from evaluate import _json_safe, harm_report, measure_latency, predict
from models.harm_model import HarmModel, ModelConfig


def _labels_each_class_covered(tax):
    """(N, C) labels where every class has 2 positives (and many negatives)."""
    c = tax.num_classes
    labels = np.zeros((2 * c, c), dtype=np.float32)
    for k in range(c):
        labels[2 * k, k] = 1.0
        labels[2 * k + 1, k] = 1.0
    return labels


def test_harm_report_perfect_predictions_pass_targets():
    tax = load_taxonomy()
    labels = _labels_each_class_covered(tax)
    probs = labels * 0.9 + (1 - labels) * 0.1  # perfect separation
    report = harm_report(probs, labels, tax)

    assert set(report["per_class"]) == set(tax.all_classes)
    assert set(report["per_category"]) == set(tax.harm_categories)
    assert abs(report["macro_map"] - 1.0) < 1e-9
    assert all(report["targets"].values())  # every §9 target met


def test_harm_report_reversed_predictions_fail_targets():
    tax = load_taxonomy()
    labels = _labels_each_class_covered(tax)
    probs = 1.0 - (labels * 0.9 + (1 - labels) * 0.1)  # anti-correlated
    report = harm_report(probs, labels, tax)
    assert not any(report["targets"].values())


def test_harm_report_per_class_metrics_present():
    tax = load_taxonomy()
    labels = _labels_each_class_covered(tax)
    probs = labels * 0.9 + (1 - labels) * 0.1
    report = harm_report(probs, labels, tax)
    m = report["per_class"]["vio_gunshot"]
    assert set(m) == {"ap", "auroc", "recall_at_fpr"}
    assert abs(m["auroc"] - 1.0) < 1e-9


def test_report_serializes_to_strict_json_despite_nan():
    tax = load_taxonomy()
    # a confusable class with no positives -> nan metrics in the report
    labels = _labels_each_class_covered(tax)
    labels[:, tax.index_of("asmr")] = 0.0
    probs = labels * 0.9 + (1 - labels) * 0.1
    report = harm_report(probs, labels, tax)
    # strict JSON (allow_nan=False) must succeed after sanitizing
    text = json.dumps(_json_safe(report), allow_nan=False)
    assert json.loads(text)["per_class"]["asmr"]["ap"] is None


def test_predict_and_latency_smoke():
    tax = load_taxonomy()
    model = HarmModel(tax.num_classes, ModelConfig(backbone_out_dim=32))
    x = torch.randn(6, 1, 32, 32)
    y = (torch.rand(6, tax.num_classes) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    device = torch.device("cpu")

    probs, labels = predict(model, loader, device)
    assert probs.shape == (6, tax.num_classes)
    assert labels.shape == (6, tax.num_classes)
    assert (probs >= 0).all() and (probs <= 1).all()

    ms = measure_latency(model, x[:1], device, repeats=3)
    assert ms > 0
