"""Threshold-fitting helpers from .autorun/cascade_offline.py (imported without running it)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    """Import the script module without executing main() (guarded by __main__)."""
    for p in (_ROOT / "src", _ROOT / "scripts", _ROOT / ".autorun", _ROOT / "distill"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(
        "cascade_offline", _ROOT / ".autorun/cascade_offline.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:  # heavy deps (transformers/torchaudio) unavailable
        pytest.skip(f"cascade_offline import needs project deps: {type(e).__name__}: {e}")
    return m


def test_thr_at_fpr_respects_budget(mod):
    neg = np.linspace(0, 1, 100)
    for fpr in (0.01, 0.05, 0.1, 0.5):
        thr = mod.thr_at_fpr(neg, fpr)
        assert (neg >= thr).mean() <= fpr + 1e-9


def test_thr_at_fpr_zero_budget_excludes_all(mod):
    neg = np.linspace(0, 1, 50)
    assert (neg >= mod.thr_at_fpr(neg, 0.0)).sum() == 0


def test_thr_at_fpr_respects_budget_with_ties(mod):
    """Quantized scores saturate to identical values — the k-th score can over-admit."""
    neg = np.concatenate([np.full(100, 0.1), np.full(100, 0.5), np.full(100, 0.9)])
    for fpr in (0.05, 0.2, 0.34, 0.5):
        thr = mod.thr_at_fpr(neg, fpr)
        assert (neg >= thr).mean() <= fpr + 1e-9, f"fpr={fpr} thr={thr}"


def test_thr_at_fpr_all_identical_negatives(mod):
    neg = np.full(64, 0.42)
    assert (neg >= mod.thr_at_fpr(neg, 0.05)).sum() == 0


def test_thr_at_recall_keeps_target(mod):
    pos = np.linspace(0, 1, 200)
    for r in (0.9, 0.95, 0.98, 1.0):
        thr = mod.thr_at_recall(pos, r)
        assert (pos >= thr).mean() >= r - 1e-9


def test_empty_inputs_do_not_crash(mod):
    assert 0.0 <= mod.thr_at_fpr(np.array([]), 0.05) <= 1.0
    assert mod.thr_at_recall(np.array([]), 0.98) == 0.0


def test_metrics_counts(mod):
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    m = mod._metrics(s, y, 0.5)
    assert m == {"recall": 0.5, "fpr": 0.5, "n_pos": 2, "n_neg": 2}
