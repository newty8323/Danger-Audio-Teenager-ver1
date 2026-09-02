import numpy as np
import torch

from datasets.taxonomy import load_taxonomy
from infer_stream import StreamRiskInference, make_model_predictor, top_events
from models.harm_model import HarmModel, ModelConfig
from preprocess.normalize import NormStats
from risk.policy import load_risk_policy

SR = 16_000


class StubScorer:
    """Returns a fixed risk regardless of probs, so the driver's stride/level
    logic can be tested without a fitted model."""

    def __init__(self, value):
        self.value = value

    def score(self, probs, require_fitted=True):
        return self.value


def _fixed_probs(tax):
    p = np.zeros(tax.num_classes)
    p[tax.index_of("vio_gunshot")] = 0.8
    p[tax.index_of("vio_scream")] = 0.5
    p[tax.index_of("door")] = 0.3
    return p


def _infer(scorer_value, tax=None):
    tax = tax or load_taxonomy()
    return StreamRiskInference(tax, StubScorer(scorer_value), load_risk_policy(), sample_rate=SR)


def test_top_events_returns_sorted_top3():
    tax = load_taxonomy()
    ev = top_events(_fixed_probs(tax), tax, k=3)
    assert [e["class"] for e in ev] == ["vio_gunshot", "vio_scream", "door"]
    assert ev[0]["prob"] == 0.8


def test_default_stride_when_safe():
    tax = load_taxonomy()
    infer = _infer(0.1, tax)
    wave = np.zeros(40 * SR, dtype=np.float32)
    results = infer.run(wave, lambda w: _fixed_probs(tax))
    starts = [r.start_sec for r in results]
    assert starts == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # 5s stride
    assert all(r.risk_level == "safe" for r in results)


def test_warn_densifies_stride():
    tax = load_taxonomy()
    infer = _infer(0.5, tax)  # 0.5 -> warn
    wave = np.zeros(40 * SR, dtype=np.float32)
    results = infer.run(wave, lambda w: _fixed_probs(tax))
    assert results[0].risk_level == "warn"
    assert results[0].stride_s == 2.5
    assert results[1].start_sec == 2.5  # densified


def test_three_warns_escalate_to_block_in_stream():
    tax = load_taxonomy()
    infer = _infer(0.5, tax)
    wave = np.zeros(40 * SR, dtype=np.float32)
    results = infer.run(wave, lambda w: _fixed_probs(tax))
    assert [r.risk_level for r in results[:3]] == ["warn", "warn", "block"]


def test_result_matches_9b_schema():
    tax = load_taxonomy()
    infer = _infer(0.5, tax)
    wave = np.zeros(10 * SR, dtype=np.float32)  # exactly one window
    results = infer.run(wave, lambda w: _fixed_probs(tax), clip_id="demo")
    assert len(results) == 1
    d = results[0].to_dict()
    assert {"clip_id", "probs", "risk_score", "risk_level", "top_events"} <= set(d)
    assert len(d["probs"]) == tax.num_classes
    assert len(d["top_events"]) == 3
    assert d["clip_id"] == "demo@0s"


def test_no_windows_when_audio_shorter_than_window():
    tax = load_taxonomy()
    infer = _infer(0.1, tax)
    wave = np.zeros(5 * SR, dtype=np.float32)  # < 10s window
    assert infer.run(wave, lambda w: _fixed_probs(tax)) == []


def test_make_model_predictor_smoke():
    tax = load_taxonomy()
    model = HarmModel(tax.num_classes, ModelConfig())
    stats = NormStats(mean=np.zeros(128, np.float32), std=np.ones(128, np.float32))
    predict = make_model_predictor(model, stats, torch.device("cpu"))
    probs = predict(np.random.randn(10 * SR).astype(np.float32))
    assert probs.shape == (tax.num_classes,)
    assert (probs >= 0).all() and (probs <= 1).all()
