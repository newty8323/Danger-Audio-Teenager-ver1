import json

import pytest

from cascade.decision import Thresholds, decide, load_thresholds, save_thresholds


@pytest.fixture
def thr():
    return Thresholds(gate=0.3, acoustic=0.7, text=0.6)


def test_gate_suppresses_trigger(thr):
    d = decide(thr, gate_score=0.1, acoustic_prob=0.99)
    assert not d.gate_passed
    assert d.acoustic_prob is None      # trigger never ran on device
    assert not d.escalate and d.reasons == ()


def test_acoustic_escalates_when_gate_passes(thr):
    d = decide(thr, gate_score=0.5, acoustic_prob=0.8)
    assert d.gate_passed and d.escalate and d.reasons == ("acoustic",)


def test_acoustic_below_threshold_does_not_escalate(thr):
    d = decide(thr, gate_score=0.9, acoustic_prob=0.69)
    assert d.gate_passed and not d.escalate


def test_thresholds_are_inclusive(thr):
    assert decide(thr, gate_score=0.3, acoustic_prob=0.7).escalate
    assert decide(thr, gate_score=0.9, acoustic_prob=0.6, text_prob=0.6).reasons == ("text",)


def test_text_branch_is_independent_of_the_acoustic_gate(thr):
    """Speech runs on its own duty cycle: a gate-suppressed clip can still escalate."""
    d = decide(thr, gate_score=0.0, acoustic_prob=0.9, text_prob=0.95, transcript="…")
    assert not d.gate_passed and d.acoustic_prob is None
    assert d.escalate and d.reasons == ("text",)


def test_both_branches_reported(thr):
    d = decide(thr, gate_score=0.9, acoustic_prob=0.9, text_prob=0.9)
    assert d.reasons == ("acoustic", "text")


def test_gate_disabled_path(thr):
    """Playback capture needs no tier-1 gate (OS playback state gates it for free)."""
    d = decide(thr, gate_score=None, acoustic_prob=0.8, gate_enabled=False)
    assert d.gate_score is None and d.gate_passed and d.escalate


def test_gate_enabled_requires_score(thr):
    with pytest.raises(ValueError):
        decide(thr, gate_score=None, acoustic_prob=0.8)


def test_no_text_score_means_no_text_reason(thr):
    d = decide(thr, gate_score=0.9, acoustic_prob=0.1, text_prob=None)
    assert not d.escalate


def test_threshold_roundtrip(tmp_path, thr):
    p = tmp_path / "sub" / "cascade_thresholds.json"
    save_thresholds(Thresholds(gate=0.3, acoustic=0.7, text=0.6, meta={"fit_split": "val"}), p)
    got = load_thresholds(p)
    assert (got.gate, got.acoustic, got.text) == (0.3, 0.7, 0.6)
    assert got.meta["fit_split"] == "val"
    assert json.loads(p.read_text())["meta"]["fit_split"] == "val"
