import pytest

from datasets.manifest import validate_manifest
from datasets.taxonomy import load_taxonomy
from mining.candidates import FALSE_POSITIVE, UNCERTAIN, ReviewCandidate
from mining.config import MiningConfig
from mining.hnm import (
    fp_distribution,
    promote_false_positives,
    promote_positives,
    should_stop,
)


def _cand(clip_id, cls, reason=FALSE_POSITIVE):
    return ReviewCandidate(
        clip_id=clip_id, source="youtube", source_id=f"v_{clip_id}",
        start_sec=0.0, duration=10.0, top_harm_class=cls, top_harm_prob=0.8,
        reason=reason, priority=0.8,
    )


def test_fp_distribution_counts_only_false_positives():
    cands = [
        _cand("a", "vio_scream"),
        _cand("b", "vio_scream"),
        _cand("c", "vio_gunshot"),
        _cand("d", "vio_impact", reason=UNCERTAIN),  # not counted
    ]
    assert fp_distribution(cands) == {"vio_scream": 2, "vio_gunshot": 1}


def test_promote_false_positives_builds_valid_train_records():
    tax = load_taxonomy()
    cands = [_cand("a", "vio_scream"), _cand("b", "vio_impact")]
    recs = promote_false_positives(cands, {"a": "chair_scrape", "b": "door"}, tax)
    assert len(recs) == 2
    assert all(r.split == "train" and r.label_confidence == "verified" for r in recs)
    assert {r.labels[0] for r in recs} == {"chair_scrape", "door"}
    assert validate_manifest(recs, tax) == []  # taxonomy-valid


def test_promote_rejects_harm_label():
    tax = load_taxonomy()
    with pytest.raises(ValueError):
        promote_false_positives([_cand("a", "vio_scream")], {"a": "vio_gunshot"}, tax)


def test_promote_positives_builds_harm_train_records():
    tax = load_taxonomy()
    cands = [_cand("a", "vio_scream", reason=UNCERTAIN)]
    recs = promote_positives(cands, {"a": "vio_scream"}, tax)  # confirmed missed positive
    assert len(recs) == 1
    assert recs[0].labels == ["vio_scream"] and recs[0].split == "train"
    assert recs[0].label_confidence == "verified"


def test_promote_positives_rejects_confusable_label():
    tax = load_taxonomy()
    with pytest.raises(ValueError):
        promote_positives([_cand("a", "vio_scream", reason=UNCERTAIN)], {"a": "door"}, tax)


def test_promote_rejects_unknown_label_and_clip():
    tax = load_taxonomy()
    cands = [_cand("a", "vio_scream")]
    with pytest.raises(ValueError):
        promote_false_positives(cands, {"a": "not_a_class"}, tax)
    with pytest.raises(KeyError):
        promote_false_positives(cands, {"ghost": "door"}, tax)


def test_should_stop_on_max_iterations():
    cfg = MiningConfig(max_iterations=3)
    assert should_stop([0.10, 0.08, 0.06], cfg) is True
    assert should_stop([0.10, 0.08], cfg) is False


def test_should_stop_on_plateau():
    cfg = MiningConfig(max_iterations=5, min_fpr_improvement=0.005)
    assert should_stop([0.10, 0.098], cfg) is True   # improved only 0.002
    assert should_stop([0.10, 0.09], cfg) is False    # improved 0.01


def test_should_stop_first_iteration_continues():
    assert should_stop([0.10], MiningConfig(max_iterations=3)) is False
