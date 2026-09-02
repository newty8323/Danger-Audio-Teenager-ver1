import pytest

from datasets.manifest import validate_manifest
from datasets.taxonomy import load_taxonomy
from mining.candidates import FALSE_POSITIVE as FP_REASON
from mining.candidates import UNCERTAIN, ReviewCandidate, write_review_queue
from mining.review import ReviewSession


def _cand(clip_id, cls, reason=FP_REASON):
    return ReviewCandidate(
        clip_id=clip_id, source="youtube", source_id=f"v_{clip_id}",
        start_sec=0.0, duration=10.0, top_harm_class=cls, top_harm_prob=0.8,
        reason=reason, priority=0.8,
    )


def _session():
    return ReviewSession([
        _cand("a", "vio_scream"),
        _cand("b", "vio_gunshot"),
        _cand("c", "vio_impact", reason=UNCERTAIN),
    ])


def test_progress_and_pending():
    s = _session()
    assert s.progress() == (0, 3)
    s.decide("a", "false_positive", "chair_scrape")
    assert s.progress() == (1, 3)
    assert [c.clip_id for c in s.pending()] == ["b", "c"]


def test_skip_moves_clip_to_back_of_pending():
    s = _session()
    s.decide("a", "skip")
    assert s.progress() == (0, 3)  # skip is not "done"
    pend = [c.clip_id for c in s.pending()]
    assert pend[0] != "a"  # skip advances to the next clip
    assert pend == ["b", "c", "a"]  # skipped clip pushed to the back


def test_decide_validates_label_kind_when_taxonomy_set():
    s = ReviewSession(_session().candidates, taxonomy=load_taxonomy())
    with pytest.raises(ValueError):
        s.decide("a", "false_positive", "vio_gunshot")  # harm label for an FP
    with pytest.raises(ValueError):
        s.decide("a", "positive", "door")  # confusable label for a positive
    s.decide("a", "false_positive", "chair_scrape")  # valid
    s.decide("b", "positive", "vio_gunshot")  # valid


def test_decide_validates_action_and_label():
    s = _session()
    with pytest.raises(KeyError):
        s.decide("ghost", "reject")
    with pytest.raises(ValueError):
        s.decide("a", "bogus")
    with pytest.raises(ValueError):
        s.decide("a", "false_positive")  # missing label


def test_export_builds_fp_and_positive_records():
    tax = load_taxonomy()
    s = _session()
    s.decide("a", "false_positive", "chair_scrape")  # hard negative
    s.decide("b", "reject")                            # dropped
    s.decide("c", "positive", "vio_impact")            # recovered harm positive
    recs = s.export(tax)
    assert {r.clip_id for r in recs} == {"a", "c"}
    labels = {r.clip_id: r.labels[0] for r in recs}
    assert labels == {"a": "chair_scrape", "c": "vio_impact"}
    assert validate_manifest(recs, tax) == []


def test_decisions_save_load_roundtrip(tmp_path):
    tax = load_taxonomy()
    s = _session()
    s.decide("a", "false_positive", "chair_scrape")
    s.decide("c", "positive", "vio_impact")
    path = tmp_path / "decisions.jsonl"
    s.save_decisions(path)

    resumed = _session().load_decisions(path)
    assert resumed.progress() == (2, 3)
    assert {r.clip_id for r in resumed.export(tax)} == {"a", "c"}


def test_from_queue(tmp_path):
    path = tmp_path / "q.jsonl"
    write_review_queue([_cand("a", "vio_scream")], path)
    s = ReviewSession.from_queue(path)
    assert [c.clip_id for c in s.candidates] == ["a"]
