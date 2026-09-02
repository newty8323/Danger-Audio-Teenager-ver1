import numpy as np
import pytest

from datasets.taxonomy import load_taxonomy
from mining.candidates import (
    FALSE_POSITIVE,
    UNCERTAIN,
    PoolClip,
    read_review_queue,
    select_candidates,
    write_review_queue,
)
from mining.config import MiningConfig


def _pool(n):
    return [PoolClip(f"c{i}", "youtube", f"vid{i}", 0.0, 10.0) for i in range(n)]


def _probs(tax, specs):
    p = np.zeros((len(specs), tax.num_classes))
    for i, (cls, pr) in enumerate(specs):
        p[i, tax.index_of(cls)] = pr
    return p


def test_selects_fp_and_uncertain_and_skips_low():
    tax = load_taxonomy()
    pool = _pool(4)
    probs = _probs(tax, [
        ("vio_gunshot", 0.9),   # FP
        ("vio_scream", 0.5),    # uncertain
        ("vio_impact", 0.2),    # ignored (low)
        ("vio_verbal", 0.65),   # FP
    ])
    cands = select_candidates(pool, probs, tax, MiningConfig())
    # FP first (by prob desc), then uncertain; low one dropped
    assert [c.clip_id for c in cands] == ["c0", "c3", "c1"]
    assert cands[0].reason == FALSE_POSITIVE
    assert cands[0].top_harm_class == "vio_gunshot"
    assert cands[0].top_harm_prob == 0.9
    assert cands[2].reason == UNCERTAIN


def test_top_k_caps_queue():
    tax = load_taxonomy()
    pool = _pool(3)
    probs = _probs(tax, [("vio_gunshot", 0.9), ("vio_scream", 0.8), ("vio_verbal", 0.7)])
    cands = select_candidates(pool, probs, tax, MiningConfig(top_k=2))
    assert len(cands) == 2
    assert [c.clip_id for c in cands] == ["c0", "c1"]  # highest-prob FPs


def test_uncertain_priority_favors_midpoint():
    tax = load_taxonomy()
    pool = _pool(2)
    probs = _probs(tax, [("vio_scream", 0.5), ("vio_scream", 0.41)])
    cands = select_candidates(pool, probs, tax, MiningConfig())
    assert [c.clip_id for c in cands] == ["c0", "c1"]  # 0.5 (closest to .5) first


def test_clap_pseudo_label_attached():
    tax = load_taxonomy()
    pool = _pool(1)
    probs = _probs(tax, [("vio_gunshot", 0.9)])
    cands = select_candidates(pool, probs, tax, clap_labels={"c0": "firework"})
    assert cands[0].clap_pseudo_label == "firework"


def test_length_mismatch_raises():
    tax = load_taxonomy()
    with pytest.raises(ValueError):
        select_candidates(_pool(2), _probs(tax, [("vio_gunshot", 0.9)]), tax)


def test_wrong_probs_shape_raises():
    tax = load_taxonomy()
    with pytest.raises(ValueError):
        select_candidates(_pool(1), np.zeros((1, 5)), tax)  # not num_classes wide
    with pytest.raises(ValueError):
        select_candidates(_pool(1), np.zeros(tax.num_classes), tax)  # 1-D


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        MiningConfig(top_k=0)
    with pytest.raises(ValueError):
        MiningConfig(max_iterations=0)


def test_queue_roundtrip(tmp_path):
    tax = load_taxonomy()
    pool = _pool(2)
    probs = _probs(tax, [("vio_gunshot", 0.9), ("vio_scream", 0.5)])
    cands = select_candidates(pool, probs, tax)
    path = tmp_path / "queue.jsonl"
    write_review_queue(cands, path)
    back = read_review_queue(path)
    assert [c.clip_id for c in back] == [c.clip_id for c in cands]
    assert back[0].top_harm_class == cands[0].top_harm_class
