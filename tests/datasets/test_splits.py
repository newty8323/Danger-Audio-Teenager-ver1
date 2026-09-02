import pytest

from datasets.manifest import ClipRecord
from datasets.splits import (
    SplitError,
    apply_split,
    assert_source_disjoint,
    assign_splits,
)


def _make_records(n_sources=30, clips_per_source=4, source="audioset"):
    recs = []
    for s in range(n_sources):
        for c in range(clips_per_source):
            recs.append(
                ClipRecord(
                    clip_id=f"{source}_{s}_{c}",
                    source=source,
                    source_id=f"{source}_vid_{s}",
                    start_sec=0.0,
                    duration=10.0,
                    labels=["vio_scream"],
                    label_confidence="weak",
                    split="train",  # placeholder, overwritten by assignment
                )
            )
    return recs


def test_assign_is_source_disjoint_and_deterministic():
    recs = _make_records()
    a1 = assign_splits(recs, seed=42)
    a2 = assign_splits(recs, seed=42)
    assert a1 == a2  # deterministic

    split_recs = apply_split(recs, a1)
    assert_source_disjoint(split_recs)  # must not raise


def test_ratios_are_approximately_honored():
    recs = _make_records(n_sources=100, clips_per_source=5)  # 500 clips
    assignment = assign_splits(recs, ratios=(0.7, 0.15, 0.15), seed=7)
    split_recs = apply_split(recs, assignment)
    counts = {"train": 0, "val": 0, "test": 0}
    for r in split_recs:
        counts[r.split] += 1
    total = sum(counts.values())
    assert abs(counts["train"] / total - 0.70) < 0.05
    assert abs(counts["val"] / total - 0.15) < 0.05
    assert abs(counts["test"] / total - 0.15) < 0.05


def test_in_the_wild_forced_to_test():
    recs = _make_records(n_sources=20) + _make_records(n_sources=5, source="in_the_wild")
    assignment = assign_splits(recs, seed=1)
    split_recs = apply_split(recs, assignment)
    for r in split_recs:
        if r.source == "in_the_wild":
            assert r.split == "test"


def test_assert_source_disjoint_detects_leak():
    recs = _make_records(n_sources=2, clips_per_source=2)
    recs[0].split = "train"
    recs[1].split = "val"  # same source_id as recs[0]
    with pytest.raises(SplitError):
        assert_source_disjoint(recs)


def test_bad_ratios_raise():
    with pytest.raises(SplitError):
        assign_splits(_make_records(), ratios=(0.5, 0.5, 0.5))
