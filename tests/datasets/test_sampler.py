import pytest

from datasets.manifest import ClipRecord
from datasets.sampler import BalancedBatchSampler
from datasets.taxonomy import load_taxonomy


def _rec(clip_id, labels, source_id=None):
    return ClipRecord(
        clip_id=clip_id,
        source="audioset",
        source_id=source_id or f"vid_{clip_id}",
        start_sec=0.0,
        duration=10.0,
        labels=labels,
        label_confidence="weak",
        split="train",
    )


def _records(n_harm, n_other):
    recs = [_rec(f"h{i}", ["vio_scream"]) for i in range(n_harm)]
    recs += [_rec(f"o{i}", ["asmr"]) for i in range(n_other)]
    return recs


def test_batches_are_balanced():
    tax = load_taxonomy()
    recs = _records(n_harm=40, n_other=40)
    sampler = BalancedBatchSampler(recs, tax, batch_size=8, seed=1)
    harm_ids = set(range(40))
    for batch in sampler:
        assert len(batch) == 8
        n_harm = sum(1 for i in batch if i in harm_ids)
        assert n_harm == 4  # 1:1 within the batch


def test_deterministic_with_seed_and_epoch():
    tax = load_taxonomy()
    recs = _records(30, 50)
    s1 = BalancedBatchSampler(recs, tax, batch_size=10, seed=7)
    s2 = BalancedBatchSampler(recs, tax, batch_size=10, seed=7)
    assert list(s1) == list(s2)


def test_epoch_changes_order():
    tax = load_taxonomy()
    recs = _records(30, 50)
    s = BalancedBatchSampler(recs, tax, batch_size=10, seed=7)
    s.set_epoch(0)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert e0 != e1


def test_majority_pool_is_covered():
    tax = load_taxonomy()
    recs = _records(n_harm=10, n_other=60)  # other is majority
    sampler = BalancedBatchSampler(recs, tax, batch_size=8, seed=3)
    seen_other = set()
    for batch in sampler:
        seen_other.update(i for i in batch if i >= 10)
    # Every "other" index (10..69) should appear at least once in an epoch.
    assert seen_other == set(range(10, 70))


def test_length_matches_iteration():
    tax = load_taxonomy()
    recs = _records(25, 40)
    sampler = BalancedBatchSampler(recs, tax, batch_size=10, seed=2)
    assert len(sampler) == len(list(sampler))


def test_batch_size_too_small_raises():
    tax = load_taxonomy()
    with pytest.raises(ValueError):
        BalancedBatchSampler(_records(2, 2), tax, batch_size=1)


def test_no_duplicate_indices_within_batch():
    tax = load_taxonomy()
    recs = _records(n_harm=20, n_other=20)
    sampler = BalancedBatchSampler(recs, tax, batch_size=8, seed=5)
    for batch in sampler:
        assert len(set(batch)) == len(batch)  # no trivial SupCon positives


def test_warns_when_pool_below_quota():
    tax = load_taxonomy()
    recs = _records(n_harm=1, n_other=50)  # 1 harm, quota is 4
    with pytest.warns(UserWarning):
        BalancedBatchSampler(recs, tax, batch_size=8)
