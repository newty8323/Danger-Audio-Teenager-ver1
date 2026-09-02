import numpy as np

from datasets.dataset import AugmentConfig, LogMelDataset
from datasets.manifest import ClipRecord
from datasets.taxonomy import load_taxonomy
from preprocess.normalize import NormStats

F, T = 128, 50


def _rec(clip_id, labels):
    return ClipRecord(
        clip_id=clip_id,
        source="audioset",
        source_id=f"vid_{clip_id}",
        start_sec=0.0,
        duration=10.0,
        labels=labels,
        label_confidence="weak",
        split="train",
    )


def _setup(tmp_path, n=4):
    tax = load_taxonomy()
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        labels = ["sex_moan"] if i % 2 == 0 else ["asmr"]
        records.append(_rec(f"c{i}", labels))
        np.save(feature_root / f"c{i}.npy", rng.standard_normal((1, F, T)).astype(np.float32))
    stats = NormStats(mean=np.zeros(F, dtype=np.float32), std=np.ones(F, dtype=np.float32))
    return records, feature_root, tax, stats


def test_eval_shapes_and_labels(tmp_path):
    records, feature_root, tax, stats = _setup(tmp_path)
    ds = LogMelDataset(records, feature_root, tax, stats, train=False)
    feat, label = ds[0]
    assert tuple(feat.shape) == (1, F, T)
    assert tuple(label.shape) == (tax.num_classes,)
    assert label[tax.index_of("sex_moan")] == 1.0


def test_eval_is_deterministic(tmp_path):
    records, feature_root, tax, stats = _setup(tmp_path)
    ds = LogMelDataset(records, feature_root, tax, stats, train=False)
    f1, _ = ds[1]
    f2, _ = ds[1]
    np.testing.assert_array_equal(f1.numpy(), f2.numpy())


def test_train_with_no_aug_matches_eval(tmp_path):
    records, feature_root, tax, stats = _setup(tmp_path)
    no_aug = AugmentConfig(gain_p=0.0, time_shift_p=0.0, spec_p=0.0, mixup_p=0.0)
    train_ds = LogMelDataset(records, feature_root, tax, stats, train=True, augment_cfg=no_aug)
    eval_ds = LogMelDataset(records, feature_root, tax, stats, train=False)
    ft, _ = train_ds[2]
    fe, _ = eval_ds[2]
    np.testing.assert_allclose(ft.numpy(), fe.numpy(), atol=1e-5)


def test_augmentation_varies_across_epochs(tmp_path):
    records, feature_root, tax, stats = _setup(tmp_path)
    ds = LogMelDataset(records, feature_root, tax, stats, train=True)
    ds.set_epoch(0)
    f0, _ = ds[0]
    ds.set_epoch(1)
    f1, _ = ds[0]
    assert not np.array_equal(f0.numpy(), f1.numpy())


def test_train_getitem_deterministic_within_epoch(tmp_path):
    records, feature_root, tax, stats = _setup(tmp_path)
    ds = LogMelDataset(records, feature_root, tax, stats, train=True)
    ds.set_epoch(3)
    a, la = ds[0]
    b, lb = ds[0]
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(la.numpy(), lb.numpy())
