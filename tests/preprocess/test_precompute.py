import numpy as np

from datasets.manifest import ClipRecord, read_manifest, write_manifest
from preprocess.config import PreprocessConfig
from preprocess.normalize import NormStats
from preprocess.precompute import feature_path, precompute_manifest


def _rec(clip_id, split, labels=("vio_scream",)):
    return ClipRecord(
        clip_id=clip_id,
        source="audioset",
        source_id=f"vid_{clip_id}",
        start_sec=0.0,
        duration=1.0,
        labels=list(labels),
        label_confidence="weak",
        split=split,
    )


def _sine(wav_writer, path, sr=16_000, seconds=1.0, amp=0.4, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    wav_writer(path, amp * np.sin(2 * np.pi * freq * t), sr)


def test_precompute_end_to_end(tmp_path, wav_writer):
    cfg = PreprocessConfig()
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    feature_root = tmp_path / "features"

    records = [
        _rec("t1", "train"),
        _rec("t2", "train"),
        _rec("v1", "val"),
        _rec("quiet", "train"),  # will be RMS-gated
        _rec("missing", "test"),  # no audio file on disk
    ]
    # Audible clips
    for cid in ("t1", "t2", "v1"):
        _sine(wav_writer, audio_root / f"{cid}.wav")
    # Near-silent clip -> gated out
    _sine(wav_writer, audio_root / "quiet.wav", amp=1e-4)

    manifest_in = tmp_path / "in.jsonl"
    write_manifest(records, manifest_in)

    result = precompute_manifest(
        manifest_path=manifest_in,
        audio_root=audio_root,
        feature_root=feature_root,
        stats_path=tmp_path / "artifacts" / "norm.npz",
        output_manifest_path=tmp_path / "out.jsonl",
        cfg=cfg,
    )

    assert set(result.processed_ids) == {"t1", "t2", "v1"}
    assert result.dropped_ids == ["quiet"]
    assert result.missing_ids == ["missing"]

    # Feature files exist for processed clips, not for dropped/missing
    for cid in ("t1", "t2", "v1"):
        assert feature_path(cid, feature_root).exists()
    assert not feature_path("quiet", feature_root).exists()

    # Output manifest drops gated + missing clips
    out = read_manifest(tmp_path / "out.jsonl")
    assert {r.clip_id for r in out} == {"t1", "t2", "v1"}


def test_precompute_stats_fit_on_train_only(tmp_path, wav_writer):
    cfg = PreprocessConfig()
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    feature_root = tmp_path / "features"

    records = [_rec("t1", "train"), _rec("t2", "train"), _rec("v1", "val")]
    for cid in ("t1", "t2", "v1"):
        _sine(wav_writer, audio_root / f"{cid}.wav")
    manifest_in = tmp_path / "in.jsonl"
    write_manifest(records, manifest_in)

    stats_path = tmp_path / "norm.npz"
    precompute_manifest(
        manifest_path=manifest_in,
        audio_root=audio_root,
        feature_root=feature_root,
        stats_path=stats_path,
        output_manifest_path=tmp_path / "out.jsonl",
        cfg=cfg,
    )

    stats = NormStats.load(str(stats_path))
    assert stats.n_mels == cfg.n_mels

    # Stats must equal a manual fit over ONLY the two train features.
    train_feats = [np.load(feature_path(c, feature_root)) for c in ("t1", "t2")]
    concat = np.concatenate([f[0] for f in train_feats], axis=1)
    np.testing.assert_allclose(stats.mean, concat.mean(axis=1), atol=1e-4)
