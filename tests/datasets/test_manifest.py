import pytest

from datasets.manifest import (
    ClipRecord,
    ManifestError,
    load_manifest,
    read_manifest,
    training_records,
    validate_manifest,
    write_manifest,
)
from datasets.taxonomy import load_taxonomy


def _rec(
    clip_id,
    split="train",
    labels=None,
    confidence="verified",
    flagged=False,
    source="audioset",
):
    return ClipRecord(
        clip_id=clip_id,
        source=source,
        source_id=f"vid_{clip_id}",
        start_sec=0.0,
        duration=10.0,
        labels=labels if labels is not None else ["vio_scream"],
        label_confidence=confidence,
        split=split,
        flagged=flagged,
    )


def test_roundtrip_jsonl(tmp_path):
    recs = [_rec("a"), _rec("b", split="val", labels=["asmr", "door"])]
    path = tmp_path / "m.jsonl"
    write_manifest(recs, path)
    back = read_manifest(path)
    assert [r.clip_id for r in back] == ["a", "b"]
    assert back[1].labels == ["asmr", "door"]
    assert back[1].split == "val"


def test_validate_good_manifest_passes():
    tax = load_taxonomy()
    assert validate_manifest([_rec("a"), _rec("b", split="test")], tax) == []


def test_validate_flags_bad_fields():
    tax = load_taxonomy()
    bad = [
        _rec("x", confidence="guessed"),
        _rec("y", split="holdout"),
        _rec("z", labels=["not_a_class"]),
        ClipRecord("neg", "s", "sid", -1.0, 0.0, ["door"], "weak", "train"),
    ]
    errors = validate_manifest(bad, tax)
    problems = " ".join(e.problem for e in errors)
    assert "label_confidence" in problems
    assert "split" in problems
    assert "unknown label" in problems
    assert "duration" in problems


def test_validate_duplicate_clip_id():
    tax = load_taxonomy()
    errors = validate_manifest([_rec("dup"), _rec("dup", split="val")], tax)
    assert any("duplicate" in e.problem for e in errors)


def test_load_manifest_raises_on_invalid(tmp_path):
    tax = load_taxonomy()
    path = tmp_path / "m.jsonl"
    write_manifest([_rec("a", labels=["bogus"])], path)
    with pytest.raises(ManifestError):
        load_manifest(path, tax)


def test_training_records_excludes_val_test_and_flagged():
    recs = [
        _rec("train_keep", split="train"),
        _rec("train_flagged", split="train", flagged=True),
        _rec("val_rec", split="val"),
        _rec("test_rec", split="test"),
    ]
    kept = [r.clip_id for r in training_records(recs)]
    assert kept == ["train_keep"]


def test_multihot_from_record():
    tax = load_taxonomy()
    vec = _rec("a", labels=["vio_scream", "clap"]).multihot(tax)
    assert vec.sum() == 2.0
