from __future__ import annotations

import json

import pytest

from mobile_asr.manifest import domain_counts, load_manifest


def _write(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    return path


def test_teacher_text_is_used_only_without_human_text(tmp_path):
    rows = [
        {"id": "a", "audio": "a.wav", "text": "사람 정답", "teacher_text": "교사 출력",
         "domain": "general", "split": "train", "source_id": "s1"},
        {"id": "b", "audio": "b.wav", "teacher_text": "노래 가사",
         "domain": "song", "split": "val", "source_id": "s2"},
    ]
    loaded = load_manifest(_write(tmp_path, rows), require_audio=False)
    assert loaded[0].text == "사람 정답"
    assert loaded[0].transcript_source == "human"
    assert loaded[1].text == "노래 가사"
    assert loaded[1].transcript_source == "teacher"
    assert domain_counts(loaded) == {"general": 1, "movie": 0, "song": 1, "no_speech": 0}


def test_no_speech_requires_empty_text(tmp_path):
    rows = [{"id": "a", "audio": "a.wav", "text": "환각",
             "domain": "no_speech", "split": "test", "source_id": "s1"}]
    with pytest.raises(ValueError, match="no_speech text must be empty"):
        load_manifest(_write(tmp_path, rows), require_audio=False)


def test_source_cannot_cross_splits(tmp_path):
    rows = [
        {"id": "a", "audio": "a.wav", "text": "하나", "domain": "general",
         "split": "train", "source_id": "same"},
        {"id": "b", "audio": "b.wav", "text": "둘", "domain": "movie",
         "split": "test", "source_id": "same"},
    ]
    with pytest.raises(ValueError, match="source-disjoint"):
        load_manifest(_write(tmp_path, rows), require_audio=False)

