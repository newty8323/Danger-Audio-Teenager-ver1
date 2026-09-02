from pathlib import Path

import pytest

from mobile_asr.manifest import ASRItem
from mobile_asr.metrics import evaluate_predictions, normalize_text


def _row(item_id, text, domain, terms=()):
    return ASRItem(item_id, Path("dummy.wav"), text, domain, "test", item_id,
                   harm_terms=terms)


def test_normalization_keeps_korean_and_latin_code_switch():
    assert normalize_text("안녕, But I Love!") == "안녕butilove"


def test_metrics_separate_speech_domains_and_no_speech_hallucination():
    rows = [
        _row("g", "오늘 학교에 갔다", "general"),
        _row("m", "당장 나가", "movie", ("나가",)),
        _row("s", "널 사랑해", "song"),
        _row("n", "", "no_speech"),
    ]
    result = evaluate_predictions(rows, {"g": "오늘 학교에 갔다", "m": "당장 나가",
                                         "s": "널 사랑해", "n": "그렇지 그렇지"})
    assert result["general"]["cer"] == 0
    assert result["movie"]["cer"] == 0
    assert result["song"]["cer"] == 0
    assert result["no_speech"]["false_transcript_rate"] == 1
    assert result["harm_term_recall"] == 1


def test_missing_hypothesis_is_an_error():
    with pytest.raises(ValueError, match="missing hypotheses"):
        evaluate_predictions([_row("g", "안녕", "general")], {})

