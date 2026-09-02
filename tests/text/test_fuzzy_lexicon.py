"""Fuzzy/phonetic recovery of ASR-mangled harm nouns (no model needed — pure jamo)."""

import pytest

from text.fuzzy_lexicon import fuzzy_harm_terms


@pytest.mark.parametrize("text,term", [
    ("필루폰 좀 구해줘", "필로폰"),      # 필로폰 -> 필루폰 (1 jamo)
    ("코케인 어디서 사", "코카인"),      # 코카인 -> 코케인
    ("엑스터씨 오늘 밤에 하자", "엑스터시"),
    ("필하우스 떴다", "풀하우스"),
])
def test_recovers_asr_mangled_nouns(text, term):
    hits = fuzzy_harm_terms(text)
    assert any(h.term == term and 0 < h.distance <= 2 for h in hits), (text, hits)


@pytest.mark.parametrize("text", [
    "코코아 마실래", "필통 어디 있어", "판다 보러 가자", "커피 마시자",
    "로션 발라", "오늘 회의 미루자", "코스 요리 먹자",
])
def test_no_false_match_on_benign(text):
    assert fuzzy_harm_terms(text) == [], (text, fuzzy_harm_terms(text))


def test_exact_match_is_left_to_lexicon():
    # exact term present -> fuzzy does not double-report it (distance would be 0).
    assert all(h.distance > 0 for h in fuzzy_harm_terms("필로폰 좀 구해줘"))
