"""Hybrid text scorer: the lexicon must add slang recall without firing on ordinary Korean."""
import numpy as np
import pytest

from cascade.pipeline import HybridTextScorer, split_harm_text_units


class FakeClf:
    """Stands in for KoELECTRA — the point is what the LEXICON adds, not the classifier."""

    def __init__(self, value=0.05):
        self.value = value
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        return np.full(len(texts), self.value, dtype=np.float32)


@pytest.fixture(scope="module")
def hybrid():
    return HybridTextScorer(classifier=FakeClf(0.05))


@pytest.mark.parametrize("text", [
    "오늘 밤에 사다리 한 판 돌리자",      # gambling slang, phrase-gated
    "먹튀 없는 사이트니까 걱정 마",
    "환전 빠른 총판 아는 사람",
    "일탈계인데 디엠 열어둘게",
    "물뽕 있으면 좀 나눠줘",
    "작대기 맞으면 기분 끝내준다던데",
    "야 우리 오늘 떨 한번 할래",
])
def test_slang_is_flagged(hybrid, text):
    assert hybrid.score([text])[0] > 0.5, text


@pytest.mark.parametrize("text", [
    "내일 회의는 세 시에 시작합니다 자료는 미리 공유해 주세요",
    "오늘 조건이 안 맞아서 회의를 미뤘어요",          # 조건 alone must not fire
    "그 썰 들었어? 어제 발표 잘했다더라",             # 썰 alone must not fire
    "사다리 타고 올라가서 전구 갈았어",               # 사다리 as a real ladder
    "고기 사러 마트 갔다 왔어",                      # deliberately not in the lexicon
    "아이스 아메리카노 한 잔 주세요",
])
def test_ordinary_korean_does_not_fire(hybrid, text):
    assert hybrid.score([text])[0] <= 0.5, text


def test_observed_asr_corruption_is_covered(hybrid):
    """예짤 / 목키 are what Moonshine produced for 야짤 / 먹튀 — listed exactly, not fuzzy."""
    assert hybrid.score(["예짤 보내줄 사람 있나"])[0] > 0.5
    assert hybrid.score(["목키 없는 사이트니까 걱정 마"])[0] > 0.5


def test_fuzzy_recovers_long_term_near_miss(hybrid):
    assert hybrid.lexicon_score("필루폰 구할 수 있어") > 0.5      # 필로폰, 1 jamo off


def test_short_fuzzy_stays_disabled():
    """2-syllable fuzzy matched "야빨" in 4/300 real sentences — it must stay off."""
    from text.fuzzy_lexicon import FUZZY_TERMS_SHORT
    assert FUZZY_TERMS_SHORT == {}


def test_classifier_still_dominates_when_it_is_more_alarmed():
    h = HybridTextScorer(classifier=FakeClf(0.97))
    assert h.score(["아무 문제 없는 문장입니다"])[0] == pytest.approx(0.97, abs=1e-3)


def test_score_length_matches_input(hybrid):
    assert len(hybrid.score(["가", "나", "다"])) == 3
    assert len(hybrid.score([])) == 0


def test_long_transcript_exposes_short_abusive_ending(hybrid):
    units = split_harm_text_units(
        "우리 형님이 저런 분으로 보여? 우리 형님이 거짓할게 이 새끼야!"
    )
    assert "우리 형님이 거짓할게 이 새끼야!" in units
    assert hybrid.score(["앞부분은 평범한 설명입니다. 이 새끼야!"])[0] > 0.9


def test_does_not_invent_short_suffixes_from_long_safe_sentence():
    units = split_harm_text_units(
        "상급자의 조치를 비롯한 경찰 내부 검증 시스템이 왜 작동하지 않았는지는 "
        "반드시 규명해야 할 과제로 꼽힙니다."
    )
    assert "할 과제로 꼽힙니다." not in units
