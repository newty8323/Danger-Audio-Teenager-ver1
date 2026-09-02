"""Speech gating: the cases that caused a live false escalation must be rejected."""
import numpy as np

from app.vad import SR, is_degenerate, speech_score
from tests.app.audio_fixtures import hum as _hum
from tests.app.audio_fixtures import noise as _white
from tests.app.audio_fixtures import speechlike as _speechlike


def test_speech_scores_higher_than_noise():
    assert speech_score(_speechlike()) > speech_score(_white())


def test_loud_white_noise_is_rejected_by_default_threshold():
    """The live failure: loud crowd/applause noise passed an RMS gate and hit ASR."""
    assert speech_score(_white(amp=0.5)) < 0.35


def test_steady_hum_is_rejected():
    assert speech_score(_hum()) < 0.35


def test_silence_is_zero():
    assert speech_score(np.zeros(SR * 5, dtype=np.float32)) == 0.0


def test_too_short_is_zero():
    assert speech_score(np.ones(100, dtype=np.float32) * 0.1) == 0.0


def test_degenerate_catches_the_observed_hallucination():
    text = "와! " * 34
    assert is_degenerate(text)


def test_degenerate_catches_repeated_phrase_loop():
    assert is_degenerate("스탑 너무 좋아. 스탑 너무 좋아. 스탑 너무 좋아.")
    assert is_degenerate("당신은 당신을 찾아와. 당신은 당신을 찾아와. 당신은 당신을 찾아와.")


def test_emphatic_repetition_in_real_speech_is_not_degenerate():
    """Observed on a real movie clip — this filter used to discard the whole window."""
    assert not is_degenerate("뭐야 우리 형님이 거짓의 깨 깨 깨 확 가뜩이나 요즘 가오 죽어서")
    assert not is_degenerate("야 야 야 잠깐만 기다려 봐 지금 나간다고")


def test_long_single_token_loop_is_still_degenerate():
    assert is_degenerate("깨 깨 깨 깨 깨 깨 깨 깨")


def test_real_sentences_pass():
    for s in ("묻어버릴 줄 알아 진짜 칼 들고 찾아갈 거야",
              "오늘 밤 도박장 가자 진짜 이번 판 올인이다",
              "내일 회의는 세 시에 시작합니다 자료는 미리 공유해 주세요"):
        assert not is_degenerate(s), s


def test_short_text_is_not_judged():
    assert not is_degenerate("살려줘")
    assert not is_degenerate("")
