import numpy as np

from app.ver1_audio import compact_audible_audio, remove_leading_overlap


def test_compact_audible_audio_removes_long_silence_and_keeps_voice():
    sr = 16_000
    voice = np.full(sr, 0.1, dtype=np.float32)
    wave = np.concatenate([np.zeros(sr), voice, np.zeros(sr), voice])

    result = compact_audible_audio(wave, threshold_db=-50.0)

    assert result["kept_seconds"] >= 2.0
    assert result["removed_seconds"] > 1.0
    assert result["audio"].size > 2 * sr


def test_remove_leading_overlap_deduplicates_context_words():
    previous = "아 의미 없어 모든 게"
    recovered = "모든 게 들렸어 다시 잠에 들었어"

    assert remove_leading_overlap(previous, recovered) == "들렸어 다시 잠에 들었어"
