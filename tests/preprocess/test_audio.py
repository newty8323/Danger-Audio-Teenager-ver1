import numpy as np
import pytest

from preprocess.audio import load_audio, passes_rms_gate, rms_dbfs


def test_load_audio_resamples_to_target_sr(sine_wav):
    path, _ = sine_wav(freq=440.0, seconds=10.0, sr=44_100)
    audio = load_audio(str(path), sample_rate=16_000)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # 10s at 16kHz, allow a few samples of codec slack
    assert abs(len(audio) - 160_000) < 2_000


def test_rms_dbfs_full_scale_sine():
    sr = 16_000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 440 * t)  # amplitude 1.0
    # RMS of a full-scale sine is 1/sqrt(2) -> ~ -3.01 dBFS
    assert rms_dbfs(x) == pytest.approx(-3.01, abs=0.1)


def test_rms_dbfs_silence_is_neg_inf():
    assert rms_dbfs(np.zeros(1000)) == float("-inf")
    assert rms_dbfs(np.array([])) == float("-inf")


def test_rms_gate_keeps_loud_drops_quiet():
    sr = 16_000
    t = np.arange(sr) / sr
    loud = 0.5 * np.sin(2 * np.pi * 440 * t)  # ~ -9 dBFS
    quiet = 1e-4 * np.sin(2 * np.pi * 440 * t)  # ~ -80 dBFS
    assert passes_rms_gate(loud, threshold_dbfs=-45.0) is True
    assert passes_rms_gate(quiet, threshold_dbfs=-45.0) is False


def test_rms_gate_drops_silence(silent_wav):
    path, _ = silent_wav()
    audio = load_audio(str(path), sample_rate=16_000)
    assert passes_rms_gate(audio, threshold_dbfs=-45.0) is False
