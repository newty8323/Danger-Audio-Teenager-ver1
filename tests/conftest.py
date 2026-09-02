"""Shared fixtures: synthesize test WAVs with the stdlib (no extra deps)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a mono 16-bit PCM WAV from float samples in [-1, 1]."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def wav_writer():
    """Return the low-level WAV writer for building custom fixtures."""
    return _write_wav


@pytest.fixture
def sine_wav(tmp_path: Path):
    def _make(freq: float = 440.0, seconds: float = 10.0, sr: int = 16_000, amp: float = 0.5):
        t = np.arange(int(seconds * sr)) / sr
        samples = amp * np.sin(2 * np.pi * freq * t)
        return _write_wav(tmp_path / f"sine_{int(freq)}_{amp}.wav", samples, sr), sr

    return _make


@pytest.fixture
def silent_wav(tmp_path: Path):
    def _make(seconds: float = 10.0, sr: int = 16_000):
        samples = np.zeros(int(seconds * sr), dtype=np.float32)
        return _write_wav(tmp_path / "silence.wav", samples, sr), sr

    return _make
