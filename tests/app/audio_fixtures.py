"""Synthetic waveforms shared by the app tests (no data files needed)."""
import numpy as np

SR = 16000


def speechlike(sec: float = 10.0, seed: int = 0, amp: float = 0.4) -> np.ndarray:
    """Band-limited, harmonic, syllable-modulated — passes app.vad.speech_score.

    The pitch contour must be integrated into PHASE (cumulative sum), not multiplied by t:
    `sin(2*pi*f0(t)*t)` sweeps its instantaneous frequency upward without bound, so after a
    few seconds the harmonics leave the voice band and the fixture stops looking like speech.
    """
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    t = np.arange(n) / SR
    f0 = 130 + 25 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in (1, 2, 3, 4, 6, 8))
    syllable = (np.sin(2 * np.pi * 4.0 * t) > -0.2).astype(np.float32)
    env = np.convolve(syllable, np.hanning(400) / 200, mode="same")
    sig = voiced * env + 0.01 * rng.standard_normal(n)
    return (sig / np.abs(sig).max() * amp).astype(np.float32)


def noise(sec: float = 10.0, amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """White noise — loud but NOT speech (rejected by the speech gate)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sec * SR)) * amp).astype(np.float32)


def hum(sec: float = 10.0, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
