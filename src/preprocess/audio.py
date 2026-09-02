"""Audio loading and the loudness gate (spec §4).

Decoding goes through ffmpeg so that any container/codec (wav, mp3, m4a,
opus, ...) resolves to the same canonical 16 kHz mono float32 stream. This
matches the "ffmpeg -> 16kHz mono" step in the spec and avoids depending on a
torchaudio/soundfile backend being present.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

_FFMPEG = shutil.which("ffmpeg")


class FFmpegNotFoundError(RuntimeError):
    pass


class AudioDecodeError(RuntimeError):
    pass


def load_audio(path: str, sample_rate: int = 16_000, mono: bool = True) -> np.ndarray:
    """Decode ``path`` to a 1-D float32 array in [-1, 1] at ``sample_rate``.

    Uses ffmpeg for resampling and downmixing. Returns mono samples.
    """
    if _FFMPEG is None:
        raise FFmpegNotFoundError("ffmpeg not found on PATH")

    channels = 1 if mono else 2
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-loglevel", "error",
        "-i", path,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "f32le",  # raw 32-bit float little-endian
        "-acodec", "pcm_f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg failed for {path!r}: {msg}")

    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not mono:  # interleaved -> mono via mean
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def rms_dbfs(waveform: np.ndarray) -> float:
    """Root-mean-square level in dBFS (full-scale reference = 1.0).

    Silent input returns -inf.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def passes_rms_gate(waveform: np.ndarray, threshold_dbfs: float = -45.0) -> bool:
    """True if the clip is loud enough to keep (RMS >= threshold)."""
    return rms_dbfs(waveform) >= threshold_dbfs
