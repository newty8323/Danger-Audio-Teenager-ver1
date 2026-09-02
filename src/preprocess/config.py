"""Preprocessing hyperparameters (spec §4).

Single source of truth for the log-mel pipeline. Mirrored by
``configs/data/preprocess.yaml`` for hydra; keep the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    # Target audio format
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    mono: bool = True

    # Loudness gate: drop clip if RMS < this (dBFS, full-scale ref = 1.0)
    rms_gate_dbfs: float = -45.0

    # STFT
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024

    # Mel filterbank
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0

    # log(mel + offset)
    log_offset: float = 1e-6

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def expected_frames(self) -> int:
        """Frame count for a full clip with center-padded STFT."""
        return self.clip_samples // self.hop_length + 1
