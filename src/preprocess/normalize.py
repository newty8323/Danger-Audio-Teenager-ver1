"""Per-mel-bin normalization stats (spec §4).

Mean/std are computed per mel bin over all time frames of the *train* split
and saved as a versioned artifact (``.npz``). Features are stored unnormalized;
normalization is applied at load time so the stats stay swappable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray  # (n_mels,)
    std: np.ndarray  # (n_mels,)

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape or self.mean.ndim != 1:
            raise ValueError("mean/std must be matching 1-D arrays")

    @property
    def n_mels(self) -> int:
        return int(self.mean.shape[0])

    def save(self, path: str) -> None:
        np.savez(
            path,
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str) -> NormStats:
        data = np.load(path)
        return cls(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))


def fit_norm_stats(logmels: Iterable[np.ndarray]) -> NormStats:
    """Fit per-bin mean/std from an iterable of (1, n_mels, T) log-mels.

    Uses a streaming sum/sum-of-squares so it never holds all clips in memory.
    """
    total = None  # sum over frames, per bin
    total_sq = None
    count = 0

    for lm in logmels:
        arr = np.asarray(lm, dtype=np.float64)
        if arr.ndim == 3:  # (1, n_mels, T) -> (n_mels, T)
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"expected (1, n_mels, T) or (n_mels, T), got shape {arr.shape}")
        if total is None:
            total = np.zeros(arr.shape[0], dtype=np.float64)
            total_sq = np.zeros(arr.shape[0], dtype=np.float64)
        total += arr.sum(axis=1)
        total_sq += np.square(arr).sum(axis=1)
        count += arr.shape[1]

    if total is None or count == 0:
        raise ValueError("no frames to fit normalization stats")

    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(var)
    return NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_norm(logmel: np.ndarray, stats: NormStats) -> np.ndarray:
    """Normalize a (1, n_mels, T) or (n_mels, T) log-mel per bin."""
    arr = np.asarray(logmel, dtype=np.float32)
    mean = stats.mean.astype(np.float32)
    std = stats.std.astype(np.float32) + _EPS
    if arr.ndim == 3:
        return (arr - mean[None, :, None]) / std[None, :, None]
    if arr.ndim == 2:
        return (arr - mean[:, None]) / std[:, None]
    raise ValueError(f"expected 2-D or 3-D log-mel, got shape {arr.shape}")
