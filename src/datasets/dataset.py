"""Torch dataset over precomputed log-mel features (spec §4-§5).

Loads a clip's ``.npy`` log-mel, normalizes with the saved train stats, and (in
train mode) applies feature-domain augmentation:

    gain (unnormalized) -> normalize -> mixup -> time-shift -> SpecAugment

Eval mode loads + normalizes only, so it is deterministic. Augmentation
randomness is seeded by ``(seed, epoch, index)`` — reproducible and stable
across DataLoader workers. Call :meth:`set_epoch` each epoch for fresh masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import augment
from datasets.manifest import ClipRecord
from datasets.taxonomy import Taxonomy
from preprocess.normalize import NormStats, apply_norm
from preprocess.paths import feature_path


@dataclass(frozen=True)
class AugmentConfig:
    # gain (spec: ±6 dB, p.5)
    gain_max_db: float = 6.0
    gain_p: float = 0.5
    # time shift (spec: ±1s -> 50 frames at hop 320 / 16 kHz, p.5)
    time_shift_max_frames: int = 50
    time_shift_p: float = 0.5
    # SpecAugment (spec: t<=64x2, f<=16x2, p.8)
    spec_n_time_masks: int = 2
    spec_time_width: int = 64
    spec_n_freq_masks: int = 2
    spec_freq_width: int = 16
    spec_p: float = 0.8
    # mixup (spec: Beta(.5,.5) label-union, p.5)
    mixup_alpha: float = 0.5
    mixup_p: float = 0.5


class LogMelDataset(Dataset):
    def __init__(
        self,
        records: list[ClipRecord],
        feature_root: str | Path,
        taxonomy: Taxonomy,
        norm_stats: NormStats,
        train: bool = True,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.feature_root = Path(feature_root)
        self.taxonomy = taxonomy
        self.norm_stats = norm_stats
        self.train = train
        self.aug = augment_cfg or AugmentConfig()
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, idx))

    def _load_normalized(self, idx: int, rng: np.random.Generator | None) -> np.ndarray:
        """Load a clip's log-mel, optionally gain-augment, then normalize."""
        logmel = np.load(feature_path(self.records[idx].clip_id, self.feature_root))
        logmel = logmel.astype(np.float32)
        if logmel.shape[-2] != self.norm_stats.n_mels:
            raise ValueError(
                f"feature has {logmel.shape[-2]} mel bins but norm stats expect "
                f"{self.norm_stats.n_mels} (clip {self.records[idx].clip_id!r})"
            )
        if self.train and rng is not None:
            # Gain is an additive log-domain shift on the UNNORMALIZED log-mel
            # (exact where mel >> 1e-6; near-silent bins deviate slightly).
            logmel = augment.apply_gain(logmel, self.aug.gain_max_db, self.aug.gain_p, rng)
        return apply_norm(logmel, self.norm_stats)

    def _label(self, idx: int) -> np.ndarray:
        return self.records[idx].multihot(self.taxonomy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.train:
            feat = self._load_normalized(idx, rng=None)
            label = self._label(idx)
            return torch.from_numpy(feat), torch.from_numpy(label)

        rng = self._rng(idx)
        feat = self._load_normalized(idx, rng)
        label = self._label(idx)

        # mixup with a random partner (label union); exclude self to avoid a no-op mix
        if rng.random() < self.aug.mixup_p and len(self.records) > 1:
            partner = int(rng.integers(len(self.records) - 1))
            if partner >= idx:
                partner += 1
            feat_b = self._load_normalized(partner, rng)
            label_b = self._label(partner)
            feat, label = augment.mixup(feat, label, feat_b, label_b, self.aug.mixup_alpha, rng)

        feat = augment.time_shift(
            feat, self.aug.time_shift_max_frames, self.aug.time_shift_p, rng
        )
        feat = augment.spec_augment(
            feat,
            self.aug.spec_n_time_masks,
            self.aug.spec_time_width,
            self.aug.spec_n_freq_masks,
            self.aug.spec_freq_width,
            self.aug.spec_p,
            rng=rng,
        )
        return torch.from_numpy(np.ascontiguousarray(feat)), torch.from_numpy(label)
