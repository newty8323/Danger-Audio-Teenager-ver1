"""Hard-negative mining config (spec §7). Mirrors configs/mining/hnm.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mining" / "hnm.yaml"


@dataclass(frozen=True)
class MiningConfig:
    fp_prob_threshold: float = 0.6
    uncertain_low: float = 0.4
    uncertain_high: float = 0.6
    top_k: int = 500
    max_iterations: int = 3
    min_fpr_improvement: float = 0.005

    def __post_init__(self) -> None:
        if not (0.0 <= self.uncertain_low < self.uncertain_high <= 1.0):
            raise ValueError("require 0 <= uncertain_low < uncertain_high <= 1")
        if not (0.0 <= self.fp_prob_threshold <= 1.0):
            raise ValueError("fp_prob_threshold must be in [0, 1]")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.min_fpr_improvement < 0.0:
            raise ValueError("min_fpr_improvement must be non-negative")


def load_mining_config(path: str | Path | None = None) -> MiningConfig:
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return MiningConfig(
        fp_prob_threshold=float(raw.get("fp_prob_threshold", 0.6)),
        uncertain_low=float(raw.get("uncertain_low", 0.4)),
        uncertain_high=float(raw.get("uncertain_high", 0.6)),
        top_k=int(raw.get("top_k", 500)),
        max_iterations=int(raw.get("max_iterations", 3)),
        min_fpr_improvement=float(raw.get("min_fpr_improvement", 0.005)),
    )
