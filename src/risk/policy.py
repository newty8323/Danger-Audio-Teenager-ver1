"""Risk policy: weights, thresholds, levels (spec §8).

Config-driven and independent of model code (spec §8: unit-tested, standalone).
Weights and thresholds are a versioned artifact, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from datasets.taxonomy import Taxonomy

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "risk_policy" / "default.yaml"

SAFE = "safe"
WARN = "warn"
BLOCK = "block"


@dataclass(frozen=True)
class RiskPolicy:
    version: str
    weights: dict[str, float]
    tau_warn: float = 0.4
    tau_block: float = 0.7
    ema_lambda: float = 0.3
    consecutive_warns_to_block: int = 3
    stride_default_s: float = 5.0
    stride_densified_s: float = 2.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_warn <= self.tau_block <= 1.0):
            raise ValueError("require 0 <= tau_warn <= tau_block <= 1")
        if not (0.0 < self.ema_lambda <= 1.0):
            raise ValueError("ema_lambda must be in (0, 1]")


def load_risk_policy(path: str | Path | None = None) -> RiskPolicy:
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return RiskPolicy(
        version=str(raw["version"]),
        weights={str(k): float(v) for k, v in raw["weights"].items()},
        tau_warn=float(raw.get("tau_warn", 0.4)),
        tau_block=float(raw.get("tau_block", 0.7)),
        ema_lambda=float(raw.get("ema_lambda", 0.3)),
        consecutive_warns_to_block=int(raw.get("consecutive_warns_to_block", 3)),
        stride_default_s=float(raw.get("stride_default_s", 5.0)),
        stride_densified_s=float(raw.get("stride_densified_s", 2.5)),
    )


def validate_weights(policy: RiskPolicy, taxonomy: Taxonomy) -> list[str]:
    """Return weight keys that are not harm classes in the taxonomy."""
    return [c for c in policy.weights if c not in taxonomy.harm_classes]


def weight_vector(policy: RiskPolicy, taxonomy: Taxonomy) -> np.ndarray:
    """Full (num_classes,) weight vector aligned to taxonomy order; 0 where unset.

    Confusable/safe classes get weight 0, so they never contribute to risk.
    """
    unknown = validate_weights(policy, taxonomy)
    if unknown:
        raise ValueError(f"risk weights reference non-harm classes: {unknown}")
    vec = np.zeros(taxonomy.num_classes, dtype=np.float64)
    for cls, w in policy.weights.items():
        vec[taxonomy.index_of(cls)] = w
    return vec


def risk_level(score: float, policy: RiskPolicy) -> str:
    """Single-clip level from a risk score (no stream state)."""
    if score >= policy.tau_block:
        return BLOCK
    if score >= policy.tau_warn:
        return WARN
    return SAFE
