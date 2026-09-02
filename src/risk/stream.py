"""Streaming risk tracker — Task C (spec §1, §8).

Consumes per-window risk scores from a 5s-stride sliding window, smooths them
with an EMA (λ=0.3), and emits a level with escalation rules:

  - base level from thresholds: safe / warn / block
  - 3 consecutive warns escalate to block (spec §8)
  - on warn, densify the stride 5s -> 2.5s (spec §8)

Stateful and decoupled from the model: it takes an already-computed risk score
so it can be unit-tested independently (spec §8).
"""

from __future__ import annotations

from dataclasses import dataclass

from risk.policy import BLOCK, WARN, RiskPolicy, risk_level


@dataclass(frozen=True)
class StreamState:
    raw: float  # this window's risk score
    smoothed: float  # EMA-smoothed risk
    level: str  # safe | warn | block (after escalation)
    base_level: str  # level from thresholds, before consecutive-warn escalation
    consecutive_warns: int
    stride_s: float  # suggested stride for the next window


class StreamRiskTracker:
    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy
        self._ema: float | None = None
        self._consecutive_warns = 0

    def reset(self) -> None:
        self._ema = None
        self._consecutive_warns = 0

    @property
    def smoothed(self) -> float | None:
        return self._ema

    def update(self, raw_score: float) -> StreamState:
        lam = self.policy.ema_lambda
        self._ema = raw_score if self._ema is None else lam * raw_score + (1.0 - lam) * self._ema

        base = risk_level(self._ema, self.policy)

        # Track consecutive warns (only an uninterrupted run of base==warn counts).
        if base == WARN:
            self._consecutive_warns += 1
        else:
            self._consecutive_warns = 0

        level = base
        if base == BLOCK or self._consecutive_warns >= self.policy.consecutive_warns_to_block:
            level = BLOCK

        stride = self.policy.stride_densified_s if level == WARN else self.policy.stride_default_s

        return StreamState(
            raw=float(raw_score),
            smoothed=float(self._ema),
            level=level,
            base_level=base,
            consecutive_warns=self._consecutive_warns,
            stride_s=stride,
        )
