"""Cascade decision logic — pure functions, no ML imports (unit-testable).

Semantics (model_light.md §1):
  - tier-1 gate (always-on, ~0.3M): cheap suspicion score. Below threshold -> the
    acoustic trigger is never woken (duty-cycle saving). The gate NEVER escalates
    by itself; it only wakes tier 2.
  - tier-2 acoustic trigger (CED-mini int8): violence probability. At/above its
    threshold -> escalate to server.
  - text branch (ASR -> KoELECTRA int8): any-harm probability on the transcript.
    Independent duty cycle (runs on speech, not gated by the acoustic gate). At/above
    its threshold -> escalate.
  - server (Qwen-Omni) judges degree(%) — outside this module (see server_stub).

Thresholds are a versioned artifact (fit on the VAL split by
.autorun/fit_cascade_thresholds.py), never hardcoded.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Absolute so loading works regardless of CWD (scripts write it as <repo>/artifacts/...).
DEFAULT_THRESHOLDS_PATH = _REPO_ROOT / "artifacts/cascade_thresholds.json"


@dataclass(frozen=True)
class Thresholds:
    gate: float           # tier-1 wake threshold (any-vio score)
    acoustic: float       # tier-2 escalation threshold (any-vio prob)
    text: float           # text-branch escalation threshold (1 - P(safe))
    meta: dict = field(default_factory=dict)  # fit provenance (split, targets, date)


def save_thresholds(thr: Thresholds, path: str | Path = DEFAULT_THRESHOLDS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(thr), indent=1, ensure_ascii=False))


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> Thresholds:
    d = json.loads(Path(path).read_text())
    return Thresholds(gate=float(d["gate"]), acoustic=float(d["acoustic"]),
                      text=float(d["text"]), meta=d.get("meta", {}))


@dataclass(frozen=True)
class ClipDecision:
    gate_score: float | None      # None when the gate is disabled
    gate_passed: bool             # trigger was (or would be) woken
    acoustic_prob: float | None   # None when gate suppressed the trigger
    text_prob: float | None       # None when no transcript (no speech / ASR off)
    transcript: str | None
    escalate: bool
    reasons: tuple[str, ...]      # e.g. ("acoustic",), ("text",), ("acoustic", "text")


def decide(thr: Thresholds,
           gate_score: float | None,
           acoustic_prob: float | None,
           text_prob: float | None = None,
           transcript: str | None = None,
           gate_enabled: bool = True) -> ClipDecision:
    """Combine branch scores into one escalation decision.

    Callers evaluating offline may pass acoustic_prob even for gate-suppressed clips;
    it is ignored (set to None in the decision) so the decision reflects what the
    deployed cascade would actually compute.
    """
    if gate_enabled and gate_score is None:
        raise ValueError("gate_enabled=True requires a gate_score")
    gate_passed = (not gate_enabled) or gate_score >= thr.gate
    a = acoustic_prob if gate_passed else None
    reasons = []
    if a is not None and a >= thr.acoustic:
        reasons.append("acoustic")
    if text_prob is not None and text_prob >= thr.text:
        reasons.append("text")
    return ClipDecision(
        gate_score=gate_score if gate_enabled else None,
        gate_passed=gate_passed,
        acoustic_prob=a,
        text_prob=text_prob,
        transcript=transcript,
        escalate=bool(reasons),
        reasons=tuple(reasons),
    )
