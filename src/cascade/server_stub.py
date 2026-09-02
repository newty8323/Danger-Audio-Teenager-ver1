"""Server (Qwen-Omni) escalation stub.

The real server stage judges degree(%) from the escalated clip + transcript (spec Stage-2,
not built yet). This stub freezes the REQUEST CONTRACT so the on-device side is complete:
it builds the exact payload the server will receive and appends it to a jsonl log.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from cascade.decision import ClipDecision

DEFAULT_LOG = Path(__file__).resolve().parents[2] / "data_dl/artifacts/server_escalations.jsonl"


class OmniServerStub:
    def __init__(self, log_path: str | Path = DEFAULT_LOG):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, clip_id: str, decision: ClipDecision, audio_path: str | None = None) -> dict:
        """Package an escalation. Returns the payload (server response would replace this)."""
        payload = {
            "ts": time.time(),
            "clip_id": clip_id,
            "audio_path": audio_path,          # server fetches the flagged window
            "decision": asdict(decision),
            "request": {
                "task": "harm_degree_percent",  # Omni judges 0-100% + category rationale
                "branches": list(decision.reasons),
            },
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
