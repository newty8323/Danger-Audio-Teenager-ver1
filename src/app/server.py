"""Linux-side escalation receiver (Stage-2 host).

Today it accepts the frozen escalation contract, stores the payload, and returns a
placeholder verdict — the Qwen-Omni degree(%) judgment is not built yet (spec Stage-2).
Running it now means the MacBook client is exercised against a real network hop instead of
a mock, and swapping in the model later touches only `judge()`.

    uv run python -m app.server --host 0.0.0.0 --port 8770

Note the audio itself is NOT uploaded yet: the payload carries the client's local path plus
the decision. Adding the waveform is a contract change (base64 field) to make when Omni
lands, together with a decision about what may leave the device.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = _ROOT / "data_dl/app_escalations/server_received.jsonl"
MAX_BODY = 8 * 1024 * 1024


_JUDGE = None                                   # set by main(); StubJudge if not configured


def _get_judge():
    global _JUDGE
    if _JUDGE is None:
        from app.judge import StubJudge
        _JUDGE = StubJudge()
    return _JUDGE


def judge(payload: dict) -> dict:
    """Stage-2 verdict for an escalated clip (spec Task B).

    Accepts both payload shapes: an "event" (one incident, the current client behaviour) and
    a single "window" (the earlier per-window form, kept so old clients still work). The
    audio is used when the client uploaded it; otherwise the model sees only the transcript,
    which is a strictly weaker input and is reported as such.
    """
    ev, w = payload.get("event"), payload.get("window", {})
    branches = payload.get("request", {}).get("branches", [])
    scores = ({"acoustic": ev.get("peak_acoustic"), "text": ev.get("peak_text")} if ev
              else {"acoustic": w.get("acoustic"), "text": w.get("text")})
    transcript = " / ".join(ev.get("transcripts") or []) if ev else (w.get("transcript") or "")

    wav = None
    if payload.get("audio_wav_b64"):
        from app.escalate import decode_wav_b64
        wav, _ = decode_wav_b64(payload["audio_wav_b64"])

    j = _get_judge()
    verdict = j.judge(wav, transcript)
    return {"status": "accepted", "clip_id": payload.get("clip_id"),
            "model": j.name, "heard_audio": wav is not None,
            "branches": branches, "kind": "event" if ev else "window",
            "duration_sec": ev.get("duration") if ev else None,
            "device_scores": scores, **verdict}


class _Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            return self._send(413, {"error": "missing or oversized body"})
        try:
            payload = json.loads(self.rfile.read(n))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._send(400, {"error": f"bad json: {type(e).__name__}"})
        try:
            verdict = judge(payload)
        except Exception as e:                  # a broken model must not kill the receiver
            verdict = {"status": "accepted", "clip_id": payload.get("clip_id"),
                       "model": "error", "degree_percent": None,
                       "error": f"{type(e).__name__}: {e}",
                       "kind": "event" if payload.get("event") else "window",
                       "branches": payload.get("request", {}).get("branches", []),
                       "device_scores": {}, "duration_sec": None}
            print(f"[server] judge failed: {verdict['error']}", flush=True)
        # audio is logged as a path/size, never inlined: a base64 clip per line would make the
        # log unreadable and duplicate what the client already stores
        log_payload = {k: v for k, v in payload.items() if k != "audio_wav_b64"}
        if payload.get("audio_wav_b64"):
            log_payload["audio_wav_b64_bytes"] = len(payload["audio_wav_b64"])
        record = {"received_at": time.time(), "remote": self.client_address[0],
                  "payload": log_payload, "verdict": verdict}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        s = verdict.get("device_scores") or {}
        dur = f" {verdict['duration_sec']}s" if verdict.get("duration_sec") else ""
        deg = verdict.get("degree_percent")
        deg_s = f"degree={deg}%" if deg is not None else "degree=None"
        print(f"[server] {verdict.get('kind')} {payload.get('clip_id')}{dur} from "
              f"{self.client_address[0]} branches={verdict.get('branches')} "
              f"acoustic={s.get('acoustic')} text={s.get('text')} | {verdict.get('model')} "
              f"{deg_s} cat={verdict.get('category')} "
              f"heard_audio={verdict.get('heard_audio')}", flush=True)
        if verdict.get("reason"):
            print(f"           reason: {verdict['reason']}", flush=True)
        self._send(200, verdict)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        return


def main(argv=None):
    p = argparse.ArgumentParser(description="Escalation receiver (Stage-2 host)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept the MacBook client")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--judge", default="stub", choices=["stub", "qwen-omni", "qwen-audio"],
                   help="stub records only; qwen-* actually listens (needs --group server)")
    p.add_argument("--no-4bit", action="store_true", help="load the judge in bf16 (>=18GB VRAM)")
    a = p.parse_args(argv)
    if a.judge != "stub":
        from app.judge import make_judge
        global _JUDGE
        print(f"[server] loading judge '{a.judge}' (4bit={not a.no_4bit}) — "
              f"this takes a while on first run …", flush=True)
        _JUDGE = make_judge(a.judge, four_bit=not a.no_4bit)
        print(f"[server] judge ready: {_JUDGE.name}", flush=True)
    handler = type("Handler", (_Handler,), {"log_path": Path(a.log)})
    print(f"[server] listening on http://{a.host}:{a.port}  (POST /  ·  GET /health)"
          f"\n[server] judge={_get_judge().name}  logging to {a.log}", flush=True)
    ThreadingHTTPServer((a.host, a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
