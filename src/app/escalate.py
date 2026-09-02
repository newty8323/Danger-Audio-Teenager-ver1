"""Escalation transport: MacBook (on-device) -> Linux server (Stage-2 Qwen-Omni).

Contract frozen by cascade.server_stub: the server receives the flagged window's audio plus
the on-device decision, and answers with degree(%). Until that server model exists, this
client posts to a receiver that logs the request (app/server.py), and always keeps a local
jsonl + wav copy so a demo works with no network.

Uploading happens on a background thread: a slow or unreachable server must never stall
audio capture. The queue is bounded — under sustained overload we drop the OLDEST pending
escalation and count it, rather than growing memory without limit.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

SR = 16000
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _ROOT / "data_dl/app_escalations"


def write_wav(path: Path, wav: np.ndarray, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        _fill_wav(w, wav, sr)


def _fill_wav(w: wave.Wave_write, wav: np.ndarray, sr: int) -> None:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def encode_wav_b64(wav: np.ndarray, sr: int = SR) -> str:
    """16-bit mono wav as base64 — a 10 s window is ~430 KB encoded, fine for one HTTP POST."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        _fill_wav(w, wav, sr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_wav_b64(b64: str) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sr


class Escalator:
    def __init__(self, server_url: str | None = None, out_dir: str | Path | None = None,
                 # 30 s, not 5: the server runs a model before answering (the Omni judge
                 # spends ~6 s per event, more on a busy GPU), and a timeout here throws away
                 # an escalation that the server actually processed.
                 save_audio: bool = True, timeout: float = 30.0, max_pending: int = 32,
                 keep_recent: int = 50, upload_audio: bool = False):
        self.server_url = server_url
        self.dir = Path(out_dir or DEFAULT_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "escalations.jsonl"
        self.save_audio = save_audio
        # Sending the waveform is what lets the server LISTEN (Stage-2 is an audio model), and
        # it is also the moment captured audio leaves the device — so it is opt-in and logged.
        self.upload_audio = upload_audio
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._recent: list[dict] = []
        self._keep = keep_recent
        self._lock = threading.Lock()
        self.dropped = 0
        self._inflight = 0
        self.sent = 0
        self.failed = 0
        self._worker = threading.Thread(target=self._run, name="escalator", daemon=True)
        self._worker.start()

    # ---------- producer side (audio thread) ----------

    def submit_event(self, wav: np.ndarray, event) -> dict:
        """One incident -> one submission (see app.engine.HarmEvent).

        `wav` is the highest-scoring window of the event, not the whole stretch: it is the
        evidence a human or the server model needs, and it keeps the payload bounded.
        """
        ts = time.time()
        clip_id = f"event_{int(ts)}_{event.start:.0f}-{event.end:.0f}s"
        audio_path = None
        if self.save_audio and wav.size:
            audio_path = self.dir / f"{clip_id}.wav"
            write_wav(audio_path, wav)
        payload = {
            "ts": ts, "clip_id": clip_id, "kind": "event",
            "audio_path": str(audio_path) if audio_path else None,
            "event": event.to_dict(),
            "request": {"task": "harm_degree_percent", "branches": list(event.reasons)},
        }
        if self.upload_audio and wav.size:
            payload["audio_wav_b64"] = encode_wav_b64(wav)
            payload["audio_sr"] = SR
        self._record(payload)
        return payload

    def submit(self, wav: np.ndarray, result, decision) -> dict:
        ts = time.time()
        clip_id = f"live_{int(ts)}_{result.t_start:.0f}s"
        audio_path = None
        if self.save_audio:
            audio_path = self.dir / f"{clip_id}.wav"
            write_wav(audio_path, wav)
        payload = {
            "ts": ts, "clip_id": clip_id,
            "audio_path": str(audio_path) if audio_path else None,
            "window": result.to_dict(),
            "decision": asdict(decision),
            "request": {"task": "harm_degree_percent", "branches": list(decision.reasons)},
        }
        self._record(payload)
        return payload

    def _record(self, payload: dict) -> None:
        """Log locally (always), keep for the dashboard, and queue for the server (if any)."""
        with self.log_path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        keys = [k for k in ("ts", "clip_id", "kind", "window", "event", "request")
                if k in payload]
        with self._lock:
            self._recent.append({k: payload[k] for k in keys})
            del self._recent[:-self._keep]
        if self.server_url:
            self._enqueue(payload)

    def _enqueue(self, payload: dict) -> None:
        while True:
            try:
                self._q.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()          # drop oldest, keep the freshest evidence
                    self.dropped += 1
                except queue.Empty:
                    return

    # ---------- consumer side (background thread) ----------

    def _run(self) -> None:
        while True:
            payload = self._q.get()
            self._inflight += 1                 # counted separately: get() already emptied the
            try:                                # queue, so qsize() alone would look "done"
                self._post(payload)
                self.sent += 1
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.failed += 1
                print(f"[escalate] server post failed ({type(e).__name__}: {e}) — "
                      f"kept locally at {payload.get('audio_path')}")
            finally:
                self._inflight -= 1

    def _post(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.server_url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            r.read()

    def drain(self, timeout: float = 60.0) -> bool:
        """Wait for queued escalations to finish posting. Returns False if some remain.

        Needed at shutdown: the Stage-2 judge can take seconds per event, so without this the
        last event of a session was still in flight when the process exited (observed with the
        Omni judge, which spends ~6 s per verdict).
        """
        deadline = time.monotonic() + timeout
        while (self._q.qsize() or self._inflight) and time.monotonic() < deadline:
            time.sleep(0.1)
        return not (self._q.qsize() or self._inflight)

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return list(self._recent[-n:])

    def transport_stats(self) -> dict:
        return {"server_url": self.server_url, "sent": self.sent, "failed": self.failed,
                "dropped": self.dropped, "pending": self._q.qsize()}
