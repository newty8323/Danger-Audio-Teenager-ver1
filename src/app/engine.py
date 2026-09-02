"""Streaming cascade engine — the on-device half of the system (model_light §1).

Pipeline per window (10 s, hop 2 s — the window matches the training clip length, so the
thresholds in artifacts/cascade_thresholds.json apply unchanged):

  playback audio ─► acoustic branch: CED-mini ──────────────────────────────┐
                 └► original HTDemucs ─► silence compaction                │
                                         └► Whisper Base ─► KoELECTRA ─────┤
                                                                           ├─► decide()
                                                                           └─► Qwen server

No tier-1 CNN gate: measured on 2026-07-30 it costs more CPU than it saves (model_light §2-2),
and playback capture is already gated for free by the OS playback state (§0).

The text branch runs on its own duty cycle: ASR is the expensive part, so it only runs when a
window has speech-like energy AND at most every `text_every_sec` seconds.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.escalate import Escalator
from app.vad import is_degenerate, speech_score
from cascade.decision import ClipDecision, Thresholds, decide, load_thresholds

SR = 16000
WINDOW_SEC = 10.0
HOP_SEC = 2.0


@dataclass
class WindowResult:
    t_start: float                  # seconds since capture start
    acoustic: float | None
    text: float | None
    transcript: str | None
    language_gate: dict | None
    escalate: bool
    reasons: tuple[str, ...]
    risk: float                     # smoothed 0..1 display value
    level: str                      # "ok" | "watch" | "alert"
    rms: float
    latency_ms: float

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reasons"] = list(self.reasons)
        return d


@dataclass
class HarmEvent:
    """One suspicious stretch of audio, not one window.

    Windows overlap (10 s / 2 s hop), so a single scene escalated 3-5 times and the server
    received the same incident repeatedly (observed live 2026-07-30). Consecutive escalating
    windows are merged into an event and submitted ONCE, when the event closes.
    """
    start: float
    end: float
    windows: int = 0
    peak_acoustic: float = 0.0
    peak_text: float = 0.0
    reasons: tuple[str, ...] = ()
    transcripts: tuple[str, ...] = ()
    peak_score: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["reasons"] = list(self.reasons)
        d["transcripts"] = list(self.transcripts)
        d["duration"] = self.duration
        return d


@dataclass
class EngineConfig:
    thresholds_path: str | Path | None = None
    window_sec: float = WINDOW_SEC
    hop_sec: float = HOP_SEC
    text_enabled: bool = True
    text_every_sec: float = 4.0     # ASR duty cycle; short enough to avoid merged utterances
    speech_rms_min: float = 0.01    # skip ASR on near-silence
    # DISABLED (0.0) after measurement on real audio: app/vad.speech_score scores real movie
    # dialogue at ~0.00 and gunshots up to 0.96 — it does not separate speech from non-speech
    # outside the synthetic fixture it was tuned on, and it silently suppressed ASR on a
    # profanity-heavy clip (2026-07-30). Hallucinations are handled after ASR by
    # is_degenerate(), which was validated on real output. Raise this only with a real VAD.
    speech_min: float = 0.0
    text_min_sec: float = 4.0       # shortest slice worth transcribing (ASR needs context)
    asr_model_id: str | None = None  # None -> cascade.pipeline default (moonshine-tiny-ko)
    ver1_demucs_packages: str | Path | None = None
    ver1_demucs_repo: str | Path | None = None
    ver1_demucs_model_name: str = "htdemucs"
    ver1_whisper_cli: str | Path | None = None
    ver1_whisper_model: str | Path | None = None
    ver1_silence_threshold_db: float = -50.0
    ver1_whisper_language: str = "ko"
    ver1_demucs_device: str = "auto"
    uvr_onnx_model: str | Path | None = None
    text_lexicon: bool = True        # add the slang lexicon to the classifier score
    # Recall-first Korean routing before ASR. The acoustic branch always receives raw audio.
    language_gate_enabled: bool = False
    language_gate_vad_path: str | Path | None = None
    language_gate_checkpoint_path: str | Path | None = None
    language_gate_deepfilter_exe: str | Path | None = None
    language_gate_device: str = "auto"
    language_gate_vad_threshold: float = 0.10
    language_gate_fail_open: bool = True
    ema_alpha: float = 0.4          # risk smoothing
    watch_frac: float = 0.6         # "watch" = acoustic >= watch_frac * threshold
    server_url: str | None = None   # Linux server endpoint; None -> local jsonl only
    # Quiet time needed to close a harm event. None -> derived, because the TEXT branch can
    # only escalate on windows where ASR ran, i.e. at most every `text_every_sec`: a gap
    # threshold at or below the duty cycle would split every talk-driven incident into one
    # event per ASR call (exactly what the live 6s/12s/18s escalations did).
    event_gap_sec: float | None = None
    keep_results: int = 600
    save_escalation_audio: bool = True
    upload_audio: bool = False       # send the waveform to the server (it leaves the device)


class CascadeEngine:
    """Feeds frames in, emits one WindowResult per hop. Thread-safe for a reader UI."""

    def __init__(self, cfg: EngineConfig | None = None, thresholds: Thresholds | None = None,
                 trigger=None, asr=None, text=None, language_gate=None):
        self.cfg = cfg or EngineConfig()
        if thresholds is not None:
            self.thr = thresholds
        elif self.cfg.thresholds_path:
            self.thr = load_thresholds(self.cfg.thresholds_path)
        else:
            self.thr = load_thresholds()         # versioned artifact, repo-root relative
        self.trigger, self.asr, self.text = trigger, asr, text
        self.language_gate = language_gate
        if self.cfg.event_gap_sec is None:
            duty = self.cfg.text_every_sec if self.cfg.text_enabled else 0.0
            self.cfg.event_gap_sec = max(2 * self.cfg.hop_sec, duty + 2 * self.cfg.hop_sec)
        self.escalator = Escalator(server_url=self.cfg.server_url,
                                   save_audio=self.cfg.save_escalation_audio,
                                   upload_audio=self.cfg.upload_audio)
        self._buf = np.zeros(0, dtype=np.float32)
        self._consumed = 0                      # samples dropped from the front of the stream
        self._samples_seen = 0
        self._next_window_at = 0                 # absolute sample index of the next window start
        self._risk = 0.0
        self._last_text_at = -1e9
        self._lock = threading.Lock()
        self.results: deque[WindowResult] = deque(maxlen=self.cfg.keep_results)
        self._event: HarmEvent | None = None      # open event being accumulated
        self._event_wav: np.ndarray | None = None  # audio of its highest-scoring window only
        self._last_escalation_at = -1e9
        self.events: deque[HarmEvent] = deque(maxlen=100)
        self.stats = {"windows": 0, "escalations": 0, "events": 0, "asr_runs": 0,
                      "asr_skipped_nonspeech": 0, "asr_degenerate": 0, "started": time.time()}
        self.stats.update({"language_gate_runs": 0, "language_gate_selected_groups": 0,
                           "language_gate_suppressed_groups": 0,
                           "language_gate_fail_open": 0,
                           "asr_skipped_language": 0})

    # ---------- model wiring ----------

    def load_models(self, int8: bool = True, asr_device: str | None = None) -> None:
        """Load whatever is not already injected (tests inject fakes instead)."""
        from cascade.pipeline import (
            ASR_DEFAULT,
            HybridTextScorer,
            TextScorer,
            VER1_DEMUCS_WHISPER_BASE,
            VER1_UVR_ONNX_WHISPER_BASE,
            load_trigger,
            make_asr,
        )
        if self.trigger is None:
            self.trigger = load_trigger(int8=int8)
        if self.cfg.text_enabled:
            if self.text is None:
                # hybrid = classifier OR lexicon; the lexicon is what catches Korean slang
                # (slang recall .68 -> .91 on perfect transcripts) at no model-size cost
                self.text = (HybridTextScorer(int8=int8) if self.cfg.text_lexicon
                             else TextScorer(int8=int8))
            if self.asr is None:
                model_id = self.cfg.asr_model_id or ASR_DEFAULT
                options = {}
                if model_id == VER1_DEMUCS_WHISPER_BASE:
                    options = {
                        "silence_threshold_db": self.cfg.ver1_silence_threshold_db,
                        "language": self.cfg.ver1_whisper_language,
                        "demucs_device": self.cfg.ver1_demucs_device,
                        "demucs_model_name": self.cfg.ver1_demucs_model_name,
                    }
                    optional_paths = {
                        "demucs_packages": self.cfg.ver1_demucs_packages,
                        "demucs_repo": self.cfg.ver1_demucs_repo,
                        "whisper_cli": self.cfg.ver1_whisper_cli,
                        "whisper_model": self.cfg.ver1_whisper_model,
                    }
                    options.update({key: value for key, value in optional_paths.items() if value})
                elif model_id == VER1_UVR_ONNX_WHISPER_BASE:
                    options = {
                        "silence_threshold_db": self.cfg.ver1_silence_threshold_db,
                        "language": self.cfg.ver1_whisper_language,
                    }
                    optional_paths = {
                        "uvr_model": self.cfg.uvr_onnx_model,
                        "whisper_cli": self.cfg.ver1_whisper_cli,
                        "whisper_model": self.cfg.ver1_whisper_model,
                    }
                    options.update({key: value for key, value in optional_paths.items() if value})
                self.asr = make_asr(model_id, asr_device, **options)
                self.stats["asr_backend"] = getattr(self.asr, "model_id", model_id)
            if self.cfg.language_gate_enabled and self.language_gate is None:
                from app.language_gate import KoreanLanguageGate, LanguageGateConfig

                if not (self.cfg.language_gate_vad_path
                        and self.cfg.language_gate_checkpoint_path):
                    raise ValueError(
                        "language gate needs a Silero VAD model and Whisper LID checkpoint"
                    )
                gate_cfg = LanguageGateConfig(
                    vad_threshold=self.cfg.language_gate_vad_threshold,
                    fail_open_on_error=self.cfg.language_gate_fail_open,
                )
                self.language_gate = KoreanLanguageGate.from_paths(
                    vad_path=self.cfg.language_gate_vad_path,
                    checkpoint_path=self.cfg.language_gate_checkpoint_path,
                    deepfilter_exe=self.cfg.language_gate_deepfilter_exe,
                    device=self.cfg.language_gate_device,
                    cfg=gate_cfg,
                )

    # ---------- streaming ----------

    @property
    def window_n(self) -> int:
        return int(self.cfg.window_sec * SR)

    @property
    def hop_n(self) -> int:
        return int(self.cfg.hop_sec * SR)

    def push(self, frame: np.ndarray) -> list[WindowResult]:
        """Append audio; return results for every complete window that became available."""
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32)])
        self._samples_seen += len(frame)
        out = []
        while self._samples_seen - self._next_window_at >= self.window_n:
            start_abs = self._next_window_at
            lo = start_abs - self._consumed
            win = self._buf[lo:lo + self.window_n]
            out.append(self._process(win, start_abs / SR))
            self._next_window_at += self.hop_n
            # drop audio no longer reachable by any future window (bounded memory)
            keep_from = self._next_window_at - self._consumed
            if keep_from > 0:
                self._buf = self._buf[keep_from:].copy()
                self._consumed += keep_from
        return out

    def _process(self, win: np.ndarray, t_start: float) -> WindowResult:
        t0 = time.time()
        rms = float(np.sqrt(np.mean(win ** 2)) + 1e-12)
        a = self._acoustic(win)
        t_prob, transcript, language_gate = self._maybe_text(win, t_start, rms)
        dec = decide(self.thr, None, a, t_prob, transcript, gate_enabled=False)
        base = a if a is not None else 0.0
        if t_prob is not None:
            base = max(base, t_prob)
        with self._lock:
            self._risk = (1 - self.cfg.ema_alpha) * self._risk + self.cfg.ema_alpha * base
            risk = self._risk
        res = WindowResult(t_start=round(t_start, 2), acoustic=a, text=t_prob,
                           transcript=transcript, language_gate=language_gate,
                           escalate=dec.escalate, reasons=dec.reasons,
                           risk=round(risk, 4), level=self._level(dec, a), rms=round(rms, 5),
                           latency_ms=round((time.time() - t0) * 1000, 1))
        with self._lock:
            self.results.append(res)
            self.stats["windows"] += 1
            if dec.escalate:
                self.stats["escalations"] += 1
        self._update_event(win, res, base)
        return res

    # ---------- event aggregation (one submission per incident, not per window) ----------

    def _update_event(self, win: np.ndarray, res: WindowResult, base: float) -> None:
        end = res.t_start + self.cfg.window_sec
        if res.escalate:
            with self._lock:
                if self._event is None:
                    self._event = HarmEvent(start=res.t_start, end=end)
                ev = self._event
                ev.end = max(ev.end, end)
                ev.windows += 1
                ev.peak_acoustic = max(ev.peak_acoustic, res.acoustic or 0.0)
                ev.peak_text = max(ev.peak_text, res.text or 0.0)
                ev.reasons = tuple(dict.fromkeys(ev.reasons + res.reasons))
                if res.transcript and res.transcript.strip():
                    ev.transcripts = tuple(dict.fromkeys(
                        ev.transcripts + (res.transcript.strip(),)))[-5:]
                new_peak = base > ev.peak_score
                if new_peak:
                    ev.peak_score = round(base, 4)
                self._last_escalation_at = res.t_start
            if new_peak:                       # keep only the loudest evidence: one window
                self._event_wav = win.copy()
        elif self._event is not None and \
                res.t_start - self._last_escalation_at >= self.cfg.event_gap_sec:
            self._close_event()

    def _close_event(self) -> None:
        with self._lock:
            ev, wav = self._event, self._event_wav
            self._event, self._event_wav = None, None
            if ev is not None:
                self.events.append(ev)
                self.stats["events"] += 1
        if ev is None:
            return
        try:
            self.escalator.submit_event(wav if wav is not None else np.zeros(0, np.float32), ev)
        except Exception as e:                 # never let escalation kill the capture loop
            print(f"[engine] event escalation failed: {type(e).__name__}: {e}")

    def flush(self) -> None:
        """Close an event still open at end of stream so it is not silently lost."""
        if self._event is not None:
            self._close_event()

    def _acoustic(self, win: np.ndarray) -> float | None:
        if self.trigger is None:
            return None
        import torch
        with torch.no_grad():
            x = torch.from_numpy(win).unsqueeze(0)
            logits = self.trigger(x, return_projection=False)["logits"]
            return float(torch.sigmoid(logits).max())

    def _maybe_text(self, win: np.ndarray, t_start: float, rms: float):
        """Return ``(text_prob, transcript, language_gate_metadata)``.

        ``text_prob`` is None whenever the window is not trustworthy speech: silence,
        non-speech audio, a rejected language group, or an ASR hallucination loop.
        """
        if not (self.cfg.text_enabled and self.asr and self.text):
            return None, None, None
        if rms < self.cfg.speech_rms_min:
            return None, None, None
        if t_start - self._last_text_at < self.cfg.text_every_sec:
            return None, None, None
        sp = speech_score(win)
        if sp < self.cfg.speech_min:                 # loud is not the same as speech
            with self._lock:
                self.stats["asr_skipped_nonspeech"] += 1
            return None, None, None
        # Transcribe only audio the previous ASR call did not cover. Windows overlap by
        # (window - hop), so passing the whole window re-transcribes seconds already seen and
        # the same sentence shows up twice in the feed (observed live on 2026-07-30).
        seen_until = self._last_text_at + self.cfg.window_sec
        new_sec = max(self.cfg.text_min_sec,
                      min(self.cfg.window_sec, t_start + self.cfg.window_sec - seen_until))
        seg = win if new_sec >= self.cfg.window_sec else win[-int(new_sec * SR):]
        self._last_text_at = t_start

        gate_meta = None
        if self.cfg.language_gate_enabled:
            if self.language_gate is None:
                raise RuntimeError("language gate is enabled but was not loaded")
            try:
                gate_result = self.language_gate.route(seg)
                gate_meta = gate_result.metadata()
                with self._lock:
                    self.stats["language_gate_runs"] += 1
                    self.stats["language_gate_selected_groups"] += gate_meta["selected_groups"]
                    self.stats["language_gate_suppressed_groups"] += gate_meta["suppressed_groups"]
                    if gate_meta["fail_open"]:
                        self.stats["language_gate_fail_open"] += 1
                if gate_result.audio is None or len(gate_result.audio) < SR // 10:
                    with self._lock:
                        self.stats["asr_skipped_language"] += 1
                    return None, None, gate_meta
                seg = gate_result.audio
            except Exception as exc:
                if not self.cfg.language_gate_fail_open:
                    raise
                gate_meta = {
                    "fail_open": True,
                    "fail_open_reason": f"{type(exc).__name__}: {exc}",
                    "selected_groups": 0,
                    "suppressed_groups": 0,
                }
                with self._lock:
                    self.stats["language_gate_runs"] += 1
                    self.stats["language_gate_fail_open"] += 1

        with self._lock:
            self.stats["asr_runs"] += 1
            self.stats["asr_sec"] = round(self.stats.get("asr_sec", 0.0) + len(seg) / SR, 1)
        transcript = self.asr.transcribe(seg)
        asr_metrics = getattr(self.asr, "last_metrics", None)
        if asr_metrics:
            with self._lock:
                self.stats["demucs_runs"] = self.stats.get("demucs_runs", 0) + 1
                self.stats["demucs_seconds"] = round(
                    self.stats.get("demucs_seconds", 0.0)
                    + float(asr_metrics.get("demucs_seconds", 0.0)), 3
                )
                self.stats["whisper_seconds"] = round(
                    self.stats.get("whisper_seconds", 0.0)
                    + float(asr_metrics.get("whisper_seconds", 0.0)), 3
                )
                if asr_metrics.get("context_recovered"):
                    self.stats["asr_context_recoveries"] = (
                        self.stats.get("asr_context_recoveries", 0) + 1
                    )
        if not transcript.strip():
            return None, transcript, gate_meta
        if is_degenerate(transcript):
            # Moonshine loops on non-speech; scoring the loop caused a false escalation in
            # the 2026-07-30 live run. Keep the text for display, discard the score.
            with self._lock:
                self.stats["asr_degenerate"] += 1
            return None, transcript, gate_meta
        return float(self.text.score([transcript])[0]), transcript, gate_meta

    def _level(self, dec: ClipDecision, a: float | None) -> str:
        if dec.escalate:
            return "alert"
        if a is not None and a >= self.cfg.watch_frac * self.thr.acoustic:
            return "watch"
        return "ok"

    # ---------- readers ----------

    def snapshot(self, last: int = 120) -> dict:
        with self._lock:
            rs = list(self.results)[-last:]
            stats = dict(self.stats)
            risk = self._risk
        with self._lock:
            evs = [e.to_dict() for e in list(self.events)[-12:]]
            open_ev = self._event.to_dict() if self._event else None
        return {"risk": round(risk, 4), "stats": stats,
                "thresholds": {"acoustic": self.thr.acoustic, "text": self.thr.text},
                "results": [r.to_dict() for r in rs],
                "events": evs, "open_event": open_ev,
                "escalations": self.escalator.recent(),
                "transport": self.escalator.transport_stats()}


def run_source(engine: CascadeEngine, source, on_result=None, stop: threading.Event | None = None):
    """Drive an engine from an AudioSource until it ends or `stop` is set."""
    with source:
        for frame in source.frames():
            for res in engine.push(frame):
                if on_result:
                    on_result(res)
            if stop is not None and stop.is_set():
                return
