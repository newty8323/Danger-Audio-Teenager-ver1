"""Event aggregation: one incident must reach the server once, not once per window.

Windows overlap (10 s / 2 s hop) so a single scene escalated 3-5 times in the live run and the
server logged the same incident repeatedly.
"""
import numpy as np
import pytest

from app.engine import SR, CascadeEngine, EngineConfig
from cascade.decision import Thresholds
from tests.app.audio_fixtures import noise, speechlike

THR = Thresholds(gate=0.3, acoustic=0.7, text=0.6)


class ScriptedTrigger:
    """Emits a preset acoustic probability per call (cycling the last value)."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def __call__(self, x, return_projection=False):
        import torch
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        logit = float(np.log(p / (1 - p)))
        out = torch.full((x.shape[0], 4), -20.0)
        out[0, 0] = logit
        return {"logits": out}


class Recorder:
    def __init__(self):
        self.events = []

    def submit_event(self, wav, event):
        self.events.append((event, len(wav)))
        return {}

    def recent(self, n=20):
        return []

    def transport_stats(self):
        return {}


def _engine(probs, **kw):
    cfg = EngineConfig(text_enabled=False, save_escalation_audio=False, **kw)
    e = CascadeEngine(cfg, thresholds=THR, trigger=ScriptedTrigger(probs))
    e.escalator = Recorder()
    return e


def _push(e, n_windows, amp=0.2):
    """Feed enough audio for `n_windows` windows (10 s + (n-1) hops)."""
    e.push(noise(10.0, amp=amp))
    for _ in range(n_windows - 1):
        e.push(noise(e.cfg.hop_sec, amp=amp, seed=1))


def test_consecutive_escalations_become_one_event():
    e = _engine([0.95] * 4)
    _push(e, 4)
    e.flush()
    assert e.stats["escalations"] == 4
    assert len(e.escalator.events) == 1, "one incident must be submitted once"
    ev, _ = e.escalator.events[0]
    assert ev.windows == 4
    assert ev.start == 0.0 and ev.end == pytest.approx(16.0)   # 3 hops later + window
    assert ev.duration == pytest.approx(16.0)


def test_gap_closes_the_event_and_a_later_one_opens_another():
    # escalate, then 4 quiet windows (8 s > event_gap 6 s), then escalate again
    e = _engine([0.95] + [0.01] * 4 + [0.95] * 2, event_gap_sec=6.0)
    _push(e, 7)
    e.flush()
    assert len(e.escalator.events) == 2
    assert e.stats["events"] == 2


def test_default_gap_exceeds_the_asr_duty_cycle():
    """The text branch can only fire every `text_every_sec`; the gap must be larger or every
    talk-driven incident splits into one event per ASR call (the live 6/12/18s case)."""
    cfg = EngineConfig(text_enabled=True, text_every_sec=6.0, hop_sec=2.0)
    e = CascadeEngine(cfg, thresholds=THR, trigger=ScriptedTrigger([0.01]))
    assert e.cfg.event_gap_sec > cfg.text_every_sec


def test_escalations_one_duty_cycle_apart_are_one_event():
    """Reproduces the live pattern: text escalations at t=0, 6, 12 -> ONE event."""
    probs = [0.01] * 9
    e = _engine(probs)                        # acoustic silent; text drives escalation
    e.cfg.text_enabled = True
    e.cfg.text_every_sec = 6.0
    e.cfg.event_gap_sec = 10.0
    e.asr, e.text = _SeqASR(), _HighText()
    e.push(speechlike(10.0))
    for i in range(8):                        # windows at t=2..16
        e.push(speechlike(e.cfg.hop_sec, seed=i + 2))
    e.flush()
    assert len(e.escalator.events) == 1, [ev.to_dict() for ev, _ in e.escalator.events]


def test_event_keeps_peak_scores_and_reasons():
    e = _engine([0.75, 0.99, 0.80])
    _push(e, 3)
    e.flush()
    ev, _ = e.escalator.events[0]
    assert ev.peak_acoustic == pytest.approx(0.99, abs=1e-2)
    assert ev.reasons == ("acoustic",)
    assert ev.peak_score == pytest.approx(0.99, abs=1e-2)


def test_open_event_is_flushed_at_end_of_stream():
    e = _engine([0.95])
    _push(e, 1)
    assert len(e.escalator.events) == 0      # still open — audio might continue
    e.flush()
    assert len(e.escalator.events) == 1


def test_no_escalation_no_event():
    e = _engine([0.01] * 3)
    _push(e, 3)
    e.flush()
    assert e.escalator.events == [] and e.stats["events"] == 0


def test_only_the_peak_window_audio_is_kept():
    """Memory: an event holds one window of audio regardless of how long it runs."""
    e = _engine([0.95] * 30)
    _push(e, 30)
    e.flush()
    _, n = e.escalator.events[0]
    assert n == int(e.cfg.window_sec * SR)


def test_transcripts_are_collected_and_capped():
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, save_escalation_audio=False)
    e = CascadeEngine(cfg, thresholds=THR, trigger=ScriptedTrigger([0.01] * 9),
                      asr=_SeqASR(), text=_HighText())
    e.escalator = Recorder()
    e.push(speechlike(10.0))
    for i in range(8):
        e.push(speechlike(cfg.hop_sec, seed=i + 2))
    e.flush()
    ev, _ = e.escalator.events[0]
    assert 1 <= len(ev.transcripts) <= 5      # deduped and capped
    assert ev.reasons == ("text",)


class _SeqASR:
    def __init__(self):
        self.i = 0

    def transcribe(self, wav):
        self.i += 1
        return f"칼 들고 찾아갈 거야 {self.i}"


class _HighText:
    def score(self, texts):
        return np.full(len(texts), 0.95, dtype=np.float32)


def test_snapshot_exposes_events():
    e = _engine([0.95] * 2)
    _push(e, 2)
    s = e.snapshot()
    assert s["open_event"] is not None and s["open_event"]["windows"] == 2
    e.flush()
    s = e.snapshot()
    assert s["open_event"] is None and len(s["events"]) == 1
