"""Streaming engine: windowing, memory bound, duty cycle, escalation — with fake models."""
import numpy as np
import pytest

from app.engine import SR, CascadeEngine, EngineConfig
from cascade.decision import Thresholds
from tests.app.audio_fixtures import noise, speechlike

THR = Thresholds(gate=0.3, acoustic=0.7, text=0.6)


class FakeTrigger:
    """Returns a preset probability per call; mimics HarmModel's dict output."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def __call__(self, x, return_projection=False):
        import torch
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        logit = float(np.log(p / (1 - p)))
        return {"logits": torch.full((x.shape[0], 4), -20.0).index_put_(
            (torch.tensor([0]), torch.tensor([0])), torch.tensor(logit))}


class FakeASR:
    def __init__(self, text="칼 들고 찾아갈 거야"):
        self.text = text
        self.calls = 0

    def transcribe(self, wav):
        self.calls += 1
        return self.text


class FakeText:
    def __init__(self, score=0.9):
        self.s = score
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        return np.full(len(texts), self.s, dtype=np.float32)


def _engine(cfg=None, **kw):
    cfg = cfg or EngineConfig(text_enabled=False, save_escalation_audio=False)
    return CascadeEngine(cfg, thresholds=THR, **kw)


def _noise(sec, amp=0.2, seed=0):
    return noise(sec, amp, seed)


def test_no_window_before_full_10s():
    e = _engine(trigger=FakeTrigger([0.1]))
    assert e.push(_noise(9.5)) == []
    assert e.stats["windows"] == 0


def test_windows_emitted_on_hop():
    e = _engine(trigger=FakeTrigger([0.1]))
    out = e.push(_noise(10.0))
    assert len(out) == 1 and out[0].t_start == 0.0
    out = e.push(_noise(2.0))                  # one hop later
    assert len(out) == 1 and out[0].t_start == pytest.approx(2.0)


def test_several_windows_in_one_big_push():
    e = _engine(trigger=FakeTrigger([0.1]))
    out = e.push(_noise(16.0))                 # 10s + 3 hops
    assert [r.t_start for r in out] == [0.0, 2.0, 4.0, 6.0]


def test_buffer_stays_bounded_over_long_stream():
    """The O(N) memory bug class: the ring must not grow with stream length."""
    e = _engine(trigger=FakeTrigger([0.1]))
    for _ in range(60):                        # 120 s of audio
        e.push(_noise(2.0, seed=1))
    assert len(e._buf) <= e.window_n + 2 * e.hop_n
    assert e.stats["windows"] >= 55


def test_acoustic_escalation_and_reasons():
    e = _engine(trigger=FakeTrigger([0.95]))
    r = e.push(_noise(10.0))[0]
    assert r.acoustic == pytest.approx(0.95, abs=1e-3)
    assert r.escalate and r.reasons == ("acoustic",) and r.level == "alert"


def test_below_threshold_is_watch_or_ok():
    watch = _engine(trigger=FakeTrigger([0.5])).push(_noise(10.0))[0]   # >= 0.6*0.7
    ok = _engine(trigger=FakeTrigger([0.05])).push(_noise(10.0))[0]
    assert watch.level == "watch" and not watch.escalate
    assert ok.level == "ok"


def test_text_branch_runs_and_escalates():
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, save_escalation_audio=False)
    asr, txt = FakeASR(), FakeText(0.9)
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr, text=txt)
    r = e.push(speechlike(10.0))[0]
    assert asr.calls == 1 and txt.calls == 1
    assert r.text == pytest.approx(0.9) and r.reasons == ("text",) and r.escalate


def test_spectral_gate_is_off_by_default():
    """It scored real movie dialogue at 0.00 and gunshots at 0.96 — it must not gate ASR.

    Loud non-speech therefore DOES reach ASR now; the hallucination it can produce is caught
    afterwards by is_degenerate (see test_degenerate_transcript_is_not_scored).
    """
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, save_escalation_audio=False)
    assert cfg.speech_min == 0.0
    asr = FakeASR()
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr,
                      text=FakeText(0.99))
    e.push(_noise(10.0, amp=0.5))
    assert asr.calls == 1
    assert e.stats["asr_skipped_nonspeech"] == 0


def test_explicit_speech_gate_still_works_when_enabled():
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, speech_min=0.9,
                       save_escalation_audio=False)
    asr = FakeASR()
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr,
                      text=FakeText(0.99))
    r = e.push(_noise(10.0, amp=0.5))[0]
    assert asr.calls == 0 and r.text is None
    assert e.stats["asr_skipped_nonspeech"] == 1


def test_degenerate_transcript_is_not_scored():
    """Even if ASR runs, a hallucination loop must not produce a text score."""
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, save_escalation_audio=False)
    asr, txt = FakeASR("와! " * 34), FakeText(0.95)
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr, text=txt)
    r = e.push(speechlike(10.0))[0]
    assert asr.calls == 1 and txt.calls == 0          # scorer never consulted
    assert r.text is None and not r.escalate
    assert r.transcript.startswith("와")              # kept for display
    assert e.stats["asr_degenerate"] == 1


def test_text_skipped_on_silence():
    cfg = EngineConfig(text_enabled=True, text_every_sec=0.0, save_escalation_audio=False)
    asr = FakeASR()
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr,
                      text=FakeText(0.9))
    r = e.push(np.zeros(10 * SR, dtype=np.float32))[0]
    assert asr.calls == 0 and r.text is None and not r.escalate


def test_text_duty_cycle_limits_asr_runs():
    cfg = EngineConfig(text_enabled=True, text_every_sec=6.0, save_escalation_audio=False)
    asr = FakeASR()
    e = CascadeEngine(cfg, thresholds=THR, trigger=FakeTrigger([0.01]), asr=asr,
                      text=FakeText(0.1))
    e.push(speechlike(20.0))                   # 6 windows (t=0..10) but ASR only every 6 s
    assert 1 <= asr.calls <= 3, f"duty cycle not honored: {asr.calls} ASR runs"


def test_snapshot_shape():
    e = _engine(trigger=FakeTrigger([0.9]))
    e.push(_noise(10.0))
    s = e.snapshot()
    assert set(s) >= {"risk", "stats", "thresholds", "results", "escalations", "transport"}
    assert s["results"][-1]["level"] == "alert"
    assert isinstance(s["results"][-1]["reasons"], list)


def test_risk_is_smoothed_not_instantaneous():
    e = _engine(trigger=FakeTrigger([0.9, 0.9]))
    r1 = e.push(_noise(10.0))[0]
    r2 = e.push(_noise(2.0))[0]
    assert r1.risk < 0.9              # EMA has not caught up yet
    assert r2.risk > r1.risk          # but it is rising
