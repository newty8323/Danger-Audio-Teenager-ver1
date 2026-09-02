"""ASR must transcribe only audio the previous call did not cover.

Windows overlap by (window - hop), so passing the whole window every time re-transcribes
seconds already seen — the same sentence appeared twice in the live feed on 2026-07-30.
"""
import numpy as np

from app.engine import SR, CascadeEngine, EngineConfig
from cascade.decision import Thresholds
from tests.app.audio_fixtures import speechlike

THR = Thresholds(gate=0.3, acoustic=0.7, text=0.6)


class RecordingASR:
    def __init__(self):
        self.lengths = []

    def transcribe(self, wav):
        self.lengths.append(round(len(wav) / SR, 2))
        return "내일 회의는 세 시에 시작합니다 자료는 미리 공유해 주세요"


class FlatText:
    def score(self, texts):
        return np.zeros(len(texts), dtype=np.float32)


class QuietTrigger:
    def __call__(self, x, return_projection=False):
        import torch
        return {"logits": torch.full((x.shape[0], 4), -8.0)}


def _engine(asr, **kw):
    cfg = EngineConfig(text_enabled=True, save_escalation_audio=False, **kw)
    return CascadeEngine(cfg, thresholds=THR, trigger=QuietTrigger(), asr=asr, text=FlatText())


def test_first_call_transcribes_the_whole_window():
    asr = RecordingASR()
    e = _engine(asr, text_every_sec=0.0)
    e.push(speechlike(10.0))
    assert asr.lengths == [10.0]


def test_later_calls_only_get_the_new_audio():
    asr = RecordingASR()
    e = _engine(asr, text_every_sec=6.0)
    e.push(speechlike(20.0))                 # windows at t=0,2,4,6,8,10 -> ASR at 0 and 6
    assert asr.lengths[0] == 10.0
    assert len(asr.lengths) >= 2
    assert all(x <= 6.0 for x in asr.lengths[1:]), asr.lengths


def test_slice_never_shorter_than_text_min_sec():
    """Very short slices starve the ASR of context, so they are floored."""
    asr = RecordingASR()
    e = _engine(asr, text_every_sec=2.0, text_min_sec=4.0)
    e.push(speechlike(20.0))
    assert min(asr.lengths) >= 4.0, asr.lengths


def test_transcribed_seconds_are_tracked():
    asr = RecordingASR()
    e = _engine(asr, text_every_sec=6.0)
    e.push(speechlike(20.0))
    assert e.stats["asr_sec"] == sum(asr.lengths)
