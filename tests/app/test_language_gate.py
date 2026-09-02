"""Recall-first language gate: fusion, suppression, and engine wiring."""
import numpy as np

from app.engine import SR, CascadeEngine, EngineConfig
from app.language_gate import (
    KoreanLanguageGate,
    LanguageGateConfig,
    LanguageGateResult,
)
from cascade.decision import Thresholds
from tests.app.audio_fixtures import speechlike

THR = Thresholds(gate=0.3, acoustic=0.7, text=0.6)


class SequenceVAD:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, _waveform):
        return self.outputs.pop(0)


class SequenceLID:
    threshold = 0.55

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def probability(self, _waveform):
        return self.probabilities.pop(0)


class ScaleEnhancer:
    def __call__(self, waveform):
        return waveform * 0.8, {"method": "fake", "removed_rms": 0.01}


class BrokenEnhancer:
    def __call__(self, _waveform):
        raise RuntimeError("enhancer unavailable")


def _cfg(**overrides):
    values = dict(
        min_group_sec=0.5,
        max_group_sec=2.0,
        max_join_gap_sec=0.1,
        min_language_evidence_sec=0.5,
        separator_ms=200,
    )
    values.update(overrides)
    return LanguageGateConfig(**values)


def test_dual_vad_union_recovers_enhanced_only_speech():
    raw_span = [(0, SR)]
    enhanced_only_span = [(2 * SR, 3 * SR)]
    gate = KoreanLanguageGate(
        vad=SequenceVAD([raw_span, enhanced_only_span]),
        lid=SequenceLID([0.9, 0.8, 0.9, 0.8]),
        pre_vad_enhancer=ScaleEnhancer(),
        speech_enhancer=ScaleEnhancer(),
        cfg=_cfg(),
    )
    result = gate.route(np.ones(4 * SR, dtype=np.float32) * 0.1)
    assert result.raw_spans == raw_span
    assert result.enhanced_spans == enhanced_only_span
    assert result.fused_spans == raw_span + enhanced_only_span
    assert result.metadata()["selected_groups"] == 2
    assert len(result.audio) == int(2.2 * SR)


def test_uncertain_language_is_retained_for_korean_recall():
    gate = KoreanLanguageGate(
        vad=SequenceVAD([[(0, SR)], [(0, SR)]]),
        lid=SequenceLID([0.31, 0.28]),
        pre_vad_enhancer=ScaleEnhancer(),
        speech_enhancer=ScaleEnhancer(),
        cfg=_cfg(),
    )
    result = gate.route(np.ones(2 * SR, dtype=np.float32) * 0.1)
    assert result.audio is not None
    assert result.rows[0]["selected"] is True
    assert result.rows[0]["reason"] == "uncertain_recall_first"


def test_sustained_high_confidence_non_korean_is_filtered():
    spans = [(0, SR), (2 * SR, 3 * SR)]
    gate = KoreanLanguageGate(
        vad=SequenceVAD([spans, spans]),
        lid=SequenceLID([0.01, 0.02, 0.01, 0.02]),
        pre_vad_enhancer=ScaleEnhancer(),
        speech_enhancer=ScaleEnhancer(),
        cfg=_cfg(non_korean_enter_count=2),
    )
    result = gate.route(np.ones(4 * SR, dtype=np.float32) * 0.1)
    assert [row["selected"] for row in result.rows] == [True, False]
    assert result.metadata()["suppressed_groups"] == 1
    assert result.audio is not None                 # first evidence passes fail-open


def test_empty_vad_on_non_silent_audio_fails_open():
    waveform = np.ones(2 * SR, dtype=np.float32) * 0.1
    gate = KoreanLanguageGate(
        vad=SequenceVAD([[], []]),
        lid=SequenceLID([]),
        pre_vad_enhancer=ScaleEnhancer(),
        speech_enhancer=ScaleEnhancer(),
        cfg=_cfg(fail_open_on_no_speech=True),
    )
    result = gate.route(waveform)
    assert result.fail_open is True
    assert result.fail_open_reason == "vad_empty_non_silent"
    assert np.allclose(result.audio, waveform * 0.8)


def test_internal_gate_error_fails_open_with_original_audio():
    waveform = np.ones(2 * SR, dtype=np.float32) * 0.1
    gate = KoreanLanguageGate(
        vad=SequenceVAD([]),
        lid=SequenceLID([]),
        pre_vad_enhancer=BrokenEnhancer(),
        cfg=_cfg(fail_open_on_error=True),
    )
    result = gate.route(waveform)
    assert result.fail_open is True
    assert result.fail_open_reason == "RuntimeError: enhancer unavailable"
    assert np.array_equal(result.audio, waveform)


class QuietTrigger:
    def __call__(self, x, return_projection=False):
        import torch
        return {"logits": torch.full((x.shape[0], 4), -8.0)}


class RecordingASR:
    def __init__(self):
        self.seconds = []

    def transcribe(self, waveform):
        self.seconds.append(len(waveform) / SR)
        return "내일 회의는 세 시에 시작합니다"


class SafeText:
    def score(self, texts):
        return np.zeros(len(texts), dtype=np.float32)


class RoutingGate:
    def route(self, waveform):
        assert len(waveform) == 10 * SR
        return LanguageGateResult(
            audio=waveform[:2 * SR],
            rows=[{"selected": True}],
            raw_spans=[(0, 2 * SR)],
            enhanced_spans=[(0, 2 * SR)],
            fused_spans=[(0, 2 * SR)],
            enhancement={"method": "fake"},
        )


def test_engine_routes_language_gate_output_to_asr():
    cfg = EngineConfig(
        text_enabled=True,
        text_every_sec=0.0,
        language_gate_enabled=True,
        save_escalation_audio=False,
    )
    asr = RecordingASR()
    engine = CascadeEngine(
        cfg, thresholds=THR, trigger=QuietTrigger(), asr=asr, text=SafeText(),
        language_gate=RoutingGate(),
    )
    result = engine.push(speechlike(10.0))[0]
    assert asr.seconds == [2.0]
    assert result.language_gate["selected_groups"] == 1
    assert engine.stats["language_gate_runs"] == 1


class BrokenGate:
    def route(self, _waveform):
        raise RuntimeError("model unavailable")


def test_engine_gate_error_fails_open_by_default():
    cfg = EngineConfig(
        text_enabled=True,
        text_every_sec=0.0,
        language_gate_enabled=True,
        language_gate_fail_open=True,
        save_escalation_audio=False,
    )
    asr = RecordingASR()
    engine = CascadeEngine(
        cfg, thresholds=THR, trigger=QuietTrigger(), asr=asr, text=SafeText(),
        language_gate=BrokenGate(),
    )
    result = engine.push(speechlike(10.0))[0]
    assert asr.seconds == [10.0]
    assert result.language_gate["fail_open"] is True
    assert engine.stats["language_gate_fail_open"] == 1
