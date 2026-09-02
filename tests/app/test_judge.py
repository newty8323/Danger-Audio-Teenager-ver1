"""Stage-2 verdict parsing and the escalation transport's shutdown behaviour.

The models themselves need weights + a GPU, so what is tested here is everything around
them: never invent a degree, tolerate chatty output, and do not lose an escalation at exit.
"""
import numpy as np
import pytest

from app.judge import StubJudge, make_judge, normalize
from app.judge import _extract_json as extract_json


def test_stub_returns_no_degree():
    v = StubJudge().judge(np.zeros(16000, dtype=np.float32), "무슨 말")
    assert v["degree_percent"] is None and v["category"] is None


def test_json_is_extracted_from_chatty_output():
    raw = ('네, 분석했습니다.\n```json\n{"degree": 70, "category": "violence", '
           '"reason": "총성이 들립니다", "confident": true}\n```\n도움이 되었기를')
    v = normalize(extract_json(raw), raw)
    assert v["degree_percent"] == 70
    assert v["category"] == "violence"
    assert v["reason"] == "총성이 들립니다"
    assert v["confident"] is True
    assert v["raw"] is None                  # parsed cleanly -> the text is not kept


def test_unparseable_output_keeps_raw_and_no_degree():
    raw = "잘 모르겠습니다. 오디오가 너무 짧습니다."
    v = normalize(extract_json(raw), raw)
    assert v["degree_percent"] is None
    assert v["raw"].startswith("잘 모르겠습니다")


def test_out_of_range_degree_is_rejected_not_clamped():
    """A model that answers 900 is confused; guessing 100 would launder that away."""
    assert normalize({"degree": 900})["degree_percent"] is None
    assert normalize({"degree": -5})["degree_percent"] is None
    assert normalize({"degree": "높음"})["degree_percent"] is None


def test_unknown_category_becomes_none():
    assert normalize({"degree": 10, "category": "무서움"})["category"] is None
    assert normalize({"degree": 10, "category": "GAMBLING"})["category"] == "gambling"


def test_degree_is_rounded_to_int():
    assert normalize({"degree": 62.6})["degree_percent"] == 63


def test_make_judge_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_judge("gpt-9")


def test_make_judge_stub_needs_no_weights():
    assert make_judge("stub").name == "stub"


# ---------- transport shutdown ----------

class _SlowEscalator:
    """Exercises drain() without a network: mimics the in-flight accounting."""

    def __init__(self):
        from app.escalate import Escalator
        self.e = Escalator(server_url=None, out_dir="/tmp/_esc_test", save_audio=False)

    def __enter__(self):
        return self.e

    def __exit__(self, *exc):
        pass


def test_drain_returns_true_when_idle():
    with _SlowEscalator() as e:
        assert e.drain(1.0) is True


def test_drain_waits_for_an_inflight_post():
    with _SlowEscalator() as e:
        e._inflight = 1                        # pretend a POST is being processed
        assert e.drain(0.3) is False           # must NOT report success
        e._inflight = 0
        assert e.drain(0.3) is True


def test_post_timeout_exceeds_a_model_call():
    """A 5 s timeout discarded escalations the server had actually judged (~6 s with Omni)."""
    with _SlowEscalator() as e:
        assert e.timeout >= 30.0


def test_wav_base64_roundtrip():
    from app.escalate import decode_wav_b64, encode_wav_b64
    wav = (np.sin(np.linspace(0, 50, 16000)) * 0.5).astype(np.float32)
    got, sr = decode_wav_b64(encode_wav_b64(wav))
    assert sr == 16000 and len(got) == len(wav)
    assert np.abs(got - wav).max() < 1e-3
