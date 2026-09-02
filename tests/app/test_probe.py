"""The capture probe must tell silence apart from real audio.

A silent stream makes the cascade emit one constant score forever (observed on macOS when
the tap had no permission), so this is the diagnostic that isolates capture from the models.
"""
import numpy as np

from app import sources as S


class _Fake(S.AudioSource):
    name = "fake"

    def __init__(self, frame, n=10):
        self.frame, self.n = frame, n

    def frames(self):
        for _ in range(self.n):
            yield self.frame.copy()


def _patch(monkeypatch, src):
    monkeypatch.setattr(S, "auto_source", lambda *a, **k: src)


def test_probe_reports_real_audio(monkeypatch):
    rng = np.random.default_rng(0)
    frame = (rng.standard_normal(S.SR) * 0.2).astype(np.float32)
    _patch(monkeypatch, _Fake(frame, n=3))
    r = S.probe(seconds=2.0)
    assert r["samples"] >= 2 * S.SR
    assert r["peak"] > 0.1 and r["rms"] > 0.05
    assert r["nonzero_frac"] > 0.9


def test_probe_reports_silence(monkeypatch):
    _patch(monkeypatch, _Fake(np.zeros(S.SR, dtype=np.float32), n=3))
    r = S.probe(seconds=2.0)
    assert r["peak"] == 0.0 and r["rms"] == 0.0 and r["nonzero_frac"] == 0.0


def test_probe_handles_a_backend_that_yields_nothing(monkeypatch):
    _patch(monkeypatch, _Fake(np.zeros(0, dtype=np.float32), n=2))
    r = S.probe(seconds=1.0)
    assert r["samples"] == 0 and r["rms"] == 0.0 and r["nonzero_frac"] == 0.0


def test_probe_stops_at_the_requested_duration(monkeypatch):
    frame = np.full(S.SR, 0.1, dtype=np.float32)
    _patch(monkeypatch, _Fake(frame, n=100))
    r = S.probe(seconds=3.0)
    assert 3.0 <= r["seconds"] <= 4.0        # stops one frame after reaching the target
