"""Sources (format detection, file replay) and the escalation transport."""
import json
import wave

import numpy as np
import pytest

from app.escalate import Escalator, write_wav
from app.sources import SR, AudioTeeSource, FileSource, auto_source
from cascade.decision import Thresholds, decide


def test_audiotee_detects_s16_payload():
    """audiotee emits s16 whenever it resamples (always, at 16 kHz on a Mac)."""
    pcm = (np.sin(np.linspace(0, 40, 4096)) * 20000).astype("<i2")
    assert AudioTeeSource._looks_like_f32(pcm.tobytes()) is False


def test_audiotee_detects_f32_payload():
    f = (np.sin(np.linspace(0, 40, 2048)) * 0.5).astype(np.float32)
    assert AudioTeeSource._looks_like_f32(f.tobytes()) is True


def test_audiotee_silence_is_undecidable():
    assert AudioTeeSource._looks_like_f32(np.zeros(1024, dtype=np.float32).tobytes()) is None


def test_file_source_reads_16k_mono_wav(tmp_path):
    wav = (np.sin(np.linspace(0, 100, SR)) * 0.5).astype(np.float32)
    p = tmp_path / "a.wav"
    write_wav(p, wav)
    got = np.concatenate(list(FileSource(p).frames()))
    assert len(got) == len(wav)
    assert got.dtype == np.float32
    assert np.abs(got - wav).max() < 1e-3        # 16-bit round-trip


def test_write_wav_is_16k_mono_pcm(tmp_path):
    p = tmp_path / "b.wav"
    write_wav(p, np.zeros(SR, dtype=np.float32))
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)


def test_auto_source_rejects_unknown_backend():
    with pytest.raises(ValueError):
        auto_source("nope")


class _Res:
    t_start = 12.0

    def to_dict(self):
        return {"t_start": 12.0, "acoustic": 0.91, "text": None, "level": "alert"}


def _Dec():
    """The real decision object — the transport must serialize what decide() returns."""
    return decide(Thresholds(gate=0.3, acoustic=0.7, text=0.6), None, 0.91,
                  gate_enabled=False)


def test_escalation_writes_jsonl_and_audio(tmp_path):
    es = Escalator(server_url=None, out_dir=tmp_path)
    payload = es.submit(np.zeros(SR, dtype=np.float32), _Res(), _Dec())
    rows = [json.loads(x) for x in (tmp_path / "escalations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["request"]["branches"] == ["acoustic"]
    assert (tmp_path / f"{payload['clip_id']}.wav").exists()
    assert es.recent()[-1]["clip_id"] == payload["clip_id"]


def test_escalation_without_server_does_not_queue(tmp_path):
    es = Escalator(server_url=None, out_dir=tmp_path, save_audio=False)
    es.submit(np.zeros(100, dtype=np.float32), _Res(), _Dec())
    st = es.transport_stats()
    assert st["pending"] == 0 and st["sent"] == 0 and st["dropped"] == 0


def test_escalation_queue_drops_oldest_when_full(tmp_path):
    """A dead server must not grow memory without limit."""
    es = Escalator(server_url="http://127.0.0.1:1/", out_dir=tmp_path, save_audio=False,
                   max_pending=2)
    es._worker.join(timeout=0)                  # worker may drain; only assert bounded growth
    for _ in range(20):
        es._enqueue({"x": 1})
    assert es._q.qsize() <= 2
    assert es.dropped >= 1
