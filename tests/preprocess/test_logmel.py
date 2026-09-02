import numpy as np

from preprocess.config import PreprocessConfig
from preprocess.logmel import LogMelExtractor
from preprocess.pipeline import fix_length, preprocess_clip


def test_logmel_output_shape():
    cfg = PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    wav = 0.1 * np.random.randn(cfg.clip_samples).astype(np.float32)
    logmel = extractor(wav)
    assert logmel.shape == (1, cfg.n_mels, cfg.expected_frames)
    assert logmel.dtype == np.float32
    assert np.isfinite(logmel).all()


def test_logmel_accepts_2d_input():
    cfg = PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    wav = 0.1 * np.random.randn(1, cfg.clip_samples).astype(np.float32)
    logmel = extractor(wav)
    assert logmel.shape == (1, cfg.n_mels, cfg.expected_frames)


def test_pipeline_gates_out_silence(silent_wav):
    cfg = PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    path, _ = silent_wav()
    assert preprocess_clip(str(path), extractor, cfg) is None


def test_pipeline_returns_logmel_for_audible_clip(sine_wav):
    cfg = PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    path, _ = sine_wav(freq=440.0, seconds=10.0, sr=16_000, amp=0.5)
    logmel = preprocess_clip(str(path), extractor, cfg)
    assert logmel is not None
    assert logmel.shape[0] == 1 and logmel.shape[1] == cfg.n_mels


def test_fix_length_pads_and_truncates():
    assert len(fix_length(np.ones(100), 160)) == 160  # pad
    assert len(fix_length(np.ones(200), 160)) == 160  # truncate
    assert np.array_equal(fix_length(np.ones(160), 160), np.ones(160))  # identity
    padded = fix_length(np.ones(100), 160)
    assert (padded[:100] == 1).all() and (padded[100:] == 0).all()  # zeros on the right


def test_pipeline_fixed_length_regardless_of_duration(sine_wav):
    # Short (5s) and long (12s) clips must both yield the same fixed frame count.
    cfg = PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    for seconds in (5.0, 12.0):
        path, _ = sine_wav(freq=440.0, seconds=seconds, sr=16_000, amp=0.5)
        logmel = preprocess_clip(str(path), extractor, cfg)
        assert logmel.shape == (1, cfg.n_mels, cfg.expected_frames)
