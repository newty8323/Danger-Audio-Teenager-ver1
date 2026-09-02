import math

import numpy as np

from datasets import augment


def test_gain_is_constant_log_shift():
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((1, 128, 50)).astype(np.float32)
    out = augment.apply_gain(feat, max_db=6.0, p=1.0, rng=np.random.default_rng(1))
    diff = out - feat
    # A gain is a single additive constant across the whole log-mel.
    assert np.allclose(diff, diff.flat[0], atol=1e-5)
    # ...and within the ±6 dB -> ±(6*ln10/10) range.
    assert abs(float(diff.flat[0])) <= 6.0 * math.log(10) / 10 + 1e-6


def test_gain_p_zero_is_identity():
    feat = np.random.default_rng(2).standard_normal((1, 8, 8)).astype(np.float32)
    out = augment.apply_gain(feat, p=0.0, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(out, feat)


def test_time_shift_rolls():
    feat = np.arange(20, dtype=np.float32).reshape(1, 1, 20)
    out = augment.time_shift(feat, max_frames=5, p=1.0, rng=np.random.default_rng(4))
    # Output is some circular roll of the input (same multiset of values).
    assert sorted(out.flatten()) == sorted(feat.flatten())


def test_spec_augment_masks_and_preserves_shape():
    feat = np.ones((1, 64, 100), dtype=np.float32)
    out = augment.spec_augment(feat, p=1.0, rng=np.random.default_rng(5))
    assert out.shape == feat.shape
    assert (out == 0.0).any()  # something got masked
    assert not (out == 0.0).all()


def test_spec_augment_deterministic_with_seed():
    feat = np.random.default_rng(6).standard_normal((1, 64, 100)).astype(np.float32)
    a = augment.spec_augment(feat, p=1.0, rng=np.random.default_rng(7))
    b = augment.spec_augment(feat, p=1.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_spec_augment_does_not_mutate_input():
    feat = np.ones((1, 32, 40), dtype=np.float32)
    _ = augment.spec_augment(feat, p=1.0, rng=np.random.default_rng(8))
    assert (feat == 1.0).all()


def test_mixup_label_union_and_convex_feature():
    a = np.zeros((1, 4, 4), dtype=np.float32)
    b = np.ones((1, 4, 4), dtype=np.float32)
    la = np.array([1, 0, 0, 0], dtype=np.float32)
    lb = np.array([0, 1, 0, 0], dtype=np.float32)
    feat, label = augment.mixup(a, la, b, lb, alpha=0.5, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(label, [1, 1, 0, 0])  # union, not interpolation
    assert (feat >= 0.0).all() and (feat <= 1.0).all()  # convex combo of 0 and 1


# ---- waveform-domain augmentations (raw-wav BEATs fine-tune path) ----


def test_wav_gain_constant_amplitude_scale():
    wav = np.random.default_rng(10).standard_normal(1000).astype(np.float32)
    out = augment.wav_gain(wav, max_db=6.0, p=1.0, rng=np.random.default_rng(11))
    ratio = out[wav != 0] / wav[wav != 0]
    assert np.allclose(ratio, ratio[0], atol=1e-5)          # single constant scale
    assert 10 ** (-6 / 20) - 1e-4 <= abs(ratio[0]) <= 10 ** (6 / 20) + 1e-4


def test_wav_gain_p_zero_is_identity():
    wav = np.random.default_rng(12).standard_normal(100).astype(np.float32)
    out = augment.wav_gain(wav, p=0.0, rng=np.random.default_rng(13))
    np.testing.assert_array_equal(out, wav)


def test_wav_time_shift_is_circular_roll():
    wav = np.arange(16000, dtype=np.float32)
    out = augment.wav_time_shift(wav, max_sec=1.0, sample_rate=16000, p=1.0,
                                 rng=np.random.default_rng(14))
    assert out.shape == wav.shape
    assert sorted(out.tolist()) == sorted(wav.tolist())     # same multiset


def test_wav_add_noise_matches_target_snr():
    rng = np.random.default_rng(15)
    wav = rng.standard_normal(20000).astype(np.float32)
    noise = rng.standard_normal(20000).astype(np.float32)
    out = augment.wav_add_noise(wav, noise, snr_db_range=(10.0, 10.0), p=1.0,
                                rng=np.random.default_rng(16))
    residual = out - wav                                    # == scale * noise
    sig_p = float(np.mean(wav ** 2))
    noise_p = float(np.mean(residual ** 2))
    measured_snr = 10 * math.log10(sig_p / noise_p)
    assert abs(measured_snr - 10.0) < 0.5


def test_wav_add_noise_p_zero_is_identity():
    wav = np.random.default_rng(17).standard_normal(500).astype(np.float32)
    out = augment.wav_add_noise(wav, None, p=0.0, rng=np.random.default_rng(18))
    np.testing.assert_array_equal(out, wav)


def test_wav_add_noise_tiles_short_noise():
    wav = np.ones(1000, dtype=np.float32)
    short_noise = np.ones(10, dtype=np.float32)             # shorter than wav -> tiled
    out = augment.wav_add_noise(wav, short_noise, snr_db_range=(5.0, 5.0), p=1.0,
                                rng=np.random.default_rng(19))
    assert out.shape == wav.shape and np.isfinite(out).all()


def test_wav_mixup_label_union_and_convex():
    a = np.zeros(64, dtype=np.float32)
    b = np.ones(64, dtype=np.float32)
    la = np.array([1, 0, 0], dtype=np.float32)
    lb = np.array([0, 1, 0], dtype=np.float32)
    wav, label = augment.wav_mixup(a, la, b, lb, alpha=0.5, rng=np.random.default_rng(20))
    np.testing.assert_array_equal(label, [1, 1, 0])         # union
    assert (wav >= 0.0).all() and (wav <= 1.0).all()        # convex combo of 0 and 1


def test_wav_augment_deterministic_with_seed():
    wav = np.random.default_rng(21).standard_normal(4000).astype(np.float32)
    kw = dict(snr_db_range=(8.0, 8.0), p=1.0)
    a = augment.wav_add_noise(wav, None, rng=np.random.default_rng(22), **kw)
    b = augment.wav_add_noise(wav, None, rng=np.random.default_rng(22), **kw)
    np.testing.assert_array_equal(a, b)
