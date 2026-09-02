import numpy as np
import pytest

from preprocess.normalize import NormStats, apply_norm, fit_norm_stats


def test_fit_norm_stats_recovers_mean_std():
    rng = np.random.default_rng(42)
    n_mels = 128
    true_mean = rng.uniform(-5, 5, size=n_mels)
    true_std = rng.uniform(0.5, 3.0, size=n_mels)
    # Many frames so empirical stats converge
    frames = true_mean[:, None] + true_std[:, None] * rng.standard_normal((n_mels, 50_000))
    stats = fit_norm_stats([frames[None, :, :]])
    assert stats.mean.shape == (n_mels,)
    np.testing.assert_allclose(stats.mean, true_mean, atol=0.1)
    np.testing.assert_allclose(stats.std, true_std, rtol=0.05)


def test_fit_across_multiple_clips_matches_concatenation():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal((1, 8, 100)) for _ in range(5)]
    streamed = fit_norm_stats(clips)
    concat = np.concatenate([c[0] for c in clips], axis=1)  # (8, 500)
    np.testing.assert_allclose(streamed.mean, concat.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(streamed.std, concat.std(axis=1), atol=1e-6)


def test_apply_norm_standardizes():
    rng = np.random.default_rng(1)
    lm = 3.0 + 2.0 * rng.standard_normal((1, 16, 400))
    stats = fit_norm_stats([lm])
    out = apply_norm(lm, stats)
    assert out.shape == lm.shape
    assert abs(float(out.mean())) < 0.05
    assert abs(float(out.std()) - 1.0) < 0.05


def test_apply_norm_2d_and_3d_consistent():
    rng = np.random.default_rng(2)
    lm = rng.standard_normal((1, 8, 50)).astype(np.float32)
    stats = fit_norm_stats([lm])
    out3d = apply_norm(lm, stats)
    out2d = apply_norm(lm[0], stats)
    np.testing.assert_allclose(out3d[0], out2d, atol=1e-6)


def test_norm_stats_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    stats = fit_norm_stats([rng.standard_normal((1, 32, 200))])
    path = tmp_path / "norm.npz"
    stats.save(str(path))
    loaded = NormStats.load(str(path))
    np.testing.assert_allclose(loaded.mean, stats.mean)
    np.testing.assert_allclose(loaded.std, stats.std)


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        fit_norm_stats([])
