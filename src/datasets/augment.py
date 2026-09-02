"""Feature-domain (log-mel) augmentations (spec §4).

These operate on precomputed log-mel spectrograms, so only the augmentations
that are exact/meaningful in the feature domain live here:

    SpecAugment | time-shift | gain (log-domain additive) | mixup (label-union)

Waveform-domain augmentations (MUSAN noise mixing, MP3/Opus codec simulation)
cannot be applied to a precomputed log-mel and are handled by caching extra
feature variants at the precompute stage (spec §4 "codec sim ... cached").

Every function is driven by an explicit ``numpy.random.Generator`` for
reproducibility. Features are float32 (1, F, T) unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Waveform gain g (in dB) scales power by g**2, i.e. adds ln(g**2) to a
# natural-log log-mel: shift = (dB / 20) * ln(10) * 2 = dB * ln(10) / 10.
_GAIN_DB_TO_LN = math.log(10.0) / 10.0


def apply_gain(logmel: np.ndarray, max_db: float = 6.0, p: float = 0.5,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Random gain in [-max_db, +max_db], applied as an additive log-domain shift.

    Must run on the *unnormalized* log-mel (models the raw loudness change).
    """
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return logmel
    db = rng.uniform(-max_db, max_db)
    return (logmel + db * _GAIN_DB_TO_LN).astype(np.float32)


def time_shift(feat: np.ndarray, max_frames: int = 50, p: float = 0.5,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Circularly roll along the time axis by up to ``max_frames`` (spec: ±1s)."""
    rng = rng or np.random.default_rng()
    if rng.random() >= p or max_frames <= 0:
        return feat
    shift = int(rng.integers(-max_frames, max_frames + 1))
    if shift == 0:
        return feat
    return np.roll(feat, shift, axis=-1)


def spec_augment(feat: np.ndarray, n_time_masks: int = 2, time_width: int = 64,
                 n_freq_masks: int = 2, freq_width: int = 16, p: float = 0.8,
                 mask_value: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """SpecAugment time/frequency masking (spec §4: t<=64x2, f<=16x2, p.8).

    Applied *after* normalization, so ``mask_value=0`` masks to the per-bin mean.
    Frequency is the second-to-last axis, time the last.
    """
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return feat
    out = feat.copy()
    n_freq = out.shape[-2]
    n_time = out.shape[-1]

    for _ in range(n_freq_masks):
        w = int(rng.integers(0, min(freq_width, n_freq) + 1))
        if w == 0:
            continue
        f0 = int(rng.integers(0, n_freq - w + 1))
        out[..., f0:f0 + w, :] = mask_value

    for _ in range(n_time_masks):
        w = int(rng.integers(0, min(time_width, n_time) + 1))
        if w == 0:
            continue
        t0 = int(rng.integers(0, n_time - w + 1))
        out[..., :, t0:t0 + w] = mask_value

    return out


def mixup(feat_a: np.ndarray, label_a: np.ndarray,
          feat_b: np.ndarray, label_b: np.ndarray,
          alpha: float = 0.5,
          rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mixup two samples; labels are unioned, not interpolated (spec §4 "label-union").

    feat = lam * a + (1 - lam) * b, lam ~ Beta(alpha, alpha);
    label = max(a, b) elementwise (multi-label union).
    """
    rng = rng or np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    feat = (lam * feat_a + (1.0 - lam) * feat_b).astype(np.float32)
    label = np.maximum(label_a, label_b).astype(np.float32)
    return feat, label


# ---------------------------------------------------------------------------
# Waveform-domain augmentations (spec §4), for the raw-waveform BEATs fine-tune
# path. BEATs consumes a raw 16 kHz waveform and computes its own fbank inside
# the backbone, so these run on the waveform BEFORE the model. SpecAugment stays
# feature-domain (BEATs' internal fbank is not exposed) and is intentionally not
# reproduced here; MUSAN-noise mixing accepts an external noise clip when the
# corpus is available and otherwise falls back to Gaussian noise (a stand-in).
# ---------------------------------------------------------------------------


@dataclass
class WaveAugmentConfig:
    """On-the-fly waveform augmentation knobs (spec §4)."""

    gain_max_db: float = 6.0
    gain_p: float = 0.5
    time_shift_max_sec: float = 1.0
    time_shift_p: float = 0.5
    # additive noise at a random SNR in [snr_min, snr_max] dB (spec: MUSAN 0-20 dB)
    noise_snr_min: float = 0.0
    noise_snr_max: float = 20.0
    noise_p: float = 0.5
    mixup_alpha: float = 0.5
    mixup_p: float = 0.5


def wav_gain(wav: np.ndarray, max_db: float = 6.0, p: float = 0.5,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Random gain in [-max_db, +max_db] dB applied as a linear amplitude scale."""
    rng = rng or np.random.default_rng()
    if rng.random() >= p or max_db <= 0:
        return wav
    db = rng.uniform(-max_db, max_db)
    return (wav * (10.0 ** (db / 20.0))).astype(np.float32)


def wav_time_shift(wav: np.ndarray, max_sec: float = 1.0, sample_rate: int = 16000,
                   p: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Circularly roll the waveform by up to ``max_sec`` seconds (spec: ±1 s)."""
    rng = rng or np.random.default_rng()
    max_samples = int(max_sec * sample_rate)
    if rng.random() >= p or max_samples <= 0:
        return wav
    shift = int(rng.integers(-max_samples, max_samples + 1))
    if shift == 0:
        return wav
    return np.roll(wav, shift).astype(np.float32)


def _fit_noise(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tile (if shorter) or random-crop (if longer) ``noise`` to length ``n``."""
    m = noise.shape[-1]
    if m == n:
        return noise.astype(np.float32)
    if m < n:
        reps = int(np.ceil(n / m))
        noise = np.tile(noise, reps)[:n]
        return noise.astype(np.float32)
    start = int(rng.integers(0, m - n + 1))
    return noise[start:start + n].astype(np.float32)


def wav_add_noise(wav: np.ndarray, noise: np.ndarray | None = None,
                  snr_db_range: tuple[float, float] = (0.0, 20.0), p: float = 0.5,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Add noise at a random SNR in ``snr_db_range`` (spec §4: MUSAN SNR 0-20 dB).

    ``noise`` is a waveform at the same sample rate; it is tiled/cropped to match.
    When ``noise is None`` (no MUSAN corpus yet), Gaussian noise is used as a
    stand-in. The noise is rescaled so the resulting SNR matches the drawn value.
    """
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return wav
    n = wav.shape[-1]
    if noise is None:
        noise = rng.standard_normal(n).astype(np.float32)
    else:
        noise = _fit_noise(noise, n, rng)
    sig_power = float(np.mean(wav.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    snr = float(rng.uniform(*snr_db_range))
    scale = math.sqrt((sig_power / (10.0 ** (snr / 10.0))) / noise_power)
    return (wav + scale * noise).astype(np.float32)


def wav_mixup(wav_a: np.ndarray, label_a: np.ndarray,
              wav_b: np.ndarray, label_b: np.ndarray, alpha: float = 0.5,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Waveform mixup with multi-label union (spec §4 "label-union", not interpolated)."""
    rng = rng or np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    wav = (lam * wav_a + (1.0 - lam) * wav_b).astype(np.float32)
    label = np.maximum(label_a, label_b).astype(np.float32)
    return wav, label
