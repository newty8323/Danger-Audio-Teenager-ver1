"""End-to-end clip preprocessing (spec §4).

    raw file -> 16 kHz mono -> RMS gate -> fix length (10s) -> log-mel (1, 128, ~501)

Gated-out (too quiet) clips return ``None`` so callers can drop them from the
manifest. Normalization is intentionally *not* applied here — features are
stored unnormalized and normalized at load time via a saved ``NormStats``.
"""

from __future__ import annotations

import numpy as np

from preprocess.audio import load_audio, passes_rms_gate
from preprocess.config import PreprocessConfig
from preprocess.logmel import LogMelExtractor


def fix_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (right, zeros) or truncate a waveform to exactly ``target_len`` samples.

    Real AudioSet segments can be shorter than the requested 10s (the source video
    ended early), which otherwise yields variable-T features that break batching
    and mixup. spec §4 defines a fixed 10s clip, so we enforce it here.
    """
    n = len(waveform)
    if n == target_len:
        return waveform
    if n > target_len:
        return waveform[:target_len]
    out = np.zeros(target_len, dtype=waveform.dtype)
    out[:n] = waveform
    return out


def preprocess_clip(
    path: str,
    extractor: LogMelExtractor,
    cfg: PreprocessConfig | None = None,
) -> np.ndarray | None:
    """Return the fixed-length log-mel (1, n_mels, T) for a clip, or None if RMS-gated.

    ``extractor`` is passed in (not constructed here) so the mel filterbank is
    built once and reused across a whole dataset.
    """
    cfg = cfg or extractor.cfg
    waveform = load_audio(path, sample_rate=cfg.sample_rate, mono=cfg.mono)
    if not passes_rms_gate(waveform, cfg.rms_gate_dbfs):
        return None
    waveform = fix_length(waveform, cfg.clip_samples)
    return extractor(waveform)
