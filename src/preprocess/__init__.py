"""Audio preprocessing: raw clip -> normalized log-mel spectrogram (spec §4)."""

from preprocess.audio import load_audio, passes_rms_gate, rms_dbfs
from preprocess.config import PreprocessConfig
from preprocess.logmel import LogMelExtractor
from preprocess.normalize import NormStats, apply_norm, fit_norm_stats

__all__ = [
    "PreprocessConfig",
    "LogMelExtractor",
    "NormStats",
    "load_audio",
    "rms_dbfs",
    "passes_rms_gate",
    "fit_norm_stats",
    "apply_norm",
]
