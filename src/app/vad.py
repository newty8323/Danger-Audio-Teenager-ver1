"""Filters for the text branch. `is_degenerate` is in use; `speech_score` is NOT — see below.

Both were written after a measured failure (2026-07-30 live run): on a non-speech crowd-noise
window, Moonshine-KR hallucinated a repetition loop ("와! 와! 와! …") which the text
classifier scored 0.908 → a false escalation.

1. `is_degenerate(text)` — **in use, post-ASR.** Catches the hallucination signature: a
   transcript dominated by one repeated token/phrase carries no information. Validated on
   real Moonshine output (the loop above, and "스탑 너무 좋아" ×3 style repeats).

2. `speech_score(wav)` — **DISABLED as a gate** (`EngineConfig.speech_min = 0.0`). The idea
   was to skip ASR on non-speech windows: voice-band energy ratio × spectral non-flatness ×
   syllable-rate modulation. It works on synthetic speech but NOT on real audio — measured
   medians: real movie dialogue **0.00**, TTS speech 0.79, gunshots up to 0.96. It suppressed
   ASR on a profanity-heavy movie clip entirely, i.e. it produced exactly the false negatives
   the system must not have. Kept as a diagnostic, and as the honest record of a rejected
   design; a real gate needs a trained VAD (e.g. silero-vad), measured the same way first.
"""
from __future__ import annotations

import re

import numpy as np

SR = 16000


def speech_score(wav: np.ndarray, sr: int = SR) -> float:
    """0..1 speech likelihood. Cheap: one STFT-free framing + rFFT per frame."""
    x = np.asarray(wav, dtype=np.float32)
    if x.size < sr // 10:
        return 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-3:                                    # digital silence / near-silence
        return 0.0

    n = 512
    hop = 256
    nframes = 1 + (len(x) - n) // hop
    if nframes < 8:
        return 0.0
    idx = np.arange(n)[None, :] + hop * np.arange(nframes)[:, None]
    frames = x[idx] * np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-10   # (frames, 257)
    freqs = np.fft.rfftfreq(n, 1 / sr)

    band = (freqs >= 300) & (freqs <= 3400)
    band_ratio = float(spec[:, band].sum() / spec.sum())

    # spectral flatness: geometric/arithmetic mean. Noise -> ~1, tonal speech -> << 1.
    logs = np.log(spec)
    flatness = float(np.exp(logs.mean(axis=1)).mean() / spec.mean(axis=1).mean())
    peakiness = 1.0 - min(1.0, flatness * 3.0)

    # syllable-rate modulation: speech energy swings frame to frame; steady noise does not.
    e = spec.sum(axis=1)
    modulation = float(np.std(e) / (np.mean(e) + 1e-12))
    mod_score = min(1.0, modulation / 0.6)

    band_score = min(1.0, max(0.0, (band_ratio - 0.25) / 0.35))
    return float(band_score * peakiness * mod_score) ** (1 / 3)


_TOKEN = re.compile(r"[가-힣a-zA-Z0-9]+")


def is_degenerate(text: str, min_tokens: int = 4, max_repeat_frac: float = 0.5,
                  min_unique_frac: float = 0.35) -> bool:
    """True when the transcript looks like an ASR hallucination loop rather than speech."""
    toks = _TOKEN.findall(text or "")
    if len(toks) < min_tokens:
        return False                                   # too short to judge; let it through
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    if max(counts.values()) / len(toks) > max_repeat_frac:
        return True                                    # one token dominates
    if len(counts) / len(toks) < min_unique_frac:
        return True                                    # very low lexical diversity
    return _has_repeated_phrase(toks)


def _has_repeated_phrase(toks: list[str], span: int = 4, times: int = 3,
                         single_token_times: int = 5) -> bool:
    """Detect an n-gram (n<=span) repeated in a row — the classic decode loop.

    A SINGLE word repeated three times is normal emphatic speech (observed on real audio:
    "거짓의 깨 깨 깨" from a movie line, which this filter wrongly discarded), so one-token
    runs need more repeats than phrase-level runs to count as a hallucination.
    """
    for n in range(1, span + 1):
        need = single_token_times if n == 1 else times
        if len(toks) < n * need:
            continue
        for i in range(len(toks) - n * need + 1):
            first = toks[i:i + n]
            if all(toks[i + k * n:i + (k + 1) * n] == first for k in range(1, need)):
                return True
    return False
