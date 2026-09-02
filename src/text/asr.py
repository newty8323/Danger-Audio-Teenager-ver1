"""Speech-to-text via Whisper (multimodal language layer).

Transcribes speech to text (Korean by default) so the text-harm layer can flag
dangerous *content* that the acoustic model can't hear (threats, gambling talk).
Whisper runs locally; the model downloads on first use.
"""

from __future__ import annotations

import numpy as np

_MODELS: dict[str, object] = {}


def _get_model(name: str):
    if name not in _MODELS:
        import whisper  # lazy: heavy import, optional 'asr' dep group
        _MODELS[name] = whisper.load_model(name)
    return _MODELS[name]


def transcribe(
    audio: str | np.ndarray,
    model: str = "small",
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """Transcribe an audio file path or 16kHz float32 mono waveform to text.

    ``language=None`` auto-detects (Korean + English); pass "ko"/"en" to force.
    ``prompt`` biases recognition toward domain terms (Whisper ``initial_prompt``) —
    passing the harm lexicon markedly improves Korean recall of terms like
    "제삿날"/"필로폰" without a larger model. See ``harm_text.vocabulary_prompt``.
    """
    m = _get_model(model)
    src = np.asarray(audio, dtype=np.float32) if isinstance(audio, np.ndarray) else audio
    return m.transcribe(src, language=language, fp16=False, initial_prompt=prompt)["text"].strip()
