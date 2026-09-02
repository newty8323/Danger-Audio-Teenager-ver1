"""Multimodal language layer: ASR (speech->text) + text-harm scoring."""

from text.harm_text import TextHarmResult, load_lexicon, score_text

__all__ = ["TextHarmResult", "load_lexicon", "score_text"]
