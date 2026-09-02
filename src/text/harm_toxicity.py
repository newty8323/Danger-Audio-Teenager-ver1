"""Pretrained toxicity/hate-speech detector for the language branch (threat/sexual/abuse).

Our synthetic classifier stayed keyword-bound and needed idiom-patching. Pretrained Korean
toxicity models — trained on 10k-100k+ real labeled comments — generalize much better:
they read "이 영화 죽인다 최고" / "보고싶어 죽겠어" as benign without any idiom-specific
training, while still catching real toxic/threatening speech. We use them for the toxicity
side (threat / sexual / abuse); gambling & drug are NOT toxicity (they are topics), so those
stay on the lexicon (see harm_combined).

Default model: sgunderscore/hatescore-korean-hate-speech (cleanest precision + idiom
generalization in our probe). Extra models can be ensembled by max for higher recall.
Needs the 'nlp' dep group; models download on first use and run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODELS = ("sgunderscore/hatescore-korean-hate-speech",)
# labels that mean "not harmful" across the candidate models
_BENIGN = {"none", "clean", "neutral", "정상", "not_hate", "l0", "label_0"}
# coarse map from a model's raw label to our category (best-effort; risk is what matters)
_SEXUAL_HINT = ("성", "sexual", "sex")


@dataclass(frozen=True)
class ToxicityResult:
    risk: float                    # max non-benign probability in [0, 1]
    category: str | None           # 'sexual' | 'threat' | 'abuse' | None
    label: str | None = None       # raw model label (for transparency)


def _to_category(label: str) -> str:
    low = label.lower()
    if any(h in low for h in _SEXUAL_HINT):
        return "sexual"
    return "abuse"    # hate/insult/profanity/threat -> abuse (threat refined in the combiner)


class ToxicityClassifier:
    """Ensembles one or more pretrained toxicity models; risk = max non-benign score."""

    def __init__(self, models: tuple[str, ...] = DEFAULT_MODELS):
        self.model_names = models
        self._pipes: list = []

    def _ensure_loaded(self) -> None:
        if self._pipes:
            return
        from transformers import pipeline  # lazy: optional 'nlp' group
        for name in self.model_names:
            self._pipes.append(
                pipeline("text-classification", model=name, top_k=None,
                         function_to_apply="sigmoid"))

    def score(self, text: str) -> ToxicityResult:
        if not (text or "").strip():
            return ToxicityResult(risk=0.0, category=None)
        self._ensure_loaded()
        best_risk, best_label = 0.0, None
        for pipe in self._pipes:
            for o in pipe(text)[0]:
                if o["label"].lower() in _BENIGN:
                    continue
                if o["score"] > best_risk:
                    best_risk, best_label = o["score"], o["label"]
        cat = _to_category(best_label) if (best_label and best_risk >= 0.5) else None
        return ToxicityResult(risk=round(float(best_risk), 4), category=cat, label=best_label)


_DEFAULT: ToxicityClassifier | None = None


def get_classifier() -> ToxicityClassifier:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ToxicityClassifier()
    return _DEFAULT
