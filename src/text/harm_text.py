"""Text-harm scoring on an ASR transcript (multimodal language layer).

Substring-matches a versioned Korean lexicon (threat / gambling terms) and returns
per-category matches + a text risk in [0, 1]. Transparent and deterministic; a
semantic/LLM classifier is the upgrade path for implicit threats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "text" / "harm_lexicon.yaml"


@dataclass(frozen=True)
class TextHarmResult:
    text_risk: float                      # max category risk in [0, 1]
    categories: dict[str, float] = field(default_factory=dict)   # category -> risk
    matched: dict[str, list[str]] = field(default_factory=dict)  # category -> matched terms

    @property
    def top_category(self) -> str | None:
        return max(self.categories, key=self.categories.get) if self.categories else None


def load_lexicon(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else _DEFAULT) as f:
        return yaml.safe_load(f)


def vocabulary_prompt(lexicon: dict | None = None, per_category: int = 14) -> str:
    """Space-joined lexicon terms for Whisper's ``initial_prompt`` — biases ASR to
    recognize domain words (esp. Korean: "제삿날"/"필로폰"), improving recall cheaply.

    Takes ``per_category`` terms per category to stay within the prompt budget (Whisper
    keeps only the last ~224 tokens), sampled from the FRONT and BACK of each list: the
    lists are Korean-first / English-last, so taking only the front silently dropped every
    English term once Korean slang was added (caught by test_vocabulary_prompt_contains_terms).
    """
    lexicon = lexicon or load_lexicon()
    terms: list[str] = []
    for spec in lexicon["categories"].values():
        ts = [t["t"] for t in spec["terms"]]
        if len(ts) <= per_category:
            terms += ts
            continue
        head = per_category - per_category // 3
        terms += ts[:head] + ts[-(per_category - head):]
    return " ".join(dict.fromkeys(terms))


def score_text(text: str, lexicon: dict | None = None) -> TextHarmResult:
    lexicon = lexicon or load_lexicon()
    low = (text or "").lower()  # case-insensitive (English); Korean unaffected
    cats: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for cat, spec in lexicon["categories"].items():
        hits = [term["t"] for term in spec["terms"] if term["t"].lower() in low]
        if not hits:
            continue
        weights = {term["t"]: float(term["w"]) for term in spec["terms"]}
        raw = min(1.0, sum(weights[t] for t in hits))
        cats[cat] = round(raw * float(spec.get("weight", 1.0)), 4)
        matched[cat] = hits
    text_risk = max(cats.values(), default=0.0)
    return TextHarmResult(text_risk=text_risk, categories=cats, matched=matched)
