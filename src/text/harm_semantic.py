"""Semantic harmful-speech classifier (multilingual, generalizing).

The lexicon (``harm_text.py``) matches enumerated terms and therefore misses
*implicit* harm ("두고 봐, 후회하게 만들어줄게" — a threat with no listed word) and
misfires on idioms ("이 영화 죽인다"). This module classifies by MEANING instead:
a multilingual sentence encoder (multilingual-e5) embeds the input and a versioned
bank of semantic prototypes (``configs/text/harm_prototypes.yaml``); the input is
scored by cosine similarity to the nearest prototypes per category. Novel phrasings
that share no words with any prototype still land near the right cluster — that is
the generalization the keyword filter cannot provide.

Decision uses the MARGIN between the nearest harmful prototype and the nearest safe
prototype (raw e5 cosines are compressed ~0.75-0.92, so only the margin separates harm
from benign): ``semantic_risk = sigmoid(K * (margin - DELTA))`` and ``top_category`` = the
nearest harmful class (None when the margin is below the decision boundary). Requires the
optional ``nlp`` dependency group (transformers + sentence-transformers); the model
downloads on first use and runs on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_PROTOS = Path(__file__).resolve().parents[2] / "configs" / "text" / "harm_prototypes.yaml"
MODEL_NAME = "intfloat/multilingual-e5-base"
SAFE = "safe"
# Decision uses the MARGIN between the nearest harmful prototype and the nearest safe
# prototype: raw e5 cosines are compressed (~0.75-0.92) so their absolute value is
# uninformative — only (best_harmful - safe) separates harm from benign. Calibrated on a
# diagnostic set: DELTA=0.02 gives ~1/16 benign FP and ~1/11 harmful miss; K spreads the
# margin over [0,1] with risk=0.5 exactly at the decision boundary.
DELTA = 0.02
K = 40.0


@dataclass(frozen=True)
class SemanticHarmResult:
    semantic_risk: float                       # sigmoid(K*(margin-DELTA)) in [0, 1]
    top_category: str | None                   # nearest harmful category, or None if benign
    margin: float = 0.0                        # best_harmful_sim - safe_sim (the signal)
    categories: dict[str, float] = field(default_factory=dict)  # harmful category -> cosine
    safe_sim: float = 0.0                       # nearest-safe-prototype cosine
    nearest: dict[str, str] = field(default_factory=dict)       # category -> nearest prototype


def load_prototypes(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else _DEFAULT_PROTOS) as f:
        return yaml.safe_load(f)["categories"]


class SemanticHarmClassifier:
    """Prototype-similarity harmful-speech classifier. Lazy-loads the encoder + caches
    prototype embeddings so repeated ``score`` calls are cheap."""

    def __init__(self, prototypes: dict | None = None, model_name: str = MODEL_NAME,
                 delta: float = DELTA, k: float = K):
        self.protos = prototypes or load_prototypes()
        self.model_name = model_name
        self.delta = delta
        self.k = k
        self._model = None
        self._proto_emb: dict[str, object] = {}   # category -> (n, d) normalized embeddings
        self._proto_txt: dict[str, list[str]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy: optional 'nlp' group
        self._model = SentenceTransformer(self.model_name)
        for cat, texts in self.protos.items():
            self._proto_txt[cat] = list(texts)
            # e5 convention: prefix passages with "passage: "
            self._proto_emb[cat] = self._model.encode(
                [f"passage: {t}" for t in texts], normalize_embeddings=True)

    def score(self, text: str) -> SemanticHarmResult:
        if not (text or "").strip():   # Whisper returns "" on silence — a normal case
            return SemanticHarmResult(semantic_risk=0.0, top_category=None)
        self._ensure_loaded()
        import numpy as np
        q = self._model.encode([f"query: {text}"], normalize_embeddings=True)[0]

        # per-category nearest-prototype cosine similarity + which prototype
        sims: dict[str, float] = {}
        nearest: dict[str, str] = {}
        for cat, emb in self._proto_emb.items():
            cos = emb @ q
            j = int(np.argmax(cos))
            sims[cat] = float(cos[j])
            nearest[cat] = self._proto_txt[cat][j]

        safe_sim = sims.get(SAFE, 0.0)
        harmful = {c: v for c, v in sims.items() if c != SAFE}
        top = max(harmful, key=harmful.get) if harmful else None
        margin = (harmful[top] - safe_sim) if top is not None else -1.0
        # margin -> risk: sigmoid centered on the decision boundary DELTA.
        risk = 1.0 / (1.0 + math.exp(-self.k * (margin - self.delta)))
        flagged = margin > self.delta
        return SemanticHarmResult(
            semantic_risk=round(risk, 4),
            top_category=top if flagged else None,
            margin=round(margin, 4),
            categories={c: round(v, 4) for c, v in harmful.items()},
            safe_sim=round(safe_sim, 4),
            nearest={c: nearest[c] for c in sims},
        )


_DEFAULT_CLF: SemanticHarmClassifier | None = None


def score_semantic(text: str) -> SemanticHarmResult:
    """Module-level convenience using a cached default classifier."""
    global _DEFAULT_CLF
    if _DEFAULT_CLF is None:
        _DEFAULT_CLF = SemanticHarmClassifier()
    return _DEFAULT_CLF.score(text)
