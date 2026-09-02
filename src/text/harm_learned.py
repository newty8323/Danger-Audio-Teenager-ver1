"""Learned text-harm classifier: frozen multilingual-e5 + a trained MLP head.

This is the production upgrade over the prototype-cosine scorer (harm_semantic): a small
MLP trained on a research-scale corpus + real benign negatives (see scripts/train_text_head.py).
On real held-out data it cut the false-positive rate 3.5%->0.1% and lifted real-speech
Korean recall 0.47->0.74 — the implicit-harm ceiling the prototype couldn't pass.

Frozen-encoder + linear/MLP probe is the standard transfer-learning recipe [linear-probe];
the encoder is multilingual-e5 [e5]. Loads ``artifacts/text_head.pt`` (versioned); if the
head or the 'nlp' stack is unavailable, callers fall back to the prototype/lexicon path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_HEAD_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "text_head.pt"
MODEL_NAME = "intfloat/multilingual-e5-base"
SAFE = "safe"


@dataclass(frozen=True)
class LearnedHarmResult:
    harm_risk: float                       # 1 - P(safe), in [0, 1]
    top_category: str | None               # argmax harmful class, or None if benign
    categories: dict[str, float] = field(default_factory=dict)  # harmful category -> prob


class LearnedHarmClassifier:
    """e5 embedding -> trained MLP head -> per-category probabilities."""

    def __init__(self, head_path: str | Path = _HEAD_PATH, model_name: str = MODEL_NAME):
        self.head_path = Path(head_path)
        self.model_name = model_name
        self._model = None
        self._head = None
        self._cats: list[str] = []

    def available(self) -> bool:
        return self.head_path.exists()

    def _ensure_loaded(self) -> None:
        if self._head is not None:
            return
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer  # optional 'nlp' group

        ckpt = torch.load(self.head_path, map_location="cpu", weights_only=False)
        self._cats = ckpt["cats"]
        head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(self._cats)))
        # state_dict keys are "net.0.*"/"net.3.*" from the training Head wrapper
        head.load_state_dict({k.replace("net.", ""): v for k, v in ckpt["state"].items()})
        head.eval()
        self._head = head
        self._model = SentenceTransformer(self.model_name)

    def score(self, text: str) -> LearnedHarmResult:
        if not (text or "").strip():
            return LearnedHarmResult(harm_risk=0.0, top_category=None)
        self._ensure_loaded()
        import torch
        emb = self._model.encode([f"query: {text}"], normalize_embeddings=True)
        with torch.no_grad():
            probs = torch.softmax(self._head(torch.from_numpy(emb).float()), dim=1)[0]
        p = {c: float(probs[i]) for i, c in enumerate(self._cats)}
        harmful = {c: round(v, 4) for c, v in p.items() if c != SAFE}
        top = max(harmful, key=harmful.get) if harmful else None
        risk = round(1.0 - p.get(SAFE, 0.0), 4)
        return LearnedHarmResult(harm_risk=risk, top_category=top if risk >= 0.5 else None,
                                 categories=harmful)


_DEFAULT: LearnedHarmClassifier | None = None


def get_classifier() -> LearnedHarmClassifier:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LearnedHarmClassifier()
    return _DEFAULT
