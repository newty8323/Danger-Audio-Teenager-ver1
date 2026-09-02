"""Class taxonomy (spec §2).

Loaded from ``configs/data/classes.yaml`` (versioned source of truth). Output
node order is fixed: harm groups first (sex, vio, gmb), then confusables. Code
must never hardcode the class list — always go through :func:`load_taxonomy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Repo-root-relative default config path.
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "classes.yaml"

_CONFUSABLE = "confusable"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    harm_classes: tuple[str, ...]  # ordered, sex -> vio -> gmb
    confusable_classes: tuple[str, ...]
    # class name -> category ("sex" | "vio" | "gmb" for harm, "confusable" otherwise)
    categories: dict[str, str] = field(default_factory=dict)

    @property
    def all_classes(self) -> tuple[str, ...]:
        """Ordered class list; index == output-node index."""
        return self.harm_classes + self.confusable_classes

    @property
    def num_classes(self) -> int:
        return len(self.all_classes)

    @property
    def harm_indices(self) -> tuple[int, ...]:
        """Column indices of the harm classes (for slicing (N, C) probs/labels)."""
        return tuple(range(len(self.harm_classes)))

    @property
    def harm_categories(self) -> tuple[str, ...]:
        """Distinct harm categories in first-seen order (sex, vio, gmb)."""
        seen: list[str] = []
        for c in self.harm_classes:
            cat = self.categories[c]
            if cat not in seen:
                seen.append(cat)
        return tuple(seen)

    def index_of(self, class_name: str) -> int:
        try:
            return self.all_classes.index(class_name)
        except ValueError as e:
            raise KeyError(f"unknown class {class_name!r}") from e

    def is_harm(self, class_name: str) -> bool:
        return class_name in self.harm_classes

    def category_of(self, class_name: str) -> str:
        if class_name not in self.categories:
            raise KeyError(f"unknown class {class_name!r}")
        return self.categories[class_name]

    def encode(self, labels: list[str]) -> np.ndarray:
        """Multi-hot encode a label list into a (num_classes,) float32 vector."""
        vec = np.zeros(self.num_classes, dtype=np.float32)
        for label in labels:
            vec[self.index_of(label)] = 1.0
        return vec


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    harm_groups: dict = raw["harm"]
    harm_classes: list[str] = []
    categories: dict[str, str] = {}
    for category, names in harm_groups.items():
        for name in names:
            harm_classes.append(name)
            categories[name] = category

    confusable_classes: list[str] = list(raw.get("confusable", []))
    for name in confusable_classes:
        categories[name] = _CONFUSABLE

    all_names = harm_classes + confusable_classes
    if len(all_names) != len(set(all_names)):
        raise ValueError("duplicate class names in taxonomy config")

    return Taxonomy(
        version=str(raw["version"]),
        harm_classes=tuple(harm_classes),
        confusable_classes=tuple(confusable_classes),
        categories=categories,
    )
