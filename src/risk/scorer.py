"""Risk scorer — Task B (spec §1, §8).

    R = sigmoid(a * max_i(w_i p_i) + b * sum_i(w_i p_i) + c)

The two features (weighted max and weighted sum of harm probabilities) are
combined by a post-hoc logistic regression fit on the val split; the fitted
(a, b, c) are a versioned artifact. Weights come from the risk policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.taxonomy import Taxonomy
from risk.policy import RiskPolicy, weight_vector


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip before exp: np.where evaluates both branches, so large |z| would raise a
    # spurious overflow warning even though the result is correct.
    z = np.clip(z, -709.0, 709.0)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _features(probs: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (weighted_max, weighted_sum) over classes for (N, C) or (C,) probs."""
    wp = np.asarray(probs, dtype=np.float64) * weights
    return wp.max(axis=-1), wp.sum(axis=-1)


def _fit_logreg(x: np.ndarray, y: np.ndarray, iters: int = 5000, lr: float = 0.5):
    """Fit logistic regression on 2 features. Standardizes internally for stable
    gradient descent, then folds the scaling back into raw coefficients."""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    xs = (x - mu) / sigma

    w = np.zeros(2)
    b = 0.0
    n = len(y)
    for _ in range(iters):
        p = _sigmoid(xs @ w + b)
        gw = xs.T @ (p - y) / n
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb

    # z = w·(x-mu)/sigma + b  ->  raw coeffs on x
    a_raw, b_raw = w / sigma
    c_raw = b - float((w * mu / sigma).sum())
    return float(a_raw), float(b_raw), float(c_raw)


@dataclass
class RiskScorer:
    weights: np.ndarray  # (num_classes,) aligned to taxonomy order
    a: float = 1.0  # coeff on weighted max
    b: float = 0.0  # coeff on weighted sum
    c: float = 0.0  # bias
    fitted: bool = False
    policy_version: str | None = None  # provenance for the fitted coeffs

    @classmethod
    def from_policy(cls, policy: RiskPolicy, taxonomy: Taxonomy) -> RiskScorer:
        return cls(weights=weight_vector(policy, taxonomy), policy_version=policy.version)

    def score(self, probs: np.ndarray, require_fitted: bool = True) -> np.ndarray | float:
        """Risk score(s) in (0, 1) for (N, C) or (C,) probabilities.

        Raises unless the coefficients have been fit/loaded: an unfitted scorer
        (a=1,b=0,c=0) gives R=sigmoid(max_i w_i p_i) >= 0.5 for ALL inputs, so it
        silently over-flags. Pass ``require_fitted=False`` only for raw/testing use.
        """
        if require_fitted and not self.fitted:
            raise RuntimeError("RiskScorer is not fitted; call fit() or load_params() first")
        f_max, f_sum = _features(probs, self.weights)
        r = _sigmoid(self.a * f_max + self.b * f_sum + self.c)
        return float(r) if np.ndim(r) == 0 else r

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> RiskScorer:
        """Fit (a, b, c) on val: targets = 1 if the clip is harmful (spec §1 Task B)."""
        f_max, f_sum = _features(probs, self.weights)
        x = np.stack([f_max, f_sum], axis=1)
        self.a, self.b, self.c = _fit_logreg(x, np.asarray(targets, dtype=np.float64))
        self.fitted = True
        return self

    def save_params(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": self.a, "b": self.b, "c": self.c,
            "fitted": self.fitted, "policy_version": self.policy_version,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_params(self, path: str | Path) -> RiskScorer:
        with open(path) as f:
            p = json.load(f)
        self.a, self.b, self.c = p["a"], p["b"], p["c"]
        self.fitted = p.get("fitted", True)
        saved_version = p.get("policy_version")
        if saved_version is not None and self.policy_version is not None \
                and saved_version != self.policy_version:
            raise ValueError(
                f"risk params fit against policy {saved_version!r} but scorer uses "
                f"{self.policy_version!r}; coefficients and weights are mismatched"
            )
        return self
