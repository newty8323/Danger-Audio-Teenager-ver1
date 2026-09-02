import numpy as np
import pytest

from datasets.taxonomy import load_taxonomy
from risk.policy import load_risk_policy
from risk.scorer import RiskScorer
from training.metrics import auroc


def _harmful_vs_safe(rng, tax, n=200):
    """Half harmful (high vio_gunshot prob), half safe (all low)."""
    c = tax.num_classes
    probs = rng.uniform(0.0, 0.1, size=(n, c))
    targets = np.zeros(n)
    gi = tax.index_of("vio_gunshot")
    harm = rng.choice(n, n // 2, replace=False)
    probs[harm, gi] = rng.uniform(0.7, 0.95, size=n // 2)
    targets[harm] = 1.0
    return probs, targets


def test_score_requires_fitted_by_default():
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    with pytest.raises(RuntimeError):
        scorer.score(np.zeros(tax.num_classes))  # unfitted -> would over-flag


def test_score_in_unit_interval():
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    probs = np.random.default_rng(0).uniform(0, 1, size=(10, tax.num_classes))
    scores = scorer.score(probs, require_fitted=False)
    assert scores.shape == (10,)
    assert (scores > 0).all() and (scores < 1).all()


def test_single_clip_returns_float():
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    r = scorer.score(np.zeros(tax.num_classes), require_fitted=False)
    assert isinstance(r, float)


def test_fit_separates_harmful_from_safe():
    rng = np.random.default_rng(1)
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    probs, targets = _harmful_vs_safe(rng, tax)
    scorer.fit(probs, targets)
    assert scorer.fitted
    scores = scorer.score(probs)
    # spec §9: risk binary AUC >= .95
    assert auroc(targets, scores) >= 0.95


def test_higher_harm_prob_yields_higher_risk():
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    rng = np.random.default_rng(2)
    probs, targets = _harmful_vs_safe(rng, tax)
    scorer.fit(probs, targets)
    hi = np.zeros(tax.num_classes)
    hi[tax.index_of("vio_gunshot")] = 0.95
    lo = np.zeros(tax.num_classes)
    lo[tax.index_of("vio_gunshot")] = 0.05
    assert scorer.score(hi) > scorer.score(lo)


def test_weighting_matters():
    # sex_ambient (w=0.6) at the same prob is less risky than vio_gunshot (w=1.0)
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    a = np.zeros(tax.num_classes)
    a[tax.index_of("sex_ambient")] = 0.9
    b = np.zeros(tax.num_classes)
    b[tax.index_of("vio_gunshot")] = 0.9
    assert scorer.score(b, require_fitted=False) > scorer.score(a, require_fitted=False)


def test_save_load_roundtrip(tmp_path):
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    rng = np.random.default_rng(3)
    probs, targets = _harmful_vs_safe(rng, tax)
    scorer.fit(probs, targets)
    path = tmp_path / "risk_params.json"
    scorer.save_params(path)

    fresh = RiskScorer.from_policy(load_risk_policy(), tax).load_params(path)
    assert (fresh.a, fresh.b, fresh.c) == (scorer.a, scorer.b, scorer.c)
    assert fresh.fitted
    np.testing.assert_allclose(fresh.score(probs), scorer.score(probs))


def test_load_params_detects_policy_version_mismatch(tmp_path):
    tax = load_taxonomy()
    scorer = RiskScorer.from_policy(load_risk_policy(), tax)
    rng = np.random.default_rng(4)
    probs, targets = _harmful_vs_safe(rng, tax)
    scorer.fit(probs, targets)
    path = tmp_path / "p.json"
    scorer.save_params(path)

    stale = RiskScorer.from_policy(load_risk_policy(), tax)
    stale.policy_version = "different-version"
    with pytest.raises(ValueError):
        stale.load_params(path)
