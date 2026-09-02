import numpy as np
import pytest

from datasets.taxonomy import load_taxonomy
from risk.policy import (
    BLOCK,
    SAFE,
    WARN,
    RiskPolicy,
    load_risk_policy,
    risk_level,
    validate_weights,
    weight_vector,
)


def test_shipped_policy_loads_and_matches_spec_weights():
    p = load_risk_policy()
    assert p.weights["sex_moan"] == 1.0
    assert p.weights["vio_gunshot"] == 1.0
    assert p.weights["vio_verbal"] == 0.6
    assert p.weights["gmb_machine"] == 0.7
    assert p.tau_warn == 0.4 and p.tau_block == 0.7
    assert p.ema_lambda == 0.3 and p.consecutive_warns_to_block == 3


def test_shipped_weights_are_all_harm_classes():
    p = load_risk_policy()
    assert validate_weights(p, load_taxonomy()) == []


def test_weight_vector_zero_for_confusables():
    tax = load_taxonomy()
    vec = weight_vector(load_risk_policy(), tax)
    assert vec.shape == (tax.num_classes,)
    assert vec[tax.index_of("vio_gunshot")] == 1.0
    assert vec[tax.index_of("asmr")] == 0.0  # confusable -> no risk contribution


def test_weight_vector_rejects_non_harm_class():
    tax = load_taxonomy()
    bad = RiskPolicy(version="t", weights={"asmr": 1.0})
    with pytest.raises(ValueError):
        weight_vector(bad, tax)


def test_risk_level_thresholds():
    p = load_risk_policy()
    assert risk_level(0.1, p) == SAFE
    assert risk_level(0.4, p) == WARN  # boundary is warn
    assert risk_level(0.55, p) == WARN
    assert risk_level(0.7, p) == BLOCK  # boundary is block
    assert risk_level(0.9, p) == BLOCK


def test_policy_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        RiskPolicy(version="t", weights={}, tau_warn=0.8, tau_block=0.5)
    with pytest.raises(ValueError):
        RiskPolicy(version="t", weights={}, ema_lambda=0.0)


def test_weight_vector_dtype():
    vec = weight_vector(load_risk_policy(), load_taxonomy())
    assert vec.dtype == np.float64
