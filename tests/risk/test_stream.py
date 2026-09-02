import pytest

from risk.policy import BLOCK, SAFE, WARN, load_risk_policy
from risk.stream import StreamRiskTracker


@pytest.fixture
def policy():
    return load_risk_policy()


def test_first_update_smoothed_equals_raw(policy):
    t = StreamRiskTracker(policy)
    s = t.update(0.5)
    assert s.smoothed == 0.5


def test_ema_smoothing(policy):
    t = StreamRiskTracker(policy)  # lambda 0.3
    t.update(0.0)
    s = t.update(1.0)
    # 0.3*1.0 + 0.7*0.0 = 0.3
    assert abs(s.smoothed - 0.3) < 1e-9
    assert s.level == SAFE  # 0.3 < tau_warn


def test_three_consecutive_warns_escalate_to_block(policy):
    t = StreamRiskTracker(policy)
    s1 = t.update(0.5)
    s2 = t.update(0.5)
    s3 = t.update(0.5)
    assert (s1.level, s1.consecutive_warns) == (WARN, 1)
    assert (s2.level, s2.consecutive_warns) == (WARN, 2)
    assert s3.base_level == WARN and s3.level == BLOCK  # escalated
    assert s3.consecutive_warns == 3


def test_block_by_threshold_immediately(policy):
    t = StreamRiskTracker(policy)
    s = t.update(0.9)
    assert s.base_level == BLOCK and s.level == BLOCK
    assert s.consecutive_warns == 0


def test_safe_interrupts_consecutive_warns(policy):
    t = StreamRiskTracker(policy)
    t.update(0.5)  # warn, consec 1
    t.update(0.5)  # warn, consec 2
    s3 = t.update(0.1)  # 0.3*0.1+0.7*0.5=0.38 -> safe, resets
    assert s3.level == SAFE and s3.consecutive_warns == 0
    s4 = t.update(0.5)  # back to warn, consec 1 (not block)
    assert s4.level == WARN and s4.consecutive_warns == 1


def test_stride_densifies_on_warn(policy):
    t = StreamRiskTracker(policy)
    warn = t.update(0.5)
    assert warn.level == WARN and warn.stride_s == policy.stride_densified_s
    t.reset()
    safe = t.update(0.1)
    assert safe.level == SAFE and safe.stride_s == policy.stride_default_s
    t.reset()
    block = t.update(0.9)
    assert block.level == BLOCK and block.stride_s == policy.stride_default_s


def test_reset_clears_state(policy):
    t = StreamRiskTracker(policy)
    t.update(0.5)
    t.update(0.5)
    t.reset()
    assert t.smoothed is None
    s = t.update(0.5)
    assert s.consecutive_warns == 1  # counter restarted
