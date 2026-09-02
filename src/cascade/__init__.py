"""3-tier on-device cascade (model_light.md §1): gate -> acoustic trigger + text branch
-> server escalation. Pure decision logic in `decision`, model wiring in `pipeline`."""
from cascade.decision import ClipDecision, Thresholds, decide, load_thresholds, save_thresholds

__all__ = ["ClipDecision", "Thresholds", "decide", "load_thresholds", "save_thresholds"]
