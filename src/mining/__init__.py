"""Hard-negative mining loop: candidate selection + orchestration (spec §7)."""

from mining.candidates import (
    FALSE_POSITIVE,
    UNCERTAIN,
    PoolClip,
    ReviewCandidate,
    read_review_queue,
    select_candidates,
    write_review_queue,
)
from mining.config import MiningConfig, load_mining_config
from mining.hnm import (
    fp_distribution,
    promote_false_positives,
    promote_positives,
    should_stop,
)
from mining.review import Decision, ReviewSession

__all__ = [
    "PoolClip",
    "ReviewCandidate",
    "FALSE_POSITIVE",
    "UNCERTAIN",
    "select_candidates",
    "write_review_queue",
    "read_review_queue",
    "MiningConfig",
    "load_mining_config",
    "fp_distribution",
    "promote_false_positives",
    "promote_positives",
    "should_stop",
    "ReviewSession",
    "Decision",
]
