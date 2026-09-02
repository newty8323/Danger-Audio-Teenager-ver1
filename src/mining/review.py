"""Annotator review session (spec §7 step 3).

Pure, UI-independent logic behind the Streamlit annotator: load a candidate
queue, record one-click decisions, and export confirmed decisions as train
records. Kept in ``src`` (not ``tools/``) so it is unit-tested; the Streamlit app
is a thin wrapper over this.

Decision actions:
  - false_positive: confirmed hard negative -> label is a confusable class
  - positive:       confirmed missed harm -> label is a harm class
  - reject:         not a useful example -> dropped
  - skip:           undecided (revisit later)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets.manifest import ClipRecord
from datasets.taxonomy import Taxonomy
from mining.candidates import ReviewCandidate, read_review_queue
from mining.hnm import promote_false_positives, promote_positives

FALSE_POSITIVE = "false_positive"
POSITIVE = "positive"
REJECT = "reject"
SKIP = "skip"
_ACTIONS = {FALSE_POSITIVE, POSITIVE, REJECT, SKIP}


@dataclass
class Decision:
    action: str
    label: str | None = None


class ReviewSession:
    def __init__(
        self, candidates: list[ReviewCandidate], taxonomy: Taxonomy | None = None
    ) -> None:
        self.candidates = candidates
        self._by_id = {c.clip_id: c for c in candidates}
        self.decisions: dict[str, Decision] = {}
        # If set, decide() validates the label kind immediately (fp->confusable,
        # positive->harm) instead of deferring the error to export().
        self.taxonomy = taxonomy

    @classmethod
    def from_queue(cls, path: str | Path, taxonomy: Taxonomy | None = None) -> ReviewSession:
        return cls(read_review_queue(path), taxonomy=taxonomy)

    def decide(self, clip_id: str, action: str, label: str | None = None) -> None:
        if clip_id not in self._by_id:
            raise KeyError(f"unknown candidate {clip_id!r}")
        if action not in _ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action in (FALSE_POSITIVE, POSITIVE):
            if not label:
                raise ValueError(f"action {action!r} requires a label")
            if self.taxonomy is not None:
                self._validate_label(action, label)
        self.decisions[clip_id] = Decision(action=action, label=label)

    def _validate_label(self, action: str, label: str) -> None:
        if label not in self.taxonomy.categories:
            raise ValueError(f"unknown label {label!r}")
        is_harm = self.taxonomy.is_harm(label)
        if action == FALSE_POSITIVE and is_harm:
            raise ValueError(f"false_positive needs a confusable label, got harm {label!r}")
        if action == POSITIVE and not is_harm:
            raise ValueError(f"positive needs a harm label, got confusable {label!r}")

    def pending(self) -> list[ReviewCandidate]:
        """Undecided candidates first, then skipped ones (so a skip moves on)."""
        undecided = [c for c in self.candidates if c.clip_id not in self.decisions]
        skipped = [
            c for c in self.candidates
            if c.clip_id in self.decisions and self.decisions[c.clip_id].action == SKIP
        ]
        return undecided + skipped

    def progress(self) -> tuple[int, int]:
        done = sum(1 for d in self.decisions.values() if d.action != SKIP)
        return done, len(self.candidates)

    def export(self, taxonomy: Taxonomy) -> list[ClipRecord]:
        """New train records from confirmed decisions (fp -> confusable, positive -> harm)."""
        fp = {cid: d.label for cid, d in self.decisions.items() if d.action == FALSE_POSITIVE}
        pos = {cid: d.label for cid, d in self.decisions.items() if d.action == POSITIVE}
        records = promote_false_positives(self.candidates, fp, taxonomy)
        records += promote_positives(self.candidates, pos, taxonomy)
        return records

    def save_decisions(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for clip_id, d in self.decisions.items():
                f.write(json.dumps({"clip_id": clip_id, **asdict(d)}) + "\n")

    def load_decisions(self, path: str | Path) -> ReviewSession:
        """Resume: apply a previously saved decisions log."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.decide(obj["clip_id"], obj["action"], obj.get("label"))
        return self
