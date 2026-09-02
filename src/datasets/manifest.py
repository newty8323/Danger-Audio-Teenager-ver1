"""Clip manifest schema and jsonl I/O (spec §3).

One JSON object per line. Fields mirror the spec manifest. Validation is against
the class taxonomy and the confidence/split enums so a malformed manifest fails
loudly at load time rather than silently mid-training.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets.taxonomy import Taxonomy

CONFIDENCE_LEVELS = ("verified", "pseudo", "weak")
SPLITS = ("train", "val", "test")


@dataclass
class ClipRecord:
    clip_id: str
    source: str  # dataset/source name (audioset, fsd50k, youtube, in_the_wild, ...)
    source_id: str  # video/channel id — split disjointness is enforced on this
    start_sec: float
    duration: float
    labels: list[str]
    label_confidence: str  # one of CONFIDENCE_LEVELS
    split: str  # one of SPLITS
    annotator: str | None = None
    snr_est: float | None = None
    # Ambiguous clips are flagged and excluded from *training* (spec §3 weak-label rule).
    flagged: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> ClipRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in obj.items() if k in known})

    def multihot(self, taxonomy: Taxonomy):
        return taxonomy.encode(self.labels)


@dataclass
class ValidationError:
    clip_id: str
    problem: str


def validate_record(record: ClipRecord, taxonomy: Taxonomy) -> list[ValidationError]:
    errors: list[ValidationError] = []

    def err(msg: str) -> None:
        errors.append(ValidationError(record.clip_id, msg))

    if not record.clip_id:
        err("empty clip_id")
    if record.label_confidence not in CONFIDENCE_LEVELS:
        err(f"bad label_confidence {record.label_confidence!r}")
    if record.split not in SPLITS:
        err(f"bad split {record.split!r}")
    if record.start_sec < 0:
        err(f"negative start_sec {record.start_sec}")
    if record.duration <= 0:
        err(f"non-positive duration {record.duration}")
    for label in record.labels:
        if label not in taxonomy.categories:
            err(f"unknown label {label!r}")
    return errors


def validate_manifest(records: Iterable[ClipRecord], taxonomy: Taxonomy) -> list[ValidationError]:
    records = list(records)
    errors: list[ValidationError] = []
    seen_ids: set[str] = set()
    for r in records:
        if r.clip_id in seen_ids:
            errors.append(ValidationError(r.clip_id, "duplicate clip_id"))
        seen_ids.add(r.clip_id)
        errors.extend(validate_record(r, taxonomy))
    return errors


class ManifestError(ValueError):
    pass


def read_manifest(path: str | Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{path}:{lineno}: invalid JSON: {e}") from e
            records.append(ClipRecord.from_json(obj))
    return records


def write_manifest(records: Iterable[ClipRecord], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def load_manifest(path: str | Path, taxonomy: Taxonomy) -> list[ClipRecord]:
    """Read and validate a manifest; raise on any validation error."""
    records = read_manifest(path)
    errors = validate_manifest(records, taxonomy)
    if errors:
        preview = "; ".join(f"{e.clip_id}: {e.problem}" for e in errors[:10])
        raise ManifestError(f"{len(errors)} validation error(s): {preview}")
    return records


def training_records(records: Iterable[ClipRecord]) -> Iterator[ClipRecord]:
    """Train-split records that are not flagged as ambiguous (spec §3)."""
    for r in records:
        if r.split == "train" and not r.flagged:
            yield r
