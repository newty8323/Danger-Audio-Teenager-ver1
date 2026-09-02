"""Build a manifest from AudioSet segment CSVs (spec §3).

AudioSet ships weak segment labels as CSV (``YTID, start_seconds, end_seconds,
"mid,mid,..."``); audio is fetched separately from YouTube (see
``collect.download``). This module maps AudioSet mids to our taxonomy, selects
matching 10s segments, and emits a :class:`ClipRecord` manifest with weak labels.

The mid->class map is a versioned config; validate it against the real
``ontology.json`` before a run (``validate_label_map``) so a stale/typo'd mid
fails loudly instead of silently dropping a class.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from datasets.manifest import ClipRecord, write_manifest


@dataclass(frozen=True)
class AudioSetSegment:
    ytid: str
    start: float
    end: float
    mids: tuple[str, ...]

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def parse_ontology(path: str | Path) -> dict[str, str]:
    """Return {mid: name} from AudioSet ontology.json."""
    with open(path) as f:
        entries = json.load(f)
    return {e["id"]: e["name"] for e in entries}


def parse_segments_csv(path: str | Path) -> Iterator[AudioSetSegment]:
    """Yield segments from an AudioSet CSV, skipping the '#' header lines.

    Uses the csv module (skipinitialspace) so the quoted, comma-joined label
    field is parsed as a single column.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                continue
            ytid, start, end = row[0], row[1], row[2]
            mids = tuple(m for m in row[3].split(",") if m)
            yield AudioSetSegment(ytid=ytid, start=float(start), end=float(end), mids=mids)


def load_label_map(path: str | Path) -> dict[str, list[str]]:
    """Return {our_class: [mid, ...]} from the versioned yaml."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {cls: list(mids) for cls, mids in raw["map"].items()}


def invert_label_map(label_map: dict[str, list[str]]) -> dict[str, set[str]]:
    """Return {mid: {our_class, ...}} (a mid may map to several classes)."""
    inv: dict[str, set[str]] = defaultdict(set)
    for cls, mids in label_map.items():
        for mid in mids:
            inv[mid].add(cls)
    return dict(inv)


def validate_label_map(label_map: dict[str, list[str]], ontology: dict[str, str]) -> list[str]:
    """Return mids referenced by the map that are absent from the ontology."""
    unknown = []
    for mids in label_map.values():
        for mid in mids:
            if mid not in ontology:
                unknown.append(mid)
    return unknown


def describe_label_map(
    label_map: dict[str, list[str]], ontology: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Return (our_class, mid, ontology_name) rows for human eyeballing.

    Existence validation can't tell a real-but-misassigned mid from a correct one;
    printing the ontology name lets a human catch e.g. clap -> "Applause".
    """
    rows: list[tuple[str, str, str]] = []
    for cls, mids in label_map.items():
        for mid in mids:
            rows.append((cls, mid, ontology.get(mid, "<MISSING>")))
    return rows


def select_segments(
    segments: Iterable[AudioSetSegment],
    label_map: dict[str, list[str]],
    per_class_cap: int | None = None,
    split: str = "train",
    label_confidence: str = "weak",
) -> list[ClipRecord]:
    """Map matching segments to ClipRecords (weak labels, source_id = YTID).

    ``per_class_cap`` is an approximate per-class floor (spec §3: 1k-5k/class): a
    segment is kept if at least one of its classes is still under cap, then all its
    classes increment — so a class can slightly overshoot via co-occurring labels.
    """
    mid_to_classes = invert_label_map(label_map)
    counts: dict[str, int] = defaultdict(int)
    records: list[ClipRecord] = []

    for seg in segments:
        classes: set[str] = set()
        for mid in seg.mids:
            classes |= mid_to_classes.get(mid, set())
        if not classes:
            continue
        if per_class_cap is not None and all(counts[c] >= per_class_cap for c in classes):
            continue
        for c in classes:
            counts[c] += 1
        records.append(
            ClipRecord(
                clip_id=f"{seg.ytid}_{seg.start:g}",
                source="audioset",
                source_id=seg.ytid,
                start_sec=seg.start,
                duration=seg.duration,
                labels=sorted(classes),
                label_confidence=label_confidence,
                split=split,
            )
        )
    return records


def build_manifest(
    csv_paths: Iterable[str | Path],
    label_map_path: str | Path,
    out_path: str | Path,
    per_class_cap: int | None = None,
    split: str = "train",
) -> list[ClipRecord]:
    label_map = load_label_map(label_map_path)
    segments: list[AudioSetSegment] = []
    for p in csv_paths:
        segments.extend(parse_segments_csv(p))
    records = select_segments(segments, label_map, per_class_cap=per_class_cap, split=split)
    write_manifest(records, out_path)
    return records


# ---- CLI ----

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AudioSet manifest builder / validator.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a manifest from segment CSV(s)")
    b.add_argument("--csv", nargs="+", required=True)
    b.add_argument("--label-map", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--per-class-cap", type=int, default=None)
    b.add_argument("--split", default="train")

    v = sub.add_parser("validate", help="check the label map against ontology.json")
    v.add_argument("--label-map", required=True)
    v.add_argument("--ontology", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.cmd == "validate":
        label_map = load_label_map(args.label_map)
        ontology = parse_ontology(args.ontology)
        for cls, mid, name in describe_label_map(label_map, ontology):
            print(f"  {cls:14s} {mid:14s} {name}")
        unknown = validate_label_map(label_map, ontology)
        if unknown:
            raise SystemExit(f"{len(unknown)} unknown mid(s): {unknown}")
        print("OK: all mids exist in ontology — eyeball the names above for mismatches")
        return
    records = build_manifest(
        args.csv, args.label_map, args.out,
        per_class_cap=args.per_class_cap, split=args.split,
    )
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
