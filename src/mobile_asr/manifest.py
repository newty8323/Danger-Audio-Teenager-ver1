"""Manifest contract for the mobile Korean ASR experiment.

The experiment deliberately keeps four domains separate so a gain on songs cannot hide a
regression on ordinary speech.  Paths are resolved relative to the manifest file, which makes
the same manifest portable between the Mac development machine and a training host.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DOMAINS = ("general", "movie", "song", "no_speech")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ASRItem:
    item_id: str
    audio: Path
    text: str
    domain: str
    split: str
    source_id: str
    transcript_source: str = "human"
    harm_terms: tuple[str, ...] = ()

    @property
    def is_no_speech(self) -> bool:
        return self.domain == "no_speech"


def load_manifest(path: str | Path, *, require_audio: bool = True) -> list[ASRItem]:
    """Load and validate a JSONL manifest.

    Required fields: ``id``, ``audio``, ``domain``, ``split``, ``source_id`` and ``text``.
    A row may omit ``text`` only when it contains ``teacher_text``.  Human text always wins;
    teacher text is sequence-level distillation data, not a replacement for available truth.
    """
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    rows: list[ASRItem] = []
    seen_ids: set[str] = set()
    errors: list[str] = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        missing = [k for k in ("id", "audio", "domain", "split", "source_id") if not raw.get(k)]
        if missing:
            errors.append(f"line {line_no}: missing {', '.join(missing)}")
            continue
        item_id = str(raw["id"])
        if item_id in seen_ids:
            errors.append(f"line {line_no}: duplicate id {item_id!r}")
            continue
        seen_ids.add(item_id)
        domain = str(raw["domain"])
        split = str(raw["split"])
        if domain not in DOMAINS:
            errors.append(f"line {line_no}: domain must be one of {DOMAINS}, got {domain!r}")
            continue
        if split not in SPLITS:
            errors.append(f"line {line_no}: split must be one of {SPLITS}, got {split!r}")
            continue
        human_text = raw.get("text")
        teacher_text = raw.get("teacher_text")
        if human_text is None and teacher_text is None:
            errors.append(f"line {line_no}: text or teacher_text is required")
            continue
        text = str(human_text if human_text is not None else teacher_text).strip()
        transcript_source = "human" if human_text is not None else "teacher"
        if domain == "no_speech" and text:
            errors.append(f"line {line_no}: no_speech text must be empty")
            continue
        if domain != "no_speech" and not text:
            errors.append(f"line {line_no}: {domain} text must not be empty")
            continue
        audio = Path(str(raw["audio"])).expanduser()
        if not audio.is_absolute():
            audio = manifest.parent / audio
        audio = audio.resolve()
        if require_audio and not audio.is_file():
            errors.append(f"line {line_no}: audio not found: {audio}")
            continue
        harm_terms = raw.get("harm_terms", [])
        if not isinstance(harm_terms, list) or not all(isinstance(x, str) for x in harm_terms):
            errors.append(f"line {line_no}: harm_terms must be a string list")
            continue
        rows.append(ASRItem(
            item_id=item_id,
            audio=audio,
            text=text,
            domain=domain,
            split=split,
            source_id=str(raw["source_id"]),
            transcript_source=transcript_source,
            harm_terms=tuple(x.strip() for x in harm_terms if x.strip()),
        ))
    if errors:
        preview = "\n  - ".join(errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"invalid ASR manifest {manifest}:\n  - {preview}{suffix}")
    _check_source_disjoint(rows)
    return rows


def _check_source_disjoint(rows: Iterable[ASRItem]) -> None:
    by_source: dict[str, set[str]] = {}
    for row in rows:
        by_source.setdefault(row.source_id, set()).add(row.split)
    leaked = {source: splits for source, splits in by_source.items() if len(splits) > 1}
    if leaked:
        examples = ", ".join(
            f"{source}={sorted(splits)}" for source, splits in list(leaked.items())[:10]
        )
        raise ValueError(f"source-disjoint split violation: {examples}")


def domain_counts(rows: Iterable[ASRItem], split: str | None = None) -> dict[str, int]:
    counts = {domain: 0 for domain in DOMAINS}
    for row in rows:
        if split is None or row.split == split:
            counts[row.domain] += 1
    return counts
