"""Metrics that keep ordinary speech, movies, songs and no-speech failures visible."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping

from mobile_asr.manifest import DOMAINS, ASRItem

_KEEP = re.compile(r"[^가-힣a-zA-Z0-9]")


def normalize_text(text: str) -> str:
    return _KEEP.sub("", unicodedata.normalize("NFC", text or "")).lower()


def character_error_rate(refs: Iterable[str], hyps: Iterable[str]) -> float | None:
    import jiwer

    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    return float(jiwer.cer([r for r, _ in pairs], [h or "-" for _, h in pairs]))


def evaluate_predictions(rows: list[ASRItem], hypotheses: Mapping[str, str]) -> dict:
    """Return per-domain metrics without averaging away the no-speech failure mode."""
    missing = [row.item_id for row in rows if row.item_id not in hypotheses]
    if missing:
        raise ValueError(f"missing hypotheses for {len(missing)} rows: {missing[:5]}")
    by_domain: dict[str, list[ASRItem]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)

    result: dict[str, dict] = {}
    for domain in DOMAINS:
        items = by_domain.get(domain, [])
        hyps = [hypotheses[row.item_id] for row in items]
        entry = {"n": len(items)}
        if domain == "no_speech":
            entry["false_transcript_rate"] = (
                sum(bool(normalize_text(h)) for h in hyps) / len(hyps) if hyps else None
            )
            entry["cer"] = None
        else:
            entry["cer"] = character_error_rate([row.text for row in items], hyps)
            entry["empty_rate"] = (
                sum(not normalize_text(h) for h in hyps) / len(hyps) if hyps else None
            )
        result[domain] = entry

    term_total = 0
    term_hits = 0
    for row in rows:
        hyp = normalize_text(hypotheses[row.item_id])
        for term in row.harm_terms:
            term_total += 1
            term_hits += normalize_text(term) in hyp
    result["harm_term_recall"] = term_hits / term_total if term_total else None
    result["harm_terms"] = term_total
    return result
