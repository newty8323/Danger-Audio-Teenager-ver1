"""Phonetic/fuzzy recovery of ASR-mangled harm NOUNS (Korean-focused).

End-to-end eval showed Whisper mis-hears domain nouns by 1-2 jamo: 필로폰->필루폰,
코카인->코케인, 엑스터시->엑스터씨, 판돈->판똥. Exact substring matching (harm_text)
then misses them. This layer decomposes Korean to jamo and matches a curated list of
high-value harm nouns within a small edit distance, recovering the near-miss without a
bigger ASR model. Deliberately narrow — only concrete drug/gambling nouns, tight
threshold — so it adds recall on garbled transcripts without inflating false positives
on clean text (a fuzzy match of a common word would be risky, so common words are excluded).
"""

from __future__ import annotations

from dataclasses import dataclass

# curated ASR-recovery targets: distinctive, >=3-syllable harm nouns. Short terms (잭팟,
# 룰렛) and ones that collide with common Korean at d1 were dropped after a real-corpus
# eval (kor_unsmile) showed them false-matching ("실루엣"->룰렛, "가카가"->바카라).
FUZZY_TERMS: dict[str, list[str]] = {
    "drug": ["필로폰", "히로뽕", "코카인", "헤로인", "엑스터시", "펜타닐", "대마초",
             # slang added 2026-07-30 after measuring ASR mangling: 작대기 -> "닭대기"
             "작대기", "약쟁이"],
    "gambling": ["풀하우스", "슬롯머신", "블랙잭", "카지노"],
    "sexual": ["일탈계"],
}
# Two-syllable slang stays OUT, confirming the original warning: adding 야짤 fuzzy-matched
# "야빨" in 4 of 300 real kor_unsmile sentences (2026-07-30). Observed ASR corruptions of
# short slang are handled as EXACT lexicon entries instead (configs/text/harm_lexicon.yaml),
# which cannot false-match.
FUZZY_TERMS_SHORT: dict[str, list[str]] = {}
MAX_JAMO_DIST = 1   # only 1 jamo error (d2 coincidentally matched real text everywhere)


def _decompose(text: str) -> list:
    """Korean syllable -> (cho, jung, jong) jamo indices; other chars pass through."""
    out: list = []
    for ch in text:
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            i = c - 0xAC00
            out.extend((("C", i // 588), ("V", (i % 588) // 28), ("T", i % 28)))
        else:
            out.append(ch)
    return out


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class FuzzyHit:
    category: str
    term: str            # canonical harm term
    matched: str         # the (mangled) span from the text
    distance: int        # jamo edit distance


def fuzzy_harm_terms(text: str, max_dist: int = MAX_JAMO_DIST) -> list[FuzzyHit]:
    """Find curated harm nouns in ``text`` allowing small jamo edit distance.

    Slides a window sized to each term over the whitespace tokens joined, comparing jamo
    sequences. Exact matches are left to the lexicon; this only fires on near-misses.
    """
    low = (text or "").replace(" ", "")
    if not low:
        return []
    # (category, term, minimum term length). Long terms are safe anywhere; the curated
    # two-syllable slang list is allowed at length 2 (see FUZZY_TERMS_SHORT).
    pool = [(c, t, 3) for c, ts in FUZZY_TERMS.items() for t in ts]
    pool += [(c, t, 2) for c, ts in FUZZY_TERMS_SHORT.items() for t in ts]
    hits: list[FuzzyHit] = []
    for cat, term, min_len in pool:
        if len(term) < min_len:     # too short to match safely
            continue
        hit = _best_match(low, cat, term, max_dist)
        if hit is not None:
            hits.append(hit)
    return hits


def _best_match(low: str, cat: str, term: str, max_dist: int) -> FuzzyHit | None:
    """Closest near-miss of `term` in `low`, or None (exact matches are the lexicon's job)."""
    tj = _decompose(term)
    term_cho = tj[0]               # leading consonant of the term
    n = len(term)
    best: FuzzyHit | None = None
    # scan windows of length n-1 .. n+1 (jamo distance tolerates length drift)
    for w in (n - 1, n, n + 1):
        if w < 2:
            continue
        for i in range(0, len(low) - w + 1):
            span = low[i:i + w]
            if span == term:
                return None        # exact -> lexicon handles it, skip fuzzy
            sj = _decompose(span)
            # require the leading consonant to match: ASR errors preserve the initial
            # sound, so this kills coincidental matches ("제로인" != 헤로인).
            if not sj or sj[0] != term_cho:
                continue
            d = _edit_distance(sj, tj)
            if d <= max_dist and (best is None or d < best.distance):
                best = FuzzyHit(cat, term, span, d)
    return best
