"""Combine the two language-branch signals into one text-harm verdict.

- lexicon (``harm_text.score_text``): high-precision, explainable, but keyword-bound.
- semantic (``harm_semantic.score_semantic``): generalizes to implicit/novel phrasings.

Combination:
  - semantic extends recall to harmful *meaning* the lexicon can't see (implicit threats);
  - the lexicon adds explicit-term precision + an explanation;
  - but when the semantic classifier is *confidently* safe, it VETOES a lexicon match — a
    keyword hit inside a benign context is almost always an idiom/homograph ("이 영화 죽인다
    최고야", "she killed it on stage"). This is how the branch stops filtering mere words.

So ``text_risk = semantic`` normally, raised to ``max(lexicon, semantic)`` unless semantic
is confidently safe (< VETO_CEILING), in which case the lexicon hit is suppressed. Semantic
is optional (needs the 'nlp' dep group + a model download); if unavailable we degrade to
lexicon-only rather than fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from text.harm_text import TextHarmResult, score_text

# A confidently-safe semantic verdict (< VETO_CEILING) vetoes a lexicon keyword hit as an
# idiom/homograph/stance FP ("필로폰은 나쁜 거야" mentions a drug to condemn it). But NEVER
# veto a THREAT keyword: a real threat the encoder merely under-scores ("죽여버리기 전에 꺼져",
# ~0.24) must not be silently suppressed — missing a threat is the worst error for a safety
# system, so threats err toward catching (accepting rare idiom FPs like "영화 죽인다").
VETO_CEILING = 0.30
NO_VETO_CATEGORY = "threat"


# risk assigned to a fuzzy-recovered harm noun (ASR near-miss); d<=1 stronger than d==2.
FUZZY_RISK = {1: 0.75, 2: 0.6}
# a curated drug/gambling noun is strong evidence, so it is vetoed only when semantic is
# VERY confidently safe (< this, lower than the lexicon VETO_CEILING) — i.e. condemnation/
# education ("뽕 같은 거 하면 안 돼", sem ~0.08), not a mangled solicitation (sem >= ~0.28).
FUZZY_VETO_CEILING = 0.20


@dataclass(frozen=True)
class CombinedTextResult:
    text_risk: float                       # max(lexicon/semantic-combined, fuzzy)
    top_category: str | None
    lexicon_risk: float
    semantic_risk: float
    lexicon: TextHarmResult
    semantic: object | None = None         # Learned/SemanticHarmResult or None
    semantic_error: str | None = None      # why the model path was skipped, if it was
    fuzzy_risk: float = 0.0                # ASR-mangled harm-noun recovery
    fuzzy_category: str | None = None
    mode: str = "lexicon"                  # "learned" | "prototype" | "lexicon"


def score_text_all(text: str, use_semantic: bool = True) -> CombinedTextResult:
    """Score text. Prefers the LEARNED head (production); falls back to the prototype +
    lexicon + fuzzy path if the trained head is unavailable, then to lexicon-only if the
    'nlp' stack is missing."""
    lex = score_text(text)
    sem = None
    sem_err = None
    sem_risk = 0.0
    mode = "lexicon"
    if use_semantic:
        try:
            from text.harm_learned import get_classifier
            clf = get_classifier()
            if clf.available():
                sem = clf.score(text)          # learned e5+MLP head (primary)
                sem_risk = sem.harm_risk
                mode = "learned"
            else:                              # no trained head -> prototype scorer
                from text.harm_semantic import score_semantic
                sem = score_semantic(text)
                sem_risk = sem.semantic_risk
                mode = "prototype"
        except ImportError as e:  # 'nlp' group not installed
            sem_err = f"model classifier unavailable (install 'nlp' group): {e}"
        except Exception as e:  # model load failure -> degrade, don't crash
            sem_err = f"model classifier failed: {e}"

    if mode == "learned":
        # the learned head already handles idioms/stance/implicit (trained on real negatives)
        # with a 0.1% real FP — so it is the base risk. Lexicon is kept for explanation only;
        # letting it raise risk would re-introduce the keyword false positives.
        risk = sem_risk
        top = sem.top_category
    else:
        # prototype fallback: lexicon raises risk unless the prototype is confidently safe
        # (veto), and the lexicon hit isn't a threat (threats are never vetoed).
        vetoable = lex.top_category != NO_VETO_CATEGORY
        if sem is not None and sem_risk < VETO_CEILING and vetoable:
            risk = sem_risk                   # idiom/homograph/stance veto
            top = None
        else:
            risk = max(lex.text_risk, sem_risk)
            top = sem.top_category if (sem is not None and sem_risk >= lex.text_risk) \
                else lex.top_category

    # fuzzy recovery of ASR-mangled harm nouns (both modes) — high-precision (d<=1 + initial
    # consonant), vetoed only when the model is VERY confidently safe (stance/condemnation).
    fuzzy_risk, fuzzy_cat = 0.0, None
    from text.fuzzy_lexicon import fuzzy_harm_terms
    fhits = fuzzy_harm_terms(text)
    if fhits:
        best = min(fhits, key=lambda h: h.distance)
        fuzzy_cat = best.category
        fuzzy_vetoed = sem is not None and sem_risk < FUZZY_VETO_CEILING
        fuzzy_risk = 0.0 if fuzzy_vetoed else FUZZY_RISK.get(best.distance, 0.0)
        if fuzzy_risk > risk:
            risk, top = fuzzy_risk, fuzzy_cat

    return CombinedTextResult(
        text_risk=round(risk, 4),
        top_category=top,
        lexicon_risk=lex.text_risk,
        semantic_risk=round(sem_risk, 4),
        lexicon=lex,
        semantic=sem,
        semantic_error=sem_err,
        mode=mode,
        fuzzy_risk=round(fuzzy_risk, 4),
        fuzzy_category=fuzzy_cat,
    )
