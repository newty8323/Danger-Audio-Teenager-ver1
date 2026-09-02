"""Do profanity (욕설) and slang (은어) actually survive the pipeline? — user requirement.

The existing ko harm testset has almost no profanity (5/173 rows, all "죽여/꺼져" level) and no
slang, so this capability had never been measured. Two things can break independently:

  stage 1 — ASR: does the transcript still CONTAIN the harmful token? Read-speech ASR often
            mangles or sanitizes swear words and has never seen drug/gambling slang.
  stage 2 — classifier: given a correct transcript, does it fire?

So this script reports, per row: keyword survival through real Moonshine output, and the
classifier score on both the clean text and the ASR output. Negatives (kor_unsmile clean)
set the FPR15% threshold exactly as in eval_text_asr_noise.py, so the recall numbers are
comparable to §2-4.

Env: N_NEG 300, SNRS "10,5" (clean always run), SEED 13, ASR_MODEL base|tiny,
     HYBRID=1 to score with the classifier+lexicon hybrid (cascade.pipeline.HybridTextScorer),
     OUT data_dl/asr/profanity_slang_results.json
Run: uv run --group nlp --group asr python .autorun/eval_profanity_slang.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts", _ROOT / ".autorun"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")

N_NEG = int(os.environ.get("N_NEG", "300"))
SNRS = [float(s) for s in os.environ.get("SNRS", "10,5").split(",") if s.strip()]
SEED = int(os.environ.get("SEED", "13"))
ASR_MODEL = os.environ.get("ASR_MODEL", "base")
FPR = 0.15
SR = 16000
OUT = os.environ.get("OUT", "data_dl/asr/profanity_slang_results.json")
TESTSET = _ROOT / "configs/text/profanity_slang_testset.jsonl"

# The token that MUST survive ASR for the row to be catchable by a keyword-ish model.
# Chosen per row as the most harm-bearing substring; matching is on the normalized text.
KEYWORDS = {
    "abuse": ["씨발", "개새", "병신", "지랄", "미친", "좆", "애미", "찌질", "존나", "등신", "븅신"],
    "threat": ["죽여", "죽고", "패버", "뒤지", "깨버", "쳐맞", "칼"],
    "drug": ["떨", "아이스", "작대기", "허브", "총", "물뽕", "약쟁이", "고기"],
    "gambling": ["사다리", "토토", "홀짝", "먹튀", "존버", "올인", "총판", "슬롯", "바카라", "마틴"],
    "sexual": ["조건", "일탈계", "야짤", "스폰", "썰", "능욕"],
}


def load_rows():
    rows = []
    for line in TESTSET.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_negatives(rng):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/valid-00000-of-00001.parquet",
                        repo_type="dataset")
    neg = [str(s) for s in pd.read_parquet(p).query("clean == 1")["문장"]]
    idx = rng.permutation(len(neg))[:N_NEG]
    return [neg[int(i)] for i in idx]


def keywords_for(row) -> list[str]:
    return [k for k in KEYWORDS.get(row["label"], []) if k in row["text"]]


def survived(row, hyp: str) -> bool | None:
    """True/False when the row has an identifiable keyword, None when it has none."""
    kws = keywords_for(row)
    if not kws:
        return None
    return any(k in hyp for k in kws)


def thr_at_fpr(neg: np.ndarray, fpr: float = FPR) -> float:
    k = int(np.floor(fpr * len(neg)))
    srt = np.sort(neg)[::-1]
    return float(srt[max(0, k - 1)]) if k >= 1 else float(srt[0] + 1e-6)


def main():
    from app.escalate import write_wav
    from cascade.pipeline import MOONSHINE_BASE, MOONSHINE_TINY, MoonshineASR, TextScorer
    sys.path.insert(0, str(_ROOT / ".autorun"))
    from eval_e2e_text import TTS, load_noise_pool, mix_snr

    rng = np.random.default_rng(SEED)
    rows = load_rows()
    neg_texts = load_negatives(rng)
    print(f"[prof] rows {len(rows)} (profanity "
          f"{sum(1 for r in rows if r['kind'] == 'profanity')}, "
          f"slang {sum(1 for r in rows if r['kind'] == 'slang')}) | neg {len(neg_texts)}",
          flush=True)

    model_dir = os.environ.get("MODEL_DIR") or None
    if os.environ.get("HYBRID"):
        from cascade.pipeline import HybridTextScorer
        scorer = HybridTextScorer(classifier=TextScorer(model_dir=model_dir))
        print(f"[prof] scorer = classifier({model_dir or 'default'}) + lexicon (hybrid)",
              flush=True)
    else:
        scorer = TextScorer(model_dir=model_dir)   # int8 CPU — the deployed configuration
        print(f"[prof] scorer = classifier only ({model_dir or 'default'})", flush=True)
    neg_scores = scorer.score(neg_texts)
    thr_clean = thr_at_fpr(neg_scores)

    # --- stage 2 alone: perfect transcript ---
    texts = [r["text"] for r in rows]
    s_clean = scorer.score(texts)
    res = {"threshold_clean_text": round(thr_clean, 4),
           "text_only": {"recall@fpr15": round(float((s_clean >= thr_clean).mean()), 4)}}
    for kind in ("profanity", "slang"):
        m = np.array([r["kind"] == kind for r in rows])
        res["text_only"][kind] = round(float((s_clean[m] >= thr_clean).mean()), 4)
    for lab in sorted({r["label"] for r in rows}):
        m = np.array([r["label"] == lab for r in rows])
        res["text_only"][f"label_{lab}"] = round(float((s_clean[m] >= thr_clean).mean()), 4)
    print(f"[prof] text-only (perfect transcript) recall@FPR15 = "
          f"{res['text_only']['recall@fpr15']:.3f} "
          f"(profanity {res['text_only']['profanity']:.3f} / "
          f"slang {res['text_only']['slang']:.3f})", flush=True)

    # --- stage 1+2: TTS -> noise -> real ASR -> classifier ---
    wav_dir = Path("data_dl/asr/prof_tts")
    wav_dir.mkdir(parents=True, exist_ok=True)
    tts = TTS("cuda")
    paths = []
    for i, r in enumerate(rows):
        p = wav_dir / f"row_{i:03d}.wav"
        if not p.exists():
            write_wav(p, tts.say(r["text"]))
        paths.append(p)
    del tts
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    import soundfile as sf
    noise = load_noise_pool(rng, k=80)
    mid = {"base": MOONSHINE_BASE, "tiny": MOONSHINE_TINY}.get(ASR_MODEL, ASR_MODEL)
    asr = MoonshineASR(None, model_id=mid)
    res["asr_model"] = mid
    res["conditions"] = {}
    samples = []

    for cond in ["clean"] + [f"snr{int(s)}" for s in SNRS]:
        snr = None if cond == "clean" else float(cond[3:])
        crng = np.random.default_rng(SEED + (0 if snr is None else int(snr)))
        hyps, t0 = [], time.time()
        for p in paths:
            w, _ = sf.read(p, dtype="float32")
            if snr is not None:
                w = mix_snr(w, noise[int(crng.integers(len(noise)))], snr, crng)
            hyps.append(asr.transcribe(w))
        s_asr = scorer.score(hyps)
        surv = [survived(r, h) for r, h in zip(rows, hyps, strict=True)]
        judged = [x for x in surv if x is not None]
        row_res = {
            "keyword_survival": round(float(np.mean(judged)), 4) if judged else None,
            "n_judged": len(judged),
            "recall@fpr15": round(float((s_asr >= thr_clean).mean()), 4),
            "sec": round(time.time() - t0, 1),
        }
        for kind in ("profanity", "slang"):
            m = np.array([r["kind"] == kind for r in rows])
            row_res[f"recall_{kind}"] = round(float((s_asr[m] >= thr_clean).mean()), 4)
            sk = [survived(r, h) for r, h in zip(rows, hyps, strict=True) if r["kind"] == kind]
            sk = [x for x in sk if x is not None]
            row_res[f"survival_{kind}"] = round(float(np.mean(sk)), 4) if sk else None
        res["conditions"][cond] = row_res
        print(f"  {cond:6s} keyword-survival={row_res['keyword_survival']}  "
              f"recall@FPR15={row_res['recall@fpr15']:.3f}  "
              f"(profanity {row_res['recall_profanity']:.3f} / "
              f"slang {row_res['recall_slang']:.3f})", flush=True)
        if cond == "clean":
            samples = [{"ref": r["text"], "hyp": h, "kind": r["kind"], "label": r["label"],
                        "survived": s, "score": round(float(sc), 3)}
                       for r, h, s, sc in zip(rows, hyps, surv, s_asr, strict=True)]

    res["samples_clean"] = samples
    out = _ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[prof] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
