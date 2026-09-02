"""Build training text that carries REAL ASR errors: text -> TTS -> noise -> ASR -> text.

The synthetic jamo corruption used in eval_text_asr_noise.py turned out to be optimistic by
2-19pt vs real Moonshine output (model_light §2-4 condition ③), because real errors are not
jamo substitutions — they replace a harmful word with a plausible BENIGN one ("떨" ->
"오른쪽", "먹튀" -> "목키") or delete it. Training on the real thing is the honest version of
the augmentation that already doubled recall at CER20.

Each input row becomes up to len(SNRS)+1 output rows (clean + one per SNR), keeping the
ORIGINAL label: the model learns that the corrupted string still means the same harm.
Rows whose transcript is empty or an ASR artifact are dropped. Progress is checkpointed, so
a re-run resumes instead of re-synthesizing.

NOT every corruption is worth learning. When the ASR turns a harmful word into a COMMON one
("떨" -> "또", "허브" -> "커브"), the evidence is gone and the only thing left to learn is
ordinary phrasing ("또 한번 할래") — training on that manufactures false positives. When it
turns into a DISTINCTIVE non-word ("먹튀" -> "목키", "야짤" -> "예짤", "작대기" -> "닭대기"),
the string is learnable and safe. So a row is kept only if it still contains at least one
token absent from real benign Korean (kor_unsmile clean vocabulary). Set REQUIRE_RARE=0 to
keep everything and see the false-positive cost for yourself.

Env: SOURCES (default "configs/text/slang_train_corpus.jsonl"), LIMIT (0=all),
     SNRS "10" (clean always produced), ASR_MODEL tiny|base, SEED 5, REQUIRE_RARE 1,
     OUT configs/text/asr_corrupted_corpus.jsonl
Run: uv run --group nlp --group asr python .autorun/make_asr_corrupted_corpus.py
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

SOURCES = os.environ.get("SOURCES", "configs/text/slang_train_corpus.jsonl").split(",")
LIMIT = int(os.environ.get("LIMIT", "0"))
SNRS = [float(s) for s in os.environ.get("SNRS", "10").split(",") if s.strip()]
ASR_MODEL = os.environ.get("ASR_MODEL", "tiny")
SEED = int(os.environ.get("SEED", "5"))
REQUIRE_RARE = os.environ.get("REQUIRE_RARE", "1") == "1"
OUT = _ROOT / os.environ.get("OUT", "configs/text/asr_corrupted_corpus.jsonl")
SR = 16000
ARTIFACTS = {"audiotext", "audio text"}


def benign_vocabulary() -> set[str]:
    """Whitespace tokens of real benign Korean (kor_unsmile TRAIN clean)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/train-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    vocab: set[str] = set()
    for s in df[df["clean"] == 1]["문장"]:
        vocab.update(str(s).split())
    return vocab


def has_rare_token(text: str, vocab: set[str], min_len: int = 2) -> bool:
    """True if some token never occurs in benign Korean — the learnable, safe case."""
    return any(len(t) >= min_len and t not in vocab for t in text.split())


def load_rows() -> list[dict]:
    rows = []
    for src in SOURCES:
        p = _ROOT / src.strip()
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("label") and r.get("text"):
                    rows.append({"text": r["text"], "label": r["label"]})
    return rows[:LIMIT] if LIMIT else rows


def main():
    from eval_e2e_text import TTS, load_noise_pool, mix_snr

    from cascade.pipeline import MOONSHINE_BASE, MOONSHINE_TINY, MoonshineASR

    rng = np.random.default_rng(SEED)
    rows = load_rows()
    done = set()
    if OUT.exists():                            # resume
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["source_text"])
        print(f"[corrupt] resuming: {len(done)} source rows already done", flush=True)
    todo = [r for r in rows if r["text"] not in done]
    print(f"[corrupt] {len(todo)} of {len(rows)} rows to process "
          f"(x{len(SNRS) + 1} conditions)", flush=True)
    if not todo:
        return

    tts = TTS("cuda")
    mid = {"tiny": MOONSHINE_TINY, "base": MOONSHINE_BASE}.get(ASR_MODEL, ASR_MODEL)
    asr = MoonshineASR(None, model_id=mid)
    noise = load_noise_pool(rng, k=80)
    vocab = benign_vocabulary() if REQUIRE_RARE else set()
    print(f"[corrupt] tts=mms-tts-kor asr={mid} noise clips={len(noise)} "
          f"benign vocab={len(vocab)} require_rare={REQUIRE_RARE}", flush=True)

    t0, written, kept, dropped_common = time.time(), 0, 0, 0
    with OUT.open("a") as f:
        for i, r in enumerate(todo):
            try:
                wav = tts.say(r["text"])
            except Exception:
                continue
            variants = [("clean", wav)]
            for snr in SNRS:
                nz = noise[int(rng.integers(len(noise)))]
                variants.append((f"snr{int(snr)}", mix_snr(wav, nz, snr, rng)))
            for cond, w in variants:
                hyp = asr.transcribe(w).strip()
                written += 1
                if not hyp or hyp.strip(".!?").lower() in ARTIFACTS:
                    continue
                if hyp.replace(" ", "") == r["text"].replace(" ", ""):
                    continue                     # unchanged: adds nothing over the source row
                if REQUIRE_RARE and not has_rare_token(hyp, vocab):
                    dropped_common += 1          # evidence gone -> would teach benign phrasing
                    continue
                f.write(json.dumps({"text": hyp, "label": r["label"], "lang": "ko",
                                    "kind": "asr_real", "cond": cond,
                                    "source_text": r["text"]}, ensure_ascii=False) + "\n")
                kept += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)} rows · kept {kept}/{written} variants · "
                      f"{el:.0f}s ({el / (i + 1):.2f}s/row)", flush=True)
    print(f"[corrupt] done: kept {kept}/{written} variants "
          f"(dropped {dropped_common} whose harmful word became a common word) -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
