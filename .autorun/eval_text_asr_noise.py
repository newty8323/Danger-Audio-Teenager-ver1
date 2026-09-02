"""Evaluate on-device text classifier vs teacher under ASR noise (model_light §2-4 eval).

Models:
  - student: KoELECTRA-small-v3 fine-tuned (artifacts/koelectra_small_harm)
  - teacher: frozen e5-base + MLP head (artifacts/text_head.pt)

Protocol (deployment-aligned):
  - positives = harm rows of configs/text/harm_language_testset.jsonl, KOREAN subset
    (the on-device pipeline is Korean-only by construction: Moonshine-KR ASR).
  - negatives = kor_unsmile VALID clean (real-world benign Korean).
  - score = 1 - P(safe)  (any-harm suspicion).
  - metric = recall @ FPR15% (threshold set on the negatives; spec §9 text Stage-1).
  - ASR-noise injection at MEASURED CER levels {5, 20, 40}% (asr_cer_eval.py results:
    clean 5.6 / SNR10 20.7 / SNR5 38.9): jamo-level substitution/deletion + spacing errors,
    applied to BOTH positives and negatives (threshold re-fit per condition — the deployed
    system would calibrate on ASR-output text).

Honest limits: synthetic corruption approximates real ASR errors (confusions are not
Moonshine's actual error distribution); English rows excluded (server-side scope).
Env: CERS default "5,20,40", SEED 7.
"""
from __future__ import annotations
import json, os, sys, unicodedata
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
CATS = ["threat", "sexual", "gambling", "drug", "abuse", "safe"]
SAFE = CATS.index("safe")
CERS = [float(x) / 100 for x in os.environ.get("CERS", "5,20,40").split(",")]
SEED = int(os.environ.get("SEED", "7"))
FPR = 0.15

# ---------- hangul jamo corruption ----------
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
# phonetically-near substitution pools (coarse ASR-style confusions)
NEAR_CHO = {"ㄱ": "ㅋㄲ", "ㄷ": "ㅌㄸ", "ㅂ": "ㅍㅃ", "ㅈ": "ㅊㅉ", "ㅅ": "ㅆ", "ㄴ": "ㅁㄹ", "ㅁ": "ㄴㅂ", "ㄹ": "ㄴ"}
NEAR_JUNG = {"ㅏ": "ㅑㅓ", "ㅓ": "ㅏㅗ", "ㅗ": "ㅜㅓ", "ㅜ": "ㅗㅡ", "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅡ": "ㅜㅣ", "ㅣ": "ㅢㅔ"}


def _decomp(s):
    o = ord(s) - 0xAC00
    return CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]


def _comp(c, j, g):
    return chr(0xAC00 + CHO.index(c) * 588 + JUNG.index(j) * 28 + JONG.index(g))


def corrupt(text, cer, rng):
    """Apply jamo substitutions / syllable deletions / spacing errors at ~CER rate."""
    out = []
    for ch in text:
        if "가" <= ch <= "힣" and rng.random() < cer:
            op = rng.random()
            if op < 0.25:
                continue                      # deletion
            c, j, g = _decomp(ch)
            if op < 0.75:                     # initial/vowel confusion
                if rng.random() < 0.5 and c in NEAR_CHO:
                    c = NEAR_CHO[c][rng.integers(len(NEAR_CHO[c]))]
                elif j in NEAR_JUNG:
                    j = NEAR_JUNG[j][rng.integers(len(NEAR_JUNG[j]))]
            else:                             # final-consonant drop
                g = " "
            try:
                out.append(_comp(c, j, g))
            except ValueError:
                out.append(ch)
        elif ch == " " and rng.random() < cer:
            continue                          # spacing error (ASR joins words)
        else:
            out.append(ch)
    return "".join(out)


# ---------- data ----------

def load_eval():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    pos = []
    for line in (_ROOT / "configs/text/harm_language_testset.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("lang") == "ko" and r["label"] != "safe":
            pos.append(r["text"])
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/valid-00000-of-00001.parquet",
                        repo_type="dataset")
    dv = pd.read_parquet(p)
    neg = [str(s) for s in dv[dv["clean"] == 1]["문장"]]
    return pos, neg


# ---------- models ----------

class Student:
    name = "KoELECTRA-small(14M)"
    DIR = "artifacts/koelectra_small_harm"

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        d = _ROOT / self.DIR
        self.tok = AutoTokenizer.from_pretrained(d)
        self.m = AutoModelForSequenceClassification.from_pretrained(d).to("cuda").eval()

    @torch.no_grad()
    def scores(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            b = self.tok(texts[i:i + 64], truncation=True, max_length=128, padding=True,
                         return_tensors="pt").to("cuda")
            p = torch.softmax(self.m(**b).logits, -1)[:, SAFE]
            out.append((1 - p).cpu().numpy())
        return np.concatenate(out)


class Teacher:
    name = "e5-base+MLP(teacher,278M)"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch.nn as nn
        self.e5 = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
        ck = torch.load(_ROOT / "artifacts/text_head.pt", map_location="cpu", weights_only=False)
        d, h = 768, 256
        self.head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(h, len(CATS)))
        sd = ck["state"]
        self.head.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
        self.head.eval()

    @torch.no_grad()
    def scores(self, texts):
        X = self.e5.encode([f"query: {t}" for t in texts], normalize_embeddings=True,
                           batch_size=64, show_progress_bar=False)
        p = torch.softmax(self.head(torch.from_numpy(np.asarray(X, dtype=np.float32))), -1)[:, SAFE]
        return (1 - p).numpy()


def recall_at_fpr(pos_s, neg_s, fpr=FPR):
    thr = np.sort(neg_s)[::-1][max(0, int(np.floor(fpr * len(neg_s))) - 1)]
    return float((pos_s >= thr).mean()), float(thr)


def main():
    rng = np.random.default_rng(SEED)
    pos, neg = load_eval()
    print(f"[text-eval] pos(ko harm)={len(pos)}  neg(unsmile valid clean)={len(neg)}", flush=True)
    conds = {"clean": (pos, neg)}
    for cer in CERS:
        r2 = np.random.default_rng(SEED + int(cer * 100))
        conds[f"cer{int(cer*100)}"] = ([corrupt(t, cer, r2) for t in pos],
                                       [corrupt(t, cer, r2) for t in neg])
    class StudentAug(Student):
        name = "KoELECTRA-small+ASRaug(14M)"
        DIR = "artifacts/koelectra_small_harm_asraug"

    models = [Student, Teacher]
    if (_ROOT / StudentAug.DIR).exists():
        models.insert(1, StudentAug)
    results = {}
    for cls in models:
        m = cls()
        row = {}
        for tag, (P, N) in conds.items():
            r, thr = recall_at_fpr(m.scores(P), m.scores(N))
            row[tag] = round(r, 4)
        results[m.name] = row
        print(f"  {m.name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
        del m; torch.cuda.empty_cache()
    out = _ROOT / "data_dl/asr/text_asr_noise_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[text-eval] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
