"""On-device text-harm classifier — fine-tune KoELECTRA-small-v3 (14M) (model_light §2-4).

Training data replicates scripts/train_text_head.py EXACTLY (fair comparison with the
e5+MLP teacher): harm_train_corpus.jsonl (synthetic, eval-leakage-excluded) + kor_unsmile
TRAIN clean (3000 real negatives) + unsmile toxic split into threat(violence marker)/
abuse(strong slur), 6-way {threat,sexual,gambling,drug,abuse,safe}, class-weighted CE.

Full fine-tune (14M is cheap). Saves artifacts/koelectra_small_harm/ (HF format).

Env: EPOCHS(4), BATCH(32), LR(3e-5), SEED(1234 — same as teacher).
  ASR_AUG=1   duplicate the corpus with SYNTHETIC jamo corruption   -> _asraug
  SLANG=1     add configs/text/slang_train_corpus.jsonl (Korean 은어 the base corpus lacks)
  ASR_REAL=1  add configs/text/asr_corrupted_corpus.jsonl (text that went through TTS ->
              noise -> real Moonshine, i.e. the actual error distribution rather than a
              simulated one; built by .autorun/make_asr_corrupted_corpus.py)
  SLANG+ASR_REAL                                                    -> _slang
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
CATS = ["threat", "sexual", "gambling", "drug", "abuse", "safe"]
CAT_IDX = {c: i for i, c in enumerate(CATS)}
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = int(os.environ.get("BATCH", "32"))
LR = float(os.environ.get("LR", "3e-5"))
SEED = int(os.environ.get("SEED", "1234"))
MODEL_ID = "monologg/koelectra-small-v3-discriminator"
OUT = _ROOT / "artifacts/koelectra_small_harm"


def build_training_set():
    """Same recipe as scripts/train_text_head.py main() — kept in sync manually."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rng = np.random.default_rng(SEED)
    eval_texts = set()
    for f in ["configs/text/harm_semantic_eval.jsonl", "configs/text/harm_language_testset.jsonl"]:
        for line in (_ROOT / f).read_text().splitlines():
            if line.strip():
                eval_texts.add(json.loads(line)["text"].replace(" ", ""))
    texts, labels = [], []
    for line in (_ROOT / "configs/text/harm_train_corpus.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["text"].replace(" ", "") in eval_texts:
            continue
        texts.append(r["text"]); labels.append(CAT_IDX[r["label"]])
    vpat = re.compile(r"죽여|죽인|죽어|총살|쏴|쏘|칼|패버|때려|후려|숨통|끝장|없애|족쳐|"
                      r"파묻|린치|목졸|찔러|살인|테러|폭행|구타|주먹|팰|뚝배기|대가리")
    strong = re.compile(r"씨발|시발|병신|ㅄ|ㅂㅅ|개새끼|개새|개색|좆|존나게|지랄|썅|또라이|미친놈|"
                        r"미친새|니미|느금|엄창|창녀|걸레|후장|보지|자지|꺼져라|닥쳐라|엿먹어")
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/train-00000-of-00001.parquet",
                        repo_type="dataset")
    dftr = pd.read_parquet(p)
    clean = [str(s) for s in dftr[dftr["clean"] == 1]["문장"]]
    rng.shuffle(clean)
    for s in clean[:3000]:
        texts.append(s); labels.append(CAT_IDX["safe"])
    for s in (str(x) for x in dftr[dftr["악플/욕설"] == 1]["문장"]):
        if vpat.search(s):
            texts.append(s); labels.append(CAT_IDX["threat"])
        elif strong.search(s):
            texts.append(s); labels.append(CAT_IDX["abuse"])
    return texts, np.array(labels)


def _add_jsonl(texts: list[str], labels: list[int], path: Path, eval_texts: set[str]) -> int:
    """Append rows of a {text,label} jsonl, skipping anything that appears in an eval set."""
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"] not in CAT_IDX or r["text"].replace(" ", "") in eval_texts:
            continue
        texts.append(r["text"]); labels.append(CAT_IDX[r["label"]]); n += 1
    return n


def _eval_keys() -> set[str]:
    keys = set()
    for f in ("configs/text/harm_semantic_eval.jsonl",
              "configs/text/harm_language_testset.jsonl",
              "configs/text/profanity_slang_testset.jsonl"):
        p = _ROOT / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    keys.add(json.loads(line)["text"].replace(" ", ""))
    return keys


def main():
    torch.manual_seed(SEED)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts, y = build_training_set()
    extra_tag = ""
    if os.environ.get("SLANG") == "1" or os.environ.get("ASR_REAL") == "1":
        tl, yl = list(texts), list(y)
        keys = _eval_keys()
        if os.environ.get("SLANG") == "1":
            n = _add_jsonl(tl, yl, _ROOT / "configs/text/slang_train_corpus.jsonl", keys)
            print(f"[koelectra] +{n} slang rows", flush=True)
        if os.environ.get("ASR_REAL") == "1":
            p = _ROOT / "configs/text/asr_corrupted_corpus.jsonl"
            if not p.exists():
                raise SystemExit(f"{p} missing — run .autorun/make_asr_corrupted_corpus.py")
            n = _add_jsonl(tl, yl, p, keys)
            print(f"[koelectra] +{n} real-ASR-corrupted rows", flush=True)
        texts, y = tl, np.array(yl)
        extra_tag = "_slang"
    if os.environ.get("ASR_AUG") == "1":
        # ASR-noise augmentation: duplicate corpus with jamo corruption at CER~U(5,40)%
        # (same corruptor as the eval, but train uses its own rng -> no leakage of eval seeds)
        sys.path.insert(0, str(_ROOT / ".autorun"))
        import importlib.util as _il
        _s = _il.spec_from_file_location("ev", _ROOT / ".autorun/eval_text_asr_noise.py")
        _ev = _il.module_from_spec(_s); _s.loader.exec_module(_ev)
        rng_a = np.random.default_rng(SEED + 999)
        aug = [_ev.corrupt(t, rng_a.uniform(0.05, 0.4), rng_a) for t in texts]
        texts = texts + aug
        y = np.concatenate([y, y])
        global OUT
        OUT = _ROOT / f"artifacts/koelectra_small_harm_asraug{extra_tag}"
        print(f"[koelectra] ASR_AUG on -> {len(texts)} samples, OUT={OUT.name}", flush=True)
    elif extra_tag:
        OUT = _ROOT / f"artifacts/koelectra_small_harm{extra_tag}"
    print(f"[koelectra] train {len(texts)}  dist={np.bincount(y, minlength=6).tolist()} = {CATS}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(CATS)).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[koelectra] {MODEL_ID} -> {n_par:.1f}M params, device {device}", flush=True)

    counts = np.bincount(y, minlength=len(CATS)).astype(np.float32)
    w = torch.from_numpy(counts.sum() / (len(CATS) * np.maximum(counts, 1))).float().to(device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    enc = tok(texts, truncation=True, max_length=128, padding=False)["input_ids"]

    idx = np.arange(len(texts))
    rng = np.random.default_rng(SEED)
    steps = 0
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(idx); tot = 0.0; nb = 0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            batch = tok.pad({"input_ids": [enc[j] for j in b]}, return_tensors="pt").to(device)
            out = model(**batch)
            loss = lossf(out.logits, torch.from_numpy(y[b]).long().to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1; steps += 1
        print(f"[koelectra] ep{ep} loss={tot/nb:.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    (OUT / "cats.json").write_text(json.dumps(CATS))
    print(f"[koelectra] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
