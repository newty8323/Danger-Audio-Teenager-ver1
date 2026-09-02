"""Pick the ASR for the app by the metric that matters: does the HARMFUL WORD survive?

CER on read speech (asr_cer_eval.py) picked moonshine-base-ko, but on the profanity/slang set
base-ko returns the literal string "audiotext" for some utterances — including
"야 이 씨발놈아 당장 나와", which tiny-ko transcribes fine. CER cannot see that failure mode,
so this script scores candidates on:

  keyword survival — is the harm-bearing token still in the transcript?
  recall@FPR15     — does the deployed int8 classifier fire? (threshold from unsmile clean)
  artifact rate    — how often the model returns junk/nothing instead of a transcript
  RTF (CPU)        — is it affordable on the client

Candidates are given as MODELS="moonshine:tiny,moonshine:base,whisper:small,
whisper:large-v3-turbo". Whisper runs through faster-whisper (int8 on CPU, int8_float16 on
CUDA). Audio is the TTS profanity/slang set built by eval_profanity_slang.py.

Env: MODELS (see above), SNRS "10" (clean always run), SEED 13, N_NEG 300, DEVICE auto,
     OUT data_dl/asr/asr_harm_comparison.json
Run: uv run --group nlp --group asr python .autorun/compare_asr_harm.py
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

MODELS = os.environ.get("MODELS", "moonshine:tiny,moonshine:base,whisper:large-v3-turbo")
SNRS = [float(s) for s in os.environ.get("SNRS", "10").split(",") if s.strip()]
SEED = int(os.environ.get("SEED", "13"))
N_NEG = int(os.environ.get("N_NEG", "300"))
OUT = os.environ.get("OUT", "data_dl/asr/asr_harm_comparison.json")
SR = 16000
TTS_DIR = Path("data_dl/asr/prof_tts")

ARTIFACTS = {"audiotext", "audio text", ""}


class MoonshineBackend:
    def __init__(self, size: str, device: str | None):
        from cascade.pipeline import MoonshineASR
        self.name = f"moonshine-{size}-ko"
        self.asr = MoonshineASR(device, model_id=f"UsefulSensors/moonshine-{size}-ko")
        self.params = sum(p.numel() for p in self.asr.m.parameters()) / 1e6

    def transcribe(self, wav):
        return self.asr.transcribe(wav)

    def close(self):
        del self.asr


class WhisperBackend:
    """faster-whisper (CTranslate2). NOTE: CUDA needs cuBLAS/cuDNN installed separately from
    torch's bundled copies — when they are missing it raises "libcublas.so.12 is not found",
    so WHISPER_DEVICE=cpu is the portable choice (int8, and the client is a laptop anyway)."""

    def __init__(self, size: str, device: str | None):
        from faster_whisper import WhisperModel
        dev = device or os.environ.get("WHISPER_DEVICE", "cpu")
        ct = "int8_float16" if dev == "cuda" else "int8"
        self.name = f"faster-whisper-{size}({ct})"
        self.m = WhisperModel(size, device=dev, compute_type=ct)
        self.params = float("nan")

    def transcribe(self, wav):
        segs, _ = self.m.transcribe(wav, language="ko", beam_size=5, vad_filter=False)
        return "".join(s.text for s in segs).strip()

    def close(self):
        del self.m


def make_backend(spec: str, device: str | None):
    kind, _, size = spec.partition(":")
    if kind == "moonshine":
        return MoonshineBackend(size, device)
    if kind == "whisper":
        return WhisperBackend(size, device)
    raise SystemExit(f"unknown backend: {spec}")


def main():
    import soundfile as sf
    from eval_e2e_text import load_noise_pool, mix_snr
    from eval_profanity_slang import (
        KEYWORDS,
        load_negatives,
        load_rows,
        survived,
        thr_at_fpr,
    )

    from cascade.pipeline import TextScorer

    assert KEYWORDS, "keyword table missing"
    rng = np.random.default_rng(SEED)
    rows = load_rows()
    paths = [TTS_DIR / f"row_{i:03d}.wav" for i in range(len(rows))]
    if not all(p.exists() for p in paths):
        raise SystemExit(f"missing TTS audio in {TTS_DIR} — "
                         "run .autorun/eval_profanity_slang.py first")

    scorer = TextScorer()
    thr = thr_at_fpr(scorer.score(load_negatives(rng)))
    noise = load_noise_pool(rng, k=80)
    print(f"[cmp] {len(rows)} rows · classifier threshold {thr:.3f} @FPR15", flush=True)

    audio = {"clean": [sf.read(p, dtype="float32")[0] for p in paths]}
    for snr in SNRS:
        crng = np.random.default_rng(SEED + int(snr))
        audio[f"snr{int(snr)}"] = [
            mix_snr(w, noise[int(crng.integers(len(noise)))], snr, crng)
            for w in audio["clean"]]

    results = {}
    for spec in [s.strip() for s in MODELS.split(",") if s.strip()]:
        try:
            be = make_backend(spec, None)
        except Exception as e:
            print(f"  [skip] {spec}: {type(e).__name__}: {e}", flush=True)
            continue
        entry = {"params_M": None if be.params != be.params else round(be.params, 1)}
        for cond, wavs in audio.items():
            t0 = time.time()
            hyps = [be.transcribe(w) for w in wavs]
            dt = time.time() - t0
            dur = sum(len(w) for w in wavs) / SR
            s = scorer.score([h or "-" for h in hyps])
            surv = [survived(r, h) for r, h in zip(rows, hyps, strict=True)]
            judged = [x for x in surv if x is not None]
            art = float(np.mean([h.strip().strip(".!?").lower() in ARTIFACTS for h in hyps]))
            e = {"keyword_survival": round(float(np.mean(judged)), 3) if judged else None,
                 "recall@fpr15": round(float((s >= thr).mean()), 3),
                 "artifact_rate": round(art, 3), "rtf": round(dt / dur, 3)}
            for kind in ("profanity", "slang"):
                m = np.array([r["kind"] == kind for r in rows])
                e[f"recall_{kind}"] = round(float((s[m] >= thr).mean()), 3)
            entry[cond] = e
            print(f"  {be.name:34s} {cond:6s} survival={e['keyword_survival']} "
                  f"recall={e['recall@fpr15']:.3f} (prof {e['recall_profanity']:.3f} / "
                  f"slang {e['recall_slang']:.3f}) artifact={e['artifact_rate']:.2f} "
                  f"RTF={e['rtf']:.3f}", flush=True)
            if cond == "clean":
                entry["samples"] = [{"ref": r["text"], "hyp": h}
                                    for r, h in zip(rows[:8], hyps[:8], strict=False)]
        results[be.name] = entry
        be.close()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    out = _ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threshold": thr, "models": results}, ensure_ascii=False,
                              indent=1))
    print(f"[cmp] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
