"""On-device Korean ASR CER measurement (model_light.md §5-2, tracks a+b).

Candidates:
  - moonshine-tiny-ko (27M, UsefulSensors/moonshine-tiny-ko, HF transformers, GPU)
  - sherpa-onnx zipformer-korean int8 (offline transducer, CPU)
  - faster-whisper large-v3 (upper bound / server reference, GPU int8_float16)

Track (a) clean CER: Zeroth-Korean test split (public reference transcripts).
Track (b) noise robustness: same utterances mixed with OUR project noise clips
  (violence + confusable sounds) at SNR {10, 5, 0} dB -> CER degradation curve.
  This approximates the "harmful-situation" acoustic domain while keeping ground truth.

CER normalization: NFC, remove punctuation/symbols, remove ALL whitespace (Korean CER
convention) -> character error rate via jiwer.cer.

Env: N_CLEAN (default 200), N_NOISE (default 100), SNRS (default "10,5,0"), SEED,
     OUT (default data_dl/asr/cer_results.json).
"""
from __future__ import annotations
import json, os, re, sys, time, unicodedata
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")

N_CLEAN = int(os.environ.get("N_CLEAN", "200"))
N_NOISE = int(os.environ.get("N_NOISE", "100"))
SNRS = [float(s) for s in os.environ.get("SNRS", "10,5,0").split(",")]
SEED = int(os.environ.get("SEED", "42"))
OUT = os.environ.get("OUT", "data_dl/asr/cer_results.json")
SHERPA_DIR = Path("data_dl/asr/sherpa-onnx-zipformer-korean-2024-06-24")
SR = 16000

_punct = re.compile(r"[^가-힣a-zA-Z0-9]")  # keep hangul syllables + alnum


def norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    return _punct.sub("", t)


def cer(refs, hyps):
    import jiwer
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # skip empty refs
    if not pairs:
        return float("nan")
    return jiwer.cer([p[0] for p in pairs], [p[1] or "-" for p in pairs])


# ---------- data ----------

def load_zeroth(n, rng):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("kresnik/zeroth_korean", "data/test-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    idx = rng.permutation(len(df))[:n]
    items = []
    import io, soundfile as sf
    for i in idx:
        row = df.iloc[int(i)]
        au = row["audio"]
        wav, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
        if sr != SR:
            import torchaudio.functional as AF, torch
            wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
        items.append({"id": str(row.get("id", i)), "wav": wav, "text": row["text"]})
    return items


def load_noise_pool(rng, k=200):
    """Noise clips from OUR dataset: violence + confusables (the deployment soundscape)."""
    import combined_data as CD
    CD.VIOLENCE = "data_dl/manifests/violence_v2.jsonl"
    CD.GAMBLING = "data_dl/manifests/gambling.jsonl"
    from train_beats_finetune import _has_clip
    from preprocess.audio import load_audio
    tr, _, _ = CD.build_combined_records(exists_fn=_has_clip)
    # exclude vio_verbal (contains speech -> would corrupt reference transcript validity)
    pool = [r for r in tr if "vio_verbal" not in r.labels]
    picks = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    out = []
    for i in picks:
        try:
            w = load_audio(f"data_dl/clips/{pool[int(i)].clip_id}.wav", sample_rate=SR)
            if np.abs(w).max() > 1e-4:
                out.append(w.astype(np.float32))
        except Exception:
            pass
    return out


def mix_snr(speech, noise, snr_db, rng):
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    off = rng.integers(0, len(noise) - len(speech) + 1)
    noise = noise[off:off + len(speech)]
    ps = np.mean(speech ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise
    peak = np.abs(mixed).max()
    return (mixed / peak * 0.95).astype(np.float32) if peak > 1 else mixed.astype(np.float32)


# ---------- models ----------

class Moonshine:
    name = "moonshine-tiny-ko(27M)"
    model_id = "UsefulSensors/moonshine-tiny-ko"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(self.model_id)
        self.m = MoonshineForConditionalGeneration.from_pretrained(
            self.model_id).to("cuda").eval()
        self.torch = torch

    def transcribe(self, wav):
        t = self.torch
        with t.no_grad():
            inp = self.proc(wav, sampling_rate=SR, return_tensors="pt").to("cuda")
            # generous cap — the 6.5 tok/s English convention TRUNCATES Korean (verified:
            # sentences cut mid-way). Korean BPE needs far more tokens per second.
            max_new = max(32, int(len(wav) / SR * 20))
            ids = self.m.generate(**inp, max_new_tokens=max_new)
            return self.proc.decode(ids[0], skip_special_tokens=True)


class Zipformer:
    name = "sherpa-zipformer-ko-int8(~74M)"

    def __init__(self):
        import sherpa_onnx
        d = SHERPA_DIR
        enc = next(d.glob("encoder-*.int8.onnx"))
        dec = next(d.glob("decoder-*.onnx"))
        joi = next(d.glob("joiner-*.int8.onnx"))
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(enc), decoder=str(dec), joiner=str(joi),
            tokens=str(d / "tokens.txt"), num_threads=4)

    def transcribe(self, wav):
        s = self.rec.create_stream()
        s.accept_waveform(SR, wav)
        self.rec.decode_stream(s)
        return s.result.text


class WhisperLarge:
    name = "faster-whisper-large-v3(ref)"

    def __init__(self):
        from faster_whisper import WhisperModel
        self.m = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

    def transcribe(self, wav):
        segs, _ = self.m.transcribe(wav, language="ko", beam_size=5, vad_filter=False)
        return "".join(s.text for s in segs)


def run_set(model, items, tag):
    hyps, t0 = [], time.time()
    for it in items:
        try:
            hyps.append(model.transcribe(it["wav"]))
        except Exception as e:
            print(f"  [warn] {model.name} failed on {it['id']}: {type(e).__name__}", flush=True)
            hyps.append("")
    dt = time.time() - t0
    c = cer([it["text"] for it in items], hyps)
    dur = sum(len(it["wav"]) for it in items) / SR
    print(f"  {model.name:34s} {tag:12s} CER={c:.4f}  ({dt:.0f}s for {dur:.0f}s audio, "
          f"RTF={dt/max(dur,1e-9):.2f})", flush=True)
    return {"cer": c, "rtf": dt / max(dur, 1e-9), "n": len(items)}


def main():
    rng = np.random.default_rng(SEED)
    print(f"[asr-cer] loading Zeroth test (n={N_CLEAN}) …", flush=True)
    clean = load_zeroth(N_CLEAN, rng)
    print(f"[asr-cer] loading noise pool from our clips …", flush=True)
    noise = load_noise_pool(rng)
    print(f"[asr-cer] {len(clean)} utts, {len(noise)} noise clips", flush=True)

    noisy_sets = {}
    sub = clean[:N_NOISE]
    for snr in SNRS:
        ns = []
        for it in sub:
            nz = noise[rng.integers(len(noise))]
            ns.append({**it, "wav": mix_snr(it["wav"], nz, snr, rng)})
        noisy_sets[snr] = ns

    # MODELS lets a run compare Moonshine sizes only (e.g. MODELS="moonshine:tiny,moonshine:base")
    # instead of re-measuring the rejected/reference systems every time.
    spec = os.environ.get("MODELS", "")
    if spec:
        classes = []
        for s in (x.strip() for x in spec.split(",") if x.strip()):
            if s.startswith("moonshine:"):
                size = s.split(":", 1)[1]
                mid = f"UsefulSensors/moonshine-{size}-ko"
                classes.append(type(f"Moonshine_{size}", (Moonshine,),
                                    {"name": f"moonshine-{size}-ko", "model_id": mid}))
            elif s == "zipformer":
                classes.append(Zipformer)
            elif s == "whisper":
                classes.append(WhisperLarge)
            else:
                raise SystemExit(f"unknown MODELS entry: {s}")
    else:
        classes = [Moonshine, Zipformer, WhisperLarge]

    results = {}
    for cls in classes:
        print(f"[asr-cer] init {cls.name} …", flush=True)
        try:
            m = cls()
        except Exception as e:
            print(f"  [skip] {cls.name}: {type(e).__name__}: {e}", flush=True)
            continue
        r = {"clean": run_set(m, clean, "clean")}
        for snr, ns in noisy_sets.items():
            r[f"snr{int(snr)}"] = run_set(m, ns, f"SNR{int(snr)}dB")
        results[m.name] = r
        del m
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[asr-cer] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
