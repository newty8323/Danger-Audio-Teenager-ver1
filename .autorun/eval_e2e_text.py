"""End-to-end validation of the language branch — model_light.md §2-4 condition ③.

Closes the caveat that the ASR-noise gain was measured with SYNTHETIC jamo corruption
sharing the augmentation's error model. Here the corruption comes from the real ASR:

  Korean text  --MMS-TTS-kor-->  speech  --mix project noise @SNR-->  waveform
               --Moonshine-tiny-ko-->  transcript  --KoELECTRA-small int8-->  score

Reported per condition (clean / SNR10 / SNR5 / SNR0):
  - real CER of Moonshine on these utterances (comparable to asr_cer_eval.py's Zeroth CER)
  - recall @FPR15% with the threshold RE-FIT on that condition's negatives
    (comparable to eval_text_asr_noise.py's synthetic-corruption numbers)
  - recall/FPR at the FIXED deployed threshold from artifacts/cascade_thresholds.json

Positives: ko harm rows of configs/text/harm_language_testset.jsonl.
Negatives: kor_unsmile valid clean (subsampled, N_NEG).
Noise: our violence/confusable clips, vio_verbal excluded (would add real speech).

Honest limits: TTS speech is cleaner and more canonical than spontaneous speech, so CER
here is optimistic vs a real user; single TTS voice = no speaker variation.

Env: N_NEG 300, SNRS "10,5,0", SEED 11, TTS_BATCH 8,
     OUT data_dl/asr/e2e_text_results.json, SAVE_SAMPLES 1
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts", _ROOT / ".autorun"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")

N_NEG = int(os.environ.get("N_NEG", "300"))
N_CTRL = int(os.environ.get("N_CTRL", "100"))
SNRS = [float(s) for s in os.environ.get("SNRS", "10,5,0").split(",")]
SEED = int(os.environ.get("SEED", "11"))
FPR = 0.15
SR = 16000
OUT = os.environ.get("OUT", "data_dl/asr/e2e_text_results.json")
SAVE_SAMPLES = os.environ.get("SAVE_SAMPLES", "1") == "1"

_keep = re.compile(r"[^가-힣a-zA-Z0-9]")


def norm(t: str) -> str:
    return _keep.sub("", unicodedata.normalize("NFC", t))


def cer(refs, hyps):
    import jiwer
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return float("nan")
    return float(jiwer.cer([r for r, _ in pairs], [h or "-" for _, h in pairs]))


def thr_at_fpr(neg: np.ndarray, fpr: float = FPR) -> float:
    k = int(np.floor(fpr * len(neg)))
    srt = np.sort(neg)[::-1]
    return float(srt[max(0, k - 1)]) if k >= 1 else float(srt[0] + 1e-6)


# ---------- data ----------

def load_texts(rng):
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
    neg_all = [str(s) for s in pd.read_parquet(p).query("clean == 1")["문장"]]
    idx = rng.permutation(len(neg_all))[:N_NEG]
    return pos, [neg_all[int(i)] for i in idx]


def load_control_texts(rng):
    """Zeroth-Korean test transcripts (text column only — the audio bytes stay unread)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("kresnik/zeroth_korean", "data/test-00000-of-00001.parquet",
                        repo_type="dataset")
    texts = pq.read_table(p, columns=["text"])["text"].to_pylist()
    idx = rng.permutation(len(texts))[:N_CTRL]
    return [str(texts[int(i)]) for i in idx]


def load_noise_pool(rng, k=120):
    import combined_data as CD
    CD.VIOLENCE = "data_dl/manifests/violence_v2.jsonl"
    CD.GAMBLING = "data_dl/manifests/gambling.jsonl"
    from train_beats_finetune import _has_clip

    from preprocess.audio import load_audio
    tr, _, _ = CD.build_combined_records(exists_fn=_has_clip)
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
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    off = int(rng.integers(0, len(noise) - len(speech) + 1))
    noise = noise[off:off + len(speech)]
    ps = float(np.mean(speech ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    mixed = speech + np.sqrt(ps / (pn * 10 ** (snr_db / 10))) * noise
    peak = float(np.abs(mixed).max())
    return (mixed / peak * 0.95).astype(np.float32) if peak > 1 else mixed.astype(np.float32)


# ---------- TTS ----------

class TTS:
    """facebook/mms-tts-kor (VITS, 36M, 16 kHz).

    MMS models are trained on uroman-ROMANIZED text and their tokenizer drops Hangul
    entirely (vocab is 26 Latin symbols) — feeding Hangul yields an empty sequence and
    crashes in padding. Romanizing first is the documented MMS path, not a workaround.
    """

    def __init__(self, device="cuda"):
        import uroman as ur
        from transformers import AutoTokenizer, VitsModel
        self.u = ur.Uroman()
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
        self.m = VitsModel.from_pretrained("facebook/mms-tts-kor").to(device).eval()
        self.device = device
        self.sr = self.m.config.sampling_rate

    @torch.no_grad()
    def say(self, text: str) -> np.ndarray:
        b = self.tok(self.u.romanize_string(text), return_tensors="pt").to(self.device)
        if b["input_ids"].shape[-1] == 0:
            raise ValueError("empty token sequence after romanization")
        w = self.m(**b).waveform[0].float().cpu().numpy()
        if self.sr != SR:
            import torchaudio.functional as AF
            w = AF.resample(torch.from_numpy(w), self.sr, SR).numpy()
        peak = float(np.abs(w).max())
        return (w / peak * 0.9).astype(np.float32) if peak > 0 else w.astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pos_t, neg_t = load_texts(rng)
    print(f"[e2e] texts: pos {len(pos_t)}  neg {len(neg_t)}", flush=True)
    noise = load_noise_pool(rng)
    print(f"[e2e] noise clips {len(noise)}", flush=True)

    # --- 1) synthesize once, keep on disk (memory: never hold the whole corpus in RAM) ---
    import soundfile as sf
    wav_dir = Path("data_dl/asr/e2e_tts")
    wav_dir.mkdir(parents=True, exist_ok=True)
    tts = TTS(dev)
    items = []  # (path, ref_text, is_pos)
    t0 = time.time()
    for grp, texts in (("pos", pos_t), ("neg", neg_t)):
        for i, t in enumerate(texts):
            p = wav_dir / f"{grp}_{i:04d}.wav"
            if not p.exists():
                try:
                    sf.write(p, tts.say(t), SR)
                except Exception as e:
                    print(f"  [warn] TTS failed {grp}_{i}: {type(e).__name__}", flush=True)
                    continue
            items.append((p, t, grp == "pos"))
    # CONTROL for the TTS domain gap: same Zeroth sentences whose HUMAN-read CER we already
    # measured (asr_cer_eval.py: clean 5.6%). TTS-clean CER minus 5.6% = the synthetic-voice
    # bias, so the numbers below can be read against the real-speech baseline.
    ctrl = []
    if N_CTRL:
        ctrl_dir = wav_dir / "control"
        ctrl_dir.mkdir(exist_ok=True)
        for i, t in enumerate(load_control_texts(rng)):
            p = ctrl_dir / f"ctrl_{i:04d}.wav"
            if not p.exists():
                try:
                    sf.write(p, tts.say(t), SR)
                except Exception:
                    continue
            ctrl.append((p, t))
    del tts
    torch.cuda.empty_cache()
    print(f"[e2e] TTS done: {len(items)} utts (+{len(ctrl)} control) in {time.time()-t0:.0f}s "
          f"-> {wav_dir}", flush=True)

    # --- 2) ASR + text scoring per condition ---
    from cascade.pipeline import MoonshineASR, TextScorer
    asr = MoonshineASR(dev)
    scorer = TextScorer()          # int8 CPU, the deployed configuration
    conds = ["clean"] + [f"snr{int(s)}" for s in SNRS]
    results, samples = {}, {}

    if ctrl:
        chyps = [asr.transcribe(sf.read(p, dtype="float32")[0]) for p, _ in ctrl]
        ccer = cer([t for _, t in ctrl], chyps)
        results["_control_tts_zeroth_clean"] = {
            "cer": round(ccer, 4), "n": len(ctrl),
            "human_read_cer_reference": 0.056,
            "tts_bias": round(ccer - 0.056, 4),
            "note": "same corpus as asr_cer_eval.py clean (human-read CER 5.6%); the delta "
                    "is the synthetic-voice bias in every number below"}
        print(f"  control TTS-Zeroth CER={ccer:.3f} vs human-read 0.056 "
              f"(TTS bias {ccer-0.056:+.3f})", flush=True)

    for cond in conds:
        snr = None if cond == "clean" else float(cond[3:])
        refs, hyps, is_pos = [], [], []
        crng = np.random.default_rng(SEED + (0 if snr is None else int(snr) + 100))
        t0 = time.time()
        for p, ref, ip in items:
            w, _ = sf.read(p, dtype="float32")
            if snr is not None:
                w = mix_snr(w, noise[int(crng.integers(len(noise)))], snr, crng)
            hyps.append(asr.transcribe(w))
            refs.append(ref)
            is_pos.append(ip)
            del w
        is_pos = np.asarray(is_pos)
        c = cer(refs, hyps)
        s = scorer.score(hyps)
        pos_s, neg_s = s[is_pos], s[~is_pos]
        thr = thr_at_fpr(neg_s)
        results[cond] = {"cer": round(c, 4),
                         "recall@fpr15_refit": round(float((pos_s >= thr).mean()), 4),
                         "thr_refit": round(thr, 4),
                         "empty_hyp_rate": round(float(np.mean([not h.strip() for h in hyps])), 4),
                         "n_pos": int(is_pos.sum()), "n_neg": int((~is_pos).sum()),
                         "asr_sec": round(time.time() - t0, 1)}
        samples[cond] = [{"ref": refs[i], "hyp": hyps[i]} for i in range(min(5, len(refs)))]
        r = results[cond]
        print(f"  {cond:6s} CER={c:.3f}  recall@FPR15(refit)={r['recall@fpr15_refit']:.3f}"
              f"  empty={r['empty_hyp_rate']:.2f}  ({r['asr_sec']:.0f}s)", flush=True)

        # fixed deployed threshold (if the cascade artifact exists)
        tp = _ROOT / "artifacts/cascade_thresholds.json"
        if tp.exists():
            from cascade.decision import load_thresholds
            ft = load_thresholds(tp).text
            results[cond]["recall@fixed_thr"] = round(float((pos_s >= ft).mean()), 4)
            results[cond]["fpr@fixed_thr"] = round(float((neg_s >= ft).mean()), 4)
            results[cond]["fixed_thr"] = round(ft, 4)
            print(f"         fixed thr {ft:.3f}: recall {results[cond]['recall@fixed_thr']:.3f}"
                  f"  FPR {results[cond]['fpr@fixed_thr']:.3f}", flush=True)

    out = {"results": results, "samples": samples if SAVE_SAMPLES else {},
           "setup": {"tts": "facebook/mms-tts-kor", "asr": "UsefulSensors/moonshine-tiny-ko",
                     "text": "artifacts/koelectra_small_harm_asraug int8",
                     "n_utts": len(items), "seed": SEED,
                     "note": "TTS speech is cleaner than spontaneous speech -> CER optimistic"}}
    Path(_ROOT / OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(_ROOT / OUT).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[e2e] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
