"""Cascade assembly: fit thresholds on VAL, evaluate the assembled cascade on TEST.

Acoustic path only (the text branch is validated end-to-end by eval_e2e_text.py; its
threshold is fit here from clean Korean text so the artifact is complete).

Phase 1 (fit, VAL split):
  gate threshold   -> smallest threshold keeping >= GATE_RECALL of val any-vio positives
                      (tier-1 must not lose recall; its job is duty-cycle saving)
  acoustic thresh. -> FPR = ACOUSTIC_FPR on val negatives (spec §9 operating point)
  text threshold   -> FPR = TEXT_FPR on kor_unsmile valid-clean negatives
  -> artifacts/cascade_thresholds.json (versioned, with fit provenance)

Phase 2 (eval, TEST split):
  - CED-mini alone @ its threshold   (reference)
  - gate -> CED-mini cascade         (deployed path)
  reports recall / FPR / wake-rate (duty-cycle saving) + per-clip CPU latency.

Run from the repo root with the nlp group available (transformers/pandas/pyarrow):
    uv run --group nlp python .autorun/cascade_offline.py

Env: GATE_RECALL 0.98, ACOUSTIC_FPR 0.05, TEXT_FPR 0.15, LIMIT 0, LAT_N 30,
     NO_TEXT (skip text threshold fit), OUT artifacts/cascade_thresholds.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts", _ROOT / ".autorun", _ROOT / "distill"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")

import combined_data as CD  # noqa: E402

CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

from torch.utils.data import DataLoader  # noqa: E402
from train_beats_finetune import RawAudioDataset, _has_clip  # noqa: E402
from train_ced_vio import _strip  # noqa: E402

from cascade.decision import Thresholds, decide, save_thresholds  # noqa: E402
from cascade.pipeline import load_gate, load_trigger  # noqa: E402
from datasets.taxonomy import load_taxonomy  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402

GATE_RECALL = float(os.environ.get("GATE_RECALL", "0.98"))
ACOUSTIC_FPR = float(os.environ.get("ACOUSTIC_FPR", "0.05"))
TEXT_FPR = float(os.environ.get("TEXT_FPR", "0.15"))
LIMIT = int(os.environ.get("LIMIT", "0"))
LAT_N = int(os.environ.get("LAT_N", "30"))
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
OUT = os.environ.get("OUT", "artifacts/cascade_thresholds.json")
REPORT = os.environ.get("REPORT", "data_dl/artifacts/cascade_eval.json")


def _has_both(cid):  # identical filter to train_ced_vio -> same test set as probs npz
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


@torch.no_grad()
def _scores(model, records, tax, cfg_pp, tag, batch=1):
    """Per-clip any-vio score (max over the 4 violence sigmoids) + labels. CPU.

    batch=1 on purpose: int8 dynamic quantization derives activation scales PER BATCH, so
    the same clip scores differently at batch 8 vs batch 1 (measured up to 0.048 — same
    order as the int8-vs-fp32 error itself). Thresholds must be fit in the shape the device
    actually runs (one clip at a time).
    """
    loader = DataLoader(RawAudioDataset(records, tax, cfg_pp, train=False), batch_size=batch)
    S, Y = [], []
    t0 = time.time()
    for x, y in loader:
        p = torch.sigmoid(model(x, return_projection=False)["logits"]).float().numpy()
        S.append(p.max(1))
        Y.append(np.asarray(y).max(1))
    s, y = np.concatenate(S), np.concatenate(Y).astype(int)
    dt = time.time() - t0
    print(f"  [{tag}] {len(s)} clips in {dt:.0f}s ({dt/max(len(s),1)*1000:.0f} ms/clip, "
          f"batch={batch})", flush=True)
    return s, y


def thr_at_fpr(neg: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on `neg` is <= fpr.

    Ties matter after quantization (scores saturate at identical values), so the k-th
    score is not always a valid threshold: if it repeats, `>= thr` admits more than k
    negatives. Step up to the next strictly-greater score in that case.
    """
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(fpr * len(neg)))
    srt = np.sort(neg)[::-1]
    if k < 1:
        return float(srt[0] + 1e-6)
    thr = float(srt[k - 1])
    if (neg >= thr).sum() > k:                       # tie spilling over the budget
        higher = srt[srt > thr]
        thr = float(higher[-1]) if len(higher) else float(srt[0] + 1e-6)
    return thr


def thr_at_recall(pos: np.ndarray, recall: float) -> float:
    """Largest threshold keeping >= `recall` of positives (high-recall gate)."""
    if len(pos) == 0:
        return 0.0
    srt = np.sort(pos)[::-1]
    k = min(len(pos) - 1, max(0, int(np.ceil(recall * len(pos))) - 1))
    return float(srt[k])


def _fit_text_threshold():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    # HybridTextScorer, not TextScorer: the app scores max(classifier, lexicon), so the
    # operating point has to be fitted on the same quantity it will be compared against.
    from cascade.pipeline import HybridTextScorer
    p = hf_hub_download("smilegate-ai/kor_unsmile", "data/valid-00000-of-00001.parquet",
                        repo_type="dataset")
    neg = [str(s) for s in pd.read_parquet(p).query("clean == 1")["문장"]]
    ts = HybridTextScorer()
    s = ts.score(neg)
    thr = thr_at_fpr(s, TEXT_FPR)
    print(f"  [text] {len(neg)} clean negatives -> thr {thr:.4f} @FPR{TEXT_FPR:.0%}", flush=True)
    del ts
    return thr, len(neg)


def _metrics(s, y, thr):
    pos, neg = s[y == 1], s[y == 0]
    return {"recall": float((pos >= thr).mean()), "fpr": float((neg >= thr).mean()),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


@torch.no_grad()
def _latency(gate, trigger, records, tax, cfg_pp, n):
    """Single-clip CPU latency (deployment shape: batch 1)."""
    ds = RawAudioDataset(records[:n], tax, cfg_pp, train=False)
    out = {}
    for name, m in (("gate", gate), ("trigger", trigger)):
        if m is None:
            continue
        xs = [ds[i][0].unsqueeze(0) for i in range(min(n, len(ds)))]
        m(xs[0], return_projection=False)  # warm-up
        t0 = time.time()
        for x in xs:
            m(x, return_projection=False)
        out[name] = (time.time() - t0) / len(xs) * 1000
    return out


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    tax = load_taxonomy(str(_ROOT / "configs/data/classes_vio.yaml"))
    cfg_pp = PreprocessConfig()
    _, va, te = CD.build_combined_records(exists_fn=_has_both)
    va, te = _strip(va, tax), _strip(te, tax)
    if LIMIT:
        va, te = va[:LIMIT], te[:LIMIT]
    print(f"[cascade] val {len(va)} (pos {sum(1 for r in va if r.labels)}) | "
          f"test {len(te)} (pos {sum(1 for r in te if r.labels)})", flush=True)

    print("[cascade] loading gate (student s1) + trigger (CED-mini int8) …", flush=True)
    gate = load_gate("s1")
    trigger = load_trigger(int8=True)

    print("[cascade] --- phase 1: fit thresholds on VAL ---", flush=True)
    g_va, y_va = _scores(gate, va, tax, cfg_pp, "gate/val")
    a_va, _ = _scores(trigger, va, tax, cfg_pp, "trigger/val")
    t_gate = thr_at_recall(g_va[y_va == 1], GATE_RECALL)
    t_ac = thr_at_fpr(a_va[y_va == 0], ACOUSTIC_FPR)
    print(f"  gate thr {t_gate:.4f} (val recall {(g_va[y_va==1] >= t_gate).mean():.3f}, "
          f"wake-rate {(g_va >= t_gate).mean():.3f})", flush=True)
    print(f"  acoustic thr {t_ac:.4f} (val FPR {(a_va[y_va==0] >= t_ac).mean():.3f}, "
          f"recall {(a_va[y_va==1] >= t_ac).mean():.3f})", flush=True)

    t_text, n_text_neg = (0.5, 0)
    if not os.environ.get("NO_TEXT"):
        print("[cascade] fitting text threshold (KoELECTRA int8, unsmile clean) …", flush=True)
        t_text, n_text_neg = _fit_text_threshold()

    thr = Thresholds(gate=t_gate, acoustic=t_ac, text=t_text, meta={
        "fit_split": "val", "n_val": len(va), "gate_recall_target": GATE_RECALL,
        "acoustic_fpr_target": ACOUSTIC_FPR, "text_fpr_target": TEXT_FPR,
        "text_negatives": "kor_unsmile valid clean", "n_text_neg": n_text_neg,
        "gate_model": "distill/student_s1.pt (0.32M)",
        "acoustic_model": "ckpt_ced_mini_vio/best.ckpt int8-dynamic",
        "text_model": "artifacts/koelectra_small_harm_asraug_slang int8-dynamic "
                      "+ configs/text/harm_lexicon.yaml (hybrid max)",
        "inference_shape": "batch=1 CPU (int8 dynamic activation scales are per-batch: the "
                           "same clip shifts up to ~0.05 at batch 8, so the fit must use the "
                           "deployed shape)",
    })
    save_thresholds(thr, _ROOT / OUT)
    print(f"[cascade] saved thresholds -> {OUT}", flush=True)

    print("[cascade] --- phase 2: evaluate on TEST ---", flush=True)
    g_te, y_te = _scores(gate, te, tax, cfg_pp, "gate/test")
    a_te, _ = _scores(trigger, te, tax, cfg_pp, "trigger/test")

    np.savez(_ROOT / "data_dl/artifacts/cascade_scores.npz", gate_val=g_va, acoustic_val=a_va,
             y_val=y_va, gate_test=g_te, acoustic_test=a_te, y_test=y_te)

    ced_alone = _metrics(a_te, y_te, t_ac)
    # In-sample reference: threshold fit ON the test negatives. Published recall@FPR numbers
    # (compare_vio.py, model_light §2-1) are in-sample; the gap vs `ced_alone` above is the
    # cost of transferring a val-fit operating point to unseen data — a deployment reality.
    t_ac_in = thr_at_fpr(a_te[y_te == 0], ACOUSTIC_FPR)
    ced_insample = _metrics(a_te, y_te, t_ac_in)
    # assembled cascade: the decision module is the single source of truth
    esc = np.array([decide(thr, float(g), float(a)).escalate
                    for g, a in zip(g_te, a_te, strict=True)])
    woke = g_te >= t_gate
    casc = {"recall": float(esc[y_te == 1].mean()), "fpr": float(esc[y_te == 0].mean()),
            "n_pos": int((y_te == 1).sum()), "n_neg": int((y_te == 0).sum())}
    gate_m = _metrics(g_te, y_te, t_gate)
    lat = _latency(gate, trigger, te, tax, cfg_pp, LAT_N)
    # Duty-cycle accounting: the gate only pays off if gate_cost + wake_rate * trigger_cost
    # is cheaper than trigger_cost alone.
    if "gate" in lat and "trigger" in lat:
        cost_gated = lat["gate"] + float(woke.mean()) * lat["trigger"]
        lat["cost_ms_gated_path"] = cost_gated
        lat["cost_ms_trigger_only"] = lat["trigger"]
        lat["gate_saves_compute"] = bool(cost_gated < lat["trigger"])

    rep = {"thresholds": {"gate": t_gate, "acoustic": t_ac, "text": t_text},
           "test": {"gate": gate_m, "ced_alone_val_fit_thr": ced_alone,
                    "ced_alone_insample_thr": ced_insample, "insample_thr": t_ac_in,
                    "cascade": casc, "wake_rate": float(woke.mean()),
                    "missed_by_gate": int(((y_te == 1) & ~woke).sum())},
           "latency_ms_per_clip_cpu_batch1": lat}
    Path(_ROOT / REPORT).parent.mkdir(parents=True, exist_ok=True)
    Path(_ROOT / REPORT).write_text(json.dumps(rep, indent=1))

    print("\n=============== CASCADE (TEST) ===============")
    print(f" gate  @{t_gate:.3f}: recall {gate_m['recall']:.3f}  wake-rate {woke.mean():.3f} "
          f"(trigger sleeps {1-woke.mean():.1%} of the time)")
    print(f" CED alone @{t_ac:.3f} (val-fit thr): recall {ced_alone['recall']:.3f}  "
          f"FPR {ced_alone['fpr']:.3f}")
    print(f" CED alone @{t_ac_in:.3f} (in-sample thr, = published convention): "
          f"recall {ced_insample['recall']:.3f}  FPR {ced_insample['fpr']:.3f}")
    print(f" CASCADE          : recall {casc['recall']:.3f}  FPR {casc['fpr']:.3f}  "
          f"(Δrecall {casc['recall']-ced_alone['recall']:+.3f})")
    print(f" positives lost by gate: {rep['test']['missed_by_gate']}/{casc['n_pos']}")
    print(" latency (CPU, batch1): " +
          "  ".join(f"{k} {v:.0f}ms" for k, v in lat.items() if isinstance(v, float)))
    if "gate_saves_compute" in lat:
        verdict = "SAVES" if lat["gate_saves_compute"] else "COSTS MORE THAN it saves"
        print(f" duty-cycle: gated path {lat['cost_ms_gated_path']:.0f}ms vs trigger-only "
              f"{lat['trigger']:.0f}ms -> tier-1 gate {verdict}")
    print(f" text threshold (for the language branch): {t_text:.4f} @FPR{TEXT_FPR:.0%}")
    print("==============================================")


if __name__ == "__main__":
    main()
