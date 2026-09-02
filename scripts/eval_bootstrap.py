"""Tier-0 statistical rigor: bootstrap CIs + paired-bootstrap significance for the
Model A (original) vs Model B (full) comparison, on the IDENTICAL test set.

Answers "are the reported improvements real or noise?" — the core brutal question.
- Per harm class: AP, AUROC, recall@FPR{5,10,20}% with 95% bootstrap CI.
- Paired bootstrap for B-A (same resampled clips for both) -> 95% CI on the delta and
  P(B>A). recall@FPR1% is dropped as primary (threshold set by ~9 negatives = meaningless);
  reported only for reference.

Env: CLIP_DIR, VIOLENCE_MANIFEST, GAMBLING_MANIFEST, BEATS_CKPT, CKPT_A, CKPT_B, N_BOOT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

os.environ.setdefault("CLIP_DIR", "data_dl/clips")
os.environ.setdefault("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
os.environ.setdefault("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

import combined_data as CD  # noqa: E402
CD.VIOLENCE = os.environ["VIOLENCE_MANIFEST"]
CD.GAMBLING = os.environ["GAMBLING_MANIFEST"]

from datasets.taxonomy import load_taxonomy  # noqa: E402
from evaluate import predict  # noqa: E402
from models.beats_finetune import build_finetune_model  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402
from training.metrics import auroc, average_precision, recall_at_fpr  # noqa: E402
from training.trainer import resolve_device  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from train_beats_finetune import RawAudioDataset, _has_clip  # noqa: E402

BEATS_CKPT = os.environ.get("BEATS_CKPT")
CKPT_A = os.environ["CKPT_A"]
CKPT_B = os.environ["CKPT_B"]
N_BOOT = int(os.environ.get("N_BOOT", "2000"))
HARM = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal", "gmb_machine", "gmb_table"]


def load_and_predict(ckpt_path, tax, loader, device):
    model = build_finetune_model(tax.num_classes, head_ckpt=None,
                                 beats_ckpt=BEATS_CKPT, unfreeze_top_k=4)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    model.to(device)
    return predict(model, loader, device)


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    tax = load_taxonomy()
    _, _, te = CD.build_combined_records(exists_fn=_has_clip)
    loader = DataLoader(RawAudioDataset(te, tax, PreprocessConfig(), train=False), batch_size=8)
    device = resolve_device("auto")
    print(f"test n={len(te)} | bootstrap N={N_BOOT}")
    pA, lab = load_and_predict(CKPT_A, tax, loader, device)
    pB, _ = load_and_predict(CKPT_B, tax, loader, device)
    n = len(lab)
    rng = np.random.default_rng(0)
    idx_boot = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    metrics = {
        "AP": lambda yt, ys: average_precision(yt, ys),
        "AUROC": lambda yt, ys: auroc(yt, ys),
        "R@FPR5%": lambda yt, ys: recall_at_fpr(yt, ys, 0.05),
        "R@FPR10%": lambda yt, ys: recall_at_fpr(yt, ys, 0.10),
        "R@FPR1%": lambda yt, ys: recall_at_fpr(yt, ys, 0.01),
    }
    ci_col = os.environ.get("CLIP_DIR")  # unused; keep import tidy
    for cls in HARM:
        c = tax.all_classes.index(cls)
        yt = lab[:, c]
        pos = int(yt.sum())
        print(f"\n=== {cls}  (test positives={pos}) ===")
        print(f"{'metric':10s}{'A':>8}{'B':>8}{'Δ(B-A)':>9}{'95% CI of Δ':>18}{'P(B>A)':>9}  verdict")
        for mname, mfn in metrics.items():
            a = mfn(yt, pA[:, c]); b = mfn(yt, pB[:, c])
            dboot = []
            for bi in idx_boot:
                ytb = yt[bi]
                if ytb.sum() == 0 or (~ytb.astype(bool)).sum() == 0:
                    continue
                dboot.append(mfn(ytb, pB[bi, c]) - mfn(ytb, pA[bi, c]))
            dboot = np.array(dboot)
            lo, hi = ci(dboot)
            pgt = float((dboot > 0).mean())
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"  # 95% CI excludes 0
            av = f"{a:.3f}" if not np.isnan(a) else "nan"
            bv = f"{b:.3f}" if not np.isnan(b) else "nan"
            print(f"{mname:10s}{av:>8}{bv:>8}{b-a:>+9.3f}   [{lo:+.3f},{hi:+.3f}]{pgt:>9.2f}  {sig}")


if __name__ == "__main__":
    main()
