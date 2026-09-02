"""Any-violence TRIGGER comparison across models with different taxonomies.
Handles both 4-col (violence-only v2.0) and 23-col (v1.0) prob dumps: any-violence
score/label = max over the 4 violence columns (all 4 for vio-only npz; the v1.0 vio
indices for 23-col npz). recall@fixed-FPR with bootstrap CI + paired Δ vs a baseline.

Edit MODELS below (label -> npz). BASELINE picks the paired-Δ reference.
"""
import os, sys
import numpy as np
sys.path.insert(0, "src")
from datasets.taxonomy import load_taxonomy

VIO = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal"]
V1 = "configs/data/classes.yaml"
FPRS = [0.01, 0.05, 0.10]
NBOOT = 2000
rng = np.random.default_rng(0)
_v1tax = load_taxonomy(V1)
_v1idx = [_v1tax.index_of(c) for c in VIO]

MODELS = {  # label -> npz  (only those that exist are used)
    "full-fp32":          "data_dl/artifacts/probs_beats_fp32.npz",
    "full-int8(quant)":   "data_dl/artifacts/probs_beats_int8.npz",
    "L6(23cls,eval-vio)": "data_dl/artifacts/probs_beats_L6.npz",
    "student-s1(0.32M)":  "data_dl/artifacts/probs_student_s1.npz",
    "student-s2(0.94M)":  "data_dl/artifacts/probs_student_s2.npz",
    "student-s3(2.9M)":   "data_dl/artifacts/probs_student_s3.npz",
    "CED-mini(10M,vio)":  "data_dl/artifacts/probs_ced_mini.npz",
    "CED-mini-int8(10MB)": "data_dl/artifacts/probs_ced_mini_int8.npz",
}
BASELINE = "full-fp32"  # student/int8 Δ vs teacher fp32


def load_anyvio(path):
    d = np.load(path); probs, labels = d["probs"], d["labels"]
    idx = [0, 1, 2, 3] if probs.shape[1] == 4 else _v1idx
    y = labels[:, idx].max(1).astype(int)
    s = probs[:, idx].max(1)
    return s, y


def thr_at_fpr(s, y, fpr):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return np.inf
    k = max(0, int(np.floor(fpr * len(neg))) - 1)
    return neg[min(k, len(neg) - 1)]


def recall_at(s, y, thr):
    pos = s[y == 1]
    return float((pos >= thr).mean()) if len(pos) else float("nan")


def ap(s, y):
    order = np.argsort(-s); yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(yy.sum(), 1)
    a = 0.0; prev = 0.0
    for p, r in zip(prec, rec):
        a += p * (r - prev); prev = r
    return a


data = {}
for name, p in MODELS.items():
    if os.path.exists(p):
        data[name] = load_anyvio(p)
    else:
        print(f"(skip {name}: no {p})")

# sanity: same test set / same violence positives across models
ys = [tuple(y.tolist()) for _, y in data.values()]
same = all(y == ys[0] for y in ys)
npos = int(np.array(next(iter(data.values()))[1]).sum())
print(f"\nmodels={list(data)}  |  same-test-labels={same}  any-vio pos={npos}\n")

print("=== any-violence recall @ fixed FPR (in-sample thr; bootstrap 95% CI) ===")
for name, (s, y) in data.items():
    line = [f"[{name:26s}] AP={ap(s,y):.3f}"]
    for fpr in FPRS:
        thr = thr_at_fpr(s, y, fpr); r = recall_at(s, y, thr)
        boot = []
        idx = np.arange(len(y))
        for _ in range(NBOOT):
            bi = rng.choice(idx, size=len(idx), replace=True)
            sb, yb = s[bi], y[bi]
            if yb.sum() == 0 or (yb == 0).sum() == 0:
                continue
            boot.append(recall_at(sb, yb, thr_at_fpr(sb, yb, fpr)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        line.append(f"@{int(fpr*100)}%={r:.3f}[{lo:.2f},{hi:.2f}]")
    print("  " + "  ".join(line))

if BASELINE in data and same:
    sb0, yb0 = data[BASELINE]
    print(f"\n=== paired Δ vs {BASELINE} (95% CI; SIG if excludes 0) ===")
    for name, (s, y) in data.items():
        if name == BASELINE:
            continue
        cells = []
        idx = np.arange(len(y))
        for fpr in FPRS:
            d_obs = recall_at(s, y, thr_at_fpr(s, y, fpr)) - recall_at(sb0, yb0, thr_at_fpr(sb0, yb0, fpr))
            boot = []
            for _ in range(NBOOT):
                bi = rng.choice(idx, size=len(idx), replace=True)
                yb = y[bi]
                if yb.sum() == 0 or (yb == 0).sum() == 0:
                    continue
                d = recall_at(s[bi], yb, thr_at_fpr(s[bi], yb, fpr)) - \
                    recall_at(sb0[bi], yb, thr_at_fpr(sb0[bi], yb, fpr))
                boot.append(d)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            cells.append(f"@{int(fpr*100)}%:{d_obs:+.3f}[{lo:+.2f},{hi:+.2f}]{sig}")
        print(f"  [{name:26s}] " + "  ".join(cells))
