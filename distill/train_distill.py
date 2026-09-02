"""Distill BEATs -> TinyMelCNN student (violence trigger). Trains the student to match the
teacher's soft violence logits (dark knowledge) + projection embedding, with a small hard-label
term. Same combined_data split; teacher targets from dump_teacher_targets.py (keyed by clip_id).
Dumps test probs (same order as probs_beats_fp32.npz) for .autorun/compare_vio.py.

Env: TAG (default s1), EPOCHS, SEED, ALPHA/BETA/GAMMA/TEMP (loss weights), LR, BATCH, WANDB_* .
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")
import combined_data as CD
CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

from datasets.taxonomy import load_taxonomy
from datasets.sampler import BalancedBatchSampler
from preprocess.config import PreprocessConfig
from train_beats_finetune import RawAudioDataset, _has_clip
from student_models import TinyMelCNN

FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
V1_CFG = str(_ROOT / "configs/data/classes.yaml")
VIO_CFG = str(_ROOT / "configs/data/classes_vio.yaml")
# size sweep presets: SIZE -> (conv widths, emb dim) ~ {s1:0.32M, s2:~0.9M, s3:~2.8M}.
# emb dim MUST equal the teacher projection dim (256) — feature distillation compares them directly.
_PRESET = {"s1": ((32, 64, 128), 256), "s2": ((56, 112, 224), 256), "s3": ((100, 200, 400), 256)}
SIZE = os.environ.get("SIZE", "s1")
WIDTHS, EMB = _PRESET.get(SIZE, _PRESET["s1"])
TAG = os.environ.get("TAG", SIZE)
EPOCHS = int(os.environ.get("EPOCHS", "40"))
SEED = int(os.environ.get("SEED", "42"))
ALPHA = float(os.environ.get("ALPHA", "1.0"))   # soft
BETA = float(os.environ.get("BETA", "1.0"))     # feature
GAMMA = float(os.environ.get("GAMMA", "0.3"))   # hard
TEMP = float(os.environ.get("TEMP", "2.0"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "32"))
DDIR = Path(__file__).resolve().parent
CKPT = DDIR / f"student_{TAG}.pt"
OUT_PROBS = f"data_dl/artifacts/probs_student_{TAG}.npz"


def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


def _load_targets(name):
    # Read each array EXACTLY once into a contiguous ndarray + a clip_id->row index map.
    # (NpzFile.__getitem__ has no cache and integer-indexing returns a VIEW pinning the whole
    # array — the old dict-of-rows form was O(N^2) memory and re-decompressed the npz 3N times,
    # which OOM-killed the box.)
    with np.load(DDIR / f"teacher_targets_{name}.npz") as d:
        arrs = (d["vio_logits"], d["emb"], d["hard"])
        idx = {str(c): i for i, c in enumerate(d["clip_ids"])}
    return arrs, idx


class DistillDataset(Dataset):
    """(clean wav, teacher_logits[4], teacher_emb[256], hard[4]) — wav via RawAudioDataset
    (23-class tax just for its loader; its label is ignored). No aug in v1 (teacher target is
    for the clean clip)."""
    def __init__(self, records, tax_full, cfg_pp, targets):
        self.base = RawAudioDataset(records, tax_full, cfg_pp, train=False)
        self.records = records
        (self.tl, self.te, self.hd), self.idx = targets   # 3 shared arrays + id->row map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        wav, _ = self.base[i]
        j = self.idx[str(self.records[i].clip_id)]
        # .copy() detaches the row from the big shared array so torch.from_numpy / the
        # collated batch never pins the full (N x D) array (esp. across DataLoader workers).
        return (wav, torch.from_numpy(self.tl[j].copy()),
                torch.from_numpy(self.te[j].copy()),
                torch.from_numpy(self.hd[j].copy()))


def _ap(s, y):
    o = np.argsort(-s); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(yy.sum(), 1); a = 0.0; prev = 0.0
    for pp, rr in zip(p, r):
        a += pp * (rr - prev); prev = rr
    return float(a)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval(); P, Y = [], []
    for wav, _, _, hd in loader:
        P.append(torch.sigmoid(model(wav.to(device))["logits"]).cpu().numpy()); Y.append(hd.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    s = P.max(1); y = Y.max(1).astype(int)
    m = {"anyvio_ap": _ap(s, y)}
    neg = np.sort(s[y == 0])[::-1]; pos = s[y == 1]
    for f in (0.01, 0.05, 0.10):
        k = max(0, int(np.floor(f * len(neg))) - 1); thr = neg[min(k, len(neg) - 1)] if len(neg) else 1e9
        m[f"recall@fpr{int(f*100)}"] = float((pos >= thr).mean()) if len(pos) else float("nan")
    return m, P, Y


def _wandb():
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "audio-harm"),
               group=os.environ.get("WANDB_GROUP", "distill"),
               id=os.environ.get("WANDB_RUN_ID", f"distill-{TAG}"), resume="allow",
               config={"tag": TAG, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "temp": TEMP,
                       "lr": LR, "batch": BATCH, "epochs": EPOCHS, "seed": SEED})
    return wandb.log


def main():
    torch.manual_seed(SEED)
    tax_full = load_taxonomy(V1_CFG); tax_vio = load_taxonomy(VIO_CFG)
    cfg_pp = PreprocessConfig()
    tr, va, te = CD.build_combined_records(exists_fn=_has_both)
    tgt = {n: _load_targets(n) for n in ("train", "val", "test")}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    student = TinyMelCNN(num_classes=4, widths=WIDTHS, emb_dim=EMB).to(device)
    print(f"[distill-{TAG}] SIZE={SIZE} widths={WIDTHS} emb={EMB} | student {student.num_params():.2f}M params | "
          f"train {len(tr)} val {len(va)} "
          f"test {len(te)} | device {device}", flush=True)

    nw = int(os.environ.get("NUM_WORKERS", "2"))
    sampler = BalancedBatchSampler(tr, tax_vio, BATCH, seed=SEED)
    tl = DataLoader(DistillDataset(tr, tax_full, cfg_pp, tgt["train"]), batch_sampler=sampler, num_workers=nw)
    vl = DataLoader(DistillDataset(va, tax_full, cfg_pp, tgt["val"]), batch_size=32, num_workers=max(1, nw // 2))
    tel = DataLoader(DistillDataset(te, tax_full, cfg_pp, tgt["test"]), batch_size=32, num_workers=max(1, nw // 2))

    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    log = _wandb()
    best, no_imp = -1.0, 0
    nb = max(len(tl), 1)
    for ep in range(EPOCHS):
        student.train(); tot = cs = cf = ch = 0.0
        for wav, tlog, temb, hd in tl:
            wav, tlog, temb, hd = wav.to(device), tlog.to(device), temb.to(device), hd.to(device)
            out = student(wav, return_projection=True)
            soft_t = torch.sigmoid(tlog / TEMP)
            # soft: temperature-softened BCE (no T^2 — keep ~O(0.5) so the 3 terms are comparable).
            l_soft = F.binary_cross_entropy_with_logits(out["logits"] / TEMP, soft_t)
            # feat: cosine distance (O(0..2)) — MSE-over-256 was ~0.008 and drowned out.
            l_feat = (1.0 - F.cosine_similarity(out["embeddings"], temb, dim=1)).mean()
            l_hard = F.binary_cross_entropy_with_logits(out["logits"], hd)
            loss = ALPHA * l_soft + BETA * l_feat + GAMMA * l_hard
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); cs += float(l_soft); cf += float(l_feat); ch += float(l_hard)
        sched.step()
        m, _, _ = _eval(student, vl, device)
        key = m["anyvio_ap"]   # select on val AP: R@FPR on a 74-positive val is too noisy
                               # (it saved an early undertrained ckpt for s3 -> false regression).
        if key > best + 1e-6:
            best, no_imp = key, 0; torch.save({"model": student.state_dict(), "tag": TAG}, CKPT)
        else:
            no_imp += 1
        print(f"[distill-{TAG}] ep{ep} loss={tot/nb:.4f} (soft={cs/nb:.3f} feat={cf/nb:.3f} hard={ch/nb:.3f}) "
              f"val_anyvioAP={m['anyvio_ap']:.3f} R@1%={m['recall@fpr1']:.3f} R@5%={m['recall@fpr5']:.3f} "
              f"R@10%={m['recall@fpr10']:.3f} {'*best' if no_imp==0 else ''}", flush=True)
        if log:
            log({"epoch": ep, "train_loss": tot / nb, "loss/soft": cs / nb, "loss/feat": cf / nb,
                 "loss/hard": ch / nb, **{f"val/{k}": v for k, v in m.items()}})
        if no_imp >= 8:
            print(f"[distill-{TAG}] early stop @ep{ep}", flush=True); break

    student.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    _, P, Y = _eval(student, tel, device)
    Path(OUT_PROBS).parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PROBS, probs=P, labels=Y)
    print(f"[distill-{TAG}] best val AP={best:.3f} | saved test probs {P.shape} -> {OUT_PROBS}", flush=True)


if __name__ == "__main__":
    main()
