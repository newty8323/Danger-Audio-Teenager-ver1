"""Phase 2 — BEATs fine-tuning (strategy B) on Kaggle GPU. Attacks recall@FPR1%.

Raw audio -> trainable BEATs (top-k blocks) -> MIL + heads, warm-started from the
frozen-head checkpoint (ckpt_beats_combined). Reuses the project Trainer, so it inherits
resume-auto, the 11h Kaggle time-guard, AMP(cuda), SupCon curriculum, early-stop val-mAP,
and per-epoch wandb logging. Everything is path-configurable via env so the same file runs
locally and on Kaggle (dataset mounted at /kaggle/input).

    # local dev (only when explicitly allowed):
    uv run python scripts/train_beats_finetune.py
    # kaggle: paths via env, wandb key from Kaggle Secret WANDB_API_KEY

Env: CLIP_DIR, VIOLENCE_MANIFEST, GAMBLING_MANIFEST, BEATS_CKPT, HEAD_CKPT, CKPT_DIR,
     EPOCHS, UNFREEZE_TOP_K, BATCH_SIZE, GRAD_ACCUM, WANDB_PROJECT, WANDB_RUN_ID.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import combined_data as CD  # noqa: E402

# --- path config (env-overridable for Kaggle) ---
CLIP_DIR = os.environ.get("CLIP_DIR", "data/clips")
CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", CD.VIOLENCE)
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", CD.GAMBLING)
BEATS_CKPT = os.environ.get("BEATS_CKPT", "weights/beats/BEATs_iter3_plus_AS2M.pt")
HEAD_CKPT = os.environ.get("HEAD_CKPT", "artifacts/ckpt_beats_combined/best.ckpt")
CKPT_DIR = os.environ.get("CKPT_DIR", "artifacts/ckpt_beats_finetune")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
UNFREEZE_TOP_K = int(os.environ.get("UNFREEZE_TOP_K", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
# Augmentation is OFF by default: the single-seed A/B (2026-07-18) showed it did not
# improve the target (violence recall@FPR1% regressed, gunshot -.184; mixup blurs sharp
# transients). Kept toggle-able (AUGMENT=1) for a later MUSAN/multi-seed revisit.
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
SEED = int(os.environ.get("SEED", "42"))  # multi-seed: varies data order (sampler) + init

from datasets import augment  # noqa: E402
from datasets.sampler import BalancedBatchSampler  # noqa: E402
from datasets.taxonomy import load_taxonomy  # noqa: E402
from evaluate import _json_safe, harm_report, predict  # noqa: E402
from losses.combined import CombinedLoss, LossConfig  # noqa: E402
from models.beats_finetune import build_finetune_model  # noqa: E402
from preprocess.audio import load_audio  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402
from preprocess.pipeline import fix_length  # noqa: E402
from training.config import CurriculumStage, TrainConfig  # noqa: E402
from training.trainer import Trainer, resolve_device  # noqa: E402


class RawAudioDataset(Dataset):
    """Serves (raw 16kHz waveform (N,), multihot) — BEATs computes its own fbank.

    In train mode, applies on-the-fly waveform augmentation (spec §4: gain, time-shift,
    additive noise, mixup). Eval mode is always clean. Reproducible via a per-(seed,
    epoch, idx) RNG, mirroring LogMelDataset.
    """

    def __init__(self, records, tax, cfg_pp: PreprocessConfig, train: bool = False,
                 aug: augment.WaveAugmentConfig | None = None, seed: int = 42):
        self.records = records
        self.tax = tax
        self.sr = cfg_pp.sample_rate
        self.n = cfg_pp.clip_samples
        self.train = train
        self.aug = aug or augment.WaveAugmentConfig()
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, idx))

    def _load(self, i):
        r = self.records[i]
        try:
            wav = load_audio(f"{CLIP_DIR}/{r.clip_id}.wav", sample_rate=self.sr)
            wav = fix_length(wav, self.n).astype(np.float32)
        except Exception as e:
            # Corrupt/undecodable clip (e.g. a truncated download): don't crash a
            # multi-hour run — substitute silence with an all-negative label (safe).
            print(f"[warn] undecodable clip {r.clip_id}: {type(e).__name__} -> silence", flush=True)
            return (np.zeros(self.n, dtype=np.float32),
                    np.zeros(self.tax.num_classes, dtype=np.float32))
        return wav, r.multihot(self.tax)

    def __getitem__(self, i):
        wav, label = self._load(i)
        if not (self.train and AUGMENT):
            return torch.from_numpy(wav), torch.from_numpy(label)

        rng, a = self._rng(i), self.aug
        # mixup with a random partner (label union); exclude self to avoid a no-op mix
        if rng.random() < a.mixup_p and len(self.records) > 1:
            partner = int(rng.integers(len(self.records) - 1))
            if partner >= i:
                partner += 1
            wav_b, label_b = self._load(partner)
            wav, label = augment.wav_mixup(wav, label, wav_b, label_b, a.mixup_alpha, rng)
        wav = augment.wav_gain(wav, a.gain_max_db, a.gain_p, rng)
        wav = augment.wav_time_shift(wav, a.time_shift_max_sec, self.sr, a.time_shift_p, rng)
        wav = augment.wav_add_noise(wav, None, (a.noise_snr_min, a.noise_snr_max), a.noise_p, rng)
        return torch.from_numpy(np.ascontiguousarray(wav)), torch.from_numpy(label)


def _has_clip(clip_id: str) -> bool:
    return os.path.exists(f"{CLIP_DIR}/{clip_id}.wav")


def _make_wandb_logger():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        print("(no WANDB_API_KEY — skipping wandb)")
        return None
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "audio-harm"),
               group=os.environ.get("WANDB_GROUP", "beats-finetune"),
               id=os.environ.get("WANDB_RUN_ID"), resume="allow",
               config={"epochs": EPOCHS, "unfreeze_top_k": UNFREEZE_TOP_K,
                       "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM})
    return lambda info: wandb.log(info)


def _class_alpha(records, tax, device, lo=0.5, hi=0.8):
    """Per-class focal alpha (positive weight) ~ inverse-sqrt class frequency.

    Rarer / lower-data classes (e.g. gambling vs the more numerous violence, or the
    weaker vio_scream / gmb_table) get more gradient on their positives, which directly
    targets recall@FPR1% and counters the 'class exchange' where a plain fine-tune
    sacrifices low-data classes to improve the abundant/easy ones. Toggle with
    env CLASS_BALANCE=0 to fall back to unweighted focal (A/B baseline).
    """
    counts = np.zeros(tax.num_classes, dtype=np.float64)
    for r in records:
        counts += np.asarray(r.multihot(tax), dtype=np.float64)
    alpha = np.full(tax.num_classes, lo, dtype=np.float64)
    pos = counts > 0
    if pos.sum() >= 2:
        inv = 1.0 / np.sqrt(counts[pos])
        rng = float(inv.max() - inv.min())
        norm = (inv - inv.min()) / rng if rng > 0 else np.zeros_like(inv)
        alpha[pos] = lo + (hi - lo) * norm
    return torch.tensor(alpha, dtype=torch.float32, device=device), counts


def main() -> None:
    torch.manual_seed(SEED)  # weight-init/dropout; data order is seeded via cfg.seed below
    tax = load_taxonomy()
    cfg_pp = PreprocessConfig()
    tr, va, te = CD.build_combined_records(exists_fn=_has_clip)
    # Controlled data-effect experiment: drop clips listed in EXCLUDE_MANIFESTS from TRAIN
    # only (val/test come from the same fixed split), so an "original-data" model and a
    # "full-data" model are evaluated on an IDENTICAL held-out test set.
    excl = os.environ.get("EXCLUDE_MANIFESTS", "").strip()
    if excl:
        drop = set()
        for p in excl.split(","):
            p = p.strip()
            if p and os.path.exists(p):
                for line in open(p):
                    drop.add(json.loads(line)["clip_id"])
        before = len(tr)
        tr = [r for r in tr if r.clip_id not in drop]
        print(f"EXCLUDE_MANIFESTS: dropped {before - len(tr)} train clips ({before}->{len(tr)})")
    # Keep only TRAIN clips whose wav predates this epoch time (Kaggle-original clips have
    # early mtime; this session's downloads are later). Clean way to build an "original-data"
    # train set for the Phase-2 controlled experiment, since re-listed AudioSet segments make
    # clip_id-based exclusion unreliable. val/test unchanged (same fixed source-disjoint split).
    mtb = os.environ.get("TRAIN_MTIME_BEFORE", "").strip()
    if mtb:
        thr = float(mtb)
        before = len(tr)
        tr = [r for r in tr if os.path.getmtime(f"{CLIP_DIR}/{r.clip_id}.wav") < thr]
        print(f"TRAIN_MTIME_BEFORE({thr:.0f}): kept original-mtime train clips ({before}->{len(tr)})")
    print(f"finetune data: train {len(tr)} val {len(va)} test {len(te)}  (clips in {CLIP_DIR})  seed={SEED}")

    model = build_finetune_model(tax.num_classes, head_ckpt=HEAD_CKPT,
                                 beats_ckpt=BEATS_CKPT, unfreeze_top_k=UNFREEZE_TOP_K)
    print(f"trainable: BEATs top-{UNFREEZE_TOP_K} {model.backbone.trainable_parameters()/1e6:.1f}M "
          f"+ heads {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone'))/1e6:.2f}M")

    # Precision: BEATs NaNs in fp16 (narrow range), so use bf16 on Ampere+ GPUs
    # (RTX 30xx/40xx, A100 — fp32 range, no overflow, ~2x faster than fp32) and fall
    # back to fp32 elsewhere (e.g. T4, which has no bf16). Never fp16 for this model.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu/mps"
    print(f"precision: {'bf16 AMP' if use_bf16 else 'fp32'}  (device: {gpu})")
    cfg = TrainConfig(
        device="auto", batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM, num_workers=2,
        lr_heads=1e-4, lr_backbone=1e-5, layer_decay=1.0, warmup_pct=0.05, patience=8,
        amp=use_bf16, amp_dtype="bf16", time_guard_hours=11.0, ckpt_dir=CKPT_DIR, seed=SEED,
        curriculum=(CurriculumStage("finetune", EPOCHS, freeze_backbone=False, use_supcon=True),),
    )
    device = resolve_device("auto")
    if os.environ.get("CLASS_BALANCE", "1") == "1":
        alpha, counts = _class_alpha(tr, tax, device)
        top = sorted(zip(tax.all_classes, alpha.tolist(), counts.tolist(), strict=True),
                     key=lambda x: -x[1])[:4]
        print("class-balanced focal alpha (top-weighted): " +
              ", ".join(f"{n} a={a:.2f}(n={int(c)})" for n, a, c in top))
    else:
        alpha = None
        print("class balance OFF (unweighted focal baseline)")
    print(f"augmentation: {'ON (gain/time-shift/noise/mixup, spec §4)' if AUGMENT else 'OFF'}")
    sampler = BalancedBatchSampler(tr, tax, BATCH_SIZE, seed=cfg.seed)
    train_ds = RawAudioDataset(tr, tax, cfg_pp, train=True, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.num_workers)
    val_loader = DataLoader(RawAudioDataset(va, tax, cfg_pp, train=False), batch_size=BATCH_SIZE)
    test_loader = DataLoader(RawAudioDataset(te, tax, cfg_pp, train=False), batch_size=BATCH_SIZE)

    trainer = Trainer(model, CombinedLoss(LossConfig(), alpha=alpha), cfg,
                      on_epoch=_make_wandb_logger())
    res = trainer.fit(train_loader, val_loader, resume="auto")
    print(f"\n[finetune] status={res.status} best_val_mAP={res.best_metric:.3f}")

    probs, labels = predict(trainer.model, test_loader, device)
    report = harm_report(probs, labels, tax)
    report["run"] = {"status": res.status, "best_val_map": float(res.best_metric),
                     "unfreeze_top_k": UNFREEZE_TOP_K, "epochs": EPOCHS}
    out = Path(CKPT_DIR) / "eval_finetune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2))
    print(f"test macro mAP {report['macro_map']:.3f} · harm AUROC {report['macro_auroc_harm']:.3f}"
          f" · recall@FPR1% vio {report['per_category'].get('vio',{}).get('recall_at_fpr')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
