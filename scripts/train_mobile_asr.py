"""Fine-tune a sub-100 MB Korean Whisper-tiny student on four audio domains.

The manifest may contain human transcripts or ``teacher_text`` generated offline by a large
teacher.  That is sequence-level distillation: the phone only receives the tiny student.

Run:
  PYTHONPATH=src uv run --group nlp --group asr python scripts/train_mobile_asr.py
Override examples:
  ... manifest=/data/mobile_asr.jsonl train.batch_size=8 device=cuda
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.optimization import get_cosine_schedule_with_warmup

from mobile_asr.data import WhisperCollator, WhisperManifestDataset, balanced_sampler
from mobile_asr.manifest import domain_counts, load_manifest


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    found = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            found.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(found, default=(0, None))[1]


def _save_checkpoint(model, processor, optimizer, scheduler, output_dir: Path,
                     step: int, epoch: int, best_val: float) -> Path:
    target = output_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    processor.save_pretrained(target)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               target / "training_state.pt")
    (target / "trainer_state.json").write_text(json.dumps(
        {"step": step, "epoch": epoch, "best_val": best_val}, indent=2), encoding="utf-8")
    return target


@torch.no_grad()
def _validation_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@hydra.main(version_base=None, config_path="../configs/asr", config_name="mobile_whisper_tiny")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    manifest_path = Path(to_absolute_path(str(cfg.manifest)))
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest_path)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    if not train_rows or not val_rows:
        raise SystemExit("manifest needs non-empty train and val splits")
    print(f"[mobile-asr] train {domain_counts(train_rows)}", flush=True)
    print(f"[mobile-asr] val   {domain_counts(val_rows)}", flush=True)

    checkpoint = _latest_checkpoint(output_dir) if str(cfg.resume) == "auto" else None
    model_source = str(checkpoint or cfg.model_id)
    processor_source = str(checkpoint or cfg.model_id)
    processor = WhisperProcessor.from_pretrained(
        processor_source, language=str(cfg.language), task=str(cfg.task)
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_source)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if bool(cfg.train.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    device = _device(str(cfg.device))
    model.to(device)

    collator = WhisperCollator(processor)
    train_ds = WhisperManifestDataset(train_rows, float(cfg.max_seconds), random_crop=True)
    val_ds = WhisperManifestDataset(val_rows, float(cfg.max_seconds), random_crop=False)
    ratios = {k: float(v) for k, v in cfg.domain_ratios.items()}
    sampler = balanced_sampler(train_rows, ratios, seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        collate_fn=collator,
        num_workers=int(cfg.train.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(cfg.train.num_workers),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay)
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(cfg.train.grad_accum))
    total_steps = updates_per_epoch * int(cfg.train.epochs)
    warmup_steps = int(total_steps * float(cfg.train.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if checkpoint:
        state_file = checkpoint / "training_state.pt"
        meta_file = checkpoint / "trainer_state.json"
        if state_file.is_file() and meta_file.is_file():
            state = torch.load(state_file, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            start_epoch = int(meta["epoch"])
            global_step = int(meta["step"])
            best_val = float(meta.get("best_val", best_val))
            print(f"[mobile-asr] resumed {checkpoint.name} at epoch {start_epoch}", flush=True)

    (output_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    started = time.monotonic()
    limit_sec = float(cfg.train.time_guard_hours) * 3600
    grad_accum = int(cfg.train.grad_accum)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        running = []
        for batch_index, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running.append(float(loss.detach().cpu()) * grad_accum)
            should_update = (
                (batch_index + 1) % grad_accum == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if time.monotonic() - started >= limit_sec:
                saved = _save_checkpoint(model, processor, optimizer, scheduler, output_dir,
                                         global_step, epoch, best_val)
                print(f"[mobile-asr] time guard: saved {saved}", flush=True)
                return
        val_loss = _validation_loss(model, val_loader, device)
        best_val = min(best_val, val_loss)
        saved = _save_checkpoint(model, processor, optimizer, scheduler, output_dir,
                                 global_step, epoch + 1, best_val)
        print(f"[mobile-asr] epoch {epoch + 1}: train_loss={np.mean(running):.4f} "
              f"val_loss={val_loss:.4f} -> {saved.name}", flush=True)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[mobile-asr] complete -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
