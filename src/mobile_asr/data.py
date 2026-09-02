"""Audio loading and balanced sampling for mobile Whisper training."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from mobile_asr.manifest import DOMAINS, ASRItem

SR = 16000


def load_audio(path, max_seconds: float = 30.0, *, random_crop: bool = False) -> np.ndarray:
    import soundfile as sf

    wav, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sample_rate != SR:
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sample_rate, SR).numpy()
    limit = int(max_seconds * SR)
    if len(wav) > limit:
        start = random.randint(0, len(wav) - limit) if random_crop else 0
        wav = wav[start:start + limit]
    return np.asarray(wav, dtype=np.float32)


class WhisperManifestDataset(Dataset):
    def __init__(self, rows: list[ASRItem], max_seconds: float = 30.0,
                 random_crop: bool = False):
        self.rows = rows
        self.max_seconds = max_seconds
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "id": row.item_id,
            "domain": row.domain,
            "audio": load_audio(row.audio, self.max_seconds, random_crop=self.random_crop),
            "text": row.text,
        }


@dataclass
class WhisperCollator:
    processor: object

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        audio = self.processor.feature_extractor(
            [row["audio"] for row in features],
            sampling_rate=SR,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )
        tokenized = self.processor.tokenizer(
            [row["text"] for row in features],
            return_tensors="pt",
            padding=True,
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        decoder_start = self.processor.tokenizer.bos_token_id
        if labels.shape[1] and torch.all(labels[:, 0].eq(decoder_start)):
            labels = labels[:, 1:]
        batch = {"input_features": audio.input_features, "labels": labels}
        if hasattr(audio, "attention_mask"):
            batch["attention_mask"] = audio.attention_mask
        return batch


def balanced_sampler(rows: list[ASRItem], ratios: dict[str, float], seed: int,
                     num_samples: int | None = None) -> WeightedRandomSampler:
    unknown = set(ratios) - set(DOMAINS)
    if unknown:
        raise ValueError(f"unknown domain ratios: {sorted(unknown)}")
    total_ratio = sum(float(ratios.get(domain, 0.0)) for domain in DOMAINS)
    if total_ratio <= 0:
        raise ValueError("domain ratios must have a positive sum")
    counts = {domain: sum(row.domain == domain for row in rows) for domain in DOMAINS}
    missing = [domain for domain in DOMAINS if ratios.get(domain, 0.0) > 0 and counts[domain] == 0]
    if missing:
        raise ValueError(f"training manifest has no rows for weighted domains: {missing}")
    weights = [float(ratios.get(row.domain, 0.0)) / counts[row.domain] for row in rows]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(rows),
        replacement=True,
        generator=generator,
    )
