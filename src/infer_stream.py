"""Streaming risk inference — Task C (spec §1, §9b, §11).

Slides a 10s window over audio (default 5s stride, densified to 2.5s while in
`warn`), scores each window's harm probabilities into a risk score, smooths with
the streaming tracker, and emits a per-window result in the §9b schema:

    { clip_id, probs{class:float}, risk_score, risk_level, top_events[3] }

The scoring is decoupled: `run` takes a ``predict(window) -> probs`` callable, so
the driver is unit-testable without a model. ``make_model_predictor`` wires the
real preprocess + model path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from datasets.taxonomy import Taxonomy, load_taxonomy
from models.harm_model import HarmModel
from preprocess.audio import load_audio
from preprocess.config import PreprocessConfig
from preprocess.logmel import LogMelExtractor
from preprocess.normalize import NormStats, apply_norm
from risk.policy import RiskPolicy, load_risk_policy
from risk.scorer import RiskScorer
from risk.stream import StreamRiskTracker
from training.trainer import resolve_device

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass
class StreamResult:
    """One window's result (§9b schema + streaming extras).

    Note: ``risk_level`` reflects streaming escalation (spec §8), so it can be
    ``block`` while ``risk_score`` is still in the warn band (e.g. after 3
    consecutive warns). Don't assume ``risk_level == threshold(risk_score)``.
    """

    clip_id: str
    window_index: int
    start_sec: float
    probs: dict[str, float]
    risk_score: float  # EMA-smoothed risk (drives the base level)
    raw_risk: float
    risk_level: str
    top_events: list[dict] = field(default_factory=list)
    stride_s: float = 5.0
    consecutive_warns: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def top_events(probs: np.ndarray, taxonomy: Taxonomy, k: int = 3) -> list[dict]:
    order = np.argsort(-probs)[:k]
    return [{"class": taxonomy.all_classes[i], "prob": float(probs[i])} for i in order]


def _build_result(clip_id, idx, start_sec, probs, taxonomy, state) -> StreamResult:
    return StreamResult(
        clip_id=f"{clip_id}@{start_sec:g}s",
        window_index=idx,
        start_sec=float(start_sec),
        probs={taxonomy.all_classes[i]: float(probs[i]) for i in range(taxonomy.num_classes)},
        risk_score=float(state.smoothed),
        raw_risk=float(state.raw),
        risk_level=state.level,
        top_events=top_events(probs, taxonomy),
        stride_s=float(state.stride_s),
        consecutive_warns=state.consecutive_warns,
    )


class StreamRiskInference:
    def __init__(
        self,
        taxonomy: Taxonomy,
        scorer: RiskScorer,
        policy: RiskPolicy,
        sample_rate: int = 16_000,
        window_s: float = 10.0,
    ) -> None:
        self.taxonomy = taxonomy
        self.scorer = scorer
        self.policy = policy
        self.sample_rate = sample_rate
        self.window_n = int(round(window_s * sample_rate))
        self.tracker = StreamRiskTracker(policy)

    def run(
        self, waveform: np.ndarray, predict: Predict, clip_id: str = "clip"
    ) -> list[StreamResult]:
        """Slide over ``waveform``; stride adapts to each window's level."""
        self.tracker.reset()
        results: list[StreamResult] = []
        pos = 0
        idx = 0
        n = len(waveform)
        while pos + self.window_n <= n:
            window = waveform[pos:pos + self.window_n]
            probs = np.asarray(predict(window), dtype=np.float64).reshape(-1)
            raw = float(self.scorer.score(probs))
            state = self.tracker.update(raw)
            results.append(
                _build_result(clip_id, idx, pos / self.sample_rate, probs, self.taxonomy, state)
            )
            pos += max(1, int(round(state.stride_s * self.sample_rate)))
            idx += 1
        return results


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Device equality that ignores an unset index.

    ``resolve_device`` returns an index-less device ("mps", "cuda") while a moved
    module reports ``mps:0`` / ``cuda:0``, and ``torch.device`` compares those as
    unequal. Treat an unset index as "whatever the other side is on".
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


def make_model_predictor(
    model: HarmModel,
    norm_stats: NormStats,
    device: torch.device,
    cfg: PreprocessConfig | None = None,
) -> Predict:
    """Real predictor: window waveform -> log-mel -> normalize -> model -> probs."""
    cfg = cfg or PreprocessConfig()
    extractor = LogMelExtractor(cfg)
    model_device = next(model.parameters()).device
    if not _same_device(model_device, device):
        raise ValueError(f"model is on {model_device} but predictor targets {device}")

    @torch.no_grad()
    def predict(window: np.ndarray) -> np.ndarray:
        logmel = apply_norm(extractor(window), norm_stats)  # (1, F, T)
        x = torch.from_numpy(logmel).unsqueeze(0).to(device)  # (1, 1, F, T)
        probs = model.predict_proba(x)
        return probs.squeeze(0).cpu().numpy()

    return predict


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Streaming risk inference over an audio file.")
    p.add_argument("--audio", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--risk-params", required=True, help="fitted risk (a,b,c) json")
    p.add_argument("--classes", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--out", default=None, help="write per-window results as JSONL")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.classes)
    policy = load_risk_policy(args.policy)
    scorer = RiskScorer.from_policy(policy, taxonomy).load_params(args.risk_params)

    device = resolve_device("auto")
    model = HarmModel.from_checkpoint(args.ckpt, taxonomy.num_classes, map_location=device)
    stats = NormStats.load(args.stats)

    waveform = load_audio(args.audio, sample_rate=PreprocessConfig().sample_rate)
    infer = StreamRiskInference(taxonomy, scorer, policy)
    predict = make_model_predictor(model, stats, device)
    results = infer.run(waveform, predict, clip_id=Path(args.audio).stem)

    if not results:
        seconds = len(waveform) / PreprocessConfig().sample_rate
        print(f"no full 10s windows in {args.audio} ({seconds:.1f}s of audio)")
        return

    for r in results:
        print(f"{r.start_sec:6.1f}s  {r.risk_level:5s}  R={r.risk_score:.3f}  "
              f"top={r.top_events[0]['class']}({r.top_events[0]['prob']:.2f})")
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"\nwrote {len(results)} windows -> {args.out}")


if __name__ == "__main__":
    main()
