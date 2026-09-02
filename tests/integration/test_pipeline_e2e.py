"""End-to-end integration test: every module seam on tiny synthetic data.

Runs collect -> precompute -> train -> evaluate -> risk -> stream -> mining in one
pass (no network, no GPU). This is the regression guard that the pipeline actually
composes; unit tests cover each module in isolation.
"""

import numpy as np
import torch

from collect.audioset import build_manifest
from datasets.manifest import read_manifest, write_manifest
from datasets.taxonomy import load_taxonomy
from evaluate import harm_report, predict
from infer_stream import StreamRiskInference, make_model_predictor
from losses.combined import CombinedLoss, LossConfig
from mining.candidates import PoolClip, select_candidates
from mining.config import MiningConfig
from mining.review import ReviewSession
from models.harm_model import HarmModel, ModelConfig
from preprocess.normalize import NormStats, apply_norm
from preprocess.precompute import feature_path, precompute_manifest
from risk.policy import load_risk_policy
from risk.scorer import RiskScorer
from training.config import CurriculumStage, TrainConfig
from training.trainer import Trainer

SR = 16_000
LABEL_MAP = "configs/data/audioset_labels.yaml"

# AudioSet mids present in the shipped label map.
SCREAM, GUNSHOT, DOOR = "/m/03qc9zr", "/m/032s66", "/m/02dgv"


def _sine(path, seconds, wav_writer, freq=440.0):
    t = np.arange(int(seconds * SR)) / SR
    wav_writer(path, 0.4 * np.sin(2 * np.pi * freq * t), SR)


def _make_csv(path):
    rows = ["# header", "# header2", "# YTID, start, end, labels"]
    specs = [("sc0", SCREAM), ("sc1", SCREAM), ("sc2", SCREAM),
             ("gn0", GUNSHOT), ("gn1", GUNSHOT), ("gn2", GUNSHOT),
             ("dr0", DOOR), ("dr1", DOOR), ("dr2", DOOR)]
    for ytid, mid in specs:
        rows.append(f'{ytid}, 0.000, 1.000, "{mid}"')
    path.write_text("\n".join(rows) + "\n")


def test_full_pipeline_e2e(tmp_path, wav_writer):
    tax = load_taxonomy()

    # --- 1. collect: CSV -> manifest ---
    csv_path = tmp_path / "seg.csv"
    _make_csv(csv_path)
    manifest0 = tmp_path / "m0.jsonl"
    records = build_manifest([csv_path], LABEL_MAP, manifest0, split="train")
    assert len(records) == 9

    # assign val/test splits (source-disjoint by construction: distinct ytids)
    by_id = {r.clip_id: r for r in records}
    by_id["sc2_0"].split = "val"
    by_id["dr2_0"].split = "val"
    by_id["gn2_0"].split = "test"
    manifest_in = tmp_path / "m_in.jsonl"
    write_manifest(records, manifest_in)

    # --- synth audio for each clip ---
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    for i, r in enumerate(records):
        _sine(audio_root / f"{r.clip_id}.wav", 1.0, wav_writer, freq=300 + 40 * i)

    # --- 2. precompute: audio -> features + norm stats ---
    feature_root = tmp_path / "features"
    stats_path = tmp_path / "norm.npz"
    out_manifest = tmp_path / "m_feat.jsonl"
    result = precompute_manifest(manifest_in, audio_root, feature_root, stats_path, out_manifest)
    assert result.n_processed == 9 and not result.dropped_ids

    # --- 3. train: build loaders + fit 1 epoch ---
    from train import build_loaders  # imported here to keep top-level deps explicit

    cfg = TrainConfig(
        device="cpu", batch_size=2, grad_accum_steps=1, num_workers=0,
        ckpt_dir=str(tmp_path / "ckpts"),
        curriculum=(CurriculumStage("s1", 1, freeze_backbone=False, use_supcon=True),),
    )
    train_loader, val_loader, num_classes = build_loaders(
        str(out_manifest), str(feature_root), str(stats_path), tax, cfg
    )
    assert num_classes == tax.num_classes
    model = HarmModel(num_classes, ModelConfig(backbone_out_dim=32))
    trainer = Trainer(model, CombinedLoss(LossConfig()), cfg)
    fit = trainer.fit(train_loader, val_loader, resume="none")
    assert fit.status == "completed"
    assert (tmp_path / "ckpts" / "last.ckpt").exists()

    # --- 4. evaluate: §9 report ---
    device = torch.device("cpu")
    probs, labels = predict(trainer.model, val_loader, device)
    report = harm_report(probs, labels, tax)
    assert set(report["per_class"]) == set(tax.all_classes)
    assert "targets" in report

    # --- 5. risk: fit scorer on val, score ---
    stats = NormStats.load(str(stats_path))
    harm_idx = [tax.index_of(n) for n in tax.harm_classes]
    risk_targets = (labels[:, harm_idx].max(axis=1) > 0).astype(float)
    scorer = RiskScorer.from_policy(load_risk_policy(), tax).fit(probs, risk_targets)
    r = scorer.score(probs)
    assert r.shape == (probs.shape[0],)
    assert (r > 0).all() and (r < 1).all()

    # --- 6. infer_stream: 12s clip -> windowed risk (§9b) ---
    stream_wav = tmp_path / "stream.wav"
    _sine(stream_wav, 12.0, wav_writer)
    from preprocess.audio import load_audio
    waveform = load_audio(str(stream_wav), sample_rate=SR)
    infer = StreamRiskInference(tax, scorer, load_risk_policy())
    predictor = make_model_predictor(trainer.model, stats, device)
    stream_results = infer.run(waveform, predictor, clip_id="demo")
    assert len(stream_results) >= 1
    d = stream_results[0].to_dict()
    assert {"clip_id", "probs", "risk_score", "risk_level", "top_events"} <= set(d)

    # --- 7. mining: pool predictions -> candidates -> review -> promote ---
    val_recs = [r for r in read_manifest(out_manifest) if r.split == "val"]
    pool = [PoolClip(r.clip_id, r.source, r.source_id, r.start_sec, r.duration) for r in val_recs]
    feats = np.stack([
        apply_norm(np.load(feature_path(r.clip_id, feature_root)), stats)[0] for r in val_recs
    ])
    with torch.no_grad():
        pool_probs = trainer.model.predict_proba(
            torch.from_numpy(feats).unsqueeze(1)
        ).numpy()
    cands = select_candidates(pool, pool_probs, tax, MiningConfig(fp_prob_threshold=0.0, top_k=10))
    assert len(cands) == len(pool)  # threshold 0 -> all are FP candidates

    session = ReviewSession(cands, taxonomy=tax)
    session.decide(cands[0].clip_id, "false_positive", "chair_scrape")
    new_records = session.export(tax)
    assert len(new_records) == 1
    assert new_records[0].labels == ["chair_scrape"]
    assert new_records[0].split == "train"

    # --- 8. CLIs that close the pipeline: risk.fit + mining.run (via from_checkpoint) ---
    ckpt_path = str(tmp_path / "ckpts" / "last.ckpt")
    from risk.fit import main as fit_risk_main
    risk_params = tmp_path / "risk.json"
    fit_risk_main([
        "--manifest", str(out_manifest), "--feature-root", str(feature_root),
        "--stats", str(stats_path), "--ckpt", ckpt_path, "--split", "val",
        "--out", str(risk_params),
    ])
    assert risk_params.exists()
    RiskScorer.from_policy(load_risk_policy(), tax).load_params(str(risk_params))

    from mining.run import main as mining_run_main
    queue = tmp_path / "queue.jsonl"
    mining_run_main([
        "--pool-manifest", str(out_manifest), "--feature-root", str(feature_root),
        "--stats", str(stats_path), "--ckpt", ckpt_path, "--out", str(queue),
    ])
    assert queue.exists()
