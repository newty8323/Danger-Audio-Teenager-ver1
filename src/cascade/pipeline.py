"""Model wiring for the on-device cascade (loaders + per-clip inference).

Everything runs on CPU by default (the deployment target); ASR may use CUDA for offline
experiments. Loaders are lazy so offline eval scripts can wire only the branches they need.

Component contracts (verified against the training/eval scripts):
  - gate: distill/student_models.TinyMelCNN (s1 preset, 0.32M) — raw 16 kHz wav ->
    4 vio logits; score = max sigmoid.
  - trigger: .autorun/train_ced_vio.CEDRawBackbone + models.harm_model.HarmModel
    ("passthrough"), int8 dynamic PTQ on Linear (exactly .autorun/quantize_ced.py) —
    raw wav -> 4 vio probs; score = max.
  - text: KoELECTRA-small (artifacts/koelectra_small_harm_asraug_slang), int8 dynamic PTQ —
    transcript -> 1 - P(safe).
  - asr: Moonshine-tiny-ko; Korean needs ~20 tok/s max_new_tokens (6.5 tok/s truncates).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

from cascade.decision import ClipDecision, Thresholds, decide

_ROOT = Path(__file__).resolve().parents[2]
SR = 16000

_STUDENT_WIDTHS = {"s1": (32, 64, 128), "s2": (56, 112, 224), "s3": (100, 200, 400)}


def _set_quant_engine() -> str:
    """fbgemm on x86, qnnpack on ARM (Apple Silicon / phones). Hardcoding fbgemm crashes
    on ARM; accuracy is engine-independent, speed is not."""
    avail = torch.backends.quantized.supported_engines
    for eng in ("fbgemm", "qnnpack"):
        if eng in avail:
            torch.backends.quantized.engine = eng
            return eng
    return torch.backends.quantized.engine


def _add_paths():
    for p in (_ROOT / "distill", _ROOT / ".autorun", _ROOT / "scripts", _ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# Weights are NOT in git (too large) — they come from the data-v1 release. Say so instead of
# letting torch/transformers raise a bare FileNotFoundError deep in a traceback.
_FETCH_HINT = ("run `bash scripts/fetch_data.sh --models` from the repo root "
               "(~0.2 GB; needs `gh auth login`)")

# Adopted text classifier: base recipe + Korean slang corpus + text carrying REAL Moonshine
# errors (2026-07-30). Measured against the previous _asraug model, recall@FPR15 on the
# profanity/slang set through real ASR: .525 -> .925 (clean), .400 -> .850 (SNR10); the
# original ko-harm testset also improved (.855 -> .902), so this is not a trade.
TEXT_MODEL_DIR = "artifacts/koelectra_small_harm_asraug_slang"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}\n  -> {_FETCH_HINT}")
    return path


# transformers treats a local dir WITHOUT config.json as a Hub repo id and then complains
# "Repo id must be in the form 'repo_name'..." — checking the contents turns that into a
# straight answer (an incomplete extraction of ckpt_final.tar).
_HF_DIR_FILES = ("config.json", "tokenizer_config.json")
_HF_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _hf_dir_problem(d: Path) -> str | None:
    """None when `d` is a usable local transformers model dir, else what is wrong."""
    if not d.exists():
        return "directory is missing"
    missing = [f for f in _HF_DIR_FILES if not (d / f).exists()]
    if not any((d / w).exists() for w in _HF_WEIGHTS):
        missing.append(f"one of {'/'.join(_HF_WEIGHTS)}")
    if missing:
        have = sorted(p.name for p in d.iterdir()) or ["(empty)"]
        return f"incomplete — missing {', '.join(missing)}; contains {', '.join(have)}"
    return None


def _require_hf_dir(d: Path, what: str) -> Path:
    problem = _hf_dir_problem(d)
    if problem:
        raise FileNotFoundError(f"{what} at {d}: {problem}\n  -> {_FETCH_HINT}")
    return d


def missing_artifacts(text: bool = True) -> list[str]:
    """Required-but-unusable model files, as human-readable lines (empty when ready to run)."""
    out = []
    ckpt = _ROOT / "ckpt_ced_mini_vio/best.ckpt"
    if not ckpt.exists():
        out.append(f"violence trigger checkpoint (CED-mini): {ckpt} — file is missing")
    d = _ROOT / TEXT_MODEL_DIR
    if text:
        problem = _hf_dir_problem(d)
        if problem:
            out.append(f"text classifier (KoELECTRA-small): {d} — {problem}")
    return out


def load_gate(size: str = "s1", ckpt: str | Path | None = None) -> torch.nn.Module:
    _add_paths()
    from student_models import TinyMelCNN
    m = TinyMelCNN(num_classes=4, widths=_STUDENT_WIDTHS[size], emb_dim=256)
    p = _require(Path(ckpt or _ROOT / f"distill/student_{size}.pt"),
                 f"distilled gate checkpoint ({size})")
    ck = torch.load(p, map_location="cpu", weights_only=True)  # ckpt is {"model", "tag"} only
    m.load_state_dict(ck["model"])
    return m.eval()


def load_trigger(ckpt: str | Path | None = None, int8: bool = True) -> torch.nn.Module:
    _add_paths()
    from train_ced_vio import CEDRawBackbone

    from models.harm_model import HarmModel, ModelConfig
    _set_quant_engine()
    # checked BEFORE the backbone downloads from HF, so a missing weight fails fast
    p = _require(Path(ckpt or _ROOT / "ckpt_ced_mini_vio/best.ckpt"),
                 "violence trigger checkpoint (CED-mini)")
    bb = CEDRawBackbone()
    m = HarmModel(4, ModelConfig(backbone="passthrough", backbone_out_dim=bb.out_dim))
    m.backbone = bb
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    if int8:
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    return m


class TextScorer:
    """KoELECTRA-small -> any-harm score (1 - P(safe)). CPU int8 by default."""

    def __init__(self, model_dir: str | Path | None = None, int8: bool = True):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        d = _require_hf_dir(Path(model_dir or _ROOT / TEXT_MODEL_DIR),
                            "text classifier (KoELECTRA-small + ASR aug + slang)")
        # local_files_only: never fall back to the Hub, so a broken local dir says so
        self.tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
        m = AutoModelForSequenceClassification.from_pretrained(
            str(d), local_files_only=True).eval()
        if int8:
            _set_quant_engine()
            m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        self.m = m
        import json
        cats = json.loads((d / "cats.json").read_text())
        self.safe_idx = cats.index("safe")

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            b = self.tok(texts[i:i + 32], truncation=True, max_length=128, padding=True,
                         return_tensors="pt")
            p = torch.softmax(self.m(**b).logits, -1)[:, self.safe_idx]
            out.append((1 - p).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def split_harm_text_units(text: str, window_tokens: int = 8) -> list[str]:
    """Keep the full transcript and add sentence/clause-sized KoELECTRA views.

    A single neutral lead-in can dilute a short abusive ending. Whisper segment newlines and
    punctuation are the primary boundaries; overlapping word windows cover weak punctuation.
    Very short synthetic suffixes are deliberately not created: KoELECTRA produced a strong
    false alarm for an ordinary four-word news suffix. Explicit short insults are handled by
    the high-confidence lexicon instead.
    """
    full = re.sub(r"[ \t]+", " ", (text or "").strip())
    if not full:
        return []
    units: list[str] = [full]
    sentences = re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", full)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        units.append(sentence)
        clauses = re.split(r"\s*(?:[,;:…]+|[-–—]\s+)\s*", sentence)
        for clause in clauses:
            clause = clause.strip()
            if clause:
                units.append(clause)
            words = clause.split()
            if len(words) > window_tokens:
                stride = max(2, window_tokens // 2)
                for start in range(0, len(words) - 1, stride):
                    chunk = words[start:start + window_tokens]
                    if len(chunk) == window_tokens:
                        units.append(" ".join(chunk))
    return list(dict.fromkeys(units))


class HybridTextScorer:
    """Classifier OR lexicon — whichever is more alarmed.

    Measured 2026-07-30: the classifier scores Korean *slang* near zero even when the ASR
    transcribed it correctly (슬롯 0.12, 스폰 0.33, 썰 0.27) because that vocabulary is absent
    from its training corpus. A bigger ASR does not fix that — whisper-small (9x the size)
    still only reached slang recall 0.50. The lexicon does, at zero model-size cost:
      - exact/phrase match on configs/text/harm_lexicon.yaml (ambiguous short words such as
        떨 / 조건 / 썰 only count inside a phrase, so ordinary speech does not fire)
      - jamo-level fuzzy match for ASR near-misses (야짤 -> "예짤")
    The final score is max(classifier, lexicon risk), so the lexicon can only ADD recall;
    its false-positive cost is measured on kor_unsmile clean like every other operating point.
    """

    def __init__(self, classifier: TextScorer | None = None, use_fuzzy: bool = True,
                 lexicon_path: str | Path | None = None, int8: bool = True):
        import sys
        if str(_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(_ROOT / "src"))
        from text.harm_text import load_lexicon, score_text
        self.clf = classifier if classifier is not None else TextScorer(int8=int8)
        self._score_text = score_text
        self.lexicon = load_lexicon(lexicon_path)
        self.use_fuzzy = use_fuzzy
        if use_fuzzy:
            from text.fuzzy_lexicon import fuzzy_harm_terms
            self._fuzzy = fuzzy_harm_terms
        # a recovered near-miss is worth slightly less than an exact hit
        self.fuzzy_weight = 0.85

    def lexicon_score(self, text: str) -> float:
        risk = float(self._score_text(text, self.lexicon).text_risk)
        if self.use_fuzzy and risk < 1.0:
            cats = self.lexicon["categories"]
            for hit in self._fuzzy(text):
                w = float(cats.get(hit.category, {}).get("weight", 1.0))
                risk = max(risk, self.fuzzy_weight * w)
        return min(1.0, risk)

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        groups = [split_harm_text_units(text) or [text] for text in texts]
        flat = [unit for group in groups for unit in group]
        classifier_scores = self.clf.score(flat)
        output = []
        offset = 0
        self.last_details = []
        for original, units in zip(texts, groups):
            end = offset + len(units)
            unit_scores = classifier_scores[offset:end]
            classifier_max = float(np.max(unit_scores)) if len(unit_scores) else 0.0
            lexicon_max = max((self.lexicon_score(unit) for unit in units), default=0.0)
            best_index = int(np.argmax(unit_scores)) if len(unit_scores) else 0
            best_unit = units[best_index] if units else original
            final = max(classifier_max, lexicon_max)
            output.append(final)
            self.last_details.append({
                "full_text": original,
                "best_unit": best_unit,
                "classifier_max": classifier_max,
                "lexicon_max": lexicon_max,
                "final": final,
                "units": len(units),
            })
            offset = end
        return np.asarray(output, dtype=np.float32)


# ---------- ASR ----------
#
# Chosen by "does the harmful word survive?", NOT by read-speech CER — CER picked
# moonshine-base-ko, which turned out to emit the literal string "audiotext" on 15% of the
# profanity/slang set. Measured 2026-07-30 (.autorun/compare_asr_harm.py, 40 rows,
# recall@FPR15 clean/SNR10):
#
#   model                        recall       profanity    slang        artifact  CPU RTF
#   moonshine-tiny-ko  (27M)     .53 / .40    .889 / .611  .23 / .23    0.00      0.04
#   moonshine-base-ko  (61.5M)   .43 / .43    .611 / .667  .27 / .23    0.15      0.05
#   whisper-large-v3-turbo int8  .55 / .60    .778 / .722  .36 / .50    0.00      1.28  (too slow)
#   whisper-small int8 (244M)    .65 / .68    .833 / .833  .50 / .55    0.00      0.32
#   whisper-tiny int8  (39M)     .38 / .33    .444 / .444  .32 / .23    0.00      0.09
#   whisper-base int8  (74M)     .50 / .58    .556 / .556  .46 / .59    0.00      0.13
#
# whisper-small measures best on every axis and stays real-time — but at ~250MB int8 it is
# over the trigger-tier budget (model_light §0: 10-90MB), i.e. PHONE-INFEASIBLE, so it is
# opt-in only. DEFAULT = moonshine-tiny: best profanity survival per parameter (.889 equals
# the perfect-transcript ceiling), zero artifacts, 27MB. Slang is unreachable for every
# in-budget ASR (<=.59) — that gap is closed on the TEXT side (HybridTextScorer + the
# slang-retrained classifier), not by a bigger ASR.
WHISPER_TINY = "whisper:tiny"
WHISPER_BASE = "whisper:base"
WHISPER_SMALL = "whisper:small"
WHISPER_TURBO = "whisper:large-v3-turbo"
VER1_DEMUCS_WHISPER_BASE = "ver1:htdemucs+whisper-base-fp16"
VER1_UVR_ONNX_WHISPER_BASE = "ver1:uvr-mdx-net-onnx+whisper-base-fp16"
MOONSHINE_TINY = "UsefulSensors/moonshine-tiny-ko"
MOONSHINE_BASE = "UsefulSensors/moonshine-base-ko"
MOONSHINE_DEFAULT = MOONSHINE_TINY       # 27M: best profanity survival per parameter
ASR_DEFAULT = MOONSHINE_TINY


def make_asr(model_id: str = ASR_DEFAULT, device: str | None = None, **options):
    """Build the ASR backend for `model_id` ("whisper:<size>" or a Moonshine HF id)."""
    if model_id == VER1_DEMUCS_WHISPER_BASE:
        from app.ver1_audio import DemucsWhisperBaseASR

        return DemucsWhisperBaseASR(**options)
    if model_id == VER1_UVR_ONNX_WHISPER_BASE:
        from app.ver1_audio import UVRMDXWhisperBaseASR

        return UVRMDXWhisperBaseASR(**options)
    if model_id.startswith("whisper:"):
        return WhisperASR(model_id.split(":", 1)[1], device)
    return MoonshineASR(device, model_id=model_id)


class WhisperASR:
    """faster-whisper (CTranslate2), int8. Default device is CPU: CUDA needs cuBLAS/cuDNN
    installed outside torch, and the client is a laptop anyway."""

    def __init__(self, size: str = "small", device: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is missing — install the asr group:\n"
                "  uv sync --group nlp --group asr\n"
                "(or run with --asr-model tiny to use Moonshine instead)") from e
        dev = device or "cpu"
        self.model_id = f"whisper:{size}"
        self.compute_type = "int8_float16" if dev == "cuda" else "int8"
        self.m = WhisperModel(size, device=dev, compute_type=self.compute_type)
        self.device = dev

    def transcribe(self, wav: np.ndarray) -> str:
        segs, _ = self.m.transcribe(np.asarray(wav, dtype=np.float32), language="ko",
                                    beam_size=5, vad_filter=False)
        return "".join(s.text for s in segs).strip()


class MoonshineASR:
    """Korean ASR. `model_id` is swappable so a bigger model can be traded for accuracy."""

    def __init__(self, device: str | None = None, model_id: str = MOONSHINE_DEFAULT):
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.m = MoonshineForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        # The checkpoint ships max_length=194, which collides with our per-second token
        # budget and makes transformers warn on every call. max_new_tokens is what we mean.
        if getattr(self.m, "generation_config", None) is not None:
            self.m.generation_config.max_length = None
        self.device = device

    # moonshine-base-ko emits these instead of a transcript on some utterances (measured:
    # 6/40 rows of the profanity/slang set, including "야 이 씨발놈아 당장 나와", where tiny-ko
    # transcribed fine). Treat them as "no transcript" so nothing downstream scores them.
    _ARTIFACTS = {"audiotext", "audio text", "audiotexte"}

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray) -> str:
        inp = self.proc(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        max_new = max(32, int(len(wav) / SR * 20))  # Korean ~20 tok/s (6.5 truncates)
        ids = self.m.generate(**inp, max_new_tokens=max_new)
        text = self.proc.decode(ids[0], skip_special_tokens=True)
        if text.strip().strip(".!?").lower() in self._ARTIFACTS:
            return ""
        return text


class CascadePipeline:
    """Per-clip cascade. Branches not loaded are skipped (score None)."""

    def __init__(self, thresholds: Thresholds, gate=None, trigger=None,
                 asr: MoonshineASR | None = None, text: TextScorer | None = None):
        self.thr = thresholds
        self.gate, self.trigger, self.asr, self.text = gate, trigger, asr, text

    @torch.no_grad()
    def _gate_score(self, wav_t: torch.Tensor) -> float:
        out = self.gate(wav_t, return_projection=False)
        return float(torch.sigmoid(out["logits"]).max())

    @torch.no_grad()
    def _trigger_prob(self, wav_t: torch.Tensor) -> float:
        out = self.trigger(wav_t, return_projection=False)
        return float(torch.sigmoid(out["logits"]).max())

    def process_clip(self, wav: np.ndarray, run_text: bool = False) -> ClipDecision:
        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        gate_enabled = self.gate is not None
        g = self._gate_score(wav_t) if gate_enabled else None
        a = None
        if self.trigger is not None and ((not gate_enabled) or g >= self.thr.gate):
            a = self._trigger_prob(wav_t)
        t_prob, transcript = None, None
        if run_text and self.asr is not None and self.text is not None:
            transcript = self.asr.transcribe(np.asarray(wav, dtype=np.float32))
            if transcript.strip():
                t_prob = float(self.text.score([transcript])[0])
        return decide(self.thr, g, a, t_prob, transcript, gate_enabled=gate_enabled)
