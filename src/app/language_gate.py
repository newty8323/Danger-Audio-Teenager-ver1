"""Recall-first Korean speech routing for the real-time text branch.

The acoustic violence trigger must keep seeing the original 10-second window.  This module
only narrows the audio sent to ASR:

    raw audio ─┬─ Silero VAD ───────────────────────────────┐
               └─ DeepFilterNet ─► Silero VAD ─► timestamp union
                                                    │
                                      raw speech ─► Wiener enhancement
                                                    │
                                                    └─ Whisper-tiny encoder LID
                                                       └─ recall-first temporal policy

Uncertain speech is retained.  Only sustained, high-confidence non-Korean evidence is
suppressed, because a missed Korean utterance is more costly than an extra ASR call.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SR = 16_000
_VAD_CHUNK = 512
_LID_CLIP_SAMPLES = 64_000
_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 80
_D_MODEL = 384
_ATTENTION_HEADS = 6
_FFN_DIM = 1_536
_MAX_POSITIONS = 1_500


class Enhancer(Protocol):
    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, dict]: ...


class VAD(Protocol):
    def __call__(self, waveform: np.ndarray) -> list[tuple[int, int]]: ...


class LID(Protocol):
    threshold: float

    def probability(self, waveform: np.ndarray) -> float: ...


@dataclass
class LanguageGateConfig:
    vad_threshold: float = 0.10
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    speech_pad_ms: int = 200
    min_group_sec: float = 2.0
    max_group_sec: float = 5.0
    max_join_gap_sec: float = 1.5
    min_language_evidence_sec: float = 2.0
    korean_accept: float | None = None
    non_korean_evidence_max: float = 0.15
    non_korean_enter_count: int = 2
    non_korean_switch_count: int = 3
    separator_ms: int = 200
    fail_open_on_no_speech: bool = True
    fail_open_on_error: bool = True


@dataclass
class SpeechGroup:
    start: int
    end: int
    spans: tuple[tuple[int, int], ...]
    raw: np.ndarray

    @property
    def voiced_sec(self) -> float:
        return sum(end - start for start, end in self.spans) / SR


@dataclass
class LanguageGateResult:
    audio: np.ndarray | None
    rows: list[dict]
    raw_spans: list[tuple[int, int]]
    enhanced_spans: list[tuple[int, int]]
    fused_spans: list[tuple[int, int]]
    enhancement: dict
    fail_open: bool = False
    fail_open_reason: str | None = None

    def metadata(self) -> dict:
        return {
            "raw_vad_sec": round(_span_seconds(self.raw_spans), 3),
            "enhanced_vad_sec": round(_span_seconds(self.enhanced_spans), 3),
            "fused_vad_sec": round(_span_seconds(self.fused_spans), 3),
            "groups": len(self.rows),
            "selected_groups": sum(row["selected"] for row in self.rows),
            "suppressed_groups": sum(not row["selected"] for row in self.rows),
            "decisions": [dict(row) for row in self.rows],
            "enhancement": dict(self.enhancement),
            "fail_open": self.fail_open,
            "fail_open_reason": self.fail_open_reason,
        }


class IdentityEnhancer:
    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, dict]:
        x = _float_audio(waveform)
        return x.copy(), _enhancement_metrics(x, x, "identity")


class WienerEnhancer:
    """Cross-platform fallback when a DeepFilterNet runtime is unavailable."""

    def __init__(self, strength: float = 1.0, noise_frame_ratio: float = 0.20,
                 gain_floor: float = 0.15):
        self.strength = strength
        self.noise_frame_ratio = noise_frame_ratio
        self.gain_floor = gain_floor

    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, dict]:
        x = torch.from_numpy(_float_audio(waveform))
        if x.numel() < 512 or float(x.square().mean().sqrt()) < 1e-8:
            y = x.numpy().copy()
            return y, _enhancement_metrics(x.numpy(), y, "wiener")
        n_fft, hop = 512, 128
        window = torch.hann_window(n_fft)
        spectrum = torch.stft(
            x, n_fft=n_fft, hop_length=hop, window=window, center=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        frame_power = power.mean(dim=0)
        count = max(1, round(frame_power.numel() * self.noise_frame_ratio))
        noise_idx = torch.topk(frame_power, count, largest=False).indices
        noise_psd = power[:, noise_idx].median(dim=1).values.unsqueeze(1) * self.strength
        speech_psd = (power - noise_psd).clamp_min(0.0)
        gain = speech_psd / (speech_psd + noise_psd + 1e-10)
        gain = F.avg_pool2d(
            gain[None, None], kernel_size=(3, 5), stride=1, padding=(1, 2)
        )[0, 0]
        gain = gain.clamp(self.gain_floor, 1.0)
        y = torch.istft(
            spectrum * gain, n_fft=n_fft, hop_length=hop, window=window,
            center=True, length=x.numel(),
        ).clamp(-1.0, 1.0).numpy()
        metrics = _enhancement_metrics(x.numpy(), y, "wiener")
        metrics["mean_gain"] = round(float(gain.mean()), 6)
        return y.astype(np.float32, copy=False), metrics


class DeepFilterCliEnhancer:
    """DeepFilterNet command-line adapter used before VAD.

    The official runtime consumes 48 kHz WAV.  A temporary file is used per ASR duty-cycle
    slice; production mobile ports should replace this adapter with a streaming native API.
    """

    def __init__(self, executable: str | Path, attenuation_limit_db: float = 20.0):
        self.executable = Path(executable)
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.attenuation_limit_db = attenuation_limit_db

    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, dict]:
        import torchaudio.functional as AF

        x = _float_audio(waveform)
        started = time.perf_counter()
        x48 = AF.resample(torch.from_numpy(x), SR, 48_000).numpy()
        with tempfile.TemporaryDirectory(prefix="language_gate_deepfilter_") as tmp:
            root = Path(tmp)
            input_path = root / "input_48k.wav"
            output_dir = root / "output"
            output_dir.mkdir()
            _write_pcm16(input_path, x48, 48_000)
            completed = subprocess.run(
                [
                    str(self.executable),
                    "--compensate-delay",
                    "--atten-lim-db",
                    str(self.attenuation_limit_db),
                    "--output-dir",
                    str(output_dir),
                    str(input_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            candidates = list(output_dir.glob("*.wav"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one DeepFilterNet output, found {candidates}")
            y48, output_sr = _read_pcm_wav(candidates[0])
            if output_sr != 48_000:
                raise ValueError(f"unexpected DeepFilterNet sample rate: {output_sr}")
        y = AF.resample(torch.from_numpy(y48), 48_000, SR).numpy()
        y = _match_length(y, len(x))
        metrics = _enhancement_metrics(x, y, "deepfilternet")
        metrics.update({
            "attenuation_limit_db": self.attenuation_limit_db,
            "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
            "stderr_tail": completed.stderr[-300:],
        })
        return y, metrics


class SileroVAD:
    """Self-contained TorchScript Silero VAD adapter (16 kHz, offline slice mode)."""

    def __init__(self, model_path: str | Path, threshold: float = 0.10,
                 min_speech_ms: int = 250, min_silence_ms: int = 300,
                 speech_pad_ms: int = 200):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.threshold = threshold
        self.min_speech = int(SR * min_speech_ms / 1000)
        self.min_silence = int(SR * min_silence_ms / 1000)
        self.pad = int(SR * speech_pad_ms / 1000)
        self._lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> list[tuple[int, int]]:
        x = torch.from_numpy(_float_audio(waveform))
        if x.numel() == 0:
            return []
        probs: list[float] = []
        with self._lock:
            self.model.reset_states()
            for start in range(0, x.numel(), _VAD_CHUNK):
                chunk = x[start:start + _VAD_CHUNK]
                if chunk.numel() < _VAD_CHUNK:
                    chunk = F.pad(chunk, (0, _VAD_CHUNK - chunk.numel()))
                probs.append(float(self.model(chunk, SR).item()))
        return _timestamps_from_probs(
            probs, len(x), self.threshold, self.min_speech, self.min_silence, self.pad
        )


class _WhisperAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(_D_MODEL, _D_MODEL, bias=False)
        self.v_proj = nn.Linear(_D_MODEL, _D_MODEL)
        self.q_proj = nn.Linear(_D_MODEL, _D_MODEL)
        self.out_proj = nn.Linear(_D_MODEL, _D_MODEL)
        self.head_dim = _D_MODEL // _ATTENTION_HEADS
        self.scaling = self.head_dim**-0.5

    def _shape(self, tensor: Tensor) -> Tensor:
        batch, frames, _ = tensor.shape
        return tensor.view(batch, frames, _ATTENTION_HEADS, self.head_dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> Tensor:
        query = self._shape(self.q_proj(hidden)) * self.scaling
        key = self._shape(self.k_proj(hidden))
        value = self._shape(self.v_proj(hidden))
        weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)).float(), dim=-1)
        attended = torch.matmul(weights.to(value.dtype), value)
        attended = attended.transpose(1, 2).contiguous().view(hidden.shape)
        return self.out_proj(attended)


class _WhisperEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _WhisperAttention()
        self.self_attn_layer_norm = nn.LayerNorm(_D_MODEL)
        self.fc1 = nn.Linear(_D_MODEL, _FFN_DIM)
        self.fc2 = nn.Linear(_FFN_DIM, _D_MODEL)
        self.final_layer_norm = nn.LayerNorm(_D_MODEL)

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = residual + self.self_attn(self.self_attn_layer_norm(hidden))
        residual = hidden
        hidden = self.final_layer_norm(hidden)
        return residual + self.fc2(F.gelu(self.fc1(hidden)))


class _WhisperTinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(_N_MELS, _D_MODEL, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(_D_MODEL, _D_MODEL, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(_MAX_POSITIONS, _D_MODEL)
        self.layers = nn.ModuleList(_WhisperEncoderLayer() for _ in range(4))
        self.layer_norm = nn.LayerNorm(_D_MODEL)

    def forward(self, log_mel: Tensor) -> Tensor:
        hidden = F.gelu(self.conv1(log_mel))
        hidden = F.gelu(self.conv2(hidden)).transpose(1, 2)
        if hidden.shape[1] > _MAX_POSITIONS:
            raise ValueError(f"too many encoder frames: {hidden.shape[1]}")
        hidden = hidden + self.embed_positions.weight[:hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


class _WhisperLogMel(nn.Module):
    def __init__(self, mel_filters: Tensor) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(_N_FFT), persistent=False)
        self.register_buffer("mel_filters", mel_filters.float(), persistent=True)

    def forward(self, waveform: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):
            stft = torch.stft(
                waveform.float(), _N_FFT, _HOP_LENGTH, window=self.window,
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs().square()
            log_spec = torch.clamp(torch.matmul(self.mel_filters, magnitudes), min=1e-10)
            log_spec = log_spec.log10()
            log_spec = torch.maximum(
                log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0
            )
            return (log_spec + 4.0) / 4.0


class _LinearHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(_D_MODEL * 2), nn.Linear(_D_MODEL * 2, 1)
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.layers(embedding).squeeze(1)


class _MlpHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(_D_MODEL * 2), nn.Linear(_D_MODEL * 2, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.layers(embedding).squeeze(1)


class WhisperKoreanLID:
    """Frozen Whisper-tiny encoder plus the trained Korean/non-Korean head."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "whisper-tiny-encoder-binary-lid-v1":
            raise ValueError("unsupported compact LID checkpoint format")
        self.device = _choose_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.threshold = float(checkpoint["threshold"])
        self.encoder = _WhisperTinyEncoder()
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        self.head = _LinearHead() if checkpoint["head_type"] == "linear" else _MlpHead()
        self.head.load_state_dict(checkpoint["head_state"], strict=True)
        self.frontend = _WhisperLogMel(checkpoint["mel_filters"].float())
        self.encoder = self.encoder.to(self.device, dtype=self.dtype).eval()
        self.head = self.head.to(self.device, dtype=torch.float32).eval()
        self.frontend = self.frontend.to(self.device).eval()

    @torch.inference_mode()
    def probability(self, waveform: np.ndarray) -> float:
        x = torch.from_numpy(_float_audio(waveform))
        windows = _lid_windows(x)
        batch = torch.stack(windows).to(self.device)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            hidden = self.encoder(self.frontend(batch).to(self.dtype))
        embedding = torch.cat((hidden.float().mean(1), hidden.float().std(1)), dim=1)
        return float(torch.sigmoid(self.head(embedding)).mean().cpu())


class RecallFirstPolicy:
    """Keep uncertainty; suppress only sustained, very strong non-Korean evidence."""

    def __init__(self, cfg: LanguageGateConfig):
        self.cfg = cfg
        self.state = "unknown"
        self.non_korean_streak = 0

    def decide(
        self, probability: float, voiced_sec: float, korean_accept: float
    ) -> tuple[bool, str]:
        if probability >= korean_accept:
            self.state = "korean"
            self.non_korean_streak = 0
            return True, "korean"
        strong_non_korean = (
            voiced_sec >= self.cfg.min_language_evidence_sec
            and probability <= self.cfg.non_korean_evidence_max
        )
        if not strong_non_korean:
            self.non_korean_streak = 0
            return True, "uncertain_recall_first"
        self.non_korean_streak += 1
        required = (
            self.cfg.non_korean_switch_count
            if self.state == "korean"
            else self.cfg.non_korean_enter_count
        )
        if self.non_korean_streak >= required:
            self.state = "non_korean"
            return False, f"sustained_non_korean_x{self.non_korean_streak}"
        return True, f"non_korean_pending_x{self.non_korean_streak}"


class KoreanLanguageGate:
    def __init__(self, vad: VAD, lid: LID,
                 pre_vad_enhancer: Enhancer | None = None,
                 speech_enhancer: Enhancer | None = None,
                 cfg: LanguageGateConfig | None = None):
        self.cfg = cfg or LanguageGateConfig()
        self.vad = vad
        self.lid = lid
        self.pre_vad_enhancer = pre_vad_enhancer or WienerEnhancer()
        self.speech_enhancer = speech_enhancer or WienerEnhancer()
        self.policy = RecallFirstPolicy(self.cfg)

    @classmethod
    def from_paths(cls, vad_path: str | Path, checkpoint_path: str | Path,
                   deepfilter_exe: str | Path | None = None, device: str = "auto",
                   cfg: LanguageGateConfig | None = None) -> "KoreanLanguageGate":
        config = cfg or LanguageGateConfig()
        vad = SileroVAD(
            vad_path,
            threshold=config.vad_threshold,
            min_speech_ms=config.min_speech_ms,
            min_silence_ms=config.min_silence_ms,
            speech_pad_ms=config.speech_pad_ms,
        )
        lid = WhisperKoreanLID(checkpoint_path, device=device)
        pre_vad_enhancer: Enhancer = (
            DeepFilterCliEnhancer(deepfilter_exe) if deepfilter_exe else WienerEnhancer()
        )
        return cls(
            vad=vad,
            lid=lid,
            pre_vad_enhancer=pre_vad_enhancer,
            speech_enhancer=WienerEnhancer(),
            cfg=config,
        )

    def route(self, waveform: np.ndarray) -> LanguageGateResult:
        raw = _float_audio(waveform)
        try:
            return self._route(raw)
        except Exception as exc:
            if not self.cfg.fail_open_on_error:
                raise
            return LanguageGateResult(
                audio=raw.copy(),
                rows=[],
                raw_spans=[],
                enhanced_spans=[],
                fused_spans=[],
                enhancement={
                    "method": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                fail_open=True,
                fail_open_reason=f"{type(exc).__name__}: {exc}",
            )

    def _route(self, raw: np.ndarray) -> LanguageGateResult:
        pre_vad_audio, pre_vad_metrics = self.pre_vad_enhancer(raw)
        pre_vad_audio = _match_length(_float_audio(pre_vad_audio), len(raw))
        pre_vad_method = pre_vad_metrics.get("method", "unknown")
        enhancement = {
            "method": f"{pre_vad_method}->not_run",
            "pre_vad": dict(pre_vad_metrics),
            "speech_groups": [],
        }
        raw_spans = self.vad(raw)
        enhanced_spans = self.vad(pre_vad_audio)
        fused = merge_spans(raw_spans, enhanced_spans)
        if not fused:
            fail_open = self.cfg.fail_open_on_no_speech and _rms(raw) >= 1e-4
            return LanguageGateResult(
                audio=pre_vad_audio if fail_open else None,
                rows=[],
                raw_spans=raw_spans,
                enhanced_spans=enhanced_spans,
                fused_spans=fused,
                enhancement=enhancement,
                fail_open=fail_open,
                fail_open_reason="vad_empty_non_silent" if fail_open else None,
            )
        groups = make_groups(raw, fused, self.cfg)
        korean_accept = (
            float(self.lid.threshold)
            if self.cfg.korean_accept is None
            else self.cfg.korean_accept
        )
        rows: list[dict] = []
        selected: list[np.ndarray] = []
        for index, group in enumerate(groups):
            # DeepFilterNet locates speech. The established extraction path then enhances the
            # selected RAW speech with Wiener filtering before LID and ASR.
            speech_enhanced, speech_metrics = self.speech_enhancer(group.raw)
            speech_enhanced = _match_length(speech_enhanced, len(group.raw))
            enhancement["speech_groups"].append(dict(speech_metrics))
            enhancement["method"] = (
                f"{pre_vad_method}->{speech_metrics.get('method', 'unknown')}"
            )
            # Max fusion protects Korean recall if post-VAD enhancement helps one domain but
            # hurts another.
            raw_probability = float(self.lid.probability(group.raw))
            enhanced_probability = float(self.lid.probability(speech_enhanced))
            probability = max(raw_probability, enhanced_probability)
            keep, reason = self.policy.decide(
                probability, group.voiced_sec, korean_accept
            )
            rows.append({
                "group": index,
                "start_sec": round(group.start / SR, 3),
                "end_sec": round(group.end / SR, 3),
                "voiced_sec": round(group.voiced_sec, 3),
                "korean_probability": round(probability, 6),
                "raw_korean_probability": round(raw_probability, 6),
                "enhanced_korean_probability": round(enhanced_probability, 6),
                "speech_enhancement": dict(speech_metrics),
                "selected": keep,
                "reason": reason,
                "state": self.policy.state,
            })
            if keep:
                selected.append(speech_enhanced)
        audio = _join_audio(selected, self.cfg.separator_ms)
        return LanguageGateResult(
            audio=audio,
            rows=rows,
            raw_spans=raw_spans,
            enhanced_spans=enhanced_spans,
            fused_spans=fused,
            enhancement=enhancement,
        )


def merge_spans(*sets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(
        (max(0, int(start)), max(0, int(end)))
        for spans in sets for start, end in spans if end > start
    )
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def make_groups(raw: np.ndarray, spans: list[tuple[int, int]],
                cfg: LanguageGateConfig) -> list[SpeechGroup]:
    groups: list[SpeechGroup] = []
    current: list[tuple[int, int]] = []
    voiced = 0
    max_samples = max(1, int(cfg.max_group_sec * SR))
    min_samples = max(1, int(cfg.min_group_sec * SR))
    max_gap = int(cfg.max_join_gap_sec * SR)

    def flush() -> None:
        nonlocal current, voiced
        if not current:
            return
        groups.append(_build_group(raw, current))
        current, voiced = [], 0

    previous_end: int | None = None
    for span_start, span_end in spans:
        start, end = max(0, span_start), min(len(raw), span_end)
        if end <= start:
            continue
        if previous_end is not None and start - previous_end > max_gap:
            flush()
        cursor = start
        while cursor < end:
            room = max_samples - voiced
            take_end = min(end, cursor + room)
            current.append((cursor, take_end))
            voiced += take_end - cursor
            cursor = take_end
            if voiced >= max_samples:
                flush()
        if voiced >= min_samples:
            flush()
        previous_end = end
    flush()
    return groups


def _build_group(raw: np.ndarray, spans: list[tuple[int, int]]) -> SpeechGroup:
    raw_parts = [raw[start:end] for start, end in spans]
    return SpeechGroup(
        start=spans[0][0],
        end=spans[-1][1],
        spans=tuple(spans),
        raw=np.concatenate(raw_parts).astype(np.float32, copy=False),
    )


def _timestamps_from_probs(probs: list[float], audio_len: int, threshold: float,
                           min_speech: int, min_silence: int,
                           pad: int) -> list[tuple[int, int]]:
    neg_threshold = max(threshold - 0.15, 0.01)
    triggered = False
    start = 0
    temp_end: int | None = None
    spans: list[tuple[int, int]] = []
    for index, prob in enumerate(probs):
        sample = index * _VAD_CHUNK
        if prob >= threshold:
            if not triggered:
                triggered, start = True, sample
            temp_end = None
            continue
        if triggered and prob < neg_threshold:
            temp_end = sample if temp_end is None else temp_end
            if sample - temp_end >= min_silence:
                if temp_end - start >= min_speech:
                    spans.append((start, temp_end))
                triggered, temp_end = False, None
    if triggered and audio_len - start >= min_speech:
        spans.append((start, audio_len))
    padded = [(max(0, start - pad), min(audio_len, end + pad)) for start, end in spans]
    return merge_spans(padded)


def _lid_windows(waveform: Tensor) -> list[Tensor]:
    if waveform.numel() < _LID_CLIP_SAMPLES:
        output = torch.zeros(_LID_CLIP_SAMPLES)
        offset = (_LID_CLIP_SAMPLES - waveform.numel()) // 2
        output[offset:offset + waveform.numel()] = waveform
        return [output]
    if waveform.numel() == _LID_CLIP_SAMPLES:
        return [waveform]
    return [waveform[:_LID_CLIP_SAMPLES], waveform[-_LID_CLIP_SAMPLES:]]


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _join_audio(parts: list[np.ndarray], separator_ms: int) -> np.ndarray | None:
    if not parts:
        return None
    separator = np.zeros(int(SR * separator_ms / 1000), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index:
            joined.append(separator)
        joined.append(part)
    return np.concatenate(joined).astype(np.float32, copy=False)


def _enhancement_metrics(raw: np.ndarray, enhanced: np.ndarray, method: str) -> dict:
    x = _float_audio(raw)
    y = _match_length(_float_audio(enhanced), len(x))
    removed = x - y
    correlation = 1.0
    if len(x) > 1 and float(np.std(x)) > 1e-8 and float(np.std(y)) > 1e-8:
        correlation = float(np.corrcoef(x, y)[0, 1])
    input_rms, output_rms = _rms(x), _rms(y)
    return {
        "method": method,
        "input_rms": round(input_rms, 8),
        "output_rms": round(output_rms, 8),
        "rms_change_db": round(
            20 * math.log10((output_rms + 1e-10) / (input_rms + 1e-10)), 4
        ),
        "removed_rms": round(_rms(removed), 8),
        "correlation": round(correlation, 6),
        "clipping_ratio": round(float(np.mean(np.abs(y) >= 0.999)), 8),
    }


def _float_audio(waveform: np.ndarray) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def _match_length(waveform: np.ndarray, length: int) -> np.ndarray:
    x = _float_audio(waveform)
    if len(x) >= length:
        return x[:length].copy()
    return np.pad(x, (0, length - len(x))).astype(np.float32)


def _rms(waveform: np.ndarray) -> float:
    x = _float_audio(waveform)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _span_seconds(spans: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in spans) / SR


def _write_pcm16(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate
