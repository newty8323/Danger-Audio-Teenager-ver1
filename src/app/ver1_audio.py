"""Ver1 speech branch: original HTDemucs -> silence compaction -> Whisper Base FP16.

The acoustic branch must continue to receive the untouched waveform.  This module is only
the ASR backend used by the text branch.  It deliberately uses subprocesses so Demucs and
whisper.cpp release their large working buffers after every duty-cycle run; that matches the
planned sequential mobile execution and gives us a clean boundary for the later ONNX port.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np

from app.escalate import write_wav

SR = 16_000
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMUCS_PACKAGES = ROOT / "vendor/demucs-python"
DEFAULT_DEMUCS_REPO = ROOT / "artifacts/ver1/demucs"
DEFAULT_QUANTIZED_DEMUCS_REPO = ROOT / "artifacts/ver1/demucs-quantized"
DEFAULT_WHISPER_CLI = ROOT / "artifacts/ver1/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = ROOT / "artifacts/ver1/whisper/ggml-base.bin"
DEFAULT_QUANTIZED_WHISPER_MODEL = ROOT / "artifacts/ver1/whisper/ggml-base-q6_k.bin"


def _normalised_token(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"[^가-힣a-z0-9]", "", text)


def remove_leading_overlap(previous_text: str, current_text: str) -> str:
    """Remove exact token overlap introduced by a context-recovery retry."""
    previous = previous_text.split()
    current = current_text.split()
    for size in range(min(len(previous), len(current), 20), 0, -1):
        left = [_normalised_token(token) for token in previous[-size:]]
        right = [_normalised_token(token) for token in current[:size]]
        if left == right and all(left):
            return " ".join(current[size:]).strip()
    return current_text.strip()


def compact_audible_audio(
    wave: np.ndarray,
    threshold_db: float = -50.0,
    frame_ms: int = 40,
    min_sound_ms: int = 160,
    keep_gap_ms: int = 280,
    pad_ms: int = 120,
    separator_ms: int = 120,
) -> dict:
    """Remove long quiet gaps from a Demucs vocal stem without chopping words."""
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not x.size:
        return {"audio": x, "spans": [], "kept_seconds": 0.0,
                "original_seconds": 0.0, "removed_seconds": 0.0}

    frame_n = max(1, int(SR * frame_ms / 1000))
    padded = np.pad(x, (0, (-len(x)) % frame_n))
    frames = padded.reshape(-1, frame_n)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    active = 20.0 * np.log10(rms + 1e-8) > float(threshold_db)

    max_gap = max(0, int(np.ceil(keep_gap_ms / frame_ms)))
    inactive_start = None
    for index, is_active in enumerate(np.r_[active, True]):
        if not is_active and inactive_start is None:
            inactive_start = index
        elif is_active and inactive_start is not None:
            if index - inactive_start <= max_gap:
                active[inactive_start:index] = True
            inactive_start = None

    min_frames = max(1, int(np.ceil(min_sound_ms / frame_ms)))
    pad_frames = max(0, int(np.ceil(pad_ms / frame_ms)))
    spans: list[tuple[int, int]] = []
    run_start = None
    for index, is_active in enumerate(np.r_[active, False]):
        if is_active and run_start is None:
            run_start = index
        elif not is_active and run_start is not None:
            if index - run_start >= min_frames:
                start = max(0, run_start - pad_frames) * frame_n
                end = min(len(active), index + pad_frames) * frame_n
                span = (start, min(len(x), end))
                if spans and span[0] <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(spans[-1][1], span[1]))
                else:
                    spans.append(span)
            run_start = None

    separator = np.zeros(int(SR * separator_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, (start, end) in enumerate(spans):
        if index:
            parts.append(separator)
        parts.append(x[start:end])
    audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    kept_seconds = sum(end - start for start, end in spans) / SR
    original_seconds = len(x) / SR
    return {
        "audio": audio,
        "spans": spans,
        "kept_seconds": kept_seconds,
        "original_seconds": original_seconds,
        "removed_seconds": max(0.0, original_seconds - kept_seconds),
    }


class DemucsWhisperBaseASR:
    """Stateful Ver1 ASR backend with one-shot recovery for blank boundary windows."""

    model_id = "ver1:htdemucs+whisper-base-fp16"

    def __init__(
        self,
        demucs_packages: str | Path = DEFAULT_DEMUCS_PACKAGES,
        demucs_repo: str | Path = DEFAULT_DEMUCS_REPO,
        demucs_model_name: str = "htdemucs",
        whisper_cli: str | Path = DEFAULT_WHISPER_CLI,
        whisper_model: str | Path = DEFAULT_WHISPER_MODEL,
        silence_threshold_db: float = -50.0,
        language: str = "ko",
        context_seconds: float = 2.0,
        demucs_device: str = "auto",
    ):
        self.demucs_packages = Path(demucs_packages)
        self.demucs_repo = Path(demucs_repo)
        self.demucs_model_name = demucs_model_name
        self.whisper_cli = Path(whisper_cli)
        self.whisper_model = Path(whisper_model)
        self.silence_threshold_db = float(silence_threshold_db)
        self.language = language
        self.demucs_device = demucs_device
        self.model_id = (
            "ver1:htdemucs+whisper-base-fp16"
            if self.demucs_model_name == "htdemucs" and self.whisper_model.name == "ggml-base.bin"
            else f"ver1:{self.demucs_model_name}+{self.whisper_model.stem}"
        )
        self.context_samples = max(0, int(float(context_seconds) * SR))
        self._previous_audio = np.zeros(0, dtype=np.float32)
        self._previous_text = ""
        self.last_metrics: dict = {}
        self._validate()
        self._load_demucs()
        self._warmup_demucs()

    def _validate(self) -> None:
        required = {
            "vendored Demucs packages": self.demucs_packages / "demucs/__init__.py",
            "whisper.cpp CLI": self.whisper_cli,
            "Whisper model": self.whisper_model,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
        if self.demucs_model_name == "htdemucs":
            demucs_files = [self.demucs_repo / "htdemucs.yaml",
                            self.demucs_repo / "955717e8-8726e21a.th"]
        else:
            demucs_files = list(self.demucs_repo.glob(f"{self.demucs_model_name}-*.th"))
        if not demucs_files or not all(path.is_file() for path in demucs_files):
            missing.append(
                f"Demucs model {self.demucs_model_name}: {self.demucs_repo}"
            )
        if missing:
            raise FileNotFoundError("Ver1 speech artifacts are missing:\n  - " + "\n  - ".join(missing))

    def _load_demucs(self) -> None:
        """Load HTDemucs once; the old implementation reloaded it for every 10 s window."""
        if str(self.demucs_packages) not in sys.path:
            sys.path.insert(0, str(self.demucs_packages))
        import torch
        from demucs.pretrained import get_model

        started = time.perf_counter()
        self._demucs_model = get_model(
            self.demucs_model_name, repo=self.demucs_repo
        ).eval()
        if self.demucs_device == "auto":
            if torch.backends.mps.is_available():
                self._demucs_device = "mps"
            elif torch.cuda.is_available():
                self._demucs_device = "cuda"
            else:
                self._demucs_device = "cpu"
        else:
            self._demucs_device = self.demucs_device
        self._demucs_model.to(self._demucs_device)
        self._demucs_load_seconds = time.perf_counter() - started

    def _warmup_demucs(self) -> None:
        """Compile/cache the accelerator path before the first real utterance arrives."""
        self._demucs_warmup_seconds = 0.0
        if self._demucs_device not in {"mps", "cuda"}:
            return
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="danger-audio-ver1-warmup-") as directory:
            self._demucs_vocals(np.zeros(SR, dtype=np.float32), Path(directory))
        self._demucs_warmup_seconds = time.perf_counter() - started

    def _demucs_vocals(self, wave: np.ndarray, root: Path) -> tuple[np.ndarray, float]:
        del root  # kept in the signature so ONNX and subprocess backends share one boundary
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        model = self._demucs_model
        mix = torch.from_numpy(np.asarray(wave, dtype=np.float32)).reshape(1, -1)
        mix = convert_audio(mix, SR, model.samplerate, model.audio_channels)
        reference = mix.mean(0)
        mean = reference.mean()
        std = reference.std().clamp_min(1e-8)
        mix = (mix - mean) / std
        started = time.perf_counter()
        separated = apply_model(
            model, mix[None], device=self._demucs_device, shifts=0, split=True,
            overlap=0.1, progress=False, num_workers=0, segment=7,
        )[0]
        if self._demucs_device == "mps":
            torch.mps.synchronize()
        elif self._demucs_device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        separated = separated * std + mean
        vocals = separated[model.sources.index("vocals")].detach().cpu()
        vocals = convert_audio(vocals, model.samplerate, SR, 1)[0]
        return vocals.numpy().astype(np.float32, copy=False), elapsed

    def _whisper(self, wave: np.ndarray, root: Path, name: str) -> tuple[str, float]:
        if not wave.size:
            return "", 0.0
        path = root / f"{name}.wav"
        write_wav(path, wave, SR)
        command = [
            str(self.whisper_cli), "-m", str(self.whisper_model), "-f", str(path),
            "-l", self.language, "-nt", "-np",
        ]
        started = time.perf_counter()
        # whisper.cpp occasionally ends a streamed Korean line between UTF-8 bytes. Decode
        # defensively so one truncated final glyph cannot terminate the live capture loop.
        completed = subprocess.run(command, capture_output=True)
        elapsed = time.perf_counter() - started
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode:
            raise RuntimeError(
                "Whisper inference failed:\n" + (stderr or stdout)[-3000:]
            )
        lines = []
        for line in stdout.splitlines():
            cleaned = re.sub(r"^\s*-\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)
        # whisper.cpp prints one decoded segment per line. Preserve that boundary instead
        # of flattening a whole 10 s window into one visually and semantically long sentence.
        return "\n".join(lines).strip(), elapsed

    def transcribe(self, wav: np.ndarray) -> str:
        original = np.asarray(wav, dtype=np.float32).reshape(-1)
        with tempfile.TemporaryDirectory(prefix="danger-audio-ver1-") as directory:
            root = Path(directory)
            vocals, demucs_seconds = self._demucs_vocals(original, root)
            compacted = compact_audible_audio(vocals, self.silence_threshold_db)
            asr_audio = compacted["audio"]
            transcript, whisper_seconds = self._whisper(asr_audio, root, "asr-input")

            recovered = False
            recovery_seconds = 0.0
            raw_recovery = ""
            if not transcript and self._previous_audio.size and asr_audio.size:
                context = self._previous_audio[-self.context_samples:]
                recovery_audio = np.concatenate([context, asr_audio]).astype(np.float32)
                raw_recovery, recovery_seconds = self._whisper(
                    recovery_audio, root, "asr-recovery-input"
                )
                transcript = remove_leading_overlap(self._previous_text, raw_recovery)
                recovered = bool(transcript)

        self._previous_audio = asr_audio.copy()
        if transcript:
            self._previous_text = transcript
        self.last_metrics = {
            "backend": self.model_id,
            "input_seconds": len(original) / SR,
            "vocals_seconds": len(vocals) / SR,
            "kept_seconds": compacted["kept_seconds"],
            "removed_seconds": compacted["removed_seconds"],
            "demucs_seconds": demucs_seconds,
            "demucs_load_seconds": self._demucs_load_seconds,
            "demucs_warmup_seconds": self._demucs_warmup_seconds,
            "demucs_device": self._demucs_device,
            "whisper_seconds": whisper_seconds + recovery_seconds,
            "context_recovered": recovered,
            "raw_recovery_transcript": raw_recovery,
        }
        return transcript


class UVRMDXWhisperBaseASR(DemucsWhisperBaseASR):
    """Live ASR branch using the ONNX UVR MDX-Net separator instead of Demucs.

    This is intentionally an opt-in experiment.  The acoustic branch still sees
    the original 16 kHz waveform; only the ASR branch is separated.
    """

    model_id = "ver1:uvr-mdx-net-onnx+whisper-base-fp16"

    def __init__(
        self,
        uvr_model: str | Path = ROOT / "artifacts/onnx/uvr_mdxnet_3_9662/UVR_MDXNET_3_9662.onnx",
        whisper_cli: str | Path = DEFAULT_WHISPER_CLI,
        whisper_model: str | Path = DEFAULT_WHISPER_MODEL,
        silence_threshold_db: float = -50.0,
        language: str = "ko",
        context_seconds: float = 2.0,
    ):
        self.uvr_model = Path(uvr_model)
        self.whisper_cli = Path(whisper_cli)
        self.whisper_model = Path(whisper_model)
        self.silence_threshold_db = float(silence_threshold_db)
        self.language = language
        self.context_samples = max(0, int(float(context_seconds) * SR))
        self._previous_audio = np.zeros(0, dtype=np.float32)
        self._previous_text = ""
        self.last_metrics: dict = {}
        missing = [path for path in (self.uvr_model, self.whisper_cli, self.whisper_model)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError("UVR/Whisper artifacts are missing:\n  - " +
                                    "\n  - ".join(map(str, missing)))
        # The reusable separator lives in scripts so file experiments and live
        # capture run exactly the same ONNX pre/post-processing.
        from scripts.uvr_mdx_onnx_separate import SAMPLE_RATE, UVRMDXNet
        self._uvr_sample_rate = SAMPLE_RATE
        self._uvr = UVRMDXNet(self.uvr_model)

    def _uvr_vocals(self, wave: np.ndarray) -> tuple[np.ndarray, float]:
        import librosa

        stereo = librosa.resample(np.asarray(wave, dtype=np.float32), orig_sr=SR,
                                  target_sr=self._uvr_sample_rate)
        mix = np.stack((stereo, stereo))
        started = time.perf_counter()
        vocals = self._uvr.separate(mix)
        elapsed = time.perf_counter() - started
        mono = np.mean(vocals, axis=0)
        return librosa.resample(mono, orig_sr=self._uvr_sample_rate,
                                target_sr=SR).astype(np.float32), elapsed

    def transcribe(self, wav: np.ndarray) -> str:
        original = np.asarray(wav, dtype=np.float32).reshape(-1)
        with tempfile.TemporaryDirectory(prefix="danger-audio-uvr-") as directory:
            root = Path(directory)
            vocals, separator_seconds = self._uvr_vocals(original)
            compacted = compact_audible_audio(vocals, self.silence_threshold_db)
            asr_audio = compacted["audio"]
            transcript, whisper_seconds = self._whisper(asr_audio, root, "asr-input")

            recovered = False
            recovery_seconds = 0.0
            raw_recovery = ""
            if not transcript and self._previous_audio.size and asr_audio.size:
                recovery_audio = np.concatenate(
                    [self._previous_audio[-self.context_samples:], asr_audio]
                ).astype(np.float32)
                raw_recovery, recovery_seconds = self._whisper(
                    recovery_audio, root, "asr-recovery-input"
                )
                transcript = remove_leading_overlap(self._previous_text, raw_recovery)
                recovered = bool(transcript)

        self._previous_audio = asr_audio.copy()
        if transcript:
            self._previous_text = transcript
        self.last_metrics = {
            "backend": self.model_id,
            "input_seconds": len(original) / SR,
            "vocals_seconds": len(vocals) / SR,
            "kept_seconds": compacted["kept_seconds"],
            "removed_seconds": compacted["removed_seconds"],
            "uvr_onnx_seconds": separator_seconds,
            "uvr_providers": self._uvr.providers,
            "whisper_seconds": whisper_seconds + recovery_seconds,
            "context_recovered": recovered,
            "raw_recovery_transcript": raw_recovery,
        }
        return transcript
