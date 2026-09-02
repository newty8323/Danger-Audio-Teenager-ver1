"""On-device app entry point — watch what this machine PLAYS and flag harmful content.

MacBook (primary target, Apple Silicon, macOS 14.2+):
    uv run --group nlp python -m app.main --server http://<linux-host>:8770/

Linux dev box (same code path, PipeWire loopback):
    uv run --group nlp python -m app.main --source pipewire

Offline replay of a file (no capture backend needed):
    uv run --group nlp python -m app.main --source file --file some.wav --realtime

Then open http://127.0.0.1:8765 . Ctrl-C stops.
"""
from __future__ import annotations

import argparse
import os
import platform
import signal
import sys
import threading
import time
from pathlib import Path

from app import sources
from app.dashboard import serve
from app.engine import CascadeEngine, EngineConfig


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Playback harm monitor (on-device cascade)")
    p.add_argument("--source", default=None,
                   choices=["auto", "audiotee", "pipewire", "device", "file"],
                   help="capture backend (default: auto — audiotee on macOS, pipewire on Linux)")
    p.add_argument("--file", help="audio file for --source file")
    p.add_argument("--realtime", action="store_true",
                   help="replay a file at wall-clock speed (demo pacing)")
    p.add_argument("--device", help="input device name/index for --source device")
    p.add_argument("--server", default=None,
                   help="escalation endpoint, e.g. http://192.168.0.10:8770/ (default: local only)")
    p.add_argument("--no-text", action="store_true", help="disable the ASR + text branch")
    p.add_argument("--no-lexicon", action="store_true",
                   help="score text with the classifier only (drop the slang lexicon)")
    p.add_argument("--language-gate", action="store_true",
                   help="enhance + dual Silero VAD + Whisper-tiny Korean LID before ASR")
    p.add_argument("--language-gate-vad", default=None,
                   help="silero_vad.jit path (or SILERO_VAD_MODEL)")
    p.add_argument("--language-gate-checkpoint", default=None,
                   help="Whisper-tiny Korean LID .pt path (or KOREAN_LID_CHECKPOINT)")
    p.add_argument("--deepfilter-exe", default=None,
                   help="DeepFilterNet executable (or DEEPFILTER_EXE); Wiener fallback if omitted")
    p.add_argument("--language-gate-device", default="auto",
                   help="device for Korean LID: auto|cpu|mps|cuda")
    p.add_argument("--language-gate-vad-threshold", type=float, default=0.10,
                   help="low-recall-loss Silero threshold (default: 0.10)")
    p.add_argument("--language-gate-strict", action="store_true",
                   help="stop instead of passing the original ASR slice when the gate errors")
    p.add_argument("--upload-audio", action="store_true",
                   help="send the flagged audio to --server so its model can LISTEN "
                        "(Stage-2 is an audio model). Captured audio leaves this machine.")
    p.add_argument("--text-every", type=float, default=4.0,
                   help="ASR duty cycle in seconds (Ver1 default: shorter 4 s utterances)")
    p.add_argument("--asr-device", default=None, help="torch device for ASR (cpu/mps/cuda)")
    p.add_argument("--asr-model", default="ver1",
                   help="ver1 = quantized MDX Extra Demucs + Whisper Base FP16; "
                        "ver1-uvr-onnx = fast UVR MDXNET-3 ONNX + Whisper Base FP16; "
                        "tiny|base = Moonshine-KR (27M/61.5M, phone budget); "
                        "whisper-tiny|whisper-base|whisper-small = faster-whisper int8 "
                        "(~40/80/250MB, better slang); or any HF model id")
    p.add_argument("--ver1-demucs-packages", default=None,
                   help="override vendored Demucs Python package directory")
    p.add_argument("--ver1-demucs-repo", default=None,
                   help="override the default local quantized MDX Extra checkpoint repository")
    p.add_argument("--ver1-demucs-model", default=None,
                   help="override Demucs model name/signature (default: quantized MDX Extra)")
    p.add_argument("--ver1-whisper-cli", default=None,
                   help="override bundled whisper.cpp CLI")
    p.add_argument("--ver1-whisper-model", default=None,
                   help="override bundled Whisper Base FP16 GGML model")
    p.add_argument("--ver1-silence-db", type=float, default=-50.0,
                   help="keep Demucs vocal frames louder than this dBFS value")
    p.add_argument("--ver1-whisper-language", default="ko",
                   help="Whisper language code for Ver1 (default: ko; use auto to detect)")
    p.add_argument("--ver1-demucs-device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="persistent HTDemucs device (default: auto; MPS on Apple Silicon)")
    p.add_argument("--uvr-onnx-model", default=None,
                   help="override UVR MDX-Net ONNX model for --asr-model ver1-uvr-onnx")
    p.add_argument("--fp32", action="store_true", help="skip int8 quantization (debug)")
    p.add_argument("--all-quantized", action="store_true",
                   help="use MDX Extra quantized Demucs + Whisper Base Q6_K; "
                        "CED/KoELECTRA remain int8")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-ui", action="store_true", help="console only")
    p.add_argument("--thresholds", default=None, help="override cascade_thresholds.json path")
    return p.parse_args(argv)


def _make_source(a):
    if a.source == "file":
        if not a.file:
            sys.exit("--source file requires --file")
        return sources.FileSource(a.file, realtime=a.realtime)
    prefer = None if a.source in (None, "auto") else a.source
    return sources.auto_source(prefer, device=a.device)


def _asr_model_id(name: str) -> str:
    from cascade.pipeline import (
        MOONSHINE_BASE,
        MOONSHINE_TINY,
        WHISPER_BASE,
        WHISPER_SMALL,
        WHISPER_TINY,
        WHISPER_TURBO,
        VER1_DEMUCS_WHISPER_BASE,
        VER1_UVR_ONNX_WHISPER_BASE,
    )
    return {"whisper-tiny": WHISPER_TINY, "whisper-base": WHISPER_BASE,
            "whisper-small": WHISPER_SMALL, "whisper-turbo": WHISPER_TURBO,
            "ver1": VER1_DEMUCS_WHISPER_BASE,
            "ver1-demucs-base": VER1_DEMUCS_WHISPER_BASE,
            "ver1-uvr-onnx": VER1_UVR_ONNX_WHISPER_BASE,
            "tiny": MOONSHINE_TINY, "base": MOONSHINE_BASE}.get(name, name)


def _default_asr_device(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"        # Apple Silicon: ASR on the GPU, int8 branches stay on CPU
        except Exception:
            pass
        return "cpu"
    return None                      # cascade.pipeline picks cuda when present


def _preflight(text_enabled: bool, language_gate: bool = False,
               vad_path: str | None = None, checkpoint_path: str | None = None,
               deepfilter_exe: str | None = None) -> None:
    """Fail with one actionable message before any model/HF download happens."""
    from cascade.pipeline import missing_artifacts
    missing = missing_artifacts(text=text_enabled)
    if missing:
        lines = "\n".join(f"  - {m}" for m in missing)
        sys.exit("[app] model weights are missing (they are not in git):\n"
                 f"{lines}\n"
                 "[app] fetch them with:  bash scripts/fetch_data.sh --models\n"
                 "[app] (needs `gh auth login`; ~0.2 GB — the full training bundle is not "
                 "required to run the app)")
    if language_gate:
        required = {
            "Silero VAD (--language-gate-vad or SILERO_VAD_MODEL)": vad_path,
            "Korean LID (--language-gate-checkpoint or KOREAN_LID_CHECKPOINT)": checkpoint_path,
        }
        absent = [
            name for name, value in required.items()
            if not value or not Path(value).is_file()
        ]
        if deepfilter_exe and not Path(deepfilter_exe).is_file():
            absent.append("DeepFilterNet (--deepfilter-exe or DEEPFILTER_EXE)")
        if absent:
            lines = "\n".join(f"  - {name}" for name in absent)
            sys.exit("[app] language-gate artifacts are missing:\n" + lines)


def _artifact_path(explicit: str | None, environment_name: str) -> str | None:
    return explicit or os.environ.get(environment_name)


def main(argv=None) -> None:
    a = _parse(argv)
    root = Path(__file__).resolve().parents[2]
    # Ver1's research baseline keeps the transcription model at FP16 because
    # Q6_K lost Korean lyric/dialogue accuracy in the local comparison.  Only
    # Demucs is quantized by default to reduce the heaviest front-end cost.
    if _asr_model_id(a.asr_model) == "ver1:htdemucs+whisper-base-fp16":
        if a.ver1_demucs_repo is None:
            a.ver1_demucs_repo = str(root / "artifacts/ver1/demucs-quantized")
        if a.ver1_demucs_model is None:
            a.ver1_demucs_model = "14fc6a69"
        if a.ver1_whisper_model is None:
            a.ver1_whisper_model = str(root / "artifacts/ver1/whisper/ggml-base.bin")
    if _asr_model_id(a.asr_model) == "ver1:uvr-mdx-net-onnx+whisper-base-fp16":
        if a.uvr_onnx_model is None:
            a.uvr_onnx_model = str(
                root / "artifacts/onnx/uvr_mdxnet_3_9662/UVR_MDXNET_3_9662.onnx"
            )
        if a.ver1_whisper_model is None:
            a.ver1_whisper_model = str(root / "artifacts/ver1/whisper/ggml-base.bin")
    if a.all_quantized:
        a.ver1_demucs_repo = str(root / "artifacts/ver1/demucs-quantized")
        a.ver1_demucs_model = "14fc6a69"
        a.ver1_whisper_model = str(
            root / "artifacts/ver1/whisper/ggml-base-q6_k.bin"
        )
    gate_vad = _artifact_path(a.language_gate_vad, "SILERO_VAD_MODEL")
    gate_checkpoint = _artifact_path(
        a.language_gate_checkpoint, "KOREAN_LID_CHECKPOINT"
    )
    deepfilter_exe = _artifact_path(a.deepfilter_exe, "DEEPFILTER_EXE")
    _preflight(text_enabled=not a.no_text, language_gate=a.language_gate,
               vad_path=gate_vad, checkpoint_path=gate_checkpoint,
               deepfilter_exe=deepfilter_exe)
    src = _make_source(a)
    cfg = EngineConfig(thresholds_path=a.thresholds, text_enabled=not a.no_text,
                       text_every_sec=a.text_every, server_url=a.server,
                       asr_model_id=_asr_model_id(a.asr_model),
                       ver1_demucs_packages=a.ver1_demucs_packages,
                       ver1_demucs_repo=a.ver1_demucs_repo,
                       ver1_demucs_model_name=a.ver1_demucs_model,
                       ver1_whisper_cli=a.ver1_whisper_cli,
                       ver1_whisper_model=a.ver1_whisper_model,
                       ver1_silence_threshold_db=a.ver1_silence_db,
                       ver1_whisper_language=a.ver1_whisper_language,
                       ver1_demucs_device=a.ver1_demucs_device,
                       uvr_onnx_model=a.uvr_onnx_model,
                       text_lexicon=not a.no_lexicon,
                       language_gate_enabled=a.language_gate,
                       language_gate_vad_path=gate_vad,
                       language_gate_checkpoint_path=gate_checkpoint,
                       language_gate_deepfilter_exe=deepfilter_exe,
                       language_gate_device=a.language_gate_device,
                       language_gate_vad_threshold=a.language_gate_vad_threshold,
                       language_gate_fail_open=not a.language_gate_strict,
                       upload_audio=a.upload_audio)
    engine = CascadeEngine(cfg)
    print(f"[app] thresholds: acoustic {engine.thr.acoustic:.3f} · text {engine.thr.text:.3f}"
          f"  (fit {engine.thr.meta.get('fit_split', '?')})", flush=True)
    branch_precision = "fp32" if a.fp32 else "int8"
    if a.all_quantized:
        asr_precision = "; MDX Extra=quantized, Whisper Base=Q6_K"
    else:
        labels = {
            "ver1:htdemucs+whisper-base-fp16": "; MDX Extra=quantized, Whisper Base=FP16",
            "ver1:uvr-mdx-net-onnx+whisper-base-fp16": "; UVR MDXNET-3=ONNX, Whisper Base=FP16",
        }
        asr_precision = labels.get(_asr_model_id(a.asr_model), "")
    print(f"[app] loading models (CED/KoELECTRA={branch_precision}{asr_precision}) …",
          flush=True)
    engine.load_models(int8=not a.fp32, asr_device=_default_asr_device(a.asr_device))
    language_gate_label = ""
    if a.language_gate:
        language_gate_label = "  language-gate=" + (
            "deepfilter" if deepfilter_exe else "wiener"
        )
    print(f"[app] source: {src.name}"
          f"{'' if cfg.text_enabled else '  (text branch off)'}"
          f"{'  asr=' + a.asr_model if cfg.text_enabled else ''}"
          f"{language_gate_label}",
          flush=True)

    if not a.no_ui:
        serve(engine, source_name=src.name, host=a.host, port=a.port)
        print(f"[app] dashboard: http://{a.host}:{a.port}", flush=True)

    stop = threading.Event()

    def _sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    silent = {"n": 0, "warned": False}

    def on_result(r):
        mark = {"ok": " ", "watch": "!", "alert": "#"}[r.level]
        line = (f"{mark} t={r.t_start:7.1f}s  rms={r.rms:.4f}  acoustic={_f(r.acoustic)}  "
                f"text={_f(r.text)}  risk={r.risk:.2f}  {r.latency_ms:.0f}ms")
        if r.reasons:
            line += f"  -> ESCALATE ({', '.join(r.reasons)})"
        if r.language_gate:
            line += (f"  lang={r.language_gate.get('selected_groups', 0)}/"
                     f"{r.language_gate.get('groups', 0)}")
        print(line, flush=True)
        if r.transcript and r.transcript.strip():
            print(f"    asr: {r.transcript[:100]}", flush=True)
        # A silent stream scores a constant value forever and never runs ASR — that looks
        # like "the model is stuck" but means the capture backend is delivering zeros.
        silent["n"] = silent["n"] + 1 if r.rms < 1e-4 else 0
        if silent["n"] == 3 and not silent["warned"]:
            silent["warned"] = True
            print(f"[app] WARNING: {src.name} has delivered only silence for 3 windows "
                  f"(rms < 1e-4).\n"
                  "[app] Audio is not reaching the app. Check, in order:\n"
                  "[app]   1) something is actually playing to the DEFAULT output device\n"
                  "[app]   2) macOS audio-recording permission was granted (System Settings ->\n"
                  "[app]      Privacy & Security -> check for this terminal/app)\n"
                  "[app]   3) audiotee is code-signed — unsigned builds capture silence with\n"
                  "[app]      no prompt: codesign --force --sign - $(which audiotee)\n"
                  "[app]   4) isolate capture from the models:  uv run python -m app.sources\n",
                  flush=True)

    worker = threading.Thread(target=_drive, args=(engine, src, on_result, stop), daemon=True)
    worker.start()
    try:
        while worker.is_alive() and not stop.is_set():
            time.sleep(0.2)
    finally:
        stop.set()
        src.close()
        engine.flush()          # submit an event still open at end of stream
        if a.server:            # let the last POSTs land (a judge can take seconds per event)
            if not engine.escalator.drain(60.0):
                print("[app] warning: some escalations were still queued at exit "
                      "(kept locally)", flush=True)
        s = engine.snapshot()
        print(f"\n[app] {s['stats']['windows']} windows · {s['stats']['escalations']} escalating"
              f" · {s['stats']['events']} events sent · {s['stats']['asr_runs']} ASR runs"
              f" · transport {s['transport']}", flush=True)


def _drive(engine, src, on_result, stop):
    try:
        for frame in src.frames():
            for res in engine.push(frame):
                on_result(res)
            if stop.is_set():
                return
    except Exception as e:
        print(f"[app] capture stopped: {type(e).__name__}: {e}", flush=True)
    finally:
        stop.set()


def _f(v):
    return "  –  " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
