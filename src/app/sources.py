"""Audio sources — capture DEVICE PLAYBACK (not the mic) as 16 kHz mono float32 frames.

Why playback-only: spec §1b source ②. Playback audio needs no always-on tier-1 gate (the OS
playback state gates for free) and matches our training domain exactly (stream-original
clips), unlike mic re-recording which has an untested domain shift.

Platforms:
  - Linux (PipeWire/PulseAudio): `pw-record` on the default sink's monitor. No permission
    prompt, no extra install.
  - macOS 14.2+: `audiotee` (Core Audio process taps) — audio-only TCC category
    (NSAudioCaptureUsageDescription), separate from the mic. Needs a SIGNED binary or the
    permission prompt never fires.
  - macOS < 14.2 (or taps unavailable): a loopback input DEVICE (BlackHole/Loopback) read
    via sounddevice, after the user routes output through a Multi-Output Device.
  - File: offline replay for tests/demos.

All sources yield np.float32 mono frames at 16 kHz via `.frames()`, and are context managers.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SR = 16000
_FRAME = 4096  # samples per yielded frame (~256 ms) — small enough for a responsive hop


class AudioSource(ABC):
    name = "base"

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield mono float32 frames at SR until the source ends or close() is called."""

    def close(self) -> None:
        """Release the backend. Subclasses that hold a process/stream override this."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _PipeSource(AudioSource):
    """Common base for 'spawn a CLI that writes raw f32 mono PCM to stdout'."""

    cmd: list[str] = []

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def frames(self) -> Iterator[np.ndarray]:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        nbytes = _FRAME * 4
        try:
            while True:
                buf = self.proc.stdout.read(nbytes)
                if not buf:
                    err = (self.proc.stderr.read() or b"").decode(errors="replace")[:400]
                    if err.strip():
                        print(f"[{self.name}] capture ended: {err}", file=sys.stderr)
                    return
                if len(buf) < nbytes:                    # short final read
                    buf = buf[: len(buf) // 4 * 4]
                yield np.frombuffer(buf, dtype=np.float32).copy()
        finally:
            self.close()

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()


class PipeWireLoopback(_PipeSource):
    """Linux: capture the default sink's monitor (everything the machine plays)."""

    name = "pipewire-loopback"

    def __init__(self, latency_ms: int = 100):
        super().__init__()
        self.cmd = ["pw-record", "-P", "{ stream.capture.sink=true }",
                    "--rate", str(SR), "--channels", "1", "--format", "f32",
                    "--latency", f"{latency_ms}ms", "-"]

    @staticmethod
    def available() -> bool:
        return shutil.which("pw-record") is not None


class AudioTeeSource(AudioSource):
    """macOS 14.2+ (primary path on Apple Silicon): Core Audio process taps via `audiotee`.

    Writes raw little-endian PCM chunks to stdout, all logging on stderr. Sample format is
    f32 normally but **s16 whenever sample-rate conversion happens** — and asking for 16 kHz
    on a Mac (devices run 44.1/48 kHz) always converts. Rather than trust either, we detect
    the width from the first non-silent chunk: reading s16 data as f32 yields absurd
    magnitudes / non-finite values, which is unmistakable on real audio.
    """

    name = "audiotee"

    def __init__(self, extra_args: list[str] | None = None, chunk_sec: float = 0.2,
                 dtype: str = "auto"):
        self.cmd = ["audiotee", "--sample-rate", str(SR),
                    "--chunk-duration", str(chunk_sec), *(extra_args or [])]
        self.dtype = dtype          # "auto" | "f32" | "s16"
        self.proc: subprocess.Popen | None = None

    @staticmethod
    def available() -> bool:
        if platform.system() != "Darwin" or shutil.which("audiotee") is None:
            return False
        try:                                    # taps need macOS >= 14.2
            parts = platform.mac_ver()[0].split(".")
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return False
        return (major, minor) >= (14, 2)

    @staticmethod
    def _looks_like_f32(buf: bytes) -> bool | None:
        """None when the chunk is silent (undecidable), else True for f32, False for s16."""
        a = np.frombuffer(buf, dtype=np.float32)
        if not np.isfinite(a).all():        # s16 bytes read as f32 -> NaN/Inf patterns
            return False
        if not np.any(a):
            return None
        return bool(np.abs(a).max() <= 4.0)

    def frames(self) -> Iterator[np.ndarray]:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        dtype = None if self.dtype == "auto" else self.dtype
        nbytes = _FRAME * 4                      # read in f32-sized blocks; s16 gives 2x frames
        try:
            while True:
                buf = self.proc.stdout.read(nbytes)
                if not buf:
                    err = (self.proc.stderr.read() or b"").decode(errors="replace")[:400]
                    if err.strip():
                        print(f"[audiotee] capture ended: {err}", file=sys.stderr)
                    return
                buf = buf[: len(buf) // 4 * 4]
                if dtype is None:
                    guess = self._looks_like_f32(buf)
                    if guess is None:            # silence: emit zeros, keep probing
                        yield np.zeros(len(buf) // 4, dtype=np.float32)
                        continue
                    dtype = "f32" if guess else "s16"
                    print(f"[audiotee] sample format detected: {dtype}", file=sys.stderr)
                if dtype == "f32":
                    yield np.frombuffer(buf, dtype=np.float32).copy()
                else:
                    yield (np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0)
        finally:
            self.close()

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()


class LoopbackDeviceSource(AudioSource):
    """Any platform: read from a loopback INPUT device (BlackHole/Loopback on macOS,
    a monitor source on Linux). Requires the user to route playback into it."""

    name = "loopback-device"

    def __init__(self, device: str | int | None = None, blocksize: int = _FRAME):
        self.device = device
        self.blocksize = blocksize
        self._stream = None

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def find_device(patterns=("blackhole", "loopback", "monitor")) -> int | None:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and any(p in d["name"].lower() for p in patterns):
                return i
        return None

    def frames(self) -> Iterator[np.ndarray]:
        import sounddevice as sd
        dev = self.device if self.device is not None else self.find_device()
        if dev is None:
            raise RuntimeError("no loopback input device found (install BlackHole and route "
                               "output through a Multi-Output Device)")
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=self.blocksize, device=dev)
        with self._stream:
            while True:
                buf, _ = self._stream.read(self.blocksize)
                yield buf[:, 0].copy()

    def close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
            except Exception:
                pass


class FileSource(AudioSource):
    """Offline replay (tests, demos). Decodes via ffmpeg when it is not already 16k mono wav."""

    name = "file"

    def __init__(self, path: str | Path, realtime: bool = False):
        self.path = Path(path)
        self.realtime = realtime

    def frames(self) -> Iterator[np.ndarray]:
        wav = self._read()
        for i in range(0, len(wav), _FRAME):
            chunk = wav[i:i + _FRAME]
            if len(chunk) == 0:
                break
            if self.realtime:
                import time
                time.sleep(len(chunk) / SR)
            yield chunk.astype(np.float32)

    def _read(self) -> np.ndarray:
        if self.path.suffix.lower() == ".wav":
            with wave.open(str(self.path), "rb") as w:
                if w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    raw = w.readframes(w.getnframes())
                    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.path), "-f", "f32le",
                              "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy()


def auto_source(prefer: str | None = None, device: str | int | None = None) -> AudioSource:
    """Pick the best playback-capture source for this machine.

    prefer: "pipewire" | "audiotee" | "device" | None (auto — native backend per OS).
    """
    if prefer == "pipewire":
        if not PipeWireLoopback.available():
            raise RuntimeError("pw-record not found (install pipewire-utils)")
        return PipeWireLoopback()
    if prefer == "audiotee":
        if not AudioTeeSource.available():
            raise RuntimeError("audiotee not found or macOS < 14.2 — see app/README.md")
        return AudioTeeSource()
    if prefer == "device":
        if not LoopbackDeviceSource.available():
            raise RuntimeError("sounddevice not installed")
        return LoopbackDeviceSource(device)
    if prefer is not None:
        raise ValueError(f"unknown source: {prefer}")
    mac = platform.system() == "Darwin"
    if mac and AudioTeeSource.available():
        return AudioTeeSource()
    if not mac and PipeWireLoopback.available():
        return PipeWireLoopback()
    if LoopbackDeviceSource.available():
        return LoopbackDeviceSource(device)
    if mac:
        raise RuntimeError("macOS: install audiotee (Core Audio taps, macOS 14.2+) or "
                           "BlackHole — see app/README.md")
    raise RuntimeError("no playback capture backend available; see app/README.md")


def probe(prefer: str | None = None, seconds: float = 5.0, device=None) -> dict:
    """Capture for `seconds` and report signal level — isolates capture from the models.

    A silent stream makes the cascade emit one constant score forever, which looks like a
    stuck model. Run this first when that happens.
    """
    src = auto_source(prefer, device=device)
    want = int(seconds * SR)
    got, peak, energy, nonzero = 0, 0.0, 0.0, 0
    with src:
        for frame in src.frames():
            if frame.size:
                peak = max(peak, float(np.abs(frame).max()))
                energy += float(np.sum(frame.astype(np.float64) ** 2))
                nonzero += int(np.count_nonzero(frame))
                got += frame.size
            if got >= want:
                break
    rms = (energy / got) ** 0.5 if got else 0.0
    return {"source": src.name, "samples": got, "seconds": round(got / SR, 2),
            "peak": round(peak, 6), "rms": round(rms, 6),
            "nonzero_frac": round(nonzero / got, 4) if got else 0.0}


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Capture self-test (level probe)")
    p.add_argument("--source", default=None,
                   choices=["auto", "audiotee", "pipewire", "device"])
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    print("[probe] play some audio now (music, a video — anything on the default output)",
          flush=True)
    prefer = None if a.source in (None, "auto") else a.source
    r = probe(prefer, a.seconds, a.device)
    print(f"[probe] {r}", flush=True)
    if r["samples"] == 0:
        print("[probe] FAIL: the backend produced no data at all — it is not running.")
    elif r["peak"] < 1e-4:
        print("[probe] FAIL: captured only silence.\n"
              "  - is audio really playing to the DEFAULT output device?\n"
              "  - macOS: was the audio-recording permission granted? (System Settings ->\n"
              "    Privacy & Security). An UNSIGNED audiotee captures silence and never\n"
              "    prompts: codesign --force --sign - $(which audiotee)\n"
              "  - Bluetooth/AirPlay outputs are not always covered by the tap; try the\n"
              "    built-in speakers.")
    elif r["peak"] > 1.5 or r["nonzero_frac"] < 0.5:
        print("[probe] SUSPECT: level looks wrong — the sample format may be misdetected.")
    else:
        print("[probe] OK: audio is reaching the app.")


if __name__ == "__main__":
    _main()
