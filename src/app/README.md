# On-device app — playback harm monitor

Captures what the machine **plays** (not the mic) and runs the on-device cascade live:
CED-mini int8 violence trigger + optional Korean language gate + Moonshine-KR → KoELECTRA
int8 text branch, escalating suspicious 10 s windows to the Linux Stage-2 host.

```
audio source ─► 10 s windows (hop 2 s) ─┬─► CED-mini int8 (raw) ──────────────┐
                                        └─► enhancement + dual Silero VAD     │
                                             └─► Whisper-tiny Korean LID      │
                                                  └─► ASR ─► KoELECTRA int8 ─┤
                                                                              ▼
                                decision (artifacts/cascade_thresholds.json) ─► escalate
```

No tier-1 CNN gate: measured 2026-07-30 it costs more CPU than it saves (`model_light.md`
§2-2), and playback capture is already gated for free by the OS playback state.

## Optional Korean language gate

`--language-gate` adds a recall-first router immediately before ASR. DeepFilterNet first
helps locate speech; raw and enhanced Silero timestamps are unioned; then the corresponding
raw speech is Wiener-enhanced for LID and ASR. Uncertain language is retained; only sustained,
high-confidence non-Korean groups are suppressed. The CED-mini acoustic branch always sees
the untouched window.

```powershell
$env:PYTHONUTF8 = "1"
uv run --group nlp python -m app.main `
  --source file --file C:\path\to\movie.wav --realtime `
  --language-gate `
  --language-gate-vad C:\models\silero_vad.jit `
  --language-gate-checkpoint C:\models\whisper_tiny_encoder_lid.pt `
  --deepfilter-exe C:\models\deep-filter.exe
```

The three added artifacts in the tested Windows setup total 45.66MB: DeepFilterNet CLI
26.91MB, Silero VAD 2.27MB, and the Whisper-tiny encoder LID checkpoint 16.48MB. Omitting
`--deepfilter-exe` selects the model-free Wiener fallback. See
[`docs/08-korean-language-gate.md`](../../docs/08-korean-language-gate.md) for the policy,
model-size caveats, actual Korean-clip smoke test, and mobile-port limits.

## MacBook (primary target — Apple Silicon, macOS 14.2+)

macOS **can** capture app/system audio without the microphone, via Core Audio process taps.
That permission (`NSAudioCaptureUsageDescription`) is a **separate TCC category from the
mic** — this app never asks for microphone access.

1. Get the repo and the model weights onto the Mac (the checkpoints are not in git — they
   live in the `data-v1` release; `--models` fetches only what inference needs, ~0.2 GB
   instead of the full 5.4 GB training bundle):

   ```bash
   gh auth login                                   # account with access to the private repo
   git clone https://github.com/soysaucecrab/Danger-Audio-Teenager.git
   cd Danger-Audio-Teenager                        # every command below runs from here
   bash scripts/fetch_data.sh --models
   ```

2. Build the capture helper (`audiotee`, MIT, Swift 5.9+; no Homebrew formula yet):

   ```bash
   git clone https://github.com/makeusabrew/audiotee.git && cd audiotee
   swift build -c release
   sudo cp .build/release/audiotee /usr/local/bin/     # must be on PATH
   ```

   **The binary must be signed** or macOS never shows the permission prompt and capture
   silently returns nothing. If `swift build` produced an unsigned binary, sign it locally:

   ```bash
   codesign --force --sign - /usr/local/bin/audiotee   # ad-hoc signature
   ```

3. Install the project deps and run (from the repo root — `-m app.main` needs the project):

   ```bash
   uv sync --group nlp
   uv run --group nlp python -m app.main --server http://<linux-host>:8770/
   ```

4. macOS shows the audio-capture prompt on the first window — approve it. Open
   <http://127.0.0.1:8765> for the dashboard.

Smoke-test the install without any capture backend (point it at any audio file you have):

```bash
uv run --group nlp python -m app.main --source file --file /path/to/any.wav --no-ui
```

**If the acoustic score is frozen at one value and no transcript ever appears, the capture is
delivering silence** — the models are fine, the audio is not arriving. Isolate it:

```bash
uv run --group nlp python -m app.sources        # play something, then read the level report
```

`peak`/`rms` near zero means silence. In order of likelihood: nothing is playing to the
**default** output; the audio-recording permission was never granted; `audiotee` is unsigned
(it then captures silence and never prompts — `codesign --force --sign - $(which audiotee)`);
or the output is a Bluetooth/AirPlay device the tap does not cover (try built-in speakers).

Notes for Apple Silicon:
- int8 branches (trigger, text) run on CPU with the `qnnpack` engine — selected
  automatically (`cascade.pipeline._set_quant_engine`).
- ASR defaults to **MPS** on macOS; `--asr-device cpu` if you hit an MPS op gap.
- `audiotee` emits 16-bit PCM whenever it resamples (always, when we ask for 16 kHz) and
  32-bit float otherwise; the source auto-detects which.

**Older macOS or taps unavailable** → route playback through a virtual device:
install [BlackHole](https://existential.audio/blackhole/) 2ch, create a Multi-Output Device
(Audio MIDI Setup) containing your speakers + BlackHole, select it as system output, then

```bash
uv run --group nlp --with sounddevice python -m app.main --source device
```

(`sounddevice` is only needed for this fallback, so it is not a project dependency.)

## Linux (dev box / same code path)

```bash
uv run --group nlp python -m app.main --source pipewire
```

Uses `pw-record` on the default sink's monitor — no permissions, no extra install.

## Server (Linux Stage-2 host)

```bash
uv run --group nlp python -m app.server --host 0.0.0.0 --port 8770
```

Accepts the escalation contract and logs to `data_dl/app_escalations/server_received.jsonl`.
Without `--judge` it records the request and returns `degree=None`; with `--judge qwen-omni`
it actually listens and answers — see **Stage-2** below.

## Offline replay (no capture backend, good for demos)

```bash
uv run --group nlp python -m app.main --source file --file some.wav --realtime
```

## Useful flags

| flag | effect |
|---|---|
| `--no-text` | acoustic branch only (no ASR — lowest CPU) |
| `--text-every 6` | ASR duty cycle in seconds (default 6) |
| `--asr-model` | `tiny`/`base` = Moonshine-KR (27M/61.5M) · `whisper-tiny/base/small` int8 (~40/80/250MB). tiny is the default: best profanity survival per parameter (`cascade/pipeline.py` has the table) |
| `--no-lexicon` | classifier only, drop the slang lexicon (it is what catches 은어) |
| `--language-gate` | enable enhancement + dual Silero VAD + Korean LID before ASR |
| `--language-gate-vad PATH` | standalone `silero_vad.jit` model path |
| `--language-gate-checkpoint PATH` | compact Whisper-tiny encoder Korean LID checkpoint |
| `--deepfilter-exe PATH` | DeepFilterNet CLI; omit to use the Wiener fallback |
| `--language-gate-strict` | raise a gate error instead of the default Korean-recall fail-open |
| `--server URL` | escalate over HTTP; without it, escalations stay local |
| `--upload-audio` | also send the waveform so the server's model can listen |
| `--no-ui` | console only |
| `--fp32` | skip int8 (debugging) |
| `--thresholds PATH` | use a different fitted operating point |

## Measured behaviour (Linux, 2026-07-30)

- ~17–21 ms CPU per 10 s window for the acoustic branch; 100–300 ms when ASR also runs.
- Live playback capture → violent segment escalated, escalations delivered to the server.
- Consecutive escalating windows are merged into ONE event before submission (windows overlap,
  so a single scene used to be reported 3–5 times).
- **Moonshine hallucinates repetition loops on non-speech** ("와! 와! 와! …") and the text
  classifier scored one 0.908 → false escalation. Degenerate transcripts are now shown but
  never scored. The spectral speech gate written for the same bug is **disabled**: measured on
  real audio it scored movie dialogue 0.00 and gunshots 0.96, i.e. it suppressed ASR on real
  speech (`app/vad.py` documents the refutation).

## Limitations

- Playback capture only. The **mic path is deliberately absent**: no training clip is
  mic-re-recorded, so its domain shift is unmeasured (`spec.md` §1b).
- Windows are 10 s to match the training clip length, so the fitted thresholds apply
  unchanged; detection latency is therefore up to ~10 s after a scene starts.
- Degenerate-transcript suppression can also drop a genuine single-word repetition
  ("살려줘 살려줘 …"). That case is covered by the acoustic branch (`vio_verbal`), which runs
  on the same window — hence dropping the *text* score is safe here.

## Stage-2: the server verdict (degree %)

The device only answers "suspicious?". The degree comes from an audio LLM on the Linux host:

```bash
uv sync --group nlp --group server        # bitsandbytes + accelerate + Omni's image deps
uv run --group nlp --group server python -m app.server --host 0.0.0.0 --port 8770 \
    --judge qwen-omni
```

and the client has to actually send the sound, which is opt-in because that is the moment
captured audio leaves the machine:

```bash
uv run --group nlp python -m app.main --server http://<host>:8770/ --upload-audio
```

Measured on this box (RTX 3060 12 GB, 2026-07-30): Qwen2.5-Omni-7B loads in **4-bit** in 21 s
using **6.3 GB VRAM**, and answers in **~6 s per event**. Only the Thinker is loaded — the
Talker exists to speak the answer, which a server never needs. 7B in 4-bit is the largest that
fits 12 GB; bigger needs another card or an API (and an API would send the audio off-machine).

Verdicts from the real profanity clip and a benign control:

| clip | degree | category | reason |
|---|---|---|---|
| movie profanity scene | 80% | abuse | 어른이 청소년을 비하하고 폄하하는 내용이 포함되어 있습니다 |
| benign TTS sentence | 0% | none | 회의 시간 정보이며 청소년 유해 콘텐츠와 관련이 없습니다 |

`--judge stub` (the default) keeps the contract exercised with no model, so a demo works on a
machine with no GPU. A model that answers out of range or unparseably yields `degree=None`
plus the raw text — it is never rounded into a number that looks authoritative.
