"""Stage-2 judgment: how harmful is this escalated clip, and why (spec Task B).

This is the server's job, not the device's. The device answers "suspicious?" with two cheap
binary triggers; here a large audio LLM listens to the actual audio plus the device's
transcript and returns a degree in percent, a category and a short reason.

Backends
  stub        — records the request, returns degree=None. Keeps the contract exercised with
                no model. Default, so a demo works on a machine with no GPU.
  qwen-omni   — Qwen2.5-Omni-7B (thinker only; the speech-output Talker is not loaded) in
                4-bit. Chosen because it takes AUDIO directly: the whole point of tier 3 is
                to hear what the on-device text branch cannot read (screams, tone, context).
  qwen-audio  — Qwen2-Audio-7B-Instruct in 4-bit; fallback if Omni cannot be loaded.

Size note: 7B in 4-bit is what fits a 12 GB GPU. Anything larger needs a bigger card or an
API — and an API would send captured audio off the machine, which is a policy decision, not
a technical one.

The model is asked for strict JSON. Free-form text is parsed defensively and, if it cannot be
read, returned as `raw` with degree=None rather than guessed at.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

SR = 16000
MAX_AUDIO_SEC = 30.0

CATEGORIES = ("violence", "sexual", "gambling", "drug", "abuse", "none")

PROMPT = """이 오디오는 청소년 유해 콘텐츠 탐지 시스템이 의심 구간으로 표시한 것입니다.
오디오를 직접 듣고 판단하세요.
참고로 온디바이스 받아쓰기 결과입니다(오류가 많을 수 있습니다):
{transcript}

다음 JSON만 출력하세요. 설명이나 코드블록 없이 JSON 한 개만:
{{"degree": <0-100 정수, 청소년에게 유해한 정도>,
  "category": <"violence"|"sexual"|"gambling"|"drug"|"abuse"|"none">,
  "reason": "<한국어 한 문장, 무엇을 근거로 판단했는지>",
  "confident": <true|false>}}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalize(verdict: dict[str, Any] | None, raw: str = "") -> dict[str, Any]:
    """Coerce a model answer into the response contract; never invent a degree."""
    out: dict[str, Any] = {"degree_percent": None, "category": None, "reason": None,
                           "confident": None, "raw": raw[:600] or None}
    if not verdict:
        return out
    d = verdict.get("degree")
    if isinstance(d, (int, float)) and 0 <= float(d) <= 100:
        out["degree_percent"] = int(round(float(d)))
    cat = str(verdict.get("category", "")).lower().strip()
    out["category"] = cat if cat in CATEGORIES else None
    reason = verdict.get("reason")
    out["reason"] = str(reason)[:300] if reason else None
    if isinstance(verdict.get("confident"), bool):
        out["confident"] = verdict["confident"]
    if out["degree_percent"] is not None:
        out["raw"] = None                      # parsed cleanly, no need to keep the text
    return out


class StubJudge:
    name = "stub"

    def judge(self, wav: np.ndarray | None, transcript: str) -> dict[str, Any]:
        return normalize(None)


class _AudioLLMJudge:
    """Shared plumbing for the two audio-LLM backends."""

    model_id = ""
    name = ""

    NEED_VRAM_GB = 7.5                          # 7B in 4-bit measured at 6.3 GB, plus headroom

    def __init__(self, device_map: str = "auto", four_bit: bool = True,
                 max_new_tokens: int = 160, model_id: str | None = None):
        self.max_new_tokens = max_new_tokens
        self.four_bit = four_bit
        self.device_map = device_map
        # A server may have the several-GB checkpoint copied locally instead of
        # having access to Hugging Face.  Transformers accepts either form.
        if model_id:
            self.model_id = model_id
        self.check_vram()
        self._load()

    @classmethod
    def check_vram(cls, four_bit: bool = True) -> None:
        """A busy GPU makes transformers report 'Some modules are dispatched on the CPU or the
        disk', which never mentions the actual cause. Say it plainly instead."""
        try:
            import torch
        except ImportError:                     # pragma: no cover
            return
        if not torch.cuda.is_available():
            return
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if four_bit and free_gb < cls.NEED_VRAM_GB:
            raise RuntimeError(
                f"only {free_gb:.1f} GB VRAM free; this judge needs ~{cls.NEED_VRAM_GB} GB in "
                "4-bit.\n  Something else is probably still holding the GPU — check\n"
                "  nvidia-smi --query-compute-apps=pid,used_memory --format=csv\n"
                "  and kill it (killing a shell can leave its python process alive).")

    def _quant_config(self):
        if not self.four_bit:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:                # pragma: no cover - env dependent
            raise ImportError("bitsandbytes/accelerate missing — uv sync --group server") from e
        import torch
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)

    def _load(self):                            # pragma: no cover - needs the weights
        raise NotImplementedError

    def _prepare(self, wav: np.ndarray | None, transcript: str):
        """Trim audio to MAX_AUDIO_SEC and build the chat inputs."""
        audio = None
        if wav is not None and wav.size:
            audio = np.asarray(wav, dtype=np.float32)[: int(MAX_AUDIO_SEC * SR)]
        text = PROMPT.format(transcript=(transcript or "(전사 없음)")[:500])
        return audio, text

    def judge(self, wav: np.ndarray | None, transcript: str) -> dict[str, Any]:  # pragma: no cover
        raw = self._generate(*self._prepare(wav, transcript))
        return normalize(_extract_json(raw), raw)

    def _generate(self, audio, text) -> str:    # pragma: no cover
        raise NotImplementedError


class QwenOmniJudge(_AudioLLMJudge):
    model_id = "Qwen/Qwen2.5-Omni-7B"
    name = "qwen2.5-omni-7b-4bit"

    def _load(self):                            # pragma: no cover - needs the weights
        import torch
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        # Thinker only: the Talker exists to SPEAK the answer, which this server never does,
        # and skipping it saves several GB.
        print("[judge] loading the Thinker only — transformers will list the checkpoint's "
              "talker.* / token2wav.* tensors as UNEXPECTED.\n"
              "[judge] that is the speech-output half (~5B params) and skipping it is the "
              "point; 'MISSING' entries would be the problem, and there are none.", flush=True)
        self.proc = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        self.m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=self.device_map,
            quantization_config=self._quant_config(), low_cpu_mem_usage=True).eval()

    def _generate(self, audio, text) -> str:    # pragma: no cover
        import torch
        content = ([{"type": "audio", "audio": audio}] if audio is not None else []) + \
                  [{"type": "text", "text": text}]
        chat = [{"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(chat, add_generation_prompt=True,
                                               tokenize=False)
        inputs = self.proc(text=prompt, audio=[audio] if audio is not None else None,
                           sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.m.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        n_in = inputs["input_ids"].shape[-1]
        return self.proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


class QwenAudioJudge(_AudioLLMJudge):
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    name = "qwen2-audio-7b-4bit"

    def _load(self):                            # pragma: no cover - needs the weights
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(self.model_id)
        self.m = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=self.device_map,
            quantization_config=self._quant_config(), low_cpu_mem_usage=True).eval()

    def _generate(self, audio, text) -> str:    # pragma: no cover
        import torch
        content = ([{"type": "audio", "audio_url": "clip.wav"}] if audio is not None else []) + \
                  [{"type": "text", "text": text}]
        chat = [{"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(chat, add_generation_prompt=True,
                                               tokenize=False)
        inputs = self.proc(text=prompt, audio=[audio] if audio is not None else None,
                           sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.m.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        n_in = inputs["input_ids"].shape[-1]
        return self.proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


def make_judge(kind: str = "stub", **kw):
    kinds = {"stub": StubJudge, "qwen-omni": QwenOmniJudge, "qwen-audio": QwenAudioJudge}
    if kind not in kinds:
        raise ValueError(f"unknown judge: {kind} (choose from {sorted(kinds)})")
    return StubJudge() if kind == "stub" else kinds[kind](**kw)
