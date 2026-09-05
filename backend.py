"""Backend abstraction — Ollama and MLX as peer LLM hosts.

Two implementations share one streaming contract so the agent loop, TUI,
and CLI never have to know which daemon they are talking to:

  * `OllamaBackend`  — thin wrapper around `ollama.AsyncClient` that
    delegates to the existing watchdog/retry helpers in `ollama_io.py`.

  * `MLXBackend`     — loads the MLX model in-process and streams via
    `mlx_lm.stream_generate` directly. Default for the TUI (reliable,
    no HTTP). The model is held in a class-level cache so subsequent
    requests reuse the loaded weights.

  * `MLXServerBackend` — talks to a running `mlx_lm.server` (or any
    OpenAI-compatible MLX server) over HTTP. Enabled when `MLX_SERVER_URL`
    or `MLX_HOST` is set, or when `LLM_BACKEND=mlx-server`. Use this
    for parallel batch testing: N agent clients → one server with
    continuous batching, one model copy in VRAM.

`detect_backend()` picks an LLM daemon at session start. The rule:

  1. Honor `LLM_BACKEND=ollama|mlx|auto` when set (CLI `--backend` counts too).
  2. On macOS (`darwin`), if neither env nor argument picks a backend,
     default to **MLX** (Apple GPU). Linux and others default to `auto`.
  3. With preference `auto`: probe both; if MLX_MODEL is set or a single
     local MLX model is discoverable → MLX. Otherwise check Ollama.
  4. With preference `mlx` or `ollama`: force that daemon or raise.

For MLX, "which model" comes from:
  1. `MLX_MODEL` env var (explicit path or HF id)
  2. Whatever oMLX already has **loaded** (live `/v1/models/status` — not
     merely discovered/unloaded; TUI can type a goal without `/load`)
  3. The first local MLX folder under `~/MLX_Models/`
  4. Otherwise: raise — there's nothing to load.
  glm5_next / deepseek_v4 / qwen4_exp always talk to oMLX, never in-process mlx_lm.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Iterable, Literal

import ollama

from ollama_io import (
    Candidate,
    DeliberationDetector,
    DiagnoseBloatDetector,
    RepetitionDetector,
    StreamResult,
    _should_grace_inline_data_bloat,
    stream_chat_with_retry,
)


# Tags that are clearly NOT chat models. Same list as chat.py used to
# carry — kept here so detection can filter ps results consistently
# across drivers.
_NON_CHAT_TAG_FRAGMENTS: tuple[str, ...] = (
    "z-image", "stable-diffusion", "sdxl", "flux",
    "embed", "embedding", "minilm", "bge-", "rerank",
    "whisper", "tts-", "voxcpm", "speech-", "voice-",
)


def _is_chat_capable_tag(name: str) -> bool:
    n = (name or "").lower()
    return not any(frag in n for frag in _NON_CHAT_TAG_FRAGMENTS)


# Vision-Language Model (VLM) name patterns (Item 4 in chat.py /list).
# A VLM can accept image input alongside text — Claude / GPT can do this
# via API, and several open-weight model families ship VLM variants
# (Qwen-VL, LLaVA, DeepSeek-VL, MiniCPM-V, Pixtral, etc.).
#
# Why this matters: the agent's VLM-critique path (chat.py /vlm,
# agent.use_vlm_critique) sends the latest game screenshot to the model
# so it can SEE the rendered output and adjust on visual evidence
# (e.g. "the player sprite isn't visible because Mario was drawn off-
# canvas"). For text-only models that path is a no-op — they can't read
# images. Showing the modality in /list lets the user pick the right
# tool for the job: text-only model for a small/fast iter, VLM when a
# visual bug needs eyes.
#
# We classify by NAME (substring match, case-insensitive) — not by
# probing the model with a test image, which would be expensive. The
# agent still does a real probe via `_detect_vlm` at session start;
# this name-based classifier is purely for the /list UI and may miss
# variants we haven't catalogued. When the name doesn't match either
# bucket we return "text" (the safe default — most models are
# text-only and the runtime probe will detect the rare VLM that
# slipped through the catalog).
#
# Adding a new VLM family: drop one or more substring patterns into
# `_VLM_NAME_SUBSTRINGS`. Be specific enough that ordinary text-only
# coding models don't accidentally match — e.g. don't add bare "vision"
# without a unique surrounding token.

# Substrings that, when present in the model NAME, indicate VLM.
# Cross-checked against the model families shipped on HuggingFace as
# of 2026-Q2. Patterns are matched case-insensitive against the full
# tag (Ollama) or path basename (MLX).
_VLM_NAME_SUBSTRINGS: tuple[str, ...] = (
    # Alibaba Qwen family
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen3.6-vl",
    "qwen-omni", "qwen2.5-omni", "qwen3-omni",
    # 2026-05-15 user correction: Qwen3.6 unified vision into the
    # base 27B (and 7B etc.), dropping the "-VL" suffix that earlier
    # Qwen families used. The base `Qwen3.6-27B` from Alibaba ships
    # as a VLM out of the box — see HF model card for
    # mlx-community/Qwen3.6-27B-bf16 (pipeline_tag: image-text-to-text,
    # library: mlx-vlm). Match the family prefix so any quant
    # (-bf16, -mxfp8, -8bit, etc.) is correctly labeled. Earlier Qwen3
    # (without ".6"/".8") was NOT unified — keep that prefix OUT of this
    # list so plain Qwen3-30B etc. stay labeled text-only.
    "qwen3.6-27b", "qwen3.6-7b", "qwen3.6-72b", "qwen3.6-235b",
    "qwen3.6:27b", "qwen3.6:7b", "qwen3.6:72b", "qwen3.6:235b",
    # 2026-08-15: Qwen3.8 keeps the same unified-vision packaging
    # (mlx-community/Qwen3.8-27B-mxfp8 is image-text-to-text via
    # mlx-vlm). Dense sizes only — do not add bare "qwen3.8" because
    # Qwen3.8-2.4T-A95B is text-only MoE.
    "qwen3.8-vl",
    "qwen3.8-27b", "qwen3.8-8b", "qwen3.8-7b", "qwen3.8-6b", "qwen3.8-5b",
    "qwen3.8:27b", "qwen3.8:8b", "qwen3.8:7b", "qwen3.8:6b", "qwen3.8:5b",
    # 2026-08-29: Qwen3.8-Flash-Next is a qwen4_exp VLM. Name-classifying
    # it as text routes in-process MLX through mlx_lm, which raises
    # "Model type qwen4_exp not supported" (trace 20260829_135708).
    "qwen3.8-flash",
    "qwen3.8:flash",
    # 2026-09-04: GLM-5.3-Flash (`glm5_next`) is Glm5NextForConditionalGeneration
    # with vision_config glm5_next_vision. /list showed [text] because no
    # substring matched. Do NOT add bare "glm-5" — GLM-5.2 is text-only.
    "glm-5.3", "glm_5.3", "glm5.3", "glm5_next",
    # Meta Muse Glimmer (2026-08) — multimodal agentic; mlx-vlm only
    # (model_type muse_glimmer). Without this tag, MLXBackend routes through
    # mlx_lm and fails with "Model type muse_glimmer not supported".
    "muse-glimmer", "muse_glimmer",
    # LLaVA family
    "llava", "bakllava",
    # DeepSeek vision
    "deepseek-vl",
    # OpenGVLab InternVL
    "internvl",
    # MiniCPM vision
    "minicpm-v", "minicpm-llama3-v",
    # Mistral / Pixtral
    "pixtral",
    # Google Gemma 3/4 (multimodal) — gemma3 + unified encoder-free gemma4
    "gemma3", "gemma-3", "gemma4", "gemma-4",
    # PaLI / SigLIP-based
    "pali", "paligemma",
    # CogVLM family
    "cogvlm", "cogagent",
    # Bunny (small VLM)
    "bunny-v",
    # Moondream
    "moondream",
    # HuggingFace M4 Idefics
    "idefics",
    # Florence-2
    "florence-2",
    # mPLUG-Owl
    "mplug-owl",
    # Microsoft Phi multimodal
    "phi-3-vision", "phi-3.5-vision", "phi-4-multimodal",
    # Cloud VLMs (Anthropic / OpenAI) — all current Claude / GPT-4o
    # / o-series models accept images via API. Match generously: any
    # gpt-4o*, gpt-5*, claude-*, claude-opus*, claude-sonnet* matches.
    "gpt-4o", "gpt-4.1", "gpt-5", "o1-", "o3-", "o4-",
    "claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-fable",
    "claude-haiku-3", "claude-haiku-4",
)


def _delta_thinking_text(delta: Any) -> str:
    """Hidden CoT from an OpenAI-style stream delta (dict or object).

    oMLX GLM-5.3 puts thinking in `reasoning_content` / `reasoning`.
    Those chunks never hit `content`, so the TUI showed 0 tokens while
    the model was talking to itself (BATTLEZO 20260904_095910).
    """
    if delta is None:
        return ""
    keys = ("reasoning_content", "reasoning")
    if isinstance(delta, dict):
        for key in keys:
            v = delta.get(key)
            if isinstance(v, str) and v:
                return v
        return ""
    for key in keys:
        v = getattr(delta, key, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _notify_thinking(on_token: Callable[[str], None] | None, piece: str) -> None:
    """Forward hidden-CoT chunks to `on_token.on_thinking` if the TUI bound one."""
    if not piece or on_token is None:
        return
    cb = getattr(on_token, "on_thinking", None)
    if cb is None:
        return
    try:
        cb(piece)
    except Exception:
        pass


def classify_model_modality(name: str | None) -> str:
    """Return "vlm" if the model NAME is a known Vision-Language Model
    pattern, else "text". Case-insensitive substring match.

    The classification is NAME-based only and may miss novel VLM
    families we haven't catalogued. Callers that need a definitive
    answer (e.g., the agent's runtime VLM-critique path) should
    additionally probe the live model — see `GameAgent._detect_vlm`.
    The /list TUI uses this name classifier to label rows with a
    `[VLM]` or `[text]` badge so the user can pick the right model
    without probing first.
    """
    if not name:
        return "text"
    low = name.lower()
    for sub in _VLM_NAME_SUBSTRINGS:
        if sub in low:
            return "vlm"
    return "text"


# Official Qwen3.8-27B chat_template.jinja (and mlx-community conversion):
#   reasoning_effort in {xhigh, medium, low} only. Native template default
#   is xhigh; harness default is medium (tag parser + plan turns — DK/SF
#   Aug 15 xhigh plans ran 15k–22k tokens before tags). There is NO "high"
#   — passing it raises in jinja. Aliases below never forward an illegal
#   value. Opt in to max: QWEN_REASONING_EFFORT=xhigh. Off:
#   QWEN_ENABLE_THINKING=0. Close think before code prefill so xhigh still
#   codes if someone opts back in.
_QWEN38_REASONING_EFFORTS = ("xhigh", "medium", "low")
_QWEN38_REASONING_ALIASES = {
    "high": "xhigh",  # not a legal template value; map to native default
    "max": "xhigh",
    "x-high": "xhigh",
}


# Stage-aware effort (Sept 2026 /640png traces: 91-98% of completion tokens
# were hidden CoT; patch turns burned 60-105 s for <=67 visible tokens).
# Applies ONLY when QWEN_REASONING_EFFORT is NOT set explicitly.
# "critic" (code/visual review sidecar) also runs low: it must finish
# while the coder streams, and its output is a short bullet list.
_QWEN38_STAGE_EFFORT_LOW = ("fix", "patch", "critic")


def chat_template_thinking_kwargs(
    model: str | None, stage: str | None = None,
) -> dict[str, Any]:
    """Qwen3.8 apply_chat_template kwargs. Harness default is medium.

    Native jinja default is xhigh; that fights <plan>/<html_file> parsing
    on long CoT. Must pass enable_thinking=True on the mlx_vlm path —
    that helper otherwise defaults thinking OFF. Other families return {}.

    `stage`: "plan" / "first_build" keep the default (medium); "fix" /
    "patch" drop to low. An explicit QWEN_REASONING_EFFORT wins for
    every stage.
    """
    if not model or "qwen3.8" not in model.lower():
        return {}
    enable_raw = os.environ.get("QWEN_ENABLE_THINKING", "1").strip().lower()
    if enable_raw in ("0", "false", "no", "off"):
        return {"enable_thinking": False}
    env_effort = os.environ.get("QWEN_REASONING_EFFORT")
    if env_effort is None and stage in _QWEN38_STAGE_EFFORT_LOW:
        return {"enable_thinking": True, "reasoning_effort": "low"}
    effort = (env_effort if env_effort is not None else "medium").strip().lower()
    effort = _QWEN38_REASONING_ALIASES.get(effort, effort)
    if effort not in _QWEN38_REASONING_EFFORTS:
        effort = "medium"
    return {"enable_thinking": True, "reasoning_effort": effort}


def omlx_messages_close_think_prefill(
    messages: list[dict], model: str | None,
) -> list[dict]:
    """Close Qwen `<think>` before an assistant prefill on the oMLX HTTP path.

    In-process MLX does this in `append_assistant_prefill` (DK 20260815_085321).
    oMLX applies the chat template server-side, so a trailing assistant
    `<html_file>` prefill lands *inside* the open think block and the game
    never appears as `content` (trace 20260829_165958). Plan turns (last
    role=user) are unchanged — the model still thinks. Thinking stays ON.
    """
    kw = chat_template_thinking_kwargs(model)
    if not kw.get("enable_thinking") or not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return messages
    content = last.get("content")
    if not isinstance(content, str) or not content:
        return messages
    if content.lstrip().startswith("</think>"):
        return messages
    out = list(messages)
    out[-1] = {**last, "content": "</think>\n\n" + content}
    return out


def apply_chat_template_safe(
    apply_fn, model: str | None, *args, _stage: str | None = None, **kwargs,
):
    """apply_chat_template with Qwen3.8 think kwargs; never fail the turn.

    Illegal effort (`high`) would jinja-raise. We only pass legal values,
    and if the tokenizer/template still TypeError/ValueError, retry with
    no extra kwargs instead of dropping to a naive prompt concat.
    `_stage` (harness turn stage) selects the reasoning effort; it is
    consumed here and never forwarded to `apply_fn`.
    """
    extra = chat_template_thinking_kwargs(model, stage=_stage)
    if extra:
        try:
            return apply_fn(*args, **kwargs, **extra)
        except (TypeError, ValueError):
            pass
    return apply_fn(*args, **kwargs)


def append_assistant_prefill(prompt: str, prefill_content: str) -> str:
    """Glue Continue.dev assistant prefill onto a chat-template prompt.

    Qwen3.x thinking templates (including xhigh) leave an OPEN `<think>`
    when thinking is on. Appending `<html_file>` there puts the whole
    first-build inside the reasoning channel (DK 20260815_085321). Close
    think first so max/xhigh still emits parser-visible code.
    """
    if not prefill_content:
        return prompt
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        return stripped + "\n</think>\n\n" + prefill_content
    return prompt + prefill_content


# -----------------------------------------------------------------------------
# Public types.
# -----------------------------------------------------------------------------


@dataclass
class BackendInfo:
    """Resolved backend identity. What the TUI prints, what the agent uses."""

    name: Literal["ollama", "mlx", "openai", "anthropic"]
    model: str
    source: str               # human-readable provenance ("loaded in ollama (/api/ps): 'qwen3.6:27b'")
    endpoint: str             # base URL — "http://127.0.0.1:11434" or "http://127.0.0.1:8080"
    context_length: int | None = None


@dataclass
class OllamaAutopinResult:
    """Outcome of best-effort per-GPU Ollama daemon setup for the TUI."""

    mode: Literal["off", "manual", "auto-pinned", "fallback"]
    message: str = ""
    endpoints: dict[int, str] | None = None


# Cloud-backend fallbacks — used when the Models API list is unreachable
# (offline, bad key, pytest). Live `/list` and short aliases (`fable`,
# `gpt`, `opus`) prefer GET /v1/models; see resolve_cloud_alias().
# API keys are read from env at request time — never from disk, never
# embedded in BackendInfo.
_OPENAI_DEFAULT_MODEL = "gpt-5.6"
_OPENAI_MINI_FALLBACK = "gpt-5-mini"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
_ANTHROPIC_FABLE_FALLBACK = "claude-fable-5-1"
_ANTHROPIC_SONNET_FALLBACK = "claude-sonnet-4-6"
_ANTHROPIC_HAIKU_FALLBACK = "claude-haiku-4-5"
_OPENAI_MODELS: tuple[str, ...] = (_OPENAI_DEFAULT_MODEL,)
# Curated /list fallback — live inventory replaces these when the
# Models API answers.
_ANTHROPIC_MODELS: tuple[str, ...] = (
    _ANTHROPIC_FABLE_FALLBACK,
    _ANTHROPIC_DEFAULT_MODEL,
)


class Backend(ABC):
    """Common interface implemented by OllamaBackend, MLXBackend,
    OpenAIBackend, and AnthropicBackend."""

    info: BackendInfo

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        # `on_progress(stage, current, total)` is fired during the
        # pre-token phase when the backend exposes progress. Today only
        # MLX surfaces it (mlx_lm's prompt_progress_callback):
        #   stage="prompt_eval", current=N, total=M
        # Ollama ignores the parameter — its API doesn't expose
        # prompt-processing progress mid-stream.
        #
        # `cancel_event` (asyncio.Event) lets the caller request a
        # mid-stream stop — set by the TUI when the user hits Ctrl-D so
        # the agent doesn't have to wait until the current iter finishes.
        # MLXBackend polls it between tokens. OllamaBackend currently
        # accepts but does not act on it (the ollama Python client has
        # its own retry/stall flow); a cancel still works by cancelling
        # the consuming asyncio task.
        ...

    @abstractmethod
    async def is_vlm(self) -> bool:
        ...

    async def best_of_n(
        self,
        messages: list[dict],
        *,
        n: int = 3,
        temperatures: Iterable[float] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        scorer: Callable[[str], Awaitable[tuple[float, dict]]],
        on_progress: Callable[[int, str], None] | None = None,
        early_exit_score: float = 1.0,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[Candidate, list[Candidate]]:
        """Sequential best-of-N with early exit. Backend-agnostic.

        Local LLMs serialize generation at the daemon (Ollama) or the
        single-model-per-process (MLX), so parallel sampling never
        helps — it just queues the second candidate behind the first
        and trips the stall watchdog. Sequential with early exit is
        the right shape on both backends.
        """
        if temperatures is None:
            temperatures = [0.2, 0.6, 0.9][:n]
        temps = list(temperatures)
        if len(temps) < n:
            temps += [temps[-1]] * (n - len(temps))

        base_options = dict(options or {})
        cands: list[Candidate] = []
        for i, t in enumerate(temps):
            opts = dict(base_options)
            opts["temperature"] = t
            if on_progress is not None:
                on_progress(i, f"start (T={t})")
            if cancel_event is not None and cancel_event.is_set():
                break
            result = await self.stream_chat(
                messages,
                on_token=None,
                options=opts,
                keep_alive=keep_alive,
                stall_seconds=stall_seconds,
                overall_seconds=overall_seconds,
                max_retries=0,
                cancel_event=cancel_event,
            )
            if on_progress is not None:
                tag = "stalled" if result.stalled else f"{result.tokens} tok in {result.duration_s:.1f}s"
                on_progress(i, f"generated ({tag})")
            try:
                score, extra = await scorer(result.text)
            except Exception as e:
                score, extra = -1.0, {"scorer_error": str(e)}
            if on_progress is not None:
                on_progress(i, f"scored {score:+.2f}")
            cands.append(Candidate(
                text=result.text,
                score=score,
                extra=extra,
                tokens=result.tokens,
                duration_s=result.duration_s,
                stalled=result.stalled,
            ))
            if score >= early_exit_score:
                if on_progress is not None:
                    on_progress(i, f"early-exit at {score:+.2f}")
                break
        cands.sort(key=lambda c: (c.score, -c.duration_s), reverse=True)
        return cands[0], cands

    async def warm_prefix(
        self,
        messages: list[dict],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """Pre-fill this backend's KV cache for `messages` without
        producing meaningful output.

        Used to hide cross-slot prompt-reprefill cost: when role A
        streams on slot A and then control hands off to role B on slot
        B, slot B has never seen this conversation and pays the full
        prefill on its first request. The 2026-05-22 chess trace had
        slot 1 (coder) idle for 58 s during architect + asset/sound gen,
        then iter 1 spent its own prefill time on tokens slot 3
        (architect) had already processed. Firing this method during
        that idle window means slot 1's KV is hot before iter 1's
        stream starts.

        Default implementation: stream_chat with 1-token cap, output
        discarded. Ollama caches the prompt KV across requests by
        prefix-match; a subsequent stream_chat with the SAME messages
        reuses the cached KV. Subclasses can override for backends that
        expose a cheaper prompt-only path.

        Returns a dict with `ok`, `elapsed_s`, optional `tokens`,
        `error`. Never raises — the caller treats warm as advisory.
        """
        import time as _time
        opts = dict(options or {})
        # Two common knobs; backends pick whichever they honor.
        opts.setdefault("num_predict", 1)
        opts.setdefault("max_tokens", 1)
        opts.setdefault("temperature", 0.0)
        started_at = _time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.stream_chat(
                    messages,
                    on_token=None,
                    options=opts,
                    keep_alive=keep_alive,
                    stall_seconds=timeout_s,
                    overall_seconds=timeout_s,
                    max_retries=0,
                ),
                timeout=timeout_s,
            )
            return {
                "ok": True,
                "elapsed_s": round(_time.monotonic() - started_at, 2),
                "tokens": getattr(result, "tokens", None),
                "stalled": getattr(result, "stalled", False),
            }
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "elapsed_s": round(_time.monotonic() - started_at, 2),
                "error": "timeout",
            }
        except Exception as e:
            return {
                "ok": False,
                "elapsed_s": round(_time.monotonic() - started_at, 2),
                "error": str(e)[:200],
            }

    async def close(self) -> None:
        """Best-effort cleanup of any underlying connection pools."""
        return None


# -----------------------------------------------------------------------------
# Ollama implementation.
# -----------------------------------------------------------------------------


class OllamaBackend(Backend):
    """Wraps `ollama.AsyncClient` so the existing Ollama code path stays intact."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        # `ollama.AsyncClient` reads OLLAMA_HOST internally. We pass `host`
        # explicitly only when info.endpoint differs from the default so we
        # don't override the user's env in unexpected ways.
        if info.endpoint and info.endpoint not in ("http://127.0.0.1:11434", "http://localhost:11434"):
            self._client = ollama.AsyncClient(host=info.endpoint)
        else:
            self._client = ollama.AsyncClient()

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        # `on_progress` and `cancel_event` are accepted for API symmetry
        # with MLXBackend. Ollama's /api/chat doesn't expose prompt-eval
        # progress mid-stream; cancellation here propagates through
        # task.cancel() — the ollama AsyncClient closes its socket and
        # the call unwinds.
        # Harness-internal `_stage` (agent_stream plan cap / effort) must
        # not reach Ollama's options; Ollama spells the output cap
        # `num_predict`, so translate `max_tokens` when the caller set one.
        if options and ("_stage" in options or "max_tokens" in options):
            options = {k: v for k, v in options.items() if not k.startswith("_")}
            if "max_tokens" in options:
                options.setdefault("num_predict", int(options.pop("max_tokens")))
        return await stream_chat_with_retry(
            self._client,
            self.info.model,
            messages,
            on_token=on_token,
            options=options,
            keep_alive=keep_alive,
            stall_seconds=stall_seconds,
            overall_seconds=overall_seconds,
            max_retries=max_retries,
            on_stall=on_stall,
        )

    async def is_vlm(self) -> bool:
        try:
            info = await self._client.show(model=self.info.model)
        except Exception:
            return False
        caps = getattr(info, "capabilities", None) or []
        return any(str(c).lower() == "vision" for c in caps)


# -----------------------------------------------------------------------------
# MLX implementation.
# -----------------------------------------------------------------------------


# --- In-process MLX cross-turn KV prompt cache (Sept 2026) ------------------
# Before this, `stream_generate` ran with no `prompt_cache`, so every turn
# re-prefilled the whole conversation (60-120 s on 24-35k-token prompts).
# The agent loop is append-only most of the time (system + history stay
# byte-identical; only the newest user turn changes), so we keep the KV
# cache from the previous turn and only prefill the divergent suffix.
# MLX_PROMPT_CACHE=0 disables. Minimum shared prefix worth reusing:
_MLX_PROMPT_CACHE_MIN_REUSE_TOKENS = 64


def mlx_prompt_cache_enabled() -> bool:
    raw = (os.environ.get("MLX_PROMPT_CACHE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def plan_prompt_cache_reuse(
    cached_tokens: list[int] | None,
    new_tokens: list[int],
    *,
    trimmable: bool,
    min_reuse: int = _MLX_PROMPT_CACHE_MIN_REUSE_TOKENS,
) -> tuple[int, int]:
    """Decide how much of the previous turn's KV cache to keep.

    Returns ``(reuse, trim)``:
      * ``reuse`` — number of leading tokens of ``new_tokens`` already in
        the cache (0 → start from an empty cache).
      * ``trim``  — number of trailing cache entries to drop first
        (cache held ``len(cached_tokens)`` tokens; only ``reuse`` match).

    Rules (pure, unit-tested — no Metal):
      * Common prefix shorter than ``min_reuse`` → (0, 0): not worth it.
      * At least ONE new token must be fed to the model, so a prompt that
        equals the cached sequence exactly reuses ``len(new)-1``.
      * A non-trimmable cache (rotating / hybrid layers) can only be
        reused when nothing needs trimming (pure append); otherwise reset.
    """
    if not cached_tokens or not new_tokens:
        return 0, 0
    n = min(len(cached_tokens), len(new_tokens))
    k = 0
    while k < n and cached_tokens[k] == new_tokens[k]:
        k += 1
    if k >= len(new_tokens):
        k = len(new_tokens) - 1
    if k < min_reuse:
        return 0, 0
    trim = len(cached_tokens) - k
    if trim and not trimmable:
        return 0, 0
    return k, trim


def _usage_cached_prompt_tokens(usage: Any) -> int | None:
    """Pull the prefix-cache hit count out of an OpenAI-style `usage` dict.

    Accepts both the standard `prompt_tokens_details.cached_tokens` and a
    flat `cached_tokens` / `prompt_cache_hit_tokens` (DeepSeek-style).
    """
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return int(details["cached_tokens"])
    for key in ("cached_tokens", "prompt_cache_hit_tokens"):
        if isinstance(usage.get(key), int):
            return int(usage[key])
    return None


def omlx_hot_cache_status(endpoint: str | None) -> str | None:
    """Read the local oMLX settings file and report its KV hot-cache size.

    Returns ``"off"`` when `cache.hot_cache_max_size` is 0 (prefix reuse
    disabled — every turn re-prefills), the configured size string when
    set, or ``None`` when unknown (remote endpoint, no file, unreadable).
    Only consulted for loopback endpoints: the settings file describes the
    oMLX on THIS machine.
    """
    try:
        if not endpoint or not endpoint_is_omlx(endpoint):
            return None
        from urllib.parse import urlparse
        host = (urlparse(endpoint).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            return None
        path = os.path.expanduser("~/.omlx/settings.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        cache_cfg = data.get("cache") if isinstance(data, dict) else None
        if not isinstance(cache_cfg, dict):
            return None
        size = cache_cfg.get("hot_cache_max_size")
        if size is None:
            return None
        text = str(size).strip()
        if text in ("0", "0B", "0GB", "0MB"):
            return "off"
        return text
    except Exception:
        return None


# MLX request fields we forward when present in the agent's `options` dict.
# Anything else (including `num_ctx`) is silently dropped — MLX uses the
# model's native context, no equivalent knob.
_MLX_OPTION_KEYS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p",
    "seed", "max_tokens",
)

# BATTLEZO 20260904_095910: first-token watchdog for MLXServerBackend when
# the model is already resident (see _stream_once). min() with stall_seconds.
_MLX_SERVER_FIRST_TOKEN_WATCHDOG_S = 300.0

# SSE keepalive / prompt-eval progress inside mlx_lm.server streams.
_MLX_PROGRESS_RE = re.compile(
    r"(?:keepalive|Prompt processing progress)[:\s]+(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)

# DeepSeek-V4 Flash needs an even smaller prefill chunk than Pro
# because its Indexer attention path materializes O(L^2 * k) Metal
# buffers per chunk and crashes mid-stream at chunk_size > 512
# (observed 2026-05-15 DK trace: 11K-token generation crashed after
# a 12K-prompt prefill at chunk_size 1024). General default stays at
# 1024 for everything else; Flash auto-downshifts via
# `_resolve_prefill_step_size`. Env MLX_PREFILL_STEP_SIZE always wins.
# (The earlier `scripts/mlx_v4_server.sh` wrapper that passed this
# flag to the separate HTTP server has been removed — MLX runs
# in-process now, so the flag is applied here directly.)
_MLX_PREFILL_STEP_SIZE_DEFAULT = 1024
_MLX_PREFILL_STEP_SIZE_FLASH = 512


def _resolve_prefill_step_size(model_path: str) -> int:
    """Pick the prefill chunk size for `model_path`.

    Env override (`MLX_PREFILL_STEP_SIZE`) wins. Otherwise, model
    names containing 'flash' (case-insensitive) get 512 — empirically
    required for DeepSeek-V4 Flash; safe for any other model that
    happens to share the substring. Everything else gets 1024.
    """
    env_val = (os.environ.get("MLX_PREFILL_STEP_SIZE") or "").strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    if "flash" in (model_path or "").lower():
        return _MLX_PREFILL_STEP_SIZE_FLASH
    return _MLX_PREFILL_STEP_SIZE_DEFAULT


def _mark_open_fds_noninheritable() -> None:
    """Stop Playwright/oMLX pipes from being passed into mlx load forks.

    After Chromium is up, mlx_lm/mlx_vlm.load (huggingface/safetensors)
    forks and `_posixsubprocess.fork_exec` raises
    `ValueError: bad value(s) in fds_to_keep` (BATTLEZ2/4/5/6 20260904 —
    Qwen3.8/3.6 died at 0 tokens once GLM work started Playwright first).
    Same trap as `assets.preload()` in chat.py main().
    """
    fd_dir = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return
    for name in names:
        try:
            fd = int(name)
        except ValueError:
            continue
        if fd < 3:
            continue
        try:
            os.set_inheritable(fd, False)
        except OSError:
            pass


_SPAWNV_PASSFDS_PATCHED = False


def _install_spawnv_passfds_filter() -> None:
    """Drop closed FDs from multiprocessing spawn (Playwright leftovers).

    `subprocess.Popen(close_fds=False)` is not enough: huggingface/tokenizers
    use `multiprocessing.util.spawnv_passfds`, which still calls
    `_posixsubprocess.fork_exec` and raises fds_to_keep (BATTLEZ8/9 20260904).
    Filtering invalid fds is safe for Playwright's own spawns. GLM/oMLX HTTP
    never hits this.
    """
    global _SPAWNV_PASSFDS_PATCHED
    if _SPAWNV_PASSFDS_PATCHED:
        return
    try:
        import multiprocessing.util as mp_util
    except Exception:
        return
    orig = getattr(mp_util, "spawnv_passfds", None)
    if orig is None:
        return

    def _filtered(path, args, passfds):
        cleaned = []
        for fd in passfds:
            try:
                os.fstat(int(fd))
            except (OSError, TypeError, ValueError):
                continue
            cleaned.append(fd)
        return orig(path, args, cleaned)

    mp_util.spawnv_passfds = _filtered
    _SPAWNV_PASSFDS_PATCHED = True


@contextmanager
def _mlx_subprocess_fork_guard():
    """Let in-process mlx load spawn while Playwright pipes are already open.

    `close_fds=True` (Python default) walks the fd table and raises
    `ValueError: bad value(s) in fds_to_keep` when Chromium has dangling
    pipes. BATTLEZ7 20260904: Qwen3.8 died in 0.79s at planning because
    the GLM session had already started the browser; set_inheritable
    alone was not enough. GLM/oMLX HTTP never enters this guard.
    """
    _install_spawnv_passfds_filter()
    _mark_open_fds_noninheritable()
    orig = subprocess.Popen

    def _popen(*args, **kwargs):
        kwargs["close_fds"] = False
        kwargs.pop("pass_fds", None)
        return orig(*args, **kwargs)

    subprocess.Popen = _popen
    try:
        yield
    finally:
        subprocess.Popen = orig


class MLXBackend(Backend):
    """In-process MLX backend. Loads the model into this process's GPU
    VRAM on first request and streams generations via
    `mlx_lm.stream_generate`. No HTTP, no `mlx_lm.server`.

    Model + tokenizer are held at the class level so subsequent
    requests within a session reuse the loaded weights. A 27B mxfp8
    model is ~15 GB; on Apple unified memory this coexists fine with
    Z-Image-Turbo and Chromium on 64 GB+ Macs.

    Cancellation: the worker thread that iterates `stream_generate`
    checks a `threading.Event` between yields. When the agent's
    `_stop_event` is set (Ctrl-D in the TUI), the next iteration
    exits cleanly and `stream_chat` returns the partial result with
    `stalled=True`. The asyncio caller can also be cancelled — that
    raises `CancelledError` in `stream_chat`, which sets the worker
    event and re-raises so the agent's run-loop can wind down.
    """

    # Class-level model cache. Switching MLX_MODEL between sessions
    # frees the previous model first to keep VRAM bounded.
    #
    # Two slots: text-only models load via `mlx_lm` and use
    # (_loaded_model, _loaded_tokenizer); VLM models load via
    # `mlx_vlm` and use (_loaded_vlm_model, _loaded_vlm_processor,
    # _loaded_vlm_config). Only one slot at a time should hold
    # weights for any given path — `_load_sync` and `_load_vlm_sync`
    # both null the OTHER slot on a new load to keep VRAM bounded.
    _loaded_model: Any = None
    _loaded_tokenizer: Any = None
    _loaded_path: str | None = None
    _loaded_vlm_model: Any = None
    _loaded_vlm_processor: Any = None
    _loaded_vlm_config: Any = None
    _loaded_vlm_path: str | None = None
    # Cross-turn KV prompt cache (text pipeline only; see
    # `plan_prompt_cache_reuse`). `_prompt_cache_tokens` is the exact token
    # sequence the cache currently holds (prompt + generated ids). Dropped
    # with the weights in `release_weights` / on a different-path load.
    _prompt_cache: Any = None
    _prompt_cache_tokens: list[int] | None = None
    _load_lock: asyncio.Lock | None = None
    # All MLX work runs on this single dedicated thread. MLX/Metal
    # binds GPU contexts to the calling thread; if we loaded the model
    # on a worker from the asyncio default executor and then ran
    # stream_generate on a different threading.Thread, Metal would
    # segfault deep in Mtl/objc code (seen on macOS 26 + DeepSeek-V4).
    # Pinning everything to one thread eliminates that class of crash.
    _mlx_thread: Any = None  # concurrent.futures.ThreadPoolExecutor

    def __init__(self, info: BackendInfo) -> None:
        self.info = info

    @classmethod
    def _get_load_lock(cls) -> asyncio.Lock:
        if cls._load_lock is None:
            cls._load_lock = asyncio.Lock()
        return cls._load_lock

    @classmethod
    def _get_mlx_executor(cls):
        """Single-thread executor that owns the Metal context."""
        if cls._mlx_thread is None:
            from concurrent.futures import ThreadPoolExecutor
            # Daemon=True so we don't block process exit on shutdown.
            cls._mlx_thread = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mlx",
            )
        return cls._mlx_thread

    @classmethod
    def _clear_metal_cache(cls) -> None:
        try:
            import mlx.core as mx  # type: ignore
            metal = getattr(mx, "metal", None)
            if metal is not None and hasattr(metal, "clear_cache"):
                metal.clear_cache()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

    @classmethod
    def release_weights(cls, *, wait_for_metal: bool = False) -> str | None:
        """Drop cached MLX/VLM weights and clear the Metal allocator.

        Returns the path that was resident (if any). Safe from any thread.
        When called from the dedicated MLX worker thread, Metal cleanup runs
        inline (submitting back to the same single-thread executor deadlocks).
        """
        freed_path = cls._loaded_path or cls._loaded_vlm_path
        cls._loaded_model = None
        cls._loaded_tokenizer = None
        cls._loaded_path = None
        # KV prompt cache belongs to these weights — drop it with them.
        cls._prompt_cache = None
        cls._prompt_cache_tokens = None
        cls._loaded_vlm_model = None
        cls._loaded_vlm_processor = None
        cls._loaded_vlm_config = None
        cls._loaded_vlm_path = None
        import gc
        gc.collect()

        import threading
        on_mlx_thread = threading.current_thread().name.startswith("mlx")
        if on_mlx_thread:
            cls._clear_metal_cache()
            return freed_path

        try:
            fut = cls._get_mlx_executor().submit(cls._clear_metal_cache)
            if wait_for_metal:
                fut.result(timeout=30)
        except Exception:
            pass
        return freed_path

    @classmethod
    def _drop_after_crash(cls) -> None:
        """Free GPU state after the MLX worker raised mid-stream.

        Without this, a single crash made the rest of the session
        unusable until the user killed and restarted chat.py (the
        process-wide Metal allocator was full of dead tensors).
        """
        cls.release_weights()

    @classmethod
    def _load_sync(cls, path: str) -> tuple[Any, Any]:
        """Blocking load. MUST run on the dedicated MLX thread (see
        `_get_mlx_executor`) — Metal binds to the calling thread.
        """
        if cls._loaded_path == path and cls._loaded_model is not None:
            return cls._loaded_model, cls._loaded_tokenizer
        # Drop any resident MLX/VLM weights before loading a different path.
        if cls._loaded_model is not None or cls._loaded_vlm_model is not None:
            cls.release_weights()
        # Defer the mlx_lm import so Ollama-only users don't pay its
        # ~1-2 s cold-start cost. Import+load both fork — guard while
        # Playwright pipes may already be open (BATTLEZ7 20260904).
        with _mlx_subprocess_fork_guard():
            from mlx_lm import load as _mlx_load  # type: ignore
            model, tokenizer = _mlx_load(path)
        cls._loaded_model = model
        cls._loaded_tokenizer = tokenizer
        cls._loaded_path = path
        # Free the VLM slot if it's holding a different model — only
        # one model should be in VRAM at a time.
        if cls._loaded_vlm_model is not None and cls._loaded_vlm_path != path:
            cls._loaded_vlm_model = None
            cls._loaded_vlm_processor = None
            cls._loaded_vlm_config = None
            cls._loaded_vlm_path = None
            import gc as _gc
            _gc.collect()
        return model, tokenizer

    @classmethod
    def _load_vlm_sync(cls, path: str) -> tuple[Any, Any, Any]:
        """Blocking load via `mlx_vlm`. Returns (model, processor, config).
        MUST run on the dedicated MLX thread.

        Used when the model NAME classifies as a VLM (e.g. Qwen3.6-27B,
        LLaVA, MiniCPM-V). The mlx_vlm pipeline loads BOTH the language
        model and the vision tower — same files on disk as the mlx_lm
        load, different python objects, different VRAM footprint.
        """
        if (
            cls._loaded_vlm_path == path
            and cls._loaded_vlm_model is not None
        ):
            return (
                cls._loaded_vlm_model,
                cls._loaded_vlm_processor,
                cls._loaded_vlm_config,
            )
        # Free either slot if it holds a different (or any) model.
        if cls._loaded_vlm_model is not None or cls._loaded_model is not None:
            cls.release_weights()
        with _mlx_subprocess_fork_guard():
            from mlx_vlm import load as _vlm_load  # type: ignore
            from mlx_vlm.utils import load_config as _vlm_load_config  # type: ignore
            model, processor = _vlm_load(path)
            config = _vlm_load_config(path)
        cls._loaded_vlm_model = model
        cls._loaded_vlm_processor = processor
        cls._loaded_vlm_config = config
        cls._loaded_vlm_path = path
        return model, processor, config

    def warm_load(self) -> None:
        """Load weights on the MLX thread before Playwright is up.

        mlx_lm/mlx_vlm.load forks. After Chromium opens IPC pipes that
        fork raises fds_to_keep (BATTLEZ4 20260904 Qwen3.8). Same trap
        as assets.preload() in chat.py main(). Idempotent if cached.
        """
        path = self.info.model
        is_vlm = classify_model_modality(path) == "vlm"
        if is_vlm:
            try:
                import mlx_vlm  # noqa: F401
            except ImportError:
                is_vlm = False

        def _do() -> None:
            if is_vlm:
                self._load_vlm_sync(path)
            else:
                self._load_sync(path)

        self._get_mlx_executor().submit(_do).result(timeout=600)

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        # max_retries is accepted for API symmetry with OllamaBackend
        # but is a no-op in-process: retrying against the same loaded
        # model with the same prompt produces the same result (modulo
        # sampler temperature, which the caller controls).
        return await self._stream_once(
            messages,
            on_token=on_token,
            options=options,
            stall_seconds=stall_seconds,
            overall_seconds=overall_seconds,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    async def _stream_once(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None,
        options: dict[str, Any] | None,
        stall_seconds: float,
        overall_seconds: float,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        opts = dict(options or {})
        # Harness turn stage (agent_stream._stream) → Qwen reasoning effort.
        turn_stage = opts.get("_stage")
        # Drop fields that mean nothing to MLX (e.g. num_ctx). Carry
        # only the sampler knobs MLX understands.
        sampler_opts = {k: opts[k] for k in _MLX_OPTION_KEYS if k in opts}
        # Output cap. Current generation of local MLX models — Qwen3.6,
        # DeepSeek V4, GLM 5.1, MiniMax M2 — all ship with 256K+ native
        # context, and the 16384 / 131072 caps in prior versions
        # truncated full <html_file> rewrites mid-stream.
        #
        # Measured peak across the donkey-kong session that motivated
        # the recent default bumps: 5 659 completion tokens. That's
        # 4.3% of 131 K and 2.2% of 256 K. Real coding-game workloads
        # don't come anywhere near either cap. The number's job is
        # purely "don't be the bottleneck" — a runaway generation
        # guard, not a working limit. 131 072 gives ~23x headroom
        # over observed peaks, leaves room for unusually long full
        # rewrites, and matches what the agent loop already chooses
        # for its own truncation heuristic. Per-machine override via
        # MLX_MAX_TOKENS env var when a model genuinely needs more.
        env_cap = os.environ.get("MLX_MAX_TOKENS", "").strip()
        if env_cap.isdigit() and int(env_cap) > 0:
            default_max = int(env_cap)
        else:
            default_max = 131072
        max_tokens = int(sampler_opts.get("max_tokens") or default_max)
        temperature = float(sampler_opts.get("temperature") or 0.0)
        # Tail-truncation defaults (added 2026-05-31). PRIOR behavior left
        # top_p/top_k/min_p at 0 whenever a caller didn't set them — and
        # callers only ever pass `temperature` (see agent.py), so EVERY MLX
        # turn sampled at temp>0 over the FULL vocabulary with NO nucleus or
        # top-k truncation. mlx_lm's make_sampler skips each filter unless
        # top_p in (0,1) / top_k>0 / min_p>0, so zeros = "no filter". That is
        # the danger zone Qwen's own docs warn against: "DO NOT use greedy
        # decoding ... can lead to endless repetitions", and the same applies
        # to an untruncated tail — once the model emits a structurally
        # identical line (e.g. `let cpuIsBlocking=false;`) nothing pulls it off
        # that attractor. The 2026-05-31 dojo-fight trace died exactly this way
        # twice (run_20260531_214215): repetition-loop abort mid-`<html_file>`,
        # zero usable builds.
        #
        # Defaults are the VENDOR thinking-mode / precise-coding preset for
        # Qwen3.6 (temp 0.6, top_p 0.95, top_k 20, min_p 0; repetition_penalty
        # stays 1.0 — a rep penalty HURTS code, which legitimately repeats `}`,
        # `const`, `ctx.`). These are model-agnostic good hygiene: sane tail
        # truncation helps every local model, not just Qwen. Per-machine
        # override via MLX_TOP_P / MLX_TOP_K / MLX_MIN_P; a caller that passes
        # a positive value still wins. We do NOT inject a temperature default
        # here — greedy (temp=0) planning stages bypass the sampler entirely
        # (make_sampler returns argmax), and explicit per-stage temps must
        # pass through untouched.
        def _env_float(name: str, fallback: float) -> float:
            raw = os.environ.get(name, "").strip()
            try:
                return float(raw) if raw else fallback
            except ValueError:
                return fallback

        def _env_int(name: str, fallback: int) -> int:
            raw = os.environ.get(name, "").strip()
            return int(raw) if raw.lstrip("-").isdigit() else fallback

        top_p = float(sampler_opts.get("top_p") or 0.0) or _env_float("MLX_TOP_P", 0.95)
        top_k = int(sampler_opts.get("top_k") or 0) or _env_int("MLX_TOP_K", 20)
        min_p = float(sampler_opts.get("min_p") or 0.0) or _env_float("MLX_MIN_P", 0.0)

        prefill_step_size = _resolve_prefill_step_size(self.info.model)

        started = time.monotonic()
        # Last-activity timestamp for the stall watchdog. Bumped on
        # EITHER (a) token emission, or (b) prefill progress
        # (prompt_progress_callback firing). Old behavior measured
        # stall purely as wall-clock since `started`, which killed
        # streams during long prefills (DK trace 20260513_173528:
        # 17K-token prompt on cold-loaded DeepSeek-V4 takes >60s
        # to prefill before any generation token; the watchdog fired
        # at exactly 60.00s with 0 tokens). MLX has been doing real
        # work the whole time — the stall watchdog just couldn't see
        # it.
        last_activity_at = started
        parts: list[str] = []
        n_tokens = 0
        stalled = False
        looped = False
        silent = False
        stall_at: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        cached_prompt_tokens: int | None = None  # KV prefix reused this turn
        # Shared repetition detector — same class both backends use.
        repeat = RepetitionDetector()
        # A2: shared deliberation detector for unique-text reasoning loops.
        delib = DeliberationDetector()
        deliberated = False
        # Diagnose-bloat guard (chess-trace fix): abort an unclosed <diagnose>.
        diag = DiagnoseBloatDetector()
        diagnose_bloat = False
        loop_grace_used = False
        loop_grace_reason: str | None = None

        # Route decision: VLM models (Qwen3.6-27B, LLaVA, MiniCPM-V,
        # etc.) go through `mlx_vlm` so the agent can pass screenshot
        # bytes per-iter. Text-only goes through `mlx_lm` as before.
        # `mlx_vlm` must be importable; if it's not installed but the
        # name classifies as VLM, fall back to text-only mode so the
        # session still works (with images silently dropped — same
        # behavior as before 2026-05-15).
        is_vlm_model = classify_model_modality(self.info.model) == "vlm"
        if is_vlm_model:
            try:
                # First import can subprocess-fork (tokenizers/hf). Same
                # fds_to_keep trap as load if Chromium is already up.
                with _mlx_subprocess_fork_guard():
                    import mlx_vlm  # noqa: F401
            except ImportError:
                is_vlm_model = False
        # Track separately so the load-check below picks the right slot.
        needs_load = (
            (
                is_vlm_model
                and (
                    self._loaded_vlm_path != self.info.model
                    or self._loaded_vlm_model is None
                )
            )
            or (
                not is_vlm_model
                and (
                    self._loaded_path != self.info.model
                    or self._loaded_model is None
                )
            )
        )
        if needs_load and on_progress is not None:
            try:
                # Visible signal so the TUI shows "loading MLX model"
                # instead of a silent multi-second wait on first request.
                on_progress("mlx_load", 0, 1)
            except Exception:
                pass

        # asyncio <-> dedicated-MLX-thread bridge. Load + stream_generate
        # both run on the same single-thread executor; the worker pushes
        # tuples; the consumer below reads them with a per-item timeout.
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        worker_cancel = threading.Event()

        def _prompt_progress(cur: int, tot: int) -> None:
            # Runs on the MLX thread. Hop back to the event loop AND
            # bump last_activity_at so the stall watchdog knows the
            # backend is still doing real prefill work — even with
            # zero generated tokens. Without this, long prefills on
            # big prompts trip the stall guard before generation can
            # even start.
            loop.call_soon_threadsafe(_bump_activity)
            if on_progress is not None:
                loop.call_soon_threadsafe(
                    lambda c=cur, t=tot: _safe_call(on_progress, "prompt_eval", c, t)
                )

        def _bump_activity() -> None:
            nonlocal last_activity_at
            last_activity_at = time.monotonic()

        def _safe_call(fn: Callable, *args) -> None:
            try:
                fn(*args)
            except Exception:
                pass

        # The full pipeline runs in one thread:
        #   1. Load model+tokenizer if not cached.
        #   2. Apply chat template.
        #   3. Iterate stream_generate, pushing each delta to the queue.
        # Doing everything here ensures the Metal context is bound to
        # this single thread for the entire lifetime of the model.
        info_model = self.info.model

        # Strip "images" from each message before chat-template
        # rendering — chat templates expect string content. We also
        # collect the bytes here so the VLM pipeline can write them
        # to temp files. The "images" key is what `agent._stream`
        # sets when attaching a screenshot to a turn.
        def _split_images(msgs: list[dict]) -> tuple[list[dict], list[bytes]]:
            cleaned: list[dict] = []
            images: list[bytes] = []
            for m in msgs:
                imgs = m.get("images") if isinstance(m, dict) else None
                if imgs:
                    images.extend(b for b in imgs if isinstance(b, (bytes, bytearray)))
                # Keep only text-template-safe keys.
                cleaned.append({
                    k: v for k, v in m.items()
                    if k in ("role", "content")
                })
            return cleaned, images

        cleaned_messages, image_bytes_list = _split_images(messages)

        def _pipeline_textonly() -> None:
            try:
                model, tokenizer = self._load_sync(info_model)
                if needs_load:
                    loop.call_soon_threadsafe(q.put_nowait, ("loaded", None, None, None))
                # Build prompt via chat template. Falls back to a naive
                # role/content concat if the tokenizer lacks the template
                # (rare with modern Instruct models).
                # Support assistant prefill: if the last message is an assistant message,
                # we run the chat template on the preceding messages with add_generation_prompt=True,
                # and then manually append the assistant message content to the prompt.
                has_prefill = (len(cleaned_messages) > 0 and cleaned_messages[-1].get("role") == "assistant")
                if has_prefill:
                    history = cleaned_messages[:-1]
                    prefill_content = cleaned_messages[-1].get("content", "")
                else:
                    history = cleaned_messages
                    prefill_content = ""

                try:
                    prompt = apply_chat_template_safe(
                        tokenizer.apply_chat_template,
                        info_model,
                        history,
                        tokenize=False,
                        add_generation_prompt=True,
                        _stage=turn_stage,
                    )
                except Exception:
                    prompt = "\n\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in history
                    ) + "\n\nassistant:"

                if has_prefill:
                    prompt = append_assistant_prefill(prompt, prefill_content)

                from mlx_lm.sample_utils import make_sampler  # type: ignore
                sampler = make_sampler(
                    temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k,
                )

                from mlx_lm import stream_generate  # type: ignore
                # --- Cross-turn KV prompt cache -----------------------------
                # Tokenize here (same bos rule as stream_generate) so the
                # common prefix with the previous turn can be measured and
                # only the divergent suffix is prefilled. Full prompt length
                # is reported as prompt_tokens; the reused count rides the
                # "done" payload (StreamResult.cached_prompt_tokens).
                gen_kwargs: dict[str, Any] = {}
                n_prompt_full: int | None = None
                cached_reused = 0
                gen_token_ids: list[int] = []
                cache_active = False
                if mlx_prompt_cache_enabled():
                    try:
                        from mlx_lm.models import cache as _mlx_cache  # type: ignore
                        import mlx.core as _mx  # type: ignore
                        bos = getattr(tokenizer, "bos_token", None)
                        add_special = bos is None or not prompt.startswith(bos)
                        full_ids = list(tokenizer.encode(prompt, add_special_tokens=add_special))
                        n_prompt_full = len(full_ids)
                        pc = self.__class__._prompt_cache
                        pc_tokens = self.__class__._prompt_cache_tokens
                        if pc is None or pc_tokens is None:
                            pc = _mlx_cache.make_prompt_cache(model)
                            pc_tokens = []
                        reuse, trim = plan_prompt_cache_reuse(
                            pc_tokens, full_ids,
                            trimmable=_mlx_cache.can_trim_prompt_cache(pc),
                        )
                        if reuse == 0 and pc_tokens:
                            # Prefix diverged too early (or untrimmable) —
                            # start from a fresh cache instead of trimming.
                            pc = _mlx_cache.make_prompt_cache(model)
                            pc_tokens = []
                        elif trim:
                            _mlx_cache.trim_prompt_cache(pc, trim)
                        cached_reused = reuse
                        gen_kwargs["prompt_cache"] = pc
                        prompt = _mx.array(full_ids[reuse:])
                        # Cache will hold the full prompt once prefilled;
                        # generated ids are appended as they stream.
                        self.__class__._prompt_cache = pc
                        self.__class__._prompt_cache_tokens = list(full_ids)
                        cache_active = True
                    except Exception:
                        # Any cache plumbing failure → plain uncached turn.
                        gen_kwargs = {}
                        cached_reused = 0
                        cache_active = False
                        self.__class__._prompt_cache = None
                        self.__class__._prompt_cache_tokens = None
                last_gen = None
                for gen in stream_generate(
                    model, tokenizer, prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_progress_callback=_prompt_progress,
                    prefill_step_size=prefill_step_size,
                    **gen_kwargs,
                ):
                    # Record the id BEFORE the cancel check: generate_step
                    # has already fed every yielded token to the model, so
                    # the cache holds it even if we stop consuming here.
                    if cache_active:
                        gen_token_ids.append(int(gen.token))
                    if worker_cancel.is_set():
                        break
                    last_gen = gen
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        (
                            "text", gen.text,
                            n_prompt_full if n_prompt_full is not None else gen.prompt_tokens,
                            gen.generation_tokens,
                        ),
                    )
                if cache_active and self.__class__._prompt_cache_tokens is not None:
                    # generate_step prefetches one step ahead, so every
                    # yielded token has already been fed to the model —
                    # the cache holds prompt + all yielded ids.
                    self.__class__._prompt_cache_tokens.extend(gen_token_ids)
                pt = (
                    n_prompt_full if n_prompt_full is not None
                    else (getattr(last_gen, "prompt_tokens", None) if last_gen else None)
                )
                ct = getattr(last_gen, "generation_tokens", None) if last_gen else None
                loop.call_soon_threadsafe(
                    q.put_nowait, ("done", cached_reused if cache_active else None, pt, ct)
                )
            except BaseException as e:  # noqa: BLE001 - surface MLX errors too
                # Format the exception ON this thread — the consumer
                # is in asyncio land and doesn't have the live frame.
                # `format_exception_only` works for BaseException
                # subclasses (MemoryError, Metal RuntimeError, ...).
                import traceback as _tb
                err_text = "".join(
                    _tb.format_exception_only(type(e), e)
                ).strip() or repr(e)
                loop.call_soon_threadsafe(
                    q.put_nowait, ("error", err_text, None, None)
                )

        def _pipeline_vlm() -> None:
            """VLM streaming via mlx_vlm. Writes any attached image
            bytes to temp files; passes their paths to mlx_vlm's
            `stream_generate` alongside the text prompt. Same queue +
            cancel + error-text plumbing as the text-only path.
            """
            import tempfile
            from pathlib import Path as _Path
            tmp_dir = tempfile.mkdtemp(prefix="mlx_vlm_chat_")
            image_paths: list[str] = []
            try:
                model, processor, config = self._load_vlm_sync(info_model)
                if needs_load:
                    loop.call_soon_threadsafe(q.put_nowait, ("loaded", None, None, None))
                # Write image bytes to temp PNGs.
                for i, img in enumerate(image_bytes_list):
                    p = _Path(tmp_dir) / f"img_{i:02d}.png"
                    p.write_bytes(img)
                    image_paths.append(str(p))

                # Build prompt via mlx_vlm's chat-template helper.
                # It needs num_images so the right number of <image>
                # placeholders are inserted in the prompt.
                from mlx_vlm.prompt_utils import (  # type: ignore
                    apply_chat_template as _vlm_template,
                )
                # Support assistant prefill in VLM template
                has_prefill = (len(cleaned_messages) > 0 and cleaned_messages[-1].get("role") == "assistant")
                if has_prefill:
                    history = cleaned_messages[:-1]
                    prefill_content = cleaned_messages[-1].get("content", "")
                else:
                    history = cleaned_messages
                    prefill_content = ""

                try:
                    prompt = apply_chat_template_safe(
                        _vlm_template,
                        info_model,
                        processor, config, history,
                        num_images=len(image_paths),
                        _stage=turn_stage,
                    )
                except Exception:
                    # Same naive fallback as text-only.
                    prompt = "\n\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in history
                    ) + "\n\nassistant:"

                if has_prefill:
                    prompt = append_assistant_prefill(prompt, prefill_content)

                from mlx_vlm import stream_generate as _vlm_stream  # type: ignore
                # mlx_vlm.stream_generate doesn't take a `sampler` like
                # mlx_lm; sampling kwargs pass through directly.
                kwargs: dict[str, Any] = {
                    "max_tokens": max_tokens,
                    "prefill_step_size": prefill_step_size,
                }
                if temperature > 0:
                    kwargs["temperature"] = temperature
                if top_p > 0:
                    kwargs["top_p"] = top_p
                # Pass image arg only when present — mlx_vlm handles
                # text-only prompts cleanly when image=None.
                image_arg: Any = None
                if image_paths:
                    image_arg = image_paths if len(image_paths) > 1 else image_paths[0]

                last_gen = None
                for gen in _vlm_stream(
                    model, processor, prompt,
                    image=image_arg,
                    **kwargs,
                ):
                    if worker_cancel.is_set():
                        break
                    last_gen = gen
                    pt = getattr(gen, "prompt_tokens", None)
                    ct = getattr(gen, "generation_tokens", None)
                    loop.call_soon_threadsafe(
                        q.put_nowait, ("text", gen.text, pt, ct)
                    )
                pt = getattr(last_gen, "prompt_tokens", None) if last_gen else None
                ct = getattr(last_gen, "generation_tokens", None) if last_gen else None
                loop.call_soon_threadsafe(q.put_nowait, ("done", None, pt, ct))
            except BaseException as e:  # noqa: BLE001
                import traceback as _tb
                err_text = "".join(
                    _tb.format_exception_only(type(e), e)
                ).strip() or repr(e)
                loop.call_soon_threadsafe(
                    q.put_nowait, ("error", err_text, None, None)
                )
            finally:
                # Best-effort temp-dir cleanup.
                try:
                    for p in image_paths:
                        try:
                            _Path(p).unlink()
                        except Exception:
                            pass
                    _Path(tmp_dir).rmdir()
                except Exception:
                    pass

        _pipeline = _pipeline_vlm if is_vlm_model else _pipeline_textonly

        # Submit on the dedicated MLX executor (single thread).
        # Future is awaited implicitly via the queue; we just need to
        # kick it off here.
        async with self._get_load_lock():
            self._get_mlx_executor().submit(_pipeline)
            # If a cold load is needed, drain the "loaded" sentinel
            # before falling through to the read loop so the on_progress
            # callback can flip the status panel back to "prompt eval".
            if needs_load:
                try:
                    first = await asyncio.wait_for(q.get(), timeout=overall_seconds)
                except asyncio.TimeoutError:
                    worker_cancel.set()
                    self._drop_after_crash()
                    return StreamResult(
                        text="", tokens=0,
                        duration_s=time.monotonic() - started,
                        stalled=True, looped=False, crashed=True,
                        stall_at_token=None,
                        error_message=(
                            f"MLX cold-load did not return within "
                            f"{overall_seconds:.0f}s. Model never finished "
                            "loading into Metal."
                        ),
                    )
                kind = first[0]
                if kind == "error":
                    err_payload = first[1] if isinstance(first[1], str) else None
                    self._drop_after_crash()
                    return StreamResult(
                        text="", tokens=0,
                        duration_s=time.monotonic() - started,
                        stalled=True, looped=False, crashed=True,
                        stall_at_token=None,
                        error_message=err_payload,
                    )
                if kind == "loaded":
                    if on_progress is not None:
                        try:
                            on_progress("mlx_load", 1, 1)
                        except Exception:
                            pass
                else:
                    # Got tokens before a "loaded" sentinel — model was
                    # already loaded (race with another stream_chat).
                    # Push the item back for the main consumer.
                    q.put_nowait(first)

        try:
            while True:
                # Check the external cancel event (Ctrl-D). We also poll
                # it via the per-item wait_for timeout below to avoid
                # busy-looping on a slow stream.
                if cancel_event is not None and cancel_event.is_set():
                    worker_cancel.set()
                    stalled = True
                    stall_at = n_tokens
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=min(stall_seconds, 5.0))
                except asyncio.TimeoutError:
                    # Either the model is stalled OR we just want to
                    # poll the cancel event. Stall is measured from
                    # the LAST activity (prefill progress or token
                    # emission), not from stream start — so a slow
                    # prefill on a big prompt isn't mistaken for a
                    # hang.
                    if (
                        time.monotonic() - last_activity_at > stall_seconds
                        and n_tokens == 0
                    ):
                        stalled = True
                        stall_at = 0
                        worker_cancel.set()
                        break
                    continue

                kind, payload, pt, ct = item
                if kind == "error":
                    worker_cancel.set()
                    # Hard MLX/Metal error mid-stream. Surface the real
                    # exception text (worker formatted it at the catch
                    # site) and free GPU state so the next stream
                    # doesn't inherit a stuck Metal allocator.
                    crashed = True
                    err_payload = payload if isinstance(payload, str) else None
                    self._drop_after_crash()
                    return StreamResult(
                        text="".join(parts), tokens=n_tokens,
                        duration_s=time.monotonic() - started,
                        stalled=True, looped=False, crashed=True,
                        stall_at_token=stall_at,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        error_message=err_payload,
                    )
                if kind == "done":
                    if isinstance(pt, int):
                        prompt_tokens = pt
                    if isinstance(ct, int):
                        completion_tokens = ct
                    # KV prompt-cache reuse count rides the done payload.
                    if isinstance(payload, int):
                        cached_prompt_tokens = payload
                    break
                # kind == "text"
                piece = payload or ""
                if isinstance(pt, int):
                    prompt_tokens = pt
                if isinstance(ct, int):
                    completion_tokens = ct
                if not piece:
                    # Silent-stream guard — mirrors ollama_io.stream_chat.
                    # If the model emits only empty `text` events (e.g. all
                    # generation goes to a reasoning channel) with no backend
                    # activity for the floor window, abort instead of waiting
                    # for stall_seconds (default 600s) which is too long.
                    #
                    # Measure from last_activity_at, NOT stream start.
                    # Holochess trace 20260623 (GLM-5.2-MLX): 27–34K-token
                    # feedback prompts prefilled for 5–8 minutes while
                    # prefill progress kept bumping last_activity_at. The old
                    # started-based guard fired the instant prefill finished
                    # (wall clock already >180s) and killed a healthy stream
                    # as "MLX stall". Same fix pattern as the stall watchdog
                    # (test_mlx_stall_activity.py).
                    if (
                        n_tokens == 0
                        and (time.monotonic() - last_activity_at) >= 180.0
                    ):
                        silent = True
                        stall_at = 0
                        worker_cancel.set()
                        break
                    # Empty generation events still count as activity —
                    # a thinking model may stream non-visible tokens for a
                    # long time without ever landing visible content.
                    last_activity_at = time.monotonic()
                    continue
                parts.append(piece)
                n_tokens += 1
                # Token arrived → reset the no-activity stall window.
                # Without this, a model that emits very slowly (a token
                # every 30s) could still trip stall_seconds even though
                # it's making real progress.
                last_activity_at = time.monotonic()
                if on_token is not None:
                    _safe_call(on_token, piece)
                if delib.feed(piece):
                    deliberated = True
                    stall_at = n_tokens
                    worker_cancel.set()
                    break
                if delib.code_emission_started and repeat.feed(piece):
                    if _should_grace_inline_data_bloat(
                        stall_reason=repeat.stall_reason,
                        assembled_text="".join(parts),
                        grace_already_used=loop_grace_used,
                        completion_tokens=n_tokens,
                    ):
                        loop_grace_used = True
                        # BATTLE10: reason names the graced detector (bloat or spam).
                        loop_grace_reason = f"{repeat.stall_reason}_unclosed_output_block"
                        # Reset the detector while the output block is still
                        # open — SEARCH bodies legitimately repeat source lines.
                        repeat = RepetitionDetector()
                        continue
                    looped = True
                    stall_at = n_tokens
                    worker_cancel.set()
                    break
                if diag.feed(piece):
                    diagnose_bloat = True
                    stall_at = n_tokens
                    worker_cancel.set()
                    break
        except asyncio.CancelledError:
            # Caller (agent's task) was cancelled. Stop the worker and
            # re-raise so the agent's run-loop unwinds cleanly.
            worker_cancel.set()
            raise

        return StreamResult(
            text="".join(parts),
            tokens=n_tokens,
            duration_s=time.monotonic() - started,
            stalled=stalled or looped or deliberated or silent or diagnose_bloat,
            stall_at_token=stall_at,
            looped=looped,
            deliberated=deliberated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            crashed=False,
            loop_kind=repeat.stall_reason if looped else None,
            loop_line=repeat.loop_line if looped else None,
            loop_grace_used=loop_grace_used,
            loop_grace_reason=loop_grace_reason,
            silent=silent,
            diagnose_bloat=diagnose_bloat,
            cached_prompt_tokens=cached_prompt_tokens,
        )

    async def is_vlm(self) -> bool:
        """True when (a) the model name classifies as a VLM AND (b)
        `mlx_vlm` is installed so we can actually serve images to it.

        Until 2026-05-15 this returned False unconditionally — the
        MLX stream path only knew about `mlx_lm` (text-only), so
        even if a user loaded a VLM, attached images would be
        silently dropped. Now `stream_chat` routes VLM models
        through `mlx_vlm.stream_generate` with images, so we can
        honestly advertise the capability.
        """
        if classify_model_modality(self.info.model) != "vlm":
            return False
        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            return False
        return True


def _strip_ollama_only_fields(messages: list[dict]) -> list[dict]:
    """Remove fields Ollama uses that mlx_lm.server doesn't understand."""
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        out.append({k: v for k, v in m.items() if k != "images"})
    return out


class MLXServerBackend(Backend):
    """Talks to `mlx_lm.server` via its OpenAI-compatible HTTP API.

    Use when `MLX_SERVER_URL` / `MLX_HOST` is set or
    `LLM_BACKEND=mlx-server`. Multiple agent processes can share one
    server process for continuous batching — one model load in VRAM.
    """

    def __init__(self, info: BackendInfo) -> None:
        self.info = info

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        last: StreamResult | None = None
        # GLM-5.3 6-bit (~289GB) cannot share Metal with Qwen Flash-Next (~198GB).
        # Pinned models are not LRU-evicted; oMLX returns 507 instead
        # (BATTLEZO 20260904_091702). Drop every other resident oMLX model first.
        if requires_omlx_server(self.info.model) or endpoint_is_omlx(
            self.info.endpoint
        ):
            try:
                omlx_unload_loaded_for_inprocess(
                    keep=omlx_api_model_id(self.info.model),
                    endpoint=self.info.endpoint,
                )
            except Exception:
                pass
        for attempt in range(max_retries + 1):
            result = await self._stream_once(
                messages,
                on_token=on_token,
                options=options,
                stall_seconds=stall_seconds,
                overall_seconds=overall_seconds,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            last = result
            if result.crashed:
                return result
            if not result.stalled:
                return result
            if on_stall is not None:
                try:
                    on_stall(result, attempt)
                except Exception:
                    pass
            if attempt < max_retries:
                await asyncio.sleep(2.0)
        assert last is not None
        return last

    async def _stream_once(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None,
        options: dict[str, Any] | None,
        stall_seconds: float,
        overall_seconds: float,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        import httpx

        opts = dict(options or {})
        # oMLX expects basename ids; chat may still hold absolute MLX_Models paths.
        # Fold PNG bytes into OpenAI image_url parts BEFORE stripping the
        # raw `images` key. Hardcoded is_vlm=False + strip meant /list
        # showed [VLM] (name classifier) while /ref warned text-only and
        # screenshots never left the client (bomberman 20260901 oMLX Flash-Next).
        body: dict[str, Any] = {
            "model": mlx_server_api_model_id(self.info.model, self.info.endpoint),
            "messages": omlx_messages_close_think_prefill(
                _strip_ollama_only_fields(
                    _openai_messages_with_images(messages)
                ),
                self.info.model,
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for key in _MLX_OPTION_KEYS:
            if key in opts:
                body[key] = opts[key]
        env_cap = (os.environ.get("MLX_MAX_TOKENS") or "").strip()
        # Same default as in-process MLXBackend (131072). 16384 truncated
        # first-builds (classic-doom 20260512) and the Flash-Next oMLX HTML
        # turn (20260829_165958: finish_reason=length at exactly 16384).
        default_max = int(env_cap) if env_cap.isdigit() and int(env_cap) > 0 else 131072
        body.setdefault("max_tokens", default_max)
        # Keep Qwen3.8 thinking ON (medium). oMLX already defaults thinking on;
        # pass kwargs so effort is medium, not native xhigh. `_stage` (harness
        # turn stage, never sent on the wire) drops fix turns to low.
        think_kw = chat_template_thinking_kwargs(
            self.info.model, stage=opts.get("_stage"),
        )
        if think_kw:
            body["chat_template_kwargs"] = dict(think_kw)
            if "reasoning_effort" in think_kw:
                body["reasoning_effort"] = think_kw["reasoning_effort"]

        started = time.monotonic()
        parts: list[str] = []
        n_tokens = 0
        n_think = 0
        stalled = False
        looped = False
        crashed = False
        error_message: str | None = None
        stall_at: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        cached_prompt_tokens: int | None = None  # oMLX usage.prompt_tokens_details
        max_tokens_hit = False
        repeat = RepetitionDetector()
        prompt_eval_done_at: float | None = None
        _MLX_GENERATION_KICKOFF_SECONDS = 30.0
        # Activity-aware stall (mirrors in-process MLX): SSE prefill progress
        # lines reset the per-read wait_for but must NOT extend the quiet
        # window forever — tune_round1_r2 hung 2h+ on planning with only
        # stream_start in the trace while progress kept the socket alive.
        last_activity_at = started
        _line_poll_s = min(stall_seconds, 30.0)
        # BATTLEZO 20260904_095910: 1800 s at 0 tokens with the model already
        # resident. The quiet-window checks below live in the read-TIMEOUT
        # branch, so SSE keepalive lines (no digits → not progress) kept
        # wait_for from ever timing out and only the cold-load cap ended it.
        # First-token watchdog: runs every loop pass; keepalives do not count
        # as activity; fires only when /v1/models/status says loaded=true
        # (a genuine cold load still gets the full overall_seconds).
        first_token_watchdog_s = min(stall_seconds, _MLX_SERVER_FIRST_TOKEN_WATCHDOG_S)
        # Next monotonic time the loaded-status probe may run (re-armed
        # after a "not loaded" answer so a cold load is re-checked later).
        first_token_watchdog_next = started + first_token_watchdog_s

        try:
            async with httpx.AsyncClient(
                base_url=self.info.endpoint, timeout=None,
            ) as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=body,
                    headers={"accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    ait = response.aiter_lines().__aiter__()
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            stalled = True
                            stall_at = n_tokens
                            break
                        if (
                            n_tokens == 0
                            and n_think == 0
                            and time.monotonic() >= first_token_watchdog_next
                            and time.monotonic() - last_activity_at > first_token_watchdog_s
                        ):
                            first_token_watchdog_next = (
                                time.monotonic() + first_token_watchdog_s
                            )
                            if await self._model_reported_loaded():
                                stalled = True
                                stall_at = 0
                                error_message = (
                                    f"oMLX model loaded but emitted no tokens in "
                                    f"{first_token_watchdog_s:.0f}s (first-token watchdog)"
                                )
                                break
                        if time.monotonic() - started > overall_seconds:
                            stalled = True
                            stall_at = n_tokens
                            error_message = (
                                f"mlx_lm.server exceeded overall timeout "
                                f"({overall_seconds:.0f}s)"
                            )
                            break
                        try:
                            line = await asyncio.wait_for(
                                ait.__anext__(), timeout=_line_poll_s,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            # Must run on read timeout — after prefill the SSE
                            # socket goes quiet until tokens arrive; if generate
                            # wedged, no lines arrive and the old check below
                            # this block never ran (tune_round1_r3 hung 2.5m+).
                            if (
                                prompt_eval_done_at is not None
                                and n_tokens == 0
                                and n_think == 0
                                and time.monotonic() - prompt_eval_done_at
                                > _MLX_GENERATION_KICKOFF_SECONDS
                            ):
                                crashed = True
                                error_message = (
                                    "mlx_lm.server finished prompt eval but emitted "
                                    "no tokens (Metal OOM / generate thread crash?)"
                                )
                                stall_at = 0
                                break
                            if (
                                time.monotonic() - last_activity_at > stall_seconds
                                and n_tokens == 0
                                and n_think == 0
                            ):
                                stalled = True
                                stall_at = 0
                                error_message = (
                                    f"mlx_lm.server quiet for {stall_seconds:.0f}s "
                                    "before first token"
                                )
                                break
                            continue

                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].lstrip()
                        if not payload:
                            continue
                        if payload == "[DONE]":
                            break
                        if payload.startswith(":"):
                            m = _MLX_PROGRESS_RE.search(payload)
                            if m:
                                cur, tot = int(m.group(1)), int(m.group(2))
                                last_activity_at = time.monotonic()
                                if on_progress is not None:
                                    try:
                                        on_progress("prompt_eval", cur, tot)
                                    except Exception:
                                        pass
                                if cur >= tot and prompt_eval_done_at is None:
                                    prompt_eval_done_at = time.monotonic()
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            pt = usage.get("prompt_tokens")
                            ct = usage.get("completion_tokens")
                            if isinstance(pt, int):
                                prompt_tokens = pt
                            if isinstance(ct, int):
                                completion_tokens = ct
                            # KV prefix-cache hit size (OpenAI-compatible
                            # field; oMLX emits it when its tiered cache
                            # served part of the prompt). Absent → None.
                            cpt = _usage_cached_prompt_tokens(usage)
                            if cpt is not None:
                                cached_prompt_tokens = cpt
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        fr = choices[0].get("finish_reason")
                        if fr in ("length", "max_tokens"):
                            max_tokens_hit = True
                        # Hidden CoT is not parser-visible but must reset the
                        # stall clock (20260829_165958) and TUI think counter
                        # (BATTLEZO 20260904_095910 — GLM thinking looked like
                        # a dead 0-token stream).
                        think_piece = _delta_thinking_text(delta)
                        if think_piece:
                            last_activity_at = time.monotonic()
                            n_think += 1
                            _notify_thinking(on_token, think_piece)
                        piece = delta.get("content") or ""
                        if not piece:
                            continue
                        parts.append(piece)
                        n_tokens += 1
                        last_activity_at = time.monotonic()
                        if on_token is not None:
                            try:
                                on_token(piece)
                            except Exception:
                                pass
                        if repeat.feed(piece):
                            looped = True
                            stall_at = n_tokens
                            break
        except (httpx.HTTPError, OSError) as e:
            stalled = True
            crashed = True
            error_message = f"{type(e).__name__}: {e}"
            if stall_at is None:
                stall_at = n_tokens

        if (
            stalled
            and not crashed
            and prompt_eval_done_at is not None
            and n_tokens == 0
            and n_think == 0
        ):
            crashed = True
            if error_message is None:
                error_message = (
                    "mlx_lm.server stalled after prompt eval with no output tokens"
                )

        return StreamResult(
            text="".join(parts),
            tokens=n_tokens,
            duration_s=time.monotonic() - started,
            stalled=stalled or looped or crashed,
            stall_at_token=stall_at,
            looped=looped,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            crashed=crashed,
            error_message=error_message,
            loop_kind=repeat.stall_reason if looped else None,
            loop_line=repeat.loop_line if looped else None,
            max_tokens_hit=max_tokens_hit,
            thinking_tokens=n_think,
            cached_prompt_tokens=cached_prompt_tokens,
        )

    async def _model_reported_loaded(self) -> bool:
        """True when the server lists this model `loaded=true`.

        Used by the first-token watchdog (BATTLEZO 20260904_095910). Plain
        mlx_lm.server has no /v1/models/status → [] → False → the watchdog
        never fires there and the cold-load cap applies as before.
        """
        try:
            ids = await asyncio.to_thread(
                omlx_list_loaded_model_ids, self.info.endpoint,
            )
        except Exception:
            return False
        want = omlx_api_model_id(self.info.model)
        return any(omlx_api_model_id(x) == want for x in ids)

    async def is_vlm(self) -> bool:
        """Same name classifier as /list [VLM]. Do not hardcode False.

        oMLX Flash-Next (Qwen3.8-Flash-Next) is a VLM; /list already
        badged it. Returning False here latched GameAgent._is_vlm and
        made /ref print "text-only" (trace runtime_modes.model_is_vlm=true,
        no vlm_detected).
        """
        return classify_model_modality(self.info.model) == "vlm"


# -----------------------------------------------------------------------------
# Cloud backends — OpenAI and Anthropic.
#
# Both read their API key from the environment at request time
# (OPENAI_API_KEY, ANTHROPIC_API_KEY). The key never enters BackendInfo,
# the trace log, or the message history. SDK calls also pull through
# the standard SDK env vars (OPENAI_BASE_URL, ANTHROPIC_BASE_URL) so
# proxies / Azure / Bedrock relays work without code changes.
#
# StreamResult fields populated:
#   text / tokens (chunk count) / duration_s / stalled / prompt_tokens /
#   completion_tokens.
# The richer Ollama-specific fields (looped, deliberated, crashed) stay
# False — those detectors live in ollama_io.py and don't translate.
# -----------------------------------------------------------------------------


def _openai_messages_with_images(messages: list[dict]) -> list[dict]:
    """Return messages with any attached PNG `images` folded into Chat
    Completions multimodal `content` parts.

    `run_visual_critic` passes screenshots as `{"role":"user",
    "content":prompt,"images":[png_bytes,...]}`. The Chat Completions API
    ignores the top-level `images` key, so without this conversion the
    vision model never sees the screenshot. Messages without images are
    passed through unchanged (the `images` key, if present and empty, is
    stripped so the API never sees an unknown field).
    """
    if not any(isinstance(m, dict) and m.get("images") for m in messages):
        return messages
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        imgs = m.get("images")
        if imgs and isinstance(imgs, (list, tuple)):
            parts: list[dict[str, Any]] = []
            text = m.get("content") or ""
            if text:
                parts.append({"type": "text", "text": text})
            for b in imgs:
                if isinstance(b, (bytes, bytearray)):
                    b64 = base64.b64encode(bytes(b)).decode("ascii")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
            out.append({"role": m.get("role", "user"), "content": parts})
        else:
            # Drop a possibly-empty `images` key so the API sees a clean msg.
            out.append({k: v for k, v in m.items() if k != "images"})
    return out


class OpenAIBackend(Backend):
    """OpenAI Chat Completions backend. Streaming, async."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. Run: "
                ".venv/bin/pip install 'openai>=1.50'"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell "
                "(or 1Password / Keychain) before starting chat.py."
            )
        # The SDK reads OPENAI_API_KEY + OPENAI_BASE_URL automatically.
        self._client = AsyncOpenAI()

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        opts = dict(options or {})
        # Convert any attached PNG bytes (run_visual_critic passes
        # `images=[png,...]` on a message) into Chat Completions multimodal
        # content parts. The raw `images` key is not understood by the API
        # and was silently ignored, leaving the vision model blind (same
        # class of bug as the Anthropic path — trace 20260613_213711).
        norm_messages = _openai_messages_with_images(messages)
        params: dict[str, Any] = {
            "model": self.info.model,
            "messages": norm_messages,
            "stream": True,
            # include_usage gives us prompt/completion token counts in
            # the final chunk so the right-hand status panel can show
            # token spend (which is also $ spend for cloud backends).
            "stream_options": {"include_usage": True},
        }
        if "max_tokens" in opts:
            # GPT-5 and other reasoning-trained models prefer
            # max_completion_tokens. The legacy max_tokens still works
            # for non-reasoning models — try the new name first, fall
            # back on TypeError.
            params["max_completion_tokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            params["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            params["top_p"] = float(opts["top_p"])
        if "seed" in opts:
            params["seed"] = int(opts["seed"])

        parts: list[str] = []
        tokens = 0
        think_tokens = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        t0 = time.monotonic()
        cancelled = False

        max_tokens_hit = False

        async def _run(call_params: dict[str, Any]) -> None:
            nonlocal tokens, think_tokens, prompt_tokens, completion_tokens, cancelled
            nonlocal max_tokens_hit
            stream = await self._client.chat.completions.create(**call_params)
            async for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    think_piece = _delta_thinking_text(delta)
                    if think_piece:
                        think_tokens += 1
                        _notify_thinking(on_token, think_piece)
                    piece = getattr(delta, "content", None)
                    if piece:
                        parts.append(piece)
                        tokens += 1
                        if on_token is not None:
                            on_token(piece)
                    # OpenAI Chat Completions: finish_reason on the
                    # final chunk's choice. "length" means we hit
                    # max_completion_tokens; the model would have kept
                    # going. Routed by the agent to a "your reply was
                    # capped, emit a smaller change" coach.
                    fr = getattr(chunk.choices[0], "finish_reason", None)
                    if fr == "length":
                        max_tokens_hit = True
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)

        try:
            await _run(params)
        except TypeError as e:
            # Older model API surface — retry without max_completion_tokens
            # by swapping it back to the legacy max_tokens key.
            if "max_completion_tokens" in params and "max_completion_tokens" in str(e):
                params["max_tokens"] = params.pop("max_completion_tokens")
                try:
                    await _run(params)
                except Exception as e2:
                    err_payload = f"{type(e2).__name__}: {e2}"
                    print(
                        f"openai stream_chat error: {err_payload}",
                        file=sys.stderr,
                    )
                    # Phase 5a: real crash, not a stall. Lets the agent
                    # route to fallback / retry rather than synthesizing
                    # a misleading "stalled at <stall_seconds>s" message.
                    return StreamResult(
                        text="".join(parts),
                        tokens=tokens,
                        duration_s=time.monotonic() - t0,
                        stalled=False,
                        crashed=True,
                        error_message=err_payload,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        thinking_tokens=think_tokens,
                    )
            else:
                raise
        except Exception as e:
            # Connection / auth / quota / timeout errors come through
            # here. Surface the real exception class + message to
            # stderr so callers can diagnose without enabling debug
            # logging — 429 insufficient_quota is a billing fix, 401
            # is a key fix.
            err_payload = f"{type(e).__name__}: {e}"
            print(
                f"openai stream_chat error: {err_payload}",
                file=sys.stderr,
            )
            # Phase 5a: real crash, not a stall (see Anthropic note).
            return StreamResult(
                text="".join(parts),
                tokens=tokens,
                duration_s=time.monotonic() - t0,
                stalled=False,
                crashed=True,
                error_message=err_payload,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=think_tokens,
            )

        return StreamResult(
            text="".join(parts),
            tokens=tokens,
            duration_s=time.monotonic() - t0,
            stalled=cancelled,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_tokens_hit=max_tokens_hit,
            thinking_tokens=think_tokens,
        )

    async def is_vlm(self) -> bool:
        # GPT-4o / GPT-4.1 / GPT-5 all support vision input. The agent
        # gates screenshot inclusion on this flag.
        return True

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass


def _anthropic_prepare_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict[str, str]]]:
    """Split system prompts from history and sanitize for Anthropic API rules.

    Belt-and-suspenders safety net for tag-opener assistant prefill:
    newer Claude models (Opus 4.7+) hard-reject ANY trailing assistant
    message — they require the final message to be from the user. The
    agent-level fix in `agent._stream()` folds tag prefills into the
    last user message for backend=anthropic. If a caller forgets that,
    this layer detects a SHORT tag-opener assistant turn and folds it
    into the preceding user message here instead of letting the API
    return a 400 'does not support assistant message prefill'.
    """
    system_parts: list[str] = []
    msgs: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        # Convert any attached PNG bytes (run_visual_critic passes
        # `images=[png,...]` on the message) into Anthropic vision content
        # blocks. Without this the `images` key was silently dropped and the
        # critic model received text only — it then truthfully replied "no
        # screenshot was provided" and the blind verdict was parsed as real
        # failures (trace 20260613_213711, Opus 4.8). Image blocks first,
        # then the text prompt, matching vision_judge._png_block's shape.
        imgs = m.get("images")
        if imgs and isinstance(imgs, (list, tuple)):
            blocks: list[dict[str, Any]] = []
            for b in imgs:
                if isinstance(b, (bytes, bytearray)):
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(bytes(b)).decode("ascii"),
                        },
                    })
            if blocks:
                new_content: list[dict[str, Any]] = list(blocks)
                if content:
                    new_content.append({"type": "text", "text": content})
                msgs.append({"role": role, "content": new_content})
                continue
        msgs.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts).strip() or None

    # Safety-net fold: trailing assistant tag opener -> user format hint.
    # Trigger ONLY when the trailing assistant message looks like a bare
    # tag opener (short + starts with `<`). Longer assistant content is
    # a real model reply we must preserve verbatim.
    if (
        len(msgs) >= 2
        and msgs[-1].get("role") == "assistant"
        and msgs[-2].get("role") == "user"
        # Only fold when the user content is plain text — a user message
        # carrying vision content blocks (list) is never a tag-opener case.
        and isinstance(msgs[-2].get("content"), str)
        and isinstance(msgs[-1].get("content"), str)
    ):
        tail = str(msgs[-1].get("content") or "").rstrip()
        looks_like_opener = (
            tail.startswith("<")
            and len(tail) <= 200
            # First non-space line should be the bare opener.
            and tail.split("\n", 1)[0].strip().endswith(">")
        )
        if looks_like_opener:
            first_line = tail.split("\n", 1)[0].strip()
            hint = (
                "\n\nFORMAT: begin your reply with exactly `"
                + first_line
                + "` (no prose before it; no extra whitespace)."
            )
            user_content = str(msgs[-2].get("content") or "")
            msgs[-2] = {
                "role": "user",
                "content": user_content + hint,
            }
            msgs = msgs[:-1]
            return system_text, msgs

    # Fix-mode assistant prefill ends with "\n" (e.g. "<diagnose>\n"); Anthropic
    # 400s when the final assistant turn has trailing whitespace.
    if msgs and msgs[-1].get("role") == "assistant":
        msgs[-1] = {
            "role": "assistant",
            "content": str(msgs[-1].get("content") or "").rstrip(),
        }
    return system_text, msgs


class AnthropicBackend(Backend):
    """Anthropic Messages backend. Streaming, async."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run: "
                ".venv/bin/pip install 'anthropic>=0.40'"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell "
                "(or 1Password / Keychain) before starting chat.py."
            )
        self._client = AsyncAnthropic()

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: float | str | None = None,
        stall_seconds: float = 600.0,
        overall_seconds: float = 1800.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StreamResult:
        system_text, msgs = _anthropic_prepare_messages(messages)

        opts = dict(options or {})
        # Anthropic max_tokens is REQUIRED.
        #
        # 8192 was the original default — and the DK trace
        # 20260513_135011 burned 3 consecutive iters (iter 1/2/3) on
        # truncated <html_file> rewrites where Claude generated exactly
        # 8192 completion tokens and got cut off mid-document. Iter 1
        # was a 17,654-byte stream missing every closing tag.
        # Sonnet 4.6 supports 64K output, Opus 4.8 supports 32K. 32768
        # is the safe-everywhere ceiling for "write a full HTML game
        # file in one go" and covers ~120 KB of output. Override via
        # options["max_tokens"] or env ANTHROPIC_MAX_TOKENS for runs
        # that need to push higher (Sonnet) or lower (rate-limit
        # mitigation).
        env_cap = os.environ.get("ANTHROPIC_MAX_TOKENS", "").strip()
        try:
            env_max = int(env_cap) if env_cap else 0
        except ValueError:
            env_max = 0
        default_max = env_max if env_max > 0 else 32768
        max_tok = int(opts.get("max_tokens") or default_max)
        kwargs: dict[str, Any] = {
            "model": self.info.model,
            "messages": msgs,
            "max_tokens": max_tok,
        }
        if system_text:
            kwargs["system"] = system_text
        if "temperature" in opts:
            kwargs["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            kwargs["top_p"] = float(opts["top_p"])

        parts: list[str] = []
        tokens = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        t0 = time.monotonic()
        cancelled = False

        max_tokens_hit = False

        async def _run_anthropic(call_kwargs: dict[str, Any]) -> None:
            nonlocal tokens, prompt_tokens, completion_tokens, cancelled
            nonlocal max_tokens_hit
            async with self._client.messages.stream(**call_kwargs) as stream:
                async for piece in stream.text_stream:
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                    if piece:
                        parts.append(piece)
                        tokens += 1
                        if on_token is not None:
                            on_token(piece)
                if not cancelled:
                    final = await stream.get_final_message()
                    usage = getattr(final, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "input_tokens", None)
                        completion_tokens = getattr(usage, "output_tokens", None)
                    # Capture the cut-by-API signal. The Anthropic SDK
                    # exposes `stop_reason` on the final message; the
                    # value "max_tokens" means we hit the cap and the
                    # model would have kept going. Distinct from
                    # "end_turn" (natural finish) or "stop_sequence".
                    stop = getattr(final, "stop_reason", None)
                    if stop == "max_tokens":
                        max_tokens_hit = True

        try:
            await _run_anthropic(kwargs)
        except Exception as e:
            # Some Claude models (Opus 4.x reasoning class) reject
            # `temperature` with a 400 — auto-retry once with the
            # parameter dropped before bubbling the error up.
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                try:
                    parts.clear()
                    tokens = 0
                    await _run_anthropic(kwargs)
                except Exception as e2:
                    err_payload = f"{type(e2).__name__}: {e2}"
                    print(
                        f"anthropic stream_chat error: {err_payload}",
                        file=sys.stderr,
                    )
                    return StreamResult(
                        text="".join(parts),
                        tokens=tokens,
                        duration_s=time.monotonic() - t0,
                        stalled=False,
                        crashed=True,
                        error_message=err_payload,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            else:
                # Phase 5a: mark this as a real CRASH, not a stall, so
                # the agent can route to the fallback / retry path.
                # Trace 2 (chess 20260522_104235) had Anthropic raise in
                # 0.48s and the agent reported "stalling at 600s" because
                # this branch returned `stalled=True` with no error_message.
                err_payload = f"{type(e).__name__}: {e}"
                print(
                    f"anthropic stream_chat error: {err_payload}",
                    file=sys.stderr,
                )
                return StreamResult(
                    text="".join(parts),
                    tokens=tokens,
                    duration_s=time.monotonic() - t0,
                    stalled=False,
                    crashed=True,
                    error_message=err_payload,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        return StreamResult(
            text="".join(parts),
            tokens=tokens,
            duration_s=time.monotonic() - t0,
            stalled=cancelled,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_tokens_hit=max_tokens_hit,
        )

    async def is_vlm(self) -> bool:
        # All Claude 4.x models accept images.
        return True

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Detection.
# -----------------------------------------------------------------------------


def detect_backend(prefer: str | None = None) -> BackendInfo:
    """Resolve which LLM daemon to use.

    Resolution order: non-empty ``prefer`` argument, else ``LLM_BACKEND`` env,
    else platform default (``mlx`` on macOS, ``auto`` on other platforms).

    Effective preference string:
      "auto"        — probe both; MLX wins ties; Ollama fallback if neither loaded.
      "ollama"      — force Ollama; raise if unreachable.
      "mlx"         — force MLX; raise if unreachable.

    Set ``LLM_BACKEND=auto`` on a Mac (or pass ``prefer=\"auto\"``) to probe
    both daemons instead of defaulting to MLX.
    """
    default_pref = "mlx" if sys.platform == "darwin" else "auto"
    prefer = (prefer or os.environ.get("LLM_BACKEND") or default_pref).strip().lower()

    if prefer == "ollama":
        info = _try_ollama_with_loaded() or _ollama_full_fallback()
        if info is None:
            raise RuntimeError(
                "LLM_BACKEND=ollama but no model is loaded and /api/tags is unreachable. "
                "Start ollama and `ollama run <model>` first."
            )
        return info

    if prefer == "mlx":
        info = _try_mlx()
        if info is None:
            if _mlx_server_mode_requested():
                raise RuntimeError(
                    "MLX server mode selected but mlx_lm.server is not reachable.\n"
                    f"Start it with: mlx_lm.server --model <path> --port 8080\n"
                    f"Or set MLX_SERVER_URL (currently {_mlx_server_endpoint_url()!r})."
                )
            raise RuntimeError(
                "MLX backend selected but no MLX model could be resolved.\n"
                "Set MLX_MODEL=<path-or-id> to point at a downloaded MLX "
                "model, or place a model under ~/MLX_Models/ so it's "
                "auto-discovered."
            )
        return info

    if prefer == "mlx-server":
        info = _try_mlx_server()
        if info is None:
            raise RuntimeError(
                "LLM_BACKEND=mlx-server but mlx_lm.server is not reachable.\n"
                f"Start it with: mlx_lm.server --model <path> --port 8080\n"
                f"Endpoint probed: {_mlx_server_endpoint_url()!r}"
            )
        return info

    if prefer in ("openai", "oai"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "LLM_BACKEND=openai but OPENAI_API_KEY is not set. "
                "Export it in your shell first."
            )
        return BackendInfo(
            name="openai",
            model=_openai_default_model(),
            source="LLM_BACKEND=openai (OPENAI_API_KEY set)",
            endpoint="https://api.openai.com",
        )

    if prefer in ("anthropic", "claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set. "
                "Export it in your shell first."
            )
        return BackendInfo(
            name="anthropic",
            model=_anthropic_default_model(),
            source="LLM_BACKEND=anthropic (ANTHROPIC_API_KEY set)",
            endpoint="https://api.anthropic.com",
        )

    # Auto: probe both, prefer whichever has a loaded model. MLX wins ties.
    mlx_info = _try_mlx()
    ollama_info = _try_ollama_with_loaded()
    if mlx_info is not None:
        return mlx_info
    if ollama_info is not None:
        return ollama_info
    # Nothing loaded — fall back to ollama with /api/tags.
    fallback = _ollama_full_fallback()
    if fallback is not None:
        return fallback
    raise RuntimeError(
        "No LLM backend reachable. Either:\n"
        "  • Start Ollama: `ollama run <model>`  (port 11434), or\n"
        "  • Set MLX_MODEL=<path> to a local MLX model, or place one "
        "under ~/MLX_Models/."
    )


def make_backend(info: BackendInfo) -> Backend:
    if info.name == "ollama":
        return OllamaBackend(info)
    if info.name == "mlx":
        # BATTLEZ2 20260904_094928: detect_backend handed GLM-5.3 with
        # endpoint=in-process → mlx_lm "glm5_next not supported", 0 tokens.
        if requires_omlx_server(info.model):
            ep = (info.endpoint or "").strip()
            if not ep or ep == _MLX_IN_PROCESS_ENDPOINT:
                info = replace(
                    info,
                    model=omlx_api_model_id(info.model),
                    endpoint=omlx_default_endpoint(),
                )
            else:
                info = replace(info, model=omlx_api_model_id(info.model))
            return MLXServerBackend(info)
        ep = (info.endpoint or "").strip()
        if ep and ep != _MLX_IN_PROCESS_ENDPOINT:
            return MLXServerBackend(info)
        return MLXBackend(info)
    if info.name == "openai":
        return OpenAIBackend(info)
    if info.name == "anthropic":
        return AnthropicBackend(info)
    raise ValueError(f"unknown backend: {info.name!r}")


# Endpoint sentinels for cloud backends. The cloud SDKs ignore these —
# the real base URL comes from OPENAI_BASE_URL / ANTHROPIC_BASE_URL env
# vars if set, otherwise the SDK default. The string just keeps
# BackendInfo's endpoint field non-empty so existing UI strings work.
_OPENAI_ENDPOINT = "https://api.openai.com"
_ANTHROPIC_ENDPOINT = "https://api.anthropic.com"


def openai_endpoint_url() -> str:
    return _OPENAI_ENDPOINT


def anthropic_endpoint_url() -> str:
    return _ANTHROPIC_ENDPOINT


def list_openai_inventory() -> tuple[list[str], str | None]:
    """(available_models, default_or_None) — for /list display.

    Empty when OPENAI_API_KEY is not set, so the TUI hides the entries
    instead of dangling unusable picks. Prefers the newest flagship id
    from GET /v1/models (metadata only, cached for the process). Falls
    back to `_OPENAI_DEFAULT_MODEL` when the list is unreachable.
    Pin with OPENAI_MODEL if you do not want the live pick.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return [], None
    default = _openai_default_model()
    return [default], default


def list_anthropic_inventory() -> tuple[list[str], str | None]:
    """Mirror of list_openai_inventory for Anthropic / Claude.

    /list stays two rows: newest Fable + newest Opus (coding default).
    Short aliases (`fable`, `opus`, `sonnet`) resolve via
    resolve_cloud_alias(). Pin with ANTHROPIC_MODEL for the default.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return [], None
    ids = _fetch_anthropic_model_ids() or []
    fable = _pick_newest_anthropic_family(ids, "fable") or _ANTHROPIC_MODELS[0]
    opus = _anthropic_default_model()
    models = [fable]
    if opus not in models:
        models.append(opus)
    return models, opus


# Short TUI aliases → Claude family or OpenAI flagship/mini. Full ids
# (claude-fable-5-1, gpt-5.6-sol) pass through unchanged.
_CLOUD_ALIAS_FAMILY: dict[str, str] = {
    "claude": "sonnet",  # /check reviewer — keep the cheap family
    "sonnet": "sonnet",
    "opus": "opus",
    "fable": "fable",
    "haiku": "haiku",
    "gpt": "openai_flagship",
    "openai": "openai_flagship",
    "gpt5": "openai_flagship",
    "gpt-5-mini": "openai_mini",
    "gpt5-mini": "openai_mini",
}

# Process cache for GET /v1/models. None = fetch failed (do not retry
# every /list). Missing key = not attempted yet.
_CLOUD_MODEL_IDS_CACHE: dict[str, list[str] | None] = {}

_OPENAI_SKIP_SUBSTR: tuple[str, ...] = (
    "embedding", "whisper", "tts", "dall-e", "dalle", "realtime",
    "audio", "transcribe", "search", "moderation", "davinci",
    "babbage", "ada", "sora", "gpt-image", "computer-use",
)


def _ids_from_models_payload(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for row in data.get("data") or []:
        if isinstance(row, dict):
            mid = (row.get("id") or "").strip()
            if mid:
                out.append(mid)
        elif isinstance(row, str) and row.strip():
            out.append(row.strip())
    return out


def _is_dated_model_snapshot(mid: str) -> bool:
    low = mid.lower()
    return bool(
        re.search(r"-\d{8}$", low) or re.search(r"-\d{4}-\d{2}-\d{2}$", low)
    )


def _pick_newest_anthropic_family(ids: list[str], family: str) -> str | None:
    """Newest claude-{family}-* id. Prefer alias ids over dated snapshots."""
    family = (family or "").strip().lower()
    if not family or not ids:
        return None
    prefix = f"claude-{family}-"
    matched = [
        mid for mid in ids
        if mid.lower().startswith(prefix) or mid.lower() == f"claude-{family}"
    ]
    if not matched:
        return None

    def _key(mid: str) -> tuple:
        rest = mid.lower().split(prefix, 1)[-1] if prefix in mid.lower() else ""
        nums = tuple(int(x) for x in re.findall(r"\d+", rest))
        return (0 if _is_dated_model_snapshot(mid) else 1, nums)

    return max(matched, key=_key)


def _gpt_version_tuple(mid: str) -> tuple[int, ...]:
    m = re.match(r"gpt-(\d+(?:\.\d+)*)", mid.lower())
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


def _openai_flagship_tier_rank(mid: str) -> int:
    """Prefer gpt-5.6 (alias) over gpt-5.6-sol; skip mini/terra/luna."""
    low = mid.lower()
    if _is_dated_model_snapshot(low):
        return -10
    if any(tok in low for tok in ("-mini", "-nano", "-luna", "-terra", "-codex")):
        return -5
    rest = re.sub(r"^gpt-\d+(?:\.\d+)*", "", low)
    if rest == "":
        return 3
    if rest == "-sol":
        return 2
    return 0


def _pick_openai_flagship(ids: list[str]) -> str | None:
    """Newest GPT chat flagship. Skips embeddings/audio/mini/terra/luna."""
    cands: list[str] = []
    for mid in ids:
        low = mid.lower()
        if not low.startswith("gpt-"):
            continue
        if any(s in low for s in _OPENAI_SKIP_SUBSTR):
            continue
        cands.append(mid)
    if not cands:
        return None
    return max(cands, key=lambda m: (_gpt_version_tuple(m), _openai_flagship_tier_rank(m)))


def _pick_openai_mini(ids: list[str]) -> str | None:
    cands = [
        mid for mid in ids
        if mid.lower().startswith("gpt-")
        and "-mini" in mid.lower()
        and "codex-mini" not in mid.lower()
        and not _is_dated_model_snapshot(mid)
    ]
    if not cands:
        return None
    return max(cands, key=_gpt_version_tuple)


def _http_get_json_auth(
    url: str, headers: dict[str, str], timeout: float = 2.5,
) -> Any:
    """Authenticated GET. Fail-soft like `_http_get_json` — never raises."""
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _fetch_openai_model_ids() -> list[str] | None:
    if "openai" in _CLOUD_MODEL_IDS_CACHE:
        return _CLOUD_MODEL_IDS_CACHE["openai"]
    # Pytest stays offline; tests mock this helper when they need live picks.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    data = _http_get_json_auth(
        "https://api.openai.com/v1/models",
        {"Authorization": f"Bearer {key}"},
    )
    ids = _ids_from_models_payload(data)
    _CLOUD_MODEL_IDS_CACHE["openai"] = ids or None
    return _CLOUD_MODEL_IDS_CACHE["openai"]


def _fetch_anthropic_model_ids() -> list[str] | None:
    if "anthropic" in _CLOUD_MODEL_IDS_CACHE:
        return _CLOUD_MODEL_IDS_CACHE["anthropic"]
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    data = _http_get_json_auth(
        "https://api.anthropic.com/v1/models?limit=100",
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    ids = _ids_from_models_payload(data)
    _CLOUD_MODEL_IDS_CACHE["anthropic"] = ids or None
    return _CLOUD_MODEL_IDS_CACHE["anthropic"]


def _openai_default_model() -> str:
    pinned = (os.environ.get("OPENAI_MODEL") or "").strip()
    if pinned:
        return pinned
    return _pick_openai_flagship(_fetch_openai_model_ids() or []) or _OPENAI_MODELS[0]


def _anthropic_default_model() -> str:
    pinned = (os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if pinned:
        return pinned
    return (
        _pick_newest_anthropic_family(_fetch_anthropic_model_ids() or [], "opus")
        or _ANTHROPIC_DEFAULT_MODEL
    )


def resolve_cloud_alias(name: str) -> str:
    """Map `fable` / `gpt` / `opus` / … to a concrete API id.

    Live Models API list (cached) when the matching key is set; otherwise
    the fallback constants. Unknown names (full ids, local MLX paths)
    are returned unchanged.
    """
    raw = (name or "").strip()
    if not raw:
        return raw
    family = _CLOUD_ALIAS_FAMILY.get(raw.lower())
    if family is None:
        return raw
    if family == "openai_flagship":
        return _openai_default_model()
    if family == "openai_mini":
        return (
            _pick_openai_mini(_fetch_openai_model_ids() or [])
            or _OPENAI_MINI_FALLBACK
        )
    fallback = {
        "opus": _ANTHROPIC_DEFAULT_MODEL,
        "fable": _ANTHROPIC_FABLE_FALLBACK,
        "sonnet": _ANTHROPIC_SONNET_FALLBACK,
        "haiku": _ANTHROPIC_HAIKU_FALLBACK,
    }.get(family, _ANTHROPIC_DEFAULT_MODEL)
    return (
        _pick_newest_anthropic_family(_fetch_anthropic_model_ids() or [], family)
        or fallback
    )


# -----------------------------------------------------------------------------
# Ollama detection helpers (extracted from chat.py — same probe chain).
# -----------------------------------------------------------------------------


def _normalize_ollama_host(raw: str, *, default_port: int = 11434) -> str:
    s = (raw or "").strip().rstrip("/")
    if not s:
        return f"http://127.0.0.1:{default_port}"
    if not s.startswith("http"):
        s = "http://" + s
    return s


def _ollama_endpoint() -> str:
    return _normalize_ollama_host(os.environ.get("OLLAMA_HOST") or "")


def _ollama_endpoint_for_slot(slot: int) -> str:
    """HTTP base for Ollama slot 1/2/3 (3-model runs on separate daemons).

    Slot 1: ``OLLAMA_HOST`` (default ``http://127.0.0.1:11434``).
    Slot 2: ``OLLAMA_HOST2``, else slot 1.
    Slot 3: ``OLLAMA_HOST3``, else slot 1.
    """
    if slot <= 1:
        return _ollama_endpoint()
    key = "OLLAMA_HOST2" if slot == 2 else "OLLAMA_HOST3"
    raw = (os.environ.get(key) or "").strip()
    if raw:
        return _normalize_ollama_host(raw)
    return _ollama_endpoint()


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _manual_ollama_slot_hosts_set() -> bool:
    return bool(
        (os.environ.get("OLLAMA_HOST2") or "").strip()
        or (os.environ.get("OLLAMA_HOST3") or "").strip()
    )


def _ollama_cli_candidates() -> list[str]:
    """Resolve `ollama`; Cursor-launched Python often has a thin PATH."""
    out: list[str] = []
    seen: set[str] = set()
    for c in (
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/snap/bin/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
    ):
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            out.append(c)
    return out


def _resolve_ollama_models_dir() -> str:
    """Choose the model store for auto-started slot daemons."""
    raw = (os.environ.get("OLLAMA_MODELS") or "").strip()
    candidates = [raw] if raw else []
    candidates.extend([
        "/usr/share/ollama/.ollama/models",
        os.path.expanduser("~/.ollama/models"),
    ])
    for path in candidates:
        if not path:
            continue
        manifests = os.path.join(path, "manifests")
        if os.path.isdir(manifests):
            return path
    return raw or os.path.expanduser("~/.ollama/models")


def _is_four_gpu_linux_nvidia_workstation() -> tuple[bool, str]:
    """Strict autopin gate: Linux + exactly 4 large NVIDIA GPUs."""
    if sys.platform == "darwin":
        return False, "macOS/MLX host"
    if not sys.platform.startswith("linux"):
        return False, f"non-Linux platform {sys.platform!r}"
    try:
        import gpu_status as gs
    except Exception:
        return False, "gpu_status unavailable"
    snap = gs.snapshot_gpus(force=True)
    if snap is None or not snap.gpus:
        return False, "nvidia-smi unavailable"
    gpus = sorted(snap.gpus, key=lambda g: g.index)
    if len(gpus) != 4:
        return False, f"{len(gpus)} visible GPUs"
    if not all("nvidia" in (g.name or "").lower() for g in gpus):
        return False, "non-NVIDIA or mixed GPU inventory"
    if not all((g.memory_total_mib or 0) >= 40000 for g in gpus):
        return False, "not all GPUs are 48 GB-class"
    if [g.index for g in gpus] != [0, 1, 2, 3]:
        return False, f"unexpected GPU indices {[g.index for g in gpus]}"
    return True, "4x NVIDIA 48 GB-class workstation"


def _port_owner_pid(port: int) -> int | None:
    """PID listening on TCP port, if `ss` can see it."""
    try:
        r = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    marker = f":{port}"
    for line in (r.stdout or "").splitlines():
        if marker not in line:
            continue
        m = re.search(r"pid=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return data.replace(b"\0", b" ").decode(errors="replace").strip()


def _proc_environ(pid: int) -> dict[str, str]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _pid_is_same_user(pid: int) -> bool:
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def _pid_is_ollama_serve(pid: int) -> bool:
    cmd = _proc_cmdline(pid).lower()
    return "ollama" in cmd and "serve" in cmd


def _endpoint_ready(base: str) -> bool:
    return isinstance(_http_get_json(base.rstrip("/") + "/api/tags", timeout=1.5), dict)


def _wait_endpoint_ready(base: str, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _endpoint_ready(base):
            return True
        time.sleep(0.25)
    return _endpoint_ready(base)


def _terminate_same_user_ollama_serve(pid: int, *, port: int) -> tuple[bool, str]:
    if not _pid_is_same_user(pid) or not _pid_is_ollama_serve(pid):
        return False, (
            f"port {port} is owned by pid {pid}, not a same-user ollama serve; "
            "left untouched"
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return False, f"could not stop ollama serve pid {pid}: {e!r}"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _port_owner_pid(port) != pid:
            return True, f"stopped stale ollama serve pid {pid} on {port}"
        time.sleep(0.2)
    return False, f"ollama serve pid {pid} on {port} did not stop"


def _slot_daemon_matches(pid: int, *, gpu: int, models_dir: str) -> bool:
    env = _proc_environ(pid)
    visible = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    store = (env.get("OLLAMA_MODELS") or "").strip()
    return visible == str(gpu) and (not store or os.path.abspath(store) == os.path.abspath(models_dir))


def _start_ollama_slot_daemon(
    *,
    exe: str,
    port: int,
    gpu: int,
    models_dir: str,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = models_dir
    env["OLLAMA_SCHED_SPREAD"] = "2"
    try:
        subprocess.Popen(
            [exe, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        return False, f"failed to start ollama on {port}/GPU{gpu}: {e!r}"
    base = f"http://127.0.0.1:{port}"
    if not _wait_endpoint_ready(base):
        return False, f"ollama on {port}/GPU{gpu} did not become reachable"
    return True, f"started ollama {port} on GPU{gpu}"


def _slot_endpoint_env(slot: int, port: int) -> tuple[str, str]:
    key = "OLLAMA_HOST" if slot == 1 else f"OLLAMA_HOST{slot}"
    return key, f"http://127.0.0.1:{port}"


def ensure_ollama_slot_daemons_for_chat(
    *,
    enabled: bool,
    prefer: str | None = None,
) -> OllamaAutopinResult:
    """Auto-start one Ollama daemon per LLM slot on the 4-GPU Linux box.

    This is intentionally a no-op everywhere except the strict workstation
    shape. It never uses sudo/systemd and never changes context size.
    """
    if not enabled:
        return OllamaAutopinResult("off", "not a 3-slot Ollama run")
    if _env_truthy("AGENT_NO_AUTO_OLLAMA_PIN"):
        return OllamaAutopinResult("off", "AGENT_NO_AUTO_OLLAMA_PIN=1")
    if _manual_ollama_slot_hosts_set():
        return OllamaAutopinResult("manual", "manual OLLAMA_HOST2/HOST3 in use")

    pref = (prefer or os.environ.get("LLM_BACKEND") or "").strip().lower()
    if pref in ("mlx", "openai", "oai", "anthropic", "claude"):
        return OllamaAutopinResult("off", f"backend {pref!r} is not Ollama")
    if pref in ("", "auto"):
        try:
            if _try_mlx() is not None:
                return OllamaAutopinResult("off", "auto backend selected MLX")
        except Exception:
            pass

    ok, reason = _is_four_gpu_linux_nvidia_workstation()
    if not ok:
        return OllamaAutopinResult("off", reason)

    exe_candidates = _ollama_cli_candidates()
    if not exe_candidates:
        return OllamaAutopinResult("fallback", "no ollama executable found")
    exe = exe_candidates[0]
    models_dir = _resolve_ollama_models_dir()
    slots = ((1, 11434, 1), (2, 11435, 2), (3, 11436, 3))
    messages: list[str] = []

    for slot, port, gpu in slots:
        base = f"http://127.0.0.1:{port}"
        owner = _port_owner_pid(port)
        if owner is not None:
            if _slot_daemon_matches(owner, gpu=gpu, models_dir=models_dir) and _endpoint_ready(base):
                messages.append(f"{port}->GPU{gpu} already pinned")
            else:
                loaded = _ollama_running_models(base)
                if loaded:
                    unload_all_ollama_models(base)
                stopped, msg = _terminate_same_user_ollama_serve(owner, port=port)
                messages.append(msg)
                if not stopped:
                    return OllamaAutopinResult("fallback", "; ".join(messages))
                started, msg = _start_ollama_slot_daemon(
                    exe=exe, port=port, gpu=gpu, models_dir=models_dir,
                )
                messages.append(msg)
                if not started:
                    return OllamaAutopinResult("fallback", "; ".join(messages))
        else:
            started, msg = _start_ollama_slot_daemon(
                exe=exe, port=port, gpu=gpu, models_dir=models_dir,
            )
            messages.append(msg)
            if not started:
                return OllamaAutopinResult("fallback", "; ".join(messages))

        key, value = _slot_endpoint_env(slot, port)
        os.environ[key] = value

    endpoints = {slot: value for slot, port, _ in slots for _, value in [_slot_endpoint_env(slot, port)]}
    return OllamaAutopinResult(
        "auto-pinned",
        " · ".join(messages),
        endpoints=endpoints,
    )


def _ollama_endpoints() -> list[str]:
    """Loopback bases to probe — IDE-launched Python often differs from shell.

    First entry is the env-derived endpoint (used when constructing
    OllamaBackend); the rest are loopback fallbacks so detection still
    succeeds when OLLAMA_HOST is unset and the daemon binds only one
    of v4 / v6.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in (
        os.environ.get("OLLAMA_HOST") or "",
        "127.0.0.1:11434",
        "localhost:11434",
        "[::1]:11434",
    ):
        s = raw.strip().rstrip("/")
        if not s:
            continue
        if not s.startswith("http"):
            s = "http://" + s
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _http_get_json(url: str, timeout: float = 5.0) -> Any:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _http_post_json(url: str, payload: dict, timeout: float = 5.0) -> Any:
    """POST a JSON body and return the parsed JSON response, or None on
    any error. Mirrors `_http_get_json` semantics — best-effort, never
    raises, returns None on URL / timeout / decode failures."""
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _ollama_running_models(base: str) -> list[dict]:
    """List of models currently loaded at `base` (via /api/ps), with metadata."""
    data = _http_get_json(base.rstrip("/") + "/api/ps")
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        details = m.get("details") or {}
        out.append({
            "name": name,
            "expires_at": m.get("expires_at") or "",
            "parameter_size": (details.get("parameter_size") or "").strip(),
            "context_length": m.get("context_length") or 0,
        })
    return out


def _ollama_installed_models(base: str) -> list[str]:
    data = _http_get_json(base.rstrip("/") + "/api/tags")
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if name:
            out.append(name)
    return out


def _try_ollama_with_loaded() -> BackendInfo | None:
    """Return BackendInfo if Ollama has a chat-capable model LOADED. None otherwise.

    Honors OLLAMA_MODEL / CHAT_OLLAMA_MODEL — these are STRONG hints from
    the user, so when set we return immediately even if the tag isn't
    currently in /api/ps (Ollama will load it on first request).
    """
    endpoint = _ollama_endpoint()

    for key in ("OLLAMA_MODEL", "CHAT_OLLAMA_MODEL"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return BackendInfo(
                name="ollama", model=raw,
                source=f"{key} env",
                endpoint=endpoint,
            )

    # Probe each loopback base; first reachable one wins.
    for base in _ollama_endpoints():
        running = _ollama_running_models(base)
        chat_running = [m for m in running if _is_chat_capable_tag(m["name"])]
        if not chat_running:
            continue
        # Sort by expires_at descending — Ollama bumps TTL on every use,
        # so the freshest entry is what the user most recently ran.
        chat_running.sort(key=lambda m: m.get("expires_at") or "", reverse=True)
        chosen = chat_running[0]
        names = [m["name"] for m in chat_running]
        skipped = [m["name"] for m in running if not _is_chat_capable_tag(m["name"])]
        tail = f" (skipped non-chat: {skipped})" if skipped else ""
        if len(names) == 1:
            src = f"loaded in ollama (/api/ps): {chosen['name']!r}{tail}"
        else:
            src = (
                f"loaded in ollama: {names} — picking most-recently-used "
                f"{chosen['name']!r} (latest expires_at){tail}"
            )
        return BackendInfo(
            name="ollama", model=chosen["name"],
            source=src,
            endpoint=base,
            context_length=chosen.get("context_length") or None,
        )
    return None


def _ollama_show_context_length(base: str, model: str) -> int | None:
    """Best-effort: pull the native context length from `/api/show`.

    Ollama's show payload nests it under `model_info.<arch>.context_length`
    where `<arch>` is e.g. `qwen2`, `llama`, `deepseek2`. We walk the
    model_info dict and return the first key ending in `.context_length`.
    Returns None on failure — caller treats absence as "unknown".
    """
    try:
        url = base.rstrip("/") + "/api/show"
        data = _http_post_json(url, {"name": model})
        if not isinstance(data, dict):
            return None
        info = data.get("model_info") or {}
        if not isinstance(info, dict):
            return None
        for k, v in info.items():
            if isinstance(k, str) and k.endswith(".context_length") and isinstance(v, int):
                return v
    except Exception:
        pass
    return None


def _ollama_full_fallback() -> BackendInfo | None:
    """When no model is LOADED — pick first installed chat-capable model.

    Used only as a last resort. /api/tags is a guess (the daemon will
    have to load the model on first request, which is slow).
    """
    endpoint = _ollama_endpoint()
    for base in _ollama_endpoints():
        installed = _ollama_installed_models(base)
        chat_installed = [n for n in installed if _is_chat_capable_tag(n)]
        if chat_installed:
            chosen = chat_installed[0]
            return BackendInfo(
                name="ollama", model=chosen,
                source=f"first installed (no model loaded): {chosen!r} of {chat_installed}",
                endpoint=base,
                context_length=_ollama_show_context_length(base, chosen),
            )
    # Daemon unreachable — nothing more to do.
    return None


# -----------------------------------------------------------------------------
# MLX detection helpers.
# -----------------------------------------------------------------------------


_MLX_IN_PROCESS_ENDPOINT = "in-process"
_MLX_PROC_MODEL_RE = re.compile(r"--model[=\s]+(\S+)")


def _mlx_server_mode_requested() -> bool:
    """True when the agent should talk to mlx_lm.server over HTTP."""
    prefer = (os.environ.get("LLM_BACKEND") or "").strip().lower()
    if prefer == "mlx-server":
        return True
    return bool(
        (os.environ.get("MLX_SERVER_URL") or os.environ.get("MLX_HOST") or "").strip()
    )


def _mlx_server_endpoint_url() -> str:
    raw = (
        os.environ.get("MLX_SERVER_URL")
        or os.environ.get("MLX_HOST")
        or ""
    ).strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:8080"
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw


def _mlx_endpoint() -> str:
    """Endpoint label for status surfaces."""
    if _mlx_server_mode_requested():
        return _mlx_server_endpoint_url()
    return _MLX_IN_PROCESS_ENDPOINT


def _mlx_process_model_arg() -> str | None:
    """Read `--model X` from a running mlx_lm.server process."""
    try:
        r = subprocess.run(
            ["ps", "-axo", "command"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if "mlx_lm.server" not in line and "mlx_lm/server.py" not in line:
            continue
        m = _MLX_PROC_MODEL_RE.search(line)
        if m:
            return m.group(1)
    return None


def _mlx_backend_info_for_model(model: str, *, source: str) -> BackendInfo:
    """Attach oMLX HTTP for glm5_next / deepseek_v4 / qwen4_exp, else in-process."""
    raw = (model or "").strip()
    ep = mlx_endpoint_for_model(raw)
    mid = omlx_api_model_id(raw) if requires_omlx_server(raw) else raw
    ctx_src = os.path.expanduser(raw)
    return BackendInfo(
        name="mlx",
        model=mid,
        source=source,
        endpoint=ep,
        context_length=_read_mlx_context_length(ctx_src),
    )


def _try_omlx_already_loaded() -> BackendInfo | None:
    """Session default when oMLX already has a chat model in Metal.

    Live `loaded=true` only — discovered-but-unloaded rows are ignored
    (BATTLEZ2 20260904: user typed a goal because GLM was already resident;
    TUI still sent glm5_next through in-process mlx_lm).
    """
    ep = omlx_default_endpoint()
    if not omlx_reachable(ep, timeout=0.4):
        return None
    data = _http_get_json(ep + "/v1/models/status", timeout=0.8)
    if not isinstance(data, dict):
        return None
    loaded_rows: list[dict] = []
    for m in data.get("models") or []:
        if not isinstance(m, dict) or not m.get("loaded") or not m.get("id"):
            continue
        mid = str(m["id"])
        if not _is_chat_capable_tag(mid):
            continue
        loaded_rows.append(m)
    if not loaded_rows:
        return None
    pinned = [m for m in loaded_rows if m.get("pinned")]
    row = pinned[0] if pinned else loaded_rows[0]
    mid = str(row["id"])
    disk = os.path.join(_omlx_model_dir(), mid)
    return BackendInfo(
        name="mlx",
        model=mid,
        source=f"oMLX already loaded: {mid}",
        endpoint=ep,
        context_length=_read_mlx_context_length(disk),
    )


def _try_mlx() -> BackendInfo | None:
    """Resolve MLX: explicit env, else oMLX resident, else first local folder."""
    if _mlx_server_mode_requested():
        return _try_mlx_server()
    env_model = (os.environ.get("MLX_MODEL") or "").strip()
    if env_model:
        return _mlx_backend_info_for_model(
            env_model, source=f"MLX_MODEL env: {env_model!r}",
        )
    resident = _try_omlx_already_loaded()
    if resident is not None:
        return resident
    return _try_mlx_in_process()


def _try_mlx_in_process() -> BackendInfo | None:
    """Resolve which local MLX folder to use when nothing is already loaded."""
    local = list_local_mlx_models()
    chat_local = [p for p in local if _is_chat_capable_tag(p)]
    if len(chat_local) == 1:
        path = chat_local[0]
        return _mlx_backend_info_for_model(
            path,
            source=f"only local MLX chat model: {os.path.basename(path)!r}",
        )
    if chat_local:
        path = chat_local[0]
        return _mlx_backend_info_for_model(
            path,
            source=(
                f"first of {len(chat_local)} local MLX models: "
                f"{os.path.basename(path)!r} "
                "(set MLX_MODEL to override)"
            ),
        )
    return None


def _read_mlx_context_length(model_path: str) -> int | None:
    """Pull the model's native context length from its config.json.

    Tries common key names in order (`max_position_embeddings` for
    Llama/Qwen, `max_seq_len` for some Mistral variants,
    `model_max_length` as a tokenizer-fallback). Returns None when the
    path isn't a local dir or the config lacks any of these keys —
    the status panel hides the row in that case.

    Best-effort: any exception swallowed (a malformed config shouldn't
    break backend resolution).
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    try:
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("max_position_embeddings", "max_seq_len", "model_max_length"):
            v = cfg.get(key)
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass
    return None


def _try_mlx_server() -> BackendInfo | None:
    """Return BackendInfo when mlx_lm.server is reachable. None otherwise.

    Model resolution order:
      1. MLX_MODEL env var.
      2. `--model` on the running mlx_lm.server process.
      3. Sole entry in /v1/models, else first entry.
    """
    endpoint = _mlx_server_endpoint_url()
    data = _http_get_json(endpoint.rstrip("/") + "/v1/models", timeout=1.0)
    if data is None:
        return None
    available: list[str] = []
    for m in (data.get("data") or []) if isinstance(data, dict) else []:
        if isinstance(m, dict) and m.get("id"):
            available.append(str(m["id"]))

    env_model = (os.environ.get("MLX_MODEL") or "").strip()
    if env_model:
        return BackendInfo(
            name="mlx", model=env_model,
            source=f"MLX_MODEL env → mlx_lm.server at {endpoint!r}",
            endpoint=endpoint,
            context_length=_read_mlx_context_length(env_model),
        )

    proc_model = _mlx_process_model_arg()
    if proc_model:
        return BackendInfo(
            name="mlx", model=proc_model,
            source=f"mlx_lm.server --model {proc_model!r}",
            endpoint=endpoint,
            context_length=_read_mlx_context_length(proc_model),
        )

    if len(available) == 1:
        mid = available[0]
        return BackendInfo(
            name="mlx", model=mid,
            source=f"only MLX model in /v1/models: {mid!r}",
            endpoint=endpoint,
        )
    if available:
        mid = available[0]
        return BackendInfo(
            name="mlx", model=mid,
            source=(
                f"first of {len(available)} in /v1/models: {mid!r} "
                "(set MLX_MODEL or restart mlx_lm.server with --model)"
            ),
            endpoint=endpoint,
        )
    return None


# -----------------------------------------------------------------------------
# oMLX — models stock mlx-lm cannot load (deepseek_v4, glm5_next)
# -----------------------------------------------------------------------------
# chat.py auto-starts oMLX when the user /model-picks DeepSeek-V4-Flash or
# GLM-5.3-Flash. GLM-5.2 / Qwen / MiniMax stay in-process; we do NOT set
# MLX_SERVER_URL globally.


def omlx_default_endpoint() -> str:
    """oMLX OpenAI base URL (default :8000). Override with OMLX_SERVER_URL."""
    raw = (os.environ.get("OMLX_SERVER_URL") or "").strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:8000"
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw


def requires_omlx_server(model: str) -> bool:
    """True when this MLX id/path needs oMLX (stock mlx-lm cannot load it).

    Matches DeepSeek-V4-Flash (`deepseek_v4`), GLM-5.3-Flash (`glm5_next`),
    and Qwen3.8-Flash-Next (`qwen4_exp`) by folder/HF name, or a local
    config.json with those model_type values.
    GLM-5.2 stays in-process mlx-lm — do not match a bare "glm-5" prefix.
    Dense Qwen3.8-27B stays in-process mlx-vlm — do not match bare "qwen3.8".
    """
    if not model or not str(model).strip():
        return False
    text = str(model).strip()
    low = text.lower().replace("\\", "/")
    base = os.path.basename(low.rstrip("/"))
    # Path or basename heuristics (HF folders, aliases).
    if "deepseek-v4" in low or "deepseek_v4" in low:
        return True
    if base.startswith("deepseek-v4") or "v4-flash" in base:
        return True
    # GLM-5.3-Flash (`glm5_next`) — not GLM-5.2 (`glm_moe_dsa`).
    if "glm-5.3" in low or "glm_5.3" in low or "glm5.3" in low or "glm5_next" in low:
        return True
    # Qwen3.8-Flash-Next (`qwen4_exp`). In-process mlx-vlm 0.6.17 loads the
    # language tower but rejects the 76 native-MTP tensors (strict
    # "Received 76 parameters not in model: language_model.mtp.*").
    # Vontra's 8bit-MTP card requires oMLX 0.6.3+ with qwen4_exp MTP.
    if "qwen3.8-flash" in low or "qwen4_exp" in low:
        return True
    # Local dir with authoritative config.
    cfg_path = os.path.join(text, "config.json") if os.path.isdir(text) else ""
    if cfg_path and os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            mt = str((cfg or {}).get("model_type") or "").lower()
            if mt in ("deepseek_v4", "glm5_next", "qwen4_exp"):
                return True
        except Exception:
            pass
    return False


def omlx_api_model_id(model: str) -> str:
    """Map a local folder path to the id oMLX advertises on GET /v1/models.

    oMLX discovers models by scanning --model-dir and registers each by
    **basename** (e.g. DeepSeek-V4-Flash-0731-MXFP4-MLX). Chat listing often
    keeps the absolute path; POSTing that path yields 404:
    Model '/Users/.../DeepSeek-...' not found.
    """
    text = (model or "").strip().rstrip("/\\")
    if not text:
        return text
    expanded = os.path.expanduser(text)
    # Path-like → basename; bare API ids / HF names unchanged.
    if os.path.isabs(expanded) or (os.sep in expanded) or (
        os.altsep is not None and os.altsep in expanded
    ):
        return os.path.basename(expanded)
    return text


def mlx_server_api_model_id(model: str, endpoint: str | None = None) -> str:
    """OpenAI `model` field for MLXServerBackend toward oMLX vs mlx_lm.server."""
    if requires_omlx_server(model):
        return omlx_api_model_id(model)
    ep = (endpoint or "").rstrip("/")
    if ep and ep == omlx_default_endpoint().rstrip("/"):
        return omlx_api_model_id(model)
    return model


def omlx_unload_model(
    model: str,
    endpoint: str | None = None,
    *,
    timeout: float = 180.0,
) -> tuple[bool, str]:
    """POST /v1/models/{id}/unload so Flash weights leave unified memory.

    Needed when switching TUI from oMLX Flash → in-process GLM/Qwen: in-process
    relief only clears MLXBackend._loaded_path (null while Flash was on oMLX),
    so Flash would stay resident (~150GB+) and the next GLM load OOM-kills chat.
    """
    model_id = omlx_api_model_id(model)
    if not model_id:
        return False, "empty model id"
    ep = (endpoint or omlx_default_endpoint()).rstrip("/")
    from urllib.parse import quote

    url = f"{ep}/v1/models/{quote(model_id, safe='')}/unload"
    try:
        req = urllib.request.Request(
            url,
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict) and data.get("status") == "ok":
                return True, f"unloaded {model_id}"
            return True, f"unload ok ({model_id}): {raw[:120]}"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            pass
        low = body.lower()
        # Already unloaded is success for relief purposes.
        if e.code == 400 and "not loaded" in low:
            return True, f"already unloaded: {model_id}"
        return False, f"HTTP {e.code}: {body[:200] or e.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, str(e)


def omlx_list_loaded_model_ids(
    endpoint: str | None = None,
    *,
    timeout: float = 3.0,
) -> list[str]:
    """Ids currently resident in oMLX (`GET /v1/models/status`). Unloaded = omitted."""
    ep = (endpoint or omlx_default_endpoint()).rstrip("/")
    data = _http_get_json(ep + "/v1/models/status", timeout=timeout)
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for m in data.get("models") or []:
        if isinstance(m, dict) and m.get("loaded") and m.get("id"):
            out.append(str(m["id"]))
    return out


def mlx_name_is_resident(
    name: str,
    *,
    in_process_active: str | None = None,
    omlx_loaded: list[str] | None = None,
) -> bool:
    """True when /list should show * for this MLX path or basename.

    `list_mlx_inventory` only knows the in-process mlx_lm cache
    (`MLXBackend._loaded_path`). GLM-5.3 lives in oMLX, so a pinned
    load never got a star (TUI 20260904). Compare via omlx_api_model_id
    because listing rows are full disk paths and oMLX ids are basenames.
    """
    if not name:
        return False
    if in_process_active:
        if name == in_process_active:
            return True
        if omlx_api_model_id(name) == omlx_api_model_id(in_process_active):
            return True
    ids = omlx_loaded if omlx_loaded is not None else omlx_list_loaded_model_ids()
    want = omlx_api_model_id(name)
    return any(omlx_api_model_id(x) == want for x in ids)


def omlx_unload_loaded_for_inprocess(
    *,
    keep: str | None = None,
    endpoint: str | None = None,
) -> list[str]:
    """Unload every resident oMLX model except optional `keep` basename.

    Used before an in-process GLM/Qwen load so orphaned Flash (left behind by
    a dead chat after /model swap) does not OOM the next session.
    """
    keep_id = omlx_api_model_id(keep) if keep else ""
    # Sept 2026: every model staged for a role this session (coder /
    # critic / architect on the same oMLX) is also kept — otherwise the
    # coder's per-stream unload evicts the critic's weights and vice versa.
    keep_ids = {keep_id} if keep_id else set()
    keep_ids |= {omlx_api_model_id(m) for m in OMLX_SESSION_KEEP_MODELS if m}
    freed: list[str] = []
    for mid in omlx_list_loaded_model_ids(endpoint):
        if mid in keep_ids:
            continue
        ok, _detail = omlx_unload_model(mid, endpoint)
        if ok:
            freed.append(mid)
    return freed


# Model ids/paths every staged role backend uses this session (filled by
# chat.py / coder.py when slots are staged). Read by
# `omlx_unload_loaded_for_inprocess` so role models are never evicted by
# each other's pre-stream cleanup. Same-model setups are unaffected.
OMLX_SESSION_KEEP_MODELS: set[str] = set()


def endpoint_is_omlx(endpoint: str | None) -> bool:
    """True when endpoint is the configured oMLX base (default :8000)."""
    ep = (endpoint or "").rstrip("/")
    if not ep:
        return False
    return ep == omlx_default_endpoint().rstrip("/")


def omlx_reachable(endpoint: str | None = None, *, timeout: float = 1.0) -> bool:
    """True when oMLX (or any OpenAI-compatible server) answers GET /v1/models."""
    ep = (endpoint or omlx_default_endpoint()).rstrip("/")
    data = _http_get_json(ep + "/v1/models", timeout=timeout)
    return isinstance(data, dict)


def _resolve_omlx_bin() -> str | None:
    """Find an `omlx` CLI: PATH, ~/.omlx/bin, or ~/MLX_Models/.omlx-venv."""
    which = shutil.which("omlx")
    if which:
        return which
    home = os.path.expanduser("~")
    for path in (
        os.path.join(home, ".omlx", "bin", "omlx"),
        os.path.join(home, "MLX_Models", ".omlx-venv", "bin", "omlx"),
    ):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _omlx_model_dir() -> str:
    """Directory oMLX should scan for MLX model folders."""
    env = (os.environ.get("OMLX_MODEL_DIR") or os.environ.get("MLX_MODELS_DIR") or "").strip()
    if env:
        # MLX_MODELS_DIR may be ':'-separated; take the first existing root.
        for part in env.split(":"):
            p = os.path.expanduser(part.strip())
            if p and os.path.isdir(p):
                return p
    default = os.path.join(os.path.expanduser("~"), "MLX_Models")
    return default


def _spawn_omlx_serve(endpoint: str) -> tuple[bool, str]:
    """Detached `omlx serve` or open oMLX.app. Does not wait for readiness."""
    bin_path = _resolve_omlx_bin()
    model_dir = _omlx_model_dir()
    # Parse port from endpoint for --port (default 8000).
    port = "8000"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint if "://" in endpoint else "http://" + endpoint)
        if parsed.port:
            port = str(parsed.port)
    except Exception:
        pass

    if bin_path:
        cmd = [
            bin_path,
            "serve",
            "--model-dir",
            model_dir,
            "--hot-cache-max-size",
            # oMLX 0.5.7 CLI rejects percentages ("20%") — use absolute GB.
            # Admin UI may show "%"; settings.json must be parseable by serve.
            "32GB",
            "--port",
            port,
        ]
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"spawned {' '.join(cmd)}"
        except OSError as e:
            return False, f"failed to spawn omlx serve: {e!r}"

    # Fall back to menu-bar app (auto_start_on_launch in ~/.omlx/settings.json).
    app_candidates = (
        "/Applications/oMLX.app",
        os.path.expanduser("~/Applications/oMLX.app"),
    )
    for app in app_candidates:
        if os.path.isdir(app):
            try:
                subprocess.Popen(
                    ["open", app],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True, f"opened {app}"
            except OSError as e:
                return False, f"failed to open {app}: {e!r}"
    try:
        subprocess.Popen(
            ["open", "-a", "oMLX"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, "open -a oMLX"
    except OSError as e:
        return False, (
            "oMLX not installed (no `omlx` CLI and no oMLX.app). "
            f"Install oMLX 0.6.3+ then retry. ({e!r})"
        )


def ensure_omlx_server(*, timeout_s: float = 120.0) -> str:
    """Ensure oMLX is reachable; spawn it if needed. Returns endpoint URL.

    Raises RuntimeError with an install/start hint when the server never comes up.
    """
    endpoint = omlx_default_endpoint()
    if omlx_reachable(endpoint):
        return endpoint

    ok, detail = _spawn_omlx_serve(endpoint)
    if not ok:
        raise RuntimeError(
            f"This architecture needs oMLX but could not start it: {detail}\n"
            "Install oMLX 0.6.3+ (app or `omlx` CLI), or start it once; "
            "chat will reuse http://127.0.0.1:8000 afterward."
        )

    # Longer poll than Ollama — oMLX cold start / Metal init is slower.
    deadline = time.monotonic() + max(5.0, float(timeout_s))
    while time.monotonic() < deadline:
        if omlx_reachable(endpoint, timeout=1.5):
            return endpoint
        time.sleep(0.5)

    raise RuntimeError(
        f"oMLX did not become reachable at {endpoint!r} within {timeout_s:.0f}s "
        f"(start detail: {detail}). Check /Applications/oMLX.app or "
        "`omlx serve --model-dir ~/MLX_Models`, then /model again."
    )


def mlx_endpoint_for_model(model: str) -> str:
    """Endpoint for an MLX model pick: oMLX URL for deepseek_v4 / glm5_next /
    qwen4_exp, else in-process unless the user explicitly requested mlx-server via env.
    """
    if requires_omlx_server(model):
        return omlx_default_endpoint()
    if _mlx_server_mode_requested():
        return _mlx_server_endpoint_url()
    return _MLX_IN_PROCESS_ENDPOINT


# -----------------------------------------------------------------------------
# Convenience listing — used by chat.py /list and similar surfaces.
# -----------------------------------------------------------------------------


def unload_ollama_model(name: str, endpoint: str | None = None) -> tuple[bool, str]:
    """Tell Ollama to evict `name` from VRAM by POSTing keep_alive=0.

    Tries /api/chat (empty messages) then /api/generate — the agent loads
    models via chat, and either endpoint accepts keep_alive=0 to drop VRAM.
    """
    base = (endpoint or _ollama_endpoint()).rstrip("/")
    attempts = [
        (
            base + "/api/chat",
            {"model": name, "messages": [], "keep_alive": 0},
        ),
        (
            base + "/api/generate",
            {"model": name, "prompt": "", "keep_alive": 0},
        ),
    ]
    last_err = ""
    for url, body in attempts:
        data = _http_post_json(url, body, timeout=30.0)
        if data is None:
            last_err = f"no response from {url}"
            continue
        if data.get("done_reason") == "unload" or data.get("done") is True:
            return True, f"unloaded {name!r}"
    if last_err:
        return False, last_err
    # Active chat sessions use keep_alive=-1 and can reload immediately.
    time.sleep(0.4)
    still = [
        m for m in _ollama_running_models(base)
        if name in (m.get("name") or "")
    ]
    if still:
        return (
            False,
            f"still loaded at {base.rsplit(':', 1)[-1]} after unload "
            "(stop the running game session, or another client is holding the model)",
        )
    return True, f"unloaded {name!r}"


def auto_fix_ollama_tensor_split(endpoint: str | None = None) -> tuple[bool, str]:
    """On large-GPU workstations, drop tensor-split Ollama VRAM before a session.

    Uses existing ``/unload`` machinery (``keep_alive=0``). No manual
    ``OLLAMA_SCHED_SPREAD`` or systemd edits required. Skipped on small-GPU
    topologies where split may be intentional. Disable with
    ``AGENT_NO_AUTO_OLLAMA_GPU_FIX=1``.
    """
    if os.environ.get("AGENT_NO_AUTO_OLLAMA_GPU_FIX", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return False, ""
    try:
        import gpu_status as gs
    except Exception:
        return False, ""
    snap = gs.snapshot_gpus(force=True)
    if not gs.ollama_is_tensor_split(snap):
        return False, ""
    if not gs.prefer_single_gpu_workstation(snap):
        return False, ""
    results = unload_all_ollama_models(endpoint)
    if not results:
        return False, ""
    names = [n for n, ok, _ in results if ok]
    if not names:
        err = results[0][2] if results else "unload failed"
        return False, err
    gs.snapshot_gpus(force=True)
    return True, (
        "released split Ollama VRAM — next LLM request reloads "
        f"({', '.join(names)})"
    )


# Standard multi-GPU slot ports (mirrors ensure_ollama_slot_daemons_for_chat).
_OLLAMA_SLOT_PORTS = (11434, 11435, 11436)


def ollama_unload_probe_bases() -> list[str]:
    """All Ollama HTTP bases to probe for /unload all (deduped by port)."""
    try:
        import gpu_status as gs
        bases = list(gs.ollama_all_api_bases())
    except Exception:
        bases = [_ollama_endpoint()]
    by_port: dict[int, str] = {}
    for raw in bases:
        m = re.search(r":(\d+)$", raw.rstrip("/"))
        if not m:
            continue
        port = int(m.group(1))
        by_port.setdefault(port, f"http://127.0.0.1:{port}")
    for port in _OLLAMA_SLOT_PORTS:
        base = f"http://127.0.0.1:{port}"
        if port not in by_port and _endpoint_ready(base):
            by_port[port] = base
    return [by_port[p] for p in sorted(by_port)]


def _unload_all_at_endpoint(base: str) -> list[tuple[str, bool, str]]:
    loaded = _ollama_running_models(base)
    return [
        (m["name"], *unload_ollama_model(m["name"], endpoint=base))
        for m in loaded
    ]


def unload_all_ollama_models(endpoint: str | None = None) -> list[tuple[str, bool, str]]:
    """Walk /api/ps and unload every loaded model.

    When ``endpoint`` is None, every reachable Ollama daemon is probed
    (slot 1–3 on the 4-GPU box, not only ``OLLAMA_HOST``). Messages
    include the endpoint so multi-daemon runs are easy to audit.
    """
    if endpoint is not None:
        return _unload_all_at_endpoint(endpoint.rstrip("/"))

    results: list[tuple[str, bool, str]] = []
    for base in ollama_unload_probe_bases():
        for name, ok, msg in _unload_all_at_endpoint(base):
            port = base.rsplit(":", 1)[-1]
            results.append((name, ok, f"{msg} ({port})"))
    return results


def mlx_server_pids() -> list[int]:
    """PIDs of running mlx_lm.server processes — for /unload mlx hints."""
    try:
        r = subprocess.run(
            ["ps", "-axo", "pid,command"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0:
        return []
    pids: list[int] = []
    for line in r.stdout.splitlines():
        if "mlx_lm.server" not in line and "mlx_lm/server.py" not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def list_ollama_inventory() -> tuple[list[str], set[str]]:
    """(installed_chat_tags, currently_loaded_tags) — for /list display.

    Filters non-chat tags (z-image, embedders, ...) so the numbered list
    only includes models that can actually answer /load N. The "loaded"
    set merges /api/ps across every probed Ollama slot (11434–11436), not
    only OLLAMA_HOST — so ``*`` in /list matches what /unload all touches.
    """
    installed: list[str] = []
    loaded: set[str] = set()
    bases = ollama_unload_probe_bases()
    for base in bases:
        if not installed:
            installed = _ollama_installed_models(base)
        for m in _ollama_running_models(base):
            if m.get("name"):
                loaded.add(m["name"])
    return [n for n in installed if _is_chat_capable_tag(n)], loaded


def list_mlx_inventory() -> tuple[list[str], str | None]:
    """(downloaded_chat_models, active_model_or_None) — for /list display."""
    if _mlx_server_mode_requested():
        endpoint = _mlx_server_endpoint_url()
        data = _http_get_json(endpoint.rstrip("/") + "/v1/models", timeout=1.0)
        if data is None:
            return [], None
        all_ids = [
            m["id"] for m in (data.get("data") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        active = (
            _mlx_process_model_arg()
            or (os.environ.get("MLX_MODEL") or "").strip()
            or None
        )
        downloaded = [name for name in all_ids if _is_chat_capable_tag(name)]
        if active and active not in downloaded and active in all_ids:
            downloaded.append(active)
        return downloaded, active

    active = MLXBackend._loaded_path or (os.environ.get("MLX_MODEL") or "").strip() or None
    local_paths = list_local_mlx_models()

    merged: list[str] = []
    seen: set[str] = set()
    for path in local_paths:
        base = os.path.basename(path)
        if path in seen or base in seen:
            continue
        # The chat-cap check has to look at the full path: HF cache
        # layouts put the SHA in basename and the model id 2 levels up
        # (`hub/models--<org>--<name>/snapshots/<sha>`), so checking
        # only `base` would let embedding models slip through.
        if not _is_chat_capable_tag(path):
            continue
        merged.append(path)
        seen.add(path)
        seen.add(base)
    if active and active not in seen:
        merged.append(active)
    return merged, active


def discover_local_vlm() -> str | None:
    """First locally-downloaded MLX model whose name classifies as VLM.

    Used by GameAgent to auto-enable the vision_judge visual-progress
    check when a vision-capable model is available on disk — keeps the
    judge entirely local (no Anthropic fallback). Returns None when no
    local VLM is found; callers must treat None as "no signal", not as
    an error.
    """
    downloaded, _active = list_mlx_inventory()
    for entry in downloaded:
        base = entry.split("/")[-1] if "/" in entry else entry
        if classify_model_modality(base) == "vlm":
            return entry
    # Also scan dirs that list_mlx_inventory's _is_chat_capable_tag may
    # filter out — some VLM packagings don't ship a chat template marker
    # in the path. Fall through to a direct scan of the same dirs.
    for root in _default_mlx_search_dirs():
        for path in _scan_mlx_models_dir(root):
            base = path.split("/")[-1] if "/" in path else path
            if classify_model_modality(base) == "vlm":
                return path
    return None


def _default_mlx_search_dirs() -> list[str]:
    """Where to look for locally-downloaded MLX models.

    Override per-machine via the MLX_MODELS_DIR env var (single path or
    `:`-separated list). Defaults cover the common machine layouts the
    user has used: `~/MLX_Models`, then HF cache.
    """
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "MLX_Models"),
        os.path.join(home, "Models_MLX"),
        os.path.join(home, ".cache", "huggingface", "hub"),
        "/opt/mlx_models",
    ]


def _is_mlx_model_dir(path: str) -> bool:
    """A directory looks like an MLX model when it has config.json plus
    at least one .safetensors file."""
    try:
        if not os.path.isfile(os.path.join(path, "config.json")):
            return False
        for name in os.listdir(path):
            if name.endswith(".safetensors"):
                return True
    except OSError:
        return False
    return False


def _scan_mlx_models_dir(root: str) -> list[str]:
    """Find downloaded MLX model directories under `root`.

    Direct children that look like model dirs win. We also walk one
    level into HF-cache style layouts (`models--org--name/snapshots/<sha>/`)
    so the HF cache is covered without a separate scanner.
    """
    out: list[str] = []
    if not root or not os.path.isdir(root):
        return out
    seen: set[str] = set()
    try:
        children = list(os.scandir(root))
    except OSError:
        return out
    for entry in children:
        if not entry.is_dir(follow_symlinks=False):
            continue
        if _is_mlx_model_dir(entry.path):
            ap = os.path.abspath(entry.path)
            if ap not in seen:
                out.append(ap)
                seen.add(ap)
            continue
        snapshots = os.path.join(entry.path, "snapshots")
        if os.path.isdir(snapshots):
            try:
                for snap in os.scandir(snapshots):
                    if snap.is_dir(follow_symlinks=False) and _is_mlx_model_dir(snap.path):
                        ap = os.path.abspath(snap.path)
                        if ap not in seen:
                            out.append(ap)
                            seen.add(ap)
            except OSError:
                pass
    out.sort()
    return out


def list_local_mlx_models() -> list[str]:
    """All locally-downloaded MLX model paths the user can launch.

    Walks every entry in MLX_MODELS_DIR (env-overridable, `:`-separated)
    plus the platform defaults from `_default_mlx_search_dirs`. Result
    is a stable, deduped list of absolute directory paths suitable for
    passing to `mlx_lm.server --model <path>`.
    """
    raw_env = (os.environ.get("MLX_MODELS_DIR") or "").strip()
    roots: list[str] = []
    if raw_env:
        roots.extend(p.strip() for p in raw_env.split(":") if p.strip())
    roots.extend(_default_mlx_search_dirs())
    out: list[str] = []
    seen: set[str] = set()
    for r in roots:
        for p in _scan_mlx_models_dir(os.path.expanduser(r)):
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out


def mlx_endpoint_url() -> str:
    """Public alias for the MLX endpoint (in-process label or server URL)."""
    return _mlx_endpoint()


def ollama_endpoint_url(slot: int = 1) -> str:
    """Public alias for the resolved Ollama endpoint URL (per model slot)."""
    return _ollama_endpoint_for_slot(slot)


def ollama_context_length(endpoint: str, model: str) -> int | None:
    """Native context length from ``/api/show``; None if unreachable."""
    return _ollama_show_context_length(endpoint, model)
