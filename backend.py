"""Backend abstraction — Ollama and MLX as peer LLM hosts.

Two implementations share one streaming contract so the agent loop, TUI,
and CLI never have to know which daemon they are talking to:

  * `OllamaBackend`  — thin wrapper around `ollama.AsyncClient` that
    delegates to the existing watchdog/retry helpers in `ollama_io.py`.

  * `MLXBackend`     — loads the MLX model in-process and streams via
    `mlx_lm.stream_generate` directly. No HTTP, no `mlx_lm.server`,
    no broken pipes. The model is held in a class-level cache so
    subsequent requests reuse the loaded weights. Cancellation is
    plumbed through a `threading.Event` that the worker thread checks
    between tokens, so Ctrl-D in the TUI actually stops a mid-stream
    call.

`detect_backend()` picks an LLM daemon at session start. The rule:

  1. Honor `LLM_BACKEND=ollama|mlx|auto` when set (CLI `--backend` counts too).
  2. On macOS (`darwin`), if neither env nor argument picks a backend,
     default to **MLX** (Apple GPU). Linux and others default to `auto`.
  3. With preference `auto`: probe both; if MLX_MODEL is set or a single
     local MLX model is discoverable → MLX. Otherwise check Ollama.
  4. With preference `mlx` or `ollama`: force that daemon or raise.

For MLX, "which model" comes from:
  1. `MLX_MODEL` env var (explicit path or HF id)
  2. The single MLX model found in `~/.MLX_Models/` (or HF cache)
  3. Otherwise: raise — there's nothing to load.
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
from dataclasses import dataclass
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
    # (without ".6") was NOT unified — keep that prefix OUT of this
    # list so plain Qwen3-30B etc. stay labeled text-only.
    "qwen3.6-27b", "qwen3.6-7b", "qwen3.6-72b", "qwen3.6-235b",
    "qwen3.6:27b", "qwen3.6:7b", "qwen3.6:72b", "qwen3.6:235b",
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


# Cloud-backend defaults. The curated single-model-each shape keeps /list
# tight; edit these constants (or add more entries to the inventory lists
# below) to expose more variants. API keys are read from env at request
# time — never from disk, never embedded in BackendInfo.
_OPENAI_DEFAULT_MODEL = "gpt-5"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"
_OPENAI_MODELS: tuple[str, ...] = (_OPENAI_DEFAULT_MODEL,)
# Curated /list inventory — edit here to expose more Claude variants.
_ANTHROPIC_MODELS: tuple[str, ...] = (
    "claude-fable-5",
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


# MLX request fields we forward when present in the agent's `options` dict.
# Anything else (including `num_ctx`) is silently dropped — MLX uses the
# model's native context, no equivalent knob.
_MLX_OPTION_KEYS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p",
    "seed", "max_tokens",
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
    def _drop_after_crash(cls) -> None:
        """Free GPU state after the MLX worker raised mid-stream.

        Nulls the cached model/tokenizer references and queues a
        `mlx.core.metal.clear_cache()` on the MLX thread so the next
        stream re-enters the load path with a clean Metal context.
        Without this, a single crash made the rest of the session
        unusable until the user killed and restarted chat.py (the
        process-wide Metal allocator was full of dead tensors).

        Safe to call from any thread — only touches Python refs here
        and schedules the actual Metal work back onto the MLX thread.
        """
        cls._loaded_model = None
        cls._loaded_tokenizer = None
        cls._loaded_path = None
        cls._loaded_vlm_model = None
        cls._loaded_vlm_processor = None
        cls._loaded_vlm_config = None
        cls._loaded_vlm_path = None
        import gc
        gc.collect()

        def _clear_on_mlx_thread() -> None:
            try:
                import mlx.core as mx  # type: ignore
                metal = getattr(mx, "metal", None)
                if metal is not None and hasattr(metal, "clear_cache"):
                    metal.clear_cache()
            except Exception:
                pass
            import gc as _gc
            _gc.collect()

        try:
            cls._get_mlx_executor().submit(_clear_on_mlx_thread)
        except Exception:
            pass

    @classmethod
    def _load_sync(cls, path: str) -> tuple[Any, Any]:
        """Blocking load. MUST run on the dedicated MLX thread (see
        `_get_mlx_executor`) — Metal binds to the calling thread.
        """
        if cls._loaded_path == path and cls._loaded_model is not None:
            return cls._loaded_model, cls._loaded_tokenizer
        # If a different model was previously loaded, drop the
        # reference so Python+MLX can reclaim memory before the new
        # weights arrive.
        if cls._loaded_model is not None:
            cls._loaded_model = None
            cls._loaded_tokenizer = None
            cls._loaded_path = None
            import gc
            gc.collect()
        # Defer the mlx_lm import so Ollama-only users don't pay its
        # ~1-2 s cold-start cost.
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
        if cls._loaded_vlm_model is not None:
            cls._loaded_vlm_model = None
            cls._loaded_vlm_processor = None
            cls._loaded_vlm_config = None
            cls._loaded_vlm_path = None
        if cls._loaded_model is not None and cls._loaded_path != path:
            cls._loaded_model = None
            cls._loaded_tokenizer = None
            cls._loaded_path = None
        import gc as _gc
        _gc.collect()
        from mlx_vlm import load as _vlm_load  # type: ignore
        from mlx_vlm.utils import load_config as _vlm_load_config  # type: ignore
        model, processor = _vlm_load(path)
        config = _vlm_load_config(path)
        cls._loaded_vlm_model = model
        cls._loaded_vlm_processor = processor
        cls._loaded_vlm_config = config
        cls._loaded_vlm_path = path
        return model, processor, config

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
                    prompt = tokenizer.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = "\n\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in history
                    ) + "\n\nassistant:"

                if has_prefill:
                    prompt += prefill_content

                from mlx_lm.sample_utils import make_sampler  # type: ignore
                sampler = make_sampler(
                    temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k,
                )

                from mlx_lm import stream_generate  # type: ignore
                last_gen = None
                for gen in stream_generate(
                    model, tokenizer, prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_progress_callback=_prompt_progress,
                    prefill_step_size=prefill_step_size,
                ):
                    if worker_cancel.is_set():
                        break
                    last_gen = gen
                    loop.call_soon_threadsafe(
                        q.put_nowait, ("text", gen.text, gen.prompt_tokens, gen.generation_tokens)
                    )
                pt = getattr(last_gen, "prompt_tokens", None) if last_gen else None
                ct = getattr(last_gen, "generation_tokens", None) if last_gen else None
                loop.call_soon_threadsafe(q.put_nowait, ("done", None, pt, ct))
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
                    prompt = _vlm_template(
                        processor, config, history,
                        num_images=len(image_paths),
                    )
                except Exception:
                    # Same naive fallback as text-only.
                    prompt = "\n\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in history
                    ) + "\n\nassistant:"

                if has_prefill:
                    prompt += prefill_content

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
                    # generation goes to a reasoning channel) past the
                    # wall-clock floor, abort instead of waiting for
                    # stall_seconds (default 600s) which is too long.
                    if (
                        n_tokens == 0
                        and (time.monotonic() - started) >= 180.0
                    ):
                        silent = True
                        stall_at = 0
                        worker_cancel.set()
                        break
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
                if repeat.feed(piece):
                    if _should_grace_inline_data_bloat(
                        stall_reason=repeat.stall_reason,
                        assembled_text="".join(parts),
                        grace_already_used=loop_grace_used,
                        completion_tokens=n_tokens,
                    ):
                        loop_grace_used = True
                        loop_grace_reason = "inline_data_bloat_unclosed_html_file"
                        # One-shot grace only; if repetition returns after
                        # reset, we abort as usual.
                        repeat = RepetitionDetector()
                        continue
                    looped = True
                    stall_at = n_tokens
                    worker_cancel.set()
                    break
                if delib.feed(piece):
                    deliberated = True
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
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        t0 = time.monotonic()
        cancelled = False

        max_tokens_hit = False

        async def _run(call_params: dict[str, Any]) -> None:
            nonlocal tokens, prompt_tokens, completion_tokens, cancelled
            nonlocal max_tokens_hit
            stream = await self._client.chat.completions.create(**call_params)
            async for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if chunk.choices:
                    delta = chunk.choices[0].delta
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
            raise RuntimeError(
                "MLX backend selected but no MLX model could be resolved.\n"
                "Set MLX_MODEL=<path-or-id> to point at a downloaded MLX "
                "model, or place a model under ~/MLX_Models/ so it's "
                "auto-discovered."
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
            model=_OPENAI_DEFAULT_MODEL,
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
            model=_ANTHROPIC_DEFAULT_MODEL,
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
    instead of dangling unusable picks. The list is a curated constant
    (edit _OPENAI_MODELS in this file to add more); we don't probe the
    API just to populate /list — that would burn quota every time the
    user typed it.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return [], None
    return list(_OPENAI_MODELS), _OPENAI_DEFAULT_MODEL


def list_anthropic_inventory() -> tuple[list[str], str | None]:
    """Mirror of list_openai_inventory for Anthropic / Claude."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return [], None
    return list(_ANTHROPIC_MODELS), _ANTHROPIC_DEFAULT_MODEL


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


def _mlx_endpoint() -> str:
    """Pseudo-endpoint string for the in-process MLX backend.

    Kept as a public function so callers / status displays that show
    'where is the model running?' have something stable to print.
    No HTTP is involved; the model lives in this Python process's
    GPU VRAM.
    """
    return _MLX_IN_PROCESS_ENDPOINT


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


def _try_mlx() -> BackendInfo | None:
    """Resolve which MLX model to load in-process. None if nothing usable.

    Resolution order:
      1. MLX_MODEL env var (path or HF id).
      2. Single locally-downloaded MLX model under MLX_MODELS_DIR /
         ~/MLX_Models / HF cache (unambiguous).
      3. First locally-downloaded MLX model (with a warning in source).
    """
    env_model = (os.environ.get("MLX_MODEL") or "").strip()
    if env_model:
        return BackendInfo(
            name="mlx", model=env_model,
            source=f"MLX_MODEL env: {env_model!r}",
            endpoint=_MLX_IN_PROCESS_ENDPOINT,
            context_length=_read_mlx_context_length(env_model),
        )

    local = list_local_mlx_models()
    chat_local = [p for p in local if _is_chat_capable_tag(p)]
    if len(chat_local) == 1:
        path = chat_local[0]
        return BackendInfo(
            name="mlx", model=path,
            source=f"only local MLX chat model: {os.path.basename(path)!r}",
            endpoint=_MLX_IN_PROCESS_ENDPOINT,
            context_length=_read_mlx_context_length(path),
        )
    if chat_local:
        path = chat_local[0]
        return BackendInfo(
            name="mlx", model=path,
            source=(
                f"first of {len(chat_local)} local MLX models: "
                f"{os.path.basename(path)!r} "
                "(set MLX_MODEL to override)"
            ),
            endpoint=_MLX_IN_PROCESS_ENDPOINT,
            context_length=_read_mlx_context_length(path),
        )
    return None


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
    """Always returns [] now that MLX runs in-process.

    Kept as a no-op shim for chat.py /unload-mlx surfaces; once those
    surfaces are reworked for the in-process model this function can
    be deleted.
    """
    return []


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
    """(downloaded_chat_models, active_model_or_None) — for /list display.

    "Active" = whatever the in-process MLX backend has currently loaded
    (MLXBackend._loaded_path), or the env-set MLX_MODEL if nothing's
    loaded yet this session.

    The list is built from a local disk scan of MLX_MODELS_DIR + the
    platform defaults; the active model is appended even if it would
    otherwise be filtered.
    """
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
    """Public alias for the MLX endpoint string. Returns "in-process" now
    that MLX runs in this Python process — kept as a function so callers
    that render an endpoint label have something stable to display.
    """
    return _mlx_endpoint()


def ollama_endpoint_url(slot: int = 1) -> str:
    """Public alias for the resolved Ollama endpoint URL (per model slot)."""
    return _ollama_endpoint_for_slot(slot)


def ollama_context_length(endpoint: str, model: str) -> int | None:
    """Native context length from ``/api/show``; None if unreachable."""
    return _ollama_show_context_length(endpoint, model)
