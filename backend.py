"""Backend abstraction — Ollama and MLX as peer LLM hosts.

Two implementations share one streaming contract so the agent loop, TUI,
and CLI never have to know which daemon they are talking to:

  * `OllamaBackend`  — thin wrapper around `ollama.AsyncClient` that
    delegates to the existing watchdog/retry helpers in `ollama_io.py`.
    Behavior on the Ollama path is byte-for-byte unchanged.

  * `MLXBackend`     — talks to `mlx_lm.server` over HTTP using its
    OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming.
    Reuses the same per-chunk stall watchdog and the same repetition
    detector as the Ollama path so a wedged MLX stream is detected the
    same way a wedged Ollama stream is.

`detect_backend()` probes both daemons at session start. The picking
rule (per user decision):

  1. Honor `LLM_BACKEND=ollama|mlx` if set.
  2. If both have a model loaded → MLX wins (faster on Apple Silicon).
  3. If exactly one has a model loaded → that one.
  4. If neither has a model loaded but Ollama is reachable → Ollama
     with /api/tags fallback (matches today's behavior).
  5. Otherwise raise.

For MLX, "what's loaded" is read from the running `mlx_lm.server`
process's `--model` arg (mirrors how `ollama ps` works), falling back
to `/v1/models[0]` when no `--model` flag was given. `MLX_MODEL=<id>`
overrides.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Literal

import httpx
import ollama

from ollama_io import (
    Candidate,
    RepetitionDetector,
    StreamResult,
    stream_chat_with_retry,
)


# Tags that are clearly NOT chat models. Same list as chat.py used to
# carry — kept here so detection can filter ps results consistently
# across drivers.
_NON_CHAT_TAG_FRAGMENTS: tuple[str, ...] = (
    "z-image", "stable-diffusion", "sdxl", "flux",
    "embed", "embedding", "minilm", "bge-", "rerank",
    "whisper", "tts-",
)


def _is_chat_capable_tag(name: str) -> bool:
    n = (name or "").lower()
    return not any(frag in n for frag in _NON_CHAT_TAG_FRAGMENTS)


# -----------------------------------------------------------------------------
# Public types.
# -----------------------------------------------------------------------------


@dataclass
class BackendInfo:
    """Resolved backend identity. What the TUI prints, what the agent uses."""

    name: Literal["ollama", "mlx"]
    model: str
    source: str               # human-readable provenance ("loaded in ollama (/api/ps): 'qwen3.6:27b'")
    endpoint: str             # base URL — "http://127.0.0.1:11434" or "http://127.0.0.1:8080"
    context_length: int | None = None


class Backend(ABC):
    """Common interface implemented by OllamaBackend and MLXBackend."""

    info: BackendInfo

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        stall_seconds: float = 90.0,
        overall_seconds: float = 600.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
    ) -> StreamResult:
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
        stall_seconds: float = 90.0,
        overall_seconds: float = 600.0,
        scorer: Callable[[str], Awaitable[tuple[float, dict]]],
        on_progress: Callable[[int, str], None] | None = None,
        early_exit_score: float = 1.0,
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
            result = await self.stream_chat(
                messages,
                on_token=None,
                options=opts,
                stall_seconds=stall_seconds,
                overall_seconds=overall_seconds,
                max_retries=0,
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
        stall_seconds: float = 90.0,
        overall_seconds: float = 600.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
    ) -> StreamResult:
        return await stream_chat_with_retry(
            self._client,
            self.info.model,
            messages,
            on_token=on_token,
            options=options,
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
# Anything else (including `num_ctx`) is silently dropped — mlx_lm.server
# uses the model's native context, no equivalent knob.
_MLX_OPTION_KEYS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "repetition_context_size",
    "seed", "max_tokens",
)


class MLXBackend(Backend):
    """Talks to `mlx_lm.server` via its OpenAI-compatible HTTP API."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        # No persistent client: each stream opens a fresh httpx.AsyncClient
        # so cancellation cleanly closes connections. mlx_lm.server is
        # local; connection setup is essentially free.

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        on_token: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        stall_seconds: float = 90.0,
        overall_seconds: float = 600.0,
        max_retries: int = 1,
        on_stall: Callable[[StreamResult, int], None] | None = None,
    ) -> StreamResult:
        last: StreamResult | None = None
        for attempt in range(max_retries + 1):
            result = await self._stream_once(
                messages,
                on_token=on_token,
                options=options,
                stall_seconds=stall_seconds,
                overall_seconds=overall_seconds,
            )
            last = result
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
    ) -> StreamResult:
        opts = dict(options or {})
        body: dict[str, Any] = {
            "model": self.info.model,
            "messages": _strip_ollama_only_fields(messages),
            "stream": True,
            # Ask mlx_lm.server to emit a final SSE frame carrying token
            # counts in `usage` (OpenAI-compatible). Without this we have
            # no input-token visibility on the MLX path.
            "stream_options": {"include_usage": True},
        }
        # Carry agent-controlled sampler params through to mlx_lm.server.
        for key in _MLX_OPTION_KEYS:
            if key in opts:
                body[key] = opts[key]
        # Reasonable token ceiling so mlx_lm.server's 512 default doesn't
        # truncate full HTML game files. Caller can lift it via options.
        body.setdefault("max_tokens", 16384)

        started = time.monotonic()
        parts: list[str] = []
        n_tokens = 0
        stalled = False
        looped = False
        stall_at: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        # Shared repetition detector — same class both backends use, so
        # tuning happens in exactly one place (ollama_io.RepetitionDetector).
        repeat = RepetitionDetector()

        try:
            async with httpx.AsyncClient(base_url=self.info.endpoint, timeout=None) as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=body,
                    headers={"accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    ait = response.aiter_lines().__aiter__()
                    while True:
                        try:
                            line = await asyncio.wait_for(
                                ait.__anext__(), timeout=stall_seconds
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            stalled = True
                            stall_at = n_tokens
                            break

                        if time.monotonic() - started > overall_seconds:
                            stalled = True
                            stall_at = n_tokens
                            break

                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].lstrip()
                        if not payload:
                            continue
                        if payload == "[DONE]":
                            break
                        # SSE "comment" frames (mlx_lm.server uses these
                        # for prompt-processing keepalives like
                        # "data: : keepalive 50/200").
                        if payload.startswith(":"):
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        # Final usage frame (when include_usage=True): the
                        # OpenAI spec sends choices=[] with usage populated.
                        # Capture before the choices-empty skip below.
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            pt = usage.get("prompt_tokens")
                            ct = usage.get("completion_tokens")
                            if isinstance(pt, int):
                                prompt_tokens = pt
                            if isinstance(ct, int):
                                completion_tokens = ct
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or ""
                        if not piece:
                            # Some servers send an empty delta as the
                            # first frame to set role=assistant. Skip.
                            continue
                        parts.append(piece)
                        n_tokens += 1
                        if on_token is not None:
                            try:
                                on_token(piece)
                            except Exception:
                                pass

                        # Shared repetition detector — see ollama_io.RepetitionDetector
                        # for the two-window strategy. Identical behavior on
                        # both backends.
                        if repeat.feed(piece):
                            looped = True
                            stall_at = n_tokens
                            break
        except (httpx.HTTPError, OSError):
            stalled = True
            if stall_at is None:
                stall_at = n_tokens

        return StreamResult(
            text="".join(parts),
            tokens=n_tokens,
            duration_s=time.monotonic() - started,
            stalled=stalled or looped,
            stall_at_token=stall_at,
            looped=looped,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def is_vlm(self) -> bool:
        # mlx_lm.server's OpenAI-compatible API is text-only today; vision
        # MLX models exist but the server doesn't expose image input the
        # way ollama.show() lets us check capabilities. Treat as False so
        # the agent keeps screenshots out of MLX prompts.
        return False


def _strip_ollama_only_fields(messages: list[dict]) -> list[dict]:
    """Remove fields Ollama uses that mlx_lm.server doesn't understand.

    Today the only such field is `images` (Ollama's bytes-list shape for
    vision). Returns a copy; the agent's stored history is untouched.
    """
    out: list[dict] = []
    for m in messages:
        if "images" in m:
            mm = dict(m)
            mm.pop("images", None)
            out.append(mm)
        else:
            out.append(m)
    return out


# -----------------------------------------------------------------------------
# Detection.
# -----------------------------------------------------------------------------


def detect_backend(prefer: str | None = None) -> BackendInfo:
    """Probe both daemons and decide which to use.

    `prefer` overrides the LLM_BACKEND env var. Accepts:
      "auto" | None — probe both, MLX wins ties (per user decision).
      "ollama"      — force Ollama; raise if unreachable.
      "mlx"         — force MLX; raise if unreachable.
    """
    prefer = (prefer or os.environ.get("LLM_BACKEND") or "auto").strip().lower()

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
            endpoint = _mlx_endpoint()
            raise RuntimeError(
                f"LLM_BACKEND=mlx but mlx_lm.server is not reachable at {endpoint}. "
                "Start it with: mlx_lm.server --model <hf-id> --port 8080"
            )
        return info

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
        "No LLM backend reachable. Start either:\n"
        "  • Ollama:           `ollama run <model>`  (port 11434)\n"
        "  • mlx_lm.server:    `mlx_lm.server --model <hf-id> --port 8080`"
    )


def make_backend(info: BackendInfo) -> Backend:
    if info.name == "ollama":
        return OllamaBackend(info)
    if info.name == "mlx":
        return MLXBackend(info)
    raise ValueError(f"unknown backend: {info.name!r}")


# -----------------------------------------------------------------------------
# Ollama detection helpers (extracted from chat.py — same probe chain).
# -----------------------------------------------------------------------------


def _ollama_endpoint() -> str:
    raw = (os.environ.get("OLLAMA_HOST") or "").strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:11434"
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw


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
            )
    # Daemon unreachable — nothing more to do.
    return None


# -----------------------------------------------------------------------------
# MLX detection helpers.
# -----------------------------------------------------------------------------


def _mlx_endpoint() -> str:
    raw = (os.environ.get("MLX_HOST") or "").strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:8080"
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw


def _try_mlx() -> BackendInfo | None:
    """Return a BackendInfo if mlx_lm.server is reachable. None otherwise.

    Resolution order for the model id (per user decision):
      1. MLX_MODEL env var.
      2. `--model X` arg on the running mlx_lm.server process.
      3. /v1/models[0] (only one MLX model in HF cache → unambiguous;
         many → still pick the first, with a clearly-labeled source).
    """
    endpoint = _mlx_endpoint()
    data = _http_get_json(endpoint.rstrip("/") + "/v1/models", timeout=1.0)
    if data is None:
        return None
    available = []
    for m in (data.get("data") or []) if isinstance(data, dict) else []:
        if isinstance(m, dict) and m.get("id"):
            available.append(m["id"])

    env_model = (os.environ.get("MLX_MODEL") or "").strip()
    if env_model:
        return BackendInfo(
            name="mlx", model=env_model,
            source="MLX_MODEL env",
            endpoint=endpoint,
        )

    proc_model = _mlx_process_model_arg()
    if proc_model:
        return BackendInfo(
            name="mlx", model=proc_model,
            source=f"mlx_lm.server --model {proc_model!r}",
            endpoint=endpoint,
        )

    if len(available) == 1:
        return BackendInfo(
            name="mlx", model=available[0],
            source=f"only MLX model in /v1/models: {available[0]!r}",
            endpoint=endpoint,
        )
    if available:
        return BackendInfo(
            name="mlx", model=available[0],
            source=(
                f"first of {len(available)} in /v1/models: {available[0]!r} "
                "(set MLX_MODEL or restart mlx_lm.server with --model to disambiguate)"
            ),
            endpoint=endpoint,
        )
    # Server is up but no models discovered — unusual; surface so caller
    # can decide whether to error or fall back.
    return None


_MLX_PROC_MODEL_RE = re.compile(r"--model[=\s]+(\S+)")


def _mlx_process_model_arg() -> str | None:
    """Read `--model X` from the running mlx_lm.server process command line.

    Mirrors `ollama ps` semantics — we want to know what's *actually*
    running, not just what's in the HF cache. Returns None if no
    mlx_lm.server process is found or if it was launched without
    --model (i.e. loads on first request).
    """
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


# -----------------------------------------------------------------------------
# Convenience listing — used by chat.py /list and similar surfaces.
# -----------------------------------------------------------------------------


def unload_ollama_model(name: str, endpoint: str | None = None) -> tuple[bool, str]:
    """Tell Ollama to evict `name` from VRAM by POSTing keep_alive=0.

    Returns (ok, message). Uses /api/generate with prompt="" — Ollama
    treats keep_alive=0 as "drop this model now". The model stays
    installed on disk; only the VRAM allocation is released.

    Why /api/generate and not /api/chat: /api/chat ignores keep_alive
    when the model isn't already loaded for chat. /api/generate is the
    documented unload path.
    """
    base = (endpoint or _ollama_endpoint()).rstrip("/")
    url = base + "/api/generate"
    payload = json.dumps({"model": name, "prompt": "", "keep_alive": 0}).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"{e!r}"
    return True, f"unloaded {name!r}"


def unload_all_ollama_models(endpoint: str | None = None) -> list[tuple[str, bool, str]]:
    """Walk /api/ps and unload every loaded model. Returns per-model results."""
    base = endpoint or _ollama_endpoint()
    loaded = _ollama_running_models(base)
    return [
        (m["name"], *unload_ollama_model(m["name"], endpoint=base))
        for m in loaded
    ]


def mlx_server_pids() -> list[int]:
    """PIDs of running mlx_lm.server processes — for the /unload mlx hint.

    Returns empty list if `ps` is unavailable or no server is running.
    """
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
    set is returned unfiltered so a currently-running diffuser is still
    visible in status surfaces.
    """
    installed: list[str] = []
    loaded: set[str] = set()
    for base in _ollama_endpoints():
        if not installed:
            installed = _ollama_installed_models(base)
        if not loaded:
            loaded = {m["name"] for m in _ollama_running_models(base)}
        if installed and loaded:
            break
    return [n for n in installed if _is_chat_capable_tag(n)], loaded


def list_mlx_inventory() -> tuple[list[str], str | None]:
    """(downloaded_chat_models, active_model_or_None) — for /list display.

    "Active" = whatever `--model` arg the running server has, falling back
    to the env-set MLX_MODEL when the server was launched without --model.

    Filters non-chat ids (FLUX, Z-Image, embedding models, ...) using the
    same fragment list as the Ollama path. The active model is returned
    as-is even if it would otherwise be filtered, so the user always sees
    what's actually loaded.
    """
    endpoint = _mlx_endpoint()
    data = _http_get_json(endpoint.rstrip("/") + "/v1/models", timeout=1.0)
    if data is None:
        return [], None
    all_ids = [
        m["id"] for m in (data.get("data") or [])
        if isinstance(m, dict) and m.get("id")
    ]
    active = _mlx_process_model_arg() or (os.environ.get("MLX_MODEL") or "").strip() or None
    downloaded = [name for name in all_ids if _is_chat_capable_tag(name)]
    # If the active server is on a "non-chat" id (rare — user explicitly
    # started mlx_lm.server on it), surface it anyway so it's selectable.
    if active and active not in downloaded and active in all_ids:
        downloaded.append(active)
    return downloaded, active


def mlx_endpoint_url() -> str:
    """Public alias for the resolved mlx_lm.server endpoint URL."""
    return _mlx_endpoint()


def ollama_endpoint_url() -> str:
    """Public alias for the resolved Ollama endpoint URL."""
    return _ollama_endpoint()
