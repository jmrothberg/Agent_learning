"""Low-level Ollama streaming helpers with a stall watchdog and best-of-N.

Why a separate module: the previous version of the agent embedded a raw
`async for chunk in self._client.chat(stream=True)` directly in the loop. That
call has no timeout — if Ollama silently stops yielding tokens (which we saw
with `gpt-oss:latest` at iteration 2 once the conversation grew past
num_ctx), the agent freezes forever. There is no exception, no log, no exit.

What this module gives us:

  * `stream_chat()` — same shape as the old call, but every awaited chunk has
    a per-chunk inactivity timeout AND an overall deadline. A stall raises
    `StreamStalled` so the caller can recover or abort cleanly.

  * `stream_chat_collect()` — convenience wrapper that returns the full text
    plus token rate, used when we don't care about per-token streaming
    (e.g. background best-of-N samples).

  * `best_of_n()` — fan out N chat requests in parallel, score each result
    with a caller-supplied scorer (typically: run the produced HTML in a
    headless browser, score = 1 if test passes, else partial), return the
    winner. With local models this is cheap and is the single biggest
    quality lever we have for weak models.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import ollama


class StreamStalled(RuntimeError):
    """Raised when the model goes silent for longer than `stall_seconds`.

    The exception message includes how many tokens we got before the stall and
    how long we waited. The caller is expected to log this, abandon the
    response, and try again with a smaller context or a backoff.
    """


@dataclass
class StreamResult:
    """What stream_chat returns when it completes (or stalls)."""

    text: str           # full assembled assistant text (may be partial on stall)
    tokens: int         # rough chunk count (not BPE tokens)
    duration_s: float   # wall-clock elapsed on the streaming call
    stalled: bool       # True if we returned because of a stall
    stall_at_token: int | None = None  # which token index we stalled on


async def stream_chat(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    stall_seconds: float = 45.0,
    overall_seconds: float = 600.0,
) -> StreamResult:
    """Stream a chat completion with a stall watchdog.

    `stall_seconds` is the per-chunk inactivity budget — if no token arrives
    for that long we stop waiting. `overall_seconds` is the absolute ceiling
    (a long but slow stream is still a problem if it never ends).

    On a clean finish we return the full text and stalled=False. On stall we
    return what we collected so far AND set stalled=True so the caller can
    decide to use the partial result, retry, or escalate.
    """
    options = dict(options or {})

    started = time.monotonic()
    parts: list[str] = []
    n_tokens = 0
    stalled = False
    stall_at: int | None = None

    # ollama.AsyncClient.chat returns an async iterator of dicts. We pull
    # .__aiter__() so we can wrap each .__anext__() in asyncio.wait_for.
    stream = await client.chat(
        model=model, messages=messages, stream=True, options=options
    )
    ait = stream.__aiter__()

    try:
        while True:
            # Per-chunk inactivity timeout. This is the watchdog.
            try:
                chunk = await asyncio.wait_for(ait.__anext__(), timeout=stall_seconds)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                stalled = True
                stall_at = n_tokens
                break

            # Overall deadline check (cheap; keeps stuck-at-trickle bounded).
            if time.monotonic() - started > overall_seconds:
                stalled = True
                stall_at = n_tokens
                break

            piece = chunk.get("message", {}).get("content", "") or ""
            if not piece:
                continue
            parts.append(piece)
            n_tokens += 1
            if on_token is not None:
                try:
                    on_token(piece)
                except Exception:
                    # A misbehaving UI callback must never kill the stream.
                    pass
    finally:
        # Best-effort close in both branches. Ollama's AsyncStream exposes
        # .aclose() in newer versions; older versions don't, hence the guard.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

    return StreamResult(
        text="".join(parts),
        tokens=n_tokens,
        duration_s=time.monotonic() - started,
        stalled=stalled,
        stall_at_token=stall_at,
    )


async def stream_chat_with_retry(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    stall_seconds: float = 45.0,
    overall_seconds: float = 600.0,
    max_retries: int = 1,
    on_stall: Callable[[StreamResult, int], None] | None = None,
) -> StreamResult:
    """Wrapper that retries on stall, optionally calling `on_stall` for logging.

    Backoff is fixed at 2s — the failure mode we're guarding against is the
    daemon getting wedged on a single huge prompt; sleeping briefly and
    retrying with the SAME prompt rarely helps, but the caller may have
    pruned the conversation between attempts.
    """
    last: StreamResult | None = None
    for attempt in range(max_retries + 1):
        result = await stream_chat(
            client,
            model,
            messages,
            on_token,
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


# -----------------------------------------------------------------------------
# Best-of-N: fan out N samples, pick the winner.
# -----------------------------------------------------------------------------


@dataclass
class Candidate:
    """One sampled completion + its score from the user-supplied scorer."""

    text: str
    score: float
    extra: dict[str, Any]   # whatever the scorer wants to attach (test report, etc)
    tokens: int
    duration_s: float
    stalled: bool


async def best_of_n(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    *,
    n: int = 3,
    temperatures: Iterable[float] | None = None,
    options: dict[str, Any] | None = None,
    stall_seconds: float = 45.0,
    overall_seconds: float = 600.0,
    scorer: Callable[[str], Awaitable[tuple[float, dict]]],
    on_progress: Callable[[int, str], None] | None = None,
) -> tuple[Candidate, list[Candidate]]:
    """Sample N completions in parallel, score each, return (best, all_sorted).

    `temperatures`: per-sample temps. If None we use a default spread that
    keeps one near-greedy sample (precision) and a couple of explorers.
    `scorer(text) -> (score, extra)` is awaited concurrently with the next
    sample's generation when possible.

    Returns ((winner, all_candidates_sorted_high_to_low). Caller uses winner
    for the next agent step; the rest are typically traced for audit.
    """
    if temperatures is None:
        # 0.2 = greedy-ish (best for precision fixes), 0.6/0.9 = exploration.
        # We trim the list to length n so callers can shrink it.
        temperatures = [0.2, 0.6, 0.9][:n]
    temps = list(temperatures)
    if len(temps) < n:
        # Pad by reusing the last temperature.
        temps += [temps[-1]] * (n - len(temps))

    base_options = dict(options or {})

    async def one(i: int, t: float) -> Candidate:
        opts = dict(base_options)
        opts["temperature"] = t
        if on_progress is not None:
            on_progress(i, f"start (T={t})")
        result = await stream_chat(
            client,
            model,
            messages,
            on_token=None,  # silent; we'll only show the winner
            options=opts,
            stall_seconds=stall_seconds,
            overall_seconds=overall_seconds,
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
        return Candidate(
            text=result.text,
            score=score,
            extra=extra,
            tokens=result.tokens,
            duration_s=result.duration_s,
            stalled=result.stalled,
        )

    # Generation can run in parallel; scoring (browser-test) is also async, so
    # asyncio.gather lets the browser pool all candidates concurrently. The
    # one shared resource is the Ollama daemon itself — most local servers
    # serialize generation requests anyway, so "parallel" here means we
    # overlap generation with scoring of earlier samples, not pure parallel
    # token production.
    cands = await asyncio.gather(*(one(i, t) for i, t in enumerate(temps)))
    cands.sort(key=lambda c: (c.score, -c.duration_s), reverse=True)
    return cands[0], cands
