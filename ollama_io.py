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


# Mid-stream repetition detector tunables. Local LLMs (qwen3.6, gpt-oss)
# occasionally enter a "looping" state where they emit the same lines
# forever. Two distinct failure shapes we have to catch:
#
#   (A) SHORT-LINE LOOP — `</body></html>\n</html_file>\n` × 400. See
#       games/traces/missile-command_20260505_224321.log. Tight repetition,
#       short lines, easy to spot: the unique-count over the window
#       collapses to 1-2 in a few seconds.
#
#   (B) NEAR-DUPLICATE LONG-LINE LOOP — 200+ lines of
#       `{"name":"minimap_compiler<N>","prompt":"…","size":"16x16"},`
#       where every line differs only in a small numeric suffix. See
#       games/traces/a-first-person-shooter_20260509_134453.log: the
#       model burned 9906 tokens / 25 minutes before the stall watchdog
#       finally fired. Short-line detector misses these because each line
#       is ~155 chars and unique by exact string. We catch them with a
#       NORMALIZED hash that strips trailing digits / whitespace so the
#       per-line numeric variant collapses to one bucket.
#
# A real long file (Space Invaders, Asteroids) emits thousands of unique
# lines even after normalization, so 25+ unique values in a 30-line
# window is the floor a healthy stream comfortably clears.
_REPEAT_WINDOW_LINES = 30
_REPEAT_MIN_LINES = 12      # need this many lines before we even consider
_REPEAT_MAX_UNIQUE = 2      # window collapses to ≤2 unique values = looping
_REPEAT_LINE_MAX_LEN = 80   # short-line detector only watches lines ≤ 80 chars
                            # (long-line variants are caught by the
                            # normalized detector below)


# Strip trailing digits / underscored numeric suffixes / whitespace from a
# line so near-duplicates collapse to a single bucket. Examples:
#   `{"name":"minimap_compiler179","prompt":"…"},`     →
#   `{"name":"minimap_compiler","prompt":"…"},`
#   `      'p_47',`                                    →  `'p',`
#   `<patch>id="enemy42">`                             →  `<patch>id="enemy">`
# We only collapse digit runs; alphabetic content stays exact so legitimate
# lines like `const score = 0;` and `const lives = 0;` are still distinct.
import re as _re
_DIGIT_RUN_RE = _re.compile(r"\d+")


def _normalize_line_for_repeat(s: str) -> str:
    """Bucket near-duplicate lines so a model spamming numbered variants
    of the same template (asset_1, asset_2, …) collapses to one entry."""
    return _DIGIT_RUN_RE.sub("", s).strip()


class RepetitionDetector:
    """Streaming repetition detector shared by both backends.

    Why a class instead of inline loops in each backend: the detection
    logic was duplicated across `ollama_io.stream_chat` (Ollama) and
    `backend.MLXBackend._stream_once` (MLX). Two copies = two places to
    keep in sync the next time we tune thresholds or add a third
    detector. Whichever backend the user happens to be running, the
    behavior is now byte-for-byte identical.

    Usage:
        detector = RepetitionDetector()
        for piece in stream:
            if detector.feed(piece):
                # model is looping; abort
                break

    State is per-instance — construct one per stream. The two windows
    use the same `_REPEAT_*` thresholds defined at module top.
    """

    __slots__ = ("_line_buf", "_recent_lines", "_recent_lines_norm")

    def __init__(self) -> None:
        self._line_buf = ""
        # Window 1: short-line exact-match. Catches the
        # `</body></html>` × 400 case from the missile-command trace.
        self._recent_lines: list[str] = []
        # Window 2: ALL lines, digit-stripped. Catches the
        # numbered-template loop (asset_1, asset_2, …) from the
        # first-person-shooter trace.
        self._recent_lines_norm: list[str] = []

    def feed(self, piece: str) -> bool:
        """Append `piece` to the internal line buffer; on every newline,
        update both windows and check thresholds. Returns True when
        either window's unique-count has collapsed to ≤
        `_REPEAT_MAX_UNIQUE` after at least `_REPEAT_MIN_LINES` entries
        — i.e., the model is looping and the caller should abort.
        """
        self._line_buf += piece
        if "\n" not in self._line_buf:
            return False
        *complete, self._line_buf = self._line_buf.split("\n")
        for ln in complete:
            s = ln.strip()
            if not s:
                continue
            if len(s) <= _REPEAT_LINE_MAX_LEN:
                self._recent_lines.append(s)
                if len(self._recent_lines) > _REPEAT_WINDOW_LINES:
                    self._recent_lines.pop(0)
            norm = _normalize_line_for_repeat(s)
            if norm:
                self._recent_lines_norm.append(norm)
                if len(self._recent_lines_norm) > _REPEAT_WINDOW_LINES:
                    self._recent_lines_norm.pop(0)
        if (
            len(self._recent_lines) >= _REPEAT_MIN_LINES
            and len(set(self._recent_lines)) <= _REPEAT_MAX_UNIQUE
        ):
            return True
        if (
            len(self._recent_lines_norm) >= _REPEAT_MIN_LINES
            and len(set(self._recent_lines_norm)) <= _REPEAT_MAX_UNIQUE
        ):
            return True
        return False


@dataclass
class StreamResult:
    """What stream_chat returns when it completes (or stalls)."""

    text: str           # full assembled assistant text (may be partial on stall)
    tokens: int         # rough chunk count (not BPE tokens)
    duration_s: float   # wall-clock elapsed on the streaming call
    stalled: bool       # True if we returned because of a stall
    stall_at_token: int | None = None  # which token index we stalled on
    # True when we aborted because the model entered a repetition loop
    # (distinct from a true stall). Caller can log this differently.
    looped: bool = False
    # BPE prompt tokens, captured from the backend's final chunk (Ollama:
    # `prompt_eval_count`; MLX: `usage.prompt_tokens`). None if the backend
    # didn't surface it (e.g. early stall before the final frame). This is
    # the only way to see how much we're paying for the inlined-file truth
    # source on every fix turn.
    prompt_tokens: int | None = None
    # BPE completion tokens from the backend (vs. `tokens`, which counts
    # streaming chunks). Often differs from chunk count — kept distinct so
    # the chunk count stays meaningful for stall diagnostics.
    completion_tokens: int | None = None


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
    looped = False
    stall_at: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Shared repetition detector used by both backends — see
    # `RepetitionDetector` for the two-window strategy and rationale.
    repeat = RepetitionDetector()

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

            # Final Ollama chunk carries token counts in `prompt_eval_count`
            # and `eval_count` plus done=True. The piece is usually empty
            # on that frame, so capture before the empty-piece skip below.
            if chunk.get("done"):
                pec = chunk.get("prompt_eval_count")
                ec = chunk.get("eval_count")
                if isinstance(pec, int):
                    prompt_tokens = pec
                if isinstance(ec, int):
                    completion_tokens = ec

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

            # ---- repetition detector --------------------------------
            if repeat.feed(piece):
                looped = True
                stall_at = n_tokens
                break
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
        # `stalled` covers both the original stall semantics AND a
        # repetition loop, so existing callers that only check `.stalled`
        # still abort correctly. `looped` lets callers distinguish.
        stalled=stalled or looped,
        stall_at_token=stall_at,
        looped=looped,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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
    early_exit_score: float = 1.0,
) -> tuple[Candidate, list[Candidate]]:
    """Sample candidates SEQUENTIALLY with early exit, score each, return winner.

    Originally we ran candidates in parallel via asyncio.gather, but local
    Ollama serializes generation requests at the daemon level — so "parallel"
    just queued the second request behind the first AND tripped the stall
    watchdog while the second candidate sat waiting. Sequential is faster
    in wall time AND correct.

    Early exit: as soon as a candidate scores at least `early_exit_score`
    (default 1.0 = passes the test), we stop sampling. So when the first
    sample is good, best-of-N costs roughly the same as best-of-1.
    """
    if temperatures is None:
        # Sample 0 = near-greedy (precision). Subsequent samples explore.
        # Anything beyond 3 reuses the last temp.
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

        result = await stream_chat(
            client,
            model,
            messages,
            on_token=None,
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
