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

# Window 3 (BLOCK detector): catches the case where the model duplicates a
# large structured literal (a maze 2D array, a tilemap, a const-table) 3+
# times within one response. The line-based detectors above miss this
# because the rows *inside* one block are individually unique; it's the
# block AS A WHOLE that repeats. We hash an N-line sliding window and
# count appearances. See games/traces (DOOM/FPS sessions) and the live
# maze/broken-pipe failure observed 2026-05-10.
_BLOCK_WINDOW_LINES = 8       # 8 consecutive lines form one "block hash"
_BLOCK_MIN_BYTES = 200        # skip trivially short blocks (e.g. whitespace)
_BLOCK_MAX_REPEATS = 3        # 3 identical blocks within one response → loop

# Window 4 (ADJACENT spam): catches the dead-state-reset block failure
# observed in donkey-kong-game-matching-orig_20260516_142445 iter 1,
# where the model emitted `p.onGirder = false; p.onLadder = false;`
# alternating ~16 times. Windows 1/2 above already catch this but only
# after _REPEAT_MIN_LINES=12 entries. Adjacency is a strictly stronger
# signal — N IDENTICAL consecutive lines means the model is stuck right
# now, not just "the window happens to be low-cardinality." Fires at 4
# entries, ~3× faster than Window 1 for the donkey-kong shape.
_ADJACENT_SPAM_REPEATS = 4    # 4 identical consecutive normalized lines


# Strip numeric SUFFIXES on identifiers so near-duplicate template spam
# collapses (asset_1, asset_2, ...), without erasing standalone numeric
# literals that are normal in real game code (coordinates, dimensions).
# Examples:
#   `{"name":"minimap_compiler179","prompt":"…"},`     →
#   `{"name":"minimap_compiler","prompt":"…"},`
#   `const id_47 = "x";`                               →  `const id_ = "x";`
# but:
#   `{ x1: 0, x2: 800 }` keeps `0` and `800` intact.
import re as _re
_IDENT_SUFFIX_DIGITS_RE = _re.compile(r"(?<=[A-Za-z_])\d+\b")
_DIM_TOKEN_RE = _re.compile(r"\b\d+x\d+\b", _re.IGNORECASE)
_HAS_SIGNAL_RE = _re.compile(r"[A-Za-z_]")


def _normalize_line_for_repeat(s: str) -> str:
    """Bucket near-duplicate lines so a model spamming numbered variants
    of the same template (asset_1, asset_2, …) collapses to one entry."""
    out = _DIM_TOKEN_RE.sub("x", s)
    out = _IDENT_SUFFIX_DIGITS_RE.sub("", out)
    return out.strip()


def _is_repeat_signal_line(s: str) -> bool:
    """True for lines that carry semantic signal.

    Repetition guards should ignore punctuation-only structural closers
    (`});`, `}`, `],`) because those frequently appear in bursts in
    healthy JS output and were causing false loop aborts on one-shot
    long HTML generations.
    """
    if not s:
        return False
    return bool(_HAS_SIGNAL_RE.search(s))


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

    __slots__ = (
        "_line_buf", "_recent_lines", "_recent_lines_norm",
        "_all_lines", "_block_counts", "stall_reason",
        "_adjacent_tail", "loop_line",
    )

    def __init__(self) -> None:
        self._line_buf = ""
        # Window 1: short-line exact-match. Catches the
        # `</body></html>` × 400 case from the missile-command trace.
        self._recent_lines: list[str] = []
        # Window 2: ALL lines, digit-stripped. Catches the
        # numbered-template loop (asset_1, asset_2, …) from the
        # first-person-shooter trace.
        self._recent_lines_norm: list[str] = []
        # Window 3: rolling N-line sliding window for block-level
        # duplication (maze/tilemap repetition). We accumulate all lines
        # and hash the last `_BLOCK_WINDOW_LINES` after each new line.
        self._all_lines: list[str] = []
        self._block_counts: dict[str, int] = {}
        # Window 4: last _ADJACENT_SPAM_REPEATS normalized lines as a
        # tiny ring. When all entries are the same non-empty string the
        # model is stuck emitting one statement on repeat. Strictly
        # tighter than Window 1.
        self._adjacent_tail: list[str] = []
        # When feed() returns True the caller can read this to discriminate
        # ("short_line_loop" / "near_dup_template_loop" / "inline_data_bloat"
        # / "adjacent_line_spam") and customize the recovery message.
        self.stall_reason: str | None = None
        # The actual repeated line (normalized) — surfaced in coaching so
        # the model can see *what* it was looping on.
        self.loop_line: str | None = None

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
            if _is_repeat_signal_line(s) and len(s) <= _REPEAT_LINE_MAX_LEN:
                self._recent_lines.append(s)
                if len(self._recent_lines) > _REPEAT_WINDOW_LINES:
                    self._recent_lines.pop(0)
            norm = _normalize_line_for_repeat(s)
            if norm and _is_repeat_signal_line(norm):
                self._recent_lines_norm.append(norm)
                if len(self._recent_lines_norm) > _REPEAT_WINDOW_LINES:
                    self._recent_lines_norm.pop(0)
                # Window 4 (adjacency): keep just the last N normalized
                # lines. If they're all identical the model is stuck.
                self._adjacent_tail.append(norm)
                if len(self._adjacent_tail) > _ADJACENT_SPAM_REPEATS:
                    self._adjacent_tail.pop(0)
                if (
                    len(self._adjacent_tail) == _ADJACENT_SPAM_REPEATS
                    and len(set(self._adjacent_tail)) == 1
                ):
                    self.stall_reason = "adjacent_line_spam"
                    self.loop_line = self._adjacent_tail[0]
                    return True
            # Window 3 (block-level): hash the trailing N lines as a
            # single block. If we see the same block hash >
            # _BLOCK_MAX_REPEATS times in one response, the model is
            # duplicating a structured literal (maze/tilemap/const-table).
            self._all_lines.append(s)
            if len(self._all_lines) >= _BLOCK_WINDOW_LINES:
                tail = self._all_lines[-_BLOCK_WINDOW_LINES:]
                joined = "\n".join(tail)
                if len(joined) >= _BLOCK_MIN_BYTES:
                    h = hash(joined)
                    self._block_counts[h] = self._block_counts.get(h, 0) + 1
                    if self._block_counts[h] > _BLOCK_MAX_REPEATS:
                        self.stall_reason = "inline_data_bloat"
                        return True
        if (
            len(self._recent_lines) >= _REPEAT_MIN_LINES
            and len(set(self._recent_lines)) <= _REPEAT_MAX_UNIQUE
        ):
            self.stall_reason = "short_line_loop"
            # Most-frequent line in the window — what the model is stuck on.
            from collections import Counter
            self.loop_line = Counter(self._recent_lines).most_common(1)[0][0]
            return True
        if (
            len(self._recent_lines_norm) >= _REPEAT_MIN_LINES
            and len(set(self._recent_lines_norm)) <= _REPEAT_MAX_UNIQUE
        ):
            self.stall_reason = "near_dup_template_loop"
            from collections import Counter
            self.loop_line = Counter(self._recent_lines_norm).most_common(1)[0][0]
            return True
        return False


# A2: smaller LLMs (qwen3.6:27b/35b on MLX) sometimes fall into a
# "deliberation loop" — endless unique paragraphs of "Let me think...
# wait... actually... hmm..." that never emit a recognized output tag.
# RepetitionDetector misses this because each paragraph is unique by
# string; the model isn't repeating, it's stalling on indecision. The
# game-of-space-invaders_20260512_084906 trace shows iter 2 burning
# 14,013 tokens / 13 min in this state.
#
# DeliberationDetector tracks "tokens-since-last-tag-opener" and trips
# when the prefix grows past a budget without producing any of the
# expected output tags. The agent's recovery path treats this as a
# `deliberation_loop` stall reason and injects a coaching user-turn.
#
# Disable for AB testing via DISABLE_DELIBERATION_DETECTOR=1.
# Only the agent's official output tags count as a real "output has
# begun" signal. Earlier versions of this regex also accepted bare
# `<!DOCTYPE html>` and `<html>` literals so the model could deliver
# raw HTML without the `<html_file>` wrapper — but those literals turn
# up constantly inside reasoning prose ("I'd start with `<!DOCTYPE
# html>`...") and falsely latched the detector on the classic-doom
# 20260512_111015 trace, contributing to a 37-min iter-1 disaster.
# Dropping them is a strict improvement: false-no-abort surface shrinks,
# false-abort surface unchanged (the agent's materializer always
# requires `<html_file>` wrapping anyway, so a raw-HTML reply would
# fail to materialize regardless of whether the detector latches).
# ` ```html ` fences are kept because they're how the model legitimately
# delivers code in seed-build paths and rarely appear in reasoning prose.
_TAG_OPENER_RE = _re.compile(
    r"<(?:plan|patch|html_file|diagnose|notes|criteria|probes|assets|sounds|"
    r"done|confirm_done|lookup_bullet)\b|```(?:html|js)?\b",
    _re.IGNORECASE,
)


class DeliberationDetector:
    """Abort a stream that has not produced any output-tag opener after
    `threshold_chars` characters. Disabled if the env var
    DISABLE_DELIBERATION_DETECTOR=1 is set.

    Why character-count rather than token-count: streaming pieces from
    Ollama/MLX vary wildly in size (sometimes one BPE token = 4 chars,
    sometimes a single piece = a whole 60-char line). Char-count gives
    a backend-agnostic budget. Empirically the qwen3.6:27b deliberation
    chains produce ~4–5 chars per piece, so 6000 chars ≈ 1500 pieces.

    <think>-awareness (from classic-doom-style 20260512_111015 trace):
    a reasoning-mode model emits its chain-of-thought inside
    <think>...</think> first. When the CoT mentions output-tag literals
    in prose ("I'd write <!DOCTYPE html>\n<html lang='en'>..."), the
    naive opener regex used to latch on those as if real output had
    begun — and the detector then sat silent while the model burned
    42096 BPE tokens / 37 wall-clock minutes producing zero usable code.
    Fix: track open `<think>` count incrementally; while inside one,
    skip the opener latch AND use a higher per-piece char budget
    (`think_threshold_chars`) before aborting. Outside `<think>`, the
    original 6000-char limit catches pre-tag rambling that isn't even
    wrapped in a reasoning block.
    """

    __slots__ = (
        "_buf", "_total_chars", "_seen_tag", "_threshold", "_think_threshold",
        "_disabled", "_think_depth", "_carry", "stall_reason",
    )

    def __init__(
        self,
        # Item 2, trace build-a-donkey-kong-clone-in-o_20260514_214747:
        # tightened from 6000 → 4000 chars (outside <think>) and
        # 15000 → 8000 chars (inside <think>). The DK trace's iter 1
        # and iter 3 both spent 1000+ lines of <think> reasoning
        # before emitting code — the old 15K think-threshold caught
        # them but only after ~200 lines of pure deliberation. With
        # 8K, abort fires at ~100 lines, saving 5-10 minutes of
        # wall-clock time per stuck iter. False-positive risk on
        # legitimately complex problems is low because the
        # _TAG_OPENER_RE matches the moment ANY output tag begins
        # (including ```html or ```js fences in seed builds), so
        # any model that's actually about to emit code latches
        # before the threshold trips.
        threshold_chars: int = 4000,
        *,
        think_threshold_chars: int = 8000,
    ) -> None:
        import os as _os
        # `_buf` holds a trailing slice for tag-opener regex matching
        # (bounded ≤ 4 KB). `_total_chars` is the cumulative count of
        # chars seen and is what the abort threshold compares against —
        # an earlier bug compared the abort against `len(_buf)`, which
        # is bounded, so any threshold above the trim ceiling never
        # fired. The classic-doom 20260512_111015 trace exposed this:
        # default threshold 6000 vs buf cap 4096 = detector silent.
        self._buf = ""
        self._total_chars = 0
        self._seen_tag = False
        self._threshold = threshold_chars
        self._think_threshold = max(threshold_chars, think_threshold_chars)
        self._disabled = _os.environ.get("DISABLE_DELIBERATION_DETECTOR") == "1"
        # <think>-open count minus </think>-close count. Strictly clamped
        # at 0 below (a stray closing tag without a matching opener
        # shouldn't drop us into "negative depth" where everything looks
        # like real output).
        self._think_depth = 0
        # Last ~15 chars carried across feed() calls so a <think> tag
        # split across pieces still matches (e.g. piece1 ends "<thi",
        # piece2 starts "nk>x"). Longer than the longest tag we count
        # ("</think>" = 8 chars).
        self._carry = ""
        self.stall_reason: str | None = None

    def feed(self, piece: str) -> bool:
        if self._disabled or self._seen_tag:
            # Even when latched on a real output tag, keep updating
            # think_depth so a subsequent block this object reuses
            # stays accurate. But since we early-return, depth-tracking
            # is moot once latched — skip it.
            return False
        # --- update <think> depth incrementally ----------------------
        # Tags may straddle the boundary between two streamed pieces
        # (e.g. piece1 ends "<thi", piece2 starts "nk>"). We keep a
        # ~15-char carry of the prior piece's tail so a boundary-tag
        # gets matched. The trick: any tags already FULLY inside the
        # carry were counted in the *previous* feed() call, so we
        # subtract those out to avoid double-counting.
        window = self._carry + piece
        opens_window = window.count("<think>")
        closes_window = window.count("</think>")
        opens_carry = self._carry.count("<think>")
        closes_carry = self._carry.count("</think>")
        new_opens = opens_window - opens_carry
        new_closes = closes_window - closes_carry
        self._think_depth = max(0, self._think_depth + new_opens - new_closes)
        # Carry must be long enough to span "</think>" (8 chars) boundary
        # split — 15 gives generous safety without bounded growth.
        self._carry = window[-15:]
        # --- buffer trimming (memory bound) + cumulative counter -----
        self._buf += piece
        self._total_chars += len(piece)
        if len(self._buf) > 4096:
            self._buf = self._buf[-2048:]
        # --- latch check --------------------------------------------
        # Only count opener literals when we're outside <think>. Inside
        # <think> they're reasoning prose ("the spec says <!DOCTYPE html>
        # is required") rather than real output.
        if self._think_depth == 0 and _TAG_OPENER_RE.search(self._buf):
            self._seen_tag = True
            return False
        # --- abort check --------------------------------------------
        # Compare against cumulative chars, NOT buffer size — the buf
        # is trimmed at 4 KB and would never reach the higher
        # inside-think threshold otherwise.
        limit = self._think_threshold if self._think_depth > 0 else self._threshold
        if self._total_chars >= limit:
            self.stall_reason = "deliberation_loop"
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
    # True when the backend raised an exception mid-generation
    # (in-process MLX worker, cloud API error, etc.). Distinct from
    # `stalled` so the agent can surface a specific recovery path.
    # Pair with `error_message` for the actual cause — without it the
    # agent has to guess (and the guess used to be wrong; see
    # 2026-05-15 DK trace where a hardcoded "mlx_lm.server crashed"
    # message survived the move to in-process MLX).
    crashed: bool = False
    # Real exception text from the backend when `crashed=True`.
    # Captured via `traceback.format_exception_only(type(e), e)` so it
    # works for BaseException subclasses (MemoryError, KeyboardInterrupt,
    # Metal RuntimeError, ...). None on a clean finish.
    error_message: str | None = None
    # A2: model produced `deliberation_threshold` chars of pure reasoning
    # with no output tag (no <plan>, <patch>, <html_file>, ```html, etc).
    # Folded into `stalled` for backward compatibility; standalone field
    # lets the agent route to the coaching prompt instead of generic retry.
    deliberated: bool = False
    # Cloud-only signal: the API capped the response at max_tokens
    # (Anthropic stop_reason="max_tokens", OpenAI finish_reason="length").
    # Distinct from a model that finished naturally; the model was cut
    # off mid-emission and would have continued. Lets the agent route
    # to a "your reply was capped, emit a smaller change" coach instead
    # of treating it as a generic truncation. False on Ollama/MLX.
    max_tokens_hit: bool = False
    # When `looped=True`, which RepetitionDetector window fired:
    # "short_line_loop" / "near_dup_template_loop" / "inline_data_bloat"
    # / "adjacent_line_spam". Used by the recovery coach in agent.py to
    # name the failure shape back to the model.
    loop_kind: str | None = None
    # When `looped=True`, the actual line the model was repeating
    # (normalized). Surfaces in coaching so the model sees *what* it
    # was stuck on instead of just "your reply was aborted." None when
    # the detector didn't capture one (older code paths).
    loop_line: str | None = None


async def stream_chat(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    stall_seconds: float = 600.0,
    overall_seconds: float = 1800.0,
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
    # A2: shared deliberation detector. Aborts streams that produce
    # only reasoning paragraphs with no output tag for too long.
    delib = DeliberationDetector()
    deliberated = False

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

            # ---- deliberation detector (A2) -------------------------
            if delib.feed(piece):
                deliberated = True
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
        # `stalled` covers stall, repetition loop, AND deliberation loop
        # for back-compat with callers that only check `.stalled`. Each
        # specific cause is also exposed as its own boolean so the agent
        # can route to a tailored recovery message.
        stalled=stalled or looped or deliberated,
        stall_at_token=stall_at,
        looped=looped,
        deliberated=deliberated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        loop_kind=repeat.stall_reason if looped else None,
        loop_line=repeat.loop_line if looped else None,
    )


async def stream_chat_with_retry(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    stall_seconds: float = 600.0,
    overall_seconds: float = 1800.0,
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
    stall_seconds: float = 600.0,
    overall_seconds: float = 1800.0,
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
