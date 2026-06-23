"""Low-level Ollama streaming helpers with a stall watchdog and best-of-N.

Why a separate module: the previous version of the agent embedded a raw
`async for chunk in self._client.chat(stream=True)` directly in the loop. That
call has no timeout — if Ollama silently stops yielding tokens (which we saw
with `gpt-oss:latest` at iteration 2 once the conversation grew past
num_ctx), the agent freezes forever. There is no exception, no log, no exit.

What this module gives us:

  * `stream_chat()` — same shape as the old call, but every awaited chunk has
    a per-chunk inactivity timeout. Slow-but-active streams are allowed to
    finish; repetition and deliberation detectors handle runaway output.

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
_BLOCK_MIN_BYTES = 100        # skip trivially short blocks (e.g. whitespace)
_BLOCK_MAX_REPEATS = 3        # 3 identical blocks within one response → loop
# Lowered from 200 → 100 once `;` joined `\n` as a statement boundary:
# semicolon-split lines are roughly half the length of newline-split
# lines, so an 8-line repeating cycle that's pathological now joins to
# ~150 bytes instead of ~300. 100 still excludes pure-whitespace blocks
# (8 lines × 12 char floor) while letting template-rotation loops with
# multiple distinct RHS values (e.g. const X=440; const Y=48; const
# Y=140;) be caught by block-level repetition.

# Window 4 (ADJACENT spam): catches the dead-state-reset block failure
# observed in donkey-kong-game-matching-orig_20260516_142445 iter 1,
# where the model emitted `p.onGirder = false; p.onLadder = false;`
# alternating ~16 times. Windows 1/2 above already catch this but only
# after _REPEAT_MIN_LINES=12 entries. Adjacency is a strictly stronger
# signal — N IDENTICAL consecutive lines means the model is stuck right
# now, not just "the window happens to be low-cardinality." Fires at 4
# entries, ~3× faster than Window 1 for the donkey-kong shape.
_ADJACENT_SPAM_REPEATS = 4    # 4 identical consecutive normalized lines

# Window 5 (INTRA-LINE repetition): Windows 1-4 all key off completed
# lines (split on \n / ;). A stream that NEVER emits a boundary —
# e.g. a prose enumeration "a menuLoop, a menuLoopStart, a
# menuLoopStartIndex, …" that degenerates into "StartStartStart…" — has
# no \n and no ; ever, so feed()'s early return kept buffering forever
# and the loop ran to ~80k tokens completely unseen (the architect turn
# in a-animateed-fighing… 20260530). Prose turns (plan/architect/
# diagnose) are exactly where boundary-free loops live. This window
# inspects the unterminated buffer's TAIL for a short unit repeated many
# times consecutively. Length is only the gate to START checking; the
# repeat is what trips, so a long HEALTHY single line (minified HTML,
# base64 data URI — high entropy, no 40× consecutive repeat) never fires.
_INTRA_LINE_MIN_CHARS = 1200   # only scan once the unterminated line is this long
_INTRA_LINE_SCAN_STEP = 200    # re-scan every +200 chars (cheap amortization)
_INTRA_LINE_MAX_BUFFER = 4000  # trim buffer to this trailing slice (bound memory)
_INTRA_LINE_MAX_UNIT = 40      # repeated unit must be ≤ this many chars
_INTRA_LINE_MIN_REPEATS = 40   # … repeated ≥ this many times consecutively
_INTRA_LINE_TAIL = 2400        # only inspect this many trailing chars


def _degenerate_intra_line_unit(buf: str) -> str | None:
    """Return the repeated unit if the TAIL of `buf` is one short substring
    repeated ≥ `_INTRA_LINE_MIN_REPEATS` times back-to-back, else None.

    Tail-only so a long healthy prefix can't dilute the signal; a unit of
    length L up to `_INTRA_LINE_MAX_UNIT` also covers small 2-3 token cycles
    (e.g. 'ab'*N collapses to unit 'ab'). Whitespace-only units are ignored.
    """
    s = buf[-_INTRA_LINE_TAIL:]
    n = len(s)
    for L in range(1, _INTRA_LINE_MAX_UNIT + 1):
        if n < L * _INTRA_LINE_MIN_REPEATS:
            break
        unit = s[-L:]
        if not unit.strip():
            continue
        reps = 0
        i = n
        while i >= L and s[i - L:i] == unit:
            reps += 1
            i -= L
        if reps >= _INTRA_LINE_MIN_REPEATS:
            return unit
    return None


# Strip numeric runs INSIDE identifiers (suffix OR mid-identifier) so
# near-duplicate template spam collapses regardless of where the counter
# sits, without erasing standalone numeric literals that are normal in
# real game code (coordinates, dimensions).
# Examples:
#   `{"name":"minimap_compiler179","prompt":"…"},`     →
#   `{"name":"minimap_compiler","prompt":"…"},`
#   `const id_47 = "x";`                               →  `const id_ = "x";`
#   `const LADDER784_Y2 = 140;`                        →  `const LADDER_Y = 140;`
# but:
#   `{ x1: 0, x2: 800 }` keeps `0` and `800` intact (lookbehind only
#   fires when the digit run follows a letter/underscore, not whitespace
#   or punctuation).
#
# The original `\b` anchor at the tail required the digit run to END at
# a word boundary, which `_` (a word char) breaks — so `LADDER784_Y2`
# kept `784` because `4` is followed by `_`. The current pattern omits
# the trailing anchor so mid-identifier digit runs are also stripped,
# letting the donkey-kong-style `const LADDERnnn_Yn=v;…` loops collapse
# to one template.
import re as _re
_IDENT_SUFFIX_DIGITS_RE = _re.compile(r"(?<=[A-Za-z_])\d+")
_DIM_TOKEN_RE = _re.compile(r"\b\d+x\d+\b", _re.IGNORECASE)
_HAS_SIGNAL_RE = _re.compile(r"[A-Za-z_]")
# Logical-unit boundaries inside the streaming line buffer. Newlines are
# the obvious separator, but a model that fires off a long
# `const A1=…;const A2=…;…` chain on ONE line never produces \n, so the
# detector previously never flushed and missed the loop entirely. `;`
# is the canonical statement terminator in every C-family language the
# coder emits (JS, CSS-in-JS, JSON-with-comments), so treating it as a
# co-equal boundary makes single-line statement spam visible to the
# repetition windows without changing behavior for healthy code (where
# `;`-chained statements differ from each other after normalization and
# stay below `_REPEAT_MAX_UNIQUE`).
_STATEMENT_BOUNDARY_RE = _re.compile(r"[\n;]")


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


def _in_unclosed_html_file_block(text: str) -> bool:
    """True when streamed text has opened `<html_file>` but not closed it.

    Used by the one-shot inline-data grace path: long first-build HTML
    streams can legitimately include large repeated-looking tables/lists
    before finally closing `</html_file>`. We grant one continuation before
    aborting so normal long emissions are less likely to be cut mid-output.
    """
    if not text:
        return False
    lower = text.lower()
    last_open = lower.rfind("<html_file>")
    if last_open < 0:
        return False
    last_close = lower.rfind("</html_file>")
    return last_close < last_open


# Completion-token ceiling above which the one-shot first-build grace is
# DENIED. The grace exists so a legitimately long first-build <html_file>
# (big data tables / level maps) isn't cut mid-output by the repetition
# detector. But past this many tokens the stream is already far longer than
# any clean single-file game (observed clean builds finish well under ~12k
# tokens), so a detector trip here is almost certainly a real repetition loop
# — the model concatenating multiple drafts. Trace pin: the centipede build
# 20260615_154952 looped to 22k tokens / 26 minutes on this slow model. This
# is NOT a blanket cutoff: it only fires when the detector has ALREADY flagged
# a loop AND we are past the ceiling, so legitimate builds under the ceiling
# are untouched.
_LOOP_GRACE_TOKEN_CEILING = 18000


def _should_grace_inline_data_bloat(
    *,
    stall_reason: str | None,
    assembled_text: str,
    grace_already_used: bool,
    completion_tokens: int = 0,
) -> bool:
    """One-shot grace gate for `inline_data_bloat` repetition aborts.

    `completion_tokens` is the running count of emitted tokens for this
    stream; past `_LOOP_GRACE_TOKEN_CEILING` the grace is denied so a detected
    loop aborts immediately instead of getting another full detection window.
    """
    if completion_tokens >= _LOOP_GRACE_TOKEN_CEILING:
        return False
    return (
        not grace_already_used
        and stall_reason == "inline_data_bloat"
        and _in_unclosed_html_file_block(assembled_text)
    )


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
        "_adjacent_tail", "loop_line", "_intra_scan_at",
    )

    def __init__(self) -> None:
        self._line_buf = ""
        # Window 5 — last buffer length at which we ran the intra-line scan,
        # so we only re-scan every _INTRA_LINE_SCAN_STEP chars (not per token).
        self._intra_scan_at = 0
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
        if "\n" not in self._line_buf and ";" not in self._line_buf:
            # Window 5 — no statement boundary yet. A normal stream flushes
            # here harmlessly, but a boundary-free degenerate loop (prose
            # enumeration spiralling into "StartStart…") would grow forever
            # and escape every line-based window. Scan the unterminated tail.
            buf_len = len(self._line_buf)
            if (
                buf_len >= _INTRA_LINE_MIN_CHARS
                and buf_len - self._intra_scan_at >= _INTRA_LINE_SCAN_STEP
            ):
                self._intra_scan_at = buf_len
                unit = _degenerate_intra_line_unit(self._line_buf)
                if unit is not None:
                    self.stall_reason = "intra_line_repetition"
                    self.loop_line = unit
                    return True
                # Not degenerate — bound memory so a long healthy line
                # (minified HTML / base64) can't balloon the buffer.
                if buf_len > _INTRA_LINE_MAX_BUFFER:
                    self._line_buf = self._line_buf[-_INTRA_LINE_MAX_BUFFER:]
                    self._intra_scan_at = len(self._line_buf)
            return False
        *complete, self._line_buf = _STATEMENT_BOUNDARY_RE.split(self._line_buf)
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
            # Hash NORMALIZED lines (digit-stripped templates) rather
            # than raw text, so a rotating cycle of K>2 unique templates
            # (which Window 2's ≤2-unique threshold misses) is still
            # caught when the same K-cycle of templates repeats across
            # the rolling window — see donkey-kong 20260523_091509 where
            # `_X=440;_Y1=48;_Y2=140;` rotated for 2588 tokens without
            # tripping any line-unique check because the rotation has 3
            # distinct templates.
            self._all_lines.append(norm if norm else s)
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
# Latch when the model has clearly started producing real output. The
# previous version required line-start (`^` or `\n`) to avoid latching
# on inline prose mentions like "I need to emit `<html_file>`", but that
# was over-strict: it killed real long first-build streams whose chunking
# happened to put the opener mid-buffer after a buffer trim. Wolfenstein
# 2026-05-25 trace [04] is the canonical case — `<html_file>` was at
# position 0 of the visible reply but the detector still fired at 6001
# chars.
#
# Two latch families now:
#   _TAG_OPENER_RE  — the canonical agent output tags. Match anywhere
#                     in the buffer; an honest mention of one of these
#                     means the model is engaged with the task, not
#                     rambling in pre-tag deliberation.
#   _CODE_OPENER_RE — substantive HTML/JS/CSS produced text. Once the
#                     model has emitted `<!DOCTYPE html>`, `<script>`,
#                     `<canvas`, a `function` declaration, etc, it is
#                     past deliberation and writing code. Length stops
#                     being a useful fail signal.
_TAG_OPENER_RE = _re.compile(
    # Output tag — anywhere in buffer, but NOT preceded by a backtick.
    # `(?<!`)` excludes inline markdown code spans like
    # `` `<html_file>` `` where the model is referring to the tag in
    # prose rather than emitting it (doom 2026-05-17 trace protection).
    # A bare `<html_file>` after non-backtick prose (Wolfenstein
    # 2026-05-25 trace) still latches.
    r"(?<!`)"
    r"<(?:plan|patch|html_file|diagnose|notes|criteria|probes|assets|sounds|"
    r"done|confirm_done|lookup_bullet)\b"
    r"|```(?:html|js)?\b",
    _re.IGNORECASE,
)
# HTML structural openers — case-insensitive (model may write
# `<!doctype HTML>` or `<HTML>`).
_HTML_OPENER_RE = _re.compile(
    r"<!DOCTYPE\s+html\b"
    r"|<html\b"
    r"|<script\b"
    r"|<style\b"
    r"|<canvas\b"
    r"|<body\b",
    _re.IGNORECASE,
)
# JavaScript code openers — case-sensitive. `let`/`const`/`var`/`function`/
# `class` are reserved words and only valid as code when lowercase. The
# patterns also require code-shaped context (identifier or `=` next) so
# prose like "Let me think" or "function of the game" doesn't latch.
# No IGNORECASE flag — case-sensitive on purpose.
_JS_OPENER_RE = _re.compile(
    r"\bfunction\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\("
    r"|\bclass\s+[A-Za-z_$][A-Za-z0-9_$]*\s*[{<]"
    r"|\b(?:const|let|var)\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*="
)
# Broader "the model is writing real code" signals (minecraft trace
# 20260621_182845): a model can plan a build in bullet prose that is
# clearly code — `world = {}`, `new Uint8Array(...)`, arrow functions —
# without ever using the function/const/class keywords above. The 6000-
# char fallback then aborted a stream that was WORKING. Per the standing
# rule (never abort a working stream; only abort genuine off-the-rails
# repetition), these high-precision JS shapes also latch the detector.
# Kept tight so ordinary English does not match: arrow `=>`, a `new X(`
# constructor call, or an assignment to an object/array literal.
_CODE_DISCUSSION_RE = _re.compile(
    r"=>"
    r"|\bnew\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\("
    r"|[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*[\{\[]"
)


class DeliberationDetector:
    """Abort a stream that is rambling in pre-tag deliberation —
    NOT a stream that is doing work.

    The detector fires only when, after `threshold_chars` characters,
    NEITHER a canonical agent output tag (<html_file>, <patch>, etc)
    NOR substantive code content (<!DOCTYPE html>, <script>, <canvas,
    `function foo`, class declarations, top-level const/let/var) has
    appeared outside any `<think>` block. Once any of those land, the
    detector latches and length stops being a fail signal — the model
    can stream a 20 KB+ first build to completion without interruption.

    Wolfenstein 2026-05-25 lesson: an earlier version required the
    opener regex to match at line-start (`(?:^|\\n)\\s*<html_file\\b`)
    and only matched the agent-format tags. That killed real first
    builds whose chunking happened to put the opener mid-buffer after
    a 4 KB trim. Current version (a) drops the line-start anchor —
    inline `<html_file>` is still a real signal that the model is
    engaged with the task — and (b) adds a second latch family for
    raw HTML/JS/CSS so a model that goes straight into `<!DOCTYPE>`
    without wrapping in `<html_file>` still latches.

    Disabled if env var DISABLE_DELIBERATION_DETECTOR=1 is set.

    Why character-count rather than token-count: streaming pieces from
    Ollama/MLX vary wildly in size (sometimes one BPE token = 4 chars,
    sometimes a single piece = a whole 60-char line). Char-count gives
    a backend-agnostic budget for the pre-tag rambling case.

    <think>-awareness (from classic-doom-style 20260512_111015 trace):
    a reasoning-mode model emits its chain-of-thought inside
    <think>...</think> first. Opener literals INSIDE a think block are
    reasoning prose ("the spec says <!DOCTYPE html> is required") and
    don't latch — otherwise the detector would sleep while the model
    burned tens of thousands of tokens producing zero real output.
    Higher `think_threshold_chars` lets reasoning take more room before
    we call it a deliberation loop. Outside `<think>`, the regular
    `threshold_chars` catches pre-tag rambling that isn't even wrapped
    in a reasoning block.
    """

    __slots__ = (
        "_buf", "_total_chars", "_seen_tag", "_threshold", "_think_threshold",
        "_disabled", "_think_depth", "_carry", "stall_reason",
    )

    def __init__(
        self,
        # History: 4000/8000 → 6000/12000 (donkey-kong 20260516_170758,
        # premature cutoffs on valid long transitions into code).
        # 2026-06-21 minecraft trace 20260621_182845: a GLM first build
        # streamed 6000 chars of genuine planning (chunk sizes, block
        # IDs, BufferGeometry) WITHOUT a <think> wrapper, so it got the
        # tight 6000 plain budget and was aborted mid-plan — a working
        # stream cut off. Per the standing rule "never abort a working
        # stream; only abort genuine off-the-rails repetition", the plain
        # (outside-<think>) budget now matches the inside-<think> budget:
        # a model that plans in prose is given the same room as one that
        # plans in a reasoning channel. The repetition detector remains
        # the real-time off-the-rails guard; this is only a far backstop
        # for a pure-prose stream that never produces a single code shape.
        threshold_chars: int = 12000,
        *,
        think_threshold_chars: int = 12000,
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
        # Latch on either: (a) a canonical agent output tag (<html_file>,
        # <patch>, <plan>, etc), or (b) substantive HTML/JS/CSS that
        # signals the model is past deliberation and into producing real
        # code (<!DOCTYPE html>, <script>, function foo, ...). Either is
        # sufficient evidence the stream is doing work; length should
        # stop being a fail signal.
        #
        # Only outside <think>: inside the thinking channel these
        # literals are reasoning prose ("the spec says <!DOCTYPE html>
        # is required"), not real output. The original doom 20260512
        # trace failure was exactly this — model emitted opener literals
        # in thinking that latched falsely, then never produced real
        # output. After </think>, the buf may contain post-think content
        # where a fresh opener will latch normally.
        if self._think_depth == 0 and (
            _TAG_OPENER_RE.search(self._buf)
            or _HTML_OPENER_RE.search(self._buf)
            or _JS_OPENER_RE.search(self._buf)
            or _CODE_DISCUSSION_RE.search(self._buf)
        ):
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


class DiagnoseBloatDetector:
    """Abort a fix-turn stream whose `<diagnose>` block never closes.

    `DeliberationDetector` latches the moment ANY output tag appears —
    including `<diagnose>` — so it CANNOT catch a stream that opens
    `<diagnose>` then rambles for tens of thousands of chars without ever
    emitting `</diagnose>` or moving on to `<patch>`/`<html_file>`. That is
    exactly the chess-trace iter-2 failure (674 s, 6707 tokens, an unclosed
    `<diagnose>` + probe-JSON loop, no usable patch — see
    a-game-of-chess-player-vs-cpu_20260621_193434).

    Fix-turn-only and genre-free by construction: it ARMS only after a
    `<diagnose>` opener is seen (first builds emit `<html_file>`, never
    `<diagnose>` — so a long legitimate first build is never touched), and
    it DISARMS the moment `</diagnose>` closes OR a `<patch>`/`<html_file>`
    opener appears (the model committed to code — honoring the standing
    rule "latch on code-emission, not length"). Budget is in CHARS for the
    same backend-agnostic reason as `DeliberationDetector`. Disable with
    DISABLE_DIAGNOSE_BLOAT_DETECTOR=1.
    """

    __slots__ = (
        "_armed", "_disarmed", "_chars_since_open", "_carry",
        "_budget", "_disabled", "stall_reason",
    )

    def __init__(self, budget_chars: int = 2400) -> None:
        import os as _os
        self._armed = False
        self._disarmed = False
        self._chars_since_open = 0
        # ~12-char carry so a tag split across two streamed pieces
        # ("<diag" + "nose>") still matches.
        self._carry = ""
        self._budget = budget_chars
        self._disabled = (
            _os.environ.get("DISABLE_DIAGNOSE_BLOAT_DETECTOR") == "1"
        )
        self.stall_reason: str | None = None

    def feed(self, piece: str) -> bool:
        if self._disabled or self._disarmed:
            return False
        window = self._carry + piece
        if not self._armed:
            if "<diagnose>" in window:
                self._armed = True
                idx = window.rfind("<diagnose>") + len("<diagnose>")
                self._chars_since_open = max(0, len(window) - idx)
            self._carry = window[-12:]
            return False
        # Armed: disarm on a close OR on committing to code output.
        if (
            "</diagnose>" in window
            or "<patch>" in window
            or "<html_file>" in window
        ):
            self._disarmed = True
            self._carry = window[-12:]
            return False
        self._chars_since_open += len(piece)
        self._carry = window[-12:]
        if self._chars_since_open >= self._budget:
            self.stall_reason = "diagnose_bloat"
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
    # One-shot safety valve: True when we observed an inline-data-bloat
    # repetition signal while a `<html_file>` was still open, granted one
    # continuation, and resumed streaming.
    loop_grace_used: bool = False
    # Textual reason for the grace event (currently one value).
    loop_grace_reason: str | None = None
    # Silent-stream signal — see _SILENT_STREAM_SECONDS_FLOOR in
    # `stream_chat`. True when the model produced ZERO non-empty
    # content pieces for the full wall-clock floor, indicating its
    # entire generation went to a reasoning/thinking channel that
    # surfaces as empty content. Folded into `stalled` for back-compat;
    # standalone field so the agent can route to a specific recovery
    # ("emit a tag immediately, skip the reasoning preamble").
    silent: bool = False
    # Diagnose-bloat signal (chess-trace fix 2026-06-22): the model opened
    # <diagnose> and never closed it within the char budget, nor moved on
    # to <patch>/<html_file> — the iter-2 unclosed-diagnose runaway. Folded
    # into `stalled`; standalone field routes the agent to patch-first
    # coaching instead of a generic retry.
    diagnose_bloat: bool = False


async def stream_chat(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: float | str | None = None,
    stall_seconds: float = 600.0,
    overall_seconds: float = 1800.0,
) -> StreamResult:
    """Stream a chat completion with a stall watchdog.

    `stall_seconds` is the per-chunk inactivity budget — if no token arrives
    for that long we stop waiting. `overall_seconds` is accepted for caller
    compatibility and telemetry, but it is NOT an active-stream cutoff: a
    model that keeps producing tokens is working and must be allowed to close
    its `<html_file>`. Runaway output is bounded by repetition/deliberation
    detectors and backend max-token settings, not by wall-clock time.

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
    silent = False
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
    # Diagnose-bloat guard (chess-trace fix): abort an unclosed <diagnose>.
    diag = DiagnoseBloatDetector()
    diagnose_bloat = False
    loop_grace_used = False
    loop_grace_reason: str | None = None
    # Silent-stream guard. The DeliberationDetector and RepetitionDetector
    # both feed on the message *content* (`piece`). When a model emits
    # ALL of its tokens via a reasoning/thinking channel that surfaces as
    # empty `content` strings — observed on qwen3.6:27b 2026-05-23 doom
    # iter 4: 32,777 completion tokens, ZERO non-empty pieces, 1356 s
    # wall-clock — neither detector ever fires. This guard catches that
    # exact shape: no visible content AND no backend activity for the
    # floor window. Uses last_activity_at (chunk arrival / prefill
    # progress), not stream start — holochess/GLM 20260623 fix.
    _SILENT_STREAM_SECONDS_FLOOR = 180.0
    last_activity_at = started

    # ollama.AsyncClient.chat returns an async iterator of dicts. We pull
    # .__aiter__() so we can wrap each .__anext__() in asyncio.wait_for.
    stream = await client.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options,
        keep_alive=keep_alive,
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
                # Silent-stream guard — see _SILENT_STREAM_SECONDS_FLOOR
                # comment block above. Fires only when n_tokens == 0
                # (no visible content has EVER arrived) AND no backend
                # activity for the floor window. Measure from
                # last_activity_at (not stream start) so a long Ollama
                # prefill on a big prompt is not mistaken for silence —
                # same holochess/GLM fix as backend.MLXBackend.
                if (
                    n_tokens == 0
                    and (time.monotonic() - last_activity_at) >= _SILENT_STREAM_SECONDS_FLOOR
                ):
                    silent = True
                    stall_at = 0
                    break
                # Empty chunk still counts as daemon activity.
                last_activity_at = time.monotonic()
                continue
            last_activity_at = time.monotonic()
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
                if _should_grace_inline_data_bloat(
                    stall_reason=repeat.stall_reason,
                    assembled_text="".join(parts),
                    grace_already_used=loop_grace_used,
                    completion_tokens=n_tokens,
                ):
                    loop_grace_used = True
                    loop_grace_reason = "inline_data_bloat_unclosed_html_file"
                    # Reset the detector so we only abort if the same
                    # loop shape appears again after this grace.
                    repeat = RepetitionDetector()
                    continue
                looped = True
                stall_at = n_tokens
                break

            # ---- deliberation detector (A2) -------------------------
            if delib.feed(piece):
                deliberated = True
                stall_at = n_tokens
                break

            # ---- diagnose-bloat detector ----------------------------
            if diag.feed(piece):
                diagnose_bloat = True
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
        stalled=stalled or looped or deliberated or silent or diagnose_bloat,
        stall_at_token=stall_at,
        looped=looped,
        deliberated=deliberated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        loop_kind=repeat.stall_reason if looped else None,
        loop_line=repeat.loop_line if looped else None,
        loop_grace_used=loop_grace_used,
        loop_grace_reason=loop_grace_reason,
        silent=silent,
        diagnose_bloat=diagnose_bloat,
    )


async def stream_chat_with_retry(
    client: ollama.AsyncClient,
    model: str,
    messages: list[dict],
    on_token: Callable[[str], None] | None = None,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: float | str | None = None,
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
        try:
            result = await stream_chat(
                client,
                model,
                messages,
                on_token,
                options=options,
                keep_alive=keep_alive,
                stall_seconds=stall_seconds,
                overall_seconds=overall_seconds,
            )
        except ollama.ResponseError as e:
            msg = str(e).lower()
            opts = dict(options or {})
            # On single-GPU-pinned Ollama daemons, `num_gpu=999` can force an
            # impossible all-GPU memory layout for Q8 + 262K ctx. Keep num_ctx
            # intact, drop only the offload hint, and let Ollama choose layout.
            if (
                "memory layout cannot be allocated" in msg
                and "num_gpu" in msg
                and "num_gpu" in opts
            ):
                retry_opts = dict(opts)
                retry_opts.pop("num_gpu", None)
                result = await stream_chat(
                    client,
                    model,
                    messages,
                    on_token,
                    options=retry_opts,
                    keep_alive=keep_alive,
                    stall_seconds=stall_seconds,
                    overall_seconds=overall_seconds,
                )
            else:
                raise
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
    keep_alive: float | str | None = None,
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
            keep_alive=keep_alive,
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
