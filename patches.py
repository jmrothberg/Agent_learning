"""SEARCH/REPLACE patch parsing and application.

Why we use this format instead of asking the model to re-emit the full file:

  * Smaller models (7B-30B) drop tokens during long completions. We saw the
    asteroids run produce a structurally-correct file but with `KEYMAP[e.code]`
    silently rendered as `KEYMAP` in the live token stream, because the model
    momentarily lost track of what it was writing. The .html file on disk was
    actually correct, but the same class of failure DOES corrupt files in
    other runs.

  * Patches are short. A 5-line SEARCH/REPLACE has roughly 1/100th the
    surface area for the model to mess up vs a 250-line file rewrite.

  * Patches force the model to look at the existing code instead of writing
    from memory. This dramatically improves the "I'll just rewrite this
    similar but different" failure mode.

  * Failed application is a PROPER ERROR: if the SEARCH block doesn't appear
    verbatim in the file, we tell the model exactly what didn't match instead
    of silently producing wrong code.

Format (Aider-style, slightly relaxed):

    <patch>
    <<<<<<< SEARCH
    exact lines from the current file
    =======
    new lines that replace them
    >>>>>>> REPLACE
    </patch>

Multiple <patch> blocks are allowed in one reply; they apply in order. An
empty SEARCH block with content in REPLACE means "prepend"; an empty REPLACE
block means "delete".

The agent always falls back to a full-file rewrite if patches don't apply,
so format errors degrade gracefully instead of being fatal.

Match cascade (most specific to most lenient):
  1. Exact substring match.
  2. Character-preserving normalization: smart-quotes / dash variants /
     NBSP and other unicode spaces collapsed to ASCII. Each transform is
     1:1, so positions in normalized space map directly to original
     indices. This rescues the most common small-model failures (model
     emits `'` where the file has `'`, or `—` where the file has `-`).
  3. Whitespace-collapse (runs of spaces/tabs treated as a single space).
  4. Trim (strip leading/trailing whitespace on both sides).

Cross-patch validation:
  * Per patch, count matches. >1 ⇒ ambiguous, reject with a prescriptive
    "add more surrounding context" error.
  * Across patches, sort matches by start index; reject any pair whose
    spans overlap with a "merge edits N and M into one patch" error.
  * Apply in REVERSE source-order so each splice keeps earlier offsets
    valid. (Pi-mono's edit-diff engine pattern.)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_PATCH_RE = re.compile(
    r"<patch>\s*"
    r"<{5,}\s*SEARCH\s*\n"
    r"(?P<search>.*?)\n?"
    r"={5,}\s*\n"
    r"(?P<replace>.*?)\n?"
    r">{5,}\s*REPLACE\s*\n?"
    r"\s*</patch>",
    re.DOTALL | re.IGNORECASE,
)

# Detects a SEARCH/REPLACE marker as a STANDALONE LINE (no surrounding
# code on the same line). Used to reject patches whose body contains
# embedded markers — a real failure mode where the model writes a
# malformed patch with two `>>>>>>> REPLACE` markers and the regex
# above backtracks past the first one, capturing it (plus a chunk of
# real code) into the REPLACE body. Applying that splices literal
# `>>>>>>> REPLACE` lines into the file. See the missile-command
# session in games/traces/game-of-misile-command-good-gr_*.log:
# turn 12 caused turn 13 to fail with "Unexpected token '>>>'" via
# exactly this path.
#
# Uses MULTILINE + ^/$ so we ONLY match a marker that is the WHOLE line
# (modulo whitespace). Avoids false positives on legitimate code that
# happens to contain '=======' inline (e.g. JS assignments using long
# divider comments).
_EMBEDDED_MARKER_RE = re.compile(
    r"^\s*(?:<{5,}\s*SEARCH|={5,}|>{5,}\s*REPLACE)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _has_embedded_marker(text: str) -> bool:
    return bool(_EMBEDDED_MARKER_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Patch-delta token-repetition detector
# ---------------------------------------------------------------------------
#
# When a model's REPLACE block contains the same non-trivial line repeated
# many times in a row, that's almost always a degenerate sampling loop, not
# a legitimate edit. DK trace 20260514_104131 iter 3: the model emitted a
# patch whose REPLACE added `}else{` × 11 consecutively. micro_probes
# detected the pattern AFTER the patch had applied; the token-spam shipped
# to disk and the next test ran on the broken file.
#
# This detector runs at apply time so we can reject the patch BEFORE writing
# anything. Mirrors the shape of `_detect_block_bloat` in tools.py but
# operates on the REPLACE delta only (not the full file), which keeps
# legitimate switch-statement chains (`case X:` / `case Y:`) safe because
# the lines are non-identical even though the structure looks repetitive.
#
# Defenses against false positives:
#   - Minimum trimmed length 6 chars (skip lone `}`, `});`, `;`).
#   - Skip pure-punctuation lines (closing braces, semicolons, commas).
#   - Skip lines inside an unbalanced backtick region — template literals
#     legally contain repeated identical lines (e.g. multi-line strings).

_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def _detect_replace_repetition(
    replace_body: str,
    *,
    min_run: int = 3,
    min_line_len: int = 6,
) -> tuple[str, int] | None:
    """Detect a run of ≥`min_run` consecutive identical non-trivial lines
    in a patch REPLACE body.

    Returns (line, count) for the largest run when one exists, None
    otherwise. See module-level comment for defenses.
    """
    if not replace_body:
        return None
    lines = replace_body.splitlines()
    if len(lines) < min_run:
        return None

    # Mark which lines fall inside an unbalanced backtick (template
    # literal) region. Toggle on each line by parity of backtick count.
    in_template: list[bool] = []
    ticks_open = False
    for ln in lines:
        in_template.append(ticks_open)
        if ln.count("`") % 2 == 1:
            ticks_open = not ticks_open

    def _trivial(stripped: str, idx: int) -> bool:
        return (
            len(stripped) < min_line_len
            or bool(_PUNCT_ONLY_RE.match(stripped))
            or in_template[idx]
        )

    best_line: str | None = None
    best_count: int = 0
    cur_line: str | None = None
    cur_count: int = 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if _trivial(stripped, i):
            cur_line = None
            cur_count = 0
            continue
        if stripped == cur_line:
            cur_count += 1
        else:
            cur_line = stripped
            cur_count = 1
        if cur_count > best_count:
            best_count = cur_count
            best_line = cur_line
    if best_count >= min_run and best_line is not None:
        return (best_line, best_count)
    return None


# ---------------------------------------------------------------------------
# Char-preserving normalization (1:1 mapping, no length change)
# ---------------------------------------------------------------------------
#
# Each entry maps a single unicode code point to a single ASCII code point,
# so str.translate() is character-preserving: position i in the normalized
# string corresponds to position i in the original string. This lets us
# match in normalized space and splice in original space without an index
# map. Pi-mono's edit-diff.ts uses the same set; small models (qwen3.6,
# gpt-oss) frequently emit smart quotes / em-dashes after "polishing"
# their patch text, which then fails exact match against the LF-clean
# file we wrote ourselves.

_SMART_QUOTES = str.maketrans({
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK (also "apostrophe")
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9
    "′": "'",  # PRIME
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "‟": '"',  # DOUBLE HIGH-REVERSED-9
    "″": '"',  # DOUBLE PRIME
})

_DASHES = str.maketrans({
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "‒": "-",  # FIGURE DASH
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "―": "-",  # HORIZONTAL BAR
    "−": "-",  # MINUS SIGN
})

_SPACES = str.maketrans({
    " ": " ",  # NO-BREAK SPACE
    " ": " ",  # OGHAM SPACE MARK
    " ": " ",  # EN QUAD
    " ": " ",  # EM QUAD
    " ": " ",  # EN SPACE
    " ": " ",  # EM SPACE
    " ": " ",  # THREE-PER-EM SPACE
    " ": " ",  # FOUR-PER-EM SPACE
    " ": " ",  # SIX-PER-EM SPACE
    " ": " ",  # FIGURE SPACE
    " ": " ",  # PUNCTUATION SPACE
    " ": " ",  # THIN SPACE
    " ": " ",  # HAIR SPACE
    " ": " ",  # NARROW NO-BREAK SPACE
    " ": " ",  # MEDIUM MATHEMATICAL SPACE
    "　": " ",  # IDEOGRAPHIC SPACE
})


def _normalize_chars(s: str) -> str:
    """Char-preserving fuzzy normalization.

    Applies smart-quote / dash / unicode-space → ASCII translations. Each
    translation is 1:1, so len(_normalize_chars(s)) == len(s) and position
    i in the result maps to position i in the original.
    """
    return s.translate(_SMART_QUOTES).translate(_DASHES).translate(_SPACES)


# ---------------------------------------------------------------------------
# Repair layer (run before extract_patches)
# ---------------------------------------------------------------------------

# Strip a UTF-8 BOM at the very start of the reply (some Ollama proxies
# inject one). Cheap, harmless when absent.
_BOM = "﻿"

# Match ```html or ```js or ``` on its own line, used to strip stray code
# fences from inside <patch> bodies. Models sometimes wrap each side of
# the SEARCH/REPLACE in a fence, which then fails exact match.
_PATCH_INTERNAL_FENCE_RE = re.compile(
    r"^[ \t]*```[a-zA-Z]*[ \t]*\n?|\n?[ \t]*```[ \t]*$",
    re.MULTILINE,
)

# Outer ```language\n...\n``` fence whose body contains a real tag.
# Local models (DeepSeek-V4, Qwen) sometimes wrap the entire <patch> or
# <html_file> in a markdown code fence ("just to be safe"). The tag
# regexes downstream don't care about surrounding context, BUT when the
# fence body has its own internal ```fences they collide with the body
# parsing and the patch fails to extract. The DK trace 20260513_185815
# burned 7 turns on a fenced <patch>; strip it proactively here so the
# downstream parser sees the same shape as a model that obeyed the
# format directly.
_OUTER_FENCE_RE = re.compile(
    r"^[ \t]*```[a-zA-Z]*[ \t]*\n(.*?)\n[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def _strip_outer_fences_around_tags(reply: str) -> str:
    """Remove ```...``` fences whose body contains <patch> or <html_file>.

    Only strips a fence when its body REALLY contains one of our tags so
    we don't damage fences in regular prose, in <plan> bodies, or in
    pre-existing HTML inside <html_file> (the wrapper itself protects
    its body).
    """
    if "```" not in reply or ("<patch>" not in reply.lower() and "<html_file>" not in reply.lower()):
        return reply

    def _maybe_strip(m: re.Match) -> str:
        body = m.group(1)
        low = body.lower()
        if "<patch>" in low or "<html_file>" in low:
            return body
        return m.group(0)

    return _OUTER_FENCE_RE.sub(_maybe_strip, reply)


def _normalize_lines(s: str) -> str:
    """Normalize line endings to LF.

    Drops CR characters; the result is shorter when CRLF or bare-CR was
    present. This is NOT character-preserving — only call this on whole
    text that we treat uniformly afterwards (whole reply, whole source).
    """
    if "\r" not in s:
        return s
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _strip_internal_fences(body: str) -> str:
    """Remove stray ```lang / ``` fences from a SEARCH or REPLACE body.

    The model occasionally wraps its patch body in a markdown fence ("just
    to be safe"). Those literal fence lines then fail to match the file
    on disk. We strip the FIRST opening fence and LAST closing fence; we
    do NOT strip every fence-looking line, because legitimate code may
    contain "```" inside a string literal.
    """
    if "```" not in body:
        return body
    new = _PATCH_INTERNAL_FENCE_RE.sub("", body, count=2)
    return new


def repair_reply(reply: str) -> str:
    """Pi-mono-style "prepareArguments" pass: fix common malformations
    BEFORE the regex parser sees the text.

    Currently handles:
      * UTF-8 BOM at start
      * CRLF / bare-CR line endings (normalized to LF)
      * Reasoning-model `</think>` prelude (everything up to the last
        `</think>` is chain-of-thought, not the answer — without
        stripping, a CoT that mentions `` `<patch>` `` in markdown
        backticks corrupts the patch parser the same way assets/sounds
        were corrupted in
        games/traces/game-of-space-invaders-with-gr_20260511_093225).
      * Loose spacing inside SEARCH/REPLACE/DIVIDER markers (normalized to
        single space / standard format).
      * Consecutive duplicate `=======` divider lines (collapsed to one).
      * Unclosed <patch> blocks ending with REPLACE marker (closes them).

    We do NOT touch the body of <patch> blocks here; per-patch repairs
    (fence stripping) happen after extraction so we don't accidentally
    eat a fence that lives in REGULAR text outside a patch.
    """
    if not reply:
        return reply
    if reply.startswith(_BOM):
        reply = reply[1:]
    reply = _normalize_lines(reply)
    idx = reply.rfind("</think>")
    if idx >= 0:
        reply = reply[idx + len("</think>"):]
    reply = _strip_outer_fences_around_tags(reply)

    # 1. Normalize spaces inside SEARCH / REPLACE / DIVIDER markers
    reply = re.sub(r"^[ \t]*<{7,}[ \t]*SEARCH[ \t]*$", "<<<<<<< SEARCH", reply, flags=re.MULTILINE | re.IGNORECASE)
    reply = re.sub(r"^[ \t]*={7,}[ \t]*$", "=======", reply, flags=re.MULTILINE)
    reply = re.sub(r"^[ \t]*>{7,}[ \t]*REPLACE[ \t]*$", ">>>>>>> REPLACE", reply, flags=re.MULTILINE | re.IGNORECASE)

    # 2. Collapse consecutive duplicate/empty ======= divider lines
    # This matches multiple ======= lines separated only by whitespace or newlines.
    reply = re.sub(r"\n=======(?:\s*=======)+", "\n=======", reply)

    # 3. Auto-close <patch> blocks that end with the REPLACE marker but are missing </patch>
    # We do this by finding '<patch>' (case-insensitive) where there's no matching '</patch>' ahead
    # before another '<patch>', and we see '>>>>>>> REPLACE'.
    # A simple but extremely robust line-by-line or block-based correction:
    lines = reply.splitlines()
    open_patch_idx = -1
    has_replace = False
    has_close = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        if "<patch>" in line_stripped:
            open_patch_idx = i
            has_replace = False
            has_close = False
        elif ">>>>>>> replace" in line_stripped:
            has_replace = True
        elif "</patch>" in line_stripped:
            has_close = True
            open_patch_idx = -1
        
        # If we reach another <patch> or the end of the file, and we have an open patch
        # with a REPLACE marker but no </patch> closer, let's fix it.
        is_last_line = (i == len(lines) - 1)
        next_is_patch = (not is_last_line and "<patch>" in lines[i+1].strip().lower())
        
        if open_patch_idx != -1 and has_replace and not has_close and (is_last_line or next_is_patch):
            # Insert </patch> right after the REPLACE marker line
            # Let's locate the replace line index starting backwards from i
            for j in range(i, open_patch_idx, -1):
                if ">>>>>>> replace" in lines[j].strip().lower():
                    lines.insert(j + 1, "</patch>")
                    break
            # Reset trackers
            open_patch_idx = -1
            has_replace = False
            has_close = False
            
    reply = "\n".join(lines)
    return reply


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Patch:
    """A single SEARCH/REPLACE block parsed from the model's reply."""

    search: str
    replace: str

    @property
    def is_prepend(self) -> bool:
        return self.search.strip() == ""

    @property
    def is_delete(self) -> bool:
        return self.replace.strip() == "" and self.search.strip() != ""


@dataclass
class PatchResult:
    """Outcome of trying to apply patches to a source file."""

    text: str                # the (possibly edited) file content
    applied: int             # how many patches landed
    failed: list[tuple[int, Patch, str]]  # (index, patch, reason) for each failure


def find_anchor(source: str, search: str, *, ctx_lines: int = 4) -> str | None:
    """Best-effort "where did the model probably mean?" excerpt.

    When a patch SEARCH doesn't match, the typical 27B-model failure is
    "model copied an old version of the line that's since been edited."
    A correction turn needs to show the model what the file ACTUALLY
    says near where it was aiming. We find the longest non-trivial line
    from the SEARCH that appears in the source and return ±ctx_lines
    around it.

    Returns None when no useful anchor exists (no SEARCH line ≥8 chars
    appears in source — the patch was wildly wrong).
    """
    if not source or not search:
        return None
    src_lines = source.splitlines()
    src_norm = [_normalize_chars(ln) for ln in src_lines]
    candidates = [ln.strip() for ln in search.splitlines() if len(ln.strip()) >= 8]
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        cnorm = _normalize_chars(cand)
        for i, sn in enumerate(src_norm):
            if cnorm in sn:
                lo = max(0, i - ctx_lines)
                hi = min(len(src_lines), i + ctx_lines + 1)
                # Prefix line numbers (1-based) to make the excerpt useful
                # to the model as a positional hint.
                width = len(str(hi))
                out = []
                for j in range(lo, hi):
                    marker = ">" if j == i else " "
                    out.append(f"{marker} {str(j + 1).rjust(width)} | {src_lines[j]}")
                return "\n".join(out)
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_patches(reply: str) -> list[Patch]:
    """Pull every <patch>...</patch> block out of a model reply, in order.

    Runs `repair_reply` first so CRLF / BOM / etc. don't break the regex.
    Strips stray markdown fences from each patch body so a model that
    wrapped its SEARCH/REPLACE in ```html``` doesn't fail the literal
    match on the fence lines.
    """
    reply = repair_reply(reply)
    out: list[Patch] = []
    for m in _PATCH_RE.finditer(reply):
        search = _strip_internal_fences(m.group("search"))
        replace = _strip_internal_fences(m.group("replace"))
        out.append(Patch(search=search, replace=replace))
    return out


# ---------------------------------------------------------------------------
# Match cascade
# ---------------------------------------------------------------------------


def _find_all_exact(text: str, needle: str) -> list[tuple[int, int]]:
    """All non-overlapping (start, end) positions of `needle` in `text`."""
    out: list[tuple[int, int]] = []
    L = len(needle)
    if L == 0:
        return out
    pos = 0
    while True:
        i = text.find(needle, pos)
        if i < 0:
            return out
        out.append((i, i + L))
        pos = i + L  # non-overlapping: jump past the match


def _find_all_normalized(text: str, needle: str) -> list[tuple[int, int]]:
    """Match after char-preserving normalization (smart quotes / dashes /
    unicode spaces). Because the normalization is 1:1, positions in the
    normalized text equal positions in the original text.
    """
    norm_text = _normalize_chars(text)
    norm_needle = _normalize_chars(needle)
    if norm_needle == needle and norm_text == text:
        return []  # nothing changed; exact would have caught it
    return _find_all_exact(norm_text, norm_needle)


_WS_RUN = re.compile(r"[ \t]+")


def _norm_ws(s: str) -> str:
    """Collapse runs of spaces/tabs into one space. Newlines preserved."""
    return _WS_RUN.sub(" ", s)


def _orig_offset(text: str, norm_pos: int) -> int | None:
    """Return the index into `text` corresponding to `norm_pos` after
    whitespace collapsing. None if `norm_pos` is past the end.
    """
    n = 0
    i = 0
    L = len(text)
    while i < L:
        if n == norm_pos:
            return i
        ch = text[i]
        if ch == " " or ch == "\t":
            j = i
            while j < L and text[j] in " \t":
                j += 1
            i = j
            n += 1
        else:
            i += 1
            n += 1
    if n == norm_pos:
        return L
    return None


def _find_all_lenient_ws(text: str, needle: str) -> list[tuple[int, int]]:
    """Match after collapsing runs of spaces/tabs in both sides.

    Walks the normalized text and for each match recovers the original-
    text bounds via `_orig_offset`. Newlines are preserved unchanged so
    line structure stays meaningful.
    """
    norm_needle = _norm_ws(needle)
    if not norm_needle:
        return []
    norm_text = _norm_ws(text)
    out: list[tuple[int, int]] = []
    pos = 0
    while True:
        i = norm_text.find(norm_needle, pos)
        if i < 0:
            return out
        start = _orig_offset(text, i)
        end = _orig_offset(text, i + len(norm_needle))
        if start is None or end is None or start >= end:
            return out
        out.append((start, end))
        pos = i + len(norm_needle)


def _find_all_trimmed(text: str, needle: str) -> list[tuple[int, int]]:
    """Last-chance match: strip leading/trailing whitespace from `needle`
    and search again. Useful for patches that copied an extra blank line.
    """
    stripped = needle.strip()
    if not stripped or stripped == needle:
        return []
    return _find_all_exact(text, stripped)


def _locate(text: str, search: str) -> tuple[list[tuple[int, int]], str]:
    """Run the match cascade. Returns (matches, layer_label).

    Stops at the first layer that produces ≥1 matches; later layers are
    not consulted. layer_label is a short tag for trace/error messages.
    """
    m = _find_all_exact(text, search)
    if m:
        return m, "exact"
    m = _find_all_normalized(text, search)
    if m:
        return m, "char-norm"
    m = _find_all_lenient_ws(text, search)
    if m:
        return m, "ws-collapse"
    m = _find_all_trimmed(text, search)
    if m:
        return m, "trimmed"
    return [], "none"


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_patches(source: str, patches: list[Patch]) -> PatchResult:
    """Apply a sequence of SEARCH/REPLACE patches to `source`.

    Cross-patch contract:
      * Each non-prepend patch must match EXACTLY ONCE in the source. >1
        match returns an "ambiguous" failure so the model adds more
        surrounding context.
      * Patches must not overlap in source. Overlapping pairs return a
        "merge edits N and M" failure for both members.
      * Surviving patches are applied in REVERSE source-order so earlier
        offsets stay valid (pi-mono's edit-diff engine pattern).

    Failed patches do NOT abort the rest — partial application is still
    written out so the test harness can give the model meaningful feedback
    on the next turn.
    """
    # Normalize source line endings once. The model's SEARCH text was
    # already LF-normalized inside repair_reply (called from extract_patches),
    # so both sides are now in the same line-ending world.
    source = _normalize_lines(source)

    failed: list[tuple[int, Patch, str]] = []
    # (orig_index, start, end, replacement_text, layer)
    spans: list[tuple[int, int, int, str, str]] = []
    # Prepend patches don't have a source position; we handle them last by
    # accumulating into a prefix string in their original reply order.
    prepend_buf: list[tuple[int, str]] = []

    for i, p in enumerate(patches):
        # --- early reject: embedded markers ---------------------------
        if _has_embedded_marker(p.search) or _has_embedded_marker(p.replace):
            failed.append((i, p, (
                "patch body contains an embedded SEARCH/REPLACE marker "
                "(<<<<<<<, =======, or >>>>>>>) on its own line — looks "
                "like nested or malformed headers. Re-emit ONE marker "
                "set per <patch> block; do not put '>>>>>>> REPLACE' "
                "anywhere except as the closing line."
            )))
            continue

        # --- early reject: token-repetition loop in REPLACE body ------
        # DK trace 20260514_104131 iter 3: REPLACE added `}else{` × 11
        # consecutively; micro_probes caught it after-the-fact but the
        # spam shipped. Block here so the patch never lands.
        rep = _detect_replace_repetition(p.replace)
        if rep is not None:
            rep_line, rep_count = rep
            snippet = (
                (rep_line[:80] + "...") if len(rep_line) > 80 else rep_line
            )
            failed.append((i, p, (
                f"patch REPLACE block contains the same line repeated "
                f"{rep_count} times consecutively: {snippet!r}. This is "
                "a token-repetition loop, not a legitimate edit. Re-emit "
                "a focused patch with NO repeated lines; if you truly "
                "need to repeat a construct, refactor into a loop or "
                "helper function."
            )))
            continue

        # --- prepend (empty SEARCH) ----------------------------------
        if p.is_prepend:
            prepend_buf.append((i, p.replace))
            continue

        # --- locate ----------------------------------------------------
        matches, layer = _locate(source, p.search)
        if not matches:
            snippet = (p.search[:120] + "...") if len(p.search) > 120 else p.search
            failed.append((i, p, (
                f"SEARCH block not found in file: {snippet!r}. The CURRENT "
                "FILE ON DISK is the truth — re-copy the lines exactly, "
                "including indentation."
            )))
            continue
        if len(matches) > 1:
            snippet = (p.search[:80] + "...") if len(p.search) > 80 else p.search
            failed.append((i, p, (
                f"SEARCH block matched {len(matches)} places in the file — "
                "ambiguous. Add more SURROUNDING CONTEXT (e.g. the function "
                f"name above and a unique line below) so it matches exactly "
                f"once. SEARCH was: {snippet!r}"
            )))
            continue

        start, end = matches[0]
        spans.append((i, start, end, p.replace, layer))

    # --- non-overlap validation across surviving spans ----------------
    # Sort by start; any pair with prev.end > next.start is an overlap.
    if len(spans) >= 2:
        sorted_spans = sorted(spans, key=lambda s: s[1])
        bad_indices: set[int] = set()
        for a, b in zip(sorted_spans, sorted_spans[1:]):
            if a[2] > b[1]:
                bad_indices.add(a[0])
                bad_indices.add(b[0])
        if bad_indices:
            for (orig_i, start, end, rep, layer) in list(spans):
                if orig_i in bad_indices:
                    failed.append((orig_i, patches[orig_i], (
                        f"patch #{orig_i + 1} overlaps another patch in this "
                        "reply — both touch the same region of the file. "
                        "Merge them into ONE <patch> with combined SEARCH "
                        "and REPLACE; pi-mono-style: edits[].oldText is "
                        "matched against the ORIGINAL file, so overlapping "
                        "patches are ambiguous."
                    )))
            spans = [s for s in spans if s[0] not in bad_indices]

    # --- apply in REVERSE source-order so offsets stay stable -------
    text = source
    spans.sort(key=lambda s: s[1], reverse=True)
    applied = 0
    for (_orig_i, start, end, rep, _layer) in spans:
        text = text[:start] + rep + text[end:]
        applied += 1

    # --- prepend after spans applied (so prepend lands at NEW pos 0) -
    if prepend_buf:
        prefix_parts: list[str] = []
        for (_i, content) in prepend_buf:
            prefix_parts.append(
                content + ("" if content.endswith("\n") else "\n")
            )
            applied += 1
        text = "".join(prefix_parts) + text

    return PatchResult(text=text, applied=applied, failed=failed)


# ---------------------------------------------------------------------------
# Format-failure classification
# ---------------------------------------------------------------------------
#
# When `extract_patches` returns [] AND no <html_file> body could be pulled
# from the reply, we want to tell the model WHY — generic "I could not find
# a <patch> or <html_file> block" sends the model into a death-spiral on
# the same broken shape (DK trace 20260513_185815: 7 consecutive rejections
# on the same fenced-patch reply).
#
# `classify_format_failure` runs a series of detectors and returns the
# first match. Order: most-specific first (fenced tag is more informative
# than "no tags at all").


@dataclass
class FormatRejection:
    """Structured reason why a model reply could not be materialized.

    Returned by `classify_format_failure` when both <patch> and
    <html_file> extraction fail. The caller (agent.py) translates this
    into a specific user-turn hint instead of the generic fallback.
    """

    kind: str            # short tag — "tags_in_fence", "bare_markers", ...
    hint: str            # one-line human-readable summary (for logs / UI)
    detail: str          # multi-line model-facing message describing the fix


# Standalone SEARCH/REPLACE markers (whole-line match required).
_BARE_SEARCH_RE = re.compile(r"^[ \t]*<{5,}\s*SEARCH\s*$", re.MULTILINE | re.IGNORECASE)
_BARE_REPLACE_RE = re.compile(r"^[ \t]*>{5,}\s*REPLACE\s*$", re.MULTILINE | re.IGNORECASE)
# A `<html>...</html>` element NOT wrapped in <html_file>. Used to catch
# the "model emitted bare HTML element with no doctype" variant.
_BARE_HTML_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_BARE_HTML_CLOSE_RE = re.compile(r"</html\s*>", re.IGNORECASE)


def classify_format_failure(reply: str) -> FormatRejection | None:
    """Diagnose why a reply has no parseable <patch> / <html_file>.

    Returns None when the reply doesn't look like a botched code-emission
    attempt at all (e.g. pure prose, plan-only). The caller handles the
    "no code emitted" case separately — this function only fires when
    there's evidence the model TRIED to emit code but the shape was off.

    Detectors, in priority order:
      * tags_in_fence       — <patch>/<html_file> inside ``` fence
      * bare_markers        — SEARCH/REPLACE markers without <patch> wrapper
      * unclosed_patch      — <patch> with no </patch>
      * unclosed_html_file  — <html_file> with no </html_file>
      * wrong_tag_html      — <html>...</html> with no <html_file> wrapper
      * wrong_tag_patches   — <patches> (plural) instead of <patch>
    """
    if not reply:
        return None
    low = reply.lower()

    # 1. <patch> or <html_file> trapped inside a markdown fence. We check
    # AFTER repair_reply has already stripped outer fences whose body
    # contains a tag (so this fires only on shapes the proactive stripper
    # couldn't fix — e.g. nested fences or fences with no closer).
    for fm in re.finditer(r"```[a-zA-Z]*[ \t]*\n(.*?)(?:\n```|$)", reply, re.DOTALL):
        body_low = fm.group(1).lower()
        if "<patch>" in body_low or "<html_file>" in body_low:
            return FormatRejection(
                kind="tags_in_fence",
                hint=(
                    "Your <patch> / <html_file> block was inside a "
                    "```markdown fence. Emit raw tags."
                ),
                detail=(
                    "PARSE ERROR: I found a <patch> or <html_file> tag "
                    "INSIDE a ```...``` markdown code fence. The harness "
                    "parses raw tags only — fenced content is invisible "
                    "to it. Re-emit the SAME tag with NO surrounding "
                    "```fence```. Your reply should open with <patch> or "
                    "<html_file> directly (after at most a <diagnose>...</diagnose> "
                    "or <notes>...</notes> block)."
                ),
            )

    # 2. Bare SEARCH/REPLACE markers — model wrote the body of a patch
    # but forgot the <patch>...</patch> wrapper. Exclude the case where
    # the wrapper exists but is the wrong tag name (`<patches>`) — that
    # gets a more specific message below.
    if (
        _BARE_SEARCH_RE.search(reply)
        and _BARE_REPLACE_RE.search(reply)
        and "<patch>" not in low
        and not re.search(r"<patches\b", reply, re.IGNORECASE)
    ):
        return FormatRejection(
            kind="bare_markers",
            hint=(
                "You emitted SEARCH/REPLACE markers without the "
                "<patch>...</patch> wrapper."
            ),
            detail=(
                "PARSE ERROR: I see <<<<<<< SEARCH and >>>>>>> REPLACE "
                "markers but no <patch>...</patch> wrapper around them. "
                "Wrap the block:\n"
                "<patch>\n<<<<<<< SEARCH\n... old lines ...\n=======\n"
                "... new lines ...\n>>>>>>> REPLACE\n</patch>\n"
                "One <patch> wrapper per SEARCH/REPLACE pair."
            ),
        )

    # 3. <patch> opened but never closed. Could be a stream stall.
    if "<patch>" in low and "</patch>" not in low:
        return FormatRejection(
            kind="unclosed_patch",
            hint="Your <patch> block has no closing </patch>.",
            detail=(
                "PARSE ERROR: I see <patch> but no closing </patch>. "
                "Either the reply was cut off mid-stream, or you "
                "forgot the closing tag. Re-emit the COMPLETE <patch>"
                "...</patch> block with the closing tag on its own line "
                "after >>>>>>> REPLACE."
            ),
        )

    # 4. <html_file> opened but never closed. The extractor already
    # accepts this shape (variant 2/3 in _extract_html_inner), so this
    # only fires when the extractor itself rejected — meaning no </html>
    # was found either. Distinct from #3 because the fix is different.
    if "<html_file>" in low and "</html_file>" not in low and "</html>" not in low:
        return FormatRejection(
            kind="unclosed_html_file",
            hint="Your <html_file> block has no closing </html_file> or </html>.",
            detail=(
                "PARSE ERROR: I see <html_file> but no closing tag and "
                "no </html> inside the body. The stream may have been "
                "cut off. Re-emit the COMPLETE <html_file>...</html_file> "
                "block, ending with </html></html_file> on the last lines."
            ),
        )

    # 5. Bare <html>...</html> element with no <html_file> wrapper.
    # The extractor's bare-doctype path (variant 5) catches docs that
    # start with <!DOCTYPE>; this catches the no-doctype shape.
    if (
        "<html_file>" not in low
        and "<patch>" not in low
        and "<!doctype" not in low
        and _BARE_HTML_RE.search(reply)
        and _BARE_HTML_CLOSE_RE.search(reply)
    ):
        return FormatRejection(
            kind="wrong_tag_html",
            hint=(
                "You used <html>...</html> instead of "
                "<html_file>...</html_file>."
            ),
            detail=(
                "PARSE ERROR: I see an <html>...</html> document but "
                "no <html_file> wrapper. Wrap the COMPLETE HTML "
                "document in <html_file>...</html_file> tags and "
                "include a <!DOCTYPE html> directive on the first line "
                "of the body."
            ),
        )

    # 6. <patches> (plural) — common typo, also <patch:>, <Patches>, etc.
    if re.search(r"<patches\b", reply, re.IGNORECASE) and "<patch>" not in low:
        return FormatRejection(
            kind="wrong_tag_patches",
            hint="You used <patches> instead of <patch>.",
            detail=(
                "PARSE ERROR: I see <patches> (plural) but the tag is "
                "<patch> (singular). Use one <patch>...</patch> block "
                "per edit; multiple <patch> blocks per reply are "
                "allowed and apply together."
            ),
        )

    return None
