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

    We do NOT touch the body of <patch> blocks here; per-patch repairs
    (fence stripping) happen after extraction so we don't accidentally
    eat a fence that lives in REGULAR text outside a patch.
    """
    if not reply:
        return reply
    if reply.startswith(_BOM):
        reply = reply[1:]
    reply = _normalize_lines(reply)
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
