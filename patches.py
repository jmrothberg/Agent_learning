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
"""

from __future__ import annotations

import re
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


def extract_patches(reply: str) -> list[Patch]:
    """Pull every <patch>...</patch> block out of a model reply, in order."""
    out: list[Patch] = []
    for m in _PATCH_RE.finditer(reply):
        out.append(Patch(search=m.group("search"), replace=m.group("replace")))
    return out


def apply_patches(source: str, patches: list[Patch]) -> PatchResult:
    """Apply a sequence of SEARCH/REPLACE patches to `source`.

    A patch fails (is recorded but does NOT abort the rest) if its SEARCH
    block doesn't appear verbatim in the current text. We return both the
    final text and the list of failures so the caller can re-prompt the
    model with the specific failures.

    Whitespace strategy: we try exact match first, then a "lenient" match
    where we collapse runs of spaces/tabs in both source and search to a
    single space. That handles models that re-indent by accident without
    breaking on intentional whitespace differences in strings.
    """
    text = source
    failed: list[tuple[int, Patch, str]] = []
    applied = 0

    for i, p in enumerate(patches):
        if p.is_prepend:
            # No SEARCH means: insert REPLACE at the start of file.
            text = p.replace + ("\n" if not p.replace.endswith("\n") else "") + text
            applied += 1
            continue

        # 1) exact match — by far the most common case
        if p.search in text:
            text = text.replace(p.search, p.replace, 1)
            applied += 1
            continue

        # 2) lenient whitespace match — find a slice of `text` that matches
        #    SEARCH after we collapse internal runs of spaces/tabs.
        lenient = _lenient_find_replace(text, p.search, p.replace)
        if lenient is not None:
            text = lenient
            applied += 1
            continue

        # 3) trimmed — strip leading/trailing whitespace on both sides
        if p.search.strip() in text:
            text = text.replace(p.search.strip(), p.replace, 1)
            applied += 1
            continue

        # 4) failed — record what we tried so the model can correct it
        snippet = (p.search[:120] + "...") if len(p.search) > 120 else p.search
        failed.append((i, p, f"SEARCH block not found in file: {snippet!r}"))

    return PatchResult(text=text, applied=applied, failed=failed)


def _lenient_find_replace(text: str, search: str, replace: str) -> str | None:
    """Try to apply replace by matching search modulo whitespace runs.

    We can't do this with a simple .replace because the slice we replace must
    be the actual substring in `text`, not the whitespace-normalized version.
    So we walk the text looking for a window whose normalized form equals the
    normalized search, then splice.
    """
    norm_search = _norm_ws(search)
    if not norm_search:
        return None

    # Build a normalized index → original index map so we can recover the
    # actual substring once we find a normalized match. Simple approach: walk
    # in O(N*M) which is fine for files ≤ ~50KB.
    norm_text = _norm_ws(text)
    pos = norm_text.find(norm_search)
    if pos == -1:
        return None

    # Find the slice bounds in the original text. We do it by re-walking
    # both strings in lock-step and counting normalized chars consumed.
    start = _orig_offset(text, pos)
    end = _orig_offset(text, pos + len(norm_search))
    if start is None or end is None or start > end:
        return None
    return text[:start] + replace + text[end:]


_WS_RUN = re.compile(r"[ \t]+")


def _norm_ws(s: str) -> str:
    """Collapse runs of spaces/tabs into one space. Newlines preserved."""
    return _WS_RUN.sub(" ", s)


def _orig_offset(text: str, norm_pos: int) -> int | None:
    """Return the index into `text` corresponding to `norm_pos` after
    whitespace collapsing. None if norm_pos is past the end.
    """
    n = 0
    i = 0
    L = len(text)
    while i < L:
        if n == norm_pos:
            return i
        ch = text[i]
        if ch == " " or ch == "\t":
            # consume the entire run, count as ONE normalized char
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
