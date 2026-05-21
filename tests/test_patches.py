"""Tests for patches.py.

Covers the pi-mono-inspired upgrades:
  - Char-preserving fuzzy normalization (smart quotes, dashes, unicode spaces)
  - Uniqueness check: >1 source match → ambiguous failure
  - Cross-patch non-overlap validation
  - Reverse-order application preserves offsets
  - Repair layer (BOM, CRLF, internal fence stripping)

Plus regression tests for the pre-existing behavior we preserve:
  - Exact match
  - Whitespace-collapse lenient match
  - Trimmed match
  - Embedded-marker rejection
  - Prepend (empty SEARCH)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from project root: `.venv/bin/python -m pytest tests/test_patches.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import patches  # noqa: E402
from patches import Patch, apply_patches, extract_patches, find_anchor, repair_reply  # noqa: E402


# ---------------------------------------------------------------------------
# find_anchor — used by the patch retry prompt to show the model the
# region of the file it was probably aiming at when its SEARCH didn't
# match. The most useful failure mode to fix is "model edited from a
# stale copy of the line."
# ---------------------------------------------------------------------------


def test_find_anchor_returns_window_around_longest_matching_line():
    source = "\n".join([
        "function foo() {",
        "  const x = 1;",
        "  const y = computeWidget(x);",
        "  return y * 2;",
        "}",
    ])
    # The model's SEARCH copied a stale version of line 3 — but line 4
    # is unchanged, so we anchor on it.
    search = "\n".join([
        "  const y = oldCompute(x);",
        "  return y * 2;",
    ])
    anchor = find_anchor(source, search, ctx_lines=1)
    assert anchor is not None
    assert "return y * 2" in anchor
    # The closest hit gets the > marker.
    assert ">" in anchor


def test_find_anchor_returns_none_when_search_is_alien():
    source = "alpha\nbeta\ngamma\n"
    # No line of SEARCH appears in source; nothing to anchor on.
    assert find_anchor(source, "completely_unrelated_token_xyzzy") is None


def test_find_anchor_handles_empty_inputs():
    assert find_anchor("", "anything") is None
    assert find_anchor("anything", "") is None


# ---------------------------------------------------------------------------
# extract_patches / repair_reply
# ---------------------------------------------------------------------------


def test_extract_patches_basic():
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "old line\n"
        "=======\n"
        "new line\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    out = extract_patches(reply)
    assert len(out) == 1
    assert out[0].search == "old line"
    assert out[0].replace == "new line"


def test_extract_patches_strips_internal_fence():
    """Model wrapped its SEARCH/REPLACE body in ```html fences — repair
    layer should strip them so the literal match works."""
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "```html\n"
        "<div>old</div>\n"
        "```\n"
        "=======\n"
        "```html\n"
        "<div>new</div>\n"
        "```\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    out = extract_patches(reply)
    assert len(out) == 1
    assert "```" not in out[0].search
    assert "```" not in out[0].replace
    assert "<div>old</div>" in out[0].search
    assert "<div>new</div>" in out[0].replace


def test_repair_reply_normalizes_crlf():
    s = "line1\r\nline2\r\nline3"
    out = repair_reply(s)
    assert out == "line1\nline2\nline3"


def test_repair_reply_strips_bom():
    s = "﻿foo bar"
    out = repair_reply(s)
    assert out == "foo bar"
    assert not out.startswith("﻿")


def test_repair_reply_handles_bare_cr():
    s = "line1\rline2"
    out = repair_reply(s)
    assert out == "line1\nline2"


# ---------------------------------------------------------------------------
# apply_patches — exact match (regression)
# ---------------------------------------------------------------------------


def test_apply_exact_match():
    src = "hello world\nfoo bar\nbaz qux\n"
    patch = Patch(search="foo bar", replace="FOO BAR")
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert result.failed == []
    assert "FOO BAR" in result.text
    assert "foo bar" not in result.text


def test_apply_multiple_non_overlapping():
    src = "alpha\nbeta\ngamma\ndelta\n"
    patches_ = [
        Patch(search="alpha", replace="ALPHA"),
        Patch(search="gamma", replace="GAMMA"),
    ]
    result = apply_patches(src, patches_)
    assert result.applied == 2
    assert result.failed == []
    assert "ALPHA" in result.text
    assert "GAMMA" in result.text
    assert "beta" in result.text


# ---------------------------------------------------------------------------
# Char-preserving fuzzy match — pi-mono #1
# ---------------------------------------------------------------------------


def test_apply_smart_quote_match():
    """Model emits ASCII apostrophe; file has a curly apostrophe (or
    vice-versa). Char-preserving normalization should rescue this."""
    src = "const greeting = ‘hello’;\n"  # curly quotes in file
    patch = Patch(
        search="const greeting = 'hello';",  # ASCII quotes in patch
        replace="const greeting = 'goodbye';",
    )
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert "goodbye" in result.text


def test_apply_em_dash_match():
    """File has ASCII hyphen; model emits em-dash."""
    src = "const x = a - b;\n"
    patch = Patch(
        search="const x = a — b;",  # em-dash
        replace="const x = a + b;",
    )
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert "a + b" in result.text


def test_apply_nbsp_match():
    """File has NBSP; model emits regular space."""
    src = "ship speed = 5;\n"  # NBSP between "ship" and "speed"
    patch = Patch(
        search="ship speed = 5;",  # regular space
        replace="ship speed = 10;",
    )
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert "= 10" in result.text


def test_normalize_chars_is_length_preserving():
    """Each entry in our normalization tables must map a single char to a
    single ASCII char so positions in normalized space map directly to
    positions in original space."""
    samples = [
        "foo ‘bar’",
        "a — b",
        "ship speed",
        "“double” quotes",
        "′prime″ marks",
    ]
    for s in samples:
        n = patches._normalize_chars(s)
        assert len(n) == len(s), f"length changed: {s!r} → {n!r}"


# ---------------------------------------------------------------------------
# Uniqueness check — pi-mono #2 (part 1)
# ---------------------------------------------------------------------------


def test_apply_ambiguous_search_rejected():
    """SEARCH that matches twice in the file is ambiguous — must reject
    with a prescriptive 'add more context' error."""
    src = "var x = 1;\nvar y = 2;\nvar x = 1;\n"  # "var x = 1;" appears twice
    patch = Patch(search="var x = 1;", replace="var x = 99;")
    result = apply_patches(src, [patch])
    assert result.applied == 0
    assert len(result.failed) == 1
    reason = result.failed[0][2]
    assert "ambiguous" in reason.lower() or "2 places" in reason or "add more" in reason.lower()


def test_apply_unique_search_passes():
    """Negative-pair to the ambiguous test: when SEARCH appears once, apply."""
    src = "var x = 1;\nvar y = 2;\n"
    patch = Patch(search="var x = 1;", replace="var x = 99;")
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert result.failed == []
    assert "var x = 99;" in result.text


# ---------------------------------------------------------------------------
# Non-overlap validation — pi-mono #2 (part 2)
# ---------------------------------------------------------------------------


def test_apply_overlapping_patches_rejected():
    """Two patches whose SEARCH matches overlap in the source must both fail
    with a 'merge' instruction; surviving non-overlap patches still apply."""
    src = "alpha beta gamma delta\n"
    patches_ = [
        Patch(search="alpha beta gamma", replace="X"),
        Patch(search="beta gamma delta", replace="Y"),
    ]
    result = apply_patches(src, patches_)
    assert result.applied == 0
    assert len(result.failed) == 2
    for (_i, _p, reason) in result.failed:
        assert "overlap" in reason.lower() or "merge" in reason.lower()


def test_apply_overlapping_with_third_clean_patch():
    """Mix: two overlapping (rejected) + one independent (applies)."""
    src = "alpha beta gamma\n\nstandalone line\n"
    patches_ = [
        Patch(search="alpha beta", replace="X"),
        Patch(search="beta gamma", replace="Y"),
        Patch(search="standalone line", replace="LANDED"),
    ]
    result = apply_patches(src, patches_)
    assert result.applied == 1
    assert len(result.failed) == 2
    assert "LANDED" in result.text


# ---------------------------------------------------------------------------
# Reverse-order application — pi-mono #2 (part 3)
# ---------------------------------------------------------------------------


def test_apply_reverse_order_preserves_offsets():
    """Two patches in document order where the FIRST patch's REPLACE has
    a different length than its SEARCH must not invalidate the SECOND
    patch's offsets. The pi-mono pattern: collect all spans, then apply
    in reverse source-order."""
    src = "AAAAA xxx BBBBB yyy CCCCC\n"
    patches_ = [
        Patch(search="AAAAA", replace="A_REPLACED_LONG"),  # grows
        Patch(search="CCCCC", replace="C_REPL"),            # shrinks
    ]
    result = apply_patches(src, patches_)
    assert result.applied == 2
    assert "A_REPLACED_LONG" in result.text
    assert "C_REPL" in result.text
    assert "BBBBB" in result.text


def test_apply_reverse_with_growth_then_shrink():
    """First patch grows; second (later in source) shrinks. Sanity that
    both edits land at the right locations."""
    src = "FIRST middle LAST"
    patches_ = [
        Patch(search="FIRST", replace="FIRST_GREW_BIGGER"),
        Patch(search="LAST", replace="L"),
    ]
    result = apply_patches(src, patches_)
    assert result.applied == 2
    assert result.text == "FIRST_GREW_BIGGER middle L"


# ---------------------------------------------------------------------------
# Pre-existing behavior we preserve
# ---------------------------------------------------------------------------


def test_lenient_whitespace_match():
    """Two-space indent in file vs four-space indent in patch should match
    via the whitespace-collapse layer."""
    src = "function foo() {\n  return 42;\n}\n"
    patch = Patch(
        search="function foo() {\n    return 42;\n}",  # 4-space indent
        replace="function foo() {\n    return 99;\n}",
    )
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert "return 99" in result.text


def test_trimmed_match():
    """Search with extra leading newline should still match."""
    src = "line\n"
    patch = Patch(search="\n\n  line  \n\n", replace="newline")
    result = apply_patches(src, [patch])
    assert result.applied == 1
    assert "newline" in result.text


def test_embedded_marker_still_rejected():
    """Patches whose body contains a SEARCH/REPLACE marker on a standalone
    line are malformed and must be rejected, not applied."""
    src = "real content\n"
    bad = Patch(
        search="real content",
        replace="new\n>>>>>>> REPLACE\nstuff",  # embedded marker in REPLACE
    )
    result = apply_patches(src, [bad])
    assert result.applied == 0
    assert len(result.failed) == 1
    assert "embedded" in result.failed[0][2].lower() or "marker" in result.failed[0][2].lower()


def test_prepend_with_empty_search():
    src = "existing line\n"
    p = Patch(search="", replace="prepended line")
    result = apply_patches(src, [p])
    assert result.applied == 1
    assert result.text.startswith("prepended line\n")
    assert "existing line" in result.text


def test_search_not_found_returns_failure():
    src = "alpha\n"
    p = Patch(search="missing pattern", replace="replacement")
    result = apply_patches(src, [p])
    assert result.applied == 0
    assert len(result.failed) == 1
    assert "not found" in result.failed[0][2].lower()
    # File unchanged.
    assert result.text == "alpha\n"


# ---------------------------------------------------------------------------
# End-to-end via extract_patches
# ---------------------------------------------------------------------------


def test_end_to_end_smart_quote():
    """Full path: model emits a patch with smart quotes; file has ASCII.
    extract_patches → apply_patches should still land the change."""
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "msg = “hello”;\n"  # smart quotes in patch
        "=======\n"
        "msg = \"goodbye\";\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    src = 'msg = "hello";\n'  # ASCII quotes in file
    p = extract_patches(reply)
    assert len(p) == 1
    result = apply_patches(src, p)
    assert result.applied == 1
    assert "goodbye" in result.text


def test_end_to_end_overlap_error_surfaces():
    """Two overlapping patches in one reply — both should land in failed
    with prescriptive merge text, file unchanged."""
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "alpha beta gamma\n"
        "=======\n"
        "X\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "beta gamma delta\n"
        "=======\n"
        "Y\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    src = "alpha beta gamma delta\n"
    parsed = extract_patches(reply)
    assert len(parsed) == 2
    result = apply_patches(src, parsed)
    assert result.applied == 0
    assert len(result.failed) == 2
    assert result.text == src


def test_repair_reply_duplicate_dividers():
    """Consecutive or duplicate ======= dividers are collapsed."""
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "old code\n"
        "=======\n"
        "=======\n"
        "new code\n"
        ">>>>>>> REPLACE\n"
        "</patch>"
    )
    repaired = repair_reply(reply)
    # Count how many divider lines are left
    assert repaired.count("=======") == 1
    
    # Extract should now succeed perfectly
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    assert parsed[0].search.strip() == "old code"
    assert parsed[0].replace.strip() == "new code"


def test_repair_reply_unclosed_patch():
    """An unclosed <patch> ending with a REPLACE marker is auto-closed."""
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "old code\n"
        "=======\n"
        "new code\n"
        ">>>>>>> REPLACE"
    )
    repaired = repair_reply(reply)
    assert "</patch>" in repaired
    
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    assert parsed[0].search.strip() == "old code"
    assert parsed[0].replace.strip() == "new code"


def test_repair_reply_marker_spaces():
    """Spaces around SEARCH/REPLACE/DIVIDER markers are normalized."""
    reply = (
        "<patch>\n"
        "  <<<<<<<      SEARCH  \n"
        "old code\n"
        "   =======   \n"
        "new code\n"
        "   >>>>>>>   REPLACE  \n"
        "</patch>"
    )
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    assert parsed[0].search.strip() == "old code"
    assert parsed[0].replace.strip() == "new code"

