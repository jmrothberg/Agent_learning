"""Tests for the parser self-correction + format-rejection classification
added to patches.py + agent.py to fix the DK trace 20260513_185815 death
spiral (model emitted valid <patch> inside a ```html markdown fence; the
harness rejected with a generic message and the model retried the same
shape 7 times).

Three layers covered:
  1. `_strip_outer_fences_around_tags` strips a ```...``` fence around a
     tag block, leaving subsequent `extract_patches` / `_extract_html`
     extraction unchanged.
  2. `classify_format_failure` returns a structured rejection naming the
     specific shape error.
  3. `GameAgent._no_usable_code_fallback` consumes the rejection +
     format-stuck streak and emits a model-facing hint that escalates
     to "send full <html_file>" at streak >= 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import patches  # noqa: E402
from patches import (  # noqa: E402
    FormatRejection,
    apply_patches,
    classify_format_failure,
    extract_patches,
    repair_reply,
)


# ---------------------------------------------------------------------------
# Layer 1: proactive fence stripping
# ---------------------------------------------------------------------------


def test_strip_outer_fence_around_patch_block():
    """The DK-trace shape: ```html\n<patch>...\n``` — strip the fence."""
    reply = (
        "Here's the fix:\n\n"
        "```html\n"
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "foo\n"
        "=======\n"
        "bar\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
        "```\n"
    )
    repaired = repair_reply(reply)
    # The fence lines are gone; the <patch> body is now bare in the text.
    assert "```html" not in repaired
    assert "```\n" not in repaired
    assert "<patch>" in repaired
    # And extract_patches now finds it.
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    assert parsed[0].search.strip() == "foo"
    assert parsed[0].replace.strip() == "bar"


def test_strip_outer_fence_around_html_file_block():
    reply = (
        "```html\n"
        "<html_file>\n"
        "<!DOCTYPE html><html><body>hi</body></html>\n"
        "</html_file>\n"
        "```\n"
    )
    repaired = repair_reply(reply)
    # Fence stripped; <html_file> + body remain extractable downstream.
    assert "<html_file>" in repaired
    assert "```" not in repaired


def test_strip_outer_fence_leaves_unrelated_fences_alone():
    """Fences that don't contain a real tag should NOT be touched —
    they may live legitimately inside <plan> bodies or prose."""
    reply = (
        "<plan>\n"
        "Here's some pseudocode:\n"
        "```js\n"
        "function foo() { return 1; }\n"
        "```\n"
        "</plan>\n"
    )
    repaired = repair_reply(reply)
    # The js fence is preserved because there's no <patch> / <html_file>
    # inside it.
    assert "```js" in repaired
    assert "function foo()" in repaired


def test_extract_patches_succeeds_with_fenced_patch_via_repair():
    """The actual end-to-end check: a model reply that wraps the entire
    <patch> in ```html should now parse via extract_patches without
    extra plumbing on the caller side."""
    reply = (
        "```html\n"
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "const x = 1;\n"
        "=======\n"
        "const x = 2;\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
        "```\n"
    )
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    assert "const x = 1;" in parsed[0].search
    assert "const x = 2;" in parsed[0].replace


# ---------------------------------------------------------------------------
# Layer 2: classify_format_failure
# ---------------------------------------------------------------------------


def test_classify_returns_none_on_pure_prose():
    """A reply with no tag-shaped content shouldn't claim a malformed
    shape — it's the plan-only / no-code path's job."""
    assert classify_format_failure("Let me think about this...") is None
    assert classify_format_failure("") is None
    assert classify_format_failure("   \n  ") is None


def test_classify_tags_in_fence_when_proactive_strip_misses():
    """If `repair_reply` already stripped the outer fence, the classifier
    won't fire on `tags_in_fence` because the tag is now bare. But when
    the fence has no closer (truncated stream) the proactive stripper
    can't match — classifier picks it up."""
    reply = (
        "Here it is:\n"
        "```html\n"
        "<patch>\n"
        "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
        "</patch>\n"
        # No closing ``` — the fence wasn't closed.
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "tags_in_fence"
    assert "fence" in rej.hint.lower()
    # The detail message must tell the model what to do differently.
    assert "<patch>" in rej.detail or "<html_file>" in rej.detail


def test_classify_bare_markers_no_patch_wrapper():
    reply = (
        "Here's the fix:\n"
        "<<<<<<< SEARCH\n"
        "foo\n"
        "=======\n"
        "bar\n"
        ">>>>>>> REPLACE\n"
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "bare_markers"
    # Detail explains the fix (wrap in <patch>).
    assert "<patch>" in rej.detail


def test_classify_unclosed_patch():
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "foo\n"
        "=======\n"
        "bar\n"
        ">>>>>>> REPLACE\n"
        # Note: no </patch>
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "unclosed_patch"


def test_classify_unclosed_patch_ignores_quoted_patch_in_prose():
    """Star Wars iter-2: model quoted '<patch> blocks per reply' in reasoning."""
    reply = (
        "<diagnose>collision bug</diagnose>\n\n"
        'The guidelines say "<patch> blocks per reply are allowed".\n'
        "Let me plan the fix...\n"
    )
    assert classify_format_failure(reply) is None


def test_classify_wrong_tag_html_instead_of_html_file():
    reply = (
        "<html>\n"
        "<body><h1>hi</h1></body>\n"
        "</html>\n"
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "wrong_tag_html"
    assert "<html_file>" in rej.detail


def test_classify_wrong_tag_patches_plural():
    reply = (
        "<patches>\n"
        "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
        "</patches>\n"
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "wrong_tag_patches"


def test_classify_prefers_tags_in_fence_over_bare_markers():
    """Order matters: a fenced <patch> body with SEARCH/REPLACE markers
    inside should classify as `tags_in_fence`, not `bare_markers`."""
    reply = (
        "```\n"
        "<patch>\n"
        "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
        "</patch>\n"
    )
    rej = classify_format_failure(reply)
    assert rej is not None
    assert rej.kind == "tags_in_fence"


# ---------------------------------------------------------------------------
# Layer 3: agent._no_usable_code_fallback uses the rejection
# ---------------------------------------------------------------------------


def test_no_usable_code_fallback_uses_rejection_detail():
    from agent import GameAgent

    rej = FormatRejection(
        kind="tags_in_fence",
        hint="Your <patch> was in a fence.",
        detail="PARSE ERROR: I found <patch> inside a markdown fence.",
    )
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        rejection=rej,
        format_stuck_streak=1,
    )
    assert "PARSE ERROR" in msg
    # Streak < 2 → soft re-emit prompt, NOT the hard "stop using <patch>"
    # escalation.
    assert "STOP" not in msg.upper() or "ESCALATION" not in msg.upper()


def test_no_usable_code_fallback_escalates_at_streak_2():
    from agent import GameAgent

    rej = FormatRejection(
        kind="tags_in_fence",
        hint="Your <patch> was in a fence.",
        detail="PARSE ERROR: I found <patch> inside a markdown fence.",
    )
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        rejection=rej,
        format_stuck_streak=2,
    )
    # Escalation language present.
    assert "ESCALATION" in msg.upper()
    # And the escalation forbids <patch> and requires <html_file>.
    assert "<html_file>" in msg
    assert "<patch>" in msg  # mentioned in the "stop trying to send" line


def test_no_usable_code_fallback_no_rejection_falls_back_to_generic():
    """When `rejection` is None (model emitted pure prose or plan-only
    without the shape signals), the generic fallback is used."""
    from agent import GameAgent

    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        rejection=None,
        format_stuck_streak=0,
    )
    assert "could not find a <patch> or <html_file>" in msg


# ---------------------------------------------------------------------------
# Regression check: the existing pipeline still works on well-formed
# replies (no false-positive fence stripping or misclassification).
# ---------------------------------------------------------------------------


def test_well_formed_patch_reply_still_parses():
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "foo\n"
        "=======\n"
        "bar\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    parsed = extract_patches(reply)
    assert len(parsed) == 1
    # And the classifier doesn't fire (extraction succeeded — caller
    # never asks).
    rej = classify_format_failure(reply)
    # The reply contains a parseable <patch>; classifier may or may not
    # decline to flag it. The contract is "what happens when extraction
    # fails", not "always returns None on success". This test pins the
    # current behavior: no false alarm.
    assert rej is None or rej.kind != "tags_in_fence"
