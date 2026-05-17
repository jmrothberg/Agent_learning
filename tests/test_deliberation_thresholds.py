"""Tests for the deliberation guard thresholds (Item 2).

Trace evidence — `build-a-donkey-kong-clone-in-o_20260514_214747`
iter 1 and iter 3 both spent 1000+ lines of `<think>` reasoning
before emitting code (deliberation loops, model "thinking out loud"
without ever committing). The previous defaults (6000 outside-think
/ 15000 inside-think chars) caught them but only after ~200 lines —
several minutes of wall-clock per stuck iter.

Defaults are now 6000 / 12000. These tests pin the new defaults AND
the relationship `think_threshold >= threshold` (the inside-think
budget must be at least as large as the outside-think budget; the
detector internally clamps the inside threshold to max(threshold,
think_threshold), but pinning the input here documents the intent).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_io import DeliberationDetector  # noqa: E402


def test_default_thresholds_pinned_at_6000_and_12000():
    """The default thresholds — raised from 4000/8000 to 6000/12000
    after donkey-kong 20260516_170758 showed premature cutoffs on valid
    long transitions to code. If you change them, update this test and
    document the rationale in `DeliberationDetector.__init__`."""
    d = DeliberationDetector()
    assert d._threshold == 6000
    assert d._think_threshold == 12000


def test_inside_think_threshold_at_least_outside():
    """The constructor clamps inside-think threshold to be ≥ outside,
    so a misconfigured (low inside) threshold doesn't accidentally
    abort sooner inside <think> than outside. Pin this invariant."""
    d = DeliberationDetector(threshold_chars=2000, think_threshold_chars=1000)
    assert d._think_threshold >= d._threshold


def test_abort_after_threshold_chars_no_tag():
    """Outside <think>, ~`threshold_chars` of pre-tag rambling triggers
    abort. Feed 3999 chars (just under) — no abort. Then one more
    piece pushing over — abort fires."""
    d = DeliberationDetector(threshold_chars=6000, think_threshold_chars=12000)
    # 5900 chars of rambling, no tag opener.
    chunk = "a" * 100
    aborted = False
    for _ in range(59):
        if d.feed(chunk):
            aborted = True
            break
    assert aborted is False
    # One more 100-char piece pushes us over 6000. Should abort.
    if d.feed(chunk):
        aborted = True
    assert aborted is True
    assert d.stall_reason == "deliberation_loop"


def test_no_abort_when_tag_opens_before_threshold():
    """If the stream produces a known output-tag opener (e.g.
    `<patch>`, `<html_file>`, `<plan>`, …) BEFORE the threshold, the
    detector latches on it and stops counting."""
    d = DeliberationDetector(threshold_chars=6000, think_threshold_chars=12000)
    # Push 1000 chars of rambling.
    d.feed("a" * 1000)
    # Tag opens — detector latches.
    d.feed("here is the code: <html_file>")
    # Now push another 7000 chars; should NOT abort (latched).
    aborted = False
    for _ in range(70):
        if d.feed("a" * 100):
            aborted = True
            break
    assert aborted is False


def test_inside_think_higher_budget():
    """Inside `<think>...</think>` the budget is higher (12000 vs 6000).
    Feed ~4700 chars of `<think>` body — must NOT abort under the new
    12000-char inside-think threshold."""
    d = DeliberationDetector(threshold_chars=6000, think_threshold_chars=12000)
    # Enter <think>
    d.feed("<think>\n")
    aborted = False
    for _ in range(45):
        if d.feed("reasoning text " * 7):  # ~105 chars per chunk × 45 = ~4725
            aborted = True
            break
    # Under 8000 chars inside think — must not abort yet.
    assert aborted is False


def test_disabled_via_env_var(monkeypatch):
    """`DISABLE_DELIBERATION_DETECTOR=1` env var fully disables the
    abort path — used for A/B testing."""
    monkeypatch.setenv("DISABLE_DELIBERATION_DETECTOR", "1")
    d = DeliberationDetector(threshold_chars=10, think_threshold_chars=10)
    # Even with tiny thresholds, abort must not fire when disabled.
    for _ in range(100):
        assert d.feed("xxxxxxxxxx") is False  # 10 chars each, way over
