"""Stall-recovery fallback tests (Phase 0D).

Motivating trace: Fieldrunners tower-defense 20260626_102307, GLM-5.2,
~15 tok/s.
  - iter 4: the model rambled 707s of pre-tag reasoning prose with NO output
    tag. The deliberation guard aborted -> no code saved. The user had asked
    for NEW enemy art + head-direction fixes.
  - iter 5: after a retry, the model bulk-emitted markdown SEARCH/REPLACE
    patches; the repeated config rows tripped a repetition loop -> aborted.

These assert that `_no_usable_code_fallback` produces ASSET-aware recovery
when art is pending (start with <assets>, few names, no inline data dumps)
and stays patch-focused otherwise. Pure-function: no model, no browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402

_fallback = GameAgent._no_usable_code_fallback


# ---------------------------------------------------------------------------
# Deliberation recovery
# ---------------------------------------------------------------------------


def test_deliberation_recovery_art_pending_steers_to_assets():
    """iter-4 shape: deliberation abort while NEW ART is pending -> the
    fallback must tell the model to START with <assets>, not patch-only."""
    msg, reset = _fallback(
        plan_only=True,
        has_existing_file=True,
        consecutive_plan_only=1,
        prior_stream_deliberated=True,
        art_pending=True,
    )
    assert "DELIBERATION RECOVERY" in msg
    assert "<assets>" in msg
    # Must not invite another reasoning essay.
    assert "first non-whitespace text MUST be `<assets>`" in msg
    assert reset is True


def test_deliberation_recovery_no_art_stays_patch_focused():
    """Deliberation abort with NO pending art -> diagnose+patch, no <assets>."""
    msg, reset = _fallback(
        plan_only=True,
        has_existing_file=True,
        consecutive_plan_only=1,
        prior_stream_deliberated=True,
        art_pending=False,
    )
    assert "DELIBERATION RECOVERY" in msg
    assert "<assets>" not in msg
    assert "<patch>" in msg or "<diagnose>" in msg
    assert reset is True


# ---------------------------------------------------------------------------
# Repetition-loop recovery
# ---------------------------------------------------------------------------


def test_loop_recovery_art_pending_steers_to_small_assets():
    """iter-5 shape: loop abort during bulk data while NEW ART is pending ->
    recover with ONE small <assets> block, explicitly NOT inline data dumps."""
    msg, _reset = _fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        prior_stream_looped=True,
        art_pending=True,
    )
    assert "REPETITION-LOOP RECOVERY" in msg
    assert "<assets>" in msg
    assert "inline" in msg.lower()


def test_loop_recovery_no_art_stays_minimal_patch():
    """Loop abort with NO pending art -> the existing minimal-<patch> path."""
    msg, _reset = _fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        prior_stream_looped=True,
        art_pending=False,
    )
    assert "REPETITION-LOOP RECOVERY" in msg
    assert "<patch>" in msg
    assert "<assets>" not in msg


def test_silent_recovery_unchanged():
    """Silent-stream recovery is independent of art and must still fire."""
    msg, _reset = _fallback(
        plan_only=True,
        has_existing_file=True,
        consecutive_plan_only=1,
        prior_stream_silent=True,
        art_pending=True,
    )
    assert "SILENT STREAM RECOVERY" in msg
