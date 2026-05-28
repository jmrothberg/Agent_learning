from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_vision_note_rejects_trace_fragments():
    assert GameAgent._clean_actionable_vision_note("- Image 1") == ""
    assert (
        GameAgent._clean_actionable_vision_note(
            "2. **Compare with the goal:** The goal is"
        )
        == ""
    )
    assert (
        GameAgent._clean_actionable_vision_note(
            "- The game state has progressed slightly (Lives went"
        )
        == ""
    )


def test_vision_note_keeps_actionable_feedback():
    assert (
        GameAgent._clean_actionable_vision_note(
            "- Player sprite is missing from the canvas"
        )
        == "Player sprite is missing from the canvas"
    )


# ---------- relevance gate: drop pure screenshot narration ----------------
#
# Evidence: fighing-game trace 20260519_153115. Two notes from the local
# VLM passed the prior filter even though they were purely descriptive —
# they told the coding model nothing actionable, and were paid context
# tokens with negative information value.


def test_vision_note_drops_iter3_controls_listed_at_bottom():
    """Trace iter 3 actual note. Pure layout description, no action."""
    assert (
        GameAgent._clean_actionable_vision_note(
            "Controls are listed at the bottom"
        )
        == ""
    )


def test_vision_note_drops_iter4_low_resolution_style():
    """Trace iter 4 actual note. Style description, no action."""
    assert (
        GameAgent._clean_actionable_vision_note(
            "Both images show very low-resolution, pixel-art style"
        )
        == ""
    )


def test_vision_note_keeps_iter1_no_complex_super():
    """Trace iter 1 actual note (paraphrased to fit one line). The
    'no complex' negation IS actionable — model can infer 'add complex'.
    The relevance gate must keep this."""
    assert (
        GameAgent._clean_actionable_vision_note(
            "look very basic, just standing poses. There are no complex super"
        )
        != ""
    )


def test_vision_note_keeps_explicit_change_verb():
    """`add`, `fix`, `make`, etc. are explicit actionable verbs."""
    assert (
        GameAgent._clean_actionable_vision_note(
            "Add a score display to the top of the canvas"
        )
        != ""
    )
    assert (
        GameAgent._clean_actionable_vision_note(
            "Fix the player sprite scale"
        )
        != ""
    )


def test_vision_note_drops_purely_descriptive():
    """Generic descriptive note with no actionable token gets dropped."""
    assert (
        GameAgent._clean_actionable_vision_note(
            "The game shows two characters facing each other"
        )
        == ""
    )
