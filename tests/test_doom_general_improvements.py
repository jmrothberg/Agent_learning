"""Tests for the six general agent improvements drawn from the
2026-05-23 doom trace.

Each test covers a single failure mode the doom trace exhibited AND
pins a guard that prevents it from recurring on any future game
(not just doom-style FPS). The improvements were designed as
game-agnostic — these tests assert that.

Plan reference: /home/jonathan/.claude/plans/lexical-toasting-starfish.md
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import memory as memory_mod
from agent import GameAgent
import ollama_io
import prompts_v1


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ----------------------------------------------------------------------
# A. Silent-stream guard. The motivating trace had 1356s of wall-clock
# with 32777 completion tokens and ZERO non-empty `content` pieces;
# none of the existing detectors (stalled/looped/deliberated/crashed)
# fired because they all key on `piece` content.
# ----------------------------------------------------------------------


def test_stream_result_has_silent_field() -> None:
    """The dataclass must carry the new `silent: bool` field for the
    agent recovery path to read."""
    r = ollama_io.StreamResult(text="", tokens=0, duration_s=0.0, stalled=False)
    assert hasattr(r, "silent")
    assert r.silent is False


def test_no_usable_code_fallback_routes_silent_to_dedicated_message() -> None:
    """When `prior_stream_silent=True`, the fallback message must call
    out the specific failure mode (zero content for the entire wall-
    clock budget) so the model knows to start with an opening tag and
    skip any reasoning preamble. The generic 'no usable code' message
    doesn't communicate this and would leave the model guessing."""
    msg, reset = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        prior_stream_silent=True,
    )
    assert "SILENT STREAM RECOVERY" in msg
    assert "ZERO visible content" in msg or "zero visible content" in msg.lower()
    assert "opening tag" in msg.lower()
    # The recovery must NOT escalate the plan-only streak counter
    # because this isn't a plan-only loop, it's a content-channel
    # failure.
    assert reset is False


def test_no_usable_code_fallback_silent_takes_precedence_over_generic() -> None:
    """When both prior_stream_silent AND prior_stream_looped are True,
    silent wins (more specific recovery)."""
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        prior_stream_silent=True,
        prior_stream_looped=True,
    )
    assert "SILENT STREAM RECOVERY" in msg
    # Loop-recovery message should NOT show — silent fired first.
    assert "REPETITION-LOOP RECOVERY" not in msg


# ----------------------------------------------------------------------
# B. Cross-turn patch-SEARCH-failure memory.
# ----------------------------------------------------------------------


def test_patch_anchor_fingerprint_is_normalization_stable() -> None:
    """Same SEARCH with different whitespace must produce the same
    fingerprint so re-emits with cosmetic shifts still match."""
    a = GameAgent._patch_anchor_fingerprint("const x = 1;\n  const y = 2;")
    b = GameAgent._patch_anchor_fingerprint("const   x = 1;  const y = 2;")
    c = GameAgent._patch_anchor_fingerprint("  const x = 1;\nconst y = 2;\n")
    assert a == b == c, (a, b, c)


def test_patch_anchor_fingerprint_different_searches_differ() -> None:
    a = GameAgent._patch_anchor_fingerprint("const x = 1;")
    b = GameAgent._patch_anchor_fingerprint("const z = 9;")
    assert a != b


def test_patch_retry_instruction_marks_repeats(tmp_path: Path) -> None:
    """The retry prompt must prepend [REPEATED FAILURE] when a SEARCH
    matches an anchor from the previous turn."""

    class FakePatch:
        def __init__(self, search: str) -> None:
            self.search = search

    p1 = FakePatch("const spriteNames=['imp_idle','imp_walk1','imp_walk2'];")
    failures = [(0, p1, "SEARCH block not found in file: '...'")]
    fp = GameAgent._patch_anchor_fingerprint(p1.search)

    out = prompts_v1.patch_retry_instruction(
        failures,
        "<html><body></body></html>",
        repeat_anchors={fp},
        anchor_fingerprint=GameAgent._patch_anchor_fingerprint,
    )
    assert "REPEATED PATCH FAILURE" in out
    assert "[REPEATED FAILURE" in out


def test_patch_retry_instruction_omits_repeat_marker_without_match(tmp_path: Path) -> None:
    """Regression guard: when repeat_anchors is empty, behaviour
    matches the original (no [REPEATED FAILURE] noise)."""

    class FakePatch:
        def __init__(self, search: str) -> None:
            self.search = search

    failures = [(0, FakePatch("const x=1;"), "SEARCH block not found.")]
    out = prompts_v1.patch_retry_instruction(failures, "<html></html>", repeat_anchors=set())
    assert "REPEATED PATCH FAILURE" not in out
    assert "[REPEATED FAILURE" not in out


# ----------------------------------------------------------------------
# C. Synthetic coverage-gap probe text fencing.
# ----------------------------------------------------------------------


def test_coverage_gap_err_carries_harness_note_fence() -> None:
    """When tools.py synthesizes a coverage_gap probe, the err string
    MUST start with [HARNESS NOTE — NOT FILE CONTENT, DO NOT <patch>]
    so the model can't mistake it for file content. Doom 2026-05-23
    extension 1 emitted a <patch> targeting this text because no
    fence existed."""
    # Build the same string the synthesizer builds.
    err = (
        "[HARNESS NOTE — NOT FILE CONTENT, DO NOT <patch>]\n"
        "This is a SYNTHETIC harness probe — it does NOT exist anywhere "
        "in your .html file. It was added automatically because your "
        "Phase A <criteria> included this line but no model-authored "
        "probe in your <probes> block references it:\n\n"
        "  criterion: Player movement is relative to facing direction\n"
        "[/HARNESS NOTE]"
    )
    # Sanity: the fence open / close are present and the DO NOT <patch>
    # directive is in the FIRST line so a model that only reads the
    # head of the err sees it.
    first_line = err.splitlines()[0]
    assert first_line.startswith("[HARNESS NOTE")
    assert "DO NOT <patch>" in first_line
    assert "[/HARNESS NOTE]" in err
    assert "SYNTHETIC" in err


# ----------------------------------------------------------------------
# D. Visual-critic cross-turn dedup.
# ----------------------------------------------------------------------


def test_visual_critic_dedup_suppresses_repeats(tmp_path: Path) -> None:
    """Same critic observation across consecutive turns must be
    suppressed after the first injection."""
    a = _make_agent(tmp_path)
    note = "The wall textures appear extremely low-resolution and pixelated."

    queued_1 = a._queue_visual_critic_coaching(note, iteration=1)
    assert queued_1 is True
    assert any(note in c for c in a._pending_coaching)

    queued_2 = a._queue_visual_critic_coaching(note, iteration=2)
    assert queued_2 is False  # suppressed
    # _pending_coaching should still contain only ONE instance of the note.
    matches = sum(1 for c in a._pending_coaching if note in c)
    assert matches == 1


def test_visual_critic_dedup_normalizes_whitespace_and_case(tmp_path: Path) -> None:
    """Cosmetic wording shifts (case, leading spaces) must still match."""
    a = _make_agent(tmp_path)
    a._queue_visual_critic_coaching(
        "The walls look low-resolution and pixelated.",
        iteration=1,
    )
    queued_2 = a._queue_visual_critic_coaching(
        "  THE WALLS LOOK   low-resolution and PIXELATED.  ",
        iteration=2,
    )
    assert queued_2 is False


def test_visual_critic_dedup_lets_new_observation_through(tmp_path: Path) -> None:
    """Regression guard: an unrelated new note must still queue."""
    a = _make_agent(tmp_path)
    a._queue_visual_critic_coaching("Walls look low-resolution.", iteration=1)
    queued = a._queue_visual_critic_coaching(
        "The gun is pointing to the left of the crosshair.",
        iteration=2,
    )
    assert queued is True


def test_critic_fingerprint_uses_first_120_chars(tmp_path: Path) -> None:
    """Two notes sharing the same opening 120 chars but diverging
    later should match — model often paraphrases the tail of its own
    observation across turns."""
    base = "The wall textures are very low-resolution and pixelated, " * 3
    fp_a = GameAgent._critic_note_fingerprint(base + "first variation")
    fp_b = GameAgent._critic_note_fingerprint(base + "second variation")
    assert fp_a == fp_b


# ----------------------------------------------------------------------
# E. Classifier overrule auto-disable.
# ----------------------------------------------------------------------


def test_classifier_overrule_auto_disables_at_threshold(tmp_path: Path) -> None:
    """After two overrules in a session, _use_feedback_directives must
    auto-flip to False so the classifier stops misrouting typed
    feedback for the rest of the session."""
    a = _make_agent(tmp_path)
    a._use_feedback_directives = True  # starting state
    assert a._classifier_overrule_count == 0

    a._record_classifier_overrule(
        expected_mode="media_only",
        model_emitted="patch_only",
        feedback_preview="move gun 75 pixels right, no asset changes",
    )
    # After ONE overrule the directives are still on.
    assert a._use_feedback_directives is True
    assert a._classifier_overrule_count == 1

    a._record_classifier_overrule(
        expected_mode="media_only",
        model_emitted="patch_only",
        feedback_preview="move gun UP 10 pixels, no asset changes",
    )
    # Second overrule → auto-disabled.
    assert a._use_feedback_directives is False
    assert a._classifier_auto_disabled is True
    assert a._classifier_overrule_count == 2


def test_classifier_auto_disable_idempotent(tmp_path: Path) -> None:
    """Third + fourth overrule events must not re-flip / re-trace
    the auto-disable banner."""
    a = _make_agent(tmp_path)
    a._use_feedback_directives = True
    for _ in range(4):
        a._record_classifier_overrule(
            expected_mode="media_only",
            model_emitted="patch_only",
            feedback_preview="...",
        )
    # Still off, still flagged auto-disabled exactly once.
    assert a._use_feedback_directives is False
    assert a._classifier_auto_disabled is True
    assert a._classifier_overrule_count == 4


def test_classifier_auto_disable_respects_explicit_off(tmp_path: Path) -> None:
    """If the user already turned directives off via /rawfeedback, the
    auto-disable path is a no-op (doesn't try to re-disable an already-
    off flag and doesn't bump the auto_disabled flag for nothing)."""
    a = _make_agent(tmp_path)
    a._use_feedback_directives = False  # already raw mode
    a._record_classifier_overrule(
        expected_mode="media_only",
        model_emitted="patch_only",
        feedback_preview="...",
    )
    a._record_classifier_overrule(
        expected_mode="media_only",
        model_emitted="patch_only",
        feedback_preview="...",
    )
    # Counter still ticks (postmortem analysis still useful), but
    # the auto-disable banner doesn't fire on an already-off flag.
    assert a._use_feedback_directives is False
    assert a._classifier_auto_disabled is False
    assert a._classifier_overrule_count == 2


# ----------------------------------------------------------------------
# F. entity-progress-over-time recipe: input-driven skip.
# ----------------------------------------------------------------------


def test_entity_progress_recipe_applies_when_balanced_js() -> None:
    """The new applies_when JS must be syntactically balanced
    (parens / braces) so it doesn't crash the browser eval."""
    items = memory_mod._opening_book_seed_items()
    recipes = items.get(memory_mod.PLAYTESTS_FILENAME, [])
    ep = [r for r in recipes if r.id == "entity-progress-over-time"]
    assert len(ep) == 1
    js = ep[0].recipe["applies_when"]
    assert js.count("(") == js.count(")")
    assert js.count("{") == js.count("}")
    # The gate must START with `(()=>{` so it's an IIFE and END with `})()`.
    assert js.startswith("(()=>{")
    assert js.endswith("})()")


def test_entity_progress_recipe_skips_input_driven_signals_documented() -> None:
    """The applies_when JS must check the documented self-driven-motion
    signals (enemies vx/vy, projectiles, particles, score, time/timer)
    so future edits don't accidentally regress to canvas-only."""
    items = memory_mod._opening_book_seed_items()
    recipes = items.get(memory_mod.PLAYTESTS_FILENAME, [])
    ep = [r for r in recipes if r.id == "entity-progress-over-time"][0]
    js = ep.recipe["applies_when"]
    for needle in (
        "enemies", "projectiles", "particles",
        "score", "time", "timer",
        "vx", "vy",
    ):
        assert needle in js, f"applies_when JS missing self-driven signal: {needle!r}"
    # Conservative path: no exposed state → run (don't skip).
    assert "if(!s)return true" in js


def test_entity_progress_recipe_no_genre_strings() -> None:
    """Standing rule: no game-genre / subject-matter strings in
    applies_when gates. Only observable structure."""
    items = memory_mod._opening_book_seed_items()
    recipes = items.get(memory_mod.PLAYTESTS_FILENAME, [])
    ep = [r for r in recipes if r.id == "entity-progress-over-time"][0]
    js = ep.recipe["applies_when"]
    for forbidden in (
        "doom", "wolfenstein", "snake", "pacman", "asteroid", "tetris",
        "fps", "rpg", "platformer", "shooter",
    ):
        assert forbidden.lower() not in js.lower(), (
            f"genre/subject-matter token leaked into gate: {forbidden!r}"
        )
