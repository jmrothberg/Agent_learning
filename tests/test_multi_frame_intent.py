"""Phase 0.8 — goal-text scanner for multi-frame asset intent.

The agent's planning policy used to silently force "one sprite per
visual entity (idle only)" even when the user explicitly asked for
animation frames in the goal text. This detector surfaces that intent
so the planner can flip its policy.

Tests are genre-free — example goals describe rendering modality
("walking sprites", "3 frames per enemy"), not subject matter.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts_v1 import _detect_multi_frame_intent  # noqa: E402


# Goals that MUST trigger multi-frame intent. Each entry is a tuple of
# (goal_text, at_least_one_of_these_substrings_should_appear_in_match).
_POSITIVE_GOALS = [
    (
        "smooth interpolated piece movement with idle bob and walk "
        "stride animations",
        ("walk", "stride", "idle"),
    ),
    (
        "each character has 3 sprites: idle, walking, attacking",
        ("3 sprites", "walking", "attacking", "idle"),
    ),
    (
        "spritesheet of 4 frames per enemy showing run cycle",
        ("4 frames", "frames", "run cycle", "run-cycle"),
    ),
    (
        "platformer with animated player — idle, walk, jump, fall, "
        "attack states",
        ("animation", "animated", "states", "walk", "jump"),
    ),
    (
        "make each enemy animated with multiple frames",
        ("animated", "multiple", "frames"),
    ),
    (
        "two sprites for each character",
        ("two sprites",),
    ),
    (
        "shooter with walk-cycle animation frames per soldier",
        ("walk-cycle", "frames"),
    ),
]


# Goals that MUST NOT trigger — they're single-frame requests or
# unrelated. Detector must be conservative; false positives shrink the
# iter-1 budget unnecessarily.
_NEGATIVE_GOALS = [
    "snake game with a single sprite per body segment",
    "tic-tac-toe with X and O glyphs",
    "minimax chess engine with no animations needed",
    "calculator app, DOM only",
    "color picker",
    "asteroids with bullets and rocks",   # single sprite per entity
    "space invaders with one ship and one alien type",
    # run_15 Dragon's Lair: probe-coaching "idle-wait" must NOT raise the
    # multi-frame asset cap (bare "idle" token was a false positive).
    "Dynamic probes must FORCE the event — never idle-wait for score.",
]


def test_idle_wait_probe_coaching_is_not_multi_frame():
    assert _detect_multi_frame_intent(
        "never idle-wait for score/food/collision"
    ) == []


def test_dragons_lair_canonical_goal_not_multi_frame_from_pose_words():
    """Pose words in the QTE goal are prompt-edit instructions, not a
    multi-frame roster ask. Cap must stay default unless frames/walk/etc."""
    from prompt_library import load_prompt_library
    goal = next(p["prompt"] for p in load_prompt_library() if p["name"] == "dragons-lair")
    assert "ASSET MUSTS" not in goal
    assert _detect_multi_frame_intent(goal) == []


def test_positive_multi_frame_goals_match():
    for goal, expected_substrings in _POSITIVE_GOALS:
        matched = _detect_multi_frame_intent(goal)
        assert matched, f"expected multi-frame intent for: {goal!r}"
        joined = " ".join(matched).lower()
        assert any(s in joined for s in expected_substrings), (
            f"goal {goal!r} matched {matched!r} but missed any of "
            f"{expected_substrings}"
        )


def test_negative_goals_do_not_match():
    for goal in _NEGATIVE_GOALS:
        matched = _detect_multi_frame_intent(goal)
        assert not matched, (
            f"goal {goal!r} must NOT trigger multi-frame intent — got {matched!r}"
        )


def test_empty_goal_returns_empty():
    assert _detect_multi_frame_intent("") == []
    assert _detect_multi_frame_intent(None) == []  # type: ignore[arg-type]


def test_returned_keywords_are_lowercase_and_dedup():
    # Same phrase repeated twice should appear ONCE in the match list.
    goal = "make each piece animated, with idle and walk and attack frames"
    matched = _detect_multi_frame_intent(goal)
    # All entries lowercase.
    assert all(m == m.lower() for m in matched)
    # No duplicates.
    assert len(matched) == len(set(matched))
