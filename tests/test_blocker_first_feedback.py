"""Regression tests for blocker-first feedback routing.

When a browser/micro-probe blocker is active, fresh user feedback should not
compete with the failing test report. This protects local models from chasing
the newest visual request while input/rendering is still broken.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent_stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._fix_mode = True
    a._previous_report_ok = False
    a._previous_report = {
        "ok": False,
        "soft_warnings": [
            "PROBE FAILED [input_responsive]: keyboard input must produce a visible canvas change",
        ],
    }
    a._pending_answer = None
    a._pending_feedback = ["change the pointer for the player; it is wrong"]
    a._pending_bullet_lookups = []
    a._pending_probe_quarantine_notices = []
    a._pending_coaching = []
    a._last_drained_feedback = []
    a._token_cb = None
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    # Defaults needed when the feedback-processing path runs (Phase 0.1
    # routes media-only feedback through this path even during a blocker).
    a._recent_feedback_texts = []
    a._repeat_sig_streak = 0
    a._scoped_change_active = False
    a._scoped_constraints = {}
    a._allow_one_rewrite = False
    a._session_assets = {}
    a._session_sounds = {}
    a._recent_deferred_signatures = []
    return a


def test_active_blocker_defers_feedback_without_consuming_queue() -> None:
    a = _agent_stub()

    prompt = a._flush_user_injections("FIX THE FAILING INPUT REPORT")

    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in prompt
    assert "FIX THE FAILING INPUT REPORT" in prompt
    assert "USER FEEDBACK (HIGHEST PRIORITY)" not in prompt
    assert a._pending_feedback == ["change the pointer for the player; it is wrong"]
    assert a._last_drained_feedback == []
    assert a._last_turn_contract["blocker_feedback_deferred"] is True
    assert any(e.get("kind") == "feedback_deferred_blocker" for e in a._trace_events)


def test_explicit_blocker_override_is_not_deferred() -> None:
    a = _agent_stub()
    a._pending_feedback = ["ignore the test failure and change the pointer"]

    assert a._should_defer_feedback_for_blocker() is False


# Regression: 2026-05-22 chess trace — user typed natural-language overrides
# three times ("the game works great", "do not change the GAME", "game plays
# fine") and all three were deferred behind a kbCursor draw warning. The
# narrow regex above this fix didn't recognise any of them. Each phrase
# below is taken verbatim or near-verbatim from real user feedback and must
# now bypass the deferral.
_NATURAL_LANGUAGE_OVERRIDES = [
    "the game plays fine",
    "the game works great",
    "it works just fine now",
    "the game is fine",
    "it is great",
    "the game plays again, just make the new assets",
    "do not change the GAME, just make the images",
    "don't change the game, only add the new sprites",
    "leave the gameplay alone, regenerate the sprites",
    "keep the code as-is",
    "the game is a perfectly working game already",
    "it is working perfectly, only fix the art",
]


def test_natural_language_blocker_overrides_are_not_deferred() -> None:
    for phrase in _NATURAL_LANGUAGE_OVERRIDES:
        a = _agent_stub()
        a._pending_feedback = [phrase]
        assert a._should_defer_feedback_for_blocker() is False, (
            f"natural-language override should bypass deferral: {phrase!r}"
        )


def test_unrelated_complaint_still_defers() -> None:
    # Negative control — phrases that should NOT trigger override.
    for phrase in [
        "change the player pointer color",
        "the enemy is too fast",
        "add a high-score screen",
        "make the music loop",
    ]:
        a = _agent_stub()
        a._pending_feedback = [phrase]
        assert a._should_defer_feedback_for_blocker() is True, (
            f"plain feature ask must still defer behind a blocker: {phrase!r}"
        )


# Phase 0.1 — media-only feedback must route through to the directive
# ladder even when a code blocker is active, because image/sound gen runs
# on GPU 0 in parallel with the coder on slot 1. The 2026-05-22 chess
# trace had the user ask three times for img2img animation frames seeded
# from existing piece sprites; all three were deferred. After this fix,
# the same asks should reach the model in the same turn the coder is
# fixing the unrelated draw warning.
def _media_aware_stub(
    feedback: list[str],
    asset_names: list[str] | None = None,
    sound_names: list[str] | None = None,
) -> GameAgent:
    a = _agent_stub()
    a._pending_feedback = list(feedback)
    a._session_assets = {n: f"/tmp/{n}.png" for n in (asset_names or [])}
    a._session_sounds = {n: f"/tmp/{n}.ogg" for n in (sound_names or [])}
    a._recent_feedback_texts = []
    a._recent_deferred_signatures = []
    return a


def test_art_only_feedback_routes_through_during_blocker() -> None:
    # Exact user feedback from the 2026-05-22 chess trace.
    chess_assets = [
        "white_pawn", "white_rook", "white_knight", "white_bishop",
        "white_queen", "white_king",
        "black_pawn", "black_rook", "black_knight", "black_bishop",
        "black_queen", "black_king",
    ]
    a = _media_aware_stub(
        feedback=[
            "make new assets of each of the existing pieces showing them "
            "walking, use the existing assets for each pawn, bishop, rook, "
            "knight, queen and king as starting point but then show each "
            "walking, and each smashing another piece",
        ],
        asset_names=chess_assets,
    )

    prompt = a._flush_user_injections("FIX THE FAILING DRAW WARNING")

    # The media-only ask reached the directive ladder, not just the
    # deferral block.
    assert "USER FEEDBACK (HIGHEST PRIORITY)" in prompt, (
        "media-only feedback must surface in the user-feedback block"
    )
    # And it should NOT be deferred (no code items in the partition).
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" not in prompt
    assert any(
        e.get("kind") == "media_only_parallel_inject" for e in a._trace_events
    )
    # Queue is drained for media items.
    assert a._pending_feedback == []


def test_pure_code_feedback_still_defers_during_blocker() -> None:
    # Pure behavior ask — no art noun, no asset-name match. Must defer.
    a = _media_aware_stub(
        feedback=["the AI is too slow, takes 30 seconds to move"],
        asset_names=["player"],
    )

    prompt = a._flush_user_injections("FIX THE FAILING REPORT")

    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in prompt
    assert "USER FEEDBACK (HIGHEST PRIORITY)" not in prompt
    # No media partition.
    assert not any(
        e.get("kind") == "media_only_parallel_inject" for e in a._trace_events
    )
    # Queue intact for next turn.
    assert a._pending_feedback == ["the AI is too slow, takes 30 seconds to move"]


def test_mixed_feedback_partitions_media_and_code_during_blocker() -> None:
    a = _media_aware_stub(
        feedback=[
            "make new sprites for each piece walking",   # media-only
            "the AI takes 30 seconds to move, that's a bug",   # code/behavior
        ],
        asset_names=["white_pawn", "black_pawn"],
    )

    prompt = a._flush_user_injections("FIX THE BLOCKER")

    # Both blocks must be present in the prompt.
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in prompt
    assert "USER FEEDBACK (HIGHEST PRIORITY)" in prompt
    # Deferral block must mention only the code item, not the sprite ask.
    defer_section = prompt.split("BLOCKER-FIRST FEEDBACK DEFERRAL", 1)[1]
    defer_section = defer_section.split("=================================================================", 1)[0]
    assert "AI takes 30 seconds" in defer_section
    assert "sprites" not in defer_section.lower()
    # And the parallel-note breadcrumb tells the model the partition happened.
    assert "media-only" in defer_section.lower() or "parallel" in defer_section.lower()
    # Code item stays queued for next turn.
    assert a._pending_feedback == ["the AI takes 30 seconds to move, that's a bug"]
    assert any(
        e.get("kind") == "media_only_parallel_inject"
        and e.get("media_count") == 1
        and e.get("code_deferred_count") == 1
        for e in a._trace_events
    )


def test_fuzzy_stem_resolution_for_pawn_king_pieces() -> None:
    # User says "pawn" / "king" — must resolve to the actual asset names
    # in the roster so the model knows which `from_image` parents to chain.
    from agent import _resolve_fuzzy_asset_stems

    assets = [
        "white_pawn", "black_pawn",
        "white_king", "black_king",
        "white_queen", "black_queen",
    ]
    out = _resolve_fuzzy_asset_stems(
        "make new sprites of each pawn and king walking", assets
    )
    assert out.get("pawn") == ["white_pawn", "black_pawn"], out
    assert out.get("king") == ["white_king", "black_king"], out
    # The user did NOT mention "queen" — must not appear.
    assert "queen" not in out

    # Colors alone do NOT trigger over-matching:
    out2 = _resolve_fuzzy_asset_stems(
        "the white background is wrong", assets
    )
    assert out2 == {}, out2


def test_media_change_directive_includes_stem_map_during_blocker() -> None:
    chess_assets = [
        "white_pawn", "black_pawn",
        "white_king", "black_king",
    ]
    a = _media_aware_stub(
        feedback=[
            "make new sprites of each pawn and king walking, use the "
            "existing assets as starting point",
        ],
        asset_names=chess_assets,
    )

    prompt = a._flush_user_injections("FIX THE BLOCKER")

    # Directive must surface the stem map so the model chains correctly.
    assert "MEDIA-CHANGE DIRECTIVE" in prompt
    assert "Stems the user referenced map to existing assets:" in prompt
    assert "'pawn' → [black_pawn, white_pawn]" in prompt
    assert "'king' → [black_king, white_king]" in prompt
    # And from_image reinforcement is present so the model uses img2img.
    assert "from_image" in prompt


def test_code_feedback_deferred_three_times_escalates_on_third_turn() -> None:
    # Simulate the chess scenario: a code-touching ask that keeps getting
    # deferred. Two prior deferrals → third attempt force-honors.
    a = _agent_stub()
    # Pure-behavior ask, no art noun, so it routes to code_to_defer.
    fb = "the AI is too slow, takes 30 seconds to move"

    # Turn 1: first defer.
    a._pending_feedback = [fb]
    p1 = a._flush_user_injections("FIX THE REPORT")
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in p1
    assert "FEEDBACK ESCALATION" not in p1

    # Turn 2: same feedback, second defer.
    a._pending_feedback = [fb]
    p2 = a._flush_user_injections("FIX THE REPORT")
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in p2
    assert "FEEDBACK ESCALATION" not in p2

    # Turn 3: same intent — must escalate, NOT defer.
    a._pending_feedback = [fb]
    p3 = a._flush_user_injections("FIX THE REPORT")
    assert "FEEDBACK ESCALATION" in p3, (
        "third deferral of the same intent must escalate"
    )
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" not in p3, (
        "escalated feedback must not also appear in the deferral block"
    )
    # Trace event for the escalation.
    assert any(
        e.get("kind") == "feedback_deferral_escalated"
        for e in a._trace_events
    )
    # Queue drained on the escalated turn.
    assert a._pending_feedback == []


def test_different_feedback_does_not_falsely_escalate() -> None:
    # First two turns: defer ask A. Third turn: brand new ask B —
    # signatures don't overlap, must defer normally (no escalation).
    a = _agent_stub()
    a._pending_feedback = ["the AI is too slow"]
    a._flush_user_injections("FIX THE REPORT")
    a._pending_feedback = ["the AI is too slow"]
    a._flush_user_injections("FIX THE REPORT")
    # New feedback, unrelated:
    a._pending_feedback = ["the music doesn't loop properly"]
    prompt = a._flush_user_injections("FIX THE REPORT")
    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in prompt
    assert "FEEDBACK ESCALATION" not in prompt


def test_behavior_bug_phrased_with_art_noun_still_defers() -> None:
    # Negative case: a phrase that mentions art nouns/asset names but is
    # clearly a behavior-bug complaint. Must defer, not route through —
    # protects against false partitioning that would push a real code bug
    # to the diffuser. `_feedback_is_behavior_bug` fires on explicit
    # complaint nouns ("broken", "frozen", "stuck", "glitching") and on
    # negation + behavior verbs.
    a = _media_aware_stub(
        feedback=["the player sprite is frozen and the game is broken"],
        asset_names=["player"],
    )

    a._flush_user_injections("FIX THE REPORT")

    # Behavior_bug classifier suppresses media routing, so this defers.
    assert a._pending_feedback == [
        "the player sprite is frozen and the game is broken"
    ]
    assert not any(
        e.get("kind") == "media_only_parallel_inject" for e in a._trace_events
    )
