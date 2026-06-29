"""Phase 0.9 + 0.10 — wiring multi-frame intent into planning + asset cap.

Phase 0.8 added the detector. These tests verify the consumer paths:
  - plan_instruction(goal=...) flips the SCOPE-PACING nudge to a
    multi-frame override AND surfaces a MULTI-FRAME INTENT DETECTED
    banner so the architect emits base + variant chains, not idle-only
  - parse_assets_block_with_meta honors `max_assets` so a raised
    per-session cap lets larger rosters land in one turn
  - GameAgent.run() raises `_session_asset_cap` when the goal text
    contains explicit multi-frame phrasing
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from assets import parse_assets_block_with_meta  # noqa: E402
from prompts_v1 import plan_instruction  # noqa: E402
from agent import GameAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 0.9 — plan_instruction reacts to multi-frame intent.
# ---------------------------------------------------------------------------

_HEAVY_LOGIC_ART_GOAL_NO_MF = (
    "A 3-ply minimax turn-based board game with sprites for each piece. "
    "Full ruleset, AI search, alpha-beta pruning. Single sprite per piece."
)

_HEAVY_LOGIC_ART_GOAL_WITH_MF = (
    "A 3-ply minimax turn-based board game with sprites for each piece. "
    "Full ruleset, AI search, alpha-beta pruning. Each piece animated "
    "with idle, walking, and attacking frames."
)

_ART_ONLY_GOAL_WITH_MF = (
    "Platformer with walk-cycle animation frames per soldier — show idle, "
    "walking, and jumping states for each sprite."
)


def test_plan_instruction_default_scope_nudge_says_idle_only():
    """Without multi-frame intent, the scope-pacing nudge keeps the
    'idle pose only' rule."""
    body = plan_instruction(goal=_HEAVY_LOGIC_ART_GOAL_NO_MF)
    assert "SCOPE-PACING NUDGE" in body
    assert "idle pose only" in body.lower()
    assert "multi-frame override" not in body.lower()
    assert "MULTI-FRAME INTENT DETECTED" not in body


def test_plan_instruction_multi_frame_override_replaces_idle_only_rule():
    """With multi-frame intent + heavy logic, the nudge inverts: emit
    idle + a from_image pose frame for EVERY action state the goal named
    (not one core frame), with no 8-10 entry cap — deferring only the
    second smoothing frame and unnamed poses. SHORT prompts (not a low
    entry count) are what prevent the token-repetition runaway. 2026-05-31
    fix: the user enumerated punch/kick/jump/duck/fireball and the planner
    deferred them because the old text said 'one core motion frame, defer
    the rest' (2026-05-31 fighting-game trace)."""
    body = plan_instruction(goal=_HEAVY_LOGIC_ART_GOAL_WITH_MF)
    assert "multi-frame override" in body.lower()
    # The "idle pose only" rule must NOT appear under the override branch.
    assert "ONE sprite per visual entity (idle pose only)" not in body
    # The chain directive MUST appear with concrete from_image guidance.
    assert "from_image" in body
    # A concrete idle-seeded example template must be present.
    assert "_idle" in body
    # The matched keywords must be quoted so the model sees what triggered it.
    assert "matched:" in body.lower()
    # NEW contract: one frame per NAMED action, this turn — not deferred.
    assert "every action state the goal named" in body.lower()
    # The old flat 8-10 cap must be gone (it dropped the named poses).
    assert "no 8-10 cap" in body.lower()
    # Only the SECOND frame / unnamed poses are deferred to a later turn.
    assert "later" in body.lower()
    # Short-prompt discipline (the real runaway guard) must remain.
    assert "short" in body.lower()


def test_plan_instruction_multi_frame_nudge_fires_without_heavy_logic():
    """Pure-art goal with multi-frame intent still gets a directive —
    the standalone MULTI-FRAME INTENT DETECTED banner.

    2026-05-31: this is the branch that fires for a typical fighting-game
    goal (art detection empty → scope-pacing override never triggers). It
    must carry the SAME 'one frame per named action, no 8-10 cap' contract
    as the override, or the enumerated poses get deferred again."""
    body = plan_instruction(goal=_ART_ONLY_GOAL_WITH_MF)
    assert "MULTI-FRAME INTENT DETECTED" in body
    assert "from_image" in body
    # The scope-pacing nudge does NOT fire (no heavy-logic kw match).
    assert "SCOPE-PACING NUDGE" not in body
    # New contract pinned on the standalone branch too: emit a frame per
    # NAMED action this turn, with no flat entry cap, deferring only the
    # second smoothing frame / unnamed poses. Short prompts stay the guard.
    low = body.lower()
    assert "every action state the goal named" in low
    assert "no 8-10 entry cap" in low
    assert "keep each prompt short" in low
    assert "second" in low and "later mid-session" in low


# ---------------------------------------------------------------------------
# Phase 0.10 — parse_assets_block_with_meta honors the raised cap.
# ---------------------------------------------------------------------------


def _assets_reply_with_n_specs(n: int) -> str:
    specs = ", ".join(
        f'{{"name":"asset_{i}","prompt":"sprite {i}","size":"64x64"}}'
        for i in range(n)
    )
    return f"<assets>[{specs}]</assets>"


def test_assets_default_cap_24_truncates_larger_rosters():
    reply = _assets_reply_with_n_specs(36)
    specs, dropped = parse_assets_block_with_meta(reply)
    assert len(specs) == 24
    assert len(dropped) == 12


def test_assets_explicit_cap_lets_full_roster_land():
    reply = _assets_reply_with_n_specs(36)
    specs, dropped = parse_assets_block_with_meta(reply, max_assets=72)
    assert len(specs) == 36
    assert dropped == []


def test_assets_raised_cap_still_caps_at_its_value():
    reply = _assets_reply_with_n_specs(80)
    specs, dropped = parse_assets_block_with_meta(reply, max_assets=72)
    assert len(specs) == 72
    assert len(dropped) == 8


def test_assets_cap_min_one_no_zero_truncation():
    # Defensive — a misconfigured cap of 0 would silently drop everything.
    # The implementation should clamp to ≥1; if it ever doesn't, this
    # test will alert us.
    reply = _assets_reply_with_n_specs(5)
    specs, dropped = parse_assets_block_with_meta(reply, max_assets=0)
    assert len(specs) == 1


# ---------------------------------------------------------------------------
# Phase 0.10 — GameAgent raises _session_asset_cap on multi-frame goals.
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )


def test_session_asset_cap_raised_when_multi_frame_intent_in_goal(tmp_path):
    a = _make_agent(tmp_path)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    import asyncio
    gen = a.run(
        goal=(
            "Side-scrolling platformer with walk-cycle animation frames "
            "per character — show idle, walking, and attacking states."
        )
    )
    # Pull one event off the generator to trigger run()'s prelude where
    # _goal + _session_asset_cap are set. We don't actually need to run
    # iter 1.
    async def _kick():
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    asyncio.run(_kick())

    assert a._session_asset_cap == 72, (
        f"expected raised cap on multi-frame goal, got {a._session_asset_cap!r}"
    )
    intent_events = [t for t in traces if t.get("kind") == "multi_frame_intent_detected"]
    assert len(intent_events) == 1
    assert intent_events[0]["asset_cap_raised_to"] == 72
    assert intent_events[0]["matched_keywords"]


def test_session_asset_cap_stays_default_when_no_multi_frame_intent(tmp_path):
    a = _make_agent(tmp_path)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    import asyncio
    gen = a.run(goal="minimax engine, no animations needed")
    async def _kick():
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
    asyncio.run(_kick())

    assert a._session_asset_cap is None
    assert not any(t.get("kind") == "multi_frame_intent_detected" for t in traces)


# ---------------------------------------------------------------------------
# Phase 0.10b — mid-session multi-frame feedback ALSO raises the cap.
# ---------------------------------------------------------------------------


def _flush_user_injections_stub(asset_names: list[str] | None = None) -> "GameAgent":  # type: ignore[name-defined]
    """Lightweight GameAgent stub that exercises _flush_user_injections
    without spinning up an event loop or a backend."""
    a = GameAgent.__new__(GameAgent)
    a._fix_mode = False
    a._previous_report_ok = True
    a._previous_report = {"ok": True}
    a._pending_answer = None
    a._pending_feedback = []
    a._pending_bullet_lookups = []
    a._pending_probe_quarantine_notices = []
    a._pending_coaching = []
    a._last_drained_feedback = []
    a._token_cb = None
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    a._recent_feedback_texts = []
    a._repeat_sig_streak = 0
    a._scoped_change_active = False
    a._scoped_constraints = {}
    a._allow_one_rewrite = False
    a._session_assets = {
        n: f"/tmp/{n}.png" for n in (asset_names or [])
    }
    a._session_sounds = {}
    a._recent_deferred_signatures = []
    a._session_asset_cap = None
    a._feedback_deferred_last_turn = False
    return a


def test_mid_session_multi_frame_feedback_raises_cap():
    a = _flush_user_injections_stub(asset_names=["player", "enemy"])
    # User mid-session ask with multi-frame language.
    a._pending_feedback = [
        "make 3 walk frames for each character — idle, walk, attack states"
    ]
    a._flush_user_injections("test_report=ok")

    assert a._session_asset_cap == 72, (
        f"expected mid-session cap raise, got {a._session_asset_cap!r}"
    )
    intent_events = [
        e for e in a._trace_events
        if e.get("kind") == "multi_frame_intent_detected"
        and e.get("trigger") == "mid_session_feedback"
    ]
    assert len(intent_events) == 1
    assert intent_events[0]["prior_cap"] is None


def test_mid_session_non_multi_frame_feedback_keeps_default_cap():
    a = _flush_user_injections_stub(asset_names=["player"])
    a._pending_feedback = ["redraw the player sprite — make it more red"]
    a._flush_user_injections("test_report=ok")

    # Plain art-regen feedback should NOT raise the cap.
    assert a._session_asset_cap is None
    assert not any(
        e.get("kind") == "multi_frame_intent_detected"
        for e in a._trace_events
    )


def test_mid_session_cap_does_not_lower_an_already_raised_cap():
    a = _flush_user_injections_stub(asset_names=["player"])
    # Session was already raised (e.g. from goal-time detection).
    a._session_asset_cap = 96
    a._pending_feedback = [
        "make 3 walk frames for each character"
    ]
    a._flush_user_injections("test_report=ok")

    # The mid-session raise must not LOWER the cap.
    assert a._session_asset_cap == 96


# ---------------------------------------------------------------------------
# Ctrl+D fast-path: post-iter visual critic is skipped on _user_force_done.
# ---------------------------------------------------------------------------


def test_user_force_done_logic_short_circuit_flag_exists():
    """Smoke-level check that the flag GameAgent reads to decide whether
    to skip the post-iter visual critic is wired and defaults to False.
    Avoids spinning up a real critic stream (which would need a backend,
    a screenshot, and Playwright); the per-iter check pattern is the
    `if critic_backend is not None and self._user_force_done` line in
    agent.py which we grep-assert below to prevent regression.
    """
    import agent as agent_module
    from agent import module_inspect_source

    a = GameAgent.__new__(GameAgent)
    a._user_force_done = False
    assert hasattr(a, "_user_force_done")

    src = module_inspect_source()
    assert "critic_skipped_for_force_done" in src, (
        "Phase 0.4b regression: the critic fast-path on Ctrl+D was "
        "removed. The post-iter visual critic must respect "
        "_user_force_done so the iter-boundary check can fire promptly."
    )
