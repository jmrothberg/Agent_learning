"""Phase 0.13 + 0.14 — immediate fixes for the 2026-05-22 third chess trace
failures.

Phase 0.13: when the model emits a mid-session <assets> block AND the
user has queued feedback asking for a style rebrand, the agent must
DEFER the asset gen and queue a coaching note. Generating sprites
guaranteed to be replaced burns ~3 s per sprite × N entries with zero
useful output.

Phase 0.14: when a single stream emits > 15k completion tokens, the
agent must emit a `runaway_stream_warning` trace event so the user can
see it in the TUI and Ctrl+D out instead of waiting silently.

All checks are general (no genre logic).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent_with_mid_session_state(tmp_path: Path) -> GameAgent:
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    # Mid-session state — already have some assets generated.
    a._session_assets = {
        "white_pawn_idle": Path("/tmp/white_pawn_idle.png"),
        "white_pawn_walk": Path("/tmp/white_pawn_walk.png"),
        "white_pawn_smash": Path("/tmp/white_pawn_smash.png"),
        "black_pawn_idle": Path("/tmp/black_pawn_idle.png"),
    }
    a._current_file = "<html><body><canvas></canvas></body></html>"
    # Ensure no global config interference.
    a._scoped_constraints = {}
    return a


def test_phase_0_13_defers_mid_session_assets_when_user_wants_style_rebrand(tmp_path):
    a = _agent_with_mid_session_state(tmp_path)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)
    # User typed mid-session: clearly a style rebrand.
    a._pending_feedback = [
        "All of the images for the chess pieces need to look like fantasy monsters, "
        "not regular chess pieces"
    ]
    a._pending_coaching = []

    # Model emits <assets> for the missing names. Identical style to
    # what's already on disk — guaranteed-to-be-discarded if we let it
    # generate.
    reply = (
        "<assets>["
        '{"name":"black_pawn_walk","prompt":"pixel-art black chess pawn mid-step"},'
        '{"name":"black_pawn_smash","prompt":"pixel-art black chess pawn impact"}'
        "]</assets>"
    )

    async def _drive():
        events = []
        async for ev in a._maybe_generate_assets_and_sounds(
            reply, trigger="mid_session"
        ):
            events.append(ev)
        return events

    asyncio.run(_drive())

    # Deferred — no new files generated, coaching queued for next turn.
    deferral_events = [
        t for t in traces
        if t.get("kind") == "mid_session_assets_deferred_for_user_style"
    ]
    assert len(deferral_events) == 1, (
        "deferral event must fire exactly once when a style-rebrand "
        "intent is pending and the model emits mid-session <assets>"
    )
    assert deferral_events[0]["deferred_names"] == ["black_pawn_walk", "black_pawn_smash"]
    # Coaching queued.
    assert any(
        "MID-SESSION ASSET DEFERRAL" in c for c in a._pending_coaching
    )
    # User's feedback STAYS in the queue — Phase 0.1's partitioner picks
    # it up at the next user-turn boundary.
    assert len(a._pending_feedback) == 1


def test_phase_0_13_does_not_defer_when_no_style_rebrand_pending(tmp_path):
    a = _agent_with_mid_session_state(tmp_path)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)
    # Pending feedback is an unrelated behavior bug — must NOT defer.
    a._pending_feedback = ["the AI is too slow, takes 30 seconds to move"]
    a._pending_coaching = []
    # Mock the image generator so we don't try to load real diffusers.
    a._asset_generator = None  # forces _maybe_generate_assets_and_sounds early-exit
    # Actually let's not get into the gen path — we just check the
    # deferral didn't fire by emitting a tiny reply.
    reply = (
        "<assets>["
        '{"name":"black_pawn_walk","prompt":"pixel-art black chess pawn mid-step"}'
        "]</assets>"
    )

    async def _drive():
        try:
            async for _ in a._maybe_generate_assets_and_sounds(
                reply, trigger="mid_session"
            ):
                pass
        except Exception:
            # Diffuser path may fail in a stubbed env — we only care about
            # the trace event from the deferral check, which fires before
            # any gen.
            pass

    asyncio.run(_drive())

    assert not any(
        t.get("kind") == "mid_session_assets_deferred_for_user_style"
        for t in traces
    ), "deferral must not fire on non-rebrand feedback"


def test_phase_0_13_skipped_on_phase_a(tmp_path):
    # Phase A asset gen is the architect's initial roster — there CAN'T
    # be queued user feedback because the goal text has been received but
    # no user has typed yet. We still test that the deferral never fires
    # on phase_a even with queued feedback (a defensive check).
    a = _agent_with_mid_session_state(tmp_path)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)
    a._pending_feedback = ["make all the sprites look like monsters"]
    a._pending_coaching = []
    reply = "<assets>[" '{"name":"player","prompt":"player sprite"}' "]</assets>"

    async def _drive():
        try:
            async for _ in a._maybe_generate_assets_and_sounds(
                reply, trigger="phase_a"
            ):
                pass
        except Exception:
            pass

    asyncio.run(_drive())

    assert not any(
        t.get("kind") == "mid_session_assets_deferred_for_user_style"
        for t in traces
    ), "deferral must only fire on mid_session trigger"


# ---------------------------------------------------------------------------
# Phase 0.14 — runaway stream warning
# ---------------------------------------------------------------------------


def test_phase_0_14_runaway_warning_threshold():
    """The runaway-token floor must be reasonable — not so low it spams
    on normal large rewrites, not so high it misses real runaways. The
    2026-05-22 trace had a 36,736-token stream; the threshold must fire
    well before that."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    # Verify the floor and the warning kind appear together.
    assert "_RUNAWAY_TOKEN_FLOOR" in src
    assert "runaway_stream_warning" in src
    # Floor is between 10k and 25k — a typical iter is 1-4k; a giant
    # legitimate rewrite is ~10k. 15k is the chosen middle ground.
    import re
    m = re.search(r"_RUNAWAY_TOKEN_FLOOR\s*=\s*(\d+)", src)
    assert m is not None
    floor = int(m.group(1))
    assert 10_000 <= floor <= 25_000, (
        f"runaway floor of {floor} is outside the sensible 10k-25k range — "
        "either too aggressive (will spam normal large rewrites) or too "
        "lax (will miss the 25-minute / 36k-token runaway the trace showed)"
    )


def test_phase_0_14_warning_fires_once_per_stream():
    """The runaway flag is fire-once per stream (`runaway_warned` in
    hb_state) so a single stream doesn't spam multiple warnings every
    30s heartbeat. Regression guard."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    assert '"runaway_warned": False' in src, (
        "runaway-warned fire-once flag must be initialized to False"
    )
    # And the warning emission must guard on it.
    assert 'not hb_state["runaway_warned"]' in src, (
        "runaway emission must check the fire-once flag to avoid spam"
    )
