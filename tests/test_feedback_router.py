"""LLM Feedback Router (chess-trace fix 2026-06-22).

The router interprets a user-feedback batch into a routing decision that
OVERRIDES the brittle regex classifiers. These tests cover the pure /
deterministic parts (no live model):

  1. `_parse_feedback_route_json` — tolerant JSON extraction + validation.
  2. Deferral reform — a route with `honor_user_now` stops the blocker
     deferral; `defer_behind_blocker` forces it (the chess-trace iter-3
     bug: "no new assets, just show the full screen the bottom row is cut
     off" was deferred behind a stale blocker for three turns).
  3. Stale asset-reprompt clear — a route that says NO new art clears an
     outstanding `_unhonored_asset_request` (the chess-trace attempt-3
     reprompt still quoted the iter-2 message after "no new assets").

All checks are genre-free.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    a._trace = lambda obj: None
    return a


# ---- 1. JSON parsing -------------------------------------------------

def test_parse_route_plain_json(tmp_path):
    a = _agent(tmp_path)
    txt = (
        '{"primary_intent": "code_fix", "honor_user_now": true, '
        '"allow_assets_block": false, "allow_patch": true, '
        '"defer_behind_blocker": false, "user_visible_issue": "row clipped", '
        '"harness_blocker_ack": "", "confidence": 0.9}'
    )
    route = a._parse_feedback_route_json(txt)
    assert route is not None
    assert route["primary_intent"] == "code_fix"
    assert route["honor_user_now"] is True
    assert route["allow_assets_block"] is False


def test_parse_route_fenced_json(tmp_path):
    a = _agent(tmp_path)
    txt = (
        "Here is the routing:\n```json\n"
        '{"primary_intent": "generate_new_assets", "allow_assets_block": true}\n'
        "```\n"
    )
    route = a._parse_feedback_route_json(txt)
    assert route is not None
    assert route["primary_intent"] == "generate_new_assets"
    assert route["allow_assets_block"] is True
    # Missing fields fall back to safe defaults.
    assert route["honor_user_now"] is True


def test_parse_route_prose_wrapped(tmp_path):
    a = _agent(tmp_path)
    txt = (
        "The user is reporting a layout problem. "
        '{"primary_intent":"code_fix","allow_assets_block":false} '
        "That is my decision."
    )
    route = a._parse_feedback_route_json(txt)
    assert route is not None
    assert route["primary_intent"] == "code_fix"


def test_parse_route_invalid_returns_none(tmp_path):
    a = _agent(tmp_path)
    assert a._parse_feedback_route_json("") is None
    assert a._parse_feedback_route_json("no json here at all") is None
    # Unknown intent is rejected.
    assert a._parse_feedback_route_json('{"primary_intent": "nonsense"}') is None


# ---- 2. Deferral reform ----------------------------------------------

def _arm_blocker(a: GameAgent) -> None:
    # Active blocker = fix mode after a failed report.
    a._fix_mode = True
    a._previous_report_ok = False


def test_route_honor_now_stops_deferral(tmp_path):
    a = _agent(tmp_path)
    _arm_blocker(a)
    a._pending_feedback = [
        "no new assets, just show the full screen the bottom row is cut off"
    ]
    a._feedback_route = {
        "primary_intent": "code_fix",
        "honor_user_now": True,
        "allow_assets_block": False,
        "defer_behind_blocker": False,
    }
    assert a._should_defer_feedback_for_blocker() is False


def test_route_defer_behind_blocker_forces_deferral(tmp_path):
    a = _agent(tmp_path)
    _arm_blocker(a)
    a._pending_feedback = ["fix the crash first, then we'll talk layout"]
    a._feedback_route = {
        "primary_intent": "code_fix",
        "honor_user_now": False,
        "allow_assets_block": False,
        "defer_behind_blocker": True,
    }
    assert a._should_defer_feedback_for_blocker() is True


def test_no_route_falls_back_to_regex(tmp_path):
    # Without a route, the legacy regex override behavior is preserved:
    # generic feedback under a blocker still defers.
    a = _agent(tmp_path)
    _arm_blocker(a)
    a._pending_feedback = ["the bottom row is cut off"]
    a._feedback_route = None
    assert a._should_defer_feedback_for_blocker() is True


# ---- 3. Stale asset-reprompt clear -----------------------------------

def test_route_no_assets_clears_stale_reprompt(tmp_path):
    a = _agent(tmp_path)
    a._session_assets = {"white_pawn_idle": Path("/tmp/white_pawn_idle.png")}
    # A stale outstanding asset request from an earlier turn.
    a._unhonored_asset_request = "make the pieces look like dragons"
    a._asset_reprompt_count = 2
    a._pending_feedback = ["no new assets, just show the full screen"]
    a._feedback_route = {
        "primary_intent": "code_fix",
        "honor_user_now": True,
        "allow_assets_block": False,
        "defer_behind_blocker": False,
    }
    out = a._flush_user_injections("REPORT: still failing")
    # The contradicted reprompt is cleared, and no ASSET GENERATION
    # REQUIRED banner is emitted this turn.
    assert a._unhonored_asset_request is None
    assert a._asset_reprompt_count == 0
    assert "ASSET GENERATION REQUIRED" not in out


def test_route_retains_reprompt_on_vague_retry(tmp_path):
    """Phase 0D-6 (Fieldrunners trace 20260626_102307 iter 5): a CONTENT-FREE
    retry nudge routes as code_fix / allow_assets_block=false, but it does NOT
    contradict a still-unhonored art request. The standing request must be
    RETAINED (and re-armed this turn), not silently dropped."""
    a = _agent(tmp_path)
    a._session_assets = {"missile_head_n": Path("/tmp/missile_head_n.png")}
    a._unhonored_asset_request = "give each tower its own unique head sprite"
    a._asset_reprompt_count = 1
    a._pending_feedback = ["no usable code was identified by the agent, try again"]
    a._feedback_route = {
        "primary_intent": "code_fix",
        "honor_user_now": True,
        "allow_assets_block": False,
        "defer_behind_blocker": False,
    }
    out = a._flush_user_injections("REPORT: still failing")
    # The standing art request survives the vague retry.
    assert a._unhonored_asset_request == "give each tower its own unique head sprite"
    # And it is re-surfaced this turn so the <assets> scaffold fires.
    assert "ASSET GENERATION REQUIRED" in out


def test_route_wants_assets_arms_reprompt(tmp_path):
    a = _agent(tmp_path)
    a._session_assets = {"white_pawn_idle": Path("/tmp/white_pawn_idle.png")}
    a._unhonored_asset_request = None
    a._asset_reprompt_count = 0
    a._pending_feedback = ["add a brand new red dragon boss sprite"]
    a._feedback_route = {
        "primary_intent": "generate_new_assets",
        "honor_user_now": True,
        "allow_assets_block": True,
        "defer_behind_blocker": False,
    }
    out = a._flush_user_injections("REPORT: ok")
    assert "ASSET GENERATION REQUIRED" in out
    assert a._unhonored_asset_request is not None


# ---- 4. DOOM3DF3 20260911: state_fields + repeat-complaint escalation ----

_DOOM_LIKE_HTML = """<html><script>
window.state = {phase:'play', player:{x:1,z:2,yaw:0,pitch:0}};
const p = state.player;
document.addEventListener('mousemove', (e) => {
  p.yaw -= e.movementX * 0.002;
  p.pitch -= e.movementY * 0.002;
});
function syncCamera(){ camera.rotation.y = p.yaw; }
function update(dt){
  if(state.phase!=='play') return;
  p.yaw=camera.rotation.y; p.pitch=camera.rotation.x;
  syncCamera();
}
</script></html>"""


def test_parse_route_extracts_state_fields(tmp_path):
    a = _agent(tmp_path)
    txt = (
        '{"primary_intent":"code_fix","state_fields":["player.yaw","player.pitch",'
        '"bad field!", 42, "a.b.c.d.e.f"]}'
    )
    route = a._parse_feedback_route_json(txt)
    assert route["state_fields"] == ["player.yaw", "player.pitch", "a.b.c.d.e.f"]
    # Missing → empty list, never None.
    assert a._parse_feedback_route_json('{"primary_intent":"code_fix"}')["state_fields"] == []


def test_static_state_writer_lines_resolves_alias_and_names_frame_writer():
    """The focused slice only saw literal `state.player.yaw` writes; the
    DOOM3DF3 culprit was `p.yaw=camera.rotation.y` via `const p=state.player`
    inside update(). The audit must list BOTH writers with their function."""
    from tools import static_state_writer_lines
    rows = static_state_writer_lines(_DOOM_LIKE_HTML, "player.yaw")
    where = {r["in"]: r["text"] for r in rows}
    assert "on mousemove" in where and "p.yaw -= e.movementX" in where["on mousemove"]
    assert "update" in where and "p.yaw=camera.rotation.y" in where["update"]
    # `camera.rotation.y = p.yaw` is a READ of yaw, not a write.
    assert not any("camera.rotation.y = p.yaw" in r["text"] for r in rows)
    assert static_state_writer_lines(_DOOM_LIKE_HTML, "player.nope") == []


def test_state_field_evidence_block_names_overwrite(tmp_path):
    """Writer audit + runtime stickiness → one measured block under the user
    note, with a VERDICT when the poke reverted."""
    import asyncio

    class _B:
        async def state_field_stickiness(self, fields):
            return {"player.yaw": {"before": 0, "poked": 0.5, "after": 0, "drift": False, "reverted": True}}

    a = _agent(tmp_path)
    a.browser = _B()
    a._current_file = tmp_path / "game.html"
    a._current_file.write_text(_DOOM_LIKE_HTML)
    block = asyncio.run(a._build_state_field_evidence(["player.yaw"]))
    assert "STATE-FIELD EVIDENCE" in block
    assert "VERDICT: state.player.yaw is OVERWRITTEN EVERY FRAME" in block
    assert "in update: p.yaw=camera.rotation.y" in block
    assert "in on mousemove" in block


def test_raw_mode_repeat_complaint_escalates_and_attaches_evidence(tmp_path):
    """Raw feedback mode (the default) never ran repeat detection, so the
    same ask was re-patched 4× in the handler. Second identical complaint
    must: prefix the note, add the CHANGE APPROACH block naming prior
    attempts, attach the measured evidence, and arm the diagnose prefill."""
    a = _agent(tmp_path)
    a._use_feedback_directives = False  # raw mode
    a._recent_feedback_texts = ["use the mouse to turn the player and look around"]
    a._fix_attempt_ledger = [{"applied_heads": ["document.addEventListener('mousemove'"], "notes": "wired mousemove yaw"}]
    a._feedback_state_evidence = "================ STATE-FIELD EVIDENCE (harness, measured) ================\nx"
    a._pending_feedback = ["the mouse still does not turn the player to look around"]
    a._feedback_route = {"primary_intent": "code_fix", "honor_user_now": True,
                         "allow_assets_block": False, "defer_behind_blocker": False}
    out = a._flush_user_injections("REPORT: ok")
    assert "USER HAS RAISED THIS BEFORE" in out
    assert "REPEAT COMPLAINT — CHANGE APPROACH" in out
    assert "attempt 1: edited near document.addEventListener('mousemove'" in out
    assert "STATE-FIELD EVIDENCE" in out
    assert a._repeat_feedback_escalated is True


def test_raw_mode_first_complaint_does_not_escalate(tmp_path):
    a = _agent(tmp_path)
    a._use_feedback_directives = False
    a._recent_feedback_texts = []
    a._pending_feedback = ["use the mouse to turn the player"]
    a._feedback_route = {"primary_intent": "code_fix", "honor_user_now": True,
                         "allow_assets_block": False, "defer_behind_blocker": False}
    out = a._flush_user_injections("REPORT: ok")
    assert "REPEAT COMPLAINT" not in out
    assert not getattr(a, "_repeat_feedback_escalated", False)
