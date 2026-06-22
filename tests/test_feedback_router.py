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
