"""Re-assert 'emit <assets>' until the model honors a new-art request (2026-05-31).

here-s-a-tight-test 20260530: the user asked twice for a red opponent's sprites;
the model (blocker-distracted, raw-feedback mode) replied with code/diagnose and
never emitted an <assets> block, so ZERO new assets were generated. General fix:
track the unhonored art request and re-inject an ASSET GENERATION REQUIRED
directive each turn (capped) until an <assets> block appears. Genre/model-agnostic.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402
from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(model="stub:1b", out_path=out, browser=MagicMock(),
                     max_iters=2, memory_root=str(tmp_path / "memory"))


# ---- source-pin: the wiring exists -----------------------------------------

def test_flush_has_asset_generation_required_directive():
    src = inspect.getsource(GameAgent._flush_user_injections)
    assert "ASSET GENERATION REQUIRED" in src
    assert "_unhonored_asset_request" in src
    assert "asset_request_reprompt" in src
    assert "_reprompts < 3" in src  # bounded


def test_assets_block_clears_the_reprompt():
    src = inspect.getsource(GameAgent._maybe_generate_assets_and_sounds)
    assert "_unhonored_asset_request = None" in src
    assert "asset_specs and self._unhonored_asset_request" in src


# ---- behavioral: directive fires, is bounded, clears on <assets> -----------

def test_reprompt_fires_repeats_and_caps(tmp_path):
    a = _make_agent(tmp_path)
    a._session_assets = {"fighter_idle": Path("x.png")}
    a._use_feedback_directives = True
    # an explicit make-new-art request
    a._pending_feedback = ["please make a new red fighter sprite, same moves"]
    out1 = a._flush_user_injections("please make a new red fighter sprite, same moves")
    assert "ASSET GENERATION REQUIRED" in out1
    assert a._unhonored_asset_request is not None
    assert a._asset_reprompt_count == 1
    # next turns: no new feedback, but the request is still unhonored → re-assert
    a._pending_feedback = []
    out2 = a._flush_user_injections("")
    assert "ASSET GENERATION REQUIRED" in out2 and a._asset_reprompt_count == 2
    a._flush_user_injections("")            # count 3
    out4 = a._flush_user_injections("")      # >3 → give up, stop nagging
    assert "ASSET GENERATION REQUIRED" not in out4
    assert a._unhonored_asset_request is None


def test_reprompt_clears_when_assets_emitted(tmp_path):
    a = _make_agent(tmp_path)
    a._session_assets = {"fighter_idle": Path("x.png")}
    a._use_feedback_directives = True
    a._pending_feedback = ["make a new red fighter sprite"]
    a._flush_user_injections("make a new red fighter sprite")
    assert a._unhonored_asset_request is not None
    # simulate the clear that _maybe_generate_assets_and_sounds does on <assets>
    asset_specs = [{"name": "p2_idle", "prompt": "red fighter", "size": (768, 768)}]
    if asset_specs and a._unhonored_asset_request is not None:
        a._unhonored_asset_request = None
        a._asset_reprompt_count = 0
    a._pending_feedback = []
    out = a._flush_user_injections("")
    assert "ASSET GENERATION REQUIRED" not in out
