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


def test_internal_media_notice_does_not_arm_reprompt(tmp_path):
    """2026-06-12 (trace 20260612_004616): the agent's own 'Mid-session
    asset/sound/video additions' notice rode _pending_feedback, classified
    as a USER art request, and fired 8 spurious ASSET GENERATION REQUIRED
    banners demanding <assets> for files that already existed. Internally
    queued texts must never arm the unhonored-asset-request detector."""
    a = _make_agent(tmp_path)
    a._session_assets = {"knight_idle": Path("x.png")}
    a._use_feedback_directives = True
    notice = (
        "Mid-session asset/sound/video additions — load these in your "
        "next patch and use them where appropriate. The files exist on "
        "disk now:\n\n================ GENERATED ASSETS (sprites) "
        "================\nZ-Image sprite: knight_dash (knight_dash.png)"
    )
    a._queue_internal_feedback(notice)
    out = a._flush_user_injections("")
    # The notice itself still reaches the model…
    assert "Mid-session asset/sound/video additions" in out
    # …but must NOT be treated as an unhonored user art request.
    assert "ASSET GENERATION REQUIRED" not in out
    assert a._unhonored_asset_request is None
    assert a._asset_reprompt_count == 0


def test_genuine_user_art_request_still_arms_reprompt_after_notice(tmp_path):
    """The internal-text exclusion must not blind the detector to REAL
    user art requests queued in the same batch."""
    a = _make_agent(tmp_path)
    a._session_assets = {"knight_idle": Path("x.png")}
    a._use_feedback_directives = True
    a._queue_internal_feedback(
        "Mid-session asset/sound/video additions — load these in your "
        "next patch. GENERATED ASSETS (sprites): knight_dash"
    )
    a.add_user_feedback("please make a new red dragon sprite")
    out = a._flush_user_injections("")
    assert "ASSET GENERATION REQUIRED" in out
    assert a._unhonored_asset_request == "please make a new red dragon sprite"


def test_help_screen_feedback_with_location_asset_stays_code_path(tmp_path):
    """Trace 20260621_150955: "skull beach" matched beach_bg as a fuzzy
    asset stem, so a help-screen/game-logic request was misrouted into
    ASSET GENERATION REQUIRED instead of a code patch request."""
    a = _make_agent(tmp_path)
    a._session_assets = {"beach_bg": Path("beach.png")}
    a._use_feedback_directives = True
    a._fix_mode = True
    a._previous_report_ok = False
    a._pending_feedback = [
        "Add a help screen. check the logic of the game, "
        "for example how do you dig at skull beach"
    ]
    out = a._flush_user_injections("")
    assert "ASSET GENERATION REQUIRED" not in out
    assert "media-only items" not in out
    assert a._unhonored_asset_request is None


def test_ui_feature_patterns_match_help_hint_button_phrasing():
    """Fix 1 (2026-06-21 seed trace): the narrow `help screen` pattern
    missed `add a help button` / `hint button` phrasings, so those UI
    asks were classified as plain code and queued behind a blocker. The
    widened genre-free patterns must now recognise them."""
    from agent import _feedback_is_ui_feature

    for phrase in [
        "add a help button",
        "add a hint",
        "please add a help overlay",
        "add a hint button",
        "show a hint panel",
        "we need a help modal",
        "add help",
    ]:
        assert _feedback_is_ui_feature(phrase), (
            f"should classify as UI feature: {phrase!r}"
        )

    # Negative controls — these are NOT UI-feature feedback.
    for phrase in [
        "the enemy moves too fast",
        "make the music loop",
        "the player sprite is the wrong color",
    ]:
        assert not _feedback_is_ui_feature(phrase), (
            f"should NOT classify as UI feature: {phrase!r}"
        )


def test_reprompt_cleared_by_subject_matching_patches(tmp_path):
    """2026-06-12 (trace 20260612_132314): 'need to improve animation for
    jump,duck, left and right' was hard-classified as an art request, but
    the model correctly shipped procedural-animation PATCHES (img2img
    cannot change pose). The reprompt only cleared on <assets>, so it
    nagged 3 turns. Patches that touch the request's subject terms must
    stand it down."""
    a = _make_agent(tmp_path)
    a._unhonored_asset_request = (
        "need to improve animation for jump,duck, left and right"
    )
    a._asset_reprompt_count = 2
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "function drawKnight(pose) {\n"
        "=======\n"
        "function drawKnight(pose) {\n"
        "  // procedural animation: squash on jump, bob on duck,\n"
        "  // sway left / right while dashing\n"
        "  applyPoseAnimation(pose);\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    a._maybe_clear_asset_reprompt_via_code(reply)
    assert a._unhonored_asset_request is None
    assert a._asset_reprompt_count == 0


def test_reprompt_kept_when_patches_do_not_touch_subject(tmp_path):
    """The original here-s-a-tight-test failure mode stays covered: a
    reply whose patches are unrelated to the art request keeps the
    reprompt armed."""
    a = _make_agent(tmp_path)
    a._unhonored_asset_request = "please make a new red dragon sprite"
    a._asset_reprompt_count = 1
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "score += 10;\n"
        "=======\n"
        "score += 25;\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )
    a._maybe_clear_asset_reprompt_via_code(reply)
    assert a._unhonored_asset_request == "please make a new red dragon sprite"
    assert a._asset_reprompt_count == 1


def test_reprompt_kept_when_reply_has_no_patches(tmp_path):
    a = _make_agent(tmp_path)
    a._unhonored_asset_request = "make a new red dragon sprite"
    a._asset_reprompt_count = 1
    a._maybe_clear_asset_reprompt_via_code(
        "<notes>I will add the dragon sprite next turn.</notes>"
    )
    assert a._unhonored_asset_request == "make a new red dragon sprite"


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
