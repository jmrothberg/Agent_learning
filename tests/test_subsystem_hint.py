"""Tests for `_subsystem_hint` + focused-slice biasing.

DK trace 20260514_104131 background:
  - 100% of recent sessions had `mistake_signature` containing
    "INPUT_DEAD" (set by `memory.signature_for_report` when
    input_test ran but produced no canvas change) or the
    human-readable "Controls are not wired up" soft-warning text.
  - The model's <patch> blocks did NOT touch `addEventListener` /
    `keydown` code on any iter; they kept editing the higher-level
    mechanic the user's complaint named.
  - Existing coaching at `_repeat_sig_streak >= 2` said "AUTHOR a
    runtime-state probe" — too abstract for a 27B model to
    translate into "rewrite the keydown handler."

The fix has two halves:
  1. `_focused_slice` biases its keyset toward the implicated
     subsystem's identifier tokens, so the model SEES the right code
     region in its fix prompt (even on iter 1, before any streak).
  2. The coaching message at streak ≥ 2 names the subsystem and the
     code area, instead of the generic probe-authoring text.

These tests cover the pure helper + the keyset injection. The
coaching-message wording is asserted by string-presence checks against
the in-memory `_pending_coaching` list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _subsystem_hint  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: returns shape, identifiers, fix phrase
# ---------------------------------------------------------------------------


def test_input_dead_signature_matches_input_subsystem():
    """The literal token `INPUT_DEAD` from `memory.signature_for_report`
    (memory.py:664) must trigger the input hint."""
    h = _subsystem_hint("PROBE FAILED [foo] | INPUT_DEAD")
    assert h is not None
    assert h["name"] == "input"
    assert "addEventListener" in h["identifiers"]
    assert "keydown" in h["identifiers"]
    assert "keyup" in h["identifiers"]
    assert "keydown" in h["fix_phrase"] or "input" in h["fix_phrase"].lower()


def test_controls_not_wired_text_matches_input_subsystem():
    """The human-readable form of the input failure — from
    tools.py's HEURISTIC soft-warning — also matches."""
    h = _subsystem_hint(
        "HEURISTIC: pressed ArrowUp, ArrowDown, ArrowLeft, "
        "ArrowRight - canvas pixels never changed. Controls "
        "are not wired up (or input handler is broken)."
    )
    assert h is not None
    assert h["name"] == "input"


def test_frozen_canvas_matches_draw_or_raf_subsystem():
    """The `FROZEN` token signature (memory.py:661) or the
    human-readable 'did not change between two samples' both
    match the draw-or-RAF subsystem."""
    h = _subsystem_hint("Some error | FROZEN")
    assert h is not None
    assert h["name"] == "draw_or_raf"
    assert "requestAnimationFrame" in h["identifiers"]

    h2 = _subsystem_hint(
        "HEURISTIC: canvas drew SOMETHING but did not change "
        "between two samples 1s apart"
    )
    assert h2 is not None
    assert h2["name"] == "draw_or_raf"


def test_canvas_uniform_matches_raf_start_subsystem():
    """The 'canvas pixels are uniform' soft-warning fires when the
    canvas literally hasn't been drawn to yet (RAF kick-off
    missing). Maps to a different subsystem than FROZEN — the
    fix is wiring `loadAssets().then(() => requestAnimationFrame(...))`,
    not patching the draw function."""
    h = _subsystem_hint(
        "HEURISTIC: canvas pixels are uniform AND keyboard input "
        "didn't change anything either"
    )
    assert h is not None
    assert h["name"] == "raf_start"
    assert "loadAssets" in h["identifiers"]


def test_unrecognized_signature_returns_none():
    """When the signature doesn't match any of the known shapes,
    the helper returns None and callers fall back to existing
    generic behavior."""
    assert _subsystem_hint("totally unrelated error: bullets array is empty") is None
    assert _subsystem_hint("") is None
    assert _subsystem_hint(None or "") is None


def test_helper_is_case_insensitive():
    """Signature matching is case-insensitive so a model that
    writes 'Input Dead' or the harness that writes 'INPUT_DEAD'
    both work."""
    assert _subsystem_hint("input_dead") is not None
    assert _subsystem_hint("INPUT_DEAD") is not None
    assert _subsystem_hint("Input_Dead") is not None


def test_literal_trace_20260514_175012_signature_matches_input():
    """Pin the literal mistake_signature from
    games/traces/a-game-of-donkey-kong-all-char_20260514_175012.jsonl.
    The trace's coaching_injected event fired the GENERIC fallback
    because it pre-dates the Item 1a deploy; this test confirms the
    new code WOULD have engaged on that exact sig — no path bug."""
    sig = (
        "HEURISTIC: pressed ArrowUp, ArrowDown, ArrowLeft, "
        "ArrowRight, Space, KeyW, KeyA, KeyS, KeyD - canvas "
        "pixels never changed. Controls are not wired up "
        "(or input handler is broken). | PROBE FAILED "
        "[input_responsive]: `(harness) keyboard input must "
        "produce a visible canvas change` — pressed [...] and "
        "canvas pixels never changed — controls promised in "
        "<criteria> but not wired."
    )
    h = _subsystem_hint(sig)
    assert h is not None
    assert h["name"] == "input"
    assert "addEventListener" in h["identifiers"]
    assert "keydown" in h["identifiers"]
    assert "keyup" in h["identifiers"]


# ---------------------------------------------------------------------------
# Focused-slice biasing (Item 1b)
# ---------------------------------------------------------------------------


def _slice_agent(last_sig: str = "") -> GameAgent:
    """Build a minimum-viable GameAgent without spinning up the
    backend, with the fields _focused_slice needs."""
    a = GameAgent.__new__(GameAgent)
    a._last_mistake_sig = last_sig
    # _trace is called inside _focused_slice on the biasing branch;
    # stub it.
    a._trace = lambda obj: None
    return a


_DK_LIKE_HTML = """<!DOCTYPE html><html><body><canvas></canvas><script>
const state = { player: { x: 100, y: 100, climbing: false } };
function climbLadder(p) {
  if (p.climbing) {
    p.y -= 1;
  }
}
function nearestLadder(p, mode) {
  return null;
}
function update(dt) {
  climbLadder(state.player);
}
function frame() {
  update(0.016);
  requestAnimationFrame(frame);
}
function setupKeys() {
  const KEYMAP = { ArrowUp: 'up', ArrowDown: 'down' };
  window.addEventListener('keydown', function (e) {
    const k = KEYMAP[e.code];
    if (k) state.keys[k] = true;
  });
  window.addEventListener('keyup', function (e) {
    const k = KEYMAP[e.code];
    if (k) state.keys[k] = false;
  });
}
""" + ("/* filler line to push file over the 12 KB inject limit */\n" * 400) + """
</script></body></html>"""


def test_focused_slice_without_hint_misses_input_code():
    """Pin the failure mode: when only error signals ("update",
    "climbLadder") drive the keyset, the input-handler functions
    don't show up in the slice — the model never sees the keydown
    code it needs to rewrite."""
    agent = _slice_agent(last_sig="")  # No subsystem hint.
    fake_report = {
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "soft_warnings": ["climbLadder did not advance the player"],
    }
    slice_text = agent._focused_slice(
        _DK_LIKE_HTML, fake_report,
        "Basic: climbLadder lets Mario climb",
    )
    assert slice_text is not None
    assert "climbLadder" in slice_text
    # Input wiring is invisible — that's the bug we're fixing.
    assert "addEventListener" not in slice_text
    assert "setupKeys" not in slice_text


def test_focused_slice_with_input_hint_pulls_in_input_code():
    """With the input-subsystem hint set on the agent via
    `_last_mistake_sig`, the slice now biases toward keydown /
    addEventListener / KEYMAP code. The model gets to SEE the input
    handler, not just the climb math."""
    agent = _slice_agent(
        last_sig="HEURISTIC: pressed [keys] - canvas pixels never "
                 "changed. Controls are not wired up. | INPUT_DEAD",
    )
    fake_report = {
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "soft_warnings": ["climbLadder did not advance the player"],
    }
    slice_text = agent._focused_slice(
        _DK_LIKE_HTML, fake_report,
        "Basic: climbLadder lets Mario climb",
    )
    assert slice_text is not None
    # Input wiring code now present in the slice.
    assert "setupKeys" in slice_text or "addEventListener" in slice_text or "KEYMAP" in slice_text


def test_focused_slice_returns_none_for_small_files_regardless_of_hint():
    """The hint biasing must not bypass the small-file short-circuit
    (`len(html) <= _FULL_FILE_INJECT_LIMIT` returns None). Tiny seed
    files always go through the full-file path."""
    agent = _slice_agent(
        last_sig="INPUT_DEAD",
    )
    small = "<html><body><script>function foo(){}</script></body></html>"
    assert agent._focused_slice(small, {}, "criteria") is None
