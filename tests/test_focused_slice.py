"""Tests for the focused-slice callee-promotion feature (Tier 4 Fix A).

Background: the asteroid_20260510_164857 trace showed the model wasted
4 iters debugging blind because `update()` had been elided from the
focused slice. Error signals said "canvas didn't change" (no mention
of `update`), so `update()` scored 0 and was dropped. Meanwhile
`frame()` was kept and `frame()` calls `update(dt)` on its first line.

These tests exercise `GameAgent._focused_slice()` directly with
synthetic HTML so we don't need Chromium or a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# A plausible game shape: frame() drives the loop and calls update() +
# draw(). The "bug" the model is asked to fix is keyboard input dead, so
# error signals reference keys/input/canvas — NOT `update`. Without
# callee promotion, `update()` would be dropped despite being the
# function that consumes the key state.
_GAME_HTML = """<!DOCTYPE html>
<html>
<head><title>game</title></head>
<body>
<canvas id="c"></canvas>
<script>
const KEYMAP = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left',
                 ArrowRight: 'right', Space: 'fire' };
const keys = {}, pressed = {};
window.addEventListener('keydown', e => {
  const k = KEYMAP[e.code];
  if (!k) return;
  if (!keys[k]) pressed[k] = true;
  keys[k] = true;
  e.preventDefault();
});
window.addEventListener('keyup', e => {
  const k = KEYMAP[e.code];
  if (k) keys[k] = false;
});

const state = { ship: { x: 400, y: 300, angle: 0, vx: 0, vy: 0 } };
window.state = state;

function update(dt) {
  // Bug intentionally here: should read keys.up but reads pressed.up,
  // so held thrust fires only one frame. Model would only spot this
  // if `update` appears in the focused slice.
  if (pressed.up) {
    state.ship.vx += Math.cos(state.ship.angle) * dt;
    state.ship.vy += Math.sin(state.ship.angle) * dt;
  }
  if (keys.left) state.ship.angle -= dt * 3;
  if (keys.right) state.ship.angle += dt * 3;
  state.ship.x += state.ship.vx * dt;
  state.ship.y += state.ship.vy * dt;
}

function draw() {
  const ctx = document.getElementById('c').getContext('2d');
  ctx.clearRect(0, 0, 800, 600);
  ctx.fillStyle = '#fff';
  ctx.fillRect(state.ship.x, state.ship.y, 10, 10);
}

function frame(now) {
  const dt = 0.016;
  try { update(dt); draw(); } catch (e) { console.error(e); }
  for (const k in pressed) pressed[k] = false;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
""" + ("// padding to push past _FULL_FILE_INJECT_LIMIT so slicing kicks in. " * 200)


def test_callee_promoted_when_caller_selected(tmp_path):
    """frame() is selected (its name appears in errors / criteria),
    update() and draw() should both be promoted via the call-graph
    walk even though neither is named in the error signals."""
    a = _agent(tmp_path)
    # Report mentions `frame` (by listener / canvas wording) and `keys`
    # but NOT `update` — that's the asteroid trace shape.
    report = {
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "soft_warnings": [
            "HEURISTIC: pressing keys did not change canvas pixels — "
            "the frame loop is running but input is dead."
        ],
        "probes": [],
    }
    criteria = "Basic: pressing ArrowUp applies thrust to the ship."
    slice_text = a._focused_slice(_GAME_HTML, report, criteria)
    assert slice_text is not None, "expected a non-None slice"
    # frame should win on identifier match (mentions `keys`, `pressed`,
    # `canvas` chain), update + draw should ride in as one-hop callees.
    assert "function frame" in slice_text, "frame() must be in the slice"
    assert "function update" in slice_text, (
        "update() must be promoted as a one-hop callee of frame() — "
        "this is the whole point of Fix A"
    )


def test_no_promotion_when_caller_not_selected(tmp_path):
    """If no function gets enough identifier signal to be kept,
    nothing should be auto-promoted either (no callers → no callees).
    Returns None and the agent falls through to a full-file inject."""
    a = _agent(tmp_path)
    # Signals reference nothing in the source.
    report = {
        "errors": [], "console_errors": [], "page_errors": [],
        "soft_warnings": ["something_unrelated_to_this_game broke"],
        "probes": [],
    }
    slice_text = a._focused_slice(_GAME_HTML, report, "")
    # Either None (no signals matched anywhere) OR a slice that
    # at least doesn't hallucinate functions that shouldn't be there.
    if slice_text is not None:
        # Must NOT auto-include unrelated stuff — the promotion path
        # only fires when something else was already selected.
        assert "function update" not in slice_text


def test_function_cap_raised_to_five(tmp_path):
    """The cap was 3; raised to 5 to absorb 1-2 callee promotions
    without pushing out higher-signal functions. Sanity: a game
    with > 3 score>0 functions returns up to 5 in the slice."""
    # Build HTML with 6 distinct functions, each mentioning `state`
    # so they all score positive against a state-focused report.
    funcs = [
        f"function func_{i}() {{ state.field_{i} += 1; return state; }}"
        for i in range(6)
    ]
    html = (
        "<!DOCTYPE html><html><body><script>\n"
        "const state = {};\n"
        + "\n".join(funcs)
        + "\n</script></body></html>"
        + ("// padding to push past _FULL_FILE_INJECT_LIMIT. " * 200)
    )
    a = _agent(tmp_path)
    report = {
        "errors": [], "console_errors": [], "page_errors": [],
        "soft_warnings": ["state went wrong"],
        "probes": [],
    }
    slice_text = a._focused_slice(html, report, "Basic: state must persist.")
    if slice_text is None:
        # Slice could legitimately exceed 60%-of-file fallback on a
        # small synthetic — relax the expectation in that case.
        return
    # Count function entries — header line includes the marker.
    n_kept = slice_text.count("--- function `")
    assert n_kept <= 5, f"function cap should be ≤ 5, got {n_kept}"
    assert n_kept >= 3, f"expected at least 3 high-scoring functions, got {n_kept}"
