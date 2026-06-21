"""Tests for `_detect_skeleton_payload` and the agent's `<html_file>`
rejection of pseudocode-skeleton bodies.

Trace evidence — `build-a-donkey-kong-clone-in-o_20260514_214747`
iter 3 (turn 10):
  - Model emitted an `<html_file>` whose JS body was 374 bytes of
    `// Asset loading`, `// Sound loading`, `function loadAssets() {
    ... }` comment-headers with `{ ... }` placeholder bodies.
  - The harness wrote that to disk as the baseline.
  - Iter 4 had no real code to patch against and the model burned
    another deliberation loop trying to rebuild.

The detector trips when ALL FOUR conditions hold (see `_detect_
skeleton_payload` docstring): tiny total size, near-zero post-comment
JS volume, ≤2 function definitions, AND a placeholder marker like
`{ ... }` or `// ...`. False-positive risk on real-but-tiny games is
controlled by the placeholder requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    _detect_skeleton_payload,
    _is_placeholder_first_build,
    _SKELETON_MAX_BYTES,
    _SKELETON_MIN_BODY_BYTES,
    _PLACEHOLDER_FIRST_BUILD_MIN_CODE,
)


# ---------------------------------------------------------------------------
# Positive cases — must trip the detector
# ---------------------------------------------------------------------------


_DK_TRACE_SKELETON = """<!DOCTYPE html><html><body><canvas></canvas><script>
(function() {
  "use strict";
  // Canvas setup
  const cvs = document.getElementById("c");
  // ...
  // Asset loading
  const ASSETS = {};
  async function loadAssets() { ... }
  // Sound loading
  const SOUNDS = {};
  function loadSounds() { ... }
  // Input handling
  const keys = {};
})();
</script></body></html>"""


def test_dk_trace_374byte_skeleton_triggers_detector():
    """The literal trace 20260514_214747 iter 3 shape — must reject."""
    result = _detect_skeleton_payload(_DK_TRACE_SKELETON)
    assert result is not None
    # Reason mentions the diagnostic numbers so the model can act on them.
    assert "code after" in result.lower() or "placeholder" in result.lower()


def test_short_skeleton_with_only_comments_and_placeholders():
    """Even smaller variant — all comment headers, no real code."""
    skel = (
        "<!DOCTYPE html><html><body><script>"
        "// game setup\n// init\n// loop\nfunction game() { ... }\n"
        "</script></body></html>"
    )
    assert _detect_skeleton_payload(skel) is not None


def test_pseudocode_with_TODO_markers():
    """Variant: model uses `// TODO` placeholder markers instead of
    `// ...` to mark unfilled regions."""
    skel = (
        "<!DOCTYPE html><html><body><script>"
        "function init() { // TODO: implement\n}\n"
        "function loop() {}\n"
        "</script></body></html>"
    )
    # Tiny body + TODO marker — should trip.
    assert _detect_skeleton_payload(skel) is not None


# ---------------------------------------------------------------------------
# Negative cases — real code must NOT trigger
# ---------------------------------------------------------------------------


def test_real_game_does_not_trigger():
    """A regular small-but-real game body should pass through."""
    body = (
        "var x = 0;\n"
        "var y = 100;\n"
        "function update(dt) { x += 1; y -= 1; }\n"
        "function draw() { ctx.clearRect(0,0,800,600); }\n"
        "function frame() { update(0.016); draw(); requestAnimationFrame(frame); }\n"
        "frame();\n"
    ) * 10  # repeat to get past min body bytes
    html = (
        "<!DOCTYPE html><html><body><canvas></canvas><script>"
        f"{body}</script></body></html>"
    )
    assert _detect_skeleton_payload(html) is None


def test_dom_only_app_passes():
    """A DOM-only app (calculator, todo list) has no `<script>` and
    must NOT be flagged — they're legitimately tiny."""
    html = (
        "<!DOCTYPE html><html><body>"
        "<h1>Calculator</h1>"
        "<input id='a'>"
        "<button>+</button>"
        "</body></html>"
    )
    assert _detect_skeleton_payload(html) is None


def test_legitimate_tiny_game_without_placeholders_passes():
    """A genuinely small one-script game (e.g. a 600-byte snake) — no
    placeholder markers — must not false-positive even if it's compact.
    The placeholder-marker requirement is the safety against this."""
    html = (
        "<!DOCTYPE html><html><body><canvas></canvas><script>"
        "var s={x:0};function f(){s.x++;requestAnimationFrame(f);}f();"
        "</script></body></html>"
    )
    # < 4 KB, < 800 bytes body, ≤ 2 functions — BUT no `{ ... }` or
    # `// ...` markers. Must pass.
    assert _detect_skeleton_payload(html) is None


def test_large_file_with_placeholders_passes():
    """A larger file (>4 KB total) is over the size cap and must not
    be flagged even if it happens to contain a `// ...` somewhere."""
    body = (
        "function realFunction() {\n"
        "  // This is a normal comment that happens to say `// ...` in prose.\n"
        "  return 1;\n"
        "}\n"
        "// Lots of real code:\n"
    ) + "var x = 1; var y = 2; var z = 3;\n" * 200
    html = (
        "<!DOCTYPE html><html><body><canvas></canvas><script>"
        f"{body}</script></body></html>"
    )
    assert _detect_skeleton_payload(html) is None


def test_empty_input_returns_none():
    assert _detect_skeleton_payload("") is None
    assert _detect_skeleton_payload(None or "") is None


# ---------------------------------------------------------------------------
# Threshold constants pinned
# ---------------------------------------------------------------------------


def test_thresholds_pinned():
    """Pin the threshold constants so a refactor doesn't silently drift
    them. If you genuinely need to change these, update this test
    AND document the rationale in the docstring of
    `_detect_skeleton_payload`."""
    assert _SKELETON_MAX_BYTES == 4_000
    assert _SKELETON_MIN_BODY_BYTES == 800
    assert _PLACEHOLDER_FIRST_BUILD_MIN_CODE == 24


# ---------------------------------------------------------------------------
# `_is_placeholder_first_build` — marker-free, canvas-required sibling.
# Dragon's-lair trace 20260621_091419 iter 1: a 593-byte canvas+comment-only
# stub slipped past `_detect_skeleton_payload` (no `{ ... }` marker) and
# shipped to Chromium. This helper catches that shape so the existing
# first-build prefill RETRY rescue arms instead.
# ---------------------------------------------------------------------------


# Approximation of the dragon iter-1 stub: a <canvas> game with a
# <script> body that is ONLY comments — no executable statements.
_DRAGON_STYLE_STUB = (
    "<!DOCTYPE html><html><head><title>Dragon's Lair</title></head><body>"
    "<canvas id='c' width='800' height='600'></canvas>"
    "<script>\n"
    "// Dragon's Lair game\n"
    "// set up the canvas and context\n"
    "// load the animation frames\n"
    "// handle input and scene transitions\n"
    "// main game loop goes here\n"
    "</script></body></html>"
)


def test_comment_only_canvas_stub_is_placeholder():
    """The dragon iter-1 shape — canvas present, script body all comments."""
    assert _is_placeholder_first_build(_DRAGON_STYLE_STUB) is True


def test_canvas_with_no_inline_script_is_placeholder():
    """A <canvas> declared but no inline <script> body at all is a stub."""
    html = (
        "<!DOCTYPE html><html><body>"
        "<canvas id='c'></canvas>"
        "</body></html>"
    )
    assert _is_placeholder_first_build(html) is True


def test_real_tiny_canvas_game_is_not_placeholder():
    """The same 59-char one-liner snake used above — real statements, must
    NOT be flagged (would otherwise lose a working first build)."""
    html = (
        "<!DOCTYPE html><html><body><canvas></canvas><script>"
        "var s={x:0};function f(){s.x++;requestAnimationFrame(f);}f();"
        "</script></body></html>"
    )
    assert _is_placeholder_first_build(html) is False


def test_cdn_first_script_does_not_mask_real_inline_body():
    """Protects the working three.js DOOM one-shot: a CDN `<script src=...>`
    first tag has an empty body, but the REAL inline <script> after it
    must keep the file out of the placeholder bucket. All bodies are
    concatenated, so the real code wins."""
    html = (
        "<!DOCTYPE html><html><head>"
        "<script src='https://example.com/three.min.js'></script>"
        "</head><body><canvas></canvas><script>"
        "const scene=init();function loop(){update();render(scene);"
        "requestAnimationFrame(loop);}loop();"
        "</script></body></html>"
    )
    assert _is_placeholder_first_build(html) is False


def test_dom_only_app_is_not_placeholder():
    """Pure-DOM app (no <canvas>) is exempt — different intent."""
    html = (
        "<!DOCTYPE html><html><body>"
        "<h1>Calculator</h1><input id='a'><button>+</button>"
        "<script>// nothing real here either</script>"
        "</body></html>"
    )
    assert _is_placeholder_first_build(html) is False


def test_sizable_file_is_never_placeholder():
    """Any file at/over the size ceiling is treated as a real build and
    never inspected (belt-and-suspenders for the DOOM 22 KB one-shot)."""
    big = (
        "<!DOCTYPE html><html><body><canvas></canvas><script>"
        + ("// just a comment line\n" * 400)  # >4 KB but all comments
        + "</script></body></html>"
    )
    assert len(big) >= _SKELETON_MAX_BYTES
    assert _is_placeholder_first_build(big) is False


def test_empty_input_is_not_placeholder():
    assert _is_placeholder_first_build("") is False
