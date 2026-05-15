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
    _SKELETON_MAX_BYTES,
    _SKELETON_MIN_BODY_BYTES,
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
