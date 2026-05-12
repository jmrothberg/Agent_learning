"""Tests for the two bloat detectors that catch the maze-repetition
failure mode local LLMs fall into:

  1. ollama_io.RepetitionDetector — streams text, fires on the 3rd
     repeat of an 8-line block. Catches duplication LIVE so the agent
     can abort early.

  2. agent._detect_block_bloat — scans already-materialized HTML for
     the same pattern. Last line of defense before writing to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (  # noqa: E402
    _detect_block_bloat,
    _is_degenerate_baseline,
    _truncation_reason,
)
from ollama_io import RepetitionDetector  # noqa: E402


# ---------------------------------------------------------------------------
# _detect_block_bloat — operates on already-assembled text.
# ---------------------------------------------------------------------------


def _maze_block(seed: int) -> str:
    """Plausible-looking 8-line maze chunk that's also >200 bytes."""
    rows = [
        f"  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  // s={seed}",
    ] * 8
    return "\n".join(rows)


def test_clean_html_returns_none():
    """A normal-looking HTML game has no duplicated 8-line block."""
    html = "<!DOCTYPE html>\n<html><body>\n" + "\n".join(
        f"<div id='r{i}'>row {i}</div>" for i in range(80)
    ) + "\n</body></html>\n"
    assert _detect_block_bloat(html) is None


def test_three_repeats_is_clean():
    """Exactly 3 repeats stays under the > 3 threshold."""
    block = _maze_block(0)
    text = "\n\n".join([block] * 3)
    assert _detect_block_bloat(text) is None


def test_four_repeats_flags():
    """4 identical 8-line blocks ≥ 200 bytes each → bloat detected."""
    block = _maze_block(0)
    text = "\n\n".join([block] * 4)
    result = _detect_block_bloat(text)
    assert result is not None
    assert "appears" in result
    assert "8-line block" in result


def test_short_blocks_skipped():
    """Repeated but short blocks shouldn't trigger (most legit HTML
    has plenty of these — closing tags, repeated CSS rules)."""
    short_block = "</div>\n" * 8  # 8 lines, but well under 200 bytes
    text = (short_block + "\n") * 10
    assert _detect_block_bloat(text) is None


def test_too_few_total_lines_skipped():
    """If the whole text is shorter than 8 * 4 lines, can't even
    physically have 4 repeats of an 8-line block."""
    block = _maze_block(0)
    # Only 24 lines total — not enough for the detector to consider.
    text = "\n".join(block.splitlines()[:24])
    assert _detect_block_bloat(text) is None


# ---------------------------------------------------------------------------
# RepetitionDetector — block-level window (window 3).
# ---------------------------------------------------------------------------


def test_repdetector_clean_stream():
    """A varied stream with lexically distinct lines doesn't trip the
    detector. Note: lines that only differ by a numeric suffix are
    *supposed* to trip the near-dup-template detector (window 2), so
    this stream is intentionally varied across both digits AND words."""
    det = RepetitionDetector()
    words = [
        "ship", "asteroid", "bullet", "score", "lives", "particle",
        "explosion", "thrust", "rotate", "wrap", "collide", "spawn",
    ]
    for i in range(200):
        w = words[i % len(words)]
        line = f"const {w}{i // len(words)} = computeSomething({i}, '{w}');\n"
        if det.feed(line):
            assert False, f"false positive on line {i}: {det.stall_reason}"


def test_repdetector_maze_block_dup_triggers():
    """4 identical 8-line ≥ 200-byte blocks streamed → inline_data_bloat."""
    det = RepetitionDetector()
    block = _maze_block(0) + "\n"
    tripped_on = None
    for repeat_idx in range(6):
        if det.feed(block):
            tripped_on = repeat_idx
            break
    assert tripped_on is not None, "detector did not fire on the bloat pattern"
    assert det.stall_reason == "inline_data_bloat"


def test_repdetector_short_line_loop_still_works():
    """The original short-line detector (window 1) still fires."""
    det = RepetitionDetector()
    fired = False
    for _ in range(40):
        if det.feed("</body></html>\n"):
            fired = True
            break
    assert fired
    assert det.stall_reason == "short_line_loop"


def test_repdetector_numbered_template_loop_still_works():
    """Window 2 (digit-stripped) catches numbered template variants."""
    det = RepetitionDetector()
    fired = False
    for i in range(40):
        if det.feed(f'  {{"name":"asset_{i}","prompt":"foo"}}\n'):
            fired = True
            break
    assert fired
    assert det.stall_reason == "near_dup_template_loop"


# ---------------------------------------------------------------------------
# _is_degenerate_baseline — recognizes a truncated skeleton on disk so
# the agent allows a full <html_file> rewrite next iter instead of
# forcing patch-mode against placeholder comments.
#
# The exact failure shape we're protecting against came from
# games/traces/classic-doom-style-first-perso_20260512_101944: iter 1
# hit the MLX 16384-token cap mid-stream, the harness wrote the 835-byte
# placeholder-comment skeleton, and iter 2's correct full rewrite was
# then rejected. With this detector iter 2 recovers.
# ---------------------------------------------------------------------------


_DOOM_SKELETON = """<!DOCTYPE html>
<html>
<head><style>/* Dark theme */</style></head>
<body>
  <canvas id="c"></canvas>
  <script>
    // Constants
    // Asset/sound paths
    // DOM refs
    // Input handling
    // Audio system
    // Game loop (RAF)
  </script>
</body>
</html>"""


_REAL_GAME_STUB = """<!DOCTYPE html>
<html>
<head><style>body{margin:0}canvas{display:block}</style></head>
<body>
  <canvas id="c" width="800" height="600"></canvas>
  <script>
    const cvs = document.getElementById("c");
    const ctx = cvs.getContext("2d");
    const W = cvs.width, H = cvs.height;
    const player = { x: W/2, y: H/2, hp: 100 };
    function update(dt) {
      player.x += 1;
      if (player.x > W) player.x = 0;
    }
    function draw() {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#0f0';
      ctx.fillRect(player.x, player.y, 20, 20);
    }
    let last = performance.now();
    function frame(now) {
      const dt = (now - last) / 1000;
      last = now;
      try { update(dt); draw(); } catch (e) { console.error(e); }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  </script>
</body>
</html>"""


def test_degenerate_baseline_detects_doom_skeleton():
    """The exact 835-byte truncated placeholder from the classic-doom
    trace must be classified as degenerate."""
    assert _is_degenerate_baseline(_DOOM_SKELETON) is True


def test_degenerate_baseline_passes_real_stub():
    """A small but real working game must NOT be flagged as degenerate
    — patches against it would still work."""
    # Pad to clear the 2 KB size floor while keeping the same shape.
    padded = _REAL_GAME_STUB.replace(
        "function draw() {",
        "function draw() {\n" + "      // real comment line " * 60,
    )
    assert len(padded) >= 2048
    assert _is_degenerate_baseline(padded) is False


def test_degenerate_baseline_empty_string():
    """An empty string is degenerate (no baseline at all)."""
    assert _is_degenerate_baseline("") is True
    assert _is_degenerate_baseline(None) is True  # type: ignore[arg-type]


def test_degenerate_baseline_no_canvas():
    """A document with no <canvas> can't be a canvas-game baseline."""
    html = "<!DOCTYPE html><html><body>" + ("x" * 3000) + "</body></html>"
    assert _is_degenerate_baseline(html) is True


def test_degenerate_baseline_script_only_comments():
    """A 5 KB document whose script body is just comments must be
    classified as degenerate — pads of `//`-comment placeholder lines
    is exactly the truncation shape."""
    body = "\n".join([f"    // section {i} placeholder" for i in range(200)])
    html = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        f"<script>{body}</script></body></html>"
    )
    assert len(html) > 4000
    assert _is_degenerate_baseline(html) is True


# ---------------------------------------------------------------------------
# _truncation_reason — generic across-models recognition of a stream
# that ended before closing its outer HTML/body/script tags.
#
# Trace evidence: classic-doom-style 20260512_153449 (DeepSeek-V4-mxfp8)
# emitted a stray ``` markdown-fence terminator inside the <html_file>
# body, ending the actual HTML before </script></body></html>.
# ---------------------------------------------------------------------------


def _doom_trace_153449_body() -> str:
    """Reconstruct the failure shape from the trace: open <html>, open
    <body>, open <script>, but stream stops before any close. Real-game
    sized so the size check in _is_degenerate_baseline can't short-circuit
    on the small-file path."""
    body = "\n".join([f"  const v{i} = {i};" for i in range(200)])
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'><head><title>X</title></head>\n"
        "<body>\n"
        "<canvas id='c' width='800' height='600'></canvas>\n"
        f"<script>\n{body}\n"
        # No </script>, no </body>, no </html> — truncated mid-stream.
    )


def test_truncation_reason_detects_unclosed_html():
    html = "<!DOCTYPE html><html lang='en'><body><canvas id='c'></canvas><script>const x=1;</script></body>"
    assert _truncation_reason(html) == "unclosed <html>"


def test_truncation_reason_detects_unclosed_body():
    html = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const x=1;</script></html>"
    assert _truncation_reason(html) == "unclosed <body>"


def test_truncation_reason_detects_unclosed_script():
    html = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const x=1;</body></html>"
    assert _truncation_reason(html) == "unclosed <script>"


def test_truncation_reason_returns_none_for_complete_file():
    html = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const x=1;</script></body></html>"
    assert _truncation_reason(html) is None


def test_truncation_reason_returns_none_for_empty_or_pure_text():
    assert _truncation_reason("") is None
    assert _truncation_reason(None) is None  # type: ignore[arg-type]
    assert _truncation_reason("Just some plain text, no tags at all.") is None


def test_truncation_reason_case_insensitive():
    """Some models capitalize HTML tags. The check is case-insensitive."""
    html = "<!DOCTYPE HTML><HTML><BODY><canvas id='c'></canvas><SCRIPT>const x=1;</SCRIPT></BODY>"
    assert _truncation_reason(html) == "unclosed <html>"


def test_doom_trace_153449_fixture_is_degenerate():
    """The exact failure shape from the user's trace must be classified
    as a degenerate baseline so the rewrite-gate carve-out kicks in.
    The detector reports the outermost unclosed tag first (<html>),
    which is the highest-leverage signal — even if <body> and <script>
    are also unclosed, fixing the outermost matters for recovery."""
    broken = _doom_trace_153449_body()
    assert _truncation_reason(broken) == "unclosed <html>"
    assert _is_degenerate_baseline(broken) is True


def test_truncation_reports_outermost_first():
    """Ordering: <html> takes priority over <body>, which takes priority
    over <script>. The outermost open tag is the most useful recovery
    signal."""
    # Only <script> unclosed:
    only_script = "<!DOCTYPE html><html><body><canvas></canvas><script>const x=1;</body></html>"
    assert _truncation_reason(only_script) == "unclosed <script>"
    # <body> and <script> unclosed but <html> closed (unusual but possible):
    body_script = "<!DOCTYPE html><html><body><canvas></canvas><script>const x=1;</html>"
    assert _truncation_reason(body_script) == "unclosed <body>"


def test_real_game_with_close_tags_is_not_truncated():
    """Regression guard: a complete game whose script body happens to
    contain '<html' as a string literal (not a tag) must NOT be flagged
    as truncated."""
    html = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        "<script>"
        "const note = '<html>';\n"
        + ("const filler = 'x'.repeat(100);\n" * 80)
        + "console.log(note);\n"
        "</script></body></html>"
    )
    assert _truncation_reason(html) is None
