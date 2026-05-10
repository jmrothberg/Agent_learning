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

from agent import _detect_block_bloat  # noqa: E402
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
