"""Tests for the mid-stream repetition detector and asset/sound parser
dedupe.

Both layers protect against the same family of bug: a local model
entering a degenerate state where it emits ~200 lines of near-duplicate
output before the stall watchdog fires. The original detector watched
only short lines (≤ 80 chars) with exact match, which let through
templated long-line loops like:

    {"name":"asset_1",  "prompt":"green computer", "size":"16x16"},
    {"name":"asset_2",  "prompt":"green computer", "size":"16x16"},
    ...
    {"name":"asset_208","prompt":"green computer", "size":"16x16"},

The fix has two layers:
  1. `RepetitionDetector` (in ollama_io) runs two windows — short-line
     exact-match AND all-line digit-stripped — so numbered template
     loops are caught by the second window. The class is SHARED by
     both the Ollama and MLX streams, so tuning lives in one place.
  2. `parse_assets_block` / `parse_sounds_block` dedupe by
     (normalized_prompt, size) so even if the loop slipped past the
     stream detector, generation is still bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402
import ollama_io  # noqa: E402
import sounds  # noqa: E402


# ---------------------------------------------------------------------------
# Stream-level: _normalize_line_for_repeat
# ---------------------------------------------------------------------------


def test_normalize_collapses_trailing_numeric_suffix():
    a = ollama_io._normalize_line_for_repeat(
        '{"name":"minimap_compiler179","prompt":"green computer","size":"16x16"},'
    )
    b = ollama_io._normalize_line_for_repeat(
        '{"name":"minimap_compiler208","prompt":"green computer","size":"16x16"},'
    )
    assert a == b, "numbered variants must hash to the same bucket"


def test_normalize_preserves_distinct_alphabetic_content():
    """`const score = 0;` and `const lives = 0;` differ in identifier;
    digit-stripping alone shouldn't conflate them — that would cause
    false-positive loop detection on real code."""
    a = ollama_io._normalize_line_for_repeat("const score = 0;")
    b = ollama_io._normalize_line_for_repeat("const lives = 0;")
    assert a != b


def test_normalize_strips_inner_digits_in_size_suffix():
    """An attacker / lazy model emitting `16x16`, `32x32`, `64x64`
    suffixes on otherwise identical templated lines should still be
    detected — the size variation is incidental, not meaningful."""
    a = ollama_io._normalize_line_for_repeat('{"name":"x","size":"16x16"},')
    b = ollama_io._normalize_line_for_repeat('{"name":"x","size":"32x32"},')
    # After stripping all digits, both reduce to the same shape:
    assert a == b


def test_normalize_handles_blank_and_whitespace():
    assert ollama_io._normalize_line_for_repeat("") == ""
    assert ollama_io._normalize_line_for_repeat("   \t  ") == ""
    # Surrounding whitespace doesn't matter.
    assert (
        ollama_io._normalize_line_for_repeat("  foo  ")
        == ollama_io._normalize_line_for_repeat("foo")
    )


def test_unclosed_html_file_block_detection():
    body = "<html_file>\n<html><body><script>const x = 1;\n"
    assert ollama_io._in_unclosed_html_file_block(body) is True
    assert ollama_io._in_unclosed_html_file_block(body + "</script></body></html></html_file>") is False


def test_inline_data_bloat_grace_gate():
    partial = "<html_file>\n<html><body>...\n"
    assert ollama_io._should_grace_inline_data_bloat(
        stall_reason="inline_data_bloat",
        assembled_text=partial,
        grace_already_used=False,
    ) is True
    assert ollama_io._should_grace_inline_data_bloat(
        stall_reason="inline_data_bloat",
        assembled_text=partial,
        grace_already_used=True,
    ) is False
    assert ollama_io._should_grace_inline_data_bloat(
        stall_reason="adjacent_line_spam",
        assembled_text=partial,
        grace_already_used=False,
    ) is False


def test_inline_data_bloat_grace_denied_past_token_ceiling():
    """Past the completion-token ceiling the first-build grace is DENIED so a
    detected loop aborts immediately (trace centipede 20260615_154952: a slow
    model looped to 22k tokens / 26 min). Below the ceiling, grace still
    applies, so legitimate long builds are untouched."""
    partial = "<html_file>\n<html><body>...\n"
    # Just under the ceiling — grace still granted.
    assert ollama_io._should_grace_inline_data_bloat(
        stall_reason="inline_data_bloat",
        assembled_text=partial,
        grace_already_used=False,
        completion_tokens=ollama_io._LOOP_GRACE_TOKEN_CEILING - 1,
    ) is True
    # At/over the ceiling — grace denied → caller aborts the runaway loop.
    assert ollama_io._should_grace_inline_data_bloat(
        stall_reason="inline_data_bloat",
        assembled_text=partial,
        grace_already_used=False,
        completion_tokens=ollama_io._LOOP_GRACE_TOKEN_CEILING,
    ) is False


# ---------------------------------------------------------------------------
# RepetitionDetector — the shared class used by BOTH backends. These tests
# pin its behavior directly so we don't rely on the Ollama or MLX wrappers
# to assert correctness. If the user is on MLX (most Macs) and sees a model
# go off the rails, this is the code that catches it.
# ---------------------------------------------------------------------------


def test_detector_catches_short_line_loop():
    """Original failure shape: `</body></html>\\n</html_file>\\n` × N.
    Lines are short and exactly identical — Window 1 catches them."""
    d = ollama_io.RepetitionDetector()
    fired = False
    for _ in range(50):
        if d.feed("</body></html>\n</html_file>\n"):
            fired = True
            break
    assert fired


def test_detector_catches_long_line_numbered_template_loop():
    """The bug that motivated this rewrite: 200+ JSON entries, each ~155
    chars, identical except for a numeric suffix. Window 2 (digit-
    stripped) catches them; Window 1 wouldn't (lines are too long
    AND they're all distinct strings before normalization)."""
    d = ollama_io.RepetitionDetector()
    fired_at = None
    for i in range(1, 100):
        line = (
            '{"name":"minimap_compiler' + str(i) +
            '","prompt":"pixel-art green computer for minimap '
            'compiler room marker, transparent background","size":"16x16"},\n'
        )
        if d.feed(line):
            fired_at = i
            break
    # Should fire by the time the digit-stripped window has 12 entries
    # (well before 100 iterations).
    assert fired_at is not None
    assert fired_at < 30


def test_detector_does_not_false_positive_on_real_code():
    """A short healthy stream of distinct lines must NOT trip the
    detector. We feed lines from a fictitious but realistic Snake game
    and assert the detector stays quiet."""
    d = ollama_io.RepetitionDetector()
    snippets = [
        "<!DOCTYPE html>\n",
        "<html>\n",
        "<head><title>Snake</title></head>\n",
        "<body>\n",
        "<canvas id='cvs' width='800' height='600'></canvas>\n",
        "<script>\n",
        "const cvs = document.getElementById('cvs');\n",
        "const ctx = cvs.getContext('2d');\n",
        "let snake = [{x: 10, y: 10}];\n",
        "let dir = {x: 1, y: 0};\n",
        "let food = {x: 15, y: 15};\n",
        "let score = 0;\n",
        "function update() {\n",
        "  const head = {x: snake[0].x + dir.x, y: snake[0].y + dir.y};\n",
        "  snake.unshift(head);\n",
        "  if (head.x === food.x && head.y === food.y) {\n",
        "    score += 1;\n",
        "    food = {x: Math.floor(Math.random()*40), y: Math.floor(Math.random()*30)};\n",
        "  } else {\n",
        "    snake.pop();\n",
        "  }\n",
        "}\n",
        "requestAnimationFrame(update);\n",
    ]
    for s in snippets:
        assert not d.feed(s), f"false positive on real code at: {s!r}"


def test_normalize_collapses_mid_identifier_digit_runs():
    """The donkey-kong 20260523_091509 trace: model emitted

        const LADDER783_X=440;const LADDER784_Y1=48;const LADDER784_Y2=140;…

    for 55 minutes / 2588 tokens before being cut off externally. The
    original normalizer used `\\d+\\b` which required the digit run to
    end at a word boundary, and `_` is a word char, so `LADDER784_Y2`
    kept `784` and the templates never collapsed to one bucket. The
    current pattern drops the boundary anchor so mid-identifier counters
    are stripped too."""
    a = ollama_io._normalize_line_for_repeat("const LADDER783_X=440")
    b = ollama_io._normalize_line_for_repeat("const LADDER784_Y1=48")
    c = ollama_io._normalize_line_for_repeat("const LADDER999_Y2=140")
    # Counter portions collapse; assignment values (preceded by `=`,
    # not a letter/underscore) stay intact and disambiguate when they
    # legitimately differ.
    assert "783" not in a and "784" not in b and "999" not in c
    # Right-hand-side numeric literals must survive — these are real
    # values, not counters, and erasing them would conflate distinct
    # statements.
    assert "440" in a and "48" in b and "140" in c


def test_detector_catches_semicolon_chained_template_loop_with_no_newlines():
    """Reproduces the donkey-kong 20260523_091509 stream shape exactly:
    one long line, semicolon-terminated statements, embedded counters,
    no `\\n` anywhere. Before the fix the detector's line buffer never
    flushed (it only split on `\\n`) so all four windows stayed empty
    and the model burned ~55 min at 0.8 tok/s.

    Streamed as one piece (no newline) — the detector must still see
    the per-statement boundaries via `;` and fire."""
    d = ollama_io.RepetitionDetector()
    chain = "".join(
        f"const LADDER{i}_X=440;const LADDER{i}_Y1=48;const LADDER{i}_Y2=140;"
        for i in range(1, 30)
    )
    fired = d.feed(chain)
    assert fired, "semicolon-chained template loop must fire even with no \\n"


def test_detector_does_not_false_positive_on_semicolon_chained_distinct_statements():
    """Real coder output often packs multiple distinct statements onto
    one line:

        const W=480,H=640;const canvas=document.getElementById('c');
        const ctx=canvas.getContext('2d');ctx.scale(DPR,DPR);

    The `;`-aware split must NOT trip on this. Each statement is
    structurally unique after normalization."""
    d = ollama_io.RepetitionDetector()
    line = (
        "const W=480,H=640;const canvas=document.getElementById('c');"
        "const ctx=canvas.getContext('2d');ctx.scale(DPR,DPR);"
        "canvas.width=W*DPR;canvas.height=H*DPR;"
        "canvas.style.width=W+'px';canvas.style.height=H+'px';"
    )
    # Feed the line several times in a row to be safe — these distinct
    # statements should never collapse to ≤2 unique templates.
    for _ in range(3):
        assert not d.feed(line), "false positive on healthy single-line code"


def test_detector_state_is_per_instance():
    """Constructing a new detector resets state — so a wedged previous
    stream can't poison the next one."""
    d1 = ollama_io.RepetitionDetector()
    for _ in range(50):
        d1.feed("dup\n")
    # d1 has fired by now; verify a fresh instance is clean.
    d2 = ollama_io.RepetitionDetector()
    assert not d2.feed("alpha\n")
    assert not d2.feed("beta\n")


# ---------------------------------------------------------------------------
# parse_assets_block — dedupe
# ---------------------------------------------------------------------------


def test_parse_assets_block_dedupes_numbered_template_loop():
    """Reproduces the failure mode that motivated this fix: 200+ entries
    that all describe the same sprite, only differing in a numbered
    `name` field. After dedupe the parser returns ONE entry, not 200,
    so generate_assets makes one GPU call instead of being hit with the
    `_MAX_ASSETS_PER_TURN` cap silently."""
    items = ",\n".join(
        f'{{"name":"minimap_compiler{i}","prompt":"green computer for minimap","size":"16x16"}}'
        for i in range(1, 201)
    )
    reply = f"<assets>\n[\n{items}\n]\n</assets>"
    out = assets.parse_assets_block(reply)
    assert len(out) == 1
    assert out[0]["prompt"] == "green computer for minimap"
    assert out[0]["size"] == (16, 16)


def test_parse_assets_block_keeps_distinct_prompts():
    """Healthy plans with several distinct sprites must NOT be deduped."""
    reply = (
        "<assets>"
        '[{"name":"ship",     "prompt":"pixel-art spaceship",  "size":"64x64"},'
        ' {"name":"asteroid", "prompt":"grey rocky asteroid",  "size":"64x64"},'
        ' {"name":"explosion","prompt":"orange explosion",     "size":"96x96"}]'
        "</assets>"
    )
    out = assets.parse_assets_block(reply)
    assert len(out) == 3
    assert {a["name"] for a in out} == {"ship", "asteroid", "explosion"}


def test_parse_assets_block_dedupes_case_and_whitespace_variants():
    """Cache key already normalizes; dedupe must use the same shape so
    `Pixel Spaceship` and `pixel  spaceship  ` don't both generate."""
    reply = (
        "<assets>"
        '[{"name":"a","prompt":"Pixel Spaceship","size":"64x64"},'
        ' {"name":"b","prompt":"pixel  spaceship","size":"64x64"}]'
        "</assets>"
    )
    out = assets.parse_assets_block(reply)
    assert len(out) == 1


def test_parse_assets_block_dedupe_respects_size():
    """Same prompt at different sizes is not the same asset — both
    legitimately need separate generation."""
    reply = (
        "<assets>"
        '[{"name":"small","prompt":"pixel ship","size":"32x32"},'
        ' {"name":"big",  "prompt":"pixel ship","size":"128x128"}]'
        "</assets>"
    )
    out = assets.parse_assets_block(reply)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# parse_sounds_block — dedupe (mirror of assets)
# ---------------------------------------------------------------------------


def test_parse_sounds_block_dedupes_numbered_template_loop():
    items = ",\n".join(
        f'{{"name":"laser_{i}","prompt":"short retro arcade laser","duration":0.4}}'
        for i in range(1, 51)
    )
    reply = f"<sounds>[\n{items}\n]</sounds>"
    out = sounds.parse_sounds_block(reply)
    assert len(out) == 1


def test_parse_sounds_block_dedupe_respects_duration_and_loop():
    """A short SFX and a long loop with the same prompt are different."""
    reply = (
        "<sounds>"
        '[{"name":"a","prompt":"chiptune","duration":0.5,"loop":false},'
        ' {"name":"b","prompt":"chiptune","duration":12.0,"loop":true}]'
        "</sounds>"
    )
    out = sounds.parse_sounds_block(reply)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# A2 — DeliberationDetector
# ---------------------------------------------------------------------------


def test_deliberation_detector_fires_after_threshold_with_no_tag():
    """Unique-text rambling with no output tag must trip the detector
    once the char budget is exhausted. This is the failure mode from
    games/traces/game-of-space-invaders_20260512_084906: iter 2 burned
    14k tokens of pure reasoning before being aborted by overall_seconds.
    """
    d = ollama_io.DeliberationDetector(threshold_chars=200)
    fired = False
    text = (
        "Let me think about this. Wait, actually, hmm. "
        "But what about the alien bullets array? "
        "Maybe state.player is undefined. Let me reconsider. "
        "Actually wait. But state.player should exist. Hmm. "
        "Let me think again. Maybe the issue is somewhere else entirely."
    )
    # Feed one char at a time to simulate streaming.
    for ch in text:
        if d.feed(ch):
            fired = True
            break
    assert fired, "no-tag rambling should trip after threshold_chars"
    assert d.stall_reason == "deliberation_loop"


def test_deliberation_detector_does_not_fire_when_tag_appears_early():
    """Real productive replies start with <diagnose> or <patch> — the
    detector must not fire if any output tag opener has appeared.
    """
    d = ollama_io.DeliberationDetector(threshold_chars=100)
    text = "<diagnose>The alien bullets array is being mutated mid-loop.</diagnose>" + "x" * 500
    for ch in text:
        assert not d.feed(ch), f"tripped on legal output containing tag at len={len(d._buf)}"


def test_deliberation_detector_does_not_fire_on_full_html_file():
    """Phase B first build streams a full <html_file>...</html_file>;
    that's hundreds of KB of text but always begins with a tag opener.
    """
    d = ollama_io.DeliberationDetector(threshold_chars=50)
    payload = "<html_file>\n<!DOCTYPE html><html><body>" + "x" * 10000 + "</body></html>\n</html_file>"
    for chunk in [payload[i:i + 17] for i in range(0, len(payload), 17)]:
        assert not d.feed(chunk)


def test_deliberation_detector_respects_disable_env(monkeypatch):
    """DISABLE_DELIBERATION_DETECTOR=1 is the AB-flag hook so the bench
    script can compare ON vs OFF arms without code edits."""
    monkeypatch.setenv("DISABLE_DELIBERATION_DETECTOR", "1")
    d = ollama_io.DeliberationDetector(threshold_chars=50)
    for _ in range(200):
        assert not d.feed("more reasoning, no tag, ")
    assert d.stall_reason is None


def test_deliberation_detector_catches_tag_split_across_pieces():
    """When `<html_file>` arrives split across two stream pieces, the
    detector must still recognize it — otherwise long replies that
    happen to land a boundary mid-tag would trip the guard.
    """
    d = ollama_io.DeliberationDetector(threshold_chars=10000)
    # Pump 270 chars of prose so the buffer is sticky, then split a
    # line-start tag across pieces.
    for ch in "preamble " * 30:
        d.feed(ch)
    assert d.feed("\n<html_") is False
    assert d.feed("file>") is False
    # 'seen_tag' should now be latched True.
    for _ in range(10000):
        assert not d.feed("x")


# ---------------------------------------------------------------------------
# <think>-aware deliberation (classic-doom 20260512_111015 regression)
#
# The failure: iter 1 emitted 134,850 chars of reply, all of it inside a
# never-closed <think> block, and the old detector latched on a literal
# `<!DOCTYPE html>` the model had quoted in its reasoning prose. The
# detector then sat silent through 32K+ more tokens of pure CoT.
# ---------------------------------------------------------------------------


def test_deliberation_inside_think_ignores_doctype_literal():
    """Model writes `<!DOCTYPE html>` inside <think> — must NOT latch."""
    d = ollama_io.DeliberationDetector(
        threshold_chars=6000, think_threshold_chars=20000,
    )
    text = (
        "<think>Let me plan the build. I'd start with `<!DOCTYPE html>` "
        "then `<html lang='en'>` and a head with meta-viewport. "
    )
    for ch in text:
        d.feed(ch)
    assert d._think_depth == 1
    assert d._seen_tag is False, (
        "DOCTYPE/<html> inside <think> must not count as real output"
    )


def test_deliberation_inside_think_aborts_at_higher_threshold():
    """The 134KB iter-1 disaster: model rambles inside <think> forever.
    The detector must abort once the inside-think budget is exhausted.
    """
    d = ollama_io.DeliberationDetector(
        threshold_chars=6000, think_threshold_chars=15000,
    )
    # Emit <think> opener then keep streaming reasoning prose with
    # casual HTML mentions — old detector would have latched on the
    # first `<html` and never aborted.
    fired = False
    chunks = ["<think>"]
    for i in range(2000):
        chunks.append(
            f"step {i}: the spec says <!DOCTYPE html>\\n<html lang='en'>... "
        )
    for piece in chunks:
        if d.feed(piece):
            fired = True
            break
    assert fired, "must abort inside an unbounded <think> block"
    assert d.stall_reason == "deliberation_loop"


def test_deliberation_outside_think_latches_normally():
    """Without any <think> wrapping, real output tags still latch the
    same as before — no regression on non-reasoning models."""
    d = ollama_io.DeliberationDetector(threshold_chars=6000)
    # 500 chars of preamble (well under 6000) then a real <html_file>.
    for _ in range(50):
        d.feed("preamble preamble preamble ")
    assert d.feed("\n<html_file>") is False
    assert d._seen_tag is True


def test_deliberation_after_think_close_latches_on_real_output():
    """Model emits <think>...</think> then real output. Once depth
    returns to 0, the detector resumes latching on opener tags."""
    d = ollama_io.DeliberationDetector(
        threshold_chars=6000, think_threshold_chars=20000,
    )
    d.feed("<think>some short reasoning</think>\n\n<patch>SEARCH/REPLACE</patch>")
    assert d._think_depth == 0
    assert d._seen_tag is True


def test_deliberation_think_open_close_split_across_pieces():
    """Streaming may split `</think>` mid-tag. Depth bookkeeping must
    survive: piece1 = `<think>x</thi`, piece2 = `nk>`."""
    d = ollama_io.DeliberationDetector(threshold_chars=6000)
    d.feed("<think>x</thi")
    assert d._think_depth == 1, "open seen, close not yet"
    d.feed("nk>")
    assert d._think_depth == 0, "close completed across boundary"


def test_deliberation_doesnt_drop_below_zero_on_stray_close():
    """A stray </think> with no matching open must clamp depth at 0,
    not drop negative (which would let openers inside future <think>
    blocks accidentally latch).
    """
    d = ollama_io.DeliberationDetector(threshold_chars=6000)
    d.feed("</think>")
    assert d._think_depth == 0
    d.feed("<think>")
    assert d._think_depth == 1
