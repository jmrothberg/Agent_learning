"""Tests for `_detect_replace_repetition` and the apply_patches rejection.

DK trace 20260514_104131 iter 3 pin:
  - The model emitted a <patch> whose REPLACE body added the line
    `}else{` 11 times consecutively (token-repetition loop, a
    27B-class degenerate sampling failure mode).
  - The patch applied to disk. The next test ran against the
    poisoned file. `run_micro_probes` flagged the pattern AFTER
    the write as a WARNING, not a rejection.

The detector runs at apply time so the patch is rejected BEFORE
writing. Defenses against false positives (so legitimate switch
chains, closing braces, and template literals aren't blocked):
  - Trimmed length ≥ 6 chars.
  - Skip pure-punctuation lines.
  - Skip lines inside unbalanced backticks (template literals).
  - Requires CONSECUTIVE identical lines (so switch `case X:`
    chains stay safe — those lines are non-identical).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patches import (  # noqa: E402
    Patch,
    _detect_replace_repetition,
    apply_patches,
)


# ---------------------------------------------------------------------------
# Detector — positive cases (should trigger)
# ---------------------------------------------------------------------------


def test_dk_else_brace_11x_triggers():
    """The literal DK trace 20260514_104131 iter 3 shape."""
    body = "\n".join(["}else{"] * 11)
    result = _detect_replace_repetition(body)
    assert result is not None
    line, count = result
    assert line == "}else{"
    assert count == 11


def test_three_identical_normal_lines_trigger():
    """Minimum run is 3."""
    body = "state.x = 0;\nstate.x = 0;\nstate.x = 0;"
    result = _detect_replace_repetition(body)
    assert result is not None
    line, count = result
    assert "state.x = 0;" in line
    assert count == 3


def test_run_with_leading_trailing_distinct_lines():
    """The run can be embedded inside distinct surrounding code."""
    body = (
        "function foo() {\n"
        "  bar();\n"
        "  bar();\n"
        "  bar();\n"
        "  baz();\n"
        "}"
    )
    result = _detect_replace_repetition(body)
    assert result is not None
    line, count = result
    assert "bar();" in line
    # bar() is 6 chars after trim — exactly at the threshold. Pin it.
    assert count >= 3


# ---------------------------------------------------------------------------
# Detector — negative cases (must not false-positive)
# ---------------------------------------------------------------------------


def test_switch_statement_with_distinct_cases_safe():
    """A switch chain has consecutive lines that LOOK uniform but
    are NOT identical — every `case X:` line has a different X."""
    body = "\n".join(f"case CMD_{i}: return i;" for i in range(10))
    assert _detect_replace_repetition(body) is None


def test_two_identical_lines_below_min_run():
    """Below `min_run=3` — must not fire."""
    body = "return foo;\nreturn foo;"
    assert _detect_replace_repetition(body) is None


def test_lone_closing_brace_below_min_length():
    """Pure `}` is 1 char, below the 6-char min — never fires even
    when present many times."""
    body = "\n".join(["}"] * 10)
    assert _detect_replace_repetition(body) is None


def test_short_closer_below_min_length():
    """`});` is 3 chars after trim — still below threshold."""
    body = "\n".join(["});"] * 10)
    assert _detect_replace_repetition(body) is None


def test_pure_punctuation_line_skipped():
    """A line that's only `,` or `;` repeated must not fire even if
    it appears 10+ times consecutively."""
    body = "\n".join([","] * 10)
    assert _detect_replace_repetition(body) is None


def test_template_literal_with_repeated_lines_safe():
    """Multi-line backtick string with repeated identical lines is
    legal JS — the detector must skip lines inside unbalanced
    backticks."""
    body = (
        "const banner = `\n"
        "==========\n"
        "==========\n"
        "==========\n"
        "==========\n"
        "==========\n"
        "`;"
    )
    assert _detect_replace_repetition(body) is None


def test_empty_replace_body_returns_none():
    assert _detect_replace_repetition("") is None
    assert _detect_replace_repetition(None or "") is None


def test_one_line_replace_body_returns_none():
    assert _detect_replace_repetition("const x = 1;") is None


def test_two_line_replace_body_returns_none():
    """Even when the two lines are identical and long enough, two
    is below min_run."""
    assert _detect_replace_repetition("state.frame++;\nstate.frame++;") is None


# ---------------------------------------------------------------------------
# Integration — apply_patches rejects the patch with a specific error
# ---------------------------------------------------------------------------


def test_apply_patches_rejects_repetition_patch():
    """End-to-end: a patch with `}else{` × 11 in REPLACE is rejected
    by apply_patches before the source is touched."""
    source = "function foo() {\n  if (x) doSomething();\n}\n"
    p = Patch(
        search="if (x) doSomething();",
        replace="\n".join(["}else{"] * 11),
    )
    result = apply_patches(source, [p])
    # Patch did NOT apply.
    assert result.applied == 0
    assert result.text == source  # Source unchanged.
    # Rejection includes the offending line and the count.
    assert len(result.failed) == 1
    _, _, reason = result.failed[0]
    assert "}else{" in reason
    assert "11" in reason
    assert "repetition" in reason.lower()


def test_apply_patches_accepts_legitimate_patch_with_close_braces():
    """A patch that adds normal code with repeated short closing
    braces still applies — the defenses keep `}` / `});` chains
    safe."""
    source = "let a = 1;\n"
    p = Patch(
        search="let a = 1;",
        replace=(
            "let a = 1;\n"
            "function foo() {\n"
            "  return {\n"
            "    bar: 1\n"
            "  };\n"
            "}"
        ),
    )
    result = apply_patches(source, [p])
    assert result.applied == 1
    assert result.failed == []
    assert "function foo()" in result.text


def test_apply_patches_accepts_switch_chain():
    """Switch statements with many `case X:` lines must still apply."""
    source = "function dispatch(cmd) {\n  /* body */\n}\n"
    cases = "\n".join(f"    case CMD_{i}: return i;" for i in range(8))
    p = Patch(
        search="  /* body */",
        replace=(
            "  switch (cmd) {\n"
            f"{cases}\n"
            "    default: return -1;\n"
            "  }"
        ),
    )
    result = apply_patches(source, [p])
    assert result.applied == 1
    assert result.failed == []
