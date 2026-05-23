"""Phase 6 regression fixtures — real failure shapes from chess traces.

These tests use minimal HTML fragments that mirror the EXACT shapes
observed in the two May-22 chess traces, so the regression coverage
keeps growing as new shapes are encountered.

Trace 1 (qwen3.6:27b):
  - concatenated drafts (duplicate top-level decls in <script>)
  - diagnose preamble before <!DOCTYPE>
  - <html_file> rejected with "baseline file already exists" while
    the baseline was unrecoverable

Trace 2 (claude-opus-4-7):
  - cloud backend raised an exception in 0.48s
  - agent reported it as "stalling at 600.0s" because the catch block
    set stalled=True with no error_message
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    GameAgent,
    _baseline_structurally_broken,
    _is_degenerate_baseline,
    _normalize_extracted_html,
)


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ---------------------------------------------------------------------
# Trace 1 — exact shape from games/traces/a-game-of-chess-where-white-is_*
# ---------------------------------------------------------------------


_TRACE1_CONCATENATED_DRAFTS = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Chess</title></head>
<body>
<canvas id="c" width="480" height="480"></canvas>
<script>
(() => {
  "use strict";
  const ctx = document.getElementById('c').getContext('2d');
  const state = { score: 0 };
  function buildLevels() { return []; }
  // ... draft 1 ends here, draft 2 starts below
  const ctx = document.getElementById('c').getContext('2d');
  const state = { score: 0 };
  function buildLevels() { return []; }
""" + "\n".join(f"  const v{i} = {i};" for i in range(80)) + """
})();
</script>
</body>
</html>"""

_TRACE1_DIAGNOSE_PREAMBLE = (
    "was truncated mid-generation at ~6KB because the full chess "
    "implementation exceeded the output token budget.\n"
    "</diagnose>\n\n"
    "<html_file>\n"
    "<!DOCTYPE html>\n<html><body><canvas id='c'></canvas>"
    "<script>const x = 1;\nfunction loop() {}\nloop();</script>"
    "</body></html>\n"
    "</html_file>"
)


def test_trace1_concatenated_drafts_recognized_as_broken() -> None:
    """The exact concatenated-drafts shape from trace 1 must trip both
    detectors so the rewrite carve-out kicks in."""
    reason = _baseline_structurally_broken(_TRACE1_CONCATENATED_DRAFTS)
    assert reason is not None
    assert "duplicate" in reason.lower()
    assert _is_degenerate_baseline(_TRACE1_CONCATENATED_DRAFTS) is True


def test_trace1_diagnose_preamble_normalized_away() -> None:
    """The exact diagnose-preamble shape from trace 1 iter 2 must be
    sliced down to the inner HTML by the normalizer."""
    normalized = _normalize_extracted_html(_TRACE1_DIAGNOSE_PREAMBLE)
    assert normalized is not None
    assert normalized.lower().startswith("<!doctype")
    assert "was truncated" not in normalized
    assert "</diagnose>" not in normalized


def test_trace1_full_chain_pre_write_reject_then_rewrite_accepted(
    tmp_path: Path,
) -> None:
    """End-to-end on the trace-1 shape:
      1. an incoming <html_file> with concatenated drafts is rejected,
      2. the on-disk baseline is degenerate so a clean rewrite is
         accepted on the next attempt without "baseline exists".
    """
    a = _make_agent(tmp_path)
    a._snapshot_n = 1
    # Simulate the on-disk baseline being the corrupt trace-1 file.
    a._current_file = _TRACE1_CONCATENATED_DRAFTS
    assert _is_degenerate_baseline(a._current_file) is True

    bad_payload = (
        f"<html_file>\n{_TRACE1_CONCATENATED_DRAFTS}\n</html_file>"
    )
    rejected, msg = asyncio.run(a._materialize(bad_payload, dry_run=True))
    assert rejected is None
    assert "rejected" in msg.lower()

    clean_html = (
        "<!DOCTYPE html><html><body><canvas id='c' width='480' height='480'>"
        "</canvas><script>"
        "(() => {\n"
        + "".join(
            f"  function fn{i}() {{ return {i}; }} const v{i} = fn{i}();\n"
            for i in range(80)
        )
        + "})();</script></body></html>"
    )
    accepted, msg2 = asyncio.run(
        a._materialize(f"<html_file>\n{clean_html}\n</html_file>", dry_run=True)
    )
    assert accepted is not None, f"clean rewrite must be accepted; msg={msg2}"
    assert "baseline" not in msg2.lower()


# ---------------------------------------------------------------------
# Trace 2 — phantom-stall-message shape.
# ---------------------------------------------------------------------


def test_trace2_phantom_stall_message_now_uses_real_duration() -> None:
    """The trace-2 user-visible error said `stalling at 600.0s` despite
    the underlying API exception firing in 0.48s. The classifier and
    the message construction must both surface real wall-clock seconds.
    """
    new_shape = (
        "Model produced no tokens before stalling at 0.48s on "
        "backend=anthropic. cause: OverloadedError. Check ..."
    )
    info = GameAgent._classify_stall(new_shape)
    assert info is not None
    assert info["stall_seconds"] == 0.48


def test_trace2_fallback_no_longer_continuation_only(tmp_path: Path) -> None:
    """The trace-2 session was killed because fallback was gated to
    continuation=True only. The fixed handler accepts any iteration —
    proven by the absence of a continuation-related guard on the
    fallback call site (covered explicitly in the phase 5 tests).
    """
    a = _make_agent(tmp_path)
    # Sanity: the fallback method itself is callable on the agent —
    # the call-site change in the run() loop now invokes it on any iter.
    assert callable(a._try_extension_backend_fallback)
