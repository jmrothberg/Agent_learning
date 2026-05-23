"""Phase 2 regression tests — fix-turn coaching escapes the rewrite trap.

When the on-disk baseline is structurally broken (concatenated drafts,
wrapper preamble, truncated tags), the agent must:
  - report `rewrite_allowed: true` in turn_contract,
  - mark <html_file> as allowed in derive_allowed_forbidden_tags,
  - route _build_fix_prompt through truncation_recovery_instruction
    so the model is told WHY patches won't anchor and rewrite is open.

Trace 1 (chess 20260522_000304) burned iters 3-6 fighting this gate;
this test suite locks the fix in.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _is_degenerate_baseline  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _concatenated_drafts_html() -> str:
    return (
        "<!DOCTYPE html><html><body><canvas id='c' width='480' height='480'>"
        "</canvas><script>"
        "(() => {\n"
        "  const ctx = document.getElementById('c').getContext('2d');\n"
        "  const state = { x: 0 };\n"
        "  const ctx = document.getElementById('c').getContext('2d');\n"
        "  const state = { x: 0 };\n"
        + "  const filler = 'x'.repeat(8);\n" * 200
        + "})();\n"
        "</script></body></html>"
    )


def test_degenerate_baseline_flips_turn_contract_rewrite_allowed(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._snapshot_n = 1  # mid-session
    a._current_file = _concatenated_drafts_html()
    assert _is_degenerate_baseline(a._current_file) is True

    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert "<html_file>" in allowed, (
        f"degenerate baseline must allow <html_file>; got allowed={allowed}"
    )
    assert "<html_file>" not in forbidden


def test_clean_baseline_keeps_html_file_forbidden(tmp_path: Path) -> None:
    """Sanity: when the baseline is healthy and rewrite is not armed,
    <html_file> stays forbidden — the trap-escape is opt-in based on
    actual structural breakage, not always-on.
    """
    a = _make_agent(tmp_path)
    a._snapshot_n = 1
    real_html = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>"
        "(() => {\n"
        + "".join(
            f"  function fn{i}() {{ return {i}; }} const v{i} = fn{i}();\n"
            for i in range(80)
        )
        + "})();</script></body></html>"
    )
    a._current_file = real_html
    assert _is_degenerate_baseline(a._current_file) is False

    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert "<html_file>" in forbidden
    assert "<html_file>" not in allowed


def test_materialize_accepts_rewrite_when_baseline_degenerate(tmp_path: Path) -> None:
    """Direct end-to-end: with a degenerate baseline on disk, a clean
    <html_file> must materialize cleanly with no `baseline exists` reject.
    """
    a = _make_agent(tmp_path)
    a._snapshot_n = 1
    a._current_file = _concatenated_drafts_html()

    fresh_html = (
        "<!DOCTYPE html><html><body><canvas id='c' width='480' height='480'>"
        "</canvas><script>"
        "(() => { const x = 1; let y = 2; })();"
        + ("\n// real-line " * 250)
        + "</script></body></html>"
    )
    materialized, msg = asyncio.run(
        a._materialize(f"<html_file>\n{fresh_html}\n</html_file>", dry_run=True)
    )
    assert materialized is not None, f"rewrite must succeed; got msg={msg}"
    assert "baseline" not in msg.lower(), msg
