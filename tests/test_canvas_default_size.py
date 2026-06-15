"""Tests for the fix round "BoN Asset Paths, Patch Safety, Tiny Canvas,
Noise Gates" (fight trace 20260611_145321, both user reports).

Covered, all pure-function / source-pinned — no model, no Chromium:

  0. BoN candidates tested from out_path.parent (relative assets resolve)
  1. CANVAS-DEFAULT-SIZE live warning + report-header default callout
  2. Pre-browser micro-probe warning for unsized <canvas>
  3. Skeleton-preservation line in the first-build prompt
  4. Pre-commit patch bracket validation (reject, keep baseline)
  5. Noise gates: perf criteria advisory-only; asset_stats accepts onload
  5b. Orientation audit selection / verdict parse / mirror flip
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _patch_set_bracket_break  # noqa: E402
from tools import (  # noqa: E402
    LiveBrowser,
    _canvas_default_size_warning,
    _is_unverifiable_perf_criterion,
    format_report_for_model,
    run_micro_probes,
)


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=4,
        memory_root=str(tmp_path / "memory"),
    )


BALANCED_HTML = (
    "<!DOCTYPE html><html><body>"
    "<canvas id=\"c\" width=\"800\" height=\"500\"></canvas>"
    "<script>\nfunction f(){ if(a){ b(); } }\nf();\n</script>"
    "</body></html>"
)


# ---------------------------------------------------------------------------
# 0. BoN candidate temp path
# ---------------------------------------------------------------------------

def test_bon_candidates_written_to_out_path_parent():
    """The scorer must test candidates from the SAME dir as the real game
    so relative ./<session>_assets/ paths resolve (the blocks-instead-of-
    sprites failure)."""
    src = inspect.getsource(GameAgent._generate_and_score_candidates)
    assert "self.out_path.parent" in src
    assert "self.snapshots_dir / f\"cand_" not in src


def test_bon_candidate_temp_file_cleaned_up():
    src = inspect.getsource(GameAgent._generate_and_score_candidates)
    assert "finally" in src
    assert "unlink" in src


# ---------------------------------------------------------------------------
# 1. CANVAS-DEFAULT-SIZE live warning
# ---------------------------------------------------------------------------

def test_default_size_unsized_canvas_warns():
    html = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>draw();</script></body></html>"
    msg = _canvas_default_size_warning({"width": 300, "height": 150}, html)
    assert msg and "CANVAS-DEFAULT-SIZE" in msg


def test_default_size_sized_by_attribute_silent():
    html = "<canvas id='c' width='800' height='500'></canvas>"
    assert _canvas_default_size_warning({"width": 300, "height": 150}, html) is None


def test_default_size_sized_by_js_silent():
    html = "<canvas id='c'></canvas><script>cvs.width = 800; cvs.height = 500;</script>"
    assert _canvas_default_size_warning({"width": 300, "height": 150}, html) is None


def test_default_size_non_default_dimensions_silent():
    html = "<canvas id='c'></canvas>"
    assert _canvas_default_size_warning({"width": 800, "height": 500}, html) is None


def test_default_size_no_canvas_info_silent():
    assert _canvas_default_size_warning(None, "<canvas></canvas>") is None


def test_load_and_test_wires_default_size_check():
    src = inspect.getsource(LiveBrowser.load_and_test)
    assert "_canvas_default_size_warning" in src


def _report_base() -> dict:
    return {
        "errors": [], "page_errors": [], "console_errors": [],
        "soft_warnings": [], "warnings": [], "logs": [],
        "input_listeners": {}, "body_chars": 0, "body_sample": "",
    }


def test_report_header_flags_browser_default():
    report = {
        "ok": False,
        "title": "t",
        "canvas": {"width": 300, "height": 150, "raf_ran": True, "blank": False},
        **_report_base(),
    }
    txt = format_report_for_model(report)
    assert "300x150" in txt and "BROWSER DEFAULT" in txt


def test_report_header_normal_size_no_flag():
    report = {
        "ok": True,
        "title": "t",
        "canvas": {"width": 800, "height": 500, "raf_ran": True, "blank": False},
        **_report_base(),
    }
    txt = format_report_for_model(report)
    assert "800x500" in txt and "BROWSER DEFAULT" not in txt


# ---------------------------------------------------------------------------
# 2. Pre-browser micro-probe warning
# ---------------------------------------------------------------------------

def _pad(html: str) -> str:
    return html + "<!--" + "x" * 250 + "-->"


def test_micro_probe_unsized_canvas_warns():
    html = _pad(
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        "<script>function draw(){}</script></body></html>"
    )
    rep = run_micro_probes(html)
    assert any("300x150" in w for w in rep["warnings"])
    # warning, never an error — Chromium check is authoritative
    assert not any("300x150" in e for e in rep["errors"])


def test_micro_probe_sized_canvas_silent():
    html = _pad(
        "<!DOCTYPE html><html><body><canvas id='c' width='800' height='500'></canvas>"
        "<script>function draw(){}</script></body></html>"
    )
    rep = run_micro_probes(html)
    assert not any("300x150" in w for w in rep["warnings"])


def test_micro_probe_js_sized_canvas_silent():
    html = _pad(
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        "<script>const c=document.querySelector('canvas'); c.width=800;</script>"
        "</body></html>"
    )
    rep = run_micro_probes(html)
    assert not any("300x150" in w for w in rep["warnings"])


# ---------------------------------------------------------------------------
# 3. Skeleton-preservation line in first-build prompt
# ---------------------------------------------------------------------------

def test_first_build_prompt_preserves_canvas_sizing():
    from prompts_v1 import first_build_instruction
    txt = first_build_instruction("<canvas width='800'></canvas>")
    assert "canvas sizing" in txt
    assert "300x150" in txt


# ---------------------------------------------------------------------------
# 4. Pre-commit patch bracket validation
# ---------------------------------------------------------------------------

class _P:
    def __init__(self, search: str, replace: str):
        self.search = search
        self.replace = replace


def test_bracket_break_detected_and_block_named():
    broken = BALANCED_HTML.replace("if(a){ b(); } }", "if(a){ b(); }")
    msg = _patch_set_bracket_break(
        BALANCED_HTML, broken, [_P("if(a){ b(); } }", "if(a){ b(); }")],
    )
    assert msg and msg.startswith("patch set rejected")
    assert "block 1" in msg and "{}" in msg


def test_balanced_patch_passes():
    assert _patch_set_bracket_break(
        BALANCED_HTML, BALANCED_HTML, [_P("b();", "c();")],
    ) is None


def test_already_broken_baseline_not_blamed_on_patch():
    broken = BALANCED_HTML.replace("} }", "}")
    assert _patch_set_bracket_break(broken, broken, [_P("x", "y")]) is None


def test_materialize_rejects_bracket_breaking_patch(tmp_path):
    """Integration: a matching patch that amputates a closing brace must be
    rejected atomically — _materialize returns None and the baseline is
    untouched (the iter-2 sprite() amputation from the trace)."""
    agent = _make_agent(tmp_path)
    agent._current_file = BALANCED_HTML
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "function f(){ if(a){ b(); } }\n"
        "=======\n"
        "function f(){ if(a){ b();\n"
        ">>>>>>> REPLACE\n"
        "</patch>"
    )
    html, msg = asyncio.run(agent._materialize(reply, dry_run=True))
    assert html is None
    assert "patch set rejected" in msg
    assert agent._current_file == BALANCED_HTML


def test_materialize_accepts_balanced_patch(tmp_path):
    agent = _make_agent(tmp_path)
    agent._current_file = BALANCED_HTML
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "function f(){ if(a){ b(); } }\n"
        "=======\n"
        "function f(){ if(a){ b(); c(); } }\n"
        ">>>>>>> REPLACE\n"
        "</patch>"
    )
    html, msg = asyncio.run(agent._materialize(reply, dry_run=True))
    assert html is not None and "c();" in html
    assert "applied 1/1" in msg


def test_bracket_reject_retry_message_is_targeted():
    """The run loop must send the bracket message itself, not the generic
    patch_retry_instruction (whose failed-list would be empty)."""
    src = inspect.getsource(GameAgent.run)
    assert 'materialize_msg.startswith("patch set rejected")' in src


# ---------------------------------------------------------------------------
# 5. Noise gates
# ---------------------------------------------------------------------------

def test_perf_criterion_detected():
    assert _is_unverifiable_perf_criterion(
        "Performance stays smooth (60fps) under stress "
        "(20 fireballs + 10 hit sparks active simultaneously)"
    )
    assert _is_unverifiable_perf_criterion("no slowdown with many enemies")
    assert _is_unverifiable_perf_criterion("frame rate stays stable")


def test_behavioral_criterion_not_perf():
    assert not _is_unverifiable_perf_criterion("player can jump over the pit")
    assert not _is_unverifiable_perf_criterion(
        "fireball travels across the screen and damages the opponent"
    )


def test_perf_gap_skipped_in_synthesis():
    src = inspect.getsource(LiveBrowser.load_and_test)
    assert "_is_unverifiable_perf_criterion" in src


def test_asset_stats_accepts_onload():
    src = inspect.getsource(LiveBrowser._run_opening_book_recipes)
    assert '"onload" in html_text' in src


# ---------------------------------------------------------------------------
# 5b. Sprite orientation is a memory/code convention, not pipeline policy.
# The pin/audit/flip machinery was removed; assert it is gone so it does not
# silently creep back (facing lives in the playbook, not the asset pipeline).
# ---------------------------------------------------------------------------

def test_orientation_pipeline_machinery_removed():
    import assets
    for gone in (
        "pin_sprite_orientation",
        "select_orientation_audit_targets",
        "parse_orientation_verdicts",
        "flip_sprite_horizontal",
    ):
        assert not hasattr(assets, gone), f"assets.{gone} should be removed"
    assert not hasattr(GameAgent, "_audit_sprite_orientation")
