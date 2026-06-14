"""Catch "MISSING <asset>" placeholders the agent used to ship silently.

Covers the three fixes from the 2026-06-14 dragon's-lair trace:
  1. Deterministic asset-miss probe (sprite() helper + injector).
  2. Defect-aware vision-judge note parser.
  3. Structured-checklist routing through the local VLM.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import inspect
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import vision_judge  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ----------------------------------------------------------------------
# Tier 1 — deterministic asset-miss probe.
# ----------------------------------------------------------------------


def test_sprite_helper_records_misses() -> None:
    """The recommended sprite() helper template must record unresolved
    keys to window.__assetMisses so the harness can probe them."""
    import assets

    src = inspect.getsource(assets)
    assert "window.__assetMisses" in src
    assert "__assetMisses = window.__assetMisses || {}" in src


def test_asset_miss_probe_injected_when_assets_present(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"hero_idle": "x/hero_idle.png"}
    a._probes = []
    a._maybe_inject_asset_miss_probe()
    names = [p.get("name") for p in a._probes]
    assert "auto_no_missing_asset_placeholders" in names


def test_asset_miss_probe_absent_without_assets(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {}
    a._probes = []
    a._maybe_inject_asset_miss_probe()
    assert a._probes == []


def test_asset_miss_probe_not_duplicated(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"hero_idle": "x/hero_idle.png"}
    a._probes = []
    a._maybe_inject_asset_miss_probe()
    a._maybe_inject_asset_miss_probe()
    names = [p.get("name") for p in a._probes]
    assert names.count("auto_no_missing_asset_placeholders") == 1


# ----------------------------------------------------------------------
# Tier 2 — defect-aware note parser.
# ----------------------------------------------------------------------


def test_parser_recovers_missing_finding_over_trailing_line() -> None:
    """The exact failure shape from the dragon's-lair trace: the real
    finding is buried mid-prose; the reply trails off with a UI line."""
    raw = (
        "**1. Analyze the Screenshot:**\n"
        "*   **Visuals:** In the center there's a large magenta rectangle "
        'with the text "MISSING hero_idle". This indicates a missing asset.\n'
        "*   Bottom: Touch controls (Up, Down, Left, Right, Jump)\n"
    )
    _progress, note = vision_judge._parse(raw)
    low = note.lower()
    assert "missing" in low or "magenta" in low
    assert "touch controls" not in low


def test_parser_prefers_defect_sentence_not_last_line() -> None:
    raw = (
        "The scene looks okay overall.\n"
        "The player sprite is clipped at the right edge.\n"
        "Score is shown in the top-left corner.\n"
    )
    _p, note = vision_judge._parse(raw)
    assert "clipped" in note.lower()


def test_parser_falls_back_to_last_line_without_cue() -> None:
    raw = "Looks fine.\nThe HUD shows the score and lives.\n"
    _p, note = vision_judge._parse(raw)
    assert "hud" in note.lower() or "score" in note.lower()


def test_defect_cue_note_survives_actionability_gate() -> None:
    """A defect sentence with no change-verb (e.g. 'colored box') must
    not be dropped by _clean_actionable_vision_note."""
    note = "The hero appears as a colored box instead of a sprite."
    cleaned = GameAgent._clean_actionable_vision_note(note)
    assert cleaned  # not suppressed
    assert "colored box" in cleaned.lower()


# ----------------------------------------------------------------------
# Tier 3 — structured-checklist routing through the local VLM.
# ----------------------------------------------------------------------


def test_run_local_vlm_prompt_exported() -> None:
    assert hasattr(vision_judge, "run_local_vlm_prompt")
    assert inspect.iscoroutinefunction(vision_judge.run_local_vlm_prompt)


def test_vision_judge_routes_to_structured_when_recipe_active() -> None:
    """When a visual recipe matched, _run_vision_judge must try the
    structured checklist before the open-ended progress judge."""
    src = inspect.getsource(GameAgent._run_vision_judge)
    assert "_run_structured_local_vlm_critique" in src
    assert "_active_visual_playtest_recipe_id" in src


def test_structured_critique_helper_uses_checklist_pipeline() -> None:
    src = inspect.getsource(GameAgent._run_structured_local_vlm_critique)
    assert "_build_visual_playtest_prompt" in src
    assert "_parse_visual_playtest_response" in src
    assert "_format_visual_playtest_critique" in src
    assert "run_local_vlm_prompt" in src
