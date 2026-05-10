"""Tests for the harness changes that make probes ENFORCING (A1)
and require a clean-streak before <done/> ships (A2).

All pure-function — no Chromium, no model. Each test < 5 ms.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402


# ---------------------------------------------------------------------------
# A1 — probe failures must surface as soft_warnings (gating ok=True via
# the existing formula). We exercise the exact code shape the harness
# uses: build a stub report + probes list, run the helper, assert.
# ---------------------------------------------------------------------------


def _stub_report() -> dict:
    """Mimic what _build_report returns BEFORE the post-step that
    promotes probe failures into soft_warnings."""
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "soft_warnings": [],
        "logs": [],
        "title": "test",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "body_chars": 100,
        "body_sample": "...",
        "screenshot": None,
        "screenshot_before": None,
        "frozen_canvas": False,
        "input_test": {"ran": True, "any_change": True, "first_responsive_key": "ArrowRight"},
        "probes": [],
    }


def _apply_probe_gate(report: dict) -> None:
    """Inline copy of the post-step that lives in
    LiveBrowser.load_and_test (tools.py around line 1057). Test-only
    so we don't need to spin up Chromium to verify A1.
    """
    for p in (report.get("probes") or []):
        if p.get("ok"):
            continue
        name = p.get("name", "probe")
        expr = (p.get("expr") or "")[:80]
        err = p.get("err") or "evaluated falsy"
        report["soft_warnings"].append(
            f"PROBE FAILED [{name}]: `{expr}` — {err}. "
            "Your Phase A acceptance criterion is unmet; fix the "
            "game so it evaluates truthy."
        )
    report["ok"] = (
        len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
    )


def test_a1_no_probes_keeps_ok_true():
    r = _stub_report()
    _apply_probe_gate(r)
    assert r["ok"] is True
    assert r["soft_warnings"] == []


def test_a1_all_probes_pass_keeps_ok_true():
    r = _stub_report()
    r["probes"] = [
        {"name": "canvas_present", "expr": "...", "ok": True, "err": ""},
        {"name": "player_alive",   "expr": "...", "ok": True, "err": ""},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is True


def test_a1_one_probe_fails_flips_ok_to_false():
    """The exact failure mode from the doom-fps trace."""
    r = _stub_report()
    r["probes"] = [
        {"name": "canvas_present",  "expr": "!!document.querySelector('canvas')", "ok": True, "err": ""},
        {"name": "player_alive",    "expr": "window.game.player.health > 0",      "ok": False, "err": ""},
        {"name": "monster_exists",  "expr": "window.game.monsters.length > 0",    "ok": False, "err": ""},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is False
    # Both failed probes get a clear soft_warning so the model can act.
    matching = [w for w in r["soft_warnings"] if "PROBE FAILED" in w]
    assert len(matching) == 2
    assert any("player_alive" in w for w in matching)
    assert any("monster_exists" in w for w in matching)


def test_a1_failed_probe_with_runtime_error_surfaces_message():
    """If a probe threw at evaluate-time, its `err` should appear in
    the soft_warning so the model knows it's a syntax/typo issue."""
    r = _stub_report()
    r["probes"] = [
        {"name": "broken", "expr": "window.gam.player", "ok": False,
         "err": "TypeError: Cannot read properties of undefined"},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is False
    found = [w for w in r["soft_warnings"] if "broken" in w]
    assert found
    assert "TypeError" in found[0]


# ---------------------------------------------------------------------------
# A2 — clean-streak counter increments on ok=True, resets on ok=False.
# Default streak threshold is 2; <done/> shouldn't ship at streak=1.
# ---------------------------------------------------------------------------


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


def test_a2_streak_initial_state(tmp_path):
    a = _make_agent(tmp_path)
    assert a._consecutive_clean_iters == 0
    assert a._min_clean_streak_to_ship == 2


def test_a2_streak_increments_on_clean(tmp_path):
    """Simulate the per-iter counter update from agent.run()."""
    a = _make_agent(tmp_path)

    def step(ok: bool):
        if ok:
            a._consecutive_clean_iters += 1
        else:
            a._consecutive_clean_iters = 0

    step(True);  assert a._consecutive_clean_iters == 1
    step(True);  assert a._consecutive_clean_iters == 2
    step(True);  assert a._consecutive_clean_iters == 3
    step(False); assert a._consecutive_clean_iters == 0
    step(True);  assert a._consecutive_clean_iters == 1


def test_a2_streak_threshold_reflects_done_eligibility(tmp_path):
    """The conditional in agent.run() requires
    consecutive_clean_iters >= min_clean_streak_to_ship before honoring
    a model-issued <done/>."""
    a = _make_agent(tmp_path)
    a._consecutive_clean_iters = 1
    streak_ok = a._consecutive_clean_iters >= a._min_clean_streak_to_ship
    assert streak_ok is False  # one clean iter ≠ ship

    a._consecutive_clean_iters = 2
    streak_ok = a._consecutive_clean_iters >= a._min_clean_streak_to_ship
    assert streak_ok is True


# ---------------------------------------------------------------------------
# B1 — 3D intent detector
# ---------------------------------------------------------------------------


def test_b1_3d_keyword_first_person():
    """Two-word phrases like 'first person' detected via word-join."""
    kws = prompts_v1._detect_3d_intent("make a first person shooter")
    assert "firstperson" in kws or "first-person" in kws


def test_b1_3d_keyword_doom_like():
    kws = prompts_v1._detect_3d_intent("doom like maze game with monsters")
    # Either single 'doom' (not in our list — genre-free) won't match,
    # but 'doomlike' or 'doom-like' joined form should.
    assert "doomlike" in kws or "doom-like" in kws


def test_b1_3d_keyword_three_d():
    kws = prompts_v1._detect_3d_intent("3D voxel adventure")
    assert "3d" in kws
    assert "voxel" in kws


def test_b1_3d_no_match_for_2d():
    """Plain 2D goals should not trigger the 3D nudge."""
    assert prompts_v1._detect_3d_intent("make a snake game") == []
    assert prompts_v1._detect_3d_intent("calculator app with light theme") == []
    # Importantly — bare 'doom' alone (no 'like' / 'fps' / 'first-person')
    # doesn't fire. We don't hardcode genre names.
    assert prompts_v1._detect_3d_intent("doom") == []


def test_b1_3d_nudge_appears_in_plan_instruction():
    out = prompts_v1.plan_instruction(goal="first person shooter with monsters")
    assert "3D INTENT DETECTED" in out
    assert "three.js" in out


def test_b1_no_3d_nudge_for_2d_goal():
    out = prompts_v1.plan_instruction(goal="snake game with pixel art")
    assert "3D INTENT DETECTED" not in out


def test_b1_both_nudges_can_fire_together():
    """A '3D first person shooter with sprite art' goal should trigger
    BOTH the ART INTENT and the 3D INTENT callouts — they're additive."""
    out = prompts_v1.plan_instruction(goal="first person shooter with sprite art")
    assert "ART INTENT DETECTED" in out
    assert "3D INTENT DETECTED" in out


# ---------------------------------------------------------------------------
# B2 — Mixed sprite/procedural guideline appears in system prompt
# ---------------------------------------------------------------------------


def test_b2_sprite_procedural_mix_in_system_prompt():
    sp = prompts_v1.SYSTEM_PROMPT
    # The new guideline says: sprites for static, procedural for
    # destructible/state-rich. Look for the key concept.
    assert "DESTRUCTIBLE" in sp or "destructible" in sp
    assert "procedural" in sp.lower()


# ---------------------------------------------------------------------------
# B3 — Sprite-orientation guidance in the asset paths block
# ---------------------------------------------------------------------------


def test_b3_orientation_block_contains_rotate_pattern(tmp_path):
    import assets
    asset_dir = tmp_path / "x_assets"
    asset_dir.mkdir()
    ship_png = asset_dir / "ship.png"
    ship_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder; filter only checks exists()
    block = assets.render_asset_paths_block(
        {"ship": ship_png},
        tmp_path / "x.html",
    )
    assert "ORIENTATION" in block
    assert "ctx.rotate" in block
    assert "ctx.translate" in block
    # The pattern shows the right save/restore frame for rotation.
    assert "ctx.save" in block
    assert "ctx.restore" in block
