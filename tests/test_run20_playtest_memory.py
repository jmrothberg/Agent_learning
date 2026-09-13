"""Guards run_20 playtest memory fixes (class craft + recipe probes)."""

from __future__ import annotations

import json
from pathlib import Path

import memory as mem

REPO = Path(__file__).resolve().parent.parent


def _playbook_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in (REPO / "memory/playbook.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        out[b["id"]] = b
    return out


def _outline(oid: str) -> dict:
    for line in (REPO / "memory/implementation_outlines.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["id"] == oid:
            return o
    raise AssertionError(oid)


def _recipe(rid: str) -> dict:
    for line in (REPO / "memory/visual_playtests.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["id"] == rid:
            return o["recipe"]
    raise AssertionError(rid)


def test_run20_playbook_class_bullets_exist():
    by = _playbook_by_id()
    assert "lives-init-matches-hud" in by
    assert "fixed-shooter-field-scatter-and-death-flash" in by
    assert "overworld-enemies-controls-hud" in by
    assert "BOTTOM" in by["ramp-hazard-roll-then-tumble"]["content"]
    assert "2–3 tiles" in by["discrete-tile-stepping"]["content"] or "2-3 tiles" in by[
        "discrete-tile-stepping"
    ]["content"]
    # Fire guidance lives on outline + recipe probe (keep wireframe-fps short for Jaccard).
    assert "wireframe-fps-movement-vectors" in by
    wtraps = _outline("outline-vector-wireframe")["recipe"]["traps"]
    assert any("Space" in t or "fire" in t.lower() for t in wtraps)


def test_run20_outline_traps_cover_feedback():
    assert any("lives" in t.lower() for t in _outline("outline-lane-crossing")["recipe"]["traps"])
    vtraps = _outline("outline-vertical-platformer")["recipe"]["traps"]
    assert any("BOTTOM" in t or "bottom" in t.lower() for t in vtraps)
    assert any("SLANT" in t or "slant" in t.lower() for t in vtraps)
    ftraps = _outline("outline-fixed-shooter")["recipe"]["traps"]
    assert any("scatter" in t.lower() or "mushroom" in t.lower() for t in ftraps)
    assert any("solid" in t.lower() and "red" in t.lower() for t in ftraps)
    wtraps = _outline("outline-vector-wireframe")["recipe"]["traps"]
    assert any("Space" in t or "fire" in t.lower() for t in wtraps)


def test_run20_recipe_auto_probes_and_conditional_ensures():
    lane = _recipe("canvas-lane-crossing")
    assert any(p["name"] == "auto_lives_start_at_least_2" for p in lane["auto_probes"])
    vert = _recipe("canvas-vertical-platformer")
    assert any(p["name"] == "auto_vertical_player_starts_lower_half" for p in vert["auto_probes"])
    fix = _recipe("canvas-fixed-shooter")
    assert any(p["name"] == "auto_field_props_not_only_top_band" for p in fix["auto_probes"])
    wire = _recipe("canvas-vector-wireframe")
    assert any(p["name"] == "auto_wireframe_space_fires_projectile" for p in wire["auto_probes"])
    over = _recipe("canvas-overworld-rpg")
    assert any(p["name"] == "auto_overworld_has_combat_enemies" for p in over["auto_probes"])

    # Pac-Man pins maze bullets; Dig Dug does not.
    pac = " ".join(
        mem.collect_recipe_ensure_ids(
            "Build a Pac-Man game with pellets ghosts maze",
            active_recipe_id="canvas-grid-navigation",
        )
    )
    assert "pacman-maze-copy-dont-generate" in pac
    dig = " ".join(
        mem.collect_recipe_ensure_ids(
            "Build a Dig Dug game dig tunnels harpoon soil",
            active_recipe_id="canvas-grid-navigation",
        )
    )
    assert "pacman-maze-copy-dont-generate" not in dig
    assert "discrete-tile-stepping" in dig


def test_prompt_library_run20_title_nudges():
    by_n = {}
    for line in (REPO / "memory/prompt_library.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_n[r["n"]] = r
    assert "3 lives" in by_n[7]["prompt_640"] or "Start with 3" in by_n[7]["prompt_640"]
    assert "BOTTOM" in by_n[2]["prompt_640"]
    assert "Scatter mushrooms" in by_n[3]["prompt_640"] or "scatter" in by_n[3]["prompt_640"].lower()
    assert "Space MUST fire" in by_n[28]["prompt_640"] or "Space MUST fire" in by_n[28]["prompt"]
