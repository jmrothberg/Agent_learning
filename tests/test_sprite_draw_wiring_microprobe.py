"""Micro-probe: catch fillRect-only entity draws when session PNGs exist."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import run_micro_probes  # noqa: E402


def _minimal_html(body: str) -> str:
    return (
        "<!DOCTYPE html><html><head></head><body>"
        f"<canvas width='800' height='600'></canvas><script>{body}</script>"
        "</body></html>"
    )


def test_sprite_draw_wiring_flags_fillrect_only(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const ctx=document.querySelector('canvas').getContext('2d');"
        "function draw(){"
        "ctx.fillRect(0,0,48,48);ctx.fillRect(50,0,48,48);"
        "ctx.fillRect(100,0,48,48);ctx.fillRect(150,0,48,48);"
        "}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert rep["ok"] is False
    assert any("SPRITE_DRAW_WIRING" in e for e in rep["errors"])


def test_sprite_draw_wiring_passes_when_sprite_called(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "function sprite(k){ctx.drawImage(ASSETS[k],0,0);}"
        "function draw(){for(const k of Object.keys(ASSETS))sprite(k);}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert not any("SPRITE_DRAW_WIRING" in e for e in rep.get("errors") or [])


def test_paths_key_coverage_flags_missing_paths_keys(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const PATHS={tower_basic:'tower_0.png'};"
        "function drawSprite(k){ctx.drawImage(load(PATHS[k]),0,0);}"
        "function draw(){drawSprite('tower_flame');drawSprite('enemy_basic');}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert rep["ok"] is False
    assert any("PATHS_KEY_COVERAGE" in e for e in rep["errors"])


def test_paths_key_coverage_passes_when_keys_present(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const PATHS={tower_flame:'tower_0.png',enemy_basic:'tower_1.png'};"
        "function drawSprite(k){ctx.drawImage(load(PATHS[k]),0,0);}"
        "function draw(){drawSprite('tower_flame');drawSprite('enemy_basic');}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert not any("PATHS_KEY_COVERAGE" in e for e in rep.get("errors") or [])
