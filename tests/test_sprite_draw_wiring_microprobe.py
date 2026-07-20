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


def test_paths_key_coverage_passes_array_of_pairs(tmp_path: Path) -> None:
    """Array PATHS/ASSET_PATHS is a valid loader class — must not false-fail."""
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const PATHS=["
        "['tower_flame','./game_20260703_assets/tower_0.png'],"
        "['enemy_basic','./game_20260703_assets/tower_1.png'],"
        "['bomb','./game_20260703_assets/tower_2.png'],"
        "['soft_block','./game_20260703_assets/tower_3.png']"
        "];"
        "function drawSprite(k){ctx.drawImage(load(PATHS[k]),0,0);}"
        "function draw(){"
        "drawSprite('tower_flame');drawSprite('enemy_basic');"
        "drawSprite('bomb');drawSprite('soft_block');"
        "}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert not any("PATHS_KEY_COVERAGE" in e for e in rep.get("errors") or [])


def test_paths_key_coverage_passes_asset_paths_alias(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"t{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const ASSET_PATHS=["
        "['ship','./game_20260703_assets/t0.png'],"
        "['asteroid','./game_20260703_assets/t1.png']"
        "];"
        "function draw(){sprite('ship');sprite('asteroid');}"
        "function sprite(k){ctx.drawImage(ASSETS[k],0,0);}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert not any("PATHS_KEY_COVERAGE" in e for e in rep.get("errors") or [])


def test_paths_key_coverage_flags_missing_in_array_form(tmp_path: Path) -> None:
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"tower_{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "const PATHS=[['tower_basic','tower_0.png']];"
        "function drawSprite(k){ctx.drawImage(load(PATHS[k]),0,0);}"
        "function draw(){drawSprite('tower_flame');drawSprite('enemy_basic');}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert rep["ok"] is False
    assert any("PATHS_KEY_COVERAGE" in e for e in rep["errors"])


def test_paths_key_coverage_passes_loadassets_entries(tmp_path: Path) -> None:
    """loadAssets() entries=[['name', '…_assets/…png']] is a valid loader class."""
    assets = tmp_path / "game_20260703_assets"
    assets.mkdir()
    for i in range(6):
        (assets / f"t{i}.png").write_bytes(b"fake")
    out = tmp_path / "game_20260703.html"
    out.write_text("x")
    html = _minimal_html(
        "async function loadAssets(){"
        "const entries=["
        "['bomb','./game_20260703_assets/t0.png'],"
        "['soft_block','./game_20260703_assets/t1.png'],"
        "['hard_block','./game_20260703_assets/t2.png'],"
        "['exit','./game_20260703_assets/t3.png']"
        "];"
        "for(const [name,src] of entries){ASSETS[name]=new Image();ASSETS[name].src=src;}"
        "}"
        "function drawSprite(k){ctx.drawImage(ASSETS[k],0,0);}"
        "function draw(){"
        "drawSprite('bomb');drawSprite('soft_block');"
        "drawSprite('hard_block');drawSprite('exit');"
        "}"
        "requestAnimationFrame(draw);"
    )
    rep = run_micro_probes(html, out_path=out)
    assert not any("PATHS_KEY_COVERAGE" in e for e in rep.get("errors") or [])
