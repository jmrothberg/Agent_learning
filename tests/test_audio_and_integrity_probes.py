"""HTMLAudioElement allowlist + unused-asset integrity probe."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import (
    _check_api_allowlist,
    _check_unused_assets,
    run_micro_probes,
    screenshot_delta,
)


def test_audio_play_is_allowed():
    """`audio.play()` is the most common audio call; it must NOT flag."""
    js = "const audio = new Audio('x.ogg'); audio.play();"
    hits = _check_api_allowlist(js)
    # No hits about `audio.play` — it's a real method.
    assert not any(method == "play" for _, method, _ in hits)


def test_audio_hallucinated_method_is_flagged():
    """`audio.startWithFadeIn()` is an invented method; flag it."""
    js = "const sfx = new Audio('boom.ogg'); sfx.startWithFadeIn();"
    hits = _check_api_allowlist(js)
    assert any(
        recv == "sfx" and method.lower() == "startwithfadein"
        for recv, method, _ in hits
    )


def test_audio_clonenode_is_allowed():
    """The overlap-safe audio pattern uses cloneNode(); must not flag."""
    js = "audio.cloneNode().play();"
    hits = _check_api_allowlist(js)
    assert not any(method == "clonenode" for _, method, _ in hits)


def test_unused_sprite_warns(tmp_path: Path):
    """A generated PNG that the HTML never references → warning."""
    base = tmp_path
    (base / "game_assets").mkdir()
    (base / "game_assets" / "player.png").write_bytes(b"fake")
    (base / "game_assets" / "enemy.png").write_bytes(b"fake")
    html = (
        "<!doctype html><html><body>"
        '<img src="./game_assets/player.png">'
        "<script>console.log('ok')</script></body></html>"
    )
    out = base / "game.html"
    warns = _check_unused_assets(html, out)
    assert any("enemy.png" in w for w in warns)
    assert not any("player.png" in w for w in warns)


def test_unused_sound_warns(tmp_path: Path):
    """Same for OGG files under <slug>_sounds/."""
    base = tmp_path
    (base / "game_sounds").mkdir()
    (base / "game_sounds" / "jump.ogg").write_bytes(b"fake")
    (base / "game_sounds" / "win.ogg").write_bytes(b"fake")
    html = (
        "<!doctype html><html><body><script>"
        "new Audio('./game_sounds/jump.ogg').play();"
        "</script></body></html>"
    )
    out = base / "game.html"
    warns = _check_unused_assets(html, out)
    assert any("win.ogg" in w for w in warns)
    assert not any("jump.ogg" in w for w in warns)


def test_micro_probes_surface_unused_assets_warning(tmp_path: Path):
    """Integration: run_micro_probes should bubble the unused-asset
    warnings into the report.warnings list and stats."""
    base = tmp_path
    (base / "g_assets").mkdir()
    (base / "g_assets" / "orphan.png").write_bytes(b"fake")
    # Run_micro_probes returns early on <200-byte HTML, so the body
    # needs enough text to pass that gate. Bracket counts must balance.
    html = (
        "<!doctype html><html><head><title>game</title></head><body>"
        "<canvas id='c' width='800' height='600'></canvas>"
        "<script>"
        "function start() { console.log('hi'); }"
        "function loop() { requestAnimationFrame(loop); }"
        "requestAnimationFrame(loop);"
        "document.addEventListener('keydown', function(e) { console.log(e.key); });"
        "</script></body></html>"
    )
    assert len(html) >= 200  # sanity: above the empty-file threshold
    out = base / "g.html"
    report = run_micro_probes(html, out_path=out)
    assert any("orphan.png" in w for w in report.get("warnings", []))
    assert report["stats"].get("unused_assets", 0) >= 1


def test_screenshot_delta_identical_zero():
    """Identical PNGs → delta near zero."""
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="PNG")
    png = buf.getvalue()
    delta = screenshot_delta(png, png)
    assert delta is not None
    assert delta < 0.001


def test_screenshot_delta_disjoint_large():
    """Black vs white → delta near 1.0."""
    from io import BytesIO

    from PIL import Image
    a_buf, b_buf = BytesIO(), BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(a_buf, format="PNG")
    Image.new("RGB", (64, 64), (255, 255, 255)).save(b_buf, format="PNG")
    delta = screenshot_delta(a_buf.getvalue(), b_buf.getvalue())
    assert delta is not None
    assert delta > 0.95


def test_screenshot_delta_none_input_returns_none():
    """Missing input → None (not zero, not crash)."""
    assert screenshot_delta(None, b"x") is None
    assert screenshot_delta(b"x", None) is None
    assert screenshot_delta(None, None) is None
