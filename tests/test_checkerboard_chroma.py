"""Checkerboard fake-transparency chroma-key (FLUX2-klein path)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402


def _checkerboard_with_fg(size: int = 128) -> "object":
    from PIL import Image

    white = (255, 255, 255)
    gray = (204, 206, 205)
    img = Image.new("RGB", (size, size))
    px = img.load()
    cx0, cy0, cx1, cy1 = size // 4, size // 4, 3 * size // 4, 3 * size // 4
    for y in range(size):
        for x in range(size):
            if cx0 <= x < cx1 and cy0 <= y < cy1:
                px[x, y] = (200, 0, 0)
            else:
                px[x, y] = white if (x // 16 + y // 16) % 2 == 0 else gray
    return img


def test_checkerboard_border_becomes_transparent():
    img = _checkerboard_with_fg()
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert keyed.mode == "RGBA"
    assert stats["checkerboard_bg"] is not None
    assert stats["alpha_pixel_ratio"] > 0.5
    assert keyed.getpixel((0, 0))[3] == 0
    mid = img.width // 2
    assert keyed.getpixel((mid, mid))[3] == 255


def test_flux2_prompt_avoids_transparent_background():
    class _FakeFlux:
        pass

    gen = _FakeFlux()
    gen.__class__.__name__ = "Flux2KleinMfluxGenerator"

    out = assets._ensure_sprite_bg_prompt(
        "cel-shaded red ninja fighter, transparent background",
        gen,
    )
    assert "transparent background" not in out.lower()
    assert "solid pure white" in out.lower()
    assert "no checkerboard" in out.lower()


def test_zimage_prompt_keeps_transparent_suffix():
    out = assets._ensure_sprite_bg_prompt("pixel ship facing right", object())
    assert "transparent background" in out.lower()


def test_flux2_base_cmd_omits_negative_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    models_root = tmp_path / "Diffusion_Models"
    (models_root / "FLUX2-klein-9B-mlx-8bit").mkdir(parents=True)
    mflux_bin = tmp_path / "mflux-generate-flux2"
    mflux_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mflux_bin.chmod(0o755)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_FLUX2", str(mflux_bin))
    gen = assets.Flux2KleinMfluxGenerator()
    cmd = gen._base_cmd("test sprite", "/tmp/out.png")
    assert "--negative-prompt" not in cmd
