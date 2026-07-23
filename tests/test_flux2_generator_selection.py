"""Selection logic for FLUX2 klein (mflux) sprite generation.

These tests are intentionally light: they validate that the harness prefers
FLUX2 klein on macOS when both the model directory and the mflux binary are
available, without depending on any real model weights or GPU stack.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_macos_prefers_flux2_when_model_and_binary_present(tmp_path: Path, monkeypatch) -> None:
    # Pretend we're on macOS regardless of the CI runner OS.
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)

    # Create a fake FLUX2 model directory in a custom diffusion tree.
    models_root = tmp_path / "Diffusion_Models"
    (models_root / "FLUX2-klein-9B-mlx-8bit").mkdir(parents=True, exist_ok=True)

    # Provide a fake mflux binary.
    mflux_bin = tmp_path / "bin" / "mflux-generate-flux2"
    _write_executable(mflux_bin)

    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_FLUX2", str(mflux_bin))
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)

    gen = assets.try_load_image_generator()
    assert gen is not None
    assert type(gen).__name__ == "Flux2KleinMfluxGenerator"
    assert os.path.basename(str(getattr(gen, "model_path", ""))) == "FLUX2-klein-9B-mlx-8bit"


def test_macos_never_falls_back_to_zimage(monkeypatch) -> None:
    """Policy: Apple Silicon Macs use FLUX2 only — never Z-Image-Turbo."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(assets, "_resolve_flux2_path", lambda: None)
    monkeypatch.setattr(assets, "_resolve_mflux_generate_flux2", lambda: None)
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    assert assets._construct_generator() is None


def test_macos_studio_happy_path_still_flux2(tmp_path: Path, monkeypatch) -> None:
    """Mac Studio regression: FLUX2 weights + mflux => same generator as before."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    models_root = tmp_path / "Diffusion_Models"
    (models_root / "FLUX2-klein-9B-mlx-8bit").mkdir(parents=True)
    mflux_bin = tmp_path / "bin" / "mflux-generate-flux2"
    _write_executable(mflux_bin)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_FLUX2", str(mflux_bin))
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    # Even if a complete Z-Image tree exists, macOS must ignore it.
    zimg = tmp_path / "Z-Image-Turbo"
    zimg.mkdir()
    monkeypatch.setattr(assets, "_resolve_zimage_path", lambda: str(zimg))
    gen = assets._construct_generator()
    assert type(gen).__name__ == "Flux2KleinMfluxGenerator"
    assert "FLUX2-klein-9B-mlx-8bit" in str(gen.model_path)


def test_linux_still_uses_zimage_when_no_flux2(monkeypatch) -> None:
    """Linux beast path unchanged — Z-Image remains the default there."""
    monkeypatch.setattr(assets.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(assets, "_resolve_flux2_path", lambda: None)
    monkeypatch.setattr(assets, "_resolve_mflux_generate_flux2", lambda: None)
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    import importlib.util as iu

    real = iu.find_spec

    def _spec(name, *a, **k):
        if name in ("torch", "diffusers"):
            return object()
        return real(name, *a, **k)

    monkeypatch.setattr(iu, "find_spec", _spec)
    gen = assets._construct_generator()
    assert type(gen).__name__ == "ZImageTurboGenerator"

