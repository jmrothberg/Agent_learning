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

