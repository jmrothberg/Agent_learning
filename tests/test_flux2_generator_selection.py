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
    # Klein remains the backup when the Qwen-Image-2.1 tree is absent.
    monkeypatch.setattr(assets, "_resolve_qwen21_path", lambda: None)

    gen = assets.try_load_image_generator()
    assert gen is not None
    assert type(gen).__name__ == "Flux2KleinMfluxGenerator"
    assert os.path.basename(str(getattr(gen, "model_path", ""))) == "FLUX2-klein-9B-mlx-8bit"


def test_macos_never_falls_back_to_zimage(monkeypatch) -> None:
    """Policy: Apple Silicon Macs use FLUX2 only — never Z-Image-Turbo."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(assets, "_resolve_flux2_path", lambda: None)
    monkeypatch.setattr(assets, "_resolve_mflux_generate_flux2", lambda: None)
    monkeypatch.setattr(assets, "_resolve_qwen21_path", lambda: None)
    monkeypatch.setattr(assets, "_resolve_mflux_generate_qwen21", lambda: None)
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
    monkeypatch.setattr(assets, "_resolve_qwen21_path", lambda: None)
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


def _fake_qwen21_tree(root: Path) -> Path:
    """mflux shard layout (transformer/0.safetensors), not the mlx-community file."""
    model = root / "Qwen-Image-2.1-MLX-4bit-mflux"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer" / "0.safetensors").write_bytes(b"")
    return model


def test_macos_prefers_klein_over_qwen_for_speed(tmp_path: Path, monkeypatch) -> None:
    """Both installed → FLUX2-klein. Qwen-Image-2.1 is ~40 steps, not the fast path."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    models_root = tmp_path / "Diffusers"
    _fake_qwen21_tree(models_root)
    klein = models_root / "FLUX2-klein-9B-mlx-8bit"
    klein.mkdir()
    qbin = tmp_path / "bin" / "mflux-generate-qwen-2.1"
    _write_executable(qbin)
    kbin = tmp_path / "bin" / "mflux-generate-flux2"
    _write_executable(kbin)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_QWEN21", str(qbin))
    monkeypatch.setenv("MFLUX_GENERATE_FLUX2", str(kbin))
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    gen = assets._construct_generator()
    assert type(gen).__name__ == "Flux2KleinMfluxGenerator"
    assert gen.model_path == str(klein)


def test_macos_qwen21_when_klein_missing(tmp_path: Path, monkeypatch) -> None:
    """Qwen is the macOS fallback only when the 4-step klein tree is absent."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    models_root = tmp_path / "Diffusers"
    qwen = _fake_qwen21_tree(models_root)
    qbin = tmp_path / "bin" / "mflux-generate-qwen-2.1"
    _write_executable(qbin)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_QWEN21", str(qbin))
    monkeypatch.delenv("MFLUX_GENERATE_FLUX2", raising=False)
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.setattr(assets._os.path, "expanduser", lambda _p: str(empty_home))
    monkeypatch.setattr(assets, "_MODEL_SEARCH_DIRS", [])
    gen = assets._construct_generator()
    assert type(gen).__name__ == "QwenImage21MfluxGenerator"
    assert gen.model_path == str(qwen)


def test_macos_qwen21_missing_falls_back_to_klein(tmp_path: Path, monkeypatch) -> None:
    """No Qwen tree → FLUX2-klein, including when only the non-mflux quant is present."""
    monkeypatch.setattr(assets.sys, "platform", "darwin", raising=False)
    models_root = tmp_path / "Diffusers"
    # mlx-community layout: single model.safetensors, no transformer/0.safetensors.
    community = models_root / "Qwen-Image-2.1-MLX-4bit" / "transformer"
    community.mkdir(parents=True)
    (community / "model.safetensors").write_bytes(b"")
    (models_root / "FLUX2-klein-9B-mlx-8bit").mkdir()
    kbin = tmp_path / "bin" / "mflux-generate-flux2"
    _write_executable(kbin)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_FLUX2", str(kbin))
    monkeypatch.delenv("MFLUX_GENERATE_QWEN21", raising=False)
    monkeypatch.delenv("DIFFUSER_TXT2IMG_BACKBONE", raising=False)
    # Stay inside the temp tree — do not see this machine's ~/Diffusers.
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.setattr(assets._os.path, "expanduser", lambda _p: str(empty_home))
    monkeypatch.setattr(assets, "_MODEL_SEARCH_DIRS", [])
    gen = assets._construct_generator()
    assert type(gen).__name__ == "Flux2KleinMfluxGenerator"


def test_linux_ignores_qwen21_and_keeps_zimage(tmp_path: Path, monkeypatch) -> None:
    """Other systems stay on Z-Image even if a Qwen MLX tree is on disk."""
    monkeypatch.setattr(assets.sys, "platform", "linux", raising=False)
    models_root = tmp_path / "Diffusers"
    _fake_qwen21_tree(models_root)
    qbin = tmp_path / "bin" / "mflux-generate-qwen-2.1"
    _write_executable(qbin)
    monkeypatch.setenv("DIFFUSION_MODELS_DIR", str(models_root))
    monkeypatch.setenv("MFLUX_GENERATE_QWEN21", str(qbin))
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

