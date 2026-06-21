"""Auto-unload diffusers after sprite/sound batches when MLX coder is huge."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from backend import MLXBackend  # noqa: E402


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


def _fake_huge_model_dir(tmp_path: Path, size_bytes: int) -> tuple[Path, MagicMock]:
    model_dir = tmp_path / "mlx_model"
    model_dir.mkdir()
    fake = MagicMock()
    fake.is_file.return_value = True
    fake.stat.return_value.st_size = size_bytes
    return model_dir, fake


def test_mlx_coder_memory_pressure_trips_for_huge_model(
    tmp_path: Path, monkeypatch,
) -> None:
    model_dir, fake = _fake_huge_model_dir(tmp_path, int(250 * 1e9))

    def _rglob(self: Path, pattern: str):
        if self == model_dir:
            return [fake]
        return []

    monkeypatch.setattr(Path, "rglob", _rglob)
    agent = _make_agent(tmp_path)
    agent._backend = MagicMock()
    agent._backend.info.name = "mlx"
    agent._backend.info.model = str(model_dir)
    monkeypatch.setattr(MLXBackend, "_loaded_path", None)

    def _sysconf(name: str) -> int:
        if name == "SC_PHYS_PAGES":
            return int(512 * 1e9 / 4096)
        if name == "SC_PAGE_SIZE":
            return 4096
        return 0

    monkeypatch.setattr(os, "sysconf", _sysconf)
    tripped, llm_gb, _ = agent._mlx_coder_memory_pressure()
    assert tripped is True
    assert llm_gb is not None and llm_gb > 100


def test_mlx_coder_memory_pressure_skips_small_model(
    tmp_path: Path, monkeypatch,
) -> None:
    model_dir, fake = _fake_huge_model_dir(tmp_path, int(30 * 1e9))

    def _rglob(self: Path, pattern: str):
        if self == model_dir:
            return [fake]
        return []

    monkeypatch.setattr(Path, "rglob", _rglob)
    agent = _make_agent(tmp_path)
    agent._backend = MagicMock()
    agent._backend.info.name = "mlx"
    agent._backend.info.model = str(model_dir)
    monkeypatch.setattr(MLXBackend, "_loaded_path", None)

    def _sysconf(name: str) -> int:
        if name == "SC_PHYS_PAGES":
            return int(512 * 1e9 / 4096)
        if name == "SC_PAGE_SIZE":
            return 4096
        return 0

    monkeypatch.setattr(os, "sysconf", _sysconf)
    tripped, _, _ = agent._mlx_coder_memory_pressure()
    assert tripped is False


def test_release_diffusers_vram_clears_session_generators(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    asset_gen = MagicMock()
    asset_gen.cleanup = MagicMock()
    sound_gen = MagicMock()
    sound_gen.cleanup = MagicMock()
    agent._asset_generator = asset_gen
    agent._sound_generator = sound_gen
    freed = agent._release_diffusers_vram()
    asset_gen.cleanup.assert_called_once()
    sound_gen.cleanup.assert_called_once()
    assert agent._asset_generator is None
    assert agent._sound_generator is None
    assert "Stable-Audio" in freed
