"""VRAM relief when hot-swapping MLX coder models mid-session."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from backend import MLXBackend  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    agent._backend = MagicMock()
    agent._backend.info.name = "mlx"
    return agent


def test_relieve_on_upsizing_swap_unloads_mlx_and_diffusers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = str(tmp_path / "small")
    large = str(tmp_path / "large")
    agent = _make_agent(tmp_path)
    asset_gen = MagicMock()
    asset_gen.cleanup = MagicMock()
    agent._asset_generator = asset_gen
    MLXBackend._loaded_path = small
    MLXBackend._loaded_model = object()

    def _disk_gb(path: str | None) -> float | None:
        if path == small:
            return 30.0
        if path == large:
            return 250.0
        return None

    monkeypatch.setattr(GameAgent, "_mlx_model_disk_gb", staticmethod(_disk_gb))
    released: list[bool] = []

    def _fake_release(*, wait_for_metal: bool = False) -> str:
        released.append(wait_for_metal)
        MLXBackend._loaded_model = None
        MLXBackend._loaded_path = None
        return small

    monkeypatch.setattr(
        MLXBackend,
        "release_weights",
        classmethod(lambda cls, **kw: _fake_release(**kw)),
    )

    out = agent._relieve_vram_for_mlx_model_swap(large)
    assert out["skipped"] is False
    assert out["upsizing"] is True
    assert any("MLX-LLM" in x for x in out["freed"])
    asset_gen.cleanup.assert_called_once()
    assert released == [True]
    assert MLXBackend._loaded_path is None


def test_relieve_skips_when_same_model_path(tmp_path: Path) -> None:
    path = str(tmp_path / "m")
    agent = _make_agent(tmp_path)
    MLXBackend._loaded_path = path
    MLXBackend._loaded_model = object()

    out = agent._relieve_vram_for_mlx_model_swap(path)
    assert out["skipped"] is True
    assert out["freed"] == []
    assert MLXBackend._loaded_path == path


def test_relieve_on_any_path_swap_even_small_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    agent = _make_agent(tmp_path)
    MLXBackend._loaded_path = a
    MLXBackend._loaded_model = object()

    monkeypatch.setattr(
        GameAgent,
        "_mlx_model_disk_gb",
        staticmethod(lambda path: 30.0 if path else None),
    )
    called: list[bool] = []

    def _fake_release(*, wait_for_metal: bool = False) -> str:
        called.append(wait_for_metal)
        return a

    monkeypatch.setattr(
        MLXBackend,
        "release_weights",
        classmethod(lambda cls, **kw: _fake_release(**kw)),
    )

    out = agent._relieve_vram_for_mlx_model_swap(b)
    assert out["skipped"] is False
    assert called == [True]
