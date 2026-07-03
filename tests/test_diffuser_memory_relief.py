"""Auto-unload diffusers when free RAM is low; skip small MLX models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def _fake_model_dir(tmp_path: Path, size_bytes: int) -> tuple[Path, MagicMock]:
    model_dir = tmp_path / "mlx_model"
    model_dir.mkdir()
    fake = MagicMock()
    fake.is_file.return_value = True
    fake.stat.return_value.st_size = size_bytes
    return model_dir, fake


def _wire_mlx_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_dir: Path,
    fake_file: MagicMock,
) -> GameAgent:
    def _rglob(self: Path, pattern: str):
        if self == model_dir:
            return [fake_file]
        return []

    monkeypatch.setattr(Path, "rglob", _rglob)
    agent = _make_agent(tmp_path)
    agent._backend = MagicMock()
    agent._backend.info.name = "mlx"
    agent._backend.info.model = str(model_dir)
    monkeypatch.setattr(MLXBackend, "_loaded_path", None)
    return agent


def test_mlx_coder_memory_pressure_trips_below_64gb_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relief trips when free RAM is under the 64 GB default (e.g. DOOM + GLM stack)."""
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (48.0, 512.0)),
    )
    tripped, avail, _ = agent._mlx_coder_memory_pressure()
    assert tripped is True
    assert avail == 48.0


def test_mlx_coder_memory_pressure_skips_at_or_above_64gb_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (72.0, 512.0)),
    )
    tripped, _, _ = agent._mlx_coder_memory_pressure()
    assert tripped is False


def test_should_release_diffusers_after_media_on_96gb_phys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """96 GB Mac: plenty of free pages but still unload after sprite gen."""
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (72.0, 96.0)),
    )
    assert agent._should_release_diffusers_after_media() is True


def test_should_release_diffusers_on_96gb_with_small_27b_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3.6-27B (~30 GB on disk): phys gate must still trip on 96 GB hosts."""
    model_dir, fake = _fake_model_dir(tmp_path, int(30 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (72.0, 96.0)),
    )
    tripped, _, phys = agent._mlx_coder_memory_pressure()
    assert tripped is False
    assert phys is None
    assert agent._should_release_diffusers_after_media() is True


def test_should_release_diffusers_after_media_skips_512gb_phys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (200.0, 512.0)),
    )
    assert agent._should_release_diffusers_after_media() is False


def test_mlx_coder_memory_pressure_trips_when_ram_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (16.0, 512.0)),
    )
    tripped, avail, phys = agent._mlx_coder_memory_pressure()
    assert tripped is True
    assert avail == 16.0
    assert phys == 512.0


def test_mlx_coder_memory_pressure_skips_when_ram_plenty_even_huge_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLM-sized on-disk tree must not trip relief on a roomy box."""
    model_dir, fake = _fake_model_dir(tmp_path, int(420 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (200.0, 512.0)),
    )
    tripped, _, _ = agent._mlx_coder_memory_pressure()
    assert tripped is False


def test_mlx_coder_memory_pressure_skips_small_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, fake = _fake_model_dir(tmp_path, int(30 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (8.0, 64.0)),
    )
    tripped, _, _ = agent._mlx_coder_memory_pressure()
    assert tripped is False


def test_mlx_coder_memory_pressure_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ENABLE_MEMORY_RELIEF", "0")
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (8.0, 64.0)),
    )
    tripped, _, _ = agent._mlx_coder_memory_pressure()
    assert tripped is False


def test_free_memory_before_video_never_drops_mlx_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (16.0, 512.0)),
    )
    monkeypatch.setattr(MLXBackend, "_loaded_model", object(), raising=False)
    out = agent._free_memory_before_video()
    assert "MLX-LLM" not in out.get("freed", [])
    assert out.get("forced") is True


def test_free_memory_before_video_always_releases_diffusers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_09: jetsam during Wan video even when free RAM looked fine — relief
    must run unconditionally, not skip when pressure gate is false."""
    model_dir, fake = _fake_model_dir(tmp_path, int(250 * 1e9))
    agent = _wire_mlx_agent(tmp_path, monkeypatch, model_dir=model_dir, fake_file=fake)
    monkeypatch.setattr(
        GameAgent,
        "_available_system_memory_gb",
        staticmethod(lambda: (260.0, 549.0)),
    )
    freed_calls: list[str] = []

    def _fake_release():
        freed_calls.append("diffusers")
        return ["Z-Image"]

    import assets as assets_mod
    monkeypatch.setattr(assets_mod, "release_preloaded_diffusers", _fake_release)
    out = agent._free_memory_before_video()
    assert freed_calls == ["diffusers"]
    assert out.get("forced") is True
    assert "Z-Image" in out.get("freed", [])


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
