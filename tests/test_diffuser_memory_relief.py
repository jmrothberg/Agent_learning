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


def _wire_ollama_agent(tmp_path: Path) -> GameAgent:
    agent = _make_agent(tmp_path)
    agent._backend = MagicMock()
    agent._backend.info.name = "ollama"
    agent._backend.info.model = "qwen3.6-27b-ud-q8:latest"
    agent._backend.info.endpoint = "http://127.0.0.1:11434"
    return agent


def test_should_release_diffusers_after_media_ollama_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux/Ollama+CUDA: always drop in-process diffusers for the coder."""
    agent = _wire_ollama_agent(tmp_path)
    monkeypatch.setattr(
        GameAgent, "_cuda_available_for_relief", lambda self: True,
    )
    assert agent._should_release_diffusers_after_media() is True


def test_should_release_diffusers_ollama_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ENABLE_MEMORY_RELIEF", "0")
    agent = _wire_ollama_agent(tmp_path)
    monkeypatch.setattr(
        GameAgent, "_cuda_available_for_relief", lambda self: True,
    )
    assert agent._should_release_diffusers_after_media() is False


def test_should_release_diffusers_ollama_no_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _wire_ollama_agent(tmp_path)
    monkeypatch.setattr(
        GameAgent, "_cuda_available_for_relief", lambda self: False,
    )
    assert agent._should_release_diffusers_after_media() is False


def test_free_memory_before_video_unloads_ollama_when_vram_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _wire_ollama_agent(tmp_path)
    unload_calls: list[object] = []

    def _fake_unload(endpoint=None):
        unload_calls.append(endpoint)
        return [("qwen3.6-27b-ud-q8:latest", True, "unloaded")]

    import assets as assets_mod
    import backend as backend_mod
    import gpu_status as gs

    monkeypatch.setattr(assets_mod, "release_preloaded_diffusers", lambda: [])
    monkeypatch.setattr(backend_mod, "unload_all_ollama_models", _fake_unload)
    monkeypatch.setattr(gs, "diffuser_has_dedicated_gpu", lambda snap=None: False)
    monkeypatch.setattr(
        GameAgent, "_video_target_cuda_free_gb", lambda self: 4.0,
    )
    monkeypatch.setenv("AGENT_VIDEO_MIN_FREE_VRAM_GB", "12")
    out = agent._free_memory_before_video()
    assert unload_calls == [None]
    assert out.get("ollama_unloaded") == ["qwen3.6-27b-ud-q8:latest"]
    assert any(x.startswith("Ollama:") for x in out.get("freed", []))


def test_free_memory_before_video_skips_ollama_on_dedicated_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _wire_ollama_agent(tmp_path)
    unload_calls: list[object] = []

    def _fake_unload(endpoint=None):
        unload_calls.append(endpoint)
        return [("qwen", True, "unloaded")]

    import assets as assets_mod
    import backend as backend_mod
    import gpu_status as gs

    monkeypatch.setattr(assets_mod, "release_preloaded_diffusers", lambda: [])
    monkeypatch.setattr(backend_mod, "unload_all_ollama_models", _fake_unload)
    monkeypatch.setattr(gs, "diffuser_has_dedicated_gpu", lambda snap=None: True)
    monkeypatch.setattr(
        GameAgent, "_video_target_cuda_free_gb", lambda self: 1.0,
    )
    out = agent._free_memory_before_video()
    assert unload_calls == []
    assert out.get("ollama_unload_skipped") == "dedicated_diffuser_gpu"


def test_ensure_vram_for_diffuser_media_unloads_before_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Space Invaders 20260723: sprites OOM'd because Ollama still held VRAM."""
    agent = _wire_ollama_agent(tmp_path)
    unload_calls: list[object] = []

    def _fake_unload(endpoint=None):
        unload_calls.append(endpoint)
        return [("qwen3.6-27b-ud-q8:latest", True, "unloaded")]

    import backend as backend_mod
    import gpu_status as gs

    monkeypatch.setattr(backend_mod, "unload_all_ollama_models", _fake_unload)
    monkeypatch.setattr(gs, "diffuser_has_dedicated_gpu", lambda snap=None: False)
    monkeypatch.setattr(
        GameAgent, "_video_target_cuda_free_gb", lambda self: 2.0,
    )
    monkeypatch.setenv("AGENT_VIDEO_MIN_FREE_VRAM_GB", "12")
    out = agent._ensure_vram_for_diffuser_media(reason="assets")
    assert unload_calls == [None]
    assert out.get("ollama_unloaded") == ["qwen3.6-27b-ud-q8:latest"]
    assert out.get("media_reason") == "assets"


def test_ollama_cpu_offload_fraction() -> None:
    import gpu_status as gs
    from unittest.mock import patch

    models = [
        {"size_bytes": 40_000_000_000, "size_vram_bytes": 15_000_000_000},
    ]
    with patch.object(gs, "ollama_ps_at_endpoint", return_value=models):
        frac = gs.ollama_cpu_offload_fraction("http://127.0.0.1:11434")
    assert frac is not None
    assert abs(frac - 0.625) < 0.01

    with patch.object(
        gs,
        "ollama_ps_at_endpoint",
        return_value=[
            {"size_bytes": 40_000_000_000, "size_vram_bytes": 40_000_000_000},
        ],
    ):
        assert gs.ollama_cpu_offload_fraction("http://127.0.0.1:11434") == 0.0

    with patch.object(gs, "ollama_ps_at_endpoint", return_value=[]):
        assert gs.ollama_cpu_offload_fraction("http://127.0.0.1:11434") is None
