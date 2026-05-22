"""Tests for status-panel-side helpers added in the panel revamp:

  * GameAgent._estimate_ctx_fill — message-char sum used to render
    the live `Ctx: X / Y (Z%)` row.
  * backend._read_mlx_context_length — reads the model's native
    context window from its config.json so the panel can render the
    `Y` side of that row for MLX-resolved sessions.

Pure functions; no GPU / Chromium / network involvement.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import backend as backend_mod  # noqa: E402
from backend import _read_mlx_context_length  # noqa: E402
from chat import CodingBoxApp  # noqa: E402
import gpu_status  # noqa: E402


@pytest.fixture(autouse=True)
def clear_ollama_slot_env(monkeypatch):
    for key in ("OLLAMA_HOST2", "OLLAMA_HOST3"):
        monkeypatch.delenv(key, raising=False)


def _agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ---------------------------------------------------------------------------
# _estimate_ctx_fill
# ---------------------------------------------------------------------------


def test_estimate_ctx_fill_empty(tmp_path: Path) -> None:
    a = _agent(tmp_path)
    assert a._estimate_ctx_fill() == 0


def test_estimate_ctx_fill_sums_message_content(tmp_path: Path) -> None:
    a = _agent(tmp_path)
    a._messages = [
        {"role": "system", "content": "x" * 100},
        {"role": "user", "content": "y" * 250},
        {"role": "assistant", "content": "z" * 500},
    ]
    assert a._estimate_ctx_fill() == 850


def test_estimate_ctx_fill_skips_non_string_content(tmp_path: Path) -> None:
    a = _agent(tmp_path)
    a._messages = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": None},
        {"role": "user", "content": ["multimodal", "list"]},
        {"role": "user", "content": "world"},
    ]
    # Only the two string contents count: 5 + 5 = 10
    assert a._estimate_ctx_fill() == 10


def test_tui_defaults_to_wait_mode_profile() -> None:
    app = CodingBoxApp()
    assert app._run_profile == "local_manual"


def test_chat_process_gpu_vram() -> None:
    snap = gpu_status.GpuSnapshot(
        processes=[gpu_status.GpuProcess(0, 42, "python", 20416)],
    )
    assert gpu_status.chat_process_gpu_vram(snap, 42) == [(0, 20416)]
    assert gpu_status.chat_process_gpu_vram(snap, 99) == []


def test_activity_header_includes_role() -> None:
    app = CodingBoxApp()
    app._session_backend3 = object()
    app._model3_is_streaming = True
    app._model3_stream_tokens = 50
    app._session_role3 = "architect"
    app._session_model3 = "qwen3.6:27b"
    app._activity_label = "architect note"
    app._activity_role = "architect"
    app._model3_stream_started_at = time.monotonic() - 5.0
    app._model3_last_token_at = time.monotonic() - 1.0
    line = app._render_activity_line()
    assert "Activity (architect):" in line
    assert "50 tok" in line
    assert "Activity (coder):" in line


def test_role_slot_for_stream_maps_sidecars() -> None:
    app = CodingBoxApp()
    app._session_backend2 = object()
    app._session_backend3 = object()
    app._session_role2 = "critic"
    app._session_role3 = "architect"
    assert app._role_slot_for_stream("coder") is None
    assert app._role_slot_for_stream("critic") == 2
    assert app._role_slot_for_stream("architect") == 3


# ---------------------------------------------------------------------------
# _read_mlx_context_length
# ---------------------------------------------------------------------------


def _write_config(dir_: Path, payload: dict) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = dir_ / "config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return cfg


def test_read_mlx_context_length_max_position_embeddings(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {"max_position_embeddings": 32768})
    assert _read_mlx_context_length(str(tmp_path / "m")) == 32768


def test_read_mlx_context_length_max_seq_len(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {"max_seq_len": 8192})
    assert _read_mlx_context_length(str(tmp_path / "m")) == 8192


def test_read_mlx_context_length_model_max_length(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {"model_max_length": 4096})
    assert _read_mlx_context_length(str(tmp_path / "m")) == 4096


def test_read_mlx_context_length_prefers_max_position_embeddings(tmp_path: Path) -> None:
    # When multiple keys are present, the canonical Llama/Qwen key wins.
    _write_config(tmp_path / "m", {
        "max_position_embeddings": 32768,
        "max_seq_len": 8192,
        "model_max_length": 4096,
    })
    assert _read_mlx_context_length(str(tmp_path / "m")) == 32768


def test_read_mlx_context_length_missing_config(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert _read_mlx_context_length(str(tmp_path / "empty")) is None


def test_read_mlx_context_length_invalid_path() -> None:
    assert _read_mlx_context_length("") is None
    assert _read_mlx_context_length("/no/such/path/xyz") is None


def test_read_mlx_context_length_malformed_config(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "config.json").write_text("not json", encoding="utf-8")
    assert _read_mlx_context_length(str(tmp_path / "m")) is None


def test_cuda_device_label_visible_devices() -> None:
    import os
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        assert gpu_status.cuda_device_label("cuda", 0) == "GPU 2 (cuda:0)"
        assert gpu_status.cuda_device_label("cuda", 1) == "GPU 3 (cuda:1)"
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def test_cuda_device_label_plain_cuda() -> None:
    import os
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert gpu_status.cuda_device_label("cuda", 1) == "GPU 1"
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def test_pick_least_loaded_cuda_index() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(0, "A", memory_used_mib=40000, memory_total_mib=49140),
            gpu_status.GpuInfo(1, "B", memory_used_mib=28000, memory_total_mib=49140),
            gpu_status.GpuInfo(2, "C", memory_used_mib=2000, memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_used_mib=27000, memory_total_mib=49140),
        ],
    )
    assert gpu_status.pick_least_loaded_cuda_index(snap) == 2


def test_pick_least_loaded_skips_llm_gpus() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(0, "A", memory_used_mib=40000, memory_total_mib=49140),
            gpu_status.GpuInfo(1, "B", memory_used_mib=28000, memory_total_mib=49140),
            gpu_status.GpuInfo(2, "C", memory_used_mib=2000, memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_used_mib=27000, memory_total_mib=49140),
        ],
        processes=[
            gpu_status.GpuProcess(0, 99, "ollama", 28000),
            gpu_status.GpuProcess(1, 99, "ollama", 27000),
        ],
    )
    assert gpu_status.pick_least_loaded_cuda_index(snap) == 2


def _four_gpu_workstation_snap(
    *,
    used: tuple[int, int, int, int] = (1000, 2000, 2000, 43000),
    processes: list | None = None,
) -> gpu_status.GpuSnapshot:
    name = "NVIDIA RTX 6000 Ada Generation"
    return gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(i, name, memory_used_mib=used[i], memory_total_mib=49140)
            for i in range(4)
        ],
        processes=processes or [],
    )


def test_pick_diffuser_prefers_gpu0_on_workstation() -> None:
    """Empty GPU 1 must not steal diffusers from reserved GPU 0."""
    snap = _four_gpu_workstation_snap(used=(1000, 500, 2000, 43000))
    assert gpu_status.pick_diffuser_cuda_index(snap) == 0


def test_pick_diffuser_skips_ollama_slot_gpus() -> None:
    snap = _four_gpu_workstation_snap(
        used=(1000, 20000, 10000, 43000),
        processes=[gpu_status.GpuProcess(3, 9, "ollama", 43000)],
    )
    assert gpu_status.pick_diffuser_cuda_index(snap) == 0


def test_pick_diffuser_reuse_cuda_index() -> None:
    snap = _four_gpu_workstation_snap()
    assert gpu_status.pick_diffuser_cuda_index(snap, reuse_cuda_index=0) == 0


def test_format_gpu_indices_label_never_bare_question() -> None:
    assert "GPU ?" not in gpu_status.format_gpu_indices_label([], None, pending=True)
    assert "GPU ?" not in gpu_status.format_gpu_indices_label([], None, vram_gib=20.5)
    snap = gpu_status.GpuSnapshot(
        gpus=[gpu_status.GpuInfo(1, "B", memory_used_mib=1000, memory_total_mib=49140)],
    )
    label = gpu_status.format_gpu_indices_label([1], snap)
    assert label.startswith("GPU 1")
    assert "GPU ?" not in label


def test_infer_ollama_gpu_indices_from_processes() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[gpu_status.GpuInfo(1, "B", memory_used_mib=28000, memory_total_mib=49140)],
        processes=[gpu_status.GpuProcess(1, 42, "/usr/bin/ollama", 28000)],
    )
    assert gpu_status.infer_ollama_gpu_indices(snap) == [1]


def test_ollama_tensor_split_warning() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[],
        processes=[
            gpu_status.GpuProcess(1, 7, "ollama", 28000),
            gpu_status.GpuProcess(3, 7, "ollama", 27000),
        ],
    )
    warn = gpu_status.ollama_tensor_split_warning(snap)
    assert warn is not None
    assert "GPU 1" in warn and "3" in warn


def test_format_four_gpu_summary() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(0, "A", memory_used_mib=20480, memory_total_mib=49140),
            gpu_status.GpuInfo(1, "B", memory_used_mib=0, memory_total_mib=49140),
        ],
    )
    s = gpu_status.format_four_gpu_summary(snap)
    assert "0 [" in s and "1 [" in s


def test_format_model_gpu_placement_multi_gpu() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(1, "B", memory_used_mib=28000, memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_used_mib=27000, memory_total_mib=49140),
        ],
    )
    line = gpu_status.format_model_gpu_placement([1, 3], snap, vram_gib=53.1)
    assert line == "GPU 1 + 3 (~53.1 GB)"
    assert "GPU ?" not in line


def test_prefer_single_gpu_workstation() -> None:
    big = gpu_status.GpuSnapshot(
        gpus=[gpu_status.GpuInfo(0, "A", memory_total_mib=49140)],
    )
    assert gpu_status.prefer_single_gpu_workstation(big) is True
    small = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(0, "A", memory_total_mib=8192),
            gpu_status.GpuInfo(1, "B", memory_total_mib=8192),
        ],
    )
    assert gpu_status.prefer_single_gpu_workstation(small) is False


def test_ollama_chat_load_options_big_box() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[gpu_status.GpuInfo(0, "A", memory_total_mib=49140)],
    )
    assert gpu_status.ollama_chat_load_options(snap) == {"num_gpu": 999}
    small = gpu_status.GpuSnapshot(
        gpus=[gpu_status.GpuInfo(0, "A", memory_total_mib=8192)],
    )
    assert gpu_status.ollama_chat_load_options(small) == {}


def test_ollama_split_tip_short() -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(0, "A", memory_total_mib=49140),
            gpu_status.GpuInfo(1, "B", memory_total_mib=49140),
        ],
        processes=[
            gpu_status.GpuProcess(1, 7, "ollama", 28000),
            gpu_status.GpuProcess(3, 7, "ollama", 27000),
        ],
    )
    tip = gpu_status.ollama_split_tip_short(snap)
    assert tip is not None
    assert "/unload" in tip
    assert "pid" not in tip.lower()


def test_ollama_split_tip_suppressed_for_multi_daemon(monkeypatch) -> None:
    snap = gpu_status.GpuSnapshot(
        processes=[
            gpu_status.GpuProcess(1, 7, "ollama", 28000),
            gpu_status.GpuProcess(3, 7, "ollama", 27000),
        ],
    )
    monkeypatch.setenv("OLLAMA_HOST2", "http://127.0.0.1:11435")
    assert gpu_status.ollama_split_tip_short(snap) is None


def test_gpu_indices_for_ollama_loaded_model_single_card() -> None:
    snap = gpu_status.GpuSnapshot(
        processes=[
            gpu_status.GpuProcess(2, 10, "ollama", 28000),
            gpu_status.GpuProcess(1, 11, "ollama", 4000),
        ],
    )
    vram_bytes = 28 * 1024 * 1024 * 1024
    assert gpu_status.gpu_indices_for_ollama_loaded_model(
        snap, vram_bytes=vram_bytes,
    ) == [2]


def test_auto_fix_ollama_tensor_split_unloads(monkeypatch) -> None:
    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(1, "B", memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_total_mib=49140),
        ],
        processes=[
            gpu_status.GpuProcess(1, 7, "ollama", 28000),
            gpu_status.GpuProcess(3, 7, "ollama", 27000),
        ],
    )
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: snap)
    monkeypatch.setattr(
        backend_mod,
        "unload_all_ollama_models",
        lambda endpoint=None: [("qwen3.6:27b-q8_0", True, "ok")],
    )
    ok, msg = backend_mod.auto_fix_ollama_tensor_split()
    assert ok is True
    assert "released split" in msg


def test_render_gpu_block_three_model_rows(monkeypatch) -> None:
    app = CodingBoxApp()
    tag = "qwen3.6:27b-q8_0"
    bi = backend_mod.BackendInfo(
        name="ollama",
        model=tag,
        source="test",
        endpoint="http://127.0.0.1:11434",
    )
    agent = MagicMock()
    agent.model = tag
    agent._backend = MagicMock(info=bi)
    agent._backend2 = MagicMock(info=bi)
    agent._backend3 = MagicMock(info=bi)
    agent._model2_role = "architect"
    agent._model3_role = "critic"
    agent._asset_generator = None
    agent._sound_generator = None
    app.agent = agent
    app._session_model2 = tag
    app._session_model3 = tag

    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(1, "B", memory_used_mib=28000, memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_used_mib=27000, memory_total_mib=49140),
        ],
        processes=[
            gpu_status.GpuProcess(1, 7, "ollama", 28000),
            gpu_status.GpuProcess(3, 7, "ollama", 27000),
        ],
    )

    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: snap)
    monkeypatch.setattr(
        gpu_status, "ollama_loaded_models",
        lambda: [{"name": tag, "vram_gib": 53.1}],
    )

    block = app._render_gpu_placement_block()
    assert "GPU ?" not in block
    assert "Ollama VRAM" not in block
    assert "shared load" not in block
    assert "Model 1" in block and "Model 2" in block and "Model 3" in block
    assert tag in block
    assert "same VRAM" in block
    assert "LLM" in block
    assert "Diffusers" in block
    assert "Z-Image-Turbo" in block
    assert "not loaded" in block
    assert "still split" in block or "/unload" in block


def test_render_gpu_block_separate_ollama_endpoints(monkeypatch) -> None:
    app = CodingBoxApp()
    tag = "qwen3.6:27b-q8_0"
    bi1 = backend_mod.BackendInfo(
        name="ollama", model=tag, source="test",
        endpoint="http://127.0.0.1:11434",
    )
    bi2 = backend_mod.BackendInfo(
        name="ollama", model=tag, source="test",
        endpoint="http://127.0.0.1:11435",
    )
    bi3 = backend_mod.BackendInfo(
        name="ollama", model=tag, source="test",
        endpoint="http://127.0.0.1:11436",
    )
    agent = MagicMock()
    agent.model = tag
    agent._backend = MagicMock(info=bi1)
    agent._backend2 = MagicMock(info=bi2)
    agent._backend3 = MagicMock(info=bi3)
    agent._model2_role = "critic"
    agent._model3_role = "architect"
    agent._asset_generator = None
    agent._sound_generator = None
    app.agent = agent
    app._session_model2 = tag
    app._session_model3 = tag

    snap = gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(1, "B", memory_used_mib=44000, memory_total_mib=49140),
            gpu_status.GpuInfo(2, "C", memory_used_mib=33000, memory_total_mib=49140),
            gpu_status.GpuInfo(3, "D", memory_used_mib=430, memory_total_mib=49140),
        ],
    )
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: snap)
    monkeypatch.setattr(
        gpu_status,
        "ollama_endpoint_gpu_index",
        lambda endpoint: {
            "http://127.0.0.1:11434": 1,
            "http://127.0.0.1:11435": 2,
            "http://127.0.0.1:11436": 3,
        }.get(endpoint),
    )
    monkeypatch.setattr(
        gpu_status,
        "ollama_loaded_models",
        lambda: [
            {"name": tag, "endpoint": "http://127.0.0.1:11434", "vram_gib": 41.8},
            {"name": tag, "endpoint": "http://127.0.0.1:11435", "vram_gib": 32.8},
        ],
    )

    block = app._render_gpu_placement_block()
    assert "Model 1 (coder)" in block and "GPU 1" in block
    assert "Model 2 (critic)" in block and "GPU 2" in block
    assert "Model 3 (architect)" in block and "GPU 3" in block
    assert "pinned, not loaded" in block
    assert "same VRAM" not in block


def test_render_activity_lines_all_roles() -> None:
    app = CodingBoxApp()
    app.agent = MagicMock()
    app.agent.model = "qwen3.6:27b"
    app.agent._backend2 = MagicMock()
    app.agent._backend3 = MagicMock()
    app.agent._model2_activity = "proposing playtest"
    app.agent._model3_activity = "idle"
    app._session_model2 = "qwen3.6:27b"
    app._session_model3 = "qwen3.6:27b"
    app._session_role2 = "critic"
    app._session_role3 = "architect"
    app._is_streaming = True
    app._stream_tokens = 120
    app._stream_started_at = 1000.0
    app._last_token_at = 1005.0
    app._activity_label = "iter 2 reply"
    app._activity_role = "coder"
    app._model2_is_streaming = True
    app._model2_stream_tokens = 40
    app._model2_stream_started_at = 1001.0
    app._model2_last_token_at = 1004.0
    monkeypatch_now = 1010.0
    orig = time.monotonic
    try:
        time.monotonic = lambda: monkeypatch_now
        block = app._render_activity_line()
    finally:
        time.monotonic = orig
    assert "Activity (coder)" in block
    assert "Activity (critic)" in block
    assert "Activity (architect)" in block
    assert "120 tok" in block
    assert "40 tok" in block
    assert "tok/s" in block


def test_diffuser_placement_stable_audio_gpu_index() -> None:
    class StableAudioGenerator:
        _device = "cuda"
        _cuda_device_index = 2

    assert "GPU 2" in gpu_status.diffuser_placement(StableAudioGenerator())


def test_diffuser_kind_labels() -> None:
    class ZImageTurboGenerator:
        pass
    class Img2ImgGenerator:
        pass
    class StableAudioGenerator:
        pass
    assert gpu_status.diffuser_kind(ZImageTurboGenerator()) == "Z-Image-Turbo"
    assert gpu_status.diffuser_kind(Img2ImgGenerator()) == "SD-Turbo img2img"
    assert gpu_status.diffuser_kind(StableAudioGenerator()) == "Stable-Audio"


def test_read_mlx_context_length_ignores_non_positive(tmp_path: Path) -> None:
    # Don't surface 0 or -1 as a context window — these would mislead
    # the user. None means "unknown" and hides the row.
    _write_config(tmp_path / "m", {"max_position_embeddings": 0})
    assert _read_mlx_context_length(str(tmp_path / "m")) is None
    _write_config(tmp_path / "m2", {"max_seq_len": -1})
    assert _read_mlx_context_length(str(tmp_path / "m2")) is None
