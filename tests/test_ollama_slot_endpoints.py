"""Per-slot OLLAMA_HOST2/OLLAMA_HOST3 resolution for 3-model runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: E402
import gpu_status  # noqa: E402


@pytest.fixture(autouse=True)
def clear_ollama_hosts(monkeypatch):
    for key in ("OLLAMA_HOST", "OLLAMA_HOST2", "OLLAMA_HOST3"):
        monkeypatch.delenv(key, raising=False)
    yield
    for key in ("OLLAMA_HOST", "OLLAMA_HOST2", "OLLAMA_HOST3"):
        monkeypatch.delenv(key, raising=False)


def test_ollama_endpoint_slot_defaults() -> None:
    assert backend.ollama_endpoint_url(1) == "http://127.0.0.1:11434"
    assert backend.ollama_endpoint_url(2) == "http://127.0.0.1:11434"
    assert backend.ollama_endpoint_url(3) == "http://127.0.0.1:11434"


def test_ollama_endpoint_per_slot(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_HOST2", "http://127.0.0.1:11435")
    monkeypatch.setenv("OLLAMA_HOST3", "http://127.0.0.1:11436")
    assert backend.ollama_endpoint_url(1) == "http://127.0.0.1:11434"
    assert backend.ollama_endpoint_url(2) == "http://127.0.0.1:11435"
    assert backend.ollama_endpoint_url(3) == "http://127.0.0.1:11436"


def _large_gpu_snapshot(count: int) -> gpu_status.GpuSnapshot:
    return gpu_status.GpuSnapshot(
        gpus=[
            gpu_status.GpuInfo(
                index=i,
                name="NVIDIA RTX 6000 Ada Generation",
                memory_used_mib=0,
                memory_total_mib=49140,
            )
            for i in range(count)
        ],
    )


def test_autopin_respects_manual_slot_hosts(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST2", "http://127.0.0.1:11435")
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "manual"


def test_autopin_skips_macos(monkeypatch) -> None:
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "off"
    assert "macOS" in result.message


def test_autopin_skips_two_gpu_linux(monkeypatch) -> None:
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: _large_gpu_snapshot(2))
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "off"
    assert "2 visible GPUs" in result.message


def test_autopin_skips_when_nvidia_smi_missing(monkeypatch) -> None:
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: None)
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "off"
    assert "nvidia-smi" in result.message


def test_autopin_starts_three_slot_daemons(monkeypatch) -> None:
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: _large_gpu_snapshot(4))
    monkeypatch.setattr(backend, "_try_mlx", lambda: None)
    monkeypatch.setattr(backend, "_ollama_cli_candidates", lambda: ["/usr/local/bin/ollama"])
    monkeypatch.setattr(backend, "_resolve_ollama_models_dir", lambda: "/usr/share/ollama/.ollama/models")
    monkeypatch.setattr(backend, "_port_owner_pid", lambda port: None)
    calls: list[tuple[int, int, str]] = []

    def fake_start(*, exe: str, port: int, gpu: int, models_dir: str):
        calls.append((port, gpu, models_dir))
        return True, f"started {port}"

    monkeypatch.setattr(backend, "_start_ollama_slot_daemon", fake_start)
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "auto-pinned"
    assert calls == [
        (11434, 1, "/usr/share/ollama/.ollama/models"),
        (11435, 2, "/usr/share/ollama/.ollama/models"),
        (11436, 3, "/usr/share/ollama/.ollama/models"),
    ]
    assert backend.ollama_endpoint_url(2) == "http://127.0.0.1:11435"
    assert backend.ollama_endpoint_url(3) == "http://127.0.0.1:11436"


def test_autopin_unloads_split_before_same_user_restart(monkeypatch) -> None:
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(gpu_status, "snapshot_gpus", lambda **_: _large_gpu_snapshot(4))
    monkeypatch.setattr(backend, "_try_mlx", lambda: None)
    monkeypatch.setattr(backend, "_ollama_cli_candidates", lambda: ["/usr/local/bin/ollama"])
    monkeypatch.setattr(backend, "_resolve_ollama_models_dir", lambda: "/models")
    monkeypatch.setattr(
        backend,
        "_port_owner_pid",
        lambda port: 99 if port == 11434 else None,
    )
    monkeypatch.setattr(backend, "_slot_daemon_matches", lambda *a, **k: False)
    monkeypatch.setattr(
        backend,
        "_ollama_running_models",
        lambda base: [{"name": "qwen3.6:27b-q8_0"}] if base.endswith(":11434") else [],
    )
    unloaded: list[str] = []
    monkeypatch.setattr(
        backend,
        "unload_all_ollama_models",
        lambda base: unloaded.append(base) or [("qwen3.6:27b-q8_0", True, "ok")],
    )
    monkeypatch.setattr(
        backend,
        "_terminate_same_user_ollama_serve",
        lambda pid, port: (True, f"stopped {pid}"),
    )
    monkeypatch.setattr(
        backend,
        "_start_ollama_slot_daemon",
        lambda **kwargs: (True, f"started {kwargs['port']}"),
    )
    result = backend.ensure_ollama_slot_daemons_for_chat(enabled=True, prefer="ollama")
    assert result.mode == "auto-pinned"
    assert unloaded == ["http://127.0.0.1:11434"]


def test_list_ollama_inventory_merges_all_slots(monkeypatch) -> None:
    def fake_running(base: str) -> list[dict]:
        if base.endswith(":11435"):
            return [{"name": "qwen3.6:27b"}]
        return []

    monkeypatch.setattr(
        backend,
        "ollama_unload_probe_bases",
        lambda: ["http://127.0.0.1:11434", "http://127.0.0.1:11435"],
    )
    monkeypatch.setattr(backend, "_ollama_installed_models", lambda _b: ["qwen3.6:27b"])
    monkeypatch.setattr(backend, "_ollama_running_models", fake_running)
    _installed, loaded = backend.list_ollama_inventory()
    assert "qwen3.6:27b" in loaded


def test_unload_all_probes_three_slot_ports(monkeypatch) -> None:
    """/unload all must hit 11434/11435/11436, not only OLLAMA_HOST."""
    calls: list[str] = []

    def fake_ready(base: str) -> bool:
        return base.endswith((":11434", ":11435", ":11436"))

    def fake_running(base: str) -> list[dict]:
        if base.endswith(":11435"):
            return [{"name": "qwen3.6:27b"}]
        return []

    def fake_unload(name: str, endpoint: str | None = None) -> tuple[bool, str]:
        calls.append(endpoint or "")
        return True, "ok"

    monkeypatch.setattr(backend, "_endpoint_ready", fake_ready)
    monkeypatch.setattr(backend, "_ollama_running_models", fake_running)
    monkeypatch.setattr(backend, "unload_ollama_model", fake_unload)
    monkeypatch.setattr(
        backend,
        "ollama_unload_probe_bases",
        lambda: [
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11435",
            "http://127.0.0.1:11436",
        ],
    )
    results = backend.unload_all_ollama_models()
    assert len(results) == 1
    assert "http://127.0.0.1:11435" in calls
