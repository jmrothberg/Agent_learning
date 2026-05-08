"""Unit tests for backend.detect_backend() — pure-function probes via mocks.

The detection functions reach out to Ollama (urllib) and to MLX
(urllib + `ps` subprocess). We stub all three so the test runs offline
and deterministically. No real daemon required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes for the three probes detect_backend uses.
# ---------------------------------------------------------------------------


def _fake_http(routes: dict[str, dict | None]):
    """Returns a function suitable for monkey-patching backend._http_get_json.

    `routes` maps a URL substring → JSON payload (or None to simulate a
    network error). The first matching substring wins.
    """
    def fake(url: str, timeout: float = 5.0):
        for needle, payload in routes.items():
            if needle in url:
                return payload
        return None
    return fake


def _fake_proc(model_arg: str | None):
    """Returns a function suitable for monkey-patching backend._mlx_process_model_arg."""
    def fake() -> str | None:
        return model_arg
    return fake


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Strip env vars that would otherwise short-circuit detection."""
    for key in ("OLLAMA_MODEL", "CHAT_OLLAMA_MODEL", "MLX_MODEL", "LLM_BACKEND",
                "OLLAMA_HOST", "MLX_HOST"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Detection scenarios.
# ---------------------------------------------------------------------------


def test_only_ollama_loaded_picks_ollama(monkeypatch):
    """Ollama up with a chat model loaded, MLX down → Ollama."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {"parameter_size": "27B"},
                "context_length": 32768,
            }]},
        }),
    )
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    info = backend.detect_backend("auto")
    assert info.name == "ollama"
    assert info.model == "qwen3.6:27b"
    assert "loaded in ollama" in info.source


def test_only_mlx_loaded_picks_mlx(monkeypatch):
    """MLX up with a model in /v1/models, Ollama down → MLX."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [{"id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"}]},
        }),
    )
    monkeypatch.setattr(
        backend, "_mlx_process_model_arg",
        _fake_proc("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"),
    )

    info = backend.detect_backend("auto")
    assert info.name == "mlx"
    assert info.model == "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
    assert "mlx_lm.server" in info.source


def test_both_loaded_mlx_wins(monkeypatch):
    """Both daemons reachable with loaded models → MLX wins (per user decision)."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {}, "context_length": 32768,
            }]},
            "/v1/models": {"data": [{"id": "mlx-community/Llama-3-8B-Instruct-4bit"}]},
        }),
    )
    monkeypatch.setattr(
        backend, "_mlx_process_model_arg",
        _fake_proc("mlx-community/Llama-3-8B-Instruct-4bit"),
    )

    info = backend.detect_backend("auto")
    assert info.name == "mlx"


def test_llm_backend_env_forces_ollama(monkeypatch):
    """LLM_BACKEND=ollama overrides MLX preference even when both loaded."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {}, "context_length": 32768,
            }]},
            "/v1/models": {"data": [{"id": "mlx-community/Foo"}]},
        }),
    )
    monkeypatch.setattr(
        backend, "_mlx_process_model_arg",
        _fake_proc("mlx-community/Foo"),
    )
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    info = backend.detect_backend()
    assert info.name == "ollama"


def test_mlx_process_arg_resolves_model(monkeypatch):
    """When the MLX server has --model X, that's the active model id."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [
                {"id": "mlx-community/A"},
                {"id": "mlx-community/B"},
                {"id": "mlx-community/C"},
            ]},
        }),
    )
    monkeypatch.setattr(
        backend, "_mlx_process_model_arg",
        _fake_proc("mlx-community/B"),
    )

    info = backend.detect_backend("mlx")
    assert info.model == "mlx-community/B"
    assert "mlx_lm.server" in info.source


def test_mlx_falls_back_to_first_v1_models(monkeypatch):
    """No --model arg → use /v1/models[0] with a clearly-labeled source."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [
                {"id": "mlx-community/A"},
                {"id": "mlx-community/B"},
            ]},
        }),
    )
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    info = backend.detect_backend("mlx")
    assert info.model == "mlx-community/A"
    assert "first of 2" in info.source


def test_mlx_model_env_overrides_process(monkeypatch):
    """MLX_MODEL env wins over the running process's --model arg."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({"/v1/models": {"data": [{"id": "mlx-community/Whatever"}]}}),
    )
    monkeypatch.setattr(
        backend, "_mlx_process_model_arg",
        _fake_proc("mlx-community/FromProcess"),
    )
    monkeypatch.setenv("MLX_MODEL", "mlx-community/FromEnv")

    info = backend.detect_backend("mlx")
    assert info.model == "mlx-community/FromEnv"


def test_ollama_model_env_overrides_loaded(monkeypatch):
    """OLLAMA_MODEL env wins over the freshest /api/ps entry."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {}, "context_length": 32768,
            }]},
        }),
    )
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:latest")

    info = backend.detect_backend("auto")
    assert info.name == "ollama"
    assert info.model == "gpt-oss:latest"
    assert "OLLAMA_MODEL" in info.source


def test_neither_reachable_raises(monkeypatch):
    """No daemon up at all → RuntimeError with a useful hint."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    with pytest.raises(RuntimeError, match="No LLM backend reachable"):
        backend.detect_backend("auto")


def test_force_mlx_when_unreachable_raises(monkeypatch):
    """LLM_BACKEND=mlx but mlx_lm.server is down → clear error."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    with pytest.raises(RuntimeError, match="mlx_lm.server is not reachable"):
        backend.detect_backend("mlx")


def test_picks_freshest_ollama_by_expires_at(monkeypatch):
    """When /api/ps lists multiple, sort by expires_at desc and pick freshest."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [
                {"name": "old:7b", "expires_at": "2020-01-01T00:00:00Z",
                 "details": {}, "context_length": 8192},
                {"name": "fresh:27b", "expires_at": "2030-01-01T00:00:00Z",
                 "details": {}, "context_length": 32768},
            ]},
        }),
    )
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    info = backend.detect_backend("ollama")
    assert info.model == "fresh:27b"


def test_filters_non_chat_tags(monkeypatch):
    """Z-Image / embedding models in /api/ps must not be picked."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [
                # Most-recently-used is a diffuser; should be skipped.
                {"name": "x/z-image-turbo:latest", "expires_at": "2030-01-01T00:00:00Z",
                 "details": {}, "context_length": 0},
                {"name": "qwen3.6:27b", "expires_at": "2025-01-01T00:00:00Z",
                 "details": {}, "context_length": 32768},
            ]},
        }),
    )
    monkeypatch.setattr(backend, "_mlx_process_model_arg", _fake_proc(None))

    info = backend.detect_backend("ollama")
    assert info.model == "qwen3.6:27b"


def test_mlx_process_regex_parses_command_line():
    """Exercise _MLX_PROC_MODEL_RE against representative ps output lines."""
    samples = [
        ("/usr/bin/python -m mlx_lm.server --model mlx-community/Foo --port 8080",
         "mlx-community/Foo"),
        ("python mlx_lm.server --model=mlx-community/Bar-4bit --host 0.0.0.0",
         "mlx-community/Bar-4bit"),
        ("python -m mlx_lm.server --port 8080",  # no --model arg
         None),
    ]
    for cmd, expected in samples:
        m = backend._MLX_PROC_MODEL_RE.search(cmd)
        got = m.group(1) if m else None
        assert got == expected, f"{cmd!r} → {got!r}, expected {expected!r}"
