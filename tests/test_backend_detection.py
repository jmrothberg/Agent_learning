"""Unit tests for backend.detect_backend() — pure-function probes via mocks.

The Ollama path still reaches out via urllib (/api/ps, /api/tags). The
MLX path now resolves entirely locally: MLX_MODEL env, then a filesystem
scan for downloaded models (no HTTP, no mlx_lm.server). We stub the
relevant probes so the test runs offline and deterministically.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch  # noqa: F401 - retained for downstream test additions

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: E402


def test_ollama_backend_forwards_keep_alive(monkeypatch):
    seen: dict = {}

    async def fake_stream_chat_with_retry(client, model, messages, **kwargs):
        seen.update(kwargs)
        return backend.StreamResult(
            text="ok", tokens=1, duration_s=0.01, stalled=False
        )

    monkeypatch.setattr(
        backend, "stream_chat_with_retry", fake_stream_chat_with_retry
    )
    be = backend.OllamaBackend(backend.BackendInfo(
        name="ollama",
        model="qwen3.6:27b-q8_0",
        source="test",
        endpoint="http://127.0.0.1:11434",
    ))

    result = asyncio.run(be.stream_chat(
        [{"role": "user", "content": "hi"}],
        options={"temperature": 0.1},
        keep_alive=-1,
    ))

    assert result.text == "ok"
    assert seen["keep_alive"] == -1
    assert seen["options"] == {"temperature": 0.1}


# ---------------------------------------------------------------------------
# Fakes for the probes detect_backend uses.
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


def _fake_local_mlx(paths: list[str]):
    """Returns a function suitable for monkey-patching backend.list_local_mlx_models."""
    def fake() -> list[str]:
        return list(paths)
    return fake


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Strip env vars that would otherwise short-circuit detection."""
    for key in ("OLLAMA_MODEL", "CHAT_OLLAMA_MODEL", "MLX_MODEL", "LLM_BACKEND",
                "OLLAMA_HOST", "MLX_SERVER_URL", "MLX_HOST"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def isolate_mlx_state():
    """Ensure MLXBackend's class-level model cache doesn't leak across tests
    (a previous test loading a fake path would otherwise show up in
    list_mlx_inventory's "active" return)."""
    prev_path = backend.MLXBackend._loaded_path
    prev_model = backend.MLXBackend._loaded_model
    prev_tok = backend.MLXBackend._loaded_tokenizer
    backend.MLXBackend._loaded_path = None
    backend.MLXBackend._loaded_model = None
    backend.MLXBackend._loaded_tokenizer = None
    yield
    backend.MLXBackend._loaded_path = prev_path
    backend.MLXBackend._loaded_model = prev_model
    backend.MLXBackend._loaded_tokenizer = prev_tok


# ---------------------------------------------------------------------------
# Detection scenarios.
# ---------------------------------------------------------------------------


def test_only_ollama_loaded_picks_ollama(monkeypatch):
    """Ollama up with a chat model loaded, no local MLX models → Ollama."""
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
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))

    info = backend.detect_backend("auto")
    assert info.name == "ollama"
    assert info.model == "qwen3.6:27b"
    assert "loaded in ollama" in info.source


def test_only_mlx_local_picks_mlx(monkeypatch):
    """A single local MLX model present, Ollama down → MLX."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/home/u/MLX_Models/Qwen3.6-27B-mxfp8"]),
    )

    info = backend.detect_backend("auto")
    assert info.name == "mlx"
    assert info.model == "/home/u/MLX_Models/Qwen3.6-27B-mxfp8"
    assert "only local MLX" in info.source
    assert info.endpoint == "in-process"


def test_both_available_mlx_wins(monkeypatch):
    """Ollama loaded AND a local MLX model present → MLX wins."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {}, "context_length": 32768,
            }]},
        }),
    )
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/m/Llama-3-8B-Instruct-4bit"]),
    )

    info = backend.detect_backend("auto")
    assert info.name == "mlx"


def test_llm_backend_env_forces_ollama(monkeypatch):
    """LLM_BACKEND=ollama overrides MLX preference even when MLX is available."""
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/api/ps": {"models": [{
                "name": "qwen3.6:27b", "expires_at": "2030-01-01T00:00:00Z",
                "details": {}, "context_length": 32768,
            }]},
        }),
    )
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/m/Foo"]),
    )
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    info = backend.detect_backend()
    assert info.name == "ollama"


def test_mlx_env_overrides_local_scan(monkeypatch):
    """MLX_MODEL env wins over any local-disk scan results."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/disk/SomeOther"]),
    )
    monkeypatch.setenv("MLX_MODEL", "/explicit/path/Qwen-via-env")

    info = backend.detect_backend("mlx")
    assert info.model == "/explicit/path/Qwen-via-env"
    assert "MLX_MODEL" in info.source


def test_mlx_multiple_local_picks_first_with_warning(monkeypatch):
    """Multiple local MLX models → pick first with a 'set MLX_MODEL' hint."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/m/A", "/m/B", "/m/C"]),
    )

    info = backend.detect_backend("mlx")
    assert info.model == "/m/A"
    assert "set MLX_MODEL" in info.source


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
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:latest")

    info = backend.detect_backend("auto")
    assert info.name == "ollama"
    assert info.model == "gpt-oss:latest"
    assert "OLLAMA_MODEL" in info.source


def test_neither_reachable_raises(monkeypatch):
    """No daemon up and no local MLX → RuntimeError with a useful hint."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))

    with pytest.raises(RuntimeError, match="No LLM backend reachable"):
        backend.detect_backend("auto")


def test_force_mlx_when_no_model_raises(monkeypatch):
    """LLM_BACKEND=mlx but no MLX_MODEL and no local scan → clear error."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))

    with pytest.raises(RuntimeError, match="no MLX model could be resolved"):
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
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))

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
    monkeypatch.setattr(backend, "list_local_mlx_models", _fake_local_mlx([]))

    info = backend.detect_backend("ollama")
    assert info.model == "qwen3.6:27b"


def test_mlx_local_filters_non_chat(monkeypatch):
    """Local MLX scan must skip embedding/diffuser-shaped paths."""
    monkeypatch.setattr(backend, "_http_get_json", _fake_http({}))
    # First entry contains a non-chat fragment ("z-image"); should be skipped.
    monkeypatch.setattr(
        backend, "list_local_mlx_models",
        _fake_local_mlx(["/m/z-image-turbo", "/m/Qwen3.6-27B-mxfp8"]),
    )

    info = backend.detect_backend("mlx")
    assert info.model == "/m/Qwen3.6-27B-mxfp8"


def test_mlx_server_url_forces_server_backend(monkeypatch):
    monkeypatch.setenv("MLX_SERVER_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [{"id": "Qwen3.6-27B-mxfp8"}]},
        }),
    )

    info = backend.detect_backend("mlx")
    assert info.name == "mlx"
    assert info.endpoint == "http://127.0.0.1:8080"
    assert info.model == "Qwen3.6-27B-mxfp8"
    assert "mlx_lm.server" in info.source or "/v1/models" in info.source


def test_llm_backend_mlx_server(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mlx-server")
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [{"id": "local-model"}]},
        }),
    )

    info = backend.detect_backend()
    assert info.endpoint == "http://127.0.0.1:8080"
    assert info.model == "local-model"


def test_make_backend_routes_http_mlx_to_server():
    info = backend.BackendInfo(
        name="mlx",
        model="foo",
        source="test",
        endpoint="http://127.0.0.1:8080",
    )
    be = backend.make_backend(info)
    assert isinstance(be, backend.MLXServerBackend)


def test_make_backend_routes_in_process_mlx():
    info = backend.BackendInfo(
        name="mlx",
        model="/m/foo",
        source="test",
        endpoint="in-process",
    )
    be = backend.make_backend(info)
    assert isinstance(be, backend.MLXBackend)


def test_mlx_server_url_forces_server_backend(monkeypatch):
    monkeypatch.setenv("MLX_SERVER_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [{"id": "Qwen3.6-27B-mxfp8"}]},
        }),
    )

    info = backend.detect_backend("mlx")
    assert info.name == "mlx"
    assert info.endpoint == "http://127.0.0.1:8080"
    assert info.model == "Qwen3.6-27B-mxfp8"
    assert "mlx_lm.server" in info.source or "/v1/models" in info.source


def test_llm_backend_mlx_server(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mlx-server")
    monkeypatch.setattr(
        backend, "_http_get_json",
        _fake_http({
            "/v1/models": {"data": [{"id": "local-model"}]},
        }),
    )

    info = backend.detect_backend()
    assert info.endpoint == "http://127.0.0.1:8080"
    assert info.model == "local-model"


# --- oMLX / DeepSeek-V4 helpers ----------------------------------------------


def test_requires_omlx_server_by_name():
    assert backend.requires_omlx_server(
        "/Users/x/MLX_Models/DeepSeek-V4-Flash-0731-MXFP4-MLX"
    )
    assert backend.requires_omlx_server("DeepSeek-V4-Flash-0731-MXFP4-MLX")
    assert backend.requires_omlx_server("vontra/deepseek_v4_flash")
    assert not backend.requires_omlx_server("GLM-5.2-MLX-4bit")
    assert not backend.requires_omlx_server("Qwen3.6-27B-mxfp8")
    assert not backend.requires_omlx_server("")


def test_requires_omlx_server_from_config_json(tmp_path: Path):
    d = tmp_path / "some_custom_dir"
    d.mkdir()
    (d / "config.json").write_text(
        '{"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}\n',
        encoding="utf-8",
    )
    assert backend.requires_omlx_server(str(d))
    other = tmp_path / "glm"
    other.mkdir()
    (other / "config.json").write_text(
        '{"model_type": "glm_moe_dsa"}\n', encoding="utf-8"
    )
    assert not backend.requires_omlx_server(str(other))


def test_omlx_default_endpoint_env(monkeypatch):
    monkeypatch.delenv("OMLX_SERVER_URL", raising=False)
    assert backend.omlx_default_endpoint() == "http://127.0.0.1:8000"
    monkeypatch.setenv("OMLX_SERVER_URL", "127.0.0.1:9000")
    assert backend.omlx_default_endpoint() == "http://127.0.0.1:9000"


def test_mlx_endpoint_for_model_routes_flash_to_omlx(monkeypatch):
    monkeypatch.delenv("MLX_SERVER_URL", raising=False)
    monkeypatch.delenv("MLX_HOST", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    flash = "/m/DeepSeek-V4-Flash-0731-MXFP4-MLX"
    assert backend.mlx_endpoint_for_model(flash) == "http://127.0.0.1:8000"
    assert backend.mlx_endpoint_for_model("/m/GLM-5.2-MLX-4bit") == "in-process"


def test_ensure_omlx_server_already_up(monkeypatch):
    monkeypatch.setattr(backend, "omlx_reachable", lambda *a, **k: True)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        return False, "should not spawn"

    monkeypatch.setattr(backend, "_spawn_omlx_serve", boom)
    assert backend.ensure_omlx_server() == "http://127.0.0.1:8000"
    assert calls == []


def test_ensure_omlx_server_spawns_then_ready(monkeypatch):
    hits = {"n": 0}

    def reachable(*_a, **_k):
        hits["n"] += 1
        return hits["n"] >= 2  # first probe fail, after spawn succeed

    monkeypatch.setattr(backend, "omlx_reachable", reachable)
    monkeypatch.setattr(
        backend, "_spawn_omlx_serve", lambda ep: (True, "spawned test")
    )
    monkeypatch.setattr(backend.time, "sleep", lambda *_: None)
    assert backend.ensure_omlx_server(timeout_s=5.0) == "http://127.0.0.1:8000"


def test_ensure_omlx_server_fails_clearly(monkeypatch):
    monkeypatch.setattr(backend, "omlx_reachable", lambda *a, **k: False)
    monkeypatch.setattr(
        backend, "_spawn_omlx_serve", lambda ep: (False, "no binary")
    )
    with pytest.raises(RuntimeError, match="oMLX"):
        backend.ensure_omlx_server(timeout_s=1.0)
