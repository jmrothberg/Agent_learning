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
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from backend import _read_mlx_context_length  # noqa: E402
from chat import CodingBoxApp  # noqa: E402


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


def test_read_mlx_context_length_ignores_non_positive(tmp_path: Path) -> None:
    # Don't surface 0 or -1 as a context window — these would mislead
    # the user. None means "unknown" and hides the row.
    _write_config(tmp_path / "m", {"max_position_embeddings": 0})
    assert _read_mlx_context_length(str(tmp_path / "m")) is None
    _write_config(tmp_path / "m2", {"max_seq_len": -1})
    assert _read_mlx_context_length(str(tmp_path / "m2")) is None
