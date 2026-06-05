"""Defaults and /ctx parsing for Ollama num_ctx."""

from __future__ import annotations

import pytest

from agent import (
    DEFAULT_NUM_CTX,
    MAX_NUM_CTX,
    MIN_NUM_CTX,
    default_num_ctx,
    parse_num_ctx_arg,
)


def test_default_num_ctx_is_100k() -> None:
    # Raised from 32K → 100K (2026-05-29): 32K was also the compaction pressure
    # denominator, so the lossy compaction fired every turn and shredded
    # history. 100K is the speed/headroom sweet spot — coder prompts stay
    # ~10-45K, so pressure stays under the 0.70 trigger (full history kept)
    # while KV-cache/prefill cost stays far below a 250K reservation.
    assert DEFAULT_NUM_CTX == 100_000
    assert DEFAULT_NUM_CTX <= MAX_NUM_CTX


def test_parse_num_ctx_presets() -> None:
    assert parse_num_ctx_arg("100k") == 100_000
    assert parse_num_ctx_arg("131k") == 131_072
    assert parse_num_ctx_arg("262k") == MAX_NUM_CTX
    assert parse_num_ctx_arg("full") == MAX_NUM_CTX
    assert parse_num_ctx_arg("100000") == 100_000


def test_parse_num_ctx_k_suffix() -> None:
    assert parse_num_ctx_arg("150k") == 150_000


def test_parse_num_ctx_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        parse_num_ctx_arg("4096")
    with pytest.raises(ValueError):
        parse_num_ctx_arg("999999")


def test_default_num_ctx_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_BOX_NUM_CTX", "131k")
    assert default_num_ctx() == 131_072
    monkeypatch.delenv("CODING_BOX_NUM_CTX", raising=False)
    assert default_num_ctx() == DEFAULT_NUM_CTX


def test_bounds_constants() -> None:
    assert MIN_NUM_CTX == 8192
    assert MAX_NUM_CTX == 262_144
