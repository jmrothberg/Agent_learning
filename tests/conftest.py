"""Shared pytest fixtures for GameAgent harness tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent import GameAgent


@pytest.fixture
def tmp_memory(tmp_path: Path) -> Path:
    mem = tmp_path / "memory"
    mem.mkdir()
    return mem


@pytest.fixture
def agent(tmp_path: Path, tmp_memory: Path) -> GameAgent:
    """Minimal GameAgent with stub browser and tmp memory root."""
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=4,
        memory_root=str(tmp_memory),
    )


@pytest.fixture
def agent_no_browser(tmp_path: Path, tmp_memory: Path) -> GameAgent:
    """Materialization-only agent (no Chromium)."""
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=None,
        max_iters=4,
        memory_root=str(tmp_memory),
    )
