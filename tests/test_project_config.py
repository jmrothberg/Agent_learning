"""Tests for AGENTS.md / CLAUDE.md project-config injection (roadmap #1).

The agent reads AGENTS.md and CLAUDE.md from the working tree at session
start and folds them into the system prompt as a <project-context>
block. These tests exercise the helper directly — no model is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402


def test_no_project_config_returns_empty(tmp_path: Path) -> None:
    text, sources = agent._read_project_config(tmp_path)
    assert text == ""
    assert sources == []


def test_reads_agents_md_alone(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "# Project rules\nAlways use Phaser. Never use React.\n"
    )
    text, sources = agent._read_project_config(tmp_path)
    assert "## AGENTS.md" in text
    assert "Always use Phaser" in text
    assert any(s.endswith("AGENTS.md") for s in sources)


def test_reads_claude_md_alone(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Use vanilla JS only.")
    text, sources = agent._read_project_config(tmp_path)
    assert "## CLAUDE.md" in text
    assert "vanilla JS" in text


def test_concats_both_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("RULE A: foo")
    (tmp_path / "CLAUDE.md").write_text("RULE B: bar")
    text, sources = agent._read_project_config(tmp_path)
    assert "RULE A: foo" in text
    assert "RULE B: bar" in text
    assert "## AGENTS.md" in text
    assert "## CLAUDE.md" in text
    # Per the read order, AGENTS.md first.
    assert text.index("RULE A") < text.index("RULE B")
    assert len(sources) == 2


def test_truncates_oversize_file(tmp_path: Path) -> None:
    """Files past _PROJECT_CONFIG_MAX_CHARS get a clear truncation marker."""
    big = "x" * (agent._PROJECT_CONFIG_MAX_CHARS + 1000)
    (tmp_path / "AGENTS.md").write_text(big)
    text, _ = agent._read_project_config(tmp_path)
    assert len(text) <= agent._PROJECT_CONFIG_MAX_CHARS + 200  # marker overhead
    assert "truncated" in text.lower()


def test_empty_file_skipped(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n  \n")
    text, sources = agent._read_project_config(tmp_path)
    assert text == ""
    assert sources == []


def test_directory_with_only_other_md_files_no_op(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("not an agent file")
    text, sources = agent._read_project_config(tmp_path)
    assert text == ""
    assert sources == []
