"""Tests for legacy maintainer-doc detection on disk.

game_rules.md injection was removed — standing constraints belong in
memory/playbook.jsonl. AGENTS.md / CLAUDE.md must never be injected into
the game model; `_deprecated_project_config_on_disk` traces them for migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402


def test_deprecated_files_detected_for_trace(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("RULE A: foo")
    (tmp_path / "CLAUDE.md").write_text("RULE B: bar")
    found = agent._deprecated_project_config_on_disk(tmp_path)
    assert len(found) == 2


def test_empty_deprecated_files_ignored(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n")
    found = agent._deprecated_project_config_on_disk(tmp_path)
    assert found == []


def test_no_deprecated_files_returns_empty(tmp_path: Path) -> None:
    assert agent._deprecated_project_config_on_disk(tmp_path) == []
