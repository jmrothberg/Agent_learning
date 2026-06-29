"""Optional git_sha on session_outcome trace line (run_07 batch correlation)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_current_git_sha_returns_short_hash_in_repo() -> None:
    expected = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
    ).strip()
    assert GameAgent._current_git_sha() == expected
