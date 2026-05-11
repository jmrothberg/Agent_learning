"""Tests for GameAgent._classify_stall.

The stall classifier inspects a stream-failure exception message and
returns a structured `mlx_stall` event payload when it detects the
"model produced no tokens" signature. Substring + regex only — no
backend identity check, so future stalls from any backend with similar
text are caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_stall_classifier_matches_mlx_message():
    msg = "Model produced no tokens before stalling at 60.0s"
    info = GameAgent._classify_stall(msg)
    assert info is not None
    assert info["kind"] == "no_tokens_stall"
    assert info["stall_seconds"] == 60.0


def test_stall_classifier_returns_none_for_unrelated():
    assert GameAgent._classify_stall("ConnectionRefused: no route to host") is None
    assert GameAgent._classify_stall("") is None


def test_stall_classifier_handles_message_without_seconds():
    info = GameAgent._classify_stall("Model produced no tokens; aborting.")
    assert info is not None
    assert info["stall_seconds"] is None
