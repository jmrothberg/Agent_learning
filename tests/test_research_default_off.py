"""Tests pinning the REMOVAL of the /wiki Wikipedia research path.

History: research.fetch returned 0/10 hits on representative game goals
(asteroids, pacman, donkey kong, space invaders, missile command, street
fighter, doom, snake, 2d roguelike, tetris) for a cumulative ~38s of
network latency with no benefit, and the HTTPS handshake failed outright
on the framework Python build. The feature was first defaulted OFF, then
fully REMOVED 2026-06-24 in favor of the curated memory/ opening library.

These tests pin that the plan-time research path stays gone so a future
refactor doesn't silently re-introduce the network lookup (and the 110s
"agent looks frozen" gap from trace 20260519_111209).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import research  # noqa: E402
from agent import GameAgent  # noqa: E402


def test_research_fetch_is_inert_stub() -> None:
    """research.fetch is a deprecated no-op that returns an empty string,
    never a <reference> block or a network result."""
    assert research.fetch("missile command, good graphics") == ""
    assert research.fetch("anything at all") == ""


def test_agent_has_no_research_toggle() -> None:
    """The GameAgent research toggle/setter are gone — nothing can flip a
    plan-time Wikipedia lookup back on."""
    assert not hasattr(GameAgent, "set_research_enabled")


def test_chat_has_no_wiki_command() -> None:
    """The /wiki command handler was removed from the TUI."""
    import chat as chat_mod

    assert not hasattr(chat_mod.CodingBoxApp, "_cmd_toggle_wiki")
