"""Phase 0A modality disambiguation (Fieldrunners trace 20260626_102307).

The beat-em-up detector's weak trigger "waves" also appears in tower-defense
goals ("waves of enemies"), which wrongly injected a side-scrolling brawler
nudge onto a TD plan turn. The detector now suppresses itself when
tower-defense SHAPE tokens co-occur. Genuine brawler goals still fire.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
import memory as memory_module  # noqa: E402
from prompts_v1 import _detect_beat_em_up_intent  # noqa: E402


def test_suppression_comes_from_data_not_hardcode(monkeypatch):
    """Phase 4 (4A): proves the suppression is DATA-driven. With the
    visual_playtests `suppresses_nudges` loader stubbed empty, the TD goal
    NO LONGER suppresses — confirming `_detect_beat_em_up_intent` reads the
    memory file (single source of truth), not a hardcoded token list."""
    td_goal = (
        "tower defense: survive 10 waves of creeps along the path"
    )
    # Baseline: real data suppresses.
    assert _detect_beat_em_up_intent(td_goal) == []
    # Stub the loader empty -> suppression must disappear (data was the source).
    monkeypatch.setattr(memory_module, "_load_nudge_suppressors", lambda: tuple())
    assert _detect_beat_em_up_intent(td_goal) != []


def test_goal_suppresses_nudge_loader_direct():
    """The single loader function is the public disambiguation API."""
    assert memory_module.goal_suppresses_nudge(
        "open-field fieldrunners tower defense with turrets", "beat-em-up"
    ) is True
    assert memory_module.goal_suppresses_nudge(
        "a side-scrolling brawler beat-em-up on each floor", "beat-em-up"
    ) is False


def test_fieldrunners_td_goal_does_not_fire_beat_em_up():
    """The literal trace goal must NOT get a brawler nudge."""
    goal = (
        "Build an open-field Fieldrunners-style tower defense where you place "
        "turrets to stop waves of enemies from crossing the field."
    )
    assert _detect_beat_em_up_intent(goal) == []


def test_generic_tower_defense_waves_suppressed():
    assert _detect_beat_em_up_intent(
        "tower defense: survive 10 waves of creeps along the path"
    ) == []
    assert _detect_beat_em_up_intent(
        "place turrets on a grid to stop waves before they leak"
    ) == []


def test_genuine_brawler_still_fires():
    """A real side-scrolling brawler (no TD shape tokens) must still nudge."""
    hits = _detect_beat_em_up_intent(
        "a side-scrolling beat-em-up brawler: clear waves of thugs on each "
        "floor then fight the boss"
    )
    assert hits, "genuine brawler goal should still trigger the nudge"
    assert "brawler" in hits or "beat-em-up" in hits


def test_waves_alone_without_td_or_brawler_tokens():
    """'waves' as the only token still fires (unchanged) when no TD shape
    token is present — suppression is TD-specific, not a blanket disable."""
    assert _detect_beat_em_up_intent("survive endless waves of attackers") == [
        "waves"
    ]
