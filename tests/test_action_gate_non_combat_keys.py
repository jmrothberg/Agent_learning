"""Phase 0B non-combat key exclusion for ACTION_DRAWN_NOT_SPRITED
(Fieldrunners trace 20260626_102307).

Criteria like "Space starts a wave" named a flow/menu key; the gate treated
Space as a combat action key and produced a false "faked kick" diagnosis.
`_non_combat_action_keys` now excludes keys whose nearby phrase describes a
start-wave / pause / menu / build / sell control. Genre-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import _non_combat_action_keys, _parse_action_keys  # noqa: E402


def test_space_start_wave_excluded():
    crit = "Space starts a wave of enemies; ArrowKeys pan the camera."
    assert "Space" in _parse_action_keys(crit)
    assert "Space" in _non_combat_action_keys(crit)


def test_pause_and_menu_keys_excluded():
    crit = "KeyP pauses the game. Enter opens the build menu."
    non_combat = _non_combat_action_keys(crit)
    assert "KeyP" in non_combat
    assert "Enter" in non_combat


def test_build_and_sell_keys_excluded():
    crit = "Digit1 places a turret, KeyS sells the selected tower."
    non_combat = _non_combat_action_keys(crit)
    assert "Digit1" in non_combat
    assert "KeyS" in non_combat


def test_genuine_attack_key_not_excluded():
    """A real attack key with no flow verb nearby must stay in the gate set."""
    crit = "KeyF throws a punch; KeyK performs a special attack."
    non_combat = _non_combat_action_keys(crit)
    assert "KeyF" not in non_combat
    assert "KeyK" not in non_combat


def test_empty_criteria_safe():
    assert _non_combat_action_keys("") == set()
    assert _non_combat_action_keys(None or "") == set()
