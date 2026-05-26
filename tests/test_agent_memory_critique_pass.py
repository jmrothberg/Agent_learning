"""Tests for the 2026-05-25 agent-memory critique-pass additions:

(1) 6 new edit/scope-discipline playbook bullets in `memory/playbook.jsonl`
    retrieve on realistic scoped-feedback goals and stay absent from
    unrelated simple goals.

(2) New auto_probes added to 4 visual_playtest recipes that lacked them.
    Each is a valid IIFE and conservative (returns true when state shape
    is absent — never false-fails on a game whose state we don't know).

(3) New optional `fix_hint` field on visual_playtest recipes — rendered
    into critic coaching ONLY when the recipe has failures (not just
    unclears, not when all-pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK_PATH = PROJECT_ROOT / "memory" / "playbook.jsonl"
PLAYTESTS_PATH = PROJECT_ROOT / "memory" / "visual_playtests.jsonl"


# ---------------------------------------------------------------------------
# Round-1: edit/scope-discipline playbook bullets retrieve
# ---------------------------------------------------------------------------

NEW_PLAYBOOK_IDS = {
    "scope-flip-minimal-change",
    "scope-locked-by-user-language",
    "add-feature-do-not-touch-working-code",
    "previous-user-fix-is-locked",
    "patch-budget-when-scope-locked",
    "vlm-critic-can-mislead-on-orientation",
}


def test_new_edit_discipline_bullets_exist_in_playbook():
    """All 6 bullets present in the JSONL file with non-empty content + tags."""
    seen = {}
    with PLAYBOOK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] in NEW_PLAYBOOK_IDS:
                seen[d["id"]] = d
    missing = NEW_PLAYBOOK_IDS - seen.keys()
    assert not missing, f"missing playbook bullets: {missing}"
    for bid, b in seen.items():
        assert b.get("content"), f"{bid} has empty content"
        assert b.get("tags"), f"{bid} has empty tags"
        assert isinstance(b["tags"], list)


def test_edit_discipline_bullets_retrieve_on_flip_scoped_feedback():
    """A goal-text simulating the Street Fighter feedback ('flip just the
    punch, no other changes') must surface the edit-discipline bullets in
    the top retrieval hits.
    """
    import os

    from memory import Playbook

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        pb = Playbook(base_root="memory")
        goal = (
            "flip just the punch direction 180 degrees no other changes "
            "only punch do not touch other states"
        )
        hits = pb.retrieve(goal, k=8, stage="code")
    finally:
        os.chdir(old_cwd)
    retrieved_ids = {h.bullet.id for h in hits}
    # At least 3 of the 6 new bullets must surface — they overlap heavily
    # on tags like 'only', 'scope', 'flip', 'change'.
    new_in_top = NEW_PLAYBOOK_IDS & retrieved_ids
    assert len(new_in_top) >= 3, (
        f"only {len(new_in_top)} of {len(NEW_PLAYBOOK_IDS)} new bullets "
        f"retrieved on scoped-flip feedback. retrieved: {sorted(retrieved_ids)}"
    )


def test_edit_discipline_bullets_retrieve_on_add_feature_feedback():
    """A goal simulating 'add a flying fireball, do not change other code'
    must retrieve the add-feature-don't-touch-working-code bullet.
    """
    import os

    from memory import Playbook

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        pb = Playbook(base_root="memory")
        goal = (
            "add a flying fireball that travels across the screen "
            "do not change any other code do not make new assets"
        )
        hits = pb.retrieve(goal, k=6, stage="code")
    finally:
        os.chdir(old_cwd)
    retrieved_ids = {h.bullet.id for h in hits}
    assert "add-feature-do-not-touch-working-code" in retrieved_ids, (
        f"add-feature bullet missing; retrieved: {sorted(retrieved_ids)}"
    )


def test_edit_discipline_bullets_absent_on_simple_unrelated_goal():
    """The bullets target scoped-feedback failure shapes — they must NOT
    flood retrieval on simple unrelated goals like 'snake game with arrow
    keys' or 'tic-tac-toe'. False positives there would dilute the
    retrieval slot.
    """
    import os

    from memory import Playbook

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        pb = Playbook(base_root="memory")
        for goal in [
            "snake game with arrow keys",
            "tic tac toe",
            "classic asteroids vector graphics",
        ]:
            hits = pb.retrieve(goal, k=5, stage="code")
            retrieved_ids = {h.bullet.id for h in hits}
            overlap = NEW_PLAYBOOK_IDS & retrieved_ids
            assert not overlap, (
                f"goal {goal!r}: new edit-discipline bullets crowded in "
                f"on a simple unrelated goal: {overlap}"
            )
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Round-2: new auto_probes are valid IIFEs and conservative
# ---------------------------------------------------------------------------

NEW_AUTO_PROBE_NAMES = {
    "auto_actors_face_each_other_strict",
    "auto_platformer_has_multiple_platforms",
    "auto_platformer_player_has_vertical_motion_state",
    "auto_fp_player_has_yaw_or_angle",
    "auto_board_is_2d_array_and_turn_exposed",
}


def test_new_auto_probes_exist_and_are_iifes():
    """Each new auto_probe must be an IIFE `(()=>{...})()` (matches the
    test_visual_playtest_auto_probes.py guard) and have balanced parens.
    """
    seen = {}
    with PLAYTESTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for ap in d.get("recipe", {}).get("auto_probes", []):
                if ap["name"] in NEW_AUTO_PROBE_NAMES:
                    seen[ap["name"]] = ap
    missing = NEW_AUTO_PROBE_NAMES - seen.keys()
    assert not missing, f"missing auto_probes: {missing}"
    for name, ap in seen.items():
        expr = ap["expr"]
        assert expr.startswith("(()=>{"), f"{name} not an IIFE"
        assert expr.endswith("})()"), f"{name} not an IIFE close"
        assert expr.count("(") == expr.count(")"), f"{name} parens unbalanced"
        assert expr.count("{") == expr.count("}"), f"{name} braces unbalanced"


# ---------------------------------------------------------------------------
# Round-3: fix_hint rendering in critic coaching
# ---------------------------------------------------------------------------


class _StubRecipe:
    def __init__(self, rid: str, checklist: list[str], fix_hint: str | None = None):
        self.id = rid
        self.recipe = {"checklist": checklist}
        if fix_hint is not None:
            self.recipe["fix_hint"] = fix_hint


def test_fix_hint_renders_when_failures_present():
    """When the recipe has a fix_hint AND there's at least one FAILED
    answer, the critique text appends "Minimal fix shape: ..." with the
    hint contents.
    """
    recipe = _StubRecipe(
        "test",
        ["Is X visible?", "Is Y aligned?"],
        fix_hint="Wrap drawImage in ctx.save/scale/restore; do not refactor draw().",
    )
    parsed = {
        "answers": {1: ("no", "X not visible"), 2: ("yes", None)},
        "n_questions": 2,
        "parse_rate": 1.0,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, recipe)
    assert out is not None
    assert "FAILED" in out
    assert "Minimal fix shape:" in out
    assert "ctx.save/scale/restore" in out


def test_fix_hint_omitted_when_all_pass():
    """All-pass critiques return None — no critique text at all, so
    fix_hint never surfaces.
    """
    recipe = _StubRecipe(
        "test", ["Is X visible?"],
        fix_hint="this should not appear",
    )
    parsed = {"answers": {1: ("yes", None)}, "n_questions": 1, "parse_rate": 1.0}
    out = GameAgent._format_visual_playtest_critique(parsed, recipe)
    assert out is None


def test_fix_hint_omitted_when_only_unclears():
    """UNCLEAR answers don't trigger fix_hint — the hint speaks to
    concrete failures; an unclear means the critic couldn't tell,
    which doesn't need a fix suggestion.
    """
    recipe = _StubRecipe(
        "test",
        ["Is X visible?", "Is Y aligned?"],
        fix_hint="this should not appear when only unclears",
    )
    parsed = {
        "answers": {1: ("unclear", "can't tell"), 2: ("yes", None)},
        "n_questions": 2,
        "parse_rate": 1.0,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, recipe)
    assert out is not None
    assert "UNCLEAR" in out
    assert "Minimal fix shape:" not in out


def test_fix_hint_absent_field_degrades_gracefully():
    """Recipes without a fix_hint field render normally (no error, no
    "Minimal fix shape:" line). Backwards compatibility with the 10 of
    19 recipes that don't have a hint set today.
    """
    recipe = _StubRecipe("test", ["Is X visible?"], fix_hint=None)
    parsed = {
        "answers": {1: ("no", "X missing")},
        "n_questions": 1,
        "parse_rate": 1.0,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, recipe)
    assert out is not None
    assert "FAILED" in out
    assert "Minimal fix shape:" not in out


def test_fix_hint_present_on_high_priority_recipes():
    """The recipes most likely to surface critic coaching that misled the
    model in prior traces (canvas-two-actors-facing was the Street Fighter
    case) must have fix_hint set. Pin the must-have list so a future
    edit can't quietly remove them.
    """
    must_have_hint = {
        "canvas-two-actors-facing",
        "canvas-side-scroll-platformer",
        "canvas-3d-first-person",
        "canvas-board-game",
        "canvas-controllable-player",
    }
    found = set()
    with PLAYTESTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] in must_have_hint and d.get("recipe", {}).get("fix_hint"):
                found.add(d["id"])
    missing = must_have_hint - found
    assert not missing, f"recipes missing fix_hint: {missing}"
