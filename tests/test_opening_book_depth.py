"""Tests for the 2026-06-11 Deep Universal Opening Book round.

The root-level opening book (`memory/implementation_outlines.jsonl`) grew
from 22 one-sentence outlines with empty `recipe: {}` into a chess-style
book: every entry carries a structured `recipe` (state / order / traps /
tuning / probes) and ~10 new mechanism families were added (tower-defense,
match-3, rhythm, stealth, roguelike, pinball, idle, bullet-hell,
word/typing, physics-stacking).

Local-model-first constraints pinned here:
 1. Every outline parses and has a non-empty recipe with state/order/traps.
 2. Hard per-field caps: order <= 7 lines, traps 3-6 x <= 140 chars,
    tuning <= 3, probes <= 3 — so the deep render stays terse for a 27B.
 3. No code fences inside recipes (components.jsonl owns code snippets).
 4. Deep render appears at the plan budget and NOT in the shallow
    (code-stage) render, so iterate-loop prompts never grow.
 5. Rendered deep block stays within the plan-stage budget.
 6. New family outlines retrieve on representative goals; the canonical
    fight/asteroids free-form goals route correctly (dilution guard).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTLINES_PATH = PROJECT_ROOT / "memory" / "implementation_outlines.jsonl"

PLAN_BUDGET = 3600   # agent.py plan-stage char_budget
CODE_BUDGET = 1400   # agent.py code-stage char_budget (shallow only)

NEW_FAMILY_IDS = [
    "outline-tower-defense",
    "outline-match3",
    "outline-rhythm",
    "outline-stealth",
    "outline-roguelike",
    "outline-pinball",
    "outline-idle-clicker",
    "outline-bullet-hell",
    "outline-word-typing",
    "outline-stacking-physics",
]

# Representative goal -> outline id that MUST win (broadened-book coverage).
FAMILY_GOALS = [
    ("build a tower defense game with waves of creeps", "outline-tower-defense"),
    ("match-3 puzzle game like bejeweled with cascades", "outline-match3"),
    ("rhythm game where notes scroll and you hit them on the beat", "outline-rhythm"),
    ("stealth game where you sneak past guards with vision cones", "outline-stealth"),
    ("roguelike dungeon crawler with procedural rooms and fog of war", "outline-roguelike"),
    ("pinball table with flippers and bumpers", "outline-pinball"),
    ("idle clicker game like cookie clicker with generators and prestige", "outline-idle-clicker"),
    ("bullet hell boss fight with spiral patterns", "outline-bullet-hell"),
    ("typing game where you type falling words before they land", "outline-word-typing"),
    ("stacking game where you drop blocks from a swinging crane to build a tower", "outline-stacking-physics"),
]

# Dilution guards: book growth must not steal these canonical goals.
CANONICAL_GOALS = [
    ("build a single-screen 2d fighting game with punches and kicks", "outline-two-actors-facing"),
    ("asteroids game with thrust and irregular rocks", "outline-top-down-action"),
]


def _load_outlines() -> list[dict]:
    out = []
    with OUTLINES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _memory():
    from memory import GameMemory
    return GameMemory(root=str(PROJECT_ROOT / "memory"))


# ---------------------------------------------------------------- shape ---

def test_book_has_32_entries_including_new_families():
    outlines = _load_outlines()
    assert len(outlines) >= 32, f"expected >=32 outlines, got {len(outlines)}"
    ids = {o["id"] for o in outlines}
    missing = [i for i in NEW_FAMILY_IDS if i not in ids]
    assert not missing, f"missing new family outlines: {missing}"


def test_every_outline_has_deep_recipe():
    for o in _load_outlines():
        r = o.get("recipe")
        assert isinstance(r, dict) and r, f"{o['id']} has empty recipe"
        assert str(r.get("state", "")).strip(), f"{o['id']} recipe missing state"
        assert r.get("order"), f"{o['id']} recipe missing order"
        assert r.get("traps"), f"{o['id']} recipe missing traps"


def test_recipe_field_caps_for_local_model_prompt_discipline():
    """Per-field caps keep the deep render terse for a local 27B. If this
    fires, trim the entry — do not raise the caps."""
    for o in _load_outlines():
        r = o["recipe"]
        rid = o["id"]
        assert 1 <= len(r["order"]) <= 7, f"{rid}: order has {len(r['order'])} lines (max 7)"
        assert 3 <= len(r["traps"]) <= 6, f"{rid}: traps has {len(r['traps'])} lines (3-6)"
        for t in r["traps"]:
            assert len(t) <= 140, f"{rid}: trap exceeds 140 chars: {t!r}"
        assert 1 <= len(r.get("tuning", ["x"])) <= 3, f"{rid}: tuning > 3 lines"
        assert 1 <= len(r.get("probes", ["x"])) <= 3, f"{rid}: probes > 3 lines"


def test_no_code_fences_in_recipes():
    """The book is prose/structured data — components.jsonl owns code."""
    for o in _load_outlines():
        blob = json.dumps(o["recipe"])
        assert "```" not in blob, f"{o['id']} recipe contains a code fence"


def test_traps_are_single_lines():
    for o in _load_outlines():
        for t in o["recipe"]["traps"]:
            assert "\n" not in t, f"{o['id']} trap spans multiple lines: {t!r}"


# --------------------------------------------------------------- render ---

def test_deep_render_appears_at_plan_budget_not_code_budget():
    from memory import render_opening_book_block

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = _memory()
        outline = mem.retrieve_implementation_outline(
            "two character fighting game with punch kick fireball"
        )
        assert outline is not None
        deep = render_opening_book_block(
            outline, [], [], [], char_budget=PLAN_BUDGET, deep=True,
        )
        shallow = render_opening_book_block(
            outline, [], [], [], char_budget=CODE_BUDGET,
        )
    finally:
        os.chdir(old_cwd)
    assert "traps:" in deep, "plan-stage render is missing the deep recipe"
    assert "state:" in deep and "order:" in deep
    assert "traps:" not in shallow, "code-stage render must stay shallow"
    assert "state:" not in shallow


def test_deep_render_fits_plan_budget_for_every_outline():
    """Deep render of EVERY outline (alone) must fit the plan budget with
    room for playtests/audits — cap each outline's deep section at 2600
    so the 3600 budget keeps ~1000 chars of headroom."""
    from memory import (
        OpeningBookHit, _render_outline_recipe, render_opening_book_block,
    )
    from memory import ImplementationOutline

    for o in _load_outlines():
        item = ImplementationOutline.from_record(dict(o), source_tier="root")
        hit = OpeningBookHit(item=item, score=0.5)
        block = render_opening_book_block(
            hit, [], [], [], char_budget=PLAN_BUDGET, deep=True,
        )
        assert "[opening book truncated by budget]" not in block, (
            f"{o['id']} deep render alone exceeds the plan budget"
        )
        section = len(item.content) + len(_render_outline_recipe(item.recipe))
        assert section <= 2600, (
            f"{o['id']} outline section is {section} chars — trim the entry "
            f"so playtests/audits keep headroom in the 3600 budget"
        )


def test_render_handles_empty_recipe_gracefully():
    from memory import _render_outline_recipe
    assert _render_outline_recipe({}) == ""
    assert _render_outline_recipe(None) == ""


# ------------------------------------------------------------ retrieval ---

def test_new_families_retrieve_on_representative_goals():
    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = _memory()
        misses = []
        for goal, expected in FAMILY_GOALS:
            hit = mem.retrieve_implementation_outline(goal)
            actual = hit.item.id if hit else None
            if actual != expected:
                misses.append((goal, expected, actual))
    finally:
        os.chdir(old_cwd)
    assert not misses, "family goal misroutes:\n" + "\n".join(
        f"  {g!r} expected {e}, got {a}" for g, e, a in misses
    )


def test_canonical_goals_survive_book_growth():
    """Dilution guard: adding 10 entries + tags must not steal the
    canonical fight / asteroids goals (DEV.md regression pair)."""
    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = _memory()
        misses = []
        for goal, expected in CANONICAL_GOALS:
            hit = mem.retrieve_implementation_outline(goal)
            actual = hit.item.id if hit else None
            if actual != expected:
                misses.append((goal, expected, actual))
    finally:
        os.chdir(old_cwd)
    assert not misses, "canonical goal misroutes:\n" + "\n".join(
        f"  {g!r} expected {e}, got {a}" for g, e, a in misses
    )


def test_dig_dug_and_qbert_route_correctly():
    """The two preset mis-routings fixed by tag enrichment (data-only)."""
    lib_path = PROJECT_ROOT / "memory" / "prompt_library.jsonl"
    prompts = {}
    with lib_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                prompts[g["name"]] = g
    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = _memory()
        for name, expected in (
            ("dig-dug", "outline-grid-navigation"),
            ("qbert", "outline-pyramid-hopper"),
        ):
            hit = mem.retrieve_implementation_outline(prompts[name]["prompt"])
            actual = hit.item.id if hit else None
            assert actual == expected, f"{name}: expected {expected}, got {actual}"
            # expect block in the library must agree with live routing
            assert prompts[name]["expect"]["outline"] == expected, (
                f"{name}: prompt_library expect.outline is stale"
            )
    finally:
        os.chdir(old_cwd)
