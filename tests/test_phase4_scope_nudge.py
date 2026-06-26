"""Phase 4 regression tests — scope guardrails (genre-free, prompt-only).

Verifies:
  - `_detect_heavy_logic_intent` matches generic logic keywords (no genres).
  - The plan-instruction nudge only fires when art (or 3D) AND heavy-logic
    intents BOTH match.
  - `_no_usable_code_fallback` names the duplicate-decl failure shape
    when materialize rejected on that ground.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory as memory_module  # noqa: E402
from agent import GameAgent  # noqa: E402
from prompts_v1 import (  # noqa: E402
    _detect_art_intent,
    _detect_heavy_logic_intent,
    plan_instruction,
)


def test_plan_nudge_prose_comes_from_data_loader(monkeypatch):
    """Phase 4 (4A): plan-turn nudge PROSE lives in memory/plan_nudges.jsonl;
    prompts_v1 reads it through memory.load_plan_nudge. Proven by stubbing the
    loader: a sentinel body for "art" must appear verbatim in plan_instruction
    output (the kws placeholder filled), confirming the data file is the source
    of truth, not a hardcoded string."""
    # Clear any cache so the monkeypatch takes effect for this loader.
    monkeypatch.setattr(memory_module, "_PLAN_NUDGES_CACHE", None, raising=False)
    real = memory_module.load_plan_nudge

    def fake(nudge_id: str) -> str:
        if nudge_id == "art":
            return "\n\nSENTINEL-ART {kws} BODY\n"
        return real(nudge_id)

    monkeypatch.setattr(memory_module, "load_plan_nudge", fake)
    out = plan_instruction(goal="make a sprite-based pixel-art shooter")
    assert "SENTINEL-ART" in out
    # The {kws} slot was interpolated by prompts_v1, not left literal.
    assert "{kws}" not in out


def test_plan_nudges_file_has_expected_ids():
    """All migrated nudge ids resolve to non-empty bodies (no silent drop)."""
    for nid in ("art", "3d", "beat-em-up", "audio", "video", "qte",
                "wireframe-perspective", "wireframe-flat", "scope-pacing",
                "scope-pacing-multiframe", "multi-frame", "minimal-first-build"):
        assert memory_module.load_plan_nudge(nid).strip(), nid


def test_heavy_logic_keywords_match_logic_terms_not_genres() -> None:
    """The detector matches computational shape, not subject matter."""
    assert _detect_heavy_logic_intent("3 ply minimax search opponent") != []
    assert _detect_heavy_logic_intent("multiplayer save undo with bot AI") != []
    # Genre tokens like "chess" / "rts" / "doom" are NOT in the keyword set.
    assert _detect_heavy_logic_intent("a game of chess") == []
    assert _detect_heavy_logic_intent("simple pong with paddle") == []


def test_scope_nudge_only_when_art_and_logic_both_match() -> None:
    """The pacing nudge requires BOTH art (or 3D) intent AND heavy-logic."""
    art_only = "make a sprite-based shooter with pixel art"
    logic_only = "chess engine with 3 ply minimax search"
    combined = "monster sprites with 3 ply minimax search and full ruleset"

    text_art = plan_instruction(goal=art_only)
    text_logic = plan_instruction(goal=logic_only)
    text_combined = plan_instruction(goal=combined)

    assert "SCOPE-PACING NUDGE" not in text_art
    assert "SCOPE-PACING NUDGE" not in text_logic
    assert "SCOPE-PACING NUDGE" in text_combined
    # The combined nudge must surface the matched logic keywords.
    assert "minimax" in text_combined.lower()
    assert _detect_art_intent(combined) != []


def test_scope_nudge_genre_free_no_chess_or_pacman() -> None:
    """Sanity: the nudge text never names a specific game genre."""
    combined = "monster sprites with 3 ply minimax search and full ruleset"
    text = plan_instruction(goal=combined)
    lower = text.lower()
    # No genre names should appear in the nudge body.
    for genre in ("chess", "pacman", "doom", "asteroids", "snake", "tetris"):
        assert genre not in lower, f"genre '{genre}' leaked into prompt"


def test_no_usable_code_fallback_names_duplicate_decl_shape() -> None:
    """When materialize rejected on duplicate top-level declarations,
    the fallback message must surface that exact shape and ask for
    a single-pass body."""
    reason = (
        "<html_file> rejected: duplicate top-level declaration(s) in "
        "<script>: `ctx`, `state` declared 2+ times at the same scope "
        "— looks like two drafts got concatenated."
    )
    msg, _reset = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        materialize_reject_reason=reason,
    )
    low = msg.lower()
    assert "duplicate" in low
    assert "concatenated" in low or "single coherent body" in low
    assert "narrow scope" in low or "<question>" in low.lower()


def test_no_usable_code_fallback_unrelated_reason_keeps_default() -> None:
    """Sanity: a non-duplicate reject reason routes through normal paths.
    """
    msg, _reset = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        materialize_reject_reason="no <patch> or <html_file> in reply",
    )
    # The duplicate-decl branch must not fire on unrelated rejections.
    assert "DUPLICATE TOP-LEVEL DECLARATIONS" not in msg
