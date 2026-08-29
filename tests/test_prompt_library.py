"""Tests for the curated /games prompt library (prompt_library.py + the
shipped memory/prompt_library.jsonl)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_library import load_prompt_library, get_prompt  # noqa: E402

_REPO = Path(__file__).parent.parent
_SHIPPED = _REPO / "memory" / "prompt_library.jsonl"


def test_shipped_library_loads_and_is_numbered_contiguously():
    games = load_prompt_library(_SHIPPED)
    assert len(games) == 54
    # Numbered 1..N with no gaps, sorted.
    assert [g["n"] for g in games] == list(range(1, len(games) + 1))


def test_showcase_genres_present():
    """Beyond the arcade classics, the library must cover the genres the
    agent has dedicated recipes+outlines+skeletons for (2026-05-31 audit)."""
    names = {g["name"] for g in load_prompt_library(_SHIPPED)}
    for genre in ("doom", "minecraft", "outrun", "chess", "zelda", "monkey-island"):
        assert genre in names, f"missing showcase genre: {genre}"
    assert "dragons-lair" in names
    for vector_game in ("battlezone", "star-wars"):
        assert vector_game in names, f"missing vector wireframe game: {vector_game}"


def test_first_three_are_the_named_games():
    games = load_prompt_library(_SHIPPED)
    by_n = {g["n"]: g for g in games}
    assert by_n[1]["name"] == "street-fighter"
    assert by_n[2]["name"] == "donkey-kong"
    assert by_n[3]["name"] == "centipede"


def test_every_entry_has_title_and_nonempty_prompt():
    for g in load_prompt_library(_SHIPPED):
        assert g.get("title")
        assert isinstance(g["prompt"], str) and len(g["prompt"]) > 40


def test_character_prompts_name_action_poses():
    """The animation showcase prompts must enumerate the poses so the
    multi-frame planner generates a frame per named action up front."""
    by_name = {g["name"]: g for g in load_prompt_library(_SHIPPED)}
    sf = by_name["street-fighter"]["prompt"].lower()
    for pose in ("idle", "punch", "kick", "jump", "duck", "fireball"):
        assert pose in sf, f"street-fighter prompt missing '{pose}'"


def test_get_prompt_by_number():
    g = get_prompt(1, _SHIPPED)
    assert g is not None and g["name"] == "street-fighter"
    assert get_prompt(999, _SHIPPED) is None


def test_every_entry_has_prompt_640_for_simulator():
    """/640 mode needs a pixel-map variant for every curated game."""
    from prompt_library import effective_prompt

    for g in load_prompt_library(_SHIPPED):
        p640 = g.get("prompt_640")
        assert isinstance(p640, str) and len(p640) > 40, g["name"]
        assert "TARGET=/640" in p640 or "pixel" in p640.lower(), g["name"]
        assert "image-to-video" not in p640.lower()
        # Forbid requiring media tags (footer says do NOT emit them).
        assert "emit <assets>" not in p640.lower()
        assert "generate video" not in p640.lower()
        # Media prompt kept intact for non-/640 runs.
        assert g["prompt"].startswith(("Build a ", "Build an "))
        assert effective_prompt(g, simulator_mode=True) == p640.strip()
        assert effective_prompt(g, simulator_mode=False) == g["prompt"].strip()


def test_prompt_640_fieldrunners_demands_pixel_sprites_not_boxes():
    by_name = {g["name"]: g for g in load_prompt_library(_SHIPPED)}
    p = by_name["tower-defense-openfield"]["prompt_640"].lower()
    assert "pixel" in p
    assert "colored squares" in p or "not colored" in p
    assert "generate <assets>" not in p


def test_prompt_640_strips_video_pipeline_from_fighters():
    by_name = {g["name"]: g for g in load_prompt_library(_SHIPPED)}
    for name in ("street-fighter", "mortal-kombat", "super-mario"):
        p = by_name[name]["prompt_640"]
        assert "image-to-video" not in p.lower()
        assert "generate video cutscenes" not in p.lower()


def test_malformed_lines_are_skipped(tmp_path: Path):
    p = tmp_path / "lib.jsonl"
    p.write_text(
        '{"n": 1, "name": "a", "prompt": "build a"}\n'
        "not json at all\n"
        '{"n": 2, "prompt": ""}\n'              # empty prompt -> skipped
        '{"name": "c", "prompt": "no number"}\n'  # missing n -> skipped
        '{"n": 3, "name": "c", "prompt": "build c"}\n',
        encoding="utf-8",
    )
    games = load_prompt_library(p)
    assert [g["n"] for g in games] == [1, 3]
    assert games[0]["title"] == "a"  # title defaults to name


def test_missing_file_returns_empty(tmp_path: Path):
    assert load_prompt_library(tmp_path / "nope.jsonl") == []


def test_prompts_use_direct_openers_no_style_hedging():
    """Library goals open with a direct game name, not '-style' hedging."""
    hedging_tokens = ("should", "try to", "strongly prefer")
    for g in load_prompt_library(_SHIPPED):
        prompt = g["prompt"]
        assert prompt.startswith(("Build a ", "Build an ")), (
            f"#{g['n']} {g['name']}: prompt must start with 'Build a/an'"
        )
        first_sentence = prompt.split(".")[0]
        low_first = first_sentence.lower()
        assert "-style" not in low_first, (
            f"#{g['n']} {g['name']}: first sentence contains '-style'"
        )
        assert " style " not in low_first, (
            f"#{g['n']} {g['name']}: first sentence contains ' style '"
        )
        head = prompt[:120].lower()
        for tok in hedging_tokens:
            assert tok not in head, (
                f"#{g['n']} {g['name']}: hedging token {tok!r} in first 120 chars"
            )
