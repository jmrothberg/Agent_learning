"""Visual-playtest end-to-end wiring tests.

The matcher itself is covered by tests/test_visual_playtest_matcher.py.
This file covers the prompt-build + response-parse + format pipeline
in `agent.run_visual_critic`. The wiring tests use synthetic recipes
and synthetic VLM responses — no actual backend, no actual screenshot.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import memory as memory_mod  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _stub_recipe() -> memory_mod.VisualPlaytestRecipe:
    return memory_mod.VisualPlaytestRecipe(
        id="test-recipe",
        kind="visual_playtest",
        content="test",
        tags=[],
        source_tier="root",
        verified=True,
        helpful=0, harmful=0,
        recipe={
            "applies_keywords": [],
            "checklist": [
                "Is the player visible?",
                "Is the player within the canvas bounds?",
                "Are there enemies visible?",
                "Is there a visible HUD?",
            ],
            "format": "yes_no_per_line",
        },
        trace_ids=[],
        pass_count=0,
        false_positive_count=0,
        last_verified_at="",
    )


# ----------------------------------------------------------------------
# Prompt builder.
# ----------------------------------------------------------------------


def test_prompt_includes_numbered_checklist(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._goal = "build a snake game"
    prompt = a._build_visual_playtest_prompt(_stub_recipe(), before_png=None)
    assert "Q1: Is the player visible?" in prompt
    assert "Q4: Is there a visible HUD?" in prompt
    assert "build a snake game" in prompt


def test_prompt_two_image_mode(tmp_path: Path) -> None:
    """With before_png present the prompt must clarify Image 1 / 2."""
    a = _make_agent(tmp_path)
    a._goal = "test"
    prompt = a._build_visual_playtest_prompt(_stub_recipe(), before_png=b"png_bytes")
    assert "Image 1" in prompt
    assert "Image 2" in prompt
    assert "AFTER" in prompt


def test_prompt_enforces_stop_after_list(tmp_path: Path) -> None:
    """The instruction must tell the VLM to stop after the questions."""
    a = _make_agent(tmp_path)
    prompt = a._build_visual_playtest_prompt(_stub_recipe(), before_png=None)
    assert "Stop after" in prompt


def test_prompt_example_response_present(tmp_path: Path) -> None:
    """Example response shape primes the VLM toward the right format."""
    a = _make_agent(tmp_path)
    prompt = a._build_visual_playtest_prompt(_stub_recipe(), before_png=None)
    assert "Example response" in prompt
    assert "Q1: yes" in prompt
    assert "Q2: no" in prompt


# ----------------------------------------------------------------------
# Response parser.
# ----------------------------------------------------------------------


def test_parse_clean_response() -> None:
    response = (
        "Q1: yes\n"
        "Q2: no — player is clipped at right edge\n"
        "Q3: yes\n"
        "Q4: unclear\n"
    )
    parsed = GameAgent._parse_visual_playtest_response(response, _stub_recipe())
    assert parsed["n_questions"] == 4
    assert parsed["parse_rate"] == 1.0
    answers = parsed["answers"]
    assert answers[1] == ("yes", "")
    assert answers[2] == ("no", "player is clipped at right edge")
    assert answers[3] == ("yes", "")
    assert answers[4] == ("unclear", "")


def test_parse_tolerates_format_variations() -> None:
    """`Qn.`, `Qn -`, case variations, single-letter Y/N must parse."""
    response = (
        "Q1. YES\n"
        "Q2 - N — bad facing\n"
        "Q3) y\n"
        "Q4: ✗ — clipped\n"
    )
    parsed = GameAgent._parse_visual_playtest_response(response, _stub_recipe())
    assert parsed["parse_rate"] == 1.0
    a = parsed["answers"]
    assert a[1][0] == "yes"
    assert a[2][0] == "no"
    assert a[3][0] == "yes"
    # ✗ isn't in our emoji set (we accept ❌ ✖); test it falls
    # through to "unclear" rather than crashing.
    assert a[4][0] in ("unclear", "no")


def test_parse_skips_prose_lines() -> None:
    """Lines that don't match the pattern are silently skipped."""
    response = (
        "Here's my review of the screenshot:\n"
        "Q1: yes\n"
        "Some other text the VLM rambled about.\n"
        "Q2: no\n"
        "\n"
        "Q4: yes\n"
    )
    parsed = GameAgent._parse_visual_playtest_response(response, _stub_recipe())
    # Q3 wasn't answered → parse_rate = 3/4.
    assert parsed["parse_rate"] == 0.75
    assert 1 in parsed["answers"]
    assert 2 in parsed["answers"]
    assert 3 not in parsed["answers"]
    assert 4 in parsed["answers"]


def test_parse_empty_response() -> None:
    parsed = GameAgent._parse_visual_playtest_response("", _stub_recipe())
    assert parsed["parse_rate"] == 0.0
    assert parsed["answers"] == {}


def test_parse_invalid_question_numbers_ignored() -> None:
    """A 4-question recipe should ignore answers for Q5+ that the VLM hallucinated."""
    response = "Q1: yes\nQ2: yes\nQ3: no\nQ4: yes\nQ5: yes — extra\nQ99: no\n"
    parsed = GameAgent._parse_visual_playtest_response(response, _stub_recipe())
    assert 5 not in parsed["answers"]
    assert 99 not in parsed["answers"]
    assert parsed["n_questions"] == 4
    assert parsed["parse_rate"] == 1.0


# ----------------------------------------------------------------------
# Formatter — all-pass returns None; failures show up as bullet list.
# ----------------------------------------------------------------------


def test_format_all_pass_returns_none() -> None:
    parsed = {
        "answers": {1: ("yes", ""), 2: ("yes", ""), 3: ("yes", ""), 4: ("yes", "")},
        "parse_rate": 1.0,
        "n_questions": 4,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, _stub_recipe())
    assert out is None


def test_format_failures_listed_with_remarks() -> None:
    parsed = {
        "answers": {
            1: ("yes", ""),
            2: ("no", "player clipped at right edge"),
            3: ("yes", ""),
            4: ("no", "HUD overlapping playfield"),
        },
        "parse_rate": 1.0,
        "n_questions": 4,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, _stub_recipe())
    assert out is not None
    assert "[VISUAL PLAYTEST — test-recipe]" in out
    assert "2 of 4 check(s) failed" in out
    assert "Q2" in out
    assert "player clipped at right edge" in out
    assert "Q4" in out
    assert "HUD overlapping playfield" in out
    # passed-checks listed once (not per-pass-question — that would
    # bloat the prompt).
    assert "Q1" not in out  # yes-checks not enumerated
    assert "Q3" not in out


def test_format_unclear_counted_separately() -> None:
    parsed = {
        "answers": {
            1: ("yes", ""),
            2: ("unclear", "VLM couldn't see clearly"),
            3: ("yes", ""),
            4: ("yes", ""),
        },
        "parse_rate": 1.0,
        "n_questions": 4,
    }
    out = GameAgent._format_visual_playtest_critique(parsed, _stub_recipe())
    assert out is not None
    assert "0 of 4 check(s) failed" in out
    assert "1 unclear" in out
    assert "UNCLEAR" in out


def test_format_no_answers_returns_none() -> None:
    parsed = {"answers": {}, "parse_rate": 0.0, "n_questions": 4}
    out = GameAgent._format_visual_playtest_critique(parsed, _stub_recipe())
    assert out is None


# ----------------------------------------------------------------------
# End-to-end: matcher retrieves the right recipe for real user goals.
# ----------------------------------------------------------------------


def test_e2e_real_goals_pick_expected_recipe(tmp_path: Path) -> None:
    """Sanity check: the canonical committed library
    `memory/visual_playtests.jsonl` resolves the user's actual past
    goals to the right mechanism recipe."""
    repo_memory = Path(__file__).parent.parent / "memory"
    mem = memory_mod.GameMemory(root=str(repo_memory))
    cases = [
        ("write a game of doom, first person shooter", "canvas-3d-first-person"),
        ("mortal kombat / street fighter two player versus", "canvas-two-actors-facing"),
        ("collect dots while avoiding ghosts in corridors", "canvas-grid-navigation"),
        ("asteroids clone with player ship in the middle", "canvas-top-down-action"),
        ("puzzle game where shapes fall and you stack rows", "canvas-puzzle-grid"),
        ("side scrolling platformer hero jumping on platforms", "canvas-side-scroll-platformer"),
    ]
    for goal, expected_id in cases:
        r, diag = mem.find_visual_playtest_for(goal=goal)
        assert r is not None, f"no recipe for {goal!r}; diag={diag}"
        assert r.id == expected_id, (
            f"goal {goal!r} matched {r.id!r}, expected {expected_id!r}; "
            f"top: {diag['top_candidates']}"
        )


def test_e2e_novel_goal_returns_none(tmp_path: Path) -> None:
    """Goals genuinely outside the mechanism library return None;
    caller falls back to the legacy generic critic prompt."""
    repo_memory = Path(__file__).parent.parent / "memory"
    mem = memory_mod.GameMemory(root=str(repo_memory))
    r, _ = mem.find_visual_playtest_for(
        goal="a breathing meditation timer with calming sounds"
    )
    assert r is None


# ----------------------------------------------------------------------
# #3 — critic retry-once-with-terse-reformat before dropping unparseable.
# ----------------------------------------------------------------------

class _StubReply:
    def __init__(self, text: str):
        self.text = text


class _RetryBackend:
    """Returns rambling prose first (unparseable), a clean checklist on retry."""
    info = type("_Info", (), {"model": "stub", "name": "stub"})()

    def __init__(self, first: str, second: str):
        self._replies = [first, second]
        self.calls = 0

    async def is_vlm(self):
        # run_visual_critic now skips non-vision backends (2026-06-02 guard);
        # this stub stands in for a real VLM so the retry path is exercised.
        return True

    async def stream_chat(self, messages, **kwargs):
        i = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return _StubReply(self._replies[i])


def _critic_agent(tmp_path: Path, backend) -> GameAgent:
    a = _make_agent(tmp_path)
    a._backend = backend
    a._use_vlm_critique = True
    a._all_roles_enabled = True  # single-slot critic path via coder
    a._goal = ""
    a._criteria = ""
    return a


import asyncio  # noqa: E402


def test_visual_critic_retries_then_uses_reparsed(tmp_path: Path, monkeypatch) -> None:
    recipe = _stub_recipe()
    monkeypatch.setattr(
        memory_mod.GameMemory, "find_visual_playtest_for",
        lambda self, **kw: (recipe, {}),
    )
    # First reply rambles (0 parseable lines); retry answers Q3: no cleanly.
    backend = _RetryBackend(
        "Well, let me think about this screenshot in great detail...",
        "Q1: yes\nQ2: yes\nQ3: no\nQ4: yes",
    )
    a = _critic_agent(tmp_path, backend)
    out = asyncio.run(a.run_visual_critic(b"\x89PNG-current"))
    # The retry was issued (two backend calls) and its parsed failure surfaced.
    assert backend.calls == 2
    assert out is not None
    assert "enemies" in out.lower() or "Q3" in out or "3" in out


def test_visual_critic_drops_when_retry_also_unparseable(tmp_path: Path, monkeypatch) -> None:
    recipe = _stub_recipe()
    monkeypatch.setattr(
        memory_mod.GameMemory, "find_visual_playtest_for",
        lambda self, **kw: (recipe, {}),
    )
    backend = _RetryBackend("rambling one", "still rambling, no Qn lines here")
    a = _critic_agent(tmp_path, backend)
    out = asyncio.run(a.run_visual_critic(b"\x89PNG-current"))
    assert backend.calls == 2
    # Nothing parseable from either pass -> dropped (never raw chain-of-thought).
    assert out is None
