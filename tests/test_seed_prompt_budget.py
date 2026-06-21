"""Seed-continuation prompt size + memory budget (2026-06-21 trace).

A large /seed file (~26KB) inlined in full ballooned iter-1 context to
15K+ prompt tokens and drove the local model into repetition/deliberation
loops; the lean budget also dropped the `help-overlay-modal` component
exactly when the weak model needed the snippet. These tests pin the three
general fixes: seed HTML excerpting, structural-token extraction, and
component-protected lean budget. All pure-function — no model/Chromium.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from prompts_v1 import seed_build_instruction  # noqa: E402


def _stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._criteria = ""
    a._last_test_report = None
    a._lean_prompt = True  # force lean budget active for the budget tests
    a._trace = lambda obj: None
    return a


# ---- Fix 2: seed HTML excerpting -------------------------------------------

def test_small_seed_html_passed_in_full() -> None:
    a = _stub()
    html = "<html><body>" + ("x" * 500) + "</body></html>"
    out, truncated = a._seed_html_for_prompt(html, None)
    assert out == html
    assert truncated is False


def test_large_seed_html_is_excerpted_and_bounded() -> None:
    a = _stub()
    # Comfortably over _FULL_FILE_INJECT_LIMIT (12_000).
    head = "<head><style>#helpBtn{}</style></head>\n"
    middle = "\n".join(f"function logic{i}() {{ return {i}; }}" for i in range(2000))
    tail = "\n<script>boot();</script></html>"
    html = head + middle + tail
    assert len(html) > GameAgent._FULL_FILE_INJECT_LIMIT

    out, truncated = a._seed_html_for_prompt(html, None)
    assert truncated is True
    # Excerpt + the elision marker stays comfortably under 2x the limit.
    assert len(out) < GameAgent._FULL_FILE_INJECT_LIMIT + 500
    # Head structure (UI anchors) is preserved.
    assert "#helpBtn" in out
    # Boot/init tail is preserved.
    assert "boot();" in out
    # The elision marker tells the model the full file is on disk.
    assert "elided" in out
    assert "on-disk file" in out


def test_seed_build_instruction_truncated_wording() -> None:
    full = seed_build_instruction("<html>x</html>", "/tmp/seed.html")
    assert "EXISTING FILE:" in full
    assert "EXCERPT" not in full

    excerpted = seed_build_instruction(
        "<html>x</html>", "/tmp/seed.html", truncated=True,
    )
    assert "EXCERPT" in excerpted
    assert "on-disk file" in excerpted


# ---- Fix 3b: structural token extraction -----------------------------------

def test_seed_structural_tokens_pull_ids_and_functions() -> None:
    html = (
        '<div id="inventory"></div>'
        '<button id="helpBtn">?</button>'
        '<div class="hotspot scene"></div>'
        '<script>function digHotspot(){} function showDialog(){}</script>'
    )
    toks = GameAgent._seed_structural_tokens(html)
    assert "inventory" in toks
    assert "helpbtn" in toks
    assert "hotspot" in toks
    assert "dighotspot" in toks
    # Generic layout noise is dropped.
    assert "div" not in toks
    assert "canvas" not in toks


def test_seed_structural_tokens_empty_html() -> None:
    assert GameAgent._seed_structural_tokens("") == []


# ---- Fix 3c: lean budget protects components on seed -----------------------

def test_protect_components_keeps_component_when_opening_fills_budget() -> None:
    a = _stub()
    budget = GameAgent._LEAN_MEMORY_COMBINED_BUDGET
    # Opening already (nearly) fills the budget; without protection the
    # components block would be dropped.
    opening = "O" * (budget - 100)
    components = "C" * 1500
    playbook = "P" * 1500

    # Default behavior drops components (and playbook).
    o0, c0, p0 = a._apply_lean_memory_budget(opening, components, playbook)
    assert c0 == ""

    # protect_components keeps the component, sacrifices playbook instead.
    o1, c1, p1 = a._apply_lean_memory_budget(
        opening, components, playbook, protect_components=True,
    )
    assert o1 == opening
    assert c1 == components
    assert p1 == ""


def test_protect_components_noop_when_no_component() -> None:
    a = _stub()
    o, c, p = a._apply_lean_memory_budget(
        "O" * 100, "", "P" * 100, protect_components=True,
    )
    assert c == ""
