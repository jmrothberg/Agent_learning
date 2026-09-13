"""Phase 3 guard: no game-title string literals in harness Python.

Title/class tokens (doom, wolfenstein, quake, tetris, battlezone, pinball)
belong in memory/*.jsonl recipe fields. Python must load them via memory.py
helpers — not branch on `"doom" in goal` style literals.

Allowlist: comments, docstrings, and thin migration shims that only *load*
from JSON (function names / recipe-field keys, not title vocabulary sets).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Standalone quoted literals only — "canvas-pinball" / doom_assets do not match.
_TITLE_LIT_RE = re.compile(
    r"""(?<![\w])(['"])(doom|wolfenstein|quake|tetris|battlezone|pinball)\1(?![\w])""",
    re.IGNORECASE,
)

_SCAN_FILES = (
    "tools.py",
    "prompts_v1.py",
    "modality.py",
    "memory.py",
    "agent.py",
    "agent_helpers.py",
    "agent_feedback.py",
    "agent_prompts.py",
    "agent_compaction.py",
    "agent_stream.py",
    "agent_probes.py",
    "agent_memory.py",
    "agent_gates.py",
    "agent_critic.py",
    "agent_assets.py",
)

# Lines that may still mention a title while only loading recipe JSON.
_ALLOW_SUBSTRINGS = (
    "disambiguation_signals",
    "viewpoint_entity",
    "auto_probe_requires_words",
    "detect_plan_nudge_keywords",
    "detect_recipe_intent_keywords",
    "recipe_viewpoint_entities",
    "recipe_disambiguation_signals",
    "collect_recipe_ensure_ids",
    "visual_recipe_get",
    "pinball-table",  # plan_nudge id, not a title branch
    "canvas-pinball",  # recipe id
    "title-literal-allow:",  # explicit escape hatch in a comment
)


def _code_lines_without_docstrings(text: str) -> list[tuple[int, str]]:
    """Yield (lineno, line) for non-docstring, non-#comment-only code lines."""
    out: list[tuple[int, str]] = []
    in_doc = False
    doc_delim = ""
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if not in_doc:
            if stripped.startswith("#"):
                continue
            # Opening docstring (module/func) — possibly same-line close.
            for delim in ('"""', "'''"):
                if delim in stripped:
                    # Count delimiters on this line.
                    count = stripped.count(delim)
                    if count == 1:
                        in_doc = True
                        doc_delim = delim
                    # count >= 2 ⇒ opened and closed on one line → skip line
                    break
            else:
                out.append((i, line))
            continue
        # Inside docstring: look for closer.
        if doc_delim and doc_delim in stripped:
            in_doc = False
            doc_delim = ""
        # skip docstring lines entirely
    return out


@pytest.mark.parametrize("rel", _SCAN_FILES)
def test_no_game_title_string_literals(rel: str) -> None:
    path = _REPO / rel
    assert path.is_file(), f"missing scan target {rel}"
    text = path.read_text(encoding="utf-8")
    offenders: list[str] = []
    for i, line in _code_lines_without_docstrings(text):
        if not _TITLE_LIT_RE.search(line):
            continue
        if _is_allowed(line):
            continue
        offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not offenders, (
        "Game-title string literals must live in memory JSONL recipes "
        "(Phase 3). Offenders:\n  " + "\n  ".join(offenders)
    )
