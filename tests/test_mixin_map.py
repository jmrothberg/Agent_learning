"""AGENTS.md mixin map matches actual method locations."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent

# Methods named in AGENTS.md §1b method-groups table.
_EXPECTED = {
    "_build_fix_prompt": "agent_prompts",
    "_build_structured_summary": "agent_prompts",
    "_apply_undrawn_art_intent_gate": "agent_gates",
    "_apply_dropped_assets_pending_gate": "agent_gates",
    "_apply_scoped_check_to_report": "agent_feedback",
    "_parse_feedback_route_json": "agent_feedback",
    "_route_user_feedback_llm": "agent_feedback",
    "_prune_messages": "agent_compaction",
    "_maybe_reset_continuation_context": "agent_compaction",
    "_extract_html": "agent_stream",
    "_run_format_doctor": "agent_stream",
}


@pytest.mark.parametrize("method,prefix", list(_EXPECTED.items()))
def test_mixin_map_method_module(method: str, prefix: str) -> None:
    fn = getattr(GameAgent, method)
    mod = inspect.getmodule(fn)
    assert mod is not None
    assert mod.__name__ == prefix, f"{method} expected in {prefix}, got {mod.__name__}"
