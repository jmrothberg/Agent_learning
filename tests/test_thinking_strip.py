"""Regression test for the reasoning-model `</think>` parse bug.

Failure mode (observed in
games/traces/game-of-space-invaders-with-gr_20260511_093225):

Qwen3.6-27B-mxfp8 emits its chain-of-thought first, which mentions
`` `<assets>` `` in markdown backticks as part of an instruction
checklist. The greedy non-greedy <assets>...</assets> regex then
matches from that FIRST occurrence in the CoT all the way through to
the REAL `</assets>` block lower down, capturing the thinking prose
as the body. The body isn't valid JSON, the parse returns [], and
the agent silently skips all asset generation even though the model
correctly emitted 13 assets + 10 sounds.

Fix: strip everything up to and including the last `</think>` before
any tag parser sees the reply. Applied in:
  - assets._strip_thinking → _extract_assets_body / _extract_sounds_body
  - agent._strip_thinking → all _extract_* static methods
  - patches.repair_reply → extract_patches
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _strip_thinking  # noqa: E402
from assets import _strip_thinking as _strip_thinking_assets, parse_assets_block  # noqa: E402
from sounds import parse_sounds_block  # noqa: E402
from patches import repair_reply  # noqa: E402


_REPLY_WITH_THINKING = """The user wants Space Invaders. I need to emit:
1. `<plan>` - design overview
2. `<criteria>` - acceptance bullets
3. `<probes>` - executable JS
4. `<assets>` - visual entities for the diffuser
5. `<sounds>` - audio events
</think>

<plan>arcade shooter</plan>
<criteria>moves, fires, scores</criteria>
<probes>[{"name":"canvas","expr":"!!document.querySelector('canvas')"}]</probes>
<assets>
[
  {"name": "player_ship", "prompt": "pixel-art cannon"},
  {"name": "alien", "prompt": "pixel-art crab alien"}
]
</assets>
<sounds>
[
  {"name": "fire", "prompt": "8-bit laser pew", "duration": 0.2}
]
</sounds>
"""


def test_strip_thinking_drops_cot_prelude():
    out = _strip_thinking(_REPLY_WITH_THINKING)
    # CoT must be gone.
    assert "I need to emit" not in out
    # The real tags must remain.
    assert "<plan>arcade shooter</plan>" in out


def test_strip_thinking_assets_mirror_matches():
    # The assets.py mirror must match agent.py's helper exactly so the
    # two parser families behave identically on the same input.
    assert _strip_thinking_assets(_REPLY_WITH_THINKING) == _strip_thinking(_REPLY_WITH_THINKING)


def test_strip_thinking_noop_when_no_close_tag():
    plain = "<plan>x</plan>\n<assets>[]</assets>"
    assert _strip_thinking(plain) == plain


def test_strip_thinking_uses_last_close_tag():
    multi = (
        "<think>first</think>\n"
        "intermediate\n"
        "<think>refining</think>\n"
        "<plan>final</plan>"
    )
    out = _strip_thinking(multi)
    assert out.strip() == "<plan>final</plan>"


def test_parse_assets_after_strip():
    a = parse_assets_block(_REPLY_WITH_THINKING)
    assert len(a) == 2
    names = {x["name"] for x in a}
    assert names == {"player_ship", "alien"}


def test_parse_sounds_after_strip():
    s = parse_sounds_block(_REPLY_WITH_THINKING)
    assert len(s) == 1
    assert s[0]["name"] == "fire"


def test_parse_probes_after_strip():
    probes = GameAgent._extract_probes(_REPLY_WITH_THINKING)
    assert len(probes) == 1
    assert probes[0]["name"] == "canvas"


def test_parse_criteria_after_strip():
    crit = GameAgent._extract_criteria(_REPLY_WITH_THINKING)
    assert crit == "moves, fires, scores"


def test_repair_reply_strips_thinking():
    # patches.repair_reply is the universal pre-parser for <patch>
    # extraction; it must also drop the CoT prelude so a CoT that
    # mentions `<patch>` in backticks doesn't blow up extract_patches.
    out = repair_reply(_REPLY_WITH_THINKING)
    assert "I need to emit" not in out
    assert "<plan>arcade shooter</plan>" in out
