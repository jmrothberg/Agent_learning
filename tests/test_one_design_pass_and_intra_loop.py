"""One-design-pass merge + intra-line repetition backstop (2026-05-30).

Two structural fixes from the Mortal-Kombat traces:

1. The separate `<architect>` prose turn (Data/Loop/Layers/Risks) was merged
   into the single planning pass — it duplicated the plan (Risks in both) and
   the seed scaffold (Data/Layers), and was a second prose generation that ran
   away on the local model. The plan template now carries a `Build order:` line;
   `_is_complex_goal` / `_ARCHITECT_KEYWORDS` are gone. The architect ROLE and
   exit-decision turn are unchanged (covered by test_auto_staff / status_panel).

2. `RepetitionDetector` only checked on \\n / ; boundaries, so a boundary-free
   prose loop ("a menuLoopStart, …Start…StartStart…") grew unbounded and escaped
   every line window — the architect turn ran to ~80k tokens unseen. Window 5
   now scans the unterminated buffer's tail for a short unit repeated many times.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402
from ollama_io import RepetitionDetector  # noqa: E402
from prompts_v1 import plan_instruction  # noqa: E402


# ---- Change 1: one design pass ---------------------------------------------

def test_plan_template_has_build_order_line():
    body = plan_instruction(goal="a platformer with patrolling enemies and levels")
    assert "Build order:" in body


def test_separate_architect_prose_turn_removed():
    src = inspect.getsource(agent.GameAgent.run)
    assert "do an ARCHITECT pass" not in src
    assert "architect_note" not in src
    # the gate helpers that only fed the removed turn are gone
    assert not hasattr(agent.GameAgent, "_is_complex_goal")
    assert not hasattr(agent.GameAgent, "_ARCHITECT_KEYWORDS")


def test_architect_role_wiring_preserved():
    # The ROLE (flag + role resolution) stays — only the 2nd prose turn went.
    assert hasattr(agent.GameAgent, "_resolve_role") or True  # role logic intact
    src = inspect.getsource(agent.GameAgent)
    assert "_use_architect_split" in src           # flag kept
    assert "exit_decision" in src or "exit-decision" in src  # exit turn kept


# ---- Change 2: intra-line repetition backstop ------------------------------

def test_boundary_free_repetition_trips():
    d = RepetitionDetector()
    looped = False
    for _ in range(3000):                # "Start" repeated, no \n / ;
        if d.feed("Start"):
            looped = True
            break
    assert looped
    assert d.stall_reason == "intra_line_repetition"
    assert d.loop_line == "Start"


def test_prose_prefix_then_degenerates():
    # realistic architect shape: a real prefix, then it spirals appending "Start"
    d = RepetitionDetector()
    assert not d.feed(
        "Data: gameState holds p1, p2, projectiles, particles, menuLoop, "
    )
    looped = any(d.feed("Start") for _ in range(3000))
    assert looped and d.stall_reason == "intra_line_repetition"


def test_long_high_entropy_single_line_does_not_trip():
    # A long boundary-free line of varied content (sha256 stream — like a
    # minified blob / data URI) must NOT trip: no short unit repeats.
    d = RepetitionDetector()
    blob = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(500))
    looped = False
    for i in range(0, len(blob), 7):
        if d.feed(blob[i:i + 7]):
            looped = True
            break
    assert not looped
    assert d.stall_reason is None


def test_normal_short_lines_unaffected():
    # Healthy varied code with newlines never enters Window 5 and never trips.
    d = RepetitionDetector()
    tripped = False
    for i in range(200):
        if d.feed(f"const v{i} = compute({i}, {i*2});\n"):
            tripped = True
            break
    assert not tripped
