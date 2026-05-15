"""Tests for the widened probe re-parse gate.

DK trace 20260514_104131 evidence: when the user runs /seed against a
known game, the model writes Phase-A probes WITHOUT seeing the file.
On that session, 4/5 Phase-A probes referenced state names that don't
exist in the seed (`state.grid`, `state.player.onLadder`, `state.reset`,
`#instructions`). In iter 1 the model re-emitted 5 corrected probes (3
dynamic, matching the actual file shape) — but the harness's
coverage-gap-only gate dropped them. This file pins the new gate
shape: re-parse fires when ANY of:
  (A) Phase-A coverage gaps exist
  (B) The prior iter's report had probe failures
  (C) Seed session, very first iter
…and the new probe list is at least as large as the current one.

The gate decision is a tiny pure function we can unit-test without
spinning up the whole agent loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _decide_allow_reparse(
    *,
    planning_coverage_gaps: list[str],
    prev_probe_failures: int,
    seed_iter1: bool,
) -> bool:
    """Mirror of the gate added to agent.run()'s coverage-gap probe
    re-parse branch."""
    return (
        bool(planning_coverage_gaps)
        or prev_probe_failures > 0
        or seed_iter1
    )


# ---------------------------------------------------------------------------
# Negative cases — gate stays closed
# ---------------------------------------------------------------------------


def test_no_reparse_when_nothing_is_failing_and_no_gaps():
    """Healthy session, no gaps, no prior failures, not a seed iter
    1 — the gate stays shut and probes remain immutable."""
    assert _decide_allow_reparse(
        planning_coverage_gaps=[],
        prev_probe_failures=0,
        seed_iter1=False,
    ) is False


# ---------------------------------------------------------------------------
# Positive cases — gate opens
# ---------------------------------------------------------------------------


def test_reparse_when_phase_a_coverage_gap_exists():
    """Existing behavior (original gate) — preserved."""
    assert _decide_allow_reparse(
        planning_coverage_gaps=["restart works"],
        prev_probe_failures=0,
        seed_iter1=False,
    ) is True


def test_reparse_when_prior_probes_failed():
    """DK 20260514 pin: the model's Phase-A probes evaluated falsy on
    iter 1 because they referenced names that don't exist; the gate
    must open so a corrected probe set can be adopted."""
    assert _decide_allow_reparse(
        planning_coverage_gaps=[],
        prev_probe_failures=4,
        seed_iter1=False,
    ) is True


def test_reparse_on_seed_iter_1():
    """The model just saw the seed file for the first time — invite
    it to correct any blind Phase-A probes."""
    assert _decide_allow_reparse(
        planning_coverage_gaps=[],
        prev_probe_failures=0,
        seed_iter1=True,
    ) is True


def test_reparse_when_multiple_triggers_fire():
    """Multiple triggers is just OR — no special behavior, but pin
    it so a refactor doesn't accidentally narrow the gate."""
    assert _decide_allow_reparse(
        planning_coverage_gaps=["some gap"],
        prev_probe_failures=2,
        seed_iter1=True,
    ) is True


# ---------------------------------------------------------------------------
# Count-defensive check
# ---------------------------------------------------------------------------


def test_count_check_pins_minimum_probe_set_size():
    """The agent's re-parse path requires the new probe list to be at
    least as large as the current set, preventing a model from
    shrinking the surface to mask regressions."""
    # Pure-function pin: the assertion is `len(new) >= len(old)`.
    current = [{"name": "a", "expr": "true"},
               {"name": "b", "expr": "true"}]
    # Strictly smaller — should be rejected.
    smaller = [{"name": "a", "expr": "true"}]
    assert (len(smaller) >= len(current)) is False
    # Same size — accepted.
    same = [{"name": "x", "expr": "true"}, {"name": "y", "expr": "true"}]
    assert (len(same) >= len(current)) is True
    # Larger — accepted.
    larger = [{"name": "a", "expr": "true"},
              {"name": "b", "expr": "true"},
              {"name": "c", "expr": "true"}]
    assert (len(larger) >= len(current)) is True


# ---------------------------------------------------------------------------
# DK-trace pin — the literal Phase-A probes that should have been
# replaced but weren't under the old gate
# ---------------------------------------------------------------------------


def test_dk_trace_phase_a_probes_would_invite_reparse():
    """The literal probes the model wrote at turn 02 of
    donkey-kong-game-animated-donk_20260514_104131 referenced
    `state.grid`, `state.player.onLadder`, `state.reset`,
    `#instructions` — none in the seed. After iter 1 ran with these
    probes, the harness would report 4 probe failures. The new gate
    fires on `prev_probe_failures > 0`, opening the door for the
    model's iter-1 corrected probe block."""
    # Simulated iter-1 report: 4 of 5 probes failed.
    assert _decide_allow_reparse(
        planning_coverage_gaps=[],     # gate (A) closed
        prev_probe_failures=4,          # gate (B) open — this is the fix
        seed_iter1=False,               # this is iter 2 evaluating the iter-1 reply
    ) is True


# ---------------------------------------------------------------------------
# Seed prompt addendum
# ---------------------------------------------------------------------------


def test_seed_build_instruction_now_invites_probe_correction():
    """The seed prompt must explicitly tell the model that Phase-A
    probes may reference names that don't exist in the file, and
    invite a corrected <probes> block."""
    from prompts_v1 import seed_build_instruction

    p = seed_build_instruction(
        "<html><script>const state = { player: { x: 0 } };</script></html>",
        "/some/path/x.best.html",
    )
    assert "Phase A" in p
    assert "<probes>" in p
    # Specifically calls out the blind-authoring problem.
    assert "WITHOUT seeing this file" in p or "without seeing" in p.lower()


def test_seed_build_instruction_preserves_existing_anchors():
    """The new sentence must not displace the existing instructions
    (path, patch preference, full file inline)."""
    from prompts_v1 import seed_build_instruction

    p = seed_build_instruction("<html>x</html>", "/p.html")
    assert "SEED FILE: /p.html" in p
    assert "PREFER one or more <patch>" in p
    assert "EXISTING FILE:" in p
    assert "<html>x</html>" in p
