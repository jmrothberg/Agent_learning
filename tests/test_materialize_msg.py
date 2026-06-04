"""Tests for the materialize message surfacing patch-failure detail.

The DK trace 20260513_153626 iter 4 emitted a malformed patch (extra
`=======` line between the REPLACE body and `>>>>>>> REPLACE`). The
existing `_has_embedded_marker` check in patches.py correctly
detected the stray delimiter, but the user-visible log said only
"all 1 patches failed to apply" — debugging required digging into
the trace. These tests pin the contract that the first per-patch
failure reason is included in the materialize message returned to
the caller (and surfaced into the agent's info-event log)."""

import asyncio

from agent import GameAgent
from backend import BackendInfo, make_backend


def test_materialize_surfaces_malformed_marker_reason(tmp_path):
    """A patch with an extra `=======` inside the REPLACE body should
    fail with the specific 'embedded SEARCH/REPLACE marker' reason,
    not the generic 'all 1 patches failed to apply'."""
    info = BackendInfo(
        name="ollama", model="dummy:0",
        source="test", endpoint="http://127.0.0.1:0",
    )
    agent = GameAgent(
        backend=make_backend(info),
        out_path=tmp_path / "game.html",
        max_iters=1,
    )
    # Plant a baseline so the patch path is taken.
    baseline = "ALPHA\nBETA\nGAMMA\n"
    agent._current_file = baseline

    # A patch with a stray `=======` that is NOT auto-recoverable: there is
    # REPLACE-body content (OMEGA) AFTER the stray divider, so repair_reply
    # can't safely collapse it (unlike the simpler "divider directly before
    # >>>>>>> REPLACE" shape, which is now repaired — see
    # test_visual_critic_failsafe.test_repair_collapses_doubled_divider_before_replace).
    # This still trips `_has_embedded_marker` and must surface the specific reason.
    reply = """<patch>
<<<<<<< SEARCH
ALPHA
=======
ZETA
=======
OMEGA
>>>>>>> REPLACE
</patch>"""
    new_html, msg = asyncio.run(agent._materialize(reply, dry_run=True))
    assert new_html is None
    assert "failed" in msg.lower()
    # The reason from _has_embedded_marker must be present so the user
    # log shows the actual problem.
    assert "embedded" in msg.lower() or "marker" in msg.lower(), (
        f"materialize_msg should name the malformed-marker reason, got: {msg!r}"
    )


def test_materialize_surfaces_search_not_found_reason(tmp_path):
    """A patch whose SEARCH block isn't in the file should fail with
    the 'SEARCH block not found' reason, not the generic message."""
    info = BackendInfo(
        name="ollama", model="dummy:0",
        source="test", endpoint="http://127.0.0.1:0",
    )
    agent = GameAgent(
        backend=make_backend(info),
        out_path=tmp_path / "game.html",
        max_iters=1,
    )
    agent._current_file = "ONE\nTWO\nTHREE\n"
    reply = """<patch>
<<<<<<< SEARCH
NONEXISTENT_LINE
=======
REPLACEMENT
>>>>>>> REPLACE
</patch>"""
    new_html, msg = asyncio.run(agent._materialize(reply, dry_run=True))
    assert new_html is None
    assert "not found" in msg.lower() or "search" in msg.lower(), (
        f"materialize_msg should name the search-miss reason, got: {msg!r}"
    )
