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


# ---------------------------------------------------------------------------
# First-build parse-error salvage (DOOM3DF2 20260911_170915)
# ---------------------------------------------------------------------------

def _agent(tmp_path):
    info = BackendInfo(
        name="ollama", model="dummy:0",
        source="test", endpoint="http://127.0.0.1:0",
    )
    return GameAgent(
        backend=make_backend(info),
        out_path=tmp_path / "game.html",
        max_iters=1,
    )


def _complete_doc(*, parse_error: bool) -> str:
    """A real-sized, complete document. With `parse_error`, its ONLY defect
    is one bad line (DOOM3DF2 shape: a stray token + one unclosed paren, so
    the bracket probe trips and node confirms the SyntaxError)."""
    body = "\n".join(
        f"function fn{i}(a,b){{ var s=0; for(var k=0;k<a;k++){{ s+=k*b; }} return s; }}"
        for i in range(120)
    )
    bad = "var ceilMat = ({ map: per-pixel ceilTex };\n" if parse_error else ""
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>T</title></head><body>\n"
        "<canvas id=\"c\" width=\"640\" height=\"480\"></canvas>\n<script>\n"
        + body
        + "\nvar ceilTex = 1;\n" + bad
        + "requestAnimationFrame(function loop(){ requestAnimationFrame(loop); });\n"
        "</script>\n</body></html>\n"
    )


def _complete_doc_with_parse_error() -> str:
    return _complete_doc(parse_error=True)


def test_first_build_parse_error_is_saved_not_discarded(tmp_path):
    """DOOM3DF2: a complete 26 KB build was thrown away for one bad token and
    the retry regenerated everything (26 min at 5 tok/s). With no baseline,
    a parse-error-only document must be written so the next turn patches."""
    agent = _agent(tmp_path)
    assert not agent._current_file
    doc = _complete_doc_with_parse_error()
    reply = "<html_file>\n" + doc + "</html_file>\n<notes>first build</notes>"
    new_html, msg = asyncio.run(agent._materialize(reply))
    assert new_html is not None, msg
    assert "ceilTex" in new_html
    assert "parse error" in msg.lower()
    # The model is coached to patch, not re-emit.
    assert any("<patch>" in c for c in agent._pending_coaching)


def test_parse_error_salvage_only_on_first_build(tmp_path):
    """Once a baseline exists the broken rewrite stays rejected (the
    stop-losing-to-one-shot rule is unchanged)."""
    agent = _agent(tmp_path)
    agent._current_file = _complete_doc(parse_error=False)  # healthy baseline
    reply = "<html_file>\n" + _complete_doc_with_parse_error() + "</html_file>"
    new_html, msg = asyncio.run(agent._materialize(reply))
    assert new_html is None
    assert "rejected" in msg.lower()


def test_parse_error_salvage_refuses_truncated_or_prose_docs():
    """Only JS parse/bracket reasons on a complete doc qualify; leading
    prose, tiny bodies and truncated files still need a re-emit."""
    from agent_helpers import _first_build_parse_error_salvageable as ok

    doc = _complete_doc_with_parse_error()
    assert ok(doc, "inline <script> has a JavaScript syntax error: x") is True
    assert ok(doc, "unbalanced () brackets in <script>: extra opening ( by 3") is True
    assert ok(doc, "file does not start with a HTML document") is False
    assert ok(doc, "elision marker found in source: '...}'") is False
    assert ok("<!DOCTYPE html><html><script>x(</script></html>", "unbalanced ()") is False
    truncated = doc.split("</script>")[0]  # no closing tags at all
    assert ok(truncated, "unbalanced () brackets") is False
