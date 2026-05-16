"""End-to-end-ish tests for the harness changes motivated by the two
Donkey Kong traces (2026-05-16):

  - 20260516_124628 — concatenated-drafts + iter-1 probes-only +
    dotted elision marker. Iters 4/5 were full <html_file> rewrites.
  - 20260516_142445 — dead-state-reset block triggered a token-
    repetition loop; ~16 `p.onGirder = false;` repeats; partial
    output left an unclosed <html_file>.

Each test exercises ONE of the harness changes and asserts the
coaching / detector path responds the way the plan describes. Pure
function calls — no model, no Chromium, no GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import GameAgent  # noqa: E402
from learner import detect_failure_shapes  # noqa: E402
from ollama_io import RepetitionDetector  # noqa: E402
from patches import FormatRejection  # noqa: E402


# ---------------------------------------------------------------------------
# A1 — adjacency trigger surfaces stall_reason + loop_line
# ---------------------------------------------------------------------------


def test_adjacency_fires_with_loop_line_captured():
    """Donkey-kong 20260516_142445 shape: 4 identical lines abort the
    stream. The detector must capture WHICH line was repeated so the
    recovery coach can name it back to the model."""
    det = RepetitionDetector()
    fired = False
    for _ in range(10):
        if det.feed("p.onGirder = false;\n"):
            fired = True
            break
    assert fired
    assert det.stall_reason == "adjacent_line_spam"
    assert det.loop_line == "p.onGirder = false;"


def test_adjacency_distinct_lines_do_not_fire():
    """Negative: 4 distinct lines must NOT trip the adjacency check."""
    det = RepetitionDetector()
    lines = ["a();", "b();", "c();", "d();"]
    for ln in lines:
        assert not det.feed(ln + "\n")
    assert det.stall_reason is None


# ---------------------------------------------------------------------------
# B1 — recovery coaching when guard fires inside an unclosed <html_file>
# ---------------------------------------------------------------------------


def test_unclosed_html_post_loop_coaches_with_repeated_line():
    """When prior_stream_looped=True AND rejection.kind=unclosed_html_file,
    the coach names the loop shape AND the repeated line so the model
    can avoid re-emitting the same broken branch."""
    rejection = FormatRejection(
        kind="unclosed_html_file",
        hint="missing </html_file>",
        detail="(detail body)",
    )
    text, reset = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=False,
        consecutive_plan_only=0,
        rejection=rejection,
        format_stuck_streak=1,
        prior_stream_looped=True,
        prior_loop_kind="adjacent_line_spam",
        prior_loop_line="p.onGirder = false;",
    )
    low = text.lower()
    assert "token-repetition loop" in low
    assert "no closing tag" in low
    assert "p.ongirder = false" in low      # specific line surfaces
    assert "n times in a row" in low        # shape labelled
    # The coach must NOT just say "try again" — it must offer a
    # divergent path (question OR smaller html_file).
    assert "<question>" in low
    assert "smaller" in low or "omits" in low
    assert reset is False


def test_unclosed_html_without_prior_loop_uses_generic_path():
    """The new coach only fires when both conditions hold. Without a
    prior loop signal, the generic rejection-detail path runs."""
    rejection = FormatRejection(
        kind="unclosed_html_file",
        hint="missing </html_file>",
        detail="THE REJECTION DETAIL",
    )
    text, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=False,
        consecutive_plan_only=0,
        rejection=rejection,
        format_stuck_streak=1,
        prior_stream_looped=False,
    )
    assert "THE REJECTION DETAIL" in text
    assert "token-repetition loop" not in text.lower()


# ---------------------------------------------------------------------------
# B2 — probes-only reply coaching
# ---------------------------------------------------------------------------


def test_probes_only_reply_coaches_specifically():
    """Donkey-kong 20260516_124628 iter 1 emitted only `<probes>` re-
    statement. The coach should call that shape out by name."""
    text, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=False,
        consecutive_plan_only=0,
        rejection=None,
        format_stuck_streak=0,
        probes_only=True,
        media_only=False,
    )
    low = text.lower()
    assert "<probes>" in text
    assert "live in session state" in low
    assert "code-changing tag is required" in low


def test_media_only_reply_coaches_specifically():
    """Same path but for <assets> / <sounds>-only emissions."""
    text, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=False,
        consecutive_plan_only=0,
        rejection=None,
        format_stuck_streak=0,
        probes_only=False,
        media_only=True,
    )
    assert "<assets>" in text
    assert "<sounds>" in text


# ---------------------------------------------------------------------------
# B3 — probe-sanity lint
# ---------------------------------------------------------------------------


def test_lint_flags_tautological_probe_donkey_kong_mario_moves_shape():
    """Donkey-kong 20260516_142445 `mario_moves`:
        `(()=>{const x0=s.player.x; setTimeout(()=>{}, 100); return true;})()`
    The probe binds x0, never reads it again, and returns true. The
    lint must flag it."""
    probes = [{
        "name": "mario_moves",
        "expr": (
            "(()=>{const x0=s.player.x; setTimeout(()=>{}, 100); "
            "return true;})()"
        ),
    }]
    findings = GameAgent._lint_probes(probes)
    assert len(findings) == 1
    assert findings[0]["kind"] == "tautological_constant_return"
    assert "mario_moves" in findings[0]["message"]
    assert "`x0`" in findings[0]["message"]


def test_lint_does_not_flag_real_dynamic_probe():
    """Negative: a probe that actually uses its temp var should pass."""
    probes = [{
        "name": "mario_moves_real",
        "expr": (
            "(()=>{const x0=s.player.x; return new Promise(r => "
            "setTimeout(() => r(s.player.x !== x0), 500));})()"
        ),
    }]
    findings = GameAgent._lint_probes(probes)
    # The regex MAY match the `return` shape, but the dead-temp guard
    # (count occurrences of `x0`) requires the temp to be unused.
    # `x0` appears twice here (decl + comparison), so the lint passes.
    assert findings == [], findings


def test_lint_flags_probe_referencing_unassigned_property():
    """Donkey-kong 20260516_124628 `barrels_move` reads `b.x0` which is
    never assigned anywhere in the game body. After materialize, the
    lint must surface this."""
    probes = [{
        "name": "barrels_move",
        "expr": (
            "(()=>{const s=window.state; if(!s||!Array.isArray(s.barrels)"
            "||s.barrels.length===0) return false; const b=s.barrels[0]; "
            "if (typeof b.x0 === 'undefined') return false; "
            "const x0=b.x; return new Promise(r => setTimeout(() => "
            "r(b.x !== x0), 300));})()"
        ),
    }]
    # Game body that defines `x`, `y`, etc. but NEVER assigns `x0`.
    html = (
        "<html><body><canvas id='c'></canvas><script>"
        "const state = { barrels: [{ x: 10, y: 20 }] };"
        "window.state = state;"
        "</script></body></html>"
    )
    findings = GameAgent._probes_referencing_unassigned_props(probes, html)
    assert len(findings) == 1
    assert findings[0]["kind"] == "unassigned_property_read"
    assert "barrels_move" in findings[0]["message"]
    assert "`x0`" in findings[0]["message"]


def test_lint_skips_dom_properties_on_unassigned_check():
    """Negative: a probe that reads `canvas.width` from the DOM should
    not be flagged just because the game doesn't `canvas.width = X`.
    DOM properties are in the ignore set."""
    probes = [{
        "name": "canvas_visible",
        "expr": (
            "(()=>{const c=document.querySelector('canvas'); "
            "return c.width > 0 && c.height > 0;})()"
        ),
    }]
    html = (
        "<html><body><canvas id='c' width='400' height='300'></canvas>"
        "<script>const x = 1;</script></body></html>"
    )
    findings = GameAgent._probes_referencing_unassigned_props(probes, html)
    assert findings == []


# ---------------------------------------------------------------------------
# C1 — learner shape-detectors
# ---------------------------------------------------------------------------


def _write_trace(rows: list[dict]) -> Path:
    """Helper: write a tiny trace fixture and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def test_shape_detector_proposes_dead_state_block_bullet():
    """Trace with stream_done.looped + adjacent_line_spam on a flag
    assignment line → 'no-dead-state-reset-fallthrough' bullet."""
    trace = _write_trace([
        {"kind": "session_start", "session_id": "test", "goal": "x"},
        {
            "kind": "stream_done",
            "tokens": 100, "duration_s": 30.0,
            "stalled": True, "looped": True, "deliberated": False,
            "loop_kind": "adjacent_line_spam",
            "loop_line": "p.onGirder = false;",
        },
    ])
    try:
        bullets = detect_failure_shapes(trace)
        assert len(bullets) == 1
        assert bullets[0]["id"] == "no-dead-state-reset-fallthrough"
        assert "token-loop" in bullets[0]["tags"]
    finally:
        trace.unlink()


def test_shape_detector_proposes_concatenated_drafts_bullet():
    """Trace with a console error 'Identifier X has already been
    declared' → 'no-concatenated-drafts' bullet."""
    trace = _write_trace([
        {"kind": "session_start", "session_id": "test", "goal": "x"},
        {
            "kind": "event",
            "event": "test",
            "data": {
                "ok": False,
                "errors": [
                    "Uncaught SyntaxError: Identifier 'ctx' has already "
                    "been declared",
                ],
            },
        },
    ])
    try:
        bullets = detect_failure_shapes(trace)
        assert len(bullets) == 1
        assert bullets[0]["id"] == "no-concatenated-drafts"
    finally:
        trace.unlink()


def test_shape_detector_idempotent_across_repeated_signals():
    """Two `stream_done.looped=True` events in the same trace must
    produce ONE bullet, not two — the curator's dedup would handle
    it but we shouldn't even emit the dup."""
    trace = _write_trace([
        {"kind": "session_start", "session_id": "test", "goal": "x"},
        {
            "kind": "stream_done",
            "looped": True, "loop_kind": "adjacent_line_spam",
            "loop_line": "x = false;",
        },
        {
            "kind": "stream_done",
            "looped": True, "loop_kind": "adjacent_line_spam",
            "loop_line": "x = false;",
        },
    ])
    try:
        bullets = detect_failure_shapes(trace)
        assert len(bullets) == 1
    finally:
        trace.unlink()


def test_shape_detector_clean_trace_returns_nothing():
    """Negative: a clean trace with no loop / no dup-decl events
    proposes nothing."""
    trace = _write_trace([
        {"kind": "session_start", "session_id": "test", "goal": "x"},
        {
            "kind": "event", "event": "test",
            "data": {"ok": True, "errors": []},
        },
        {"kind": "event", "event": "done"},
    ])
    try:
        assert detect_failure_shapes(trace) == []
    finally:
        trace.unlink()
