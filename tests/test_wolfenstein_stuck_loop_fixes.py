"""Tests for the universal harness fixes derived from the Wolfenstein
2026-05-24 stuck-loop trace (`make-a-browser-playable-wolfen_20260524_205821`).

That trace burned ~3.5 wall-clock hours across 3 restart attempts and
shipped nothing. Root causes were three orthogonal harness gaps. Each
gate below keys on observable signals already in the trace stream and
fires regardless of genre / goal text.

Gates covered:

  1. `_extract_html_inner` variant 6 — bare `<html>...</html>` document
     without `<!DOCTYPE>` is salvaged with a synthetic doctype line.
  2. `_reply_fingerprint` + identical-reply branch in
     `_no_usable_code_fallback` — when the same rejected reply lands
     twice in a row, the fallback returns a scope-reduction escalation
     instead of the generic "I could not find <patch> or <html_file>".
  3. `fix_instruction(..., context_pressure=True)` — when the prior
     stream's prompt_tokens hit >=85% of num_ctx, the fix prompt omits
     the inlined CURRENT FILE block and demands a minimal patch.
  4. `scope_reduction_instruction` — fires when iter <= 2 ships a file
     with RAF dead AND input dead. The detector + branch are wired in
     agent.py; this test exercises the prompt builder directly.
  5. `_attempt_failure_signature` + `plan_instruction(force_minimal_
     first_build=True)` — when two consecutive restart attempts hit
     the same failure shape, the next attempt's planning prompt asks
     for a minimal first build.
  6. New playbook bullets retrieve above the 0.02 Jaccard noise floor
     on a synthetic ambitious goal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from patches import classify_format_failure  # noqa: E402
from prompts_v1 import (  # noqa: E402
    fix_instruction,
    plan_instruction,
    scope_reduction_instruction,
)


# ---------------------------------------------------------------------------
# Gate 1: bare <html>...</html> salvage extractor
# ---------------------------------------------------------------------------


def test_variant6_extracts_bare_html_document_without_doctype():
    """A reply containing <html>...</html> with no <!DOCTYPE> and
    enough body should extract with a synthetic doctype prepended.
    """
    body_padding = "x" * 300  # bulk above the 200-byte threshold
    reply = (
        "Here is your game:\n\n"
        "<html><head><title>Game</title></head><body>\n"
        f"<canvas id=c></canvas><script>/*{body_padding}*/\n"
        "(function(){var c=document.getElementById('c');var ctx=c.getContext('2d');\n"
        "function frame(){ctx.fillRect(10,10,100,100);requestAnimationFrame(frame);}\n"
        "requestAnimationFrame(frame);window.gameState={score:0};})();\n"
        "</script></body></html>\n\n"
        "Hope this works!"
    )
    out = GameAgent._extract_html(reply)
    assert out is not None
    assert out.startswith("<!DOCTYPE html>")
    assert "<html" in out
    assert "</html>" in out


def test_variant6_rejects_tiny_html_probe_expression():
    """A small inline `<html></html>` (e.g. inside prose or a probe
    expression) is below the 200-byte threshold and should NOT be
    salvaged — otherwise the harness would write garbage to disk.
    """
    reply = "Check this returns true: <html></html> here."
    assert GameAgent._extract_html(reply) is None


def test_variant1_still_wins_when_html_file_wrapper_present():
    """The new variant 6 must not regress the canonical extraction
    path — when <html_file>...</html_file> is present, variants 1-5
    still take precedence.
    """
    reply = (
        "<html_file><!DOCTYPE html><html><body>x</body></html></html_file>"
    )
    out = GameAgent._extract_html(reply)
    assert out is not None
    assert "<!DOCTYPE" in out


def test_variant5_doctype_still_handled():
    """Existing variant-5 path with <!DOCTYPE>...</html> in prose must
    keep working — variant 6 should only fire when variant 5 doesn't.
    """
    reply = (
        "Here you go: <!DOCTYPE html>\n"
        "<html><body>" + ("x" * 300) + "</body></html>\nthanks"
    )
    out = GameAgent._extract_html(reply)
    assert out is not None
    assert out.startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
# Gate 2: identical-reply loop breaker
# ---------------------------------------------------------------------------


def test_reply_fingerprint_stable_for_identical_text():
    """Identical replies fingerprint the same; different lengths
    fingerprint differently (length bucket guard).
    """
    a = "hello world " * 50
    b = "hello world " * 50
    assert GameAgent._reply_fingerprint(a) == GameAgent._reply_fingerprint(b)
    c = "hello world " * 100  # different length bucket
    assert GameAgent._reply_fingerprint(a) != GameAgent._reply_fingerprint(c)


def test_no_usable_code_fallback_identical_repeat_branch():
    """When `identical_repeat=True`, the fallback returns the scope-
    reduction escalation regardless of other flags.
    """
    msg, reset = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        identical_repeat=True,
    )
    assert "IDENTICAL-REPLY LOOP DETECTED" in msg
    assert reset is True


def test_no_usable_code_fallback_identical_repeat_overrides_silent_branch():
    """The identical-repeat escalation takes priority over the silent-
    stream recovery prompt so two identical streams that both surfaced
    as silent don't get the wrong coaching.
    """
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        prior_stream_silent=True,
        identical_repeat=True,
    )
    assert "IDENTICAL-REPLY LOOP DETECTED" in msg
    assert "SILENT STREAM RECOVERY" not in msg


def test_no_usable_code_fallback_no_repeat_uses_generic_branch():
    """Without `identical_repeat`, the generic fallback still fires."""
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=True,
        consecutive_plan_only=0,
        identical_repeat=False,
    )
    assert "IDENTICAL-REPLY" not in msg


# ---------------------------------------------------------------------------
# Gate 3: context-pressure detector — fix_instruction omits file
# ---------------------------------------------------------------------------


def test_fix_instruction_pressure_mode_omits_current_file():
    """When `context_pressure=True`, the inlined CURRENT FILE block is
    replaced by a brief "CONTEXT IS FULL" banner so the model has
    headroom for its reply. The pressure banner is allowed to mention
    "CURRENT FILE ON DISK" as part of the explanation — what must be
    absent is the actual file body.
    """
    file_body = "<!DOCTYPE html><body>UNIQUE_PRESSURE_MARKER_42</body>"
    out = fix_instruction(
        "REPORT", file_body, "",
        context_pressure=True,
    )
    assert "CONTEXT IS FULL" in out
    assert "UNIQUE_PRESSURE_MARKER_42" not in out
    assert "<!DOCTYPE" not in out


def test_fix_instruction_pressure_overrides_focused_slice():
    """Pressure mode should win even when a focused slice is provided
    — the whole point is to free context for the reply.
    """
    out = fix_instruction(
        "REPORT", "<!DOCTYPE html><body>X</body>", "",
        context_pressure=True,
        focused_slice="function foo(){ return 42; }",
    )
    assert "CONTEXT IS FULL" in out
    assert "function foo" not in out


def test_fix_instruction_normal_mode_inlines_file():
    """Backwards compatibility — without the flag, behavior is
    unchanged and the file is inlined.
    """
    out = fix_instruction(
        "REPORT", "<!DOCTYPE html><body>UNIQUE_BODY_42</body>", "",
    )
    assert "CURRENT FILE ON DISK" in out
    assert "UNIQUE_BODY_42" in out
    assert "CONTEXT IS FULL" not in out


# ---------------------------------------------------------------------------
# Gate 4: dead-first-build scope-reduction prompt
# ---------------------------------------------------------------------------


def test_scope_reduction_instruction_contains_recovery_directives():
    """The scope-reduction prompt names the dead-first-build cause and
    demands a smaller intentionally-minimal rewrite (not a patch).
    """
    out = scope_reduction_instruction("--TEST REPORT--")
    assert "--TEST REPORT--" in out
    assert "DEAD-FIRST-BUILD DETECTED" in out
    assert "INTENTIONALLY SMALLER" in out
    assert "<html_file>" in out


def test_scope_reduction_instruction_mentions_dual_state_exposure():
    """The recovery prompt should remind the model to wire RAF + the
    dual state exposure (window.gameState + window.state) so the same
    failure doesn't repeat on the next first build.
    """
    out = scope_reduction_instruction("rep")
    assert "requestAnimationFrame" in out
    assert "window.gameState" in out
    assert "window.state" in out


# ---------------------------------------------------------------------------
# Gate 5: restart-signature-repeat → minimal first build
# ---------------------------------------------------------------------------


class _AttemptStub:
    """Minimal duck-typed stand-in for a GameAgent — exposes only the
    fields `_attempt_failure_signature` reads. Avoids spinning up a
    full agent (which would require a browser, backend, etc).
    """

    def __init__(
        self,
        *,
        dead: int = 0,
        identical: int = 0,
        fmt_iter1: int = 0,
        restart_threshold: float = 60.0,
    ) -> None:
        self._dead_first_build_recoveries = dead
        self._identical_reply_loops_this_attempt = identical
        self._format_rejections_iter1_this_attempt = fmt_iter1
        self.restart_score_threshold = restart_threshold


def test_build_fix_prompt_with_pressure_flag_does_not_NameError(tmp_path):
    """Regression for the 2026-05-25 13:22 crash:
    `_build_fix_prompt` referenced `iteration` in a `_trace` call but
    `iteration` is not a local var of that function. This test pins the
    pressure-flag path: build a fix prompt with the flag set, must
    return a string (not raise NameError).
    """
    from unittest.mock import MagicMock

    from agent import GameAgent

    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    # Substantive file body so the truncation-recovery branch (which
    # would short-circuit before our pressure-mitigation code) doesn't
    # fire first. Must include closing </script></body></html> so
    # _truncation_reason() returns None.
    a._current_file = (
        "<!DOCTYPE html>\n<html><head><title>X</title></head><body>\n"
        "<canvas id='c'></canvas><script>(function(){\n"
        + "var x = 1; // padding line\n" * 80
        + "console.log('init');\n"
        "})();</script></body></html>"
    )
    a._context_pressure_pending = True
    a._context_pressure_streak = 2
    a._last_tested_iter = 3
    fake_report = {
        "ok": False,
        "errors": ["x"],
        "console_errors": [],
        "page_errors": [],
        "probe_errors": [],
        "soft_warnings": [],
        "warnings": [],
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": {"ran": False, "any_change": None, "keys_tried": []},
        "frozen_canvas": False,
        "body_chars": 100,
        "body_sample": "",
        "probes": [],
    }
    # Must not raise NameError on `iteration`.
    prompt = a._build_fix_prompt(
        report=fake_report, regressed=False, partial_failed=[],
    )
    assert isinstance(prompt, str)
    assert "CONTEXT IS FULL" in prompt  # pressure-mitigation path fired
    # Flag is consumed (one-shot).
    assert a._context_pressure_pending is False


def test_build_fix_prompt_with_dead_first_build_flag_does_not_NameError(tmp_path):
    """Same regression class as above, for the dead-first-build branch:
    `_build_fix_prompt` referenced `iteration` in a `_trace` call inside
    the dead-first-build-recovery branch. This test pins the path so a
    future edit can't reintroduce the NameError.
    """
    from unittest.mock import MagicMock

    from agent import GameAgent

    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    a._current_file = (
        "<!DOCTYPE html>\n<html><head><title>X</title></head><body>\n"
        "<canvas id='c'></canvas><script>(function(){\n"
        + "var x = 1; // padding line\n" * 80
        + "console.log('init');\n"
        "})();</script></body></html>"
    )
    a._dead_first_build_pending = True
    a._dead_first_build_recoveries = 1
    a._last_tested_iter = 2
    fake_report = {
        "ok": False,
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "probe_errors": [],
        "soft_warnings": ["raf dead + input dead"],
        "warnings": [],
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": False},
        "input_listeners": {"total": 0, "document": 0, "window": 0, "body": 0, "other": 0},
        "input_test": {"ran": True, "any_change": False, "keys_tried": ["ArrowUp"]},
        "frozen_canvas": False,
        "body_chars": 100,
        "body_sample": "",
        "probes": [],
    }
    # Must not raise NameError on `iteration`.
    prompt = a._build_fix_prompt(
        report=fake_report, regressed=False, partial_failed=[],
    )
    assert isinstance(prompt, str)
    assert "DEAD-FIRST-BUILD DETECTED" in prompt
    # Flag is consumed.
    assert a._dead_first_build_pending is False


def test_attempt_failure_signature_priority_order():
    """Signature picks the FIRST flagged condition (dead_first_build >
    identical_reply_loop > format_rejection_iter1 > low_score > ok).
    """
    # dead_first_build wins over everything
    a = _AttemptStub(dead=2, identical=5, fmt_iter1=3)
    assert (
        GameAgent._attempt_failure_signature(a, score=20)
        == "dead_first_build"
    )

    # identical_reply_loop wins over format_rejection_iter1
    a = _AttemptStub(dead=0, identical=1, fmt_iter1=3)
    assert (
        GameAgent._attempt_failure_signature(a, score=20)
        == "identical_reply_loop"
    )

    # format_rejection_iter1 wins over low_score
    a = _AttemptStub(dead=0, identical=0, fmt_iter1=1)
    assert (
        GameAgent._attempt_failure_signature(a, score=10)
        == "format_rejection_iter1"
    )

    # No flags + low score → low_score
    a = _AttemptStub()
    assert (
        GameAgent._attempt_failure_signature(a, score=40) == "low_score"
    )

    # No flags + above threshold → ok
    a = _AttemptStub(restart_threshold=60.0)
    assert GameAgent._attempt_failure_signature(a, score=85) == "ok"


def test_plan_instruction_force_minimal_appends_nudge():
    """When `force_minimal_first_build=True`, the planning prompt
    appends a "RESTART RECOVERY — MINIMAL FIRST BUILD" nudge that
    scopes the first build down to 2-3 acceptance bullets.
    """
    out = plan_instruction(
        goal="build a complex 3D dungeon crawler with enemies, HUD, sound",
        force_minimal_first_build=True,
    )
    assert "RESTART RECOVERY" in out
    assert "MINIMAL FIRST BUILD" in out


def test_plan_instruction_default_omits_minimal_nudge():
    """Backwards compatibility — the nudge is opt-in. Default plan
    instruction omits it entirely.
    """
    out = plan_instruction(goal="build a snake game")
    assert "RESTART RECOVERY" not in out


# ---------------------------------------------------------------------------
# Gate 6: new playbook bullets retrieve above the noise floor
# ---------------------------------------------------------------------------


def test_new_playbook_bullets_retrieve_on_ambitious_goal():
    """The three new bullets (first-build-scope-discipline,
    patch-only-when-context-fills, dual-state-exposure-for-probes)
    must clear the 0.02 Jaccard noise floor on a representative
    ambitious complex-game goal. If they don't, they'll never reach
    the prompt and the fix is silently dead.
    """
    import os

    from memory import Playbook

    # Run from the project root so the playbook path resolves
    # regardless of where pytest was invoked from.
    project_root = Path(__file__).resolve().parent.parent
    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        pb = Playbook(base_root="memory")
        goal = (
            "Make a browser-playable Wolfenstein 3D-style first-person "
            "shooter raycast castle maze stone walls doors rooms enemies "
            "pickups health ammo score exit arrow mouse shoot weapon gun "
            "crosshair muzzle flash animation guards patrol alert chasing "
            "aiming shooting pain hit dying dead minimap"
        )
        hits = pb.retrieve(goal, k=15, stage="plan")
    finally:
        os.chdir(old_cwd)
    retrieved_ids = {h.bullet.id for h in hits}
    expected = {
        "first-build-scope-discipline",
        "patch-only-when-context-fills",
        "dual-state-exposure-for-probes",
    }
    missing = expected - retrieved_ids
    assert not missing, (
        f"new playbook bullets failed to clear noise floor: {missing}. "
        f"Retrieved top-{len(hits)}: {sorted(retrieved_ids)}"
    )


# ---------------------------------------------------------------------------
# Round 2 gates from the same trace
# ---------------------------------------------------------------------------


def test_compaction_marker_uses_html_comment_not_html_file_wrapper():
    """Round 2: the new compaction marker must use an HTML comment
    shape with no <html_file> / <patch> substrings ANYWHERE. Wolfenstein
    2026-05-24 trace: the old `<html_file>[omitted: N bytes]</html_file>`
    marker was shaped like valid output and a confused model parroted
    it back — sending the agent into an identical-reply loop.
    """
    import asyncio
    from unittest.mock import MagicMock

    backend = MagicMock()
    backend.info.name = "ollama"
    backend.info.model = "qwen3.6:27b-q8_0"
    a = GameAgent(
        backend=backend,
        out_path=Path("/tmp/round2_compaction_test.html"),
        browser=None,
        memory_root=str(Path(__file__).resolve().parent.parent / "memory"),
        playbook_top_k=0,
    )
    big_body = "a" * 5000
    summarized = a._summarize_content(
        f"prefix <html_file>{big_body}</html_file> tail"
    )
    assert "<html_file>" not in summarized
    assert "</html_file>" not in summarized
    assert "<patch>" not in summarized
    assert "HARNESS-OMITTED-PRIOR-HTML" in summarized
    # Marker must be an HTML comment so the model can't extract it
    assert summarized.count("<!--") >= 1
    assert summarized.count("-->") >= 1


def test_compaction_marker_echo_classified():
    """`classify_format_failure` must catch a reply that echoes either
    the legacy `[omitted: N bytes]` or new `HARNESS-OMITTED-PRIOR-*`
    marker. Naming the cause unblocks the model on the next turn.
    """
    # Legacy shape (what the Wolfenstein trace's confused model emitted)
    rej = classify_format_failure(
        "<html_file>[omitted: 25206 bytes of HTML; see snapshot]</html_file>"
    )
    assert rej is not None
    assert rej.kind == "compaction_marker_echoed"
    assert "compaction marker" in rej.detail.lower()

    # New comment shape — should also flag if echoed
    rej2 = classify_format_failure(
        "<diagnose>foo</diagnose>\n"
        "<!-- HARNESS-OMITTED-PRIOR-HTML: 12 bytes -->"
    )
    assert rej2 is not None
    assert rej2.kind == "compaction_marker_echoed"


def test_compaction_marker_detector_does_not_flag_normal_reply():
    """A clean <patch> or <html_file> reply must NOT trigger the new
    compaction-echo detector — false positives there would break
    every normal turn.
    """
    clean_patch = (
        "<patch>\n"
        "<<<<<<< SEARCH\nold line\n=======\nnew line\n>>>>>>> REPLACE\n"
        "</patch>"
    )
    assert classify_format_failure(clean_patch) is None

    clean_html = (
        "<html_file><!DOCTYPE html><html><body><script>"
        "console.log('hi');</script></body></html></html_file>"
    )
    assert classify_format_failure(clean_html) is None


def test_model_give_up_detector_fires_on_wolfenstein_notes():
    """`_notes_signal_give_up` must flag the exact Wolfenstein 2026-05-24
    turn [04] notes shape.
    """
    notes_text = (
        "(a) What works: ... "
        "(b) What's still broken: The <html_file> tag has been "
        "consistently failing to parse across multiple attempts, "
        "likely due to invisible formatting characters or XML "
        "parsing issues in the harness. "
        "(c) Workaround: The user can manually copy the complete "
        "HTML code from the last successful response into a file."
    )
    assert GameAgent._notes_signal_give_up(notes_text) is True


def test_model_give_up_detector_does_not_flag_progress_notes():
    """Normal progress notes (what was added, what's still TODO) must
    NOT trigger the give-up detector. False positives here would
    suppress legitimate <done/> shipping.
    """
    progress = (
        "Added mouse-look + WASD movement. Hooked up the start button. "
        "Still TODO: enemy AI alert state, sound wiring for the gun "
        "shot. The current build is playable end to end."
    )
    assert GameAgent._notes_signal_give_up(progress) is False
    assert GameAgent._notes_signal_give_up(None) is False
    assert GameAgent._notes_signal_give_up("") is False


# ---------------------------------------------------------------------------
# Round 3: DeliberationDetector must not kill long working streams
# ---------------------------------------------------------------------------


def test_deliberation_detector_latches_on_html_file_at_start():
    """The exact Wolfenstein 2026-05-25 trace [04] shape: stream starts
    with `<html_file>` and produces a 20 KB+ first build. Detector
    MUST latch on the opener and never abort, regardless of length.
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    # First chunk is the literal opener — should latch immediately.
    assert d.feed("<html_file>") is False
    # Now stream a 20 KB body in 200-char pieces; detector must stay
    # latched and never abort.
    body_chunk = "<!DOCTYPE html><body>" + ("x" * 180) + "</body>\n"
    for _ in range(100):  # ~20 KB total
        assert d.feed(body_chunk) is False


def test_deliberation_detector_latches_on_inline_html_file():
    """Pre-fix bug: opener regex required line-start. A reply that opens
    with prose then `<html_file>` on the same line failed to latch and
    aborted at the threshold. Now mid-line opener must latch.
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    assert d.feed("Sure, here's the file: <html_file>") is False
    # Stream more — should NOT abort
    for _ in range(50):
        assert d.feed("<!DOCTYPE html>" + ("x" * 200)) is False


def test_deliberation_detector_latches_on_doctype_without_wrapper():
    """A model that goes straight into `<!DOCTYPE html>` without
    wrapping in `<html_file>` (e.g. tried and forgot the wrapper)
    must still latch — the second latch family (code openers) was
    added for exactly this case.
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    assert d.feed("<!DOCTYPE html>\n<html><body>") is False
    for _ in range(50):
        assert d.feed("<script>function foo(){}</script>" + ("x" * 200)) is False


def test_deliberation_detector_latches_on_function_declaration():
    """`function foo()` is unambiguous code — detector should latch
    even if no HTML structure has appeared yet (e.g. model is writing
    a `<script>` body inline).
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    # First a bit of prose / scaffolding
    assert d.feed("Building the game now...\n") is False
    # Then a function declaration — should latch
    assert d.feed("function setup() { return 42; }\n") is False
    # Continue streaming — no abort
    for _ in range(50):
        assert d.feed("more code " * 50) is False


def test_deliberation_detector_still_aborts_on_pure_prose_deliberation():
    """The detector exists for a reason: a model that produces 6000+
    chars of pure prose / reasoning without ANY tag opener or code
    content must still abort. Drop this assertion and we lose the
    safety net entirely.
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    # 6100 chars of pure prose with no tag / no code
    prose_chunk = (
        "Let me think about this. First I need to consider the requirements. "
        "Actually, let me approach this differently. The user wants a game "
        "but I need to plan it carefully before writing any code. "
    )
    aborted = False
    total = 0
    for _ in range(40):
        if d.feed(prose_chunk):
            aborted = True
            break
        total += len(prose_chunk)
    assert aborted, f"expected abort, no abort after {total} chars"
    assert d.stall_reason == "deliberation_loop"


def test_deliberation_detector_respects_think_block_for_opener_literals():
    """Opener literals INSIDE `<think>` are reasoning prose, not real
    output. They must not latch (preserves the doom 20260512 trace
    protection: model mentioning `<html_file>` in CoT then producing
    zero real output).
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    # Inside <think>, even a literal <html_file> mention must not latch.
    assert d.feed("<think>\nI need to emit <html_file> later.\n") is False
    # Now stream more reasoning — still inside think — should eventually
    # abort because nothing real has been emitted.
    aborted = False
    for _ in range(50):
        if d.feed("Still thinking. " * 30):
            aborted = True
            break
    assert aborted, "expected abort inside unclosed <think>"


def test_deliberation_detector_latches_after_think_close():
    """After `</think>` closes, a fresh opener in the post-think buffer
    MUST latch. The think-only-detection protects against reasoning
    that names tags; after the model exits reasoning and starts real
    output, the latch should fire normally.
    """
    from ollama_io import DeliberationDetector

    d = DeliberationDetector()
    assert d.feed("<think>\nPlanning the game.\n</think>\n<html_file>") is False
    # Continue streaming the build — no abort
    for _ in range(50):
        assert d.feed("<!DOCTYPE html>" + ("x" * 200)) is False


def test_new_playbook_bullets_absent_on_simple_goal():
    """The new bullets target ambitious goals — they should NOT
    crowd out simpler retrievals on minimal goals (snake, tic-tac-toe).
    """
    import os

    from memory import Playbook

    project_root = Path(__file__).resolve().parent.parent
    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        pb = Playbook(base_root="memory")
        hits = pb.retrieve("snake game with arrow keys", k=5, stage="plan")
    finally:
        os.chdir(old_cwd)
    retrieved_ids = {h.bullet.id for h in hits}
    new_bullets = {
        "first-build-scope-discipline",
        "patch-only-when-context-fills",
        "dual-state-exposure-for-probes",
    }
    overlap = new_bullets & retrieved_ids
    assert not overlap, (
        f"new bullets retrieved on simple goal where they shouldn't: "
        f"{overlap}"
    )
