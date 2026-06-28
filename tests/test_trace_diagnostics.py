"""Phase 4 (4D) trace-diagnostics tests.

The persisted .jsonl is an LLM-ONLY artifact (humans never read it). These
tests pin three properties:

  1. Verbosity budget — high-frequency live-monitoring events
     (`stream_heartbeat`, `stream_progress`) are NOT persisted; `status_snapshot`
     (deduped, low-volume, feedback-queue observability) IS kept.
  2. Failure classification — `_classify_failure` buckets an iter into the layer
     that needs the fix (harness_bug / memory_gap / local_llm_limit / none).
  3. Digest — `render_run_summary` surfaces `failure_class` so one row answers
     "where does this fix go?".

Pure-function / no model / no browser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, render_run_summary  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    return a


def _kinds(a: GameAgent) -> list[str]:
    return [
        json.loads(line)["kind"]
        for line in a.trace_path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1. Verbosity budget
# ---------------------------------------------------------------------------


def test_live_monitoring_events_not_persisted(tmp_path):
    a = _agent(tmp_path)
    a._trace({"kind": "stream_heartbeat", "tokens": 5, "tok_per_s": 1.2})
    a._trace({"kind": "stream_progress", "stage": "prompt_eval", "current": 1})
    a._trace({"kind": "iter_summary", "iteration": 1, "ok": True})
    kinds = _kinds(a)
    # The two genuine spam events must NOT reach the LLM-facing .jsonl.
    assert "stream_heartbeat" not in kinds
    assert "stream_progress" not in kinds
    # Milestone/diagnostic events still persist.
    assert "iter_summary" in kinds


def test_status_snapshot_still_persisted(tmp_path):
    # status_snapshot is deduped/low-volume and is the only record of
    # feedback-queue timing — it must stay in the persisted trace.
    a = _agent(tmp_path)
    a._trace({"kind": "status_snapshot", "activity": "idle", "pending_feedback": 1})
    assert "status_snapshot" in _kinds(a)


def test_diagnostic_events_still_persisted(tmp_path):
    a = _agent(tmp_path)
    for k in ("slow_prefill", "patch_outcome", "feedback_router_decision",
              "coaching_suppressed_clean_pass", "no_usable_code"):
        a._trace({"kind": k})
    kinds = _kinds(a)
    for k in ("slow_prefill", "patch_outcome", "feedback_router_decision",
              "coaching_suppressed_clean_pass", "no_usable_code"):
        assert k in kinds


# ---------------------------------------------------------------------------
# 2. Failure classification (pure static method, precedence-ordered)
# ---------------------------------------------------------------------------


def _classify(**kw):
    base = dict(
        ok=False, materialized=True, stall_reason=None,
        coaching_suppressed=False, asset_reprompt_cleared=False,
        art_intent=False, undrawn_present=False,
    )
    base.update(kw)
    return GameAgent._classify_failure(**base)


def test_class_harness_bug_coaching_suppressed():
    cls, reason = _classify(
        materialized=False, stall_reason="deliberation_loop",
        coaching_suppressed=True,
    )
    assert cls == "harness_bug"
    assert "coaching" in reason


def test_class_harness_bug_asset_reprompt_cleared():
    cls, reason = _classify(
        materialized=False, stall_reason="repetition_loop",
        asset_reprompt_cleared=True,
    )
    assert cls == "harness_bug"
    assert "art request" in reason


def test_class_harness_bug_precedence_over_stall():
    # A harness contradiction masks the model-side stall (most actionable).
    cls, _ = _classify(
        materialized=False, stall_reason="repetition_loop",
        coaching_suppressed=True, asset_reprompt_cleared=True,
    )
    assert cls == "harness_bug"


def test_class_memory_gap_undrawn_on_art_intent():
    cls, reason = _classify(
        ok=False, materialized=True, art_intent=True, undrawn_present=True,
    )
    assert cls == "memory_gap"
    assert "undrawn" in reason


def test_class_local_llm_limit_on_stall():
    cls, reason = _classify(materialized=False, stall_reason="repetition_loop")
    assert cls == "local_llm_limit"
    assert "repetition_loop" in reason


def test_class_none_on_clean_iter():
    cls, reason = _classify(ok=True, materialized=True)
    assert cls == "none"
    assert reason == ""


def test_class_memory_gap_only_when_undrawn_and_art():
    # Art intent but assets ARE drawn -> not a memory gap.
    cls, _ = _classify(ok=True, materialized=True, art_intent=True,
                       undrawn_present=False)
    assert cls == "none"


# --- Exhaustive input matrix (golden trace build-a-dragon-s-lair-laserdis_*) --
# Proves generality of the `not ok` guard over the whole input space instead of
# relying on the single real-world trace we have. _classify_failure is a pure
# function of a handful of flags, so the matrix is small and decisive.


def test_class_ok_clean_iter_always_none_matrix():
    """A CLEAN iter (ok=True) with NO stall and NO harness-bug flag must always
    be `none`, for every art_intent x undrawn combination. This is the iter-2
    golden case (ok=True, art_intent=True, undrawn_present=True) plus every
    other clean art / non-art game (Zelda, chess, tower-defense, novel art)."""
    for art_intent in (True, False):
        for undrawn_present in (True, False):
            cls, reason = _classify(
                ok=True, materialized=True,
                art_intent=art_intent, undrawn_present=undrawn_present,
                stall_reason=None,
                coaching_suppressed=False, asset_reprompt_cleared=False,
            )
            assert cls == "none", (art_intent, undrawn_present, cls)
            assert reason == ""


def test_class_ok_with_stall_is_local_llm_limit():
    """A clean-result iter that nonetheless stalled mid-stream still surfaces
    the stall (local_llm_limit) — the `not ok` guard only suppresses the
    memory_gap mislabel, it does NOT swallow a genuine stall signal."""
    for art_intent in (True, False):
        for undrawn_present in (True, False):
            cls, reason = _classify(
                ok=True, materialized=True,
                art_intent=art_intent, undrawn_present=undrawn_present,
                stall_reason="repetition_loop",
            )
            assert cls == "local_llm_limit"
            assert "repetition_loop" in reason


def test_class_failing_art_still_memory_gap():
    """Failing art iter with undrawn assets keeps memory_gap triage (iter-1
    golden case) — the guard must not erase failing-iter diagnostics."""
    cls, reason = _classify(
        ok=False, materialized=True, art_intent=True, undrawn_present=True,
    )
    assert cls == "memory_gap"
    assert "undrawn" in reason


def test_class_harness_bug_precedence_even_when_ok():
    """A genuine harness contradiction (coaching suppressed on a clean prior
    iter / standing art request cleared) is worth surfacing regardless of ok —
    harness_bug precedence must win even on ok=True."""
    cls, _ = _classify(ok=True, materialized=True, coaching_suppressed=True)
    assert cls == "harness_bug"
    cls2, _ = _classify(ok=True, materialized=True, asset_reprompt_cleared=True)
    assert cls2 == "harness_bug"


# ---------------------------------------------------------------------------
# 3. Digest surfaces failure_class
# ---------------------------------------------------------------------------


def test_image_skipped_not_persisted(tmp_path):
    a = _agent(tmp_path)
    a._trace({"kind": "image_skipped", "name": "enemy_goblin", "reason": "cached"})
    a._trace({"kind": "iter_summary", "iteration": 1, "ok": True})
    kinds = _kinds(a)
    assert "image_skipped" not in kinds
    assert "iter_summary" in kinds


def test_synthetic_report_no_browser():
    report = GameAgent._synthetic_report_no_browser({"warnings": ["x"]})
    assert report["test_skipped"] == "no_browser"
    assert report["ok"] is True
    assert report["warnings"] == ["x"]


def test_render_digest_surfaces_test_skipped():
    records = [
        {"kind": "session_start", "goal": "bigger towers", "model_name": "stub"},
        {"kind": "iter_summary", "iteration": 1, "ok": True,
         "test_skipped": "no_browser", "probes_passed": 0, "probes_total": 0,
         "failure_class": "none"},
    ]
    text = render_run_summary(records, artifact_id="td__run_x")
    assert "test_skipped:no_browser" in text


def test_render_digest_surfaces_failure_class():
    records = [
        {"kind": "session_start", "goal": "tower defense", "model_name": "stub"},
        {"kind": "code_snapshot", "iteration": 1, "size": 1000,
         "materialize": "full"},
        {"kind": "iter_summary", "iteration": 1, "ok": False,
         "probes_passed": 5, "probes_total": 7, "patch_applied": "3/3",
         "router_intent": "generate_new_assets", "tok_per_s": 14.2,
         "failure_class": "memory_gap", "soft_warnings": []},
        {"kind": "no_usable_code", "failure_class": "harness_bug",
         "identical_repeat": False},
    ]
    text = render_run_summary(records, artifact_id="td__run_x")
    # Header has the new diagnostic columns.
    assert "| class |" in text
    assert "patch" in text and "router" in text
    # The iter row carries the fix-layer bucket + correlation fields.
    assert "memory_gap" in text
    assert "generate_new_assets" in text
    assert "3/3" in text
    # The no-code turn is tagged with its bucket.
    assert "harness_bug" in text
