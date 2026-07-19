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

import asyncio
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import AgentEvent, GameAgent, render_run_summary  # noqa: E402
from backend import StreamResult  # noqa: E402


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


def test_oversized_noncanonical_trace_is_at_most_2kb(tmp_path):
    a = _agent(tmp_path)
    a._trace({
        "kind": "arbitrary_diagnostic",
        "iteration": 7,
        "failure_class": "harness_bug",
        "err": "identity-error",
        "huge": "界🙂\n" * 5000,
        "nested": [{"report": "R" * 5000} for _ in range(50)],
    })
    raw = a.trace_path.read_bytes().splitlines()[-1]
    row = json.loads(raw)
    assert len(raw) <= 2048
    assert row["kind"] == "arbitrary_diagnostic"
    assert row["iteration"] == 7
    assert row["failure_class"] == "harness_bug"
    assert row["err"] == "identity-error"
    assert row["_trace_compacted"] is True
    assert row["_trace_original_bytes"] > len(raw)


def test_canonical_turn_and_reply_preserve_exact_unicode_multiline(tmp_path):
    a = _agent(tmp_path)
    turn = {"role": "user", "content": "第一行🙂\nsecond line\r\n\t끝"}
    reply = "答え🙂\n```js\nconst café = '☕';\n```\r\n"
    a._trace({"kind": "stream_start", "turn_input": turn})
    a._trace({"kind": "assistant_reply", "iteration": 1, "reply": reply})
    rows = [
        json.loads(line)
        for line in a.trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("kind") in {"stream_start", "assistant_reply"}
    ]
    assert rows[0]["turn_input"] == turn
    assert rows[1]["reply"] == reply
    assert "_trace_compacted" not in rows[0]
    assert "_trace_compacted" not in rows[1]


def test_generic_event_has_bounded_preview_and_no_full_canonical_copy(tmp_path):
    a = _agent(tmp_path)
    full = "PLAN🙂\n" * 1000
    a._record(AgentEvent("plan", full, {
        "plan": full,
        "reply": full,
        "compact_flag": True,
    }))
    row = json.loads(a.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["kind"] == "event"
    assert row["text_len"] == len(full)
    assert len(row["text_preview"]) <= 320
    assert "plan" not in row["data"] and "reply" not in row["data"]
    assert row["data"]["plan_len"] == len(full)
    assert len(row["data"]["plan_preview"]) <= 160
    assert len(a.trace_path.read_bytes().splitlines()[-1]) <= 2048


def test_trace_slimming_reduces_representative_serialized_bytes(tmp_path):
    full_recipe = {
        "state": ["player", "waves", "towers"],
        "probes": ["window.__gameState exists"] * 80,
        "steps": "render and verify\n" * 300,
    }
    before_payload = {
        "ts": "2026-07-19T00:00:00Z",
        "kind": "opening_book_retrieved",
        "stage": "plan",
        "hits": [{
            "kind": "outline", "id": "outline-tower-defense",
            "tier": "root", "score": 0.91, "recipe": full_recipe,
        }],
    }
    before = len(json.dumps(
        before_payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    a = _agent(tmp_path)
    a._trace(before_payload)
    raw = a.trace_path.read_bytes().splitlines()[-1]
    after = len(raw)
    row = json.loads(raw)
    assert after < before
    assert after <= 2048
    assert row["hits"][0]["id"] == "outline-tower-defense"
    assert "recipe" not in row["hits"][0]


def test_system_prompt_trace_is_manifest_under_1kb(tmp_path):
    a = _agent(tmp_path)
    prompt = "SYSTEM🙂\n" * 5000
    a._trace({
        "kind": "system_prompt_built",
        "model_class": "small",
        "chars": len(prompt),
        "system_prompt": prompt,
    })
    raw = a.trace_path.read_bytes().splitlines()[-1]
    row = json.loads(raw)
    assert len(raw) <= 1024
    assert "system_prompt" not in row
    assert row["system_prompt_chars"] == len(prompt)
    assert len(row["system_prompt_sha256"]) == 64


def test_media_generation_trace_keeps_outcomes_not_prompt_prose(tmp_path):
    a = _agent(tmp_path)
    a._trace({
        "kind": "assets_generated",
        "requested": 2,
        "produced": 1,
        "names": ["hero"],
        "paths": {"hero": "/tmp/hero.png"},
        "failures": {"enemy": "generator unavailable"},
        "per_asset": [
            {"name": "hero", "prompt": "long repeated art prose", "cache_hit": True},
        ],
    })
    row = json.loads(a.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["paths"] == {"hero": "/tmp/hero.png"}
    assert row["failures"] == {"enemy": "generator unavailable"}
    assert row["per_asset"] == [{"name": "hero", "cache_hit": True}]
    assert "long repeated art prose" not in json.dumps(row, ensure_ascii=False)


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


def test_class_memory_gap_launch_playfield_over_undrawn():
    cls, reason = _classify(
        ok=False, materialized=True, art_intent=True, undrawn_present=True,
        launch_playfield_probe_failed=True,
    )
    assert cls == "memory_gap"
    assert "launch/playfield" in reason


def test_class_harness_bug_green_probes_only_soft_warnings():
    """T-1 (run_04): ok=False but every model probe passed, no page errors,
    only soft_warnings blocked -> harness false positive (PLAYER-STUCK /
    keyboard-HEURISTIC / board-input), formerly mislabeled `none`."""
    cls, reason = _classify(
        ok=False, materialized=True,
        probes_all_passed=True, has_page_errors=False, has_soft_warnings=True,
    )
    assert cls == "harness_bug"
    assert "all model probes passed" in reason


def test_class_harness_fp_not_fired_with_page_errors():
    # A real page error means the model build IS broken -> not a harness FP.
    cls, _ = _classify(
        ok=False, materialized=True,
        probes_all_passed=True, has_page_errors=True, has_soft_warnings=True,
    )
    assert cls != "harness_bug"


def test_class_harness_fp_yields_to_memory_gap_on_undrawn_art():
    # First-occurrence undrawn on an art build stays memory_gap even when
    # probes pass — the wiring gap is the more actionable signal there.
    cls, reason = _classify(
        ok=False, materialized=True, art_intent=True, undrawn_present=True,
        probes_all_passed=True, has_page_errors=False, has_soft_warnings=True,
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


def _load_enrich_trace():
    import importlib.util
    path = Path(__file__).parent.parent / "scripts" / "enrich_trace.py"
    spec = importlib.util.spec_from_file_location("_enrich_trace_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wasted_iters_counts_blocked_iter_identical_to_shipped():
    """T-4: an ok=False iter whose code sha equals the final shipped build is
    a wasted (harness-friction) iter; an ok=False iter with DIFFERENT code is
    a genuine fix iter and is NOT counted."""
    et = _load_enrich_trace()
    records = [
        {"kind": "code_snapshot", "iteration": 1, "html_sha256": "aaa"},
        {"kind": "iter_summary", "iteration": 1, "ok": False},
        {"kind": "code_snapshot", "iteration": 2, "html_sha256": "bbb"},
        {"kind": "iter_summary", "iteration": 2, "ok": False,
         "shipped_unchanged_after_block": True},
        {"kind": "code_snapshot", "iteration": 3, "html_sha256": "bbb"},
        {"kind": "iter_summary", "iteration": 3, "ok": True},
    ]
    n, iters = et._compute_wasted_iters(records)
    # iter 2 (sha == shipped 'bbb', ok=False) is wasted; iter 1 ('aaa') is not.
    assert n == 1 and iters == [2]


def test_retrieval_first_clean_join():
    """M-2: legacy traces without retrieved_ids fall back to playbook_retrieved."""
    et = _load_enrich_trace()
    records = [
        {"kind": "playbook_retrieved", "ids": ["b1", "b2"]},
        {"kind": "iter_summary", "iteration": 1, "ok": False},
        {"kind": "playbook_retrieved", "ids": ["b2", "b3"]},
        {"kind": "iter_summary", "iteration": 2, "ok": True},
    ]
    first_clean, retrieved = et._retrieval_first_clean(records)
    assert first_clean == 2
    assert retrieved == ["b1", "b2", "b3"]  # ordered-unique union


def test_retrieval_first_clean_prefers_per_iter_retrieved_ids():
    """Post-clean feedback retrieval must not be credited to first-clean."""
    et = _load_enrich_trace()
    records = [
        {"kind": "iter_summary", "iteration": 1, "ok": True, "retrieved_ids": ["b1"]},
        {"kind": "playbook_retrieved", "ids": ["b2"], "feedback_in_query": True},
        {"kind": "iter_summary", "iteration": 2, "ok": False, "retrieved_ids": ["b2"]},
    ]
    first_clean, credited = et._retrieval_first_clean(records)
    assert first_clean == 1
    assert credited == ["b1"]


def test_session_outcome_carries_materialized_and_crash_flags():
    """Offline credit eligibility needs code_materialized + backend_crashed."""
    src = GameAgent.class_inspect_source()
    assert '"code_materialized": code_materialized' in src
    assert '"backend_crashed": backend_crashed' in src


def test_iter_summary_carries_shipped_unchanged_and_probe_digest():
    """T-2/T-3: iter_summary records the shipped-unchanged false-positive
    marker and a bounded per-probe digest, both from data the agent holds."""
    src = GameAgent.run_loop_inspect_source()
    # T-2 boolean + the code-hash roll-forward that backs it.
    assert '"shipped_unchanged_after_block": _shipped_unchanged_after_block' in src
    assert "_cur_iter_code_sha" in src
    assert "self._prev_iter_code_sha = _cur_sha" in src
    # T-3 compact probe digest (name + ok + short expr), bounded.
    assert "_probe_digest = [" in src
    assert '"probes": _probe_digest' in src


def test_iter_summary_carries_retrieved_ids():
    """Scoreboard/credit: active playbook ids fold onto iter_summary."""
    src = GameAgent.run_loop_inspect_source()
    assert '"retrieved_ids": list(getattr(self, "_active_bullet_ids"' in src


def test_session_outcome_carries_numeric_score():
    """session_outcome.score uses _score_attempt so offline tools need no join."""
    src = GameAgent.class_inspect_source()
    assert '"kind": "session_outcome"' in src
    assert '"score": session_score' in src
    assert "session_score = float(self._score_attempt())" in src


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


def test_render_digest_surfaces_agent_crash():
    records = [
        {"kind": "session_start", "goal": "tower defense", "model_name": "stub"},
        {"kind": "agent_crash", "source": "run", "exc_type": "NameError",
         "err": "name 'Path' is not defined", "iteration": 2,
         "traceback": "  File \"agent_compaction.py\", line 240"},
    ]
    text = render_run_summary(records, artifact_id="td__run_x")
    assert "**AGENT CRASH**" in text
    assert "NameError" in text
    assert "Path" in text
    assert "agent_crash" in text


def test_one_screen_diagnostics_keep_core_trace_signals():
    et = _load_enrich_trace()
    records = [
        {"kind": "session_start", "goal": "tower defense", "model_name": "stub"},
        {"kind": "playbook_retrieved", "ids": ["tower-loop", "range-ring"]},
        {"kind": "iter_summary", "iteration": 1, "ok": False,
         "probes_passed": 4, "probes_total": 6, "patch_applied": "2/3",
         "failure_class": "memory_gap", "soft_warnings": ["collision blocker"],
         "retrieved_ids": ["tower-loop", "range-ring"]},
        {"kind": "agent_crash", "source": "run", "exc_type": "RuntimeError",
         "err": "edit crashed", "iteration": 2},
    ]
    digest = render_run_summary(records, artifact_id="td__run_x")
    assert "memory_gap" in digest
    assert "collision blocker" in digest
    assert "4/6" in digest
    assert "2/3" in digest
    assert "RuntimeError" in digest and "edit crashed" in digest
    first_clean, retrieved = et._retrieval_first_clean(records)
    assert first_clean is None
    assert retrieved == ["tower-loop", "range-ring"]
    assert "memory_gap ->" in et._format_edit_first(records)


def test_failing_iteration_trace_stage_contract(tmp_path):
    """A failed production loop iteration records stages in causal order."""
    class FakeBrowser:
        async def load_and_test(self, path, **kwargs):
            return {
                "ok": False,
                "page_errors": ["scripted browser failure"],
                "console_errors": [],
                "errors": [],
                "soft_warnings": [],
                "warnings": [],
                "title": "scripted failure",
                "canvas": None,
                "input_listeners": {"keydown": 1},
                "input_test": None,
                "frozen_canvas": None,
                "body_chars": 1,
                "body_sample": "",
                "logs": [],
                "probes": [],
                "screenshot": None,
                "screenshot_before": None,
                "screenshot_action": None,
            }

    agent = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=FakeBrowser(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
        playbook_writeback=False,  # Production-loop test must never credit tracked memory.
    )
    agent._goal = "scripted trace-stage contract"
    agent._criteria = ""
    agent._probes = []
    agent._messages = [{"role": "user", "content": "Build the scripted game."}]
    agent._use_autonomous_feedback = False
    agent._use_vlm_critique = False
    agent.set_auto_step_on_failure(False)  # Keep this unattended regression non-blocking.

    first_reply = """<html_file>
<!DOCTYPE html><html><body><canvas id="game" width="320" height="200"></canvas>
<script>
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
let x = 10;
addEventListener("keydown", e => { if (e.key === "ArrowRight") x += 2; });
function loop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillRect(x, 20, 20, 20);
  requestAnimationFrame(loop);
}
loop();
</script></body></html>
</html_file>"""
    second_reply = "\n".join([
        "<patch>",
        "<" * 7 + " SEARCH",
        "let x = 10;",
        "=" * 7,
        "let x = 12;",
        ">" * 7 + " REPLACE",
        "</patch>",
    ])
    replies = iter((first_reply, second_reply))

    async def scripted_stream_chat(*args, **kwargs):
        text = next(replies)
        return StreamResult(
            text=text, tokens=1, duration_s=0.01, stalled=False,
        )

    # Script below production _stream so stream_start remains production-wired.
    agent._backend.stream_chat = scripted_stream_chat

    async def drive_loop():
        async for _ in agent._run_build_iterate_loop(continuation=False):
            pass

    asyncio.run(drive_loop())
    records = [
        json.loads(line)
        for line in agent.trace_path.read_text().splitlines()
        if line.strip()
    ]

    # Normalize only contract stages: a traced AgentEvent("test") is persisted
    # as kind=event/event=test, and the next prompt is embedded in stream_start.
    normalized = []
    for record in records:
        kind = record.get("kind")
        if kind in {"stream_start", "assistant_reply", "code_snapshot", "iter_summary"}:
            normalized.append(kind)
        elif kind == "event" and record.get("event") == "test":
            normalized.append("test")

    expected = [
        "stream_start",
        "assistant_reply",
        "code_snapshot",
        "test",
        "iter_summary",
        "stream_start",
    ]
    assert normalized[:len(expected)] == expected
    second_stream = [r for r in records if r.get("kind") == "stream_start"][1]
    assert second_stream["turn_input"]["role"] == "user"
    assert "scripted browser failure" in second_stream["turn_input"]["content"]
