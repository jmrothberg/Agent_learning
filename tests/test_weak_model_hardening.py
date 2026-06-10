"""Regression tests for the 2026-06-10 weak-model hardening pass.

Trace evidence: two runs of the same fight-game goal —
  - build-a-single-screen-2d-fight_20260610_140129 (DeepSeek-V4-Flash)
  - build-a-single-screen-2d-fight_20260610_151443 (Qwen3.6-27B)

Both models produced correct content that the harness discarded:
  1. tools.py UnboundLocalError crashed the browser test exactly when
     action keys worked (report["action_frames"] assigned pre-creation).
  2. The harness-crash message told the model to "simplify the page".
  3. Recovery prompts ORDERED a full <html_file> and then the
     baseline-exists gate rejected the compliant reply.
  4. The ASSET SANITY WARNING advisory was misclassified as a user art
     request, arming 3 turns of "ASSET GENERATION REQUIRED".
  5. An end-of-stream <patch> missing `>>>>>>> REPLACE` was dropped.
  6. A noise playbook bullet (sim 0.0319) was injected every fix turn.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    GameAgent,
    _HARNESS_ADVISORY_SENTINEL,
    _feedback_is_art_change,
    render_run_summary,
)
from memory import Bullet, Playbook  # noqa: E402
from patches import extract_patches  # noqa: E402
import prompts_v1  # noqa: E402

ROOT = Path(__file__).parent.parent


def _make_agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _real_html(marker: str = "v2") -> str:
    # Distinct (non-repeating) statements so the fixture clears BOTH the
    # 2KB `_is_degenerate_baseline` size floor and `_detect_block_bloat`.
    filler = "\n".join(
        f"  const pad_{marker}_{i} = Math.sin({i}) * {i} + x * {i + 1};"
        for i in range(40)
    )
    return (
        "<!DOCTYPE html><html><head><title>t</title></head><body>"
        "<canvas id='c' width='480' height='480'></canvas><script>\n"
        "(() => {\n"
        "  const ctx = document.getElementById('c').getContext('2d');\n"
        f"  const tag = '{marker}';\n"
        "  let x = 0;\n"
        "  window.addEventListener('keydown', (e) => { x += 1; });\n"
        + filler + "\n"
        "  function loop() {\n"
        "    ctx.fillStyle = '#123';\n"
        "    ctx.fillRect(0, 0, 480, 480);\n"
        "    ctx.fillStyle = '#fff';\n"
        "    ctx.fillRect(x % 480, 100, 20, 20);\n"
        "    requestAnimationFrame(loop);\n"
        "  }\n"
        "  window.state = { player: { x: 0 } };\n"
        "  loop();\n"
        "})();\n"
        "</script></body></html>"
    )


# ---------------------------------------------------------------------------
# 1. tools.py: `report` must be created before any report[...] access in
#    load_and_test (AST ordering check — guards the UnboundLocalError).
# ---------------------------------------------------------------------------


def test_load_and_test_never_touches_report_before_creation() -> None:
    src = (ROOT / "tools.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "load_and_test":
            fn = node
            break
    assert fn is not None, "load_and_test not found in tools.py"
    first_assign_line = None
    subscript_lines: list[int] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "report":
                    if first_assign_line is None or node.lineno < first_assign_line:
                        first_assign_line = node.lineno
        if isinstance(node, ast.Subscript):
            v = node.value
            if isinstance(v, ast.Name) and v.id == "report":
                subscript_lines.append(node.lineno)
    assert first_assign_line is not None
    early = [ln for ln in subscript_lines if ln < first_assign_line]
    assert not early, (
        f"report[...] used at lines {early} before `report = ...` at line "
        f"{first_assign_line} — this is the UnboundLocalError that crashed "
        "the harness on good games."
    )


# ---------------------------------------------------------------------------
# 2. Harness-crash message: no more "simplify the page" blame.
# ---------------------------------------------------------------------------


def test_harness_crash_message_does_not_blame_the_game() -> None:
    src = (ROOT / "agent.py").read_text(encoding="utf-8")
    assert "Please simplify the page and try again" not in src
    assert "HARNESS FAILURE (not a game bug)" in src
    # Traceback capture exists on the crash path.
    assert '"harness_crash"' in src or "'harness_crash'" in src


# ---------------------------------------------------------------------------
# 3. Rewrite-exemption invariant: every fallback prompt that ORDERS a full
#    <html_file> must be recognized by _prompt_orders_full_rewrite.
# ---------------------------------------------------------------------------


def test_duplicate_decl_coaching_orders_rewrite_and_is_detected() -> None:
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False, has_existing_file=True, consecutive_plan_only=0,
        materialize_reject_reason=(
            "<html_file> rejected: duplicate top-level declaration(s) in "
            "<script>: `sprite` declared 2+ times"
        ),
    )
    assert "DUPLICATE TOP-LEVEL DECLARATIONS" in msg
    assert GameAgent._prompt_orders_full_rewrite(msg)


def test_plan_only_with_file_orders_rewrite_and_is_detected() -> None:
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=True, has_existing_file=True, consecutive_plan_only=1,
    )
    assert GameAgent._prompt_orders_full_rewrite(msg)


def test_scope_reduction_prompt_is_detected() -> None:
    msg = prompts_v1.scope_reduction_instruction("OK: False")
    assert GameAgent._prompt_orders_full_rewrite(msg)


def test_generic_and_loop_escalations_do_not_arm_exemption() -> None:
    # Generic fallback: conditional mention of <html_file>, not an order.
    generic, _ = GameAgent._no_usable_code_fallback(
        plan_only=False, has_existing_file=True, consecutive_plan_only=0,
    )
    assert not GameAgent._prompt_orders_full_rewrite(generic)
    # Identical-reply escalation explicitly says do NOT re-emit <html_file>.
    loop_msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False, has_existing_file=True, consecutive_plan_only=0,
        identical_repeat=True,
    )
    assert not GameAgent._prompt_orders_full_rewrite(loop_msg)


# ---------------------------------------------------------------------------
# 4. Truthful fallback for baseline-exists rejections.
# ---------------------------------------------------------------------------


def test_baseline_exists_rejection_gets_truthful_fallback() -> None:
    msg, _ = GameAgent._no_usable_code_fallback(
        plan_only=False, has_existing_file=True, consecutive_plan_only=0,
        materialize_reject_reason=(
            "<html_file> rejected: a baseline file already exists. "
            "Send <patch> SEARCH/REPLACE blocks instead."
        ),
    )
    # Must acknowledge the html_file WAS received (not "could not find").
    assert "WAS received" in msg
    assert "could not find" not in msg.lower()
    assert "<patch>" in msg


# ---------------------------------------------------------------------------
# 5. ASSET SANITY WARNING never classifies as a user art request.
# ---------------------------------------------------------------------------


def test_sentinel_advisory_is_not_an_art_request() -> None:
    advisory = (
        _HARNESS_ADVISORY_SENTINEL + "\n"
        "ASSET SANITY WARNING — these generated sprites came out nearly "
        "identical to the `from_image` parent: `cpu_block` 96% identical "
        "to parent `cpu_idle`."
    )
    names = ["cpu_block", "cpu_idle", "player_punch"]
    assert _feedback_is_art_change(advisory, names) is False
    # A genuine user ask still classifies.
    assert _feedback_is_art_change(
        "please redraw the cpu_block sprite in a red gi", names,
    ) is True


# ---------------------------------------------------------------------------
# 6. Failing-baseline salvage in _materialize.
# ---------------------------------------------------------------------------


def test_materialize_accepts_rewrite_when_baseline_failing(tmp_path) -> None:
    a = _make_agent(tmp_path)
    a._current_file = _real_html("v1")
    a.out_path.write_text(a._current_file, encoding="utf-8")
    a._snapshot_n = 1
    a._previous_report_ok = False  # baseline is failing the harness
    reply = "<html_file>\n" + _real_html("v2") + "\n</html_file>"
    html, msg = asyncio.run(a._materialize(reply))
    assert html is not None, f"rewrite rejected: {msg}"
    assert "v2" in html


def test_materialize_rejects_rewrite_when_baseline_working(tmp_path) -> None:
    a = _make_agent(tmp_path)
    a._current_file = _real_html("v1")
    a.out_path.write_text(a._current_file, encoding="utf-8")
    a._snapshot_n = 1
    a._previous_report_ok = True  # working baseline: keep the ban
    reply = "<html_file>\n" + _real_html("v2") + "\n</html_file>"
    html, msg = asyncio.run(a._materialize(reply))
    assert html is None
    assert "baseline file already exists" in msg


# ---------------------------------------------------------------------------
# 7. repair_reply: end-of-stream patch missing >>>>>>> REPLACE recovered.
# ---------------------------------------------------------------------------


def test_truncated_patch_missing_replace_terminator_is_recovered() -> None:
    reply = (
        "<diagnose>RAF gated on asset load.</diagnose>\n"
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "  loadAssets().then(() => requestAnimationFrame(frame));\n"
        "=======\n"
        "  requestAnimationFrame(frame);\n"
        "  loadAssets();"
        # stream died here — no >>>>>>> REPLACE, no </patch>
    )
    patches = extract_patches(reply)
    assert len(patches) == 1
    assert "loadAssets().then" in patches[0].search
    assert "requestAnimationFrame(frame);" in patches[0].replace


def test_mid_reply_malformed_patch_still_dropped() -> None:
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "  old();\n"
        "=======\n"
        "  new();\n"
        "</patch>\n"  # malformed: closed WITHOUT the REPLACE marker
        "Some prose after the patch block.\n"
    )
    # Not end-of-stream truncation (reply continues past the block) —
    # the malformed block must not be silently "repaired".
    assert extract_patches(reply) == []


def test_truncation_at_divider_is_not_repaired() -> None:
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "  old();\n"
        "=======\n"
    )
    # Stream died AT the divider — intended replacement unknowable; do
    # not synthesize a deletion patch.
    assert extract_patches(reply) == []


# ---------------------------------------------------------------------------
# 8. Playbook code-stage relevance floor.
# ---------------------------------------------------------------------------


def test_code_stage_floor_drops_noise_hit_plan_keeps_it(tmp_path) -> None:
    goal = (
        "build a single screen fighting game with punch kick fireball "
        "health bars dojo stage blue red fighters"
    )
    noise = Bullet(
        id="fps-noise",
        content=(
            "first person camera forward vector strafe pitch yaw raycast "
            "column wall slice texture column height distance shading "
            "minimap render fov projection plane fireball spread"
        ),
        tags=["3d", "raycast", "camera"],
    )
    pb = Playbook(root=str(tmp_path / "memory"))
    pb.ensure()
    pb._save_all([noise])
    plan_hits = pb.retrieve(goal, stage="plan")
    code_hits = pb.retrieve(goal, stage="code")
    assert plan_hits, "plan stage should keep low-sim hits (broad by design)"
    assert 0.02 <= plan_hits[0].score < 0.035, (
        f"fixture sim {plan_hits[0].score:.4f} is outside the noise band "
        "this test is meant to cover — adjust the bullet text"
    )
    assert code_hits == [], "code stage must drop sub-floor noise hits"


# ---------------------------------------------------------------------------
# 9. render_run_summary renders the per-iter table from jsonl records.
# ---------------------------------------------------------------------------


def test_render_run_summary_table() -> None:
    records = [
        {"kind": "session_start", "goal": "build a fight game",
         "model_name": "stub-27b"},
        {"kind": "code_snapshot", "iteration": 1, "size": 15813,
         "materialize": "full <html_file> rewrite"},
        {"kind": "iter_summary", "iteration": 1, "ok": False,
         "probes_passed": 6, "probes_total": 10,
         "soft_warnings": ["PROBE FAILED [game_state_exists]: falsy"],
         "fail_reason": "page_errors:1"},
        {"kind": "no_usable_code", "plan_only": False,
         "identical_repeat": False},
        {"kind": "harness_crash", "err": "boom"},
        {"kind": "event", "event": "restart",
         "text_preview": "restart winner: attempt 2 score=88"},
        {"kind": "session_outcome", "ok": True, "iterations": 4,
         "best_path_exists": True},
    ]
    text = render_run_summary(records, artifact_id="fight__run_x")
    assert "fight__run_x" in text
    assert "build a fight game" in text
    assert "| 1 | full <html_file> rewrite | 15813 | False | 6/10 |" in text
    assert "No-usable-code turns: 1" in text
    assert "Harness crashes: 1" in text
    assert "restart winner: attempt 2 score=88" in text
    assert "Outcome: ok=True iterations=4 best_exists=True" in text
