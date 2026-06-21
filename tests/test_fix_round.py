"""Tests for the fix round (plan fix_unwinnable_gates_and_stalls).

Seven trace-backed fixes from run 20260610_185238, all covered with
pure-function / source-pinned tests — no model, no Chromium:

  1. Control-recovery re-test (permanent stun-lock detection)
  2. Sprite-audit gate demotion (UNDRAWN persistence, CODE_DRAWN_OVER_SPRITE,
     ACTION_DRAWN_NOT_SPRITED declared-keys filter)
  3. Absolute prompt-token compaction ceiling + silent-stall recovery
  4. _SUMMARIZE_FENCE_RE content sniff (no more prose mangling)
  5. Stuck best-of-2 guard on deterministic audit blockers
  6. Critic action-frame fairness (skip, don't fail; payload dedupe)
  7. Sprite orientation pinning (facing-right by construction)
"""

from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402
from agent import GameAgent, _COMPACT_TOKEN_CEILING  # noqa: E402
from tools import (  # noqa: E402
    control_not_recovered_verdict,
    summarize_state_timeline,
)


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=4,
        memory_root=str(tmp_path / "memory"),
    )


def _green_report() -> dict:
    return {
        "ok": False,
        "errors": [],
        "page_errors": [],
        "soft_warnings": ["ASSETS_LOADED_BUT_UNDRAWN [x]: cosmetic"],
        "warnings": [],
        "probes": [{"name": "p1", "ok": True}, {"name": "p2", "ok": True}],
    }


def _failing_report() -> dict:
    r = _green_report()
    r["probes"] = [{"name": "p1", "ok": False}]
    r["page_errors"] = ["TypeError: boom"]
    return r


# ---------------------------------------------------------------------------
# 1. Control-recovery re-test
# ---------------------------------------------------------------------------

def test_control_recovery_verdict_flags_permanent_lock():
    # Moved early, frozen on recheck, retry still frozen -> flag.
    assert control_not_recovered_verdict(
        has_position_state=True, moved_early=True,
        recheck_moved=False, retry_moved=False,
    ) is True


def test_control_recovery_verdict_tolerates_transient_hit_stun():
    # Frozen once but the retry (after the grace wait) moves -> no flag.
    assert control_not_recovered_verdict(
        has_position_state=True, moved_early=True,
        recheck_moved=False, retry_moved=True,
    ) is False


def test_control_recovery_verdict_recovered_immediately():
    assert control_not_recovered_verdict(
        has_position_state=True, moved_early=True,
        recheck_moved=True, retry_moved=None,
    ) is False


def test_control_recovery_verdict_never_moved_is_player_stuck_job():
    # PLAYER-STUCK owns "never moved"; no double flag.
    assert control_not_recovered_verdict(
        has_position_state=True, moved_early=False,
        recheck_moved=False, retry_moved=False,
    ) is False


def test_control_recovery_verdict_skips_without_position_state():
    # Menu/board games: arrows are not motion -> not applicable.
    assert control_not_recovered_verdict(
        has_position_state=False, moved_early=True,
        recheck_moved=False, retry_moved=False,
    ) is False


def test_smoke_test_wires_recovery_retest():
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    assert '"control_not_recovered"' in src
    assert "control_not_recovered_verdict" in src
    # Re-dispatches the remembered movement key and rechecks its leaves.
    assert "recovery_key" in src and "recovery_leaves" in src


def test_load_and_test_gates_control_not_recovered():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "CONTROL-NOT-RECOVERED" in src
    start = src.index("CONTROL-NOT-RECOVERED")
    excerpt = src[start:start + 900]
    # Gating channel + names the early-return-above-timer cause pattern.
    block = src[src.rfind("soft_warnings", 0, start):start + 900]
    assert "soft_warnings" in block
    assert "early-return" in excerpt or "early return" in excerpt
    assert "timer" in excerpt.lower()


def test_snapshot_js_captures_short_string_leaves():
    js = tools_module._GAMESTATE_SNAPSHOT_JS
    assert "typeof obj === 'string'" in js
    assert "16" in js  # the short-string cap


def test_timeline_flags_stuck_state_string():
    samples = [
        {"player.action": "hit", "player.x": 100.0 + i, "t": float(i)}
        for i in range(6)
    ]
    out = summarize_state_timeline(samples)
    assert "player.action stuck at 'hit'" in out
    assert "all 6 samples" in out


def test_timeline_ignores_defaultish_and_non_stateish_strings():
    # 'idle' is a default value; 'name' is not a state-machine leaf.
    samples = [
        {"player.action": "idle", "player.name": "Ryu",
         "player.x": 100.0 + i, "t": float(i)}
        for i in range(6)
    ]
    out = summarize_state_timeline(samples)
    assert "stuck at" not in out


def test_timeline_ignores_changing_state_string():
    vals = ["idle", "hit", "idle", "punch", "idle", "kick"]
    samples = [
        {"player.action": vals[i], "player.x": 100.0 + i, "t": float(i)}
        for i in range(6)
    ]
    out = summarize_state_timeline(samples)
    assert "stuck at" not in out


def test_playbook_has_stun_timer_bullet():
    import json
    pb = Path(__file__).parent.parent / "memory" / "playbook.jsonl"
    recs = [json.loads(l) for l in pb.read_text().splitlines() if l.strip()]
    ids = {r["id"] for r in recs}
    assert "stun-timer-before-early-return" in ids


# ---------------------------------------------------------------------------
# 2. Sprite-audit gate demotion
# ---------------------------------------------------------------------------

def test_undrawn_gates_first_occurrence_then_demotes():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    # Persistence flag drives the channel decision.
    assert "_undrawn_seen_before" in src
    start = src.index("ASSETS_LOADED_BUT_UNDRAWN")
    window = src[start:start + 4200]
    # Demotion requires green probes/no errors, plus either a repeat or a
    # no-missing-entity behaviorally green report.
    assert "_probes_green" in window and "_no_errors" in window
    assert "_entity_missing_count" in window
    assert "_opening_hard_green" in window
    assert "ADVISORY (non-blocking)" in window
    # Non-green first occurrence still gates via soft_warnings.
    assert 'report["soft_warnings"].append(_undrawn_text)' in window


def test_undrawn_streak_resets_when_finding_clears():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "self._undrawn_seen_before = False" in src


def test_code_drawn_over_sprite_never_gates():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    start = src.index("CODE_DRAWN_OVER_SPRITE")
    # The emit call must target the non-gating warnings channel.
    emit_block = src[src.rfind("report.setdefault", 0, start):start + 200]
    assert '"warnings"' in emit_block
    assert "advisory" in src[start:start + 200].lower()
    # And no soft_warnings append remains for this finding.
    after = src[start:start + 800]
    assert 'soft_warnings"].append' not in after


def test_action_drawn_not_sprited_only_declared_keys():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    fake_i = src.index("ACTION_DRAWN_NOT_SPRITED")
    cond = src[src.rfind("faked = [", 0, fake_i):fake_i]
    assert "_declared_action_keys" in cond
    assert "kc in _declared_action_keys" in cond
    # Declared keys come from criteria MINUS movement defaults (KeyA/Arrow*).
    decl_i = src.rfind("_declared_action_keys = ", 0, fake_i)
    decl_block = src[decl_i:fake_i]
    assert "_parse_action_keys(criteria" in decl_block
    assert "_MOVEMENT_KEYS" in decl_block


def test_smoke_test_induces_combat_before_recovery_retest():
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    assert "_induce_combat_before_recovery" in src
    assert "await _induce_combat_before_recovery()" in src
    # Only criteria-declared non-movement keys — genre-free.
    block = src[src.index("async def _induce_combat_before_recovery"):src.index("async def _recovery_leaves_move_again")]
    assert "_MOVEMENT_KEYS" in block
    assert "_RESTART_KEYS" in block
    assert "_parse_action_keys(criteria" in block


def test_recovery_key_prefers_lateral_position_movers():
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    assert "x_leaves" in src
    assert 'l.endswith(".x")' in src


# ---------------------------------------------------------------------------
# 3. Token ceiling + silent-stall recovery
# ---------------------------------------------------------------------------

def _stuffed_agent(tmp_path, n_msgs: int = 12) -> GameAgent:
    a = _make_agent(tmp_path)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"turn {i} " + "x" * 200})
    a._messages = msgs
    return a


def test_token_ceiling_forces_structured_compaction(tmp_path):
    a = _stuffed_agent(tmp_path)
    a._last_prompt_tokens = _COMPACT_TOKEN_CEILING + 1000
    a._last_prompt_pressure = 0.46  # below _COMPACT_PRESSURE
    a._prune_messages()
    anchor = [m for m in a._messages if "STATE ANCHOR" in (m.get("content") or "")]
    assert len(anchor) == 1


def test_below_ceiling_low_pressure_no_compaction(tmp_path):
    a = _stuffed_agent(tmp_path)
    a._last_prompt_tokens = _COMPACT_TOKEN_CEILING - 1000
    a._last_prompt_pressure = 0.46
    a._prune_messages()
    anchor = [m for m in a._messages if "STATE ANCHOR" in (m.get("content") or "")]
    assert not anchor


def test_silent_stall_flag_forces_compaction_once(tmp_path):
    a = _stuffed_agent(tmp_path)
    a._last_prompt_tokens = 9000
    a._last_prompt_pressure = 0.09
    a._force_compact_after_stall = True
    a._prune_messages()
    anchor = [m for m in a._messages if "STATE ANCHOR" in (m.get("content") or "")]
    assert len(anchor) == 1
    # One-shot: consumed by the prune pass.
    assert a._force_compact_after_stall is False


def test_silent_abort_handler_sets_stall_flag():
    src = inspect.getsource(GameAgent)
    i = src.index('"kind": "stream_silent_aborted"')
    before = src[max(0, i - 600):i]
    assert "_force_compact_after_stall = True" in before


# ---------------------------------------------------------------------------
# 4. Fence sniff
# ---------------------------------------------------------------------------

def test_prose_js_fence_stays_verbatim(tmp_path):
    a = _make_agent(tmp_path)
    prose = "const x = 1; // " + ("reasoning prose " * 100)  # > 1024 bytes
    content = f"Some analysis.\n```js\n{prose}\n```\nMore analysis."
    out = a._summarize_content(content)
    assert "HARNESS-OMITTED-PRIOR-FENCE" not in out
    assert prose in out


def test_html_document_fence_still_elides(tmp_path):
    a = _make_agent(tmp_path)
    body = "<!DOCTYPE html>\n<html><body>" + ("x" * 1200) + "</body></html>"
    content = f"```html\n{body}\n```"
    out = a._summarize_content(content)
    assert "HARNESS-OMITTED-PRIOR-FENCE" in out
    assert "x" * 1200 not in out


def test_bare_fence_with_html_root_elides(tmp_path):
    a = _make_agent(tmp_path)
    body = "<html><body>" + ("y" * 1200) + "</body></html>"
    out = a._summarize_content(f"```\n{body}\n```")
    assert "HARNESS-OMITTED-PRIOR-FENCE" in out


def test_cross_fence_prose_not_swallowed(tmp_path):
    """The trace-20260610 failure: closing fence + prose + next opening
    fence matched as one 'fence body', eliding the prose between two ```js
    blocks. With the sniff, the prose survives."""
    a = _make_agent(tmp_path)
    filler = "p " * 600
    content = (
        "```js\nconst a = 1;\n```\n"
        f"IMPORTANT diagnosis prose {filler}\n"
        "```js\nconst b = 2;\n```"
    )
    out = a._summarize_content(content)
    assert "IMPORTANT diagnosis prose" in out
    assert "HARNESS-OMITTED-PRIOR-FENCE" not in out


# ---------------------------------------------------------------------------
# 5. Stuck best-of-2 guard
# ---------------------------------------------------------------------------

def test_bon_guard_skips_on_green_audit_only_report():
    f = GameAgent._should_escalate_stuck_bon
    # Probes green + no errors -> deterministic audit blocker; skip.
    assert f(2, 1, 0, last_report=_green_report()) is False


def test_bon_guard_fires_on_failing_probes_or_errors():
    f = GameAgent._should_escalate_stuck_bon
    assert f(2, 1, 0, last_report=_failing_report()) is True
    r = _green_report()
    r["page_errors"] = ["ReferenceError: nope"]
    assert f(2, 1, 0, last_report=r) is True


def test_bon_guard_backward_compatible_without_report():
    f = GameAgent._should_escalate_stuck_bon
    assert f(2, 1, 0) is True
    assert f(1, 1, 0) is False
    assert f(2, 2, 0) is False


# ---------------------------------------------------------------------------
# 6. Critic action-frame fairness
# ---------------------------------------------------------------------------

class _Recipe:
    def __init__(self, checklist, fix_hints=None):
        self.id = "test-recipe"
        self.recipe = {"checklist": checklist}
        if fix_hints is not None:
            self.recipe["fix_hints"] = fix_hints


_CHECKLIST = [
    "Are TWO distinct character figures visible?",
    "In the ACTION frame (the mid-input image), is a distinct attack pose visible?",
    "Are there 2 health indicators visible?",
    "Does each character keep its look between the resting frame and the action frame?",
]


def test_action_frame_question_indices():
    idxs = GameAgent._action_frame_question_indices(_CHECKLIST)
    assert idxs == [1, 3]
    assert GameAgent._action_frame_question_indices([]) == []
    assert GameAgent._action_frame_question_indices(["Is the canvas blank?"]) == []


def test_strip_action_frame_questions_renumbers_fix_hints(tmp_path):
    a = _make_agent(tmp_path)
    recipe = _Recipe(
        list(_CHECKLIST),
        fix_hints={"1": "hint one", "2": "hint two", "3": "hint three", "4": "hint four"},
    )
    stripped, skipped = a._strip_action_frame_questions(recipe)
    assert len(skipped) == 2
    assert all("action frame" in q.lower() or "mid-input" in q.lower() for q in skipped)
    assert stripped.recipe["checklist"] == [_CHECKLIST[0], _CHECKLIST[2]]
    # Old Q3 ("hint three") is the new Q2; action-frame hints dropped.
    assert stripped.recipe["fix_hints"] == {"1": "hint one", "2": "hint three"}
    # Original recipe untouched (clone semantics).
    assert recipe.recipe["checklist"] == _CHECKLIST


def test_strip_noop_without_action_questions(tmp_path):
    a = _make_agent(tmp_path)
    recipe = _Recipe(["Is the canvas blank?", "Is a HUD visible?"])
    stripped, skipped = a._strip_action_frame_questions(recipe)
    assert stripped is recipe
    assert skipped == []


def test_run_visual_critic_wires_skip_and_advisory():
    src = inspect.getsource(GameAgent.run_visual_critic)
    assert "_strip_action_frame_questions" in src
    # Strips ONLY when no action frame was captured + animation expected.
    i = src.index("_strip_action_frame_questions")
    guard = src[src.rfind("if recipe is not None", 0, i):i]
    assert "action_png is None" in guard
    assert "_animation_expected()" in guard
    # One-per-attempt deterministic advisory, pointing at the real cause.
    assert "_no_action_frame_advisory_sent" in src
    assert "CONTROL-NOT-RECOVERED" in src
    # Payload-fingerprint dedupe skips the VLM call.
    assert "visual_critic_skipped_duplicate" in src
    assert "_suppressed_critic_payload_fp" in src


def test_suppression_records_payload_fingerprint():
    src = inspect.getsource(GameAgent._queue_visual_critic_coaching)
    assert "_suppressed_critic_payload_fp" in src
    assert "_current_critic_payload_fp" in src


def test_reset_attempt_state_clears_critic_fairness_state(tmp_path):
    a = _make_agent(tmp_path)
    a._no_action_frame_advisory_sent = True
    a._suppressed_critic_payload_fp = "abc"
    a._current_critic_payload_fp = "def"
    a._reset_attempt_state()
    assert a._no_action_frame_advisory_sent is False
    assert a._suppressed_critic_payload_fp is None
    assert a._current_critic_payload_fp is None


# ---------------------------------------------------------------------------
# 7. Sprite orientation is NOT pinned/rewritten by the pipeline anymore.
# The facing convention now lives in the playbook (directional-art-faces-right);
# generate_assets must pass prompts through verbatim and never tag a stat with
# orientation_pinned. (Art-direction policy belongs in memory, not in code.)
# ---------------------------------------------------------------------------

def test_generate_assets_does_not_pin_orientation(tmp_path):
    """The pipeline must NOT mutate prompts or tag orientation_pinned —
    facing is a code/memory convention, not an asset-pipeline policy."""
    from assets import generate_assets
    from PIL import Image

    sent_prompts: list[str] = []

    class StubGen:
        last_stats: list = []

        def generate(self, prompt: str):
            sent_prompts.append(prompt)
            p = tmp_path / f"gen_{len(sent_prompts)}.png"
            Image.new("RGB", (64, 64), (255, 0, 255)).save(p)
            return str(p)

    specs = [
        {"name": "player_kick", "prompt": "ninja flying kick, transparent background", "size": (64, 64)},
        {"name": "dojo_bg", "prompt": "dojo interior, warm light", "size": (64, 64)},
    ]
    gen = StubGen()
    out = generate_assets(
        specs, tmp_path / "session",
        cache_dir=tmp_path / "cache",
        image_generator=gen,
        img2img_generator=None,
    )
    assert "player_kick" in out and "dojo_bg" in out
    # prompts passed through verbatim — nothing appended by the pipeline
    assert "ninja flying kick, transparent background" in sent_prompts
    assert all("side view, facing right" not in p for p in sent_prompts)
    stats = {s["name"]: s for s in gen.last_stats}
    assert "orientation_pinned" not in stats["player_kick"]
    assert "orientation_pinned" not in stats["dojo_bg"]


def test_playbook_has_directional_art_bullet():
    import json
    pb = Path(__file__).parent.parent / "memory" / "playbook.jsonl"
    recs = [json.loads(l) for l in pb.read_text().splitlines() if l.strip()]
    ids = {r["id"] for r in recs}
    assert "directional-art-faces-right" in ids


def test_playbook_covers_major_genres_and_concepts():
    """Memory must know the major game genres + cross-cutting concepts.
    These bullets fill the gaps the recipes/outlines didn't already cover."""
    import json
    pb = Path(__file__).parent.parent / "memory" / "playbook.jsonl"
    recs = [json.loads(l) for l in pb.read_text().splitlines() if l.strip()]
    ids = {r["id"] for r in recs}
    required = {
        "endless-runner-autoscroll",
        "box-push-sokoban",
        "flood-reveal-grid",
        "gravity-drop-board",
        "deckbuilder-card-economy",
        "save-load-persistence",
        "difficulty-progression-scaling",
    }
    missing = required - ids
    assert not missing, f"playbook missing major-genre/concept bullets: {sorted(missing)}"


def test_directional_art_bullet_has_no_pipeline_claim():
    """The facing convention is model guidance now — it must NOT claim the
    asset pipeline pins orientation (that machinery was removed)."""
    import json
    pb = Path(__file__).parent.parent / "memory" / "playbook.jsonl"
    bullet = next(
        json.loads(l) for l in pb.read_text().splitlines()
        if l.strip() and json.loads(l)["id"] == "directional-art-faces-right"
    )
    assert "asset pipeline pins" not in bullet["content"]
    assert "hero_left" in bullet["content"]  # teaches: don't name by direction
