"""Async event-driven coding agent for HTML games.

Public surface is unchanged from the previous version so chat.py and coder.py
keep working:

    GameAgent(model, out_path, browser, max_iters)
        .set_token_callback(cb)
        .add_user_feedback(text)
        .add_user_answer(text)
        .request_done()
        .has_pending_user_input() -> bool
        .run(goal) -> AsyncIterator[AgentEvent]

Internally the agent now layers six things on top of the original loop:

  1. Streaming watchdog (ollama_io.stream_chat). The old loop hung
     indefinitely if Ollama stopped yielding tokens; we now abort on a
     per-chunk inactivity timeout and recover.

  2. Patch-based editing. After the first build, the model emits
     <patch>SEARCH/REPLACE</patch> blocks against the current file on disk.
     Falls back to a full <html_file> if patches don't parse or don't apply.

  3. Persistent memory (memory.py). On a new goal we retrieve the closest
     past skeleton; on a failed test we retrieve past mistakes whose
     signature matches and surface them in the diagnose prompt.

  4. Best-of-N. On failed iterations we sample N candidate fixes in
     parallel (different temperatures) and pick the one whose patches
     actually pass the test.

  5. Diagnose-then-fix in ONE turn. The fix prompt asks the model to emit
     <diagnose>root cause in 2 sentences</diagnose> BEFORE its patches.
     The diagnosis is stashed in memory if the resulting fix lands clean.

  6. VLM screenshot review. When the model is vision-capable, the latest
     screenshot is attached AND the prompt explicitly tells the model to
     use it. Half the wiring was already there; the prompt half was not.

We also aggressively prune old <html_file> blobs out of conversation
history every turn so context stays bounded regardless of iteration count.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from assets import (
    _DERIVED_FRAME_MIN_DELTA,
    generate_assets,
    parse_assets_block,
    parse_assets_block_with_meta,
    render_asset_paths_block,
    try_load_image_generator,
)
from sounds import (
    generate_sounds,
    parse_sounds_block,
    render_sound_paths_block,
    try_load_audio_generator,
)
# Video cutscenes (<videos> tag) — sibling of assets/sounds. Wan2.2 runs
# in a subprocess (scripts/generate_video.py), so this import is cheap.
from videos import (
    generate_videos,
    parse_videos_block,
    render_video_paths_block,
    try_load_video_generator,
)
from backend import Backend, BackendInfo, detect_backend, make_backend
from memory import (
    CANVAS_SKELETON_V2,
    CANVAS_SKELETON_V2_NAME,
    DEFAULT_SKELETON,
    DEFAULT_SKELETON_NAME,
    GameMemory,
    OpeningBookItem,
    Playbook,
    PLAYTESTS_FILENAME,
    ASSET_AUDITS_FILENAME,
    ANIMATION_AUDITS_FILENAME,
    SkeletonHit,
    lookup_bullet,
    signature_for_report,
)
from ollama_io import Candidate, StreamResult
from patches import FormatRejection, apply_patches, classify_format_failure, extract_patches

# Prompt-module routing: v1 is the production prompt module (`prompts_v1.py`).
# v0 (`prompts.py`) was retired — it never grew the playbook / criteria /
# probes machinery v1 ships with, and every live driver was passing
# `prompt_version="v1"` already. Future revisions should add `prompts_v2.py`
# alongside v1 and route via the `prompt_version` constructor argument.

from tools import (
    LiveBrowser,
    format_micro_probes_for_model,
    format_report_for_model,
    run_micro_probes,
    score_test_report,
)
# Probe-result handlers (impossible-probe downgrade, eval-error quarantine,
# all-quarantine ship gate) live in agent_probes.py; GameAgent inherits them.
from agent_probes import ProbeHandlingMixin
# First-build/plan-turn memory retrieval (opening-book, components, lean budget,
# open-domain detection) lives in agent_memory.py; GameAgent inherits it.
from agent_memory import MemoryRetrievalMixin
from agent_prompts import PromptBuildingMixin
from agent_compaction import CompactionMixin
from agent_stream import StreamMaterializeMixin, _repetition_loop_abort_message
from agent_feedback import FeedbackRoutingMixin
from agent_feedback import (
    _ART_NOUNS,
    _BEHAVIOR_BUG_COMPLAINT_RE,
    _BEHAVIOR_BUG_NEGATION_RE,
    _BEHAVIOR_VERB_ALT,
    _BLOCKER_NEGATION_RE,
    _HARNESS_ADVISORY_SENTINEL,
    _MEDIA_VERBS,
    _SEED_EDIT_MAX_PATCHES,
    _feedback_is_art_change,
    _feedback_is_behavior_bug,
    _feedback_is_orientation_change,
    _feedback_is_sound_change,
    _feedback_is_strict_scope,
    _feedback_is_ui_feature,
    _feedback_locks_code,
    _feedback_mentions_scoped_behavior_change,
    _feedback_requests_existing_media,
    _feedback_requests_explicit_new_art,
    _feedback_requests_img2img_chain,
    _feedback_requests_size_change,
    _feedback_requests_style_rebrand,
    _feedback_vocab,
    _goal_is_small_scope_edit,
    _has_audio_context,
    _matched_names_in_text,
    _name_in_text,
    _phrase_is_negated,
    _resolve_fuzzy_asset_stems,
    _scoped_probe_keywords,
    _subsystem_hint,
)



# Pure helpers — seed media, HTML parsing, compaction constants (agent_helpers.py).
from agent_helpers import (
    _ANTHROPIC_NON_RETRYABLE_400_PHRASES,
    _ASSETS_OPEN_RE,
    _BARE_DOCTYPE_RE,
    _BARE_HTML_ELEMENT_RE,
    _BLOAT_BLOCK_LINES,
    _BLOAT_MAX_REPEATS,
    _BLOAT_MIN_BLOCK_BYTES,
    _COMPACT_MESSAGE_CAP,
    _COMPACT_PRESSURE,
    _COMPACT_TOKEN_CEILING,
    _CONFIRM_RE,
    _COSMETIC_SPRITE_WARNING_PREFIXES,
    _CRITERIA_RE,
    _DIAGNOSE_RE,
    _DONE_RE,
    _HTML_FENCE_RE,
    _HTML_RE,
    _IMAGE_EXTS,
    _LOOKUP_BULLET_RE,
    _MAX_BULLET_LOOKUPS_PER_TURN,
    _NOTES_RE,
    _PLACEHOLDER_FIRST_BUILD_MIN_CODE,
    _PLAN_OPEN_RE,
    _POLISH_TURN_CAP,
    _PROBES_OPEN_RE,
    _PRUNE_KEEP_RECENT_TURNS,
    _QUESTION_RE,
    _REPORT_BLOCK_BEGIN,
    _REPORT_BLOCK_END,
    _REPORT_BLOCK_RE,
    _SEED_ASSET_RE,
    _SEED_SOUND_RE,
    _SKELETON_MAX_BYTES,
    _SKELETON_MIN_BODY_BYTES,
    _SOUNDS_OPEN_RE,
    _SOUND_EXTS,
    _STRUCTURED_PRUNE_THRESHOLD,
    _STUCK_BON_ESCALATION_CAP,
    _SUMMARIZE_MIN_HTML_BYTES,
    _SUMMARIZE_MIN_PROBES_BYTES,
    _TODOS_RE,
    _UNCLOSED_HTML_FILE_RE,
    _baseline_structurally_broken,
    _detect_block_bloat,
    _detect_skeleton_payload,
    _is_degenerate_baseline,
    _is_placeholder_first_build,
    _looks_like_placeholder_html_payload,
    _normalize_extracted_html,
    _patch_set_bracket_break,
    _png_dims,
    _deprecated_project_config_on_disk,
    _report_green_except_cosmetic_sprites,
    _scan_seed_media,
    _strip_thinking,
    _truncation_reason,
)


@dataclass
class AgentEvent:
    kind: str           # phase | token | plan | code | test | question | done | error | info | diagnose | patch | best_of_n | memory | activity | assets | sounds | videos | streak
    text: str = ""
    data: dict = field(default_factory=dict)


_TRACE_MAX_BYTES = 2048
_TRACE_PREVIEW_CHARS = 320
_TRACE_CANONICAL_KINDS = frozenset({"stream_start", "assistant_reply"})
_TRACE_IDENTITY_KEYS = (
    "ts", "kind", "event", "iteration", "failure_class", "ok", "err", "error",
    "exc_type", "source", "stage", "recovery", "recovery_action", "stall_reason",
    "last_stall_reason", "html_sha256", "code_sha256", "previous_code_sha256",
    "ids", "retrieved_ids", "patch_applied", "patch_outcome", "probes_passed",
    "probes_total", "failing_probes", "probes", "blocker", "fail_reason",
    "failure_reason", "soft_warnings", "page_errors", "console_errors",
    "shipped_unchanged_after_block", "materialized", "router_intent",
    "coaching_action", "task_ledger_done", "stream_tokens", "stream_duration_s",
    "tok_per_s", "prefill_s", "frozen_canvas", "action_frame_captured",
    "action_key", "static_action", "prompt_tokens", "message_count", "lean_prompt",
    "pending_feedback_count", "drawn_asset_check", "asset_decode_settle",
    "traceback", "edit_first", "model_role", "model_name",
)
_TRACE_REQUIRED_KEYS = (
    "ts", "kind", "event", "iteration", "failure_class",
    "err", "error", "exc_type",
)


def _trace_json_bytes(obj: dict) -> int:
    """Serialized UTF-8 size using the exact compact trace encoding."""
    return len(json.dumps(
        obj, ensure_ascii=False, default=str, separators=(",", ":"),
    ).encode("utf-8"))


def _trace_preview(text: str, limit: int = _TRACE_PREVIEW_CHARS) -> str:
    """Deterministic character preview; JSON byte budgeting happens afterward."""
    return text if len(text) <= limit else text[:limit]


def _compact_trace_value(value: Any, *, string_limit: int, list_limit: int, depth: int = 0) -> Any:
    """Bound nested diagnostic values without modifying their runtime owners."""
    if isinstance(value, str):
        return _trace_preview(value, string_limit)
    if isinstance(value, dict):
        if depth >= 5:
            return {"_type": "dict", "_len": len(value)}
        return {
            str(key): _compact_trace_value(
                item, string_limit=string_limit, list_limit=list_limit, depth=depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        compacted = [
            _compact_trace_value(
                item, string_limit=string_limit, list_limit=list_limit, depth=depth + 1,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            compacted.append({"_omitted": len(value) - list_limit})
        return compacted
    return value


def _compact_trace_payload(payload: dict) -> dict:
    """Return a deterministic <=2 KiB noncanonical trace projection.

    The exact model turn and reply rows are canonical conversation artifacts and
    are deliberately exempt. Everything else is diagnostic data: long nested
    strings/lists are bounded, and an identity-first fallback keeps routing,
    failure, hash, probe, patch, and recovery fields visible.
    """
    kind = payload.get("kind")
    if kind in _TRACE_CANONICAL_KINDS:
        return payload

    original_bytes = _trace_json_bytes(payload)
    projected = copy.deepcopy(payload)

    # The prompt row is a manifest, not a third copy of canonical prompt prose.
    if kind == "system_prompt_built" and isinstance(projected.get("system_prompt"), str):
        prompt = projected.pop("system_prompt")
        projected["system_prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        projected["system_prompt_chars"] = len(prompt)

    # Generic UI rows may mirror a report/plan/reply. Keep only bounded previews
    # and lengths in the persisted trace; the in-memory AgentEvent is unchanged.
    if kind == "event":
        text = str(projected.get("text_preview") or "")
        projected["text_preview"] = _trace_preview(text)
        if isinstance(projected.get("data"), dict):
            def _event_data_projection(value: Any, depth: int = 0) -> Any:
                if isinstance(value, dict):
                    out: dict[str, Any] = {}
                    for key, item in value.items():
                        if key in {
                            "reply", "report", "plan", "content", "prompt",
                            "negative_prompt", "system_prompt", "spec", "prose",
                        }:
                            rendered = item if isinstance(item, str) else json.dumps(
                                item, ensure_ascii=False, default=str, separators=(",", ":"),
                            )
                            out[f"{key}_len"] = len(rendered)
                            if depth == 0 and key in {"reply", "report", "plan", "content"}:
                                out[f"{key}_preview"] = _trace_preview(rendered, 160)
                            continue
                        out[str(key)] = _event_data_projection(item, depth + 1)
                    return out
                if isinstance(value, list):
                    return [_event_data_projection(item, depth + 1) for item in value]
                return value

            data = _event_data_projection(projected["data"])
            projected["data"] = _compact_trace_value(
                data, string_limit=240, list_limit=12,
            )

    # Retrieval traces carry attribution, never reconstructible recipe prose.
    if kind in {
        "opening_book_retrieved", "plan_opening_book_injected",
        "playbook_retrieved", "playbook_injected", "components_injected",
        "outline_traps_injected",
    }:
        def _without_recipe(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _without_recipe(item)
                    for key, item in value.items()
                    if key not in {"recipe", "rendered_text", "content", "prompt"}
                }
            if isinstance(value, list):
                return [_without_recipe(item) for item in value]
            return value
        projected = _without_recipe(projected)

    # Media generation diagnostics retain outcomes and compact numeric/cache
    # stats, but not repeated generation prompt prose.
    if kind in {"assets_generated", "sounds_generated"}:
        stat_key = "per_asset" if kind == "assets_generated" else "per_sound"
        compact_stats = []
        for stat in projected.get(stat_key) or []:
            if not isinstance(stat, dict):
                continue
            compact_stats.append({
                key: value for key, value in stat.items()
                if key not in {"prompt", "negative_prompt", "spec", "prose"}
            })
        projected[stat_key] = compact_stats

    if _trace_json_bytes(projected) <= _TRACE_MAX_BYTES and projected == payload:
        return projected

    projected["_trace_compacted"] = True
    projected["_trace_original_bytes"] = original_bytes
    projected = _compact_trace_value(projected, string_limit=320, list_limit=16)
    if _trace_json_bytes(projected) <= _TRACE_MAX_BYTES:
        return projected

    projected = _compact_trace_value(projected, string_limit=120, list_limit=8)
    if _trace_json_bytes(projected) <= _TRACE_MAX_BYTES:
        return projected

    # Deterministic final fallback: preserve diagnostic identity first, then add
    # remaining compact fields only while the serialized row stays in budget.
    fallback = {
        key: _compact_trace_value(projected[key], string_limit=96, list_limit=6)
        for key in _TRACE_REQUIRED_KEYS
        if key in projected
    }
    fallback["_trace_compacted"] = True
    fallback["_trace_original_bytes"] = original_bytes
    for key in _TRACE_IDENTITY_KEYS:
        if key not in projected or key in fallback:
            continue
        candidate = dict(fallback)
        candidate[key] = _compact_trace_value(
            projected[key], string_limit=96, list_limit=6,
        )
        if _trace_json_bytes(candidate) <= _TRACE_MAX_BYTES:
            fallback = candidate
    for key, value in projected.items():
        if key in fallback:
            continue
        candidate = dict(fallback)
        candidate[key] = _compact_trace_value(value, string_limit=96, list_limit=6)
        if _trace_json_bytes(candidate) <= _TRACE_MAX_BYTES:
            fallback = candidate
    dropped_fields = max(
        0,
        len(projected) - len([
            key for key in fallback
            if not key.startswith("_trace_")
        ]),
    )
    with_dropped = dict(fallback)
    with_dropped["_trace_dropped_fields"] = dropped_fields
    if _trace_json_bytes(with_dropped) <= _TRACE_MAX_BYTES:
        fallback = with_dropped
    # Exception identity is retained even for pathological multi-byte inputs.
    while _trace_json_bytes(fallback) > _TRACE_MAX_BYTES:
        reduced = False
        for key in ("traceback", "err", "error", "fail_reason", "blocker", "soft_warnings"):
            value = fallback.get(key)
            if isinstance(value, str) and len(value) > 24:
                fallback[key] = value[:max(24, len(value) // 2)]
                reduced = True
                break
            if isinstance(value, list) and value:
                fallback[key] = value[:max(1, len(value) // 2)]
                reduced = True
                break
        if not reduced:
            # Metadata/key names alone can only exceed the limit for adversarial
            # input; retain the required identities and omit ancillary fields.
            fallback = {
                key: fallback[key]
                for key in ("ts", "kind", "iteration", "failure_class", "err", "error", "exc_type")
                if key in fallback
            } | {
                "_trace_compacted": True,
                "_trace_original_bytes": original_bytes,
            }
            break
    return fallback


from agent_gates import GateProcessingMixin
from agent_critic import CriticMixin
from agent_assets import AssetGenerationMixin


def render_run_summary(records: list[dict], artifact_id: str = "") -> str:
    """Render a compact per-run markdown summary from parsed jsonl records.

    Pure function (testable without a session). Added 2026-06-10: answering
    "what happened in this run" previously required grepping a multi-hundred-
    KB log; the data was already in the `iter_summary` / `code_snapshot` /
    `no_usable_code` / `harness_crash` trace records — this just renders it
    as one table per run.
    """
    goal = ""
    model = ""
    iters: dict[int, dict] = {}
    no_code_turns: list[str] = []
    harness_crashes = 0
    agent_crashes: list[dict] = []
    restart_lines: list[str] = []
    outcome: dict | None = None
    for rec in records:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if not goal and rec.get("goal"):
            goal = str(rec.get("goal"))
        if not model and rec.get("model_name"):
            model = str(rec.get("model_name"))
        if kind == "code_snapshot":
            it = int(rec.get("iteration") or 0)
            row = iters.setdefault(it, {})
            row["materialize"] = str(rec.get("materialize") or "")[:60]
            row["bytes"] = rec.get("size")
        elif kind == "iter_summary":
            it = int(rec.get("iteration") or 0)
            row = iters.setdefault(it, {})
            row["ok"] = rec.get("ok")
            row["probes"] = (
                f"{rec.get('probes_passed')}/{rec.get('probes_total')}"
            )
            sw = rec.get("soft_warnings") or []
            row["blocker"] = (
                str(sw[0])[:70] if sw else str(rec.get("fail_reason") or "")[:70]
            )
            # Phase 4 (4D): correlation fields for the one-row digest.
            row["patch"] = rec.get("patch_applied") or ""
            row["router"] = rec.get("router_intent") or ""
            row["toks"] = rec.get("tok_per_s") or ""
            row["class"] = rec.get("failure_class") or ""
            if rec.get("test_skipped"):
                row["blocker"] = f"test_skipped:{rec.get('test_skipped')}"
        elif kind == "no_usable_code":
            reason_bits = [
                key for key in ("plan_only", "probes_only", "media_only",
                                "identical_repeat")
                if rec.get(key)
            ]
            # Phase 4 (4D.2): tag the no-code turn with its fix-layer bucket so
            # the digest shows WHERE the iter-4/5-class failure lives.
            _cls = rec.get("failure_class") or ""
            _label = ",".join(reason_bits) or "rejected/unparsed"
            no_code_turns.append(
                f"{_label} [{_cls}]" if _cls and _cls != "none" else _label
            )
        elif kind == "harness_crash":
            harness_crashes += 1
        elif kind == "agent_crash":
            agent_crashes.append(rec)
        elif kind == "event" and rec.get("event") == "restart":
            restart_lines.append(str(rec.get("text_preview") or "")[:90])
        elif kind == "session_outcome":
            outcome = rec
    lines: list[str] = [f"# Run summary — {artifact_id}".rstrip(" —")]
    if goal:
        lines.append(f"Goal: {goal[:200]}")
    if model:
        lines.append(f"Model: {model}")
    lines.append("")
    if agent_crashes:
        cr = agent_crashes[-1]
        where = cr.get("source") or "agent_loop"
        it = cr.get("iteration")
        it_note = f" iter {it}" if it is not None else ""
        lines.append(
            f"**AGENT CRASH** ({where}{it_note}): "
            f"{cr.get('exc_type', 'Exception')}: {cr.get('err', '')}"
        )
        lines.append(
            "(full traceback on the `agent_crash` row in the .jsonl — "
            "grep kind agent_crash)"
        )
        lines.append("")
    if iters:
        # Phase 4 (4D): the table is an LLM-facing digest — one dense row per
        # iter carrying the fix-layer (`class`) + router/patch/tok-s correlation
        # so "why did this iter go this way?" is answerable from ONE row. The
        # original columns keep their order/position for back-compat.
        lines.append(
            "| iter | materialize | bytes | ok | probes | patch | router "
            "| tok/s | class | blocker |"
        )
        lines.append(
            "|------|-------------|-------|----|--------|-------|--------"
            "|-------|-------|---------|"
        )
        for it in sorted(iters):
            row = iters[it]
            lines.append(
                f"| {it} | {row.get('materialize', '')} "
                f"| {row.get('bytes', '')} | {row.get('ok', '')} "
                f"| {row.get('probes', '')} | {row.get('patch', '')} "
                f"| {row.get('router', '')} | {row.get('toks', '')} "
                f"| {row.get('class', '')} | {row.get('blocker', '')} |"
            )
        lines.append("")
    if no_code_turns:
        lines.append(
            f"No-usable-code turns: {len(no_code_turns)} "
            f"({'; '.join(no_code_turns[:8])})"
        )
    if harness_crashes:
        lines.append(f"Harness crashes: {harness_crashes}")
    for rl in restart_lines:
        lines.append(f"Restart: {rl}")
    if outcome is not None:
        lines.append(
            f"Outcome: ok={outcome.get('ok')} "
            f"iterations={outcome.get('iterations')} "
            f"best_exists={outcome.get('best_path_exists')}"
        )
    return "\n".join(lines) + "\n"


# Default context window. This value is ALSO the denominator for the
# compaction pressure check (prompt_tokens / num_ctx). The old 32K default
# made that ratio exceed 1.0 within a couple of feedback turns, so the lossy
# state-anchor compaction fired EVERY turn and shredded the playbook + prior
# user-feedback + the model's view of the file (observed 2026-05-29
# fighting-game trace: pressure 1.19 at 8 messages on a 200K-context model).
# 100K is the speed/headroom sweet spot: observed coder prompts run ~10-45K
# even deep into a feedback session, so pressure stays well under the 0.70
# compaction trigger (full history retained), while keeping Ollama KV-cache /
# prefill cost far lower than a 250K reservation. Raise toward 200K for very
# long sessions, or lower on tight-VRAM hosts, via CODING_BOX_NUM_CTX / /ctx.
DEFAULT_NUM_CTX = 100_000
MIN_NUM_CTX = 8192
MAX_NUM_CTX = 262_144

_NUM_CTX_PRESETS: dict[str, int] = {
    "default": DEFAULT_NUM_CTX,
    "32k": DEFAULT_NUM_CTX,
    "64k": 65_536,
    "100k": 100_000,
    "131k": 131_072,
    "200k": 200_000,
    "262k": MAX_NUM_CTX,
    "full": MAX_NUM_CTX,
    "max": MAX_NUM_CTX,
    "native": MAX_NUM_CTX,
}


def default_num_ctx() -> int:
    """Resolve Ollama num_ctx: env CODING_BOX_NUM_CTX overrides the default."""
    raw = (os.environ.get("CODING_BOX_NUM_CTX") or "").strip()
    if not raw:
        return DEFAULT_NUM_CTX
    return parse_num_ctx_arg(raw)


def parse_num_ctx_arg(arg: str) -> int:
    """Parse /ctx or env values: 100000, 100k, 262k, full, native, …"""
    s = (arg or "").strip().lower().replace("_", "").replace(",", "")
    if not s:
        raise ValueError("empty num_ctx")
    if s in _NUM_CTX_PRESETS:
        return _NUM_CTX_PRESETS[s]
    if s.endswith("k"):
        try:
            n = float(s[:-1])
        except ValueError as e:
            raise ValueError(f"invalid num_ctx: {arg!r}") from e
        v = int(n * 1000)
    else:
        try:
            v = int(s)
        except ValueError as e:
            raise ValueError(f"invalid num_ctx: {arg!r}") from e
    if v < MIN_NUM_CTX or v > MAX_NUM_CTX:
        raise ValueError(
            f"num_ctx {v} out of range [{MIN_NUM_CTX}, {MAX_NUM_CTX}]"
        )
    return v


def _parse_ollama_keep_alive_env() -> float | str:
    """Ollama chat keep_alive override.

    Default -1 keeps local models resident between feedback turns. Reject 0
    because it immediately evicts after every call and recreates the stall.
    """
    raw = (os.environ.get("OLLAMA_KEEP_ALIVE") or "").strip()
    if not raw:
        return -1
    low = raw.lower()
    if re.fullmatch(r"0+(?:\.0+)?(?:ms|s|m|h)?", low):
        raise ValueError(
            "OLLAMA_KEEP_ALIVE=0 would unload the model after every call; "
            "use -1, 10m, 1h, or unset it for the default -1."
        )
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if numeric == 0:
        raise ValueError(
            "OLLAMA_KEEP_ALIVE=0 would unload the model after every call; "
            "use -1, 10m, 1h, or unset it for the default -1."
        )
    return int(numeric) if numeric.is_integer() else numeric


class GameAgent(
    PromptBuildingMixin,
    CompactionMixin,
    StreamMaterializeMixin,
    AssetGenerationMixin,
    CriticMixin,
    GateProcessingMixin,
    FeedbackRoutingMixin,
    ProbeHandlingMixin,
    MemoryRetrievalMixin,
):
    """Drives the planning/coding/critique loop. One instance per session."""

    def __init__(
        self,
        model: str | None = None,
        out_path: Path | None = None,
        browser: LiveBrowser | None = None,
        max_iters: int = 6,
        *,
        # Resolved LLM backend (Ollama or MLX). Drivers (chat.py, coder.py)
        # build it via `make_backend(detect_backend(...))`. When omitted,
        # we construct a legacy OllamaBackend from `model` so older callers
        # (and unit tests that pass `model="stub"` without ever streaming)
        # keep working unchanged.
        backend: Backend | None = None,
        # Default 1: single candidate per turn. Best-of-N was briefly
        # default-3 (Phase 0.6 after the 2026-05-22 chess trace) but
        # combined with the multi-slot autopin it oversubscribed the
        # 4-GPU box on first build (2026-05-23). Opt in with --best-of-n
        # when you actually have spare slots staged.
        best_of_n: int = 1,
        # Ollama context window. qwen3.6:27b/35b natively supports 128K+,
        # gpt-oss supports 128K — at 8K we were truncating mid-<assets>
        # block on long planning turns (see games/traces/make-a-small-
        # first-person-shoo_20260506_222042). Bumped default to 32768
        # which fits the system prompt + plan + first-build with room
        # for several feedback iterations before structured compaction.
        # Ollama context window. Current-gen local coding models
        # (Qwen3.6-27B, DeepSeek V4 Flash, GLM 5.1, MiniMax M2) all
        # ship with 256K native context. The harness ceiling matches
        # so a long multi-iter session — full HTML + history +
        # focused slice + screenshots-as-text — never gets clipped
        # by us before the model. Was 65536; the cap was leaving
        # ~48 KB output headroom which iter 1 of a complex game
        # could clear, but later iters with reply history accumulated
        # would hit the wall (the original 16384 cap caused exactly
        # this in classic-doom 20260512_101944).
        # Memory cost note: Ollama KV-cache scales linearly with
        # num_ctx — 100K default (~4 GB KV on a 27B) fits 48 GB-class
        # multi-slot boxes; 262K is ~10 GB. Override via CODING_BOX_NUM_CTX,
        # chat /ctx, or coder --num-ctx. Changing num_ctx forces an Ollama
        # reload — preload with `ollama run --ctx-size N <model>` first.
        num_ctx: int | None = None,
        # Per-stream stall budget (no-activity quiet window) — single
        # generous value for every model. Activity-aware MLX watchdog
        # bumps this on prefill progress and every emitted token,
        # so 10 minutes of *silence* is what trips it, not 10 minutes
        # of compute. No model name parsing, no bracket table.
        stall_seconds: float = 600.0,
        # Hard ceiling per stream. 30 minutes covers any realistic
        # generation including pathological MoE prefills + multi-KB
        # html_file rewrites. Runaway generation is still bounded.
        overall_seconds: float = 1800.0,
        memory_root: str | Path = "memory",
        # Optional path to an existing HTML file to start from. When set,
        # the agent skips memory-skeleton retrieval and uses this file as
        # the baseline; the model is asked to ADAPT it (via patches) to
        # the user's goal rather than build from scratch.
        seed_file: str | Path | None = None,
        # Which prompt module to load. "v1" = prompts_v1.py (production
        # default). Future revisions should ship as prompts_v2.py /
        # prompts_v3.py / etc. and pass `prompt_version="v2"`. The
        # retired v0 (prompts.py) was deleted; passing "v0" or any
        # missing module raises ImportError immediately.
        prompt_version: str = "v1",
        # How to seed the first build. "retrieve" (default) = best-match
        # skeleton from past wins; "default" = always use the bundled
        # canvas_basic skeleton (good for tune mode — measures from-scratch
        # ability); "none" = no skeleton, model writes blank-slate.
        skeleton_mode: str = "retrieve",
        # How many playbook bullets to inject per render.
        playbook_top_k: int = 6,
        # When True, increment helpful/harmful counters on the playbook
        # bullets that were active during each iteration based on the
        # outcome. Off by default so tune-mode A/B experiments can
        # compare a frozen playbook. Flip on once a baseline is locked.
        playbook_writeback: bool = False,
        # ----- behavior bundles (independently testable) -----------------
        # Continue.dev-style assistant prefill: open the model's turn
        # with `<plan>\n` (Phase A) or `<diagnose>\n` (fix turns) so
        # format compliance is forced. Cost: ~0 tokens, just changes
        # request shape.
        use_prefill: bool = False,
        # Always attach the latest screenshot to Phase C self-critique
        # when the model is a VLM. Today the screenshot is only attached
        # on FAIL turns; this extends it to clean+done so polish bugs
        # ("ship is half off-canvas", "score invisible") stop slipping
        # past CONFIRM_DONE.
        use_vlm_critique: bool = False,
        # Automatic stuck best-of-2 escalation (cap 2/session). Default
        # OFF — on slow single-GPU MLX it doubles wall time with little
        # gain when failures are structural. Opt in via /bestof on (TUI)
        # or --stuck-bon (coder.py). Explicit --best-of-n N is separate.
        stuck_bon_enabled: bool = False,
        # Capture two screenshots — t=startup and t=after-input-press —
        # so the model can see motion/state-change for fix turns. Costs
        # one extra Playwright screenshot call per iter.
        use_double_screenshot: bool = False,
        # On detected complex first-builds, do a 2-call architect/editor
        # split: model-1 produces an English plan describing data
        # structures + render layers, model-2 (same Ollama model, fresh
        # turn) writes the code. Aider's pattern. Doubles wall-clock on
        # the FIRST iter only — gated on a complexity heuristic so
        # simple goals stay one-shot.
        use_architect_split: bool = False,
        # Prompt-size + retrieval-budget trim. "auto" (default) maps
        # to "small" — the lean ~5 KB schema biased for mid-size local
        # LLMs and one-shot strength on simple games. Pass "large"
        # explicitly when running a frontier-tier model that can absorb
        # the full schema. NEVER hardwire detection by model name —
        # the user rotates models constantly.
        model_class: str = "auto",
        # Stop-Losing-To-OneShot Track A: when iter 1 of a session
        # ends with score < restart_score_threshold, throw the
        # session away and restart from scratch. Up to restart_n
        # total attempts. Best-by-score wins. Mid-size LLMs one-shot
        # small games well — restarting beats polishing a stinker.
        # restart_n=1 disables the wrapper (default keeps existing
        # behavior so callers that don't opt in are unchanged).
        restart_n: int = 1,
        restart_score_threshold: float = 60.0,
        backend2: Backend | None = None,
        model2_role: str | None = None,
        backend3: Backend | None = None,
        model3_role: str | None = None,
    ):
        # Backend resolution. Legacy callers pass `model="..."` without
        # `backend=` (notably the unit-test fixtures that never stream);
        # build a default OllamaBackend in that case so behavior is
        # identical to before this refactor.
        if backend is None:
            if not model:
                raise TypeError(
                    "GameAgent requires either `backend=` (resolved via "
                    "backend.detect_backend()) or `model=<tag>` for legacy callers"
                )
            backend = make_backend(BackendInfo(
                name="ollama", model=model,
                source="legacy: GameAgent(model=...) without backend=",
                endpoint=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            ))
        self._backend: Backend = backend
        self._backend2: Backend | None = backend2
        self._model2_role: str | None = model2_role
        self._backend3: Backend | None = backend3
        self._model3_role: str | None = model3_role
        self._model2_activity: str = "idle"
        self._model3_activity: str = "idle"
        # Role passed to the in-flight _stream() call — chat.py routes
        # token counters to Model 2/3 lines when this is critic/architect.
        self._last_stream_role: str = "coder"
        self.out_path = Path(out_path)
        self.browser = browser
        self.max_iters = max_iters
        self.best_of_n = max(1, best_of_n)
        self.num_ctx = num_ctx if num_ctx is not None else default_num_ctx()
        self.stall_seconds = stall_seconds
        self.overall_seconds = overall_seconds
        self._ollama_keep_alive: float | str = _parse_ollama_keep_alive_env()
        self.seed_file: Path | None = Path(seed_file) if seed_file else None
        self._messages: list[dict] = []
        self._pending_feedback: list[str] = []
        # Texts queued by the AGENT itself (not the user) via
        # _queue_internal_feedback. Detectors that must only fire on
        # genuine user feedback (unhonored-asset-request) skip these.
        self._internal_feedback_texts: set[str] = set()
        # Animation frames that came back near-identical to their from_image
        # parent (dead animation: the limbs never moved). name -> parent_delta.
        # Populated during asset generation; cleared when the frame is later
        # regenerated distinctly. While non-empty it HARD-BLOCKS <done/> via
        # _apply_dead_animation_check_to_report — a sliding static sprite is
        # not the animation the user asked for.
        self._dead_anim_frames: dict[str, float] = {}
        # True once the model has declared any from_image-derived (animation)
        # frame this session — a signal-driven "the user wants motion" flag
        # the visual critic uses to add a context-specific animation question.
        self._declared_anim_frames: bool = False
        # An explicit "generate new art" request that the model has NOT yet
        # honored with an <assets> block. Re-asserted each turn (capped) until
        # the model emits one — a model distracted by a blocker, or in
        # raw-feedback mode, otherwise replies with code/diagnose and the new
        # sprite never gets generated (here-s-a-tight-test 20260530).
        self._unhonored_asset_request: str | None = None
        self._asset_reprompt_count: int = 0
        self._feedback_deferred_last_turn: bool = False
        # LLM Feedback Router (chess-trace fix 2026-06-22): a small LLM
        # call interprets the pending user feedback batch into a routing
        # decision (intent + honor-now + allow-assets) that OVERRIDES the
        # brittle regex classifiers below when present. The regex helpers
        # (`_feedback_is_art_change` etc.) stay as the offline / parse-fail
        # fallback (see `_route_user_feedback_llm`). `_feedback_route` is
        # the most-recent decision dict (or None); `_feedback_route_key`
        # is the cache key (feedback-hash + asset-count + last_report_ok)
        # so a re-flush in the same turn reuses it instead of re-calling.
        self._feedback_route: dict | None = None
        self._feedback_route_key: str | None = None
        # Most recent feedback batch consumed by _flush_user_injections.
        # Used to restore feedback if a stream fails before any assistant
        # reply lands (extension fallback / backend failure path).
        self._last_drained_feedback: list[str] = []
        # Set by `_flush_user_injections` when the user locks the turn
        # ("no code changes", "only X"). The fix-mode prompt reads this
        # and suppresses the failing-test "fix these" framing so the
        # local model isn't asked to balance "ignore test failures" vs.
        # "fix these test failures" in the same prompt (DK trace
        # 2026-05-15). Self-clears after each fix-prompt build.
        self._scoped_change_active: bool = False
        # Per-turn scoped routing metadata persisted when feedback is drained.
        # Applies deterministic guards before materialization:
        #   - route mode: "single_patch" | "media_only"
        #   - max patch blocks
        #   - existing-name lock for media-only turns
        #   - optional scoped-check probe requirement (behavior asks)
        self._scoped_constraints: dict[str, Any] | None = None
        # Fix B (seed-edit escape hatch): count consecutive scoped violations
        # so a seed-edit lock can stop thrashing and apply usable code after
        # 2 strikes. Reset on lock clear and after any successful materialize.
        self._scoped_violation_streak: int = 0
        # Routing decisions saved by _flush_user_injections so _stream
        # can emit a single turn_contract trace event with allowed/
        # forbidden tags and classifier flags. None until first turn.
        self._last_turn_contract: dict[str, Any] | None = None
        # One-turn scoped-check keywords carried into the next test report.
        # When non-empty, we require at least one matching probe pass.
        self._pending_scoped_check_keywords: list[str] = []
        # Probe eval-error recovery: distinguish broken probes from
        # game-false probes, quarantine repeated eval-error probes, and
        # avoid re-feeding the same harness-side SyntaxError forever.
        self._probe_eval_error_streak: dict[str, int] = {}
        self._probe_eval_error_shape_streak: dict[str, int] = {}
        self._probe_names_ever_passed: set[str] = set()
        self._pending_probe_quarantine_notices: list[str] = []
        # When quarantine empties the entire model-authored probe set, the
        # build would otherwise ship "clean" with ZERO behavioral self-checks
        # (battlezone 20260622 iter 4: 7/7 probes syntax-quarantined ->
        # probes_total:0, ok:true). We gate that clean ship for a BOUNDED
        # number of iters (cap below) so a model that simply cannot author a
        # parseable probe does not loop forever (mirrors the impossible-probe
        # downgrade's anti-stuck contract). Counter resets each session.
        self._all_probes_quarantined_gate_used: int = 0
        # Partial-quarantine gate (serial10 chess game 5): when SOME probes
        # survive but a behavioral probe was syntax-quarantined on a
        # recipe-matched game, the surviving probes can report a clean pass
        # that masks the dead gate. Counter (bounded by
        # _PARTIAL_QUARANTINE_GATE_CAP) blocks the clean ship until a valid
        # replacement probe is emitted. Per-session (new GameAgent per run).
        self._partial_quarantine_gate_used: int = 0
        # Last few raw feedback strings for repeated-request detection.
        self._recent_feedback_texts: list[str] = []
        # Phase 0.2 — keyword signatures of feedback items that have been
        # deferred behind a blocker. When the same intent shows up for the
        # 3rd time, escalation kicks in and bypasses the deferral so the
        # user doesn't get silently swallowed across turns (the chess
        # 2026-05-22 trace had 3 deferrals of the same ask).
        self._recent_deferred_signatures: list[str] = []
        self._pending_answer: str | None = None
        # A2: short coaching strings queued when the deliberation detector
        # aborts the last stream or another agent-side guard wants the
        # next user-turn to carry a one-shot corrective note. Drained
        # by _flush_user_injections before any base message is appended.
        self._pending_coaching: list[str] = []
        # Point-and-click VLM grounding: {bg_name: {object: {cell,nx,ny}}}.
        self._pointclick_grounding: dict[str, dict[str, dict[str, Any]]] = {}
        self._pointclick_grounding_block: str = ""
        # A5: track consecutive same-signature errors to drive runtime-
        # state probe coaching.
        self._last_mistake_sig: str | None = None
        # Hard-gate flag (Item 2, plan 20260514_175012): when
        # _repeat_sig_streak hits 3 AND _subsystem_hint matches, this
        # is set to the matching hint dict. The next user-turn
        # assembly substitutes a <question>-only prompt and clears
        # the flag. Reset to None whenever a clean iter resets the
        # streak (see `_repeat_sig_streak = 0` sites).
        self._force_question_subsystem: dict | None = None
        self._repeat_sig_streak: int = 0
        # Fix #3 (classic-doom 20260512_111015): on the FIRST turn after
        # fresh user feedback is drained, exempt a single <html_file>
        # rewrite from the snapshot_n>=1 rejection. Multi-issue feedback
        # (gun + mouse + powerups + demons) is often easier to address
        # holistically than as 4 fragile patches; that's exactly when
        # the model wants a rewrite and the existing policy was burning
        # iters rejecting it. Cleared after one materialization
        # attempt — does NOT persist across extension turns.
        self._allow_one_rewrite: bool = False
        # Phase 1b: set when a continuation turn loaded a corrupt on-disk
        # baseline. Lets the fix-prompt + turn_contract surface the fault
        # to the model so it can rewrite cleanly instead of patching garbage.
        self._continuation_baseline_corrupt: bool = False
        # Phase 3: set by the exit-decision turn when the model emitted
        # <done/> while the on-disk file was still structurally broken.
        # Overrides session_outcome so a confident-but-wrong <notes> handoff
        # can't masquerade as success.
        self._exit_done_over_broken_file: bool = False
        self._user_force_done = False
        # Plan-only loop detector: counts consecutive iterations where the
        # model emitted <plan> but neither <patch> nor <html_file>. The
        # "no usable code" fallback escalates the next user-turn prompt
        # when this hits 2, and the loop-break message is sent at >=2.
        # Resets to 0 whenever a reply successfully materializes code.
        self._consecutive_plan_only: int = 0
        # Criteria lines that no probe references — surfaced at Phase A
        # parse so the gap is visible upfront. Empty when probes cover
        # everything or when criteria/probes are missing.
        self._planning_coverage_gaps: list[str] = []
        # asyncio.Event that the MLX backend polls between tokens. Set by
        # request_done() so Ctrl-D in the TUI actually stops mid-stream,
        # not just at the next iter boundary. Created lazily on first use
        # so the agent can be constructed outside a running event loop
        # (some tests instantiate it that way).
        self._stop_event: asyncio.Event | None = None
        # Step-mode (Stop-Losing-To-OneShot todo #1): when True, the iter
        # loop pauses BETWEEN iterations and waits for explicit user input
        # before querying the model again. Strictly stronger than any
        # harness check for mid-tier models — the user becomes the
        # verifier between iters. Toggled via /wait (chat.py) or --step
        # (coder.py). Off by default; existing autonomous behavior is
        # preserved when the flag stays False.
        self._step_mode: bool = False
        # Wikipedia research lookup REMOVED (2026-06-24): empirical 0/10 hit
        # rate on common game goals; the curated opening library in memory/
        # (outlines, components, playtests, audits) is the source of grounding
        # instead. No plan-time network lookup, no <reference> block.
        # Auto-step-mode (DK trace 20260513_122154): when the first
        # iter fails, the harness auto-arms /wait so the user can
        # intervene before the cascade. Fires exactly once per session;
        # `_step_auto_disabled` is set by set_step_mode(False) so a
        # subsequent user `/wait off` permanently opts out of the auto-
        # arm for the rest of the run.
        self._auto_step_armed: bool = False
        self._step_auto_disabled: bool = False
        # Master switch for auto-arming step mode on first failed iter.
        # Drivers can disable this for uninterrupted AUTO sessions.
        self._auto_step_on_failure: bool = True
        # Released by signal_step_continue() to unblock a step-mode wait
        # without adding any user feedback. add_user_feedback also
        # unblocks (via has_pending_user_input becoming True).
        self._step_continue: bool = False

        # Game/session basename comes from out_path and may intentionally be
        # reused for seed/continuation workflows so the live HTML, best.html,
        # assets, and sounds keep the same canonical names.
        #
        # Trace artifacts must NOT reuse that bare basename: a later seeded
        # run against the same game can have a different user goal. Mixing a
        # new .log/.conversation.md with an old appended .jsonl makes traces
        # untrustworthy. Give every GameAgent construction its own artifact id
        # while keeping `_session_id` stable for game assets.
        basename = self.out_path.stem or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = basename
        self._artifact_id = (
            f"{basename}__run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        out_dir = self.out_path.parent
        self.trace_path: Path = out_dir / "traces" / f"{self._artifact_id}.jsonl"
        self.snapshots_dir: Path = out_dir / "snapshots" / self._artifact_id
        self.best_path: Path = out_dir / f"{basename}.best.html"
        self.conversation_path: Path = (
            out_dir / "traces" / f"{self._artifact_id}.conversation.md"
        )
        self._previous_report_ok: bool | None = None
        # Stop-Losing-To-OneShot todo #3 — track the full previous report
        # so the <done/> gate can ask richer questions than just "ok?".
        # Specifically: criteria_uncovered (todo #2 coverage check),
        # page_errors, and per-probe ok. None until the first iter ran.
        self._previous_report: dict | None = None
        # Stop-Losing-To-OneShot todo #4 — auto-revert grants bonus iters
        # (capped per run) so a regression that gets auto-rolled back
        # doesn't punish the user's max_iters budget. Reset to 0 at the
        # top of each run().
        self._iter_budget_bonus: int = 0
        # Per-iter flag: was mid-session media regenerated this turn?
        # Set inside _maybe_generate_assets_and_sounds when at least one
        # asset or sound was successfully produced. Read after
        # materialize to decide whether a "no usable code" turn was
        # purely preparatory (model emitted <assets> but no code), in
        # which case we grant a bonus iter rather than punishing the
        # user's max_iters budget — the regen work is real progress.
        # DK trace 20260513_153626 iter 3 burned a slot on assets-only
        # emission and never got to use the new sprites in code; this
        # flag prevents that loss. Reset at the top of every iter.
        self._media_regenerated_this_iter: bool = False
        # A2 — streak of consecutive clean iterations. <done/> is only
        # honored when streak >= 2, so the model can't ship after a
        # single passing iter. Resets to 0 on any failed test, including
        # probe failures (see A1). awaiting_confirm bypasses this check
        # because the post-self-critique <confirm_done/> is a separate
        # signal that already represents "model verified twice."
        self._consecutive_clean_iters: int = 0
        self._min_clean_streak_to_ship: int = 2
        self._snapshot_n: int = 0
        self._is_vlm: bool | None = None
        self._next_image_bytes: bytes | None = None
        self._fix_mode: bool = False
        self._memory = GameMemory(root=memory_root)
        self._memory.ensure()
        self._playbook = Playbook(root=memory_root)
        self._playbook.ensure()
        self._playbook_top_k = max(0, int(playbook_top_k))
        self._playbook_writeback = bool(playbook_writeback)
        # Bullet IDs retrieved on the most recent prompt render — used by
        # the writeback feedback loop to credit/blame them after the next
        # test result.
        self._active_bullet_ids: list[str] = []
        self._active_opening_book_recipes: list[dict] = []
        self._active_skeleton: str | None = None
        # Visual playtest recipe id matched for this session, plus the
        # names of any auto_probes injected into self._probes by it.
        # Surfaced in the TUI status panel so the user can see WHICH
        # mechanism recipe is guiding the VLM critic + which extra
        # state-shape assertions are running each iter.
        self._active_visual_playtest_recipe_id: str | None = None
        self._active_visual_playtest_auto_probes: list[str] = []
        # True when the matched recipe declares `still_frame` (laserdisc /
        # cutscene cadence). Used to downgrade the FROZEN-AT-IDLE warning to a
        # neutral advisory so the model is not coached to add an unrequested
        # breathing/bob motion to a deliberately still-frame game (trace
        # 20260613_213711, Opus 4.8 iter 5).
        self._active_visual_playtest_still_frame: bool = False
        self._skeleton_mode = skeleton_mode
        self._prompt_version = prompt_version
        self._p = self._load_prompt_module(prompt_version)
        self._last_diagnose: str | None = None
        self._stuck_streak: int = 0
        # Capability-round item 2 — polish-phase state. `_polish_turns_used`
        # counts polish prompts sent this session (cap _POLISH_TURN_CAP);
        # `_polish_pending` marks "the turn now streaming is a polish turn"
        # so a regression on its report ends the polish phase;
        # `_last_critic_note` keeps the visual critic's latest finding for
        # the polish prompt.
        self._polish_turns_used: int = 0
        self._polish_pending: bool = False
        self._last_critic_note: str | None = None
        # Prefill-broken latch (2026-06-12): set True after the visual
        # critic's "Q1: " assistant prefill yields an empty completion
        # (some backends don't continue prefills — trace 20260612_004616
        # wasted one VLM call per iteration, 13/13). Once latched, the
        # critic skips the prefill for the rest of the session. Not
        # reset between restart attempts — the backend doesn't change.
        self._critic_prefill_broken: bool = False
        # Fix-round item 6 — critic action-frame fairness state.
        # `_no_action_frame_advisory_sent`: the one-per-attempt deterministic
        # advisory replacing repeated unanswerable action-frame failures.
        # `_current_critic_payload_fp` / `_suppressed_critic_payload_fp`:
        # payload-fingerprint dedupe so an identical critic payload whose
        # critique was already suppressed skips the VLM call entirely.
        self._no_action_frame_advisory_sent: bool = False
        self._current_critic_payload_fp: str | None = None
        self._suppressed_critic_payload_fp: str | None = None
        # Capability-round item 5 — count of stuck best-of-2 escalations
        # used this session (cap _STUCK_BON_ESCALATION_CAP).
        self._stuck_bon_escalations: int = 0
        # Iterations left after the current one (set each loop pass); the
        # polish branch only fires when another iter exists to test it.
        self._iters_remaining: int = 0
        # DK trace 20260513_185815 burned 7 consecutive turns on the same
        # broken format (model emitted <patch> inside a markdown fence,
        # harness rejected with generic "no <patch> or <html_file>" and
        # the model retried the same shape). `_format_stuck_streak`
        # counts consecutive turns where _materialize returned None
        # without the model having emitted a parseable structure. At
        # streak >= 2 we (a) escalate the prompt to require a full
        # <html_file>, and (b) optionally invoke a format-doctor
        # subagent (see _run_format_doctor). Reset to 0 on any
        # successful materialize.
        self._format_stuck_streak: int = 0
        # Streaming-abort flags from the most recent _stream() call.
        # Used by the format-rejection branch to escalate the doctor:
        # a `looped` abort means the model was emitting the same 1-2
        # short lines on repeat — strong evidence of confusion, fire
        # the doctor at streak=1 instead of waiting for streak=2.
        # DK trace 20260514_104131 post-seed iter 2 hit exactly this:
        # repetition loop + bare_markers rejection, session ended
        # before streak reached 2.
        self._last_stream_looped: bool = False
        self._last_stream_stalled: bool = False
        self._last_stream_deliberated: bool = False
        # True when the most recent stream produced ZERO non-empty
        # content pieces past the wall-clock floor — the ollama_io /
        # MLX silent-stream guard fired. Indicates the model burned
        # tokens through a reasoning/thinking channel that surfaces
        # as empty `content`. Tracked separately so the recovery
        # prompt can call out "start with an opening tag, skip the
        # silent reasoning preamble" instead of generic stall coach.
        self._last_stream_silent: bool = False
        # Guard-abort AgentEvents queued inside _stream() for the TUI.
        # _stream is not an async generator — run() must drain and yield.
        self._pending_stream_ui_events: list[AgentEvent] = []
        # Cross-turn patch-SEARCH-failure memory. Stores fingerprints
        # (sha1 of normalized first 80 chars) of every SEARCH block that
        # failed to apply on the most recent retry turn. When the next
        # turn re-emits a SEARCH that matches one of these fingerprints,
        # the retry prompt marks it [REPEATED FAILURE] so the model
        # gets a clearer signal than the generic "re-read the file"
        # nudge. Motivating trace: doom 2026-05-23 extensions 1/2/3
        # where the same `spriteNames=['imp_idle'...]` SEARCH failed
        # three turns in a row because the FIRST patch already changed
        # those lines and the model didn't read the updated file.
        # Updated by the no_usable_code retry branch in run(); set is
        # bounded by the per-turn failure count, no explicit cap needed.
        self._last_failed_patch_anchors: set[str] = set()
        # Trace 20260612_225857: repeated bracket-rejected patches against
        # the same function burned feedback turns. Count bracket rejects so
        # the next prompt can switch into tiny source-slice surgery mode.
        self._patch_bracket_reject_streak: int = 0
        # Cross-turn identical-reply detector. Wolfenstein 2026-05-24
        # trace burned 5 consecutive iters where the model emitted
        # bit-identical 7838-token replies and the agent kept sending
        # the same generic "no <patch> or <html_file>" fallback. We
        # fingerprint each reply that triggers `no_usable_code` (sha1
        # of normalized first 4 KB) and remember the previous one.
        # When the SAME fingerprint fires twice in a row we know the
        # loop is genuine and standard fallback won't move the model
        # — escalate unconditionally to format_doctor + scope-reduction
        # coaching. Reset on any successful materialize so a real
        # cycle of "fail → fix → fail differently" isn't punished.
        self._last_no_usable_code_fingerprint: str | None = None
        # Context-pressure detector. Wolfenstein 2026-05-24 trace had
        # prompt_tokens pinned at 32711 / num_ctx=32768 across the
        # whole stuck-loop tail — the model literally had ~50 tokens
        # of headroom to emit a complete <html_file>. Universal fix:
        # when stream_done reports prompt_tokens >= 85% of num_ctx,
        # set this flag so the NEXT fix_instruction call omits the
        # inlined CURRENT FILE block and coaches the model to send
        # a minimal patch only. Flag is one-shot (cleared after the
        # mitigated prompt is built). `_context_pressure_streak`
        # tracks consecutive high-pressure turns so the trace event
        # fires once per streak rather than every turn.
        self._context_pressure_pending: bool = False
        self._context_pressure_streak: int = 0
        # Last observed coder prompt size, for token-aware compaction. 0.0
        # until the first coder turn reports usage — _prune_messages then
        # falls back to the message-count safety cap only.
        self._last_prompt_tokens: int = 0
        self._last_prompt_pressure: float = 0.0
        # Fix-round item 3: one-shot flag set by the silent-stall handler so
        # the NEXT _prune_messages forces a structured compaction instead of
        # rebuilding the same prompt that just produced 0 tokens for 180s+.
        self._force_compact_after_stall: bool = False
        # Mid-session assets generated but not yet referenced in HTML PATHS.
        self._new_assets_not_in_html: set[str] = set()
        # Dead-first-build detector. Wolfenstein 2026-05-24 trace iter 2
        # loaded a file but RAF never fired AND the input smoke test
        # registered zero state/canvas delta — the file is structurally
        # broken. Patches on top of a dead first build deepen the hole.
        # When detected on iter <= 2, the next fix turn uses a
        # scope-reduction prompt ("ship a smaller intentionally minimal
        # html_file") instead of diagnose-then-fix. Counter caps
        # recoveries per attempt — after 2, the agent ends the attempt
        # so the restart loop can apply a fresh seed/recipe.
        self._dead_first_build_pending: bool = False
        self._dead_first_build_recoveries: int = 0
        self._dead_first_build_abort_attempt: bool = False
        # Per-attempt counters used by the restart loop to compute a
        # failure signature. When two consecutive attempts hit the SAME
        # signature, the next attempt's plan_instruction is given
        # `force_minimal_first_build=True` so the model writes a
        # smaller first build rather than re-attempting the same
        # ambitious one. Universal: keys on observable counter shape,
        # not goal text. All three reset in `_reset_attempt_state`.
        self._identical_reply_loops_this_attempt: int = 0
        self._format_rejections_iter1_this_attempt: int = 0
        # Carries the previous attempt's signature into the next
        # attempt's run_with_restarts loop so plan_instruction can
        # detect "same failure shape twice in a row". None on the
        # first attempt of a session.
        self._prev_attempt_signature: str | None = None
        self._force_minimal_first_build: bool = False
        # Visual-critic cross-turn dedup. Bounded deque of fingerprints
        # (sha1 of normalized first 120 chars) for critic notes injected
        # in the last few turns. Before queueing a new critic note we
        # check this; matches are suppressed (with a
        # `coaching_suppressed_repeated` trace) so the same observation
        # doesn't bloat the prompt iter after iter. Motivating trace:
        # doom 2026-05-23 where the critic flagged "low-resolution wall
        # textures" in iters 1, 3, 5, 7 — each repeat added ~400 chars
        # of prompt noise for zero new information.
        from collections import deque as _deque
        self._recent_critic_note_fingerprints: _deque = _deque(maxlen=3)
        # Per-session scoped-classifier overrule counter. When the
        # model picks a different output mode than the classifier
        # expected (patch when classifier said media_only, etc.), this
        # ticks up. At threshold (2) we auto-flip
        # `_use_feedback_directives = False` for the rest of this
        # session — the classifier has demonstrated it's misroutíng
        # the user's typed feedback, and per the standing rule
        # "agent must beat zero-shot" we step out of the way. Resets
        # to 0 on /new (since this object is reconstructed). The
        # threshold is intentionally low so it kicks in fast on
        # iter-position-tweak sessions like doom 2026-05-23 where the
        # model overruled the classifier twice in a row on
        # pixel-shift feedback ("move gun 75 pixels right, no asset
        # changes" → classifier=media_only, model emitted a code
        # patch).
        self._classifier_overrule_count: int = 0
        self._classifier_auto_disabled: bool = False
        # When the previous stream looped, which RepetitionDetector window
        # fired and (if captured) what line was looping. Surfaced in the
        # format-rejection fallback so coaching can name the failure shape
        # back to the model (donkey-kong 20260516_142445 iter 1).
        self._last_stream_loop_kind: str | None = None
        self._last_stream_loop_line: str | None = None
        # Probe-sanity lint findings (tautological / unassigned-property
        # reads). Refreshed each iter; surfaced into the diagnose-then-
        # fix prompt when the iter fails. Empty list when probes are
        # healthy. See _lint_probes / _probes_referencing_unassigned_props.
        self._probe_lint_findings: list[dict] = []
        # Tracked separately from `stalled` (which the backend sets on
        # crash too). Format-doctor early-escalation reads this so it
        # doesn't burn another stream trying to "reformat" 30+ KB of
        # mid-stream-crash wreckage. The fix is to start the next iter
        # fresh, not to coax the doctor through the same context.
        self._last_stream_crashed: bool = False
        # Sticky for session_outcome.backend_crashed — offline credit must not
        # treat infra/backend deaths as playbook-harmful (trace-schema audit).
        self._session_backend_crashed: bool = False
        # Tracks the most recent iteration where _materialize wrote a
        # file to disk vs the most recent iteration where the verifier
        # ran. If they diverge at end-of-run we run one final test on
        # the last shipped code (DK trace 20260513_185815 ended with
        # correct code in Turn 14 that was never run because the loop
        # hit its budget).
        self._last_materialized_iter: int = 0
        self._last_tested_iter: int = 0
        # Most recent test report (full dict). Used by the end-of-run
        # final-iter test guarantee to fold the late test into the
        # outcome record.
        self._last_test_report: dict | None = None
        # Persistence counter for harness `warnings`. Each non-empty
        # warning string in a test report counts as +1 per consecutive
        # iter; absent strings reset to 0. When the count reaches
        # _WARNING_COMPACT_THRESHOLD (3rd consecutive appearance), the
        # model-facing rendering replaces the body with a one-line
        # collapsed form. The full warning text stays in `report` (and
        # therefore in the trace JSONL) for postmortem.
        # Evidence: fighing-game trace 20260519_153115 — 8 unused-asset
        # warnings (~1.6 KB) repeated verbatim every iter for 4 iters
        # while the model stopped reacting. Carrying that text into
        # the working set after it has clearly been seen wastes
        # context. Domain-neutral by construction — the dedup is keyed
        # by exact-string equality, not by warning content.
        self._warning_persistence: dict[str, int] = {}
        self._use_prefill = bool(use_prefill)
        self._use_vlm_critique = bool(use_vlm_critique)
        self._stuck_bon_enabled = bool(stuck_bon_enabled)
        # Auto-staff flag (2026-05-21): True when _use_vlm_critique was
        # flipped by the auto-enable hook in _detect_vlm rather than by an
        # explicit /vlm-critique on. Surfaced in the TUI status panel so
        # users see "[auto]" and know they can override.
        self._vlm_critique_auto: bool = False
        self._use_double_screenshot = bool(use_double_screenshot)
        self._use_architect_split = bool(use_architect_split)
        # Auto-staff flag for architect split (same shape as above).
        self._architect_split_auto: bool = False
        # todo #6 — resolve "auto" via simple substring-match. Adding a
        # name to _MID_MODEL_TAGS is a one-line opt-in for new families.
        self._model_class: str = (
            model_class if model_class in ("small", "mid", "large")
            else self._classify_model(model)
        )
        self._trace({"kind": "model_class_resolved", "model": model, "model_class": self._model_class})
        # Lean system-prompt mode (2026-06-13): render the compact `small`
        # schema for LOCAL models (MLX/Ollama) even when they classify as
        # `mid`, so a local VLM like qwen3.6:27b spends its attention on the
        # game rather than on a ~20KB schema + 6KB project-doc it must read
        # before writing a line. None = auto (on for local non-large);
        # override via `/leanprompt on|off`. Retrieval budgets keep using
        # self._model_class — only the SYSTEM PROMPT tier changes.
        self._lean_prompt: bool | None = None
        # Most-recent before/after screenshot bytes for the VLM. Filled
        # by the verifier on each iter; consumed by `_stream`.
        self._last_screenshot_before: bytes | None = None
        self._last_screenshot_after: bytes | None = None
        # Sibling path strings so the image_attached trace event can say
        # WHICH screenshot file was sent to the model. Without these the
        # trace shows bytes/dims only and you can't correlate to a file
        # on disk to inspect.
        self._last_screenshot_before_path: str | None = None
        self._last_screenshot_after_path: str | None = None
        # Last screenshot fed to the vision-progress judge — kept so the
        # NEXT iter's judge call can compare "current vs previous" and
        # judge whether real visible progress happened. Independent from
        # `_last_screenshot_after` (which is for VLM critique attachment
        # and gated on `_is_vlm`); this one runs out-of-band regardless.
        self._prev_judge_png: bytes | None = None
        # Resolved local MLX-VLM path for the visual-progress judge.
        # Populated lazily on first `_run_vision_judge` call. None means
        # "no local VLM discovered" — we then skip the judge rather than
        # silently calling a cloud model (the user's "never silent cloud
        # calls" rule). Set to the sentinel "" to mean "scanned, none
        # found" so we don't rescan every iter.
        self._local_vlm_path: str | None = None
        # /allroles bundle: when True, vlm-critique runs on the coder slot
        # instead of the local MLX-VLM vision-judge fallback. Synced from
        # chat.py after construction (mirrors `_use_autonomous_feedback`).
        self._all_roles_enabled: bool = False
        # Per-iter screenshot delta against previous iter (mean abs pixel
        # diff, 0..1). Stashed for the loop's regression detector.
        self._last_screenshot_delta: float | None = None
        # Last vision-judge verdict — preserved across compaction so the
        # state-anchor summary can still tell the model "as of iter N
        # the game looked like <note>". Surface, not signal: this is a
        # one-line hint, not a structural assertion.
        self._last_vision_verdict_iter: int | None = None
        self._last_vision_verdict_progress: bool | None = None
        self._last_vision_verdict_note: str = ""
        # Acceptance criteria the model emitted during Phase A — fed back
        # into fix prompts so the model self-checks against its own bar.
        self._criteria: str = ""
        # Executable acceptance probes — JS expressions the model proposes
        # in Phase A. Each iter's verifier runs them in the page; results
        # join the report. Empty list = no model probes (universal probes
        # still run).
        self._probes: list[dict] = []
        # Latest continuation request. Kept separate from `_goal` so an
        # extension can change the requested runtime shape without
        # rewriting the original session label.
        self._continuation_feedback: str = ""
        # Classification of the most-recent probe set into structural-
        # vs-dynamic. {"dynamic": [...], "structural": [...], "ratio":
        # float}. Populated after Phase A; consumed when assembling
        # the first-build user message to inject a nudge when zero
        # probes verify dynamic behavior. DK trace 20260513_185815
        # rationale (probes only checked structural presence; a static
        # HUD passed all of them).
        self._probe_quality: dict = {
            "dynamic": [], "structural": [], "ratio": 0.0,
        }
        # <todos> artifact (deepagents-style): mutable plan checklist
        # the model emits and rewrites each turn. Persisted to disk
        # next to the trace, replayed via the state-anchor so it
        # survives compaction. Empty until the model emits the first
        # <todos> block; optional — universal probes / criteria still
        # cover acceptance even when the model never uses it.
        self._todos_text: str = ""
        # Autonomous-recipe skip cache (2026-06-12): (recipe_id, code
        # hash) pairs whose applicability gate already came back falsy
        # for the current code. Any patch changes the hash and re-opens
        # the gate; identical re-evals are skipped silently.
        self._recipe_skip_cache: set[tuple[str, str]] = set()
        # Todo-driven execution (frontier-agent pattern, 2026-06-12).
        # Parsed view of _todos_text: list of (done, text) items. When a
        # clean iter leaves unchecked items, the post-clean prompt names
        # the FIRST unchecked one as the CURRENT TASK so each turn has
        # ONE objective (the biggest reliability lever for 27B-class
        # local models). Telemetry only on mismatch — never a cutoff.
        self._todos_items: list[tuple[bool, str]] = []
        # Phase 2: True while the ledger is HARNESS-SEEDED (from goal clauses /
        # outline order) and the model has not yet emitted its own <todos>.
        # Gates conservative harness done-marking; cleared the moment the model
        # takes ownership via `_capture_todos`, so model control is preserved.
        self._todos_seeded_by_harness: bool = False
        # Phase 4B (gated experiment): records HOW the harness seed was derived
        # ("goal_clauses" | "outline_order" | None). Only "outline_order" (a
        # complex fresh build with a strong outline match >=0.5 and >=3 ordered
        # steps) arms the one-objective-first-build nudge.
        self._ledger_source: str | None = None
        # The todo injected as CURRENT TASK in the most recent prompt
        # (None when no contract was injected). Consumed at the next
        # streak update for the todo_drift telemetry check.
        self._current_todo: str | None = None
        # Per-task injection counts so a todo the model refuses to
        # check off doesn't get re-nagged forever (cap 2 per task).
        self._todo_nag_counts: dict[str, int] = {}
        self._token_cb = None
        self._goal: str = ""
        # Tracks the most recent test-report summary for memory.record_outcome.
        self._last_report_summary: str = ""
        self._last_iter_run: int = 0
        # Tracks the most recent file content actually written to disk. We
        # always inline THIS in fix prompts (instead of asking the model to
        # remember its own previous reply).
        self._current_file: str = ""
        # Bullet bodies queued by <lookup_bullet> tags in the most recent
        # assistant reply. Drained into the next user message so the model
        # actually receives the requested body. Pi-mono "skills" pattern.
        self._pending_bullet_lookups: list[str] = []
        # Lazy ImageGenerator for Z-Image-Turbo sprite generation. Only
        # loaded if the model emits an <assets> block in Phase A; the
        # diffusers / torch import + pipeline init costs ~30-60s, so we
        # never pay for it on sessions that don't request art.
        self._asset_generator: Any = None
        # Resolved asset paths from Phase A (name → absolute path); used
        # by the first-build prompt assembler.
        self._session_assets: dict[str, Path] = {}
        # Asset names dropped by the per-turn cap (asset_overflow) that still
        # need mid-session generation — re-warned every iter until on disk.
        self._pending_dropped_assets: list[str] = []
        # Full specs for harness-dropped sprites (name/prompt/size) — autogen
        # at the next iter boundary instead of waiting for the model to re-emit.
        self._pending_dropped_asset_specs: list[dict] = []
        # Phase 0.10 — per-session cap on assets generated per <assets>
        # block. Default `None` means "use module default" (24). Raised
        # at session start when the goal explicitly asks for multi-frame
        # rosters via `prompts_v1._detect_multi_frame_intent`. The raise
        # lets a user-requested N entities × M frames roster land in one
        # turn instead of getting silently truncated to 24.
        self._session_asset_cap: int | None = None
        self._post_clean_shrink_detected: bool = False
        # Phase 1.5 — autonomous self-feedback loop. Mirrors the chat.py
        # toggle so the agent can check this flag without depending on
        # the TUI. Default ON; /playtest off (alias /feedback off) flips it.
        # The standard error-reporting paths (test reports, patch
        # diagnostics, visual critic) are NOT gated by this — only the
        # extra autonomous direction.
        self._use_autonomous_feedback: bool = True
        # When False, every harness-added directive that wraps USER
        # feedback (MEDIA-CHANGE, ORIENTATION-CHANGE, FEEDBACK SCOPE
        # ARBITRATION, asset stem mappings, scoped constraint config)
        # is suppressed and the user's text is passed through with only
        # the basic USER FEEDBACK (HIGHEST PRIORITY) wrapper. Use when
        # the classifier is misrouting your guidance (Doom trace
        # 2026-05-23: "down key moves you forward" got wrapped with
        # "the feedback above is about ART/SOUND, not code"). Flip via
        # /rawfeedback in the TUI.
        self._use_feedback_directives: bool = True
        # Cycle counter for the budget governor. Resets per session.
        # Capped at _AUTONOMOUS_MAX_CYCLES; auto-stops if two consecutive
        # playtests find nothing new.
        self._autonomous_playtest_cycle: int = 0
        self._autonomous_no_findings_streak: int = 0
        # Phase 1A — background task handle for the non-blocking visual
        # critic. When the critic backend is a DIFFERENT slot than the
        # coder, the critic runs as an asyncio.Task overlapping iter N+1's
        # coder stream prefill + early tokens, and its coaching lands in
        # `_pending_coaching` whenever the task completes. When critic
        # backend == coder backend (single-slot fallback), we await
        # inline (no benefit, no extra complexity).
        self._critic_task: "asyncio.Task | None" = None
        # Same lazy-load pattern for Stable Audio Open. Only loaded when
        # the model emits a <sounds> block in Phase A.
        self._sound_generator: Any = None
        # Resolved sound paths (name → absolute path) and the subset
        # that was declared loop=true. The loop set is preserved
        # separately so render_sound_paths_block can mark them in the
        # injected loader pattern.
        self._session_sounds: dict[str, Path] = {}
        self._session_looping: set[str] = set()
        # Video cutscenes (<videos> block). The generator is a cheap
        # subprocess wrapper (no in-process model load) — Wan2.2 weights
        # live in the scripts/generate_video.py child process per batch.
        self._video_generator: Any = None
        self._session_videos: dict[str, Path] = {}
        self.restart_n: int = max(1, int(restart_n))
        self.restart_score_threshold: float = float(restart_score_threshold)
        # restart-N attempt index (0-based). Attempt 0 keeps the default
        # decode profile; later attempts apply a diversified profile so a
        # restart doesn't replay the same generation trajectory.
        self._restart_attempt_idx: int = 0
        self._restart_seed_base: int = (
            (int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF) or 1
        )
        self._restart_attempt_seed: int | None = None
        # First-build rescue flag. If iter 1 emits prose without code,
        # force the next first-build turn to start from an <html_file>
        # prefill stub to break the loop.
        self._force_first_build_prefill: bool = False
        # One-shot "same-iteration" retry budget for first-build no-code
        # failures. We grant one bonus iter so format-only recovery does
        # not consume the user's regular iteration budget.
        self._first_build_retry_bonus_used: bool = False
        # Phase 0D-4: same-iter auto-retry after a STALL on a feedback turn.
        # When the model deliberated / looped / went silent on a turn that
        # carried user feedback and saved no code (Fieldrunners trace
        # 20260626_102307 iters 4-5), retry immediately (skip the step-mode
        # pause) on a bonus iter so the edit is not silently lost. Bounded.
        self._auto_retry_pending: bool = False
        self._feedback_no_code_retries: int = 0
        # Did the most recent iteration actually materialize code? Drives the
        # honest step-mode pause label ("complete" vs "produced no code").
        self._last_iter_materialized: bool = False

    # Read-through to the resolved backend's model id. Existing call sites
    # (trace metadata, conversation dump, memory.record_outcome, ...) used
    # `self.model` as a string; keeping it as a property means the agent
    # always reports whatever the backend resolved to without callers
    # having to know about Backend internals.
    @property
    def model(self) -> str:
        return self._backend.info.model

    def _planning_role(self) -> str:
        """Role label for planning / exit-decision turns.

        Returns 'architect' only when planning will actually run on a
        distinct backend — i.e. the user explicitly staged a slot with
        role=architect, OR `_use_architect_split` (/allroles bundle)
        is on. Otherwise returns 'coder' so the UI and trace don't
        claim a role the user never enabled. (Bug observed 2026-05-23
        doom run: UI showed "Activity (architect)" when only coder +
        critic were staged.)
        """
        if getattr(self, "_use_architect_split", False):
            return "architect"
        arch_backend = self.get_backend("architect")
        if arch_backend is not None and arch_backend is not self._backend:
            return "architect"
        return "coder"

    def get_backend(self, role: str) -> Backend:
        """Dynamic hierarchical routing for multi-GPU configurations."""
        if role == "coder":
            return self._backend  # Always Model 1
            
        elif role == "architect":
            # Check if Model 2 or Model 3 is explicitly configured as architect
            if getattr(self, "_model2_role", None) == "architect" and getattr(self, "_backend2", None) is not None:
                return self._backend2
            if getattr(self, "_model3_role", None) == "architect" and getattr(self, "_backend3", None) is not None:
                return self._backend3
            return self._backend  # Fallback to Model 1
            
        elif role == "critic":
            # Check if Model 3 or Model 2 is explicitly configured as visual critic
            if getattr(self, "_model3_role", None) == "critic" and getattr(self, "_backend3", None) is not None:
                return self._backend3
            if getattr(self, "_model2_role", None) == "critic" and getattr(self, "_backend2", None) is not None:
                return self._backend2
            # Unified single critic (2026-06-14): there is ONE structured
            # visual-critic path, gated only by /vlm-critique. When no separate
            # critic model is staged, the CODER itself reviews its own
            # screenshot — provided it can see (it's a VLM). /allroles also
            # routes here (coder slot). The old lightweight open-ended
            # vision-judge fallback is retired. Returns None only when the
            # coder is text-only AND no critic is staged — then vision is
            # impossible and the deterministic probes carry verification.
            if getattr(self, "_use_vlm_critique", False) and (
                getattr(self, "_all_roles_enabled", False)
                or getattr(self, "_is_vlm", False)
            ):
                return self._backend
            return None

    def _keep_alive_for_backend(self, backend: Backend | None) -> float | str | None:
        if backend is not None and getattr(backend, "info", None) and backend.info.name == "ollama":
            return self._ollama_keep_alive
        return None

    def _set_role_activity(self, role: str, activity: str) -> None:
        """Set TUI status panel text for Model 2 and Model 3 dynamically."""
        if getattr(self, "_model2_role", None) == role:
            self._model2_activity = activity
        if getattr(self, "_model3_role", None) == role:
            self._model3_activity = activity


    @staticmethod
    def _patch_anchor_fingerprint(search: str) -> str:
        """Deterministic short fingerprint for a patch SEARCH block.

        Used by `_last_failed_patch_anchors` to detect cross-turn repeats
        of the same failing SEARCH. Normalizes whitespace + takes first
        ~80 chars so a model that re-emits the same SEARCH with slightly
        different indentation or trailing spaces still matches.
        """
        import hashlib as _hashlib
        normalized = " ".join((search or "").split())
        head = normalized[:80].encode("utf-8", "ignore")
        return _hashlib.sha1(head).hexdigest()[:12]

    # Phrases inside <notes> that indicate the model has given up —
    # it's emitting <done/> not because the game ships but because it
    # can't escape a parse / harness loop. Wolfenstein 2026-05-24 trace
    # turn [04] is the canonical case: model said `<done/>` with notes
    # "The <html_file> tag has been consistently failing to parse...
    # The user can manually copy the complete HTML code..." — the
    # agent took the <done/> at face value and shipped a broken file.
    # Universal: keys on the model's own confession of harness trouble,
    # not on goal text or genre. Detector below is intentionally narrow
    # (must combine a "failing to ship" verb with a harness/parse/manual
    # noun) so it doesn't false-fire on legitimate cosmetic notes.
    _GIVE_UP_NOTES_PATTERNS: tuple[str, ...] = (
        "consistently failing",
        "consistently fail",
        "failing to parse",
        "fails to parse",
        "cannot parse",
        "can't parse",
        "parser keeps rejecting",
        "manually copy",
        "manual copy",
        "user can manually",
        "harness parsing",
        "harness rejected",
        "harness issue",
        "broken parser",
    )

    @staticmethod
    def _notes_signal_give_up(notes_text: str | None) -> bool:
        """Return True when <notes> text indicates the model is asking
        for help (parse loop, manual-copy workaround) rather than
        legitimately shipping a working game.

        Single-phrase match is enough — the give-up phrases are
        distinctive and don't show up in normal "what works / what's
        still broken" notes.
        """
        if not notes_text:
            return False
        low = notes_text.lower()
        return any(p in low for p in GameAgent._GIVE_UP_NOTES_PATTERNS)

    @staticmethod
    def _reply_fingerprint(reply: str) -> str:
        """Deterministic short fingerprint for a full model reply.

        Used by `_last_no_usable_code_fingerprint` to detect when the
        model emits a bit-identical (or near-identical) unparseable reply
        twice in a row. Wolfenstein 2026-05-24 trace: same 7838-token
        reply on 5 consecutive iters, all rejected with the same generic
        fallback. The generic fallback gives the model no new signal, so
        the loop is self-reinforcing.

        Normalization: lowercased + whitespace-collapsed + first 4 KB.
        Stable across cosmetic reshuffles (trailing spaces, line
        ending shifts) so two replies with the same content but a
        different whitespace tail still fingerprint together.
        """
        import hashlib as _hashlib
        normalized = " ".join((reply or "").lower().split())
        head = normalized[:4000].encode("utf-8", "ignore")
        # Salt with the length bucket so a stream that gets cut short
        # at exactly the same point fingerprints separately from one
        # that fully completed at a different length — protects against
        # false positives when the model genuinely changed its mind.
        length_bucket = str(len(reply or "") // 100)
        head = head + b"|len:" + length_bucket.encode("ascii")
        return _hashlib.sha1(head).hexdigest()[:12]

    _REJECTED_REPLY_STUB_HEAD = 400

    def _probe_quality_nudge(self) -> str:
        """Return a directive to add dynamic-behavior probes, or "".

        Fires when every Phase-A probe is structural-only (no time-based
        check, no numeric threshold, no canvas-pixel delta). Surfaces in
        the first-build user message so the model can re-emit <probes>
        with at least one dynamic check. Re-emission is allowed via the
        existing `_planning_coverage_gaps` re-parse path in run() — no
        extra plumbing needed.

        Empty when probes are absent (universal probes will run instead)
        or already include at least one dynamic check. Returns a short
        block; the goal is a nudge, not a lecture.
        """
        pq = self._probe_quality
        if not pq or pq.get("ratio", 0.0) > 0.0:
            return ""
        if not self._probes:
            return ""
        return (
            "PROBE-QUALITY NUDGE: your Phase-A <probes> only verify "
            "structural presence (e.g. `!!window.state`, "
            "`typeof X === 'object'`). A game that renders a static "
            "HUD will pass them all. In your reply, re-emit "
            "<probes>[...]</probes> alongside the code, INCLUDING at "
            "least 2 probes that verify dynamic behavior over time. "
            "Example shapes:\n"
            "  - `await new Promise(r => setTimeout(r, 500)); return "
            "state.score >= 0 && state.frame > 30;` (RAF actually "
            "firing)\n"
            "  - `(()=>{const t0=performance.now(); return new Promise("
            "r=>setTimeout(()=>r(state.player.x !== window.__x0 || "
            "state.player.y !== window.__y0), 500));})();` (state delta "
            "over 500 ms)\n"
            "  - `(()=>{const c=document.querySelector('canvas'); "
            "const g=c.getContext('2d'); const a=g.getImageData(0,0,"
            "c.width,c.height).data; return a.some((v,i)=>i%4!==3 && "
            "v!==0);})();` (canvas has non-black pixels — confirms "
            "rendering)\n"
            "Each new probe is a JSON object {\"name\": ..., \"expr\": "
            "...}. Keep existing structural probes; ADD the dynamic "
            "ones."
        )

    @classmethod
    def _classify_model(cls, model: str | None) -> str:
        """Default model class.

        We dynamically classify the model based on substring matching of its
        name, falling back to 'small' for smaller models or unknown systems.
        """
        if not model:
            return "small"
        m = str(model).lower()
        # Large/frontier models
        large_keywords = [
            "70b", "72b", "100b", "110b", "120b", "132b", "141b", "236b", "314b", "405b",
            "gpt-4", "gpt-4o", "o1-", "o3-", "claude-3", "claude-3.5", "claude-4",
            "opus", "sonnet", "fable"
        ]
        if any(kw in m for kw in large_keywords):
            return "large"
        # Mid-tier capable models (14B-35B class + notable architectures)
        mid_keywords = [
            "14b", "16b", "20b", "22b", "27b", "32b", "33b", "34b", "35b", "moe",
            "qwen3.6", "qwen", "deepseek", "gemini", "gemma"
        ]
        if any(kw in m for kw in mid_keywords):
            return "mid"
        return "small"

    @staticmethod
    def _load_prompt_module(version: str):
        """Resolve the prompt module for `version` (e.g. "v1" → prompts_v1).

        v0 (`prompts.py`) was retired; only `prompts_v{N}.py` modules
        are supported. An unknown version raises ImportError immediately
        so misconfigured runs fail fast instead of silently using a
        stale prompt set.
        """
        import importlib
        return importlib.import_module(f"prompts_{version}")

    # Playbook / opening-book / component retrieval methods were moved
    # VERBATIM to agent_memory.MemoryRetrievalMixin (GameAgent inherits them).
    # See that module for the memory retrieval logic.

    @staticmethod
    def _report_blocker_query(report: dict) -> str:
        """Compact text describing the report's blockers, used as the
        retrieval query for fix-turn component injection. Empty when the
        report has no concrete blockers (then no component is injected)."""
        parts: list[str] = []
        try:
            for p in (report.get("probes") or []):
                if not p.get("ok"):
                    parts.append(str(p.get("name") or ""))
            for key in ("page_errors", "console_errors", "errors"):
                vals = report.get(key) or []
                if vals:
                    parts.append(str(vals[0])[:160])
            if report.get("frozen_canvas"):
                parts.append("frozen canvas animation loop stalled")
            for w in (report.get("soft_warnings") or []):
                ws = str(w)
                if any(
                    tok in ws
                    for tok in (
                        "ASSETS_LOADED_BUT_UNDRAWN",
                        "CONTROL-NOT-RECOVERED",
                        "FROZEN-CANVAS",
                        "PROBE FAILED",
                    )
                ):
                    parts.append(ws[:160])
            cnr = report.get("control_not_recovered")
            if isinstance(cnr, dict) and cnr.get("key"):
                parts.append(
                    f"control not recovered after {cnr.get('key')}"
                )
            it = report.get("input_test") or {}
            if it.get("ran") and not it.get("any_change"):
                parts.append("input keyboard keys not moving player")
        except Exception:
            return ""
        return " ".join(x for x in parts if x).strip()

    def _extract_and_queue_lookups(self, reply: str) -> None:
        """Find <lookup_bullet>id</lookup_bullet> tags in an assistant reply,
        resolve each against the playbook, and queue rendered bodies for
        injection into the next user-turn message. Pi-mono skills pattern.

        Capped at _MAX_BULLET_LOOKUPS_PER_TURN per reply so a chatty
        model can't bloat context. Unknown IDs are surfaced as
        "NOT FOUND" entries so the model knows its lookup missed.
        """
        if not reply:
            return
        raw_ids = [m.group(1).strip() for m in _LOOKUP_BULLET_RE.finditer(reply)]
        if not raw_ids:
            return
        seen: set[str] = set()
        resolved: list[str] = []
        for bid in raw_ids[:_MAX_BULLET_LOOKUPS_PER_TURN]:
            if not bid or bid in seen:
                continue
            seen.add(bid)
            b = lookup_bullet(self._playbook, bid)
            if b is None:
                resolved.append(
                    f"## [{bid}] — NOT FOUND in current playbook "
                    "(typo, or that ID is no longer available)"
                )
                continue
            tag_str = ",".join(b.tags[:5]) if b.tags else "untagged"
            resolved.append(
                f"## [{b.id}]  tags=[{tag_str}]\n{b.content}"
            )
        if not resolved:
            return
        block = (
            "================ PLAYBOOK LOOKUP RESULTS ================\n"
            "You requested these bullet bodies via <lookup_bullet> in your "
            "previous turn. Apply them where relevant — they were on the "
            "INDEX list and you asked for the body, so the body is now "
            "yours to use this turn.\n\n"
            + "\n\n".join(resolved)
            + "\n========================================================="
        )
        self._pending_bullet_lookups.append(block)
        self._trace({
            "kind": "bullet_lookups_resolved",
            "ids": list(seen),
            "count": len(resolved),
        })

    # -- TUI-facing setters -------------------------------------------------

    def add_user_feedback(self, text: str) -> None:
        text = text.strip()
        if text:
            self._pending_feedback.append(text)
            self._feedback_deferred_last_turn = False
            self._trace({"kind": "feedback_queued", "text": text})
            # Fix #2 (2026-05-22 Pac-Man trace) — refresh the autonomous
            # playtest budget when the user types fresh feedback. The
            # 12-iter Pac-Man session exhausted the 3-cycle cap by iter
            # 6-7 and then ran 5 more iters with no autonomous oversight
            # even after the user typed two new bug reports. New
            # feedback is a strong "verify again" signal — refresh the
            # cycle counter by one and clear the no-findings streak.
            # Model-agnostic; works regardless of session length.
            try:
                prior_cycle = getattr(self, "_autonomous_playtest_cycle", 0)
                prior_streak = getattr(self, "_autonomous_no_findings_streak", 0)
                refreshed = False
                if prior_cycle > 0:
                    self._autonomous_playtest_cycle = max(0, prior_cycle - 1)
                    refreshed = True
                if prior_streak > 0:
                    self._autonomous_no_findings_streak = 0
                    refreshed = True
                if refreshed:
                    self._trace({
                        "kind": "autonomous_budget_refreshed_on_feedback",
                        "prior_cycle": prior_cycle,
                        "new_cycle": getattr(self, "_autonomous_playtest_cycle", 0),
                        "prior_no_findings_streak": prior_streak,
                    })
            except Exception:
                # Field may not exist on every agent path (early-init,
                # tests). Refresh is advisory; never block feedback.
                pass

    def add_user_answer(self, text: str) -> None:
        self._pending_answer = text.strip()
        self._feedback_deferred_last_turn = False
        self._trace({"kind": "answer_queued", "text": self._pending_answer})

    def unqueue_pending_input(self, which: str = "") -> dict[str, Any]:
        """Drop queued user input before the next user-turn boundary.

        Used by the TUI ``/unqueue`` command when the user typed feedback
        by accident while the agent was still streaming.

        Bare ``/unqueue`` (default) removes ONLY the most recently typed
        **feedback** line — the last thing appended while watching the
        stream. It never touches a queued model-question answer and never
        removes older feedback you queued earlier.

        Explicit forms (power-user):
          - ``all`` — clear every queued feedback item and any pending
            answer.
          - ``answer`` — clear only the pending answer to a model
            ``<question>``.
          - ``N`` (1-based) — remove feedback item *N* (same numbering as
            the status panel ``Queued`` list).

        Returns ``{"ok": True, ...}`` or ``{"ok": False, "error": ...}``.
        Does not touch conversation history or the on-disk game file.
        """
        key = (which or "").strip().lower()
        removed: list[dict[str, str]] = []

        def _preview(text: str) -> str:
            t = (text or "").strip()
            return t[:80] + ("…" if len(t) > 80 else "")

        if key in ("all", "clear", "reset"):
            for fb in self._pending_feedback:
                removed.append({"kind": "feedback", "preview": _preview(fb)})
            if self._pending_answer is not None:
                removed.append({
                    "kind": "answer",
                    "preview": _preview(self._pending_answer),
                })
            count = len(self._pending_feedback) + (
                1 if self._pending_answer is not None else 0
            )
            self._pending_feedback = []
            self._pending_answer = None
            if count:
                self._trace({
                    "kind": "feedback_unqueued",
                    "which": "all",
                    "count": count,
                })
            return {
                "ok": True,
                "which": "all",
                "removed": removed,
                "remaining_feedback": 0,
                "remaining_answer": False,
            }

        if key in ("answer", "ans"):
            if self._pending_answer is None:
                return {"ok": False, "error": "no queued answer to remove"}
            preview = _preview(self._pending_answer)
            self._pending_answer = None
            self._trace({
                "kind": "feedback_unqueued",
                "which": "answer",
                "preview": preview,
            })
            return {
                "ok": True,
                "which": "answer",
                "removed": [{"kind": "answer", "preview": preview}],
                "remaining_feedback": len(self._pending_feedback),
                "remaining_answer": False,
            }

        # Default (bare /unqueue): drop ONLY the last typed feedback line.
        if key in ("", "last", "feedback", "fb"):
            if not self._pending_feedback:
                return {
                    "ok": False,
                    "error": (
                        "no queued feedback to remove — only typed feedback "
                        "can be unqueued with bare /unqueue "
                        "(use /unqueue answer for a queued model-question reply)"
                    ),
                }
            text = self._pending_feedback.pop()
            preview = _preview(text)
            self._trace({
                "kind": "feedback_unqueued",
                "which": "last_feedback",
                "preview": preview,
                "remaining": len(self._pending_feedback),
            })
            return {
                "ok": True,
                "which": "last_feedback",
                "removed": [{"kind": "feedback", "preview": preview}],
                "remaining_feedback": len(self._pending_feedback),
                "remaining_answer": self._pending_answer is not None,
            }

        try:
            idx = int(key)
        except ValueError:
            return {
                "ok": False,
                "error": (
                    f"unknown /unqueue target {which!r} — bare /unqueue "
                    "drops your last typed feedback; also: all, answer, "
                    "or a 1-based item number"
                ),
            }
        if idx < 1 or idx > len(self._pending_feedback):
            return {
                "ok": False,
                "error": (
                    f"no queued feedback item #{idx} "
                    f"(have {len(self._pending_feedback)})"
                ),
            }
        text = self._pending_feedback.pop(idx - 1)
        preview = _preview(text)
        self._trace({
            "kind": "feedback_unqueued",
            "which": str(idx),
            "preview": preview,
            "remaining": len(self._pending_feedback),
        })
        return {
            "ok": True,
            "which": str(idx),
            "removed": [{"kind": "feedback", "preview": preview}],
            "remaining_feedback": len(self._pending_feedback),
            "remaining_answer": self._pending_answer is not None,
        }

    def has_pending_user_input(self) -> bool:
        return bool(self._pending_feedback) or self._pending_answer is not None

    def _has_genuine_user_input(self) -> bool:
        """Like has_pending_user_input, but AGENT-generated feedback
        (autonomous-playtest findings etc., tracked in
        `_internal_feedback_texts`) doesn't count.

        Added 2026-06-12 (trace 20260612_132314): at every clean iter an
        [AUTONOMOUS PLAYTEST] finding was pending, so the todo CURRENT
        TASK contract — gated on "no pending user input" — never fired
        all session. Internal findings should share the turn with the
        contract, not displace it; only REAL user input wins the turn.
        """
        if self._pending_answer is not None:
            return True
        internal = getattr(self, "_internal_feedback_texts", set())
        return any(fb not in internal for fb in self._pending_feedback)

    def request_done(self) -> None:
        self._user_force_done = True
        # Signal the in-flight stream (if any) to stop now. The
        # MLXBackend worker polls this between tokens; the next yield
        # exits, stream_chat returns a partial result with stalled=True,
        # and the iter-boundary check in run() ships best.html.
        try:
            ev = self._ensure_stop_event()
            ev.set()
        except RuntimeError:
            # No running event loop yet — the flag above will be
            # picked up at the next iter-boundary check anyway.
            pass

    def _ensure_stop_event(self) -> asyncio.Event:
        """Lazily create the stop event on the running event loop.

        Raises RuntimeError if there's no running loop (called from
        outside an async context, e.g. an early TUI hook).
        """
        if self._stop_event is None:
            self._stop_event = asyncio.Event()
        return self._stop_event

    # Step-mode controls (Stop-Losing-To-OneShot todo #1).
    def set_step_mode(self, on: bool) -> None:
        """Turn step-mode on/off. When on, the iter loop pauses after
        each iteration boundary and waits for explicit user input before
        querying the model again. Drivers wake the wait by either
        signal_step_continue() (no feedback) or add_user_feedback() (the
        existing path).

        Explicit user-driven /wait off also flips `_step_auto_disabled`
        so the auto-arm logic doesn't immediately re-enable step-mode
        after the next failed iter — once the user says "no thanks,"
        we don't keep asking.
        """
        new = bool(on)
        if self._step_mode and not new:
            # Explicit disable — opt out of auto-arm for the rest of
            # the session.
            self._step_auto_disabled = True
        self._step_mode = new
        self._trace({"kind": "step_mode_set", "on": self._step_mode})

    def signal_step_continue(self) -> None:
        """Release the current step-mode wait without adding feedback.
        No-op when no wait is active."""
        self._step_continue = True
        self._trace({"kind": "step_continue_signal"})

    def _step_pause_should_wait(self) -> bool:
        """True while the between-iter step-mode pause should keep sleeping.

        Ctrl+D / 'done' set _user_force_done without queuing feedback; the
        wait loop must wake so the top-of-iter ship check can exit.
        """
        if self._user_force_done:
            return False
        if self._step_continue:
            return False
        if self.has_pending_user_input() and not self._feedback_deferred_last_turn:
            return False
        return True

    def _step_pause_flush_feedback(self) -> None:
        """Fold feedback typed during a step-mode pause into the queued
        next-user turn.

        In step mode the next user message is ALREADY appended at the end
        of the previous iter. We pop it and re-flush so the feedback banner
        leads the report-driven base prompt.

        [STEP-PAUSE POLISH FIX 20260622 — battlezone trace] When the queued
        base was a polish / GAME-FEEL turn (built last iter because no
        feedback was pending), reusing it here keeps that polish framing on
        what is now a FIX turn. The model then bundles a "juice" feature
        onto the real fix, bloating the reply until it truncates and the
        whole patch set is bracket-rejected. So if a polish turn is pending,
        DISCARD it and rebuild a clean fix base from the last report — with
        feedback now pending, _build_fix_prompt skips polish on its own. A
        NON-polish base is preserved untouched (the working single-fix path,
        e.g. battlezone asks 2 & 3, must not change).
        """
        if not (self._messages and self._messages[-1].get("role") == "user"):
            return
        base = self._messages.pop()["content"]
        if self._polish_pending and self._last_test_report is not None:
            self._polish_pending = False
            base = self._build_fix_prompt(
                report=self._last_test_report,
                regressed=False,
                partial_failed=[],
            )
            # Match the end-of-iter queueing path so the rebuilt report
            # block still collapses once superseded.
            base = self._wrap_report_block(base, self._last_test_report)
            self._trace({"kind": "step_pause_polish_discarded"})
        self._messages.append({
            "role": "user",
            "content": self._flush_user_injections(base),
        })

    def set_auto_step_on_failure(self, on: bool) -> None:
        """Enable/disable auto step-mode arming on first failed iter."""
        self._auto_step_on_failure = bool(on)
        self._trace({
            "kind": "auto_step_on_failure_set",
            "on": self._auto_step_on_failure,
        })

    # Generic DOM/structural attributes whose values are pure layout noise,
    # not feature signals — dropped from seed structural tokens.
    _SEED_TOKEN_STOPWORDS = frozenset({
        "canvas", "ctx", "context", "div", "span", "body", "html", "head",
        "script", "style", "container", "wrapper", "main", "root", "app",
        "game", "gamecontainer", "screen", "overlay", "true", "false",
    })

    def _is_local_backend(self) -> bool:
        """True for local backends (MLX/Ollama)."""
        return self._backend.info.name in {"mlx", "ollama"}

    def _lean_prompt_active(self) -> bool:
        """Whether to render the compact `small` system-prompt schema.

        Explicit `/leanprompt on|off` wins; otherwise auto-on for local
        backends (MLX/Ollama) on non-large tiers. SOTA / large / cloud keep
        the full schema.
        """
        if self._lean_prompt is not None:
            return bool(self._lean_prompt)
        return self._is_local_backend() and self._model_class != "large"

    def _system_prompt_class(self) -> str:
        """Effective model_class for SYSTEM PROMPT rendering only.

        In lean mode a `mid` local model renders the `small` schema (67%
        smaller). Retrieval budgets continue to key off self._model_class,
        so only the system prompt shrinks.
        """
        if self._lean_prompt_active() and self._model_class == "mid":
            return "small"
        return self._model_class

    def set_lean_prompt(self, value: bool | None) -> None:
        """TUI hook for `/leanprompt on|off|auto`. None resets to auto."""
        self._lean_prompt = value

    # `_LEAN_MEMORY_COMBINED_BUDGET`, `_OPEN_DOMAIN_OUTLINE_FLOOR`,
    # `_apply_lean_memory_budget`, and `_detect_open_domain_build` were moved
    # VERBATIM to agent_memory.MemoryRetrievalMixin (GameAgent inherits them).

    _PLAN_ELIDE_RE = re.compile(r"<plan>.*?</plan>", re.DOTALL)

    def _lean_compact_planning_message(self) -> None:
        """Elide the verbose <plan> prose from the retained Phase-A assistant
        turn (lean mode, after first build). The plan's <criteria>/<probes>
        were already extracted into self._criteria / self._probes and survive
        compaction independently, so the full plan prose re-loading on every
        coder prefill is pure overhead for a local model. Idempotent."""
        for msg in self._messages:
            if msg.get("phase") != "planning":
                continue
            content = msg.get("content") or ""
            if "<plan>" not in content or "[plan prose elided" in content:
                return
            new_content = self._PLAN_ELIDE_RE.sub(
                "<plan>[plan prose elided after first build to save local "
                "context; criteria/probes retained]</plan>",
                content,
                count=1,
            )
            if new_content != content:
                msg["content"] = new_content
                self._trace({
                    "kind": "lean_plan_prose_elided",
                    "saved_chars": len(content) - len(new_content),
                })
            return

    def _local_first_build_nudge(self) -> str:
        """Local-only first-build contract + optional compact-code hint."""
        if not self._is_local_backend():
            return ""
        parts = [
            "LOCAL FIRST-BUILD CONTRACT: Plan is ACCEPTED — do NOT restate "
            "requirements or re-plan. Brief reasoning OK; first output tag "
            "must be `<html_file>` (raw tag, no ```html fence). Start the "
            "loop: call requestAnimationFrame(loop) unconditionally after "
            "asset load.",
        ]
        n_assets = len(self._session_assets)
        n_sounds = len(self._session_sounds)
        if n_assets >= 10 or n_sounds >= 6:
            parts.append(
                "LOCAL MODEL SAFETY NUDGE: Keep first-build code compact to "
                "avoid token loops. Use short name arrays + loops for media "
                "loaders; do NOT hand-enumerate long repeated `[name, path]` "
                "blocks. Use ONLY sound/sprite names present in the GENERATED "
                "ASSETS/SOUNDS blocks above."
            )
        if n_assets >= 1:
            parts.append(
                "SPRITE DRAW (required): PNG paths were generated above — "
                "every tower/enemy/projectile draw path MUST call "
                "`sprite(key)` or `ctx.drawImage` with the EXACT asset names. "
                "Do NOT draw entities with fillRect/arc placeholders when a "
                "PNG exists; the harness counts undrawn sprites as a failure."
            )
        return "\n".join(parts)

    def _should_pre_lean_plan_before_first_build(self) -> bool:
        """Gate pre-iter-1 plan prose elision: local backend + heavy context."""
        if not self._is_local_backend():
            return False
        if len(self._session_assets) >= 10 or len(self._session_sounds) >= 6:
            return True
        for msg in self._messages:
            if msg.get("phase") != "planning":
                continue
            content = msg.get("content") or ""
            if "[plan prose elided" in content:
                return False
            if len(content) > 8000:
                return True
            m = self._PLAN_ELIDE_RE.search(content)
            if m and len(m.group(0)) > 2000:
                return True
        return False

    def _one_objective_first_build_nudge(self) -> str:
        """Phase 4B (gated experiment): on a slow local backend building a
        COMPLEX fresh game from a strong outline match, tell the model to
        implement ONLY the first checklist step this turn instead of the whole
        recipe at once. Fieldrunners trace 20260626_102307 iter 1 burned ~49
        min / 45K tok trying to author everything in one stream; one-objective
        builds keep the first stream small and let the iterate loop add the
        rest. OFF by default — opt in via AGENT_ONE_STEP_FIRST_BUILD=1 so the
        completion-token effect can be trace-measured before it ships on.

        Gate (ALL must hold): experiment flag on · local backend · ledger
        HARNESS-SEEDED from "outline_order" (which already implies outline
        match >=0.5 AND >=3 ordered steps via _seed_task_ledger_from_goal).
        Returns "" when any condition fails.
        """
        if os.environ.get("AGENT_ONE_STEP_FIRST_BUILD", "") not in ("1", "true", "True"):
            return ""
        if not self._is_local_backend():
            return ""
        if not self._todos_seeded_by_harness or self._ledger_source != "outline_order":
            return ""
        steps = [t for _, t in self._todos_items]
        if len(steps) < 3:
            return ""
        step1 = steps[0]
        self._trace({
            "kind": "one_objective_first_build_applied",
            "step1": step1,
            "step_count": len(steps),
        })
        return (
            "ONE-OBJECTIVE FIRST BUILD: This is a multi-step build. For THIS "
            f"turn implement ONLY step 1 — \"{step1}\" — as a complete, "
            "runnable <html_file> that loads and renders. Do NOT try to build "
            "every step at once. Later turns will add the remaining steps from "
            "the checklist one at a time."
        )

    def _local_should_fallback_skeleton(self, skel: SkeletonHit) -> tuple[bool, str]:
        """Guard local backends from mismatched won-skeleton media naming."""
        if not self._is_local_backend():
            return (False, "")
        if skel.source_goal is None:
            return (False, "")
        refs_assets = self._scan_html_for_asset_refs(skel.html)
        refs_sounds = self._scan_html_for_sound_refs(skel.html)
        refs = refs_assets | refs_sounds
        if not refs:
            return (False, "")
        available = set(self._session_assets.keys()) | set(self._session_sounds.keys())
        if not available:
            return (False, "")
        overlap = refs & available
        ratio = len(overlap) / max(1, len(refs))
        if ratio >= 0.25 or len(overlap) >= 2:
            return (False, "")
        return (
            True,
            (
                "low media-name overlap on local backend "
                f"(skeleton refs={len(refs)}, overlap={len(overlap)}, "
                f"ratio={ratio:.2f})"
            ),
        )

    def set_token_callback(self, cb) -> None:
        self._token_cb = cb

    def _estimate_ctx_fill(self) -> int:
        """Return the total character count across `_messages`.

        Cheap to call repeatedly — the TUI hits this on every status
        tick. Caller divides by a chars-per-token factor to get the
        token estimate; we keep the conversion out of here so the agent
        stays backend-agnostic.
        """
        total = 0
        for msg in self._messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
        return total

    # Exact prompt provenance lives with the prompt builders in
    # `agent_prompts.py`; it observes `_messages` without changing them.

    def trace_status(self, snapshot: dict) -> None:
        """Persist a TUI status-panel snapshot to the trace .jsonl.

        Called by the TUI's _update_status hook. Snapshots are
        deduped here by comparing against the last accepted payload —
        successive updates that change nothing meaningful (e.g. token
        counter ticks) are skipped to avoid trace bloat.

        The recorded payload is whatever the caller provides; by
        convention it mirrors the status panel keys: activity, phase,
        iteration, total_iters, streak_clean, streak_stuck, backend,
        model, goal, files. Best-effort: any exception is swallowed
        so the TUI never crashes on a logging failure.
        """
        try:
            sig_keys = (
                "activity", "phase", "iteration", "total_iters",
                "streak_clean", "streak_stuck", "backend", "model",
                # Queue observability (2026-06-11): a feedback item being
                # queued/drained must produce a snapshot row even when the
                # activity label hasn't changed yet.
                "pending_feedback", "ledger_tail",
                # Guard-abort sticky note — must re-snapshot when set/cleared
                # (donkey-kong 20260628: dedupe hid last_stall_reason=null).
                "last_stall_reason",
            )
            sig = tuple(snapshot.get(k) for k in sig_keys)
            if sig == getattr(self, "_last_status_sig", None):
                return
            self._last_status_sig = sig
            # Trace-conciseness (2026-06-11): `goal` and `files` are static
            # within a session yet were repeated on every row (~300 bytes ×
            # 100 rows). Carry-forward semantics: write them only when they
            # change; readers take the last seen value.
            row = dict(snapshot)
            static = {k: row.get(k) for k in ("goal", "files")}
            if static == getattr(self, "_last_status_static", None):
                for k in ("goal", "files"):
                    row.pop(k, None)
            else:
                self._last_status_static = static
            self._trace({"kind": "status_snapshot", **row})
        except Exception:
            pass

    def _token_cb_wrapper(self, piece: str) -> None:
        if self._token_cb is not None:
            try:
                self._token_cb(piece)
            except Exception:
                pass

    def _role_token_cb(self, role: str):
        """Token callback that tags stream pieces for the TUI status panel."""
        def _cb(piece: str) -> None:
            self._last_stream_role = role
            self._token_cb_wrapper(piece)
        return _cb

    @staticmethod
    def _activity_idle_event(role: str = "coder") -> AgentEvent:
        return AgentEvent("activity", "idle", {"role": role})

    # -- trace / snapshot helpers ------------------------------------------

    # Phase 4 (4D.0): the persisted .jsonl is an LLM-ONLY diagnostic artifact
    # (humans never read it). These high-frequency events exist only to drive
    # the live TUI panel/heartbeat and have NO trace-file consumer — the TUI
    # reads in-memory stashes (`_stream_progress_*`) and the token callback,
    # not the .jsonl. Persisting them spammed the reviewing LLM (~98
    # heartbeats on a 49-min stream, ~40 progress rows/iter). Their one piece
    # of diagnostic value (tokens / tok-s / duration) is folded onto
    # `iter_summary` and `stream_done` instead. `status_snapshot` is NOT here:
    # it is already deduped (low-volume) and is the only record of
    # feedback-queue timing ("was my feedback acknowledged?").
    _EPHEMERAL_TRACE_KINDS = frozenset({
        "stream_heartbeat", "stream_progress",
        # Phase 4 post-eval: fires every iter when no screenshot is queued for
        # the building model (text-only coder, or browser=None eval). TUI-only.
        "image_skipped",
    })

    def _trace(self, obj: dict) -> None:
        try:
            # Drop live-monitoring-only events from the persisted trace.
            if obj.get("kind") in self._EPHEMERAL_TRACE_KINDS:
                return
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            traced = dict(obj)
            # Add dynamic multi-agent telemetry role & name metadata
            if "model_role" not in traced and "model_name" not in traced:
                role = getattr(self, "_last_stream_role", "coder")
                backend = self.get_backend(role)
                if backend:
                    traced["model_role"] = role
                    traced["model_name"] = backend.info.model
            payload = _compact_trace_payload({
                "ts": datetime.utcnow().isoformat() + "Z",
                **traced,
            })
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    payload, ensure_ascii=False, default=str, separators=(",", ":"),
                ) + "\n")
        except Exception:
            pass

    def _record(self, ev: AgentEvent) -> AgentEvent:
        text = ev.text or ""
        data = ev.data
        # Trace conciseness (2026-06-13): the `test` event carries the FULL
        # report dict (~6.6KB) which dominated the jsonl (~35%) and fully
        # duplicates iter_summary. Store a compact projection in the TRACE
        # only; the live AgentEvent returned to the TUI is unchanged.
        if ev.kind == "test" and isinstance(data, dict):
            probes = data.get("probes") or []
            data = {
                "ok": data.get("ok"),
                "probes_passed": sum(1 for p in probes if p.get("ok")),
                "probes_total": len(probes),
                "failing_probes": [
                    p.get("name") for p in probes if not p.get("ok")
                ][:8],
                "soft_warnings": len(data.get("soft_warnings") or []),
                "page_errors": len(data.get("page_errors") or []),
                "console_errors": len(data.get("console_errors") or []),
                "frozen_canvas": data.get("frozen_canvas"),
                "_slimmed": True,
            }
        self._trace({
            "kind": "event",
            "event": ev.kind,
            "text_preview": text[:_TRACE_PREVIEW_CHARS],
            "text_len": len(text),
            "data": data,
        })
        return ev

    def _queue_stream_ui_event(self, ev: AgentEvent) -> None:
        """Trace an event and queue it for TUI yield after _stream() returns.

        Guard-abort notifications fire inside _stream(), which is not an
        async generator — callers must drain and yield them or the status
        panel never sees the stall reason (donkey-kong 20260628).
        """
        self._record(ev)
        self._pending_stream_ui_events.append(ev)

    def _drain_stream_ui_events(self) -> list[AgentEvent]:
        """Return and clear UI events queued during the last _stream()."""
        out = self._pending_stream_ui_events
        self._pending_stream_ui_events = []
        return out

    def _trace_exception(self, kind: str, e: Exception, **extra) -> None:
        """Trace an exception WITH a capped traceback.

        Added 2026-06-10: the dojo-fight trace 20260610_151443 recorded the
        harness UnboundLocalError only as the one-line str(e) — no stack —
        which is why a hard tools.py bug survived undetected. str(e) alone
        is not enough to debug harness/agent-loop failures from a trace.
        """
        self._trace({
            "kind": kind,
            "err": str(e)[:300],
            "traceback": traceback.format_exc()[-4000:],
            **extra,
        })

    def _trace_agent_crash(self, e: Exception, *, source: str = "agent_loop") -> None:
        """Record a fatal loop/TUI crash in the jsonl so the trace is self-sufficient.

        TUI/CLI consumers also log to the screen, but the .jsonl must carry
        enough for an LLM to debug without re-running (TD trace 20260630_103853:
        NameError in compaction was visible in the TUI only).
        """
        if getattr(self, "_crash_traced_this_session", False):
            return
        self._crash_traced_this_session = True
        self._trace_exception(
            "agent_crash",
            e,
            source=source,
            exc_type=type(e).__name__,
            iteration=getattr(self, "_last_iter_run", None),
            snapshot_n=getattr(self, "_snapshot_n", None),
            fix_mode=getattr(self, "_fix_mode", None),
        )

    def _save_snapshot(self, html: str) -> Path | None:
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_n += 1
            p = self.snapshots_dir / f"iter_{self._snapshot_n:02d}.html"
            p.write_text(html, encoding="utf-8")
            return p
        except Exception:
            return None

    def _mlx_model_on_disk_gb(self) -> float | None:
        """Sum on-disk weight bytes for the active MLX coder model path."""
        try:
            if not self._backend or getattr(self._backend.info, "name", None) != "mlx":
                return None
            from backend import MLXBackend
            path = getattr(MLXBackend, "_loaded_path", None) or self._backend.info.model
            return self._mlx_model_disk_gb(path)
        except Exception:
            return None

    @staticmethod
    def _mlx_model_disk_gb(path: str | None) -> float | None:
        """On-disk weight size (GB) for an MLX model directory."""
        if not path:
            return None
        try:
            from pathlib import Path as _Path
            p = _Path(path)
            if not p.exists():
                return None
            total = 0
            for f in p.rglob("*"):
                try:
                    if f.is_file():
                        total += f.stat().st_size
                except Exception:
                    pass
            return round(total / 1e9, 1)
        except Exception:
            return None

    def _relieve_vram_for_mlx_model_swap(self, new_model_path: str) -> dict:
        """Free VRAM before hot-swapping MLX coder models mid-session.

        Pinball trace 20260701_163948: Qwen3.6-27B stayed resident while
        diffusers were loaded; /model GLM-5.2 only changed the backend pointer
        and the process OOM-killed on the first GLM load. Unload the previous
        MLX weights and diffuser pipelines when swapping paths, or when loading
        a large model while diffusers are still resident.
        """
        from backend import MLXBackend

        old_path = MLXBackend._loaded_path or MLXBackend._loaded_vlm_path
        info: dict = {
            "old_path": old_path,
            "new_path": new_model_path,
            "freed": [],
            "skipped": False,
        }
        if not new_model_path:
            info["skipped"] = True
            self._trace({"kind": "mlx_model_swap_relief", **info})
            return info

        new_disk = self._mlx_model_disk_gb(new_model_path)
        old_disk = self._mlx_model_disk_gb(old_path) if old_path else None
        small_disk_gb = float(
            os.environ.get("AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB", "50")
        )
        swapping = old_path is not None and old_path != new_model_path
        upsizing = (
            swapping
            and new_disk is not None
            and old_disk is not None
            and new_disk > old_disk
        )
        loading_large = new_disk is not None and new_disk >= small_disk_gb
        diffusers_loaded = (
            self._asset_generator is not None
            or self._sound_generator is not None
        )

        should_relief = swapping or (loading_large and diffusers_loaded)
        if not should_relief:
            info["skipped"] = True
            info["new_disk_gb"] = new_disk
            info["old_disk_gb"] = old_disk
            self._trace({"kind": "mlx_model_swap_relief", **info})
            return info

        info["new_disk_gb"] = new_disk
        info["old_disk_gb"] = old_disk
        info["upsizing"] = upsizing
        info["freed"].extend(self._release_diffusers_vram())
        if swapping:
            freed_path = MLXBackend.release_weights(wait_for_metal=True)
            if freed_path:
                info["freed"].append(f"MLX-LLM ({Path(freed_path).name})")
        self._trace({"kind": "mlx_model_swap_relief", **info})
        return info

    @staticmethod
    def _available_system_memory_gb() -> tuple[float | None, float | None]:
        """Best-effort (available_gb, phys_gb) for memory-pressure gating."""
        import os as _os
        import sys as _sys
        try:
            phys_gb = (
                _os.sysconf("SC_PHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
            ) / 1e9
        except Exception:
            phys_gb = None
        available_gb: float | None = None
        try:
            if _sys.platform == "darwin":
                import re
                import subprocess
                proc = subprocess.run(
                    ["vm_stat"], capture_output=True, text=True, timeout=2,
                )
                text = proc.stdout or ""
                page_size = 4096
                m = re.search(r"page size of (\d+)", text)
                if m:
                    page_size = int(m.group(1))
                pages = 0
                for prefix in (
                    "Pages free:",
                    "Pages inactive:",
                    "Pages speculative:",
                ):
                    for line in text.splitlines():
                        if line.startswith(prefix):
                            pages += int(line.split(":")[1].strip().rstrip("."))
                available_gb = pages * page_size / 1e9
            else:
                avail_pages = _os.sysconf("SC_AVPHYS_PAGES")
                page_size = _os.sysconf("SC_PAGE_SIZE")
                available_gb = avail_pages * page_size / 1e9
        except Exception:
            available_gb = None
        if available_gb is not None:
            available_gb = round(available_gb, 1)
        if phys_gb is not None:
            phys_gb = round(phys_gb, 1)
        return available_gb, phys_gb

    def _memory_relief_opt_out(self) -> bool:
        return os.environ.get("AGENT_ENABLE_MEMORY_RELIEF", "").strip().lower() in (
            "0", "false", "no", "off",
        )

    def _mlx_coder_memory_pressure(self) -> tuple[bool, float | None, float | None]:
        """True when diffusers should unload before stacking more GPU work.

        **On by default** when system RAM is tight. Skips entirely for small MLX
        models (on-disk weights below ``AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB``,
        default 50). Uses **available** RAM (vm_stat / SC_AVPHYS_PAGES), not
        on-disk folder size — GLM-5.2's huge shard tree must not trip relief on
        a 512 GB box that still has hundreds of GB free.

        Opt out: ``AGENT_ENABLE_MEMORY_RELIEF=0``.
        Tune: ``AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB`` (default 64).
        """
        if self._memory_relief_opt_out():
            return False, None, None
        try:
            if not self._backend or getattr(self._backend.info, "name", None) != "mlx":
                return False, None, None
            disk_gb = self._mlx_model_on_disk_gb()
            small_disk_gb = float(
                os.environ.get("AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB", "50")
            )
            if disk_gb is not None and disk_gb < small_disk_gb:
                return False, disk_gb, None
            available_gb, phys_gb = self._available_system_memory_gb()
            if available_gb is None:
                return False, disk_gb, phys_gb
            min_available = float(
                os.environ.get("AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB", "64")
            )
            tripped = available_gb < min_available
            # Second value: available_gb when probed, else disk_gb for traces.
            metric = available_gb if available_gb is not None else disk_gb
            return tripped, metric, phys_gb
        except Exception:
            return False, None, None

    def _should_release_diffusers_after_media(self) -> bool:
        """Unload Z-Image before MLX codegen on tight unified-memory hosts.

        Fieldrunners trace 20260703 (96 GB Mac): vm_stat still showed >64 GB
        free after sprite gen, so the old available-RAM-only gate kept
        Z-Image resident on MPS while the 27B MLX coder prefilled — silent
        stall on iter 2. Also trip when physical RAM is at or below
        AGENT_MEMORY_RELIEF_MAX_PHYS_GB (default 128) so 96 GB-class boxes
        always drop diffusers after <assets>/<sounds> even if free pages
        look comfortable.
        """
        if self._memory_relief_opt_out():
            return False
        try:
            if not self._backend or getattr(self._backend.info, "name", None) != "mlx":
                return False
        except Exception:
            return False
        # Phys-RAM ceiling is independent of the small-model opt-out in
        # _mlx_coder_memory_pressure (Qwen3.6-27B ~30 GB on disk must still
        # unload Z-Image on 96 GB hosts).
        _, phys_gb = self._available_system_memory_gb()
        max_phys = float(
            os.environ.get("AGENT_MEMORY_RELIEF_MAX_PHYS_GB", "128")
        )
        if phys_gb is not None and phys_gb <= max_phys:
            return True
        tripped, _, _ = self._mlx_coder_memory_pressure()
        if tripped:
            return True
        return False

    def _maybe_release_diffusers_before_coder_stream(self) -> list[str]:
        """Drop in-process diffusers before a coder stream on MLX when relief trips.

        Same gate as post-<assets> unload — do NOT drop on roomy 512 GB boxes
        just because generators are still resident; only when memory pressure
        or the phys-RAM ceiling says to.
        """
        if not self._should_release_diffusers_after_media():
            return []
        return self._release_diffusers_vram()

    def _release_diffusers_vram(self) -> list[str]:
        """Drop in-process Z-Image / Stable-Audio pipelines; keep MLX LLM."""
        freed: list[str] = []
        try:
            import assets as _assets
            freed.extend(_assets.release_preloaded_diffusers())
        except Exception:
            pass
        try:
            gen = self._asset_generator
            if gen is not None and hasattr(gen, "cleanup"):
                gen.cleanup()
                label = "Z-Image-Turbo (session)"
                if label not in freed:
                    freed.append(label)
        except Exception:
            pass
        self._asset_generator = None
        try:
            sg = self._sound_generator
            if sg is not None and hasattr(sg, "cleanup"):
                sg.cleanup()
                freed.append("Stable-Audio")
        except Exception:
            pass
        self._sound_generator = None
        try:
            vg = self._video_generator
            if vg is not None:
                self._video_generator = None
                freed.append("VideoGenerator (session)")
        except Exception:
            pass
        return freed

    def _vision_judge_headroom_ok(self) -> bool:
        """Return False when stacking a local MLX-VLM on the resident coder
        LLM would likely trigger macOS memory-pressure kills. Same available-
        RAM gate as `_free_memory_before_video`; Ollama daemons are not measured.
        """
        tripped, _, _ = self._mlx_coder_memory_pressure()
        return not tripped

    def _free_memory_before_video(self) -> dict:
        """Free diffuser pipelines BEFORE launching the Wan video subprocess.

        Always releases Z-Image / Stable-Audio resident pipelines — assets and
        sounds are already on disk by Phase A video time, and keeping diffusers
        loaded alongside the MLX coder LLM + Wan subprocess triggered macOS
        jetsam (vm-compressor-space-shortage) on run_09 games 5–7 even when
        free-RAM looked plentiful. Never drops the in-process MLX LLM.
        """
        info: dict = {"freed": [], "forced": True, "available_gb": None}
        try:
            _, metric_gb, phys_gb = self._mlx_coder_memory_pressure()
            info["available_gb"] = metric_gb
            # Diffusers only (Z-Image / Stable-Audio) — NOT the MLX coder LLM.
            try:
                import assets as _assets
                info["freed"].extend(_assets.release_preloaded_diffusers())
            except Exception:
                pass
            try:
                sg = self._sound_generator
                if sg is not None and hasattr(sg, "cleanup"):
                    sg.cleanup()
                    self._sound_generator = None
                    info["freed"].append("Stable-Audio")
            except Exception:
                pass
            self._asset_generator = None
            self._trace({
                "kind": "video_memory_relief",
                "available_gb": info["available_gb"],
                "phys_gb": phys_gb,
                "freed": info["freed"],
                "forced": True,
            })
        except Exception as e:
            self._trace({"kind": "video_memory_relief_error", "err": str(e)})
        return info

    # ------------------------------------------------------------------ revert
    #
    # User-triggered escape hatch for "the model just broke a working game."
    # Audit of 10 recent traces (2026-05-25) showed harness gates catch only
    # ~5-10% of "model takes liberty beyond user scope" failures across the
    # full feedback shape. The cheaper, higher-leverage answer is to let the
    # user roll back a regression in one keystroke rather than typing "you
    # broke X, fix it" and watching the model break more.
    #
    # `/revert` calls revert_to_iter(None) → pick the most-recent iter whose
    # `iter_summary` event had ok=True. `/revert N` calls revert_to_iter(N)
    # → pick that specific snapshot. Both fall back to best.html when no
    # clean snapshot is available; both error cleanly when neither exists.
    #
    # Iter counter is intentionally NOT reset — we're rewinding the FILE,
    # not the conversation history. The next turn will see the reverted
    # bytes as CURRENT FILE ON DISK and any feedback the user types.

    def _clean_iters_from_trace(self) -> list[int]:
        """Read the trace JSONL and return iter numbers whose iter_summary
        event had ok=True, in chronological order. Source of truth is the
        trace file (not in-memory state) so /revert works even after a
        restart inside the same artifact.
        """
        out: list[int] = []
        try:
            if not self.trace_path.exists():
                return out
            with self.trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or '"iter_summary"' not in line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("kind") != "iter_summary":
                        continue
                    if not obj.get("ok"):
                        continue
                    it = obj.get("iteration")
                    if isinstance(it, int):
                        out.append(it)
        except Exception:
            return out
        return out

    def _resolve_revert_source(
        self, requested_iter: int | None,
    ) -> tuple[Path | None, int | None, str, str | None]:
        """Resolve the revert target. Returns (source_path, iter_n, source_kind, error_msg).

        - requested_iter=None → most-recent clean iter snapshot, else best.html
        - requested_iter=N    → snapshot for iter N specifically
        - error_msg non-None → callers surface to user; source_path is None
        """
        if requested_iter is not None:
            snap = self.snapshots_dir / f"iter_{requested_iter:02d}.html"
            if not snap.exists():
                available = sorted(
                    int(p.stem.split("_", 1)[1])
                    for p in self.snapshots_dir.glob("iter_*.html")
                    if p.stem.split("_", 1)[1].isdigit()
                ) if self.snapshots_dir.exists() else []
                avail_str = (
                    ", ".join(str(i) for i in available) if available
                    else "(none — no iters have completed yet)"
                )
                return None, None, "", (
                    f"iter {requested_iter} snapshot not found at "
                    f"{snap}. Available iters: {avail_str}"
                )
            return snap, requested_iter, "snapshot", None

        clean_iters = self._clean_iters_from_trace()
        if clean_iters:
            target_iter = clean_iters[-1]
            snap = self.snapshots_dir / f"iter_{target_iter:02d}.html"
            if snap.exists():
                return snap, target_iter, "snapshot", None
            # Fall through — clean iter was recorded but snapshot is gone

        if self.best_path.exists():
            return self.best_path, None, "best", None

        return None, None, "", (
            "no clean iter snapshot AND no best.html — nothing to revert to. "
            "(this can happen if no iter has shipped cleanly yet in this session.)"
        )

    def revert_to_iter(
        self, requested_iter: int | None = None,
    ) -> dict[str, Any]:
        """Roll back the on-disk game file to a previous iter's snapshot.

        Returns a dict the caller (TUI / CLI) can render:
            {
                "ok": bool,
                "error": str | None,       # set when ok is False
                "to_iter": int | None,     # iter we landed on, or None for best.html
                "source": "snapshot" | "best",
                "source_path": str,        # absolute path of the source file
                "from_iter": int | None,   # last iter that ran, for the banner
                "file_bytes": int,         # size of the file after revert
            }
        """
        source_path, to_iter, source_kind, err = self._resolve_revert_source(
            requested_iter,
        )
        if err is not None:
            return {
                "ok": False, "error": err, "to_iter": None,
                "source": "", "source_path": "", "from_iter": None,
                "file_bytes": 0,
            }
        try:
            content = source_path.read_text(encoding="utf-8")
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(content, encoding="utf-8")
        except Exception as e:
            return {
                "ok": False, "error": f"failed to write {self.out_path}: {e}",
                "to_iter": None, "source": "", "source_path": "",
                "from_iter": None, "file_bytes": 0,
            }
        from_iter = max(
            self._last_tested_iter or 0,
            self._last_materialized_iter or 0,
            self._snapshot_n or 0,
        )
        # Update in-memory file mirror so the next fix-prompt's CURRENT
        # FILE block reflects what's actually on disk.
        self._current_file = content
        # Reset stale per-iter state that would mislead the next turn.
        # We don't touch _messages or counters — the conversation
        # continues; only iter-local signals get cleared.
        self._previous_report_ok = None
        self._previous_report = None
        # T-2: per-iter code-hash trackers for shipped_unchanged_after_block.
        self._cur_iter_code_sha = None
        self._prev_iter_code_sha = None
        self._last_test_report = None
        self._last_failed_patch_anchors = set()
        self._pending_coaching = []
        self._last_no_usable_code_fingerprint = None
        # Recovery-flag flags from the round-1/2 Wolfenstein fix bundle —
        # any pending recovery prompt is stale because the file changed.
        self._dead_first_build_pending = False
        # Polish phase (item 2): an in-flight polish turn is stale after a
        # manual revert; the turn counter is NOT reset (per-session cap).
        self._polish_pending = False
        self._context_pressure_pending = False
        self._context_pressure_streak = 0
        # Format-stuck streak and stream-status flags reset too — the
        # last failure's evidence no longer reflects current state.
        self._format_stuck_streak = 0
        self._last_stream_looped = False
        self._last_stream_stalled = False
        self._last_stream_silent = False
        self._last_stream_crashed = False
        file_bytes = len(content)
        self._trace({
            "kind": "user_revert",
            "from_iter": from_iter,
            "to_iter": to_iter,
            "source": source_kind,
            "source_path": str(source_path),
            "file_bytes": file_bytes,
        })
        return {
            "ok": True, "error": None,
            "to_iter": to_iter, "source": source_kind,
            "source_path": str(source_path),
            "from_iter": from_iter,
            "file_bytes": file_bytes,
        }

    def _save_best(self, html: str) -> Path | None:
        try:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.best_path.write_text(html, encoding="utf-8")
            return self.best_path
        except Exception:
            return None

    def _read_best_or_empty(self) -> str:
        try:
            if self.best_path.exists():
                return self.best_path.read_text(encoding="utf-8")
        except Exception:
            pass
        return ""

    @staticmethod
    def _file_hash16(path: Path | str | None) -> str | None:
        """Return a short sha256 for a media file, or None if unreadable."""
        if not path:
            return None
        try:
            import hashlib

            p = Path(path)
            if not p.exists() or not p.is_file():
                return None
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 128), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception:
            return None

    def _dump_conversation(self) -> None:
        # One-complete-trace (2026-06-14): the .conversation.md sibling is
        # retired. Canonical user turns (with all injected blocks) and assistant
        # replies are captured directly in the .jsonl; the system prompt has a
        # compact hash/size manifest. This redundant file is no longer written.
        # Kept as a no-op (rather than ripping out ~6 call sites) to keep the
        # change surgical; existing .conversation.md files on disk are untouched.
        return
        try:
            self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = [
                f"# Conversation dump — {self.model}",
                f"_session: {self._session_id}_  ",
                f"_artifact: {self._artifact_id}_  ",
                f"_goal: {self._goal or '(not set)'}_  ",
                f"_trace: {self.trace_path}_  ",
                f"_game: {self.out_path}_  ",
                f"_iteration count: {self._snapshot_n}_  ",
                f"_messages: {len(self._messages)}_",
                "",
            ]
            for i, msg in enumerate(self._messages):
                role = msg.get("role", "?")
                content = msg.get("content", "") or ""
                model_role = msg.get("model_role")
                model_name = msg.get("model_name")
                
                role_label = f"{role.upper()}"
                if role == "assistant":
                    m_role = model_role or "coder"
                    m_name = model_name or self.model
                    role_label += f" — Role: {m_role} ({m_name})"
                elif role == "system" and model_role:
                    role_label += f" — Role: {model_role}"
                    
                lines.append(f"## [{i:02d}] {role_label}")
                lines.append("")
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
            self.conversation_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    # -- conversation pruning ----------------------------------------------

    _SUMMARIZE_HTML_RE = re.compile(
        r"<html_file>\s*(.*?)\s*</html_file>", re.DOTALL | re.IGNORECASE
    )
    _SUMMARIZE_FENCE_RE = re.compile(
        r"```(?:html|HTML)?\n(.*?)\n```", re.DOTALL
    )
    # Capability-round item 3: stale probe re-emissions in pruned
    # assistant turns.
    _SUMMARIZE_PROBES_RE = re.compile(
        r"<probes>\s*(.*?)\s*</probes>", re.DOTALL | re.IGNORECASE
    )

    @staticmethod
    def _endpoint_supports_concurrency(endpoint: str) -> bool:
        """Fix #5 — does this endpoint accept concurrent requests in
        parallel, or do they queue at a single local daemon?

        Heuristic: **loopback URLs serialize** (a single Ollama daemon
        on 127.0.0.1:11434 handles one stream at a time). **Non-loopback
        HTTP(S) endpoints are assumed to support concurrency** — cloud
        provider APIs (Anthropic, OpenAI, future providers) accept many
        concurrent requests per account, and self-hosted multi-worker
        inference servers on a LAN typically do too.

        Detection is by ENDPOINT SHAPE, not provider name. Adding a
        new provider in the future works automatically; no string-match
        on `"anthropic"` or `"openai"` anywhere.
        """
        if not endpoint:
            return False
        ep = endpoint.lower()
        # Loopback patterns — single-daemon, serialize.
        loopback_markers = ("127.", "localhost", "[::1]", "0.0.0.0")
        return not any(m in ep for m in loopback_markers)

    def _available_sampler_slots(self) -> list[tuple["Backend", str]]:
        """List of (backend, label) tuples that can run a best-of-N
        candidate independently of slot 1. Excludes any slot whose
        endpoint matches slot 1's (would queue at the same daemon)
        AND any slot currently busy with a concurrent critic task.

        Always returns slot 1 first; slots 2 and 3 only when truly
        independent. On a single-slot / single-GPU config this returns
        just slot 1, which lets the caller fall back to the existing
        sequential best_of_n path.
        """
        out: list[tuple["Backend", str]] = [(self._backend, "slot1")]
        seen_endpoints: set[str] = set()
        try:
            ep1 = getattr(getattr(self._backend, "info", None), "endpoint", "") or ""
            if ep1:
                seen_endpoints.add(ep1)
        except Exception:
            pass

        # Detect which slot the critic is currently using so we don't
        # steal it from a concurrent visual-critic task spawned by Phase 1A.
        critic_busy_backend = None
        if self._critic_task is not None and not self._critic_task.done():
            try:
                critic_busy_backend = self.get_backend("critic")
            except Exception:
                critic_busy_backend = None

        for label, bk in (
            ("slot2", getattr(self, "_backend2", None)),
            ("slot3", getattr(self, "_backend3", None)),
        ):
            if bk is None or bk is self._backend:
                continue
            try:
                ep = getattr(getattr(bk, "info", None), "endpoint", "") or ""
            except Exception:
                ep = ""
            if (
                ep
                and ep in seen_endpoints
                and not self._endpoint_supports_concurrency(ep)
            ):
                # Same loopback daemon as another slot — would just
                # queue. Fix #5: for non-loopback endpoints (cloud
                # APIs, LAN inference servers) concurrent requests
                # run in parallel and we DO want both slots in the
                # fan-out set.
                continue
            if critic_busy_backend is not None and bk is critic_busy_backend:
                continue
            if ep:
                seen_endpoints.add(ep)
            out.append((bk, label))
        return out

    @staticmethod
    def _should_escalate_stuck_bon(
        stuck_streak: int,
        best_of_n: int,
        escalations_used: int,
        cap: int = _STUCK_BON_ESCALATION_CAP,
        last_report: dict | None = None,
    ) -> bool:
        """Pure trigger predicate for item 5 (stuck best-of-2 repair).

        Fires only for best-of-1 sessions: a session-wide best_of_n > 1
        already samples every turn, so escalation would be redundant.

        Fix-round item 5: when the last report shows ALL probes passing and
        zero errors/page_errors, the blocker is a deterministic audit
        verdict (sprite-audit soft_warnings) — resampling cannot change it,
        so don't spend an escalation (~25 min of sequential MLX sampling in
        trace 20260610_185238 went to exactly this). Escalation stays
        reserved for failing probes / runtime errors, which fresh samples
        CAN fix.
        """
        if not (
            stuck_streak >= 2
            and best_of_n == 1
            and escalations_used < cap
        ):
            return False
        if isinstance(last_report, dict):
            probes = last_report.get("probes") or []
            probes_green = bool(probes) and all(p.get("ok") for p in probes)
            no_errors = (
                not last_report.get("errors")
                and not last_report.get("page_errors")
            )
            if probes_green and no_errors:
                return False
        return True

    def _bon_candidate_path(self, candidate_index: int) -> Path:
        """Visible on-disk path for a best-of-N candidate HTML file."""
        iter_n = self._snapshot_n + 1
        d = self.out_path.parent / "candidates" / f"iter_{iter_n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"cand_{candidate_index}.html"

    def set_stuck_bon_enabled(self, enabled: bool) -> None:
        """Enable/disable automatic stuck best-of-2 escalation."""
        self._stuck_bon_enabled = bool(enabled)

    async def _generate_and_score_candidates(
        self,
        n: int,
    ) -> tuple[Candidate, list[Candidate]]:
        """Sample N completions and score each by running its result through
        the test harness against a temp file. Used when fixing a failed iter.

        Phase 2A: when multiple independent slots are available, fan out
        across them in parallel. When only slot 1 is independent (single-
        slot / single-GPU / critic holding the others), fall back to the
        existing sequential path.

        Scorer is a continuous quality score in [0, 100] from
        `score_test_report` so partial-credit candidates ("almost works")
        win over fully-broken ones. Pass = 100; "applied but several
        errors" lands ~30; "wouldn't apply at all" returns -10 so it
        always loses to anything that ran.
        """
        # We deliberately DO NOT stream tokens for these candidates — only
        # the winning one is replayed visually after we pick it.
        _score_idx = [0]

        async def scorer(text: str) -> tuple[float, dict]:
            cand_i = _score_idx[0]
            _score_idx[0] += 1
            extra: dict = {
                "kind": "candidate",
                "text_len": len(text),
                "candidate_index": cand_i,
            }
            scoped_violation = self._scoped_reply_violation(text)
            if scoped_violation:
                extra["scoped_violation"] = scoped_violation
                return -20.0, extra
            html, applied_msg = await self._materialize(text, dry_run=True)
            extra["materialized"] = bool(html)
            extra["materialize_msg"] = applied_msg
            if not html:
                return -10.0, extra
            # Fix round: candidates MUST be tested from the same directory as
            # the real game file (out_path.parent) so relative asset/sound
            # paths like ./<session>_assets/x.png resolve. Visible paths under
            # candidates/iter_NN/ so you can open them in Chrome for manual
            # testing — no dotfiles, no auto-delete.
            cand_path = self._bon_candidate_path(cand_i)
            extra["candidate_path"] = str(cand_path)
            try:
                cand_path.write_text(html, encoding="utf-8")
                report = await self.browser.load_and_test(
                    cand_path, screenshot_path=None,
                    probes=self._probes or None,
                    opening_book_recipes=getattr(self, "_active_opening_book_recipes", []),
                    # todo #2: pass criteria so the harness can flag
                    # coverage gaps even on best-of-N candidate scoring.
                    criteria=self._criteria or None,
                    goal=self._goal or "",
                    visual_recipe_id=getattr(self, "_active_visual_playtest_recipe_id", None),
                )
                extra["report_ok"] = report.get("ok", False)
                extra["report_summary"] = format_report_for_model(report)[:400]
                return score_test_report(report), extra
            except Exception as e:
                extra["scorer_exception"] = str(e)
                # Scorer crashed — treat as worse than "applied but
                # broken" but better than "didn't apply".
                return 10.0, extra

        # Phase 2A — parallel fan-out across independent slots.
        slots = self._available_sampler_slots()
        if len(slots) >= 2 and n >= 2:
            winner, all_cands = await self._fan_out_best_of_n_across_slots(
                slots=slots, n=n, scorer=scorer,
            )
            return winner, all_cands

        # Single-slot fallback — preserved exactly as before.
        winner, all_cands = await self._backend.best_of_n(
            self._messages,
            n=n,
            options={"num_ctx": self.num_ctx},
            keep_alive=self._keep_alive_for_backend(self._backend),
            stall_seconds=self.stall_seconds,
            overall_seconds=self.overall_seconds,
            scorer=scorer,
            # score_test_report() returns 0-100; passing test = 100, so
            # we early-exit at 100 (any partial-credit candidate keeps
            # sampling).
            early_exit_score=100.0,
            on_progress=lambda i, msg: self._trace({
                "kind": "best_of_n_progress",
                "candidate": i,
                "msg": msg,
                "candidate_dir": str(
                    self.out_path.parent / "candidates"
                    / f"iter_{self._snapshot_n + 1:02d}"
                ),
            }),
            cancel_event=self._ensure_stop_event(),
        )
        return winner, all_cands

    async def _fan_out_best_of_n_across_slots(
        self,
        *,
        slots: list[tuple["Backend", str]],
        n: int,
        scorer,
    ) -> tuple[Candidate, list[Candidate]]:
        """Phase 2A — sample N candidates in parallel across `slots`.

        Each candidate runs on a different slot with a different
        temperature (and optional alternate fix-strategy prompt for
        2B). Slot endpoints are independent — verified by
        `_available_sampler_slots` — so the LLM daemons don't queue
        the requests. Candidates are scored sequentially against the
        SAME Chromium browser (single LiveBrowser instance) to avoid
        race conditions on the test harness; only the LLM generation
        is parallel. That's where the time savings live anyway
        (generation: 30-120 s vs scoring: 5-10 s).
        """
        from backend import Candidate as _Candidate
        # Pair each candidate with a slot + temperature. When n > len(slots),
        # subsequent candidates wrap back to slot 1 to keep total count = n.
        temps = [0.2, 0.6, 0.9]
        while len(temps) < n:
            temps.append(temps[-1])
        pairs: list[tuple["Backend", str, float]] = []
        for i in range(n):
            bk, label = slots[i % len(slots)]
            pairs.append((bk, label, temps[i]))

        cancel = self._ensure_stop_event()

        async def _run_one(idx: int, backend, label: str, temp: float):
            t0 = asyncio.get_event_loop().time()
            try:
                result = await backend.stream_chat(
                    self._messages,
                    on_token=None,
                    options={"num_ctx": self.num_ctx, "temperature": temp},
                    keep_alive=self._keep_alive_for_backend(backend),
                    stall_seconds=self.stall_seconds,
                    overall_seconds=self.overall_seconds,
                    max_retries=0,
                    cancel_event=cancel,
                )
            except Exception as e:
                self._trace_exception(
                    "best_of_n_candidate_error", e,
                    candidate=idx, slot=label, temperature=temp,
                )
                return _Candidate(
                    text="", score=-100.0,
                    extra={"slot": label, "err": str(e)[:200]},
                    tokens=0, duration_s=0.0, stalled=True,
                )
            duration = asyncio.get_event_loop().time() - t0
            self._trace({
                "kind": "best_of_n_candidate_generated",
                "candidate": idx,
                "slot": label,
                "temperature": temp,
                "tokens": result.tokens,
                "duration_s": round(duration, 2),
            })
            return _Candidate(
                text=result.text,
                score=0.0,  # scored below
                extra={"slot": label, "temperature": temp},
                tokens=result.tokens,
                duration_s=duration,
                stalled=result.stalled,
            )

        # Phase: fan out generations in parallel.
        gen_tasks = [
            _run_one(i, bk, label, temp)
            for i, (bk, label, temp) in enumerate(pairs)
        ]
        candidates_raw = await asyncio.gather(*gen_tasks, return_exceptions=False)

        # Phase: score sequentially (single Chromium).
        scored: list[_Candidate] = []
        for i, c in enumerate(candidates_raw):
            if not c.text:
                scored.append(c)
                continue
            try:
                score, extra = await scorer(c.text)
            except Exception as e:
                score = -1.0
                extra = {"scorer_error": str(e)[:200]}
            extra = {**(c.extra or {}), **extra}
            scored.append(_Candidate(
                text=c.text, score=score, extra=extra,
                tokens=c.tokens, duration_s=c.duration_s, stalled=c.stalled,
            ))
            self._trace({
                "kind": "best_of_n_candidate_scored",
                "candidate": i,
                "slot": (c.extra or {}).get("slot"),
                "score": round(score, 2),
                "report_ok": extra.get("report_ok"),
            })

        # Highest score wins; tie-break by shorter text (smaller diff).
        scored.sort(key=lambda c: (c.score, -len(c.text)), reverse=True)
        winner = scored[0] if scored else None
        if winner is not None:
            winning_slot = (winner.extra or {}).get("slot", "?")
            # Phase 3 surprise: a non-slot-1 candidate winning is a
            # signal worth surfacing — it means the alternate slots'
            # extra capacity is actually pulling its weight, and
            # justifies the multi-slot configuration.
            self._trace({
                "kind": "best_of_n_attempt",
                "n": n,
                "winner_slot": winning_slot,
                "winner_score": round(winner.score, 2),
                "winner_temperature": (winner.extra or {}).get("temperature"),
                "candidate_summary": [
                    {
                        "slot": (c.extra or {}).get("slot"),
                        "temperature": (c.extra or {}).get("temperature"),
                        "score": round(c.score, 2),
                        "tokens": c.tokens,
                    }
                    for c in scored
                ],
            })
            if winning_slot != "slot1":
                self._trace({
                    "kind": "surprise",
                    "category": "non_slot1_bon_winner",
                    "winner_slot": winning_slot,
                    "winner_score": round(winner.score, 2),
                    "hint": (
                        "An alternate slot won the fan-out — the "
                        "additional slot capacity is generating better "
                        "candidates than slot 1 alone. Worth comparing "
                        "the winning temperature against slot 1's for "
                        "schedule tuning."
                    ),
                })
        return winner, scored

    # -- text → file: extract patches OR <html_file> -----------------------

    @staticmethod
    def _extract_notes(reply: str) -> str | None:
        reply = _strip_thinking(reply)
        m = _NOTES_RE.search(reply)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_criteria(reply: str) -> str | None:
        reply = _strip_thinking(reply)
        m = _CRITERIA_RE.search(reply)
        return m.group(1).strip() if m else None

    # (Removed 2026-05-30: `_ARCHITECT_KEYWORDS` + `_is_complex_goal` gated the
    # separate `<architect>` prose turn, which was merged into the single
    # planning pass. The architect ROLE and exit-decision turn are unchanged.)

    @staticmethod
    def _failure_blames_code(report: dict[str, Any]) -> bool:
        """True only when a failing iter is attributable to a real CODE
        defect — so playbook bullets should be penalized (`harmful++`).

        Why this gate exists (trace 20260613_213711, Opus 4.8): a failing
        iter used to bump `harmful` on every injected bullet whenever the
        stuck-streak hit 3, regardless of WHY it failed. But the blocker
        there was a model-authored `<probes>` failure (`state_room0` racing
        the input smoke test) — nothing to do with the QTE playbook bullets,
        which were correct. Penalizing them corrupted hand-curated seeds.

        Model-probe-authoring artifacts (the model's own failing `<probes>`
        and synthetic `coverage_gap__*` probes) surface as soft_warnings
        prefixed `PROBE FAILED [`. Genuine code defects surface either as
        page_errors or as soft_warnings WITHOUT that prefix (FROZEN-CANVAS,
        ENTITY-NOT-RENDERED, HEURISTIC controls-not-wired, OPENING BOOK
        CHECK FAILED, JS-SOURCE-IN-BODY, …). The harness-synthesized
        `input_responsive` probe is a real behavioral defect (controls not
        wired), so a failing one counts as code-attributable even though it
        carries the PROBE FAILED prefix. Genre-free, model-agnostic.
        """
        if report.get("page_errors"):
            return True
        soft = report.get("soft_warnings") or []
        non_probe = [
            str(w) for w in soft
            if not str(w).startswith("PROBE FAILED [")
        ]
        # Harness per-turn cap drops — not a code defect; autogen handles it.
        if non_probe and all(
            "ASSETS_DROPPED_PENDING" in w for w in non_probe
        ):
            return False
        for w in soft:
            if not str(w).startswith("PROBE FAILED ["):
                return True
        # Harness behavioral probe (controls-not-wired) is a real defect.
        for p in (report.get("probes") or []):
            if (not p.get("ok")) and p.get("name") == "input_responsive":
                return True
        return False

    @staticmethod
    def _extract_todos(reply: str) -> str | None:
        """Pull a <todos>...</todos> block out of a reply.

        Returns the body text trimmed of leading/trailing whitespace,
        or None when no block is present. The body is the literal
        checklist as the model wrote it — we don't normalize the
        `[ ]` / `[x]` markers because models use varied dialects
        (`- [ ]`, `* [ ]`, `[ ]`) and forcing a single shape would
        cost more parses than it saves.
        """
        reply = _strip_thinking(reply)
        m = _TODOS_RE.search(reply)
        if not m:
            return None
        body = m.group(1).strip()
        return body or None

    @staticmethod
    def _parse_todo_items(text: str) -> list[tuple[bool, str]]:
        """Parse a <todos> body into (done, text) items.

        Tolerant of the marker dialects models actually use:
        `[ ] item`, `- [ ] item`, `* [x] item`, `[X] item`. Lines
        without a checkbox marker are skipped (headers, blank lines).
        Added 2026-06-12 for todo-driven execution — the captured
        text used to be opaque; now the agent can select the next
        unchecked item as the turn's CURRENT TASK.
        """
        items: list[tuple[bool, str]] = []
        if not text:
            return items
        import re as _re
        pat = _re.compile(r"^\s*[-*]?\s*\[([ xX])\]\s*(.+?)\s*$")
        for line in text.splitlines():
            m = pat.match(line)
            if not m:
                continue
            done = m.group(1).lower() == "x"
            body = m.group(2).strip()
            if body:
                items.append((done, body))
        return items

    # Phase 2 (task ledger): split a multi-part goal/edit into ordered step
    # strings. Genre-free — purely structural splitting of the user's own
    # words on list markers / newlines / `then` / `;` / commas.
    _STEP_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
    _STEP_INLINE_SPLIT_RE = re.compile(r"\s*(?:;|,|\bthen\b)\s*", re.IGNORECASE)
    # P1a (run_04 trace): a SINGLE-LINE goal longer than this many words is
    # treated as descriptive PROSE, not an imperative list — we do NOT comma-
    # split it (see _parse_task_steps). 30 words comfortably covers a real
    # "do X, then Y, and add Z" ask while excluding 200-word game specs.
    _MAX_INLINE_SPLIT_WORDS = 30
    # Mid-line enumerated markers ("1) … 2) …", "- … - …") — when present in a
    # long single line the user really did write a list, so splitting is fine.
    _INLINE_LIST_MARKER_RE = re.compile(r"(?:^|\s)(?:\d+[.)]|[-*•])\s+\S")

    @staticmethod
    def _parse_task_steps(text: str) -> list[str]:
        """Split a multi-part goal into ordered steps for the <todos> seed.

        Used ONLY to seed the ledger when the model hasn't emitted its own
        <todos> yet (multi-part seed edits / comma-listed asks). Conservative:
        explicit list lines win; otherwise split a single line on `;` / `,` /
        `then`. Fragments under 3 words are dropped, deduped, capped at 8.
        Bare " and " is intentionally NOT a separator (compound nouns like
        "cat and mouse" must not split). Returns [] for a terse single clause.

        P1a (run_04 holochess/Dragon traces): a long DESCRIPTIVE single-line
        paragraph (a rich game spec, 200 words of commas) must NOT be comma-
        split — doing so produced 8 sentence-FRAGMENT "todos" (e.g. "but a
        chess game that teleports pieces…") that the todo contract then nagged
        one-by-one. Only inline-split a single line when it is SHORT
        (<= _MAX_INLINE_SPLIT_WORDS words) OR carries explicit enumerated list
        markers; otherwise return [] so the caller falls back to the outline
        build ORDER (or seeds no ledger at all). Multi-line goals are unchanged
        (genuine line-per-step lists).
        """
        if not text or not text.strip():
            return []
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(raw_lines) > 1:
            candidates = raw_lines
        else:
            line = raw_lines[0] if raw_lines else text.strip()
            is_prose = (
                len(line.split()) > GameAgent._MAX_INLINE_SPLIT_WORDS
                and not GameAgent._INLINE_LIST_MARKER_RE.search(line)
            )
            if is_prose:
                return []
            candidates = GameAgent._STEP_INLINE_SPLIT_RE.split(line)
        steps: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            c = GameAgent._STEP_LIST_MARKER_RE.sub("", c or "").strip().rstrip(".")
            if len(c.split()) < 3:
                continue
            key = " ".join(c.lower().split())
            if key in seen:
                continue
            seen.add(key)
            steps.append(c)
            if len(steps) >= 8:
                break
        return steps

    def _outline_order_steps(self) -> list[str]:
        """Phase 2 source (b): the matched implementation outline's build
        ORDER, as a checklist. For a complex FRESH build (Fieldrunners-class)
        the goal is one descriptive clause, but the opening book's outline
        already carries an ordered recipe (`order: BFS -> placement -> draw`).
        Surface it as steps so iter-2+ patches target the next build phase
        instead of wandering. Gated on a real match (score >= the open-domain
        floor) so a weak fallback match doesn't seed irrelevant steps. Entries
        are already curated step phrases, so they skip the word-count filter."""
        try:
            hit = self._memory.retrieve_implementation_outline(self._goal)
        except Exception:
            return []
        if hit is None or getattr(hit, "score", 0.0) < self._OPEN_DOMAIN_OUTLINE_FLOOR:
            return []
        recipe = getattr(hit.item, "recipe", None)
        order = (recipe or {}).get("order") if isinstance(recipe, dict) else None
        steps = [str(s).strip() for s in (order or []) if str(s).strip()]
        return steps[:8]

    def _seed_task_ledger_from_goal(self) -> None:
        """Seed `_todos_text` / `_todos_items` when the model has not emitted a
        <todos> block yet. Gives a weak local model a ONE-objective-per-turn
        checklist from turn 1 (the biggest reliability lever) instead of waiting
        for it to invent one. The model's own first <todos> emission overwrites
        this seed via `_capture_todos`, so model control is preserved.

        Two sources, in priority order:
          (a) the goal's own clauses (multi-part seed edits / comma-listed asks)
          (b) the matched outline's build ORDER (complex fresh builds whose goal
              is a single descriptive clause) — only on a fresh build.
        No-op for a terse single-clause goal with no strong outline match."""
        if self._todos_items:
            return
        steps = self._parse_task_steps(self._goal)
        source = "goal_clauses"
        if len(steps) < 2 and self.seed_file is None:
            outline_steps = self._outline_order_steps()
            if len(outline_steps) >= 3:
                steps = outline_steps
                source = "outline_order"
        if len(steps) < 2:
            return
        self._todos_text = "\n".join(f"- [ ] {s}" for s in steps)
        self._todos_items = self._parse_todo_items(self._todos_text)
        self._todos_seeded_by_harness = True
        # Phase 4B: remember the seed source so the one-objective-first-build
        # experiment can gate on "outline_order" (strong-match complex build).
        self._ledger_source = source
        self._trace({
            "kind": "task_ledger_seeded",
            "source": source,
            "step_count": len(steps),
            "steps": steps,
        })

    # Tokens too common to identify a ledger step (skipped when token-matching
    # a step against the materialized file for harness done-marking).
    _LEDGER_STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "add", "make", "draw", "show", "use", "set", "then", "each", "all",
        "your", "this", "that", "it", "is", "are", "be", "do", "game", "screen",
    })

    def _mark_ledger_progress(self) -> None:
        """Phase 2: conservatively mark HARNESS-SEEDED ledger steps done after a
        materialize. A step is marked done ONLY when ALL of its distinctive
        tokens (>=2 of them, stopwords removed) appear in the current file —
        an under-claiming heuristic (missing a done step is safe; falsely
        marking one done would let the model skip real work, so we require a
        full multi-token match). No-op once the model owns its own <todos>."""
        if not self._todos_seeded_by_harness or not self._todos_items:
            return
        haystack = (self._current_file or "").lower()
        if not haystack:
            return
        new_items: list[tuple[bool, str]] = []
        changed = False
        for done, text in self._todos_items:
            if not done:
                toks = [
                    t for t in re.findall(r"[a-zA-Z]{3,}", text.lower())
                    if t not in self._LEDGER_STOPWORDS
                ]
                if len(toks) >= 2 and all(t in haystack for t in toks):
                    done = True
                    changed = True
            new_items.append((done, text))
        if changed:
            self._todos_items = new_items
            self._todos_text = "\n".join(
                f"- [{'x' if d else ' '}] {t}" for d, t in new_items
            )
            self._trace({
                "kind": "task_ledger_progress_marked",
                "done": sum(1 for d, _ in new_items if d),
                "total": len(new_items),
            })

    def _capture_todos(self, reply: str) -> None:
        """If the reply contains a <todos> block, update in-memory
        state and persist to disk. No-op when absent — the model
        controls the cadence.
        """
        todos = self._extract_todos(reply)
        if not todos:
            return
        # Cap at 6 KB so a runaway emission can't bloat the state-
        # anchor (which is already char-budgeted by compaction).
        todos = todos[:6000]
        self._todos_text = todos
        # Keep the structured view in sync for todo-driven execution.
        self._todos_items = self._parse_todo_items(todos)
        # Phase 2: the model now owns the ledger — stop harness done-marking.
        self._todos_seeded_by_harness = False
        # One-complete-trace (2026-06-14): the .todos.md sibling is retired.
        # The full todos text is now stored inline in the todos_captured trace
        # event (was len + file path only), so the .jsonl carries it and no
        # redundant file is written.
        self._trace({
            "kind": "todos_captured",
            "len": len(todos),
            "todos": todos,
        })

    @staticmethod
    def _norm_todo(text: str) -> str:
        """Whitespace-collapsed lowercase form for fuzzy todo matching
        (models lightly rephrase items when re-emitting the list)."""
        return " ".join((text or "").lower().split())

    def _todo_drift_check(self) -> None:
        """After a clean iter that was given a CURRENT TASK contract,
        check the model marked the task [x] in its re-emitted <todos>.
        Mismatch fires a `todo_drift` trace event — telemetry ONLY,
        never a cutoff or a forced retry. Consumes `_current_todo`.
        """
        if not self._current_todo:
            return
        want = self._norm_todo(self._current_todo)
        still_open = any(
            (not done) and (
                self._norm_todo(t) == want
                or want in self._norm_todo(t)
                or self._norm_todo(t) in want
            )
            for done, t in self._todos_items
        )
        if still_open:
            self._trace({
                "kind": "todo_drift",
                "todo": self._current_todo[:200],
                "detail": "clean iter but CURRENT TASK still "
                          "unchecked in re-emitted <todos>",
            })
        self._current_todo = None

    def _select_next_todo(self) -> str | None:
        """First unchecked todo eligible for a CURRENT TASK contract.

        Returns None when the list is empty, everything is checked, or
        the first unchecked item has already been nagged twice (the
        model keeps declining it — stop re-asserting so the prompt
        doesn't bloat; mirrors the asset-reprompt cap philosophy).
        """
        for done, body in getattr(self, "_todos_items", []) or []:
            if done:
                continue
            key = self._norm_todo(body)
            if self._todo_nag_counts.get(key, 0) >= 2:
                continue
            return body
        return None

    @staticmethod
    def _classify_stall(err_text: str) -> dict | None:
        """Parse a stream-failure exception message for a no-tokens
        stall signature. Returns a dict with stall_seconds if matched,
        else None.

        Model-agnostic: substring + regex over the message, no backend
        identity check. The MLX backend produces:
            "Model produced no tokens before stalling at 60.0s"
        Future Ollama / other backends emitting similar text are
        caught by the same matcher.
        """
        if not err_text or "no tokens" not in err_text:
            return None
        import re
        m = re.search(r"stalling at\s+([0-9]+(?:\.[0-9]+)?)\s*s", err_text)
        seconds = float(m.group(1)) if m else None
        return {
            "kind": "no_tokens_stall",
            "stall_seconds": seconds,
            "message_preview": err_text[:400],
        }

    async def _try_extension_backend_fallback(
        self,
        *,
        stall: dict | None,
        iteration: int,
    ) -> tuple[bool, str]:
        """Try one local fallback (cloud -> mlx OR ollama) on extension stalls.

        P0c (MK trace 20260528): when Anthropic / OpenAI fails, fall back
        to ANY available local backend, not just Ollama. The original
        loop only accepted `resolved.name == "ollama"`, so MK trace
        landed on `resolved mlx:DeepSeek-V4-Flash (not ollama)` and gave
        up — even though MLX was the active local backend the user had
        loaded. The new loop accepts mlx OR ollama (whichever the host
        detection resolves first), matching the original intent of the
        function: escape a transient cloud outage by switching to
        whatever local model is available.

        MLX stays excluded as the SOURCE backend: a user who picked MLX
        wants the local model, and silently switching to Ollama on a
        transient Metal / Ctrl+D stall produces a confusing cross-
        backend error cascade (DK trace 20260523_081532). MLX failures
        still surface with the MLX-specific recovery hint and stop
        there; only the cloud→local safety net is broadened.
        """
        info = getattr(self._backend, "info", None)
        backend_name = getattr(info, "name", None)
        if backend_name == "mlx":
            return False, (
                "MLX stall — staying on MLX (no silent fallback to Ollama). "
                "Use the MLX recovery hint above, or switch backends explicitly "
                "with /backend ollama + /load <N>."
            )
        if backend_name not in ("anthropic", "openai"):
            return False, (
                f"fallback skipped: current backend is {backend_name} "
                "(only cloud backends fall back to a local model)"
            )
        if not stall or stall.get("kind") != "no_tokens_stall":
            return False, "fallback skipped: stall shape is not no-token"

        # Accept any local backend (mlx OR ollama). detect_backend(prefer="auto")
        # already implements the host's preference order; we just stop
        # rejecting a non-ollama result.
        candidate = None
        errs: list[str] = []
        for prefer in ("auto", "mlx", "ollama"):
            try:
                resolved = detect_backend(prefer=prefer)
            except Exception as e:
                errs.append(f"{prefer}: {e}")
                continue
            if resolved.name in ("mlx", "ollama"):
                candidate = resolved
                break
            errs.append(
                f"{prefer}: resolved {resolved.name}:{resolved.model} "
                "(not a local backend)"
            )
        if candidate is None:
            reason = " | ".join(errs) if errs else "no local backend available"
            self._trace({
                "kind": "extension_backend_fallback_unavailable",
                "iteration": iteration,
                "reason": reason[:500],
            })
            return False, (
                "Extension fallback unavailable: no local MLX or Ollama "
                f"backend could be resolved ({reason[:220]})."
            )

        old = self._backend
        old_name = f"{old.info.name}:{old.info.model}"
        try:
            new_backend = make_backend(candidate)
        except Exception as e:
            self._trace({
                "kind": "extension_backend_fallback_failed",
                "iteration": iteration,
                "reason": f"make_backend failed ({candidate.name}): {e}",
            })
            return False, (
                f"Extension fallback failed while initializing "
                f"{candidate.name} backend: {e}"
            )
        try:
            await old.close()
        except Exception:
            pass
        self._backend = new_backend
        self._trace({
            "kind": "extension_backend_fallback_switched",
            "iteration": iteration,
            "from": old_name,
            "to": f"{candidate.name}:{candidate.model}",
            "local_kind": candidate.name,
        })
        return True, (
            f"Extension fallback: switched backend from {old_name} to "
            f"{candidate.name}:{candidate.model} for this turn."
        )

    async def run(
        self,
        goal: str,
        *,
        continuation: bool = False,
        plan_only: bool = False,
        patch_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Drive a planning + iteration session.

        patch_only=True: skip Phase A planning + phase_a asset/sound generation
        and jump straight to the seed-file build loop. Requires seed_file.
        Used by eval/eval_seed_edits.py --patch-only to measure patch
        materialization without plan-time GPU burn.

        plan_only=True: run ONLY Phase A — plan, criteria, probes, the
        plan-time memory injection and analysis — then emit a consolidated
        `plan_summary` trace and return BEFORE any asset/sound generation or
        build iteration. No diffuser, no browser. Used by the prompt-eval
        harness (eval/eval_prompts_plan.py) to test a critical feature of each
        prompt cheaply while keeping a complete, informative trace.

        continuation=False (default): fresh session. Resets _messages, runs
        phase A planning, picks a memory skeleton, and seeds the first build.

        continuation=True: extend a previously-finished session. `goal` is
        treated as new user feedback for the existing game on disk; planning
        and first-build are skipped, the iteration loop resumes immediately
        with a continuation prompt. _messages, _current_file, _snapshot_n,
        browser, and model are all reused.

        BUG fixed in this version: `self._goal` was being OVERWRITTEN with
        the new feedback on continuation, so the structured-compaction
        summary's "Goal" line ended up reading "use the art you generate"
        (the latest feedback) instead of the original "doom shooter"
        request. Now we only set `_goal` on a fresh session and treat the
        continuation `goal` arg as feedback, leaving the original intact.
        """
        # ---- run() TOC (see AGENTS.md §1b) --------------------------------
        # Session init + continuation baseline sanitize
        # ---- PHASE A: planning ------------------------------------------
        # ---- seed file OR memory skeleton for the first build ----------
        # ---- PHASE B: build/iterate -------------------------------------
        self._run_session_complete = False
        self._crash_traced_this_session = False
        try:
            async for ev in self._run_phase_a_and_first_build(
                goal,
                continuation=continuation,
                plan_only=plan_only,
                patch_only=patch_only,
            ):
                yield ev
            if self._run_session_complete:
                return
            async for ev in self._run_build_iterate_loop(
                continuation=continuation,
            ):
                yield ev
            if self._run_session_complete:
                return
            async for ev in self._run_exit_and_finalize():
                yield ev
        except Exception as e:
            self._trace_agent_crash(e, source="run")
            raise

    async def _run_phase_a_and_first_build(
        self,
        goal: str,
        *,
        continuation: bool,
        plan_only: bool,
        patch_only: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Phase A planning, optional assets, first-build message assembly."""
        if not continuation:
            self._goal = goal
            self._continuation_feedback = ""
            # Phase 2: seed the <todos> ledger from the goal's clauses so a
            # multi-part edit / listed ask has a per-turn checklist from turn 1
            # (reuses the existing todos injection + CURRENT TASK machinery).
            self._seed_task_ledger_from_goal()
            # Phase 0.10 — when the goal explicitly asks for multi-frame
            # rosters, raise this session's asset cap so the architect
            # can fit base + variant frames in one turn. Default cap is
            # 24 (`_MAX_ASSETS_PER_TURN`); the raised cap allows ~12
            # entities × 3 frames = 36, with headroom up to 72 for
            # larger rosters. Genre-free intent detection in
            # `prompts_v1._detect_multi_frame_intent`.
            try:
                from prompts_v1 import _detect_multi_frame_intent as _mfi
                mf_kws = _mfi(goal)
                if mf_kws:
                    self._session_asset_cap = 72
                    self._trace({
                        "kind": "multi_frame_intent_detected",
                        "matched_keywords": mf_kws,
                        "asset_cap_raised_to": 72,
                    })
                else:
                    self._session_asset_cap = None
            except Exception as e:
                # Detector is advisory; never block session start on it.
                self._session_asset_cap = None
                self._trace({"kind": "multi_frame_intent_error", "err": str(e)})
        # Persist for `turn_contract` trace event; reset per call so a
        # fresh session after an extension does not inherit True.
        self._continuation = bool(continuation)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        if continuation:
            # Make sure we have something to extend.
            if not self._current_file:
                try:
                    self._current_file = self.out_path.read_text(encoding="utf-8")
                except Exception as e:
                    yield self._record(AgentEvent(
                        "error", f"can't extend: no current file ({e})"
                    ))
                    return
            # Phase 1b: sanitize a corrupt on-disk baseline before extension.
            # Trace 1 (chess 20260522_000304) ended a session with a file
            # whose bytes started with diagnose preamble before <!DOCTYPE,
            # making every patch attempt useless. If we can recover real
            # HTML from the corrupt blob, rewrite the file in place; if not,
            # leave the bytes alone but tell the model the baseline is broken
            # and a clean <html_file> rewrite is permitted this turn.
            self._continuation_baseline_corrupt = False
            try:
                broken_reason = _baseline_structurally_broken(self._current_file)
            except Exception:
                broken_reason = None
            if broken_reason:
                normalized = _normalize_extracted_html(self._current_file)
                if normalized and not _baseline_structurally_broken(normalized):
                    try:
                        self.out_path.write_text(normalized, encoding="utf-8")
                        self._current_file = normalized
                        self._save_snapshot(normalized)
                        yield self._record(AgentEvent(
                            "info",
                            "[continuation] on-disk baseline had leading garbage; "
                            "sanitized to start at <!DOCTYPE> before resuming.",
                        ))
                        self._trace({
                            "kind": "continuation_baseline_sanitized",
                            "reason": broken_reason[:180],
                            "size_after": len(normalized),
                        })
                    except Exception as e:
                        self._trace({
                            "kind": "continuation_baseline_sanitize_failed",
                            "err": str(e)[:180],
                        })
                        self._continuation_baseline_corrupt = True
                else:
                    # Could not recover by normalization (truly broken/empty).
                    # Arm the rewrite-exemption flag so the next turn can emit
                    # a fresh <html_file> without hitting "baseline exists".
                    self._continuation_baseline_corrupt = True
                    self._allow_one_rewrite = True
                    yield self._record(AgentEvent(
                        "info",
                        f"[continuation] on-disk baseline is unrecoverable "
                        f"({broken_reason[:120]}); rewrite permitted this turn.",
                    ))
                    self._trace({
                        "kind": "continuation_baseline_unrecoverable",
                        "reason": broken_reason[:180],
                    })
            # Fresh-context continuation (frontier-agent pattern,
            # 2026-06-12). Capture whether the prior session ended clean
            # BEFORE the line below forces _previous_report_ok=True. When
            # it did, the accumulated history is debugging residue the new
            # request doesn't need — trace 20260612_004616 carried ~61K
            # prompt tokens into its continuation turns despite 7
            # compactions. SOTA models shrug that off; 27B-class local
            # models degrade well before it. The reset below replaces the
            # history with [system, state anchor] before the continuation
            # message is appended. Only changes what we SEND — never cuts
            # model output.
            _prior_clean = bool(self._previous_report_ok) and bool(
                (self._previous_report or {}).get("ok")
            )
            # Reset transient state so the loop runs again fresh.
            self._user_force_done = False
            if self._stop_event is not None:
                self._stop_event.clear()
            self._fix_mode = True
            self._previous_report_ok = True
            # Queue the new request as user feedback so _flush_user_injections
            # folds it into the next prompt with the standard "USER FEEDBACK"
            # banner the model already knows how to react to.
            self.add_user_feedback(goal)
            self._continuation_feedback = goal
            self._trace({"kind": "continuation_start", "feedback": goal})
            # Apply the context reset (see _maybe_reset_continuation_context
            # for the gates and the fallback behavior).
            self._maybe_reset_continuation_context(_prior_clean)
            yield self._record(AgentEvent(
                "info", f"continuing on existing file with new request: {goal[:160]}"
            ))
            # Inline the actual file as CURRENT FILE ON DISK truth source.
            # Without this, patch-search text routinely hallucinated
            # variable names against a file the model couldn't see (e.g.
            # donkey-kong trace 20260512_201139: model wrote a patch
            # targeting `state.princessTimer` when the file actually has
            # `state.princess.timer`). Worst on restarts where the
            # message history starts empty. Mirrors fix_instruction's
            # behavior on regular fix turns.
            if hasattr(self._p, "continuation_instruction"):
                cont_msg = self._p.continuation_instruction(self._current_file)
            else:
                # v0 prompt module — no helper; inline the same shape.
                cont_msg = (
                    "CONTINUATION TURN: the user has new feedback above. "
                    "Patch against the CURRENT FILE ON DISK block below "
                    "character-for-character.\n\n"
                    "CURRENT FILE ON DISK:\n"
                    "```html\n"
                    f"{self._current_file}\n"
                    "```\n\n"
                    "Reply with one or more <patch> blocks. Use a full "
                    "<html_file> only if patches truly cannot express "
                    "the change."
                )
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(cont_msg),
            })
        else:
            # eval --patch-only: seed edit with no Phase A / asset gen.
            _patch_only_seed = bool(patch_only) and self.seed_file is not None
            if patch_only and self.seed_file is None:
                yield self._record(AgentEvent(
                    "error", "patch_only requires seed_file",
                ))
                return

            # Wikipedia research lookup REMOVED (2026-06-24): empirical 0/10
            # hit rate on common game goals made it pure plan-time latency.
            # Grounding now comes from the curated opening library in memory/
            # (outlines / components / playtests / audits), not a network
            # fetch. reference_block stays empty so plan_instruction renders
            # no <reference> block (it no-ops on an empty string).
            reference_block = ""

            # P1 (MK trace 20260528): rehydrate seed media BEFORE the
            # planning prompt is built so the planner can see existing
            # asset/sound names and suppress the "MUST emit <assets>"
            # directive. Without this, the model emitted a fresh asset
            # roster every seed restart and the harness regenerated
            # every sprite — wiping the user's existing art.
            early_seed_assets = 0
            early_seed_sounds = 0
            if self.seed_file is not None:
                early_seed_assets, early_seed_sounds = (
                    self._early_rehydrate_seed_media()
                )

            if hasattr(self._p, "plan_instruction"):
                # v1+ planner takes goal so it can detect art-modality
                # keywords ("sprite", "art", "graphics") and escalate
                # the <assets> directive to REQUIRED for that turn.
                # Tolerant of older signatures: try with goal first,
                # then with force_minimal_first_build (if the restart
                # loop set it after a same-signature attempt), then
                # fall back to the simplest reference-only call.
                fmfb = bool(getattr(self, "_force_minimal_first_build", False))
                # P1: seed continuation — pass discovered names so
                # plan_instruction can suppress the MUST-emit <assets>
                # directive and tell the model to reuse existing media.
                from_seed_kwargs = {}
                if self.seed_file is not None and (
                    early_seed_assets or early_seed_sounds
                ):
                    from_seed_kwargs = {
                        "from_seed": True,
                        "seed_asset_names": sorted(self._session_assets.keys()),
                        "seed_sound_names": sorted(self._session_sounds.keys()),
                    }
                try:
                    _plan_nudge_ids: list[str] = []
                    plan_msg = self._p.plan_instruction(
                        reference_block=reference_block,
                        goal=goal,
                        force_minimal_first_build=fmfb,
                        model_class=self._system_prompt_class(),
                        nudge_ids_out=_plan_nudge_ids,
                        **from_seed_kwargs,
                    )
                    self._last_plan_nudge_ids = _plan_nudge_ids
                    if fmfb:
                        self._trace({
                            "kind": "force_minimal_first_build_applied",
                            "attempt_idx": getattr(
                                self, "_restart_attempt_idx", 0,
                            ),
                            "prev_signature": getattr(
                                self, "_prev_attempt_signature", None,
                            ),
                        })
                except TypeError:
                    try:
                        plan_msg = self._p.plan_instruction(
                            reference_block=reference_block, goal=goal,
                        )
                    except TypeError:
                        plan_msg = self._p.plan_instruction(
                            reference_block=reference_block,
                        )
            elif reference_block:
                # v0 prompt module — no plan_instruction() helper. Manually
                # prepend the reference and an authority sentence.
                plan_msg = (
                    f"{reference_block}\n\n"
                    "AUTHORITY: the <reference> block above is from Wikipedia "
                    "and describes the actual game the user named. Treat its "
                    "mechanics as authoritative. Your plan and criteria MUST "
                    "match those mechanics — do not invent different ones.\n\n"
                    f"{self._p.PLAN_INSTRUCTION}"
                )
            else:
                plan_msg = self._p.PLAN_INSTRUCTION

            # B1: the plan turn is where the model writes its <criteria>/<probes>;
            # at 1200 chars the deep outline's state/order/traps contract was
            # truncated, so probes were authored against fields the game would
            # not have (the dominant impossible-probe failure). 3000 gives the
            # full contract here (first-build still gets 3600) so the self-tests
            # read real state.
            plan_opening_block, plan_opening_hits = self._retrieve_opening_book_block(
                goal, stage="plan", char_budget=3000, deep=True,
            )
            if plan_opening_block:
                plan_msg = (
                    f"{plan_opening_block}\n\n"
                    "Use the opening-book recipes above when choosing your "
                    "plan, acceptance criteria, and executable probes — adapt "
                    "them to the user's goal when it specifies different "
                    "counts, style, or mechanics. Include the relevant state "
                    "and puzzle/help checks in the plan contract when they "
                    "apply.\n\n"
                    + plan_msg
                )
            if plan_opening_hits:
                self._trace({
                    "kind": "plan_opening_book_injected",
                    "hits": [
                        {key: value for key, value in hit.items() if key != "recipe"}
                        for hit in plan_opening_hits
                    ],
                    "selected_chars": sum(
                        len(json.dumps(
                            hit.get("recipe") or {},
                            ensure_ascii=False,
                            default=str,
                            separators=(",", ":"),
                        ))
                        for hit in plan_opening_hits
                    ),
                    "rendered_chars": len(plan_opening_block or ""),
                })

            # B2: a THIN playbook at plan time so the model sees the top
            # loop/input/facing rules BEFORE it commits to an approach instead
            # of discovering the rule three iterations later. Uses the narrow
            # code-stage retrieval (top few, full bodies, small budget) so the
            # smallest prompt of the session stays small.
            plan_playbook_block = self._retrieve_playbook_block(goal, stage="code")
            if plan_playbook_block:
                plan_msg = f"{plan_playbook_block}\n\n" + plan_msg
                self._trace({
                    "kind": "plan_playbook_injected",
                    "chars": len(plan_playbook_block),
                })

            # Stop-Losing-To-OneShot todo #6 — when the active prompt
            # module exposes build_system_prompt (v1+), pass model_class
            # so mid-tier models get a trimmed prompt. v0 falls back
            # to the static SYSTEM_PROMPT constant unchanged.
            lean_active = self._lean_prompt_active()
            sys_class = self._system_prompt_class()
            if hasattr(self._p, "build_system_prompt"):
                sys_prompt = self._p.build_system_prompt(
                    goal, model_class=sys_class,
                )
            else:
                sys_prompt = self._p.SYSTEM_PROMPT.replace("{goal}", goal)
            self._trace({
                "kind": "system_prompt_built",
                "model_class": self._model_class,
                "system_prompt_class": sys_class,
                "lean": lean_active,
                "chars": len(sys_prompt),
                # _trace converts this to a compact hash/size manifest without
                # changing the runtime prompt sent to the model.
                "system_prompt": sys_prompt,
            })
            # Runtime-mode observability (2026-06-13): one concise row so a
            # trace shows whether the user's /allroles + /playtest stack is
            # actually doing what they think — in particular whether the
            # per-iter VISUAL CRITIC will run (it needs a VLM-capable model;
            # on a text-only model /allroles only relabels planning).
            try:
                from backend import classify_model_modality as _cmm
                _model_is_vlm = _cmm(self._backend.info.model) == "vlm"
            except Exception:
                _model_is_vlm = False
            self._trace({
                "kind": "runtime_modes",
                "architect_split": bool(getattr(self, "_use_architect_split", False)),
                "vlm_critique": bool(getattr(self, "_use_vlm_critique", False)),
                "model_is_vlm": _model_is_vlm,
                "visual_critic_will_run": bool(
                    getattr(self, "_use_vlm_critique", False) and _model_is_vlm
                ),
                "lean_prompt": lean_active,
                "autonomous_feedback": bool(getattr(self, "_use_autonomous_feedback", False)),
                "step_mode": bool(getattr(self, "_step_mode", False)),
            })
            # Legacy maintainer docs (AGENTS.md / CLAUDE.md) must not be
            # injected into the game model — trace if still on disk.
            _dep_cfg = _deprecated_project_config_on_disk(Path.cwd())
            if not _dep_cfg:
                _dep_cfg = _deprecated_project_config_on_disk(self.out_path.parent.parent)
            if _dep_cfg:
                self._trace({
                    "kind": "project_config_deprecated_source",
                    "sources": _dep_cfg,
                    "hint": (
                        "maintainer docs belong in AGENTS.md + DEV.md (Cursor only); "
                        "game model uses prompts_v1 + memory/playbook.jsonl"
                    ),
                })

            # #5: surface PROVEN pose prompts from past animated sessions.
            # When this goal implies character motion and we hold recipes whose
            # pose words overlap it, inject them as a planning hint so the model
            # reuses txt2img-merged prompts that previously produced a REAL
            # (moved) pose rather than an idle clone. Best-effort; never blocks.
            try:
                if self._animation_expected():
                    from asset_library import AssetLibrary
                    _recipes = AssetLibrary().retrieve_pose_recipes(goal, k=4)
                    if _recipes:
                        _lines = "\n".join(
                            f'- {r.get("pose", "?")}: "{r.get("prompt", "")}"'
                            for r in _recipes
                        )
                        plan_msg = (
                            plan_msg
                            + "\n\n<proven-pose-prompts>\n"
                            + "These pose prompts produced REAL distinct poses "
                            + "(not idle clones) in past animated games. Reuse "
                            + "or adapt them for matching from_image frames:\n"
                            + _lines
                            + "\n</proven-pose-prompts>"
                        )
                        self._trace({
                            "kind": "pose_recipes_injected",
                            "count": len(_recipes),
                            "poses": [r.get("pose") for r in _recipes],
                        })
            except Exception as e:
                self._trace({"kind": "pose_recipes_error", "err": str(e)})

            self._messages = [
                {"role": "system", "content": sys_prompt},
            ] + (
                [] if _patch_only_seed else [
                    {"role": "user", "content": plan_msg},
                ]
            )

            self._trace({
                "kind": "session_start",
                "session_id": self._session_id,
                "artifact_id": self._artifact_id,
                "model": self.model,
                "goal": goal,
                "out_path": str(self.out_path),
                "trace_path": str(self.trace_path),
                "conversation_path": str(self.conversation_path),
                "snapshots_dir": str(self.snapshots_dir),
                "best_path": str(self.best_path),
                "max_iters": self.max_iters,
                "best_of_n": self.best_of_n,
                "num_ctx": self.num_ctx,
                "stall_seconds": self.stall_seconds,
                "memory_root": str(self._memory.root),
                "reference_chars": len(reference_block),
            })
            yield self._record(AgentEvent(
                "info",
                f"Trace: {self.trace_path}  (snapshots: {self.snapshots_dir})",
            ))

            if _patch_only_seed:
                self._trace({
                    "kind": "patch_only_mode",
                    "goal": goal,
                    "seed_file": str(self.seed_file),
                })
                yield self._record(AgentEvent(
                    "info",
                    "patch-only mode: skipping Phase A planning and asset "
                    "generation — jumping to seed edit.",
                ))

            if not _patch_only_seed:
                # ---- PHASE A: planning ------------------------------------------
                yield self._record(AgentEvent("phase", "planning"))
                self._plan_retry_done = False
                self._probe_quality_retry_done = False
                self._plan_syntax_retry_done = False
                # Phase 1B — when the diffuser pipeline lives on a GPU that
                # is NOT shared with any LLM slot, pre-warm it during
                # architect streaming so the cold-load (~30-60 s) is hidden
                # under the plan stream. The 2026-05-22 chess traces showed
                # ~37 s wasted between phase-A end and first asset gen.
                # SKIPPED on single-GPU systems where pre-warm would
                # compete with the architect for VRAM and slow it down.
                # General behavior — observable detection, no genre logic.
                self._maybe_prewarm_diffusers_during_phase_a()
                planning_role = self._planning_role()
                yield self._record(AgentEvent(
                    "activity", "streaming",
                    {"label": "planning reply", "role": planning_role},
                ))
                try:
                    plan_reply = await self._stream(self._token_cb_wrapper, role=planning_role)
                except Exception as e:
                    yield self._record(AgentEvent("activity", "idle"))
                    err_msg = (
                        f"{self._backend.info.name.upper()} call failed "
                        f"during planning: {e}"
                    )
                    yield self._record(AgentEvent("error", err_msg))
                    stall = self._classify_stall(str(e))
                    if stall:
                        yield self._record(AgentEvent(
                            "mlx_stall", err_msg,
                            {**stall, "phase": "planning"},
                        ))
                    return
                for _ev in self._drain_stream_ui_events():
                    yield _ev
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({
                    "role": "assistant",
                    "content": plan_reply,
                    "model_role": planning_role,
                    "model_name": self.get_backend(planning_role).info.model,
                    # Tag so lean mode can elide the verbose <plan> prose from
                    # this turn after the first build (criteria/probes already
                    # extracted into self._criteria / self._probes), stopping it
                    # from re-loading on every later coder prefill.
                    "phase": "planning",
                })
                self._extract_and_queue_lookups(plan_reply)
                self._capture_todos(plan_reply)
                self._dump_conversation()
                yield self._record(AgentEvent("plan", plan_reply))
                crit = self._extract_criteria(plan_reply)
                if crit:
                    self._criteria = crit
                    self._trace({"kind": "criteria", "text": crit[:600]})
                probes = self._extract_probes(plan_reply)
                # Planning-turn retry on an UNUSABLE plan. A degenerate token-
                # repetition tail (e.g. "cooldown reset cooldown reset ..." ×100,
                # or "st.x = st.x + st.x;" ×100) makes the model emit EOS mid-
                # structured-output — below the RepetitionDetector threshold — so
                # the plan parses with no <criteria> and/or no <probes>. Proceeding
                # would burn the whole session on an empty plan. Re-stream ONCE with
                # a terse corrective reprompt (mirrors the visual-critic reparse
                # retry). Trace-evidenced 2026-05-31: street-fighter + bomberman
                # both hit this and both recovered on a fresh attempt. This RECOVERS
                # empty output — it only fires when the plan parsed empty, so it
                # never truncates a legitimate long plan (per the no-early-cutoff rule).
                if (not self._criteria or not probes) and not getattr(
                    self, "_plan_retry_done", False
                ):
                    self._plan_retry_done = True
                    self._trace({
                        "kind": "plan_incomplete_retry",
                        "had_criteria": bool(self._criteria),
                        "had_probes": bool(probes),
                        "reply_chars": len(plan_reply or ""),
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "planning reply was incomplete (likely a repetition "
                        "cut-off) — retrying the plan once.",
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your previous reply was cut off before a complete "
                            "plan (missing <criteria> and/or <probes>). Re-emit the "
                            "FULL <plan>, <criteria>, <probes>"
                            + (", <assets>" if self._animation_expected() else "")
                            + " now, concisely. Do NOT repeat any word or phrase "
                            "more than a few times — write each item exactly once."
                        ),
                    })
                    _retry_role = self._planning_role()
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": "re-streaming incomplete plan",
                         "role": _retry_role},
                    ))
                    retry_reply = None
                    try:
                        retry_reply = await self._stream(
                            self._token_cb_wrapper, role=_retry_role)
                    except Exception:
                        retry_reply = None
                    for _ev in self._drain_stream_ui_events():
                        yield _ev
                    yield self._record(AgentEvent("activity", "idle"))
                    _nc = self._extract_criteria(retry_reply) if retry_reply else None
                    _np = self._extract_probes(retry_reply) if retry_reply else None
                    if retry_reply and (_nc or _np):
                        # Replace the failed planning blob instead of stacking
                        # a second 50k-115k char essay in context (DK trace).
                        replaced = False
                        for msg in reversed(self._messages):
                            if (
                                msg.get("role") == "assistant"
                                and msg.get("phase") == "planning"
                            ):
                                msg["content"] = retry_reply
                                msg["model_role"] = _retry_role
                                msg["model_name"] = (
                                    self.get_backend(_retry_role).info.model
                                )
                                replaced = True
                                break
                        if not replaced:
                            self._messages.append({
                                "role": "assistant", "content": retry_reply,
                                "model_role": _retry_role,
                                "model_name": self.get_backend(_retry_role).info.model,
                                "phase": "planning",
                            })
                        self._extract_and_queue_lookups(retry_reply)
                        self._capture_todos(retry_reply)
                        self._dump_conversation()
                        yield self._record(AgentEvent("plan", retry_reply))
                        plan_reply = retry_reply
                        if _nc:
                            self._criteria = _nc
                            self._trace({"kind": "criteria",
                                         "text": _nc[:600], "retry": True})
                        if _np:
                            probes = _np
                        self._trace({"kind": "plan_retry_recovered",
                                     "criteria": bool(_nc), "probes": len(_np or [])})

                # Probe-quality retry: the plan parsed fine (criteria + probes
                # present) but EVERY probe is structural-only — no input→delta,
                # no time-based check, no canvas-pixel read. A game that renders
                # a static HUD passes them all, so the harness `ok=True` signal
                # would be wrong (DEV.md "harness signal must be right"). The
                # passive nudge alone (injected into the first-build message)
                # did NOT move the model: the 2026-05-31 prompt-library sweep
                # showed 24/26 prompts still emitting 0 dynamic probes
                # (probe_quality_ratio 0.0). So ESCALATE to a corrective
                # re-prompt here, mirroring the empty-plan retry above. Bounded
                # to once; fires only when probes EXIST and ALL are structural,
                # so a plan that already has one dynamic probe is never
                # re-streamed (no long-plan regression). The retry is adopted
                # ONLY if it actually adds a dynamic probe — otherwise the
                # original probes are kept and the passive nudge below still
                # fires as a backstop.
                if (
                    probes
                    and not getattr(self, "_probe_quality_retry_done", False)
                    and self._classify_probes_dynamic(probes)["ratio"] == 0.0
                ):
                    self._probe_quality_retry_done = True
                    self._trace({
                        "kind": "probe_quality_retry",
                        "probe_count": len(probes),
                        "names": [p.get("name") for p in probes],
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "every Phase-A probe is structural-only (a game that "
                        "renders but never PLAYS would pass them all) — "
                        "re-prompting the plan once for an input→delta probe.",
                    ))
                    from prompts_v1 import input_moves_player_probe_expr
                    _dyn_probe = input_moves_player_probe_expr(
                        goal=self._goal or "",
                        code=getattr(self, "_current_file", "") or "",
                    )
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your <probes> only check structural PRESENCE "
                            "(elements exist, state is an object, arrays are "
                            "non-empty). A game that draws a static screen but "
                            "never responds to input would pass every one of "
                            "them — so they cannot verify the game actually "
                            "PLAYS. Re-emit the FULL <probes>...</probes> block "
                            "now, keeping your structural probes AND adding at "
                            "least one DYNAMIC probe that simulates an input "
                            "and asserts a state delta. Copy this shape, "
                            "swapping in your real control key and state path:\n"
                            f'  {{"name":"input_moves_player","expr":"{_dyn_probe}"}}\n'
                            "The exposed state global it reads (window.state, "
                            "or whatever you expose) MUST exist for the probe "
                            "to work. Do NOT repeat any word or phrase more "
                            "than a few times."
                        ),
                    })
                    _pq_role = self._planning_role()
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": "re-streaming plan for dynamic probes",
                         "role": _pq_role},
                    ))
                    pq_reply = None
                    try:
                        pq_reply = await self._stream(
                            self._token_cb_wrapper, role=_pq_role)
                    except Exception:
                        pq_reply = None
                    for _ev in self._drain_stream_ui_events():
                        yield _ev
                    yield self._record(AgentEvent("activity", "idle"))
                    _np = self._extract_probes(pq_reply) if pq_reply else None
                    _nc = self._extract_criteria(pq_reply) if pq_reply else None
                    # Adopt ONLY if the retry produced probes that now include a
                    # dynamic check — never regress to a worse/empty/still-
                    # structural set.
                    if _np and self._classify_probes_dynamic(_np)["ratio"] > 0.0:
                        self._messages.append({
                            "role": "assistant", "content": pq_reply,
                            "model_role": _pq_role,
                            "model_name": self.get_backend(_pq_role).info.model,
                        })
                        self._extract_and_queue_lookups(pq_reply)
                        self._capture_todos(pq_reply)
                        self._dump_conversation()
                        yield self._record(AgentEvent("plan", pq_reply))
                        plan_reply = pq_reply
                        probes = _np
                        if _nc:
                            self._criteria = _nc
                            self._trace({"kind": "criteria", "text": _nc[:600],
                                         "retry": "probe_quality"})
                        self._trace({
                            "kind": "probe_quality_retry_recovered",
                            "ratio": self._classify_probes_dynamic(probes)["ratio"],
                            "probes": len(probes),
                        })
                    else:
                        self._trace({
                            "kind": "probe_quality_retry_no_improvement",
                            "got_probes": len(_np or []),
                        })
                        # When the corrective re-stream still leaves every probe
                        # structural-only, inject one harness-authored dynamic
                        # probe so static-HUD plans cannot ship with ratio 0.0.
                        _inject_name = "input_moves_player"
                        _existing_names = {
                            str(p.get("name") or "") for p in (probes or [])
                        }
                        if _inject_name not in _existing_names:
                            from prompts_v1 import input_moves_player_probe_expr
                            _dyn_expr = input_moves_player_probe_expr(
                                goal=self._goal or "",
                                code=getattr(self, "_current_file", "") or "",
                            )
                            probes = list(probes or [])
                            probes.append({
                                "name": _inject_name,
                                "expr": _dyn_expr,
                                "harness_injected": True,
                            })
                            self._trace({
                                "kind": "probe_quality_harness_inject",
                                "name": _inject_name,
                                "ratio_after": (
                                    self._classify_probes_dynamic(probes)["ratio"]
                                ),
                            })

                if probes:
                    self._probes = probes
                    # 2.1: log full probe text alongside the count so brittle
                    # probes (instantaneous-state, getImageData-based,
                    # window-global-without-bind) are visible upfront and we
                    # don't have to dig through a 1000-char preview to find
                    # the expression that's blocking ship.
                    self._trace({
                        "kind": "probes_parsed",
                        "count": len(probes),
                        "names": [p.get("name") for p in probes],
                        "full": [
                            {
                                "name": str(p.get("name", "?"))[:60],
                                "expr": str(p.get("expr", ""))[:300],
                            }
                            for p in probes
                        ],
                    })
                    # Planning-turn coverage check: surface criteria that no
                    # probe references so the model can see the gap on iter 1
                    # rather than only at the end. Local LLMs often skip
                    # writing a probe for a stress/behavioral criterion;
                    # naming the gap in the first build prompt helps them
                    # close it. We don't block the loop on this — the model
                    # may not recover gracefully — just inject a nudge.
                    if self._criteria:
                        from tools import _criteria_coverage_gaps as _gaps_fn
                        gaps = _gaps_fn(self._criteria, probes)
                        if gaps:
                            self._planning_coverage_gaps = gaps[:6]
                            self._trace({
                                "kind": "planning_coverage_gaps",
                                "uncovered": self._planning_coverage_gaps,
                            })
                            yield self._record(AgentEvent(
                                "info",
                                "criteria without matching probes: "
                                + "; ".join(self._planning_coverage_gaps),
                            ))

                    # Probe-quality gate (DK trace 20260513_185815: all five
                    # probes — state_exists, player_exists, barrels_array,
                    # princess_exists, hud_visible — only checked structural
                    # presence; a game that rendered a static HUD would
                    # pass them all). Classify with the regex heuristic; if
                    # zero probes verify dynamic behavior, stash a nudge
                    # that gets injected into the first-build user message.
                    pq = self._classify_probes_dynamic(probes)
                    self._probe_quality = pq
                    self._trace({
                        "kind": "probe_quality",
                        "dynamic": pq["dynamic"],
                        "structural": pq["structural"],
                        "ratio": pq["ratio"],
                    })
                    if pq["ratio"] == 0.0 and probes:
                        yield self._record(AgentEvent(
                            "info",
                            "all probes are structural-only (e.g. "
                            "`!!window.x`) — nudging model to add "
                            "dynamic-behavior probes.",
                        ))
                    # Static probe-sanity lint: catches tautological probes
                    # that bind a temp and discard it before returning a
                    # constant. The undefined-property check runs later (it
                    # needs the on-disk HTML). Donkey-kong trace
                    # 20260516_142445 `mario_moves` is the canonical case.
                    taut_findings = GameAgent._lint_probes(probes)
                    self._probe_lint_findings = list(taut_findings)
                    if taut_findings:
                        self._trace({
                            "kind": "probe_lint",
                            "findings": taut_findings,
                        })
                        for f in taut_findings:
                            yield self._record(AgentEvent(
                                "info",
                                f"[yellow]probe lint[/yellow] {f['message']}",
                            ))
                    # Syntax lint: malformed probe expr burns iter 1 at test
                    # time (OutRun trace). Re-stream the plan once before build.
                    syntax_findings = GameAgent._lint_probe_syntax(probes)
                    if (
                        syntax_findings
                        and probes
                        and not getattr(self, "_plan_syntax_retry_done", False)
                    ):
                        self._plan_syntax_retry_done = True
                        bad = ", ".join(
                            f["name"] for f in syntax_findings[:4]
                        )
                        self._trace({
                            "kind": "plan_probe_syntax_retry",
                            "findings": syntax_findings,
                        })
                        yield self._record(AgentEvent(
                            "info",
                            f"Phase-A probe(s) have JS syntax errors "
                            f"({bad}) — re-prompting the plan once.",
                        ))
                        detail = "\n".join(
                            f"- {f['message']}" for f in syntax_findings[:4]
                        )
                        self._messages.append({
                            "role": "user",
                            "content": (
                                "Your <probes> block contains JavaScript "
                                "syntax errors — a malformed probe burns "
                                "the first build iteration. Fix and re-emit "
                                "the FULL <probes>...</probes> block now "
                                "(keep <plan> and <criteria> unchanged):\n"
                                f"{detail}\n"
                                "Common trap: missing `)` before `;` inside "
                                "async KeyboardEvent probes."
                            ),
                        })
                        _syn_role = self._planning_role()
                        yield self._record(AgentEvent(
                            "activity", "streaming",
                            {"label": "re-streaming plan for probe syntax",
                             "role": _syn_role},
                        ))
                        syn_reply = None
                        try:
                            syn_reply = await self._stream(
                                self._token_cb_wrapper, role=_syn_role,
                            )
                        except Exception:
                            syn_reply = None
                        for _ev in self._drain_stream_ui_events():
                            yield _ev
                        yield self._record(AgentEvent("activity", "idle"))
                        _np = (
                            self._extract_probes(syn_reply) if syn_reply else None
                        )
                        if _np:
                            _syn2 = GameAgent._lint_probe_syntax(_np)
                            if not _syn2:
                                self._messages.append({
                                    "role": "assistant",
                                    "content": syn_reply,
                                    "model_role": _syn_role,
                                    "model_name": (
                                        self.get_backend(_syn_role).info.model
                                    ),
                                    "phase": "planning",
                                })
                                self._extract_and_queue_lookups(syn_reply)
                                self._capture_todos(syn_reply)
                                self._dump_conversation()
                                yield self._record(AgentEvent("plan", syn_reply))
                                plan_reply = syn_reply
                                probes = _np
                                self._probes = _np
                                self._trace({
                                    "kind": "plan_probe_syntax_retry_recovered",
                                    "probes": len(_np),
                                })
                            else:
                                self._trace({
                                    "kind": "plan_probe_syntax_retry_no_fix",
                                    "still_bad": [f["name"] for f in _syn2],
                                })
                        else:
                            self._trace({
                                "kind": "plan_probe_syntax_retry_no_probes",
                            })
                # Visual-playtest auto-probes injection. Run AFTER the
                # model's own <probes> are parsed (or skipped) so the
                # injected probes ride alongside whatever the model wrote.
                # Deterministic safety net for the mechanism — even if the
                # model's probes miss the failure class (e.g. mortal-
                # kombat 2026-05-24 had no facing assertion), the injected
                # probe catches it. See VisualPlaytestRecipe.auto_probes.
                self._maybe_inject_visual_playtest_auto_probes()
                # Deterministic asset-miss probe: fails any iter where the
                # game draws a MISSING <key> placeholder for a generated asset
                # (key drift / loader not awaited). No VLM needed.
                self._maybe_inject_media_probes()

                q = self._extract_question(plan_reply)
                if q is not None:
                    yield self._record(AgentEvent("question", q))
                    while self._pending_answer is None:
                        await asyncio.sleep(0.1)
                    # Planning phase: the question came BEFORE any build, so
                    # asking for <plan> next is correct. The build-phase
                    # handler has its own rewrite-vs-patch routing below.
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "Your answer is recorded above. Now produce the "
                            "<plan> per the original instructions. Do NOT ask "
                            "another <question> this turn — the answer above "
                            "is sufficient to plan."
                        ),
                    })
                    planning_role = self._planning_role()
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": "streaming plan after question", "role": planning_role},
                    ))
                    try:
                        plan_reply = await self._stream(self._token_cb_wrapper, role=planning_role)
                    except Exception as e:
                        yield self._record(AgentEvent("activity", "idle"))
                        err_msg = (
                            f"{self._backend.info.name.upper()} call failed: {e}"
                        )
                        yield self._record(AgentEvent("error", err_msg))
                        stall = self._classify_stall(str(e))
                        if stall:
                            yield self._record(AgentEvent(
                                "mlx_stall", err_msg,
                                {**stall, "phase": "planning_after_question"},
                            ))
                        return
                    for _ev in self._drain_stream_ui_events():
                        yield _ev
                    yield self._record(AgentEvent("activity", "idle"))
                    self._messages.append({
                        "role": "assistant",
                        "content": plan_reply,
                        "model_role": planning_role,
                        "model_name": self.get_backend(planning_role).info.model
                    })
                    self._extract_and_queue_lookups(plan_reply)
                    self._capture_todos(plan_reply)
                    self._dump_conversation()
                    yield self._record(AgentEvent("plan", plan_reply))

                # Consolidated plan-outcome trace — one event a reviewer (or the
                # prompt-eval harness) can read to see WHAT planning produced
                # (criteria size, probe names, coverage gaps, probe quality)
                # without scanning the whole token stream. Added 2026-05-31 from
                # the prompt-library eval: a harness that stopped at the `plan`
                # event lost the criteria/probes traces that fire just after it.
                _prose_chars, _canonical_chars = self._p.measure_plan_reply(
                    plan_reply or ""
                )
                self._trace({
                    "kind": "plan_summary",
                    "criteria_chars": len(self._criteria or ""),
                    "probe_count": len(self._probes or []),
                    "probe_names": [p.get("name") for p in (self._probes or [])][:20],
                    "coverage_gaps": getattr(self, "_planning_coverage_gaps", []),
                    "probe_quality_ratio": (
                        getattr(self, "_probe_quality", {}) or {}
                    ).get("ratio"),
                    "prose_chars": _prose_chars,
                    "canonical_chars": _canonical_chars,
                    "nudge_ids": getattr(self, "_last_plan_nudge_ids", []),
                })
            else:
                plan_reply = ""
            if plan_only:
                # Dry-run: stop cleanly after Phase A. No coder-warm, no
                # asset/sound generation (no diffuser), no build loop.
                self._dump_conversation()
                yield self._record(AgentEvent(
                    "info",
                    "plan-only mode: Phase A complete — stopping before "
                    "asset generation and build.",
                ))
                return

            if not _patch_only_seed:
                # Phase 0.4 — pre-warm the coder slot's KV cache while asset
                # / sound generation runs on GPU 0. The architect just ran on
                # slot 3 (GPU 3); slot 1 (GPU 1, coder) has never seen this
                # conversation. Without this, iter 1's first request pays
                # a full prefill before any token streams; the 2026-05-22
                # chess trace had a 740 s prefill stall on iter 2 (cold
                # cross-slot cache + a fix-mode prompt that embedded the
                # full file). Fires-and-forgets — emits a `prefill_warm`
                # trace event on completion. Coder slot may differ from
                # architect slot only on the 4-GPU workstation; otherwise
                # `get_backend('coder')` returns the same backend as the
                # architect just used and warm_prefix is a no-op cache hit.
                coder_backend = self.get_backend("coder")
                architect_backend = self.get_backend("architect")
                if (
                    coder_backend is not None
                    and architect_backend is not None
                    and coder_backend is not architect_backend
                ):
                    snapshot_messages = list(self._messages)
                    async def _warm_coder_slot() -> None:
                        res = await coder_backend.warm_prefix(
                            snapshot_messages,
                            options={"num_ctx": self.num_ctx},
                            keep_alive=self._keep_alive_for_backend(coder_backend),
                            timeout_s=120.0,
                        )
                        self._trace({
                            "kind": "prefill_warm",
                            "slot": "coder",
                            "trigger": "phase_a_end",
                            **res,
                        })
                    # Background task — runs concurrent with asset/sound gen.
                    self._coder_warm_task = asyncio.create_task(_warm_coder_slot())

                # ---- Phase A → first-build: optional asset + sound generation --
                # Delegated to a shared helper so the same code path also
                # runs mid-session when the model emits a fresh <assets>/
                # <sounds> block in response to user feedback.
                async for ev in self._maybe_generate_assets_and_sounds(
                    plan_reply, trigger="phase_a",
                ):
                    yield ev

            # ---- seed file OR memory skeleton for the first build ----------
            if self.seed_file is not None:
                # User explicitly handed us a starting file. Skip memory
                # retrieval entirely; pre-populate the on-disk file and the
                # in-memory baseline so patches can apply against it from
                # the very first iteration. The plan is already done, so
                # the model immediately sees: goal + plan + existing code.
                try:
                    seed_html = self.seed_file.read_text(encoding="utf-8")
                except Exception as e:
                    yield self._record(AgentEvent(
                        "error",
                        f"could not read seed file {self.seed_file}: {e}",
                    ))
                    return
                # Land it on disk as the working file before iteration 1
                # so a patch-only first reply has something to patch.
                self.out_path.write_text(seed_html, encoding="utf-8")
                self._current_file = seed_html
                # Rehydrate media state from the seed before iteration 1.
                # chat.py resolves seeds (including snapshots and
                # .best.html siblings) back to games/<basename>.html
                # before constructing this agent, so out_path is the
                # canonical live game and out_path.parent is the games
                # dir — which is exactly where <basename>_assets/ lives.
                # Anchor the scan there (NOT at self.seed_file.parent,
                # which for a snapshot seed would be the snapshots
                # subdir and miss the assets folder one level up).
                seed_assets, seed_sounds, _, _ = (
                    _scan_seed_media(seed_html, self.out_path)
                )
                if seed_assets:
                    self._session_assets.update(seed_assets)
                if seed_sounds:
                    self._session_sounds.update(seed_sounds)
                if seed_assets or seed_sounds:
                    yield self._record(AgentEvent("memory", (
                        f"rehydrated from seed: "
                        f"{len(seed_assets)} asset(s), "
                        f"{len(seed_sounds)} sound(s)"
                    ), {
                        "seed_assets": {n: str(p) for n, p in seed_assets.items()},
                        "seed_sounds": {n: str(p) for n, p in seed_sounds.items()},
                    }))
                self._active_skeleton = f"seed: {self.seed_file.name}"
                yield self._record(AgentEvent("memory", (
                    f"using user-provided seed file: {self.seed_file} "
                    f"({len(seed_html)} bytes) — memory skeleton skipped"
                ), {
                    "seed_file": str(self.seed_file),
                    "seed_bytes": len(seed_html),
                }))
                # Patch-only skips Phase A where auto-probes normally inject
                # (~5090). Seed edits still match canvas-two-actors-facing etc.
                # via goal — inject deterministic facing probes before iter 1.
                self._maybe_inject_visual_playtest_auto_probes()
                # First build = plan-stage equivalent: model is choosing
                # the implementation shape, broad context helps.
                pb_block = self._retrieve_playbook_block(
                    goal, code=seed_html, stage="plan",
                    ensure_ids=self._first_build_playbook_ensure_ids(goal),
                )
                # Seed retrieval: bias the outline match with structural
                # tokens from the working file so a terse goal ("add a
                # button") doesn't out-rank what the file actually is.
                seed_struct_tokens = self._seed_structural_tokens(seed_html)
                if seed_struct_tokens:
                    self._trace({
                        "kind": "seed_structural_tokens",
                        "tokens": seed_struct_tokens,
                    })
                opening_block, opening_hits = self._retrieve_opening_book_block(
                    goal, stage="plan", extra_tokens=seed_struct_tokens,
                )
                self._active_opening_book_recipes = opening_hits
                # Component skill library — tested mechanics snippets, same
                # as the skeleton path. The seed path previously skipped
                # this, so a seed/continuation build lost the copy-paste-
                # correct snippets exactly when a weak model needs them.
                # Retrieved before assembly so the lean budget weighs all three.
                components_block = self._retrieve_components_block(
                    goal, stage="plan", k=3,
                )
                # Lean mode: cap the COMBINED size of the three memory blocks.
                # protect_components: on a seed continuation the components
                # are the snippets the weak model copies from — keep them
                # even if the opening book already filled the budget.
                opening_block, components_block, pb_block = (
                    self._apply_lean_memory_budget(
                        opening_block, components_block, pb_block,
                        protect_components=True,
                        protect_playbook=bool(self._session_assets),
                    )
                )
                pb_kwargs = {"playbook_block": pb_block} if pb_block else {}
                # Cap inlined seed size so a large working file doesn't
                # balloon iter-1 context into a repetition/deliberation loop
                # (2026-06-21 point-and-click seed trace). Full file stays on
                # disk; the prompt tells the model to patch against it.
                seed_html_for_prompt, seed_truncated = self._seed_html_for_prompt(
                    seed_html, self._last_test_report,
                )
                if seed_truncated:
                    self._trace({
                        "kind": "seed_html_excerpted",
                        "full_bytes": len(seed_html),
                        "prompt_bytes": len(seed_html_for_prompt),
                    })
                build_msg = self._p.seed_build_instruction(
                    seed_html_for_prompt, str(self.seed_file),
                    truncated=seed_truncated, **pb_kwargs,
                )
                if opening_block:
                    build_msg = (
                        f"{opening_block}\n\n"
                        "Use the opening-book recipes above as verified "
                        "implementation and test guidance.\n\n"
                        + build_msg
                    )
                if components_block:
                    build_msg = f"{components_block}\n\n" + build_msg
                asset_block = render_asset_paths_block(
                    self._session_assets, self.out_path,
                )
                sound_block = render_sound_paths_block(
                    self._session_sounds, self.out_path,
                    looping_names=self._session_looping,
                )
                video_block = render_video_paths_block(
                    self._session_videos, self.out_path,
                )
                seed_media_contract = self._render_seed_media_contract(
                    seed_html,
                    asset_names=list(self._session_assets.keys()),
                    sound_names=list(self._session_sounds.keys()),
                )
                prelude = "\n\n".join(
                    b for b in (
                        seed_media_contract, asset_block, sound_block,
                        video_block,
                        getattr(self, "_pointclick_grounding_block", "") or "",
                    ) if b
                )
                if prelude:
                    build_msg = prelude + "\n\n" + build_msg
                local_nudge = self._local_first_build_nudge()
                if local_nudge:
                    build_msg = local_nudge + "\n\n" + build_msg
                probe_nudge = self._probe_quality_nudge()
                if probe_nudge:
                    build_msg = probe_nudge + "\n\n" + build_msg
                # MK trace 20260517_220025: apply scope lock from the
                # initial goal so iter 1 of a seed edit doesn't sprawl.
                build_msg = self._apply_initial_goal_scoping(goal, build_msg)
                if self._session_assets:
                    build_msg = (
                        build_msg + "\n\n"
                        + self._p.generated_sprite_draw_contract()
                    )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(build_msg),
                })
            else:
                if self._skeleton_mode == "default":
                    # Force the bundled scaffold — used by tune mode so we
                    # measure the agent's reasoning, not skeleton-of-self
                    # retrieval. Bypasses memory entirely.
                    skel = SkeletonHit(
                        name=DEFAULT_SKELETON_NAME,
                        html=DEFAULT_SKELETON,
                        score=0.0,
                        source_goal=None,
                    )
                elif self._skeleton_mode == "default_v2":
                    # Bigger bug-hardened scaffold — pre-empts focus-loss,
                    # restart-cleanup, dt-cap, DPR-resize and ~7 other
                    # playbook bullets in the seed itself. The model only
                    # has to fill update/draw/state.
                    skel = SkeletonHit(
                        name=CANVAS_SKELETON_V2_NAME,
                        html=CANVAS_SKELETON_V2,
                        score=0.0,
                        source_goal=None,
                    )
                else:
                    skel = self._memory.retrieve_skeleton(goal)
                should_fallback, fallback_reason = (
                    self._local_should_fallback_skeleton(skel)
                )
                if should_fallback:
                    skel = SkeletonHit(
                        name=DEFAULT_SKELETON_NAME,
                        html=DEFAULT_SKELETON,
                        score=0.0,
                        source_goal=None,
                    )
                    yield self._record(AgentEvent(
                        "memory",
                        "local skeleton guard: fallback to default scaffold "
                        f"({fallback_reason})",
                        {
                            "fallback_reason": fallback_reason,
                            "skeleton": skel.name,
                            "backend": self._backend.info.name,
                        },
                    ))
                self._active_skeleton = skel.name
                memory_msg = (
                    f"using skeleton: {skel.name}"
                    + (f" (sim={skel.score:.2f}, src goal: {skel.source_goal!r})"
                       if skel.source_goal else " (default)")
                    + (f" [skeleton_mode={self._skeleton_mode}]" if self._skeleton_mode != "retrieve" else "")
                )
                yield self._record(AgentEvent("memory", memory_msg, {
                    "skeleton": skel.name, "score": skel.score,
                    "source_goal": skel.source_goal,
                    "skeleton_mode": self._skeleton_mode,
                }))

                # First build with memory skeleton — plan stage (broad).
                # First-build retrieval queries against goal ONLY.
                # The skeleton is canvas_basic.html (generic) for every
                # fresh build, so including it in the query drags in
                # the same ~10 bullets every time regardless of goal —
                # defeats goal-specific retrieval. User-supplied seed
                # files go through a different code path that does
                # include code.
                pb_block = self._retrieve_playbook_block(
                    goal, stage="plan",
                    ensure_ids=self._first_build_playbook_ensure_ids(goal),
                )
                opening_block, opening_hits = self._retrieve_opening_book_block(
                    goal, stage="plan",
                )
                self._active_opening_book_recipes = opening_hits
                # Open-domain detection (genre-free) — see
                # MemoryRetrievalMixin._detect_open_domain_build.
                open_domain_build, _outline_rows = self._detect_open_domain_build(
                    opening_hits
                )
                # Component skill library — tested mechanics snippets. Retrieved
                # HERE (before assembly) so the lean combined-memory budget can
                # weigh all three blocks together. Open-domain builds pull one
                # extra snippet (k=4) and PIN the engine skeleton (game loop +
                # buffered input) so a weak model copies a working loop instead
                # of inventing a broken RAF/input from scratch.
                _ensure_ids = (
                    ["fixed-timestep-game-loop", "input-manager-buffered"]
                    if open_domain_build else None
                )
                components_block = self._retrieve_components_block(
                    goal, stage="plan", k=4 if open_domain_build else 3,
                    ensure_ids=_ensure_ids,
                )
                # Lean mode: cap the COMBINED size of the three memory blocks
                # (opening > components > playbook) so a local model isn't
                # buried in overlapping past-lessons before the task. For
                # open-domain builds PROTECT the components so the working
                # game-loop+input snippets survive truncation (the playbook
                # yields first) instead of the model inventing a broken loop.
                opening_block, components_block, pb_block = (
                    self._apply_lean_memory_budget(
                        opening_block, components_block, pb_block,
                        protect_components=open_domain_build,
                        protect_playbook=bool(self._session_assets),
                    )
                )
                if open_domain_build:
                    self._trace({
                        "kind": "open_domain_components_boost",
                        "outline_id": (
                            _outline_rows[0].get("id") if _outline_rows else None
                        ),
                        "outline_score": (
                            round(_outline_rows[0].get("score", 0.0), 4)
                            if _outline_rows else None
                        ),
                        "floor": self._OPEN_DOMAIN_OUTLINE_FLOOR,
                        "components_k": 4,
                        "pinned": ["fixed-timestep-game-loop", "input-manager-buffered"],
                        "protect_components": True,
                    })
                pb_kwargs = {"playbook_block": pb_block} if pb_block else {}

                # Design happens in ONE pass: the planning turn (above, on
                # the architect role) already emits the full decomposition —
                # mechanics, controls, risky bits, criteria, probes, media,
                # and a `Build order` line. The separate `<architect>` prose
                # turn that used to run here was removed (2026-05-30): it
                # restated the plan (Risks duplicated, Data/Layers already in
                # the seed scaffold) and was a second boundary-free prose
                # generation that ran away on the local model. The architect
                # ROLE and the exit-decision turn are unchanged.

                # Fix B (model-agnostic): pass the current session's
                # asset/sound directory basenames so first_build_instruction
                # can scrub stale `./<other_session>_assets` paths out of
                # the retrieved seed skeleton. Without this, any model is
                # liable to copy the seed's session-specific path literals
                # verbatim (classic-doom 20260512_153449 caught this on
                # DeepSeek-V4). Names derived from the first asset path in
                # each map; if either map is empty we pass None and the
                # scrubber substitutes a self-describing sentinel.
                cur_asset_dir = None
                if self._session_assets:
                    any_asset = next(iter(self._session_assets.values()))
                    cur_asset_dir = any_asset.parent.name
                cur_sound_dir = None
                if self._session_sounds:
                    any_sound = next(iter(self._session_sounds.values()))
                    cur_sound_dir = any_sound.parent.name
                build_msg = self._p.first_build_instruction(
                    skel.html, skel.source_goal,
                    current_asset_dir=cur_asset_dir,
                    current_sound_dir=cur_sound_dir,
                    has_generated_assets=bool(self._session_assets),
                    **pb_kwargs,
                )
                if opening_block:
                    build_msg = (
                        f"{opening_block}\n\n"
                        "Use the opening-book recipes above as verified "
                        "implementation and test guidance.\n\n"
                        + build_msg
                    )
                # Component skill library (retrieved + budget-capped above).
                if components_block:
                    build_msg = f"{components_block}\n\n" + build_msg
                asset_block = render_asset_paths_block(
                    self._session_assets, self.out_path,
                )
                sound_block = render_sound_paths_block(
                    self._session_sounds, self.out_path,
                    looping_names=self._session_looping,
                )
                video_block = render_video_paths_block(
                    self._session_videos, self.out_path,
                )
                prelude = "\n\n".join(
                    b for b in (
                        asset_block, sound_block, video_block,
                        getattr(self, "_pointclick_grounding_block", "") or "",
                    ) if b
                )
                if prelude:
                    build_msg = prelude + "\n\n" + build_msg
                local_nudge = self._local_first_build_nudge()
                if local_nudge:
                    build_msg = local_nudge + "\n\n" + build_msg
                # Phase 4B (gated): one-objective-first-build on a slow local
                # backend with a strong outline match — prepended LAST so it
                # leads the first-build turn. No-op unless opted in.
                one_obj_nudge = self._one_objective_first_build_nudge()
                if one_obj_nudge:
                    build_msg = one_obj_nudge + "\n\n" + build_msg
                probe_nudge = self._probe_quality_nudge()
                if probe_nudge:
                    build_msg = probe_nudge + "\n\n" + build_msg
                # Mirror the seed-file branch: apply scope lock from
                # the initial goal so iter 1 of a strict goal honors it.
                build_msg = self._apply_initial_goal_scoping(goal, build_msg)
                # Run06: draw contract must be the last line the model sees
                # before writing iter 1 (after asset prelude + nudges).
                if self._session_assets:
                    build_msg = (
                        build_msg + "\n\n"
                        + self._p.generated_sprite_draw_contract()
                    )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(build_msg),
                })


    async def _run_build_iterate_loop(
        self,
        *,
        continuation: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Phase B iteration loop and cap-reached bonus turn."""
        # ---- PHASE B: build/iterate -------------------------------------
        self._awaiting_confirm = False

        # In continuation mode the iteration counter resumes from where the
        # previous run left off; the user-visible label uses a relative count
        # so they see "extension 1/N" instead of confusing "iteration 4/9".
        start_iter = self._snapshot_n + 1 if continuation else 1
        end_iter = start_iter + self.max_iters - 1
        # Stop-Losing-To-OneShot todo #4 — auto-revert can grant bonus
        # iters (each revert += 1, capped at max_iters/2) so the user's
        # budget isn't burned by reverted regressions. range() upper
        # bound is the hard cap; the in-loop early break gates each
        # bonus iter on a revert having actually happened.
        self._iter_budget_bonus = 0
        revert_bonus_cap = max(1, self.max_iters // 2)
        hard_max = end_iter + revert_bonus_cap

        for iteration in range(start_iter, hard_max + 1):
            if iteration > end_iter + self._iter_budget_bonus:
                break
            # Lean mode: after the first build, elide the verbose <plan>
            # prose from the retained Phase-A turn so it stops re-loading on
            # every coder prefill (criteria/probes are already tracked
            # separately). Idempotent; only fires once.
            if iteration > start_iter and self._lean_prompt_active():
                self._lean_compact_planning_message()
            # Pre-iter-1: lean verbose plan prose BEFORE the first build on
            # heavy local sessions so the model does not re-deliberate it.
            if (
                iteration == start_iter
                and self._should_pre_lean_plan_before_first_build()
            ):
                self._lean_compact_planning_message()
            # Polish phase (item 2): how many iters remain AFTER this one.
            # The polish branch in _build_fix_prompt only fires when >= 1
            # so a polish patch always gets a test pass before shipping.
            self._iters_remaining = (end_iter + self._iter_budget_bonus) - iteration
            # Dead-first-build attempt-abort. After 2 dead first builds
            # in this attempt, the detector flips this flag so the
            # iteration loop exits cleanly and the restart loop can
            # apply a fresh seed/recipe. Universal: keys on observable
            # raf_ran=false + input dead structural signal, no genre.
            if getattr(self, "_dead_first_build_abort_attempt", False):
                self._trace({
                    "kind": "dead_first_build_attempt_aborted",
                    "iteration": iteration,
                    "recoveries": self._dead_first_build_recoveries,
                    "hint": (
                        "Aborting this attempt after 2 dead first "
                        "builds so the restart loop can try a fresh "
                        "seed/scope."
                    ),
                })
                break
            # Reset per-iter flags so the media-only-bonus check sees
            # only THIS iter's regen state.
            self._media_regenerated_this_iter = False
            self._new_assets_not_in_html = set()
            # Harness-owned recovery: generate sprites dropped by the
            # per-turn cap on the prior turn (Dragon's Lair trace).
            async for ev in self._maybe_autogen_pending_dropped_assets():
                yield ev
            # Phase 4 (4D.1/4D.2): reset per-iter trace-correlation fields so
            # iter_summary reflects only THIS iter (no stale bleed from a prior
            # iter's patch count / coaching / router-state action).
            self._last_patch_applied = None
            self._last_patch_total = None
            self._last_coaching_action = "none"
            self._last_asset_reprompt_cleared = False
            # Decay stale diagnose context after clean passes so extension
            # prompts don't keep carrying an obsolete root-cause string.
            if self._consecutive_clean_iters >= 1 and self._last_diagnose:
                self._trace({
                    "kind": "last_diagnose_decayed",
                    "reason": "clean_streak",
                    "clean_iters": self._consecutive_clean_iters,
                })
                self._last_diagnose = None
            # User hard-stop: Ctrl-D in the TUI sets _user_force_done. Honor
            # it at the top of every iteration so the agent never starts a
            # new stream after the user asked to stop, even when probes
            # haven't passed. Whatever's in best.html (or out_path) ships.
            if self._user_force_done:
                # Phase 1A — cancel any in-flight background critic
                # before exiting. The critic generates coaching for an
                # iter that's not going to run; finishing it wastes
                # 20-300 s of user wait time.
                if self._critic_task is not None and not self._critic_task.done():
                    self._critic_task.cancel()
                    try:
                        await self._critic_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._critic_task = None
                yield self._record(AgentEvent(
                    "info", "user requested ship - exiting iteration loop",
                ))
                async for ev in self._final_iter_test_if_needed():
                    yield ev
                self._record_session_outcome(ok=self.best_path.exists())
                yield self._record(AgentEvent(
                    "done",
                    "User-requested ship.",
                    {"best_exists": self.best_path.exists()},
                ))
                return
            # Step-mode pause (Stop-Losing-To-OneShot todo #1): between
            # iterations, wait for explicit user go-ahead so the user can
            # verify the just-completed iter before the model runs again.
            # First iter is exempt (no prior result to inspect yet). The
            # next-user message is ALREADY appended at the end of the
            # previous iter; if the user types feedback during the pause
            # we pop that message and re-append with the feedback banner
            # prepended via _flush_user_injections so the model sees it
            # before the report-driven base prompt.
            if self._step_mode and iteration > start_iter and self._auto_retry_pending:
                # Phase 0D-4: a stalled feedback turn saved no code and armed an
                # auto-retry — skip the step pause and re-stream immediately so
                # the user's edit is not lost behind a "complete" prompt.
                self._auto_retry_pending = False
                self._trace({
                    "kind": "step_pause_skipped_auto_retry",
                    "iteration": iteration,
                })
            elif self._step_mode and iteration > start_iter:
                # Phase 0D-4: honest label — say "produced no code" when the
                # prior iter materialized nothing, instead of always "complete".
                _iter_status = (
                    "complete" if self._last_iter_materialized
                    else "produced no code"
                )
                yield self._record(AgentEvent(
                    "await_user",
                    f"step-mode: iter {iteration - 1} {_iter_status} — "
                    "Enter to continue, or type feedback",
                    {
                        "just_finished_iter": iteration - 1,
                        "materialized": self._last_iter_materialized,
                    },
                ))
                while self._step_pause_should_wait():
                    await asyncio.sleep(0.1)
                # Ship requested during the pause (Ctrl+D / 'done') — re-enter
                # the iter loop so the top-of-loop force_done check exits.
                if self._user_force_done:
                    continue
                # LLM router (chess-trace fix): route the feedback the user
                # typed during the pause BEFORE flushing it, so the deferral
                # / asset decisions inside _flush_user_injections see it.
                await self._precompute_feedback_route()
                if self.has_pending_user_input() and not self._feedback_deferred_last_turn:
                    self._step_pause_flush_feedback()
                self._step_continue = False  # consume signal

            self._last_iter_run = iteration
            if continuation:
                rel = iteration - start_iter + 1
                phase_text = f"extension {rel}/{self.max_iters}"
            else:
                phase_text = f"iteration {iteration}/{self.max_iters}"
            yield self._record(AgentEvent("phase", phase_text))

            # Prune older turns before generating, so we don't hit context.
            self._prune_messages()

            # Best-of-N fan-out fires whenever we have N>1 candidates
            # to sample. Originally gated on `_fix_mode` (iter 2+) under
            # the assumption that iter 1 had no test signal to score
            # against, but the scorer at _generate_and_score_candidates
            # IS the test harness itself (Chromium + probes + structural
            # checks) — every iter has that signal, including iter 1.
            # Removing the gate uses the multi-slot GPU capacity from
            # session start; on the 4-GPU box this means GPUs 2 and 3
            # are hot during the first build instead of sitting idle.
            use_bon = self.best_of_n > 1
            bon_n = self.best_of_n
            # Capability-round item 5: stuck best-of-2 repair. Being stuck
            # is precisely when sampling pays — a stream costs minutes, a
            # browser score costs seconds. One-turn escalation to n=2
            # (backend default temps 0.2 / 0.6), capped per session.
            if (
                self._stuck_bon_enabled
                and not use_bon
                and self._should_escalate_stuck_bon(
                    self._stuck_streak, self.best_of_n, self._stuck_bon_escalations,
                    last_report=self._last_test_report,
                )
            ):
                use_bon = True
                bon_n = 2
                self._stuck_bon_escalations += 1
                self._trace({
                    "kind": "stuck_bon_escalation",
                    "iteration": iteration,
                    "stuck_streak": self._stuck_streak,
                    "escalations_used": self._stuck_bon_escalations,
                    "cap": _STUCK_BON_ESCALATION_CAP,
                })
                yield self._record(AgentEvent(
                    "best_of_n",
                    f"stuck {self._stuck_streak} iters — escalating to "
                    f"best-of-2 for this turn "
                    f"({self._stuck_bon_escalations}/{_STUCK_BON_ESCALATION_CAP} "
                    "escalations used)",
                    {
                        "stuck_streak": self._stuck_streak,
                        "escalations_used": self._stuck_bon_escalations,
                    },
                ))
            fallback_attempted = False
            # Phase 5d: one transparent retry of the SAME backend on a
            # crashed cloud result before swapping to local Ollama.
            # Trace 2 (chess 20260522_104235) showed transient Anthropic
            # overloads usually clear in seconds; retrying once is much
            # cheaper than hot-swapping the whole backend.
            cloud_retry_attempted = False
            self._mlx_stall_retries_this_iter = 0
            while True:
                try:
                    if use_bon:
                        yield self._record(AgentEvent(
                            "best_of_n",
                            f"sampling {bon_n} candidates",
                            {"n": bon_n},
                        ))
                        winner, all_cands = await self._generate_and_score_candidates(bon_n)
                        # Replay the winner visually for the user — feel as if
                        # the model just wrote it now, even though it generated
                        # silently.
                        for piece in self._chunk_for_display(winner.text):
                            self._token_cb_wrapper(piece)
                        reply = winner.text
                        _cand_paths = [
                            (c.extra or {}).get("candidate_path")
                            for c in all_cands
                            if (c.extra or {}).get("candidate_path")
                        ]
                        _winner_path = (winner.extra or {}).get("candidate_path")
                        yield self._record(AgentEvent(
                            "best_of_n",
                            f"picked candidate score={winner.score:+.2f} from {len(all_cands)}"
                            + (f" · {_winner_path}" if _winner_path else ""),
                            {
                                "winner_score": winner.score,
                                "all_scores": [c.score for c in all_cands],
                                "winner_extra": winner.extra,
                                "winner_path": _winner_path,
                                "candidate_paths": _cand_paths,
                            },
                        ))
                    else:
                        # Prefill diagnose tag on fix turns so format compliance
                        # is forced. First-build (iter 1, fix_mode False) doesn't
                        # use diagnose, so prefill is empty there.
                        reply_prefill = ""
                        prefill_force = False
                        if self._use_prefill and self._fix_mode:
                            # Patch-first recovery (chess-trace iter 4/5 fix):
                            # after a prior format failure (prose essay /
                            # unclosed_patch that yielded no usable code), skip
                            # the <diagnose> essay and prefill straight into a
                            # <patch> so the model commits to an edit instead
                            # of rambling. Only when there is a file to patch
                            # AND the router did not ask for new art (which
                            # needs an <assets> block, not a patch).
                            _route = getattr(self, "_feedback_route", None)
                            _wants_assets = bool(
                                _route and _route.get("allow_assets_block")
                            )
                            # User-feedback fix turns: commit to <patch>
                            # immediately (Star Wars trace iter 2 burned 520s
                            # on a diagnose essay; iter 3 with patch prefill
                            # applied 6/6 patches in 286s).
                            _has_feedback = bool(
                                getattr(self, "_last_drained_feedback", None)
                            )
                            _art_pending = bool(
                                _wants_assets
                                or getattr(self, "_unhonored_asset_request", None)
                            )
                            _stall_light = getattr(
                                self, "_mlx_stall_retries_this_iter", 0
                            ) >= 1
                            if _stall_light:
                                # Post-stall retry: lighter prefill than
                                # patch_first — heavy prefill + VLM image
                                # just produced 0 tokens for 180s+.
                                reply_prefill = "<diagnose>"
                                self._trace({
                                    "kind": "mlx_stall_light_prefill",
                                    "iteration": iteration,
                                })
                            elif (
                                self._current_file
                                and not _wants_assets
                                and (
                                    self._format_stuck_streak >= 1
                                    or _has_feedback
                                )
                            ):
                                reply_prefill = "<patch>\n<<<<<<< SEARCH\n"
                                self._trace({
                                    "kind": "patch_first_prefill",
                                    "format_stuck_streak": self._format_stuck_streak,
                                    "had_feedback": _has_feedback,
                                    "iteration": iteration,
                                })
                            elif _art_pending and (
                                _has_feedback or self._last_stream_deliberated
                                or self._last_stream_looped
                            ):
                                # Phase 0E-2 (Fieldrunners trace 20260626_102307
                                # iters 4-5): when the user asked for NEW ART,
                                # seed the turn with <assets> so the model starts
                                # IN-FORMAT instead of rambling pre-tag prose
                                # (iter 4 deliberation) or bulk-emitting markdown
                                # patches (iter 5). Forcing the tag is the cheap,
                                # in-codebase version of constrained decoding —
                                # same trick the critic uses with "Q1: ".
                                reply_prefill = "<assets>"
                                prefill_force = True
                                self._trace({
                                    "kind": "assets_first_prefill",
                                    "had_feedback": _has_feedback,
                                    "deliberated": bool(self._last_stream_deliberated),
                                    "looped": bool(self._last_stream_looped),
                                    "iteration": iteration,
                                })
                            else:
                                # No trailing newline — Anthropic 400s on final
                                # assistant whitespace; backend also rstrip()s.
                                reply_prefill = "<diagnose>"
                        elif (not self._current_file) and (
                            self._force_first_build_prefill
                            or self._is_local_backend()
                        ):
                            # First-build rescue after a no-code turn, OR a
                            # PROACTIVE iter-1 prefill for local MLX/Ollama:
                            # GLM-5.2-MLX rambles 15-18k tokens of pre-<html_file>
                            # deliberation on rich-spec first builds before any
                            # code (run_05 DK iter1 47k tok/51min, Doom iter1 37k
                            # tok/42min — both tripped runaway_stream_warning,
                            # then shipped clean). Forcing the opening tag is
                            # constrained decoding, NOT an abort/cutoff: the full
                            # file still streams to completion; only the rambling
                            # preamble is skipped. Scoped to local backends so the
                            # cloud (Anthropic) path is untouched. Once iter 1
                            # materializes, _current_file is set and this branch
                            # no longer fires (iters 2+ patch as before).
                            reply_prefill = "<html_file>\n<!DOCTYPE html>\n"
                            prefill_force = True
                        yield self._record(AgentEvent(
                            "activity", "streaming",
                            {"label": f"iter {iteration} reply", "role": "coder"},
                        ))
                        reply = await self._stream(
                            self._token_cb_wrapper,
                            prefill=reply_prefill,
                            prefill_force=prefill_force,
                        )
                        for _ev in self._drain_stream_ui_events():
                            yield _ev
                        yield self._record(AgentEvent("activity", "idle"))
                    break
                except Exception as e:
                    yield self._record(AgentEvent("activity", "idle"))
                    err_msg = (
                        f"{self._backend.info.name.upper()} call failed: {e}"
                    )
                    yield self._record(AgentEvent("error", err_msg))
                    stall = self._classify_stall(str(e))
                    if stall:
                        yield self._record(AgentEvent(
                            "mlx_stall", err_msg,
                            {**stall, "phase": "iterate", "iteration": iteration},
                        ))
                    # Phase 5d: one transparent retry on a transient cloud
                    # crash (Anthropic / OpenAI). Recognized by the same
                    # error-text shape produced by the backend's catch
                    # block: `cause: <ExceptionClass>: ...`.
                    backend_name = getattr(
                        getattr(self._backend, "info", None),
                        "name",
                        None,
                    )
                    # MLX stall recovery — Fieldrunners 20260703: one retry
                    # after unloading diffusers + compaction instead of
                    # aborting the whole session on a silent 289s stall.
                    if stall and backend_name == "mlx":
                        _mlx_retries = getattr(
                            self, "_mlx_stall_retries_this_iter", 0
                        )
                        if _mlx_retries < 1:
                            self._mlx_stall_retries_this_iter = _mlx_retries + 1
                            self._force_compact_after_stall = True
                            # Compact NOW — _prune_messages already ran at
                            # iter start; retry must not resend the same
                            # giant prompt that just stalled.
                            self._prune_messages()
                            freed = await asyncio.to_thread(
                                self._release_diffusers_vram
                            )
                            self._trace({
                                "kind": "mlx_stall_recovery",
                                "iteration": iteration,
                                "attempt": _mlx_retries + 1,
                                "freed": freed,
                            })
                            if freed:
                                yield self._record(AgentEvent(
                                    "info",
                                    "MLX stall — unloaded "
                                    f"{', '.join(freed)}; compacting "
                                    "context and retrying once.",
                                ))
                            else:
                                yield self._record(AgentEvent(
                                    "info",
                                    "MLX stall — compacting context "
                                    "and retrying once.",
                                ))
                            continue
                    # P0b (MK trace 20260528): some Anthropic 400s are
                    # payload-shape errors that retrying the SAME payload
                    # can never fix. The trace burned two identical
                    # requests on "does not support assistant message
                    # prefill" before falling back. Detect those and
                    # skip the retry — go straight to local fallback.
                    err_str_lower = str(e).lower()
                    is_anthropic_shape_400 = (
                        backend_name == "anthropic"
                        and any(
                            phrase in err_str_lower
                            for phrase in _ANTHROPIC_NON_RETRYABLE_400_PHRASES
                        )
                    )
                    if is_anthropic_shape_400:
                        self._trace({
                            "kind": "anthropic_payload_shape_error",
                            "iteration": iteration,
                            "matched": next(
                                (
                                    p for p in _ANTHROPIC_NON_RETRYABLE_400_PHRASES
                                    if p in err_str_lower
                                ),
                                "",
                            ),
                            "err": str(e)[:300],
                        })
                        yield self._record(AgentEvent(
                            "info",
                            "[yellow]anthropic payload-shape 400[/yellow] — "
                            "skipping same-payload retry (would 400 again); "
                            "falling back to local backend if available.",
                        ))
                        # Do NOT consume the cloud_retry_attempted budget —
                        # leave it for an actual transient outage on a
                        # later iteration.
                    is_cloud_crash = (
                        backend_name in ("anthropic", "openai")
                        and "cause:" in str(e).lower()
                        and not cloud_retry_attempted
                        and not is_anthropic_shape_400
                    )
                    if is_cloud_crash:
                        cloud_retry_attempted = True
                        self._trace({
                            "kind": "cloud_crash_retry",
                            "iteration": iteration,
                            "backend": backend_name,
                            "err": str(e)[:200],
                        })
                        yield self._record(AgentEvent(
                            "info",
                            f"[dim]{backend_name} returned a transient "
                            "error; retrying same backend once before "
                            "falling back to a local backend.[/dim]",
                        ))
                        continue
                    # Phase 5c: drop the `continuation`-only guard. Trace 2
                    # (chess 20260522_104235) had a clean iter 1 then iter
                    # 2 hit a 0.48s Anthropic crash on the INITIAL run
                    # (continuation=False) — fallback was never attempted
                    # and the whole session was killed despite working
                    # iter 1 code on disk. Allow fallback on any iter.
                    if not fallback_attempted:
                        fallback_attempted = True
                        switched, note = await self._try_extension_backend_fallback(
                            stall=stall,
                            iteration=iteration,
                        )
                        if switched:
                            yield self._record(AgentEvent("info", note))
                            continue
                        if note:
                            yield self._record(AgentEvent("info", note))
                    if continuation and self._last_drained_feedback:
                        # Stream failed before we got any assistant reply; put
                        # the just-consumed feedback back in queue so extension
                        # requests are not silently dropped.
                        self._clear_scoped_constraints()
                        self._pending_feedback = (
                            self._last_drained_feedback + self._pending_feedback
                        )
                        self._trace({
                            "kind": "feedback_requeued_after_stream_failure",
                            "count": len(self._last_drained_feedback),
                        })
                    return

            self._messages.append({"role": "assistant", "content": reply})
            self._last_drained_feedback = []
            self._extract_and_queue_lookups(reply)
            self._capture_todos(reply)
            self._dump_conversation()
            self._trace({
                "kind": "assistant_reply",
                "iteration": iteration,
                "len": len(reply),
                # One-complete-trace (2026-06-14): store the FULL reply, not a
                # 600-char preview, so the .jsonl is self-sufficient (this used
                # to be recoverable only from the now-retired .conversation.md).
                "reply": reply,
            })
            violation = self._scoped_reply_violation(reply)
            if violation:
                # Fix B (seed-edit escape hatch): a seed-edit lock must always
                # make forward progress. Count consecutive violations; after 2
                # strikes, if the reply carries ANY usable code (patches or a
                # full <html_file> — the full seed is visible so a rewrite is a
                # real attempt), clear the lock and fall through to materialize
                # instead of burning every iter to nothing. Auto-revert remains
                # the safety net. Mid-session locks (no is_seed_edit) unchanged.
                cfg = self._scoped_constraints or {}
                self._scoped_violation_streak += 1
                has_usable_code = bool(extract_patches(reply)) or (
                    self._extract_html(reply) is not None
                )
                if (
                    cfg.get("is_seed_edit")
                    and self._scoped_violation_streak >= 2
                    and has_usable_code
                ):
                    self._trace({
                        "kind": "scoped_escape_hatch_used",
                        "iteration": iteration,
                        "violation": violation,
                        "streak": self._scoped_violation_streak,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "scoped guard: seed-edit escape hatch — applying "
                        "usable code after repeated scope violations.",
                    ))
                    self._clear_scoped_constraints()
                    # Fall through (no continue): the usable code is
                    # materialized by the normal path below.
                else:
                    self._trace({
                        "kind": "scoped_reply_rejected",
                        "iteration": iteration,
                        "violation": violation,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"scoped guard: {violation}",
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            self._scoped_retry_instruction(violation)
                        ),
                    })
                    continue

            # ---- mid-session asset / sound generation ------------------
            # If the model emitted a fresh <assets> or <sounds> block in
            # this iteration's reply (typically in response to user
            # feedback like "add proper sprites for the invaders"), run
            # the diffuser / audio pipeline now. The helper merges into
            # self._session_assets / self._session_sounds and queues a
            # USER FEEDBACK injection so the next user turn shows the
            # new file paths to the model. No-op when neither block is
            # present, so the hot path stays free.
            async for ev in self._maybe_generate_assets_and_sounds(
                reply, trigger="mid_session",
            ):
                yield ev

            # ---- coverage-gap probe re-parse ---------------------------
            # If the model emitted a fresh <probes> block alongside
            # usable code, decide whether to adopt it. Probes are
            # otherwise immutable after Phase A; this is the legitimate
            # mid-session update path.
            #
            # Gate fires when ANY of:
            #   (A) Phase-A coverage gap exists (the original trigger).
            #   (B) The prior iter's report had probe failures.
            #       Reason — DK trace 20260514_104131 post-seed:
            #       model wrote Phase-A probes BLIND (no seed file
            #       visible yet) referencing `state.grid`,
            #       `state.player.onLadder`, `state.reset`, none of
            #       which exist in the actual file. 4/5 of those
            #       probes evaluate falsy forever. In iter 1 the
            #       model re-emitted 5 corrected probes (3 dynamic)
            #       that DO match the file shape — but the old
            #       coverage-gaps-only gate dropped them. With this
            #       branch, failing probes invite an update.
            #   (C) Seed session, very first iter — the model just
            #       saw the seed file for the first time and may
            #       want to fix probes authored blind in Phase A.
            #
            # Defensive: the new probe list must have at least as many
            # entries as the current set, so the model can't shrink
            # the probe surface to mask regressions.
            reply_low = reply.lower()
            has_code = ("<patch>" in reply_low) or ("<html_file>" in reply_low)
            continuation_full_rewrite = bool(
                getattr(self, "_continuation", False)
                and "<html_file>" in reply_low
            )
            continuation_probe_refresh_adopted = False
            prev_report = self._previous_report or {}
            prev_probes = prev_report.get("probes") or []
            prev_probe_failures = sum(
                1 for p in prev_probes if not p.get("ok")
            )
            seed_iter1 = bool(self.seed_file) and iteration == start_iter
            allow_probe_reparse = (
                bool(self._planning_coverage_gaps)
                or prev_probe_failures > 0
                or seed_iter1
                or bool((self._scoped_constraints or {}).get("require_scope_probe"))
                or continuation_full_rewrite
            )
            if (
                allow_probe_reparse
                and "<probes>" in reply_low
                and has_code
            ):
                scoped_probe_required = bool(
                    (self._scoped_constraints or {}).get("require_scope_probe")
                )
                new_probes = self._extract_probes(reply)
                merged_scoped = False
                adopted_probes = list(new_probes)
                if (
                    scoped_probe_required
                    and new_probes
                    and len(new_probes) < len(self._probes)
                ):
                    # Scoped-check turns may re-emit a tiny probe set.
                    # Merge new probes into the current list instead of
                    # dropping them on the floor due to the length guard.
                    merged_scoped = True
                    seen = {
                        (
                            str(p.get("name") or "").strip().lower(),
                            str(p.get("expr") or "").strip().lower(),
                        )
                        for p in self._probes
                    }
                    adopted_probes = list(self._probes)
                    for p in new_probes:
                        sig = (
                            str(p.get("name") or "").strip().lower(),
                            str(p.get("expr") or "").strip().lower(),
                        )
                        if sig in seen:
                            continue
                        seen.add(sig)
                        adopted_probes.append(p)
                if (
                    adopted_probes
                    and (
                        continuation_full_rewrite
                        or len(adopted_probes) >= len(self._probes)
                    )
                ):
                    from tools import _criteria_coverage_gaps as _gaps_fn
                    fresh_criteria = self._extract_criteria(reply)
                    if continuation_full_rewrite:
                        if fresh_criteria:
                            self._criteria = fresh_criteria
                            self._trace({
                                "kind": "continuation_criteria_refreshed",
                                "iteration": iteration,
                                "chars": len(fresh_criteria),
                            })
                        else:
                            self._criteria = ""
                    new_gaps = _gaps_fn(self._criteria or "", adopted_probes)
                    self._trace({
                        "kind": "probes_reparsed",
                        "iteration": iteration,
                        "old_count": len(self._probes),
                        "new_count": len(adopted_probes),
                        "scoped_merged": merged_scoped,
                        "remaining_gaps": new_gaps[:6],
                        "trigger": (
                            "continuation_full_rewrite" if continuation_full_rewrite
                            else "coverage_gap" if self._planning_coverage_gaps
                            else "prev_probe_failures" if prev_probe_failures > 0
                            else "seed_iter1" if seed_iter1
                            else "scoped_probe"
                        ),
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"probes re-emitted ({len(self._probes)} → "
                        f"{len(adopted_probes)}); remaining coverage gaps: "
                        f"{len(new_gaps)}",
                    ))
                    self._probes = adopted_probes
                    # Model <probes> must not drop recipe auto_probes (facing eval).
                    self._maybe_inject_visual_playtest_auto_probes()
                    continuation_probe_refresh_adopted = continuation_full_rewrite
                    self._planning_coverage_gaps = new_gaps[:6]

            # ---- diagnose extraction (logged + memory-keyed) -----------
            diag = self._extract_diagnose(reply)
            if diag:
                self._last_diagnose = diag
                yield self._record(AgentEvent("diagnose", diag))
                # Shotgun-shape detector: flag when the model emitted a
                # ranked-hypothesis list. We don't reject the turn (the
                # patch may still be good); we just trace the violation
                # so postmortem can credit/blame this pattern, and
                # surface an info event so the user sees it too.
                if self._diagnose_is_shotgun(diag):
                    self._trace({
                        "kind": "diagnose_shotgun",
                        "preview": diag[:240],
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "format violation: <diagnose> emitted a ranked "
                        "hypothesis list. The fix_instruction prompt "
                        "asks for ONE root cause; this turn's patch "
                        "will still be tried.",
                    ))

            notes = self._extract_notes(reply)
            if notes:
                yield self._record(AgentEvent("info", f"notes: {notes[:200]}"))

            # ---- handle <question> from the model ----------------------
            q = self._extract_question(reply)
            html_in_reply = self._extract_html(reply)
            patches_in_reply = extract_patches(reply)

            # ---- diagnose-vs-patch subsystem-coherence note ------------
            # DK trace 20260514_175012 turn [08]: model's <diagnose>
            # named "barrel drop threshold + fallback coords" while
            # the harness had reported INPUT failure (signature
            # "Controls are not wired up") for 3 iterations. The
            # patches in the same turn touched barrels/coords, not
            # input handlers. Light touch: queue a coaching message
            # for the next turn surfacing the mismatch. Doesn't
            # reject the patch — the model may still have caught a
            # real bug, and the test report will tell us either way.
            #
            # Conditions:
            #   - a recent mistake_signature is set (i.e., we've had
            #     at least one failing iter and the harness named a
            #     subsystem),
            #   - the model emitted patches this turn (we skip the
            #     check on <html_file> replies — full rewrites might
            #     legitimately touch input without the SEARCH/REPLACE
            #     shape we scan),
            #   - <diagnose> body and patches BOTH ignore the
            #     subsystem identifiers.
            if (
                self._last_mistake_sig
                and patches_in_reply
                and diag is not None
            ):
                _hint = _subsystem_hint(self._last_mistake_sig)
                if _hint:
                    _idents = _hint["identifiers"]
                    if (
                        not self._diagnose_mentions_subsystem(diag, _idents)
                        and not self._patches_touch_subsystem_idents(
                            patches_in_reply, _idents,
                        )
                    ):
                        _idents_text = ", ".join(
                            f"`{i}`" for i in list(_idents)[:5]
                        )
                        self._pending_coaching.append(
                            f"COHERENCE NOTE — the harness has been "
                            f"reporting a {_hint['name'].upper()} "
                            "failure (signature: "
                            f"'{self._last_mistake_sig[:120]}'), but "
                            "your last <diagnose> and patches did "
                            "not mention or target any "
                            f"{_hint['name']} code "
                            f"({_idents_text}). Were you "
                            "intentionally addressing a different "
                            "bug, or did you miss this issue? If "
                            "the harness signal is real, target "
                            f"your next <patch> at {_hint['fix_phrase']}."
                        )
                        self._trace({
                            "kind": "diagnose_patch_coherence_mismatch",
                            "subsystem": _hint["name"],
                            "signature_preview": self._last_mistake_sig[:120],
                        })
            if q is not None and html_in_reply is None and not patches_in_reply:
                yield self._record(AgentEvent("question", q))
                while self._pending_answer is None:
                    await asyncio.sleep(0.1)
                # Tailor the post-answer prompt to whether there's already
                # a working file: a continuation answer often means "throw
                # out the old, rewrite". Spell out both options explicitly
                # so the model doesn't fall into a plan-only loop while it
                # tries to figure out which output tag applies.
                has_existing = bool(self._current_file)
                if has_existing:
                    followup = (
                        "Your answer is recorded above. Now produce CODE, "
                        "not another <plan>. Choose exactly one:\n"
                        "  - If your answer implies a major rewrite of the "
                        "existing file, emit one complete <html_file>...</html_file>.\n"
                        "  - Otherwise emit one or more <patch>...</patch> blocks "
                        "against the current file.\n"
                        "Do NOT re-emit <plan>, <criteria>, or <probes> this turn."
                    )
                else:
                    followup = (
                        "Your answer is recorded above. Now emit a complete "
                        "<html_file>...</html_file> for the first build. Do "
                        "NOT re-emit <plan>, <criteria>, or <probes>."
                    )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(followup),
                })
                continue

            # ---- handle done / confirm-done with no new code ----------
            # Two paths land here:
            #   (a) we asked the critique question (self._awaiting_confirm) and the
            #       model replied <confirm_done/>;
            #   (b) the previous iter passed cleanly and we sent the post-
            #       clean prompt encouraging <done/> — model replied <done/>
            #       (or <confirm_done/>) with no new code.
            # Either way: nothing to apply, nothing to test, ship it.
            said_done_or_confirm = bool(
                _CONFIRM_RE.search(reply) or _DONE_RE.search(reply)
            )
            # A2: <done/> needs a clean-streak of N iters (default 2).
            # self._awaiting_confirm bypasses the streak — the post-critique
            # <confirm_done/> already represents independent verification.
            streak_ok = (
                self._consecutive_clean_iters >= self._min_clean_streak_to_ship
            )
            # Stop-Losing-To-OneShot todo #3 — defend iter 1.
            # Mid-tier models often produce a working build on iter 1
            # then degrade it during the polish-encouraging clean-streak
            # gate. When the previous iter was honestly-clean (criteria
            # fully covered, no page errors, all model probes passed),
            # one clean iter is enough; the streak gate becomes
            # unnecessary tax, not insurance. Two clean iters is still
            # required when the harness signal is weaker (uncovered
            # criteria — coverage check still complains, or any probe
            # failed — soft_warning still appended; either case sets
            # `report["ok"]=False` so we never reach this branch).
            prev = self._previous_report or {}
            criteria_covered = not bool(prev.get("criteria_uncovered"))
            no_page_errors = not bool(prev.get("page_errors"))
            probes_all_passed = all(
                bool(p.get("ok"))
                for p in (prev.get("probes") or [])
            )
            single_clean_ship_ok = (
                self._consecutive_clean_iters >= 1
                and self._previous_report_ok is True
                and criteria_covered
                and no_page_errors
                and probes_all_passed
            )
            ship_ready = (
                self._awaiting_confirm
                or (self._previous_report_ok is True and streak_ok)
                or single_clean_ship_ok
            )
            # Model-gives-up gate (Wolfenstein 2026-05-24 trace [04]).
            # When the model emits <done/> with <notes> that confess
            # harness / parse trouble ("consistently failing to parse",
            # "user can manually copy"), it is NOT shipping a working
            # game — it's asking for help. Override the ship gate and
            # route through a recovery prompt that explicitly tells the
            # model the prior loop is the agent's bug to break, and that
            # it should send a fresh minimal <patch> targeting the
            # specific failing symptom rather than another <done/>.
            give_up_notes_text = self._extract_notes(reply) or ""
            model_gave_up = (
                said_done_or_confirm
                and GameAgent._notes_signal_give_up(give_up_notes_text)
            )
            if model_gave_up:
                self._trace({
                    "kind": "model_give_up_detected",
                    "iteration": iteration,
                    "notes_preview": give_up_notes_text[:240],
                    "hint": (
                        "Model emitted <done/> with notes confessing "
                        "harness/parse trouble. Treating as recovery "
                        "request instead of ship."
                    ),
                })
                # Drop the <done/> intent for THIS turn and inject a
                # recovery user message. The next iter will receive a
                # fresh stream prompt; the ship branch below is
                # bypassed because we `continue` here.
                recovery = (
                    "MODEL GIVE-UP DETECTED: your previous <notes> "
                    "block said the harness was failing to parse / the "
                    "user should manually copy the file. That is the "
                    "harness's bug to fix, not yours — you should not "
                    "ship with `<done/>` while the file is broken.\n\n"
                    "Recovery for THIS turn:\n"
                    "  - Read the CURRENT FILE ON DISK block in the "
                    "most recent fix prompt; it is the truth source.\n"
                    "  - Pick the ONE most concrete symptom from the "
                    "most recent test report and emit ONE small "
                    "<patch> with 3-5 lines of SEARCH context that "
                    "fixes only that one thing.\n"
                    "  - Do NOT emit <done/> again until the test "
                    "report comes back clean.\n"
                    "  - Do NOT emit a full <html_file> rewrite this "
                    "turn — the loop you were stuck in was caused by "
                    "re-emitting large bodies."
                )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(recovery),
                })
                continue
            if (
                said_done_or_confirm
                and html_in_reply is None
                and not patches_in_reply
                and ship_ready
            ):
                if self.has_pending_user_input():
                    yield self._record(AgentEvent(
                        "info",
                        "Model said done but user feedback is pending - applying it instead of exiting.",
                    ))
                    self._awaiting_confirm = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "You were about to ship, but the user just sent the "
                            "feedback above. Address it now and re-send a fix as <patch>."
                        ),
                    })
                    continue
                if single_clean_ship_ok and not (self._awaiting_confirm or streak_ok):
                    reason = (
                        "Model declared done after a clean iter with "
                        "covered criteria, no page errors, all probes passed."
                    )
                else:
                    reason = (
                        "Model confirmed after self-critique."
                        if self._awaiting_confirm
                        else "Model declared done after a clean run."
                    )
                yield self._record(AgentEvent("done", reason))
                self._record_session_outcome(ok=True)
                return
            # Stop-Losing-To-OneShot todo #3 (continued) — when the model
            # said done/confirm with no new code AND the previous iter
            # WAS clean but the gate refuses to ship (streak too short
            # AND coverage/probes weren't strong enough for the single-
            # clean shortcut), route into Phase C self-critique instead
            # of emitting "no usable code". The streak gate's whole
            # purpose is "second independent pass" — use the model's
            # <done/> as the trigger to enter that pass rather than as
            # an error. Also fixes the iter-2 deadlock from the
            # asteroids log (game-os-asteroids-vector-graph_20260507_).
            if (
                said_done_or_confirm
                and html_in_reply is None
                and not patches_in_reply
                and self._previous_report_ok is True
                and not self._awaiting_confirm
            ):
                yield self._record(AgentEvent("phase", "self-critique"))
                self._awaiting_confirm = True
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        self._p.CRITIQUE_INSTRUCTION
                    ),
                })
                self._fix_mode = False
                continue

            # ---- materialize: patches OR full file --------------------
            _pre_materialize_bytes = len(self._current_file or "")
            new_html, materialize_msg = await self._materialize(reply)
            if (
                new_html is not None
                and self._previous_report_ok is True
                and _pre_materialize_bytes
                and len(new_html) < int(_pre_materialize_bytes * 0.80)
            ):
                self._post_clean_shrink_detected = True
                self._trace({
                    "kind": "post_clean_shrink_detected",
                    "iteration": iteration,
                    "before_bytes": _pre_materialize_bytes,
                    "after_bytes": len(new_html),
                })
            else:
                self._post_clean_shrink_detected = False

            # Format-self-correction (DK trace 20260513_185815): when
            # materialize fails AND the failure looks like a malformed
            # shape (not just no-code-at-all), classify and — at streak
            # >= 2 — invoke an isolated-context format-doctor subagent
            # to reformat. Same model, fresh chat, narrow context.
            format_rejection: FormatRejection | None = None
            if new_html is None and not patches_in_reply:
                format_rejection = classify_format_failure(reply)
                if format_rejection is not None:
                    self._format_stuck_streak += 1
                    # Track iter-1 format rejections for the restart-
                    # signature comparison. Repeated iter-1 format
                    # failures are the strongest signal the model is
                    # over-scoped for what it can emit in one stream.
                    if iteration == 1:
                        self._format_rejections_iter1_this_attempt += 1
                    self._trace({
                        "kind": "format_rejection",
                        "rejection_kind": format_rejection.kind,
                        "streak": self._format_stuck_streak,
                        "iteration": iteration,
                    })
                    # Doctor invocation: normally streak == 2 (one
                    # bad reply may be a fluke). EARLY ESCALATION at
                    # streak == 1 when the stream was aborted as a
                    # repetition-loop OR stalled — those signals say
                    # the model was confused mid-emit; a second strike
                    # is wasted compute. DK trace 20260514_104131 hit
                    # this: post-seed iter 2 looped after 12706 tokens
                    # and emitted bare SEARCH/REPLACE markers; the
                    # session ended at streak=1 with no doctor call.
                    looped_or_stalled = (
                        self._last_stream_looped or self._last_stream_stalled
                    )
                    # A crash returns partial text with stalled=True, but
                    # format-doctor (re-stream from same context) is the
                    # wrong response: the underlying worker died, not the
                    # output shape. DK trace 2026-05-15: doctor burned ~5
                    # min re-streaming 34 KB after a crash and also truncated.
                    # Skip doctor entirely when the prior stream crashed —
                    # the GPU state was reset by `_drop_after_crash`, so
                    # the next iter starts fresh.
                    if self._last_stream_crashed:
                        self._trace({
                            "kind": "format_doctor_skipped_on_crash",
                            "iteration": iteration,
                            "rejection_kind": format_rejection.kind,
                        })
                        invoke_doctor = False
                    elif GameAgent._should_skip_format_doctor(
                        last_stream_looped=self._last_stream_looped,
                        last_stream_loop_kind=self._last_stream_loop_kind,
                        rejection_kind=format_rejection.kind,
                    ):
                        self._trace({
                            "kind": "format_doctor_skipped_inline_data_bloat",
                            "iteration": iteration,
                            "rejection_kind": format_rejection.kind,
                            "loop_kind": self._last_stream_loop_kind,
                        })
                        invoke_doctor = False
                    else:
                        invoke_doctor = (
                            self._format_stuck_streak == 2
                            or (
                                self._format_stuck_streak == 1
                                and looped_or_stalled
                            )
                        )
                    if invoke_doctor:
                        if looped_or_stalled and self._format_stuck_streak == 1:
                            self._trace({
                                "kind": "format_doctor_early_escalation",
                                "looped": self._last_stream_looped,
                                "stalled": self._last_stream_stalled,
                                "iteration": iteration,
                            })
                        yield self._record(AgentEvent(
                            "activity",
                            "format_doctor",
                            {"label": "format-doctor recovery"},
                        ))
                        try:
                            doctor_reply = await self._run_format_doctor(
                                reply, format_rejection,
                            )
                        finally:
                            yield self._record(AgentEvent("activity", "idle"))
                        if doctor_reply:
                            d_html, _d_msg = await self._materialize(
                                doctor_reply, dry_run=True,
                            )
                            # Trace 20260518_220003 (street-fighter): the
                            # doctor stream itself was cut off mid-output
                            # and produced a 183-byte stub. The dry-run
                            # materializer accepted it as non-None; the
                            # recovery branch declared `format_doctor_recovered`;
                            # only the downstream micro-probe pass caught
                            # the empty file. Validate the doctor's HTML
                            # the same way the regular pre-flight does
                            # before declaring recovery — if it would
                            # immediately fail micro-probes (essentially
                            # empty / unclosed / no <script>), reject the
                            # recovery here so the existing truncation-
                            # recovery path can take over without the
                            # misleading "recovered" trace event.
                            d_validation_ok = d_html is not None
                            d_validation_errors: list[str] = []
                            if d_html is not None:
                                d_mp = run_micro_probes(d_html)
                                d_validation_ok = bool(d_mp.get("ok", False))
                                if not d_validation_ok:
                                    d_validation_errors = list(
                                        d_mp.get("errors") or []
                                    )
                                    self._trace({
                                        "kind": "format_doctor_validation_failed",
                                        "rejection_kind": format_rejection.kind,
                                        "iteration": iteration,
                                        "size_bytes": len(d_html or ""),
                                        "errors": d_validation_errors[:3],
                                    })
                            if d_html is not None and d_validation_ok:
                                yield self._record(AgentEvent(
                                    "info",
                                    "[format-doctor] reformatted "
                                    f"unparseable reply "
                                    f"({format_rejection.kind}); using "
                                    "the corrected version this iter.",
                                ))
                                self._trace({
                                    "kind": "format_doctor_recovered",
                                    "rejection_kind": format_rejection.kind,
                                    "iteration": iteration,
                                })
                                # Replace the failed assistant reply in
                                # the conversation so subsequent turns
                                # see the parseable shape — avoids the
                                # model self-correcting backwards.
                                if (
                                    self._messages
                                    and self._messages[-1].get("role") == "assistant"
                                ):
                                    self._messages[-1] = {
                                        "role": "assistant",
                                        "content": doctor_reply,
                                    }
                                reply = doctor_reply
                                patches_in_reply = extract_patches(reply)
                                new_html, materialize_msg = (
                                    await self._materialize(reply)
                                )
                                self._format_stuck_streak = 0
            # First-build stub rescue (dragon's-lair trace 20260621_091419
            # iter 1): a placeholder-only first build — a <canvas> game was
            # intended but the <script> body is just comments/elisions —
            # is treated as no-usable-code so the EXISTING prefill retry
            # rescue below arms, instead of shipping the dead stub to
            # Chromium and recovering via the slower dead-build detour.
            # Gated to the first build (no baseline) so later patch turns
            # are untouched; `_is_placeholder_first_build` requires a
            # <canvas> so pure-DOM apps stay exempt. NOT a termination
            # change — only steers how the next attempt starts.
            if (
                new_html is not None
                and not self._current_file
                and _is_placeholder_first_build(new_html)
            ):
                self._trace({
                    "kind": "first_build_stub_rejected",
                    "iteration": iteration,
                    "size_bytes": len(new_html),
                })
                new_html = None
                materialize_msg = (
                    "first build was a placeholder stub (canvas present "
                    "but <script> body has no real code) — arming the "
                    "first-build prefill retry rescue"
                )
            if (
                new_html is None
                and self._media_regenerated_this_iter
                and self._current_file
                and (self._scoped_constraints or {}).get("mode") == "media_only"
            ):
                # Scoped media-only turns can be valid with no code edit:
                # regenerated files are picked up by existing loaders.
                new_html = self._current_file
                materialize_msg = "scoped media-only regeneration (no code patch)"
                self._trace({
                    "kind": "scoped_media_only_turn_accepted",
                    "iteration": iteration,
                })
            if new_html is None:
                self._last_iter_materialized = False
                if not self._current_file:
                    # Keep first-build rescue armed until code lands.
                    self._force_first_build_prefill = True
                # Phase 0D-4: same-iter auto-retry after a STALL on a feedback
                # turn. iters 4-5 (Fieldrunners) both stalled (deliberation /
                # loop) with the user's art+code feedback active and saved
                # nothing — then step mode said "iter complete" and waited for
                # Enter, so the edit silently vanished. Re-stream once
                # immediately (the step pause is skipped while
                # `_auto_retry_pending`) with the stall-aware fallback + the
                # retained asset reprompt, on a bonus iter so the user's budget
                # is not charged for the dead turn. Bounded to 2 retries.
                _stalled_no_code = bool(
                    self._last_stream_deliberated
                    or self._last_stream_looped
                    or self._last_stream_silent
                )
                _had_feedback = bool(
                    getattr(self, "_last_drained_feedback", None)
                    or getattr(self, "_unhonored_asset_request", None)
                )
                if (
                    self._current_file
                    and _stalled_no_code
                    and _had_feedback
                    and self._feedback_no_code_retries < 2
                    and self._iter_budget_bonus < revert_bonus_cap
                ):
                    self._feedback_no_code_retries += 1
                    self._iter_budget_bonus += 1
                    self._auto_retry_pending = True
                    _stall_reason = (
                        self._last_stream_loop_kind
                        or ("deliberation_loop" if self._last_stream_deliberated
                            else "silent_stream" if self._last_stream_silent
                            else "repetition_loop")
                    )
                    self._trace({
                        "kind": "feedback_no_code_retry",
                        "iteration": iteration,
                        "stall_reason": _stall_reason,
                        "bonus_total": self._iter_budget_bonus,
                        "retries": self._feedback_no_code_retries,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "[dim]feedback produced no code (stream stalled — "
                        f"{_stall_reason}); retrying this turn immediately on a "
                        "bonus iter so the edit isn't lost.[/dim]",
                    ))
                trunc = self._truncation_diagnosis(reply)
                if trunc:
                    yield self._record(AgentEvent("error", f"TRUNCATED REPLY — {trunc}"))
                else:
                    yield self._record(AgentEvent("info", f"no usable code: {materialize_msg}"))
                # First-build format-only recovery: grant one bonus iter so
                # the retry does not consume the normal iteration budget.
                if (
                    not self._current_file
                    and not self._first_build_retry_bonus_used
                    and self._iter_budget_bonus < revert_bonus_cap
                ):
                    self._first_build_retry_bonus_used = True
                    self._iter_budget_bonus += 1
                    self._trace({
                        "kind": "first_build_format_retry_bonus",
                        "iteration": iteration,
                        "bonus_total": self._iter_budget_bonus,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "[dim]first-build format recovery: granting +1 bonus "
                        "iter so this retry keeps the same effective build slot.[/dim]",
                    ))
                # Bonus iter for media-only emission. When the model
                # emitted <assets>/<sounds> (regen succeeded) but no
                # code, the harness picked up real work even though
                # this iter "failed" the code-output check. Don't
                # charge the user's max_iters budget for preparatory
                # work — the next iter is where the code emission
                # actually happens. Shares the budget pool with the
                # auto-revert bonus (cap = max_iters // 2).
                if (
                    self._media_regenerated_this_iter
                    and self._current_file
                    and self._iter_budget_bonus < revert_bonus_cap
                ):
                    self._iter_budget_bonus += 1
                    self._trace({
                        "kind": "media_only_bonus_iter",
                        "iteration": iteration,
                        "bonus_total": self._iter_budget_bonus,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"[dim]media-regen this iter; granting +1 iter "
                        f"(bonus total: {self._iter_budget_bonus}/"
                        f"{revert_bonus_cap}) so the next turn can "
                        "actually use the new assets.[/dim]"
                    ))
                # If the reply had patches but they failed to apply, give the
                # model the specific failures + current file so it can retry.
                if (
                    patches_in_reply
                    and self._current_file
                    and materialize_msg.startswith("patch set rejected")
                ):
                    # Pre-commit bracket rejection: the patches MATCHED but
                    # would have broken brace balance, so patch_retry_
                    # instruction (built from match failures) has nothing
                    # to say. Send the targeted bracket message instead.
                    self._patch_bracket_reject_streak += 1
                    _slice = ""
                    try:
                        first_search = patches_in_reply[0].search or ""
                        idx = self._current_file.find(first_search)
                        if idx >= 0:
                            start = max(0, idx - 500)
                            end = min(len(self._current_file), idx + len(first_search) + 500)
                            _slice = self._current_file[start:end]
                    except Exception:
                        _slice = ""
                    surgery = (
                        "\n\nPATCH SURGERY MODE — this is bracket rejection "
                        f"#{self._patch_bracket_reject_streak} in a row. "
                        "Do ONE tiny complete replacement only. Include the "
                        "entire enclosing function/block with balanced braces "
                        "in SEARCH and REPLACE. Do NOT emit <probes>, notes, "
                        "or a full <html_file> in this turn."
                    )
                    if _slice:
                        surgery += (
                            "\n\nCURRENT SOURCE SLICE AROUND THE FAILED PATCH "
                            "(patch this exact text):\n"
                            "```html\n" + _slice[:1800] + "\n```"
                        )
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            materialize_msg
                            + "\nThe file on disk is UNCHANGED (still the last "
                            "working version). Re-emit ONLY the corrected "
                            "<patch> block(s); count the braces in SEARCH vs "
                            "REPLACE before answering."
                            + surgery
                        ),
                    })
                elif patches_in_reply and self._current_file:
                    # Snapshot prior-turn failures BEFORE this turn's
                    # re-apply so the [REPEATED FAILURE] marker reflects
                    # cross-turn repeats, not within-turn duplicates.
                    prior_failed_anchors = set(self._last_failed_patch_anchors)
                    res = apply_patches(self._current_file, patches_in_reply)
                    # Current-turn failures → fingerprints for next turn.
                    new_failed_anchors = {
                        self._patch_anchor_fingerprint(p.search or "")
                        for (_i, p, _r) in res.failed
                    }
                    # Intersection = SEARCH blocks that failed last turn
                    # AND failed again this turn. patch_retry_instruction
                    # uses this to flag those bullets.
                    repeat_anchors = prior_failed_anchors & new_failed_anchors
                    if repeat_anchors:
                        self._trace({
                            "kind": "patch_search_repeat_detected",
                            "count": len(repeat_anchors),
                            "anchors": sorted(repeat_anchors),
                        })
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            self._p.patch_retry_instruction(
                                res.failed,
                                self._current_file,
                                repeat_anchors=repeat_anchors,
                                anchor_fingerprint=self._patch_anchor_fingerprint,
                            )
                        ),
                    })
                    # Remember THIS turn's failures so the next retry
                    # can flag repeats again.
                    self._last_failed_patch_anchors = new_failed_anchors
                else:
                    # Detect "plan-only" — the model emitted <plan> but no
                    # code. This is the failure mode from the
                    # model-8_20260511_111729 trace: after answering a
                    # <question> the model kept re-emitting the same
                    # <plan> indefinitely. Branch the fallback on whether
                    # an existing file is present and on how many
                    # consecutive plan-only turns we've seen.
                    plan_only = bool(_PLAN_OPEN_RE.search(reply))
                    if plan_only:
                        self._consecutive_plan_only += 1
                    else:
                        self._consecutive_plan_only = 0
                    # Detect probes-only / media-only re-emissions so the
                    # fallback can name the specific failure shape.
                    has_probes = bool(_PROBES_OPEN_RE.search(reply))
                    has_assets = bool(_ASSETS_OPEN_RE.search(reply))
                    has_sounds = bool(_SOUNDS_OPEN_RE.search(reply))
                    probes_only = (
                        has_probes and not has_assets and not has_sounds
                        and not plan_only
                    )
                    media_only = (
                        (has_assets or has_sounds) and not has_probes
                        and not plan_only
                    )
                    # Cross-turn identical-reply detector. When the SAME
                    # rejected reply lands twice in a row, the standard
                    # fallback (which is also identical) won't move the
                    # model — pass the signal into the fallback so it
                    # picks the scope-reduction escalation branch.
                    current_fp = GameAgent._reply_fingerprint(reply)
                    identical_repeat = (
                        current_fp != ""
                        and self._last_no_usable_code_fingerprint == current_fp
                    )
                    if identical_repeat:
                        self._identical_reply_loops_this_attempt += 1
                        self._trace({
                            "kind": "identical_reply_loop_detected",
                            "fingerprint": current_fp,
                            "reply_len": len(reply or ""),
                            "iteration": iteration,
                            "count_this_attempt": (
                                self._identical_reply_loops_this_attempt
                            ),
                        })
                    self._last_no_usable_code_fingerprint = current_fp
                    # Phase 0F: content-shape + stall fields on the no_usable_code
                    # event. When a reply produced no code, the single most
                    # useful debugging question is "what SHAPE was the reply?".
                    # iter 5 (Fieldrunners) looked like a mystery until we saw it
                    # was correct patches in the WRONG envelope (markdown
                    # SEARCH/REPLACE, no <patch> tag). These flags surface that
                    # in one line — no extra trace rows, no per-token spam.
                    _reply = reply or ""
                    # Phase 4 (4D.2): classify the fix-layer for THIS no-code
                    # turn. The canonical Fieldrunners iters 4-5 produced no
                    # code (so they emit THIS event, not iter_summary) — putting
                    # failure_class here is what makes "harness bug vs local-LLM
                    # limit" one-grep visible on exactly those failures.
                    _nuc_stall = (
                        self._last_stream_loop_kind
                        or ("deliberation_loop" if self._last_stream_deliberated
                            else "silent_stream" if self._last_stream_silent
                            else None)
                    )
                    _nuc_class, _nuc_reason = self._classify_failure(
                        ok=False,
                        materialized=False,
                        stall_reason=_nuc_stall,
                        coaching_suppressed=(
                            getattr(self, "_last_coaching_action", "none")
                            == "suppressed"
                        ),
                        asset_reprompt_cleared=bool(
                            getattr(self, "_last_asset_reprompt_cleared", False)
                        ),
                        art_intent=bool(self._session_assets),
                        undrawn_present=False,
                    )
                    self._trace({
                        "kind": "no_usable_code",
                        "failure_class": _nuc_class,
                        "failure_reason": _nuc_reason,
                        "plan_only": plan_only,
                        "probes_only": probes_only,
                        "media_only": media_only,
                        "consecutive_plan_only": self._consecutive_plan_only,
                        "has_existing_file": bool(self._current_file),
                        "identical_repeat": identical_repeat,
                        # content shape — why didn't this parse to code?
                        "reply_len": len(_reply),
                        "has_markdown_fence": "```" in _reply,
                        "has_search_replace": bool(
                            re.search(r"(?:^|\n)\s*(?:SEARCH|REPLACE)\s*:", _reply, re.I)
                        ),
                        "has_assets_tag": "<assets>" in _reply,
                        "has_patch_tag": "<patch>" in _reply,
                        "format_rejection_kind": (
                            format_rejection.kind if format_rejection is not None
                            else None
                        ),
                        # prior-stream stall context (drives recovery branch)
                        "prior_deliberated": bool(self._last_stream_deliberated),
                        "prior_looped": bool(self._last_stream_looped),
                        "prior_silent": bool(self._last_stream_silent),
                        "prior_loop_kind": self._last_stream_loop_kind,
                    })
                    # Rejected-reply stub (trace 20260611_213744): a
                    # format-rejected reply with NOTHING usable (no plan/
                    # probes/media either) is pure prompt poison — replace
                    # the just-appended assistant message with a short head
                    # + elision marker. Full text stays in the trace/.log.
                    # Fingerprinting above already ran on the full reply.
                    if (
                        format_rejection is not None
                        and not (plan_only or probes_only or media_only)
                        and self._messages
                        and self._messages[-1].get("role") == "assistant"
                        and self._messages[-1].get("content") == reply
                    ):
                        stub = GameAgent._stub_rejected_reply(
                            reply, format_rejection.kind
                        )
                        if stub is not None:
                            self._messages[-1]["content"] = stub
                            self._trace({
                                "kind": "rejected_reply_stubbed",
                                "iteration": iteration,
                                "rejection_kind": format_rejection.kind,
                                "chars_elided": len(reply) - len(stub),
                            })
                    fallback, reset_streak = (
                        GameAgent._no_usable_code_fallback(
                            plan_only=plan_only,
                            has_existing_file=bool(self._current_file),
                            consecutive_plan_only=self._consecutive_plan_only,
                            rejection=format_rejection,
                            format_stuck_streak=self._format_stuck_streak,
                            probes_only=probes_only,
                            media_only=media_only,
                            prior_stream_looped=self._last_stream_looped,
                            prior_stream_silent=self._last_stream_silent,
                            prior_stream_deliberated=self._last_stream_deliberated,
                            prior_loop_kind=self._last_stream_loop_kind,
                            prior_loop_line=self._last_stream_loop_line,
                            is_local_backend=(
                                self._backend.info.name in {"mlx", "ollama"}
                            ),
                            materialize_reject_reason=materialize_msg or "",
                            identical_repeat=identical_repeat,
                            # Phase 0D-2: art is pending when an asset reprompt is
                            # armed OR the router said the user wants new art. The
                            # deliberation/loop recovery then steers to <assets>
                            # first (the iter-2 pattern) instead of patch-only.
                            art_pending=bool(
                                getattr(self, "_unhonored_asset_request", None)
                                or (
                                    self._feedback_route is not None
                                    and self._feedback_route.get("allow_assets_block")
                                )
                            ),
                        )
                    )
                    if reset_streak:
                        self._consecutive_plan_only = 0
                        # Clear the fingerprint too: the escalation
                        # changed the prompt, the model should reply
                        # differently. Resetting prevents re-triggering
                        # on the next turn's reply even if the model
                        # ignores the escalation.
                        self._last_no_usable_code_fingerprint = None
                    # If this fallback ORDERS a full rewrite (duplicate-decl
                    # coaching, plan-only-with-file, format-stuck escalation,
                    # loop recovery), arm the one-shot exemption so the
                    # baseline-exists gate accepts the compliant reply.
                    # Qwen trace 151443 iters 4-6: the model obeyed the
                    # duplicate-decl coaching and got rejected three times.
                    if self._prompt_orders_full_rewrite(fallback):
                        self._allow_one_rewrite = True
                        self._trace({
                            "kind": "rewrite_exemption_armed_by_prompt",
                            "source": "no_usable_code_fallback",
                        })
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(fallback),
                    })
                continue

            # Code materialized — clear the plan-only loop counter so a
            # later plan-only reply doesn't inherit a stale streak. Also
            # clear the format-stuck streak: a successful parse means
            # whatever shape the model picked just worked, regardless
            # of what came before.
            # Phase 0D-4: record that this iter saved code, for the honest
            # step-mode pause label and the auto-retry gate.
            self._last_iter_materialized = True
            self._feedback_no_code_retries = 0
            # Phase 2: mark any harness-seeded ledger steps now present in the
            # materialized file (no-op once the model emits its own <todos>).
            self._mark_ledger_progress()
            if self._scoped_constraints is not None:
                scoped_check_keys = list(self._pending_scoped_check_keywords)
                self._trace({
                    "kind": "scoped_constraints_cleared_after_materialize",
                    "iteration": iteration,
                })
                self._clear_scoped_constraints()
                self._pending_scoped_check_keywords = scoped_check_keys
            self._force_first_build_prefill = False
            self._consecutive_plan_only = 0
            self._format_stuck_streak = 0
            self._patch_bracket_reject_streak = 0
            # Successful materialize means the model emitted a parseable
            # reply — any previous identical-reply loop has been broken.
            self._last_no_usable_code_fingerprint = None
            self._last_materialized_iter = iteration

            if continuation_full_rewrite and not continuation_probe_refresh_adopted:
                old_probe_count = len(self._probes)
                fresh_criteria = self._extract_criteria(reply)
                if fresh_criteria:
                    self._criteria = fresh_criteria
                    self._trace({
                        "kind": "continuation_criteria_refreshed",
                        "iteration": iteration,
                        "chars": len(fresh_criteria),
                    })
                else:
                    self._criteria = ""
                if old_probe_count:
                    self._probes = []
                    self._planning_coverage_gaps = []
                    self._probe_lint_findings = [
                        f for f in self._probe_lint_findings
                        if f.get("kind") not in (
                            "unassigned_property_read",
                            "probe_bait_flag",
                        )
                    ]
                    self._trace({
                        "kind": "continuation_stale_probes_retired",
                        "iteration": iteration,
                        "old_count": old_probe_count,
                        "reason": "full_html_rewrite_without_fresh_probes",
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "continuation rewrote the game shape; retired old "
                        "probes so stale checks do not force compatibility "
                        "with the previous build.",
                    ))

            # Track partial-patch failures for the next prompt even on
            # successful materialize.
            partial_failed: list[tuple[int, object, str]] = []
            if patches_in_reply and self._current_file:
                res = apply_patches(self._current_file, patches_in_reply)
                partial_failed = res.failed

            # Probe-sanity lint pass 2: now that we have the on-disk
            # HTML, check whether any probe reads `obj.prop` for a
            # property the game never assigns. Surfaces a soft warning
            # (one info event per finding) that flows into the next
            # diagnose-then-fix prompt via the test report. Donkey-kong
            # trace 20260516_124628 `barrels_move` is the canonical case
            # — gated on `b.x0` which no iter ever assigned.
            if self._probes:
                unassigned = GameAgent._probes_referencing_unassigned_props(
                    self._probes, new_html,
                )
                # Probe-sanity lint pass 3 (MK 20260517_220025 fix):
                # detect bait probes where the SAME iter's patch added
                # a literal flag the probe just reads back. The patch
                # REPLACE texts are the only evidence; pull them from
                # the applied patches.
                applied_replaces = [
                    getattr(p, "replace", "") or ""
                    for p in (patches_in_reply or [])
                ]
                baited = GameAgent._probes_baited_by_patches(
                    self._probes, applied_replaces,
                )
                if continuation_full_rewrite and unassigned:
                    stale_names = {
                        str(f.get("name") or "")
                        for f in unassigned
                        if f.get("name")
                    }
                    before_count = len(self._probes)
                    self._probes = [
                        p for p in self._probes
                        if str(p.get("name") or "") not in stale_names
                    ]
                    removed_count = before_count - len(self._probes)
                    if removed_count:
                        self._trace({
                            "kind": "continuation_stale_probes_removed",
                            "iteration": iteration,
                            "removed": sorted(stale_names),
                            "remaining": len(self._probes),
                        })
                        yield self._record(AgentEvent(
                            "info",
                            "continuation rewrite removed stale probes that "
                            "referenced runtime fields absent from the new "
                            "game shape.",
                        ))
                    unassigned = []
                if unassigned or baited:
                    # Combine with the tautological findings from Phase A;
                    # both flow to the model the same way.
                    self._probe_lint_findings = (
                        [
                            f for f in self._probe_lint_findings
                            if f.get("kind")
                            not in ("unassigned_property_read", "probe_bait_flag")
                        ]
                        + unassigned
                        + baited
                    )
                    self._trace({
                        "kind": "probe_lint_postbuild",
                        "iteration": iteration,
                        "findings": (unassigned + baited),
                    })

            # The reply materialized real code — if an unhonored art
            # request is outstanding and these patches touch its subject,
            # stand down the ASSET GENERATION REQUIRED reprompt (the
            # model chose a code solution; see the method docstring).
            self._maybe_clear_asset_reprompt_via_code(reply)

            # Save + per-iter snapshot. Keep the prior HTML in memory so a
            # visual/procedural regression can restore the better baseline
            # even when no clean best.html exists yet.
            _pre_iter_html = self._current_file
            self.out_path.write_text(new_html, encoding="utf-8")
            self._current_file = new_html
            if self._new_assets_not_in_html:
                refs = self._scan_html_for_asset_refs(new_html)
                self._new_assets_not_in_html -= refs
            self._scoped_violation_streak = 0  # Fix B: progress made, reset thrash counter
            snap_path = self._save_snapshot(new_html)
            shot_path = snap_path.with_suffix(".png") if snap_path else None
            # 2.3: per-iter HTML sha256 so test events can be correlated
            # back to the exact code that produced them. Iter snapshots
            # share this hash with their .html sibling on disk.
            import hashlib as _hashlib
            html_sha = _hashlib.sha256(
                new_html.encode("utf-8", "replace")
            ).hexdigest()[:16]
            # T-2: remember THIS iter's code hash so the NEXT iter_summary can
            # flag shipped_unchanged_after_block (a prior ok=False block that
            # changed nothing — the definitive false-positive marker).
            self._cur_iter_code_sha = html_sha
            self._trace({
                "kind": "code_snapshot",
                "iteration": iteration,
                "html_sha256": html_sha,
                "size": len(new_html),
                "snapshot": str(snap_path) if snap_path else None,
                "screenshot": str(shot_path) if shot_path else None,
                "materialize": materialize_msg,
                "patches_applied": len(patches_in_reply) - len(partial_failed),
                "patches_failed": len(partial_failed),
                "diagnose": self._last_diagnose,
            })
            yield self._record(AgentEvent(
                "code",
                str(self.out_path),
                {
                    "size": len(new_html),
                    "html_sha256": html_sha,
                    "snapshot": str(snap_path) if snap_path else None,
                    "screenshot": str(shot_path) if shot_path else None,
                    "materialize": materialize_msg,
                    "patches_applied": len(patches_in_reply) - len(partial_failed),
                    "patches_failed": len(partial_failed),
                },
            ))

            # ---- asset-reference alignment scan -------------------------
            # Static check before Chromium load: do the asset names the
            # HTML references actually have files on disk? In the
            # donkey-kong trace 20260513_122154 this gap drove 3 wasted
            # iters of "fix drawImage" patching. Coaching the model
            # with the exact missing names lets it either emit a fresh
            # <assets> block (mid-session regen fulfills it) or stop
            # referencing names that don't exist.
            try:
                self._check_asset_alignment(new_html)
            except Exception:
                # Never let the scan crash the loop.
                pass
            try:
                self._check_sound_alignment(new_html)
            except Exception:
                # Never let the scan crash the loop.
                pass

            # ---- pre-Chromium micro-probes (OpenCoder #4) ---------------
            # Cheap structural sanity check before paying the ~3s Chromium
            # round-trip. Catches truncation, empty scripts, badly-
            # unbalanced braces, and elision markers. Only ERRORS skip
            # the browser; warnings pass through and Chromium gets the
            # final word.
            mp = run_micro_probes(new_html, out_path=self.out_path)
            if continuation_full_rewrite:
                suppressed = (
                    self._mark_unused_media_as_stale_for_continuation(mp)
                )
                if suppressed:
                    self._trace({
                        "kind": "continuation_stale_media_context",
                        "iteration": iteration,
                        "unused_assets": (
                            (mp.get("stats") or {}).get("unused_assets")
                        ),
                        "suppressed_warning_count": suppressed,
                    })
            self._trace({
                "kind": "micro_probes",
                "ok": mp.get("ok", False),
                "errors": list(mp.get("errors") or []),
                "warnings": list(mp.get("warnings") or []),
                "stats": mp.get("stats") or {},
            })
            if not mp.get("ok", True):
                mp_text = format_micro_probes_for_model(mp)
                self._last_report_summary = mp_text
                # Surface as a "test" event so the TUI prints it the
                # same way it shows browser test failures.
                yield self._record(AgentEvent(
                    "test",
                    mp_text,
                    {
                        "ok": False,
                        "errors": mp.get("errors") or [],
                        "soft_warnings": [],
                        "warnings": mp.get("warnings") or [],
                        # Keep parity with the browser-report shape so
                        # downstream consumers (best.html save, regression
                        # detection) don't trip on a missing key.
                        "title": "(skipped browser — pre-flight failed)",
                        "canvas": None,
                        "input_listeners": {},
                        "input_test": None,
                        "frozen_canvas": None,
                        "body_chars": 0,
                        "body_sample": "",
                        "logs": [],
                        "probes": [],
                        "screenshot": None,
                        "screenshot_before": None,
                    },
                ))
                # Build the same kind of fix prompt the test loop builds
                # below; bypass the Chromium path entirely.
                self._stuck_streak += 1
                fake_report = {
                    "ok": False,
                    "errors": mp.get("errors") or [],
                    "soft_warnings": [],
                    "warnings": mp.get("warnings") or [],
                    "title": "(pre-flight)",
                    "canvas": None,
                    "input_listeners": {},
                    "input_test": None,
                    "frozen_canvas": None,
                    "body_chars": 0,
                    "body_sample": "",
                    "logs": [],
                    "probes": [],
                }
                next_user = self._build_fix_prompt(
                    report=fake_report, regressed=False, partial_failed=partial_failed,
                )
                # Item 3: wrap so a superseded pre-flight report collapses
                # to its digest once it ages out of the keep window.
                next_user = self._wrap_report_block(next_user, fake_report)
                self._fix_mode = True
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(next_user),
                })
                self._previous_report_ok = False
                self._previous_report = fake_report  # todo #3 — full report
                continue

            # Skip Chromium when mid-session art landed but HTML was not
            # patched with PATHS/loader entries (Fieldrunners 20260703).
            _paths_pending = self._new_assets_not_in_html
            if _paths_pending and self.browser is not None:
                pending_list = ", ".join(sorted(_paths_pending))
                self._trace({
                    "kind": "browser_test_skipped",
                    "iteration": iteration,
                    "reason": "paths_pending_after_mid_session_assets",
                    "pending_assets": sorted(_paths_pending),
                    "micro_probes_ok": bool(mp.get("ok", True)),
                })
                yield self._record(AgentEvent(
                    "info",
                    "[dim]browser skipped — mid-session sprites exist on "
                    "disk but HTML PATHS/loader not updated yet.[/dim]",
                ))
                self._stuck_streak += 1
                fake_report = {
                    "ok": False,
                    "errors": [
                        "MID-SESSION ASSETS NOT WIRED: generated sprites "
                        f"({pending_list}) exist on disk but this HTML does "
                        "not reference them. Emit a <patch> adding PATHS/loader "
                        "entries and wire drawSprite/sprite for these exact "
                        "keys before browser verification.",
                    ],
                    "soft_warnings": [],
                    "warnings": mp.get("warnings") or [],
                    "title": "(skipped browser — paths pending)",
                    "canvas": None,
                    "input_listeners": {},
                    "input_test": None,
                    "frozen_canvas": None,
                    "body_chars": 0,
                    "body_sample": "",
                    "logs": [],
                    "probes": [],
                    "test_skipped": "paths_pending_after_mid_session_assets",
                }
                mp_text = format_micro_probes_for_model(mp) if not mp.get("ok", True) else (
                    fake_report["errors"][0]
                )
                self._last_report_summary = mp_text
                yield self._record(AgentEvent("test", mp_text, fake_report))
                next_user = self._build_fix_prompt(
                    report=fake_report, regressed=False, partial_failed=partial_failed,
                )
                next_user = self._wrap_report_block(next_user, fake_report)
                self._fix_mode = True
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(next_user),
                })
                self._previous_report_ok = False
                self._previous_report = fake_report
                continue

            # ---- run the test ------------------------------------------
            shot_before_path = (
                snap_path.with_name(snap_path.stem + "_before.png")
                if (snap_path and self._use_double_screenshot)
                else None
            )
            shot_action_path = (
                snap_path.with_name(snap_path.stem + "_action.png")
                if snap_path else None
            )
            if self.browser is None:
                report = self._synthetic_report_no_browser(mp)
                self._trace({
                    "kind": "browser_test_skipped",
                    "iteration": iteration,
                    "reason": "no_browser",
                    "micro_probes_ok": bool(mp.get("ok", True)),
                })
                yield self._record(AgentEvent("activity", "idle"))
            else:
                yield self._record(AgentEvent(
                    "activity", "browser",
                    {"label": f"loading iter {iteration} in Chromium"},
                ))
                # Harness-crash handling (2026-06-10, dojo-fight trace 151443):
                # retry ONCE before charging the model an iteration — crashes can
                # be data-shape-dependent (the closing verification on the same
                # session ran fine), and a single retry often succeeds. If both
                # attempts crash, tell the model the truth: the harness failed,
                # the GAME WAS NOT TESTED, and it must NOT change the game in
                # response. The old message ("simplify the page, try again")
                # blamed the game for a Python bug — Qwen obeyed and spent
                # 2 iterations shrinking a working build.
                report = None
                harness_crash: Exception | None = None
                for _test_attempt in (1, 2):
                    try:
                        report = await self.browser.load_and_test(
                            self.out_path, screenshot_path=shot_path,
                            screenshot_before_path=shot_before_path,
                            screenshot_action_path=shot_action_path,
                            probes=self._probes or None,
                            opening_book_recipes=getattr(self, "_active_opening_book_recipes", []),
                            # todo #2: pass criteria so the harness can flag
                            # coverage gaps as a soft_warning (forces the model
                            # to add probes that actually test what it promised).
                            criteria=self._criteria or None,
                            goal=self._goal or "",
                            visual_recipe_id=getattr(self, "_active_visual_playtest_recipe_id", None),
                        )
                        harness_crash = None
                        break
                    except Exception as e:
                        harness_crash = e
                        self._trace_exception(
                            "harness_crash", e,
                            iteration=iteration, test_attempt=_test_attempt,
                        )
                if harness_crash is not None or report is None:
                    yield self._record(AgentEvent("activity", "idle"))
                    yield self._record(AgentEvent(
                        "info",
                        f"browser harness crashed (after retry): {harness_crash}",
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "HARNESS FAILURE (not a game bug): the browser test "
                            f"harness itself crashed with an internal error: "
                            f"{harness_crash}\n"
                            "Your game was NOT tested this iteration — there is "
                            "NO test signal, good or bad. Do NOT change the game "
                            "in response to this message, and do NOT simplify or "
                            "rewrite working code. Continue with your open "
                            "<todos> items if any remain, or reply <done/> if "
                            "you believe the build is complete."
                        ),
                    })
                    continue

                yield self._record(AgentEvent("activity", "idle"))
            if partial_failed:
                # Partial patch-apply means intended fixes are not fully
                # landed, even if probes happen to pass. Force one focused
                # recovery turn so unresolved SEARCH targets are addressed.
                sw = list(report.get("soft_warnings") or [])
                sw.append(
                    f"partial patch apply: {len(partial_failed)} patch block(s) "
                    "did not apply; fix not complete."
                )
                report["soft_warnings"] = sw
                report["ok"] = False
                self._trace({
                    "kind": "partial_patch_forced_retry",
                    "failed_count": len(partial_failed),
                    "iteration": iteration,
                })
            self._handle_probe_eval_errors(report, iteration)
            self._apply_scoped_check_to_report(report)
            self._apply_dead_animation_check_to_report(report)
            # Phase 0C: promote undrawn-art advisory back to a blocking
            # soft_warning on iter-1 / art-feedback-pending (golden-iter-2 safe).
            self._apply_undrawn_art_intent_gate(report)
            self._apply_dropped_assets_pending_gate(report)
            self._apply_still_frame_frozen_downgrade(report)
            self._apply_player_stuck_downgrade(report)
            # Demote structurally-impossible self-probes (probe_lint already
            # flagged them) so they stop permanently gating ok=false. Runs
            # after _handle_probe_eval_errors so eval-error probes are routed
            # to their own quarantine path first.
            self._apply_impossible_probe_downgrade_to_report(report)
            # Advance the warnings-persistence counter ONCE per iter,
            # before any format_report_for_model rendering. Compaction
            # is then applied uniformly to the test event text and to
            # the next user turn's report block.
            self._advance_warning_persistence(report.get("warnings") or [])
            report_text = self._format_report_for_model(report)
            self._last_report_summary = report_text
            self._last_test_report = report
            self._last_tested_iter = iteration
            yield self._record(AgentEvent("test", report_text, report))

            # Phase 3 — per-iter summary in the trace. Structured,
            # one record per iter, with enough signal to spot patterns
            # across sessions without re-grepping heartbeats.
            try:
                probes_list = report.get("probes") or []
                probes_passed = sum(1 for p in probes_list if p.get("ok"))
                probes_total = len(probes_list)
                soft_warnings = report.get("soft_warnings") or []
                page_errors = report.get("page_errors") or []
                console_errors = report.get("console_errors") or []
                # fail_reason lists BLOCKING causes only, and only when the
                # iter actually failed (ok=False). frozen_canvas is NOT added
                # here: a true freeze already appears as a FROZEN-CANVAS
                # soft_warning, and an input-responsive "frozen" canvas is
                # idle-by-design (advisory). Trace 20260613_213711 showed
                # frozen_canvas listed as a fail_reason on ok=True iters,
                # which was misleading. Advisories go in their own field.
                fail_reasons: list[str] = []
                if page_errors:
                    fail_reasons.append(f"page_errors:{len(page_errors)}")
                if console_errors:
                    fail_reasons.append(f"console_errors:{len(console_errors)}")
                if soft_warnings:
                    fail_reasons.append(f"soft_warnings:{len(soft_warnings)}")
                advisories: list[str] = []
                if report.get("frozen_canvas"):
                    advisories.append(
                        "frozen_canvas_idle_by_design"
                        if report.get("frozen_canvas_input_responsive")
                        else "frozen_canvas"
                    )
                _ok = bool(report.get("ok"))
                entity_check = report.get("entity_render_check") or {}
                missing_entities = (entity_check.get("missing") or []) if isinstance(entity_check, dict) else []
                _static_action = report.get("static_action")
                # Phase 4 (4D.1): the stall reason for THIS iter's stream — used
                # both as a field and as the failure_class input below.
                _stall_reason = (
                    self._last_stream_loop_kind
                    or ("deliberation_loop" if self._last_stream_deliberated
                        else "silent_stream" if self._last_stream_silent
                        else None)
                )
                # Phase 4 (4D.1): patch result "N/M" for this iter (None when no
                # patch turn — e.g. a full <html_file> build).
                _patch_applied = getattr(self, "_last_patch_applied", None)
                _patch_total = getattr(self, "_last_patch_total", None)
                _patch_str = (
                    f"{_patch_applied}/{_patch_total}"
                    if _patch_applied is not None and _patch_total is not None
                    else None
                )
                # Phase 4 (4D.1): task-ledger progress "done/total" (harness-seeded
                # or model-emitted; "" when no ledger).
                _ledger_items = getattr(self, "_todos_items", None) or []
                _ledger_str = (
                    f"{sum(1 for d, _ in _ledger_items if d)}/{len(_ledger_items)}"
                    if _ledger_items else ""
                )
                # Phase 4 (4D.2): which LAYER needs the fix (advisory triage).
                _undrawn_present = any(
                    isinstance(w, str) and "ASSETS_LOADED_BUT_UNDRAWN" in w
                    for w in (
                        list(report.get("warnings") or [])
                        + list(soft_warnings)
                    )
                )
                _launch_pf_fail = any(
                    not p.get("ok")
                    and isinstance(p.get("name"), str)
                    and p["name"].startswith("auto_")
                    and any(
                        tok in p["name"]
                        for tok in ("launch", "playfield", "enter")
                    )
                    for p in (report.get("probes") or [])
                )
                _failure_class, _failure_reason = self._classify_failure(
                    ok=_ok,
                    materialized=True,
                    stall_reason=_stall_reason,
                    coaching_suppressed=(
                        getattr(self, "_last_coaching_action", "none") == "suppressed"
                    ),
                    asset_reprompt_cleared=bool(
                        getattr(self, "_last_asset_reprompt_cleared", False)
                    ),
                    art_intent=bool(self._session_assets),
                    undrawn_present=_undrawn_present,
                    # T-1: feed the "model right, harness wrong" signature so a
                    # green-probes build blocked only by a soft_warning gate is
                    # tagged harness_bug (was mislabeled `none`).
                    probes_all_passed=(
                        probes_total > 0 and probes_passed >= probes_total
                    ),
                    has_page_errors=bool(page_errors),
                    has_soft_warnings=bool(soft_warnings),
                    launch_playfield_probe_failed=_launch_pf_fail,
                )
                self._last_failure_class = _failure_class
                # T-2: ground-truth false-positive marker. True when the PRIOR
                # iter was ok=False yet this iter's code is byte-identical to it
                # (same sha) — i.e. the block changed nothing, so it was a
                # harness false positive (Holochess/Dragon shipped unchanged
                # after a blocked iter). Computed from hashes the agent already
                # holds; no new event, one boolean on the existing summary.
                _cur_sha = getattr(self, "_cur_iter_code_sha", None)
                _shipped_unchanged_after_block = bool(
                    self._previous_report_ok is False
                    and _cur_sha is not None
                    and _cur_sha == getattr(self, "_prev_iter_code_sha", None)
                )
                # T-3: compact per-probe digest (name + ok + short expr) for
                # EVERY probe, passing and failing. Previously iter_summary
                # carried only probes_passed/total counts and the failing-probe
                # text lived only in soft_warnings — an LLM reading the trace
                # could not answer "which probe asserted what?" from the summary
                # alone (hit on the Dragon failing iter). Bounded so it stays a
                # digest, not a dump: <=24 probes, expr clipped to 120 chars.
                _probe_digest = [
                    {
                        "name": str(p.get("name") or "")[:60],
                        "ok": bool(p.get("ok")),
                        "expr": str(p.get("expr") or "")[:120],
                    }
                    for p in (report.get("probes") or [])
                ][:24]
                summary_payload = {
                    "kind": "iter_summary",
                    "iteration": iteration,
                    "ok": bool(report.get("ok")),
                    "shipped_unchanged_after_block": _shipped_unchanged_after_block,
                    # Phase 0F: code landed this iter (reaching iter_summary means
                    # it did) + any stall on the stream that produced it — one
                    # scannable line covering the same fields as no_usable_code.
                    "materialized": True,
                    "last_stall_reason": _stall_reason,
                    # Phase 4 (4D.1): correlation fields folded onto iter_summary
                    # so ONE row answers "why did this iter go the way it did?"
                    # without joining stream_done / patch_outcome / router /
                    # coaching events. tok/s separates a slow LOCAL MODEL from a
                    # harness stall; prefill_s flags cold-KV time-to-first-token.
                    "stream_tokens": int(getattr(self, "_last_stream_tokens", 0) or 0),
                    "stream_duration_s": float(getattr(self, "_last_stream_duration_s", 0.0) or 0.0),
                    "tok_per_s": float(getattr(self, "_last_stream_tok_per_s", 0.0) or 0.0),
                    "prefill_s": float(getattr(self, "_last_prefill_s", 0.0) or 0.0),
                    "router_intent": getattr(self, "_last_router_intent", None),
                    "patch_applied": _patch_str,
                    "coaching_action": getattr(self, "_last_coaching_action", "none"),
                    "task_ledger_done": _ledger_str,
                    # Phase 4 (4D.2): the fix-layer bucket + one-line evidence.
                    "failure_class": _failure_class,
                    "failure_reason": _failure_reason,
                    "probes_passed": probes_passed,
                    "probes_total": probes_total,
                    # Offline scoreboard / credit: active playbook ids without
                    # joining a separate playbook_retrieved event.
                    "retrieved_ids": list(getattr(self, "_active_bullet_ids", []) or []),
                    # T-3: per-probe name/ok/expr digest (see _probe_digest above).
                    "probes": _probe_digest,
                    "soft_warnings_count": len(soft_warnings),
                    "page_errors_count": len(page_errors),
                    "console_errors_count": len(console_errors),
                    # One-complete-trace (2026-06-14): the actual error STRINGS
                    # in full (not just counts), so a broken-game run is fully
                    # debuggable from the .jsonl alone. *_count kept above for
                    # render_run_summary / /revert back-compat.
                    "page_errors": list(page_errors),
                    "console_errors": list(console_errors),
                    "frozen_canvas": bool(report.get("frozen_canvas")),
                    "entity_missing_count": len(missing_entities),
                    # Verification observability (so a future trace answers
                    # "did the critic even see an action?" with one grep):
                    "action_frame_captured": bool(report.get("screenshot_action")),
                    "action_key": report.get("action_key"),
                    "static_action": _static_action,
                    # Blocking causes only when ok=False; "ok" otherwise.
                    "fail_reason": (",".join(fail_reasons) or "blocked") if not _ok else "ok",
                    "advisories": advisories,
                    # Prompt-bloat + context signals (2026-06-13) so the
                    # "agent worse than single-shot" symptom is visible per
                    # iter without joining stream_done by hand.
                    "prompt_tokens": int(getattr(self, "_last_prompt_tokens", 0) or 0),
                    "message_count": len(self._messages),
                    "lean_prompt": self._lean_prompt_active(),
                    # Higher signal-to-noise debugging (2026-05-31): record WHY
                    # it blocked (the actual soft-warning texts, not just a
                    # count), the non-blocking warnings, the frozen-canvas
                    # false-positive classifier, and any queued feedback that a
                    # non-ok iter will defer — so "stuck + user request starved"
                    # is visible from this one event.
                    # One-complete-trace (2026-06-14): full text, no artificial
                    # caps — record every soft_warning / warning / pending
                    # feedback item in full so nothing the model saw is lost.
                    "soft_warnings": [str(w) for w in soft_warnings],
                    "warnings": [str(w) for w in (report.get("warnings") or [])],
                    "frozen_canvas_input_responsive": report.get("frozen_canvas_input_responsive"),
                    "pending_feedback": list(self._pending_feedback or []),
                    "pending_feedback_count": len(self._pending_feedback or []),
                    # Undrawn-FP observability (run_10): persist the drawn-asset
                    # audit + decode-settle result so a trace alone answers
                    # "was ASSETS_LOADED_BUT_UNDRAWN a false positive?".
                    "drawn_asset_check": report.get("drawn_asset_check"),
                    "asset_decode_settle": report.get("asset_decode_settle"),
                }
                if report.get("test_skipped"):
                    summary_payload["test_skipped"] = report.get("test_skipped")
                self._trace(summary_payload)
                # T-2: roll the code-hash forward so the NEXT iter can compare
                # against THIS iter's shipped bytes (pairs with the existing
                # _previous_report_ok roll-forward below).
                self._prev_iter_code_sha = _cur_sha
                # Phase 3 surprise rules — fire on signals worth
                # postmortem attention. Surprise events are a separate
                # kind so they're easy to grep out of the trace.
                if missing_entities and report.get("ok"):
                    # Probes passed but entity-render check found gaps.
                    # This is the Pac-Man-without-Pac-Man shape.
                    self._trace({
                        "kind": "surprise",
                        "category": "state_vs_render_gap",
                        "iteration": iteration,
                        "missing_entities": [
                            m.get("name") for m in missing_entities
                            if isinstance(m, dict)
                        ],
                        "hint": (
                            "Probes passed but the entity-render check "
                            "flagged entities that exist in state but "
                            "aren't drawn on the canvas. Strong signal "
                            "to add a dynamic probe that samples canvas "
                            "pixels at the player position."
                        ),
                    })
                if not report.get("ok") and self._previous_report_ok is True:
                    # Last iter was clean; this one regressed.
                    self._trace({
                        "kind": "surprise",
                        "category": "regression_after_clean_iter",
                        "iteration": iteration,
                        "fail_reason": summary_payload["fail_reason"],
                        "hint": (
                            "Iter N-1 passed all probes; iter N "
                            "regressed. The fix-mode prompt should "
                            "favour reverting recent edits, not "
                            "elaborating on them. Common cause: model "
                            "rewrote working code while trying to add "
                            "a feature."
                        ),
                    })
                # Dead-first-build detector (universal). On iter 1 or 2,
                # if the file loaded but RAF never fired AND the input
                # smoke test registered no state/canvas change, the file
                # is structurally dead — patches will not save it. Set
                # a one-shot flag so the next fix turn uses the scope-
                # reduction prompt instead of diagnose-then-fix. Wolfenstein
                # 2026-05-24 trace burned 6+ iters trying to patch a dead
                # first build before timing out.
                try:
                    canv = report.get("canvas") or {}
                    input_test = report.get("input_test") or {}
                    raf_dead = (
                        isinstance(canv, dict)
                        and canv.get("raf_ran") is False
                    )
                    input_dead = (
                        isinstance(input_test, dict)
                        and input_test.get("ran") is True
                        and input_test.get("any_change") is False
                    )
                    if (
                        iteration <= 2
                        and not report.get("ok")
                        and raf_dead
                        and input_dead
                    ):
                        self._dead_first_build_recoveries += 1
                        self._dead_first_build_pending = True
                        self._trace({
                            "kind": "dead_first_build_detected",
                            "iteration": iteration,
                            "raf_ran": canv.get("raf_ran"),
                            "input_any_change": input_test.get("any_change"),
                            "recoveries_this_attempt": (
                                self._dead_first_build_recoveries
                            ),
                            "hint": (
                                "First build loaded but RAF never fired "
                                "and input did nothing — structurally "
                                "dead file. Next turn will request a "
                                "smaller intentionally-minimal rewrite "
                                "instead of patching."
                            ),
                        })
                        # Two recoveries in one attempt: the model can't
                        # ship a working minimal build either. Flag for
                        # the restart loop to apply a fresh seed/recipe
                        # rather than wasting more of this attempt's
                        # iteration budget.
                        if self._dead_first_build_recoveries >= 2:
                            self._dead_first_build_abort_attempt = True
                            self._trace({
                                "kind": "dead_first_build_abort_attempt",
                                "iteration": iteration,
                                "recoveries": (
                                    self._dead_first_build_recoveries
                                ),
                                "hint": (
                                    "Two dead first builds in one "
                                    "attempt; ending the attempt "
                                    "early so the restart loop picks "
                                    "up with a different seed."
                                ),
                            })
                except Exception as e:
                    self._trace({
                        "kind": "dead_first_build_detector_error",
                        "iteration": iteration,
                        "err": str(e)[:200],
                    })
            except Exception as e:
                self._trace_exception(
                    "iter_summary_error", e, iteration=iteration,
                )

            # Auto-arm step-mode on the FIRST failing iter. Rationale
            # (donkey-kong trace 20260513_122154): iter 2 burned 7 min
            # writing a 22 KB file that loaded with ERR_FILE_NOT_FOUND
            # on every sprite. The user only enabled /wait AFTER iter 3
            # had also failed, so two more iters were wasted before the
            # human could intervene. The natural intervention point is
            # the very first non-clean iter — that's where the model
            # most needs course correction and where it's cheapest to
            # provide. Self-arms exactly once per session; user can
            # /wait off to opt out for the rest of the run.
            if (
                self._auto_step_on_failure
                and
                not self._auto_step_armed
                and not report.get("ok", False)
                and not self._step_mode
                and not getattr(self, "_step_auto_disabled", False)
            ):
                self._step_mode = True
                self._auto_step_armed = True
                self._trace({
                    "kind": "step_mode_set",
                    "on": True,
                    "reason": "auto_armed_on_first_failure",
                    "iteration": iteration,
                })
                yield self._record(AgentEvent(
                    "info",
                    f"[yellow]step-mode auto-armed[/yellow] — iter {iteration} "
                    "test failed. The agent will pause after each iter so "
                    "you can intervene before more iters are spent. Type "
                    "`/wait off` to disable, or press Enter at the next "
                    "pause to continue."
                ))

            # Queue screenshot bytes for VLM attachment on next turn.
            # ALSO captured unconditionally (independent of `_is_vlm`) so
            # the vision-progress judge can compare current vs. previous
            # even when the building model is text-only.
            after_bytes: bytes | None = None
            after_path: str | None = None
            if shot_path is not None and report.get("screenshot"):
                after_path = str(report["screenshot"])
                try:
                    after_bytes = Path(after_path).read_bytes()
                except Exception:
                    after_bytes = None
                if self._is_vlm is not False:
                    self._next_image_bytes = after_bytes
                    self._last_screenshot_after = after_bytes
                    self._last_screenshot_after_path = after_path if after_bytes else None
                elif after_bytes is None:
                    self._next_image_bytes = None
                    self._last_screenshot_after = None
                    self._last_screenshot_after_path = None

            before_bytes: bytes | None = None
            before_path: str | None = None
            if shot_path is not None and report.get("screenshot_before"):
                before_path = str(report["screenshot_before"])
                try:
                    before_bytes = Path(before_path).read_bytes()
                except Exception:
                    before_bytes = None

            # Action frame: the harness captured this at the moment a held key
            # produced its largest canvas change (game mid-ACTION, not at
            # rest). Handed to the visual critic as a 3rd image so it can judge
            # whether a deliberate action animation actually renders.
            action_bytes: bytes | None = None
            if shot_path is not None and report.get("screenshot_action"):
                try:
                    action_bytes = Path(
                        str(report["screenshot_action"])
                    ).read_bytes()
                except Exception:
                    action_bytes = None

            # Compute the screenshot delta BEFORE the vision judge runs
            # — the judge rotates `_prev_judge_png` to the current frame
            # at the end of its call, so we need to read against the
            # prior frame here. Delta is mean per-pixel RGB diff in
            # [0, 1]; >0.15 means a substantial visual change. The
            # judge uses this to detect visible regressions (high delta
            # paired with "no progress" verdict).
            if after_bytes is not None:
                try:
                    from tools import screenshot_delta as _sshot_delta
                    self._last_screenshot_delta = _sshot_delta(
                        self._prev_judge_png, after_bytes,
                    )
                except Exception:
                    self._last_screenshot_delta = None
            if before_bytes is not None and after_bytes is not None:
                try:
                    from tools import screenshot_delta as _sshot_delta
                    input_playtest_delta = _sshot_delta(before_bytes, after_bytes)
                except Exception:
                    input_playtest_delta = None
                if (
                    input_playtest_delta is not None
                    and input_playtest_delta < 0.005
                ):
                    self._pending_coaching.append(
                        "STATIC SCREEN WARNING: The screen was completely "
                        "static before and after input simulation "
                        f"(pixel delta {input_playtest_delta:.4f}). This "
                        "usually means controls are unwired, state is not "
                        "updating, or the RAF loop is drawing the same frame."
                    )
                    self._trace({
                        "kind": "static_screen_warning",
                        "iteration": iteration,
                        "input_playtest_delta": input_playtest_delta,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"[yellow]warning[/yellow] (iter {iteration}): "
                        "static screen detected after simulated input "
                        f"(delta {input_playtest_delta:.4f})"
                    ))

            input_test = report.get("input_test") or {}
            if isinstance(input_test, dict) and input_test.get("ran"):
                evidence = input_test.get("responsive_evidence") or {}
                if isinstance(evidence, dict) and evidence:
                    movement_fields: list[str] = []
                    input_buffer_fields: list[str] = []
                    other_fields: list[str] = []
                    movement_exact = {"x", "y", "px", "py", "col", "row"}
                    movement_names = ("pos", "position", "coord")
                    input_exact = {"dir", "nextdir", "keys", "pressed", "input"}
                    input_names = ("lastfire", "lastinput", "key", "press")
                    for changed_fields in evidence.values():
                        if not isinstance(changed_fields, list):
                            continue
                        for field in changed_fields:
                            field_text = str(field)
                            field_low = field_text.lower()
                            leaf = field_low.rsplit(".", 1)[-1]
                            if (
                                leaf in movement_exact
                                or any(name in leaf for name in movement_names)
                            ):
                                movement_fields.append(field_text)
                            if (
                                leaf in input_exact
                                or any(name in leaf for name in input_names)
                            ):
                                input_buffer_fields.append(field_text)
                            elif not (
                                leaf in movement_exact
                                or any(name in leaf for name in movement_names)
                            ):
                                other_fields.append(field_text)
                    if (
                        input_buffer_fields
                        and not movement_fields
                        and not other_fields
                    ):
                        self._pending_coaching.append(
                            "STATE LOCOMOTION WARNING: Input simulation changed "
                            "input-buffer/state fields but did NOT change any "
                            "player coordinate fields (x, y, px, py, row/col, "
                            "position). This suggests the controllable player "
                            "is trapped or movement logic is not wired through, "
                            "for example by a grid-alignment snap or collision "
                            "check that cancels movement every frame."
                        )
                        self._trace({
                            "kind": "state_locomotion_warning",
                            "iteration": iteration,
                            "input_buffer_fields": input_buffer_fields[:8],
                        })
                        yield self._record(AgentEvent(
                            "info",
                            f"[yellow]warning[/yellow] (iter {iteration}): "
                            "input changed buffers but no player coordinates"
                        ))
            # Per-iter screenshot review — one toggle (/vlm-critique), ONE
            # structured critic path. get_backend('critic') resolves the model:
            # a staged critic slot (model2/3), else the coder itself when it can
            # see (VLM), else /allroles coder slot. None → no VLM available, skip
            # (deterministic probes carry it). Cloud review stays explicit via
            # /check in chat.py.
            if after_bytes is not None and getattr(self, "_use_vlm_critique", False):
                critic_backend = self.get_backend("critic")
                # Fast-path: if the user has already requested ship (Ctrl+D),
                # skip the post-iter visual critic. The critic is advisory
                # — it generates coaching for a hypothetical NEXT iter that
                # is never going to run. Without this skip, a force-done
                # waits the full critic duration (~20–300 s in observed
                # traces) before the iter-boundary check at the top of
                # the loop can fire. General behavior, not a per-genre
                # heuristic.
                if critic_backend is not None and self._user_force_done:
                    yield self._record(AgentEvent(
                        "info",
                        "[dim]skipping visual critic — user requested ship[/dim]",
                    ))
                    self._trace({
                        "kind": "critic_skipped_for_force_done",
                        "iteration": iteration,
                    })
                    critic_backend = None
                if critic_backend is not None:
                    critic_bk = self.get_backend("critic")
                    if critic_bk is getattr(self, "_backend3", None):
                        vc_role = getattr(self, "_model3_role", None) or "critic"
                    elif critic_bk is getattr(self, "_backend2", None):
                        vc_role = getattr(self, "_model2_role", None) or "critic"
                    else:
                        vc_role = "critic"
                    # Phase 1A — when the critic is on a separate slot
                    # from the coder, spawn it as a background task so its
                    # compute overlaps with iter N+1's coder stream. The
                    # coaching lands in `_pending_coaching` whenever the
                    # task completes (one-turn lag at worst) and is
                    # drained at the next user-turn boundary. On
                    # single-slot configs (critic backend == coder
                    # backend) we await inline — concurrent runs would
                    # just queue at the daemon, no benefit.
                    if self._critic_runs_on_independent_slot(critic_backend):
                        # If a previous critic task is still in flight,
                        # let it finish first (or drain non-blocking).
                        # We don't pile up tasks — one critic at a time.
                        await self._drain_pending_critic_task(wait=False)
                        yield self._record(AgentEvent(
                            "info",
                            f"[dim]visual critic spawned on slot ({vc_role}) — "
                            f"runs in parallel with iter {iteration + 1}[/dim]",
                        ))
                        self._trace({
                            "kind": "visual_critic_spawned_concurrent",
                            "iteration": iteration,
                            "vc_role": vc_role,
                        })
                        try:
                            self._critic_task = asyncio.create_task(
                                self._spawn_visual_critic(
                                    after_bytes, before_bytes, iteration, vc_role,
                                    action_bytes=action_bytes,
                                )
                            )
                        except Exception as exc:
                            self._trace({
                                "kind": "visual_critic_spawn_error",
                                "iteration": iteration,
                                "error": str(exc)[:240],
                            })
                    else:
                        # Blocking inline path — preserved for single-slot
                        # / single-GPU configurations.
                        try:
                            yield self._record(AgentEvent(
                                "activity",
                                "streaming",
                                {"label": "visual critic", "role": vc_role},
                            ))
                            critique = await self.run_visual_critic(
                                after_bytes,
                                before_bytes,
                                action_png=action_bytes,
                            )
                            yield self._record(self._activity_idle_event(vc_role))
                            if not critique:
                                await self._run_structured_local_vlm_critique(
                                    after_bytes, iteration,
                                )
                            elif critique:
                                cleaned = critique.strip()
                                if cleaned and "ok" not in cleaned.lower()[:30]:
                                    queued = self._queue_visual_critic_coaching(
                                        cleaned, iteration=iteration, vc_role=vc_role,
                                    )
                                    if queued:
                                        yield self._record(AgentEvent(
                                            "info",
                                            f"[magenta]visual critic[/magenta] (iter {iteration}): {cleaned}"
                                        ))
                                    else:
                                        yield self._record(AgentEvent(
                                            "info",
                                            f"[dim]visual critic (iter {iteration}): same observation as a recent turn — suppressed[/dim]"
                                        ))
                        except Exception as exc:
                            self._trace({
                                "kind": "visual_critic_error",
                                "iteration": iteration,
                                "error": str(exc),
                            })
                else:
                    # No staged critic slot — fall back to in-process mlx_vlm
                    # checklist when discoverable (e.g. mlx-server strips images).
                    if not self._user_force_done:
                        handled = await self._run_structured_local_vlm_critique(
                            after_bytes, iteration,
                        )
                        if not handled:
                            self._trace({
                                "kind": "visual_critic_skipped",
                                "iteration": iteration,
                                "reason": "no_vlm_backend_available",
                            })
            # Point-and-click: compare declared hotspot rects to VLM
            # grounding from generated bg PNGs.
            if getattr(self, "_use_vlm_critique", False):
                try:
                    await self._verify_pointclick_hotspots_vs_grounding(
                        iteration=iteration,
                    )
                except Exception as exc:
                    self._trace({
                        "kind": "pointclick_grounding_verify_error",
                        "iteration": iteration,
                        "error": str(exc)[:200],
                    })
            try:
                async for _ob_ev in self._run_opening_book_sidecars(report, iteration):
                    yield _ob_ev
            except Exception as exc:
                self._trace({
                    "kind": "opening_book_sidecars_failed",
                    "iteration": iteration,
                    "error": str(exc),
                })
            # Phase 1.5 — autonomous self-feedback. Only fires on clean
            # iters (probes_ok) and respects the /playtest toggle + the
            # budget governor + the Ctrl+D fast-path. Findings flow into
            # _pending_feedback so Phase 0.1's partitioner handles them
            # like real user feedback.
            try:
                async for _af_ev in self._run_autonomous_playtest(iteration, report):
                    yield _af_ev
            except Exception as exc:
                self._trace({
                    "kind": "autonomous_playtest_error",
                    "iteration": iteration,
                    "error": str(exc)[:240],
                })
            if (
                self._use_double_screenshot
                and before_path is not None
                and before_bytes is not None
            ):
                try:
                    self._last_screenshot_before = before_bytes
                    self._last_screenshot_before_path = before_path
                except Exception:
                    self._last_screenshot_before = None
                    self._last_screenshot_before_path = None

            # Stop-Losing-To-OneShot todo #4 — auto-revert on regression.
            # Mid-tier models often "polish" a working build into a worse
            # one. When the previous iter passed but this iter introduced
            # NEW page errors, FEWER probes passing, or NEW criteria-
            # coverage gaps (relative to the last working report), drop
            # this iter's HTML, restore .best.html on disk, and ask the
            # model for a minimal patch. The revert grants one bonus iter
            # (capped) so the user's max_iters isn't punished by the
            # rollback. Generic and behavioral — operates only on harness
            # signals, no genre awareness needed.
            prev = self._previous_report or {}
            prev_ok = self._previous_report_ok is True
            current_ok = bool(report.get("ok"))
            # Polish phase (item 2): consume the "this report answers a
            # polish turn" flag. A polish turn that regressed the green
            # build ends the polish phase for the session — the revert
            # paths below restore the file; we just stop polishing.
            was_polish_turn = bool(self._polish_pending)
            self._polish_pending = False
            if was_polish_turn and not current_ok:
                self._trace({
                    "kind": "polish_regression_revert",
                    "iteration": iteration,
                    "polish_turns_used_before_stop": self._polish_turns_used,
                })
                self._polish_turns_used = _POLISH_TURN_CAP
            if (
                prev_ok
                and not current_ok
                and self._iter_budget_bonus < revert_bonus_cap
            ):
                prev_probes = prev.get("probes") or []
                cur_probes = report.get("probes") or []
                prev_passing = sum(1 for p in prev_probes if p.get("ok"))
                cur_passing = sum(1 for p in cur_probes if p.get("ok"))
                new_page_errors = (
                    len(report.get("page_errors") or [])
                    > len(prev.get("page_errors") or [])
                )
                fewer_probes = (cur_passing < prev_passing)
                new_coverage_gaps = (
                    bool(report.get("criteria_uncovered"))
                    and not bool(prev.get("criteria_uncovered"))
                )
                if new_page_errors or fewer_probes or new_coverage_gaps:
                    best_html = self._read_best_or_empty()
                    if best_html:
                        try:
                            self.out_path.write_text(best_html, encoding="utf-8")
                        except Exception:
                            pass
                        self._current_file = best_html
                        self._iter_budget_bonus += 1
                        problems: list[str] = []
                        if new_page_errors:
                            problems.append("new uncaught page errors")
                        if fewer_probes:
                            problems.append(
                                f"only {cur_passing}/{len(cur_probes)} probes pass "
                                f"now (was {prev_passing}/{len(prev_probes)})"
                            )
                        if new_coverage_gaps:
                            problems.append(
                                "a previously-covered criterion is no longer covered"
                            )
                        problems_str = "; ".join(problems)
                        self._trace({
                            "kind": "auto_revert",
                            "iteration": iteration,
                            "problems": problems,
                            "bonus_used": self._iter_budget_bonus,
                            "bonus_cap": revert_bonus_cap,
                        })
                        yield self._record(AgentEvent(
                            "info",
                            f"REGRESSION on iter {iteration}: {problems_str} — "
                            f"auto-reverted to last working file. "
                            f"Bonus iter granted (count: {self._iter_budget_bonus}).",
                            {
                                "iter": iteration,
                                "problems": problems,
                                "bonus_used": self._iter_budget_bonus,
                                "bonus_cap": revert_bonus_cap,
                            },
                        ))
                        # Polish phase (item 2): a regressing polish turn
                        # ends polish — re-offer <done/> instead of asking
                        # for another change.
                        if was_polish_turn:
                            revert_msg = (
                                "REGRESSION DETECTED: your polish change degraded "
                                f"the working build ({problems_str}). The harness "
                                "has auto-reverted the file on disk to the previous "
                                "working version. The polish phase is over — send "
                                "<done/> to ship the working version as-is."
                            )
                        else:
                            revert_msg = (
                                "REGRESSION DETECTED: your last change degraded the "
                                f"working build ({problems_str}). The harness has "
                                "auto-reverted the file on disk to the previous "
                                "working version. Send a MINIMAL <patch> that "
                                "addresses only the original feedback without "
                                "breaking what already worked. If you cannot make a "
                                "small change without regressing, send <done/> to "
                                "ship the working version as-is."
                            )
                        self._messages.append({
                            "role": "user",
                            "content": self._flush_user_injections(revert_msg),
                        })
                        self._fix_mode = True
                        # Skip streak/playbook/save_best below so revert state
                        # is identical to "this iter didn't happen for streak
                        # purposes". _previous_report* stays unchanged.
                        continue

            # Non-clean visual regression guard: the prior iteration may not
            # have been clean enough for best.html, but it can still be a
            # better baseline. Trace 20260612_225857 peaked visually before a
            # later patch reintroduced PROCEDURAL_REGRESSION/pink placeholders.
            _iter_has_crash = bool(report.get("page_errors")) or any(
                tok in str(e).lower()
                for e in (report.get("errors") or [])
                for tok in ("referenceerror", "typeerror", "syntaxerror")
            )
            if (
                not prev_ok
                and prev
                and _pre_iter_html
                and not current_ok
                and self._iter_budget_bonus < revert_bonus_cap
                and not _iter_has_crash
            ):
                prev_probes = prev.get("probes") or []
                cur_probes = report.get("probes") or []
                prev_passing = sum(1 for p in prev_probes if p.get("ok"))
                cur_passing = sum(1 for p in cur_probes if p.get("ok"))
                visual_terms = (
                    "PROCEDURAL_REGRESSION_SUSPECTED",
                    "ASSETS_LOADED_BUT_UNDRAWN",
                    "MISSING",
                    "pink",
                )
                prev_soft = "\n".join(str(w) for w in prev.get("soft_warnings") or [])
                cur_soft = "\n".join(str(w) for w in report.get("soft_warnings") or [])
                added_visual_warning = any(
                    t in cur_soft and t not in prev_soft for t in visual_terms
                )
                if added_visual_warning and cur_passing <= prev_passing:
                    try:
                        self.out_path.write_text(_pre_iter_html, encoding="utf-8")
                    except Exception:
                        pass
                    self._current_file = _pre_iter_html
                    self._iter_budget_bonus += 1
                    self._trace({
                        "kind": "visual_regression_snapshot_revert",
                        "iteration": iteration,
                        "cur_passing": cur_passing,
                        "prev_passing": prev_passing,
                        "bonus_used": self._iter_budget_bonus,
                    })
                    yield self._record(AgentEvent(
                        "info",
                        "VISUAL REGRESSION: last patch did not improve probes "
                        "and added procedural/missing-asset warnings. Restored "
                        "the previous snapshot and granted a bonus iter.",
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "VISUAL REGRESSION DETECTED: the harness restored "
                            "the previous, better snapshot. Your last patch did "
                            "not improve probe count and added procedural/missing "
                            "asset warnings. Send one minimal <patch> that fixes "
                            "the user-visible issue without reintroducing large "
                            "procedural placeholders or pink MISSING boxes."
                        ),
                    })
                    self._fix_mode = True
                    continue

            said_done = bool(_DONE_RE.search(reply))
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Track stuck-streak — used by v1's fix prompt to switch to
            # the "5-7 different sources" reflection ladder after repeat
            # failures on the same goal.
            if report["ok"]:
                self._stuck_streak = 0
                # Skipped browser runs (eval_seed_edits) are not verified
                # gameplay — do not advance the clean-streak gate.
                if not report.get("test_skipped"):
                    self._consecutive_clean_iters += 1
                # Todo-driven execution: telemetry-only drift check on
                # the CURRENT TASK contract (never a cutoff or retry).
                self._todo_drift_check()
            else:
                self._stuck_streak += 1
                self._consecutive_clean_iters = 0
                # Failed iter: the contract task wasn't completed because
                # the build broke — drop it so a later clean iter doesn't
                # report stale drift. It stays open in _todos_items and
                # will be re-selected once the build is clean again.
                self._current_todo = None
            self._trace({
                "kind": "streak_update",
                "consecutive_clean_iters": self._consecutive_clean_iters,
                "stuck_streak": self._stuck_streak,
                "min_to_ship": self._min_clean_streak_to_ship,
            })
            yield self._record(AgentEvent(
                "streak", "",
                {
                    "consecutive_clean_iters": self._consecutive_clean_iters,
                    "stuck_streak": self._stuck_streak,
                    "min_to_ship": self._min_clean_streak_to_ship,
                },
            ))

            # Online playbook counter feedback. Off by default (tune
            # baselines should not write back). When on:
            #   - pass + active bullets → helpful++ for each
            #   - 3rd+ consecutive failure + active bullets → harmful++
            if self._playbook_writeback and self._active_bullet_ids:
                ids = list(self._active_bullet_ids)
                delta_label: str | None = None
                if report["ok"]:
                    self._playbook.update_counters(ids, helpful_delta=1)
                    delta_label = "helpful+1"
                elif self._stuck_streak >= 3 and self._failure_blames_code(report):
                    self._playbook.update_counters(ids, harmful_delta=1)
                    delta_label = "harmful+1"
                if delta_label is not None:
                    self._trace({
                        "kind": "playbook_writeback",
                        "ids": ids,
                        "delta": delta_label,
                    })
                    # Surface to the TUI/log too — the user explicitly
                    # asked to see what's happening to bullet scores.
                    # Without this the only signal is in the JSONL
                    # trace, which they have no reason to grep.
                    yield self._record(AgentEvent(
                        "memory",
                        f"playbook {delta_label}: "
                        + ", ".join(ids[:5])
                        + (f" (+{len(ids)-5} more)" if len(ids) > 5 else ""),
                        {"delta": delta_label, "ids": ids},
                    ))

            # Save best.html on every clean iteration AND record the success
            # in memory so future similar goals can retrieve this code.
            if report["ok"]:
                best = self._save_best(new_html)
                if best is not None:
                    yield self._record(AgentEvent(
                        "info", f"saved working version to {best}"
                    ))
            elif _report_green_except_cosmetic_sprites(report):
                # Trace 20260612_171752: ok stayed False on cosmetic sprite
                # findings alone (ACTION_DRAWN_NOT_SPRITED) for an entire
                # session, so a 7/7-probes playable build was never saved —
                # best_exists=False and continuation turns had no revert
                # anchor. Behaviorally green (all probes pass, zero errors)
                # + only cosmetic-sprite soft_warnings → save best.html
                # anyway; the warnings still reach the model unchanged.
                best = self._save_best(new_html)
                if best is not None:
                    self._trace({
                        "kind": "best_saved_cosmetic_only",
                        "soft_warnings": [
                            w[:80] for w in report.get("soft_warnings") or []
                        ],
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"saved best.html (all probes pass, zero errors; "
                        f"only cosmetic sprite warnings gate ok) to {best}",
                    ))
                # If the previous turn had a diagnose, we now know that
                # diagnosis led to a good fix — record it as a winning
                # mistake/fix pair for next time.
                if self._last_diagnose:
                    self._memory.record_mistake(
                        signature_for_report(report),  # current sig (often empty since OK)
                        f"diagnosis worked: {self._last_diagnose[:300]}",
                    )
                    self._last_diagnose = None

            # ---- USER force-done shortcut ------------------------------
            # Ctrl+D wins unconditionally: ship the current passing build.
            # Earlier behavior tried to apply autonomous/typed feedback that
            # had landed concurrently with the ship request, which (a)
            # contradicted the user's explicit intent and (b) re-entered the
            # iter loop while _stop_event was still set from request_done(),
            # causing the next stream to bail at 0.0s with no tokens.
            if self._user_force_done and report["ok"]:
                dropped = len(self._pending_feedback) + (1 if self._pending_answer else 0)
                if dropped:
                    self._pending_feedback.clear()
                    self._pending_answer = None
                    yield self._record(AgentEvent(
                        "info",
                        f"[dim]shipping per Ctrl+D — dropping {dropped} queued "
                        f"feedback item(s) (re-send after ship if still wanted)[/dim]",
                    ))
                yield self._record(AgentEvent("done", "User confirmed - shipping current build."))
                self._record_session_outcome(ok=True)
                return

            if self._user_force_done and not report["ok"]:
                # Hard stop: probes failed but the user asked to ship.
                # Don't loop another fix turn — exit with whatever we have.
                # best.html is shipped if it exists (a prior iter passed);
                # otherwise the current new_html is on disk at out_path.
                async for ev in self._final_iter_test_if_needed():
                    yield ev
                best_exists = self.best_path.exists()
                yield self._record(AgentEvent(
                    "info",
                    "user requested ship with failing probes - exiting with current build",
                ))
                self._record_session_outcome(ok=best_exists)
                yield self._record(AgentEvent(
                    "done",
                    "User-requested ship (probes failed).",
                    {"best_exists": best_exists, "report_ok": False},
                ))
                return

            # ---- self-critique on first clean+done ---------------------
            if report["ok"] and said_done and not self._awaiting_confirm:
                yield self._record(AgentEvent("phase", "self-critique"))
                self._awaiting_confirm = True
                notice = self._consumed_feedback_summary()
                if notice:
                    yield self._record(AgentEvent("info", notice))
                # Critique always uses single-sample (we want the model's
                # honest call, not a vote). When VLM-critique feature is
                # on AND we have an "after" screenshot, attach it AND
                # append a visual review note so confirm_done is gated on
                # actually seeing the rendered game.
                critique_msg = self._p.CRITIQUE_INSTRUCTION
                if (
                    self._use_vlm_critique
                    and self._is_vlm
                    and self._last_screenshot_after
                ):
                    critique_msg = critique_msg + (
                        "\n\nA SCREENSHOT of the final game state is "
                        "attached. Look at it. In <notes>, name one "
                        "concrete visual thing you SEE (HUD position, "
                        "ship visible, score legible). If anything looks "
                        "broken — ship off-canvas, score not visible, "
                        "modal blocking gameplay — that IS a crash-class "
                        "bug for the player; send a <patch>. Otherwise "
                        "<confirm_done/>."
                    )
                    self._next_image_bytes = self._last_screenshot_after
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(critique_msg),
                })
                self._previous_report_ok = report["ok"]
                self._previous_report = report  # todo #3 — full report
                self._pending_scoped_check_keywords = []
                self._fix_mode = False
                continue

            if self._awaiting_confirm:
                self._awaiting_confirm = False

            # ---- record mistake on regression --------------------------
            # We can't tell what the right fix is yet, so we record the
            # signature only — the next clean turn will pair it with the
            # diagnosis that fixed it (above).
            if not report["ok"]:
                sig = signature_for_report(report)
                if sig:
                    self._trace({"kind": "mistake_signature", "sig": sig})
                    # A5: when the same error signature repeats, the model
                    # is patching the wrong location. Nudge it to author a
                    # runtime-state probe so the next test report SHOWS
                    # the data it needs instead of the model guessing.
                    if sig == self._last_mistake_sig:
                        self._repeat_sig_streak += 1
                    else:
                        self._repeat_sig_streak = 1
                    self._last_mistake_sig = sig
                    if self._repeat_sig_streak >= 2:
                        # Tailor the coaching to the failure shape. For
                        # asset-load signatures the runtime-probe advice
                        # is irrelevant — the model needs to look at
                        # paths, not state. The DK trace 20260513_122154
                        # spent 3 iters patching drawImage because the
                        # generic coaching pointed it at probes when
                        # the bug was a missing file.
                        sig_low = sig.lower()
                        asset_hints = (
                            "err_file_not_found",
                            "failed to load resource",
                            "naturalwidth",
                            "broken state",
                            "broken' state",
                            "invalidstateerror",
                        )
                        if any(h in sig_low for h in asset_hints):
                            self._pending_coaching.append(
                                "Same asset-load failure for 2 iterations — "
                                "your patches keep guarding drawImage but the "
                                "Image never loaded. The browser said "
                                "net::ERR_FILE_NOT_FOUND or InvalidStateError "
                                "because the FILE on disk does not exist at "
                                "the path your code is requesting. Stop "
                                "adding try/catch or .complete checks. "
                                "Instead this turn: (a) compare every "
                                "ASSETS[name] reference against the "
                                "assets actually available in the GENERATED "
                                "ASSETS block above, and (b) for any name "
                                "that's referenced but not generated, emit "
                                "an `<assets>` block requesting it OR remove "
                                "the reference. The harness will regen the "
                                "missing names mid-session."
                            )
                        else:
                            # Subsystem-pointing coaching (DK 20260514_104131
                            # fix): when the signature implicates a specific
                            # code region (input wiring, frame loop, RAF
                            # kickoff), tell the model EXACTLY which code
                            # area to look at. 27B models follow directive
                            # ("edit the keydown handler") more reliably
                            # than abstract ("author a probe"). Generic
                            # fallback preserved when no hint matches.
                            sub_hint = _subsystem_hint(sig)
                            if sub_hint:
                                idents = ", ".join(
                                    f"`{i}`" for i in sub_hint["identifiers"][:6]
                                )
                                self._pending_coaching.append(
                                    f"Same {sub_hint['name'].upper()} "
                                    f"failure for {self._repeat_sig_streak} "
                                    "iterations — the harness keeps "
                                    "reporting the same broken subsystem and "
                                    "your patches keep targeting unrelated "
                                    "code. Signature: "
                                    f"'{sig[:140]}'. The implicated region "
                                    f"matches identifiers: {idents}. THIS "
                                    "TURN: emit a <patch> whose SEARCH "
                                    f"targets {sub_hint['fix_phrase']}, OR "
                                    "a focused <html_file> rewriting only "
                                    "that subsystem. Stop patching the "
                                    "higher-level mechanic — the bug is "
                                    "upstream of where you've been editing."
                                )
                                self._trace({
                                    "kind": "subsystem_hint_coaching",
                                    "subsystem": sub_hint["name"],
                                    "streak": self._repeat_sig_streak,
                                })
                                # Stuck-loop HARD GATE (Item 2, plan
                                # 20260514_175012): at streak >= 3 the
                                # model has had two iterations of
                                # directive coaching and ignored it.
                                # Force a <question> to the user this
                                # turn — block any other output format.
                                # Pulling the human in is the most
                                # reliable way to break the loop when
                                # the model can't translate the
                                # subsystem signal into a real fix.
                                if self._repeat_sig_streak >= 3:
                                    self._force_question_subsystem = sub_hint
                                    self._trace({
                                        "kind": "stuck_hard_gate_armed",
                                        "subsystem": sub_hint["name"],
                                        "streak": self._repeat_sig_streak,
                                    })
                            else:
                                self._pending_coaching.append(
                                    "Same error signature for 2 iterations — your "
                                    "patches are missing the real cause. In this "
                                    "turn, AUTHOR a runtime-state probe inside "
                                    "<probes>...</probes> that captures the data "
                                    "you're missing (e.g. "
                                    "`name=\"alien_bullets_len\", expr=\"window.state && state.alienBullets.length\"` "
                                    "or `JSON.stringify(state.alienBullets.slice(0,3))`). "
                                    "The next test report will include the probe's "
                                    "value — letting you see runtime state directly "
                                    "instead of guessing from the stack trace."
                                )
            else:
                self._last_mistake_sig = None
                self._repeat_sig_streak = 0
                # A clean iter dissolves any armed hard-gate (the model
                # made progress on its own; no need to force a question).
                self._force_question_subsystem = None

            # ---- build next user turn ---------------------------------
            # LLM router (chess-trace fix): route any feedback the user typed
            # during this iter's stream/test BEFORE the next-turn assembly
            # consumes it, so deferral / asset decisions see the decision.
            await self._precompute_feedback_route()
            notice = self._consumed_feedback_summary()
            if notice:
                yield self._record(AgentEvent("info", notice))

            next_user = self._build_fix_prompt(
                report=report, regressed=regressed, partial_failed=partial_failed,
            )
            # Item 3 (context discipline): wrap the report turn in collapse
            # sentinels so it shrinks to a 3-line digest once superseded.
            next_user = self._wrap_report_block(next_user, report)
            # Adaptive temperature: failed → low (precision). Clean+keep-going
            # path goes through post_clean which says "prefer done".
            self._fix_mode = not report["ok"]
            # Track D: a real gameplay/code bug the harness missed (clean
            # report) still needs a precision fix turn — let the feedback
            # router force fix_mode on so the next turn diagnoses-then-fixes
            # instead of treating the bug report as cosmetic polish.
            self._route_forces_fix_mode()
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(next_user),
            })
            self._previous_report_ok = report["ok"]
            self._previous_report = report  # todo #3 — full report
            self._pending_scoped_check_keywords = []

        # ---- iteration cap reached ------------------------------------
        if self.has_pending_user_input():
            yield self._record(AgentEvent(
                "info", "Iteration cap reached but user feedback pending - one extra turn.",
            ))
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(
                    "Iteration cap was reached but the user sent the feedback above. "
                    "One last turn: address it with a <patch> against the current file."
                ),
            })
            yield self._record(AgentEvent(
                "activity", "streaming",
                {"label": "bonus turn (cap reached)", "role": "coder"},
            ))
            try:
                reply = await self._stream(self._token_cb_wrapper)
                for _ev in self._drain_stream_ui_events():
                    yield _ev
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({"role": "assistant", "content": reply})
                self._extract_and_queue_lookups(reply)
                self._dump_conversation()
                self._trace({
                    "kind": "assistant_reply",
                    "iteration": self.max_iters + 1,
                    "len": len(reply),
                    # One-complete-trace (2026-06-14): full reply text (was
                    # len-only) so the .jsonl carries the bonus-turn output.
                    "reply": reply,
                })
                violation = self._scoped_reply_violation(reply)
                # Fix B (seed-edit escape hatch), bonus turn: same forward-
                # progress guarantee as the main loop — after 2 consecutive
                # violations with usable code on a seed-edit lock, clear the
                # lock and apply instead of rejecting again. Mid-session
                # locks (no is_seed_edit) keep today's reject-and-retry.
                if violation:
                    cfg = self._scoped_constraints or {}
                    self._scoped_violation_streak += 1
                    has_usable_code = bool(extract_patches(reply)) or (
                        self._extract_html(reply) is not None
                    )
                    if (
                        cfg.get("is_seed_edit")
                        and self._scoped_violation_streak >= 2
                        and has_usable_code
                    ):
                        self._trace({
                            "kind": "scoped_escape_hatch_used",
                            "iteration": self.max_iters + 1,
                            "violation": violation,
                            "streak": self._scoped_violation_streak,
                        })
                        self._clear_scoped_constraints()
                        violation = None  # fall through to materialize below
                if violation:
                    yield self._record(AgentEvent(
                        "info",
                        f"scoped guard (bonus turn): {violation}",
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            self._scoped_retry_instruction(violation)
                        ),
                    })
                else:
                    # Mid-session asset/sound re-parse also applies on the
                    # bonus turn — the user's feedback that triggered this
                    # turn often says "add sprites" and the model's reply
                    # may include a fresh <assets> block.
                    async for ev in self._maybe_generate_assets_and_sounds(
                        reply, trigger="mid_session",
                    ):
                        yield ev
                    new_html, materialize_msg = await self._materialize(reply)
                    if new_html is not None:
                        self.out_path.write_text(new_html, encoding="utf-8")
                        self._current_file = new_html
                        self._scoped_violation_streak = 0  # Fix B: progress, reset
                        self._save_snapshot(new_html)
                        yield self._record(AgentEvent(
                            "code", str(self.out_path),
                            {"size": len(new_html), "materialize": materialize_msg},
                        ))
            except Exception as e:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent("error", f"Final feedback turn failed: {e}"))


    async def _run_exit_and_finalize(self) -> AsyncIterator[AgentEvent]:
        """Exit-decision turn, final test, session outcome."""
        # ---- Item 5: exit-decision turn before silent loop end ------
        # DK trace 20260514_175012 ended with patches emitted but no
        # <done/> / <confirm_done/> — the user got back a half-fixed
        # game and no clear signal whether the agent had given up or
        # was waiting. Force one final ship-or-ask decision when:
        #   - last test failed
        #   - self._awaiting_confirm is False (no in-flight done/confirm cycle)
        #   - no pending user feedback (the bonus-turn branch above
        #     already handled that case)
        #   - user didn't force-ship
        if (
            self._previous_report_ok is False
            and not self._awaiting_confirm
            and not self.has_pending_user_input()
            and not self._user_force_done
        ):
            yield self._record(AgentEvent(
                "info",
                "iter cap reached with failing build — asking the "
                "model to ship-or-ask before exiting silently.",
            ))
            # Phase 3: tell the model NOT to ship if the on-disk file is
            # structurally broken. Trace 1 (chess 20260522_000304) ended
            # with a confident <done/> + <notes> over a file that wouldn't
            # even parse. The post-done verification below also catches
            # this, but warning the model up front avoids the wasted turn.
            broken_now = None
            try:
                broken_now = _baseline_structurally_broken(
                    self._current_file or ""
                )
            except Exception:
                broken_now = None
            ship_warning = ""
            if broken_now:
                ship_warning = (
                    "\nWARNING: micro-probes report the on-disk file is "
                    f"structurally broken ({broken_now[:160]}). "
                    "Do NOT use <done/> over a broken file — the harness "
                    "will record the session as failed. Use <question> "
                    "to ask the user to narrow scope, OR ship anyway "
                    "with explicit acknowledgement in <notes> that the "
                    "file does not run.\n"
                )
            last_report_facts = ""
            try:
                lr = self._previous_report or {}
                probes = lr.get("probes") or []
                failed_probe_names = [
                    str(p.get("name") or "probe")
                    for p in probes
                    if isinstance(p, dict) and not p.get("ok")
                ]
                facts = [
                    f"best_path_exists={self.best_path.exists()}",
                    f"ok={bool(lr.get('ok'))}",
                    f"failed_probes={failed_probe_names[:6]}",
                    f"page_errors={(lr.get('page_errors') or [])[:2]}",
                    f"soft_warnings={(lr.get('soft_warnings') or [])[:3]}",
                ]
                last_report_facts = (
                    "\nFACTUAL LAST REPORT — your <notes> MUST NOT "
                    "contradict or omit these facts:\n- "
                    + "\n- ".join(str(f) for f in facts)
                    + "\nIf best_path_exists=False, say no clean build was saved.\n"
                )
            except Exception:
                last_report_facts = ""
            exit_prompt = (
                "EXIT DECISION TURN — the iteration cap has been "
                "reached and the last test was not clean. THIS TURN "
                "you MUST emit EXACTLY ONE of the following:\n\n"
                "  1. <done/> followed by <notes>...</notes> — ship "
                "the current build as-is. Your <notes> should name "
                "(a) what works, (b) what's still broken, (c) any "
                "workaround the user can use. The harness will run "
                "one final verification against the file on disk and "
                "record the outcome. Use this when you've made some "
                "progress but can't fix everything in the remaining "
                "budget.\n\n"
                "  2. <question>...</question> — pause and ask the "
                "user a specific question. Use this when you "
                "genuinely don't know how to proceed and a one-line "
                "answer would unblock you. Be concrete; name the "
                "specific decision.\n\n"
                f"{ship_warning}"
                f"{last_report_facts}"
                "Do NOT emit <patch>, <html_file>, <plan>, "
                "<diagnose>, or any other tag this turn. The "
                "session ends after this reply."
            )
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(exit_prompt),
            })
            self._trace({"kind": "exit_decision_turn_prompted"})
            exit_role = self._planning_role()
            yield self._record(AgentEvent(
                "activity", "streaming",
                {"label": "exit decision", "role": exit_role},
            ))
            try:
                exit_reply = await self._stream(self._token_cb_wrapper, role=exit_role)
                for _ev in self._drain_stream_ui_events():
                    yield _ev
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({
                    "role": "assistant",
                    "content": exit_reply,
                    "model_role": exit_role,
                    "model_name": self.get_backend(exit_role).info.model
                })
                self._dump_conversation()
                self._trace({
                    "kind": "exit_decision_reply",
                    "len": len(exit_reply),
                    "preview": exit_reply[:300],
                })
                # Capture <notes> for the trace + UI; the actual ship
                # decision is reflected in the final-iter test below.
                exit_done_chosen = bool(_DONE_RE.search(exit_reply))
                if exit_done_chosen:
                    notes = self._extract_notes(exit_reply)
                    if notes:
                        yield self._record(AgentEvent(
                            "info",
                            f"exit notes (model's handoff summary): "
                            f"{notes[:400]}",
                        ))
                    self._trace({"kind": "exit_decision_done"})
                    # Phase 3: post-done structural verification. If the
                    # on-disk file fails micro-probes, override the model's
                    # ship-it claim with a hard error event AND mark the
                    # session as failed regardless of <notes>. Trace 1
                    # ended with confident notes over an unplayable file.
                    try:
                        on_disk = ""
                        if self.out_path.exists():
                            on_disk = self.out_path.read_text(encoding="utf-8")
                        broken_at_done = _baseline_structurally_broken(on_disk)
                    except Exception as e:
                        broken_at_done = f"could not read out_path: {e}"
                    if broken_at_done:
                        self._exit_done_over_broken_file = True
                        self._trace({
                            "kind": "exit_decision_done_over_broken_file",
                            "reason": (broken_at_done or "")[:200],
                        })
                        yield self._record(AgentEvent(
                            "error",
                            "shipped broken — file on disk fails "
                            f"structural checks ({broken_at_done[:160]}). "
                            "Session will be recorded as failed.",
                        ))
                # Handle <question>: surface to user, wait for one
                # answer, then exit. Skip when <done/> already chose ship
                # (run_11 Pac-Man 20260702: model mentioned both tags in
                # one reply → waited forever under --no-auto-step).
                # Unattended batch (--no-auto-step) must never block here.
                q = (
                    self._extract_question(exit_reply)
                    if not exit_done_chosen else None
                )
                if q is not None:
                    yield self._record(AgentEvent("question", q))
                    _exit_q_wait_s = 0.0
                    _exit_q_max_s = (
                        120.0 if self._auto_step_on_failure else 0.0
                    )
                    while (
                        self._pending_answer is None
                        and not self._user_force_done
                        and _exit_q_wait_s < _exit_q_max_s
                    ):
                        await asyncio.sleep(0.1)
                        _exit_q_wait_s += 0.1
                    if self._pending_answer is not None:
                        yield self._record(AgentEvent(
                            "info",
                            "user answered the exit question; "
                            "session ending — start /new with the "
                            "answer in mind to continue.",
                        ))
                        self._pending_answer = None  # consume
                    self._trace({"kind": "exit_decision_question"})
            except Exception as e:
                yield self._record(AgentEvent(
                    "activity", "idle",
                ))
                yield self._record(AgentEvent(
                    "error", f"exit-decision turn failed: {e}",
                ))

        yield self._record(AgentEvent(
            "info", f"reached max iterations ({self.max_iters}) - stopping"
        ))
        async for ev in self._final_iter_test_if_needed():
            yield ev
        # Outcome: ok if best.html exists (we passed at least once).
        # Phase 3: a <done/> over a structurally broken file flips this
        # to ok=False even when an earlier iter happened to ship a best.
        # The model's notes claimed success; the file says otherwise.
        ok_outcome = self.best_path.exists() and not self._exit_done_over_broken_file
        self._record_session_outcome(ok=ok_outcome)
        yield self._record(AgentEvent("done", "Iteration cap reached."))

    @classmethod
    def run_loop_inspect_source(cls) -> str:
        """Concatenate run-loop method sources (for tests that grep the loop body)."""
        import inspect
        return "\n".join(
            inspect.getsource(getattr(cls, name))
            for name in (
                "run",
                "_run_phase_a_and_first_build",
                "_run_build_iterate_loop",
                "_run_exit_and_finalize",
            )
        )

    @classmethod
    def class_inspect_source(cls) -> str:
        """GameAgent + mixin method sources (for regression guards on moved code)."""
        import inspect
        seen: set[str] = set()
        parts: list[str] = []
        try:
            parts.append(inspect.getsource(cls))
        except TypeError:
            pass
        for base in cls.__mro__:
            if base in (cls, object):
                continue
            for name, obj in base.__dict__.items():
                if name in seen:
                    continue
                if not (inspect.isfunction(obj) or inspect.ismethoddescriptor(obj)):
                    continue
                try:
                    parts.append(inspect.getsource(obj))
                    seen.add(name)
                except (TypeError, OSError):
                    pass
        return "\n".join(parts)

    async def run_with_restarts(
        self,
        goal: str,
        *,
        continuation: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Wrap run() with restart-N: if iter 1 of attempt k produces a
        score below `restart_score_threshold`, throw the session away
        and try again from scratch. Up to `restart_n` total attempts.
        Best-by-score wins.

        Mid-size LLMs (qwen-coder 32B, deepseek-coder 33B class) one-shot
        small games well; the agent's own multi-iter loop empirically
        regresses them. Restart-N leans into one-shot strength: rather
        than polish a bad start through 5 fix-turns, throw it away and
        try again.

        Z-Image asset cache (hash-keyed) is reused across restarts so we
        don't pay for sprite generation N times. Browser is reused.

        When restart_n=1 (default), this is a thin pass-through to run()
        and existing callers see no behavior change. continuation=True
        also passes through unchanged — restarts only make sense for
        fresh sessions.
        """
        if continuation or self.restart_n <= 1:
            self._restart_attempt_idx = 0
            self._restart_attempt_seed = None
            async for ev in self.run(goal, continuation=continuation):
                yield ev
            return

        attempts: list[tuple[float, int, Path]] = []  # (score, idx, snapshot_path)
        canonical_best = self.best_path
        for k in range(self.restart_n):
            if k > 0:
                self._reset_attempt_state()
                yield self._record(AgentEvent(
                    "phase", f"restart attempt {k+1}/{self.restart_n}",
                ))
            self._trace({
                "kind": "restart_attempt_start",
                "attempt_idx": k,
                "restart_n": self.restart_n,
            })
            self._restart_attempt_idx = k
            if k > 0:
                self._restart_attempt_seed = (
                    (self._restart_seed_base + (k * 7919)) & 0x7FFFFFFF
                ) or (k + 1)
                self._force_first_build_prefill = True
            else:
                self._restart_attempt_seed = None
                self._force_first_build_prefill = False
            self._trace({
                "kind": "restart_attempt_profile",
                "attempt_idx": k,
                "seed": self._restart_attempt_seed,
                "temp_bias": self._restart_temperature_bias(k),
                "force_first_build_prefill": self._force_first_build_prefill,
            })
            async for ev in self.run(goal):
                yield ev
            score = self._score_attempt()
            attempt_snap = canonical_best.with_name(
                f"{canonical_best.stem}.attempt_{k}.html"
            )
            try:
                if canonical_best.exists():
                    attempt_snap.write_text(
                        canonical_best.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    attempts.append((score, k, attempt_snap))
                else:
                    attempts.append((score, k, canonical_best))
            except Exception as e:
                self._trace_exception("restart_snapshot_failed", e)
                attempts.append((score, k, canonical_best))
            # Restart-signature-repeat escalation. Compute the dominant
            # failure shape of THIS attempt and compare to the previous
            # attempt's. When two attempts in a row hit the same shape,
            # the next attempt's plan_instruction is given
            # `force_minimal_first_build=True` so the model scopes
            # down rather than re-attempting the same ambitious build.
            # Universal: signature is derived from counters that fired
            # on observable failure events, no goal-text branching.
            current_signature = self._attempt_failure_signature(score=score)
            signature_repeat = (
                self._prev_attempt_signature is not None
                and current_signature == self._prev_attempt_signature
                and current_signature != "ok"
            )
            if signature_repeat:
                self._force_minimal_first_build = True
                self._trace({
                    "kind": "restart_signature_repeat",
                    "attempt_idx": k,
                    "signature": current_signature,
                    "score": score,
                    "hint": (
                        "Two consecutive attempts hit the same failure "
                        "shape; next attempt's plan_instruction will "
                        "demand a smaller intentionally-minimal first "
                        "build."
                    ),
                })
            else:
                # Same-signature streak broken — clear any prior force
                # so a future re-occurrence triggers fresh.
                self._force_minimal_first_build = False
            self._prev_attempt_signature = current_signature
            self._trace({
                "kind": "restart_attempt_end",
                "attempt_idx": k,
                "score": score,
                "signature": current_signature,
            })
            yield self._record(AgentEvent(
                "restart",
                f"restart attempt {k+1}/{self.restart_n}: score={score:.0f}",
                {
                    "attempt": k + 1,
                    "total": self.restart_n,
                    "score": score,
                    "winner": False,
                },
            ))
            if score >= 100.0:
                break
            if k == 0 and score >= self.restart_score_threshold:
                # iter-1 was close enough; prefer iterating in-place
                # (which we just did) over restarting from a clean slate.
                break

        if not attempts:
            return
        attempts.sort(key=lambda t: -t[0])
        best_score, best_idx, best_path = attempts[0]
        self._trace({
            "kind": "restart_winner",
            "winner_idx": best_idx,
            "winner_score": best_score,
            "all": [(s, i) for (s, i, _p) in attempts],
        })
        # Install winner as canonical best.html (it may already be the
        # current contents — this is a no-op in that case).
        try:
            if best_path != canonical_best and best_path.exists():
                canonical_best.write_text(
                    best_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        except Exception as e:
            self._trace_exception("restart_install_failed", e)
        yield self._record(AgentEvent(
            "restart",
            f"restart winner: attempt {best_idx+1} score={best_score:.0f}",
            {
                "attempt": best_idx + 1,
                "total": self.restart_n,
                "score": best_score,
                "winner": True,
                "all_scores": [
                    {"attempt": i + 1, "score": s}
                    for (s, i, _p) in attempts
                ],
            },
        ))

    def _attempt_failure_signature(self, *, score: float) -> str:
        """Reduce the per-attempt counters to a single short signature
        so the restart loop can detect "same shape twice."

        Priority order — pick the FIRST that fired:
          1. `dead_first_build` — file ran but RAF + input both dead.
          2. `identical_reply_loop` — model emitted byte-identical
             unparseable reply twice in a row.
          3. `format_rejection_iter1` — iter-1 reply structurally
             malformed (unclosed tag, fence trap, bare markers).
          4. `low_score` — attempt finished without any flagged
             condition but score is below the restart threshold.
          5. `ok` — attempt finished cleanly enough that the restart
             loop will probably accept it; signature reset.
        Universal: no goal text, no genre.
        """
        if getattr(self, "_dead_first_build_recoveries", 0) >= 1:
            return "dead_first_build"
        if getattr(self, "_identical_reply_loops_this_attempt", 0) >= 1:
            return "identical_reply_loop"
        if getattr(self, "_format_rejections_iter1_this_attempt", 0) >= 1:
            return "format_rejection_iter1"
        if score < getattr(self, "restart_score_threshold", 60):
            return "low_score"
        return "ok"

    def _reset_attempt_state(self) -> None:
        """Reset the per-attempt mutable state so a fresh restart begins
        from a clean slate. Keeps cross-attempt resources (browser,
        backend, memory, playbook, generated assets/sounds cache).
        """
        self._messages = []
        self._last_drained_feedback = []
        self._previous_report_ok = None
        self._previous_report = None
        self._iter_budget_bonus = 0
        self._consecutive_clean_iters = 0
        self._snapshot_n = 0
        self._fix_mode = False
        self._last_diagnose = None
        self._stuck_streak = 0
        self._criteria = ""
        self._probes = []
        self._current_file = ""
        self._last_iter_run = 0
        self._last_report_summary = ""
        self._pending_bullet_lookups = []
        self._pending_coaching = []
        self._last_mistake_sig = None
        self._repeat_sig_streak = 0
        self._force_question_subsystem = None
        self._allow_one_rewrite = False
        self._scoped_constraints = None
        self._pending_scoped_check_keywords = []
        self._probe_eval_error_streak = {}
        self._probe_eval_error_shape_streak = {}
        self._probe_names_ever_passed = set()
        self._pending_probe_quarantine_notices = []
        self._recent_feedback_texts = []
        self._user_force_done = False
        # Todo-driven execution: a fresh attempt gets a fresh checklist
        # (the model will re-emit <todos> from its new plan/build).
        self._todos_text = ""
        self._todos_items = []
        self._ledger_source = None  # Phase 4B: re-derive on the fresh attempt
        self._current_todo = None
        self._todo_nag_counts = {}
        # Recipe skip cache is code-hash keyed; a fresh attempt rebuilds
        # the file from scratch, so stale entries can't match anyway —
        # clear for hygiene.
        self._recipe_skip_cache = set()
        if self._stop_event is not None:
            self._stop_event.clear()
        self._step_continue = False
        self._last_screenshot_before = None
        self._last_screenshot_after = None
        self._active_bullet_ids = []
        self._active_opening_book_recipes = []
        self._active_visual_playtest_recipe_id = None
        self._active_visual_playtest_auto_probes = []
        self._restart_attempt_idx = 0
        self._restart_attempt_seed = None
        self._force_first_build_prefill = False
        self._first_build_retry_bonus_used = False
        # Reset warnings-persistence so a restart attempt starts the
        # streak counter fresh — otherwise a warning seen in attempt 0
        # would already be in the "compact" state on attempt 1's iter 1
        # and the model would see it as already-stale on first contact.
        self._warning_persistence = {}
        # Per-attempt failure-signature counters. Cleared so the next
        # attempt's signature reflects only its own behavior.
        self._dead_first_build_recoveries = 0
        self._dead_first_build_pending = False
        self._dead_first_build_abort_attempt = False
        # Polish phase (item 2): an in-flight polish flag is stale across
        # restart attempts. `_polish_turns_used` / `_stuck_bon_escalations`
        # are deliberately NOT reset — their caps are per SESSION.
        self._polish_pending = False
        # Fix-round item 6: a fresh attempt gets one new no-action-frame
        # advisory and a clean critic payload-dedupe slate.
        self._no_action_frame_advisory_sent = False
        self._current_critic_payload_fp = None
        self._suppressed_critic_payload_fp = None
        self._identical_reply_loops_this_attempt = 0
        self._format_rejections_iter1_this_attempt = 0
        # NOTE: _prev_attempt_signature and _force_minimal_first_build
        # are set BY the restart loop AFTER an attempt ends, so they
        # must NOT be reset here.

    def _score_attempt(self) -> float:
        """Score the just-finished attempt. Reuses tools.score_test_report
        on the most recent test report; falls back to 0 when nothing
        ran (e.g. crash before iter 1 produced a report)."""
        if self._previous_report_ok is True:
            return 100.0
        return score_test_report(self._previous_report or {})

    @staticmethod
    def _restart_temperature_bias(attempt_idx: int) -> float:
        """Small deterministic temp offsets for restart attempts."""
        if attempt_idx <= 0:
            return 0.0
        pattern = (-0.20, +0.10, -0.30, +0.20)
        return pattern[(attempt_idx - 1) % len(pattern)]

    # -- helpers ----------------------------------------------------------

    async def _final_iter_test_if_needed(
        self,
    ) -> AsyncIterator[AgentEvent]:
        """If the last materialized file was never tested, run one final
        test before recording the session outcome.

        DK trace 20260513_185815 ended with Turn 14 containing a correct
        full <html_file> that was NEVER run because the loop hit its
        max_iters / format-stuck budget. The user reported "no game
        graphics" because what got tested was the broken first build,
        not the final code. Closing this gap is a one-time deterministic
        check at every exit point; saves a session that would otherwise
        ship a stale best.html or an empty out_path.

        Side effects (on success only):
          - Updates self._last_test_report / _last_report_summary
          - If the test passes AND a baseline isn't already saved,
            promotes the file to best.html so _record_session_outcome
            reports ok=True.
        Yields AgentEvent('test', ...) so the UI shows the result.
        """
        if not self._last_materialized_iter:
            return
        if self._last_tested_iter >= self._last_materialized_iter:
            return
        if not self._current_file:
            return
        if self.browser is None:
            return
        try:
            self.out_path.write_text(self._current_file, encoding="utf-8")
        except Exception as e:
            self._trace({"kind": "final_iter_test_write_failed", "err": str(e)})
            return
        yield self._record(AgentEvent(
            "info",
            f"[final-test] last shipped code (iter "
            f"{self._last_materialized_iter}) was never tested — running "
            "one closing verification.",
        ))
        # Same retry-once + traceback capture as the iteration-loop test
        # (harness crashes are data-shape-dependent; see harness_crash trace).
        report = None
        for _test_attempt in (1, 2):
            try:
                report = await self.browser.load_and_test(
                    self.out_path,
                    screenshot_path=None,
                    screenshot_before_path=None,
                    probes=self._probes or None,
                    opening_book_recipes=getattr(self, "_active_opening_book_recipes", []),
                    criteria=self._criteria or None,
                    goal=self._goal or "",
                    visual_recipe_id=getattr(self, "_active_visual_playtest_recipe_id", None),
                )
                break
            except Exception as e:
                self._trace_exception(
                    "harness_crash", e,
                    context="final_test", test_attempt=_test_attempt,
                )
                if _test_attempt == 2:
                    yield self._record(AgentEvent(
                        "info",
                        f"[final-test] browser harness crashed (after retry): {e}",
                    ))
                    return
        if report is None:
            return
        report_text = format_report_for_model(report)
        self._last_report_summary = report_text
        self._last_test_report = report
        self._last_tested_iter = self._last_materialized_iter
        yield self._record(AgentEvent("test", report_text, report))
        if report.get("ok"):
            if not self.best_path.exists():
                self._save_best(self._current_file)
            self._trace({
                "kind": "final_iter_test_passed",
                "iteration": self._last_materialized_iter,
            })

    @staticmethod
    def _current_git_sha() -> str:
        """Short repo SHA for batch/trace correlation (best-effort)."""
        import subprocess
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            return out.strip()
        except Exception:
            return ""

    def _record_session_outcome(self, ok: bool) -> None:
        try:
            self._memory.record_outcome(
                session_id=self._session_id,
                goal=self._goal,
                model=self.model,
                iterations=self._last_iter_run,
                ok=ok,
                best_html_path=self.best_path if self.best_path.exists() else None,
                last_report_summary=self._last_report_summary,
            )
            git_sha = self._current_git_sha()
            # Numeric session score (same helper as restart selection) so
            # offline compare_runs / credit tools need no probe re-join.
            try:
                session_score = float(self._score_attempt())
            except Exception:
                session_score = 100.0 if ok else 0.0
            # Offline credit / compare_runs: distinguish "never clean" memory
            # fails from sessions that never produced code (backend crash).
            code_materialized = int(getattr(self, "_snapshot_n", 0) or 0) > 0
            backend_crashed = bool(
                getattr(self, "_session_backend_crashed", False)
                or getattr(self, "_last_stream_crashed", False)
            )
            outcome: dict[str, object] = {
                "kind": "session_outcome",
                "ok": ok,
                "iterations": self._last_iter_run,
                "best_path_exists": self.best_path.exists(),
                "score": session_score,
                "code_materialized": code_materialized,
                "backend_crashed": backend_crashed,
            }
            if git_sha:
                outcome["git_sha"] = git_sha
            self._trace(outcome)
        except Exception as e:
            self._trace({"kind": "outcome_record_failed", "err": str(e)})
        # One-complete-trace (2026-06-14): the .summary.md sibling is retired —
        # its iter table is fully derivable from the .jsonl, which is now the
        # single complete artifact. The render_run_summary() function stays
        # available (unit-tested, runnable on demand against a .jsonl); we just
        # no longer auto-write the redundant file each run.

    def _write_run_summary(self) -> None:
        try:
            if not self.trace_path.exists():
                return
            records: list[dict] = []
            with self.trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
            text = render_run_summary(records, artifact_id=self._artifact_id)
            summary_path = self.trace_path.with_name(
                self.trace_path.stem + ".summary.md"
            )
            summary_path.write_text(text, encoding="utf-8")
            self._trace({
                "kind": "run_summary_written",
                "path": str(summary_path),
            })
        except Exception as e:
            self._trace_exception("run_summary_failed", e)

    @staticmethod
    def _chunk_for_display(text: str, chunk: int = 120) -> list[str]:
        """Split a string into pseudo-tokens so the silent best-of-N winner
        still feels streamy when we replay it through the on_token callback.
        """
        out: list[str] = []
        i = 0
        L = len(text)
        while i < L:
            j = min(i + chunk, L)
            # Try to break at whitespace for readability.
            k = text.rfind(" ", i, j)
            if k > i + chunk // 2:
                j = k + 1
            out.append(text[i:j])
            i = j
        return out


def module_inspect_source() -> str:
    """agent.py + extracted mixin modules (for regression guards on moved code)."""
    import inspect
    import sys
    import agent_compaction
    import agent_feedback as _agent_feedback
    import agent_helpers
    import agent_memory
    import agent_probes
    import agent_prompts
    import agent_stream
    return "\n".join(
        inspect.getsource(m)
        for m in (
            sys.modules[__name__],
            agent_helpers,
            _agent_feedback,
            agent_prompts,
            agent_compaction,
            agent_stream,
            __import__("agent_gates"),
            __import__("agent_critic"),
            __import__("agent_assets"),
            agent_probes,
            agent_memory,
        )
    )
