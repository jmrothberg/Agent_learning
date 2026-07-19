"""Streaming and materialization extracted from agent.py.

Moved VERBATIM from `GameAgent` (no behavior change).
"""

from __future__ import annotations


import asyncio
import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any


from agent_helpers import (

    _ASSETS_OPEN_RE,
    _BARE_DOCTYPE_RE,
    _BARE_HTML_ELEMENT_RE,
    _baseline_structurally_broken,
    _BLOAT_BLOCK_LINES,

    _CONFIRM_RE,
    _DIAGNOSE_RE,
    _DONE_RE,
    _HTML_FENCE_RE,
    _HTML_RE,
    _PLAN_OPEN_RE,
    _PROBES_OPEN_RE,
    _QUESTION_RE,
    _SOUNDS_OPEN_RE,

    _UNCLOSED_HTML_FILE_RE,

    _detect_block_bloat,

    _detect_skeleton_payload,

    _is_degenerate_baseline,

    _is_placeholder_first_build,

    _looks_like_placeholder_html_payload,

    _normalize_extracted_html,

    _patch_set_bracket_break,

    _png_dims,

    _strip_thinking,

    _truncation_reason,

)

from patches import apply_patches, classify_format_failure, extract_patches, FormatRejection

from ollama_io import StreamResult


def _repetition_loop_abort_message(
    *,
    tokens: int,
    duration_s: float,
    loop_kind: str | None,
    loop_line: str | None,
) -> str:
    """Human-readable guard-abort text keyed to RepetitionDetector stall_reason."""
    kind = loop_kind or "unknown"
    desc_by_kind = {
        "inline_data_bloat": (
            "duplicated the same structured 8-line block repeatedly"
        ),
        "adjacent_line_spam": "emitted identical consecutive lines",
        "short_line_loop": "emitted the same 1-2 short lines on repeat",
        "near_dup_template_loop": (
            "emitted numbered template variants on repeat"
        ),
        "intra_line_repetition": (
            "degenerated into a boundary-free character repeat"
        ),
    }
    desc = desc_by_kind.get(kind, "entered a token-repetition loop")
    msg = (
        f"Repetition loop detected — model {desc} after "
        f"{tokens} tokens ({duration_s:.0f}s). "
        "Aborted stream and kept partial output."
    )
    if loop_line:
        preview = loop_line[:80] + ("..." if len(loop_line) > 80 else "")
        msg += f" reason={kind} sample='{preview}'"
    else:
        msg += f" reason={kind}"
    return msg




class StreamMaterializeMixin:
    """Stream + materialize for GameAgent (see module docstring)."""

    _REJECTED_REPLY_STUB_HEAD = 400

    @staticmethod
    def _stub_rejected_reply(reply: str, rejection_kind: str) -> str | None:
        """Return a stubbed replacement for a format-rejected reply, or None.

        DeepSeek-V4-Flash FPS trace 20260611_213744: three rejected
        replies (~86 KB, crashed/looped streams with unclosed
        <html_file>) stayed verbatim in `_messages`, ballooning the
        prompt 9K → 47K tokens before the compaction ceiling fired.
        Unclosed <html_file> bodies don't match the per-turn elision
        regex (no closing tag), so structured pruning can't shrink them
        either. The full text is already preserved in the trace
        (`assistant_reply`) and the .log mirror — history only needs a
        recognizable head plus an explicit "this was rejected" marker.

        Returns None when the reply is short enough that stubbing
        wouldn't save anything meaningful.
        """
        text = reply or ""
        head_n = StreamMaterializeMixin._REJECTED_REPLY_STUB_HEAD
        # Stub only when there is real bulk to elide (head + marker
        # would otherwise be longer than the original).
        if len(text) <= head_n + 200:
            return None
        elided = len(text) - head_n
        return (
            text[:head_n]
            + f"\n[harness: remaining {elided} chars elided — this reply "
            f"was rejected ({rejection_kind}) and nothing in it was "
            "usable; do not repeat this output]"
        )

    @staticmethod
    def _should_skip_format_doctor(
        *,
        last_stream_looped: bool,
        last_stream_loop_kind: str | None,
        rejection_kind: str,
    ) -> bool:
        """Skip doctor when a known high-cost/non-recovering pattern hits.

        Donkey-kong traces showed this sequence repeatedly:
          inline_data_bloat loop abort -> unclosed_html_file rejection.
        Re-streaming the same huge partial reply through format-doctor
        usually burns minutes and still truncates. Let the next iter use
        direct recovery coaching instead.
        """
        return (
            last_stream_looped
            and last_stream_loop_kind == "inline_data_bloat"
            and rejection_kind in ("unclosed_html_file", "unclosed_patch")
        )

    @staticmethod
    def _no_usable_code_fallback(
        *,
        plan_only: bool,
        has_existing_file: bool,
        consecutive_plan_only: int,
        rejection: FormatRejection | None = None,
        format_stuck_streak: int = 0,
        probes_only: bool = False,
        media_only: bool = False,
        prior_stream_looped: bool = False,
        prior_stream_silent: bool = False,
        prior_stream_deliberated: bool = False,
        prior_loop_kind: str | None = None,
        prior_loop_line: str | None = None,
        is_local_backend: bool = False,
        materialize_reject_reason: str = "",
        identical_repeat: bool = False,
        art_pending: bool = False,
    ) -> tuple[str, bool]:
        """Pick the fallback message + decide whether to reset the
        plan-only streak counter.

        Returns (fallback_text, should_reset_streak).

        Branching:
          1. `consecutive_plan_only >= 2` — hard loop-break with the
             strongest directive. Counter is reset so escalation can't
             stack forever. Handles the centipede/model-8 trace pattern
             where the model re-emits <plan> indefinitely even with a
             baseline file present.
          2. `plan_only AND NOT has_existing_file` — escalate immediately
             on the FIRST strike. With no baseline yet, code emission
             isn't optional: Phase A already supplied the plan, and
             repeating it on iter 1 is pure waste. Saves ~60-120s vs.
             waiting for a second strike. The asteroid_20260510_173200
             and DK 20260513_153626 traces both showed this exact
             one-strike pattern on reasoner-class models.
          3. `plan_only AND has_existing_file` — softer "stop re-
             emitting plan, write the rewrite" message. The model may
             have a legitimate reason (e.g. user asked for a redesign);
             we don't want to escalate on the first strike here.
          4. Default — no <plan>, no code: generic "send patches or
             html_file" reminder.

        When `rejection` is non-None the model emitted a structurally-
        recognizable but malformed reply (e.g. <patch> inside a ```
        fence). Prepend the rejection.detail to the chosen fallback so
        the model sees WHY parsing failed. At `format_stuck_streak >= 2`
        also append a hard "stop using <patch>, send full <html_file>"
        escalation — the DK trace 20260513_185815 burned 7 turns on the
        same broken shape because the generic fallback gave the model
        no signal to change strategy.
        """
        # Identical-reply loop escalation. When the model emits a
        # bit-identical (or near-identical) unparseable reply twice in
        # a row, the standard fallback prompt is also identical and the
        # model has no new signal to change behavior. Wolfenstein
        # 2026-05-24 trace: 5 consecutive iters with 7838-token
        # identical replies, each burning ~340s of GPU. Universal
        # escalation: drop any thought of re-emitting the file, demand
        # a minimal patch keyed on a single named symptom. Reset the
        # plan-only streak so this branch only fires once before
        # routing back to other handlers — if the next reply is ALSO
        # identical, we'll have fingerprinted differently because the
        # prompt changed; if it isn't, we want the normal branches.
        if identical_repeat:
            fallback = (
                "IDENTICAL-REPLY LOOP DETECTED: your previous two "
                "replies were byte-identical (or near-identical) and "
                "the parser rejected BOTH. Re-emitting the same text "
                "will be rejected again — you must change shape.\n\n"
                "Recovery for THIS turn:\n"
                "  - Do NOT re-emit <html_file>. The file you keep "
                "trying to send is either too large for what remains "
                "of the context window, or contains a parse-defeating "
                "pattern (stray markdown fence, unclosed tag, "
                "duplicated declaration).\n"
                "  - Pick the SINGLE most important symptom from the "
                "most recent test report below (frozen canvas, RAF "
                "dead, console error, failing probe) and emit ONE "
                "<patch>...</patch> with at most 5-10 lines of SEARCH "
                "context that addresses just that one thing.\n"
                "  - Start your reply immediately with <patch> as the "
                "first non-whitespace text. No preamble, no <diagnose>, "
                "no <plan>, no <html_file>."
            )
            return fallback, True
        # Silent-stream recovery — when ollama_io / MLX aborted the
        # previous stream because it produced ZERO non-empty content
        # for >=180s (all output went to a reasoning/thinking channel
        # that surfaces as empty `content`). Motivating trace: doom
        # 2026-05-23 iter 4 — 1356s wall-clock, 32,777 completion
        # tokens, ZERO visible pieces, deliberation detector didn't
        # fire because it feeds on `piece` content. The recovery
        # message tells the model EXPLICITLY to start with an opening
        # tag and skip any reasoning preamble.
        if prior_stream_silent:
            fallback = (
                "SILENT STREAM RECOVERY: your previous reply produced "
                "ZERO visible content for the entire wall-clock budget. "
                "Either the model spent all of its tokens inside a "
                "reasoning/thinking channel that surfaces as empty "
                "content, or it generated only whitespace.\n\n"
                "Recovery for THIS turn:\n"
                "  - Start your reply DIRECTLY with one of the opening "
                "tags: <patch>, <html_file>, <plan>, or <done/>. The "
                "very first non-whitespace text MUST be a tag.\n"
                "  - Do NOT begin with `<think>`, prose, or any "
                "explanatory preamble. Skip reasoning entirely if your "
                "model has a reasoning mode — go straight to the tag.\n"
                "  - Keep this reply small: one focused <patch> if the "
                "file is healthy, or a complete <html_file> if you "
                "need a fresh draft. Either way, the first token "
                "matters more than the length."
            )
            return fallback, False
        # Phase 0D-2 (Fieldrunners trace 20260626_102307 iter 4): deliberation
        # recovery. The model rambled 707s of pre-tag prose and the deliberation
        # guard aborted with NO output tag. The generic fallback below tells the
        # model "I could not find a <patch>" but gives no anti-deliberation
        # directive and never mentions <assets> — even when the user explicitly
        # asked for new art. Name the failure and force a tag-first reply, asset
        # generation FIRST when art is pending (the iter-2 pattern that worked).
        if prior_stream_deliberated:
            if art_pending:
                fallback = (
                    "DELIBERATION RECOVERY: your previous reply was pure "
                    "reasoning prose with no output tag — the deliberation "
                    "guard aborted it, so NOTHING was saved.\n\n"
                    "The user asked for NEW ART. Recovery for THIS turn:\n"
                    "  - Start IMMEDIATELY with an <assets> block (a bare JSON "
                    "array) naming a FEW new sprites, then optionally one small "
                    "<patch> to load/draw them.\n"
                    "  - Do NOT think out loud, do NOT weigh options, do NOT "
                    "re-plan. The first non-whitespace text MUST be `<assets>`.\n"
                    "  - Keep it small: a handful of new names this turn, not "
                    "the entire roster."
                )
            else:
                fallback = (
                    "DELIBERATION RECOVERY: your previous reply was pure "
                    "reasoning prose with no output tag — the deliberation "
                    "guard aborted it, so NOTHING was saved.\n\n"
                    "Recovery for THIS turn:\n"
                    "  - Write ONE short sentence inside "
                    "<diagnose>...</diagnose> naming the single line/variable "
                    "responsible, then IMMEDIATELY emit ONE "
                    "<patch>...</patch>.\n"
                    "  - Do NOT think out loud, do NOT explore alternatives. "
                    "The first non-whitespace text MUST be `<diagnose>` or "
                    "`<patch>`."
                )
            return fallback, True
        # Phase 4: duplicate-decl-aware coaching. When `_materialize`
        # rejected the reply because the inbound HTML had concatenated
        # drafts (duplicate top-level `const` / `function`), naming the
        # specific shape lets the model emit ONE single-pass body next
        # turn instead of guessing why it was rejected. Trace 1 (chess
        # 20260522_000304) burned several iters on this exact shape.
        reject_low = (materialize_reject_reason or "").lower()
        if (
            "duplicate" in reject_low
            and "declaration" in reject_low
        ):
            fallback = (
                "Your last reply was rejected because the <html_file> "
                "body contained DUPLICATE TOP-LEVEL DECLARATIONS — two "
                "drafts of the same function or const got concatenated "
                "in one stream. The browser would refuse to run that "
                f"file (\"{materialize_reject_reason[:200]}\").\n\n"
                "Recovery for THIS turn:\n"
                "  - Emit ONE complete <html_file>...</html_file> as a "
                "single coherent body. No duplicate `const ctx`, no "
                "duplicate `function loop()`, no second `(() => { ... })()` "
                "below the first.\n"
                "  - If you were running out of room, narrow scope: "
                "build the core skeleton this turn and defer extra "
                "polish to a later <patch>.\n"
                "  - If you genuinely cannot fit the file in one stream, "
                "emit a <question> asking the user to narrow scope "
                "instead of shipping concatenated drafts.\n"
                "Do NOT include <plan>, <criteria>, or <probes>."
            )
            return fallback, False
        # Baseline-exists rejection (2026-06-10 dojo-fight traces): the
        # old flow fell through to the generic "I could not find a <patch>
        # or <html_file> block in your reply" — factually FALSE (the reply
        # contained a full <html_file>; the gate rejected it) — so the
        # model dropped its in-flight fix instead of re-sending it as
        # patches. Tell it the truth and ask for the SAME changes as
        # <patch> blocks.
        if "baseline file already exists" in reject_low:
            fallback = (
                "Your <html_file> WAS received, but it was REJECTED: a "
                "working baseline file already exists on disk, and full "
                "rewrites of a working game are banned (regression risk). "
                "Your changes were NOT applied and NOT saved.\n\n"
                "Recovery for THIS turn: re-send the SAME fixes you just "
                "wrote, but as one or more <patch> SEARCH/REPLACE blocks "
                "against the CURRENT file on disk (shown in an earlier "
                "message). Keep each SEARCH block small (5-10 lines) and "
                "copy it EXACTLY from the current file. Do NOT re-emit "
                "<html_file>."
            )
            return fallback, False
        if consecutive_plan_only >= 2:
            fallback = (
                "LOOP DETECTED: you have emitted only <plan> for "
                f"{consecutive_plan_only} iterations with no "
                "code. This iteration MUST produce code, or the "
                "session will be aborted. Emit exactly one of:\n"
                "  - a complete <html_file>...</html_file> (for a "
                "full rewrite), or\n"
                "  - one or more <patch>...</patch> blocks (for "
                "incremental changes).\n"
                "Do NOT include <plan>, <criteria>, or <probes>."
            )
            return fallback, True
        if plan_only and not has_existing_file:
            fallback = (
                "BUILD PHASE — code emission is REQUIRED this turn. "
                "Phase A already collected the plan, criteria, and "
                "probes; there is no baseline file on disk yet, so "
                "the iteration is meaningless without a complete "
                "<html_file>...</html_file>. Do NOT re-emit <plan>, "
                "<criteria>, or <probes> — those were already "
                "accepted last turn and live in the session state. "
                "If you also need to refine art per user feedback, "
                "you MAY emit a small <assets> block alongside the "
                "code (the mid-session regen pipeline will fulfill "
                "it); but the <html_file> is required either way. "
                "Emit the game's full HTML now."
            )
            return fallback, False
        if plan_only and has_existing_file:
            fallback = (
                "You already provided a <plan>. The user wants "
                "a full rewrite of the existing file. Stop "
                "re-emitting <plan>. Emit one complete "
                "<html_file>...</html_file> now containing the "
                "new game. Do NOT include <plan>, <criteria>, "
                "or <probes> in this reply."
            )
            return fallback, False
        # Probes-only / media-only: the model re-emitted Phase-A signals
        # (<probes>, <assets>, <sounds>) without any <html_file> or
        # <patch>. Observed in donkey-kong traces 20260516_124628 iter 1
        # (probes-only) and 20260516_142445 iter 1 (probes preamble +
        # later <html_file>). The generic fallback below leaves the
        # model guessing why the turn was rejected; this one names the
        # specific shape so the next turn skips the redundant block.
        if probes_only or media_only:
            tag_name = "<probes>" if probes_only else "<assets> / <sounds>"
            fallback = (
                f"You emitted {tag_name} but no <html_file> or <patch> "
                "this turn. Iter 1 must produce the first build, and on "
                "any iter after that a code-changing tag is required. "
                "The probes / assets / sounds from Phase A are still in "
                "force — re-emitting them here is not needed. Emit a "
                "complete <html_file>...</html_file> now (or one or more "
                "<patch> blocks if a baseline file already exists). "
                "Do NOT include <probes>, <assets>, or <sounds> in this "
                "reply unless you are intentionally adding to them; "
                "they live in session state."
            )
            return fallback, False
        # Repetition-loop recovery path (plan item: loop-recovery-minpatch).
        # After a loop abort on an existing baseline, force a tiny patch-only
        # turn so we recover deterministically instead of re-streaming another
        # large draft.
        if prior_stream_looped and has_existing_file:
            if art_pending:
                # Phase 0D-2 (Fieldrunners iter 5): the loop happened while the
                # model bulk-emitted enemy data / repeated SEARCH-REPLACE
                # scaffolding for a big art ask. Telling it "ONE minimal patch"
                # ignores the pending art request and drops it. Route to a SMALL
                # <assets> turn instead — few names, no inline data dumps (the
                # thing that looped).
                fallback = (
                    "REPETITION-LOOP RECOVERY: your previous stream was aborted "
                    "after repeating tokens while emitting bulk data.\n\n"
                    "The user asked for NEW ART. Recover with ONE small "
                    "<assets> block this turn:\n"
                    "  - Start immediately with <assets> (a bare JSON array) "
                    "naming only a FEW new sprites — not the whole roster.\n"
                    "  - Do NOT inline large data arrays/tables in this reply; "
                    "that is what looped.\n"
                    "  - You MAY add one small <patch> after the <assets> block, "
                    "but assets come first. No long reasoning, no <html_file>."
                )
            else:
                fallback = (
                    "REPETITION-LOOP RECOVERY: your previous stream was aborted "
                    "after repeating tokens. Recover with ONE minimal "
                    "<patch>...</patch> only.\n"
                    "Rules for this turn:\n"
                    "  - Start immediately with <patch> as the first non-whitespace text.\n"
                    "  - No long reasoning, no re-deriving prior context, no <html_file>.\n"
                    "  - Change only the smallest failing region.\n"
                    "If you are uncertain, patch one symbol/path and let the next "
                    "test report guide the next step."
                )
            return fallback, False
        # Repetition-loop + unclosed <html_file>: the most common
        # sequence is "model entered a token loop inside a code block, the
        # RepetitionDetector aborted the stream mid-emit, and the parser
        # rejected the partial reply as `unclosed_html_file`." Surfacing
        # this combination prescriptively lets the model recover without
        # blindly re-issuing the same draft. Observed in donkey-kong
        # trace 20260516_142445 iter 1 (16+ `p.onGirder = false;` repeats
        # in a dead state-reset block). Without this branch the model
        # sees a generic "your tags were malformed" and has no signal
        # about WHAT broke.
        if (
            prior_stream_looped
            and rejection is not None
            and rejection.kind in ("unclosed_html_file", "unclosed_patch")
        ):
            kind_label = (
                "an `<html_file>`"
                if rejection.kind == "unclosed_html_file"
                else "a `<patch>`"
            )
            loop_shape = {
                "adjacent_line_spam": "the same line N times in a row",
                "short_line_loop": "the same 1-2 short lines cycling",
                "near_dup_template_loop": "near-duplicate template lines",
                "inline_data_bloat": "an 8-line block duplicated 3+ times",
                "intra_line_repetition": "a short phrase repeated over and over on one line",
            }.get(prior_loop_kind or "", "the same content on repeat")
            line_clue = ""
            if prior_loop_line:
                # Truncate long lines so the coaching stays small.
                clue = prior_loop_line[:80]
                if len(prior_loop_line) > 80:
                    clue += "…"
                line_clue = (
                    f" The repeated content was: `{clue}`."
                )
            if is_local_backend:
                fallback = (
                    f"Your previous reply hit a token-repetition loop and the "
                    f"stream was aborted, so {kind_label} block has no closing "
                    f"tag. The loop shape was: {loop_shape}.{line_clue}\n\n"
                    "DO NOT ask the user a question this turn. Recover "
                    "autonomously by emitting a smaller complete "
                    "<html_file>...</html_file> that OMITS the branch that "
                    "was looping. If it was a fall-through state-reset block "
                    "where every flag was already cleared upstream, delete the "
                    "block entirely — don't pad with redundant "
                    "`flag = false; flag = false;` statements (known loop "
                    "trigger on local models).\n\n"
                    "Keep this turn short and code-only."
                )
            else:
                fallback = (
                    f"Your previous reply hit a token-repetition loop and the "
                    f"stream was aborted, so {kind_label} block has no closing "
                    f"tag. The loop shape was: {loop_shape}.{line_clue}\n\n"
                    "DO NOT re-emit the same draft — the section that was "
                    "looping is the root cause; restarting will hit the same "
                    "wall. Instead, choose ONE:\n"
                    "  - emit a `<question>` describing what you were trying "
                    "to compute when the loop started (preferred when you're "
                    "unsure how to proceed without the dead branch), OR\n"
                    "  - emit a smaller `<html_file>...</html_file>` that "
                    "OMITS the branch that was looping. If it was a "
                    "fall-through state-reset block where every flag was "
                    "already cleared upstream, delete the block entirely — "
                    "don't pad with redundant `flag = false; flag = false;` "
                    "statements (those are a known token-loop trigger).\n\n"
                    "Whichever you choose, keep this turn SHORT."
                )
            return fallback, False
        # Structured-rejection path: model emitted something tag-shaped
        # but malformed. Surface the specific reason BEFORE the generic
        # reminder so the model can pattern-match on what to change.
        if rejection is not None:
            parts = [rejection.detail]
            if format_stuck_streak >= 2:
                parts.append(
                    "ESCALATION — format-stuck streak: "
                    f"{format_stuck_streak} consecutive parse failures. "
                    "Stop trying to send <patch> this turn. Send a "
                    "complete <html_file>...</html_file> containing the "
                    "full corrected file, with NO ```markdown``` fences "
                    "anywhere around or inside the tag. Raw "
                    "<html_file>...</html_file> as the first non-prose "
                    "lines of your reply."
                )
            else:
                parts.append(
                    "Re-emit your fix as either ONE <patch>...</patch> "
                    "block or a complete <html_file>...</html_file>. "
                    "No markdown fences."
                )
            return "\n\n".join(parts), False
        if not has_existing_file:
            fallback = (
                "FIRST BUILD REQUIRED — FORMAT-ONLY RECOVERY: your previous "
                "reply had no usable <html_file>/<patch>. Re-emit this turn "
                "as CODE ONLY.\n"
                "Start your reply immediately with `<html_file>` as the first "
                "non-whitespace token (no preamble, no reasoning prose), then "
                "emit the complete HTML document and close with `</html_file>`."
            )
            return fallback, False
        fallback = (
            "I could not find a <patch> or <html_file> block "
            "in your reply. If this is the first build, send "
            "a complete <html_file>. Otherwise send <patch> "
            "blocks."
        )
        return fallback, False



    async def _run_format_doctor(
        self, failed_reply: str, rejection: FormatRejection,
    ) -> str | None:
        """One-shot reformat pass on an unparseable reply.

        Same backend / same loaded model, but a FRESH isolated message
        history: the doctor sees only the failed reply + the structured
        rejection, never the agent's full conversation. Output is the
        reformatted reply (string) or None on any failure.

        Grounded by external verifier signal (the parser rejection), not
        free reflection — keeps it inside the "what works for weak local
        models" envelope (TACL 2024 / arXiv 2411.17501: weak models
        can't reliably self-critique without an external check).
        """
        # Build the narrow doctor prompt.
        sys_prompt = (
            "You are a format-doctor. Your ONE job is to reformat a "
            "previous reply so the harness can parse it. You do not "
            "judge the content. You do not refactor. You do not add or "
            "remove logic. You output ONLY a corrected <patch>...</patch> "
            "block OR a complete <html_file>...</html_file> block, "
            "nothing else — no prose, no <plan>, no <notes>, no "
            "markdown fences around the output tags. If the input "
            "contained a <patch> body, preserve the SEARCH and REPLACE "
            "text exactly; only fix the wrapping. If the input contained "
            "a complete HTML document, wrap it in <html_file>...</html_file>."
        )
        user_msg = (
            f"PARSER REJECTION: {rejection.detail}\n\n"
            "PREVIOUS REPLY (unparseable):\n"
            "================================\n"
            f"{failed_reply}\n"
            "================================\n\n"
            "Re-emit the corrected output now. Output ONLY the "
            "<patch>...</patch> or <html_file>...</html_file> tag and "
            "its body — nothing before, nothing after, no fences."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        # Keep doctor bounded so a malformed reply can't trap the session in
        # a long opaque recovery sub-step while user feedback queues up.
        doctor_stall_seconds = min(self.stall_seconds, 120.0)
        doctor_overall_seconds = min(self.overall_seconds, 240.0)
        self._trace({
            "kind": "format_doctor_start",
            "stall_seconds": doctor_stall_seconds,
            "overall_seconds": doctor_overall_seconds,
            "rejection_kind": rejection.kind,
        })
        try:
            result = await self._backend.stream_chat(
                messages,
                on_token=None,
                options={"temperature": 0.1, "num_ctx": self.num_ctx},
                keep_alive=self._keep_alive_for_backend(self._backend),
                stall_seconds=doctor_stall_seconds,
                overall_seconds=doctor_overall_seconds,
                max_retries=0,
                cancel_event=self._ensure_stop_event(),
            )
            text = result.text or ""
        except Exception as e:
            # traceback included (2026-06-10): str(e) alone hid a harness
            # bug for days — see _trace_exception docstring.
            self._trace_exception("format_doctor_error", e)
            return None
        self._trace({
            "kind": "format_doctor_stream_done",
            "len": len(text),
            "preview": text[:300],
            "stalled": bool(getattr(result, "stalled", False)),
            "looped": bool(getattr(result, "looped", False)),
            "deliberated": bool(getattr(result, "deliberated", False)),
            "crashed": bool(getattr(result, "crashed", False)),
        })
        return text or None

    async def _stream(
        self, on_token, *,
        override_temp: float | None = None,
        prefill: str = "",
        prefill_force: bool = False,
        role: str = "coder",
    ) -> str:
        """Stream once, with watchdog. Recovers from stalls by raising/logging.

        Image attachment: if VLM is detected and self._next_image_bytes is set,
        attach to the LAST user message and clear the buffer.

        Prefill (Continue.dev pattern): when non-empty AND use_prefill is on,
        a trailing assistant message with `prefill` content is appended so
        Ollama continues from there. The prefill is prepended to the
        returned text so downstream parsers see the full output.
        """
        from agent import AgentEvent  # late import — agent loads this module first

        # Determine the target backend and its VLM capability dynamically
        active_backend = self.get_backend(role)
        if active_backend is None:
            raise RuntimeError(f"no backend configured for role={role!r}")
        is_vlm_active = await self._detect_vlm(role)

        self._last_stream_role = role
        self._set_role_activity(role, f"streaming {role}…")
        self._pending_stream_ui_events = []

        if role == "coder":
            freed = self._maybe_release_diffusers_before_coder_stream()
            if freed:
                self._trace({
                    "kind": "diffuser_release_before_coder_stream",
                    "freed": freed,
                    "model_role": role,
                })

        if (
            is_vlm_active
            and self._messages
            and self._messages[-1].get("role") == "user"
        ):
            # Multi-image attach: prefer the before/after pair when the
            # double-screenshot feature is on and both are present.
            imgs: list[bytes] = []
            sources: list[str] = []
            if self._use_double_screenshot:
                if self._last_screenshot_before:
                    imgs.append(self._last_screenshot_before)
                    sources.append(self._last_screenshot_before_path or "<before>")
                if self._last_screenshot_after:
                    imgs.append(self._last_screenshot_after)
                    sources.append(self._last_screenshot_after_path or "<after>")
            elif self._next_image_bytes:
                imgs.append(self._next_image_bytes)
                sources.append(self._last_screenshot_after_path or "<queued>")
            if imgs:
                self._messages[-1]["images"] = imgs
                self._trace({
                    "kind": "image_attached",
                    "iteration": self._snapshot_n,
                    "count": len(imgs),
                    "bytes": sum(len(b) for b in imgs),
                    "sources": sources,
                    "dims": [_png_dims(b) for b in imgs],
                    "model_is_vlm": True,
                })
                self._next_image_bytes = None
            else:
                # VLM model but nothing to attach this turn — usually
                # the first turn (no screenshot yet) or a critique turn
                # without double-screenshot enabled. Logging this makes
                # "is the model getting eyes on the game" grep-answerable.
                self._trace({
                    "kind": "image_skipped",
                    "iteration": self._snapshot_n,
                    "reason": "no screenshot bytes queued",
                    "model_is_vlm": True,
                })

        # Optional Continue.dev-style assistant prefill. Only applied
        # when feature is on AND `prefill` is provided. We insert a
        # trailing assistant message; Ollama's chat API treats it as a
        # partial completion to extend.
        #
        # Anthropic exception (MK trace 20260528, iter 2): newer Claude
        # models (Opus 4.7+) hard-reject ANY trailing assistant prefill
        # with a 400 "does not support assistant message prefill" — even
        # whitespace-stripped. For backend=anthropic we FOLD the tag
        # opener into the last user message as a format hint instead,
        # and still prepend it locally to the returned text so the
        # downstream <plan>/<diagnose> regex parsers see the same shape.
        prefill_used = False
        anthropic_prefill_folded = False
        _orig_last_user_content: str | None = None
        prefill_enabled = bool(prefill) and (self._use_prefill or prefill_force)
        if prefill_enabled:
            is_anthropic = (
                getattr(getattr(active_backend, "info", None), "name", "") == "anthropic"
            )
            if is_anthropic:
                # Fold: append a hint to the last user message (in place).
                # If there is no trailing user message we cannot fold —
                # fall back to skipping prefill on Anthropic entirely
                # rather than re-introducing the 400.
                if (
                    self._messages
                    and self._messages[-1].get("role") == "user"
                ):
                    _orig_last_user_content = self._messages[-1].get("content", "") or ""
                    # Use just the first non-whitespace line of the prefill
                    # as the literal opener the model must reproduce.
                    first_line = prefill.strip().split("\n", 1)[0].strip()
                    hint = (
                        "\n\nFORMAT: begin your reply with exactly `"
                        + first_line
                        + "` (no prose before it; no extra whitespace)."
                    )
                    self._messages[-1] = {
                        **self._messages[-1],
                        "content": _orig_last_user_content + hint,
                    }
                    anthropic_prefill_folded = True
                    prefill_used = True
                    self._trace({
                        "kind": "anthropic_prefill_folded",
                        "tag": first_line[:120],
                        "len": len(prefill),
                        "forced": bool(prefill_force and not self._use_prefill),
                    })
                else:
                    # No user turn to fold into — skip prefill on this
                    # turn rather than risk a 400. Local prepend below
                    # is gated by prefill_used so it also does not run.
                    self._trace({
                        "kind": "anthropic_prefill_skipped",
                        "reason": "no trailing user message to fold into",
                        "len": len(prefill),
                    })
            else:
                self._messages.append({"role": "assistant", "content": prefill})
                prefill_used = True
                self._trace({
                    "kind": "prefill",
                    "len": len(prefill),
                    "forced": bool(prefill_force and not self._use_prefill),
                })

        # Build-turn temperature. 0.6 is the Qwen3.6 vendor "thinking-mode /
        # precise-coding (WebDev)" preset (temp 0.6, top_p 0.95, top_k 20 —
        # the tail-truncation half lives in backend.MLXBackend._stream_once).
        # Was 0.7; lowered 2026-05-31 alongside wiring up top_p/top_k, after
        # the dojo-fight trace looped at temp 0.7 with NO tail truncation.
        # Fix-mode patch turns stay tighter (0.25) — surgical edits want
        # determinism, deliberately below the build preset.
        temp = override_temp if override_temp is not None else (
            0.25 if self._fix_mode else 0.6
        )
        if override_temp is None and self._restart_attempt_idx > 0:
            bias = self._restart_temperature_bias(self._restart_attempt_idx)
            temp = max(0.05, min(1.2, temp + bias))
            self._trace({
                "kind": "restart_temp_bias_applied",
                "attempt_idx": self._restart_attempt_idx,
                "bias": bias,
                "result_temp": temp,
            })
        # One row per stream summarizing the turn's routing contract.
        # Read by future debugging tools; does not affect runtime
        # behavior. Best-effort: any KeyError /
        # type error is swallowed by the outer _trace try/except.
        try:
            contract = dict(self._last_turn_contract or {})
            scoped_mode = (self._scoped_constraints or {}).get("mode") or "none"
            contract.update({
                "kind": "turn_contract",
                "fix_mode": self._fix_mode,
                "continuation": bool(getattr(self, "_continuation", False)),
                "scoped_mode": scoped_mode,
                # Phase 2: rewrite is also allowed when the on-disk
                # baseline is structurally broken (`_is_degenerate_baseline`).
                # Surface BOTH paths so the trace truthfully shows the
                # gate state the materializer will actually use.
                "rewrite_allowed": bool(
                    self._allow_one_rewrite
                    or (
                        self._current_file
                        and _is_degenerate_baseline(self._current_file)
                    )
                ),
                "scoped_change_active": bool(self._scoped_change_active),
                "force_question": self._force_question_subsystem is not None,
                "snapshot_n": self._snapshot_n,
                "temperature": temp,
            })
            contract.update(self._prompt_provenance_fields())
            allowed, forbidden = self._derive_allowed_forbidden_tags()
            contract["allowed_tags"] = allowed
            contract["forbidden_tags"] = forbidden
            contract["keep_alive"] = self._keep_alive_for_backend(active_backend)
            self._trace(contract)
        except Exception:
            pass
        # One-complete-trace (2026-06-14): capture the EXACT outgoing turn the
        # model is about to answer — the last message in full (the user turn
        # assembled by _flush_user_injections with all injected playbook /
        # feedback / coaching / report blocks, or an assistant prefill). This
        # single chokepoint records every turn's real input across all ~20
        # append sites, so the .jsonl holds the full conversation that used to
        # live only in .conversation.md. The system prompt is captured once in
        # the system_prompt_built event, so it is not repeated here.
        try:
            _outgoing = self._messages[-1] if self._messages else {}
            _turn_input = {
                "role": _outgoing.get("role"),
                "content": _outgoing.get("content") or "",
            }
        except Exception:
            _turn_input = None
        self._trace({
            "kind": "stream_start",
            "temperature": temp,
            "fix_mode": self._fix_mode,
            "keep_alive": self._keep_alive_for_backend(active_backend),
            "turn_input": _turn_input,
        })

        # Heartbeat wrapper around the caller's on_token. Every
        # _STREAM_HEARTBEAT_SECONDS of wall clock, we trace a
        # `stream_heartbeat` event carrying token count, tok/s, and the
        # last ~120 chars of the stream. This makes a long stream
        # visible in the .log / .jsonl as it runs — without this, a
        # 25-minute degenerate generation looks identical to a healthy
        # stream that's writing to a different file (the user-facing
        # symptom that motivated this change). Cheap: at most one
        # trace event every 30 seconds.
        import time as _time
        # Phase 4 (4D.1): reset per-stream prefill timing; set when the
        # slow-prefill watchdog fires below. Folded onto iter_summary so a
        # cold-KV stall (long time-to-first-token) is one-row visible.
        self._last_prefill_s = 0.0
        hb_state = {
            "started": _time.monotonic(),
            "last_hb": _time.monotonic(),
            "tokens": 0,
            "tail": "",
            # Phase 0.7 — fire-once flag so the slow-prefill surprise
            # event only emits once per stream when the condition first
            # holds; subsequent regular heartbeats keep flowing.
            "slow_prefill_emitted": False,
            # Phase 0.14 — fire-once flag for runaway-stream warning.
            "runaway_warned": False,
        }
        _STREAM_HEARTBEAT_SECONDS = 30.0
        _STREAM_HEARTBEAT_TAIL_CHARS = 120
        # Phase 0.7 — cold KV-cache stalls (cross-slot role switches with
        # no warm_prefix) produce minutes of near-zero token output. The
        # 2026-05-22 chess trace had iter 2 emit 1 token in 740s before
        # ramping. Threshold below auto-flags this in the trace so
        # future trace mining doesn't have to grep heartbeats by hand.
        _SLOW_PREFILL_TOK_FLOOR = 5
        _SLOW_PREFILL_ELAPSED_FLOOR = 120.0
        # Phase 0.14 — runaway-generation watchdog. The 2026-05-22 third
        # chess trace had a coder stream emit 36,736 completion tokens
        # over 25 minutes. The user couldn't tell whether the model was
        # making progress or stuck in a loop; their queued feedback sat
        # invisible the entire time. Threshold below is intentionally
        # conservative — a typical iter is 1–4k tokens, a giant
        # legitimate rewrite is ~10k. Past 15k something has usually
        # gone wrong. Fires-once, no behavior change — pure visibility.
        _RUNAWAY_TOKEN_FLOOR = 15000

        def _heartbeat_on_token(piece: str) -> None:
            if on_token is not None:
                try:
                    on_token(piece)
                except Exception:
                    pass
            hb_state["tokens"] += 1
            # Maintain a small tail buffer; cheap O(1) amortized.
            tail = hb_state["tail"] + piece
            if len(tail) > _STREAM_HEARTBEAT_TAIL_CHARS * 2:
                tail = tail[-_STREAM_HEARTBEAT_TAIL_CHARS * 2:]
            hb_state["tail"] = tail
            now = _time.monotonic()
            if now - hb_state["last_hb"] >= _STREAM_HEARTBEAT_SECONDS:
                hb_state["last_hb"] = now
                elapsed = now - hb_state["started"]
                tok_per_s = hb_state["tokens"] / elapsed if elapsed > 0 else 0.0
                self._trace({
                    "kind": "stream_heartbeat",
                    "tokens": hb_state["tokens"],
                    "elapsed_s": round(elapsed, 1),
                    "tok_per_s": round(tok_per_s, 2),
                    "tail": hb_state["tail"][-_STREAM_HEARTBEAT_TAIL_CHARS:],
                })
                if (
                    not hb_state["slow_prefill_emitted"]
                    and hb_state["tokens"] < _SLOW_PREFILL_TOK_FLOOR
                    and elapsed >= _SLOW_PREFILL_ELAPSED_FLOOR
                ):
                    hb_state["slow_prefill_emitted"] = True
                    self._last_prefill_s = round(elapsed, 1)
                    self._trace({
                        "kind": "slow_prefill",
                        "tokens": hb_state["tokens"],
                        "elapsed_s": round(elapsed, 1),
                        "model_role": role,
                        "model_name": getattr(
                            getattr(active_backend, "info", None),
                            "model",
                            "unknown",
                        ),
                        "hint": (
                            "tokens<5 after 120s+ usually means a cold KV "
                            "cache after a cross-slot role switch — "
                            "Backend.warm_prefix on the next role's slot "
                            "during the prior role's stream avoids it."
                        ),
                    })
                if (
                    not hb_state["runaway_warned"]
                    and hb_state["tokens"] >= _RUNAWAY_TOKEN_FLOOR
                ):
                    hb_state["runaway_warned"] = True
                    self._trace({
                        "kind": "runaway_stream_warning",
                        "tokens": hb_state["tokens"],
                        "elapsed_s": round(elapsed, 1),
                        "tok_per_s": round(tok_per_s, 2),
                        "model_role": role,
                        "model_name": getattr(
                            getattr(active_backend, "info", None),
                            "model",
                            "unknown",
                        ),
                        "hint": (
                            f"completion >{_RUNAWAY_TOKEN_FLOOR} tokens — "
                            "typical iter is 1-4k. Likely a token-repetition "
                            "loop, an oversized rewrite, or the model "
                            "concatenating multiple drafts. Press Ctrl+D to "
                            "ship the current best build and re-queue your "
                            "feedback if waiting feels wrong."
                        ),
                    })

        # Backend-reported pre-token progress (today only MLX surfaces
        # it — parsed from mlx_lm.server's SSE keepalive frames during
        # prompt processing). We trace it AND stash on the agent so
        # `chat.py`'s status panel can render "prompt eval N/M" instead
        # of a bare "waiting Ns" during the long pre-token wait.
        # Resets on every _stream call so stale progress from a prior
        # turn doesn't render forever.
        self._stream_progress_stage: str | None = None
        self._stream_progress_current: int = 0
        self._stream_progress_total: int = 0
        # Phase 0F: throttle the stream_progress TRACE (not the live stash the
        # status panel reads). The callback fires on every prompt-eval chunk —
        # ~40 near-identical rows per iter that the heartbeat already covers.
        # Trace only on stage change, quartile boundaries, and completion.
        _progress_trace_state = {"stage": None, "bucket": -1}

        def _on_progress(stage: str, current: int, total: int) -> None:
            self._stream_progress_stage = stage
            self._stream_progress_current = current
            self._stream_progress_total = total
            bucket = (current * 4 // total) if total else -1
            is_done = bool(total) and current >= total
            if (
                stage != _progress_trace_state["stage"]
                or bucket != _progress_trace_state["bucket"]
                or is_done
            ):
                _progress_trace_state["stage"] = stage
                _progress_trace_state["bucket"] = bucket
                self._trace({
                    "kind": "stream_progress",
                    "stage": stage,
                    "current": current,
                    "total": total,
                })

        # Rich first-build turns (no baseline file yet) can be long on
        # local models; give them a larger overall wall-clock cap while
        # keeping normal patch turns at the configured default.
        effective_overall_seconds = self.overall_seconds
        # Fix C: a seed iter-1 build is a "rich first turn" too (full seed in
        # the prompt + a multi-part edit goal) but it sets _current_file to the
        # seed HTML, so the no-baseline branch above misses it (trace
        # 194238 iter 1 ran 1750s vs the 1800s default — a near miss; a larger
        # edit gets cut mid-stream → no_usable_code). Give it the same headroom.
        seed_first_build = self.seed_file is not None and self._snapshot_n == 0
        no_baseline = self._current_file is None
        if no_baseline or seed_first_build:
            effective_overall_seconds = max(self.overall_seconds, 2400.0)
        if effective_overall_seconds != self.overall_seconds:
            self._trace({
                "kind": "stream_timeout_override",
                "base_overall_seconds": self.overall_seconds,
                "effective_overall_seconds": effective_overall_seconds,
                "reason": "no_baseline_file" if no_baseline else "seed_first_build",
            })

        result = None
        try:
            opts: dict[str, Any] = {"temperature": temp, "num_ctx": self.num_ctx}
            if self._restart_attempt_seed is not None:
                opts["seed"] = int(self._restart_attempt_seed)
            if getattr(active_backend, "info", None) and active_backend.info.name == "ollama":
                try:
                    import gpu_status as _gs
                    opts.update(_gs.ollama_chat_load_options())
                except Exception:
                    pass
            result = await active_backend.stream_chat(
                self._messages,
                on_token=_heartbeat_on_token,
                options=opts,
                keep_alive=self._keep_alive_for_backend(active_backend),
                stall_seconds=self.stall_seconds,
                overall_seconds=effective_overall_seconds,
                max_retries=1,
                on_stall=lambda r, attempt: self._trace({
                    "kind": "stream_stalled",
                    "attempt": attempt,
                    "tokens_before_stall": r.tokens,
                    "duration_s": r.duration_s,
                }),
                on_progress=_on_progress,
                cancel_event=self._ensure_stop_event(),
            )
        except Exception as exc:
            self._set_role_activity(role, f"failed ({type(exc).__name__})")
            raise
        finally:
            # Always remove our prefill scaffolding before returning so
            # the message history we save & feed to subsequent turns
            # contains a single coherent assistant message.
            if anthropic_prefill_folded and _orig_last_user_content is not None:
                # Restore the user message we mutated in place so the
                # saved history doesn't drift with appended format hints
                # across turns.
                if self._messages and self._messages[-1].get("role") == "user":
                    self._messages[-1] = {
                        **self._messages[-1],
                        "content": _orig_last_user_content,
                    }
            elif prefill_used and self._messages and self._messages[-1].get("role") == "assistant":
                self._messages.pop()
            if result is not None:
                if getattr(result, "crashed", False):
                    self._set_role_activity(role, "crashed")
                else:
                    self._set_role_activity(role, "idle")

        # Stash the streaming-abort signals so callers (the format-
        # rejection branch in run()) can consult them without
        # re-plumbing every _stream() call. Cleared at the top of every
        # _stream so a previous turn's signal doesn't leak.
        self._last_stream_looped = bool(result.looped)
        self._last_stream_stalled = bool(result.stalled)
        self._last_stream_deliberated = bool(result.deliberated)
        self._last_stream_crashed = bool(result.crashed)
        # Sticky session flag for offline credit eligibility (don't mark
        # playbook bullets harmful when the backend dies before code lands).
        if result.crashed:
            self._session_backend_crashed = True
        self._last_stream_silent = bool(getattr(result, "silent", False))
        self._last_stream_loop_kind = (
            getattr(result, "loop_kind", None) if result.looped else None
        )
        self._last_stream_loop_line = (
            getattr(result, "loop_line", None) if result.looped else None
        )
        # Phase 4 (4D.1): fold the just-finished stream's cost signal onto the
        # agent so `iter_summary` is self-contained (one-row diagnosis) without
        # the reviewing LLM joining `stream_done`. tok/s separates a slow LOCAL
        # MODEL (low tok/s) from a HARNESS stall (high tok/s but no output).
        self._last_stream_tokens = int(result.tokens or 0)
        self._last_stream_duration_s = round(result.duration_s, 2)
        self._last_stream_tok_per_s = (
            round(result.tokens / result.duration_s, 2)
            if result.duration_s and result.duration_s > 0 else 0.0
        )
        self._trace({
            "kind": "stream_done",
            "tokens": result.tokens,
            "duration_s": round(result.duration_s, 2),
            "stalled": result.stalled,
            "looped": result.looped,
            "deliberated": result.deliberated,
            "crashed": result.crashed,
            "silent": bool(getattr(result, "silent", False)),
            "len": len(result.text),
            # Backend-reported BPE counts when available. `tokens` above is
            # streaming chunk count; these are the real cost numbers used
            # to chart prompt size over a session and to spot when the
            # inlined-file truth source has bloated the input.
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "max_tokens_hit": result.max_tokens_hit,
        })
        # Context-pressure detector. When prompt_tokens approaches
        # num_ctx, the model has no headroom to emit a complete
        # <html_file>; output truncates, parser rejects, the agent
        # falls into an identical-reply loop. Universal mitigation:
        # set a one-shot flag the next fix_instruction call reads to
        # omit the inlined CURRENT FILE block and coach a minimal
        # patch. Trace fires only on the FIRST turn of a high-pressure
        # streak — subsequent turns in the same streak suppress the
        # trace to avoid log spam.
        try:
            _ptokens = (
                int(result.prompt_tokens) if result.prompt_tokens is not None
                else 0
            )
            _num_ctx = int(getattr(self, "num_ctx", 0) or 0)
        except Exception:
            _ptokens, _num_ctx = 0, 0
        if _ptokens > 0 and _num_ctx > 0 and role == "coder":
            _pressure = _ptokens / _num_ctx
            # Persist for token-aware compaction (_prune_messages): we only
            # throw away conversation history when the context window is
            # actually filling, not at an arbitrary message count. Lets a
            # 200k-ctx local model keep full history through a long feedback
            # session instead of losing the playbook / earlier user asks.
            self._last_prompt_tokens = _ptokens
            self._last_prompt_pressure = _pressure
            if _pressure >= 0.85:
                self._context_pressure_streak += 1
                self._context_pressure_pending = True
                if self._context_pressure_streak == 1:
                    self._trace({
                        "kind": "context_pressure_warning",
                        "prompt_tokens": _ptokens,
                        "num_ctx": _num_ctx,
                        "pressure": round(_pressure, 3),
                        "streak": self._context_pressure_streak,
                        "hint": (
                            "prompt_tokens >= 85% of num_ctx; the next "
                            "fix turn will omit the inlined CURRENT "
                            "FILE block and require a minimal patch. "
                            "This prevents the identical-reply loop "
                            "the Wolfenstein 2026-05-24 trace burned "
                            "5 iters on."
                        ),
                    })
            else:
                self._context_pressure_streak = 0
        # Silent-stream surface — when the new ollama_io / MLX guard
        # aborts a stream that produced ZERO visible content for >=180s
        # (all output went to a reasoning channel that surfaces as empty
        # `content`), emit a dedicated trace so the doom-iter-4 failure
        # class is visible in postmortem analysis. Recovery flows
        # through the existing "no usable code" path below since the
        # reply text is empty.
        if getattr(result, "silent", False):
            # Fix-round item 3: force a structured compaction before the
            # retry — resending the same giant prompt that just stalled
            # (61-69K tokens in trace 20260610_185238) stalls again.
            self._force_compact_after_stall = True
            self._trace({
                "kind": "stream_silent_aborted",
                "duration_s": round(result.duration_s, 2),
                "completion_tokens": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "model_role": role,
                "model_name": getattr(
                    getattr(active_backend, "info", None),
                    "model",
                    "unknown",
                ),
                "hint": (
                    "Stream produced zero visible content for >=180s. "
                    "Likely all generation went to a reasoning/thinking "
                    "channel that surfaces as empty `content`. The model "
                    "should be coached to start replies directly with an "
                    "opening tag like <patch> or <html_file>."
                ),
            })
        # Short-stream warning — symmetric to runaway_stream_warning.
        # Counterpart to the 2026-05-23 SOTA chess trace where a coder
        # role emitted 8 tokens in 1.74s and the agent accepted it as
        # a clean reply. Local 27B models also produce abnormally short
        # replies (cold KV cache, prompt formatting confusion, refusal
        # patterns). Model-agnostic: signal is the token count + role
        # combination, no model name involved.
        try:
            ctokens = (
                result.completion_tokens
                if result.completion_tokens is not None
                else result.tokens
            )
        except Exception:
            ctokens = result.tokens
        _SHORT_STREAM_FLOOR = 50
        _is_done_reply = (
            "<confirm_done/>" in result.text
            or "<done/>" in result.text
            or result.text.strip() == ""
        )
        if (
            role in ("coder", "architect")
            and not result.stalled
            and not result.looped
            and not result.crashed
            and not _is_done_reply
            and ctokens is not None
            and ctokens < _SHORT_STREAM_FLOOR
        ):
            self._trace({
                "kind": "short_stream_warning",
                "role": role,
                "completion_tokens": ctokens,
                "duration_s": round(result.duration_s, 2),
                "len": len(result.text),
                "tail": (result.text or "")[-160:],
                "hint": (
                    f"stream emitted <{_SHORT_STREAM_FLOOR} completion "
                    "tokens with no stall/loop/crash flag and no <done/> "
                    "marker. Likely a degenerate reply — context "
                    "confusion, format refusal, or the model declared "
                    "completion implicitly. Worth a coaching nudge or "
                    "Ctrl+D if this repeats."
                ),
            })
        if bool(getattr(result, "loop_grace_used", False)):
            self._trace({
                "kind": "loop_grace_used",
                "reason": getattr(result, "loop_grace_reason", None),
                "tokens": result.tokens,
                "len": len(result.text),
            })
        # Output-cap detection.
        #
        # Two paths, preferring the explicit signal when available:
        #   1. Cloud backends (Anthropic / OpenAI) populate
        #      result.max_tokens_hit directly from stop_reason /
        #      finish_reason. This is the exact signal — the model
        #      would have continued.
        #   2. Local backends (Ollama, MLX) don't expose a clean
        #      "cut by cap" boolean, so we keep the legacy heuristic:
        #      completion_tokens lands on a round power-of-2 cap AND
        #      no </html_file> closer is present.
        #
        # Either path queues a coaching message so the NEXT user turn
        # tells the model the cap was the failure cause and asks for a
        # smaller emission. Without this, Claude in the DK trace
        # 20260513_135011 spent iters 1/2/3 re-emitting truncated
        # 17 KB <html_file> rewrites with no idea the API was clipping
        # them.
        cap_hit_explicit = bool(result.max_tokens_hit)
        cap_hit_heuristic = bool(
            result.completion_tokens
            and result.completion_tokens in (
                # Cloud-typical caps:
                8192, 16384, 32768, 65536,
                # Local-model native-context caps:
                131072, 200000, 262144,
            )
            and "</html_file>" not in result.text
            and "<html_file>" in result.text
        )
        if cap_hit_explicit or cap_hit_heuristic:
            ct = result.completion_tokens or 0
            self._record(AgentEvent(
                "info",
                f"[yellow]reply hit max_tokens cap[/yellow] at "
                f"{ct} BPE tokens — output was truncated mid-stream "
                f"({'API stop_reason' if cap_hit_explicit else 'heuristic'}). "
                "Coaching the model to emit a smaller change next turn."
            ))
            # Queue an iter-1-of-the-recovery coaching message. The
            # deliberation guard's _pending_coaching pipeline renders
            # this in the next user turn just before the base
            # instruction (agent.py:1557). The guidance is the same
            # whichever signal fired — the cure (smaller output) is
            # identical.
            self._pending_coaching.append(
                "Your previous reply was cut off by the model's max_tokens "
                f"cap ({ct} BPE tokens emitted) before reaching the closing "
                "</html_file> or </patch> tag. Re-emit a SMALLER change "
                "this turn:\n"
                "  - If you can express the change with `<patch>` blocks, "
                "do that — they are typically 100-500 bytes each.\n"
                "  - If a full rewrite is unavoidable, DROP every JS / CSS "
                "comment, condense whitespace, and avoid duplicate helper "
                "functions. The goal is a working file in fewer tokens, "
                "not a polished one.\n"
                "Do NOT re-emit the same long reply — it will hit the "
                "same cap and produce another wasted iter."
            )
        if result.crashed:
            # The backend's generation raised mid-stream. The MLX
            # backend now formats the actual exception at the catch
            # site and surfaces it via `result.error_message`, and
            # also drops the loaded model + clears Metal cache so
            # the next stream re-enters the load path on a clean GPU.
            # So: print the REAL exception (no more hardcoded "Metal
            # OOM" guess) and one short recovery hint.
            err = (result.error_message or "").strip() or "(no exception text captured)"
            backend_name = (
                getattr(self._backend, "info", None)
                and self._backend.info.name
            ) or "unknown"
            if backend_name == "mlx":
                hint = (
                    "GPU state has been reset — the next turn will reload the "
                    "model on a clean Metal context. If it keeps crashing on "
                    "the same prompt: lower [b]MLX_MAX_TOKENS[/b], drop "
                    "[b]MLX_PREFILL_STEP_SIZE[/b] to 512, or raise "
                    "[b]iogpu.wired_limit_mb[/b] (Metal memory cap)."
                )
            else:
                hint = (
                    "Retry the turn. If it persists, check the backend's logs "
                    "and rate limits."
                )
            self._queue_stream_ui_event(AgentEvent(
                "error",
                f"[red]backend crashed mid-generation[/red] after "
                f"{result.tokens} tok / {result.duration_s:.0f}s.\n"
                f"  cause: {err}\n"
                f"  {hint}",
                {
                    "tokens_at_crash": result.tokens,
                    "duration_s": round(result.duration_s, 2),
                    "error_message": err,
                    "backend": backend_name,
                },
            ))
        # ---- Abort lifecycle (Phase 0F cross-reference) -------------------
        # Stream aborts originate in ollama_io.py detectors and land here:
        #   RepetitionDetector  -> result.looped     (loop_kind: inline_data_bloat
        #                          / adjacent_line_spam / short_line_loop; graced
        #                          while inside an open <patch>/<html_file>/markdown
        #                          patch block, bounded by _LOOP_GRACE_TOKEN_CEILING)
        #   DeliberationDetector -> result.deliberated (pre-tag prose, no output tag)
        #   silent-stream guard  -> result.silent      (zero visible content)
        # Each sets a `_last_stream_*` flag above, queues stall-aware coaching
        # below, and selects a recovery branch in `_no_usable_code_fallback`
        # (prior_stream_looped / prior_stream_deliberated / prior_stream_silent,
        # all art-aware via `art_pending`). On a feedback turn with no code, the
        # same flags arm the Phase 0D-4 same-iter auto-retry.
        if result.looped:
            # Visible to the user via the agent log so they understand why
            # the stream cut off mid-output. Trim trailing whitespace from
            # the partial text so downstream regexes see a clean tail.
            loop_kind = getattr(result, "loop_kind", None) or "unknown"
            loop_line = (getattr(result, "loop_line", None) or "").strip()
            self._queue_stream_ui_event(AgentEvent(
                "info",
                _repetition_loop_abort_message(
                    tokens=result.tokens,
                    duration_s=result.duration_s,
                    loop_kind=loop_kind,
                    loop_line=loop_line or None,
                ),
                # Structured payload so the TUI status panel can show a
                # sticky "Last stall" line, not just a scrolled-past log
                # row (asked for after the minecraft 20260621 trace where
                # the only record of the abort vanished up the log).
                {
                    "stall_reason": "repetition_loop",
                    "loop_kind": loop_kind,
                    "tokens": result.tokens,
                    "duration_s": round(result.duration_s, 1),
                },
            ))
        if result.deliberated:
            # A2: smaller LLMs sometimes ramble pre-tag for thousands of
            # tokens without ever emitting <patch> / <html_file>. We
            # aborted; queue a coaching message so the next user turn
            # tells the model to commit to one root cause + one patch.
            self._queue_stream_ui_event(AgentEvent(
                "info",
                f"Deliberation loop detected — {result.tokens} "
                "tokens of pre-tag reasoning with no <patch>/<html_file>. "
                "Aborted stream; coaching the model to skip the essay.",
                {
                    "stall_reason": "deliberation_loop",
                    "tokens": result.tokens,
                    "duration_s": round(result.duration_s, 1),
                },
            ))
            # Phase 0D-3: when the user asked for NEW ART (asset reprompt armed
            # or router allow_assets_block), the coaching MUST mention <assets>
            # and split the multi-part ask — otherwise the model re-deliberates
            # on "rotate vs regenerate" (the exact iter-4 essay). Name the exact
            # tags so the model does not fall back to markdown patches.
            _art_pending = bool(
                getattr(self, "_unhonored_asset_request", None)
                or (
                    self._feedback_route is not None
                    and self._feedback_route.get("allow_assets_block")
                )
            )
            if _art_pending:
                self._pending_coaching.append(
                    "Your last reply was pure reasoning prose with no output "
                    "tag — aborted by the deliberation guard. The user asked "
                    "for NEW ART. Do ONE thing this turn: emit an <assets> "
                    "block (a bare JSON array) naming a FEW new sprites — NOT "
                    "the whole roster — then optionally one small <patch> to "
                    "wire them. Patches MUST be wrapped in <patch>...</patch> "
                    "and assets in <assets>...</assets>; markdown "
                    "`SEARCH:/REPLACE:` is NOT parsed. No preamble, no "
                    "weighing of options, no inline data dumps."
                )
            else:
                self._pending_coaching.append(
                    "Your last reply was pure reasoning prose with no output "
                    "tag — aborted by the deliberation guard. Do NOT think out "
                    "loud this turn. Emit ONE line inside <diagnose>...</diagnose> "
                    "naming the single line:variable responsible, then ONE "
                    "<patch>...</patch> or <html_file>...</html_file>. No "
                    "preamble, no exploration."
                )
        if getattr(result, "diagnose_bloat", False):
            # Chess-trace fix: the model opened <diagnose> and never closed
            # it (iter-2: 674 s of unclosed diagnose + probe JSON, no patch).
            # Coach patch-first: a one-sentence diagnose, then commit to a
            # <patch> immediately.
            self._queue_stream_ui_event(AgentEvent(
                "info",
                f"Diagnose-bloat detected — {result.tokens} "
                "tokens inside an unclosed <diagnose> with no <patch>. "
                "Aborted stream; coaching the model to close it and patch.",
                {
                    "stall_reason": "diagnose_bloat",
                    "tokens": result.tokens,
                    "duration_s": round(result.duration_s, 1),
                },
            ))
            self._pending_coaching.append(
                "Your last reply opened <diagnose> and never closed it — "
                "aborted by the diagnose-bloat guard. This turn: write ONE "
                "short sentence inside <diagnose>...</diagnose> (close the "
                "tag!), then IMMEDIATELY emit ONE <patch>...</patch> against "
                "the current file. Do not list probes, do not re-plan, do not "
                "think out loud."
            )
        if (result.stalled or result.crashed) and not result.text.strip():
            # Backend-aware recovery hint. "num_ctx" is Ollama-specific;
            # MLX has its own knobs (MLX_MAX_TOKENS, Metal wired-memory
            # limit); cloud backends rarely zero-stall but if they do
            # it's usually a rate-limit / connectivity issue. DK trace
            # 20260513_153626 showed the generic message pointing at
            # num_ctx during an MLX run, which is the wrong knob.
            backend_name = (
                getattr(self._backend, "info", None)
                and self._backend.info.name
            ) or "unknown"
            if backend_name == "mlx":
                hint = (
                    "Try lowering MLX_MAX_TOKENS, raising "
                    "iogpu.wired_limit_mb (Metal memory), or restarting "
                    "the chat process to release stuck VRAM."
                )
            elif backend_name == "ollama":
                hint = (
                    f"Try a smaller context (num_ctx={self.num_ctx}) "
                    "or restart `ollama serve`."
                )
            elif backend_name in ("openai", "anthropic"):
                hint = (
                    f"Check network / API key / rate limits — "
                    f"{backend_name} should not zero-stall under normal "
                    "load."
                )
            else:
                hint = "Try a different model or restart the backend."
            # Phase 5b: report ACTUAL stream wall-clock, not the configured
            # stall ceiling. Trace 2 (chess 20260522_104235) had a 0.48s
            # Anthropic API failure surface as "stalling at 600.0s" because
            # we used self.stall_seconds. Real duration is far more useful
            # for diagnosis.
            actual_seconds = round(getattr(result, "duration_s", 0.0) or 0.0, 2)
            cause = ""
            if getattr(result, "error_message", None):
                cause = f" cause: {result.error_message}."
            raise RuntimeError(
                f"Model produced no tokens before stalling at "
                f"{actual_seconds}s on backend={backend_name}.{cause} "
                f"{hint}"
            )
        # Prepend the prefill so downstream parsers (regex for <plan>,
        # <diagnose>, etc.) match against the full intended output.
        return (prefill + result.text) if prefill_used else result.text

    # -- best-of-N for fix iterations --------------------------------------

    # ---- Phase 2A — independent-slot detection for fan-out ----------



    async def _materialize(
        self, reply: str, *, dry_run: bool = False
    ) -> tuple[str | None, str]:
        """Turn a model reply into the resulting file content.

        Three paths, tried in order:
          1. <patch> blocks → apply against current file on disk.
          2. <html_file>...</html_file> → use it directly.
          3. Neither → return (None, reason).

        Returns (final_html_or_None, human_message). If `dry_run`, we don't
        write to disk or update self._current_file — used for best-of-N
        scoring where we test against a temp file.
        """
        patches = extract_patches(reply)
        if patches:
            base = self._current_file or self._read_best_or_empty()
            if not base:
                # No baseline yet — we shouldn't be using patches. Reject.
                return None, "patch reply but no baseline file yet"
            res = apply_patches(base, patches)
            # Phase 0.5 — structured per-block outcome trace. The
            # patch_retry_instruction prompt already shows the model
            # nearest-anchor diagnostics; this records the same info
            # in JSONL so cross-session pattern mining (e.g. "the same
            # SEARCH for `let kbCursor` failed in 3 sessions — the
            # model has a persistent blind spot here") works.
            if not dry_run:
                try:
                    from patches import find_anchor as _find_anchor
                    failed_idxs = {i for (i, _p, _r) in res.failed}
                    blocks = []
                    for idx, p in enumerate(patches):
                        if idx in failed_idxs:
                            reason = next(r for (i, _p, r) in res.failed if i == idx)
                            search_head = (p.search or "").splitlines()[0] if p.search else ""
                            entry = {
                                "idx": idx,
                                "applied": False,
                                "kind": (
                                    "prepend" if p.is_prepend
                                    else ("delete" if p.is_delete else "edit")
                                ),
                                "search_head": search_head[:120],
                                "reason": reason.replace("\n", " ").strip()[:240],
                            }
                            if (p.search or "").strip():
                                anchor = _find_anchor(base, p.search)
                                if anchor:
                                    entry["nearest_anchor_preview"] = (
                                        anchor.splitlines()[0][:160]
                                    )
                            blocks.append(entry)
                        else:
                            blocks.append({
                                "idx": idx,
                                "applied": True,
                                "kind": (
                                    "prepend" if p.is_prepend
                                    else ("delete" if p.is_delete else "edit")
                                ),
                            })
                    self._trace({
                        "kind": "patch_outcome",
                        "applied": res.applied,
                        "failed": len(res.failed),
                        "total": len(patches),
                        "blocks": blocks,
                    })
                    # Phase 4 (4D.1): fold this iter's patch result onto the
                    # agent so iter_summary shows "applied N/M" inline.
                    self._last_patch_applied = res.applied
                    self._last_patch_total = len(patches)
                except Exception as e:
                    self._trace_exception("patch_outcome_error", e)
            if res.applied == 0:
                # Surface the FIRST per-patch reason in the materialize
                # message so the user log shows WHY (the model already
                # gets the full failure list via patch_retry_instruction).
                # The DK trace 20260513_153626 hit a malformed-delimiter
                # patch (extra `=======` line) and the user-visible log
                # just said "all 1 patches failed to apply" — debugging
                # required digging into the trace. Showing the reason
                # inline turns it into a 5-second triage.
                reason_tail = ""
                if res.failed:
                    first_reason = res.failed[0][2]
                    # Trim to one line, cap so the log row stays tidy.
                    one_line = first_reason.replace("\n", " ").strip()
                    if len(one_line) > 200:
                        one_line = one_line[:197] + "..."
                    reason_tail = f" — {one_line}"
                return None, (
                    f"all {len(patches)} patches failed to apply{reason_tail}"
                )
            if res.failed and not dry_run:
                # Partial-apply: still write what landed, but the caller
                # gets a non-empty failed list to retry on.
                pass
            # Pre-commit bracket validation: never commit a patched file
            # that turned a balanced baseline into a syntax-broken one.
            failed_idxs2 = {i for (i, _p, _r) in res.failed}
            applied_patches = [
                p for i, p in enumerate(patches) if i not in failed_idxs2
            ]
            bracket_reject = _patch_set_bracket_break(
                base, res.text, applied_patches,
            )
            if bracket_reject:
                if not dry_run:
                    self._trace({
                        "kind": "patch_bracket_reject",
                        "applied": res.applied,
                        "total": len(patches),
                        "reason": bracket_reject[:400],
                    })
                return None, bracket_reject
            # Harness honesty: a patch can report "applied" yet leave the
            # file byte-identical (REPLACE == SEARCH, or only whitespace-
            # normalized differences). That looks like progress in the log
            # but nothing changed — the 2026-06-21 seed trace showed
            # `applied 1/1` with an unchanged SHA. Surface a non-gating
            # advisory so "applied but nothing changed" is debuggable.
            if not dry_run and res.applied > 0 and res.text == base:
                self._trace({
                    "kind": "patch_noop",
                    "applied": res.applied,
                    "total": len(patches),
                    "note": "patches applied but file content unchanged",
                })
            return res.text, f"applied {res.applied}/{len(patches)} patches"

        html = self._extract_html(reply)
        if html is not None:
            normalized = _normalize_extracted_html(html)
            if normalized:
                html = normalized
            broken = _baseline_structurally_broken(html)
            if broken is not None:
                return None, (
                    f"<html_file> rejected: {broken}. "
                    "Emit ONE complete document with a single <script> "
                    "body (no concatenated drafts, no prose before "
                    "<!DOCTYPE). If the on-disk file is already broken, "
                    "a full rewrite is allowed when the baseline is "
                    "degenerate — otherwise use <patch>."
                )
            if _looks_like_placeholder_html_payload(html):
                return None, (
                    "<html_file> rejected: extracted body is a tiny placeholder "
                    "(e.g. `...`) rather than a real HTML document."
                )
            # Stop-Losing-To-OneShot: ban full <html_file> rewrites once
            # a baseline exists. The DOOM trace burned 5 consecutive iters
            # on truncated rewrites. Force the model into <patch> mode.
            # Escape hatch: AGENT_ALLOW_FULL_REWRITE=1 lets the rare
            # genuinely-structural rewrite through. dry_run is exempted
            # so best-of-N candidate scoring still works on iter 1.
            allow_rewrite = (
                os.environ.get("AGENT_ALLOW_FULL_REWRITE", "0").lower()
                in ("1", "true", "yes")
            )
            # Degenerate-baseline carve-out (classic-doom trace
            # 20260512_101944): iter 1 hit the MLX 16384-token cap and
            # only an 835-byte placeholder-comment skeleton landed on
            # disk. Iter 2 correctly diagnosed this and tried a full
            # rewrite — but the snapshot_n>=1 check above rejected it,
            # forcing patches against a file that had no real code to
            # anchor to. Recognize a non-runnable baseline and let the
            # rewrite through.
            baseline_degenerate = _is_degenerate_baseline(self._current_file)
            # Fix #3 (classic-doom 20260512_111015): one-shot exemption
            # armed by a fresh user-feedback drain. Multi-issue feedback
            # is what triggered the rewrite-rejection cascade in that
            # trace; let the model choose rewrite over fragile patches
            # for the first turn after feedback. Consume the flag once
            # we've decided to honor it — even if the rewrite fails the
            # bloat check below, we don't re-arm.
            feedback_exempt = bool(self._allow_one_rewrite) and not dry_run
            if feedback_exempt:
                self._allow_one_rewrite = False
                self._trace({"kind": "rewrite_exemption_consumed"})
            # Failing-baseline salvage (2026-06-10, both dojo-fight traces):
            # when the on-disk baseline is itself FAILING the harness
            # (previous report not ok), a structurally-clean inbound rewrite
            # is more valuable than the rejection — local models pushed into
            # rewrite mode do not switch back to <patch> on command
            # (DeepSeek AND Qwen each burned iters 4-6 on this exact
            # rejection). The inbound HTML already passed the duplicate-decl
            # micro-probes above, and still faces the bloat / skeleton
            # checks below. Rewrites on a WORKING baseline
            # (_previous_report_ok is True) stay banned — that is the
            # regression-amplification case the gate exists for.
            failing_baseline_salvage = (
                not dry_run
                and self._previous_report_ok is not True
            )
            if (
                not dry_run
                and self._current_file
                and self._snapshot_n >= 1
                and not allow_rewrite
                and not baseline_degenerate
                and not feedback_exempt
                and not failing_baseline_salvage
            ):
                return None, (
                    "<html_file> rejected: a baseline file already exists. "
                    "Send <patch> SEARCH/REPLACE blocks instead. (Override: "
                    "AGENT_ALLOW_FULL_REWRITE=1 — only when patches truly "
                    "cannot express the structural change.)"
                )
            if (
                failing_baseline_salvage
                and self._current_file
                and self._snapshot_n >= 1
                and not allow_rewrite
                and not baseline_degenerate
                and not feedback_exempt
            ):
                current_len = len(self._current_file or "")
                dropped_media_refs = (
                    (self._current_file or "").count("_assets/") > html.count("_assets/")
                    or (self._current_file or "").count("_videos/") > html.count("_videos/")
                    or (self._current_file or "").count("_sounds/") > html.count("_sounds/")
                )
                if current_len and (
                    len(html) < int(current_len * 0.80) or dropped_media_refs
                ):
                    return None, (
                        "<html_file> rejected: full rewrite on a failing "
                        "baseline would shrink the existing architecture or "
                        "drop generated media wiring. Repair the syntax/runtime "
                        "error with a minimal <patch> instead; keep existing "
                        "asset/sound/video loaders and scene structure unless "
                        "the user explicitly asked for a redesign."
                    )
                # The gate above would have rejected this rewrite before the
                # salvage rule; record that the salvage is what let it through.
                self._trace({
                    "kind": "rewrite_accepted_failing_baseline",
                    "previous_report_ok": self._previous_report_ok,
                    "html_bytes": len(html),
                })
            # Materialize-time bloat detector: even when the streaming
            # repetition detector lets a reply through, scan the final
            # HTML for duplicated blocks (typical: a maze 2D literal
            # emitted 3+ times, or `const T=16;` repeated 50 times).
            # Reject before writing to disk so the next user-turn names
            # the suspected duplication.
            bloat = _detect_block_bloat(html)
            if bloat is not None:
                return None, (
                    f"<html_file> rejected: detected duplicated block "
                    f"({bloat}). This typically means the model entered a "
                    f"regeneration loop on a large literal. Replace inline "
                    f"data with a seeded generator function and retry."
                )
            # Skeleton-payload detector (Item 1, trace
            # build-a-donkey-kong-clone-in-o_20260514_214747 iter 3):
            # the model emitted a 374-byte `<html_file>` body whose JS
            # was pseudocode comment-headers (`// Asset loading`,
            # `// Sound loading`, …) plus `{ ... }` placeholders, and
            # the harness wrote that to disk as the baseline. The
            # NEXT iter then had no real code to patch against and
            # the model burned another deliberation loop trying to
            # rebuild. Reject BEFORE writing so the prior real
            # baseline (if any) is preserved.
            skeleton = _detect_skeleton_payload(html)
            if skeleton is not None:
                return None, (
                    f"<html_file> rejected: body looks like a "
                    f"pseudocode skeleton, not a real game "
                    f"({skeleton}). Emit the COMPLETE implementation "
                    "this turn — every function body fully written, "
                    "no `{ ... }` placeholders, no `// Asset loading` "
                    "comment-headers without code beneath them. If "
                    "you cannot fit the whole file in one reply, "
                    "send a `<question>` to ask the user how to "
                    "narrow scope instead of shipping a stub."
                )
            return html, "full <html_file> rewrite"

        return None, "no <patch> or <html_file> in reply"

    @staticmethod


    @staticmethod
    def _truncation_diagnosis(reply: str) -> str | None:
        """If the reply looks like an HTML game cut off mid-stream, describe
        the truncation. Returns None when the reply contains a complete
        document — even if some outer wrapper tags are missing.

        The key check is `</html>`. If `</html>` is present, the HTML
        document itself is complete; a missing `</html_file>` or
        `</body>` is a wrapper-syntax detail, not a real truncation.
        DK trace 20260513_181731 burned an iter because the harness
        emitted "TRUNCATED REPLY — missing </html_file>" on a reply
        that contained a complete <!DOCTYPE html>...</html> body
        — calling that a stall was wrong.
        """
        low = reply.lower()
        has_doctype = "<!doctype" in low
        has_html_open = "<html" in low
        if not (has_doctype or has_html_open):
            return None
        # If the inner HTML document closed properly, the reply is
        # NOT truncated. Missing outer tags (</html_file>, </body>)
        # are wrapper artifacts the extractor handles. The script
        # check stays — an open <script> with no </script> means a
        # real cutoff that the extractor can't recover.
        if "</html>" in low and ("</script>" in low or "<script" not in low):
            return None
        ends = {
            "</html>": "</html>" in low,
            "</body>": "</body>" in low,
            "</script>": "</script>" in low or "<script" not in low,
        }
        missing = [tag for tag, present in ends.items() if not present]
        if not missing:
            return None
        return (
            f"reply began an HTML document ({len(reply):,} bytes streamed) "
            f"but was cut off — missing closing tags: {missing}. "
            f"Likely a stream stall mid-output."
        )

    @staticmethod
    def _extract_html(reply: str) -> str | None:
        reply = _strip_thinking(reply)
        return StreamMaterializeMixin._extract_html_inner(reply)

    @staticmethod
    def _extract_html_inner(reply: str) -> str | None:
        """Pull a complete HTML game out of a model reply.

        We accept six formats so we never throw away a valid game just
        because the model ignored the <html_file> anchor (a common failure
        mode of smaller models like qwen3.6:27b — they wrap output in a
        markdown fence or emit bare <!DOCTYPE>):

          1. <html_file>BODY</html_file>           ← preferred
          2. <html_file>```html\\nBODY\\n```        ← model double-wrapped
          3. <html_file>BODY (no closing tag, but BODY contains </html>)
             — common after stream stalls truncate the closing tag
          4. ```html\\n<!DOCTYPE html>...</html>\\n```   ← markdown fence only
          5. <!DOCTYPE html>...</html>             ← bare document
          6. <html>...</html>                       ← no doctype; we prepend
        """
        # 1. Canonical wrapper.
        m = _HTML_RE.search(reply)
        if m:
            body = m.group(1).strip()
            if body.startswith("```"):
                body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
                body = re.sub(r"\n?```$", "", body)
            body = body.strip()
            normalized = _normalize_extracted_html(body)
            if normalized:
                return normalized
        # 2/3. <html_file> opener but no proper close — pull the embedded doc.
        m = _UNCLOSED_HTML_FILE_RE.search(reply)
        if m:
            normalized = _normalize_extracted_html(m.group(1).strip())
            if normalized:
                return normalized
        # 4. Markdown fence whose contents look like HTML.
        for fm in _HTML_FENCE_RE.finditer(reply):
            inner = fm.group(1).strip()
            if "<html" in inner.lower() and "</html" in inner.lower():
                normalized = _normalize_extracted_html(inner)
                if normalized:
                    return normalized
        # 5. Bare doctype...html fragment anywhere in the reply.
        m = _BARE_DOCTYPE_RE.search(reply)
        if m:
            return m.group(1).strip()
        # 6. Bare <html>...</html> document with NO <!DOCTYPE> — salvage by
        # prepending a synthetic doctype line so the browser doesn't enter
        # quirks mode. Wolfenstein 2026-05-24 trace lesson: the model
        # occasionally emits a complete html element without the doctype
        # anchor; today classify_format_failure flags it (wrong_tag_html)
        # but the iter is wasted asking the model to retry. Salvaging here
        # turns the wasted iter into a working file.
        m = _BARE_HTML_ELEMENT_RE.search(reply)
        if m:
            body = m.group(1).strip()
            # Guard against picking up something tiny like an empty
            # <html></html> probe expression — a real document is at least
            # a few hundred bytes once script/style content is in it.
            if len(body) >= 200:
                return "<!DOCTYPE html>\n" + body
        return None

    @staticmethod
    def _extract_question(reply: str) -> str | None:
        reply = _strip_thinking(reply)
        m = _QUESTION_RE.search(reply)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_diagnose(reply: str) -> str | None:
        reply = _strip_thinking(reply)
        m = _DIAGNOSE_RE.search(reply)
        return m.group(1).strip() if m else None

    # ---- harness `warnings` persistence dedup --------------------------
    #
    # See `_warning_persistence` docstring in __init__. Threshold is the
    # iter on which we START compacting (so warnings are shown in full
    # for the first two iters, then compacted from the third onward).
    _WARNING_COMPACT_THRESHOLD: int = 3

    @staticmethod


    @staticmethod
    def _hash_warning(text: str) -> str:
        """Stable short hash for warning-string equality keying."""
        h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        return h[:16]



    def _advance_warning_persistence(self, warnings: list[str]) -> None:
        """Update per-warning consecutive-iter counts for the current
        iter's harness warnings. Call EXACTLY ONCE per iter — usually
        right after the test report is computed and before
        `_format_report_for_model` is called. Streak resets to zero
        when a previously-seen warning is absent this iter.
        """
        seen: set[str] = set()
        for w in warnings or []:
            h = self._hash_warning(str(w))
            seen.add(h)
            self._warning_persistence[h] = self._warning_persistence.get(h, 0) + 1
        # Drop counters for warnings not present this iter (streak broken).
        self._warning_persistence = {
            h: c for h, c in self._warning_persistence.items() if h in seen
        }



    @classmethod
    def _clean_actionable_vision_note(cls, note: str) -> str:
        """Return a model-facing vision note, or '' for fragments.

        The local VLM sometimes returns analysis scraps ("Image 1",
        "compare with the goal", unmatched parentheses). Those are
        useful in raw traces but harmful as prompt coaching for small
        local coding models. Keep this structural and domain-neutral.

        Beyond the structural filters, also drop notes that contain no
        actionable token (negation, change verb, modal, or comparative
        — see `_VISION_NOTE_ACTIONABLE_TOKENS`). Pure screenshot
        narration is one of the dominant noise sources from local VLMs
        and the model cannot act on it.
        """
        text = str(note or "").strip()
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
        if not text:
            return ""
        low = text.lower()
        if (
            low.startswith(("image ", "wait,", "let me "))
            or "compare with the goal" in low
            or "re-read the prompt" in low
        ):
            return ""
        words = re.findall(r"[A-Za-z0-9]+", text)
        if len(words) < 3:
            return ""
        if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
            return ""
        if words[-1].lower() in {
            "the", "a", "an", "and", "or", "but", "with", "from",
            "to", "of", "in", "on", "went", "is", "are",
        }:
            return ""
        # Relevance gate: keep only notes with at least one actionable
        # token. Generic list, not goal-derived. See class docstring
        # for `_VISION_NOTE_ACTIONABLE_TOKENS`. A defect-cue phrase
        # (e.g. "colored box", "magenta", "clipped") also qualifies — it
        # names a concrete visual defect even without a change verb, and
        # is the same cue set the parser uses to recover the finding.
        low_words = {w.lower() for w in words}
        if not (cls._VISION_NOTE_ACTIONABLE_TOKENS & low_words):
            try:
                from vision_judge import DEFECT_CUES
                if not any(cue in low for cue in DEFECT_CUES):
                    return ""
            except Exception:
                return ""
        return text

