"""VLM / visual playtest / autonomous critic for GameAgent.

Moved VERBATIM from `GameAgent` (no behavior change).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from backend import Backend
from memory import (
    ANIMATION_AUDITS_FILENAME,
    ASSET_AUDITS_FILENAME,
    PLAYTESTS_FILENAME,
    OpeningBookItem,
)
from tools import format_report_for_model
from agent import AgentEvent


class CriticMixin:

    """VLM / visual playtest / autonomous critic for GameAgent."""

    @staticmethod
    def _critic_note_fingerprint(text: str) -> str:

        """Deterministic short fingerprint for a visual-critic note.



        Used by `_recent_critic_note_fingerprints` to detect cross-turn

        repeats. Normalizes whitespace + lowercases + takes first 120

        chars so semantically-identical notes ("the wall textures are

        low-resolution" vs "Wall textures appear low-resolution") with

        small wording shifts still match.

        """

        import hashlib as _hashlib

        normalized = " ".join((text or "").lower().split())

        head = normalized[:120].encode("utf-8", "ignore")

        return _hashlib.sha1(head).hexdigest()[:12]

    def _ensure_local_vlm_path(self) -> bool:
        """Resolve local VLM once per session.

        Honors SMOKE_VLM_MODEL / MLX_VLM_MODEL (same pin as smoke scripts)
        before discover_local_vlm().
        """
        if self._local_vlm_path is not None:
            return bool(self._local_vlm_path)
        resolved = None
        query = os.environ.get("SMOKE_VLM_MODEL") or os.environ.get("MLX_VLM_MODEL")
        if query:
            try:
                from vision_judge import _resolve_local_mlx_vlm

                resolved = _resolve_local_mlx_vlm(query)
            except Exception:
                resolved = None
        if not resolved:
            try:
                from backend import discover_local_vlm

                resolved = discover_local_vlm()
            except Exception:
                resolved = None
        self._local_vlm_path = resolved or ""
        if resolved:
            self._trace({"kind": "vision_judge_local_vlm", "path": resolved})
        return bool(self._local_vlm_path)

    _MIN_VLM_SCREENSHOT_BYTES = 12 * 1024

    def _vlm_screenshot_untrusted_reason(self) -> str | None:
        """Return why the last iter screenshot must not drive VLM coaching.

        Magenta MISSING boxes / pre-decode frames make Q4 'face each other'
        fail for the wrong reason and poison the coder on the next turn.
        """
        report = self._last_test_report or {}
        decode = report.get("asset_decode_settle") or {}
        if not decode.get("skipped"):
            if decode.get("need") is True and decode.get("ready") is not True:
                return "asset_decode_settle not ready"
        missing = (report.get("entity_render_check") or {}).get("missing") or []
        if missing:
            return f"entity_render_check missing ({len(missing)} entity/ies)"
        canvas = report.get("canvas") or {}
        if canvas.get("blank") is True:
            return "canvas blank"
        shot = self._last_screenshot_after or b""
        if shot and len(shot) < self._MIN_VLM_SCREENSHOT_BYTES:
            return f"screenshot too small ({len(shot)} bytes)"
        return None

    def _queue_visual_critic_coaching(self, cleaned: str, *, iteration: int, vc_role: str = "critic") -> bool:

        """Append a visual-critic note to `_pending_coaching` with cross-turn dedup.



        Returns True when the note was queued, False when suppressed as

        a repeat of one we already queued in the last 3 critic turns.

        """

        untrusted = self._vlm_screenshot_untrusted_reason()
        if untrusted:
            self._trace({
                "kind": "vlm_critique_skipped_untrusted_screenshot",
                "iteration": iteration,
                "vc_role": vc_role,
                "reason": untrusted,
                "preview": cleaned[:200],
            })
            return False

        prefix = "[VLM-CRITIQUE] "

        full = prefix + cleaned

        # Polish phase (item 2): remember the latest finding regardless of

        # dedup so polish_instruction can surface it.

        self._last_critic_note = cleaned

        fp = self._critic_note_fingerprint(cleaned)

        if fp in self._recent_critic_note_fingerprints:

            # Fix-round item 6: remember the payload that produced this

            # suppressed critique so run_visual_critic can skip an

            # identical VLM call next iter (payload-fingerprint dedupe).

            self._suppressed_critic_payload_fp = getattr(

                self, "_current_critic_payload_fp", None,

            )

            self._trace({

                "kind": "coaching_suppressed_repeated",

                "iteration": iteration,

                "vc_role": vc_role,

                "fingerprint": fp,

                "preview": cleaned[:200],

                "reason": (

                    "visual critic emitted the same observation in the "

                    "last 3 critic turns; suppressing to avoid prompt "

                    "bloat. The coder either can't fix it (asset-side) "

                    "or already addressed it and the critic misread the "

                    "new screenshot."

                ),

            })

            return False

        self._pending_coaching.append(full)

        self._recent_critic_note_fingerprints.append(fp)

        # Loop visibility: the vision critique looked at the screen and is

        # handing the problem to the coder, which fixes it on the next turn.

        self._trace({

            "kind": "vlm_critique_finding_sent_to_coder",

            "vision": True,

            "iteration": iteration,

            "vc_role": vc_role,

            "preview": cleaned[:200],

        })

        return True



    async def _detect_vlm(self, role: str = "coder") -> bool:

        backend = self.get_backend(role)

        if not backend:

            return False

        if not hasattr(self, "_vlm_cache"):

            self._vlm_cache = {}

        if backend in self._vlm_cache:

            val = self._vlm_cache[backend]

            if role == "coder":

                self._is_vlm = val

            return val

        val = await backend.is_vlm()

        self._vlm_cache[backend] = val

        # `self._is_vlm` is read by /ref (chat.py) and the fix-prompt

        # VLM_REVIEW_NOTE gate (agent.py:12195). Without this assignment

        # `_is_vlm` stays None for the session and /ref always warns

        # "active model is text-only" even on Claude Opus / VLM locals.

        if role == "coder":

            self._is_vlm = val

        if val:

            self._trace({"kind": "vlm_detected", "model": backend.info.model, "role": role})

        return val



    def _maybe_inject_visual_playtest_auto_probes(self) -> None:

        """Inject auto-probes from the matched visual_playtest recipe

        into `self._probes`. Idempotent — checks names against the

        existing probe set so repeated calls don't duplicate entries.



        Auto-probes are the DETERMINISTIC layer paired with the VLM

        checklist: the VLM might miss "both characters face the same

        direction" on a screenshot, but the `auto_actors_face_each_other`

        probe asserts `Math.sign(p1.facing) !== Math.sign(p2.facing)`

        directly against state — fails any iter where the wholesale

        flip lands. Mortal-kombat 2026-05-24 iter 12 is the motivating

        case.



        Conservative: each probe returns `true` (passes) when the

        relevant state shape isn't exposed. Never fails a game that

        simply doesn't have e.g. `state.player.facing`.



        Called once at end of Phase A (after the model's own probes

        are parsed) and again after first-build materializes (in case

        the planning phase had no probes but assets give a stronger

        recipe match).

        """

        try:

            recipe, diag = self._memory.find_visual_playtest_for(

                goal=self._goal or "",

                plan_text=self._criteria or "",

                asset_names=list(self._session_assets.keys()),

                code=getattr(self, "_current_file", "") or "",

            )

        except Exception as e:

            self._trace({

                "kind": "visual_playtest_auto_probes_error",

                "error": str(e)[:200],

            })

            return

        if recipe is None:

            return

        # Record the matched recipe id so the TUI status panel can show

        # which mechanism is guiding this session, even on recipes that

        # don't carry any auto_probes.

        self._active_visual_playtest_recipe_id = recipe.id

        # Capture the still-frame cadence flag so the report post-processor

        # can neutralize the FROZEN-AT-IDLE "add a breathing bob" coaching

        # for laserdisc/cutscene games where static frames are intended.

        self._active_visual_playtest_still_frame = bool(

            recipe.recipe.get("still_frame")

        )

        auto = recipe.recipe.get("auto_probes") or []

        if not auto:

            return

        existing_names = {(p.get("name") or "") for p in (self._probes or [])}

        added: list[str] = []

        for ap in auto:

            name = (ap.get("name") or "").strip()

            expr = (ap.get("expr") or "").strip()

            if not name or not expr or name in existing_names:

                continue

            if self._probes is None:

                self._probes = []

            self._probes.append({"name": name, "expr": expr})

            existing_names.add(name)

            added.append(name)

        if added:

            # Append rather than overwrite — same recipe re-matches

            # on subsequent retrieval calls and we want the union.

            for n in added:

                if n not in self._active_visual_playtest_auto_probes:

                    self._active_visual_playtest_auto_probes.append(n)

            self._trace({

                "kind": "visual_playtest_auto_probes_injected",

                "recipe_id": recipe.id,

                "added": added,

                "total_probes": len(self._probes or []),

            })



    def _maybe_inject_media_probes(self) -> None:

        """Phase 3: inject deterministic media-wiring probes for the assets /

        sounds / videos this session generated, so a missing-wiring gap is

        caught by the verifier instead of waiting for the user to report it.



        Conservative by design — only probes with a CLEAN in-page signal are

        added, and each passes-when-uncertain so it never false-fails a good

        game (probes gate ok). Drawn-ratio / load-failure / procedural-

        regression coverage stays in tools.py's `drawn_asset_check` /

        `missing_asset_load_check` advisory channels (golden-iter-2 safe);

        duplicating them here as gating probes would risk false negatives.

        """

        # 1. Asset placeholders (MISSING <key>) — existing, kept.

        self._maybe_inject_asset_miss_probe()

        if self._probes is None:

            self._probes = []

        existing_names = {(p.get("name") or "") for p in self._probes}

        # 2. Video present — when the session generated cutscene MP4s, a build

        # that plays them must mount a <video> element. Deterministic: fails

        # only when videos exist on disk but no <video> tag is in the DOM.

        if self._session_videos and "auto_video_present" not in existing_names:

            self._probes.append({

                "name": "auto_video_present",

                "expr": "!!document.querySelector('video')",

            })

            self._trace({

                "kind": "media_probe_injected",

                "name": "auto_video_present",

                "total_probes": len(self._probes),

            })



    def _maybe_inject_asset_miss_probe(self) -> None:

        """Inject a deterministic probe that fails when the game draws a

        `MISSING <key>` placeholder for a generated asset.



        The recommended `sprite()` helper records unresolved keys to

        `window.__assetMisses` (see assets.py). A non-empty map means the

        code asked for an asset that didn't resolve (key drift / loader

        not awaited) and drew a loud placeholder instead of the art —

        exactly the bug the user had to report manually in the

        dragon's-lair 2026-06-14 trace.



        Gated on `self._session_assets`: only games that generated assets

        can miss one. Conservative: passes (true) when `__assetMisses` is

        undefined (helper not used, or nothing drawn yet).

        """

        if not getattr(self, "_session_assets", None):

            return

        name = "auto_no_missing_asset_placeholders"

        existing_names = {(p.get("name") or "") for p in (self._probes or [])}

        if name in existing_names:

            return

        expr = (

            "(()=>{const m=window.__assetMisses;"

            "return !m||Object.keys(m).length===0;})()"

        )

        if self._probes is None:

            self._probes = []

        self._probes.append({"name": name, "expr": expr})

        self._trace({

            "kind": "asset_miss_probe_injected",

            "name": name,

            "total_probes": len(self._probes or []),

        })



    def _animation_expected(self) -> bool:

        """True when the goal/session implies animated character motion.



        Signal-driven: the model declared from_image frames, frames came back

        dead, OR the goal/criteria imply game actions. No genre/verb table.

        """

        if self._declared_anim_frames or self._dead_anim_frames:

            return True

        try:

            from tools import expects_game_controls as _egc

            return bool(_egc(self._goal or "", self._criteria or ""))

        except Exception:

            return False



    def _augment_recipe_for_animation(self, recipe):

        """Append a context-specific animation question to a recipe's checklist.



        The VLM (not a hard-wired rule) identifies the motion the GOAL describes

        — walk, kick, punch, … — and judges whether the body pose actually

        changes vs. the same sprite sliding. Returns a CLONE so the cached

        recipe is untouched; returns the original when no animation is expected

        or the recipe already asks about it.

        """

        if recipe is None or not self._animation_expected():

            return recipe

        existing = " ".join(recipe.recipe.get("checklist") or []).lower()

        if "same sprite" in existing or "mid-stride" in existing or "same character mid-move" in existing:

            return recipe  # recipe already covers animation (e.g. fighters)

        q = (

            "When the character moves or acts, do its body/limbs visibly change "

            "pose between the resting frame and the action frame — performing the "

            "SPECIFIC motion the goal describes (walking: legs in a different "

            "mid-stride position; kicking: a leg extended; punching: an arm "

            "extended) — rather than the SAME sprite image just repositioned? "

            "Answer NO if it is the identical picture slid across the screen."

        )

        clone = copy.copy(recipe)

        base = dict(recipe.recipe or {})

        base["checklist"] = list(base.get("checklist") or []) + [q]

        anim_hint = (

            "Sliding/dead animation is a COSMETIC sprite-quality note, not a "

            "code bug: do NOT redraw limbs in code and do NOT try to regenerate "

            "the pose frames (img2img can't change a pose, and a fresh txt2img "

            "frame won't stay consistent with the character already in the "

            "game). Cycle whatever motion frames exist over the active window "

            "and keep going — this does not block shipping."

        )

        base["fix_hint"] = (

            (base.get("fix_hint") or "").strip() + "\n" + anim_hint

        ).strip()

        clone.recipe = base

        return clone



    @staticmethod

    def _action_frame_question_indices(checklist: list) -> list[int]:

        """0-based indices of checklist questions that can ONLY be judged

        against a captured mid-action frame (fix-round item 6).



        Sniff is textual and mechanism-level: the question references the

        ACTION frame / mid-input image. When no action frame was captured,

        these questions are unanswerable — the anti-rubber-stamp NOTE then

        forces a deterministic FAIL that names a harness capture gap, not a

        game bug (trace 20260610_185238 failed Q5 identically 6 times).

        Pure function."""

        out: list[int] = []

        for i, q in enumerate(checklist or []):

            ql = str(q).lower()

            if "action frame" in ql or "mid-input" in ql or "mid-action" in ql:

                out.append(i)

        return out



    def _strip_action_frame_questions(self, recipe):

        """Return (recipe', skipped_questions). When no action frame exists,

        drop action-frame-dependent questions from a CLONE of the recipe so

        prompt + parser + formatter all see the reduced list; per-question

        fix_hints are renumbered to match. The VLM cannot answer YES to a

        question it never sees, so the 2026-05-29 anti-rubber-stamp

        guarantee is preserved — without converting a harness capture gap

        into a repeated fake game-bug. Returns the original recipe untouched

        when nothing needs stripping."""

        checklist = list(recipe.recipe.get("checklist") or [])

        idxs = set(self._action_frame_question_indices(checklist))

        if not idxs:

            return recipe, []

        skipped = [checklist[i] for i in sorted(idxs)]

        kept = [(i, q) for i, q in enumerate(checklist) if i not in idxs]

        clone = copy.copy(recipe)

        base = dict(recipe.recipe or {})

        base["checklist"] = [q for _, q in kept]

        per_q = base.get("fix_hints")

        if isinstance(per_q, dict) and per_q:

            remapped: dict[str, str] = {}

            for new_i, (old_i, _q) in enumerate(kept, start=1):

                h = per_q.get(str(old_i + 1)) or per_q.get(old_i + 1)

                if h:

                    remapped[str(new_i)] = h

            base["fix_hints"] = remapped

        clone.recipe = base

        return clone, skipped

    def _adjust_checklist_for_goal(self, recipe):
        """Clone recipe checklist when goal says no HUD — drop health-bar Qs."""
        if recipe is None:
            return recipe
        from memory import filter_vlm_checklist_for_goal

        checklist = list(recipe.recipe.get("checklist") or [])
        filtered = filter_vlm_checklist_for_goal(checklist, self._goal or "")
        if filtered == checklist:
            return recipe
        clone = copy.copy(recipe)
        base = dict(recipe.recipe or {})
        base["checklist"] = filtered
        clone.recipe = base
        return clone

    @classmethod

    def _critic_abstained(cls, text: str) -> bool:

        """True ONLY when the reply indicates the model never received/saw the

        IMAGE itself — not when it reports not seeing a game element (which is a

        legitimate visual finding). See _CRITIC_ABSTAIN_RE for why this is

        anchored to image/screenshot objects."""

        return bool(text) and bool(cls._CRITIC_ABSTAIN_RE.search(text))



    @staticmethod

    def _parse_visual_playtest_response(text: str, recipe) -> dict:

        """Parse a structured-checklist response into {q_index: (answer, remark)}.



        Tolerant: accepts YES/NO/UNCLEAR in any case, with or without

        leading whitespace; accepts `Qn:`, `Qn.`, `Qn -`, etc. Lines

        that don't match the pattern are skipped silently. Returns a

        dict keyed by 1-based question index.



        Returns also a `parse_rate` (matched / expected) so the

        caller can detect low-quality responses and fall back.

        """

        if not text or not recipe:

            return {"answers": {}, "parse_rate": 0.0, "n_questions": 0}

        checklist = recipe.recipe.get("checklist") or []

        n_q = len(checklist)

        answers: dict[int, tuple[str, str]] = {}

        import re as _re

        # Match `Q1: yes — remark` / `Q12. NO` / `q3 - unclear` / etc.

        # The `\b` after the answer alternation forces a word break, but

        # emoji code-points don't trigger \b in Python's re — so we split

        # the alternation into two branches: ASCII words (need \b) and

        # symbols (no \b).

        pat = _re.compile(

            r"^\s*Q?\s*(\d+)\s*[:.\-)]\s*"

            # Tolerate an optional repeated ordinal the model sometimes emits

            # after a "Qn: " prefill (e.g. "Q1: 1. YES") — skip a leading

            # "<num>." / "<num>)" before the answer word. Added 2026-06-03.

            r"(?:\d+\s*[:.)]\s*)?"

            r"(?:(yes|no|unclear|y|n|u)\b|([✅❌✔✖✓✗✘]))"

            r"\s*[\-—]?\s*(.*)$",

            _re.IGNORECASE,

        )

        for line in text.splitlines():

            m = pat.match(line)

            if not m:

                continue

            idx = int(m.group(1))

            if idx < 1 or idx > n_q:

                continue

            ans_word = m.group(2)

            ans_sym = m.group(3)

            ans = (ans_word or ans_sym or "").strip().lower()

            remark = (m.group(4) or "").strip()

            # Normalize to yes/no/unclear.

            if ans in ("y", "yes", "✅", "✔", "✓"):

                norm = "yes"

            elif ans in ("n", "no", "❌", "✖", "✗", "✘"):

                norm = "no"

            else:

                norm = "unclear"

            answers[idx] = (norm, remark)

        return {

            "answers": answers,

            "parse_rate": (len(answers) / n_q) if n_q else 0.0,

            "n_questions": n_q,

        }



    @staticmethod

    def _format_visual_playtest_critique(parsed: dict, recipe) -> str | None:

        """Turn the parsed checklist results into a single critique

        string. Returns None when EVERY check passed (no coaching

        needed — same shape as the legacy "Visual Critic: OK" path).

        """

        answers = parsed.get("answers") or {}

        n_q = parsed.get("n_questions") or 0

        if not answers or not n_q:

            return None

        checklist = recipe.recipe.get("checklist") or []

        failures: list[str] = []

        unclears: list[str] = []

        failed_idxs: list[int] = []

        for i, q in enumerate(checklist, start=1):

            entry = answers.get(i)

            if entry is None:

                continue

            ans, remark = entry

            line_intro = f"Q{i} ({q[:80]}{'...' if len(q) > 80 else ''})"

            if ans == "no":

                tail = f" — {remark}" if remark else ""

                failures.append(f"{line_intro} FAILED{tail}")

                failed_idxs.append(i)

            elif ans == "unclear":

                tail = f" — {remark}" if remark else ""

                unclears.append(f"{line_intro} UNCLEAR{tail}")

        if not failures and not unclears:

            return None  # all-pass; caller treats as "OK"

        head = (

            f"[VLM-CRITIQUE — {recipe.id}] "

            f"{len(failures)} of {n_q} check(s) failed"

            + (f" + {len(unclears)} unclear" if unclears else "")

            + ":"

        )

        body = "\n".join(failures + unclears)

        # Playbook cross-links for failed checklist items.

        playbook_refs = recipe.recipe.get("playbook_refs") or {}

        ref_ids: list[str] = []

        if isinstance(playbook_refs, dict):

            for idx in failed_idxs:

                for bid in playbook_refs.get(str(idx)) or playbook_refs.get(idx) or []:

                    s = str(bid).strip()

                    if s and s not in ref_ids:

                        ref_ids.append(s)

        playbook_tail = ""

        if ref_ids:

            playbook_tail = (

                "\n\nPlaybook: "

                + ", ".join(ref_ids)

                + " (use <lookup_bullet>id</lookup_bullet> to fetch bodies)"

            )

        # Fix-hint. Prefer a PER-QUESTION map (`fix_hints`: {q_index: hint}) so

        # we surface ONLY the advice for checks that actually FAILED. The old

        # behavior dumped the whole blob `fix_hint` on any failure — which made

        # the model apply facing-flip advice (ctx.scale(-1,1)) even when facing

        # PASSED and only Q4/Q5 failed, breaking correct facing (the iter1✓→

        # iter2✗→iter3✓ oscillation observed 2026-06-03 on the two-kickers run).

        # Falls back to the blob `fix_hint` when no per-question map exists.

        if failures:

            per_q = recipe.recipe.get("fix_hints")

            if isinstance(per_q, dict) and per_q:

                hints = []

                for i in failed_idxs:

                    h = (per_q.get(str(i)) or per_q.get(i) or "").strip()

                    if h and h not in hints:

                        hints.append(h)

                if hints:

                    return (

                        head + "\n" + body + "\n\nMinimal fix shape:\n"

                        + "\n".join(

                            f"  - (Q{idx}) {h}" for idx, h in zip(failed_idxs, hints)

                        )

                        + playbook_tail

                    )

            fix_hint = (recipe.recipe.get("fix_hint") or "").strip()

            if fix_hint:

                return head + "\n" + body + "\n\nMinimal fix shape: " + fix_hint + playbook_tail

        return head + "\n" + body + playbook_tail



    # Patterns stripped from /ask replies — read-only turn must never

    # materialize code even if the model disobeys the prompt contract.

    _ASK_FORBIDDEN_TAG_RES: tuple = (

        re.compile(r"<patch>.*?</patch>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<html_file>.*?</html_file>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<assets>.*?</assets>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<sounds>.*?</sounds>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<videos>.*?</videos>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<diagnose>.*?</diagnose>", re.DOTALL | re.IGNORECASE),

        re.compile(r"<done\s*/>", re.IGNORECASE),

        re.compile(r"<confirm_done\s*/>", re.IGNORECASE),

    )



    @staticmethod

    def _sanitize_ask_reply(text: str) -> tuple[str, bool]:

        out = text or ""

        stripped = False

        for pat in CriticMixin._ASK_FORBIDDEN_TAG_RES:

            if pat.search(out):

                stripped = True

                out = pat.sub("", out)

        return out.strip(), stripped



    def _ask_html_excerpt(self, html: str, report: dict | None) -> str:

        """Bounded best.html slice for /ask grounding."""

        max_chars = 10_000

        if not html:

            return "(no file)"

        if len(html) <= max_chars:

            return html

        report = report or {}

        if not report.get("ok", True):

            try:

                slice_text = self._focused_slice(

                    html, report, self._criteria or "",

                )

                if slice_text and len(slice_text) <= max_chars:

                    return slice_text

            except Exception:

                pass

        return (

            html[:max_chars]

            + f"\n\n[... truncated {len(html) - max_chars} chars ...]"

        )



    async def run_ask_turn(self, question: str) -> AsyncIterator[AgentEvent]:

        """One read-only Q&A turn for `/ask` — no harness, no code changes."""

        question = (question or "").strip()

        if not question:

            yield self._record(AgentEvent("error", "empty question"))

            return

        if not self.best_path.exists():

            yield self._record(AgentEvent(

                "error",

                "nothing built yet — run at least one iteration, then /ask",

            ))

            return



        try:

            html = self.best_path.read_text(encoding="utf-8")

        except Exception as e:

            yield self._record(AgentEvent(

                "error", f"could not read {self.best_path.name}: {e}",

            ))

            return



        report = self._last_test_report or {}

        report_text = (

            self._format_report_for_model(report) if report else ""

        )

        html_excerpt = self._ask_html_excerpt(html, report)

        ask_prompt = self._p.ask_instruction(

            question=question,

            goal=self._goal or "",

            criteria=self._criteria or "",

            report_text=report_text,

            html_excerpt=html_excerpt,

            asset_names=sorted(self._session_assets.keys()),

        )



        snapshot_n = self._snapshot_n

        saved_messages = list(self._messages)

        yield self._record(AgentEvent(

            "activity", "streaming",

            {"label": "ask reply", "role": "coder"},

        ))

        reply = ""

        try:

            # Lean context: ask_instruction already embeds goal, report, and
            # best.html excerpt — replaying the full iter history only bloats
            # prefill (torch-dungeon 20260701: 78k history + 13k ask prompt).
            system_msgs = [
                m for m in saved_messages if m.get("role") == "system"
            ][:1]
            ask_messages = list(system_msgs) + [{

                "role": "user",

                "content": ask_prompt,

                "phase": "ask",

            }]
            history_chars_dropped = sum(
                len(str(m.get("content") or ""))
                for m in saved_messages
                if m not in system_msgs
            )
            self._trace({

                "kind": "user_ask_context",

                "history_chars_dropped": history_chars_dropped,

                "ask_prompt_chars": len(ask_prompt),

                "message_count": len(ask_messages),

            })
            self._messages = ask_messages

            reply = await self._stream(

                self._token_cb_wrapper,

                role="coder",

                override_temp=0.3,

            )

            for _ev in self._drain_stream_ui_events():

                yield _ev

        except Exception as e:

            self._trace({"kind": "user_ask_error", "err": str(e)[:300]})

            yield self._record(AgentEvent("error", f"ask turn failed: {e}"))

            yield self._record(self._activity_idle_event("coder"))

            return

        finally:

            self._messages = saved_messages



        if self._snapshot_n != snapshot_n:

            self._trace({

                "kind": "user_ask_snapshot_guard",

                "before": snapshot_n,

                "after": self._snapshot_n,

            })



        if not (reply or "").strip():

            yield self._record(AgentEvent("error", "model returned nothing"))

            yield self._record(self._activity_idle_event("coder"))

            return



        clean_reply, tags_stripped = self._sanitize_ask_reply(reply)

        if not clean_reply:

            clean_reply = (

                "(model reply contained only code tags — omitted on this "

                "read-only ask turn)"

            )

        self._trace({

            "kind": "user_ask",

            "question": question[:500],

            "reply": clean_reply,

            "reply_preview": clean_reply[:500],

            "reply_chars": len(clean_reply),

            "tags_stripped": tags_stripped,

            "snapshot_n": self._snapshot_n,

            "phase": "ask",

        })

        yield self._record(AgentEvent(

            "info",

            clean_reply,

            {"ask": True, "tags_stripped": tags_stripped},

        ))

        yield self._record(self._activity_idle_event("coder"))



    async def run_visual_critic(

        self,

        current_png: bytes,

        before_png: bytes | None = None,

        action_png: bytes | None = None,

    ) -> str | None:

        """Run the configured out-of-band Visual Critic model on current_png.



        `action_png`, when present, is the frame the harness captured at the

        moment a held control produced its largest canvas change — the game

        mid-ACTION rather than at rest. It is supplied as a 3rd image so the

        critic can judge whether a deliberate action animation actually

        renders, instead of forever returning UNCLEAR on a resting frame.

        """

        backend = self.get_backend("critic")

        if backend is None:

            return None



        # Vision guard (added 2026-06-02, dragons-lair /allroles trace). The

        # visual critic feeds the model screenshots, so the backend MUST

        # actually serve vision. A text-only model handed an image does not

        # error — it HALLUCINATES a confident description of pixels it never

        # saw (verified: DeepSeek-V4-Flash answered a green-on-red circle with

        # "a single solid black circle"). That fabricated critique then gets

        # parsed and fed into the coaching loop as if real. So if the backend

        # can't do vision, skip the visual critic entirely and let the

        # behavioral probes carry verification. Genre-free, model-agnostic.

        try:

            if not await backend.is_vlm():

                self._trace({

                    "kind": "visual_critic_skipped",

                    "reason": "backend_not_vlm",

                    "model": getattr(getattr(backend, "info", None), "model", None),

                })

                return None

        except Exception:

            # is_vlm() probe failed — fail safe by skipping rather than

            # risking a hallucinated critique on an unknown backend.

            self._trace({"kind": "visual_critic_skipped", "reason": "is_vlm_probe_failed"})

            return None



        self._set_role_activity("critic", "Auditing screenshot...")

        # Try the structured-checklist path first (2026-05-24). Match a

        # mechanism recipe via goal + plan-grade criteria text + asset

        # names; if one resolves, build a closed-class yes/no prompt

        # from the recipe's checklist. The VLM's answer is parsed

        # line-by-line into a structured critique. If retrieval misses

        # OR the response doesn't parse cleanly, we fall back to the

        # legacy open-ended prompt below.

        recipe = None

        recipe_diag: dict = {}

        try:

            recipe, recipe_diag = self._memory.find_visual_playtest_for(

                goal=self._goal or "",

                plan_text=self._criteria or "",

                asset_names=list(self._session_assets.keys()),

                code=getattr(self, "_current_file", "") or "",

            )

        except Exception as _vp_e:

            self._trace({

                "kind": "visual_playtest_retrieval_error",

                "error": str(_vp_e)[:200],

            })

            recipe = None

        # Context-specific animation check: when motion is expected, append a

        # goal-driven "is it actually walking/kicking?" question to a clone of

        # the recipe so the prompt + parser + formatter all see it.

        recipe = self._augment_recipe_for_animation(recipe)

        recipe = self._adjust_checklist_for_goal(recipe)

        # Fix-round item 6: with NO action frame captured, action-frame

        # questions are unanswerable — strip them from the payload (skip,

        # don't fail) and surface the capture gap ONCE per attempt as a

        # deterministic advisory pointing at the real cause.

        if recipe is not None and action_png is None and self._animation_expected():

            recipe, _skipped_qs = self._strip_action_frame_questions(recipe)

            if _skipped_qs:

                self._trace({

                    "kind": "visual_playtest_action_questions_skipped",

                    "recipe_id": recipe.id,

                    "reason": "no_action_frame_captured",

                    "skipped": [q[:120] for q in _skipped_qs],

                })

                if not self._no_action_frame_advisory_sent:

                    self._no_action_frame_advisory_sent = True

                    self._pending_coaching.append(

                        "HARNESS ADVISORY: no peak-action frame was captured "

                        "— pressing the action keys produced no visible "

                        "canvas change. Either the action does not render, "

                        "or the player cannot act (if the report shows "

                        "CONTROL-NOT-RECOVERED, fix that first). The visual "

                        "critic's action-pose checks were SKIPPED this turn, "

                        "not failed."

                    )

            # Degenerate recipe (every question was action-dependent):

            # fall back to the generic open-ended critic path.

            if not (recipe.recipe.get("checklist") or []):

                recipe = None

        try:

            using_recipe = recipe is not None

            if using_recipe:

                prompt = self._build_visual_playtest_prompt(

                    recipe, before_png, action_png=action_png,

                )

                # Mirror the matched recipe id onto the active field

                # so the TUI status panel sees the same recipe id that

                # the VLM is being asked to evaluate against. Idempotent

                # — same recipe across iters just overwrites with the

                # same value.

                self._active_visual_playtest_recipe_id = recipe.id

                self._trace({

                    "kind": "visual_playtest_recipe_used",

                    "id": recipe.id,

                    "top_candidates": recipe_diag.get("top_candidates", []),

                    "match_tokens_sample": recipe_diag.get("match_tokens_sample", []),

                })

            elif before_png is not None:

                action_clause = (

                    ""

                    if action_png is None else

                    "3. Image 3 is captured at the moment of PEAK on-screen "

                    "change while a control was held — it shows the game "

                    "mid-ACTION (e.g. an attack, jump, or ability animation), "

                    "NOT a resting pose.\n"

                )

                action_guidance = (

                    ""

                    if action_png is None else

                    "  - Action visibility: use Image 3 to judge whether a "

                    "deliberate action animation actually renders. Image 3 IS "

                    "the active-input frame, so do not answer 'unclear' on "

                    "whether an action is visible — commit. If the goal implies "

                    "attacks/abilities and Image 3 shows no distinct action "

                    "pose (e.g. no extended arm / raised leg / projectile), "

                    "that absence is itself the finding.\n"

                )

                prompt = (

                    "You are an expert out-of-band Visual PlayTester and Critic "

                    "for a game development sandbox.\n"

                    "You are looking at screenshots of the latest generated "

                    "HTML5 canvas game:\n"

                    "1. Image 1 is taken before simulated inputs/playtesting.\n"

                    "2. Image 2 is taken after simulated inputs/playtesting "

                    "(resting state).\n"

                    f"{action_clause}\n"

                    f"GOAL FROM THE USER: {self._goal}\n\n"

                    "Compare the screenshots carefully for:\n"

                    "  - Lack of player locomotion or unresponsiveness: if the "

                    "goal implies a controllable player and simulated inputs "

                    "should move it, does the player appear stuck in the same "

                    "place across both images?\n"

                    f"{action_guidance}"

                    "  - Visual, positioning, or rendering bugs: wrong facing "

                    "direction, misaligned sprites, overlapping/clipped HUD, "

                    "blank canvas, or visibly frozen gameplay.\n\n"

                    "If everything looks correct and responsive, write "

                    "'Visual Critic: OK'.\n"

                    "If you spot a clear rendering defect, lack of movement, "

                    "frozen state, or control bug, write a 2-sentence visual "

                    "critique. Keep it objective, name the specific visual "

                    "evidence, and state what likely needs adjusting. Do NOT "

                    "output code or patches."

                )

                self._trace({

                    "kind": "visual_playtest_recipe_generic",

                    "reason": "no_recipe_matched",

                    "top_candidates": recipe_diag.get("top_candidates", []),

                })

            else:

                action_intro = (

                    "You are looking at a screenshot of the latest generated "

                    "HTML5 canvas game.\n\n"

                    if action_png is None else

                    "You are looking at TWO screenshots of the latest generated "

                    "HTML5 canvas game: Image 1 is the resting state; Image 2 "

                    "is captured at the moment of PEAK on-screen change while a "

                    "control was held (the game mid-ACTION). Use Image 2 to "

                    "judge whether a deliberate action animation actually "

                    "renders — commit, do not answer 'unclear'.\n\n"

                )

                prompt = (

                    "You are an expert out-of-band Visual PlayTester and Critic for a game development sandbox. "

                    + action_intro

                    + f"GOAL FROM THE USER: {self._goal}\n\n"

                    "Review the attached screenshot(s) carefully for visual, positioning, or rendering bugs. Examples:\n"

                    "  - Are projectiles spawning in the wrong direction?\n"

                    "  - Are character sprites misaligned or offset?\n"

                    "  - Are HUD elements overlapping or clipped?\n"

                    "  - Is the canvas completely blank or frozen?\n\n"

                    "If everything looks correct, write 'Visual Critic: OK'.\n"

                    "If you spot a clear rendering defect or bug, write a 2-sentence visual critique. "

                    "Keep it objective, name the specific visual evidence (e.g., 'character facing right but attack rendering to the left'), "

                    "and state what likely needs adjusting. Do NOT output code or patches."

                )

                self._trace({

                    "kind": "visual_playtest_recipe_generic",

                    "reason": "no_recipe_matched",

                    "top_candidates": recipe_diag.get("top_candidates", []),

                })

            # Ordered to match the prompt's "Image 1/2/3" numbering.

            if before_png is not None:

                images = [before_png, current_png]

            else:

                images = [current_png]

            if action_png is not None:

                images.append(action_png)

            messages = [

                {"role": "user", "content": prompt, "images": images}

            ]

            # Fix-round item 6: cheap dedupe. If this exact payload (recipe +

            # prompt + image bytes) produced a critique that was suppressed

            # as a repeat last time, the VLM verdict cannot differ — skip the

            # call (~12s each on MLX; 5 of 6 calls in trace 20260610_185238

            # were spent re-deriving an already-suppressed note).

            import hashlib as _hashlib

            _fp = _hashlib.sha1()

            _fp.update((recipe.id if recipe else "generic").encode("utf-8", "ignore"))

            _fp.update(prompt.encode("utf-8", "ignore"))

            for _b in images:

                if isinstance(_b, (bytes, bytearray)):

                    _fp.update(_hashlib.sha1(bytes(_b)).digest())

            _payload_fp = _fp.hexdigest()[:16]

            self._current_critic_payload_fp = _payload_fp

            if _payload_fp == getattr(self, "_suppressed_critic_payload_fp", None):

                self._trace({

                    "kind": "visual_critic_skipped_duplicate",

                    "recipe_id": recipe.id if recipe else None,

                    "payload_fp": _payload_fp,

                    "reason": (

                        "identical payload to the last critique that was "

                        "suppressed as a repeat — calling the VLM again "

                        "cannot produce a different verdict."

                    ),

                })

                self._set_role_activity("critic", "idle")

                return None

            self._trace({

                "kind": "visual_critic_start",

                "model": backend.info.model,

                "image_count": len(images),

                "recipe_id": recipe.id if recipe else None,

            })

            # Finding-1 instrumentation (2026-06-02): the model SEES images in

            # isolation yet returned "I can't see the screenshot" every iter in

            # the live /allroles run. Record exactly what the critic backend is

            # being handed — message count/roles, content sizes, and the real

            # byte payload of each image — so the next live trace proves whether

            # the pixels actually reach stream_chat (vs being empty/str/stripped)

            # instead of guessing. Pure observability; no behavior change.

            try:

                self._trace({

                    "kind": "visual_critic_payload",

                    "n_messages": len(messages),

                    "messages": [

                        {

                            "role": m.get("role"),

                            "content_chars": len(m.get("content") or ""),

                            "n_images": len(m.get("images") or []),

                            "image_bytes": [

                                (len(b) if isinstance(b, (bytes, bytearray)) else f"non-bytes:{type(b).__name__}")

                                for b in (m.get("images") or [])

                            ],

                        }

                        for m in messages

                    ],

                })

            except Exception:

                pass

            critic_role = getattr(self, "_model3_role", None)

            if critic_role != "critic":

                critic_role = getattr(self, "_model2_role", None) or "critic"

            on_tok = (

                self._role_token_cb(critic_role)

                if self._token_cb is not None else None

            )

            # Format-forcing prefill (added 2026-06-03). ROOT CAUSE of the

            # critic's useless output: qwen3.6-27B (thinking-mode VLM) SEES the

            # screenshot fine but answers in reasoning prose ("Wait, let me look

            # closer…") and never emits the Q1: yes/no lines, so parse_rate=0

            # and the whole critique is dropped — the safety net that should

            # catch a fighter facing the wrong way never fires. Seeding the

            # assistant turn with "Q1: " forces generation to START inside the

            # required format, skipping the reasoning preamble. The backend's

            # assistant-prefill path appends this to the prompt; we prepend it

            # back onto the reply before parsing. Recipe path only (the path

            # that expects the Qn: format). Genre/model-agnostic.

            #

            # Prefill-broken latch (2026-06-12): on some backends the

            # assistant prefill yields an EMPTY completion — trace

            # 20260612_004616 burned a wasted VLM call on 13/13 iterations

            # (raw_preview "Q1: ", parse_rate 0.0) before the terse retry

            # succeeded every time. After one empty prefilled response we

            # latch _critic_prefill_broken and skip the prefill for the

            # rest of the session, saving one VLM round-trip per iter

            # (30-60s/iter on local VLMs). The prefill stays ON for

            # backends where it works (qwen thinking-VLMs need it).

            _critic_prefill = (

                "Q1: "

                if (

                    using_recipe

                    and recipe is not None

                    and not getattr(self, "_critic_prefill_broken", False)

                )

                else None

            )

            _critic_messages = messages

            if _critic_prefill:

                _critic_messages = messages + [

                    {"role": "assistant", "content": _critic_prefill}

                ]

            result = await backend.stream_chat(

                _critic_messages,

                on_token=on_tok,

                options={"temperature": 0.2, "num_ctx": 4096},

                keep_alive=self._keep_alive_for_backend(backend),

                stall_seconds=120.0,

                overall_seconds=300.0,

                max_retries=1,

            )

            critique_raw = (result.text or "").strip()

            if _critic_prefill and len(critique_raw) < 3:

                # Empty/near-empty completion after a prefill: this

                # backend doesn't continue assistant prefills. Latch off

                # for the session; the reparse below still recovers THIS

                # call's critique.

                self._critic_prefill_broken = True

                self._trace({

                    "kind": "critic_prefill_disabled",

                    "raw_len": len(critique_raw),

                    "reason": "empty_completion_after_prefill",

                })

            # Re-attach the format-forcing prefix so the parser sees the full

            # "Q1: <answer>" first line (the model only generated what FOLLOWS

            # "Q1: "). Guard against the model having echoed it anyway.

            if _critic_prefill and not critique_raw[:6].lower().startswith("q1"):

                critique_raw = _critic_prefill + critique_raw

            # ABSTAIN guard (added 2026-06-02; TIGHTENED same day after a false

            # positive). Goal: drop a critique the model fabricated because it

            # couldn't see the image — WITHOUT dropping a genuine critique that

            # merely happens to contain a hedging phrase mid-analysis.

            #

            # The distinguishing signal is NOT "does it contain a refusal

            # phrase" (a real analysis can say "I can't see any HUD" etc.) — it

            # is "did the model actually render a verdict, or did it fall back

            # to a uniform default because it was blind". So we abstain ONLY

            # when BOTH hold: (a) a refusal phrase is present, AND (b) the reply

            # has no genuine verdict — either nothing parsed, OR every parsed

            # answer is the same fallback value (e.g. all "no"/all "unclear",

            # the classic "couldn't see → defaulted everything" shape from the

            # Street Fighter trace). A real critique that judges at least one

            # check differently (some yes, some no) is KEPT. Verified blind run:

            # qwen3.6-27B demonstrably saw + reasoned about the screenshots

            # (coder said "the screenshot shows the drawbridge background IS

            # drawn…"), so a blanket phrase-match was over-dropping real signal.

            if self._critic_abstained(critique_raw):

                _ab_parsed = self._parse_visual_playtest_response(critique_raw, recipe) if recipe else {"answers": {}}

                _ans = [a for (a, _r) in (_ab_parsed.get("answers") or {}).values()]

                _degenerate = (not _ans) or (len(set(_ans)) <= 1)

                if _degenerate:

                    self._trace({

                        "kind": "visual_critic_abstained",

                        "recipe_id": recipe.id if recipe else None,

                        "n_answers": len(_ans),

                        "distinct_answers": sorted(set(_ans)),

                        "raw_preview": critique_raw[:200],

                    })

                    self._set_role_activity("critic", "idle")

                    return None

                # Refusal phrase present but the model gave a real, MIXED

                # verdict — treat it as a genuine critique and fall through to

                # normal parsing/coaching.

                self._trace({

                    "kind": "visual_critic_abstain_overridden",

                    "recipe_id": recipe.id if recipe else None,

                    "reason": "mixed_verdict_present",

                    "distinct_answers": sorted(set(_ans)),

                })

            # If we used a recipe, parse the structured response and

            # format the failures into a single coaching string. Fall

            # back to the raw text when the VLM didn't follow the

            # format (parse_rate below threshold) — small VLMs

            # occasionally skip the Q1: prefix and revert to prose,

            # but we still want to surface whatever they said.

            if using_recipe and recipe is not None:

                parsed = self._parse_visual_playtest_response(critique_raw, recipe)

                parse_rate = parsed.get("parse_rate", 0.0)

                if parse_rate >= 0.5:

                    formatted = self._format_visual_playtest_critique(parsed, recipe)

                    self._trace({

                        "kind": "visual_playtest_parsed",

                        "recipe_id": recipe.id,

                        "parse_rate": round(parse_rate, 2),

                        "n_questions": parsed.get("n_questions", 0),

                        "n_answered": len(parsed.get("answers", {})),

                        "n_failures": sum(

                            1 for (a, _r) in parsed["answers"].values()

                            if a == "no"

                        ),

                        # Always log raw + per-Q answers so success path is auditable
                        # (trace 161245: "(all checks passed)" hid Q4:YES on wrong art).

                        "raw_preview": critique_raw[:300],

                        "answers": {

                            str(k): v[0]

                            for k, v in parsed.get("answers", {}).items()

                        },

                    })

                    # All-pass returns None — same shape as "Visual

                    # Critic: OK" in the legacy path.

                    if formatted is None:

                        self._trace({"kind": "visual_critic_end", "critique": "(all checks passed)"})

                        return None

                    self._trace({"kind": "visual_critic_end", "critique": formatted})

                    return formatted

                else:

                    self._trace({

                        "kind": "visual_playtest_unparseable",

                        "recipe_id": recipe.id,

                        "parse_rate": round(parse_rate, 2),

                        "raw_preview": critique_raw[:200],

                    })

                    # The VLM ignored the format (e.g. rambled for paragraphs

                    # on one question — adventure/SF traces 2026-05-30). RETRY

                    # ONCE with a hard "answer ONLY Qn: yes/no" reformat before

                    # giving up: small VLMs (qwen) ramble on the first pass but

                    # comply when the format is restated tersely, and losing ALL

                    # visual feedback (the critic's whole purpose) is worse than

                    # one extra cheap call. Only drop if the retry also fails.

                    checklist = recipe.recipe.get("checklist") or []

                    numbered = "\n".join(

                        f"Q{i + 1}: {q}" for i, q in enumerate(checklist)

                    )

                    reformat = (

                        "Answer the checklist below for the attached "

                        "screenshot(s). Output ONLY one line per question in "

                        "EXACTLY this form, nothing else — no reasoning, no "

                        "prose, no preamble:\n"

                        "Q1: yes\nQ2: no\nQ3: unclear\n...\n"

                        "Use only yes / no / unclear.\n\n" + numbered

                    )

                    try:

                        retry = await backend.stream_chat(

                            [{"role": "user", "content": reformat,

                              "images": images}],

                            on_token=on_tok,

                            options={"temperature": 0.0, "num_ctx": 4096},

                            keep_alive=self._keep_alive_for_backend(backend),

                            stall_seconds=120.0, overall_seconds=300.0,

                            max_retries=1,

                        )

                        reparsed = self._parse_visual_playtest_response(

                            (retry.text or "").strip(), recipe,

                        )

                        if reparsed.get("parse_rate", 0.0) >= 0.5:

                            parsed = reparsed

                            self._trace({

                                "kind": "visual_playtest_reparse_ok",

                                "recipe_id": recipe.id,

                                "parse_rate": round(reparsed["parse_rate"], 2),

                            })

                    except Exception as _re_e:

                        self._trace({

                            "kind": "visual_playtest_reparse_failed",

                            "error": str(_re_e)[:160],

                        })

                    # Surface whatever parsed (from the retry if it worked);

                    # never the raw chain-of-thought.

                    formatted = self._format_visual_playtest_critique(parsed, recipe)

                    self._trace({"kind": "visual_critic_end",

                                 "critique": formatted or "(unparseable — dropped after retry)"})

                    return formatted

            self._trace({"kind": "visual_critic_end", "critique": critique_raw})

            return critique_raw

        except Exception as e:

            self._trace({"kind": "visual_critic_failed", "error": str(e)})

            return None

        finally:

            self._set_role_activity("critic", "idle")



    async def _run_opening_book_sidecars(self, report: dict, iteration: int):

        """Let warm side models propose live recipes; store only after browser evidence.



        Yields activity events so the TUI can show per-role tok/s while sidecars run.

        """

        if not report.get("ok"):

            return

        sidecars: list[tuple[str, str, str]] = []

        if self.get_backend("architect") is not self._backend:

            sidecars.append(("architect", PLAYTESTS_FILENAME, "playtest"))

        critic_bk = self.get_backend("critic")

        if critic_bk is not None and critic_bk is not self._backend:

            sidecars.append(("critic", ASSET_AUDITS_FILENAME, "asset_audit"))

            sidecars.append(("critic", ANIMATION_AUDITS_FILENAME, "animation_audit"))

        if not sidecars:

            return

        for role, filename, kind in sidecars:

            backend = self.get_backend(role)

            if backend is None:

                continue

            self._set_role_activity(role, f"proposing {kind}")

            yield self._record(AgentEvent(

                "activity",

                "streaming",

                {"label": f"proposing {kind}", "role": role},

            ))

            prompt = (

                "You are proposing one compact reusable opening-book memory "

                "item for a single-file HTML game agent. Only propose a "

                "generic, transferable, executable or audit-style recipe. "

                "Do not mention this specific session name. Output STRICT JSON "

                "ONLY with keys: id, content, tags, recipe.\n\n"

                f"GOAL: {self._goal}\n"

                f"ITERATION: {iteration}\n"

                "BROWSER REPORT SUMMARY:\n"

                f"{format_report_for_model(report)[:1600]}\n"

            )

            try:

                on_tok = (

                    self._role_token_cb(role)

                    if self._token_cb is not None else None

                )

                result = await backend.stream_chat(

                    [{"role": "user", "content": prompt}],

                    on_token=on_tok,

                    options={"temperature": 0.2, "num_ctx": 4096},

                    keep_alive=self._keep_alive_for_backend(backend),

                    stall_seconds=120.0,

                    overall_seconds=300.0,

                    max_retries=0,

                )

                text = (result.text or "").strip()

                m = re.search(r"\{.*\}", text, re.DOTALL)

                if not m:

                    continue

                rec = json.loads(m.group(0))

                item = OpeningBookItem(

                    id=str(rec.get("id") or "").strip()[:80],

                    kind=kind,

                    content=str(rec.get("content") or "").strip()[:800],

                    tags=[str(t).strip() for t in (rec.get("tags") or [])[:8] if str(t).strip()],

                    source_tier="live",

                    verified=True,

                    helpful=0,

                    harmful=0,

                    recipe=dict(rec.get("recipe") or {}),

                    trace_ids=[self._session_id],

                    pass_count=1,

                    false_positive_count=0,

                    last_verified_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

                )

                if item.id and item.content:

                    ok = self._memory.append_live_opening_book_item(filename, item)

                    self._trace({

                        "kind": "opening_book_sidecar_proposal",

                        "role": role,

                        "filename": filename,

                        "id": item.id,

                        "stored": ok,

                    })

            except Exception as e:

                self._trace({

                    "kind": "opening_book_sidecar_error",

                    "role": role,

                    "kind_name": kind,

                    "err": str(e)[:200],

                })

            finally:

                self._set_role_activity(role, "idle")

                yield self._record(self._activity_idle_event(role))



    # ---- Phase 1B — diffuser pre-warm during Phase A --------------------



    def _critic_runs_on_independent_slot(self, critic_backend) -> bool:

        """True when the critic backend is a different slot than the

        coder, AND either points at a different endpoint OR shares a

        concurrent-capable endpoint (cloud / LAN inference server).



        The 2026-05-23 SOTA chess trace showed all three roles pointing

        at the same Anthropic endpoint; the pre-fix-#5 check then forced

        sequential best-of-N because endpoints matched. For cloud APIs,

        concurrent requests against the same account run in parallel

        just fine — fall-through to sequential there is over-conservative.

        """

        if critic_backend is None or critic_backend is self._backend:

            return False

        try:

            ce = getattr(getattr(critic_backend, "info", None), "endpoint", "")

            be = getattr(getattr(self._backend, "info", None), "endpoint", "")

            if ce and be and ce == be:

                # Same endpoint — concurrent only if the endpoint shape

                # supports it (cloud / LAN multi-worker).

                return self._endpoint_supports_concurrency(ce)

        except Exception:

            pass

        return True



    async def _drain_pending_critic_task(self, *, wait: bool) -> bool:

        """Optionally await the in-flight critic task and surface its

        coaching. Returns True if a task was drained (whether it

        produced coaching or not). Used at iter-boundary cleanup."""

        task = self._critic_task

        if task is None:

            return False

        if not wait and not task.done():

            return False

        try:

            if not task.done():

                await task

        except asyncio.CancelledError:

            pass

        except Exception as e:

            self._trace({"kind": "visual_critic_error", "error": str(e)[:240]})

        finally:

            self._critic_task = None

        return True



    async def _spawn_visual_critic(

        self,

        after_bytes: bytes,

        before_bytes: bytes | None,

        iteration: int,

        vc_role: str,

        action_bytes: bytes | None = None,

    ) -> None:

        """Background worker that runs the visual critic and appends to

        `_pending_coaching` on completion. Errors are swallowed +

        traced so a critic crash never kills the iter loop."""

        try:

            critique = await self.run_visual_critic(

                after_bytes, before_bytes, action_png=action_bytes,

            )

        except Exception as exc:

            self._trace({

                "kind": "visual_critic_error",

                "iteration": iteration,

                "error": str(exc)[:240],

            })

            return

        if not critique:

            return

        cleaned = critique.strip()

        if not cleaned or "ok" in cleaned.lower()[:30]:

            return

        queued = self._queue_visual_critic_coaching(

            cleaned, iteration=iteration, vc_role=vc_role,

        )

        self._trace({

            "kind": "visual_critic_concurrent_completed",

            "iteration": iteration,

            "vc_role": vc_role,

            "critique_preview": cleaned[:240],

            "queued": queued,

        })



    # ---- Phase 1.5 — autonomous self-feedback ---------------------------

    #

    # `_run_autonomous_playtest` is the per-iter hook that:

    #   1. Reads the root-playbook "behavior_playtest" recipes

    #   2. Skips any whose `applies_when` JS gate evaluates falsy in the

    #      current page (so a turn-based board game never runs the

    #      directional-movement recipe and vice versa — applicability

    #      is observable, not genre-named)

    #   3. Runs the surviving recipes via LiveBrowser.record_playtest

    #   4. Evaluates the per-recipe `check_kind` against the timeline

    #   5. If any recipe produced a finding, asks the critic slot for ONE

    #      paragraph of user-style feedback, prepends "[AUTONOMOUS PLAYTEST]"

    #      and routes it into _pending_feedback so Phase 0.1's partitioner

    #      handles it like a real user message

    # The budget governor (3 cycles/session, 2-no-finding auto-stop) and

    # the `/playtest off` kill-switch are both checked first; this method

    # is a no-op when either gate is closed.



    # Post-green cap (run_10 Q*bert/Galaga): polish cycles after a green
    # build cost 15-20 min each at collapsed tok/s and produced churn the
    # user read as "random changes to a working game". One cycle catches a
    # real behavioral miss; stop there. (Was 3.)
    _AUTONOMOUS_MAX_CYCLES = 1

    _AUTONOMOUS_FACING_TOLERANCE_DEG = 25.0

    _AUTONOMOUS_MIN_MOVE_PX = 6.0



    def _autonomous_playtest_disabled(self) -> tuple[bool, str]:

        """Gate check: return (disabled, reason)."""

        if not getattr(self, "_use_autonomous_feedback", True):

            return True, "feedback_off"

        if self._user_force_done:

            return True, "force_done"

        if self._autonomous_playtest_cycle >= self._AUTONOMOUS_MAX_CYCLES:

            return True, "budget_exhausted"

        if self._autonomous_no_findings_streak >= 2:

            return True, "no_findings_streak"

        return False, ""



    def _evaluate_behavior_playtest_check(

        self,

        recipe: dict,

        timeline: dict,

    ) -> dict | None:

        """Apply the recipe's `check_kind` to the captured timeline.



        Returns a finding dict {check_kind, finding_label, evidence} or

        None when the check passed. Each `check_kind` is genre-free —

        the recipe owns its applicability gate AND the human-readable

        finding label; this method just turns sampled state into a

        pass/fail call.

        """

        check_kind = (recipe.get("check_kind") or "").lower()

        samples = timeline.get("samples") or []

        if not samples or not check_kind:

            return None

        label = recipe.get("finding_label") or check_kind



        def _player_xy(state: dict | None) -> tuple[float, float] | None:

            if not state:

                return None

            for px, py in (

                ("player.x", "player.y"), ("ship.x", "ship.y"),

                ("hero.x", "hero.y"), ("x", "y"),

            ):

                if isinstance(state.get(px), (int, float)) and isinstance(state.get(py), (int, float)):

                    return float(state[px]), float(state[py])

            return None



        def _player_facing_rad(state: dict | None) -> float | None:

            if not state:

                return None

            for path in (

                "player.facing", "player.angle", "player.heading",

                "player.rot", "player.rotation",

                "ship.facing", "ship.angle", "ship.heading",

                "hero.facing", "hero.angle", "hero.heading",

                "facing", "angle", "heading", "rot", "rotation",

            ):

                v = state.get(path)

                if isinstance(v, (int, float)):

                    # If the value is > ~6.5 we assume degrees; convert

                    # to radians so the comparison is consistent. Generic

                    # heuristic; doesn't matter which the game uses.

                    import math

                    return math.radians(v) if abs(v) > 6.5 else float(v)

            return None



        if check_kind == "any_progress":

            # No new info in 10s of observation = the game is stuck.

            base_hash = samples[0].get("canvas_hash")

            base_state = samples[0].get("state") or {}

            changed = False

            for s in samples[1:]:

                if s.get("canvas_hash") and s["canvas_hash"] != base_hash:

                    changed = True

                    break

                s_state = s.get("state") or {}

                # Any numeric leaf differing from baseline counts as progress.

                for k, v in s_state.items():

                    if isinstance(v, (int, float)) and base_state.get(k) != v:

                        changed = True

                        break

                if changed:

                    break

            if not changed:

                return {

                    "check_kind": check_kind,

                    "finding_label": label,

                    "evidence": "no canvas/state delta across 10s of observation",

                }

            return None



        if check_kind == "facing_matches_movement":

            import math

            s0 = samples[0]

            s1 = samples[-1]

            xy0 = _player_xy(s0.get("state"))

            xy1 = _player_xy(s1.get("state"))

            facing = _player_facing_rad(s0.get("state"))

            if xy0 is None or xy1 is None or facing is None:

                return None  # game stopped exposing state mid-test

            dx = xy1[0] - xy0[0]

            dy = xy1[1] - xy0[1]

            mag = (dx * dx + dy * dy) ** 0.5

            if mag < self._AUTONOMOUS_MIN_MOVE_PX:

                # No movement at all — different recipe will catch this.

                return None

            move_angle = math.atan2(-dy, dx)  # screen-Y flipped to math-Y

            diff = abs(((move_angle - facing) + math.pi) % (2 * math.pi) - math.pi)

            tol = math.radians(self._AUTONOMOUS_FACING_TOLERANCE_DEG)

            if diff > tol:

                evidence = (

                    f"facing≈{math.degrees(facing):.0f}°, "

                    f"movement≈{math.degrees(move_angle):.0f}° "

                    f"(Δ={math.degrees(diff):.0f}°), |move|={mag:.1f}px"

                )

                return {

                    "check_kind": check_kind,

                    "finding_label": label,

                    "evidence": evidence,

                }

            return None



        if check_kind == "stays_in_canvas":

            # Player left the canvas during a 3s held-key window.

            # Canvas size sampled via custom_expr in the recipe would

            # be cleaner; here we approximate by checking the largest

            # observed coordinate against a generous bound (4000 px).

            # Real games rarely have canvases bigger than that.

            xy_last = _player_xy((samples[-1].get("state") or {}))

            if xy_last is None:

                return None

            x, y = xy_last

            if not (-50.0 <= x <= 4000.0 and -50.0 <= y <= 4000.0):

                return {

                    "check_kind": check_kind,

                    "finding_label": label,

                    "evidence": f"player ended at ({x:.0f},{y:.0f}) — out of plausible canvas range",

                }

            return None



        # Unknown check_kind — log + skip rather than throw.

        return None



    async def _run_autonomous_playtest(

        self,

        iteration: int,

        report: dict,

    ):

        """One cycle of the autonomous self-feedback loop.



        Caller is the iter loop; we only run when:

          - /playtest is on (default)

          - probes passed this iter (report.ok is True)

          - the budget governor allows another cycle

          - no Ctrl+D in flight



        Yields AgentEvents for status panel; queues findings into

        _pending_feedback for the NEXT iter's _flush_user_injections.

        """

        # Phase 1.5.2 — single skip-reason trace event so we can see WHY

        # the loop didn't run in future sessions. The 2026-05-22 Pac-Man

        # session had no autonomous events at all and we couldn't tell

        # whether the loop disabled, the iter failed, or recipes were

        # missing. Every silent return now leaves a breadcrumb.

        disabled, reason = self._autonomous_playtest_disabled()

        if disabled:

            self._trace({

                "kind": "autonomous_playtest_skipped",

                "reason": f"disabled:{reason}",

                "iteration": iteration,

            })

            return

        if not report.get("ok"):

            self._trace({

                "kind": "autonomous_playtest_skipped",

                "reason": "iter_failed",

                "iteration": iteration,

            })

            return

        if self.browser is None:

            self._trace({

                "kind": "autonomous_playtest_skipped",

                "reason": "no_browser",

                "iteration": iteration,

            })

            return

        try:

            # Load directly from the seed function — guarantees we see

            # the behavior_playtest recipes even on a fresh install where

            # the on-disk root playbook hasn't been hydrated yet. The

            # heavier in-memory `OpeningBookStore` is used by the model-

            # facing render path, but for an in-process check the seed

            # list is faster and avoids file-IO timing flakes in tests.

            from memory import _opening_book_seed_items, PLAYTESTS_FILENAME

            seed = _opening_book_seed_items()

            recipes = seed.get(PLAYTESTS_FILENAME, [])

        except Exception as e:

            self._trace({

                "kind": "autonomous_playtest_skipped",

                "reason": "recipe_load_error",

                "err": str(e)[:200],

                "iteration": iteration,

            })

            return

        behavior_recipes = [

            r for r in recipes

            if isinstance(getattr(r, "recipe", None), dict)

            and (r.recipe.get("type") or "").lower() == "behavior_playtest"

        ]

        if not behavior_recipes:

            self._trace({

                "kind": "autonomous_playtest_skipped",

                "reason": "no_behavior_recipes",

                "iteration": iteration,

                "total_recipes": len(recipes),

            })

            return



        self._autonomous_playtest_cycle += 1

        yield self._record(AgentEvent(

            "info",

            f"[dim]autonomous playtest: cycle {self._autonomous_playtest_cycle}/{self._AUTONOMOUS_MAX_CYCLES} "

            f"({len(behavior_recipes)} recipes available)[/dim]",

        ))



        # Fix #3 — diagnostic probes evaluated only when a recipe's

        # applies_when gate fails. Each entry is a small structural

        # check the gate is likely testing for; the boolean map gets

        # stashed on the skip event so future trace mining can see

        # which specific condition was false (no state global, no

        # canvas, no exposed player.x/.y, etc.) without re-running

        # the session. Genre-free, applies to any HTML game shape.

        _GATE_DIAGNOSTICS_JS = (

            "(()=>{const out={};"

            "out.has_state=!!window.state;"

            "out.has_gameState=!!window.gameState;"

            "const s=window.state||window.gameState||null;"

            "out.has_state_or_gameState=!!s;"

            "out.has_canvas=!!document.querySelector('canvas');"

            "out.canvas_has_dims=(()=>{const c=document.querySelector('canvas');"

            "return !!c&&c.width>0&&c.height>0;})();"

            "out.has_player_xy=!!(s&&(typeof s.player?.x==='number'&&typeof s.player?.y==='number'));"

            "out.has_player_facing=!!(s&&(typeof s.player?.facing==='number'||"

            "typeof s.player?.angle==='number'||typeof s.player?.heading==='number'||"

            "typeof s.player?.rot==='number'||typeof s.player?.rotation==='number'));"

            "out.top_level_xy_count=(()=>{if(!s||typeof s!=='object')return 0;let n=0;"

            "for(const k in s){try{const v=s[k];if(v&&typeof v==='object'&&"

            "typeof v.x==='number'&&typeof v.y==='number')n++}catch(e){}};return n;})();"

            "return out;})()"

        )



        findings: list[dict] = []

        # Skip-decision cache (2026-06-12): the same recipes re-fail the

        # same applicability gate with identical diagnostics on every

        # clean iter when the code hasn't changed (trace 20260612_004616

        # logged 3 recipes x many iters of identical gate noise). Key on

        # (recipe_id, code hash) so any patch invalidates the cache; a

        # cached skip costs zero browser evals and zero trace events.

        import hashlib as _hashlib

        # getattr defaults so partially-constructed test stubs (which

        # skip __init__) don't AttributeError here.

        _code_hash = _hashlib.sha256(

            (getattr(self, "_current_file", "") or "")

            .encode("utf-8", errors="replace")

        ).hexdigest()[:16]

        if not hasattr(self, "_recipe_skip_cache"):

            self._recipe_skip_cache = set()

        for rec in behavior_recipes:

            r = rec.recipe

            _rec_id = getattr(rec, "id", "")

            if (_rec_id, _code_hash) in self._recipe_skip_cache:

                continue

            applies_js = r.get("applies_when") or "true"

            try:

                applies = await self.browser._safe_eval(applies_js)

            except Exception:

                applies = False

            if not applies:

                self._recipe_skip_cache.add((_rec_id, _code_hash))

                # Run the diagnostic probes so the trace shows WHICH

                # condition was missing on this game's state shape.

                try:

                    diag = await self.browser._safe_eval(_GATE_DIAGNOSTICS_JS)

                except Exception as e:

                    diag = {"diag_error": str(e)[:120]}

                self._trace({

                    "kind": "autonomous_recipe_skipped",

                    "recipe_id": _rec_id,

                    "reason": "applicability_gate_falsy",

                    "diagnostics": diag if isinstance(diag, dict) else {},

                })

                continue

            timeline = await self.browser.record_playtest(

                input_script=r.get("input_script") or [],

                sample_times_s=r.get("sample_times_s") or [0.0],

            )

            self._trace({

                "kind": "autonomous_recipe_ran",

                "recipe_id": getattr(rec, "id", ""),

                "samples": len(timeline.get("samples") or []),

                "errors": timeline.get("errors") or [],

            })

            try:

                finding = self._evaluate_behavior_playtest_check(r, timeline)

            except Exception as e:

                self._trace({

                    "kind": "autonomous_check_error",

                    "recipe_id": getattr(rec, "id", ""),

                    "err": str(e)[:200],

                })

                finding = None

            if finding:

                finding["recipe_id"] = getattr(rec, "id", "")

                findings.append(finding)



        self._trace({

            "kind": "autonomous_playtest_summary",

            "iteration": iteration,

            "cycle": self._autonomous_playtest_cycle,

            "ran": len(behavior_recipes),

            "findings": len(findings),

            "finding_ids": [f.get("recipe_id") for f in findings],

        })



        if not findings:

            self._autonomous_no_findings_streak += 1

            return

        self._autonomous_no_findings_streak = 0



        # Build the user-style feedback string. Keep this TIGHT — local

        # 27B models lose focus past ~200 tokens of pre-amble. We don't

        # call the critic at all in this round; instead we synthesize

        # one paragraph from the recipe-supplied finding_labels +

        # evidence. A future enhancement can route to the critic slot

        # for a freer-form summary, but the trade-off is cost + drift.

        bullets = []

        for f in findings:

            bullets.append(f"- {f['finding_label']} ({f['evidence']})")

        feedback_text = (

            "[AUTONOMOUS PLAYTEST] I ran a short scripted playtest after "

            "iter {iter} passed probes and noticed:\n{body}\n"

            "Treat this as one user observation, not a hard failure — "

            "address what's actionable; ignore what's already correct. "

            # Brevity nudge (2026-06-12, trace 20260612_132314): this turn

            # produced an 18K-token deliberation essay on a local 27B.

            # Prompt-only — the stream is never cut.

            "Keep your reply brief — a short <diagnose> (1-3 lines) plus "

            "minimal <patch> blocks; no extended analysis."

        ).format(iter=iteration, body="\n".join(bullets))

        # Agent-generated — must not trip user-feedback detectors.

        self._queue_internal_feedback(feedback_text)

        # Loop visibility: the no-vision critique found problems and is

        # handing them to the coder, which fixes them on the next turn.

        self._trace({

            "kind": "critique_findings_sent_to_coder",

            "vision": False,

            "iteration": iteration,

            "finding_ids": [f.get("recipe_id") for f in findings],

        })

        yield self._record(AgentEvent(

            "info",

            f"[magenta]critique findings sent to the coder[/magenta] "

            f"({len(findings)} finding(s) from cycle "

            f"{self._autonomous_playtest_cycle})",

        ))



    async def _run_structured_local_vlm_critique(

        self, current_png: bytes, iteration: int,

    ) -> bool:

        """Run the matched visual-playtest checklist through the local VLM.



        Used when `/vlm-critique` is ON but there's no dedicated critic

        slot (the coder is a VLM that multiplexes one model). Reuses the

        structured prompt/parse/format pipeline that the out-of-band

        critic uses, so the model answers concrete yes/no questions

        (e.g. "is the hero a real sprite, not a placeholder box?")

        instead of the open-ended progress judge that lost the

        "MISSING hero_idle" finding in the 2026-06-14 dragon's-lair run.



        Returns True when it handled the critique (queued coaching OR

        confirmed all-pass), False to let the caller fall through to the

        open-ended progress judge.

        """

        recipe_id = getattr(self, "_active_visual_playtest_recipe_id", None)

        if not recipe_id:

            return False

        try:

            recipe, _diag = self._memory.find_visual_playtest_for(

                goal=self._goal or "",

                plan_text=self._criteria or "",

                asset_names=list(self._session_assets.keys()),

                code=getattr(self, "_current_file", "") or "",

            )

        except Exception:

            recipe = None

        if recipe is None or recipe.id != recipe_id:

            return False

        recipe = self._adjust_checklist_for_goal(recipe)

        # Resolve in-process mlx_vlm once per session (same path as vision judge).

        if not self._ensure_local_vlm_path():

            return False

        try:

            from vision_judge import run_local_vlm_prompt

            prompt = self._build_visual_playtest_prompt(

                recipe, before_png=None, action_png=None,

            )

            raw = await run_local_vlm_prompt(

                prompt=prompt,

                images=[current_png],

                model_path=self._local_vlm_path,

            )

        except Exception as exc:

            self._trace({

                "kind": "structured_critic_via_local_vlm_error",

                "iteration": iteration,

                "error": str(exc)[:240],

            })

            return False

        if not raw:

            return False

        parsed = self._parse_visual_playtest_response(raw, recipe)

        critique = self._format_visual_playtest_critique(parsed, recipe)

        self._trace({

            "kind": "structured_critic_via_local_vlm",

            "iteration": iteration,

            "recipe_id": recipe.id,

            "parse_rate": parsed.get("parse_rate"),

            "had_findings": bool(critique),

        })

        if critique:

            queued = self._queue_visual_critic_coaching(

                critique.strip(), iteration=iteration, vc_role="critic",

            )

            if queued:

                self._record(AgentEvent(

                    "info",

                    f"[magenta]vlm-critique[/magenta] (iter {iteration}): "

                    f"{critique.strip()}",

                ))

        return True



    async def _run_vision_judge(self, current_png: bytes, iteration: int) -> None:

        """Ask a vision model whether this iter made visible progress

        toward the goal, and queue the verdict for the next user turn.



        Local-first by design: this runs OUT-OF-BAND from the building

        backend. The user's local model keeps writing code (as today);

        only the visual judgment uses the local MLX-VLM resolved at

        session start. If no local VLM is discoverable, we skip rather

        than silently calling a cloud model (user rule: never silent

        cloud calls). If the judge is unreachable or returns nothing,

        we log a single trace event and continue — never block the run.

        """

        # Gated on /vlm-critique (same toggle as the structured visual

        # critic). This path is the local MLX-VLM fallback when no

        # dedicated critic slot is staged and /allroles is off.

        if not getattr(self, "_use_vlm_critique", False):

            return

        if not self._vision_judge_headroom_ok():

            self._trace({

                "kind": "vision_judge_skipped",

                "iteration": iteration,

                "reason": "tight_headroom",

            })

            return

        try:

            from vision_judge import is_enabled, judge_visual_progress

        except Exception:

            return

        if not is_enabled():

            return

        if not self._goal:

            return

        # Resolve a local VLM once per session. We never auto-use a

        # cloud model here — the chat.py /check command remains the

        # explicit user-triggered path for that.

        if not self._ensure_local_vlm_path():

            return

        # Structured-checklist path: when a visual recipe matched this

        # session, run its yes/no checklist through the local VLM instead

        # of the open-ended progress judge. The checklist asks concrete

        # questions ("is the character a real sprite, not a placeholder

        # box?") that the open-ended judge missed in the dragon's-lair

        # 2026-06-14 trace. Falls through to the progress judge when no

        # recipe matched or the structured call produced nothing.

        if getattr(self, "_active_visual_playtest_recipe_id", None):

            handled = await self._run_structured_local_vlm_critique(

                current_png, iteration,

            )

            self._prev_judge_png = current_png

            if handled:

                return

        prev_png = self._prev_judge_png

        verdict = await judge_visual_progress(

            goal=self._goal,

            current_png=current_png,

            previous_png=prev_png,

            model=self._local_vlm_path,

        )

        # Rotate for next iter's comparison even when the judge skipped,

        # so a transient API failure doesn't permanently break "compare

        # against prior" once it recovers.

        self._prev_judge_png = current_png

        if verdict is None:

            self._trace({

                "kind": "vision_judge_skipped",

                "iteration": iteration,

                "reason": "no verdict (disabled, no key, or call failed)",

            })

            return

        delta = self._last_screenshot_delta

        # Log the raw model reply (truncated) alongside the parsed

        # fields. Parsing can fail silently (model emits a verdict in

        # a shape that doesn't match the PROGRESS:/MISSING: regex) and

        # without the raw text we cannot tell whether the judge saw

        # the screenshot at all — diagnosed 2026-05-16 from a session

        # where `progress: null` showed up on every iter and there was

        # no way to know why.

        raw_excerpt = (verdict.raw or "")[:500]

        parse_failed = verdict.progress is None and not verdict.note

        # `image_count == 0` here would mean the judge call itself

        # got no PNG — a more fundamental failure than parse_failed.

        # Surface both flags so the user can grep one event and know

        # exactly which layer broke.

        self._trace({

            "kind": "vision_judge",

            "iteration": iteration,

            "progress": verdict.progress,

            "note": verdict.note,

            "model": verdict.model,

            "screenshot_delta": delta,

            "raw": raw_excerpt,

            "parse_failed": parse_failed,

            "image_count": verdict.image_count,

            "prompt_chars": verdict.prompt_chars,

            "result_chars": verdict.result_chars,

        })

        # Stash the last verdict so it survives compaction and can be

        # surfaced in the state-anchor summary (item 8). The text

        # marker is light enough to embed in the anchor string without

        # needing a multi-modal message shape.

        self._last_vision_verdict_iter = iteration

        self._last_vision_verdict_progress = verdict.progress

        self._last_vision_verdict_note = (verdict.note or "").strip()

        # Surface the verdict to the user — short and unambiguous.

        prog_label = (

            "progress" if verdict.progress is True

            else ("no progress" if verdict.progress is False else "unclear")

        )

        self._record(AgentEvent(

            "info",

            f"[magenta]vision judge[/magenta] (iter {iteration}): "

            f"{prog_label} — {verdict.note or '(no note)'}"

        ))

        # Queue the "what's still missing" line for the next user turn

        # so the building model gets concrete visual feedback. Only when

        # it's actionable (non-empty, and not "nothing obvious").

        raw_note = (verdict.note or "").strip()

        note = self._clean_actionable_vision_note(raw_note)

        if raw_note and not note:

            self._trace({

                "kind": "vision_judge_coaching_suppressed",

                "iteration": iteration,

                "reason": "non_actionable_fragment",

                "note": raw_note[:160],

            })

        if note and note.lower().strip(".") != "nothing obvious":

            prefix = (

                "VISUAL JUDGE (looked at the screenshot of your last "

                "iteration): "

            )

            # Visible-regression escalation: a "no progress" verdict

            # combined with a substantial pixel-delta against the prior

            # frame means this iter visibly CHANGED the canvas without

            # making it better. That's the regression signature the

            # screenshot-diff detector is for. Flag it explicitly so

            # the next iter's fix prompt is tuned for rollback rather

            # than further forward edits.

            regression = (

                verdict.progress is False

                and delta is not None

                and delta > 0.15

            )

            if regression:

                prefix += (

                    "REGRESSION SUSPECTED — this iter visibly changed "

                    f"the canvas (pixel delta {delta:.2f}) but the "

                    "result is NOT closer to the goal. Consider "

                    "rolling back the last patch and trying a smaller "

                    "change. "

                )

            elif verdict.progress is False:

                prefix += "this iteration did NOT visibly move toward the goal. "

            elif verdict.progress is True:

                prefix += "this iteration made progress, but "

            else:

                prefix += "(progress unclear). "

            self._pending_coaching.append(

                prefix + "Still visibly missing/wrong: " + note +

                " — address this on the next iter."

            )



