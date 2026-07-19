"""Prompt builders extracted from agent.py for readability.

Fix prompts, structured compaction summary, seed HTML injection,
focused slices, and report formatting. Moved VERBATIM from
`GameAgent` (no behavior change).
"""

from __future__ import annotations


import json
import re
from pathlib import Path
from typing import Any


from agent_feedback import _HARNESS_ADVISORY_SENTINEL, _subsystem_hint

from agent_helpers import (
    _POLISH_TURN_CAP,
    _REPORT_BLOCK_BEGIN,
    _REPORT_BLOCK_END,
    _SUMMARIZE_MIN_HTML_BYTES,
    _SUMMARIZE_MIN_PROBES_BYTES,
    _baseline_structurally_broken,
    _is_degenerate_baseline,
    _truncation_reason,
)

from patches import extract_patches
from memory import signature_for_report
from tools import format_report_for_model


class PromptBuildingMixin:

    """Prompt building for GameAgent (see module docstring)."""

    # Observation-only boundaries for prompt provenance. These do not
    # participate in prompt assembly; unmatched bytes are kept as `other`.
    _PROMPT_SECTION_BOUNDARIES: tuple[tuple[str, str, str], ...] = (
        ("user_answer", "user", "================ USER ANSWER (HIGHEST PRIORITY)"),
        ("user_feedback", "user", "================ USER FEEDBACK (HIGHEST PRIORITY)"),
        ("scope_arbitration", "harness", "================ FEEDBACK SCOPE ARBITRATION"),
        ("media_directive", "harness", "================ MEDIA-CHANGE DIRECTIVE"),
        ("scoped_change", "harness", "================ SCOPED-CHANGE DIRECTIVE"),
        ("agent_coaching", "harness", "================ AGENT COACHING"),
        ("generated_assets", "generated_media", "================ GENERATED ASSETS"),
        ("generated_sounds", "generated_media", "================ GENERATED SOUNDS"),
        ("state_anchor", "history_summary", "# Session state anchor"),
        ("opening_book", "memory", "<opening_book>"),
        ("components", "memory", "<components>"),
        ("playbook", "memory", "<playbook>"),
        ("outline_traps", "memory", "OUTLINE TRAPS (match your failure — do not add scope):"),
        ("test_report", "harness_report", _REPORT_BLOCK_BEGIN),
        ("current_file", "working_file", "CURRENT FILE ON DISK"),
        ("seed_file", "seed_file", "EXISTING FILE:"),
    )

    @classmethod
    def _collect_prompt_sections(cls, prompt_text: str) -> list[dict[str, str]]:
        """Partition a prompt exactly at stable, observation-only boundaries."""
        if not prompt_text:
            return []
        hits: list[tuple[int, str, str]] = []
        for section_id, source, marker in cls._PROMPT_SECTION_BOUNDARIES:
            for match in re.finditer(
                rf"(?m)^{re.escape(marker)}",
                prompt_text,
            ):
                hits.append((match.start(), section_id, source))
        hits.sort(key=lambda item: item[0])
        # File blocks consume the remainder of the prompt. Never inspect
        # their payload for marker-shaped user HTML or JavaScript text.
        terminal_starts = [
            start for start, section_id, _source in hits
            if section_id in {"current_file", "seed_file"}
        ]
        if terminal_starts:
            terminal_start = min(terminal_starts)
            hits = [
                hit for hit in hits
                if hit[0] <= terminal_start
            ]

        sections: list[dict[str, str]] = []
        if not hits:
            return [{"id": "core", "source": "other", "text": prompt_text}]
        if hits[0][0] > 0:
            sections.append({
                "id": "core",
                "source": "other",
                "text": prompt_text[:hits[0][0]],
            })

        seen: dict[str, int] = {}
        for i, (start, section_id, source) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(prompt_text)
            seen[section_id] = seen.get(section_id, 0) + 1
            occurrence = seen[section_id]
            unique_id = section_id if occurrence == 1 else f"{section_id}_{occurrence}"
            sections.append({
                "id": unique_id,
                "source": source,
                "text": prompt_text[start:end],
            })
        return sections

    @staticmethod
    def _bounded_prompt_section_manifest(
        sections: list[dict[str, str]],
        *,
        max_entries: int = 12,
        max_serialized_bytes: int = 1024,
    ) -> tuple[list[dict[str, int | str]], int]:
        """Strip text and bound the trace manifest; return folded char count."""
        manifest: list[dict[str, int | str]] = []
        other_chars = 0
        folded = False
        for section in sections:
            entry: dict[str, int | str] = {
                "id": section["id"],
                "source": section["source"],
                "chars": len(section["text"]),
            }
            candidate = manifest + [entry]
            candidate_bytes = len(json.dumps(
                candidate, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8"))
            if (
                folded
                or len(candidate) > max_entries
                or candidate_bytes > max_serialized_bytes
            ):
                folded = True
                other_chars += int(entry["chars"])
                continue
            manifest.append(entry)
        return manifest, other_chars

    def _prompt_provenance_fields(self) -> dict[str, Any]:
        """Return compact totals and an exact last-user section manifest."""
        system_chars = 0
        if self._messages and self._messages[0].get("role") == "system":
            system_content = self._messages[0].get("content")
            if isinstance(system_content, str):
                system_chars = len(system_content)

        last_user = ""
        for message in reversed(self._messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    last_user = content
                break
        sections = self._collect_prompt_sections(last_user)
        manifest, other_chars = self._bounded_prompt_section_manifest(sections)
        return {
            "prompt_system_chars": system_chars,
            "prompt_history_chars": self._estimate_ctx_fill(),
            "prompt_last_user_chars": len(last_user),
            "prompt_sections": manifest,
            "prompt_other_chars": other_chars,
        }


    @staticmethod
    def _prompt_orders_full_rewrite(prompt_text: str) -> bool:
        """True when a harness-authored prompt ORDERS a full <html_file>.

        2026-06-10 (both dojo-fight traces): several recovery prompts
        instruct the model to emit a complete <html_file>, but the
        baseline-exists gate in `_materialize` then rejected the compliant
        reply — the harness contradicting itself. Every caller that sends
        a prompt matching this predicate must arm `_allow_one_rewrite` so
        the requested rewrite is actually accepted.

        Matches ORDERS only, not conditionals: the generic fallback
        ("If this is the first build, send a complete <html_file>.") and
        the identical-reply escalation ("Do NOT re-emit <html_file>")
        must NOT match.
        """
        low = prompt_text.lower()
        return (
            "emit one complete <html_file>" in low
            # format-stuck escalation: "Stop trying to send <patch> this
            # turn. Send a complete <html_file>..."
            or "stop trying to send <patch>" in low
            # stream-loop recovery offers "emit a smaller `<html_file>`"
            or "emit a smaller `<html_file>`" in low
        )

    @classmethod
    def _seed_structural_tokens(cls, html: str, limit: int = 12) -> list[str]:
        """Pull genre-free structural tokens from a seed file for retrieval.

        Data-driven (tokens come from the file, not a hardcoded vocabulary):
        DOM `id=`/`class=` values and top-level function names are the
        strongest signal of the file's actual shape. When the goal is terse
        ("add a help button") these tokens let plan-time outline retrieval
        match on what the file IS (hotspots, inventory, scenes, verbs) rather
        than the goal's single weak DOM word. Returns a small de-duped list.
        """
        if not html:
            return []
        from collections import Counter
        counts: Counter[str] = Counter()
        for m in re.finditer(r'\bid\s*=\s*["\']([A-Za-z_][\w-]*)["\']', html):
            counts[m.group(1).lower()] += 2  # ids weighted highest
        for m in re.finditer(r'\bclass\s*=\s*["\']([A-Za-z_][\w\s-]*)["\']', html):
            for cls_name in m.group(1).split():
                counts[cls_name.lower()] += 1
        for m in re.finditer(r'\bfunction\s+([A-Za-z_$][\w$]*)', html):
            counts[m.group(1).lower()] += 1
        out: list[str] = []
        for tok, _n in counts.most_common():
            if tok in cls._SEED_TOKEN_STOPWORDS or len(tok) < 3:
                continue
            out.append(tok)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _report_digest_lines(report: dict) -> str:
        """3-line digest of a test report for collapsed history turns:
        ok flag, probes passing x/y, first blocker. Pure function."""
        probes = report.get("probes") or []
        passing = sum(1 for p in probes if p.get("ok"))
        blocker = ""
        for p in probes:
            if not p.get("ok"):
                blocker = f"probe '{p.get('name') or '?'}' failed"
                break
        if not blocker:
            for key in ("page_errors", "console_errors", "errors"):
                vals = report.get(key) or []
                if vals:
                    blocker = str(vals[0])
                    break
        # Sanitize so the digest can't close its wrapping HTML comment.
        blocker = blocker.replace("-->", "->").replace("\n", " ")[:120]
        return (
            f"ok={bool(report.get('ok'))}\n"
            f"probes passing: {passing}/{len(probes)}\n"
            f"first blocker: {blocker or '(none recorded)'}"
        )



    def _wrap_report_block(self, text: str, report: dict) -> str:
        """Wrap a test-report user turn in collapse sentinels (item 3).

        The digest is embedded in the BEGIN marker at append time so
        collapse needs no access to the original report later.
        """
        try:
            digest = self._report_digest_lines(report or {})
        except Exception:
            digest = "ok=unknown\nprobes passing: ?/?\nfirst blocker: (digest failed)"
        return (
            f"{_REPORT_BLOCK_BEGIN}\n{digest}\n-->\n"
            f"{text}\n"
            f"{_REPORT_BLOCK_END}"
        )



    def _build_structured_summary(self) -> str:
        """Pi-mono-style structured compaction summary.

        Replaces older raw turns with a fixed-skeleton snapshot built
        deterministically from agent state. Skeleton mirrors pi's
        compaction prompt — Goal / Constraints / Progress / Key Decisions
        / Files / Critical Context — but our build is data-driven (no
        extra LLM round-trip) since we already track every field.

        Useful when iteration count grows past _STRUCTURED_PRUNE_THRESHOLD:
        the model still gets a coherent state anchor instead of a wall of
        elided messages, AND we don't pay a summarizer call. The message
        is injected as role="user" with a loud STATE-ANCHOR prefix —
        Ollama's chat API treats multiple system roles inconsistently
        across providers; a labeled user message is the portable choice.
        """
        lines: list[str] = ["# Session state anchor (older turns elided)", ""]

        lines += ["## Goal", self._goal or "(not set)", ""]

        if self._criteria:
            lines += [
                "## Acceptance criteria (from your Phase A plan)",
                self._criteria.strip(),
                "",
            ]

        if self._probes:
            names = [str(p.get("name", "?")) for p in self._probes]
            lines += [
                "## Executable probes (verifier runs each iter)",
                "  - " + ", ".join(names),
                "",
            ]

        # Progress
        prog: list[str] = ["## Progress"]
        if self._snapshot_n == 0:
            prog.append("- not yet built")
        else:
            if self._previous_report_ok is True:
                prog.append(f"- iteration {self._snapshot_n}: PASSED all tests")
            elif self._previous_report_ok is False:
                prog.append(f"- iteration {self._snapshot_n}: FAILING")
            else:
                prog.append(f"- iteration {self._snapshot_n}: status unknown")
            if self._stuck_streak >= 2:
                prog.append(
                    f"- stuck-streak: {self._stuck_streak} consecutive "
                    "failures on this issue"
                )
            if self.best_path.exists():
                prog.append(
                    f"- last known-good saved at {self.best_path.name} "
                    "(treat as the baseline; don't regress it)"
                )
        lines += prog + [""]

        # Key decisions / diagnoses
        if self._last_diagnose:
            lines += [
                "## Key decisions",
                f"- last diagnose: {self._last_diagnose[:300]}",
                "",
            ]

        # Last test report (truncated — pi-mono caps tool results to ~2000 chars)
        if self._last_report_summary:
            lines += [
                "## Last test report",
                self._last_report_summary[:800],
                "",
            ]

        # Files in session
        files: list[str] = ["## Files in session"]
        cur_size = len(self._current_file)
        files.append(
            f"- {self.out_path.name}: working file ({cur_size:,} bytes)"
        )
        if self.best_path.exists() and self.best_path.name != self.out_path.name:
            files.append(f"- {self.best_path.name}: last clean version")
        lines += files + [""]

        # Generated assets — REQUIRED in summary so the model still
        # knows the PNG names after compaction wipes earlier turns.
        # Without this, "use the art you generated" feedback can't be
        # acted on because the model has forgotten the asset paths.
        if self._session_assets:
            html_dir = self.out_path.resolve().parent
            asset_lines: list[str] = ["## Generated assets (USE these — not procedural fillRect)"]
            for name, path in self._session_assets.items():
                try:
                    rel = Path(path).resolve().relative_to(html_dir)
                except ValueError:
                    rel = path
                asset_lines.append(f"- {name}: ./{rel}")
            asset_lines.append(
                "Load with `new Image()` + `await img.decode()`, then "
                "draw with `ctx.drawImage(img, x, y, w, h)`. Procedural "
                "drawing for entities covered above IS A REGRESSION."
            )
            lines += asset_lines + [""]

        # Generated sounds — same rationale as assets above. Compaction
        # would otherwise drop the OGG paths and the model would forget
        # they exist, shipping a silent game on later iterations.
        if self._session_sounds:
            html_dir = self.out_path.resolve().parent
            sound_lines: list[str] = ["## Generated sounds (USE these — silent games are a regression)"]
            for name, path in self._session_sounds.items():
                try:
                    rel = Path(path).resolve().relative_to(html_dir)
                except ValueError:
                    rel = path
                loop_tag = " (looping)" if name in self._session_looping else ""
                sound_lines.append(f"- {name}: ./{rel}{loop_tag}")
            sound_lines.append(
                "Load via `new Audio('./<name>.ogg')`; play SFX with "
                "`audio.cloneNode().play()` (overlap-safe), looping "
                "music with `audio.loop=true; audio.play()`. Browsers "
                "require a user gesture before audio plays — unlock on "
                "first keydown / pointerdown."
            )
            lines += sound_lines + [""]

        # Last vision-judge verdict — preserved through compaction so
        # the model still knows what the game LOOKED like at the most
        # recent visual check, not just what its probes said. Cheap
        # one-liner; the full screenshot is re-attached on the next
        # VLM-capable turn via _last_screenshot_after.
        if self._last_vision_verdict_iter is not None and self._last_vision_verdict_note:
            prog = self._last_vision_verdict_progress
            tag = (
                "made visible progress" if prog is True
                else ("did NOT make visible progress" if prog is False
                      else "progress unclear")
            )
            lines += [
                "## Visual state at last judge",
                f"- iter {self._last_vision_verdict_iter}: {tag}.",
                f"- still missing/wrong: {self._last_vision_verdict_note}",
                "",
            ]

        # Phase 2: task-progress checklist (structured view of the ledger,
        # whether harness-seeded from goal clauses / outline order or model-
        # emitted). Survives compaction so a long session keeps a visible
        # "done vs left" map. Compact: a count line + the parsed items.
        if self._todos_items:
            done_n = sum(1 for d, _ in self._todos_items if d)
            prog_lines = [
                "## Task progress",
                f"- {done_n}/{len(self._todos_items)} steps done",
            ]
            for d, t in self._todos_items[:12]:
                prog_lines.append(f"  - [{'x' if d else ' '}] {t}")
            lines += prog_lines + [""]

        # Mutable todos artifact (deepagents-style). Replayed across
        # compaction so a long session doesn't lose track of "what's
        # left to ship". Empty when the model never used the tag.
        if self._todos_text:
            lines += [
                "## Open todos (your most recent <todos> snapshot — "
                "re-emit and update it as items complete)",
                self._todos_text[:1500],
                "",
            ]

        # Critical context — preserved across compaction so the model
        # never forgets the truth-source contract.
        lines += [
            "## Critical context",
            "- The CURRENT FILE ON DISK shown inline in the most recent "
            "fix prompt is the source of truth — patch against THAT, "
            "character-for-character. Do NOT trust earlier turns' code.",
            "- Combine related fixes into one multi-patch reply.",
            "- Working > perfect: prefer <done/> after a clean test.",
        ]

        return "\n".join(lines)



    def _build_visual_playtest_prompt(
        self, recipe, before_png: bytes | None, action_png: bytes | None = None,
    ) -> str:
        """Build a structured-checklist VLM prompt from a recipe.

        Small VLMs answer closed-class yes/no questions much more
        reliably than open-ended "what's wrong?" prompts (mortal-
        kombat 2026-05-24 trace had 6 paraphrased complaints in 6
        iters). The recipe carries 6-9 high-signal questions; we
        wrap them in a strict response format and tell the VLM to
        stop after the list.

        When `action_png` is present, a third image (the game captured
        mid-ACTION at peak input-attributable change) is appended by the
        caller. The prompt then routes action/animation questions to that
        image so a brief attack/ability is no longer invisible.
        """
        checklist = recipe.recipe.get("checklist") or []
        # Render as Q1..Qn so the response parser can match without
        # depending on exact question text.
        numbered = "\n".join(
            f"Q{i+1}: {q}" for i, q in enumerate(checklist)
        )
        intro = (
            "You are reviewing one screenshot of a game called: "
            f"{self._goal[:300] or '(no goal specified)'}\n\n"
            "Answer each numbered question below by RE-EMITTING the "
            "question's number with YES, NO, or UNCLEAR, plus an "
            "optional short remark after a dash. ONE LINE per "
            "question, in order. Stop after the last question. Do "
            "NOT add prose, do NOT guess at code causes, do NOT "
            "describe the background unless a question asks.\n\n"
        )
        if before_png is not None and action_png is not None:
            intro = (
                "You are reviewing THREE screenshots of a game called: "
                f"{self._goal[:300] or '(no goal specified)'}\n\n"
                "Image 1: BEFORE simulated input. Image 2: AFTER (resting "
                "state). Image 3: the PEAK ACTION frame, captured mid-input "
                "when on-screen change was largest.\n\n"
                "Answer each numbered question by RE-EMITTING the question's "
                "number with YES, NO, or UNCLEAR. ONE LINE per question, in "
                "order. Judge ACTION / ANIMATION / attack questions against "
                "Image 3; judge LAYOUT / HUD / resting-position questions "
                "against Image 2. Image 3 IS the active-input frame, so for "
                "'is the action visible' questions, commit to YES or NO rather "
                "than UNCLEAR. Stop after the last question.\n\n"
            )
        elif before_png is not None:
            intro = (
                "You are reviewing TWO screenshots of a game called: "
                f"{self._goal[:300] or '(no goal specified)'}\n\n"
                "Image 1: BEFORE simulated input. Image 2: AFTER.\n\n"
                "Answer each numbered question by RE-EMITTING the "
                "question's number with YES, NO, or UNCLEAR. ONE "
                "LINE per question, in order. Refer to Image 2 (the "
                "AFTER image) for each answer. Stop after the last "
                "question.\n\n"
            )
        elif action_png is not None:
            intro = (
                "You are reviewing TWO screenshots of a game called: "
                f"{self._goal[:300] or '(no goal specified)'}\n\n"
                "Image 1: the resting state. Image 2: the PEAK ACTION frame, "
                "captured mid-input when on-screen change was largest.\n\n"
                "Answer each numbered question by RE-EMITTING the question's "
                "number with YES, NO, or UNCLEAR. ONE LINE per question, in "
                "order. Judge ACTION / ANIMATION / attack questions against "
                "Image 2 and commit to YES or NO rather than UNCLEAR. Stop "
                "after the last question.\n\n"
            )
        # Anti-rubber-stamp: when NO mid-action frame was captured but the
        # goal/criteria imply the game has actions (attacks/abilities), the
        # VLM must NOT confirm an action is visible from two resting frames —
        # that is exactly how a static, never-animated attack got "Q5: YES"
        # every iteration in the 2026-05-29 fighting-game trace.
        if action_png is None and self._animation_expected():
            intro = intro + (
                "NOTE: no active-input (mid-action) frame was captured this "
                "run. If a question asks whether an ACTION / ATTACK / "
                "ABILITY / animation (e.g. a walk cycle, kick, punch) is "
                "VISIBLE or PLAYING, answer NO or UNCLEAR —"
                " do NOT answer YES, because there is no mid-action frame "
                "here to confirm it.\n\n"
            )
        # Pixel analysis already measured the generated animation frames; if
        # any came back near-identical to idle, tell the VLM so its judgment
        # and the objective floor reinforce each other.
        if self._dead_anim_frames:
            names = ", ".join(sorted(self._dead_anim_frames))
            intro = intro + (
                "NOTE: pixel analysis found these animation frames nearly "
                f"identical to the idle pose — {names}. Their limbs likely do "
                "not move, so the character would slide as a static image. "
                "Weigh that when judging any animation question.\n\n"
            )
        example = (
            "Example response shape:\n"
            "Q1: yes\n"
            "Q2: no — player overlaps the right wall\n"
            "Q3: unclear\n"
            "...\n\n"
        )
        return intro + numbered + "\n\n" + example

    # Phrases a VLM uses when it did NOT actually receive/see an image. When
    # any of these appear, the critique is an ABSTAIN, not a judgement — its
    # Qn: answers (if any) are fabricated and must never be parsed as real
    # visual failures or fed to the coaching loop. Added 2026-06-02 after the
    # Street Fighter trace parsed a "can't see the image" reply that ALSO
    # emitted Q1:no..Q5:no as a genuine "5 of 5 checks FAILED". Genre/model
    # agnostic — these are generic refusal/blindness phrasings.
    # TIGHTENED 2026-06-03: the abstain test must fire ONLY when the model says
    # it can't see the IMAGE/SCREENSHOT itself — never on a legitimate visual
    # observation about the game's contents. The old `(?:can't|don't) …see`
    # clause false-matched "I don't see a projectile in the slingshot" (a CORRECT
    # finding from a model that saw the screenshot fine) and wrongly dropped the
    # whole critique. Every blindness alternative below is anchored to an
    # image/screenshot/picture object, so "don't see a <game element>" no longer
    # trips it. Verified against the angry-birds critic trace where the VLM
    # described the slingshot accurately yet was being discarded as "blind".
    _IMG = r"(?:image|images|screenshot|screenshots|picture|pictures|photo|attachment)"
    _CRITIC_ABSTAIN_RE = __import__("re").compile(
        r"(?:"
        rf"no {_IMG}\b|"
        rf"without (?:a |the |any )?{_IMG}\b|"
        rf"(?:can(?:no|')t|cannot|could ?n'?t|unable to|do not|don'?t) (?:\w+ ){{0,3}}(?:see|view|access|open|load|analyze|review|make out) (?:\w+ ){{0,3}}{_IMG}\b|"
        rf"(?:did|do) ?n'?t (?:receive|get|see) (?:the |a |an |any )?{_IMG}\b|"
        rf"no {_IMG} (?:was |were )?(?:provided|attached|included|shared|uploaded|present)|"
        rf"(?:please |kindly )?(?:share|attach|provide|upload|re-?upload|send)(?: (?:the|a|an|your))? {_IMG}\b|"
        rf"i (?:don'?t|do not) have (?:access to )?(?:an? |the |any )?{_IMG}\b|"
        rf"if you (?:can |could )?(?:share|attach|provide|send)(?: (?:the|a|an|your))? {_IMG}\b|"
        r"no (?:visual|file) (?:was )?(?:provided|attached|included)"
        r")",
        __import__("re").IGNORECASE,
    )

    def _compact_warnings_for_prompt(self, warnings: list[str]) -> list[str]:
        """Return a model-facing warnings list with persistent items
        replaced by a one-line collapsed form. Does NOT advance
        counters — call `_advance_warning_persistence` separately
        once per iter. Original `warnings` list is not mutated.
        """
        out: list[str] = []
        for w in warnings or []:
            text = str(w)
            count = self._warning_persistence.get(self._hash_warning(text), 0)
            if count >= self._WARNING_COMPACT_THRESHOLD:
                # First non-empty line, capped — enough for the model to
                # remember which warning this is without re-reading the
                # full body each iter.
                first_line = text.strip().splitlines()[0] if text.strip() else ""
                preview = first_line[:80].rstrip()
                out.append(
                    f"persistent warning [seen {count}× in a row]: {preview}…"
                )
            else:
                out.append(text)
        return out



    def _format_report_for_model(self, report: dict) -> str:
        """Wrapper around `tools.format_report_for_model` that compacts
        persistent harness warnings before formatting. Trace and any
        other consumers of the original `report` see full warnings;
        only the prompt-rendering path uses the compacted view.

        Harness report slimming (chess-trace fix 2026-06-22): on a turn
        that carries genuine USER feedback, drop the cosmetic ASSET SANITY
        advisories from the non-gating `warnings` channel so the user's ask
        isn't buried under harness narration (iter 2 had ~15.6K chars of
        report text competing with the user's one-line layout request). The
        advisory is non-gating, so `ok` is unaffected; the trace still keeps
        the full report.
        """
        if not report:
            return format_report_for_model(report)
        warnings = self._compact_warnings_for_prompt(
            report.get("warnings") or []
        )
        # Genuine user feedback pending? (agent-internal notices excluded.)
        _internal = getattr(self, "_internal_feedback_texts", set())
        _user_pending = [
            fb for fb in (getattr(self, "_pending_feedback", None) or [])
            if fb not in _internal
        ]
        if _user_pending:
            slimmed = [
                w for w in warnings
                if "ASSET SANITY" not in str(w)
                and not str(w).startswith(_HARNESS_ADVISORY_SENTINEL)
            ]
            if len(slimmed) != len(warnings):
                self._trace({
                    "kind": "report_slimmed_for_feedback_turn",
                    "dropped": len(warnings) - len(slimmed),
                    "kept": len(slimmed),
                })
            warnings = slimmed
        rfp = dict(report)
        rfp["warnings"] = warnings
        return format_report_for_model(rfp)

    # Generic, domain-neutral tokens that indicate the VLM note is
    # ACTIONABLE coaching rather than pure screenshot narration. Used
    # by `_clean_actionable_vision_note` below as a relevance gate
    # alongside the existing structural filters.
    #
    # Evidence: fighing-game trace 20260519_153115 surfaced two purely
    # descriptive notes that the prior filter let through:
    #   iter 3: "Controls are listed at the bottom"
    #   iter 4: "Both images show very low-resolution, pixel-art style"
    # Both told the model nothing it could act on. The first iter's
    # useful note ("look very basic ... There are no complex super…")
    # passes this gate via the negation token "no".
    #
    # The list is intentionally short and general — not goal-derived,
    # not genre-specific. False negatives (dropping a legitimately
    # useful note) are preferable to false positives (injecting
    # descriptive narration as coaching) on local 27B-class models
    # that already have limited context to spend on guidance.
    _VISION_NOTE_ACTIONABLE_TOKENS: frozenset[str] = frozenset({
        # Negation / absence.
        "no", "none", "nothing", "without", "missing", "lacks",
        "lack", "never",
        # Direct change verbs.
        "add", "fix", "make", "replace", "remove", "move", "change",
        "improve",
        # Need / should / want.
        "needs", "need", "should", "must",
        # Comparative-implies-change.
        "too", "over", "under",
    })

    @staticmethod
    def _identifiers(text: str) -> set[str]:
        """Pull plausible identifier tokens from arbitrary text. Used to
        bias the focused-slice toward the function bodies the model is
        most likely to need to patch."""
        if not text:
            return set()
        # Skip JS keywords + small/numeric tokens that aren't useful.
        skip = {
            "true", "false", "null", "undefined", "function", "return",
            "const", "let", "var", "if", "else", "for", "while", "this",
            "new", "void", "from", "with", "in", "of", "do", "try", "catch",
            "throw", "break", "continue", "switch", "case", "default",
            "Math", "console", "window", "document", "Object", "Array",
            "true", "yes", "no",
        }
        out: set[str] = set()
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text):
            if tok in skip:
                continue
            out.add(tok)
        return out



    @staticmethod
    def _identifier_occurrence_slice(
        html: str,
        identifiers: list[str] | tuple[str, ...],
        *,
        radius: int = 1,
        max_chars: int = 2400,
    ) -> str:
        """Return a compact line-window block containing all identifier hits."""
        if not html or not identifiers:
            return ""
        lines = html.splitlines()
        chosen: set[int] = set()
        idents = [i for i in identifiers if i]
        for i, ln in enumerate(lines):
            if any(tok in ln for tok in idents):
                lo = max(0, i - radius)
                hi = min(len(lines), i + radius + 1)
                for j in range(lo, hi):
                    chosen.add(j)
        if not chosen:
            return ""
        ordered = sorted(chosen)
        out_lines: list[str] = []
        used = 0
        for j in ordered:
            row = f"{j + 1:5d}: {lines[j]}"
            row_len = len(row) + 1
            if used + row_len > max_chars:
                break
            out_lines.append(row)
            used += row_len
        return "\n".join(out_lines)



    @staticmethod
    def _signature_focus_identifiers(sig: str) -> list[str]:
        """Extract dotted identifier paths from a failure signature."""
        if not sig:
            return []
        browser_noise = {
            "Page.evaluate",
            "UtilityScript.evaluate",
            "UtilityScript.anonymous",
            "Page.ev",
        }
        browser_prefixes = ("UtilityScript.", "Page.")
        out: list[str] = []
        seen: set[str] = set()
        for tok in re.findall(
            r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*){1,4}",
            sig,
        ):
            if tok in browser_noise or tok.startswith(browser_prefixes):
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= 8:
                break
        return out

    @staticmethod


    @staticmethod
    def _diagnose_is_shotgun(diag: str) -> bool:
        """Detect ranked-hypothesis shape: >=3 lines starting with `1.`,
        `2)`, `(3)`, `- `, or `* ` patterns. Mid-size models default to
        shotgun lists when the prompt allows it; the new fix_instruction
        forbids them, but we keep this detector so we can both log the
        violation and tighten the next user turn's reminder.
        """
        if not diag:
            return False
        list_re = re.compile(r"^\s*(?:[-*]|\(?[1-9][0-9]?\)?[.)])\s+\S", re.MULTILINE)
        return len(list_re.findall(diag)) >= 3

    @staticmethod


    def _diagnose_mentions_subsystem(
        diagnose_text: str | None,
        identifiers: tuple[str, ...] | list[str],
    ) -> bool:
        """True when the `<diagnose>` body contains ANY of the hint's
        identifier tokens (case-insensitive substring match).

        Used by the diagnose-vs-patch coherence check to detect when
        the model's stated root cause ignores the subsystem the harness
        has been flagging. DK trace 20260514_175012 turn [08]:
        `<diagnose>` named "barrel drop threshold + procedural fallback
        coordinate bug" while the harness had been reporting INPUT
        failure for 3 iterations — the diagnose contained none of
        addEventListener/keydown/KeyboardEvent/keys.
        """
        if not diagnose_text or not identifiers:
            return False
        low = diagnose_text.lower()
        return any(i.lower() in low for i in identifiers)

    @staticmethod


    def _patches_touch_subsystem_idents(
        patches_in_reply,
        identifiers: tuple[str, ...] | list[str],
    ) -> bool:
        """True when ANY patch's SEARCH or REPLACE text contains ANY
        of the hint's identifier tokens. Lowercase substring match
        — the identifiers in `_SUBSYSTEM_HINTS` are short and mostly
        unambiguous within JS code (`addEventListener`, `keydown`,
        etc.). Scanning BOTH sides catches both:
          - patches that touch existing input code (SEARCH matches),
          - patches that ADD input wiring to a region without it
            (REPLACE matches).
        """
        if not patches_in_reply or not identifiers:
            return False
        ident_set_low = [i.lower() for i in identifiers]
        for p in patches_in_reply:
            blob = ((p.search or "") + "\n" + (p.replace or "")).lower()
            if any(i in blob for i in ident_set_low):
                return True
        return False

    def _seed_html_for_prompt(
        self, seed_html: str, report: dict | None = None,
    ) -> tuple[str, bool]:
        """Bound the seed file inlined into the first /seed build prompt.

        A large working seed (this 2026-06-21 trace: ~26KB) inlined in full
        balloons iter-1 context to 15K+ prompt tokens and drives the weak
        local model into repetition/deliberation loops. Returns
        (html_for_prompt, truncated). Genre-free, structural:

        - Under _SEED_FULL_FILE_INJECT_LIMIT: full file (so EVERY patch
          target stays visible — the primary fix for the 2026-06-25 forced-
          rewrite loop; see the constant's comment above).
        - Otherwise: when a failing report exists, prefer the focused slice;
          else a head+tail excerpt (DOM/CSS anchors live near the top, boot
          code near the bottom — the regions UI/additive patches anchor to).
        The full file stays on disk; the prompt tells the model to patch
        against the on-disk file, not the excerpt.
        """
        if not seed_html or len(seed_html) <= self._SEED_FULL_FILE_INJECT_LIMIT:
            return seed_html, False
        limit = self._SEED_FULL_FILE_INJECT_LIMIT
        if report and not report.get("ok", True):
            try:
                sliced = self._focused_slice(
                    seed_html, report, self._criteria or "",
                )
                if sliced and len(sliced) <= limit:
                    return sliced, True
            except Exception:
                pass
        # Head + tail with elided middle. Head gets the larger share so the
        # <head>/<style>/<body>-open structure (where UI anchors live) is
        # intact; tail keeps the boot/init region.
        head_budget = int(limit * 0.7)
        tail_budget = limit - head_budget
        head = seed_html[:head_budget]
        tail = seed_html[-tail_budget:]
        elided = len(seed_html) - head_budget - tail_budget
        return (
            head
            + f"\n\n<!-- ... {elided} chars elided — full file is on disk; "
              "patch against the on-disk file, do not rewrite ... -->\n\n"
            + tail,
            True,
        )


    def _focused_slice(self, html: str, report: dict, criteria: str) -> str | None:
        """Build a focused slice of the current file, biased toward the
        functions / regions implicated by the failing probes, the
        console/page errors, and the model's own <criteria>.

        Returns None when slicing isn't worth it (file is small, or no
        signals to focus on, or the slice would cover most of the file
        anyway). Caller in that case sends the full file.

        Stays genre-free: identifier matching is purely structural —
        no hardcoded function names or game-type heuristics.
        """
        if not html or len(html) <= self._FULL_FILE_INJECT_LIMIT:
            return None
        # Collect failure signals.
        sig_text_parts: list[str] = []
        for k in ("errors", "console_errors", "page_errors", "soft_warnings"):
            v = report.get(k) or []
            if isinstance(v, list):
                sig_text_parts.extend(str(x) for x in v)
            else:
                sig_text_parts.append(str(v))
        for p in (report.get("probes") or []):
            if not p.get("ok"):
                sig_text_parts.append(str(p.get("expr", "")))
                sig_text_parts.append(str(p.get("error", "")))
        # Criteria identifiers protect against the asteroids regression:
        # `vx = cos(angle)*speed` stays in scope even when the failing
        # probe doesn't mention `vx` directly.
        keyset = self._identifiers("\n".join(sig_text_parts)) | self._identifiers(criteria or "")
        # Subsystem-hint biasing (DK trace 20260514_104131): when the
        # most-recent mistake_signature implicates a specific code
        # region (input handler, RAF loop, etc.), pull its identifier
        # tokens into the keyset. The slice then surfaces functions
        # in that region even when the error signals don't directly
        # name them — so the model SEES the keydown handler on iter 1,
        # not just the climb-math function the user's complaint named.
        sig_hint = _subsystem_hint(getattr(self, "_last_mistake_sig", "") or "")
        if sig_hint:
            keyset = keyset | set(sig_hint["identifiers"])
            self._trace({
                "kind": "subsystem_hint_biased_slice",
                "subsystem": sig_hint["name"],
                "added_identifiers": list(sig_hint["identifiers"]),
            })
        if not keyset:
            return None

        # Score each <script> body's function definitions.
        scripts = re.findall(
            r"(<script[^>]*>)(.*?)(</script>)",
            html, re.DOTALL | re.IGNORECASE,
        )
        # Function-shaped chunks: `function foo(...) { ... }`,
        # `const foo = (...) => { ... }`, `foo() { ... }` (method).
        # Keep the regex simple — we match opening lines then balance braces.
        #
        # We extract EVERY plausible function into `pool` (regardless of
        # score) so the callee-promotion step below can pull in functions
        # called by selected ones. `chunks` is the score>0 subset that
        # enters the ranking. This matters when the model is debugging a
        # symptom whose error signals don't name the buggy function
        # (asteroid trace: input dead → no mention of update() → update()
        # scored 0 → dropped → model spent 4 iters blind).
        chunks: list[tuple[int, str, str]] = []  # (score, name, body_with_header)
        pool: list[tuple[str, str]] = []          # (name, body_with_header) — ALL extracted
        for (_open, body, _close) in scripts:
            for m in re.finditer(
                r"(?:function\s+([A-Za-z_$][\w$]*)|"
                r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^)]*\)?\s*=>|"
                r"([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{)",
                body,
            ):
                name = m.group(1) or m.group(2) or m.group(3) or ""
                # Skip control-flow keywords that the third pattern
                # spuriously matches (`if (cond) { ... }`,
                # `for (init; cond; step) { ... }` etc).
                if not name or name in {
                    "if", "else", "for", "while", "switch", "do",
                    "try", "catch", "finally", "return", "throw",
                }:
                    continue
                start = m.start()
                # Find the opening `{` and balance to find the body end.
                brace_at = body.find("{", start)
                if brace_at < 0 or brace_at - start > 200:
                    continue
                depth = 0
                end = brace_at
                for i in range(brace_at, min(len(body), brace_at + 4000)):
                    c = body[i]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                segment = body[start:end]
                if len(segment) < 30 or len(segment) > 3500:
                    continue
                pool.append((name, segment))
                seg_ids = self._identifiers(segment)
                hits = len(seg_ids & keyset)
                # Always include functions whose NAME hits a key.
                if name in keyset:
                    hits += 5
                if hits == 0:
                    continue
                chunks.append((hits, name, segment))

        if not chunks:
            return None
        chunks.sort(key=lambda t: -t[0])

        # One-hop callee promotion: when a selected function calls another
        # extracted function that didn't score on its own, pull the callee
        # into the candidate list at a low score (1 — below genuine name/
        # identifier hits, above the cut). This fixes the case where the
        # error signals describe a symptom (`canvas didn't change`) rather
        # than the buggy function's name, so the function with the bug
        # never enters the slice.
        all_names = {n for (n, _seg) in pool}
        # Compute promotion target set from the existing top picks. We use
        # all current chunks (not just the top-K) because the byte cap
        # below may drop some — we want every potential selection to
        # carry its callees in. Caller-of-callee scanning is bounded by
        # the regex on each chunk's body, so it's cheap.
        selected_names: set[str] = {n for (_s, n, _seg) in chunks}
        called: set[str] = set()
        # Pre-compile to avoid recompiling on every chunk.
        callee_re = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
        for (_s, _n, seg) in chunks:
            for cm in callee_re.finditer(seg):
                cn = cm.group(1)
                if cn in all_names and cn not in selected_names:
                    called.add(cn)
        if called:
            # Use a name → body map so we look each callee up once.
            pool_map: dict[str, str] = {}
            for (n, seg) in pool:
                pool_map.setdefault(n, seg)  # first definition wins on duplicates
            added = 0
            for cn in called:
                seg = pool_map.get(cn)
                if seg is None:
                    continue
                chunks.append((1, cn, seg))
                added += 1
            if added:
                # Trace this so we can see in jsonl when callee promotion
                # rescued a function the symptom-based scoring missed.
                self._trace({
                    "kind": "focused_slice_callees_added",
                    "callees": sorted(called),
                    "count": added,
                })
                chunks.sort(key=lambda t: -t[0])

        kept: list[str] = []
        used = 0
        for (_score, name, seg) in chunks:
            if used + len(seg) > 5000:
                break
            kept.append(f"// --- function `{name}` (focused slice) ---\n{seg}")
            used += len(seg) + 60
            # Cap raised from 3 → 5 to absorb 1–2 callee promotions
            # without pushing out higher-signal functions. Byte cap
            # (5000) still gates total size.
            if len(kept) >= 5:
                break
        if not kept:
            return None
        if used > len(html) * 0.6:
            return None

        # If the selected logic reads `state.foo`, also show nearby writes
        # to `state.foo` elsewhere in the file. Game bugs often hide in
        # reset/init code that silently overwrites the value the model is
        # trying to fix in update/draw logic.
        kept_text = "\n\n".join(kept)
        state_props = {
            p for p in re.findall(r"\bstate\.([A-Za-z_$][\w$]*)\b", kept_text)
            if len(p) >= 3
        }
        assignment_snips: list[str] = []
        seen_windows: set[tuple[int, int, int]] = set()
        if state_props:
            prop_alt = "|".join(re.escape(p) for p in sorted(state_props))
            state_write_re = re.compile(
                rf"\bstate\.({prop_alt})(?:\.[A-Za-z_$][\w$]*)*\s*(?:[+\-*/%]?=|\+\+|--)"
                rf"|\b({prop_alt})\s*:",
            )
            for script_idx, (_open, body, _close) in enumerate(scripts):
                lines = body.splitlines()
                for line_idx, line in enumerate(lines):
                    if not state_write_re.search(line):
                        continue
                    stripped = line.strip()
                    if stripped and stripped in kept_text:
                        continue
                    start = max(0, line_idx - 2)
                    end = min(len(lines), line_idx + 3)
                    key = (script_idx, start, end)
                    if key in seen_windows:
                        continue
                    seen_windows.add(key)
                    snippet = "\n".join(lines[start:end])
                    if len("\n\n".join(assignment_snips + [snippet])) > 1600:
                        break
                    assignment_snips.append(snippet)
                if len("\n\n".join(assignment_snips)) > 1500:
                    break
        if assignment_snips:
            self._trace({
                "kind": "focused_slice_state_assignments_added",
                "state_props": sorted(state_props),
                "count": len(assignment_snips),
            })
            kept_text += (
                "\n\n// --- related state assignments (focused slice) ---\n"
                + "\n\n".join(assignment_snips)
            )
        return kept_text

    def _partial_patch_recovery_block(
        self,
        partial_failed: list[tuple[int, object, str]],
    ) -> str:
        """Prompt addendum for partial patch application retries."""
        if not partial_failed or not self._current_file:
            return ""
        from patches import find_anchor  # local import to avoid module cycle

        lines = [
            "PATCH-APPLY RECOVERY (previous reply partially applied):",
            "Send ONE consolidated <patch> that fixes the unresolved region.",
            "Do NOT scatter multiple overlapping patches for this retry.",
        ]
        for (i, p, reason) in partial_failed[:3]:
            lines.append(f"- unresolved patch #{i + 1}: {reason}")
            search = (getattr(p, "search", "") or "").strip()
            if search:
                preview = search.splitlines()[0][:180]
                lines.append(f"  failed SEARCH head: {preview!r}")
                anchor = find_anchor(self._current_file, search)
                if anchor:
                    lines.append("  nearest current-file anchor:")
                    lines.extend(f"    {ln}" for ln in anchor.splitlines()[:8])
        if len(partial_failed) > 3:
            lines.append(f"- (+{len(partial_failed) - 3} more unresolved patches)")
        return "\n".join(lines)



    def _repeat_error_fastpath_block(self, report: dict) -> str:
        """Force a narrow one-patch retry after repeated same-signature failures."""
        if self._repeat_sig_streak < 2 or not self._current_file:
            return ""
        sig = self._last_mistake_sig or signature_for_report(report) or ""
        hint = _subsystem_hint(sig)
        identifiers: list[str] = []
        if hint:
            identifiers.extend(list(hint["identifiers"]))
        identifiers.extend(self._signature_focus_identifiers(sig))
        deduped: list[str] = []
        seen: set[str] = set()
        for tok in identifiers:
            if tok in seen:
                continue
            seen.add(tok)
            deduped.append(tok)
            if len(deduped) >= 12:
                break
        occurrence_block = self._identifier_occurrence_slice(
            self._current_file,
            deduped,
        )
        lines = [
            "REPEATED-ERROR FAST PATH:",
            f"The same failure signature has repeated for {self._repeat_sig_streak} consecutive iterations.",
            "THIS TURN: emit exactly ONE minimal <patch> targeting the failing symbol/path.",
            "Do NOT refactor, rename unrelated code, or emit a full <html_file>.",
        ]
        if hint:
            lines.append(f"Target subsystem: {hint['name']} ({hint['fix_phrase']}).")
        if deduped:
            lines.append("Implicated identifiers: " + ", ".join(f"`{x}`" for x in deduped[:8]))
        if occurrence_block:
            lines.append("All matching identifier occurrences in CURRENT FILE:")
            lines.append("```text")
            lines.append(occurrence_block)
            lines.append("```")
        return "\n".join(lines)

    def _build_fix_prompt(
        self,
        *,
        report: dict,
        regressed: bool,
        partial_failed: list[tuple[int, object, str]],
    ) -> str:
        """Construct the next user message after a test result.

        Branches:
          - **stuck hard-gate** → force <question>-only turn when the
                                  same subsystem has failed 3+ times
                                  (Item 2, plan 20260514_175012)
          - report ok           → post_clean (encourage <done/>)
          - regressed           → revert prompt with last-good code inline
          - structurally broken → truncation-recovery prompt that does
                                  NOT inline the broken file (saves ~5-8K
                                  BPE tokens of prompt and removes a
                                  misleading "truth source")
          - failed              → diagnose-then-fix combined turn (with
                                  mistake hints and current file inline;
                                  VLM note appended if applicable)
        """
        # Hard-gate check — fires before any other branch. When the
        # subsystem-hint coaching has been ignored for 3 iterations
        # in a row, force the model into <question>-only mode this
        # turn. Pulls the human in to break the loop. Flag is set in
        # the streak-handling branch and cleared here on consumption.
        if self._force_question_subsystem is not None:
            hint = self._force_question_subsystem
            self._force_question_subsystem = None  # consume once
            idents = ", ".join(f"`{i}`" for i in hint["identifiers"][:5])
            self._trace({
                "kind": "stuck_hard_gate_prompt_built",
                "subsystem": hint["name"],
            })
            return (
                "STUCK-LOOP HARD GATE — the harness has reported the "
                f"same {hint['name'].upper()} failure for "
                f"{self._repeat_sig_streak} consecutive iterations and "
                "your patches haven't addressed it. The implicated "
                f"region matches identifiers: {idents}.\n\n"
                "THIS TURN you MUST emit exactly ONE "
                "<question>...</question> tag asking the user one of "
                "the following (pick the one that best matches your "
                "uncertainty):\n"
                f"  (a) \"Should I rewrite {hint['fix_phrase']} from "
                "scratch?\"\n"
                "  (b) \"Is there a specific approach you want me to "
                f"try for the {hint['name']} subsystem?\"\n"
                "  (c) \"Do you want to ship the partial game as-is "
                "and accept the known failure?\"\n\n"
                "Do NOT emit <patch>, <html_file>, <plan>, "
                "<diagnose>, or any other tag this turn. The "
                "session will resume after the user answers. The "
                "most recent test report (for context — do NOT "
                "act on it this turn):\n\n"
                f"{self._format_report_for_model(report)}"
            )

        report_text = self._format_report_for_model(report)

        # SCOPED-CHANGE override: when the user explicitly locked the
        # turn ("no code changes", "only X"), the failing-probes report
        # must NOT be framed as "fix these failures" — the user told
        # the model to ignore them this turn. Without this gate, the
        # full report text travels downstream as "fix these" context
        # and competes with the SCOPED-CHANGE directive that says
        # "ignore them". DK trace 2026-05-15 iter 3 is the case study:
        # 4 KB of failing-probe text drowned out the user's scope-lock
        # and the model "fixed" 5 things instead of the 1 thing asked.
        #
        # We still record that issues existed (1 short line of
        # context) so the model knows the session isn't shippable yet —
        # just not actively pushed to fix them this turn.
        if self._scoped_change_active and not report["ok"]:
            n_errs = len(report.get("errors") or [])
            n_warn = len(report.get("soft_warnings") or [])
            probes = report.get("probes") or []
            n_probe_fail = sum(1 for p in probes if not p.get("ok"))
            report_text = (
                "NOTE: previous iter had "
                f"{n_errs} error(s), {n_warn} soft warning(s), and "
                f"{n_probe_fail} failing probe(s). The user has scoped "
                "THIS turn to ONLY the change in the USER FEEDBACK and "
                "SCOPED-CHANGE blocks above — do NOT address the "
                "previous iter's failures this turn. They will come "
                "back into scope on a later turn once the user has "
                "verified the scoped change landed."
            )
            self._trace({
                "kind": "scoped_change_report_suppressed",
                "n_errors": n_errs,
                "n_soft_warnings": n_warn,
                "n_probes_failed": n_probe_fail,
            })
            # Consume the flag — only applies to THIS fix-prompt build.
            self._scoped_change_active = False

        if report["ok"]:
            # Todo-driven execution (2026-06-12): when the model's own
            # <todos> list still has unchecked items after a clean iter,
            # name the FIRST unchecked one as the turn's CURRENT TASK —
            # one objective per turn instead of "everything not yet
            # done". Frontier-agent pattern; biggest reliability lever
            # for 27B-class local models. Fires before polish (open work
            # beats game-feel polish). Skipped when user feedback is
            # pending (their wish wins) or the user asked to ship. The
            # contract itself re-offers <done/> so it never blocks
            # shipping.
            # Gate on GENUINE user input only (2026-06-12): agent-
            # generated findings ride the same feedback queue and were
            # starving this contract at every clean iter. An internal
            # finding still flows into this turn via
            # _flush_user_injections — the model gets the finding AND
            # one scoped task.
            #
            # P1b (run_04 holochess/Dragon/snake): if the harness SEEDED the
            # ledger (the model never emitted its own <todos>) and this iter is
            # honestly clean — all model probes passed AND no page errors — the
            # working build already satisfies the seeded checklist. Auto-close
            # it and skip the todo contract instead of nagging the model to
            # "work ONLY on" a goal fragment for another full turn (each of
            # those games burned ~1 turn re-marking todos [x] on a 7/7-probe
            # build). The model's OWN self-declared <todos> are left untouched.
            if (
                getattr(self, "_todos_seeded_by_harness", False)
                and self._todos_items
                and any(not _d for _d, _ in self._todos_items)
            ):
                _all_probes_ok = all(
                    bool(p.get("ok")) for p in (report.get("probes") or [])
                )
                _no_page_errors = not (report.get("page_errors") or [])
                if _all_probes_ok and _no_page_errors:
                    self._todos_items = [
                        (True, _t) for _d, _t in self._todos_items
                    ]
                    self._todos_text = "\n".join(
                        f"- [x] {_t}" for _d, _t in self._todos_items
                    )
                    self._trace({
                        "kind": "task_ledger_auto_closed",
                        "reason": "clean_iter_all_probes_passed",
                        "closed": len(self._todos_items),
                    })
            _todo_task = self._select_next_todo()
            if (
                _todo_task is not None
                and self._iters_remaining >= 1
                and not self._has_genuine_user_input()
                and not self._user_force_done
            ):
                _key = self._norm_todo(_todo_task)
                self._todo_nag_counts[_key] = (
                    self._todo_nag_counts.get(_key, 0) + 1
                )
                self._current_todo = _todo_task
                n_open = sum(
                    1 for d, _t in self._todos_items if not d
                )
                self._trace({
                    "kind": "todo_contract_injected",
                    "todo": _todo_task[:200],
                    "open_count": n_open,
                    "nag_count": self._todo_nag_counts[_key],
                })
                base = self._p.post_clean_instruction(report_text)
                contract = (
                    "\n\nCURRENT TASK (from your own <todos> list — "
                    f"{n_open} item(s) still open):\n"
                    f"  {_todo_task}\n"
                    "Work ONLY on this item this turn: emit <patch> "
                    "blocks scoped to it, then re-emit the FULL <todos> "
                    "list with it marked [x]. If it is already complete "
                    "or no longer worth doing, just re-emit <todos> "
                    "with it marked [x] (or removed) — and if nothing "
                    "real remains, ship with <done/>."
                )
                cf = self._current_file or ""
                if cf and len(cf) <= 60_000:
                    # Same truth-source inject as the pending-feedback
                    # path below: a <patch> is likely this turn.
                    return (
                        f"{base}{contract}\n\n"
                        "CURRENT FILE ON DISK (this is the SOURCE OF "
                        "TRUTH — if you emit a <patch>, its SEARCH must "
                        "match THIS exact text, character-for-character; "
                        "earlier turns' code may be stale):\n"
                        "```html\n"
                        f"{cf}\n"
                        "```\n"
                    )
                return f"{base}{contract}"
            # Capability-round item 2: polish phase. Probes are green,
            # iteration budget remains, and the per-session polish cap is
            # unmet — spend a turn on game feel instead of pushing <done/>.
            # Skipped when user feedback is pending (their wish wins) or
            # the user already asked to ship. Never blocks shipping: the
            # prompt itself re-offers <done/>.
            if (
                self._polish_turns_used < _POLISH_TURN_CAP
                and self._iters_remaining >= 1
                and not self.has_pending_user_input()
                and not self._user_force_done
                and hasattr(self._p, "polish_instruction")
            ):
                self._polish_turns_used += 1
                self._polish_pending = True
                # 1 juice component (item 1 synergy): query goal + feel
                # terms so the snippet fits the game's modality.
                juice_block = self._retrieve_components_block(
                    f"{self._goal} juice feel polish particles screen shake "
                    "easing tween audio hit feedback",
                    stage="code", k=1,
                )
                cf = self._current_file or ""
                self._trace({
                    "kind": "polish_turn_started",
                    "turn": self._polish_turns_used,
                    "cap": _POLISH_TURN_CAP,
                    "iters_remaining": self._iters_remaining,
                    "has_critic_note": bool(self._last_critic_note),
                    "has_component": bool(juice_block),
                })
                return self._p.polish_instruction(
                    report_text,
                    current_file=cf if (cf and len(cf) <= 60_000) else "",
                    critic_note=self._last_critic_note or "",
                    component_block=juice_block,
                    turn=self._polish_turns_used,
                    cap=_POLISH_TURN_CAP,
                )
            # Truth-source inject for post-clean follow-up turns.
            # Evidence: fighing-game trace 20260519_153115 iter 3→4 — the
            # post_clean instruction does NOT inline the current file, so
            # when the user gave feedback after a clean iter the model
            # patched against memory, hallucinated drawFighter's structure,
            # and SEARCH failed (1/2 patches applied). Same pattern as
            # continuation_instruction / fix_instruction — give the model
            # the on-disk truth so its <patch> SEARCH matches.
            base = self._p.post_clean_instruction(report_text)
            cf = self._current_file or ""
            # Only inject when (a) feedback is queued or pending so a
            # <patch> is likely this turn, AND (b) the file is non-empty
            # and not so huge it would blow the context. Keep the cap
            # generous so we rarely skip.
            file_likely_used = bool(self._pending_feedback) or bool(
                getattr(self, "_pending_answer", None)
            )
            if cf and file_likely_used and len(cf) <= 60_000:
                self._trace({
                    "kind": "post_clean_truth_source_injected",
                    "file_bytes": len(cf),
                    "reason": "pending_feedback",
                })
                return (
                    f"{base}\n\n"
                    "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — "
                    "if you emit a <patch>, its SEARCH must match THIS "
                    "exact text, character-for-character; earlier turns' "
                    "code may be stale):\n"
                    "```html\n"
                    f"{cf}\n"
                    "```\n"
                )
            return base

        if regressed:
            best = self._read_best_or_empty()
            base = self._p.regression_instruction(report_text, best)
            if getattr(self, "_post_clean_shrink_detected", False):
                base = (
                    "POST-CLEAN REGRESSION: your last patch broke a WORKING "
                    "build (file shrank >20% — likely accidental truncation). "
                    "Revert to the last known-good version below, then apply "
                    "ONE minimal fix.\n\n"
                    + base
                )
            return base

        # Fix C (model-agnostic): when the on-disk file is structurally
        # truncated (open <html>/<body>/<script> without matching close),
        # patches can't anchor — there's nothing to anchor against. Route
        # through a short-form recovery prompt that asks for a fresh
        # rewrite. The existing rewrite-gate already allows the rewrite
        # through (degenerate-baseline carve-out, which we extended to
        # include this truncation case).
        trunc_reason = _truncation_reason(self._current_file)
        if trunc_reason:
            self._trace({
                "kind": "truncation_recovery",
                "reason": trunc_reason,
                "broken_file_bytes": len(self._current_file),
            })
            return self._p.truncation_recovery_instruction(
                report_text=report_text,
                truncation_reason=trunc_reason,
                broken_size_bytes=len(self._current_file),
            )

        # Phase 2: structural-broken recovery for non-truncation shapes —
        # concatenated drafts (duplicate top-level declarations), wrapper
        # preamble before <!DOCTYPE, etc. _is_degenerate_baseline opens
        # the rewrite gate in `_materialize`; route the same recovery
        # prompt so the model knows WHY patches will fail to anchor.
        # Trace 1 (chess 20260522_000304) sat in this state for 4 iters.
        try:
            structural_reason = _baseline_structurally_broken(
                self._current_file
            )
        except Exception:
            structural_reason = None
        if structural_reason and _is_degenerate_baseline(self._current_file):
            self._trace({
                "kind": "structural_recovery",
                "reason": structural_reason[:200],
                "broken_file_bytes": len(self._current_file),
            })
            return self._p.truncation_recovery_instruction(
                report_text=report_text,
                truncation_reason=structural_reason,
                broken_size_bytes=len(self._current_file),
            )

        # Dead-first-build recovery (Wolfenstein 2026-05-24 lesson):
        # iter 1 or 2 loaded a file with raf_ran=false AND input dead.
        # Patching can't fix a fundamentally non-running file. Route to
        # the scope-reduction prompt that asks for a smaller intentional
        # rewrite. The flag is consumed so this only fires once per
        # detection; if the model ships another dead first build the
        # detector will set it again and the attempt-abort counter will
        # eventually flag the restart loop.
        if (
            getattr(self, "_dead_first_build_pending", False)
            and hasattr(self._p, "scope_reduction_instruction")
        ):
            self._dead_first_build_pending = False
            # The scope-reduction prompt ORDERS a complete <html_file>;
            # arm the one-shot exemption so the baseline-exists gate in
            # `_materialize` accepts the compliant rewrite. DeepSeek trace
            # 140129 attempt 2: the model obeyed this exact prompt and its
            # rewrite (containing the PLAYER_X fix) was rejected.
            self._allow_one_rewrite = True
            self._trace({
                "kind": "dead_first_build_recovery_prompt_used",
                # `_build_fix_prompt` runs after the iter's test report
                # lands, so `_last_tested_iter` is the iteration this
                # recovery is responding to. `iteration` is NOT in
                # scope here — earlier draft referenced the loop var
                # by name and crashed with NameError.
                "iteration": self._last_tested_iter,
                "recoveries_this_attempt": (
                    self._dead_first_build_recoveries
                ),
                "rewrite_exemption_armed": True,
            })
            return self._p.scope_reduction_instruction(report_text)

        # Failed: combined diagnose+fix prompt with memory hints inline.
        sig = signature_for_report(report)
        hints_list = self._memory.retrieve_mistakes(sig, k=3) if sig else []
        hints = ""
        if hints_list:
            hints = "\n".join(
                f"- {h.fix_summary}" for h in hints_list
            )

        # ONE combined turn: diagnose → patches → notes. The format anchor
        # goes BEFORE the report so it's the first thing the model sees,
        # which dramatically increases the chance the model actually emits
        # <diagnose>. (Empirical: when the format ask was below the report,
        # gpt-oss skipped it ~100% of the time.)
        # Fix-turn = code stage (narrow): only validated patterns,
        # tighter char budget, no net-harmful bullets.
        blocker_query = self._report_blocker_query(report)
        fix_query = (
            f"{self._goal} {blocker_query}".strip()
            if blocker_query else self._goal
        )
        ensure_ids = self._playbook_ensure_ids_for_report(report) or None
        pb_block = self._retrieve_playbook_block(
            fix_query,
            code=self._current_file,
            stage="code",
            ensure_ids=ensure_ids or None,
        )
        opening_block, opening_hits = self._retrieve_opening_book_block(
            self._goal, stage="code",
        )
        self._active_opening_book_recipes = opening_hits
        fix_kwargs: dict = {}
        if pb_block:
            fix_kwargs["playbook_block"] = pb_block
        # Track stuck-streak so the fix prompt can switch to the
        # 5-7-causes reflection ladder after repeated failures.
        fix_kwargs["stuck_streak"] = self._stuck_streak
        # Feed the model its own Phase-A acceptance criteria so each fix
        # is anchored to "what does the working game owe me?" instead of
        # only "what does the report say is wrong?".
        if self._criteria:
            fix_kwargs["criteria_block"] = self._criteria
        failing_probes = [
            p for p in (report.get("probes") or []) if not p.get("ok")
        ]
        if failing_probes:
            lines = []
            for p in failing_probes[:4]:
                name = str(p.get("name") or "probe")
                lines.append(
                    f"- {name}: executable probe returned falsy — fix this "
                    "behavior before cosmetic warnings."
                )
            fix_kwargs["probe_failure_block"] = "\n".join(lines)
        # Build a focused slice when the file is large; falls back to
        # full-file inject for small files (slice would lose context for
        # marginal gain). The slice protects against context-pollution
        # on long sessions where the file passes 12 KB.
        try:
            slice_text = self._focused_slice(
                self._current_file, report, self._criteria,
            )
        except Exception as e:
            slice_text = None
            self._trace({"kind": "focused_slice_failed", "err": str(e)})
        if slice_text:
            fix_kwargs["focused_slice"] = slice_text
            self._trace({
                "kind": "focused_slice_used",
                "slice_bytes": len(slice_text),
                "full_bytes": len(self._current_file),
            })
        # Context-pressure one-shot: when the prior stream pinned >=85%
        # of num_ctx, omit the CURRENT FILE block from the fix prompt
        # this turn and force a minimal patch. Consumed once then
        # cleared so a transient spike doesn't lock the agent into
        # patch-only mode forever.
        if getattr(self, "_context_pressure_pending", False):
            fix_kwargs["context_pressure"] = True
            self._trace({
                "kind": "context_pressure_mitigation_applied",
                # Same scope bug as dead_first_build above — `iteration`
                # is not local to `_build_fix_prompt`. Use the tracked
                # iteration of the test report this prompt is reacting to.
                "iteration": self._last_tested_iter,
                "streak": self._context_pressure_streak,
            })
            self._context_pressure_pending = False
        fix = self._p.fix_instruction(
            report_text, self._current_file, hints, **fix_kwargs,
        )
        if opening_block:
            fix = (
                f"{opening_block}\n\n"
                "Use only opening-book recipes that directly match this failure; "
                "do not add unrelated scope.\n\n"
                + fix
            )
        traps_block = self._retrieve_outline_traps_block(
            self._goal,
            report_text,
            failure_class=getattr(self, "_last_failure_class", None),
        )
        if traps_block:
            fix = f"{traps_block}\n\n" + fix
        # Capability-round item 1: fix-turn component injection. Query is
        # the BLOCKER text (failed probes / errors), not the goal, so a
        # snippet only appears when it matches the actual failure. k=1.
        if blocker_query:
            components_block = self._retrieve_components_block(
                blocker_query, stage="code", k=1,
            )
            if components_block:
                fix = (
                    f"{components_block}\n\n"
                    "The component above matches this failure — adapt it "
                    "to your existing code via <patch>; do not bolt it on "
                    "as-is.\n\n"
                    + fix
                )
        repeat_fastpath = self._repeat_error_fastpath_block(report)
        if partial_failed:
            fix += "\n\n" + self._partial_patch_recovery_block(partial_failed)
        # Probe-sanity findings: surface tautological-probe or
        # unassigned-property warnings so the model can fix the probes
        # alongside the code. Without this the model often "fixes" the
        # gameplay only to find the probe still false-fails.
        if self._probe_lint_findings:
            fix += "\n\nPROBE LINT — these probes look broken:\n" + "\n".join(
                f"  - {f['message']}" for f in self._probe_lint_findings
            ) + (
                "\nRe-emit `<probes>[...]</probes>` alongside your patch "
                "this turn, rewriting the flagged probes so they actually "
                "test the behavior they claim to test."
            )
        eval_error_count = sum(
            1 for p in (report.get("probes") or [])
            if p.get("kind") == "eval_error" and not p.get("ok")
        )
        if eval_error_count:
            fix += (
                "\n\nPROBES NEED REPAIR: "
                f"{eval_error_count} probe(s) errored at eval time last iter. "
                "You may emit `<probes>[...]</probes>` alongside your patch "
                "this turn to replace them; the harness will adopt the new set."
            )
        if self._is_vlm and self._next_image_bytes:
            fix += "\n\n" + self._p.VLM_REVIEW_NOTE
        if repeat_fastpath:
            fix += "\n\n" + repeat_fastpath

        format_anchor = (
            "REPLY FORMAT FOR THIS TURN — emit these tags IN THIS ORDER:\n"
            "  1. <diagnose>EXACTLY ONE root cause in ≤2 sentences. Name the "
            "function or variable. Required. Do NOT enumerate hypotheses; "
            "do NOT emit a numbered or bulleted list.</diagnose>\n"
            "  2. ONE <patch>...SEARCH/REPLACE...</patch> block against the "
            "current file (or, only if patches truly cannot express the "
            "change AND a baseline does not yet exist OR you are explicitly "
            "permitted, a single <html_file>...</html_file>). Multiple "
            "patches in one reply are allowed only when they target the "
            "same root cause.\n"
            "  3. <notes>one sentence</notes>\n\n"
            "EXAMPLE:\n"
            "<diagnose>The keyup handler is referencing `keys` instead of "
            "`keys[k]`, so held keys never clear.</diagnose>\n"
            "<patch>\n"
            "<<<<<<< SEARCH\n"
            'addEventListener("keyup", e => { const k = KEYMAP[e.code]; if (k) keys = false; });\n'
            "=======\n"
            'addEventListener("keyup", e => { const k = KEYMAP[e.code]; if (k) keys[k] = false; });\n'
            ">>>>>>> REPLACE\n"
            "</patch>\n"
            "<notes>Fixed broken keyup so movement keys release.</notes>\n\n"
        )
        return format_anchor + fix



