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
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import ollama

from memory import GameMemory, signature_for_report
from ollama_io import Candidate, StreamResult, best_of_n as _best_of_n
from ollama_io import stream_chat, stream_chat_with_retry
from patches import apply_patches, extract_patches
from prompts import (
    CRITIQUE_INSTRUCTION,
    PLAN_INSTRUCTION,
    SYSTEM_PROMPT,
    VLM_REVIEW_NOTE,
    diagnose_instruction,
    first_build_instruction,
    fix_instruction,
    patch_retry_instruction,
    post_clean_instruction,
    regression_instruction,
    seed_build_instruction,
)
from tools import LiveBrowser, format_report_for_model


_HTML_RE = re.compile(r"<html_file>\s*(.*?)\s*</html_file>", re.DOTALL | re.IGNORECASE)
_DONE_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)
_CONFIRM_RE = re.compile(r"<confirm[_-]?done\s*/?>", re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)
_DIAGNOSE_RE = re.compile(r"<diagnose>\s*(.*?)\s*</diagnose>", re.DOTALL | re.IGNORECASE)
_NOTES_RE = re.compile(r"<notes>\s*(.*?)\s*</notes>", re.DOTALL | re.IGNORECASE)


# Once the messages list has more turns than this, older turns get their
# embedded code (<html_file> bodies, ```html fences) replaced with summaries
# so context stays bounded. Tunable; the agent always passes the CURRENT
# file inline in the fix prompt, so old code in history is just bloat.
_PRUNE_KEEP_RECENT_TURNS = 4


@dataclass
class AgentEvent:
    kind: str           # phase | token | plan | code | test | question | done | error | info | diagnose | patch | best_of_n | memory
    text: str = ""
    data: dict = field(default_factory=dict)


class GameAgent:
    """Drives the planning/coding/critique loop. One instance per session."""

    def __init__(
        self,
        model: str,
        out_path: Path,
        browser: LiveBrowser,
        max_iters: int = 6,
        *,
        best_of_n: int = 1,
        # 8192 matches Ollama's default load size for gpt-oss. Sending a
        # different value forces a model reload every request and was the
        # root cause of "Ollama call failed" errors after several runs.
        # Conversation pruning (_prune_messages) keeps us under this even
        # across many iterations.
        num_ctx: int = 8192,
        # 90s per-chunk inactivity. With 16K ctx + 20B model, time-to-first
        # token can be 20-40s on a fresh load; we want headroom but still
        # detect a true wedge promptly.
        stall_seconds: float = 90.0,
        memory_root: str | Path = "games/memory",
        # Optional path to an existing HTML file to start from. When set,
        # the agent skips memory-skeleton retrieval and uses this file as
        # the baseline; the model is asked to ADAPT it (via patches) to
        # the user's goal rather than build from scratch.
        seed_file: str | Path | None = None,
    ):
        self.model = model
        self.out_path = Path(out_path)
        self.browser = browser
        self.max_iters = max_iters
        self.best_of_n = max(1, best_of_n)
        self.num_ctx = num_ctx
        self.stall_seconds = stall_seconds
        self.seed_file: Path | None = Path(seed_file) if seed_file else None
        self._client = ollama.AsyncClient()
        self._messages: list[dict] = []
        self._pending_feedback: list[str] = []
        self._pending_answer: str | None = None
        self._user_force_done = False

        # All per-session artifact paths share the out_path stem so a session
        # named e.g. "asteroids_20260503_175727.html" produces matching
        # asteroids_20260503_175727.{jsonl,log,best.html,conversation.md} and
        # snapshots/asteroids_20260503_175727/. The driver (chat.py / coder.py)
        # is responsible for making the stem unique + meaningful — usually
        # "<goal-slug>_<timestamp>".
        basename = self.out_path.stem or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = basename
        out_dir = self.out_path.parent
        self.trace_path: Path = out_dir / "traces" / f"{basename}.jsonl"
        self.snapshots_dir: Path = out_dir / "snapshots" / basename
        self.best_path: Path = out_dir / f"{basename}.best.html"
        self.conversation_path: Path = out_dir / "traces" / f"{basename}.conversation.md"
        self._previous_report_ok: bool | None = None
        self._snapshot_n: int = 0
        self._is_vlm: bool | None = None
        self._next_image_bytes: bytes | None = None
        self._fix_mode: bool = False
        self._memory = GameMemory(root=memory_root)
        self._memory.ensure()
        self._last_diagnose: str | None = None
        self._token_cb = None
        self._goal: str = ""
        # Tracks the most recent test-report summary for memory.record_outcome.
        self._last_report_summary: str = ""
        self._last_iter_run: int = 0
        # Tracks the most recent file content actually written to disk. We
        # always inline THIS in fix prompts (instead of asking the model to
        # remember its own previous reply).
        self._current_file: str = ""

    # -- TUI-facing setters -------------------------------------------------

    def add_user_feedback(self, text: str) -> None:
        text = text.strip()
        if text:
            self._pending_feedback.append(text)
            self._trace({"kind": "feedback_queued", "text": text})

    def add_user_answer(self, text: str) -> None:
        self._pending_answer = text.strip()
        self._trace({"kind": "answer_queued", "text": self._pending_answer})

    def has_pending_user_input(self) -> bool:
        return bool(self._pending_feedback) or self._pending_answer is not None

    def request_done(self) -> None:
        self._user_force_done = True

    def set_token_callback(self, cb) -> None:
        self._token_cb = cb

    def _token_cb_wrapper(self, piece: str) -> None:
        if self._token_cb is not None:
            try:
                self._token_cb(piece)
            except Exception:
                pass

    # -- trace / snapshot helpers ------------------------------------------

    def _trace(self, obj: dict) -> None:
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": datetime.utcnow().isoformat() + "Z", **obj}
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _record(self, ev: AgentEvent) -> AgentEvent:
        text = ev.text or ""
        self._trace({
            "kind": "event",
            "event": ev.kind,
            "text_preview": text[:1000],
            "text_len": len(text),
            "data": ev.data,
        })
        return ev

    def _save_snapshot(self, html: str) -> Path | None:
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_n += 1
            p = self.snapshots_dir / f"iter_{self._snapshot_n:02d}.html"
            p.write_text(html, encoding="utf-8")
            return p
        except Exception:
            return None

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

    def _dump_conversation(self) -> None:
        try:
            self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = [
                f"# Conversation dump — {self.model}",
                f"_session: {self._session_id}_  ",
                f"_iteration count: {self._snapshot_n}_  ",
                f"_messages: {len(self._messages)}_",
                "",
            ]
            for i, msg in enumerate(self._messages):
                role = msg.get("role", "?")
                content = msg.get("content", "") or ""
                lines.append(f"## [{i:02d}] {role}")
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

    def _summarize_content(self, c: str) -> str:
        """Replace embedded HTML blobs with size markers — keep tags + notes."""
        def html_repl(m):
            n = len(m.group(1))
            return f"<html_file>[omitted: {n} bytes of HTML; see snapshot]</html_file>"

        def fence_repl(m):
            n = len(m.group(1))
            return f"```html\n[omitted: {n} bytes of HTML; see snapshot]\n```"

        c = self._SUMMARIZE_HTML_RE.sub(html_repl, c)
        c = self._SUMMARIZE_FENCE_RE.sub(fence_repl, c)
        return c

    def _prune_messages(self) -> None:
        """Compress code-heavy old turns so context stays bounded.

        We always keep the system prompt (index 0) and the last
        _PRUNE_KEEP_RECENT_TURNS messages intact. Everything between gets
        its inline HTML replaced with `[omitted: N bytes]`.
        """
        if len(self._messages) <= 1 + _PRUNE_KEEP_RECENT_TURNS:
            return
        cutoff = len(self._messages) - _PRUNE_KEEP_RECENT_TURNS
        for i in range(1, cutoff):
            msg = self._messages[i]
            c = msg.get("content", "") or ""
            new_c = self._summarize_content(c)
            if new_c != c:
                msg["content"] = new_c

    # -- user-injection plumbing -------------------------------------------

    def _consumed_feedback_summary(self) -> str | None:
        bits: list[str] = []
        if self._pending_answer is not None:
            ans = self._pending_answer
            bits.append(f"answer: {ans[:80]!r}")
        if self._pending_feedback:
            for fb in self._pending_feedback:
                bits.append(f"feedback: {fb[:80]!r}")
        if not bits:
            return None
        return "→ applying your input to next turn: " + "; ".join(bits)

    def _flush_user_injections(self, base_message: str) -> str:
        parts: list[str] = []
        if self._pending_answer is not None:
            ans = self._pending_answer
            parts.append(
                "================ USER ANSWER (HIGHEST PRIORITY) ================\n"
                f"{ans}\n"
                "================================================================"
            )
            self._trace({"kind": "answer_injected", "text": ans})
            self._pending_answer = None
        if self._pending_feedback:
            joined = "\n- ".join(self._pending_feedback)
            parts.append(
                "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                "The user just typed this while watching your game. It OVERRIDES\n"
                "any plan or default behavior. Address it explicitly in this turn:\n"
                f"\n- {joined}\n"
                "=================================================================="
            )
            for fb in self._pending_feedback:
                self._trace({"kind": "feedback_injected", "text": fb})
            self._pending_feedback.clear()
        if base_message:
            parts.append(base_message)
        return "\n\n".join(parts)

    # -- streaming ----------------------------------------------------------

    async def _detect_vlm(self) -> bool:
        try:
            info = await self._client.show(model=self.model)
        except Exception:
            return False
        caps = getattr(info, "capabilities", None) or []
        return any(str(c).lower() == "vision" for c in caps)

    async def _stream(self, on_token, *, override_temp: float | None = None) -> str:
        """Stream once, with watchdog. Recovers from stalls by raising/logging.

        Image attachment: if VLM is detected and self._next_image_bytes is set,
        attach to the LAST user message and clear the buffer.
        """
        if self._is_vlm is None:
            self._is_vlm = await self._detect_vlm()
            if self._is_vlm:
                self._trace({"kind": "vlm_detected", "model": self.model})

        if (
            self._is_vlm
            and self._next_image_bytes
            and self._messages
            and self._messages[-1].get("role") == "user"
        ):
            self._messages[-1]["images"] = [self._next_image_bytes]
            self._trace({"kind": "image_attached", "bytes": len(self._next_image_bytes)})
            self._next_image_bytes = None

        temp = override_temp if override_temp is not None else (
            0.25 if self._fix_mode else 0.7
        )
        self._trace({"kind": "stream_start", "temperature": temp, "fix_mode": self._fix_mode})

        result = await stream_chat_with_retry(
            self._client,
            self.model,
            self._messages,
            on_token=on_token,
            options={"temperature": temp, "num_ctx": self.num_ctx},
            stall_seconds=self.stall_seconds,
            overall_seconds=600.0,
            max_retries=1,
            on_stall=lambda r, attempt: self._trace({
                "kind": "stream_stalled",
                "attempt": attempt,
                "tokens_before_stall": r.tokens,
                "duration_s": r.duration_s,
            }),
        )
        self._trace({
            "kind": "stream_done",
            "tokens": result.tokens,
            "duration_s": round(result.duration_s, 2),
            "stalled": result.stalled,
            "len": len(result.text),
        })
        if result.stalled and not result.text.strip():
            raise RuntimeError(
                f"Model produced no tokens before stalling at "
                f"{self.stall_seconds}s. Try a smaller context (num_ctx={self.num_ctx}) "
                "or different model."
            )
        return result.text

    # -- best-of-N for fix iterations --------------------------------------

    async def _generate_and_score_candidates(
        self,
        n: int,
    ) -> tuple[Candidate, list[Candidate]]:
        """Sample N completions and score each by running its result through
        the test harness against a temp file. Used when fixing a failed iter.

        Scorer:
          +1.0 if patches/file produced and test passed
          +0.5 if patches/file applied cleanly but test still failed
          0.0  if patches/file extracted but couldn't be applied
          -1.0 if no usable code came back at all
        """
        # We deliberately DO NOT stream tokens for these candidates — only
        # the winning one is replayed visually after we pick it.
        async def scorer(text: str) -> tuple[float, dict]:
            extra: dict = {"kind": "candidate", "text_len": len(text)}
            html, applied_msg = await self._materialize(text, dry_run=True)
            extra["materialized"] = bool(html)
            extra["materialize_msg"] = applied_msg
            if not html:
                return -1.0, extra
            # Score by running headless via the LiveBrowser. The browser is
            # currently showing the LAST file we wrote; we'll switch back to
            # it after scoring. For scoring we write to a side-by-side temp.
            tmp_path = self.snapshots_dir / f"cand_{self._snapshot_n+1:02d}_{abs(hash(text))%10000:04d}.html"
            try:
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(html, encoding="utf-8")
                report = await self.browser.load_and_test(tmp_path, screenshot_path=None)
                extra["report_ok"] = report.get("ok", False)
                extra["report_summary"] = format_report_for_model(report)[:400]
                return (1.0 if report["ok"] else 0.5), extra
            except Exception as e:
                extra["scorer_exception"] = str(e)
                return 0.5, extra

        winner, all_cands = await _best_of_n(
            self._client,
            self.model,
            self._messages,
            n=n,
            options={"num_ctx": self.num_ctx},
            stall_seconds=self.stall_seconds,
            overall_seconds=600.0,
            scorer=scorer,
            on_progress=lambda i, msg: self._trace({
                "kind": "best_of_n_progress",
                "candidate": i,
                "msg": msg,
            }),
        )
        return winner, all_cands

    # -- text → file: extract patches OR <html_file> -----------------------

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
            if res.applied == 0:
                return None, f"all {len(patches)} patches failed to apply"
            if res.failed and not dry_run:
                # Partial-apply: still write what landed, but the caller
                # gets a non-empty failed list to retry on.
                pass
            return res.text, f"applied {res.applied}/{len(patches)} patches"

        html = self._extract_html(reply)
        if html is not None:
            return html, "full <html_file> rewrite"

        return None, "no <patch> or <html_file> in reply"

    @staticmethod
    def _extract_html(reply: str) -> str | None:
        m = _HTML_RE.search(reply)
        if not m:
            return None
        body = m.group(1).strip()
        if body.startswith("```"):
            body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
            body = re.sub(r"\n?```$", "", body)
        return body.strip() or None

    @staticmethod
    def _extract_question(reply: str) -> str | None:
        m = _QUESTION_RE.search(reply)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_diagnose(reply: str) -> str | None:
        m = _DIAGNOSE_RE.search(reply)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_notes(reply: str) -> str | None:
        m = _NOTES_RE.search(reply)
        return m.group(1).strip() if m else None

    # -- main loop ----------------------------------------------------------

    async def run(
        self,
        goal: str,
        *,
        continuation: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Drive a planning + iteration session.

        continuation=False (default): fresh session. Resets _messages, runs
        phase A planning, picks a memory skeleton, and seeds the first build.

        continuation=True: extend a previously-finished session. `goal` is
        treated as new user feedback for the existing game on disk; planning
        and first-build are skipped, the iteration loop resumes immediately
        with a continuation prompt. _messages, _current_file, _snapshot_n,
        browser, and model are all reused.
        """
        self._goal = goal
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
            # Reset transient state so the loop runs again fresh.
            self._user_force_done = False
            self._fix_mode = True
            self._previous_report_ok = True
            # Queue the new request as user feedback so _flush_user_injections
            # folds it into the next prompt with the standard "USER FEEDBACK"
            # banner the model already knows how to react to.
            self.add_user_feedback(goal)
            self._trace({"kind": "continuation_start", "feedback": goal})
            yield self._record(AgentEvent(
                "info", f"continuing on existing file with new request: {goal[:160]}"
            ))
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(
                    "CONTINUATION TURN: the user has new feedback above for the "
                    "game you previously shipped. The current file is unchanged "
                    "on disk. Reply with one or more <patch> blocks that address "
                    "the feedback. Use a full <html_file> only if patches truly "
                    "cannot express the change."
                ),
            })
        else:
            self._messages = [
                {"role": "system", "content": SYSTEM_PROMPT.replace("{goal}", goal)},
                {"role": "user", "content": PLAN_INSTRUCTION},
            ]

            self._trace({
                "kind": "session_start",
                "model": self.model,
                "goal": goal,
                "out_path": str(self.out_path),
                "trace_path": str(self.trace_path),
                "snapshots_dir": str(self.snapshots_dir),
                "best_path": str(self.best_path),
                "max_iters": self.max_iters,
                "best_of_n": self.best_of_n,
                "num_ctx": self.num_ctx,
                "stall_seconds": self.stall_seconds,
                "memory_root": str(self._memory.root),
            })
            yield self._record(AgentEvent(
                "info",
                f"Trace: {self.trace_path}  (snapshots: {self.snapshots_dir})",
            ))

            # ---- PHASE A: planning ------------------------------------------
            yield self._record(AgentEvent("phase", "planning"))
            try:
                plan_reply = await self._stream(self._token_cb_wrapper)
            except Exception as e:
                yield self._record(AgentEvent("error", f"Ollama call failed during planning: {e}"))
                return
            self._messages.append({"role": "assistant", "content": plan_reply})
            self._dump_conversation()
            yield self._record(AgentEvent("plan", plan_reply))

            q = self._extract_question(plan_reply)
            if q is not None:
                yield self._record(AgentEvent("question", q))
                while self._pending_answer is None:
                    await asyncio.sleep(0.1)
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        "Thanks. Now produce the <plan> per the original instructions."
                    ),
                })
                try:
                    plan_reply = await self._stream(self._token_cb_wrapper)
                except Exception as e:
                    yield self._record(AgentEvent("error", f"Ollama call failed: {e}"))
                    return
                self._messages.append({"role": "assistant", "content": plan_reply})
                self._dump_conversation()
                yield self._record(AgentEvent("plan", plan_reply))

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
                yield self._record(AgentEvent("memory", (
                    f"using user-provided seed file: {self.seed_file} "
                    f"({len(seed_html)} bytes) — memory skeleton skipped"
                ), {
                    "seed_file": str(self.seed_file),
                    "seed_bytes": len(seed_html),
                }))
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        seed_build_instruction(seed_html, str(self.seed_file))
                    ),
                })
            else:
                skel = self._memory.retrieve_skeleton(goal)
                memory_msg = (
                    f"using skeleton: {skel.name}"
                    + (f" (sim={skel.score:.2f}, src goal: {skel.source_goal!r})"
                       if skel.source_goal else " (default)")
                )
                yield self._record(AgentEvent("memory", memory_msg, {
                    "skeleton": skel.name, "score": skel.score, "source_goal": skel.source_goal,
                }))

                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        first_build_instruction(skel.html, skel.source_goal)
                    ),
                })

        # ---- PHASE B: build/iterate -------------------------------------
        awaiting_confirm = False

        # In continuation mode the iteration counter resumes from where the
        # previous run left off; the user-visible label uses a relative count
        # so they see "extension 1/N" instead of confusing "iteration 4/9".
        start_iter = self._snapshot_n + 1 if continuation else 1
        end_iter = start_iter + self.max_iters - 1

        for iteration in range(start_iter, end_iter + 1):
            self._last_iter_run = iteration
            if continuation:
                rel = iteration - start_iter + 1
                phase_text = f"extension {rel}/{self.max_iters}"
            else:
                phase_text = f"iteration {iteration}/{self.max_iters}"
            yield self._record(AgentEvent("phase", phase_text))

            # Prune older turns before generating, so we don't hit context.
            self._prune_messages()

            # Best-of-N is only used in fix mode (iter 2+ after a failure)
            # AND only when we have N>1. The first build is always single
            # because there's no test signal yet to score against.
            use_bon = self._fix_mode and self.best_of_n > 1
            try:
                if use_bon:
                    yield self._record(AgentEvent(
                        "best_of_n",
                        f"sampling {self.best_of_n} candidates",
                        {"n": self.best_of_n},
                    ))
                    winner, all_cands = await self._generate_and_score_candidates(self.best_of_n)
                    # Replay the winner visually for the user — feel as if
                    # the model just wrote it now, even though it generated
                    # silently.
                    for piece in self._chunk_for_display(winner.text):
                        self._token_cb_wrapper(piece)
                    reply = winner.text
                    yield self._record(AgentEvent(
                        "best_of_n",
                        f"picked candidate score={winner.score:+.2f} from {len(all_cands)}",
                        {
                            "winner_score": winner.score,
                            "all_scores": [c.score for c in all_cands],
                            "winner_extra": winner.extra,
                        },
                    ))
                else:
                    reply = await self._stream(self._token_cb_wrapper)
            except Exception as e:
                yield self._record(AgentEvent("error", f"Ollama call failed: {e}"))
                return

            self._messages.append({"role": "assistant", "content": reply})
            self._dump_conversation()
            self._trace({
                "kind": "assistant_reply",
                "iteration": iteration,
                "len": len(reply),
                "preview": reply[:600],
            })

            # ---- diagnose extraction (logged + memory-keyed) -----------
            diag = self._extract_diagnose(reply)
            if diag:
                self._last_diagnose = diag
                yield self._record(AgentEvent("diagnose", diag))

            notes = self._extract_notes(reply)
            if notes:
                yield self._record(AgentEvent("info", f"notes: {notes[:200]}"))

            # ---- handle <question> from the model ----------------------
            q = self._extract_question(reply)
            html_in_reply = self._extract_html(reply)
            patches_in_reply = extract_patches(reply)
            if q is not None and html_in_reply is None and not patches_in_reply:
                yield self._record(AgentEvent("question", q))
                while self._pending_answer is None:
                    await asyncio.sleep(0.1)
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        "Thanks. Continue building the game."
                    ),
                })
                continue

            # ---- handle done / confirm-done with no new code ----------
            # Two paths land here:
            #   (a) we asked the critique question (awaiting_confirm) and the
            #       model replied <confirm_done/>;
            #   (b) the previous iter passed cleanly and we sent the post-
            #       clean prompt encouraging <done/> — model replied <done/>
            #       (or <confirm_done/>) with no new code.
            # Either way: nothing to apply, nothing to test, ship it.
            said_done_or_confirm = bool(
                _CONFIRM_RE.search(reply) or _DONE_RE.search(reply)
            )
            if (
                said_done_or_confirm
                and html_in_reply is None
                and not patches_in_reply
                and (awaiting_confirm or self._previous_report_ok is True)
            ):
                if self.has_pending_user_input():
                    yield self._record(AgentEvent(
                        "info",
                        "Model said done but user feedback is pending - applying it instead of exiting.",
                    ))
                    awaiting_confirm = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "You were about to ship, but the user just sent the "
                            "feedback above. Address it now and re-send a fix as <patch>."
                        ),
                    })
                    continue
                reason = (
                    "Model confirmed after self-critique."
                    if awaiting_confirm
                    else "Model declared done after a clean run."
                )
                yield self._record(AgentEvent("done", reason))
                self._record_session_outcome(ok=True)
                return

            # ---- materialize: patches OR full file --------------------
            new_html, materialize_msg = await self._materialize(reply)
            if new_html is None:
                yield self._record(AgentEvent("info", f"no usable code: {materialize_msg}"))
                # If the reply had patches but they failed to apply, give the
                # model the specific failures + current file so it can retry.
                if patches_in_reply and self._current_file:
                    res = apply_patches(self._current_file, patches_in_reply)
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            patch_retry_instruction(res.failed, self._current_file)
                        ),
                    })
                else:
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "I could not find a <patch> or <html_file> block in your "
                            "reply. If this is the first build, send a complete "
                            "<html_file>. Otherwise send <patch> blocks."
                        ),
                    })
                continue

            # Track partial-patch failures for the next prompt even on
            # successful materialize.
            partial_failed: list[tuple[int, object, str]] = []
            if patches_in_reply and self._current_file:
                res = apply_patches(self._current_file, patches_in_reply)
                partial_failed = res.failed

            # Save + per-iter snapshot
            self.out_path.write_text(new_html, encoding="utf-8")
            self._current_file = new_html
            snap_path = self._save_snapshot(new_html)
            shot_path = snap_path.with_suffix(".png") if snap_path else None
            yield self._record(AgentEvent(
                "code",
                str(self.out_path),
                {
                    "size": len(new_html),
                    "snapshot": str(snap_path) if snap_path else None,
                    "screenshot": str(shot_path) if shot_path else None,
                    "materialize": materialize_msg,
                    "patches_applied": len(patches_in_reply) - len(partial_failed),
                    "patches_failed": len(partial_failed),
                },
            ))

            # ---- run the test ------------------------------------------
            try:
                report = await self.browser.load_and_test(
                    self.out_path, screenshot_path=shot_path,
                )
            except Exception as e:
                yield self._record(AgentEvent("info", f"browser harness crashed: {e}"))
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        f"The browser test harness itself crashed: {e}\n"
                        "Please simplify the page and try again."
                    ),
                })
                continue

            report_text = format_report_for_model(report)
            self._last_report_summary = report_text
            yield self._record(AgentEvent("test", report_text, report))

            # Queue screenshot bytes for VLM attachment on next turn.
            if shot_path is not None and report.get("screenshot"):
                if self._is_vlm is not False:
                    try:
                        self._next_image_bytes = Path(report["screenshot"]).read_bytes()
                    except Exception:
                        self._next_image_bytes = None

            said_done = bool(_DONE_RE.search(reply))
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Save best.html on every clean iteration AND record the success
            # in memory so future similar goals can retrieve this code.
            if report["ok"]:
                best = self._save_best(new_html)
                if best is not None:
                    yield self._record(AgentEvent(
                        "info", f"saved working version to {best}"
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
            if self._user_force_done and report["ok"]:
                if self.has_pending_user_input():
                    yield self._record(AgentEvent(
                        "info",
                        "Ship requested but new user feedback arrived - applying it before shipping.",
                    ))
                    self._user_force_done = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "The user wanted to ship but then sent the feedback above. "
                            "Address that feedback. Send a <patch>."
                        ),
                    })
                    continue
                yield self._record(AgentEvent("done", "User confirmed - shipping current build."))
                self._record_session_outcome(ok=True)
                return

            if self._user_force_done and not report["ok"]:
                # Build a normal fix prompt, but tell the model to ship now.
                next_user = self._build_fix_prompt(
                    report=report, regressed=regressed, partial_failed=partial_failed,
                ) + (
                    "\n\nNOTE: the user wants to SHIP NOW. Fix ONLY the errors above; "
                    "do not add features or polish."
                )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(next_user),
                })
                self._previous_report_ok = report["ok"]
                self._fix_mode = True
                continue

            # ---- self-critique on first clean+done ---------------------
            if report["ok"] and said_done and not awaiting_confirm:
                yield self._record(AgentEvent("phase", "self-critique"))
                awaiting_confirm = True
                notice = self._consumed_feedback_summary()
                if notice:
                    yield self._record(AgentEvent("info", notice))
                # Critique always uses single-sample (we want the model's
                # honest call, not a vote).
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(CRITIQUE_INSTRUCTION),
                })
                self._previous_report_ok = report["ok"]
                self._fix_mode = False
                continue

            if awaiting_confirm:
                awaiting_confirm = False

            # ---- record mistake on regression --------------------------
            # We can't tell what the right fix is yet, so we record the
            # signature only — the next clean turn will pair it with the
            # diagnosis that fixed it (above).
            if not report["ok"]:
                sig = signature_for_report(report)
                if sig:
                    self._trace({"kind": "mistake_signature", "sig": sig})

            # ---- build next user turn ---------------------------------
            notice = self._consumed_feedback_summary()
            if notice:
                yield self._record(AgentEvent("info", notice))

            next_user = self._build_fix_prompt(
                report=report, regressed=regressed, partial_failed=partial_failed,
            )
            # Adaptive temperature: failed → low (precision). Clean+keep-going
            # path goes through post_clean which says "prefer done".
            self._fix_mode = not report["ok"]
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(next_user),
            })
            self._previous_report_ok = report["ok"]

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
            try:
                reply = await self._stream(self._token_cb_wrapper)
                self._messages.append({"role": "assistant", "content": reply})
                self._dump_conversation()
                self._trace({
                    "kind": "assistant_reply",
                    "iteration": self.max_iters + 1,
                    "len": len(reply),
                })
                new_html, materialize_msg = await self._materialize(reply)
                if new_html is not None:
                    self.out_path.write_text(new_html, encoding="utf-8")
                    self._current_file = new_html
                    self._save_snapshot(new_html)
                    yield self._record(AgentEvent(
                        "code", str(self.out_path),
                        {"size": len(new_html), "materialize": materialize_msg},
                    ))
            except Exception as e:
                yield self._record(AgentEvent("error", f"Final feedback turn failed: {e}"))

        yield self._record(AgentEvent(
            "info", f"reached max iterations ({self.max_iters}) - stopping"
        ))
        # Outcome: ok if best.html exists (we passed at least once).
        self._record_session_outcome(ok=self.best_path.exists())
        yield self._record(AgentEvent("done", "Iteration cap reached."))

    # -- helpers ----------------------------------------------------------

    def _build_fix_prompt(
        self,
        *,
        report: dict,
        regressed: bool,
        partial_failed: list[tuple[int, object, str]],
    ) -> str:
        """Construct the next user message after a test result.

        Branches:
          - report ok  → post_clean (encourage <done/>)
          - regressed  → revert prompt with last-good code inline
          - failed     → diagnose-then-fix combined turn (with mistake hints
                         and current file inline; VLM note appended if
                         applicable)
        """
        report_text = format_report_for_model(report)

        if report["ok"]:
            return post_clean_instruction(report_text)

        if regressed:
            best = self._read_best_or_empty()
            return regression_instruction(report_text, best)

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
        fix = fix_instruction(report_text, self._current_file, hints)
        if partial_failed:
            fix += (
                "\n\nNOTE: some of your previous patches did not apply. "
                "When fixing this turn, also re-send corrected versions of:\n"
                + "\n".join(f"  - {reason}" for (_i, _p, reason) in partial_failed)
            )
        if self._is_vlm and self._next_image_bytes:
            fix += "\n\n" + VLM_REVIEW_NOTE

        format_anchor = (
            "REPLY FORMAT FOR THIS TURN — emit these tags IN THIS ORDER:\n"
            "  1. <diagnose>...root cause in ≤2 sentences. Name the function or "
            "variable. Required.</diagnose>\n"
            "  2. one or more <patch>...SEARCH/REPLACE...</patch> blocks against "
            "the current file (or, only if patches truly cannot express the "
            "change, a single <html_file>...</html_file>).\n"
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
            self._trace({
                "kind": "session_outcome",
                "ok": ok,
                "iterations": self._last_iter_run,
                "best_path_exists": self.best_path.exists(),
            })
        except Exception as e:
            self._trace({"kind": "outcome_record_failed", "err": str(e)})

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
