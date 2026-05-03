"""Async agent loop, refactored out of coder.py so the Textual TUI can drive it.

Design:
    - GameAgent owns the message history, iteration count, and the LiveBrowser.
    - It exposes ONE async generator: `run(goal)` which yields AgentEvent
      objects as the loop progresses (planning, streaming tokens, test results,
      questions, done, error, ...).
    - The TUI consumes the generator and pushes events into widgets.
    - Two methods let the TUI inject signals at any time:
        add_user_feedback(text)  - queued, prepended to the next user turn
        add_user_answer(text)    - same, but tagged as USER ANSWER (response
                                   to the model's <question>)
        request_done()           - tells the agent the human is satisfied; on
                                   the next clean run the agent will exit
                                   without entering the self-critique pass.

The CLI in coder.py is unchanged - it still uses the sync chat_stream there.
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

from prompts import CRITIQUE_INSTRUCTION, PLAN_INSTRUCTION, SYSTEM_PROMPT
from tools import LiveBrowser, format_report_for_model


# Same regexes as coder.py - kept duplicated (not imported) so this module
# stands on its own and the CLI doesn't accidentally couple to it.
_HTML_RE = re.compile(r"<html_file>\s*(.*?)\s*</html_file>", re.DOTALL | re.IGNORECASE)
_DONE_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)
_CONFIRM_RE = re.compile(r"<confirm[_-]?done\s*/?>", re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Events emitted by the agent. The TUI matches on .kind and updates widgets.
# Kept as a single dataclass with a `kind` discriminator (rather than a class
# hierarchy) so handlers can be a simple if/elif - small TUI, small dispatch.
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    kind: str           # one of: phase, token, plan, code, test, question, done, error, info
    text: str = ""      # human-readable payload (used by most kinds)
    data: dict = field(default_factory=dict)  # structured payload (e.g. test report)


class GameAgent:
    """Drives the planning/coding/critique loop. One instance per session."""

    def __init__(
        self,
        model: str,
        out_path: Path,
        browser: LiveBrowser,
        max_iters: int = 6,
    ):
        self.model = model
        self.out_path = Path(out_path)
        self.browser = browser
        self.max_iters = max_iters
        self._client = ollama.AsyncClient()
        # Conversation history. Re-seeded by run() for each fresh goal.
        self._messages: list[dict] = []
        # Pending injections from the user. Flushed into the next user turn.
        self._pending_feedback: list[str] = []
        self._pending_answer: str | None = None
        # Set by the TUI when the human says "ship it". On the next clean run
        # we skip the self-critique pass and exit.
        self._user_force_done = False

        # --- audit / regression-protection paths ---
        # All sit next to out_path so one game folder stays self-contained.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.out_path.parent
        self.trace_path: Path = out_dir / "traces" / f"{ts}.jsonl"
        self.snapshots_dir: Path = out_dir / "snapshots" / ts
        self.best_path: Path = out_dir / "best.html"
        # _previous_report_ok lets us detect a working-then-broken regression
        # in the very next iteration so we can tell the model to revert.
        self._previous_report_ok: bool | None = None
        # Iteration counter (separate from the for-loop var so snapshot files
        # increase even when an iteration restarts on a "no <html_file>" nudge).
        self._snapshot_n: int = 0
        # Vision-capable model? Filled lazily the first time we need to know.
        # None = unknown, True/False = decided.
        self._is_vlm: bool | None = None
        # If VLM, bytes of the most recent screenshot get attached to the next
        # user turn (and then cleared). Avoids ballooning context with every
        # past frame.
        self._next_image_bytes: bytes | None = None
        # Adaptive temperature (item C): True when the next generation is a
        # FIX (precision wanted), False on first attempt or after a clean run
        # (creativity wanted). _stream() reads this.
        self._fix_mode: bool = False
        # Conversation dump path (item D): full untruncated message log lives
        # here, refreshed after every assistant turn.
        self.conversation_path: Path = out_dir / "traces" / f"{ts}.conversation.md"

    # --- TUI-facing setters ------------------------------------------------

    def add_user_feedback(self, text: str) -> None:
        """Queue free-form feedback to inject at the next user turn."""
        text = text.strip()
        if text:
            self._pending_feedback.append(text)
            # Trace queueing immediately - we use this later to verify the
            # feedback actually made it into a message (vs being dropped).
            self._trace({"kind": "feedback_queued", "text": text})

    def add_user_answer(self, text: str) -> None:
        """Provide the answer to the model's most recent <question>."""
        self._pending_answer = text.strip()
        self._trace({"kind": "answer_queued", "text": self._pending_answer})

    def has_pending_user_input(self) -> bool:
        """True if there is queued feedback OR an unanswered question reply.

        Used by exit paths so we never silently drop the user's words on
        confirm-done / force-done / max-iters boundaries.
        """
        return bool(self._pending_feedback) or self._pending_answer is not None

    def request_done(self) -> None:
        """User says they're happy. Agent exits at the next safe boundary."""
        self._user_force_done = True

    # --- trace / snapshot helpers ----------------------------------------
    # All file I/O is best-effort: tracing must NEVER crash the agent loop.

    def _trace(self, obj: dict) -> None:
        """Append one JSON line to the session trace file."""
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": datetime.utcnow().isoformat() + "Z", **obj}
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _record(self, ev: AgentEvent) -> AgentEvent:
        """Wrap a yield: record into trace and return the event unchanged.

        Usage: `yield self._record(AgentEvent(...))`. Keeps yield sites tidy.
        """
        # Truncate long text fields so the trace stays readable - full content
        # already lives in messages history / snapshots.
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
        """Write the current iteration's HTML to snapshots/iter_NN.html."""
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_n += 1
            p = self.snapshots_dir / f"iter_{self._snapshot_n:02d}.html"
            p.write_text(html, encoding="utf-8")
            return p
        except Exception:
            return None

    def _save_best(self, html: str) -> Path | None:
        """Mirror the latest CLEAN-test HTML to best.html."""
        try:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.best_path.write_text(html, encoding="utf-8")
            return self.best_path
        except Exception:
            return None

    def _best_code_block_or_empty(self) -> str:
        """Return the last known-good HTML wrapped in a fenced block for use
        in fix/regression prompts (item B). Empty string if no clean version
        has been saved yet, so the caller can just `f"{block}"` unconditionally.
        """
        try:
            if not self.best_path.exists():
                return ""
            html = self.best_path.read_text(encoding="utf-8")
            if not html.strip():
                return ""
            # Cap at a sane size: a 100KB game would blow context. 60KB is
            # plenty for a single-file game and keeps tokens reasonable.
            MAX = 60_000
            if len(html) > MAX:
                html = html[:MAX] + "\n<!-- ...truncated for prompt budget... -->"
            return (
                "LAST KNOWN-GOOD VERSION (this passed all tests). Patch from THIS "
                "baseline; do not rewrite from scratch:\n"
                "```html\n" + html + "\n```\n\n"
            )
        except Exception:
            return ""

    def _dump_conversation(self) -> None:
        """Item D: write the full message history to <ts>.conversation.md.
        Best-effort, never raises. Called after every assistant turn so a
        crash mid-session still leaves a complete record up to that point.
        """
        try:
            self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = [
                f"# Conversation dump - {self.model}",
                f"_iteration count: {self._snapshot_n}_  ",
                f"_messages: {len(self._messages)}_",
                "",
            ]
            for i, msg in enumerate(self._messages):
                role = msg.get("role", "?")
                content = msg.get("content", "") or ""
                lines.append(f"## [{i:02d}] {role}")
                lines.append("")
                # Wrap as a fenced block so html / xml tags render verbatim.
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
            self.conversation_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    # --- internal helpers --------------------------------------------------

    def _consumed_feedback_summary(self) -> str | None:
        """Return a short human-readable summary of pending feedback to be
        flushed, or None if there's nothing pending. Caller should yield an
        info event with this BEFORE calling _flush_user_injections.
        """
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
        """Prepend any queued feedback / answer onto the next user turn.

        IMPORTANT change vs the old version: feedback used to render as a
        single line `USER FEEDBACK: ...` which 27B-class models would skim
        past. We now put it inside a loud fenced block at the TOP of the
        prompt so the model literally cannot miss it. Same for ANSWER.
        Also tracks injection in the trace so we can audit "did the user's
        words actually reach the model?"
        """
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

    async def _stream(self, on_token) -> str:
        """Stream a chat completion; call on_token(piece) for each chunk.

        Returns the full assembled assistant text. Errors propagate to the
        caller so run() can emit a clean AgentEvent('error') and stop.

        VLM hook: if self._next_image_bytes is set AND we've decided the model
        is vision-capable, we attach the image to the LAST user message in
        history (the one we're about to send the model). The image is then
        cleared so we don't re-send stale frames.
        """
        # Lazily decide whether the model accepts images. Cached after first
        # check so we don't pay for ollama.show() on every turn.
        if self._is_vlm is None:
            self._is_vlm = await self._detect_vlm()
            if self._is_vlm:
                self._trace({"kind": "vlm_detected", "model": self.model})

        # Inject image into the last user message if we have one queued and
        # the model can use it. Mutating the dict in-place is safe because
        # ollama just serializes it for the request.
        if (
            self._is_vlm
            and self._next_image_bytes
            and self._messages
            and self._messages[-1].get("role") == "user"
        ):
            self._messages[-1]["images"] = [self._next_image_bytes]
            self._next_image_bytes = None

        # Adaptive temperature (item C): fix iterations want PRECISION
        # (low temp), first attempts and post-clean turns want CREATIVITY
        # (moderate temp). _fix_mode is set at end of each iteration based
        # on whether the test passed.
        temp = 0.25 if self._fix_mode else 0.7
        self._trace({"kind": "stream_start", "temperature": temp, "fix_mode": self._fix_mode})
        parts: list[str] = []
        async for chunk in await self._client.chat(
            model=self.model,
            messages=self._messages,
            stream=True,
            options={"temperature": temp, "num_ctx": 8192},
        ):
            piece = chunk.get("message", {}).get("content", "") or ""
            if not piece:
                continue
            parts.append(piece)
            on_token(piece)
        return "".join(parts)

    async def _detect_vlm(self) -> bool:
        """Return True if Ollama reports this model has 'vision' capability.

        Newer Ollama models advertise capabilities (e.g. ['completion','vision'])
        via /api/show. If that field is missing or the call fails we assume
        text-only - safer default.
        """
        try:
            info = await self._client.show(model=self.model)
        except Exception:
            return False
        caps = getattr(info, "capabilities", None) or []
        return any(str(c).lower() == "vision" for c in caps)

    @staticmethod
    def _extract_html(reply: str) -> str | None:
        m = _HTML_RE.search(reply)
        if not m:
            return None
        body = m.group(1).strip()
        # Defensive: strip a stray ```html ... ``` fence.
        if body.startswith("```"):
            body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
            body = re.sub(r"\n?```$", "", body)
        return body.strip() or None

    @staticmethod
    def _extract_question(reply: str) -> str | None:
        m = _QUESTION_RE.search(reply)
        return m.group(1).strip() if m else None

    # --- main loop ---------------------------------------------------------

    async def run(self, goal: str) -> AsyncIterator[AgentEvent]:
        """Drive the whole session as an async event stream.

        Yields AgentEvent objects. The TUI iterates with `async for`.
        """
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._messages = [
            # NOTE: use .replace not .format - the skeleton in SYSTEM_PROMPT
            # contains literal CSS/JS braces which .format would try to parse.
            {"role": "system", "content": SYSTEM_PROMPT.replace("{goal}", goal)},
            {"role": "user", "content": PLAN_INSTRUCTION},
        ]

        # Mark the trace with session metadata so each JSONL is self-describing.
        self._trace({
            "kind": "session_start",
            "model": self.model,
            "goal": goal,
            "out_path": str(self.out_path),
            "trace_path": str(self.trace_path),
            "snapshots_dir": str(self.snapshots_dir),
            "best_path": str(self.best_path),
            "max_iters": self.max_iters,
        })
        # Surface trace location in the TUI / log so the user can find it.
        yield self._record(AgentEvent(
            "info",
            f"Trace: {self.trace_path}  (snapshots: {self.snapshots_dir})",
        ))

        # ---- PHASE A: planning ------------------------------------------
        yield self._record(AgentEvent("phase", "planning"))
        try:
            # Same token callback the TUI uses for iterations - planning tokens
            # should stream to the agent log just like everything else.
            plan_reply = await self._stream(self._token_cb_wrapper)
        except Exception as e:
            yield self._record(AgentEvent("error", f"Ollama call failed during planning: {e}"))
            return

        # Emit one big token-burst event AFTER the streaming finishes (the TUI
        # already saw live tokens via the on_tok callback wired in chat.py;
        # see _make_token_callback there). The full text is also stashed in
        # the conversation.
        self._messages.append({"role": "assistant", "content": plan_reply})
        self._dump_conversation()
        yield self._record(AgentEvent("plan", plan_reply))

        # If the model asked a question instead of (or alongside) a plan,
        # surface it now and wait for the user's answer.
        q = self._extract_question(plan_reply)
        if q is not None:
            yield self._record(AgentEvent("question", q))
            # Spin until add_user_answer() is called.
            while self._pending_answer is None:
                await asyncio.sleep(0.1)
            # Don't re-plan from scratch - the model still has its previous
            # turn in history; we just feed the answer in and ask it to plan.
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

        # Hand off to phase B.
        self._messages.append({
            "role": "user",
            "content": self._flush_user_injections(
                "Plan accepted. Now write the FIRST version of the game per "
                "your plan. Output the COMPLETE file in <html_file>...</html_file> "
                "tags as instructed."
            ),
        })

        # ---- PHASE B: build/iterate -------------------------------------
        awaiting_confirm = False

        for iteration in range(1, self.max_iters + 1):
            yield self._record(AgentEvent("phase", f"iteration {iteration}/{self.max_iters}"))

            # Fresh streaming buffer per iteration. on_token closures are set
            # by the TUI via set_token_callback() between iterations - here we
            # just call self._token_cb if present.
            try:
                reply = await self._stream(self._token_cb_wrapper)
            except Exception as e:
                yield self._record(AgentEvent("error", f"Ollama call failed: {e}"))
                return

            self._messages.append({"role": "assistant", "content": reply})
            self._dump_conversation()
            # Also trace the reply so we can reconstruct conversations later.
            self._trace({
                "kind": "assistant_reply",
                "iteration": iteration,
                "len": len(reply),
                "preview": reply[:600],
            })

            # ---- handle <question> from the model ----------------------
            q = self._extract_question(reply)
            html = self._extract_html(reply)
            if q is not None and html is None:
                yield self._record(AgentEvent("question", q))
                while self._pending_answer is None:
                    await asyncio.sleep(0.1)
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        "Thanks. Continue building the game in <html_file> tags."
                    ),
                })
                continue

            # ---- handle confirm-done after critique --------------------
            if awaiting_confirm and html is None and _CONFIRM_RE.search(reply):
                # Never silently drop fresh user feedback. If the user typed
                # something while the model was streaming its <confirm_done/>,
                # treat that as "actually keep going - do this first."
                if self.has_pending_user_input():
                    yield self._record(AgentEvent(
                        "info",
                        "Model said <confirm_done/> but user feedback is pending - applying it instead of exiting.",
                    ))
                    awaiting_confirm = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "You were about to confirm done, but the user just sent "
                            "the feedback above. Address it now and re-send the "
                            "complete game in <html_file>...</html_file>."
                        ),
                    })
                    continue
                yield self._record(AgentEvent("done", "Model confirmed after self-critique."))
                return

            # ---- no <html_file> at all - nudge once --------------------
            if html is None:
                yield self._record(AgentEvent("info", "no <html_file> tag found - nudging model"))
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        "I could not find a <html_file>...</html_file> block in "
                        "your reply. Please re-send the COMPLETE game wrapped in "
                        "those exact tags."
                    ),
                })
                continue

            # ---- save + test in the visible browser --------------------
            self.out_path.write_text(html, encoding="utf-8")
            # Also keep an immutable per-iteration snapshot so we can diff later.
            snap_path = self._save_snapshot(html)
            # Per-iteration screenshot path lives next to the HTML snapshot so
            # iter_NN.html and iter_NN.png pair up by name.
            shot_path = None
            if snap_path is not None:
                shot_path = snap_path.with_suffix(".png")
            yield self._record(AgentEvent(
                "code",
                str(self.out_path),
                {
                    "size": len(html),
                    "snapshot": str(snap_path) if snap_path else None,
                    "screenshot": str(shot_path) if shot_path else None,
                },
            ))

            try:
                report = await self.browser.load_and_test(
                    self.out_path,
                    screenshot_path=shot_path,
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

            yield self._record(AgentEvent("test", format_report_for_model(report), report))

            # If we know (or will discover) the model is vision-capable, queue
            # the latest screenshot bytes so they get attached to the NEXT
            # user message inside _stream(). Never read the file when the model
            # is text-only - saves memory and avoids surprises.
            if shot_path is not None and report.get("screenshot"):
                if self._is_vlm is not False:  # None or True
                    try:
                        self._next_image_bytes = Path(report["screenshot"]).read_bytes()
                    except Exception:
                        self._next_image_bytes = None

            said_done = bool(_DONE_RE.search(reply))

            # Detect "had a working version, just broke it" regression. Used
            # below to switch the next prompt from "fix errors" to "REVERT".
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Mirror to best.html on every clean iteration so the user always
            # has a known-good copy even if the agent regresses afterwards.
            if report["ok"]:
                best = self._save_best(html)
                if best is not None:
                    yield self._record(AgentEvent(
                        "info", f"saved working version to {best}"
                    ))

            # ---- USER force-done shortcut ------------------------------
            # The human explicitly said "ship it". If the run is clean, we
            # exit immediately without a critique pass. If there are still
            # errors, we tell the model to fix them THEN exit.
            if self._user_force_done and report["ok"]:
                # Same protection as confirm-done: pending feedback wins over
                # the user's earlier "ship it" press.
                if self.has_pending_user_input():
                    yield self._record(AgentEvent(
                        "info",
                        "Ship requested but new user feedback arrived - applying it before shipping.",
                    ))
                    # Consume the force-done flag - if the user still wants to
                    # ship after their feedback is applied, they'll press it again.
                    self._user_force_done = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "The user wanted to ship but then sent the feedback above. "
                            "Address that feedback. Re-send the complete game."
                        ),
                    })
                    continue
                yield self._record(AgentEvent("done", "User confirmed - shipping current build."))
                return
            if self._user_force_done and not report["ok"]:
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        f"{format_report_for_model(report)}\n\n"
                        "The user wants to SHIP NOW but errors remain. Fix ONLY "
                        "the errors above (no new features) and re-send the "
                        "complete file."
                    ),
                })
                continue

            # ---- enter self-critique on first clean+done ---------------
            if report["ok"] and said_done and not awaiting_confirm:
                yield self._record(AgentEvent("phase", "self-critique"))
                awaiting_confirm = True
                # Announce feedback consumption BEFORE building the message
                # so the user sees exactly when their words land.
                notice = self._consumed_feedback_summary()
                if notice:
                    yield self._record(AgentEvent("info", notice))
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(CRITIQUE_INSTRUCTION),
                })
                # Track that we're about to head into critique on a clean run.
                self._previous_report_ok = report["ok"]
                continue

            if awaiting_confirm:
                # Model produced new code during a critique round - leave
                # critique mode and treat as a normal iteration.
                awaiting_confirm = False

            # ---- build next user turn ---------------------------------
            # FIX-MODE PROMPTS (B): when the test failed, attach the LAST
            # KNOWN-GOOD code so the model patches against a concrete baseline
            # instead of trying to remember its own previous reply. Empirical-
            # ly this is the biggest single win against regression spirals on
            # 27B-class models.
            best_block = self._best_code_block_or_empty() if not report["ok"] else ""
            if report["ok"]:
                # Toned down: explicitly DISCOURAGE big rewrites when clean.
                # The previous prompt invited "improve" and caused regressions.
                base = (
                    f"{format_report_for_model(report)}\n\n"
                    "No errors. The game already works. STRONGLY prefer ending "
                    "with <done/>. Only re-send a new file if you have a SINGLE "
                    "small concrete improvement and you are confident it will "
                    "not regress. Do NOT make sweeping rewrites."
                )
            elif regressed:
                # Hard regression handler: previous turn passed, this one
                # broke. Tell the model to revert, not to spiral on "fix it".
                base = (
                    "REGRESSION: the previous iteration passed all tests. Your "
                    "latest change introduced these problems:\n\n"
                    f"{format_report_for_model(report)}\n\n"
                    f"{best_block}"
                    "REVERT to the previously-working version above. Re-send it "
                    "as the COMPLETE game in <html_file>...</html_file> tags, "
                    "then either stop with <done/> or attempt a different, "
                    "smaller fix on the next turn."
                )
            else:
                base = (
                    f"{format_report_for_model(report)}\n\n"
                    f"{best_block}"
                    "Fix every ERROR and ISSUE above. Re-send the COMPLETE game in "
                    "<html_file>...</html_file> tags. If a previously-working "
                    "version is shown above, prefer patching IT over rewriting."
                )
            # Same announce: tell the user their feedback (if any) is landing.
            notice = self._consumed_feedback_summary()
            if notice:
                yield self._record(AgentEvent("info", notice))
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(base),
            })
            # Remember this turn's status so the NEXT iteration can detect
            # a clean-then-broken regression.
            self._previous_report_ok = report["ok"]
            # Adaptive temperature (C): if this iteration failed, the next
            # generation is a FIX → use low temp for precision. If clean, the
            # next is exploratory polish (or done) → keep moderate.
            self._fix_mode = not report["ok"]

        # If the user typed feedback right at the iteration cap, give them ONE
        # extra turn so we don't drop their words. Beyond that we really stop.
        if self.has_pending_user_input():
            yield self._record(AgentEvent(
                "info", "Iteration cap reached but user feedback pending - one extra turn.",
            ))
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(
                    "Iteration cap was reached but the user sent the feedback above. "
                    "One last turn: address it and re-send the complete game."
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
                    "preview": reply[:600],
                })
                html = self._extract_html(reply)
                if html is not None:
                    self.out_path.write_text(html, encoding="utf-8")
                    self._save_snapshot(html)
                    yield self._record(AgentEvent(
                        "code", str(self.out_path), {"size": len(html)}
                    ))
            except Exception as e:
                yield self._record(AgentEvent("error", f"Final feedback turn failed: {e}"))

        yield self._record(AgentEvent(
            "info", f"reached max iterations ({self.max_iters}) - stopping"
        ))
        yield self._record(AgentEvent("done", "Iteration cap reached."))

    # --- token callback plumbing ------------------------------------------
    # The TUI sets self._token_cb to a function that pushes characters into
    # the agent log widget. We wrap it so a missing callback (CLI use) is a
    # no-op rather than a crash.

    _token_cb = None  # type: ignore[assignment]

    def set_token_callback(self, cb) -> None:
        """TUI calls this once after construction to receive streaming tokens."""
        self._token_cb = cb

    def _token_cb_wrapper(self, piece: str) -> None:
        if self._token_cb is not None:
            try:
                self._token_cb(piece)
            except Exception:
                # Never let a UI exception kill the agent loop.
                pass
