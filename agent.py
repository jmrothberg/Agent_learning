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

from memory import (
    CANVAS_SKELETON_V2,
    CANVAS_SKELETON_V2_NAME,
    DEFAULT_SKELETON,
    DEFAULT_SKELETON_NAME,
    GameMemory,
    Playbook,
    SkeletonHit,
    render_playbook_block,
    signature_for_report,
)
from ollama_io import Candidate, StreamResult, best_of_n as _best_of_n
from ollama_io import stream_chat, stream_chat_with_retry
from patches import apply_patches, extract_patches

# Prompt-module routing: v0 is the original prompts.py kept for backward
# compatibility; v1+ are siblings (prompts_v1.py, prompts_v2.py, ...). The
# agent picks one at construction time via the `prompt_version` argument.
import prompts as _prompts_v0  # noqa: E402

from tools import LiveBrowser, format_report_for_model, score_test_report


_HTML_RE = re.compile(r"<html_file>\s*(.*?)\s*</html_file>", re.DOTALL | re.IGNORECASE)
# Models that don't follow the <html_file> wrapper instruction often emit a
# markdown ```html fence instead, or just a bare <!DOCTYPE html>...</html>
# block. We accept both as fallbacks so we don't throw away an otherwise
# valid game just because the format anchor was ignored.
_HTML_FENCE_RE = re.compile(
    r"```(?:html|HTML)?\s*\n(.*?\n)```",
    re.DOTALL,
)
_BARE_DOCTYPE_RE = re.compile(
    r"(<!DOCTYPE\s+html[^>]*>.*?</html\s*>)",
    re.DOTALL | re.IGNORECASE,
)
# Some models also write <html_file> with a stray opening but never close it
# (especially after a stall). If we see an opener and a complete <html>
# document inside, we accept the document.
_UNCLOSED_HTML_FILE_RE = re.compile(
    r"<html_file>\s*(?:```(?:html)?\s*\n)?(<!DOCTYPE\s+html.*?</html\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_DONE_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)
_CONFIRM_RE = re.compile(r"<confirm[_-]?done\s*/?>", re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)
_DIAGNOSE_RE = re.compile(r"<diagnose>\s*(.*?)\s*</diagnose>", re.DOTALL | re.IGNORECASE)
_NOTES_RE = re.compile(r"<notes>\s*(.*?)\s*</notes>", re.DOTALL | re.IGNORECASE)
_CRITERIA_RE = re.compile(r"<criteria>\s*(.*?)\s*</criteria>", re.DOTALL | re.IGNORECASE)
_PROBES_RE = re.compile(r"<probes>\s*(.*?)\s*</probes>", re.DOTALL | re.IGNORECASE)


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
        # Total wall-clock budget for one stream. 600s is enough for a 20B
        # model to write a 5K-token game; 35B+ models need more, especially
        # for verbose outputs like full Space Invaders. chat.py bumps this
        # automatically based on detected model parameter_size.
        overall_seconds: float = 600.0,
        memory_root: str | Path = "games/memory",
        # Optional path to an existing HTML file to start from. When set,
        # the agent skips memory-skeleton retrieval and uses this file as
        # the baseline; the model is asked to ADAPT it (via patches) to
        # the user's goal rather than build from scratch.
        seed_file: str | Path | None = None,
        # Which prompt module to load. "v0" = prompts.py (original);
        # "v1", "v2", ... = prompts_vN.py (research-tuned variants). Falls
        # back to v0 if the requested module isn't installed.
        prompt_version: str = "v0",
        # How to seed the first build. "retrieve" (default) = best-match
        # skeleton from past wins; "default" = always use the bundled
        # canvas_basic skeleton (good for tune mode — measures from-scratch
        # ability); "none" = no skeleton, model writes blank-slate.
        skeleton_mode: str = "retrieve",
        # How many playbook bullets to inject per render. v1+ prompts use
        # this; v0 ignores it.
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
    ):
        self.model = model
        self.out_path = Path(out_path)
        self.browser = browser
        self.max_iters = max_iters
        self.best_of_n = max(1, best_of_n)
        self.num_ctx = num_ctx
        self.stall_seconds = stall_seconds
        self.overall_seconds = overall_seconds
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
        self._playbook = Playbook(root=memory_root)
        self._playbook.ensure()
        self._playbook_top_k = max(0, int(playbook_top_k))
        self._playbook_writeback = bool(playbook_writeback)
        # Bullet IDs retrieved on the most recent prompt render — used by
        # the writeback feedback loop to credit/blame them after the next
        # test result.
        self._active_bullet_ids: list[str] = []
        self._skeleton_mode = skeleton_mode
        self._prompt_version = prompt_version
        self._p = self._load_prompt_module(prompt_version)
        self._last_diagnose: str | None = None
        self._stuck_streak: int = 0
        self._use_prefill = bool(use_prefill)
        self._use_vlm_critique = bool(use_vlm_critique)
        self._use_double_screenshot = bool(use_double_screenshot)
        self._use_architect_split = bool(use_architect_split)
        # Most-recent before/after screenshot bytes for the VLM. Filled
        # by the verifier on each iter; consumed by `_stream`.
        self._last_screenshot_before: bytes | None = None
        self._last_screenshot_after: bytes | None = None
        # Acceptance criteria the model emitted during Phase A — fed back
        # into fix prompts so the model self-checks against its own bar.
        self._criteria: str = ""
        # Executable acceptance probes — JS expressions the model proposes
        # in Phase A. Each iter's verifier runs them in the page; results
        # join the report. Empty list = no model probes (universal probes
        # still run).
        self._probes: list[dict] = []
        self._token_cb = None
        self._goal: str = ""
        # Tracks the most recent test-report summary for memory.record_outcome.
        self._last_report_summary: str = ""
        self._last_iter_run: int = 0
        # Tracks the most recent file content actually written to disk. We
        # always inline THIS in fix prompts (instead of asking the model to
        # remember its own previous reply).
        self._current_file: str = ""

    @staticmethod
    def _load_prompt_module(version: str):
        """Resolve the prompt module for `version`. Falls back to v0 silently
        if the requested module isn't importable, and traces the fallback so
        misconfigured tune runs are visible.
        """
        if version == "v0" or not version:
            return _prompts_v0
        try:
            import importlib
            return importlib.import_module(f"prompts_{version}")
        except Exception:
            return _prompts_v0

    def _retrieve_playbook_block(self, goal: str, *, code: str = "") -> str:
        """Get top-K bullets and render them as a `<playbook>` block.

        Empty string when nothing matches OR when the active prompt module
        has set PLAYBOOK_DISABLED = True (gives a v0-prompt the option to
        opt out wholesale).

        Logs the retrieved bullet IDs to the trace so the offline learner
        can later credit/blame each bullet for the eventual outcome.
        """
        if self._playbook_top_k <= 0:
            return ""
        if getattr(self._p, "PLAYBOOK_DISABLED", False):
            return ""
        try:
            hits = self._playbook.retrieve(goal, code=code, k=self._playbook_top_k)
            if hits:
                ids = [h.bullet.id for h in hits]
                self._trace({
                    "kind": "playbook_retrieved",
                    "ids": ids,
                    "scores": [round(h.score, 4) for h in hits],
                    "goal_preview": goal[:120],
                })
                self._active_bullet_ids = list(ids)
            return render_playbook_block(hits)
        except Exception:
            return ""

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

    async def _stream(
        self, on_token, *,
        override_temp: float | None = None,
        prefill: str = "",
    ) -> str:
        """Stream once, with watchdog. Recovers from stalls by raising/logging.

        Image attachment: if VLM is detected and self._next_image_bytes is set,
        attach to the LAST user message and clear the buffer.

        Prefill (Continue.dev pattern): when non-empty AND use_prefill is on,
        a trailing assistant message with `prefill` content is appended so
        Ollama continues from there. The prefill is prepended to the
        returned text so downstream parsers see the full output.
        """
        if self._is_vlm is None:
            self._is_vlm = await self._detect_vlm()
            if self._is_vlm:
                self._trace({"kind": "vlm_detected", "model": self.model})

        if (
            self._is_vlm
            and self._messages
            and self._messages[-1].get("role") == "user"
        ):
            # Multi-image attach: prefer the before/after pair when the
            # double-screenshot feature is on and both are present.
            imgs: list[bytes] = []
            if self._use_double_screenshot:
                if self._last_screenshot_before:
                    imgs.append(self._last_screenshot_before)
                if self._last_screenshot_after:
                    imgs.append(self._last_screenshot_after)
            elif self._next_image_bytes:
                imgs.append(self._next_image_bytes)
            if imgs:
                self._messages[-1]["images"] = imgs
                self._trace({
                    "kind": "image_attached",
                    "count": len(imgs),
                    "bytes": sum(len(b) for b in imgs),
                })
                self._next_image_bytes = None

        # Optional Continue.dev-style assistant prefill. Only applied
        # when feature is on AND `prefill` is provided. We insert a
        # trailing assistant message; Ollama's chat API treats it as a
        # partial completion to extend.
        prefill_used = False
        if self._use_prefill and prefill:
            self._messages.append({"role": "assistant", "content": prefill})
            prefill_used = True
            self._trace({"kind": "prefill", "len": len(prefill)})

        temp = override_temp if override_temp is not None else (
            0.25 if self._fix_mode else 0.7
        )
        self._trace({"kind": "stream_start", "temperature": temp, "fix_mode": self._fix_mode})

        try:
            result = await stream_chat_with_retry(
                self._client,
                self.model,
                self._messages,
                on_token=on_token,
                options={"temperature": temp, "num_ctx": self.num_ctx},
                stall_seconds=self.stall_seconds,
                overall_seconds=self.overall_seconds,
                max_retries=1,
                on_stall=lambda r, attempt: self._trace({
                    "kind": "stream_stalled",
                    "attempt": attempt,
                    "tokens_before_stall": r.tokens,
                    "duration_s": r.duration_s,
                }),
            )
        finally:
            # Always remove our prefill scaffolding before returning so
            # the message history we save & feed to subsequent turns
            # contains a single coherent assistant message.
            if prefill_used and self._messages and self._messages[-1].get("role") == "assistant":
                self._messages.pop()

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
        # Prepend the prefill so downstream parsers (regex for <plan>,
        # <diagnose>, etc.) match against the full intended output.
        return (prefill + result.text) if prefill_used else result.text

    # -- best-of-N for fix iterations --------------------------------------

    async def _generate_and_score_candidates(
        self,
        n: int,
    ) -> tuple[Candidate, list[Candidate]]:
        """Sample N completions and score each by running its result through
        the test harness against a temp file. Used when fixing a failed iter.

        Scorer is a continuous quality score in [0, 100] from
        `score_test_report` so partial-credit candidates ("almost works")
        win over fully-broken ones. Pass = 100; "applied but several
        errors" lands ~30; "wouldn't apply at all" returns -10 so it
        always loses to anything that ran.
        """
        # We deliberately DO NOT stream tokens for these candidates — only
        # the winning one is replayed visually after we pick it.
        async def scorer(text: str) -> tuple[float, dict]:
            extra: dict = {"kind": "candidate", "text_len": len(text)}
            html, applied_msg = await self._materialize(text, dry_run=True)
            extra["materialized"] = bool(html)
            extra["materialize_msg"] = applied_msg
            if not html:
                return -10.0, extra
            tmp_path = self.snapshots_dir / f"cand_{self._snapshot_n+1:02d}_{abs(hash(text))%10000:04d}.html"
            try:
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(html, encoding="utf-8")
                report = await self.browser.load_and_test(
                    tmp_path, screenshot_path=None,
                    probes=self._probes or None,
                )
                extra["report_ok"] = report.get("ok", False)
                extra["report_summary"] = format_report_for_model(report)[:400]
                return score_test_report(report), extra
            except Exception as e:
                extra["scorer_exception"] = str(e)
                # Scorer crashed — treat as worse than "applied but
                # broken" but better than "didn't apply".
                return 10.0, extra

        winner, all_cands = await _best_of_n(
            self._client,
            self.model,
            self._messages,
            n=n,
            options={"num_ctx": self.num_ctx},
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
    def _truncation_diagnosis(reply: str) -> str | None:
        """If the reply looks like an HTML game cut off mid-stream, describe
        the truncation so the user knows it was a stall, not a model dud.

        Returns None if the reply doesn't look truncated (e.g. it was just a
        plan or a chat response with no HTML attempt).
        """
        low = reply.lower()
        has_doctype = "<!doctype" in low
        has_html_open = "<html" in low
        if not (has_doctype or has_html_open):
            return None
        ends = {
            "</html>": "</html>" in low,
            "</body>": "</body>" in low,
            "</script>": "</script>" in low or "<script" not in low,
            "</html_file>": "</html_file>" in low or "<html_file>" not in low,
        }
        missing = [tag for tag, present in ends.items() if not present]
        if not missing:
            return None
        return (
            f"reply began an HTML document ({len(reply):,} bytes streamed) "
            f"but was cut off — missing closing tags: {missing}. "
            f"Likely a stream stall mid-output. Consider a smaller goal, a "
            f"smaller model, or `/iters 1` to re-roll."
        )

    @staticmethod
    def _extract_html(reply: str) -> str | None:
        """Pull a complete HTML game out of a model reply.

        We accept four formats so we never throw away a valid game just
        because the model ignored the <html_file> anchor (a common failure
        mode of smaller models like qwen3.6:27b — they wrap output in a
        markdown fence or emit bare <!DOCTYPE>):

          1. <html_file>BODY</html_file>           ← preferred
          2. <html_file>```html\\nBODY\\n```        ← model double-wrapped
          3. <html_file>BODY (no closing tag, but BODY contains </html>)
             — common after stream stalls truncate the closing tag
          4. ```html\\n<!DOCTYPE html>...</html>\\n```   ← markdown fence only
          5. <!DOCTYPE html>...</html>             ← bare document
        """
        # 1. Canonical wrapper.
        m = _HTML_RE.search(reply)
        if m:
            body = m.group(1).strip()
            if body.startswith("```"):
                body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
                body = re.sub(r"\n?```$", "", body)
            body = body.strip()
            if body:
                return body
        # 2/3. <html_file> opener but no proper close — pull the embedded doc.
        m = _UNCLOSED_HTML_FILE_RE.search(reply)
        if m:
            return m.group(1).strip()
        # 4. Markdown fence whose contents look like HTML.
        for fm in _HTML_FENCE_RE.finditer(reply):
            inner = fm.group(1).strip()
            if "<html" in inner.lower() and "</html" in inner.lower():
                return inner
        # 5. Bare doctype...html fragment anywhere in the reply.
        m = _BARE_DOCTYPE_RE.search(reply)
        if m:
            return m.group(1).strip()
        return None

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

    @staticmethod
    def _extract_criteria(reply: str) -> str | None:
        m = _CRITERIA_RE.search(reply)
        return m.group(1).strip() if m else None

    _ARCHITECT_KEYWORDS = (
        "level", "boss", "multi", "stage", "wave", "campaign",
        "physics", "raycast", "particle", "shader", "3d", "three.js",
        "phaser", "engine", "ai opponent", "tournament", "tile", "rpg",
        "platformer", "scrolling", "parallax", "inventory", "puzzle",
        "minesweeper", "tower defense", "flappy", "endless", "shoot em up",
    )

    @classmethod
    def _is_complex_goal(cls, goal: str) -> bool:
        """Heuristic gate for the architect/editor split. Conservative —
        prefers single-call when uncertain so we don't double the wall-
        clock on simple goals.
        """
        if not goal:
            return False
        g = goal.lower()
        if len(g) > 90:
            return True
        if sum(1 for k in cls._ARCHITECT_KEYWORDS if k in g) >= 1:
            return True
        # Multi-clause goals ("X with Y and Z") are usually richer.
        if g.count(" with ") + g.count(" and ") >= 2:
            return True
        return False

    @staticmethod
    def _extract_probes(reply: str) -> list[dict]:
        """Pull a JSON list-of-{name,expr} out of <probes>...</probes>.

        Tolerant: accepts either a JSON list of objects, or a list of
        plain strings (treated as `expr`, with name auto-assigned). Bad
        JSON returns []; the agent shouldn't ever crash on a probe parse
        failure since universal probes still cover the basics.
        """
        m = _PROBES_RE.search(reply)
        if not m:
            return []
        body = m.group(1).strip()
        # Strip a fenced ```json``` if present.
        body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
        try:
            obj = json.loads(body)
        except Exception:
            return []
        out: list[dict] = []
        if isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                if isinstance(item, dict) and item.get("expr"):
                    out.append({
                        "name": str(item.get("name") or f"probe_{i}")[:60],
                        "expr": str(item["expr"])[:600],
                    })
                elif isinstance(item, str):
                    out.append({"name": f"probe_{i}", "expr": item[:600]})
        return out[:8]   # cap so a chatty model can't bloat the verifier

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
                {"role": "system", "content": self._p.SYSTEM_PROMPT.replace("{goal}", goal)},
                {"role": "user", "content": self._p.PLAN_INSTRUCTION},
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
            crit = self._extract_criteria(plan_reply)
            if crit:
                self._criteria = crit
                self._trace({"kind": "criteria", "text": crit[:600]})
            probes = self._extract_probes(plan_reply)
            if probes:
                self._probes = probes
                self._trace({
                    "kind": "probes_parsed",
                    "count": len(probes),
                    "names": [p.get("name") for p in probes],
                })

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
                pb_block = self._retrieve_playbook_block(goal, code=seed_html)
                pb_kwargs = {"playbook_block": pb_block} if (pb_block and self._prompt_version != "v0") else {}
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        self._p.seed_build_instruction(
                            seed_html, str(self.seed_file), **pb_kwargs,
                        )
                    ),
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

                pb_block = self._retrieve_playbook_block(goal, code=skel.html)
                pb_kwargs = {"playbook_block": pb_block} if (pb_block and self._prompt_version != "v0") else {}

                # Optional architect step — produce an English design
                # before code. Only fires on detected complex goals AND
                # when the feature is on. The architect note becomes
                # part of the first-build user turn so the editor model
                # has a concrete plan to execute.
                architect_note = ""
                if (
                    self._use_architect_split
                    and self._is_complex_goal(goal)
                ):
                    yield self._record(AgentEvent(
                        "phase", "architect",
                        {"goal_chars": len(goal)},
                    ))
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Before code, do an ARCHITECT pass — describe the "
                            "implementation in English. No code, no <html_file>, "
                            "no <patch>. Use this format:\n\n"
                            "<architect>\n"
                            "Data: <key globals / state shape>\n"
                            "Loop: <update / draw responsibilities>\n"
                            "Layers: <draw order, e.g. bg → entities → fx → hud>\n"
                            "Risks: <2-3 places this typically goes wrong>\n"
                            "</architect>\n\n"
                            "Keep it short — 1-2 sentences per line. The next "
                            "turn will hand this to the editor model along with "
                            "the seed code."
                        ),
                    })
                    try:
                        arch_reply = await self._stream(self._token_cb_wrapper)
                    except Exception as e:
                        yield self._record(AgentEvent(
                            "info", f"architect call failed, continuing single-shot: {e}",
                        ))
                        # Pop the architect user message so we don't leave
                        # the conversation in a half-state.
                        if self._messages and self._messages[-1].get("role") == "user":
                            self._messages.pop()
                    else:
                        self._messages.append({"role": "assistant", "content": arch_reply})
                        m = re.search(r"<architect>\s*(.*?)\s*</architect>",
                                      arch_reply, re.DOTALL | re.IGNORECASE)
                        if m:
                            architect_note = m.group(1).strip()
                            self._trace({
                                "kind": "architect_note",
                                "len": len(architect_note),
                            })
                            yield self._record(AgentEvent(
                                "info",
                                f"architect: {architect_note[:160]}",
                            ))

                build_msg = self._p.first_build_instruction(
                    skel.html, skel.source_goal, **pb_kwargs,
                )
                if architect_note:
                    build_msg = (
                        "ARCHITECT NOTE (your own plan from the previous turn — "
                        "follow it):\n"
                        f"<architect>\n{architect_note}\n</architect>\n\n"
                        + build_msg
                    )
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(build_msg),
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
                    # Prefill diagnose tag on fix turns so format compliance
                    # is forced. First-build (iter 1, fix_mode False) doesn't
                    # use diagnose, so prefill is empty there.
                    reply_prefill = ""
                    if self._use_prefill and self._fix_mode:
                        reply_prefill = "<diagnose>\n"
                    reply = await self._stream(
                        self._token_cb_wrapper, prefill=reply_prefill,
                    )
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
                trunc = self._truncation_diagnosis(reply)
                if trunc:
                    yield self._record(AgentEvent("error", f"TRUNCATED REPLY — {trunc}"))
                else:
                    yield self._record(AgentEvent("info", f"no usable code: {materialize_msg}"))
                # If the reply had patches but they failed to apply, give the
                # model the specific failures + current file so it can retry.
                if patches_in_reply and self._current_file:
                    res = apply_patches(self._current_file, patches_in_reply)
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            self._p.patch_retry_instruction(res.failed, self._current_file)
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
            shot_before_path = (
                snap_path.with_name(snap_path.stem + "_before.png")
                if (snap_path and self._use_double_screenshot)
                else None
            )
            try:
                report = await self.browser.load_and_test(
                    self.out_path, screenshot_path=shot_path,
                    screenshot_before_path=shot_before_path,
                    probes=self._probes or None,
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
                        after_bytes = Path(report["screenshot"]).read_bytes()
                        self._next_image_bytes = after_bytes
                        self._last_screenshot_after = after_bytes
                    except Exception:
                        self._next_image_bytes = None
                        self._last_screenshot_after = None
            if self._use_double_screenshot and report.get("screenshot_before"):
                try:
                    self._last_screenshot_before = Path(
                        report["screenshot_before"]
                    ).read_bytes()
                except Exception:
                    self._last_screenshot_before = None

            said_done = bool(_DONE_RE.search(reply))
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Track stuck-streak — used by v1's fix prompt to switch to
            # the "5-7 different sources" reflection ladder after repeat
            # failures on the same goal.
            if report["ok"]:
                self._stuck_streak = 0
            else:
                self._stuck_streak += 1

            # Online playbook counter feedback. Off by default (tune
            # baselines should not write back). When on:
            #   - pass + active bullets → helpful++ for each
            #   - 3rd+ consecutive failure + active bullets → harmful++
            if self._playbook_writeback and self._active_bullet_ids:
                if report["ok"]:
                    self._playbook.update_counters(
                        list(self._active_bullet_ids), helpful_delta=1,
                    )
                    self._trace({
                        "kind": "playbook_writeback",
                        "ids": list(self._active_bullet_ids),
                        "delta": "helpful+1",
                    })
                elif self._stuck_streak >= 3:
                    self._playbook.update_counters(
                        list(self._active_bullet_ids), harmful_delta=1,
                    )
                    self._trace({
                        "kind": "playbook_writeback",
                        "ids": list(self._active_bullet_ids),
                        "delta": "harmful+1",
                    })

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
            return self._p.post_clean_instruction(report_text)

        if regressed:
            best = self._read_best_or_empty()
            return self._p.regression_instruction(report_text, best)

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
        pb_block = self._retrieve_playbook_block(self._goal, code=self._current_file)
        fix_kwargs: dict = {}
        if pb_block and self._prompt_version != "v0":
            fix_kwargs["playbook_block"] = pb_block
        # Track stuck-streak so v1's fix prompt can switch to the
        # 5-7-causes reflection ladder after repeated failures.
        if self._prompt_version != "v0":
            fix_kwargs["stuck_streak"] = self._stuck_streak
        # Feed the model its own Phase-A acceptance criteria so each fix
        # is anchored to "what does the working game owe me?" instead of
        # only "what does the report say is wrong?". Criteria are emitted
        # by v1's PLAN_INSTRUCTION; v0 doesn't ask for them, so this is
        # naturally a no-op there.
        if self._prompt_version != "v0" and self._criteria:
            fix_kwargs["criteria_block"] = self._criteria
        try:
            fix = self._p.fix_instruction(
                report_text, self._current_file, hints, **fix_kwargs,
            )
        except TypeError:
            # v0's signature doesn't take playbook/stuck/criteria kwargs.
            fix = self._p.fix_instruction(report_text, self._current_file, hints)
        if partial_failed:
            fix += (
                "\n\nNOTE: some of your previous patches did not apply. "
                "When fixing this turn, also re-send corrected versions of:\n"
                + "\n".join(f"  - {reason}" for (_i, _p, reason) in partial_failed)
            )
        if self._is_vlm and self._next_image_bytes:
            fix += "\n\n" + self._p.VLM_REVIEW_NOTE

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
