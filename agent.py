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
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from assets import (
    generate_assets,
    parse_assets_block,
    render_asset_paths_block,
    try_load_image_generator,
)
from sounds import (
    generate_sounds,
    parse_sounds_block,
    render_sound_paths_block,
    try_load_audio_generator,
)
from backend import Backend, BackendInfo, make_backend
from memory import (
    CANVAS_SKELETON_V2,
    CANVAS_SKELETON_V2_NAME,
    DEFAULT_SKELETON,
    DEFAULT_SKELETON_NAME,
    GameMemory,
    Playbook,
    SkeletonHit,
    lookup_bullet,
    render_playbook_block,
    signature_for_report,
)
from ollama_io import Candidate, StreamResult
from patches import apply_patches, extract_patches

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


# Pi-mono pattern: read AGENTS.md / CLAUDE.md from the working tree at
# session start and append it as <project-context> in the system prompt.
# Lets a repo enforce house-style ("always vanilla JS, no React") once
# instead of re-saying it via feedback every session.
_PROJECT_CONFIG_FILES = ("AGENTS.md", "CLAUDE.md")
# Cap so a sprawling project README doesn't crowd out the rest of the
# system prompt. ~6KB ≈ 1500 tokens, still room for the goal + workflow.
_PROJECT_CONFIG_MAX_CHARS = 6000


def _read_project_config(base_dir: Path) -> tuple[str, list[str]]:
    """Read AGENTS.md / CLAUDE.md (in that order) from `base_dir`.

    Returns (concat_text, source_paths). `concat_text` is empty if no
    project-config files exist or are readable. Total length is capped
    at _PROJECT_CONFIG_MAX_CHARS; truncation appends a marker so the
    model knows it was cut.
    """
    parts: list[str] = []
    sources: list[str] = []
    used = 0
    for name in _PROJECT_CONFIG_FILES:
        p = base_dir / name
        try:
            if not p.is_file():
                continue
            body = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not body.strip():
            continue
        sources.append(str(p))
        remaining = _PROJECT_CONFIG_MAX_CHARS - used
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[:remaining] + (
                f"\n\n[... {name} truncated to fit project-context budget ...]"
            )
        # Prefix each file with its name so the model can tell them apart
        # if the project ships both.
        parts.append(f"## {name}\n\n{body.strip()}")
        used += len(body) + len(name) + 8
    return ("\n\n".join(parts), sources)


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
# Pi-mono "skills" pattern: <lookup_bullet>id</lookup_bullet> requests
# the full body of a playbook bullet whose index-entry was inlined in
# hybrid mode. Resolved + injected at the next user-turn boundary.
_LOOKUP_BULLET_RE = re.compile(
    r"<lookup_bullet>\s*(.*?)\s*</lookup_bullet>", re.DOTALL | re.IGNORECASE
)
# Cap to keep one chatty reply from blowing up context.
_MAX_BULLET_LOOKUPS_PER_TURN = 5


# Block-level bloat detector for full-rewrite paths. Local LLMs sometimes
# duplicate a large literal (a maze 2D array, a tilemap, a const-table)
# 3+ times within one response — the streaming detector in ollama_io
# catches most cases live, but this is the materialize-time safety net.
# Returns a short human-readable description of the duplication, or None.
_BLOAT_BLOCK_LINES = 8       # 8 consecutive lines = one "block" hash
_BLOAT_MIN_BLOCK_BYTES = 200 # skip whitespace-y blocks
_BLOAT_MAX_REPEATS = 3       # > 3 identical blocks = bloat


def _detect_block_bloat(text: str) -> str | None:
    """Scan `text` for repeated N-line blocks. None if clean."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < _BLOAT_BLOCK_LINES * (_BLOAT_MAX_REPEATS + 1):
        return None
    counts: dict[int, int] = {}
    for i in range(len(lines) - _BLOAT_BLOCK_LINES + 1):
        block = "\n".join(lines[i:i + _BLOAT_BLOCK_LINES])
        if len(block) < _BLOAT_MIN_BLOCK_BYTES:
            continue
        h = hash(block)
        counts[h] = counts.get(h, 0) + 1
        if counts[h] > _BLOAT_MAX_REPEATS:
            sample = block.replace("\n", " ↵ ")[:160]
            return (
                f"the same {_BLOAT_BLOCK_LINES}-line block appears "
                f"{counts[h]}+ times: '{sample}...'"
            )
    return None


# Once the messages list has more turns than this, older turns get their
# embedded code (<html_file> bodies, ```html fences) replaced with summaries
# so context stays bounded. Tunable; the agent always passes the CURRENT
# file inline in the fix prompt, so old code in history is just bloat.
_PRUNE_KEEP_RECENT_TURNS = 4

# Above this total-message count, switch from per-turn HTML elision to a
# pi-mono-style STRUCTURED COMPACTION: replace messages 1..cutoff with a
# single deterministic summary that captures goal, criteria, progress,
# stuck-streak, and last test report. The system prompt + last
# _PRUNE_KEEP_RECENT_TURNS turns survive intact. Threshold tuned so a 6-iter
# run rarely triggers it (planning + first build + ~5 fix turns ≈ 12 msgs)
# but a long extension session does.
_STRUCTURED_PRUNE_THRESHOLD = 14


@dataclass
class AgentEvent:
    kind: str           # phase | token | plan | code | test | question | done | error | info | diagnose | patch | best_of_n | memory | activity | assets | streak
    text: str = ""
    data: dict = field(default_factory=dict)


class GameAgent:
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
        best_of_n: int = 1,
        # Ollama context window. qwen3.6:27b/35b natively supports 128K+,
        # gpt-oss supports 128K — at 8K we were truncating mid-<assets>
        # block on long planning turns (see games/traces/make-a-small-
        # first-person-shoo_20260506_222042). Bumped default to 32768
        # which fits the system prompt + plan + first-build with room
        # for several feedback iterations before structured compaction.
        # Override explicitly via constructor or via the
        # CODING_BOX_NUM_CTX env var if your model needs different.
        # Note: changing num_ctx between calls forces an Ollama model
        # reload — to avoid that, preload at the desired size with
        # `ollama run --ctx-size 32768 <model>` before starting a session.
        num_ctx: int = 32768,
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
        self.out_path = Path(out_path)
        self.browser = browser
        self.max_iters = max_iters
        self.best_of_n = max(1, best_of_n)
        self.num_ctx = num_ctx
        self.stall_seconds = stall_seconds
        self.overall_seconds = overall_seconds
        self.seed_file: Path | None = Path(seed_file) if seed_file else None
        self._messages: list[dict] = []
        self._pending_feedback: list[str] = []
        self._pending_answer: str | None = None
        self._user_force_done = False
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
        # Released by signal_step_continue() to unblock a step-mode wait
        # without adding any user feedback. add_user_feedback also
        # unblocks (via has_pending_user_input becoming True).
        self._step_continue: bool = False

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
        self._skeleton_mode = skeleton_mode
        self._prompt_version = prompt_version
        self._p = self._load_prompt_module(prompt_version)
        self._last_diagnose: str | None = None
        self._stuck_streak: int = 0
        self._use_prefill = bool(use_prefill)
        self._use_vlm_critique = bool(use_vlm_critique)
        self._use_double_screenshot = bool(use_double_screenshot)
        self._use_architect_split = bool(use_architect_split)
        # todo #6 — resolve "auto" via simple substring-match. Adding a
        # name to _MID_MODEL_TAGS is a one-line opt-in for new families.
        self._model_class: str = (
            model_class if model_class in ("small", "mid", "large")
            else self._classify_model(model)
        )
        self._trace({"kind": "model_class_resolved", "model": model, "model_class": self._model_class})
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
        # Same lazy-load pattern for Stable Audio Open. Only loaded when
        # the model emits a <sounds> block in Phase A.
        self._sound_generator: Any = None
        # Resolved sound paths (name → absolute path) and the subset
        # that was declared loop=true. The loop set is preserved
        # separately so render_sound_paths_block can mark them in the
        # injected loader pattern.
        self._session_sounds: dict[str, Path] = {}
        self._session_looping: set[str] = set()
        self.restart_n: int = max(1, int(restart_n))
        self.restart_score_threshold: float = float(restart_score_threshold)

    # Read-through to the resolved backend's model id. Existing call sites
    # (trace metadata, conversation dump, memory.record_outcome, ...) used
    # `self.model` as a string; keeping it as a property means the agent
    # always reports whatever the backend resolved to without callers
    # having to know about Backend internals.
    @property
    def model(self) -> str:
        return self._backend.info.model

    @classmethod
    def _classify_model(cls, model: str) -> str:
        """Default model class.

        We deliberately do NOT inspect the model name. The user runs a
        rotating set of mid-size local LLMs (~27B-class) — qwen3.6, the
        next qwen, whatever ships next quarter — and a model-name table
        would go stale every release. The class is "small" by default:
        the lean ~5 KB system prompt + drop of the <assets>/<sounds>/
        <lookup_bullet> pipelines, biased for one-shot strength on simple
        games. Pass `model_class="large"` explicitly when running a
        frontier-tier model that can absorb the full schema.
        """
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

    # OpenCoder #1 — two-stage retrieval (broad-then-narrow). Plan stage
    # gets a wider, more permissive cut of the playbook (small models
    # benefit from "see the whole space"); code stage gets a tighter cut
    # of validated patterns only (no net-harmful bullets, fewer entries,
    # smaller char budget). Mirrors OpenCoder's two-stage SFT — broad
    # first, narrow second.
    _PLAN_STAGE_TOP_K_BONUS = 2          # plan retrieves K + bonus bullets
    _CODE_STAGE_TOP_K = 3                # narrow cut at code time
    _PLAN_STAGE_CHAR_BUDGET = 4500       # ~1100 tokens, broad context
    _CODE_STAGE_CHAR_BUDGET = 2400       # ~600 tokens, tight context

    def _retrieve_playbook_block(
        self,
        goal: str,
        *,
        code: str = "",
        stage: str = "code",
    ) -> str:
        """Get top-K bullets and render them as a `<playbook>` block.

        `stage` selects OpenCoder-style two-stage retrieval:
          - "plan" (Stage-1, broad): top_k+bonus bullets, all positive
            relevance hits including net-harmful (exposure to history),
            larger char budget.
          - "code" (Stage-2, narrow, default): top-3 only, drops bullets
            with score ≤ -2 (validated-only patterns), smaller budget.

        After retrieval, `render_playbook_block` runs shingle dedup
        (OpenCoder #5) and budget capping (OpenCoder #2) before emitting
        the prompt block.

        Empty string when nothing matches OR when the active prompt module
        has set PLAYBOOK_DISABLED = True (gives a v0-prompt the option to
        opt out wholesale). Logs retrieved bullet IDs + stage to the trace
        so the offline learner can later credit/blame each bullet for the
        eventual outcome.
        """
        if self._playbook_top_k <= 0:
            return ""
        if getattr(self._p, "PLAYBOOK_DISABLED", False):
            return ""
        try:
            if stage == "plan":
                k = self._playbook_top_k + self._PLAN_STAGE_TOP_K_BONUS
                budget = self._PLAN_STAGE_CHAR_BUDGET
                # Stop-Losing-To-OneShot todo #6 — mid-tier models lose
                # focus when the playbook bloats the planning context;
                # collapse the plan-stage budget to match code-stage so
                # the goal stays prominent. The retrieval still fetches
                # k+bonus bullets (more diversity) — only the rendered
                # char budget is tightened.
                if self._model_class in ("mid", "small"):
                    budget = self._CODE_STAGE_CHAR_BUDGET
                # Plan stage advertises breadth: top-3 full + the rest as
                # ID-only index. Model emits <lookup_bullet> if it wants
                # the body of any indexed entry. Pi-mono "skills" pattern.
                render_mode = "hybrid"
            else:
                k = min(self._playbook_top_k, self._CODE_STAGE_TOP_K)
                budget = self._CODE_STAGE_CHAR_BUDGET
                # Code stage already narrowly retrieves; full bodies on all.
                render_mode = "full"
            hits = self._playbook.retrieve(
                goal, code=code, k=k, stage=stage,
            )
            if hits:
                ids = [h.bullet.id for h in hits]
                self._trace({
                    "kind": "playbook_retrieved",
                    "stage": stage,
                    "ids": ids,
                    "scores": [round(h.score, 4) for h in hits],
                    "goal_preview": goal[:120],
                    "char_budget": budget,
                    "render_mode": render_mode,
                })
                self._active_bullet_ids = list(ids)
            return render_playbook_block(
                hits, char_budget=budget, mode=render_mode,
            )
        except Exception:
            return ""

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
            self._trace({"kind": "feedback_queued", "text": text})

    def add_user_answer(self, text: str) -> None:
        self._pending_answer = text.strip()
        self._trace({"kind": "answer_queued", "text": self._pending_answer})

    def has_pending_user_input(self) -> bool:
        return bool(self._pending_feedback) or self._pending_answer is not None

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
        existing path)."""
        self._step_mode = bool(on)
        self._trace({"kind": "step_mode_set", "on": self._step_mode})

    def signal_step_continue(self) -> None:
        """Release the current step-mode wait without adding feedback.
        No-op when no wait is active."""
        self._step_continue = True
        self._trace({"kind": "step_continue_signal"})

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

    def _prune_messages(self) -> None:
        """Compress old turns so context stays bounded.

        Two strategies, by message count:
          * ≤ _PRUNE_KEEP_RECENT_TURNS+1 messages: no-op.
          * ≤ _STRUCTURED_PRUNE_THRESHOLD: per-turn HTML elision (the
            original behavior — keeps message shape, strips embedded HTML).
          * > _STRUCTURED_PRUNE_THRESHOLD: pi-mono-style structured
            compaction — replace messages 1..cutoff with a single
            deterministic state-anchor message; keep system prompt and
            last _PRUNE_KEEP_RECENT_TURNS turns.

        The system prompt (index 0) and the most recent K turns are
        always preserved verbatim.
        """
        n = len(self._messages)
        if n <= 1 + _PRUNE_KEEP_RECENT_TURNS:
            return

        if n > _STRUCTURED_PRUNE_THRESHOLD:
            cutoff = n - _PRUNE_KEEP_RECENT_TURNS
            summary = self._build_structured_summary()
            anchor_msg = {
                "role": "user",
                "content": (
                    "================ STATE ANCHOR (compaction) ================\n"
                    "Older turns were elided to keep context bounded. The "
                    "snapshot below is a deterministic summary of session "
                    "state — treat it as authoritative for goal, criteria, "
                    "progress, and critical context.\n\n"
                    f"{summary}\n"
                    "==========================================================="
                ),
            }
            new_messages = [self._messages[0], anchor_msg] + self._messages[cutoff:]
            self._trace({
                "kind": "structured_compaction",
                "original_messages": n,
                "kept_recent": _PRUNE_KEEP_RECENT_TURNS,
                "summary_chars": len(summary),
                "new_messages": len(new_messages),
            })
            self._messages = new_messages
            return

        # Default elision path: keep message shape, strip embedded HTML
        # bodies. Cheap, lossy on iteration history, but safe.
        cutoff = n - _PRUNE_KEEP_RECENT_TURNS
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
        # Snapshot the queue BEFORE consuming so we can push a visible
        # "✓ APPLIED to this turn" confirmation into the agent log via
        # the TUI's token callback. Without this, only the right-hand
        # status panel reflects the queue draining — the left-hand log
        # (where the user's eye lives) shows nothing, leaving them
        # uncertain whether their typing actually reached the model.
        consumed_items: list[str] = []
        if self._pending_answer is not None:
            consumed_items.append(f"answer: {self._pending_answer[:120]}")
        for fb in self._pending_feedback:
            consumed_items.append(f"feedback: {fb[:120]}")

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
        # Drain any <lookup_bullet> resolutions queued by the previous
        # assistant reply. These come BEFORE the base message so the
        # model sees them as fresh material before the iteration prompt.
        if self._pending_bullet_lookups:
            for block in self._pending_bullet_lookups:
                parts.append(block)
            self._pending_bullet_lookups.clear()
        if base_message:
            parts.append(base_message)

        # Push a confirmation line into the TUI agent log via the token
        # callback. Plain text only — the TUI renders streamed tokens
        # via Rich's Text() (no markup parsing) so any [tag] would
        # appear literally. Newlines bracket the line so the streaming
        # buffer flushes it as a discrete log line. Bypassed on CLI
        # runs (no callback wired) — those users see feedback_injected
        # events in the trace instead.
        if consumed_items and self._token_cb is not None:
            try:
                preview = "; ".join(consumed_items)
                self._token_cb(f"\n>> APPLIED to this turn: {preview}\n")
            except Exception:
                pass

        return "\n\n".join(parts)

    # -- streaming ----------------------------------------------------------

    async def _detect_vlm(self) -> bool:
        return await self._backend.is_vlm()

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
        hb_state = {
            "started": _time.monotonic(),
            "last_hb": _time.monotonic(),
            "tokens": 0,
            "tail": "",
        }
        _STREAM_HEARTBEAT_SECONDS = 30.0
        _STREAM_HEARTBEAT_TAIL_CHARS = 120

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

        def _on_progress(stage: str, current: int, total: int) -> None:
            self._stream_progress_stage = stage
            self._stream_progress_current = current
            self._stream_progress_total = total
            self._trace({
                "kind": "stream_progress",
                "stage": stage,
                "current": current,
                "total": total,
            })

        try:
            result = await self._backend.stream_chat(
                self._messages,
                on_token=_heartbeat_on_token,
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
                on_progress=_on_progress,
                cancel_event=self._ensure_stop_event(),
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
            "looped": result.looped,
            "crashed": result.crashed,
            "len": len(result.text),
            # Backend-reported BPE counts when available. `tokens` above is
            # streaming chunk count; these are the real cost numbers used
            # to chart prompt size over a session and to spot when the
            # inlined-file truth source has bloated the input.
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        })
        if result.crashed:
            # mlx_lm.server's generate thread died mid-flight (Metal
            # wired-memory limit, OOM, segfault). HTTP layer is still
            # answering /v1/models, but the loaded model can't generate
            # anything. Surface the specific recovery hint instead of
            # the generic stall message — saves the user from digging
            # through mlx_lm.server's stderr in another terminal.
            self._record(AgentEvent(
                "error",
                "[red]mlx_lm.server crashed mid-generation[/red] — almost "
                "always Metal wired-memory exhaustion. Recover with:\n"
                "  1. [b]pkill -f mlx_lm.server[/b]\n"
                "  2. [b]sudo sysctl iogpu.wired_limit_mb=$N[/b] "
                "(see README §MLX memory limit on Apple Silicon)\n"
                "  3. relaunch [b]mlx_lm.server[/b] and try again",
                {
                    "tokens_at_crash": result.tokens,
                    "duration_s": round(result.duration_s, 2),
                },
            ))
        if result.looped:
            # Visible to the user via the agent log so they understand why
            # the stream cut off mid-output. Trim trailing whitespace from
            # the partial text so downstream regexes see a clean tail.
            self._record(AgentEvent(
                "info",
                f"[yellow]repetition loop detected[/yellow] — model was emitting "
                f"the same 1-2 short lines on repeat after {result.tokens} tokens "
                f"({result.duration_s:.0f}s). Aborted stream and kept partial output."
            ))
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
                    # todo #2: pass criteria so the harness can flag
                    # coverage gaps even on best-of-N candidate scoring.
                    criteria=self._criteria or None,
                )
                extra["report_ok"] = report.get("ok", False)
                extra["report_summary"] = format_report_for_model(report)[:400]
                return score_test_report(report), extra
            except Exception as e:
                extra["scorer_exception"] = str(e)
                # Scorer crashed — treat as worse than "applied but
                # broken" but better than "didn't apply".
                return 10.0, extra

        winner, all_cands = await self._backend.best_of_n(
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
            cancel_event=self._ensure_stop_event(),
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
            if (
                not dry_run
                and self._current_file
                and self._snapshot_n >= 1
                and not allow_rewrite
            ):
                return None, (
                    "<html_file> rejected: a baseline file already exists. "
                    "Send <patch> SEARCH/REPLACE blocks instead. (Override: "
                    "AGENT_ALLOW_FULL_REWRITE=1 — only when patches truly "
                    "cannot express the structural change.)"
                )
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

    # Threshold above which we consider the current file too large to
    # inject in full on every fix turn. Below this, full-file inject is
    # cheap and removes any risk of the slice missing context.
    _FULL_FILE_INJECT_LIMIT = 12_000

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
        return "\n\n".join(kept)

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

        BUG fixed in this version: `self._goal` was being OVERWRITTEN with
        the new feedback on continuation, so the structured-compaction
        summary's "Goal" line ended up reading "use the art you generate"
        (the latest feedback) instead of the original "doom shooter"
        request. Now we only set `_goal` on a fresh session and treat the
        continuation `goal` arg as feedback, leaving the original intact.
        """
        if not continuation:
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
            if self._stop_event is not None:
                self._stop_event.clear()
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
            # Open-domain research: try Wikipedia for the goal before
            # planning. If we get a hit, prepend the reference to the
            # planning user-turn so the model's <plan> + <criteria> are
            # grounded in real mechanics instead of model priors. The
            # missile-command session in games/traces/ shipped Space
            # Invaders with the labels swapped — that's the failure
            # mode this prevents.
            #
            # Networked + has its own rate-limit sleeps, so run in a
            # worker thread to keep the TUI responsive. Total budget
            # ~3s for a typical hit, ~6s worst case.
            reference_block = ""
            try:
                import research as _research
                reference_block = await asyncio.to_thread(_research.fetch, goal) or ""
            except Exception as e:
                yield self._record(AgentEvent(
                    "info", f"research lookup failed: {e!r}"
                ))

            if hasattr(self._p, "plan_instruction"):
                # v1+ planner takes goal so it can detect art-modality
                # keywords ("sprite", "art", "graphics") and escalate
                # the <assets> directive to REQUIRED for that turn.
                # Tolerant of older signatures: try with goal first,
                # fall through to the reference-only call.
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

            # Pi-mono-style project-config injection. Reads AGENTS.md /
            # CLAUDE.md from cwd; falls back to the out_path's parent so
            # `python coder.py` from outside the project tree still picks
            # them up. Appended INSIDE the system message rather than as
            # a separate user turn — keeps it ambient instead of
            # interactive.
            # Stop-Losing-To-OneShot todo #6 — when the active prompt
            # module exposes build_system_prompt (v1+), pass model_class
            # so mid-tier models get a trimmed prompt. v0 falls back
            # to the static SYSTEM_PROMPT constant unchanged.
            if hasattr(self._p, "build_system_prompt"):
                sys_prompt = self._p.build_system_prompt(
                    goal, model_class=self._model_class,
                )
            else:
                sys_prompt = self._p.SYSTEM_PROMPT.replace("{goal}", goal)
            cfg_text, cfg_sources = _read_project_config(Path.cwd())
            if not cfg_text:
                cfg_text, cfg_sources = _read_project_config(self.out_path.parent.parent)
            if cfg_text:
                sys_prompt = (
                    sys_prompt
                    + "\n\n<project-context>\n"
                    + "Project-level configuration loaded from the working tree. "
                    + "Treat its rules as additional hard requirements; they "
                    + "override anything in <hard-rules> or <iteration-policy> "
                    + "where they conflict.\n\n"
                    + cfg_text
                    + "\n</project-context>"
                )
                self._trace({
                    "kind": "project_config_loaded",
                    "sources": cfg_sources,
                    "chars": len(cfg_text),
                })

            self._messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": plan_msg},
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
                "reference_chars": len(reference_block),
            })
            yield self._record(AgentEvent(
                "info",
                f"Trace: {self.trace_path}  (snapshots: {self.snapshots_dir})",
            ))
            if reference_block:
                # Surface the matched title so the user sees what we pulled.
                first_line = reference_block.splitlines()[1] if "\n" in reference_block else ""
                yield self._record(AgentEvent(
                    "info",
                    f"research: pulled Wikipedia reference ({len(reference_block)} chars) "
                    f"— {first_line}"
                ))
            else:
                yield self._record(AgentEvent(
                    "info",
                    "research: no Wikipedia match for this goal "
                    "(planning from model priors)"
                ))

            # ---- PHASE A: planning ------------------------------------------
            yield self._record(AgentEvent("phase", "planning"))
            yield self._record(AgentEvent("activity", "streaming", {"label": "planning reply"}))
            try:
                plan_reply = await self._stream(self._token_cb_wrapper)
            except Exception as e:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent(
                    "error",
                    f"{self._backend.info.name.upper()} call failed during planning: {e}",
                ))
                return
            yield self._record(AgentEvent("activity", "idle"))
            self._messages.append({"role": "assistant", "content": plan_reply})
            self._extract_and_queue_lookups(plan_reply)
            self._dump_conversation()
            yield self._record(AgentEvent("plan", plan_reply))
            crit = self._extract_criteria(plan_reply)
            if crit:
                self._criteria = crit
                self._trace({"kind": "criteria", "text": crit[:600]})
            probes = self._extract_probes(plan_reply)
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
                yield self._record(AgentEvent(
                    "activity", "streaming",
                    {"label": "streaming plan after question"},
                ))
                try:
                    plan_reply = await self._stream(self._token_cb_wrapper)
                except Exception as e:
                    yield self._record(AgentEvent("activity", "idle"))
                    yield self._record(AgentEvent(
                        "error",
                        f"{self._backend.info.name.upper()} call failed: {e}",
                    ))
                    return
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({"role": "assistant", "content": plan_reply})
                self._extract_and_queue_lookups(plan_reply)
                self._dump_conversation()
                yield self._record(AgentEvent("plan", plan_reply))

            # ---- Phase A → first-build: optional asset generation ----------
            # If the model emitted <assets> in its plan AND the local
            # Z-Image-Turbo diffuser is reachable, generate sprites and
            # collect their paths. Both halves are optional: no-asset
            # plans skip everything; reachable-but-no-GPU systems log
            # and skip; failures on individual assets don't abort.
            asset_specs = parse_assets_block(plan_reply)
            if asset_specs:
                yield self._record(AgentEvent(
                    "info",
                    f"plan requested {len(asset_specs)} asset(s); "
                    "loading Z-Image-Turbo (first call only, ~30-60s)…",
                    {"assets_requested": [s["name"] for s in asset_specs]},
                ))
                yield self._record(AgentEvent(
                    "activity", "generating_assets",
                    {
                        "label": "generating sprites",
                        "requested": len(asset_specs),
                        "produced": 0,
                    },
                ))
                if self._asset_generator is None:
                    # asyncio.to_thread keeps the TUI responsive during
                    # the (slow) first-load of the diffusion pipeline.
                    self._asset_generator = await asyncio.to_thread(
                        try_load_image_generator,
                    )
                if self._asset_generator is None:
                    yield self._record(AgentEvent("activity", "idle"))
                    yield self._record(AgentEvent(
                        "info",
                        "Z-Image-Turbo not reachable (no CUDA / no diffusers / "
                        "Colossal_Cave/diffusion_manager.py missing) — "
                        "proceeding without assets, model will draw "
                        "procedurally."
                    ))
                else:
                    session_assets_dir = (
                        self.out_path.parent / f"{self._session_id}_assets"
                    )
                    try:
                        self._session_assets = await asyncio.to_thread(
                            generate_assets,
                            asset_specs,
                            session_assets_dir,
                            image_generator=self._asset_generator,
                        )
                    except Exception as e:
                        yield self._record(AgentEvent("activity", "idle"))
                        yield self._record(AgentEvent(
                            "info",
                            f"asset generation crashed: {e!r} — proceeding without."
                        ))
                        self._session_assets = {}
                    # 2.2: pull per-asset stats (prompt, native_size,
                    # bg_color, alpha_pixel_ratio, gen_seconds, errors)
                    # off the generator instance — generate_assets
                    # stashes them there. Lets the trace record exactly
                    # which assets were cache hits vs fresh, what bg
                    # color was detected, and how transparent the
                    # chroma-key actually got.
                    per_asset = getattr(
                        self._asset_generator, "last_stats", None,
                    ) or []
                    self._trace({
                        "kind": "assets_generated",
                        "requested": len(asset_specs),
                        "produced": len(self._session_assets),
                        "names": list(self._session_assets.keys()),
                        "session_dir": str(session_assets_dir),
                        "per_asset": per_asset,
                    })
                    # Always emit the structured assets event (even on 0
                    # produced) so the TUI can show "0/N generated" with
                    # paths and per-asset error reasons.
                    yield self._record(AgentEvent(
                        "assets",
                        f"{len(self._session_assets)}/{len(asset_specs)} generated",
                        {
                            "requested": len(asset_specs),
                            "produced": len(self._session_assets),
                            "session_dir": str(session_assets_dir),
                            "paths": {n: str(p) for n, p in self._session_assets.items()},
                            "per_asset": per_asset,
                        },
                    ))
                    yield self._record(AgentEvent("activity", "idle"))
                    if self._session_assets:
                        yield self._record(AgentEvent(
                            "info",
                            f"generated {len(self._session_assets)}/"
                            f"{len(asset_specs)} sprites at "
                            f"{session_assets_dir}",
                            {"assets": {n: str(p) for n, p in self._session_assets.items()}},
                        ))
                    # When we asked for assets but produced fewer than
                    # requested, surface the per-asset error reasons as
                    # info events so the .log mirror picks them up
                    # (status panel alone is too easy to miss). Each
                    # failure gets one line so the user sees the actual
                    # diffuser error — fp16 NaN, model path miss,
                    # NSFW filter, etc — instead of just "0/N generated".
                    if len(self._session_assets) < len(asset_specs):
                        failed = [
                            s for s in per_asset
                            if isinstance(s, dict) and s.get("error")
                        ]
                        if failed:
                            yield self._record(AgentEvent(
                                "info",
                                f"asset gen: {len(failed)}/{len(asset_specs)} "
                                f"failed — see per-asset reasons below"
                            ))
                            for s in failed:
                                name = s.get("name", "?")
                                err = s.get("error", "(no reason captured)")
                                # Truncate so a giant traceback doesn't
                                # blow up the log line — the trace JSONL
                                # has the full untruncated error.
                                err_line = str(err)[:400]
                                yield self._record(AgentEvent(
                                    "info",
                                    f"  - {name}: {err_line}"
                                ))

            # ---- Phase A → first-build: optional sound generation ----------
            # Mirrors the asset block above. If the model emitted <sounds>
            # AND Stable Audio Open is reachable, generate OGGs and
            # collect their paths. Both halves are optional: silent plans
            # skip everything; reachable-but-no-GPU systems log and skip;
            # individual failures don't abort the batch.
            sound_specs = parse_sounds_block(plan_reply)
            if sound_specs:
                yield self._record(AgentEvent(
                    "info",
                    f"plan requested {len(sound_specs)} sound(s); "
                    "loading Stable Audio Open (first call only, ~30-60s)…",
                    {"sounds_requested": [s["name"] for s in sound_specs]},
                ))
                yield self._record(AgentEvent(
                    "activity", "generating_sounds",
                    {
                        "label": "generating sounds",
                        "requested": len(sound_specs),
                        "produced": 0,
                    },
                ))
                if self._sound_generator is None:
                    self._sound_generator = await asyncio.to_thread(
                        try_load_audio_generator,
                    )
                if self._sound_generator is None:
                    yield self._record(AgentEvent("activity", "idle"))
                    yield self._record(AgentEvent(
                        "info",
                        "Stable Audio Open not reachable (no CUDA / no MPS / "
                        "diffusers or soundfile missing) — proceeding "
                        "without audio, model will ship a silent game."
                    ))
                else:
                    session_sounds_dir = (
                        self.out_path.parent / f"{self._session_id}_sounds"
                    )
                    try:
                        self._session_sounds = await asyncio.to_thread(
                            generate_sounds,
                            sound_specs,
                            session_sounds_dir,
                            audio_generator=self._sound_generator,
                        )
                    except Exception as e:
                        yield self._record(AgentEvent("activity", "idle"))
                        yield self._record(AgentEvent(
                            "info",
                            f"sound generation crashed: {e!r} — proceeding without."
                        ))
                        self._session_sounds = {}
                    # Track which produced names were declared with
                    # loop=true so the loader pattern can mark them.
                    self._session_looping = {
                        str(s.get("name", "")).strip()
                        for s in sound_specs if s.get("loop")
                    } & set(self._session_sounds.keys())
                    per_sound = getattr(
                        self._sound_generator, "last_stats", None,
                    ) or []
                    self._trace({
                        "kind": "sounds_generated",
                        "requested": len(sound_specs),
                        "produced": len(self._session_sounds),
                        "names": list(self._session_sounds.keys()),
                        "looping": sorted(self._session_looping),
                        "session_dir": str(session_sounds_dir),
                        "per_sound": per_sound,
                    })
                    yield self._record(AgentEvent(
                        "sounds",
                        f"{len(self._session_sounds)}/{len(sound_specs)} generated",
                        {
                            "requested": len(sound_specs),
                            "produced": len(self._session_sounds),
                            "session_dir": str(session_sounds_dir),
                            "paths": {n: str(p) for n, p in self._session_sounds.items()},
                            "looping": sorted(self._session_looping),
                            "per_sound": per_sound,
                        },
                    ))
                    yield self._record(AgentEvent("activity", "idle"))
                    if self._session_sounds:
                        yield self._record(AgentEvent(
                            "info",
                            f"generated {len(self._session_sounds)}/"
                            f"{len(sound_specs)} sounds at "
                            f"{session_sounds_dir}",
                            {"sounds": {n: str(p) for n, p in self._session_sounds.items()}},
                        ))
                    if len(self._session_sounds) < len(sound_specs):
                        failed = [
                            s for s in per_sound
                            if isinstance(s, dict) and s.get("error")
                        ]
                        if failed:
                            yield self._record(AgentEvent(
                                "info",
                                f"sound gen: {len(failed)}/{len(sound_specs)} "
                                f"failed — see per-sound reasons below"
                            ))
                            for s in failed:
                                name = s.get("name", "?")
                                err = s.get("error", "(no reason captured)")
                                err_line = str(err)[:400]
                                yield self._record(AgentEvent(
                                    "info",
                                    f"  - {name}: {err_line}"
                                ))

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
                # First build = plan-stage equivalent: model is choosing
                # the implementation shape, broad context helps.
                pb_block = self._retrieve_playbook_block(
                    goal, code=seed_html, stage="plan",
                )
                pb_kwargs = {"playbook_block": pb_block} if pb_block else {}
                build_msg = self._p.seed_build_instruction(
                    seed_html, str(self.seed_file), **pb_kwargs,
                )
                asset_block = render_asset_paths_block(
                    self._session_assets, self.out_path,
                )
                sound_block = render_sound_paths_block(
                    self._session_sounds, self.out_path,
                    looping_names=self._session_looping,
                )
                prelude = "\n\n".join(b for b in (asset_block, sound_block) if b)
                if prelude:
                    build_msg = prelude + "\n\n" + build_msg
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
                pb_block = self._retrieve_playbook_block(
                    goal, code=skel.html, stage="plan",
                )
                pb_kwargs = {"playbook_block": pb_block} if pb_block else {}

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
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": "streaming architect note"},
                    ))
                    try:
                        arch_reply = await self._stream(self._token_cb_wrapper)
                    except Exception as e:
                        yield self._record(AgentEvent("activity", "idle"))
                        yield self._record(AgentEvent(
                            "info", f"architect call failed, continuing single-shot: {e}",
                        ))
                        # Pop the architect user message so we don't leave
                        # the conversation in a half-state.
                        if self._messages and self._messages[-1].get("role") == "user":
                            self._messages.pop()
                    else:
                        yield self._record(AgentEvent("activity", "idle"))
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
                asset_block = render_asset_paths_block(
                    self._session_assets, self.out_path,
                )
                sound_block = render_sound_paths_block(
                    self._session_sounds, self.out_path,
                    looping_names=self._session_looping,
                )
                prelude = "\n\n".join(b for b in (asset_block, sound_block) if b)
                if prelude:
                    build_msg = prelude + "\n\n" + build_msg
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
            # User hard-stop: Ctrl-D in the TUI sets _user_force_done. Honor
            # it at the top of every iteration so the agent never starts a
            # new stream after the user asked to stop, even when probes
            # haven't passed. Whatever's in best.html (or out_path) ships.
            if self._user_force_done:
                yield self._record(AgentEvent(
                    "info", "user requested ship - exiting iteration loop",
                ))
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
            if self._step_mode and iteration > start_iter:
                yield self._record(AgentEvent(
                    "await_user",
                    f"step-mode: iter {iteration - 1} complete — "
                    "Enter to continue, or type feedback",
                    {"just_finished_iter": iteration - 1},
                ))
                while not self._step_continue and not self.has_pending_user_input():
                    await asyncio.sleep(0.1)
                if self.has_pending_user_input():
                    if self._messages and self._messages[-1].get("role") == "user":
                        base = self._messages.pop()["content"]
                        self._messages.append({
                            "role": "user",
                            "content": self._flush_user_injections(base),
                        })
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
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": f"streaming iter {iteration} reply"},
                    ))
                    reply = await self._stream(
                        self._token_cb_wrapper, prefill=reply_prefill,
                    )
                    yield self._record(AgentEvent("activity", "idle"))
            except Exception as e:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent(
                    "error",
                    f"{self._backend.info.name.upper()} call failed: {e}",
                ))
                return

            self._messages.append({"role": "assistant", "content": reply})
            self._extract_and_queue_lookups(reply)
            self._dump_conversation()
            self._trace({
                "kind": "assistant_reply",
                "iteration": iteration,
                "len": len(reply),
                "preview": reply[:600],
            })

            # ---- coverage-gap probe re-parse ---------------------------
            # If planning detected uncovered criteria (the synthetic
            # `coverage_gap__*` probes are still failing), and the model
            # included a new <probes> block in this reply, swap it in
            # before we run the test. Probes are otherwise immutable
            # after Phase A — this is the one legitimate mid-session
            # path, gated on an unresolved coverage gap so the model
            # can't churn probes turn-over-turn.
            if self._planning_coverage_gaps and "<probes>" in reply.lower():
                new_probes = self._extract_probes(reply)
                if new_probes:
                    from tools import _criteria_coverage_gaps as _gaps_fn
                    new_gaps = _gaps_fn(self._criteria or "", new_probes)
                    self._trace({
                        "kind": "probes_reparsed",
                        "iteration": iteration,
                        "old_count": len(self._probes),
                        "new_count": len(new_probes),
                        "remaining_gaps": new_gaps[:6],
                    })
                    yield self._record(AgentEvent(
                        "info",
                        f"probes re-emitted ({len(self._probes)} → "
                        f"{len(new_probes)}); remaining coverage gaps: "
                        f"{len(new_gaps)}",
                    ))
                    self._probes = new_probes
                    self._planning_coverage_gaps = new_gaps[:6]

            # ---- diagnose extraction (logged + memory-keyed) -----------
            diag = self._extract_diagnose(reply)
            if diag:
                self._last_diagnose = diag
                yield self._record(AgentEvent("diagnose", diag))
                # Shotgun-shape detector: flag when the model emitted a
                # ranked-hypothesis list. We don't reject the turn (the
                # patch may still be good); we just trace the violation
                # so the offline learner can credit/blame this pattern,
                # and surface an info event so the user sees it too.
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
            # A2: <done/> needs a clean-streak of N iters (default 2).
            # awaiting_confirm bypasses the streak — the post-critique
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
                awaiting_confirm
                or (self._previous_report_ok is True and streak_ok)
                or single_clean_ship_ok
            )
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
                    awaiting_confirm = False
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            "You were about to ship, but the user just sent the "
                            "feedback above. Address it now and re-send a fix as <patch>."
                        ),
                    })
                    continue
                if single_clean_ship_ok and not (awaiting_confirm or streak_ok):
                    reason = (
                        "Model declared done after a clean iter with "
                        "covered criteria, no page errors, all probes passed."
                    )
                else:
                    reason = (
                        "Model confirmed after self-critique."
                        if awaiting_confirm
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
                and not awaiting_confirm
            ):
                yield self._record(AgentEvent("phase", "self-critique"))
                awaiting_confirm = True
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        self._p.CRITIQUE_INSTRUCTION
                    ),
                })
                self._fix_mode = False
                continue

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
            # 2.3: per-iter HTML sha256 so test events can be correlated
            # back to the exact code that produced them. Iter snapshots
            # share this hash with their .html sibling on disk.
            import hashlib as _hashlib
            html_sha = _hashlib.sha256(
                new_html.encode("utf-8", "replace")
            ).hexdigest()[:16]
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

            # ---- pre-Chromium micro-probes (OpenCoder #4) ---------------
            # Cheap structural sanity check before paying the ~3s Chromium
            # round-trip. Catches truncation, empty scripts, badly-
            # unbalanced braces, and elision markers. Only ERRORS skip
            # the browser; warnings pass through and Chromium gets the
            # final word.
            mp = run_micro_probes(new_html)
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
                self._fix_mode = True
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(next_user),
                })
                self._previous_report_ok = False
                self._previous_report = fake_report  # todo #3 — full report
                continue

            # ---- run the test ------------------------------------------
            shot_before_path = (
                snap_path.with_name(snap_path.stem + "_before.png")
                if (snap_path and self._use_double_screenshot)
                else None
            )
            yield self._record(AgentEvent(
                "activity", "browser",
                {"label": f"loading iter {iteration} in Chromium"},
            ))
            try:
                report = await self.browser.load_and_test(
                    self.out_path, screenshot_path=shot_path,
                    screenshot_before_path=shot_before_path,
                    probes=self._probes or None,
                    # todo #2: pass criteria so the harness can flag
                    # coverage gaps as a soft_warning (forces the model
                    # to add probes that actually test what it promised).
                    criteria=self._criteria or None,
                )
            except Exception as e:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent("info", f"browser harness crashed: {e}"))
                self._messages.append({
                    "role": "user",
                    "content": self._flush_user_injections(
                        f"The browser test harness itself crashed: {e}\n"
                        "Please simplify the page and try again."
                    ),
                })
                continue

            yield self._record(AgentEvent("activity", "idle"))
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
                        self._messages.append({
                            "role": "user",
                            "content": self._flush_user_injections(
                                "REGRESSION DETECTED: your last change degraded the "
                                f"working build ({problems_str}). The harness has "
                                "auto-reverted the file on disk to the previous "
                                "working version. Send a MINIMAL <patch> that "
                                "addresses only the original feedback without "
                                "breaking what already worked. If you cannot make a "
                                "small change without regressing, send <done/> to "
                                "ship the working version as-is."
                            ),
                        })
                        self._fix_mode = True
                        # Skip streak/playbook/save_best below so revert state
                        # is identical to "this iter didn't happen for streak
                        # purposes". _previous_report* stays unchanged.
                        continue

            said_done = bool(_DONE_RE.search(reply))
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Track stuck-streak — used by v1's fix prompt to switch to
            # the "5-7 different sources" reflection ladder after repeat
            # failures on the same goal.
            if report["ok"]:
                self._stuck_streak = 0
                self._consecutive_clean_iters += 1
            else:
                self._stuck_streak += 1
                self._consecutive_clean_iters = 0
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
                # Hard stop: probes failed but the user asked to ship.
                # Don't loop another fix turn — exit with whatever we have.
                # best.html is shipped if it exists (a prior iter passed);
                # otherwise the current new_html is on disk at out_path.
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
                self._previous_report = report  # todo #3 — full report
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
            self._previous_report = report  # todo #3 — full report

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
                {"label": "streaming bonus turn (cap reached)"},
            ))
            try:
                reply = await self._stream(self._token_cb_wrapper)
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({"role": "assistant", "content": reply})
                self._extract_and_queue_lookups(reply)
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
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent("error", f"Final feedback turn failed: {e}"))

        yield self._record(AgentEvent(
            "info", f"reached max iterations ({self.max_iters}) - stopping"
        ))
        # Outcome: ok if best.html exists (we passed at least once).
        self._record_session_outcome(ok=self.best_path.exists())
        yield self._record(AgentEvent("done", "Iteration cap reached."))

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
                self._trace({"kind": "restart_snapshot_failed", "err": str(e)})
                attempts.append((score, k, canonical_best))
            self._trace({
                "kind": "restart_attempt_end",
                "attempt_idx": k,
                "score": score,
            })
            yield self._record(AgentEvent(
                "info",
                f"restart attempt {k+1}/{self.restart_n}: score={score:.0f}",
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
            self._trace({"kind": "restart_install_failed", "err": str(e)})
        yield self._record(AgentEvent(
            "info",
            f"restart winner: attempt {best_idx+1} score={best_score:.0f}",
        ))

    def _reset_attempt_state(self) -> None:
        """Reset the per-attempt mutable state so a fresh restart begins
        from a clean slate. Keeps cross-attempt resources (browser,
        backend, memory, playbook, generated assets/sounds cache).
        """
        self._messages = []
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
        self._user_force_done = False
        if self._stop_event is not None:
            self._stop_event.clear()
        self._step_continue = False
        self._last_screenshot_before = None
        self._last_screenshot_after = None
        self._active_bullet_ids = []

    def _score_attempt(self) -> float:
        """Score the just-finished attempt. Reuses tools.score_test_report
        on the most recent test report; falls back to 0 when nothing
        ran (e.g. crash before iter 1 produced a report)."""
        if self._previous_report_ok is True:
            return 100.0
        return score_test_report(self._previous_report or {})

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
        # Fix-turn = code stage (narrow): only validated patterns,
        # tighter char budget, no net-harmful bullets.
        pb_block = self._retrieve_playbook_block(
            self._goal, code=self._current_file, stage="code",
        )
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
        fix = self._p.fix_instruction(
            report_text, self._current_file, hints, **fix_kwargs,
        )
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
        # Close the playbook learning loop. Traces sit in trace_path.parent;
        # learner.py reads them and merges deltas into playbook.jsonl so
        # retrieval actually compounds across sessions. Default-on; opt
        # out via LEARNER_AUTO_APPLY=0 (e.g. tune.py runs that don't want
        # to mutate the shared playbook).
        if os.environ.get("LEARNER_AUTO_APPLY", "1").lower() not in ("0", "false", "no"):
            self._auto_apply_learner()

    def _auto_apply_learner(self) -> None:
        try:
            import subprocess
            import sys
            traces_dir = self.trace_path.parent
            learner_script = Path(__file__).parent / "learner.py"
            if not learner_script.exists():
                return
            proc = subprocess.run(
                [sys.executable, str(learner_script), "apply", str(traces_dir),
                 "--tests", self._session_id],
                capture_output=True, text=True, timeout=300, check=False,
            )
            self._trace({
                "kind": "learner_auto_apply",
                "rc": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-400:],
                "stderr_tail": (proc.stderr or "")[-200:],
            })
        except Exception as e:
            self._trace({"kind": "learner_auto_apply_failed", "err": str(e)})

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
