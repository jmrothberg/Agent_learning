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


# Pi-mono pattern: read AGENTS.md / CLAUDE.md from the working tree at
# session start and append it as <project-context> in the system prompt.
# Lets a repo enforce house-style ("always vanilla JS, no React") once
# instead of re-saying it via feedback every session.
_PROJECT_CONFIG_FILES = ("AGENTS.md", "CLAUDE.md")
# Cap so a sprawling project README doesn't crowd out the rest of the
# system prompt. ~6KB ≈ 1500 tokens, still room for the goal + workflow.
_PROJECT_CONFIG_MAX_CHARS = 6000


def _png_dims(png_bytes: bytes) -> tuple[int, int] | None:
    """Read width/height from the PNG IHDR chunk without decoding the
    image. Returns None if the bytes aren't a valid PNG. Used by the
    image_attached trace event so the log says EXACTLY what dimensions
    of screenshot the model received."""
    if not png_bytes or len(png_bytes) < 24:
        return None
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        import struct
        w, h = struct.unpack(">II", png_bytes[16:24])
        return (int(w), int(h))
    except Exception:
        return None


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


# Asset/sound rehydration from a seed file. Lifts every
# "./<prefix>_assets/<name>.<ext>" and "./<prefix>_sounds/<name>.<ext>"
# reference out of the seed HTML and, if the file exists on disk next
# to the seed, returns a name→path mapping. Tolerates straight or
# escaped quotes and an optional leading "./". Designed to be robust to
# whatever path style the model used when it first wrote the file.
_SEED_ASSET_RE = re.compile(
    r"""['"](?:\./)?([A-Za-z0-9_\-]+)_assets/([A-Za-z0-9_\-]+\.(?:png|jpg|jpeg|webp|gif))['"]""",
    re.IGNORECASE,
)
_SEED_SOUND_RE = re.compile(
    r"""['"](?:\./)?([A-Za-z0-9_\-]+)_sounds/([A-Za-z0-9_\-]+\.(?:ogg|mp3|wav|m4a))['"]""",
    re.IGNORECASE,
)


_IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif"}
)
_SOUND_EXTS: frozenset[str] = frozenset(
    {".ogg", ".mp3", ".wav", ".m4a"}
)


def _scan_seed_media(
    seed_html: str, seed_path: Path,
) -> tuple[
    dict[str, Path], dict[str, Path], Path | None, Path | None,
]:
    """Discover the seed's full media roster — both files referenced
    in the HTML AND any other files sitting in the canonical media
    folders on disk.

    Why both: a prior session may have generated mario_idle, mario_jump,
    donkey_kong_idle, etc. into `<basename>_assets/`, but the current
    seed HTML only references hero.png + princess.png. If we only
    looked at HTML refs, the model would re-invent fresh names instead
    of reusing the rich existing art that's right there on disk. By
    listing the FOLDER too, the model sees the complete roster in the
    system summary and can choose to load whichever existing PNGs fit.

    Returns (asset_paths, sound_paths, assets_dir, sounds_dir):
      - asset_paths / sound_paths: name (filename stem) → absolute Path,
        union of HTML-referenced AND folder-resident files
      - assets_dir / sounds_dir: kept for backwards compat with callers
        that still inspect them; current agent code reads these values
        but doesn't act on them (the canonical dir is derived from
        _session_id instead).

    Pure function — no side effects on the agent.
    """
    seed_dir = seed_path.parent.resolve()
    asset_paths: dict[str, Path] = {}
    sound_paths: dict[str, Path] = {}
    asset_dirs: set[Path] = set()
    sound_dirs: set[Path] = set()

    # Pass 1 — HTML refs. These are what the seed code IS LOADING
    # right now; the model should treat them as "currently wired up".
    for m in _SEED_ASSET_RE.finditer(seed_html):
        prefix, fname = m.group(1), m.group(2)
        adir = seed_dir / f"{prefix}_assets"
        full = adir / fname
        if full.exists():
            asset_paths[Path(fname).stem] = full.resolve()
            asset_dirs.add(adir.resolve())
    for m in _SEED_SOUND_RE.finditer(seed_html):
        prefix, fname = m.group(1), m.group(2)
        sdir = seed_dir / f"{prefix}_sounds"
        full = sdir / fname
        if full.exists():
            sound_paths[Path(fname).stem] = full.resolve()
            sound_dirs.add(sdir.resolve())

    # Pass 2 — list the canonical media folders directly. The seed
    # path here is already the canonical `games/<basename>.html`
    # (chat.py resolves snapshots and .best.html before constructing
    # the agent), so `<basename>_assets/` and `<basename>_sounds/`
    # next to it are exactly the dirs we want to walk. Files not
    # referenced in the HTML still get added to the roster so the
    # model knows they're available to reload.
    canonical_assets = seed_dir / f"{seed_path.stem}_assets"
    canonical_sounds = seed_dir / f"{seed_path.stem}_sounds"
    if canonical_assets.is_dir():
        for f in canonical_assets.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in _IMAGE_EXTS:
                continue
            name = f.stem
            if name not in asset_paths:
                asset_paths[name] = f.resolve()
        if canonical_assets.exists():
            asset_dirs.add(canonical_assets.resolve())
    if canonical_sounds.is_dir():
        for f in canonical_sounds.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in _SOUND_EXTS:
                continue
            name = f.stem
            if name not in sound_paths:
                sound_paths[name] = f.resolve()
        if canonical_sounds.exists():
            sound_dirs.add(canonical_sounds.resolve())

    assets_dir = next(iter(asset_dirs)) if len(asset_dirs) == 1 else None
    sounds_dir = next(iter(sound_dirs)) if len(sound_dirs) == 1 else None
    return asset_paths, sound_paths, assets_dir, sounds_dir


def _strip_thinking(reply: str) -> str:
    """Drop everything up to and including the LAST `</think>` tag.

    Reasoning-mode models (Qwen3.6, DeepSeek-V3.x, etc.) stream their
    chain-of-thought first, terminated by `</think>`. The CoT may
    legitimately MENTION tag names in markdown backticks
    (`` `<assets>` ``, `` `<patch>` ``), and the greedy non-greedy tag
    regexes below would otherwise match from the first occurrence in
    the prose all the way to the real closing tag, capturing the
    thinking text as the body and breaking parse — observed in
    games/traces/game-of-space-invaders-with-gr_20260511_093225 where
    13 asset specs + 10 sound specs were silently dropped because the
    CoT mentioned `` `<assets>` `` in a checklist.

    Stripping at the LAST `</think>` is safe: if the model uses
    multiple think segments, the real answer follows the last one. If
    no `</think>` is present, return the reply unchanged.

    Mirrored in assets.py for the asset/sound parsers — those modules
    can't import agent.py without a cycle.
    """
    if not reply:
        return reply
    idx = reply.rfind("</think>")
    if idx < 0:
        return reply
    return reply[idx + len("</think>"):]


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
_TODOS_RE = re.compile(r"<todos>\s*(.*?)\s*</todos>", re.DOTALL | re.IGNORECASE)
_PLAN_OPEN_RE = re.compile(r"<plan\b", re.IGNORECASE)
# Phase-A signals re-emitted in a build/iter turn instead of code.
# Used in the no-usable-code fallback to route to probes-only / media-only
# coaching that names the shape back to the model (donkey-kong traces
# 20260516_124628 iter 1, 20260516_142445 iter 1).
_PROBES_OPEN_RE = re.compile(r"<probes\b", re.IGNORECASE)
_ASSETS_OPEN_RE = re.compile(r"<assets\b", re.IGNORECASE)
_SOUNDS_OPEN_RE = re.compile(r"<sounds\b", re.IGNORECASE)
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


def _truncation_reason(html: str) -> str | None:
    """Return a short human description if `html` looks structurally
    truncated (an open tag with no close), else None.

    Generic across models: the classic-doom 20260512_153449 trace
    showed DeepSeek-V4 emit a stray ``` markdown fence inside an
    `<html_file>` body and stop before the closing tags. Smaller
    models hit the same shape when they run out of tokens. Either way
    the resulting file has open-but-not-closed `<html>` / `<body>` /
    `<script>` and the next fix turn should NOT inline it as a patch
    truth source — there's nothing to anchor patches against.
    """
    if not html:
        return None
    low = html.lower()
    # Single-tag opens (no attribute closing-bracket needed for opener
    # detection beyond the `<tag` prefix). Each pair is (opener-needle,
    # closer-needle, reason).
    for opener, closer, reason in (
        ("<html", "</html>", "unclosed <html>"),
        ("<body", "</body>", "unclosed <body>"),
        ("<script", "</script>", "unclosed <script>"),
    ):
        if opener in low and closer not in low:
            return reason
    return None


def _is_degenerate_baseline(html: str) -> bool:
    """True when `html` looks like a placeholder skeleton rather than a
    real game: too small, OR no <canvas>, OR no <script>, OR a <script>
    body containing only comments / placeholder markers, OR structurally
    truncated (open tag with no matching close).

    Used by `_materialize` to allow a full <html_file> rewrite when the
    on-disk baseline was the truncated output of a prior iter that hit
    the model's max_tokens cap (classic-doom trace 20260512_101944) OR
    that ended on a stray markdown fence inside the `<html_file>` body
    (classic-doom trace 20260512_153449). Without this carve-out the
    agent forces patch-mode against a file that has no real code to
    anchor to, and every subsequent iter is wasted.
    """
    if not html or len(html) < 2048:
        return True
    if _truncation_reason(html) is not None:
        return True
    low = html.lower()
    if "<canvas" not in low or "<script" not in low:
        return True
    # Extract first <script>...</script> body and check whether it
    # contains real code or just placeholder comments.
    m = re.search(r"<script\b[^>]*>(.*?)</script\s*>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return True
    body = m.group(1)
    # Strip JS comments (line + block) and whitespace.
    stripped = re.sub(r"//[^\n]*", "", body)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()
    # A real game's <script> body is ≥ 2 KB after stripping comments.
    # The DOOM iter 1 placeholder stripped to ~0 bytes.
    return len(stripped) < 512


def _detect_skeleton_payload(html: str) -> str | None:
    """Detect a `<html_file>` body that's a pseudocode placeholder, not
    real game code. Returns a human-readable reason string when the
    body looks like a skeleton, or None when it looks legitimate.

    Trace evidence — `build-a-donkey-kong-clone-in-o_20260514_214747`
    iter 3 (turn 10): the model emitted an `<html_file>` whose JS
    body was almost entirely pseudocode comment-headers like::

        (function() {
          "use strict";
          // Canvas setup
          const cvs = document.getElementById("c");
          // ...
          // Asset loading
          const ASSETS = {};
          async function loadAssets() { ... }
          // Sound loading
          const SOUNDS = {};
          function loadSounds() { ... }
          // ...

    The placeholder `{ ... }` bodies parse as valid JS (an object with
    a single `...` would be a spread but here it's not even that — it
    just lands in the DOM as a 374-byte stub). The harness wrote that
    to disk as the baseline, and the next iter had nothing real to
    patch against. The model then hit the deliberation guard again
    trying to rebuild from scratch. This rejects at materialize time
    so the prior real baseline is preserved.

    Heuristic (all four must hold):
      1. Total HTML byte size below `_SKELETON_MAX_BYTES`.
      2. First `<script>` body, stripped of JS comments and
         whitespace, is shorter than `_SKELETON_MIN_BODY_BYTES`.
      3. The stripped body contains at most a tiny handful of
         function-like definitions (`function NAME(` or `const NAME = `
         lambda forms). Real games have many.
      4. The stripped body contains a placeholder ellipsis (` { ... }`
         or `// ...`) that's the model's "fill this in later" marker.

    The combination of #1 + #2 + #3 is the strong signal; #4 is a
    safety check so we don't false-positive on a tiny legitimate game
    (e.g. a one-liner toy). All four make a false positive on real
    code virtually impossible.

    Sibling of `_is_degenerate_baseline` — that function tests EXISTING
    on-disk files to allow rewrite carve-outs; this one tests an
    INCOMING `<html_file>` body to reject the write.
    """
    if not html:
        return None
    if len(html) > _SKELETON_MAX_BYTES:
        return None  # Too big to be a skeleton.
    m = re.search(r"<script\b[^>]*>(.*?)</script\s*>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        # No <script> = either DOM-only app (legitimate; skip) or
        # malformed (other detectors catch it).
        return None
    body = m.group(1)
    # Strip JS comments (line + block) and condense whitespace so we
    # measure the volume of REAL CODE, not the model's commentary.
    stripped = re.sub(r"//[^\n]*", "", body)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if len(stripped) >= _SKELETON_MIN_BODY_BYTES:
        return None  # Real code present.
    # Count function-shape definitions in the stripped body.
    fn_count = len(re.findall(
        r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(", stripped,
    )) + len(re.findall(
        r"\b(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?"
        r"(?:function|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
        stripped,
    ))
    if fn_count > 2:
        return None  # Real game has multiple functions; skip.
    # Placeholder ellipsis markers — the model literally wrote "..."
    # where code should be. Check the ORIGINAL body (not stripped)
    # because the markers often live inside `{ ... }` placeholders.
    placeholder = bool(
        re.search(r"\{\s*\.\.\.\s*\}", body)
        or re.search(r"//\s*\.\.\.", body)
        or re.search(r"//\s*(?:rest|existing|TODO|stub|placeholder)", body, re.I)
    )
    if not placeholder:
        return None
    # All four conditions held → skeleton.
    return (
        f"<script> body is {len(stripped)} chars of code after "
        f"stripping comments, with only {fn_count} function "
        "definition(s) and `{ ... }` / `// ...` placeholder markers — "
        "looks like a pseudocode outline, not a real game"
    )


# Skeleton-payload thresholds (Item 1, plan 20260514). Pulled out as
# named constants so the regression test pins them and a future
# refactor doesn't drift them silently.
_SKELETON_MAX_BYTES = 4_000        # body cap; 374-byte DK trace fits easily
_SKELETON_MIN_BODY_BYTES = 800     # post-comment-strip JS volume below this is suspicious


def _looks_like_placeholder_html_payload(html: str) -> bool:
    """True when extracted `<html_file>` body is a tiny placeholder.

    Guards against format-doctor fallback outputs like `...` being treated as
    a valid full rewrite, which then writes a 3-byte baseline and burns the
    next iterations on recovery.
    """
    if not html:
        return True
    body = html.strip()
    if body in {"...", "…"}:
        return True
    has_html_marker = ("<html" in body.lower()) or ("<!doctype" in body.lower())
    if not has_html_marker and len(body) < 256:
        return True
    return False


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

# Only elide genuinely large inline HTML blobs during message compaction.
# Small examples (e.g. `<html_file>...</html_file>` in instructions) must
# remain verbatim or we mutate the semantics of prior user guidance.
_SUMMARIZE_MIN_HTML_BYTES = 1024


# Centipede trace 20260512_180020: user typed
#   "only change the centipiede_tail no other asset or code,
#    just that one asset no changes to the code"
# and the model replied "I can't generate new image assets in this
# environment - I can only modify the HTML file" — then rewrote a
# drawSprite() call into procedural ctx.* code (regression).
# The agent already supports mid-session asset re-render
# (_maybe_generate_assets_and_sounds); the model just didn't know it.
# These detectors light up the right path when art-change feedback
# arrives: inject a directive pointing at <assets>, and DON'T arm the
# one-shot <html_file> rewrite exemption when the user explicitly
# locked the code (that exemption fueled the second-attempt regression
# where the model "fixed" the asset by clobbering the sprite call).
_CODE_LOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "no changes to the code", "no change to code"
    re.compile(r"\bno\s+changes?\s+(?:to\s+)?(?:the\s+)?code\b", re.I),
    # "no code changes"
    re.compile(r"\bno\s+code\s+(?:change|edit|modification)s?\b", re.I),
    # "no other code", "no other asset or code"
    re.compile(r"\bno\s+other\s+\w+(?:\s+or\s+code)?\b.*\bcode\b", re.I),
    # "don't change/touch/modify the code", "do not change code"
    re.compile(
        r"\b(?:don['’]?t|do\s+not)\s+(?:change|touch|modify|edit)"
        r"\s+(?:the\s+|any\s+)?code\b",
        re.I,
    ),
    # "without changing/touching/modifying the code"
    re.compile(
        r"\bwithout\s+(?:changing|touching|modifying|editing)\s+"
        r"(?:the\s+)?code\b",
        re.I,
    ),
    # "only/just (the/that/this) (one) asset|sprite|image|art|png"
    re.compile(
        r"\b(?:only|just)\s+(?:the\s+|that\s+|this\s+)?(?:one\s+)?"
        r"(?:asset|sprite|image|art|png|graphic|picture|icon)s?\b",
        re.I,
    ),
    # Trace 20260514_175012 fix: user said "this is trivial no other
    # changes there" but the existing patterns all required the
    # literal word "code" or specific media nouns. None matched, the
    # rewrite exemption armed, and the model emitted a full <html_file>
    # rewrite instead of the minimal swap the user asked for. The
    # patterns below catch common minimal-scope phrasings without
    # requiring "code" or a media noun.
    #
    # "no other changes" / "no other changes there/please/needed"
    re.compile(r"\bno\s+other\s+changes?\b", re.I),
    # "this is trivial" / "trivial change|fix|swap" — minimal-scope intent
    re.compile(
        r"\btrivial\b.*\b(?:change|fix|swap|edit|update|tweak)\b", re.I,
    ),
    re.compile(
        r"\b(?:change|fix|swap|edit|update|tweak)\b.*\btrivial\b", re.I,
    ),
    # "just swap|switch|rename|move|flip|toggle" — narrow-verb scope-lock
    re.compile(
        r"\bjust\s+(?:swap|switch|rename|move|flip|toggle|rewire|"
        r"rebind|reassign)\b",
        re.I,
    ),
    # "only swap|switch|rename|move|flip|toggle"
    re.compile(
        r"\bonly\s+(?:swap|switch|rename|move|flip|toggle|rewire|"
        r"rebind|reassign)\b",
        re.I,
    ),
    # "leave the rest (alone|as-is)" — explicit out-of-scope signal
    re.compile(r"\bleave\s+(?:the\s+)?rest\b", re.I),
    # "nothing else (changes|to change|to fix)"
    re.compile(r"\bnothing\s+else\b", re.I),
)

_MEDIA_VERBS: tuple[str, ...] = (
    "change", "swap", "replace", "redraw", "regenerate", "remake",
    "redo", "redesign", "update", "make", "render", "rerender",
    "rerecord", "rebuild",
    # Added 2026-05-15: user said "fix the images" / "fix the
    # animations" and got a CODE rewrite instead of sprite regen.
    # "fix" alone is too generic to route on, but the gate also
    # requires an art noun (see _feedback_is_art_change), so
    # "fix the keyboard handler" still correctly stays in code-fix
    # mode (no art noun → no MEDIA-CHANGE).
    "fix", "improve",
)
_ART_NOUNS: tuple[str, ...] = (
    "asset", "assets", "sprite", "sprites", "image", "images",
    "png", "art", "graphic", "graphics", "picture", "pictures",
    "icon", "icons", "appearance",
    # Added 2026-05-15: in user vocabulary, "animation" / "animations"
    # = "the moving picture on screen" = the sprite. Missing this
    # caused MEDIA-CHANGE to never fire for "replace the animations"
    # / "fix the annimations", routing the request to <patch> (code
    # change) when the user explicitly wanted sprite regeneration.
    # "annimation" with the extra 'n' is the user's persistent typo —
    # listed so detection is robust to that misspelling. "frame" /
    # "frames" cover "redo the run frames" / "the walk frames look
    # wrong" phrasings; behavior_bug still suppresses MEDIA-CHANGE
    # for phrases like "frames are stuttering" because that hits the
    # bug-complaint regex first.
    "animation", "animations",
    "annimation", "annimations",
    "anim", "anims",
    "frame", "frames",
    "spritesheet", "spritesheets",
)
_SOUND_NOUNS: tuple[str, ...] = (
    "sound", "sounds", "audio", "sfx", "music", "song",
    "beep", "chime", "sample", "clip", "track", "ogg", "tune",
    "soundtrack",
)


def _feedback_locks_code(text: str) -> bool:
    """User explicitly forbade code changes for this turn.

    Matches phrases like "no changes to the code", "only the asset",
    "just that one sprite", "don't touch the code". Used to suppress
    the one-shot <html_file> rewrite exemption that fresh feedback
    would otherwise arm — when the user locked the code, the rewrite
    license is exactly what we DON'T want.
    """
    return any(p.search(text) for p in _CODE_LOCK_PATTERNS)


# Behavior verbs — gameplay actions a player or game object performs.
# Used to detect "user is reporting a behavior bug" patterns like
# "mario doesn't climb" / "barrels don't roll" / "nothing happens when
# I press space". When fired, the MEDIA-CHANGE DIRECTIVE is suppressed
# even if an asset name appears in the feedback (DK trace
# 20260514_104131 burned 7 consecutive turns because "mario" + "ladder"
# matched the art-change classifier and the model was told the
# feedback was about ART/SOUND when the user was clearly reporting a
# code bug).
#
# Genre-free / game-agnostic: every entry describes input or state
# transitions, not subject matter. Visual verbs (look, appear, render,
# show) are intentionally EXCLUDED so "the dragon doesn't look right"
# still routes to the art-change path.
_BEHAVIOR_VERBS: tuple[str, ...] = (
    "climb", "climbs", "climbing", "climbed",
    "move", "moves", "moving", "moved",
    "jump", "jumps", "jumping", "jumped",
    "run", "runs", "running",
    "walk", "walks", "walking", "walked",
    "fire", "fires", "firing", "fired",
    "shoot", "shoots", "shooting", "shot",
    "work", "works", "working", "worked",
    "respond", "responds", "responding", "responded",
    "react", "reacts", "reacting", "reacted",
    "fall", "falls", "falling", "fell",
    "fly", "flies", "flying", "flew",
    "spawn", "spawns", "spawning", "spawned",
    "reset", "resets", "resetting",
    "restart", "restarts", "restarting", "restarted",
    "trigger", "triggers", "triggering", "triggered",
    "hit", "hits", "hitting",
    "register", "registers", "registering", "registered",
    "roll", "rolls", "rolling", "rolled",
    "drop", "drops", "dropping", "dropped",
    "happen", "happens", "happening", "happened",
    "play", "plays", "playing", "played",
    "load", "loads", "loading", "loaded",
    "update", "updates", "updating", "updated",
    "collide", "collides", "colliding", "collided",
    "die", "dies", "dying", "died",
    "score", "scores", "scoring", "scored",
)
_BEHAVIOR_VERB_ALT = "|".join(re.escape(v) for v in _BEHAVIOR_VERBS)
_BEHAVIOR_BUG_NEGATION_RE = re.compile(
    r"\b(?:doesn['’]?t|don['’]?t|didn['’]?t|can['’]?t|cannot|"
    r"won['’]?t|wouldn['’]?t|isn['’]?t|aren['’]?t|never|"
    r"nothing|unable\s+to|fails?\s+to|not)\s+"
    r"(?:\w+\s+){0,3}"
    rf"(?:{_BEHAVIOR_VERB_ALT})\b",
    re.IGNORECASE,
)
_BEHAVIOR_BUG_COMPLAINT_RE = re.compile(
    r"\b(?:bug|broken|crash(?:ing|ed|es)?|freez(?:ing|es)?|frozen|"
    r"stuck|hang(?:ing|s|ed)?|glitch(?:ing|ed|es)?)\b",
    re.IGNORECASE,
)


def _feedback_is_behavior_bug(text: str) -> bool:
    """User is reporting a behavior / code bug — "X doesn't Y",
    "nothing happens when …", "the game is broken / frozen / crashing".

    DK trace 20260514_104131 fix: the existing art-change classifier
    fires True whenever an asset name appears in feedback, which
    misroutes "mario does not climb the ladder" as an ART/SOUND change
    request (because "mario" and "ladder" are asset names). Detecting
    behavior-bug language lets the directive injector suppress the
    misrouting.

    Patterns matched:
      - negation + behavior verb within 3 words ("does not climb",
        "won't reset", "isn't responding")
      - explicit complaint nouns ("bug", "broken", "crashing",
        "frozen", "stuck", "glitching")

    Visual-only verbs (look, appear, render, show) are NOT in the
    behavior-verb set, so "the dragon doesn't look right" stays on
    the art-change path.
    """
    if not text:
        return False
    if _BEHAVIOR_BUG_COMPLAINT_RE.search(text):
        return True
    if _BEHAVIOR_BUG_NEGATION_RE.search(text):
        return True
    return False


def _name_in_text(text_lower: str, names: list[str]) -> bool:
    """Match a known asset/sound name in feedback, tolerating
    underscores ↔ hyphens ↔ spaces (so "centipede tail" matches
    "centipede_tail" and vice versa)."""
    canon_text = re.sub(r"[\s\-]+", "_", text_lower)
    for name in names:
        n = (name or "").strip().lower()
        if not n:
            continue
        canon_name = re.sub(r"[\s\-]+", "_", n)
        if canon_name and canon_name in canon_text:
            return True
    return False


def _feedback_is_art_change(text: str, asset_names: list[str]) -> bool:
    """User feedback is asking to change visual art.

    Heuristic: a known asset name appears in the text, OR text contains
    an art-noun ("sprite", "image", "art", …) together with a media
    verb ("change", "redraw", "make", …). Misses are safe (no directive
    injected, regular flow proceeds). False positives are also safe —
    the directive only advises the model to prefer <assets>.
    """
    lo = text.lower()
    if _name_in_text(lo, asset_names):
        return True
    has_noun = any(re.search(rf"\b{re.escape(n)}\b", lo) for n in _ART_NOUNS)
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo) for v in _MEDIA_VERBS
    )
    return has_noun and has_verb


def _feedback_is_sound_change(text: str, sound_names: list[str]) -> bool:
    """Same shape as `_feedback_is_art_change`, for `<sounds>`."""
    lo = text.lower()
    if _name_in_text(lo, sound_names):
        return True
    has_noun = any(
        re.search(rf"\b{re.escape(n)}\b", lo) for n in _SOUND_NOUNS
    )
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo) for v in _MEDIA_VERBS
    )
    return has_noun and has_verb


# ---------------------------------------------------------------------------
# Subsystem hints — map mistake_signature shapes to a code region the model
# should look at. Used by (a) the coaching message at _repeat_sig_streak >= 2
# and (b) the focused-slice keyset biaser. DK trace 20260514_104131 evidence:
# 100% of recent sessions had signatures containing "INPUT_DEAD" / "Controls
# are not wired up" AND the model's patches didn't touch any addEventListener
# / keydown code — the verifier signal pointed to the input layer but the
# model kept patching the higher-level mechanic the user named (climb math,
# barrel physics, etc.). Existing coaching at _repeat_sig_streak >= 2 said
# "AUTHOR a runtime-state probe" — too abstract for a 27B model already
# focused on the wrong area.
#
# Genre-free / model-agnostic: each entry describes the SHAPE of a failure
# (input wiring, frame/draw loop, RAF kick-off), not subject matter.
# Asset-load is intentionally OMITTED here — the existing asset-specific
# coaching branch handles that case (DK 20260513_122154 fix).
# ---------------------------------------------------------------------------

# (signature_substrings_lower, name, identifier_tokens, fix_phrase)
_SUBSYSTEM_HINTS: tuple[
    tuple[tuple[str, ...], str, tuple[str, ...], str], ...
] = (
    (
        (
            "input_dead",
            "controls are not wired up",
            "controls not wired",
            "input handler is broken",
        ),
        "input",
        (
            "addEventListener", "keydown", "keyup", "KeyboardEvent",
            "code", "KEYMAP", "keys",
        ),
        "the keydown/keyup handler (e.g. `window.addEventListener("
        "'keydown', ...)` and the key-state map)",
    ),
    (
        (
            "frozen",
            "did not change between two samples",
            "canvas drew something but did not change",
        ),
        "draw_or_raf",
        (
            "requestAnimationFrame", "frame", "render", "draw", "ctx",
        ),
        "the frame/draw loop (the function called by "
        "`requestAnimationFrame`)",
    ),
    (
        (
            "canvas pixels are uniform",
            "not rendering",
        ),
        "raf_start",
        (
            "requestAnimationFrame", "loadAssets", "then",
            "startGame", "init",
        ),
        "the RAF kick-off (e.g. `loadAssets().then(() => "
        "requestAnimationFrame(frame))`)",
    ),
)


def _subsystem_hint(signature: str) -> dict | None:
    """Map a mistake_signature to a structured subsystem hint.

    Returns dict {name, identifiers, fix_phrase} or None when no
    entry matches. Callers use:
      - `identifiers` to bias _focused_slice's keyset.
      - `fix_phrase` to build a directive coaching message that names
        the specific subsystem to patch this turn.
    """
    if not signature:
        return None
    low = signature.lower()
    for substrings, name, identifiers, fix_phrase in _SUBSYSTEM_HINTS:
        if any(sub in low for sub in substrings):
            return {
                "name": name,
                "identifiers": identifiers,
                "fix_phrase": fix_phrase,
            }
    return None


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
        # num_ctx — 256K on a 27B model is ~10 GB of VRAM, which is
        # fine on a 64+ GB Mac but tight on a 32 GB laptop. Override
        # downward via the CODING_BOX_NUM_CTX env var if you hit OOM.
        # Note: changing num_ctx between calls forces an Ollama model
        # reload — to avoid that, preload at the desired size with
        # `ollama run --ctx-size 262144 <model>` before starting.
        num_ctx: int = 262144,
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
        # Set by `_flush_user_injections` when the user locks the turn
        # ("no code changes", "only X"). The fix-mode prompt reads this
        # and suppresses the failing-test "fix these" framing so the
        # local model isn't asked to balance "ignore test failures" vs.
        # "fix these test failures" in the same prompt (DK trace
        # 2026-05-15). Self-clears after each fix-prompt build.
        self._scoped_change_active: bool = False
        self._pending_answer: str | None = None
        # A2: short coaching strings queued when the deliberation detector
        # aborts the last stream or another agent-side guard wants the
        # next user-turn to carry a one-shot corrective note. Drained
        # by _flush_user_injections before any base message is appended.
        self._pending_coaching: list[str] = []
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
        self._skeleton_mode = skeleton_mode
        self._prompt_version = prompt_version
        self._p = self._load_prompt_module(prompt_version)
        self._last_diagnose: str | None = None
        self._stuck_streak: int = 0
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

    # Read-through to the resolved backend's model id. Existing call sites
    # (trace metadata, conversation dump, memory.record_outcome, ...) used
    # `self.model` as a string; keeping it as a property means the agent
    # always reports whatever the backend resolved to without callers
    # having to know about Backend internals.
    @property
    def model(self) -> str:
        return self._backend.info.model

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
            and rejection_kind == "unclosed_html_file"
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
        prior_loop_kind: str | None = None,
        prior_loop_line: str | None = None,
        is_local_backend: bool = False,
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
                "FIRST BUILD REQUIRED — your previous reply had no usable "
                "<html_file>/<patch>. Re-emit this turn as CODE ONLY.\n"
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
                stall_seconds=doctor_stall_seconds,
                overall_seconds=doctor_overall_seconds,
                max_retries=0,
                cancel_event=self._ensure_stop_event(),
            )
            text = result.text or ""
        except Exception as e:
            self._trace({"kind": "format_doctor_error", "err": str(e)[:200]})
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

    def set_auto_step_on_failure(self, on: bool) -> None:
        """Enable/disable auto step-mode arming on first failed iter."""
        self._auto_step_on_failure = bool(on)
        self._trace({
            "kind": "auto_step_on_failure_set",
            "on": self._auto_step_on_failure,
        })

    # -- asset-reference alignment scan ----------------------------------
    #
    # In the donkey-kong trace 20260513_122154 the model's Phase A
    # produced 8 sprites but the iter-1 HTML referenced 14 by name. The
    # 6 unbacked names produced net::ERR_FILE_NOT_FOUND on every load,
    # and the model spent iters 3+4 patching drawImage symptoms instead
    # of noticing the files weren't there. The scan compares names the
    # HTML references vs. names actually generated, so the harness can
    # tell the model exactly which assets need a mid-session regen
    # BEFORE the browser test wastes another iter on the same symptom.

    # Pattern catches:
    #   ASSETS['name']  /  ASSETS["name"]   (subscript)
    #   ASSETS.name                          (dot access)
    # Captures the identifier as group 1.
    _ASSET_SUBSCRIPT_RE = __import__("re").compile(
        r"""ASSETS\s*\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"""
    )
    _ASSET_DOT_RE = __import__("re").compile(
        r"""\bASSETS\.([A-Za-z_][A-Za-z0-9_]*)\b"""
    )
    # Loose path pattern: any '<anything>_assets/<name>.png' string
    # literal. Catches inlined paths even without ASSETS[] indirection.
    _ASSET_PATH_RE = __import__("re").compile(
        r"""['"][^'"\s]*_assets/([A-Za-z_][A-Za-z0-9_]*)\.png['"]"""
    )
    # Array-of-string-names commonly named `assetList`, `asset_names`,
    # `assetPaths`, `sprites`, etc., followed by a `[...]` literal of
    # bare string identifiers. The DK trace failure used exactly this
    # pattern — assetList = ['mario_idle', …] then mapped at runtime.
    _ASSET_LIST_RE = __import__("re").compile(
        r"""\b(?:assetList|asset_names|assetNames|spriteNames|spriteList)\s*=\s*\[([^\]]+)\]""",
        __import__("re").IGNORECASE,
    )
    _ASSET_LIST_NAME_RE = __import__("re").compile(
        r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""
    )
    # Sound alignment scan mirrors the asset scan above so missing OGG
    # references are surfaced before Chromium wastes an iteration on 404s.
    _SOUND_SUBSCRIPT_RE = __import__("re").compile(
        r"""SOUNDS\s*\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"""
    )
    _SOUND_DOT_RE = __import__("re").compile(
        r"""\bSOUNDS\.([A-Za-z_][A-Za-z0-9_]*)\b"""
    )
    _SOUND_PATH_RE = __import__("re").compile(
        r"""['"][^'"\s]*_sounds/([A-Za-z_][A-Za-z0-9_]*)\.(?:ogg|mp3|wav|m4a)['"]""",
        __import__("re").IGNORECASE,
    )
    _SOUND_LIST_RE = __import__("re").compile(
        r"""\b(?:soundNames|soundList|sfxNames|audioNames)\s*=\s*\[([^\]]+)\]""",
        __import__("re").IGNORECASE,
    )
    _SOUND_LIST_NAME_RE = __import__("re").compile(
        r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""
    )

    @classmethod
    def _scan_html_for_asset_refs(cls, html: str) -> set[str]:
        """Return the set of asset *names* the HTML references.

        Static analysis — fast, deterministic, no JS execution. Covers
        the three patterns the model produces in practice:
          1. ASSETS['name'] / ASSETS["name"] / ASSETS.name
          2. Literal paths '<dir>_assets/<name>.png'
          3. Array literals assigned to assetList / spriteNames / etc.
        """
        refs: set[str] = set()
        for m in cls._ASSET_SUBSCRIPT_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._ASSET_DOT_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._ASSET_PATH_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._ASSET_LIST_RE.finditer(html):
            for nm in cls._ASSET_LIST_NAME_RE.finditer(m.group(1)):
                refs.add(nm.group(1))
        return refs

    def _check_asset_alignment(self, html: str) -> set[str]:
        """Compare HTML asset references against generated files.

        Returns the set of names that are REFERENCED but not present in
        `self._session_assets`. When non-empty, queues a coaching
        message naming the missing files so the next user turn tells
        the model to either remove the references or emit an `<assets>`
        block to request the gap (which the existing mid-session regen
        pipeline will then fulfill).
        """
        refs = self._scan_html_for_asset_refs(html)
        if not refs:
            return set()
        available = set(self._session_assets.keys())
        missing = refs - available
        if not missing:
            return set()
        self._trace({
            "kind": "asset_alignment_gap",
            "referenced": sorted(refs),
            "available": sorted(available),
            "missing": sorted(missing),
        })
        miss_list = ", ".join(sorted(missing))
        avail_list = ", ".join(sorted(available)) or "(none)"
        self._pending_coaching.append(
            "Asset references don't match the files on disk. Your code "
            f"references these names that were never generated: {miss_list}. "
            f"Available assets: {avail_list}. The browser is returning "
            "net::ERR_FILE_NOT_FOUND for the missing ones, not a drawImage "
            "bug. To fix the root cause, EITHER (a) emit an `<assets>...</assets>` "
            "block in this turn requesting the missing names — the harness "
            "will regenerate them mid-session and your code will work as "
            "written — OR (b) edit the code to only reference assets that "
            "exist. Do NOT add more drawImage try/catch guards — those "
            "hide the load failure, they don't fix it."
        )
        return missing

    @classmethod
    def _scan_html_for_sound_refs(cls, html: str) -> set[str]:
        """Return the set of sound *names* the HTML references."""
        refs: set[str] = set()
        for m in cls._SOUND_SUBSCRIPT_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._SOUND_DOT_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._SOUND_PATH_RE.finditer(html):
            refs.add(m.group(1))
        for m in cls._SOUND_LIST_RE.finditer(html):
            for nm in cls._SOUND_LIST_NAME_RE.finditer(m.group(1)):
                refs.add(nm.group(1))
        return refs

    def _check_sound_alignment(self, html: str) -> set[str]:
        """Compare HTML sound references against generated files."""
        refs = self._scan_html_for_sound_refs(html)
        if not refs:
            return set()
        available = set(self._session_sounds.keys())
        missing = refs - available
        if not missing:
            return set()
        self._trace({
            "kind": "sound_alignment_gap",
            "referenced": sorted(refs),
            "available": sorted(available),
            "missing": sorted(missing),
        })
        miss_list = ", ".join(sorted(missing))
        avail_list = ", ".join(sorted(available)) or "(none)"
        self._pending_coaching.append(
            "Sound references don't match the files on disk. Your code "
            f"references these sound names that were never generated: {miss_list}. "
            f"Available sounds: {avail_list}. The browser is returning "
            "net::ERR_FILE_NOT_FOUND for missing OGGs, not an Audio.play() "
            "bug. To fix the root cause, EITHER (a) emit a `<sounds>...</sounds>` "
            "block in this turn requesting the missing names — the harness "
            "will regenerate them mid-session and your code will work as "
            "written — OR (b) edit the code to only reference sounds that "
            "exist. Do NOT add more try/catch around play(); that hides the "
            "load failure, it doesn't fix it."
        )
        return missing

    def _is_local_backend(self) -> bool:
        """True for local backends (MLX/Ollama)."""
        return self._backend.info.name in {"mlx", "ollama"}

    def _local_first_build_nudge(self) -> str:
        """Small local-only nudge to reduce long repetitive first builds."""
        if not self._is_local_backend():
            return ""
        n_assets = len(self._session_assets)
        n_sounds = len(self._session_sounds)
        if n_assets < 10 and n_sounds < 6:
            return ""
        return (
            "LOCAL MODEL SAFETY NUDGE: Keep first-build code compact to avoid "
            "token loops. Use short name arrays + loops for media loaders; do "
            "NOT hand-enumerate long repeated `[name, path]` blocks. Use ONLY "
            "sound/sprite names present in the GENERATED ASSETS/SOUNDS blocks "
            "above."
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
            )
            sig = tuple(snapshot.get(k) for k in sig_keys)
            if sig == getattr(self, "_last_status_sig", None):
                return
            self._last_status_sig = sig
            self._trace({"kind": "status_snapshot", **snapshot})
        except Exception:
            pass

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
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
            return f"<html_file>[omitted: {n} bytes of HTML; see snapshot]</html_file>"

        def fence_repl(m):
            n = len(m.group(1))
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
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
            # Do NOT rewrite user/system turns here. Mutating prior user
            # instructions (especially format examples) creates false context
            # and can derail one-shot generations.
            if msg.get("role") != "assistant":
                continue
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

            # Detect intent BEFORE clearing the queue so we can shape the
            # follow-up directives. Centipede trace 20260512_180020 is
            # the motivating case.
            asset_names = (
                list(self._session_assets.keys())
                if self._session_assets else []
            )
            sound_names = (
                list(self._session_sounds.keys())
                if self._session_sounds else []
            )
            locks_code = _feedback_locks_code(joined)
            art_change = bool(asset_names) and _feedback_is_art_change(
                joined, asset_names,
            )
            sound_change = bool(sound_names) and _feedback_is_sound_change(
                joined, sound_names,
            )
            behavior_bug = _feedback_is_behavior_bug(joined)

            self._pending_feedback.clear()

            # DK trace 20260514_104131 fix: when the user is clearly
            # reporting a behavior bug ("mario does not climb the
            # ladder"), suppress the MEDIA-CHANGE DIRECTIVE — that
            # directive tells the model "your feedback is about
            # ART/SOUND, not code" and fires the model into emitting
            # an <assets> re-render instead of fixing the code. The
            # asset-name match in `_feedback_is_art_change` is a
            # weak signal that needs an explicit suppressor when the
            # feedback contains behavior-bug language.
            if behavior_bug and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "behavior_bug",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })

            # MEDIA-CHANGE DIRECTIVE — fires only when the feedback
            # carries INDEPENDENT art/sound semantics (asset/sprite/sound
            # vocabulary). The old gate also fired on `locks_code` alone,
            # which mis-routed any "no code changes" feedback to the art
            # path even when the user really wanted a code-side tweak
            # (DK trace 2026-05-15 iter 3: "make 4x larger, no code
            # changes" routed to <assets> regeneration instead of a
            # drawImage size patch — same regenerated PNGs at the same
            # canvas size achieve nothing visible). The new SCOPED-CHANGE
            # block below handles the code-lock case directly.
            if (
                (asset_names or sound_names)
                and (art_change or sound_change)
                and not behavior_bug
            ):
                lines: list[str] = [
                    "================ MEDIA-CHANGE DIRECTIVE ================",
                    "The feedback above is about ART/SOUND, not code. The",
                    "harness can regenerate any sprite or sound in place:",
                    "emit a fresh block with the EXISTING name and a new",
                    "prompt — no JS edit needed. The existing drawSprite()",
                    "/ new Audio() call already in the file automatically",
                    "picks up the regenerated file.",
                ]
                if asset_names:
                    asset_list = ", ".join(sorted(asset_names))
                    lines.extend([
                        "",
                        "Sprites — use <assets> to re-render:",
                        "  <assets>[{\"name\":\"<existing_name>\","
                        "\"prompt\":\"<new visual prompt>\"}]</assets>",
                        f"  Existing asset names: {asset_list}",
                    ])
                if sound_names:
                    sound_list = ", ".join(sorted(sound_names))
                    lines.extend([
                        "",
                        "Sounds — use <sounds> to re-render:",
                        "  <sounds>[{\"name\":\"<existing_name>\","
                        "\"prompt\":\"<new audio prompt>\","
                        "\"duration\":1.0}]</sounds>",
                        f"  Existing sound names: {sound_list}",
                    ])
                lines.extend([
                    "",
                    "Do NOT swap an existing drawSprite(name,…) or",
                    "new Audio(path) call for inline procedural code here —",
                    "that loses the media path and regresses. If the user",
                    "truly asked for code changes too, address those with a",
                    "small <patch>.",
                    "========================================================",
                ])
                parts.append("\n".join(lines))
                self._trace({
                    "kind": "media_change_directive_injected",
                    "locks_code": locks_code,
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "asset_count": len(asset_names),
                    "sound_count": len(sound_names),
                })

            # SCOPED-CHANGE DIRECTIVE — fires whenever the user locked
            # the turn ("no code changes", "only X", "leave the rest").
            # Sits ABOVE rewrite-exemption gating and the downstream
            # fix-mode test-failure framing (which is also suppressed
            # for this turn — see `_scoped_change_active` flag below).
            #
            # Why it exists: 2026-05-15 DK trace iter 3. User said "make
            # 4x larger, DO NOT change other code, no code changes". The
            # agent dutifully added the user's text with HIGHEST PRIORITY
            # framing, then in the same prompt also injected the prior
            # iter's failing-probes block, the (mis-routed) MEDIA-CHANGE
            # DIRECTIVE, and standard fix-mode coaching. The 27B local
            # model tried to satisfy all four directives, shipped a 2x
            # scale + four unrelated "fixes", and the user typed back
            # "YOU DIDNT LISTEN". A frontier model can balance contra-
            # dictory prompts; a local model cannot. The fix is to stop
            # contradicting ourselves in the same prompt.
            if locks_code:
                scoped_lines = [
                    "================ SCOPED-CHANGE DIRECTIVE ================",
                    "The user scoped this turn — address ONLY what is in",
                    "the USER FEEDBACK block above.",
                    "",
                    "Routing the request:",
                    "  - If the user wants the SPRITES TO LOOK DIFFERENT",
                    "    (different style, pose, color, etc.) — emit",
                    "    <assets> ONLY with the existing names + a new",
                    "    prompt. No <patch>, no <html_file>.",
                    "  - If the user wants the SPRITES TO BE A DIFFERENT",
                    "    SIZE on screen ('4x larger', 'smaller', 'half'",
                    "    'half the size') — emit ONE small <patch> that",
                    "    ONLY changes the drawImage width and height",
                    "    arguments. Do not touch any other function,",
                    "    constant, or variable. No <assets> needed.",
                    "  - If the user wants new BEHAVIOR (different speed,",
                    "    new key binding, new rule) — emit ONE small",
                    "    <patch> that ONLY changes the relevant value or",
                    "    branch. Nothing else.",
                    "",
                    "HARD RULES for this turn:",
                    "  - Do NOT fix unrelated test failures from the",
                    "    previous iter. The user told you to ignore them",
                    "    this turn.",
                    "  - Do NOT clean up code you think looks suspicious.",
                    "  - Do NOT refactor, rename, or 'improve' anything",
                    "    not named by the user.",
                    "  - Do NOT emit a full <html_file>; use one tight",
                    "    <patch>.",
                    "  - If the user says 'Nx larger' or 'Nx smaller',",
                    "    the changed numbers in your <patch> must be",
                    "    EXACTLY that factor of the originals. Not",
                    "    'close enough', not 'approximately'.",
                    "  - If you cannot satisfy the user's request",
                    "    without a code change but they said 'no code",
                    "    changes', make the MINIMAL code edit that",
                    "    achieves the intent and nothing else.",
                    "=========================================================",
                ]
                parts.append("\n".join(scoped_lines))
                self._trace({
                    "kind": "scoped_change_directive_injected",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
                # Flag consumed by the fix-mode prompt builder to drop
                # the "fix these failing probes" framing for this turn.
                # Cleared after one fix-mode build cycle (caller resets).
                self._scoped_change_active = True

            # Rewrite-exemption gating — unchanged from before, just
            # moved below SCOPED-CHANGE so the order in the prompt is:
            # USER FEEDBACK → (MEDIA-CHANGE if applicable) → SCOPED-CHANGE
            # → other context. SUPPRESSED when the user locked the code,
            # because granting a full <html_file> rewrite while telling
            # the model "don't change other code" is the exact contra-
            # diction we're trying to eliminate.
            if locks_code:
                self._trace({
                    "kind": "rewrite_exemption_suppressed",
                    "reason": "code_locked",
                })
            elif self._repeat_sig_streak >= 2:
                # The model has been failing on the same error twice in
                # a row. Granting another full <html_file> rewrite when
                # surgical patches haven't fixed the root cause just
                # fuels regression (cf. DK trace 20260513_122154: 3
                # consecutive 22-25 KB rewrites all hit the same
                # ERR_FILE_NOT_FOUND). Force the model to send a
                # focused <patch> this turn so it has to articulate
                # what's actually different.
                self._trace({
                    "kind": "rewrite_exemption_suppressed",
                    "reason": "repeat_signature",
                    "streak": self._repeat_sig_streak,
                })
            else:
                self._allow_one_rewrite = True
                self._trace({"kind": "rewrite_exemption_armed"})
        # Drain any <lookup_bullet> resolutions queued by the previous
        # assistant reply. These come BEFORE the base message so the
        # model sees them as fresh material before the iteration prompt.
        if self._pending_bullet_lookups:
            for block in self._pending_bullet_lookups:
                parts.append(block)
            self._pending_bullet_lookups.clear()
        # A2/A5: agent-queued coaching lines (deliberation guard recovery,
        # repeat-error nudges). Rendered as a single high-priority block
        # so the model sees them before the base instruction.
        if self._pending_coaching:
            joined = "\n- ".join(self._pending_coaching)
            parts.append(
                "================ AGENT COACHING ================\n"
                f"- {joined}\n"
                "================================================"
            )
            for c in self._pending_coaching:
                self._trace({"kind": "coaching_injected", "text": c})
            self._pending_coaching.clear()
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
        if self._local_vlm_path is None:
            try:
                from backend import discover_local_vlm
                resolved = discover_local_vlm()
            except Exception:
                resolved = None
            # Sentinel "" = scanned, nothing found. Stops us rescanning.
            self._local_vlm_path = resolved or ""
            if resolved:
                self._trace({"kind": "vision_judge_local_vlm", "path": resolved})
        if not self._local_vlm_path:
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
        note = (verdict.note or "").strip()
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

    async def _stream(
        self, on_token, *,
        override_temp: float | None = None,
        prefill: str = "",
        prefill_force: bool = False,
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
        prefill_used = False
        prefill_enabled = bool(prefill) and (self._use_prefill or prefill_force)
        if prefill_enabled:
            self._messages.append({"role": "assistant", "content": prefill})
            prefill_used = True
            self._trace({
                "kind": "prefill",
                "len": len(prefill),
                "forced": bool(prefill_force and not self._use_prefill),
            })

        temp = override_temp if override_temp is not None else (
            0.25 if self._fix_mode else 0.7
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

        # Rich first-build turns (no baseline file yet) can be long on
        # local models; give them a larger overall wall-clock cap while
        # keeping normal patch turns at the configured default.
        effective_overall_seconds = self.overall_seconds
        if self._current_file is None:
            effective_overall_seconds = max(self.overall_seconds, 2400.0)
        if effective_overall_seconds != self.overall_seconds:
            self._trace({
                "kind": "stream_timeout_override",
                "base_overall_seconds": self.overall_seconds,
                "effective_overall_seconds": effective_overall_seconds,
                "reason": "no_baseline_file",
            })

        try:
            opts: dict[str, Any] = {"temperature": temp, "num_ctx": self.num_ctx}
            if self._restart_attempt_seed is not None:
                opts["seed"] = int(self._restart_attempt_seed)
            result = await self._backend.stream_chat(
                self._messages,
                on_token=_heartbeat_on_token,
                options=opts,
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
        finally:
            # Always remove our prefill scaffolding before returning so
            # the message history we save & feed to subsequent turns
            # contains a single coherent assistant message.
            if prefill_used and self._messages and self._messages[-1].get("role") == "assistant":
                self._messages.pop()

        # Stash the streaming-abort signals so callers (the format-
        # rejection branch in run()) can consult them without
        # re-plumbing every _stream() call. Cleared at the top of every
        # _stream so a previous turn's signal doesn't leak.
        self._last_stream_looped = bool(result.looped)
        self._last_stream_stalled = bool(result.stalled)
        self._last_stream_deliberated = bool(result.deliberated)
        self._last_stream_crashed = bool(result.crashed)
        self._last_stream_loop_kind = (
            getattr(result, "loop_kind", None) if result.looped else None
        )
        self._last_stream_loop_line = (
            getattr(result, "loop_line", None) if result.looped else None
        )
        self._trace({
            "kind": "stream_done",
            "tokens": result.tokens,
            "duration_s": round(result.duration_s, 2),
            "stalled": result.stalled,
            "looped": result.looped,
            "deliberated": result.deliberated,
            "crashed": result.crashed,
            "len": len(result.text),
            # Backend-reported BPE counts when available. `tokens` above is
            # streaming chunk count; these are the real cost numbers used
            # to chart prompt size over a session and to spot when the
            # inlined-file truth source has bloated the input.
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "max_tokens_hit": result.max_tokens_hit,
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
            self._record(AgentEvent(
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
        if result.looped:
            # Visible to the user via the agent log so they understand why
            # the stream cut off mid-output. Trim trailing whitespace from
            # the partial text so downstream regexes see a clean tail.
            loop_kind = getattr(result, "loop_kind", None) or "unknown"
            loop_line = (getattr(result, "loop_line", None) or "").strip()
            extra = f" reason={loop_kind}"
            if loop_line:
                preview = loop_line[:80] + ("..." if len(loop_line) > 80 else "")
                extra += f" sample='{preview}'"
            self._record(AgentEvent(
                "info",
                f"[yellow]repetition loop detected[/yellow] — model was emitting "
                f"the same 1-2 short lines on repeat after {result.tokens} tokens "
                f"({result.duration_s:.0f}s). Aborted stream and kept partial output.{extra}"
            ))
        if result.deliberated:
            # A2: smaller LLMs sometimes ramble pre-tag for thousands of
            # tokens without ever emitting <patch> / <html_file>. We
            # aborted; queue a coaching message so the next user turn
            # tells the model to commit to one root cause + one patch.
            self._record(AgentEvent(
                "info",
                f"[yellow]deliberation loop detected[/yellow] — {result.tokens} "
                f"tokens of pre-tag reasoning with no <patch>/<html_file>. "
                "Aborted stream; coaching the model to skip the essay."
            ))
            self._pending_coaching.append(
                "Your last reply was pure reasoning prose with no output tag — "
                "aborted by the deliberation guard. Do NOT think out loud this "
                "turn. Emit ONE line inside <diagnose>...</diagnose> naming the "
                "single line:variable responsible, then ONE <patch>...</patch> "
                "or <html_file>...</html_file>. No preamble, no exploration."
            )
        if result.stalled and not result.text.strip():
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
            raise RuntimeError(
                f"Model produced no tokens before stalling at "
                f"{self.stall_seconds}s on backend={backend_name}. {hint}"
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
            return res.text, f"applied {res.applied}/{len(patches)} patches"

        html = self._extract_html(reply)
        if html is not None:
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
            if (
                not dry_run
                and self._current_file
                and self._snapshot_n >= 1
                and not allow_rewrite
                and not baseline_degenerate
                and not feedback_exempt
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
        return GameAgent._extract_html_inner(reply)

    @staticmethod
    def _extract_html_inner(reply: str) -> str | None:
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
        reply = _strip_thinking(reply)
        m = _QUESTION_RE.search(reply)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_diagnose(reply: str) -> str | None:
        reply = _strip_thinking(reply)
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

    # Properties whose presence/size is established at init, NOT at
    # runtime — comparisons against them aren't dynamic even when they
    # use `>` / `<`. `state.barrels.length > 0` is "we spawned a
    # barrel", which the game does on iter 1; `c.toDataURL().length >
    # 200` is "the canvas has rendered SOMETHING", true the moment the
    # HUD draws. Excluding these closes the DK trace 20260514_104131
    # false-negative (5/5 'structural-with-floor' probes scored as
    # 40% dynamic, nudge stayed silent on a game where gameplay never
    # advanced).
    _STRUCTURAL_NUMERIC_PROPS = (
        "length", "size", "width", "height", "bytelength",
        "innerwidth", "innerheight", "clientwidth", "clientheight",
        "offsetwidth", "offsetheight",
    )
    _CMP_RE = re.compile(r"(\S+?)\s*([<>]=?)\s*(-?\d+)")
    _CMP_REV_RE = re.compile(r"(-?\d+)\s*([<>]=?)\s*(\S+)")
    # Probes that explicitly try to capture a time-delta — IIFE or
    # ternary patterns that store a `t0` / `before` snapshot and
    # compare later. Genuine dynamic checks; only fire when paired with
    # an explicit time signal (await/setTimeout/etc.) so an `x !== 0`
    # at init doesn't false-positive.

    @staticmethod
    def _is_dynamic_probe(expr: str) -> bool:
        """True if `expr` verifies behavior over time, not just existence.

        Heuristic (no LLM call — this is a regex problem, not a
        reasoning problem; weak models can't reliably self-critique
        their own probe list per TACL 2024 / arXiv 2411.17501).

        Dynamic signals (each is sufficient):
          - awaits or returns a Promise (`await`, `.then(`, `Promise.`)
          - references a timer / clock (`setTimeout`, `setInterval`,
            `requestAnimationFrame`, `performance.now`, `Date.now`)
          - reads canvas state via getImageData (the resulting pixel
            array is genuine runtime state, not a `.length` check)
          - compares against a NON-ZERO numeric threshold AND the LHS
            isn't a known structural property (.length / .size /
            .width / .height etc.)

        Structural-only (returns False):
          - `!!window.state`, `typeof X === 'object'`
          - `X.length > 0`, `X.length > 200`, `width > 0` (existence
            with a floor; true at init for any rendered game)
          - `state.player.x > 0` — at init most games have positive
            starting coordinates; "non-zero" doesn't mean "moved"
          - bare delta-marker identifiers (`prev`, `t0`, `lastFire`)
            — too many false positives like `lastFireTime > 0` which
            just means "fired at init"

        DK trace 20260514_104131 pin: the five probes (`canvas_exists`,
        `player_state`, `barrels_active`, `score_visible`,
        `game_not_blank`) ALL classify as structural under this rule,
        so the nudge fires.
        """
        if not expr:
            return False
        e = expr.strip()
        low = e.lower()
        # 1. Awaits / promises — only way a probe can actually wait.
        if "await " in e or ".then(" in e or "promise." in low:
            return True
        # 2. Timers / clocks — the probe is taking a time-delta.
        for tok in (
            "settimeout", "setinterval", "requestanimationframe",
            "performance.now", "date.now",
        ):
            if tok in low:
                return True
        # 3. Canvas pixel read — getImageData returns runtime state.
        # Exclude `getImageData(...).data.length` (still a presence
        # check on the returned typed array).
        if "getimagedata" in low and ".data.length" not in low:
            return True
        # 4. Numeric comparison against a non-trivial threshold (|N| >=
        # 1) where the LHS isn't a structural-presence property.
        def _is_structural_lhs(lhs: str) -> bool:
            low_lhs = lhs.lower()
            for prop in GameAgent._STRUCTURAL_NUMERIC_PROPS:
                if "." + prop in low_lhs:
                    return True
            return False

        for m in GameAgent._CMP_RE.finditer(e):
            lhs = m.group(1)
            threshold = int(m.group(3))
            if threshold == 0:
                continue
            if _is_structural_lhs(lhs):
                continue
            return True
        for m in GameAgent._CMP_REV_RE.finditer(e):
            rhs = m.group(3)
            threshold = int(m.group(1))
            if threshold == 0:
                continue
            if _is_structural_lhs(rhs):
                continue
            return True
        return False

    @staticmethod
    def _classify_probes_dynamic(probes: list[dict]) -> dict:
        """Bulk-classify probes into structural vs dynamic.

        Returns {"dynamic": [names], "structural": [names], "ratio": float}
        where ratio is dynamic_count / total (0.0 when probes is empty).
        """
        if not probes:
            return {"dynamic": [], "structural": [], "ratio": 0.0}
        dyn: list[str] = []
        struct: list[str] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if GameAgent._is_dynamic_probe(expr):
                dyn.append(name)
            else:
                struct.append(name)
        total = len(dyn) + len(struct)
        return {
            "dynamic": dyn,
            "structural": struct,
            "ratio": (len(dyn) / total) if total else 0.0,
        }

    # Pattern 1: probe binds a local `const x0 = …` (or `let`/`var`), then
    # immediately returns a literal `true`/`false` without ever reading
    # x0 again. The temp binding is wasted; the probe is tautological.
    # Donkey-kong trace 20260516_142445 iter 1 `mario_moves`:
    #   `(()=>{const x0=s.player.x; setTimeout(()=>{}, 100); return true;})()`
    # Pattern 2: probe asserts `typeof obj.NAME === 'undefined'` (or the
    # short form `obj.NAME === undefined`) and returns false on that
    # path, but `obj.NAME` is never assigned anywhere in the game body.
    # That check is permanently true → probe always returns false →
    # zero signal. Donkey-kong trace 20260516_124628 iter 1 `barrels_move`:
    #   `if (typeof b.x0 === 'undefined') return false; …`  (b.x0 is
    #   never assigned).
    # The undefined-property check needs the on-disk HTML, so it runs
    # later — at materialize time, not at probe-parse time. The
    # tautological-temp check is purely structural and runs immediately.
    # Match: `const NAME = …;` followed (eventually) by `return <literal>`.
    # The `.*?` is lazy so we don't span over an outer return that
    # belongs to a different function body. We accept newlines (DOTALL)
    # and braces inside (e.g. an inner `setTimeout(()=>{})`) because
    # the "is the temp dead?" check below uses `expr.count(temp_name)`
    # to confirm the binding is actually unused.
    _PROBE_TAUTOLOGY_RE = re.compile(
        r"(?:const|let|var)\s+(\w+)\s*=\s*[^;]+;"
        r".*?return\s+(?:true|false|0|1)\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def _lint_probes(probes: list[dict]) -> list[dict]:
        """Return a list of `{name, kind, message}` lint findings for
        probes that pass JSON parse but are structurally tautological.

        Catches the donkey-kong 20260516_142445 `mario_moves` shape —
        the probe binds a `const x0` then returns `true` without ever
        comparing against x0. The setTimeout callback is empty, so the
        probe provides no behavioral signal.
        """
        findings: list[dict] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            m = GameAgent._PROBE_TAUTOLOGY_RE.search(expr)
            if m:
                temp_name = m.group(1)
                # Confirm the temp is referenced ONLY in its declaration:
                # split on the temp name; if there's only one occurrence
                # left after stripping the decl prefix, the temp is dead.
                if expr.count(temp_name) <= 1:
                    findings.append({
                        "name": name,
                        "kind": "tautological_constant_return",
                        "message": (
                            f"probe `{name}` binds `{temp_name}` but "
                            f"never reads it again before returning a "
                            f"constant — the probe always returns the "
                            f"same value regardless of game behavior."
                        ),
                    })
        return findings

    @staticmethod
    def _probes_referencing_unassigned_props(
        probes: list[dict], html: str,
    ) -> list[dict]:
        """For each probe, find `obj.PROP` accesses where PROP is never
        assigned anywhere in `html`. Returns the same `{name, kind,
        message}` shape as `_lint_probes`.

        Catches the donkey-kong 20260516_124628 `barrels_move` shape:
        the probe gates on `b.x0` but the game code never sets `b.x0`,
        so the probe trivially returns false. The check is permissive —
        only properties that look like real game-state names (≥ 2 chars,
        not a reserved word, not a built-in property) are checked, to
        avoid flagging legitimate property reads on DOM / built-in
        objects.
        """
        if not probes or not html:
            return []
        # Cheap negation: skip if the HTML body is too small to contain
        # any real game state. The check below assumes a parseable script.
        if "<script" not in html.lower():
            return []
        findings: list[dict] = []
        # Properties to never flag: DOM, canvas, common built-ins. If a
        # probe accesses `c.width` and the HTML never says `c.width = X`,
        # that's fine — `width` is a DOM property of the canvas element.
        _IGNORE = {
            "length", "size", "width", "height", "x", "y", "left", "top",
            "right", "bottom", "data", "value", "textContent",
            "innerText", "innerHTML", "style", "className", "id", "name",
            "parent", "child", "next", "prev", "node", "type", "kind",
        }
        # Extract candidate property *accesses* (not method *calls*).
        # Negative lookahead `(?!\s*\()` skips `obj.method(args)` —
        # methods aren't assignable state, so flagging them as
        # "unassigned" produces false positives (e.g. `obj.toString`,
        # `document.querySelector`). Method-call hallucinations are
        # caught by the separate API-allowlist micro-probe in tools.py.
        prop_re = re.compile(
            r"\b\w+\.([A-Za-z_$][\w$]*)\b(?!\s*\()"
        )
        # Build the assignment set ONCE per HTML — assignments look like
        # `obj.prop =`, `obj.prop +=`, `obj.prop:` (object literal),
        # `prop:` inside object literals at all depths.
        # Conservatively, scan for both patterns.
        assigned: set[str] = set()
        for m in re.finditer(
            r"\.([A-Za-z_$][\w$]*)\s*(?:=|\+=|-=|\*=|/=)(?!=)",
            html,
        ):
            assigned.add(m.group(1))
        # Object-literal entries: `name:` at line start or after `,` /
        # `{` / `(`. This is a loose match (catches some non-assignments
        # like ternary labels) which is FINE — false negatives (skipping
        # a real warning) are cheaper than false positives.
        for m in re.finditer(
            r"(?:[,{(\n]\s*|^\s*)([A-Za-z_$][\w$]*)\s*:",
            html,
        ):
            assigned.add(m.group(1))
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            missing: set[str] = set()
            for m in prop_re.finditer(expr):
                prop = m.group(1)
                if prop in _IGNORE or prop.startswith("_"):
                    continue
                # The probe accesses obj.prop; if `prop` is never
                # assigned anywhere in the file, flag it.
                if prop not in assigned:
                    missing.add(prop)
            if missing:
                sample = ", ".join(f"`{p}`" for p in sorted(missing)[:3])
                findings.append({
                    "name": name,
                    "kind": "unassigned_property_read",
                    "message": (
                        f"probe `{name}` reads {sample} but the game "
                        f"code never assigns those properties — the "
                        f"probe will trivially fail (or short-circuit "
                        f"return) regardless of game behavior. Either "
                        f"fix the probe to read a real game-state "
                        f"property, or add the assignment to the game."
                    ),
                })
        return findings

    @staticmethod
    def _extract_probes(reply: str) -> list[dict]:
        """Pull a JSON list-of-{name,expr} out of <probes>...</probes>.

        Tolerant: accepts either a JSON list of objects, or a list of
        plain strings (treated as `expr`, with name auto-assigned). Bad
        JSON returns []; the agent shouldn't ever crash on a probe parse
        failure since universal probes still cover the basics.
        """
        reply = _strip_thinking(reply)
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
        try:
            path = self.trace_path.parent / f"{self._session_id}.todos.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(todos + "\n", encoding="utf-8")
            self._trace({
                "kind": "todos_captured",
                "len": len(todos),
                "path": str(path),
            })
        except Exception as e:
            self._trace({"kind": "todos_write_failed", "err": str(e)})

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

    async def _maybe_generate_assets_and_sounds(
        self, reply: str, *, trigger: str,
    ) -> AsyncIterator[AgentEvent]:
        """Parse <assets>/<sounds> in a model reply and run the diffuser /
        audio pipeline if either block is present.

        Called from two sites:
        - Phase A plan reply (initial generation).
        - Any Phase B / Phase C reply (mid-session add when the model
          emits a fresh <assets>/<sounds> block in response to user
          feedback like "add proper sprites for the invaders").

        `trigger` is stamped onto the trace events ("phase_a" |
        "mid_session") so offline analysis can distinguish first-load
        from mid-session adds.

        Mid-session semantics: MERGE into self._session_assets /
        self._session_sounds — never overwrite. New assets are
        additive. After generation, push a freshly-rendered asset/sound
        paths block into _pending_feedback so the next user turn shows
        the new file paths to the model (mirrors the Phase A → first
        build prelude path).

        Per-asset success timing is logged here too — one info line per
        sprite when count is small, summary line when >5 to avoid log
        spam. Failure reasons are surfaced one-per-line either way.
        """
        asset_specs, dropped_asset_names = parse_assets_block_with_meta(reply)
        sound_specs = parse_sounds_block(reply)
        if not asset_specs and not sound_specs:
            return

        # Asset overflow: model asked for more than _MAX_ASSETS_PER_TURN.
        # Across four DK traces, this drove the dominant failure mode —
        # the model requested 14 sprites, only the first 8 generated,
        # the rest 404'd in the browser, and the model spent multiple
        # iters patching drawImage symptoms. Surface the gap LOUDLY:
        # trace event the user sees, AND coach the model next turn so
        # it knows to split the request or use img2img chaining.
        if dropped_asset_names:
            self._trace({
                "kind": "asset_overflow",
                "requested": len(asset_specs) + len(dropped_asset_names),
                "generated_cap": len(asset_specs),
                "dropped": dropped_asset_names,
            })
            yield self._record(AgentEvent(
                "info",
                f"[yellow]asset overflow[/yellow] — you requested "
                f"{len(asset_specs) + len(dropped_asset_names)} sprites "
                f"but the per-turn cap is {len(asset_specs)}. Dropped: "
                f"{', '.join(dropped_asset_names)}. Coaching the model to "
                "request the rest in a follow-up turn."
            ))
            self._pending_coaching.append(
                "Your <assets> block requested "
                f"{len(asset_specs) + len(dropped_asset_names)} sprites but "
                f"the harness only generates up to {len(asset_specs)} per "
                "turn. These were DROPPED and will not exist on disk: "
                f"{', '.join(dropped_asset_names)}. To fix: either (a) emit "
                "another <assets> block on a later turn with just the "
                "missing names — the agent will fulfill it mid-session — "
                "or (b) use `from_image` chaining (one base sprite + N "
                "img2img variants) so multiple animation frames cost a "
                "single base generation. Do NOT reference dropped names in "
                "the code until they've been generated."
            )

        # Mid-session: capture pre-existing assets so the feedback block
        # we synthesize at the end reflects only the *new* additions.
        new_asset_paths: dict[str, Path] = {}
        new_sound_paths: dict[str, Path] = {}
        new_looping: set[str] = set()

        if asset_specs:
            yield self._record(AgentEvent(
                "info",
                f"{trigger}: requested {len(asset_specs)} asset(s); "
                "loading Z-Image-Turbo (first call only, ~30-60s)…",
                {
                    "trigger": trigger,
                    "assets_requested": [s["name"] for s in asset_specs],
                },
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
                # Always derive the assets dir from _session_id. When the
                # session was started from a seed (chat.py reuses the
                # seed's path as out_path), _session_id IS the seed's
                # basename, so generation merges into the seed's existing
                # `<basename>_assets/` folder. Fresh sessions get a fresh
                # folder. No override mechanism needed.
                session_assets_dir = (
                    self.out_path.parent / f"{self._session_id}_assets"
                )
                try:
                    produced = await asyncio.to_thread(
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
                    produced = {}
                # Merge — mid-session adds extend, never overwrite.
                # Same-name keys take the newer path (model re-rendered
                # an existing asset on purpose).
                new_asset_paths = dict(produced)
                self._session_assets.update(produced)
                per_asset = getattr(
                    self._asset_generator, "last_stats", None,
                ) or []
                self._trace({
                    "kind": "assets_generated",
                    "trigger": trigger,
                    "requested": len(asset_specs),
                    "produced": len(produced),
                    "names": list(produced.keys()),
                    "session_dir": str(session_assets_dir),
                    "per_asset": per_asset,
                })
                yield self._record(AgentEvent(
                    "assets",
                    f"{len(produced)}/{len(asset_specs)} generated",
                    {
                        "trigger": trigger,
                        "requested": len(asset_specs),
                        "produced": len(produced),
                        "session_dir": str(session_assets_dir),
                        "paths": {n: str(p) for n, p in produced.items()},
                        "per_asset": per_asset,
                    },
                ))
                yield self._record(AgentEvent("activity", "idle"))
                if produced:
                    yield self._record(AgentEvent(
                        "info",
                        f"generated {len(produced)}/"
                        f"{len(asset_specs)} sprites at "
                        f"{session_assets_dir}",
                        {"assets": {n: str(p) for n, p in produced.items()}},
                    ))
                # Per-asset success timing: one line each when small,
                # else a summary. Cache hits (gen_seconds≈0) called out.
                ok_stats = [
                    s for s in per_asset
                    if isinstance(s, dict) and not s.get("error")
                    and s.get("name") in produced
                ]
                if ok_stats:
                    if len(ok_stats) <= 5:
                        for s in ok_stats:
                            secs = float(s.get("gen_seconds") or 0.0)
                            cached = secs < 0.5  # heuristic; cache hits are <100ms
                            tag = " (cached)" if cached else ""
                            yield self._record(AgentEvent(
                                "info",
                                f"  asset {s.get('name','?')}: "
                                f"{secs:.1f}s{tag}",
                            ))
                    else:
                        total = sum(
                            float(s.get("gen_seconds") or 0.0)
                            for s in ok_stats
                        )
                        cached = sum(
                            1 for s in ok_stats
                            if float(s.get("gen_seconds") or 0.0) < 0.5
                        )
                        avg = total / max(1, len(ok_stats))
                        yield self._record(AgentEvent(
                            "info",
                            f"asset gen: {len(ok_stats)}/{len(asset_specs)} "
                            f"in {total:.1f}s (avg {avg:.1f}s/sprite, "
                            f"{cached} cached)",
                        ))
                if len(produced) < len(asset_specs):
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
                            err_line = str(err)[:400]
                            yield self._record(AgentEvent(
                                "info",
                                f"  - {name}: {err_line}"
                            ))

        if sound_specs:
            yield self._record(AgentEvent(
                "info",
                f"{trigger}: requested {len(sound_specs)} sound(s); "
                "loading Stable Audio Open (first call only, ~30-60s)…",
                {
                    "trigger": trigger,
                    "sounds_requested": [s["name"] for s in sound_specs],
                },
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
                # Mirror the asset path above: dir comes from _session_id,
                # which already matches the seed's basename when seeded.
                session_sounds_dir = (
                    self.out_path.parent / f"{self._session_id}_sounds"
                )
                try:
                    produced = await asyncio.to_thread(
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
                    produced = {}
                new_sound_paths = dict(produced)
                self._session_sounds.update(produced)
                # Recompute looping subset from the latest spec set
                # combined with already-installed sounds.
                new_looping = {
                    str(s.get("name", "")).strip()
                    for s in sound_specs if s.get("loop")
                } & set(produced.keys())
                self._session_looping |= new_looping
                per_sound = getattr(
                    self._sound_generator, "last_stats", None,
                ) or []
                self._trace({
                    "kind": "sounds_generated",
                    "trigger": trigger,
                    "requested": len(sound_specs),
                    "produced": len(produced),
                    "names": list(produced.keys()),
                    "looping": sorted(new_looping),
                    "session_dir": str(session_sounds_dir),
                    "per_sound": per_sound,
                })
                yield self._record(AgentEvent(
                    "sounds",
                    f"{len(produced)}/{len(sound_specs)} generated",
                    {
                        "trigger": trigger,
                        "requested": len(sound_specs),
                        "produced": len(produced),
                        "session_dir": str(session_sounds_dir),
                        "paths": {n: str(p) for n, p in produced.items()},
                        "looping": sorted(new_looping),
                        "per_sound": per_sound,
                    },
                ))
                yield self._record(AgentEvent("activity", "idle"))
                if produced:
                    yield self._record(AgentEvent(
                        "info",
                        f"generated {len(produced)}/"
                        f"{len(sound_specs)} sounds at "
                        f"{session_sounds_dir}",
                        {"sounds": {n: str(p) for n, p in produced.items()}},
                    ))
                if len(produced) < len(sound_specs):
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

        # Mid-session only: synthesize a feedback line that re-emits the
        # asset/sound paths block, so the model's next user turn sees
        # the new files via the existing _flush_user_injections channel.
        # Phase A doesn't need this — the first-build assembler already
        # renders these blocks inline.
        if trigger == "mid_session" and (new_asset_paths or new_sound_paths):
            # Mark the iter as having done real preparatory work, so a
            # follow-up "no usable code" outcome doesn't get charged
            # against max_iters. See _media_regenerated_this_iter init.
            self._media_regenerated_this_iter = True
            blocks: list[str] = []
            if new_asset_paths:
                blocks.append(render_asset_paths_block(
                    new_asset_paths, self.out_path,
                ))
            if new_sound_paths:
                blocks.append(render_sound_paths_block(
                    new_sound_paths, self.out_path,
                    looping_names=new_looping,
                ))
            blocks = [b for b in blocks if b]
            if blocks:
                msg = (
                    "Mid-session asset/sound additions — load these in "
                    "your next patch and use them where appropriate. "
                    "The files exist on disk now:\n\n"
                    + "\n\n".join(blocks)
                )
                self._pending_feedback.append(msg)
                self._trace({
                    "kind": "midsession_asset_injection_queued",
                    "asset_names": list(new_asset_paths.keys()),
                    "sound_names": list(new_sound_paths.keys()),
                })

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
            yield self._record(AgentEvent("activity", "idle"))
            self._messages.append({"role": "assistant", "content": plan_reply})
            self._extract_and_queue_lookups(plan_reply)
            self._capture_todos(plan_reply)
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
                yield self._record(AgentEvent(
                    "activity", "streaming",
                    {"label": "streaming plan after question"},
                ))
                try:
                    plan_reply = await self._stream(self._token_cb_wrapper)
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
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({"role": "assistant", "content": plan_reply})
                self._extract_and_queue_lookups(plan_reply)
                self._capture_todos(plan_reply)
                self._dump_conversation()
                yield self._record(AgentEvent("plan", plan_reply))

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
                local_nudge = self._local_first_build_nudge()
                if local_nudge:
                    build_msg = local_nudge + "\n\n" + build_msg
                probe_nudge = self._probe_quality_nudge()
                if probe_nudge:
                    build_msg = probe_nudge + "\n\n" + build_msg
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
                    **pb_kwargs,
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
                local_nudge = self._local_first_build_nudge()
                if local_nudge:
                    build_msg = local_nudge + "\n\n" + build_msg
                probe_nudge = self._probe_quality_nudge()
                if probe_nudge:
                    build_msg = probe_nudge + "\n\n" + build_msg
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
            # Reset per-iter flags so the media-only-bonus check sees
            # only THIS iter's regen state.
            self._media_regenerated_this_iter = False
            # User hard-stop: Ctrl-D in the TUI sets _user_force_done. Honor
            # it at the top of every iteration so the agent never starts a
            # new stream after the user asked to stop, even when probes
            # haven't passed. Whatever's in best.html (or out_path) ships.
            if self._user_force_done:
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
                    prefill_force = False
                    if self._use_prefill and self._fix_mode:
                        reply_prefill = "<diagnose>\n"
                    elif (not self._current_file) and self._force_first_build_prefill:
                        # First-build rescue after a no-code turn.
                        reply_prefill = "<html_file>\n<!DOCTYPE html>\n"
                        prefill_force = True
                    yield self._record(AgentEvent(
                        "activity", "streaming",
                        {"label": f"streaming iter {iteration} reply"},
                    ))
                    reply = await self._stream(
                        self._token_cb_wrapper,
                        prefill=reply_prefill,
                        prefill_force=prefill_force,
                    )
                    yield self._record(AgentEvent("activity", "idle"))
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
                return

            self._messages.append({"role": "assistant", "content": reply})
            self._extract_and_queue_lookups(reply)
            self._capture_todos(reply)
            self._dump_conversation()
            self._trace({
                "kind": "assistant_reply",
                "iteration": iteration,
                "len": len(reply),
                "preview": reply[:600],
            })

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
            )
            if (
                allow_probe_reparse
                and "<probes>" in reply_low
                and has_code
            ):
                new_probes = self._extract_probes(reply)
                if new_probes and len(new_probes) >= len(self._probes):
                    from tools import _criteria_coverage_gaps as _gaps_fn
                    new_gaps = _gaps_fn(self._criteria or "", new_probes)
                    self._trace({
                        "kind": "probes_reparsed",
                        "iteration": iteration,
                        "old_count": len(self._probes),
                        "new_count": len(new_probes),
                        "remaining_gaps": new_gaps[:6],
                        "trigger": (
                            "coverage_gap" if self._planning_coverage_gaps
                            else "prev_probe_failures" if prev_probe_failures > 0
                            else "seed_iter1"
                        ),
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
                            if d_html is not None:
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
            if new_html is None:
                if not self._current_file:
                    # Keep first-build rescue armed until code lands.
                    self._force_first_build_prefill = True
                trunc = self._truncation_diagnosis(reply)
                if trunc:
                    yield self._record(AgentEvent("error", f"TRUNCATED REPLY — {trunc}"))
                else:
                    yield self._record(AgentEvent("info", f"no usable code: {materialize_msg}"))
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
                if patches_in_reply and self._current_file:
                    res = apply_patches(self._current_file, patches_in_reply)
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            self._p.patch_retry_instruction(res.failed, self._current_file)
                        ),
                    })
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
                    self._trace({
                        "kind": "no_usable_code",
                        "plan_only": plan_only,
                        "probes_only": probes_only,
                        "media_only": media_only,
                        "consecutive_plan_only": self._consecutive_plan_only,
                        "has_existing_file": bool(self._current_file),
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
                            prior_loop_kind=self._last_stream_loop_kind,
                            prior_loop_line=self._last_stream_loop_line,
                            is_local_backend=(
                                self._backend.info.name in {"mlx", "ollama"}
                            ),
                        )
                    )
                    if reset_streak:
                        self._consecutive_plan_only = 0
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
            self._force_first_build_prefill = False
            self._consecutive_plan_only = 0
            self._format_stuck_streak = 0
            self._last_materialized_iter = iteration

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
                if unassigned:
                    # Combine with the tautological findings from Phase A;
                    # both flow to the model the same way.
                    self._probe_lint_findings = (
                        [f for f in self._probe_lint_findings
                         if f.get("kind") != "unassigned_property_read"]
                        + unassigned
                    )
                    self._trace({
                        "kind": "probe_lint_postbuild",
                        "iteration": iteration,
                        "findings": unassigned,
                    })

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
            self._last_test_report = report
            self._last_tested_iter = iteration
            yield self._record(AgentEvent("test", report_text, report))

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
            # Visual-progress judge: auto-runs when a local MLX-VLM is
            # discoverable on disk (honors the "never silent cloud calls"
            # rule — `_run_vision_judge` skips cleanly when no local VLM
            # is found, and does NOT fall back to Anthropic). The user
            # can still invoke a cloud judge explicitly via `/check with
            # <model>` in chat.py. Disable entirely with VISION_JUDGE=0.
            if after_bytes is not None:
                try:
                    await self._run_vision_judge(after_bytes, iteration)
                except Exception as exc:
                    self._trace({
                        "kind": "vision_judge_error",
                        "iteration": iteration,
                        "error": str(exc),
                    })
            if self._use_double_screenshot and report.get("screenshot_before"):
                try:
                    before_path = str(report["screenshot_before"])
                    self._last_screenshot_before = Path(before_path).read_bytes()
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
                ids = list(self._active_bullet_ids)
                delta_label: str | None = None
                if report["ok"]:
                    self._playbook.update_counters(ids, helpful_delta=1)
                    delta_label = "helpful+1"
                elif self._stuck_streak >= 3:
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
                    self._save_snapshot(new_html)
                    yield self._record(AgentEvent(
                        "code", str(self.out_path),
                        {"size": len(new_html), "materialize": materialize_msg},
                    ))
            except Exception as e:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent("error", f"Final feedback turn failed: {e}"))

        # ---- Item 5: exit-decision turn before silent loop end ------
        # DK trace 20260514_175012 ended with patches emitted but no
        # <done/> / <confirm_done/> — the user got back a half-fixed
        # game and no clear signal whether the agent had given up or
        # was waiting. Force one final ship-or-ask decision when:
        #   - last test failed
        #   - awaiting_confirm is False (no in-flight done/confirm cycle)
        #   - no pending user feedback (the bonus-turn branch above
        #     already handled that case)
        #   - user didn't force-ship
        if (
            self._previous_report_ok is False
            and not awaiting_confirm
            and not self.has_pending_user_input()
            and not self._user_force_done
        ):
            yield self._record(AgentEvent(
                "info",
                "iter cap reached with failing build — asking the "
                "model to ship-or-ask before exiting silently.",
            ))
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
                "Do NOT emit <patch>, <html_file>, <plan>, "
                "<diagnose>, or any other tag this turn. The "
                "session ends after this reply."
            )
            self._messages.append({
                "role": "user",
                "content": self._flush_user_injections(exit_prompt),
            })
            self._trace({"kind": "exit_decision_turn_prompted"})
            yield self._record(AgentEvent(
                "activity", "streaming",
                {"label": "streaming exit-decision turn"},
            ))
            try:
                exit_reply = await self._stream(self._token_cb_wrapper)
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({
                    "role": "assistant", "content": exit_reply,
                })
                self._dump_conversation()
                self._trace({
                    "kind": "exit_decision_reply",
                    "len": len(exit_reply),
                    "preview": exit_reply[:300],
                })
                # Capture <notes> for the trace + UI; the actual ship
                # decision is reflected in the final-iter test below.
                if _DONE_RE.search(exit_reply):
                    notes = self._extract_notes(exit_reply)
                    if notes:
                        yield self._record(AgentEvent(
                            "info",
                            f"exit notes (model's handoff summary): "
                            f"{notes[:400]}",
                        ))
                    self._trace({"kind": "exit_decision_done"})
                # Handle <question>: surface to user, wait for one
                # answer, then exit. The session ends regardless of
                # what the user types — this is a "what blocker?"
                # ask, not a new iter.
                q = self._extract_question(exit_reply)
                if q is not None:
                    yield self._record(AgentEvent("question", q))
                    while (
                        self._pending_answer is None
                        and not self._user_force_done
                    ):
                        await asyncio.sleep(0.1)
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
                self._trace({"kind": "restart_snapshot_failed", "err": str(e)})
                attempts.append((score, k, canonical_best))
            self._trace({
                "kind": "restart_attempt_end",
                "attempt_idx": k,
                "score": score,
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
            self._trace({"kind": "restart_install_failed", "err": str(e)})
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
        self._pending_coaching = []
        self._last_mistake_sig = None
        self._repeat_sig_streak = 0
        self._force_question_subsystem = None
        self._allow_one_rewrite = False
        self._user_force_done = False
        if self._stop_event is not None:
            self._stop_event.clear()
        self._step_continue = False
        self._last_screenshot_before = None
        self._last_screenshot_after = None
        self._active_bullet_ids = []
        self._restart_attempt_idx = 0
        self._restart_attempt_seed = None
        self._force_first_build_prefill = False

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
                f"{format_report_for_model(report)}"
            )

        report_text = format_report_for_model(report)

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
            return self._p.post_clean_instruction(report_text)

        if regressed:
            best = self._read_best_or_empty()
            return self._p.regression_instruction(report_text, best)

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
        try:
            report = await self.browser.load_and_test(
                self.out_path,
                screenshot_path=None,
                screenshot_before_path=None,
                probes=self._probes or None,
                criteria=self._criteria or None,
            )
        except Exception as e:
            yield self._record(AgentEvent(
                "info", f"[final-test] browser harness crashed: {e}",
            ))
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
