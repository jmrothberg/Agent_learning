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
    flip_sprite_horizontal,
    generate_assets,
    parse_assets_block,
    parse_assets_block_with_meta,
    parse_orientation_verdicts,
    render_asset_paths_block,
    select_orientation_audit_targets,
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
    OpeningBookHit,
    OpeningBookItem,
    Playbook,
    PLAYTESTS_FILENAME,
    ASSET_AUDITS_FILENAME,
    ANIMATION_AUDITS_FILENAME,
    SkeletonHit,
    lookup_bullet,
    render_components_block,
    render_opening_book_block,
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
# Reply emits a complete <html>...</html> document without <!DOCTYPE>. The
# Wolfenstein 2026-05-24 stuck-loop trace burned 5 consecutive iters because
# the model emitted text the parser rejected; one rejection class is the
# "html element with no doctype" shape that classify_format_failure flags
# (wrong_tag_html) but doesn't salvage. Extraction variant 6 below recovers
# the document by prepending a synthetic <!DOCTYPE html>. Stays universal:
# any goal where the model omits the doctype benefits identically.
_BARE_HTML_ELEMENT_RE = re.compile(
    r"(<html\b[^>]*>.*?</html\s*>)",
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

# Anthropic 400s that are payload-SHAPE errors — retrying the same
# payload cannot fix them. The MK trace 20260528 burned two identical
# requests on "does not support assistant message prefill" before the
# fallback logic kicked in. Match against the lowercased error text.
_ANTHROPIC_NON_RETRYABLE_400_PHRASES: tuple[str, ...] = (
    "does not support assistant message prefill",
    "must end with a user message",
)
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


def _normalize_extracted_html(body: str) -> str | None:
    """Return HTML starting at the first document anchor.

    Models sometimes put <diagnose> tails or a second <html_file> opener
    inside the wrapper body (chess trace 20260522_000304 iter 2: the
    file on disk began with ``was truncated…</diagnose>`` before
    <!DOCTYPE). Slice from the first <!DOCTYPE or <html so materialize
    never writes tag garbage ahead of the real document.
    """
    if not body:
        return None
    body = body.strip()
    m = re.search(r"<!DOCTYPE\s+html|<html\b", body, re.IGNORECASE)
    if not m:
        return None
    return body[m.start() :].strip()


def _baseline_structurally_broken(html: str) -> str | None:
    """If the HTML cannot pass pre-Chromium checks, return a short reason."""
    if not html:
        return "empty file"
    stripped = html.lstrip()
    if not stripped.lower().startswith(("<!doctype", "<html")):
        return (
            "file does not start with a HTML document "
            "(leading prose or stray tags before <!DOCTYPE)"
        )
    report = run_micro_probes(html)
    if report.get("ok"):
        return None
    errors = report.get("errors") or []
    if not errors:
        return None
    return errors[0][:240]


def _patch_set_bracket_break(base: str, patched: str, patches) -> str | None:
    """Pre-commit patch validation (fix round, fight trace 20260611_145321):
    if applying `patches` turned a bracket-balanced baseline into an
    imbalanced file, return a rejection message naming the offending
    patch block(s); else None.

    The trace's iter-2 patch applied "2/2 OK" but chopped a function body
    mid-block (+2 `{`), leaving a broken file on disk that burned a full
    structural-recovery rewrite + two ~7-minute best-of-N generations.
    Rejecting at apply time keeps the working baseline on disk and turns
    the failure into a cheap one-patch retry.
    """
    from tools import _SCRIPT_BLOCK_RE, _bracket_imbalance

    def _total_imbalance(html: str) -> dict[str, int]:
        totals = {"{}": 0, "()": 0, "[]": 0}
        for m in _SCRIPT_BLOCK_RE.finditer(html or ""):
            attrs, body = m.group(1) or "", m.group(2) or ""
            if "src=" in attrs:
                continue
            for kind, v in _bracket_imbalance(body).items():
                totals[kind] += v
        return totals

    base_imb = _total_imbalance(base)
    if any(base_imb.values()):
        return None  # baseline already imbalanced — not the patch's fault
    patched_imb = _total_imbalance(patched)
    broken_kinds = [k for k, v in patched_imb.items() if v]
    if not broken_kinds:
        return None
    # Attribute: per-block bracket delta = imbalance(REPLACE) - imbalance(SEARCH).
    culprits: list[str] = []
    for idx, p in enumerate(patches):
        search_imb = _bracket_imbalance(getattr(p, "search", "") or "")
        replace_imb = _bracket_imbalance(getattr(p, "replace", "") or "")
        deltas = []
        for kind in broken_kinds:
            d = replace_imb[kind] - search_imb[kind]
            if d:
                deltas.append(f"{kind} {d:+d}")
        if deltas:
            head = ((getattr(p, "search", "") or "").splitlines() or [""])[0]
            culprits.append(
                f"block {idx + 1} (SEARCH starts `{head[:80]}`) changes {', '.join(deltas)}"
            )
    detail = "; ".join(culprits) if culprits else (
        "net imbalance "
        + ", ".join(f"{k} {patched_imb[k]:+d}" for k in broken_kinds)
    )
    return (
        "patch set rejected: applying it would break bracket balance "
        f"({detail}). The REPLACE text must close every brace it opens — "
        "include the full surrounding lines so SEARCH and REPLACE have "
        "matching bracket counts."
    )


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
    if _baseline_structurally_broken(html) is not None:
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

# Token-aware compaction gate. The lossy structured anchor only fires when the
# last coder prompt used >= this fraction of num_ctx — NOT merely past a
# message count. Local models with a 200k window keep full history (playbook,
# every prior user-feedback item, diagnoses) through long feedback sessions
# instead of compacting at message 15 while 90% of the window is unused.
_COMPACT_PRESSURE = 0.70
# Hard safety cap on total messages, used ONLY when token stats are missing
# (pressure defaults 0.0) so a pathological run can't grow unbounded.
_COMPACT_MESSAGE_CAP = 60

# Fix-round item 3 — absolute prompt-token ceiling for structured compaction.
# The pressure gate above is relative to num_ctx (0.70 of 100K = 70K), but the
# 20260610_185238 trace showed MLX streams stalling silently (0 visible tokens
# for 180-212s) already at 61-69K prompt tokens — the session died before
# pressure ever fired. Compact when the last coder prompt reaches this many
# tokens regardless of pressure. Env-overridable for big-prompt-tolerant
# backends (set 0 to disable the ceiling entirely).
_COMPACT_TOKEN_CEILING = int(
    os.environ.get("AGENT_COMPACT_TOKEN_CEILING", "45000") or "45000"
)

# Capability-round item 2 — polish phase. After probes pass, up to this many
# turns per session are spent on game feel (juice rubric in
# prompts_v1.polish_instruction) instead of immediately pushing <done/>.
# A polish turn that regresses the green build auto-reverts and ends the
# polish phase. Polish never blocks shipping.
_POLISH_TURN_CAP = 2

# Capability-round item 5 — best-of-2 repair on stuck blockers. When the
# session runs with best_of_n == 1 and two consecutive reports failed
# (_stuck_streak >= 2), that turn samples 2 candidates (backend default
# temps 0.2 / 0.6) and ships the better-scoring one. Capped per session to
# bound cost on slow local backends.
_STUCK_BON_ESCALATION_CAP = 2

# Only elide genuinely large inline HTML blobs during message compaction.
# Small examples (e.g. `<html_file>...</html_file>` in instructions) must
# remain verbatim or we mutate the semantics of prior user guidance.
_SUMMARIZE_MIN_HTML_BYTES = 1024

# Capability-round item 3 — context discipline. Each test-report user turn
# is wrapped in these harness sentinels (with a 3-line digest embedded in
# the BEGIN marker). `_prune_messages` collapses superseded wrapped reports
# older than the keep window down to the digest; the newest report always
# stays verbatim. User feedback appended by `_flush_user_injections` sits
# OUTSIDE the wrapper and survives collapse untouched.
_REPORT_BLOCK_BEGIN = "<!-- HARNESS-REPORT-BLOCK digest:"
_REPORT_BLOCK_END = "<!-- /HARNESS-REPORT-BLOCK -->"
_REPORT_BLOCK_RE = re.compile(
    r"<!-- HARNESS-REPORT-BLOCK digest:\n(.*?)\n-->\n(.*?)<!-- /HARNESS-REPORT-BLOCK -->",
    re.DOTALL,
)

# Stale <probes> re-emissions in pruned assistant turns are ~1.5KB each and
# superseded the moment self._probes updates. Bodies above this size are
# collapsed by _summarize_content; small examples stay verbatim.
_SUMMARIZE_MIN_PROBES_BYTES = 300


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
    # "don't change/touch/modify the code", "do not change code",
    # AND "do not change the <noun> code" (e.g. "the game code", "the
    # player code", "the AI code", "the physics code"). Allowing up to
    # two intervening words covers natural-English phrasings like
    # "do not change the chess game code" without firing on neutral
    # text — _feedback_is_strict_scope still requires the explicit
    # forbid-verb so loosening the noun slot can't false-positive.
    re.compile(
        r"\b(?:don['’]?t|do\s+not)\s+(?:change|touch|modify|edit)"
        r"\s+(?:the\s+|any\s+)?(?:\w+\s+){0,2}code\b",
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
    "rerecord", "rebuild", "add", "missing",
    # Added 2026-05-15: user said "fix the images" / "fix the
    # animations" and got a CODE rewrite instead of sprite regen.
    # "fix" alone is too generic to route on, but the gate also
    # requires an art noun (see _feedback_is_art_change), so
    # "fix the keyboard handler" still correctly stays in code-fix
    # mode (no art noun → no MEDIA-CHANGE).
    "fix", "improve",
    # Phase 0.11 — descriptive verbs that pair with an art noun to
    # signal a STYLE rebrand without using an imperative verb. Real
    # user examples that previously failed classification:
    #   "all the images need to be animated as fantasy monsters"
    #   "i want all new graphics so the pieces look like monsters"
    #   "the sprites should look more like X"
    # The art-noun + verb gate keeps these from false-firing on
    # behavior asks ("the player should jump higher" has no art noun
    # → still classified as behavior, not art).
    "look", "looks", "looking",
    "want", "wants", "wanting",
    "need", "needs", "needing",
    "should",
    "feel", "feels", "feeling",
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

# Scoped-behavior terms: if these appear in a strict scoped feedback turn,
# route to a ONE-PATCH behavior fix path (not media-only regen). Added for
# Mortal Kombat trace where "turn around / facing / CPU behavior" got
# misrouted into art regeneration.
_SCOPED_BEHAVIOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bturn(?:\s*-\s*|\s+)around\b", re.I),
    re.compile(r"\bfac(?:e|es|ing)\b", re.I),
    re.compile(r"\bcpu\b", re.I),
    re.compile(r"\bai\b", re.I),
    re.compile(r"\bbehavior\b", re.I),
    re.compile(r"\baction(?:s)?\b", re.I),
    re.compile(r"\b(?:kick|kicks|kicking)\b", re.I),
    re.compile(r"\b(?:punch|punches|punching)\b", re.I),
    re.compile(r"\b(?:attack|attacks|attacking)\b", re.I),
    re.compile(r"\banimation\s+for\b", re.I),
)
_SCOPED_SIZE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:x|times?)\s*(?:larger|smaller|bigger)\b", re.I,
    ),
    re.compile(r"\b(?:larger|smaller|bigger)\b", re.I),
    re.compile(r"\b(?:scale|size)\b", re.I),
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


_STRICT_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bonly\b.*\b(?:change|fix|edit|touch|update|do)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:don['’]?t|do\s+not)\s+change\s+anything\s+else\b",
        re.I,
    ),
    re.compile(r"\bnothing\s+else\b", re.I),
)


def _feedback_is_strict_scope(text: str) -> bool:
    """True when feedback explicitly narrows the turn to one scoped change."""
    if not text:
        return False
    if _feedback_locks_code(text):
        return True
    return any(p.search(text) for p in _STRICT_SCOPE_PATTERNS)


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

# INVERTED-BEHAVIOR patterns — the user is reporting that behavior IS
# happening but in the WRONG way (reversed/swapped/inverted). The
# existing negation regex only catches "X doesn't Y", not "X does the
# opposite of Y". Doom trace 2026-05-23: user wrote "down key moves
# you forward" and "directions and views getting reversed" — clear
# code bugs that slipped past the classifier and got misrouted to
# MEDIA-CHANGE because no behavior_bug suppression fired.
#
# These patterns intentionally pair an INPUT/CONTROL noun with a
# WRONG-DIRECTION qualifier so they don't false-positive on art
# language ("the sprite is inverted" stays orientation-only).
_BEHAVIOR_BUG_INVERTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "direction(s) ... reversed/wrong/inverted/swapped/backwards" within ~6 words
    re.compile(
        r"\bdirection(?:s)?\b[\w\s,]{0,40}\b"
        r"(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|"
        r"flipped|backwards?)\b",
        re.I,
    ),
    # Symmetric: "reversed/wrong/... ... direction(s)" within ~6 words
    re.compile(
        r"\b(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|flipped)\b"
        r"[\w\s,]{0,40}\bdirection(?:s)?\b",
        re.I,
    ),
    # "movement / controls / motion / input / axis ... wrong-shape"
    re.compile(
        r"\b(?:movement|controls?|motion|input|axis|axes)\b[\w\s,]{0,40}\b"
        r"(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|"
        r"flipped|backwards?)\b",
        re.I,
    ),
    # "wrong way" or "opposite way" as a standalone control complaint
    # (already covered by orientation_change for sprite contexts; here
    # we add it to behavior_bug so the input-mismatch case routes too).
    re.compile(r"\b(?:wrong|opposite)\s+(?:way|direction)\b", re.I),
    # Input key + verb + opposite output: "down key moves you forward",
    # "up arrow goes backwards", "left button sends you right".
    re.compile(
        r"\b(?:up|down|left|right|forward|back|backwards?|w|a|s|d)\s+"
        r"(?:key|arrow|button)\b[\w\s,]{0,40}\b"
        r"(?:moves?|sends?|goes|going|turns?|takes?|points?)\b"
        r"[\w\s,]{0,20}\b"
        r"(?:up|down|left|right|forward|back|backwards?|opposite|wrong)\b",
        re.I,
    ),
)


def _feedback_is_behavior_bug(text: str) -> bool:
    """User is reporting a behavior / code bug — "X doesn't Y",
    "nothing happens when …", "the game is broken / frozen / crashing",
    OR "X is happening but in the WRONG / REVERSED way" (input-control
    mismatch).

    DK trace 20260514_104131 fix: the existing art-change classifier
    fires True whenever an asset name appears in feedback, which
    misroutes "mario does not climb the ladder" as an ART/SOUND change
    request (because "mario" and "ladder" are asset names). Detecting
    behavior-bug language lets the directive injector suppress the
    misrouting.

    Doom trace 2026-05-23 fix: also detect INVERTED-behavior complaints
    ("down key moves you forward", "directions getting reversed"). The
    old negation regex only matched "X doesn't Y", missing the "X does
    the OPPOSITE of Y" failure class entirely.

    Patterns matched:
      - negation + behavior verb within 3 words ("does not climb",
        "won't reset", "isn't responding")
      - explicit complaint nouns ("bug", "broken", "crashing",
        "frozen", "stuck", "glitching")
      - inverted-behavior pairs ("direction(s) reversed", "movement
        inverted", "down key moves you forward")

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
    if any(p.search(text) for p in _BEHAVIOR_BUG_INVERTED_PATTERNS):
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


def _matched_names_in_text(text_lower: str, names: list[str]) -> set[str]:
    """Return canonical known names found in feedback text."""
    canon_text = re.sub(r"[\s\-]+", "_", text_lower)
    matched: set[str] = set()
    for name in names:
        n = (name or "").strip().lower()
        if not n:
            continue
        canon_name = re.sub(r"[\s\-]+", "_", n)
        if canon_name and canon_name in canon_text:
            matched.add(canon_name)
    return matched


# Tokens that appear as common prefixes/suffixes in asset names and are NOT
# distinctive on their own — "white" alone shouldn't pull in every white_*
# sprite in the roster. Used by `_resolve_fuzzy_asset_stems` to skip stems
# the user almost certainly didn't mean as a piece identifier.
_NON_DISTINCTIVE_ASSET_STEMS: frozenset[str] = frozenset({
    "white", "black", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey",
    "left", "right", "up", "down", "front", "back", "side", "view",
    "small", "large", "big", "tiny", "huge",
    "light", "dark", "bright",
    "player", "enemy", "npc",  # too broad to disambiguate
    "frame", "anim", "idle", "walk", "run", "attack", "hurt", "die",
    "1", "2", "3", "4",
})


def _resolve_fuzzy_asset_stems(
    text: str, asset_names: list[str]
) -> dict[str, list[str]]:
    """Map distinctive last-tokens of asset names to the parent assets.

    Example: assets = [white_pawn, black_pawn, white_king, black_king];
    user text = "make the pawns walk and the king attack". Returns
    {"pawn": ["white_pawn", "black_pawn"], "king": ["white_king",
    "black_king"]}.

    Used to surface "user said 'pawn', which maps to [white_pawn,
    black_pawn] in your asset map" inside the MEDIA-CHANGE directive
    so the model can emit the right `from_image` chains without having
    to guess. Token-level match; color/side prefixes ("white", "black")
    are skipped via `_NON_DISTINCTIVE_ASSET_STEMS`.
    """
    if not text or not asset_names:
        return {}
    lo = text.lower()
    out: dict[str, list[str]] = {}
    for name in asset_names:
        canon = re.sub(r"[\s\-]+", "_", (name or "").strip().lower())
        if not canon:
            continue
        tokens = [t for t in canon.split("_") if t]
        if not tokens:
            continue
        # Prefer the LAST distinctive token. Walk right-to-left so
        # ("white", "pawn") picks "pawn"; ("hero", "idle", "1") picks
        # "hero" (since "idle" and "1" are non-distinctive).
        stem: str | None = None
        for tok in reversed(tokens):
            if len(tok) < 3 or tok in _NON_DISTINCTIVE_ASSET_STEMS:
                continue
            stem = tok
            break
        if stem is None:
            continue
        if not re.search(rf"\b{re.escape(stem)}s?\b", lo):
            continue
        out.setdefault(stem, [])
        if name not in out[stem]:
            out[stem].append(name)
    return out


# Prefix for feedback items written by the HARNESS (not the user). The
# media-request classifiers must never treat these as user asks — see
# `_feedback_is_art_change`. The model still sees the full text.
_HARNESS_ADVISORY_SENTINEL = "[HARNESS ADVISORY — informational, not a request]"


def _has_audio_context(text_lower: str) -> bool:
    """True when feedback language is explicitly about audio."""
    if any(re.search(rf"\b{re.escape(n)}\b", text_lower) for n in _SOUND_NOUNS):
        return True
    audio_words = (
        "volume", "louder", "quieter", "quiet", "loud", "mute",
        "unmute", "pitch", "echo", "bass", "treble",
    )
    return any(re.search(rf"\b{re.escape(w)}\b", text_lower) for w in audio_words)


def _feedback_is_art_change(text: str, asset_names: list[str]) -> bool:
    """User feedback is asking to change visual art.

    Heuristic: a known asset name appears in the text (literal OR via
    fuzzy stem — Phase 0.11), OR text contains an art-noun ("sprite",
    "image", "art", …) together with a media verb ("change", "redraw",
    "make", "look", "want", …). Misses are safe (no directive injected,
    regular flow proceeds). False positives are also safe — the
    directive only advises the model to prefer <assets>.

    Phase 0.11: when the user says "all the pawns" and the session has
    `white_pawn_idle, black_pawn_idle` etc., the fuzzy stem match
    counts. Without this, a session with N-frame entity names (every
    asset is `<entity>_<state>`) would skip art-change classification
    even when the user clearly referenced an entity.

    Harness-authored advisories (sentinel-prefixed) are NEVER art
    requests: 2026-06-10 dojo-fight traces show the ASSET SANITY
    WARNING (which itself says "do NOT regenerate") being classified
    as a user art request here, arming 3 turns of "ASSET GENERATION
    REQUIRED — The user asked you to GENERATE NEW ART" that
    contradicted both the advisory and the recovery prompt in the
    same message.
    """
    if text.lstrip().startswith(_HARNESS_ADVISORY_SENTINEL):
        return False
    lo = text.lower()
    if _name_in_text(lo, asset_names):
        return True
    if _resolve_fuzzy_asset_stems(text, asset_names):
        return True
    has_noun = any(re.search(rf"\b{re.escape(n)}\b", lo) for n in _ART_NOUNS)
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo) for v in _MEDIA_VERBS
    )
    return has_noun and has_verb


def _feedback_is_sound_change(text: str, sound_names: list[str]) -> bool:
    """Classifier heuristic for `<sounds>` regen feedback.

    Action words that double as combat-game sound names ("kick",
    "punch", "block", "hit", "jump", "attack", "fireball",
    "fatality") are AMBIGUOUS — they appear in graphics-only feedback
    too ("the CPU kick is facing the wrong way", MK trace
    20260517_220025). When ONLY those names matched, require explicit
    audio vocabulary (a `_SOUND_NOUN` or volume/pitch word) before
    routing to sound regen. False positives steer the model toward
    `<sounds>` regeneration on a graphics turn, polluting the prompt
    and (in the trace) burning iters; false negatives keep the turn
    on its current path which is the safer outcome.
    """
    lo = text.lower()
    matched = _matched_names_in_text(lo, sound_names)
    if matched:
        ambiguous = {
            "kick", "punch", "block", "hit", "jump", "attack",
            "fireball", "fatality",
        }
        unambiguous_match = any(name not in ambiguous for name in matched)
        if unambiguous_match:
            return True
        if _has_audio_context(lo):
            return True
        # All matched names are ambiguous AND no audio context — do
        # NOT fall through to the noun+verb gate, which has False
        # negatives we already accept. Returning False here pins the
        # behavior observed by the MK regression tests.
        return False
    # No sound-name match at all → require BOTH a sound noun and a
    # media verb. This catches generic "redo the music" / "redesign
    # the soundtrack" without an existing-name match.
    has_noun = any(
        re.search(rf"\b{re.escape(n)}\b", lo) for n in _SOUND_NOUNS
    )
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo) for v in _MEDIA_VERBS
    )
    return has_noun and has_verb


_EXISTING_MEDIA_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdon['’]?t\s+(?:redo|redraw|regenerate|remake)\b", re.I),
    re.compile(r"\bdo\s+not\s+(?:redo|redraw|regenerate|remake)\b", re.I),
    re.compile(r"\buse\s+(?:the\s+)?(?:existing|original|old|current)\b", re.I),
    re.compile(r"\balready\s+exists?\b", re.I),
    re.compile(r"\buse\s+them\b", re.I),
)


def _feedback_requests_existing_media(text: str) -> bool:
    """User wants existing media wired/used, not regenerated."""
    if not text:
        return False
    return any(p.search(text) for p in _EXISTING_MEDIA_ONLY_PATTERNS)


# Phase 0.1 — phrases that explicitly request img2img CHAINING (seed from
# an existing sprite, produce variants/animation frames). Distinct from
# "use the existing PNG verbatim" — the user wants NEW frames seeded from
# an existing one, which is exactly what SD-Turbo img2img handles. See
# 2026-05-22 chess trace: "use the existing assets for each pawn... as
# starting point but then show each walking" — pure img2img.
_IMG2IMG_CHAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bas\s+(?:a\s+|the\s+)?(?:starting\s+point|base|seed|basis|reference)\b", re.I),
    re.compile(r"\bbased\s+on\s+(?:the\s+)?(?:existing|current|original)\b", re.I),
    re.compile(r"\bseed(?:ed)?\s+from\b", re.I),
    re.compile(r"\b(?:show|showing|make)\s+(?:them|it|each|the\s+\w+\s+)?(?:walking|running|attacking|moving|jumping|dancing|fighting|smashing|killing|dying|capturing)\b", re.I),
    re.compile(r"\b(?:more|additional|extra|new)\s+(?:frame|frames|variant|variants|version|versions|animation)\b", re.I),
    re.compile(r"\banimated\s+(?:series|sequence|set|version)\b", re.I),
    re.compile(r"\banimat(?:e|ing)\s+(?:the\s+)?(?:existing|current|pieces|sprites?|characters?)\b", re.I),
    re.compile(r"\bwalk(?:ing)?\s+cycle\b", re.I),
)


def _feedback_requests_img2img_chain(text: str) -> bool:
    """True when the user is asking for animation frames or variants
    chained from EXISTING sprites — the SD-Turbo `from_image` path."""
    if not text:
        return False
    return any(p.search(text) for p in _IMG2IMG_CHAIN_PATTERNS)


# Phase 0.11 — phrases that signal the user wants a STYLE REBRAND of
# existing assets (regenerate every existing asset NAME with a NEW
# prompt that bakes in a different style). Distinct from:
#   - existing_media: "use the existing X, don't regen" (KEEP as-is)
#   - img2img_chain: "use existing as starting point for animation
#                     frames" (REGEN as variants via from_image)
#   - style_rebrand: "ALL of the images need to look like X" /
#                    "all new graphics, look like monsters" — REGEN
#                    each name in place via txt2img (NOT from_image,
#                    because that would carry the old style forward)
_STYLE_REBRAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blook(?:s|ing)?\s+like\b", re.I),
    re.compile(r"\binstead\s+of\b", re.I),
    re.compile(r"\bnot\s+look\s+like\b", re.I),
    re.compile(r"\bnot\s+(?:regular|standard|normal|plain|generic)\b", re.I),
    re.compile(r"\b(?:themed?|styled?)\s+(?:as|like)\b", re.I),
    re.compile(r"\b(?:in\s+the\s+)?style\s+of\b", re.I),
    re.compile(r"\bnew\s+(?:style|look|theme|art|graphics|design|visuals?)\b", re.I),
    re.compile(r"\bdifferent\s+(?:style|look|theme|art|design|aesthetic)\b", re.I),
    re.compile(r"\b(?:all|every)\s+(?:new|fresh)\s+(?:graphics?|sprites?|images?|art|assets?)\b", re.I),
    re.compile(r"\b(?:re\s*-?\s*)?(?:design|theme|skin|reskin)\s+(?:them|the\s+\w+|all)\b", re.I),
    re.compile(r"\bmake\s+(?:them|all|every)\b.*\blook\b", re.I | re.DOTALL),
)


def _feedback_requests_style_rebrand(text: str) -> bool:
    """True when the user wants existing assets re-rendered with a new
    visual style (txt2img with new prompts, not from_image)."""
    if not text:
        return False
    return any(p.search(text) for p in _STYLE_REBRAND_PATTERNS)


def _feedback_requests_size_change(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _SCOPED_SIZE_PATTERNS)


# Orientation-change vocabulary: "invert / mirror / flip / face the
# other way / facing wrong / horizontally". Genre-free: describes a
# rendering modality (mirror a sprite) not subject matter. A false
# positive routes the turn to a one-patch canvas-flip recipe instead
# of asset regeneration, so the gate also REQUIRES the absence of
# explicit style-change verbs ("redraw", "redesign", "new art") and
# does not fire when the user explicitly says "new asset" / "make a
# new" — those are regen requests, not mirror requests.
_ORIENTATION_VERB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\binvert(?:ed|ing)?\b", re.I),
    re.compile(r"\bmirror(?:ed|ing)?\b", re.I),
    re.compile(r"\bflip(?:ped|ping)?\b", re.I),
    re.compile(r"\b(?:facing|faces|face)\s+(?:the\s+)?(?:wrong|other|opposite|right|left)\b", re.I),
    re.compile(r"\bwrong\s+(?:way|direction)\b", re.I),
    re.compile(r"\bhorizontal(?:ly)?\b", re.I),
    re.compile(r"\brotat(?:e|ed|ing|ion)\s+(?:just|only|the)\b", re.I),
)
_ORIENTATION_REGEN_BLOCKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnew\s+asset\b", re.I),
    re.compile(r"\bmake\s+(?:a\s+)?new\b", re.I),
    re.compile(r"\bregenerat(?:e|ed|ing)\b", re.I),
    re.compile(r"\bredraw\b", re.I),
    re.compile(r"\bredesign\b", re.I),
    re.compile(r"\bdifferent\s+style\b", re.I),
)

# Negation tokens that, when they precede a blocker phrase, INVERT the
# blocker's meaning ("do NOT make a new asset" forbids regen — the
# opposite of the bare blocker's "user wants regen" reading). The
# doom trace 2026-05-23 was the motivating case: user wrote "do not
# make a NEW asset, i think the pistal maybe facing the wrong way"
# and the literal `\bnew\s+asset\b` blocker fired, suppressing the
# orientation route and letting MEDIA-CHANGE re-render the pistol —
# exactly what the user forbade.
_BLOCKER_NEGATION_RE = re.compile(
    r"\b(?:not|no|don['’]?t|doesn['’]?t|never|without|please\s+don['’]?t)\b",
    re.I,
)


def _phrase_is_negated(text: str, match_start: int, window_chars: int = 30) -> bool:
    """True when a negation word appears within `window_chars` BEFORE match_start."""
    pre = text[max(0, match_start - window_chars): match_start]
    return bool(_BLOCKER_NEGATION_RE.search(pre))


def _feedback_is_orientation_change(text: str) -> bool:
    """Classifier heuristic: user wants a sprite mirrored/flipped on
    the canvas (one small <patch>), not regenerated. False positives
    push a code patch path when the user actually wanted regen, so
    blockers like "new asset" / "redraw" suppress the route — UNLESS
    those blocker phrases are themselves negated ("do NOT make a new
    asset"), in which case the user is forbidding regen and the
    orientation route should fire.

    Genre-free: vocabulary describes rendering modality, not subject
    matter. Returns True only when an orientation verb fires AND no
    un-negated regen blocker fires.
    """
    if not text:
        return False
    for p in _ORIENTATION_REGEN_BLOCKERS:
        for m in p.finditer(text):
            if not _phrase_is_negated(text, m.start()):
                return False
    return any(p.search(text) for p in _ORIENTATION_VERB_PATTERNS)


def _feedback_mentions_scoped_behavior_change(text: str) -> bool:
    if not text:
        return False
    if _feedback_is_behavior_bug(text):
        return True
    return any(p.search(text) for p in _SCOPED_BEHAVIOR_PATTERNS)


def _scoped_probe_keywords(text: str, limit: int = 3) -> list[str]:
    """Pick a tiny deterministic keyword set for scoped-check probes.

    Keep this intentionally short for local-model correction prompts.
    """
    if not text:
        return []
    lo = text.lower()
    ordered = (
        "turn around",
        "facing",
        "cpu",
        "ai",
        "behavior",
        "action",
        "kick",
        "punch",
        "jump",
        "attack",
    )
    out: list[str] = []
    for key in ordered:
        if key in lo:
            out.append(key)
        if len(out) >= limit:
            break
    return out


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
    kind: str           # phase | token | plan | code | test | question | done | error | info | diagnose | patch | best_of_n | memory | activity | assets | sounds | videos | streak
    text: str = ""
    data: dict = field(default_factory=dict)


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
        elif kind == "no_usable_code":
            reason_bits = [
                key for key in ("plan_only", "probes_only", "media_only",
                                "identical_repeat")
                if rec.get(key)
            ]
            no_code_turns.append(",".join(reason_bits) or "rejected/unparsed")
        elif kind == "harness_crash":
            harness_crashes += 1
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
    if iters:
        lines.append("| iter | materialize | bytes | ok | probes | blocker |")
        lines.append("|------|-------------|-------|----|--------|---------|")
        for it in sorted(iters):
            row = iters[it]
            lines.append(
                f"| {it} | {row.get('materialize', '')} "
                f"| {row.get('bytes', '')} | {row.get('ok', '')} "
                f"| {row.get('probes', '')} | {row.get('blocker', '')} |"
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
        # Wikipedia research toggle. OFF by default — empirical test
        # 2026-05-19 returned 0/10 hits on common game goals. /wiki on
        # in chat.py (or set _research_enabled=True on the agent) opts
        # in for users who want to test the matcher.
        self._research_enabled: bool = False
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
        # Phase 0.10 — per-session cap on assets generated per <assets>
        # block. Default `None` means "use module default" (24). Raised
        # at session start when the goal explicitly asks for multi-frame
        # rosters via `prompts_v1._detect_multi_frame_intent`. The raise
        # lets a user-requested N entities × M frames roster land in one
        # turn instead of getting silently truncated to 24.
        self._session_asset_cap: int | None = None
        # Phase 1.5 — autonomous self-feedback loop. Mirrors the chat.py
        # toggle so the agent can check this flag without depending on
        # the TUI. Default ON; one slash command (/feedback off) flips it.
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
            # Single-LLM "all roles" path: when the user opted into VLM critique
            # but no dedicated critic slot exists, run the critic on slot 1.
            # Gated on the toggle so the default-off path keeps returning None
            # (which lets _run_vision_judge take over when a local VLM is on disk).
            if getattr(self, "_use_vlm_critique", False):
                return self._backend
            return None  # Fallback to standard non-visual automated testing

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

    _CLASSIFIER_AUTO_DISABLE_THRESHOLD = 2

    def _record_classifier_overrule(self, *, expected_mode: str, model_emitted: str, feedback_preview: str) -> None:
        """Bump the overrule counter + auto-disable directives at threshold.

        Centralizes the bookkeeping so both overrule emit sites stay in
        sync. When the running count reaches the threshold we flip
        `_use_feedback_directives = False` for the rest of the session
        (a per-session decision; new sessions start fresh).
        """
        self._trace({
            "kind": "scoped_classifier_overruled_by_model",
            "expected_mode": expected_mode,
            "model_emitted": model_emitted,
            "feedback_preview": feedback_preview,
            "count": self._classifier_overrule_count + 1,
        })
        self._classifier_overrule_count += 1
        if (
            not self._classifier_auto_disabled
            and self._classifier_overrule_count >= self._CLASSIFIER_AUTO_DISABLE_THRESHOLD
            and getattr(self, "_use_feedback_directives", True)
        ):
            self._use_feedback_directives = False
            self._classifier_auto_disabled = True
            self._trace({
                "kind": "classifier_auto_disabled_after_repeated_overrules",
                "count": self._classifier_overrule_count,
                "threshold": self._CLASSIFIER_AUTO_DISABLE_THRESHOLD,
                "reason": (
                    "model overruled the scoped classifier "
                    f"{self._classifier_overrule_count} times this session; "
                    "switching to raw feedback mode for the rest of the "
                    "session per the agent-must-beat-zero-shot rule. "
                    "Resets next /new."
                ),
            })

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

    def _queue_visual_critic_coaching(self, cleaned: str, *, iteration: int, vc_role: str = "critic") -> bool:
        """Append a visual-critic note to `_pending_coaching` with cross-turn dedup.

        Returns True when the note was queued, False when suppressed as
        a repeat of one we already queued in the last 3 critic turns.
        """
        prefix = "VISUAL CRITIC (looked at the screenshot of your last iteration): "
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
        return True

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
        head_n = GameAgent._REJECTED_REPLY_STUB_HEAD
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
            and rejection_kind == "unclosed_html_file"
        )

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
        prior_loop_kind: str | None = None,
        prior_loop_line: str | None = None,
        is_local_backend: bool = False,
        materialize_reject_reason: str = "",
        identical_repeat: bool = False,
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
        for postmortem inspection of which bullets fired.
        """
        if self._playbook_top_k <= 0:
            return ""
        if getattr(self._p, "PLAYBOOK_DISABLED", False):
            return ""
        try:
            if stage == "plan":
                k = self._playbook_top_k + self._PLAN_STAGE_TOP_K_BONUS
                if self._model_class in ("mid", "small"):
                    k = 2
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
                full_top_n_val = 1 if self._model_class in ("mid", "small") else 3
            else:
                k = min(self._playbook_top_k, self._CODE_STAGE_TOP_K)
                if self._model_class in ("mid", "small"):
                    k = 1
                budget = self._CODE_STAGE_CHAR_BUDGET
                # Code stage already narrowly retrieves; full bodies on all.
                render_mode = "full"
                full_top_n_val = 3
            # Modality-token expansion (added 2026-05-21): when the goal
            # mentions a known modality (board/DOM/3D/etc), append those
            # keywords to the playbook query so the modality-tagged
            # bullets retrieve above the 0.05 Jaccard noise floor. The
            # detectors are the same ones the skeleton retrieval uses;
            # importing here keeps the helper public-API stable.
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_board_intent, _detect_dom_intent, _detect_3d_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            # Fix B (2026-05-25): include the most recent user feedback
            # in the retrieval query so scope-discipline bullets
            # (scope-locked-by-user-language, patch-budget-when-scope-
            # locked, vlm-critic-can-mislead-on-orientation, etc.) fire
            # when the user types scope-lock language like "JUST do X"
            # or "do not touch other code". Without this, those bullets
            # only retrieve when the GOAL itself contains scope vocab —
            # which it never does, so the bullets sit unused across the
            # whole session. The doom trace 2026-05-25 made this gap
            # visible: user typed "JUST change the direction the player
            # moves with the arrow keys they are REVERSED" four times
            # in a row, and the scope-locked / patch-budget bullets
            # never surfaced because retrieval keyed only on the goal.
            #
            # Source of feedback text: prefer `_last_drained_feedback`
            # (the feedback that just landed in this turn's user-turn
            # message) over `_pending_feedback` (queued for next turn).
            # On a fix-prompt build the feedback for THIS turn has
            # already been drained, so `_last_drained_feedback` is the
            # right scope. Fall back to `_pending_feedback` when the
            # call site runs before drain.
            recent_feedback = (
                getattr(self, "_last_drained_feedback", None)
                or getattr(self, "_pending_feedback", None)
                or []
            )
            feedback_text = " ".join(str(f) for f in recent_feedback[-2:])
            # For small/mid model classes the playbook surfaces k=1 hit;
            # the goal text dominates the query by sheer length, so any
            # bullet matched only by feedback words gets buried. We
            # repeat the feedback 3x so its tokens compete on count
            # with the longer goal. Larger model classes already get
            # k=3-6 retrievals and don't need the boost as much.
            if feedback_text and self._model_class in ("mid", "small"):
                feedback_weighted = " ".join([feedback_text] * 3)
            else:
                feedback_weighted = feedback_text
            query = goal if not feedback_text else f"{goal} {feedback_weighted}"
            hits = self._playbook.retrieve(
                query, code=code, k=k, stage=stage,
                modality_tokens=mod_toks,
            )
            if hits:
                ids = [h.bullet.id for h in hits]
                self._trace({
                    "kind": "playbook_retrieved",
                    "stage": stage,
                    "ids": ids,
                    "scores": [round(h.score, 4) for h in hits],
                    "goal_preview": goal[:120],
                    "feedback_in_query": bool(feedback_text),
                    "feedback_preview": feedback_text[:200] if feedback_text else "",
                    "char_budget": budget,
                    "render_mode": render_mode,
                })
                self._active_bullet_ids = list(ids)
            block = render_playbook_block(
                hits, char_budget=budget, mode=render_mode, full_top_n=full_top_n_val,
            )
            # Injection observability: distinguishes "retrieved but rendered
            # empty" from "actually placed in the prompt". In the 2026-05-29
            # fighting trace it was impossible to tell whether the animation
            # bullets ever reached the model — this makes it one grep.
            self._trace({
                "kind": "playbook_injected",
                "stage": stage,
                "ids": [h.bullet.id for h in hits] if hits else [],
                "chars": len(block or ""),
                "rendered": bool(block),
            })
            return block
        except Exception:
            return ""

    def _retrieve_opening_book_block(
        self,
        goal: str,
        *,
        stage: str = "plan",
    ) -> tuple[str, list[dict]]:
        """Retrieve compact root/live opening-book recipes with hard caps."""
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            outline = self._memory.retrieve_implementation_outline(goal, mod_toks)
            playtests = self._memory.retrieve_playtests(
                goal, mod_toks, k=3 if stage == "plan" else 1,
            )
            asset_audits = self._memory.retrieve_asset_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            animation_audits = self._memory.retrieve_animation_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            # Plan stage deep-renders the ONE matched outline's recipe
            # (state/order/traps/tuning/probes) under a 3600-char cap —
            # ~+600 tokens in the smallest prompt of the session. Code
            # stage stays shallow at 1400 so iterate prompts never grow.
            block = render_opening_book_block(
                outline, playtests, asset_audits, animation_audits,
                char_budget=3600 if stage == "plan" else 1400,
                deep=(stage == "plan"),
            )

            def _row(kind: str, hit: OpeningBookHit) -> dict:
                return {
                    "kind": kind,
                    "id": hit.item.id,
                    "tier": hit.item.source_tier,
                    "score": round(hit.score, 4),
                    "recipe": hit.item.recipe,
                }

            hits: list[dict] = []
            if outline:
                hits.append(_row("outline", outline))
            hits.extend(_row("playtest", h) for h in playtests)
            hits.extend(_row("asset_audit", h) for h in asset_audits)
            hits.extend(_row("animation_audit", h) for h in animation_audits)
            if hits:
                self._trace({
                    "kind": "opening_book_retrieved",
                    "stage": stage,
                    "hits": hits,
                    "modality_tokens": mod_toks,
                })
            return block, hits
        except Exception as e:
            self._trace({"kind": "opening_book_error", "stage": stage, "err": str(e)})
            return "", []

    def _retrieve_components_block(
        self,
        query: str,
        *,
        stage: str = "plan",
        k: int = 3,
    ) -> str:
        """Retrieve component-library snippets as a <components> block.

        Capability-round item 1: tested, mechanics-level JS the model
        pastes and ADAPTS (memory/components.jsonl). `query` is the goal
        at first build; at fix turns it is the blocker text so a snippet
        is only injected when it matches the actual failure. Returns ""
        when nothing matches (safe degradation).
        """
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(query)
                    + _detect_board_intent(query)
                    + _detect_dom_intent(query)
                )
            except Exception:
                mod_toks = []
            hits = self._memory.retrieve_components(query, mod_toks, k=k)
            block = render_components_block(
                hits, char_budget=2200 if stage == "plan" else 1400,
            )
            if hits:
                self._trace({
                    "kind": "components_injected",
                    "stage": stage,
                    "ids": [h.item.id for h in hits],
                    "scores": [round(h.score, 4) for h in hits],
                    "chars": len(block or ""),
                    "rendered": bool(block),
                })
            return block or ""
        except Exception as e:
            self._trace({"kind": "components_error", "stage": stage, "err": str(e)})
            return ""

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

    def _queue_internal_feedback(self, text: str) -> None:
        """Queue an AGENT-generated notice into the feedback channel.

        Added 2026-06-12: agent-injected texts (mid-session media loader
        notices, scoped-media locks, autonomous-playtest findings) ride
        the same `_pending_feedback` queue as real user feedback, so the
        unhonored-asset-request detector in `_flush_user_injections`
        classified them as USER art requests — trace 20260612_004616
        fired 8 spurious ASSET GENERATION REQUIRED banners demanding
        <assets> for files that already existed on disk. Texts queued
        here are remembered in `_internal_feedback_texts` so detectors
        that must only fire on genuine user feedback can skip them.
        The model still receives the text unchanged.
        """
        text = (text or "").strip()
        if not text:
            return
        # Lazy-init so partially-constructed test stubs (which skip
        # __init__) don't AttributeError here.
        if not hasattr(self, "_internal_feedback_texts"):
            self._internal_feedback_texts = set()
        self._internal_feedback_texts.add(text)
        self._pending_feedback.append(text)

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

    def set_research_enabled(self, on: bool) -> None:
        """Toggle Wikipedia research lookup before planning. OFF by
        default per /wiki slash command in chat.py — empirical test
        2026-05-19 returned 0/10 hits on common game goals so the
        lookup is pure latency unless the operator opts in to test or
        improve the matcher.
        """
        self._research_enabled = bool(on)
        self._trace({
            "kind": "research_enabled_set",
            "on": self._research_enabled,
        })

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

    def _early_rehydrate_seed_media(self) -> tuple[int, int]:
        """Populate `_session_assets` / `_session_sounds` from the seed
        BEFORE Phase A asset generation runs.

        P1 (MK trace 20260528): the previous run loop rehydrated AFTER
        `_maybe_generate_assets_and_sounds(trigger="phase_a")` had
        already regenerated every sprite the model re-requested in its
        plan. By rehydrating first, the skip-guard inside
        `_maybe_generate_assets_and_sounds` can short-circuit phase_a
        generation when on-disk media already covers the game.

        Idempotent: a second call from the existing later branch is a
        no-op (dict.update with the same paths). Returns (n_assets,
        n_sounds) found on disk, or (0, 0) when there is no seed.
        """
        if not self.seed_file:
            return 0, 0
        try:
            seed_html = self.seed_file.read_text(encoding="utf-8")
        except Exception as e:
            self._trace({
                "kind": "seed_media_early_rehydrate_failed",
                "err": str(e)[:200],
                "seed_file": str(self.seed_file),
            })
            return 0, 0
        try:
            seed_assets, seed_sounds, _, _ = _scan_seed_media(
                seed_html, self.out_path
            )
        except Exception as e:
            self._trace({
                "kind": "seed_media_early_rehydrate_failed",
                "err": str(e)[:200],
                "stage": "_scan_seed_media",
            })
            return 0, 0
        if seed_assets:
            self._session_assets.update(seed_assets)
        if seed_sounds:
            self._session_sounds.update(seed_sounds)
        self._trace({
            "kind": "seed_media_early_rehydrate",
            "assets": len(seed_assets),
            "sounds": len(seed_sounds),
            "asset_names": sorted(seed_assets.keys())[:24],
            "sound_names": sorted(seed_sounds.keys())[:24],
        })
        return len(seed_assets), len(seed_sounds)

    def _render_seed_media_contract(
        self,
        html: str,
        *,
        asset_names: list[str] | None = None,
        sound_names: list[str] | None = None,
    ) -> str:
        """Compact contract for seed runs: existing media are first-class.

        Seeded edits should wire or reuse media already on disk before asking
        for new generations. The contract lists referenced vs available names
        and highlights available-but-unused names that often solve "missing
        animation" requests by adding loader entries, not regenerating art.
        """
        assets = sorted(asset_names if asset_names is not None else self._session_assets.keys())
        sounds = sorted(sound_names if sound_names is not None else self._session_sounds.keys())
        refs_a = sorted(self._scan_html_for_asset_refs(html or ""))
        refs_s = sorted(self._scan_html_for_sound_refs(html or ""))
        unused_a = sorted(set(assets) - set(refs_a))
        unused_s = sorted(set(sounds) - set(refs_s))

        def _fmt(names: list[str], cap: int = 28) -> str:
            if not names:
                return "(none)"
            shown = names[:cap]
            more = f" (+{len(names) - cap} more)" if len(names) > cap else ""
            return ", ".join(shown) + more

        lines = [
            "================ SEED MEDIA CONTRACT ================",
            "This is an EXISTING seeded game. Treat assets/sounds already",
            "on disk as the current game's media API. Use/wire existing",
            "names before asking for new art or sound.",
            "",
            f"Available assets: {_fmt(assets)}",
            f"Referenced assets in HTML: {_fmt(refs_a)}",
            f"Existing assets not currently referenced/loaded: {_fmt(unused_a)}",
        ]
        if sounds:
            lines.extend([
                "",
                f"Available sounds: {_fmt(sounds)}",
                f"Referenced sounds in HTML: {_fmt(refs_s)}",
                f"Existing sounds not currently referenced/loaded: {_fmt(unused_s)}",
            ])
        lines.extend([
            "",
            "Rules for seed edits:",
            "  - If the user says missing / use existing / don't redo, patch",
            "    loader or draw mapping to use existing names.",
            "  - Emit <assets>/<sounds> only when the user explicitly asks",
            "    for new art/sound or no existing name can satisfy the request.",
            "  - Do not design a new game; adapt this seed.",
            "=====================================================",
        ])
        return "\n".join(lines)

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

    # Markers used by `_estimate_prompt_section_chars` to attribute
    # the most recent user message to known sections. Position of the
    # first occurrence is used as the section start; the section ends
    # at the next marker (or end-of-string).
    _PROMPT_SECTION_MARKERS: tuple[tuple[str, str], ...] = (
        ("user_answer", "================ USER ANSWER (HIGHEST PRIORITY)"),
        ("user_feedback", "================ USER FEEDBACK (HIGHEST PRIORITY)"),
        ("scope_arbitration", "================ FEEDBACK SCOPE ARBITRATION"),
        ("media_directive", "================ MEDIA-CHANGE DIRECTIVE"),
        ("scoped_change", "================ SCOPED-CHANGE DIRECTIVE"),
        ("agent_coaching", "================ AGENT COACHING"),
        ("generated_assets", "================ GENERATED ASSETS"),
        ("generated_sounds", "================ GENERATED SOUNDS"),
        ("state_anchor", "# Session state anchor"),
        ("current_file", "CURRENT FILE ON DISK"),
        ("seed_file", "EXISTING FILE:"),
    )

    def _estimate_prompt_section_chars(self) -> dict:
        """Approximate char counts per section in the most recent user
        message, plus totals for system prompt and message history.
        Read by the `turn_contract` trace event. Best-effort: silently
        returns minimal data if the message list is empty.
        """
        out: dict[str, int] = {}
        # System prompt size (first message, role=system).
        if self._messages and self._messages[0].get("role") == "system":
            sys_content = self._messages[0].get("content") or ""
            out["system"] = len(sys_content) if isinstance(sys_content, str) else 0
        else:
            out["system"] = 0
        # Total message-history chars (matches _estimate_ctx_fill).
        out["history_total"] = self._estimate_ctx_fill()
        # Most-recent user message section breakdown.
        last_user_content = ""
        for msg in reversed(self._messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    last_user_content = content
                break
        if not last_user_content:
            return out
        out["last_user_chars"] = len(last_user_content)
        # Find every marker's first-occurrence offset; sort by offset
        # and slice between consecutive markers. Sections that don't
        # appear in this message are omitted (cheap, no noise).
        hits: list[tuple[int, str]] = []
        for label, marker in self._PROMPT_SECTION_MARKERS:
            idx = last_user_content.find(marker)
            if idx >= 0:
                hits.append((idx, label))
        hits.sort()
        for i, (start, label) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(last_user_content)
            out[f"section_{label}"] = end - start
        return out

    def _apply_initial_goal_scoping(self, goal: str, build_msg: str) -> str:
        """Apply scoped constraints + SCOPED-CHANGE directive when the
        initial goal carries strict-scope language ("no other changes",
        "just X", "only Y"). Returns the (possibly augmented)
        build_msg. MK trace 20260517_220025 motivation: goal said
        "ROTATING just the CPU punch horizontally make NO other
        changes" but scoping only kicked in for mid-session feedback,
        not the very first build turn.
        """
        if not goal:
            return build_msg
        if not (_feedback_is_strict_scope(goal) or _feedback_locks_code(goal)):
            return build_msg
        locks_code = _feedback_locks_code(goal)
        asset_names = list(self._session_assets.keys()) if self._session_assets else []
        sound_names = list(self._session_sounds.keys()) if self._session_sounds else []
        art_change = bool(asset_names) and _feedback_is_art_change(goal, asset_names)
        sound_change = bool(sound_names) and _feedback_is_sound_change(goal, sound_names)
        self._configure_scoped_constraints(
            joined_feedback=goal,
            locks_code=locks_code,
            art_change=art_change,
            sound_change=sound_change,
        )
        self._trace({
            "kind": "initial_goal_scoping_applied",
            "locks_code": locks_code,
            "art_change": art_change,
            "sound_change": sound_change,
            "scoped_mode": (self._scoped_constraints or {}).get("mode"),
        })
        # Prepend a compact SCOPED-CHANGE notice so the model sees the
        # narrowed contract right at iter 1. Full SCOPED-CHANGE block
        # also fires automatically on any subsequent feedback turn.
        scoped_mode = (self._scoped_constraints or {}).get("mode") or "single_patch"
        mode_label = (
            "MEDIA-ONLY (existing names only)"
            if scoped_mode == "media_only"
            else "ONE-SMALL-PATCH (no rewrite, max one patch block)"
        )
        scoped_notice = (
            "================ INITIAL-GOAL SCOPE LOCK ================\n"
            "Your goal above carries strict-scope language. Treat this\n"
            "entire session as scoped:\n"
            f"  - Turn mode: {mode_label}\n"
            "  - Address ONLY what the goal explicitly names.\n"
            "  - Do NOT touch unrelated functions, variables, or files.\n"
            "  - Do NOT refactor or 'improve' anything not asked for.\n"
            "==========================================================="
        )
        return f"{scoped_notice}\n\n{build_msg}"

    def _derive_allowed_forbidden_tags(self) -> tuple[list[str], list[str]]:
        """Compute the turn's allowed/forbidden output tags from the
        current scoped/rewrite state. Used by the `turn_contract`
        trace event for visibility; does NOT enforce — enforcement
        lives in `_scoped_reply_violation` and downstream materializer.
        """
        scoped = self._scoped_constraints or {}
        mode = scoped.get("mode")
        if mode == "media_only":
            return (
                ["<assets>", "<sounds>"],
                ["<patch>", "<html_file>"],
            )
        if mode == "single_patch":
            return (
                ["<patch>"],
                ["<html_file>"],
            )
        # No scoped lock: default workflow rules.
        if self._snapshot_n == 0:
            # First build: complete file expected.
            return (["<html_file>"], [])
        # Mid-session: patches preferred; rewrite only when armed.
        # Phase 2: also allow rewrite when the on-disk baseline is itself
        # degenerate — `_materialize` already opens this carve-out via
        # `_is_degenerate_baseline`, but `turn_contract` was reporting
        # rewrite_allowed=False which left the model guessing why its
        # rewrites were getting rejected. Surface the carve-out here so
        # the model and the trace agree.
        allowed = ["<patch>"]
        forbidden: list[str] = []
        rewrite_via_arm = bool(self._allow_one_rewrite)
        rewrite_via_degenerate = False
        try:
            rewrite_via_degenerate = bool(
                self._current_file
                and _is_degenerate_baseline(self._current_file)
            )
        except Exception:
            rewrite_via_degenerate = False
        if rewrite_via_arm or rewrite_via_degenerate:
            allowed.append("<html_file>")
        else:
            forbidden.append("<html_file>")
        return (allowed, forbidden)

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

    def _trace(self, obj: dict) -> None:
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # Add dynamic multi-agent telemetry role & name metadata
            if "model_role" not in obj and "model_name" not in obj:
                role = getattr(self, "_last_stream_role", "coder")
                backend = self.get_backend(role)
                if backend:
                    obj["model_role"] = role
                    obj["model_name"] = backend.info.model
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

    def _save_snapshot(self, html: str) -> Path | None:
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_n += 1
            p = self.snapshots_dir / f"iter_{self._snapshot_n:02d}.html"
            p.write_text(html, encoding="utf-8")
            return p
        except Exception:
            return None

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

    def _summarize_content(self, c: str) -> str:
        """Replace embedded HTML blobs with size markers — keep tags + notes.

        Wolfenstein 2026-05-24 trace lesson: the OLD marker
        `<html_file>[omitted: N bytes of HTML; see snapshot]</html_file>`
        was shaped like a valid output tag, so a confused model parroted
        it back verbatim in its next reply (turns [02], [06], [08] of
        that trace's conversation.md). The parser saw an `<html_file>`
        wrapper around prose, the body didn't normalize to a document,
        materialize failed, and the agent entered an identical-reply
        loop sending the generic "no <patch> or <html_file>" fallback.

        The new marker uses an HTML COMMENT — no `<html_file>` /
        `<patch>` / markdown-fence wrapper. The model literally cannot
        copy-paste it as a fresh tag emission, and the embedded
        instruction tells the model what to do instead of just naming
        a byte count. Universal: no goal text, no genre.
        """
        def html_repl(m):
            n = len(m.group(1))
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
            # Marker intentionally uses no `<html_file>` / `<patch>`
            # substrings (not even inside prose) so a stressed model
            # has nothing to copy-paste as a fresh tag emission. The
            # comment shape also can't extract through the html
            # regex variants 1-6 since none of them match comments.
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-HTML: {n} bytes of HTML "
                f"body written to disk in this earlier turn. The "
                f"current file is shown inline below in the CURRENT "
                f"FILE ON DISK block; patch against that, do NOT "
                f"re-emit this marker. -->"
            )

        def fence_repl(m):
            body = m.group(1)
            n = len(body)
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
            # Fix-round item 4: content sniff. The regex's optional `html`
            # tag means it also matches ```js / ``` fences — and worse, a
            # CLOSING fence followed by the NEXT opening fence, eliding the
            # PROSE between them and falsely asserting it was HTML written
            # to disk (trace 20260610_185238 turn [08] had the marker
            # spliced mid-sentence between two ```js reasoning fences).
            # Only elide when the body actually looks like an HTML document.
            body_head = body.lstrip()[:32].lower()
            if not (body_head.startswith("<!doctype") or body_head.startswith("<html")):
                return m.group(0)
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-FENCE: {n} bytes of "
                f"fenced HTML written to disk in this earlier turn. "
                f"Do NOT re-emit this marker; patch against the "
                f"CURRENT FILE ON DISK block below. -->"
            )

        def probes_repl(m):
            body = m.group(1)
            if len(body) < _SUMMARIZE_MIN_PROBES_BYTES:
                return m.group(0)
            # Rough def count: each probe object carries a "name" key.
            n_defs = body.count('"name"')
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-PROBES: {n_defs} probe defs "
                f"from this earlier turn are superseded — the current "
                f"probe set lives in session state and runs every iter. "
                f"Do NOT re-emit this marker. -->"
            )

        c = self._SUMMARIZE_HTML_RE.sub(html_repl, c)
        c = self._SUMMARIZE_FENCE_RE.sub(fence_repl, c)
        c = self._SUMMARIZE_PROBES_RE.sub(probes_repl, c)
        return c

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

    def _maybe_reset_continuation_context(self, prior_clean: bool) -> bool:
        """Fresh-context continuation (frontier-agent pattern, 2026-06-12).

        Replace the accumulated history with [system prompt, state
        anchor] before the continuation message is appended. Gated:
          - prior session ended clean (mid-debugging history is never
            discarded),
          - a real system prompt sits at index 0,
          - enough history to be worth replacing (> 3 messages).
        Falls back to plain append (returns False) on any anchor-build
        failure. Only changes what we SEND — never cuts model output.
        Returns True when the reset was applied.
        """
        if not (
            prior_clean
            and len(self._messages) > 3
            and (self._messages[0].get("role") == "system")
        ):
            return False
        try:
            before_msgs = len(self._messages)
            before_chars = sum(
                len(str(m.get("content") or "")) for m in self._messages
            )
            summary = self._build_structured_summary()
            anchor_msg = {
                "role": "user",
                "content": (
                    "================ STATE ANCHOR (fresh-context "
                    "continuation) ================\n"
                    "The previous build passed its tests and shipped. "
                    "Older turns were replaced by the snapshot below — "
                    "treat it as authoritative for goal, criteria, "
                    "progress, and critical context. The user's NEW "
                    "request follows in the next message.\n\n"
                    f"{summary}\n"
                    "==========================================================="
                ),
            }
            self._messages = [self._messages[0], anchor_msg]
            after_chars = sum(
                len(str(m.get("content") or "")) for m in self._messages
            )
            self._trace({
                "kind": "continuation_context_reset",
                "before_messages": before_msgs,
                "after_messages": len(self._messages),
                "before_chars": before_chars,
                "after_chars": after_chars,
                # ~4 chars/token heuristic, telemetry only.
                "est_tokens_saved": max(
                    0, (before_chars - after_chars) // 4
                ),
            })
            return True
        except Exception as e:
            # Anchor build failed — keep the appended-history behavior;
            # never block the continuation itself.
            self._trace({
                "kind": "continuation_context_reset_failed",
                "err": str(e)[:180],
            })
            return False

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

        # Token-aware gate: only do the LOSSY structured compaction when the
        # context window is actually filling (last coder prompt >=
        # _COMPACT_PRESSURE of num_ctx), OR as a hard safety cap when token
        # stats are unavailable. A high message count alone no longer triggers
        # it — so a big-context local model keeps full history (playbook +
        # every prior user ask) until the window is genuinely under pressure.
        pressure = float(getattr(self, "_last_prompt_pressure", 0.0) or 0.0)
        last_tokens = int(getattr(self, "_last_prompt_tokens", 0) or 0)
        # Forward-looking projection (2026-06-12, trace 20260612_132314):
        # the reactive gates above look at the PREVIOUS stream's prompt
        # size, so a single-turn jump slips through — a post-clean turn
        # (full file inline + playtest feedback + history) ballooned
        # 33K → 60K in one assembly step and the 27B silently emitted 0
        # tokens for 184s. _prune_messages runs AFTER the new user turn
        # is appended, so estimating the message list (~4 chars/token)
        # projects the prompt we are ABOUT to send. Images aren't
        # counted — this is a floor estimate, which is the safe side.
        projected_tokens = sum(
            len(str(m.get("content") or "")) for m in self._messages
        ) // 4
        compact_reason = None
        if pressure >= _COMPACT_PRESSURE:
            compact_reason = "token_pressure"
        elif _COMPACT_TOKEN_CEILING > 0 and last_tokens >= _COMPACT_TOKEN_CEILING:
            # Fix-round item 3: absolute ceiling — local backends stall
            # silently well before the relative pressure gate fires.
            compact_reason = "token_ceiling"
        elif _COMPACT_TOKEN_CEILING > 0 and projected_tokens >= _COMPACT_TOKEN_CEILING:
            compact_reason = "projected_ceiling"
        elif getattr(self, "_force_compact_after_stall", False):
            # Fix-round item 3: a silent 0-token stall just aborted — do NOT
            # rebuild the same giant prompt for the retry; compact first.
            compact_reason = "silent_stall_recovery"
        elif n > _COMPACT_MESSAGE_CAP:
            compact_reason = "count_cap"
        # One-shot flag: consumed whether or not it was the deciding reason.
        self._force_compact_after_stall = False

        if compact_reason is not None:
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
            # Post-compaction projection so future trace mining can tell
            # whether compaction actually got the next prompt under the
            # ceiling (it may not when the bulk lives in the kept-recent
            # window, e.g. a fresh full-file inline).
            projected_after = sum(
                len(str(m.get("content") or "")) for m in new_messages
            ) // 4
            self._trace({
                "kind": "structured_compaction",
                "reason": compact_reason,
                "prompt_tokens": getattr(self, "_last_prompt_tokens", 0),
                "projected_tokens": projected_tokens,
                "projected_tokens_after": projected_after,
                "num_ctx": getattr(self, "num_ctx", 0),
                "pressure": round(pressure, 3),
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
        # Capability-round item 3: the NEWEST wrapped test report always
        # stays verbatim, even if it has fallen below the keep window.
        newest_report_idx = -1
        for i, m in enumerate(self._messages):
            if (
                m.get("role") == "user"
                and _REPORT_BLOCK_BEGIN in (m.get("content") or "")
            ):
                newest_report_idx = i
        collapsed_reports = 0
        for i in range(1, cutoff):
            msg = self._messages[i]
            # Do NOT rewrite user/system turns here. Mutating prior user
            # instructions (especially format examples) creates false context
            # and can derail one-shot generations.
            # EXCEPTION (item 3): turns wrapped in HARNESS-REPORT-BLOCK
            # sentinels are harness-authored test reports, not user
            # instructions. Superseded ones collapse to their 3-line
            # digest; feedback appended outside the wrapper survives.
            if msg.get("role") == "user":
                c = msg.get("content", "") or ""
                if _REPORT_BLOCK_BEGIN in c and i != newest_report_idx:
                    def _collapse(m_):
                        return (
                            "[superseded test report — collapsed by "
                            "harness; the newest report below is the "
                            "current truth]\n"
                            + m_.group(1).strip()
                        )
                    new_c = _REPORT_BLOCK_RE.sub(_collapse, c)
                    if new_c != c:
                        msg["content"] = new_c
                        collapsed_reports += 1
                continue
            if msg.get("role") != "assistant":
                continue
            c = msg.get("content", "") or ""
            new_c = self._summarize_content(c)
            if new_c != c:
                msg["content"] = new_c
        if collapsed_reports:
            self._trace({
                "kind": "report_turns_collapsed",
                "count": collapsed_reports,
                "newest_kept_idx": newest_report_idx,
            })

    # -- user-injection plumbing -------------------------------------------

    def _clear_scoped_constraints(self) -> None:
        self._scoped_constraints = None
        self._pending_scoped_check_keywords = []

    def _previous_iter_clean_for_scope_guard(self) -> bool:
        prev = self._previous_report or {}
        probes = prev.get("probes") or []
        probes_all_pass = bool(probes) and all(bool(p.get("ok")) for p in probes)
        return (
            self._previous_report_ok is True
            and probes_all_pass
            and not (prev.get("errors") or [])
            and not (prev.get("soft_warnings") or [])
            and not (prev.get("page_errors") or [])
            and not (prev.get("console_errors") or [])
        )

    def _configure_scoped_constraints(
        self,
        *,
        joined_feedback: str,
        locks_code: bool,
        art_change: bool,
        sound_change: bool,
    ) -> None:
        """Persist deterministic scoped routing for the next model reply."""
        if not locks_code:
            self._clear_scoped_constraints()
            return
        behavior_scope = _feedback_mentions_scoped_behavior_change(joined_feedback)
        size_scope = _feedback_requests_size_change(joined_feedback)
        # Preserve working baselines on strict scoped tweaks: default to one
        # small patch unless this is an explicit style-only media request.
        media_only = bool(art_change or sound_change) and not behavior_scope and not size_scope
        mode = "media_only" if media_only else "single_patch"
        probe_keywords = (
            _scoped_probe_keywords(joined_feedback)
            if mode == "single_patch" and behavior_scope
            else []
        )
        self._scoped_constraints = {
            "mode": mode,
            "max_patch_count": 1,
            "allowed_asset_names": sorted(self._session_assets.keys()),
            "allowed_sound_names": sorted(self._session_sounds.keys()),
            "media_name_lock": mode == "media_only",
            "require_scope_probe": bool(probe_keywords),
            "probe_keywords": probe_keywords,
            "preserve_baseline": self._previous_iter_clean_for_scope_guard(),
            "feedback_preview": joined_feedback[:220],
        }
        self._pending_scoped_check_keywords = list(probe_keywords)
        self._trace({
            "kind": "scoped_constraints_set",
            "mode": mode,
            "max_patch_count": 1,
            "require_scope_probe": bool(probe_keywords),
            "probe_keywords": probe_keywords,
            "media_name_lock": mode == "media_only",
            "preserve_baseline": self._scoped_constraints["preserve_baseline"],
            "art_change": bool(art_change),
            "sound_change": bool(sound_change),
            "behavior_scope": behavior_scope,
            "size_scope": size_scope,
        })

    def _scoped_reply_violation(self, reply: str) -> str | None:
        """Return a compact deterministic violation message, else None."""
        cfg = self._scoped_constraints
        if not cfg:
            return None
        stripped = (reply or "").lstrip()
        if not stripped:
            return "SCOPED FORMAT: empty reply; emit required tag only."
        if stripped.lower().startswith("<think"):
            return "SCOPED FORMAT: do not emit <think>; start with required tag."
        if not stripped.startswith("<"):
            return "SCOPED FORMAT: no prose preamble; start with required tag."
        low = stripped.lower()
        has_assets = bool(_ASSETS_OPEN_RE.search(low))
        has_sounds = bool(_SOUNDS_OPEN_RE.search(low))
        patches = extract_patches(reply)
        patch_count = len(patches)
        has_html = self._extract_html(reply) is not None
        mode = str(cfg.get("mode") or "single_patch")
        # Be permissive about which tag the model chose. The classifier
        # that picked `media_only` vs `single_patch` reads the user's
        # English feedback through a regex pattern — it's necessarily
        # incomplete (natural language is unbounded). When the model
        # picked a different tag than we expected, it usually has a
        # reason: e.g., DOOM trace 20260523_171650 iter 4 — user wrote
        # "do not change any graphics... just shift the pistol 15 px
        # right" which scored as art_change → media_only, but the model
        # correctly chose a CSS <patch>. Rejecting that patch with
        # "SCOPED MEDIA: emit <assets>/<sounds> only" sent the model
        # into a doom loop where it regenerated the gun asset with a
        # wrong prompt. Auto-revert is the safety net for genuinely bad
        # patches; the scope gate's only remaining job is to block full
        # <html_file> rewrites on small-scope feedback (a single patch
        # is hard to misjudge in a way auto-revert can't catch).
        if has_html:
            return "SCOPED: full <html_file> rewrite blocked; send one <patch>."
        if not (patch_count or has_assets or has_sounds):
            return "SCOPED: emit one <patch>, or <assets>/<sounds> with existing names."
        max_patch = int(cfg.get("max_patch_count") or 1)
        if patch_count > max_patch:
            # Wording matches the historical "SCOPED PATCH: send exactly
            # one <patch>" message so existing tests keep working.
            return f"SCOPED: send exactly one <patch> (got {patch_count})."
        # Soft trace when the model's tag choice diverges from the
        # classifier's expectation. Useful for postmortem to spot
        # patterns without us having to enumerate them in regex.
        if mode == "media_only" and patch_count > 0 and not (has_assets or has_sounds):
            self._record_classifier_overrule(
                expected_mode=mode,
                model_emitted="patch_only",
                feedback_preview=str(cfg.get("feedback_preview") or "")[:200],
            )
        elif mode == "single_patch" and (has_assets or has_sounds) and patch_count == 0:
            self._record_classifier_overrule(
                expected_mode=mode,
                model_emitted="media_only",
                feedback_preview=str(cfg.get("feedback_preview") or "")[:200],
            )
        if bool(cfg.get("require_scope_probe")):
            probes = self._extract_probes(reply)
            if not probes:
                return "SCOPED CHECK: include one compact <probes> check for requested behavior."
            keys = [str(k).lower() for k in (cfg.get("probe_keywords") or []) if k]
            if keys:
                probe_text = " ".join(
                    f"{p.get('name','')} {p.get('expr','')}".lower()
                    for p in probes
                )
                if not any(k in probe_text for k in keys):
                    want = ", ".join(keys[:3])
                    return (
                        "SCOPED CHECK: probe must mention requested behavior "
                        f"(e.g. {want})."
                    )
        return None

    def _scoped_retry_instruction(self, violation: str) -> str:
        cfg = self._scoped_constraints or {}
        probe_line = ""
        if cfg.get("require_scope_probe"):
            probe_line = " Include one compact <probes> entry verifying the requested behavior."
        return (
            f"{violation}\n"
            "Retry: emit either one small <patch> (preserve the working "
            "baseline — change only the scoped request) OR an <assets>/"
            "<sounds> block re-using EXISTING names."
            f"{probe_line}"
        )

    def _apply_scoped_check_to_report(self, report: dict[str, Any]) -> None:
        keys = [k for k in self._pending_scoped_check_keywords if k]
        if not keys:
            return
        probes = report.get("probes") or []
        matched = [
            p for p in probes
            if any(
                key in (
                    f"{str(p.get('name') or '').lower()} "
                    f"{str(p.get('expr') or '').lower()}"
                )
                for key in keys
            )
        ]
        passed = any(bool(p.get("ok")) for p in matched)
        report["scoped_check"] = {
            "required": True,
            "keywords": keys[:3],
            "matched": len(matched),
            "pass": passed,
        }
        if not passed:
            sw = list(report.get("soft_warnings") or [])
            sw.append(
                "SCOPED CHECK FAILED: requested behavior change was not "
                "verified by a passing scoped probe."
            )
            report["soft_warnings"] = sw
            report["ok"] = False

    def _apply_dead_animation_check_to_report(self, report: dict[str, Any]) -> None:
        """Surface dead animation frames as ADVISORY — never a hard ok=False gate.

        A `from_image` frame that came back near-identical to its idle parent
        (tracked in `self._dead_anim_frames`) means the limbs never moved — the
        character will just slide as a static image. That's worth telling the
        model, but it must NOT block shipping or starve the session.

        WHY ADVISORY, NOT BLOCKING (changed 2026-06-01, trace
        build-a-single-screen-2d-fight_20260531_214215 run_…214215): this used
        to flip report["ok"]=False, which created an UNWINNABLE loop. The dojo-
        fight session — with BOTH a local model (qwen3.6-27B) and SOTA
        (Opus 4.8) — corrected the actual gameplay perfectly (patches applied
        4/4 then 3/3, behavioral probes 8/8, input test PASS) yet every iter
        stayed ok=False on this one cosmetic sprite warning. Two compounding
        traps made it impossible to clear:
          1. The prescribed fix (re-emit `from_image` strength 0.55-0.65) is
             the SAME img2img path the user's own A/B finding documents as
             BROKEN — "pose frames must be TXT2IMG, not img2img" — so the regen
             came back dead again and RE-armed the block.
          2. While ok stayed False, the user's real gameplay feedback (slow the
             animation, flip the CPU facing) was deferred behind this blocker
             (`_should_defer_feedback_for_blocker`), so the model was never even
             allowed to make the simple code fix the user asked for.
        A cosmetic asset-quality signal the model cannot reliably fix must not
        gate a behaviorally-correct build. Behavioral probes gate; cosmetics
        inform. The signal still reaches the model three other ways that do NOT
        hard-fail: the rendered `warnings` block, the coaching channel, and the
        VLM critic note (see `_build_visual_critic_*`). The set still clears
        automatically when a frame regenerates with a real delta.
        """
        dead = self._dead_anim_frames
        if not dead:
            return
        names = ", ".join(f"`{n}`" for n in sorted(dead))
        # `warnings` (NOT `soft_warnings`) is the non-gating channel: tools.py
        # computes ok = (no errors) and (no soft_warnings), so anything in
        # soft_warnings is a hard fail. Route this advisory to `warnings` so it
        # is shown to the model without flipping ok. Do NOT touch report["ok"].
        #
        # Do NOT tell the model to "regenerate the pose frame" here. Both
        # regeneration routes are dead ends and suggesting either wastes the
        # correction loop (see CLAUDE.md "Things to avoid" + the user memory
        # feedback_sprite_animation_from_image.md):
        #   - img2img can't change a pose at all (guidance_scale=0 keeps it
        #     locked to idle — proven A/B 2026-05-30), and
        #   - a fresh txt2img replacement frame will NOT stay consistent with
        #     the character already placed in the running game (per the user:
        #     consistency is the hard constraint), and in the dojo-fight trace
        #     the txt2img path STILL returned near-idle `block` frames anyway.
        # So this is purely informational: name the dead frames, say it's
        # cosmetic, and let the behaviorally-correct build ship.
        warns = list(report.get("warnings") or [])
        warns.append(
            "DEAD ANIMATION (advisory — does not block shipping): these frames "
            f"came back near-identical to their idle parent — {names}. Their "
            "limbs barely move, so the character looks near-static for that "
            "pose. This is a cosmetic sprite-quality note, NOT a code bug: do "
            "not change game logic for it and do not try to regenerate the "
            "frames in your patch. It will not stop a behaviorally-correct game "
            "from shipping."
        )
        report["warnings"] = warns
        report["dead_anim_frames"] = dict(dead)

    @staticmethod
    def _probe_shape_key(p: dict[str, Any]) -> str:
        name = str(p.get("name") or "probe").strip().lower()
        expr = str(p.get("expr") or "").strip().lower()
        err_class = str(p.get("error_class") or "eval_error").strip().lower()
        return f"{name}:{err_class}:{expr[:80]}"

    @staticmethod
    def _refresh_probe_error_fields(report: dict[str, Any]) -> None:
        probes = list(report.get("probes") or [])
        report["probe_errors"] = [
            f"{p.get('name','?')}: {p.get('err','')[:160]}"
            for p in probes
            if not p.get("ok") and p.get("err")
        ]
        report["probe_eval_errors"] = [
            {
                "name": p.get("name", "probe"),
                "expr_preview": (p.get("expr") or "")[:120],
                "error_class": p.get("error_class") or "eval_error",
                "err": (p.get("err") or "")[:200],
            }
            for p in probes
            if not p.get("ok") and p.get("kind") == "eval_error"
        ]

    @staticmethod
    def _remove_probe_warnings(report: dict[str, Any], names: set[str]) -> None:
        if not names:
            return
        kept: list[str] = []
        for w in list(report.get("soft_warnings") or []):
            drop = False
            for name in names:
                if (
                    f"PROBE FAILED [{name}]" in w
                    or f"PROBE BROKEN [{name}]" in w
                ):
                    drop = True
                    break
            if not drop:
                kept.append(w)
        report["soft_warnings"] = kept

    def _handle_probe_eval_errors(self, report: dict[str, Any], iteration: int) -> None:
        """Trace, soften, and quarantine probes that fail at eval time.

        Falsy probes are still game failures. Eval-error probes are broken
        probe expressions; after two consecutive eval errors for the same
        name, drop the probe from future iterations so it stops drowning the
        model in stale harness errors.
        """
        probes = list(report.get("probes") or [])
        if not probes:
            return

        failing = [p for p in probes if not p.get("ok")]
        eval_failing = [
            p for p in failing
            if p.get("kind") == "eval_error"
        ]

        for p in probes:
            name = str(p.get("name") or "probe")
            if p.get("ok"):
                self._probe_names_ever_passed.add(name)
                self._probe_eval_error_streak[name] = 0
                self._probe_eval_error_shape_streak[self._probe_shape_key(p)] = 0
                continue
            if p.get("kind") == "eval_error":
                shape = self._probe_shape_key(p)
                self._probe_eval_error_streak[name] = (
                    self._probe_eval_error_streak.get(name, 0) + 1
                )
                self._probe_eval_error_shape_streak[shape] = (
                    self._probe_eval_error_shape_streak.get(shape, 0) + 1
                )
                self._trace({
                    "kind": "probe_eval_error",
                    "iteration": iteration,
                    "name": name,
                    "expr_preview": (p.get("expr") or "")[:120],
                    "error_class": p.get("error_class") or "eval_error",
                    "streak": self._probe_eval_error_streak[name],
                    "shape_streak": self._probe_eval_error_shape_streak[shape],
                })
            else:
                # A real falsy result breaks the eval-error streak.
                self._probe_eval_error_streak[name] = 0

        # Two-strike rule for general eval errors, ONE-STRIKE for
        # SyntaxError. Syntax errors mean the probe expression itself
        # is malformed JavaScript — re-evaluating it next iter cannot
        # produce a different result; the only way it would "pass" is
        # if the model re-emits a corrected probe. Holding the broken
        # probe in the active set for a second iter just lets the
        # syntax error mask other harness signals (the 2026-05-23
        # chess trace's iter 2/3 chased a `new Promise(r=>{...})`
        # parser error for two iters while the actual
        # ASSETS_LOADED_BUT_UNDRAWN problem was hidden behind it).
        # Model-agnostic: any model that emits unparseable JS triggers
        # the fast-path; legitimate runtime-eval errors still get the
        # tolerant 2-strike treatment.
        def _is_syntax_error(p: dict) -> bool:
            err_class = (p.get("error_class") or "").lower()
            err_text = (p.get("err") or "").lower()
            return (
                "syntaxerror" in err_class
                or "syntaxerror" in err_text
                or "unexpected token" in err_text
                or "unexpected identifier" in err_text
                or "unterminated" in err_text
            )
        quarantine_names: set[str] = set()
        for p in eval_failing:
            name = str(p.get("name") or "probe")
            streak = self._probe_eval_error_streak.get(name, 0)
            if streak >= 2 or _is_syntax_error(p):
                quarantine_names.add(name)
        if quarantine_names:
            for p in eval_failing:
                name = str(p.get("name") or "probe")
                if name not in quarantine_names:
                    continue
                is_syntax = _is_syntax_error(p)
                self._trace({
                    "kind": "probe_quarantined",
                    "iteration": iteration,
                    "name": name,
                    "expr_preview": (p.get("expr") or "")[:120],
                    "eval_error_class": p.get("error_class") or "eval_error",
                    "streak": self._probe_eval_error_streak.get(name, 0),
                    "fast_path": "syntax_error" if is_syntax else None,
                })
                if is_syntax:
                    notice = (
                        f"PROBE {name} QUARANTINED IMMEDIATELY: the "
                        "expression is unparseable JavaScript (syntax "
                        "error). Re-emit <probes>...</probes> with a "
                        "corrected expression — re-running malformed "
                        "JS cannot produce a different result and "
                        "would mask other harness signals."
                    )
                else:
                    notice = (
                        f"PROBE {name} QUARANTINED: had eval errors twice; "
                        "re-emit <probes>...</probes> to replace, or leave it "
                        "dropped if it was not testing real behavior."
                    )
                if notice not in self._pending_probe_quarantine_notices:
                    self._pending_probe_quarantine_notices.append(notice)

            self._probes = [
                p for p in self._probes
                if str(p.get("name") or "probe") not in quarantine_names
            ]
            report["probes"] = [
                p for p in probes
                if str(p.get("name") or "probe") not in quarantine_names
            ]
            self._remove_probe_warnings(report, quarantine_names)
            warnings = list(report.get("warnings") or [])
            warnings.append(
                "Probe quarantine: dropped "
                f"{len(quarantine_names)} probe(s) after repeated eval-time "
                "errors; re-emit <probes> to replace if needed."
            )
            report["warnings"] = warnings
            self._refresh_probe_error_fields(report)

        # C1: before a probe is quarantined, avoid blocking on eval-error-only
        # probes when the rest of the harness says the game is healthy.
        remaining = list(report.get("probes") or [])
        remaining_failing = [p for p in remaining if not p.get("ok")]
        if remaining_failing and all(
            p.get("kind") == "eval_error" for p in remaining_failing
        ):
            input_test = report.get("input_test") or {}
            canvas = report.get("canvas") or {}
            healthy_page = not (report.get("page_errors") or report.get("console_errors"))
            healthy_harness = (
                input_test.get("ran") is True
                and input_test.get("any_change") is True
                and canvas.get("blank") is False
                and healthy_page
            )
            proven_probe_bug = all(
                (
                    str(p.get("name") or "probe") in self._probe_names_ever_passed
                    or self._probe_eval_error_shape_streak.get(self._probe_shape_key(p), 0) >= 2
                )
                for p in remaining_failing
            )
            if healthy_harness and proven_probe_bug:
                softened = {str(p.get("name") or "probe") for p in remaining_failing}
                for p in remaining_failing:
                    p["ok"] = True
                    p["downgraded"] = (
                        "probe eval error softened: input/canvas/page checks "
                        "were healthy and this probe shape repeatedly failed "
                        "before evaluating"
                    )
                self._remove_probe_warnings(report, softened)
                warnings = list(report.get("warnings") or [])
                warnings.append(
                    "Probe eval errors softened: all remaining probe failures "
                    "were eval-time errors while input/canvas/page checks were healthy."
                )
                report["warnings"] = warnings
                self._refresh_probe_error_fields(report)

        report["ok"] = (
            len(report.get("errors") or []) == 0
            and len(report.get("soft_warnings") or []) == 0
        )

    def _consumed_feedback_summary(self) -> str | None:
        if self._should_defer_feedback_for_blocker():
            return (
                "→ queued your input, but deferring it until the current "
                "test blocker is fixed."
            )
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

    # Phrases that count as a user override of the blocker-first deferral.
    # The narrow explicit forms ("ignore the test", "ship as-is") are joined
    # by natural-language signals — the chess trace had the user say "the
    # game works great" / "the game plays fine" / "do not change the GAME"
    # three times and none matched the original regex, so all three asks
    # were deferred. False positives here are safe: an override only stops
    # the deferral, it does not silence the test report the model still
    # sees. False negatives are the actual user-facing failure.
    _BLOCKER_OVERRIDE_RE = re.compile(
        r"\b("
        r"ignore\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"override\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"abandon\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"skip\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"ship\s+(?:it\s+)?(?:as[- ]is|anyway)|"
        r"ship\s+the\s+partial|"
        r"accept\s+(?:the\s+)?(?:known\s+)?failure|"
        r"(?:the\s+)?(?:game|it)\s+(?:works|plays|runs)"
        r"(?:\s+(?:great|fine|well|now|again|already|just\s+fine))?|"
        r"(?:the\s+)?(?:game|it)\s+is\s+(?:fine|great|working|playing|ok|okay)|"
        r"do(?:es)?\s*n[o']?t?\s+change\s+(?:the\s+)?"
        r"(?:game|code|behavior|gameplay|logic|mechanics)|"
        r"(?:leave|keep)\s+(?:the\s+)?(?:game|code|gameplay)\s+"
        r"(?:as[- ]is|alone|the\s+same)|"
        r"perfectly\s+working\s+(?:game|build)|"
        r"working\s+(?:perfectly|fine|great)"
        r")\b",
        re.I,
    )

    def _has_active_blocker(self) -> bool:
        """True when the previous tested build failed and the next turn is a fix."""
        return bool(self._fix_mode and self._previous_report_ok is False)

    def _should_defer_feedback_for_blocker(self) -> bool:
        """Keep fresh feedback from distracting a failing repair turn.

        The queue is left intact so the exact feedback is applied after a clean
        report. Only explicit "ignore/ship anyway" style overrides are allowed
        through while a blocker is active.
        """
        if not (self._pending_feedback and self._has_active_blocker()):
            return False
        joined = "\n".join(self._pending_feedback)
        return self._BLOCKER_OVERRIDE_RE.search(joined) is None

    @staticmethod
    def _feedback_shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(words) < n:
            return {tuple(words)} if words else set()
        return {tuple(words[i:i + n]) for i in range(0, len(words) - n + 1)}

    @staticmethod
    def _feedback_keywords(text: str) -> set[str]:
        stop = {
            "the", "a", "an", "to", "and", "or", "but", "it", "is", "are",
            "be", "of", "for", "with", "when", "still", "need", "needs",
            "make", "also", "not", "just", "this", "that", "their",
        }
        return {
            w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in stop
        }

    def _deferral_signature(self, text: str) -> str:
        """Compact keyword signature for cross-turn deferral matching.

        Keyword bag drops stop-words and length-2 tokens (`_feedback_keywords`).
        Sorted + joined into a stable key. Two phrasings of the same ask
        ("make new pawn frames" vs "more frames of the pawn please") will
        produce highly overlapping signatures.
        """
        kw = self._feedback_keywords(text)
        return " ".join(sorted(kw))

    def _count_recent_deferrals(self, text: str) -> int:
        """How many recent deferrals share intent with `text`."""
        if not self._recent_deferred_signatures:
            return 0
        cur_set = set(self._deferral_signature(text).split())
        if not cur_set:
            return 0
        count = 0
        for prev_sig in self._recent_deferred_signatures:
            prev_set = set(prev_sig.split())
            if not prev_set:
                continue
            union = cur_set | prev_set
            overlap = len(cur_set & prev_set) / max(1, len(union))
            if overlap >= 0.5:
                count += 1
        return count

    def _detect_repeated_feedback(self, text: str) -> dict | None:
        """Return overlap metadata when latest feedback repeats a recent ask."""
        cur = self._feedback_shingles(text)
        cur_kw = self._feedback_keywords(text)
        if not cur:
            return None
        best: tuple[float, int, set[str]] | None = None
        for idx, prev in enumerate(self._recent_feedback_texts[-2:]):
            prev_set = self._feedback_shingles(prev)
            if not prev_set:
                continue
            union = cur | prev_set
            shingle_overlap = len(cur & prev_set) / max(1, len(union))
            prev_kw = self._feedback_keywords(prev)
            kw_union = cur_kw | prev_kw
            kw_overlap = len(cur_kw & prev_kw) / max(1, len(kw_union))
            overlap = max(shingle_overlap, kw_overlap)
            shared = {" ".join(s) for s in (cur & prev_set)}
            shared.update(cur_kw & prev_kw)
            if best is None or overlap > best[0]:
                # idx here is relative to the final two entries.
                best = (overlap, len(self._recent_feedback_texts[-2:]) - idx, shared)
        if not best or best[0] < 0.5:
            return None
        shared_words: list[str] = []
        for term in sorted(best[2])[:5]:
            if term:
                shared_words.append(term)
        return {
            "overlap": round(best[0], 3),
            "overlap_with_recent_turns_ago": best[1],
            "shared_terms": shared_words,
        }

    def _maybe_clear_asset_reprompt_via_code(self, reply: str) -> None:
        """Stand down the ASSET GENERATION REQUIRED reprompt when the
        model substantively addressed the request in CODE.

        Evidence (trace 20260612_132314): "need to improve animation for
        jump,duck, left and right" hard-classified as an art request, but
        the model — correctly, since img2img cannot change a pose —
        shipped procedural-animation patches. The reprompt only cleared
        on an <assets> block, so it nagged for 3 turns and gave up.

        Heuristic: the reply's applied patch bodies share subject
        keywords with the request (>= 2 terms, or all of them for a
        one/two-keyword request). Replies that neither emit <assets> nor
        touch the requested subject keep the reprompt — the original
        here-s-a-tight-test failure mode stays covered.
        """
        req = getattr(self, "_unhonored_asset_request", None)
        if not req:
            return
        try:
            patch_list = extract_patches(reply or "")
        except Exception:
            patch_list = []
        if not patch_list:
            return
        req_kw = self._feedback_keywords(req)
        if not req_kw:
            return
        patch_text = "\n".join(
            f"{p.search or ''}\n{p.replace or ''}" for p in patch_list
        )
        patch_kw = self._feedback_keywords(patch_text)
        overlap = req_kw & patch_kw
        if len(overlap) >= min(2, len(req_kw)):
            self._unhonored_asset_request = None
            self._asset_reprompt_count = 0
            self._trace({
                "kind": "asset_request_honored_via_code",
                "request": req[:200],
                "matched_terms": sorted(overlap)[:8],
                "patch_count": len(patch_list),
            })

    @staticmethod
    def _is_post_clean_instruction(base_message: str) -> bool:
        """True when `base_message` is the clean-report post-clean prompt.

        Fresh user feedback after a clean pass should not drag a full test
        report and stale "prefer <done/>" text into the next model turn.
        The local MK trace showed this confusing small models after the
        user asked for one narrow missing-animation fix.
        """
        if not base_message:
            return False
        return (
            "No errors. The game works. STRONGLY prefer ending with <done/>"
            in base_message
        )

    def _compact_post_clean_context(self) -> str:
        prev = self._previous_report or {}
        probes = prev.get("probes") or []
        passed = sum(1 for p in probes if p.get("ok"))
        total = len(probes)
        return (
            "PREVIOUS BUILD WAS CLEAN: "
            f"{passed}/{total} probes passed, no page errors, no console errors. "
            "Treat that version as the baseline."
        )

    def _flush_user_injections(self, base_message: str) -> str:
        """Main router: drain queued user input, inject directives, and
        record the routing decisions consumed by `_stream` to emit one
        `turn_contract` trace row per stream. False positives in the
        classifier helpers (`_feedback_is_art_change`,
        `_feedback_is_sound_change`, `_feedback_is_orientation_change`)
        flip allowed/forbidden output tags and are expensive — keep
        them genre-free and prefer false negatives over false positives.
        """
        parts: list[str] = []
        self._last_drained_feedback = []
        # Reset routing record so a turn without feedback shows up as
        # "no feedback" rather than inheriting the previous turn's flags.
        # Fields are filled in below as decisions are made.
        self._last_turn_contract = {
            "had_feedback": False,
            "locks_code": False,
            "art_change": False,
            "sound_change": False,
            "behavior_scope": False,
            "behavior_bug": False,
            "orientation_change": False,
            "blocker_feedback_deferred": False,
        }
        # Snapshot the queue BEFORE consuming so we can push a visible
        # "✓ APPLIED to this turn" confirmation into the agent log via
        # the TUI's token callback. Without this, only the right-hand
        # status panel reflects the queue draining — the left-hand log
        # (where the user's eye lives) shows nothing, leaving them
        # uncertain whether their typing actually reached the model.
        # Phase 0.1 — partition feedback before the deferral decision so
        # MEDIA-ONLY items (art/sound regen, animation frame chains via
        # `from_image`) bypass the blocker-first deferral. They run on
        # GPU 0 / Z-Image / Stable-Audio / SD-Turbo; the coder runs on
        # slot 1. They don't compete. The 2026-05-22 chess trace had the
        # user ask three times for img2img animation frames from existing
        # piece sprites; all three were deferred behind an irrelevant
        # `kbCursor` draw warning and the session shipped without ever
        # honoring the asset ask. See memory `feedback-media-requests-
        # never-defer.md`.
        defer_predicate = self._should_defer_feedback_for_blocker()
        _session_assets = getattr(self, "_session_assets", None) or {}
        _session_sounds = getattr(self, "_session_sounds", None) or {}
        asset_names = list(_session_assets.keys())
        sound_names = list(_session_sounds.keys())
        # GENERAL (2026-05-31, genre/model-agnostic): an explicit "generate new
        # art" request only produces sprites if the model emits an <assets>
        # block. A model distracted by a test blocker — or in raw-feedback mode
        # where the media wrapper is suppressed — often replies with code or a
        # diagnosis and NO <assets>, so the art never gets made (here-s-a-tight
        # -test 20260530: asked twice for a red fighter, zero new assets). Track
        # the outstanding request and re-assert "emit <assets>" each turn until
        # the model actually emits one (capped). This is a machine-level
        # directive (NOT suppressed by raw mode) and survives the blocker; it
        # is cleared in _maybe_generate_assets_and_sounds the moment an <assets>
        # block is parsed.
        # getattr defaults so partially-constructed test stubs (which skip
        # __init__) don't AttributeError here.
        _unhonored = getattr(self, "_unhonored_asset_request", None)
        _reprompts = getattr(self, "_asset_reprompt_count", 0)
        # Skip AGENT-generated notices (mid-session media loaders etc.) —
        # only genuine USER feedback can arm the unhonored-asset-request
        # detector. Trace 20260612_004616: the agent's own "Mid-session
        # asset/sound/video additions" notice classified as an art change
        # and fired 8 spurious ASSET GENERATION REQUIRED banners.
        _internal_texts = getattr(self, "_internal_feedback_texts", set())
        _art_reqs = [
            fb for fb in self._pending_feedback
            if fb not in _internal_texts
            and _feedback_is_art_change(fb, asset_names)
        ]
        if _art_reqs:
            _unhonored = _art_reqs[-1]
            _reprompts = 0
        if _unhonored and _reprompts < 3:
            _reprompts += 1
            self._unhonored_asset_request = _unhonored
            self._asset_reprompt_count = _reprompts
            parts.append(
                "================ ASSET GENERATION REQUIRED ================\n"
                "The user asked you to GENERATE NEW ART:\n"
                f'  "{_unhonored[:300]}"\n'
                "New art does NOT exist until you emit an <assets> block. THIS "
                "turn, emit <assets> with a NEW `name` for each new sprite and a "
                "SHORT prompt for each (for a recolor/variant, reuse the base "
                "entity's description with the requested change — e.g. a second "
                "fighter that is the same character in a different color, one "
                "entry per pose). You MAY also <patch> the code to load and draw "
                "them. Do NOT reply with only a diagnosis or code — the sprites "
                "will not appear without an <assets> block.\n"
                "==========================================================="
            )
            self._trace({
                "kind": "asset_request_reprompt",
                "attempt": _reprompts,
                "request": _unhonored[:200],
            })
        elif _unhonored and _reprompts >= 3:
            # Gave it 3 turns; stop nagging so the prompt doesn't bloat.
            self._trace({
                "kind": "asset_request_giveup",
                "request": _unhonored[:200],
            })
            self._unhonored_asset_request = None
            self._asset_reprompt_count = 0
        media_to_process_now: list[str] = []
        code_to_defer: list[str] = []
        force_honor_via_escalation: list[str] = []
        if defer_predicate and self._pending_feedback:
            for fb in self._pending_feedback:
                is_art = _feedback_is_art_change(fb, asset_names)
                is_sound = _feedback_is_sound_change(fb, sound_names)
                is_bug = _feedback_is_behavior_bug(fb)
                is_orient = _feedback_is_orientation_change(fb)
                if (is_art or is_sound) and not is_bug and not is_orient:
                    media_to_process_now.append(fb)
                    continue
                # Phase 0.2 — same intent deferred ≥2 times before? Force
                # it through. The model is told the user has asked N
                # times; the test report still appears so the blocker
                # isn't silently dropped, but the user's input is no
                # longer swallowed for a third turn in a row.
                prior_defer_count = self._count_recent_deferrals(fb)
                if prior_defer_count >= 2:
                    force_honor_via_escalation.append(fb)
                    self._trace({
                        "kind": "feedback_deferral_escalated",
                        "prior_defer_count": prior_defer_count,
                        "text": fb[:240],
                    })
                else:
                    code_to_defer.append(fb)
            # Rewrite the queue so the downstream "process feedback" block
            # consumes media items + force-honored items this turn. The
            # plain code items get re-queued at the end of this method so
            # they remain pending for the next turn.
            self._pending_feedback = list(media_to_process_now) + list(force_honor_via_escalation)
            if media_to_process_now:
                self._trace({
                    "kind": "media_only_parallel_inject",
                    "media_count": len(media_to_process_now),
                    "code_deferred_count": len(code_to_defer),
                    "escalated_count": len(force_honor_via_escalation),
                    "asset_names": asset_names,
                    "sound_names": sound_names,
                    "preview": "\n- ".join(media_to_process_now)[:400],
                })
        defer_block_active = defer_predicate and bool(code_to_defer)

        consumed_items: list[str] = []
        if self._pending_answer is not None:
            consumed_items.append(f"answer: {self._pending_answer[:120]}")
        for fb in self._pending_feedback:
            consumed_items.append(f"feedback: {fb[:120]}")

        self._feedback_deferred_last_turn = False
        answer_was_consumed = False
        if self._pending_answer is not None:
            ans = self._pending_answer
            parts.append(
                "================ USER ANSWER (HIGHEST PRIORITY) ================\n"
                f"{ans}\n"
                "================================================================"
            )
            self._trace({"kind": "answer_injected", "text": ans})
            self._pending_answer = None
            answer_was_consumed = True
        if defer_block_active:
            self._feedback_deferred_last_turn = True
            feedback_items = list(code_to_defer)
            preview = "\n- ".join(fb[:240] for fb in feedback_items)
            parallel_note = ""
            if media_to_process_now:
                parallel_note = (
                    "\nNOTE: media-only items in the same user feedback "
                    "(art/sound regen / animation frames) are being processed "
                    "in parallel via the diffuser pipeline on GPU 0 this turn. "
                    "The block above lists ONLY the code-touching items that "
                    "wait on the blocker.\n"
                )
            parts.append(
                "================ BLOCKER-FIRST FEEDBACK DEFERRAL ================\n"
                "The previous browser/micro-probe report is still failing, so\n"
                "code-touching user feedback is queued for after the blocker\n"
                "is clean. THIS turn must fix the failing test report below first.\n"
                f"{parallel_note}"
                f"\nDeferred feedback:\n- {preview}\n"
                "================================================================="
            )
            self._last_turn_contract["blocker_feedback_deferred"] = True
            self._trace({
                "kind": "feedback_deferred_blocker",
                "count": len(feedback_items),
                "previous_report_ok": self._previous_report_ok,
                "fix_mode": self._fix_mode,
                "media_parallel_count": len(media_to_process_now),
                "preview": "\n- ".join(feedback_items)[:400],
            })
            # Phase 0.2 — record each deferred item's signature so the
            # next-turn escalation count is accurate.
            for fb in feedback_items:
                sig = self._deferral_signature(fb)
                if sig:
                    self._recent_deferred_signatures.append(sig)
            # Cap the history at 6 entries — enough to span ~3 ask-defer
            # cycles; older signatures shed naturally.
            self._recent_deferred_signatures = self._recent_deferred_signatures[-6:]
        if force_honor_via_escalation:
            # Phase 0.2 — escalation directive prepends the regular USER
            # FEEDBACK block. The model gets the test report AND the
            # user's persistent ask; it must address both this turn.
            parts.append(
                "================ FEEDBACK ESCALATION (USER REPEATED) ================\n"
                "The user has stated the following request three or more times "
                "across recent turns; it was deferred twice behind a code "
                "blocker. The blocker is real and still appears below — fix it. "
                "AT THE SAME TIME, address the user's ask in this turn or "
                "explicitly explain in <notes> why a single turn cannot do "
                "both. Do not defer again.\n"
                f"\nRepeated ask(s):\n- "
                + "\n- ".join(fb[:240] for fb in force_honor_via_escalation)
                + "\n=================================================================="
            )
        if self._pending_feedback and not getattr(self, "_use_feedback_directives", True):
            # RAW FEEDBACK MODE — /rawfeedback on. Pass every queued
            # user note through verbatim, with ONLY the basic USER
            # FEEDBACK (HIGHEST PRIORITY) wrapper. Skip strict-scope
            # arbitration, classifier calls, MEDIA-CHANGE / ORIENTATION-
            # CHANGE / SCOPE ARBITRATION directives, asset stem mapping,
            # and scoped-constraint configuration. The model sees what
            # the user typed and decides for itself whether to <patch>
            # or <assets>. Use when the classifier is misrouting your
            # guidance (Doom trace 2026-05-23 was the motivating case).
            feedback_items = list(self._pending_feedback)
            joined_raw = "\n- ".join(feedback_items)
            parts.append(
                "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                "The user just typed this while watching your game. It OVERRIDES\n"
                "any plan or default behavior. Address it explicitly in this turn:\n"
                f"\n[USER NOTE]\n- {joined_raw}\n[/USER NOTE]\n"
                "=================================================================="
            )
            for fb in feedback_items:
                self._trace({"kind": "feedback_injected", "text": fb, "raw_mode": True})
            self._trace({
                "kind": "feedback_directives_suppressed",
                "reason": "raw_feedback_mode",
                "count": len(feedback_items),
            })
            self._last_drained_feedback = list(feedback_items)
            self._recent_feedback_texts.extend(feedback_items)
            self._recent_feedback_texts = self._recent_feedback_texts[-3:]
            self._pending_feedback.clear()
            self._clear_scoped_constraints()
            self._last_turn_contract.update({
                "had_feedback": True,
                "locks_code": False,
                "art_change": False,
                "sound_change": False,
                "behavior_scope": False,
                "behavior_bug": False,
                "orientation_change": False,
                "existing_media_request": False,
                "raw_feedback_mode": True,
            })
        elif self._pending_feedback:
            feedback_items = list(self._pending_feedback)
            strict_idxs = [
                i for (i, fb) in enumerate(feedback_items)
                if _feedback_is_strict_scope(fb)
            ]
            strict_scope_dropped = 0
            if strict_idxs:
                # Latest strict scope wins. Earlier queued asks are
                # intentionally suppressed for this turn so local models
                # don't try to satisfy contradictory objectives.
                keep_i = strict_idxs[-1]
                selected_feedback = [feedback_items[keep_i]]
                strict_scope_dropped = len(feedback_items) - 1
            else:
                selected_feedback = feedback_items
            joined = "\n- ".join(selected_feedback)
            repeated = self._detect_repeated_feedback(joined)
            repeat_prefix = ""
            if repeated:
                repeat_prefix = (
                    "[USER HAS RAISED THIS BEFORE — review screenshots "
                    "and prior fix]\n"
                )
                self._trace({
                    "kind": "repeated_user_request",
                    **repeated,
                    "text_preview": joined[:240],
                })
            post_clean_feedback = self._is_post_clean_instruction(base_message)
            if post_clean_feedback:
                parts.append(
                    "================ POST-CLEAN FEEDBACK CONTRACT ================\n"
                    f"{self._compact_post_clean_context()}\n"
                    "The user is asking for a follow-up on a working game.\n"
                    "Make ONLY the requested change. Do not refactor,\n"
                    "rewrite, rebalance, or address stale coaching unless\n"
                    "the user explicitly asks for that broader work.\n"
                    "=============================================================="
                )
                self._trace({
                    "kind": "post_clean_feedback_contract",
                    "feedback_preview": joined[:240],
                })

            # [USER NOTE] markers added 2026-05-21: explicit labels survive
            # the structured-prune compaction step. Without them, late-
            # session prompts lose user voice — the compactor summarizes
            # the banner away. The marker is also a literal token the
            # model can search-and-attend to.
            parts.append(
                "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                "The user just typed this while watching your game. It OVERRIDES\n"
                "any plan or default behavior. Address it explicitly in this turn:\n"
                f"\n[USER NOTE]\n{repeat_prefix}- {joined}\n[/USER NOTE]\n"
                "=================================================================="
            )
            for fb in selected_feedback:
                self._trace({"kind": "feedback_injected", "text": fb})
            if strict_scope_dropped:
                parts.append(
                    "================ FEEDBACK SCOPE ARBITRATION ================\n"
                    "A strict scope lock was present in the latest feedback.\n"
                    "For THIS turn, treat ONLY that latest scoped request as\n"
                    "in-scope. Ignore earlier queued requests.\n"
                    "============================================================"
                )
                self._trace({
                    "kind": "feedback_scope_arbitration",
                    "kept_latest_scoped": selected_feedback[0][:200],
                    "dropped_count": strict_scope_dropped,
                })

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
            # Drop the bool(asset_names) / bool(sound_names) short-circuits
            # so first-time art/audio requests can route to the MEDIA-CHANGE
            # directive ladder. The classifiers themselves require an art
            # noun + media verb (and audio noun + media verb), so neutral
            # feedback like "fix the bug" still won't fire here. The
            # directive renderer below already branches on whether asset/
            # sound names exist when picking which recipe to emit.
            art_change = _feedback_is_art_change(joined, asset_names)
            sound_change = _feedback_is_sound_change(joined, sound_names)
            existing_media_request = _feedback_requests_existing_media(joined)
            # Phase 0.1 — when the user says BOTH "use the existing X" AND
            # "as a starting point" / "show them walking" / "more frames",
            # they want img2img CHAINING (new frames seeded from existing
            # sprites), not "use the existing PNG verbatim". The old
            # suppression treated all "use existing" mentions as the
            # latter and killed the MEDIA-CHANGE directive entirely; the
            # 2026-05-22 chess trace shows the user asked for img2img
            # animation chains three times and got nothing.
            img2img_chain_request = _feedback_requests_img2img_chain(joined)
            style_rebrand_request = _feedback_requests_style_rebrand(joined)
            if (
                existing_media_request
                and (art_change or sound_change)
                and not img2img_chain_request
                and not style_rebrand_request
            ):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "use_existing_media",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
                art_change = False
                sound_change = False
            elif img2img_chain_request and (art_change or sound_change):
                self._trace({
                    "kind": "img2img_chain_directive_active",
                    "existing_media_request": existing_media_request,
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
            if style_rebrand_request and (art_change or sound_change):
                self._trace({
                    "kind": "style_rebrand_directive_active",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
            # Phase 0.10b — multi-frame intent in MID-SESSION feedback
            # ("make 3 walk frames for each character", "use existing
            # images for X to seed an animation"). When the goal text
            # at session start didn't trigger the cap-raise, a later
            # animation ask would silently hit the default 24 cap. Mirror
            # the session-start path: detect multi-frame intent on the
            # feedback text and raise the session cap if it triggers.
            # General behavior — no genre logic.
            try:
                from prompts_v1 import _detect_multi_frame_intent as _mfi_mid
                mf_mid = _mfi_mid(joined)
            except Exception:
                mf_mid = []
            _prior_mid_cap = getattr(self, "_session_asset_cap", None)
            if mf_mid and (_prior_mid_cap is None or _prior_mid_cap < 72):
                self._session_asset_cap = 72
                self._trace({
                    "kind": "multi_frame_intent_detected",
                    "trigger": "mid_session_feedback",
                    "matched_keywords": mf_mid,
                    "asset_cap_raised_to": 72,
                    "prior_cap": _prior_mid_cap,
                })
            behavior_bug = _feedback_is_behavior_bug(joined)
            behavior_scope = _feedback_mentions_scoped_behavior_change(joined)
            orientation_change = _feedback_is_orientation_change(joined)
            # Record routing flags so `_stream` can emit the
            # `turn_contract` trace event. Mode + tag derivation happens
            # after `_configure_scoped_constraints` runs below.
            self._last_turn_contract.update({
                "had_feedback": True,
                "locks_code": locks_code,
                "art_change": art_change,
                "sound_change": sound_change,
                "behavior_scope": behavior_scope,
                "behavior_bug": behavior_bug,
                "orientation_change": orientation_change,
                "existing_media_request": existing_media_request,
            })
            self._configure_scoped_constraints(
                joined_feedback=joined,
                locks_code=locks_code,
                art_change=art_change,
                sound_change=sound_change,
            )
            self._last_drained_feedback = list(selected_feedback)
            self._recent_feedback_texts.extend(selected_feedback)
            self._recent_feedback_texts = self._recent_feedback_texts[-3:]

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
            if (behavior_bug or (locks_code and behavior_scope)) and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": (
                        "behavior_scope_on_strict_turn"
                        if locks_code and behavior_scope
                        else "behavior_bug"
                    ),
                    "art_change": art_change,
                    "sound_change": sound_change,
                })

            # ORIENTATION-CHANGE suppression — MK trace
            # 20260517_220025: user asked "is there a way to INVERT
            # the asset we use for the player kick?" which mentions an
            # existing asset name, so `art_change` fires; the standard
            # MEDIA-CHANGE directive then tells the model to regen the
            # sprite, which is the wrong fix (a one-line ctx.scale(-1,1)
            # canvas mirror was the right one). When orientation
            # vocabulary is present AND no regen blocker, route to the
            # one-patch canvas mirror path instead.
            if orientation_change and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "orientation_change",
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
            #
            # Two branches:
            #   - has_existing_media: regen using the existing name(s).
            #   - else: emit a fresh <assets>/<sounds> block (NEW ART) +
            #     ONE small <patch> wiring the loader. The 2026-05-21
            #     chess trace is the motivating case — first-time art
            #     request after a working game shipped, no sprites yet,
            #     model defaulted to inline SVG instead of the diffuser.
            has_existing_media = bool(asset_names or sound_names)
            if (
                (art_change or sound_change)
                and not behavior_bug
                and not (locks_code and behavior_scope)
                and not orientation_change
            ):
                if has_existing_media:
                    lines: list[str] = [
                        "================ MEDIA-CHANGE DIRECTIVE ================",
                        "The feedback above is about ART/SOUND, not code. The",
                        "harness can regenerate any sprite or sound in place:",
                        "emit a fresh block with the EXISTING name and a new",
                        "prompt — no JS edit needed. The existing drawSprite()",
                        "/ new Audio() call already in the file automatically",
                        "picks up the regenerated file.",
                    ]
                    # Phase 0.1 — surface the user's fuzzy asset stems
                    # ("pawn" → [white_pawn, black_pawn]) so the model
                    # emits the right `from_image` chains for animation
                    # frames seeded from existing pieces.
                    stem_map = _resolve_fuzzy_asset_stems(joined, asset_names)
                    if asset_names:
                        asset_list = ", ".join(sorted(asset_names))
                        lines.extend([
                            "",
                            "Sprites — use <assets> to re-render:",
                            "  <assets>[{\"name\":\"<existing_name>\","
                            "\"prompt\":\"<new visual prompt>\"}]</assets>",
                            f"  Existing asset names: {asset_list}",
                        ])
                        if stem_map:
                            lines.append("")
                            lines.append("Stems the user referenced map to existing assets:")
                            for stem, names in sorted(stem_map.items()):
                                joined_names = ", ".join(sorted(names))
                                lines.append(f"  '{stem}' → [{joined_names}]")
                            lines.append(
                                "When the user asks for animation FRAMES or"
                                " VARIANTS of these existing sprites (e.g."
                                " walk1/walk2/capture), declare each new"
                                " frame with `from_image: <existing_name>`"
                                " and `strength: 0.35-0.55` so SD-Turbo"
                                " img2img chains it from the parent. Do NOT"
                                " regenerate from scratch via txt2img — the"
                                " new frames will look like different"
                                " characters and break visual continuity."
                            )
                        # Phase 0.11 — STYLE REBRAND branch. User wants
                        # EVERY existing asset re-rendered with a new
                        # visual style (e.g. "all the images should look
                        # like monsters, not regular chess pieces").
                        # Different from img2img chains: this is full
                        # txt2img regeneration with NEW prompts, because
                        # from_image would carry the OLD style forward.
                        if style_rebrand_request:
                            lines.append("")
                            lines.append("STYLE REBRAND DETECTED:")
                            lines.append(
                                "The user wants ALL existing assets re-rendered"
                                " with a new visual style. For EACH existing"
                                " asset name, emit one entry in <assets> with"
                                " the SAME name + a NEW prompt that bakes in"
                                " the requested style. Do NOT use `from_image`"
                                " — chaining would carry the OLD style forward."
                                " The harness regenerates each PNG in place."
                            )
                            lines.append(
                                "Roster-size note: a full rebrand of N existing"
                                " assets = N new <assets> entries. If your"
                                " session has many entries, split the LAST"
                                " batch into a follow-up turn — never silently"
                                " regenerate only a subset."
                            )
                        lines.extend([
                            "",
                            "If the user says an animation/image is MISSING",
                            "and no existing name matches that state, you may",
                            "emit a NEW same-pattern sprite name plus ONE small",
                            "<patch> that only adds the name to the existing",
                            "asset loader/list. Do not rewrite gameplay code.",
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
                else:
                    # NEW MEDIA branch — session has no generated assets/
                    # sounds yet. Tell the model to emit a fresh <assets>
                    # / <sounds> block + ONE small <patch> that loads
                    # them. Without this branch, the directive ladder
                    # was suppressed entirely and the model fell back to
                    # inline SVG / ctx.fillRect / AudioContext beeps.
                    lines = [
                        "================ MEDIA-CHANGE DIRECTIVE (NEW MEDIA) ================",
                        "The feedback above is asking for generated MEDIA, but",
                        "this session has no sprites/sounds yet. The harness",
                        "runs Z-Image-Turbo (sprites) and Stable-Audio-Open",
                        "(sounds) locally — emit the right block this turn",
                        "and the PNG/OGG paths arrive in the next user turn.",
                    ]
                    if art_change:
                        lines.extend([
                            "",
                            "Sprites — emit fresh names:",
                            "  <assets>[",
                            "    {\"name\":\"<short_id>\",",
                            "     \"prompt\":\"<one short visual sentence,"
                            " transparent bg>\"},",
                            "    ...one entry per visual entity the player"
                            " sees...",
                            "  ]</assets>",
                            "",
                            "Then ONE small <patch> that:",
                            "  1. Adds `const ASSETS = { name: new Image(),"
                            " ... }` and `await Promise.all(Object.values("
                            "ASSETS).map(i => i.decode()))` before the loop"
                            " starts.",
                            "  2. Replaces the procedural drawing site with"
                            " `ctx.drawImage(ASSETS.<name>, x, y, w, h)`.",
                            "Do NOT draw the entities with inline SVG, ctx",
                            "primitives, Unicode glyphs, or emoji — the user",
                            "explicitly asked for generated art and that's",
                            "exactly the failure mode this directive prevents.",
                        ])
                    if sound_change:
                        lines.extend([
                            "",
                            "Sounds — emit fresh names:",
                            "  <sounds>[",
                            "    {\"name\":\"<short_id>\",",
                            "     \"prompt\":\"<one short audio sentence>\",",
                            "     \"duration\":<0.3-1.5>},",
                            "    ...one entry per discrete audible event...",
                            "  ]</sounds>",
                            "",
                            "Then ONE small <patch> that does",
                            "`new Audio('./<dir>/<name>.ogg').play()` on the",
                            "matching event. Do NOT synthesize beeps with",
                            "AudioContext oscillators — the user asked for",
                            "real sound.",
                        ])
                    lines.extend([
                        "",
                        "Path conventions (the harness sets these for you):",
                        "  - Sprites land in `./<session>_assets/<name>.png`.",
                        "  - Sounds land in `./<session>_sounds/<name>.ogg`.",
                        "Reference the names you emit; the next user turn",
                        "will surface the exact relative paths.",
                        "===================================================================",
                    ])
                parts.append("\n".join(lines))
                self._trace({
                    "kind": "media_change_directive_injected",
                    "locks_code": locks_code,
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "behavior_scope": behavior_scope,
                    "asset_count": len(asset_names),
                    "sound_count": len(sound_names),
                    # Tag the branch so offline analysis can tell first-
                    # time art requests from regen requests at a glance.
                    "branch": "existing" if has_existing_media else "new",
                })

            # ORIENTATION-CHANGE DIRECTIVE — when the user wants a
            # sprite mirrored on the canvas (not regenerated) emit a
            # short canvas-mirror recipe so the model emits ONE small
            # <patch> wrapping the existing draw call rather than
            # reaching for <assets>. Genre-free, code-only.
            if orientation_change:
                orient_lines = [
                    "================ ORIENTATION-CHANGE DIRECTIVE ================",
                    "The user wants a SPRITE MIRRORED/FLIPPED on the canvas,",
                    "not regenerated. Emit ONE small <patch> that wraps the",
                    "existing drawImage() call for the named sprite with a",
                    "ctx.save() / ctx.scale(-1, 1) / ctx.restore() block.",
                    "Do NOT emit <assets> — the existing PNG is fine and",
                    "another generation will not change its facing direction",
                    "in a reliable way.",
                    "",
                    "Canonical recipe (adapt names to this file's helper):",
                    "  if (currentSpriteName === '<name>') {",
                    "    ctx.save();",
                    "    ctx.translate(x + w, 0);",
                    "    ctx.scale(-1, 1);",
                    "    ctx.drawImage(sprite, 0, y, w, h);",
                    "    ctx.restore();",
                    "  } else {",
                    "    ctx.drawImage(sprite, x, y, w, h);",
                    "  }",
                    "",
                    "If a facing-aware branch already exists in the helper,",
                    "KEEP it for all sprites and ADD a separate sprite-name",
                    "guard with an EXTRA scale(-1,1). Never replace the",
                    "existing facing branch with an `else if`.",
                    "===============================================================",
                ]
                parts.append("\n".join(orient_lines))
                self._trace({
                    "kind": "orientation_change_directive_injected",
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "locks_code": locks_code,
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
                if self._scoped_constraints is not None:
                    scoped_mode = self._scoped_constraints.get("mode")
                    if scoped_mode == "media_only":
                        scoped_lines.insert(
                            3,
                            "Turn mode: MEDIA-ONLY (existing names only).",
                        )
                    else:
                        scoped_lines.insert(
                            3,
                            "Turn mode: ONE-SMALL-PATCH (no rewrite, max one patch block).",
                        )
                parts.append("\n".join(scoped_lines))
                self._trace({
                    "kind": "scoped_change_directive_injected",
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "mode": (
                        (self._scoped_constraints or {}).get("mode")
                        or "single_patch"
                    ),
                })
                # Flag consumed by the fix-mode prompt builder to drop
                # the "fix these failing probes" framing for this turn.
                # Cleared after one fix-mode build cycle (caller resets).
                self._scoped_change_active = True

            # BOUNDED-ASSET-ONLY TURN — when scoped_mode is media_only,
            # append a hard, narrow output spec at the end of the user
            # message. MK trace 20260517_220025: the model streamed an
            # asset prompt fragment in a loop for 30 minutes because
            # the user message had multiple competing instructions
            # (USER FEEDBACK + MEDIA-CHANGE + SCOPED-CHANGE) and no
            # explicit "stop after one block" constraint. Adding the
            # cap LAST so it's the most recent instruction the model
            # sees before generating.
            scoped_mode_now = (self._scoped_constraints or {}).get("mode")
            if scoped_mode_now == "media_only":
                allowed_assets = sorted(asset_names)
                bounded_lines = [
                    "================ BOUNDED OUTPUT — MEDIA ONLY ================",
                    "Emit exactly ONE <assets> JSON array. No other tags.",
                    "  - Use ONLY these existing names (regen replaces the",
                    "    file on disk; the existing drawImage call picks",
                    "    up the new pixels automatically):",
                ]
                if allowed_assets:
                    bounded_lines.append(
                        "    " + ", ".join(allowed_assets)
                    )
                else:
                    bounded_lines.append(
                        "    (no existing asset names — abort and ask "
                        "the user instead)"
                    )
                bounded_lines.extend([
                    "  - Each prompt: ONE short sentence, ≤ 200 chars.",
                    "  - Close the block with </assets> and STOP. Do NOT",
                    "    repeat or restate the JSON, do NOT add prose,",
                    "    do NOT emit <patch>, <html_file>, <plan>,",
                    "    <criteria>, <probes>, or <notes>.",
                    "=============================================================",
                ])
                parts.append("\n".join(bounded_lines))
                self._trace({
                    "kind": "bounded_asset_only_injected",
                    "asset_count": len(allowed_assets),
                })

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

        # Probe quarantine notices are not normal "coaching"; they explain
        # that the agent dropped a broken model-authored probe after repeated
        # eval-time errors. Always surface once so the next model turn knows
        # it may re-emit <probes> if the dropped assertion mattered.
        if self._pending_probe_quarantine_notices:
            joined = "\n- ".join(self._pending_probe_quarantine_notices)
            parts.append(
                "================ PROBE QUARANTINE NOTICE ================\n"
                f"- {joined}\n"
                "========================================================="
            )
            for c in self._pending_probe_quarantine_notices:
                self._trace({"kind": "probe_quarantine_notice_injected", "text": c})
            self._pending_probe_quarantine_notices.clear()

        # A2/A5: agent-queued coaching lines (deliberation guard recovery,
        # repeat-error nudges). Rendered as a single high-priority block
        # so the model sees them before the base instruction.
        if self._pending_coaching:
            scoped_lock_active = bool(self._scoped_constraints is not None)
            if scoped_lock_active:
                self._trace({
                    "kind": "coaching_suppressed_scoped_lock",
                    "count": len(self._pending_coaching),
                })
                self._pending_coaching.clear()
            else:
                prev = self._previous_report or {}
                probes = prev.get("probes") or []
                full_probe_pass = bool(probes) and all(bool(p.get("ok")) for p in probes)
                clean_report = (
                    self._previous_report_ok is True
                    and full_probe_pass
                    and not (prev.get("errors") or [])
                    and not (prev.get("soft_warnings") or [])
                    and not (prev.get("page_errors") or [])
                    and not (prev.get("console_errors") or [])
                )
                coaching_to_inject = self._pending_coaching
                if clean_report:
                    # Visual/static/player-motion warnings are generated
                    # specifically because the normal report looked clean.
                    # Preserve those so "passes but not playable" issues
                    # still reach the next coder turn.
                    must_keep_keywords = (
                        "VISUAL CRITIC",
                        "VISUAL JUDGE",
                        "REGRESSION SUSPECTED",
                        "STATIC SCREEN",
                        "STATE LOCOMOTION",
                        "Asset references",
                        "Sound references",
                    )
                    coaching_to_inject = [
                        c for c in self._pending_coaching
                        if any(kw in c for kw in must_keep_keywords)
                    ]
                    suppressed = [
                        c for c in self._pending_coaching
                        if c not in coaching_to_inject
                    ]
                    self._trace({
                        "kind": "coaching_suppressed_clean_pass",
                        "count": len(suppressed),
                        "preserved_count": len(coaching_to_inject),
                    })
                if coaching_to_inject:
                    joined = "\n- ".join(coaching_to_inject)
                    # [CRITIC] marker (added 2026-05-21): mirrors the
                    # [USER NOTE] pattern so critic findings get the same
                    # compaction-survival treatment as user feedback.
                    # Distinct label so the model can tell which voice is
                    # speaking (user override > critic suggestion).
                    has_critic = any(
                        "VISUAL CRITIC" in c or "VISUAL JUDGE" in c
                        for c in coaching_to_inject
                    )
                    label_open = "[CRITIC]\n" if has_critic else ""
                    label_close = "\n[/CRITIC]" if has_critic else ""
                    parts.append(
                        "================ AGENT COACHING ================\n"
                        f"{label_open}- {joined}{label_close}\n"
                        "================================================"
                    )
                    for c in coaching_to_inject:
                        self._trace({"kind": "coaching_injected", "text": c})
                self._pending_coaching.clear()
        post_clean_with_feedback = bool(
            base_message
            and self._last_turn_contract
            and self._last_turn_contract.get("had_feedback")
            and self._is_post_clean_instruction(base_message)
        )
        if base_message:
            if post_clean_with_feedback:
                # Replace the large clean report with one compact line; the
                # POST-CLEAN FEEDBACK CONTRACT above carries the important
                # baseline signal without drowning the user's fresh request.
                parts.append(self._compact_post_clean_context())
                self._trace({"kind": "post_clean_report_compacted"})
            else:
                parts.append(base_message)

        # Inline CURRENT FILE ON DISK on post-clean feedback turns and
        # whenever an answer is being injected. Without this, the model
        # is asked to patch concrete code ("remove the circles", "shift
        # the muzzle flash up") with no file in context — after compaction
        # the original <html_file> is gone from history, and post-clean
        # / answer turns don't otherwise carry it (unlike fix_instruction).
        # The DOOM trace 20260523_152317 has the model literally saying
        # "I genuinely do not have the file contents in my context this
        # turn" and giving up. Mirrors the file block from
        # continuation_instruction.
        if (post_clean_with_feedback or answer_was_consumed) and self._current_file:
            parts.append(
                "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — patch "
                "against THIS exact text, character-for-character; earlier "
                "turns' code may be stale or absent from this prompt):\n"
                "```html\n"
                f"{self._current_file}\n"
                "```"
            )
            self._trace({
                "kind": "current_file_inlined",
                "reason": (
                    "post_clean_with_feedback"
                    if post_clean_with_feedback
                    else "answer_consumed"
                ),
                "bytes": len(self._current_file),
            })

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

        # Phase 0.1 — restore code-touching feedback that was partitioned
        # out for parallel media processing. These items remain pending
        # behind the code blocker; they re-evaluate next turn.
        if code_to_defer:
            for fb in code_to_defer:
                if fb not in self._pending_feedback:
                    self._pending_feedback.append(fb)

        return "\n\n".join(parts)

    # -- streaming ----------------------------------------------------------

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
            # Auto-staff (added 2026-05-21): if a local VLM is on ANY
            # slot (coder/critic/architect) and vlm-critique is still off,
            # enable it. Single-model users with a VLM coder were getting
            # nothing from the proven-useful visual critic loop because
            # /vlm-critique stayed off by default. User can still override
            # with /vlm-critique off, and the explicit toggle path remains
            # the source of truth.
            if not self._use_vlm_critique:
                self._use_vlm_critique = True
                self._vlm_critique_auto = True
                self._trace({
                    "kind": "auto_enabled",
                    "feature": "vlm_critique",
                    "reason": f"local VLM detected on role={role}",
                    "model": backend.info.model,
                })
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
            r"(?:\d+\s*[.)]\s*)?"
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
            f"[VISUAL PLAYTEST — {recipe.id}] "
            f"{len(failures)} of {n_q} check(s) failed"
            + (f" + {len(unclears)} unclear" if unclears else "")
            + ":"
        )
        body = "\n".join(failures + unclears)
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
                    return head + "\n" + body + "\n\nMinimal fix shape:\n" + "\n".join(
                        f"  - (Q{idx}) {h}" for idx, h in zip(failed_idxs, hints)
                    )
            fix_hint = (recipe.recipe.get("fix_hint") or "").strip()
            if fix_hint:
                return head + "\n" + body + "\n\nMinimal fix shape: " + fix_hint
        return head + "\n" + body

    async def _audit_sprite_orientation(
        self,
        produced: dict[str, Any],
        per_asset: list[dict],
    ) -> list[str]:
        """Fix round 5b — post-generation orientation audit. For directional
        pose sprites whose prompt was orientation-pinned (kick/punch/strike/
        throw…), ask the local critic VLM which way the action points and
        mirror-flip any LEFT answers in place. NEVER regenerates a frame
        (lossless transpose preserves the character). Advisory only: any
        failure (no VLM, bad parse) skips silently and returns []."""
        targets = select_orientation_audit_targets(per_asset, produced)
        if not targets:
            return []
        backend = self.get_backend("critic")
        if backend is None:
            return []
        try:
            if not await backend.is_vlm():
                return []
        except Exception:
            return []
        images: list[bytes] = []
        names: list[str] = []
        paths: list[Any] = []
        for name, path in targets:
            try:
                images.append(Path(path).read_bytes())
            except OSError:
                continue
            names.append(name)
            paths.append(path)
        if not images:
            return []
        listing = "\n".join(
            f"{i + 1}. {n}" for i, n in enumerate(names)
        )
        prompt = (
            "You are auditing game sprites. Each numbered image below shows "
            "a character performing an action (the order matches this "
            "list):\n"
            f"{listing}\n\n"
            "For EACH image, answer which horizontal direction the main "
            "action (extended leg/arm, strike, throw) points. Reply with "
            "ONE line per image, format strictly:\n"
            "<number>: LEFT or <number>: RIGHT\n"
            "No other text."
        )
        messages = [{"role": "user", "content": prompt, "images": images}]
        try:
            result = await backend.stream_chat(
                messages,
                options={"temperature": 0.0, "num_ctx": 4096},
                keep_alive=self._keep_alive_for_backend(backend),
                stall_seconds=120.0,
                overall_seconds=300.0,
                max_retries=1,
            )
            reply = (result.text or "").strip()
        except Exception as e:
            self._trace_exception("orientation_audit_vlm_error", e)
            return []
        verdicts = parse_orientation_verdicts(reply, len(names))
        flipped: list[str] = []
        for idx, direction in verdicts.items():
            if direction != "left":
                continue
            name = names[idx - 1]
            if flip_sprite_horizontal(paths[idx - 1]):
                flipped.append(name)
                for stat in per_asset:
                    if isinstance(stat, dict) and stat.get("name") == name:
                        stat["orientation_flipped"] = True
        self._trace({
            "kind": "orientation_flip_audit",
            "targets": names,
            "verdicts": {names[i - 1]: d for i, d in verdicts.items()},
            "flipped": flipped,
            "reply_preview": reply[:200],
        })
        return flipped

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
                    last_verified_at=datetime.utcnow().isoformat() + "Z",
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

    def _maybe_prewarm_diffusers_during_phase_a(self) -> None:
        """Background-load Z-Image-Turbo + Stable Audio Open during the
        architect's streaming window — but ONLY when the diffuser GPU
        is independent of any LLM slot.

        On the 4-GPU workstation the diffusers run on GPU 0 and the
        LLM on GPUs 1-3, so the warmup is pure speedup (~30-60 s
        hidden). On a single-GPU box the diffuser and the architect
        share VRAM, so pre-warming would steal VRAM from the in-flight
        plan stream — skip in that case.

        Fires-and-forgets: each pipeline's `ensure_loaded()` runs in
        `asyncio.to_thread`; if loading raises, we trace + move on.
        The user-facing wins are captured by `prefill_warm` trace
        events tagged with `target: "z_image" | "stable_audio"`.
        """
        try:
            import gpu_status as _gs
            snap = _gs.snapshot_gpus()
            if not _gs.diffuser_has_dedicated_gpu(snap):
                self._trace({
                    "kind": "diffuser_prewarm_skipped",
                    "reason": "no_dedicated_gpu",
                    "n_gpus": len(snap.gpus) if snap and snap.gpus else 0,
                })
                return
        except Exception as e:
            self._trace({
                "kind": "diffuser_prewarm_skipped",
                "reason": "gpu_probe_error",
                "err": str(e)[:200],
            })
            return

        async def _prewarm_image() -> None:
            import time as _t
            t0 = _t.monotonic()
            try:
                import assets as _assets
                gen = self._asset_generator or _assets.ZImageTurboGenerator()
                self._asset_generator = gen
                ok = await asyncio.to_thread(gen._lazy_init)
                self._trace({
                    "kind": "prefill_warm",
                    "target": "z_image",
                    "elapsed_s": round(_t.monotonic() - t0, 2),
                    "ok": bool(ok),
                    "hidden_under_phase_a": True,
                })
            except Exception as e:
                self._trace({
                    "kind": "prefill_warm",
                    "target": "z_image",
                    "elapsed_s": round(_t.monotonic() - t0, 2),
                    "ok": False,
                    "err": str(e)[:200],
                })

        async def _prewarm_audio() -> None:
            import time as _t
            t0 = _t.monotonic()
            try:
                import sounds as _sounds
                gen = self._sound_generator or _sounds.StableAudioGenerator()
                self._sound_generator = gen
                ok = await asyncio.to_thread(gen._lazy_init)
                self._trace({
                    "kind": "prefill_warm",
                    "target": "stable_audio",
                    "elapsed_s": round(_t.monotonic() - t0, 2),
                    "ok": bool(ok),
                    "hidden_under_phase_a": True,
                })
            except Exception as e:
                self._trace({
                    "kind": "prefill_warm",
                    "target": "stable_audio",
                    "elapsed_s": round(_t.monotonic() - t0, 2),
                    "ok": False,
                    "err": str(e)[:200],
                })

        # Fire-and-forget. We don't await — these run concurrently with
        # the architect's streaming. The tasks complete on their own
        # schedule; the next `<assets>` / `<sounds>` block uses the
        # already-loaded pipeline.
        try:
            asyncio.create_task(_prewarm_image())
            asyncio.create_task(_prewarm_audio())
        except Exception:
            pass

    # ---- Phase 1A — non-blocking visual critic --------------------------
    #
    # When the critic role is bound to a SEPARATE Ollama slot from the
    # coder (the normal multi-GPU configuration), running the critic
    # serially after each iter wastes 20-300 s of wall-clock — the
    # critic compute could overlap with the next iter's coder stream.
    # Spawning it as `asyncio.Task` lets that overlap happen and lets
    # the coaching land in `_pending_coaching` whenever the task
    # finishes (one-turn lag at worst). When the critic backend is the
    # SAME as the coder backend (single-slot / single-GPU fallback),
    # concurrent runs just queue at the daemon — no benefit, so we
    # fall back to the existing blocking await.

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
    # the `/feedback off` kill-switch are both checked first; this method
    # is a no-op when either gate is closed.

    _AUTONOMOUS_MAX_CYCLES = 3
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
          - /feedback is on (default)
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
        yield self._record(AgentEvent(
            "info",
            f"[magenta]autonomous feedback queued[/magenta] "
            f"({len(findings)} finding(s) from cycle "
            f"{self._autonomous_playtest_cycle})",
        ))

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
        # Determine the target backend and its VLM capability dynamically
        active_backend = self.get_backend(role)
        if active_backend is None:
            raise RuntimeError(f"no backend configured for role={role!r}")
        is_vlm_active = await self._detect_vlm(role)

        self._last_stream_role = role
        self._set_role_activity(role, f"streaming {role}…")

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
                "prompt_sections": self._estimate_prompt_section_chars(),
                "temperature": temp,
            })
            allowed, forbidden = self._derive_allowed_forbidden_tags()
            contract["allowed_tags"] = allowed
            contract["forbidden_tags"] = forbidden
            contract["keep_alive"] = self._keep_alive_for_backend(active_backend)
            self._trace(contract)
        except Exception:
            pass
        self._trace({
            "kind": "stream_start",
            "temperature": temp,
            "fix_mode": self._fix_mode,
            "keep_alive": self._keep_alive_for_backend(active_backend),
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
        self._last_stream_silent = bool(getattr(result, "silent", False))
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
        async def scorer(text: str) -> tuple[float, dict]:
            extra: dict = {"kind": "candidate", "text_len": len(text)}
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
            # paths like ./<session>_assets/x.png resolve. Testing from
            # snapshots_dir made every PNG 404 → sprite fallbacks drew blocks
            # and scoring was biased against asset-using candidates.
            tmp_path = self.out_path.parent / f".cand_{self._snapshot_n+1:02d}_{abs(hash(text))%10000:04d}.html"
            try:
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(html, encoding="utf-8")
                report = await self.browser.load_and_test(
                    tmp_path, screenshot_path=None,
                    probes=self._probes or None,
                    opening_book_recipes=getattr(self, "_active_opening_book_recipes", []),
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
            finally:
                # Candidate files are scoring scratch — never leave them
                # beside the real game in games/.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

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
        """
        if not report:
            return format_report_for_model(report)
        compacted = self._compact_warnings_for_prompt(
            report.get("warnings") or []
        )
        rfp = dict(report)
        rfp["warnings"] = compacted
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
        # for `_VISION_NOTE_ACTIONABLE_TOKENS`.
        low_words = {w.lower() for w in words}
        if not (cls._VISION_NOTE_ACTIONABLE_TOKENS & low_words):
            return ""
        return text

    @staticmethod
    def _mark_unused_media_as_stale_for_continuation(mp: dict) -> int:
        """Rewrite unused-media warnings for full continuation rewrites.

        When a full rewrite changes the runtime shape, generated media
        from the previous build may simply be stale. In that context,
        "wire it in" is the wrong instruction; the model should use or
        request media that fits the current request.

        Returns the number of old unused-media warnings suppressed.
        """
        if not (mp.get("stats") or {}).get("unused_assets"):
            return 0
        old_warnings = list(mp.get("warnings") or [])
        kept = [
            w for w in old_warnings
            if "NEVER referenced in the HTML" not in str(w)
        ]
        suppressed = len(old_warnings) - len(kept)
        if suppressed:
            kept.append(
                "Generated media from the previous build appears stale "
                "after this full rewrite. Ignore old asset/sound names "
                "unless they still fit the current request; request or "
                "wire media for the current game shape instead."
            )
            mp["warnings"] = kept
        return suppressed

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

    # MK trace 20260517_220025 fix. Iter 2 probe:
    #   (()=>{try{const s=window.state; if(!s)return false;
    #          return s.cpuPunchFlipped===true;}catch(e){return false;}})()
    # was bait — the SAME iter's patch added `cpuPunchFlipped: true`
    # to state.reset(), so the probe trivially passed without
    # verifying the actual draw transform. Detect this shape so the
    # next user-turn fix prompt surfaces the bait. The probe-side
    # pattern matches any `<ident>.<prop> === true|false` shape (the
    # probe in the MK trace used `s.X` where `s` was a local alias
    # for `window.state`). False-positive risk is bounded by the
    # second-pass check that `<prop>` was also assigned to a literal
    # boolean by the same iter's patch.
    _PROBE_BAIT_PROP_RE = re.compile(
        r"\b\w+\.(\w+)\s*===\s*(?:true|false)\b",
        re.IGNORECASE,
    )
    # Inline constant assignment matchers — both object-literal
    # (`cpuPunchFlipped: true`) and statement (`state.X = true`,
    # `window.X = true`) forms. Conservative on purpose: we only
    # match LITERAL true/false to flag the exact bait shape.
    _PROBE_BAIT_OBJLIT_RE = re.compile(
        r"\b(\w+)\s*:\s*(?:true|false)\b",
        re.IGNORECASE,
    )
    _PROBE_BAIT_ASSIGN_RE = re.compile(
        r"(?:state|window|self|globalThis)\.(\w+)\s*=\s*(?:true|false)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _probes_baited_by_patches(
        probes: list[dict], applied_replaces: list[str],
    ) -> list[dict]:
        """Detect probes whose only assertion is `state.X === true|false`
        when the SAME iter's applied patch also writes that exact flag
        as a literal constant. Returns the same `{name, kind, message}`
        shape as `_lint_probes`. MK trace 20260517_220025 iter 2 is
        the case study: a probe checked `state.cpuPunchFlipped===true`
        while the patch added `cpuPunchFlipped: true` to reset() — the
        probe passed without actually testing the rendered flip.

        Conservative: requires LITERAL true/false on both sides, and
        only flags when the same property name appears in BOTH the
        probe expr and a REPLACE constant assignment. False positives
        would noisily lint legitimate flag checks; false negatives
        keep the existing behavior.
        """
        if not probes or not applied_replaces:
            return []
        # Build the set of property names assigned to a literal
        # boolean by the applied patches.
        assigned_to_bool: set[str] = set()
        for replace_text in applied_replaces:
            if not isinstance(replace_text, str) or not replace_text:
                continue
            for m in GameAgent._PROBE_BAIT_OBJLIT_RE.finditer(replace_text):
                assigned_to_bool.add(m.group(1))
            for m in GameAgent._PROBE_BAIT_ASSIGN_RE.finditer(replace_text):
                assigned_to_bool.add(m.group(1))
        if not assigned_to_bool:
            return []
        findings: list[dict] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            for m in GameAgent._PROBE_BAIT_PROP_RE.finditer(expr):
                prop = m.group(1)
                if prop in assigned_to_bool:
                    findings.append({
                        "name": name,
                        "kind": "probe_bait_flag",
                        "message": (
                            f"probe `{name}` checks "
                            f"`{prop} === true|false` but the same "
                            f"iter's patch sets `{prop}` to a literal "
                            f"boolean — the probe passes without "
                            f"verifying the actual behavior. Replace "
                            f"the flag check with a draw-path / DOM / "
                            f"canvas-state assertion that fails when "
                            f"the visual change is missing."
                        ),
                    })
                    break
        return findings

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
                        f"return) regardless of game behavior. Prefer "
                        f"fixing the probe to read real runtime state; "
                        f"do not add a constant pass-flag assignment "
                        f"unless it is genuinely updated by gameplay."
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

    @staticmethod
    def _filter_media_specs_to_allowed(
        specs: list[dict[str, Any]],
        allowed_names: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not specs:
            return specs, []
        if not allowed_names:
            dropped = [
                str(spec.get("name") or "").strip()
                for spec in specs
                if str(spec.get("name") or "").strip()
            ]
            return [], dropped
        kept: list[dict[str, Any]] = []
        dropped: list[str] = []
        for spec in specs:
            name = str(spec.get("name") or "").strip()
            if name in allowed_names:
                kept.append(spec)
            elif name:
                dropped.append(name)
        return kept, dropped

    async def _maybe_generate_assets_and_sounds(
        self, reply: str, *, trigger: str,
    ) -> AsyncIterator[AgentEvent]:
        """Parse <assets>/<sounds>/<videos> in a model reply and run the
        diffuser / audio / video pipeline for each block present.

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
        # Phase 0.10 — per-session asset cap. Defaults to module-level
        # _MAX_ASSETS_PER_TURN; raised at session start when the goal
        # text explicitly asks for multi-frame rosters
        # (`prompts_v1._detect_multi_frame_intent`). The raise lets a
        # user-requested N entities × M frames roster land in one turn
        # instead of getting silently truncated.
        session_cap = getattr(self, "_session_asset_cap", None)
        asset_specs, dropped_asset_names = parse_assets_block_with_meta(
            reply, max_assets=session_cap,
        )
        sound_specs = parse_sounds_block(reply)
        video_specs = parse_videos_block(reply)
        # The model emitted an <assets> block → any outstanding "generate new
        # art" request is now honored; stop re-asserting it (see the
        # ASSET GENERATION REQUIRED directive in _flush_user_injections).
        if asset_specs and self._unhonored_asset_request is not None:
            self._unhonored_asset_request = None
            self._asset_reprompt_count = 0

        # P1 (MK trace 20260528): on a seed restart with on-disk media,
        # SKIP phase_a generation entirely. The model can still emit
        # <assets> in its plan, but we won't burn 90+ seconds re-
        # rendering sprites the user already has. Mid-session triggers
        # are unaffected — explicit user requests like "add a new boss
        # sprite" still flow through the generator below.
        if (
            trigger == "phase_a"
            and self.seed_file is not None
            and (self._session_assets or self._session_sounds)
            and (asset_specs or sound_specs)
        ):
            self._trace({
                "kind": "seed_phase_a_media_skipped",
                "have_assets": len(self._session_assets),
                "have_sounds": len(self._session_sounds),
                "requested_assets": [
                    str(s.get("name") or "") for s in asset_specs
                ],
                "requested_sounds": [
                    str(s.get("name") or "") for s in sound_specs
                ],
            })
            yield self._record(AgentEvent(
                "info",
                f"[dim]phase_a asset/sound generation skipped — seed has "
                f"{len(self._session_assets)} asset(s) and "
                f"{len(self._session_sounds)} sound(s) on disk; "
                f"reusing existing media.[/dim]",
            ))
            return

        # Phase 0.13 — when the model emits a mid-session <assets> block
        # BUT the user has queued feedback asking for a style rebrand,
        # the model is about to generate sprites guaranteed to be
        # discarded next turn. Defer the asset gen and queue a coaching
        # note so the next user-turn re-emits the same <assets> with the
        # user's style baked in. Prevents the "model emitted 14 wrong-
        # style sprites while user feedback waited 25 minutes" failure
        # mode from the 2026-05-22 trace. General behavior — fires only
        # on the conjunction of mid-session model-driven asset gen AND
        # style-rebrand intent in pending feedback.
        if (
            trigger == "mid_session"
            and asset_specs
            and self._pending_feedback
        ):
            joined_pending = "\n".join(self._pending_feedback)
            try:
                wants_rebrand = _feedback_requests_style_rebrand(joined_pending)
            except Exception:
                wants_rebrand = False
            if wants_rebrand:
                # Echo the asset names back so the model knows what to
                # re-emit; include the user feedback preview so the
                # coaching is grounded in their own words.
                deferred_names = [str(s.get("name", "")) for s in asset_specs]
                fb_preview = joined_pending[:240].replace("\n", " | ")
                self._trace({
                    "kind": "mid_session_assets_deferred_for_user_style",
                    "deferred_names": deferred_names,
                    "feedback_preview": fb_preview,
                })
                yield self._record(AgentEvent(
                    "info",
                    f"[yellow]mid-session <assets> deferred[/yellow] — "
                    f"user has queued style-rebrand feedback. "
                    f"{len(deferred_names)} asset spec(s) will be re-emitted "
                    "next turn with the user's style applied.",
                ))
                # Coaching note carried into the next user-turn so the
                # model knows the prior <assets> got dropped and must be
                # re-emitted with the new style.
                self._pending_coaching.append(
                    "MID-SESSION ASSET DEFERRAL — your last reply emitted "
                    f"<assets> for {len(deferred_names)} entries "
                    f"({', '.join(deferred_names[:6])}"
                    f"{'…' if len(deferred_names) > 6 else ''}) but the user "
                    "had QUEUED feedback asking for a style rebrand. The "
                    "asset gen was SKIPPED so you don't burn GPU on sprites "
                    "guaranteed to be replaced. Re-emit the SAME asset "
                    "names in THIS turn's <assets> block but with NEW "
                    "prompts that bake in the style the user described. "
                    "Do NOT use `from_image` for the rebrand (it would "
                    "carry the old style forward)."
                )
                # Bail out — no gen, no path injection, no session-asset
                # bookkeeping. The next iter's flush will inject user
                # feedback + this coaching note + the regular MEDIA-CHANGE
                # / STYLE-REBRAND directive.
                return
        # Strict scoped-media lock: when the user said "regenerate only the
        # existing media", reject additive names early (before generation).
        scoped = self._scoped_constraints or {}
        if (
            trigger == "mid_session"
            and scoped.get("mode") == "media_only"
            and scoped.get("media_name_lock")
        ):
            allowed_assets = set(scoped.get("allowed_asset_names") or [])
            allowed_sounds = set(scoped.get("allowed_sound_names") or [])
            asset_specs, dropped_new_assets = self._filter_media_specs_to_allowed(
                asset_specs, allowed_assets,
            )
            sound_specs, dropped_new_sounds = self._filter_media_specs_to_allowed(
                sound_specs, allowed_sounds,
            )
            if dropped_new_assets or dropped_new_sounds:
                allowed_asset_line = ", ".join(sorted(allowed_assets)) or "(none)"
                allowed_sound_line = ", ".join(sorted(allowed_sounds)) or "(none)"
                dropped_line = ", ".join(dropped_new_assets + dropped_new_sounds)
                self._trace({
                    "kind": "scoped_media_new_names_rejected",
                    "dropped": sorted(dropped_new_assets + dropped_new_sounds),
                    "allowed_assets": sorted(allowed_assets),
                    "allowed_sounds": sorted(allowed_sounds),
                })
                self._queue_internal_feedback(
                    "SCOPED MEDIA NAME LOCK: new names were ignored "
                    f"[{dropped_line}]. Use existing names only. Assets: "
                    f"{allowed_asset_line}. Sounds: {allowed_sound_line}."
                )
                yield self._record(AgentEvent(
                    "info",
                    "scoped media lock: dropped new names; only existing keys are allowed this turn.",
                ))
        if not asset_specs and not sound_specs and not video_specs:
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
        new_video_paths: dict[str, Path] = {}
        new_looping: set[str] = set()
        pre_asset_hashes: dict[str, str | None] = {}
        pre_sound_hashes: dict[str, str | None] = {}
        if trigger == "mid_session":
            for spec in asset_specs:
                nm = str(spec.get("name", "")).strip()
                if nm and nm in self._session_assets:
                    pre_asset_hashes[nm] = self._file_hash16(self._session_assets.get(nm))
            for spec in sound_specs:
                nm = str(spec.get("name", "")).strip()
                if nm and nm in self._session_sounds:
                    pre_sound_hashes[nm] = self._file_hash16(self._session_sounds.get(nm))

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
                # At-a-glance "did each pose frame actually move off idle?"
                # signal (2026-05-31): {name: delta-vs-parent} for from_image
                # frames. A reviewer can spot a cloned pose (delta < ~0.04)
                # without digging into per_asset or regenerating. Genre-free.
                pose_deltas = {
                    s.get("name"): s.get("parent_delta")
                    for s in per_asset
                    if isinstance(s, dict) and s.get("from_image")
                    and isinstance(s.get("parent_delta"), (int, float))
                }
                # Fix round 5b: VLM orientation audit + mirror flip for
                # directional pose sprites the diffuser drew facing left
                # (prompt pins are advisory — the kick frame in trace
                # 20260611_145321 disobeyed). Runs BEFORE the trace so
                # per_asset carries orientation_flipped flags.
                if produced:
                    try:
                        _flipped = await self._audit_sprite_orientation(
                            produced, per_asset,
                        )
                        if _flipped:
                            yield self._record(AgentEvent(
                                "info",
                                f"orientation audit: mirrored "
                                f"{len(_flipped)} sprite(s) to face right: "
                                f"{', '.join(_flipped)}",
                            ))
                    except Exception as _oa_e:
                        self._trace_exception("orientation_audit_error", _oa_e)
                self._trace({
                    "kind": "assets_generated",
                    "trigger": trigger,
                    "requested": len(asset_specs),
                    "produced": len(produced),
                    "names": list(produced.keys()),
                    "session_dir": str(session_assets_dir),
                    "pose_deltas": pose_deltas,
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

                # Derived-frame sanity: any sprite chained from a parent
                # (from_image) that came out near-identical to that parent
                # probably did NOT render the requested pose change — the
                # classic "punch sprite looks like idle" silent failure. The
                # model can't see its own art, so log it AND queue it as
                # actionable feedback for the next turn. Genre-free.
                derived = [
                    s for s in per_asset
                    if isinstance(s, dict)
                    and s.get("name") in produced
                    and s.get("from_image")
                    and isinstance(s.get("parent_delta"), (int, float))
                ]
                if derived:
                    # Signal-driven: the model declared animation frames, so
                    # the visual critic should verify the motion reads.
                    self._declared_anim_frames = True
                near_identical = [
                    s for s in derived
                    if s["parent_delta"] < _DERIVED_FRAME_MIN_DELTA
                ]
                # Update the done-gate set: a frame that just regenerated
                # with a real delta clears; a dead one (re)arms the block.
                for s in derived:
                    if s["parent_delta"] >= _DERIVED_FRAME_MIN_DELTA:
                        self._dead_anim_frames.pop(s["name"], None)
                for s in near_identical:
                    self._dead_anim_frames[s["name"]] = float(s["parent_delta"])
                if near_identical:
                    self._trace({
                        "kind": "derived_frame_near_identical",
                        "trigger": trigger,
                        "assets": [
                            {"name": s["name"], "from_image": s["from_image"],
                             "parent_delta": s["parent_delta"]}
                            for s in near_identical
                        ],
                    })
                    warn_lines = [
                        # Sentinel keeps this advisory OUT of the user-art-
                        # request classifier (it is not a request to make art;
                        # it explicitly says regen is a dead end).
                        _HARNESS_ADVISORY_SENTINEL,
                        "ASSET SANITY WARNING — these generated sprites came "
                        "out nearly identical to the `from_image` parent they "
                        "were derived from, so the requested pose change "
                        "likely did NOT render (e.g. a 'punch' frame that "
                        "looks just like idle). The pixels barely differ:",
                    ]
                    for s in near_identical:
                        pct = round((1.0 - float(s["parent_delta"])) * 100)
                        line = (
                            f"  - `{s['name']}` ≈{pct}% identical to parent "
                            f"`{s['from_image']}` (delta "
                            f"{s['parent_delta']:.3f})"
                        )
                        warn_lines.append(line)
                        yield self._record(AgentEvent("info", line.strip()))
                    warn_lines.append(
                        "This is COSMETIC and does not block shipping. Do NOT "
                        "try to regenerate these frames to fix it: img2img "
                        "cannot change a pose (it stays locked to idle), and a "
                        "fresh txt2img frame will not stay consistent with the "
                        "character already in the game — consistency is the "
                        "hard constraint. Keep using the sprites you have, "
                        "never draw the limb in code, and move on to the "
                        "behavior the user actually asked for."
                    )
                    self._queue_internal_feedback("\n".join(warn_lines))

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

        # Video cutscenes — generated AFTER assets so an `image` field can
        # seed image-to-video from a key-art sprite produced THIS turn.
        # Each clip costs minutes of GPU; surface per-clip progress.
        if video_specs:
            yield self._record(AgentEvent(
                "info",
                f"{trigger}: requested {len(video_specs)} cutscene "
                "video(s); Wan2.2 runs per-clip in a subprocess "
                "(~2-5 min per clip)…",
                {
                    "trigger": trigger,
                    "videos_requested": [s["name"] for s in video_specs],
                },
            ))
            yield self._record(AgentEvent(
                "activity", "generating_videos",
                {
                    "label": "generating cutscene videos",
                    "requested": len(video_specs),
                    "produced": 0,
                },
            ))
            if self._video_generator is None:
                self._video_generator = try_load_video_generator()
            if self._video_generator is None:
                yield self._record(AgentEvent("activity", "idle"))
                yield self._record(AgentEvent(
                    "info",
                    "video backend not reachable (macOS: .venv-video/"
                    "mlx-gen missing; Linux: torch/diffusers missing) — "
                    "proceeding without cutscenes."
                ))
            else:
                session_videos_dir = (
                    self.out_path.parent / f"{self._session_id}_videos"
                )
                try:
                    produced = await asyncio.to_thread(
                        generate_videos,
                        video_specs,
                        session_videos_dir,
                        video_generator=self._video_generator,
                        asset_paths=self._session_assets,
                    )
                except Exception as e:
                    yield self._record(AgentEvent("activity", "idle"))
                    yield self._record(AgentEvent(
                        "info",
                        f"video generation crashed: {e!r} — proceeding without."
                    ))
                    produced = {}
                new_video_paths = dict(produced)
                self._session_videos.update(produced)
                per_video = getattr(
                    self._video_generator, "last_stats", None,
                ) or []
                self._trace({
                    "kind": "videos_generated",
                    "trigger": trigger,
                    "requested": len(video_specs),
                    "produced": len(produced),
                    "names": list(produced.keys()),
                    "session_dir": str(session_videos_dir),
                    "per_video": per_video,
                })
                yield self._record(AgentEvent(
                    "videos",
                    f"{len(produced)}/{len(video_specs)} generated",
                    {
                        "trigger": trigger,
                        "requested": len(video_specs),
                        "produced": len(produced),
                        "session_dir": str(session_videos_dir),
                        "paths": {n: str(p) for n, p in produced.items()},
                        "per_video": per_video,
                    },
                ))
                yield self._record(AgentEvent("activity", "idle"))
                if produced:
                    yield self._record(AgentEvent(
                        "info",
                        f"generated {len(produced)}/"
                        f"{len(video_specs)} cutscene videos at "
                        f"{session_videos_dir}",
                        {"videos": {n: str(p) for n, p in produced.items()}},
                    ))
                    for s in per_video:
                        if isinstance(s, dict) and not s.get("error"):
                            secs = float(s.get("gen_seconds") or 0.0)
                            tag = " (cached)" if s.get("cache_hit") else (
                                " (i2v)" if s.get("i2v") else ""
                            )
                            yield self._record(AgentEvent(
                                "info",
                                f"  video {s.get('name','?')}: "
                                f"{secs:.0f}s{tag}",
                            ))
                if len(produced) < len(video_specs):
                    failed = [
                        s for s in per_video
                        if isinstance(s, dict) and s.get("error")
                    ]
                    for s in failed:
                        yield self._record(AgentEvent(
                            "info",
                            f"  - video {s.get('name','?')}: "
                            f"{str(s.get('error',''))[:400]}"
                        ))

        # Mid-session only: synthesize a feedback line that re-emits the
        # asset/sound paths block, so the model's next user turn sees
        # the new files via the existing _flush_user_injections channel.
        # Phase A doesn't need this — the first-build assembler already
        # renders these blocks inline.
        if trigger == "mid_session" and (
            new_asset_paths or new_sound_paths or new_video_paths
        ):
            # Mark the iter as having done real preparatory work, so a
            # follow-up "no usable code" outcome doesn't get charged
            # against max_iters. See _media_regenerated_this_iter init.
            self._media_regenerated_this_iter = True
            # MK trace 20260517_220025 fix: when an existing asset is
            # regenerated with the SAME name, the file on disk has
            # already been replaced — the existing drawImage(ASSETS.X)
            # call picks up the new pixels automatically. Injecting the
            # heavy ASSETS_LOADER block in that case told the model "add
            # a new loader entry", which on a small model led to a 30-
            # minute repeating asset-prompt stream. Split into:
            #   - already_in_html: regen confirmation only
            #   - new_to_html: full loader block
            html_for_refs = self._current_file or ""
            html_asset_refs = self._scan_html_for_asset_refs(html_for_refs)
            html_sound_refs = self._scan_html_for_sound_refs(html_for_refs)
            already_assets = {
                n: p for n, p in new_asset_paths.items()
                if n in html_asset_refs
            }
            new_assets = {
                n: p for n, p in new_asset_paths.items()
                if n not in html_asset_refs
            }
            already_sounds = {
                n: p for n, p in new_sound_paths.items()
                if n in html_sound_refs
            }
            new_sounds = {
                n: p for n, p in new_sound_paths.items()
                if n not in html_sound_refs
            }
            asset_replacements = {
                n: {
                    "old_hash": pre_asset_hashes.get(n),
                    "new_hash": self._file_hash16(p),
                    "changed": (
                        pre_asset_hashes.get(n) is not None
                        and self._file_hash16(p) is not None
                        and pre_asset_hashes.get(n) != self._file_hash16(p)
                    ),
                }
                for n, p in already_assets.items()
            }
            sound_replacements = {
                n: {
                    "old_hash": pre_sound_hashes.get(n),
                    "new_hash": self._file_hash16(p),
                    "changed": (
                        pre_sound_hashes.get(n) is not None
                        and self._file_hash16(p) is not None
                        and pre_sound_hashes.get(n) != self._file_hash16(p)
                    ),
                }
                for n, p in already_sounds.items()
            }

            msg_parts: list[str] = []
            if already_assets or already_sounds:
                # Compact confirmation; no loader pattern, no procedural-
                # vs-sprite warnings, no orientation reminder. The
                # existing draw / Audio() call already references these
                # names by path, so the regen is transparent.
                lines = [
                    "================ MEDIA REGEN COMPLETE ================",
                    "These names already exist in your HTML. The files on",
                    "disk have been REPLACED with newly generated content.",
                    "No JS patch is required — the existing drawImage() /",
                    "new Audio() calls will pick up the new pixels / audio",
                    "automatically on the next Chromium load.",
                ]
                if already_assets:
                    lines.append("")
                    lines.append("Regenerated sprites:")
                    for n in sorted(already_assets):
                        rep = asset_replacements.get(n) or {}
                        status = (
                            "changed"
                            if rep.get("changed") is True
                            else "unchanged/unknown"
                        )
                        lines.append(f"  - {n} ({status})")
                if already_sounds:
                    lines.append("")
                    lines.append("Regenerated sounds:")
                    for n in sorted(already_sounds):
                        rep = sound_replacements.get(n) or {}
                        status = (
                            "changed"
                            if rep.get("changed") is True
                            else "unchanged/unknown"
                        )
                        lines.append(f"  - {n} ({status})")
                lines.append(
                    "======================================================"
                )
                msg_parts.append("\n".join(lines))

            if new_assets or new_sounds or new_video_paths:
                # New names → emit the full loader block so the model
                # knows how to wire them in via a <patch>.
                blocks: list[str] = []
                if new_assets:
                    blocks.append(render_asset_paths_block(
                        new_assets, self.out_path,
                    ))
                if new_sounds:
                    new_looping_subset = {
                        n for n in new_looping if n in new_sounds
                    }
                    blocks.append(render_sound_paths_block(
                        new_sounds, self.out_path,
                        looping_names=new_looping_subset,
                    ))
                if new_video_paths:
                    # Videos are always treated as new — the loader block
                    # carries the full <video>-overlay wiring pattern.
                    blocks.append(render_video_paths_block(
                        new_video_paths, self.out_path,
                    ))
                blocks = [b for b in blocks if b]
                if blocks:
                    msg_parts.append(
                        "Mid-session asset/sound/video additions — load "
                        "these in your next patch and use them where "
                        "appropriate. The files exist on disk now:\n\n"
                        + "\n\n".join(blocks)
                    )

            if msg_parts:
                # Agent-generated media notice — queued via the internal
                # channel so the unhonored-asset-request detector can't
                # mistake it for a user art request (8 spurious reprompts
                # in trace 20260612_004616).
                self._queue_internal_feedback("\n\n".join(msg_parts))
                self._trace({
                    "kind": "midsession_asset_injection_queued",
                    "asset_names": list(new_asset_paths.keys()),
                    "sound_names": list(new_sound_paths.keys()),
                    "video_names": list(new_video_paths.keys()),
                    "already_in_html_assets": sorted(already_assets.keys()),
                    "already_in_html_sounds": sorted(already_sounds.keys()),
                    "new_assets": sorted(new_assets.keys()),
                    "new_sounds": sorted(new_sounds.keys()),
                    "asset_replacements": asset_replacements,
                    "sound_replacements": sound_replacements,
                })

    # -- main loop ----------------------------------------------------------

    async def run(
        self,
        goal: str,
        *,
        continuation: bool = False,
        plan_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Drive a planning + iteration session.

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
        if not continuation:
            self._goal = goal
            self._continuation_feedback = ""
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
            # Wikipedia research is OFF by default — empirical test on
            # 10 representative goals (asteroids, pacman, donkey kong,
            # space invaders, missile command, street fighter, doom,
            # snake, 2d roguelike, tetris) returned 0 matches in ~38s
            # cumulative latency. Filter is too strict: opensearch +
            # word-overlap rejection drops everything. Until the matcher
            # is rewritten, the lookup is pure tax with no benefit.
            # Toggle via the `/wiki on` slash command in chat.py
            # (mirrors `/wait` style). Trace 20260519_111209
            # (build-a-complete-playable-2d-r) is the canonical "agent
            # looked frozen for ~110s with no trace events" case this
            # guards against — see the visibility events below for the
            # opt-in path.
            if self._research_enabled:
                yield self._record(AgentEvent(
                    "phase", "research",
                ))
                yield self._record(AgentEvent(
                    "activity", "research",
                    {"label": "looking up reference"},
                ))
                yield self._record(AgentEvent(
                    "info",
                    "researching goal on Wikipedia "
                    "(/wiki on; 8s/request, 6s typical)…",
                ))
                try:
                    import research as _research
                    reference_block = await asyncio.to_thread(
                        _research.fetch, goal,
                    ) or ""
                except Exception as e:
                    yield self._record(AgentEvent(
                        "info", f"research lookup failed: {e!r}"
                    ))
                yield self._record(AgentEvent("activity", "idle"))
                self._trace({
                    "kind": "research_attempted",
                    "enabled_via": "/wiki on",
                    "got_reference": bool(reference_block),
                    "reference_chars": len(reference_block),
                })
            else:
                self._trace({
                    "kind": "research_skipped",
                    "reason": "disabled_by_default",
                    "hint": "type /wiki on to enable",
                })

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
                    plan_msg = self._p.plan_instruction(
                        reference_block=reference_block,
                        goal=goal,
                        force_minimal_first_build=fmfb,
                        **from_seed_kwargs,
                    )
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
                {"role": "user", "content": plan_msg},
            ]

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
            self._plan_retry_done = False
            self._probe_quality_retry_done = False
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
                yield self._record(AgentEvent("activity", "idle"))
                _nc = self._extract_criteria(retry_reply) if retry_reply else None
                _np = self._extract_probes(retry_reply) if retry_reply else None
                if retry_reply and (_nc or _np):
                    self._messages.append({
                        "role": "assistant", "content": retry_reply,
                        "model_role": _retry_role,
                        "model_name": self.get_backend(_retry_role).info.model,
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
            # would be wrong (CLAUDE.md "harness signal must be right"). The
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
                        "  {\"name\":\"input_moves_player\",\"expr\":\""
                        "(async()=>{if(!window.state||!state.player)return "
                        "false;const x0=state.player.x;window.dispatchEvent("
                        "new KeyboardEvent('keydown',{code:'ArrowRight',"
                        "bubbles:true}));await new Promise(r=>setTimeout(r,"
                        "250));window.dispatchEvent(new KeyboardEvent("
                        "'keyup',{code:'ArrowRight',bubbles:true}));return "
                        "state.player.x!==x0;})()\"}\n"
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
            # Visual-playtest auto-probes injection. Run AFTER the
            # model's own <probes> are parsed (or skipped) so the
            # injected probes ride alongside whatever the model wrote.
            # Deterministic safety net for the mechanism — even if the
            # model's probes miss the failure class (e.g. mortal-
            # kombat 2026-05-24 had no facing assertion), the injected
            # probe catches it. See VisualPlaytestRecipe.auto_probes.
            self._maybe_inject_visual_playtest_auto_probes()

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
            self._trace({
                "kind": "plan_summary",
                "criteria_chars": len(self._criteria or ""),
                "probe_count": len(self._probes or []),
                "probe_names": [p.get("name") for p in (self._probes or [])][:20],
                "coverage_gaps": getattr(self, "_planning_coverage_gaps", []),
                "probe_quality_ratio": (
                    getattr(self, "_probe_quality", {}) or {}
                ).get("ratio"),
            })
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
                # First build = plan-stage equivalent: model is choosing
                # the implementation shape, broad context helps.
                pb_block = self._retrieve_playbook_block(
                    goal, code=seed_html, stage="plan",
                )
                opening_block, opening_hits = self._retrieve_opening_book_block(
                    goal, stage="plan",
                )
                self._active_opening_book_recipes = opening_hits
                pb_kwargs = {"playbook_block": pb_block} if pb_block else {}
                build_msg = self._p.seed_build_instruction(
                    seed_html, str(self.seed_file), **pb_kwargs,
                )
                if opening_block:
                    build_msg = (
                        f"{opening_block}\n\n"
                        "Use the opening-book recipes above as verified "
                        "implementation and test guidance.\n\n"
                        + build_msg
                    )
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
                )
                opening_block, opening_hits = self._retrieve_opening_book_block(
                    goal, stage="plan",
                )
                self._active_opening_book_recipes = opening_hits
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
                    **pb_kwargs,
                )
                if opening_block:
                    build_msg = (
                        f"{opening_block}\n\n"
                        "Use the opening-book recipes above as verified "
                        "implementation and test guidance.\n\n"
                        + build_msg
                    )
                # Capability-round item 1: component skill library —
                # tested mechanics snippets injected beside the opening
                # book so the first build adapts working code instead of
                # synthesizing the same machinery from prose.
                components_block = self._retrieve_components_block(
                    goal, stage="plan", k=3,
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
                prelude = "\n\n".join(
                    b for b in (asset_block, sound_block, video_block) if b
                )
                if prelude:
                    build_msg = prelude + "\n\n" + build_msg
                local_nudge = self._local_first_build_nudge()
                if local_nudge:
                    build_msg = local_nudge + "\n\n" + build_msg
                probe_nudge = self._probe_quality_nudge()
                if probe_nudge:
                    build_msg = probe_nudge + "\n\n" + build_msg
                # Mirror the seed-file branch: apply scope lock from
                # the initial goal so iter 1 of a strict goal honors it.
                build_msg = self._apply_initial_goal_scoping(goal, build_msg)
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
            if self._step_mode and iteration > start_iter:
                yield self._record(AgentEvent(
                    "await_user",
                    f"step-mode: iter {iteration - 1} complete — "
                    "Enter to continue, or type feedback",
                    {"just_finished_iter": iteration - 1},
                ))
                while self._step_pause_should_wait():
                    await asyncio.sleep(0.1)
                # Ship requested during the pause (Ctrl+D / 'done') — re-enter
                # the iter loop so the top-of-loop force_done check exits.
                if self._user_force_done:
                    continue
                if self.has_pending_user_input() and not self._feedback_deferred_last_turn:
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
            if not use_bon and self._should_escalate_stuck_bon(
                self._stuck_streak, self.best_of_n, self._stuck_bon_escalations,
                last_report=self._last_test_report,
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
                            # No trailing newline — Anthropic 400s on final
                            # assistant whitespace; backend also rstrip()s.
                            reply_prefill = "<diagnose>"
                        elif (not self._current_file) and self._force_first_build_prefill:
                            # First-build rescue after a no-code turn.
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
                "preview": reply[:600],
            })
            violation = self._scoped_reply_violation(reply)
            if violation:
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
                if not self._current_file:
                    # Keep first-build rescue armed until code lands.
                    self._force_first_build_prefill = True
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
                    self._messages.append({
                        "role": "user",
                        "content": self._flush_user_injections(
                            materialize_msg
                            + "\nThe file on disk is UNCHANGED (still the last "
                            "working version). Re-emit ONLY the corrected "
                            "<patch> block(s); count the braces in SEARCH vs "
                            "REPLACE before answering."
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
                    self._trace({
                        "kind": "no_usable_code",
                        "plan_only": plan_only,
                        "probes_only": probes_only,
                        "media_only": media_only,
                        "consecutive_plan_only": self._consecutive_plan_only,
                        "has_existing_file": bool(self._current_file),
                        "identical_repeat": identical_repeat,
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
                            prior_loop_kind=self._last_stream_loop_kind,
                            prior_loop_line=self._last_stream_loop_line,
                            is_local_backend=(
                                self._backend.info.name in {"mlx", "ollama"}
                            ),
                            materialize_reject_reason=materialize_msg or "",
                            identical_repeat=identical_repeat,
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
                fail_reasons: list[str] = []
                if page_errors:
                    fail_reasons.append(f"page_errors:{len(page_errors)}")
                if console_errors:
                    fail_reasons.append(f"console_errors:{len(console_errors)}")
                if soft_warnings:
                    fail_reasons.append(f"soft_warnings:{len(soft_warnings)}")
                if report.get("frozen_canvas"):
                    fail_reasons.append("frozen_canvas")
                entity_check = report.get("entity_render_check") or {}
                missing_entities = (entity_check.get("missing") or []) if isinstance(entity_check, dict) else []
                _static_action = report.get("static_action")
                summary_payload = {
                    "kind": "iter_summary",
                    "iteration": iteration,
                    "ok": bool(report.get("ok")),
                    "probes_passed": probes_passed,
                    "probes_total": probes_total,
                    "soft_warnings_count": len(soft_warnings),
                    "page_errors_count": len(page_errors),
                    "console_errors_count": len(console_errors),
                    "frozen_canvas": bool(report.get("frozen_canvas")),
                    "entity_missing_count": len(missing_entities),
                    # Verification observability (so a future trace answers
                    # "did the critic even see an action?" with one grep):
                    "action_frame_captured": bool(report.get("screenshot_action")),
                    "action_key": report.get("action_key"),
                    "static_action": _static_action,
                    "fail_reason": ",".join(fail_reasons) or "ok",
                    # Higher signal-to-noise debugging (2026-05-31): record WHY
                    # it blocked (the actual soft-warning texts, not just a
                    # count), the non-blocking warnings, the frozen-canvas
                    # false-positive classifier, and any queued feedback that a
                    # non-ok iter will defer — so "stuck + user request starved"
                    # is visible from this one event.
                    "soft_warnings": [str(w)[:160] for w in soft_warnings[:6]],
                    "warnings": [str(w)[:140] for w in (report.get("warnings") or [])[:4]],
                    "frozen_canvas_input_responsive": report.get("frozen_canvas_input_responsive"),
                    "pending_feedback": [fb[:120] for fb in (self._pending_feedback or [])[:4]],
                    "pending_feedback_count": len(self._pending_feedback or []),
                }
                self._trace(summary_payload)
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
            # Visual-progress judge: auto-runs when a local MLX-VLM is
            # discoverable on disk (honors the "never silent cloud calls"
            # rule — `_run_vision_judge` skips cleanly when no local VLM
            # is found, and does NOT fall back to Anthropic). The user
            # can still invoke a cloud judge explicitly via `/check with
            # <model>` in chat.py. Disable entirely with VISION_JUDGE=0.
            if after_bytes is not None:
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
                            if critique:
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
                    try:
                        await self._run_vision_judge(after_bytes, iteration)
                    except Exception as exc:
                        self._trace({
                            "kind": "vision_judge_error",
                            "iteration": iteration,
                            "error": str(exc),
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
            # iters (probes_ok) and respects the /feedback toggle + the
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

            said_done = bool(_DONE_RE.search(reply))
            regressed = (self._previous_report_ok is True) and (not report["ok"])

            # Track stuck-streak — used by v1's fix prompt to switch to
            # the "5-7 different sources" reflection ladder after repeat
            # failures on the same goal.
            if report["ok"]:
                self._stuck_streak = 0
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
                self._pending_scoped_check_keywords = []
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
            # Item 3 (context discipline): wrap the report turn in collapse
            # sentinels so it shrinks to a 3-line digest once superseded.
            next_user = self._wrap_report_block(next_user, report)
            # Adaptive temperature: failed → low (precision). Clean+keep-going
            # path goes through post_clean which says "prefer done".
            self._fix_mode = not report["ok"]
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
                yield self._record(AgentEvent("activity", "idle"))
                self._messages.append({"role": "assistant", "content": reply})
                self._extract_and_queue_lookups(reply)
                self._dump_conversation()
                self._trace({
                    "kind": "assistant_reply",
                    "iteration": self.max_iters + 1,
                    "len": len(reply),
                })
                violation = self._scoped_reply_violation(reply)
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
                if _DONE_RE.search(exit_reply):
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
        # Phase 3: a <done/> over a structurally broken file flips this
        # to ok=False even when an earlier iter happened to ship a best.
        # The model's notes claimed success; the file says otherwise.
        ok_outcome = self.best_path.exists() and not self._exit_done_over_broken_file
        self._record_session_outcome(ok=ok_outcome)
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
        pb_block = self._retrieve_playbook_block(
            self._goal, code=self._current_file, stage="code",
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
        # Capability-round item 1: fix-turn component injection. Query is
        # the BLOCKER text (failed probes / errors), not the goal, so a
        # snippet only appears when it matches the actual failure. k=1.
        blocker_query = self._report_blocker_query(report)
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
        # Per-run summary.md (2026-06-10): render the iter table from the
        # jsonl trace so a failed run is diagnosable in one screen instead
        # of grepping the full log. Best-effort; never blocks the outcome.
        self._write_run_summary()

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
