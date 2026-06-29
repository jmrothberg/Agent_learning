"""Pure helpers extracted from agent.py for readability.

Seed/media scan, HTML normalize/extract regexes, compaction constants, and
related pure functions. Moved VERBATIM from `agent.py` (no behavior change).
`agent.py` re-exports public names so tests keep `from agent import _DONE_RE`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from tools import run_micro_probes

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


# Trace 20260612_171752: cosmetic sprite-audit findings that may gate ok on
# their FIRST occurrence but never indicate a behaviorally-broken build.
# A report whose only gating soft_warnings are from this family — with all
# probes passing and zero errors — is still worth saving as best.html.
_COSMETIC_SPRITE_WARNING_PREFIXES = (
    "ASSETS_LOADED_BUT_UNDRAWN",
    "ACTION_DRAWN_NOT_SPRITED",
    "CODE_DRAWN_OVER_SPRITE",
)


def _report_green_except_cosmetic_sprites(report: dict) -> bool:
    """True when a test report failed `ok` solely on cosmetic sprite-family
    soft_warnings while all behavioral signals are green: every probe
    passed, no console/page errors. Such a build is playable and must not
    be lost (trace 20260612_171752 ended best_exists=False on a 7/7-probes
    game because ACTION_DRAWN_NOT_SPRITED held ok=False every iter).
    """
    if report.get("ok"):
        return False
    probes = report.get("probes") or []
    if not probes or not all(p.get("ok") for p in probes):
        return False
    if report.get("errors") or report.get("page_errors") \
            or report.get("console_errors"):
        return False
    softs = report.get("soft_warnings") or []
    if not softs:
        return False
    return all(
        any(prefix in w for prefix in _COSMETIC_SPRITE_WARNING_PREFIXES)
        for w in softs
    )


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

# First-build placeholder rescue threshold. An all-comment/elision stub
# strips to ~0 chars of code; even a 59-char one-liner toy game strips to
# well above this. Kept deliberately low (content-based, NOT size-based)
# so a genuinely tiny real game never trips it.
_PLACEHOLDER_FIRST_BUILD_MIN_CODE = 24


def _is_placeholder_first_build(html: str) -> bool:
    """True when a FIRST-build `<html_file>` declared a <canvas> game but
    its inline <script> body/bodies have effectively NO executable code —
    only comments, whitespace, or `...` elisions.

    Lets the caller route that first build into the EXISTING prefill RETRY
    rescue (`_force_first_build_prefill`) instead of shipping a dead stub
    to Chromium. Dragon's-lair trace 20260621_091419 iter 1 emitted a
    593-byte canvas+comment-only stub that `_detect_skeleton_payload` let
    through because it lacked the `{ ... }` placeholder marker that detector
    requires (its condition #4). This is the marker-free, canvas-required
    sibling — it does NOT change any termination threshold; it only changes
    how the NEXT first-build attempt starts.

    Conservative by design:
      - <canvas> required (pure-DOM apps are not our case -> False),
      - ALL inline <script> bodies are concatenated, so a CDN
        `<script src=...>` first tag cannot mask a real inline body
        (protects the working three.js DOOM one-shot),
      - size ceiling (`_SKELETON_MAX_BYTES`): any sizable file is treated
        as a real build and never inspected.
    """
    if not html:
        return False
    if len(html) >= _SKELETON_MAX_BYTES:
        return False  # sizable -> a real build, never a stub
    low = html.lower()
    if "<canvas" not in low:
        return False  # pure-DOM intent -> exempt (keep DOM-app exemption)
    bodies = re.findall(
        r"<script\b[^>]*>(.*?)</script\s*>", html, re.IGNORECASE | re.DOTALL,
    )
    body = "\n".join(bodies)
    # Measure REAL code volume: drop comments, `...` elisions, whitespace.
    stripped = re.sub(r"//[^\n]*", "", body)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"\.\.\.", "", stripped)
    stripped = re.sub(r"\s+", "", stripped)
    return len(stripped) < _PLACEHOLDER_FIRST_BUILD_MIN_CODE


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
