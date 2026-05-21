"""learner.py — offline Reflector + Curator over agent traces.

Walks `games/traces/*.jsonl` (or a tune run's traces tree), distills each
session into a compact record, asks an LLM Reflector to propose playbook
deltas (new bullets + counter updates), and applies them via the
deterministic Curator. Mirrors the ACE pattern (arXiv 2510.04618):
localized deltas only, never wholesale rewrites.

Subcommands:
    learner walk [paths...]        # parse traces, print 1-line per session
    learner show <trace_path>      # full structured dump of one session
    learner reflect <paths...>     # run Reflector, print proposed deltas
    learner apply <paths...>       # reflect + curate (writes playbook.jsonl)

Notes:
  - The Reflector dynamically falls back to the currently loaded model
    (via detect_backend()) when no explicit --model is provided.
  - Deltas are applied deterministically: ADD a new bullet if id is
    novel; UPDATE counters on existing bullets; never rewrite content of
    seed bullets without explicit `--allow-overwrite-seed`.
  - Idempotency: re-running on the same traces will increment counters
    again — gate with `--once` or use `--apply` only on new sessions.

Trace events the learner reads (see agent.py):
  session_start, event:phase, event:plan, event:memory, event:test,
  event:diagnose, event:done, event:error, playbook_retrieved,
  session_outcome, mistake_signature, vlm_detected, stream_done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import backend as backend_mod
from backend import Backend
from memory import Bullet, Playbook


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYBOOK_ROOT = REPO_ROOT / "memory"
DEFAULT_TRACES_DIR = REPO_ROOT / "games" / "traces"
# Surgical change: Set default reflector model to None so it dynamically
# falls back to the currently loaded model via detect_backend().
DEFAULT_REFLECTOR_MODEL = None


# ---------------------------------------------------------------------------
# Session shape
# ---------------------------------------------------------------------------


@dataclass
class IterReport:
    n: int
    ok: bool
    errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    diagnose: str = ""
    notes: str = ""


@dataclass
class Session:
    trace_path: Path
    session_id: str = ""
    goal: str = ""
    model: str = ""
    started_at: str = ""
    final_ok: bool | None = None
    iters: list[IterReport] = field(default_factory=list)
    plan: str = ""
    skeleton_used: str = ""
    skeleton_score: float | None = None
    bullets_retrieved: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def _safe_load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return []
    return out


_PLAN_TAG_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.DOTALL | re.IGNORECASE)


def parse_session(path: Path) -> Session:
    """Build a structured Session from one agent trace file."""
    s = Session(trace_path=path)
    rows = _safe_load_jsonl(path)
    if not rows:
        return s

    cur_iter: int = 0
    last_diagnose = ""
    last_notes = ""
    for r in rows:
        kind = r.get("kind") or r.get("event") or ""
        if kind == "session_start":
            s.session_id = str(r.get("session_id") or path.stem)
            s.goal = str(r.get("goal") or "")
            s.model = str(r.get("model") or "")
            s.started_at = str(r.get("ts") or "")
        elif kind == "event":
            ev = r.get("event") or ""
            text = r.get("text_preview") or ""
            data = r.get("data") or {}
            if ev == "memory":
                s.skeleton_used = str(data.get("skeleton") or "")
                s.skeleton_score = data.get("score") if "score" in data else None
            elif ev == "plan":
                m = _PLAN_TAG_RE.search(text)
                s.plan = (m.group(1).strip() if m else text)[:1500]
            elif ev == "phase":
                t = (text or "").strip()
                if t.startswith("iteration"):
                    try:
                        cur_iter = int(t.split()[1].split("/")[0])
                    except Exception:
                        pass
            elif ev == "diagnose":
                last_diagnose = text[:600]
            elif ev == "info" and text.startswith("notes:"):
                last_notes = text[len("notes:"):].strip()[:300]
            elif ev == "test":
                # data is the report dict
                ok = bool(data.get("ok", False))
                errs = list(data.get("errors") or [])[:5]
                soft = list(data.get("soft_warnings") or [])[:5]
                s.iters.append(IterReport(
                    n=cur_iter or len(s.iters) + 1,
                    ok=ok,
                    errors=[str(e)[:240] for e in errs],
                    soft_warnings=[str(w)[:240] for w in soft],
                    diagnose=last_diagnose,
                    notes=last_notes,
                ))
                last_diagnose = ""
                last_notes = ""
            elif ev == "done":
                s.final_ok = True
            elif ev == "error":
                if s.final_ok is None:
                    s.final_ok = False
        elif kind == "playbook_retrieved":
            ids = r.get("ids") or []
            for i in ids:
                if i not in s.bullets_retrieved:
                    s.bullets_retrieved.append(i)
        elif kind == "session_outcome":
            s.final_ok = bool(r.get("ok"))
        elif kind == "stream_done":
            try:
                s.duration_s += float(r.get("duration_s") or 0.0)
            except Exception:
                pass

    if s.final_ok is None:
        s.final_ok = bool(s.iters and s.iters[-1].ok)
    s.duration_s = round(s.duration_s, 1)
    return s


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------


def walk_traces(roots: list[Path]) -> list[Session]:
    """Find every *.jsonl trace under each root and parse it.

    Skips non-trace files (we identify a trace by presence of a
    `session_start` row) so accidental .jsonl artifacts are ignored.
    """
    seen: set[Path] = set()
    sessions: list[Session] = []
    for root in roots:
        if root.is_file() and root.suffix == ".jsonl":
            sessions.append(parse_session(root))
            seen.add(root.resolve())
            continue
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.jsonl")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            s = parse_session(p)
            if s.goal or s.iters:
                sessions.append(s)
    return sessions


def cmd_walk(args) -> int:
    sessions = walk_traces([Path(r) for r in (args.paths or [str(DEFAULT_TRACES_DIR)])])
    if not sessions:
        print("no traces found")
        return 0
    for s in sessions:
        ok = "OK  " if s.final_ok else "FAIL"
        iters = len(s.iters)
        bullets = f"bullets={len(s.bullets_retrieved)}" if s.bullets_retrieved else ""
        print(
            f"{ok}  {s.session_id:<40}  "
            f"iters={iters}  {s.duration_s:>7.1f}s  "
            f"goal={s.goal[:60]!r}  {bullets}".strip()
        )
    n_ok = sum(1 for s in sessions if s.final_ok)
    print(f"\n== {n_ok}/{len(sessions)} sessions OK")
    return 0


def cmd_show(args) -> int:
    p = Path(args.trace_path)
    if not p.exists():
        # Allow passing a session id; search default tree.
        candidates = list(DEFAULT_TRACES_DIR.rglob(f"{args.trace_path}*.jsonl"))
        if not candidates:
            print(f"no trace at {p}", file=sys.stderr)
            return 2
        p = candidates[0]
    s = parse_session(p)
    print(f"trace:    {p}")
    print(f"session:  {s.session_id}")
    print(f"goal:     {s.goal}")
    print(f"model:    {s.model}")
    print(f"started:  {s.started_at}")
    print(f"final:    {'OK' if s.final_ok else 'FAIL'}")
    print(f"duration: {s.duration_s}s")
    print(f"plan:")
    for line in (s.plan or "(none)").splitlines():
        print(f"  {line}")
    if s.bullets_retrieved:
        print(f"bullets retrieved ({len(s.bullets_retrieved)}):")
        for b in s.bullets_retrieved:
            print(f"  - {b}")
    print(f"iters ({len(s.iters)}):")
    for it in s.iters:
        tag = "OK" if it.ok else "FAIL"
        print(f"  iter {it.n} [{tag}]")
        for e in it.errors:
            print(f"    err: {e}")
        for w in it.soft_warnings:
            print(f"    iss: {w}")
        if it.diagnose:
            print(f"    diagnose: {it.diagnose[:200]}")
        if it.notes:
            print(f"    notes:    {it.notes}")
    return 0


# ---------------------------------------------------------------------------
# Reflector — LLM-driven proposal of playbook deltas
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM = """You are an offline reflector for a coding agent
that builds single-file HTML5/JS games. Your job is to look at a single
agent session and propose updates to a persistent "playbook" of rules.

The playbook is a list of structured bullets. Each bullet has:
  - id (kebab-case)
  - tags (short keywords used for retrieval against future goals)
  - content (one paragraph: a concrete, transferable code rule)
  - helpful / harmful counters (incremented based on outcomes)

You propose DELTAS only, never rewrites:
  - new_bullets:      bullets to ADD that capture a transferable lesson
  - counter_updates:  helpful++ for bullets whose content matches what
                      ultimately worked; harmful++ for bullets that
                      were retrieved but the run still failed

ULTRA IMPORTANT rules:
  - Only propose new bullets that are GENERIC and TRANSFERABLE. Do not
    propose bullets that are specific to one goal ("for asteroids X..."
    is fine; "in this game, when the asteroid is named Bob..." is not).
  - Do not duplicate existing playbook bullets. If the lesson is already
    captured by an existing bullet, propose a counter_update instead.
  - Bullets should be ONE PARAGRAPH. Be concrete: name functions,
    variables, conventions, with code-snippet phrasing where helpful.
  - If nothing learnable came out of this session, return empty arrays.

Output STRICT JSON ONLY. No prose outside the JSON. Schema:

{
  "observations": ["short string per significant event"],
  "new_bullets": [
    {"id": "kebab-id", "tags": ["t1","t2"], "content": "one paragraph"}
  ],
  "counter_updates": [
    {"id": "existing-id", "helpful": 1, "harmful": 0, "reason": "..."}
  ]
}
"""


def _format_session_for_reflector(s: Session, existing_bullets: list[Bullet]) -> str:
    lines: list[str] = []
    lines.append(f"GOAL: {s.goal}")
    lines.append(f"MODEL: {s.model}")
    lines.append(f"OUTCOME: {'OK' if s.final_ok else 'FAIL'}")
    lines.append(f"DURATION: {s.duration_s}s across {len(s.iters)} iterations")
    if s.skeleton_used:
        ss = f" (sim={s.skeleton_score:.2f})" if s.skeleton_score is not None else ""
        lines.append(f"SKELETON: {s.skeleton_used}{ss}")
    if s.plan:
        lines.append("\nPLAN:")
        lines.append(s.plan)
    if s.bullets_retrieved:
        lines.append("\nPLAYBOOK BULLETS RETRIEVED THIS SESSION:")
        by_id = {b.id: b for b in existing_bullets}
        for bid in s.bullets_retrieved:
            b = by_id.get(bid)
            if b:
                lines.append(f"  - [{bid}] tags={b.tags}  helpful={b.helpful} harmful={b.harmful}")
                lines.append(f"      {b.content[:200]}")
            else:
                lines.append(f"  - [{bid}] (no longer in playbook)")
    lines.append("\nITERATIONS:")
    for it in s.iters:
        tag = "OK" if it.ok else "FAIL"
        lines.append(f"  iter {it.n} [{tag}]")
        for e in it.errors[:3]:
            lines.append(f"    err: {e}")
        for w in it.soft_warnings[:3]:
            lines.append(f"    iss: {w}")
        if it.diagnose:
            lines.append(f"    diagnose: {it.diagnose[:240]}")
        if it.notes:
            lines.append(f"    notes:    {it.notes}")
    if s.bullets_retrieved:
        lines.append("\nIMPORTANT: when proposing counter_updates, prefer "
                     "the IDs above (they were live this session).")
    lines.append("\nExisting playbook IDs (do NOT duplicate):")
    lines.append(", ".join(b.id for b in existing_bullets))
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


async def reflect_one(
    session: Session,
    existing_bullets: list[Bullet],
    *,
    backend: Backend | None = None,
    model: str | None = None,
    client=None,  # legacy ollama.AsyncClient — kept for backward compat
) -> dict:
    """Run the Reflector on one session; return the parsed JSON proposal.

    Returns an empty proposal on parse / network failure rather than
    raising — the curator will simply have nothing to apply.

    Either pass `backend=` (preferred) or the legacy `model=`/`client=`
    pair (auto-wraps in an OllamaBackend).
    """
    if backend is None:
        # Legacy path: synthesize an OllamaBackend from `model` + `client`.
        if not model:
            raise TypeError("reflect_one requires either backend= or model=")
        backend = backend_mod.make_backend(backend_mod.BackendInfo(
            name="ollama", model=model,
            source="legacy reflect_one(model=...)",
            endpoint=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
        ))
    user_msg = _format_session_for_reflector(session, existing_bullets)
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = await backend.stream_chat(
            messages,
            on_token=lambda _t: None,
            options={"temperature": 0.25, "num_ctx": 8192},
            stall_seconds=120.0,
            overall_seconds=900.0,
            max_retries=1,
        )
        text = (result.text or "").strip()
    except Exception as e:
        return {"observations": [f"reflector exception: {e}"],
                "new_bullets": [], "counter_updates": []}
    m = _JSON_RE.search(text)
    if not m:
        return {"observations": ["reflector returned no JSON"],
                "new_bullets": [], "counter_updates": []}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # Try to fix common JSON sins (trailing commas).
        cleaned = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            obj = json.loads(cleaned)
        except Exception as e:
            return {"observations": [f"json parse failed: {e}"],
                    "new_bullets": [], "counter_updates": []}
    obj.setdefault("observations", [])
    obj.setdefault("new_bullets", [])
    obj.setdefault("counter_updates", [])
    return obj


# ---------------------------------------------------------------------------
# Curator — deterministic merge
# ---------------------------------------------------------------------------


@dataclass
class CurationLog:
    added: list[str] = field(default_factory=list)
    updated_counters: list[tuple[str, int, int]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def curate(
    proposals: list[dict],
    playbook: Playbook,
    *,
    apply: bool = False,
    allow_overwrite_seed: bool = False,
) -> CurationLog:
    """Apply Reflector proposals to a playbook deterministically.

    `proposals` is the list of reflector JSON outputs (one per session).
    `apply=False` means dry-run: log what would change, don't write.
    """
    log = CurationLog()
    existing = {b.id: b for b in playbook.load_all()}

    for prop in proposals:
        for nb in prop.get("new_bullets") or []:
            try:
                bid = str(nb["id"]).strip()
                content = str(nb["content"]).strip()
                tags = [str(t).strip() for t in (nb.get("tags") or []) if str(t).strip()]
            except Exception:
                continue
            if not bid or not content:
                continue
            if bid in existing:
                ex = existing[bid]
                if ex.source == "seed" and not allow_overwrite_seed:
                    log.skipped.append((bid, "seed bullet, not overwriting"))
                    continue
                # Treat ADD-on-existing as a content refresh: keep counters.
                ex.content = content
                ex.tags = tags or ex.tags
                ex.source = "learned"
                if apply:
                    playbook.add(ex)
                log.added.append(bid + " (refreshed)")
                continue
            new_b = Bullet(
                id=bid, content=content, tags=tags, source="learned",
            )
            existing[bid] = new_b
            if apply:
                playbook.add(new_b)
            log.added.append(bid)

        for cu in prop.get("counter_updates") or []:
            try:
                bid = str(cu["id"]).strip()
                helpful = int(cu.get("helpful") or 0)
                harmful = int(cu.get("harmful") or 0)
            except Exception:
                continue
            if not bid:
                continue
            if bid not in existing:
                log.skipped.append((bid, "counter_update for unknown id"))
                continue
            if apply:
                playbook.update_counters([bid],
                                          helpful_delta=helpful,
                                          harmful_delta=harmful)
            log.updated_counters.append((bid, helpful, harmful))

    return log


# ---------------------------------------------------------------------------
# Reflect / apply commands
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deterministic trace-shape detectors (no LLM call)
# ---------------------------------------------------------------------------
#
# Some failure shapes are unambiguous in the trace bytes — a token-
# repetition loop on `flag = false;` assignments, or a `console.error`
# saying `Identifier '<name>' has already been declared`. Asking the
# Reflector LLM to "notice" these shapes wastes a request and depends on
# the LLM having a stable mental model of what the harness already
# detected. Better: scan the raw rows ourselves and emit deterministic
# proposals BEFORE the LLM Reflector runs.
#
# These detectors are the "every-trace teaches the playbook" path. They
# encode the failure shapes from donkey-kong traces 20260516_124628
# (concatenated drafts → duplicate `const` declarations) and
# 20260516_142445 (dead state-reset block → adjacent-line spam).

# Regexes for the two shapes we currently detect.
_DUP_DECL_ERR_RE = re.compile(
    r"Identifier\s+['\"`]?(\w+)['\"`]?\s+has already been declared",
    re.IGNORECASE,
)
# An assignment-style line we'd expect to see repeated when the model
# emits a dead state-reset block (`p.onGirder = false;` shape).
_FLAG_ASSIGN_RE = re.compile(
    r"^[\w.\[\]]+\s*=\s*(?:false|true|null|0|undefined)\s*;?\s*$",
)


def detect_failure_shapes(trace_path: Path) -> list[dict]:
    """Scan a trace for failure shapes that always map to the same
    playbook bullet, and emit deterministic new_bullets proposals.

    Each returned dict matches the Reflector's `new_bullets` schema so
    the Curator can merge it with LLM-proposed deltas without extra
    plumbing. The Curator's existing dedup-by-id (and idempotent
    counter logic) handles "this bullet already exists" → no-op.

    Two detectors today:

    1. Adjacent-line spam on flag-assignment lines. Trigger:
       `stream_done` with `looped=True` AND `loop_kind` in
       {"adjacent_line_spam", "short_line_loop"} AND the captured
       `loop_line` looks like an assignment to a literal. This is the
       donkey-kong 20260516_142445 shape exactly.

    2. Duplicate top-level declaration. Trigger: any `console_error`
       text matches "Identifier 'X' has already been declared". This is
       the donkey-kong 20260516_124628 shape (concatenated drafts).
    """
    out: list[dict] = []
    rows = _safe_load_jsonl(trace_path)
    if not rows:
        return out

    fired_dead_state = False
    fired_dup_decl = False
    # Pending state for the legacy-trace fallback (pre-A1 traces don't
    # carry `loop_kind` / `loop_line`). When we see a `looped=True`
    # `stream_done` immediately followed by a `format_rejection` whose
    # kind is `unclosed_html_file`, that's the same failure shape the
    # adjacency detector now captures — fire the bullet on the
    # combination instead.
    pending_loop_no_metadata = False

    for r in rows:
        kind = r.get("kind") or ""

        # Detector 1: dead-state-reset block.
        if kind == "stream_done" and not fired_dead_state:
            looped = bool(r.get("looped"))
            if looped:
                loop_kind = r.get("loop_kind") or ""
                loop_line = (r.get("loop_line") or "").strip()
                if not loop_kind and not loop_line:
                    # Legacy trace: arm the unclosed-html-file
                    # confirmation below.
                    pending_loop_no_metadata = True
                if (
                    loop_kind in ("adjacent_line_spam", "short_line_loop")
                    and loop_line
                    and _FLAG_ASSIGN_RE.match(loop_line)
                ):
                    out.append({
                        "id": "no-dead-state-reset-fallthrough",
                        "tags": [
                            "code-quality", "anti-pattern",
                            "token-loop", "control-flow",
                        ],
                        "content": (
                            "When a state branch's only effect is to "
                            "re-clear flags that were already cleared "
                            "upstream, DELETE the branch entirely. Do "
                            "NOT pad it with redundant assignments like "
                            "`flag = false; flag = false;`. Adjacent "
                            "identical assignments are a well-known "
                            "token-repetition-loop trigger for local "
                            "LLMs (qwen3.6, DeepSeek-V4) and will cause "
                            "the streaming RepetitionDetector to abort "
                            "the reply mid-emit, leaving an unclosed "
                            "<html_file>."
                        ),
                    })
                    fired_dead_state = True

        # Legacy-trace confirmation: `format_rejection.rejection_kind=
        # unclosed_html_file` immediately after a `looped` `stream_done`
        # is the same shape, even without `loop_kind`/`loop_line`.
        if (
            kind == "format_rejection"
            and pending_loop_no_metadata
            and not fired_dead_state
        ):
            rk = r.get("rejection_kind") or ""
            if rk == "unclosed_html_file":
                out.append({
                    "id": "no-dead-state-reset-fallthrough",
                    "tags": [
                        "code-quality", "anti-pattern",
                        "token-loop", "control-flow",
                    ],
                    "content": (
                        "When a state branch's only effect is to "
                        "re-clear flags that were already cleared "
                        "upstream, DELETE the branch entirely. Do "
                        "NOT pad it with redundant assignments like "
                        "`flag = false; flag = false;`. Adjacent "
                        "identical assignments are a well-known "
                        "token-repetition-loop trigger for local "
                        "LLMs (qwen3.6, DeepSeek-V4) and will cause "
                        "the streaming RepetitionDetector to abort "
                        "the reply mid-emit, leaving an unclosed "
                        "<html_file>."
                    ),
                })
                fired_dead_state = True
                pending_loop_no_metadata = False

        # Detector 2: concatenated drafts → duplicate top-level decl.
        # `test` events carry the console error list; we also accept
        # the raw event:test payload's data.errors field.
        if kind == "event" and r.get("event") == "test":
            data = r.get("data") or {}
            for err in (data.get("errors") or []):
                if _DUP_DECL_ERR_RE.search(str(err)):
                    if not fired_dup_decl:
                        out.append({
                            "id": "no-concatenated-drafts",
                            "tags": [
                                "code-quality", "anti-pattern",
                                "rewrite", "syntax",
                            ],
                            "content": (
                                "When you rewrite an `<html_file>` body, "
                                "delete the previous draft COMPLETELY. "
                                "Duplicate top-level `const` / `let` / "
                                "`function` declarations at the same "
                                "scope crash the script with "
                                "`Identifier '<name>' has already been "
                                "declared`. The micro-probe catches "
                                "this pre-Chromium, but the iter is "
                                "still wasted — emit ONE complete body "
                                "per turn, not a concatenation of "
                                "first-and-second drafts."
                            ),
                        })
                        fired_dup_decl = True
                    break

        # Early exit: both fired, nothing more to find.
        if fired_dead_state and fired_dup_decl:
            break

    return out


async def cmd_reflect(args) -> int:
    return await _run_reflect_apply(args, apply=False)


async def cmd_apply(args) -> int:
    return await _run_reflect_apply(args, apply=True)


async def _run_reflect_apply(args, *, apply: bool) -> int:
    paths = [Path(p) for p in (args.paths or [str(DEFAULT_TRACES_DIR)])]
    sessions = walk_traces(paths)
    if not sessions:
        print("no traces found")
        return 0

    if args.tests:
        wanted = {s.strip() for s in args.tests.split(",") if s.strip()}
        sessions = [s for s in sessions if any(t in s.session_id or t in s.goal for t in wanted)]
        if not sessions:
            print(f"no sessions matched --tests={args.tests}")
            return 0

    if args.failures_only:
        sessions = [s for s in sessions if not s.final_ok]
    if args.successes_only:
        sessions = [s for s in sessions if s.final_ok]

    # Surgical change: Print the exact resolved model inside the print statement
    model_disp = args.model if args.model else "dynamic default"
    print(f"reflecting over {len(sessions)} session(s) using {model_disp} ...")
    pb_root = Path(args.playbook_root or DEFAULT_PLAYBOOK_ROOT)
    playbook = Playbook(root=pb_root)
    playbook.ensure()
    existing = playbook.load_all()

    info = backend_mod.detect_backend()
    if args.model:
        info = backend_mod.BackendInfo(
            name=info.name, model=args.model,
            source=f"--model {args.model!r}",
            endpoint=info.endpoint,
        )
    bk = backend_mod.make_backend(info)
    print(f"reflector backend: {info.name.upper()} · {info.model} [{info.source}]")
    proposals: list[dict] = []
    for i, s in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {s.session_id}  goal={s.goal[:50]!r}  "
              f"{'OK' if s.final_ok else 'FAIL'}")
        prop = await reflect_one(s, existing, backend=bk)
        # Deterministic trace-shape detectors run BEFORE merging into
        # the proposal so their findings get the same Curator dedup
        # treatment. The LLM Reflector still runs (it catches shapes
        # we haven't hardcoded yet), but obvious patterns no longer
        # depend on the LLM noticing them.
        shape_bullets = detect_failure_shapes(s.trace_path)
        if shape_bullets:
            prop.setdefault("new_bullets", []).extend(shape_bullets)
            prop.setdefault("observations", []).extend(
                [f"shape-detector: {b['id']}" for b in shape_bullets]
            )
        n_new = len(prop.get("new_bullets") or [])
        n_cu = len(prop.get("counter_updates") or [])
        n_obs = len(prop.get("observations") or [])
        n_shape = len(shape_bullets)
        suffix = f" (+{n_shape} from shape-detectors)" if n_shape else ""
        print(f"     → obs={n_obs}  new={n_new}  counters={n_cu}{suffix}")
        proposals.append(prop)

    log = curate(proposals, playbook, apply=apply,
                 allow_overwrite_seed=args.allow_overwrite_seed)

    print()
    print(f"== curator log ({'applied' if apply else 'dry-run'}) ==")
    if log.added:
        print(f"  added/refreshed ({len(log.added)}):")
        for x in log.added:
            print(f"    + {x}")
    if log.updated_counters:
        print(f"  counter updates ({len(log.updated_counters)}):")
        for bid, h, hh in log.updated_counters:
            print(f"    Δ {bid}  helpful{h:+d}  harmful{hh:+d}")
    if log.skipped:
        print(f"  skipped ({len(log.skipped)}):")
        for bid, reason in log.skipped:
            print(f"    - {bid}: {reason}")
    if not (log.added or log.updated_counters or log.skipped):
        print("  (no changes)")
    print(f"\nplaybook: {playbook.path}")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Offline learner for the coding agent.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("walk", help="parse trace files, one line per session")
    pw.add_argument("paths", nargs="*",
                    help=f"files or dirs (default {DEFAULT_TRACES_DIR})")

    ps = sub.add_parser("show", help="print a structured dump of one session")
    ps.add_argument("trace_path", help="path or session id prefix")

    pr = sub.add_parser("reflect", help="run Reflector on traces (dry-run, no writes)")
    pr.add_argument("paths", nargs="*",
                    help=f"trace files or dirs (default {DEFAULT_TRACES_DIR})")
    pr.add_argument("--model", default=DEFAULT_REFLECTOR_MODEL,
                    help="Reflector model (default: currently loaded/detected model)")
    pr.add_argument("--tests", default=None,
                    help="comma-separated substrings to filter session ids/goals")
    pr.add_argument("--failures-only", action="store_true",
                    help="only reflect on FAIL sessions")
    pr.add_argument("--successes-only", action="store_true",
                    help="only reflect on OK sessions")
    pr.add_argument("--playbook-root", default=None,
                    help=f"playbook dir (default {DEFAULT_PLAYBOOK_ROOT})")
    pr.add_argument("--allow-overwrite-seed", action="store_true",
                    help="allow ADD ops to overwrite seed-source bullets")

    pa = sub.add_parser("apply", help="run Reflector + Curator (writes playbook)")
    for a in pr._actions:                     # mirror reflect args
        if a.dest in ("help",):
            continue
        # add_action throws if the kwarg combo doesn't match — skip cleanly.
        try:
            pa._add_action(a)
        except Exception:
            pass

    args = p.parse_args()

    if args.cmd == "walk":
        return cmd_walk(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "reflect":
        return asyncio.run(cmd_reflect(args))
    if args.cmd == "apply":
        return asyncio.run(cmd_apply(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
