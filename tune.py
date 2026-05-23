"""tune.py — battery runner for the HTML/JS coding agent.

The inner loop for tuning the agent: run a fixed list of test goals against
a chosen model + prompt-version, collect per-test results, and write a
comparable summary so prompt / playbook / probe iterations are measurable.

Usage:
    python tune.py run                          # quick: 2 iters, single candidate
    python tune.py run --full                   # 4 iters, best-of-N=2
    python tune.py run --model qwen3.6:35b
    python tune.py run --prompt-version v1
    python tune.py run --tests asteroids,snake  # subset
    python tune.py list                         # past runs
    python tune.py show <run_id>                # show a run's SUMMARY.md
    python tune.py diff <run_a> <run_b>         # compare two runs

Artifacts per run land under games/tune/run_<TIMESTAMP>/:
    manifest.json               # model, prompt_version, mode, all results inline
    SUMMARY.md                  # human-readable
    <slug>/<slug>.html          # final game (if any)
    <slug>/<slug>.best.html     # last passing version (if any)
    <slug>/result.json          # per-test result
    <slug>/traces/<slug>.jsonl  # full agent trace for postmortem
    <slug>/snapshots/<slug>/    # per-iter HTML snapshots
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import AgentEvent, GameAgent
from tools import LiveBrowser


REPO_ROOT = Path(__file__).resolve().parent
TUNE_ROOT = REPO_ROOT / "games" / "tune"
DEFAULT_BATTERY = TUNE_ROOT / "battery.jsonl"

# qwen3.6:27b is the iteration model (faster than 35b; same architecture).
# Override with --model if you want to validate at 35b before declaring a
# tuning round done.
DEFAULT_MODEL = os.environ.get("TUNE_MODEL", "qwen3.6:27b")

# Fixed seed/temperature behavior left to the agent. The harness already
# uses 0.25 in fix_mode and 0.7 in explore — those are tuned and we don't
# want to override them here.

# Default budgets per mode. Tuned for the qwen3.6 27B/35B class.
QUICK_MAX_ITERS = 2
QUICK_BEST_OF_N = 1
FULL_MAX_ITERS = 4
FULL_BEST_OF_N = 2

# Model timeout scaling — delegates to chat.py:resolve_session_timeouts
# so the battery uses the same fail-open scale as live runs. Previously
# this had its own hardcoded substring table that drifted out of sync
# and used the old 60s/90s defaults — which silently killed any
# battery comparison that used a 27B+ MLX model on the small bracket.
def _timeouts_for_model(model: str) -> tuple[float, float, int]:
    """Return (stall_s, overall_s, num_ctx) for the model.

    8192 num_ctx matches Ollama's default load size so we don't force
    a model reload on every request during battery runs.
    """
    from chat import resolve_session_timeouts
    stall_s, overall_s = resolve_session_timeouts(model)
    return stall_s, overall_s, 8192


# ---------------------------------------------------------------------------
# Battery loading
# ---------------------------------------------------------------------------


@dataclass
class BatteryTest:
    slug: str
    goal: str
    notes: str = ""

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "BatteryTest":
        return cls(
            slug=str(obj["slug"]).strip(),
            goal=str(obj["goal"]).strip(),
            notes=str(obj.get("notes", "")).strip(),
        )


def load_battery(path: Path) -> list[BatteryTest]:
    out: list[BatteryTest] = []
    if not path.exists():
        raise SystemExit(f"battery file not found: {path}")
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
            out.append(BatteryTest.from_obj(obj))
        except Exception as e:
            raise SystemExit(f"battery {path}:{ln} parse error: {e}")
    if not out:
        raise SystemExit(f"battery {path} is empty")
    return out


# ---------------------------------------------------------------------------
# Per-test result
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    slug: str
    goal: str
    model: str
    prompt_version: str
    mode: str                    # "quick" | "full"
    max_iters: int
    best_of_n: int
    passed: bool = False
    first_pass_iter: int | None = None
    iters_used: int = 0
    final_errors: list[str] = field(default_factory=list)
    final_soft_warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    best_html_path: str | None = None
    final_html_path: str | None = None
    trace_path: str | None = None

    def short_status(self) -> str:
        if self.passed:
            return f"PASS @ iter {self.first_pass_iter}"
        if self.iters_used == 0:
            return "FAIL (no iter run)"
        what = []
        if self.final_errors:
            what.append(f"{len(self.final_errors)}err")
        if self.final_soft_warnings:
            what.append(f"{len(self.final_soft_warnings)}iss")
        return "FAIL " + ("/".join(what) if what else "")


# ---------------------------------------------------------------------------
# One test run
# ---------------------------------------------------------------------------


async def _run_one(
    test: BatteryTest,
    *,
    model: str,
    prompt_version: str,
    mode: str,
    max_iters: int,
    best_of_n: int,
    skeleton_mode: str,
    memory_root: str,
    run_dir: Path,
    browser: LiveBrowser,
    feature_kwargs: dict[str, Any] | None = None,
) -> TestResult:
    feature_kwargs = feature_kwargs or {}
    """Drive GameAgent on one battery goal; capture pass/fail + artifacts.

    Each test gets its own subdir under the run dir. The agent's natural
    artifact tree (traces/snapshots/best.html) lands inside that subdir
    because agent.py derives its paths from out_path.stem + parent.
    """
    test_dir = run_dir / test.slug
    test_dir.mkdir(parents=True, exist_ok=True)
    out_path = test_dir / f"{test.slug}.html"

    stall_s, overall_s, num_ctx = _timeouts_for_model(model)

    agent_kwargs: dict[str, Any] = dict(
        model=model,
        out_path=out_path,
        browser=browser,
        max_iters=max_iters,
        best_of_n=best_of_n,
        num_ctx=num_ctx,
        stall_seconds=stall_s,
        overall_seconds=overall_s,
        prompt_version=prompt_version,
        skeleton_mode=skeleton_mode,
        memory_root=memory_root,
        **feature_kwargs,
    )
    agent = GameAgent(**agent_kwargs)

    result = TestResult(
        slug=test.slug,
        goal=test.goal,
        model=model,
        prompt_version=prompt_version,
        mode=mode,
        max_iters=max_iters,
        best_of_n=best_of_n,
        trace_path=str(agent.trace_path),
    )

    t0 = time.monotonic()
    last_report: dict[str, Any] | None = None
    final_html = ""

    try:
        async for ev in agent.run(test.goal):
            if ev.kind == "phase":
                # phase events look like "iteration 1/2"; pick the iter
                # counter out so we can show progress.
                if ev.text.startswith("iteration"):
                    try:
                        result.iters_used = int(ev.text.split()[1].split("/")[0])
                    except Exception:
                        pass
            elif ev.kind == "test":
                last_report = ev.data
                ok = ev.data.get("ok", False)
                if ok and result.first_pass_iter is None:
                    result.first_pass_iter = result.iters_used or 1
                    result.passed = True
            elif ev.kind == "code":
                final_html_path = ev.text
                if final_html_path and Path(final_html_path).exists():
                    result.final_html_path = final_html_path
            elif ev.kind == "done":
                # Agent's own success/failure determination. We trust it.
                pass
            elif ev.kind == "error":
                result.notes.append(f"agent error: {ev.text[:200]}")
                break
    except KeyboardInterrupt:
        result.notes.append("interrupted by user")
        raise
    except Exception as e:
        result.notes.append(f"exception: {type(e).__name__}: {str(e)[:200]}")

    result.duration_s = round(time.monotonic() - t0, 1)

    if last_report is not None:
        result.final_errors = list(last_report.get("errors") or [])[:5]
        result.final_soft_warnings = list(last_report.get("soft_warnings") or [])[:5]

    if agent.best_path.exists():
        result.best_html_path = str(agent.best_path)
        # If we have a best.html the agent actually passed at some point.
        result.passed = result.passed or True

    # Write per-test result.json
    (test_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )

    return result


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------


def _summary_md(
    run_dir: Path,
    *,
    model: str,
    prompt_version: str,
    mode: str,
    started_at: str,
    duration_total_s: float,
    results: list[TestResult],
) -> str:
    n_pass = sum(1 for r in results if r.passed)
    n = len(results)
    avg_iters = (
        round(sum(r.iters_used for r in results) / max(n, 1), 2) if results else 0
    )
    avg_dur = (
        round(sum(r.duration_s for r in results) / max(n, 1), 1) if results else 0
    )
    lines = [
        f"# tune run — {run_dir.name}",
        "",
        f"- Started:        `{started_at}`",
        f"- Model:          `{model}`",
        f"- Prompt version: `{prompt_version}`",
        f"- Mode:           `{mode}`",
        f"- Total duration: `{duration_total_s:.1f}s`",
        f"- Pass rate:      **{n_pass}/{n}**",
        f"- Avg iters/test: `{avg_iters}`",
        f"- Avg dur/test:   `{avg_dur}s`",
        "",
        "## Per-test results",
        "",
        "| # | Test | Status | Iters | Dur (s) | Notes |",
        "|---|------|--------|-------|---------|-------|",
    ]
    for i, r in enumerate(results, 1):
        notes = "; ".join(r.notes)[:100]
        if r.final_errors and not notes:
            notes = f"err: {r.final_errors[0][:80]}"
        elif r.final_soft_warnings and not notes:
            notes = f"iss: {r.final_soft_warnings[0][:80]}"
        lines.append(
            f"| {i} | `{r.slug}` | {r.short_status()} | "
            f"{r.iters_used}/{r.max_iters} | {r.duration_s} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


_FEATURE_FLAGS = {
    "prefill":            "use_prefill",
    "vlm_critique":       "use_vlm_critique",
    "double_screenshot":  "use_double_screenshot",
    "architect":          "use_architect_split",
}


def _parse_features(text: str) -> dict[str, bool]:
    """Convert --features 'prefill,vlm_critique' into the GameAgent kwargs.

    Special tokens: 'all' = all of them; '' / 'none' = no features.
    Unknown tokens are ignored with a stderr warning.
    """
    kwargs: dict[str, bool] = {f: False for f in _FEATURE_FLAGS.values()}
    text = (text or "").strip().lower()
    if not text or text == "none":
        return kwargs
    if text == "all":
        return {f: True for f in _FEATURE_FLAGS.values()}
    for tok in (s.strip() for s in text.split(",")):
        if not tok:
            continue
        if tok not in _FEATURE_FLAGS:
            print(f"warn: unknown feature {tok!r}", file=sys.stderr)
            continue
        kwargs[_FEATURE_FLAGS[tok]] = True
    return kwargs


async def cmd_run(args) -> int:
    battery_path = Path(args.battery) if args.battery else DEFAULT_BATTERY
    tests = load_battery(battery_path)

    if args.tests:
        wanted = {s.strip() for s in args.tests.split(",") if s.strip()}
        tests = [t for t in tests if t.slug in wanted]
        if not tests:
            print(f"no battery tests matched --tests={args.tests}", file=sys.stderr)
            return 2

    mode = "full" if args.full else "quick"
    max_iters = FULL_MAX_ITERS if mode == "full" else QUICK_MAX_ITERS
    if args.max_iters is not None:
        max_iters = args.max_iters
    best_of_n = FULL_BEST_OF_N if mode == "full" else QUICK_BEST_OF_N
    if args.best_of_n is not None:
        best_of_n = args.best_of_n

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = TUNE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Memory root: by default each tune run gets its own copy that starts
    # from the seeded playbook + default skeleton — keeps tune results
    # attributable to agent changes rather than the user's organic memory
    # accumulation.
    if args.memory_root:
        memory_root = str(Path(args.memory_root).resolve())
    else:
        memory_root = str(run_dir / "_memory")
        Path(memory_root).mkdir(parents=True, exist_ok=True)
        # Seed the playbook from the current shared file if one exists,
        # so we test against the agent's accumulated rules even though
        # skeletons are isolated.
        shared_playbook = REPO_ROOT / "memory" / "playbook.jsonl"
        if shared_playbook.exists():
            try:
                (Path(memory_root) / "playbook.jsonl").write_text(
                    shared_playbook.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except Exception:
                pass

    started_at = datetime.now().isoformat(timespec="seconds")

    feature_kwargs = _parse_features(args.features)
    feature_str = (
        ",".join(k.replace("use_", "") for k, v in feature_kwargs.items() if v)
        or "none"
    )

    print(
        f"== tune run · model={args.model} · prompt={args.prompt_version} · "
        f"mode={mode} · iters={max_iters} · best_of_n={best_of_n} · "
        f"skeleton={args.skeleton_mode} · features={feature_str} · "
        f"tests={len(tests)}"
    )
    print(f"== artifacts: {run_dir}")
    print(f"== memory:    {memory_root}")
    print()

    browser = LiveBrowser(viewport=(800, 600), run_seconds=3.0, headless=args.headless)
    try:
        await browser.start()
    except Exception as e:
        print(f"could not launch Chromium: {e}", file=sys.stderr)
        print("hint: `playwright install chromium`", file=sys.stderr)
        return 2

    results: list[TestResult] = []
    t0 = time.monotonic()
    rc = 0
    try:
        for i, test in enumerate(tests, 1):
            print(f"[{i:>2}/{len(tests)}] {test.slug:<22}  ", end="", flush=True)
            try:
                r = await _run_one(
                    test,
                    model=args.model,
                    prompt_version=args.prompt_version,
                    mode=mode,
                    max_iters=max_iters,
                    best_of_n=best_of_n,
                    skeleton_mode=args.skeleton_mode,
                    memory_root=memory_root,
                    run_dir=run_dir,
                    browser=browser,
                    feature_kwargs=feature_kwargs,
                )
            except KeyboardInterrupt:
                print("INTERRUPTED")
                rc = 130
                break
            except Exception as e:
                print(f"CRASH: {type(e).__name__}: {e}")
                # synthesize a failure result so we still record something
                r = TestResult(
                    slug=test.slug, goal=test.goal, model=args.model,
                    prompt_version=args.prompt_version, mode=mode,
                    max_iters=max_iters, best_of_n=best_of_n,
                    notes=[f"runner exception: {type(e).__name__}: {e}"],
                )
                (run_dir / test.slug).mkdir(parents=True, exist_ok=True)
                (run_dir / test.slug / "result.json").write_text(
                    json.dumps(asdict(r), indent=2), encoding="utf-8"
                )
            results.append(r)
            print(f"{r.short_status():<22}  iters={r.iters_used}/{r.max_iters}  "
                  f"{r.duration_s:>6.1f}s")

            # Write the manifest after every test so partial results survive
            # an interrupt.
            _write_manifest(run_dir, args.model, args.prompt_version, mode,
                            started_at, time.monotonic() - t0, results)

    finally:
        try:
            await browser.close()
        except Exception:
            pass

    duration_total = time.monotonic() - t0
    _write_manifest(run_dir, args.model, args.prompt_version, mode,
                    started_at, duration_total, results)
    summary = _summary_md(run_dir, model=args.model,
                          prompt_version=args.prompt_version, mode=mode,
                          started_at=started_at, duration_total_s=duration_total,
                          results=results)
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")

    print()
    n_pass = sum(1 for r in results if r.passed)
    print(f"== {n_pass}/{len(results)} passing  ({duration_total:.1f}s total)")
    print(f"== summary: {run_dir / 'SUMMARY.md'}")

    # ---- optional auto-learn ---------------------------------------------
    if args.auto_learn and rc == 0:
        await _auto_learn(run_dir, results, args, memory_root)

    return rc


async def _auto_learn(run_dir: Path, results, args, memory_root) -> None:
    """Walk this run's traces, run the offline Reflector + Curator, write
    bullet deltas to the chosen playbook.

    This is the closed-loop learning step: every battery run can leave
    the playbook a little smarter. By default writes to the per-run
    isolated memory; pass --learn-shared to write to the project's
    canonical playbook.
    """
    from learner import walk_traces, reflect_one, curate
    from memory import Playbook
    import backend as backend_mod

    pb_root = (
        Path(REPO_ROOT) / "games" / "game-memory"
        if args.learn_shared else Path(memory_root)
    )
    playbook = Playbook(root=pb_root)
    playbook.ensure()

    # Pull every trace from inside the run dir.
    sessions = walk_traces([run_dir])
    if not sessions:
        print("== auto-learn: no traces found")
        return

    rmodel = args.reflector_model or args.model
    print(f"== auto-learn: {len(sessions)} session(s) → reflector={rmodel} → "
          f"playbook={pb_root}")
    # Reflector runs on Ollama by default — it reads existing traces and
    # doesn't benefit from the MLX speed advantage. Honor LLM_BACKEND if
    # the user has explicitly set it.
    info = backend_mod.detect_backend()
    if rmodel:
        info = backend_mod.BackendInfo(
            name=info.name, model=rmodel,
            source=f"--reflector-model {rmodel!r}",
            endpoint=info.endpoint,
        )
    bk = backend_mod.make_backend(info)
    existing = playbook.load_all()
    proposals = []
    for s in sessions:
        prop = await reflect_one(s, existing, backend=bk)
        proposals.append(prop)
        nb = len(prop.get("new_bullets") or [])
        nc = len(prop.get("counter_updates") or [])
        print(f"   {s.session_id:<40}  new={nb}  counters={nc}")

    log = curate(proposals, playbook, apply=True)
    print(f"== curator: added/refreshed {len(log.added)}, "
          f"counter updates {len(log.updated_counters)}, "
          f"skipped {len(log.skipped)}")


def _write_manifest(run_dir, model, prompt_version, mode, started_at,
                    duration_s, results) -> None:
    obj = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "model": model,
        "prompt_version": prompt_version,
        "mode": mode,
        "duration_s": round(duration_s, 1),
        "pass_count": sum(1 for r in results if r.passed),
        "test_count": len(results),
        "results": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(obj, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Subcommands: list, show, diff
# ---------------------------------------------------------------------------


def cmd_list(_args) -> int:
    if not TUNE_ROOT.exists():
        print("no tune runs yet")
        return 0
    runs = sorted([p for p in TUNE_ROOT.glob("run_*") if p.is_dir()])
    if not runs:
        print("no tune runs yet")
        return 0
    print(f"{'run_id':<32}  {'model':<18}  {'prompt':<8}  {'mode':<6}  pass  dur")
    for r in runs:
        m = _read_manifest(r)
        if m is None:
            print(f"{r.name:<32}  (no manifest.json)")
            continue
        pass_ratio = f"{m['pass_count']}/{m['test_count']}"
        print(
            f"{m['run_id']:<32}  {m['model']:<18}  {m['prompt_version']:<8}  "
            f"{m['mode']:<6}  {pass_ratio:<5} {m['duration_s']:>6.1f}s"
        )
    return 0


def cmd_show(args) -> int:
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        return 2
    summary = run_dir / "SUMMARY.md"
    if not summary.exists():
        print(f"no SUMMARY.md in {run_dir}", file=sys.stderr)
        return 2
    print(summary.read_text(encoding="utf-8"))
    return 0


def cmd_analyze(args) -> int:
    """Cluster failure signatures across a run; for each cluster, surface
    which playbook bullets WOULD have matched the failing goal.

    Useful for "which bullets to add / promote next?" decisions: if 4 of
    10 tests failed with the same blank-canvas signature and no current
    bullet matches, you need a new bullet.
    """
    from collections import Counter
    from memory import Playbook
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        return 2
    m = _read_manifest(run_dir)
    if m is None:
        print(f"no manifest in {run_dir}", file=sys.stderr)
        return 2

    pb_root = Path(args.playbook_root) if args.playbook_root else (run_dir / "_memory")
    if not (pb_root / "playbook.jsonl").exists():
        pb_root = Path(REPO_ROOT) / "games" / "game-memory"
    playbook = Playbook(root=pb_root)
    bullets = playbook.load_all()

    # ---- per-test failure signature
    def _strip(msg: str) -> str:
        # collapse numbers, hex, file URLs so similar bugs cluster.
        msg = re.sub(r"\b\d+\b", "N", msg)
        msg = re.sub(r"file://\S+", "FILE", msg)
        msg = re.sub(r"@\d+:\d+", "@N:N", msg)
        return msg[:200]

    sig_to_tests: dict[str, list[str]] = {}
    sig_to_goal: dict[str, list[str]] = {}
    for r in m["results"]:
        if r.get("passed"):
            continue
        for e in (r.get("final_errors") or []):
            sig = "ERR " + _strip(str(e))
            sig_to_tests.setdefault(sig, []).append(r["slug"])
            sig_to_goal.setdefault(sig, []).append(r["goal"])
        for w in (r.get("final_soft_warnings") or []):
            sig = "ISS " + _strip(str(w))
            sig_to_tests.setdefault(sig, []).append(r["slug"])
            sig_to_goal.setdefault(sig, []).append(r["goal"])
        if not (r.get("final_errors") or r.get("final_soft_warnings")):
            sig_to_tests.setdefault("FAIL (no error captured)", []).append(r["slug"])
            sig_to_goal.setdefault("FAIL (no error captured)", []).append(r["goal"])

    n_pass = m["pass_count"]
    n_total = m["test_count"]
    print(f"== analyze {run_dir.name}  ({n_pass}/{n_total} passing, model={m['model']}, prompt={m['prompt_version']})")

    if not sig_to_tests:
        print("no failures to analyze")
        return 0

    print()
    print(f"Failure clusters ({len(sig_to_tests)}):")
    for sig, slugs in sorted(sig_to_tests.items(), key=lambda kv: -len(kv[1])):
        goals = sig_to_goal[sig]
        print(f"\n  ── {len(slugs)}× cluster ──")
        print(f"  signature: {sig[:160]}")
        print(f"  tests:     {', '.join(slugs)}")
        # match playbook to the joined goal text — bullets whose tags
        # OR content overlap with these failing goals are candidates
        # for "should have applied".
        joined_goal = " ".join(goals)
        hits = playbook.retrieve(joined_goal, k=4)
        if hits:
            print(f"  matching playbook bullets (top {len(hits)}):")
            for h in hits:
                print(f"    {h.score:.3f}  [{h.bullet.id}]  {h.bullet.content[:100]}")
        else:
            print(f"  matching playbook bullets: NONE — candidate for a new bullet.")

    return 0


def cmd_why(args) -> int:
    """Postmortem one test in a run: prompt + plan + iter reports + bullets."""
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        return 2
    test_dir = run_dir / args.slug
    if not test_dir.exists():
        print(f"no test '{args.slug}' in {run_dir.name}", file=sys.stderr)
        return 2
    # Find the trace file. tune.py builds it under <run>/<slug>/traces/<slug>.jsonl
    traces = list((test_dir / "traces").glob("*.jsonl"))
    if not traces:
        print(f"no trace under {test_dir}", file=sys.stderr)
        return 2
    trace = traces[0]
    # Use learner.parse_session for parsing.
    from learner import parse_session
    s = parse_session(trace)

    # Per-test result.
    rj = test_dir / "result.json"
    result = json.loads(rj.read_text()) if rj.exists() else {}

    print(f"== {run_dir.name} / {args.slug}")
    print(f"goal:    {s.goal}")
    print(f"model:   {s.model}")
    print(f"prompt:  {result.get('prompt_version', '?')}")
    print(f"outcome: {'PASS' if s.final_ok else 'FAIL'}  "
          f"(iters {len(s.iters)}, {s.duration_s:.1f}s)")
    if s.skeleton_used:
        score = f" sim={s.skeleton_score:.2f}" if s.skeleton_score is not None else ""
        print(f"skeleton: {s.skeleton_used}{score}")
    if s.bullets_retrieved:
        print(f"bullets retrieved: {len(s.bullets_retrieved)}")
        for bid in s.bullets_retrieved:
            print(f"  - {bid}")
    print()
    print("plan:")
    for line in (s.plan or "(none)").splitlines():
        print(f"  {line}")
    print()
    print("iterations:")
    for it in s.iters:
        tag = "PASS" if it.ok else "FAIL"
        print(f"  iter {it.n} [{tag}]")
        for e in it.errors[:3]:
            print(f"    err: {e}")
        for w in it.soft_warnings[:3]:
            print(f"    iss: {w}")
        if it.diagnose:
            print(f"    diagnose: {it.diagnose[:240]}")
        if it.notes:
            print(f"    notes:    {it.notes}")
    print()
    print("artifacts:")
    if result.get("best_html_path"):
        print(f"  best: {result['best_html_path']}")
    if result.get("final_html_path"):
        print(f"  last: {result['final_html_path']}")
    print(f"  trace: {trace}")
    if (test_dir / "traces" / f"{args.slug}.conversation.md").exists():
        print(f"  conv:  {test_dir / 'traces' / (args.slug + '.conversation.md')}")
    return 0


async def cmd_validate_bullet(args) -> int:
    """A/B-test ONE playbook bullet by running the battery twice with
    the bullet's retrieval-score toggled. Reports the delta and a
    verdict.

    Procedure:
      1. Snapshot the bullet's current `helpful`/`harmful` counters.
      2. Run the battery with the bullet AS-IS (treatment).
      3. Force-downgrade the bullet (`harmful=999`) so the
         `1 + 0.10·tanh(score/5)` quality multiplier flips negative
         and code-stage retrieval drops it. Run the battery again
         (control).
      4. Restore the original counters.
      5. Diff the two manifests; emit a verdict (helps / neutral /
         harms).

    A bullet that does not improve iters-to-clean and does not improve
    pass count is at best neutral. The Reflector's auto-proposals
    (learner.detect_failure_shapes) can be gated on this verdict
    before they get full retrieval weight — the README thesis 'an
    agent that learns from every session' becomes 'an agent that
    learns AND validates from every session.'
    """
    from memory import Playbook  # local import: keeps top-level lean

    pb_root = (
        Path(args.playbook_root) if args.playbook_root
        else (Path(__file__).resolve().parent / "games" / "game-memory")
    )
    pb = Playbook(root=pb_root)
    pb.ensure()
    all_bullets = pb.load_all()
    target = next((b for b in all_bullets if b.id == args.id), None)
    if target is None:
        print(f"no bullet with id={args.id!r} in {pb.path}", file=sys.stderr)
        return 2

    print(f"validating bullet: {args.id}")
    print(f"  current counters: helpful={target.helpful}  harmful={target.harmful}")
    saved_helpful, saved_harmful = target.helpful, target.harmful
    tests_arg = args.tests or "asteroids"

    # Build the args shape cmd_run expects (it reads attrs on the
    # Namespace). Reuse current model / prompt-version / iter caps.
    def _make_run_args(run_id: str) -> argparse.Namespace:
        return argparse.Namespace(
            model=args.model,
            prompt_version=args.prompt_version,
            full=False,
            max_iters=args.max_iters,
            best_of_n=args.best_of_n,
            tests=tests_arg,
            battery=None,
            run_id=run_id,
            headless=True,
            skeleton_mode="default",
            memory_root=str(pb_root),
            auto_learn=False,
            learn_shared=False,
            reflector_model=None,
            features="",
        )

    treatment_id = f"validate_{args.id}_treatment"
    control_id = f"validate_{args.id}_control"

    try:
        # Step 1: treatment (bullet retrieves normally).
        print()
        print(f"[1/2] treatment run (bullet enabled) → {treatment_id}")
        await cmd_run(_make_run_args(treatment_id))

        # Step 2: control (bullet downgraded so retrieval drops it).
        print()
        print(f"[2/2] control run (bullet downgraded) → {control_id}")
        pb.update_counters(
            [args.id], helpful_delta=0, harmful_delta=999,
        )
        await cmd_run(_make_run_args(control_id))
    finally:
        # Always restore the original counters even on crash.
        post = next((b for b in pb.load_all() if b.id == args.id), None)
        if post is not None:
            pb.update_counters(
                [args.id],
                helpful_delta=saved_helpful - post.helpful,
                harmful_delta=saved_harmful - post.harmful,
            )
        print(f"\nrestored {args.id} counters: helpful={saved_helpful}  "
              f"harmful={saved_harmful}")

    # Step 3: load manifests and report.
    t_run = _resolve_run(treatment_id)
    c_run = _resolve_run(control_id)
    if t_run is None or c_run is None:
        print("could not resolve one of the run dirs", file=sys.stderr)
        return 2
    tm = _read_manifest(t_run)
    cm = _read_manifest(c_run)
    if tm is None or cm is None:
        print("missing manifest.json", file=sys.stderr)
        return 2
    t_pass = tm["pass_count"]
    c_pass = cm["pass_count"]
    delta = t_pass - c_pass

    print()
    print("== verdict ==")
    print(f"  treatment (bullet ON):  pass {t_pass}/{tm['test_count']}")
    print(f"  control   (bullet OFF): pass {c_pass}/{cm['test_count']}")
    print(f"  Δ pass: {delta:+d}")
    if delta > 0:
        verdict = "HELPS — bullet should keep current weight."
    elif delta == 0:
        verdict = (
            "NEUTRAL — bullet had no measurable effect on this slice. "
            "Consider broadening the test set, or downgrading if you "
            "want only validated bullets in the playbook."
        )
    else:
        verdict = (
            "HARMS — bullet regressed pass rate. Recommend setting "
            f"`harmful` ≥ 5 on {args.id} so the code-stage retrieval "
            "drops it."
        )
    print(f"  {verdict}")
    print()
    print(f"  full diff: python tune.py diff {control_id} {treatment_id}")
    return 0


def cmd_diff(args) -> int:
    a = _resolve_run(args.run_a)
    b = _resolve_run(args.run_b)
    if a is None or b is None:
        return 2
    ma, mb = _read_manifest(a), _read_manifest(b)
    if ma is None or mb is None:
        print("missing manifest.json in one of the runs", file=sys.stderr)
        return 2

    def by_slug(m):
        return {r["slug"]: r for r in m["results"]}
    da, db = by_slug(ma), by_slug(mb)
    slugs = sorted(set(da) | set(db))

    print(f"A: {ma['run_id']}  (pass {ma['pass_count']}/{ma['test_count']})")
    print(f"B: {mb['run_id']}  (pass {mb['pass_count']}/{mb['test_count']})")
    print(f"Δ pass: {mb['pass_count'] - ma['pass_count']:+d}")
    print()
    print(f"{'slug':<22}  A         B         Δ")
    for s in slugs:
        ra = da.get(s)
        rb = db.get(s)
        sa = ra["passed"] if ra else None
        sb = rb["passed"] if rb else None
        a_str = "PASS" if sa else ("FAIL" if sa is False else " — ")
        b_str = "PASS" if sb else ("FAIL" if sb is False else " — ")
        if sa == sb:
            d = " "
        elif sb and not sa:
            d = "+"        # newly passing
        elif sa and not sb:
            d = "-"        # regression
        else:
            d = "?"
        print(f"{s:<22}  {a_str:<8}  {b_str:<8}  {d}")
    return 0


def _read_manifest(run_dir: Path) -> dict | None:
    p = run_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_run(run_id: str) -> Path | None:
    """Accept full run id, a path, or a partial-prefix match."""
    p = Path(run_id)
    if p.exists() and p.is_dir():
        return p
    direct = TUNE_ROOT / run_id
    if direct.exists():
        return direct
    if not TUNE_ROOT.exists():
        print(f"no tune root yet: {TUNE_ROOT}", file=sys.stderr)
        return None
    matches = [d for d in TUNE_ROOT.glob("run_*") if d.name.startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"no run matching {run_id!r}", file=sys.stderr)
        return None
    print(f"ambiguous {run_id!r}; matches: {[m.name for m in matches]}",
          file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="HTML/JS coding-agent tune rig.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the battery")
    pr.add_argument("--model", default=DEFAULT_MODEL)
    pr.add_argument("--prompt-version", default="v1",
                    help="which prompt module to load; v1 = prompts_v1.py (default, and "
                         "what chat.py / coder.py use). The retired v0 prompts.py was deleted.")
    pr.add_argument("--full", action="store_true",
                    help=f"full mode (max_iters={FULL_MAX_ITERS}, best_of_n={FULL_BEST_OF_N}); "
                         f"default is quick (iters={QUICK_MAX_ITERS}, bon={QUICK_BEST_OF_N})")
    pr.add_argument("--max-iters", type=int, default=None,
                    help="override iter cap")
    pr.add_argument("--best-of-n", type=int, default=None,
                    help="override best-of-N candidate count")
    pr.add_argument("--tests", default=None,
                    help="comma-separated subset of slugs to run")
    pr.add_argument("--battery", default=None,
                    help=f"path to battery JSONL (default {DEFAULT_BATTERY})")
    pr.add_argument("--run-id", default=None,
                    help="explicit run id; default run_<timestamp>")
    pr.add_argument("--headless", action="store_true",
                    help="hide the Chromium window (default: visible — user wants to watch)")
    pr.add_argument("--skeleton-mode", default="default",
                    choices=["retrieve", "default", "default_v2"],
                    help="default (recommended baseline) forces the small "
                         "bundled skeleton; default_v2 forces the larger "
                         "bug-hardened scaffold; retrieve mirrors "
                         "production by pulling the most-similar past win")
    pr.add_argument("--memory-root", default=None,
                    help="memory dir for skeleton retrieval and playbook "
                         "(default: per-run isolated copy under the run dir)")
    pr.add_argument("--auto-learn", action="store_true",
                    help="after the battery finishes, run the offline "
                         "learner over the run's traces to update the "
                         "playbook (writes to --memory-root unless "
                         "--learn-shared is also set)")
    pr.add_argument("--learn-shared", action="store_true",
                    help="with --auto-learn, write playbook updates back "
                         "to the shared memory/playbook.jsonl "
                         "instead of the per-run isolated copy")
    pr.add_argument("--reflector-model", default=None,
                    help="model for the offline reflector (default: same "
                         "as --model, so we don't load a second model)")
    pr.add_argument("--features", default="",
                    help="comma-separated agent feature flags to enable "
                         "(prefill, vlm_critique, double_screenshot, "
                         "architect, all). Default off for fair A/B "
                         "against baseline runs.")

    sub.add_parser("list", help="list past runs")

    ps = sub.add_parser("show", help="print a run's SUMMARY.md")
    ps.add_argument("run_id")

    pd = sub.add_parser("diff", help="diff two runs by pass/fail")
    pd.add_argument("run_a")
    pd.add_argument("run_b")

    pw = sub.add_parser("why", help="postmortem one test in a run")
    pw.add_argument("run_id")
    pw.add_argument("slug")

    pan = sub.add_parser("analyze", help="cluster failures + match playbook bullets")
    pan.add_argument("run_id")
    pan.add_argument("--playbook-root", default=None,
                     help="playbook dir (default: run's isolated _memory, "
                          "fallback to project memory)")

    pv = sub.add_parser(
        "validate-bullet",
        help="A/B-test one playbook bullet by running the battery "
             "twice with the bullet's retrieval-score toggled and "
             "printing the pass-rate delta.",
    )
    pv.add_argument("--id", required=True,
                    help="bullet id to validate (must exist in playbook)")
    pv.add_argument("--tests", default="asteroids",
                    help="comma-separated test slugs to run for the A/B "
                         "(default: asteroids, the canonical regression "
                         "check)")
    pv.add_argument("--model", default=DEFAULT_MODEL)
    pv.add_argument("--prompt-version", default="v1")
    pv.add_argument("--max-iters", type=int, default=None)
    pv.add_argument("--best-of-n", type=int, default=None)
    pv.add_argument("--playbook-root", default=None,
                    help="playbook dir (default: project memory)")

    args = p.parse_args()

    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "diff":
        return cmd_diff(args)
    if args.cmd == "why":
        return cmd_why(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "validate-bullet":
        return asyncio.run(cmd_validate_bullet(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
