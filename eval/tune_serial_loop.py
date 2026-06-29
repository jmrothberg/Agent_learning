#!/usr/bin/env python3
"""Serial VLM tuning loop — one game at a time, visible browser, in-process MLX.

Ops guide (launch batch, triage, artifact paths): eval/OPERATIONS.md

Default: fully unattended — no pause between games, no per-game wall timeout,
no auto-step on test failure (runs to completion or natural agent exit).

    cd /Users/jonathanrothberg/Agent_learning
    MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/tune_serial_loop.py \\
        --goals-file eval/tune_serial10_goals.txt \\
        --out-dir games/tune_serial10/run_01

    # Overnight + Cursor watcher fixes between games (NO Enter in Terminal):
    .venv/bin/python eval/tune_serial_loop.py ... --wait-for-monitor 1800

    # Manual Enter pause (legacy):
    .venv/bin/python eval/tune_serial_loop.py --pause-between-games
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.inter_game_sync import wait_for_ready, write_pending  # noqa: E402

import backend as backend_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CODER = REPO_ROOT / "coder.py"
DEFAULT_GOALS = REPO_ROOT / "eval" / "tune_serial10_goals.txt"
DEFAULT_OUT_ROOT = REPO_ROOT / "games" / "tune_serial10"
CHECKPOINT_NAME = "tune_checkpoint.json"


@dataclass
class JobResult:
    index: int
    label: str
    goal: str
    cmd: list[str]
    exit_code: int
    duration_s: float
    out_path: str | None = None
    failure_classes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    attempts: int = 1
    outcome: str = "fresh_fail"


def _load_goals(args) -> list[tuple[str, str]]:
    goals: list[str] = list(args.goal or [])
    goals_file = args.goals_file or str(DEFAULT_GOALS)
    if goals_file:
        text = Path(goals_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            goals.append(line)
    if not goals:
        return []
    out: list[tuple[str, str]] = []
    for i, g in enumerate(goals, 1):
        slug = "".join(c if c.isalnum() else "_" for c in g.lower())[:32].strip("_")
        label = f"{i:02d}_{slug or f'job_{i:02d}'}"
        out.append((label, g))
    return out


def _build_cmd(*, goal: str, out_path: Path, args) -> list[str]:
    cmd = [
        sys.executable,
        str(CODER),
        goal,
        "--backend",
        args.backend,
        "--out",
        str(out_path),
        "--max-iters",
        str(args.max_iters),
        "--best-of-n",
        str(args.best_of_n),
        "--stall-seconds",
        str(args.stall_seconds),
    ]
    if not args.no_vlm_critique:
        cmd.append("--vlm-critique")
    cmd.append("--no-auto-step")
    if args.headless:
        cmd.append("--headless")
    if args.model:
        cmd.extend(["--model", args.model])
    return cmd


def _is_game_delivered(out_path: Path) -> bool:
    """True when a prior run left a playable artifact (resume / crash recovery).

    NOTE: this is intentionally lenient (any .html >500 bytes) and is used ONLY
    for resume/skip decisions — it does NOT decide PASS. See `_counts_as_pass`.
    """
    best = out_path.with_name(out_path.stem + ".best.html")
    if best.is_file() and best.stat().st_size > 500:
        return True
    return out_path.is_file() and out_path.stat().st_size > 500


def _trace_last_iter_ok(out_path: Path) -> bool | None:
    """Read the newest trace for this game and return the LAST iter_summary's
    `ok` flag (None when no iter_summary exists). Same trace-locating logic as
    `_triage_trace`."""
    stem = Path(out_path).stem
    candidates = sorted(
        (REPO_ROOT / "games").glob(f"**/traces/{stem}__run_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    last_ok: bool | None = None
    for line in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "iter_summary":
            last_ok = bool(ev.get("ok"))
    return last_ok


def _counts_as_pass(out_path: Path) -> bool:
    """PASS predicate (EVAL-ONLY, serial10 games 4 & 6 were falsely PASS).

    A game counts as PASS only when it actually shipped something verified:
      - a `.best.html` exists (the agent only writes it on a clean ship), OR
      - the LAST iter_summary in the trace reported ok=True.
    A bare `.html` left on disk after every iteration FAILED is NOT a PASS.
    This changes scoring ONLY — it does not affect how any game is built.
    """
    best = out_path.with_name(out_path.stem + ".best.html")
    if best.is_file() and best.stat().st_size > 500:
        return True
    return _trace_last_iter_ok(out_path) is True


def _classify_outcome(out_path: Path) -> str:
    """Verdict after a subprocess run (not resume-skip)."""
    last_ok = _trace_last_iter_ok(out_path)
    if last_ok is True:
        return "fresh_pass"
    best = out_path.with_name(out_path.stem + ".best.html")
    if best.is_file() and best.stat().st_size > 500:
        return "artifact_pass"
    return "fresh_fail"


def _outcome_counts(results: list[JobResult]) -> dict[str, int]:
    keys = ("fresh_pass", "artifact_pass", "fresh_fail", "skipped")
    counts = {k: 0 for k in keys}
    for r in results:
        o = r.outcome if r.outcome in counts else "fresh_fail"
        counts[o] += 1
    return counts


def _delivered_outcomes() -> frozenset[str]:
    return frozenset({"fresh_pass", "artifact_pass", "skipped"})


def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {"completed_labels": [], "results": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"completed_labels": [], "results": []}


def _save_checkpoint(
    *,
    path: Path,
    completed_labels: list[str],
    results: list[JobResult],
    jobs_total: int,
) -> None:
    payload = {
        "completed_labels": completed_labels,
        "completed_count": len(completed_labels),
        "jobs_total": jobs_total,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _triage_trace(out_path: Path | None) -> dict[str, int]:
    """Count failure_class tags in the newest trace under games/."""
    if out_path is None:
        return {}
    stem = Path(out_path).stem
    trace_dir = REPO_ROOT / "games"
    candidates = sorted(
        trace_dir.glob(f"**/traces/{stem}__run_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    counts: dict[str, int] = {}
    for line in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        fc = ev.get("failure_class")
        if fc:
            counts[fc] = counts.get(fc, 0) + 1
    return counts


def _write_summary(
    *,
    path: Path,
    model: str,
    results: list[JobResult],
    status: str,
    vlm_critique: bool = True,
) -> None:
    counts = _outcome_counts(results)
    summary = {
        "status": status,
        "backend": "mlx-in-process",
        "model": model,
        "vlm_critique": vlm_critique,
        "headless": False,
        "jobs": len(results),
        "passed": counts["fresh_pass"],
        "fresh_passed": counts["fresh_pass"],
        "artifact_passed": counts["artifact_pass"],
        "fresh_failed": counts["fresh_fail"],
        "skipped": counts["skipped"],
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _reconcile_exit_code(
    exit_code: int,
    tail_lines: list[str],
    out_path: Path,
) -> tuple[int, list[str]]:
    """SIGSEGV/SIGABRT during atexit after a delivered game → treat as success."""
    notes: list[str] = []
    if exit_code == 0:
        return exit_code, notes
    if _counts_as_pass(out_path):
        text = "\n".join(tail_lines)
        completed = "Final game saved to:" in text or "DONE —" in text
        if completed or exit_code < 0:
            notes.append(
                f"exit={exit_code} — verified artifact on disk, counted as PASS"
            )
            return 0, notes
    return exit_code, notes


async def _run_job(
    *,
    index: int,
    label: str,
    goal: str,
    cmd: list[str],
    out_path: Path,
    env: dict[str, str],
    job_timeout: float,
) -> JobResult:
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    notes: list[str] = []
    tail_lines: list[str] = []

    async def _stream_stdout() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            tail_lines.append(text.rstrip("\n"))

    stream_task = asyncio.create_task(_stream_stdout())
    exit_code = 0
    try:
        if job_timeout > 0:
            await asyncio.wait_for(proc.wait(), timeout=job_timeout)
        else:
            await proc.wait()
        exit_code = proc.returncode or 0
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        exit_code = 124
        notes.append(f"job_timeout: exceeded {job_timeout:.0f}s — subprocess killed")
    finally:
        await stream_task
    dt = time.monotonic() - t0
    if tail_lines:
        notes.append("last line: " + tail_lines[-1][:200])
    exit_code, recon_notes = _reconcile_exit_code(exit_code, tail_lines, out_path)
    notes.extend(recon_notes)
    failure_classes = _triage_trace(out_path)
    return JobResult(
        index=index,
        label=label,
        goal=goal,
        cmd=cmd,
        exit_code=exit_code,
        duration_s=dt,
        out_path=str(out_path),
        failure_classes=failure_classes,
        notes=notes,
    )


async def main_async(args) -> int:
    jobs = _load_goals(args)
    if not jobs:
        print("no goals selected", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "tune_summary.json"
    checkpoint_path = out_root / CHECKPOINT_NAME

    child_env = os.environ.copy()
    child_env.pop("MLX_SERVER_URL", None)
    child_env["LLM_BACKEND"] = args.backend
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    if args.model:
        child_env["MLX_MODEL"] = args.model

    try:
        info = backend_mod.detect_backend(args.backend)
    except Exception as e:  # noqa: BLE001
        print(f"backend resolution failed ({args.backend!r}): {e}", file=sys.stderr)
        return 2

    model = args.model or info.model
    vlm_on = not args.no_vlm_critique

    if args.dry_run:
        for label, goal in jobs:
            out_path = out_root / f"{label}.html"
            cmd = _build_cmd(goal=goal, out_path=out_path, args=args)
            print(" ".join(cmd))
        return 0

    print(
        f"tune_serial_loop · backend={args.backend} · model={model!r} · "
        f"jobs={len(jobs)} · vlm-critique={'ON' if vlm_on else 'OFF'} · no-auto-step · "
        f"resume={args.resume} · retries={args.retries} · "
        f"headless={args.headless} · stall_seconds={args.stall_seconds} · "
        f"job_timeout={'none' if args.job_timeout <= 0 else f'{args.job_timeout:.0f}s'}"
    )
    print(f"artifacts: {out_root}")
    print("Unattended serial run — crash recovery ON, always advances to next game.\n")

    completed_labels: list[str] = []
    results: list[JobResult] = []
    if args.resume:
        ck = _load_checkpoint(checkpoint_path)
        completed_labels = list(ck.get("completed_labels") or [])
        for raw in ck.get("results") or []:
            try:
                results.append(JobResult(**raw))
            except TypeError:
                pass

    for i, (label, goal) in enumerate(jobs, 1):
        out_path = out_root / f"{label}.html"
        if args.resume and (label in completed_labels or _is_game_delivered(out_path)):
            print(f"=== skip {i}/{len(jobs)} · {label} (already delivered) ===")
            if not any(r.label == label for r in results):
                skip_result = JobResult(
                    index=i,
                    label=label,
                    goal=goal,
                    cmd=[],
                    exit_code=0,
                    duration_s=0.0,
                    out_path=str(out_path),
                    outcome="artifact_pass",
                    notes=["resume skip — artifact on disk"],
                )
                results.append(skip_result)
                print(f"\n[artifact_pass] {label} · skipped (already delivered)")
            if label not in completed_labels:
                completed_labels.append(label)
                _save_checkpoint(
                    path=checkpoint_path,
                    completed_labels=completed_labels,
                    results=results,
                    jobs_total=len(jobs),
                )
            continue

        cmd = _build_cmd(goal=goal, out_path=out_path, args=args)
        print(f"=== game {i}/{len(jobs)} · {label} ===")
        print(f"goal: {goal[:120]}{'…' if len(goal) > 120 else ''}\n")

        max_attempts = 1 + max(0, args.retries)
        result: JobResult | None = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(
                    f"\n↻ retry {attempt}/{max_attempts} for {label} "
                    f"(prior exit={result.exit_code if result else '?'}) — "
                    f"sleep {args.retry_delay:.0f}s\n",
                    flush=True,
                )
                await asyncio.sleep(args.retry_delay)
            result = await _run_job(
                index=i,
                label=label,
                goal=goal,
                cmd=cmd,
                out_path=out_path,
                env=child_env,
                job_timeout=args.job_timeout,
            )
            result.attempts = attempt
            if result.exit_code == 0 or _counts_as_pass(out_path):
                if result.exit_code != 0 and _counts_as_pass(out_path):
                    result.exit_code = 0
                    result.notes.append(
                        "verified artifact on disk after failed exit — artifact_pass"
                    )
                break

        assert result is not None
        result.outcome = _classify_outcome(out_path)
        if any(
            "verified artifact on disk after failed exit" in n for n in result.notes
        ) and result.outcome == "fresh_fail" and _counts_as_pass(out_path):
            result.outcome = "artifact_pass"
        results.append(result)
        fc_str = ", ".join(f"{k}={v}" for k, v in sorted(result.failure_classes.items()))
        print(
            f"\n[{result.outcome}] {label} · {result.duration_s:.0f}s · exit={result.exit_code}"
            + (f" · attempts={result.attempts}" if result.attempts > 1 else "")
            + (f" · failure_class: {fc_str}" if fc_str else "")
        )
        for note in result.notes:
            print(f"  note: {note}")

        if result.outcome in _delivered_outcomes():
            if label not in completed_labels:
                completed_labels.append(label)

        _save_checkpoint(
            path=checkpoint_path,
            completed_labels=completed_labels,
            results=results,
            jobs_total=len(jobs),
        )
        _write_summary(
            path=summary_path, model=model, results=results,
            status="running", vlm_critique=vlm_on,
        )

        if i < len(jobs) and args.wait_for_monitor > 0:
            trace_hits = sorted(out_dir.glob(f"traces/{label}__run_*.jsonl"))
            write_pending(
                out_dir,
                {
                    "label": label,
                    "goal_index": i,
                    "jobs_total": len(jobs),
                    "outcome": result.outcome,
                    "exit_code": result.exit_code,
                    "failure_classes": result.failure_classes,
                    "trace": str(trace_hits[-1].relative_to(REPO_ROOT))
                    if trace_hits
                    else None,
                    "html": str(out_path.relative_to(REPO_ROOT))
                    if out_path.exists()
                    else None,
                },
            )
            print(
                f"\nWaiting for watcher (inter_game_ready.json, timeout "
                f"{args.wait_for_monitor:.0f}s) — triage + fix in Cursor, then:\n"
                f"  .venv/bin/python eval/tune_inter_game_ready.py "
                f"--out-dir {out_dir.relative_to(REPO_ROOT)} --note '…'\n",
                flush=True,
            )
            released = wait_for_ready(
                out_dir, timeout_s=args.wait_for_monitor, poll_s=5.0,
            )
            if not released:
                print(
                    f"warning: monitor wait timed out after {args.wait_for_monitor:.0f}s "
                    f"— continuing to next game without confirmed fix",
                    file=sys.stderr,
                )
        elif i < len(jobs) and args.pause_between_games:
            try:
                input("\nPause — triage trace, apply general fixes, Enter for next game… ")
            except EOFError:
                print("(stdin closed — continuing)", file=sys.stderr)

    counts = _outcome_counts(results)
    all_done = len(completed_labels) >= len(jobs)
    _write_summary(
        path=summary_path, model=model, results=results,
        status="done" if all_done else "incomplete",
        vlm_critique=vlm_on,
    )
    _save_checkpoint(
        path=checkpoint_path,
        completed_labels=completed_labels,
        results=results,
        jobs_total=len(jobs),
    )
    print(
        f"\n{len(completed_labels)}/{len(jobs)} checkpoint complete · "
        f"{counts['fresh_pass']}/{len(jobs)} fresh pass · "
        f"{counts['artifact_pass']} artifact pass · "
        f"{counts['fresh_fail']} fresh fail · "
        f"summary: {summary_path}"
    )
    return 0 if all_done else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal", action="append", help="Build goal (repeatable)")
    ap.add_argument(
        "--goals-file",
        default=str(DEFAULT_GOALS),
        help=f"One goal per line (default: {DEFAULT_GOALS.name})",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_ROOT / "run_01"),
        help="Output directory for HTML + traces",
    )
    ap.add_argument(
        "--backend",
        default="mlx",
        help="LLM backend for child coder.py (default mlx in-process; never mlx-server here)",
    )
    ap.add_argument("--model", default=None, help="MLX model path or HF id (sets MLX_MODEL in child)")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--best-of-n", type=int, default=1)
    ap.add_argument(
        "--stall-seconds",
        type=float,
        default=1200.0,
        help="Per-stream activity stall budget passed to coder.py (default 1200s)",
    )
    ap.add_argument(
        "--job-timeout",
        type=float,
        default=0.0,
        help="Wall-clock cap per game subprocess (default 0 = no limit — run to completion)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless (not recommended for tuning)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip games already in tune_checkpoint.json or with .best.html (default ON)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-run every game from scratch",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Extra attempts per game after crash/fail (default 2 → up to 3 tries)",
    )
    ap.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Seconds to wait before retrying a crashed game (default 30)",
    )
    ap.add_argument(
        "--pause-between-games",
        action="store_true",
        help="Wait for Enter between games (use --wait-for-monitor for overnight + agent fixes)",
    )
    ap.add_argument(
        "--wait-for-monitor",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "After each game, block until eval/tune_inter_game_ready.py writes "
            "inter_game_ready.json (Cursor watcher triage). 0=off (default). "
            "Typical overnight: 1800 (30 min cap per game handoff)."
        ),
    )
    ap.add_argument(
        "--no-vlm-critique",
        action="store_true",
        help="Disable structured visual critic (default: ON for serial tuning)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print child commands and exit")
    args = ap.parse_args()
    if args.backend == "mlx-server":
        print(
            "tune_serial_loop uses in-process MLX for VLM — do not pass --backend mlx-server",
            file=sys.stderr,
        )
        return 2
    if args.no_vlm_critique:
        print(
            "warning: --no-vlm-critique set — orientation/art defects will not be vision-checked",
            file=sys.stderr,
        )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
