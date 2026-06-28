#!/usr/bin/env python3
"""Run multiple game builds in parallel against one batched MLX server.

Each job is a separate ``coder.py`` (or ``eval_seed_edits.py``) subprocess —
clients share one ``mlx_lm.server`` load and continuous batching, not N
in-process MLX copies.

Usage:
    # Terminal 1 — start the server once
    .venv/bin/mlx_lm.server --model ~/MLX_Models/Qwen3.6-27B-mxfp8 --port 8080

    # Terminal 2 — parallel builds (2 at a time)
    MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \\
        --jobs 2 --goal "snake wraparound" --goal "breakout paddle"

    # Art-heavy Round 1 — prefer --jobs 1 to avoid MLX stream_stalled under load
    env -u PLAYWRIGHT_BROWSERS_PATH MLX_SERVER_URL=http://127.0.0.1:8080 \\
        .venv/bin/python eval/batch_parallel.py \\
        --jobs 1 --goals-file eval/tune_round1_goals.txt --headless --max-iters 6 \\
        --stall-seconds 1200 --out-dir games/batch_parallel/tune_round1_r2

    # From a goals file (one goal per line; # comments ok)
    MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \\
        --jobs 5 --goals-file my_goals.txt --headless --max-iters 4

    # Seed-edit harness scenarios (patch-only, no browser)
    MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \\
        --seed-edits --jobs 5 --patch-only --max-iters 2
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

import backend as backend_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CODER = REPO_ROOT / "coder.py"
EVAL_SEED = REPO_ROOT / "eval" / "eval_seed_edits.py"
DEFAULT_OUT_ROOT = REPO_ROOT / "games" / "batch_parallel"

SEED_EDIT_SCENARIOS = [
    "bigger_towers",
    "recolor_creeps",
    "faster_creeps",
    "second_path",
    "tower_range_ring",
]


@dataclass
class JobResult:
    label: str
    goal: str
    cmd: list[str]
    exit_code: int
    duration_s: float
    out_path: str | None = None
    notes: list[str] = field(default_factory=list)


def _load_goals(args) -> list[tuple[str, str]]:
    """Return [(label, goal_text), ...]."""
    if args.seed_edits:
        names = list(SEED_EDIT_SCENARIOS)
        if args.names:
            wanted = {n.strip().lower() for n in args.names.split(",")}
            names = [n for n in names if n.lower() in wanted]
        return [(n, n) for n in names]

    goals: list[str] = list(args.goal or [])
    if args.goals_file:
        text = Path(args.goals_file).read_text(encoding="utf-8")
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
        label = slug or f"job_{i:02d}"
        out.append((label, g))
    return out


def _build_cmd(
    *,
    goal: str,
    label: str,
    out_root: Path,
    args,
) -> tuple[list[str], Path | None]:
    py = sys.executable
    if args.seed_edits:
        cmd = [
            py, str(EVAL_SEED),
            "--names", label,
            "--max-iters", str(args.max_iters),
            "--backend", "mlx-server",
        ]
        if args.patch_only:
            cmd.append("--patch-only")
        if args.model:
            cmd.extend(["--model", args.model])
        return cmd, out_root / label

    out_path = out_root / f"{label}.html"
    cmd = [
        py, str(CODER),
        goal,
        "--backend", "mlx-server",
        "--out", str(out_path),
        "--max-iters", str(args.max_iters),
        "--best-of-n", str(args.best_of_n),
        "--stall-seconds", str(args.stall_seconds),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.model:
        cmd.extend(["--model", args.model])
    return cmd, out_path


def _write_summary(
    *,
    summary_path: Path,
    server_url: str,
    model: str,
    jobs_total: int,
    results: list[JobResult],
    wall_s: float,
    status: str = "running",
) -> None:
    passed = sum(1 for r in results if r.exit_code == 0)
    summary = {
        "status": status,
        "server": server_url,
        "model": model,
        "jobs": jobs_total,
        "completed": len(results),
        "passed": passed,
        "duration_s": round(wall_s, 1),
        "results": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


async def _preflight_server(server_url: str, model: str, timeout: float = 90.0) -> None:
    """Fail fast if mlx_lm.server is wedged before launching hour-long builds."""
    import httpx

    async with httpx.AsyncClient(base_url=server_url, timeout=timeout) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                "max_tokens": 4,
                "stream": False,
            },
        )
        r.raise_for_status()


async def _run_job(
    sem: asyncio.Semaphore,
    *,
    label: str,
    goal: str,
    cmd: list[str],
    out_path: Path | None,
    env: dict[str, str],
    job_timeout: float,
) -> JobResult:
    async with sem:
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        notes: list[str] = []
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=job_timeout,
            )
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            exit_code = 124
            notes.append(
                f"job_timeout: exceeded {job_timeout:.0f}s wall — subprocess killed"
            )
            stdout = b""
        dt = time.monotonic() - t0
        if stdout:
            tail = stdout.decode("utf-8", errors="replace").strip().splitlines()
            if tail:
                notes.append("last line: " + tail[-1][:200])
        return JobResult(
            label=label,
            goal=goal,
            cmd=cmd,
            exit_code=exit_code,
            duration_s=dt,
            out_path=str(out_path) if out_path else None,
            notes=notes,
        )


async def main_async(args) -> int:
    jobs = _load_goals(args)
    if not jobs:
        print("no goals/scenarios selected", file=sys.stderr)
        return 2

    server_url = (args.server or os.environ.get("MLX_SERVER_URL") or "").strip()
    if not server_url:
        server_url = backend_mod._mlx_server_endpoint_url()
    os.environ["MLX_SERVER_URL"] = server_url
    os.environ["LLM_BACKEND"] = "mlx-server"

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "batch_summary.json"

    child_env = os.environ.copy()
    child_env["MLX_SERVER_URL"] = server_url
    child_env["LLM_BACKEND"] = "mlx-server"
    child_env["PYTHONUNBUFFERED"] = "1"
    # Cursor/IDE sandboxes set PLAYWRIGHT_BROWSERS_PATH to a temp dir without
    # Chromium — subprocesses must use the project venv install (see setup.sh).
    child_env.pop("PLAYWRIGHT_BROWSERS_PATH", None)

    if args.dry_run:
        for label, goal in jobs:
            cmd, _ = _build_cmd(
                goal=goal, label=label, out_root=out_root, args=args,
            )
            print(" ".join(cmd))
        return 0

    try:
        info = backend_mod.detect_backend("mlx-server")
    except Exception as e:  # noqa: BLE001
        print(f"mlx server not reachable at {server_url!r}: {e}", file=sys.stderr)
        print(
            "Start the server first, e.g.:\n"
            "  .venv/bin/mlx_lm.server --model ~/MLX_Models/Qwen3.6-27B-mxfp8 --port 8080",
            file=sys.stderr,
        )
        return 2

    if not args.skip_preflight:
        try:
            await _preflight_server(server_url, info.model, timeout=args.preflight_timeout)
            print(f"preflight OK · model={info.model!r}")
        except Exception as e:  # noqa: BLE001
            print(
                f"mlx server preflight failed at {server_url!r}: {e}\n"
                "Restart mlx_lm.server before batching (wedged server caused "
                "tune_round1_r2 planning hang).",
                file=sys.stderr,
            )
            return 2

    print(
        f"batch_parallel · server={server_url} · model={info.model!r} · "
        f"jobs={len(jobs)} · concurrency={args.jobs} · "
        f"stall_seconds={args.stall_seconds} · job_timeout={args.job_timeout}s"
    )
    print(f"artifacts: {out_root}")

    sem = asyncio.Semaphore(max(1, args.jobs))
    t0 = time.monotonic()
    results: list[JobResult] = []

    async def _one(label: str, goal: str) -> JobResult:
        cmd, out_path = _build_cmd(
            goal=goal, label=label, out_root=out_root, args=args,
        )
        return await _run_job(
            sem, label=label, goal=goal, cmd=cmd, out_path=out_path, env=child_env,
            job_timeout=args.job_timeout,
        )

    tasks = [asyncio.create_task(_one(label, goal)) for label, goal in jobs]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        status = "PASS" if result.exit_code == 0 else "FAIL"
        print(
            f"[{status}] {result.label:<20} {result.duration_s:6.0f}s  "
            f"exit={result.exit_code}"
            + (f"  out={result.out_path}" if result.out_path else "")
        )
        _write_summary(
            summary_path=summary_path,
            server_url=server_url,
            model=info.model,
            jobs_total=len(jobs),
            results=results,
            wall_s=time.monotonic() - t0,
            status="running" if len(results) < len(jobs) else "done",
        )

    total_s = time.monotonic() - t0
    passed = sum(1 for r in results if r.exit_code == 0)
    _write_summary(
        summary_path=summary_path,
        server_url=server_url,
        model=info.model,
        jobs_total=len(jobs),
        results=results,
        wall_s=total_s,
        status="done",
    )

    print(f"\n{passed}/{len(results)} passed · {total_s:.0f}s wall · summary: {summary_path}")
    return 0 if passed == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default=None, help="MLX server base URL (sets MLX_SERVER_URL)")
    ap.add_argument(
        "--jobs", type=int, default=1,
        help="Max concurrent subprocesses (default 1 for art-heavy full builds)",
    )
    ap.add_argument("--goal", action="append", help="Build goal (repeatable)")
    ap.add_argument("--goals-file", help="Text file: one goal per line")
    ap.add_argument(
        "--seed-edits",
        action="store_true",
        help="Run eval_seed_edits scenarios instead of full coder builds",
    )
    ap.add_argument("--names", help="Comma-separated seed-edit scenario names")
    ap.add_argument("--patch-only", action="store_true")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--best-of-n", type=int, default=1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--stall-seconds",
        type=float,
        default=600.0,
        help="Quiet-window budget passed to coder.py (default 600; activity-based on server)",
    )
    ap.add_argument(
        "--job-timeout",
        type=float,
        default=7200.0,
        help="Kill each coder subprocess after this many seconds (default 7200 = 2h)",
    )
    ap.add_argument(
        "--preflight-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for a tiny mlx server completion before batch start",
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip mlx server smoke completion (not recommended)",
    )
    ap.add_argument("--model", default=None, help="Override model id sent to the server")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
