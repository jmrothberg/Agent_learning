#!/usr/bin/env python3
"""Autonomous A/B benchmark: same goal with playbook ON vs OFF.

Drives `coder.py` headlessly against the live MLX backend. Captures
each run's trace JSONL and scores by (passed, first_pass_iter,
stuck_streak). Emits a JSON summary and prints a small ASCII table
so the user can see whether the playbook helps or hurts on real
goals without having to run sessions by hand.

Designed for autonomous loops: pass --rounds N to repeat the matched
pair N times, --goals goal1,goal2 to test multiple prompts, and
--max-iters K (default 2) to keep total runtime bounded. Tracks
which bullets retrieved on the ON arm and credits/blames them based
on whether that arm beat the OFF baseline.

Usage:
    .venv/bin/python scripts/bench_playbook_ab.py \\
        --goals "snake game with arrow keys" \\
        --rounds 1 --max-iters 2

Output: games/bench/<timestamp>/summary.json plus stdout table.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO / "games" / "traces"
BENCH_DIR = REPO / "games" / "bench"


@dataclass
class Outcome:
    arm: str            # "on" or "off"
    goal: str
    round_idx: int
    passed: bool = False
    first_pass_iter: int | None = None
    iters_used: int = 0
    duration_s: float = 0.0
    bullets_retrieved: list[str] = field(default_factory=list)
    trace_path: str | None = None
    final_probes_passed: int = 0
    final_probes_total: int = 0
    error: str | None = None


def run_one(goal: str, *, playbook_on: bool, max_iters: int, bench_dir: Path,
            round_idx: int, timeout_s: float) -> Outcome:
    """Run coder.py once with the given config; return parsed Outcome."""
    arm = "on" if playbook_on else "off"
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:30]
    stamp = datetime.now().strftime("%H%M%S")
    out_html = bench_dir / f"{slug}_r{round_idx}_{arm}_{stamp}.html"
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "coder.py"),
        goal,
        "--max-iters", str(max_iters),
        "--best-of-n", "1",
        "--headless",
        "--out", str(out_html),
    ]
    if not playbook_on:
        cmd.append("--no-playbook")
    # Freeze writeback during benchmarking so the A and B arms don't
    # mutate the playbook differently while we're measuring.
    cmd.append("--no-playbook-writeback")

    started = time.monotonic()
    outcome = Outcome(arm=arm, goal=goal, round_idx=round_idx)
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), timeout=timeout_s,
            capture_output=True, text=True,
        )
        outcome.duration_s = time.monotonic() - started
        # coder.py exits 0 on ship, nonzero otherwise. But the real
        # signal is in the trace JSONL — read it directly.
        # Resolve trace path from the HTML stem.
        stem = out_html.stem
        trace = TRACES_DIR / f"{stem}.jsonl"
        outcome.trace_path = str(trace) if trace.exists() else None
        if outcome.trace_path:
            _populate_from_trace(outcome, trace)
        else:
            outcome.error = f"no trace at {trace}: stderr={proc.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        outcome.duration_s = time.monotonic() - started
        outcome.error = f"timeout after {timeout_s:.0f}s"
    except Exception as e:
        outcome.duration_s = time.monotonic() - started
        outcome.error = f"{type(e).__name__}: {e}"
    return outcome


def _populate_from_trace(outcome: Outcome, trace: Path) -> None:
    """Walk trace JSONL and pull out passed/iter/bullets/probes."""
    last_test = None
    bullet_ids: set[str] = set()
    iters_used = 0
    for raw in trace.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        kind = d.get("kind")
        if kind == "playbook_retrieved":
            for bid in d.get("ids", []):
                bullet_ids.add(bid)
        elif kind == "event" and d.get("event") == "phase":
            txt = d.get("text_preview", "")
            if txt.startswith("iteration "):
                # "iteration 3/6" → 3
                try:
                    iters_used = max(iters_used, int(txt.split()[1].split("/")[0]))
                except Exception:
                    pass
        elif kind == "event" and d.get("event") == "test":
            last_test = d
        elif kind == "session_outcome":
            # Some traces emit a structured outcome at the end.
            if d.get("ok"):
                outcome.passed = True
                outcome.first_pass_iter = d.get("first_pass_iter") or iters_used
    outcome.iters_used = iters_used
    outcome.bullets_retrieved = sorted(bullet_ids)
    # Final probe stats from the last test event, if present.
    if last_test and isinstance(last_test.get("data"), dict):
        rep = last_test["data"]
        probes = rep.get("probes") or []
        outcome.final_probes_total = len(probes)
        outcome.final_probes_passed = sum(1 for p in probes if p.get("ok"))
        # If no session_outcome event, infer pass from the last test ok.
        if not outcome.passed and rep.get("ok"):
            outcome.passed = True
            outcome.first_pass_iter = outcome.first_pass_iter or iters_used


def _format_table(outcomes: list[Outcome]) -> str:
    rows: list[str] = []
    rows.append(f"{'goal':36} {'arm':4} {'rnd':>3} {'pass':>5} "
                f"{'iter':>4} {'probes':>7} {'dur':>5}  bullets")
    rows.append("-" * 100)
    for o in outcomes:
        probes = f"{o.final_probes_passed}/{o.final_probes_total}" \
                 if o.final_probes_total else "—"
        passed = "Y" if o.passed else "N"
        nbul = len(o.bullets_retrieved)
        rows.append(
            f"{o.goal[:36]:36} {o.arm:4} {o.round_idx:>3} {passed:>5} "
            f"{o.iters_used:>4} {probes:>7} {o.duration_s:>4.0f}s  {nbul}"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", default="snake game with arrow keys",
                    help="comma-separated goals")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=420.0,
                    help="per-run cap. 27B MLX cold-load ~30s + planning + "
                         "2 iters at ~60-120s each. Default 7 min is comfortable.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    goals = [g.strip() for g in args.goals.split(",") if g.strip()]
    bench_dir = Path(args.out_dir) if args.out_dir else \
        BENCH_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] goals={goals} rounds={args.rounds} "
          f"max_iters={args.max_iters} dir={bench_dir}")

    outcomes: list[Outcome] = []
    for r in range(args.rounds):
        for g in goals:
            for on in (True, False):
                arm = "ON" if on else "OFF"
                print(f"[bench] round {r+1}/{args.rounds}  goal={g!r}  arm={arm}")
                o = run_one(
                    g, playbook_on=on, max_iters=args.max_iters,
                    bench_dir=bench_dir, round_idx=r + 1,
                    timeout_s=args.timeout_s,
                )
                outcomes.append(o)
                status = "PASS" if o.passed else ("ERR" if o.error else "FAIL")
                tail = f" err={o.error}" if o.error else ""
                print(f"        -> {status} iter={o.iters_used} "
                      f"probes={o.final_probes_passed}/{o.final_probes_total} "
                      f"dur={o.duration_s:.0f}s "
                      f"bullets={len(o.bullets_retrieved)}{tail}")
                # Flush a partial summary after every run so a ctrl-c
                # mid-loop doesn't lose data.
                (bench_dir / "summary.json").write_text(
                    json.dumps([asdict(x) for x in outcomes], indent=2)
                )

    print()
    print(_format_table(outcomes))
    print()
    summary_path = bench_dir / "summary.json"
    summary_path.write_text(json.dumps(
        [asdict(o) for o in outcomes], indent=2
    ))
    print(f"[bench] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
