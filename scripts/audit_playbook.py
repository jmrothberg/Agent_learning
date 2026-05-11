#!/usr/bin/env python3
"""Offline playbook audit: which bullets pull their weight, which don't.

Walks every games/traces/*.jsonl session, joins playbook_retrieved
events to the session's final outcome (passed / failed / unknown), and
emits a per-bullet table:

  fires    pass-rate    avg-iter   tags     id

Bullets are sorted by a simple "earnings" score:
    earnings = (passes * 1.0) - (fails * 0.5)
so a bullet that retrieves often but is on the losing arm gets
penalized; one that consistently rides winners climbs the table.
Bullets that never retrieve appear at the bottom with fires=0 — those
are the cheap-to-prune candidates.

This is the same signal `learner.py reflect` would compute via LLM
calls, but cheaper (no model invocation) and reproducible. The user
explicitly asked for a way to validate whether bullets help vs hurt;
this gives them numbers based on real session data.

Usage:
    .venv/bin/python scripts/audit_playbook.py
    .venv/bin/python scripts/audit_playbook.py --traces-dir games/traces --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class BulletStats:
    bid: str
    fires: int = 0
    on_pass: int = 0
    on_fail: int = 0
    on_unknown: int = 0
    iters_when_retrieved: list[int] = field(default_factory=list)
    goals_seen: list[str] = field(default_factory=list)

    def earnings(self) -> float:
        return self.on_pass * 1.0 - self.on_fail * 0.5

    def pass_rate(self) -> float | None:
        decided = self.on_pass + self.on_fail
        if decided == 0:
            return None
        return self.on_pass / decided

    def avg_iter(self) -> float | None:
        if not self.iters_when_retrieved:
            return None
        return sum(self.iters_when_retrieved) / len(self.iters_when_retrieved)


@dataclass
class SessionSummary:
    session_id: str
    goal: str
    passed: bool | None  # None = unknown (no terminal signal)
    iters_used: int
    retrieved_ids: list[str] = field(default_factory=list)


def _scan_session(trace: Path) -> SessionSummary | None:
    """Walk one trace JSONL and pull out goal, outcome, retrieved ids."""
    sid = trace.stem
    goal = ""
    passed: bool | None = None
    iters_used = 0
    retrieved_ids: set[str] = set()
    found_signal = False
    try:
        for raw in trace.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            found_signal = True
            kind = d.get("kind")
            if kind == "session_start":
                goal = d.get("goal", "")[:80]
            elif kind == "playbook_retrieved":
                for bid in d.get("ids", []) or []:
                    retrieved_ids.add(bid)
            elif kind == "event" and d.get("event") == "phase":
                txt = d.get("text_preview", "")
                if txt.startswith("iteration "):
                    try:
                        iters_used = max(
                            iters_used,
                            int(txt.split()[1].split("/")[0]),
                        )
                    except Exception:
                        pass
            elif kind == "session_outcome":
                # Explicit outcome event.
                if d.get("ok") is True:
                    passed = True
                elif d.get("ok") is False:
                    passed = False
            elif kind == "event" and d.get("event") == "test":
                rep = d.get("data") or {}
                if rep.get("ok"):
                    passed = True
                elif passed is None:
                    passed = False  # tentative; later test can flip back
    except Exception:
        return None
    if not found_signal:
        return None
    return SessionSummary(
        session_id=sid,
        goal=goal,
        passed=passed,
        iters_used=iters_used,
        retrieved_ids=sorted(retrieved_ids),
    )


def audit(traces_dir: Path) -> tuple[list[BulletStats], list[SessionSummary]]:
    stats: dict[str, BulletStats] = defaultdict(
        lambda: BulletStats(bid="?")
    )
    sessions: list[SessionSummary] = []
    # Walk recursively so games/bench/<ts>/traces/*.jsonl gets included
    # when traces_dir is games/traces. Top-level traces and bench
    # sub-traces both count as "sessions" for earnings attribution.
    candidates = list(sorted(traces_dir.glob("*.jsonl")))
    if traces_dir.name == "traces":
        bench_root = traces_dir.parent / "bench"
        if bench_root.exists():
            for sub in sorted(bench_root.glob("*/traces/*.jsonl")):
                candidates.append(sub)
    for trace in candidates:
        s = _scan_session(trace)
        if s is None:
            continue
        sessions.append(s)
        for bid in s.retrieved_ids:
            st = stats[bid]
            st.bid = bid
            st.fires += 1
            st.iters_when_retrieved.append(s.iters_used)
            if s.goal:
                st.goals_seen.append(s.goal[:40])
            if s.passed is True:
                st.on_pass += 1
            elif s.passed is False:
                st.on_fail += 1
            else:
                st.on_unknown += 1

    # Also list every bullet in the playbook, including never-retrieved.
    try:
        import memory
        pb = memory.Playbook(str(REPO / "games" / "memory"))
        for b in pb.load_all():
            if b.id not in stats:
                stats[b.id] = BulletStats(bid=b.id)
    except Exception:
        pass

    ordered = sorted(
        stats.values(),
        key=lambda s: (-s.earnings(), -s.fires, s.bid),
    )
    return ordered, sessions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces-dir",
        default=str(REPO / "games" / "traces"),
        help="directory of session trace JSONLs",
    )
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of the table")
    ap.add_argument("--include-zero-fire", action="store_true",
                    help="include bullets that never retrieved (default: "
                         "hide; these are pure dead weight worth pruning)")
    args = ap.parse_args()

    stats, sessions = audit(Path(args.traces_dir))
    if args.json:
        print(json.dumps({
            "sessions": [asdict(s) for s in sessions],
            "bullets": [asdict(s) for s in stats],
        }, indent=2))
        return 0

    decided = [s for s in sessions if s.passed is not None]
    passes = sum(1 for s in sessions if s.passed is True)
    fails = sum(1 for s in sessions if s.passed is False)
    print(f"Sessions: {len(sessions)} total  "
          f"({passes} pass, {fails} fail, "
          f"{len(sessions) - len(decided)} unknown)")
    print()
    print(f"{'bullet':35} {'fires':>5} {'pass':>4} {'fail':>4} "
          f"{'unk':>4} {'rate':>5} {'avg-it':>6} {'earn':>5}")
    print("-" * 80)
    for st in stats:
        if st.fires == 0 and not args.include_zero_fire:
            continue
        rate = st.pass_rate()
        rate_s = f"{rate*100:.0f}%" if rate is not None else "—"
        avg = st.avg_iter()
        avg_s = f"{avg:.1f}" if avg is not None else "—"
        print(f"{st.bid:35} {st.fires:>5} {st.on_pass:>4} "
              f"{st.on_fail:>4} {st.on_unknown:>4} {rate_s:>5} "
              f"{avg_s:>6} {st.earnings():>+5.1f}")
    # Always show the dead-weight count separately so the user has a
    # one-line signal even when zero-fires are hidden.
    dead = sum(1 for s in stats if s.fires == 0)
    if dead and not args.include_zero_fire:
        print()
        print(f"[{dead} bullets never retrieved in any scanned session — "
              "pure dead weight. Pass --include-zero-fire to list them.]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
