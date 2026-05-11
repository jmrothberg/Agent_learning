#!/usr/bin/env python3
"""Data-driven playbook prune.

Reads scripts/audit_playbook.py output across recent traces. For each
bullet, decides one of three actions:

  KEEP   — has positive earnings OR has zero fires but is too new to
           judge (fewer than --min-sessions sessions scanned).
  HARMFUL — fires often AND has pass-rate below --harmful-rate; the
           bullet is bumped harmful++ via Playbook.update_counters so
           the existing quality multiplier ranks it lower at retrieval.
  PRUNE   — fires often with terrible earnings, OR has zero fires after
           --min-sessions sessions in playbook.jsonl. The bullet is
           removed from games/memory/playbook.jsonl. (Seed list in
           memory.py is left alone; a future `git pull` rehydrates.)

The script REFUSES to prune if --min-sessions hasn't been hit, so a
single trace can't accidentally demolish the playbook. Default
thresholds are deliberately conservative: --min-sessions 8,
--harmful-rate 0.20, --prune-rate 0.05, --prune-fires 4.

Dry-run by default. Pass --apply to actually mutate playbook.jsonl.

Usage:
    .venv/bin/python scripts/prune_playbook.py
    .venv/bin/python scripts/prune_playbook.py --apply
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass
class Decision:
    bid: str
    action: str       # "keep" | "harmful" | "prune"
    reason: str
    fires: int = 0
    pass_rate: float | None = None
    earnings: float = 0.0


def _load_audit() -> dict:
    """Shell out to audit_playbook.py so we use the canonical scanner."""
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "audit_playbook.py"),
        "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if out.returncode != 0:
        raise SystemExit(
            f"audit failed: {out.stderr[-400:]}\n(cmd: {' '.join(cmd)})"
        )
    return json.loads(out.stdout)


def decide(
    bullets: list[dict],
    *,
    n_sessions: int,
    min_sessions: int,
    harmful_rate: float,
    prune_rate: float,
    prune_fires: int,
) -> list[Decision]:
    decisions: list[Decision] = []
    can_prune = n_sessions >= min_sessions
    for b in bullets:
        bid = b["bid"]
        fires = b.get("fires", 0)
        on_pass = b.get("on_pass", 0)
        on_fail = b.get("on_fail", 0)
        decided = on_pass + on_fail
        rate = (on_pass / decided) if decided else None
        earnings = float(b.get("earnings", on_pass - 0.5 * on_fail))
        # Zero-fire bullets: dead weight after a healthy session count.
        if fires == 0:
            if can_prune:
                decisions.append(Decision(
                    bid=bid, action="prune",
                    reason=f"never retrieved in {n_sessions} sessions",
                    fires=0, earnings=earnings,
                ))
            else:
                decisions.append(Decision(
                    bid=bid, action="keep",
                    reason=(f"never retrieved yet, but only "
                            f"{n_sessions} sessions scanned (< "
                            f"min-sessions={min_sessions})"),
                    fires=0, earnings=earnings,
                ))
            continue
        # Heavy losers: prune outright.
        if can_prune and fires >= prune_fires and rate is not None \
                and rate <= prune_rate:
            decisions.append(Decision(
                bid=bid, action="prune",
                reason=(f"fires={fires} rate={rate*100:.0f}% "
                        f"earnings={earnings:.1f}"),
                fires=fires, pass_rate=rate, earnings=earnings,
            ))
            continue
        # Soft penalty: bump harmful so retrieval quality multiplier
        # downranks it without losing the bullet entirely.
        if rate is not None and rate <= harmful_rate and fires >= 2:
            decisions.append(Decision(
                bid=bid, action="harmful",
                reason=(f"fires={fires} rate={rate*100:.0f}% — "
                        f"penalize, don't prune yet"),
                fires=fires, pass_rate=rate, earnings=earnings,
            ))
            continue
        decisions.append(Decision(
            bid=bid, action="keep",
            reason=(f"fires={fires} rate="
                    f"{(rate*100 if rate is not None else 0):.0f}% "
                    f"earnings={earnings:+.1f}"),
            fires=fires, pass_rate=rate, earnings=earnings,
        ))
    return decisions


def apply(decisions: list[Decision]) -> None:
    """Mutate playbook.jsonl: remove pruned, bump harmful counters."""
    import memory
    pb = memory.Playbook(str(REPO / "games" / "memory"))
    bullets = pb.load_all()
    prune_ids = {d.bid for d in decisions if d.action == "prune"}
    harmful_ids = [d.bid for d in decisions if d.action == "harmful"]
    if prune_ids:
        kept = [b for b in bullets if b.id not in prune_ids]
        pb._save_all(kept)
        print(f"[prune] removed {len(prune_ids)} bullets from "
              f"playbook.jsonl: {sorted(prune_ids)}")
    if harmful_ids:
        pb.update_counters(harmful_ids, harmful_delta=1)
        print(f"[prune] bumped harmful+1 on {len(harmful_ids)}: "
              f"{sorted(harmful_ids)}")
    if not prune_ids and not harmful_ids:
        print("[prune] nothing to do")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sessions", type=int, default=8,
                    help="don't prune until at least this many sessions "
                         "have been scanned (default 8)")
    ap.add_argument("--harmful-rate", type=float, default=0.20,
                    help="bump harmful++ on bullets with pass-rate at or "
                         "below this (default 0.20)")
    ap.add_argument("--prune-rate", type=float, default=0.05,
                    help="prune bullets with pass-rate at or below this "
                         "(default 0.05)")
    ap.add_argument("--prune-fires", type=int, default=4,
                    help="require at least N fires before pruning a low "
                         "performer (default 4)")
    ap.add_argument("--apply", action="store_true",
                    help="actually mutate playbook.jsonl. Default is "
                         "dry-run, so you can sanity check.")
    args = ap.parse_args()

    audit = _load_audit()
    n_sessions = len(audit.get("sessions", []))
    bullets = audit.get("bullets", [])
    decisions = decide(
        bullets,
        n_sessions=n_sessions,
        min_sessions=args.min_sessions,
        harmful_rate=args.harmful_rate,
        prune_rate=args.prune_rate,
        prune_fires=args.prune_fires,
    )
    by_action: dict[str, list[Decision]] = {"prune": [], "harmful": [], "keep": []}
    for d in decisions:
        by_action[d.action].append(d)

    print(f"Sessions scanned: {n_sessions} (need ≥ "
          f"{args.min_sessions} to prune)")
    print()
    for action in ("prune", "harmful", "keep"):
        rows = by_action[action]
        if not rows:
            continue
        print(f"[{action.upper()}] {len(rows)} bullets")
        for d in rows:
            print(f"  {d.bid:35} {d.reason}")
        print()

    if args.apply:
        if n_sessions < args.min_sessions and any(
            d.action == "prune" for d in decisions
        ):
            print(f"REFUSED: --apply requires at least "
                  f"{args.min_sessions} sessions; only {n_sessions} "
                  "scanned. Run more sessions first.")
            return 2
        apply(decisions)
    else:
        print("[dry-run] pass --apply to actually mutate playbook.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
