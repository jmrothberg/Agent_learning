#!/usr/bin/env python3
"""Credit/blame playbook bullets using matched-pair bench A/B outcomes.

The signal: for each goal in a bench run, we have an ON arm and an
OFF arm with the same model, seed, and prompt machinery — only the
playbook differs. If ON outscores OFF, the bullets that retrieved on
ON earned a +1; if OFF outscored ON, they earned a -1. This is a
clean attribution signal that the normal session-only writeback
(which has no counterfactual) can't produce.

Aggregates across all matched pairs in a bench dir, then applies the
deltas to playbook.jsonl via Playbook.update_counters. Dry-run by
default.

Score signal:
- Compare winner_score from the final restart_winner event (0-100).
  Higher = closer to passing. If absent, fall back to probes_passed
  ratio.
- A goal where ON > OFF by ≥ 5 points → bullets active on ON arm get
  helpful+1.
- A goal where OFF > ON by ≥ 5 points → those bullets get harmful+1.
- Within the noise band (|delta| < 5), no update — single-trial bench
  data is too noisy to act on small differences.

Usage:
    .venv/bin/python scripts/learn_from_bench.py \\
        games/bench/<timestamp>/
    .venv/bin/python scripts/learn_from_bench.py \\
        games/bench/<timestamp>/ --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Allow `import memory` when invoked from any cwd.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass
class PairOutcome:
    goal: str
    on_score: float
    off_score: float
    on_bullets: list[str]
    off_bullets: list[str]


def _score_from_trace(trace: Path) -> tuple[float, list[str]] | None:
    """Return (final_score_0_to_100, retrieved_bullet_ids) or None."""
    if not trace.exists():
        return None
    final_score: float | None = None
    bullets: set[str] = set()
    probes_pass = 0
    probes_total = 0
    for raw in trace.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        kind = d.get("kind")
        if kind == "playbook_retrieved":
            for bid in d.get("ids", []) or []:
                bullets.add(bid)
        elif kind == "restart_winner":
            # winner_score lives at the top level of the event.
            sc = d.get("winner_score")
            if isinstance(sc, (int, float)):
                final_score = float(sc)
        elif kind == "event" and d.get("event") == "test":
            rep = d.get("data") or {}
            probes = rep.get("probes") or []
            probes_total = len(probes)
            probes_pass = sum(1 for p in probes if p.get("ok"))
    if final_score is None and probes_total:
        # Fallback: scale probe pass-rate to 0-100 for comparable units.
        final_score = 100.0 * probes_pass / probes_total
    if final_score is None:
        return None
    return final_score, sorted(bullets)


def collect_pairs(bench_dir: Path) -> list[PairOutcome]:
    """Group trace files by goal slug and pair ON with OFF."""
    traces = list((bench_dir / "traces").glob("*.jsonl"))
    by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    # Naming convention from bench_playbook_ab.py:
    #   <goal-slug>_r<N>_<on|off>_<HHMMSS>.jsonl
    for t in traces:
        stem = t.stem
        # Split from the right: "..._r<N>_<arm>_<stamp>"
        parts = stem.rsplit("_", 3)
        if len(parts) < 4:
            continue
        goal_slug, round_marker, arm, _stamp = parts
        if arm not in ("on", "off"):
            continue
        by_key[(goal_slug, round_marker)].append(t)
    pairs: list[PairOutcome] = []
    for key, paths in by_key.items():
        on_t = next((p for p in paths if "_on_" in p.stem), None)
        off_t = next((p for p in paths if "_off_" in p.stem), None)
        if not on_t or not off_t:
            continue
        on = _score_from_trace(on_t)
        off = _score_from_trace(off_t)
        if not on or not off:
            continue
        # Goal text from the session_start event would be more
        # accurate, but the slug is sufficient for grouping.
        goal_slug = key[0]
        pairs.append(PairOutcome(
            goal=goal_slug,
            on_score=on[0], off_score=off[0],
            on_bullets=on[1], off_bullets=off[1],
        ))
    return pairs


def aggregate_deltas(
    pairs: list[PairOutcome], noise_band: float
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    helpful: dict[str, int] = defaultdict(int)
    harmful: dict[str, int] = defaultdict(int)
    notes: list[str] = []
    for p in pairs:
        delta = p.on_score - p.off_score
        if abs(delta) < noise_band:
            notes.append(
                f"  {p.goal:36} ON={p.on_score:.0f} OFF={p.off_score:.0f} "
                f"Δ={delta:+.0f} (in noise band — no update)"
            )
            continue
        if delta >= noise_band:
            target = helpful
            label = "helpful+1"
            ids = p.on_bullets
        else:
            target = harmful
            label = "harmful+1"
            ids = p.on_bullets  # blame the bullets that lost
        for bid in ids:
            target[bid] += 1
        notes.append(
            f"  {p.goal:36} ON={p.on_score:.0f} OFF={p.off_score:.0f} "
            f"Δ={delta:+.0f} → {label} on {len(ids)} bullets"
        )
    return dict(helpful), dict(harmful), notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir", type=Path,
                    help="games/bench/<timestamp>/")
    ap.add_argument("--apply", action="store_true",
                    help="actually update playbook counters")
    ap.add_argument("--noise-band", type=float, default=5.0,
                    help="ignore |on-off| score deltas below this "
                         "(default 5 points — single-trial bench is noisy)")
    args = ap.parse_args()

    if not args.bench_dir.exists():
        raise SystemExit(f"no such dir: {args.bench_dir}")

    pairs = collect_pairs(args.bench_dir)
    if not pairs:
        print("no matched ON/OFF pairs found in "
              f"{args.bench_dir}/traces/")
        return 1

    helpful, harmful, notes = aggregate_deltas(pairs, args.noise_band)
    print(f"Pairs analyzed: {len(pairs)}  (noise band: ±{args.noise_band})")
    print()
    for line in notes:
        print(line)
    print()
    if helpful:
        print(f"[helpful++] {len(helpful)} bullets")
        for bid, n in sorted(helpful.items(), key=lambda kv: -kv[1]):
            print(f"  {bid:35} +{n}")
    if harmful:
        print(f"[harmful++] {len(harmful)} bullets")
        for bid, n in sorted(harmful.items(), key=lambda kv: -kv[1]):
            print(f"  {bid:35} +{n}")
    if not helpful and not harmful:
        print("[no deltas] all pairs were within the noise band")
    if args.apply and (helpful or harmful):
        import memory
        pb = memory.Playbook(str(REPO / "games" / "memory"))
        # Aggregate to single update calls for atomicity.
        for bid, n in helpful.items():
            pb.update_counters([bid], helpful_delta=n)
        for bid, n in harmful.items():
            pb.update_counters([bid], harmful_delta=n)
        print()
        print(f"[apply] updated playbook.jsonl: "
              f"{len(helpful)} helpful, {len(harmful)} harmful")
    elif helpful or harmful:
        print()
        print("[dry-run] pass --apply to mutate playbook.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
