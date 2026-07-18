#!/usr/bin/env python3
"""Offline playbook helpful/harmful credit from finished batch traces.

Joins playbook retrieval ids to iter outcomes (same join as
``enrich_trace._retrieval_first_clean``) and applies
``memory.Playbook.update_counters``. Dedupes by trace stem in
``memory/playbook_credit_ledger.jsonl`` so a trace is never double-credited.

Usage:
  # Dry-run on a tune batch (no writes):
  .venv/bin/python scripts/credit_bullets.py games/tune_serial10/run_15 --dry-run

  # Apply credits:
  .venv/bin/python scripts/credit_bullets.py games/tune_serial10/run_15

  # Hygiene report only (never-retrieved / harmful>helpful):
  .venv/bin/python scripts/credit_bullets.py --hygiene games/tune_serial10/run_14 games/tune_serial10/run_15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from enrich_trace import _load_events, _retrieval_first_clean  # noqa: E402
from memory import Playbook  # noqa: E402

LEDGER_NAME = "playbook_credit_ledger.jsonl"
DEFAULT_MEMORY = REPO_ROOT / "memory"


def _ledger_path(memory_root: Path) -> Path:
    return memory_root / LEDGER_NAME


def _load_ledger(memory_root: Path) -> set[str]:
    path = _ledger_path(memory_root)
    credited: set[str] = set()
    if not path.exists():
        return credited
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stem = rec.get("trace_stem")
        if stem:
            credited.add(str(stem))
    return credited


def _append_ledger(
    memory_root: Path,
    *,
    trace_stem: str,
    helpful_ids: list[str],
    harmful_ids: list[str],
) -> None:
    path = _ledger_path(memory_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "trace_stem": trace_stem,
        "credited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "helpful_ids": helpful_ids,
        "harmful_ids": harmful_ids,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _iter_trace_paths(batch_dir: Path) -> list[Path]:
    traces_dir = batch_dir / "traces"
    if not traces_dir.is_dir():
        # Allow a single .jsonl file or a traces-only dir
        if batch_dir.is_file() and batch_dir.suffix == ".jsonl":
            return [batch_dir]
        if batch_dir.is_dir():
            return sorted(
                t for t in batch_dir.glob("*.jsonl") if ".enriched." not in t.name
            )
        return []
    return sorted(
        t for t in traces_dir.glob("*.jsonl") if ".enriched." not in t.name
    )


def _credit_ineligible(records: list[dict]) -> bool:
    """True when the session must not update helpful/harmful counters.

    Infra/backend crashes and sessions that never materialized code would
    otherwise poison playbook scores (seed-edit deepseek_v4 crash pattern).
    """
    for rec in records:
        if not isinstance(rec, dict) or rec.get("kind") != "session_outcome":
            continue
        if rec.get("backend_crashed"):
            return True
        if "code_materialized" in rec and not rec.get("code_materialized"):
            return True
    has_iter = any(
        isinstance(r, dict) and r.get("kind") == "iter_summary" for r in records
    )
    if has_iter:
        return False
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("kind") == "stream_done" and rec.get("crashed"):
            return True
        if rec.get("kind") in ("mlx_stall", "mlx_stall_recovery"):
            return True
    has_code = any(
        isinstance(r, dict) and r.get("kind") == "code_snapshot" for r in records
    )
    return not has_code


def credit_from_trace(records: list[dict]) -> tuple[list[str], list[str]]:
    """Return (helpful_ids, harmful_ids) for one session.

    Helpful: bullets active at the first clean iter (per-iter retrieved_ids).
    Harmful: active bullets when no clean iter — but NEVER for infra/backend
    crashes or sessions that never materialized code.
    """
    if _credit_ineligible(records):
        return [], []
    first_clean, retrieved = _retrieval_first_clean(records)
    if not retrieved:
        return [], []
    if first_clean is not None:
        return list(retrieved), []
    return [], list(retrieved)


def credit_batch(
    batch_dir: Path,
    *,
    memory_root: Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    already = _load_ledger(memory_root)
    helpful_counter: Counter[str] = Counter()
    harmful_counter: Counter[str] = Counter()
    credited_traces: list[str] = []
    skipped: list[str] = []

    for trace in _iter_trace_paths(batch_dir):
        stem = trace.stem
        if stem in already:
            skipped.append(stem)
            continue
        try:
            records = _load_events(trace)
        except Exception:
            skipped.append(stem)
            continue
        helpful_ids, harmful_ids = credit_from_trace(records)
        if not helpful_ids and not harmful_ids:
            skipped.append(stem)
            continue
        for bid in helpful_ids:
            helpful_counter[bid] += 1
        for bid in harmful_ids:
            harmful_counter[bid] += 1
        credited_traces.append(stem)
        if not dry_run:
            pb = Playbook(base_root=str(memory_root))
            if helpful_ids:
                pb.update_counters(helpful_ids, helpful_delta=1)
            if harmful_ids:
                pb.update_counters(harmful_ids, harmful_delta=1)
            _append_ledger(
                memory_root,
                trace_stem=stem,
                helpful_ids=helpful_ids,
                harmful_ids=harmful_ids,
            )

    return {
        "batch": str(batch_dir),
        "dry_run": dry_run,
        "credited_traces": credited_traces,
        "skipped_traces": skipped,
        "helpful_deltas": dict(helpful_counter.most_common()),
        "harmful_deltas": dict(harmful_counter.most_common()),
    }


def hygiene_report(batch_dirs: list[Path], *, memory_root: Path) -> str:
    """List never-retrieved bullets and harmful>helpful for manual curation."""
    retrieved: set[str] = set()
    for batch in batch_dirs:
        for trace in _iter_trace_paths(batch):
            try:
                records = _load_events(trace)
            except Exception:
                continue
            _, ids = _retrieval_first_clean(records)
            retrieved.update(ids)

    pb = Playbook(base_root=str(memory_root))
    bullets = pb.load_all()
    never = [b for b in bullets if b.id not in retrieved]
    net_neg = [b for b in bullets if b.harmful > b.helpful]

    lines = [
        "# Playbook hygiene (manual curation only — no deletes)",
        "",
        f"Batches scanned: {', '.join(d.name for d in batch_dirs)}",
        f"Unique retrieved ids: {len(retrieved)} / {len(bullets)} bullets",
        "",
        f"## Never retrieved in scanned batches ({len(never)})",
        "",
    ]
    for b in sorted(never, key=lambda x: x.id)[:80]:
        lines.append(f"- `{b.id}` (h={b.helpful}/H={b.harmful})")
    if len(never) > 80:
        lines.append(f"- … +{len(never) - 80} more")
    lines.extend(["", f"## harmful > helpful ({len(net_neg)})", ""])
    for b in sorted(net_neg, key=lambda x: x.harmful - x.helpful, reverse=True)[:40]:
        lines.append(f"- `{b.id}` (h={b.helpful}/H={b.harmful})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        help="Batch dir(s) under games/tune_serial10/ (or traces dirs)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deltas; do not update playbook or ledger",
    )
    ap.add_argument(
        "--hygiene",
        action="store_true",
        help="Print never-retrieved / net-harmful report only",
    )
    ap.add_argument(
        "--memory-root",
        default=str(DEFAULT_MEMORY),
        help="memory/ directory (default: repo memory/)",
    )
    args = ap.parse_args(argv)
    memory_root = Path(args.memory_root).resolve()
    if not args.paths:
        ap.error("at least one batch path required")
    paths = [Path(p).resolve() for p in args.paths]

    if args.hygiene:
        print(hygiene_report(paths, memory_root=memory_root))
        return 0

    # Default to dry-run unless --dry-run omitted AND user passes apply?
    # Plan: dry-run on run_15 shows deltas before applying. CLI: --dry-run
    # flag; without it we APPLY. Safer default: require explicit apply.
    # Plan says: credit_bullets.py --dry-run on run_15, then apply.
    # I'll use --dry-run as optional; without flag = apply. Document clearly.
    dry = bool(args.dry_run)
    for batch in paths:
        report = credit_batch(batch, memory_root=memory_root, dry_run=dry)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
