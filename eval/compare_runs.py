#!/usr/bin/env python3
"""Cross-run scoreboard for tune_serial batches (real data, not anecdotes).

Reads ``games/tune_serial10/run_XX/tune_summary.json`` plus each run's
``traces/*.jsonl``. Reuses ``scripts/enrich_trace.py`` helpers for
``wasted_iters`` / first-clean / retrieval joins — do not duplicate that
logic.

Usage:
  .venv/bin/python eval/compare_runs.py run_14 run_15
  .venv/bin/python eval/compare_runs.py games/tune_serial10/run_14 games/tune_serial10/run_15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from enrich_trace import (  # noqa: E402
    _compute_wasted_iters,
    _load_events,
    _retrieval_first_clean,
)

DEFAULT_BASE = REPO_ROOT / "games" / "tune_serial10"


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir() and (p / "tune_summary.json").exists():
        return p.resolve()
    cand = DEFAULT_BASE / arg
    if cand.is_dir() and (cand / "tune_summary.json").exists():
        return cand.resolve()
    # Allow bare run_XX even if summary is mid-batch
    if cand.is_dir():
        return cand.resolve()
    raise SystemExit(f"run dir not found: {arg}")


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "tune_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_metrics(run_dir: Path) -> dict[str, Any]:
    """Aggregate per-trace enrich_trace diagnostics for one run dir."""
    traces_dir = run_dir / "traces"
    traces = sorted(traces_dir.glob("*.jsonl")) if traces_dir.is_dir() else []
    # Prefer final attempts over .enriched siblings
    traces = [t for t in traces if ".enriched." not in t.name]

    wasted_total = 0
    first_cleans: list[int] = []
    never_clean = 0
    infra_failed = 0
    tok_rates: list[float] = []
    fail_hist: Counter[str] = Counter()
    n_traces = 0

    for trace in traces:
        try:
            records = _load_events(trace)
        except Exception:
            continue
        if not records:
            continue
        n_traces += 1
        n_wasted, _ = _compute_wasted_iters(records)
        wasted_total += n_wasted
        n_iter = sum(
            1 for r in records
            if isinstance(r, dict) and r.get("kind") == "iter_summary"
        )
        # Sessions with zero iter_summary are infra/backend deaths — not
        # memory_gap "never clean" (would inflate efficiency scoreboards).
        outcome = next(
            (r for r in records
             if isinstance(r, dict) and r.get("kind") == "session_outcome"),
            None,
        )
        is_infra = n_iter == 0 or bool(
            outcome and (
                outcome.get("backend_crashed")
                or (
                    "code_materialized" in outcome
                    and not outcome.get("code_materialized")
                )
            )
        )
        if is_infra and n_iter == 0:
            infra_failed += 1
        else:
            first_clean, _ = _retrieval_first_clean(records)
            if first_clean is None:
                never_clean += 1
            else:
                first_cleans.append(int(first_clean))
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") != "iter_summary":
                continue
            tps = rec.get("tok_per_s")
            if tps is not None:
                try:
                    v = float(tps)
                    if v > 0:
                        tok_rates.append(v)
                except (TypeError, ValueError):
                    pass
            fc = rec.get("failure_class")
            if fc and fc not in ("none", "ok", None):
                fail_hist[str(fc)] += 1
            elif rec.get("ok") is False and not fc:
                fail_hist["unknown"] += 1

    avg_wasted = (wasted_total / n_traces) if n_traces else 0.0
    avg_first = (sum(first_cleans) / len(first_cleans)) if first_cleans else None
    avg_tok = (sum(tok_rates) / len(tok_rates)) if tok_rates else None
    return {
        "n_traces": n_traces,
        "avg_wasted_iters": round(avg_wasted, 2),
        "avg_first_clean": round(avg_first, 2) if avg_first is not None else None,
        "never_clean": never_clean,
        "infra_failed": infra_failed,
        "avg_tok_per_s": round(avg_tok, 2) if avg_tok is not None else None,
        "failure_class": dict(fail_hist.most_common()),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = _load_summary(run_dir)
    metrics = _trace_metrics(run_dir)
    results = summary.get("results") or []
    outcomes = Counter(str(r.get("outcome") or "unknown") for r in results)
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "status": summary.get("status"),
        "jobs": summary.get("jobs"),
        "passed": summary.get("passed"),
        "fresh_passed": summary.get("fresh_passed"),
        "artifact_passed": summary.get("artifact_passed"),
        "fresh_failed": summary.get("fresh_failed"),
        "skipped": summary.get("skipped"),
        "model": summary.get("model"),
        "outcomes": dict(outcomes),
        **metrics,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return str(v)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown scoreboard table + failure_class histograms."""
    lines: list[str] = []
    lines.append("# Cross-run scoreboard")
    lines.append("")
    lines.append(
        "| run | status | jobs | fresh_pass | artifact_pass | fail | "
        "avg wasted_iters | avg first_clean | never_clean | infra_failed | avg tok/s |"
    )
    lines.append(
        "|-----|--------|------|------------|---------------|------|"
        "-----------------|-----------------|-------------|--------------|-----------|"
    )
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt(r.get('status'))} | {_fmt(r.get('jobs'))} | "
            f"{_fmt(r.get('fresh_passed'))} | {_fmt(r.get('artifact_passed'))} | "
            f"{_fmt(r.get('fresh_failed'))} | {_fmt(r.get('avg_wasted_iters'))} | "
            f"{_fmt(r.get('avg_first_clean'))} | {_fmt(r.get('never_clean'))} | "
            f"{_fmt(r.get('infra_failed'))} | {_fmt(r.get('avg_tok_per_s'))} |"
        )
    lines.append("")
    lines.append("## failure_class histogram (iter_summary, non-ok classes)")
    lines.append("")
    for r in rows:
        hist = r.get("failure_class") or {}
        infra = r.get("infra_failed") or 0
        if not hist and not infra:
            lines.append(f"- **{r['name']}**: (none recorded)")
            continue
        parts = ", ".join(f"{k}={v}" for k, v in hist.items()) if hist else "(no iter_summary classes)"
        extra = f"; infra_failed={infra}" if infra else ""
        lines.append(f"- **{r['name']}**: {parts}{extra}")
    lines.append("")
    lines.append(
        "_Source: `tune_summary.json` + `traces/*.jsonl` via "
        "`scripts/enrich_trace` wasted_iters / first_clean helpers._"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "runs",
        nargs="+",
        help="Run dir names (run_14) or paths under games/tune_serial10/",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown",
    )
    args = ap.parse_args(argv)
    rows = [summarize_run(_resolve_run_dir(a)) for a in args.runs]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
