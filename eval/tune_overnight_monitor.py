#!/usr/bin/env python3
"""Poll tune_serial checkpoint + traces; emit status for overnight agent triage.

Usage:
  .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_05
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = "tune_checkpoint.json"
STATUS_NAME = "agent_monitor.json"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _trace_signals(trace_path: Path) -> dict:
    out: dict = {
        "lines": 0,
        "iter_summaries": 0,
        "last_iter_ok": None,
        "runaway_warnings": 0,
        "stream_dones": 0,
        "failure_classes": {},
        "plan_summary": None,
        "max_tokens_hit": False,
    }
    if not trace_path.is_file():
        return out
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        out["lines"] += 1
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("kind")
        if kind == "iter_summary":
            out["iter_summaries"] += 1
            out["last_iter_ok"] = bool(ev.get("ok"))
        elif kind == "runaway_stream_warning":
            out["runaway_warnings"] += 1
        elif kind == "stream_done":
            out["stream_dones"] += 1
            if ev.get("max_tokens_hit"):
                out["max_tokens_hit"] = True
        elif kind == "plan_summary":
            out["plan_summary"] = {
                "prose_chars": ev.get("prose_chars"),
                "looped": ev.get("looped"),
            }
        fc = ev.get("failure_class")
        if fc:
            out["failure_classes"][fc] = out["failure_classes"].get(fc, 0) + 1
    return out


def _newest_trace(out_dir: Path, label: str) -> Path | None:
    stem = label
    candidates = sorted(
        out_dir.glob(f"traces/{stem}__run_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _tail_log(log_path: Path, n: int = 5) -> list[str]:
    if not log_path.is_file():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _parse_pass_fail(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hits = []
    for m in re.finditer(r"\[(PASS|FAIL)\]\s+(\S+)", text):
        hits.append({"status": m.group(1), "label": m.group(2)})
    return hits


def snapshot(out_dir: Path, jobs_total: int) -> dict:
    ck_path = out_dir / CHECKPOINT
    ck = _load_json(ck_path)
    completed = list(ck.get("completed_labels") or [])
    log_path = out_dir / "overnight.log"
    results_raw = ck.get("results") or []
    game_rows = []
    for raw in results_raw:
        label = raw.get("label", "")
        trace = _newest_trace(out_dir, label) if label else None
        sig = _trace_signals(trace) if trace else {}
        game_rows.append({
            "label": label,
            "exit_code": raw.get("exit_code"),
            "failure_classes": raw.get("failure_classes") or {},
            "trace": str(trace.relative_to(REPO)) if trace else None,
            "signals": sig,
        })
    runaway_games = sum(
        1 for g in game_rows if g.get("signals", {}).get("runaway_warnings", 0) > 0
    )
    no_iter_games = sum(
        1 for g in game_rows if g.get("signals", {}).get("iter_summaries", 0) == 0
    )
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_count": len(completed),
        "jobs_total": jobs_total,
        "completed_labels": completed,
        "log_tail": _tail_log(log_path),
        "pass_fail": _parse_pass_fail(log_path),
        "games": game_rows,
        "patterns": {
            "runaway_games": runaway_games,
            "no_iter_summary_games": no_iter_games,
            "mid_batch_prefill_gate": runaway_games >= 2 or (
                runaway_games >= 1 and no_iter_games >= 1
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="games/tune_serial10/run_05")
    ap.add_argument("--jobs-total", type=int, default=12)
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / STATUS_NAME

    while True:
        payload = snapshot(out_dir, args.jobs_total)
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"monitor: {payload['completed_count']}/{payload['jobs_total']} "
            f"runaway_games={payload['patterns']['runaway_games']} "
            f"gate={payload['patterns']['mid_batch_prefill_gate']}",
            flush=True,
        )
        if args.once or payload["completed_count"] >= args.jobs_total:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
