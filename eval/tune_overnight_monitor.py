#!/usr/bin/env python3
"""Poll tune_serial checkpoint + traces; emit status for overnight agent triage.

Usage (overnight run_07 — batch in Terminal, watcher in Cursor):

  # Terminal — runs all night, blocks between games until agent releases:
  caffeinate -dims .venv/bin/python eval/tune_serial_loop.py ... --wait-for-monitor 1800

  # Cursor — poll + auto-release if agent stuck (optional safety net):
  .venv/bin/python eval/tune_overnight_monitor.py \\
      --out-dir games/tune_serial10/run_07_big --jobs-total 6 --interval 60 \\
      --sync-loop --auto-release 3600

When inter_game_pending.json appears, triage the trace, patch code/memory/prompts,
then release:
  .venv/bin/python eval/tune_inter_game_ready.py --out-dir games/tune_serial10/run_07_big \\
      --note "what you fixed"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.inter_game_sync import (  # noqa: E402
    pending_awaiting_fix,
    pending_age_seconds,
    sync_status,
    write_ready,
)
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
        "session_ok": None,
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
        elif kind == "session_outcome":
            out["session_ok"] = bool(ev.get("ok"))
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
    tag = r"(?:PASS|FAIL|fresh_pass|artifact_pass|fresh_fail|skipped)"
    for m in re.finditer(rf"\[({tag})\]\s+(\S+)", text):
        hits.append({"status": m.group(1), "label": m.group(2)})
    return hits


def _effective_outcome(raw: dict, sig: dict) -> str:
    """Prefer checkpoint outcome; infer from trace when missing (old checkpoints)."""
    outcome = raw.get("outcome")
    if outcome in ("fresh_pass", "artifact_pass", "fresh_fail", "skipped"):
        return outcome
    if sig.get("last_iter_ok") is True:
        return "fresh_pass"
    if sig.get("iter_summaries", 0) == 0 and sig.get("session_ok") is False:
        return "fresh_fail"
    if raw.get("exit_code") == 0 and sig.get("iter_summaries", 0) > 0:
        return "fresh_pass"
    return "fresh_fail"


def snapshot(out_dir: Path, jobs_total: int) -> dict:
    ck_path = out_dir / CHECKPOINT
    ck = _load_json(ck_path)
    completed = list(ck.get("completed_labels") or [])
    log_path = out_dir / "overnight.log"
    results_raw = ck.get("results") or []
    game_rows = []
    pass_fail = []
    for raw in results_raw:
        label = raw.get("label", "")
        trace = _newest_trace(out_dir, label) if label else None
        sig = _trace_signals(trace) if trace else {}
        outcome = _effective_outcome(raw, sig)
        game_rows.append({
            "label": label,
            "exit_code": raw.get("exit_code"),
            "outcome": outcome,
            "failure_classes": raw.get("failure_classes") or {},
            "trace": str(trace.relative_to(REPO)) if trace else None,
            "signals": sig,
        })
        pass_fail.append({"status": outcome, "label": label})
    runaway_games = sum(
        1 for g in game_rows if g.get("signals", {}).get("runaway_warnings", 0) > 0
    )
    no_iter_games = sum(
        1 for g in game_rows if g.get("signals", {}).get("iter_summaries", 0) == 0
    )
    fresh_pass = sum(1 for g in game_rows if g.get("outcome") == "fresh_pass")
    fresh_fail = sum(1 for g in game_rows if g.get("outcome") == "fresh_fail")
    inter_game = sync_status(out_dir)
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_count": len(completed),
        "jobs_total": jobs_total,
        "completed_labels": completed,
        "fresh_pass_count": fresh_pass,
        "fresh_fail_count": fresh_fail,
        "inter_game": inter_game,
        "watcher_action": (
            "triage_and_fix_then_release"
            if inter_game.get("awaiting_agent_fix")
            else None
        ),
        "log_tail": _tail_log(log_path),
        "pass_fail": pass_fail,
        "games": game_rows,
        "patterns": {
            "runaway_games": runaway_games,
            "no_iter_summary_games": no_iter_games,
            "mid_batch_prefill_gate": runaway_games >= 2 or (
                runaway_games >= 1 and no_iter_games >= 1
            ),
        },
    }


def _active_run07_batch() -> tuple[str, Path, int]:
    """Return (name, out_dir, jobs_total) for the batch still in progress."""
    big_dir = (REPO / "games/tune_serial10/run_07_big").resolve()
    vlm_dir = (REPO / "games/tune_serial10/run_07_vlm").resolve()
    big_ck = _load_json(big_dir / CHECKPOINT)
    vlm_ck = _load_json(vlm_dir / CHECKPOINT)
    big_n = len(big_ck.get("completed_labels") or [])
    vlm_n = len(vlm_ck.get("completed_labels") or [])
    if big_n < 6:
        return "big", big_dir, 6
    if vlm_n < 5:
        return "vlm", vlm_dir, 5
    return "done", vlm_dir, 5


def _run_run07_chain_monitor(args) -> int:
    """Poll both run_07 batches; auto-release on the active batch's pending handoff."""
    chain_dir = (REPO / "games/tune_serial10/run_07").resolve()
    chain_dir.mkdir(parents=True, exist_ok=True)
    status_path = chain_dir / STATUS_NAME

    while True:
        batch_name, active_dir, jobs_total = _active_run07_batch()
        big_snap = snapshot((REPO / "games/tune_serial10/run_07_big").resolve(), 6)
        vlm_snap = snapshot((REPO / "games/tune_serial10/run_07_vlm").resolve(), 5)

        # auto_release=0 on --run07-chain: instant pass-through (no sleep wait).
        if batch_name != "done":
            ig = (big_snap if batch_name == "big" else vlm_snap).get("inter_game") or {}
            age = pending_age_seconds(active_dir)
            instant = args.auto_release == 0
            timed = args.auto_release > 0 and age >= args.auto_release
            if ig.get("awaiting_agent_fix") and (instant or timed):
                write_ready(
                    active_dir,
                    note=(
                        "instant pass-through (auto-release=0)"
                        if instant
                        else f"auto-release after {args.auto_release:.0f}s wait"
                    ),
                    released_by="monitor_instant" if instant else "monitor_timeout",
                )
                if batch_name == "big":
                    big_snap = snapshot((REPO / "games/tune_serial10/run_07_big").resolve(), 6)
                else:
                    vlm_snap = snapshot((REPO / "games/tune_serial10/run_07_vlm").resolve(), 5)

        active_snap = big_snap if batch_name == "big" else vlm_snap
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run07_chain": True,
            "active_batch": batch_name,
            "active_out_dir": str(active_dir.relative_to(REPO)),
            "batch_a": big_snap,
            "batch_b": vlm_snap,
            "awaiting_agent_fix": (active_snap.get("inter_game") or {}).get("awaiting_agent_fix"),
            "watcher_action": active_snap.get("watcher_action"),
            "inter_game": active_snap.get("inter_game"),
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Mirror per-batch monitor files for tools that read them directly.
        (REPO / "games/tune_serial10/run_07_big" / STATUS_NAME).write_text(
            json.dumps(big_snap, indent=2), encoding="utf-8",
        )
        (REPO / "games/tune_serial10/run_07_vlm" / STATUS_NAME).write_text(
            json.dumps(vlm_snap, indent=2), encoding="utf-8",
        )

        ig = payload.get("inter_game") or {}
        print(
            f"run07-chain: active={batch_name} "
            f"A={big_snap['completed_count']}/6 B={vlm_snap['completed_count']}/5 "
            f"awaiting_fix={payload.get('awaiting_agent_fix')} "
            f"pending_age={ig.get('pending_age_s', 0)}s",
            flush=True,
        )
        if payload.get("awaiting_agent_fix"):
            pending = ig.get("pending") or {}
            rel = active_dir.relative_to(REPO)
            print(
                f"  → WATCHER: triage {pending.get('trace') or pending.get('label')} "
                f"then: .venv/bin/python eval/tune_inter_game_ready.py "
                f"--out-dir {rel} --note '…'",
                flush=True,
            )
        if args.once or batch_name == "done":
            break
        time.sleep(args.interval)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="games/tune_serial10/run_05")
    ap.add_argument("--jobs-total", type=int, default=12)
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--sync-loop",
        action="store_true",
        help="Include inter_game handoff state; use with --wait-for-monitor batch",
    )
    ap.add_argument(
        "--auto-release",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "If pending awaits fix longer than SECONDS, write inter_game_ready "
            "(safety net if Cursor agent stuck). 0=disabled."
        ),
    )
    ap.add_argument(
        "--run07-chain",
        action="store_true",
        help=(
            "Watch run_07_big (6) then run_07_vlm (5); write "
            "games/tune_serial10/run_07/agent_monitor.json"
        ),
    )
    args = ap.parse_args()

    if args.run07_chain:
        return _run_run07_chain_monitor(args)
    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / STATUS_NAME

    while True:
        payload = snapshot(out_dir, args.jobs_total)
        if args.sync_loop or args.auto_release > 0:
            ig = payload.get("inter_game") or {}
            if (
                args.auto_release > 0
                and ig.get("awaiting_agent_fix")
                and pending_age_seconds(out_dir) >= args.auto_release
            ):
                write_ready(
                    out_dir,
                    note=f"auto-release after {args.auto_release:.0f}s wait",
                    released_by="monitor_timeout",
                )
                payload = snapshot(out_dir, args.jobs_total)
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ig = payload.get("inter_game") or {}
        print(
            f"monitor: {payload['completed_count']}/{payload['jobs_total']} "
            f"runaway_games={payload['patterns']['runaway_games']} "
            f"gate={payload['patterns']['mid_batch_prefill_gate']} "
            f"awaiting_fix={ig.get('awaiting_agent_fix')} "
            f"pending_age={ig.get('pending_age_s')}s",
            flush=True,
        )
        if ig.get("awaiting_agent_fix"):
            pending = ig.get("pending") or {}
            print(
                f"  → WATCHER: triage {pending.get('trace') or pending.get('label')} "
                f"then: .venv/bin/python eval/tune_inter_game_ready.py "
                f"--out-dir {out_dir.relative_to(REPO)} --note '…'",
                flush=True,
            )
        if args.once or payload["completed_count"] >= args.jobs_total:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
