#!/usr/bin/env python3
"""Back-fill diagnostics into an existing trace .jsonl.

The trace events the agent wrote BEFORE the trace-detail upgrade (commit
4372532, 2026-05-16) are sparse: `image_attached` carried only count +
bytes, and `vision_judge` carried only progress + note + model. When a
session showed `progress: null, note: ""` on every iter you could not
tell whether (a) the judge call got no image, (b) the model returned
empty text, or (c) the model emitted unparseable prose -- a different
fix per layer.

Most of that information is still recoverable from disk:

- Each `image_attached` event happens within seconds of an iter writing
  `iter_NN.png` into the session's snapshots dir. We can match by
  timestamp and back-fill dims + source path.
- Each `vision_judge` event sat against a `_prev_judge_png` rotation;
  on single-screenshot sessions it ships exactly ONE image to mlx_vlm
  (image_count = 1) unless the screenshot capture failed.
- The screenshots themselves are still on disk, so we can re-read PNG
  IHDR and add the actual width/height.

We do NOT mutate the original trace file -- output goes to a sibling
`.enriched.jsonl` so the raw provenance is preserved.

Also prints a one-screen summary of what happened in the session:
VLM detected? Screenshots attached? Judge fired? Parse fails?

Usage:
    .venv/bin/python scripts/enrich_trace.py <session_id-or-trace-path>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _png_dims(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as fh:
            sig = fh.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            fh.read(8)  # IHDR length + type
            w, h = struct.unpack(">II", fh.read(8))
            return (int(w), int(h))
    except Exception:
        return None


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _resolve_paths(arg: str) -> tuple[Path, Path]:
    """Accept either a path to a .jsonl trace or a session-id substring.
    Returns (trace_path, snapshots_dir)."""
    p = Path(arg)
    if p.is_file() and p.suffix == ".jsonl":
        trace = p
    else:
        traces_dir = REPO_ROOT / "games" / "traces"
        matches = sorted(traces_dir.glob(f"*{arg}*.jsonl"))
        if not matches:
            raise SystemExit(f"no trace matched {arg!r} in {traces_dir}")
        if len(matches) > 1:
            print("multiple matches; using most recent:")
            for m in matches:
                print(f"  - {m.name}")
            print(f"using: {matches[-1].name}")
        trace = matches[-1]
    session_id = trace.stem
    snapshots_dir = REPO_ROOT / "games" / "snapshots" / session_id
    return trace, snapshots_dir


def _load_events(trace: Path) -> list[dict]:
    events: list[dict] = []
    with trace.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _index_snapshots(snapshots_dir: Path) -> list[tuple[int, Path, tuple[int, int] | None]]:
    """List (iter_n, path, dims) for every iter_NN.png on disk."""
    if not snapshots_dir.is_dir():
        return []
    out: list[tuple[int, Path, tuple[int, int] | None]] = []
    for png in sorted(snapshots_dir.glob("iter_*.png")):
        stem = png.stem  # "iter_01"
        try:
            n = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((n, png, _png_dims(png)))
    return out


def enrich(trace: Path, snapshots_dir: Path) -> tuple[list[dict], dict]:
    events = _load_events(trace)
    snapshots = _index_snapshots(snapshots_dir)

    # Index snapshots in iter order so we can hand them out as
    # image_attached events appear. Single-screenshot sessions attach
    # one PNG per fix turn, so the Nth image_attached event corresponds
    # to iter_NN.png. (Double-screenshot mode would attach two; we
    # detect that by looking at the original count field.)
    snap_iter = iter(snapshots)
    consumed: list[tuple[int, Path, tuple[int, int] | None]] = []
    enriched: list[dict] = []

    summary = {
        "session_id": trace.stem,
        "events_total": len(events),
        "snapshots_on_disk": len(snapshots),
        "vlm_detected": False,
        "vlm_path": None,
        "image_attached_count": 0,
        "image_attached_total_bytes": 0,
        "vision_judge_calls": 0,
        "vision_judge_parse_ok": 0,
        "vision_judge_parse_failed": 0,
        "first_user_feedback": None,
    }

    for ev in events:
        kind = ev.get("kind")
        out = dict(ev)

        if kind == "vlm_detected":
            summary["vlm_detected"] = True
            summary["vlm_path"] = ev.get("model")

        elif kind == "image_attached":
            summary["image_attached_count"] += 1
            summary["image_attached_total_bytes"] += int(ev.get("bytes") or 0)
            if "dims" not in out or "sources" not in out:
                # Best-effort match: pop the next on-disk snapshot.
                count = int(ev.get("count") or 1)
                taken: list[tuple[int, Path, tuple[int, int] | None]] = []
                try:
                    for _ in range(count):
                        taken.append(next(snap_iter))
                except StopIteration:
                    pass
                consumed.extend(taken)
                if taken:
                    out.setdefault("dims", [t[2] for t in taken])
                    out.setdefault("sources", [str(t[1]) for t in taken])
                    out.setdefault("iteration_recovered", [t[0] for t in taken])
                    out["enriched"] = True
                else:
                    out["enriched"] = False
                    out["enrich_note"] = "no matching iter_NN.png on disk"

        elif kind == "vision_judge":
            summary["vision_judge_calls"] += 1
            progress = ev.get("progress")
            note = ev.get("note") or ""
            parse_failed = (progress is None) and not note
            out.setdefault("parse_failed", parse_failed)
            if parse_failed:
                summary["vision_judge_parse_failed"] += 1
            else:
                summary["vision_judge_parse_ok"] += 1
            # image_count wasn't logged in old traces; if the session
            # ran with single-screenshot (the default), the judge call
            # shipped exactly one PNG when current_png was non-empty.
            # We can't recover the actual count after the fact, but we
            # CAN say "screenshot for iter N existed on disk", which
            # is the most useful proxy.
            iter_n = ev.get("iteration")
            if isinstance(iter_n, int):
                hit = next(
                    (s for s in snapshots if s[0] == iter_n), None
                )
                if hit is not None:
                    out.setdefault("screenshot_for_iter_existed", True)
                    out.setdefault("screenshot_dims_recovered", hit[2])
                    out.setdefault("screenshot_path_recovered", str(hit[1]))
                else:
                    out.setdefault("screenshot_for_iter_existed", False)
            if "raw" not in out:
                out.setdefault("raw", None)
                out.setdefault("raw_note", (
                    "raw model reply was not captured by the agent at "
                    "this session's commit (pre-4372532). Re-run for "
                    "post-mortem-grade trace."
                ))

        elif kind == "user_feedback" and summary["first_user_feedback"] is None:
            summary["first_user_feedback"] = (ev.get("text") or "")[:200]

        enriched.append(out)

    return enriched, summary


def _print_summary(summary: dict) -> None:
    print(f"\n=== session {summary['session_id']} ===")
    print(f"events total            : {summary['events_total']}")
    print(f"snapshots on disk       : {summary['snapshots_on_disk']}")
    print(f"VLM detected            : {summary['vlm_detected']}")
    if summary["vlm_path"]:
        print(f"VLM path                : {summary['vlm_path']}")
    print(f"image_attached events   : {summary['image_attached_count']}")
    print(f"image_attached bytes    : {summary['image_attached_total_bytes']:,}")
    print(f"vision_judge calls      : {summary['vision_judge_calls']}")
    print(f"  parse OK              : {summary['vision_judge_parse_ok']}")
    print(f"  parse FAILED          : {summary['vision_judge_parse_failed']}")
    if summary["first_user_feedback"]:
        print(f"first user feedback     : {summary['first_user_feedback']}")
    print()


def _compute_wasted_iters(records: list[dict]) -> tuple[int, list[int]]:
    """T-4 (offline, zero runtime cost): count `ok=False` iters whose code is
    byte-identical to the eventually-shipped build — pure harness friction
    (the block changed nothing). Computed from data the trace already holds:
    the per-iter `code_snapshot.html_sha256` vs the LAST snapshot's hash. Falls
    back to the agent-computed `shipped_unchanged_after_block` flag (T-2) for
    iters that carry it. Returns (count, sorted iter indices)."""
    sha_by_iter: dict[int, str] = {}
    last_sha: str | None = None
    flagged: set[int] = set()
    ok_by_iter: dict[int, bool] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "code_snapshot":
            it = int(rec.get("iteration") or 0)
            sha = rec.get("html_sha256")
            if sha:
                sha_by_iter[it] = str(sha)
                last_sha = str(sha)
        elif kind == "iter_summary":
            it = int(rec.get("iteration") or 0)
            ok_by_iter[it] = bool(rec.get("ok"))
            if rec.get("shipped_unchanged_after_block"):
                flagged.add(it)
    wasted: set[int] = set(flagged)
    if last_sha is not None:
        for it, ok in ok_by_iter.items():
            if not ok and sha_by_iter.get(it) == last_sha:
                wasted.add(it)
    return len(wasted), sorted(wasted)


def _retrieval_first_clean(
    records: list[dict],
) -> tuple[int | None, list[str]]:
    """M-2 (offline join, no new recording, no auto-learner): map the playbook
    bullet ids RETRIEVED this session to the first `ok=True` iter index. Joining
    the existing `playbook_retrieved` events to the first-clean iter — aggregated
    over a batch this yields a "bullet id x first-pass-clean" table that INFORMS
    hand-curation (counters stay hand-curated per CLAUDE.md). Returns
    (first_clean_iter or None, ordered-unique retrieved bullet ids)."""
    first_clean: int | None = None
    retrieved: list[str] = []
    seen: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "iter_summary" and rec.get("ok"):
            it = int(rec.get("iteration") or 0)
            if first_clean is None or it < first_clean:
                first_clean = it
        elif kind == "playbook_retrieved":
            for bid in (rec.get("ids") or []):
                if bid not in seen:
                    seen.add(bid)
                    retrieved.append(str(bid))
    return first_clean, retrieved


def _print_timeline(trace: Path) -> None:
    """Phase 4 (4D.3): one-screen per-iter digest for the reviewing LLM.

    Reuses `agent.render_run_summary` (the single source of truth for the iter
    table) so the timeline can't drift from what the harness records. Each row
    carries the fix-layer bucket (`class`: harness_bug / memory_gap /
    local_llm_limit) so "where does this fix go?" is answerable at a glance.

    Appends two offline diagnostics computed here (no runtime cost, no new
    recording): T-4 `wasted_iters` (harness-friction count) and M-2 the
    retrieved-bullet -> first-clean-iter join.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from agent import render_run_summary
    except Exception as e:  # pragma: no cover - import guard
        print(f"(timeline unavailable: {e})")
        return
    records = _load_events(trace)
    print(render_run_summary(records, artifact_id=trace.stem))
    # T-4: harness-friction scalar (ok=False iters identical to shipped build).
    n_wasted, wasted_iters = _compute_wasted_iters(records)
    detail = f" (iters {wasted_iters})" if wasted_iters else ""
    print(f"wasted_iters (ok=False, code == shipped build): {n_wasted}{detail}")
    # M-2: retrieval -> first-clean-iter join to inform playbook curation.
    first_clean, retrieved = _retrieval_first_clean(records)
    print(
        "playbook retrieval -> first clean iter: "
        f"first_clean={first_clean if first_clean is not None else 'never'}; "
        f"retrieved_ids={retrieved if retrieved else '[]'}"
    )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "target",
        help="Path to a .jsonl trace, or a session-id substring (e.g. 'donkey-kong').",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path (default: sibling .enriched.jsonl).",
    )
    ap.add_argument(
        "--timeline",
        action="store_true",
        help=(
            "Print the one-screen per-iter digest (iter/ok/probes/patch/router/"
            "tok-s/failure_class/blocker) and exit — no enrichment written."
        ),
    )
    args = ap.parse_args()

    trace, snapshots_dir = _resolve_paths(args.target)

    # Timeline is a read-only digest: print and exit without writing files.
    if args.timeline:
        _print_timeline(trace)
        return 0

    enriched, summary = enrich(trace, snapshots_dir)

    out_path = Path(args.out) if args.out else trace.with_suffix(".enriched.jsonl")
    with out_path.open("w", encoding="utf-8") as fh:
        for ev in enriched:
            fh.write(json.dumps(ev, separators=(",", ":")) + "\n")

    _print_summary(summary)
    print(f"enriched trace: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
