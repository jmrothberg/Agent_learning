"""Remove what the agent "learned" from a single past session.

Use when a session shipped a wrong-but-passing game (e.g. the May 5
Missile Command run that was actually Space Invaders with the labels
swapped — its `won_*.html` would otherwise get retrieved as the
starting skeleton next time someone asked for Missile Command).

What gets deleted:
  - memory/skeletons/won_<session_id>.{html,json}
  - memory/goals/<session_id>/  (the win-record dir)
  - memory/mistakes.jsonl entries whose 'session' field matches
    (only some entries carry one; we keep the rest)

What is NOT touched:
  - playbook.jsonl  (curated bullets — none came from a single session)
  - per-session traces / .best.html / .conversation.md under games/
    (those are read-only history; delete by hand if you also want them
    gone)

Usage:
    .venv/bin/python scripts/forget_session.py <session_id>
    .venv/bin/python scripts/forget_session.py --list      # show won_* + goals
    .venv/bin/python scripts/forget_session.py --dry-run <id>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MEM = REPO_ROOT / "memory"


def list_state() -> int:
    skel_dir = MEM / "skeletons"
    goals_dir = MEM / "goals"
    print("won_* skeletons:")
    for p in sorted(skel_dir.glob("won_*.json")) if skel_dir.exists() else []:
        try:
            d = json.loads(p.read_text())
            sid = d.get("session_id", "?")
            goal = (d.get("goal") or "")[:80]
            print(f"  {sid}\n    goal: {goal}")
        except Exception as e:
            print(f"  {p.name}  (unreadable: {e})")
    print()
    print("goals/ records:")
    for p in sorted(goals_dir.iterdir()) if goals_dir.exists() else []:
        if p.is_dir():
            print(f"  {p.name}")
    return 0


def forget(session_id: str, *, dry_run: bool) -> int:
    skel_dir = MEM / "skeletons"
    goals_dir = MEM / "goals"
    targets: list[Path] = []
    for ext in (".html", ".json"):
        f = skel_dir / f"won_{session_id}{ext}"
        if f.exists():
            targets.append(f)
    g = goals_dir / session_id
    if g.exists():
        targets.append(g)

    if not targets:
        print(f"nothing to forget for session_id={session_id!r}", file=sys.stderr)
        print("(use --list to see what's stored)", file=sys.stderr)
        return 1

    for t in targets:
        kind = "dir " if t.is_dir() else "file"
        if dry_run:
            print(f"[dry-run] would remove {kind}: {t.relative_to(REPO_ROOT)}")
            continue
        if t.is_dir():
            shutil.rmtree(t)
        else:
            t.unlink()
        print(f"removed {kind}: {t.relative_to(REPO_ROOT)}")

    # Best-effort: prune mistakes.jsonl entries with a matching session
    # field. Most entries don't carry one (the writer only adds it for
    # newer traces), so this rewrite is small and safe.
    mistakes = MEM / "mistakes.jsonl"
    if mistakes.exists():
        kept: list[str] = []
        dropped = 0
        for line in mistakes.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                kept.append(line)
                continue
            if obj.get("session") == session_id:
                dropped += 1
                continue
            kept.append(line)
        if dropped:
            if dry_run:
                print(f"[dry-run] would prune {dropped} mistakes.jsonl entry/entries")
            else:
                mistakes.write_text("\n".join(kept) + ("\n" if kept else ""))
                print(f"pruned {dropped} mistakes.jsonl entry/entries")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_id", nargs="?", help="basename of the session (no extension, no path)")
    p.add_argument("--list", action="store_true", help="show what's stored, then exit")
    p.add_argument("--dry-run", action="store_true", help="print what would be removed, do nothing")
    args = p.parse_args(argv[1:])
    if args.list:
        return list_state()
    if not args.session_id:
        p.print_help(sys.stderr)
        return 2
    return forget(args.session_id, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
