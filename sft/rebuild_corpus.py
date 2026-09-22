"""Rebuild html_corpus.jsonl from every HTML game on disk.

Uses rows.messages_for_html (real <html_file> labels). Replaces the sqlite
index so kinds match. Does not delete raw downloads.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rows import JMR_STORAGE, ROOT, connect, training_rows

GOODGAME = Path("/Users/jonathanrothberg/Agent_learning/goodgame")
RAW = ROOT / "raw"
CORPUS = ROOT / "jsonl" / "html_corpus.jsonl"
TRAIN = ROOT / "jsonl" / "train.jsonl"
STATE = ROOT / "logs" / "rebuild.json"


def _state(**kw) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if STATE.exists():
        try:
            cur = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur.update(kw)
    cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE.write_text(json.dumps(cur))


def _walk(root: Path, conn, out, label: str) -> dict:
    stats = {"gold": 0, "html": 0, "skip": 0, "unfitted": 0, "walked": 0}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in (".git", "node_modules", "dist", "vendor") and not d.startswith(".")
        ]
        for name in filenames:
            if not name.lower().endswith((".html", ".htm")):
                continue
            path = Path(dirpath) / name
            stats["walked"] += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                stats["skip"] += 1
                continue
            rows = training_rows(text, path)
            if not rows:
                import hashlib
                sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                if len(text) >= 80:
                    conn.execute(
                        "INSERT OR REPLACE INTO games (sha, path, title, kind, bytes) VALUES (?,?,?,?,?)",
                        (sha, str(path), path.stem[:120], "unfitted", len(text.encode("utf-8", errors="replace"))),
                    )
                    stats["unfitted"] += 1
                else:
                    stats["skip"] += 1
                continue
            for row in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO games (sha, path, title, kind, bytes) VALUES (?,?,?,?,?)",
                    (row["sha"], row["source"], row["title"], row["kind"], path.stat().st_size if path.exists() else 0),
                )
                out.write(json.dumps({k: row[k] for k in ("sha", "kind", "title", "source", "messages")}, ensure_ascii=False) + "\n")
                stats[row["kind"]] += 1
            if stats["walked"] % 500 == 0:
                conn.commit()
                _state(state="rebuilding", source=label, **stats)
    conn.commit()
    return stats


def main() -> None:
    conn = connect()
    conn.execute("DELETE FROM games")
    conn.commit()
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = CORPUS.with_suffix(".jsonl.tmp")
    total = {"gold": 0, "html": 0, "skip": 0, "unfitted": 0, "walked": 0}
    _state(state="rebuilding", source="start")
    with tmp.open("w", encoding="utf-8") as out:
        for root, label in (
            (JMR_STORAGE, "jmr_storage"),
            (GOODGAME, "goodgame"),
            (RAW, "raw"),
        ):
            if not root.is_dir():
                continue
            part = _walk(root, conn, out, label)
            for k in total:
                total[k] += part.get(k, 0)
    tmp.replace(CORPUS)
    TRAIN.write_text(CORPUS.read_text(encoding="utf-8"))
    train_rows = total["gold"] + total["html"]
    _state(state="done", train_rows=train_rows, **total)
    print(json.dumps({"train_rows": train_rows, **total}))


if __name__ == "__main__":
    main()
