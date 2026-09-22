"""Download HTML/JS games and append /640png training rows.

No license filter and no count cap. A game detector still skips pages
that are not games. Unique games are sha256 of the file bytes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

from rows import ROOT, already, connect, is_jmr_storage, looks_like_game, training_rows

RAW = ROOT / "raw"
# New and newly split files land here. The trainer reloads this at
# each 30-minute checkpoint. Do not rewrite html_corpus.jsonl.
JSONL = ROOT / "jsonl" / "added.jsonl"
SEED = ROOT / "jsonl" / "seed.jsonl"
STATE = ROOT / "logs" / "ingest.json"
GOODGAME = Path("/Users/jonathanrothberg/Agent_learning/goodgame")

# Public HTML-game trees. 404s are skipped. js13k org repos are added at runtime.
SEED_REPOS = [
    "https://github.com/phaserjs/examples.git",
    "https://github.com/melonjs/examples.git",
    "https://github.com/KilledByAPixel/LittleJS.git",
    "https://github.com/gabrielecirulli/2048.git",
    "https://github.com/mozilla/BrowserQuest.git",
    "https://github.com/mozdevs/gamedev-js-tiles.git",
    "https://github.com/photonstorm/phaser3-examples.git",
    "https://github.com/silver-ai/html5-game-collection.git",
    "https://github.com/pixijs/examples.git",
    "https://github.com/end3r/Gamedev-Canvas-workshop.git",
    "https://github.com/straker/kontra.git",
    "https://github.com/kaplayjs/kaplay.git",
    "https://github.com/jakesgordon/javascript-racer.git",
    "https://github.com/jakesgordon/javascript-breakout.git",
    "https://github.com/jakesgordon/javascript-pong.git",
    "https://github.com/maryrosecook/coquette.git",
    "https://github.com/craftyjs/Crafty.git",
    "https://github.com/playcanvas/engine.git",
]


def _write_state(**kw) -> None:
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


def _unique(conn) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    except sqlite3.OperationalError:
        return -1


def absorb_file(conn, path: Path, out) -> str:
    """Index one HTML file. Returns gold|plan|dup|skip."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "skip"
    if not is_jmr_storage(path) and not looks_like_game(text):
        return "skip"
    rows = training_rows(text, path)
    if not rows:
        # Count the game even when the row does not fit the token budget.
        import hashlib
        sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        try:
            if already(conn, sha):
                return "dup"
            conn.execute(
                "INSERT INTO games (sha, path, title, kind, bytes) VALUES (?,?,?,?,?)",
                (sha, str(path), path.stem[:120], "unfitted", len(text.encode("utf-8", errors="replace"))),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        return "unfitted"
    wrote = 0
    for row in rows:
        try:
            if already(conn, row["sha"]):
                continue
            conn.execute(
                "INSERT INTO games (sha, path, title, kind, bytes) VALUES (?,?,?,?,?)",
                (row["sha"], row["source"], row["title"], row["kind"], path.stat().st_size if path.exists() else 0),
            )
        except sqlite3.OperationalError:
            # Rebuild holds the database. The jsonl line is what training reads.
            pass
        out.write(json.dumps({k: row[k] for k in ("sha", "kind", "title", "source", "messages")}, ensure_ascii=False) + "\n")
        wrote += 1
    try:
        conn.commit()
    except sqlite3.OperationalError:
        pass
    out.flush()
    if wrote == 0:
        return "dup"
    return rows[0]["kind"]


def absorb_tree(conn, folder: Path, out, label: str) -> None:
    n = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [
            d for d in dirnames
            if d not in (".git", "node_modules", "dist", "vendor") and not d.startswith(".")
        ]
        for name in filenames:
            if not name.lower().endswith((".html", ".htm")):
                continue
            kind = absorb_file(conn, Path(dirpath) / name, out)
            n += 1
            if n % 50 == 0:
                _write_state(state="parsing", source=label, files_walked=n, unique=_unique(conn), last=kind)
    _write_state(state="parsing", source=label, files_walked=n, unique=_unique(conn))


def _clone(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)],
        capture_output=True, text=True, timeout=600,
    )
    return proc.returncode == 0


def _gh_clone_urls() -> list[str]:
    urls: list[str] = []
    queries = [
        ["gh", "api", "--paginate", "orgs/js13kGames/repos?per_page=100", "--jq", ".[].clone_url"],
    ]
    # Each search is a different slice of public HTML/JS games. Existing
    # clone folders are skipped later, so repeating a query does not re-download.
    for q in (
        "js13k game",
        "topic:js13k",
        "topic:html5-game",
        "topic:html5-canvas",
        "topic:canvas-game",
        "phaser html5 game",
        "kaboom.js game",
        "kontra canvas game",
        "vanilla javascript canvas game",
        "html5 game jam",
    ):
        queries.append(["gh", "search", "repos", q, "--limit", "100", "--json", "url"])
    for cmd in queries:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        text = proc.stdout or ""
        if text.strip().startswith("[") or text.strip().startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("url"):
                        urls.append(item["url"] + ".git")
                    elif isinstance(item, str) and "github.com" in item:
                        urls.append(item if item.endswith(".git") else item + ".git")
        else:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("https://github.com/"):
                    urls.append(line if line.endswith(".git") else line + ".git")
    # Keep order, drop dupes.
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _search_queries() -> list[str]:
    """HTML + JavaScript that opens in Chrome, including three.js."""
    topics = (
        "three.js game html",
        "threejs webgl game",
        "vanilla javascript canvas game",
        "html javascript canvas game",
        "single file html canvas game",
        "pure javascript browser game",
        "javascript requestAnimationFrame canvas game",
        "html css javascript game canvas",
        "three.js mini game",
        "webgl game html javascript",
        "html5 canvas game javascript",
        "single html file game javascript",
        "vanilla js game canvas",
    )
    out = list(topics)
    for year in range(2013, 2027):
        for base in ("three.js game", "javascript canvas game", "html canvas game", "vanilla javascript game"):
            out.append(f"{base} created:{year}-01-01..{year}-12-31")
    return out


def _urls_for_query(q: str) -> list[str]:
    cmd = ["gh", "search", "repos", q, "--limit", "100", "--json", "url"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    urls = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("url"):
                u = item["url"]
                urls.append(u if u.endswith(".git") else u + ".git")
    return urls


def _known_sources() -> set[str]:
    """Paths already in the training jsonl. A long file split into slices counts once."""
    known: set[str] = set()
    for path in (ROOT / "jsonl" / "html_corpus.jsonl", JSONL):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A bad line can be a bare number. Skip it instead of dying.
            if not isinstance(obj, dict):
                continue
            known.add(obj.get("source") or "")
    known.discard("")
    return known


def _split_on_disk(conn, known: set[str]) -> int:
    """Append split HTML/JS for files already downloaded. Does not rewrite the corpus."""
    from rows import training_rows
    added = 0
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(RAW):
            dirnames[:] = [
                d for d in dirnames
                if d not in (".git", "node_modules", "dist", "vendor") and not d.startswith(".")
            ]
            for name in filenames:
                if not name.lower().endswith((".html", ".htm")):
                    continue
                path = Path(dirpath) / name
                if str(path) in known:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rows = training_rows(text, path)
                if not rows:
                    known.add(str(path))
                    continue
                for row in rows:
                    if already(conn, row["sha"]):
                        continue
                    conn.execute(
                        "INSERT INTO games (sha, path, title, kind, bytes) VALUES (?,?,?,?,?)",
                        (row["sha"], row["source"], row["title"], row["kind"], path.stat().st_size if path.exists() else 0),
                    )
                    out.write(json.dumps({k: row[k] for k in ("sha", "kind", "title", "source", "messages")}, ensure_ascii=False) + "\n")
                    added += 1
                conn.commit()
                out.flush()
                known.add(str(path))
                if added and added % 50 == 0:
                    _write_state(state="splitting", source=str(path), unique=_unique(conn), html_added=added)
    return added


def main() -> None:
    """Clone new HTML/JS games all night. Append rows. Do not stop the trainer."""
    RAW.mkdir(parents=True, exist_ok=True)
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "seed.ready").write_text("html")
    conn = connect()
    seen_path = ROOT / "logs" / "seen_repos.txt"
    seen = set()
    if seen_path.exists():
        seen = {ln.strip() for ln in seen_path.read_text().splitlines() if ln.strip()}
    known = _known_sources()
    # SKIP_SPLIT=1 searches the web only. The full disk walk waits until
    # the rebuild is done so the two jobs do not lock the same database.
    if os.environ.get("SKIP_SPLIT") == "1":
        added_total = 0
    else:
        added_total = _split_on_disk(conn, known)
        _write_state(state="split_done", source="raw", unique=_unique(conn), html_added=added_total)
    while True:
        for q in _search_queries():
            _write_state(state="listing", source=q, unique=_unique(conn), html_added=added_total)
            for url in _urls_for_query(q):
                if url in seen:
                    continue
                name = url.rstrip("/").split("/")[-1].removesuffix(".git")
                org = url.rstrip("/").split("/")[-2]
                dest = RAW / "github" / org / name
                if dest.exists():
                    seen.add(url)
                    continue
                _write_state(state="cloning", source=url, unique=_unique(conn), html_added=added_total)
                try:
                    ok = _clone(url, dest)
                except subprocess.TimeoutExpired:
                    ok = False
                if not ok or not dest.exists():
                    _write_state(state="clone_failed", source=url, unique=_unique(conn), html_added=added_total)
                    continue
                seen.add(url)
                seen_path.write_text("\n".join(sorted(seen)) + "\n")
                try:
                    before = conn.execute(
                        "SELECT COUNT(*) FROM games WHERE kind IN ('html','gold')"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    before = added_total
                with JSONL.open("a", encoding="utf-8") as out:
                    absorb_tree(conn, dest, out, url)
                try:
                    after = conn.execute(
                        "SELECT COUNT(*) FROM games WHERE kind IN ('html','gold')"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    after = before
                added_total += max(0, after - before)
                _write_state(state="parsing", source=url, unique=_unique(conn), html_added=added_total)
        _write_state(state="sleeping", source="next search pass", unique=_unique(conn), html_added=added_total)
        time.sleep(300)


if __name__ == "__main__":
    main()
