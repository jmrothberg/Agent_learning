"""Local page for training and download progress. http://127.0.0.1:8765/"""
from __future__ import annotations

import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/Users/jonathanrothberg/MLX_Models/html_game_sft")
DB = ROOT / "games.sqlite"
HTML = Path(__file__).resolve().parent / "progress.html"
_LOSS = re.compile(r"Iter \d+: Train loss .*?([0-9]+\.[0-9]+)")


def status() -> dict:
    unique = gold = html = plan = unfitted = 0
    gold_files = html_files = 0
    if DB.exists():
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
        try:
            unique = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            for kind, n in conn.execute("SELECT kind, COUNT(*) FROM games GROUP BY kind"):
                if kind == "gold":
                    gold = n
                elif kind == "html":
                    html = n
                elif kind == "plan":
                    plan = n
                elif kind == "unfitted":
                    unfitted = n
            gold_files = conn.execute(
                "SELECT COUNT(DISTINCT path) FROM games WHERE kind='gold'"
            ).fetchone()[0]
            html_files = conn.execute(
                "SELECT COUNT(DISTINCT path) FROM games WHERE kind='html'"
            ).fetchone()[0]
        except sqlite3.Error:
            pass
        finally:
            conn.close()
    train = {}
    ingest = {}
    rebuild = {}
    for name, dest in (("state.json", "train"), ("ingest.json", "ingest"), ("rebuild.json", "rebuild")):
        path = ROOT / "logs" / name
        if path.exists():
            try:
                blob = json.loads(path.read_text())
            except json.JSONDecodeError:
                blob = {}
            if dest == "train":
                train = blob
            elif dest == "rebuild":
                rebuild = blob
            else:
                ingest = blob
    tail = ""
    log = ROOT / "logs" / "train.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        # Older slices stay in this file. Show only the slice that is running.
        cut = 0
        for i, line in enumerate(lines):
            if line.startswith("--- slice"):
                cut = i
        tail = "\n".join(lines[cut:][-24:])
    # This GPU run starts at the last fresh LoRA. Losses after that are the plot.
    losses: list[float] = []
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        start = 0
        for i, line in enumerate(lines):
            if "resume=None" in line:
                start = i
        for line in lines[start:]:
            m = _LOSS.search(line)
            if m:
                losses.append(float(m.group(1)))
    return {
        "unique_html_games": unique,
        "gold_rows": gold,
        "gold_files": gold_files,
        "html_rows": html,
        "html_files": html_files,
        "plan_rows": plan,
        "unfitted": unfitted,
        "training_rows": gold + html,
        "train": train,
        "ingest": ingest,
        "rebuild": rebuild,
        "log_tail": tail,
        "losses": losses,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/status.json"):
            self._send(200, json.dumps(status()).encode(), "application/json")
            return
        if self.path in ("/", "/progress.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    # 8765 is chat.py Asset Studio. This page stays on 8766.
    print("progress http://127.0.0.1:8766/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
