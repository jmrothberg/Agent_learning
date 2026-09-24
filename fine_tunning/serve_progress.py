"""Local page for training and download progress. http://127.0.0.1:8766/ (LORA_PORT)"""
from __future__ import annotations

import calendar
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# MULTI-LORA: env overrides (see README.md). Defaults = the HTML-game LoRA.
ROOT = Path(os.environ.get("LORA_ROOT", "/Users/jonathanrothberg/MLX_Models/html_game_sft")).expanduser()
HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("LORA_PORT", "8766"))
DB = ROOT / "games.sqlite"
# Page ships with the code in fine_tunning/, not in the data folder.
HTML = HERE / "progress.html"
PY = os.path.expanduser(os.environ.get("LORA_PY", "/Users/jonathanrothberg/Agents/.venv/bin/python"))
HOLD = ROOT / "logs" / "hold.json"
_LOSS = re.compile(r"Iter \d+: Train loss .*?([0-9]+\.[0-9]+)")
# SMALL MODEL: train_small.py logs Tokens/sec on every Iter line; the page plots it.
_SPEED = re.compile(r"Tokens/sec ([0-9]+\.?[0-9]*)")
_STAMP = re.compile(r"\d{8}T\d{6}Z")


def _python_script_pids(script_name: str) -> list[int]:
    # Match the Python process only. A shell whose command line mentions the
    # script path is not the trainer and must not be signaled.
    out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True, errors="replace")
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        # MULTI-LORA: fine_tunning/ copy, or the older <data>/scripts/ copy still in use.
        needles = (f"{HERE}/{script_name}", f"/{ROOT.name}/scripts/{script_name}")
        # SMALL MODEL: train_small.py is started from fine_tunning/small/ with a bare name.
        if script_name == "train_small.py":
            needles += (f"Python {script_name}", f"small/{script_name}")
        if not any(n in cmd for n in needles) or "Python" not in cmd or "zsh" in cmd:
            continue
        found.append(int(pid_s))
    return found


def _agent_model_pids() -> list[int]:
    # chat.py holds the 27B in unified memory. Training cannot load beside it.
    out = subprocess.check_output(
        ["ps", "-ax", "-o", "pid=,rss=,command="], text=True, errors="replace"
    )
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid_s, rss_s, cmd = parts
        if "Python" not in cmd or "zsh" in cmd:
            continue
        # SMALL MODEL: train_small.py is a trainer (>20 GB), never treat it as the agent model.
        if any(skip in cmd for skip in ("serve_progress.py", "train_lora.py", "run_slices.py", "ingest.py",
                                        "train_small.py")):
            continue
        named = any(mark in cmd for mark in ("chat.py", "coder.py", "mlx_vlm.server", "mlx_lm.server"))
        huge = int(rss_s) > 20_000_000  # rss is KB; 20 GB is a loaded 27B, not the dashboard
        if named or huge:
            found.append(int(pid_s))
    return found


def _signal_gone(pids: list[int], sig: int, wait_s: float) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass
    deadline = time.time() + wait_s
    while time.time() < deadline:
        alive = False
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if not alive:
            return
        time.sleep(0.5)


def _checkpoints() -> list[dict]:
    """Dated snapshot folders, oldest first. Nothing here deletes an older one."""
    root = ROOT / "snapshots"
    if not root.is_dir():
        return []
    items = []
    for path in root.iterdir():
        if not path.is_dir() or not _STAMP.fullmatch(path.name):
            continue
        weights = path / "adapters.safetensors"
        if not weights.exists():
            weights = path / "model.safetensors"  # SMALL MODEL: full-weight snapshots
        if not weights.exists() or weights.stat().st_size < 100_000_000:
            continue
        parsed = time.strptime(path.name, "%Y%m%dT%H%M%SZ")
        when = time.strftime("%b %d, %Y, %I:%M %p", time.localtime(calendar.timegm(parsed)))
        items.append({"name": path.name, "when": when, "path": str(path)})
    items.sort(key=lambda item: item["name"])
    return items


def _latest_snapshot() -> str | None:
    root = ROOT / "snapshots"
    if not root.is_dir():
        return None
    best = None
    for path in root.iterdir():
        if not path.is_dir() or not _STAMP.fullmatch(path.name):
            continue
        weights = path / "adapters.safetensors"
        if weights.exists() and weights.stat().st_size > 100_000_000:
            if best is None or path.name > best.name:
                best = path
    return str(best) if best else None


def _control() -> dict:
    try:
        # SMALL MODEL: train_small.py counts as a running trainer too.
        trainer = (_python_script_pids("train_lora.py") + _python_script_pids("run_slices.py")
                   + _python_script_pids("train_small.py"))
        agent = [] if trainer else _agent_model_pids()
        checkpoint = _latest_snapshot()
        running = bool(trainer)
    except (OSError, subprocess.SubprocessError, ValueError):
        # Unknown is not safe. Do not tell the page the GPU is free.
        return {
            "trainer_running": True,
            "agent_loaded": False,
            "safe_to_test": False,
            "checkpoint": None,
        }
    return {
        "trainer_running": running,
        "agent_loaded": bool(agent),
        "safe_to_test": (not running) and (not agent) and bool(checkpoint),
        "checkpoint": checkpoint,
    }


def hold_training() -> dict:
    # The live supervisor was started before it knew this file. Stopping the
    # Python PIDs is what frees the GPU. The file stops the next start.
    HOLD.parent.mkdir(parents=True, exist_ok=True)
    HOLD.write_text(json.dumps({"state": "held"}))
    _signal_gone(_python_script_pids("train_lora.py"), signal.SIGTERM, 45)
    _signal_gone(_python_script_pids("train_lora.py"), signal.SIGKILL, 10)
    _signal_gone(_python_script_pids("run_slices.py"), signal.SIGTERM, 20)
    _signal_gone(_python_script_pids("run_slices.py"), signal.SIGKILL, 10)
    return status()


def resume_training() -> dict:
    if _python_script_pids("train_lora.py") or _python_script_pids("run_slices.py"):
        return status()
    # Quit the agent model first. A second 27B does not fit beside training.
    _signal_gone(_agent_model_pids(), signal.SIGTERM, 30)
    _signal_gone(_agent_model_pids(), signal.SIGKILL, 15)
    time.sleep(3)
    HOLD.write_text(json.dumps({"state": "running"}))
    if not _python_script_pids("run_slices.py"):
        log = open(ROOT / "logs" / "supervisor.log", "a", encoding="utf-8")
        subprocess.Popen(
            [PY, str(HERE / "run_slices.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(ROOT),
        )
    deadline = time.time() + 5
    while time.time() < deadline and not _python_script_pids("run_slices.py"):
        time.sleep(0.2)
    return status()


_corpus_cache: dict = {"at": 0.0, "scan": {}}


def _scan_shards() -> dict:
    """One pass over shard metadata. Cached 60s — the files change slowly."""
    now = time.time()
    if now - _corpus_cache["at"] < 60 and _corpus_cache["scan"]:
        return _corpus_cache["scan"]
    docs = tokens = 0
    by_src: dict[str, int] = {}
    norms: set[str] = set()
    shards = ROOT / "shards"
    if shards.is_dir():
        for meta in shards.glob("*.jsonl"):
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docs += 1
                tokens += int(d.get("ntok") or 0)
                src = d.get("src") or "?"
                by_src[src] = by_src.get(src, 0) + 1
                n = d.get("norm")
                if n:
                    norms.add(n)
    scan = {"docs": docs, "unique": len(norms), "tokens": tokens, "by_src": by_src}
    _corpus_cache["at"] = now
    _corpus_cache["scan"] = scan
    return scan


def _quality_modes() -> list[str]:
    """Which CPU review passes are running: score, edu, browser. Empty if none."""
    out = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True, errors="replace")
    modes = []
    for cmd in out.splitlines():
        if "small/quality_worker.py" not in cmd or "Python" not in cmd or "zsh" in cmd:
            continue
        if "--edu" in cmd:
            modes.append("edu")
        elif "--browser" in cmd:
            modes.append("browser")
        else:
            modes.append("score")
    return modes


def _quality_rate() -> dict:
    """Last 'edu: N docs X/s' or 'browser: N pages' line. Reads the tail only."""
    info = {"edu_rate": 0.0, "edu_session": 0, "browser_session": 0}
    for name, kind in (("quality.out", "edu"), ("browser.out", "browser")):
        path = ROOT / "logs" / name
        if not path.exists():
            continue
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                f.seek(max(0, size - 12000))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if kind == "edu":
            hits = re.findall(r"edu: (\d+) docs ([0-9.]+)/s", tail)
            if hits:
                info["edu_session"] = int(hits[-1][0])
                info["edu_rate"] = float(hits[-1][1])
        else:
            hits = re.findall(r"browser: (\d+) pages", tail)
            if hits:
                info["browser_session"] = int(hits[-1])
    return info


_quality_cache: dict = {"at": 0.0, "row": {}}


def _quality_row() -> dict:
    """Counts from quality.sqlite. Cached 60s — the eligible-file count takes several seconds."""
    now = time.time()
    if now - _quality_cache["at"] < 60 and _quality_cache["row"]:
        return _quality_cache["row"]
    shards = ROOT / "shards"
    row = {
        "shards": len(list(shards.glob("*.jsonl"))) if shards.is_dir() else 0,
        "shards_done": 0, "scored": 0, "dropped": 0, "edu": 0, "edu_left": 0,
        "browser": 0, "browser_left": 0,
    }
    q = ROOT / "quality.sqlite"
    if q.exists():
        try:
            con = sqlite3.connect(f"file:{q}?mode=ro", uri=True, timeout=5)
            row["shards_done"] = con.execute("select count(*) from done").fetchone()[0]
            row["scored"] = con.execute("select count(*) from quality").fetchone()[0]
            row["dropped"] = con.execute("select count(*) from quality where weight = 0").fetchone()[0]
            row["edu"] = con.execute("select count(*) from edu").fetchone()[0]
            row["browser"] = con.execute("select count(*) from browser").fetchone()[0]
            row["browser_left"] = con.execute("select count(*) from pending_browser").fetchone()[0]
            # Github files still eligible for Stack-Edu: not already scored, not a near-dup (weight 0).
            row["edu_left"] = con.execute(
                "select count(*) from feat f "
                "left join edu e on e.norm = f.norm "
                "left join quality q on q.norm = f.norm and q.weight = 0 "
                "where f.src = 'gcc' and e.norm is null and q.norm is null"
            ).fetchone()[0]
            con.close()
        except sqlite3.Error:
            pass
    _quality_cache["at"] = now
    _quality_cache["row"] = row
    return row


def _small_corpus() -> dict:
    """Training-set size and quality-worker progress for the small-model page."""
    # Counts are cached. Running/not and the files-per-second line update every poll.
    row = {**_scan_shards(), **_quality_row(), **_quality_rate()}
    row["modes"] = _quality_modes()
    return row


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
    speeds: list[float] = []  # SMALL MODEL: Tokens/sec per Iter line
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        start = 0
        for i, line in enumerate(lines):
            if "resume=None" in line:
                start = i
        slice_starts = 0
        for line in lines[start:]:
            if line.startswith("--- slice"):
                slice_starts += 1
            m = _LOSS.search(line)
            if m:
                losses.append(float(m.group(1)))
                sp = _SPEED.search(line)
                if sp:
                    speeds.append(float(sp.group(1)))
    else:
        slice_starts = 0
    rows_this = 0
    row_match = re.search(r"(\d+)\s+rows", str(train.get("detail") or ""))
    if row_match:
        rows_this = int(row_match.group(1))
    rate = float(train.get("it_per_sec") or 0)
    epochs = (len(losses) / rows_this) if rows_this else None
    epoch_days = (rows_this / rate / 86400) if rows_this and rate > 0 else None
    checkpoints = _checkpoints()
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
        "speeds": speeds,
        "small": bool(train.get("tok_per_sec")),  # SMALL MODEL: train_small.py state.json
        "corpus": _small_corpus() if train.get("tok_per_sec") else {},
        "steps_trained": len(losses),
        "slice_starts": slice_starts,
        "epochs": epochs,
        "epoch_rows": rows_this,
        "epoch_days": epoch_days,
        "checkpoints": checkpoints,
        **_control(),
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

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/hold":
            body = hold_training()
        elif path == "/resume":
            body = resume_training()
        else:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, json.dumps(body).encode(), "application/json")


def main() -> None:
    # MULTI-LORA: LORA_PORT lets a second project's page run beside this one.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    # 8765 is chat.py Asset Studio. This page stays on 8766.
    print(f"progress http://127.0.0.1:{PORT}/  root={ROOT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
