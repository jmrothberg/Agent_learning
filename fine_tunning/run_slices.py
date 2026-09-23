"""Smoke, then repeat ~6 hour LoRA slices. Reload jsonl between slices."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from rows import game_rank

# MULTI-LORA: env overrides (see README.md). Defaults = the HTML-game LoRA.
ROOT = Path(os.environ.get("LORA_ROOT", "/Users/jonathanrothberg/MLX_Models/html_game_sft")).expanduser()
PY = os.path.expanduser(os.environ.get("LORA_PY", "/Users/jonathanrothberg/Agents/.venv/bin/python"))
BASE = os.path.expanduser(os.environ.get("LORA_BASE", "/Users/jonathanrothberg/MLX_Models/Qwen3.8-27B-mxfp8"))
# Extra train_lora.py flags, e.g. "--system full --whole-games 0" for non-game data.
TRAIN_ARGS = os.environ.get("LORA_TRAIN_ARGS", "").split()
# 1 = keep only rows game_rank() calls a game (HTML LoRA). 0 = train every row.
GAME_FILTER = os.environ.get("LORA_GAME_FILTER", "1") == "1"
SEED = ROOT / "jsonl" / "seed.jsonl"
INCOMING = ROOT / "jsonl" / "incoming.jsonl"
CORPUS = ROOT / "jsonl" / "html_corpus.jsonl"
TRAIN = ROOT / "jsonl" / "train.jsonl"
LOG = ROOT / "logs" / "train.log"
STATE = ROOT / "logs" / "state.json"
ADAPTERS = ROOT / "adapters"
READY = ROOT / "logs" / "seed.ready"
# 30-minute checkpoints so new split files join without a 6-hour wait.
SLICE_SECONDS = int(os.environ.get("LORA_SLICE_MIN", "30")) * 60

_LOSS = re.compile(
    r"Iter (\d+): Train loss .*?([0-9]+\.[0-9]+).*?It/sec ([0-9.]+).*?Peak mem ([0-9.]+) GB"
)


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


def _wait_seed() -> None:
    _state(state="waiting_for_seed", detail="indexing goodgame/")
    # MULTI-LORA: a project with rows already in jsonl/ does not need the goodgame seed.
    have_rows = any(p.exists() and p.stat().st_size > 0 for p in (CORPUS, ROOT / "jsonl" / "added.jsonl"))
    while not have_rows and (not READY.exists() or not SEED.exists() or SEED.stat().st_size == 0):
        time.sleep(2)
    _state(state="seed_ready", detail=SEED.name)


def _assemble() -> int:
    # Corpus, overnight additions, and the file the rebuild is still
    # writing. Tmp is last so the larger HTML set wins. Plan-card files
    # stay out: the same sha would replace real HTML.
    lines = []
    paths = [CORPUS, ROOT / "jsonl" / "added.jsonl", ROOT / "jsonl" / "html_corpus.jsonl.tmp"]
    for path in paths:
        if path.exists():
            lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    # Last write wins per sha so a later gold label replaces a plan row.
    by_sha = {}
    order = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "messages" not in obj:
            continue
        sha = obj.get("sha") or str(len(order))
        if sha not in by_sha:
            order.append(sha)
        by_sha[sha] = line
    # The trainer draws a random handful from this file. Keep known games
    # in it. Other HTML stays in the corpus and is not deleted.
    if GAME_FILTER:
        ranked = [(game_rank(by_sha[sha]), sha) for sha in order]
        games = [sha for rank, sha in ranked if rank]
        if not games:
            games = list(order)
        games.sort(key=lambda sha: -game_rank(by_sha[sha]))
    else:
        games = list(order)
    held = len(order) - len(games)
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text("\n".join(by_sha[s] for s in games) + ("\n" if games else ""))
    print(f"assemble games={len(games)} held_back={held}", flush=True)
    return len(games)


def _run(iters: int, resume: bool) -> tuple[int, float | None]:
    cmd = [
        PY, str(Path(__file__).resolve().parent / "train_lora.py"),
        "--jsonl", str(TRAIN),
        "--iters", str(iters),
        "--steps-per-report", "1",
        "--steps-per-save", "10",
        "--max-seq-length", "8192",
    ] + TRAIN_ARGS
    if resume and (ADAPTERS / "adapter_config.json").exists():
        cmd.extend(["--adapter-path", str(ADAPTERS)])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    it_per_sec = None
    last_step = 0
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n--- slice iters={iters} resume={resume} ---\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = _LOSS.search(line)
            if m:
                last_step = int(m.group(1))
                it_per_sec = float(m.group(3))
                _state(
                    step=last_step,
                    loss=float(m.group(2)),
                    it_per_sec=it_per_sec,
                    peak_gb=float(m.group(4)),
                    iters_this_slice=iters,
                )
        code = proc.wait()
    return code, it_per_sec


def _snapshot() -> str | None:
    src = ADAPTERS / "adapters.safetensors"
    cfg = ADAPTERS / "adapter_config.json"
    if not src.exists() or not cfg.exists():
        return None
    # New folder every save. Older snapshot directories are left in place.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    when = time.strftime("%b %d, %Y, %I:%M %p %Z", time.localtime())
    dest = ROOT / "snapshots" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "adapters.safetensors")
    shutil.copy2(cfg, dest / "adapter_config.json")
    (dest / "READY").write_text(
        f"DATE={when}\n"
        f"MLX_MODEL={BASE}\n"
        f"MLX_ADAPTER={dest}\n"
        "Then /640png before /new. Vision tower is the base model.\n"
    )
    return str(dest)


def _held() -> bool:
    # Dashboard writes this. A fresh supervisor waits here so HOLD can pause
    # between slices without loading another copy of the model.
    path = ROOT / "logs" / "hold.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("state") == "held"
    except json.JSONDecodeError:
        return False


def main() -> None:
    lock = ROOT / "logs" / "trainer.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            print(f"trainer already running pid {pid}", flush=True)
            return
        except (OSError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    _wait_seed()
    rate = 0.01
    first = True
    while True:
        while _held():
            _state(state="held", detail="paused so the agent can load a checkpoint")
            time.sleep(2)
        rows = _assemble()
        iters = max(8, int(SLICE_SECONDS * rate))
        # A restart after HOLD must keep the saved LoRA. resume=False would
        # train a new adapter and overwrite adapters/adapters.safetensors.
        has_weights = (ADAPTERS / "adapters.safetensors").exists()
        resume = (not first) or has_weights
        label = "~30m slice" if resume else "first 30m"
        _state(state="running", detail=f"{rows} rows, {label}", iters_this_slice=iters, step=0)
        t0 = time.time()
        code, new_rate = _run(iters, resume=resume)
        first = False
        if new_rate and new_rate > 0:
            rate = new_rate
        snap = _snapshot()
        if code != 0:
            _state(state="error", detail=f"slice exit {code}", snapshot=snap)
            return
        _state(
            state="between_slices",
            detail=snap or "",
            snapshot=snap,
            slice_seconds=round(time.time() - t0),
        )


if __name__ == "__main__":
    main()
