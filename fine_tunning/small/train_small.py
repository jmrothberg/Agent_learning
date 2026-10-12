"""Stage-1 continued pretraining of the small HTML/JS model (full fine-tune, MLX).

Reads the shards written by data.py, packs them into fixed blocks (no padding),
and trains every weight of the base model. Loss is on the page, not on the
request in front of it, and not on the token that only separates documents.

  python train_small.py                 # start or resume
  python train_small.py --bench 20      # timing only, nothing saved

Every --save-minutes it writes checkpoints/latest/ (weights + optimizer +
counters), overwriting that folder, and re-reads shards/ and quality.sqlite.
Every 6 hours it also keeps a dated weights-only copy in snapshots/<date>/.
Those dated copies are never deleted. New shards and quality scores join the
next blocks without a restart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load

from data import training_dir

ROOT = training_dir("SMALL_ROOT", "html_js_small")
BASE = Path(os.environ.get("SMALL_BASE", "~/MLX_Models/MiniCPM5-1B-Base")).expanduser()
SHARDS = ROOT / "shards"
CKPT = ROOT / "checkpoints"
LOG = ROOT / "logs" / "train.log"
STATE = ROOT / "logs" / "state.json"
QUALITY = ROOT / "quality.sqlite"

# How often one file is drawn, before the quality score and the length factor.
# Rank 2 = canvas or three.js with a loop. Rank 1 = canvas or a loop.
# Rank 0 = a plain file (libraries, tests, node_modules). It is not drawn.
# Those files stay on disk. They were most of GitHub and The Stack, and they
# taught this model to write ordinary JavaScript instead of a finished game.
# ask = a request, then one of our own games. close = the ending of a real page.
# The request tokens are not in the loss. See data.py --source game-sft.
# Synthetic games are one model's voice, so they are not the bulk of a pass.
SET_WEIGHT = {
    "synth": 12.0,
    "own": 25.0,
    "ask": 35.0,
    "close": 40.0,
    "gcc": 0.5,
    "stack": 0.1,
}
_RANK_FACTOR = {2: 1.0, 1: 0.6, 0: 0.0}
# quality.sqlite must not throw these out as near-copies of each other.
_LOSS_FREE = {"synth", "close", "ask"}
WEIGHTS = {(src, rank): weight * _RANK_FACTOR[rank]
           for src, weight in SET_WEIGHT.items()
           for rank in _RANK_FACTOR}


def _length_factor(trained: int) -> float:
    """A page the generator can finish is the one to imitate.

    Generation stops at a few thousand tokens. A 5,000-token page teaches the
    model to keep writing. Short pages stay at full rate. Long ones stay in
    the mix, at a lower rate, so the style is seen and not memorized.
    `trained` is the tokens that carry loss (the page, not the request).
    """
    if trained <= 2000:
        return 1.0
    if trained <= 4000:
        return 0.45
    return 0.12


def _state(**kw) -> None:
    cur = {}
    if STATE.exists():
        try:
            cur = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur.update(kw)
    cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE.write_text(json.dumps(cur))


def _quality() -> dict[str, float]:
    """norm hash -> multiplier from quality_worker (0 = drop). Empty if absent."""
    if not QUALITY.exists():
        return {}
    con = sqlite3.connect(f"file:{QUALITY}?mode=ro", uri=True, timeout=30)
    try:
        return dict(con.execute("select norm, weight from quality"))
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


# Each reload used to memmap every shard and leave the previous maps open.
# Raise the soft limit, keep one set of maps, and close that set before the next reload.
# If a machine cannot hold one map per shard, only a few stay open.
_MAX_OPEN_SHARDS = 32


def _fd_budget() -> int:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = resource.RLIM_INFINITY
    target = 65536 if hard == inf else min(int(hard), 65536)
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    # numpy memmap keeps one fd per shard. Leave room for the model and logs.
    return max(_MAX_OPEN_SHARDS, (int(soft) - 1024) // 2)


def _close_memmap(mm: np.memmap) -> None:
    mapped = getattr(mm, "_mmap", None)
    if mapped is not None:
        mapped.close()


class Blocks:
    """Weighted, deduplicated, shuffled stream of packed token blocks."""

    def __init__(self, block: int, seed: int):
        self.block = block
        q = _quality()
        self.paths: list[Path] = []
        self._open: dict[int, np.memmap] = {}
        self._lru: list[int] = []
        docs = []  # (bin index, offset, ntok, weight, skip)
        seen: set[str] = set()
        drawn: dict[str, float] = {}
        loss_drawn: dict[str, float] = {}
        # Each set keeps tokens in sets/<name>/tokens. shards/ is the old mixed folder.
        folders = []
        if SHARDS.is_dir():
            folders.append(SHARDS)
        sets = ROOT / "sets"
        if sets.is_dir():
            for folder in sorted(sets.iterdir()):
                tokens = folder / "tokens"
                if tokens.is_dir():
                    folders.append(tokens)
        for folder in folders:
            for meta in sorted(folder.glob("*.jsonl")):
                b = meta.with_suffix(".bin")
                if not b.exists() or b.stat().st_size == 0:
                    continue
                bi = len(self.paths)
                self.paths.append(b)
                for line in meta.read_text().splitlines():
                    d = json.loads(line)
                    if d["norm"] in seen:
                        continue
                    seen.add(d["norm"])
                    src = d.get("src") or ""
                    # Missing skip: do not train the leading BOS. It only separates docs.
                    skip = int(d.get("skip") or 1)
                    skip = max(1, min(skip, int(d["ntok"]) - 1))
                    qmul = 1.0 if src in _LOSS_FREE else q.get(d["norm"], 1.0)
                    trained = int(d["ntok"]) - skip
                    w = WEIGHTS.get((src, int(d["rank"])), 1.0) * qmul * _length_factor(trained)
                    if w > 0:
                        docs.append((bi, int(d["offset"]), int(d["ntok"]), w, skip))
                        drawn[src] = drawn.get(src, 0.0) + d["ntok"] * w
                        loss_drawn[src] = loss_drawn.get(src, 0.0) + trained * w
        rng = np.random.default_rng(seed)
        order = []
        for i, (_, _, _, w, _) in enumerate(docs):
            k = int(w) + (1 if rng.random() < w - int(w) else 0)
            order.extend([i] * k)
        rng.shuffle(order)
        self.docs, self.order = docs, order
        self.tokens = sum(docs[i][2] for i in order)
        self.mix, self.loss_mix = drawn, loss_drawn
        self.pos = 0
        self.buf = np.zeros(0, np.uint32)
        # 1 = this token is a trained target. 0 = request, or the doc-separator BOS.
        self.mbuf = np.zeros(0, np.float32)
        self._cap = _fd_budget()
        # Open every shard when the limit allows. A miss opens and closes a file
        # on almost every document, which is slower than holding the maps.
        if len(self.paths) <= self._cap:
            try:
                for bi, path in enumerate(self.paths):
                    self._open[bi] = np.memmap(path, dtype=np.uint32, mode="r")
                    self._lru.append(bi)
            except OSError as exc:
                if exc.errno != 24:
                    raise
                self.close()
                self._cap = _MAX_OPEN_SHARDS

    def _bin(self, bi: int) -> np.memmap:
        mm = self._open.get(bi)
        if mm is not None:
            # All shards are already open. Skip the LRU shuffle; it is O(shards) per doc.
            if len(self._open) < len(self.paths):
                self._lru.remove(bi)
                self._lru.append(bi)
            return mm
        while self._lru and len(self._open) >= self._cap:
            self._drop(self._lru.pop(0))
        mm = np.memmap(self.paths[bi], dtype=np.uint32, mode="r")
        self._open[bi] = mm
        self._lru.append(bi)
        return mm

    def _drop(self, bi: int) -> None:
        mm = self._open.pop(bi, None)
        if mm is not None:
            _close_memmap(mm)

    def close(self) -> None:
        for bi in list(self._open):
            self._drop(bi)
        self._lru.clear()

    def next(self, batch: int) -> mx.array | None:
        need = batch * (self.block + 1)
        while self.buf.size < need:
            if self.pos >= len(self.order):
                return None
            bi, off, n, _, skip = self.docs[self.order[self.pos]]
            self.pos += 1
            self.buf = np.concatenate([self.buf, self._bin(bi)[off:off + n]])
            mark = np.ones(n, np.float32)
            mark[:skip] = 0.0
            self.mbuf = np.concatenate([self.mbuf, mark])
        x, self.buf = self.buf[:need], self.buf[need:]
        w, self.mbuf = self.mbuf[:need], self.mbuf[need:]
        shape = (batch, self.block + 1)
        return mx.array(x.astype(np.int32).reshape(shape)), mx.array(w.reshape(shape))

    def mix_line(self, counts: dict[str, float]) -> str:
        total = sum(counts.values()) or 1.0
        parts = [f"{k}={v / total:.0%}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        return " ".join(parts)


def _lr(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer update")
    # 5e-5 fits a 1B. It wrecked this 2B (loss looked low because the pages were
    # easy to memorize). 1e-5 is the peak for the game run.
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200, help="optimizer updates")
    # 2e9 on this small game set repeats the same pages until they are memorized.
    p.add_argument("--total-tokens", type=float, default=2e8)
    p.add_argument("--fresh", action="store_true",
                   help="Load the base model. Fresh optimizer, counters at 0. Refuses to run if checkpoints/latest exists.")
    p.add_argument("--save-minutes", type=float, default=30)
    p.add_argument("--bench", type=int, default=0, help="time N micro-steps, save nothing")
    args = p.parse_args()
    _fd_budget()
    for d in (CKPT, LOG.parent):
        d.mkdir(parents=True, exist_ok=True)

    latest = CKPT / "latest"
    if args.fresh and (latest / "counters.json").exists() and not args.bench:
        raise SystemExit("refusing --fresh while checkpoints/latest exists; move that folder aside first")
    resume = (latest / "counters.json").exists() and not args.bench and not args.fresh
    model, _ = load(str(latest if resume else BASE))
    model.train()
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    counters = {"update": 0, "micro": 0, "tokens": 0, "seed": 0, "stop_tokens": int(args.total_tokens)}
    if resume:
        counters = json.loads((latest / "counters.json").read_text())
        counters.setdefault("stop_tokens", int(args.total_tokens))
        opt.state = tree_unflatten(list(mx.load(str(latest / "optimizer.safetensors")).items()))

    tokens_per_update = args.batch * args.block * args.accum
    total_updates = int(int(counters["stop_tokens"]) // tokens_per_update)

    def loss_fn(m, x, w):
        # w is 1 on page tokens and 0 on the request. Mean is over the page only,
        # so a long request does not dilute the game tokens.
        tok = nn.losses.cross_entropy(m(x[:, :-1]), x[:, 1:], weights=w[:, 1:], reduction="none")
        denom = mx.maximum(mx.sum(w[:, 1:]), mx.array(1.0, dtype=tok.dtype))
        return mx.sum(tok) / denom

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def micro(x, w):
        return nn.value_and_grad(model, loss_fn)(model, x, w)

    last_snap = [time.time()]

    def save() -> None:
        # latest/ is overwritten file by file (os.replace is atomic), so a crash
        # mid-save leaves the previous file, never a half-written one.
        latest.mkdir(exist_ok=True)
        model.save_weights(str(latest / "model.tmp.safetensors"))
        os.replace(latest / "model.tmp.safetensors", latest / "model.safetensors")
        mx.save_safetensors(str(latest / "optimizer.tmp.safetensors"), dict(tree_flatten(opt.state)))
        os.replace(latest / "optimizer.tmp.safetensors", latest / "optimizer.safetensors")
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json", "chat_template.jinja"):
            if (BASE / f).exists() and not (latest / f).exists():
                shutil.copy2(BASE / f, latest / f)
        (latest / "counters.tmp.json").write_text(json.dumps(counters))
        os.replace(latest / "counters.tmp.json", latest / "counters.json")
        # Dated weights-only copy every 6 h. Nothing deletes these folders.
        if time.time() - last_snap[0] > 6 * 3600:
            snap = ROOT / "snapshots" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            snap.mkdir(parents=True, exist_ok=True)
            for f in latest.iterdir():
                if f.name != "optimizer.safetensors" and f.is_file() and not f.name.endswith(".tmp"):
                    if not (snap / f.name).exists():
                        shutil.copy2(f, snap / f.name)
            last_snap[0] = time.time()

    data = Blocks(args.block, seed=counters["seed"])
    msg = (f"train base={BASE.name} block={args.block} batch={args.batch} accum={args.accum} lr={args.lr} "
           f"docs={len(data.docs)} sampled_tokens={data.tokens/1e9:.2f}B resume={resume} fresh={args.fresh} "
           f"update={counters['update']}/{total_updates} "
           f"mix[{data.mix_line(data.mix)}] loss[{data.mix_line(data.loss_mix)}]")
    print(msg, flush=True)
    with LOG.open("a") as log:
        log.write(msg + "\n")
    _state(state="running", detail=msg, iters_this_slice=total_updates)

    grads, loss_sum, n_micro = None, 0.0, 0
    t_save = t_rep = time.time()
    tok_rep = 0
    while counters["update"] < total_updates:
        batch = data.next(args.batch)
        if batch is None:  # sampled order used up: reshuffle with a new seed
            data.close()
            counters["seed"] += 1
            data = Blocks(args.block, seed=counters["seed"])
            continue
        x, w = batch
        loss, g = micro(x, w)
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(loss, grads)
        loss_sum += loss.item()
        n_micro += 1
        counters["micro"] += 1
        counters["tokens"] += x.size - args.batch
        tok_rep += x.size - args.batch
        if args.bench and counters["micro"] >= args.bench:
            dt = time.time() - t_rep
            print(f"bench micro={counters['micro']} tok/s={tok_rep/dt:.0f} "
                  f"peak={mx.get_peak_memory()/1e9:.1f}GB", flush=True)
            data.close()
            return
        if n_micro < args.accum:
            continue
        opt.learning_rate = _lr(counters["update"], total_updates, args.lr, args.warmup)
        opt.update(model, tree_map(lambda a: a / args.accum, grads))
        mx.eval(model.parameters(), opt.state)
        counters["update"] += 1
        avg = loss_sum / n_micro
        grads, loss_sum, n_micro = None, 0.0, 0
        now = time.time()
        if counters["update"] % 10 == 0:
            dt = now - t_rep
            line = (f"Iter {counters['update']}: Train loss {avg:.4f}, Learning Rate "
                    f"{float(opt.learning_rate):.3e}, It/sec {10/dt:.3f}, Tokens/sec "
                    f"{tok_rep/dt:.1f}, Trained Tokens {counters['tokens']}, Peak mem "
                    f"{mx.get_peak_memory()/1e9:.3f} GB")
            print(line, flush=True)
            with LOG.open("a") as log:
                log.write(line + "\n")
            _state(step=counters["update"], loss=avg, it_per_sec=10 / dt,
                   tok_per_sec=tok_rep / dt, tokens=counters["tokens"],
                   peak_gb=mx.get_peak_memory() / 1e9)
            t_rep, tok_rep = now, 0
        if now - t_save > args.save_minutes * 60:
            save()
            resume = True
            t_save = time.time()
            # New shards and quality scores join here; order restarts with a new seed.
            # Close the previous maps first. Leaving them open hit "too many open files".
            data.close()
            counters["seed"] += 1
            data = Blocks(args.block, seed=counters["seed"])
            t_rep, tok_rep = time.time(), 0
    data.close()
    save()
    _state(state="done")


if __name__ == "__main__":
    main()
