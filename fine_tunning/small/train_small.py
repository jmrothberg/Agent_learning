"""Stage-1 continued pretraining of the small HTML/JS model (full fine-tune, MLX).

Reads the shards written by data.py, packs them into fixed blocks (no padding,
loss on every token), and trains every weight of MiniCPM5-1B-Base.

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

ROOT = Path(os.environ.get("SMALL_ROOT", "~/MLX_Models/html_js_small")).expanduser()
BASE = Path(os.environ.get("SMALL_BASE", "~/MLX_Models/MiniCPM5-1B-Base")).expanduser()
SHARDS = ROOT / "shards"
CKPT = ROOT / "checkpoints"
LOG = ROOT / "logs" / "train.log"
STATE = ROOT / "logs" / "state.json"
QUALITY = ROOT / "quality.sqlite"

# Sampling weight per (source, rank). Own games are the target domain, so they
# repeat more; rank 2 = canvas/three.js game with a loop.
WEIGHTS = {("own", 2): 3.0, ("own", 1): 2.0, ("own", 0): 1.0,
           ("gcc", 2): 1.5, ("gcc", 1): 1.0, ("gcc", 0): 0.7,
           ("stack", 2): 1.5, ("stack", 1): 1.0, ("stack", 0): 0.7}


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


class Blocks:
    """Weighted, deduplicated, shuffled stream of packed token blocks."""

    def __init__(self, block: int, seed: int):
        self.block = block
        q = _quality()
        self.bins: list[np.memmap] = []
        docs = []  # (bin index, offset, ntok, weight)
        seen: set[str] = set()
        for meta in sorted(SHARDS.glob("*.jsonl")):
            b = meta.with_suffix(".bin")
            if not b.exists() or b.stat().st_size == 0:
                continue
            bi = len(self.bins)
            self.bins.append(np.memmap(b, dtype=np.uint32, mode="r"))
            for line in meta.read_text().splitlines():
                d = json.loads(line)
                if d["norm"] in seen:
                    continue
                seen.add(d["norm"])
                w = WEIGHTS.get((d["src"], d["rank"]), 1.0) * q.get(d["norm"], 1.0)
                if w > 0:
                    docs.append((bi, d["offset"], d["ntok"], w))
        rng = np.random.default_rng(seed)
        order = []
        for i, (_, _, _, w) in enumerate(docs):
            k = int(w) + (1 if rng.random() < w - int(w) else 0)
            order.extend([i] * k)
        rng.shuffle(order)
        self.docs, self.order = docs, order
        self.tokens = sum(docs[i][2] for i in order)
        self.pos, self.buf = 0, np.zeros(0, np.uint32)

    def next(self, batch: int) -> mx.array | None:
        need = batch * (self.block + 1)
        while self.buf.size < need:
            if self.pos >= len(self.order):
                return None
            bi, off, n, _ = self.docs[self.order[self.pos]]
            self.pos += 1
            self.buf = np.concatenate([self.buf, self.bins[bi][off:off + n]])
        x, self.buf = self.buf[:need], self.buf[need:]
        return mx.array(x.astype(np.int32).reshape(batch, self.block + 1))


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
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=200, help="optimizer updates")
    p.add_argument("--total-tokens", type=float, default=2e9)
    p.add_argument("--save-minutes", type=float, default=30)
    p.add_argument("--bench", type=int, default=0, help="time N micro-steps, save nothing")
    args = p.parse_args()
    for d in (CKPT, LOG.parent):
        d.mkdir(parents=True, exist_ok=True)

    latest = CKPT / "latest"
    resume = (latest / "counters.json").exists() and not args.bench
    model, _ = load(str(latest if resume else BASE))
    model.train()
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    counters = {"update": 0, "micro": 0, "tokens": 0, "seed": 0}
    if resume:
        counters = json.loads((latest / "counters.json").read_text())
        opt.state = tree_unflatten(list(mx.load(str(latest / "optimizer.safetensors")).items()))

    tokens_per_update = args.batch * args.block * args.accum
    total_updates = int(args.total_tokens // tokens_per_update)

    def loss_fn(m, x):
        return nn.losses.cross_entropy(m(x[:, :-1]), x[:, 1:]).mean()

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def micro(x):
        return nn.value_and_grad(model, loss_fn)(model, x)

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
           f"docs={len(data.docs)} sampled_tokens={data.tokens/1e9:.2f}B resume={resume} "
           f"update={counters['update']}/{total_updates}")
    print(msg, flush=True)
    with LOG.open("a") as log:
        log.write(msg + "\n")
    _state(state="running", detail=msg, iters_this_slice=total_updates)

    grads, loss_sum, n_micro = None, 0.0, 0
    t_save = t_rep = time.time()
    tok_rep = 0
    while counters["update"] < total_updates:
        x = data.next(args.batch)
        if x is None:  # sampled order used up: reshuffle with a new seed
            counters["seed"] += 1
            data = Blocks(args.block, seed=counters["seed"])
            continue
        loss, g = micro(x)
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
            counters["seed"] += 1
            data = Blocks(args.block, seed=counters["seed"])
            t_rep, tok_rep = time.time(), 0
    save()
    _state(state="done")


if __name__ == "__main__":
    main()
