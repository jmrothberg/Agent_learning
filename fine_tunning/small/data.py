"""HTML/JS pretraining shards for the small model (MiniCPM5-1B-Base).

Sources:
  own  — games indexed in ~/MLX_Models/html_game_sft/games.sqlite (kind html/gold)
  gcc  — codeparrot/github-code-clean, languages HTML + JavaScript, streamed
         over HTTP (parquet files are read, never stored on disk)

Output: ~/MLX_Models/html_js_small/shards/<src>_<chunk>.bin   uint32 tokens
        ~/MLX_Models/html_js_small/shards/<src>_<chunk>.jsonl one line per doc
        (offset, ntok, sha, norm hash, source, path, rank). A chunk is done
        when its .jsonl exists, so a rerun resumes.

  python data.py --source own
  python data.py --source gcc --workers 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SMALL_ROOT", "~/MLX_Models/html_js_small")).expanduser()
BASE = Path(os.environ.get("SMALL_BASE", "~/MLX_Models/MiniCPM5-1B-Base")).expanduser()
GAMES_DB = Path("~/MLX_Models/html_game_sft/games.sqlite").expanduser()
SHARDS = ROOT / "shards"

def _special_ids() -> tuple[int, int]:
    """Doc separators from the base model's config (MiniCPM5: bos 0, eos 1).
    Models without a BOS (e.g. Qwen) use EOS on both sides."""
    try:
        cfg = json.loads((BASE / "config.json").read_text())
    except (OSError, ValueError):
        return 0, 1
    eos = cfg.get("eos_token_id", 1)
    eos = eos[0] if isinstance(eos, list) else eos
    bos = cfg.get("bos_token_id")
    return (eos if bos is None else bos), eos


BOS, EOS = _special_ids()
MIN_BYTES, MAX_BYTES = 200, 200_000

_B64 = re.compile(r"[A-Za-z0-9+/=]{200,}")
_WS = re.compile(r"\s+")

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    return _tok


def quality(text: str, is_html: bool) -> str | None:
    """Return a drop reason, or None to keep."""
    n = len(text)
    if n < MIN_BYTES or n > MAX_BYTES:
        return "size"
    lines = text.splitlines() or [text]
    if n / len(lines) > 200:
        return "minified"
    if sum(len(l) for l in lines if len(l) > 1000) > 0.3 * n:
        return "minified"
    if sum(c.isalnum() for c in text) < 0.25 * n:
        return "symbols"
    if sum(len(m) for m in _B64.findall(text)) > 0.2 * n:
        return "base64"
    if is_html and "<script" not in text.lower():
        return "no_script"
    return None


def rank(text: str) -> int:
    """2 = canvas/three.js game with a loop, 1 = has canvas or a loop, 0 = other."""
    low = text.lower()
    canvas = "<canvas" in low or "getcontext" in low
    loop = "requestanimationframe" in low or "setinterval" in low
    three = "three.js" in low or "three.min.js" in low or "new three." in low
    if (canvas and loop) or three:
        return 2
    return 1 if (canvas or loop) else 0


def _write(name: str, docs: list[tuple[str, dict]]) -> dict:
    tok = _tokenizer()
    enc = tok.encode_batch([t for t, _ in docs], add_special_tokens=False)
    parts, off = [], 0
    meta_lines = []
    for (text, meta), e in zip(docs, enc):
        ids = np.asarray([BOS] + e.ids + [EOS], dtype=np.uint32)
        parts.append(ids)
        meta.update(offset=off, ntok=int(ids.size))
        # retok keeps the original norm, so quality.sqlite scores still match.
        meta.setdefault("norm", hashlib.sha1(_WS.sub(" ", text).encode()).hexdigest()[:16])
        meta_lines.append(json.dumps(meta))
        off += ids.size
    arr = np.concatenate(parts) if parts else np.zeros(0, np.uint32)
    arr.tofile(SHARDS / f"{name}.bin")
    tmp = SHARDS / f"{name}.jsonl.tmp"
    tmp.write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""))
    tmp.rename(SHARDS / f"{name}.jsonl")
    return {"docs": len(docs), "tokens": int(off)}


def _own_chunk(job: tuple[int, list[tuple[str, str]]]) -> dict:
    idx, rows = job
    name = f"own_{idx:05d}"
    drops: dict[str, int] = {}
    docs = []
    for sha, path in rows:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            drops["missing"] = drops.get("missing", 0) + 1
            continue
        why = quality(text, is_html=True)
        if why:
            drops[why] = drops.get(why, 0) + 1
            continue
        docs.append((text, {"sha": sha, "src": "own", "path": path, "rank": rank(text)}))
    out = _write(name, docs)
    out["drops"] = drops
    return out


def _gcc_chunk(job: tuple[int, str]) -> dict:
    idx, file = job
    name = f"gcc_{idx:05d}"
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    drops: dict[str, int] = {}
    docs = []
    fs = HfFileSystem()
    with fs.open(f"datasets/codeparrot/github-code-clean/{file}", "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["code", "language", "path", "repo_name"], batch_size=4096):
            d = batch.to_pydict()
            for code, lang, path, repo in zip(d["code"], d["language"], d["path"], d["repo_name"]):
                if lang not in ("HTML", "JavaScript"):
                    continue
                why = quality(code, is_html=(lang == "HTML"))
                if why:
                    drops[why] = drops.get(why, 0) + 1
                    continue
                sha = hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()
                docs.append((code, {"sha": sha, "src": "gcc", "path": f"{repo}/{path}",
                                    "lang": lang, "rank": rank(code)}))
    out = _write(name, docs)
    out["drops"] = drops
    return out


def _retok_chunk(job: tuple[str, str]) -> dict:
    """Decode one shard with the tokenizer saved beside it, re-encode for BASE."""
    from tokenizers import Tokenizer
    src_shards, name = Path(job[0]), job[1]
    old = Tokenizer.from_file(str(src_shards / "tokenizer.json"))
    ids = np.memmap(src_shards / f"{name}.bin", dtype=np.uint32, mode="r")
    docs = []
    for line in (src_shards / f"{name}.jsonl").read_text().splitlines():
        d = json.loads(line)
        text = old.decode(ids[d["offset"] + 1: d["offset"] + d["ntok"] - 1].tolist())
        docs.append((text, {k: v for k, v in d.items() if k not in ("offset", "ntok")}))
    out = _write(name, docs)
    out["drops"] = {}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("own", "gcc", "retok"), required=True)
    p.add_argument("--from", dest="from_root", default="",
                   help="retok: another SMALL_ROOT whose shards/ (+ shards/tokenizer.json) to re-tokenize for SMALL_BASE")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--chunk", type=int, default=2000, help="own: files per shard")
    p.add_argument("--max-chunks", type=int, default=0, help="gcc: stop after this many parquet files (0 = all)")
    args = p.parse_args()
    SHARDS.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    # Shards are token ids of this tokenizer; keep a copy beside them so retok can decode them later.
    if not (SHARDS / "tokenizer.json").exists():
        shutil.copy2(BASE / "tokenizer.json", SHARDS / "tokenizer.json")
    print(f"base={BASE} bos={BOS} eos={EOS}", flush=True)

    if args.source == "retok":
        src = Path(args.from_root).expanduser() / "shards"
        if src.resolve() == SHARDS.resolve():
            raise SystemExit("--from must be a different SMALL_ROOT")
        jobs = [(str(src), m.stem) for m in sorted(src.glob("*.jsonl"))]
        fn = _retok_chunk
        jobs = [j for j in jobs if not (SHARDS / f"{j[1]}.jsonl").exists()]
        print(f"retok: {len(jobs)} shards from {src}", flush=True)
    elif args.source == "own":
        con = sqlite3.connect(f"file:{GAMES_DB}?mode=ro", uri=True, timeout=30)
        rows = con.execute(
            "select sha, path from games where kind in ('html','gold') and bytes between ? and ? order by sha",
            (MIN_BYTES, MAX_BYTES)).fetchall()
        con.close()
        jobs = [(i, rows[s:s + args.chunk]) for i, s in enumerate(range(0, len(rows), args.chunk))]
        fn = _own_chunk
    else:
        from huggingface_hub import HfApi
        files = sorted(f for f in HfApi().list_repo_files("codeparrot/github-code-clean", repo_type="dataset")
                       if f.endswith(".parquet"))
        jobs = list(enumerate(files))
        if args.max_chunks:
            jobs = jobs[: args.max_chunks]
        fn = _gcc_chunk
    if args.source != "retok":
        jobs = [j for j in jobs if not (SHARDS / f"{args.source}_{j[0]:05d}.jsonl").exists()]
        print(f"{args.source}: {len(jobs)} chunks to do", flush=True)

    total = {"docs": 0, "tokens": 0, "drops": {}}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            total["docs"] += r["docs"]
            total["tokens"] += r["tokens"]
            for k, v in r["drops"].items():
                total["drops"][k] = total["drops"].get(k, 0) + v
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chunks docs={total['docs']} tokens={total['tokens']/1e9:.2f}B "
                      f"drops={total['drops']} {time.time()-t0:.0f}s", flush=True)
    (ROOT / "logs" / f"data_{args.source}.json").write_text(json.dumps(total))


if __name__ == "__main__":
    main()
