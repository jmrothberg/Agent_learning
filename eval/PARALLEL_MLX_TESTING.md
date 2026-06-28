# Parallel MLX testing — agent handoff

**Share this file** with another coding agent when you want parallel game/eval runs
using **one MLX model load** and server-side batching. Repo:
`/Users/jonathanrothberg/Agent_learning` (or clone of `jmrothberg/Agent_learning`).

---

## Primary tuning workflow (serial + VLM + visible browser)

**Use this for agent/memory tuning — not parallel mlx-server batch.**

| Requirement | Command / setting |
|-------------|-------------------|
| In-process MLX (VLM works) | `MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8` + `--backend mlx` — **do not** set `MLX_SERVER_URL` |
| VLM critique ON | TUI: `/vlm-critique on` · CLI: `coder.py --vlm-critique` |
| Visible Chromium | **No `--headless`** — run in Terminal.app |
| Serial 10-game eval | [`eval/tune_serial_loop.py`](tune_serial_loop.py) + [`eval/tune_serial10_goals.txt`](tune_serial10_goals.txt) |
| Unattended default | No pause between games · no job wall timeout · `--no-auto-step` on child (no `/wait` latch) |
| Crash recovery | `--resume` (default ON) skips delivered games · `--retries 2` per game · `tune_checkpoint.json` |
| Overnight watchdog | `nohup eval/tune_serial_overnight.sh &` — restarts loop until 10/10 in checkpoint |
| After each game | Optional `--pause-between-games` for triage; else auto-advance. Triage `failure_class`: `harness_bug` → code; `memory_gap` → JSONL |

```bash
cd /Users/jonathanrothberg/Agent_learning
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/tune_serial_loop.py \
  --goals-file eval/tune_serial10_goals.txt \
  --out-dir games/tune_serial10/run_01
```

Interactive single game (preferred while iterating):

```bash
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python chat.py
# paste goal · /vlm-critique on · watch Chromium
```

**Why not mlx-server for tuning:** `MLXServerBackend.is_vlm()` is false — images are stripped, structured visual playtests never run (`vlm_critique: false` in batch traces even when the model is VLM-capable).

### Round 1 triage (`tune_round1_r4`) — general fixes only

| Trace | failure_class | General fix (already landed) |
|-------|---------------|------------------------------|
| Street Fighter | `none` (4×) | VLM was OFF in batch — use serial + `--vlm-critique` |
| Asteroids | `none` (6×) | Same; wireframe goals skip `<assets>` via plan nudge |
| Galaga | `memory_gap` (1×) + `none` | Extended `canvas-top-down-action` Q7–Q9 + auto-probes; playbook `top-down-sprite-draw-orientation` |
| Snake | `none` | No harness change |
| Holochess | (incomplete trace) | Animation nudge already in plan_nudges |

**Do not** add per-game Python branches. Repeated `memory_gap` on undrawn assets → playbook/outline retrieval (genre-free).

### failure_class triage (serial loop)

| Tag | Action |
|-----|--------|
| `harness_bug` | Patch `agent.py` / `tools.py` / `backend.py` (one surgical fix) |
| `memory_gap` | Edit `memory/*.jsonl` (playbook, visual_playtests, plan_nudges) |
| `local_llm_limit` | Usually no agent change; note for prompt budget |
| `none` | Ship or iterate on probes/VLM |

Parallel + headless batch below remains useful for **throughput experiments** and CI smoke only.

---

## Goal (parallel throughput lab)

Run **2–5 test jobs at once** (full game builds or seed-edit harness scenarios)
without loading the MLX model separately in every Python process.

---

## Wrong vs right

| Approach | What happens | Use for parallel? |
|----------|--------------|-------------------|
| **Wrong:** N × `coder.py` with default `--backend mlx` | N in-process `mlx_lm.load()` copies (~15 GB each for 27B) | No — OOM on most Macs |
| **Wrong:** `--best-of-n 3` on MLX | Sequential samples within **one** game | No — not multi-game |
| **Right:** N clients → **one** `mlx_lm.server` | One model in VRAM; server batches concurrent streams | **Yes** |

---

## Regular chat is unchanged

**`chat.py` / interactive TUI still uses in-process MLX by default.**

Server mode activates **only** when one of these is set:

- `MLX_SERVER_URL=http://127.0.0.1:8080`
- `MLX_HOST=127.0.0.1:8080` (legacy alias)
- `LLM_BACKEND=mlx-server`
- `--backend mlx-server` on CLI

Do **not** put `MLX_SERVER_URL` in `.env` or shell profile unless you want the TUI
to use the server too. For batch runs, set it **only in the batch terminal** or
let `eval/batch_parallel.py` set it for child processes.

---

## Architecture

```
Terminal 1                          Terminal 2 (batch runner)
┌─────────────────────┐            ┌──────────────────────────────┐
│ mlx_lm.server       │            │ eval/batch_parallel.py       │
│  (one model load)   │◄──HTTP─────│  spawns N subprocesses:    │
│  continuous batch   │            │   coder.py --backend mlx-server
└─────────────────────┘            │   or eval_seed_edits.py …  │
                                   └──────────────────────────────┘
```

Implementation (already in repo):

- [`backend.py`](../backend.py) — `MLXServerBackend` (HTTP SSE to `/v1/chat/completions`);
  `MLXBackend` remains in-process default.
- [`eval/batch_parallel.py`](batch_parallel.py) — orchestrator: `--jobs N`, goals file,
  `--seed-edits`, summary JSON.
- [`coder.py`](../coder.py) — accepts `--backend mlx-server`.
- Detection: `_mlx_server_mode_requested()` in `backend.py`.

Optional stronger batching: **vllm-mlx** with `--continuous-batching` on the same
OpenAI-compatible port — same client path, not a repo dependency today.

---

## Runbook (copy-paste)

### 1. Start the server (once, separate terminal)

```bash
cd /Users/jonathanrothberg/Agent_learning
.venv/bin/mlx_lm.server \
  --model ~/MLX_Models/Qwen3.6-27B-mxfp8 \
  --port 8080
```

On big models + long context, see `scripts/setup.sh` macOS hint for
`iogpu.wired_limit_mb` if the server dies silently after prefill.

Verify:

```bash
curl -s http://127.0.0.1:8080/v1/models | head
```

### 2. Parallel full game builds (2 at a time)

```bash
cd /Users/jonathanrothberg/Agent_learning
MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \
  --jobs 2 \
  --goal "snake with wraparound board" \
  --goal "breakout with paddle and bricks" \
  --headless \
  --max-iters 4
```

Artifacts: `games/batch_parallel/` + `batch_summary.json`.

### 3. Parallel seed-edit harness (5 scenarios, lighter — no browser)

```bash
MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \
  --seed-edits \
  --jobs 5 \
  --patch-only \
  --max-iters 2
```

Scenarios: `bigger_towers`, `recolor_creeps`, `faster_creeps`, `second_path`,
`tower_range_ring` (see `eval/eval_seed_edits.py`).

### 4. Goals from a file

```bash
# my_goals.txt — one goal per line, # comments ok
MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python eval/batch_parallel.py \
  --jobs 3 \
  --goals-file my_goals.txt \
  --headless \
  --max-iters 4
```

### 5. Dry-run (print subprocess commands, no server required)

```bash
.venv/bin/python eval/batch_parallel.py --dry-run \
  --goal "snake" --goal "breakout"
```

### 6. Manual single client (without batch runner)

```bash
MLX_SERVER_URL=http://127.0.0.1:8080 .venv/bin/python coder.py \
  "snake wraparound" \
  --backend mlx-server \
  --headless \
  --out games/batch_parallel/snake.html
```

---

## `batch_parallel.py` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--jobs N` | 1 | Max concurrent subprocesses (use 1 for art-heavy Round 1; 2+ for procedural-only) |
| `--server URL` | env or `http://127.0.0.1:8080` | MLX server base URL |
| `--goal TEXT` | — | Repeatable build goal |
| `--goals-file PATH` | — | One goal per line |
| `--seed-edits` | off | Run `eval_seed_edits.py` scenarios |
| `--names a,b` | all 5 | Subset of seed-edit names |
| `--patch-only` | off | Skip Phase A (seed edits) |
| `--max-iters N` | 4 | Iterations per job |
| `--best-of-n N` | 1 | Passed to `coder.py` |
| `--stall-seconds N` | 600 | Quiet-window budget passed to `coder.py` (MLX server uses activity-based stall) |
| `--job-timeout N` | 7200 | Kill each subprocess after N seconds (prevents infinite hangs) |
| `--skip-preflight` | off | Skip mlx server smoke completion before batch |
| `--headless` | off | Chromium headless for full builds |
| `--model ID` | auto from server | Override model id in requests |
| `--out-dir PATH` | `games/batch_parallel` | Output root |
| `--dry-run` | off | Print commands only |

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `MLX_SERVER_URL` | Switch to HTTP server backend |
| `MLX_HOST` | Same (legacy) |
| `LLM_BACKEND=mlx-server` | Force server backend |
| `MLX_MODEL` | Model id/path sent to server (and server-side load hint) |
| *(unset)* | **In-process MLX** — normal `chat.py` behavior |

---

## What an agent should put in its plan

When planning parallel MLX testing, the plan should:

1. **Start one server** in a background terminal (or assume user started it).
2. **Use `eval/batch_parallel.py`** with `--jobs N`, not N bare `coder.py` with default MLX.
3. **Set `MLX_SERVER_URL` only for the batch session** — do not change default chat behavior.
4. **Pick workload:**
   - Full builds → `--goal` / `--goals-file` + `--headless`
   - Harness regression → `--seed-edits --patch-only`
5. **Read results** from `games/batch_parallel/batch_summary.json`.
6. **Do not** implement N in-process MLX processes as “parallelism.”
7. **Do not** expect `--best-of-n` to parallelize across games on MLX.

Success criteria for seed-edit batch: each scenario **materialized a changed file**
(same as `eval/eval_seed_edits.py` PASS). For full builds: inspect `batch_summary.json`
exit codes and generated HTML under `--out-dir`.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `mlx_lm.server is not reachable` | Server not started or wrong port |
| Prefill completes, zero tokens | Metal OOM or MLX server starved — use `--jobs 1`, raise `--stall-seconds`, or `iogpu.wired_limit_mb` |
| TUI suddenly uses server | `MLX_SERVER_URL` left set in `.env` / shell — unset it |
| Art-heavy builds stall mid-stream | Lower `--jobs` to 1; Round 1 tune batch uses serial jobs + 1200s stall |
| 5 full art-heavy builds OOM | Lower `--jobs` or use `--seed-edits --patch-only` |
| Slow with 5 long prompts | Expected on unified memory; try vllm-mlx continuous batching |

---

## Related docs

- [`eval/tune_serial_loop.py`](tune_serial_loop.py) — serial VLM tuning runner (primary)
- [`TEST.md`](../TEST.md) — short parallel section in Layer 2b
- [`CLAUDE.md`](../CLAUDE.md) — `MLX_SERVER_URL` env var
- [`backend.py`](../backend.py) — `MLXServerBackend`, `MLXBackend`, `detect_backend`
