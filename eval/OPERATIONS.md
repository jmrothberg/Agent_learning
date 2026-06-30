# eval/OPERATIONS.md — run tests and batches (LLM entry point)

**Start here** when the user asks to run tests, triage a tune batch, or kick off N games.
Human onboarding → `README.md`. Commands/env → `DEV.md`. Harness traps → `FOR_NEXT_LLM.md`.

---

## If the user says… → run this

| User intent | Command | Notes |
|-------------|---------|-------|
| **Run 10 games overnight (run_08 — tonight)** | Terminal: `bash eval/tune_run08.sh` · Cursor: watcher below | Batch runs **flat-out** (no pause). Watcher fixes in parallel. |
| **Run all 11 games overnight (both batches, auto-chained)** | `bash eval/tune_run07_chain.sh` in Terminal + monitor below in Cursor | **One paste** — Batch B starts automatically when A finishes. No wake-up. |
| **Run 11 games to improve the agent (run_07)** | Same as chain row above | A=GLM no VLM (6) → B=Qwen VLM on (5), watcher handoff between games. |
| **Run unit tests** / **pytest** / **after a code change** | `.venv/bin/python -m pytest tests/ -q` | ~2158 tests, no GPU. Full map: `TEST.md`. |
| **Run one test file** | `.venv/bin/python -m pytest tests/test_patches.py -v` | Swap path. |
| **Run asteroids regression** | `.venv/bin/python -m pytest tests/test_retrieval.py tests/test_patches.py -q -k asteroids` | Ship thrust + irregular asteroids. |
| **Run 3D navigation guards** | `.venv/bin/python -m pytest tests/test_3d_navigation_conventions.py tests/test_doom_trace_fixes.py -q` | Skeletons, playbook, FPS yaw/movement conventions. |
| **Run run_06 guards** | `.venv/bin/python -m pytest tests/test_run06_draw_contract.py tests/test_tune_serial_pass.py tests/test_stream_instance_method.py tests/test_grid_maze_chase_probes.py -q` | drawImage contract + honest batch PASS + get_backend fix + grid-maze probes. |
| **Prompt library coverage (no model)** | `.venv/bin/python eval/eval_prompts_plan.py --coverage` | Instant; CI runs this. |
| **Plan eval (one model turn per prompt)** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_prompts_plan.py` | No browser. |
| **Seed-edit eval** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py` | Materialization only (`browser=None`). |
| **One game headless** | `.venv/bin/python coder.py "snake" --max-iters 4 --headless` | Single session; trace under `games/traces/`. |
| **Interactive TUI** | `.venv/bin/python chat.py` | Visible Chromium; `/bestof off` default. |
| **System smoke (browser)** | `python system_tests.py run --suite smoke --three-model` | Slow; confirms full loop. |
| **Timeline a trace** | `.venv/bin/python scripts/enrich_trace.py <path-or-stem> --timeline` | Primary triage; see `HARNESS_DEBUG.md`. |
| **Batch dashboard / watcher (run_07 chain)** | `.venv/bin/python eval/tune_overnight_monitor.py --run07-chain --interval 30 --sync-loop` | Polls every **30 seconds** (not minutes). Triage + patch while batch keeps running. |
| **Parallel N games (throughput lab)** | See `eval/PARALLEL_MLX_TESTING.md` + `eval/batch_parallel.py` | One `mlx_lm.server`, N clients — **not** in-game BoN. |

---

## run_07 — both batches, one night (A → B auto-chained)

**One Terminal paste runs all 11 games back-to-back.** Batch B starts automatically when A finishes. **No pause between games** (`--wait-for-monitor 0` default). Cursor watcher runs in parallel and patches harness/memory while games continue — fixes apply to the next game(s).

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run07_chain.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --run07-chain --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_07/chain.log` | `games/tune_serial10/run_07/agent_monitor.json` |

```bash
cd /Users/jonathanrothberg/Agent_learning
bash eval/tune_run07_chain.sh
```

Cursor watcher (parallel — triage traces, patch code/memory, **do not stop the batch**):

```bash
.venv/bin/python eval/tune_overnight_monitor.py --run07-chain --interval 30 --sync-loop
```

When a game finishes, read `agent_monitor.json` / the newest trace under `traces/`, classify `failure_class`, patch source, commit if needed. **No Enter, no `inter_game_ready` release** — default batch does not block between games.

Optional blocking handoff (only if you explicitly set `TUNE_WAIT_FOR_MONITOR=1800` on the Terminal batch): after fixes, release with:

```bash
.venv/bin/python eval/tune_inter_game_ready.py \
  --out-dir games/tune_serial10/run_07_big \
  --note "what you fixed"
```

Use `--out-dir` from `active_out_dir` in `agent_monitor.json` (`run_07_big` during Batch A, `run_07_vlm` during Batch B).

| | Batch A | Batch B |
|---|---------|---------|
| Dir | `run_07_big/` | `run_07_vlm/` |
| Model | GLM-5.2-MLX-4bit | Qwen3.6-27B-mxfp8 |
| VLM | off (`--no-vlm-critique`) | **on** (default) |
| Games | 6 | 5 |

**Success criteria:** `fresh_pass` with `iter_summaries > 0` per game — not checkpoint-only complete.

---

## run_08 — tonight (10 games, flat-out + parallel watcher)

Fresh library goals not in run_07, plus **Doom slot 1** to validate the 3D FPS navigation harness fix. Goals: `eval/tune_run08_goals.txt`.

| Slot | Library | Modality |
|------|---------|----------|
| 1 | doom | 3D FPS (repeat — nav validation) |
| 2 | minecraft | 3D voxel |
| 3 | space-invaders | top-down shooter |
| 4 | dig-dug | grid carve |
| 5 | bomberman | grid bombs |
| 6 | outrun | Mode-7 racing + video |
| 7 | zelda | overworld RPG + video |
| 8 | monkey-island | point-and-click + video |
| 9 | torch-dungeon | lit maze / fog |
| 10 | bullet-hell-boss | bullet patterns |

**Two processes — both required.** Terminal batch runs **game → game with zero wait**. Cursor watcher polls every **30 seconds** (not 30 minutes), triages finished traces, and patches harness/memory/prompts **while the batch keeps going**. No Enter, no blocking, no manual release.

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run08.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_08 --jobs-total 10 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_08/overnight.log` | `games/tune_serial10/run_08/agent_monitor.json` |

```bash
cd /Users/jonathanrothberg/Agent_learning
bash eval/tune_run08.sh
```

```bash
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_08 \
  --jobs-total 10 \
  --interval 30 \
  --sync-loop
```

**Watcher loop (continuous, no blocking the batch):**

1. Poll `agent_monitor.json` — when `completed_count` advances, open the newest trace for that label.
2. **Timeline:** `.venv/bin/python scripts/enrich_trace.py <trace> --timeline`
3. **Classify** `failure_class` → patch `tools.py` / `agent_*.py` / `memory/*.jsonl` / `prompts_v1.py`
4. **Keep going** — next game already running or starts immediately; fixes apply to subsequent games.

Artifacts: `games/tune_serial10/run_08/` (`overnight.log`, `traces/`, `tune_checkpoint.json`).

---

## Serial overnight batch (manual launch)

**Primary workflow:** see **run_07 chain** above. For a custom dir/goals file:

| File | Purpose |
|------|---------|
| **`eval/tune_run08_goals.txt`** | **run_08 tonight (10 games)** — see § run_08 above |
| `eval/tune_run07_big.txt` | run_07 Batch A (6 games, GLM, no VLM) |
| `eval/tune_run07_vlm.txt` | run_07 Batch B (5 games, Qwen + VLM, `--max-iters 2`) |
| `eval/tune_serial10_goals.txt` | Full 12-game battery |
| `eval/tune_serial10_round2_goals.txt` | Round 2 subset |
| `eval/tune_serial10_round2_rerun.txt` | run_06 validation (6 games) → e.g. `run_06` or `run_07` |

**Example launch** (Terminal.app, not Cursor):

```bash
cd /Users/jonathanrothberg/Agent_learning
mkdir -p games/tune_serial10/run_06
caffeinate -dims env \
  TUNE_OUT_DIR=games/tune_serial10/run_06 \
  TUNE_GOALS_FILE=eval/tune_serial10_round2_rerun.txt \
  MLX_MODEL="$HOME/MLX_Models/GLM-5.2-MLX-4bit" \
  nohup bash eval/tune_serial_overnight.sh &
tail -f games/tune_serial10/run_06/overnight.log
```

**Defaults baked in:** `--best-of-n 1`, stuck best-of-2 **off**, `--no-vlm-critique`, `--resume`, `--retries 2`, **`--wait-for-monitor 0`** (flat-out — no pause between games).

**Change game count:** edit the goals `.txt` (one goal per non-comment line) or pass `--goal "..."` repeatedly to `eval/tune_serial_loop.py`.

**Change max iters / model:**

```bash
.venv/bin/python eval/tune_serial_loop.py \
  --goals-file eval/tune_serial10_round2_rerun.txt \
  --out-dir games/tune_serial10/run_07 \
  --model "$HOME/MLX_Models/GLM-5.2-MLX-4bit" \
  --max-iters 6 \
  --best-of-n 1 \
  --no-vlm-critique \
  --resume
```

---

## Where artifacts live (run_06 example)

```
games/tune_serial10/run_06/
  overnight.log              # full stdout — tail this while batch runs
  overnight.pid              # watchdog pid
  agent_monitor.json         # optional condensed dashboard (poller)
  tune_checkpoint.json       # resume state — completed_labels
  tune_summary.json          # live summary while running
  01_build_a_donkey_kong_game__single.html
  01_....best.html           # best passing snapshot
  01_..._assets/             # sprites
  candidates/                # visible BoN scratch (when /bestof on or --best-of-n>1)
    iter_05/cand_0.html
  traces/
    01_...__run_20260629_152359_723418.jsonl   # full session log — triage here
  snapshots/<artifact_id>/iter_02.html
```

| Artifact | Writer | Use |
|----------|--------|-----|
| `traces/*.jsonl` | `GameAgent._trace()` | **Always triage here first** — `enrich_trace.py --timeline` |
| `overnight.log` | `tune_serial_overnight.sh` | Live progress, model tokens, errors |
| `agent_monitor.json` | `eval/tune_overnight_monitor.py` | At-a-glance PASS count + log tail — **not** a substitute for traces |
| `tune_checkpoint.json` | `eval/tune_serial_loop.py` | Resume after crash; `completed_labels` |

**Open a game manually:** open `02_....html` in Chrome (same folder as `_assets/`).

---

## Triage workflow (any failed or slow game)

1. **Timeline:** `.venv/bin/python scripts/enrich_trace.py games/tune_serial10/run_06/traces/02_...jsonl --timeline`
2. **Read `failure_class`** on failed iters (`harness_bug` → Python; `memory_gap` → playbook; `local_llm_limit` → model/prompt).
3. **Play the HTML** — green `TEST OK` can still be wrong (`HARNESS_DEBUG.md` rule #1).
4. **Check BoN:** if log shows `sampling 2 candidates` / `stuck ... escalating`, stuck BoN was on — default is now off; use `/bestof on` only when you want it.
5. **Persist learnings:** copy into `FOR_NEXT_LLM.md` + optional `memory/playbook.jsonl` bullet — not into generated `games/*.html`.

### Targeted pytest after edits

| After editing… | Run |
|----------------|-----|
| `tools.py` gates | `.venv/bin/python -m pytest tests/test_probe_gate.py tests/test_microprobes.py tests/test_static_action_gate.py -q` |
| `patches.py` | `.venv/bin/python -m pytest tests/test_patches.py -q` |
| `agent_feedback.py` | `.venv/bin/python -m pytest tests/test_feedback_router.py tests/test_scoped_feedback.py tests/test_golden_feedback_flows.py -q` |
| `agent_compaction.py` | `.venv/bin/python -m pytest tests/test_compaction.py tests/test_token_aware_compaction.py -q` |
| Mixins / loop | `.venv/bin/python -m pytest tests/test_iter_loop_guards.py tests/test_trace_diagnostics.py -q` |
| Docs only | `.venv/bin/python -m pytest tests/test_doc_links.py tests/test_mixin_map.py -q` |

---

## run_06 snapshot (2026-06-29)

Model: `GLM-5.2-MLX-4bit`. Goals: `eval/tune_serial10_round2_rerun.txt` (6 games).

| Game | Result | Notes |
|------|--------|-------|
| 01 Donkey Kong | PASS | ~51 min; 1× memory_gap |
| 02 Kung-Fu Master | PASS (after retries) | Crouch gated movement → `input_moves_player` fail; cutscene setTimeout → enemies undrawn; stuck BoN burned hours when enabled |
| 03–06 | **fresh_fail** | Instant `get_backend` TypeError (~1 s traces, no `.best.html`) — fixed in repo |

Key harness changes already in repo: stuck BoN default off (`/bestof off`), visible `candidates/iter_NN/`, drawImage contract (`test_run06_draw_contract.py`).

**Batch outcome labels** (since run_06 reporting fix): `tune_serial_loop.py` and `agent_monitor.json` distinguish **checkpoint complete** (all labels recorded) from **fresh pass** (this run’s trace has `iter_summary ok=true` or new `.best.html`). **`artifact_pass`** = resume skip or SIGSEGV-after-ship reconcile when `.best.html` already existed; **`fresh_fail`** = subprocess ran but nothing verified shipped (e.g. run_06 games 03–06 hit `get_backend` TypeError in ~1 s). Do not treat `6/6 checkpoint complete` as `6/6 fresh pass`.

Corrected run_06 score: **2/6 fresh pass** (01 Donkey Kong, 02 Kung-Fu), **4/6 fresh fail** (03–06 instant MLX bug; no `.best.html` in run_06).

---

## Related docs

| Doc | When |
|-----|------|
| `TEST.md` | What each pytest file guards |
| `HARNESS_DEBUG.md` | Gates, BoN glossary, trace grep |
| `FOR_NEXT_LLM.md` | Tuning traps + batch learnings |
| `AGENTS.md` | Source vs artifacts map |
| `eval/PARALLEL_MLX_TESTING.md` | Multi-game parallel via mlx-server |
