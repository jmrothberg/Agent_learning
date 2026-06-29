# eval/OPERATIONS.md — run tests and batches (LLM entry point)

**Start here** when the user asks to run tests, triage a tune batch, or kick off N games.
Human onboarding → `README.md`. Commands/env → `CLAUDE.md`. Harness traps → `FOR_NEXT_LLM.md`.

---

## If the user says… → run this

| User intent | Command | Notes |
|-------------|---------|-------|
| **Run unit tests** / **pytest** / **after a code change** | `.venv/bin/python -m pytest tests/ -q` | ~2048 tests, no GPU. Full map: `TEST.md`. |
| **Run one test file** | `.venv/bin/python -m pytest tests/test_patches.py -v` | Swap path. |
| **Run asteroids regression** | `.venv/bin/python -m pytest tests/test_retrieval.py tests/test_patches.py -q -k asteroids` | Ship thrust + irregular asteroids. |
| **Run run_06 guards** | `.venv/bin/python -m pytest tests/test_run06_draw_contract.py tests/test_tune_serial_pass.py tests/test_stream_instance_method.py -q` | drawImage contract + honest batch PASS + get_backend fix. |
| **Prompt library coverage (no model)** | `.venv/bin/python eval/eval_prompts_plan.py --coverage` | Instant; CI runs this. |
| **Plan eval (one model turn per prompt)** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_prompts_plan.py` | No browser. |
| **Seed-edit eval** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py` | Materialization only (`browser=None`). |
| **One game headless** | `.venv/bin/python coder.py "snake" --max-iters 4 --headless` | Single session; trace under `games/traces/`. |
| **Interactive TUI** | `.venv/bin/python chat.py` | Visible Chromium; `/bestof off` default. |
| **System smoke (browser)** | `python system_tests.py run --suite smoke --three-model` | Slow; confirms full loop. |
| **Timeline a trace** | `.venv/bin/python scripts/enrich_trace.py <path-or-stem> --timeline` | Primary triage; see `HARNESS_DEBUG.md`. |
| **Batch dashboard refresh** | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_06` | Writes `agent_monitor.json`; optional. |
| **Parallel N games (throughput lab)** | See `eval/PARALLEL_MLX_TESTING.md` + `eval/batch_parallel.py` | One `mlx_lm.server`, N clients — **not** in-game BoN. |

---

## Serial overnight batch (6 games, one MLX, visible browser)

**Goal lists** (pick one):

| File | Purpose |
|------|---------|
| `eval/tune_serial10_goals.txt` | Full 12-game battery |
| `eval/tune_serial10_round2_goals.txt` | Round 2 subset |
| `eval/tune_serial10_round2_rerun.txt` | run_06 validation (6 games) |

**Launch** (Terminal.app, not Cursor — survives IDE restarts):

```bash
cd /Users/jonathanrothberg/Agent_learning
mkdir -p games/tune_serial10/run_07
caffeinate -dims env \
  TUNE_OUT_DIR=games/tune_serial10/run_07 \
  TUNE_GOALS_FILE=eval/tune_serial10_round2_rerun.txt \
  MLX_MODEL="$HOME/MLX_Models/GLM-5.2-MLX-4bit" \
  nohup bash eval/tune_serial_overnight.sh &
tail -f games/tune_serial10/run_07/overnight.log
```

**Defaults baked in:** `--best-of-n 1`, stuck best-of-2 **off** (no `--stuck-bon`), `--no-vlm-critique`, `--no-auto-step`, `--resume`, `--retries 2`, `stall_seconds=1200`.

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

---

## run_06 snapshot (2026-06-29)

Model: `GLM-5.2-MLX-4bit`. Goals: `eval/tune_serial10_round2_rerun.txt` (6 games).

| Game | Result | Notes |
|------|--------|-------|
| 01 Donkey Kong | PASS | ~51 min; 1× memory_gap |
| 02 Kung-Fu Master | PASS (after retries) | Crouch gated movement → `input_moves_player` fail; cutscene setTimeout → enemies undrawn; stuck BoN burned hours when enabled |
| 03–06 | Multiple trace retries in log | Check newest trace per label |

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
