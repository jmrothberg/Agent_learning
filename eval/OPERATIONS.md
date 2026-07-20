# eval/OPERATIONS.md — run tests and batches (LLM entry point)

**Start here** when the user asks to run tests, triage a tune batch, or kick off N games.
Human onboarding → `README.md`. Commands/env → `DEV.md`. Harness traps → `HARNESS_TUNING.md`.

---

## HARD RULES — overnight batch (never violate)

**One script / one double-click.** Prefer the Terminal Q&A (no command line). CLI flags still work for agents.

```bash
# You (Finder): double-click Overnight.command
# Or in Terminal.app:  bash eval/overnight.sh
# Asks: prompt numbers → iterations → VLM yes/no → model → Start?

# Agent / CLI:
bash eval/overnight.sh --prompts 54,28,21 --model GLM-5.2-MLX-4bit --vlm no
bash eval/overnight.sh --list
```

| Role | Where it must appear | How it starts | Forbidden |
|------|----------------------|---------------|-----------|
| **Batch** | **macOS Terminal.app** (visible Chromium) | Double-click `Overnight.command` **or** agent runs `overnight.sh` with `all` perms | Cursor integrated terminal · asking human to paste |
| **Watcher** | **Cursor IDE terminals panel** (`monitor:` lines) | Cursor **Shell**, `block_until_ms=0`, command printed after batch starts | `nohup` · skipping the watcher |

### When the user starts an overnight

1. They double-click **`Overnight.command`** (or run `bash eval/overnight.sh` in Terminal) and answer the questions.
2. Agent starts the printed watcher in a Cursor Shell (`block_until_ms=0`).
3. Patch harness from traces while it runs. **Never halt.**

If the agent must launch for them: `bash eval/overnight.sh --interactive` (opens Terminal Q&A) or CLI `--prompts/--model/--vlm`. Optional: `--run-id`, `--max-iters`, `--retries`, `--dry-run`.

### Launch recipe (agent checklist — every overnight)

```text
[ ] 1. User: double-click Overnight.command  (or agent: bash eval/overnight.sh --interactive with full OS perms)
      Confirm overnight.log shows planning/coding, NOT chrome-mac-x64 / Playwright missing.
[ ] 2. Watcher — Cursor Shell ONLY (block_until_ms=0); use the line printed in Terminal:
        .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_N --jobs-total K --interval 30 --sync-loop
[ ] 3. Tell the user both are up. Do not ask them to paste anything.
[ ] 4. Improve while it runs. Never pause the batch.
```

Legacy `eval/tune_runXX.sh` still works but **new nights use `Overnight.command` / `overnight.sh`**.

**Burned on run_18 (do not repeat):**
- Batch inside Cursor → wrong Playwright arch (`chrome-mac-x64`) → instant `fresh_fail` ×11.
- Asking the human to paste → wrong; use `Overnight.command` or `overnight.sh --interactive`.
- Watcher via `nohup` → user cannot see it in Cursor; always use Cursor Shell `block_until_ms=0`.

---

## If the user says… → run this

| User intent | Command | Notes |
|-------------|---------|-------|
| **Overnight (default)** | Double-click `Overnight.command` · Cursor Shell watcher | Interactive: prompts → iters → VLM → model. Or CLI `overnight.sh --prompts …`. |
| **Run Mr. Do! + 10 graphics/3D overnight (run_18)** | legacy `eval/tune_run18.sh` or `overnight.sh --prompts …` | **GLM-5.2-MLX-4bit**, VLM off, `--max-iters 3`. Already launched. |
| **Run 20 GRAPHICS-BEST games overnight (run_15 — tonight)** | Terminal: `bash eval/tune_run15.sh` · Cursor: watcher below | **GLM-5.2-MLX-4bit**, **`--no-vlm-critique`**, flat-out. High-confidence watcher fixes only. |
| **Run 10 NEW games overnight (run_14)** | Terminal: `bash eval/tune_run14.sh` · Cursor: watcher below | Qwen3.6-27B-mxfp8, VLM critique ON (completed). |
| **Run 10 NEW games overnight (run_13)** | Terminal: `bash eval/tune_run13.sh` · Cursor: watcher below | GLM-5.2-MLX-4bit, **`--no-vlm-critique`**, flat-out. Watcher fixes in parallel. |
| **Run 10 games overnight (run_08)** | Terminal: `bash eval/tune_run08.sh` · Cursor: watcher below | Batch runs **flat-out** (no pause). Watcher fixes in parallel. |
| **Run 10 games validation (run_10 — run_09 fix retest)** | Terminal: `bash eval/tune_run09.sh` · Cursor: watcher below | `--max-iters 4`, fresh `run_10/` dir. See § run_10. |
| **Run all 11 games overnight (both batches, auto-chained)** | Agent `osascript`s `eval/tune_run07_chain.sh` in Terminal + monitor in Cursor | Batch B starts automatically when A finishes. No wake-up. |
| **Run 11 games to improve the agent (run_07)** | Same as chain row above | A=GLM no VLM (6) → B=Qwen VLM on (5), watcher handoff between games. |
| **Run unit tests** / **pytest** / **after a code change** | `.venv/bin/python -m pytest tests/ -q` | ~2330 tests, no GPU. Full map: `TEST.md`. |
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
| **Compare tune runs (scoreboard)** | `.venv/bin/python eval/compare_runs.py run_15 run_16` | Cross-run fresh_pass / wasted_iters / failure_class — measure before/after harness changes. |
| **Run 10 graphics-heavy games (run_16)** | **completed** — `games/tune_serial10/run_16/` | **5/10 fresh_pass** · GLM-5.2 · `--max-iters 3` · scoreboard below. |
| **Offline playbook credit (dry-run)** | `.venv/bin/python scripts/credit_bullets.py games/tune_serial10/run_15 --dry-run` | Helpful/harmful deltas from traces; omit `--dry-run` to apply + ledger dedupe. |
| **Batch dashboard / watcher (run_07 chain)** | `.venv/bin/python eval/tune_overnight_monitor.py --run07-chain --interval 30 --sync-loop` | Polls every **30 seconds** (not minutes). Triage + patch while batch keeps running. |
| **Parallel N games (throughput lab)** | See `eval/PARALLEL_MLX_TESTING.md` + `eval/batch_parallel.py` | One `mlx_lm.server`, N clients — **not** in-game BoN. |

---

## run_07 — both batches, one night (A → B auto-chained)

**One Terminal.app launch runs all 11 games back-to-back** (`osascript` → `tune_run07_chain.sh`). Batch B starts automatically when A finishes. **No pause between games** (`--wait-for-monitor 0` default). Cursor watcher runs in parallel and patches harness/memory while games continue — fixes apply to the next game(s).

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

## run_18 — (Mr. Do! + 10 graphics/3D, GLM-5.2, max-iters 3)

Goals never used in run_16 or run_17: new Mr. Do!, never-run vector 3D (Battlezone, Star Wars), voxel/FPS 3D, particles, and high-sprite arcade last seen in run_14/15. Goals: `eval/tune_run18_goals.txt`.
**Model:** `~/MLX_Models/GLM-5.2-MLX-4bit`. **`--no-vlm-critique`**. **`--max-iters 3 --retries 0`**.

| # | Library | Why |
|---|---------|-----|
| 1 | mr-do | new digger arcade (sprites) |
| 2 | battlezone | 3D vector tank (never overnight) |
| 3 | star-wars | 3D vector trench (never overnight) |
| 4 | minecraft | voxel 3D (last run_14) |
| 5 | doom | three.js FPS (last run_15, not 16/17) |
| 6 | asteroids | vector ship/rocks (last run_11) |
| 7 | missile-command | particle explosions (last run_11) |
| 8 | particle-fireworks | particle FX showcase (last run_14) |
| 9 | metal-slug | run-gun sprites (last run_15, not 16/17) |
| 10 | dig-dug | digger sibling to Mr. Do! (last run_15) |
| 11 | rampage | climb-smash sprites (last run_15, not 16/17) |

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/launch_overnight_batch.sh eval/tune_run18.sh` | Cursor Shell `block_until_ms=0`: `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_18 --jobs-total 11 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_18/overnight.log` | `games/tune_serial10/run_18/agent_monitor.json` |

```bash
bash eval/launch_overnight_batch.sh eval/tune_run18.sh
```

```bash
# Cursor Shell — must be visible in IDE terminals panel (NOT nohup)
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_18 \
  --jobs-total 11 --interval 30 --sync-loop
```

---

## run_16 — (10 graphics-heavy, GLM-5.2, max-iters 3) — completed

Fresh modalities not exhausted by run_15 (shooters, animated board games, open-field TD, roguelike, pinball, bullet hell, lit dungeon). Goals: `eval/tune_run16_goals.txt`.
**Model:** `~/MLX_Models/GLM-5.2-MLX-4bit`. **`--no-vlm-critique`**. **`--max-iters 3 --retries 0`**.

| # | Outcome | Label |
|---|---------|-------|
| 1–2 | fresh_fail (infra / SIGKILL, no iter_summary) | Centipede, Galaga |
| 3–6, 10 | fresh_pass | 1942, Holochess, Checkers, Fieldrunners, Torch Dungeon |
| 7–9 | fresh_fail (quality) | Roguelike (stairs/fog + monsters_step), Pinball (playfield entry), Bullet Hell (steady-state `bullets_spawn`) |

### Scoreboard snapshot — run_15 vs run_16 (2026-07-18)

```bash
.venv/bin/python eval/compare_runs.py run_15 run_16
```

| run | status | jobs | fresh_pass | artifact_pass | fail | avg wasted_iters | avg first_clean | never_clean | infra_failed | avg tok/s |
|-----|--------|------|------------|---------------|------|-----------------|-----------------|-------------|--------------|-----------|
| run_15 | incomplete | 24 | 12 | 2 | 10 | 0.32 | 2.23 | 7 | 5 | 9.01 |
| run_16 | incomplete | 10 | 5 | 0 | 5 | 0.23 | 2.6 | 3 | 5 | 9.02 |

failure_class (non-ok iters): run_15 `memory_gap=33`, `local_llm_limit=1`; run_16 `memory_gap=12` (plus 2 early SIGKILL with no iters).

**Harness/memory landed mid-batch:** syntax soft_warning cascade suppress; partial-quarantine gate cap 2→1; ENTITY fog skip (`seen`/`explored`); pinball `auto_body` reseat; `outline-bullet-hell` / roguelike / pinball trap updates. Optional re-run Centipede/Galaga later (infra, not quality).

### Post-batch learning (traces + HTML, no new model runs — 2026-07-18)

Applied `scripts/credit_bullets.py` on run_14/15/16 (infra SIGKILL skipped). Fact-based harness FPs from shipped HTML:

| Evidence | Fix |
|----------|-----|
| DL `auto_qte_threat_position_advances` → `undefined helpers: d` (local arrow) | Skip local decls in helper diagnose; recipe inlines `Math.hypot` |
| Typing `document.dispatchEvent(KeyboardEvent)` vs `window` listener | `_patch_probe_keyboard_dispatch` dual-fires both |
| OutRun intro-only draw of `key_art` with green probes + undrawn cars | Intro/title/menu mode demotes ASSETS_UNDRAWN |
| Bullet hell `length>b0` after wait with 94 live bullets | Phase-A `fragile_length_growth_probe` lint |
| DK `jumpOverSet` never `.add` / score | `outline-vertical-platformer` trap + probe |

Repetition abort threshold left at `_BLOCK_MAX_REPEATS=3` — lowering to 2 false-positived healthy semicolon-chained streams.

---

## run_15 — (20 GRAPHICS-BEST games, GLM-5.2, no VLM critique) — completed

Asset-heavy library goals (fighters, QTE, platformers, FPS, point-click, arcade). Goals: `eval/tune_run15_goals.txt` (canonical `prompt_library.jsonl` — no speculative goal appendices).
**Model:** `~/MLX_Models/GLM-5.2-MLX-4bit`. **`--no-vlm-critique`**. `--max-iters 4 --retries 0`.

| Slot | Library | Modality |
|------|---------|----------|
| 1 | donkey-kong | animated platformer |
| 2 | frogger | lane crossing |
| 3 | dragons-lair | visual QTE + video |
| 4 | super-mario | platformer + video |
| 5 | prince-of-persia | rotoscope platformer |
| 6 | doom | FPS / 3D |
| 7 | zelda | top-down RPG |
| 8 | street-fighter | versus fighter + video |
| 9 | mortal-kombat | versus fighter + video |
| 10 | kung-fu-master | beat-em-up + video |
| 11 | joust | flap platformer |
| 12 | rampage | destruction climber |
| 13 | pac-man | maze chase |
| 14 | outrun | Mode-7 racer |
| 15 | monkey-island | point-and-click |
| 16 | metal-slug | run-and-gun |
| 17 | fighter-showcase | single-fighter poses |
| 18 | qbert | isometric hopper |
| 19 | dig-dug | digger arcade |
| 20 | bomberman | grid bombs |

**Two processes — both required.** Terminal batch runs **game → game with zero wait** and opens a **visible Chromium window**. Cursor watcher triages finished traces and patches harness/memory/prompts **only when highly confident**.

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run15.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_15 --jobs-total 20 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_15/overnight.log` | `games/tune_serial10/run_15/agent_monitor.json` |

**Open batch in Terminal.app from Cursor:**

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/jonathanrothberg/Agent_learning && bash eval/tune_run15.sh"'
```

```bash
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_15 \
  --jobs-total 20 \
  --interval 30 \
  --sync-loop
```

**Watcher loop (continuous, no blocking the batch):**

1. Poll `agent_monitor.json` — when `completed_count` advances, open the newest trace for that label.
2. **Timeline:** `.venv/bin/python scripts/enrich_trace.py <trace> --timeline`
3. **Classify** `failure_class` → patch only on high-confidence evidence (`tools.py` / `agent_*.py` / `memory/*.jsonl` / `prompts_v1.py`)
4. **Keep going** — next game already running; fixes apply to subsequent games.

Artifacts: `games/tune_serial10/run_15/` (`overnight.log`, `traces/`, `tune_checkpoint.json`).

### Scoreboard snapshot — run_14 vs run_15 (2026-07-17)

Measured with `eval/compare_runs.py` on real `tune_summary.json` + traces (run_15 still in progress when captured):

| run | status | jobs | fresh_pass | artifact_pass | fail | avg wasted_iters | avg first_clean | never_clean | avg tok/s |
|-----|--------|------|------------|---------------|------|-----------------|-----------------|-------------|-----------|
| run_14 | incomplete | 10 | 4 | 0 | 6 | 0.6 | 1.5 | 6 | 4.06 |
| run_15 | running | 18 | 7 | 2 | 9 | 0.4 | 2.38 | 12 | 9.15 |

failure_class (non-ok iters): run_14 `memory_gap=6`; run_15 `memory_gap=28`, `local_llm_limit=1`.

Re-run after 20/20: `.venv/bin/python eval/compare_runs.py run_14 run_15`.

---

## run_14 — (10 NEW games, Qwen3.6-27B-mxfp8, VLM critique ON)

All-new library goals never used in run_07–13. Goals: `eval/tune_run14_goals.txt`.
**Model:** `~/MLX_Models/Qwen3.6-27B-mxfp8`. **VLM critique ON** (default — no `--no-vlm-critique`). `--max-iters 4 --retries 0`.

| Slot | Library | Modality |
|------|---------|----------|
| 1 | tetris | falling-block puzzle |
| 2 | snake | grid arcade |
| 3 | pong | paddle duel |
| 4 | doom | FPS raycaster / 3D |
| 5 | minecraft | voxel sandbox |
| 6 | dragons-lair | visual QTE |
| 7 | particle-fireworks | particle VFX |
| 8 | tower-defense | fixed-path TD |
| 9 | cookie-clicker | idle clicker |
| 10 | typing-race | typing |

**Two processes — both required.** Terminal batch runs **game → game with zero wait** and opens a **visible Chromium window**. Cursor watcher triages each finished trace and patches harness/memory/prompts **as evidence arrives** (do not wait until the end).

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run14.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_14 --jobs-total 10 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_14/overnight.log` | `games/tune_serial10/run_14/agent_monitor.json` |

**Open batch in Terminal.app from Cursor:**

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/jonathanrothberg/Agent_learning && bash eval/tune_run14.sh"'
```

```bash
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_14 \
  --jobs-total 10 \
  --interval 30 \
  --sync-loop
```

**Watcher loop (continuous, no blocking the batch):**

1. Poll `agent_monitor.json` — when `completed_count` advances, open the newest trace for that label.
2. **Timeline:** `.venv/bin/python scripts/enrich_trace.py <trace> --timeline`
3. **Classify** `failure_class` → patch `tools.py` / `agent_*.py` / `memory/*.jsonl` / `prompts_v1.py`
4. **Keep going** — next game already running; fixes apply to subsequent games.

Artifacts: `games/tune_serial10/run_14/` (`overnight.log`, `traces/`, `tune_checkpoint.json`).

---

## run_13 — (10 NEW games, GLM-5.2, no VLM critique)

All-new library goals never used in run_07–12. Goals: `eval/tune_run13_goals.txt`.
**Model:** `~/MLX_Models/GLM-5.2-MLX-4bit`. **`--no-vlm-critique`**. `--max-iters 4 --retries 0`.

| Slot | Library | Modality |
|------|---------|----------|
| 1 | 1942 | vertical shoot-'em-up |
| 2 | stealth-infiltration | stealth / FOV |
| 3 | simcity-lite | city builder |
| 4 | elite-trader | space trading |
| 5 | match-three | match-3 puzzle |
| 6 | stacking-tower | physics stack |
| 7 | rhythm-tap | rhythm |
| 8 | solitaire | cards |
| 9 | angry-blocks | physics puzzle |
| 10 | fighter-showcase | pose / animation showcase |

**Two processes — both required.** Terminal batch runs **game → game with zero wait** and opens a **visible Chromium window**. Cursor watcher triages each finished trace and patches harness/memory/prompts **as evidence arrives** (do not wait until the end).

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run13.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_13 --jobs-total 10 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_13/overnight.log` | `games/tune_serial10/run_13/agent_monitor.json` |

**Open batch in Terminal.app from Cursor:**

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/jonathanrothberg/Agent_learning && bash eval/tune_run13.sh"'
```

```bash
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_13 \
  --jobs-total 10 \
  --interval 30 \
  --sync-loop
```

**Watcher loop (continuous, no blocking the batch):**

1. Poll `agent_monitor.json` — when `completed_count` advances, open the newest trace for that label.
2. **Timeline:** `.venv/bin/python scripts/enrich_trace.py <trace> --timeline`
3. **Classify** `failure_class` → patch `tools.py` / `agent_*.py` / `memory/*.jsonl` / `prompts_v1.py`
4. **Keep going** — next game already running; fixes apply to subsequent games.

Artifacts: `games/tune_serial10/run_13/` (`overnight.log`, `traces/`, `tune_checkpoint.json`).

---

## run_08 — (10 games, flat-out + parallel watcher)

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

**Two processes — both required.** Terminal batch runs **game → game with zero wait** and opens a **visible Chromium window** for each game (no `--headless`). **Run the batch in Terminal.app** — Cursor’s integrated terminal often hides or kills the browser window.

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run08.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_08 --jobs-total 10 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_08/overnight.log` | `games/tune_serial10/run_08/agent_monitor.json` |

**Open batch in Terminal.app from Cursor** (agent always does this — never ask the human):

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/jonathanrothberg/Agent_learning && bash eval/tune_run08.sh"'
```

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

## run_10 — validation batch (run_09 fix retest)

Re-runs run_09 failures plus games that exercise each landed fix class (sprite PATHS wiring, video memory relief, asset-overflow gate, pyramid-hopper outline, roguelike art). Goals: `eval/tune_run09_goals.txt`. **`--max-iters 4 --retries 0`**, VLM off.

| Slot | Library | Fix class under test |
|------|---------|----------------------|
| 1 | centipede | First-build PATHS / ASSETS_LOADED_BUT_UNDRAWN |
| 2 | galaga | Sprite wiring control (recovered in run_09) |
| 3 | pinball | Sprite wiring + local_llm probes-only turn |
| 4 | qbert | `canvas-pyramid-hopper` / `outline-pyramid-hopper` |
| 5 | roguelike-dungeon | Art language + map-tile ENTITY-NOT-RENDERED |
| 6 | tower-defense-openfield | `ASSETS_DROPPED_PENDING` gate |
| 7 | super-mario | Unconditional video memory relief |
| 8 | kung-fu-master | Video relief (beat-em-up + cutscene) |
| 9 | dragons-lair | Video relief (heaviest QTE + multi-video) |
| 10 | street-fighter | Multi-frame animation stress |

| | Terminal.app (once) | Cursor watcher (once) |
|---|---------------------|------------------------|
| Command | `bash eval/tune_run09.sh` | `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_10 --jobs-total 10 --interval 30 --sync-loop` |
| Log / status | `games/tune_serial10/run_10/overnight.log` | `games/tune_serial10/run_10/agent_monitor.json` |

**Open batch in Terminal.app from Cursor:**

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/jonathanrothberg/Agent_learning && bash eval/tune_run09.sh"'
```

```bash
cd /Users/jonathanrothberg/Agent_learning
bash eval/tune_run09.sh
```

```bash
.venv/bin/python eval/tune_overnight_monitor.py \
  --out-dir games/tune_serial10/run_10 \
  --jobs-total 10 \
  --interval 30 \
  --sync-loop
```

Artifacts: `games/tune_serial10/run_10/` (`overnight.log`, `traces/`, `tune_checkpoint.json`).

---

## Serial overnight batch (manual launch)

**Primary workflow:** see **run_07 chain** above. For a custom dir/goals file:

| File | Purpose |
|------|---------|
| **`eval/tune_run08_goals.txt`** | **run_08 tonight (10 games)** — see § run_08 above |
| **`eval/tune_run09_goals.txt`** | **run_10 validation (10 games)** — see § run_10 above |
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
5. **Persist learnings:** copy into `HARNESS_TUNING.md` + optional `memory/playbook.jsonl` bullet — not into generated `games/*.html`.

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
| `HARNESS_TUNING.md` | Tuning traps + batch learnings |
| `AGENTS.md` | Source vs artifacts map |
| `eval/PARALLEL_MLX_TESTING.md` | Multi-game parallel via mlx-server |
