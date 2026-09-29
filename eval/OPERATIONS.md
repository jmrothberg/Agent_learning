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

**oMLX must have native Metal kernels (Mac).** Before any batch: `curl -s :8000/api/status | jq .custom_kernels` → all `available: true`. A source/git pip install has none (Aug 4–Sep 11 2026 ran that way); install the prebuilt release wheel — `HARNESS_TUNING.md` "HARD RULE (Mac) — oMLX must run with native Metal kernels". `chat.py` / `ensure_omlx_server()` refuses a kernel-less server.

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

**New nights use `Overnight.command` / `overnight.sh`.** That script writes `games/tune_serial10/run_N/goals.txt` from the prompts you pick.

**Burned on run_18 (do not repeat):**
- Batch inside Cursor → wrong Playwright arch (`chrome-mac-x64`) → instant `fresh_fail` ×11.
- Asking the human to paste → wrong; use `Overnight.command` or `overnight.sh --interactive`.
- Watcher via `nohup` → user cannot see it in Cursor; always use Cursor Shell `block_until_ms=0`.

---

## If the user says… → run this

| User intent | Command | Notes |
|-------------|---------|-------|
| **Overnight (default)** | Double-click `Overnight.command` · Cursor Shell watcher | Interactive: prompts → iters → VLM → model. Or CLI `overnight.sh --prompts …`. |
| **Run unit tests** / **pytest** / **after a code change** | `.venv/bin/python -m pytest tests/ -q` | ~2343 tests / 195 files, no GPU. Full map: `TEST.md`. |
| **Run one test file** | `.venv/bin/python -m pytest tests/test_patches.py -v` | Swap path. |
| **Run asteroids regression** | `.venv/bin/python -m pytest tests/test_retrieval.py tests/test_patches.py -q -k asteroids` | Ship thrust + irregular asteroids. |
| **Run 3D navigation guards** | `.venv/bin/python -m pytest tests/test_3d_navigation_conventions.py tests/test_doom_trace_fixes.py -q` | Skeletons, playbook, FPS yaw/movement conventions. |
| **Run run_06 guards** | `.venv/bin/python -m pytest tests/test_run06_draw_contract.py tests/test_tune_serial_pass.py tests/test_stream_instance_method.py tests/test_grid_maze_chase_probes.py -q` | drawImage contract + honest batch PASS + get_backend fix + grid-maze probes. |
| **Prompt library coverage (no model)** | `.venv/bin/python eval/eval_prompts_plan.py --coverage` | Instant; CI runs this. |
| **Plan eval (one model turn per prompt)** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_prompts_plan.py` | No browser. |
| **Seed-edit eval** | `MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py` | Materialization only (`browser=None`). |
| **One game headless** | `.venv/bin/python coder.py "snake" --max-iters 4 --headless` | Single session; trace under `games/traces/`. |
| **Interactive TUI** | `.venv/bin/python chat.py` | Visible Chromium; `/wait on` (`local_manual`) and `/bestof off` by default. |
| **System smoke (browser)** | `python system_tests.py run --suite smoke --three-model` | Slow; confirms full loop. |
| **Timeline a trace** | `.venv/bin/python scripts/enrich_trace.py <path-or-stem> --timeline` | Primary triage; see `HARNESS_DEBUG.md`. |
| **Compare tune runs (scoreboard)** | `.venv/bin/python eval/compare_runs.py run_15 run_16` | Cross-run fresh_pass / wasted_iters / failure_class — measure before/after harness changes. Also reports avg ttft_s / cached_prompt / media_s / ttf_test_s (Phase 1 timing columns). |
| **Run 10 graphics-heavy games (run_16)** | **completed** — `games/tune_serial10/run_16/` | **5/10 fresh_pass** · GLM-5.2 · `--max-iters 3` · scoreboard below. |
| **Offline playbook credit (dry-run)** | `.venv/bin/python scripts/credit_bullets.py games/tune_serial10/run_15 --dry-run` | Helpful/harmful deltas from traces; omit `--dry-run` to apply + ledger dedupe. |
| **Batch dashboard / watcher (run_07 chain)** | `.venv/bin/python eval/tune_overnight_monitor.py --run07-chain --interval 30 --sync-loop` | Polls every **30 seconds** (not minutes). Triage + patch while batch keeps running. |
| **Parallel N games (throughput lab)** | See `eval/PARALLEL_MLX_TESTING.md` + `eval/batch_parallel.py` | One `mlx_lm.server`, N clients — **not** in-game BoN. |

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
| `overnight.log` | `eval/overnight.sh` | Live progress, model tokens, errors |
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

## Batch outcome labels

**Batch outcome labels** (since run_06 reporting fix): `tune_serial_loop.py` and `agent_monitor.json` distinguish **checkpoint complete** (all labels recorded) from **fresh pass** (this run’s trace has `iter_summary ok=true` or new `.best.html`). **`artifact_pass`** = resume skip or SIGSEGV-after-ship reconcile when `.best.html` already existed; **`fresh_fail`** = subprocess ran but nothing verified shipped (e.g. run_06 games 03–06 hit `get_backend` TypeError in ~1 s). Do not treat `6/6 checkpoint complete` as `6/6 fresh pass`.

## Related docs

| Doc | When |
|-----|------|
| `TEST.md` | What each pytest file guards |
| `HARNESS_DEBUG.md` | Gates, BoN glossary, trace grep |
| `HARNESS_TUNING.md` | Tuning traps + batch learnings |
| `AGENTS.md` | Source vs artifacts map |
| `eval/PARALLEL_MLX_TESTING.md` | Parallel oMLX / mlx-server lab only (not serial overnight) |
