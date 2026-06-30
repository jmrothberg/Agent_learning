# Testing guide

Three layers, fastest first. **Batch runs and “what command do I run?”** → **[`eval/OPERATIONS.md`](eval/OPERATIONS.md)** (natural-language → command table, overnight launch, artifact paths). Commands also appear in **`README.md`** and **`DEV.md`** — this file is the **canonical map** of what to run and what each test area guards.

## Layer 1 — unit suite (`tests/`)

Pure-function, deterministic: stub backend, mock browser, `tmp_path` memory. **Run after every harness/agent change.**

```bash
.venv/bin/python -m pytest tests/ -q                  # full suite (~2048 tests, ~1–2 min)
.venv/bin/python -m pytest tests/test_patches.py -v   # one file
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v
```

**Canonical smoke after retrieval/prompt/patch changes:** asteroids — ship direction (`vx = cos(angle)*speed`) and irregular-polygon asteroids (not perfect circles).

### Conventions

| Pattern | Use |
|---------|-----|
| `GameAgent(model="stub", browser=MagicMock(), memory_root=str(tmp_path/"mem"))` | Default agent fixture |
| Stub `agent._stream` or `backend.stream_chat` | Drive loop without GPU |
| `browser=None` | Materialization-only (no Chromium) |
| **Source grep after mixin split** | Loop body → `GameAgent.run_loop_inspect_source()`; agent+mixin methods → `GameAgent.class_inspect_source()`; module-level across mixins → `module_inspect_source()` from `agent` |

Do **not** grep `inspect.getsource(agent)` or `inspect.getsource(GameAgent)` for loop logic — that only sees the `agent.py` class body, not mixins.

### What the suite guards (by subsystem)

| Subsystem | Primary tests | What must stay true |
|-----------|---------------|---------------------|
| **Patch engine** | `test_patches.py`, `test_materialize_msg.py`, `test_format_rejection.py` | 4-tier match, non-overlap, repair_reply |
| **Verifier / gates** | `test_probe_gate.py`, `test_static_action_gate.py`, `test_microprobes.py`, `test_drawn_asset_detector.py`, `test_dead_animation_gate.py` | `ok=False` on real behavioral gaps; cosmetic sprite warnings non-gating |
| **Feedback routing** | `test_feedback_router.py`, `test_blocker_first_feedback.py`, `test_scoped_feedback.py`, `test_golden_feedback_flows.py` | User feedback authoritative; art vs code vs scope locks |
| **Agent loop** | `test_iter_loop_guards.py`, `test_stall_recovery.py`, `test_exit_decision_turn.py`, `test_final_iter_test_guarantee.py`, `test_plan_retry.py` | Phase A/B/C, stall recovery, exit honesty, final untested iter |
| **Compaction / context** | `test_compaction.py`, `test_token_aware_compaction.py`, `test_num_ctx.py` | Token-aware pressure; playbook survives feedback |
| **Assets / media** | `test_assets.py`, `test_midsession_assets.py`, `test_asset_alignment.py`, `test_seed_phase_a_skip.py`, `test_mid_session_asset_deferral_and_runaway.py` | Alignment scan, rehydrate, style-rebrand deferral |
| **Memory / prompts** | `test_retrieval.py`, `test_prompt_library*.py`, `test_opening_book_memory.py`, `test_open_domain_routing.py` | Genre-free retrieval; plan nudges data-driven |
| **Trace / diagnostics** | `test_trace_diagnostics.py`, `test_patch_outcome_trace.py`, `test_failure_class_routing.py` | `failure_class`, `iter_summary`, ephemeral events |
| **Backend / streaming** | `test_ollama_io.py`, `test_max_tokens_signal.py`, `test_repetition.py`, `test_deliberation_thresholds.py` | Sampling, repetition latch on code emission |

### Trace-backed regression guards

These pin fixes from specific production traces. Prefer **extending** an existing file when the failure class matches; add a new file only for a new failure *class*.

| File | Trace / theme |
|------|----------------|
| `test_2026_05_23_fixes.py` | Pac-Man + SOTA chess (short stream, fan-out, endpoint concurrency) |
| `test_fix_round.py` | Multi-item harness round (compaction, gates, critic dedup) |
| `test_doom_trace_fixes.py`, `test_doom_general_improvements.py`, `test_doom_feedback_misroute.py` | Doom / FPS traces |
| `test_wolfenstein_stuck_loop_fixes.py` | Stuck loop / restart signature |
| `test_qte_quality_hardening.py` | Dragon's Lair QTE wiring |
| `test_phase2_fix_coaching.py` | Degenerate baseline rewrite trap |
| `test_run06_draw_contract.py` | Serial tune run_06 drawImage contract (`memory_gap`) |

**Stub regression banks** (no model; loaded by pytest): `eval/golden_feedback_flows.jsonl`, `eval/modality_scenarios.jsonl`, `eval/seed_edit_scenarios.jsonl`, `eval/failure_class_routing.jsonl`.

### Overlap — intentional, not redundant

- **`test_seed_media_rehydrate.py` vs `test_seed_phase_a_skip.py`** — rehydrate logic vs Phase A skip guard (same pipeline, different assertions).
- **`test_post_clean_truth_source.py` vs `test_post_clean_feedback_truth_source.py`** — post-clean prompt truth vs feedback-channel truth.
- **`tools.py` source greps** — many files grep `LiveBrowser.load_and_test`; that is intentional (each guards a different gate string).

When adding coverage: **behavioral test first** (call the function with a fixture report). Use source grep only when the bug was “wiring never called”. Trace showed X → pick `failure_class` → extend the matching test file (see `eval/failure_class_routing.jsonl`).

### Shared fixtures (`tests/conftest.py`)

| Fixture | Use |
|---------|-----|
| `agent` | `GameAgent` + `MagicMock` browser + `tmp_memory` |
| `agent_no_browser` | Materialization-only (`browser=None`) |
| `tmp_memory` | Isolated `memory_root` under `tmp_path` |

Prefer these over copy-paste `_make_agent()` helpers in new tests.

---

## Layer 2 — prompt-library eval (local model, no browser)

```bash
.venv/bin/python eval/eval_prompts_plan.py --coverage   # instant memory matrix (CI)
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
```

### Layer 2b — seed-edit robustness

```bash
MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py
.venv/bin/python eval/eval_seed_edits.py --patch-only --max-iters 2
```

PASS = materialized + bytes changed (`browser=None`). Traces: `games/eval_seed_edits/`.

### Parallel batch (one MLX server, N clients)

See [`eval/PARALLEL_MLX_TESTING.md`](eval/PARALLEL_MLX_TESTING.md).

---

## Layer 3 — system tests (full loop, visible browser)

```bash
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes
```

Slow canaries only. Battery: `memory/system_battery.jsonl`.

---

## Scripts (`scripts/`)

| Prefix | Meaning |
|--------|---------|
| `_smoke_*` | GPU/model smoke — manual or optional CI |
| `archive/_apply_*` | One-shot migrations — **do not re-run** |
| `_gen_*` / `demo_*` / `live_test_*` | Session demos — ignore unless debugging that feature |
| *(no prefix)* | Supported maintainer tool |

| Script | Role | LLM |
|--------|------|-----|
| `setup.sh`, `update.sh` | Install deps / pull repo | setup |
| `install_diffuser.sh`, `install_mlx_*_fix.sh` | GPU stack / mlx-lm patches | setup |
| `clean_artifacts.sh` | Wipe stale `games/` (not `goodgame/` or `memory/`) | after batch |
| `forget_session.py` | Drop one session from memory index | hygiene |
| **`enrich_trace.py`** | **Primary triage** — timeline from `.jsonl` | **always** |
| `generate_video.py` | Standalone Wan2.2 clip | media |
| `_smoke_doom.py`, `_smoke_audio.py`, `_smoke_img2img.py` | Diffuser smoke | optional |
| **`_smoke_asset_decode_settle.py`** | **Chromium asset-decode settle + undrawn smoke** (dojo fixture) | **Playwright** |
| **`smoke_vlm_facing_sanity.py`** | **VLM must say NO on seed facing bug** (preflight before facing eval) | **MLX VLM + Playwright** |
| `asset_studio.py`, `draw_game_art.py`, `build_stock_sounds.py` | Asset tooling | as needed |
| `oneshot_game.py`, `play_folder.py` | Ad-hoc runs | optional |
| `smoke_cloud_backends.py` | Cloud API smoke | optional |
| `archive/_apply_agent_*.py`, `archive/_apply_qte_telegraph.py` | Historical splits | **ignore** |

## Eval (`eval/`)

| Script | Role |
|--------|------|
| `tune_serial_loop.py` | Serial N-game batch driver |
| `tune_serial_overnight.sh` | Watchdog wrapper for unattended batches |
| `tune_run07_chain.sh` | Auto-chained run_07 A→B |
| `tune_overnight_monitor.py` | Batch dashboard (`agent_monitor.json`) |
| `tune_inter_game_ready.py` | Release next game after mid-batch fix |
| `batch_parallel.py` | N clients → one `mlx_lm.server` |
| `eval_prompts_plan.py` | Plan-turn eval + `--coverage` matrix |
| `eval_seed_edits.py` | Seed-edit materialization eval |
| `eval_vlm_facing_fix.py` | Agent + VLM-critique facing regression (opt-in, needs model) |

Trace workflow: **`HARNESS_DEBUG.md`**. Source vs artifacts: **`AGENTS.md`**. Batch / overnight: **`eval/OPERATIONS.md`**.

---

## Chrome asset-decode settle (harness regression)

After changes to `tools.py` undrawn detection or async asset loaders:

```bash
# 1. Unit pins + fixture shape (no browser)
.venv/bin/python -m pytest tests/test_drawn_asset_detector.py tests/test_dojo_asset_smoke_fixture.py -q

# 2. Harness smoke (Chromium, about 8s)
env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python scripts/_smoke_asset_decode_settle.py

# 3. VLM smoke — Qwen3.6-27B-mxfp8 (about 1 min cold load)
env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python scripts/_smoke_asset_decode_settle.py --vlm

# Opt-in: pytest wrapper shells the smoke script
CHROMIUM_SMOKE=1 .venv/bin/python -m pytest tests/test_dojo_asset_smoke_fixture.py::test_dojo_asset_smoke_chromium_wrapper -q
```

Fixture: `eval/fixtures/dojo_fighters_asset_smoke.html` (sprites from `goodgame/build-a-single-screen-2d-fight_*_assets/`).

## VLM-critique facing regression (opt-in)

Minimal seeded bug: both fighters stay `facing: +1` with no flip toward opponent.
Exercises the full agent loop (patch-only) + `/vlm-critique` + facing probes.
**Main agent `/vlm-critique` is unchanged** — still runs for simpler visual checks when you use a VLM coder; this eval only gates *facing* on a model that passes sanity.

```bash
# 1. Preflight — VLM must say NO on the seed bug screenshot (exit 0 = trustworthy)
MLX_MODEL=~/MLX_Models/<your-vlm> \\
  .venv/bin/python scripts/smoke_vlm_facing_sanity.py

# 2. Full eval (skipped automatically if step 1 would fail on this model)
MLX_MODEL=~/MLX_Models/<your-vlm> \\
  .venv/bin/python eval/eval_vlm_facing_fix.py

# mxfp8 27B currently fails sanity (answers YES on obvious wrong facing) — eval exits 3.

# Force eval anyway (not recommended):
.venv/bin/python eval/eval_vlm_facing_fix.py --skip-sanity
```

**PASS (eval):** preflight sanity pass + post-run VLM facing Q **YES** after crossover screenshot.
**Secondary:** state probes `auto_actors_face_each_other`, `auto_actors_face_each_other_strict`, `facing_crossover_strict`.
**Trace audit:** `visual_playtest_parsed` now includes `raw_preview` and `answers` on every iter (not only parse failures).

Fixture: `eval/fixtures/seed_fighters_facing_bug.html` · assets: `eval/fixtures/seed_fighters_facing_bug_assets/`. Eval copies sprites to `games/eval_vlm_facing/seed_fighters_facing_bug_assets/`. Disables step-mode for unattended runs.
