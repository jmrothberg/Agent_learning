# Testing guide

Three layers, fastest first. Full eval/system-test commands also appear in **`README.md`**.

## Layer 1 — unit suite (`tests/`)

Pure-function, deterministic: stub backend, mock browser, `tmp_path` memory. **Run this for every
change.**

```bash
.venv/bin/python -m pytest tests/ -q                  # full suite (~1 min)
.venv/bin/python -m pytest tests/test_patches.py -v   # one file
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v
```

**Conventions:** `GameAgent(model="stub", browser=MagicMock(), memory_root=str(tmp_path/"mem"))`;
stub `backend.stream_chat` or monkeypatch `agent._stream`; no network or GPU.

| Area | Representative files |
|------|----------------------|
| Patch engine | `test_patches.py` |
| Memory / retrieval / prompts | `test_retrieval.py`, `test_architect_opening_library.py`, `test_prompt_library*.py`, `test_open_domain_routing.py`, `test_opening_book_memory.py` |
| Probes & gates | `test_probe_gate.py`, `test_static_action_gate.py`, `test_dead_animation_gate.py` |
| Assets / animation | `test_assets.py`, `test_action_frame_capture.py` |
| Agent loop / compaction / feedback | `test_*compaction*.py`, `test_feedback_router.py`, `test_session_timeouts.py` |
| Stall recovery (deliberation/loop/silent) | `test_stall_recovery.py`, `test_repetition.py` (incl. markdown-patch bloat grace) |
| Golden feedback flows + modality/seed banks | `test_golden_feedback_flows.py`, `test_modality_scenarios.py`, `test_modality_disambiguation.py`, `test_action_gate_non_combat_keys.py`, `test_seed_edit_scenarios.py` |
| Trace diagnostics (4D) | `test_trace_diagnostics.py` — ephemeral events, `failure_class`, digest |
| Backend / context | `test_num_ctx.py`, `test_max_tokens_signal.py` |

**Regression banks** (`eval/*.jsonl`, stub-only — no model): `golden_feedback_flows.jsonl`
(Fieldrunners trace `20260626_102307`: locks the working iter 2-3 art→patch / behavior-bug
classification and documents the iter 4-5 failure inputs), `modality_scenarios.jsonl`
(beat-em-up suppression + non-combat key exclusion), `seed_edit_scenarios.jsonl`
(small-scope / orientation / size edit classification).

Canonical regression after retrieval/prompt/patch changes: **asteroids** — ship direction
(`vx = cos(angle)*speed`) and irregular-polygon asteroids (not perfect circles).

## Layer 2 — prompt-library eval (local model, no browser)

```bash
.venv/bin/python eval/eval_prompts_plan.py --coverage   # instant memory matrix (also in CI)
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
```

One planning turn per curated prompt; traces under `games/eval-traces/`. See **`README.md`**.

### Layer 2b — seed-edit robustness (local model, no browser)

```bash
MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py
.venv/bin/python eval/eval_seed_edits.py --only 1 --max-iters 2   # one scenario
.venv/bin/python eval/eval_seed_edits.py --patch-only --max-iters 2   # skip Phase A (canvas seeds)
```

Runs real build/iterate turns over `eval/fixtures/seed_tower_defense.html` with
`browser=None`, so it measures **materialization** (did the model emit code that
changed the file?) not gameplay. The agent skips `load_and_test`, emits `browser_test_skipped`
+ `iter_summary` with `test_skipped:no_browser` (no spurious `harness_crash`). **PASS** = a
`<patch>`/`<html_file>` landed and the bytes differ from the seed; **FAIL** = no-code or
byte-identical. Traces under `games/eval_seed_edits/` (gitignored run artifacts).

## Layer 3 — system tests (full loop, visible browser)

```bash
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes
```

Slow; use for canaries, not per-prompt coverage. Battery: `memory/system_battery.jsonl`.
