# Testing guide

This project has **three layers** of verification, fastest first. The `tests/`
suite is pure-function (no model, no GPU, no Chromium) and runs in seconds; the
eval harness and system tests exercise the real local model.

```bash
# Layer 1 — unit suite (no model/GPU/browser; the everyday gate)
.venv/bin/python -m pytest tests/ -q                  # full suite (~50 s, 1300+ tests)
.venv/bin/python -m pytest tests/test_patches.py -v   # one file
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v
```

As of this writing: **126 test files, ~1350 test functions.** Four failures are
known/pre-existing and environmental, not regressions:
`test_each_shape_retrieves_correct_outline` (live-store outline cruft in the
gitignored `games/game-memory/`), `test_post_clean_failed_branch_unchanged`,
`test_pick_diffuser_prefers_gpu0_on_workstation` (4×GPU-host assertion), and
`test_default_battery_file_exists` (needs a generated battery file).

---

## Layer 1 — unit suite (`tests/`)

Everything in `tests/` is model-free and deterministic. Backends are stubbed,
the browser is a `MagicMock`, and memory is pointed at a `tmp_path`. New code
should land with a test here. Rough thematic groups (not exhaustive):

| Area | Representative files |
|------|----------------------|
| Patch engine (SEARCH/REPLACE, smart-quote/whitespace match) | `test_patches.py` |
| Memory & retrieval (playbook, outlines, skeletons, mistakes) | `test_retrieval.py`, `test_architect_opening_library.py`, `test_prompt_library*.py` |
| Prompt assembly & planning (intent detectors, multi-frame, scope) | `test_multi_frame_planning_and_cap.py`, `test_*planning*.py` |
| Probes & gates (probe quality, lint, coverage, static-action) | `test_probe_gate.py`, `test_probe_reparse_gate.py`, `test_static_action_gate.py` |
| Planning robustness | `test_plan_retry.py` |
| Assets / sprites / animation | `test_assets.py`, `test_action_frame_capture.py`, `test_dead_animation_gate.py` |
| Sounds | `test_sound_alignment.py` |
| Visual playtests / VLM critic | `test_visual_playtest_*.py`, `test_vlm_classifier.py` |
| Cross-session asset & prompt libraries | `test_asset_library.py`, `test_prompt_library.py` |
| Agent loop, compaction, feedback flows, TUI status | `test_*compaction*.py`, `test_status_panel*.py`, `test_session_timeouts.py` |
| Backend / context window | `test_num_ctx.py`, `test_max_tokens_signal.py`, `test_check_routing.py` |

Conventions: construct the agent with `GameAgent(model="stub", browser=MagicMock(),
memory_root=str(tmp_path/"mem"))`; stub `backend.stream_chat` (or monkeypatch
`agent._stream`) to script model replies; never hit the network or GPU.

---

## Layer 2 — prompt-library eval (real local model, no browser/diffuser)

`eval/eval_prompts_plan.py` runs ONE Phase-A planning turn per curated prompt
(`memory/prompt_library.jsonl`) against the local model via the agent's
`plan_only` mode — no build loop, no Chromium, no asset generation. It checks
each prompt's critical feature (its `expect` block) and **keeps a full trace per
prompt** under `games/eval-traces/eval_<ts>/` (per-prompt agent `.jsonl` +
`.conversation.md`, the raw `*.plan.md`, `eval_summary.jsonl`, `eval_report.md`).

```bash
# Layer 0 — memory coverage matrix, NO model, instant (which subsystem fires per prompt)
.venv/bin/python eval/eval_prompts_plan.py --coverage

# One planning turn per prompt vs the local model (weights load once, ~1-3 min/prompt)
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
.venv/bin/python eval/eval_prompts_plan.py --only 1           # one prompt by number
.venv/bin/python eval/eval_prompts_plan.py --names chess,doom  # by name
```

The Layer-0 matrix is also asserted in CI by `tests/test_prompt_library_coverage.py`
(model-free). Use the kept traces to drive improvements to the agent, the memory
files, and the trace format itself.

---

## Layer 3 — system tests (visible browser, full build loop)

`system_tests.py` runs the agent end-to-end with a real model and Chromium —
slowest, highest-fidelity. Reserve for canaries, not per-prompt coverage.

```bash
python system_tests.py run --suite smoke --three-model   # plumbing + "player doesn't move" regressions
python system_tests.py run --suite pacman --yes          # slow (~15-30 min); skips the confirm prompt
```

The canonical regression check is **asteroids** (ship-direction
`vx = cos(angle)*speed` and irregular-polygon asteroids) — verify after any
change to retrieval, prompts, or the patch engine.
