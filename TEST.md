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
| Backend / context | `test_num_ctx.py`, `test_max_tokens_signal.py` |

Canonical regression after retrieval/prompt/patch changes: **asteroids** — ship direction
(`vx = cos(angle)*speed`) and irregular-polygon asteroids (not perfect circles).

## Layer 2 — prompt-library eval (local model, no browser)

```bash
.venv/bin/python eval/eval_prompts_plan.py --coverage   # instant memory matrix (also in CI)
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
```

One planning turn per curated prompt; traces under `games/eval-traces/`. See **`README.md`**.

## Layer 3 — system tests (full loop, visible browser)

```bash
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes
```

Slow; use for canaries, not per-prompt coverage. Battery: `memory/system_battery.jsonl`.
