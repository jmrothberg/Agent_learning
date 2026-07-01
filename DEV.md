# DEV.md — maintainer reference (Cursor + humans)

Operational summary for coding agents maintaining this repo. **Not injected** into the
game-building LLM — see `AGENTS.md` §0 (who reads what).

**Repo:** `jmrothberg/Agent_learning` (`origin`). Single source of truth — commit and push to `main`.
User updates: `./scripts/update.sh` or `git pull`.

**Other docs:** router → `AGENTS.md` · human onboarding → `README.md` · tests → `TEST.md` ·
run commands → `eval/OPERATIONS.md` · tuning traps → `FOR_NEXT_LLM.md` · trace debug →
`HARNESS_DEBUG.md`

---

## What this project is

A coding agent driving a **local model** (qwen3.6 27B/35B via MLX in-process or Ollama) to write,
test, and iteratively fix **single-file HTML5 games** with real Chromium verification, in-process
Z-Image-Turbo sprites, Stable Audio, optional Wan2.2 cutscenes.

- `chat.py` — Textual TUI (default; visible Chromium)
- `coder.py` — headless CLI (`--backend {auto,ollama,mlx,mlx-server}`)
- `memory/playbook.jsonl` — hand-curated rules retrieved at runtime (`memory.py`)

---

## Common commands

Full pytest map → **`TEST.md`**. Batch / overnight → **`eval/OPERATIONS.md`**.

```bash
./scripts/setup.sh
.venv/bin/python scripts/_smoke_doom.py
.venv/bin/python chat.py
.venv/bin/python coder.py "snake" --max-iters 4 --headless
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python coder.py "snake"
.venv/bin/python -m pytest tests/ -q
python system_tests.py run --suite smoke --three-model
.venv/bin/python eval/eval_prompts_plan.py --coverage
MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py --patch-only --max-iters 2
.venv/bin/python scripts/forget_session.py --list
./scripts/clean_artifacts.sh --yes
```

MLX upgrades: MiniMax-M3 (`minimax_m3.py` copy after mlx-lm upgrade), GLM-5.2
(`./scripts/install_mlx_glm52_fix.sh`). See `README.md` quick start for full commands.

---

**Env vars that matter:**
- `LLM_BACKEND` — unset defaults to **`mlx` on macOS**, else **`auto`**. Values: `auto` | `ollama` | `mlx` | `mlx-server` | `openai` | `anthropic`
- `OLLAMA_MODEL` / `CHAT_OLLAMA_MODEL` — explicit Ollama model override
- `OLLAMA_HOST` — non-default Ollama daemon address
- `OLLAMA_KEEP_ALIVE` — default `-1` (keep loaded for chat process lifetime)
- `MLX_MODEL` — explicit MLX model path or HF id (in-process `mlx_lm.load`)
- `MLX_SERVER_URL` — HTTP to `mlx_lm.server` (parallel batch testing; see `eval/batch_parallel.py`)
- `MLX_MODELS_DIR` — `:`-separated extra model scan dirs
- `MLX_PREFILL_STEP_SIZE` — prefill chunk (512 if path contains `flash`, else 1024)
- `MLX_TOP_P` / `MLX_TOP_K` / `MLX_MIN_P` — MLX sampler (vendor coding preset; repetition penalty stays off)
- `MLX_MAX_TOKENS` — MLX output cap (default **131072**)
- `CODING_BOX_NUM_CTX` — context window (default **100000**); compaction fires near ~70% (`_COMPACT_PRESSURE`)
- `AGENT_COMPACT_TOKEN_CEILING` — absolute token ceiling for compaction (optional override)
- `AGENT_ENABLE_MEMORY_RELIEF` — set `0` to disable auto-unload of diffusers before video / after asset batches (default **on** when free RAM &lt; `AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB`, default 64). Skips small MLX models (&lt; `AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB`, default 50 GB on disk).
- `AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB` — trip relief when available RAM falls below this (default 64)
- `AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB` — never unload for coder models smaller than this on disk (default 50)
- `AGENT_NO_AUTO_OLLAMA_GPU_FIX` — set `1` to disable auto Ollama VRAM unload on `/new`
- `ANTHROPIC_MAX_TOKENS` — Anthropic output cap (default 32768)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — cloud backends only
- `DIFFUSION_MODELS_DIR`, `DIFFUSER_CUDA_DEVICE`, `TORCH_CUDA` — sprite/sound GPU stack
- `VIDEO_MODEL`, `VIDEO_VENV` — Wan2.2 cutscene subprocess overrides

---

## Architecture (the parts that span multiple files)

### The agent loop is async + event-driven

`GameAgent.run(goal)` in `agent.py` orchestrates phase methods; logic lives in **mixins** — see
**`AGENTS.md` §1b**; do not grow inline logic in `run()`.

Three phases: **A** plan → **B** build/iterate (patch → micro-probes → Chromium) → **C** self-critique.

Drivers: `chat.py`, `coder.py`. Verification: `tools.py`. Modality detectors (3D / wireframe / FPS
nav): `modality.py` — shared by `prompts_v1.py` and `memory.py` (no import cycle). Patches:
`patches.py`. Prompts: `prompts_v1.py`.

### Trace (LLM-only `.jsonl`)

`failure_class` on `iter_summary`: `harness_bug` | `memory_gap` | `local_llm_limit`. Timeline:
`scripts/enrich_trace.py <path-or-stem> --timeline` — TUI traces: substring under `games/traces/`;
tune batch: **full path** under `games/tune_serial10/run_XX/traces/`. See **`HARNESS_DEBUG.md`**.

### Memory / Playbook

`GameMemory` — skeleton retrieval (runtime fallback **`canvas_basic_v2.html`**; bundled name
`canvas_basic.html` in constants). `Playbook` — weighted Jaccard on `memory/playbook.jsonl`.
Standing game constraints belong in playbook bullets, not a root config file.

### Compaction

In `agent_compaction.py` (`_prune_messages`): ≤5 turns no-op; ≤14 HTML elision; >14 or ~70% of
`num_ctx` → state-anchor summary via `_build_structured_summary` in `agent_prompts.py`.

---

## Standing rules

- **Tune the agent, not the model** — prompts / retrieval / gates / memory
- **Genre-free in code** — modality detectors describe rendering shape, not subject matter
- **All code self-contained in Agent_learning**
- **Visible Chromium by default** (TUI); CLI `--headless` for unattended runs
- **Asteroids regression** after retrieval/prompt/patch changes
- **Never regenerate pose frames** — cosmetic sprite warnings are advisory only
- **Do not reintroduce aggressive early cutoffs** — latch on code emission, not token count alone

---

## Things to avoid

- Fix harness signal before adding loop machinery
- Don't bypass `<patch>` once `best.html` exists
- Don't gate `ok` on dead-sprite warnings
- Never defer `<videos>` out of the build that plays them
- Don't commit generated artifacts under `games/` — use `/goodgame` or `scripts/clean_artifacts.sh`
