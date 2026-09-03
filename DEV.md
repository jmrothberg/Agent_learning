# DEV.md — maintainer reference (Cursor + humans)

Operational summary for coding agents maintaining this repo. **Not injected** into the
game-building LLM — see `AGENTS.md` §0 (who reads what).

**Repo:** `jmrothberg/Agent_learning` (`origin`). Single source of truth — commit and push to `main`.
User updates: `./scripts/update.sh` or `git pull`.

**Other docs:** router → `AGENTS.md` · human onboarding → `README.md` · tests → `TEST.md` ·
run commands → `eval/OPERATIONS.md` · tuning traps → `HARNESS_TUNING.md` · trace debug →
`HARNESS_DEBUG.md`

**New Cursor agent improving the harness:** read **`HARNESS_TUNING.md` § “New agent — harness
improvement”** first (read order, harness vs memory, canonical fix loop).

---

## What this project is

A coding agent driving a **local model** (qwen3.6 27B/35B via MLX in-process or Ollama) to write,
test, and iteratively fix **single-file HTML5 games** with real Chromium verification,
**FLUX2-klein** sprites on macOS (Z-Image-Turbo on Linux), Stable Audio, optional LTX-2.5 (Mac) /
Wan2.2 cutscenes.

- `chat.py` — Textual TUI (default; visible Chromium). `/wait` **ON** (`local_manual`) so each iter pauses for inspection. `/help` for slash commands.
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
- `MLX_SERVER_URL` — HTTP base for OpenAI-compatible MLX servers (`mlx_lm.server` **or oMLX**, e.g. `http://127.0.0.1:8000`). Enables `MLXServerBackend` continuous batching — **one model load, N clients**. Set **only for that session** (not in `.env`); see `eval/PARALLEL_MLX_TESTING.md` and README **DeepSeek-V4-Flash / oMLX**.
- `MLX_HOST` — legacy alias for `MLX_SERVER_URL` host:port
- `OMLX_SERVER_URL` — override oMLX base for Flash auto-start (default `http://127.0.0.1:8000`); used by `backend.ensure_omlx_server` / `omlx_default_endpoint`
- `OMLX_MODEL_DIR` — model scan dir for spawned `omlx serve` (else first existing `MLX_MODELS_DIR` entry, else `~/MLX_Models`)
- `MLX_MODELS_DIR` — `:`-separated extra model scan dirs (in-process `/list`; oMLX spawn uses first existing root)
- `MLX_PREFILL_STEP_SIZE` — prefill chunk (512 if path contains `flash`, else 1024) — in-process only
- `MLX_TOP_P` / `MLX_TOP_K` / `MLX_MIN_P` — MLX sampler (vendor coding preset; repetition penalty stays off)
- `MLX_MAX_TOKENS` — MLX output cap (default **131072**)
- `CODING_BOX_NUM_CTX` — context window (default **100000**); compaction fires near ~70% (`_COMPACT_PRESSURE`)
- `AGENT_COMPACT_TOKEN_CEILING` — absolute token ceiling for compaction (optional override)
- `AGENT_ENABLE_MEMORY_RELIEF` — set `0` to disable auto VRAM/RAM relief (default **on**). **MLX:** unload diffusers when free RAM &lt; `AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB` (default 64) or phys RAM ≤ `AGENT_MEMORY_RELIEF_MAX_PHYS_GB`; skips small MLX models (&lt; `AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB`, default 50 GB on disk). **Linux/Ollama+CUDA:** always unload in-process Z-Image/Stable-Audio after sprite/sound gen and before coder streams so the LLM is not forced into CPU offload on 2×24 GB boxes.
- `AGENT_MEMORY_RELIEF_MIN_AVAILABLE_GB` — trip MLX relief when available RAM falls below this (default 64)
- `AGENT_MEMORY_RELIEF_MAX_PHYS_GB` — also trip MLX relief after sprite/sound gen when physical RAM is at or below this (default **128**). Use on 96 GB Macs so Z-Image unloads before the MLX coder runs even if vm_stat still shows plenty of free pages.
- `AGENT_MEMORY_RELIEF_SMALL_MODEL_DISK_GB` — never unload for MLX coder models smaller than this on disk (default 50)
- `AGENT_VIDEO_MIN_FREE_VRAM_GB` — before Z-Image / Stable-Audio / Wan on Linux/Ollama, unload Ollama if the diffuser GPU has less free VRAM than this (default **12**). Skipped when a dedicated diffuser GPU is detected (4-GPU workstation). Next coder turn reloads via chat; does not start `ollama serve`.
- MLX `/model` hot-swap — when upsizing (or loading a large model while diffusers are resident), the harness auto-unloads the previous MLX weights and Z-Image/Stable-Audio before the next generation
- `AGENT_NO_AUTO_OLLAMA_GPU_FIX` — set `1` to disable auto Ollama VRAM unload on `/new`
- `ANTHROPIC_MAX_TOKENS` — Anthropic output cap (default 32768)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — cloud backends only
- `DIFFUSION_MODELS_DIR`, `DIFFUSER_CUDA_DEVICE`, `TORCH_CUDA` — sprite/sound GPU stack
- `VIDEO_ENGINE` — `ltx` | `wan` (TUI `/ltx` `/wan`). Unset on **macOS** = **LTX** when `~/Video_Models/LTX-2.5-MLX` **and** the `ltx-2-mlx` CLI exist; otherwise Wan. `/wan` opts out. Linux stays Wan.
- `VIDEO_MODELS_DIR` — `:`-separated video-weight scan dirs (default `~/Video_Models`; sibling of `~/MLX_Models`, not LLM weights)
- `VIDEO_LTX_MODEL`, `VIDEO_LTX_BIN` — LTX-2.5 weights dir / `ltx-2-mlx` CLI path
- `VIDEO_MODEL`, `VIDEO_VENV` — Wan2.2 cutscene subprocess overrides
- `AGENT_SIMULATOR` — set `1` to start `chat.py` in simulator mode (`/media off` / `/640`: 640×480 canvas, no sidecar sprites/sounds/videos). TUI `/media on` restores the full pipeline (next `/new`).
- `AGENT_JMR_PNG` — set `1` to start in `/640png` mode (JMR V1 640×480 walls + art pipeline writing `STEM-N.png` / `jmr:spr:N`). Wins over `AGENT_SIMULATOR`.

### oMLX (preferred MLX HTTP server on Mac)

[oMLX](https://omlx.ai/) 0.6.3+ is the supported way to serve **DeepSeek-V4-Flash**
(Vontra MXFP4 at `~/MLX_Models/DeepSeek-V4-Flash-0731-MXFP4-MLX`),
**GLM-5.3-Flash** (`glm5_next`, e.g. `~/MLX_Models/GLM-5.3-Flash-MLX-6bit`),
and **Qwen3.8-Flash-Next** (`qwen4_exp` + native MTP, e.g.
`~/MLX_Models/Qwen3.8-Flash-Next-MLX-8bit-MTP`).
Stock PyPI `mlx-lm` / in-process `mlx-vlm` 0.6.17 lack those load paths
(Flash-Next: 76 `language_model.mtp.*` tensors rejected).

| Concern | Setting |
|---------|---------|
| **TUI pick Flash** | `/model` / `/load` / `/launch` on DeepSeek-V4-Flash, GLM-5.3-Flash, or Qwen3.8-Flash-Next **auto-starts oMLX** (`backend.ensure_omlx_server`) and routes that session to `:8000`; GLM-5.2 / dense Qwen3.8-27B / MiniMax stay in-process |
| **Prompt cache (check first)** | `cache.hot_cache_max_size` ≠ `"0"` (e.g. `"32GB"`). Admin UI: **Memory Management → Memory Limit (In-Memory Hot Cache)** — **not** the CACHE panel. Default `"0"` disabled; enabling cut a repeated ~24K prompt **51s → 4.6s**. oMLX CLI rejects `"20%"` — use absolute GB in `settings.json` / `omlx serve` |
| Parallel agents | `LLM_BACKEND=mlx-server` + `MLX_SERVER_URL=http://127.0.0.1:8000` — one resident model, continuous batch |
| Idle unload / “server quit” | Global idle timeout **None**; **pin** the coder model; per-model TTL off |
| Long SSE prefills | SSE keepalive `chunk`; `caffeinate` / menu-bar app auto-restart |
| MTP speed | Per-model `mtp_enabled=true`, Flash-Next `mtp_num_draft_tokens=3` in `~/.omlx/model_settings.json` — **no** companion speed-patch repos on 0.5.4rc2+. Do **not** use `vlm_mtp_enabled` (that is Gemma’s external drafter) |
| GLM-5.2 coexistence | Separate weights folder; do not expect both ~390G + ~167G resident unless you have headroom |

`OMLX_SERVER_URL` overrides the default `http://127.0.0.1:8000`. Config file:
`~/.omlx/settings.json` (admin UI at `http://127.0.0.1:8000/admin`). Parallel
runbook: **`eval/PARALLEL_MLX_TESTING.md`**.

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

### Sprite atlas packing (`/640png`)

`materialize_jmr_png_sheets` (`assets.py`) packs related poses (shared name
prefix, e.g. `hero_idle` + `hero_walk1`) onto ONE `STEM-N.png` strip instead
of one PNG per pose. **16 sheets is a file cap, not a per-sheet frame cap** —
a strip just gets wider as frames are added; the real per-turn generation
cap is 64 poses (`JMR_PNG_MAX_FRAMES`). Games crop frames with 9-arg
`drawImage` / the injected `blitSpr` helper (`sx = frameIndex * cellW`).
Full rules: `assets.py` module docstring (`--- /640png atlas packing ---`)
and `HARNESS_TUNING.md` §"/640 simulator (JMR native)".

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
