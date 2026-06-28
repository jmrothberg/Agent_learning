# CLAUDE.md

Operational summary for coding agents working in this repo. **Also injected** into the game agent’s
system prompt (capped at 6 KB) — keep it actionable, not historical.

**Repo:** `jmrothberg/Agent_learning` (`origin`). Single source of truth — commit and push to `main`.
User updates: `./scripts/update.sh` or `git pull`.

**Other docs:** deep walkthrough → `README.md` · tests → `TEST.md` · tuning traps →
`FOR_NEXT_LLM.md` · debug traces → `HARNESS_DEBUG.md`

---

## What this project is

A coding agent driving a **local model** (qwen3.6 27B/35B via MLX in-process or Ollama) to write,
test, and iteratively fix **single-file HTML5 games** with real Chromium verification, in-process
Z-Image-Turbo sprites, Stable Audio, optional Wan2.2 cutscenes.

- `chat.py` — Textual TUI (default; visible Chromium)
- `coder.py` — headless CLI
- `memory/playbook.jsonl` — hand-curated rules retrieved at runtime (`memory.py`)

---

## Common commands

```bash
# Setup (once) — DEFAULT installs full GPU stack (MPS on Mac, CUDA on Linux) + sprites + audio:
./scripts/setup.sh

# Manual alternative only if you skip the script (easy to miss Chromium / MLX / GPU):
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-mlx.txt   # Apple Silicon MLX server only
env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python -m playwright install chromium
./scripts/install_diffuser.sh                   # torch + diffusers + requirements-diffuser.txt (sprites + sound deps)
TORCH_CUDA=121 ./scripts/install_diffuser.sh    # older NVIDIA only

# Rare: no CUDA/MPS host — skips torch/diffusers (~5 GB saved): ./scripts/setup.sh --no-gpu

# Verify the diffuser pipeline end-to-end (~2 minutes first run)
.venv/bin/python scripts/_smoke_doom.py

# Generate a video cutscene clip standalone (Wan2.2 — agent uses the same wrapper for <videos>)
.venv/bin/python scripts/generate_video.py --prompt "knight charges, cinematic" --out /tmp/clip.mp4
# add --image <png> for image-to-video (animate session key art). ~3 min per 4s clip on M3 Ultra.

# Run the TUI (recommended)
.venv/bin/python chat.py
# In the TUI: /help (commands) · /help topics · /help feedback-flows · /help critique · /help vlm-critique (static detail, tui_help.py)

# One-shot CLI
.venv/bin/python coder.py "build me a snake game with a wraparound board"
.venv/bin/python coder.py "snake" --max-iters 4 --best-of-n 1 --headless

# MLX backend (Apple Silicon — usually faster than Ollama at the same param count)
# Runs IN-PROCESS — no separate mlx_lm.server, no HTTP, no broken pipes.
# The model loads into this Python process's GPU VRAM on first request
# (~30-60s cold for a 27B mxfp8). Resolution order:
#   1. MLX_MODEL env var (path or HF id)
#   2. Single local MLX model in ~/MLX_Models / HF cache (auto-discovered)
MLX_MODEL=/Users/jonathanrothberg/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python coder.py "snake"
.venv/bin/python coder.py "snake" --backend mlx   # explicit; macOS defaults to MLX anyway
# Use LLM_BACKEND=auto (or --backend auto) to probe Ollama when MLX is down.
# DeepSeek-V4 quirk: still apply ./scripts/install_mlx_v4_fix.sh (it patches
# the installed mlx_lm package directly; in-process loads pick it up).
# Per-machine prefill chunk override: MLX_PREFILL_STEP_SIZE=N. Defaults
# auto-resolve from the model path: 512 if "flash" is in the name
# (DeepSeek-V4 Flash crashes mid-generation at >512), else 1024.

# Tests (pure-function, no model/GPU/browser — see TEST.md)
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_patches.py -v
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v

# System tests (visible browser; smoke fast, pacman slow — prompts unless --yes)
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes   # skip slow-run confirmation

# Prompt-library evaluation (memory/prompt_library.jsonl, the /games library)
# Layer 0 — memory coverage matrix, NO model, instant: which subsystem fires per prompt
.venv/bin/python eval/eval_prompts_plan.py --coverage
# Layer 1 — ONE planning turn per prompt vs the local model (no browser, no diffuser):
#   parses <plan>/<criteria>/<probes>/<assets>, checks each prompt's `expect` block.
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
.venv/bin/python eval/eval_prompts_plan.py --only 1            # one prompt
.venv/bin/python eval/eval_prompts_plan.py --names chess,doom  # by name
# Layer-0 assertions also run in CI: tests/test_prompt_library_coverage.py

# Seed-EDIT robustness eval (opt-in, local model) — PASS = materialized + bytes changed.
# Runs real edit turns over eval/fixtures/seed_tower_defense.html (browser=None, so it
# measures MATERIALIZATION not gameplay). Guards the "feedback produced no saved code"
# class (Fieldrunners trace 20260626_102307 iters 4-5).
MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py
.venv/bin/python eval/eval_seed_edits.py --only 1 --max-iters 2
# Parallel batch (one mlx_lm.server, N clients): see eval/batch_parallel.py
.venv/bin/python eval/eval_seed_edits.py --patch-only --max-iters 2   # skip Phase A on canvas seeds
# Stub regression banks (no model, run in the pytest suite): eval/golden_feedback_flows.jsonl
# (golden iter 2-3 art->patch / behavior-bug classification), eval/modality_scenarios.jsonl
# (beat-em-up suppression + non-combat key exclusion), eval/seed_edit_scenarios.jsonl
# (small-scope / orientation / size edit classification).

# Memory hygiene
.venv/bin/python scripts/forget_session.py --list
.venv/bin/python scripts/forget_session.py <session_id>
./scripts/clean_artifacts.sh --yes                # bulk wipe stale per-session artifacts
```

**Env vars that matter:**
- `LLM_BACKEND` — unset defaults to **`mlx` on macOS** (Apple GPU), else **`auto`**. Values: `auto` | `ollama` | `mlx`. `auto` probes both; if a local MLX model is discoverable AND Ollama also has a model loaded, MLX wins. Set `LLM_BACKEND=auto` on a Mac to allow Ollama-only fallback again.
- `OLLAMA_MODEL` / `CHAT_OLLAMA_MODEL` — explicit Ollama model override (else: detected from `/api/ps`, then first installed)
- `OLLAMA_HOST` — non-default Ollama daemon address
- `OLLAMA_KEEP_ALIVE` — Ollama model residency between chat turns. Default `-1` keeps the model loaded for the chat process lifetime to prevent 5-minute idle evictions during user-feedback pauses; set e.g. `10m` on tight-VRAM hosts.
- `MLX_MODEL` — explicit MLX model path or HF id. Loaded in-process via `mlx_lm.load`. If unset, the backend scans `~/MLX_Models/` / `MLX_MODELS_DIR` / HF cache and picks a single discoverable chat model (multi-match → first, with a "set MLX_MODEL to override" hint in `info.source`).
- `MLX_SERVER_URL` — when set (or `LLM_BACKEND=mlx-server`), talk to a running `mlx_lm.server` over HTTP instead of loading MLX in-process. Use for parallel batch testing: N `coder.py` clients → one server with continuous batching. See `eval/batch_parallel.py` and `TEST.md`.
- `MLX_MODELS_DIR` — `:`-separated list of additional dirs to scan for downloaded MLX models. Defaults: `~/MLX_Models`, `~/Models_MLX`, `~/.cache/huggingface/hub`, `/opt/mlx_models`.
- `MLX_PREFILL_STEP_SIZE` — chunk size for prompt eval. Per-model defaults applied automatically: **512** if the model path contains `flash` (DeepSeek-V4 Flash crashes mid-generation at >512 — observed 2026-05-15 DK trace), else **1024**. Env override always wins. Raise to `2048` on small models if you want a few % more throughput.
- `CODING_BOX_NUM_CTX` — context window (default **100000** — the speed/headroom sweet spot: observed coder prompts stay ~10-45K even deep into feedback sessions, so full history is retained while KV-cache/prefill cost stays low). In the TUI use **`/ctx`** to change it (raise toward 200K for very long sessions, lower on tight-VRAM hosts). Preload Ollama at the chosen size with `ollama run --ctx-size N <model>` to avoid a reload on first request. **Compaction is token-aware:** this value is the denominator for the pressure check — the lossy state-anchor compaction only fires once a coder prompt exceeds ~70% of it (`_COMPACT_PRESSURE`), not at a fixed message count. The old 32K default made that ratio exceed 1.0 within a couple of feedback turns, compacting every turn and shredding the playbook + prior user-feedback.
- `AGENT_NO_AUTO_OLLAMA_GPU_FIX` — set to `1` to disable automatic unload of tensor-split Ollama VRAM on `/new` (default: auto-fix on 48 GB-class GPUs).
- `MLX_MAX_TOKENS` — MLX output cap (default **131072**). Sized as a runaway-generation guard, not a working limit — measured peak across observed sessions is ~5.7K completion tokens, so 131K gives ~23× headroom. Raise it (e.g. 262144) only if a future model genuinely needs more output per turn.
- `ANTHROPIC_MAX_TOKENS` — Anthropic output cap (default 32768, the safe ceiling across Sonnet 4.6 / Opus 4.8). Override only if you hit per-model API limits.
- `DIFFUSION_MODELS_DIR` — root override for weights search (sprites + sounds layout); hidden **`~/.Diffusion_Models`** / **`~/.Models_Diffusers`** are tried before visible `~/…` siblings; HuggingFace cache fallback if absent.
- `DIFFUSER_CUDA_DEVICE` — force Z-Image / Stable-Audio onto a specific CUDA index; on the auto-pinned 4×48 GB Linux box the default is **GPU 0** so Ollama slots 1–3 stay dedicated to coder/critic/architect.
- `TORCH_CUDA` — CUDA version for `install_diffuser.sh` (`130` default, `121`/`124` for older GPUs)

---

## Architecture (the parts that span multiple files)

### The agent loop is async + event-driven

`GameAgent.run(goal) -> AsyncIterator[AgentEvent]` in `agent.py` is the heart. Three phases:

1. **Phase A — planning** (1 turn). Model emits `<plan>` + `<criteria>` + `<probes>` + optional `<assets>`. The user-turn prompt is built by `prompts_v1.plan_instruction(goal=...)`, which detects art / 3D modality keywords and injects directives accordingly.
2. **Phase B — build/iterate** (up to `max_iters`). Each iter: stream model reply → materialize via `patches.extract_patches`/`apply_patches` (preferred) OR full `<html_file>` rewrite → micro-probes (cheap, no browser) → Chromium load (skipped when `browser=None` — emits `iter_summary` with `test_skipped:no_browser`) → score against the model's own `<probes>`. Failed iters get a diagnose-then-fix prompt; clean iters get a "prefer `<done/>`" prompt. **`patch_only=True`** (requires `seed_file`): skip Phase A planning + phase-A asset gen — used by `eval/eval_seed_edits.py --patch-only`.
3. **Phase C — self-critique** (1 turn after `<done/>`). Model either `<confirm_done/>` or one more `<patch>`.

Drivers (`chat.py`, `coder.py`) construct `GameAgent`, wire a token callback, and consume the event stream. `chat.py` adds a Textual TUI with mid-stream user feedback queue (drained at every user-turn boundary by `_flush_user_injections`).

### Verification — multi-layered

`tools.py`: micro-probes (pre-Chromium) → `LiveBrowser.load_and_test` (Playwright: errors, RAF,
input smoke test, model `<probes>`, behavioral gates) → optional VLM critic. Failed probes or hard
soft_warnings flip `ok=False`. `<done/>` needs `_consecutive_clean_iters >= 2` (skipped-test iters
do not advance the streak). Gate list and debug workflow: **`HARNESS_DEBUG.md`**.

### Trace (LLM-only `.jsonl`)

High-frequency events (`stream_heartbeat`, `stream_progress`, `image_skipped`) are **not**
persisted. Stream metrics fold onto `iter_summary`; `failure_class` tags which layer needs the fix
(harness_bug / memory_gap / local_llm_limit). Timeline: `scripts/enrich_trace.py <session-id> --timeline`.
Tests: `tests/test_trace_diagnostics.py`.

### Patch engine (the quality lever)

`patches.py` defines a SEARCH/REPLACE format. The matcher cascades **exact → char-preserving normalized (smart quotes / dashes / unicode spaces) → whitespace-collapse → trimmed**. Cross-patch validation enforces uniqueness (>1 source match → ambiguous error) and non-overlap; surviving patches apply in **reverse source-order** so earlier offsets stay valid. `repair_reply` strips BOM + CRLF + internal fences, collapses malformed markers, and converts markdown `SEARCH:`/`REPLACE:` fenced pairs to `<patch>` blocks.

### Memory / Playbook (compounds across sessions)

`memory.py`:
- `GameMemory` — skeleton retrieval (`canvas_basic.html` bundled default + auto-promoted `won_<session_id>.html` from past wins) and mistake retrieval keyed by error signature.
- `Playbook` — JSONL of bullets with `helpful` / `harmful` counters. Retrieval is weighted Jaccard × quality multiplier `1 + 0.10·tanh(score/5)`. `stage="plan"` returns broader top-K including net-harmful entries; `stage="code"` drops bullets with score ≤ -2.
- `render_playbook_block(hits, mode="hybrid", char_budget=...)` — pi-mono "skills" pattern: top-3 with full bodies + remaining as ID-only index entries. Model emits `<lookup_bullet>id</lookup_bullet>` to fetch a body on demand; `agent._extract_and_queue_lookups` drains lookups into the next user turn (capped at 5/turn).
- `dedup_hits` (5-gram Jaccard ≥ 0.85) + `cap_hits_by_budget` run inside `render_playbook_block` by default.
- `memory/playbook.jsonl` is hand-curated; bullets are added/edited by you, not by an offline learner pass.

### Asset generation (Z-Image-Turbo, in-process)

`assets.py` is fully self-contained — no sibling-repo imports, no servers, no subprocess. The `ZImageTurboGenerator` class lazy-loads the Z-Image-Turbo `diffusers` pipeline into the chat.py process's GPU VRAM only on the first `<assets>` request (~30-60 s; ~14 s per image after). Pipeline:

1. `parse_assets_block(plan_reply)` extracts the JSON list (tolerant of fenced ```json wrappers AND of streams that truncated before `</assets>` — the truncated-list repair recovers complete entries).
2. `generate_assets`: for each spec, hash-cache → if miss, generate at native 768×768 → resize Lanczos → `_chroma_key_to_rgba` (sample 8 corner+edge points; if ≥6/8 agree on a dominant color, alpha-mask within tolerance) → save RGBA PNG. Per-asset stats stashed on `image_generator.last_stats`.
3. `render_asset_paths_block` builds the GENERATED ASSETS injection block for the first-build user message, with the literal `const ASSETS = {}; await img.decode(); ctx.drawImage(...)` loader pattern inline.

Siblings: `sounds.py` (Stable Audio Open, in-process, OGG ≤12 s) and `videos.py` (Wan2.2-TI2V-5B cutscene MP4s via the `<videos>` tag — runs `scripts/generate_video.py` as a SUBPROCESS: mlx-gen in `.venv-video/` on macOS, diffusers WanPipeline on Linux/CUDA; `image` field seeds image-to-video from a session asset so cutscenes match sprite style; ~3 min per 4 s clip, capped 4/turn; silent no-op when no backend).

### Prompt assembly (data-driven, not a string blob)

`prompts_v1.SYSTEM_PROMPT = build_system_prompt("{goal}")`. The function walks `ALL_FORMATS` (a list of `FormatSpec(name, snippet, guidelines)` per output tag), dedupes guidelines across enabled formats, and renders `<output-tags>` + `<guidelines>` + `<hard-rules>` + `<anti-patterns>`. To swap or disable a tag, edit the `FormatSpec` list — don't hand-edit the rendered prompt.

`plan_instruction(goal=...)` calls `_detect_art_intent(goal)` and `_detect_3d_intent(goal)`; matched modality keywords inject "ART INTENT DETECTED" / "3D INTENT DETECTED" callouts that escalate `<assets>` and three.js usage from "expected" to "required this turn". The keyword sets are intentionally **genre-free** — they describe rendering modality (`sprite`, `pixel`, `first-person`, `voxel`), not subject matter. Don't add genre names (`asteroids`, `doom`) here. The nudge **prose** lives in data (`memory/plan_nudges.jsonl`, loaded via `memory.load_plan_nudge(id)`); `prompts_v1` owns only the detectors + `{kws}`/`{logic_kws}`/`{mf_kws}` slot fill. TD-vs-brawler disambiguation is also data (`memory/visual_playtests.jsonl` `recipe.suppresses_nudges`, loaded via `memory.goal_suppresses_nudge(goal, nudge)`).

### Project-config injection

At session start `agent._read_project_config(cwd)` reads `AGENTS.md` then `CLAUDE.md` (this file) from the working directory, caps the concatenation at 6 KB, and appends as `<project-context>` at the END of the system prompt. Per-repo conventions get inherited automatically — but this file is data the agent will see, not just developer notes. Keep it useful to the model too.

### Compaction

Two-tier in `agent._prune_messages`:
- ≤ `_PRUNE_KEEP_RECENT_TURNS+1` (default 5): no-op.
- ≤ `_STRUCTURED_PRUNE_THRESHOLD` (default 14): per-turn HTML elision (replace `<html_file>` bodies with `[omitted: N bytes]`).
- `> 14`: replace messages 1..cutoff with one **state-anchor message** built deterministically by `_build_structured_summary` (Goal / Acceptance criteria / Probes / Progress / Diagnoses / Last test report / Files / Generated assets / Critical context). No extra LLM call. Asset paths survive compaction so feedback like "use the art you generated" is still actionable late in a session.

---

## Standing rules from the user (in `~/.claude/projects/-home-jonathan-Agent-learning/memory/`)

These bind decisions across sessions:

- **Tune the agent, not the model.** The model is fixed (qwen3.6:27b/35b). "Tune" means prompt / retrieval / playbook / probes / loop changes — never "try a bigger model".
- **No hardcoded genre / category lists.** Open-domain HTML/JS — retrieval, probes, skeletons stay genre-free. Modality keyword detectors (`_detect_art_intent`, `_detect_3d_intent`) describe rendering shape, not subject matter; that's why they pass the rule.
- **All code self-contained in Agent_learning.** No sys.path-injecting sibling repos (pre-existing `Colossal_Cave/diffusion_manager.py` was vendored into `assets.py` for this reason). External *data* at standard system paths (Ollama models, diffusion weights) is fine.
- **Visible Chromium, not headless.** The TUI default is `headless=False` so the user can watch the game. CLI keeps `--headless` for unattended runs.
- **Asteroids is the canonical regression check.** Ship-direction (`vx = cos(angle)*speed`) and round-asteroid (must be irregular polygons, not perfect circles) are the failure pair to verify after any change to retrieval / prompts / patches.
- **Never regenerate a pose frame to “fix” a dead one — and never let a dead-sprite warning block
  shipping.** img2img cannot change pose; txt2img replacement breaks consistency. Cosmetic findings
  → advisory `warnings` only; behavioral probes gate shipping.
- **Iterate autonomously.** Drive build/test/improve loops yourself; research-grounded techniques only.
- **Do not reintroduce aggressive early cutoffs.** Abort/repetition guards must latch on code emission,
  not token count or length alone. Preserve long first-build `<html_file>` streams unless trace-backed.

---

## Things to avoid

- Don't add features without first checking `tools.py` reports `ok=True` honestly. Fix harness signal
  before adding loop machinery.
- Don't bypass `<patch>` once `best.html` exists. Full `<html_file>` rewrites on a working game are a regression-amplification risk.
- Don't expand the system prompt with always-on rules that only matter sometimes. Use the per-format `FormatSpec.guidelines` (deduped) or a goal-keyword detector (`_detect_*_intent`) to make the rule conditional.
- Don't tighten repetition/deliberation/timeout abort thresholds without trace evidence proving
  first-build streams aren't cut mid-output.
- **Don't gate `ok` on dead-sprite warnings or suggest regenerating pose frames.** Cosmetic only;
  `_apply_dead_animation_check_to_report` uses the non-gating `warnings` channel. img2img cannot
  change pose; txt2img replacement breaks consistency — see standing rules above.
- **Never defer `<videos>` out of the build that plays them.** Untestable without real MP4s on disk.
- Don't commit generated artifacts. `.gitignore` excludes routine session outputs under `games/` (`*.html`, `*_assets/`, `*_sounds/`, traces, caches, `game-memory/`). **Exception:** TUI **`/goodgame`** copies the trio into `goodgame/` (tracked, not gitignored — no `git` commands from the agent). Legacy path: commit under `games/` + matching `!` negations (chess sample). `memory/playbook.jsonl` stays hand-curated. Use `scripts/clean_artifacts.sh` to wipe stale session artifacts.
