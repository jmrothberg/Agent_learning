# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repo identity:** This checkout is **Coding Box Overlay** (`Agent_learning_overlay` on GitHub). Do not treat pushes as updates to `jmrothberg/Agent_learning` unless the user explicitly retargets `origin`.

---

## What this project is

A coding agent that drives a **local Ollama model** (qwen3.6:27b/35b is the working default) to write, test, and iteratively fix **single-file HTML5 games** with a real Chromium browser as the verifier. It ships:

- `chat.py` — Textual TUI (default entry point; visible Chromium beside the terminal)
- `coder.py` — headless CLI driver for unattended runs / scripting
- A self-contained Z-Image-Turbo sprite-generation pipeline (no server, in-process)
- A persistent cross-session **playbook** of HTML/JS rules of thumb (`memory/playbook.jsonl`, hand-curated seed bullets retrieved at runtime by `memory.py`)

The README has a deep walkthrough; this file is the operational summary.

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

# Run the TUI (recommended)
.venv/bin/python chat.py

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

# Tests (all pure-function, no model/Chromium calls; full suite ~12s)
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_patches.py -v        # one file
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v   # one test

# System tests (visible browser; smoke fast, pacman slow — prompts unless --yes)
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes   # skip slow-run confirmation

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
- `MLX_MODELS_DIR` — `:`-separated list of additional dirs to scan for downloaded MLX models. Defaults: `~/MLX_Models`, `~/Models_MLX`, `~/.cache/huggingface/hub`, `/opt/mlx_models`.
- `MLX_PREFILL_STEP_SIZE` — chunk size for prompt eval. Per-model defaults applied automatically: **512** if the model path contains `flash` (DeepSeek-V4 Flash crashes mid-generation at >512 — observed 2026-05-15 DK trace), else **1024**. Env override always wins. Raise to `2048` on small models if you want a few % more throughput.
- `CODING_BOX_NUM_CTX` — Ollama context window (default **100000** — enough for typical 6-iter game sessions; KV-cache scales linearly with ctx). In the TUI use **`/ctx`** to raise (e.g. `262k`, `full`) for long extension chats. Preload at the chosen size with `ollama run --ctx-size N <model>` to avoid a reload on first request.
- `AGENT_NO_AUTO_OLLAMA_GPU_FIX` — set to `1` to disable automatic unload of tensor-split Ollama VRAM on `/new` (default: auto-fix on 48 GB-class GPUs).
- `MLX_MAX_TOKENS` — MLX output cap (default **131072**). Sized as a runaway-generation guard, not a working limit — measured peak across observed sessions is ~5.7K completion tokens, so 131K gives ~23× headroom. Raise it (e.g. 262144) only if a future model genuinely needs more output per turn.
- `ANTHROPIC_MAX_TOKENS` — Anthropic output cap (default 32768, the safe ceiling across Sonnet 4.6 / Opus 4.7). Override only if you hit per-model API limits.
- `DIFFUSION_MODELS_DIR` — root override for weights search (sprites + sounds layout); hidden **`~/.Diffusion_Models`** / **`~/.Models_Diffusers`** are tried before visible `~/…` siblings; HuggingFace cache fallback if absent.
- `DIFFUSER_CUDA_DEVICE` — force Z-Image / Stable-Audio onto a specific CUDA index; on the auto-pinned 4×48 GB Linux box the default is **GPU 0** so Ollama slots 1–3 stay dedicated to coder/critic/architect.
- `TORCH_CUDA` — CUDA version for `install_diffuser.sh` (`130` default, `121`/`124` for older GPUs)

---

## Architecture (the parts that span multiple files)

### The agent loop is async + event-driven

`GameAgent.run(goal) -> AsyncIterator[AgentEvent]` in `agent.py` is the heart. Three phases:

1. **Phase A — planning** (1 turn). Model emits `<plan>` + `<criteria>` + `<probes>` + optional `<assets>`. The user-turn prompt is built by `prompts_v1.plan_instruction(goal=...)`, which detects art / 3D modality keywords and injects directives accordingly.
2. **Phase B — build/iterate** (up to `max_iters`). Each iter: stream model reply → materialize via `patches.extract_patches`/`apply_patches` (preferred) OR full `<html_file>` rewrite → micro-probes (cheap, no browser) → Chromium load → score against the model's own `<probes>`. Failed iters get a diagnose-then-fix prompt; clean iters get a "prefer `<done/>`" prompt.
3. **Phase C — self-critique** (1 turn after `<done/>`). Model either `<confirm_done/>` or one more `<patch>`.

Drivers (`chat.py`, `coder.py`) construct `GameAgent`, wire a token callback, and consume the event stream. `chat.py` adds a Textual TUI with mid-stream user feedback queue (drained at every user-turn boundary by `_flush_user_injections`).

### Verification — multi-layered

`tools.py` runs the harness:
- **Pre-Chromium micro-probes** (`run_micro_probes`): structural sanity (HTML completeness, `<script>` presence, bracket balance, elision sentinels, API-method allowlist for canvas2d / AudioContext / canvas-elt receivers). Cheap; rejects truncated streams before paying for a 3 s browser load.
- **Chromium load via Playwright async** (`LiveBrowser.load_and_test`). Captures `console.error` (→ `console_errors`), `pageerror` (→ `page_errors`), canvas state, RAF firing, listener counts, frozen-canvas check, automated input smoke test, screenshot per iter, model-proposed `<probes>` evaluated in the page context. Launches with `--allow-file-access-from-files` + `--disable-web-security` so `drawImage(<file:// PNG>)` doesn't taint the canvas.
- **Probes gate `ok=True`.** A failed probe (or any `soft_warning`) flips `report["ok"] = False`. Tainted-canvas probe errors are auto-downgraded to passes. `<done/>` requires `_consecutive_clean_iters >= 2`.

### Patch engine (the quality lever)

`patches.py` defines a SEARCH/REPLACE format. The matcher cascades **exact → char-preserving normalized (smart quotes / dashes / unicode spaces) → whitespace-collapse → trimmed**. Cross-patch validation enforces uniqueness (>1 source match → ambiguous error) and non-overlap; surviving patches apply in **reverse source-order** so earlier offsets stay valid. `repair_reply` strips BOM + CRLF + internal `​`​`​`html` fences before parsing.

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

### Prompt assembly (data-driven, not a string blob)

`prompts_v1.SYSTEM_PROMPT = build_system_prompt("{goal}")`. The function walks `ALL_FORMATS` (a list of `FormatSpec(name, snippet, guidelines)` per output tag), dedupes guidelines across enabled formats, and renders `<output-tags>` + `<guidelines>` + `<hard-rules>` + `<anti-patterns>`. To swap or disable a tag, edit the `FormatSpec` list — don't hand-edit the rendered prompt.

`plan_instruction(goal=...)` calls `_detect_art_intent(goal)` and `_detect_3d_intent(goal)`; matched modality keywords inject "ART INTENT DETECTED" / "3D INTENT DETECTED" callouts that escalate `<assets>` and three.js usage from "expected" to "required this turn". The keyword sets are intentionally **genre-free** — they describe rendering modality (`sprite`, `pixel`, `first-person`, `voxel`), not subject matter. Don't add genre names (`asteroids`, `doom`) here.

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
- **Iterate autonomously.** Drive build/test/improve loops yourself; research-grounded techniques only.
- **Do not reintroduce aggressive early cutoffs.** Changes to repetition / deliberation / timeout guards must be trace-backed and must preserve long first-build `<html_file>` completion (no premature guard aborts as the default behavior).

---

## Things to avoid

- Don't add features without first checking `tools.py` reports `ok=True` honestly. The harness signal must be right before adding more agent-loop machinery — see commit `044edf4` and `games/traces/using-great-graphics-that-you_…` for what happens when it isn't.
- Don't bypass `<patch>` once `best.html` exists. Full `<html_file>` rewrites on a working game are a regression-amplification risk.
- Don't expand the system prompt with always-on rules that only matter sometimes. Use the per-format `FormatSpec.guidelines` (deduped) or a goal-keyword detector (`_detect_*_intent`) to make the rule conditional.
- Don't tighten repetition/deliberation/timeout abort thresholds without concrete trace evidence and a regression run proving rich first-build streams are not cut mid-output.
- Don't commit generated artifacts. `.gitignore` excludes `games/<*>.html`, `games/snapshots/`, `games/traces/`, `games/_asset_cache/`, `games/*_assets/`, `games/_smoke/`, `games/game-memory/`, and other `memory/*` except **`memory/playbook.jsonl`** (hand-curated). Use `scripts/clean_artifacts.sh` to wipe stale session artifacts.
