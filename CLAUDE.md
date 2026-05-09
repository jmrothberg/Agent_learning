# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project is

A coding agent that drives a **local Ollama model** (qwen3.6:27b/35b is the working default) to write, test, and iteratively fix **single-file HTML5 games** with a real Chromium browser as the verifier. It ships:

- `chat.py` — Textual TUI (default entry point; visible Chromium beside the terminal)
- `coder.py` — headless CLI driver for unattended runs / scripting
- A self-contained Z-Image-Turbo sprite-generation pipeline (no server, in-process)
- A persistent cross-session **playbook** of HTML/JS rules of thumb that compounds via an offline learner

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
mlx_lm.server --model /Users/jonathanrothberg_1/MLX_Models/Qwen3.6-27B-mxfp8 --port 8080
.venv/bin/python coder.py "snake" --backend mlx   # explicit; macOS defaults to MLX anyway
# Use LLM_BACKEND=auto (or --backend auto) to probe Ollama when MLX is down.

# Tests (all pure-function, no model/Chromium calls; full suite ~12s)
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_patches.py -v        # one file
.venv/bin/python -m pytest tests/test_patches.py::test_apply_smart_quote_match -v   # one test

# Tune battery (compare prompt/playbook changes A/B)
python tune.py run                                # quick: max_iters=2, best_of_n=1
python tune.py run --full --prompt-version v1 --auto-learn
python tune.py diff baseline_v0 v1_run            # per-test pass/fail deltas
python tune.py why <run_id> <test_name>           # postmortem

# Offline learner (Reflector + Curator over traces)
python learner.py walk                            # one-line summary per past session
python learner.py reflect games/traces/           # propose deltas (no writes)
python learner.py apply games/traces/             # propose AND write to playbook.jsonl

# Memory hygiene
.venv/bin/python scripts/forget_session.py --list
.venv/bin/python scripts/forget_session.py <session_id>
./scripts/clean_artifacts.sh --yes                # bulk wipe stale per-session artifacts
```

**Env vars that matter:**
- `LLM_BACKEND` — unset defaults to **`mlx` on macOS** (Apple GPU), else **`auto`**. Values: `auto` | `ollama` | `mlx`. `auto` probes both; if both have a model loaded, MLX wins (faster on Apple Silicon). Set `LLM_BACKEND=auto` on a Mac to allow Ollama-only fallback again.
- `OLLAMA_MODEL` / `CHAT_OLLAMA_MODEL` — explicit Ollama model override (else: detected from `/api/ps`, then first installed)
- `OLLAMA_HOST` — non-default Ollama daemon address
- `MLX_MODEL` — explicit MLX model id override (else: `--model X` arg of running `mlx_lm.server`, then `/v1/models[0]`)
- `MLX_HOST` — non-default mlx_lm.server address (default `http://127.0.0.1:8080`)
- `CODING_BOX_NUM_CTX` — Ollama context window (default 32768; supports 128K+ on most local models). MLX has no equivalent; uses model native context.
- `DIFFUSION_MODELS_DIR` — override Z-Image-Turbo weights search path (Linux: `~/Models_Diffusers`, Mac: `~/Diffusion_Models`, then HuggingFace fallback)
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

`learner.py` reads completed traces and proposes bullet deltas (Reflector) which the Curator merges deterministically into `playbook.jsonl`.

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
- **Iterate autonomously.** Drive build/test/improve loops yourself; tune battery is the alignment check; research-grounded techniques only.

---

## Things to avoid

- Don't add features without first checking `tools.py` reports `ok=True` honestly. The harness signal must be right before adding more agent-loop machinery — see commit `044edf4` and `games/traces/using-great-graphics-that-you_…` for what happens when it isn't.
- Don't bypass `<patch>` once `best.html` exists. Full `<html_file>` rewrites on a working game are a regression-amplification risk.
- Don't expand the system prompt with always-on rules that only matter sometimes. Use the per-format `FormatSpec.guidelines` (deduped) or a goal-keyword detector (`_detect_*_intent`) to make the rule conditional.
- Don't commit generated artifacts. `.gitignore` already excludes `games/<*>.html`, `games/snapshots/`, `games/traces/`, `games/_asset_cache/`, `games/*_assets/`, `games/_smoke/`, `games/memory/`. Use `scripts/clean_artifacts.sh` to wipe them.
