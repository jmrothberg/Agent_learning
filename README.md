# Coding Box — Local HTML Game Agent

A specialist agent that drives a **small local LLM** (Ollama or MLX
in-process) to write, test, and iteratively fix **single-file HTML5
games** with a real Chromium browser as the verifier and a real local
diffusion pipeline as the asset generator. Everything runs on your
machine. No cloud calls in the main loop. No server processes between
you and your GPU.

The thesis in one line: **a small validated model beats a large
unvalidated one — and an agent that learns from every session beats a
static prompt.** Every reply is parsed, every patch is applied with a
4-tier match cascade, every iter is loaded in real Chromium, every
generated sprite is alpha-keyed, every clean turn is preserved, every
regression is flagged, every session feeds a playbook the agent reads
back the next time you launch it.

The asset pipeline is fully self-contained: **Z-Image-Turbo** for
sprites (txt2img, 768×768 native, downscaled and chroma-keyed to RGBA)
and **Stable Audio Open** for sounds (OGG, 0.2–12 s), with optional
**SD-Turbo img2img** for animation frame chains via the `from_image`
field. None of those touch the network at runtime once weights are
cached. A cross-session **asset library** under `memory/` lets
admitted assets compound across sessions exactly the way the playbook
does.

**Remote:** https://github.com/jmrothberg/Agent_learning

---

## Contents

- [What this is, in one screen](#what-this-is-in-one-screen)
- [Compared to other coding agents](#compared-to-other-coding-agents)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Architecture — the parts that span multiple files](#architecture--the-parts-that-span-multiple-files)
  - [The agent loop](#the-agent-loop-async-and-event-driven)
  - [Patch engine](#patch-engine)
  - [Memory and playbook](#memory-and-playbook-compounds-across-sessions)
  - [Verification harness](#verification-harness-multi-layered)
  - [Visual-progress judge (local VLM)](#visual-progress-judge-local-vlm)
  - [Sprite generation](#sprite-generation-z-image-turbo)
  - [Animation frame chains](#animation-frame-chains-sd-turbo-img2img)
  - [Sound generation](#sound-generation-stable-audio-open)
  - [Cross-session asset library](#cross-session-asset-library)
  - [Prompt assembly](#prompt-assembly-data-driven-not-a-string-blob)
  - [Compaction](#compaction-two-tier-state-anchor)
  - [Project-config injection](#project-config-injection)
- [TUI and CLI reference](#tui-and-cli-reference)
- [Tuning rig and playbook commands](#tuning-rig-and-playbook-commands)
- [Standing rules and design constraints](#standing-rules-and-design-constraints)
- [Major future improvements](#major-future-improvements)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [License](#license)

---

## What this is, in one screen

The product is a coding agent narrowly specialized for **playable
single-file HTML5 games with visual and audible feedback**, that runs
entirely on a local machine. Three drivers ship in the repo:

- `chat.py` — Textual TUI with a visible Chromium beside the terminal.
  Default. Mid-stream user feedback queue, slash commands, asset
  picker, screenshot review, model swap, playbook viewer.
- `coder.py` — Headless CLI for unattended runs. Same agent, same
  loop, no UI.
- `tune.py` — A/B battery rig. Runs the agent against a curated set
  of goals under different prompt versions / feature flags, compares
  pass/fail deltas per goal, postmortems failing tests.

The core loop is event-driven and yields a stream of `AgentEvent`
objects (`info`, `tokens`, `iter_test`, `done`, …). The drivers
consume the stream; they do not contain agent logic of their own.

**Three signals gate iter quality.** Each iter, the harness reads:

1. **Pre-Chromium micro-probes** ([tools.py](tools.py)): HTML structure,
   bracket balance, API allowlist (canvas2d, AudioContext,
   HTMLCanvasElement, HTMLAudioElement), elision sentinels,
   repetition-collapse loops, asset-path existence on disk,
   *unused-asset detection* (PNG/OGG generated but never referenced).
2. **Chromium harness** (Playwright async, visible by default in the
   TUI): console errors, page errors, RAF firing, canvas state, blank
   detection, listener counts, automated input smoke test, model-
   proposed JS probes, screenshot, frozen-canvas check, audio-events
   recording shim (`window.__audioEvents`).
3. **Visual-progress judge** ([vision_judge.py](vision_judge.py)): auto-on
   when a local MLX-VLM is discoverable on disk
   ([backend.py](backend.py) `discover_local_vlm()`); compares previous
   vs. current screenshot, reports `PROGRESS: yes|no|unclear` plus a
   one-sentence "what's still missing" note that flows into the next
   user turn as coaching. A high pixel-delta paired with `PROGRESS:
   no` escalates to REGRESSION SUSPECTED. Never silently falls back to
  a cloud model — the user's `/check <N|model>` slash command is
  the explicit path for cloud-backed judging (Anthropic or OpenAI).

**No new asset generators are introduced.** Z-Image-Turbo for sprites
and Stable Audio Open for sounds stay locked. The cross-session asset
library compounds them; nothing replaces them.

---

## Compared to other coding agents

This is a specialist tool. The comparison table below lists what each
agent is *built for*, not which is "better." For general coding,
prefer the generalists; for local HTML games with visual + audible
quality gates, prefer this.

| Agent | Built for | Local-only | Browser verifier | Asset gen (sprites + sounds) | Cross-session learning |
|---|---|---|---|---|---|
| **This repo** | Single-file HTML5 games | yes | yes (Playwright Chromium) | yes (Z-Image-Turbo + Stable Audio + img2img) | yes (playbook + asset library) |
| [opencode](https://github.com/anomalyco/opencode) (TS, MIT) | General coding assistant | optional | no | no | no |
| [Claude Code](https://www.claude.com/product/claude-code) | General coding via Anthropic API | no (cloud) | no | no | no |
| [Aider](https://aider.chat) | General Git-aware coding | optional | no | no | no |
| [pi-mono coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) | General coding (TS, multi-provider, function-call native) | optional | no | no | no |
| [OpenCoder](https://opencoder-llm.github.io/) | Training recipe for a code LLM (not an agent) | — | — | — | — |

What was borrowed from where:

- **From pi-mono**: the SEARCH/REPLACE patch matcher cascade, per-format
  prompt assembly with deduped guidelines, structured compaction with
  state-anchor messages, "prescriptive errors" (tell the model EXACTLY
  what was wrong, not "syntax error").
- **From OpenCoder**: quality-ranked + deduped + budgeted retrieval
  (Jaccard × `1 + 0.10·tanh(score/5)` quality multiplier), the
  two-stage plan/code distinction that maps to our `<plan>` /
  `<patch>` phases, the educational-execution-filter pattern that maps
  to micro-probes.
- **From opencode** (the comparison target for the most recent moat-
  widening pass): the idea of static analysis *before* the browser
  (we don't run a real LSP — we run a regex-based receiver-method
  allowlist that catches the highest-frequency hallucinations).

What is **not** in this repo and is intentionally not planned:

- A client/server architecture (opencode has one — we explicitly want
  local-only).
- Provider plugin sprawl. The backends are: Ollama, MLX in-process,
  optional cloud providers (Anthropic/OpenAI) with explicit calls only.
- A generalist coding mode. The system prompt, the retrieval index,
  the probes, the asset pipeline, and the playbook are all tuned for
  one thing: **a playable HTML5 game in a single file**.

---

## Prerequisites

- **Python 3.10+**, macOS or Linux Ubuntu.
- **Ollama** with one chat model loaded, *or* an **MLX** model
  directory under `~/MLX_Models/` (auto-discovered) on Apple Silicon.
  Working defaults: qwen3.6 27B/35B, DeepSeek-V4, GLM-5.1, MiniMax-M2.
- **Playwright Chromium**: installed by `scripts/setup.sh`.
- **Z-Image-Turbo + SD-Turbo + Stable Audio Open**: installed by
  `scripts/install_diffuser.sh` (~5 GB of weights total; auto-downloads
  on first use). Skip this only if you don't want generated assets.
- A GPU is strongly recommended. Apple Silicon: MPS via the MLX/MLX-VLM
  in-process backend and torch/diffusers MPS for the asset pipeline.
  Linux NVIDIA: CUDA 13.0 default, `TORCH_CUDA=121` for older cards.

---

## Quick start

```bash
# One-time setup (installs Python deps, Playwright Chromium, GPU stack).
./scripts/setup.sh

# Optional: pre-generate the 8-sound stock SFX pack so first-iter
# games are audible without paying audio-diffuser cost.
.venv/bin/python scripts/build_stock_sounds.py

# Smoke-test the asset pipeline end-to-end (~2 min cold).
.venv/bin/python scripts/_smoke_doom.py

# Run the TUI (recommended). Visible Chromium opens beside the terminal.
.venv/bin/python chat.py

# One-shot CLI run.
.venv/bin/python coder.py "build me a snake game with a wraparound board"
.venv/bin/python coder.py "asteroids" --max-iters 4 --best-of-n 1 --headless
```

### Backend selection

`LLM_BACKEND` controls the chat model. Default on macOS is **`mlx`**
(Apple GPU, in-process); elsewhere it's **`auto`** (probe both Ollama
and MLX, prefer MLX when available).

```bash
# Pin a specific MLX model.
MLX_MODEL=/Users/me/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python coder.py "snake"

# Force Ollama.
LLM_BACKEND=ollama OLLAMA_MODEL=qwen3.6:35b .venv/bin/python coder.py "asteroids"

# Disable the local visual judge entirely (otherwise auto-on if a
# local VLM is discoverable on disk).
VISION_JUDGE=0 .venv/bin/python coder.py "snake"

# Optional: verify cloud backends (explicit API-key check only).
.venv/bin/python scripts/smoke_cloud_backends.py
.venv/bin/python scripts/smoke_cloud_backends.py openai
.venv/bin/python scripts/smoke_cloud_backends.py anthropic
```

### Tests

```bash
.venv/bin/python -m pytest tests/ -q                          # full suite (~2 s, no GPU/browser)
.venv/bin/python -m pytest tests/test_patches.py -v           # single file
```

---

## Architecture — the parts that span multiple files

### The agent loop (async and event-driven)

[`GameAgent.run(goal) -> AsyncIterator[AgentEvent]`](agent.py) in
`agent.py` is the heart. Three phases:

**Phase A — planning (1 turn).** The model emits `<plan>`,
`<criteria>` (acceptance bar), `<probes>` (executable JS checks the
verifier runs each iter), and optional `<assets>` + `<sounds>` (asset
spec lists). The user-turn prompt is built by
`prompts_v1.plan_instruction(goal=...)` which detects art and 3D
modality keywords and escalates `<assets>` / three.js usage from
"expected" to "required this turn" when matched. Modality keywords are
intentionally genre-free.

**Phase B — build/iterate (up to `max_iters`).** Each iter: stream the
model's reply → materialize via `<patch>` (preferred) or `<html_file>`
rewrite → run micro-probes (pre-Chromium) → if clean, load in Chromium
and score the report → coach the model with verifier feedback for the
next turn. Failed iters get a diagnose-then-fix prompt; clean iters
get a "prefer `<done/>`" prompt. The vision judge runs after the
screenshot (when a local VLM is discoverable) and contributes coaching
about what's still visibly wrong.

**Phase C — self-critique (1 turn after `<done/>`).** The model either
emits `<confirm_done/>` (ship) or one more `<patch>` (one last fix).
On VLM-capable backends the latest screenshot is attached and the
critique prompt names the visual things to look for.

Drivers (`chat.py`, `coder.py`, `tune.py`) construct `GameAgent`, wire
a token callback, and consume the event stream. The TUI adds a
mid-stream user feedback queue drained at every user-turn boundary by
`_flush_user_injections`.

**Scoped-turn hard enforcement (small-local-LLM guardrails).** When
feedback explicitly locks scope ("only X", "no code changes"), the
agent now persists a deterministic one-turn contract and rejects
non-compliant replies before materialization:

- **`single_patch` mode** (default for behavior/size tweaks): one small
  `<patch>` only; no `<html_file>` rewrite; max 1 patch block.
- **`media_only` mode** (style-only redraw wording): `<assets>`/`<sounds>`
  only, with existing session names only (new keys are dropped with a
  concise correction line).
- **Format-only gate:** scoped replies must start with the required tag
  (no prose preamble, no `<think>` leakage).
- **Scoped verification hook:** behavior-scoped turns require one compact
  probe signal tied to the requested behavior; missing/failing scoped
  checks are surfaced as a test issue and block `ok=True`.
- **Coaching suppression:** while strict scoped lock is active, unrelated
  agent coaching is withheld so the model sees one objective.

**Done detection.** `<done/>` requires
`self._consecutive_clean_iters >= self._min_clean_streak_to_ship`
(default 2) AND the latest report's `ok=True`. Any failure resets the
counter to 0. The iter budget can grant one bonus iter on auto-revert
to avoid punishing rollback work.

### Patch engine

[`patches.py`](patches.py) defines a SEARCH/REPLACE format with a
**four-tier matching cascade**:

1. **Exact** substring match in the file.
2. **Character-preserving normalized** (smart quotes → ASCII, en/em
   dashes → hyphen, NBSP and unicode spaces → ASCII space; 1:1 mapping
   preserves offsets).
3. **Whitespace-collapse** (runs of spaces/tabs → single).
4. **Trim** (strip leading/trailing whitespace both sides).

**Cross-patch validation** rejects ambiguity (>1 source match → "add
context") and overlap (spans cross → "merge edits N and M"). Surviving
patches apply in **reverse source-order** so earlier offsets stay valid.

`repair_reply` strips BOM, CRLF, and internal ```html fences before
parsing. The error messages are prescriptive: the model is told which
patch number caused the problem and what to change to fix it.

### Memory and playbook (compounds across sessions)

[`memory.py`](memory.py):

- **`GameMemory`** — skeleton retrieval and mistake retrieval.
  - **Premium Default Skeletons (Autobootstrapped on boot)**: The system provides 11 generic, high-fidelity scaffolds in `memory/skeletons/` designed to give local models a perfect first-build template:
    - `canvas_basic.html`: Clean 2D canvas with DPR scaling, frame loops, and window keyboard handlers.
    - `canvas_basic_v2.html`:Denser, bug-hardened scaffold pre-empting blur-clearing, focus-loss, and cleanup.
    - `canvas_3d_basic.html`: Full 3D perspective setup utilizing CDN Three.js, lights, camera, and aspect-ratio fits.
    - `canvas_grid_basic.html`: Continuous tile-aligned corridor movement and corner snapping (e.g., Pac-Man, Sokoban).
    - `canvas_platformer_basic.html`: Gravity jumps, vertical ladder alignments, climbing, and platform landings (e.g., Donkey Kong).
    - `canvas_scrolling_basic.html`: Cam viewport horizontal scrolling and parallax backgrounds (e.g., Defender).
    - `canvas_mode7_basic.html`: Scanline perspective projection texture mapping for rotatable tracks (e.g., Mario Kart).
    - `canvas_crawler_basic.html`: Top-down dungeon rooms, spawner pools, wall-sliding, and multi-player bounding clamps (e.g., Gauntlet).
    - `canvas_mobile_basic.html`: Pointer Events touch joystick, tap-buttons, and mobile aspect letterboxing (e.g., iOS Safari).
    - `canvas_rpg_basic.html`: Grid-locked discrete stepping and lerp walking animations (e.g., Pokemon).
    - `canvas_cards_basic.html`: Mouse/touch drag-and-drop hit testing and grid snapping (e.g., Solitaire, Chess).
    - `canvas_physics_basic.html`: Gravity projectile trajectories, launch slingshots, and elastic boundary collisions.
- **`Playbook`** — JSONL of bullets with `helpful` / `harmful`
  counters. Features elite math and physics rules for retro classics (Mode 7 scanning, wall-sliding, segmented follow, angle biasing, mobile joysticks, aspect ratio letterboxing) alongside the standard set. Retrieval is weighted Jaccard × quality multiplier `1 + 0.10·tanh(score/5)`. `stage="plan"` returns broader top-K; `stage="code"` drops bullets with score ≤ -2. On-demand expansion via `<lookup_bullet>id</lookup_bullet>`.
- **Dedup + budget capping**: `dedup_hits` (5-gram Jaccard ≥ 0.85) +
  `cap_hits_by_budget` run inside `render_playbook_block` by default.
- **Won-skeleton promotion**: after `<confirm_done/>`, the agent
  copies the working HTML to `memory/skeletons/won_<session>.html`
  and indexes it so future sessions with similar goals can use it as
  the starting scaffold.

[`learner.py`](learner.py) is the offline pipeline that reads
completed traces and runs a Reflector (proposes bullet deltas) +
Curator (deterministic merge into `playbook.jsonl`, increments
helpful/harmful counters, prunes dead bullets). It runs on demand or
attached to a `tune.py run --auto-learn`.

### Verification harness (multi-layered)

[`tools.py`](tools.py) — the verifier:

**Pre-Chromium micro-probes** (`run_micro_probes`, fast, no browser):

- Structural completeness: DOCTYPE, `<html>`, `<body>`, `<script>`
  presence, bracket balance per inline script (with regex-literal
  tolerance).
- **API allowlist** for invented method calls on known-receiver
  variable names (canvas2d, AudioContext, HTMLCanvasElement, **and
  HTMLAudioElement** — added so `audio.startWithFadeIn()` and similar
  invented audio APIs fail static, not in Chromium).
- Repetition-collapse loops (`_ACTUAL_ACTUAL_ACTUAL_…` family, line
  repeats, suffix loops in large bodies).
- Elision sentinels (`// ... rest unchanged`, etc.) — fails the iter
  before shipping a half-implemented file.
- **Asset path existence**: relative `./*.png` / `./*.ogg` paths in
  the HTML are verified to exist on disk; missing paths get a
  closest-match suggestion via `difflib`.
- **Unused-asset integrity probe**: assets generated to
  `<slug>_assets/` and `<slug>_sounds/` that the HTML never references
  produce explicit warnings ("PNG generated but never used"). Catches
  the silent failure where the model declares 12 sprites and wires
  none of them.

**Chromium harness** (`LiveBrowser.load_and_test`, async, visible by
default):

- Launches with `--allow-file-access-from-files` +
  `--disable-web-security` so `drawImage(<file:// PNG>)` doesn't taint
  the canvas.
- For likely three.js/WebGL pages, runs a short second strict check in
  stock `file://` Chromium **without** those relaxed flags; strict
  failures are classified into concise repair buckets
  (`cors_blocked`, `script_load_failed`, `asset_path_missing`,
  `render_loop_missing`) and block `ok=True`. If the strict checker
  itself fails (harness infra issue), it downgrades to a warning and
  does not block shipping.
- Captures `console.error` and `pageerror`, capped at 12 lines × 240
  chars to keep small models from drowning in logs.
- Canvas state (size, RAF fired, blank detection sampling 32×32
  grid for both Canvas2D and WebGL contexts, frozen-canvas check via
  before/after hash).
- Listener count (document / window / body / other) — distinguishes
  "no input handlers wired" from "input handler attached but state
  doesn't update".
- Model-proposed JS probes run in the page context; results join the
  report. Tainted-canvas probe errors auto-downgraded to passes.
- Automated input smoke test (synthesized keydown / pointerdown
  events) so a "press space to start" gate doesn't make the canvas
  look frozen.
- **Audio-events shim**: hooks `HTMLAudioElement.play`,
  `AudioBufferSourceNode.start`, and `OscillatorNode.start` to push
  `{t, src, kind}` records into `window.__audioEvents`. Model probes
  assert against this to gate that sounds actually FIRE (not just
  load).
- Screenshot captured every iter; saved alongside the trace.

Probes gate `ok=True`. A failed probe or `soft_warning` flips
`report["ok"] = False`. `<done/>` requires the consecutive-clean-iters
threshold.

**Screenshot-delta regression detector** (`screenshot_delta`): mean
per-pixel RGB diff between consecutive iters in [0, 1]. Used in the
loop: high delta + `PROGRESS: no` from the vision judge =
REGRESSION SUSPECTED, escalated as a rollback-flavored coaching
message.

### Visual-progress judge (local VLM)

[`vision_judge.py`](vision_judge.py) compares prev vs. current
screenshots and returns:

```
PROGRESS: yes | no | unclear
MISSING: <one-line description of what's still wrong>
```

The judgment flows into the next user turn as coaching, e.g.
`"VISUAL JUDGE: this iter made progress, but Still visibly
missing/wrong: ship is offscreen — address this on the next iter."`

**Auto-on** when [`backend.discover_local_vlm()`](backend.py) finds an
MLX-VLM model (any path whose name classifies as `vlm` —
`qwen2.5-vl-*`, `llava-*`, `cogvlm-*`, etc.) under `~/MLX_Models/` or
the HF cache. Disable explicitly with `VISION_JUDGE=0`.

**Never silently calls a cloud model.** Cloud paths (Anthropic/OpenAI)
are reachable only via explicit user actions (e.g. `/check ...` or
staging a cloud backend). The agent loop never auto-falls back to cloud.

The last verdict (iter + progress + note) survives the state-anchor
compaction so the model still knows what the game *looked like* even
when the older turns get elided.

### Sprite generation (Z-Image-Turbo)

[`assets.py`](assets.py) is fully self-contained (no sibling-repo
imports, no servers, no subprocess). `ZImageTurboGenerator` lazy-loads
the diffusers pipeline into this process's GPU VRAM only on the first
`<assets>` request (~30-60 s cold, ~2-4 s per image after).

Pipeline per spec:

1. `parse_assets_block(reply)` extracts the JSON list (tolerant of
   ```json fences and truncated streams).
2. Cache key = sha256(model_id, normalized prompt, size). Cache hit →
   hardlink (or copy) into the session dir, skip generation.
3. **Cross-session library lookup** (production path only). See
   [Cross-session asset library](#cross-session-asset-library) below.
4. Cache miss + library miss → generate at native 768×768.
5. PIL Lanczos downscale to per-asset target size (default 128 px).
6. `_chroma_key_to_rgba` samples 8 corner+edge points; if ≥6/8 agree
   on a dominant color, alpha-mask within tolerance → save RGBA PNG.
7. Per-asset stats stash on `image_generator.last_stats`.

`render_asset_paths_block` builds the `GENERATED ASSETS` injection for
the first-build user message, with the literal loader pattern (`const
ASSETS = {}; await img.decode(); ctx.drawImage(...)`) inline.

### Animation frame chains (SD-Turbo img2img)

`assets.py` supports `from_image` chaining: a spec with
`"from_image": "<name-of-prior-spec>"` becomes an SD-Turbo img2img
generation seeded from that prior frame at controllable `strength`
(default 0.45, range 0.05–1.0). Frame 1 ships as txt2img; frames 2..N
inherit silhouette + palette while only the pose changes. This is the
recipe for walk cycles, attack windups, idle bobs.

Topologically sorted so children always render after parents. Falls
back to txt2img cleanly if img2img isn't available.

### Sound generation (Stable Audio Open)

[`sounds.py`](sounds.py) mirrors the sprite pipeline. `<sounds>` spec
shape is `{name, prompt, duration?, loop?}`. Duration is 0.2–12.0 s
(SFX is 0.2–1.5 s, looping background music is 8–12 s with `loop:
true`). Cache and library lookups work identically to assets. Output
is OGG Vorbis.

The prompt-side guideline tells the model to load via `new
Audio(path)`, play overlap-safe SFX with `audio.cloneNode().play()`,
loop music via `audio.loop = true; audio.play()`, and respect the
browser's user-gesture-before-audio rule (unlock on first
keydown/pointerdown).

### Cross-session asset library

[`asset_library.py`](asset_library.py) sits one layer above the
per-project `_asset_cache/`. It indexes admitted assets by tokenized
prompt and metadata, and serves them on semantically-similar prompts
in future sessions.

- **Index**: `memory/asset_index.jsonl` — one JSON line per
  entry. Schema: `id, modality, prompt, tokens, size_or_duration,
  sha, path, helpful, harmful, last_used, created_at`.
- **Storage**: `memory/asset_library/sprites/<id>.png` and
  `…/sounds/<id>.ogg`.
- **Retrieval**: Jaccard token match against the prompt, scoped to
  exact size for sprites (10% tolerance for sound duration), minimum
  score 0.5 by default. Stale paths are skipped (returned-None safe).
- **Admission**: idempotent on sha. New bytes get a fresh `id`;
  re-admitting the same bytes bumps `last_used` only.
- **LRU eviction**: capped at 2000 entries (configurable). On
  overflow, evict by oldest `last_used` and delete the underlying
  file.
- **Test hermeticity**: the library is gated on the production code
  path only (`cache_dir is None and image_generator is None`). Tests
  that inject a stub generator skip the library entirely so cross-
  session state never leaks into the test suite.

The stock-sound script
[`scripts/build_stock_sounds.py`](scripts/build_stock_sounds.py)
populates the library at install time with 8 universal SFX (`jump`,
`pickup`, `hit`, `win`, `lose`, `click`, `laser`, `explosion`).
Subsequent sessions hit cache for these without paying the audio
diffuser. Idempotent — re-running is a no-op for entries already
present.

### Prompt assembly (data-driven, not a string blob)

[`prompts_v1.py`](prompts_v1.py)'s `SYSTEM_PROMPT =
build_system_prompt("{goal}")` walks `ALL_FORMATS` (a list of
`FormatSpec(name, snippet, guidelines)` per output tag), dedupes
guidelines across enabled formats, and renders `<output-tags>` +
`<guidelines>` + `<hard-rules>` + `<anti-patterns>`. To swap or
disable a tag, edit the `FormatSpec` list — never hand-edit the
rendered prompt.

`plan_instruction(goal=...)` calls `_detect_art_intent(goal)` and
`_detect_3d_intent(goal)`. Matched keywords inject "ART INTENT
DETECTED" / "3D INTENT DETECTED" callouts that escalate `<assets>` and
three.js usage from "expected" to "required this turn." The keyword
sets describe rendering modality (`sprite`, `pixel`, `first-person`,
`voxel`), not subject matter.

`model_class` controls the trim path:

- `small` drops `<assets>`, `<sounds>`, `<lookup_bullet>`,
  `<anti-patterns>`, `<reasoning-license>`, `<user-presence>`, and
  collapses `<workflow>` + `<iteration-policy>`. Target ≤ 6 KB.
- `mid` keeps `<assets>` + `<sounds>`, drops `<anti-patterns>`.
- `large` keeps everything.

### Compaction (two-tier state-anchor)

[`agent.py:_prune_messages`](agent.py):

- ≤ `_PRUNE_KEEP_RECENT_TURNS + 1` (default 5): no-op.
- ≤ `_STRUCTURED_PRUNE_THRESHOLD` (default 14): per-turn HTML
  elision — replace `<html_file>` bodies with `[omitted: N bytes]`.
- `> 14`: replace messages 1..cutoff with **one state-anchor message**
  built deterministically by `_build_structured_summary`. Sections:
  - **Goal**
  - **Acceptance criteria** (from Phase A)
  - **Executable probes** (names only)
  - **Progress** (iter pass/fail, stuck-streak, last clean snapshot)
  - **Key decisions** (last diagnose, truncated)
  - **Last test report** (truncated)
  - **Files in session**
  - **Generated assets** (paths still actionable after compaction)
  - **Generated sounds** (same)
  - **Visual state at last judge** (iter, progress label, missing-
    note from the vision judge — preserved so the model still knows
    what the game looked like)
  - **Open todos** (latest `<todos>` snapshot)
  - **Critical context** (truth-source contract reminder)

No extra LLM call. The state-anchor message stays under ~6 KB even
for long sessions.

### Project-config injection

At session start `agent._read_project_config(cwd)` reads `AGENTS.md`
then `CLAUDE.md` from the working directory, caps the concatenation at
6 KB, and appends as `<project-context>` at the END of the system
prompt. Per-repo conventions get inherited automatically. This file
is data the agent will see, not just developer notes.

---

## TUI and CLI reference

### Slash commands (TUI)

| Command | What it does |
|---|---|
| `/help`, `/h`, `/?` | Show command list. |
| `/list`, `/models` | Inventory of MLX + Ollama models, with `[VLM]` / `[text]` labels. |
| `/model <N\|name>`, `/load <N\|name>` | Switch the chat model. |
| `/backend <auto\|ollama\|mlx\|openai\|anthropic>` | Switch the backend (cloud choices are explicit and billable). |
| `/unload` | Free the loaded MLX model from VRAM. |
| `/new` | Start a new session in the same workspace. |
| `/ship` | Force `<confirm_done/>` on the next critique turn. |
| `/quit` | Exit. |
| `/open` | Reveal the current HTML in the file browser. |
| `/log`, `/paths`, `/files` | Print log + artifact paths. |
| `/clear` | Clear the chat scrollback (does not reset the session). |
| `/iters <n>` | Change the iter budget mid-session. |
| `/seed <text>` | Inject a one-shot system seed at the next user turn. |
| `/reset` | Reset agent state to a fresh session. |
| `/status` | Print iter count, ok status, model, backend, VLM. |
| `/wait <on\|off>` | Toggle step-mode — default ON in the TUI; when on, pause after each iter; when off, iterate continuously. |
| `/iter-detail <on\|off>` | Toggle optional expanded blocker detail after the compact iter decision line (default off). |
| `/mode <local_manual\|local_auto\|local_plus_review with <model> [--auto-apply]\|custom>` | Apply a run contract: manual checkpoints, autonomous loop, or autonomous loop with an explicit reviewer hook. |
| `/playbook`, `/memory` | Print the matched playbook bullets. |
| `/prefill <on\|off>` | Toggle forcing assistant prefill tags (`<plan>`, `<diagnose>`) to prevent preamble talk and lock XML formatting (default ON). |
| `/architect <on\|off>` | Toggle architect/editor split (Aider's 2-call pattern) on complex first-builds to split planning from coding (default off). |
| `/double-screenshot <on\|off>` | Toggle capturing dual screenshots (startup and post-input) to help the model see movement/animation (default off). |
| `/vlm-critique <on\|off>` | Toggle VLM screenshot attachment during Phase C successful critique turns for layout and UI polishing (default off). |
| `/audit` | Detailed view of last iter's micro-probes + report. |
| `/restarts` | Show backend restart history. |
| `/model-class <small\|mid\|large>` | Override the prompt trim path. |
| `/launch` | Manually launch Chromium against the current best file. |
| `/check [<N\|model>]` | Run visual review on the latest screenshot. `N` selects by `/list` number. Bare `/check` uses the active VLM session model. `claude-…` routes Anthropic, `gpt-…` routes OpenAI, and other names resolve local MLX VLMs. In wait mode ON, suggestion is prefilled in input for edit/Enter; in wait mode OFF, coaching auto-queues to the next coding turn. |

### CLI flags (`coder.py`)

```bash
.venv/bin/python coder.py "<goal>" \
  [--max-iters N]            # iter budget (default 6)
  [--best-of-n N]            # best-of-N sampling per turn (default 1)
  [--headless]               # no visible Chromium
  [--backend auto|ollama|mlx]
  [--model <name>]
  [--prompt-version v1]
  [--no-playbook]            # skip playbook retrieval
  [--no-skeleton]            # ignore skeleton library
  [--no-assets]              # skip Z-Image-Turbo
  [--no-sounds]              # skip Stable Audio
```

`tune.py` flags include `--prompt-version`, `--best-of-n`,
`--max-iters`, `--feature-flags prefill,vlm_critique,…`, `--learn`,
`--auto-learn`, `--learn-shared`.

---

## Tuning rig and playbook commands

```bash
# A/B between prompt versions and feature flags.
python tune.py run                                  # quick: max_iters=2, best_of_n=1
python tune.py run --full --prompt-version v1 --auto-learn
python tune.py diff baseline_v0 v1_run              # per-test pass/fail deltas
python tune.py why <run_id> <test_name>             # postmortem one test

# Offline learner — Reflector + Curator over traces.
python learner.py walk                              # one-line summary per past session
python learner.py reflect games/traces/             # propose deltas (no writes)
python learner.py apply games/traces/               # propose AND write to playbook.jsonl

# Memory hygiene.
.venv/bin/python scripts/forget_session.py --list
.venv/bin/python scripts/forget_session.py <session_id>
./scripts/clean_artifacts.sh --yes                  # bulk wipe stale per-session artifacts

# Playbook ops.
.venv/bin/python scripts/audit_playbook.py
.venv/bin/python scripts/prune_playbook.py --negative-only
.venv/bin/python scripts/bench_playbook_ab.py
```

The canonical regression check is **asteroids**: `vx = cos(angle) *
speed` for ship direction, and asteroids drawn as **irregular
polygons** (not perfect circles). Run after any change to retrieval,
prompts, patches, or asset wiring.

### Validation matrix (loop reliability)

Use this matrix when validating `chat.py` loop/control changes:

| Scenario | Setup | Expected result |
|---|---|---|
| Asteroids regression guard | `/new asteroids` on local backend | Ship direction uses `vx = cos(angle) * speed`; asteroids stay irregular polygons. |
| Wait-mode manual control | Default TUI mode, or `/mode local_manual` | Agent pauses after each iter (`await_user`), accepts user feedback before continuing. |
| Autonomous loop control | `/mode local_auto` or `/wait off` | Agent iterates continuously without step pauses unless user explicitly enables wait mode. |
| Explicit external review (manual) | `/mode local_plus_review with <model>` and keep wait on | Reviewer guidance can be run via `/check <N\|model>`; suggestion is prefilled for edit/Enter. |
| Explicit external review (auto loop) | `/mode local_plus_review with <model> --auto-apply` with wait off | Failed iters can auto-run explicit reviewer hook and inject coaching for next turn. |
| No silent cloud guarantee | Local-only run (`/mode local_auto` or `/mode local_manual`) | No cloud calls unless user explicitly picks cloud backend or `/check ...`. |
| Active-stream cutoff regression guard | Rich-media first-build run on local MLX/Ollama model | If tokens or backend progress are still arriving, the model is working. Never stop it with an absolute wall-clock cutoff; only no-activity stalls, repetition loops, deliberation loops, backend max-token caps, hard crashes, or explicit user cancel may stop generation. |

---

## Standing rules and design constraints

These have evidence behind them and should not be broken silently.

1. **Tune the agent, not the model.** The model is held fixed; "tune"
   means prompt / retrieval / playbook / probes / loop changes. Never
   "try a bigger model" as a fix.
2. **No hardcoded genre / category lists.** Open-domain HTML/JS —
   retrieval, probes, and skeletons stay genre-free. Modality keyword
   detectors describe rendering shape (sprite, voxel, first-person),
   not subject matter (asteroids, doom).
3. **All code self-contained in `Agent_learning/`.** No sys.path-
   injecting sibling repos. External *data* at standard system paths
   (Ollama, HF cache, MLX dirs) is fine.
4. **Visible Chromium, not headless.** The TUI default is
   `headless=False` so the user can watch the game. CLI keeps
   `--headless` for unattended runs.
5. **Asteroids is the canonical regression check.** Ship direction +
   irregular-polygon asteroids must still pass after any change.
6. **Iterate autonomously.** Drive build/test/improve loops yourself.
   The tune battery is the alignment check.
7. **Never silently call a cloud model.** Cloud / paid API calls must
   be EXPLICIT (slash command, flag). Never default-on, never auto-
   fallback. The user owns the wallet.
8. **Never cut off an active local-model stream for wall-clock time.**
   MLX/Ollama context windows can be huge and rich first-build
   `<html_file>` emissions can be slow. If tokens or backend progress
   are still arriving, generation is working and must be allowed to
   finish. Safe stop reasons are: no-activity stall, repetition loop,
   deliberation/no-output loop, backend max-token cap, hard backend
   crash, or explicit user cancel. Any change to repetition /
   deliberation / timeout abort logic must cite concrete trace evidence
   and include regression validation that active first-build streams are
   not cut off mid-emission.

---

## Major future improvements

Concrete, research-grounded directions for moving small local LLMs
from "playable" to "great" at writing HTML games. Each item lists the
problem it addresses, the proposed change, and the rough effort. The
ranking is intent, not commitment; pick the ones that match your
machine and time budget.

### 1. Vision-judge gating of `<done/>`

**Problem.** Probes verify *correctness*; the vision judge today
contributes *coaching* but does not gate ship. A game can pass probes
with a blank canvas or invisible player and still get `<confirm_done/>`.

**Change.** Require the last clean iter's vision verdict to be
`PROGRESS: yes` (or `unclear` with non-empty `MISSING: nothing
obvious`) before `<done/>` is honored. Falls back to current behavior
if no local VLM is discoverable.

**Why this is the highest-leverage next move.** It is the cleanest
way to make "looks like the user asked for" a hard precondition rather
than a hint. The infrastructure is already in place; the change is a
single guard in `agent.py`'s done-detection block.

### 2. Reward-shaping the playbook from vision verdicts

**Problem.** The playbook's helpful/harmful counters increment on
session outcome (won = +1 to active bullets, regressed = -1). The
vision judge produces a per-iter signal that today only flows into
coaching.

**Change.** Reflector reads `vision_judge.{iteration, progress, note}`
records alongside test outcomes. When `PROGRESS: yes` follows a
patch derived from a specific bullet, that bullet gets a smaller
helpful increment (e.g. +0.3). When `PROGRESS: no` with high
screenshot-delta follows a bullet's pattern, that bullet gets a
proportional harmful increment. Compound the signal across sessions.

**Effect.** The playbook starts to learn what makes games *look*
better, not just what makes probes pass.

### 3. Semantic-embedding retrieval (replace Jaccard)

**Problem.** Playbook and asset-library retrieval today use Jaccard
on a tokenized prompt. This works at small scale but degrades fast as
the library grows past ~500 entries — synonymy and paraphrase miss.

**Change.** Wire a local embedding model (Sentence-Transformers
all-MiniLM, BGE-small, or a similarly tiny model on MLX/MPS) to embed
prompts at admit time and at query time. Cosine similarity replaces
Jaccard. Keep the Jaccard path as a fallback when embeddings aren't
available.

**Effect.** "explosion sprite" and "boom particle effect" hit each
other even with no token overlap. Playbook retrieval gets less brittle
to paraphrase. Asset library scales to 10–100k entries.

### 4. Multi-agent critic ensemble

**Problem.** A single coding model is both the implementer and (via
self-critique) the reviewer. Local 27–35B models are weaker at
honest self-assessment than at code generation — they shadow-pass
their own bugs.

**Change.** Ship three lightweight sidecar critics that run after
each iter:

- **Art critic** (VLM): "does the rendered sprite match the prompt?
  is it readable at this draw scale? is alpha working?"
- **Sound critic** (text + audio-event timeline): "do the audio
  events fire on the moments that matter — collision, scoring, death?
  is anything looping that should be one-shot?"
- **Gameplay critic** (text + report + screenshot): "given the
  acceptance criteria, what one thing is most missing?"

Each emits a one-sentence verdict that joins the next user turn as
coaching. Cheap to run because each critic gets a tight, scoped
prompt (~1 KB context).

**Why this matters more for small models.** Specialized critics each
get the full attention budget on their narrow domain; a single
generalist self-critique splits attention across all three.

### 5. Best-of-N at the iter level, not the session level

**Problem.** `--best-of-n N` today samples N completions for one turn
and picks the best by ranker. Sessions are still serial.

**Change.** On clean iters, fork two short branches: one tries a
small refinement, one tries a larger structural change. Run both
through the verifier in parallel. Keep the winner; discard the loser.
Costs 2× GPU on clean iters only; clean iters are cheap (no fix
needed).

**Effect.** The agent stops getting stuck in local optima of the
form "this works fine and I have no reason to change it but it
doesn't look great."

### 6. Two-model split: planner vs. coder

**Problem.** Local 27–35B models good at structured Phase A planning
are often *not* the same ones good at high-quality Phase B patches,
and vice versa.

**Change.** Optionally bind two different MLX models — a planning
model for Phase A and a coding model for Phase B/C — with weights
both in VRAM (small enough Macs only) or hot-swapped (larger Macs).
The agent already abstracts the backend; the change is per-phase
backend selection.

**Effect.** Each phase gets the model that's best at it. The cost is
VRAM (or load latency on swap).

### 7. Game-feel benchmarks

**Problem.** "Great game" is currently judged on visual appearance
and audible feedback. *Game feel* (input lag, juice, screen shake,
hitstop) is invisible to the harness.

**Change.** Extend the audio-events shim pattern to capture:

- **Input → state delta latency.** Synthesize keydown; measure ms
  until the next RAF tick mutates a tracked variable.
- **Frame timing.** Record per-RAF timestamps; report mean FPS,
  longest stall, jank counts.
- **Juice signals.** Track screen shake (canvas transform deltas
  around game events), hitstop (RAF idleness in a tracked window
  around damage events).

Add a few model-proposed probes that gate on these (e.g. "input → ship
moves in <33 ms"). The agent gets explicit feedback on the
hard-to-articulate dimensions of "good".

### 8. WebGL / three.js verifier coverage

**Problem.** Today's harness is strongest on Canvas2D. WebGL games
(first-person, voxel, true 3D) pass the harness loosely — the canvas-
hash sampler covers WebGL but the probes the model writes default to
2D patterns.

**Change.** Prompt-side: extend `<probes>` examples with WebGL/three.js
patterns (camera position read, scene graph child counts, render-loop
fired). Harness-side: a tiny shim that exposes `THREE.WebGLRenderer.
render` call counts on `window.__threeRenderCount` so probes can
assert the loop runs.

**Effect.** 3D games stop being a second-class citizen.

### 9. Active-learning asset hints

**Problem.** The model writes a generic prompt for a sprite; Z-Image-
Turbo produces a mediocre rendering; the model has no signal that the
sprite is the weak link.

**Change.** After generation, the vision judge gets a per-sprite
question: "does this sprite match the prompt? would a small change to
the prompt produce something more readable at this size?" If the
judge says "smaller, higher contrast, cleaner silhouette", the agent
prepends that to the prompt and regenerates *once* (capped).

**Effect.** First-iter sprites improve materially without growing the
model context.

### 10. Self-play curriculum

**Problem.** The tune battery is hand-curated and small (~12 goals).
Generalization to long-tail goals comes from the user, not from
training.

**Change.** Periodically, the agent generates its own goal ("simple
puzzle game where two colors must combine to create a third"), runs a
full session against it, evaluates with the vision judge + probes,
and feeds the trace to the learner. Treat goal generation as a
diversity exercise — sample from modality keywords plus a small
plot-noun bank, not a genre list (rule 2 stays).

**Effect.** The playbook compounds even when the user isn't actively
building.

### 11. Local LoRA adapter from session traces

**Problem.** The playbook compounds *retrieval-time* knowledge, not
*model-time* knowledge. The base MLX model never gets better at the
patterns it sees succeed.

**Change.** Once `games/traces/` contains enough confirmed-good
sessions (say, 50 won + 50 fixed-regression pairs), train a small
LoRA on the trace pairs. Default to a quantized adapter so it lands
under 200 MB on disk and adds <50 ms to per-token latency on Apple
Silicon. The agent loads the adapter alongside the base model.

**Effect.** The base model itself starts shipping cleaner first
patches on the patterns it has seen many times. The playbook still
covers the long tail.

### 12. Multi-modal compaction with real images

**Problem.** Today's state-anchor preserves a *text description* of
the visual state from the vision judge. VLM-capable backends could in
principle handle a 128×128 thumbnail embedded as a real image content
block.

**Change.** When the discovered backend is a VLM, the state-anchor
becomes a multi-modal message: text sections + one base64 PNG
thumbnail of the last clean iter's screenshot. Falls back to text
when the backend is text-only.

**Effect.** Long-session visual coherence improves measurably; the
model "remembers" the look of the game across compactions, not just
the words.

---

## Improving the Agent From a Trace

When a session goes wrong, the artifacts under `games/traces/` are
the single source of truth. Use this checklist instead of guessing.

**Order of inspection.**

1. `games/traces/<session>.log` — human-readable timeline. Read this
   first to identify which iteration broke and what the user said.
2. `games/traces/<session>.jsonl` — structured events, one JSON per
   line. Grep for `kind` values listed in the glossary below to see
   the agent's routing decisions.
3. `games/traces/<session>.conversation.md` — the exact prompts and
   replies the model saw. Look here when you suspect prompt bloat or
   conflicting directives.
4. `games/snapshots/<session>/iter_<N>.png` — visual regression
   evidence. Compare consecutive screenshots when the test report
   says probes pass but the user disagrees.

Seeded runs reuse the canonical game HTML/assets basename, but trace
artifacts are per-run. A seeded game like `games/foo.html` writes
fresh artifacts such as `foo__run_<timestamp>.jsonl`, `.log`,
`.conversation.md`, and `snapshots/foo__run_<timestamp>/`. This keeps
the live game/assets stable while preventing different goals from
mixing in the same trace/log/conversation files.

**Diagnostic order.**

1. Identify the user's actual intent in plain English.
2. Find the `feedback_injected` event for that turn and the
   `turn_contract` row immediately after.
3. Compare expected tags vs `allowed_tags`/`forbidden_tags`. If the
   contract is wrong, the failure is in the classifier or routing,
   not the model.
4. Check whether the harness probes test real behavior or just a
   flag (look for `probe_lint` events flagging `probe_bait_flag`).
5. Check the screenshot delta and `vision_judge` events for visual
   evidence (vs probe-pass).
6. Prefer routing / prompt / probe fixes before loop or timeout
   changes.
7. Never fix runaway streams by adding an absolute wall-clock cutoff to
   active generation. If `stream_heartbeat` tails keep changing, or MLX
   prompt/token progress is still arriving, the model is working. Let it
   finish unless a real guard fires: no-activity stall, repetition loop,
   deliberation/no-output loop, backend max-token cap, hard backend
   crash, or explicit user cancel. Prefer pre-stream containment
   (bounded prompts, narrower contracts) over stream guards.

### Trace event glossary

The `.jsonl` file mixes high-level events (`event`) with structured
agent decisions. Use these `kind` values as entry points:

| `kind` | When it fires | What it tells you |
|---|---|---|
| `session_start` | Start of every run | Includes `session_id` (game basename), `artifact_id` (per-run trace id), goal, model, trace path, conversation path |
| `feedback_queued` / `feedback_injected` | User typed feedback | Exact text the next user turn will carry |
| `turn_contract` | Once per stream, just before model call | Routing contract for this turn: allowed/forbidden tags, scoped_mode, classifier flags, prompt section sizes |
| `media_change_directive_injected` | MEDIA-CHANGE block added to prompt | Routed feedback to asset/sound regen |
| `media_change_directive_suppressed` | MEDIA-CHANGE skipped | Reason is in `reason` field: `behavior_bug`, `behavior_scope_on_strict_turn`, `orientation_change`, or `use_existing_media` |
| `orientation_change_directive_injected` | ORIENTATION-CHANGE block added | Routed feedback to canvas mirror patch |
| `scoped_constraints_set` | `_configure_scoped_constraints` ran | `mode` is `single_patch` or `media_only` |
| `scoped_change_directive_injected` | SCOPED-CHANGE block added | User locked the turn |
| `bounded_asset_only_injected` | BOUNDED OUTPUT block appended | Final hard cap for media_only turns |
| `initial_goal_scoping_applied` | First build had scope-lock language | Scope applied from goal, not feedback |
| `rewrite_exemption_armed` / `rewrite_exemption_suppressed` | `<html_file>` rewrite license toggled | `_allow_one_rewrite` is on or off |
| `assets_generated` / `midsession_asset_injection_queued` | Z-Image-Turbo regen finished | `already_in_html_assets` vs `new_assets` shows whether a loader block was injected; same-name replacements include old/new hashes and `changed` |
| `probe_lint` / `probe_lint_postbuild` | Probes flagged as broken | Includes `probe_bait_flag` for boolean-flag bait probes |
| `probe_eval_error` | A probe expression errors before yielding true/false | The probe is broken, not necessarily the game |
| `probe_quarantined` | Same probe has repeated eval-time errors | Probe removed from future runs until model re-emits a corrected probe |
| `post_clean_feedback_contract` | User gives feedback after a clean build | Full clean report is compacted; model is told to keep the clean build as baseline and make only the requested change |
| `stream_start` / `stream_done` / `stream_heartbeat` | Token streaming events | Changing tails mean active generation. Never add a wall-clock cutoff for active streams; only no-activity stall, repetition, deliberation, max-token cap, hard crash, or user cancel should stop them. Repeated identical tails indicate a degenerate loop; read `_flush_user_injections` to find what gave the model contradictory instructions |
| `no_usable_code` | Reply had no `<patch>` / `<html_file>` | `media_only=true` means asset regen with no code edit |
| `vision_judge` | VLM verdict on this iter's screenshot | `progress` is yes/no/unclear |
| `format_rejection` | Reply didn't parse as expected tags | Usually means stream stalled mid-tag |
| `status_snapshot` | TUI status panel snapshot | Persists right-hand panel state to JSONL |

When you see the same failure shape twice, write a regression test
in `tests/` and add a one-line bullet to `playbook.jsonl` via
`learner.py reflect`.

---

## Troubleshooting

- **`Playwright Chromium missing`**: run `env -u PLAYWRIGHT_BROWSERS_PATH
  .venv/bin/python -m playwright install chromium`.
- **`MLX out of memory`**: lower the wired-memory cap or pick a smaller
  quant. `MLX_PREFILL_STEP_SIZE=512` if you OOM mid-generation.
  DeepSeek-V4 Flash specifically requires 512 (auto-detected via path
  substring).
- **`Ollama context too small`**: `ollama run --ctx-size 262144 <model>`
  before launching. The agent's default is 256K.
- **`Vision judge never fires`**: check `VISION_JUDGE=0` isn't set;
  verify `backend.discover_local_vlm()` returns a path (run
  `.venv/bin/python -c "from backend import discover_local_vlm;
  print(discover_local_vlm())"`).
- **`Sprites blur when drawn small`**: the asset generator's default
  output size is 128 px; ask for a smaller `size` (`"size": "32x32"`)
  for icons rendered at < 64 px.
- **`First-iter games are silent`**: run
  `scripts/build_stock_sounds.py` once. The 8 stock sounds become
  zero-cost on subsequent sessions.
- **`Sounds load but never play`**: check `window.__audioEvents` in
  the browser devtools after pressing a key. If empty, the model
  didn't wire a `play()` call on input.
- **`Game looks great, probes still fail`**: the vision judge note is
  the human signal; the failing probe name is the machine signal.
  Reconcile them in chat.
- **`Seeded game ignores existing sprites/sounds`**: seeded prompts
  include a `SEED MEDIA CONTRACT` listing available media, currently
  referenced media, and media that exists on disk but is not loaded.
  If the user says "use existing", "don't redo", or "use the original
  ones", MEDIA-CHANGE is suppressed and the model should patch loader
  or draw mapping instead of regenerating assets.
- **`Regenerated asset still looks unchanged`**: same-name media regen
  records old/new file hashes in `midsession_asset_injection_queued`.
  If `changed=false`, the generator returned identical bytes or failed
  to replace the file; if `changed=true` but the screenshot is unchanged,
  inspect draw-state mapping or browser caching rather than blindly
  regenerating again.

---

## Dependencies

Python packages (in `requirements.txt` and `requirements-diffuser.txt`):

- **playwright** — Chromium harness.
- **ollama** — async client.
- **mlx-lm** + optional **mlx-vlm** — Apple GPU inference.
- **anthropic** / **openai** — optional cloud backends, only used when
  explicitly selected (`/backend anthropic|openai` or `/check ...`).
- **torch + diffusers + transformers + accelerate** — asset and audio
  pipelines.
- **PIL** — sprite postprocessing.
- **soundfile / libsndfile** — OGG encoding.
- **textual** — TUI.

External weights (auto-downloaded to system caches on first use):

- **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo` on HF).
- **SD-Turbo** (`stabilityai/sd-turbo` on HF).
- **Stable Audio Open** (`stabilityai/stable-audio-open-1.0` on HF;
  Stability AI Community License).
- A chat MLX model of your choice (qwen3.6, DeepSeek-V4, GLM-5.1,
  MiniMax-M2 are all known-working).
- Optional: an MLX-VLM (qwen2.5-vl, llava, etc.) for the local visual
  judge.

`tune.py` and `learner.py` have no extra dependencies beyond the
agent.

---

## License

No `LICENSE` file ships in this repository yet. If you fork or
redistribute, add a license you are comfortable with (or confirm
terms with the repository owner on GitHub).
