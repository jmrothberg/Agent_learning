# Coding Box Overlay — Local HTML Game Agent (harness fork)

> **This repo is not `Agent_learning` on GitHub.** It is the **overlay /
> harness** line (feedback routing, playtests, multi-slot fan-out, TUI
> polish). Push here: **`jmrothberg/Agent_learning_overlay`** so you do not
> overwrite the upstream program at `jmrothberg/Agent_learning`.
> Clone: `git clone https://github.com/jmrothberg/Agent_learning_overlay.git`
> Remote setup: `git remote set-url origin https://github.com/jmrothberg/Agent_learning_overlay.git`
> and optionally `git remote add upstream https://github.com/jmrothberg/Agent_learning.git`

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

**Remote (this fork):** https://github.com/jmrothberg/Agent_learning_overlay  
**Upstream program (do not push overlay work here):** https://github.com/jmrothberg/Agent_learning

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
  - [Model topologies](#model-topologies-1-2-and-3-model-runs)
  - [Status panel & GPU map](#status-panel-right-column)
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
entirely on a local machine. Two drivers ship in the repo:

- `chat.py` — Textual TUI with a visible Chromium beside the terminal.
  Default. Mid-stream user feedback queue, slash commands, asset
  picker, screenshot review, model swap, playbook viewer, live status
  panel (GPU map, per-role activity).
- `coder.py` — Headless CLI for unattended runs. Same agent, same
  loop, no UI.

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
intentionally genre-free (see **Feedback routing…** below for multi-frame
rosters, asset-cap raises, and cross-slot KV warm).

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

Drivers (`chat.py`, `coder.py`) construct `GameAgent`, wire
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

**Feedback routing, fix strategy, and loop telemetry.** The items below
are what the harness does between your typed messages and the next model
call — no separate modes to configure unless noted.

**When the last iter failed (blocker-first).** If micro-probes or
Chromium still report `ok=False`, code-touching feedback is normally
**deferred** so the model fixes the failing test first. Before that
decision runs, the queue is **partitioned**:

- **Media-only asks** (regen sprites/sounds, animation frames via
  `from_image`) are peeled off and processed **this turn** on the
  diffuser path (GPU 0) — they do not wait behind a canvas draw warning
  on the coder GPU.
- **Code-touching asks** stay deferred until the blocker is clean.

If the same code ask was deferred **twice** already, **escalation**
forces it through on the third turn: the test report still appears, but
the model is told the user has repeated the request and must honor it.
**Natural-language overrides** also bypass deferral — e.g. "the game
works great", "don't change the game", "ship anyway", "ignore the test"
— without hiding the harness failure from the prompt.

**Mid-session feedback classifiers (genre-free).** On every injected
user message, heuristics decide whether to add MEDIA-CHANGE (regen
`<assets>` / `<sounds>`), keep existing media, request **img2img chains**
(`from_image` off a prior frame), or a **style rebrand** (regenerate
every existing asset **name** with new txt2img prompts — **not**
`from_image`, which would keep the old look). Art-change detection
includes descriptive verbs (`want`, `need`, `look like`, …) and **fuzzy
entity stems** (`"all the pawns"` → `white_pawn_idle`, …). Mid-session
feedback that clearly requests a rebrand can **raise the asset cap** the
same way the initial goal can. If a rebrand is queued and the model still
emits mid-session `<assets>`, generation is **skipped** and the model is
coached to apply the rebrand directive first — avoids GPU time on sprites
that will be thrown away.

**Multi-frame goals.** Planning detects explicit walk-cycle / animation-
roster language and nudges the model to emit `from_image` variant chains
(not idle-only sprites). The per-turn asset cap rises (up to **36**
entries vs the default **24**) at session start and when mid-session
feedback repeats that intent.

**Fix turns: best-of-N and patch traces.** Failed iters sample up to
**three** patch candidates by default (`best_of_n=3` in `GameAgent` and
`coder.py --best-of-n`; set `1` to disable). Each candidate is scored;
the first that passes can ship the iter early. Every applied patch block
emits a structured **`patch_outcome`** trace row (per-block ok/error,
match tier) so postmortems can see which SEARCH/REPLACE
shapes fail repeatedly.

**Multi-slot fan-out (Ollama 2- or 3-model topology).** When the agent
detects multiple **independent** Ollama daemons (different backend
objects + different endpoint URLs + critic slot not in flight), the
best-of-N candidates are dispatched **in parallel** across slots
(`_available_sampler_slots` selects them; `_fan_out_best_of_n_across_slots`
runs `asyncio.gather` over the streams). Each candidate uses a distinct
temperature (0.2 / 0.6 / 0.9). Generations run concurrently, scoring is
sequential against the single Chromium so the harness never races
itself. On single-slot / single-GPU configurations the path falls back
to the existing sequential `Backend.best_of_n` — no behavioral
difference, just no parallel speedup. Per-candidate trace events
(`best_of_n_candidate_generated`, `best_of_n_candidate_scored`,
`best_of_n_attempt`) carry slot id, temperature, score, and rejection
reason. When a non-slot-1 candidate wins, a `surprise` event with
`category: "non_slot1_bon_winner"` fires.

**Cross-slot prompt cache (Ollama multi-model).** When the coder slot is
idle during architect planning or asset/sound generation, the agent may
call **`Backend.warm_prefix()`** on the coder backend so the first build
stream reuses prefix-matched KV instead of paying full prefill again
(same messages → Ollama cache hit on the next request).

**Streaming and queue visibility.** Long MLX/Ollama **prefill with no
tokens yet** emits a one-shot **`slow_prefill`** trace + TUI note (work
is not stalled — prefill is in progress). Completions above **~15k
tokens** without a closing tag emit **`runaway_stream_warning`** so you
can Ctrl+D; the harness does **not** hard-abort an active stream for
token count alone. While a stream is running, the status panel shows a
**Queued for next user-turn** banner whenever feedback or a plan answer
is waiting for the next user-turn boundary (input always drains after
the current stream finishes).

**Ctrl+D wins unconditionally.** A ship request (first Ctrl+D, the `/ship`
command, or typing `done` / `looks good`) ends the session at the next
iter boundary on the current passing build. Any feedback that landed in
the queue between the ship request and the boundary — including
`[AUTONOMOUS PLAYTEST]` findings — is **dropped** with a one-line notice
so you can re-send if you actually wanted it applied. Earlier behavior
auto-applied queued feedback before shipping; that contradicted the
explicit ship intent and (because `_stop_event` was still set from the
ship request) caused the next stream to bail at 0.0s with no tokens
(DK trace 20260523_081532). Second Ctrl+D within 2s force-quits the TUI.

**No silent backend fallback from MLX.** MLX stalls now surface with the
MLX-specific recovery hint (lower `MLX_MAX_TOKENS`, raise
`iogpu.wired_limit_mb`, restart the chat process) and stop there. The
former MLX→Ollama auto-fallback produced confusing cross-backend error
cascades (an MLX user gets bombed with Ollama 500s they never asked for).
The cloud→local safety net for **Anthropic** is unchanged: a transient
Anthropic stall still tries a local Ollama once before giving up.

**Autonomous playtests (default ON, `/feedback off` to disable).** After
each **clean** iter, the agent may run scripted **behavior playtests**
from `memory/playtests.jsonl` (recipes with JS `applies_when` gates —
frozen canvas, facing vs movement, held-key bounds, etc.).
`LiveBrowser.record_playtest()` drives a timeline and samples state at
multiple timestamps. Failures produce one paragraph tagged
`[AUTONOMOUS PLAYTEST]` and enter the same feedback queue as typed input
(unless you **ship with Ctrl+D** first — see above). Budget: **3 cycles**
per session max; stops after **2** consecutive cycles with no findings.
Harness failures, patch diagnostics, and the visual critic are
**unchanged** when autonomous mode is off. The always-on **state-vs-render
gap** check (below) can also emit `surprise` events with category
`state_vs_render_gap` when probes pass but an entity with `x`/`y` in
`window.state` is not visibly drawn.

### "Feedback" — four distinct flows, four separate switches

The word *feedback* is overloaded in this codebase. There are four
distinct flows, all controlled separately, and confusing them was the
source of multiple frustrating sessions:

| Flow | What it is | Switch | Default |
| --- | --- | --- | --- |
| **1. Machine bug feedback** | Real Chromium loads your HTML each iter and reports console errors, page errors, frozen-canvas state, RAF firing, listener counts, input smoke-test results, and every `<probe>` the model defined in Phase A. This is the load-bearing verifier signal — what makes the agent better than zero-shot. | **always on** (unconditional) | ON |
| **2. Playbook retrieval** | Top-K most relevant bullets from `memory/playbook.jsonl` (hand-curated recipes — e.g. FPS camera basis vectors, asteroid jitter, image-load race) injected into the prompt each turn. The model reads them and uses what's relevant. | `/playbook on\|off` | ON |
| **3. Autonomous self-playtest** | After each **clean** iter the agent runs a SECOND playtest using playbook recipes (`memory/playtests.jsonl`) and queues `[AUTONOMOUS PLAYTEST]` feedback if it finds something the probes missed. Uses #2 to know what to check. | `/feedback on\|off` | ON |
| **4. Directive wrapping on YOUR typed feedback** | When you type a feedback note into the TUI, a classifier reads your text and adds injected instructions (MEDIA-CHANGE / ORIENTATION-CHANGE / SCOPE ARBITRATION / asset-stem mapping). The model sees those instructions wrapping your literal text. When the classifier is wrong, these wrappers override your guidance. | `/rawfeedback on\|off` | **ON = directives suppressed** (raw mode) |

The agent's value-add comes from #1, #2, #3. #4 is the part that has
historically misrouted user guidance — e.g. typing "down key moves you
forward" got wrapped with "the feedback above is about ART/SOUND, not
code" because the classifier matched "view" in "maze view" as a reference
to the `pistol_view` sprite. As of 2026-05-23 the default is **raw mode
on** (wrapping suppressed) so the model sees what you typed. Flip
`/rawfeedback off` to opt in to the wrappers when the classifier reads
you correctly.

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

[`memory.py`](memory.py) uses **three tiers** (`Playbook.ensure()` /
`GameMemory.ensure()` create missing dirs and seed the playbook on first run):

| Path | Role |
|---|---|
| `memory/` | **Tracked reference** — `playbook.jsonl`, `playtests.jsonl` (behavior playtest recipes for `/feedback`), bundled skeletons, asset library index |
| `games/game-memory/` | **Local learned** (gitignored) — optional live playbook overlay, `won_*` skeletons, `mistakes.jsonl` |
| `games/goals/` | **Short-term** — per-session `goal.txt`, `best.html`, `outcome.json` |
| `games/<stem>.html` (+ `<stem>_assets/`, `<stem>_sounds/`) | **Curated showcases** (tracked when great) — most session HTML is gitignored; promote a winner by restoring/committing the trio and adding matching `!` lines in `.gitignore` (see chess sample below). |

**Curated games in repo** (open from `games/` so relative asset paths resolve):

- [`games/mechanics-standard-chess-on-an_20260522_163629.html`](games/mechanics-standard-chess-on-an_20260522_163629.html) — animated sprites, SFX, negamax AI (`AI_SEARCH_DEPTH=4`).
- [`games/game-of-mortal-kombat-fighing_20260524_101226.html`](games/game-of-mortal-kombat-fighing_20260524_101226.html) — two-player fighter (sprites, SFX, HUD). Shipped from session `.best.html`.

- **`GameMemory`** — skeleton retrieval and mistake retrieval.
  - **Premium Default Skeletons (Autobootstrapped on boot)**: The system provides 17 generic, high-fidelity scaffolds in `memory/skeletons/` designed to give local models a perfect first-build template:
    - `canvas_basic.html`: Clean 2D canvas with DPR scaling, frame loops, and window keyboard handlers. Kept for `skeleton_mode="default"` baseline; **no longer the retrieval fallback** (2026-05-21).
    - `canvas_basic_v2.html`: **New fallback (2026-05-21)** — denser bug-hardened scaffold pre-empting focus-blur, dt-cap, restart-cleanup, DPR-resize, lazy-audio, and HUD pointer-events failures. Used when no modality or sidecar scaffold matches.
    - `canvas_3d_basic.html`: Full 3D perspective setup utilizing CDN Three.js, lights, camera, and aspect-ratio fits.
    - `canvas_grid_basic.html`: Continuous tile-aligned corridor movement and corner snapping (e.g., Pac-Man, Sokoban).
    - `canvas_platformer_basic.html`: Gravity jumps, vertical ladder alignments, climbing, and platform landings (e.g., Donkey Kong).
    - `canvas_scrolling_basic.html`: Cam viewport horizontal scrolling and parallax backgrounds (e.g., Defender).
    - `canvas_mode7_basic.html`: Scanline perspective projection texture mapping for rotatable tracks (e.g., Mario Kart).
    - `canvas_crawler_basic.html`: Top-down dungeon rooms, spawner pools, wall-sliding, and multi-player bounding clamps (e.g., Gauntlet).
    - `canvas_mobile_basic.html`: Pointer Events touch joystick, tap-buttons, and mobile aspect letterboxing (e.g., iOS Safari).
    - `canvas_rpg_basic.html`: Grid-locked discrete stepping and lerp walking animations (e.g., Pokemon).
    - `canvas_cards_basic.html`: Mouse/touch drag-and-drop hit testing and grid snapping (e.g., Solitaire).
    - `canvas_physics_basic.html`: Gravity projectile trajectories, launch slingshots, and elastic boundary collisions.
    - `canvas_voxel_minecraft_basic.html`: Voxel/cube terrain, pointer-lock + WASD, build/break interaction.
    - `canvas_ar_flick_basic.html`: Pointer flick/swipe gestures with spin and gravity (e.g., Pokémon-Go-style throws).
    - `canvas_lit_dungeon_basic.html`: Composite-mode dynamic lighting + shadow gradients for top-down dungeons.
    - `canvas_vfx_particles_basic.html`: Pooled particle effects + screen shake + damage numbers.
    - `canvas_board_turn_basic.html`: **New (2026-05-21)** — 8×8 grid with click-to-select / click-to-move state machine, alternating `currentPlayer`, exposed `window.gameState` (e.g., Chess, Checkers, Go, Reversi).
    - `canvas_dom_basic.html`: **New (2026-05-21)** — DOM-only `<table>` + event delegation for UI-style apps (e.g., Tic-Tac-Toe, Calculator, Todo).
  - **Modality detector (added 2026-05-21)**: short goals like "chess" or "doom" tokenize to 1-2 non-stopword tokens — too sparse for Jaccard to clear the 0.30 sidecar threshold. `retrieve_skeleton()` now runs `_detect_board_intent` / `_detect_dom_intent` / `_detect_3d_intent` FIRST; a single strong-hook token (e.g. `chess`, `doom`, `tictactoe`, `calculator`) or ≥2 modality keywords commits the matching scaffold without going through Jaccard.
- **`Playbook`** — JSONL of bullets with `helpful` / `harmful`
  counters. Features elite math and physics rules for retro classics (Mode 7 scanning, wall-sliding, segmented follow, angle biasing, mobile joysticks, aspect ratio letterboxing) alongside the standard set, plus 2026-05-21 additions for turn-based board mechanics (`turn-based-select-move`, `board-grid-indexing`, `click-cell-from-pointer`) and the `expose-state-on-window` rule promoted from learned. Retrieval is weighted Jaccard × quality multiplier `1 + 0.10·tanh(score/5)`. `stage="plan"` returns broader top-K; `stage="code"` drops bullets with score ≤ -2. On-demand expansion via `<lookup_bullet>id</lookup_bullet>`.
- **Modality token expansion (added 2026-05-21)**: when retrieving for a goal that hits a modality detector, the matched keywords are appended to the query. Pac-man's `corner-sliding-alignment` jumps from 0.026 → 0.076 (~3×); doom's `tetris-matrix-rotation` retrieval noise disappears. Backwards-compatible: pass `modality_tokens=[]` or omit to get default behavior.
- **Dedup + budget capping**: `dedup_hits` (5-gram Jaccard ≥ 0.85) +
  `cap_hits_by_budget` run inside `render_playbook_block` by default.
- **Mistake memory (`games/game-memory/mistakes.jsonl`)**: seeded 2026-05-21 with 8 trace-derived signatures (window-state exposure, pointer-lock target, missing HUD elements, restart probe, partial-patch apply, duplicate top-level declarations). The diagnose prompt retrieves matching signatures so the model gets "you've seen this before — here's what worked" hints instead of re-diagnosing from scratch.
- **Won-skeleton promotion**: after `<confirm_done/>`, the agent
  copies the working HTML to `games/game-memory/skeletons/won_<session>.html`
  and indexes it so future sessions with similar goals can use it as
  the starting scaffold.

`memory/playbook.jsonl` is hand-curated. Add or revise bullets
manually; `memory.Playbook` rereads the file at session start.

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
  repeats, suffix loops in large bodies). Mid-stream watchdog (Ollama/MLX)
  also catches **semicolon-chained** one-line template loops and
  **mid-identifier** counter runs (`p.onGirder0`, `p.onGirder1`, …) that
  never produce newlines.
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
- **State-vs-render gap check** (`_ENTITY_RENDERED_JS`, always-on): for
  every top-level field on `window.state` / `window.gameState` that has
  numeric `.x` and `.y`, samples a 16×16 patch around the screen
  position (trying both raw-pixel and inferred-tile coordinate
  interpretations: 28/32/20/16/8-cell grids). If >80% of patch pixels
  are background-colored (low alpha OR within RGB tolerance of the
  top-left corner), appends a `ENTITY-NOT-RENDERED [name]` soft warning
  to the report. Catches the "Pac-Man without a Pac-Man" failure class
  — probes that test state existence can pass while the draw() function
  never references the sprite. Genre-free: candidates are found by
  *shape* (object with numeric x/y), not by name. The check flips
  `report["ok"]` to False so the model gets a fix turn with the
  warning visible.

**Multi-window playtest capture** (`LiveBrowser.record_playtest`): executes
a recipe's `input_script` and samples state (and optional screenshots) at
`sample_times_s` offsets. Used by the autonomous `/feedback` loop and by
`behavior_playtest` entries in `memory/playtests.jsonl` (e.g. frozen-game
progress check, forward-key vs facing-vector alignment, held-key canvas
bounds). Recipes are filtered in-page via `applies_when` — no genre names.

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
5. PIL Lanczos downscale to per-asset target size (default **512 px**
   as of 2026-05-23 — was 128 px, but tiny PNGs looked postage-stamp
   on modern displays and threw away most of the diffuser's detail;
   `drawImage` downscales at draw time if the game wants smaller).
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
| `/list`, `/models` | Inventory of MLX + Ollama models, with `[VLM]` / `[text]` labels. `*` = loaded in Ollama VRAM on any slot (11434–11436); `← active` = bound to this session (coder/model2/model3), not VRAM. |
| `/model <N\|name>`, `/load <N\|name>` | Stage the coder model (sticky across `/new`). |
| `/modelall <N\|name>`, `/loadall` | Stage the **same** model on all three Ollama slots (coder + critic + architect). Bare `/modelall` clears all three. |
| `/model2 <N\|name> [--role critic\|architect]` | Stage or hot-swap secondary model 2. Defaults to `critic` for VLMs and `architect` for text models unless role balancing chooses otherwise. |
| `/model3 <N\|name> [--role critic\|architect]` | Stage or hot-swap tertiary model 3. Used to split architect and critic roles in 3-model runs. |
| `/launch <N\|name\|path>` | Stage an MLX model for the next `/new` (in-process load on first request). |
| `/backend <auto\|ollama\|mlx\|openai\|anthropic>` | Switch the backend (cloud choices are explicit and billable). |
| `/unload` | Free VRAM: bare `/unload` evicts the active session tag; `/unload all` walks **every** Ollama slot (11434–11436) plus diffusers preload, not only `OLLAMA_HOST`; `/unload mlx` drops in-process MLX. |
| `/new` | Start a new session in the same workspace. |
| `/ship` | Force `<confirm_done/>` on the next critique turn. |
| `/quit` | Exit. |
| `/open` | Reveal the current HTML in the file browser. |
| `/log`, `/paths`, `/files` | Print log + artifact paths. |
| `/clear` | Clear the chat scrollback (does not reset the session). |
| `/iters <n>` | Change the iter budget (sticky for next `/new` and extensions). |
| `/ctx [N\|100k\|131k\|262k\|full]` | Ollama `num_ctx` / KV reservation (default **100k**; sticky). Bare `/ctx` shows current. Ollama reloads on change — preload with `ollama run --ctx-size N <model>`. |
| `/seed <path>` | Stage a baseline `.html` for the next `/new` (sticky). Bare `/seed` clears. Snapshots under `games/snapshots/<basename>/` resolve to the canonical `games/<basename>.html` so traces/assets keep one stem. |
| `/reset` | Wipe staged state (seed, models, iters, ctx → defaults). Does not stop a running session. |
| `/status` | Print model, phase, iteration, staged picks, ctx, paths. |
| `/retry` | Re-run after a bad model (keeps game file + trace; uses staged `/model` first). |
| `/wait <on\|off>` | Toggle step-mode — default ON in the TUI; when on, pause after each iter; when off, iterate continuously. |
| `/iter-detail <on\|off>` | Toggle optional expanded blocker detail after the compact iter decision line (default off). |
| `/mode <local_manual\|local_auto\|local_plus_review with <model> [--auto-apply]\|custom>` | Apply a run contract: manual checkpoints, autonomous loop, or autonomous loop with an explicit reviewer hook. |
| `/playbook`, `/memory` | Toggle playbook bullet injection (`on` / `off`; default on in production TUI). |
| `/prefill <on\|off>` | Toggle forcing assistant prefill tags (`<plan>`, `<diagnose>`) to prevent preamble talk and lock XML formatting (default ON). |
| `/architect <on\|off>` | Toggle architect/editor split (Aider's 2-call pattern) on complex first-builds — planning pass before the coder writes HTML (default off). Distinct from `/model2 … --role architect`, which assigns a **sidecar slot**. |
| `/allroles` [on\|off] | Toggle **architect-split + vlm-critique** together for **one** loaded LLM (no `/model2` / `/model3`, no extra GPUs). Bare `/allroles` flips on/off. Staged critic/architect slots still win when present. `/reset` clears the bundle. |
| `/double-screenshot <on\|off>` | Toggle capturing dual screenshots (startup and post-input) to help the model see movement/animation (default off). |
| `/vlm-critique <on\|off>` | Toggle VLM screenshot attachment during Phase C successful critique turns for layout and UI polishing (default off). Needs a VLM as the loaded model; uses slot 1 when no critic slot is staged. |
| `/feedback [on\|off]` | Toggle autonomous playtest loop (default **on**). Bare `/feedback` prints state without flipping. When on: after clean iters, genre-free behavior recipes may queue `[AUTONOMOUS PLAYTEST]` coaching. Test reports and the critic still run when off. |
| `/rawfeedback [on\|off]` | Your typed feedback goes to the model verbatim (default **on** as of 2026-05-23). The model sees only the basic USER FEEDBACK block around your literal text and decides for itself whether to `<patch>` or `<assets>`. Machine bug feedback (browser test reports, console errors, probes), playbook retrieval, autonomous playtest, and visual critic are UNAFFECTED — they always run. Flip to `off` to opt in to the classifier wrappers (MEDIA-CHANGE / ORIENTATION-CHANGE / SCOPE ARBITRATION / asset-stem-mapping). Sticky across `/new`. Aliases: `/raw`, `/raw-feedback`. |
| `/audit` | Per-bullet playbook earnings from trace history (fires, pass-rate, avg-iter). |
| `/restarts <N>` | Independent full restarts when iter-1 score is below 60 (sticky; default 2; `1` = off). |
| `/model-class <auto\|small\|mid\|large>` | Override system-prompt trim (sticky; default `auto` → lean ~5 KB schema). |
| `/check [<N\|model>]` | Run visual review on the latest screenshot. `N` selects by `/list` number. Bare `/check` uses the active VLM session model. `claude-…` routes Anthropic, `gpt-…` routes OpenAI, and other names resolve local MLX VLMs. In wait mode ON, suggestion is prefilled in input for edit/Enter; in wait mode OFF, coaching auto-queues to the next coding turn. |

### Model topologies (1, 2, and 3 model runs)

The TUI always has one **coder** model: `/model` or `/load` selects the
model that writes `<plan>`, `<html_file>`, `<patch>`, and normal fix
turns. Optional `/model2` and `/model3` sidecars do not replace the
coder; they take narrow roles so the main loop can keep one source of
truth for edits.

**1-model run.** Use only `/model`. The same model plans, codes, reads
browser reports, and decides whether to patch or ship. The deterministic
harness still runs every iter: micro-probes, Chromium load, input smoke
test, screenshots, canvas freeze checks, and model-authored probes.
If a local VLM is discoverable, the built-in vision judge can add visual
coaching; otherwise visual checks stay programmatic unless you run
`/check` explicitly. **`/allroles`** is the shortcut when you want that
one model to also run architect-split and VLM critique without staging
slot 2/3 (requires a VLM-capable model for critique screenshots).

**2-model run.** Use `/model` for the coder and `/model2 ... --role
critic` or `/model2 ... --role architect` for one sidecar:

- `--role critic` is best for a VLM. After each browser test it reviews
  the latest screenshot, and when before/after screenshots are available
  it compares them for visible movement, frozen screens, misalignment,
  clipped HUD, or other rendering bugs. Critic output becomes coaching
  for the next coder turn.
- `--role architect` is best for a strong text model. When architect
  split is enabled (`/architect on`), complex first builds get a short
  design pass before the coder writes HTML. The architect role is also
  used for the final exit-decision turn when available.

**Auto-staff (added 2026-05-21).** Setting `/model2 N --role critic` (or
`--role architect`) automatically enables the matching feature
(`vlm-critique` or `architect-split`) in one step — no separate toggle
needed. The same hook fires when a local VLM is loaded on **any** slot,
so single-model users with a VLM coder get the visual critic loop
without typing `/vlm-critique on`. The status panel marks auto-flipped
features with `[auto]`. Override anytime with `/vlm-critique off` or
`/architect off` — the explicit toggle wins and persists.

If you omit `--role`, VLMs default toward `critic` and text models
default toward `architect`.

**3-model run.** Use `/model` for the coder, then assign one sidecar as
architect and one as critic:

```text
/model 1
/model2 2 --role architect
/model3 3 --role critic
```

This gives the cleanest split: model 1 edits code, the architect handles
planning/exit decisions, and the critic reviews screenshots. If both
sidecars are assigned the same role, routing is deterministic but less
useful: architect prefers model 2, critic prefers model 3.

**Wait on vs. wait off.** Sidecar roles run after the verifier either
way. With `/wait on` (the TUI default), the agent pauses between iters
so you can inspect Chromium and add feedback before the next model turn.
With `/wait off`, the loop continues automatically; critic, vision, and
programmatic warnings are queued as coaching for the next coder turn.
Cloud models are never called silently: use an explicit cloud backend or
explicit `/check <cloud-model>` if you want one involved.

**Status panel (right column).** Refreshes about once per second while a
session runs.

- **Activity (coder / critic / architect)** — One line per role when
  configured: always shown with the same live detail (waiting for first
  token, prompt-eval progress on MLX, tok/s, STALLED). Coder uses the
  main model; sidecars use Model 2 / Model 3 when staged. Idle roles stay
  visible as `idle` so you can see all three slots at once in a 3-model
  run.
- **Model 2 / Model 3** — Sidecar model tag, backend, and GPU placement.
  `idle`, `streaming …`, `failed`, or `crashed` when something goes wrong.
  Footer may add `busy now: GPU N (role)` when multiple cards are active.
- **Queued for next user-turn** — Bold yellow banner when feedback or a
  plan answer is queued but the current stream has not finished yet
  (high-visibility; complements the lower `Queued (N):` summary).
- **GPU map → LLM** — One row per configured slot:
  - `Model 1 (coder) · <tag> · OLLAMA · GPU …`
  - `Model 2 (critic) · …` / `Model 3 (architect) · …`
  - Same tag on all three → **same GPU** on each row; Models 2–3 add
    `same VRAM` (one physical Ollama load, three logical roles).
  - `not loaded` before the first request to that tag.
- **GPU map → Diffusers** — Three lines, always shown:
  - `Z-Image-Turbo`, `SD-Turbo img2img`, `Stable-Audio`: `loaded · GPU N`
    (index shown for all three, including Stable-Audio) only after this
    session constructs the pipeline (first `<assets>` / sound generation).
    **`not loaded` does not mean GPU 0 is empty** —
    `nvidia-smi` may still show ~20 GB on `python` for the chat process
    (browser, torch, or stale VRAM). When that happens, a dim
    `chat process · GPU 0 ~20 GB` note explains it.
  - Footer: `VRAM · 0:20/48 · 1:27/48 · …` for all cards.

**Reading `nvidia-smi` alongside the panel**

| What you see | Meaning |
|---|---|
| `ollama` on GPU 1–3, ~22–35 GB each | Normal 3-model pin: one daemon per slot (11434–11436) |
| `ollama` on GPU 1+3, ~53 GB total | Tensor-split single daemon — run `/unload all` then `/new` to re-pin |
| `python` ~20 GB on GPU 0 | Usually **this TUI process**, not Ollama — not the same as “Z-Image loaded” in the panel until sprites run |
| Diffusers `not loaded` + GPU 0 busy | Normal for a text-only goal (chess, etc.) until the plan triggers asset generation |

**Multi-GPU workstation (automatic, no manual Ollama config).** On `/new`,
if Ollama is already **tensor-split** across two cards and the box has
48 GB-class GPUs, the agent **auto-unloads** those models (same mechanism
as `/unload`) so the next LLM request can reload cleanly. Diffusers pick
GPU **0** on the 4×48 GB workstation (skip Ollama slots 1–3 and cards
with large `ollama` / `python` compute).
Disable auto-unload with `AGENT_NO_AUTO_OLLAMA_GPU_FIX=1`. On **small
GPUs** (two ~8–16 GB cards), split may stay — the panel says that is
expected. If split persists, the yellow tip suggests `/unload all`.

**3-model run — one Ollama daemon per GPU (automatic on 4×48 GB Linux/NVIDIA).**
When `/model`, `/model2`, and `/model3` are all Ollama slots, `chat.py`
auto-pins them on this workstation shape:

- exactly four visible NVIDIA GPUs
- each GPU is 48 GB-class
- Linux with `nvidia-smi`
- not macOS / MLX
- no manual `OLLAMA_HOST2` / `OLLAMA_HOST3` already set

The TUI starts missing same-user daemons as:

```text
11434 → GPU 1
11435 → GPU 2
11436 → GPU 3
```

GPU 0 is left for the TUI / diffusers path. Z-Image-Turbo and Stable-Audio
load on **GPU 0** on this box (not “whichever card has the most free
VRAM” — that used to pick empty GPU 1 and collide with the coder).
Override with `DIFFUSER_CUDA_DEVICE=N`. `CODING_BOX_NUM_CTX` defaults to
**100000** (or **`/ctx`** in the TUI — e.g. `/ctx 262k` for long extensions). If an old
same-user single daemon on 11434 has a split model loaded, the TUI unloads
that model before restarting the daemon pinned to GPU 1. It does not use
sudo, does not edit systemd, and does not touch daemons owned by another
user; those become `single daemon fallback` in the status panel.

Manual override still works: set `OLLAMA_HOST2` / `OLLAMA_HOST3` yourself
and the TUI will respect them without rewriting placement.

The status panel maps each slot by endpoint, not just by model tag: the
same model on `11434` / `11435` / `11436` should show GPU 1 / GPU 2 / GPU 3,
with `pinned, not loaded` until that slot handles its first request. Default
Ollama context is **100k** (`/ctx` or `CODING_BOX_NUM_CTX`) to keep KV VRAM
reasonable on 48 GB cards. If allocation still fails, the Ollama call retries
once without the `num_gpu=999` offload hint while keeping the same `num_ctx`.
For long extension chats on one slot, raise context explicitly:

```bash
# TUI: /ctx 262k   — or before launch:
CODING_BOX_NUM_CTX=262144 .venv/bin/python chat.py
```

**Useful env vars (GPU / backends)**

| Variable | Effect |
|---|---|
| `LLM_BACKEND` | `auto`, `ollama`, or `mlx` (default: MLX on macOS, else `auto`) |
| `OLLAMA_MODEL` / `CHAT_OLLAMA_MODEL` | Force the Ollama tag for chat |
| `OLLAMA_HOST` / `OLLAMA_HOST2` / `OLLAMA_HOST3` | Ollama HTTP base per slot (coder / model2 / model3) |
| `OLLAMA_KEEP_ALIVE` | Ollama model residency between turns; default `-1` keeps the model loaded to prevent idle eviction during feedback pauses |
| `AGENT_NO_AUTO_OLLAMA_PIN=1` | Disable automatic 4-GPU per-slot Ollama daemon startup |
| `AGENT_NO_AUTO_OLLAMA_GPU_FIX=1` | Do not auto-unload split Ollama VRAM on `/new` |
| `CODING_BOX_NUM_CTX` | Ollama context window (default 100000; `/ctx` in TUI) |
| `DIFFUSION_MODELS_DIR` | Override root for Z-Image / SD-Turbo weights |
| `DIFFUSER_CUDA_DEVICE` | Force diffusers (Z-Image, Stable-Audio) onto a specific CUDA index; default on 4×48 GB Linux is GPU 0 |

### CLI flags (`coder.py`)

```bash
.venv/bin/python coder.py "<goal>" \
  [--max-iters N]            # iter budget (default 6)
  [--best-of-n N]            # best-of-N patch candidates per failed iter (default 3; 1=off)
  [--headless]               # no visible Chromium
  [--backend auto|ollama|mlx]
  [--model <name>]
  [--prompt-version v1]
  [--no-playbook]            # skip playbook retrieval
  [--no-skeleton]            # ignore skeleton library
  [--no-assets]              # skip Z-Image-Turbo
  [--no-sounds]              # skip Stable Audio
  [--num-ctx N]              # Ollama context (default 100000; env CODING_BOX_NUM_CTX)
```

---

## System tests and memory hygiene

```bash
# Visible system tests (multi-GPU plumbing + optional slow Pac-Man benchmark).
# Smoke suite (~5–10 min): plumbing + “player does not move” regressions.
python system_tests.py run --suite smoke --three-model --model qwen3.6:27b
# Pac-Man is slow (~15–30 min) — runner asks “Run now? [y/N]” unless you pass --yes.
python system_tests.py run --suite pacman --max-iters 3 --three-model
python system_tests.py run --suite full --three-model   # smoke first, then prompts for pacman-hard
python system_tests.py run --suite pacman --yes         # skip confirmation (CI / unattended)
python system_tests.py show run_20260521_120000         # SYSTEM_SUMMARY.md
python system_tests.py list

# Memory hygiene.
.venv/bin/python scripts/forget_session.py --list
.venv/bin/python scripts/forget_session.py <session_id>
./scripts/clean_artifacts.sh --yes                  # bulk wipe stale per-session artifacts
```

The canonical regression check is **asteroids**: `vx = cos(angle) *
speed` for ship direction, and asteroids drawn as **irregular
polygons** (not perfect circles). Run after any change to retrieval,
prompts, patches, or asset wiring.

### Validation matrix (loop reliability)

Use this matrix when validating `chat.py` loop/control changes:

| Scenario | Setup | Expected result |
|---|---|---|
| 1-model run | `/model <N>` only | Coder handles planning, edits, and exit decisions; harness and programmatic checks still verify every iter. |
| 2-model run | `/model <N>` plus `/model2 <N> --role critic` or `--role architect` | Coder remains the only editor; one sidecar contributes screenshot critique or architect planning/exit decisions. |
| 3-model run | `/model <N>`, `/model2 <N> --role architect`, `/model3 <N> --role critic` | Coder edits, architect plans/handles exit decisions, critic reviews screenshots and queues coaching. |
| 3-slot same model | `/modelall <N>` on 4×48 GB autopin box | Same tag on 11434/11435/11436 (GPU 1/2/3); roles coder / critic / architect; enables parallel best-of-N and non-blocking sidecars. |
| Asteroids regression guard | `/new asteroids` on local backend | Ship direction uses `vx = cos(angle) * speed`; asteroids stay irregular polygons. |
| Wait-mode manual control | Default TUI mode, or `/mode local_manual` | Agent pauses after each iter (`await_user`), accepts user feedback before continuing. |
| Autonomous loop control | `/mode local_auto` or `/wait off` | Agent iterates continuously without step pauses unless user explicitly enables wait mode. |
| Explicit external review (manual) | `/mode local_plus_review with <model>` and keep wait on | Reviewer guidance can be run via `/check <N\|model>`; suggestion is prefilled for edit/Enter. |
| Explicit external review (auto loop) | `/mode local_plus_review with <model> --auto-apply` with wait off | Failed iters can auto-run explicit reviewer hook and inject coaching for next turn. |
| No silent cloud guarantee | Local-only run (`/mode local_auto` or `/mode local_manual`) | No cloud calls unless user explicitly picks cloud backend or `/check ...`. |
| Active-stream cutoff regression guard | Rich-media first-build run on local MLX/Ollama model | If tokens or backend progress are still arriving, the model is working. Never stop it with an absolute wall-clock cutoff; only no-activity stalls, repetition loops, deliberation loops, backend max-token caps, hard crashes, or explicit user cancel may stop generation. |
| Style rebrand feedback | Mid-session: "all new graphics … look like X" with existing `_assets` | `style_rebrand_directive_active` in trace; mid-session `<assets>` may defer as `mid_session_assets_deferred_for_user_style` until coaching applies. |
| Autonomous playtest | Default TUI; `/feedback on` after clean iter on canvas game | `[AUTONOMOUS PLAYTEST]` may appear in `feedback_queued` when a `behavior_playtest` recipe fails; `/feedback off` suppresses only this path. |
| Single-LLM all roles | `/model <VLM>` then `/allroles on` | Architect-split + vlm-critique on slot 1; no second GPU. Critic routing uses the loaded model when no `/model2` critic is staged. |
| Ship while streaming | Ctrl+D or `done` during a long stream | Queued feedback (including autonomous findings) is **dropped** at ship; second Ctrl+D within 2s force-quits the TUI. |
| MLX stall | `LLM_BACKEND=mlx`, stall mid-stream | No silent fallback to Ollama — MLX-specific recovery hint only; use `/backend ollama` explicitly if you want to switch. |

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

### 4. Make critic findings harder to ignore

**Problem.** Sidecar critics and the local vision judge can now feed
coaching into the next coder turn, but most findings are still advisory.
A game can pass structural probes while the visual reviewer spots a
real playability problem, such as a player that appears stuck.

**Change.** Promote only high-confidence, generic critic findings into
harder gates: static before/after screenshots, missing player
locomotion after input simulation, blank or clipped canvas, and clear
asset/sound mismatches. Keep subjective art taste as coaching so weak
critics cannot block a working game.

**Effect.** One-, two-, and three-model runs all get stricter where the
evidence is objective, without turning the critic into an unreliable
second coder.

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

### 6. Per-phase model policy

**Problem.** The TUI already supports coder, architect, and critic
roles, but selection is still mostly manual. Some goals benefit from a
stronger architect, while others need the fastest patch model or the
best VLM critic.

**Change.** Add a lightweight policy layer that can recommend a role
assignment from local model inventory, VRAM budget, and goal modality.
The policy should only suggest local models unless the user explicitly
selects a cloud backend.

**Effect.** Users keep the current 1/2/3-model controls, but common
setups become one command instead of manual role assignment.

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
| `feedback_deferred_blocker` | Code feedback held while tests fail | Preview of deferred items; media-only may still run in parallel |
| `feedback_deferral_escalated` | Same ask deferred twice before | Third turn forces the request through with escalation text |
| `media_only_parallel_inject` | Art/sound feedback during blocker | Media peeled off the queue; runs on diffuser GPU this turn |
| `turn_contract` | Once per stream, just before model call | Routing contract for this turn: allowed/forbidden tags, scoped_mode, classifier flags, prompt section sizes |
| `patch_outcome` | After patch apply | Per-block success/failure and matcher tier for postmortem |
| `slow_prefill` | Prefill with no tokens yet | One-shot “still working” signal during long MLX/Ollama prefill |
| `media_change_directive_injected` | MEDIA-CHANGE block added to prompt | Routed feedback to asset/sound regen |
| `media_change_directive_suppressed` | MEDIA-CHANGE skipped | Reason is in `reason` field: `behavior_bug`, `behavior_scope_on_strict_turn`, `orientation_change`, `use_existing_media`, or `style_rebrand` (rebrand uses its own directive, not generic MEDIA-CHANGE) |
| `multi_frame_intent_detected` | Goal or feedback asks for animation rosters | Asset cap may be raised; plan nudge favors `from_image` chains |
| `style_rebrand_directive_active` | STYLE REBRAND block added | User asked to re-render all existing asset names with a new visual style (txt2img, not `from_image`) |
| `mid_session_assets_deferred_for_user_style` | Mid-session `<assets>` skipped | User queued a style rebrand; model tried to generate more sprites first — deferred to avoid wasted VRAM/time |
| `runaway_stream_warning` | Stream past ~15k completion tokens | TUI/log warning so you can Ctrl+D or wait; not an automatic abort |
| `autonomous_playtest_*` | After a clean iter (when `/feedback` on) | Recipe run, finding, or budget stop — see trace `detail` / `finding` fields |
| `autonomous_playtest_skipped` | Autonomous loop early-exited | `reason` field: `disabled:feedback_off / disabled:force_done / disabled:budget_exhausted / disabled:no_findings_streak / iter_failed / no_browser / no_behavior_recipes / recipe_load_error` |
| `iter_summary` | End of every iter | Structured summary: probes_passed/total, soft_warnings_count, page/console_errors, frozen_canvas, entity_missing_count, fail_reason |
| `surprise` | Auto-flagged across all phases | `category` field tags the class: `state_vs_render_gap` (probes pass but entity not drawn), `regression_after_clean_iter`, `non_slot1_bon_winner` |
| `best_of_n_attempt` | Fan-out best-of-N picked a winner | candidate_summary lists per-slot scores + temperatures; surfaces which slot won |
| `best_of_n_candidate_generated` / `_scored` | Per-candidate event during fan-out | slot, temperature, tokens, duration_s, eventual score |
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
in `tests/` and add a one-line bullet to `memory/playbook.jsonl` by
hand.

---

## Troubleshooting

- **`Playwright Chromium missing`**: run `env -u PLAYWRIGHT_BROWSERS_PATH
  .venv/bin/python -m playwright install chromium`.
- **`MLX out of memory`**: lower the wired-memory cap or pick a smaller
  quant. `MLX_PREFILL_STEP_SIZE=512` if you OOM mid-generation.
  DeepSeek-V4 Flash specifically requires 512 (auto-detected via path
  substring).
- **`Ollama context too small`**: raise with `/ctx 262k` or
  `CODING_BOX_NUM_CTX=262144`, and preload with `ollama run --ctx-size N <model>`
  before launching. Default is 100K.
- **`Vision judge never fires`**: check `VISION_JUDGE=0` isn't set;
  verify `backend.discover_local_vlm()` returns a path (run
  `.venv/bin/python -c "from backend import discover_local_vlm;
  print(discover_local_vlm())"`).
- **`Sprites blur or look postage-stamp tiny`**: default output size
  is 512 px (was 128 px before 2026-05-23). `drawImage` downscales
  cheaply if the game uses a smaller render rect. Ask for a smaller
  `size` only when you want tiny on-disk files (HUD icons:
  `"size": "32x32"`) or larger for full-screen overlays
  (`"size": 1024`).
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
- **`User wants new art style but model keeps redrawing wrong sprites`**:
  use phrasing like "all new graphics", "look like monsters", "not regular
  chess pieces" — the agent classifies **style rebrand** and injects a
  directive to regen each existing asset name with a new prompt (no
  `from_image`). If you queued that feedback during a long stream, wait
  for the iter boundary; mid-session `<assets>` may be deferred until
  the rebrand coaching lands.
- **`Feedback typed during a long stream seems ignored`**: check the
  status panel **Queued for next user-turn** banner — input injects at
  the next user-turn boundary after the current stream finishes.
- **`Autonomous coaching feels noisy`**: `/feedback off` disables only
  the extra playtest loop; harness failures and `/check` still apply.
- **`Stream ran 15k+ tokens with no end`**: look for `runaway_stream_warning`
  in the log; consider Ctrl+D and a smaller change (`<patch>`) on the next
  turn — the agent does not hard-abort active streams for token count alone.
- **`Shipped but my queued feedback never applied`**: by design after
  Ctrl+D / `done` / `/ship` — ship wins over the queue. Re-type the feedback
  and extend, or ship only after the stream ends.
- **`MLX stalled then Ollama errors appeared`**: MLX no longer auto-falls back
  to Ollama; follow the MLX hint or switch backends explicitly.
- **`Probes pass but nothing visible on canvas`**: check for
  `ENTITY-NOT-RENDERED` / `surprise` `state_vs_render_gap` — state exists but
  draw() never references the sprite.
- **`GPU map says Diffusers not loaded but nvidia-smi shows 20 GB python
  on GPU 0`**: the panel tracks **in-session** pipeline objects, not every
  byte in VRAM. GPU 0 is usually the **chat.py** process until Z-Image
  loads after the first sprite batch; restart the TUI if that VRAM is
  stale from a crashed run. Ollama VRAM is separate (often GPU 1+3).
- **`Status stuck on streaming architect`**: restart the TUI to pick up
  current code; sidecar activity should reset to `idle`, `failed`, or
  `crashed`. Check the log for `architect call failed` or
  `backend crashed mid-generation`.
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

---

## License

No `LICENSE` file ships in this repository yet. If you fork or
redistribute, add a license you are comfortable with (or confirm
terms with the repository owner on GitHub).
