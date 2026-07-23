# Coding Box — Local HTML Game Agent

> Repo: **`jmrothberg/Agent_learning`** (`origin`). Single source of truth — push to `main`.
> Remote: https://github.com/jmrothberg/Agent_learning  
> **Get newest code:** `./scripts/update.sh` or `git pull`

A specialist agent that drives a **small local LLM** (qwen3.6 27B via MLX in-process, or Ollama)
to write, test, and iteratively fix **single-file HTML5 games**, with a real **Chromium** browser
as the verifier and a local **diffusion** pipeline as the asset generator. Everything runs on your
machine — no cloud calls in the main loop, no server between you and your GPU.

Thesis: **a small *validated* model beats a large unvalidated one, and an agent that learns from
every session beats a static prompt.** Every reply is parsed, every patch applied via a 4-tier
match cascade, every iteration loaded in real Chromium, every sprite alpha-keyed, every regression
flagged, every session feeds a hand-curated playbook the agent reads back next launch.

Asset pipeline (self-contained, no runtime network once weights cache):

- **Sprites (macOS Apple Silicon):** **FLUX2-klein-9B** via **mflux** CLI (`mflux-generate-flux2`) —
  fast txt2img + init-image guidance; auto-selected when `FLUX2-klein-9B-mlx-8bit/` lives under
  `~/Diffusion_Models` (or `DIFFUSION_MODELS_DIR`). Installed by `./scripts/setup.sh` on Darwin.
- **Sprites (Linux / fallback):** **Z-Image-Turbo** (diffusers, txt2img 768×768 → downscaled,
  chroma-keyed to RGBA). SD-Turbo img2img for animation chaining only.
- **Sounds:** **Stable Audio Open** (OGG, 0.2–12 s) — preloaded at `chat.py` / `coder.py` launch
  *before* Playwright opens (required for diffusers fork safety).
- **Video:** **Wan2.2-TI2V-5B** cutscenes (MP4, 2–8 s — see [Video cutscenes](#video-cutscenes--wan22-ti2v-5b-local)).

Pose/animation frames are **txt2img with one shared character description + fixed seed** for
consistency — *not* img2img pose morphing (see [Animation](#animation--consistency-is-the-hard-constraint)).

---

## Contents
- [What this is](#what-this-is) · [How this compares](#how-this-compares-to-other-coding-agents) · [Quick start](#quick-start)
- [Overnight batches](#overnight-batches-10-games) · [Play sample games](#play-sample-games-goodgame) · [Architecture](#architecture)
- [The verification harness](#the-verification-harness-the-core-lever) · [Assets & animation](#animation--consistency-is-the-hard-constraint)
- [Memory / opening library](#memory--the-opening-library) · [TUI & CLI](#tui--cli-reference)
- [Standalone asset tools](#standalone-asset-tools) · [Video cutscenes](#video-cutscenes--wan22-ti2v-5b-local) · [System tests](#system-tests--memory-hygiene)
- [Standing rules](#standing-rules) · [Troubleshooting](#troubleshooting) · [Other docs](#other-docs)

---

## What this is

A coding agent narrowly specialized for **playable single-file HTML5 games with visual + audible
feedback**, running entirely locally. Two drivers:

- **`chat.py`** — Textual TUI with a visible Chromium beside the terminal (default). Mid-stream
  feedback queue, slash commands, asset picker, screenshot review, model swap, live status panel.
- **`coder.py`** — headless CLI for unattended runs. Same agent, same loop, no UI.

The loop (`GameAgent.run` in `agent.py`) is async and yields a stream of `AgentEvent`s; the drivers
only consume the stream. Three phases: **A — plan** (model emits `<plan>`/`<criteria>`/`<probes>` +
optional `<assets>`/`<sounds>`/`<videos>`); **B — build/iterate** (patch or full-file → micro-probes →
Chromium load → score against the model's own probes + harness gates); **C — self-critique** (one
turn after `<done/>`).

---

## How this compares to other coding agents

Thesis in one line: **a small *validated* model beats a large unvalidated one** — and **tuning
prompts + harness + memory beats swapping models.** This repo is not a general repo editor; it is a
**local, game-specialized loop** where Chromium + behavioral gates are the quality lever.

| Pattern | Cursor / Claude Code | Aider / OpenCode | **Coding Box (this repo)** |
|---|---|---|---|
| **Primary loop** | chat + tools + subagents | chat + apply_patch / edit | Phase A plan → Phase B patch/rewrite → Phase C self-critique |
| **Verification** | you run tests / linter | tests if configured | **always-on:** micro-probes → Playwright → model `<probes>` + genre-free gates (`tools.py`) |
| **Second opinion** | parallel subagents | — | `/critique` (scripted playtests) + `/vlm-critique` (VLM checklist); **serial on one GPU** — no hidden cloud reviewer |
| **Plan / progress** | plan.md, todo lists | — | Phase A `<criteria>`/`<probes>` + harness-seeded **task ledger** (goal clauses / outline order / optional model `<todos>`) |
| **Context** | rules, @files, very large ctx | repo + skills | opening book + playbook + lean prompt; **state-anchor compaction** near ~70% of `num_ctx` (default 100K) |
| **Edits** | search/replace tools | unified diff | `<patch>` SEARCH/REPLACE — 4-tier match cascade; markdown `SEARCH:`/`REPLACE:` pairs repaired in `patches.repair_reply` |
| **Assets** | none in-loop | none | **FLUX2-klein (mflux) on macOS** or Z-Image-Turbo + Stable Audio; Wan2.2 cutscenes via subprocess |
| **Memory** | repo files | conversation | hand-curated **`memory/`** opening book (JSONL — one line, no restart) |
| **Local LLM** | cloud-first | local or cloud | **MLX in-process** (macOS default) or Ollama; cloud only with explicit API key + `/backend` |
| **Regression** | CI you author | ad hoc | **pytest (~2258 tests)** + stub eval banks + opt-in `eval/eval_seed_edits.py` (materialization with `browser=None`) |

**Real advantages vs general agents:** playable-game verification (input smoke test, per-action
screenshots, sprite gates), and a full on-machine art/audio pipeline tied to the same loop.

**Known gap vs Cursor/Claude Code:** feedback passes through routing/classifiers before the model
sees it — misclassification can arm the wrong directive (art reprompt vs code patch). The feedback
router + golden eval banks (`eval/golden_feedback_flows.jsonl`) exist to guard that class.

**Trace/debug (LLM-facing):** persisted `.jsonl` carries milestone events only; use
`scripts/enrich_trace.py <session-id> --timeline` and `failure_class` on `iter_summary` — see
**`HARNESS_DEBUG.md`**.

---

## Quick start

```bash
./scripts/setup.sh                 # one-time: Python deps, Playwright Chromium, GPU stack + mflux (macOS)
.venv/bin/python scripts/_smoke_doom.py   # verify the asset pipeline end-to-end (~2 min cold)

.venv/bin/python chat.py           # TUI (recommended) — visible Chromium opens beside terminal
.venv/bin/python coder.py "build me a snake game with a wraparound board"   # one-shot CLI
```

**macOS sprite weights:** put `FLUX2-klein-9B-mlx-8bit/` under `~/Diffusion_Models` (setup writes
`DIFFUSION_MODELS_DIR` to `.env`). Linux/CUDA: use `Z-Image-Turbo/` in the same tree or let setup
download (~32 GB). `mflux` ships in `.venv` on Darwin via `requirements-diffuser.txt`.

**Backend selection** (macOS defaults to MLX; else `auto`):
```bash
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python coder.py "snake"   # pin MLX model
.venv/bin/python coder.py "snake" --backend ollama                           # force Ollama
```

**MLX upgrades — MiniMax-M3 (re-copy after every `pip install -U mlx-lm`):** PyPI `mlx-lm` ships
`minimax` (M2) but not `minimax_m3`. The pipenetwork MiniMax-M3-MLX weights bundle
`minimax_m3.py`; after you upgrade mlx-lm (or run `./scripts/install_mlx_v4_fix.sh`, which
reinstalls mlx-lm), copy it back into the venv or load fails instantly with
`Model type minimax_m3 not supported`:
```bash
cp ~/MLX_Models/MiniMax-M3-MLX-8bit/minimax_m3.py \
   .venv/lib/python3.12/site-packages/mlx_lm/models/minimax_m3.py
```
(Use the model dir you actually downloaded — path or HF id — if not under `~/MLX_Models`.)
Pick it in the TUI with `/list` then `/load` like any other MLX model.

**MLX upgrades — GLM-5.2 (`glm_moe_dsa` / IndexShare):** PyPI `mlx-lm` 0.31.3 maps
`glm_moe_dsa` to a bare `deepseek_v32` subclass and builds a DSA indexer on **every**
layer. GLM-5.2 only has indexer weights on `"full"` layers and reuses the previous
layer's top-k on `"shared"` layers. Loading without the fix fails instantly with
`ValueError: Missing 285 parameters: model.layers.*.self_attn.indexer...`.
```bash
./scripts/install_mlx_glm52_fix.sh           # install (auto-picks source; see below)
./scripts/install_mlx_glm52_fix.sh --status  # PR #1410 merge status + installed check
./scripts/install_mlx_glm52_fix.sh --rollback
```
The script checks whether [mlx-lm PR #1410](https://github.com/ml-explore/mlx-lm/pull/1410)
has merged: if yes it tries PyPI first (then `main` if the release lags), otherwise it
installs the PR head. After any mlx-lm reinstall, re-copy `minimax_m3.py` if you use
MiniMax-M3 (see above). Example weights: `pipenetwork/GLM-5.2-MLX-4bit`,
`mlx-community/GLM-5.2-mxfp4` — needs a very large unified-memory Mac (~370–480 GB
for 4-bit).

**Tests** (pure-function; see `TEST.md`; **batch / overnight commands:** `eval/OPERATIONS.md`):
```bash
.venv/bin/python -m pytest tests/ -q
```

See **`DEV.md`** for env vars (`LLM_BACKEND`, `MLX_MODEL`, `CODING_BOX_NUM_CTX`, …).

---

## Overnight batches (10 games)

**Easiest (no command line):** double-click **`Overnight.command`** in Finder. Terminal opens and asks:
1. prompt numbers (from the canned list)
2. max iterations
3. VLM critique yes/no
4. which MLX model
then starts the batch. Cursor still needs the watcher (Terminal prints the exact command).

**One sentence for another LLM:** user double-clicks `Overnight.command` (or `bash eval/overnight.sh`); agent starts `.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_N --jobs-total K --interval 30 --sync-loop` in a Cursor Shell (`block_until_ms=0`). CLI: `bash eval/overnight.sh --prompts 54,28 --model GLM-5.2-MLX-4bit --vlm no`.

Two processes run in parallel — **both required**:

| Where | Command | Purpose |
|-------|---------|---------|
| **Terminal.app** | Double-click `Overnight.command` **or** `bash eval/overnight.sh` | Interactive Q&A then builds games (visible Chromium → `games/tune_serial10/run_N/`) |
| **Cursor** | Watcher line printed by overnight (e.g. `tune_overnight_monitor.py … --jobs-total K`) | Polls every 30s; triage traces and patch harness/memory while the batch keeps going |

**Resume vs fresh:** auto next `run_N` each night (`--resume` skips completed labels). Use a new run id for a clean batch.

**Watcher loop:** when `agent_monitor.json` shows `completed_count` advanced, timeline the newest trace (`scripts/enrich_trace.py <trace> --timeline`), classify `failure_class`, patch source — do **not** stop the Terminal batch.

Full rules and legacy `tune_runNN.sh`: **`eval/OPERATIONS.md`**.

---

## Play sample games (`goodgame/`)

Curated wins live in **`goodgame/`**. **Play** links use **GitHub Pages** (same setup as
[jmrothberg/Games](https://github.com/jmrothberg/Games)) — click to run in your browser, no clone,
no GPU. Relative `*_assets/` and `*_sounds/` folders load from the same origin.

**Launcher:** [https://jmrothberg.github.io/Agent_learning/goodgame/](https://jmrothberg.github.io/Agent_learning/goodgame/)

| Game | Play |
|------|------|
| Open Field Tower Defense | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-an-open-field-tower-defe_20260625_144848%20copy.html) |
| Asteroids | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-the-classic-asteroids-ga_20260612_222054.html) |
| Centipede Arcade | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/centipede-game-with-super-nice_20260512_180020.best.html) |
| Centipede Shooter | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-a-centipede-style-fixed_20260615_154952.html) |
| Girder Climb (platformer) | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-a-single-screen-arcade-p_20260620_225138.best.html) |
| Dragon's Lair Deluxe | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/dragons-lair-deluxe.html) |
| Street Fighter | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/a-game-of-street-figher-a-two_20260525_151525.html) |
| Mortal Kombat | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/game-of-mortal-kombat-fighing_20260524_101226.html) |
| Dojo Fighters | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-a-single-screen-2d-fight_20260615_181442.html) |
| Space Invaders | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/game-of-space-invaders-with-an_20260512_165800.best.html) |
| Chess | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/mechanics-standard-chess-on-an_20260522_163629.html) |
| The Secret of Skull Island (adventure) | [Play](https://jmrothberg.github.io/Agent_learning/goodgame/build-a-point-and-click-advent_20260621_150955.html) |

After pushing a new game to `goodgame/`, wait ~1 minute for Pages to rebuild (or run
`gh api repos/jmrothberg/Agent_learning/pages/builds --method POST`).

**Local clone** (optional — same files, `file://` from disk):

```bash
open "goodgame/build-an-open-field-tower-defe_20260625_144848 copy.html"   # macOS
```

In the TUI, **`/goodgame`** copies the current session's best build into this folder for tracking.

---

## Architecture

Files that carry the weight:

| File | Role |
|---|---|
| `tools.py` | **The verifier.** `LiveBrowser.load_and_test` (Chromium), `_input_smoke_test` (presses keys, captures per-action frames, runs the gates), micro-probes. Highest-leverage file. |
| `agent.py` | Orchestrator (`GameAgent.run`); phase methods + mixin map → **`AGENTS.md` §1b** |
| `assets.py` / `sounds.py` | Sprites: **FLUX2-klein (mflux CLI)** on macOS when local weights + binary exist, else Z-Image-Turbo (diffusers). Sounds: Stable Audio Open (always preloaded before browser). `render_asset_paths_block` injects the `sprite()` loader. |
| `videos.py` | `<videos>` cutscene clips via Wan2.2-TI2V-5B in a **subprocess** (`scripts/generate_video.py` — mlx-gen on Mac, diffusers on Linux). `render_video_paths_block` injects the `<video>`-overlay loader. |
| `backend.py` | MLX (in-process `mlx_lm`/`mlx_vlm`) + Ollama backends; sampler; VLM image path. |
| `modality.py` | Genre-free rendering-shape detectors (3D, wireframe, FPS nav modality); shared by `prompts_v1.py` and `memory.py`. |
| `prompts_v1.py` | Data-driven system prompt (`build_system_prompt` walks a `FormatSpec` list — don't hand-edit the rendered blob). |
| `memory.py` | `GameMemory` (skeleton retrieval), `Playbook` (Jaccard bullet retrieval), opening-book outlines/recipes. |
| `patches.py` | SEARCH/REPLACE engine: 4-tier match cascade (exact → normalized → whitespace → trimmed), `repair_reply`. |

### Patch engine & compaction

Prefer `<patch>` once `best.html` exists. Patch cascade and compaction rules: **`DEV.md`** · tests: `tests/test_patches.py`, `tests/test_compaction.py`.

---

## The verification harness (the core lever)

**Most "agent failures" are verifier failures** — the harness said `ok=True` while the game was
broken, so the fix loop never engaged. Probes often check **state** while **pixels/behavior** are
wrong; this project layers structural, behavioral, and visual checks on every iter.

**Per iteration (when a browser is configured):**
1. **Micro-probes** (pre-Chromium) — HTML completeness, bracket balance, elision sentinels.
2. **Chromium** (`LiveBrowser.load_and_test`) — console/page errors, RAF, input smoke test, model
   `<probes>`, screenshots, behavioral gates (`PLAYER-STUCK`, `ACTION_DRAWN_NOT_SPRITED`, …).
3. **Optional reviews** — `/critique` (scripted playtests) and `/vlm-critique` (local VLM checklist).

`<done/>` needs a **clean streak** (default 2 consecutive `ok=True` iters). Gate reference,
`failure_class` triage, and the trace timeline workflow: **`HARNESS_DEBUG.md`**.

**Eval mode (`browser=None`):** seed-edit regression (`eval/eval_seed_edits.py`) skips Chromium
entirely — micro-probes still run; each iter emits `iter_summary` with `test_skipped:no_browser`
(not `harness_crash`). Use `--patch-only` to skip Phase A planning + phase-A asset gen on canvas
seeds.

---

## Animation — consistency is the hard constraint

A character that kicks/punches must be the **same** character across frames. That's the entire
reason the asset pipeline exists, and it dictates two hard rules:

- **img2img cannot change a pose** (Z-Image at `guidance_scale=0` → locked to idle). `from_image` is
  recolor/restyle only.
- **Fresh txt2img replacement breaks consistency** with art already in the game. **Never regenerate
  a pose frame to “fix” a dead one.** Near-identical frames are **cosmetic** — advisory only; must
  not flip `ok=False` or defer user gameplay feedback.

At **plan time**, request all named pose frames as **txt2img with one shared character description +
fixed seed**. In-session, cycle the frames you have and convey the action with the **sprite**, never
a code-drawn limb. Wrong sprite direction → flip in **code** (`ctx.scale(-1,1)`), don't regenerate.

Sounds: Stable Audio Open, OGG, ≤12 s. A cross-session **asset library** under `memory/` lets
admitted sprites/sounds compound across sessions like the playbook does.

---

## Two reviews: `/critique` and `/vlm-critique`

Everything collapses into **two reviews**. Both look for problems and **hand them back to the
coding agent so it fixes them next round**. The only difference is whether it uses eyes.

| Command | Looks at the screen? | Default | What it does |
|---|---|---|---|
| **`/critique`** | No | ON | Plays the game (scripted input from `memory/playtests.jsonl`) + reads the report, sends problems to the agent |
| **`/vlm-critique`** | Yes | OFF | A vision model looks at the screenshot, uses a memory checklist when one fits, sends problems to the agent |
| *(harness — always on)* | — | on | Chromium load, probes, console errors |
| `/check` | Yes | — | Look at the screen once, on demand |
| your typed notes | — | — | Type any time (`/rawfeedback` controls wrappers) |

The word **"feedback" is retired** as a name — `/playtest` and `/feedback` still work as silent
aliases of `/critique`, and `/judge` / `/vc` as aliases of `/vlm-critique`.

**`/critique` (no vision)** runs browser automation recipes (keydown timelines, state deltas,
"did pixels change?" hashes). Catches frozen loops, dead controls, wrong facing→movement math.
It **cannot** judge sprites, layout, or art.

**`/vlm-critique` (vision)** is the one that answers "does my game look good?" Which model does the
looking (first match wins):
1. **model 2** staged as critic (`/model2 … --role critic`) — the fallback when your main model can't see;
2. else the **main model**, if it is a VLM;
3. else skip (a blind model never pretends to see).

There is **one** structured critic path: it pulls a mechanism checklist from
`memory/visual_playtests.jsonl` when one matches (else an open-ended look). When no separate critic
is staged, the coder reviews its own screenshot — no second model is loaded. Staging `--role critic`
only changes **which** model looks, not the path (the old lightweight open-ended judge is retired).

For a full "is it good?" answer, run **both**.

**Typical setup (Mac + large coder LLM):**

```text
/model2 2 --role critic    # optional: a small VLM does the looking so the huge coder isn't fed screenshots
/vlm-critique on           # vision review (off by default)
/critique on               # no-vision review (on by default; complements vision, doesn't replace it)
```

Manual spot-check while iterating: `/check`, or type notes at a `/wait` pause.

---

## Memory — the opening library

The Python engine stays general; `memory/` is a hand-curated "opening book" (chess-style) that
helps **any class of game**, retrieved at runtime — add one JSONL line, no restart. Genre-free by
shape, not subject.

| File | Holds |
|---|---|
| `memory/playbook.jsonl` | code rules-of-thumb, retrieved by weighted-Jaccard on the goal (tags weigh 2×; ~0.02 floor) |
| `memory/plan_nudges.jsonl` | plan-turn modality nudge **prose** (detectors live in `prompts_v1.py`; TD-vs-brawler suppression in `visual_playtests.jsonl`) |
| `memory/feedback_patterns.jsonl` | extra feedback-classifier phrases (unioned in `agent_feedback.py`) |
| `memory/playtests.jsonl` | behavior recipes for `/critique` (scripted input, state, pixel hash — no VLM) |
| `memory/visual_playtests.jsonl` | mechanism-keyed yes/no VLM checklists + `auto_probes` + per-question `fix_hints` |
| `memory/implementation_outlines.jsonl` | architect mechanism outlines (retrieved k=1); deep render carries a probe-authoring contract tied to each outline's `state:` fields |
| `memory/components.jsonl` | paste-and-adapt JS snippets (game loop, input, camera, pointer-lock, spatial hash, loaders) |
| `memory/asset_audits.jsonl` | generated-art loader/decode/fallback audits (incl. missing-asset placeholder, chroma-key paths) |
| `memory/animation_audits.jsonl` | motion audits (midframe, facing-flip, walk-cycle advance, action-frame reset) |
| `memory/skeletons/` | first-build HTML templates per mechanism (`.html` + `.json` sidecar) |
| `memory/prompt_library.jsonl` | the curated `/games` prompts |
| `memory/system_battery.jsonl` | default system-test battery (committed — useful on every machine) |

Grounding is fully offline: the old `/wiki` Wikipedia lookup was **removed** (2026-06-24, 0/10
empirical hit rate) — the opening library above is the single source of mechanism grounding.

**Skeleton routing** (`retrieve_skeleton`): a modality detector (board/DOM/3D strong-hooks) runs
first; then the **recipe→skeleton** map reuses the already-accurate visual-recipe matcher; then a
distinctiveness-gated Jaccard fallback; else the generic `canvas_basic_v2`. A 2D-arcade goal must
never inherit a 3D/board/dungeon scaffold (regression-tested). Coverage of the curated prompts is
checked model-free by `eval/eval_prompts_plan.py --coverage`.

---

## TUI & CLI reference

**Key TUI slash commands** (`/help` lists all): `/allroles` (architect-split + visual critic on one
loaded LLM) · `/critique [on|off]` (no-vision review, default on; aliases `/playtest`, `/feedback`) ·
`/vlm-critique [on|off]` (vision review, default off; alias `/judge`) · `/wait [on|off]`
(step-mode pause per iter) · `/games [N]` (load a curated prompt) · `/ctx N` (context window) ·
`/ref <path>` (attach a reference image for a VLM turn) · `/check [with <model>]` (explicit cloud or
local VLM screenshot judge — never auto-called) · `/goodgame` (copy the trio into tracked `goodgame/`).

**`coder.py` flags:** `--backend {auto,ollama,mlx,mlx-server}` · `--model` · `--max-iters` · `--best-of-n` ·
`--num-ctx` · `--headless` · `--step` · `--restart-n` · `--playbook` (retrieval; off by default on
the CLI). Goal is the positional arg.

**Model topologies:** 1 model = all roles multiplex one loaded LLM (the `/allroles` default on a
single machine). On a multi-GPU box, slots 2/3 can host a dedicated architect/critic; the diffuser
pins to GPU 0 so the LLM slots stay free (`gpu_status.pick_diffuser_cuda_index`).

---

## Standalone asset tools

Two small terminal utilities outside the agent loop — same sprite pipeline as the agent
(FLUX2-klein on macOS, Z-Image-Turbo elsewhere) and asset cache, no LLM required.

**Draw sprites interactively** — walks you through style, single vs animation mode, and each
prompt; writes PNGs to `games/_draw/<project>_<timestamp>_assets/` (chroma-keyed RGBA, same as
the agent). Animation mode chains pose frames off a base idle frame (txt2img merge, not img2img).

```bash
.venv/bin/python scripts/draw_game_art.py
```

**Preview a folder** — drag a `*_assets/` or `*_sounds/` folder onto the terminal (or pass the
path). Audio folders play each clip in order; image folders open in Preview.

```bash
.venv/bin/python scripts/play_folder.py                    # then drag folder + Enter
.venv/bin/python scripts/play_folder.py path/to/my_assets  # or pass path directly
```

**Asset Studio (browser UI)** — drag a PNG, describe a change (or start from text only), generate
with the same sprite backend as the agent, and save into any `*_assets/` folder with the filename you
pick. Modes: new sprite/background (txt2img), modify existing (img2img / init-image on macOS), or save/rename only.

**Double-click** `scripts/Asset Studio.command` in Finder (starts the sprite stack + opens the browser —
you never type a python command). While `chat.py` is running, the same UI is already at
http://127.0.0.1:8765/ — bookmark it.

```bash
open "scripts/Asset Studio.command"   # same as double-click in Finder
# manual fallback only:
.venv/bin/python scripts/asset_studio.py
```

Requires the same GPU setup as the agent: `./scripts/install_diffuser.sh` for drawing; preview
works with macOS `afplay` / Preview only.

---

## Video cutscenes — Wan2.2-TI2V-5B (local)

The agent can generate short MP4 cutscene clips (intro / death / victory / boss reveal) the same
way it generates sprites and sounds: the model emits a `<videos>` block at plan time (or
mid-session) and the harness injects the resulting file paths plus a proven `<video>`-overlay
loader pattern into the next build prompt. Gameplay always stays on the canvas — clips are
skippable overlay moments with a mandatory continue-on-failure path, so a missing or blocked video
can never stall the game.

```text
<videos>
[
  {"name": "intro",   "prompt": "slow push-in toward the castle at dusk, bats circling", "image": "key_intro", "seconds": 4},
  {"name": "victory", "prompt": "the knight raises the sword, confetti falls, camera orbits", "seconds": 4}
]
</videos>
```

The optional `"image"` field names a key-art **asset from the same session** — the clip is then
**image-to-video**, so cutscenes stay style-consistent with the in-game sprites (the recommended
flow: declare a 768px establishing-shot asset, then animate it). Without `image` it's
text-to-video. `seconds` clamps to 2–8 (default 4). Clips are silent — pair with `<sounds>`.

**Per-platform backends** (both behind `scripts/generate_video.py`; the agent shells out, so no
video deps ever load into the agent process):

| Platform | Backend | Model | Install |
|---|---|---|---|
| macOS (Apple Silicon) | `mlx-gen` CLI in the dedicated `.venv-video/` | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` (~17 GB, lazy) | `./scripts/setup.sh` (step 7) |
| Ubuntu / Linux (CUDA) | diffusers `WanPipeline` in the main `.venv` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~25 GB, lazy) | covered by `./scripts/install_diffuser.sh` (includes `imageio` + `imageio-ffmpeg` for MP4 export) |

**Linux model path:** `VIDEO_MODEL` (or the default hub id) must be a **Diffusers** tree — `model_index.json` plus `vae/`, `transformer/`, `text_encoder/`. An original Wan checkpoint folder (flat `*.pth` / sharded safetensors, no `model_index.json`) will not load. Prefer the HF id or a local Diffusers snapshot; leave Macs on mlx-gen.

**Linux VRAM tip (2×24 GB boxes):** Wan needs a mostly free GPU. If Ollama is holding both cards, unload it before cutscenes (`curl` `keep_alive:0`, or TUI `/unload`); the next coder turn reloads the model automatically as long as `ollama serve` is running. Macs are unchanged (mlx-gen / in-process MLX).

Standalone CLI (no LLM needed) — text-to-video and image-to-video:

```bash
# T2V
.venv/bin/python scripts/generate_video.py \
    --prompt "a knight runs across a collapsing drawbridge, cinematic" \
    --out games/mygame_videos/intro.mp4

# I2V — animate a Z-Image still so the clip matches the game art
.venv/bin/python scripts/generate_video.py \
    --prompt "the dragon slowly wakes and rears its head, embers drift" \
    --image games/mygame_assets/key_dragon_wake.png \
    --out games/mygame_videos/dragon_reveal.mp4
```

Defaults: 832×480, 49 frames written at 12 fps (~4 s of slow cinematic motion). Width/height must
be multiples of 32; frames must be 4k+1. Measured cost: **~3 min per 4 s clip on an M3 Ultra**, so
the prompt guidelines cap sessions at 2–4 clips. Generated clips cache under `games/_video_cache/`
(re-runs are free). Env vars: `VIDEO_MODEL` (override model id/path for either backend),
`VIDEO_VENV` (mlxgen venv location, default `.venv-video/`), `DIFFUSER_CUDA_DEVICE` (pin the CUDA
index on Linux, same var the sprite/sound diffusers honor). If no backend is installed the whole
feature is a silent no-op — the model is told to ship without cutscenes. Skip the install with
`./scripts/setup.sh --no-video`.

---

## System tests & memory hygiene

See **`TEST.md`** and **`eval/OPERATIONS.md`**. Quick check:

```bash
.venv/bin/python -m pytest tests/ -q
```

Battery: `memory/system_battery.jsonl`.

---

## Other docs

Each file has one job — avoid duplicating long gate/env lists across them.

| File | Audience | Purpose |
|------|----------|---------|
| `AGENTS.md` | maintainers + LLMs | Source vs artifacts, mixin map, trace paths — **new maintainer start** |
| `DEV.md` | **LLM agents + humans** | Commands, env vars, architecture (maintainer only — not injected into game LLM) |
| `TEST.md` | humans + LLMs | Three-layer tests, suite map, scripts inventory |
| **`eval/OPERATIONS.md`** | **humans + LLMs** | **“Run pytest / batch N games / triage run_XX” — natural-language → command** |
| **`HARNESS_TUNING.md`** | harness tuners | **Start here for new agents** — harness vs memory, traps, fix loop |
| `HARNESS_DEBUG.md` | harness tuners | Gate list, `failure_class`, trace timeline workflow |
| `FOR_NEXT_LLM.md` | — | Legacy redirect → `HARNESS_TUNING.md` |
| `eval/PARALLEL_MLX_TESTING.md` | batch eval | One MLX server, N parallel clients |

---

## Standing rules

Full list: **`DEV.md`**. Summary: tune the agent not the model; genre-free code; visible Chromium;
Asteroids regression; cosmetic sprite warnings are advisory; don't commit `games/` output.

---

## Troubleshooting

- **All sounds fail with `bad value(s) in fds_to_keep`:** Stable Audio tried to load *after*
  Playwright opened. Restart `chat.py` / `coder.py` — `assets.preload()` now always preloads audio
  at launch (even when macOS uses FLUX2 klein for sprites). Do not set `SKIP_DIFFUSER_PRELOAD=1`
  if you want `<sounds>`.
- **Sprites download Z-Image-Turbo on macOS despite local FLUX2:** install mflux in the venv
  (`./scripts/setup.sh` or `.venv/bin/pip install -r requirements-diffuser.txt`) and ensure
  `FLUX2-klein-9B-mlx-8bit/` is under `DIFFUSION_MODELS_DIR`. Incomplete HF Z-Image snapshots
  are skipped automatically.
- **Chromium won't launch:** `env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python -m playwright install chromium`.
- **MLX cold-load is slow (~30–60 s first request):** the 27B mxfp8 loads into VRAM in-process; preload with `ollama run --ctx-size N <model>` for the Ollama path.
- **`Model type minimax_m3 not supported` after an mlx-lm upgrade:** re-copy `minimax_m3.py` from your MiniMax-M3-MLX model dir into `.venv/lib/python3.12/site-packages/mlx_lm/models/` (see **MLX upgrades — MiniMax-M3** above).
- **`Missing 285 parameters` / `indexer.k_norm.bias` loading GLM-5.2:** PyPI mlx-lm lacks IndexShare — run `./scripts/install_mlx_glm52_fix.sh` (see **MLX upgrades — GLM-5.2** above).
- **"Feedback doesn't stick" / patches fail with SEARCH-not-found:** compaction shredded the file view — check `num_ctx` (default 100K) and the `structured_compaction` trace events.
- **Game shows colored boxes instead of art:** sprite-key mismatch — the `sprite()` resolver and `ASSETS_LOADED_BUT_UNDRAWN` gate now catch it; re-run.
- **Model loops/truncates on big builds:** expected for a 27B; the MLX sampler passes `top_p`/`top_k` (vendor coding preset) to avoid the degenerate line-repeat. Judge harness signals from the trace, not always a finished game.
- **Debugging a bad session:** `HARNESS_DEBUG.md` — start with `scripts/enrich_trace.py <id> --timeline`.
  Tuning traps: `HARNESS_TUNING.md`.

## Dependencies
Python 3.12, `mlx-lm` / `mlx-vlm` (Apple Silicon) or `ollama`, Playwright Chromium, `diffusers` +
`torch` (Z-Image-Turbo + Stable Audio Open on Linux/CUDA), **`mflux`** (macOS FLUX2-klein sprites).
Install via `./scripts/setup.sh`. See `requirements*.txt` — `requirements-diffuser.txt` layers
soundfile/torchsde and `mflux>=0.18.0` on Darwin only.

## License
See repository.
