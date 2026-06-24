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

Asset pipeline (self-contained, no runtime network once weights cache): **Z-Image-Turbo** for
sprites (txt2img, 768×768 → downscaled, chroma-keyed to RGBA), **Stable Audio Open** for sounds
(OGG, 0.2–12 s), and **Wan2.2-TI2V-5B** for video cutscenes (MP4, 2–8 s — see
[Video cutscenes](#video-cutscenes--wan22-ti2v-5b-local)). Pose/animation frames are **txt2img with
one shared character description + fixed seed** for consistency — *not* img2img (see
[Animation](#animation--consistency-is-the-hard-constraint)).

---

## Contents
- [What this is](#what-this-is) · [Quick start](#quick-start) · [Architecture](#architecture)
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

## Quick start

```bash
./scripts/setup.sh                 # one-time: Python deps, Playwright Chromium, GPU stack
.venv/bin/python scripts/_smoke_doom.py   # verify the asset pipeline end-to-end (~2 min cold)

.venv/bin/python chat.py           # TUI (recommended) — visible Chromium opens beside terminal
.venv/bin/python coder.py "build me a snake game with a wraparound board"   # one-shot CLI
```

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

**Tests** (pure-function; see `TEST.md`):
```bash
.venv/bin/python -m pytest tests/ -q
```

See `CLAUDE.md` for the full env-var reference (`LLM_BACKEND`, `OLLAMA_MODEL`, `MLX_MODEL`,
`CODING_BOX_NUM_CTX`, `MLX_TOP_P`/`MLX_TOP_K`, `DIFFUSION_MODELS_DIR`, …).

---

## Architecture

Files that carry the weight:

| File | Role |
|---|---|
| `tools.py` | **The verifier.** `LiveBrowser.load_and_test` (Chromium), `_input_smoke_test` (presses keys, captures per-action frames, runs the gates), micro-probes. Highest-leverage file. |
| `agent.py` | Async `GameAgent.run` loop, prompt assembly, compaction, memory retrieval, `run_visual_critic`, fix-prompt building. |
| `assets.py` / `sounds.py` | In-process Z-Image-Turbo / Stable Audio (lazy GPU load). `render_asset_paths_block` injects the `sprite()` loader. |
| `videos.py` | `<videos>` cutscene clips via Wan2.2-TI2V-5B in a **subprocess** (`scripts/generate_video.py` — mlx-gen on Mac, diffusers on Linux). `render_video_paths_block` injects the `<video>`-overlay loader. |
| `backend.py` | MLX (in-process `mlx_lm`/`mlx_vlm`) + Ollama backends; sampler; VLM image path. |
| `prompts_v1.py` | Data-driven system prompt (`build_system_prompt` walks a `FormatSpec` list — don't hand-edit the rendered blob). Modality detectors (`_detect_art_intent`/`_detect_3d_intent`). |
| `memory.py` | `GameMemory` (skeleton retrieval), `Playbook` (Jaccard bullet retrieval), opening-book outlines/recipes. |
| `patches.py` | SEARCH/REPLACE engine: 4-tier match cascade (exact → normalized → whitespace → trimmed), `repair_reply`. |

### Patch engine
`<patch>` blocks use a SEARCH/REPLACE format matched **exact → char-preserving-normalized (smart
quotes/dashes) → whitespace-collapse → trimmed**, with cross-patch uniqueness + non-overlap
checks; surviving patches apply in reverse source-order. `repair_reply` strips BOM/CRLF/internal
fences and collapses malformed doubled `=======`/`>>>>>>> REPLACE` markers. Once `best.html`
exists, prefer patches — full `<html_file>` rewrites on a working game amplify regressions.

### Compaction (two-tier, token-aware)
Pressure = `prompt_tokens / num_ctx`. ≤5 turns: no-op. ≤14: per-turn `<html_file>` body elision.
\>14 **or** >~70% of `num_ctx`: replace history with one deterministic **state-anchor** message
(goal / criteria / probes / progress / diagnoses / last report / files / **generated asset paths**).
Default `num_ctx` is **100000** — a wrong (too-small) denominator makes compaction fire every turn
and shred the playbook + the user's instructions.

---

## The verification harness (the core lever)

**Most "agent failures" are verifier failures**: the harness said `ok=True` while the game was
broken, so the fix-loop never engaged. The trap is that probes check **state** (`state==='punch'`,
a field changed) and pass while the **pixels/behavior** are wrong. So the harness layers structural,
behavioral, and visual checks — and the highest-value work in this project is **gate fixes**.

**Layers per iteration:**
1. **Pre-Chromium micro-probes** — HTML completeness, bracket balance, elision sentinels, duplicate
   top-level declarations, API allowlist. Cheap; rejects truncated streams before a browser load.
2. **Chromium** (`load_and_test`, Playwright, visible by default) — console/page errors, RAF firing,
   blank/frozen-canvas, listener counts, the **input smoke test**, the model's `<probes>`,
   screenshots, an `__audioEvents` shim, and draw-call shims (`__drawImageEvents`, `__fillRectEvents`,
   `__strokeEvents`).
3. **Visual critic** (`run_visual_critic`, local VLM) — a mechanism-keyed yes/no checklist judged
   against the screenshots; catches what probes can't (facing direction, attack pose).

### The gates (all in `tools.py`, genre-free, all flip `ok=False`)
- **PLAYER-STUCK** — a movement key registers but no *position* leaf changes (spawned in a wall).
- **ACTION_DRAWN_NOT_SPRITED** — an action key changed the canvas by code-drawing
  (`fillRect`/`lineTo`/`stroke`) but drew **no new sprite** = a faked action (a kick drawn as lines
  over idle instead of swapping to the kick sprite).
- **CODE_DRAWN_OVER_SPRITE** — the action drew its sprite **but also** code-drew stroke/arc shapes on
  top (a "motion line + flash"/limb). The sprite conveys the move; the overlay is rejected junk.
- **ASSETS_LOADED_BUT_UNDRAWN** — a sprite loaded but never `drawImage`'d. Usual cause: a sprite-KEY
  mismatch (`'left_idle'` built vs `'left_fighter_idle'` generated) → silent fillRect block. The
  injected `sprite(key)` resolver (exact→normalized→token match) self-heals the drift and draws a
  loud `MISSING` marker on a true miss.
- **PROCEDURAL_REGRESSION_SUSPECTED** — ≥3 sprites declared but the canvas is mostly big fillRects.
- **ENTITY-NOT-RENDERED** — entity in `state` with x/y but not drawn.
- **STATIC-ACTION** — an action renders one held pose while the game animates elsewhere.

Per-action frames are saved to the trace (`iter_NN_action_<Key>.png`) so each action's graphics are
debuggable. **Advisory (never gates):** dead / near-identical sprite frames are *cosmetic*.

### The visual critic
Mechanism behind **`/vlm-critique`**: a local VLM answers a mechanism-keyed yes/no checklist against
screenshots (facing, attack pose, etc.). Requires a VLM backend; skips on text-only models. The
critic prefills `"Q1: "` for parseable output; abstains only on image blindness; per-question
`fix_hints` for failed checks only. See `FOR_NEXT_LLM.md` for tuning traps.

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
| `memory/playtests.jsonl` | behavior recipes for `/critique` (scripted input, state, pixel hash — no VLM); incl. genre-free `state_expr` checks (state-exposed, score, timer, pause) |
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

**`coder.py` flags:** `--backend {auto,ollama,mlx}` · `--model` · `--max-iters` · `--best-of-n` ·
`--num-ctx` · `--headless` · `--step` · `--restart-n` · `--playbook` (retrieval; off by default on
the CLI). Goal is the positional arg.

**Model topologies:** 1 model = all roles multiplex one loaded LLM (the `/allroles` default on a
single machine). On a multi-GPU box, slots 2/3 can host a dedicated architect/critic; the diffuser
pins to GPU 0 so the LLM slots stay free (`gpu_status.pick_diffuser_cuda_index`).

---

## Standalone asset tools

Two small terminal utilities outside the agent loop — same Z-Image-Turbo pipeline and asset
cache, no LLM required.

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
| Ubuntu / Linux (CUDA) | diffusers `WanPipeline` in the main `.venv` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~25 GB, lazy) | covered by `./scripts/install_diffuser.sh` |

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

See **`TEST.md`** for the three-layer testing guide. Quick reference:

```bash
python system_tests.py run --suite smoke --three-model
python system_tests.py run --suite pacman --yes
.venv/bin/python eval/eval_prompts_plan.py --coverage
.venv/bin/python eval/eval_prompts_plan.py
.venv/bin/python scripts/forget_session.py --list
./scripts/clean_artifacts.sh --yes
```
Battery: `memory/system_battery.jsonl` (local override: `games/system-tests/battery.jsonl`).

---

## Other docs

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Commands, env vars, architecture summary — **also injected into the game agent** (6 KB cap) |
| `TEST.md` | How to run and write tests |
| `FOR_NEXT_LLM.md` | Tuning rules and mistake traps for agent work |
| `HARNESS_DEBUG.md` | Gate reference and 5-grep trace debug workflow |

---

## Standing rules

1. **Tune the agent, not the model.** Never "try a bigger model" — fix prompts / retrieval / gates /
   scoring / memory.
2. **No hardcoded genre lists.** Detect by rendering/interaction *shape*, never subject matter.
3. **General fix → code; specific game craft → memory** (retrieval-gated).
4. **All code self-contained in `Agent_learning/`** — no sibling-repo `sys.path` injection.
5. **Visible Chromium by default** (TUI `headless=False`; CLI `--headless` for unattended).
6. **Asteroids is the canonical regression check** — ship direction (`vx = cos(angle)*speed`) and
   irregular-polygon asteroids must still pass after any retrieval/prompt/patch change.
7. **Never silently call a cloud model** — cloud needs the user's key + explicit opt-in.
8. **Don't tighten repetition/timeout aborts without trace evidence** — latch on code emission, not
   token count; preserve long first-build `<html_file>` completion.

Don't-commit: routine session outputs under `games/` are gitignored (`*.html`, `*_assets/`, traces,
caches). `memory/*.jsonl` is hand-curated.

---

## Troubleshooting

- **Chromium won't launch:** `env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python -m playwright install chromium`.
- **MLX cold-load is slow (~30–60 s first request):** the 27B mxfp8 loads into VRAM in-process; preload with `ollama run --ctx-size N <model>` for the Ollama path.
- **`Model type minimax_m3 not supported` after an mlx-lm upgrade:** re-copy `minimax_m3.py` from your MiniMax-M3-MLX model dir into `.venv/lib/python3.12/site-packages/mlx_lm/models/` (see **MLX upgrades — MiniMax-M3** above).
- **`Missing 285 parameters` / `indexer.k_norm.bias` loading GLM-5.2:** PyPI mlx-lm lacks IndexShare — run `./scripts/install_mlx_glm52_fix.sh` (see **MLX upgrades — GLM-5.2** above).
- **"Feedback doesn't stick" / patches fail with SEARCH-not-found:** compaction shredded the file view — check `num_ctx` (default 100K) and the `structured_compaction` trace events.
- **Game shows colored boxes instead of art:** sprite-key mismatch — the `sprite()` resolver and `ASSETS_LOADED_BUT_UNDRAWN` gate now catch it; re-run.
- **Model loops/truncates on big builds:** expected for a 27B; the MLX sampler passes `top_p`/`top_k` (vendor coding preset) to avoid the degenerate line-repeat. Judge harness signals from the trace, not always a finished game.
- **Debugging a bad session:** `HARNESS_DEBUG.md` (5-grep recipe). Tuning traps: `FOR_NEXT_LLM.md`.

---

## Dependencies
Python 3.12, `mlx-lm` / `mlx-vlm` (Apple Silicon) or `ollama`, Playwright Chromium, `diffusers` +
`torch` (Z-Image-Turbo, Stable Audio Open). Install via `./scripts/setup.sh`. See `requirements*.txt`.

## License
See repository.
