# Coding Box — Local HTML Game Agent

> Repo: **`jmrothberg/Agent_learning`** (`origin`). Single source of truth — push to `main`.
> Remote: https://github.com/jmrothberg/Agent_learning

A specialist agent that drives a **small local LLM** (qwen3.6 27B via MLX in-process, or Ollama)
to write, test, and iteratively fix **single-file HTML5 games**, with a real **Chromium** browser
as the verifier and a local **diffusion** pipeline as the asset generator. Everything runs on your
machine — no cloud calls in the main loop, no server between you and your GPU.

Thesis: **a small *validated* model beats a large unvalidated one, and an agent that learns from
every session beats a static prompt.** Every reply is parsed, every patch applied via a 4-tier
match cascade, every iteration loaded in real Chromium, every sprite alpha-keyed, every regression
flagged, every session feeds a hand-curated playbook the agent reads back next launch.

Asset pipeline (self-contained, no runtime network once weights cache): **Z-Image-Turbo** for
sprites (txt2img, 768×768 → downscaled, chroma-keyed to RGBA) and **Stable Audio Open** for sounds
(OGG, 0.2–12 s). Pose/animation frames are **txt2img with one shared character description + fixed
seed** for consistency — *not* img2img (see [Animation](#animation--consistency-is-the-hard-constraint)).

---

## Contents
- [What this is](#what-this-is) · [Quick start](#quick-start) · [Architecture](#architecture)
- [The verification harness](#the-verification-harness-the-core-lever) · [Assets & animation](#animation--consistency-is-the-hard-constraint)
- [Memory / opening library](#memory--the-opening-library) · [TUI & CLI](#tui--cli-reference)
- [System tests](#system-tests--memory-hygiene) · [Standing rules](#standing-rules) · [Troubleshooting](#troubleshooting)

---

## What this is

A coding agent narrowly specialized for **playable single-file HTML5 games with visual + audible
feedback**, running entirely locally. Two drivers:

- **`chat.py`** — Textual TUI with a visible Chromium beside the terminal (default). Mid-stream
  feedback queue, slash commands, asset picker, screenshot review, model swap, live status panel.
- **`coder.py`** — headless CLI for unattended runs. Same agent, same loop, no UI.

The loop (`GameAgent.run` in `agent.py`) is async and yields a stream of `AgentEvent`s; the drivers
only consume the stream. Three phases: **A — plan** (model emits `<plan>`/`<criteria>`/`<probes>` +
optional `<assets>`/`<sounds>`); **B — build/iterate** (patch or full-file → micro-probes →
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

**Tests** (pure-function, no model/Chromium; ~1 min, 1377 passing):
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
The critic is the human-eyes check for "facing the wrong way / punch renders backward." qwen3.6 is a
real VLM and sees screenshots, but three fixes were needed to make it useful:
- It answers in reasoning prose, never `Qn: yes/no` → parse_rate 0 → dropped. **Fix:** prefill the
  critic's assistant turn with `"Q1: "` to force the format (parser tolerates a doubled ordinal).
- Abstain detection is anchored to "can't see the *image*" — a real finding ("I don't see a
  projectile") must NOT be treated as blindness.
- Per-question `fix_hints` surface only failed checks' advice (a blanket hint once made the model
  flip a correctly-facing sprite → oscillation). The critic skips entirely on a non-VLM backend.

---

## Animation — consistency is the hard constraint

A character that kicks/punches must be the **same** character across frames. That's the entire
reason the asset pipeline exists, and it dictates two hard rules:

- **img2img cannot change a pose** (Z-Image/SD-Turbo run at `guidance_scale=0` → locked to idle at
  every strength; proven A/B in `animation_ab/`). `from_image` is recolor/restyle only.
- **A fresh txt2img *replacement* frame breaks consistency** with the character already in the game.
  So **never regenerate a pose frame to "fix" a dead/wrong one** — both routes are dead ends. A
  near-identical/dead frame is **cosmetic**: surfaced as an advisory `warning`, it must never flip
  `ok=False` or defer the user's gameplay feedback. (Trace `…214215` had this combo make both qwen
  and a SOTA model "fail" while their code was correct. See `feedback_sprite_animation_from_image.md`.)

At **plan time**, request all named pose frames as **txt2img with one shared character description +
fixed seed**. In-session, cycle the frames you have and convey the action with the **sprite**, never
a code-drawn limb. Wrong sprite direction → flip in **code** (`ctx.scale(-1,1)`), don't regenerate.

Sounds: Stable Audio Open, OGG, ≤12 s. A cross-session **asset library** under `memory/` lets
admitted sprites/sounds compound across sessions like the playbook does.

---

## Memory — the opening library

The Python engine stays general; `memory/` is a hand-curated "opening book" (chess-style) that
helps **any class of game**, retrieved at runtime — add one JSONL line, no restart. Genre-free by
shape, not subject.

| File | Holds |
|---|---|
| `memory/playbook.jsonl` | code rules-of-thumb, retrieved by weighted-Jaccard on the goal (tags weigh 2×; ~0.02 floor) |
| `memory/visual_playtests.jsonl` | mechanism-keyed yes/no VLM checklists + `auto_probes` + per-question `fix_hints` |
| `memory/implementation_outlines.jsonl` | architect mechanism outlines (retrieved k=1) |
| `memory/skeletons/` | first-build HTML templates per mechanism (`.html` + `.json` sidecar) |
| `memory/prompt_library.jsonl` | the curated `/games` prompts |
| `memory/system_battery.jsonl` | default system-test battery (committed — useful on every machine) |

**Skeleton routing** (`retrieve_skeleton`): a modality detector (board/DOM/3D strong-hooks) runs
first; then the **recipe→skeleton** map reuses the already-accurate visual-recipe matcher; then a
distinctiveness-gated Jaccard fallback; else the generic `canvas_basic_v2`. A 2D-arcade goal must
never inherit a 3D/board/dungeon scaffold (regression-tested). Coverage of the curated prompts is
checked model-free by `eval/eval_prompts_plan.py --coverage`.

---

## TUI & CLI reference

**Key TUI slash commands** (`/help` lists all): `/allroles` (architect-split + visual critic on one
loaded LLM) · `/feedback [on|off]` (autonomous self-feedback, default on) · `/wait [on|off]`
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

## System tests & memory hygiene

```bash
python system_tests.py run --suite smoke --three-model      # visible, fast plumbing + move regressions
python system_tests.py run --suite pacman --yes             # slow full Pac-Man build
.venv/bin/python eval/eval_prompts_plan.py --coverage       # model-free memory coverage matrix
.venv/bin/python eval/eval_prompts_plan.py                  # one planning turn per curated prompt
.venv/bin/python scripts/forget_session.py --list           # memory hygiene
./scripts/clean_artifacts.sh --yes                          # wipe stale per-session artifacts
```
The default battery is `memory/system_battery.jsonl` (committed); a local override at
`games/system-tests/battery.jsonl` wins if present.

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
8. **Don't tighten repetition/timeout aborts without trace evidence** — preserve long first-build
   `<html_file>` completion.

Don't-commit: routine session outputs under `games/` are gitignored (`*.html`, `*_assets/`, traces,
caches). `memory/*.jsonl` is hand-curated.

---

## Troubleshooting

- **Chromium won't launch:** `env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python -m playwright install chromium`.
- **MLX cold-load is slow (~30–60 s first request):** the 27B mxfp8 loads into VRAM in-process; preload with `ollama run --ctx-size N <model>` for the Ollama path.
- **"Feedback doesn't stick" / patches fail with SEARCH-not-found:** compaction shredded the file view — check `num_ctx` (default 100K) and the `structured_compaction` trace events.
- **Game shows colored boxes instead of art:** sprite-key mismatch — the `sprite()` resolver and `ASSETS_LOADED_BUT_UNDRAWN` gate now catch it; re-run.
- **Model loops/truncates on big builds:** expected for a 27B; the MLX sampler passes `top_p`/`top_k` (vendor coding preset) to avoid the degenerate line-repeat. Judge harness signals from the trace, not always a finished game.
- **Debugging a bad session:** see `HARNESS_DEBUG.md` (5-grep recipe) and `FOR_NEXT_LLM.md`.

---

## Dependencies
Python 3.12, `mlx-lm` / `mlx-vlm` (Apple Silicon) or `ollama`, Playwright Chromium, `diffusers` +
`torch` (Z-Image-Turbo, Stable Audio Open). Install via `./scripts/setup.sh`. See `requirements*.txt`.

## License
See repository.
