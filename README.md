# Coding Box — HTML Game Agent

A small, opinionated agent that drives a **local Ollama model** to write,
test, and iteratively fix single-file HTML5 games — with live feedback
from a real Chromium browser. Comes with a Textual TUI for two-way chat
and a plain CLI for unattended runs.

The core thesis: **a small validated model beats a large unvalidated one,
and an agent that learns from every session beats a static prompt.**
Every model output is run in a real browser, every error is fed back,
every clean turn is preserved, every regression is reverted, user feedback
typed mid-run is the highest-priority signal in the loop — **and every
session feeds an offline learner that grows the agent's `playbook` of
HTML/JS rules of thumb, so tomorrow's session starts smarter than today's.**

The patch engine, prompt assembly, retrieval, and pre-flight checks
borrow heavily from the best ideas in
[badlogic/pi-mono coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
(TS, multi-provider, native function-calling) and the
[OpenCoder](https://opencoder-llm.github.io/) project
(arXiv 2411.04905, the open recipe for a top-tier code LLM). See
[Compared to pi-mono and OpenCoder](#compared-to-pi-mono-and-opencoder)
for what we ported, where we deliberately diverge, and what each tool
is best at — they're not direct competitors, and this README does not
try to make them into one.

> **This isn't a one-shot tune.** The agent ships with a 30-bullet seeded
> playbook (canonical bugs and best practices distilled from the literature),
> and **every game you build, every piece of feedback you type, and every
> failed iter that gets fixed becomes input** to a Reflector + Curator
> pipeline that mints new playbook bullets, increments helpful/harmful
> counters on existing ones, and prunes rules that drift negative. After
> a month of use the playbook in `games/memory/playbook.jsonl` reflects
> *your* games, *your* preferences, and the failure modes *your* model
> actually hits. See [How the agent compounds](#how-the-agent-compounds)
> below.

**Default interactive stack (`chat.py`, `coder.py`):** both construct
`GameAgent` with **`prompt_version="v1"`** (the data-driven `prompts_v1.py`
module — see [agent.py:710](agent.py), [chat.py:2772](chat.py), [coder.py:189](coder.py)).
The full v1 stack is active by default: per-format guidelines, `<criteria>`
+ JSON `<probes>`, retrieved `<playbook>` block, the `<assets>` / `<sounds>`
generation pipeline (large-class), and the verifier-feedback loops described
below. The legacy `prompts.py` (v0) module was retired; pass `prompt_version`
to `GameAgent` only if you've added a `prompts_v{N}.py` of your own.

**Remote:** https://github.com/jmrothberg/Agent_learning/

## Contents

- [Prerequisites](#prerequisites)
- [How a small model writes big code](#how-a-small-model-writes-big-code)
- [Patch engine, retrieval & pre-flight upgrades](#patch-engine-retrieval--pre-flight-upgrades)
- [Compared to pi-mono and OpenCoder](#compared-to-pi-mono-and-opencoder)
- [How the agent compounds (it learns from your sessions)](#how-the-agent-compounds)
- [Tuning rig & playbook commands](#tuning-rig--playbook-commands)
- [Quick start](#quick-start)
- [CLI (`coder.py`)](#cli-coderpy)
- [What to type when (cheat sheet)](#what-to-type-when-the-only-cheat-sheet-you-need)
  - [How to write a prompt that ships a playable game](#how-to-write-a-prompt-that-ships-a-playable-game)
- [What the TUI looks like](#what-the-tui-looks-like)
- [Slash commands](#slash-commands)
- [Keys](#keys)
- [How feedback works](#how-feedback-works-this-is-the-important-bit)
  - [Changing one asset (or sound) without touching the code](#changing-one-asset-or-one-sound-without-touching-the-code)
- [Model selection](#model-selection)
- [File layout](#file-layout-where-to-look-when-something-fails)
- [How the loop works (in code)](#how-the-loop-works-in-code)
- [Restarting / resuming](#restarting-resuming-after-a-crash)
- [Troubleshooting](#troubleshooting)
- [Roadmap & known gaps](#roadmap--known-gaps)
- [Dependencies](#dependencies)
- [License](#license)

---

## Prerequisites

- **Python 3.10+**, macOS or Linux Ubuntu (the platforms `scripts/setup.sh`
  is tested on). Older Pythons miss async features the agent relies on.
- **A local LLM** — either:
  - **Ollama** (`ollama serve` with at least one pulled model), or
  - **MLX in-process** on Apple Silicon. `mlx-lm` is loaded directly into the
    agent's Python process — there is **no `mlx_lm.server`**, no HTTP, no
    `MLX_HOST`. Resolution order: `MLX_MODEL=<path-or-id>`, else a single
    auto-discovered model under `~/MLX_Models/` / `MLX_MODELS_DIR` / the HF
    cache. Weights load on the first request (~30–60 s for a 27B mxfp8)
    and stay resident in this process's GPU VRAM for the rest of the session.

  On macOS the agent **defaults to MLX** when `LLM_BACKEND` is unset; use
  `LLM_BACKEND=auto` to probe Ollama too. `OLLAMA_HOST` still applies for
  non-default Ollama addresses.
- **Playwright Chromium** — not installed by `pip` alone; `./scripts/setup.sh`
  runs `playwright install chromium`. If launch fails with “Executable doesn't exist”,
  your environment may set `PLAYWRIGHT_BROWSERS_PATH` to a stale dir — run
  `env -u PLAYWRIGHT_BROWSERS_PATH .venv/bin/python -m playwright install chromium`.
- **A display for `chat.py`** — the TUI launches a **visible** Chromium
  window beside the terminal. SSH-only / CI hosts should use
  `coder.py --headless` instead.
- **Context window** — defaults to `num_ctx=262144`, matching the native
  context of current local coding models (Qwen3.6, DeepSeek V4, GLM 5.1,
  MiniMax M2). Smaller via `CODING_BOX_NUM_CTX` / `--num-ctx` if you OOM
  on tight VRAM. Pre-warm the model at the matching size to skip a reload:
  ```bash
  ollama run --ctx-size 32768 qwen3.6:35b
  ```
- **Sprite + sound generation (GPU)** — **`./scripts/setup.sh` with no flags**
  installs the full stack by default (torch/MPS or CUDA + diffusers).
  Sprites use Z-Image-Turbo (public HF repo — usually downloads with no login).
  Sounds use Stable Audio Open — **often** the same: weights land in
  `~/.cache/huggingface/hub/` without prompting if you're already authenticated
  (`huggingface-cli login` from another tool, or `HF_TOKEN` in your environment).
  **Only if** you get **403/401** on download, see [HF troubleshooting](#hugging-face-login-only-if-downloads-fail).
  Use **`--no-gpu`** only if you **deliberately** want to skip that stack
  (~5 GB saved — no Z-Image / Stable Audio from the pipeline).
- **Trust** — the model writes HTML/JS that runs in a real browser from
  `file://` URLs. Only run models and seeds you trust; treat generated
  games like untrusted web pages if you re-host them.

### MLX memory limit on Apple Silicon

Apple Silicon Macs cap how much unified memory a single GPU process
is allowed to wire for Metal — it's not your full physical RAM. The cap
is `iogpu.wired_limit_mb`, default ≈ 75 % of total RAM. **Big models
with long context routinely cross it**, and when that happens the next
Metal allocation throws

```
RuntimeError: [metal::malloc] Resource limit (NNNNNN) exceeded.
```

inside whatever process is holding the model. Since this agent loads
MLX **in-process** (see `backend.MLXBackend`), that exception bubbles
up out of `mlx_lm.stream_generate`, gets caught by the agent's pipeline
worker, and the result is returned with `crashed=True` — the TUI shows
a recovery hint instead of a silent hang.

**Set the cap before launching the agent** (per-boot; sudo):

```bash
# Suggested formula: physical RAM (MB) − 16 GB headroom for OS + apps.
sudo sysctl iogpu.wired_limit_mb=$(( $(sysctl -n hw.memsize) / 1048576 - 16384 ))

# Examples:
sudo sysctl iogpu.wired_limit_mb=496000   # 512 GB Mac Studio
sudo sysctl iogpu.wired_limit_mb=176000   # 192 GB Mac Studio
sudo sysctl iogpu.wired_limit_mb=112000   #  128 GB
sudo sysctl iogpu.wired_limit_mb=48000    #  64 GB
```

`./scripts/setup.sh` reads `hw.memsize` and prints the right command for
your machine in its final banner; just paste-and-run.

**To persist across reboots**, drop a file into `/etc/sysctl.d/`:

```bash
echo 'iogpu.wired_limit_mb=496000' | sudo tee /etc/sysctl.d/50-mlx.conf
```

(macOS reads `/etc/sysctl.conf` at boot; some installs use
`/etc/sysctl.d/`. Check with `man sysctl.conf` on your version.)

**If MLX has already OOM'd once in the running TUI**, the Python
process's Metal context is poisoned — every subsequent generation will
fail the same way. Quit the TUI (`Ctrl+Q`), raise the cap, and relaunch:

```bash
# In the TUI: Ctrl+Q. Or from another shell:
pkill -f "python.*chat.py"
# raise the cap
sudo sysctl iogpu.wired_limit_mb=496000
# relaunch (model loads in-process on first request)
MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 .venv/bin/python chat.py
```

You can also free the in-process MLX model mid-session without quitting:
type **`/unload mlx`** in the TUI to drop the weights from VRAM (next
generation will reload on first request). Useful when swapping models
or recovering from a soft Metal error that didn't kill the whole
context.

This concern is **macOS-only**. Linux/CUDA hosts don't have an
equivalent cap — `nvidia-smi` shows the full GPU VRAM available to MLX-
or-PyTorch, no sysctl required.

### DeepSeek-V4 on MLX (mlx-lm 0.31.3 ships a broken V4 stub)

mlx-lm 0.31.3 — the version on PyPI as of May 2026 — includes a
**half-implemented `models/deepseek_v4.py`** that fails to load any
public DeepSeek-V4 conversion (Flash or Pro, any quantization) with

```
ValueError: Received N parameters not in model:
  lm_head.weight, model.embed_tokens.weight, ...
```

The file Apple shipped is missing the HyperConnection, Sinkhorn-Knopp
manifold reduction, FP8 e4m3 block dequant, sliced `wo_a` MLA output
projections, hash-routed early MoE, and `sqrtsoftplus` scoring that
V4 actually uses. The complete implementation lives in two open PRs
that haven't merged yet:

- [ml-explore/mlx-lm PR #1192](https://github.com/ml-explore/mlx-lm/pull/1192) — "Add DeepSeek-v4 (Flash/Pro)" (+2192 lines)
- [huggingface/transformers PR #45643](https://github.com/huggingface/transformers/pull/45643) — V4 tokenizer fixes (already merged on `main`; PR head still preferred per the mlx-lm PR description)

Until those land in a tagged release, install the PR heads with:

```bash
./scripts/install_mlx_v4_fix.sh
```

The script auto-detects which Python owns your `mlx_lm` install (different
on every machine — python.org installer Python 3.11 here, Homebrew
Python on someone else's Mac, conda elsewhere) by reading the shebang of
the installed `mlx_lm.server` script (still ships with the package even
though the agent no longer uses the server), then `pip install --user
--force-reinstall`s both PR heads into that same interpreter. The
in-process `from mlx_lm import load, stream_generate` calls pick up the
patches automatically — no extra wiring.

It verifies success by checking that `deepseek_v4.py` jumped from the
broken ~16 KB stub to the ~50 KB+ full implementation and contains the
`HyperConnection` / `HyperHead` / `hc_expand` symbols only the PR ships.

**When the upstream PRs merge** and a new mlx-lm release is on PyPI,
roll forward to it (overwriting the git-installed PR heads):

```bash
./scripts/install_mlx_v4_fix.sh --rollback
```

That wraps `pip install --upgrade --force-reinstall mlx-lm transformers`
into the same auto-detected Python. `--force-reinstall` is what makes it
work even when the PR-head version string compares equal to the new
PyPI release — useful because pre-merge branches don't always bump the
version. From that point on, normal `pip install -U mlx-lm` keeps you
current as new local models ship.

This patch is **scoped to the V4 model class only**. It does not touch
any other MLX architecture (Qwen, GLM, Gemma, MiniMax, Qwen3-MoE, etc.) —
those load through their own `mlx_lm/models/<name>.py` files which are
unchanged. Skip the patch if you don't plan to run V4 Flash or Pro; the
other ~300 MLX-community model conversions all work on stock 0.31.3.

**Critical runtime flag for V4** — even with the install patched, V4 has
a separate runtime bug: its Indexer attention path materializes a single
Metal buffer of size `O(L² × k)` during prefill, where `L` is the prefill
chunk length and `k` saturates at 512. At the default
`prefill_step_size=2048` this single allocation crosses the per-process
Metal cap (~487 GB on a 512 GB Mac) the moment any prompt is more than
~1.3K tokens, and `stream_generate` dies with
`[metal::malloc] Resource limit (NNN) exceeded`.

The agent passes a safer **`prefill_step_size=1024`** by default to
`mlx_lm.stream_generate` ([backend.py](backend.py), constant
`_MLX_PREFILL_STEP_SIZE_DEFAULT`). Override per-machine with the
`MLX_PREFILL_STEP_SIZE` env var if you want to experiment:

```bash
MLX_PREFILL_STEP_SIZE=512 .venv/bin/python chat.py   # paranoid
MLX_PREFILL_STEP_SIZE=1280 .venv/bin/python chat.py  # last safe step
```

`1024` is the documented safe-anywhere default per the PR reviewer's
`(L, k, GB)` table — the resulting 34 GB single buffer fits under
Metal's ~86 GB per-buffer cap on every M-series Mac. The Metal cap is
hardware-bound and **does NOT scale with total RAM**, so even on a
512 GB Mac you can't push above ~1280 chunk size; chunk=1500 already
overflows the per-buffer cap and crashes. There's no benefit to going
higher. Going lower (`512`, `256`) is paranoid-safe and ~5–10% slower
on prompts > 1K tokens; lower than that has no effect on shorter
prompts.

Tracking the upstream cubic-attention bug: see PR #1192's review thread
for the `(L, k, GB)` table that documents the blow-up. Once a fix lands
upstream the env var becomes optional and
`./scripts/install_mlx_v4_fix.sh --rollback` will pick up the upstream
fix.

---

## How a small model writes big code

```
                                      USER GOAL
                                          │
                                          ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  PHASE A · PLAN                                              │
        │  ─────────────────                                           │
        │  Model emits <plan>: mechanics · controls · win/lose · risk  │
        │  Forces explicit design BEFORE the first character of code.  │
        └────────────────────────────────┬─────────────────────────────┘
                                         │  goal
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  MEMORY ▸ GameMemory.retrieve_skeleton │
                    │  Nearest past WORKING game seeds     │
                    │  the very first <html_file>.         │
                    └──────────────────┬───────────────────┘
                                       │  seed file
                                       ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │            PHASE B · BUILD ⇄ TEST     (loop until <done/>)           │
   │            ────────────────────────                                  │
   │                                                                      │
   │      MODEL                                  HARNESS                  │
   │   ┌─────────────┐   <patch> blocks      ┌─────────────────────────┐  │
   │   │ <diagnose>  │                       │  REAL CHROMIUM          │  │
   │   │ <patch>×N   │ ────────────────────► │  • console errors       │  │
   │   │ <html_file> │                       │  • page errors          │  │
   │   │ <notes>     │                       │  • RAF actually firing? │  │
   │   │             │      REPORT           │  • canvas blank/frozen? │  │
   │   │             │ ◄──────────────────── │  • input listener count │  │
   │   └──────┬──────┘                       │  • smoke-press all keys │  │
   │          │                              │  • screenshot.png       │  │
   │          │  fix_mode = True             └─────────┬───────────────┘  │
   │          │  T = 0.25  (precision)                 │                  │
   │          │                                        │ failure          │
   │          │   ┌────────────────────────────────────▼─────────────┐    │
   │          │   │  MEMORY ▸ GameMemory.retrieve_mistakes(sig, k=3) │    │
   │          │   │  past root-causes for THIS failure type, inlined │    │
   │          │   │  into the next prompt as hints                   │    │
   │          │   └──────────────────────────────────────────────────┘    │
   │          ▼                                                           │
   │   ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐   │
   │   │ BEST-OF-N    │    │ VLM SCREENSHOT  │    │ USER FEEDBACK     │   │
   │   │ ─────────    │    │ ─────────────── │    │ ─────────────     │   │
   │   │ sample N     │    │ vision model    │    │ free text typed   │   │
   │   │ completions  │    │ gets the .png + │    │ at any time, in   │   │
   │   │ without TUI  │    │ "compare to     │    │ a HIGHEST-PRIORITY│   │
   │   │ streaming;   │    │  expectation"   │    │ banner, OVERRIDES │   │
   │   │ score each   │    │ catches "runs   │    │ plan + defaults   │   │
   │   │ in the same  │    │ but looks wrong"│    │ this turn         │   │
   │   │ LiveBrowser  │    │                 │    │                   │   │
   │   └──────────────┘    └─────────────────┘    └───────────────────┘   │
   └────────────────────────────────┬─────────────────────────────────────┘
                                    │  <done/>
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  PHASE C · SELF-CRITIQUE                                     │
        │  ─────────────────────                                       │
        │  One extra turn — model sends one more <patch> or            │
        │  <confirm_done/>. Catches "confidently wrong" final replies. │
        └────────────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────┐
            │  best.html SAVED  ·  memory UPDATED  ·  ready      │
            │                                                    │
            │  type more feedback ▸ AUTO-EXTEND the same file    │
            │  /new <goal>        ▸ start a fresh session        │
            └────────────────────────────────────────────────────┘
```

**Diagram note — best-of-N:** each candidate is scored by loading temporary
HTML through the **same** `LiveBrowser` instance as the rest of the session
(`tools.LiveBrowser.load_and_test`). With `chat.py` that is usually a **visible**
window; with `coder.py --headless` it stays headless. There is no separate
always-headless scorer.

### Why this competes with much larger models

A 20B local model has limited reasoning per turn. The harness multiplies
that capacity by **outsourcing verification to the runtime**: the JS
engine validates syntax, the DOM proves event listeners attach, the
canvas pixels prove rendering happens, the keyboard simulator proves
controls respond. Frontier one-shot models cheat by having more
parameters; this loop cheats by **never trusting the model's self-eval**
— if the harness can't see it work, it didn't work.

Each technique is matched to a *specific* small-model failure mode:

| Technique                    | Failure it prevents                                            | Where it lives                              |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| **Plan first**               | Code-first models skip controls / win conditions / edge cases  | `prompts.PLAN_INSTRUCTION`                  |
| **Web research grounding**   | Model "knows" the game name but not the mechanics — ships Space Invaders when asked for Missile Command | `research.fetch` → `<reference>` injected into Phase A; v1 prompt marks it AUTHORITATIVE |
| **Repetition watchdog**      | Model gets stuck looping the same 1–2 short lines while tokens keep flowing (stall watchdog wouldn't fire) | `ollama_io.stream_chat` sliding-window detector → `StreamResult.looped` |
| **Memory · skeleton**        | First-build cold start ("how do I even structure this?")       | `memory.GameMemory.retrieve_skeleton`       |
| **Patches > rewrites**       | Token budget blown re-emitting unchanged code; truncation bugs | `patches.extract_patches`, `apply_patches`  |
| **Real-browser tests**       | Hallucinated APIs, wrong event names, broken bracket indexing  | `tools.LiveBrowser.load_and_test`           |
| **Memory · past mistakes**   | Re-making the same fix five sessions in a row                  | `memory.GameMemory.retrieve_mistakes`, `record_mistake` |
| **Diagnose-then-fix**        | Patches applied without root-cause analysis; whack-a-mole      | `prompts.fix_instruction`                   |
| **Best-of-N (fix mode)**     | One sample landing on a bad local minimum                      | `agent._generate_and_score_candidates`      |
| **VLM screenshot review**    | "Game runs but looks wrong" — bugs the harness can't detect    | `agent._stream` → `image_attached`          |
| **User feedback override**   | Model running in circles, ignoring obvious user intent         | `agent._flush_user_injections`              |
| **Self-critique pass**       | "Confidently wrong" final reply ships unchallenged             | `prompts.CRITIQUE_INSTRUCTION`              |
| **Save best on every clean** | A regression destroying yesterday's working code               | `agent._save_best`                          |
| **Continuation extends**     | "Add sound" after `<done/>` means restart from scratch         | `agent.run(continuation=True)`              |
| **Adaptive temperature**     | Same temp for creative-build and bug-fix turns                 | `agent._stream` (`0.7` build → `0.25` fix)  |
| **Conversation pruning**     | Long sessions overflow context; old code crowds out new turns  | `agent._prune_messages`                     |
| **Playbook injection**       | Same dumb mistake every session — model has no long-term memory | `memory.Playbook.retrieve` → `<playbook>` in **v1+** prompts only (`prompt_version != "v0"`) |
| **Acceptance criteria**      | "Passes the smoke test" but doesn't actually fulfill the goal  | `prompts_v1.PLAN_INSTRUCTION` `<criteria>`  |
| **Runtime probes**           | Game looks fine to the heuristics but the player can't actually play | model emits `<probes>` JSON → executed by `tools.LiveBrowser._run_probe` |
| **Probes gate `ok`**         | Probe results were advisory; harness reported `ok=True` even when the model's own probes failed (e.g. `monsters: []` ships as "passed") | `tools.py:1057` post-step appends `PROBE FAILED` to `soft_warnings`; existing `ok = no errors AND no soft_warnings` formula now catches it |
| **`<done/>` clean-streak**   | Model declared done on the first ok=True iter and shipped a flaky pass | `agent._consecutive_clean_iters` ≥ 2 required before honoring `<done/>` (default; `_min_clean_streak_to_ship`) |
| **3D intent detector**       | Model hand-rolls a raycaster in 12 KB when a CDN three.js + sprite billboards would ship a real 3D game in the same effort | `prompts_v1._detect_3d_intent` triggers on `3d / fps / first-person / raycaster / voxel / minecraftlike / wolfenstein / perspective`; injects three.js / babylon CDN nudge into Phase A |
| **Mixed sprite/procedural**  | Model forces every entity into one bucket; misses that destructible-state needs procedural drawing while static character needs sprites | `prompts_v1.ASSETS_FORMAT.guidelines` explicitly teaches: sprites for static (player/enemies/walls), procedural for destructible (bunkers crumbling brick-by-brick, cracks, damage levels) |
| **Sprite-orientation pattern** | `drawImage` ships a sideways gun because the sprite was rendered facing right but drawn flat | `assets.render_asset_paths_block` includes the `save / translate / rotate / drawImage / restore` snippet inline in the asset paths block the model receives |
| **Honest probe signal**      | Probe failures triggered by harness-side canvas-tainting (file:// + drawImage made `getImageData` throw `SecurityError`); a working render was reported as failed and shipped a regression | Chromium launches with `--allow-file-access-from-files` + `--disable-web-security`; probe runner detects `tainted / cross-origin / SecurityError` and downgrades those probes to `ok=True` with a `downgraded` reason; `non_blank` example replaced with a `toDataURL`+try/catch+dimension fallback |
| **Chroma-key transparent backgrounds** | Z-Image-Turbo doesn't honor `transparent background` prompts; the model tried to chroma-key in JS at runtime, tainting the canvas and breaking RAF | `assets._chroma_key_to_rgba`: PIL pass after generation samples 8 corner+edge points, requires ≥6/8 agreement to pick a dominant bg color, alpha-masks within tolerance, saves RGBA PNG. Stats logged per-asset (`bg_color`, `alpha_pixel_ratio`) so you can see at a glance whether the chroma-key actually fired |
| **Per-iter HTML hash**       | Test events were hard to correlate with the exact code that produced them; "did iter 3 actually change anything?" required manual diffing of snapshots | `code_snapshot` trace event includes 16-hex `html_sha256` so iter→code linkage is direct |
| **Error source split**       | One catch-all `errors[]` mixed `console.error('debug')` (informational) with `UNCAUGHT TypeError` (real crash) — model treated them with equal urgency | Report fields split: `console_errors` / `page_errors` / `probe_errors`. `format_report_for_model` renders **PAGE ERRORS (must fix)** separately from CONSOLE ERRORS |
| **Stuck-loop reflection**    | Same wrong fix tried 3 times in a row                          | v1 `fix_instruction` switches to "5–7 different sources" mode after `stuck_streak >= 2` |
| **Fuzzy patch matching**     | Smart-quote / em-dash / NBSP drift between model output and file → `<patch>` "SEARCH not found" | `patches._normalize_chars` (1:1 char-preserving NFKC-lite) |
| **Patch uniqueness + non-overlap** | Ambiguous SEARCH silently picks wrong site; overlapping patches splice garbage | `patches._locate` + reverse-order apply in `apply_patches` |
| **Patch repair layer**       | BOM / CRLF / stray ```html fences inside SEARCH-REPLACE bodies fail exact match | `patches.repair_reply` + `_strip_internal_fences` |
| **Per-format guidelines**    | Big monolithic system prompt fragments rules across sections; small-model attention frays | `prompts_v1.FormatSpec` + `build_system_prompt` (deduped) |
| **Structured compaction**    | Long extension sessions drift; bug-fix context lost first when history is naively elided | `agent._build_structured_summary` + 2-tier `_prune_messages` |
| **Pre-Chromium micro-probes**| Truncated stream / unbalanced braces / elision markers waste a 3 s browser round-trip | `tools.run_micro_probes` |
| **API allowlist (hallucination guard)** | Models invent methods (`ctx.drawCircle`, `audioCtx.playSound`); Chromium TypeError eats a round-trip | `tools._check_api_allowlist` (canvas2d / AudioContext / canvas-elt receivers) |
| **Project-config injection** | Per-repo conventions ("always Phaser, never React") have to be re-stated every session | `agent._read_project_config` reads `AGENTS.md` / `CLAUDE.md` from cwd |
| **Bullet-on-demand retrieval** | Eager top-K injection burns context on bullets the model may not need | `<lookup_bullet>id</lookup_bullet>` + `memory.render_playbook_block(mode="hybrid")` |
| **Generated sprites (Z-Image-Turbo)** | Procedural canvas drawing limits arcade-style polish; agent can request real PNG sprites that the harness mints in-process — no separate server | `<assets>[{name,prompt,size?}]</assets>` in Phase A → `assets.generate_assets` → paths injected into first-build prompt |
| **Two-stage retrieval**      | Same playbook bullets injected at plan time AND code time → context bloat without precision | `agent._retrieve_playbook_block(stage="plan"/"code")` |
| **Quality-ranked retrieval** | Identical-relevance bullets returned in arbitrary order; loser equally likely as winner | `Playbook.retrieve` quality multiplier `1 + 0.10·tanh(score/5)` |
| **Shingle dedup**            | Two near-identical bullets crowd out the third diverse one     | `memory.dedup_hits` (5-gram word shingles, Jaccard ≥ 0.85) |
| **80/16 context budget**     | Playbook block grows unboundedly as the corpus grows           | `memory.cap_hits_by_budget` + char-budget arg on `render_playbook_block` |
| **Offline learner**          | Lessons learned in session N never reach session N+1           | `learner.py` (Reflector + Curator) → `playbook.jsonl`  |
| **Tune battery**             | Prompt / probe / playbook changes broke something silently     | `tune.py run` battery + `tune.py diff a b` per-test deltas |

---

## Patch engine, retrieval & pre-flight upgrades

Two adjacent code-agents in the wild taught the loop new tricks:

- **[badlogic/pi-mono `coding-agent`](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)** — a focused TS coding agent
  whose `edit-diff` engine, system-prompt assembly, and compaction patterns
  are unusually well-engineered for small/mid models.
- **[OpenCoder](https://opencoder-llm.github.io/)** (Huang et al., 2024,
  arXiv 2411.04905) — the open recipe for a top-tier code LLM. Its data-
  curation findings translate cleanly into *inference-time context-curation*
  rules, which is exactly what the playbook retrieval needed.

Ten features ported, grouped by where they live:

### `patches.py` — patch engine, pi-mono pattern

The SEARCH/REPLACE format is unchanged, but the matcher is now a cascade:

```
exact  →  char-preserving normalized  →  whitespace-collapse  →  trimmed
```

Char-preserving normalization (`_normalize_chars`) maps every smart quote
(`’`/`‘`/`“`/`”`/`″`/`′`) to ASCII, every dash variant (`–` `—` `―` `−`)
to `-`, and every unicode space (NBSP ` `, en/em/figure spaces, ideographic
`　`) to a regular ASCII space — each transform is 1:1, so positions in
normalized space map directly to original-text indices, no offset map
needed. This rescues the most common qwen3.6 / gpt-oss failure mode:
the model "polishes" its patch text, the file on disk has the original
ASCII, and `<patch>` reports SEARCH-not-found.

Cross-patch validation now mirrors pi-mono's edit-diff:

- **Uniqueness check.** If SEARCH matches more than one place in the
  source, the patch is rejected with a prescriptive error ("add more
  surrounding context"). Previous behavior silently picked the first
  match — sometimes the wrong one.
- **Non-overlap.** With multiple `<patch>` blocks per reply, the engine
  finds each match in the *original* source, sorts by start index, and
  rejects pairs whose spans overlap (with a "merge edits N and M into
  one" instruction). Then applies in **reverse source-order** so earlier
  splices keep later offsets valid.

A repair pass (`repair_reply`) runs before the regex parser:

- Strips a UTF-8 BOM at the very start.
- Normalizes CRLF / bare-CR to LF.
- After parsing, each `<patch>` body has stray markdown fences removed
  (`_strip_internal_fences`) — models occasionally wrap the body in
  ```` ```html ```` for "safety", which then fails the literal match.

Coverage: 24 unit tests in `tests/test_patches.py`.

### `prompts_v1.py` — per-format guidelines, pi-mono pattern

`SYSTEM_PROMPT` is now built from data, not written by hand:

```python
FormatSpec(
    name="<patch>",
    snippet="<patch>...</patch>  SEARCH/REPLACE block",
    guidelines=[
        "SEARCH must appear in the current file character-for-character …",
        "If SEARCH would match more than one place, the patch is rejected …",
        "Do not emit overlapping or nested patches …",
        # …
    ],
)
```

Each output tag (`<patch>`, `<html_file>`, `<question>`, `<diagnose>`,
`<criteria>`, `<probes>`, `<notes>`, `<done/>`, `<confirm_done/>`) owns
its own guidelines array. `build_system_prompt(goal, formats=...)` walks
the enabled formats, dedupes guidelines (so the same rule never appears
twice), and renders the `<output-tags>` list, the `<guidelines>` block,
and the cross-cutting `<hard-rules>` / `<anti-patterns>` blocks. The
result is the same effective prompt as before, but smaller and
maintainable from data.

### `agent.py` — structured compaction, pi-mono pattern

`_prune_messages` is now two-tier:

- **≤ 14 messages:** existing per-turn HTML elision (replace `<html_file>`
  bodies with `[omitted: N bytes]`). Cheap, lossy on iteration history,
  safe.
- **> 14 messages:** replace messages 1..cutoff with one **state-anchor
  message** built deterministically from agent state by
  `_build_structured_summary` — a fixed Markdown skeleton:

  ```
  ## Goal
  ## Acceptance criteria
  ## Executable probes
  ## Progress
  ## Key decisions
  ## Last test report
  ## Files in session
  ## Critical context
  ```

  No extra LLM call (we already track every field). The system prompt
  and the last 4 turns survive intact.

This kicks in on long extension sessions where the elision path would
have lost too much per-turn diagnostic context.

### `memory.py` — quality-ranked, deduped, budgeted retrieval (OpenCoder)

The OpenCoder paper's data-curation findings translate to inference-time
context-curation almost line-for-line:

- **Quality multiplier.** `Playbook.retrieve()` now scores each hit as
  `relevance × (1.0 + 0.10·tanh(score/5))`, a bounded ±10% boost based on
  the bullet's helpful-minus-harmful counter. Equal-relevance ties go to
  the validated winner, but a heavy-winner on an off-topic bullet can't
  outrank an on-topic newcomer.
- **Two-stage retrieval.** A new `stage="plan"|"code"` argument selects
  the OpenCoder-style broad-then-narrow split: plan-stage returns top_k+2
  bullets (lenient, even slightly-harmful entries — the model benefits
  from "see the whole space"), code-stage returns top-3 with score ≤ -2
  bullets dropped (validated patterns only). `agent._retrieve_playbook_block`
  passes `stage="plan"` for first-build calls and `stage="code"` for
  fix-turn calls.
- **Shingle dedup** (`dedup_hits`). 5-gram word shingles + Jaccard
  ≥ 0.85; near-duplicate bullets collapse to the highest-ranked one.
- **80/16 budget cap** (`cap_hits_by_budget`). The rendered `<playbook>`
  block is truncated to a char budget — 4500 chars for plan stage, 2400
  for code stage. Mirrors OpenCoder's annealing-mix finding that ~16% of
  context being "high-signal" is the sweet spot; more dilutes.

`render_playbook_block` runs dedup → cap → render by default, so callers
get the new behavior automatically.

### `tools.py` — pre-Chromium micro-probes (OpenCoder)

OpenCoder's Educational-Instruct lesson: cheap execution filters, often.
The harness already has a 3 s Chromium round-trip per iteration; we now
run a fast pre-flight first.

`run_micro_probes(html)` checks:

- **Size** (errors if < 200 bytes — the patch left the file empty).
- **Structural completeness:** `<!DOCTYPE>` / `<html>` / `</html>` /
  `<body>` / `</body>` all balanced. An unclosed root tag is a stream-
  truncation indicator.
- **Script presence:** at least one `<script>` block (or inline event
  handler for DOM-only games — open-domain, no genre lock-in).
- **Bracket balance** for each inline `<script>` body, after stripping
  comments and string/template literals. Off-by-2-or-more = error;
  off-by-1 = warning (regex-literal false positives are real).
- **Elision sentinels:** `// ... rest unchanged ...`, `// (existing code)`,
  etc. — the model occasionally slips these in even after we tell it not
  to.

On error, the agent skips the Chromium round-trip and feeds a structured
report back to the model on the next turn — same shape as a real test
report, with the title `(skipped browser — pre-flight failed)` so the
trace is unambiguous.

### Verifier-feedback loops — failures the harness used to swallow

Two consecutive Donkey Kong runs (`games/traces/donkey-kong-arcade-clone-800x6_20260513_185815` and `games/traces/donkey-kong-game-animated-donk_20260514_104131`) exposed a category of failure the harness was eating quietly: the model produces something that *looks* finished, the harness rubber-stamps it, and the user gets a broken game with the agent insisting it shipped. The research evidence on non-SOTA local coding agents (Aider leaderboards, Agentless, Kimi-Dev, TACL 2024 / arXiv 2411.17501) all points the same direction: **external-verifier strength dominates self-critique for sub-70B models.** The upgrades below tighten that verifier edge — no free-form reflection critics, no "let the model judge itself," every fix lands in the harness→model signal path.

**Parser self-correction (`patches.py`)**

The 20260513 trace burned 7 consecutive turns when the model emitted a *valid* `<patch>` wrapped in a ```` ```html ```` markdown fence — the harness rejected it with a generic *"I could not find a `<patch>` or `<html_file>` block"* and the model retried the same shape every time. Two fixes:

1. **Proactive fence stripping** in `repair_reply`: when a ` ``` ` fence body contains `<patch>` or `<html_file>`, the wrapper is removed before extraction. Unrelated fences (e.g. inside `<plan>` prose) are left intact — `_strip_outer_fences_around_tags` only fires when a real tag is inside.
2. **Structured rejection classification** via `classify_format_failure(reply) → FormatRejection | None`. When extraction still fails, the classifier names *why*: `tags_in_fence`, `bare_markers` (SEARCH/REPLACE with no `<patch>` wrapper), `unclosed_patch`, `unclosed_html_file`, `wrong_tag_html` (model used `<html>...</html>` directly), `wrong_tag_patches` (plural). Each kind ships with a specific model-facing hint instead of the generic fallback.

**Format-doctor subagent (`agent.py`)**

On the second consecutive parse failure (`_format_stuck_streak == 2`), `_run_format_doctor` fires a single one-shot inference on the **same backend / same loaded model** with a fresh isolated message history — a narrow system prompt + the failed reply + the structured rejection, nothing else. If the doctor's output parses, it replaces the failed assistant message in the conversation and the iteration continues; if not, the next user turn carries a "stop trying `<patch>`, send a complete `<html_file>`" escalation. This is the only deepagents-style subagent in the loop, kept because its input is the external rejection signal (grounded), not free self-reflection (the failure mode for weak models).

**Probe-quality classifier + first-build nudge (`agent.py`)**

The 20260513 probes (`state_exists`, `player_exists`, `barrels_array`, `princess_exists`, `hud_visible`) were all structural-presence checks; a game that rendered a static HUD passed every one. The 20260514 probes were the same pattern in slightly different clothes (`barrels.length > 0`, `textContent.length > 0`, `toDataURL().length > 200`). `_classify_probes_dynamic(probes)` walks each probe's expression and tags it dynamic only when it contains an `await` / `setTimeout` / `requestAnimationFrame` / `performance.now`, a `getImageData` pixel read, or a numeric comparison against a non-zero threshold whose LHS isn't a `.length` / `.size` / `.width` / `.height` style structural property. When zero probes classify as dynamic, the first-build user message gets a directive showing three example dynamic shapes (timer-delta, state-delta, canvas-pixel-non-black).

**Input-responsiveness probe synthesis (`tools.py`)**

The 20260514 trace shipped with `Input test: FAIL — pressed [...] and canvas pixels never changed` and `ok=True` on the same report, because the harness saw a clickable element and downgraded the failure with `"Note: keyboard test produced no canvas change, but the page has clickable elements; treating as DOM-driven."` The model used that note as permission to ship. Fix:

- `expects_game_controls(*texts)` — a tokenized keyword detector on the `<criteria>` text. Words like `arrow`, `wasd`, `key`, `press`, `move`, `climb`, `jump`, `fire` fire it; substrings inside other words don't (so "monkey" doesn't match "key"). Genre-free / game-agnostic — it describes input *modality*, not subject matter.
- When the input test fails AND `expects_game_controls(criteria)`, the harness overrides the DOM-driven carve-out and synthesizes an `input_responsive` probe into `report["probes"]` with `ok=False`. The existing probe-failure gate then promotes it to a `PROBE FAILED [input_responsive]: ...` soft-warning, which the `ok` formula already catches. `<done/>` becomes unavailable until the input is actually wired.

**Final-iteration test guarantee (`agent.py`)**

The 20260513 trace's worst single failure: the final assistant turn shipped a correct full `<html_file>` that was never tested because the loop hit its `max_iters` budget while in the rejection branch. `_final_iter_test_if_needed()` runs at every exit path — user-force-done, `<done/>`+ok, and max-iters — and if the last materialized iter index is greater than the last tested iter index, one closing browser test runs against the current file. On pass it promotes the file to `best.html` if no baseline exists; on fail nothing is promoted. `_record_session_outcome` reads from this final test, so the session never ships a "passed" report that's actually based on a stale snapshot.

**Subsystem-pointing coaching + iter-1 focused-slice biasing (`agent.py`)**

Across all four recent DK-class traces, the `mistake_signature` literally encoded which subsystem was broken — `INPUT_DEAD` from `memory.signature_for_report` when the input test pressed every key and the canvas never changed, or `FROZEN` when the canvas drew once but never re-rendered. The model could *read* the failure (iter 3's reply preview started *"1. Input test FAIL: Pressing arrow keys doesn't change canvas pixels"*) but its patches kept editing the higher-level mechanic the user's complaint named (climb math, barrel physics) instead of the actual broken layer (`addEventListener` wiring). Existing coaching at `_repeat_sig_streak >= 2` said *"AUTHOR a runtime-state probe"* — too abstract for a 27B model already focused on the wrong code area.

The fix has two halves, both routed through a new `_subsystem_hint(signature)` helper that maps signature substrings to `(name, identifier_tokens, fix_phrase)` tuples. Three entries today, each describing a SHAPE of failure (input wiring / draw-or-RAF / RAF-start) not a genre:

| Signature shape | Identifier tokens | Fix phrase |
|---|---|---|
| `INPUT_DEAD` / *"Controls are not wired up"* | `addEventListener`, `keydown`, `keyup`, `KeyboardEvent`, `code`, `KEYMAP`, `keys` | "the keydown/keyup handler" |
| `FROZEN` / *"did not change between two samples"* | `requestAnimationFrame`, `frame`, `render`, `draw`, `ctx` | "the frame/draw loop" |
| *"canvas pixels are uniform"* / *"not rendering"* | `requestAnimationFrame`, `loadAssets`, `then`, `startGame`, `init` | "the RAF kick-off" |

1. **Coaching becomes directive at `_repeat_sig_streak >= 2`**: when a hint matches the current signature, the queued coaching message names the subsystem ("Same INPUT failure for 2 iterations..."), lists the implicated identifiers, and tells the model to target `<patch>` SEARCH at the keydown/keyup handler OR send a focused `<html_file>` rewriting only that subsystem. Generic fallback preserved when no hint matches.

2. **`_focused_slice` keyset biasing on iter 1**: when `self._last_mistake_sig` matches a hint, the slice constructor pulls the hint's identifier tokens into the keyset *in addition to* error-signal and criteria tokens. The slice the model sees in its iter-1 fix prompt now surfaces functions matching `addEventListener` / `keydown` / etc., not just the higher-level functions whose names appear in the failing report. This acts one iteration earlier than the streak-based coaching — the verifier signal is already there on iter 1 when input fails.

Coverage: 9 unit tests in [tests/test_subsystem_hint.py](tests/test_subsystem_hint.py) including a pinned DK-trace-shaped HTML fixture confirming that without the hint the slice misses the input handler entirely, with the hint it surfaces.

**Patch-delta token-repetition rejection (`patches.py`)**

DK trace 20260514 iter 3 documented a different small-model degenerate sampling failure: the model emitted a `<patch>` whose REPLACE body added the line `}else{` 11 times consecutively. The existing `run_micro_probes` detector flagged the pattern AFTER the patch had applied — the token-spam had already shipped to disk and the next test ran on a poisoned file.

`_detect_replace_repetition(replace_body)` in `patches.py` runs *at apply time* inside `apply_patches`'s per-patch loop, right after the existing `_has_embedded_marker` check. Returns `(line, count)` when ≥3 consecutive identical lines appear in the REPLACE body, with defenses against false positives:

- **Trimmed length ≥ 6 chars** — skips lone `}`, `});`, `;` chains that are legitimate code structure.
- **Pure-punctuation lines excluded** — `re.match(r"^[\W_]+$", line)` returns True → skip.
- **Lines inside unbalanced backticks skipped** — template literals legally contain multi-line repeated content (banner strings, ASCII art, etc.).
- **Requires consecutive identical lines** — switch statements with `case CMD_X:` / `case CMD_Y:` chains stay safe because the lines aren't identical even though the structure repeats.

On trigger, the patch is rejected with a specific error naming the line and count (*"patch REPLACE block contains the same line repeated 11 times consecutively: `}else{` — token-repetition loop, not a legitimate edit"*). The existing `patch_retry_instruction` plumbing in `agent.py` delivers the rejection to the next model turn. Coverage: 15 unit tests in [tests/test_patch_replace_repetition.py](tests/test_patch_replace_repetition.py) including the literal DK `}else{` × 11 case and negatives for switch chains, lone braces, template literals, and short bodies.

**Probe re-parse gate widening (`agent.py` + `prompts_v1.py`)**

Phase A asks the model to author `<probes>` before it sees the seed file (on `/seed` sessions, the file isn't shown until Phase B). The 20260514 trace captures the failure mode: the model wrote `state.grid`, `state.player.onLadder`, `state.reset`, `#instructions` — none of which exist in the seed. Iter 1's report had 4/5 probes evaluating falsy forever. The model then re-emitted 5 corrected probes (3 dynamic) in its iter-1 reply that DID match the file shape — but the original coverage-gap-only gate dropped them, and iter 2's test re-ran the stale Phase-A probes.

The gate now opens when ANY of:
1. Phase-A coverage gaps exist (the original trigger — preserved).
2. The prior iter's report had at least one probe failure (new — failing probes are evidence they may be wrong).
3. The session was started with `/seed` AND this is iter 1 (new — the model just saw the file for the first time).

Defensive: the new probe list must be at least as large as the current set, so a model can't shrink the probe surface to mask regressions. `seed_build_instruction` in `prompts_v1.py` now also explicitly tells the model *"your Phase A `<probes>` were written WITHOUT seeing this file. If any reference state property, function, or DOM element names that do NOT exist in the code below, RE-EMIT a corrected `<probes>` block alongside your patch."*

**Small-LLM safety net — six fixes from the 20260514_214747 trace**

A Donkey Kong session run with a 27B local model (Qwen3.6-27B-mxfp8) exposed two clean harness gaps and three UI clarity gaps. The model hit two full deliberation loops (turns 04 and 08, each ~1000 lines of `<think>` prose with no code emission), one of which destroyed the on-disk baseline by emitting a 374-byte pseudocode skeleton inside `<html_file>` tags. The `Input test: FAIL` diagnostic was misleading because it claimed "controls are not wired up" even though `Input listeners: total=5` was logged in the same report. And the user couldn't tell at a glance whether the active model could read screenshots (VLM) or whether `/wait` step-mode was on or off.

1. **Skeleton-payload rejection** ([agent.py](agent.py) `_detect_skeleton_payload`). The trace's iter 3 emitted an `<html_file>` whose JS body was 374 bytes of `// Asset loading` / `// Sound loading` comment-headers with `function loadAssets() { ... }` placeholder bodies. The harness wrote that to disk as the baseline; iter 4 had no real code to patch against and the model spent another deliberation loop trying to rebuild. The new detector trips when ALL FOUR conditions hold: (a) total HTML < 4 KB, (b) `<script>` body shorter than 800 bytes after stripping comments and whitespace, (c) at most 2 function definitions, (d) a placeholder marker (`{ ... }`, `// ...`, `// TODO`, `// stub`). Rejection fires inside `_materialize` before disk write, preserving the prior real baseline. Specific error tells the model to "emit the COMPLETE implementation this turn — every function body fully written, no `{ ... }` placeholders" or fall back to `<question>`.

2. **Tighter deliberation guard** ([ollama_io.py](ollama_io.py) `DeliberationDetector`). Defaults were 6000 chars outside `<think>` and 15000 inside. The trace's two deliberation loops were caught but only after ~200 lines of pure reasoning — minutes of wall-clock per stuck iter. Tightened to 4000 / 8000. False-positive risk on legitimate complex reasoning stays low because the tag-opener regex matches any output-tag start (including ```` ```html ```` / ```` ```js ```` fences in seed builds), so a model that's about to emit code latches well before the threshold.

3. **Better `Input test: FAIL` diagnostic** ([tools.py](tools.py)). Previous message said *"Controls are not wired up (or input handler is broken)"* regardless of listener count. The trace's iter 2 had `Input listeners: total=5 (win=5)` AND `RAF ran: True` AND canvas was not blank — the wiring DID exist; the model's keydown handler was firing, the issue was downstream (closure-scope mismatch between the handler's `keys` object and the one `updatePlayer` / `draw()` read). New diagnostic branches on `(listeners_present AND raf_ran AND canvas_not_blank)` and points at the actual layers to check: `keys`-object scope, `e.code` vs `e.key`, RAF-before-listener ordering, `update()` early-return on game-state flags. The `input_responsive` synthetic probe carries the same matched diagnostic.

4. **VLM / text-only badge in `/list`** ([backend.py](backend.py) `classify_model_modality` + [chat.py](chat.py)). Vision-Language Models can consume screenshots — the agent's `_detect_vlm` probe and the VLM-critique path attach the current canvas to feedback turns so the model can SEE what it just shipped. For a text-only model that path is a no-op. The new classifier substring-matches the model name against a catalog of known VLM families (Qwen-VL, LLaVA, DeepSeek-VL, MiniCPM-V, Pixtral, Gemma 3, Phi-vision, MoonDream, plus all current Claude / GPT-4o variants). `/list` now shows `[VLM]` (magenta) or `[text]` (dim) next to each row. Name-based only; the runtime probe is still authoritative for the active session, but the badge lets the user pick the right model up-front. Add new VLM families by extending `_VLM_NAME_SUBSTRINGS`.

5. **Mode row in status panel** ([chat.py](chat.py) `_render_mode_row`). The previous mode-bar showed a yellow `WAIT MODE` badge when step-mode was ON and nothing when OFF — the user had to remember the toggle state. The new status panel renders an explicit colored Mode line at the top of the right-hand panel that shows BOTH states: black-on-yellow `WAIT` when step-mode is on (with descriptive text "pause after each iter — Enter or feedback to continue"), black-on-green `AUTO` when off ("continuous run — /wait on to pause per-iter"). The bottom mode-bar got the same dual-state treatment (`AUTO` green badge added). A VLM hint also rides on the Mode line — `[VLM]` if the active session model is image-capable, `[text-only]` if not — so the user knows whether attaching `/vlm` screenshots will help.

6. **Image-source marker in status assets block** ([chat.py](chat.py) `_format_assets_summary`). Each generated sprite is either txt2img (text prompt only) or img2img chained from a parent sprite (the `from_image` field in `<assets>` specs — used for animation walk-cycles, attack frames, etc., where SD-Turbo img2img preserves the parent's silhouette + palette). The status panel now shows a per-asset badge: dim `[txt2img]` for plain text-prompt generation, cyan `[img2img←<parent_name>]` for chained frames, yellow `[txt2img (img2img fallback)]` when img2img failed and the asset fell back to txt2img (which usually means the child won't visually match the parent). Helps the user spot animation-chain failures at a glance.

Coverage: 9 unit tests for the skeleton detector with the literal DK 374-byte fixture pinned ([tests/test_skeleton_payload.py](tests/test_skeleton_payload.py)), 19 for the VLM classifier across Qwen / LLaVA / DeepSeek / Claude / GPT-4o families plus negative cases for text-only models ([tests/test_vlm_classifier.py](tests/test_vlm_classifier.py)), 6 for the deliberation-threshold pins including tag-opener-latch and env-var-disable ([tests/test_deliberation_thresholds.py](tests/test_deliberation_thresholds.py)).

**Listening fixes — five edges where signals were getting dropped (`agent.py`)**

Trace `a-game-of-donkey-kong-all-char_20260514_175012` revealed five distinct places where a clear signal — from the user, from the harness, or from the model's own diagnose — was being ignored. Each fix tightens one edge:

1. **Broadened `_CODE_LOCK_PATTERNS`** ([agent.py:448](agent.py)). The user typed *"this is trivial no other changes there"* at turn [03]. The previous patterns all required the literal word "code" or a specific media noun (asset/sprite/image/...) — *"no other changes"* slipped through, `rewrite_exemption_armed` fired anyway, and the model emitted a full `<html_file>` instead of the minimal swap the user asked for. Seven new patterns cover *"no other changes"*, *"this is trivial"* / *"trivial change/fix/swap"*, *"just/only swap|switch|rename|move|flip|toggle"*, *"leave the rest"*, *"nothing else"*. Defenses: bare *"just"* / *"only"* without a minimal-scope verb don't match; behavior-bug feedback doesn't accidentally lock unless it also expresses scope-lock intent.

2. **Stuck-loop hard gate at `_repeat_sig_streak >= 3`** ([agent.py](agent.py)). The DK trace hit streak 3 on the INPUT subsystem; the model emitted patches anyway. There was no further escalation lever. Now: at streak ≥ 3 AND `_subsystem_hint` matches, `_force_question_subsystem` is set to the hint dict, and the next `_build_fix_prompt` call substitutes a `<question>`-only hard-gate prompt that lists three concrete options *(a) rewrite the keydown handler from scratch (b) ask for a specific approach (c) ship as-is with the known failure*. All other tags (`<patch>`, `<html_file>`, `<plan>`, `<diagnose>`) are blocked by prompt. Flag clears on consumption AND on any clean iter (so a session that's actually making progress never hits the gate). Pulls the human in to break the loop when the model can't translate the subsystem signal into a fix.

3. **Diagnose-vs-patch subsystem-coherence note** ([agent.py](agent.py) `_diagnose_mentions_subsystem` + `_patches_touch_subsystem_idents`). Turn [08]'s `<diagnose>` named *"barrel drop threshold + procedural fallback coordinate bug"* — but the harness had been reporting INPUT failure for 3 iterations and the patches touched barrels/coords, not input handlers. The model talked itself out of fixing the implicated subsystem. New helpers scan the diagnose body (case-insensitive substring) and each patch's SEARCH+REPLACE text for the hint's identifier tokens. When BOTH are silent on the implicated subsystem, a coaching note is queued for the next turn: *"COHERENCE NOTE — the harness has been reporting INPUT failure but your last `<diagnose>` and patches did not mention any input code. Were you intentionally addressing a different bug?"* Light touch — doesn't reject the patch, doesn't block the iter.

4. **Audit of subsystem-hint coaching path** (`tests/test_subsystem_hint.py` extended). Confirmed via regression test with the literal trace signature: `_subsystem_hint` correctly returns the input hint when given the trace's *"Controls are not wired up"* signature. The trace's `coaching_injected` event firing the generic fallback was because the trace pre-dated the Item 1a deploy; the code path is correct.

5. **Exit-decision turn before silent loop end** ([agent.py](agent.py) `run()` exit path). The DK session ended with patches applied but no `<done/>` or `<confirm_done/>` — user got back a half-fixed game with no clear handoff signal. When the iter cap is reached with `_previous_report_ok is False` (and no `awaiting_confirm`, no pending user feedback, no force-ship), the agent now injects one final EXIT DECISION TURN prompt: the model MUST emit either `<done/>` + `<notes>...</notes>` (handoff summary: what works / what's broken / workaround) OR `<question>...</question>` (specific blocker). All other tags rejected by prompt. The final-iter test guarantee then runs against whatever's on disk, and `_record_session_outcome(ok=...)` reflects the actual report — the model's `<notes>` is advisory only, not load-bearing on the outcome.

Coverage: 16 unit tests for the broadened code-lock patterns including the literal DK feedback text pin (`tests/test_feedback_code_lock.py`), 8 for the stuck hard-gate flag mechanics + prompt content (`tests/test_stuck_hard_gate.py`), 12 for the diagnose-patch coherence helpers (`tests/test_diagnose_patch_coherence.py`), 14 for the exit-decision turn gate + reply parsing + prompt content (`tests/test_exit_decision_turn.py`), plus one new regression test in `tests/test_subsystem_hint.py` pinning the literal trace signature.

**Feedback-routing fix: behavior-bug detector (`agent.py`)**

The same 20260514 trace exposed a parallel failure on the feedback side. When the user typed *"mario does not climb the ladder, even when below it and i push the key up, dont change anything else"*, the harness misclassified the feedback as an art/sound change because "mario" and "ladder" were registered asset names — the `_feedback_is_art_change` heuristic returned True on any asset-name match. The MEDIA-CHANGE DIRECTIVE then injected *"The feedback above is about ART/SOUND, not code"* into the next user turn, and the model dutifully emitted `<assets>` re-renders instead of fixing the climb bug. The .jsonl shows `media_change_directive_injected ... art_change: true` fired on 7+ consecutive feedback turns.

`_feedback_is_behavior_bug(text)` adds a gate: when the feedback contains a negation paired with a behavior verb within a small window (*"does not climb"*, *"can't move"*, *"won't reset"*, *"doesn't respond"*), or an explicit complaint noun (*"bug"*, *"broken"*, *"crashing"*, *"frozen"*, *"stuck"*, *"glitching"*), the MEDIA-CHANGE DIRECTIVE is suppressed even if `_feedback_is_art_change` would otherwise fire. The `_BEHAVIOR_VERBS` set is genre-free — gameplay actions like `climb` / `move` / `jump` / `fire` / `roll` / `spawn` / `reset` — and intentionally EXCLUDES visual verbs (`look`, `appear`, `render`, `show`) so a legitimate art complaint like *"the dragon doesn't look right"* still routes to the art-change path. The suppression is traced as `media_change_directive_suppressed`.

**Mutable `<todos>` artifact (`agent.py` + `prompts_v1.py`)**

The one deepagents idea that survived the "no free reflection" filter. `<todos>...</todos>` is an optional checklist the model can rewrite each turn; the agent persists it to `games/traces/<session>.todos.md` and replays it in `_build_structured_summary` so compaction doesn't drop it. Encouraged-not-required: when the model doesn't use it, the universal probes / criteria still cover acceptance. Dropped from the small-class system prompt (`_SMALL_DROP`) so the 6 KB budget for sub-30B models holds.

**Coverage**

19 unit tests for the VLM/text model-modality classifier across Qwen, LLaVA, DeepSeek, Claude and GPT-4o families (`tests/test_vlm_classifier.py`), 15 for `patches.py` format classification (`tests/test_format_rejection.py`), 15 for the probe-quality classifier including pinned regressions on both DK traces (`tests/test_probe_quality.py`), 15 for the patch-delta token-repetition detector including the literal DK `}else{ × 11` pin (`tests/test_patch_replace_repetition.py`), 16 for the broadened code-lock patterns with literal DK feedback text pin (`tests/test_feedback_code_lock.py`), 14 for the exit-decision turn gate + reply parsing + prompt content (`tests/test_exit_decision_turn.py`), 13 for the feedback behavior-bug detector with the literal DK feedback text pinned as a regression (`tests/test_feedback_behavior_bug.py`), 12 for the diagnose-vs-patch subsystem-coherence helpers (`tests/test_diagnose_patch_coherence.py`), 11 for the input-responsiveness keyword detector (`tests/test_input_responsive_synthesis.py`), 10 for the subsystem-hint helper + focused-slice biasing including the DK-20260514_175012 sig pin (`tests/test_subsystem_hint.py`), 9 for the widened probe re-parse gate including the DK-20260514 seed-session pin (`tests/test_probe_reparse_gate.py`), 9 for the skeleton-payload detector with the literal DK-20260514_214747 374-byte pin (`tests/test_skeleton_payload.py`), 8 for the stuck-loop hard-gate at streak≥3 (`tests/test_stuck_hard_gate.py`), 8 for the format-doctor early-escalation on looped streams (`tests/test_format_doctor_early_escalation.py`), 8 for the `<todos>` parser (`tests/test_todos_artifact.py`), 6 for the deliberation-guard threshold pins + tag-latch + env-disable (`tests/test_deliberation_thresholds.py`), 6 for the final-iter test guarantee (`tests/test_final_iter_test_guarantee.py`). All harness-side and pure-function — the full set runs in well under two seconds.

### Generated sprites — Z-Image-Turbo, no server

Most arcade-style games look better with real sprite PNGs than with
procedural `ctx.fillRect` rectangles. The agent can now request them.

In Phase A the model emits an optional `<assets>` block alongside
`<plan>` / `<criteria>` / `<probes>`:

```xml
<assets>
[
  {"name": "ship",     "prompt": "pixel-art retro arcade spaceship facing right, white outline, transparent background"},
  {"name": "asteroid", "prompt": "pixel-art irregular grey rocky asteroid, transparent bg", "size": "64x64"},
  {"name": "explosion","prompt": "pixel-art orange explosion sprite, transparent bg",        "size": 96}
]
</assets>
```

The harness then (all in-process, no server, no subprocess):

1. Calls `assets.try_load_image_generator()` which constructs an
   in-tree `ZImageTurboGenerator` (vendored from `diffusion_manager.py`
   on 2026-05-06 — Z-Image-Turbo path only, watermarking and other
   pipeline branches stripped). Pure Python `import` chain; same
   process as `chat.py`, same GPU.
2. Lazy-loads the **Z-Image-Turbo** weights once per session (~30-60 s
   the first time the agent emits `<assets>`; ~2-4 s per image after).
   Pipeline never loads on sessions that don't request assets, so the
   cost is opt-in per turn.
3. Generates each missing PNG at native 768×768, downscales with PIL
   Lanczos to the per-asset target size (default 128 px), then runs
   a **chroma-key alpha pass** (`_chroma_key_to_rgba`) to remove the
   solid background Z-Image-Turbo always produces — even when the
   prompt says "transparent background". The pass samples 8 corner
   and edge pixels, requires ≥6/8 agreement to pick a dominant bg
   color, and alpha-masks pixels within tolerance. If no clear bg
   color is detected (e.g. a foreground that fills the frame), the
   pass leaves the image alone — better to skip masking than risk
   eating real pixels. Per-asset `bg_color` + `alpha_pixel_ratio`
   are logged in the trace so you can see whether the chroma-key
   actually fired and how aggressively.
4. Caches by `sha256(model_id, normalized_prompt, size)` so repeated
   prompts across sessions are free. Cache stores the post-chroma-key
   RGBA PNG.
5. Saves PNGs into `games/<slug>_<ts>_assets/<name>.png`, hard-linked
   from the cache.
6. Prepends a `GENERATED ASSETS` block to the first-build prompt
   listing each `name → ./<rel-path>` so the model can `<img src>` or
   `new Image()` directly.

The existing `image-load-race` playbook bullet (`memory.py:858`) tells
the model to wait for `await img.decode()` before drawing; that
contract holds whether sprites are model-generated or hand-supplied.

### Animation frame chains — SD-Turbo img2img, optional

The default sprite pipeline above generates each PNG independently from
its text prompt. That's fine for a static ship or wall texture, but it
breaks for **animation sequences** — two text-only generations of
`"alien legs together"` and `"alien legs apart"` produce two *different
characters*, not two poses of the same one.

The agent has a second pipeline for this case: **SD-Turbo** (512×512,
~2 GB weights, 1–4 step img2img), wired alongside Z-Image-Turbo. The
model declares the chain in the `<assets>` block by adding a
`from_image` field referencing a sibling asset, plus a `strength` in
[0.05, 1.0] that controls how much the next frame can deviate from the
init image:

```xml
<assets>
[
  {"name": "alien_walk1", "prompt": "8-bit pixel alien, legs together"},
  {"name": "alien_walk2", "prompt": "8-bit pixel alien, legs apart",
   "from_image": "alien_walk1", "strength": 0.45}
]
</assets>
```

The pipeline is:

1. `parse_assets_block` (`assets.py:180`) preserves `from_image` /
   `strength` on each spec; strength is clamped to `[0.05, 1.0]` and
   defaults to `0.45` (preserves silhouette + palette, lets the prompt
   move the pose).
2. `_topo_sort_specs` (`assets.py:793`) orders the list so parents
   generate before children. Cycles fall back to the original order.
3. `generate_assets` (`assets.py:827`) only lazy-loads SD-Turbo when
   at least one spec has `from_image`. Root frames go through
   Z-Image-Turbo as usual; chained children route to
   `Img2ImgGenerator.generate(prompt, init_image_path, strength=...)`
   (`assets.py:522`), which wraps `diffusers.AutoPipelineForImage2Image`
   from the `stabilityai/sd-turbo` repo.
4. The cache key for chained frames includes the parent file's
   `(size, mtime)` so regenerating the parent invalidates downstream
   frames.

If SD-Turbo isn't installed (or the chain fails for any reason — OOM,
import error, NSFW filter), the agent silently **falls back to
txt2img** for the dependent frame. The session continues; just without
frame coherence.

The model is taught about the schema by the always-on asset guideline
in `prompts_v1.ASSETS_FORMAT.guidelines` (`prompts_v1.py:165–179`),
which includes the walk-cycle example above. Words like
`animation`, `walk cycle`, `frame`, `animated` in the user's goal
make the model more likely to declare the chain.

SD-Turbo cross-platform:

| Platform | Dtype | Smoke test |
| --- | --- | --- |
| Linux + NVIDIA | fp16 | `.venv/bin/python scripts/_smoke_img2img.py` |
| macOS + Apple Silicon | fp16 (MPS-stable, unlike Z-Image-Turbo which needs fp32) | same |
| No GPU | refuses to load (CPU img2img is ~30 s/frame, not useful) | n/a |

`scripts/_smoke_img2img.py` generates frame 1 (txt2img) then frame 2
(img2img from frame 1 at strength 0.45), saves both to
`games/_smoke/img2img/`, and prints a coarse luminance-similarity
score (>0.55 = clearly the same character, <0.30 = unrelated).

SD-Turbo weight locations follow the same resolution order as
Z-Image-Turbo above — `$DIFFUSION_MODELS_DIR/sd-turbo/`, then the home
bases (`~/.Diffusion_Models/sd-turbo/`, etc.), then HuggingFace cache.
`scripts/install_diffuser.sh` pre-fetches the ~2 GB weights by default
(opt out with `--skip-img2img`).

### What you need

|                          | Linux + NVIDIA              | macOS (Apple Silicon)         | No GPU         |
| ------------------------ | --------------------------- | ----------------------------- | -------------- |
| **Hardware**             | NVIDIA GPU, ≥10 GB VRAM     | Apple Silicon, ≥16 GB RAM     | n/a            |
| **Install**              | `./scripts/install_diffuser.sh` | `./scripts/install_diffuser.sh` | skip       |
| **Torch flavor**         | nightly cu130 (default; override via `TORCH_CUDA=121`) | nightly w/ MPS         | n/a            |
| **Model weights (~5 GB)**| auto-downloaded to `~/.cache/huggingface/hub/` on first `<assets>` use, OR pre-placed (see below) | same | n/a            |
| **First-run cost**       | ~30 s pipeline load + ~14 s/image (256×256) | ~60 s + ~30–60 s/image (estimate, **experimental**) | n/a |
| **Subsequent runs**      | ~14 s/image; cache hits = free hard-link | longer on MPS, still cache-free on hits | n/a |
| **When unavailable**     | `try_load_image_generator()` returns `None` silently → agent logs *"Z-Image-Turbo not reachable, drawing procedurally"* and the session continues exactly as before. | same | same   |

The Linux path is **verified working** on the user's NVIDIA GB10
(testing record in commit `504b4a0`). MPS on Apple Silicon is
experimental — Z-Image-Turbo's authors test on CUDA only; whether it
works on a given Mac depends on the diffusers / torch nightly's MPS
coverage of the model's specific ops. The pipeline detects MPS at
runtime and uses `float16` instead of `bfloat16` (MPS bf16 is uneven).

### Setup — one command

The canonical entry is `./scripts/setup.sh` (see [Quick start](#quick-start)),
which installs both sprite + sound pipelines along with the rest of the
agent. Internally it calls `scripts/install_diffuser.sh`, which you can
also run directly when you only want to refresh the GPU stack:

```bash
./scripts/install_diffuser.sh         # Linux: cu130 nightly torch
                                       # macOS: nightly w/ MPS
                                       # other: CPU torch + warning
```

The script:
1. Detects your platform via `uname -s` and picks the right torch index
   URL (`download.pytorch.org/whl/nightly/cu130` on Linux,
   `…/whl/nightly/cpu` on macOS — that's the wheel that includes MPS
   support).
2. Installs torch + torchvision + torchaudio.
3. Installs diffusers from **git HEAD** — `ZImagePipeline` and
   `StableAudioPipeline` updates land in HEAD before tagged releases.
4. Installs transformers + accelerate + safetensors + pillow.
5. **`pip install -r requirements-diffuser.txt`** — `soundfile`, `torchsde`,
   and pins for Stable Audio (`<sounds>`); `./scripts/setup.sh` gets these via this same script (no extra pip step).
6. Verifies CUDA / MPS, imports `ZImagePipeline`, **`StableAudioPipeline`**,
   **`soundfile` / `torchsde`**, and runs `assets.try_load_image_generator()`.

If you have an older NVIDIA GPU that needs CUDA 12.x:

```bash
TORCH_CUDA=121 ./scripts/install_diffuser.sh
TORCH_CUDA=124 ./scripts/install_diffuser.sh
```

After install, **smoke-test both pipelines end-to-end** before running
a real session:

```bash
.venv/bin/python scripts/_smoke_doom.py     # one PNG via Z-Image-Turbo
.venv/bin/python scripts/_smoke_audio.py    # three OGGs via Stable Audio Open
```

This generates a single 256×256 PNG from the prompt *"doom video game
cover art, demon with red skin and horns screaming"*, saves it to
`games/_smoke/doom.png`, and prints the path. ~120 s on first run
(pipeline load + first-call CUDA kernel compile); ~14 s on subsequent
runs. If this works, real sessions will work.

### How the model knows to use sprites

The agent is biased toward emitting `<assets>` for any canvas-rendered
game; small models (qwen3.6, gpt-oss) will sometimes politely skip it
without a stronger nudge. Two layers:

1. **Always-on framing.** `prompts_v1.PLAN_INSTRUCTION` says `<assets>`
   is **EXPECTED** (not optional) for canvas games and **SKIP only for
   pure-DOM apps** (todo / calculator / tic-tac-toe).
2. **Goal-keyword escalation.** `prompts_v1._detect_art_intent(goal)`
   scans the user's goal for modality words (`sprite`, `art`,
   `graphic`, `pixel`, `image`, `texture`, `asset`, `draw`, `icon`,
   `render`, `illustration`, plus polish adjectives like `cool` /
   `gorgeous` / `stunning`). Any match injects an `ART INTENT DETECTED`
   callout above the planning template with `ULTRA IMPORTANT` framing,
   making `<assets>` mandatory for that turn.

The keyword list is intentionally **genre-free** — only words about
output character, never subject matter. So "build me a doom-like
shooter" doesn't fire; "build me a doom-like shooter with cool sprite
art" does. This matches the project rule against hardcoded genre
lists for retrieval / probes / skeletons.

### Where the model weights live (cross-platform)

Z-Image-Turbo weights are *data*, not code — they live outside the
repo by design (~5 GB doesn't belong in git). The loader searches the
following paths and uses the first one that exists:

| Order | Path                                | Note                                      |
| :---: | ----------------------------------- | ----------------------------------------- |
| 1     | `$DIFFUSION_MODELS_DIR/Z-Image-Turbo/` (or `Tongyi-MAI_Z-Image-Turbo/`) | **Recommended override** |
| 2–5   | Home bases — **dot-folders first**: `~/.Diffusion_Models/`, `~/Diffusion_Models/`, then `~/.Models_Diffusers/`, `~/Models_Diffusers/` on macOS (Linux tries the *Models_Diffusers* pair first). Same subfolder names as above. |
| —     | `/home/jonathan/Models_Diffusers/Z-Image-Turbo/` | Legacy, kept for compat       |
| —     | `./models_diffusers/Z-Image-Turbo/` | Repo-relative — for portability           |
| last  | **HuggingFace** `Tongyi-MAI/Z-Image-Turbo` | Auto-download on first use → `~/.cache/huggingface/hub/` (`HF_HOME` honored). |

**Stable Audio weights** use the **same home bases** (hidden first). Expected checkpoint dir: **`stable-audio-open-1.0`** (or `audio/stable-audio-open-1.0`) under those roots, or **`$AUDIO_MODELS_DIR`**, or **`$DIFFUSION_MODELS_DIR/audio/`**. If missing, diffusers pulls **`stabilityai/stable-audio-open-1.0`** into the HF cache (often silent; if you get **403/401**, see [HF troubleshooting](#hugging-face-login-only-if-downloads-fail)). **Game output OGGs** are always written under **`games/<session>_sounds/`** — only the **model checkpoint** lives in the paths above.

To use a custom location (e.g. an external SSD), add to your shell rc:

```bash
# Linux / macOS
export DIFFUSION_MODELS_DIR=/Volumes/External/Diffusion_Models   # mac
export DIFFUSION_MODELS_DIR=/data/Diffusion_Models                # linux
```

Skip every step above and the agent still works — it just won't have
sprites. **Skip `<assets>` for DOM-only games** (todo list,
calculator, tic-tac-toe); the format spec instructs the model to emit
`<assets>` only when sprite art would help.

### Generated sounds — Stable Audio Open, no server

A direct parallel to the sprite pipeline above, but for short audio
clips. Most arcade games feel cheaper without sound; the agent can now
request SFX and looping background music in Phase A alongside `<plan>`,
`<criteria>`, `<probes>`, and `<assets>`:

```xml
<sounds>
[
  {"name": "laser",     "prompt": "short retro arcade laser shot, 8-bit synth blip", "duration": 0.4},
  {"name": "explosion", "prompt": "short pixelated explosion, 8-bit boom",            "duration": 0.8},
  {"name": "music",     "prompt": "loopable 8-bit chiptune background, 90 bpm",       "duration": 12.0, "loop": true}
]
</sounds>
```

The harness:

1. Calls `sounds.try_load_audio_generator()` which constructs a
   `StableAudioGenerator` (vendored shape, mirrors `assets.py`). Pure
   Python `import` chain; same process as `chat.py`, same GPU.
2. Lazy-loads **Stable Audio Open 1.0** weights once per session
   (~30-60 s on first `<sounds>` request; ~3-8 s per second of audio
   after on Apple Silicon MPS, faster on CUDA). Pipeline never loads
   on sessions that don't request sound, so the cost is opt-in per turn.
3. Generates each missing OGG via `diffusers.StableAudioPipeline` at
   the model's native 44.1 kHz, encodes to OGG-Vorbis via `soundfile`.
   `duration` is clamped to [0.2 s, 12.0 s] per sound; `loop: true`
   marks a clip for `<audio loop>` use in the loader.
4. Caches by `sha256(model_id, normalized_prompt, duration)` so
   repeated prompts across sessions are free.
5. Saves OGGs into `games/<slug>_<ts>_sounds/<name>.ogg`, hard-linked
   from the cache.
6. Prepends a `GENERATED SOUNDS` block to the first-build prompt
   listing each `name → ./<rel-path>` plus a complete `new Audio(...)`
   loader pattern (cloneNode for overlap-safe SFX; user-gesture unlock
   so browsers actually play audio).

A goal-keyword detector (`prompts_v1._detect_audio_intent`) escalates
`<sounds>` from "expected" to **AUDIO INTENT DETECTED — required this
turn** when words like `sound`, `audio`, `music`, `sfx`, `chiptune`
appear in the user's goal. Genre-free per the project rule — only
words about output character, never subject matter.

Compaction's state-anchor message preserves sound paths alongside asset
paths, so feedback like *"use the music you generated"* survives
30+-turn sessions.

|                          | Linux + NVIDIA / macOS Apple Silicon         | No GPU         |
| ------------------------ | -------------------------------------------- | -------------- |
| **Hardware**             | NVIDIA ≥10 GB VRAM, or Apple Silicon ≥16 GB | n/a            |
| **Install**              | `./scripts/setup.sh` (GPU stack + audio + sprites) | `./scripts/setup.sh --no-gpu` (escape hatch only) |
| **HF auth**              | Usually **none** — weights cache under `~/.cache/huggingface/hub/`. If Stable Audio fails with **403/401**, accept the license on the HF model page + `huggingface-cli login` or `export HF_TOKEN=…` | n/a |
| **Model weights (~5 GB)**| auto-downloaded to `~/.cache/huggingface/hub/` on first `<sounds>` use | n/a |
| **First-run cost**       | ~30 s pipeline load + ~3-8 s/sec of audio   | n/a            |
| **Subsequent runs**      | cache hits = free hard-link                  | n/a            |
| **When unavailable**     | `try_load_audio_generator()` returns `None` silently → agent logs *"Stable Audio Open not reachable, shipping silent game"* and the session continues exactly as before. | same |

#### Known issue: torchsde infinite recursion (handled)

Stable Audio Open's default scheduler
(`CosineDPMSolverMultistepScheduler`) uses `torchsde` for SDE noise.
On every step, float-precision drift makes the requested midway value
land ~1e-5 outside the Brownian interval bounds; `torchsde._Interval._split`
bisects the gap and recurses forever, never reaching the unreachable
target — `RecursionError` regardless of `sys.setrecursionlimit`. We
monkey-patch `_Interval._split` in `sounds.StableAudioGenerator._lazy_init`
to clamp `midway` to `[_start, _end]` before bisecting; out-of-bounds
queries get a single non-recursive `_split_exact` at the boundary.
Audio output is unaffected because `BrownianInterval.__call__` already
clamps the requested time to the interval before any of this fires.
Idempotent and process-local; only fires when audio is actually used.

### Roadmap items also shipped on this branch

Three of the highest-ROI gaps from the original roadmap landed in the
same commit family as the 10 ports above:

- **Project-config injection** (`agent._read_project_config`). At
  session start we read `AGENTS.md` and `CLAUDE.md` from the working
  directory (falling back to `out_path.parent.parent`), cap the total
  at 6 KB, and append the contents as a `<project-context>` block at
  the END of the system prompt. Pi-mono pattern: a repo can lock in
  "always vanilla JS, never React" once and every session inherits it.

- **API hallucination guard** (`tools._check_api_allowlist`). Inside
  `run_micro_probes`, scan inline scripts for `<receiver>.<method>(`
  patterns where the receiver name is one of the strict conventions
  (`ctx`, `audioCtx`, `cvs`, …) and the method is not on the canonical
  allowlist for that receiver type. Reported as a *warning*, not an
  error — false-positive risk is real (user objects named `ctx`), and
  Chromium has the final word. Bias: false negatives over false
  positives.

- **Bullet-on-demand retrieval** (pi-mono "skills" pattern). New
  `mode="hybrid"` for `render_playbook_block`: ships top-3 with full
  body + remaining bullets as ID-only index entries with their tags.
  When the model wants the body of an indexed bullet, it emits
  `<lookup_bullet>id</lookup_bullet>` in its reply; the agent's
  `_extract_and_queue_lookups` resolves and queues the body for
  injection into the *next* user-turn message via
  `_flush_user_injections`. Capped at 5 lookups per turn so a chatty
  reply can't drown context. The plan stage (broad retrieval) defaults
  to hybrid; the code stage (narrow, K ≤ 3) stays full.

Coverage: 23 unit tests in `tests/test_microprobes.py`,
20 in `tests/test_retrieval.py`,
9 in `tests/test_compaction.py`,
24 in `tests/test_patches.py`,
8 in `tests/test_lookup.py`,
7 in `tests/test_project_config.py`.
Total: **89 passing**.

---

## Compared to pi-mono and OpenCoder

These three tools are **not direct substitutes**. Pi-mono is a general-
purpose multi-language coding agent with native function-calling and a
broad tool taxonomy. OpenCoder is an open *model* + training recipe, not
an agent at all. Coding Box is a verticalized HTML/JS-game agent that
ships with browser verification, cross-session memory, and visual review
out of the box. We borrowed liberally from both; this table is meant to
make the borrowing explicit and the divergences honest.

| Capability                                    | Coding Box (this repo)                                                       | pi-mono coding-agent                              | OpenCoder                              |
| --------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------- |
| **Domain**                                    | HTML5 single-file games + browser-runnable code (verticalized)              | General-purpose multi-language coding             | (Model + training recipe, not agent)   |
| **Model providers**                           | Ollama only (deliberate)                                                    | OpenAI, Anthropic, GLM, Ollama, more              | Standalone — bring your own runtime    |
| **Tool / output format**                      | XML tags parsed from text (`<patch>`, `<html_file>`, `<probes>`, …)         | Native function-calling (TypeBox-validated)       | n/a                                    |
| **Patch / edit format**                       | SEARCH/REPLACE w/ fuzzy norm + uniqueness + non-overlap + reverse apply ★   | `edits[].oldText/newText` w/ same engine ★ source | n/a                                    |
| **Patch repair layer**                        | BOM, CRLF, internal-fence stripping ★ (ported)                              | BOM, CRLF, fuzzy normalize ★ (origin)             | n/a                                    |
| **System prompt assembly**                    | Per-format `FormatSpec` + deduped guidelines ★ (ported)                     | Per-tool `promptGuidelines` ★ (origin)            | n/a                                    |
| **Long-session compaction**                   | Two-tier: HTML elision → deterministic structured anchor ★ (ported pattern) | LLM-summarized w/ fixed skeleton ★ (origin)       | n/a                                    |
| **Retrieval quality ranking**                 | `relevance × (1 + 0.10·tanh(score/5))` ★ (inferred from #2)                 | None (single AGENTS.md context)                   | Star-count + recency at training       |
| **Two-stage broad → narrow**                  | Plan-stage broad / code-stage narrow at retrieval ★ (ported)                | Single retrieval                                  | Two-stage SFT at training ★ (origin)   |
| **Context dedup + budget cap**                | Shingle Jaccard ≥ 0.85 + char-budget per stage ★ (inferred from data dedup) | None                                              | File-level MinHash dedup at training   |
| **Pre-Chromium structural probes**            | `run_micro_probes` (size, structure, scripts, brackets, elision) ★ (ported) | None                                              | Execution-filter at training (Educational-Instruct) |
| **Real-browser verification**                 | **Yes** — Playwright Chromium, RAF, frozen-canvas, input smoke, screenshot  | bash + user-invoked tests                         | n/a                                    |
| **VLM screenshot review**                     | **Yes** — vision model gets `.png` on fix turns                             | No                                                | n/a                                    |
| **Web research grounding (per-goal)**         | **Yes** — Wikipedia title-match w/ Levenshtein typos, no genre allowlist    | No                                                | n/a                                    |
| **Cross-session memory**                      | **Yes** — playbook with helpful/harmful counters + offline learner          | AGENTS.md / CLAUDE.md (project-level, manual)     | At training only                       |
| **Acceptance criteria + executable probes**   | **Yes** — `<criteria>` + JSON `<probes>` literally executed in the page     | No                                                | n/a                                    |
| **Best-of-N w/ runtime scoring**              | **Yes** — sample N, score each in the same browser, pick winner             | No                                                | n/a                                    |
| **Diagnose-then-fix combined turn**           | **Yes** — `<diagnose>` required before `<patch>` on fix turns               | No                                                | n/a                                    |
| **Adaptive temperature (build vs fix)**       | **Yes** — 0.7 build / 0.25 fix                                              | Single temp                                       | n/a                                    |
| **Stuck-loop reflection ladder**              | **Yes** — "5–7 different sources" mode after 2+ failures                    | No                                                | n/a                                    |
| **Repetition-loop watchdog**                  | **Yes** — sliding-window detector kills wedged streams                      | Stall watchdog only                               | n/a                                    |
| **User feedback mid-stream**                  | HIGHEST-PRIORITY banner; drained at next turn boundary                      | Steering messages (similar)                       | n/a                                    |
| **Save-best-on-clean + regression detect**    | **Yes** — every clean turn → `best.html`; revert prompt if next breaks      | git (manual)                                      | n/a                                    |
| **`<done/>` then plain text auto-extends**    | **Yes** — same file, continuation mode                                      | Follow-up + steering                              | n/a                                    |
| **Architect/editor 2-call split**             | Optional, complexity-gated                                                  | No                                                | n/a                                    |
| **AGENTS.md / project-config injection**      | **Yes** — `AGENTS.md` + `CLAUDE.md` auto-loaded from cwd ★ (ported)        | Yes — both `AGENTS.md` and `CLAUDE.md`            | n/a                                    |
| **API hallucination guard**                   | **Yes** — receiver-name allowlist for canvas2d / AudioContext / canvas-elt | No                                                | n/a                                    |
| **Skills / lazy-loaded reference docs**       | **Yes** — hybrid playbook mode + `<lookup_bullet>` tag ★ (ported pattern)  | Yes — skills advertised, model `read`s on demand  | n/a                                    |
| **Model-agnostic across Anthropic/OpenAI/etc.** | **No** (deliberate; Ollama-focused)                                       | Yes                                               | n/a                                    |
| **JSON repair for streamed tool args**        | **No** — we use XML, no JSON args to repair                                 | Yes — `repairJson` + partial-json fallback        | n/a                                    |

★ = pattern ported from one of the other tools (or inspired by their published recipe).

**TL;DR positioning:**

- **Pi-mono coding-agent** is the right choice if you want a *general-
  purpose, multi-provider* coding agent that works across many languages
  with native function-calling. It has the better tool framework; we
  ported its patch engine, prompt assembly, and compaction patterns.
- **OpenCoder** is the right choice if you want to *understand or train*
  a top-tier open code LLM end-to-end. It's a model + recipe, not a
  daily-driver agent; we ported its data-curation principles to
  inference-time context curation.
- **Coding Box** is the right choice if you want to drive a small/mid
  local Ollama model to ship browser-runnable HTML/JS specifically, with
  Chromium as ground truth and a playbook that gets smarter every
  session. Verticalization buys us the runtime-validation layer; the
  cost is we don't help you write a Rust kernel module.

---

## How the agent compounds

Most "AI coding tools" are a static prompt. This one isn't. Three loops
run on different timescales, and each one *feeds the next*:

```
   ┌─────────── inside one session ────────────┐
   │  plan → build → test → diagnose → fix     │
   │  ⤷ user feedback wins every tie           │
   │  ⤷ best.html saved on every clean turn    │
   └────────────────┬──────────────────────────┘
                    │ trace, snapshots, reports
                    ▼
   ┌─────── across sessions (the playbook) ────┐
   │  Reflector reads completed traces and     │
   │  proposes bullet deltas:                  │
   │    + ADD a transferable rule we learned   │
   │    + helpful++ on bullets that fired on   │
   │      a passing run                        │
   │    + harmful++ on bullets active during   │
   │      stuck-loop failures                  │
   │  Curator merges deltas deterministically  │
   │    (no wholesale rewrites — ACE pattern)  │
   │  Pruner drops bullets where harmful>>>    │
   │    helpful so the playbook self-curates   │
   └────────────────┬──────────────────────────┘
                    │ updated playbook.jsonl
                    ▼
   ┌────── feeds the next session ─────────────┐
   │  v1 system prompt retrieves the top-K     │
   │  relevant bullets per goal and injects    │
   │  them as a <playbook> block. The first    │
   │  build is informed by every lesson the    │
   │  agent has ever learned.                  │
   └───────────────────────────────────────────┘
```

**What the playbook actually contains (sample bullets):**

```
[rotation-thrust-vector] tags=[ship,thrust,rotation,asteroids,angle]
  When applying thrust to a rotatable ship/character, compute velocity
  from its facing angle: vx = Math.cos(angle) * speed, vy = Math.sin(angle)
  * speed. NEVER use plain world-axis dx/dy. ...

[first-click-safety] tags=[grid,minesweeper,first-click,fairness]
  For grid games with hidden hazards, ensure the first player interaction
  is always safe by generating hazards AFTER the first click. ...

[obstacle-gap-bounds] tags=[obstacles,random,gaps,playability,flappy]
  When generating scrolling obstacles with randomized vertical gaps,
  clamp the gap position to guarantee it never clips the screen edges
  or becomes impassable. ...
```

The first set ships seeded — distilled from the OpenGame paper, the
Macklon canvas-bug taxonomy (arXiv 2201.07351), JS13k post-mortems, and
mining the actual system prompts of Aider / Cline / OpenHands / Bolt /
Continue. As you build games, the offline learner adds bullets like
`first-click-safety` and `obstacle-gap-bounds` automatically — those
two were learned by the agent itself during a 10-game battery.

**How user feedback makes it smarter:**

When you type feedback mid-run (e.g. "ship is moving sideways instead of
forward"), the agent applies it as the highest-priority injection AND
the trace records:

- the failing report
- the diagnosis the model produced after seeing your feedback
- the fix that landed clean

The Reflector reads that pattern back. If a similar bug surfaces in a
future session, the relevant bullet (or a freshly-minted one) can be
retrieved into the **v1** prompt — so the live model sees it only when you run
with **`prompt_version=v1`** (e.g. `tune.py` or a custom `GameAgent`); the
default **v0** `chat.py` path does not inject that block.

**Closing the loop:**

```bash
# Every battery run can refresh the playbook in one shot (isolated copy under
# the run dir unless you also pass --learn-shared).
python tune.py run --prompt-version v1 --auto-learn --learn-shared

# Or reflect over real (not battery) sessions you've shipped this week.
python learner.py apply games/traces/

# Inspect what the agent has learned.
ls games/memory/playbook.jsonl                # the file
python learner.py walk                        # past sessions (default: games/traces/)
```

For one session’s detail, pass a **trace path** (`.jsonl`) or a session-id
substring — `learner.py show` resolves `games/traces/**/*<id>*.jsonl`.

---

## Web research grounding

Local LLMs in the 20–35B class have **thin world knowledge** for arcade
games. Ask one to build "Missile Command" cold and it will cheerfully
ship Space Invaders with the labels swapped — the May 5 trace at
`games/traces/game-of-misile-command-good-gr_20260505_133453.*` is the
canonical example: player at the bottom moving left/right with arrow
keys, firing bullets *up* at "enemy bases" raining bullets *down*. Not
remotely Missile Command.

Fix: before the planning turn, the agent looks the goal up on Wikipedia
and prepends the result as a `<reference>` block. The v1 planning
prompt then says, in plain English, *treat this as authoritative*. The
same model that previously produced Space Invaders now plans cities,
crosshair, three batteries, fireballs.

**How it works** (`research.py`, `agent.py:903`):

1. `_search_queries(goal)` strips leading/trailing **filler** words
   ("game of", "make a", "good graphics", "the original arcade") so a
   long natural-language goal turns into a tight title-shaped query.
   Wikipedia's `opensearch` endpoint matches against page titles
   starting at the beginning, so "game of misile command" returns
   nothing while "misile command" returns "Missile Command" — the
   stripping is what makes the lookup work.
2. For each query (raw + ` video game` + ` arcade game` suffixes), call
   `opensearch`. Throttled to 1 req / 600 ms — anonymous Wikipedia silently
   returns empty for ~5+ rapid bursts.
3. **Open-domain title filter**: a candidate is accepted only if its
   title (sans disambiguators like `(video game)`) appears in the goal.
   Substring match for clean inputs, **squashed** match for
   concatenated names ("Pac-Man" vs goal "pacman"), and a
   token-Levenshtein fallback for typos ("misile" vs "missile",
   distance 1 → match). Crucially, **no game list is hardcoded** —
   the agent must handle any HTML/JS request open-domain.
4. **Gameishness rank**: when multiple titles pass (e.g. `Snake` the
   animal AND `Snake (video game genre)`), prefer titles tagged
   `(video game)`/`(arcade)` or whose summary description mentions
   "video game", "arcade game", "shooter", "platformer", etc. Stops
   on first gameish hit to minimize calls.
5. Render `<reference source="wikipedia">` with TITLE / DESCRIPTION /
   SOURCE / SUMMARY / GAMEPLAY (parsed from the wikitext "Gameplay"
   section when present). Capped at 1800 chars.

**When it fires:** every fresh session at Phase A. Skipped on
`continuation=True` (you're patching, not rebuilding).

**When it returns nothing** ("make a game where bunnies fight robots",
"a calculator with a sound on click"): the planning prompt falls
through to v1's normal `PLAN_INSTRUCTION` and the model plans from its
priors — same as before.

CLI for sanity-checking by hand:

```bash
.venv/bin/python research.py "missile command"
.venv/bin/python research.py "make a snake game"
.venv/bin/python research.py "make a game where bunnies fight robots"   # → no reference
```

A planning-only smoke test sits at `tests/test_research_planning.py` —
runs one Ollama planning call and checks the resulting `<plan>` against
Missile-Command vs Space-Invaders keyword lists.

---

## Memory hygiene — when learning goes wrong

The agent stores three flavors of "memory" under `games/memory/`. They
are **not all equally trustworthy**, and you should know the
difference before reasoning about why a session went the way it did.

```
games/memory/
├── playbook.jsonl   ← curated bullets w/ helpful/harmful counters
├── skeletons/
│   ├── canvas_basic.html              ← bundled default
│   └── won_<session_id>.{html,json}   ← auto-saved on every clean win
├── goals/<session_id>/                ← per-win record (best.html, outcome.json)
└── mistakes.jsonl                     ← {error_signature, fix_summary} pairs
```

**Trustworthy by construction:**

- **`playbook.jsonl`** is the long-term lessons file. Every bullet has
  a `source` field. Bullets with `source: "seed"` are the **hand-curated
  baseline** distilled from research literature (the OpenGame paper,
  Macklon's canvas-bug taxonomy, Aider/Cline/Bolt prompt mining). They
  are reviewed; they don't go stale.
- **`canvas_basic.html`** is the bundled default skeleton.

**Can be wrong, and how it goes wrong:**

- **`skeletons/won_<session_id>.html`** is dropped automatically every
  time a session passes the harness's automated test. The harness only
  checks "the game runs, accepts input, and animates" — it does **not**
  check "the game is the one the user asked for". So the May 5 broken
  Missile Command session passed (a Space-Invaders-shaped game runs
  perfectly fine!) and saved its file as `won_game-of-misile-command-…`.
  Future Missile Command goals would then *retrieve that file as the
  starting skeleton*, locking in the wrong game on iteration 1. We
  removed it as part of the same commit that added research grounding;
  see `scripts/forget_session.py`.
- **`mistakes.jsonl`** can pin a fix to the wrong root cause if the
  Reflector misreads a trace. Far less impactful than a bad skeleton —
  the entries are short hints, not starting code.
- **Auto-distilled `playbook` bullets** (`source != "seed"`) come from
  the offline learner. They go through the helpful/harmful counter
  loop, so a bad bullet that gets retrieved into stuck-loop turns gets
  pruned; in practice the seed bullets dominate retrieval. Inspect
  with `python learner.py walk` and remove individual entries by
  editing `playbook.jsonl` directly if needed.

**Why this matters now:** before the Wikipedia grounding landed, the
agent had no way to *know* the Missile Command session was wrong. The
research step is the new front-line defense — it shapes the plan before
any code is written, so wrong-game wins should be much rarer going
forward. But everything saved before that fix landed is suspect.

**Cleanup utilities — two flavors:**

For **single-session surgery** (one bad session whose skeleton +
goals you want to forget):

```bash
# See what's stored.
.venv/bin/python scripts/forget_session.py --list

# Wipe one session's skeleton + goals record + matching mistakes entries.
.venv/bin/python scripts/forget_session.py game-of-misile-command-good-gr_20260505_133453

# Dry-run first if unsure.
.venv/bin/python scripts/forget_session.py --dry-run <session_id>
```

For **bulk wipe** when `games/` has accumulated a lot of stale runs
and you want a fresh baseline:

```bash
./scripts/clean_artifacts.sh          # interactive — shows what will go, then asks
./scripts/clean_artifacts.sh --yes    # unattended
```

This deletes per-session HTML, traces, snapshots, every `won_*`
auto-promoted skeleton (but **keeps `canvas_basic.html`**), the
`goals/` cache, `mistakes.jsonl`, and tune-battery run dirs (but
keeps `tune/battery.jsonl`). The seed playbook, asset cache, and
smoke artifacts survive. First run on a busy `games/` typically
shrinks it from several megabytes to a few hundred KB; re-runs are
no-ops once the tree is clean.

What `forget_session.py` touches:

- `games/memory/skeletons/won_<id>.{html,json}`
- `games/memory/goals/<id>/`
- entries in `mistakes.jsonl` whose `session` field matches

What it leaves alone (deliberately): `playbook.jsonl` (no entries are
pinned to a single session), and the on-disk session artifacts under
`games/traces/`, `games/snapshots/`, and the per-session `.html` files
in `games/`. Those are read-only history — delete by hand if you also
want them gone.

---

## Tuning rig & playbook commands

`tune.py` and `learner.py` are the meta-tools that drive "is the agent
getting better?" measurement and offline learning respectively.

**`tune.py` — measure the agent against a fixed battery.** Used to
compare prompt / playbook / probe changes apples-to-apples.

**Defaults:** `tune.py run` uses **quick** mode unless you pass **`--full`** —
quick = `max_iters=2`, `best_of_n=1`; full = `max_iters=4`, `best_of_n=2`.
**`--prompt-version`** defaults to **`v0`**; use **`v1`** for playbook +
criteria + probes. Default model is env **`TUNE_MODEL`** or **`qwen3.6:27b`**.
Chromium is **visible** unless **`--headless`**.

```bash
# Run all goals in games/tune/battery.jsonl — quick mode, v0 prompt.
python tune.py run

# Stronger search budget + v1 prompt + apply learner to this run’s traces.
python tune.py run --full --prompt-version v1 --auto-learn

# Push curated playbook updates to the shared games/memory tree (not only the run’s _memory).
python tune.py run --prompt-version v1 --auto-learn --learn-shared

# Override skeleton: bundled default vs memory retrieval (mirrors production).
python tune.py run --skeleton-mode retrieve        # or default, default_v2

# Enable agent feature flags for A/B (comma-separated: prefill, vlm_critique,
# double_screenshot, architect, all).
python tune.py run --features prefill,architect

# Just one test (asteroids is the canonical "did anything regress?" check).
python tune.py run --tests asteroids

python tune.py list                              # past runs under games/tune/
python tune.py show <run_id>                     # print that run’s SUMMARY.md

# Compare two completed runs by per-test pass/fail.
python tune.py diff baseline_v0 v1_run

# Postmortem one test in a run.
python tune.py why v1_run flappy-bird

# Cluster failure signatures across a run; show which playbook
# bullets WOULD have matched the failing goal (= candidate new bullets).
python tune.py analyze v1_run
```

**`learner.py` — Reflector + Curator over traces.** Reads either
production traces (`games/traces/`) or a tune run's traces.

```bash
# One-line summary per past session.
python learner.py walk

# Full structured dump of one session (path to .jsonl, or session id substring).
python learner.py show games/traces/<slug>_<ts>.jsonl
python learner.py show <session-id>

# Reflect over traces and PRINT proposed deltas (no writes).
python learner.py reflect games/traces/

# Reflect AND apply (writes to playbook.jsonl).
python learner.py apply games/traces/
```

**Playbook file:** `games/memory/playbook.jsonl` (one bullet per line,
human-readable). Hand-edit it freely — the curator merges deterministically
on top, and the prompt only injects the top-K bullets relevant to the
current goal so adding a niche bullet won't crowd out the basics.

The seeded 30 bullets are tagged `source: "seed"`; learner additions are
tagged `source: "learned"`. Helpful/harmful counters live in the same
record so you can see at a glance which rules fire on passing vs failing
runs.

---

## Quick start

**One command, clean clone → working agent (macOS or Ubuntu Linux):**

```bash
git clone https://github.com/jmrothberg/Agent_learning.git
cd Agent_learning
./scripts/setup.sh
```

The setup script (idempotent — safe to re-run any time):

1. Verifies `python3 >= 3.10`.
2. Creates `.venv/` (or repairs a stale one if the repo was moved).
3. Installs `requirements.txt` (core: ollama, playwright, textual, pytest, …).
4. On **Apple Silicon**, installs `requirements-mlx.txt` (`mlx-lm`) unless `--no-mlx-tools`.
5. Runs `playwright install chromium` with `PLAYWRIGHT_BROWSERS_PATH` unset (normal OS cache).
6. Installs the full diffusion stack via **`./scripts/install_diffuser.sh`** (one invocation):
   torch, diffusers (git), transformers stack, then **`requirements-diffuser.txt`**
   (**soundfile**, **torchsde**) — sprites + Stable Audio **pip** deps together.
   (**HF weights** download on first `<assets>` / `<sounds>` or smoke scripts into the hub cache; interactive login is uncommon.)
7. Runs the pure-function pytest suite (~190 tests) as a sanity check.
8. Prints MLX / Ollama next-steps (plus optional HF recovery hints).

Useful flags:

```bash
./scripts/setup.sh --recreate-venv   # nuke .venv and start fresh
./scripts/setup.sh --no-mlx-tools    # Apple Silicon but Ollama-only — skip mlx-lm pip install
./scripts/setup.sh --skip-playwright # headless servers without a display (coder.py --headless)
./scripts/setup.sh --skip-tests
./scripts/setup.sh --no-gpu          # ONLY without CUDA/MPS — skips torch/diffusers (~5 GB)
./scripts/setup.sh -h
```

After setup, point the agent at a local LLM and start a session:

```bash
# Either: Ollama with at least one model loaded.
ollama serve &
ollama list
ollama run --ctx-size 32768 qwen3.6:35b   # warm at 32K to skip reload

# OR: MLX on Apple Silicon — loaded IN-PROCESS by the agent (no
# mlx_lm.server, no port). Point MLX_MODEL at a downloaded model, or
# place one under ~/MLX_Models/ so it's auto-discovered. First request
# of a session pays a ~30-60s cold-load cost; the weights then stay
# resident in this process's GPU VRAM for the rest of the session.
export MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8

# Run the agent:
.venv/bin/python chat.py                    # TUI (recommended)
.venv/bin/python coder.py "build snake"     # one-shot CLI
```

### Hugging Face login (only if downloads fail)

Weights for sprites and sounds normally download **automatically** into `~/.cache/huggingface/hub/` the first time you run `<assets>` / `<sounds>` or the smoke scripts below. **You often will not be prompted** for a Hugging Face password — e.g. you're already logged in from another project, or `HF_TOKEN` is set in your shell.

If Stable Audio (or another repo) returns **403** / **401**:

1. Sign in at Hugging Face, open [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0), and agree to terms if the page asks.
2. Create a read token at <https://huggingface.co/settings/tokens> if you need one.
3. In this repo's venv:
   ```bash
   .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
   # or: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
   ```

### Smoke-test sprites + sounds (do this right after setup)

```bash
.venv/bin/python scripts/_smoke_doom.py    # one PNG — exercises Z-Image cache
.venv/bin/python scripts/_smoke_audio.py   # three OGGs — exercises Stable Audio cache
```

If both complete, HF setup is done unless you later hit an auth error.

Z-Image-Turbo is a **public** HF model. Stable Audio may enforce agreement only when the hub requires it for your account.

---

## CLI (`coder.py`)

Same `GameAgent` as the TUI, without Textual. Useful for scripting, CI, or
machines without a display (`--headless`).

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--model` | backend detection | Override model id (Ollama tag or MLX path/HF id) |
| `--max-iters` | `6` | Cap plan/build iterations |
| `--out` | unique `games/<slug>_<timestamp>.html` | Output HTML path |
| `--best-of-n` | `1` | Sample N candidate fixes per failed iteration |
| `--num-ctx` | `262144` (env `CODING_BOX_NUM_CTX`) | Ollama context window. Matches the native context of Qwen3.6 / DeepSeek V4 / GLM 5.1 / MiniMax M2 so the harness never clips before the model does. Lower it (e.g. `--num-ctx 65536`) if you OOM — KV-cache scales linearly with ctx. Changing between calls forces an Ollama model reload — preload at this size first with `ollama run --ctx-size 262144 <model>`. |
| `--stall-seconds` | `90` | Per-chunk stream inactivity timeout |
| `--headless` | off | Run Chromium without a visible window |
| `--open` | off | Open the final HTML in the system browser |
| `--seed` | none | Existing `.html` to adapt (skips memory skeleton) |

Run `.venv/bin/python coder.py -h` for the full argument help text.

---

## What to type when (the only cheat sheet you need)

Once the TUI is running, the input box is the only place you interact with
the agent. Five things you'll do:

| What you want                                       | What to type                                          |
| --------------------------------------------------- | ----------------------------------------------------- |
| **First run** — describe the game                   | the description, then Enter (e.g. `snake with score`) |
| **Small change to what just shipped** (most common) | **just type it** — `make the bunkers green`           |
| **Ship as-is, stop iterating**                      | `done` / `looks good` / `ship` (or `Ctrl+D`)          |
| **Brand-new unrelated game**                        | `/new <goal>`                                         |
| **Start from an existing `.html`**                  | `/seed <path>` then `/new <goal>`                     |

That's it. After the model says `<done/>`, plain text auto-extends the *same*
game (skips planning, jumps straight to patching the existing file). You only
need `/new` when you're starting an unrelated game — small tweaks are just
typing.

`/help` inside the TUI shows the same cheat sheet plus the full command list.

### How to write a prompt that ships a playable game

These six rules are also printed at the top of every fresh TUI session
(see `chat.py` `on_mount`). They're the highest-leverage things you can
do to nudge a medium-skilled local model toward an actually-playable
result instead of a "passes the test but unplayable" demo:

1. **Be specific about controls + win/lose.** *"WASD to move, mouse to
   aim, space to shoot, lose at 0 HP, restart with R."* Vague goals →
   vague games — the model fills the gaps with whatever's easiest.
2. **Name what's on screen.** Enemies, projectiles, terrain types,
   pickups. The agent uses these names to populate `<assets>`
   automatically; if you don't name them, neither will the model.
3. **Ask for art directly.** Words like *"sprite art" / "pixel-art" /
   "cool graphics" / "great visuals"* trigger the Z-Image-Turbo
   pipeline via `_detect_art_intent`. Skip them for DOM-only apps
   (todo, calculator) where text + emojis suffice.
4. **Mark mixed graphics explicitly.** *"Sprites for X, but procedural
   for Y because Y gets destroyed brick-by-brick."* The new
   `ASSETS_FORMAT` guideline teaches this pattern; stating it in your
   prompt makes the model commit.
5. **For 3D, just say "3D" or "first-person".** `_detect_3d_intent`
   recognizes those plus *fps / raycaster / voxel / minecraftlike /
   wolfenstein* and switches the model to `three.js` via CDN. Don't
   ask for a raycaster from scratch unless you really mean it.
6. **Iterate via plain text after `<done/>`.** Mid-game tweaks like
   *"the gun is sideways, rotate 90°"* auto-extend the same file —
   you don't need `/new` for these. Use `/new <goal>` only when
   switching to an unrelated game.

The TUI also prints a `>> APPLIED to this turn: feedback: '...'` line
in the agent log every time your queued input lands in a prompt, so
you can watch each piece of feedback being consumed (see
[How feedback works](#how-feedback-works-this-is-the-important-bit)).

---

## What the TUI looks like

```
┌─ JMR's Coding Box — <model> ───────────────────────────────────────┐
│ ── planning ──                            │ Phase: extension 2/6   │
│ <plan>...</plan>                          │ Iteration: 4/6         │
│ ── iteration 1/6 ──                       │ Model: qwen3.6:35b     │
│ <html_file>...</html_file>                │ Goal: asteroids w/sound│
│ TEST FAILED (1 error, 0 issues)           │                        │
│ → applying your input to next turn: ...   │ Log: games/traces/...  │
│ ── extension 1/6 ──   ← continuation      │ Ctrl+L reprint         │
│ wrote games/asteroids_2026..._iter_04.html│                        │
│ TEST OK (0 errors, 0 issues)              │                        │
│ DONE — Model declared done after a clean  │                        │
│        run. Type to extend ▸ /help        │                        │
├────────────────────────────────────────────────────────────────────┤
│ > add sound on shoot and explosion        ▸ auto-extends the game  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Slash commands

Type at the input box at any time. The session continues even after the
model says `<done/>` — plain text auto-extends the existing game, and a
short ship phrase (`done`, `ok`, `looks good`, `lgtm`, `ship`, `perfect`,
`stop`, `finished`, …) ships immediately, same as `Ctrl+D`.

| Command              | What it does                                                                          |
| -------------------- | ------------------------------------------------------------------------------------- |
| `/help` (`/h`, `/?`) | print all commands                                                                    |
| `/list` (`/models`)  | numbered list of installed Ollama models, marks `*` for loaded                        |
| `/model <name\|N>`   | stage a model — **STICKY** across `/new`s · `/model` alone clears the staging         |
| `/seed <path>`       | stage a baseline `.html` to adapt — **STICKY** across `/new`s · `/seed` alone clears  |
| `/new <goal>`        | end current session (must be done first), start a fresh one — uses staged seed/model  |
| `/iters <N>`         | change `max_iters` for the next session/extension (also sticky)                       |
| `/reset`             | wipe ALL staged state at once (seed + model + iters → defaults)                       |
| `/wait [on\|off]`    | toggle step-mode: pause after each iter and wait for explicit input before continuing |
| `/ship`              | ship now (= Ctrl+D, or just type `done`/`looks good`/`ship`)                          |
| `/open`              | open the current `.html` in your default system browser                               |
| `/log` (`/paths`)    | print game / log / jsonl / conversation / snapshots / best paths                      |
| `/clear`             | clear the agent log pane (does not affect staged state)                               |
| `/status`            | print model, phase, iteration, paths, **what's currently staged**                     |
| `/quit`              | quit (= Ctrl+Q)                                                                       |

### Sticky staging — and how to actually start fresh

`/seed`, `/model`, and `/iters` are **sticky**. Once set, they apply to *every*
subsequent `/new` until you change them. This matches the most common workflow
("I want to iterate on this file with this model"):

```
/seed games/asteroids_20260503.html
/model qwen3.6:35b
/new add multiplayer       ← uses the seed + the model
/new add a boss enemy      ← still uses the same seed + model
/new high-score table      ← still uses them
```

To clear:

| To clear…                       | Type                          |
| ------------------------------- | ----------------------------- |
| just the staged seed            | `/seed`        (no argument)  |
| just the staged model           | `/model`       (no argument)  |
| reset iters to default (6)      | `/iters 6`                    |
| **everything staged at once**   | `/reset`                      |
| clear the log pane              | `/clear`                      |

`/reset` doesn't touch the running session, the browser, or anything on disk —
it just resets staging to defaults. Follow with `/new <goal>` to actually
start a fresh session.

### Small-model practical guide — getting the most out of a 27B local LLM

This project assumes you're running a sub-30B local model (Qwen3.6-27B,
DeepSeek-V4-flash, etc.). Frontier-tier models (Claude / GPT-5) tolerate
more abstract guidance; small models reward concrete framing. A handful
of pragmatic rules that make sessions land more often:

- **`/seed <path>` vs `/new <goal>`.** Use `/seed games/<basename>.best.html`
  + `/new <tweak goal>` when you want the model to *continue* an existing
  game. Use `/new <goal>` alone for a fresh build. `/seed` is sticky, so
  multiple `/new`s reuse the same baseline — good for "add boss enemy,
  then add high-score table." Pick `.best.html` (last passing iter) over
  `.html` (current working file, may be mid-fix). The seed mechanism
  inlines the file in the first build prompt — keep it under ~12 KB if
  you can; bigger seeds push iter-1 context past 30 KB for a 27B model.

- **`/wait on` while debugging.** Toggles step-mode: the agent pauses
  after each iter and waits for explicit input before continuing. Lets
  you inspect the screenshot, type corrective feedback, or hit Enter to
  let the next iter run. Indispensable when a session is converging
  slowly. Auto-arms on the first failing iter; toggle off with `/wait
  off` once you've stabilized.

- **How to read the test report.** Two independent gates:
  - `OK: True/False` — the harness's overall verdict. Driven by
    page-errors + soft-warnings + probe failures.
  - `Input test: PASS / FAIL` — a *separate* automated keypress test.
    For game-shaped goals (criteria mention arrows / WASD / keys), an
    `Input test: FAIL` synthesizes an `input_responsive` probe failure
    that flips `OK: False`. For pure-DOM apps (calculator, color
    picker), it's surfaced as a warning only.
  - **`soft_warnings` block `<done/>`** even when `errors` is empty.
    Anything in the `ISSUES (must fix):` block of the report counts.
  - The `<criteria>` you wrote in Phase A are echoed back at each fix
    turn so you can see what the model committed to verify.

- **Writing effective feedback.** Phrase it like you'd report a bug:
  - **Good**: *"player does not move left when I press ArrowLeft"* —
    behavior verb (`move`) + negation (`does not`) + specific control
    name. The behavior-bug detector ([agent.py](agent.py) `_feedback_is_behavior_bug`)
    suppresses the MEDIA-CHANGE DIRECTIVE so the model routes to code,
    not assets.
  - **Risky**: *"fix mario"* — mentions an asset name with no verb
    pair, can trigger MEDIA-CHANGE misrouting. Add a behavior verb +
    negation, or describe what shouldn't be happening.
  - **Avoid**: *"controls broken"* / *"doesn't work"* — too abstract.
    Small models can't translate this into a specific fix. Name the
    button, the expected behavior, what actually happens.
  - **Lock the code when changing only art/sound**: *"redraw the
    barrel sprite, don't change any code"* fires the `_feedback_locks_code`
    detector and suppresses the one-shot rewrite exemption.

- **When to expect /iters higher than 6.** The default is 6, sufficient
  for ~70% of one-shot small-model sessions. Complex games (DK,
  Pac-Man, Doom-style) want `/iters 10` or higher, plus `/restarts 2`
  so the agent can throw away a bad start and re-roll.

- **Don't chase one model on one game.** This project's standing rule:
  if the agent can't ship a working game with one model, switch
  models. Don't tune prompts or harness behavior for a single
  model — defaults stay generic. Use `/list` to see what's loaded, `/model <N>`
  to switch.

---

## Keys

| Key            | What it does                                             |
| -------------- | -------------------------------------------------------- |
| `Enter`        | submit text (goal / answer / feedback / slash command)   |
| `Ctrl+D`       | ship — agent finishes current turn and exits             |
| `Ctrl+L`       | re-print all log file paths for this session             |
| `Ctrl+Q`       | quit (browser is cleaned up automatically)               |
| `Shift`+drag   | select text in the log pane (bypasses TUI mouse capture) |
| `Ctrl+Shift+C` | copy selection (after `Shift`+drag)                      |

If your terminal stops echoing after exit, run `reset` and you're fine.

---

## How feedback works (this is the important bit)

You can type ANY text in the input box during a run. It is NOT sent to
the model immediately — it is queued and injected at the very next
user-turn boundary.

The TUI gives **five** independent signals so you always know where your
words are at every stage of the round-trip:

1. `> feedback: your text` in the **left agent log** — the moment you press Enter.
2. `✓ queued (pending: N)` in the **left agent log** — immediate ack that
   the input handler captured the text.
3. `Queued (N):` section in the **right status panel** — lists every
   pending message with a numbered preview. Updates instantly on type;
   each item disappears the moment the agent consumes it.
4. `>> APPLIED to this turn: feedback: '...'` in the **left agent log** —
   fires inside `_flush_user_injections` at the user-turn boundary, so
   you see exactly which messages reached the model in which turn.
   (The earlier `→ applying your input to next turn` line still fires
   from older code paths — both are confirmation, not duplicates.)
5. The model's next reply addresses your feedback explicitly.

Your feedback is wrapped in a loud banner at the top of the prompt:

```
================ USER FEEDBACK (HIGHEST PRIORITY) ================
The user just typed this while watching your game. It OVERRIDES
any plan or default behavior. Address it explicitly in this turn:
- your text here
==================================================================
```

**Pending feedback is never dropped.** If the model says `<confirm_done/>`
or you press `Ctrl+D` while feedback is pending, the agent applies the
feedback and continues for one more turn instead of exiting.

**After `<done/>`, plain text auto-extends.** The agent re-enters the
iteration loop in *continuation mode* — it skips planning + first-build,
loads the existing file, and treats your feedback as the next fix
prompt. No need to restart the TUI to add features.

### Changing one asset (or one sound) without touching the code

Sprites and sounds the agent generates can be **re-rendered in place**
mid-session — same name, new visual/audio prompt, no JS edit. The
trick is using language the agent recognizes as an art/sound change
plus an explicit code-lock so the rewrite gate stays closed.

**Recommended template:**

```
redraw the <name> asset as <new visual description>, no code changes
remake the <name> sound  as <new audio description>, no code changes
```

Working variants (any of these phrasings hit the detector):

- `redraw the player_ship asset as a pink heart with sparkles, only the asset`
- `regenerate the centipede_tail — rounder, two animated legs, just that one asset`
- `swap the mushroom sprite for a deep-red cap, taller, no code changes`
- `remake the laser sound — deeper, punchier 8-bit, only the audio`
- `redo the music track as a slow chiptune, just the sound`

**Two ingredients matter:**

1. **An art/sound verb** — `redraw`, `remake`, `regenerate`, `swap`,
   `replace`, `redo`, `redesign`, `update`, `change`. Paired with an
   art noun (`asset`, `sprite`, `image`, `art`, `png`) or a sound noun
   (`sound`, `audio`, `sfx`, `music`, `clip`, `track`).
2. **A code-lock phrase** — `no code changes`, `only the asset`,
   `just that one sprite`, `don't touch the code`, `without changing
   the code`. This both flags the asset-change intent AND suppresses
   the one-shot `<html_file>` rewrite exemption that fresh feedback
   would otherwise arm. Without it, the agent may decide to "improve"
   surrounding code too.

Naming the existing asset/sound (e.g. `player_ship`, `centipede_tail`)
isn't required — the agent also detects generic phrasings like *"redraw
the sprite"* — but it makes the directive land more precisely when you
have several assets.

**What happens under the hood:** [`agent._feedback_is_art_change` /
`_feedback_is_sound_change`][_detectors] fire, [`_feedback_locks_code`][_locks]
fires, and `_flush_user_injections` appends a `MEDIA-CHANGE DIRECTIVE`
block to the next user turn listing every existing asset/sound name
and steering the model toward `<assets>` / `<sounds>` re-render.
[`_maybe_generate_assets_and_sounds`][_regen] merges the new file into
the session dict, overwriting the PNG / OGG at the same path so the
`drawSprite()` / `new Audio()` call already in the HTML picks it up
with no code edit.

[_detectors]: agent.py
[_locks]:     agent.py
[_regen]:     agent.py

**Live-test evidence** (`scripts/live_test_asset_change.py`):

```
[backend] mlx model=DeepSeek-V4-Flash-mxfp8
[user]    redraw the player_ship asset as a small pink heart with
          white sparkles on a transparent background. Only the asset,
          no code changes.
[model]   <assets>[{"name":"player_ship","prompt":"small pink heart
          with white sparkles on transparent background, pixel art,
          64x64"}]</assets>
[result]  PNG pre  sha8=7b6b2fcc  17028 B
          PNG post sha8=25be7c2c  10183 B
          PNG re-rendered? True
          <patch> in reply?  False
          <html_file>?       False
          PASS
```

The stubbed sibling test (`scripts/demo_asset_change_feedback.py`) runs
the same flow with mocked diffuser in ~1 s and is wired into the test
suite — see `tests/test_asset_change_feedback.py`.

**Origin:** [`games/traces/centipede-game-with-super-nice_20260512_180020.*`](games/traces/)
captures the failure mode that drove this. The user typed
`only change the centipiede_tail no other asset or code, just that one
asset no changes to the code` and the model replied *"I can't generate
new image assets in this environment — I can only modify the HTML
file"*, then rewrote a `drawSprite()` call into procedural `ctx.*`
code → regression → auto-revert. The harness already supported
mid-session re-render; the prompt just didn't tell the model about it.

---

## Model selection

The TUI picks a model in this order:

1. `OLLAMA_MODEL` / `CHAT_OLLAMA_MODEL` env var — explicit override, always wins.
2. **Currently loaded in Ollama** (`/api/ps`), preferring the entry with the
   latest `expires_at`. Ollama bumps that TTL on every use, so the freshest
   one is the model you most recently ran. *No blacklist applied here* —
   if you have it loaded, it works for you.
3. First **installed** (`/api/tags`) skipping the broken-tag blacklist —
   only used when nothing is loaded yet.
4. Hard fallback in `coder.MODEL`.

To force a specific model:

```bash
OLLAMA_MODEL=gpt-oss:latest .venv/bin/python chat.py
```

…or, inside the TUI, `/list` then `/model <number>`.

### What the model blacklist is

`chat._KNOWN_BROKEN_TAGS` is a **stay-clear list** for auto-detection. If the
Ollama daemon on your machine returns 500 (or wedges) when asked to load
some particular tag, put that tag in the set and step 3 above will skip
over it instead of picking it as the "first installed". The list is **only
consulted in step 3**; steps 1, 2, and `/model` ignore it entirely — if
you explicitly choose a tag, you get that tag.

The set ships **empty** today. The previous default (`qwen3.6:27b/35b`) was
a stale workaround from before those models were healthy on this machine,
and leaving them in caused fresh launches to silently fall through to
`gpt-oss:latest`. If you ever discover a tag that crashes Ollama on load,
add it here and the resolver will route around it.

---

## File layout & where to look when something fails

Every session writes a unique meaningful basename: `<goal-slug>_<timestamp>`.
No artifact ever overwrites a previous session's.

```
Agent_learning/
├── chat.py              # Textual TUI (recommended entry point)
├── coder.py             # CLI agent (one-shot, no UI)
├── tune.py               # battery runner + diff/why/analyze + auto-learn
├── learner.py            # offline Reflector + Curator over traces
├── agent.py             # async event-driven agent core
├── tools.py             # Playwright browser harness + game heuristics + score_test_report
├── prompts.py           # v0 SYSTEM_PROMPT + per-phase instructions
├── prompts_v1.py        # v1 prompt: XML-structured, FormatSpec-assembled, criteria/probes/assets
├── memory.py            # skeletons + mistakes + Playbook (quality-ranked, deduped, budgeted)
├── patches.py           # SEARCH/REPLACE patch engine: fuzzy norm, uniqueness, repair layer
├── assets.py            # Z-Image-Turbo sprite generation (lazy import, content-hash cache)
├── ollama_io.py         # streaming watchdog + best-of-N sampler
├── tests/
│   ├── test_patches.py        # 24 tests — fuzzy match, uniqueness, overlap, repair
│   ├── test_compaction.py     # 9 tests — structured summary + 2-tier prune
│   ├── test_retrieval.py      # 20 tests — quality rank, two-stage, dedup, budget, hybrid
│   ├── test_microprobes.py    # 23 tests — pre-Chromium sanity + API allowlist
│   ├── test_lookup.py         # 8 tests — <lookup_bullet> tag → playbook body injection
│   ├── test_project_config.py # 7 tests — AGENTS.md / CLAUDE.md auto-loading
│   └── test_assets.py         # 22 tests — &lt;assets&gt; parsing, cache, sprite gen w/ stub
├── requirements.txt
└── games/
    ├── <slug>_<ts>.html              # the live game file
    ├── <slug>_<ts>.best.html         # last clean version (auto-saved)
    ├── snapshots/<slug>_<ts>/
    │   ├── iter_01.html              # every iteration's full HTML
    │   └── iter_01.png               # browser screenshot per iteration
    ├── traces/
    │   ├── <slug>_<ts>.log           # plain-text mirror of the TUI agent log
    │   ├── <slug>_<ts>.jsonl         # structured event stream (one JSON per line)
    │   └── <slug>_<ts>.conversation.md  # FULL message history sent to/from the model
    ├── memory/
    │   ├── skeletons/                # bundled + auto-promoted past wins
    │   ├── mistakes.jsonl            # error-signature → fix-summary log
    │   ├── playbook.jsonl            # ★ THE PLAYBOOK ★ — accumulated rules
    │   └── goals/<session-id>/       # outcome.json + best.html copy per session
    └── tune/
        ├── battery.jsonl             # 10 canonical goals
        └── run_<timestamp>/          # per-tune-run artifacts
            ├── manifest.json
            ├── SUMMARY.md
            ├── _memory/              # isolated playbook for this run
            └── <slug>/result.json
```

When a session goes wrong, the single most useful file to share for
debugging is `games/traces/<slug>_<ts>.log` — every model token, every
test report, every error with traceback, every UI event in order. For
the model's exact view of the conversation use `<slug>_<ts>.conversation.md`.

`Ctrl+L` (or `/log`) in the TUI prints all paths at once.

---

## How the loop works (in code)

Three phases, all driven by simple XML-style tags the model emits. See
the diagram above for the full picture; this is the implementation
shorthand:

1. **PHASE A · planning** (1 turn): model emits `<plan>...</plan>` only.
   With **`prompt_version=v1`**, the plan also carries **`<criteria>...</criteria>`**
   (machine-checkable acceptance checks).
2. **PHASE B · build / iterate** (up to `max_iters`): model emits
   `<patch>` blocks (preferred) or a full `<html_file>...</html_file>`.
   Harness runs it in real Chromium and reports back: console errors,
   page errors, canvas state, RAF firing, input listener count,
   frozen-canvas check, automated input smoke test, and (if the model
   is a VLM) attaches the latest screenshot for vision review.
   **v1** may emit **`<probes>...</probes>`** (JSON) — the harness executes them
   via `tools.LiveBrowser`. Failed iterations use **`score_test_report()`**
   for partial-credit scoring when ranking best-of-N candidates.
3. **PHASE C · self-critique** (1 turn): when the model says `<done/>`
   on a clean run, it gets one final pass to either send a fix or
   reply `<confirm_done/>`.

Adaptive temperature: 0.7 on first/clean turns (creative), 0.25 after a
failed test (precision for fixes). Last-known-good code is embedded
in fix prompts so the model patches concrete code instead of trying to
remember its own previous reply.

After `<done/>` the worker exits cleanly — *but* the TUI stays alive.
Type more text and the agent re-enters the loop in continuation mode
(see `agent.run(continuation=True)`): planning is skipped, the existing
file is the baseline, and your feedback is the first fix prompt.

**Best-of-N (fix iterations):** when `best_of_n > 1` on `GameAgent` (today:
pass `--best-of-n N` to `coder.py`; the TUI leaves the default `1`), the agent
samples multiple completions and scores each by testing temp HTML through the
session `LiveBrowser` (same visibility mode as the rest of the run — see the
diagram note above).

---

## Restarting / resuming after a crash

Every session is self-contained on disk. To resume:

1. **Find the latest trace.** `ls -t games/traces/*.log | head -1` → `cat`.
   Goal, conversation, failures, attempts — all there.
2. **Inspect the last working game.** `games/<slug>_<ts>.best.html` opens
   in any browser. The non-`best` `.html` is the last version on disk
   (which may be broken).
3. **Step through snapshots.** `games/snapshots/<slug>_<ts>/iter_NN.html`
   plus the matching `.png` show every step.
4. **Re-run the same goal.** Launch `chat.py` and either retype the goal
   or `/new <goal>`.

The README + the trace files are enough to hand off the project to a
fresh assistant: paste this README, paste the most recent `<slug>_<ts>.log`,
describe your next goal.

---

## Troubleshooting

| Symptom                           | Fix                                                                                    |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| `Could not launch Chromium`       | run `playwright install chromium`; ensure a display for non-headless runs (use `coder.py --headless` on servers) |
| Blank window / browser won't open | confirm you have a desktop session; try `echo $DISPLAY` (X11) or Wayland; SSH needs forwarding or headless CLI |
| Picks the wrong model             | `/list` then `/model <N>`, or `OLLAMA_MODEL=<tag> .venv/bin/python chat.py`           |
| Model picked is stale             | `/api/ps` is sorted by `expires_at` — touch your preferred model to bump it           |
| Ollama 500 on model load          | the tag is broken locally; pick another with `/list` + `/model <N>`                    |
| Terminal stops echoing after exit | `reset` (Ctrl+Q is the proper exit, not Ctrl+C)                                        |
| Can't select text in TUI          | hold `Shift` while click-dragging                                                      |
| Feedback ignored                  | check the agent log for `>> APPLIED to this turn:` (fires at every user-turn boundary) AND the right-pane `Queued (N):` panel emptying. If the panel never empties, the agent isn't draining; if the APPLIED line shows up but the model's reply ignores it, the model is choosing not to act. Also see `<slug>_<ts>.jsonl` for `feedback_queued` / `feedback_injected` events. |
| Feedback after done does nothing  | it should auto-extend; verify the bottom hint says "type feedback to extend"          |
| Typed feedback re-renders assets instead of fixing code | The MEDIA-CHANGE DIRECTIVE misroutes when an asset name (e.g. `mario`, `ladder`) appears in your text. Re-type the feedback with both a behavior verb (`move`, `climb`, `jump`, `fire`, `roll`) AND a negation (`does not`, `can't`, `won't`) — e.g. `"mario does not climb"` instead of `"fix mario climb"`. The behavior-bug detector then suppresses the directive. Look in the .jsonl for `media_change_directive_suppressed` to confirm. |
| Trivial-scope feedback got a full rewrite anyway | The model emitted `<html_file>` despite *"just swap X — no other changes"*. Phrasings that don't include the literal word "code" or a media noun used to slip the code-lock detector. Now broadened — *"this is trivial"*, *"no other changes"*, *"just swap/rename/move"*, *"leave the rest"*, *"nothing else"* all lock. If a rewrite still fires, check the .jsonl: `rewrite_exemption_armed` event should be ABSENT when your text matches one of those patterns. |
| Agent loops on the same bug 3+ iterations | At `_repeat_sig_streak >= 3` AND the `mistake_signature` matches a known subsystem (input wiring, draw/RAF loop, RAF kick-off), the harness now forces a `<question>`-only hard-gate turn. The model emits a question with three concrete options *(rewrite from scratch / try a specific approach / ship as-is)* and waits for your answer. Look for `stuck_hard_gate_armed` and `stuck_hard_gate_prompt_built` events in the .jsonl. |
| Session ended without `<done/>` or any signal | The iter loop used to fall out silently when the iter cap was reached on a failing build. Now an EXIT DECISION TURN fires forcing the model to either `<done/>` + `<notes>` (handoff summary) or `<question>` (specific blocker). Look for `exit_decision_turn_prompted` in the .jsonl. The final-iter test still runs against whatever ships. |
| Diagnose names X but patches target Y | Light-touch warning: when the harness has been reporting a specific subsystem (e.g. INPUT) for ≥1 iter AND your `<diagnose>` and patches both ignore it, a `COHERENCE NOTE` appears in the next user turn naming the identifiers you missed (`addEventListener`, `keydown`, etc.). The note doesn't block the patch — it just asks whether you intentionally addressed a different bug. Look for `diagnose_patch_coherence_mismatch` in the .jsonl. |
| Model emitted `<html_file>` as pseudocode skeleton (374 bytes, `// Asset loading` headers) | The new `_detect_skeleton_payload` check rejects this at materialize time so the prior real baseline isn't destroyed. Look for `<html_file> rejected: body looks like a pseudocode skeleton` in the test event. If this fires repeatedly, the model is at its working-memory ceiling — try a larger model or simplify the goal. |
| Stream takes 5-10 min on a 27B local model with no code emitted | Deliberation loop. The new tighter thresholds (4K outside `<think>`, 8K inside) abort earlier with `result.deliberated=true`. The agent's pending-coaching path then nudges the model to "commit to one root cause + one patch" on the next turn. |
| Can't tell if `/wait` step-mode is on or off | Look at the top of the status panel — explicit colored Mode row shows yellow `WAIT` when step-mode is on (pause per iter), green `AUTO` when off (continuous). Same dual-state badge in the bottom mode-bar. |
| Can't tell if active model can read screenshots | Mode row in the status panel shows `[VLM]` (magenta) or `[text-only]` (dim) once the agent's `_detect_vlm` probe completes (after the first stream call). `/list` shows the same badge for every installed model based on name patterns. VLM models help when you need visual feedback on what rendered (the `/vlm` critique path attaches screenshots to feedback turns). |
| Generated sprite doesn't match the rest of an animation chain | The status assets block now shows `[img2img←<parent>]` (cyan) for chained frames vs `[txt2img]` (dim) for plain text-prompt generation. `[txt2img (img2img fallback)]` (yellow) means the chain broke and the child rendered standalone — that's why it doesn't match its siblings. Re-emit the parent with a fresh prompt or change the chain's `strength` value. |
| Stream takes 10+ minutes, then aborts | `stream_done` event shows `looped=true` / `stalled=true`. A 27B model exhausted productive vocabulary on a complex fix and entered a token-repetition loop. The format-doctor escalates early on this signal (one strike, not two — see `format_doctor_early_escalation` in trace). You can interrupt with feedback or wait for the doctor to reformat. |
| Seed session: iter-1 report shows 4/5 probes failing | Phase A probes were authored WITHOUT seeing the seed file (the model can't see it until Phase B). Probes reference state names that don't exist in the seed. The iter-1 build prompt invites a corrected `<probes>` block; the harness adopts it on iter 2 via the widened reparse gate. **Trust iter 2's report, not iter 1's**, on seeded sessions. |
| Playbook / criteria never show up in traces | The agent runs `prompt_version="v1"` by default; if you've manually downgraded to a custom v0 or older module, the `<playbook>` block won't be rendered. Check `agent.py:710`, `chat.py:2772`, and `coder.py:189` — all default to v1. |

---

## Roadmap & known gaps

What's missing if you want this to be the *strongest* mid-size-LLM
coding agent. None of the rows below are in flight; this is an honest
checklist so you can decide what to PR.

> **Recently shipped (out of the roadmap):**
> - ✅ **AGENTS.md / CLAUDE.md project-config injection** — read at session start from cwd, appended as `<project-context>` (`agent._read_project_config`).
> - ✅ **API allowlist for Canvas / Audio / DOM** — receiver-name + method-allowlist scanner inside `run_micro_probes`; flags hallucinated `ctx.drawCircle`, `audioCtx.playSound`, etc., as warnings before the Chromium round-trip.
> - ✅ **Bullet-on-demand retrieval (skills pattern)** — `render_playbook_block(mode="hybrid")` ships top-3 with full body + the rest as ID-only index entries; the model emits `<lookup_bullet>id</lookup_bullet>` and the agent injects the body in the next turn.

| Gap                                          | Why it matters                                                                                                     | Effort | Notes                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Native tool-calling format (replace XML)** | Ollama supports JSON tool-calling for several models. Native-call output is more reliable than tag parsing on RLHF'd models, and we'd inherit Ollama's schema validation for free. | medium | We deferred this earlier as "real cost without measurement." Worth revisiting once a tune-battery baseline lands. |
| **Streaming patch validation**               | Today we validate patches *after* the stream finishes. Catching a malformed `<patch>` mid-stream lets us abort early and re-prompt before wasting tokens. | medium | Hook into `ollama_io.stream_chat`; partial-parse `<patch>` blocks as they close.                                                  |
| **Diff preview before patch apply**          | UX win: show the user the unified diff produced by `<patch>` before writing the file. Users could veto or hand-edit.                                  | low    | Mostly Textual layout; the patch engine already returns the spliced text.                                                         |
| **Real JS syntax check (`node --check`)**    | The pre-Chromium micro-probe uses bracket-balance heuristics. If `node` is on PATH, we could pipe each `<script>` through `node --check` for a real syntax verdict.                                             | low    | Already covered by the bracket-balance heuristic for the common cases; this is a quality-not-quantity upgrade.                    |
| **Sandboxed `file://` loading**              | Generated games run unrestricted in Chromium. The README warns about this; a real fix would route loads through a service worker or Origin Isolation policy that limits network access. | medium | Threat model is "agent generates bad code that exfils data" — currently mitigated only by user trust + local-model assumption.   |
| **Playbook auto-pruning by age**             | Bullets with low net score sit forever; the offline learner only adds. A periodic pruner that drops bullets with `harmful >> helpful` after N retrievals would keep the corpus tight. | low    | `learner.py` has the data; needs a `prune` subcommand and a freshness counter on each bullet.                                    |
| **CDN integrity hashes for `<script src=...>`** | Games sometimes pull from `cdn.example.com/phaser.js`; an SRI-style allowlist would reject typosquatted CDNs.                                          | low    | Maintain a small JSON of `{cdn_host: allowed_paths}`.                                                                             |
| **Cross-session user-preference memory**     | Beyond playbook (which is HTML/JS lessons): "user wants minimalist UI", "user prefers WASD over arrows", "this user always asks for sound" — all currently learned per-session and forgotten. | medium | Same shape as playbook; different namespace.                                                                                      |
| **Receiver-type inference for API allowlist** | Today the allowlist scanner only checks variables whose name matches a strict convention (`ctx`, `audioCtx`, …). A tiny type-inference pass (`getContext('2d')` → mark var as canvas2d) would catch hallucinations on unconventionally-named receivers too. | medium | Adds false-positive risk; gate with a tune-battery measurement first. |
| **Generalize the agent loop beyond HTML/JS** | Verticalization is what buys us the runtime-validation layer, but it also means we can't help with non-browser code (Python scripts, Rust kernels, etc.). | high   | Would essentially be a sibling agent sharing the patch engine + retrieval; non-trivial.   |

If you want the *single highest-ROI next step* per axis:

- **Token efficiency / safety on long sessions:** streaming patch validation.
- **UX for interactive use:** diff preview before patch apply.
- **Hardening:** sandboxed `file://` loading.

---

## Notes for future contributors (LLMs and humans)

A lot of this codebase grew out of evidence-driven trace diagnosis. If
you're an LLM picking this up to help improve the agent, read this section
first — it tells you the invariants, the standing rules, the patterns you
should follow, and the places where past edits have learned hard lessons.

### Standing rules (these have evidence behind them — don't break them silently)

1. **Model-agnostic.** No code path branches on a specific model name.
   The user rotates local models constantly (Qwen3.6, DeepSeek-V4-flash,
   the next thing next quarter). Tune the agent — prompts, retrieval,
   verifier signals, harness gates — never the model. The only model-
   shape signal we accept is a coarse class (`small` / `mid` / `large`)
   which the user sets explicitly via `/model-class`.
   - Memory rule: `~/.claude/projects/-Users-jonathanrothberg-Agent-learning/memory/feedback_model_agnostic.md`.

2. **Genre-free.** No code path mentions a specific game (asteroids,
   snake, DK, Doom) by name. The harness is open-domain HTML/JS;
   retrieval, probes, skeletons, hint tables stay genre-free. Modality
   keyword detectors (`_detect_art_intent`, `_detect_3d_intent`,
   `_SUBSYSTEM_HINTS`) describe rendering SHAPES — `input`, `draw`,
   `raf_start`, `assets` — not subject matter.

3. **Diagnose before propose.** When given a trace and asked "why did
   this fail?", the first response is evidence-based analysis quoting
   turn numbers, event kinds, byte counts. Don't pivot to fixes before
   the diagnosis lands.
   - Memory rule: `feedback_diagnose_before_propose.md`.

4. **Pace between chunks when non-auto.** When auto-mode is off, finish
   one item fully (code + tests + run pytest), then stop and check in.
   "First X, then Y" means stop between X and Y, even when both are
   pre-approved.

### Architecture at a glance

```
chat.py              Textual TUI (default entry point)
coder.py             headless CLI driver
agent.py             GameAgent — the core loop (planning, build, iterate,
                     critique). Most verifier-feedback logic lives here.
backend.py           LLM backends: Ollama, MLX (in-process), Anthropic,
                     OpenAI. `detect_backend()` auto-selects.
ollama_io.py         Shared streaming code path used by all backends —
                     stall watchdog, repetition detector, deliberation
                     guard.
tools.py             Chromium harness. `LiveBrowser.load_and_test`
                     runs the game and returns the test report.
                     Universal probes + game-control keyword detection.
patches.py           SEARCH/REPLACE patch engine. Char-preserving
                     fuzzy match cascade. Token-repetition rejection.
                     Format-failure classification.
prompts_v1.py        Per-format guidelines, pi-mono FormatSpec pattern.
                     `build_system_prompt(goal, model_class=...)`.
memory.py            Mistakes (signature-keyed) + playbook (compounding
                     rules-of-thumb). Quality-ranked, deduped retrieval.
assets.py            In-process Z-Image-Turbo sprite generator.
sounds.py            In-process Stable Audio Open sound generator.
learner.py           Offline Reflector + Curator over trace files.
tune.py              A/B battery for prompt/playbook changes.
research.py          Wikipedia-grounded planning context.
```

The agent loop is async + event-driven. `GameAgent.run(goal)` is an
`AsyncIterator[AgentEvent]`. Drivers consume the event stream. Three
phases: Phase A (planning, 1 turn), Phase B (build/iterate up to
`max_iters`), Phase C (self-critique on first clean `<done/>`).

### How to add a new verifier-feedback fix (the dominant pattern in this codebase)

Almost every fix in the "Verifier-feedback loops" section of this README
follows the same pattern. Future improvements will too. The recipe:

1. **Find the trace evidence.** Open the `.jsonl` file from a failing
   session in `games/traces/`. Look for the event kinds: `mistake_signature`,
   `stream_done` (with `looped`/`stalled`/`deliberated`/`crashed`/
   `max_tokens_hit`), `format_rejection`, `media_change_directive_*`,
   `probe_quality`, `coaching_injected`, `code_snapshot`, etc. Quote
   timestamps + turn numbers in your fix's comment.
2. **Write the helper as a module-level function with a thorough
   docstring.** Examples: `_subsystem_hint`, `_feedback_is_behavior_bug`,
   `_detect_skeleton_payload`, `classify_model_modality`. Each docstring
   cites the trace, names the failure mode, lists the defenses against
   false positives.
3. **Wire it in at the right site.** Most live in `agent.py` (per-iter
   loop, materialize path, fix-prompt assembly), `tools.py` (test report
   construction), or `patches.py` (apply-patches per-patch loop).
4. **Trace a structured event** for the new mechanism so it's observable
   in the `.jsonl`. Pattern: `self._trace({"kind": "your_event_kind",
   "field1": ..., ...})`. The .jsonl is the source of truth for
   post-mortem analysis; if an event isn't traced, future LLMs can't
   reason about it.
5. **Write pure-function tests.** Pin the literal trace data as a
   regression. Threshold constants get their own pin test. New tests
   go in `tests/test_<feature>.py`. The full suite must run in <2s.
6. **Document in the README.** Add an entry under "Verifier-feedback
   loops" or "Listening fixes" with the trace citation, the
   before/after behavior, and a one-line summary of the test coverage.
   Bump the test-count summary at the bottom of that section.

### Hint tables to extend (no architectural changes needed)

These are the most common "add new entries" tasks. Each is a single
data-only change — no logic refactor.

- **VLM model families:** `_VLM_NAME_SUBSTRINGS` in `backend.py`. Add
  case-insensitive substring patterns for the model name (e.g. a new
  Qwen vision variant). The `/list` UI auto-picks up the label.
- **Subsystem hints (mistake_signature → code-region pointer):**
  `_SUBSYSTEM_HINTS` in `agent.py`. Each entry is
  `(signature_substrings, name, identifier_tokens, fix_phrase)`. Used
  by Item 1a coaching + Item 1b focused-slice biasing.
- **Code-lock patterns (user feedback that means "minimal scope"):**
  `_CODE_LOCK_PATTERNS` in `agent.py`. New regex patterns add
  recognized phrasings without breaking existing ones.
- **Game-control keywords (criteria text → input-test should be hard
  signal):** `_GAME_CONTROL_KEYWORDS` in `tools.py`. New verbs / control
  surfaces (joystick, gamepad, etc.).
- **Behavior verbs (feedback shape → suppress MEDIA-CHANGE):**
  `_BEHAVIOR_VERBS` in `agent.py`. Gameplay verbs only — visual verbs
  like `look`, `appear`, `render` stay out so visual complaints route
  to the art-change path.
- **Format-failure classifiers (model emitted code in a shape the
  parser missed):** `classify_format_failure` in `patches.py`.
  Currently handles 6 shapes; add new ones as new failure modes
  appear in traces.

### Tests are documentation

Many regression tests pin the literal trace fixture that motivated
their fix. When you change behavior, update the test AND add a comment
explaining why the new behavior is correct. Examples to study:

- `test_subsystem_hint.py::test_literal_trace_20260514_175012_signature_matches_input`
- `test_feedback_code_lock.py::test_dk_trace_20260514_175012_user_text_locks_code`
- `test_skeleton_payload.py::test_dk_trace_374byte_skeleton_triggers_detector`
- `test_patch_replace_repetition.py::test_dk_else_brace_11x_triggers`

These tests are the closest thing to a regression contract. If a
"clever simplification" makes one of them fail, the simplification
probably re-introduces the bug it was meant to prevent.

### What NOT to do (lessons from past edits being reverted)

- **Don't propose fixes before doing the diagnosis.** The user calls
  this out; memory has the rule. When given a trace, the first
  response is evidence-based analysis.
- **Don't blast through pre-approved sequences in non-auto mode.**
  Pace between chunks; check in after each.
- **Don't add hidden behavior changes to existing helpers.** New
  behavior gets a new helper. Existing helpers are pinned by tests.
- **Don't tune for a specific model.** If a single trace shows a
  failure on Qwen but the same agent code works on DeepSeek, the fix
  should be model-agnostic — broaden a pattern, tighten a threshold,
  add a fallback. Never `if model_name == ...`.
- **Don't skip the trace citation in the code comment.** Future
  contributors (LLM or human) need to know which trace this fix
  came from so they can verify the regression test if the trace gets
  archived.

### Where the standing rules live

- `~/.claude/projects/-Users-jonathanrothberg-Agent-learning/memory/MEMORY.md`
  is the index. Each rule is a small markdown file. The agent reads
  this directory at session start and uses it to ground decisions.
- `CLAUDE.md` (in the repo root) is the operational summary the agent
  injects into the model's system prompt as `<project-context>`. Read
  it for the env-var matrix and common-commands quick reference.

---

## Dependencies

Three pip requirement files; `./scripts/setup.sh` installs them as follows:

| File | When |
|------|------|
| **`requirements.txt`** | Always — agent runtime (playwright **Python** package only; browser bundle is separate). |
| **`requirements-mlx.txt`** | **macOS arm64** by default (`mlx-lm` — used in-process by `backend.MLXBackend`; the `mlx_lm.server` CLI ships with the package but the agent does **not** call it). Skipped with `--no-mlx-tools`. |
| **`requirements-diffuser.txt`** | Installed automatically as the **last step inside** `./scripts/install_diffuser.sh` (`setup.sh` does not pip it separately). Skipped only with `--no-gpu`. |

**`requirements.txt`** — pure-Python deps the agent needs to run at all:

- `ollama` — Python client for the local Ollama daemon (also used by
  `learner.py reflect` / `apply` to call the Reflector; default model
  set in `learner.py` and overridable with `--model`)
- `playwright` — Python wrapper; **you still need** `playwright install chromium` (done by `setup.sh`)
- `textual` — the TUI framework
- `pillow` — used by `assets.py` for sprite resize/save
- `rich` — markup + escape helpers used by the TUI mirror layer
- `pytest` — test runner; setup.sh's verification step runs the suite

**`requirements-mlx.txt`** — `mlx-lm` on Apple Silicon. The agent
imports `mlx_lm.load` and `mlx_lm.stream_generate` directly from this
package and loads the model **in-process** (no separate server). See
[`backend.MLXBackend`](backend.py).

**`requirements-diffuser.txt`** — **`./scripts/install_diffuser.sh` pip-installs this file as its final step**
(so sprites + Stable Audio extras ship in one script when GPU stack is enabled). Powers BOTH sprite (`assets.py`) and sound (`sounds.py`)
generation:

- `transformers`, `accelerate`, `safetensors` — diffusers' transitive
  deps (also pulled by torch via `install_diffuser.sh`)
- `soundfile` — libsndfile bindings used by `sounds.py` to encode
  Stable Audio Open output as OGG-Vorbis
- `torchsde` — required by Stable Audio Open's default scheduler
  (`CosineDPMSolverMultistepScheduler`); we monkey-patch its
  `_Interval._split` to fix an infinite-recursion bug — see the
  [Generated sounds](#generated-sounds--stable-audio-open-no-server)
  section.

Torch + diffusers themselves are installed by `scripts/install_diffuser.sh`,
which ends by layering **`requirements-diffuser.txt`** (same script — cu130 nightly on Linux, MPS nightly wheel on macOS).

`tune.py` / `learner.py` have no extra pip dependencies.

---

## License

No `LICENSE` file ships in this repository yet. If you fork or redistribute
the code, add a license you are comfortable with (or confirm terms with the
repository owner on GitHub).
