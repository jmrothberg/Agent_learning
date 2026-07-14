# HARNESS_DEBUG.md — when a build goes wrong

This file is for **you** when a game session looks broken, stuck, or “TEST OK” but plays badly.

The browser test **finds problems and tells the model what to fix**. It does not write the game
for you. If the same bug keeps happening after many tries, the problem is usually one of the buckets
below — not “we need one more automatic check.”

More tuning traps: **`HARNESS_TUNING.md`**. Commands and env vars: **`DEV.md`**. Trace paths: **`AGENTS.md` §2**. Batch runs / pytest: **`eval/OPERATIONS.md`**. Test map: **`TEST.md`**.

**New agent?** Start with **`HARNESS_TUNING.md` § “New agent — harness improvement”** before editing code.

---

## Rule #1: play the game yourself

The agent can report **TEST OK** while the game is still wrong — for example the code says
`state.player.x` changed but the sprite never moved, or art loaded but never got drawn.

**Always:** open the `.html`, play it, watch Chromium if the TUI is running, look at screenshots
under the session folder. Do not trust green test text alone.

### Game looks fine but the trace says `ok=False`

Common on art-heavy builds after user feedback fixed visuals:

| What you see | What the harness still flags | Usually means |
|--------------|------------------------------|---------------|
| Sprites correct, fun to play | 1–2 **`soft_warnings`** on last iter | Probe timing (`punch_lands_damage`), partial patch apply, or `input_responsive` in headless test — not “art still wrong” |
| Sprites correct | `failure_class: memory_gap` + “assets undrawn” | **Stale triage label** — check whether `ASSETS_LOADED_BUT_UNDRAWN` is in `soft_warnings` or demoted to advisory in `report["warnings"]` |
| Playable in browser | `input_responsive` failed | Keys registered but no pixel delta in 3s smoke — often `_assetsReady` gating or closure-scope `keys` bug in **game code**; fix via memory/prompt, not always harness |

Read **`iter_summary.soft_warnings`** line by line. **`ok`** is false if **any** soft warning exists,
even when probes are 7/8 and the game looks perfect manually.

**Harness vs memory:** wrong sprite *resolver* for all parallel-prefix games → **`assets.py`**.
Wrong *wiring in one session* after feedback → often **LLM patch** + playbook (`versus-fighter-sprite-prefix`).
See **`HARNESS_TUNING.md` § harness vs memory**.

---

## What the browser checks (plain English)

These live in `tools.py`. When one fires hard, the iteration **fails** and the model gets a fix
prompt.

| Name | What it means |
|------|----------------|
| **PLAYER-STUCK** | You pressed a move key but the player did not move (often spawned inside a wall). |
| **ACTION_DRAWN_NOT_SPRITED** | Attack key did something on screen, but by drawing shapes/lines — not by showing the kick/punch sprite. |
| **CODE_DRAWN_OVER_SPRITE** | The right sprite showed, but the model also drew extra lines/flashes on top (fake “motion”). |
| **ASSETS_LOADED_BUT_UNDRAWN** | PNG files exist on disk but the game never calls `drawImage` (wrong asset name in code). **Advisory only** when probes pass and undrawn stems look state-gated, pose-only, scene-indexed off-screen backgrounds, or decode-timing false positives — see `tools.py` demotion paths. |
| **PROCEDURAL_REGRESSION_SUSPECTED** | Game declared lots of sprites but the screen is mostly colored rectangles. |
| **ENTITY-NOT-RENDERED** | Game state says an enemy/player exists at x/y but nothing was drawn there. |
| **STATIC-ACTION** | You pressed attack but the character stayed in one frozen pose while other things animated. |
| **Frozen canvas** | Screen pixels never change between frames (and input did not explain it). |
| **Micro-probes** | Cheap checks *before* opening the browser: broken/truncated HTML, unbalanced `{` `}`. |
| **Model probes** | Checks the model wrote at plan time (“does `state.player` exist?”, “did score go up?”). |

**Warnings only (do not fail the build):** ugly/dead animation frames that look like copies of idle;
pointer-lock noise from the test environment.

Action screenshots (one per key tested) save next to the trace as `iter_01_action_ArrowRight.png`,
etc.

---

## Vision review (`/vlm-critique`)

If enabled and you have a **vision-capable** model loaded, a second pass looks at screenshots and
answers yes/no questions (“Is the fighter facing left?”, “Is a projectile visible?”). Text-only
models skip this. Details: **`HARNESS_TUNING.md`**.

---

## Reading a session log (the `.jsonl` trace)

Each run writes a trace file under `games/…/traces/`. You normally **do not** read the raw file —
use the summary script:

```bash
.venv/bin/python scripts/enrich_trace.py <session-id> --timeline
```

Replace `<session-id>` with a folder name for TUI traces under `games/traces/`, or pass a **full path**
for tune batch traces (`games/tune_serial10/run_XX/traces/...jsonl`). See **`AGENTS.md` §2**.

That prints **one line per iteration**, roughly:

- Did the test pass?
- How many probes passed?
- Did patches apply? (e.g. `3/3` = three patch blocks, all applied)
- How fast was the model? (tokens per second)
- **What kind of failure was it?** (see table below)
- What blocked shipping?

If the session died mid-run (TUI shows “Agent crashed”), the timeline also prints an
**AGENT CRASH** banner. Grep the raw `.jsonl` for `"kind": "agent_crash"` — that row
includes `err`, `exc_type`, `iteration`, and a capped `traceback` so an LLM can debug
without the TUI log.

**Seed-edit eval runs** (`eval/eval_seed_edits.py`) turn the browser off on purpose. Those lines
show `test_skipped:no_browser` — that is expected, not a crash.

---

## Best-of-N and session artifacts

Three different “candidate” concepts show up in traces — do not confuse them:

| Name | What it is |
|------|------------|
| **`--best-of-n N`** (CLI) | Every fix turn samples N completions and picks the best Chromium score. Default **1** (single sample). |
| **Stuck best-of-2** | After 2+ failed iters, auto-escalate to 2 samples for one turn (cap 2/session). Default **OFF**. TUI: `/bestof on` · CLI: `--stuck-bon`. |
| **`top_candidates` in trace** | Memory/visual-playtest recipe matching — unrelated to BoN sampling. |

When BoN runs, candidate HTML is saved **visibly** next to the game (not dotfiles):

```
games/<session>/
  game.html
  game.best.html
  game_assets/
  candidates/
    iter_05/
      cand_0.html
      cand_1.html
```

Open `cand_0.html` in Chrome the same way as the main `.html` — same folder as `_assets/` so paths resolve.

**Trace vs batch monitor:** the agent writes `games/…/traces/*.jsonl` (full session log — use
`scripts/enrich_trace.py --timeline`). Optional `agent_monitor.json` from
`eval/tune_overnight_monitor.py` is a batch dashboard only — not required for debugging.

---

## Failure types — where to fix things

When something goes wrong, the trace tags a **failure type**. Use it to decide *who* should fix it:

| Tag | Plain meaning | You fix it by… |
|-----|----------------|----------------|
| **harness_bug** | The test harness or agent logic was wrong — it blocked or mis-coached a turn that should have been fine. | `tools.py` gates · `agent_*.py` mixins — see **`AGENTS.md` §1b** |
| **memory_gap** | Code saved fine, but the model made a mistake your curated hints should have prevented (e.g. art on disk but never drawn on an art-heavy game). | Adding a line to **`memory/playbook.jsonl`** or related `memory/*.jsonl`. |
| **local_llm_limit** | The local model stalled, looped, went silent, or emitted garbage — the harness did not contradict it. | Prompt/format tweaks, or accept model limits; not a browser bug. |
| **none** | Nothing special to classify. | Normal iteration friction. |

If the model never saved code that turn, look for **`no_usable_code`** instead of **`iter_summary`**
— same failure tags appear there.

---

## If you need to dig deeper (5 searches in the trace file)

After the timeline, open the `.jsonl` and search for:

1. **`agent_crash`** — session killed by an uncaught Python error (NameError, etc.); includes
   `traceback`. Distinct from a browser **`harness_crash`** (Playwright/test layer only).
2. **`iter_summary`** or **`no_usable_code`** — read `failure_class` and any `soft_warnings` text.
3. **`structured_compaction`** — fired too early? Context window may be too small; patches “don’t
   stick” because the model lost sight of the file.
4. **`playbook_injected`** / **`opening_book_retrieved`** — did the memory library actually load?
5. **`visual_critic`** — did the vision review run and parse answers?
6. **`patch_outcome`** — patch could not find its SEARCH block (often after compaction ate the
   file context).

Then **play the game again** and compare to what the log claimed.

---

## When nothing improves after many iterations

1. The **base model** may not be able to one-shot this game — that is a model limit, not always a
   harness bug.
2. **Memory** may not match your goal words (playbook did not retrieve the right bullet).
3. **Vision review** may be off, or the loaded model cannot see images.
4. The **same model** is both coding and reviewing — on one GPU there is no separate “second
   opinion” unless you stage a VLM on `/model2`.

---

## Files that matter

| File | Role |
|------|------|
| `tools.py` | Browser load, input test, gates |
| `agent.py` + mixins | Loop orchestration — map in **`AGENTS.md` §1b** |
| `modality.py` | Genre-free 3D / wireframe / FPS-nav shape detectors (planner + memory) |
| `assets.py` | Sprite generation and **injected** `sprite()` / `loadAssets` block (`render_asset_paths_block`) |
| `memory/*.jsonl` | Curated hints the model reads each run |
| `prompts_v1.py` | System prompt templates |

Quick regression (no GPU): `.venv/bin/python -m pytest tests/ -q`
