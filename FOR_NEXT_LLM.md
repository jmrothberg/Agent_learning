# FOR_NEXT_LLM.md — Working on This Codebase as a Coding Agent

You are the next LLM picking up this project. The codebase drives a **medium-size local LLM** (qwen3.6:27b/35b, DeepSeek-V4, GLM-5.1, MiniMax-M2) to write single-file HTML5 games with a real Chromium browser as the verifier. The whole thing runs on the user's machine; the agent itself (the loop, prompts, harness, learner) is what we keep improving.

This document is the **brief** for that work. Read it before you propose changes.

---

## TL;DR — the four rules that bind every change

1. **The model is fixed. Tune the agent.** Don't propose "try a bigger model". Every improvement must be in prompts, retrieval, harness checks, scoring, slot scheduling, or the playbook. The user runs ~27-35 B params on local GPUs; the work is making *that* class of model ship working games.

2. **Genre-free or it doesn't ship.** No `if "chess" in goal:` style branches anywhere. Detection runs on **observable shape** — exposed `state.player.x`, canvas dimensions, image-noun + verb co-occurrence, recipe applicability gates. Adding a genre string anywhere reintroduces the listening bug Phase 0 was built to fix.

3. **GPU assignment is fixed on the 4-GPU workstation** — GPU 0 = diffusers (Z-Image-Turbo, SD-Turbo, Stable-Audio), GPUs 1/2/3 = three Ollama daemons (coder/critic/architect on ports 11434/11435/11436). Phase 1 parallelism gates on **observable independence** (different backend objects, different endpoints), so it falls back gracefully on single-GPU. Don't propose moving slots.

4. **Listening before speed.** Every "ship faster" idea has to clear "did the user's last three messages get honored?" first. Phase 0 is the listening layer; if it regresses, the agent feels broken regardless of how fast iter 1 runs.

---

## What this agent looks like end-to-end

```
chat.py / coder.py / tune.py   ← drivers (TUI, CLI, A/B rig)
        ↓
GameAgent.run(goal)            ← agent.py — async event-stream loop
        ↓
   Phase A (planning)          ← architect slot streams <plan>, <criteria>,
        ↓                        <probes>, optional <assets>, <sounds>.
                                 Phase 1B pre-warms diffusers in parallel
                                 on dedicated-GPU configs.
   <assets>, <sounds> gen      ← assets.py (Z-Image-Turbo), sounds.py
   on GPU 0 (lazy load)         (Stable Audio Open). Live `Sprites: N/M`
                                 in the status panel.
        ↓
   Phase B (iterate)            ← coder slot. Each iter:
        ↓                        1. stream → repair → extract_patches OR
                                    <html_file> → apply_patches
                                 2. run_micro_probes (no browser)
                                 3. LiveBrowser.load_and_test
                                    - canvas, listeners, console errors
                                    - input smoke test (state delta)
                                    - probe expressions in page
                                    - opening-book recipe checks
                                    - State-vs-render entity check
                                      (Phase 1.5.1)
                                 4. score_test_report → 0-100
                                 5. visual critic (non-blocking on
                                    independent slot, Phase 1A)
                                 6. autonomous playtest (Phase 1.5)
        ↓
   Phase C (self-critique)     ← <confirm_done/> or one more <patch>
        ↓
   restart-N + outcome record  ← write best.html + trace + playbook
                                 counter updates
```

Three drivers:
- **`chat.py`** — Textual TUI (default; visible Chromium beside the terminal). Mid-stream feedback queue, slash commands, live throughput rates.
- **`coder.py`** — headless CLI for unattended runs. Same agent.
- **`tune.py`** — A/B battery comparing prompt versions / feature flags across many goals. The alignment check.

---

## The standing rules — etched into memory at `~/.claude/projects/-home-jonathan-Agent-learning/memory/`

Read the `MEMORY.md` index before touching code. Each rule was learned the hard way. Highest-priority ones for any code change:

| Rule | What it forbids |
|---|---|
| [feedback-media-requests-never-defer] | Letting blocker-first deferral swallow art/sound asks. Media runs on GPU 0, code runs on slot 1. Independent paths. |
| [multi-frame-intent-must-be-honored] | Silently emitting "idle only" when the goal asks for walk/attack/idle frames. The planner inverts to `from_image` chains. |
| [descriptive-verbs-are-art-change] | Treating "the sprites should look like X" as not-art-change. Verb set includes `look`, `want`, `need`, `should`. Fuzzy entity stems too. |
| [autonomous-mode-no-genre-logic] | Playtest recipes that gate on subject matter. Gates must be observable structure only. |
| [state-vs-render-gap] | Probes that test `state.X !== undefined` are blind to whether the entity is drawn. Always-on harness check samples canvas pixels. |
| [phase1-keeps-gpu-assignment-and-works-on-one-gpu] | Parallelism features that hardcode GPU assignments or refuse to run on single-GPU. Gate on observable independence. |
| [dont-burn-gpu-on-known-wrong-assets] | Generating mid-session `<assets>` while a style-rebrand is queued. Defer + coach. |
| [logs-for-llm-readers] | Logs written for humans only. The reflector + future you read the trace; structure for that audience. |
| [status-panel-live-rates] | Silent "agent is working" — there's always a live rate to show (tok/s, sprites/s, ETA). |
| [measure-before-changing] | Proposing big architectural changes from guessed numbers. Read recent traces first. |

---

## What changes well in this codebase, and what doesn't

### Changes that historically pay off

1. **Better harness signals.** A new probe-quality lint, an always-on render check, a structured trace event for an existing failure shape. These compound: every future session reads the new signal too. The Pac-Man-without-a-Pac-Man fix (Phase 1.5.1) is exactly this shape.

2. **Listening fixes.** Anything that prevents the agent from swallowing what the user typed. Every iter the user can intervene is an iter that can recover. Phase 0.1 / 0.2 / 0.11 / 0.12 / 0.13 are all in this bucket.

3. **Generalizable playbook bullets.** The Reflector / Curator in `learner.py` turns *trace patterns* into bullets that get retrieved on future goals. A bullet that says "for any canvas game, expose `window.state.player.x` so the input smoke test detects movement" pays you back forever. A bullet that says "in chess games…" doesn't.

4. **Visibility.** Status panel rows, structured trace events with reasons, surprise categories. The agent gets better when *you can see why a session went sideways without re-running it*.

5. **Parallelism on truly independent hardware** — best-of-N fan-out across slots, non-blocking critic, diffuser pre-warm. Each gated on observable independence, each falls back cleanly on single-GPU.

### Changes that historically regress things

1. **Hardcoding genre logic.** Always tempting when you're staring at a single trace. Always reintroduces the listening bug.

2. **Tightening token / time abort thresholds without trace evidence.** The 27-35 B model emits long legitimate streams. Premature cutoffs trade a small win on one trace for a big regression on rich first-build streams.

3. **Adding always-on system-prompt rules.** Prompts past ~3-5 KB hurt mid-tier models. Use `FormatSpec.guidelines` (deduped per enabled format) or a goal-keyword detector that makes the rule conditional.

4. **Refactoring the per-spec asset loop.** It handles caching, chroma-keying, library admission, fuzzy parent lookup, multiple fallbacks. Risk-to-win ratio is poor unless there's a specific bug to fix. (Phase 2C `generate_batch` shipped as available infrastructure for exactly this reason — wire-in is deferred.)

5. **Cross-slot KV cache "fixes" that don't account for separate daemons.** Each Ollama daemon has its own KV cache. The fix is `warm_prefix` on the target slot, not a clever cache-sharing scheme (impossible across daemons).

---

## The verification harness is the lever — understand it first

Most session failures look like agent failures but are actually **verifier failures**. The harness gates `report["ok"]`; if a broken game can pass the gates, every downstream improvement is wasted. The biggest wins this project has shipped are gate fixes:

- **`window.gameState` vs `window.state` bug** (2026-05-16, before this conversation): the input smoke test was reading the wrong global for the entire history of the code. Silently fell back to canvas-hash, which is degenerate for any auto-animating game. Fixed in `tools.py` — now the gate detects actual input response.
- **State-vs-render gap** (2026-05-22, Phase 1.5.1): probes that check `state.pacman.x !== undefined` pass on a game where the player isn't drawn. New always-on check samples canvas pixels at exposed entity positions.

When you find yourself adding agent-loop machinery to handle a failure pattern, **first** ask: is the gate detecting this correctly? If the gate flips `ok=False` reliably, the existing fix-mode loop is usually sufficient.

---

## The mid-tier-model constraint changes what works

Frontier models tolerate verbose, multi-objective prompts. Qwen3.6:27b does not. Patterns that work here:

1. **Imperative, concrete, with examples.** "Emit `<assets>` with N entries. Example shape: `[{name: 'X_idle', prompt: '...'}, {name: 'X_walk', from_image: 'X_idle', strength: 0.40, prompt: '...'}]`." Not: "the user explicitly asked for animation frames so consider whether you should use img2img chains".

2. **One directive per turn, not a stack.** When MEDIA-CHANGE, SCOPED-CHANGE, and STUCK-LOOP-FAST-PATH all fire on the same turn, the 27B model concatenates two drafts. The router suppresses conflicting directives explicitly (see `_flush_user_injections` and the `media_change_directive_suppressed` reasons).

3. **Place the user's exact words in the prompt.** `[USER NOTE]\n{user text}\n[/USER NOTE]` survives compaction; "the user wants X" rephrased by the agent gets compacted away.

4. **Surface the matched keywords back to the model.** `(matched: 'idle', 'walkstride', 'smash')` in the SCOPE-PACING NUDGE tells the model what triggered the rule. Without that, the model treats the rule as opaque.

5. **Tight prompts beat smart prompts.** Phase 1.5's self-feedback paragraph is *deterministically* built from `recipe.finding_label + evidence` — no extra LLM call. Adding a critic call to "synthesize" it would cost a second slot stream for marginal wording improvement.

---

## How to debug a session that went badly (the 5-minute version)

1. **`grep '"kind":"iter_summary"' games/traces/<session>/trace.jsonl`** — one line per iter, contains `fail_reason`. Tells you the first iter that regressed.

2. **`grep '"kind":"surprise"' games/traces/<session>/trace.jsonl`** — pre-flagged failure classes (`state_vs_render_gap`, `regression_after_clean_iter`, `non_slot1_bon_winner`). Always read these first.

3. **`grep '"kind":"autonomous_playtest_skipped"' .../trace.jsonl`** — if the autonomous loop never ran, the reason field tells you why: `disabled:feedback_off`, `iter_failed`, `no_browser`, `no_behavior_recipes`, etc.

4. **`grep '"kind":"slow_prefill"' .../trace.jsonl`** — flags cross-slot KV reprefill stalls. If present and `warm_prefix` didn't fire, the bug is in the slot-handoff timing.

5. **Open `.conversation.md`** — the user's actual typed words, in order, with the agent's replies. The classifier truth is here. If the user said the same thing three times, the agent's intent detection failed.

6. **`grep '"kind":"runaway_stream_warning"'`** — a stream past 15k completion tokens. The model went off the rails. Usually a sign of `<html_file>` rewrite when `<patch>` should have been used.

---

## What's open and worth picking up

(These are honest deferred items, not "future work" handwaving.)

1. **Phase 2C wire-in.** `ZImageTurboGenerator.generate_batch(prompts)` is shipped and tested. Wire it into `generate_assets` for txt2img-only specs grouped by size. Win: ~10 s on 12-asset batches. Risk: the per-spec cache/chroma-key/library flow has side effects that need careful preservation.

2. **Phase 2B alternate-strategy prompts.** Best-of-N today fans out across slots at different temperatures. Adding alternate fix-strategy prompts ("minimal `<patch>`" vs "subsystem refactor") inside `_fan_out_best_of_n_across_slots` is a single edit. Defer until trace data shows temperature alone isn't diverse enough.

3. **Async autonomous playtest on slot 3.** The Phase 1.5 loop runs sequentially after the critic. Once Phase 1A's non-blocking critic stabilises, the playtest could move to a parallel slot 3 task. Worth it on 3-slot configs.

4. **More playtest recipes.** Three shipped: `entity-progress-over-time`, `input-axis-matches-facing`, `held-key-stays-in-bounds`. The spec listed four more (`held-key-changes-state-twice`, `released-key-stops-action`, `ai-opponent-makes-a-move`, `game-over-actually-ends`). Each is a one-recipe-entry + one `check_kind` case in `_evaluate_behavior_playtest_check`. Recipes must stay genre-free.

5. **Cross-session embeddings retrieval for the playbook.** Currently Jaccard-weighted. Vector retrieval (sentence-transformers, fully local) would catch semantic matches Jaccard misses. The bullet schema is stable; this is a swap inside `memory.Playbook.retrieve`.

6. **Compaction reads its own structured trace.** `_build_structured_summary` is deterministic. A learner pass could spot patterns in compacted state across sessions ("agent always compresses asset paths and then re-emits them — surface them in the anchor by default").

---

## When in doubt — read these files in order

1. **`CLAUDE.md`** — the operational summary the user wrote. Loaded into every session by `agent._read_project_config`.
2. **`README.md`** — the deep walkthrough. Has a `Major future improvements` section with the project's own backlog.
3. **`~/.claude/projects/-home-jonathan-Agent-learning/memory/MEMORY.md`** — the index of standing rules. Every entry is a hard-learned constraint.
4. **`HARNESS_DEBUG.md`** — what the verifier does AND doesn't catch. The blunt version.
5. **Recent traces in `games/traces/`** — the empirical truth. If your proposed change doesn't relate to a real trace pattern, you're probably making the prompt longer for no payoff.

---

## One last thing

This codebase has been shaped by hundreds of trace iterations against a small set of physical games. The "right" answer for an agent doing chess looks identical to the "right" answer for an agent doing Pac-Man looks identical to the answer for a future agent doing a roguelike — because **none of the rules above mention chess, Pac-Man, or roguelikes by name**. Keep it that way and the code stays general; break that and you're back to fixing one game at a time forever.

The user's standing principle: *"never make changes for one game, anything helping a type of game should be in our root level memory, but needs to be very general."* That's the bar. Hold it.

— Claude Opus 4.7, 2026-05-22
