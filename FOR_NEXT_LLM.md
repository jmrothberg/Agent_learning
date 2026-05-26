# FOR_NEXT_LLM.md — Coding Box (the agent harness)

You are the next LLM picking up this project. The codebase drives a
**medium-size local LLM** (qwen3.6:27b/35b, DeepSeek-V4, GLM-5.1,
MiniMax-M2) to write single-file HTML5 games with a real Chromium
browser as the verifier. The whole thing runs on the user's machine;
the agent itself (the loop, prompts, harness, memory layers) is what
we keep improving.

This document is the **brief** for that work. Read it before you
propose changes.

> **Push target:** Harness / overlay work → **`git push origin main`**
> on `https://github.com/jmrothberg/Agent_learning_overlay` (local
> `origin`). **`upstream`** (`jmrothberg/Agent_learning`) is the program
> line — do not push overlay-only commits there unless the user explicitly
> retargets. See `CLAUDE.md` repo identity and README remote URLs.

> **For changes since the 2026-05-22 byline** — see "Recent ships" at
> the bottom. The mental model worth internalizing up front: the four
> separate "feedback" flows (machine bug feedback / playbook /
> autonomous playtest / directive wrapping). Sections below cover it.
>
> **For the highest-impact 2026-05-25 lesson** — read rule **5** in
> the TL;DR below before you propose any change to a stream-abort
> threshold. "Length is not a fail signal once the stream is
> producing" was learned the hard way. The DeliberationDetector
> story under "Recent ships" is the case study.

---

## TL;DR — the five rules that bind every change

1. **The model is fixed. Tune the agent.** Don't propose "try a bigger model". Every improvement must be in prompts, retrieval, harness checks, scoring, slot scheduling, or the playbook. The user runs ~27-35 B params on local GPUs; the work is making *that* class of model ship working games.

2. **Genre-free or it doesn't ship.** No `if "chess" in goal:` style branches anywhere. Detection runs on **observable shape** — exposed `state.player.x`, canvas dimensions, image-noun + verb co-occurrence, recipe applicability gates. Adding a genre string anywhere reintroduces the listening bug Phase 0 was built to fix.

3. **Multi-slot is opt-in.** GPU 0 = diffusers; GPUs 1/2/3 = Ollama daemons (coder/critic/architect on ports 11434/11435/11436) **only when explicitly staged** via `/model2` / `/model3` / `/modelall`. Default-on autopin was reverted 2026-05-23 — it crashed the workstation when combined with iter-1 best-of-N fan-out. Parallelism features gate on observable independence (different backends, different endpoints) so they fall back cleanly on single-GPU. Memory: `four-gpu-workstation-topology`.

4. **Listening before speed.** Every "ship faster" idea has to clear "did the user's last three messages get honored?" first. Phase 0 is the listening layer; if it regresses, the agent feels broken regardless of how fast iter 1 runs.

5. **Length is not a fail signal once the stream is producing.** This is the rule the 2026-05-25 Wolfenstein round-3 fix locked in after a painful regression. The DeliberationDetector existed to catch *pre-tag rambling* (model deliberates 6000+ chars without ever starting output), not to kill a working 20+ KB first-build stream. If you ever find yourself wanting to **bump a stream-abort threshold from X to Y** to make a long but legitimate stream survive, **STOP** — that is the wrong shape. The right fix is to *latch on observable code-emission* (`<html_file>`, `<patch>`, `<!DOCTYPE`, `<script`, `<canvas`, `function foo(`, lowercase `let/const/var foo =`) so length stops mattering once real output has begun. See `ollama_io.py:_HTML_OPENER_RE` / `_JS_OPENER_RE` and the round-3 detector tests in `test_wolfenstein_stuck_loop_fixes.py::test_deliberation_detector_*`. The user's exact pushback worth re-reading: *"if it takes 20k to get first pass how is threshold to 12000 fucking helping! it should only stop if there is an issue! not because its a long solution!"*

---

## What this agent looks like end-to-end

```
chat.py / coder.py             ← drivers (TUI, CLI)
        ↓
GameAgent.run(goal)            ← agent.py — async event-stream loop
        ↓
   Phase A (planning)          ← architect slot streams <plan>, <criteria>,
        ↓                        <probes>, optional <assets>, <sounds>.
                                 Diffusers pre-warm in parallel on
                                 dedicated-GPU configs.
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
                                 4. score_test_report → 0-100
                                 5. visual critic (non-blocking when
                                    on independent slot)
                                 6. autonomous playtest (after clean
                                    iters)
        ↓
   Phase C (self-critique)     ← <confirm_done/> or one more <patch>
        ↓
   restart-N + outcome record  ← write best.html + trace + playbook
                                 counter updates
```

Two drivers:
- **`chat.py`** — Textual TUI (default; visible Chromium beside the terminal). Mid-stream feedback queue, slash commands, live throughput rates.
- **`coder.py`** — headless CLI for unattended runs. Same agent.

---

## The standing rules — etched into memory at `~/.claude/projects/-home-jonathan-Agent-learning/memory/`

Read the `MEMORY.md` index before touching code. Each rule was learned the hard way. Highest-priority ones for any code change:

| Rule | What it forbids |
|---|---|
| [agent-must-beat-zero-shot] | **The North Star (2026-05-23).** The agent must improve on raw-LLM-plus-user-feedback. If a classifier / regex / directive ever OVERRIDES the user's typed words, the change is wrong — even if it "fixes" a specific trace. The accumulated regex patches that wrap user feedback caused the Doom misroute incident; `/rawfeedback` defaults ON now and is the structural answer. Before adding ANY classifier patch, ask: "does this help the model do something it wouldn't have done with the raw text? Or am I just patching the LAST regex's misfire?" |
| [repo-main-is-upstream] | Pushing harness-only work to `upstream` (`Agent_learning`) and overwriting the program line. Default push is `origin` → `Agent_learning_overlay`. |
| [feedback-media-requests-never-defer] | Letting blocker-first deferral swallow art/sound asks. Media runs on GPU 0, code runs on slot 1. Independent paths. |
| [multi-frame-intent-must-be-honored] | Silently emitting "idle only" when the goal asks for walk/attack/idle frames. The planner inverts to `from_image` chains. |
| [descriptive-verbs-are-art-change] | Treating "the sprites should look like X" as not-art-change. Verb set includes `look`, `want`, `need`, `should`. Fuzzy entity stems too. |
| [autonomous-mode-no-genre-logic] | Playtest recipes that gate on subject matter. Gates must be observable structure only. Applies to `memory/visual_playtests.jsonl` too — recipes are mechanism-keyed, NOT game-keyed. |
| [state-vs-render-gap] | Probes that test `state.X !== undefined` are blind to whether the entity is drawn. Always-on harness check samples canvas pixels. |
| [phase1-keeps-gpu-assignment-and-works-on-one-gpu] | Parallelism features that hardcode GPU assignments or refuse to run on single-GPU. Gate on observable independence. |
| [dont-burn-gpu-on-known-wrong-assets] | Generating mid-session `<assets>` while a style-rebrand is queued. Defer + coach. |
| [logs-for-llm-readers] | Logs written for humans only. The reflector + future you read the trace; structure for that audience. |
| [status-panel-live-rates] | Silent "agent is working" — there's always a live rate to show (tok/s, sprites/s, ETA). Memory layers in use also surface (`Memory in use:` block — skeleton, visual playtest, opening book). |
| [measure-before-changing] | Proposing big architectural changes from guessed numbers. Read recent traces first. |

---

## What changes well in this codebase, and what doesn't

### Changes that historically pay off

1. **Better harness signals.** A new probe-quality lint, an always-on render check, a structured trace event for an existing failure shape. These compound: every future session reads the new signal too. The state-vs-render gap detector (Pac-Man-without-a-Pac-Man) is exactly this shape.

2. **Listening fixes.** Anything that prevents the agent from swallowing what the user typed. Every iter the user can intervene is an iter that can recover. `/rawfeedback` defaults ON for this reason.

3. **Generalizable playbook bullets.** Hand-curated entries in `memory/playbook.jsonl` that get retrieved on future goals. A bullet that says "for any canvas game, expose `window.state.player.x` so the input smoke test detects movement" pays you back forever. A bullet that says "in chess games…" doesn't.

4. **Visibility.** Status panel rows, structured trace events with reasons, surprise categories. The agent gets better when *you can see why a session went sideways without re-running it*.

5. **Parallelism on truly independent hardware** — best-of-N fan-out across slots, non-blocking critic, diffuser pre-warm. Each gated on observable independence, each falls back cleanly on single-GPU.

### Changes that historically regress things

1. **Hardcoding genre logic.** Always tempting when you're staring at a single trace. Always reintroduces the listening bug.

2. **Tightening token / time abort thresholds without trace evidence.** The 27-35 B model emits long legitimate streams. Premature cutoffs trade a small win on one trace for a big regression on rich first-build streams.

3. **Adding always-on system-prompt rules.** Prompts past ~3-5 KB hurt mid-tier models. Use `FormatSpec.guidelines` (deduped per enabled format) or a goal-keyword detector that makes the rule conditional.

4. **Refactoring the per-spec asset loop.** It handles caching, chroma-keying, library admission, fuzzy parent lookup, multiple fallbacks. Risk-to-win ratio is poor unless there's a specific bug to fix.

5. **Cross-slot KV cache "fixes" that don't account for separate daemons.** Each Ollama daemon has its own KV cache. The fix is `warm_prefix` on the target slot, not a clever cache-sharing scheme (impossible across daemons).

---

## Memory layers — where to put a new thing

| File | Format | What goes here | When to add |
|---|---|---|---|
| `memory/playbook.jsonl` | committed JSONL (gitignored exception) | Hand-curated rules-of-thumb for code. Bullets retrieved by Jaccard against the goal each turn and injected into the prompt. Example: FPS camera basis vectors. | When you find a recurring code pattern the model gets wrong without it. Body should be actionable and complete. |
| `memory/visual_playtests.jsonl` | committed JSONL (gitignored exception) | Mechanism-keyed VLM checklists (yes/no questions). Matched by goal+plan+asset-name keyword overlap. **19 recipes today** cover the user's 25-archetype list at 100% (verified by `tests/test_visual_playtest_coverage.py`). Some carry `auto_probes` — deterministic JS state assertions that fail if the recipe's invariant breaks. | When a NEW mechanism shape appears (not a new game — mechanism). Adding a recipe is one JSONL line; no Python changes. |
| `memory/skeletons/` | committed `.html` + sibling `.json` | 17 mechanism-keyed first-build templates (`canvas_basic_v2`, `canvas_3d_basic`, `canvas_grid_basic`, `canvas_platformer_basic`, etc.). `retrieve_skeleton(goal)` picks one. | When a mechanism class genuinely needs a different scaffolding shape from what exists. Rare. |
| `memory/playtests.jsonl` | auto-seeded from Python | Behavioral playtest recipes the autonomous loop runs after clean iters. 6 today. **Different pattern from visual_playtests** — seeded from `_opening_book_seed_items` in `memory.py`. | When you find a behavior class the model regresses on AND it has a clean observable check (input + sample times → state delta). |
| `memory/asset_audits.jsonl` / `memory/animation_audits.jsonl` | auto-seeded | Asset / animation usage rules retrieved per turn into the `<opening_book>` block. | Rarely — most asset insights belong in the playbook instead. |
| `memory/mistakes.jsonl` | live-tier only (no seed) | Past mistake signatures from real sessions. Retrieved by error signature. | Don't author these by hand; the agent writes them on stuck-streak. |

**The standing pattern: game-shape knowledge lives in `memory/` data
files, NOT in Python code.** A recipe for a new mechanism is a one-line
JSONL append — no `_opening_book_seed_items` edit, no agent restart,
matches on the next session. The agent + memory infrastructure is
recipe-agnostic; the recipe carries the vocabulary.

The matcher (`memory.find_best_visual_playtest`) uses three signals:
1. **Goal text** — what the user typed.
2. **Plan text** — the model's Phase A `<criteria>` (rich mechanic vocabulary).
3. **Asset names** — the model's `<assets>` block names (genre signal).

This three-signal context is why goals that don't name the game still
match. *"Collect dots while avoiding ghosts in corridors"* resolves
to `canvas-grid-navigation` without the word "pacman" because
`corridor` + `dots` + `ghosts` are in the recipe's `applies_keywords`.

If you want to see WHICH recipe matched a session, the status panel's
"Memory in use" block shows it live. Trace events:
`visual_playtest_recipe_used`, `visual_playtest_auto_probes_injected`,
`opening_book_retrieved`.

**Playbook retrieval has a 0.02 Jaccard noise floor.** When you add a
new bullet, it must share enough words with typical goal text to
score above 0.02 — otherwise it's excluded and never reaches the
prompt. Symptom: the bullet exists, the failure class fires, but the
trace's `playbook_retrieved` event doesn't include the bullet's id.
Fix: broaden the bullet's tags (tags weight 2× content in the
Jaccard) with mechanism family words (`platformer`, `arcade`,
`shooter`), game-object words (`character`, `player`, `enemy`,
`boss`, `barrel`, `ghost`), failure-signal words from harness
reports (`entity`, `render`, `rendered`, `missing`, `invisible`),
and any specific game names that historically trigger the failure
(`donkey-kong`, `pacman`, `mario` — same pattern existing bullets
use). Keep bullet content tight (smaller union → higher Jaccard for
the same intersection). Lesson from `d6a4e38`.

---

## The four "feedback" flows — pay attention, they're separate switches

Multiple sessions burned debugging confusion between these. README
documents them in detail under "Feedback — four distinct flows".
TL;DR:

1. **Machine bug feedback** (always on, unconditional) — Chromium
   test report each iter: console errors, page errors, frozen canvas,
   RAF firing, probe results, input smoke test. This is the
   load-bearing verifier signal.
2. **Playbook retrieval** (`/playbook`, default ON) — `memory/playbook.jsonl`
   bullets retrieved per turn.
3. **Autonomous self-playtest** (`/feedback`, default ON) — after each
   clean iter, runs `memory/playtests.jsonl` recipes and queues
   `[AUTONOMOUS PLAYTEST]` findings as if user-typed.
4. **Directive wrapping on YOUR typed feedback** (`/rawfeedback`,
   **default ON = wrapping suppressed**) — classifier reads typed
   feedback, adds MEDIA-CHANGE / ORIENTATION-CHANGE / SCOPE
   ARBITRATION wrappers. **As of 2026-05-23, default is raw mode (no
   wrapping) per the [agent-must-beat-zero-shot] rule.**

When a user complains "the agent ignored what I said", check whether
flow #4 was on (`/rawfeedback off`). When the agent doesn't seem to
self-test, check #3. When the prompt is huge, check #2 + the
opening-book block size.

**TUI help for humans:** `/help` is the command cheat sheet; **`/help
<topic>`** pulls static pages from **`tui_help.py`** (30 topics:
`feedback-flows`, `feedback`, `vlm-critique`, `models`, `gpu`,
`session`, `assets`, …). **`/help topics`** is the full index. No LLM,
no GPU — works on Mac and single-GPU the same as on the 4-GPU box.
Point confused users at `/help feedback-flows` before they toggle the
wrong switch.

---

## TUI help (`tui_help.py`) — shipped, foundation for smarter help later

**Shipped 2026-05-26.** Static topic pages only — intentionally **not**
built with an LLM yet.

| Piece | Role |
|---|---|
| `tui_help.py` | `_HELP_TOPICS` (Rich-markup lines per topic), `_TOPIC_ALIASES`, `help_topics_index_lines()` |
| `chat.py` | `/help [topic]` → `_cmd_help(arg)` → `_render_help_lines()` (status panel mirror) |
| `tests/test_tui_help.py` | Aliases, topic content, app wiring |

**Design choices worth preserving when adding `/helpme` later:**

1. **Curated chunks, not README-at-runtime.** README stays long-form;
   `tui_help.py` is the retrieval corpus (one topic = one help unit).
2. **Retrieve-first, LLM-second (future).** Strong topic/alias match →
   print chunk as today. Weak free-text question + **agent idle** →
   optional small `stream_chat` with top 2–3 chunks as SOURCES only
   (2–4 sentences; say "not in help" if missing). Never route through
   `GameAgent.run()` — no feedback queue, no HTML, no session messages.
3. **GPU etiquette (future).** Retrieval always free. LLM help only when
   no slot is streaming; prefer architect slot / GPU 3 when staged, else
   coder slot when idle; never load a model just for help.
4. **Extend by editing data.** New topic = add `_HELP_TOPICS` entry +
   aliases + blurb; no prompt bloat.

**Do not build `/helpme` unless asked** — the static layer is the
foundation; see "What's open" item 8 below.

---

## Verifier guards (the 2026-05-23 + 24 layer)

Beyond the existing harness gates (probes, console errors, frozen
canvas, state-vs-render), these new detectors fire automatically.
Each carries a trace event so postmortem analysis is clean. None
require model changes; they all read existing state.

| Guard | Trace event | Catches |
|---|---|---|
| Silent-stream | `stream_silent_aborted` | Stream produces zero non-empty `content` for ≥180s (reasoning tokens hidden in `thinking` channel). Aborts in `ollama_io.py` + `backend.py`. |
| Procedural-regression | `procedural_regression` field on report; `PROCEDURAL_REGRESSION_SUSPECTED` soft-warning | Sprites declared but canvas draws ≥30 big rectangles (≥32×32 px) with `drawImage` count < big_rect/5. `fillRect` shim in `tools.py`. |
| Cross-turn patch-failure memory | `patch_search_repeat_detected` + `[REPEATED FAILURE]` prefix in prompt | Same SEARCH block failing two turns in a row (model copied stale lines). Fingerprint set on `_last_failed_patch_anchors`. |
| Coverage-gap fence | `[HARNESS NOTE — NOT FILE CONTENT, DO NOT <patch>]` wrapper in `probe_errors` | Model emitting `<patch>` targeting synthetic `coverage_gap__*` text. |
| Visual critic dedup | `coaching_suppressed_repeated` | Same critic note (normalized fingerprint of first 120 chars) across 3 consecutive critic turns. |
| Classifier overrule auto-disable | `classifier_auto_disabled_after_repeated_overrules` | Model overrules scoped classifier 2+ times in a session → `_use_feedback_directives` auto-flips False for the rest of the session. |
| `entity-progress-over-time` recipe gate | `autonomous_recipe_skipped reason=applicability_gate_falsy` | Recipe was false-positiving on input-driven games. Now requires a self-driven-motion signal in state (moving enemies / projectiles / score / timer). |
| Visual playtest auto-probes | `visual_playtest_auto_probes_injected` | Mechanism-recipe-injected JS assertions. `auto_actors_face_each_other`, `auto_player_not_in_wall`, `auto_player_within_canvas_bounds`. Conservative — return `true` when relevant state shape absent. |
| Identical-reply loop | `identical_reply_loop_detected` | Same `no_usable_code` reply (sha1 of normalized first 4 KB + length bucket) twice in a row. `_no_usable_code_fallback` returns a scope-reduction escalation: "stop re-emitting the file, send one tiny `<patch>`". Wolfenstein 2026-05-24 trace burned 5 iters before this existed. |
| Context-pressure | `context_pressure_warning` + `context_pressure_mitigation_applied` | `prompt_tokens / num_ctx >= 0.85` after a `stream_done`. Next `fix_instruction` call gets `context_pressure=True` and drops the inlined CURRENT FILE block + demands a 3-5-line-anchor patch. Prevents the model from emitting a full-file rewrite into 500 tokens of headroom. |
| Dead-first-build | `dead_first_build_detected` + `dead_first_build_attempt_aborted` | Iter ≤ 2 ships a file with `canvas.raf_ran=false` AND `input_test.any_change=false`. Next fix turn routes through `scope_reduction_instruction` ("ship a smaller intentionally-minimal rewrite"). After 2 such recoveries in one attempt, the iteration loop breaks so the restart loop can apply a fresh seed. |
| Restart-signature-repeat | `restart_signature_repeat` + `force_minimal_first_build_applied` | Two restart attempts in a row produce the same `_attempt_failure_signature` (`dead_first_build`, `identical_reply_loop`, `format_rejection_iter1`, `low_score`). Next attempt's `plan_instruction` gets `force_minimal_first_build=True` — 2-3 acceptance bullets, defer everything else. |
| Bare-html salvage | extraction variant 6 in `_extract_html_inner` | Reply contains `<html>...</html>` (≥ 200 bytes) but no `<!DOCTYPE>`. Salvages by prepending a synthetic doctype line instead of asking the model to retry. |
| Compaction-marker echo | `format_rejection` with `kind="compaction_marker_echoed"` | Model parroted the harness's own compaction marker (`[omitted: N bytes ...]` legacy or `HARNESS-OMITTED-PRIOR-*` current) back as if it were code. Specific recovery prompt names the cause. ROOT CAUSE FIX: new compaction marker uses an HTML comment with NO `<html_file>` / `<patch>` substrings so it can't be copied as a fresh tag. |
| Model-give-up | `model_give_up_detected` | `<done/>` + `<notes>` containing failure-confession phrases ("consistently failing", "manually copy", "harness parsing"). Treated as recovery request, not ship. Injects a "stop emitting `<done/>`, send a small `<patch>`" prompt. Wolfenstein 2026-05-24 trace turn [04] is the canonical case. |
| DeliberationDetector — latch on code (round 3) | NONE (no abort once latched) | Once the stream emits a real opener tag OR substantive HTML/JS (`<!DOCTYPE`, `<script`, `<canvas`, `function foo(`, `class X {`, lowercase `let/const/var foo =`), the detector latches and length stops being a fail signal — a 20-50 KB first build streams to completion. Threshold still fires on streams that emit ONLY pre-tag prose. The line-start-anchor requirement was DROPPED; negative-lookbehind for backtick preserves the doom 2026-05-17 "inline `<html_file>` mention" protection. |
| Patch breadcrumb anchor (Codex `@@`) | enriched error message names the `@@` option when SEARCH is ambiguous and no breadcrumb used | `<patch>` SEARCH can be prefixed with `@@ function_or_class_name` (stacked: `@@ class Foo` + `@@ def bar`) to scope an otherwise-ambiguous match. Parsed by `patches.py:_parse_breadcrumb_lines`; matches narrowed by `_narrow_to_breadcrumb_scope`. Advisory — if the name isn't in source, normal matching falls back. Mid-tier models reliably name functions; the breadcrumb is more actionable than "add more context." |

These are the cheap wins. Before adding a NEW guard, check this list
— the failure class may already be covered.

---

## The verification harness is the lever — understand it first

Most session failures look like agent failures but are actually **verifier failures**. The harness gates `report["ok"]`; if a broken game can pass the gates, every downstream improvement is wasted. The biggest wins this project has shipped are gate fixes:

- **`window.gameState` vs `window.state` bug** (2026-05-16, before this conversation): the input smoke test was reading the wrong global for the entire history of the code. Silently fell back to canvas-hash, which is degenerate for any auto-animating game. Fixed in `tools.py` — now the gate detects actual input response.
- **State-vs-render gap detector** (2026-05-22): probes that check `state.pacman.x !== undefined` pass on a game where the player isn't drawn. The always-on check samples canvas pixels at exposed entity positions.

When you find yourself adding agent-loop machinery to handle a failure pattern, **first** ask: is the gate detecting this correctly? If the gate flips `ok=False` reliably, the existing fix-mode loop is usually sufficient.

---

## The mid-tier-model constraint changes what works

Frontier models tolerate verbose, multi-objective prompts. Qwen3.6:27b does not. Patterns that work here:

1. **Imperative, concrete, with examples.** "Emit `<assets>` with N entries. Example shape: `[{name: 'X_idle', prompt: '...'}, {name: 'X_walk', from_image: 'X_idle', strength: 0.40, prompt: '...'}]`." Not: "the user explicitly asked for animation frames so consider whether you should use img2img chains".

2. **One directive per turn, not a stack.** When MEDIA-CHANGE, SCOPED-CHANGE, and STUCK-LOOP-FAST-PATH all fire on the same turn, the 27B model concatenates two drafts. The router suppresses conflicting directives explicitly (see `_flush_user_injections` and the `media_change_directive_suppressed` reasons).

3. **Place the user's exact words in the prompt.** `[USER NOTE]\n{user text}\n[/USER NOTE]` survives compaction; "the user wants X" rephrased by the agent gets compacted away.

4. **Surface the matched keywords back to the model.** `(matched: 'idle', 'walkstride', 'smash')` in the SCOPE-PACING NUDGE tells the model what triggered the rule. Without that, the model treats the rule as opaque.

5. **Tight prompts beat smart prompts.** The autonomous-playtest self-feedback paragraph is *deterministically* built from `recipe.finding_label + evidence` — no extra LLM call. Adding a critic call to "synthesize" it would cost a second slot stream for marginal wording improvement.

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

Concrete starting points, ranked by expected value. Full plans live
at `~/.claude/plans/vlm-critic-smarter-without-bigger.md` and
`vlm-critic-memory-driven-checklist.md`.

1. **Scope-your-fix prompt suffix.** When typed feedback contains
   direction keywords (`flipped`, `inverted`, `180`, `wrong
   direction`) AND a specific state name (`kick`, `punch`, `walk`),
   append a SCOPED-CHANGE coaching block telling the model not to
   flip a global condition. Catches the mortal-kombat-style
   wholesale-flip regression. ~30 lines in `prompts_v1`.

2. **Critic dedup hardening.** Today's dedup is sha1 of first 120
   normalized chars (`_critic_note_fingerprint`). Paraphrased
   complaints slip through. Swap in content-keyword Jaccard ≥ 0.45
   using the existing `_feedback_keywords` helper. ~10 LoC.

3. **Recipe `helpful` / `harmful` writeback.** When a visual playtest
   recipe's finding leads to a clean iter next turn → `helpful++`;
   regression after → `harmful++`. Same scoring infrastructure
   `memory.Playbook` already uses. Bad recipes silently fall out of
   retrieval; good ones surface more often.

4. **Multi-recipe-per-session checklist merging.** When 2+ recipes
   score above the floor (e.g. Pacman = grid-navigation +
   controllable-player), merge their checklists capped at ~10 total
   questions. Matcher already returns top candidates.

5. **Multi-image regression check** in `run_visual_critic`. When
   the loaded VLM supports 2-image input (LLaVA-1.6, MiniCPM-V-2.6,
   Qwen-VL), send before/after screenshots with three closed-class
   questions (ART_REGRESSION / ORIENTATION_FLIP / PROGRESS).
   `_last_screenshot_before` + `_last_screenshot_after` already exist.

6. **More behavioral playtest recipes** in `memory/playtests.jsonl`.
   Three shipped (`entity-progress-over-time`,
   `input-axis-matches-facing`, `held-key-stays-in-bounds`); the
   spec listed four more (`held-key-changes-state-twice`,
   `released-key-stops-action`, `ai-opponent-makes-a-move`,
   `game-over-actually-ends`). One recipe entry + one `check_kind`
   case per addition.

7. **Cross-session embeddings retrieval for the playbook.** Current
   Jaccard ranking has a 0.02 noise floor that excludes
   semantically-relevant bullets with low keyword overlap (see the
   "Playbook retrieval has a noise floor" note in Memory layers).
   Vector retrieval (local sentence-transformers) would fix this.
   Bullet schema is stable; swap inside `memory.Playbook.retrieve`.

8. **Smarter TUI help (`/helpme` or free-text `/help`).** **Not built.**
   Foundation is in place (`tui_help.py`). Planned shape: Jaccard/keyword
   retrieve top chunks → optional idle-gated paraphrase on sidecar slot;
   never `GameAgent`, never whole README. Human-in-the-loop topic authoring
   stays the default (append to `_HELP_TOPICS` when behavior changes).

---

## When in doubt — read these files in order

1. **`CLAUDE.md`** — the operational summary the user wrote. Loaded into every session by `agent._read_project_config`.
2. **`README.md`** — the deep walkthrough. Has a `Major future improvements` section with the project's own backlog.
3. **`tui_help.py`** — TUI topic help the human reads via `/help <topic>` (and the corpus for a future `/helpme`).
4. **`~/.claude/projects/-home-jonathan-Agent-learning/memory/MEMORY.md`** — the index of standing rules. Every entry is a hard-learned constraint.
5. **`HARNESS_DEBUG.md`** — what the verifier does AND doesn't catch. The blunt version.
6. **Recent traces in `games/traces/`** — the empirical truth. If your proposed change doesn't relate to a real trace pattern, you're probably making the prompt longer for no payoff.

---

## One last thing

This codebase has been shaped by hundreds of trace iterations against a small set of physical games. The "right" answer for an agent doing chess looks identical to the "right" answer for an agent doing Pac-Man looks identical to the answer for a future agent doing a roguelike — because **none of the rules above mention chess, Pac-Man, or roguelikes by name**. Keep it that way and the code stays general; break that and you're back to fixing one game at a time forever.

The user's standing principle: *"never make changes for one game, anything helping a type of game should be in our root level memory, but needs to be very general."* That's the bar. Hold it.

— Claude Opus 4.7, 2026-05-22 (original handoff)
— Claude Opus 4.7, 2026-05-24 (revision: memory layers reference,
  four-feedback-flows + verifier-guards sections, push target +
  multi-slot revert, current "What's open" backlog, donkey-kong
  trace lessons, "playbook noise floor" pattern, trimmed "Recent
  ships" to load-bearing lessons only)
— Claude Opus 4.7, 2026-05-25 (Wolfenstein stuck-loop fix bundle:
  6 new universal harness gates + reshaped compaction marker so
  the model can't parrot it back; 3 new broad playbook bullets;
  see verifier-guards table additions and Recent Ships entry)
— Claude Opus 4.7, 2026-05-25 LATE (rule 5 added — "length is not a
  fail signal once the stream is producing": DeliberationDetector
  latches on real code openers too, not just `<html_file>`; bumping
  the abort threshold is the wrong shape. Plus regression-test
  discipline for trace events emitted from prompt builders — they
  MUST source iteration from `self._last_tested_iter`, not a free
  variable. Plus Codex review slice 1 — `@@` breadcrumb anchor,
  plan-quality examples, "don't re-emit Phase-A" hard-rule.)
— Composer, 2026-05-26 (`tui_help.py` static `/help <topic>` pages;
  push target note fixed for `Agent_learning_overlay`; smarter `/helpme`
  documented as future work only — not implemented.)

---

## Recent ships — lessons worth carrying forward

Patterns from the 2026-05-23 → 2026-05-24 commits that aren't
obvious from the rest of the doc. Verifier guards already covered
above; this section adds lessons that are NOT in another section.

- **`/rawfeedback` default ON** (`58701ec`). Classifier directive
  wrapping on user-typed feedback is now suppressed by default.
  Anchor for the [agent-must-beat-zero-shot] rule. If you propose
  another classifier patch on typed feedback, read that memory file
  first.
- **Visual playtest library lives in JSONL** (`a3c759b`). Recipes
  used to be Python-seeded; now they're a hand-edited committed
  data file (`memory/visual_playtests.jsonl`). Adding a recipe is
  one JSONL line. The same pattern applies to playbook bullets —
  data file, no Python edit.
- **Structured VLM critic + auto-probes** (`b403eef`, `a18fea9`).
  Closed-class yes/no checklists replaced "what's wrong?".
  Mid-tier VLMs follow the structured form much more reliably.
  Three recipes also carry `auto_probes` — deterministic JS state
  assertions that catch the failure-class even if the VLM misses
  it visually.
- **Three-signal matching** (`1d1fd92`). The recipe matcher uses
  goal + plan text + asset names — not just goal. That's why
  *"collect dots while avoiding ghosts in corridors"* resolves to
  `canvas-grid-navigation` without the word "pacman" appearing.
- **25-archetype coverage at 100%** (`5e2778a` + `c2db88c`). The
  user supplied a 25-game list; `tests/test_visual_playtest_coverage.py`
  pins that every entry routes to a real mechanism recipe. When that
  test fails, that's the signal to add a recipe (one JSONL line),
  NOT to relax the test.
- **Playbook noise floor lesson** (`d6a4e38`). A relevant bullet can
  fail to retrieve because its tags don't overlap typical goal text.
  Covered in the Memory-layers section. Watch out when authoring new
  bullets.
- **DO NOT reintroduce default-on multi-slot autopin** (`ecfe067`
  reverted in `a205cd7`). It crashed the workstation when combined
  with iter-1 best-of-N fan-out. Multi-slot is opt-in only.
- **Wolfenstein 2026-05-24 stuck-loop fix bundle** (2026-05-25). A
  single ambitious-goal session burned 3.5 wall-clock hours across 3
  restart attempts and shipped nothing. Three independent harness
  gaps caused it; six new gates ship together. Read them in this
  order — they compound:
  1. Compaction marker no longer wraps itself in `<html_file>` tags
     (was: `<html_file>[omitted: N bytes ...]</html_file>`; now: HTML
     comment). The confused model literally cannot copy a fresh tag
     from a comment. Tests pin both shape and absence of `<html_file>`
     / `<patch>` substrings in the marker.
  2. `compaction_marker_echoed` detector in `classify_format_failure`
     catches any future leak of either old or new marker.
  3. `identical_reply_loop_detected` — sha1 fingerprint of normalized
     first 4 KB + length bucket. Two `no_usable_code` in a row with
     matching fingerprints → escalation prompt that demands a tiny
     `<patch>`, not another rewrite.
  4. `context_pressure_warning` — `prompt_tokens / num_ctx >= 0.85`
     sets a one-shot flag so the next `fix_instruction` drops the
     inlined CURRENT FILE block. Prevents the model from trying to
     emit a 30 KB rewrite into 50 tokens of headroom.
  5. `dead_first_build_detected` — iter ≤ 2 ships a file with
     `raf_ran=false` AND `input_test.any_change=false`. Routes
     through `scope_reduction_instruction` (ship a smaller
     intentionally-minimal rewrite). Two such recoveries in one
     attempt abort the attempt so the restart loop applies a
     fresh seed.
  6. `model_give_up_detected` — `<done/>` + `<notes>` confessing
     harness / parse trouble ("consistently failing", "manually
     copy") is treated as recovery request, not ship. Wolfenstein
     trace turn [04] is the canonical case.
  Plus `restart_signature_repeat` — two attempts with the same
  `_attempt_failure_signature` set `force_minimal_first_build=True`
  for the next attempt's `plan_instruction`. Plus three new playbook
  bullets: `first-build-scope-discipline`,
  `patch-only-when-context-fills`, `dual-state-exposure-for-probes`.
  Plus extraction variant 6: bare `<html>...</html>` with ≥ 200 byte
  body gets a synthetic doctype prepended. All gates are observable-
  signal-keyed (no goal text, no genre, no model name).

- **DeliberationDetector latch-on-code, not just on-tags** (2026-05-25
  round 3). The detector used to require `(?:^|\n)\s*<tag\b` at
  line-start to latch — chunking + 4 KB buffer trim ate the boundary
  on real long first-build streams and aborted them at exactly 6001
  chars. Wolfenstein-class FPS needs 20-30 KB of code; the gate killed
  it on the way up. Fix: (a) drop the line-start anchor (negative
  lookbehind for backtick still excludes prose `` `<html_file>` ``
  mentions per the doom 2026-05-17 protection), (b) add a SECOND latch
  family for HTML/JS code openers (`<!DOCTYPE`, `<script`, `<canvas`,
  `<body`, `function foo(`, `class Foo {`, lowercase `let/const/var
  foo =` — case-sensitive on purpose so "Let me think" doesn't match).
  Once latched, length stops being a fail signal. Threshold (6000) is
  still the backstop for streams that emit ONLY prose. Tests in
  `test_wolfenstein_stuck_loop_fixes.py::test_deliberation_*` pin all
  seven scenarios (latch on `<html_file>` at start, latch inline after
  prose, latch on bare DOCTYPE, latch on function decl, pure-prose
  still aborts, opener inside unclosed `<think>` doesn't latch, opener
  after `</think>` does latch).
- **Trace events from prompt builders must use in-scope vars** (the
  round-1 follow-up). Two `_trace(...)` calls inside `_build_fix_prompt`
  referenced `iteration` as a free variable; `_build_fix_prompt`'s
  signature has no `iteration` parameter. Detector unit tests passed —
  the gate fired its trace correctly in isolation — but the agent
  crashed with `NameError: name 'iteration' is not defined` the moment
  either flag fired in a real session. Fix: source iteration from
  `self._last_tested_iter` (the iter whose report this prompt is
  reacting to). Lesson now in root memory as
  [exercise-codepaths-not-just-detectors] and in two regression tests
  that exercise `_build_fix_prompt` with each flag set.
- **Codex review slice 1** (2026-05-25). Reviewed OpenAI's open-source
  Codex CLI (`github.com/openai/codex`) for transferable ideas. Three
  prompt-only changes shipped: (1) `@@ function_or_class_name`
  breadcrumb anchor in `<patch>` SEARCH blocks for disambiguating
  ambiguous matches — `patches.py:_parse_breadcrumb_lines` strips
  leading `@@ ident` lines and uses them to narrow `_locate` matches
  to that scope; stacked breadcrumbs supported; advisory (graceful
  fallback when the name isn't in source). (2) Good/bad plan-quality
  examples in `PLAN_INSTRUCTION` (mechanism-named GOOD examples vs
  vague LOW-QUALITY ones; all genre-free — earlier draft leaked
  "chess" / "snake" and was caught by the existing genre-free guard
  test). (3) New always-on hard-rule: *"Phase-A signals (plan,
  criteria, probes, media) persist once accepted; fix turns emit
  `<patch>` only"* — prevents `probes_only` / `media_only`
  `no_usable_code` shape before the model wastes a turn. Codex's
  `compact_remote_v2.rs` independently arrived at the same compaction
  fix shipped in round 2 (store summaries as user-role, not assistant)
  — confirms the round-2 reshape was the right call.
- **Two-failure-mode discipline for trace events** (carries forward).
  Every gate emits a trace event AND a coaching prompt downstream. The
  test for that gate MUST exercise BOTH paths: detector fires + prompt
  builder consumes the flag without crash. Unit tests on the detector
  alone miss the second path. Pattern is documented in the new memory
  file and the two regression tests added today (one each for
  context-pressure and dead-first-build).
- **Static TUI help topics** (2026-05-26). `tui_help.py` + `/help
  <topic>` / `/help topics` — 30 curated pages (feedback vs
  vlm-critique, GPU topology, session sticky state, assets, …). No LLM.
  Intended corpus for a future idle-gated `/helpme`; see the dedicated
  section above. When you change toggle behavior, update `tui_help.py`
  in the same PR — do not rely on README alone for operator confusion.

**Where to start when picking this up:** run a fresh session against
any of the 25 archetypes. Watch the right-side status panel —
"Memory in use" should show a skeleton + visual playtest recipe +
opening-book hits within ~30 s of `/new`. If it doesn't, that's the
next bug.
