# Coding Box — HTML Game Agent

A small, opinionated agent that drives a **local Ollama model** to write,
test, and iteratively fix single-file HTML5 games — with live feedback
from a real Chromium browser. Comes with a Textual TUI for two-way chat
and a plain CLI for unattended runs.

The core thesis: **a small validated model beats a large unvalidated one.**
Every model output is run in a real browser, every error is fed back,
every clean turn is preserved, every regression is reverted, and user
feedback typed mid-run is the highest-priority signal in the loop.

**Remote:** https://github.com/jmrothberg/Agent_learning/

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
                    │  MEMORY ▸ retrieve_skeleton(goal)    │
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
   │          │   │  MEMORY ▸ retrieve_mistakes(signature, k=3)      │    │
   │          │   │  past root-causes for THIS failure type, inlined │    │
   │          │   │  into the next prompt as hints                   │    │
   │          │   └──────────────────────────────────────────────────┘    │
   │          ▼                                                           │
   │   ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐   │
   │   │ BEST-OF-N    │    │ VLM SCREENSHOT  │    │ USER FEEDBACK     │   │
   │   │ ─────────    │    │ ─────────────── │    │ ─────────────     │   │
   │   │ sample N     │    │ vision model    │    │ free text typed   │   │
   │   │ silently,    │    │ gets the .png + │    │ at any time, in   │   │
   │   │ score each   │    │ "compare to     │    │ a HIGHEST-PRIORITY│   │
   │   │ in headless  │    │  expectation"   │    │ banner, OVERRIDES │   │
   │   │ Chromium,    │    │ catches "runs   │    │ plan + defaults   │   │
   │   │ keep winner  │    │ but looks wrong"│    │ this turn         │   │
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
| **Memory · skeleton**        | First-build cold start ("how do I even structure this?")       | `memory.retrieve_skeleton`                  |
| **Patches > rewrites**       | Token budget blown re-emitting unchanged code; truncation bugs | `patches.extract_patches`, `apply_patches`  |
| **Real-browser tests**       | Hallucinated APIs, wrong event names, broken bracket indexing  | `tools.LiveBrowser.load_and_test`           |
| **Memory · past mistakes**   | Re-making the same fix five sessions in a row                  | `memory.retrieve_mistakes` + `record_mistake` |
| **Diagnose-then-fix**        | Patches applied without root-cause analysis; whack-a-mole      | `prompts.fix_instruction`                   |
| **Best-of-N (fix mode)**     | One sample landing on a bad local minimum                      | `agent._generate_and_score_candidates`      |
| **VLM screenshot review**    | "Game runs but looks wrong" — bugs the harness can't detect    | `agent._stream` → `image_attached`          |
| **User feedback override**   | Model running in circles, ignoring obvious user intent         | `agent._flush_user_injections`              |
| **Self-critique pass**       | "Confidently wrong" final reply ships unchallenged             | `prompts.CRITIQUE_INSTRUCTION`              |
| **Save best on every clean** | A regression destroying yesterday's working code               | `agent._save_best`                          |
| **Continuation extends**     | "Add sound" after `<done/>` means restart from scratch         | `agent.run(continuation=True)`              |
| **Adaptive temperature**     | Same temp for creative-build and bug-fix turns                 | `agent._stream` (`0.7` build → `0.25` fix)  |
| **Conversation pruning**     | Long sessions overflow context; old code crowds out new turns  | `agent._prune_messages`                     |

---

## Quick start

```bash
# 1. clone & enter
git clone https://github.com/jmrothberg/Agent_learning.git
cd Agent_learning

# 2. python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 3. make sure ollama is running and you have a model installed
ollama serve &      # in a separate terminal if not already running
ollama list         # confirm at least one usable tag

# 4. interactive TUI (recommended)
.venv/bin/python chat.py

# 5. or one-shot CLI
.venv/bin/python coder.py "build me a simple snake game"
```

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

---

## What the TUI looks like

```
┌─ Coding Box · <model> ─────────────────────────────────────────────┐
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

The TUI gives four signals so you always know where your words are:

1. `> feedback: your text` — the moment you press Enter
2. `✓ queued (pending: N)` — immediate ack
3. `→ applying your input to next turn` — the moment the words land in a prompt
4. The model's next reply addresses your feedback explicitly

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

The blacklist (`chat._KNOWN_BROKEN_TAGS`) only affects the installed-list
fallback, never your explicit choice or what's currently loaded.

---

## File layout & where to look when something fails

Every session writes a unique meaningful basename: `<goal-slug>_<timestamp>`.
No artifact ever overwrites a previous session's.

```
Agent_learning/
├── chat.py              # Textual TUI (recommended entry point)
├── coder.py             # CLI agent (one-shot, no UI)
├── agent.py             # async event-driven agent core
├── tools.py             # Playwright browser harness + game heuristics
├── prompts.py           # SYSTEM_PROMPT + per-phase instructions
├── memory.py            # skeleton + mistake retrieval, signature hashing
├── patches.py           # SEARCH/REPLACE patch parser & applier
├── ollama_io.py         # streaming watchdog + best-of-N sampler
├── requirements.txt
└── games/
    ├── <slug>_<ts>.html              # the live game file (e.g. asteroids_20260503_185455.html)
    ├── <slug>_<ts>.best.html         # last clean version (auto-saved)
    ├── snapshots/<slug>_<ts>/
    │   ├── iter_01.html              # every iteration's full HTML
    │   └── iter_01.png               # browser screenshot per iteration
    └── traces/
        ├── <slug>_<ts>.log           # plain-text mirror of the TUI agent log
        ├── <slug>_<ts>.jsonl         # structured event stream (one JSON per line)
        └── <slug>_<ts>.conversation.md  # FULL message history sent to/from the model
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
2. **PHASE B · build / iterate** (up to `max_iters`): model emits
   `<patch>` blocks (preferred) or a full `<html_file>...</html_file>`.
   Harness runs it in real Chromium and reports back: console errors,
   page errors, canvas state, RAF firing, input listener count,
   frozen-canvas check, automated input smoke test, and (if the model
   is a VLM) attaches the latest screenshot for vision review.
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
| `Could not launch Chromium`       | run `playwright install chromium`                                                      |
| Picks the wrong model             | `/list` then `/model <N>`, or `OLLAMA_MODEL=<tag> .venv/bin/python chat.py`           |
| Model picked is stale             | `/api/ps` is sorted by `expires_at` — touch your preferred model to bump it           |
| Ollama 500 on model load          | the tag is broken locally; pick another with `/list` + `/model <N>`                    |
| Terminal stops echoing after exit | `reset` (Ctrl+Q is the proper exit, not Ctrl+C)                                        |
| Can't select text in TUI          | hold `Shift` while click-dragging                                                      |
| Feedback ignored                  | check the log for `→ applying your input` — if missing, see `<slug>_<ts>.jsonl` for `feedback_queued` / `feedback_injected` events |
| Feedback after done does nothing  | it should auto-extend; verify the bottom hint says "type feedback to extend"          |

---

## Dependencies

See `requirements.txt`. Key packages:

- `ollama` — python client for the local Ollama daemon
- `playwright` — real Chromium for game testing
- `textual` — the TUI framework
- `rich` — markup + escape helpers used by the TUI mirror layer
