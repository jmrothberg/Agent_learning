# Coding Box - HTML Game Agent

A small, opinionated agent that drives a local Ollama model to **write,
test, and iteratively fix single-file HTML5 games**, with live feedback
from a real Chromium browser. Comes with a Textual TUI for two-way chat
and a plain CLI for unattended runs.

The agent is built around one principle: **working > perfect**. A clean
turn is treated as sacred and saved to `games/best.html`; regressions
are detected and reverted; user feedback typed mid-run is the highest-
priority signal in the loop.

**Remote:** https://github.com/jmrothberg/Agent_learning/

---

## Quick start

```bash
# 1. clone & enter
git clone https://github.com/jmrothberg/Agent_learning.git
cd Agent_learning

# 2. python venv (already provisioned in .venv/, but if you blow it away):
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 3. make sure ollama is running and you have a model installed
ollama serve &      # in a separate terminal if not already running
ollama list         # confirm at least one usable tag (e.g. gpt-oss:latest)

# 4. interactive TUI (recommended)
.venv/bin/python chat.py

# 5. or one-shot CLI
.venv/bin/python coder.py "build me a simple snake game"
```

---

## What the TUI looks like

```
┌─ Coding Box - <model> ───────────────────────────────────────┐
│ agent log pane                            │ Status           │
│ ── planning ──                            │ Phase: ...       │
│ <plan>...</plan>                          │ Iteration: 2/6   │
│ ── iteration 1/6 ──                       │ Model: ...       │
│ <html_file>...</html_file>                │ Goal: ...        │
│ TEST FAILED (1 error(s), 0 issue(s))      │                  │
│ i → applying your input to next turn: ... │ Log: <path>      │
│ ── iteration 2/6 ──                       │ Ctrl+L reprint   │
│                                           │                  │
├──────────────────────────────────────────────────────────────┤
│ > type feedback any time, or Ctrl+D when done                │
└──────────────────────────────────────────────────────────────┘
```

---

## Keys

| Key            | What it does                                         |
| -------------- | ---------------------------------------------------- |
| `Enter`        | submit text (goal / answer to model question / feedback) |
| `Ctrl+D`       | "ship it" - agent finishes current turn and exits        |
| `Ctrl+L`       | re-print all log file paths for this session             |
| `Ctrl+Q`       | quit (browser is cleaned up automatically)               |
| `Shift`+drag   | select text in the log pane (bypasses TUI mouse capture) |
| `Ctrl+Shift+C` | copy selection (after Shift+drag)                        |

If your terminal stops echoing after exit, run `reset` and you're fine.

---

## How feedback works (this is the important bit)

You can type ANY text in the input box during a run. It is NOT sent to
the model immediately - it is queued and injected into the very next
user-turn the agent builds.

The TUI gives you four signals so you always know where your words are:

1. `> feedback: your text` - the moment you press Enter
2. `✓ queued (pending: N)` - immediate ack from the agent
3. `→ applying your input to next turn: feedback: '...'` - the moment
   the words actually land in a prompt to the model
4. The model's next reply should explicitly address your feedback

Your feedback is wrapped in a loud fenced banner at the top of the
prompt:

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

---

## Model selection

In order:

1. `OLLAMA_MODEL=<tag>` env var (explicit override)
2. First installed model from `ollama list` that isn't blacklisted
3. Hard fallback in `coder.py::MODEL`

Blacklisted tags live in `chat.py::_KNOWN_BROKEN_TAGS`. Currently
includes `qwen3.6:27b` because it 500-errors on load on this box.

To force a specific model:

```bash
OLLAMA_MODEL=gpt-oss:latest .venv/bin/python chat.py
```

---

## File layout & where to look when something fails

```
Agent_learning/
├── chat.py              # Textual TUI (recommended entry point)
├── coder.py             # CLI agent (one-shot, no UI)
├── agent.py             # async event-driven agent core (used by chat.py)
├── tools.py             # Playwright browser harness + game heuristics
├── prompts.py           # SYSTEM_PROMPT + per-phase instructions
├── requirements.txt
├── games/
│   ├── game.html                # the game the agent is iterating on
│   ├── best.html                # last clean version (auto-saved)
│   ├── snapshots/<ts>/iter_NN.html  # every iteration's full HTML
│   ├── snapshots/<ts>/iter_NN.png   # browser screenshot per iteration
│   └── traces/
│       ├── <ts>.log             # plain-text mirror of the TUI agent log
│       ├── <ts>.jsonl           # structured event stream (one JSON per line)
│       └── <ts>.conversation.md # FULL message history sent to/from the model
```

When a session goes wrong, the single most useful file to share for
debugging is `games/traces/<ts>.log` - it has every model token, every
test report, every error with traceback, and every UI event in order.
For the model's exact view of the conversation use `<ts>.conversation.md`.

`Ctrl+L` in the TUI prints all five paths at once.

---

## How the loop works

Three phases, all driven by simple XML-style tags the model emits:

1. **PHASE A - planning (1 turn)**: model emits `<plan>...</plan>` only.
2. **PHASE B - build/iterate (up to `max_iters`)**: model emits
   `<html_file>...</html_file>`. Harness runs it in real Chromium and
   reports back: console errors, page errors, canvas state, RAF firing,
   input listener count, frozen-canvas check, automated input smoke test,
   and (if the model is a VLM) a screenshot.
3. **PHASE C - self-critique (1 turn)**: when the model says `<done/>`
   on a clean run, it gets one final pass to either send a fix or
   reply `<confirm_done/>`.

Adaptive temperature: 0.7 on first/clean turns (creative), 0.25 after
a failed test (precision for fixes). Last known-good code is embedded
in fix prompts so the model patches concrete code instead of trying
to remember its own previous reply.

---

## Restarting a session / picking up where we left off

This project has no persistent agent state across runs - each session
starts fresh. To resume after a restart:

1. **Read the latest trace.** `ls -t games/traces/*.log | head -1`
   then `cat` it. The conversation, the goal, the failures, and what
   the model tried are all there.
2. **Inspect the last working game.** `games/best.html` opens in any
   browser. `games/current.html` is whatever the agent finished with
   (may be broken).
3. **Inspect snapshots.** `games/snapshots/<ts>/iter_NN.html` plus
   the matching `.png` show every step.
4. **Re-run with the same goal.** Tell `chat.py` your previous goal
   (or paste from `<ts>.conversation.md` cell `[01]`).

The README + the trace files together are enough to hand off the
project to a fresh assistant: paste this README, paste the most
recent `<ts>.log`, describe your next goal.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not launch Chromium` | run `playwright install chromium` |
| Model defaults to wrong tag | `OLLAMA_MODEL=<tag> .venv/bin/python chat.py` |
| Ollama 500 on model load | the tag is broken locally; pick another from `ollama list` |
| Terminal stops echoing after exit | `reset` (Ctrl+Q is the proper exit, not Ctrl+C) |
| Can't select text in TUI | hold `Shift` while click-dragging |
| Feedback ignored | check the log for `→ applying your input` - if missing, see `<ts>.jsonl` for `feedback_queued` / `feedback_injected` events |

---

## Dependencies

See `requirements.txt`. Key packages:

- `ollama` - python client for the local Ollama daemon
- `playwright` - real Chromium for game testing
- `textual` - the TUI framework
