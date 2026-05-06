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
`GameAgent` with **`prompt_version="v0"`** (classic `prompts.py`). The model
does **not** receive the retrieved `<playbook>` block, `<criteria>`, or JSON
`<probes>` on that path — those are **`prompts_v1.py`** features. Use
`tune.py run --prompt-version v1`, or pass `prompt_version="v1"` when creating
`GameAgent`, to exercise the full prompt + playbook injection described below.
(Traces may still record `playbook_retrieved` on v0 for the offline learner,
but the running model’s prompts omit that block until v1+.)

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
- [What the TUI looks like](#what-the-tui-looks-like)
- [Slash commands](#slash-commands)
- [Keys](#keys)
- [How feedback works](#how-feedback-works-this-is-the-important-bit)
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

- **Python 3.10+** recommended (async stack, typing, pathlib).
- **Ollama** running locally (`ollama serve`) with at least one pulled model.
- **Playwright Chromium**: after `pip install -r requirements.txt`, run  
  `playwright install chromium`.
- **Display for `chat.py`**: the default TUI launches a **visible** Chromium
  window. You need a working desktop/Wayland/X11 session (or X forwarding
  that Playwright can use). For SSH-only or CI machines, use  
  `coder.py --headless` instead of the TUI.
- **`OLLAMA_HOST`**: if your daemon is not on the default address, set this
  before launching (the Python client and HTTP fallbacks respect it).
- **Trust**: the model writes HTML/JS that is executed in a real browser from
  `file://` URLs. Only run models and seeds you trust; treat generated games
  like untrusted web pages if you re-host them.

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

**Cleanup utility:**

```bash
# See what's stored.
.venv/bin/python scripts/forget_session.py --list

# Wipe one session's skeleton + goals record + matching mistakes entries.
.venv/bin/python scripts/forget_session.py game-of-misile-command-good-gr_20260505_133453

# Dry-run first if unsure.
.venv/bin/python scripts/forget_session.py --dry-run <session_id>
```

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

## CLI (`coder.py`)

Same `GameAgent` as the TUI, without Textual. Useful for scripting, CI, or
machines without a display (`--headless`).

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--model` | env `OLLAMA_MODEL` or `coder.MODEL` | Ollama model tag |
| `--max-iters` | `6` | Cap plan/build iterations |
| `--out` | unique `games/<slug>_<timestamp>.html` | Output HTML path |
| `--best-of-n` | `1` | Sample N candidate fixes per failed iteration |
| `--num-ctx` | `8192` | Ollama context window |
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
├── prompts_v1.py        # v1 prompt: XML-structured, FormatSpec-assembled, criteria/probes
├── memory.py            # skeletons + mistakes + Playbook (quality-ranked, deduped, budgeted)
├── patches.py           # SEARCH/REPLACE patch engine: fuzzy norm, uniqueness, repair layer
├── ollama_io.py         # streaming watchdog + best-of-N sampler
├── tests/
│   ├── test_patches.py        # 24 tests — fuzzy match, uniqueness, overlap, repair
│   ├── test_compaction.py     # 9 tests — structured summary + 2-tier prune
│   ├── test_retrieval.py      # 20 tests — quality rank, two-stage, dedup, budget, hybrid
│   ├── test_microprobes.py    # 23 tests — pre-Chromium sanity + API allowlist
│   ├── test_lookup.py         # 8 tests — <lookup_bullet> tag → playbook body injection
│   └── test_project_config.py # 7 tests — AGENTS.md / CLAUDE.md auto-loading
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
| Feedback ignored                  | check the log for `→ applying your input` — if missing, see `<slug>_<ts>.jsonl` for `feedback_queued` / `feedback_injected` events |
| Feedback after done does nothing  | it should auto-extend; verify the bottom hint says "type feedback to extend"          |
| Playbook / criteria never show up in traces | normal for **`chat.py`**: default is **`prompt_version=v0`**. Use `tune.py run --prompt-version v1` or pass `prompt_version="v1"` into `GameAgent` to exercise injected `<playbook>` + `<criteria>` / `<probes>`. |

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

## Dependencies

See `requirements.txt`. Key packages:

- `ollama` — python client for the local Ollama daemon (also used by
  `learner.py reflect` / `apply` to call the Reflector; default model is
  set in `learner.py` and overridable with `--model`)
- `playwright` — real Chromium for game testing
- `textual` — the TUI framework
- `rich` — markup + escape helpers used by the TUI mirror layer

`tune.py` / `learner.py` have no extra pip dependencies beyond the above.

---

## License

No `LICENSE` file ships in this repository yet. If you fork or redistribute
the code, add a license you are comfortable with (or confirm terms with the
repository owner on GitHub).
