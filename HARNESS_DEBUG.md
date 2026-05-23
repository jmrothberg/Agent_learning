# HARNESS_DEBUG.md — what's in the harness, what's NOT in the harness, and how to debug when a session goes badly

## 🔥 2026-05-16 — the actual reason "dozens of tries no improvement"

**The gameplay verification path was sampling the wrong global.** [tools.py](tools.py)'s input smoke test (the harness's primary "does this game respond to input" gate) was reading `window.gameState`, but the system prompt teaches the model to expose `window.state`, and every won-skeleton on disk uses `window.state`. **They never matched.** The test silently fell back to canvas-hash, which is degenerate for any auto-animating game (asteroids drifting, fires flickering, parallax scrolling). For the entire history of this code, the harness has been blind to whether the agent's games actually respond to input.

Verified with a live-Chromium smoke (`scripts/_smoke_input_test.py`): a game that exposes `window.state` and moves `state.player.x` on `ArrowRight` now produces:

```
Input test: PASS — ArrowLeft→[player.x]; ArrowRight→[player.x]; Space→[bullets.0.x, bullets.0.y, …]
```

vs. the prior "Input test: PASS — pressed keys, canvas changed on 'ArrowRight'" (which would happen on a game that DIDN'T respond to input but had any auto-animation — false-positive).

**This is the change that should move quality.** Every prior detector was sharpening signals that fired correctly on failure shapes; this one was the gate itself, and it was silently degenerate.

Files: [tools.py:2166-2204](tools.py:2166) (the snapshot JS), [tools.py:2274-2350](tools.py:2274) (return shape + summary), [tools.py:1483-1497](tools.py:1483) (report formatter), [tests/test_input_smoke_state_global.py](tests/test_input_smoke_state_global.py), [scripts/_smoke_input_test.py](scripts/_smoke_input_test.py).

---


This doc is for someone (often: future-me) who needs to fix the agent after a session went poorly. It is deliberately blunt about the limits of the harness. **The harness detects, coaches, and learns from failure. It does not make the model better at writing code in a single session.** Most of the leverage on first-iter quality comes from the model, the playbook, and the prompts — not from the detectors. If sessions are flat-lining, the questions to ask are different from "did we add another detector."

---

## TL;DR — why "dozens of tries with no improvement" is possible

Almost every recent change is a **detection** or **recovery** improvement, not a **skill** improvement:

| Change | What it actually does | What it does NOT do |
|---|---|---|
| Adjacent-line spam trigger ([ollama_io.py](ollama_io.py)) | Aborts a stream 8 lines sooner when the model is spamming an identical line | Teach the model not to spam |
| Dotted-elision regex ([tools.py](tools.py)) | Catches `// ...rest of seed code stays same` placeholders pre-Chromium | Teach the model to emit complete files |
| Duplicate-decl micro-probe ([tools.py](tools.py)) | Names the duplicated identifier before the 3-second browser load | Teach the model not to concatenate drafts |
| Unclosed-html-after-loop coaching ([agent.py](agent.py)) | Tells the model *what* line was looping when its `<html_file>` got aborted | Prevent the loop in the first place |
| Probes-only reply coaching ([agent.py](agent.py)) | Names the failure shape ("you re-emitted probes, no code") | Make the model emit code on its own |
| Probe-sanity lint ([agent.py](agent.py)) | Flags `b.x0` (never assigned) or `return true` (after discarded work) | Make the model write better probes |

**Read that table again.** If you ran the agent 30 times and saw little quality lift, that is consistent with these changes working as designed: faster aborts, clearer coaching, slightly better playbook bullets the *next* session. None of these changes add an inch of headroom to "this model can build a playable Donkey Kong in 6 iters." That headroom comes from elsewhere (see "where the actual quality lift lives" below).

---

## What was added recently — change map for fast navigation

Each item names the file and the failure trace that motivated it. The inline comments at each site quote the trace explicitly so you can trace symptom → fix without reading this doc.

### Stream-time detection (Layer A — fire on the bytes as they arrive)

- **[ollama_io.py:67-85](ollama_io.py:67)** — `_ADJACENT_SPAM_REPEATS = 4` constant + Window-4 logic inside `RepetitionDetector.feed`. Fires when the last 4 normalized non-empty lines collapse to a single unique value. Strictly tighter than Windows 1/2 (which need ≥ 12 lines). Trace: donkey-kong `20260516_142445` iter 1 (`p.onGirder = false;` × 16).
- **[ollama_io.py:441-456](ollama_io.py:441) + [backend.py:1080-1095](backend.py:1080)** — `StreamResult.loop_kind` + `loop_line` fields. Surfaced from `RepetitionDetector.stall_reason` / `loop_line` so the agent coaching layer can name *which line was looping*. Both Ollama and MLX paths populate these.
- **[tools.py:879-937](tools.py:879)** — duplicate-top-level-declaration micro-probe. Regex-scans inline `<script>` bodies for `const`/`let`/`function` declarations at depth ≤ 1 (inside the IIFE wrapper); any name declared ≥ 2× errors with prescriptive text. Trace: donkey-kong `20260516_124628` iter 2.
- **[tools.py:962-997](tools.py:962)** — elision-marker check, regex variant added. Now matches `// ...rest of`, `// .. rest unchanged`, etc., in addition to the prior literal substrings. Trace: same iter (the literal `// ...rest of seed code stays same` was ignored before because the existing match was `// rest of`).

### Coaching (Layer B — when existing detectors fire, tell the model what to do differently)

- **[agent.py:1175-1184, 2885-2891](agent.py:1175)** — `_last_stream_loop_kind` and `_last_stream_loop_line` state, plumbed from `StreamResult`. Cleared on every fresh stream.
- **[agent.py:1418-1473](agent.py:1418)** — `_no_usable_code_fallback` branch for `prior_stream_looped` + `rejection.kind in (unclosed_html_file, unclosed_patch)`. Names the loop shape, quotes the repeated line, and offers `<question>` OR a smaller `<html_file>` that omits the dead branch. Trace: donkey-kong `20260516_142445` (the user manually typed "get back on track" because the existing coach was generic).
- **[agent.py:1399-1416](agent.py:1399), [5570-5605](agent.py:5570)** — probes-only / media-only reply detection. New `probes_only` and `media_only` flags on `_no_usable_code_fallback`. Caller computes them from `_PROBES_OPEN_RE` / `_ASSETS_OPEN_RE` / `_SOUNDS_OPEN_RE`. Coach names the specific shape.
- **[agent.py:3893-4029](agent.py:3893)** — two static methods on `GameAgent`:
  - `_lint_probes(probes)` — finds probes that bind a `const x0 = …` and then `return true|false|0|1` without using `x0`. Runs at probe-parse time.
  - `_probes_referencing_unassigned_props(probes, html)` — finds `obj.prop` reads where `prop` is never assigned anywhere in the on-disk HTML. DOM properties are in an ignore set; method calls (followed by `(`) are skipped. Runs after each `_materialize`.
- **[agent.py:4694-4711](agent.py:4694)** — Phase-A wiring: `_lint_probes` is called right after `_classify_probes_dynamic`; findings stash on `self._probe_lint_findings` and emit `[yellow]probe lint[/yellow]` info events.
- **[agent.py:5790-5810](agent.py:5790)** — post-materialize wiring: `_probes_referencing_unassigned_props` runs on `new_html`, merges findings (replacing prior post-build entries, preserving the parse-time tautology entries).
- **[agent.py:7036-7048](agent.py:7036)** — `fix_instruction` consumer: any findings on `self._probe_lint_findings` get appended to the diagnose-then-fix prompt with a directive to re-emit `<probes>` alongside the patch.

### User-facing additions

- **[chat.py](chat.py) `/ref <path>`** — attach a reference image (PNG/JPEG/WebP) to the next user turn. Use case: "make the game look like this." Magic-byte validated, 4 MB cap, VLM-only. The agent's existing `_next_image_bytes` plumbing handles the rest. See "VLM image-paste — what's possible" below for limits.
- **[chat.py](chat.py) `/check` — expanded vendor support.** Previously Claude-only; now routes through `vision_judge._cloud_vendor()` which recognizes `claude*` / `anthropic*` (→ Anthropic), `gpt*` / `openai*` / `o[1-9]-*` (→ OpenAI), and anything else (→ local MLX VLM resolver). With no `with <model>` argument, uses the active session model if it's a VLM (zero API cost for users who are already running a local VLM session). Aliases now include `gpt` / `gpt-5` / `openai` / `gpt-5-mini` alongside the prior Claude shortcuts. Error messages name all three escape paths (local / Anthropic / OpenAI) so a missing key in one vendor doesn't dead-end the user. See [vision_judge.py](vision_judge.py) `_openai_judge` for the new Responses-API call (image content uses `{"type": "input_image", "image_url": "data:image/png;base64,…"}` instead of Anthropic's source-block shape).

### Tests

- **[tests/test_microprobes.py](tests/test_microprobes.py)** — 4 new cases: dotted-elision (2), duplicate-decl positive + negative.
- **[tests/test_bloat_detectors.py](tests/test_bloat_detectors.py)** — added `test_repdetector_adjacent_line_spam_triggers`, updated the existing tests to use varied inputs so they still exercise their intended window (the adjacency check is strictly tighter than the prior windows, so test inputs needed to alternate to distinguish them).
- **[tests/test_check_routing.py](tests/test_check_routing.py)** — 7 tests covering `_cloud_vendor` (Anthropic / OpenAI / local), case-insensitivity, drift-guard between `_cloud_vendor` and `_looks_like_local_mlx`, and import-level surface for `_openai_judge`.

Total tests: 642 (was 617 before this work). Full suite still runs in ~2 s.

---

## Why dozens of tries can show no improvement — the harness layer vs. the skill layer

### What this harness IS good at

- **Aborting bad streams fast.** Once the model enters a token-repetition loop, every additional token is wasted compute. The detectors keep getting tighter (Window 4 abort at 4 lines now, was ≥ 12).
- **Naming the failure shape.** When something breaks, the coaching now quotes the actual broken thing back to the model (the duplicated identifier, the looped line, the unassigned property), rather than a generic "your reply was malformed."
- **Catching cheap structural mistakes pre-Chromium.** Elision sentinels, duplicate `const`, brace imbalance, API-allowlist hallucinations — all fire before paying the 3-second browser load and tell the model exactly which name was wrong.
- **Not silently calling the cloud.** The cloud-VLM path is `/check with <model>` only — the agent loop never auto-upgrades.

### What this harness is NOT good at (and where future quality lift actually lives)

These are the things that bound how good first-iter HTML can be on a fixed local model. Detectors don't move these:

1. **The base model's HTML/JS prior.** A 27B / 35B local model has a ceiling on how well it can hold a 16 KB-game architecture in one shot. No amount of coaching makes a 27B model write a flawless side-scroller.
2. **The playbook's coverage.** A bullet only helps if it retrieves on the goal's keywords. Today's retrieval is token-Jaccard; "explosion sprite" and "boom particle effect" don't hit each other. See README "Major future improvements" #3 (semantic embedding retrieval) — that's a real quality unlock.
3. **The vision judge doesn't gate `<done/>`.** A game can pass all probes with a blank canvas and ship. See README "Major future improvements" #1 — single-line guard in `agent.py`'s done-detection block, highest-leverage change still on the list.
4. **A single model is both implementer and reviewer.** Self-critique by a small model is weak. Specialized sidecar critics (art / sound / gameplay) each on a tight scoped prompt would catch more. See README #4.

**If you ran the agent 30 times and saw flat results, the bottleneck is one of items 1–4, not "we need another detector."**

### Honest diagnostic recipe — "why is this session not improving?"

Run these in order. The first non-empty answer is the bottleneck.

1. **Scan FAIL signatures across the recent N sessions in `games/traces/`.** Are the FAIL signatures the same every time, or different? If the same: the playbook is missing a bullet for that shape — hand-add one to `memory/playbook.jsonl`. If different: the bottleneck is base-model quality, not detection.
2. **`grep "stream_done.*looped.*true" games/traces/*.jsonl | wc -l`** — how often does the repetition detector fire? If > 1 in 4 iters, the prompt or seed code is provoking loops (see "Don't inline large literals" in the system prompt's hard-rules; the seeded-generator escape hatch may need wider mention).
3. **`grep "duplicate top-level declaration" games/traces/*.jsonl | wc -l`** — concatenated-drafts rate. If non-zero, the model is splitting reasoning across `<html_file>` boundaries. The new micro-probe catches it but does not prevent it; consider tightening the `<html_file>` instruction prefix in `prompts_v1.py`.
4. **`grep "probe lint" games/traces/*.jsonl | wc -l`** — bad-probe rate. If chronic, the probe-quality nudge isn't being respected; consider moving `_lint_probes` findings BEFORE the iter-1 user message rather than at the diagnose-then-fix turn.
5. **`grep "VISION JUDGE.*progress.*no" games/traces/*.jsonl | wc -l`** vs. **`grep "<confirm_done/>" games/traces/*.jsonl | wc -l`** — how often does a session ship while the vision judge says "no progress"? If non-trivial, vision-judge-gating-done (README future #1) is the next change to land.

---

## VLM image-paste — what's possible today, what isn't

**Today:**
- Pasting an image directly into the TUI Input field is not possible. Textual runs in a terminal and the Input widget only accepts text; images aren't representable in a terminal paste buffer.
- `/check with <model>` reads the latest *Chromium screenshot*, not a user-provided image.
- The agent has `_next_image_bytes` plumbing ([agent.py:1175, 2785-2828](agent.py:1175)) that attaches an image to the next user turn for VLM models. Until today this was only used internally for screenshots.

**Now (just added):**
- **`/ref <path/to/image.png>`** ([chat.py](chat.py)) — load a local file (PNG / JPEG / WebP, ≤ 4 MB), validate magic bytes, stash on `self.agent._next_image_bytes`. The next user message (typed plainly, no slash) goes through with the image attached. On the next iter:
  - VLM model present → the image is included in the chat-template render alongside the user text. The model "sees" both.
  - Text-only model active → the bytes are silently dropped (with a yellow warning at `/ref` time so you know to `/load <vlm-name>` first).

**Workflow for "make the game look like this":**
```
> /ref ~/Desktop/reference.png
/ref: attached reference.png (PNG, 247 KB) to the next user turn.
> make the game's overall look match this — colors, character proportions, vibe
```

The reference image only attaches once. Re-run `/ref` for each turn that needs it (single-shot to avoid silently re-attaching an old image when the conversation has moved on).

**What's still NOT possible:**
- Drag-and-drop of an image into the terminal — that requires a desktop-app shell, not the TUI. (The terminal renders the dropped file's *path*, which `/ref` accepts directly — that's the supported flow.)
- Multi-image attachment (only one image per turn until plumbing in `_stream` supports a list at the user-text level).
- Pasting from the macOS Universal Clipboard (the bytes never reach the terminal; the OS gives the terminal only the cursor position).

**To add drag-and-drop or clipboard paste later:** would require either (a) a non-Textual UI shell (Electron-style desktop wrapper), or (b) a small helper script that watches the macOS pasteboard for image bytes, writes them to a temp file, and runs `/ref` on that path via the chat's command socket (no socket today, would need adding).

---

## How to add a new playbook bullet

If a new failure shape emerges, hand-add a bullet to `memory/playbook.jsonl`:

1. Read 1-3 raw `.jsonl` traces that exhibit the shape. Identify the failing pattern.
2. Append a JSON line with the bullet schema:
   ```python
   {"id": "kebab-case-id", "content": "...", "tags": [...], "helpful": 0, "harmful": 0, "source": "seed", "created_at": "..."}
   ```
3. Verify retrieval picks it up on a goal that should match — `memory.Playbook` reads the file at session start, so just run a fresh session.

---

## Standing rules — repeat-back so you don't break them

These are from the user's standing memory:

1. **Tune the agent, not the model.** Don't propose "use a bigger model" as a fix. "Tune" means prompt / retrieval / playbook / probes / loop changes.
2. **No hardcoded genre lists.** Retrieval, probes, skeletons stay genre-free. Modality keyword detectors (`_detect_art_intent`, `_detect_3d_intent`) describe rendering shape, not subject matter.
3. **All code self-contained in `Agent_learning/`.** No sibling-repo `sys.path` injections.
4. **Visible Chromium by default.** TUI keeps `headless=False`. CLI keeps `--headless` for unattended.
5. **Asteroids is the canonical regression check.** Ship direction (`vx = cos(angle) * speed`) and irregular-polygon asteroids must still pass after every change. Run with `coder.py "asteroids" --max-iters 4 --best-of-n 1 --headless`.
6. **Never silently call a cloud model.** Cloud calls require explicit invocation (`/check with <claude-model>`). The agent loop never auto-upgrades.

---

## Files this doc tracks (so future-me knows where to look)

- [ollama_io.py](ollama_io.py) — `RepetitionDetector`, `StreamResult`
- [backend.py](backend.py) — MLX path's `StreamResult` construction
- [tools.py](tools.py) — `run_micro_probes` (elision, duplicate-decl, API allowlist, asset paths)
- [agent.py](agent.py) — `_no_usable_code_fallback`, `_lint_probes`, `_probes_referencing_unassigned_props`, format-rejection handling, `_last_stream_loop_*` state, post-materialize lint pass, `fix_instruction` consumer
- [chat.py](chat.py) — `_cmd_attach_ref_image` (`/ref`), `_cmd_check` (multi-vendor `/check`)
- [vision_judge.py](vision_judge.py) — `_cloud_vendor`, `_openai_judge` (new), `_anthropic_judge`, `_resolve_local_mlx_vlm`
- [tests/test_microprobes.py](tests/test_microprobes.py) — new dotted-elision + duplicate-decl tests
- [tests/test_bloat_detectors.py](tests/test_bloat_detectors.py) — Window-4 adjacency test + adjusted Windows-1/2 fixtures
- [tests/test_check_routing.py](tests/test_check_routing.py) — vendor routing for `/check`
