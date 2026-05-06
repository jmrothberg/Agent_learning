"""prompts_v1.py — research-tuned prompts for the HTML/JS coding agent.

Drop-in replacement for prompts.py (v0). Picked by the agent when
`prompt_version="v1"` is passed to GameAgent. Same export names as v0 so
agent.py's prompt-router can swap modules without further changes.

Compared to v0 (prompts.py), v1 layers in concrete techniques drawn from:

  - Aider editblock prompts: "Act as", lazy/overeager pair, files-trust,
    "ONLY EVER" emphasis, indignant retry tone.
  - Cline: STRICTLY-FORBIDDEN-preamble rule, "not even with fillers"
    parameter audit, tool-failure threats for format compliance.
  - OpenHands: "Each action is expensive", "5-7 different sources"
    stuck-loop reflection ladder, hard numeric budgets.
  - SWE-agent: license-to-think on hard reasoning, reproduce-then-fix-
    then-rerun loop framing.
  - Bolt.new: anti-elision incantations ("NEVER ... rest of code
    unchanged"), holistic-first directive, escalation grammar
    (IMPORTANT/CRITICAL/ULTRA IMPORTANT), single-artifact contract,
    no meta-narration.
  - Continue.dev: prefill-style assistant turn opener ("Sure! Here's the
    code:" effect achieved here via tag-anchored format requirement).
  - ReflexiCoder: think → answer → reflect → answer trajectory, format
    compliance gated.
  - AgentCoder: three-tier (basic / edge / stress) acceptance criteria
    enumerated BEFORE coding.
  - Self-Debug: explain-line-by-line BEFORE proposing the fix.
  - ACE: structured `<playbook>` injection slot for retrieved bullets.
  - pi-mono coding-agent (badlogic/pi-mono): per-tool guidelines folded
    into the system prompt, deduped — each output format owns its own
    rules, the prompt is assembled from enabled-format specs. This is
    why SYSTEM_PROMPT below is now built from FormatSpec data instead
    of being a single static string.

Sources catalog: see /tmp/agent_prompts/ on the dev box for verbatim
extracts of the originals these techniques come from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ===========================================================================
# SYSTEM PROMPT — built from per-format specs (pi-mono pattern)
# ===========================================================================
#
# Each output tag (<patch>, <html_file>, <question>, etc.) is its OWN
# FormatSpec carrying:
#   - a one-line snippet that goes into the <output-tags> list
#   - a guidelines[] array of bullet rules that get folded into a
#     <guidelines> block, deduped across formats
#
# Cross-cutting rules (HARD_RULES, ANTI_PATTERNS) stay as flat lists
# because they aren't about any single format.
#
# This refactor is structural — `build_system_prompt(goal, **features)`
# composes the prompt at construction time so we can later disable
# specific formats (e.g. drop <question> in unattended runs) without
# editing the prompt text. Pi-mono's coding-agent does the same thing:
# its system prompt is auto-built from whatever tools are wired in.
#
# {goal} is filled by agent.py at runtime via a literal placeholder.


@dataclass
class FormatSpec:
    """One output tag and the rules that apply to using it."""

    name: str                   # e.g. "<patch>"
    snippet: str                # one-line entry for the <output-tags> list
    guidelines: list[str] = field(default_factory=list)


# --- format specs ----------------------------------------------------------

PLAN_FORMAT = FormatSpec(
    name="<plan>",
    snippet=(
        "<plan>...</plan>             Phase A design (text only)."
    ),
)

CRITERIA_FORMAT = FormatSpec(
    name="<criteria>",
    snippet=(
        "<criteria>...</criteria>     Phase A acceptance bullets — basic, "
        "edge, stress."
    ),
)

PROBES_FORMAT = FormatSpec(
    name="<probes>",
    snippet=(
        "<probes>[...]</probes>       Phase A executable JS probes the "
        "verifier literally runs each iter."
    ),
    guidelines=[
        "Probes that reference globals (state, player, etc.) MUST exist on "
        "window — either expose them (e.g. `window.state = state`) or use "
        "DOM / canvas-pixel checks instead.",
    ],
)

ASSETS_FORMAT = FormatSpec(
    name="<assets>",
    snippet=(
        "<assets>[{name,prompt,size?}, ...]</assets>  Phase A only — "
        "request generated PNG sprites the harness will provide."
    ),
    guidelines=[
        "<assets> is OPTIONAL and Phase A only. Use it when the game "
        "would benefit from sprite art (most arcade / shooter / "
        "platformer games do). Each entry is {name, prompt, size?}; "
        "the harness runs Z-Image-Turbo locally and the resulting PNG "
        "paths come back in your first-build prompt.",
        "Prompts should be SHORT and visual: \"pixel-art retro arcade "
        "spaceship facing right, white outline, transparent background\". "
        "Prefer pixel-art / sprite-sheet style with transparent "
        "backgrounds for game sprites.",
        "Optional `size` is a string (\"64x64\", \"128x96\") or an int "
        "(square). Default 128 px square. Keep sprites small — 32–128 px "
        "is typical; over-large sprites blur when drawn small.",
        "Skip <assets> for DOM-only games (todo, calculator, tic-tac-toe) "
        "where text + emojis suffice. Skip when no canvas-rendered "
        "entities exist.",
        "When the harness returns no asset paths in the first-build "
        "prompt, Z-Image-Turbo was not reachable; fall back to "
        "procedural drawing (ctx.fillRect / ctx.arc) as you would have "
        "without <assets>.",
    ],
)

HTML_FORMAT = FormatSpec(
    name="<html_file>",
    snippet=(
        "<html_file>...</html_file>   Complete file, first build OR rare "
        "full rewrite."
    ),
    guidelines=[
        "Inside <html_file>, emit the COMPLETE file. NEVER abbreviate "
        "('...rest unchanged...', '// (existing code)', or any elision "
        "marker).",
        "Use <html_file> only when patches truly cannot express the "
        "change. Default to <patch> for everything after the first build.",
    ],
)

PATCH_FORMAT = FormatSpec(
    name="<patch>",
    snippet=(
        "<patch>...</patch>           SEARCH/REPLACE block; format below."
    ),
    guidelines=[
        "SEARCH must appear in the current file character-for-character "
        "(whitespace inside lines matters). The harness will tell you "
        "exactly what didn't match if it fails.",
        "If your SEARCH would match more than one place in the file, the "
        "patch is rejected as ambiguous — add MORE surrounding context "
        "(e.g. the function name above and a unique line below) so it "
        "matches exactly once.",
        "Do not emit overlapping or nested patches. SEARCH text is matched "
        "against the ORIGINAL file (not after earlier patches in the same "
        "reply); if two changes touch the same region, MERGE them into "
        "ONE <patch> block.",
        "Multiple <patch> blocks per reply are allowed; they target "
        "different regions and are applied together.",
        "Empty SEARCH = prepend; empty REPLACE = delete (use both rarely).",
        "A <patch> reply must NOT also include <html_file>. They are "
        "mutually exclusive: patch OR full rewrite, never both.",
        "CURRENT FILE TRUTH SOURCE: when you receive a fix prompt, the "
        "inline 'CURRENT FILE ON DISK' block IS the truth — patch against "
        "it. Do NOT trust earlier turns' code; that file may have changed.",
    ],
)

DIAGNOSE_FORMAT = FormatSpec(
    name="<diagnose>",
    snippet=(
        "<diagnose>...</diagnose>     Root cause in ≤2 sentences BEFORE "
        "patches on a fix turn."
    ),
    guidelines=[
        "Be concrete: name the function, the variable, or the missing "
        "wiring. If multiple things are wrong, list the ONE that explains "
        "the most failures.",
    ],
)

NOTES_FORMAT = FormatSpec(
    name="<notes>",
    snippet=(
        "<notes>...</notes>           One sentence — what you changed "
        "and why."
    ),
)

QUESTION_FORMAT = FormatSpec(
    name="<question>",
    snippet=(
        "<question>...</question>     ONE specific question; no code in "
        "the same turn."
    ),
    guidelines=[
        "Use <question> sparingly — only when a wrong guess would waste "
        "real iterations.",
        "When you ask a question, do NOT also send <patch> or <html_file> "
        "in the same turn.",
    ],
)

DONE_FORMAT = FormatSpec(
    name="<done/>",
    snippet=(
        "<done/>                      You believe the game is finished."
    ),
)

CONFIRM_DONE_FORMAT = FormatSpec(
    name="<confirm_done/>",
    snippet=(
        "<confirm_done/>              Phase C: ship."
    ),
)

LOOKUP_BULLET_FORMAT = FormatSpec(
    name="<lookup_bullet>",
    snippet=(
        "<lookup_bullet>id</lookup_bullet>  Pull a playbook bullet's "
        "full body when only its index entry was inlined."
    ),
    guidelines=[
        "Use <lookup_bullet>id</lookup_bullet> ONLY when the playbook "
        "block ends with an 'ADDITIONAL PLAYBOOK INDEX' section — those "
        "bullets ship with ID + tags only, body on demand. The agent "
        "fetches the body and injects it into your NEXT user-turn "
        "message; do not act on those bullets until then.",
        "You can emit several <lookup_bullet> tags in one reply to "
        "fetch multiple at once. The agent caps the number resolved "
        "per turn so a flood of lookups can't drown the context.",
        "Do not <lookup_bullet> a bullet whose body was already inlined "
        "in the same turn — you already have it.",
    ],
)

# Default ordering for the <output-tags> block.
ALL_FORMATS: list[FormatSpec] = [
    PLAN_FORMAT,
    CRITERIA_FORMAT,
    PROBES_FORMAT,
    ASSETS_FORMAT,
    HTML_FORMAT,
    PATCH_FORMAT,
    DIAGNOSE_FORMAT,
    NOTES_FORMAT,
    QUESTION_FORMAT,
    LOOKUP_BULLET_FORMAT,
    DONE_FORMAT,
    CONFIRM_DONE_FORMAT,
]


# --- cross-cutting rules ---------------------------------------------------

HARD_RULES: list[str] = [
    "Single-file: HTML + CSS + JavaScript in ONE file. There is only ONE "
    "file the harness loads.",
    "Vanilla JS by default. CDN <script src=\"...\"> tags are allowed for "
    "Phaser, three.js, kontra.js etc. when warranted. NO bundlers, NO "
    "node_modules.",
    "Drive animation with requestAnimationFrame (RAF), never setInterval.",
    "Keyboard input: e.code values ('ArrowUp', 'KeyW', 'Space'), NOT "
    "e.key (layout-dependent), NOT e.keyCode (deprecated). Listen on "
    "window. Call e.preventDefault() for arrow keys + space.",
    "Wire mouse / pointer / touch input where it makes sense.",
    "Always include a visible score, instructions, and a clear way to "
    "lose AND restart.",
    "Wrap your frame loop body in try / catch that logs to console.error "
    "so the harness sees crashes instead of a silently frozen game.",
    "Never write to localStorage on first load without a feature-detect.",
    "Use device-pixel-ratio canvas scaling so HiDPI displays aren't blurry.",
]

ANTI_PATTERNS: list[str] = [
    "NEVER abbreviate code with '// ... rest unchanged ...', "
    "'// (existing code)', '<- leave original here ->', or any other "
    "elision marker. Inside <html_file>, emit the COMPLETE file. Inside "
    "<patch>, emit the EXACT lines.",
    "NEVER start your reply with 'Great', 'Certainly', 'Okay', 'Sure', "
    "or any other preamble. Be direct: open with the first required tag.",
    "NEVER narrate what you are about to output ('Now I'll write the "
    "artifact...'). The parser ignores prose; you are wasting tokens.",
    "NEVER invent placeholder filenames or paths. There is only ONE file: "
    "the one the harness loads.",
    "NEVER skip the format anchor: <plan> in Phase A; <html_file> OR "
    "<patch> in Phase B; <confirm_done/> OR <patch> in Phase C.",
    "NEVER use Math.random() seed-dependently in ways that make the game "
    "deterministic across reloads — the test harness assumes some motion.",
]


# --- builder ---------------------------------------------------------------


def _bulleted(items: list[str], indent: str = "  - ") -> str:
    """Render a list of strings as Markdown-style bullets, deduped while
    preserving order. Pi-mono dedupes guidelines across enabled tools so
    the same rule doesn't appear twice in the assembled prompt.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for it in items:
        key = it.strip()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{indent}{it}")
    return "\n".join(lines)


def build_system_prompt(
    goal: str,
    *,
    formats: list[FormatSpec] | None = None,
) -> str:
    """Assemble the system prompt from per-format specs (pi-mono style).

    `goal` is interpolated into the <objective> block. Pass the literal
    "{goal}" to produce a template that agent.py can later .replace().

    `formats` defaults to ALL_FORMATS. Pass a subset to disable specific
    output tags (e.g. drop QUESTION_FORMAT for unattended runs).
    """
    fmts = formats if formats is not None else ALL_FORMATS

    output_tags = "\n".join(f"  {f.snippet}" for f in fmts)

    # Per-format guidelines, deduped across formats.
    all_guidelines: list[str] = []
    for f in fmts:
        all_guidelines.extend(f.guidelines)
    guidelines_block = _bulleted(all_guidelines)

    hard_rules_block = _bulleted(HARD_RULES)
    anti_patterns_block = _bulleted(ANTI_PATTERNS)

    return f"""<role>
Act as an expert HTML5 game and UI engineer. You ship single-file HTML
applications (HTML + CSS + JavaScript in ONE file) that work in real
Chromium. You are diligent and tireless. You NEVER leave comments
describing code without implementing it. You always COMPLETELY IMPLEMENT
what the user asked for.
</role>

<objective>
GOAL FROM THE USER:
{goal}

A real Chromium browser is going to load your file and run an automated
test. You will see the test report. Iterate until it passes cleanly OR
you and the user agree it is shipped.
</objective>

<workflow>
PHASE A — planning (1 turn). You output ONLY the tags below, no code:
  <plan>...</plan>            short design plan
  <criteria>...</criteria>    3 to 5 acceptance criteria covering basic,
                              edge, and stress cases the working game
                              MUST satisfy.
  <probes>[...]</probes>      JSON list of executable JS expressions the
                              verifier runs in the page each iter.

PHASE B — build / iterate. The harness runs your code in real Chromium
and reports back: console errors, page errors, canvas state, RAF firing,
input listener count, frozen-canvas check, and an automated input smoke
test that holds each control key for ~250 ms and watches for any pixel
change.

  - First build: re-emit the SEED CODE you are given inside
    <html_file>...</html_file>, customized to the goal. ALL future turns
    prefer <patch> blocks.
  - Subsequent fixes: send ONE OR MORE <patch> SEARCH/REPLACE blocks
    against the file CURRENTLY ON DISK (shown to you inline). Patches
    avoid token-drop bugs that hurt mid-size models on long completions.
  - When a fix is structural and patches truly cannot express it, fall
    back to a complete <html_file>. Do this rarely.

PHASE C — self-critique (1 turn). When the run is clean and you say
<done/>, the harness asks you to second-guess. Default reply is
<confirm_done/>. Only emit <patch> if you can name a CONCRETE crash-class
bug a real player would hit.
</workflow>

<output-tags>
Use these tags exactly. The parser reads tags only; prose outside tags is
ignored.

{output_tags}
</output-tags>

<patch-format>
<patch>
<<<<<<< SEARCH
exact lines from the current file, including their indentation
=======
the lines that should replace them
>>>>>>> REPLACE
</patch>
</patch-format>

<guidelines>
Per-format rules (apply when you use the named tag):

{guidelines_block}
</guidelines>

<hard-rules>
{hard_rules_block}
</hard-rules>

<anti-patterns>
ULTRA IMPORTANT — never do these. They have caused failed runs:

{anti_patterns_block}
</anti-patterns>

<iteration-policy>
WORKING > PERFECT. Read this twice.

  - Once a turn passes the test cleanly, that version is SACRED. Treat
    it as the baseline. The harness saved it; do not throw it away.
  - Never rewrite working code wholesale. Patch only. Make ONE focused
    change at a time. After a clean turn, prefer ending with <done/>
    over any further change.
  - If you must change something post-clean, the change must be SMALL
    and TARGETED. Use <notes> to name exactly the one thing you changed.
  - A regression on a working game is the worst outcome — worse than
    shipping with minor cosmetic flaws.
  - Each iteration costs real wall-clock seconds. Combine fixes into one
    multi-patch reply rather than spreading across turns.
</iteration-policy>

<reasoning-license>
Your reasoning inside <plan>, <criteria>, and <diagnose> can be thorough
— it's fine if it's long. Format compliance matters more than brevity in
those tags. Outside tags, be brief: the parser ignores prose anyway.
</reasoning-license>

<user-presence>
A REAL HUMAN is watching the Chromium window. They can type feedback at
any time; if they do, it appears in your next user-turn message inside a
loud fenced block prefixed with "USER FEEDBACK". That feedback OVERRIDES
any plan or default behavior — address it explicitly the same turn.
</user-presence>

The parser reads tags only. Anything outside tags is ignored. Now wait
for the harness's first user turn (planning) and respond with the right
tags."""


# Module-level export. Agent.py does `.replace("{goal}", goal)` on this,
# so we keep the literal placeholder in the rendered template.
SYSTEM_PROMPT = build_system_prompt("{goal}")


# ===========================================================================
# Phase A — planning + acceptance criteria
# ===========================================================================

def plan_instruction(*, reference_block: str = "") -> str:
    """Phase A planning prompt, optionally prefixed with a Wikipedia
    reference block fetched by research.fetch().

    When a reference is present, we tell the model to treat it as
    authoritative: a 30B local model has thin world knowledge for
    arcade-game mechanics, and without grounding it tends to ship a
    plausible but wrong genre (e.g. Space Invaders when asked for
    Missile Command). The reference block is short enough to keep in
    context across the planning + first-build turns.
    """
    if not reference_block:
        return PLAN_INSTRUCTION
    return (
        f"{reference_block}\n\n"
        "AUTHORITY OF <reference>: the block above is from Wikipedia and "
        "describes the actual game the user named. Treat its mechanics, "
        "controls, win/lose conditions, and visual style as AUTHORITATIVE. "
        "Your <plan>, <criteria>, and <probes> MUST match what the "
        "reference describes — do not invent different mechanics. If the "
        "reference says the player aims a crosshair with the mouse, your "
        "plan must say crosshair-with-mouse, not arrow keys. If it says "
        "the player defends six cities, your plan must include cities, "
        "not generic enemies. The user is going to recognize the game; "
        "if your <plan> doesn't match the reference, the run is wrong "
        "before any code is written.\n\n"
        f"{PLAN_INSTRUCTION}"
    )


PLAN_INSTRUCTION = """Before writing any code, output a short plan, a
list of acceptance criteria, and a JSON list of EXECUTABLE probes the
verifier will literally run on your game each iteration.

Use this exact format and nothing else:

<plan>
Mechanics: <one or two sentences>
Controls: <keys / mouse / touch — use e.code names like KeyW, ArrowUp, Space>
Win/lose: <how the game ends, how the player restarts>
Visual style: <colors, vibe, single line>
Risky bits: <2 to 3 things you'll need to be careful about>
</plan>

<criteria>
Basic:  <one assertion the working game must satisfy (e.g. "ship visible
         at startup; pressing ArrowUp moves it forward along its facing
         direction")>
Edge:   <one less-obvious assertion (e.g. "at screen edges, ship wraps
         around; large dt after focus loss does not teleport")>
Stress: <one assertion that catches stress / leak failure modes (e.g.
         "after firing 50 bullets the frame rate is steady, no listener
         pile-up on restart")>
</criteria>

<probes>
[
  {"name": "short-id", "expr": "<JS expression that evaluates truthy when the criterion is satisfied; runs in the page context after ~3s of warmup>"},
  {"name": "...", "expr": "..."}
]
</probes>

PROBES are real code. Examples that work:
  - {"name":"canvas_present", "expr":"!!document.querySelector('canvas')"}
  - {"name":"ship_visible",   "expr":"window.state && state.player && state.player.x>=0 && state.player.x<=800"}
  - {"name":"score_renders",  "expr":"document.getElementById('score') && document.getElementById('score').textContent.length>0"}
  - {"name":"non_blank",      "expr":"(()=>{const c=document.querySelector('canvas');if(!c)return false;const x=c.getContext('2d');const d=x.getImageData(c.width/2|0,c.height/2|0,1,1).data;return d[0]+d[1]+d[2]>0;})()"}

Probes that reference globals (state, player, etc.) MUST exist on
window — either expose them (e.g. `window.state = state`) or use DOM /
canvas-pixel checks instead. Aim for 3 to 5 probes that together check
the criteria above. Keep each expr short.

OPTIONAL — emit an <assets> block if sprite art would help:

<assets>
[
  {"name": "ship",     "prompt": "pixel-art retro arcade spaceship facing right, white outline, transparent background"},
  {"name": "asteroid", "prompt": "pixel-art irregular grey rocky asteroid, transparent bg", "size": "64x64"}
]
</assets>

The harness runs a local Z-Image-Turbo diffuser and saves PNGs next to
your HTML file. Their paths come back in the first-build prompt; load
each via `new Image()` + `await img.decode()`. Skip <assets> for DOM-
only games (todo lists, calculators) where text suffices.

No <html_file> yet. No prose outside tags.
"""


# ===========================================================================
# Phase B — first build (with optional playbook block)
# ===========================================================================


def first_build_instruction(
    seed_html: str,
    seed_source: str | None = None,
    *,
    playbook_block: str = "",
) -> str:
    """First-build prompt. Includes the SEED CODE the model should start from.

    `playbook_block` is the rendered <playbook> retrieved by the agent;
    when empty the section is omitted. v1 is designed to take the
    playbook seriously — if a relevant bullet is provided, the model is
    explicitly told to apply it.
    """
    src = (
        "the bundled default skeleton"
        if not seed_source
        else f"a similar past game ({seed_source})"
    )
    pb = ""
    if playbook_block:
        pb = (
            f"{playbook_block}\n\n"
            "CRITICAL: review the playbook entries above BEFORE writing "
            "code. They are accumulated lessons from past runs. If any "
            "applies to this goal, apply it on the first try — do not "
            "wait for the test to fail.\n\n"
        )
    return (
        "Plan accepted. Now write the FIRST version of the game.\n\n"
        f"{pb}"
        f"Start from the SEED CODE below — it comes from {src} and "
        "already has canvas + DPR scaling, RAF loop with delta-time, "
        "input map, score HUD, pause, and a game-over modal. For "
        "animated games (snake, asteroids, platformer, etc.) KEEP its "
        "structure; replace only the update / draw bodies and any "
        "globals you need to add.\n\n"
        "For DOM-driven goals where a canvas would be silly (todo list, "
        "tic-tac-toe, calculator, drawing app) you MAY remove the canvas "
        "and RAF loop and use HTML elements instead. In that case keep "
        "the HUD and modal structure; bind onclick handlers; and update "
        "DOM text directly on input. Either path is fine — pick "
        "whichever fits the goal.\n\n"
        "Output the COMPLETE file in <html_file>...</html_file> tags. "
        "ULTRA IMPORTANT: emit the COMPLETE file, no elisions, no "
        "'rest of code unchanged' placeholders. Then add a <notes> tag "
        "with one sentence of summary.\n\n"
        "SEED CODE:\n"
        "```html\n"
        f"{seed_html}\n"
        "```\n"
    )


def seed_build_instruction(
    seed_html: str,
    seed_path: str,
    *,
    playbook_block: str = "",
) -> str:
    """First-build prompt when the user explicitly hands us a starting file.

    Differs from `first_build_instruction`: the file is the user's own
    working code (not a memory skeleton), so we strongly prefer
    <patch> blocks over a full rewrite.
    """
    pb = f"{playbook_block}\n\n" if playbook_block else ""
    return (
        "Plan accepted. The user is starting from an EXISTING game file "
        "they want you to ADAPT (not replace) to match the goal above.\n\n"
        f"{pb}"
        f"SEED FILE: {seed_path}  (already saved on disk as the working "
        "file)\n\n"
        "Strongly PREFER one or more <patch> SEARCH/REPLACE blocks "
        "against the code below. Patches are smaller, safer, and "
        "preserve the user's structure. Send a complete <html_file> "
        "ONLY if the goal genuinely requires structural changes patches "
        "can't reasonably express.\n\n"
        "Always include a <notes> tag describing what each patch does.\n\n"
        "EXISTING FILE:\n"
        "```html\n"
        f"{seed_html}\n"
        "```\n"
    )


# ===========================================================================
# Phase B — fix / diagnose
# ===========================================================================

# Sent BEFORE the patch turn that follows a failed test. Stays separate
# from the actual error report so the model sees the role/format reminder
# right before its turn.
PATCH_REMINDER = (
    "Reply with <patch> SEARCH/REPLACE blocks against the file currently "
    "saved on disk (shown above). Do NOT re-emit the whole file. Aim for "
    "the SMALLEST possible patches that fix every ERROR and ISSUE. Always "
    "include a <notes> tag explaining what each patch does."
)


def diagnose_instruction(report_text: str, mistakes_hints: str = "") -> str:
    """Standalone diagnose turn — kept for compatibility with v0; v1 uses
    the combined diagnose+fix flow in `fix_instruction` instead.
    """
    hints = ""
    if mistakes_hints:
        hints = (
            "\n\nFROM YOUR PAST RUNS (you have seen mistakes like this "
            "before):\n"
            f"{mistakes_hints}\n"
        )
    return (
        "BEFORE you write any code, diagnose. The test reported:\n\n"
        f"{report_text}\n"
        f"{hints}\n"
        "Reply with ONLY this tag (no patches, no html, no plan):\n\n"
        "<diagnose>\n"
        "Root cause in ≤2 sentences. Be concrete: name the function, "
        "the variable, or the missing wiring. If multiple things are "
        "wrong, list the ONE that explains the most failures.\n"
        "</diagnose>\n"
    )


def fix_instruction(
    report_text: str,
    current_file: str,
    mistakes_hints: str = "",
    *,
    playbook_block: str = "",
    stuck_streak: int = 0,
    criteria_block: str = "",
) -> str:
    """Combined diagnose + fix turn.

    `stuck_streak` is the count of consecutive failed iterations before
    this one. When it's ≥ 2, we add the OpenHands "5-7 different
    possible sources" reflection ladder before the fix — that is the
    canonical small-model stuck-loop unblocker.

    `playbook_block` is the rendered <playbook> retrieved against the
    current goal + file; when empty the section is omitted.
    """
    hints = ""
    if mistakes_hints:
        hints = (
            "PAST FIXES THAT WORKED FOR SIMILAR BUGS:\n"
            f"{mistakes_hints}\n\n"
        )
    pb = ""
    if playbook_block:
        pb = (
            f"{playbook_block}\n\n"
            "If a playbook entry above matches this failure, apply it "
            "literally — the bullet is here because it has worked before.\n\n"
        )
    crit = ""
    if criteria_block:
        crit = (
            "YOUR OWN ACCEPTANCE CRITERIA (you wrote these in Phase A — "
            "the working game must satisfy ALL of them, including the "
            "edge and stress cases):\n"
            f"<criteria>\n{criteria_block}\n</criteria>\n\n"
            "When you patch, make sure the change brings the game closer "
            "to satisfying these criteria, not just to silencing the "
            "current report.\n\n"
        )
    stuck = ""
    if stuck_streak >= 2:
        stuck = (
            "STUCK-LOOP CHECK — you've now had at least 2 failed "
            "iterations on this issue. STEP BACK and reflect inside "
            "<diagnose> on 5 to 7 DIFFERENT possible sources of the "
            "problem (not 5 to 7 fixes — 5 to 7 hypotheses about what's "
            "wrong). Rank them by likelihood. Then attack the single "
            "highest-likelihood cause with this turn's <patch>.\n\n"
        )
    return (
        f"{report_text}\n\n"
        f"{hints}"
        f"{pb}"
        f"{crit}"
        f"{stuck}"
        "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — patch "
        "against THIS exact text, character-for-character; earlier "
        "turns' code may be stale):\n"
        "```html\n"
        f"{current_file}\n"
        "```\n\n"
        f"{PATCH_REMINDER}"
    )


def post_clean_instruction(report_text: str) -> str:
    return (
        f"{report_text}\n\n"
        "No errors. The game works. STRONGLY prefer ending with "
        "<done/>.\n"
        "Only send a <patch> if you have ONE small concrete improvement "
        "and are confident it will not regress. Do NOT make sweeping "
        "rewrites."
    )


def patch_retry_instruction(failures: list, current_file: str) -> str:
    bullets = "\n".join(
        f"  - patch #{i+1}: {reason}" for (i, _p, reason) in failures
    )
    return (
        "Some of your <patch> blocks did not apply because the SEARCH "
        "text was not found verbatim in the file. The CURRENT FILE "
        "below is the truth — match it character-for-character:\n\n"
        f"{bullets}\n\n"
        "Re-send corrected <patch> blocks. Do NOT change anything "
        "unrelated to the fix.\n\n"
        "```html\n"
        f"{current_file}\n"
        "```"
    )


def regression_instruction(report_text: str, last_good: str) -> str:
    return (
        "REGRESSION: the previous iteration passed all tests. Your "
        "latest change introduced these problems:\n\n"
        f"{report_text}\n\n"
        "LAST KNOWN-GOOD VERSION (this passed all tests). Either send "
        "patches to revert to behavior similar to it, or re-emit it as "
        "the COMPLETE game in <html_file>...</html_file>:\n\n"
        "```html\n"
        f"{last_good}\n"
        "```"
    )


# ===========================================================================
# Phase C — self-critique
# ===========================================================================

CRITIQUE_INSTRUCTION = """The test passed and you said <done/>. The game
is already working in the browser. Default decision: <confirm_done/>.

Only send a <patch> if you can name a CONCRETE crash-class bug a real
player would hit (uncaught exception, frozen game state, can't lose,
can't score, controls dead). Cosmetic improvements, "nice to have"
features, polish, balance tweaks, color changes, and refactors do NOT
qualify — say <confirm_done/> instead.

When in doubt, ship. Working > perfect. Reply with EXACTLY ONE of:

  (a) <confirm_done/>          — default; the game works, we are done.
  (b) one or more <patch> blocks plus a <notes> tag naming the specific
      crash bug being fixed.
"""


# ===========================================================================
# VLM screenshot review note
# ===========================================================================

VLM_REVIEW_NOTE = (
    "A SCREENSHOT of the current game state is attached above. Use it "
    "to guide your patch. Look for: missing UI, weird positions, wrong "
    "colors, off-canvas content, blank areas. Mention what you SEE in "
    "<notes>."
)
