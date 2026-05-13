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

SOUNDS_FORMAT = FormatSpec(
    name="<sounds>",
    snippet=(
        "<sounds>[{name,prompt,duration?,loop?}, ...]</sounds>  Request "
        "generated OGG audio for SFX and looping music. Phase A for first "
        "build; ALSO mid-session — same `name` as an existing sound "
        "re-renders that OGG in place, no code edit."
    ),
    guidelines=[
        "EMIT <sounds> when the goal mentions sound, audio, music, SFX, "
        "or has player-perceptible events (firing, hits, pickups, "
        "explosions, jumps). The harness runs Stable Audio Open locally "
        "and returns OGG paths in your first-build prompt; load each "
        "via `new Audio(path)` and play with `.play()` (browser auto-"
        "play rules require a user gesture first; use the loader "
        "pattern injected in the first-build prompt).",
        "MID-SESSION RE-RENDER: when the user asks to change the SOUND of "
        "an existing audio entry ('make the laser punchier', 'remake the "
        "music', 'redo the explosion sfx'), emit <sounds>[{name:"
        "<existing_name>, prompt:<new audio prompt>}]</sounds> — the "
        "harness rebuilds that OGG in place (same key in `SOUNDS`, same "
        "`new Audio(...)` call, no JS edit). Other sounds and the game "
        "code stay untouched. To CHANGE a sound, re-render; to ADD a new "
        "audio entry, emit a new name.",
        "Each entry is {name, prompt, duration?, loop?}. `duration` is "
        "seconds (default 1.0; capped at 12.0). For SFX use 0.2-1.5 s; "
        "for looping background music use 8-12 s with `\"loop\": true` "
        "so <audio loop> can repeat it without seam clicks.",
        "Prompts should be SHORT and audio-descriptive: \"short retro "
        "arcade laser shot, 8-bit synth blip\", \"short pixelated "
        "explosion, 8-bit boom\", \"loopable chiptune background, 90 "
        "bpm, upbeat\". Prefer 8-bit / chiptune / synth descriptors for "
        "arcade games; foley descriptors (footsteps on gravel, metal "
        "clang) work for realistic games.",
        "SKIP <sounds> for pure-DOM apps where audio adds nothing — "
        "calculators, color pickers, todo lists. Otherwise, default to "
        "EMITTING it: silent games feel cheaper than the same game "
        "with audio.",
        "When the harness returns no sound paths in the first-build "
        "prompt, the audio diffuser was not reachable on this machine; "
        "only THEN ship a silent game.",
    ],
)

ASSETS_FORMAT = FormatSpec(
    name="<assets>",
    snippet=(
        "<assets>[{name,prompt,size?}, ...]</assets>  Request generated "
        "PNG sprites. Phase A for first build; ALSO mid-session — same "
        "`name` as an existing asset re-renders that PNG in place, no "
        "code edit."
    ),
    guidelines=[
        "EMIT <assets> for any canvas-rendered game with visual "
        "entities the player sees: characters, enemies, projectiles, "
        "pickups, terrain tiles, decorations. Most arcade / shooter / "
        "platformer / RPG / racing / adventure games qualify. The "
        "harness runs Z-Image-Turbo locally and the PNG paths come "
        "back in your first-build prompt; load each via `new Image()` "
        "+ `await img.decode()` (see playbook bullet image-load-race).",
        "MID-SESSION RE-RENDER: when the user asks to change the LOOK of "
        "an existing sprite ('make the tail rounder', 'redraw the "
        "player', 'change the alien art'), emit <assets>[{name:"
        "<existing_name>, prompt:<new visual prompt>}]</assets> — the "
        "harness rebuilds that PNG in place (same key in `ASSETS`, same "
        "drawSprite call, no JS edit). Other assets and the game code "
        "stay untouched. Do NOT swap an existing drawSprite(name,…) "
        "call for procedural ctx.* drawing in response to an art-change "
        "request; that loses the sprite path and almost always "
        "regresses. To CHANGE the look, re-render; to ADD a new "
        "visual entity, emit a new name.",
        "MIX sprites and procedural drawing freely. Sprites are right "
        "for STATIC visual character (player, enemies, weapons, terrain "
        "tiles, pickups). Procedural drawing is right for DESTRUCTIBLE "
        "or STATE-RICH entities where runtime visual state must show "
        "through (bunkers crumbling brick-by-brick, cracks appearing "
        "on damage, health bars filling up, particle trails). Don't "
        "force everything into one bucket — the typical game has "
        "sprite walls/enemies AND procedural cell-based destructibles.",
        "Prompts should be SHORT and visual: \"pixel-art retro arcade "
        "spaceship facing right, white outline, transparent background\". "
        "Prefer pixel-art / sprite-sheet style with transparent "
        "backgrounds for game sprites.",
        "Optional `size` is a string (\"64x64\", \"128x96\") or an int "
        "(square). Default 128 px square. Keep sprites small — 32–128 px "
        "is typical; over-large sprites blur when drawn small.",
        "ANIMATION FRAMES: when you need a coherent sprite sequence "
        "(walk cycle, attack windup, flap, idle bob), declare frame 1 "
        "normally, then add `\"from_image\": \"<name-of-frame-1>\"` and "
        "`\"strength\": 0.35-0.55` on subsequent frames. The harness "
        "runs SD-Turbo img2img with the previous frame as the init "
        "image, so frame 2 inherits frame 1's silhouette + palette and "
        "only the pose changes. Without `from_image` each frame is an "
        "independent txt2img and the result looks like two different "
        "characters. Example: "
        "{\"name\":\"alien_walk1\", \"prompt\":\"8-bit alien, legs together\"}, "
        "{\"name\":\"alien_walk2\", \"prompt\":\"8-bit alien, legs apart\", "
        "\"from_image\":\"alien_walk1\", \"strength\":0.45}.",
        "SKIP <assets> ONLY for pure-DOM apps where text + emojis are "
        "enough — todo lists, calculators, tic-tac-toe, color pickers. "
        "If the canvas has any rendered entity with visual character, "
        "EMIT <assets>; do not pre-emptively decide procedural drawing "
        "would be 'good enough'.",
        "When the harness returns no asset paths in the first-build "
        "prompt, Z-Image-Turbo was not reachable on this machine; only "
        "THEN fall back to procedural drawing (ctx.fillRect / ctx.arc).",
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
        # Markdown-fence trap. Models trained with heavy markdown in their
        # corpus reflexively close fenced code blocks even when no fence
        # was opened — observed on DeepSeek-V4 in trace 20260512_153449,
        # where a stray ``` truncated the HTML body before </script>.
        "The <html_file> body is raw HTML, not a markdown code block. "
        "Do NOT wrap it in ```html ... ``` fences — a stray closing ``` "
        "inside the body will truncate your file at that point.",
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
        "<diagnose>...</diagnose>     ONE root cause in ≤2 sentences "
        "BEFORE patches on a fix turn."
    ),
    guidelines=[
        "Name EXACTLY ONE root cause: the function, the variable, or "
        "the missing wiring responsible for the most failures. Do NOT "
        "enumerate 5-7 hypotheses. Do NOT emit a numbered or bulleted "
        "list of possibilities. ≤2 sentences total.",
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
    SOUNDS_FORMAT,
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
    "Grids, mazes, tilemaps, dungeons, level data: if the layout is larger "
    "than 16 rows OR 16 columns, write a deterministic seeded generator "
    "function (e.g. function buildMaze(w, h, seed){...}) instead of inlining "
    "a literal 2D array. Inlining large literals causes local LLMs to fall "
    "into token-repetition loops mid-stream — the harness will reject the "
    "response and the work is wasted. Reset any module-level timer, "
    "counter, or cooldown variable (e.g. lastFireTime, lastSpawnTime) "
    "inside reset() so restart actually restarts.",
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
    model_class: str = "auto",
) -> str:
    """Assemble the system prompt from per-format specs (pi-mono style).

    `goal` is interpolated into the <objective> block. Pass the literal
    "{goal}" to produce a template that agent.py can later .replace().

    `formats` defaults to ALL_FORMATS. Pass a subset to disable specific
    output tags (e.g. drop QUESTION_FORMAT for unattended runs).

    `model_class` (Stop-Losing-To-OneShot todo #6): "mid" trims the
    <anti-patterns> block. Mid-tier models (qwen3.6:27b, gpt-oss:20b
    class) sometimes treat anti-patterns as features-to-add by mistake;
    dropping the block surfaces the goal earlier in attention and frees
    ~600-1000 chars of context. "small" goes further: drops the
    <assets>, <sounds>, <lookup_bullet> tags entirely (so the assets
    pipeline is opt-out for the small-model class), collapses
    <workflow> to a 4-line summary, and removes the
    <reasoning-license> + <user-presence> blocks. Target ≤ 6 KB so
    qwen2.5-coder-32B / deepseek-coder-33B class models can spend their
    capacity on the game, not on the schema. "large" / "auto" (default)
    keeps the full prompt unchanged.
    """
    is_small = (model_class == "small")
    if formats is None:
        if is_small:
            # Drop sprite + sound generation pipelines and the on-demand
            # playbook lookup mechanism. The model writes a self-contained
            # game instead of orchestrating an asset pipeline.
            _SMALL_DROP = {"<assets>", "<sounds>", "<lookup_bullet>"}
            fmts = [f for f in ALL_FORMATS if f.name not in _SMALL_DROP]
        else:
            fmts = ALL_FORMATS
    else:
        fmts = formats

    output_tags = "\n".join(f"  {f.snippet}" for f in fmts)

    # Per-format guidelines, deduped across formats.
    all_guidelines: list[str] = []
    for f in fmts:
        all_guidelines.extend(f.guidelines)
    guidelines_block = _bulleted(all_guidelines)

    hard_rules_block = _bulleted(HARD_RULES)
    anti_patterns_block = _bulleted(ANTI_PATTERNS)
    # Mid-tier trim: drop the <anti-patterns> block entirely. The
    # <hard-rules> block above already covers the highest-priority
    # invariants; ANTI_PATTERNS is largely "don't do X" cautionary
    # detail that 27B-class models tend to enumerate AT us in <notes>
    # instead of internalizing. Small-tier inherits the trim.
    skip_anti_patterns = (model_class in ("mid", "small"))

    workflow_full = """<workflow>
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
</workflow>"""
    workflow_small = """<workflow>
Phase A: emit <plan>, <criteria>, <probes>. No code.
Phase B iter 1: emit one complete <html_file>.
Phase B iter 2+: emit <patch> SEARCH/REPLACE blocks against the file on disk.
Phase C: reply with <confirm_done/> unless a real player would hit a crash.
</workflow>"""
    workflow_block = workflow_small if is_small else workflow_full

    iter_policy_full = """<iteration-policy>
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
</iteration-policy>"""
    iter_policy_small = """<iteration-policy>
WORKING > PERFECT. After a clean turn, ship with <done/>. Patches only;
ONE focused change per turn. No regressions on a working game.
</iteration-policy>"""
    iter_policy_block = iter_policy_small if is_small else iter_policy_full

    extras_block = "" if is_small else """<reasoning-license>
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

"""

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

{workflow_block}

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

{"" if skip_anti_patterns else f'''<anti-patterns>
ULTRA IMPORTANT — never do these. They have caused failed runs:

{anti_patterns_block}
</anti-patterns>'''}

{iter_policy_block}

{extras_block}The parser reads tags only. Anything outside tags is ignored. Now wait
for the harness's first user turn (planning) and respond with the right
tags."""


# Module-level export. Agent.py does `.replace("{goal}", goal)` on this,
# so we keep the literal placeholder in the rendered template.
SYSTEM_PROMPT = build_system_prompt("{goal}")


# ===========================================================================
# Phase A — planning + acceptance criteria
# ===========================================================================

# Modality keywords that signal the user wants generated art. Stays
# genre-free per the project rule (no asteroids/snake/etc.) — these are
# all about output character, not subject matter. When any of these
# words appear in the goal, plan_instruction() escalates the <assets>
# directive from "expected" to "REQUIRED THIS TURN" so qwen3.6 / gpt-oss
# size models can't politely skip it.
_ART_KEYWORDS = frozenset({
    "art", "sprite", "sprites", "spritesheet", "graphic", "graphics",
    "visual", "visuals", "pixel", "pixels", "pixelart", "image", "images",
    "texture", "textures", "asset", "assets", "draw", "drawing", "drawn",
    "icon", "icons", "render", "rendering", "illustration", "illustrations",
    "artwork", "looks", "look", "gorgeous", "beautiful", "pretty",
    "cool", "stunning", "polished",
})


# Modality keywords that signal the goal needs a 3D rendering technique.
# Genre-free per the project rule — these all describe rendering shape
# ("first-person view", "raycaster", "voxel"), not a specific game. The
# goal is to nudge the model toward a real 3D library (three.js / babylon
# / PlayCanvas via CDN) rather than hand-rolling a raycaster in 12 KB,
# which is what produced the unplayable doom session.
_3D_KEYWORDS = frozenset({
    "3d", "three", "threejs",
    "first-person", "firstperson", "fps",
    "raycaster", "raycasting", "raycast",
    "voxel", "voxels",
    "wolfenstein", "doom-like", "doomlike", "minecraft-like", "minecraftlike",
    "perspective",
})


# Modality keywords that signal the goal needs generated AUDIO. Genre-
# free per the project rule — these all describe rendering shape (sound
# events, music, sfx) rather than subject matter. When any appears in
# the goal, plan_instruction() escalates the <sounds> directive to
# REQUIRED so smaller models don't politely skip the audio pipeline.
_AUDIO_KEYWORDS = frozenset({
    "sound", "sounds", "audio", "music", "sfx",
    "chiptune", "soundtrack", "soundeffect", "soundeffects",
    "noise", "noisy", "noises",
    "beep", "beeps", "boom", "blip",
    "loud", "silent",   # "silent" included so "make it not silent" works
    "soundscape", "ambient",
    "score",            # ambiguous (could mean game-score) — kept; cheap miss
})


def _detect_audio_intent(goal: str) -> list[str]:
    """Return a list of audio-modality keywords found in `goal`. Empty
    list means no intent detected; non-empty triggers a stronger
    <sounds> directive in the planning prompt.

    Same tokenization shape as `_detect_art_intent`: lowercased word
    splitting, no genre lookup, no NER. The match list is surfaced
    back into the prompt so the model can see exactly which words
    triggered the escalation.
    """
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _AUDIO_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _detect_3d_intent(goal: str) -> list[str]:
    """Return a list of 3D-modality keywords found in `goal`. Empty list
    means the goal is plain 2D / DOM-only and needs no 3D nudge.
    Single-token matches; multi-token phrases like "first person" are
    detected by joining adjacent words and checking the joined form.

    Tokenizer keeps digits so "3D" is matched as "3d" (not stripped to
    "d"). Lowercased so the keyword set can stay all-lowercase.
    """
    if not goal:
        return []
    import re
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", goal)]
    out: list[str] = []
    seen: set[str] = set()
    # Single-word match.
    for w in words:
        if w in _3D_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    # Two-word join match for "first person", "doom like", etc.
    for i in range(len(words) - 1):
        j = words[i] + words[i + 1]
        if j in _3D_KEYWORDS and j not in seen:
            seen.add(j)
            out.append(j)
        # And with hyphen variant
        jh = words[i] + "-" + words[i + 1]
        if jh in _3D_KEYWORDS and jh not in seen:
            seen.add(jh)
            out.append(jh)
    return out


def _detect_art_intent(goal: str) -> list[str]:
    """Return a list of art-modality keywords found in `goal`. Empty
    list means no intent detected; non-empty triggers a stronger
    <assets> directive in the planning prompt.

    Tokenization is just lowercased word splitting — no genre lookup,
    no NER, nothing model-dependent. The match list is surfaced back
    into the prompt so the model can see exactly which words triggered
    the escalation.
    """
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _ART_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def plan_instruction(*, reference_block: str = "", goal: str = "") -> str:
    """Phase A planning prompt, optionally prefixed with a Wikipedia
    reference block fetched by research.fetch().

    When a reference is present, we tell the model to treat it as
    authoritative: a 30B local model has thin world knowledge for
    arcade-game mechanics, and without grounding it tends to ship a
    plausible but wrong genre (e.g. Space Invaders when asked for
    Missile Command). The reference block is short enough to keep in
    context across the planning + first-build turns.

    When `goal` contains art-modality keywords (sprite, graphic, art,
    pixel, image, …), we escalate the <assets> requirement — the model
    must emit an <assets> block this turn or be wrong. Stops qwen3.6
    from politely skipping the asset pipeline when the user explicitly
    asks for sprites.
    """
    art_keywords = _detect_art_intent(goal)
    art_nudge = ""
    if art_keywords:
        kws = ", ".join(repr(k) for k in art_keywords)
        art_nudge = (
            "\n\nART INTENT DETECTED — your goal mentions "
            f"{kws}. The user EXPLICITLY wants generated art. You MUST "
            "emit an <assets> block this turn with one entry per visual "
            "entity in the game (every character, enemy, projectile, "
            "pickup, decoration the player will see). Skipping <assets> "
            "and falling back to procedural ctx.fillRect drawing IS A "
            "FAILURE for this goal — the user will notice the difference "
            "between bare squares and real sprite art. ULTRA IMPORTANT: "
            "do not omit <assets>; do not say \"I'll draw procedurally\"; "
            "do not assume the diffuser is unavailable (the harness will "
            "tell you if it is).\n"
        )

    threed_keywords = _detect_3d_intent(goal)
    threed_nudge = ""
    if threed_keywords:
        kws = ", ".join(repr(k) for k in threed_keywords)
        threed_nudge = (
            "\n\n3D INTENT DETECTED — your goal mentions "
            f"{kws}. STRONGLY PREFER three.js (or babylon.js / "
            "PlayCanvas) via CDN over hand-rolling a raycaster or "
            "writing 3D math from scratch. A few hundred lines of "
            "`THREE.WebGLRenderer` + `Scene` + sprite-billboard "
            "`Mesh`es will outperform 1000 lines of raycaster code, "
            "and the user gets actual 3D fidelity.\n"
            "\n"
            "Suggested CDN imports (one is plenty):\n"
            "  <script src=\"https://cdn.jsdelivr.net/npm/three@0.160/build/three.min.js\"></script>\n"
            "  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/babylonjs/6.32.0/babylon.min.js\"></script>\n"
            "\n"
            "Pair three.js with <assets>: load the generated PNGs as "
            "`THREE.TextureLoader().load(path)` and apply them to "
            "PlaneGeometry walls, SpriteMaterial billboards (for enemies/"
            "weapons), or BoxGeometry tiles. That's how you ship real "
            "Doom-shaped output instead of a 12 KB raycaster sketch.\n"
            "\n"
            "ONLY hand-roll a raycaster if the goal explicitly says "
            "\"raycaster from scratch\" or \"no libraries\". Otherwise "
            "use the library.\n"
        )

    audio_keywords = _detect_audio_intent(goal)
    audio_nudge = ""
    if audio_keywords:
        kws = ", ".join(repr(k) for k in audio_keywords)
        audio_nudge = (
            "\n\nAUDIO INTENT DETECTED — your goal mentions "
            f"{kws}. The user EXPLICITLY wants generated audio. You MUST "
            "emit a <sounds> block this turn with one entry per "
            "discrete audible event the player will hear (firing, hits, "
            "pickups, explosions, jumps, level-clear) plus optionally a "
            "looping music track. Skipping <sounds> and shipping a "
            "silent game IS A FAILURE for this goal — the user will "
            "notice. ULTRA IMPORTANT: do not omit <sounds>; do not "
            "assume the audio diffuser is unavailable (the harness will "
            "tell you if it is).\n"
        )

    body = PLAN_INSTRUCTION + art_nudge + threed_nudge + audio_nudge

    if not reference_block:
        return body
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
        f"{body}"
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
  - {"name":"non_blank",      "expr":"(()=>{const c=document.querySelector('canvas');if(!c||!c.width||!c.height)return false;try{return c.toDataURL().length>200;}catch(e){return true;}})()"}

Probes that reference globals (state, player, etc.) MUST exist on
window — either expose them (e.g. `window.state = state`) or use DOM /
canvas-pixel checks instead. Aim for 3 to 5 probes that together check
the criteria above. Keep each expr short.

PROBE ROBUSTNESS — make probes test STRUCTURE, not specific
positions or rendered details that change frame-to-frame:
  - GOOD: `typeof window.state.x === 'number'` (the ball has a
    coordinate)
  - BAD:  testing a single pixel at canvas center for red — a moving
    object isn't at center most frames; test fails the moment the
    game animates correctly.
  - GOOD: `document.getElementById('score').textContent.length > 0`
    (the score is rendered)
  - BAD:  requiring a function with an exact name like
    `window.state.reset` — your own code may use `restart` or
    `resetGame`; you'll fail your own probe.
Test that the THING EXISTS and has the right SHAPE, not that the
runtime is in a specific instantaneous state.

EXPECTED — emit an <assets> block whenever the game has visual entities
the player will see (sprites, characters, projectiles, terrain). Most
canvas games qualify: shooters, platformers, RPGs, racers, adventures.

<assets>
[
  {"name": "ship",     "prompt": "pixel-art retro arcade spaceship facing right, white outline, transparent background"},
  {"name": "asteroid", "prompt": "pixel-art irregular grey rocky asteroid, transparent bg", "size": "64x64"}
]
</assets>

The harness runs a local Z-Image-Turbo diffuser and saves PNGs next to
your HTML file. Their paths come back in the first-build prompt; load
each via `new Image()` + `await img.decode()`.

SKIP <assets> ONLY for pure-DOM apps where text + emojis suffice (todo
lists, calculators, tic-tac-toe, color pickers). If the canvas has any
rendered entity with visual character, EMIT <assets> — do NOT decide
pre-emptively that procedural ctx.fillRect drawing is "good enough".

EXPECTED ALSO — emit a <sounds> block whenever the game has audible
events the player will hear (firing, hits, pickups, explosions, jumps)
or could benefit from a looping background track. Most arcade /
shooter / platformer games qualify.

<sounds>
[
  {"name": "laser",     "prompt": "short retro arcade laser shot, 8-bit synth blip", "duration": 0.4},
  {"name": "explosion", "prompt": "short pixelated explosion, 8-bit boom",            "duration": 0.8},
  {"name": "music",     "prompt": "loopable 8-bit chiptune background, 90 bpm",       "duration": 12.0, "loop": true}
]
</sounds>

The harness runs Stable Audio Open locally and saves OGGs next to your
HTML file. Their paths come back in the first-build prompt with a
loader pattern; use `new Audio(path)` and play on the relevant events.
Browsers require a user gesture before audio plays — the loader pattern
already handles that (first keydown / pointerdown unlocks audio).

SKIP <sounds> for pure-DOM apps where audio adds nothing (calculators,
color pickers). Otherwise, prefer EMITTING — silent games feel cheap.

No <html_file> yet. No prose outside tags.
"""


# ===========================================================================
# Phase B — first build (with optional playbook block)
# ===========================================================================


# Generic seed-path scrubber. Past `won_<other_session>.html` skeletons
# bake their own `_assets` / `_sounds` directory names into the JS as
# concrete `const ASSET_DIR = './<session>_assets'` strings — and any
# model (regardless of size or vendor) will copy those constants verbatim
# into the new session's code, leaving every sprite and sound broken
# because the referenced directory belongs to a different session. The
# classic-doom 20260512_153449 trace caught DeepSeek-V4 doing exactly
# this; the same failure shape is plausible on every other model.
# Solution: rewrite the seed text so the broken paths can't leak. If we
# know the current session's actual directory names, substitute them so
# the seed remains copy-pasteable. Otherwise replace with an obviously-
# wrong sentinel so a leaked copy fails loudly at asset-load time.
import re as _re_seed
_SEED_PATH_RE = _re_seed.compile(
    r"""\./[A-Za-z0-9._\-]+_(?P<kind>assets|sounds)\b""",
)


def _scrub_seed_paths(
    seed_html: str,
    *,
    current_asset_dir: str | None,
    current_sound_dir: str | None,
) -> str:
    """Replace `./*_assets/` and `./*_sounds/` directory literals in a
    retrieved seed skeleton. Returns the seed unchanged when no paths
    are found.
    """
    if not seed_html:
        return seed_html

    def _sub(m: "_re_seed.Match[str]") -> str:
        kind = m.group("kind")
        if kind == "assets":
            return f"./{current_asset_dir}" if current_asset_dir else (
                "./STALE_PATH_USE_GENERATED_ASSETS_BLOCK_ABOVE_assets"
            )
        return f"./{current_sound_dir}" if current_sound_dir else (
            "./STALE_PATH_USE_GENERATED_SOUNDS_BLOCK_ABOVE_sounds"
        )

    return _SEED_PATH_RE.sub(_sub, seed_html)


def first_build_instruction(
    seed_html: str,
    seed_source: str | None = None,
    *,
    playbook_block: str = "",
    current_asset_dir: str | None = None,
    current_sound_dir: str | None = None,
) -> str:
    """First-build prompt. Includes the SEED CODE the model should start from.

    `playbook_block` is the rendered <playbook> retrieved by the agent;
    when empty the section is omitted. v1 is designed to take the
    playbook seriously — if a relevant bullet is provided, the model is
    explicitly told to apply it.

    `current_asset_dir` / `current_sound_dir` are the basenames of the
    current session's asset / sound directories (e.g.
    `my-game_20260512_153449_assets`). When provided, the seed's stale
    path literals are rewritten to point at the current session's dirs
    so the seed remains copy-pasteable. When omitted, stale paths are
    replaced with a self-describing sentinel that fails loudly at runtime.
    """
    seed_html = _scrub_seed_paths(
        seed_html,
        current_asset_dir=current_asset_dir,
        current_sound_dir=current_sound_dir,
    )
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
    # Stop-Losing-To-OneShot todo #5 — relaxed for mid-tier models that
    # naturally rewrite. Auto-revert (todo #4) catches any rewrite that
    # regresses probes / introduces page errors, so a full <html_file>
    # is now safe; punishing it was a tax we no longer need to collect.
    "Reply with <patch> SEARCH/REPLACE blocks against the file currently "
    "saved on disk (shown above) when the change is local. For structural "
    "changes that patches cannot cleanly express, send a complete "
    "<html_file> — the harness auto-reverts any iter that regresses on "
    "probes / page-errors / criteria coverage, so a rewrite that loses "
    "something will be rolled back. Aim for the SMALLEST footprint that "
    "fixes every ERROR and ISSUE. Always include a <notes> tag explaining "
    "what changed."
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


def continuation_instruction(current_file: str) -> str:
    """User-feedback turn after `<done/>` (or restart of a previously
    finished session).

    The motivating bug: the prior text just SAID "the file is unchanged
    on disk" without including it, so the model was asked to write
    patches blind. Patch-search text routinely hallucinated variable
    names (e.g. `state.princessTimer` vs the file's actual
    `state.princess.timer`) — a one-character whitespace/struct mismatch
    is enough for SEARCH/REPLACE to fail. Worst case for weaker local
    models, and especially worst case on restarts where compaction
    may have already trimmed the file out of the message history.

    The cure mirrors fix_instruction: include the full file inline as
    the SOURCE OF TRUTH so patch anchors can be derived from it
    directly. No diagnostic phase here — there's no failing report;
    the user just typed feedback.
    """
    file_block = (
        "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — patch "
        "against THIS exact text, character-for-character; earlier "
        "turns' code may be stale or absent from this prompt):\n"
        "```html\n"
        f"{current_file}\n"
        "```\n\n"
    )
    return (
        "CONTINUATION TURN: the user has new feedback above for the "
        "game you previously shipped. The CURRENT FILE ON DISK block "
        "below is the exact text on disk right now — patch against it "
        "character-for-character. Reply with one or more <patch> "
        "blocks that address the feedback. Use a full <html_file> "
        "only if patches truly cannot express the change.\n\n"
        f"{file_block}"
    )


def fix_instruction(
    report_text: str,
    current_file: str,
    mistakes_hints: str = "",
    *,
    playbook_block: str = "",
    stuck_streak: int = 0,
    criteria_block: str = "",
    focused_slice: str = "",
) -> str:
    """Combined diagnose + fix turn.

    `stuck_streak` is the count of consecutive failed iterations before
    this one. When it's ≥ 2, the prompt forces the model to commit to
    ONE root cause + ONE patch (no shotgun hypothesis lists).

    `playbook_block` is the rendered <playbook> retrieved against the
    current goal + file; when empty the section is omitted.

    `focused_slice` (Stop-Losing-To-OneShot Track B): when the current
    file is large and a relevance-scored slice has been built, send
    that slice instead of the full file. The patch matcher applies
    against the on-disk file — the model only needs the slice for
    reasoning. Falls back to full file when empty.
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
        # A3: brevity pressure without a rigid schema. An earlier draft
        # required `LINE:N — VAR is TYPE because REASON` but that format
        # only fits "X is undefined" bugs — control-flow bugs like
        # "loop continues past splice" don't have a single misbehaving
        # variable. Conversation.md from
        # game-of-space-invaders_20260512_084906 shows the 27B writes
        # good 2-sentence diagnoses on its own once the source slice
        # (A1) is in the report. The deliberation guard (A2) handles
        # the rambling-before-tags case independently.
        stuck = (
            "STUCK-LOOP CHECK — you have ≥2 failed iterations on this "
            "issue. The report above includes SOURCE NEAR ERROR with "
            "the literal offending line. Inside <diagnose>: ONE root "
            "cause in ≤2 sentences. Name the specific function or "
            "variable or control-flow site responsible. Do NOT "
            "enumerate hypotheses; do NOT emit a numbered or bulleted "
            "list of possibilities. Then ONE <patch> targeting that "
            "site. No prose before <diagnose>.\n\n"
        )
    if focused_slice:
        file_block = (
            "FOCUSED SLICE OF THE CURRENT FILE (the regions implicated by "
            "the failing report, biased by your own <criteria>). Patch "
            "anchors will be matched against the FULL file on disk; this "
            "slice is for reasoning. If the slice doesn't show what you "
            "need, send a SEARCH/REPLACE block anyway — the matcher will "
            "find it on disk.\n"
            "```js\n"
            f"{focused_slice}\n"
            "```\n\n"
        )
    else:
        file_block = (
            "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — patch "
            "against THIS exact text, character-for-character; earlier "
            "turns' code may be stale):\n"
            "```html\n"
            f"{current_file}\n"
            "```\n\n"
        )
    return (
        f"{report_text}\n\n"
        f"{hints}"
        f"{pb}"
        f"{crit}"
        f"{stuck}"
        f"{file_block}"
        f"{PATCH_REMINDER}"
    )


def post_clean_instruction(report_text: str) -> str:
    # Stop-Losing-To-OneShot todo #5 — allow full <html_file> rewrites
    # when truly structural. Auto-revert defends against rewrites that
    # regress, but the goal here remains "ship". <done/> is the
    # strongly-preferred answer; code only if you have a concrete win.
    return (
        f"{report_text}\n\n"
        "No errors. The game works. STRONGLY prefer ending with "
        "<done/>.\n"
        "Only send code if you have ONE small concrete improvement and are "
        "confident it will not regress. A targeted <patch> is best; for "
        "structural improvements only, a complete <html_file> is acceptable "
        "(auto-revert will roll back any version that regresses)."
    )


def truncation_recovery_instruction(
    *,
    report_text: str,
    truncation_reason: str,
    broken_size_bytes: int,
) -> str:
    """Short-form fix prompt for when the on-disk file is structurally
    truncated (open <html>/<body>/<script> without matching close).

    Critical: this prompt deliberately does NOT inline the broken file.
    Patches against a truncated file can't anchor — the patch-target
    lines aren't there yet — and inlining ~7K BPE tokens of broken
    content has been measured (classic-doom 20260512_153449) to push
    the prompt past what some MLX runtimes can serve in 60s, causing
    a no-tokens stall on the next turn. The right move is a fresh
    rewrite, and the agent's rewrite-gate is already open (the
    degenerate-baseline carve-out treats truncated files as degenerate).

    Model-agnostic: every LLM occasionally truncates output (max_tokens
    cap, accidental markdown fence close, generation timeout, etc.).
    """
    return (
        f"{report_text}\n\n"
        f"================ TRUNCATION DETECTED ================\n"
        f"Your previous reply produced an <html_file> that is "
        f"STRUCTURALLY BROKEN: {truncation_reason}.\n"
        f"Approximate file size on disk: {broken_size_bytes} bytes "
        f"(truncated before completion).\n\n"
        f"DO NOT try to patch the broken file — patches need anchor "
        f"text that doesn't exist yet, and any anchor you guess will "
        f"fail to apply.\n\n"
        f"INSTEAD: emit ONE fresh, complete <html_file>...</html_file> "
        f"from scratch. Include every section the previous attempt was "
        f"trying to build (init, render loop, input handlers, asset "
        f"loader call, IIFE close, </script></body></html>). The "
        f"rewrite-rejection gate is open this turn.\n\n"
        f"Reminders that protect against the same failure repeating:\n"
        f"  - The <html_file> body is RAW HTML. Do NOT wrap it in "
        f"markdown code fences. No ```html opener, no closing ``` "
        f"before </html_file>.\n"
        f"  - Emit the COMPLETE file. No elisions, no 'rest unchanged' "
        f"markers.\n"
        f"  - End with </script></body></html> immediately followed by "
        f"</html_file>.\n"
        f"=======================================================\n"
    )


def patch_retry_instruction(failures: list, current_file: str) -> str:
    """Re-prompt after one or more <patch> blocks failed to apply.

    For each failure, we attach an "anchor" — the small region of the
    current file the model was probably aiming at. The typical 27B-class
    failure is "model copied an old version of a line that's since been
    edited"; showing the actual lines (with line numbers, ▸ marker on
    the closest hit) is far more useful than the bare reason string.
    Anchor lookup is best-effort; when it returns None the bullet just
    carries the reason as before.
    """
    from patches import find_anchor  # local import: avoid cycle at module load

    bullets: list[str] = []
    for (i, p, reason) in failures:
        line = f"  - patch #{i+1}: {reason}"
        anchor = find_anchor(current_file, getattr(p, "search", "") or "")
        if anchor:
            line += "\n    nearest match in current file:\n"
            line += "\n".join(f"      {ln}" for ln in anchor.splitlines())
        bullets.append(line)
    bullets_block = "\n".join(bullets)
    return (
        "Some of your <patch> blocks did not apply because the SEARCH "
        "text was not found verbatim in the file. The CURRENT FILE "
        "below is the truth — match it character-for-character:\n\n"
        f"{bullets_block}\n\n"
        "Re-send corrected <patch> blocks against the file as it ACTUALLY "
        "is now (line numbers above are 1-based; '>' marks the closest hit). "
        "Do NOT change anything unrelated to the fix.\n\n"
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
