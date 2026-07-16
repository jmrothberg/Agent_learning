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
    # Optional concise guideline set used ONLY on the lean/`small` system
    # prompt (local models). When None the full `guidelines` are used.
    # Lets the heavy media specs (<assets>/<sounds>/<videos>) keep their
    # essential rules without the multi-KB prose that buries a 27B.
    guidelines_small: list[str] | None = None


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
        # run_13 Elite Trader: probe called simulateClick()/distTo() that the
        # game never attached to window → falsy forever with no useful err.
        # Full path only — lean/small uses guidelines_small (size budget).
        "SELF-CONTAINED: probe exprs may ONLY read `window.state` / "
        "`window.gameState` / DOM / canvas, or call helpers the game "
        "explicitly attaches to `window` (e.g. `window.simulateClick`). "
        "Never call bare helpers like `simulateClick(...)` or `distTo(...)` "
        "unless your code sets `window.simulateClick = ...`. Prefer "
        "dispatching KeyboardEvent / PointerEvent / clicking a real DOM "
        "button inline inside the probe.",
    ],
    guidelines_small=[
        "Probes that reference globals (state, player, etc.) MUST exist on "
        "window — either expose them (e.g. `window.state = state`) or use "
        "DOM / canvas-pixel checks instead. No bare helpers "
        "(`simulateClick`) unless on window.",
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
        "STOCK SOUNDS: the cross-session asset library may already "
        "contain a generic SFX for `jump`, `pickup`, `hit`, `win`, "
        "`lose`, `click`, `laser`, and `explosion` (see "
        "`scripts/build_stock_sounds.py` for prompts). Request these "
        "by the same `name` and the harness will serve the cached OGG "
        "for free (zero generation time). Override by passing your own "
        "prompt for that name — the library will return a hit only if "
        "your prompt tokens overlap the stock prompt enough.",
        "Audio probes: every Audio play() is auto-recorded into "
        "`window.__audioEvents` ({t,src,kind}). In <probes>, assert "
        ".length>0 (or that it grew after a dispatched event via "
        "setTimeout+Promise) to gate audio actually firing — a "
        "silent-but-loaded <audio> otherwise still 'passes'.",
        "SKIP <sounds> for pure-DOM apps where audio adds nothing — "
        "calculators, color pickers, todo lists. Otherwise, default to "
        "EMITTING it: silent games feel cheaper than the same game "
        "with audio.",
        "When the harness returns no sound paths in the first-build "
        "prompt, the audio diffuser was not reachable on this machine; "
        "only THEN ship a silent game.",
    ],
    guidelines_small=[
        "EMIT <sounds> for audible events (firing, hits, pickups, jumps) "
        "or looping music. Load each returned OGG via `new Audio(path)` + "
        "`.play()` (needs a user gesture first). Same `name` mid-session "
        "re-renders that OGG in place.",
        "Entry {name, prompt, duration?, loop?}: SFX 0.2-1.5s; looping "
        "music 8-12s with \"loop\":true. Keep prompts short + audio-"
        "descriptive. Stock names jump/pickup/hit/win/lose/click/laser/"
        "explosion are served free.",
    ],
)

VIDEOS_FORMAT = FormatSpec(
    name="<videos>",
    snippet=(
        "<videos>[{name,prompt,image?,seconds?}, ...]</videos>  Request "
        "generated MP4 cutscene clips (Wan2.2, local). Phase A or "
        "mid-session. EXPENSIVE — minutes of GPU per clip; cutscenes "
        "only, never gameplay."
    ),
    guidelines=[
        "EMIT <videos> ONLY for cinematic cutscene moments — intro, "
        "death/game-over, victory, boss reveal, level transition. "
        "2-4 clips per session MAX: each ~4s clip costs minutes of GPU "
        "(vs seconds for a sprite). Gameplay itself ALWAYS stays on the "
        "canvas — a video is a skippable overlay moment, never the game.",
        "STRONGLY PREFER image-to-video: set `\"image\": "
        "\"<asset_name>\"` naming a key-art entry from your <assets> "
        "block (declare a 768px establishing-shot asset like "
        "`key_intro` first, then animate it). The clip then matches "
        "your in-game art style exactly. Without `image` the clip is "
        "text-to-video and may drift off-style. The prompt should "
        "describe MOTION (\"the dragon slowly rears its head, embers "
        "drift, camera pulls back\"), not re-describe the still.",
        "Optional `seconds` 2-8 (default 4). Clips are silent — pair "
        "them with a <sounds> music/sfx cue if the moment needs audio.",
        "WIRING: play clips via ONE absolutely-positioned muted <video> "
        "overlay covering the canvas (the harness returns the exact "
        "loader pattern with the file paths). Any key must skip; every "
        "failure path (missing file, autoplay block) must continue the "
        "game — a cutscene must NEVER be able to stall the game.",
        "When the harness returns no video paths in the first-build "
        "prompt, the video backend was not reachable on this machine; "
        "only THEN ship without cutscenes (or use a static key-art "
        "pan as the fallback).",
    ],
    guidelines_small=[
        "EMIT <videos> ONLY for cutscene moments (intro, game-over, "
        "victory) — 2-4 clips MAX, each costs minutes of GPU; gameplay "
        "always stays on the canvas. PREFER image-to-video: set "
        "\"image\":\"<key-art asset name>\" so the clip matches your art; "
        "the prompt describes MOTION, not the still.",
        "WIRING: ONE absolutely-positioned muted <video> overlay over the "
        "canvas (loader returned in the first-build prompt). Any key skips; "
        "every failure path (missing file, autoplay block, onended/onerror) "
        "MUST continue the game — a cutscene can never stall it.",
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
        "DRAW CONTRACT: every entity that HAS a generated PNG MUST be "
        "drawn via `ctx.drawImage` through `sprite(key)` (or equivalent "
        "`ASSETS[key]` lookup). Procedural `fillRect`/`arc`/`fillText` "
        "is allowed ONLY for entities with NO generated PNG — HUD bars, "
        "particles, grid lines, selection highlights, destructible overlays "
        "where runtime state must show through (bunkers crumbling brick-by-"
        "brick, health bars filling up). Drawing a plain colored box or "
        "circle for an entity that has a sprite is the same failure as "
        "skipping `<assets>`.",
        "Prompts should be SHORT and visual, and the ART STYLE should "
        "MATCH THE GOAL. Default for canvas sprite games: illustrated 2D "
        "game art — 'illustrated 2D game sprite, clean outline, cel/flat "
        "shading, transparent background' — NOT photorealistic photo, NOT "
        "3D render, NOT stock-photo realism. Only use photorealistic / "
        "cinematic / 'realistic photo' wording when the goal EXPLICITLY "
        "asks for realistic / photo / cinematic art. If the user asks for "
        "high-resolution / detailed / HD / polished modern art (without "
        "saying photo-real), describe DETAILED high-resolution ILLUSTRATED "
        "2D sprites — crisp shading, clean outline, transparent background "
        "— NOT blocky '8-bit' / 'pixel-art' unless the goal explicitly asks "
        "for retro / 8-bit / pixel style. Always use transparent backgrounds "
        "for game sprites.",
        "Optional `size` is a string (\"64x64\", \"256x192\") or an int "
        "(square). Default 512 px square — keeps Z-Image's detail and "
        "lets the game downscale at draw time as needed. Override per "
        "asset only when you need something specific (HUD icons 32-64 "
        "px, full-screen overlays 1024+).",
        "ANIMATION FRAMES & ROSTER LIMITS: Real animation is a SERIES of "
        "frames where the BODY PARTS actually move (legs stride, arm "
        "extends), cycled at runtime to simulate motion — not one image "
        "slid across the screen. Declare the idle/base frame normally, "
        "then declare each motion frame with `\"from_image\": "
        "\"<entity>_idle\"` — seed EVERY frame from the IDLE BASE, not "
        "from the previous frame (chaining frame-from-previous compounds "
        "sameness and the limbs never move). from_image keeps the SAME "
        "character; what makes the pose actually change is (a) "
        "`\"strength\": 0.5-0.6` — the limb-moving band; below ~0.45 the "
        "frame comes back as the idle pose UNCHANGED (verified) — and "
        "(b) a prompt that NAMES the moved part in a FEW WORDS. Write "
        "'left leg forward' / 'right arm extended, fist out' / 'leg "
        "raised high' — NOT vague 'legs apart' and NOT a long paragraph. "
        "KEEP EACH PROMPT SHORT (≈6-12 words): many long, near-identical "
        "prompts in one block make a local model fall into a "
        "token-repetition loop and the turn never finishes. The harness "
        "measures each derived frame against its parent: a frame that "
        "came back near-identical to idle is flagged and BLOCKS <done/> "
        "until the limbs visibly move — raise strength and name the pose "
        "harder, never draw the limb in code. For from_image frames write "
        "ONLY the pose DELTA with visible asymmetry ('hard left bank, left "
        "wing tip toward camera, right wing up' / 'left leg forward') — do "
        "NOT restate the idle orientation ('facing straight up') or the "
        "merged prompt fights itself and bank/tilt comes back as idle. "
        "If the user asks to "
        "animate an EXISTING sprite (\"make the king walk\", \"animate "
        "each piece\"), seed from that sprite the same way; the "
        "MEDIA-CHANGE DIRECTIVE block surfaces stem→asset mappings (e.g. "
        "\"pawn → [white_pawn, black_pawn]\") so you can do each side in "
        "one <assets> block. "
        "Examples (note: every motion frame seeds from `hero_idle`, not "
        "from the prior frame): "
        "{\"name\":\"hero_idle\", \"prompt\":\"8-bit hero holding sword, standing\"}, "
        "{\"name\":\"hero_walk1\", \"prompt\":\"8-bit hero mid-stride, left leg forward, right arm swung back\", \"from_image\":\"hero_idle\", \"strength\":0.55}, "
        "{\"name\":\"hero_walk2\", \"prompt\":\"8-bit hero mid-stride, right leg forward, left arm swung back\", \"from_image\":\"hero_idle\", \"strength\":0.55}. "
        "ROSTER PLANNING & TURNS: Keep the FIRST <assets> block SMALL — a "
        "big first roster of long, repetitive prompts is the main cause of "
        "a planning turn that loops forever and never ships a game. On the "
        "first turn emit ONLY each entity's idle + ONE core motion frame "
        "(roughly ≤8-10 entries total), then add the remaining frames in "
        "later mid-session <assets> turns (same `name` pattern, seeded from "
        "the same idle). This does NOT shrink the visual scope — you still "
        "get every frame, just spread across turns so no single block is "
        "huge. Write your JS loader as a list/array of sprite names and "
        "fall back to the idle frame for any not-yet-generated motion frame "
        "so the game stays playable and testable while later frames catch "
        "up. (The generator itself caps assets per turn; do not rely on "
        "that cap — keep the block you TYPE small.)",
        "SKIP <assets> ONLY for pure-DOM apps where text + emojis are "
        "enough — todo lists, calculators, tic-tac-toe, color pickers. "
        "If the canvas has any rendered entity with visual character, "
        "EMIT <assets>; do not pre-emptively decide procedural drawing "
        "would be 'good enough'.",
        "When the harness returns no asset paths in the first-build "
        "prompt, Z-Image-Turbo was not reachable on this machine; only "
        "THEN fall back to procedural drawing (ctx.fillRect / ctx.arc).",
        "POINT-AND-CLICK / HOTSPOT GAMES: bg_* prompts MUST describe an "
        "EMPTY environment only — walls, floor, sky, distant scenery; "
        "explicitly EXCLUDE people, NPCs, doors, items, and any object "
        "the player will click ('empty tavern interior, no people, no bar "
        "counter in frame'). Every clickable object gets its own transparent "
        "sprite (npc_*, item_*, prop_*) and a hotspot {art,x,y,w,h}; the "
        "game draws sprite(h.art) at the hotspot rect so clicks match the "
        "picture without measuring the PNG.",
    ],
    guidelines_small=[
        "EMIT <assets> for any canvas game with visual entities the player "
        "sees (characters, enemies, projectiles, terrain). Load each "
        "returned PNG via `new Image()` + `await img.decode()`. Match the "
        "ART STYLE to the goal (pixel-art only if the goal asks for retro). "
        "Transparent backgrounds. Same `name` mid-session re-renders in "
        "place; do NOT replace a sprite with procedural drawing on an "
        "art-change request.",
        "ANIMATION: seed every motion frame from the idle base with "
        "`\"from_image\":\"<entity>_idle\"` + `\"strength\":0.5-0.6` and a "
        "SHORT prompt naming the moved part ('left leg forward'). Below "
        "~0.45 strength the frame stays idle. Keep the FIRST <assets> block "
        "SMALL (idle + one core motion frame per entity, ~8-10 total); add "
        "more frames in later mid-session turns. A frame that comes back "
        "near-identical to idle is flagged — raise strength, never draw "
        "limbs in code.",
        "DRAW CONTRACT: entities with a generated PNG → drawImage via "
        "sprite(key); procedural fillRect/arc only for entities with NO "
        "PNG (HUD, particles, grid). "
        "SKIP <assets> only for pure-DOM apps (todo lists, calculators). "
        "If no asset paths return, Z-Image was unreachable — only THEN use "
        "procedural ctx.fillRect/arc.",
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
        "If SEARCH matches >1 place: add unique surrounding context OR "
        "prepend `@@ function_or_class_name` above SEARCH to scope it "
        "(stack two for nested scopes).",
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

TODOS_FORMAT = FormatSpec(
    name="<todos>",
    snippet=(
        "<todos>...</todos>           Mutable checklist of remaining work "
        "(append `[ ]` / `[x]` lines; update each turn)."
    ),
    guidelines=[
        "Optional but encouraged: maintain a <todos>...</todos> "
        "checklist that mirrors what's left to ship. One item per "
        "line, prefixed with `[ ]` (open) or `[x]` (done). Re-emit "
        "the FULL list each turn — the harness persists it to disk "
        "and replays it across compaction so it survives long "
        "sessions. Keep items concrete and small (\"wire space-bar "
        "to fire\", \"reset lastFireTime in reset()\"). When all items "
        "are `[x]` and the test passes, ship with <done/>.",
    ],
)

# Minimal <todos> spec for the small model class (2026-06-12). The full
# TODOS_FORMAT guideline blew the small prompt's 6 KB size budget, which
# is why <todos> used to be in _SMALL_DROP — but todo-driven execution
# (the agent naming ONE unchecked item as the turn's CURRENT TASK) helps
# small models the most, so they must know the tag. One terse guideline.
TODOS_FORMAT_SMALL = FormatSpec(
    name="<todos>",
    snippet=(
        "<todos>...</todos>           Checklist of remaining work "
        "(`[ ]` open / `[x]` done)."
    ),
    guidelines=[
        "Keep a short <todos> checklist (one `[ ]`/`[x]` item per line); "
        "re-emit the FULL updated list each turn. When a CURRENT TASK is "
        "named, work ONLY on that item, then mark it `[x]`.",
    ],
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
    VIDEOS_FORMAT,
    HTML_FORMAT,
    PATCH_FORMAT,
    DIAGNOSE_FORMAT,
    NOTES_FORMAT,
    TODOS_FORMAT,
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
    "For three.js/WebGL, output must run in normal browser file:// "
    "(no harness-only security flags).",
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
    # Added 2026-05-21 — most common probe failure across May 20-21 traces
    # (pac, dk, sf, doom, FPS). Tight phrasing so small-model prompt stays
    # under the 6KB target verified by test_prompt_size.
    "Expose state on window: `window.state = state` (required — probes "
    "read `window.state` first); `window.gameState = state` is an accepted "
    "alias. Also `window.game = { reset }`. Probes call "
    "`window.state.score`, `window.game.reset()` — un-exposed state fails "
    "probes even when the game works.",
    # Added 2026-07-01 — harness asset-settle poll (tools.py) waits on
    # window._assetsReady before sampling drawImage events.
    "After loadAssets() finishes (every Image decode/onload), set "
    "`window._assetsReady = true` so the harness knows decoding is done.",
    # Added 2026-05-25 from OpenAI Codex review. Prevents the `probes_only` /
    # `media_only` `no_usable_code` failure shape. Tag names omitted so the
    # small-class prompt-size + drops-optional-tags asserts still hold.
    "Phase-A signals (plan, criteria, probes, media) persist once accepted; "
    "fix turns emit <patch> only.",
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
    "NEVER reflect circle collisions while still overlapping — push the "
    "body out along the normal by overlap+1px BEFORE mirroring velocity; "
    "use a per-body hit cooldown (3-5 frames) so bumpers cannot re-score "
    "every frame (infinite ping-pong loop).",
    "NEVER launch a playfield body with only vertical velocity from a "
    "shooter/plunger lane — charged release needs horizontal velocity into "
    "the play area or an angled guide wall, else the body bounces straight "
    "back and never enters play.",
    "NEVER integrate fast physics once per frame without substeps or a "
    "speed cap — thin colliders (flippers, paddles, walls) tunnel at "
    "high speed; use 2-4 substeps and cap |v| around 1200 px/s.",
    "NEVER place timed-state early-return ABOVE the timer decrement — "
    "hit-stun/knockdown locks control forever (see stun-timer-before-"
    "early-return playbook bullet when CONTROL-NOT-RECOVERED fires).",
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
            # Drop unused media pipelines and the on-demand playbook lookup
            # mechanism. Trace 20260613_123726 showed the old unconditional
            # small-model drop removed <assets>/<videos> from the authoritative
            # output-tags list even when the goal explicitly required generated
            # media, so the model omitted media tags in Phase A. Keep each
            # media tag when the actual goal triggers its intent detector.
            # <todos> is RE-ENABLED for small (2026-06-12, was in the drop
            # set for size budget) via the minimal TODOS_FORMAT_SMALL —
            # todo-driven CURRENT TASK turns help small models the most.
            _SMALL_DROP = {"<lookup_bullet>"}
            if not _detect_art_intent(goal):
                _SMALL_DROP.add("<assets>")
            if not _detect_audio_intent(goal):
                _SMALL_DROP.add("<sounds>")
            if not _detect_video_intent(goal):
                _SMALL_DROP.add("<videos>")
            fmts = [
                TODOS_FORMAT_SMALL if f.name == "<todos>" else f
                for f in ALL_FORMATS if f.name not in _SMALL_DROP
            ]
        else:
            fmts = ALL_FORMATS
    else:
        fmts = formats

    output_tags = "\n".join(f"  {f.snippet}" for f in fmts)

    # Per-format guidelines, deduped across formats. On the lean/`small`
    # path, prefer each spec's concise `guidelines_small` when provided so
    # the heavy media specs don't bury a local model in multi-KB prose.
    all_guidelines: list[str] = []
    for f in fmts:
        if is_small and f.guidelines_small is not None:
            all_guidelines.extend(f.guidelines_small)
        else:
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


# Visible canvas entities — ball, flipper, ship, etc. Genre-free shape
# detector: fires canvas-entity-art plan nudge when the goal names entities
# the player sees but did not use explicit art keywords.
_CANVAS_ENTITY_KEYWORDS = frozenset({
    "ball", "balls", "flipper", "flippers", "bumper", "bumpers", "paddle",
    "paddles", "brick", "bricks", "ship", "ships", "enemy", "enemies",
    "player", "character", "hero", "table", "creeps", "creep", "turret",
    "turrets", "projectile", "projectiles", "piece", "pieces", "gem", "gems",
    "card", "cards", "tower", "towers", "block", "blocks", "sprite",
})

_PINBALL_KEYWORDS = frozenset({
    "pinball", "flipper", "flippers", "bumper", "bumpers", "plunger",
    "drain", "multiball", "tilt", "nudge", "slingshot-pin", "arcade-table",
})

# Open-field / Fieldrunners TD — beam vs rotating turret split (20260703).
_OPEN_FIELD_TD_SHAPE = frozenset({
    "open", "field", "tower", "towers", "turret", "turrets", "bfs", "maze",
    "mazing", "tesla", "flame", "creep", "creeps", "wave", "waves",
})


def _detect_open_field_td_intent(goal: str) -> list[str]:
    """Open-field / Fieldrunners TD — rotating turrets vs fixed beam towers."""
    if not goal:
        return []
    import re
    gl = goal.lower().replace("_", " ")
    gl_dash = gl.replace(" ", "-")
    if "fieldrunners" in gl or "open-field" in gl_dash:
        return ["fieldrunners"]
    words = re.findall(r"[a-zA-Z]+", gl)
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _OPEN_FIELD_TD_SHAPE and w not in seen:
            seen.add(w)
            out.append(w)
    if "open" in seen and "field" in seen and seen & {
        "tower", "towers", "turret", "turrets", "bfs", "maze", "mazing",
    }:
        return out
    return []


def _detect_canvas_entity_intent(goal: str) -> list[str]:
    """Return visible-entity keywords when goal describes canvas entities."""
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _CANVAS_ENTITY_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _detect_pinball_intent(goal: str) -> list[str]:
    """Return pinball-family keywords for table-physics plan nudge."""
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _PINBALL_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    if "pinball" in seen or len(out) >= 2:
        return out
    return []


# Modality keywords that signal the goal needs a 3D rendering technique.
# Genre-free per the project rule — these all describe rendering shape
# ("first-person view", "raycaster", "voxel"), not a specific game. The
# goal is to nudge the model toward a real 3D library (three.js / babylon
# / PlayCanvas via CDN) rather than hand-rolling a raycaster in 12 KB,
# which is what produced the unplayable doom session.
# 3D modality keyword set + detector now live in the shared modality.py
# (single source of truth) so the planner and memory retrieval can never
# disagree on whether a goal is 3D. Aliased to the old private names so the
# rest of this module — and tests that call prompts_v1._detect_3d_intent —
# keep working unchanged.
from modality import THREE_D_KEYWORDS as _3D_KEYWORDS
from modality import detect_3d_intent as _detect_3d_intent

# Modality-aware dynamic probe for input_moves_player (3D navigation fix 2026-06-30).
_INPUT_MOVES_PLAYER_DEFAULT = (
    "(async()=>{if(!window.state||!state.player)return false;const x0=state.player.x;"
    "window.dispatchEvent(new KeyboardEvent('keydown',{code:'ArrowRight',bubbles:true}));"
    "await new Promise(r=>setTimeout(r,250));"
    "window.dispatchEvent(new KeyboardEvent('keyup',{code:'ArrowRight',bubbles:true}));"
    "return state.player.x!==x0;})()"
)
_INPUT_MOVES_PLAYER_THREEJS_FPS = (
    "(async()=>{if(!window.state||!state.player)return false;const p=state.player;"
    "const x0=p.x,z0=p.z,y0=(p.yaw!==undefined?p.yaw:p.angle);"
    "window.dispatchEvent(new KeyboardEvent('keydown',{code:'ArrowUp',bubbles:true}));"
    "await new Promise(r=>setTimeout(r,250));"
    "window.dispatchEvent(new KeyboardEvent('keyup',{code:'ArrowUp',bubbles:true}));"
    "return p.x!==x0||p.z!==z0||(p.yaw!==undefined&&p.yaw!==y0);})()"
)
_INPUT_MOVES_PLAYER_WIREFRAME = (
    "(async()=>{if(!window.state||!state.player)return false;const p=state.player;"
    "const y0=(p.yaw!==undefined?p.yaw:p.angle);"
    "window.dispatchEvent(new KeyboardEvent('keydown',{code:'ArrowLeft',bubbles:true}));"
    "await new Promise(r=>setTimeout(r,250));"
    "window.dispatchEvent(new KeyboardEvent('keyup',{code:'ArrowLeft',bubbles:true}));"
    "const y1=(p.yaw!==undefined?p.yaw:p.angle);"
    "return typeof y0==='number'&&typeof y1==='number'&&y1!==y0;})()"
)
_INPUT_MOVES_PLAYER_MODE7 = (
    "(async()=>{const s=window.state;if(!s)return false;"
    "const p=s.player||s;const s0=(p.speed!==undefined?p.speed:p.angle);"
    "window.dispatchEvent(new KeyboardEvent('keydown',{code:'ArrowUp',bubbles:true}));"
    "await new Promise(r=>setTimeout(r,250));"
    "window.dispatchEvent(new KeyboardEvent('keyup',{code:'ArrowUp',bubbles:true}));"
    "const s1=(p.speed!==undefined?p.speed:p.angle);"
    "return s1!==s0||p.x!==undefined;})()"
)


def input_moves_player_probe_expr(*, goal: str = "", code: str = "") -> str:
    """Pick a dynamic movement probe matching the rendering modality."""
    from modality import detect_fps_navigation_modality

    mod = detect_fps_navigation_modality(goal=goal, code=code)
    if mod == "wireframe":
        return _INPUT_MOVES_PLAYER_WIREFRAME
    if mod == "threejs":
        return _INPUT_MOVES_PLAYER_THREEJS_FPS
    if mod == "mode7":
        return _INPUT_MOVES_PLAYER_MODE7
    return _INPUT_MOVES_PLAYER_DEFAULT


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


# Modality keywords that signal the goal wants generated VIDEO cutscenes.
# Genre-free per the project rule — these describe output modality
# (cutscene, cinematic clip), not subject matter. When any appears in
# the goal, plan_instruction() escalates the <videos> directive to
# REQUIRED so smaller models don't politely skip the (expensive but
# explicitly requested) video pipeline.
_VIDEO_KEYWORDS = frozenset({
    "cutscene", "cutscenes", "cinematic", "cinematics",
    "video", "videos", "movie", "movies", "clip", "clips",
    "trailer", "filmic",
})


def _detect_video_intent(goal: str) -> list[str]:
    """Return a list of video-modality keywords found in `goal`. Empty
    list means no intent detected; non-empty triggers a stronger
    <videos> directive in the planning prompt. Also matches the
    two-word form "cut scene" by joining adjacent words, mirroring
    `_detect_3d_intent`'s join trick.
    """
    if not goal:
        return []
    import re
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", goal)]
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _VIDEO_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    for i in range(len(words) - 1):
        j = words[i] + words[i + 1]   # "cut scene" -> "cutscene"
        if j in _VIDEO_KEYWORDS and j not in seen:
            seen.add(j)
            out.append(j)
    return out


# Mechanism keywords that signal a timed-reaction / QTE scene structure.
# Genre-free: these describe interaction shape, not subject matter. The nudge
# is deliberately short and conditional so it helps local models wire media +
# input without bloating unrelated prompts.
_QTE_KEYWORDS = frozenset({
    "qte", "quicktime", "quick-time", "quick-time-event",
    "reaction", "react", "timed", "timing", "prompt",
    "scripted", "scene", "scenes",
})


_SPATIAL_ALIGNMENT_KEYWORDS = frozenset({
    "hotspot", "hotspots", "point-and-click", "pointclick", "monkey",
    "inventory", "qte", "quick-time", "quicktime", "dragons-lair", "dragon",
    "lair", "duck", "jump", "sword", "hazard", "telegraph", "clickable",
    "point-and-click", "adventure", "cutscene", "peril",
})


def _detect_spatial_alignment_intent(goal: str) -> list[str]:
    """Return spatial-alignment keywords (hotspot/QTE hazard sync) in goal."""
    if not goal:
        return []
    import re
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", goal)]
    seen: set[str] = set()
    out: list[str] = []
    gl = goal.lower()
    for phrase in ("point-and-click", "point and click", "monkey island", "dragon's lair", "dragons lair"):
        if phrase in gl and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    for w in words:
        if w in _SPATIAL_ALIGNMENT_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


_POINT_AND_CLICK_KEYWORDS = frozenset({
    "point-and-click", "pointclick", "monkey", "inventory", "hotspot",
    "hotspots", "adventure", "lucasarts", "sierra", "myst", "maniac",
    "verb", "dialog", "dialogue", "examine", "pickup", "rooms", "scene",
    "scenes", "npc", "lucas", "kings", "quest",
})


def _detect_point_and_click_intent(goal: str) -> list[str]:
    """Return point-and-click adventure keywords found in `goal`."""
    if not goal:
        return []
    import re
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", goal)]
    seen: set[str] = set()
    out: list[str] = []
    gl = goal.lower()
    for phrase in (
        "point-and-click", "point and click", "monkey island",
        "maniac mansion", "kings quest", "king's quest",
    ):
        if phrase in gl and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    for w in words:
        if w in _POINT_AND_CLICK_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _detect_qte_intent(goal: str) -> list[str]:
    """Return timed-reaction/QTE mechanism keywords found in `goal`.

    Also matches joined variants like "quick time" -> "quicktime" and
    "quick time event" -> "quicktimeevent".
    """
    if not goal:
        return []
    import re
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", goal)]
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _QTE_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    for i in range(len(words) - 1):
        for j in (words[i] + words[i + 1], words[i] + "-" + words[i + 1]):
            if j in _QTE_KEYWORDS and j not in seen:
                seen.add(j)
                out.append(j)
    for i in range(len(words) - 2):
        j = words[i] + words[i + 1] + words[i + 2]
        if j in {"quicktimeevent"} and j not in seen:
            seen.add(j)
            out.append(j)
    return out


# _detect_3d_intent moved to modality.py (imported/aliased above).


# Phase 4: heavy-logic keyword set. These describe game-rule depth or
# AI/search complexity that — when COMBINED with an art-heavy goal —
# tends to push weak local LLMs into concatenating two drafts in one
# stream. Trace 1 (chess 20260522_000304) hit exactly this combo: 24
# sprites + 3-ply minimax + full chess rules in a single iter. Genre-free
# by design (no "chess" / "rts" / "pacman" terms); the entries describe
# computational shape, not subject matter.
_HEAVY_LOGIC_KEYWORDS = frozenset({
    "ai", "minimax", "alpha", "beta", "ply", "search", "depth",
    "pathfinding", "astar", "negamax", "evaluation", "heuristic",
    "rules", "ruleset", "turn-based", "turnbased", "physics",
    "simulation", "multiplayer", "networking", "persistent",
    "save", "savegame", "load", "loadgame", "undo", "redo",
    "replay", "opponent", "computer", "cpu", "bot",
})


def _detect_heavy_logic_intent(goal: str) -> list[str]:
    """Return a list of heavy-logic keywords found in `goal`. Empty list
    means the goal is plain action / arcade and needs no scope-pacing
    nudge.

    Used in combination with `_detect_art_intent` to detect the
    art + complex-logic shape that risks weak-LLM one-shot overload.
    Genre-free single-token matching.
    """
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _HEAVY_LOGIC_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
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


# Vector wireframe arcade (Battlezone / Star Wars / Asteroids line-art).
# Suppresses false art/3D nudges from "line art" and "first-person".
_WIREFRAME_VECTOR_PHRASES = frozenset({"vector line", "line art"})


def _detect_wireframe_vector_intent(goal: str) -> list[str]:
    """Return wireframe-vector modality keywords found in `goal`."""
    if not goal:
        return []
    import re
    gl = goal.lower()
    out: list[str] = []
    seen: set[str] = set()
    if "wireframe" in gl:
        seen.add("wireframe")
        out.append("wireframe")
    words = re.findall(r"[a-zA-Z]+", gl)
    for i in range(len(words) - 1):
        j = words[i] + " " + words[i + 1]
        if j in _WIREFRAME_VECTOR_PHRASES and j not in seen:
            seen.add(j)
            out.append(j.replace(" ", "-"))
    if "line art" in gl and "vector" in gl and "vector-line-art" not in seen:
        out.append("vector-line-art")
    return out


def _perspective_wireframe_nudge_needed(goal: str) -> bool:
    """Perspective projection nudge for FPS-style wireframe, not top-down."""
    gl = goal.lower()
    if "wireframe" in gl:
        return True
    for phrase in ("first-person", "first person", "trench", "cockpit", "tank"):
        if phrase in gl:
            return True
    return False


# Phase 0 (Fieldrunners trace 20260626_102307): tower-defense SHAPE tokens.
# The beat-em-up detector's weak trigger "waves" also appears in tower-defense
# goals ("waves of enemies"), which wrongly injected a "side-scrolling brawler"
# nudge onto a TD plan turn. When any of these mechanic-shape tokens are present
# the goal is a placement/path defense, not a brawler — suppress the nudge.
# Genre-free: every token names a rendering/mechanic shape (grid, path, turret),
# not subject matter.
def _detect_beat_em_up_intent(goal: str) -> list[str]:
    """Side-scroll beat-em-up modality (not 1v1 fighters). Genre-free tokens.

    Suppressed when the goal matches the tower-defense visual_playtest recipe:
    "waves" alone is a weak trigger shared with TD ("waves of enemies"), and a
    TD goal must not receive a brawler nudge (Fieldrunners trace
    20260626_102307 plan turn). Phase 4 (4A): the TD-vs-brawler disambiguation
    now lives in DATA (memory/visual_playtests.jsonl `suppresses_nudges`),
    loaded through the single `memory.goal_suppresses_nudge` so retrieval and
    this plan-nudge detector can't disagree. Lazy import avoids a circular
    import (memory imports nothing from prompts_v1, but keep it local for
    safety)."""
    from memory import goal_suppresses_nudge
    if goal_suppresses_nudge(goal, "beat-em-up"):
        return []
    gl = goal.lower()
    gl_dash = gl.replace("_", "-")
    out: list[str] = []
    seen: set[str] = set()
    for tok in (
        "beat-em-up", "beatemup", "brawler", "side-scroll", "side-scrolling",
        "sidescroll", "scrolling", "waves", "floor", "boss",
    ):
        if tok in gl_dash or tok.replace("-", " ") in gl:
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


# Phase 0.8 — single-token keywords that, alone, signal the user wants
# more than one frame per visual entity (walk cycles, idle bobs, attack
# sequences, multi-state animations). Combined with `_MULTI_FRAME_PHRASES`
# below for two-word matches. Same shape as `_ART_KEYWORDS`: lowercased
# token match, no genre / NER / model lookup.
_MULTI_FRAME_KEYWORDS = frozenset({
    # Frame-count nouns
    "frames", "framesets", "spritesheet", "spritesheets", "tilesheet",
    # Animation-state nouns (when paired with an art noun in the goal,
    # these are direct asks for multiple frames per entity)
    "walking", "walkcycle",
    "running", "jumping", "falling", "attacking", "shooting", "casting",
    "dying", "death", "dead", "hurt", "damaged",
    "idle", "bobbing", "bobbed", "swinging", "punching", "kicking",
    "crouching", "ducking", "blocking",
    "animated", "animating",
    # Destruction / impact action states — general, not genre-specific
    # ("smash the captured piece", "shatter on hit", "explode on death").
    # These describe per-entity destroy frames, not just game mechanics.
    "smash", "smashed", "smashing",
    "destroy", "destroyed", "destroying", "destruction",
    "shatter", "shattering", "shattered",
    "explode", "exploding", "exploded", "explosion",
    # Quantity hint
    "multiple", "several",
    # Multi-state directive
    "states", "poses",
})

# Multi-word phrases that strongly indicate multi-frame intent. The
# detector joins adjacent goal tokens (raw + hyphenated) and tests for
# membership; this catches "walk stride", "walk cycle", "3 frames",
# "animation states", "animated series" etc.
_MULTI_FRAME_PHRASES = frozenset({
    "walkcycle", "walkstride", "walkframes",
    "runcycle", "runframes",
    "idlebob", "idleframes",
    "attackframes", "deathframes", "hurtframes",
    "animationstates", "animationsequence", "animationseries",
    "animationframes", "animationsequences", "framespattern",
    "animatedseries", "animatedsequence",
    "spriteset", "spritesheet",
    "movecycle", "actioncycle",
})

# Numeric-prefixed phrase patterns: "3 frames each", "two sprites per",
# "N animations for each". A compiled regex covers these without
# enumerating every numeral; one pattern per phrasing.
import re as _re
_MULTI_FRAME_REGEXES = (
    _re.compile(
        r"\b(?:two|three|four|five|six|2|3|4|5|6|\d+)\s+"
        r"(?:frames?|sprites?|images?|poses?|states?|animations?|variants?)"
        r"(?:\s+(?:for|per|each)\b|\s+of\s+each\b)?",
        _re.I,
    ),
    _re.compile(
        r"\b(?:walk|run|jump|attack|hurt|death|idle|swing|punch|kick|smash|destroy|shatter|explode)"
        r"[-\s]?(?:cycle|cycles|frame|frames|sequence|stride|strides|animation|animations)\b",
        _re.I,
    ),
    _re.compile(
        r"\banim(?:ation|ate|ated)\s+(?:states?|frames?|series|sequence|cycle)\b",
        _re.I,
    ),
    _re.compile(
        r"\bidle\s+(?:and|\+|,)\s*(?:walk|run|attack|hurt|death)\b",
        _re.I,
    ),
    _re.compile(
        r"\bmulti(?:ple|-)?\s*(?:frames?|sprites?|poses?|states?|animations?)\b",
        _re.I,
    ),
)


def _detect_multi_frame_intent(goal: str) -> list[str]:
    """Return matched multi-frame keywords/phrases. Empty list means no
    multi-frame intent — the agent's default "one sprite per entity"
    behavior is appropriate.

    Examples of goals that match (general — no genre logic):
      - "make each piece animated walking and attacking"
      - "3 sprites for each character with idle and walk states"
      - "smooth interpolated piece movement with idle bob and walk
         stride animations"
      - "spritesheet of 4 frames per enemy"

    Examples that DO NOT match:
      - "shoot space invaders with a single ship sprite"
      - "minimax chess engine, no animations needed"

    Detection layers (any one triggers a match):
      1. Single-token presence from `_MULTI_FRAME_KEYWORDS`
      2. Adjacent-token joined phrase from `_MULTI_FRAME_PHRASES`
         (handles "walk stride" / "walk cycle" / "animation states")
      3. Numeric-prefixed regex from `_MULTI_FRAME_REGEXES`
         ("3 frames per", "two sprites each", "walk-cycle frames")
    """
    if not goal:
        return []
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    matched: list[str] = []
    seen: set[str] = set()
    # Layer 1: single tokens.
    for w in words:
        if w in _MULTI_FRAME_KEYWORDS and w not in seen:
            seen.add(w)
            matched.append(w)
    # Layer 2: adjacent-token joins (and hyphenated variants).
    for i in range(len(words) - 1):
        joined = words[i] + words[i + 1]
        if joined in _MULTI_FRAME_PHRASES and joined not in seen:
            seen.add(joined)
            matched.append(joined)
    # Layer 3: numeric / phrase patterns.
    for rx in _MULTI_FRAME_REGEXES:
        m = rx.search(goal)
        if m:
            phrase = m.group(0).lower().strip()
            if phrase not in seen:
                seen.add(phrase)
                matched.append(phrase)
    return matched


# Classic franchise goals: nudge asset prompts toward iconic silhouettes
# (user-editable specific guidance — genre-free mechanism memory stays separate).
_FRANCHISE_ASSET_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("space invaders", "space-invaders", "invaders"),
        "FRANCHISE ART — Space Invaders: prompt distinct classic arcade "
        "alien silhouettes (crab, squid, octopus rows) — NOT generic bugs.",
    ),
    (
        ("pac-man", "pacman", "ms pac"),
        "FRANCHISE ART — Pac-Man: yellow circle hero; pursuers as rounded "
        "ghost silhouettes with distinct colors.",
    ),
    (
        ("dig dug", "dig-dug"),
        "FRANCHISE ART — Dig Dug: round digger with pump hose; puffy "
        "underground monsters that inflate when pumped.",
    ),
    (
        ("minecraft",),
        "FRANCHISE ART — voxel sandbox: name blocks grass/dirt/stone (or "
        "similar) in <assets> prompts so hotbar types stay visually distinct.",
    ),
    (
        ("doom", "wolfenstein"),
        "FRANCHISE ART — FPS: textured wall/floor sprites plus billboard "
        "monster silhouettes; weapon overlay muzzle points UP in source art.",
    ),
]


def _franchise_asset_nudge(goal: str) -> str:
    if not goal:
        return ""
    gl = goal.lower()
    for keys, text in _FRANCHISE_ASSET_HINTS:
        if any(k in gl for k in keys):
            return "\n\n" + text + "\n"
    return ""


def plan_instruction(
    *,
    reference_block: str = "",
    goal: str = "",
    force_minimal_first_build: bool = False,
    from_seed: bool = False,
    seed_asset_names: list[str] | None = None,
    seed_sound_names: list[str] | None = None,
    model_class: str = "auto",
    nudge_ids_out: list[str] | None = None,
) -> str:
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

    P1 (MK trace 20260528): on a `/seed` continuation where
    `<basename>_assets/` already has sprites on disk, art/audio
    "MUST emit <assets>/<sounds>" nudges flip from REQUIRED to
    FORBIDDEN — the model should reuse existing media, not request
    fresh generation. `from_seed=True` suppresses those nudges and
    inserts a seed-continuation directive listing available media.
    """
    # Phase 4 (4A): plan-turn nudge PROSE lives in memory/plan_nudges.jsonl,
    # loaded via the single memory loader. prompts_v1 keeps only the detectors
    # (which keywords fired) + the slot interpolation. Lazy import avoids any
    # import-order coupling.
    from memory import load_plan_nudge

    def _record_nudge(nudge_id: str) -> None:
        if nudge_ids_out is not None:
            nudge_ids_out.append(nudge_id)

    # P1: seed continuation flips art / audio "MUST emit" nudges off.
    # Detection is opt-in via the from_seed kwarg so non-seed sessions
    # stay byte-for-byte identical (the existing test suite still passes).
    if from_seed:
        # Suppress art / audio MUST-emit nudges; the seed already has
        # media on disk and re-emitting <assets> wastes generator time
        # AND can wipe the user's existing art if a name collides.
        art_keywords = ()
        wireframe_keywords = _detect_wireframe_vector_intent(goal)
        beat_em_up_keywords = _detect_beat_em_up_intent(goal)
        threed_keywords = () if wireframe_keywords else _detect_3d_intent(goal)
        audio_keywords = ()
        video_keywords = ()
        qte_keywords = _detect_qte_intent(goal)
    else:
        wireframe_keywords = _detect_wireframe_vector_intent(goal)
        beat_em_up_keywords = _detect_beat_em_up_intent(goal)
        art_keywords = () if wireframe_keywords else _detect_art_intent(goal)
        threed_keywords = () if wireframe_keywords else _detect_3d_intent(goal)
        audio_keywords = _detect_audio_intent(goal)
        video_keywords = _detect_video_intent(goal)
        qte_keywords = _detect_qte_intent(goal)
    art_nudge = ""
    illustrated_sprite_nudge = ""
    if art_keywords:
        kws = ", ".join(repr(k) for k in art_keywords)
        art_nudge = load_plan_nudge("art").replace("{kws}", kws)
        _record_nudge("art")
        illustrated_sprite_nudge = load_plan_nudge("illustrated-2d-sprite")
        illustrated_sprite_nudge += (
            "\n\nDIRECTIONAL SPRITES — for any character that moves left/right, "
            "prompt sprites facing screen-RIGHT (canonical). Then a single "
            "`flip = facing < 0` (or `ctx.scale(-1,1)` when moving left) "
            "rule is always correct.\n"
        )
        _record_nudge("illustrated-2d-sprite")

    canvas_entity_keywords = ()
    canvas_entity_nudge = ""
    if (
        not from_seed
        and not wireframe_keywords
        and not art_keywords
    ):
        canvas_entity_keywords = _detect_canvas_entity_intent(goal)
        if len(canvas_entity_keywords) >= 2:
            kws = ", ".join(repr(k) for k in canvas_entity_keywords)
            canvas_entity_nudge = load_plan_nudge(
                "canvas-entity-art"
            ).replace("{kws}", kws)
            _record_nudge("canvas-entity-art")

    pinball_nudge = ""
    if not from_seed and not wireframe_keywords:
        pinball_keywords = _detect_pinball_intent(goal)
        if pinball_keywords:
            kws = ", ".join(repr(k) for k in pinball_keywords)
            pinball_nudge = load_plan_nudge("pinball-table").replace(
                "{kws}", kws
            )
            _record_nudge("pinball-table")

    open_field_td_nudge = ""
    if not from_seed and not wireframe_keywords:
        open_field_td_keywords = _detect_open_field_td_intent(goal)
        if open_field_td_keywords:
            kws = ", ".join(repr(k) for k in open_field_td_keywords)
            open_field_td_nudge = load_plan_nudge("open-field-td").replace(
                "{kws}", kws
            )
            _record_nudge("open-field-td")

    # `threed_keywords` is already set at the top of the function so a
    # from_seed continuation still keeps 3D detection but loses art/audio.
    threed_nudge = ""
    if threed_keywords:
        kws = ", ".join(repr(k) for k in threed_keywords)
        threed_nudge = load_plan_nudge("3d").replace("{kws}", kws)

    beat_em_up_nudge = ""
    if beat_em_up_keywords:
        kws = ", ".join(repr(k) for k in beat_em_up_keywords)
        beat_em_up_nudge = load_plan_nudge("beat-em-up").replace("{kws}", kws)

    wireframe_nudge = ""
    if wireframe_keywords:
        kws = ", ".join(repr(k) for k in wireframe_keywords)
        if _perspective_wireframe_nudge_needed(goal):
            wireframe_nudge = load_plan_nudge(
                "wireframe-perspective"
            ).replace("{kws}", kws)
        else:
            wireframe_nudge = load_plan_nudge(
                "wireframe-flat"
            ).replace("{kws}", kws)

    # `audio_keywords` is also set up top so from_seed suppression sticks.
    audio_nudge = ""
    if audio_keywords:
        kws = ", ".join(repr(k) for k in audio_keywords)
        audio_nudge = load_plan_nudge("audio").replace("{kws}", kws)

    # `video_keywords` is also set up top so from_seed suppression sticks.
    video_nudge = ""
    if video_keywords:
        kws = ", ".join(repr(k) for k in video_keywords)
        video_nudge = load_plan_nudge("video").replace("{kws}", kws)

    qte_nudge = ""
    if qte_keywords:
        kws = ", ".join(repr(k) for k in qte_keywords)
        qte_nudge = load_plan_nudge("qte").replace("{kws}", kws)

    spatial_alignment_nudge = ""
    spatial_keywords = _detect_spatial_alignment_intent(goal)
    if spatial_keywords:
        kws = ", ".join(repr(k) for k in spatial_keywords)
        spatial_alignment_nudge = load_plan_nudge("spatial-alignment").replace(
            "{kws}", kws
        )
        _record_nudge("spatial-alignment")

    point_and_click_nudge = ""
    pointclick_keywords = _detect_point_and_click_intent(goal)
    if pointclick_keywords:
        kws = ", ".join(repr(k) for k in pointclick_keywords)
        point_and_click_nudge = load_plan_nudge("point-and-click").replace(
            "{kws}", kws
        )
        _record_nudge("point-and-click")

    franchise_asset_nudge = _franchise_asset_nudge(goal)

    # Phase 4: scope-pacing nudge for art-heavy + logic-heavy goals.
    # Only fires when BOTH art (or 3D) AND heavy-logic keywords match —
    # so a pure-arcade goal (lots of art, simple rules) and a pure-logic
    # goal (chess engine, no art) both stay unaffected. The nudge is
    # PROMPT ONLY and genre-free; it tells the model to ship the core
    # skeleton first and defer extra walk-cycle / VFX frames to a later
    # mid-session <assets> turn.
    #
    # Phase 0.9 — when the user explicitly asks for multi-frame /
    # animation rosters in the goal (`_detect_multi_frame_intent`),
    # the "idle only" rule INVERTS: the architect MUST emit base +
    # frame chains via `from_image` this turn, because deferring an
    # explicit user request is the listening bug Phase 0 fixes.
    heavy_logic_keywords = _detect_heavy_logic_intent(goal)
    multi_frame_keywords = _detect_multi_frame_intent(goal)
    scope_nudge = ""
    if heavy_logic_keywords and (art_keywords or threed_keywords):
        logic_kws = ", ".join(repr(k) for k in heavy_logic_keywords)
        if multi_frame_keywords:
            mf_kws = ", ".join(repr(k) for k in multi_frame_keywords)
            scope_nudge = (
                load_plan_nudge("scope-pacing-multiframe")
                .replace("{logic_kws}", logic_kws)
                .replace("{mf_kws}", mf_kws)
            )
        else:
            scope_nudge = load_plan_nudge("scope-pacing").replace(
                "{logic_kws}", logic_kws
            )

    # Phase 0.9 — multi-frame nudge fires INDEPENDENTLY of scope-pacing
    # so a pure-art goal (no heavy logic) with explicit multi-frame
    # intent still gets the directive. Without this, "platformer with
    # walk and attack frames" would have no animation-specific guidance
    # because the scope-pacing nudge above only fires when both art and
    # heavy-logic keywords match.
    multi_frame_nudge = ""
    if multi_frame_keywords and not scope_nudge:
        mf_kws = ", ".join(repr(k) for k in multi_frame_keywords)
        multi_frame_nudge = load_plan_nudge("multi-frame").replace(
            "{mf_kws}", mf_kws
        )

    # P1 (MK trace 20260528): seed continuation directive.
    # When the harness is restarting a session against an EXISTING game
    # file, the on-disk `<basename>_assets/` and `<basename>_sounds/`
    # folders already contain this game's media. The model's job this
    # turn is to PATCH the existing code, not redesign the asset roster.
    seed_nudge = ""
    if from_seed:
        seed_a = sorted(seed_asset_names or [])
        seed_s = sorted(seed_sound_names or [])

        def _fmt_names(names: list[str], cap: int = 24) -> str:
            if not names:
                return "(none)"
            shown = names[:cap]
            more = f" (+{len(names) - cap} more)" if len(names) > cap else ""
            return ", ".join(shown) + more

        seed_nudge = (
            "\n\nSEED CONTINUATION — this is an EXISTING game the user "
            "is asking you to ADAPT. The harness has rehydrated this "
            "session's media from disk already; new generation in this "
            "turn would WIPE the user's existing art and is BLOCKED.\n"
            "\n"
            "ULTRA IMPORTANT — Phase A constraints for seed runs:\n"
            "  - DO NOT emit <assets> in this turn. The on-disk roster "
            "is the truth source.\n"
            "  - DO NOT emit <sounds> in this turn. Same reason.\n"
            "  - DO emit <plan>, <criteria>, <probes> focused on the "
            "code fix the user is requesting.\n"
            "  - Refer to existing media by their EXACT names; do not "
            "invent new names — the loader entries already map them.\n"
            "\n"
            f"Existing assets on disk: {_fmt_names(seed_a)}\n"
            f"Existing sounds on disk: {_fmt_names(seed_s)}\n"
            "\n"
            "If you genuinely need a brand-new visual entity the seed "
            "doesn't already have (rare on a seed restart — the user "
            "usually wants behavior/animation fixes, not new sprites), "
            "you can request it in a LATER mid-session <assets> turn "
            "AFTER iter 1 confirms the fix is working. Not now.\n"
        )

    minimal_nudge = ""
    if force_minimal_first_build:
        minimal_nudge = load_plan_nudge("minimal-first-build")
        _record_nudge("minimal-first-build")

    local_crisp_nudge = ""
    if model_class == "small":
        local_crisp_nudge = load_plan_nudge("local-plan-crisp")
        _record_nudge("local-plan-crisp")

    movement_probe = input_moves_player_probe_expr(goal=goal)
    plan_core = PLAN_INSTRUCTION.replace("{input_moves_player_probe}", movement_probe)

    body = (
        local_crisp_nudge
        + plan_core + art_nudge + illustrated_sprite_nudge + franchise_asset_nudge
        + canvas_entity_nudge + pinball_nudge + open_field_td_nudge
        + threed_nudge + wireframe_nudge
        + beat_em_up_nudge + audio_nudge
        + video_nudge
        + qte_nudge
        + spatial_alignment_nudge
        + point_and_click_nudge
        + scope_nudge + multi_frame_nudge + minimal_nudge + seed_nudge
    )

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


PLAN_INSTRUCTION = """CRITICAL FORMAT: output ONLY the structured tags
below. No prose, bullets, or planning essay outside <plan>, <criteria>,
<probes>, and optional <assets>/<sounds>/<videos>. Write each tag once.

Before writing any code, output a short plan, a list of acceptance
criteria, and a JSON list of EXECUTABLE probes the verifier will
literally run on your game each iteration.

Use this exact format and nothing else:

<plan>
Mechanics: <one or two sentences>
Controls: <keys / mouse / touch — use e.code names like KeyW, ArrowUp, Space>
Win/lose: <how the game ends, how the player restarts>
Visual style: <colors, vibe, single line>
Risky bits: <2 to 3 things you'll need to be careful about>
Build order: <2 to 4 implementation steps, in order, on ONE line — e.g. "scaffold + state → input/movement → collisions → win/lose + restart">
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

CRITERIA-PROBE BINDING: every Basic: line MUST share at least one
meaningful word with some probe name or expr (the harness uses simple
word-overlap; unbound Basic criteria are flagged as coverage gaps).
Edge/Stress lines may stay broader.

<probes>
[
  {"name": "short-id", "expr": "<JS expression that evaluates truthy when the criterion is satisfied; runs in the page context after ~3s of warmup>"},
  {"name": "...", "expr": "..."}
]
</probes>

PROBES come in two kinds and you MUST emit BOTH:

STRUCTURAL probes check that things EXIST (cheap sanity). Examples:
  - {"name":"canvas_present", "expr":"!!document.querySelector('canvas')"}
  - {"name":"ship_visible",   "expr":"window.state && state.player && state.player.x>=0 && state.player.x<=800"}
  - {"name":"non_blank",      "expr":"(()=>{const c=document.querySelector('canvas');if(!c||!c.width||!c.height)return false;try{return c.toDataURL().length>200;}catch(e){return true;}})()"}

DYNAMIC probes check that the game actually PLAYS — they simulate an
input and assert a state DELTA over time. Structural probes alone are
NOT enough: a game that renders a static HUD and never responds to a
key passes every structural probe. You MUST include at least 1 to 2
dynamic probes. Copy this template and swap in your real control key
and state path:
  - {"name":"input_moves_player", "expr":"{input_moves_player_probe}"}
  - {"name":"score_on_forced_event", "expr":"(async()=>{/* CAUSE the event first — e.g. place food on next cell / enemy on crosshair with hp=1, then key — never idle-wait */ const s0=state.score; /* setup + input here */; await new Promise(r=>setTimeout(r,400)); return state.score>s0;})()"}
A dynamic probe records a value, dispatches the input (KeyboardEvent
keydown/keyup with the e.code your game listens for, or calls an
exposed control like window.game.fire()), `await`s a short timeout so a
frame advances, then returns whether the value changed. Do NOT bare-
wait for score/food/kill/collision — FORCE the precondition (food cell,
enemy on crosshair + hp=1, clear serving), then assert.

Probes that reference globals (state, player, etc.) MUST exist on
window — either expose them (e.g. `window.state = state`) or use DOM /
canvas-pixel checks instead. Aim for 3 to 5 probes total — mostly
structural, but at least one input→delta dynamic probe. Keep each expr
short.

SELF-CONTAINED PROBES — never call bare helpers (`simulateClick`,
`distTo`) unless attached to `window`. Prefer inline KeyboardEvent /
PointerEvent / DOM `.click()` inside the expr.

PROBE ROBUSTNESS — test structure and behavior over time, not one
frame-specific pixel or exact helper names.

PROBE SYNTAX — each `expr` is eval'd in the page: valid JS, balanced
`()`, `[]`, `{}`; async probes close the IIFE `(async()=>{...})()`.

EXPECTED — emit an <assets> block whenever the game has visual entities
the player will see (sprites, characters, projectiles, terrain).

<assets>
[
  {"name": "ship",     "prompt": "pixel-art retro arcade spaceship facing right, white outline, transparent background"},
  {"name": "asteroid", "prompt": "pixel-art irregular grey rocky asteroid, transparent bg", "size": "64x64"}
]
</assets>

The harness saves generated PNGs next to your HTML file. Paths return in
the first-build prompt; load with `new Image()` + `await img.decode()`.

SKIP <assets> ONLY for pure-DOM apps. If canvas entities have visual
character, EMIT <assets> (do not silently downgrade to bare rectangles).

EXPECTED ALSO — emit a <sounds> block whenever the game has audible
events the player will hear (firing, hits, pickups, explosions, jumps)
or when a looping background track improves the game feel.

<sounds>
[
  {"name": "laser",     "prompt": "short retro arcade laser shot, 8-bit synth blip", "duration": 0.4},
  {"name": "explosion", "prompt": "short pixelated explosion, 8-bit boom",            "duration": 0.8},
  {"name": "music",     "prompt": "loopable 8-bit chiptune background, 90 bpm",       "duration": 12.0, "loop": true}
]
</sounds>

The harness saves generated OGGs next to your HTML file. Paths return in
the first-build prompt; load with `new Audio(path)` and play on events.

SKIP <sounds> only for pure-DOM apps where audio adds nothing.
Otherwise, emit sounds by default.

PLAN QUALITY — examples (good vs low-quality):

GOOD plans name the mechanism + specific controls + win/lose + risky bits.
Each line carries something testable. Two verbatim examples (not templates,
no game names — describe shape, not subject):

  Mechanics: DDA raycaster over a 16x16 tile grid; entities as billboard sprites.
  Controls: ArrowUp/Down forward/back along facing; mouse yaw under pointer lock; Space fires.
  Win/lose: reach exit tile = win; health=0 = lose; R restarts.
  Risky bits: floor/ceiling perf, sprite z-sort, listener cleanup on restart.

  Mechanics: 2D side-scrolling platformer with gravity + jump physics; two enemy types patrolling on platforms.
  Controls: ArrowLeft/Right walk; Space jump; R restart.
  Win/lose: reach right edge of last level = win; touch enemy = lose.
  Risky bits: corner-snap collision, double-jump guard, restart resets enemy positions.

LOW-QUALITY plans (these all fail because nothing is testable):
  - "Build a 3D first-person game."
  - "Add a side-scroller with platforms and enemies."
  - "Make it playable with arrow keys."

If your <plan> reads more like the LOW-QUALITY examples, rewrite before
sending. Specificity here saves iterations later.

No <html_file> yet.
"""


def measure_plan_reply(reply: str) -> tuple[int, int]:
    """Return (prose_chars, canonical_chars) for a Phase-A planning reply.

    prose_chars counts text outside structured tags; canonical_chars counts
    the bodies inside <plan>/<criteria>/<probes>/<assets>/<sounds>/<videos>.
    """
    if not reply:
        return 0, 0
    tag_re = _re.compile(
        r"<(plan|criteria|probes|assets|sounds|videos)>(.*?)</\1>",
        _re.DOTALL | _re.IGNORECASE,
    )
    canonical = 0
    for m in tag_re.finditer(reply):
        canonical += len(m.group(2))
    prose = reply
    for m in tag_re.finditer(reply):
        prose = prose.replace(m.group(0), "")
    return len(prose.strip()), canonical


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


def generated_sprite_draw_contract() -> str:
    """Genre-free draw contract injected when session assets exist."""
    return (
        "GENERATED-SPRITE DRAW CONTRACT — PNG paths were returned above. "
        "Every entity that has a generated sprite MUST be drawn with "
        "`ctx.drawImage` via `sprite(key)` (or `ASSETS[key]` after "
        "`img.decode()`). Replace the seed's procedural player/enemy/piece "
        "draw bodies (`fillRect`, `arc`, Unicode glyphs) with drawImage. "
        "The seed boilerplate is safe to keep ONLY for canvas sizing, RAF, "
        "input map, and HUD plumbing — NOT for entity draw. Procedural "
        "shapes are allowed ONLY as a labeled MISSING fallback while an "
        "image is still decoding, or for entities with NO generated PNG "
        "(grid lines, particles, health bars). `loadAssets()` alone is "
        "insufficient — each entity draw path must call drawImage when the "
        "sprite is ready.\n"
        # run_13: 3/10 games burned iter 1 on ASSETS_LOADED_BUT_UNDRAWN.
        "SELF-CHECK before emitting: every name listed in GENERATED ASSETS "
        "must appear in a draw call (`sprite('name')` or "
        "`drawImage(ASSETS.name` / `ASSETS['name']`) in this first build — "
        "loading without drawing fails the undrawn-art gate.\n"
        "Expose `window.state = state` (or `window.gameState = state`) "
        "after init so behavioral probes can read player position, score, "
        "and game flags."
    )


def first_build_instruction(
    seed_html: str,
    seed_source: str | None = None,
    *,
    playbook_block: str = "",
    current_asset_dir: str | None = None,
    current_sound_dir: str | None = None,
    has_generated_assets: bool = False,
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
    # Frame the seed differently depending on its provenance. The
    # bundled default (canvas_basic) is just plumbing — safe to follow
    # closely. A past session is NOT a quality guarantee — those files
    # are saved when STRUCTURAL probes pass and the model said done,
    # which does not mean the game worked. Telling the model to "KEEP
    # its structure" against a buggy past session was actively harmful
    # (2026-05-15 user pushback: "if they could mislead, fix that").
    if not seed_source:
        if has_generated_assets:
            seed_framing = (
                "Start from the SEED CODE below — it is the bundled empty "
                "scaffold (canvas + DPR scaling, RAF loop with delta-time, "
                "input map, score HUD, pause, game-over modal). It has no "
                "game logic of its own. KEEP the canvas/RAF/input/HUD "
                "plumbing, but REPLACE any procedural entity draw bodies "
                "(fillRect/arc circles) with drawImage via sprite(key) — "
                "generated PNGs are listed above."
            )
        else:
            seed_framing = (
                "Start from the SEED CODE below — it is the bundled empty "
                "scaffold (canvas + DPR scaling, RAF loop with delta-time, "
                "input map, score HUD, pause, game-over modal). It has no "
                "game logic of its own. Use as much or as little of it as "
                "fits your goal; the boilerplate is safe to keep."
            )
    else:
        seed_framing = (
            "REFERENCE ONLY — the SEED CODE below is from a similar "
            f"past session ({seed_source}). It was saved because "
            "structural probes passed and the model declared done, "
            "NOT because the game worked correctly — past sessions "
            "routinely contain real bugs (broken collisions, wrong "
            "physics, missing mechanics). DO NOT preserve its "
            "structure or copy its logic blindly. Treat it as one "
            "example of the surrounding boilerplate (canvas + DPR + "
            "RAF + input + HUD) and write fresh game code that "
            "actually meets the goal. If anything in the seed "
            "conflicts with the goal, ignore it."
        )
    pb = ""
    if playbook_block:
        pb = (
            f"{playbook_block}\n\n"
            "Apply relevant playbook entries on the first attempt.\n\n"
        )
    asset_contract = ""
    if has_generated_assets:
        asset_contract = (
            generated_sprite_draw_contract() + "\n\n"
        )
    base = (
        "Plan accepted. The plan, criteria, and probes are fixed — do NOT "
        "restate requirements or re-plan.\n\n"
        f"{pb}"
        f"{asset_contract}"
        f"{seed_framing}\n\n"
        "FORMAT CONTRACT: a brief reasoning preamble is fine, but your FIRST "
        "output tag MUST be `<html_file>` and it must contain the COMPLETE "
        "game. Emit a RAW `<html_file>` block — do NOT wrap the HTML in a "
        "markdown ```html fence.\n"
        "SELF-START: after loading assets, call requestAnimationFrame(loop) "
        "unconditionally at the end of init so the animation loop actually "
        "runs.\n"
        "Expose `window.state = state` (or `window.gameState = state`) after "
        "init — probes read window.state first.\n\n"
        "Output the COMPLETE file in <html_file>...</html_file> tags. "
        "ULTRA IMPORTANT: emit the COMPLETE file, no elisions, no "
        "'rest of code unchanged' placeholders. Then add a <notes> tag "
        "with one sentence of summary.\n\n"
        # Fix round (fight trace 20260611_145321): a rewrite dropped the
        # seed's canvas sizing → 300x150 browser default → black screen.
        "KEEP the seed's canvas sizing: the width/height attributes and "
        "any fit()/DPR-resize wiring must survive your rewrite (or be "
        "replaced with equivalent explicit sizing, e.g. <canvas "
        'width="800" height="500">). A <canvas> with no size renders at '
        "the useless 300x150 browser default.\n\n"
        "SEED CODE:\n"
        "```html\n"
        f"{seed_html}\n"
        "```\n"
    )
    return base


def seed_build_instruction(
    seed_html: str,
    seed_path: str,
    *,
    playbook_block: str = "",
    truncated: bool = False,
) -> str:
    """First-build prompt when the user explicitly hands us a starting file.

    Differs from `first_build_instruction`: the file is the user's own
    working code (not a memory skeleton), so we strongly prefer
    <patch> blocks over a full rewrite.

    When `truncated` is True the `seed_html` passed in is only an EXCERPT
    of a large file (the full file is on disk) — the wording changes so the
    model patches against the on-disk file rather than trusting the excerpt
    to be complete.
    """
    pb = f"{playbook_block}\n\n" if playbook_block else ""
    file_label = (
        "EXISTING FILE (EXCERPT — full file is on disk at the SEED PATH "
        "above; the middle was elided to save context. Patch against the "
        "on-disk file; if a patch SEARCH target is not shown below, request "
        "the region or send a complete <html_file>):\n"
        if truncated
        else "EXISTING FILE:\n"
    )
    return (
        "Plan accepted. The user is starting from an EXISTING game file "
        "they want you to ADAPT (not replace) to match the goal above.\n\n"
        f"{pb}"
        f"SEED FILE: {seed_path}  (already saved on disk as the working "
        "file)\n\n"
        "FORMAT-ONLY RULE: this is a code turn. Start your reply with "
        "either <patch> or <html_file> as the first non-whitespace token. "
        "No prose before the opening code tag.\n\n"
        "Strongly PREFER one or more <patch> SEARCH/REPLACE blocks "
        "against the code below. Patches are smaller, safer, and "
        "preserve the user's structure. Send a complete <html_file> "
        "ONLY if the goal genuinely requires structural changes patches "
        "can't reasonably express.\n\n"
        # DK trace 20260514_104131 fix: in /seed sessions the Phase-A
        # probes were authored BEFORE the model could see the file, so
        # 4/5 referenced state property names (`state.grid`,
        # `state.player.onLadder`, `state.reset`) that don't exist in
        # the seed. Invite a corrected probe list this turn — the
        # harness's iter-1 probe-reparse gate accepts a fresh
        # <probes>...</probes> block on seed iter 1.
        "Your Phase A <probes> were written WITHOUT seeing this file. "
        "If any probe expression references a state property, function, "
        "or DOM element name that does NOT exist in the code below, "
        "RE-EMIT a corrected <probes>...</probes> block alongside your "
        "patch — the harness will adopt the new probes for the "
        "iterations that follow.\n\n"
        "Always include a <notes> tag describing what each patch does.\n\n"
        f"{file_label}"
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
    "something will be rolled back. This is a code turn: first non-"
    "whitespace token must be <patch> or <html_file>, with no prose "
    "before it. Aim for the SMALLEST footprint that "
    "fixes every ERROR and ISSUE. Always include a <notes> tag explaining "
    "what changed."
)


def scope_reduction_instruction(report_text: str) -> str:
    """Replace diagnose-then-fix when the first build is structurally dead.

    Triggered by the dead-first-build detector when iter <= 2 ships a
    file but RAF never fires AND the input smoke test recorded no
    state/canvas delta. Patching a dead file is wasteful — the better
    move is to ship a smaller intentionally-minimal rewrite.

    Universal: keys on observable structural signals (raf_ran=false,
    input_test.any_change=false), no genre keywords. Reuses any prior
    <assets> / <sounds> on disk by reference; the model just emits new
    code that loads them.
    """
    return (
        f"{report_text}\n\n"
        "DEAD-FIRST-BUILD DETECTED: your file loaded in the browser "
        "but requestAnimationFrame NEVER fired AND the input smoke "
        "test produced zero state or canvas change. Nothing in the "
        "file is actually running. Patching on top of this will not "
        "fix it — the wiring is wrong at a level patches can't reach.\n\n"
        "Recovery for THIS turn (skip diagnose, skip patches):\n"
        "  - Emit ONE complete <html_file>...</html_file> that is "
        "INTENTIONALLY SMALLER than your previous attempt: pick the "
        "1-2 most essential features from your Phase A <plan> "
        "(renderer + one input is enough), and DEFER everything else "
        "to subsequent <patch> turns.\n"
        "  - Aim for at most ~10 KB of HTML — half what you wrote last "
        "time. A simpler scaffolding that actually runs beats an "
        "ambitious scaffolding that does nothing.\n"
        "  - Reuse the SAME asset / sound filenames as before — the "
        "diffuser already wrote those PNGs and OGGs, your loader "
        "code just needs to load and draw the ones you actually use "
        "this turn.\n"
        "  - Make sure (a) at least one `requestAnimationFrame(loop)` "
        "is called UNCONDITIONALLY at the end of init, (b) the "
        "keydown listener is on `window` (not the canvas), (c) "
        "`window.gameState = window.state = state` is set after init.\n"
        "  - Re-emit <probes> ONLY if you change which globals you "
        "expose. Otherwise leave them alone — they live in session "
        "state.\n\n"
        "Open <todos> items from the deferred features will guide "
        "later patch turns. Do not enumerate them in this reply — "
        "just ship the minimal core."
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
        "only if patches truly cannot express the change. If you emit "
        "a full <html_file> because the feedback changes the game's "
        "runtime shape, also emit a fresh <probes>...</probes> block "
        "for the new behavior; old probes may no longer apply.\n\n"
        # Surface the diffuser path on every post-ship turn. Without
        # this reminder the model defaults to a code-only mental model
        # and substitutes inline SVG / ctx primitives / AudioContext
        # beeps for what should be generated media (chess trace 2026-
        # 05-21 is the motivating case — model drew SVG instead of
        # asking the diffuser for sprites). Genre-free; works for any
        # goal, any subject matter.
        "If the feedback asks for ART / sprites / new visual entities "
        "(any genre, any subject), emit a fresh <assets> block in "
        "this turn — the harness runs Z-Image-Turbo and writes PNGs "
        "next to the HTML — plus ONE small <patch> that loads and "
        "draws them with `new Image()` + `ctx.drawImage`. Same shape "
        "for AUDIO: emit a <sounds> block + a small <patch> that "
        "plays `new Audio(path)` on the matching event. Do NOT "
        "substitute inline SVG, Unicode glyphs, ctx.fillRect, or "
        "synthesized AudioContext beeps for generated media when "
        "the user explicitly asked for art or sound.\n\n"
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
    context_pressure: bool = False,
    probe_failure_block: str = "",
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

    `probe_failure_block` — failing executable probes listed first so
    gameplay blockers beat cosmetic soft warnings.
    """
    probe_fail = ""
    if probe_failure_block:
        probe_fail = (
            "PROBE FAILURE — fix these before cosmetic warnings:\n"
            f"{probe_failure_block}\n\n"
        )
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
    if context_pressure:
        # Context-pressure escape hatch (Wolfenstein 2026-05-24 trace
        # lesson): the prior stream's prompt_tokens hit >=85% of the
        # model's context window. Including the full CURRENT FILE
        # block on top of that leaves the model no token budget to
        # emit a complete reply — output truncates, parser rejects,
        # the agent falls into an identical-reply loop. Drop the file
        # entirely and force a minimal patch. The matcher applies
        # SEARCH against the on-disk file regardless of whether the
        # model saw it inline; the model only needs enough context
        # to write a unique 3-5 line anchor.
        file_block = (
            "CONTEXT IS FULL: the previous turn used >=85% of the "
            "context window, so the CURRENT FILE ON DISK block has "
            "been OMITTED from this prompt to give your reply room "
            "to fit. Recovery shape for this turn:\n"
            "  - Emit ONE <patch>...</patch> with at most 3-5 lines "
            "of SEARCH context.\n"
            "  - Do NOT emit <html_file> — re-emitting the full file "
            "with this little headroom WILL truncate mid-stream and "
            "the parser will reject the result.\n"
            "  - Do NOT re-emit <probes> / <assets> / <sounds>; "
            "those live in session state.\n"
            "  - The patch matcher applies SEARCH against the file "
            "on disk regardless of whether you can see it here, so "
            "write a short distinctive anchor from memory.\n\n"
        )
    elif focused_slice:
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
        f"{probe_fail}"
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


def polish_instruction(
    report_text: str,
    *,
    current_file: str = "",
    critic_note: str = "",
    component_block: str = "",
    turn: int = 1,
    cap: int = 2,
) -> str:
    """Capability-round item 2: post-green polish turn.

    Sent instead of post_clean_instruction when probes pass, iteration
    budget remains, and the per-session polish cap is unmet. Asks for ONE
    concrete game-feel improvement via small <patch>es — never a
    restructure. Genre-free juice rubric; auto-revert guards regressions.
    """
    crit = ""
    if critic_note:
        crit = (
            "/vlm-critique finding (address it if it fits this "
            f"turn's ONE improvement): {critic_note}\n\n"
        )
    comp = f"{component_block}\n\n" if component_block else ""
    file_block = ""
    if current_file:
        file_block = (
            "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — your "
            "<patch> SEARCH must match THIS exact text, "
            "character-for-character):\n"
            "```html\n"
            f"{current_file}\n"
            "```\n\n"
        )
    return (
        f"{report_text}\n\n"
        f"All probes pass — the game WORKS. This is polish turn {turn}/{cap}: "
        "spend it on GAME FEEL, not features.\n\n"
        "Do NOT restructure. Pick exactly ONE concrete feel improvement and "
        "implement it with small <patch>es. Choose from this rubric (skip "
        "anything already present):\n"
        "- hit feedback on impactful events (flash, screen shake, hit-pause)\n"
        "- motion easing instead of linear snaps (UI, pickups, transitions)\n"
        "- particles on the most important event (death, score, impact)\n"
        "- audio cues on player actions (WebAudio beeps are fine — no files needed)\n"
        "- a title screen and/or game-over screen with a restart path\n"
        "- score presentation (pop on change, floating points, high score)\n\n"
        f"{crit}"
        f"{comp}"
        f"{file_block}"
        "If the game already feels complete and polished, send <done/> "
        "instead — shipping is always acceptable. Auto-revert protects the "
        "working build, but keep the change small and additive."
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


def patch_retry_instruction(
    failures: list,
    current_file: str,
    *,
    repeat_anchors: set[str] | None = None,
    anchor_fingerprint=None,
) -> str:
    """Re-prompt after one or more <patch> blocks failed to apply.

    For each failure, we attach an "anchor" — the small region of the
    current file the model was probably aiming at. The typical 27B-class
    failure is "model copied an old version of a line that's since been
    edited"; showing the actual lines (with line numbers, ▸ marker on
    the closest hit) is far more useful than the bare reason string.
    Anchor lookup is best-effort; when it returns None the bullet just
    carries the reason as before.

    `repeat_anchors`: fingerprints of SEARCH blocks that ALSO failed
    on the previous retry turn. When a current failure matches one of
    these, prepend [REPEATED FAILURE] to the bullet so the model gets
    a louder signal than "re-read the file" — that signal already
    fires every turn and clearly isn't enough on its own. Motivating
    trace: doom 2026-05-23 extensions 1/2/3 where the same
    `spriteNames=[...]` SEARCH failed three turns in a row because
    the FIRST patch already changed that line.
    """
    from patches import find_anchor  # local import: avoid cycle at module load

    repeat_anchors = repeat_anchors or set()
    bullets: list[str] = []
    repeat_count = 0
    for (i, p, reason) in failures:
        fp = ""
        if anchor_fingerprint is not None:
            try:
                fp = anchor_fingerprint(getattr(p, "search", "") or "")
            except Exception:
                fp = ""
        is_repeat = bool(fp) and fp in repeat_anchors
        if is_repeat:
            repeat_count += 1
            line = (
                f"  - patch #{i+1} [REPEATED FAILURE — same SEARCH "
                f"failed last turn]: {reason}"
            )
        else:
            line = f"  - patch #{i+1}: {reason}"
        anchor = find_anchor(current_file, getattr(p, "search", "") or "")
        if anchor:
            line += "\n    nearest match in current file:\n"
            line += "\n".join(f"      {ln}" for ln in anchor.splitlines())
        bullets.append(line)
    bullets_block = "\n".join(bullets)
    header = (
        "Some of your <patch> blocks did not apply because the SEARCH "
        "text was not found verbatim in the file. The CURRENT FILE "
        "below is the truth — match it character-for-character:\n\n"
    )
    if repeat_count:
        header = (
            "================ REPEATED PATCH FAILURE ================\n"
            f"{repeat_count} of your <patch> blocks re-emitted the SAME "
            "SEARCH text that already failed last turn. The file CHANGED "
            "between then and now (likely a sibling patch in the prior "
            "reply landed and shifted lines, OR you copied from a stale "
            "memory of the file). DO NOT re-send the same SEARCH; read "
            "the CURRENT FILE below FIRST, find the actual lines you "
            "want to change as they exist NOW, then write a fresh "
            "SEARCH block from those lines.\n"
            "========================================================\n\n"
        ) + header
    return (
        f"{header}"
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
        "LAST KNOWN-GOOD VERSION (this passed all tests). Your last patch "
        "broke a WORKING build — revert to behavior similar to the version "
        "below (or re-emit it as the COMPLETE game in "
        "<html_file>...</html_file>):\n\n"
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

EXCEPTION — unmet user requests are NOT "nice to have": before you
confirm, re-read the user's most recent feedback and check that EVERY
distinct item they explicitly asked for is actually implemented AND
still works in the current file (e.g. "animate the kick" AND "add a CPU
opponent" is two items). An item the user explicitly requested that is
missing, broken, or silently dropped DOES qualify for a <patch> — fix it
rather than confirming done. Animation that renders as a single static
held pose counts as unmet if the user asked for animation.

When in doubt, ship. Working > perfect. Reply with EXACTLY ONE of:

  (a) <confirm_done/>          — default; the game works, we are done.
  (b) one or more <patch> blocks plus a <notes> tag naming the specific
      crash bug being fixed.
"""


# ===========================================================================
# Read-only /ask turn (TUI slash command)
# ===========================================================================

_ASK_HTML_EXCERPT_MAX = 10_000


def ask_instruction(
    *,
    question: str,
    goal: str = "",
    criteria: str = "",
    report_text: str = "",
    html_excerpt: str = "",
    asset_names: list[str] | None = None,
) -> str:
    """One-turn user message for `/ask` — explain the current game only.

    Not part of ALL_FORMATS; injected ephemerally by GameAgent.run_ask_turn.
    """
    assets = ", ".join(asset_names or []) or "(none)"
    crit_block = criteria.strip() or "(not set)"
    report_block = report_text.strip() or "(no test report yet)"
    html_block = (html_excerpt or "(no file)").strip()
    if len(html_block) > _ASK_HTML_EXCERPT_MAX:
        html_block = (
            html_block[:_ASK_HTML_EXCERPT_MAX]
            + f"\n\n[... truncated {len(html_excerpt) - _ASK_HTML_EXCERPT_MAX} chars ...]"
        )
    return (
        "READ-ONLY ASK TURN — answer the user's question about the CURRENT "
        "game. Do NOT change code, do NOT emit <patch>, <html_file>, "
        "<assets>, <sounds>, <videos>, <done/>, or <confirm_done/>. "
        "Plain prose only. Cite function names or logic from the snippet "
        "when you can. If the snippet does not contain enough information, "
        "say what is missing instead of guessing.\n\n"
        f"GOAL: {goal.strip() or '(not set)'}\n\n"
        f"ACCEPTANCE CRITERIA (Phase A):\n{crit_block}\n\n"
        f"LAST TEST REPORT:\n{report_block}\n\n"
        f"SESSION ASSET NAMES: {assets}\n\n"
        f"CURRENT best.html EXCERPT:\n{html_block}\n\n"
        f"USER QUESTION: {question.strip()}"
    )


# ===========================================================================
# VLM screenshot review note
# ===========================================================================

VLM_REVIEW_NOTE = (
    "A SCREENSHOT of the current game state is attached above. Use it "
    "to guide your patch. Look for: missing UI, weird positions, wrong "
    "colors, off-canvas content, blank areas. Mention what you SEE in "
    "<notes>."
)
