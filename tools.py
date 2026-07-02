"""Browser test harness for the coding-box agent.

Single public function: `test_html_file(path, run_seconds=3.0) -> dict`.

It launches a headless Chromium via Playwright, loads the file, lets it
animate for a few seconds (so requestAnimationFrame loops actually tick),
then returns a SHORT report. Keeping the report short matters: a small
model gets confused by huge logs, so we cap and truncate aggressively.

Also exposes `run_micro_probes(html: str) -> dict` — fast pre-flight
checks (HTML structure, script presence, bracket balance) that run
BEFORE the Chromium round-trip. OpenCoder's "Educational-Instruct" lesson:
cheap execution filters, often. A micro-probe failure is structurally
unrecoverable (truncated stream, syntactically broken script) and
doesn't need a 3+ second browser load to confirm.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter as _Counter
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


# Cap how much we feed back to the model. Smaller is better for small models.
_MAX_MSGS = 12          # at most this many console lines forwarded
_MAX_MSG_LEN = 240      # truncate each line to this many chars
_MAX_BODY_TEXT = 200    # tiny snippet of body text


def _input_evidence_is_plausible(path: str) -> bool:
    """Return True when a state delta is likely caused by held input.

    Keep this structural, not genre-specific. Per-entity array-member
    deltas (`objects.3.x`, `things.0.age`) are often autonomous motion
    or animation noise, so they are weak proof that a key worked. Direct
    object fields (`player.x`, `camera.zoom`, `cursor.angle`) and array
    length changes (`shots.length`) remain useful input evidence.
    """
    parts = [p for p in str(path or "").split(".") if p]
    if not parts:
        return False
    if parts[-1] == "length":
        return True
    if any(p.isdigit() for p in parts[:-1]):
        return False
    return True


# Leaf names that represent a movable entity's POSITION. A movement key that
# registers input (sets a direction/flag) but never changes any of these means
# the entity is STUCK (spawned in a wall, collision blocking every move) — the
# Pac-Man "starts in a wall and doesn't move" failure that read as "responsive"
# because a `dir`/`nextDir` field changed. Genre-free: position field names are
# universal across tile and pixel games.
_POSITION_LEAF_NAMES = {
    "x", "y", "tx", "ty", "gx", "gy", "gridx", "gridy", "col", "row",
    "px", "py", "posx", "posy", "cx", "cy", "tilex", "tiley",
    "worldx", "worldy", "left", "top", "row", "column",
    # 3D games move on the x/z ground plane (FPS trace 20260611_163325:
    # playerPos.z wasn't recognized as position, so a healthy first-person
    # game tripped both PLAYER-STUCK and CONTROL-NOT-RECOVERED).
    "z", "tz", "gz", "pz", "posz", "cz", "tilez", "worldz",
}


def _is_position_leaf(path: str) -> bool:
    """True when the final component of a dotted state path is a position field
    (x/y/tx/ty/gridX/col/row/...). Used to tell 'the player actually moved'
    from 'a key registered but the player is stuck'."""
    parts = [p for p in str(path or "").split(".") if p]
    if not parts:
        return False
    leaf = parts[-1]
    if leaf.lower() in _POSITION_LEAF_NAMES:
        return True
    # camelCase coordinate compound: a lowercase letter immediately followed by
    # a trailing single capital X/Y/Z (heroX, playerY, enemyZ) — the entity's
    # own position field. Genre-free: matches the universal "<entity>X" naming,
    # not all-lowercase words like `index`/`prefix`/`max` (those end in a
    # lowercase letter, so they never match). Trace pin: a Dragon's-Lair QTE
    # nudged state.heroX/heroY on arrows but heroX was not recognized as a
    # position field, so a real player move read as "no position changed" and
    # falsely tripped PLAYER-STUCK (phase-a-requirement-your-plann_20260615_121048).
    return bool(re.search(r"[a-z][XYZ]$", leaf))


# Board / grid games often expose selection via cursor.r/c or selected.row/col
# instead of player.x/y. Arrows moving that cursor IS valid navigation — not
# PLAYER-STUCK. Parent names are structural (not genre); leaf names are the
# usual grid indices. Trace pin: holochess iter 3 (cursor.r/c changed but
# harness wanted player.x/y and falsely gated ok=False).
_BOARD_NAV_PARENTS = frozenset({
    "cursor", "selection", "selected", "select", "hover", "focus",
    "highlight", "active", "picked", "cell", "square", "tile",
})
_BOARD_NAV_LEAF_NAMES = frozenset({
    "r", "c", "row", "col", "column", "x", "y", "index", "idx", "tile",
})


def _is_board_navigation_leaf(path: str) -> bool:
    """True when a dotted state path is a grid/board selection cursor field."""
    parts = [p for p in str(path or "").split(".") if p]
    if len(parts) < 2:
        return False
    parent = parts[-2].lower()
    leaf = parts[-1].lower()
    return parent in _BOARD_NAV_PARENTS and leaf in _BOARD_NAV_LEAF_NAMES


def _is_effective_movement_leaf(path: str) -> bool:
    """Position leaf OR board-cursor leaf — both count as the entity moving."""
    return _is_position_leaf(path) or _is_board_navigation_leaf(path)


# Pose/animation asset suffixes that legitimately stay undrawn on a static
# first board (checkers iter 2: hop_up/hop_land idle on opening position).
_ANIMATION_POSE_SUFFIXES = (
    "_hop", "_hop_up", "_hop_land", "_hop_down",
    "_lift", "_slam", "_walk1", "_walk2", "_walk3",
    "_jump", "_duck", "_punch", "_kick", "_run1", "_run2",
)


def _stem_looks_like_animation_pose(stem: str) -> bool:
    """True when an asset stem names a state-conditional pose frame."""
    low = str(stem or "").lower()
    return any(low.endswith(sfx) or sfx[1:] in low for sfx in _ANIMATION_POSE_SUFFIXES)


def _idle_counterpart_drawn(stem: str, drawn_blob: str) -> bool:
    """True when a pose frame's idle/base sprite appears in drawn sources."""
    low = str(stem or "").lower()
    if not low:
        return False
    # gold_gumdrop_hop_up -> gold_gumdrop_idle
    base = low
    for sfx in _ANIMATION_POSE_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    if not base.endswith("_"):
        base += "_"
    return (base + "idle") in drawn_blob or base.rstrip("_") in drawn_blob


def _undrawn_are_animation_poses_only(undrawn: list[str], drawn_blob: str) -> bool:
    """All undrawn assets are pose frames and their idle counterparts drew."""
    if not undrawn:
        return False
    return all(
        _stem_looks_like_animation_pose(s) and _idle_counterpart_drawn(s, drawn_blob)
        for s in undrawn
    )


# Stems that structurally cannot draw in a short smoke window (boss, death pose,
# late-wave enemy) — advisory when behavioral probes already pass.
_UNDRAWN_STATE_GATED_RE = re.compile(
    r"(boss|death|dead|victory|defeat|wave\d|phase\d|_lift|_slam|_hop|"
    r"_attack|_hit|_hurt|_spawn|promoted|king_crown|frightened|powered)",
    re.I,
)


def _undrawn_likely_state_gated(undrawn: list[str]) -> bool:
    """True when every undrawn asset name looks state/wave gated."""
    if not undrawn:
        return False
    return all(_UNDRAWN_STATE_GATED_RE.search(str(s) or "") for s in undrawn)


def _drawn_blob_contains_asset_fname(drawn_blob: str, fname: str) -> bool:
    """True when drawImage event sources include this exact PNG filename.

    Uses a path-boundary match so short stems like ``idle.png`` do not
    false-positive against ``blue_idle.png`` substrings.
    """
    low = str(drawn_blob or "").lower()
    fn = str(fname or "").lower()
    if not fn.endswith(".png"):
        stem = fn.rsplit(".", 1)[0] if "." in fn else fn
        fn = f"{stem}.png"
    return bool(re.search(
        r"(?:^|[/\\?#])" + re.escape(fn) + r"(?:[?#&]|$)",
        low,
    ))


def _patch_probe_pointer_board_clicks(expr: str) -> str:
    """Upgrade mousedown/click dispatches in board probes to pointer events.

    Checkers trace run_05: games listen for pointerdown but model probes
    dispatch MouseEvent('mousedown') / click — selection never fires.
    Prepends a harness helper; also fires pointerdown+pointerup before any
    canvas mousedown/click dispatch when sx/sy (or clientX/Y) are in scope.
    """
    e = str(expr or "")
    if "dispatchEvent" not in e or "MouseEvent" not in e:
        return e
    helper = (
        "window.__harnessPointerClick=(c,x,y)=>{if(!c)return;"
        "const o={clientX:x,clientY:y,bubbles:true,cancelable:true,"
        "pointerId:1,pointerType:'mouse',isPrimary:true,button:0};"
        "c.dispatchEvent(new PointerEvent('pointerdown',{...o,buttons:1}));"
        "c.dispatchEvent(new PointerEvent('pointerup',{...o,buttons:0}));};"
        "window.__harnessOccupiedBoardClick=(c)=>{"
        "const s=window.state||window.gameState||window.g||{};"
        "const b=s.board||s.grid||s.cells;"
        "if(!c||!Array.isArray(b)||!b.length||!Array.isArray(b[0]))return null;"
        "const rect=c.getBoundingClientRect();"
        "const rows=b.length,cols=b[0].length;"
        "for(let r=0;r<rows;r++)for(let c0=0;c0<cols;c0++){"
        "const cell=b[r][c0];"
        "if(cell==null||cell===0||cell==='.'||cell===''||cell===' ')continue;"
        "const x=rect.left+rect.width*((c0+0.5)/cols);"
        "const y=rect.top+rect.height*((r+0.5)/rows);"
        "window.__harnessPointerClick(c,x,y);return{r,c:c0,x,y};}"
        "return null;};"
    )
    out = helper + e
    # Prefer an occupied board cell when the probe hardcodes canvas fractions.
    if "getBoundingClientRect" in out:
        out = out.replace(
            "const r=c.getBoundingClientRect();",
            "const _occ=window.__harnessOccupiedBoardClick&&"
            "window.__harnessOccupiedBoardClick(c);"
            "const r=c.getBoundingClientRect();"
            "if(_occ){await new Promise(r=>setTimeout(r,50));"
            "return !!(window.state&&(window.state.selected||window.state.selection));}",
            1,
        )
    # Mirror pointer events for games that only wire pointerdown handlers.
    for evt in ("mousedown", "click"):
        token = f"c.dispatchEvent(new MouseEvent('{evt}'"
        if token in out:
            out = out.replace(
                token,
                f"window.__harnessPointerClick(c,sx,sy);{token}",
                1,
            )
    return out


def _is_webgl_or_three_game(html_text: str) -> bool:
    """Genre-free detector for three.js / WebGL canvas games."""
    low = (html_text or "").lower()
    return (
        "three." in low
        or "webglrenderer" in low
        or "babylon." in low
        or "playcanvas" in low
    )


def _threejs_manual_navigation_basis_risk(html_text: str) -> bool:
    """True when three.js FPS likely uses manual sin/cos that mismatches camera."""
    if not _is_webgl_or_three_game(html_text):
        return False
    if "applyquaternion" in html_text.lower() or "getworlddirection" in html_text.lower():
        return False
    has_camera_yaw = "camera.rotation.y" in html_text or "camera.rotation.set" in html_text
    has_manual_sin = "math.sin(yaw)" in html_text.lower() or "math.sin(state.player.yaw)" in html_text.lower()
    # Old wrong pattern: fx = +Math.sin(yaw) with three.js camera
    has_plus_sin = "fx = math.sin" in html_text.lower() or "fx=math.sin" in html_text.lower()
    return bool(has_camera_yaw and has_manual_sin and has_plus_sin)


# Keys that should MOVE a controllable player. If one of these registers input
# (a direction/flag changes) but no position leaf ever changes, the player is
# stuck. Attack/ability keys are deliberately excluded.
_MOVEMENT_KEYS = frozenset({
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "KeyW", "KeyA", "KeyS", "KeyD",
})

# Restart/menu keys declared in criteria — pressed during smoke tests but
# must not be held during combat inducement (resets the match before the
# control-recovery re-test) or count as sprite-driven actions.
_RESTART_KEYS = frozenset({"KeyR", "Enter"})


def control_not_recovered_verdict(
    *,
    has_position_state: bool,
    moved_early: bool,
    recheck_moved: bool,
    retry_moved: bool | None,
) -> bool:
    """Pure decision for the control-recovery re-test (fix-round item 1).

    True only when a movement key moved the player's position EARLY in the
    run, the post-gameplay re-dispatch no longer moves it, AND a single
    retry (after a grace wait, so legitimate brief hit-stun expires) is
    still frozen. Catches the permanent stun-lock family: a hit/stun state
    whose expiry timer sits BELOW an early-return guard, locking the
    entity forever (trace build-a-single-screen-2d-fight_20260610_185238).

    - no position state (menu/board games)  -> False (not applicable)
    - never moved early                     -> False (PLAYER-STUCK's job)
    - recheck moved                         -> False (controls recovered)
    - recheck frozen, retry moved           -> False (transient hit-stun)
    - recheck frozen, retry frozen          -> True
    """
    if not has_position_state or not moved_early:
        return False
    if recheck_moved:
        return False
    return retry_moved is False


_PROBE_EVAL_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("syntax_error", re.compile(r"\bSyntaxError\b|missing \) after argument list", re.I)),
    ("reference_error", re.compile(r"\bReferenceError\b", re.I)),
    ("type_error", re.compile(r"\bTypeError\b", re.I)),
)


def _classify_probe_eval_error(err: str) -> str | None:
    """Return a compact class when a probe failed before yielding a value.

    These are harness/probe-expression failures, not game-state falsy
    results. Keeping the classifier small prevents ordinary failed game
    assertions from being softened.
    """
    if not err:
        return None
    for label, pat in _PROBE_EVAL_ERROR_PATTERNS:
        if pat.search(err):
            return label
    return None


_ARROW_IIFE_RE = re.compile(
    r"^\(\s*(?:async\s*)?\(?[^=(){};]*\)?\s*=>.*\)\s*$",
    re.DOTALL,
)
_FUNCTION_IIFE_RE = re.compile(
    r"^\(\s*(?:async\s+)?function\b.*\)\s*$",
    re.DOTALL,
)


def _normalize_probe_expr(expr: str) -> str:
    """Normalize model-authored probe expressions before Boolean wrapping.

    Strong models sometimes emit a bare IIFE expression with a trailing
    semicolon, e.g. `(()=>{ ... });`. Wrapping that directly as
    `Boolean(EXPR)` creates `Boolean((()=>{...});)` and throws a
    SyntaxError. Strip one trailing semicolon and invoke bare function
    expressions so the probe evaluates to the promised result.
    """
    out = str(expr or "true").strip()
    if out.endswith(";"):
        out = out[:-1].rstrip()
    # Already-invoked IIFEs end with `)()` or `)(...)`; leave them alone.
    already_invoked = bool(re.search(r"\)\s*\([^)]*\)\s*$", out))
    if not already_invoked and (
        _ARROW_IIFE_RE.match(out) or _FUNCTION_IIFE_RE.match(out)
    ):
        out += "()"
    return out or "true"


# Trace 20260612_171752: probe-ordering fix. A side-effecting probe
# (restart_resets dispatched KeyR → game reset() zeroed state.frame) ran
# right before a read-only probe (raf_firing: `frame > 0`), which then read
# the freshly-reset state and failed for 3 straight iterations — while the
# report's own state timeline showed the frame counter moving (81→267).
# These needles identify probes whose evaluation can MUTATE page state (or
# at least pass wall-clock time): dispatching input events, calling
# reset/restart/click, or awaiting timers. Read-only probes are run before
# any of these so they observe undisturbed game state.
_PROBE_SIDE_EFFECT_NEEDLES = (
    "dispatchEvent",
    "KeyboardEvent",
    "MouseEvent",
    "PointerEvent",
    ".reset(",
    ".restart(",
    ".click(",
    "await ",
)


def _probe_has_side_effects(expr: str) -> bool:
    """True when a model-authored probe expression may mutate page state
    (dispatches events, calls reset/restart/click) or awaits timers.
    Conservative: false positives only delay a probe to the second group,
    they never change its result.
    """
    e = expr or ""
    return any(n in e for n in _PROBE_SIDE_EFFECT_NEEDLES)


def _format_probe_failure_warning(p: dict[str, Any]) -> str:
    """Human/model-facing warning for one failed probe."""
    name = p.get("name", "probe")
    expr = (p.get("expr") or "")[:80]
    err = p.get("err") or "evaluated falsy"
    if p.get("kind") == "eval_error":
        error_class = p.get("error_class") or "eval_error"
        return (
            f"PROBE BROKEN [{name}]: `{expr}` — {err}. "
            f"The probe expression itself errored at eval time "
            f"({error_class}). This is the probe, not the game. "
            "Re-emit a corrected <probes>...</probes> block alongside "
            "your next code change; new probes are adopted when they "
            "accompany code."
        )
    return (
        f"PROBE FAILED [{name}]: `{expr}` — {err}. "
        "Your Phase A acceptance criterion is unmet; fix the "
        "game so it evaluates truthy."
    )


def screenshot_delta(prev_png: bytes | None, curr_png: bytes | None) -> float | None:
    """Mean per-pixel RGB delta between two PNGs, normalized to [0, 1].

    Returns None when either input is missing or PIL is unavailable.
    Used by the loop's visual-regression detector: a high delta paired
    with a "no progress" verdict from the vision judge implies the
    iter visibly changed the canvas without making it better — i.e.,
    a regression. The threshold lives in the caller.

    Implementation note: both screenshots are resized to a small fixed
    resolution before differencing so the comparison is fast and
    independent of the original game's canvas size. We use 128×128
    because that's enough to capture overall composition while keeping
    the diff under 1 ms on the harness machine.
    """
    if not prev_png or not curr_png:
        return None
    try:
        from io import BytesIO

        from PIL import Image
    except Exception:
        return None
    try:
        a = Image.open(BytesIO(prev_png)).convert("RGB").resize((128, 128))
        b = Image.open(BytesIO(curr_png)).convert("RGB").resize((128, 128))
    except Exception:
        return None
    pa = a.tobytes()
    pb = b.tobytes()
    if len(pa) != len(pb) or not pa:
        return None
    total = 0
    for x, y in zip(pa, pb):
        d = x - y
        total += d if d >= 0 else -d
    return total / (len(pa) * 255.0)


def _canvas_hash_distance(a: str | None, b: str | None) -> float | None:
    """Fraction of cells that differ between two `_CANVAS_HASH_JS` strings.

    `_CANVAS_HASH_JS` returns a comma-joined string of 1024 (32×32) base36
    color tokens. This returns the share of those cells whose token changed,
    in [0, 1] — a genre-free *magnitude* of canvas change.

    Returns None when either input is missing/empty or the two have a
    different cell count (different canvas → not comparable). Used to find
    the frame of PEAK input-attributable visual change (the "action frame")
    inside the input smoke test, with no PIL and no extra JS.
    """
    if not a or not b:
        return None
    ca = a.split(",")
    cb = b.split(",")
    if len(ca) != len(cb) or not ca:
        return None
    diff = sum(1 for x, y in zip(ca, cb) if x != y)
    return diff / len(ca)


# Stop-Losing-To-OneShot todo #2 — criteria coverage helper.
# Stop-words drop generic prose ("the player should be able to move")
# and keep meaningful tokens ("rotate", "thrust", "shoot"). Genre-free:
# we never compile a list of subject matter; we just remove function
# words and short tokens. Matches a probe to a criterion when at least
# one meaningful word appears in BOTH (the probe's `name` or `expr` and
# the criterion line). Conservative: a missing match flags the line,
# which biases toward asking the model to add probes.
_COVERAGE_STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "from", "into", "onto", "for",
    "are", "was", "were", "have", "has", "had", "been", "being", "must",
    "should", "could", "would", "will", "shall", "may", "might", "can",
    "do", "does", "did", "done", "is", "be", "an", "at", "by", "on",
    "in", "of", "to", "or", "no", "if", "as", "it", "its", "after",
    "when", "while", "than", "then", "but", "all", "any", "some",
    "press", "click", "type", "key", "keys", "test", "edge", "stress",
    "basic", "case", "value", "values", "true", "false", "render",
    "renders", "rendered", "show", "shown", "user", "game",
})
# Underscores are word boundaries (so 'ship_rotates_left' contributes
# 'ship', 'rotates', 'left' to the bag of words rather than a single
# compound token that almost never matches a criterion line).
_COVERAGE_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def _coverage_words(text: str) -> set[str]:
    """Return lowercased meaningful tokens (>=4 chars, non-stop)."""
    if not text:
        return set()
    return {
        w for w in (m.group(0).lower() for m in _COVERAGE_WORD_RE.finditer(text))
        if len(w) >= 4 and w not in _COVERAGE_STOPWORDS
    }


# Words in <criteria>, <plan>, or the goal text that signal the game
# expects keyboard / pointer input. When ANY of these appear, an
# input_test FAIL should be a hard signal (controls not wired), NOT
# auto-rationalized as "DOM-driven". DK trace 20260514_104131 shipped
# a game whose <criteria> said "ArrowRight moves the player right" and
# "climb ladders with ArrowUp/ArrowDown" while keyboard input was
# silently broken; the harness still reported `ok=True` because the
# page had a clickable restart button, which triggered the
# "treating as DOM-driven" carve-out.
#
# Model-agnostic, game-agnostic: describes input MODALITY, not subject
# matter or genre. Calculators / todo lists / drawing apps avoid these
# words and keep the existing DOM-driven carve-out.
_GAME_CONTROL_KEYWORDS = frozenset({
    "arrow", "arrows",
    "arrowleft", "arrowright", "arrowup", "arrowdown",
    "wasd", "key", "keys", "keyboard", "keypress",
    "press", "pressed", "control", "controls", "controller",
    "joystick", "gamepad",
    "move", "moves", "movement", "moving",
    "walk", "walks", "walking",
    "run", "runs", "running",
    "climb", "climbs", "climbing",
    "jump", "jumps", "jumping",
    "shoot", "shoots", "shooting",
    "fire", "fires", "firing",
    "spacebar", "space",
    "dpad", "stick", "thumbstick",
})


def expects_game_controls(*texts: str) -> bool:
    """True when any of the supplied texts mention keyboard / pointer
    controls. Used by load_and_test to decide whether an input_test
    failure should be a hard signal or a soft DOM-driven warning.

    Tokenizes on word boundaries so substrings inside other words
    (e.g. "monkey" matching "key") don't false-positive. Case-
    insensitive.
    """
    pattern = re.compile(r"[a-zA-Z]+")
    for text in texts:
        if not text:
            continue
        for m in pattern.finditer(text):
            if m.group(0).lower() in _GAME_CONTROL_KEYWORDS:
                return True
    return False


# NARROW keyboard-only keyword set (chess-trace fix 2026-06-22). Distinct
# from `_GAME_CONTROL_KEYWORDS`, which includes pointer-compatible verbs
# like "move"/"walk" that a CLICK game ("click to move the piece") also
# uses. These tokens specifically name KEYBOARD input, so their ABSENCE —
# together with mouse/pointer listeners and no keyboard listeners — is
# strong evidence the game is click-primary and a synthetic keyboard
# input_responsive failure is a false positive. Genre-free: input
# modality, not subject matter.
_KEYBOARD_ONLY_KEYWORDS = frozenset({
    "arrow", "arrows",
    "arrowleft", "arrowright", "arrowup", "arrowdown",
    "wasd", "key", "keys", "keyboard", "keypress", "keydown",
    "spacebar", "space", "press", "pressed",
})


def expects_keyboard_controls(*texts: str) -> bool:
    """True when any supplied text names KEYBOARD input specifically.

    Narrower than `expects_game_controls` — used to decide whether a
    synthetic keyboard input_responsive failure on a click-primary game
    (mouse listeners present, no keyboard listeners) should gate `ok`.
    """
    pattern = re.compile(r"[a-zA-Z]+")
    for text in texts:
        if not text:
            continue
        for m in pattern.finditer(text):
            if m.group(0).lower() in _KEYBOARD_ONLY_KEYWORDS:
                return True
    return False


# KeyboardEvent.code tokens a game might bind. Matched STRICTLY (so prose
# "press F" does NOT false-press an unrelated key) — the system prompt and
# won-skeletons instruct models to write `event.code` tokens in <criteria>.
_KEY_CODE_RE = re.compile(
    r"\b(?:Key[A-Z]|Arrow(?:Up|Down|Left|Right)|Space|Digit[0-9]"
    r"|Numpad[0-9]|Enter|ShiftLeft|ShiftRight|Tab)\b"
)


# Max in-hold canvas-hash distance (fraction of 1024 cells) below which an
# action's rendered region is considered a STATIC held pose, not animated.
# 0.01 ≈ 10 cells: a genuine multi-frame swap or continuous motion moves far
# more across 250ms; <10 cells is the same pose with only AA/sub-pixel jitter.
_STATIC_POSE_MAX_INHOLD = 0.01

# An "action" (attack/ability) is a TRANSIENT: it changes the canvas while held
# and largely REVERTS after release. Keys whose effect persists — restart
# (wipes/redraws the whole screen), pause, walk (moves to a new lasting
# position) — are NOT actions and must not become the captured action frame.
# A key is transient when its after-release residual is below this fraction of
# its in-hold change. (Restart: residual ≈ in-hold change → excluded. Punch:
# residual ≈ 0 → eligible.)
_ACTION_TRANSIENT_MAX_RATIO = 0.5
# Bound the number of per-key action-frame screenshots held in memory.
_ACTION_FRAME_KEYCAP = 16


def _parse_action_keys(*texts: str) -> list[str]:
    """Extract the literal KeyboardEvent.code tokens declared in the supplied
    texts (typically the model's <criteria>), in first-seen order, deduped.

    The input smoke test presses movement keys by default; a fighting game's
    attack keys (KeyF/KeyG/KeyK/KeyL), an ability key (KeyZ), etc. are never
    pressed otherwise, so an attack animation is never triggered and never
    captured as an action frame. Pressing the keys the SPEC names is
    input-derived, not a genre key-table.
    """
    seen: list[str] = []
    for text in texts:
        if not text:
            continue
        for m in _KEY_CODE_RE.finditer(text):
            tok = m.group(0)
            if tok not in seen:
                seen.append(tok)
    return seen


# Phase 0 (Fieldrunners trace 20260626_102307): control verbs/nouns that mark
# a declared key as NON-COMBAT (UI / flow / build), so it must not feed the
# ACTION_DRAWN_NOT_SPRITED gate. Criteria like "Space starts a wave" named a
# menu/flow key, and the gate falsely diagnosed a "faked kick" on it. Genre-
# free: these describe control intent (start/pause/build), not subject matter.
_NON_COMBAT_KEY_CONTEXT_RE = re.compile(
    r"\b(?:start|starts|starting|begin|begins|spawn|spawns|launch|launches|"
    r"pause|pauses|resume|resumes|menu|restart|restarts|reset|resets|"
    r"select|selects|build|builds|place|places|placing|sell|sells|"
    r"deploy|deploys|upgrade|upgrades|toggle|toggles|confirm|cancel|"
    r"next\s+wave|wave|round|skip|skips|continue)\b",
    re.I,
)


def _non_combat_action_keys(criteria: str) -> set[str]:
    """Key codes whose nearby phrase in the criteria describes a non-combat
    control (start wave / pause / menu / build / place / sell). These must be
    excluded from the ACTION_DRAWN_NOT_SPRITED gate — pressing them is not an
    attack/ability that needs a sprite frame.

    Window-based: a key code counts as non-combat when a control verb/noun
    appears within ~32 chars after it (or ~16 before). Genre-free."""
    if not criteria:
        return set()
    out: set[str] = set()
    for m in _KEY_CODE_RE.finditer(criteria):
        window = criteria[max(0, m.start() - 16): m.end() + 32]
        if _NON_COMBAT_KEY_CONTEXT_RE.search(window):
            out.add(m.group(0))
    return out


# Flipper/paddle keys rotate canvas segments — not sprite-swap actions.
_ROTATION_MECHANIC_RE = re.compile(
    r"\b(?:flipper|flippers|paddle|paddles|bat|batting|rotate|rotating)\b",
    re.I,
)


def _rotation_mechanic_action_keys(criteria: str) -> set[str]:
    """Keys declared near flipper/paddle/rotate — code-drawn segments, not sprites."""
    if not criteria:
        return set()
    out: set[str] = set()
    for m in _KEY_CODE_RE.finditer(criteria):
        window = criteria[max(0, m.start() - 24): m.end() + 48]
        if _ROTATION_MECHANIC_RE.search(window):
            out.add(m.group(0))
    return out


def _criteria_declares_keyboard_player_movement(criteria: str) -> bool:
    """True when criteria say WASD/arrows move the player/ship (not flippers)."""
    if not criteria:
        return False
    low = criteria.lower()
    player_words = ("player", "ship", "hero", "character", "walk", "move")
    for m in _KEY_CODE_RE.finditer(criteria):
        code = m.group(0)
        if code not in _MOVEMENT_KEYS:
            continue
        window = low[max(0, m.start() - 30): m.end() + 60]
        if any(w in window for w in player_words):
            return True
    return False


def _recovery_is_physics_ball_only(
    recovery_leaves: list[str], criteria: str,
) -> bool:
    """Skip CONTROL-NOT-RECOVERED when only state.ball moved (physics body)."""
    if not recovery_leaves:
        return False
    if not all(str(l).startswith("ball.") for l in recovery_leaves):
        return False
    return not _criteria_declares_keyboard_player_movement(criteria)


def _slugify_criterion(text: str) -> str:
    """Compact identifier-safe slug for a criterion line, used as the
    suffix of a synthetic coverage-gap probe name. Keeps the slug short
    (≤32 chars) and limited to lowercase ASCII / digit / underscore so
    downstream regex matching (the agent's Phase B re-parse) can rely
    on it. Drops the leading category label `Basic:` / `Edge:` /
    `Stress:` if present so the slug describes the criterion itself.
    """
    s = (text or "").strip()
    # Strip leading category label like "Basic:", "Edge:", "Stress:".
    if ":" in s:
        head, rest = s.split(":", 1)
        if 2 <= len(head.strip()) <= 12 and head.strip().isalpha():
            s = rest.strip()
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
        if len(out) >= 32:
            break
    return ("".join(out).strip("_")) or "criterion"


def _criteria_coverage_gaps(criteria_text: str, probes: list[dict]) -> list[str]:
    """For each non-empty criterion line, return the line if no probe's
    name+expr shares any meaningful word with it. Generic, behavioral,
    no genre awareness. Only used when both criteria and probes are
    present — empty inputs short-circuit to no gaps reported."""
    if not criteria_text or not probes:
        return []
    probe_words: set[str] = set()
    for p in probes:
        probe_words |= _coverage_words(str(p.get("name") or ""))
        probe_words |= _coverage_words(str(p.get("expr") or ""))
    if not probe_words:
        return []
    uncovered: list[str] = []
    for raw in criteria_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        words = _coverage_words(line)
        if not words:
            continue
        overlap = words & probe_words
        # Existence-only probes ("ship_exists", "asteroids_exist")
        # typically share exactly one noun with a behavioral criterion
        # ("ship rotates with arrows; asteroids drift"). Require >=2
        # overlapping tokens for criteria with >=3 meaningful words so
        # an existence probe can't satisfy a multi-verb claim by accident.
        # Terse criteria (1-2 tokens) keep the lenient >=1 rule.
        threshold = 2 if len(words) >= 3 else 1
        if len(overlap) < threshold:
            uncovered.append(line[:140])
    return uncovered


# Fix round (fight trace 20260611_145321): a "60fps under stress" criterion
# became a synthetic coverage_gap probe that no honest probe can satisfy in
# a short load test — it blocked ok=True forever and distracted the model.
# Sustained-performance criteria stay advisory (criteria_uncovered) only.
_PERF_CRITERION_RE = re.compile(
    r"\b(fps|frame\s?rates?|frames\s+per\s+second|stress(?:\s|-)?test\w*|"
    r"stress|slow(?:s|ing)?\s?down|slowdowns?|lag\w*|jank\w*|performance)\b",
    re.IGNORECASE,
)


def _is_unverifiable_perf_criterion(line: str) -> bool:
    """True for criteria about sustained performance / frame rate — a short
    harness load cannot honestly verify them, so they must never gate."""
    return bool(_PERF_CRITERION_RE.search(line or ""))


def _apply_coverage_gap_gate(
    report: dict[str, Any],
    criteria: str,
    probes: list[dict],
    probe_results: list[dict],
) -> list[dict]:
    """Synthesize one failing `coverage_gap__<slug>` probe per <criteria>
    line that no model-authored probe references, AND the matching
    PROBE FAILED soft_warning so the final ok-recompute gates on it.

    Extracted from `LiveBrowser.load_and_test` (was inline) so the gating
    invariant is unit-testable without Chromium.

    [COVERAGE-GAP OK FIX 20260622 — battlezone trace, GLM-5.2] The bug this
    fixes: the synthetic probe was appended to report["probes"] (shown as
    FAIL) but NO soft_warning was added, so `ok = no errors and no
    soft_warnings` left ok=True — the report listed a FAIL yet shipped
    GREEN. Real model-authored failing probes already gate via the loop in
    load_and_test; synthetic ones now do too, via the same
    `_format_probe_failure_warning` channel (which `_failure_blames_code`
    correctly treats as a probe-authoring artifact, NOT a code defect).

    Returns the (possibly extended) probe_results. No-op when there are no
    gaps — clean games keep ok=True.
    """
    coverage_gaps = _criteria_coverage_gaps(criteria or "", probes or [])
    if not coverage_gaps:
        return probe_results
    report["criteria_uncovered"] = coverage_gaps
    # When every model-authored probe already passes, uncovered criteria
    # are advisory only — synthetic coverage_gap probes burned a fix iter on
    # street-fighter Round 1 (8/9 model probes green, Edge criterion gap).
    model_results = [
        p for p in probe_results
        if not p.get("synthetic")
        and not str(p.get("name") or "").startswith("coverage_gap__")
    ]
    all_model_pass = bool(model_results) and all(p.get("ok") for p in model_results)
    for gap in coverage_gaps:
        # Sustained-performance criteria are advisory only — no honest probe
        # can verify "60fps under stress" in a short load, so a synthetic
        # probe would block ok=True forever.
        if _is_unverifiable_perf_criterion(gap):
            continue
        if all_model_pass:
            report.setdefault("warnings", []).append(
                "ADVISORY (non-blocking) — criteria uncovered by probes: "
                + gap[:160]
            )
            continue
        slug = _slugify_criterion(gap)
        # [HARNESS NOTE] fence (2026-05-24) — without this, the 27B-class
        # coder mistakes synthetic coverage_gap probe text for file content
        # and emits a <patch> trying to DELETE it.
        synth = {
            "name": f"coverage_gap__{slug}",
            "expr": "false  /* synthetic - no model-authored probe for this criterion */",
            "ok": False,
            "err": (
                "[HARNESS NOTE — NOT FILE CONTENT, DO NOT <patch>]\n"
                f"This is a SYNTHETIC harness probe — it does NOT "
                f"exist anywhere in your .html file. It was added "
                f"automatically because your Phase A <criteria> "
                f"included this line but no model-authored probe "
                f"in your <probes> block references it:\n\n"
                f"  criterion: {gap[:200]}\n\n"
                f"Recovery: in your NEXT reply, do your normal fix "
                f"work (<patch> or <html_file>) AND include an "
                f"updated <probes>...</probes> block that adds "
                f"ONE entry whose name OR expr shares words with "
                f"the criterion text. The probes block is re-parsed "
                f"only when accompanied by code — don't emit "
                f"probes alone. Do NOT emit a <patch> targeting "
                f"this text; it isn't in any file.\n"
                "[/HARNESS NOTE]"
            ),
            "synthetic": True,
        }
        probe_results.append(synth)
        # [COVERAGE-GAP OK FIX] mirror the model-probe gate: a failing
        # synthetic probe must add a soft_warning so report["ok"] flips
        # False. Without this the gap showed as FAIL but still shipped ok.
        report["soft_warnings"].append(_format_probe_failure_warning(synth))
    # Refresh the report views that were built earlier in load_and_test.
    report["probes"] = probe_results
    report["probe_errors"] = [
        f"{p.get('name','?')}: {p.get('err','')[:160]}"
        for p in probe_results
        if not p.get("ok") and p.get("err")
    ]
    report["probe_eval_errors"] = [
        {
            "name": p.get("name", "probe"),
            "expr_preview": (p.get("expr") or "")[:120],
            "error_class": p.get("error_class") or "eval_error",
            "err": (p.get("err") or "")[:200],
        }
        for p in probe_results
        if not p.get("ok") and p.get("kind") == "eval_error"
    ]
    return probe_results


def _truncate(s: str, n: int) -> str:
    """Truncate long strings with a clear marker so the model knows it was cut."""
    if len(s) <= n:
        return s
    return s[:n] + f"...[+{len(s) - n} chars]"


_THREE_RUNTIME_MARKERS = (
    "three.min.js",
    "three.js",
    "three@",
    "new three.",
    "three.webglrenderer",
)


def _is_threejs_candidate_html(html_text: str) -> bool:
    """Cheap gate: strict outside-agent check only for likely three.js pages."""
    if not html_text:
        return False
    low = html_text.lower()
    return any(mark in low for mark in _THREE_RUNTIME_MARKERS)


def _classify_strict_file_failure(
    page_errors: list[str],
    console_errors: list[str],
    canvas_info: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Map strict-file failures to one concise class + one concise fix hint."""
    merged = [*(page_errors or []), *(console_errors or [])]
    low = "\n".join(merged).lower()
    first = (merged[0] if merged else "").strip()
    if any(tok in low for tok in ("securityerror", "cross-origin", "blocked by cors", "cors")):
        return (
            "cors_blocked",
            first or "Security/CORS restriction triggered in stock file:// runtime.",
            "Use file://-safe texture loading (for example inline/data URL textures), not harness-only security flags.",
        )
    if any(tok in low for tok in ("err_file_not_found", "failed to load resource", "not allowed to load local resource", "404")):
        return (
            "asset_path_missing",
            first or "One or more local asset paths failed in stock file:// runtime.",
            "Fix relative asset paths and ensure each referenced local file exists next to the HTML.",
        )
    if any(tok in low for tok in ("failed to load module script", "cannot use import statement outside a module", "three is not defined")):
        return (
            "script_load_failed",
            first or "Core script/module failed in stock file:// runtime.",
            "Use a script-loading path that works in normal browser file:// mode.",
        )
    if page_errors:
        return (
            "render_loop_missing",
            first or "Uncaught runtime error prevents stable render loop in stock file:// runtime.",
            "Fix the first uncaught exception; render/update loop must run cleanly before tuning visuals.",
        )
    if canvas_info and canvas_info.get("raf_ran") is False:
        return (
            "render_loop_missing",
            "requestAnimationFrame never fired in stock file:// runtime.",
            "Start RAF after scene setup and avoid early returns that block the loop.",
        )
    return "", "", ""


def _asset_load_failed(referenced_assets: bool, error_text: str) -> bool:
    """True when the HTML referenced generated assets AND the run logged a
    concrete load failure (404 / file-not-found) for an `_assets/` path.

    Genre-free helper for the C1 missing-asset gate: a referenced sprite PNG
    that the browser could not fetch is a hard broken-art failure (the game
    falls back to colored squares), distinct from a pose sprite that merely
    did not trigger during the test window. Returns False when no art was
    referenced or no asset-path load error appears in the error text.
    """
    if not referenced_assets:
        return False
    low = (error_text or "").lower()
    if "_assets/" not in low:
        return False
    return any(tok in low for tok in (
        "404", "failed to load resource", "err_file_not_found",
        "not allowed to load",
    ))


def _run_strict_file_runtime_check(path: Path, run_seconds: float = 1.2) -> dict[str, Any]:
    """Second-pass outside-agent check: stock Chromium file:// with no relaxed flags."""
    page_errors: list[str] = []
    console_errors: list[str] = []
    canvas_info: dict[str, Any] | None = None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=[])
            context = browser.new_context(viewport={
                "width": _DEFAULT_BROWSER_VIEWPORT[0],
                "height": _DEFAULT_BROWSER_VIEWPORT[1],
            })
            try:
                context.add_init_script(_INSTRUMENTATION_JS)
                page = context.new_page()

                def on_console(msg):
                    if msg.type == "error":
                        console_errors.append(_truncate(msg.text, _MAX_MSG_LEN))

                def on_pageerror(exc):
                    page_errors.append(_truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN))

                page.on("console", on_console)
                page.on("pageerror", on_pageerror)
                try:
                    page.goto(f"file://{path}", wait_until="load", timeout=8_000)
                except Exception as e:
                    return {
                        "checked": True,
                        "status": "fail",
                        "failure_type": "script_load_failed",
                        "summary": _truncate(str(e), _MAX_MSG_LEN),
                        "hints": [
                            "Ensure scripts/assets load under normal file:// browser settings (no special flags).",
                        ],
                    }
                time.sleep(max(0.8, min(run_seconds, 1.8)))
                try:
                    canvas_info = page.evaluate(_CANVAS_PROBE_JS)
                except Exception:
                    canvas_info = None
            finally:
                try:
                    context.close()
                except Exception:
                    pass
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as e:
        # Circuit breaker: harness infra issues should not block shipping.
        return {
            "checked": True,
            "status": "infra_error",
            "failure_type": "infra_error",
            "summary": _truncate(str(e), _MAX_MSG_LEN),
            "hints": [],
        }

    failure_type, summary, hint = _classify_strict_file_failure(
        page_errors, console_errors, canvas_info
    )
    if failure_type:
        return {
            "checked": True,
            "status": "fail",
            "failure_type": failure_type,
            "summary": _truncate(summary, _MAX_MSG_LEN),
            "hints": [hint][:1],
        }
    return {
        "checked": True,
        "status": "pass",
        "failure_type": "",
        "summary": "Runs in stock file:// Chromium without harness-only flags.",
        "hints": [],
    }


# JS injected via add_init_script BEFORE any of the page's own scripts run.
# Hooks requestAnimationFrame (so we know if the loop fires) and wraps
# addEventListener (so we can count input handlers). Shared by both
# test_html_file (sync, headless) and LiveBrowser (async, visible).
_INSTRUMENTATION_JS = """
window.__rafRan = false;
window.__listenerCount = { document: 0, window: 0, body: 0, other: 0 };
// Event-TYPE tally (chess-trace fix 2026-06-22): lets the harness tell a
// click/pointer-primary game (chess, board games, point-and-click) from a
// keyboard game, so a synthetic KEYBOARD input_responsive failure can be
// downgraded to a non-gating warning when the game never wired keyboard
// input. Genre-free — it keys on the INPUT MODALITY the page registered.
window.__listenerTypes = { key: 0, mouse: 0, pointer: 0, touch: 0 };

// 2D board/grid in live state — click-primary board games (chess, checkers)
// may ALSO register keyboard listeners as a cursor-navigation fallback
// (playbook keyboard-fallback-for-click-games); pointer + board still means
// keyboard smoke-test failure is not a broken control when criteria don't
// name keyboard input.
window.__hasPointerBoardState = (() => {
    const s = window.state || window.gameState || window.g || {};
    const is2d = (a) => Array.isArray(a) && a.length > 0 && Array.isArray(a[0]);
    return !!(is2d(s.board) || is2d(s.grid) || is2d(s.cells) || is2d(s.maze));
})();

const _origRAF = window.requestAnimationFrame;
window.requestAnimationFrame = function(cb) {
    window.__rafRan = true;
    return _origRAF.call(window, cb);
};

const _origAdd = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function(type, ...rest) {
    try {
        if (this === document) window.__listenerCount.document++;
        else if (this === window) window.__listenerCount.window++;
        else if (document.body && this === document.body) window.__listenerCount.body++;
        else window.__listenerCount.other++;
        const _t = (type || '').toLowerCase();
        if (_t.indexOf('key') === 0) window.__listenerTypes.key++;
        else if (_t.indexOf('pointer') === 0) window.__listenerTypes.pointer++;
        else if (_t.indexOf('touch') === 0) window.__listenerTypes.touch++;
        else if (_t.indexOf('mouse') === 0 || _t === 'click'
                 || _t === 'dblclick' || _t === 'contextmenu' || _t === 'wheel') {
            window.__listenerTypes.mouse++;
        }
    } catch (e) { /* ignore - some targets are exotic */ }
    return _origAdd.call(this, type, ...rest);
};

// Audio-events shim — records every Audio playback start so probes can
// assert "an SFX fired in response to <event>". We wrap both the plain
// HTMLAudioElement.play() path AND the WebAudio source.start() path
// because games use either. The record shape is intentionally flat
// — {t: epoch-ms, src: source identifier} — so probes can filter with
// trivial JS without parsing nested events.
window.__audioEvents = [];
const _origAudioPlay = HTMLAudioElement && HTMLAudioElement.prototype && HTMLAudioElement.prototype.play;
if (_origAudioPlay) {
    HTMLAudioElement.prototype.play = function(...rest) {
        try {
            window.__audioEvents.push({
                t: Date.now(),
                src: (this && this.src) ? this.src : "<inline>",
                kind: "html_audio",
            });
        } catch (e) { /* ignore */ }
        return _origAudioPlay.apply(this, rest);
    };
}
if (typeof AudioBufferSourceNode !== "undefined" && AudioBufferSourceNode.prototype) {
    const _origStart = AudioBufferSourceNode.prototype.start;
    if (_origStart) {
        AudioBufferSourceNode.prototype.start = function(...rest) {
            try {
                window.__audioEvents.push({
                    t: Date.now(),
                    src: "<webaudio_buffer>",
                    kind: "webaudio",
                });
            } catch (e) { /* ignore */ }
            return _origStart.apply(this, rest);
        };
    }
}
if (typeof OscillatorNode !== "undefined" && OscillatorNode.prototype) {
    const _origOscStart = OscillatorNode.prototype.start;
    if (_origOscStart) {
        OscillatorNode.prototype.start = function(...rest) {
            try {
                window.__audioEvents.push({
                    t: Date.now(),
                    src: "<oscillator>",
                    kind: "oscillator",
                });
            } catch (e) { /* ignore */ }
            return _origOscStart.apply(this, rest);
        };
    }
}
// drawImage shim — records every CanvasRenderingContext2D.drawImage call
// so the harness can detect "asset loaded into an Image but never drawn".
// The 2026-05-23 chess trace had the model populate `ASSETS[name] = new
// Image(); img.src = ...; await img.decode()` AND then draw chess pieces
// procedurally via ctx.fillText() Unicode glyphs — the existing
// sprite-alpha-and-distinctness check tests loader existence, not actual
// drawImage usage, so it passed. Recording one event per call (with a
// best-effort source identifier) lets the harness diff loaded names
// against drawn names without re-parsing the file. Mirrors the
// __audioEvents shape: {t, src, kind} so probes can filter cheaply.
window.__drawImageEvents = [];
if (typeof CanvasRenderingContext2D !== "undefined"
    && CanvasRenderingContext2D.prototype) {
    const _origDrawImage = CanvasRenderingContext2D.prototype.drawImage;
    if (_origDrawImage) {
        CanvasRenderingContext2D.prototype.drawImage = function(image, ...rest) {
            try {
                // Best-effort source identifier — full URL preferred,
                // currentSrc as fallback for HTMLImageElement, "<canvas>"
                // or "<bitmap>" for non-image draw sources.
                let src = "<unknown>";
                if (image) {
                    if (typeof image.src === "string" && image.src) {
                        src = image.src;
                    } else if (typeof image.currentSrc === "string" && image.currentSrc) {
                        src = image.currentSrc;
                    } else if (image instanceof HTMLCanvasElement) {
                        src = "<canvas>";
                    } else if (typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap) {
                        src = "<bitmap>";
                    } else if (typeof image.tagName === "string") {
                        src = "<" + image.tagName.toLowerCase() + ">";
                    }
                }
                // Cap the buffer so a long-running game with many sprite
                // calls per frame doesn't grow window.__drawImageEvents
                // unboundedly. The harness only needs to know WHICH
                // sources have been seen, not the count per source.
                if (window.__drawImageEvents.length < 4000) {
                    window.__drawImageEvents.push({
                        t: Date.now(),
                        src: src,
                    });
                }
            } catch (e) { /* ignore - keep draw call intact */ }
            return _origDrawImage.call(this, image, ...rest);
        };
    }
    // fillRect shim — records BIG rectangle draws (≥32x32 in BOTH
    // dimensions) so the harness can detect the procedural-regression
    // failure mode: model declared N sprites in <assets>, loaded them
    // into ASSETS, then drew entities as colored rectangles instead of
    // ctx.drawImage(ASSETS.<name>, ...). Mortal-kombat 2026-05-24 trace
    // showed P2 rendered as "a massive solid blue rectangle" — the
    // drawImage shim correctly flagged the missing draw, but the
    // existing detector had no signal for "and the model drew this
    // instead". Recording size lets us filter UI elements (HUD bars,
    // borders) which are typically thin in one dimension.
    window.__fillRectEvents = [];
    const _origFillRect = CanvasRenderingContext2D.prototype.fillRect;
    if (_origFillRect) {
        CanvasRenderingContext2D.prototype.fillRect = function(x, y, w, h) {
            try {
                const aw = Math.abs(+w || 0);
                const ah = Math.abs(+h || 0);
                // 32x32 is the entity-vs-UI threshold. HUD bars
                // (200x16), score backgrounds (300x30), borders
                // (Wx2) all stay below it on one axis. A sprite
                // placeholder is typically a square ≥ player size.
                // Cap the buffer at 4000 like drawImage's.
                if (aw >= 32 && ah >= 32
                    && window.__fillRectEvents.length < 4000) {
                    window.__fillRectEvents.push({
                        t: Date.now(),
                        w: aw,
                        h: ah,
                    });
                }
            } catch (e) { /* ignore - keep draw call intact */ }
            return _origFillRect.call(this, x, y, w, h);
        };
    }
    // stroke/line shim — records code-drawn line/path calls (lineTo, stroke,
    // strokeRect). A model that fakes a "kick" by scribbling a limb in code
    // (instead of swapping to the kick SPRITE) fires these while NOT adding a
    // new drawImage source. Counting them lets the harness catch
    // "lines acting like a kick" (ACTION_DRAWN_NOT_SPRITED). Count only; no
    // payload. Capped like the others.
    window.__strokeEvents = window.__strokeEvents || { n: 0 };
    for (const _m of ["stroke", "strokeRect", "lineTo", "arc", "fill"]) {
        const _orig = CanvasRenderingContext2D.prototype[_m];
        if (_orig) {
            CanvasRenderingContext2D.prototype[_m] = function(...a) {
                try { if (window.__strokeEvents.n < 1000000) window.__strokeEvents.n++; }
                catch (e) {}
                return _orig.apply(this, a);
            };
        }
    }
}
"""

# Downsampled canvas hash. We sample a 32x32 grid spread across the canvas
# and concatenate the RGBA bytes. Cheap (~1KB string) but catches per-frame
# motion anywhere on the playfield — the older 9-pixel fingerprint missed
# slowly-moving objects in the middle. Used by the input smoke test AND the
# frozen-canvas check.
_CANVAS_HASH_JS = """
() => {
    const c = document.querySelector('canvas');
    if (!c || c.width < 4 || c.height < 4) return null;
    const N = 32;
    const w = c.width, h = c.height;
    // 2D context: getImageData per sample.
    const ctx2d = c.getContext('2d', { willReadFrequently: true });
    if (ctx2d) {
        try {
            const out = [];
            for (let iy = 0; iy < N; iy++) {
                const y = ((iy + 0.5) * h / N) | 0;
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const d = ctx2d.getImageData(x, y, 1, 1).data;
                    out.push(((d[0] << 16) | (d[1] << 8) | d[2]).toString(36));
                }
            }
            return out.join(',');
        } catch (e) { /* fall through to WebGL */ }
    }
    // WebGL context: read full backbuffer then sample at the same grid.
    // gl.readPixels has to read the whole framebuffer once because the
    // back-buffer is invalidated after a single readPixels call on most
    // drivers. 32x32 samples = 1024 pixels, still tiny.
    const gl = c.getContext('webgl2', { preserveDrawingBuffer: true })
            || c.getContext('webgl', { preserveDrawingBuffer: true });
    if (gl) {
        try {
            const buf = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            const out = [];
            for (let iy = 0; iy < N; iy++) {
                // WebGL Y is flipped relative to canvas Y.
                const y = h - 1 - (((iy + 0.5) * h / N) | 0);
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const i = (y * w + x) * 4;
                    out.push(((buf[i] << 16) | (buf[i+1] << 8) | buf[i+2]).toString(36));
                }
            }
            return out.join(',');
        } catch (e) { return null; }
    }
    return null;
}
"""

# JS run AFTER the game has had a chance to animate. Returns the canvas info
# (size, RAF flag, blank-pixel heuristic). Shared between sync + async paths.
#
# Blank detection samples a 32x32 grid (1024 samples) instead of 9 corner
# pixels. The 9-corner version was producing FALSE POSITIVES on any game
# with a centered "Press Space to Start" / score-only-in-middle screen
# because the four corner samples + four edge midpoints + one center
# sample would all hit the uniform background and miss the centered
# content. With 1024 samples, even small centered text / a single
# rendered sprite produces colors.size >= 2 → not blank.
# Phase 1.5.1 — detect "state has the entity but canvas doesn't render
# it". General, no genre logic: scans top-level fields on window.state
# / window.gameState for objects with numeric .x/.y; tries both raw-
# pixel and inferred-tile coordinate interpretations; samples a 16×16
# patch around each candidate position; flags entities where >80% of
# the patch pixels are background-colored (low alpha OR close to the
# top-left corner color). Catches the failure shape from the 2026-05-22
# Pac-Man trace where pacman_exists passed on state but no Pac-Man
# was drawn. Returns None when nothing to check, else
# {checked: N, missing: [{name, x, y, bg_fraction, position_kind}]}.
_ENTITY_RENDERED_JS = """
(() => {
  const s = window.state || window.gameState;
  if (!s || typeof s !== 'object') return null;
  const c = document.querySelector('canvas');
  if (!c || !c.width || !c.height) return null;
  let ctx;
  try {
    ctx = c.getContext('2d', {willReadFrequently: true});
  } catch (e) { return null; }
  if (!ctx) return null;
  let bg;
  try {
    bg = ctx.getImageData(0, 0, 4, 4).data;
  } catch (e) { return null; }
  const bgR = bg[0], bgG = bg[1], bgB = bg[2];

  // Find candidate entities: top-level state fields whose value is an
  // object with numeric .x AND .y. Skip the state object itself.
  // ENTITY-FP SUPPRESSION (serial10 game 4): a movement/direction
  // vector like state.dir = {x:1,y:0} is NOT a drawable entity position.
  // The heuristic used to read it as a tile/pixel coordinate, find
  // background there, and emit a false ENTITY-NOT-RENDERED soft_warning
  // that burned 3 of 4 fix iters. Skip a field when (a) its name reads as
  // a direction/velocity vector, or (b) BOTH |x|<=1 and |y|<=1 (a unit /
  // sign vector can never be a real on-canvas entity position).
  const DIR_NAME_RE = /(^|_)(dir|vel|velocity|heading|facing|delta|accel|acceleration)($|_|[A-Z0-9])/i;
  const candidates = [];
  for (const k in s) {
    if (k.startsWith('_')) continue;
    let v;
    try { v = s[k]; } catch (e) { continue; }
    if (v && typeof v === 'object' && !Array.isArray(v)
        && typeof v.x === 'number' && typeof v.y === 'number'
        && isFinite(v.x) && isFinite(v.y)) {
      if (DIR_NAME_RE.test(k)) continue;  // direction/velocity vector, not a position
      if (Math.abs(v.x) <= 1 && Math.abs(v.y) <= 1) continue;  // unit/sign vector, not a position
      candidates.push({name: k, x: v.x, y: v.y});
    }
  }
  if (candidates.length === 0) return null;

  // Tile-size inference: many grid games store coords in tile units
  // (e.g. entity.x = 14 = column 14). We don't know the grid count
  // generically, so we try a few common arcade widths and pick the
  // interpretation with the lowest background fraction for each entity.
  const tileCandidates = [28, 32, 20, 16, 8];

  const missing = [];
  for (const ent of candidates) {
    let v;
    try { v = s[ent.name]; } catch (e) { v = null; }
    const ew = (v && typeof v.w === 'number' && v.w > 0) ? v.w
             : (v && typeof v.r === 'number' && v.r > 0) ? v.r * 2
             : (v && typeof v.width === 'number' && v.width > 0) ? v.width : 32;
    const eh = (v && typeof v.h === 'number' && v.h > 0) ? v.h
             : (v && typeof v.r === 'number' && v.r > 0) ? v.r * 2
             : (v && typeof v.height === 'number' && v.height > 0) ? v.height : 32;
    const positions = [{kind: 'pixel', px: ent.x, py: ent.y}];
    // When state exposes a 2D map/grid, coords are usually tile indices —
    // run_09 roguelike: entity at (37,21) on a 40×30 map was misread as
    // pixel (37,21) and falsely flagged ENTITY-NOT-RENDERED.
    let mapGrid = null;
    try { mapGrid = s.map || s.grid || s.level || s.tiles; } catch (e) { mapGrid = null; }
    if (Array.isArray(mapGrid) && mapGrid.length > 0) {
      const mapRows = mapGrid.length;
      let mapCols = 0;
      for (let ri = 0; ri < mapRows; ri++) {
        const row = mapGrid[ri];
        if (Array.isArray(row) && row.length > mapCols) mapCols = row.length;
      }
      if (mapCols > 1 && mapRows > 1
          && ent.x >= 0 && ent.y >= 0
          && ent.x < mapCols && ent.y < mapRows) {
        const tw = c.width / mapCols;
        const th = c.height / mapRows;
        positions.push({
          kind: 'map_tile',
          px: ent.x * tw + tw / 2,
          py: ent.y * th + th / 2,
        });
      }
    }
    for (const n of tileCandidates) {
      const t = c.width / n;
      positions.push({
        kind: `tile${n}`,
        px: ent.x * t + t / 2,
        py: ent.y * t + t / 2,
      });
    }
    let bestPos = null, bestBgFrac = 1.0;
    const halfW = Math.min(Math.max(8, Math.round(ew / 2)), 28);
    const halfH = Math.min(Math.max(8, Math.round(eh / 2)), 28);
    for (const p of positions) {
      if (p.px < 4 || p.px > c.width - 4
          || p.py < 4 || p.py > c.height - 4) continue;
      let patch;
      try {
        const bx = Math.max(0, Math.round(p.px) - halfW);
        const by = Math.max(0, Math.round(p.py) - halfH);
        const bw = Math.min(c.width - bx, halfW * 2);
        const bh = Math.min(c.height - by, halfH * 2);
        patch = ctx.getImageData(bx, by, bw, bh).data;
      } catch (e) { continue; }
      let bgCount = 0, total = 0;
      for (let i = 0; i < patch.length; i += 4) {
        total++;
        const r = patch[i], g = patch[i+1], b = patch[i+2], a = patch[i+3];
        const delta = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);
        if (a < 32 || delta < 30) bgCount++;
      }
      const bgFrac = total > 0 ? bgCount / total : 1.0;
      if (bestPos === null || bgFrac < bestBgFrac) {
        bestPos = p;
        bestBgFrac = bgFrac;
      }
    }
    if (bestPos !== null && bestBgFrac > 0.80) {
      missing.push({
        name: ent.name,
        x: ent.x, y: ent.y,
        bg_fraction: Math.round(bestBgFrac * 100) / 100,
        position_kind: bestPos.kind,
      });
    }
  }
  return {checked: candidates.length, missing: missing};
})();
"""


# Flatten the game's exposed state into a {dotted-path: number}
# map of numeric leaves, depth-capped and fanout-capped so giant
# entity arrays don't explode the snapshot. Returns null when
# nothing is exposed — callers then fall back to canvas hash only.
#
# NAME-MATCHING BUG FIX (2026-05-16): for the entire history of
# this code, the snapshot looked at `window.gameState` while the
# system prompt and all won-skeletons expose `window.state`. The
# net effect was that the gameplay verification path was BLIND
# to the actual state of agent-generated games and silently
# fell back to canvas-hash (which is degenerate for any auto-
# animating game). This is the single biggest reason "input
# smoke test passed" did not correlate with "controls work."
# Now we walk a small ordered list of plausible globals and
# take the first that's an object. `state` is the documented
# convention; the others are back-compat.
#
# Hoisted from `_input_smoke_test` to module level 2026-06-10 so the
# state-timeline sampler in `load_and_test` (capability-round item 4)
# reuses the exact same flattening.
_GAMESTATE_SNAPSHOT_JS = """
() => {
    const candidates = ['state', 'gameState', 'game', 'GAME', 'world'];
    let gs = null;
    for (const name of candidates) {
        const v = window[name];
        if (v != null && typeof v === 'object') { gs = v; break; }
    }
    if (gs == null) return null;
    const out = {};
    const visit = (obj, path, depth) => {
        if (depth > 4 || obj == null) return;
        if (typeof obj === 'number' && isFinite(obj)) { out[path] = obj; return; }
        if (typeof obj === 'boolean') { out[path] = obj ? 1 : 0; return; }
        // Short string leaves (state names like player.action='hit') are
        // captured as categorical values so the state timeline can flag an
        // entity stuck in the same non-default state across all samples
        // (fix-round item 1, the permanent stun-lock signal).
        if (typeof obj === 'string') {
            if (obj.length > 0 && obj.length <= 16) out[path] = obj;
            return;
        }
        if (typeof obj !== 'object') return;
        const ks = Array.isArray(obj)
            ? obj.slice(0, 32).map((_, i) => String(i))
            : Object.keys(obj).slice(0, 64);
        for (const k of ks) {
            try { visit(obj[k], path ? path + '.' + k : k, depth + 1); }
            catch (e) { /* getters etc */ }
        }
    };
    visit(gs, '', 0);
    // Also surface array lengths so "bullets fired" registers as a delta.
    const lenVisit = (obj, path, depth) => {
        if (depth > 4 || obj == null || typeof obj !== 'object') return;
        if (Array.isArray(obj)) { out[(path || '_root') + '.length'] = obj.length; return; }
        const ks = Object.keys(obj).slice(0, 64);
        for (const k of ks) {
            try { lenVisit(obj[k], path ? path + '.' + k : k, depth + 1); }
            catch (e) {}
        }
    };
    lenVisit(gs, '', 0);
    return out;
}
"""


_CANVAS_PROBE_JS = """
() => {
    const c = document.querySelector('canvas');
    if (!c) return null;
    const out = {
        width: c.width,
        height: c.height,
        raf_ran: !!window.__rafRan,
        blank: null,
        sampled_colors: null,
        ctx_kind: null,  // "2d" | "webgl2" | "webgl" | null
    };
    if (c.width < 4 || c.height < 4) return out;
    const N = 32;
    const w = c.width, h = c.height;
    // Pass 1: try 2D.
    const ctx2d = c.getContext('2d', { willReadFrequently: true });
    if (ctx2d) {
        out.ctx_kind = "2d";
        try {
            const colors = new Set();
            for (let iy = 0; iy < N; iy++) {
                const y = ((iy + 0.5) * h / N) | 0;
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const d = ctx2d.getImageData(x, y, 1, 1).data;
                    colors.add((d[0] << 16 | d[1] << 8 | d[2]) | 0);
                }
            }
            out.sampled_colors = colors.size;
            out.blank = colors.size <= 1;
            return out;
        } catch (e) { /* fall through */ }
    }
    // Pass 2: WebGL. Reads full backbuffer (preserveDrawingBuffer required
    // to survive past the implicit swap that follows a paint).
    const gl = c.getContext('webgl2', { preserveDrawingBuffer: true })
            || c.getContext('webgl', { preserveDrawingBuffer: true });
    if (gl) {
        out.ctx_kind = gl instanceof WebGL2RenderingContext ? "webgl2" : "webgl";
        try {
            const buf = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            const colors = new Set();
            for (let iy = 0; iy < N; iy++) {
                const y = h - 1 - (((iy + 0.5) * h / N) | 0);
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const i = (y * w + x) * 4;
                    colors.add((buf[i] << 16 | buf[i+1] << 8 | buf[i+2]) | 0);
                }
            }
            out.sampled_colors = colors.size;
            out.blank = colors.size <= 1;
        } catch (e) { /* keep blank: null */ }
    }
    return out;
}
"""


def _build_report(
    errors: list[str],
    warnings: list[str],
    logs: list[str],
    title: str,
    canvas_info: dict | None,
    listener_info: dict,
    body_text: str,
) -> dict[str, Any]:
    """Assemble the final report dict + heuristic soft warnings.

    Pulled out so test_html_file (sync) and LiveBrowser.load_and_test (async)
    produce IDENTICAL report shapes from identical inputs.
    """
    errors = errors[:_MAX_MSGS]
    warnings = warnings[:_MAX_MSGS]
    logs = logs[:_MAX_MSGS]

    soft_warnings: list[str] = []
    if canvas_info is not None:
        if canvas_info.get("raf_ran") is False:
            soft_warnings.append(
                "HEURISTIC: <canvas> exists but requestAnimationFrame never fired - "
                "your animation loop is not running."
            )
        # NOTE: BLANK canvas alone is NOT a fail signal. Many legitimate
        # apps start blank (drawing canvas, paint, charting tools where
        # data is fed via input). LiveBrowser appends a stronger
        # "blank AND input did nothing" warning later in load_and_test
        # once it knows the input-smoke result.
    if listener_info["total"] == 0:
        soft_warnings.append(
            "HEURISTIC: zero addEventListener calls detected - the game probably "
            "ignores all input."
        )

    return {
        "ok": len(errors) == 0 and len(soft_warnings) == 0,
        "errors": errors,
        "warnings": warnings,
        "soft_warnings": soft_warnings,
        "logs": logs,
        "title": title,
        "canvas": canvas_info,
        "input_listeners": listener_info,
        "body_chars": len(body_text),
        "body_sample": _truncate(body_text.strip(), _MAX_BODY_TEXT),
    }


# ---------------------------------------------------------------------------
# Micro-probes: fast pre-flight checks (no browser)
# ---------------------------------------------------------------------------
#
# Run BEFORE the Chromium round-trip to catch structurally-broken output
# (truncated streams, empty scripts, badly-unbalanced braces). Cheap; runs
# in <1 ms on a typical 5KB game file. OpenCoder's Educational-Instruct
# pattern: discard samples cheaply and often.
#
# Conservative on errors. Only flags as ERROR what is almost certainly
# unrecoverable without re-prompting; everything fuzzy goes to WARNINGS
# so the Chromium load still gets a chance.

# --- API allowlist (browser API hallucination guard) ----------------------
#
# Models hallucinate methods that don't exist (`ctx.drawCircle`,
# `ctx.fillCircle`, `audioCtx.playSound`, etc). These crash at runtime
# and we eventually catch them via Chromium's console.error — but slow.
# A small allowlist of real method names per known receiver convention
# lets us flag the hallucinated call at micro-probe time, before the
# Chromium round-trip.
#
# Conservative philosophy:
#   - Only check receivers whose variable name matches a STRICT convention
#     (`ctx` for canvas2d, `audioCtx` for AudioContext, `cvs` for the
#     element). If the user named their canvas context `myThing`, we
#     don't try — false negatives are fine, false positives are not.
#   - Output as a WARNING, not an ERROR. Chromium has the final word;
#     this is a fast preview so the model can preempt obvious mistakes.
#   - Allowlist values are method names commonly used in games. NOT
#     exhaustive — the goal is to catch hallucinations, not to gate
#     legitimate calls. We bias toward false negatives.

# Receivers we treat as canvas2d. Variable names from the games
# literature (Aider/Cline/Bolt prompts, Mozilla MDN, JS13k post-mortems).
_CANVAS2D_RECEIVERS = {"ctx", "c2d", "ctx2d", "context", "gfx", "g2", "g2d"}

# Receivers we treat as AudioContext. We use audioCtx, not bare `audio`,
# because `audio` is overloaded (HTMLAudioElement OR AudioContext); if
# the user wrote `audio`, we can't know which they meant. Strict here.
_AUDIOCTX_RECEIVERS = {"audioctx", "audiocontext", "actx", "audctx"}

# Receivers we treat as HTMLCanvasElement (the <canvas> DOM element,
# distinct from its 2D rendering context).
_CANVAS_ELT_RECEIVERS = {"cvs", "canvas", "canvasel", "canvaselt"}

# Receivers we treat as HTMLAudioElement (`new Audio(...)` instances).
# Distinct from AudioContext above. Common variable names for sound-
# effect handles in browser games. We're conservative: bare `audio` is
# overloaded with AudioContext patterns in some codebases, so we skip
# it for AudioContext and treat it as HTMLAudioElement only.
_AUDIO_ELT_RECEIVERS = {"audio", "snd", "sfx", "sound", "clip", "track"}

# Real CanvasRenderingContext2D methods (subset useful for games).
# Source: MDN. Hallucinations the model often emits (drawCircle,
# fillCircle, drawLine, line, point) are NOT in this set, so flagged.
_CANVAS2D_METHODS = frozenset({
    "arc", "arcto", "beginpath", "beziercurveto", "clearrect", "clip",
    "closepath", "createimagedata", "createlineargradient", "createpattern",
    "createradialgradient", "createconicgradient", "drawimage",
    "drawfocusifneeded", "ellipse", "fill", "fillrect", "filltext",
    "getcontextattributes", "getimagedata", "getlinedash", "gettransform",
    "ispointinpath", "ispointinstroke", "lineto", "measuretext", "moveto",
    "putimagedata", "quadraticcurveto", "rect", "resettransform", "restore",
    "rotate", "roundrect", "save", "scale", "setlinedash", "settransform",
    "stroke", "strokerect", "stroketext", "transform", "translate",
    # Properties accessed as `.foo` are stripped before we check, but
    # methods called as e.g. `.translate()` ARE checked.
})

_AUDIOCTX_METHODS = frozenset({
    "createoscillator", "creategain", "createmediastreamsource",
    "createmediaelementsource", "createanalyser", "createbiquadfilter",
    "createbuffer", "createbuffersource", "createchannelmerger",
    "createchannelsplitter", "createconstantsource", "createconvolver",
    "createdelay", "createdynamicscompressor", "createiirfilter",
    "createpanner", "createperiodicwave", "createscriptprocessor",
    "createstereopanner", "createwaveshaper", "decodeaudiodata",
    "resume", "suspend", "close", "getoutputtimestamp",
    # Modern AudioContext additions
    "audioworklet",
})

_CANVAS_ELT_METHODS = frozenset({
    "getcontext", "todataurl", "toblob", "capturestream",
    "transfercontroltoOffscreen", "transfercontroltooffscreen",
    "addeventlistener", "removeeventlistener", "dispatchevent",
    "focus", "blur", "click", "getboundingclientrect",
    "queryselector", "queryselectorall",
    # Generic Element methods worth keeping
    "appendchild", "removechild", "setattribute", "getattribute",
    "remove", "contains",
    # Pointer Lock / Fullscreen — first-person games use both. The
    # classic-doom 20260512_111015 trace flagged every single
    # `cvs.requestPointerLock()` call as a hallucination across two
    # extension sessions, and the model's mouse-look bug persisted
    # plausibly because the harness kept saying the real API name was
    # wrong. Pointer Lock is on Element since ~2020; canvases inherit
    # it. Fullscreen the same.
    "requestpointerlock", "exitpointerlock",
    "requestfullscreen", "exitfullscreen",
    "webkitrequestpointerlock", "webkitexitpointerlock",
    "webkitrequestfullscreen", "webkitexitfullscreen",
    "mozrequestpointerlock", "mozrequestfullscreen",
    # Element / Selector API methods used in routing patterns
    "matches", "closest", "getrootnode",
    # Web Animations + ScrollIntoView — both legit, both occasionally
    # used in games.
    "animate", "scrollintoview",
})

# Real HTMLAudioElement / HTMLMediaElement methods (subset useful for
# games). Hallucinations the model often emits — `playWithFade`,
# `start3d`, `loopOnce` — are NOT here, so flagged.
_AUDIO_ELT_METHODS = frozenset({
    "play", "pause", "load", "canplaytype", "fastseek",
    "addtexttrack", "captureStream".lower(), "capturestream",
    # Inherited Element methods used in audio-handling patterns.
    "addeventlistener", "removeeventlistener", "dispatchevent",
    "cloneNode".lower(), "clonenode",
    "setattribute", "getattribute", "remove",
})

# Map of (lowered receiver name) -> (allowlist set, friendly label).
_RECEIVER_TYPES: dict[str, tuple[frozenset[str], str]] = {}
for r in _CANVAS2D_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS2D_METHODS, "CanvasRenderingContext2D")
for r in _AUDIOCTX_RECEIVERS:
    _RECEIVER_TYPES[r] = (_AUDIOCTX_METHODS, "AudioContext")
for r in _CANVAS_ELT_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS_ELT_METHODS, "HTMLCanvasElement")
for r in _AUDIO_ELT_RECEIVERS:
    _RECEIVER_TYPES[r] = (_AUDIO_ELT_METHODS, "HTMLAudioElement")

# Match `<word>.<word>(` in JS source. The receiver word must be on the
# left of the dot, the method on the right, with a `(` immediately after
# (i.e. it's CALLED, not just read). Greedy-tolerant of whitespace
# between tokens.
_METHOD_CALL_RE = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)


def _check_api_allowlist(js: str) -> list[tuple[str, str, str]]:
    """Find unknown-method calls on known-receiver variable names.

    Returns a list of (receiver_var, method, type_label) tuples — one
    per distinct hallucination found. Comments and string literals are
    stripped first via _strip_js_noise. Caller decides severity.
    """
    stripped = _strip_js_noise(js)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for m in _METHOD_CALL_RE.finditer(stripped):
        recv, method = m.group(1), m.group(2)
        recv_key = recv.lower()
        if recv_key not in _RECEIVER_TYPES:
            continue
        allowlist, type_label = _RECEIVER_TYPES[recv_key]
        if method.lower() in allowlist:
            continue
        key = (recv, method)
        if key in seen:
            continue
        seen.add(key)
        out.append((recv, method, type_label))
    return out


# Used for cheap brace-balance checks. Strips line + block comments and
# string literals so a `for (;;)` inside a comment doesn't trip us.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'        # double-quoted
    r"|'(?:[^'\\]|\\.)*'"       # single-quoted
    r"|`(?:[^`\\]|\\.)*`",      # template literal (good-enough)
    re.DOTALL,
)
_SCRIPT_BLOCK_RE = re.compile(
    r"<script\b([^>]*)>(.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_js_noise(js: str) -> str:
    """Remove comments and string literals so brace-counting is reliable."""
    js = _BLOCK_COMMENT_RE.sub("", js)
    js = _LINE_COMMENT_RE.sub("", js)
    js = _STRING_LITERAL_RE.sub("''", js)
    return js


# Fix round (fight trace 20260611_145321): a rewrite dropped the seed
# skeleton's canvas sizing → 300x150 browser default → game drawn mostly
# off-canvas (black screen). The harness recorded width/height but never
# flagged it. These helpers make that failure a named, gating warning.
_CANVAS_TAG_RE = re.compile(r"<canvas\b[^>]*>", re.IGNORECASE)
_CANVAS_JS_SIZE_RE = re.compile(
    r"\.width\s*=|setAttribute\(\s*['\"]width['\"]", re.IGNORECASE
)


def _canvas_default_size_warning(
    canvas_info: dict | None, html: str | None
) -> str | None:
    """Return a gating CANVAS-DEFAULT-SIZE warning when the live canvas is
    exactly the untouched 300x150 browser default AND the source confirms
    no sizing was ever attempted (no width= attribute, no .width assignment).
    Genre-free: a real game canvas is never the untouched default."""
    if not isinstance(canvas_info, dict):
        return None
    try:
        w = int(canvas_info.get("width") or 0)
        h = int(canvas_info.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if (w, h) != (300, 150):
        return None
    src = html or ""
    tags = _CANVAS_TAG_RE.findall(src)
    if tags and any("width" in t.lower() for t in tags):
        return None  # markup sizes it — 300x150 must be deliberate
    if _CANVAS_JS_SIZE_RE.search(src):
        return None  # script sizes it (e.g. a fit()/DPR resize)
    return (
        "CANVAS-DEFAULT-SIZE: your <canvas> has no width/height — it is "
        "the 300x150 browser default, so most of the game is drawn "
        "off-canvas. Fix: add width/height attributes (e.g. <canvas "
        'width="800" height="500">) or set canvas.width/height in JS '
        "before first draw."
    )


# JS-source-in-body gate (FPS trace 20260611_213744 iters 5-6): a patch broke
# a script boundary and 2,517 chars of JavaScript rendered as visible page
# text. The report showed the raw body sample but never NAMED the failure,
# so the model stayed stuck on generic blank-canvas heuristics. Each pattern
# is a distinct JS-source signature; ≥2 distinct kinds = source code, not
# HUD strings ("Health: Ammo:") or story prose.
_JS_IN_BODY_SIGNATURES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\b(?:const|let|var)\s+\w+\s*="),
    re.compile(r"=>\s*\{"),
    re.compile(r"\bfor\s*\(\s*(?:let|var|const)\b"),
    re.compile(r"\bif\s*\([^)]+\)\s*\{"),
)
_JS_IN_BODY_MIN_CHARS = 200


def _js_source_in_body_warning(body_text: str) -> str | None:
    """Return a gating JS-SOURCE-IN-BODY warning when the page's visible
    body text is JavaScript source — a broken <script> boundary. None for
    short bodies and normal HUD/story text (needs >=2 distinct JS-source
    signature kinds over a meaningful length)."""
    text = body_text or ""
    if len(text) <= _JS_IN_BODY_MIN_CHARS:
        return None
    kinds = sum(1 for pat in _JS_IN_BODY_SIGNATURES if pat.search(text))
    if kinds < 2:
        return None
    return (
        f"JS-SOURCE-IN-BODY: ~{len(text)} chars of JavaScript render as "
        "visible page text — a script boundary is broken. Usual causes: a "
        "literal `</script>` inside a JS string (split it as "
        "`'</scr'+'ipt>'`), or a patch inserted code outside the <script> "
        "tag. Find the break point and restore the boundary; do not "
        "restyle the text."
    )


def _bracket_imbalance(js: str) -> dict[str, int]:
    """Return |open - close| count per bracket type after stripping strings
    and comments. Zero = balanced.
    """
    stripped = _strip_js_noise(js)
    return {
        "{}": stripped.count("{") - stripped.count("}"),
        "()": stripped.count("(") - stripped.count(")"),
        "[]": stripped.count("[") - stripped.count("]"),
    }


_ASSET_REF_RE = re.compile(
    # Match relative paths inside string literals (single or double quote
    # or backtick). Examples we catch:
    #   './foo_assets/wall.png', "./bar_sounds/shoot.ogg"
    # We deliberately skip CDN URLs (https://...) and data: URIs.
    # Path must end in a known media extension to avoid false positives
    # on arbitrary string literals.
    r"""['"`]\s*(\./[^'"`\s]+\.(?:png|jpe?g|gif|webp|svg|ogg|mp3|wav|m4a))\s*['"`]""",
    re.IGNORECASE,
)


def _check_asset_paths(
    html: str, out_path: "Path | None"
) -> list[str]:
    """Find relative asset paths in the HTML that reference files which
    don't exist on disk. For each missing path, suggest the closest
    real file using difflib.

    Returns a list of warning strings (never errors — Chromium has the
    final word; this is just a useful soft signal).

    Bench traces from `first-person-shooter-doom-game_20260511_160924`
    showed the 27B model corrupting paths in generated HTML: numeric
    tokenizer artifacts (`20260511` → `20260_511`), inconsistent
    underscores in slugs, file renames between attempts. Chromium
    surfaced these as generic `Failed to load resource: net::
    ERR_FILE_NOT_FOUND` lines with NO URL attached, so the model
    couldn't fix them on the next turn. This check tells the model
    EXACTLY which path missed and what the closest match is.
    """
    if out_path is None:
        return []
    try:
        from pathlib import Path
        out = Path(out_path)
        base = out.parent
        if not base.is_dir():
            return []
    except Exception:
        return []

    # Collect candidate files on disk under the session base dir
    # (assets/, sounds/, anywhere ~2 levels deep). Stored as relative
    # POSIX paths so suggestions are pasteable.
    candidates: list[str] = []
    try:
        for ext in ("png", "jpg", "jpeg", "gif", "webp", "svg",
                    "ogg", "mp3", "wav", "m4a"):
            for p in base.rglob(f"*.{ext}"):
                try:
                    rel = p.relative_to(base).as_posix()
                except Exception:
                    continue
                candidates.append("./" + rel)
    except Exception:
        return []

    if not candidates:
        return []  # nothing generated to compare against

    import difflib
    seen: set[str] = set()
    out_warnings: list[str] = []
    for m in _ASSET_REF_RE.finditer(html or ""):
        ref = m.group(1)
        if ref in seen:
            continue
        seen.add(ref)
        full = base / ref[2:]  # strip leading "./"
        try:
            if full.is_file():
                continue
        except Exception:
            continue
        # Missing — find closest match by basename, then by full rel path.
        suggestion: str | None = None
        cand_basenames = [c.rsplit("/", 1)[-1] for c in candidates]
        bn = ref.rsplit("/", 1)[-1]
        close = difflib.get_close_matches(bn, cand_basenames, n=1, cutoff=0.5)
        if close:
            # Look up the full candidate path that ends with this basename.
            for c in candidates:
                if c.endswith("/" + close[0]):
                    suggestion = c
                    break
        out_warnings.append(
            f"asset reference {ref!r} does not exist on disk"
            + (f"; did you mean {suggestion!r}?" if suggestion else
               f" and no close match found among {len(candidates)} "
               "generated files. Use one of the paths from the "
               "GENERATED ASSETS block in the user-turn message verbatim.")
        )
    return out_warnings


# Chess/board engines and asset loaders legitimately repeat these lines many
# times. Holochess run_05 iter 1: a complete 33 KB game was rejected because
# `} else {` appeared 14× AND the session `_assets` dir token appeared 48×
# in the ASSETS[] loader — both false positives that blocked materialize.
_BENIGN_SCRIPT_REPEAT_LINES = frozenset({
    "} else {",
    "} else if (",
    "} else if(",
    "break;",
    "continue;",
    "return;",
})
_BENIGN_REPEAT_LINE_RE = re.compile(
    r"^\}(?:else(?:\s+if\s*\([^)]*\))?|\s*catch(?:\s*\([^)]*\))?)\s*\{?\s*$",
    re.IGNORECASE,
)


def _is_benign_script_repeat_line(line: str) -> bool:
    """True when a repeated line is normal control-flow, not a degeneration."""
    s = (line or "").strip()
    if s in _BENIGN_SCRIPT_REPEAT_LINES:
        return True
    return bool(_BENIGN_REPEAT_LINE_RE.match(s))


def _is_benign_script_repeat_identifier(tok: str) -> bool:
    """True when a high-count identifier is an asset path token, not a loop."""
    low = str(tok or "").lower()
    if low.endswith(("_assets", "_sounds", "_videos")):
        return True
    if "assets" in low and low.count("_") >= 2:
        return True
    return False


def run_micro_probes(
    html: str, out_path: "Path | None" = None
) -> dict[str, Any]:
    """Pre-Chromium structural sanity check.

    Report shape:
      ok:        bool          - True if no errors (warnings allowed).
      errors:    list[str]     - structurally-broken; Chromium will fail.
      warnings:  list[str]     - suspicious but maybe ok; Chromium continues.
      stats:     dict          - small numeric snapshot of what we measured.

    The agent uses this between materialize and Chromium: an `ok=False`
    report skips the browser round-trip and feeds errors back to the
    model on the next turn.

    `out_path` is optional. When provided, asset-path checks scan for
    relative file references in the HTML and flag any that don't exist
    on disk — useful when the model corrupts generated asset paths.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "size_bytes": len(html or ""),
    }

    if not html or len(html) < 200:
        errors.append(
            f"file is essentially empty ({len(html or '')} bytes) — likely "
            "the patch left the document near-empty or the model produced "
            "no usable content."
        )
        return {"ok": False, "errors": errors, "warnings": warnings, "stats": stats}

    low = html.lower()

    # --- structural completeness ---------------------------------------
    if "<!doctype" not in low and "<html" not in low:
        errors.append(
            "no <!DOCTYPE> or <html> root tag — the file does not look "
            "like an HTML document."
        )
    if "<html" in low and "</html" not in low:
        errors.append(
            "<html> opened but never closed — likely the stream was "
            "truncated. Re-emit the rest of the file (or a full <html_file> "
            "rewrite if patches can't recover)."
        )
    if "<body" in low and "</body" not in low:
        errors.append(
            "<body> opened but never closed — truncation indicator. "
            "Close the </body> and </html> tags."
        )

    # --- script presence -----------------------------------------------
    scripts = _SCRIPT_BLOCK_RE.findall(html)
    has_inline_handlers = bool(re.search(r"\bon[a-z]+\s*=", html, re.IGNORECASE))
    n_inline = sum(1 for (_attrs, body) in scripts if body.strip())
    n_external = sum(1 for (attrs, body) in scripts if "src=" in attrs.lower())
    stats["scripts_inline"] = n_inline
    stats["scripts_external"] = n_external
    stats["inline_event_handlers"] = has_inline_handlers

    if n_inline == 0 and n_external == 0 and not has_inline_handlers:
        errors.append(
            "no <script> blocks (inline or external) and no inline event "
            "handlers — the file has no game logic. Add a <script> with "
            "the game implementation."
        )

    # --- unsized <canvas> (fix round) -----------------------------------
    # A <canvas> with no width attribute and no .width assignment anywhere
    # in script renders at the 300x150 browser default — the black-screen
    # failure from fight trace 20260611_145321. Warning only: the live
    # CANVAS-DEFAULT-SIZE check in load_and_test is authoritative.
    canvas_tags = _CANVAS_TAG_RE.findall(html)
    if canvas_tags and not any("width" in t.lower() for t in canvas_tags):
        if not _CANVAS_JS_SIZE_RE.search(html):
            warnings.append(
                "<canvas> has no width/height attribute and the script never "
                "assigns canvas.width — it will render at the 300x150 browser "
                'default. Size it (e.g. <canvas width="800" height="500"> or '
                "canvas.width=… in JS) before drawing."
            )

    # --- bracket balance per inline script -----------------------------
    # Per pi-mono's prescriptive-error pattern: tell the model EXACTLY
    # which kind is unbalanced and by how much.
    total_imbalance = {"{}": 0, "()": 0, "[]": 0}
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        imb = _bracket_imbalance(body)
        for k, v in imb.items():
            total_imbalance[k] += v
    stats["bracket_imbalance"] = total_imbalance

    for kind, delta in total_imbalance.items():
        # Heuristic strips can over-count by ~1 in pathological corner
        # cases (regex literals look like division, etc). Allow ±1 as
        # WARNING; ±2 as ERROR.
        if abs(delta) >= 2:
            sign = "extra opening" if delta > 0 else "extra closing"
            errors.append(
                f"unbalanced {kind} brackets in <script>: {sign} "
                f"{kind[0] if delta>0 else kind[1]} by {abs(delta)} "
                "(after stripping comments and string literals). "
                "Almost certainly a syntax error — close the missing "
                "brace before re-running."
            )
        elif delta != 0:
            warnings.append(
                f"possibly unbalanced {kind} in <script> by {delta} "
                "(could be a regex-literal false-positive — Chromium "
                "will confirm)."
            )

    # --- duplicate top-level declarations -----------------------------
    # Catches the "concatenated two drafts" failure mode where the model
    # writes a first draft, starts over partway through, and emits a
    # second draft below the first WITHOUT deleting it. Result: duplicate
    # `const NAME = …` / `function NAME(…)` declarations at the same
    # scope, which Chromium correctly rejects with "Identifier '<name>'
    # has already been declared." Observed in donkey-kong trace
    # 20260516_124628 iter 2 (duplicate `const ctx`, `const state`,
    # `function buildLevels`) — Chromium reported the issue, but only
    # after a 3-second browser load. This probe catches it pre-Chromium
    # and tells the model EXACTLY which name is duplicated so the next
    # turn can target the right delete.
    _DECL_RE = re.compile(
        r"^(?P<indent>[ \t]*)(?P<kind>const|let|function)\s+"
        r"(?P<name>[A-Za-z_$][\w$]*)\b"
    )
    dup_names: list[str] = []
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        stripped = _strip_js_noise(body)
        depth = 0
        # Names declared at depth ≤ 1. Depth 0 is the bare script body;
        # depth 1 is "inside the IIFE wrapper that most agent games use"
        # — `(() => { ... })()`. Nested function bodies live at depth ≥ 2
        # and are excluded because shadowing there is legal.
        seen: dict[str, int] = {}
        for raw_line in stripped.splitlines():
            line = raw_line.rstrip()
            # Track brace depth BEFORE matching this line so a declaration
            # on the same line as an opening `{` is still attributed to
            # the enclosing scope.
            opens = line.count("{")
            closes = line.count("}")
            if depth <= 1:
                m = _DECL_RE.match(line)
                if m:
                    name = m.group("name")
                    seen[name] = seen.get(name, 0) + 1
            depth += opens - closes
            if depth < 0:
                # Unbalanced; bail out — bracket-imbalance probe already
                # reported this. Counting from negative depth would
                # produce noise.
                break
        for name, count in seen.items():
            if count >= 2:
                dup_names.append(name)
    if dup_names:
        # De-dup the message across multiple scripts; keep order stable.
        seen_msg: set[str] = set()
        unique = [n for n in dup_names if not (n in seen_msg or seen_msg.add(n))]
        sample = ", ".join(f"`{n}`" for n in unique[:5])
        errors.append(
            f"duplicate top-level declaration(s) in <script>: {sample} "
            "declared 2+ times at the same scope — looks like two drafts "
            "got concatenated. Re-emit with one body; delete the older "
            "duplicate."
        )
        stats["duplicate_declarations"] = unique

    # --- API allowlist (hallucinated method calls) --------------------
    # Scan inline scripts for `<known-receiver>.<method>(` patterns where
    # the method is not on the canonical allowlist for that receiver
    # type. Reported as warnings, not errors — Chromium has the final
    # word and the allowlist is intentionally incomplete (better to miss
    # a hallucination than to flag a real method we forgot).
    api_warnings: list[str] = []
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        for recv, method, type_label in _check_api_allowlist(body):
            api_warnings.append(
                f"`{recv}.{method}(...)` called but '{method}' is NOT a "
                f"known method on {type_label}. If `{recv}` is a "
                f"{type_label}, this is a hallucination — pick the "
                "real method name (Chromium will throw a TypeError "
                "otherwise). If `{recv}` is your own object, this "
                "warning is a false positive; ignore."
            )
    if api_warnings:
        warnings.extend(api_warnings)
        stats["api_hallucinations"] = len(api_warnings)

    # --- repetition-collapse loop --------------------------------------
    # Mid-size models occasionally degenerate into token-repeat loops
    # (e.g. `ENEMY_HISS_CHANCE_PER_SEC_ACTUAL_ACTUAL_ACTUAL...`) or emit
    # the same line dozens of times before stalling. Catch it here so we
    # surface a specific actionable error instead of forcing the user to
    # decode a 200 KB stream-stall trace.
    rep_warnings: list[str] = []
    scripts_with_rep_signal = 0
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        script_rep = False
        # Same line repeated > 10× verbatim (ignoring blank lines).
        line_counts: dict[str, int] = {}
        for ln in body.splitlines():
            stripped = ln.strip()
            if len(stripped) < 4:
                continue
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
        for ln, n in line_counts.items():
            if n > 10 and not _is_benign_script_repeat_line(ln):
                rep_warnings.append(
                    f"line repeated {n}× verbatim in <script>: "
                    f"{ln[:80]!r}{'…' if len(ln) > 80 else ''}. "
                    "This is a token-repeat loop — emit a focused "
                    "<patch> instead of rewriting the whole file."
                )
                script_rep = True
                break  # one example per script body is enough
        # Single 4+-char identifier appearing > 30× in one script body.
        # Threshold is intentionally high so legit names like `ctx`/`x`/
        # `i` don't trip; the failure pattern is identifier copies like
        # `_ACTUAL_ACTUAL_ACTUAL_…`.
        if not script_rep:
            for tok, n in _Counter(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", body)).items():
                if (
                    n > 30
                    and "_" in tok
                    and tok.count("_") >= 2
                    and not _is_benign_script_repeat_identifier(tok)
                ):
                    rep_warnings.append(
                        f"identifier `{tok}` appears {n}× in one <script> "
                        "body — almost certainly a repeat-loop degeneration. "
                        "Restart the change with a small <patch>."
                    )
                    script_rep = True
                    break
        # Suffix-loop: a 5+-char substring repeated > 25× anywhere in
        # the body. Catches the `_ACTUAL_ACTUAL_ACTUAL_…` family where
        # each full identifier is unique (so the token counter above
        # doesn't fire) but the suffix repeats.
        if not script_rep and len(body) > 2000:
            for substr in ("_ACTUAL", "_FINAL", "_REAL", "_TRUE"):
                n = body.count(substr)
                if n > 25:
                    rep_warnings.append(
                        f"suffix `{substr}` appears {n}× in one "
                        "<script> body — token-repeat loop. Send a "
                        "focused <patch>, not a rewrite."
                    )
                    script_rep = True
                    break
        if script_rep:
            scripts_with_rep_signal += 1
    if rep_warnings:
        # Promote to errors only when 2+ script bodies agree (very likely real).
        if scripts_with_rep_signal >= 2:
            errors.extend(rep_warnings[:3])
        else:
            warnings.extend(rep_warnings[:3])
        stats["repetition_signals"] = len(rep_warnings)

    # --- elision sentinels ---------------------------------------------
    # Models occasionally slip "// ... rest of code unchanged ..." into
    # a patch even after we tell them not to. Catch it here so we don't
    # ship a half-implemented file. The regex tolerates dotted variants
    # ("// ...rest of", "// .. rest of seed code stays same") which the
    # plain-substring list missed — donkey-kong trace 20260516_124628
    # iter 2 shipped "// ...rest of seed code stays same..." past the
    # detector because the literal had a space between "//" and "rest"
    # while the model emitted "// ..." (no space) instead.
    elision_markers = [
        "// ... rest unchanged",
        "// ... rest of code",
        "// rest of",
        "// (existing code)",
        "/* existing code */",
        "<- leave original",
    ]
    matched_marker: str | None = None
    for m in elision_markers:
        if m.lower() in low:
            matched_marker = m
            break
    if matched_marker is None:
        m_re = re.search(
            r"//\s*\.{2,}\s*rest\b\s+(?:of|unchanged)",
            html,
            re.IGNORECASE,
        )
        if m_re:
            matched_marker = m_re.group(0)
    if matched_marker is not None:
        errors.append(
            f"elision marker found in source: {matched_marker!r} — the file is "
            "incomplete. Re-emit the patch with the EXACT lines, no "
            "shortcuts."
        )

    # --- asset path existence check ------------------------------------
    # Catches model-corrupted file paths before Chromium does (which
    # only reports ERR_FILE_NOT_FOUND without telling the model WHICH
    # file). Soft warnings only — never errors.
    if out_path is not None:
        path_warnings = _check_asset_paths(html, out_path)
        if path_warnings:
            warnings.extend(path_warnings)
            stats["missing_asset_paths"] = len(path_warnings)

        # Inverse check: assets generated on disk but NEVER referenced
        # in the HTML. Common silent failure — model declared 12 sprites
        # in <assets>, generated them all, then forgot to wire them in.
        # Probes pass, visual judge sees a blank canvas, iter "succeeds".
        # Soft warnings only; the model decides whether to wire or drop.
        unused_warnings = _check_unused_assets(html, out_path)
        if unused_warnings:
            warnings.extend(unused_warnings)
            stats["unused_assets"] = len(unused_warnings)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def _check_unused_assets(
    html: str, out_path: "Path | None"
) -> list[str]:
    """Find generated assets (PNG/OGG) on disk under the session dir
    that are NOT referenced by the HTML. Returns warning strings.

    Why this matters: a session can pay 30s+ to generate a dozen sprites
    and a half-dozen sound clips, then ship an HTML file that never
    references any of them. The visible game looks empty/silent; the
    harness probes still pass on the structural side. This check makes
    the omission explicit so the next iter's fix prompt is specific
    instead of "the game looks blank, fix it".
    """
    if out_path is None:
        return []
    try:
        from pathlib import Path
        out = Path(out_path)
        base = out.parent
        session_prefix = (out.stem or "").lower()
        if not base.is_dir():
            return []
    except Exception:
        return []
    # Only flag files under the session's generated dirs — we ignore
    # arbitrary art the user dropped into the workspace.
    sprite_dirs: list[Path] = []
    sound_dirs: list[Path] = []
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            n = child.name.lower()
            if (
                session_prefix
                and n.startswith(session_prefix + "_")
                and n.endswith("_assets")
            ):
                sprite_dirs.append(child)
            elif (
                session_prefix
                and n.startswith(session_prefix + "_")
                and n.endswith("_sounds")
            ):
                sound_dirs.append(child)
    except OSError:
        return []
    if not sprite_dirs and not sound_dirs:
        return []
    html_text = html or ""
    out_warnings: list[str] = []
    for kind, dirs in (("sprite", sprite_dirs), ("sound", sound_dirs)):
        for d in dirs:
            try:
                files = sorted(d.iterdir())
            except OSError:
                continue
            for f in files:
                if not f.is_file():
                    continue
                # Cheap presence test: either the basename or the full
                # relative path appears verbatim in the HTML. Avoids
                # false positives from arbitrary path-mangling.
                name = f.name
                if name in html_text:
                    continue
                try:
                    rel = f.relative_to(base).as_posix()
                except Exception:
                    rel = name
                if rel in html_text:
                    continue
                out_warnings.append(
                    f"{kind} {name!r} was generated to {rel!r} but is "
                    "NEVER referenced in the HTML. Either wire it in "
                    "(use the GENERATED ASSETS path verbatim) or drop "
                    "it from the next <assets>/<sounds> request to "
                    "save generation time."
                )
                if len(out_warnings) >= 8:
                    return out_warnings
    return out_warnings


def format_micro_probes_for_model(report: dict[str, Any]) -> str:
    """Compact text version of a micro-probe report for the user-turn
    feedback message. Mirrors `format_report_for_model`'s shape so the
    model sees a familiar structure.
    """
    lines = ["MICRO-PROBE PRE-FLIGHT (structural sanity, no browser yet):"]
    lines.append(f"OK: {report.get('ok', False)}")
    stats = report.get("stats") or {}
    if stats:
        bits = [f"size={stats.get('size_bytes', 0)}b"]
        if "scripts_inline" in stats:
            bits.append(
                f"scripts(inline/external)={stats['scripts_inline']}/"
                f"{stats.get('scripts_external', 0)}"
            )
        if "bracket_imbalance" in stats:
            imb = stats["bracket_imbalance"]
            non_zero = {k: v for k, v in imb.items() if v != 0}
            if non_zero:
                bits.append(f"bracket_imbalance={non_zero}")
        lines.append("Stats: " + ", ".join(bits))
    if report.get("errors"):
        lines.append("ERRORS (must fix):")
        for e in report["errors"]:
            lines.append(f"  - {e}")
    if report.get("warnings"):
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def pointclick_opening_book_applicable(
    goal: str = "",
    visual_recipe_id: str | None = None,
) -> bool:
    """True when the session is a point-and-click adventure, not e.g. voxel/FPS.

    Opening-book playtest `pointclick-puzzle-chain` is hard-only for these goals.
    Low Jaccard overlap can retrieve it on unrelated goals (voxel trace 20260622).
    """
    if visual_recipe_id == "canvas-point-and-click":
        return True
    gl = (goal or "").lower()
    if "point-and-click" in gl or "point and click" in gl or "pointclick" in gl:
        return True
    if visual_recipe_id and "point-and-click" in str(visual_recipe_id):
        return True
    # Adventure + inventory/hotspot/scene — same shape as open-domain P&C tests.
    if "adventure" in gl and any(
        w in gl for w in ("inventory", "hotspot", "hotspots", "scene", "scenes", "monkey")
    ):
        return True
    return False


def test_html_file(path: str | Path, run_seconds: float = 3.0) -> dict[str, Any]:
    """Run an HTML file in headless Chromium and return a small report dict.

    Report shape (always these keys, so the agent can rely on it):
      ok:          bool   - True if zero errors AND zero exceptions
      errors:      list[str]  - console.error lines + page errors
      warnings:    list[str]  - console.warn lines
      logs:        list[str]  - first few console.log lines (debug aid only)
      title:       str
      canvas:      dict | None  - {width, height, raf_ran: bool} if a <canvas> exists
      body_chars:  int    - length of body innerText (rough "is anything there?" check)
      body_sample: str    - first chars of body innerText
    """
    path = Path(path).resolve()
    file_url = f"file://{path}"

    # Buffers we fill from event handlers.
    errors: list[str] = []
    warnings: list[str] = []
    logs: list[str] = []

    with sync_playwright() as pw:
        # CORS-taint fix (1.1): file:// pages that drawImage from a
        # local PNG taint the canvas, breaking getImageData and any
        # probe that relies on it. These flags re-enable cross-origin
        # reads + give file:// pages full access to local files. We're
        # already running untrusted-by-design HTML inside Playwright;
        # the flags don't widen the threat model meaningfully.
        cors_flags = [
            "--allow-file-access-from-files",
            "--disable-web-security",
        ]
        browser = pw.chromium.launch(headless=True, args=cors_flags)
        # New context per run so localStorage / cookies don't leak between iterations.
        context = browser.new_context(viewport={
            "width": _DEFAULT_BROWSER_VIEWPORT[0],
            "height": _DEFAULT_BROWSER_VIEWPORT[1],
        })
        page = context.new_page()

        # --- console capture ---
        # Playwright fires "console" for log/info/warn/error and "pageerror" for
        # uncaught JS exceptions. We split them into our three buckets.
        def on_console(msg):
            text = _truncate(msg.text, _MAX_MSG_LEN)
            t = msg.type
            if t == "error":
                errors.append(text)
            elif t == "warning":
                warnings.append(text)
            else:
                logs.append(text)

        def on_pageerror(exc):
            # `exc` stringifies to the JS error message + stack head.
            errors.append(_truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN))

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)

        # --- pre-load instrumentation ---
        # add_init_script runs in EVERY frame BEFORE any of the page's own
        # scripts. The actual JS lives in _INSTRUMENTATION_JS at module level
        # so LiveBrowser (async) can use the SAME hooks.
        page.add_init_script(_INSTRUMENTATION_JS)

        # --- load ---
        try:
            page.goto(file_url, wait_until="load", timeout=10_000)
        except Exception as e:
            # Page failed to even load - return early with a clear error.
            browser.close()
            return {
                "ok": False,
                "errors": [f"PAGE FAILED TO LOAD: {e}"],
                "warnings": [],
                "logs": [],
                "title": "",
                "canvas": None,
                "body_chars": 0,
                "body_sample": "",
            }

        # NOTE: RAF hook used to be injected here via page.evaluate. It was
        # moved into add_init_script above so it runs BEFORE the game's own
        # scripts; otherwise the game grabs the original RAF before we wrap it.

        # Let the game animate for a few seconds.
        time.sleep(run_seconds)

        # --- collect post-run facts ---
        title = page.title() or ""
        try:
            body_text = page.evaluate("document.body ? document.body.innerText : ''") or ""
        except Exception:
            body_text = ""

        canvas_info = None
        try:
            # Heuristic blank-canvas detector + RAF flag readout. The JS lives
            # in _CANVAS_PROBE_JS at module level so LiveBrowser uses the same.
            canvas_info = page.evaluate(_CANVAS_PROBE_JS)
        except Exception:
            canvas_info = None

        # Pull the listener counts injected by add_init_script. Tells us whether
        # the game actually wired up input - a real game has at least 1 keyboard
        # or mouse listener somewhere on document/window.
        listener_info = {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}
        try:
            counts = page.evaluate("window.__listenerCount || null") or {}
            for k in ("document", "window", "body", "other"):
                listener_info[k] = int(counts.get(k, 0) or 0)
            listener_info["total"] = sum(listener_info[k] for k in ("document", "window", "body", "other"))
        except Exception:
            pass

        browser.close()

    report = _build_report(errors, warnings, logs, title, canvas_info, listener_info, body_text)
    # A1: synchronous path doesn't separate page vs console errors, so just
    # scan the union for stack frames into the file we just ran.
    report["path"] = str(path)
    report["crash_source_slices"] = extract_crash_source_slices(
        list(errors), file_filter=path,
    )
    return report


def score_test_report(report: dict[str, Any]) -> float:
    """Continuous quality score in [0, 100] for a test report.

    Used by best-of-N candidate selection (and `tune why` postmortem
    rendering) so partial-credit candidates win over completely-broken
    ones, instead of a binary pass/fail. Designed to be monotone in
    "how close the candidate is to passing", so picking the max gives a
    sensible candidate even when none pass.

    Scoring (max 100):
      * test passes outright           → 100
      * else, weighted demerits / bonuses on:
        - errors (real JS exceptions)  : -8 per, capped at -40
        - soft_warnings (heuristics)   : -5 per, capped at -20
        - frozen_canvas True           : -10
        - canvas blank True            : -10
        - listener_total > 0           : +5
        - raf_ran True                 : +5
        - input_test passes            : +10
    """
    if report is None:
        return 0.0
    if report.get("ok") is True:
        return 100.0
    s = 50.0
    n_err = len(report.get("errors") or [])
    s -= min(n_err, 5) * 8
    n_iss = len(report.get("soft_warnings") or [])
    s -= min(n_iss, 4) * 5
    if report.get("frozen_canvas") is True:
        s -= 10
    canv = report.get("canvas") or {}
    if canv.get("blank") is True:
        s -= 10
    li = report.get("input_listeners") or {}
    if int(li.get("total") or 0) > 0:
        s += 5
    if canv.get("raf_ran") is True:
        s += 5
    it = report.get("input_test") or {}
    if it.get("ran") and it.get("any_change") is True:
        s += 10
    # Model-proposed acceptance probes (when present): each contributes a
    # small bonus, so a candidate that passes its own acceptance criteria
    # outranks one that doesn't.
    probes = report.get("probes") or []
    if probes:
        n_pass = sum(1 for p in probes if p.get("ok"))
        s += min(15, n_pass * 3)
    # "Feels like a game" bonuses — differentiate "compiles + dark canvas"
    # from "actually runs". These are deliberately small (max +9) so they
    # don't overpower the structural signals above; their job is to break
    # ties between candidates that are otherwise equivalent on errors /
    # listeners / RAF.
    if not (report.get("console_errors") or []):
        s += 3                                  # genuinely silent console
    if report.get("frozen_canvas") is False:
        s += 3                                  # canvas is being repainted
    if it.get("ran") and it.get("any_change") is True:
        s += 3                                  # input_test caused real motion
    return max(0.0, min(100.0, s))


# A1: when a JS error names a file:LINE:COL, splice the actual source
# region into the report so a 27B doesn't have to reverse-line-count
# a 700-line file from a raw stack trace. Stops "let me think about
# which .y access this could be" deliberation chains cold.
_STACK_FRAME_RE = re.compile(r"file://(?P<path>[^\s:)]+):(?P<line>\d+):(?P<col>\d+)")


def extract_crash_source_slices(
    error_strs: list[str],
    *,
    file_filter: str | Path | None = None,
    radius: int = 3,
    max_slices: int = 2,
) -> list[dict[str, Any]]:
    """Parse `file://...:LINE:COL` frames out of each error string and read the
    surrounding source lines from disk. Returns a list of dicts with `path`,
    `line`, `col`, `snippet` (multi-line string with `>` on the offending line).
    Dedups by (path, line) so a 5-deep stack trace doesn't produce 5 copies of
    the same neighborhood. Cap at `max_slices` so a noisy multi-error report
    doesn't blow up the prompt.
    """
    filter_resolved: str | None = None
    if file_filter is not None:
        try:
            filter_resolved = str(Path(file_filter).resolve())
        except Exception:
            filter_resolved = str(file_filter)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for err in error_strs:
        if not err:
            continue
        for m in _STACK_FRAME_RE.finditer(err):
            path_str = m.group("path")
            try:
                line_no = int(m.group("line"))
                col_no = int(m.group("col"))
            except ValueError:
                continue
            if filter_resolved:
                try:
                    cand_resolved = str(Path(path_str).resolve())
                except Exception:
                    cand_resolved = path_str
                if cand_resolved != filter_resolved:
                    continue
            key = (path_str, line_no)
            if key in seen:
                continue
            try:
                src = Path(path_str).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            src_lines = src.splitlines()
            if not src_lines or line_no < 1 or line_no > len(src_lines):
                continue
            lo = max(1, line_no - radius)
            hi = min(len(src_lines), line_no + radius)
            width = len(str(hi))
            rendered = []
            for i in range(lo, hi + 1):
                arrow = ">" if i == line_no else " "
                rendered.append(f"  {arrow} {i:>{width}}: {src_lines[i - 1]}")
            seen.add(key)
            out.append({
                "path": path_str,
                "line": line_no,
                "col": col_no,
                "snippet": "\n".join(rendered),
            })
            if len(out) >= max_slices:
                return out
    return out


def summarize_state_timeline(
    samples: list,
    *,
    window_seconds: float | None = None,
    max_lines: int = 4,
) -> str:
    """Dynamics digest over flattened game-state samples (item 4).

    Pure function. Classifies every numeric leaf present in all samples as
    constant / monotonic / changing, and flags two suspicious patterns:
    a frame/time counter that never increases (stalled loop) and an
    entity position frozen while a sibling's moves. Capped at `max_lines`
    rendered lines; returns "" when fewer than 3 samples exposed state.
    """
    dict_samples = [s for s in samples if isinstance(s, dict)]
    if len(dict_samples) < 3:
        return ""
    eps = 1e-3
    keys = set(dict_samples[0])
    for s in dict_samples[1:]:
        keys &= set(s)
    series: dict[str, list[float]] = {}
    constant: list[str] = []
    monotonic: list[str] = []
    changing: list[str] = []
    for k in sorted(keys):
        vals = [s[k] for s in dict_samples]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        series[k] = [float(v) for v in vals]
        deltas = [series[k][i + 1] - series[k][i] for i in range(len(vals) - 1)]
        if all(abs(d) <= eps for d in deltas):
            constant.append(k)
        elif all(d >= -eps for d in deltas) or all(d <= eps for d in deltas):
            monotonic.append(k)
        else:
            changing.append(k)
    if not series:
        return ""
    span = f" over {window_seconds:.1f}s" if window_seconds else ""
    lines = [
        f"State timeline ({len(dict_samples)} samples{span}, "
        f"{len(series)} numeric leaves): {len(constant)} constant, "
        f"{len(monotonic)} monotonic, {len(changing)} changing"
    ]
    moving = monotonic + changing
    if moving:
        ex = ", ".join(
            f"{k} ({series[k][0]:.6g}\u2192{series[k][-1]:.6g})"
            for k in moving[:3]
        )
        lines.append(f"moving: {ex}")
    # Suspicious 1: a frame/time-ish counter that never increases.
    stalled = [
        k for k in constant
        if k.split(".")[-1].lower()
        in ("frame", "frames", "tick", "ticks", "time", "t", "elapsed")
    ]
    if stalled:
        lines.append(
            f"SUSPICIOUS: {stalled[0]} not increasing across samples — "
            "the update loop may be stalled"
        )
    # Suspicious 3 (fix-round item 1): a state-machine string leaf (action/
    # state/mode/...) stuck on the same non-default value across every
    # sample — names the permanent stun-lock directly ("player.action stuck
    # at 'hit' for all 6 samples"). Leaf-name + default filters keep this
    # mechanism-level and quiet (names/colors/titles never match).
    if len(lines) < max_lines:
        _STATEISH_LEAVES = {
            "action", "state", "mode", "phase", "status",
            "anim", "animation", "pose",
        }
        _DEFAULTISH_VALUES = {
            "idle", "", "none", "default", "normal",
            "playing", "play", "running", "run",
        }
        for k in sorted(keys):
            leaf = k.rpartition(".")[2].lower()
            if leaf not in _STATEISH_LEAVES:
                continue
            vals = [s[k] for s in dict_samples]
            if not all(isinstance(v, str) for v in vals):
                continue
            if len(set(vals)) == 1 and vals[0].lower() not in _DEFAULTISH_VALUES:
                lines.append(
                    f"SUSPICIOUS: {k} stuck at '{vals[0]}' for all "
                    f"{len(dict_samples)} samples — does that state ever expire?"
                )
                break
    # Suspicious 2: one entity's position moves while a sibling's never does.
    if len(lines) < max_lines:
        moving_set = set(moving)
        by_leaf: dict[str, list[tuple[str, str]]] = {}
        for k in series:
            head, dot, leaf = k.rpartition(".")
            if dot and leaf in ("x", "y"):
                by_leaf.setdefault(leaf, []).append((head, k))
        for leaf, entries in sorted(by_leaf.items()):
            movers = [h for h, kk in entries if kk in moving_set]
            frozen_ents = [h for h, kk in entries if kk not in moving_set]
            if movers and frozen_ents:
                lines.append(
                    f"SUSPICIOUS: {frozen_ents[0]}.{leaf} constant across all "
                    f"samples while {movers[0]}.{leaf} changes — is "
                    f"{frozen_ents[0]} ever updated?"
                )
                break
    return "\n".join(lines[:max_lines])


def format_report_for_model(report: dict[str, Any]) -> str:
    """Turn the report dict into the SHORT plain-text block we feed to the model.

    Keeping this terse is intentional - per the project's debug-minimum rule we
    only send what the model needs to fix the next issue.
    """
    lines = []
    lines.append(f"OK: {report['ok']}")
    lines.append(f"Title: {report['title']!r}")
    if report["canvas"] is not None:
        c = report["canvas"]
        blank_str = "unknown" if c.get("blank") is None else str(c["blank"])
        # Flag the untouched browser default inline so the model sees the
        # real size every iteration (fix round, fight trace 20260611_145321).
        default_tag = (
            " (300x150 is the BROWSER DEFAULT — canvas was never sized!)"
            if (c.get("width"), c.get("height")) == (300, 150) else ""
        )
        lines.append(
            f"Canvas: {c['width']}x{c['height']}{default_tag}, "
            f"RAF ran: {c['raf_ran']}, blank: {blank_str}"
        )
    else:
        lines.append("Canvas: none")
    li = report.get("input_listeners", {})
    lines.append(
        f"Input listeners: total={li.get('total', 0)} "
        f"(doc={li.get('document', 0)}, win={li.get('window', 0)}, "
        f"body={li.get('body', 0)}, other={li.get('other', 0)})"
    )
    # Auto-input smoke test. With the gameplay-state-global fix
    # (2026-05-16), the harness now samples window.state (the
    # documented convention) and names exactly which fields moved on
    # which key. PASS reads like "ArrowRight→[player.x, player.facing];
    # Space→[bullets.length]" — names the wiring path that works. FAIL
    # distinguishes "state exposed but no field moved" (data-flow bug
    # downstream of the listener) from "no state global at all" (the
    # game never exposed window.state — your probes can't run).
    it = report.get("input_test") or {}
    if it.get("ran"):
        summary = it.get("summary") or ""
        if it.get("any_change"):
            lines.append(f"Input test: PASS — {summary}")
        else:
            lines.append(f"Input test: FAIL — {summary}")
    # New: did the canvas freeze (drawing same frame) between two samples?
    fz = report.get("frozen_canvas")
    if fz is True:
        lines.append("Frozen canvas: YES (same pixels at t=half and t=full)")
    elif fz is False:
        lines.append("Frozen canvas: no (pixels changed during run)")
    # Item 4: runtime dynamics digest (what actually happened over time).
    tl = report.get("state_timeline")
    if tl:
        lines.append(tl)
    lines.append(f"Body text length: {report['body_chars']} chars")
    if report["body_sample"]:
        lines.append(f"Body sample: {report['body_sample']!r}")
    strict = report.get("strict_file_runtime") or {}
    if strict.get("checked"):
        status = strict.get("status")
        if status == "pass":
            lines.append("Strict file:// runtime: pass")
        elif status == "fail":
            lines.append(
                "Strict file:// runtime: FAIL "
                f"[{strict.get('failure_type') or 'unknown'}] — "
                f"{strict.get('summary') or 'failed'}"
            )
            hints = strict.get("hints") or []
            if hints:
                lines.append(f"Strict fix hint: {hints[0]}")
        elif status == "infra_error":
            lines.append("Strict file:// runtime: unavailable (harness issue)")
    scoped = report.get("scoped_check") or {}
    if scoped.get("required"):
        keys = ", ".join(scoped.get("keywords") or [])
        if scoped.get("pass"):
            lines.append(
                "Scoped check: PASS"
                + (f" ({keys})" if keys else "")
            )
        else:
            lines.append(
                "Scoped check: FAIL"
                + (f" ({keys})" if keys else "")
            )
    if report["errors"]:
        # 2.4: split rendering when the harness exposed kinds. Page
        # errors (uncaught exceptions) are usually game bugs the model
        # can fix; console.error lines are often informational logs.
        # Probe errors are reported in their own probes section below;
        # we exclude them here so the model doesn't double-count.
        page_errs = report.get("page_errors") or []
        cons_errs = report.get("console_errors") or []
        if page_errs:
            lines.append("PAGE ERRORS (uncaught exceptions — must fix):")
            for e in page_errs:
                lines.append(f"  - {e}")
        if cons_errs:
            lines.append("CONSOLE ERRORS:")
            for e in cons_errs:
                lines.append(f"  - {e}")
        if not (page_errs or cons_errs):
            # Older path / sync test_html_file: fall back to the union.
            lines.append("ERRORS (must fix):")
            for e in report["errors"]:
                lines.append(f"  - {e}")
        # A1: prepend source context. If the test runner already attached
        # `crash_source_slices`, use those; otherwise compute on the fly so
        # callers that bypass LiveBrowser (sync test path, unit tests) still
        # benefit.
        slices = report.get("crash_source_slices")
        if slices is None:
            slices = extract_crash_source_slices(
                list(page_errs) + list(cons_errs),
                file_filter=report.get("path"),
            )
        for sl in slices or []:
            lines.append(f"SOURCE NEAR ERROR ({Path(sl['path']).name}:{sl['line']}):")
            lines.append(sl["snippet"])
    if report.get("soft_warnings"):
        # Heuristic broken-but-no-exception findings. Listed as ISSUES so the
        # model treats them with the same urgency as real errors.
        lines.append("ISSUES (must fix):")
        for s in report["soft_warnings"]:
            lines.append(f"  - {s}")
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report.get("probes"):
        n_pass = sum(1 for p in report["probes"] if p.get("ok"))
        n = len(report["probes"])
        lines.append(f"Acceptance probes: {n_pass}/{n} pass")
        for p in report["probes"]:
            tag = "ok " if p.get("ok") else "FAIL"
            err = f"  ({p['err']})" if p.get("err") else ""
            lines.append(f"  {tag} {p.get('name','probe')}: {p.get('expr','')[:80]}{err}")
    if report.get("opening_book_checks"):
        checks = report["opening_book_checks"]
        n_pass = sum(1 for c in checks if c.get("ok"))
        lines.append(f"Opening-book checks: {n_pass}/{len(checks)} pass")
        for c in checks[:6]:
            tag = "ok " if c.get("ok") else "FAIL"
            err = f"  ({c.get('err','')})" if c.get("err") else ""
            lines.append(f"  {tag} {c.get('id','recipe')}: {c.get('type','')}{err}")
    if report["logs"] and not report["errors"]:
        # Only show logs when there are no errors - otherwise the model focuses
        # on the wrong thing.
        lines.append("Console logs (info only):")
        for l in report["logs"][:4]:
            lines.append(f"  - {l}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LiveBrowser: VISIBLE, persistent browser used by the TUI.
#
# Different from test_html_file in three ways:
#   1. headless=False - you actually SEE the game in a real Chromium window.
#   2. The browser stays open across iterations (the TUI keeps a single
#      LiveBrowser instance for its whole session). Reload between iterations
#      is fast and visually shows the game updating.
#   3. Async API (so it cooperates with Textual's asyncio event loop instead
#      of blocking the TUI). Returns the SAME report shape as test_html_file.
#
# We use playwright.async_api here. Mixing sync_playwright and async_playwright
# in the same process is fine as long as you don't use both AT THE SAME TIME -
# the TUI uses async only, the CLI uses sync only.
# ---------------------------------------------------------------------------

# Imported lazily inside __init__ so people running the CLI never pay the
# async-playwright import cost (and so a missing async install doesn't break
# the headless path).

# Playwright viewport + visible Chromium window size. Old default 800×600 clipped
# HUD-heavy games (chess with tray + status) in the test browser while the same
# file:// looked fine in a full browser window. Override: BROWSER_VIEWPORT=WxH
# (e.g. 1280x800 or 1920x1080).
_DEFAULT_BROWSER_VIEWPORT: tuple[int, int] = (1280, 800)


def _resolve_browser_viewport() -> tuple[int, int]:
    """Return (width, height) from BROWSER_VIEWPORT env or module default."""
    raw = (os.environ.get("BROWSER_VIEWPORT") or "").strip()
    if not raw:
        return _DEFAULT_BROWSER_VIEWPORT
    sep = "x" if "x" in raw.lower() else ","
    parts = raw.lower().split(sep, 1)
    if len(parts) != 2:
        return _DEFAULT_BROWSER_VIEWPORT
    try:
        w, h = int(parts[0].strip()), int(parts[1].strip())
        if w >= 320 and h >= 240:
            return (w, h)
    except ValueError:
        pass
    return _DEFAULT_BROWSER_VIEWPORT


class LiveBrowser:
    """Persistent visible Chromium for the TUI. Async, single page, reusable.

    Usage:
        lb = LiveBrowser()
        await lb.start()
        report = await lb.load_and_test("games/game.html")
        ...
        await lb.close()
    """

    def __init__(
        self,
        viewport: tuple[int, int] | None = None,
        run_seconds: float = 3.0,
        headless: bool = False,
    ):
        self._viewport = viewport if viewport is not None else _resolve_browser_viewport()
        self._run_seconds = run_seconds
        self._headless = headless
        # Buffers reset on every load_and_test call.
        # 2.4: split error sources so format_report_for_model can show
        # them differently. _errors is kept as the combined view for
        # backward compat (downstream code reads report["errors"]).
        self._errors: list[str] = []
        self._console_errors: list[str] = []
        self._page_errors: list[str] = []
        self._warnings: list[str] = []
        self._logs: list[str] = []
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        # Fix-round item 2: cross-iteration persistence for the UNDRAWN
        # sprite-audit gate. First occurrence gates (forces one fix turn);
        # a persisting occurrence with green probes + no errors demotes to
        # the non-gating warnings channel. Reset when the finding clears.
        self._undrawn_seen_before = False
        # Trace 20260612_171752: same persistence-downgrade for the
        # ACTION_DRAWN_NOT_SPRITED gate — it held ok=False from iter 4
        # through every continuation turn on a 7/7-probes build, blocked
        # <done/>, and left best_exists=False. Behavioral probes gate;
        # cosmetics inform.
        self._action_not_sprited_seen_before = False

    async def start(self) -> None:
        """Launch the browser. Call once before load_and_test."""
        # Lazy import: keeps the sync CLI lightweight.
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        # headless=False is the whole point - the user wants to SEE the game.
        # --window-position keeps the window from landing on top of the terminal
        # by default (X/Wayland may ignore this; tile-WM users will arrange it
        # themselves anyway).
        # headless=False is the normal interactive case (TUI lets the user
        # SEE the game). headless=True is for the test driver.
        # CORS-taint fix (1.1): see comment in test_html_file. Without
        # these flags, drawImage(<file:// PNG>) taints the canvas and
        # any getImageData probe (including the harness's frozen check
        # and the model's non_blank example) throws SecurityError. The
        # cure used to be the model writing crossOrigin="anonymous" +
        # try/catch — error-prone and frequently broke working games.
        # Set the flags at launch and the issue goes away entirely.
        launch_args: list[str] = [
            "--allow-file-access-from-files",
            "--disable-web-security",
        ]
        if not self._headless:
            launch_args.extend([
                f"--window-position=850,50",
                f"--window-size={self._viewport[0]},{self._viewport[1]}",
            ])
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=launch_args,
        )
        self._context = await self._browser.new_context(
            viewport={"width": self._viewport[0], "height": self._viewport[1]}
        )
        # Init script runs on every navigation - critical for the listener
        # counter to reset between iterations.
        await self._context.add_init_script(_INSTRUMENTATION_JS)
        self._page = await self._context.new_page()

        # Wire console + pageerror handlers ONCE; the buffers themselves are
        # reset per-test (see load_and_test).
        self._page.on("console", self._on_console)
        self._page.on("pageerror", self._on_pageerror)

    def _on_console(self, msg) -> None:
        text = _truncate(msg.text, _MAX_MSG_LEN)
        t = msg.type
        if t == "error":
            # 2.4: split feed for the report. _errors stays as the
            # combined view for backward compat with downstream code.
            self._console_errors.append(text)
            self._errors.append(text)
        elif t == "warning":
            self._warnings.append(text)
        else:
            self._logs.append(text)

    # Errors that fire only because the headless/auto-input harness can't
    # provide a real user gesture. They aren't real bugs — pointer-lock
    # works fine for a human player — so routing them through page_errors
    # poisons the regression detector (agent.py auto-revert), throwing
    # away good patches on FPS-style games every other iter. Downgrade
    # to soft warnings: the model still sees them and can decide whether
    # to defensive-wrap, but the harness doesn't count them as a
    # regression. DOOM trace 20260523_152317 is the motivating case
    # (iters 5/7/11 all reverted on this).
    _HARNESS_ENV_ERROR_PATTERNS: tuple[str, ...] = (
        "not valid for pointer lock",
        "pointer lock", # generic catch for "Document is not focused" variants
    )

    def _on_pageerror(self, exc) -> None:
        text = _truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN)
        low = text.lower()
        if any(p in low for p in self._HARNESS_ENV_ERROR_PATTERNS):
            self._warnings.append(text + "  [harness-env, not counted as regression]")
            return
        self._page_errors.append(text)
        self._errors.append(text)

    _ASSET_DECODE_SETTLE_JS = (
        "(()=>{"
        "const html=document.documentElement.innerHTML;"
        "const hasRefs=/_assets\\/[A-Za-z0-9_\\-]+\\.png/.test(html);"
        "if(!hasRefs)return{need:false,ready:true};"
        "if(window._assetsReady===true)return{need:true,ready:true};"
        "if(window.__assetDecodeSettled===true)return{need:true,ready:true};"
        "try{"
        "  if(typeof ASSETS==='object'&&ASSETS){"
        "    const vals=Object.values(ASSETS);"
        "    if(vals.length>0&&vals.every(img=>img&&img.naturalWidth>0))"
        "      return{need:true,ready:true};"
        "  }"
        "}catch(e){}"
        "return{need:true,ready:false};"
        "})()"
    )

    async def _wait_for_session_assets_ready(
        self, *, timeout_ms: int = 8000, raf_ticks: int = 4,
    ) -> dict[str, Any]:
        """Poll until async asset loaders finish, then tick extra RAF frames.

        Chromium often samples __drawImageEvents before loadAssets() resolves;
        this settle window lets real drawImage calls land before the undrawn
        audit runs. Returns {ready, waited_ms, raf_ticks}.
        """
        import asyncio
        import time

        start = time.monotonic()
        ready = False
        need = False
        while True:
            try:
                status = await self._safe_eval(self._ASSET_DECODE_SETTLE_JS)
            except Exception:
                status = {"need": False, "ready": True}
            if not isinstance(status, dict):
                status = {"need": False, "ready": True}
            need = bool(status.get("need"))
            ready = bool(status.get("ready"))
            if not need or ready:
                break
            if int((time.monotonic() - start) * 1000) >= timeout_ms:
                break
            await asyncio.sleep(0.05)
        waited_ms = int((time.monotonic() - start) * 1000)
        try:
            await self._safe_eval(
                f"(async()=>{{for(let i=0;i<{int(raf_ticks)};i++)"
                "await new Promise(r=>requestAnimationFrame(r));}})()"
            )
        except Exception:
            pass
        return {"ready": ready, "waited_ms": waited_ms, "raf_ticks": int(raf_ticks)}

    async def load_and_test(
        self,
        path: str | Path,
        screenshot_path: str | Path | None = None,
        *,
        probes: list[dict] | None = None,
        opening_book_recipes: list[dict] | None = None,
        screenshot_before_path: str | Path | None = None,
        screenshot_action_path: str | Path | None = None,
        criteria: str | None = None,
        goal: str | None = None,
        visual_recipe_id: str | None = None,
        asset_decode_settle: bool = True,
    ) -> dict[str, Any]:
        """Navigate to the file, let it run, return the report.

        New (this version) compared to the old text-only test:
          - Takes a screenshot at the end (saved to screenshot_path if given).
          - Detects FROZEN canvas by sampling pixels twice (~1s apart).
          - Runs an input smoke test (arrow keys, WASD, space) and checks
            whether any of them produced a canvas pixel change. If none did,
            the game probably ignores input.

        All of that ends up as additional report fields and soft_warnings the
        model treats with the same urgency as crashes.
        """
        import asyncio

        # Auto-reopen logic: if the user (or the system) closed the browser window unexpectedly,
        # or if _page has become closed/unusable, we proactively reconstruct and reopen it.
        browser_is_closed = True
        try:
            if self._page is not None and self._browser is not None and self._browser.is_connected():
                # Test the connection to ensure the page is active and connected
                await self._page.evaluate("1")
                browser_is_closed = False
        except Exception:
            browser_is_closed = True

        if browser_is_closed:
            try:
                await self.close()
            except Exception:
                pass
            await self.start()

        # Reset per-test buffers. The handlers stay attached.
        self._errors.clear()
        self._console_errors.clear()
        self._page_errors.clear()
        self._warnings.clear()
        self._logs.clear()

        path = Path(path).resolve()
        file_url = f"file://{path}"
        try:
            await self._page.goto(file_url, wait_until="load", timeout=10_000)
            
            # --- Dynamic Splash-Clicker Start Screen Bypass ---
            # Automatically detect and click common "Start", "Play", "Enter", "Begin" overlay buttons,
            # or click the canvas/body so interactive and loop elements start running before we test.
            try:
                # Common elements representing start/play overlays or canvas buttons
                selectors = [
                    "button", "#startBtn", "#start", "#playBtn", "#play", 
                    ".start", ".play", "#restartBtn", "#restart", "canvas", "body"
                ]
                for sel in selectors:
                    el = await self._page.query_selector(sel)
                    if el and await el.is_visible() and await el.is_enabled():
                        text = (await el.inner_text() or "").lower()
                        # Match standard terms (like "start", "play", "enter", "begin", "hell")
                        if any(w in text for w in ("start", "play", "enter", "begin", "hell", "click", "tap")):
                            await el.click()
                            await asyncio.sleep(0.5) # Allow 500ms for assets/scene setup
                            break
            except Exception:
                pass
        except Exception as e:
            return _build_report(
                [f"PAGE FAILED TO LOAD: {e}"], [], [], "", None,
                {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}, "",
            )

        # Sleep half the budget; sample canvas; sleep rest; sample again.
        # If both samples are byte-identical the game is FROZEN even if
        # requestAnimationFrame fired (it's drawing the same frame).
        # We also take a 32x32 hash at each moment — the boolean equality of
        # the two hashes is what drives the FROZEN heuristic (it catches
        # slowly-moving content the 9-pixel probe misses).
        half = max(self._run_seconds / 2.0, 0.5)
        # Capability-round item 4 — state timeline. Sample the flattened
        # game state ~6x across the observation window (3 per half, total
        # sleep unchanged) so the fix prompt can reason from dynamics
        # ("cpu.x never changed") instead of booleans. Reuses the
        # _GAMESTATE_SNAPSHOT_JS flattening from the input smoke test.
        state_samples: list = []
        seg = half / 3.0
        for _ in range(3):
            await asyncio.sleep(seg)
            state_samples.append(await self._safe_eval(_GAMESTATE_SNAPSHOT_JS))
        canvas_first = await self._safe_eval(_CANVAS_PROBE_JS)
        hash_first = await self._safe_eval(_CANVAS_HASH_JS)
        # Optional "before-input" screenshot — captures the t=startup
        # state so a VLM can later see motion as a before/after pair.
        screenshot_before_saved: str | None = None
        if screenshot_before_path is not None:
            try:
                bp = Path(screenshot_before_path)
                bp.parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=str(bp), full_page=False)
                screenshot_before_saved = str(bp)
            except Exception:
                screenshot_before_saved = None
        for _ in range(3):
            await asyncio.sleep(seg)
            state_samples.append(await self._safe_eval(_GAMESTATE_SNAPSHOT_JS))
        canvas_info = await self._safe_eval(_CANVAS_PROBE_JS)
        hash_last = await self._safe_eval(_CANVAS_HASH_JS)

        # ---- input smoke test ---------------------------------------------
        # Most small-model bugs we miss are "controls don't work". Fire a few
        # standard inputs and check if pixels change. Captured pre/post
        # snapshots are compared via a key-set hash from the same probe.
        input_test = await self._input_smoke_test(criteria=criteria)

        # ---- action frame (peak input-attributable transient) -------------
        # The smoke test captures one screenshot at the moment a held key was
        # producing its largest canvas change — the game mid-ACTION rather
        # than at rest. Write it to disk if a path was given, then pop the raw
        # bytes so the report stays paths/booleans only.
        screenshot_action_saved: str | None = None
        action_png_bytes = input_test.pop("action_frame_png_bytes", None) \
            if isinstance(input_test, dict) else None
        if action_png_bytes and screenshot_action_path is not None:
            try:
                ap = Path(screenshot_action_path)
                ap.parent.mkdir(parents=True, exist_ok=True)
                ap.write_bytes(action_png_bytes)
                screenshot_action_saved = str(ap)
            except Exception:
                screenshot_action_saved = None

        # ---- ALL per-action frames (one image per named action key) --------
        # Save each so the trace is debuggable per action (J-kick, K-kick,
        # L-fireball …), not just the single peak frame. Named
        # <action_base>_<KeyCode>.png next to the action screenshot. Pop the
        # raw bytes; the report keeps paths only.
        action_frames_bytes = input_test.pop("action_frames_png_bytes", None) \
            if isinstance(input_test, dict) else None
        action_frame_paths: dict[str, str] = {}
        if isinstance(action_frames_bytes, dict) and screenshot_action_path is not None:
            base = Path(screenshot_action_path)
            stem = base.stem  # e.g. "iter_03_action"
            for keycode, png in action_frames_bytes.items():
                if not png:
                    continue
                try:
                    fp = base.with_name(f"{stem}_{keycode}.png")
                    fp.write_bytes(png)
                    action_frame_paths[str(keycode)] = str(fp)
                except Exception:
                    pass
        # NOTE: `action_frame_paths` is attached to the report AFTER
        # `_build_report` runs (below) — `report` does not exist yet here.
        # Assigning it at this point raised UnboundLocalError and crashed the
        # whole harness — and only when action keys actually produced frames,
        # i.e. exactly on good games (Qwen dojo-fight trace 20260610_151443
        # iters 1-2 never got a test report because of this).
        # Pop the raw per-key fake-action signal now; the gating decision is made
        # later (after referenced_assets is computed) — stash it on a local.
        _fake_actions = input_test.pop("fake_actions", None) \
            if isinstance(input_test, dict) else None

        # ---- model-proposed probes ----------------------------------------
        # Agent emits <probes> in Phase A — JSON list of {name, expr} where
        # expr is a JS expression that should evaluate truthy on the running
        # game. We run each in the page context. Per-probe results join the
        # report so the model sees its own assertions checked.
        probe_results: list[dict[str, Any]] = []
        if probes:
            # Probe-ordering fix (trace 20260612_171752): run READ-ONLY
            # probes first so a side-effecting probe (e.g. restart_resets
            # dispatching KeyR → reset() zeroes state.frame) can't poison a
            # later read-only probe (raf_firing: `frame > 0`). Relative
            # order is preserved within each group; the report keeps the
            # original probe order so the model sees a stable list.
            indexed = list(enumerate(probes))
            ordered = (
                [(i, p) for i, p in indexed
                 if not _probe_has_side_effects(str(p.get("expr") or ""))]
                + [(i, p) for i, p in indexed
                   if _probe_has_side_effects(str(p.get("expr") or ""))]
            )
            results_by_idx: dict[int, dict[str, Any]] = {}
            effectful_run_so_far: list[str] = []
            for orig_idx, p in ordered:
                pname = str(p.get("name") or "probe")[:60]
                # QTE-gate fix (serial10 game 9): recipe auto_probes are
                # trusted, can legitimately be long (the QTE window-gating
                # probe is 740 chars). The old [:600] cap sliced valid JS
                # mid-statement → SyntaxError → quarantine, so the real gate
                # never ran. Raise the eval cap to 2000 (report copy below is
                # still bounded at [:200], so report size is unchanged).
                pexpr = str(p.get("expr") or "true")[:2000]
                is_effectful = _probe_has_side_effects(pexpr)
                # Board-game probes often dispatch mousedown/click while the
                # game only listens for pointerdown — patch before eval.
                if is_effectful and "MouseEvent" in pexpr:
                    pexpr = _patch_probe_pointer_board_clicks(pexpr)
                # P3 (run_04 holochess iter 1): isolate CONSECUTIVE
                # side-effecting probes. select_works / move_tweens /
                # cpu_auto_replies each click + await; the first leaves the
                # game mid-animation (state.animating=true blocks clicks), so
                # the next probe's clicks are silently ignored and it reads
                # falsy on an otherwise-correct build (3 probes failed iter 1,
                # all passed by iter 3). When a prior side-effecting probe has
                # already run AND the game exposes a reset(), restore a clean
                # non-animating state first so each self-contained probe starts
                # fresh. Read-only probes ran earlier (ordering fix above) and
                # are unaffected. Genre-free: gated only on reset() existing.
                if is_effectful and effectful_run_so_far:
                    try:
                        _did_reset = await self._safe_eval(
                            "(()=>{const g=window.game||{};"
                            "const f=g.reset||g.restart;"
                            "if(typeof f==='function'){f.call(g);return true;}"
                            "return false;})()"
                        )
                        if _did_reset:
                            await asyncio.sleep(0.1)
                    except Exception:
                        pass
                ok, err, err_kind = await self._run_probe(pexpr)
                if not ok and not err and not is_effectful:
                    # Falsy read-only probe (no eval error): retry once
                    # after a short delay to absorb a startup timing race
                    # (e.g. probe sampled before the first RAF tick).
                    await asyncio.sleep(0.3)
                    ok, err, err_kind = await self._run_probe(pexpr)
                # 1.2: tainted-canvas / cross-origin / SecurityError on
                # getImageData are HARNESS-side limitations, not game
                # bugs — the actual canvas might be perfectly rendered.
                # Don't let a brittle probe block shipping a working
                # game; classify it as warning, not failure. The launch
                # flags from 1.1 should prevent this in practice but
                # defense-in-depth.
                taint_signal = bool(err and any(
                    s in err.lower()
                    for s in ("tainted", "cross-origin", "securityerror")
                ))
                entry: dict[str, Any] = {
                    "name": pname, "expr": pexpr[:200], "ok": ok, "err": err,
                }
                if err_kind:
                    entry["kind"] = "eval_error"
                    entry["error_class"] = err_kind
                if taint_signal and not ok:
                    entry["ok"] = True
                    entry["downgraded"] = (
                        "harness-side CORS/taint (probe relies on "
                        "getImageData/toDataURL) — treating as pass"
                    )
                    entry.pop("kind", None)
                    entry.pop("error_class", None)
                # Ordering hint: a falsy probe that ran AFTER a
                # side-effecting probe may be reading mutated state — say
                # so in the failure text so a small model isn't left
                # guessing at a contradiction with the state timeline.
                if not entry["ok"] and not err_kind and effectful_run_so_far:
                    entry["err"] = (
                        (entry.get("err") or "evaluated falsy")
                        + " (note: ran after side-effecting probe(s) "
                        f"[{', '.join(effectful_run_so_far[:3])}] which "
                        "dispatch events / reset state — their side effects "
                        "may have mutated the state this probe reads)"
                    )
                if is_effectful:
                    effectful_run_so_far.append(pname)
                results_by_idx[orig_idx] = entry
            probe_results = [results_by_idx[i] for i in sorted(results_by_idx)]

        title = (await self._page.title()) or ""
        try:
            body_text = await self._page.evaluate(
                "document.body ? document.body.innerText : ''"
            ) or ""
        except Exception:
            body_text = ""

        listener_info = {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}
        try:
            counts = await self._page.evaluate("window.__listenerCount || null") or {}
            for k in ("document", "window", "body", "other"):
                listener_info[k] = int(counts.get(k, 0) or 0)
            listener_info["total"] = sum(listener_info[k] for k in ("document", "window", "body", "other"))
        except Exception:
            pass

        # ---- screenshot (always taken; saved if path given) ---------------
        screenshot_saved: str | None = None
        if screenshot_path is not None:
            try:
                p = Path(screenshot_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=str(p), full_page=False)
                screenshot_saved = str(p)
            except Exception:
                screenshot_saved = None

        # ---- detect frozen canvas: 32x32 hash identical at t=half and t=full
        # The hash covers the whole playfield, so even one pixel of motion
        # somewhere flips the result. Only flag FROZEN when we have content
        # (not blank) and the loop is actually running (RAF fired).
        frozen = None
        if hash_first is not None and hash_last is not None and canvas_info:
            same = (hash_first == hash_last)
            if same and canvas_info.get("blank") is False and canvas_info.get("raf_ran"):
                frozen = True
            else:
                frozen = False

        report = _build_report(
            list(self._errors), list(self._warnings), list(self._logs),
            title, canvas_info, listener_info, body_text,
        )
        # Attach the new fields. The model never sees raw bytes - just paths
        # and small booleans / counts via format_report_for_model.
        report["screenshot"] = screenshot_saved
        report["screenshot_before"] = screenshot_before_saved
        report["screenshot_action"] = screenshot_action_saved
        # Item 4: dynamics digest over the ~6 observation-window samples.
        # Empty string when the game exposed no state global.
        try:
            report["state_timeline"] = summarize_state_timeline(
                state_samples, window_seconds=2 * half,
            )
        except Exception:
            report["state_timeline"] = ""
        # Moved here from the per-action-frame save block above: `report`
        # only exists from this point on (UnboundLocalError fix, 2026-06-10).
        if action_frame_paths:
            report["action_frames"] = action_frame_paths
        report["action_key"] = (
            input_test.get("action_key") if isinstance(input_test, dict) else None
        )
        # CANVAS-DEFAULT-SIZE gate (fix round): live canvas is the untouched
        # 300x150 browser default and the source never sizes it. Gating —
        # the game is being drawn mostly off-canvas.
        try:
            _src_html = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _src_html = ""
        _cds = _canvas_default_size_warning(canvas_info, _src_html)
        if _cds:
            report["soft_warnings"].append(_cds)
        # JS-SOURCE-IN-BODY gate (trace 20260611_213744): visible body text
        # is JavaScript source → a script boundary broke. Gating — name the
        # cause instead of leaving the model to guess from blank-canvas
        # heuristics. Uses the FULL body text, not the truncated sample.
        _jsb = _js_source_in_body_warning(body_text)
        if _jsb:
            report["soft_warnings"].append(_jsb)
        # Animation-liveness gate: a responsive action that renders a single
        # held pose (not animated) is a hard "must fix" — appended as a
        # soft_warning so the final ok-recompute flips report["ok"]=False and
        # the agent cannot ship a non-animated action. Objective + genre-free.
        _sa = input_test.get("static_action") if isinstance(input_test, dict) else None
        if isinstance(_sa, dict) and _sa.get("key"):
            report["static_action"] = _sa
            report["soft_warnings"].append(
                f"STATIC-ACTION: {_sa['key']} is responsive but renders as a "
                f"single held pose (in-hold canvas motion {_sa['delta']} < "
                f"{_STATIC_POSE_MAX_INHOLD}) while the rest of the canvas "
                f"animates. Animate it: cycle >=2 distinct frames or apply "
                f"continuous motion (translate/rotate/scale) during the "
                f"active window — do not hold one static frame."
            )
        # Stuck-player gate: a movement key registered input but the player's
        # position never changed → the player can't move (spawned in a wall,
        # collision blocking every direction). This read as "responsive" before
        # — the Pac-Man "starts in a wall and doesn't move" failure. Genre-free.
        if isinstance(input_test, dict) and input_test.get("input_registered_without_move"):
            report["player_stuck"] = True
            report["soft_warnings"].append(
                "PLAYER-STUCK: a movement key (arrows/WASD) registers input (a "
                "direction/state field changes) but the player's POSITION "
                "(x/y/tile) never changes on ANY direction — the player cannot "
                "move. Common causes: it spawned inside a wall, or the "
                "wall-collision check blocks every move (wrong coordinate units "
                "— pixel vs tile, or off-by-one). Spawn the player on a known "
                "corridor cell and make at least one direction actually change "
                "its position."
            )
        # Control-recovery gate (fix-round item 1): a movement key moved the
        # player at the start of the run but no longer does after gameplay
        # (one retry included). The permanent stun-lock family — gating,
        # because the player has lost control of the game.
        _cnr = input_test.get("control_not_recovered") if isinstance(input_test, dict) else None
        if isinstance(_cnr, dict) and _cnr.get("key"):
            report["control_not_recovered"] = _cnr
            report["soft_warnings"].append(
                f"CONTROL-NOT-RECOVERED: {_cnr['key']} moved the player at the "
                f"start of the run ({', '.join(_cnr.get('leaves', []))}), but "
                "after gameplay (taking hits) it no longer changes any "
                "position field — even after a grace wait and retry. The "
                "usual cause is a hit/stun/knockdown state that never "
                "expires: its timer is skipped by an early-return guard "
                "(e.g. `if (action === 'hit') return;` placed ABOVE the "
                "timer countdown), so the entity is locked forever. Make the "
                "stun timer decrement BEFORE any early return, and return to "
                "a controllable state when it hits 0."
            )
        report["frozen_canvas"] = frozen
        report["input_test"] = input_test
        report["probes"] = probe_results
        # Phase fix #4 (2026-05-23 traces) — a frozen canvas was being
        # set on the report but never feeding into the ok decision, so
        # iters could ship with `ok=True, frozen_canvas=True`. The
        # 2026-05-22 Pac-Man trace iter 7 is the case study. Soft
        # warnings already flip ok to False; this is a one-line wire-up.
        # Model-agnostic — applies to anything that renders to canvas.
        if frozen is True:
            # 2026-05-31: distinguish a TRUE freeze from idle-by-design. A
            # fighting / animation game sits on a static idle sprite until the
            # player presses a key — the canvas is legitimately unchanged
            # between t=0.5s and t=1.0s, but it is NOT frozen. If the input
            # smoke test proved keys DO change the canvas, this is idle-by-
            # design: report it as a non-blocking warning, do NOT add a
            # soft_warning (which would flip ok=False and starve the session
            # on a false positive — see here-s-a-tight-test-prompt 20260530).
            # A truly frozen game (input changes nothing) still hard-blocks.
            input_responsive = bool(
                input_test.get("ran") and input_test.get("any_change") is True
            )
            report["frozen_canvas_input_responsive"] = input_responsive
            # Turn-based board games (checkers trace run_05): static board
            # between clicks is idle-by-design, not a freeze — pointer-primary
            # with a 2D board in state and no keyboard-driven pixel delta.
            _turn_based_board_idle = False
            try:
                _turn_based_board_idle = bool(
                    await self._safe_eval(
                        "window.__hasPointerBoardState === true"
                        " && (window.__listenerTypes||{}).pointer > 0"
                    )
                )
            except Exception:
                _turn_based_board_idle = False
            if input_responsive:
                report.setdefault("warnings", []).append(
                    "FROZEN-AT-IDLE (not blocking): canvas is static between "
                    "t=0.5s and t=1.0s, but the input test confirms keys change "
                    "the canvas — idle-by-design, not a freeze. Add a subtle "
                    "continuous idle animation (breathing/bob) to silence this "
                    "(see playbook ambient-idle-pixel-delta)."
                )
            elif _turn_based_board_idle:
                report.setdefault("warnings", []).append(
                    "FROZEN-AT-IDLE (not blocking): turn-based board is static "
                    "between clicks while RAF is running — expected idle, not a "
                    "freeze. Pieces animate only after pointer input."
                )
            else:
                report["soft_warnings"].append(
                    "FROZEN-CANVAS: 32×32 canvas hash unchanged between t=0.5s "
                    "and t=1.0s while RAF was firing AND input did not change "
                    "the canvas. The render loop is alive but drawing the same "
                    "frame — likely a stuck game-state, a draw() that "
                    "early-returns before the entity layer, or all update "
                    "timers stopped advancing."
                )
        # 2.4: split error sources so the model + trace can tell apart
        # "console.error('...')" (game-logged, often informational) from
        # "UNCAUGHT TypeError" (real crash). Existing report["errors"]
        # is the union — kept for backward compat with downstream code.
        report["console_errors"] = list(self._console_errors)
        report["page_errors"] = list(self._page_errors)
        report["opening_book_checks"] = await self._run_opening_book_recipes(
            path, opening_book_recipes or [], input_test=input_test,
            canvas_info=canvas_info, frozen=frozen,
            goal=goal or "",
            visual_recipe_id=visual_recipe_id,
        )
        for chk in report["opening_book_checks"]:
            if chk.get("ok") is False and chk.get("hard"):
                report["soft_warnings"].append(
                    f"OPENING BOOK CHECK FAILED [{chk.get('id','recipe')}]: "
                    f"{chk.get('err','')[:180]}"
                )
        # Phase 1.5.1 — state-has-entity-but-canvas-doesn't-render-it.
        # Detects the failure shape from the 2026-05-22 Pac-Man trace:
        # gameState.pacman.x exists, the probe `gameState.pacman.x !==
        # undefined` passes, the canvas is NOT blank (maze + ghosts +
        # dots render), but the Pac-Man itself is invisible because the
        # draw() function never references the sprite.
        # General: works on any game whose state exposes a top-level
        # field with numeric .x/.y. Tries both raw-pixel and inferred-
        # tile-coordinate interpretations. Soft warning so the model
        # sees it in the report and the harness ok flag flips.
        try:
            entity_render_result = await self._safe_eval(_ENTITY_RENDERED_JS)
        except Exception:
            entity_render_result = None
        if isinstance(entity_render_result, dict):
            missing = entity_render_result.get("missing") or []
            report["entity_render_check"] = entity_render_result
            _ent_probes_green = bool(report.get("probes")) and all(
                p.get("ok") for p in report.get("probes") or []
            )
            _ent_no_errors = not report.get("errors") and not report.get("page_errors")
            for m in missing:
                if not isinstance(m, dict):
                    continue
                name = m.get("name", "?")
                bg_frac = m.get("bg_fraction", 0)
                pk = m.get("position_kind", "?")
                ex = m.get("x", 0)
                ey = m.get("y", 0)
                _ent_msg = (
                    f"ENTITY-NOT-RENDERED [{name}]: gameState.{name} is "
                    f"at ({ex},{ey}) but the canvas at that position is "
                    f"{int(bg_frac * 100)}% background "
                    f"(position interpreted as {pk}). The entity is "
                    "in state but not drawn — check the draw() / render "
                    "function references this entity, and that any "
                    "sprite/image is decoded before drawImage is called."
                )
                if _ent_probes_green and _ent_no_errors:
                    report.setdefault("warnings", []).append(
                        "ADVISORY (non-blocking) — " + _ent_msg
                        + " Behavioral probes pass; transparent sprite "
                        "padding or a blink frame may have caused a "
                        "false sample — verify draw uses center anchoring."
                    )
                else:
                    report["soft_warnings"].append(_ent_msg)
        # Drawn-asset detector — read the drawImage shim's event buffer
        # and diff against the session's known asset filenames. Catches
        # "model loaded the PNG into ASSETS[name] but never called
        # drawImage on it" — the failure shape from the 2026-05-23 chess
        # trace where the model wrote `await img.decode()` then drew
        # chess pieces with ctx.fillText Unicode glyphs.
        _decode_settle: dict[str, Any] = {}
        if asset_decode_settle:
            try:
                _decode_settle = await self._wait_for_session_assets_ready()
            except Exception:
                _decode_settle = {"ready": False, "waited_ms": 0, "raf_ticks": 0}
        else:
            _decode_settle = {"ready": False, "waited_ms": 0, "raf_ticks": 0, "skipped": True}
        report["asset_decode_settle"] = _decode_settle
        try:
            draw_events = await self._safe_eval(
                "window.__drawImageEvents || []"
            )
        except Exception:
            draw_events = None
        if isinstance(draw_events, list):
            drawn_sources = []
            for ev in draw_events:
                if isinstance(ev, dict):
                    s = ev.get("src") or ""
                    if isinstance(s, str):
                        drawn_sources.append(s)
            drawn_blob = "\n".join(drawn_sources).lower()
            # Find every PNG path the HTML references (already on disk,
            # already in the resolved-asset-paths report field if the
            # caller set it). We derive the candidate name list from
            # the HTML file's relative-path mentions to stay agent-
            # independent — the harness doesn't need to know which
            # session_dir produced the assets.
            try:
                html_text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                html_text = ""
            import re as _re
            asset_path_re = _re.compile(
                r"['\"]\./([A-Za-z0-9_\-]+_assets/[A-Za-z0-9_\-]+\.png)['\"]"
            )
            referenced_assets: dict[str, str] = {}
            for m in asset_path_re.finditer(html_text):
                rel_path = m.group(1)
                stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                referenced_assets[stem] = rel_path
            undrawn: list[str] = []
            for stem, rel_path in referenced_assets.items():
                # Drawn-sources match if the path's filename appears in
                # the recorded src URL (drawImage sees a file:// URL).
                fname = rel_path.rsplit("/", 1)[-1].lower()
                if not _drawn_blob_contains_asset_fname(drawn_blob, fname):
                    undrawn.append(stem)
            report["drawn_asset_check"] = {
                "loaded_referenced_count": len(referenced_assets),
                "drawn_event_count": len(drawn_sources),
                "undrawn": undrawn,
            }
            # Only flag when SOME assets exist AND a significant fraction
            # are unused. Genuine all-undrawn (because the page just
            # loaded and no frames rendered) shouldn't false-alarm — we
            # already gate via RAF-fired/non-blank canvas; if those held
            # and we still have undrawn assets, the gap is real.
            if (
                referenced_assets
                and undrawn
                and len(undrawn) >= max(1, len(referenced_assets) // 3)
                and canvas_info
                and canvas_info.get("raf_ran")
                and canvas_info.get("blank") is False
            ):
                preview = ", ".join(undrawn[:8])
                if len(undrawn) > 8:
                    preview += f", … (+{len(undrawn) - 8} more)"
                _undrawn_text = (
                    f"ASSETS_LOADED_BUT_UNDRAWN [{preview}]: "
                    f"{len(undrawn)}/{len(referenced_assets)} asset PNG(s) "
                    "loaded but never drawn this run. The usual cause is a "
                    "SPRITE-KEY MISMATCH: you build a lookup key (e.g. "
                    "'left_idle' from name+'_'+state) that does NOT equal the "
                    "generated asset name (e.g. 'left_fighter_idle'), so "
                    "ASSETS[key] is undefined and you silently fall back to a "
                    "fillRect block. The generated names are EXACTLY the ones "
                    "listed in GENERATED ASSETS. Fix: fetch every sprite via the "
                    "provided `sprite(key)` resolver (it tolerates key drift and "
                    "draws a loud MISSING marker on a true miss) — do NOT index "
                    "ASSETS directly, and do NOT draw a plain fillRect for an "
                    "entity that has a sprite. The undrawn assets above are: "
                    f"{preview}."
                )
                # Fix-round item 2: state-conditional pose sprites
                # (player_death, cpu_duck, …) structurally CANNOT all draw in
                # a 3s window + smoke test, so a persisting UNDRAWN finding on
                # a behaviorally-green build is cosmetic, not a blocker —
                # trace 20260610_185238 sat at 8/8 probes for 6 iters while
                # this gate kept ok=False. First occurrence still gates (real
                # key-mismatch bugs deserve one forced fix turn); a repeat
                # with all probes passing and zero errors goes to the
                # non-gating warnings channel (behavioral probes gate;
                # cosmetics inform).
                _probes_green = bool(report.get("probes")) and all(
                    p.get("ok") for p in report.get("probes") or []
                )
                _no_errors = not report.get("errors") and not report.get("page_errors")
                _entity_missing_count = len(
                    (report.get("entity_render_check") or {}).get("missing") or []
                )
                _opening_hard_green = all(
                    chk.get("ok") is not False or not chk.get("hard")
                    for chk in report.get("opening_book_checks") or []
                )
                _behavior_green_no_missing = (
                    _probes_green and _no_errors and _entity_missing_count == 0
                    and _opening_hard_green
                )
                # P2 (run_04 Dragon iter 1): scene-indexed cutscene/QTE games
                # reference one bg_* per scene but only the ACTIVE scene's bg
                # draws during the 3s smoke window — the other bg_* are
                # legitimately "loaded but undrawn", NOT a sprite-key-mismatch
                # bug. Dragon's Lair iter 1 passed 8/8 probes with 0 errors yet
                # gated on 19/24 undrawn (mostly bg_*). When live state exposes
                # a NUMERIC scene index and the undrawn set is predominantly
                # scene backgrounds, demote to advisory on a behaviorally-green
                # build. Genre-free: keys on a numeric state.scene, not subject.
                try:
                    _scene_idx = await self._safe_eval(
                        "(()=>{const s=window.state||window.gameState||window.g||{};"
                        "const v=(s.scene!==undefined)?s.scene:"
                        "((s.sceneIndex!==undefined)?s.sceneIndex:"
                        "((s.currentScene!==undefined)?s.currentScene:null));"
                        "return (typeof v==='number'&&isFinite(v))?v:null;})()"
                    )
                except Exception:
                    _scene_idx = None
                _scene_indexed = isinstance(_scene_idx, (int, float)) and not isinstance(_scene_idx, bool)
                _undrawn_mostly_bg = bool(undrawn) and (
                    sum(1 for s in undrawn if s.lower().startswith("bg_"))
                    >= max(1, len(undrawn) // 2)
                )
                _scene_offscreen_bg = (
                    _scene_indexed and _undrawn_mostly_bg
                    and _probes_green and _no_errors
                )
                _undrawn_pose_only = _undrawn_are_animation_poses_only(
                    undrawn, drawn_blob
                )
                _undrawn_state_gated = _undrawn_likely_state_gated(undrawn)
                _decode_settled = bool(
                    (_decode_settle or {}).get("ready") is True
                    and not (_decode_settle or {}).get("skipped")
                )
                _game_advancing = bool(
                    canvas_info
                    and canvas_info.get("raf_ran")
                    and canvas_info.get("blank") is False
                    and (
                        frozen is not True
                        or report.get("frozen_canvas_input_responsive")
                        or _turn_based_board_idle
                    )
                )
                _sprite_wrapper = (
                    "sprite(" in html_text and "drawImage" in html_text
                )
                _sprite_paths_wrapper = (
                    _sprite_wrapper
                    and "PATHS" in html_text
                    and "drawEntity(" in html_text
                )
                _decode_timing_demote = (
                    _decode_settled and _game_advancing and _no_errors
                    and (
                        _probes_green
                        or _sprite_paths_wrapper
                    )
                )
                if (
                    (_probes_green and _no_errors and (
                        _undrawn_pose_only
                        or _scene_offscreen_bg
                        or _undrawn_state_gated
                        or _sprite_wrapper
                        or _sprite_paths_wrapper
                    ))
                    or (self._undrawn_seen_before and _probes_green and _no_errors)
                    or _behavior_green_no_missing
                    or _decode_timing_demote
                ):
                    _advisory_tail = (
                        " Behavioral probes pass and a numeric scene index is "
                        "exposed; off-screen scene backgrounds simply have not "
                        "been reached during the test window."
                        if _scene_offscreen_bg else
                        " Behavioral probes pass; hop/lift/slam pose sprites "
                        "may simply not have triggered on the static opening "
                        "board — idle/base sprites ARE drawn."
                        if _undrawn_pose_only else
                        " Behavioral probes pass; undrawn assets look "
                        "state/wave gated and may not have triggered during "
                        "the short smoke window."
                        if _undrawn_state_gated else
                        " Behavioral probes pass; HTML uses sprite() + "
                        "drawImage — likely a draw-call instrumentation miss."
                        if _sprite_wrapper and not _sprite_paths_wrapper else
                        " Behavioral probes pass; HTML uses PATHS + drawEntity() + "
                        "sprite() — first-build wiring is present; undrawn is likely "
                        "decode timing or state-gated sprites (spider/bullet) not "
                        "seen in the smoke window."
                        if _sprite_paths_wrapper else
                        " Behavioral probes pass; asset decode settled and "
                        "the canvas is advancing — undrawn stems are likely "
                        "a Chromium timing false positive (sprites visible in "
                        "Safari). Re-check after reload if art still missing."
                        if _decode_timing_demote else
                        " Behavioral probes pass and no entity-render "
                        "failure is present; state-conditional pose sprites "
                        "may simply not have triggered during the test window."
                    )
                    report.setdefault("warnings", []).append(
                        "ADVISORY (non-blocking) — " + _undrawn_text
                        + _advisory_tail
                    )
                else:
                    report["soft_warnings"].append(_undrawn_text)
                self._undrawn_seen_before = True
            else:
                # Finding absent this run — break the persistence streak so a
                # future regression gates again.
                self._undrawn_seen_before = False
            # Missing-asset LOAD gate (C1): a generated _assets PNG the HTML
            # references but the browser failed to FETCH (404 / file-not-found)
            # is a hard broken-art failure — the game shows colored-box
            # placeholders instead of the generated art (holochess 20260623).
            # Unlike the undrawn check above (a pose sprite may simply not have
            # triggered), a load failure is unambiguous, so it gates via
            # soft_warnings regardless of whether the model's own probes pass.
            if referenced_assets and _asset_load_failed(
                True,
                "\n".join(str(e) for e in (
                    (report.get("page_errors") or [])
                    + (report.get("errors") or [])
                    + (report.get("console_errors") or [])
                )),
            ):
                report["missing_asset_load_check"] = {"load_failed": True}
                report["soft_warnings"].append(
                    "MISSING_ASSET_LOAD: a generated _assets PNG the game "
                    "references failed to load (404 / file-not-found), so it is "
                    "drawing placeholder/colored-square fallbacks instead of the "
                    "generated art. Load every sprite through the provided "
                    "sprite()/ASSETS loader using the EXACT names in GENERATED "
                    "ASSETS and verify each relative path resolves."
                )
        # ---- fake-action: code-drawn limb pretending to be a sprite --------
        # Only when the game actually uses sprites. A key that visibly changed
        # the canvas but added NO new sprite source while code-draw
        # (fillRect/lineTo/stroke) increased = a faked action (e.g. a kick drawn
        # with ctx.lineTo over the idle sprite instead of swapping to the kick
        # sprite). Gating soft_warning so the model must drive actions by sprite.
        if isinstance(_fake_actions, dict) and referenced_assets:
            # Fix-round item 2: only flag keys the game DECLARED as actions
            # in <criteria>. The smoke test also presses default movement /
            # WASD keys; in trace 20260610_185238 this gate blamed KeyA/KeyS
            # — default presses, not the game's controls (arrows + J/K/L/R)
            # — and could never be cleared. Declared action keys are the
            # ones whose animation must be sprite-driven.
            # Declared action keys only — movement arrows/WASD are pressed by
            # the smoke-test defaults and must not gate (trace KeyA/KeyS;
            # criteria may also name ArrowLeft/ArrowRight as move, not attack).
            _declared_action_keys = (
                set(_parse_action_keys(criteria or ""))
                - _MOVEMENT_KEYS
                - _RESTART_KEYS
                # Phase 0: drop keys the criteria describes as start-wave / pause
                # / menu / build / sell etc. — they are not faked attacks.
                - _non_combat_action_keys(criteria or "")
                - _rotation_mechanic_action_keys(criteria or "")
            )
            faked = [
                kc for kc, info in _fake_actions.items()
                if isinstance(info, dict)
                and kc in _declared_action_keys
                and not info.get("new_sprite_src")
                and int(info.get("code_draw_delta", 0)) > 0
            ]
            if faked:
                preview = ", ".join(faked[:6])
                _faked_text = (
                    f"ACTION_DRAWN_NOT_SPRITED [{preview}]: pressing "
                    f"{'these keys' if len(faked) > 1 else 'this key'} changed "
                    "the canvas by CODE-DRAWING (ctx.fillRect / lineTo / stroke) "
                    "but did NOT draw a different sprite frame — i.e. the action "
                    "(kick/punch/etc.) is faked with code lines over the idle "
                    "sprite, not the real action sprite. Drive the action by "
                    "swapping to its generated sprite via sprite() (e.g. while "
                    "state.kicking, draw the *_kick sprite); never scribble a "
                    "limb with fillRect/lineTo on top of the character."
                )
                # Trace 20260612_171752: persistence downgrade mirroring the
                # UNDRAWN gate above. First occurrence gates (one forced fix
                # turn); a REPEAT with all probes passing and zero errors is
                # cosmetic — it held ok=False from iter 4 through every
                # continuation turn on a 7/7-probes build, blocked <done/>,
                # and the model burned its turns on contorted sprite-flash
                # hacks instead of the user's feedback.
                _probes_green_fa = bool(report.get("probes")) and all(
                    p.get("ok") for p in report.get("probes") or []
                )
                _no_errors_fa = (
                    not report.get("errors") and not report.get("page_errors")
                )
                if (
                    self._action_not_sprited_seen_before
                    and _probes_green_fa
                    and _no_errors_fa
                ):
                    report.setdefault("warnings", []).append(
                        "ADVISORY (non-blocking) — " + _faked_text
                        + " This finding has persisted while all behavioral "
                        "probes pass; the action's visible response may "
                        "legitimately be code-drawn (particles/flash) in "
                        "this game."
                    )
                else:
                    report["soft_warnings"].append(_faked_text)
                self._action_not_sprited_seen_before = True
            else:
                # Finding absent this run — break the persistence streak so
                # a future regression gates again.
                self._action_not_sprited_seen_before = False
            # ---- code-drawn EFFECT bolted on top of a real sprite action ---
            # The other half of the fake-action problem: the model DOES swap to
            # the kick sprite (new_sprite_src True) but ALSO draws a code shape
            # — a "motion line + flash" / reach-bar / limb — on top of it during
            # the action. The sprite already conveys the kick; the code overlay
            # is exactly the "stupid coded object instead of the sprite" the
            # user rejects (two_kickers test3: a stroke line + growing arc
            # ball over the kicking sprite). Keyed on STROKE/arc/lineTo delta
            # (not fillRect — backgrounds use fillRect), with a small threshold
            # so an incidental single stroke doesn't trip it. Gating: the model
            # must let the sprite carry the action.
            over_sprite = [
                kc for kc, info in _fake_actions.items()
                if isinstance(info, dict)
                and info.get("new_sprite_src")
                and int(info.get("stroke_delta", 0)) >= 2
            ]
            if over_sprite:
                preview = ", ".join(over_sprite[:6])
                # Fix-round item 2: NEVER gates. The sprite WAS drawn — the
                # extra code-draw is a motion line / flash, exactly the juice
                # the polish rubric requests. Advisory on the non-gating
                # warnings channel; the model can tune it, shipping is not
                # blocked (trace 20260610_185238 held ok=False on this).
                report.setdefault("warnings", []).append(
                    f"CODE_DRAWN_OVER_SPRITE (advisory — does not block "
                    f"shipping) [{preview}]: the action DID draw its "
                    "sprite, but ALSO code-drew shapes (ctx.stroke / lineTo / arc "
                    "— a motion line, reach-bar, flash, or limb) on top of the "
                    "character. If it reads as a stray object rather than an "
                    "effect, remove the stroke/arc overlay and let the *_kick / "
                    "*_punch sprite carry the action."
                )
        # Procedural-regression detector. Independent of (but coordinated
        # with) ASSETS_LOADED_BUT_UNDRAWN above. The drawImage shim says
        # WHICH assets weren't drawn; this says WHAT was drawn instead.
        # Combined signal: "sprites declared + N big rectangles drawn +
        # few/no drawImage calls" → strong evidence the model regressed
        # entities to colored boxes. Mortal-kombat 2026-05-24 trace had
        # P2 rendered as "a massive solid blue rectangle"; today's
        # detector would have had no signal for that shape until the
        # critic happened to mention it.
        #
        # Conservative gate: must have ≥3 referenced assets AND ≥30 big
        # fillRect calls AND drawImage_count < big_fillRect_count // 5.
        # That ratio means big rectangles outnumber drawImage calls 5:1,
        # which only happens in true regression. Tile-based backgrounds
        # (e.g. a maze drawn as 30x30 fillRect tiles) also draw sprites
        # for entities, keeping the ratio under 5:1.
        try:
            fill_events = await self._safe_eval(
                "window.__fillRectEvents || []"
            )
        except Exception:
            fill_events = None
        if isinstance(fill_events, list) and isinstance(draw_events, list):
            big_rect_count = len(fill_events)
            draw_image_count = len(draw_events)
            if (
                len(referenced_assets) >= 3
                and big_rect_count >= 30
                and draw_image_count < max(1, big_rect_count // 5)
                and canvas_info
                and canvas_info.get("raf_ran")
                and canvas_info.get("blank") is False
            ):
                # Median rect size for the report so the model knows
                # what scale the placeholders are at.
                sizes = sorted(
                    (int(ev.get("w", 0)) * int(ev.get("h", 0)))
                    for ev in fill_events
                    if isinstance(ev, dict)
                )
                median_area = sizes[len(sizes) // 2] if sizes else 0
                likely_sites: list[str] = []
                try:
                    html_for_sites = Path(path).read_text(
                        encoding="utf-8", errors="ignore",
                    )
                    for line in html_for_sites.splitlines():
                        s = line.strip()
                        low = s.lower()
                        if(
                            ("fillrect" in low and any(
                                tok in low for tok in (
                                    "bg", "background", "missing", "fallback",
                                    "sprite", "entity", "scene",
                                )
                            ))
                            or "missing" in low
                            or "function drawbg" in low
                        ):
                            likely_sites.append(s[:140])
                        if len(likely_sites) >= 4:
                            break
                except Exception:
                    likely_sites = []
                report["procedural_regression"] = {
                    "referenced_assets": len(referenced_assets),
                    "big_rect_count": big_rect_count,
                    "draw_image_count": draw_image_count,
                    "median_rect_area_px": median_area,
                    "likely_source_sites": likely_sites,
                }
                site_text = ""
                if likely_sites:
                    site_text = (
                        " Likely source site(s): "
                        + " | ".join(likely_sites)
                        + "."
                    )
                report["soft_warnings"].append(
                    f"PROCEDURAL_REGRESSION_SUSPECTED: "
                    f"{len(referenced_assets)} sprite asset(s) are "
                    f"declared and loaded, but the canvas drew "
                    f"{big_rect_count} big rectangles "
                    f"(≥32×32 px; median area ≈ {median_area} px²) "
                    f"while only making {draw_image_count} ctx.drawImage "
                    f"call(s). Big rectangles outnumber sprite draws "
                    f"5:1, which is the signature of entities being "
                    f"rendered as colored placeholders instead of art. "
                    "Replace ctx.fillRect(x, y, w, h) entity draws with "
                    "ctx.drawImage(ASSETS.<name>, x, y, w, h). UI "
                    "elements (HUD bars, borders) are not counted — "
                    "the 32×32 minimum filters them out."
                    + site_text
                )
        # Probe errors are derived from probe_results (entries with
        # ok=False AND non-empty err).
        report["probe_errors"] = [
            f"{p.get('name','?')}: {p.get('err','')[:160]}"
            for p in probe_results
            if not p.get("ok") and p.get("err")
        ]
        report["probe_eval_errors"] = [
            {
                "name": p.get("name", "probe"),
                "expr_preview": (p.get("expr") or "")[:120],
                "error_class": p.get("error_class") or "eval_error",
                "err": (p.get("err") or "")[:200],
            }
            for p in probe_results
            if not p.get("ok") and p.get("kind") == "eval_error"
        ]
        # A1: precompute source slices once so format_report_for_model
        # doesn't re-read the file on every render. file_filter biases the
        # extractor to frames inside the current game file (deep frameworks
        # in CDN scripts won't pollute the report).
        report["path"] = str(path)
        report["crash_source_slices"] = extract_crash_source_slices(
            list(self._page_errors) + list(self._console_errors),
            file_filter=path,
        )

        # Promote the new findings into soft_warnings so they get the same
        # "must fix" treatment as RAF/blank in the report formatter.
        # Important nuance: FROZEN-without-input-change is bad; FROZEN-but-
        # input-causes-change is just "input-driven, not auto-animated" which
        # is normal for many games (turn-based, click counter, tic-tac-toe).
        # We only flag FROZEN when input ALSO doesn't move pixels.
        input_responsive = bool(
            input_test.get("ran") and input_test.get("any_change") is True
        )
        input_dead = bool(
            input_test.get("ran") and input_test.get("any_change") is False
        )
        # Blank canvas + dead keyboard = real failure for 2D games. For
        # three.js/WebGL builds (Doom trace run_05) surface as advisory only —
        # the view may be blank because the camera is not wired yet, not
        # because the harness should hard-block a working iter.
        if (canvas_info and canvas_info.get("blank") is True
                and input_dead and not input_responsive):
            if _is_webgl_or_three_game(_src_html):
                report["warnings"].append(
                    "ADVISORY (non-blocking): 3D/canvas view appears blank "
                    "(uniform pixels) and movement keys did not change the "
                    "rendered view — verify camera position, renderer size, "
                    "and that gameplay state drives the render loop."
                )
            else:
                report["soft_warnings"].append(
                    f"HEURISTIC: canvas pixels are uniform AND keyboard input "
                    f"didn't change anything either — the game is not "
                    f"rendering / not interactive."
                )
        if _threejs_manual_navigation_basis_risk(_src_html):
            report.setdefault("warnings", []).append(
                "3D NAVIGATION (advisory): manual fx=+Math.sin(yaw) movement "
                "with camera.rotation.y may not match three.js camera forward "
                "on world X — use applyQuaternion(camera.quaternion) or "
                "fx=-Math.sin(yaw); see playbook 3d-navigation-modality-invariants."
            )
        _turn_based_board_idle = False
        try:
            _turn_based_board_idle = bool(
                await self._safe_eval(
                    "window.__hasPointerBoardState === true"
                    " && (window.__listenerTypes||{}).pointer > 0"
                )
            )
        except Exception:
            _turn_based_board_idle = False
        if frozen is True and not input_responsive and not _turn_based_board_idle:
            report["soft_warnings"].append(
                "HEURISTIC: canvas drew SOMETHING but did not change between two "
                "samples 1s apart AND no key press changed anything either - "
                "the game is frozen / stuck on one frame."
            )
        elif frozen is True and not input_responsive and _turn_based_board_idle:
            report.setdefault("warnings", []).append(
                "HEURISTIC (non-blocking): turn-based board canvas unchanged "
                "between samples with no keyboard delta — expected while "
                "waiting for pointer clicks."
            )
        # Low-color-diversity check: 1024 sample points across the canvas
        # but only a handful of unique colors → game is barely rendering
        # (typical: WebGL "Three.js stub" with one cube on solid bg, or a
        # 2D canvas drawn once with a flat fill). The model can't game
        # this because the threshold runs harness-side, not as a
        # model-authored probe. Skip the check on tiny canvases (< 256
        # pixels) where low diversity is legitimate, and on the input-
        # responsive case where the canvas may be intentionally minimal
        # before user input (drawing app, tic-tac-toe blank board).
        if (canvas_info
                and canvas_info.get("sampled_colors") is not None
                and canvas_info.get("blank") is False
                and canvas_info.get("raf_ran")
                and canvas_info.get("width", 0) * canvas_info.get("height", 0) >= 256
                and not input_responsive):
            n_colors = int(canvas_info["sampled_colors"])
            if n_colors < 8:
                report["soft_warnings"].append(
                    f"HEURISTIC: only {n_colors} unique colors across 1024 "
                    f"canvas sample points - the scene is trivial (likely a "
                    f"stub/placeholder, not a finished game). Real games "
                    f"render at least 8-16 distinct colors even early in "
                    f"the first frame. Add real sprites, varied geometry, "
                    f"or proper textures."
                )
        if input_dead:
            keys_str = ", ".join(input_test.get("keys_tried", []))
            # Don't double-fire if the page only has a button (no canvas
            # input expected). We check for an interactive element below.
            has_clickable = await self._safe_eval(
                "document.querySelectorAll('button, [onclick]').length > 0"
            )
            # DK trace 20260514_104131 fix: when criteria explicitly
            # mention keyboard / pointer controls, override the
            # "treating as DOM-driven" carve-out — the game promised
            # input and the input is broken, regardless of whether a
            # restart button happens to exist.
            controls_expected = expects_game_controls(criteria or "")
            # Click-primary softening (chess-trace fix 2026-06-22): the
            # harness presses KEYBOARD keys. A chess / board / point-and-
            # click game wires mouse/pointer handlers and no keyboard
            # handler, so the keyboard test correctly finds "no change" —
            # but that is NOT a broken control, it is the wrong modality.
            # When the page registered mouse/pointer listeners, registered
            # NO keyboard listeners, and the criteria don't name keyboard
            # input, downgrade the synthetic probe to a non-gating warning
            # so the false positive can't pin ok=False for the whole
            # session (chess trace iters 1-5). Structural + genre-free;
            # works as a fallback even when the LLM router is unavailable.
            try:
                _ltypes = await self._safe_eval(
                    "window.__listenerTypes || null"
                )
            except Exception:
                _ltypes = None
            _ltypes = _ltypes if isinstance(_ltypes, dict) else {}
            _mouse_listeners = (
                int(_ltypes.get("mouse", 0) or 0)
                + int(_ltypes.get("pointer", 0) or 0)
                + int(_ltypes.get("touch", 0) or 0)
            ) > 0
            _keyboard_listeners = int(_ltypes.get("key", 0) or 0) > 0
            click_primary = (
                _mouse_listeners
                and not _keyboard_listeners
                and not expects_keyboard_controls(criteria or "")
            )
            # Board games with mouse click + optional keyboard cursor fallback
            # (holochess trace run_04 iter 2): still pointer-primary when criteria
            # don't name keyboard — same non-gating treatment as click_primary.
            _pointer_board = False
            try:
                _pointer_board = bool(
                    await self._safe_eval("window.__hasPointerBoardState === true")
                )
            except Exception:
                _pointer_board = False
            pointer_board_primary = (
                _mouse_listeners
                and _pointer_board
                and not expects_keyboard_controls(criteria or "")
            )
            input_modality_pointer = click_primary or pointer_board_primary
            # Item 3, trace build-a-donkey-kong-clone-in-o_20260514_214747:
            # the model's code had window.addEventListener + e.code +
            # e.preventDefault, and report showed `Input listeners:
            # total=5 (win=5)` — listeners WERE registered, RAF ran,
            # canvas was not blank. But the input_test still failed,
            # and the prior message ("Controls are not wired up") was
            # misleading because controls ARE wired. The actual cause
            # is downstream: handler writes to a `keys` object that
            # draw() doesn't read (closure-scope mismatch), or draw()
            # reads from a stale snapshot, or RAF is started before
            # the listener is registered. Branch the message so the
            # model sees a precise diagnostic instead of a wrong one.
            listeners_present = (
                bool(listener_info)
                and listener_info.get("total", 0) > 0
            )
            raf_ran = bool(canvas_info and canvas_info.get("raf_ran"))
            handler_present_but_no_visible_change = (
                listeners_present and raf_ran
                and canvas_info and canvas_info.get("blank") is False
            )
            if input_modality_pointer:
                # Click/pointer-primary game tested with the keyboard —
                # non-gating warning, not a soft_warning / synthetic probe.
                if click_primary:
                    _modality_note = (
                        "mouse/pointer listeners and NO keyboard listeners"
                    )
                    report["input_modality"] = "click_primary"
                else:
                    _modality_note = (
                        "mouse/pointer listeners and a 2D board/grid in "
                        "state (keyboard listeners may be cursor fallback only)"
                    )
                    report["input_modality"] = "pointer_board_primary"
                report["warnings"].append(
                    f"Note: keyboard test pressed {keys_str} with no canvas "
                    f"change, but the page registered {_modality_note} "
                    "(and the criteria don't name keyboard input) — treating "
                    "as click/pointer-primary, not a broken keyboard control."
                )
            elif not has_clickable or controls_expected:
                if handler_present_but_no_visible_change:
                    # Listeners + RAF + non-blank canvas, but keys
                    # don't move pixels: the wiring exists, the bug
                    # is in the data flow between handler and draw().
                    report["soft_warnings"].append(
                        f"HEURISTIC: pressed {keys_str} - "
                        f"{listener_info.get('total', 0)} key listener(s) "
                        "registered on window AND the RAF loop is "
                        "rendering, but pressing keys did not produce "
                        "any visible pixel change. The wiring exists "
                        "but the data flow between your keydown "
                        "handler and your draw() is broken. CHECK: "
                        "(1) your keydown handler writes to the SAME "
                        "`keys` object that updatePlayer/draw() reads "
                        "from (closure-scope mismatch is the #1 cause "
                        "on small models); (2) you use `e.code` (not "
                        "`e.key`) so layout-dependent values like "
                        "'ArrowLeft' actually match your KEYMAP; "
                        "(3) RAF is started AFTER the listener is "
                        "registered so the first frame can read "
                        "state set by held keys; (4) update() is "
                        "actually being called (no early-return on "
                        "state.over=true at startup)."
                    )
                else:
                    # Listeners absent or RAF dead → original message.
                    report["soft_warnings"].append(
                        f"HEURISTIC: pressed {keys_str} - canvas pixels never changed. "
                        "Controls are not wired up (or input handler is broken)."
                    )
                # Synthesize an `input_responsive` probe entry so the
                # failure also surfaces in the per-probe display (model
                # can't rationalize it as "just a warning"). The
                # existing probe-failure gate below promotes it to a
                # second PROBE FAILED soft_warning, which the `ok`
                # formula already catches. Only fires when controls are
                # affirmatively expected — for genuinely DOM-driven
                # pages we still want the lighter signal.
                if controls_expected:
                    if handler_present_but_no_visible_change:
                        probe_err = (
                            f"pressed {keys_str}; "
                            f"{listener_info.get('total', 0)} listener(s) "
                            "registered AND RAF is rendering, but keys "
                            "do not produce visible state change. "
                            "Verify your keydown handler writes to the "
                            "same `keys` object draw() reads (closure-"
                            "scope mismatch), and that update() runs "
                            "(no early-return on game-state flags)."
                        )
                    else:
                        probe_err = (
                            f"pressed {keys_str} and canvas pixels "
                            f"never changed — controls promised in "
                            f"<criteria> but not wired. Confirm the "
                            f"key handler runs and updates state."
                        )
                    probe_results.append({
                        "name": "input_responsive",
                        "expr": (
                            "(harness) keyboard input must produce a "
                            "visible canvas change"
                        ),
                        "ok": False,
                        "err": probe_err,
                        "synthetic": True,
                    })
            else:
                report["warnings"].append(
                    "Note: keyboard test produced no canvas change, but the "
                    "page has clickable elements; treating as DOM-driven."
                )
        # A1 — probe failures gate the test. Until this commit, probes
        # were advisory: the harness reported probe results but `ok`
        # ignored them. So a game where the model's own probes
        # (`player.health > 0`, `monsters.length > 0`) failed would
        # still get `ok=True` and ship as "passed". Now any probe
        # whose .ok is False appends a PROBE FAILED soft_warning,
        # which the existing `ok` formula catches. See games/traces/
        # game-of-doom-a-first-person-sh_20260506_230058.jsonl iter 1
        # for the failure case this fixes.
        for p in (report.get("probes") or []):
            if p.get("ok"):
                continue
            name = p.get("name", "probe")
            expr = (p.get("expr") or "")[:80]
            report["soft_warnings"].append(_format_probe_failure_warning(p))
        # Stop-Losing-To-OneShot todo #2 — criteria/probe coverage gate.
        # The model authors its own <probes>; mid-tier models often write
        # existence-only probes (`!!player`) that pass even when the
        # promised behavior is broken. Flag any <criteria> line whose
        # meaningful words do not appear in any probe's name/expr — the
        # model must add a probe that actually tests it. Genre-free:
        # uses simple word overlap, no semantic parsing.
        # Synthesize a failing probe per uncovered criterion AND its gating
        # soft_warning. Extracted to `_apply_coverage_gap_gate` so the gate
        # is unit-testable without Chromium. The soft_warning append is the
        # fix (battlezone trace 20260622): synthetic coverage_gap probes
        # showed as FAIL but never gated ok=True. Local LLMs ignore
        # "soft_warnings" prose but fix failing probes — the synthetic probe
        # carries name `coverage_gap__<slug>` so the agent's Phase B
        # re-parse can detect closure (model emits a new <probes> block
        # referencing the gap → agent replaces self._probes with the new
        # set). No-op (and ok preserved) when every criterion is covered.
        probe_results = _apply_coverage_gap_gate(
            report, criteria or "", probes or [], probe_results
        )
        # Outside-agent strict compatibility gate for likely three.js pages:
        # run a short second pass under stock file:// Chromium (no relaxed
        # security flags). Keep diagnostics compact so local small models can
        # fix in one iteration.
        strict_runtime: dict[str, Any] = {"checked": False}
        try:
            html_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            html_text = ""
        if _is_threejs_candidate_html(html_text):
            strict_runtime = _run_strict_file_runtime_check(
                path, run_seconds=min(1.4, max(0.9, self._run_seconds / 2.0))
            )
            status = strict_runtime.get("status")
            if status == "fail":
                failure_type = strict_runtime.get("failure_type") or "unknown"
                summary = strict_runtime.get("summary") or "outside-agent strict file:// check failed"
                report["soft_warnings"].append(
                    f"STRICT FILE RUNTIME FAILED [{failure_type}]: {summary}"
                )
                hints = strict_runtime.get("hints") or []
                if hints:
                    report["soft_warnings"].append(f"STRICT FIX HINT: {hints[0]}")
            elif status == "infra_error":
                # Circuit breaker: strict-check infra problems are non-blocking.
                summary = strict_runtime.get("summary") or "unknown harness issue"
                report["warnings"].append(
                    f"Strict outside-agent check unavailable: {summary}"
                )
        report["strict_file_runtime"] = strict_runtime
        # ok must reflect the fresh soft_warnings count after we appended.
        report["ok"] = len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
        return report

    async def _safe_eval(self, js: str):
        """page.evaluate that swallows errors and returns None on failure."""
        try:
            return await self._page.evaluate(js)
        except Exception:
            return None

    # ---- Phase 1.5 — multi-window playtest capture --------------------
    #
    # Generalises _input_smoke_test into a scripted timeline: drive
    # synthetic input across multiple time windows and sample
    # screenshots + game state at requested times. Used by the
    # autonomous self-feedback loop to detect bugs a single-frame
    # critic cannot — entity-stuck, input-axis-mismatch, out-of-bounds.
    #
    # Genre-free by design: no recipe-specific code lives here. Recipes
    # in the root playbook supply their own `input_script` (list of
    # action dicts) and `sample_times_s` (when to capture state); this
    # method just executes them and returns the timeline.

    async def record_playtest(
        self,
        input_script: list[dict],
        sample_times_s: list[float],
        *,
        capture_screenshots: bool = False,
        state_expr: str | None = None,
    ) -> dict[str, Any]:
        """Drive a scripted input timeline and sample state at intervals.

        `input_script` items support:
          {"type": "wait", "ms": 1000}
          {"type": "keydown", "key": "ArrowUp", "duration_ms": 1000}
            — fires keydown, sleeps duration_ms, fires keyup.
          {"type": "press", "key": "Space"}
            — single press event (~50 ms hold internally).
          {"type": "click", "x": 100, "y": 100}
            — mouse click at canvas-relative coords (clamped to viewport).

        `sample_times_s` lists offsets from the start of the script at
        which to capture state. Each sample carries:
          {
            "t_s": float,         # actual elapsed time
            "canvas_hash": str?,  # 32x32 downsample hash, None on failure
            "state": dict?,       # window.state / window.gameState
                                  #   flattened to {dotted: number}
            "screenshot_b64": str?,  # only when capture_screenshots=True
            "custom": Any?,       # `state_expr` evaluated against page
          }

        Returns: {"ok": bool, "samples": [...], "errors": [...],
                  "input_script_replay": <echoed for the trace>}.

        Never raises — caller treats the timeline as advisory and
        gracefully skips findings on incomplete captures.
        """
        import asyncio
        import base64
        import time as _t
        result: dict[str, Any] = {
            "ok": False,
            "samples": [],
            "errors": [],
            "input_script_replay": list(input_script or []),
        }
        if self._page is None:
            result["errors"].append("no page open")
            return result
        if not sample_times_s:
            result["errors"].append("no sample_times_s requested")
            return result

        try:
            await self._page.bring_to_front()
            await self._page.evaluate("if (document.body) document.body.focus();")
        except Exception:
            pass

        # Snapshot helper. State expression mirrors _input_smoke_test's
        # numeric-leaf flatten; we keep it inline here so the recipe
        # contract doesn't depend on private helpers.
        _STATE_JS = (
            "(()=>{const s=window.state||window.gameState||null;"
            "if(!s||typeof s!=='object')return null;"
            "const out={};const stack=[['',s,0]];const MAXD=4;const MAXN=80;"
            "while(stack.length){const[pref,obj,depth]=stack.pop();"
            "if(depth>MAXD)continue;let n=0;"
            "for(const k in obj){if(n++>MAXN)break;let v;try{v=obj[k]}catch(e){continue}"
            "if(typeof v==='number'&&Number.isFinite(v))out[pref?pref+'.'+k:k]=v;"
            "else if(typeof v==='boolean')out[pref?pref+'.'+k:k]=v?1:0;"
            "else if(v&&typeof v==='object'&&!Array.isArray(v))stack.push([pref?pref+'.'+k:k,v,depth+1]);}}"
            "return out;})()"
        )
        _CANVAS_HASH_LITERAL = (
            "(()=>{const c=document.querySelector('canvas');if(!c||!c.width||!c.height)return null;"
            "try{const tmp=document.createElement('canvas');tmp.width=32;tmp.height=32;"
            "const tctx=tmp.getContext('2d',{willReadFrequently:true});"
            "tctx.drawImage(c,0,0,32,32);"
            "const d=tctx.getImageData(0,0,32,32).data;let h=0;"
            "for(let i=0;i<d.length;i+=4){h=((h*131)+(d[i]<<16)+(d[i+1]<<8)+d[i+2])>>>0;}"
            "return h.toString(16);}catch(e){return null}})()"
        )

        async def _sample(at_t: float) -> dict[str, Any]:
            entry: dict[str, Any] = {"t_s": round(at_t, 3)}
            try:
                entry["canvas_hash"] = await self._safe_eval(_CANVAS_HASH_LITERAL)
            except Exception as e:
                entry["canvas_hash"] = None
                entry["errors"] = [f"hash:{e}"]
            try:
                entry["state"] = await self._safe_eval(_STATE_JS)
            except Exception as e:
                entry["state"] = None
                entry.setdefault("errors", []).append(f"state:{e}")
            if state_expr:
                try:
                    entry["custom"] = await self._safe_eval(state_expr)
                except Exception as e:
                    entry["custom"] = None
                    entry.setdefault("errors", []).append(f"custom:{e}")
            if capture_screenshots:
                try:
                    raw = await self._page.screenshot(full_page=False)
                    entry["screenshot_b64"] = base64.b64encode(raw).decode("ascii")
                except Exception as e:
                    entry["screenshot_b64"] = None
                    entry.setdefault("errors", []).append(f"shot:{e}")
            return entry

        # Run the input script and sampler concurrently. We schedule each
        # sample as a sleep + capture task so input + sampling run on the
        # same event loop without stepping on each other.
        started_at = _t.monotonic()

        async def _run_script() -> None:
            for action in input_script or []:
                kind = (action.get("type") or "").lower()
                try:
                    if kind == "wait":
                        await asyncio.sleep((action.get("ms") or 0) / 1000.0)
                    elif kind == "keydown":
                        key = action.get("key") or "ArrowRight"
                        dur = (action.get("duration_ms") or 1000) / 1000.0
                        await self._page.keyboard.down(key)
                        try:
                            await asyncio.sleep(dur)
                        finally:
                            await self._page.keyboard.up(key)
                    elif kind == "press":
                        key = action.get("key") or "Space"
                        await self._page.keyboard.press(key, delay=50)
                    elif kind == "click":
                        x = int(action.get("x") or 0)
                        y = int(action.get("y") or 0)
                        await self._page.mouse.click(x, y)
                except Exception as e:
                    result["errors"].append(f"action {kind}:{e}")

        async def _run_samples() -> None:
            for t in sample_times_s:
                target = float(t)
                # Sleep until at least `target` seconds after start.
                while True:
                    elapsed = _t.monotonic() - started_at
                    if elapsed >= target:
                        break
                    await asyncio.sleep(max(0.01, target - elapsed))
                entry = await _sample(_t.monotonic() - started_at)
                result["samples"].append(entry)

        try:
            await asyncio.gather(_run_script(), _run_samples())
            result["ok"] = True
        except Exception as e:
            result["errors"].append(f"gather:{e}")
        return result

    async def _run_opening_book_recipes(
        self,
        path: Path,
        recipes: list[dict],
        *,
        input_test: dict[str, Any],
        canvas_info: dict[str, Any] | None,
        frozen: bool | None,
        goal: str = "",
        visual_recipe_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute compact verified-memory recipes against the loaded page.

        Recipes are intentionally conservative: they either reuse existing
        harness evidence or inspect source for generated-asset wiring.
        """
        out: list[dict[str, Any]] = []
        if not recipes:
            return out
        try:
            html_text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            html_text = ""
        for rec in recipes[:8]:
            rid = str(rec.get("id") or rec.get("kind") or "recipe")[:80]
            kind = str(rec.get("kind") or "")
            recipe = rec.get("recipe") if isinstance(rec.get("recipe"), dict) else {}
            rtype = str(recipe.get("type") or "")
            check = {"id": rid, "kind": kind, "type": rtype, "ok": True, "hard": False}
            if rtype == "input_delta":
                ok = bool(input_test.get("ran") and input_test.get("any_change") is True)
                check.update({
                    "ok": ok,
                    "hard": True,
                    "err": "" if ok else "expected input to cause state or canvas delta",
                })
            elif rtype == "asset_usage":
                has_assets = "_assets/" in html_text or "ASSET_LIST" in html_text or "const ASSETS" in html_text
                # 2D canvas consumes art via drawImage; WebGL/three.js (which
                # the prompt steers 3D goals toward) NEVER calls drawImage — it
                # binds the image as a texture. Accept either so a real 3D game
                # is not hard-failed for using the correct API.
                uses_draw = "drawImage" in html_text
                uses_texture = any(s in html_text for s in (
                    "CanvasTexture", "TextureLoader", "THREE.Texture",
                    "SpriteMaterial", "new THREE.Sprite", ".map =", ".map=",
                    "texImage2D",
                ))
                ok = (not has_assets) or uses_draw or uses_texture
                check.update({
                    "ok": ok,
                    "hard": has_assets,
                    "err": "" if ok else "generated assets present but neither a drawImage call nor a WebGL texture binding uses them",
                })
            elif rtype == "asset_stats":
                has_assets = "_assets/" in html_text or "ASSET_LIST" in html_text or "const ASSETS" in html_text
                # Fix round: `img.onload` is just as valid a load path as
                # `img.decode()` — requiring decode() flagged correct
                # onload-based loaders (fight trace 20260611_145321).
                ok = (not has_assets) or (
                    "new Image" in html_text
                    and ("decode" in html_text or "onload" in html_text)
                )
                check.update({
                    "ok": ok,
                    "hard": has_assets,
                    "err": "" if ok else "asset loader does not clearly decode generated images (no decode() or onload handler)",
                })
            elif rtype in {"before_mid_after", "event_window"}:
                # Honest signal, not a RAF-in-source rubber-stamp: a responsive
                # action that renders a single HELD pose (static_action, set by
                # the in-hold canvas-hash sampler) is not real animation; a
                # frozen canvas is not either. Otherwise we lack contrary
                # evidence and don't over-coach. (Dead from_image SPRITE frames
                # are gated separately by the near-idle check in agent.py.)
                static_action = (
                    input_test.get("static_action")
                    if isinstance(input_test, dict) else None
                )
                if static_action:
                    ok, err = False, (
                        "action renders a single held pose — no intermediate "
                        "frames between start and end (teleport/held, not animated)"
                    )
                elif frozen is True:
                    ok, err = False, "canvas frozen — no animation evidence"
                else:
                    ok, err = True, ""
                check.update({"ok": ok, "hard": False, "err": err})
            elif rtype == "restart_reset":
                has_reset = any(s in html_text for s in ("resetGame", "function reset", ".reset(", "restart"))
                check.update({
                    "ok": bool(has_reset),
                    "hard": False,
                    "err": "" if has_reset else "no obvious restart/reset hook found",
                })
            elif rtype == "pointclick_puzzle_chain":
                if not pointclick_opening_book_applicable(goal, visual_recipe_id):
                    check.update({
                        "ok": True,
                        "skipped": True,
                        "hard": False,
                        "err": "not a point-and-click goal",
                    })
                    out.append(check)
                    continue
                result = await self._safe_eval("""
(() => {
  const s = window.state || window.gameState || {};
  const root = (window.SCENES || window.scenes || s.SCENES || s.scenes);
  const sceneList = Array.isArray(root) ? root :
    (root && typeof root === 'object' ? Object.values(root) : []);
  const hotspots = sceneList.flatMap(sc => Array.isArray(sc && sc.hotspots) ? sc.hotspots : []);
  const textOf = h => [h.id, h.name, h.label, h.verb, h.action, h.type, h.item, h.requiresItem, h.itemRequired, h.useWith, h.needs]
    .map(v => v == null ? '' : String(v)).join(' ').toLowerCase();
  const hasInventory = Array.isArray(s.inventory);
  const hasSelected = ['selectedItem','activeItem','heldItem','selectedInventory','inventorySelection']
    .some(k => Object.prototype.hasOwnProperty.call(s, k));
  const hasTake = hotspots.some(h => /\\b(take|get|pick\\s*up|pickup)\\b/.test(textOf(h)));
  const hasUse = hotspots.some(h => /\\b(use|dig|shovel|unlock|open|combine)\\b/.test(textOf(h)));
  const hasDialog = !!(s.dialog || s.dialogue || s.message || s.caption || s.statusText);
  const missing = [];
  if (!sceneList.length) missing.push('state.scenes/SCENES');
  if (!hotspots.length) missing.push('scene hotspots[]');
  if (!hasInventory) missing.push('state.inventory[]');
  if (!hasSelected) missing.push('selectedItem/active inventory field');
  if (!(hasTake || hasUse || hasDialog)) missing.push('take/use/dig hotspot or dialog state');
  return { ok: missing.length === 0, missing };
})()
""")
                ok = bool(isinstance(result, dict) and result.get("ok"))
                missing = (
                    ", ".join(result.get("missing") or [])
                    if isinstance(result, dict) else "runtime state probe failed"
                )
                check.update({
                    "ok": ok,
                    "hard": True,
                    "err": "" if ok else (
                        "point-and-click puzzle chain not exposed: " + missing
                    ),
                })
            elif rtype == "state_expr":
                # Generic ADVISORY runtime check: evaluate a self-contained JS
                # boolean expression in the page. Never hard-gates (hard=False)
                # so a genre-free opening-book playtest cannot permanently block
                # shipping — it only surfaces a finding the model can act on
                # (e.g. state-exposed-on-window, hud-score-visible). A page eval
                # error is treated as "skipped" (advisory), not a failure.
                expr = str(recipe.get("expr") or "").strip()
                if not expr:
                    check.update({"ok": True, "skipped": True, "err": "no expr"})
                else:
                    res = await self._safe_eval(
                        "(() => { try { return Boolean(" + expr + "); } "
                        "catch (e) { return null; } })()"
                    )
                    if res is None:
                        check.update({
                            "ok": True, "skipped": True,
                            "err": "state_expr eval error (advisory)",
                        })
                    else:
                        check.update({
                            "ok": bool(res),
                            "hard": False,
                            "err": "" if res else (
                                str(recipe.get("err") or "")
                                or f"expected `{expr}` to be truthy"
                            ),
                        })
            else:
                check.update({"ok": True, "skipped": True, "err": "unsupported recipe type"})
            out.append(check)
        return out

    async def _run_probe(self, expr: str) -> tuple[bool, str, str | None]:
        """Run one model-proposed probe expression in the page; return
        (ok, err, err_kind). Wraps in (() => Boolean(EXPR))() so the model
        can write either a boolean expression or an IIFE.
        """
        expr = _normalize_probe_expr(expr)
        # QTE-gate fix (serial10 game 9): some auto_probes are async (the
        # QTE window-gating probe dispatches a key then awaits a frame to
        # see if it wrongly registered success). The old sync wrapper did
        # `Boolean(promise)` → always true, so the async gate silently
        # no-op'd. Await the expression before coercing: `await (value)`
        # is a no-op for sync boolean probes, so this is universal and
        # cannot change any existing sync probe's result.
        wrapped = (
            "(async () => { try { return Boolean(await (" + expr + ")); } "
            "catch (e) { return { __probe_err: String(e && e.message || e) }; } })()"
        )
        try:
            res = await self._page.evaluate(wrapped)
            if isinstance(res, dict) and res.get("__probe_err"):
                err = str(res["__probe_err"])[:200]
                return False, err, _classify_probe_eval_error(err)
            return bool(res), "", None
        except Exception as e:
            err = str(e)[:200]
            return False, err, _classify_probe_eval_error(err)

    async def _input_smoke_test(self, criteria: str | None = None) -> dict[str, Any]:
        """Hold each test key for a few frames; report whether the canvas changed.

        Two big differences vs the original 9-pixel version:

          1. We HOLD each key (down/up) for ~250ms instead of press(). Held-key
             games like thrust-controlled ships only move while a key is
             actively down; press() releases instantly and the next sample
             catches the post-release frame.

          2. We use a ~32x32 downsampled hash of the FULL canvas instead of
             nine corner pixels, so slowly-moving objects in the middle of the
             playfield register as a change.

        Stop-Losing-To-OneShot todo #2 — ambient baseline + gameState delta.
        Games with continuous animation (asteroids drifting, particles,
        scrolling parallax) made the old before/after canvas-hash test
        ALWAYS report change → false-positive "TEST OK" on broken
        controls. We now:
          (a) take an ambient baseline (two hashes 250ms apart, no key
              held) and capture window.gameState if exposed;
          (b) for each held key, prefer a `gameState` numeric-leaf delta
              that did NOT change in ambient (input-attributable; works
              even when the canvas is also drifting), and only fall back
              to canvas-hash change when ambient was stable.
        Generic and behavioral — works for any HTML/JS game.
        """
        import asyncio

        has_canvas = await self._safe_eval(
            "!!document.querySelector('canvas')"
        )
        if not has_canvas:
            return {"ran": False, "reason": "no canvas"}

        # Movement defaults (always tried), plus the actual action keys the
        # model declared in <criteria> (KeyF punch, KeyG kick, KeyZ ability,
        # …) so attack/ability animations actually fire and get captured as an
        # action frame. Genre-free: we press the input tokens the spec names.
        default_keys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "KeyW", "KeyA", "KeyS", "KeyD"]
        keys = list(dict.fromkeys(default_keys + _parse_action_keys(criteria or "")))[:16]
        if not keys:
            keys = default_keys
        # Phase 0: criteria-declared non-combat keys (Space to pause, Enter to
        # start wave, …) must not be pressed during movement smoke — they toggle
        # flow state and cause false input_responsive failures (Fieldrunners /
        # snake batch_parallel 20260627).
        _non_combat = _non_combat_action_keys(criteria or "")
        if _non_combat:
            keys = [k for k in keys if k not in _non_combat]
        tried: list[str] = []
        any_change = False
        first_responsive_key: str | None = None

        try:
            await self._page.bring_to_front()
            await self._page.evaluate("if (document.body) document.body.focus();")
        except Exception:
            pass

        # Unpause games that show PAUSED in the HUD before the key sweep — many
        # grid/arcade builds start paused or use Space as pause; movement keys
        # do nothing while paused (snake batch_parallel 20260627).
        try:
            body_sample = await self._safe_eval(
                "(()=>{const t=(document.body?.innerText||'').slice(0,240);"
                "return typeof t==='string'?t:'';})()"
            )
            if isinstance(body_sample, str) and re.search(r"\bPAUSED\b", body_sample, re.I):
                await self._page.keyboard.press("Space")
                await asyncio.sleep(0.2)
        except Exception:
            pass

        # The state-flattening snapshot JS lives at module level
        # (`_GAMESTATE_SNAPSHOT_JS`) since 2026-06-10 — it is shared with
        # the state-timeline sampler in `load_and_test` (capability-round
        # item 4). Same flattening, same depth/fanout caps.

        def _gs_changed_leaves(before: Any, after: Any) -> set[str]:
            """Set of leaf paths whose value differs (epsilon for floats)
            or that appeared/disappeared between the two snapshots."""
            if not isinstance(before, dict) or not isinstance(after, dict):
                return set()
            EPS = 1e-3
            changed: set[str] = set()
            for k, b in before.items():
                a = after.get(k)
                if a is None and k not in after:
                    changed.add(k)
                    continue
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > EPS:
                        changed.add(k)
            for k in after:
                if k not in before:
                    changed.add(k)
            return changed

        # ---- ambient baseline (no key held) -------------------------------
        ambient_a = await self._safe_eval(_CANVAS_HASH_JS)
        ambient_gs_a = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
        await asyncio.sleep(0.25)
        ambient_b = await self._safe_eval(_CANVAS_HASH_JS)
        ambient_gs_b = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
        ambient_canvas_changed = bool(
            ambient_a is not None and ambient_b is not None and ambient_a != ambient_b
        )
        ambient_gs_changes = _gs_changed_leaves(ambient_gs_a, ambient_gs_b)
        has_gamestate = ambient_gs_a is not None
        # Does the game even HAVE a spatial entity (position fields)? Only then
        # is "a movement key didn't change any position" meaningful — guards the
        # stuck-player check against menu/quiz games where arrows aren't motion.
        # Exclude position leaves nested under an array index (e.g.
        # `rooms.0.from.x`): those are static layout/config CONSTANTS, not a
        # movable player's position, and they read as position only by name.
        # Same parts[:-1]-digit convention `_input_evidence_is_plausible` uses
        # to discount array-indexed leaves. Trace pin: a Dragon's-Lair QTE's
        # `state.rooms[].from.x/to.y` hazard coords falsely satisfied this,
        # making the harness think a movable player existed and tripping
        # PLAYER-STUCK (phase-a-requirement-your-plann_20260615_121048).
        def _is_movable_position_leaf(leaf: str) -> bool:
            if _is_board_navigation_leaf(leaf):
                return True
            if not _is_position_leaf(leaf):
                return False
            parts = [p for p in str(leaf or "").split(".") if p]
            return not any(p.isdigit() for p in parts[:-1])
        has_position_state = isinstance(ambient_gs_a, dict) and any(
            _is_movable_position_leaf(leaf) for leaf in ambient_gs_a
        )

        # ---- action-frame capture ----------------------------------------
        # Genre-free: capture ONE screenshot at the moment a held key is
        # producing its largest canvas change (a transient action animation
        # — punch, jump, ability — appears while held then reverts). The
        # ambient floor below ensures a continuously-animating game's
        # baseline drift never wins. The captured frame is kept only if its
        # winning key turns out to be input-attributable (in
        # responsive_evidence); otherwise it is discarded after the loop.
        ambient_floor = _canvas_hash_distance(ambient_a, ambient_b) or 0.0
        best_action_delta = 0.0
        best_action_key: str | None = None
        action_frame_png: bytes | None = None
        # Per-key action-frame candidates: key -> (held_dist, screenshot bytes),
        # captured during the hold for any key that beat the ambient floor. The
        # WINNER is chosen AFTER the loop among keys that are both responsive
        # and TRANSIENT (revert after release), so a screen-wiping restart key
        # never becomes the "action" frame.
        action_candidates: dict[str, tuple[float, bytes]] = {}
        per_key_release_dist: dict[str, float | None] = {}
        # Per-key "in-hold motion": max pairwise canvas-hash distance across
        # frames sampled WHILE the key is held. A real animation keeps
        # changing (>0); a single static pose held for the move stays ~0.
        # Used to flag attacks/abilities that render as a frozen pose.
        per_key_hold_motion: dict[str, float] = {}

        # Track WHICH state fields move on WHICH key — names the
        # exact wiring path that works, so the report can say
        # "ArrowRight → state.player.x changed" instead of just
        # "input test passed." When nothing moves we get the dual:
        # "ArrowRight: zero numeric fields on window.state changed."
        responsive_evidence: dict[str, list[str]] = {}
        # "Player moved" vs "stuck": did a movement key change a POSITION leaf
        # (real movement) or only a direction/flag (input registered, no move)?
        movement_position_changed = False
        movement_registered_without_move = False
        # Control-recovery re-test (fix-round item 1): remember the first
        # movement key that moved a POSITION leaf and which leaves it moved,
        # so we can re-dispatch it AFTER the full key sweep (during which the
        # game's own combat lands hits / enters stun states) and verify the
        # player can still move.
        recovery_key: str | None = None
        recovery_leaves: list[str] = []

        # Draw-call state snapshot — distinct drawImage SOURCES seen so far,
        # plus code-draw counts (fillRect + stroke/line). Used to tell a REAL
        # sprite action (a new sprite src appears during the hold) from a FAKE
        # one (the model scribbles a limb with lines/rects but draws no new
        # sprite). See ACTION_DRAWN_NOT_SPRITED.
        _DRAW_STATE_JS = (
            "(()=>{const di=window.__drawImageEvents||[];"
            "const srcs={};for(const e of di){if(e&&e.src)srcs[e.src]=1;}"
            "return {nSrc:Object.keys(srcs).length,"
            "nFill:(window.__fillRectEvents||[]).length,"
            "nStroke:(window.__strokeEvents&&window.__strokeEvents.n)||0};})()"
        )
        per_key_fake_action: dict[str, dict] = {}

        for k in keys:
            before = await self._safe_eval(_CANVAS_HASH_JS)
            before_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
            draw_before = await self._safe_eval(_DRAW_STATE_JS)
            if before is None:
                return {"ran": False, "reason": "canvas not sampleable", "keys_tried": tried}
            try:
                await self._page.keyboard.down(k)
                # Sample the canvas 3× across the ~250ms hold (same total wall
                # time). The last sample is `after_held` (preserves prior
                # semantics); the max pairwise distance among the 3 is the
                # "in-hold motion" — whether the rendered scene keeps changing
                # WHILE the key is held (real animation) vs holds one pose.
                hold_hashes: list[str | None] = []
                for _ in range(3):
                    await asyncio.sleep(0.083)
                    hold_hashes.append(await self._safe_eval(_CANVAS_HASH_JS))
                after_held = hold_hashes[-1]
                after_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
                _hold_pairs = [
                    _canvas_hash_distance(hold_hashes[0], hold_hashes[1]),
                    _canvas_hash_distance(hold_hashes[0], hold_hashes[2]),
                    _canvas_hash_distance(hold_hashes[1], hold_hashes[2]),
                ]
                _hold_pairs = [d for d in _hold_pairs if d is not None]
                if _hold_pairs:
                    per_key_hold_motion[k] = max(_hold_pairs)
                # Action-frame capture: while the key is STILL down, grab a
                # screenshot for any key whose held-frame change beats the
                # ambient floor. The winner is selected after the loop
                # (responsive AND transient), so restart/persistent keys don't
                # win just by having the biggest delta.
                held_dist = _canvas_hash_distance(before, after_held)
                if (
                    held_dist is not None
                    and held_dist > ambient_floor
                    and len(action_candidates) < _ACTION_FRAME_KEYCAP
                ):
                    try:
                        _png = await self._page.screenshot(full_page=False)
                        action_candidates[k] = (held_dist, _png)
                    except Exception:
                        pass
                # Fake-action detection: this key visibly changed the canvas
                # (held_dist > floor) — did it draw a NEW sprite source, or just
                # code-draw (fillRect/lines) over the existing art? A model that
                # fakes a kick by drawing a limb with ctx.lineTo/fillRect adds NO
                # new drawImage src. Recorded per key; the gating decision (only
                # when the game actually uses sprites) is made in load_and_test.
                if held_dist is not None and held_dist > ambient_floor:
                    draw_after = await self._safe_eval(_DRAW_STATE_JS)
                    if isinstance(draw_before, dict) and isinstance(draw_after, dict):
                        new_src = int(draw_after.get("nSrc", 0)) > int(draw_before.get("nSrc", 0))
                        fill_delta = int(draw_after.get("nFill", 0)) - int(draw_before.get("nFill", 0))
                        stroke_delta = int(draw_after.get("nStroke", 0)) - int(draw_before.get("nStroke", 0))
                        per_key_fake_action[k] = {
                            "new_sprite_src": bool(new_src),
                            "code_draw_delta": int(fill_delta + stroke_delta),
                            # stroke/arc/lineTo specifically — the signature of a
                            # code-drawn limb or "attack effect" line+ball, as
                            # opposed to background fillRects. Per ~3 hold frames.
                            "stroke_delta": int(stroke_delta),
                        }
                await self._page.keyboard.up(k)
            except Exception:
                continue
            tried.append(k)
            # Wait one more frame for any post-release tween / momentum.
            await asyncio.sleep(0.05)
            after_release = await self._safe_eval(_CANVAS_HASH_JS)
            # Residual change after release — small for a transient action that
            # reverts to idle, large for a persistent change (restart/walk).
            per_key_release_dist[k] = _canvas_hash_distance(before, after_release)

            # Decide responsiveness with the strongest available signal.
            # (1) state input-only delta: leaves that changed during
            # the held window but did NOT change during the ambient
            # window are attributable to the key press.
            input_only_leaves: set[str] = set()
            if has_gamestate:
                held_changes = _gs_changed_leaves(before_gs, after_gs)
                input_only_leaves = {
                    leaf for leaf in (held_changes - ambient_gs_changes)
                    if _input_evidence_is_plausible(leaf)
                }
            # (2) canvas-hash fallback: only credit when ambient was
            # stable (so the held-window change was input-induced).
            canvas_input_changed = (
                (after_held is not None and after_held != before)
                or (after_release is not None and after_release != before)
            ) and not ambient_canvas_changed

            # Movement-vs-stuck tracking: on a MOVEMENT key (arrows/WASD), did
            # any POSITION leaf change (player actually moved) or only a
            # direction/flag leaf (input registered but entity stuck)?
            if k in _MOVEMENT_KEYS and input_only_leaves:
                pos_leaves = [
                    leaf for leaf in input_only_leaves
                    if _is_effective_movement_leaf(leaf)
                ]
                if pos_leaves:
                    movement_position_changed = True
                    # Prefer lateral (player.x) movers for the recovery
                    # re-test — vertical keys (jump) still move y while
                    # a hit-stun lock freezes x; ArrowUp would false-pass.
                    x_leaves = sorted(l for l in pos_leaves if l.endswith(".x"))
                    candidate = x_leaves or sorted(pos_leaves)
                    cur_has_x = any(
                        l.endswith(".x") for l in recovery_leaves
                    )
                    if recovery_key is None or (x_leaves and not cur_has_x):
                        recovery_key = k
                        recovery_leaves = candidate[:5]
                elif any(
                    not _is_effective_movement_leaf(leaf)
                    for leaf in input_only_leaves
                ):
                    movement_registered_without_move = True

            if input_only_leaves:
                # Sort + cap so the report stays bounded.
                responsive_evidence[k] = sorted(input_only_leaves)[:5]
                any_change = True
                if first_responsive_key is None:
                    first_responsive_key = k
                # Don't break — collect evidence for ALL keys that
                # work so the report can name which control wires
                # which field. Cheap; ~250ms per key, bounded.
            elif canvas_input_changed:
                responsive_evidence[k] = ["<canvas-pixel-change>"]
                any_change = True
                if first_responsive_key is None:
                    first_responsive_key = k

        # Select the action frame: among captured candidates, keep only keys
        # that are (a) input-attributable (in responsive_evidence) and (b)
        # TRANSIENT — the canvas largely reverted after release. Pick the
        # highest in-hold delta among those. This excludes restart/pause/menu
        # keys (whose effect persists) and walk (moves to a lasting position),
        # so the static-pose gate evaluates the real attack/ability. Degrades
        # cleanly to no-action-frame when nothing qualifies.
        for _k, (_hd, _png) in action_candidates.items():
            if _k not in responsive_evidence:
                continue
            _rel = per_key_release_dist.get(_k)
            if _rel is None:
                continue  # can't prove it reverted → exclude
            if _rel >= _hd * _ACTION_TRANSIENT_MAX_RATIO:
                continue  # persistent change (restart/walk) → not an action
            if _hd > best_action_delta:
                best_action_delta = _hd
                best_action_key = _k
                action_frame_png = _png

        # Animation-liveness: if the winning action key is responsive but the
        # scene barely changed WHILE it was held (a single static pose) — and
        # yet the canvas IS animating elsewhere (so it's not a paused/static
        # game) — the action renders as a frozen pose, not an animation.
        # Genre-free: no notion of punch/jump; just "did the input's rendered
        # result keep moving while held, given the game is otherwise live."
        static_action: dict[str, Any] | None = None
        if best_action_key is not None:
            _m = per_key_hold_motion.get(best_action_key)
            if (
                _m is not None
                and _m < _STATIC_POSE_MAX_INHOLD
                and ambient_canvas_changed
            ):
                static_action = {"key": best_action_key, "delta": round(_m, 4)}

        # ---- control-recovery re-test (fix-round item 1) -------------------
        # The key sweep above reliably makes the game's combat land hits /
        # enter stun states. Re-dispatch the first movement key that moved a
        # position leaf EARLY in this run and check the SAME leaves move
        # again. One retry after a grace wait so legitimate brief hit-stun
        # never false-positives. Catches the permanent stun-lock: an expiry
        # timer placed below an `if (action === 'hit') return;` guard.
        control_not_recovered: dict[str, Any] | None = None
        _skip_cnr = _recovery_is_physics_ball_only(
            recovery_leaves, criteria or "",
        )
        if (
            recovery_key is not None
            and recovery_leaves
            and has_position_state
            and not _skip_cnr
        ):

            async def _induce_combat_before_recovery() -> None:
                """Give declared attack keys + held movement time to land hits.

                The recovery re-test only catches stun-lock AFTER gameplay —
                a single quick key sweep often never triggers a projectile
                hit (regression fixture build-a-single-screen-2d-fight_20260610).
                Walk toward engagement on a lateral key that already moved
                player.x (if any), tap declared attack keys, then wait for
                projectiles / melee to resolve. Genre-free."""
                action_keys = [
                    k for k in _parse_action_keys(criteria or "")
                    if k not in _MOVEMENT_KEYS and k not in _RESTART_KEYS
                ][:4]
                if not action_keys:
                    return
                # Prefer a lateral movement key that already moved player.x —
                # closing distance is what makes ranged/melee hits land.
                engage_key = recovery_key
                for k in ("ArrowRight", "ArrowLeft", "KeyD", "KeyA"):
                    ev = responsive_evidence.get(k) or []
                    if any(_is_position_leaf(leaf) and leaf.endswith(".x") for leaf in ev):
                        engage_key = k
                        break
                try:
                    await self._page.keyboard.down(engage_key)
                    # ~2.5s walk + attack taps while closing on the opponent.
                    for _ in range(6):
                        ak = action_keys[_ % len(action_keys)]
                        await self._page.keyboard.down(ak)
                        await asyncio.sleep(0.12)
                        try:
                            await self._page.keyboard.up(ak)
                        except Exception:
                            pass
                        await asyncio.sleep(0.3)
                    await asyncio.sleep(1.0)  # projectiles travel + hit-stun lands
                finally:
                    try:
                        await self._page.keyboard.up(engage_key)
                    except Exception:
                        pass
                    for ak in action_keys:
                        try:
                            await self._page.keyboard.up(ak)
                        except Exception:
                            pass

            async def _recovery_leaves_move_again() -> bool:
                b_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
                try:
                    await self._page.keyboard.down(recovery_key)
                    await asyncio.sleep(0.4)
                finally:
                    try:
                        await self._page.keyboard.up(recovery_key)
                    except Exception:
                        pass
                a_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
                moved = _gs_changed_leaves(b_gs, a_gs)
                if any(leaf in moved for leaf in recovery_leaves):
                    return True
                # FPS yaw fix (trace build-a-first-person-3d-shoote_20260611_
                # 163325 iter 6): in first-person games W moves RELATIVE TO
                # FACING — after the key sweep rotated the view, W changes
                # playerPos.z instead of the originally-recorded playerPos.x,
                # and the original-leaf-only check reported "frozen" on a
                # healthy game. Accept ANY position-leaf change as recovery;
                # the true stun-lock family still trips (nothing moves).
                return any(_is_position_leaf(leaf) for leaf in moved)

            try:
                await _induce_combat_before_recovery()
                # Game-over after combat is expected — movement stops when
                # lives hit 0 (Star Wars trace: 6/6 probes but CNR soft_warn).
                _game_over_after_combat = await self._safe_eval(
                    "(() => {"
                    "const s = window.state || window.gameState;"
                    "if (!s) return false;"
                    "return !!(s.over || s.lives === 0);"
                    "})()"
                )
                if not _game_over_after_combat:
                    recheck_moved = await _recovery_leaves_move_again()
                    retry_moved: bool | None = None
                    if not recheck_moved:
                        await asyncio.sleep(0.5)  # let a legit hit-stun timer expire
                        retry_moved = await _recovery_leaves_move_again()
                    if control_not_recovered_verdict(
                        has_position_state=has_position_state,
                        moved_early=True,
                        recheck_moved=recheck_moved,
                        retry_moved=retry_moved,
                    ):
                        control_not_recovered = {
                            "key": recovery_key,
                            "leaves": recovery_leaves,
                        }
            except Exception:
                pass  # browser hiccup — never block the report on the re-test

        # Concise summary line for the report. Two shapes:
        #   PASS — "ArrowRight→[player.x, player.facing], Space→[bullets.length]"
        #   FAIL — "had window.state but zero fields moved across [keys]"
        if any_change:
            parts = []
            for k, leaves in responsive_evidence.items():
                parts.append(f"{k}→[{', '.join(leaves)}]")
            summary = "; ".join(parts[:4])
        elif has_gamestate:
            summary = (
                f"window.state IS exposed but zero numeric fields "
                f"changed across {tried} — the listeners may be wired "
                f"but the data flow into state is broken."
            )
        else:
            summary = (
                f"no game state global exposed (tried "
                f"window.state/.gameState/.game) AND canvas pixels did "
                f"not change for {tried}."
            )

        return {
            "ran": True,
            "any_change": any_change,
            "keys_tried": tried,
            "first_responsive_key": first_responsive_key,
            "responsive_evidence": responsive_evidence,
            "summary": summary,
            "ambient_canvas_motion": ambient_canvas_changed,
            "ambient_gs_motion": bool(ambient_gs_changes),
            "had_gamestate": has_gamestate,
            # Action frame (peak input-attributable transient). Raw bytes are
            # internal — load_and_test writes them to disk and pops the key
            # before assembling the report JSON.
            "action_frame_png_bytes": action_frame_png,
            "action_key": best_action_key,
            "action_delta": round(best_action_delta, 4),
            # ALL per-action frames (one PNG per key that visibly changed the
            # canvas), so the trace has a debuggable image per named action —
            # not just the single peak one. load_and_test saves each to disk
            # and pops this key before the report JSON is assembled.
            "action_frames_png_bytes": {k: png for k, (_d, png) in action_candidates.items()},
            # Per-key fake-action signal: {key: {new_sprite_src, code_draw_delta}}
            # for keys that changed the canvas. Used (only when the game uses
            # sprites) to flag a code-drawn limb pretending to be a sprite action.
            "fake_actions": per_key_fake_action,
            # Animation-liveness verdict: set when the responsive action key
            # renders a single held pose (not animated). None otherwise.
            "static_action": static_action,
            # In-hold canvas motion per key (diagnostic / trace observability).
            "hold_motion": {k: round(v, 4) for k, v in per_key_hold_motion.items()},
            # Stuck-player verdict: a movement key registered input (set a
            # direction/flag) but NO position leaf ever changed → the player is
            # stuck (spawned in a wall / collision blocks every move). True only
            # when nothing actually moved the player's position.
            "input_registered_without_move": bool(
                movement_registered_without_move
                and not movement_position_changed
                and has_position_state
            ),
            # Control-recovery verdict (fix-round item 1): {key, leaves} when a
            # previously-responsive movement key no longer moves the player's
            # position after gameplay (retry included). None when recovered or
            # not applicable.
            "control_not_recovered": control_not_recovered,
        }

    async def open_url(self, url: str) -> None:
        """Navigate the visible Chromium window to an arbitrary URL."""
        if self._page is None:
            return
        try:
            await self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )
        except Exception:
            # Best-effort; never let navigation crash the TUI.
            pass

    async def show_status(self, title: str, message: str = "") -> None:
        """Replace whatever the browser is currently displaying with a small
        status page. Used between sessions so a failed new session can't
        masquerade as the previous successful one (the browser otherwise
        keeps the last-loaded game on screen).
        """
        if self._page is None:
            return
        # Inline HTML via data: URL — no temp file needed.
        from html import escape
        from urllib.parse import quote
        body = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{escape(title)}</title>"
            "<style>html,body{margin:0;height:100%;background:#0b1020;"
            "color:#e7ecff;font:16px/1.4 system-ui,sans-serif;}"
            "div{position:fixed;inset:0;display:grid;place-items:center;"
            "text-align:center;padding:20px;}"
            "h1{font-size:22px;color:#79a;margin:0 0 12px;}"
            "p{opacity:.75;max-width:600px;}</style></head>"
            f"<body><div><div><h1>{escape(title)}</h1>"
            f"<p>{escape(message)}</p></div></div></body></html>"
        )
        try:
            await self._page.goto("data:text/html;charset=utf-8," + quote(body),
                                  wait_until="load", timeout=5_000)
        except Exception:
            # Best-effort; never let a status update crash the TUI.
            pass

    async def close(self) -> None:
        """Tear down. Safe to call multiple times."""
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            if self._pw is not None:
                await self._pw.stop()
                self._pw = None
