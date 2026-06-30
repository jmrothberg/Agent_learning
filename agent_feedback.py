"""Feedback routing extracted from agent.py for readability.

Classifiers, scope locks, blocker-first deferral, LLM router, and
user-feedback injection. Moved VERBATIM from `GameAgent` (no behavior
change). `GameAgent` inherits this mixin; every `self.*` reference
resolves unchanged through normal MRO lookup.
"""

from __future__ import annotations


import asyncio
import json
import re
from typing import Any

from agent_helpers import _ASSETS_OPEN_RE, _SOUNDS_OPEN_RE, _is_degenerate_baseline
from patches import extract_patches


# Centipede trace 20260512_180020: user typed
#   "only change the centipiede_tail no other asset or code,
#    just that one asset no changes to the code"
# and the model replied "I can't generate new image assets in this
# environment - I can only modify the HTML file" — then rewrote a
# drawSprite() call into procedural ctx.* code (regression).
# The agent already supports mid-session asset re-render
# (_maybe_generate_assets_and_sounds); the model just didn't know it.
# These detectors light up the right path when art-change feedback
# arrives: inject a directive pointing at <assets>, and DON'T arm the
# one-shot <html_file> rewrite exemption when the user explicitly
# locked the code (that exemption fueled the second-attempt regression
# where the model "fixed" the asset by clobbering the sprite call).
_CODE_LOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "no changes to the code", "no change to code"
    re.compile(r"\bno\s+changes?\s+(?:to\s+)?(?:the\s+)?code\b", re.I),
    # "no code changes"
    re.compile(r"\bno\s+code\s+(?:change|edit|modification)s?\b", re.I),
    # "no other code", "no other asset or code"
    re.compile(r"\bno\s+other\s+\w+(?:\s+or\s+code)?\b.*\bcode\b", re.I),
    # "don't change/touch/modify the code", "do not change code",
    # AND "do not change the <noun> code" (e.g. "the game code", "the
    # player code", "the AI code", "the physics code"). Allowing up to
    # two intervening words covers natural-English phrasings like
    # "do not change the chess game code" without firing on neutral
    # text — _feedback_is_strict_scope still requires the explicit
    # forbid-verb so loosening the noun slot can't false-positive.
    re.compile(
        r"\b(?:don['’]?t|do\s+not)\s+(?:change|touch|modify|edit)"
        r"\s+(?:the\s+|any\s+)?(?:\w+\s+){0,2}code\b",
        re.I,
    ),
    # "without changing/touching/modifying the code"
    re.compile(
        r"\bwithout\s+(?:changing|touching|modifying|editing)\s+"
        r"(?:the\s+)?code\b",
        re.I,
    ),
    # "only/just (the/that/this) (one) asset|sprite|image|art|png"
    re.compile(
        r"\b(?:only|just)\s+(?:the\s+|that\s+|this\s+)?(?:one\s+)?"
        r"(?:asset|sprite|image|art|png|graphic|picture|icon)s?\b",
        re.I,
    ),
    # Trace 20260514_175012 fix: user said "this is trivial no other
    # changes there" but the existing patterns all required the
    # literal word "code" or specific media nouns. None matched, the
    # rewrite exemption armed, and the model emitted a full <html_file>
    # rewrite instead of the minimal swap the user asked for. The
    # patterns below catch common minimal-scope phrasings without
    # requiring "code" or a media noun.
    #
    # "no other changes" / "no other changes there/please/needed"
    re.compile(r"\bno\s+other\s+changes?\b", re.I),
    # "this is trivial" / "trivial change|fix|swap" — minimal-scope intent
    re.compile(
        r"\btrivial\b.*\b(?:change|fix|swap|edit|update|tweak)\b", re.I,
    ),
    re.compile(
        r"\b(?:change|fix|swap|edit|update|tweak)\b.*\btrivial\b", re.I,
    ),
    # "just swap|switch|rename|move|flip|toggle" — narrow-verb scope-lock
    re.compile(
        r"\bjust\s+(?:swap|switch|rename|move|flip|toggle|rewire|"
        r"rebind|reassign)\b",
        re.I,
    ),
    # "only swap|switch|rename|move|flip|toggle"
    re.compile(
        r"\bonly\s+(?:swap|switch|rename|move|flip|toggle|rewire|"
        r"rebind|reassign)\b",
        re.I,
    ),
    # "leave the rest (alone|as-is)" — explicit out-of-scope signal
    re.compile(r"\bleave\s+(?:the\s+)?rest\b", re.I),
    # "nothing else (changes|to change|to fix)"
    re.compile(r"\bnothing\s+else\b", re.I),
)

_MEDIA_VERBS: tuple[str, ...] = (
    "change", "swap", "replace", "redraw", "regenerate", "remake",
    "redo", "redesign", "update", "make", "render", "rerender",
    "rerecord", "rebuild", "add", "missing",
    # Added 2026-05-15: user said "fix the images" / "fix the
    # animations" and got a CODE rewrite instead of sprite regen.
    # "fix" alone is too generic to route on, but the gate also
    # requires an art noun (see _feedback_is_art_change), so
    # "fix the keyboard handler" still correctly stays in code-fix
    # mode (no art noun → no MEDIA-CHANGE).
    "fix", "improve",
    # Phase 0.11 — descriptive verbs that pair with an art noun to
    # signal a STYLE rebrand without using an imperative verb. Real
    # user examples that previously failed classification:
    #   "all the images need to be animated as fantasy monsters"
    #   "i want all new graphics so the pieces look like monsters"
    #   "the sprites should look more like X"
    # The art-noun + verb gate keeps these from false-firing on
    # behavior asks ("the player should jump higher" has no art noun
    # → still classified as behavior, not art).
    "look", "looks", "looking",
    "want", "wants", "wanting",
    "need", "needs", "needing",
    "should",
    "feel", "feels", "feeling",
)
_ART_NOUNS: tuple[str, ...] = (
    "asset", "assets", "sprite", "sprites", "image", "images",
    "png", "art", "graphic", "graphics", "picture", "pictures",
    "icon", "icons", "appearance",
    # Added 2026-05-15: in user vocabulary, "animation" / "animations"
    # = "the moving picture on screen" = the sprite. Missing this
    # caused MEDIA-CHANGE to never fire for "replace the animations"
    # / "fix the annimations", routing the request to <patch> (code
    # change) when the user explicitly wanted sprite regeneration.
    # "annimation" with the extra 'n' is the user's persistent typo —
    # listed so detection is robust to that misspelling. "frame" /
    # "frames" cover "redo the run frames" / "the walk frames look
    # wrong" phrasings; behavior_bug still suppresses MEDIA-CHANGE
    # for phrases like "frames are stuttering" because that hits the
    # bug-complaint regex first.
    "animation", "animations",
    "annimation", "annimations",
    "anim", "anims",
    "frame", "frames",
    "spritesheet", "spritesheets",
)
_SOUND_NOUNS: tuple[str, ...] = (
    "sound", "sounds", "audio", "sfx", "music", "song",
    "beep", "chime", "sample", "clip", "track", "ogg", "tune",
    "soundtrack",
)


def _feedback_vocab(category: str, base: tuple[str, ...]) -> tuple[str, ...]:
    """Phase 4 (4A): union the commented code vocab above with the data-file
    EXTENSION layer (memory/feedback_patterns.jsonl) so feedback classification
    can be broadened by editing the .jsonl — no code change. The code lists
    stay the canonical, trace-commented source; this only ADDS tokens. Lazy
    import avoids import-order coupling; memory's loader caches the file.
    """
    try:
        from memory import load_feedback_patterns
        extra = load_feedback_patterns(category)
    except Exception:
        extra = ()
    return base + tuple(t for t in extra if t not in base)

# Scoped-behavior terms: if these appear in a strict scoped feedback turn,
# route to a ONE-PATCH behavior fix path (not media-only regen). Added for
# Mortal Kombat trace where "turn around / facing / CPU behavior" got
# misrouted into art regeneration.
_SCOPED_BEHAVIOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bturn(?:\s*-\s*|\s+)around\b", re.I),
    re.compile(r"\bfac(?:e|es|ing)\b", re.I),
    re.compile(r"\bcpu\b", re.I),
    re.compile(r"\bai\b", re.I),
    re.compile(r"\bbehavior\b", re.I),
    re.compile(r"\baction(?:s)?\b", re.I),
    re.compile(r"\b(?:kick|kicks|kicking)\b", re.I),
    re.compile(r"\b(?:punch|punches|punching)\b", re.I),
    re.compile(r"\b(?:attack|attacks|attacking)\b", re.I),
    re.compile(r"\banimation\s+for\b", re.I),
)
_SCOPED_SIZE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:x|times?)\s*(?:larger|smaller|bigger)\b", re.I,
    ),
    re.compile(r"\b(?:larger|smaller|bigger)\b", re.I),
    re.compile(r"\b(?:scale|size)\b", re.I),
)
# Layout/position tweak vocabulary: "move/shift/nudge ... down/up/left/
# right/by", "reposition", explicit "N px / N pixels". Genre-free: this
# describes a LAYOUT-modality edit (where a UI element sits), not subject
# matter. Used ONLY to arm a one-patch scope lock on a SEED first build so
# a weak model can't be pushed into a full-file rewrite of a large seed for
# a trivial reposition (2026-06-25 trace 161300_697573: "move the weapons
# selection buttons down 50 pixels" forced a 28-min rewrite loop).
_SCOPED_POSITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:reposition|relocate|realign|nudge)\b", re.I),
    re.compile(
        r"\b(?:move|shift|slide|push|drop|raise|lower|offset)\b.*?"
        r"\b(?:up|down|left|right|over|by|to\s+the)\b",
        re.I,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:px|pixels?)\b", re.I),
)
# Recolor tweak vocabulary (companion to the position/size/orientation
# tweak detectors). Tight on purpose — only explicit recolor phrasing.
_SCOPED_RECOLOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:recolou?r)\b", re.I),
    re.compile(r"\bchange\b.{0,20}\bcolou?r\b", re.I),
)

# Patch budget for a small-scope SEED edit. An initial seed goal often
# bundles several edits ("move the buttons + add upgrade/sell + rotate the
# guns" = 8 patches in trace 194238_248190), so the single-tweak max of 1
# would reject the whole reply. Generous on purpose — the scope lock's only
# job on a seed edit is to forbid the doomed full <html_file> rewrite;
# per-patch validation + auto-revert remain the real quality gates.
_SEED_EDIT_MAX_PATCHES = 16

def _feedback_locks_code(text: str) -> bool:
    """User explicitly forbade code changes for this turn.

    Matches phrases like "no changes to the code", "only the asset",
    "just that one sprite", "don't touch the code". Used to suppress
    the one-shot <html_file> rewrite exemption that fresh feedback
    would otherwise arm — when the user locked the code, the rewrite
    license is exactly what we DON'T want.
    """
    return any(p.search(text) for p in _CODE_LOCK_PATTERNS)


_STRICT_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bonly\b.*\b(?:change|fix|edit|touch|update|do)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:don['’]?t|do\s+not)\s+change\s+anything\s+else\b",
        re.I,
    ),
    re.compile(r"\bnothing\s+else\b", re.I),
)


def _feedback_is_strict_scope(text: str) -> bool:
    """True when feedback explicitly narrows the turn to one scoped change."""
    if not text:
        return False
    if _feedback_locks_code(text):
        return True
    return any(p.search(text) for p in _STRICT_SCOPE_PATTERNS)


# Behavior verbs — gameplay actions a player or game object performs.
# Used to detect "user is reporting a behavior bug" patterns like
# "mario doesn't climb" / "barrels don't roll" / "nothing happens when
# I press space". When fired, the MEDIA-CHANGE DIRECTIVE is suppressed
# even if an asset name appears in the feedback (DK trace
# 20260514_104131 burned 7 consecutive turns because "mario" + "ladder"
# matched the art-change classifier and the model was told the
# feedback was about ART/SOUND when the user was clearly reporting a
# code bug).
#
# Genre-free / game-agnostic: every entry describes input or state
# transitions, not subject matter. Visual verbs (look, appear, render,
# show) are intentionally EXCLUDED so "the dragon doesn't look right"
# still routes to the art-change path.
_BEHAVIOR_VERBS: tuple[str, ...] = (
    "climb", "climbs", "climbing", "climbed",
    "move", "moves", "moving", "moved",
    "jump", "jumps", "jumping", "jumped",
    "run", "runs", "running",
    "walk", "walks", "walking", "walked",
    "fire", "fires", "firing", "fired",
    "shoot", "shoots", "shooting", "shot",
    "work", "works", "working", "worked",
    "respond", "responds", "responding", "responded",
    "react", "reacts", "reacting", "reacted",
    "fall", "falls", "falling", "fell",
    "fly", "flies", "flying", "flew",
    "spawn", "spawns", "spawning", "spawned",
    "reset", "resets", "resetting",
    "restart", "restarts", "restarting", "restarted",
    "trigger", "triggers", "triggering", "triggered",
    "hit", "hits", "hitting",
    "register", "registers", "registering", "registered",
    "roll", "rolls", "rolling", "rolled",
    "drop", "drops", "dropping", "dropped",
    "happen", "happens", "happening", "happened",
    "play", "plays", "playing", "played",
    "load", "loads", "loading", "loaded",
    "update", "updates", "updating", "updated",
    "collide", "collides", "colliding", "collided",
    "die", "dies", "dying", "died",
    "score", "scores", "scoring", "scored",
)
_BEHAVIOR_VERB_ALT = "|".join(re.escape(v) for v in _BEHAVIOR_VERBS)
_BEHAVIOR_BUG_NEGATION_RE = re.compile(
    r"\b(?:doesn['’]?t|don['’]?t|didn['’]?t|can['’]?t|cannot|"
    r"won['’]?t|wouldn['’]?t|isn['’]?t|aren['’]?t|never|"
    r"nothing|unable\s+to|fails?\s+to|not)\s+"
    r"(?:\w+\s+){0,3}"
    rf"(?:{_BEHAVIOR_VERB_ALT})\b",
    re.IGNORECASE,
)
_BEHAVIOR_BUG_COMPLAINT_RE = re.compile(
    r"\b(?:bug|broken|crash(?:ing|ed|es)?|freez(?:ing|es)?|frozen|"
    r"stuck|hang(?:ing|s|ed)?|glitch(?:ing|ed|es)?)\b",
    re.IGNORECASE,
)

# INVERTED-BEHAVIOR patterns — the user is reporting that behavior IS
# happening but in the WRONG way (reversed/swapped/inverted). The
# existing negation regex only catches "X doesn't Y", not "X does the
# opposite of Y". Doom trace 2026-05-23: user wrote "down key moves
# you forward" and "directions and views getting reversed" — clear
# code bugs that slipped past the classifier and got misrouted to
# MEDIA-CHANGE because no behavior_bug suppression fired.
#
# These patterns intentionally pair an INPUT/CONTROL noun with a
# WRONG-DIRECTION qualifier so they don't false-positive on art
# language ("the sprite is inverted" stays orientation-only).
_BEHAVIOR_BUG_INVERTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "direction(s) ... reversed/wrong/inverted/swapped/backwards" within ~6 words
    re.compile(
        r"\bdirection(?:s)?\b[\w\s,]{0,40}\b"
        r"(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|"
        r"flipped|backwards?)\b",
        re.I,
    ),
    # Symmetric: "reversed/wrong/... ... direction(s)" within ~6 words
    re.compile(
        r"\b(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|flipped)\b"
        r"[\w\s,]{0,40}\bdirection(?:s)?\b",
        re.I,
    ),
    # "movement / controls / motion / input / axis ... wrong-shape"
    re.compile(
        r"\b(?:movement|controls?|motion|input|axis|axes)\b[\w\s,]{0,40}\b"
        r"(?:reversed?|invert(?:ed|ing)?|wrong|opposite|swapped|"
        r"flipped|backwards?)\b",
        re.I,
    ),
    # "wrong way" or "opposite way" as a standalone control complaint
    # (already covered by orientation_change for sprite contexts; here
    # we add it to behavior_bug so the input-mismatch case routes too).
    re.compile(r"\b(?:wrong|opposite)\s+(?:way|direction)\b", re.I),
    # Input key + verb + opposite output: "down key moves you forward",
    # "up arrow goes backwards", "left button sends you right".
    re.compile(
        r"\b(?:up|down|left|right|forward|back|backwards?|w|a|s|d)\s+"
        r"(?:key|arrow|button)\b[\w\s,]{0,40}\b"
        r"(?:moves?|sends?|goes|going|turns?|takes?|points?)\b"
        r"[\w\s,]{0,20}\b"
        r"(?:up|down|left|right|forward|back|backwards?|opposite|wrong)\b",
        re.I,
    ),
)


def _feedback_is_behavior_bug(text: str) -> bool:
    """User is reporting a behavior / code bug — "X doesn't Y",
    "nothing happens when …", "the game is broken / frozen / crashing",
    OR "X is happening but in the WRONG / REVERSED way" (input-control
    mismatch).

    DK trace 20260514_104131 fix: the existing art-change classifier
    fires True whenever an asset name appears in feedback, which
    misroutes "mario does not climb the ladder" as an ART/SOUND change
    request (because "mario" and "ladder" are asset names). Detecting
    behavior-bug language lets the directive injector suppress the
    misrouting.

    Doom trace 2026-05-23 fix: also detect INVERTED-behavior complaints
    ("down key moves you forward", "directions getting reversed"). The
    old negation regex only matched "X doesn't Y", missing the "X does
    the OPPOSITE of Y" failure class entirely.

    Patterns matched:
      - negation + behavior verb within 3 words ("does not climb",
        "won't reset", "isn't responding")
      - explicit complaint nouns ("bug", "broken", "crashing",
        "frozen", "stuck", "glitching")
      - inverted-behavior pairs ("direction(s) reversed", "movement
        inverted", "down key moves you forward")

    Visual-only verbs (look, appear, render, show) are NOT in the
    behavior-verb set, so "the dragon doesn't look right" stays on
    the art-change path.
    """
    if not text:
        return False
    if _BEHAVIOR_BUG_COMPLAINT_RE.search(text):
        return True
    if _BEHAVIOR_BUG_NEGATION_RE.search(text):
        return True
    if any(p.search(text) for p in _BEHAVIOR_BUG_INVERTED_PATTERNS):
        return True
    return False


def _name_in_text(text_lower: str, names: list[str]) -> bool:
    """Match a known asset/sound name in feedback, tolerating
    underscores ↔ hyphens ↔ spaces (so "centipede tail" matches
    "centipede_tail" and vice versa)."""
    canon_text = re.sub(r"[\s\-]+", "_", text_lower)
    for name in names:
        n = (name or "").strip().lower()
        if not n:
            continue
        canon_name = re.sub(r"[\s\-]+", "_", n)
        if canon_name and canon_name in canon_text:
            return True
    return False


def _matched_names_in_text(text_lower: str, names: list[str]) -> set[str]:
    """Return canonical known names found in feedback text."""
    canon_text = re.sub(r"[\s\-]+", "_", text_lower)
    matched: set[str] = set()
    for name in names:
        n = (name or "").strip().lower()
        if not n:
            continue
        canon_name = re.sub(r"[\s\-]+", "_", n)
        if canon_name and canon_name in canon_text:
            matched.add(canon_name)
    return matched


def _feedback_referenced_asset_names(text: str, asset_names: list[str]) -> set[str]:
    """Asset names the user feedback explicitly names (literal + fuzzy stems)."""
    if not text or not asset_names:
        return set()
    lo = text.lower()
    referenced = set(_matched_names_in_text(lo, asset_names))
    for _stem, names in (_resolve_fuzzy_asset_stems(text, asset_names) or {}).items():
        for name in names:
            referenced.add(re.sub(r"[\s\-]+", "_", (name or "").strip().lower()))
    return referenced


def _feedback_art_request_already_on_disk(text: str, asset_names: list[str]) -> bool:
    """True when feedback names sprites that already exist this session.

    TD seed trace 20260630_114658: user re-asked for tower_tesla_idle /
    tower_flame_idle seconds after mid_session generation — reprompting
    <assets> was wrong; wiring/draw coaching was needed instead.
    """
    if not text or not asset_names:
        return False
    if _feedback_requests_explicit_new_art(text):
        return False
    referenced = _feedback_referenced_asset_names(text, asset_names)
    if not referenced:
        return False
    canon_assets = {
        re.sub(r"[\s\-]+", "_", (n or "").strip().lower())
        for n in asset_names
        if (n or "").strip()
    }
    return referenced.issubset(canon_assets)


# Tokens that appear as common prefixes/suffixes in asset names and are NOT
# distinctive on their own — "white" alone shouldn't pull in every white_*
# sprite in the roster. Used by `_resolve_fuzzy_asset_stems` to skip stems
# the user almost certainly didn't mean as a piece identifier.
_NON_DISTINCTIVE_ASSET_STEMS: frozenset[str] = frozenset({
    "white", "black", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey",
    "left", "right", "up", "down", "front", "back", "side", "view",
    "small", "large", "big", "tiny", "huge",
    "light", "dark", "bright",
    "player", "enemy", "npc",  # too broad to disambiguate
    "frame", "anim", "idle", "walk", "run", "attack", "hurt", "die",
    "1", "2", "3", "4",
})


def _resolve_fuzzy_asset_stems(
    text: str, asset_names: list[str]
) -> dict[str, list[str]]:
    """Map distinctive last-tokens of asset names to the parent assets.

    Example: assets = [white_pawn, black_pawn, white_king, black_king];
    user text = "make the pawns walk and the king attack". Returns
    {"pawn": ["white_pawn", "black_pawn"], "king": ["white_king",
    "black_king"]}.

    Used to surface "user said 'pawn', which maps to [white_pawn,
    black_pawn] in your asset map" inside the MEDIA-CHANGE directive
    so the model can emit the right `from_image` chains without having
    to guess. Token-level match; color/side prefixes ("white", "black")
    are skipped via `_NON_DISTINCTIVE_ASSET_STEMS`.
    """
    if not text or not asset_names:
        return {}
    lo = text.lower()
    out: dict[str, list[str]] = {}
    for name in asset_names:
        canon = re.sub(r"[\s\-]+", "_", (name or "").strip().lower())
        if not canon:
            continue
        tokens = [t for t in canon.split("_") if t]
        if not tokens:
            continue
        # Prefer the LAST distinctive token. Walk right-to-left so
        # ("white", "pawn") picks "pawn"; ("hero", "idle", "1") picks
        # "hero" (since "idle" and "1" are non-distinctive).
        stem: str | None = None
        for tok in reversed(tokens):
            if len(tok) < 3 or tok in _NON_DISTINCTIVE_ASSET_STEMS:
                continue
            stem = tok
            break
        if stem is None:
            continue
        if not re.search(rf"\b{re.escape(stem)}s?\b", lo):
            continue
        out.setdefault(stem, [])
        if name not in out[stem]:
            out[stem].append(name)
    return out


# Prefix for feedback items written by the HARNESS (not the user). The
# media-request classifiers must never treat these as user asks — see
# `_feedback_is_art_change`. The model still sees the full text.
_HARNESS_ADVISORY_SENTINEL = "[HARNESS ADVISORY — informational, not a request]"


_UI_FEATURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhelp\s+screen\b", re.I),
    re.compile(r"\bhow\s+to\s+play\b", re.I),
    re.compile(r"\binstructions?\b", re.I),
    re.compile(r"\btutorial\b", re.I),
    re.compile(r"\b(?:hud|menu|pause|settings|credits?)\s+(?:screen|overlay|panel|modal)\b", re.I),
    # Button/overlay phrasings: "add a help button", "hint button",
    # "help overlay", "hint text/panel". Genre-free UI nouns + verbs —
    # these are code/DOM features, never generated art (2026-06-21
    # point-and-click seed trace: "add a help button" missed the
    # narrower "help screen" pattern and got queued behind a blocker).
    re.compile(r"\bhelp\s+(?:button|overlay|panel|modal|dialog|box)\b", re.I),
    re.compile(r"\bhint\s+(?:button|overlay|panel|modal|dialog|box|text|system)\b", re.I),
    re.compile(r"\badd\s+(?:a\s+|an\s+)?help\b", re.I),
    re.compile(r"\badd\s+(?:a\s+|an\s+)?hint\b", re.I),
)


def _feedback_is_ui_feature(text: str) -> bool:
    """True when the user asks for UI/help, not new generated media."""
    if not text:
        return False
    return any(p.search(text) for p in _UI_FEATURE_PATTERNS)


def _has_audio_context(text_lower: str) -> bool:
    """True when feedback language is explicitly about audio."""
    if any(
        re.search(rf"\b{re.escape(n)}\b", text_lower)
        for n in _feedback_vocab("sound_nouns", _SOUND_NOUNS)
    ):
        return True
    audio_words = (
        "volume", "louder", "quieter", "quiet", "loud", "mute",
        "unmute", "pitch", "echo", "bass", "treble",
    )
    return any(re.search(rf"\b{re.escape(w)}\b", text_lower) for w in audio_words)


def _feedback_is_art_change(text: str, asset_names: list[str]) -> bool:
    """User feedback is asking to change visual art.

    Heuristic: a known asset name appears in the text (literal OR via
    fuzzy stem — Phase 0.11), OR text contains an art-noun ("sprite",
    "image", "art", …) together with a media verb ("change", "redraw",
    "make", "look", "want", …). Misses are safe (no directive injected,
    regular flow proceeds). False positives are also safe — the
    directive only advises the model to prefer <assets>.

    Phase 0.11: when the user says "all the pawns" and the session has
    `white_pawn_idle, black_pawn_idle` etc., the fuzzy stem match
    counts. Without this, a session with N-frame entity names (every
    asset is `<entity>_<state>`) would skip art-change classification
    even when the user clearly referenced an entity.

    Harness-authored advisories (sentinel-prefixed) are NEVER art
    requests: 2026-06-10 dojo-fight traces show the ASSET SANITY
    WARNING (which itself says "do NOT regenerate") being classified
    as a user art request here, arming 3 turns of "ASSET GENERATION
    REQUIRED — The user asked you to GENERATE NEW ART" that
    contradicted both the advisory and the recovery prompt in the
    same message.
    """
    if text.lstrip().startswith(_HARNESS_ADVISORY_SENTINEL):
        return False
    if _feedback_is_ui_feature(text):
        return False
    # Wire/load existing sprites — code fix, not art regen. img2img chain
    # and style-rebrand still route through MEDIA-CHANGE when appropriate.
    if (
        _feedback_requests_existing_media(text)
        and not _feedback_requests_img2img_chain(text)
        and not _feedback_requests_style_rebrand(text)
    ):
        return False
    lo = text.lower()
    if _name_in_text(lo, asset_names):
        return True
    if _resolve_fuzzy_asset_stems(text, asset_names):
        return True
    has_noun = any(
        re.search(rf"\b{re.escape(n)}\b", lo)
        for n in _feedback_vocab("art_nouns", _ART_NOUNS)
    )
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo)
        for v in _feedback_vocab("media_verbs", _MEDIA_VERBS)
    )
    return has_noun and has_verb


def _feedback_is_sound_change(text: str, sound_names: list[str]) -> bool:
    """Classifier heuristic for `<sounds>` regen feedback.

    Action words that double as combat-game sound names ("kick",
    "punch", "block", "hit", "jump", "attack", "fireball",
    "fatality") are AMBIGUOUS — they appear in graphics-only feedback
    too ("the CPU kick is facing the wrong way", MK trace
    20260517_220025). When ONLY those names matched, require explicit
    audio vocabulary (a `_SOUND_NOUN` or volume/pitch word) before
    routing to sound regen. False positives steer the model toward
    `<sounds>` regeneration on a graphics turn, polluting the prompt
    and (in the trace) burning iters; false negatives keep the turn
    on its current path which is the safer outcome.
    """
    lo = text.lower()
    matched = _matched_names_in_text(lo, sound_names)
    if matched:
        ambiguous = {
            "kick", "punch", "block", "hit", "jump", "attack",
            "fireball", "fatality",
        }
        unambiguous_match = any(name not in ambiguous for name in matched)
        if unambiguous_match:
            return True
        if _has_audio_context(lo):
            return True
        # All matched names are ambiguous AND no audio context — do
        # NOT fall through to the noun+verb gate, which has False
        # negatives we already accept. Returning False here pins the
        # behavior observed by the MK regression tests.
        return False
    # No sound-name match at all → require BOTH a sound noun and a
    # media verb. This catches generic "redo the music" / "redesign
    # the soundtrack" without an existing-name match.
    has_noun = any(
        re.search(rf"\b{re.escape(n)}\b", lo) for n in _SOUND_NOUNS
    )
    has_verb = any(
        re.search(rf"\b{re.escape(v)}\b", lo) for v in _MEDIA_VERBS
    )
    return has_noun and has_verb


_EXISTING_MEDIA_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdon['’]?t\s+(?:redo|redraw|regenerate|remake)\b", re.I),
    re.compile(r"\bdo\s+not\s+(?:redo|redraw|regenerate|remake)\b", re.I),
    re.compile(r"\buse\s+(?:the\s+)?(?:existing|original|old|current)\b", re.I),
    re.compile(r"\balready\s+exists?\b", re.I),
    re.compile(r"\buse\s+them\b", re.I),
    # Wire/load already-generated media — code work, not <assets> regen.
    # Chess trace 20260621_193434: "assets need to be loaded, they were
    # created but are not being used" mis-fired ASSET GENERATION REQUIRED.
    re.compile(r"\bneed(?:s|ed|ing)?\s+to\s+be\s+loaded\b", re.I),
    re.compile(
        r"\b(?:not|aren't|isn't|wasn't|weren't)\s+(?:being\s+)?"
        r"(?:used|loaded|loading|drawn|shown|displayed|visible|wired)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:were|was|where|are|is)\s+created\b.*\b(?:but|yet|still)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bcreated\b.*\b(?:but|and)\b.*\b(?:not|aren't|isn't)\s+"
        r"(?:being\s+)?(?:used|loaded|loading)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\b(?:load|wire|hook\s+up|integrate|connect)\s+"
        r"(?:the\s+|these\s+|those\s+)?"
        r"(?:assets?|sprites?|images?|art|graphics?|pngs?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:use|draw|show|display)\s+(?:the\s+)?"
        r"(?:generated|created)\s+"
        r"(?:assets?|sprites?|images?|art|graphics?)\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+(?:using|showing|drawing|displaying)\s+"
        r"(?:the\s+)?(?:assets?|sprites?|images?|generated|created)\b",
        re.I,
    ),
)

# When both wiring and explicit new-art language appear, regen wins.
_EXPLICIT_NEW_ART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:new|another|additional|extra|different)\s+"
        r"(?:sprite|sprites|asset|assets|image|images|art|graphic|graphics|"
        r"frame|frames|png|pngs)\b",
        re.I,
    ),
    re.compile(r"\b(?:regenerat|redraw|remake|redo)\w*\b", re.I),
    re.compile(r"\bmake\s+(?:a\s+|an\s+|me\s+)?new\b", re.I),
    re.compile(
        r"\b(?:create|generat)\w*\s+(?:new\s+)?"
        r"(?:sprite|sprites|asset|assets|image|images|art|graphic|graphics|"
        r"frame|frames)\b",
        re.I,
    ),
)


def _feedback_requests_explicit_new_art(text: str) -> bool:
    """True when feedback explicitly asks for newly generated media."""
    if not text:
        return False
    # "don't redo / do not regenerate" is a keep-existing signal, not new art.
    if re.search(
        r"\b(?:don['’]?t|do\s+not)\s+(?:redo|redraw|regenerat\w*|remake)\b",
        text,
        re.I,
    ):
        return False
    return any(p.search(text) for p in _EXPLICIT_NEW_ART_PATTERNS)


def _feedback_requests_existing_media(text: str) -> bool:
    """User wants existing media wired/used, not regenerated."""
    if not text:
        return False
    if not any(p.search(text) for p in _EXISTING_MEDIA_ONLY_PATTERNS):
        return False
    if _feedback_requests_explicit_new_art(text):
        return False
    return True


# Phase 0.1 — phrases that explicitly request img2img CHAINING (seed from
# an existing sprite, produce variants/animation frames). Distinct from
# "use the existing PNG verbatim" — the user wants NEW frames seeded from
# an existing one, which is exactly what SD-Turbo img2img handles. See
# 2026-05-22 chess trace: "use the existing assets for each pawn... as
# starting point but then show each walking" — pure img2img.
_IMG2IMG_CHAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bas\s+(?:a\s+|the\s+)?(?:starting\s+point|base|seed|basis|reference)\b", re.I),
    re.compile(r"\bbased\s+on\s+(?:the\s+)?(?:existing|current|original)\b", re.I),
    re.compile(r"\bseed(?:ed)?\s+from\b", re.I),
    re.compile(r"\b(?:show|showing|make)\s+(?:them|it|each|the\s+\w+\s+)?(?:walking|running|attacking|moving|jumping|dancing|fighting|smashing|killing|dying|capturing)\b", re.I),
    re.compile(r"\b(?:more|additional|extra|new)\s+(?:frame|frames|variant|variants|version|versions|animation)\b", re.I),
    re.compile(r"\banimated\s+(?:series|sequence|set|version)\b", re.I),
    re.compile(r"\banimat(?:e|ing)\s+(?:the\s+)?(?:existing|current|pieces|sprites?|characters?)\b", re.I),
    re.compile(r"\bwalk(?:ing)?\s+cycle\b", re.I),
)


def _feedback_requests_img2img_chain(text: str) -> bool:
    """True when the user is asking for animation frames or variants
    chained from EXISTING sprites — the SD-Turbo `from_image` path."""
    if not text:
        return False
    return any(p.search(text) for p in _IMG2IMG_CHAIN_PATTERNS)


# Phase 0.11 — phrases that signal the user wants a STYLE REBRAND of
# existing assets (regenerate every existing asset NAME with a NEW
# prompt that bakes in a different style). Distinct from:
#   - existing_media: "use the existing X, don't regen" (KEEP as-is)
#   - img2img_chain: "use existing as starting point for animation
#                     frames" (REGEN as variants via from_image)
#   - style_rebrand: "ALL of the images need to look like X" /
#                    "all new graphics, look like monsters" — REGEN
#                    each name in place via txt2img (NOT from_image,
#                    because that would carry the old style forward)
_STYLE_REBRAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blook(?:s|ing)?\s+like\b", re.I),
    re.compile(r"\binstead\s+of\b", re.I),
    re.compile(r"\bnot\s+look\s+like\b", re.I),
    re.compile(r"\bnot\s+(?:regular|standard|normal|plain|generic)\b", re.I),
    re.compile(r"\b(?:themed?|styled?)\s+(?:as|like)\b", re.I),
    re.compile(r"\b(?:in\s+the\s+)?style\s+of\b", re.I),
    re.compile(r"\bnew\s+(?:style|look|theme|art|graphics|design|visuals?)\b", re.I),
    re.compile(r"\bdifferent\s+(?:style|look|theme|art|design|aesthetic)\b", re.I),
    re.compile(r"\b(?:all|every)\s+(?:new|fresh)\s+(?:graphics?|sprites?|images?|art|assets?)\b", re.I),
    re.compile(r"\b(?:re\s*-?\s*)?(?:design|theme|skin|reskin)\s+(?:them|the\s+\w+|all)\b", re.I),
    re.compile(r"\bmake\s+(?:them|all|every)\b.*\blook\b", re.I | re.DOTALL),
)


def _feedback_requests_style_rebrand(text: str) -> bool:
    """True when the user wants existing assets re-rendered with a new
    visual style (txt2img with new prompts, not from_image)."""
    if not text:
        return False
    return any(p.search(text) for p in _STYLE_REBRAND_PATTERNS)


def _feedback_requests_size_change(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _SCOPED_SIZE_PATTERNS)


# Orientation-change vocabulary: "invert / mirror / flip / face the
# other way / facing wrong / horizontally". Genre-free: describes a
# rendering modality (mirror a sprite) not subject matter. A false
# positive routes the turn to a one-patch canvas-flip recipe instead
# of asset regeneration, so the gate also REQUIRES the absence of
# explicit style-change verbs ("redraw", "redesign", "new art") and
# does not fire when the user explicitly says "new asset" / "make a
# new" — those are regen requests, not mirror requests.
_ORIENTATION_VERB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\binvert(?:ed|ing)?\b", re.I),
    re.compile(r"\bmirror(?:ed|ing)?\b", re.I),
    re.compile(r"\bflip(?:ped|ping)?\b", re.I),
    re.compile(r"\b(?:facing|faces|face)\s+(?:the\s+)?(?:wrong|other|opposite|right|left)\b", re.I),
    re.compile(r"\bwrong\s+(?:way|direction)\b", re.I),
    re.compile(r"\bhorizontal(?:ly)?\b", re.I),
    re.compile(r"\brotat(?:e|ed|ing|ion)\s+(?:just|only|the)\b", re.I),
)
_ORIENTATION_REGEN_BLOCKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnew\s+asset\b", re.I),
    re.compile(r"\bmake\s+(?:a\s+)?new\b", re.I),
    re.compile(r"\bregenerat(?:e|ed|ing)\b", re.I),
    re.compile(r"\bredraw\b", re.I),
    re.compile(r"\bredesign\b", re.I),
    re.compile(r"\bdifferent\s+style\b", re.I),
)

# Negation tokens that, when they precede a blocker phrase, INVERT the
# blocker's meaning ("do NOT make a new asset" forbids regen — the
# opposite of the bare blocker's "user wants regen" reading). The
# doom trace 2026-05-23 was the motivating case: user wrote "do not
# make a NEW asset, i think the pistal maybe facing the wrong way"
# and the literal `\bnew\s+asset\b` blocker fired, suppressing the
# orientation route and letting MEDIA-CHANGE re-render the pistol —
# exactly what the user forbade.
_BLOCKER_NEGATION_RE = re.compile(
    r"\b(?:not|no|don['’]?t|doesn['’]?t|never|without|please\s+don['’]?t)\b",
    re.I,
)


def _phrase_is_negated(text: str, match_start: int, window_chars: int = 30) -> bool:
    """True when a negation word appears within `window_chars` BEFORE match_start."""
    pre = text[max(0, match_start - window_chars): match_start]
    return bool(_BLOCKER_NEGATION_RE.search(pre))


def _feedback_is_orientation_change(text: str) -> bool:
    """Classifier heuristic: user wants a sprite mirrored/flipped on
    the canvas (one small <patch>), not regenerated. False positives
    push a code patch path when the user actually wanted regen, so
    blockers like "new asset" / "redraw" suppress the route — UNLESS
    those blocker phrases are themselves negated ("do NOT make a new
    asset"), in which case the user is forbidding regen and the
    orientation route should fire.

    Genre-free: vocabulary describes rendering modality, not subject
    matter. Returns True only when an orientation verb fires AND no
    un-negated regen blocker fires.
    """
    if not text:
        return False
    for p in _ORIENTATION_REGEN_BLOCKERS:
        for m in p.finditer(text):
            if not _phrase_is_negated(text, m.start()):
                return False
    return any(p.search(text) for p in _ORIENTATION_VERB_PATTERNS)


def _feedback_mentions_scoped_behavior_change(text: str) -> bool:
    if not text:
        return False
    if _feedback_is_behavior_bug(text):
        return True
    return any(p.search(text) for p in _SCOPED_BEHAVIOR_PATTERNS)


def _goal_is_small_scope_edit(text: str) -> bool:
    """True when a seed-edit goal is a localized tweak — reposition,
    resize, rotate/flip, or recolor of existing elements — rather than a
    structural rebuild.

    Used ONLY to arm a single-patch scope lock on a SEED first build (the
    full file is injected for patching by `_seed_html_for_prompt`), so a
    weak local model can't fall back to a doomed full-file `<html_file>`
    rewrite of a large seed (2026-06-25 trace 161300_697573).

    Genre-free: every pattern describes a layout/rendering-modality edit
    (position, scale, orientation, color), never subject matter. Reuses the
    existing size + orientation detectors so the vocabulary stays in one
    place.
    """
    if not text:
        return False
    if _feedback_requests_size_change(text):
        return True
    if _feedback_is_orientation_change(text):
        return True
    if any(p.search(text) for p in _SCOPED_POSITION_PATTERNS):
        return True
    if any(p.search(text) for p in _SCOPED_RECOLOR_PATTERNS):
        return True
    return False


def _scoped_probe_keywords(text: str, limit: int = 3) -> list[str]:
    """Pick a tiny deterministic keyword set for scoped-check probes.

    Keep this intentionally short for local-model correction prompts.
    """
    if not text:
        return []
    lo = text.lower()
    ordered = (
        "turn around",
        "facing",
        "cpu",
        "ai",
        "behavior",
        "action",
        "kick",
        "punch",
        "jump",
        "attack",
    )
    out: list[str] = []
    for key in ordered:
        if key in lo:
            out.append(key)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Subsystem hints — map mistake_signature shapes to a code region the model
# should look at. Used by (a) the coaching message at _repeat_sig_streak >= 2
# and (b) the focused-slice keyset biaser. DK trace 20260514_104131 evidence:
# 100% of recent sessions had signatures containing "INPUT_DEAD" / "Controls
# are not wired up" AND the model's patches didn't touch any addEventListener
# / keydown code — the verifier signal pointed to the input layer but the
# model kept patching the higher-level mechanic the user named (climb math,
# barrel physics, etc.). Existing coaching at _repeat_sig_streak >= 2 said
# "AUTHOR a runtime-state probe" — too abstract for a 27B model already
# focused on the wrong area.
#
# Genre-free / model-agnostic: each entry describes the SHAPE of a failure
# (input wiring, frame/draw loop, RAF kick-off), not subject matter.
# Asset-load is intentionally OMITTED here — the existing asset-specific
# coaching branch handles that case (DK 20260513_122154 fix).
# ---------------------------------------------------------------------------

# (signature_substrings_lower, name, identifier_tokens, fix_phrase)
_SUBSYSTEM_HINTS: tuple[
    tuple[tuple[str, ...], str, tuple[str, ...], str], ...
] = (
    (
        (
            "input_dead",
            "controls are not wired up",
            "controls not wired",
            "input handler is broken",
        ),
        "input",
        (
            "addEventListener", "keydown", "keyup", "KeyboardEvent",
            "code", "KEYMAP", "keys",
        ),
        "the keydown/keyup handler (e.g. `window.addEventListener("
        "'keydown', ...)` and the key-state map)",
    ),
    (
        (
            "frozen",
            "did not change between two samples",
            "canvas drew something but did not change",
        ),
        "draw_or_raf",
        (
            "requestAnimationFrame", "frame", "render", "draw", "ctx",
        ),
        "the frame/draw loop (the function called by "
        "`requestAnimationFrame`)",
    ),
    (
        (
            "canvas pixels are uniform",
            "not rendering",
        ),
        "raf_start",
        (
            "requestAnimationFrame", "loadAssets", "then",
            "startGame", "init",
        ),
        "the RAF kick-off (e.g. `loadAssets().then(() => "
        "requestAnimationFrame(frame))`)",
    ),
)


def _subsystem_hint(signature: str) -> dict | None:
    """Map a mistake_signature to a structured subsystem hint.

    Returns dict {name, identifiers, fix_phrase} or None when no
    entry matches. Callers use:
      - `identifiers` to bias _focused_slice's keyset.
      - `fix_phrase` to build a directive coaching message that names
        the specific subsystem to patch this turn.
    """
    if not signature:
        return None
    low = signature.lower()
    for substrings, name, identifiers, fix_phrase in _SUBSYSTEM_HINTS:
        if any(sub in low for sub in substrings):
            return {
                "name": name,
                "identifiers": identifiers,
                "fix_phrase": fix_phrase,
            }
    return None


class FeedbackRoutingMixin:

    """Feedback routing for GameAgent (see module docstring)."""


    _CLASSIFIER_AUTO_DISABLE_THRESHOLD = 2


    def _record_classifier_overrule(self, *, expected_mode: str, model_emitted: str, feedback_preview: str) -> None:
        """Bump the overrule counter + auto-disable directives at threshold.

        Centralizes the bookkeeping so both overrule emit sites stay in
        sync. When the running count reaches the threshold we flip
        `_use_feedback_directives = False` for the rest of the session
        (a per-session decision; new sessions start fresh).
        """
        self._trace({
            "kind": "scoped_classifier_overruled_by_model",
            "expected_mode": expected_mode,
            "model_emitted": model_emitted,
            "feedback_preview": feedback_preview,
            "count": self._classifier_overrule_count + 1,
        })
        self._classifier_overrule_count += 1
        if (
            not self._classifier_auto_disabled
            and self._classifier_overrule_count >= self._CLASSIFIER_AUTO_DISABLE_THRESHOLD
            and getattr(self, "_use_feedback_directives", True)
        ):
            self._use_feedback_directives = False
            self._classifier_auto_disabled = True
            self._trace({
                "kind": "classifier_auto_disabled_after_repeated_overrules",
                "count": self._classifier_overrule_count,
                "threshold": self._CLASSIFIER_AUTO_DISABLE_THRESHOLD,
                "reason": (
                    "model overruled the scoped classifier "
                    f"{self._classifier_overrule_count} times this session; "
                    "switching to raw feedback mode for the rest of the "
                    "session per the agent-must-beat-zero-shot rule. "
                    "Resets next /new."
                ),
            })


    def _queue_internal_feedback(self, text: str) -> None:
        """Queue an AGENT-generated notice into the feedback channel.

        Added 2026-06-12: agent-injected texts (mid-session media loader
        notices, scoped-media locks, autonomous-playtest findings) ride
        the same `_pending_feedback` queue as real user feedback, so the
        unhonored-asset-request detector in `_flush_user_injections`
        classified them as USER art requests — trace 20260612_004616
        fired 8 spurious ASSET GENERATION REQUIRED banners demanding
        <assets> for files that already existed on disk. Texts queued
        here are remembered in `_internal_feedback_texts` so detectors
        that must only fire on genuine user feedback can skip them.
        The model still receives the text unchanged.
        """
        text = (text or "").strip()
        if not text:
            return
        # Lazy-init so partially-constructed test stubs (which skip
        # __init__) don't AttributeError here.
        if not hasattr(self, "_internal_feedback_texts"):
            self._internal_feedback_texts = set()
        self._internal_feedback_texts.add(text)
        self._pending_feedback.append(text)



    def _wire_existing_assets_coaching_block(
        self, text: str, asset_names: list[str],
    ) -> str:
        """Coaching when user art feedback names sprites already on disk."""
        referenced = sorted(_feedback_referenced_asset_names(text, asset_names))
        names_line = ", ".join(referenced) if referenced else "(named sprites)"
        return (
            "================ EXISTING SPRITES ON DISK ================\n"
            f"The user asked about art for: {names_line}\n"
            "Those PNGs were already generated this session and exist on disk.\n"
            "Do NOT re-emit <assets> unless changing the visual prompt.\n"
            "THIS turn: <patch> the loader/drawImage wiring so the game draws them.\n"
            "=========================================================="
        )



    def _apply_initial_goal_scoping(self, goal: str, build_msg: str) -> str:
        """Apply scoped constraints + SCOPED-CHANGE directive when the
        initial goal carries strict-scope language ("no other changes",
        "just X", "only Y"). Returns the (possibly augmented)
        build_msg. MK trace 20260517_220025 motivation: goal said
        "ROTATING just the CPU punch horizontally make NO other
        changes" but scoping only kicked in for mid-session feedback,
        not the very first build turn.
        """
        if not goal:
            return build_msg
        # Seed-only widening (2026-06-25 trace 161300_697573): a SEED edit
        # whose goal is a localized tweak (move/resize/rotate/recolor) but
        # carries no explicit "only/just/nothing else" phrase still must be
        # scoped — otherwise a weak model rewrites the whole large seed and
        # loops. Gated on `self.seed_file` so fresh-from-scratch first builds
        # (which have no file to patch) are completely unchanged. Safe because
        # `_seed_html_for_prompt` now injects the full seed so the patch
        # target is visible, and the lock auto-clears after iter 1.
        small_scope_seed = bool(
            self.seed_file is not None
            and self._current_file
            and _goal_is_small_scope_edit(goal)
        )
        if not (
            _feedback_is_strict_scope(goal)
            or _feedback_locks_code(goal)
            or small_scope_seed
        ):
            return build_msg
        # `locks_code=True` is what arms the single_patch lock in
        # `_configure_scoped_constraints`; a small-scope seed edit qualifies
        # even without explicit code-lock language.
        locks_code = _feedback_locks_code(goal) or small_scope_seed
        asset_names = list(self._session_assets.keys()) if self._session_assets else []
        sound_names = list(self._session_sounds.keys()) if self._session_sounds else []
        art_change = bool(asset_names) and _feedback_is_art_change(goal, asset_names)
        sound_change = bool(sound_names) and _feedback_is_sound_change(goal, sound_names)
        # Seed multi-part code tweak (2026-06-25 trace 194238_248190): a
        # move/rotate/resize/recolor goal is a CODE patch even when it names a
        # sprite ("rotate the guns" matched seed gun assets → media_only →
        # every code patch rejected). Force code mode so <patch> stays allowed,
        # zero out the false art/sound signal, and raise the patch budget so a
        # multi-part goal ("move buttons + add upgrade/sell + rotate guns")
        # isn't rejected by the single-tweak max of 1.
        seed_edit_kwargs: dict[str, Any] = {}
        if small_scope_seed:
            art_change = False
            sound_change = False
            seed_edit_kwargs = {
                "max_patch_count": _SEED_EDIT_MAX_PATCHES,
                "force_code_mode": True,
                # Fix A: never reject seed-edit work on patch count — a
                # multi-part goal can exceed any fixed cap. Fix B: mark the
                # lock so the violation handlers can escape after thrash.
                "allow_multi_patch": True,
                "is_seed_edit": True,
            }
        self._configure_scoped_constraints(
            joined_feedback=goal,
            locks_code=locks_code,
            art_change=art_change,
            sound_change=sound_change,
            **seed_edit_kwargs,
        )
        self._trace({
            "kind": "initial_goal_scoping_applied",
            "locks_code": locks_code,
            "art_change": art_change,
            "sound_change": sound_change,
            "small_scope_seed": small_scope_seed,
            "scoped_mode": (self._scoped_constraints or {}).get("mode"),
        })
        # Prepend a compact SCOPED-CHANGE notice so the model sees the
        # narrowed contract right at iter 1. Full SCOPED-CHANGE block
        # also fires automatically on any subsequent feedback turn.
        scoped_mode = (self._scoped_constraints or {}).get("mode") or "single_patch"
        scoped_max_patch = int((self._scoped_constraints or {}).get("max_patch_count") or 1)
        if scoped_mode == "media_only":
            mode_label = "MEDIA-ONLY (existing names only)"
        elif scoped_max_patch > 1:
            # Multi-part seed edit: patches allowed (one per change), but NO
            # full <html_file> rewrite of the working seed.
            mode_label = (
                "PATCH-ONLY (no full <html_file> rewrite; emit one small "
                f"<patch> per change, up to {scoped_max_patch})"
            )
        else:
            mode_label = "ONE-SMALL-PATCH (no rewrite, max one patch block)"
        scoped_notice = (
            "================ INITIAL-GOAL SCOPE LOCK ================\n"
            "Your goal above is a scoped edit of the EXISTING file. Treat\n"
            "this entire session as scoped:\n"
            f"  - Turn mode: {mode_label}\n"
            "  - Address ONLY what the goal explicitly names.\n"
            "  - Do NOT touch unrelated functions, variables, or files.\n"
            "  - Do NOT refactor or 'improve' anything not asked for.\n"
            "==========================================================="
        )
        return f"{scoped_notice}\n\n{build_msg}"



    def _derive_allowed_forbidden_tags(self) -> tuple[list[str], list[str]]:
        """Compute the turn's allowed/forbidden output tags from the
        current scoped/rewrite state. Used by the `turn_contract`
        trace event for visibility; does NOT enforce — enforcement
        lives in `_scoped_reply_violation` and downstream materializer.
        """
        scoped = self._scoped_constraints or {}
        mode = scoped.get("mode")
        if mode == "media_only":
            return (
                ["<assets>", "<sounds>"],
                ["<patch>", "<html_file>"],
            )
        if mode == "single_patch":
            return (
                ["<patch>"],
                ["<html_file>"],
            )
        # No scoped lock: default workflow rules.
        if self._snapshot_n == 0:
            # First build: complete file expected.
            return (["<html_file>"], [])
        # Mid-session: patches preferred; rewrite only when armed.
        # Phase 2: also allow rewrite when the on-disk baseline is itself
        # degenerate — `_materialize` already opens this carve-out via
        # `_is_degenerate_baseline`, but `turn_contract` was reporting
        # rewrite_allowed=False which left the model guessing why its
        # rewrites were getting rejected. Surface the carve-out here so
        # the model and the trace agree.
        allowed = ["<patch>"]
        forbidden: list[str] = []
        rewrite_via_arm = bool(self._allow_one_rewrite)
        rewrite_via_degenerate = False
        try:
            rewrite_via_degenerate = bool(
                self._current_file
                and _is_degenerate_baseline(self._current_file)
            )
        except Exception:
            rewrite_via_degenerate = False
        if rewrite_via_arm or rewrite_via_degenerate:
            allowed.append("<html_file>")
        else:
            forbidden.append("<html_file>")
        # Honor art intent on feedback (GLM-5.2 trace 20260625_124038): when
        # the LLM feedback router judged the user wants NEW art this batch
        # (allow_assets_block=true) — or an unhonored art request is still
        # outstanding — `<assets>` MUST be a usable tag this turn even on a
        # lean/small prompt that dropped it from the goal-time output-tags
        # list (the original goal had no art keywords). Surface it as allowed
        # so the turn contract and the model agree the asset pipeline is open.
        # Genre-agnostic: keys off the router decision, not any game type.
        _route = getattr(self, "_feedback_route", None)
        _wants_art = bool(_route and _route.get("allow_assets_block")) or bool(
            getattr(self, "_unhonored_asset_request", None)
        )
        if _wants_art and "<assets>" not in allowed:
            allowed.append("<assets>")
        return (allowed, forbidden)



    def _clear_scoped_constraints(self) -> None:
        self._scoped_constraints = None
        self._pending_scoped_check_keywords = []
        self._scoped_violation_streak = 0  # Fix B: reset thrash counter



    def _previous_iter_clean_for_scope_guard(self) -> bool:
        prev = self._previous_report or {}
        probes = prev.get("probes") or []
        probes_all_pass = bool(probes) and all(bool(p.get("ok")) for p in probes)
        return (
            self._previous_report_ok is True
            and probes_all_pass
            and not (prev.get("errors") or [])
            and not (prev.get("soft_warnings") or [])
            and not (prev.get("page_errors") or [])
            and not (prev.get("console_errors") or [])
        )



    def _configure_scoped_constraints(
        self,
        *,
        joined_feedback: str,
        locks_code: bool,
        art_change: bool,
        sound_change: bool,
        max_patch_count: int = 1,
        force_code_mode: bool = False,
        allow_multi_patch: bool = False,
        is_seed_edit: bool = False,
    ) -> None:
        """Persist deterministic scoped routing for the next model reply.

        max_patch_count: how many <patch> blocks the turn may emit (default
        1 for mid-session tweaks). A multi-part INITIAL seed-edit goal
        ("move the buttons + add upgrade/sell + rotate the guns") needs
        several patches, so the seed path raises this.

        force_code_mode: when True, never route to media_only — a
        move/resize/rotate/recolor seed tweak is a CODE patch even when it
        names a sprite, so <patch> must stay allowed (2026-06-25 trace
        194238_248190: a goal naming "guns"/"weapon" matched seed asset
        names → media_only → 8 valid code patches were all rejected and
        nothing was saved).

        allow_multi_patch: when True, NEVER reject a reply on patch count —
        a seed-edit multi-part goal can need more patches than any fixed cap,
        so the count gate in _scoped_reply_violation is skipped (supersedes
        the brittle max_patch number). Mid-session tweaks leave this False.

        is_seed_edit: marks the lock as an INITIAL seed-edit lock so the
        violation handlers can apply a forward-progress escape hatch (Fix B)
        without touching mid-session feedback behavior.
        """
        if not locks_code:
            self._clear_scoped_constraints()
            return
        behavior_scope = _feedback_mentions_scoped_behavior_change(joined_feedback)
        size_scope = _feedback_requests_size_change(joined_feedback)
        # Preserve working baselines on strict scoped tweaks: default to one
        # small patch unless this is an explicit style-only media request.
        media_only = (
            bool(art_change or sound_change)
            and not behavior_scope
            and not size_scope
            and not force_code_mode
        )
        mode = "media_only" if media_only else "single_patch"
        max_patch_count = max(1, int(max_patch_count))
        probe_keywords = (
            _scoped_probe_keywords(joined_feedback)
            if mode == "single_patch" and behavior_scope
            else []
        )
        self._scoped_constraints = {
            "mode": mode,
            "max_patch_count": max_patch_count,
            "allow_multi_patch": bool(allow_multi_patch),
            "is_seed_edit": bool(is_seed_edit),
            "allowed_asset_names": sorted(self._session_assets.keys()),
            "allowed_sound_names": sorted(self._session_sounds.keys()),
            "media_name_lock": mode == "media_only",
            "require_scope_probe": bool(probe_keywords),
            "probe_keywords": probe_keywords,
            "preserve_baseline": self._previous_iter_clean_for_scope_guard(),
            "feedback_preview": joined_feedback[:220],
        }
        self._pending_scoped_check_keywords = list(probe_keywords)
        self._trace({
            "kind": "scoped_constraints_set",
            "mode": mode,
            "max_patch_count": max_patch_count,
            "allow_multi_patch": bool(allow_multi_patch),
            "is_seed_edit": bool(is_seed_edit),
            "require_scope_probe": bool(probe_keywords),
            "probe_keywords": probe_keywords,
            "media_name_lock": mode == "media_only",
            "preserve_baseline": self._scoped_constraints["preserve_baseline"],
            "art_change": bool(art_change),
            "sound_change": bool(sound_change),
            "behavior_scope": behavior_scope,
            "size_scope": size_scope,
            "force_code_mode": bool(force_code_mode),
        })



    def _scoped_reply_violation(self, reply: str) -> str | None:
        """Return a compact deterministic violation message, else None."""
        cfg = self._scoped_constraints
        if not cfg:
            return None
        stripped = (reply or "").lstrip()
        if not stripped:
            return "SCOPED FORMAT: empty reply; emit required tag only."
        low = stripped.lower()
        has_assets = bool(_ASSETS_OPEN_RE.search(low))
        has_sounds = bool(_SOUNDS_OPEN_RE.search(low))
        patches = extract_patches(reply)
        patch_count = len(patches)
        has_html = self._extract_html(reply) is not None
        # A prose preamble or leading <think> block is COSMETIC, not a scope
        # violation: extract_patches()/_extract_html() locate blocks ANYWHERE
        # in the reply, so a valid fix must NOT be discarded just because the
        # model led with reasoning prose (GLM always does). Holochess trace
        # 20260623_204052: the model emitted <diagnose> + 2 valid <patch>
        # blocks behind an 880-char prose preamble and the ENTIRE fix was
        # thrown away over the leading-character check → "it won't even do a
        # simple FIX". Only reject for preamble/<think> when there is NO
        # usable patch/media/html to extract (a genuine non-answer).
        if not (patch_count or has_assets or has_sounds or has_html):
            if low.startswith("<think"):
                return "SCOPED FORMAT: do not emit <think>; start with required tag."
            if not stripped.startswith("<"):
                return "SCOPED FORMAT: no prose preamble; start with required tag."
        mode = str(cfg.get("mode") or "single_patch")
        # Be permissive about which tag the model chose. The classifier
        # that picked `media_only` vs `single_patch` reads the user's
        # English feedback through a regex pattern — it's necessarily
        # incomplete (natural language is unbounded). When the model
        # picked a different tag than we expected, it usually has a
        # reason: e.g., DOOM trace 20260523_171650 iter 4 — user wrote
        # "do not change any graphics... just shift the pistol 15 px
        # right" which scored as art_change → media_only, but the model
        # correctly chose a CSS <patch>. Rejecting that patch with
        # "SCOPED MEDIA: emit <assets>/<sounds> only" sent the model
        # into a doom loop where it regenerated the gun asset with a
        # wrong prompt. Auto-revert is the safety net for genuinely bad
        # patches; the scope gate's only remaining job is to block full
        # <html_file> rewrites on small-scope feedback (a single patch
        # is hard to misjudge in a way auto-revert can't catch).
        if has_html:
            return "SCOPED: full <html_file> rewrite blocked; send one <patch>."
        if not (patch_count or has_assets or has_sounds):
            return "SCOPED: emit one <patch>, or <assets>/<sounds> with existing names."
        max_patch = int(cfg.get("max_patch_count") or 1)
        # Fix A: a seed-edit lock (allow_multi_patch) never rejects on count —
        # a multi-part goal can need more patches than any fixed cap, and
        # auto-revert remains the safety net. Mid-session locks keep the cap.
        if not cfg.get("allow_multi_patch") and patch_count > max_patch:
            # Wording matches the historical "SCOPED PATCH: send exactly
            # one <patch>" message so existing tests keep working.
            return f"SCOPED: send exactly one <patch> (got {patch_count})."
        # Soft trace when the model's tag choice diverges from the
        # classifier's expectation. Useful for postmortem to spot
        # patterns without us having to enumerate them in regex.
        if mode == "media_only" and patch_count > 0 and not (has_assets or has_sounds):
            self._record_classifier_overrule(
                expected_mode=mode,
                model_emitted="patch_only",
                feedback_preview=str(cfg.get("feedback_preview") or "")[:200],
            )
        elif mode == "single_patch" and (has_assets or has_sounds) and patch_count == 0:
            self._record_classifier_overrule(
                expected_mode=mode,
                model_emitted="media_only",
                feedback_preview=str(cfg.get("feedback_preview") or "")[:200],
            )
        if bool(cfg.get("require_scope_probe")):
            probes = self._extract_probes(reply)
            if not probes:
                return "SCOPED CHECK: include one compact <probes> check for requested behavior."
            keys = [str(k).lower() for k in (cfg.get("probe_keywords") or []) if k]
            if keys:
                probe_text = " ".join(
                    f"{p.get('name','')} {p.get('expr','')}".lower()
                    for p in probes
                )
                if not any(k in probe_text for k in keys):
                    want = ", ".join(keys[:3])
                    return (
                        "SCOPED CHECK: probe must mention requested behavior "
                        f"(e.g. {want})."
                    )
        return None



    def _scoped_retry_instruction(self, violation: str) -> str:
        cfg = self._scoped_constraints or {}
        probe_line = ""
        if cfg.get("require_scope_probe"):
            probe_line = " Include one compact <probes> entry verifying the requested behavior."
        return (
            f"{violation}\n"
            "Retry: emit either one small <patch> (preserve the working "
            "baseline — change only the scoped request) OR an <assets>/"
            "<sounds> block re-using EXISTING names."
            f"{probe_line}"
        )



    def _apply_scoped_check_to_report(self, report: dict[str, Any]) -> None:
        keys = [k for k in self._pending_scoped_check_keywords if k]
        if not keys:
            return
        probes = report.get("probes") or []
        matched = [
            p for p in probes
            if any(
                key in (
                    f"{str(p.get('name') or '').lower()} "
                    f"{str(p.get('expr') or '').lower()}"
                )
                for key in keys
            )
        ]
        passed = any(bool(p.get("ok")) for p in matched)
        report["scoped_check"] = {
            "required": True,
            "keywords": keys[:3],
            "matched": len(matched),
            "pass": passed,
        }
        if not passed:
            sw = list(report.get("soft_warnings") or [])
            sw.append(
                "SCOPED CHECK FAILED: requested behavior change was not "
                "verified by a passing scoped probe."
            )
            report["soft_warnings"] = sw
            report["ok"] = False



    def _consumed_feedback_summary(self) -> str | None:
        if self._should_defer_feedback_for_blocker():
            return (
                "→ queued your input, but deferring it until the current "
                "test blocker is fixed."
            )
        bits: list[str] = []
        if self._pending_answer is not None:
            ans = self._pending_answer
            bits.append(f"answer: {ans[:80]!r}")
        if self._pending_feedback:
            for fb in self._pending_feedback:
                bits.append(f"feedback: {fb[:80]!r}")
        if not bits:
            return None
        return "→ applying your input to next turn: " + "; ".join(bits)

    # Phrases that count as a user override of the blocker-first deferral.
    # The narrow explicit forms ("ignore the test", "ship as-is") are joined
    # by natural-language signals — the chess trace had the user say "the
    # game works great" / "the game plays fine" / "do not change the GAME"
    # three times and none matched the original regex, so all three asks
    # were deferred. False positives here are safe: an override only stops
    # the deferral, it does not silence the test report the model still
    # sees. False negatives are the actual user-facing failure.
    _BLOCKER_OVERRIDE_RE = re.compile(
        r"\b("
        r"ignore\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"override\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"abandon\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"skip\s+(?:the\s+)?(?:test|tests|harness|failure|blocker|error)|"
        r"ship\s+(?:it\s+)?(?:as[- ]is|anyway)|"
        r"ship\s+the\s+partial|"
        r"accept\s+(?:the\s+)?(?:known\s+)?failure|"
        r"(?:the\s+)?(?:game|it)\s+(?:works|plays|runs)"
        r"(?:\s+(?:great|fine|well|now|again|already|just\s+fine))?|"
        r"(?:the\s+)?(?:game|it)\s+is\s+(?:fine|great|working|playing|ok|okay)|"
        r"do(?:es)?\s*n[o']?t?\s+change\s+(?:the\s+)?"
        r"(?:game|code|behavior|gameplay|logic|mechanics)|"
        r"(?:leave|keep)\s+(?:the\s+)?(?:game|code|gameplay)\s+"
        r"(?:as[- ]is|alone|the\s+same)|"
        r"perfectly\s+working\s+(?:game|build)|"
        r"working\s+(?:perfectly|fine|great)"
        r")\b",
        re.I,
    )



    def _has_active_blocker(self) -> bool:
        """True when the previous tested build failed and the next turn is a fix."""
        return bool(self._fix_mode and self._previous_report_ok is False)



    def _route_forces_fix_mode(self) -> bool:
        """Force precision fix mode for a feedback turn that reports a real
        gameplay/code bug, even when the prior harness report was clean.

        The harness cannot catch every gameplay bug (e.g. "the enemy walks
        through walls" on a build whose probes all passed). When the feedback
        router classifies the ask as a `behavior_bug` / `code_fix` to honor
        this turn, run the next turn in `_fix_mode` so it samples at the
        precision temperature and takes the diagnose-then-fix path instead of
        treating the message as cosmetic polish. Returns True if it flipped
        `_fix_mode` on. (Track D — behavior-bug feedback enters fix mode.)
        """
        route = getattr(self, "_feedback_route", None)
        if not isinstance(route, dict):
            return False
        intent = str(route.get("primary_intent", "")).strip()
        if intent in ("behavior_bug", "code_fix") and route.get("honor_user_now", True):
            if not self._fix_mode:
                self._fix_mode = True
                return True
        return False



    def _should_defer_feedback_for_blocker(self) -> bool:
        """Keep fresh feedback from distracting a failing repair turn.

        The queue is left intact so the exact feedback is applied after a clean
        report. Only explicit "ignore/ship anyway" style overrides are allowed
        through while a blocker is active.

        Deferral reform (chess-trace fix 2026-06-22): the LLM router decides
        timing FIRST. When it says `honor_user_now` (and not
        `defer_behind_blocker`), do NOT defer — the chess trace deferred
        "no new assets, just show the full screen the bottom row is cut off"
        behind a stale blocker for three turns because the regex override
        list didn't match "no new assets" / "cut off". The regex below stays
        as the fallback when the router didn't run / failed to parse.
        """
        if not (self._pending_feedback and self._has_active_blocker()):
            return False
        route = getattr(self, "_feedback_route", None)
        if route is not None:
            if route.get("defer_behind_blocker"):
                return True
            if route.get("honor_user_now", True):
                return False
        joined = "\n".join(self._pending_feedback)
        return self._BLOCKER_OVERRIDE_RE.search(joined) is None

    @staticmethod


    @staticmethod
    def _feedback_shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(words) < n:
            return {tuple(words)} if words else set()
        return {tuple(words[i:i + n]) for i in range(0, len(words) - n + 1)}

    @staticmethod


    @staticmethod
    def _feedback_keywords(text: str) -> set[str]:
        stop = {
            "the", "a", "an", "to", "and", "or", "but", "it", "is", "are",
            "be", "of", "for", "with", "when", "still", "need", "needs",
            "make", "also", "not", "just", "this", "that", "their",
        }
        return {
            w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in stop
        }



    def _deferral_signature(self, text: str) -> str:
        """Compact keyword signature for cross-turn deferral matching.

        Keyword bag drops stop-words and length-2 tokens (`_feedback_keywords`).
        Sorted + joined into a stable key. Two phrasings of the same ask
        ("make new pawn frames" vs "more frames of the pawn please") will
        produce highly overlapping signatures.
        """
        kw = self._feedback_keywords(text)
        return " ".join(sorted(kw))



    def _count_recent_deferrals(self, text: str) -> int:
        """How many recent deferrals share intent with `text`."""
        if not self._recent_deferred_signatures:
            return 0
        cur_set = set(self._deferral_signature(text).split())
        if not cur_set:
            return 0
        count = 0
        for prev_sig in self._recent_deferred_signatures:
            prev_set = set(prev_sig.split())
            if not prev_set:
                continue
            union = cur_set | prev_set
            overlap = len(cur_set & prev_set) / max(1, len(union))
            if overlap >= 0.5:
                count += 1
        return count



    def _detect_repeated_feedback(self, text: str) -> dict | None:
        """Return overlap metadata when latest feedback repeats a recent ask."""
        cur = self._feedback_shingles(text)
        cur_kw = self._feedback_keywords(text)
        if not cur:
            return None
        best: tuple[float, int, set[str]] | None = None
        for idx, prev in enumerate(self._recent_feedback_texts[-2:]):
            prev_set = self._feedback_shingles(prev)
            if not prev_set:
                continue
            union = cur | prev_set
            shingle_overlap = len(cur & prev_set) / max(1, len(union))
            prev_kw = self._feedback_keywords(prev)
            kw_union = cur_kw | prev_kw
            kw_overlap = len(cur_kw & prev_kw) / max(1, len(kw_union))
            overlap = max(shingle_overlap, kw_overlap)
            shared = {" ".join(s) for s in (cur & prev_set)}
            shared.update(cur_kw & prev_kw)
            if best is None or overlap > best[0]:
                # idx here is relative to the final two entries.
                best = (overlap, len(self._recent_feedback_texts[-2:]) - idx, shared)
        if not best or best[0] < 0.5:
            return None
        shared_words: list[str] = []
        for term in sorted(best[2])[:5]:
            if term:
                shared_words.append(term)
        return {
            "overlap": round(best[0], 3),
            "overlap_with_recent_turns_ago": best[1],
            "shared_terms": shared_words,
        }



    def _maybe_clear_asset_reprompt_via_code(self, reply: str) -> None:
        """Stand down the ASSET GENERATION REQUIRED reprompt when the
        model substantively addressed the request in CODE.

        Evidence (trace 20260612_132314): "need to improve animation for
        jump,duck, left and right" hard-classified as an art request, but
        the model — correctly, since img2img cannot change a pose —
        shipped procedural-animation patches. The reprompt only cleared
        on an <assets> block, so it nagged for 3 turns and gave up.

        Heuristic: the reply's applied patch bodies share subject
        keywords with the request (>= 2 terms, or all of them for a
        one/two-keyword request). Replies that neither emit <assets> nor
        touch the requested subject keep the reprompt — the original
        here-s-a-tight-test failure mode stays covered.

        GLM-5.2 trace 20260625_124038: the model emitted an <assets> block
        that FAILED to parse (wrapped in `{"sprites":[...]}`), so Z-Image
        never ran — yet the patches wired up sprite paths and shared the
        keywords "sprites"/"enemies", so this method cleared the reprompt
        and the missing art was masked. Guard against that: when the reply
        ATTEMPTED an <assets> block, the keyword-overlap clear does NOT
        apply. Successful generation already clears the reprompt upstream
        in `_maybe_generate_assets_and_sounds`; if we still hold an
        outstanding request here AND the reply emitted <assets>, the
        generation failed and we MUST keep nagging (so the model retries
        with a valid bare-array block). Only a pure code-only solution
        (no <assets> attempted) may stand down via keyword overlap.
        """
        req = getattr(self, "_unhonored_asset_request", None)
        if not req:
            return
        # If the model attempted an <assets> block this turn, do not clear
        # via code overlap — see docstring (failed-generation masking fix).
        if _ASSETS_OPEN_RE.search(reply or ""):
            return
        try:
            patch_list = extract_patches(reply or "")
        except Exception:
            patch_list = []
        if not patch_list:
            return
        req_kw = self._feedback_keywords(req)
        if not req_kw:
            return
        patch_text = "\n".join(
            f"{p.search or ''}\n{p.replace or ''}" for p in patch_list
        )
        patch_kw = self._feedback_keywords(patch_text)
        overlap = req_kw & patch_kw
        if len(overlap) >= min(2, len(req_kw)):
            self._unhonored_asset_request = None
            self._asset_reprompt_count = 0
            self._trace({
                "kind": "asset_request_honored_via_code",
                "request": req[:200],
                "matched_terms": sorted(overlap)[:8],
                "patch_count": len(patch_list),
            })

    @staticmethod


    @staticmethod
    def _is_post_clean_instruction(base_message: str) -> bool:
        """True when `base_message` is the clean-report post-clean prompt.

        Fresh user feedback after a clean pass should not drag a full test
        report and stale "prefer <done/>" text into the next model turn.
        The local MK trace showed this confusing small models after the
        user asked for one narrow missing-animation fix.
        """
        if not base_message:
            return False
        return (
            "No errors. The game works. STRONGLY prefer ending with <done/>"
            in base_message
        )



    def _compact_post_clean_context(self) -> str:
        prev = self._previous_report or {}
        probes = prev.get("probes") or []
        passed = sum(1 for p in probes if p.get("ok"))
        total = len(probes)
        return (
            "PREVIOUS BUILD WAS CLEAN: "
            f"{passed}/{total} probes passed, no page errors, no console errors. "
            "Treat that version as the baseline."
        )

    # ---- LLM Feedback Router (chess-trace fix 2026-06-22) --------------
    # Valid `primary_intent` values the router may return. Kept genre-free:
    # they describe what the user wants DONE (code vs media vs ship), never
    # a game type. The deterministic mapping in `_flush_user_injections`
    # turns these into the existing directive paths.
    _FEEDBACK_ROUTER_INTENTS = (
        "code_fix",
        "wire_existing_media",
        "generate_new_assets",
        "img2img_chain",
        "style_rebrand",
        "ui_feature",
        "behavior_bug",
        "ship_override",
    )

    @staticmethod


    def _feedback_route_cache_key(
        feedback_texts: list[str], asset_count: int, last_report_ok: bool | None,
    ) -> str:
        """Stable key so a re-flush in the same turn reuses the route.

        Keyed on the joined feedback text + the session asset count + the
        last report's ok flag — the three inputs that change the routing
        decision. Same batch ⇒ same key ⇒ one LLM call per turn.
        """
        import hashlib as _hashlib
        joined = "\u0001".join(feedback_texts)
        raw = f"{joined}|{asset_count}|{last_report_ok}".encode("utf-8", "ignore")
        return _hashlib.sha1(raw).hexdigest()[:16]



    async def _route_user_feedback_llm(
        self, feedback_texts: list[str],
    ) -> dict | None:
        """Interpret a pending user-feedback batch into a routing decision.

        Returns a dict with keys (`primary_intent`, `honor_user_now`,
        `allow_assets_block`, `allow_patch`, `defer_behind_blocker`,
        `user_visible_issue`, `harness_blocker_ack`, `confidence`) or None
        when the LLM is unavailable / the reply doesn't parse — in which
        case `_flush_user_injections` falls back to the regex classifiers.

        Design: LLM routes (nuanced intent), hard rules enforce output
        shape elsewhere. This replaces regex-as-authority for the
        art/code/media/defer decision so any natural phrasing ("no new
        assets, just show the full screen", "Bug: row clipped") routes
        correctly without magic words. See the chess trace
        a-game-of-chess-player-vs-cpu_20260621_193434.
        """
        if not feedback_texts:
            return None
        backend = self.get_backend("architect")
        if backend is None:
            backend = self._backend
        if backend is None:
            return None
        asset_names = list((self._session_assets or {}).keys())[:20]
        prev = self._previous_report or {}
        soft_warnings = [
            str(w)[:120] for w in (prev.get("soft_warnings") or [])[:3]
        ]
        quarantined = list(getattr(self, "_quarantined_probe_names", []) or [])[:5]
        unhonored = getattr(self, "_unhonored_asset_request", None)
        joined_feedback = "\n- ".join(feedback_texts)
        sys_prompt = (
            "You route a user's feedback on a single-file HTML5 game to the "
            "right action. The user's words are AUTHORITATIVE — never override "
            "them. Decide ONLY routing/timing, then reply with ONE JSON object "
            "and nothing else (no prose, no fences). Schema:\n"
            "{\n"
            '  "primary_intent": one of '
            f"{list(self._FEEDBACK_ROUTER_INTENTS)},\n"
            '  "honor_user_now": bool,   // address the user THIS turn (true '
            "unless they explicitly said to wait)\n"
            '  "allow_assets_block": bool,  // true ONLY if the user wants NEW '
            "art generated (generate_new_assets / img2img_chain / style_rebrand)\n"
            '  "allow_patch": bool,\n'
            '  "defer_behind_blocker": bool,  // almost always false; true only '
            "if the user said to fix the test/blocker first\n"
            '  "user_visible_issue": short string or "",\n'
            '  "harness_blocker_ack": short string or "",\n'
            '  "confidence": 0..1\n'
            "}\n"
            "Rules: 'use/load/wire the existing sprites', 'show the full "
            "screen', 'bottom row cut off', 'make the board bigger' are "
            "code_fix or wire_existing_media with allow_assets_block=false. "
            "'no new assets/images' MUST set allow_assets_block=false. Only an "
            "explicit request for NEW or restyled art sets allow_assets_block=true."
        )
        user_msg = (
            f"GAME GOAL: {(self._goal or '')[:200]}\n"
            f"SESSION ASSETS ({len(asset_names)}): {', '.join(asset_names) or 'none'}\n"
            f"LAST REPORT OK: {prev.get('ok')}\n"
            f"TOP SOFT WARNINGS: {soft_warnings or 'none'}\n"
            f"QUARANTINED PROBES: {quarantined or 'none'}\n"
            f"OUTSTANDING ASSET REQUEST: {(unhonored or 'none')[:160]}\n"
            "\nUSER FEEDBACK (verbatim):\n"
            f"- {joined_feedback}\n"
            "\nReply with the JSON object only."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        # Bounded like the format-doctor: a router that hangs would stall
        # the feedback turn it is supposed to speed up.
        router_stall = min(self.stall_seconds, 60.0)
        router_overall = min(self.overall_seconds, 120.0)
        self._trace({
            "kind": "feedback_router_start",
            "count": len(feedback_texts),
            "preview": joined_feedback[:200],
        })
        try:
            result = await backend.stream_chat(
                messages,
                on_token=None,
                options={"temperature": 0.1, "num_ctx": self.num_ctx},
                keep_alive=self._keep_alive_for_backend(backend),
                stall_seconds=router_stall,
                overall_seconds=router_overall,
                max_retries=0,
                cancel_event=self._ensure_stop_event(),
            )
            text = (getattr(result, "text", "") or "").strip()
        except Exception as e:
            self._trace_exception("feedback_router_error", e)
            return None
        route = self._parse_feedback_route_json(text)
        if route is None:
            self._trace({
                "kind": "feedback_router_parse_failed",
                "preview": text[:240],
            })
            return None
        self._trace({"kind": "feedback_router_decision", **route})
        # Phase 4 (4D.1): remember the routed intent so iter_summary carries it
        # (one-row view of "what did the harness think the user wanted?").
        self._last_router_intent = route.get("primary_intent")
        return route



    def _parse_feedback_route_json(self, text: str) -> dict | None:
        """Extract + validate the router JSON object. Tolerant of fences
        and surrounding prose; returns None on any malformed reply."""
        if not text:
            return None
        import json as _json
        candidate = text
        # Strip code fences if the model wrapped the JSON.
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                candidate = brace.group(0)
        try:
            obj = _json.loads(candidate)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        intent = str(obj.get("primary_intent", "")).strip()
        if intent not in self._FEEDBACK_ROUTER_INTENTS:
            return None

        def _as_bool(v, default):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes")
            return default

        return {
            "primary_intent": intent,
            "honor_user_now": _as_bool(obj.get("honor_user_now"), True),
            "allow_assets_block": _as_bool(obj.get("allow_assets_block"), False),
            "allow_patch": _as_bool(obj.get("allow_patch"), True),
            "defer_behind_blocker": _as_bool(obj.get("defer_behind_blocker"), False),
            "user_visible_issue": str(obj.get("user_visible_issue", ""))[:200],
            "harness_blocker_ack": str(obj.get("harness_blocker_ack", ""))[:200],
            "confidence": float(obj.get("confidence", 0.0))
            if isinstance(obj.get("confidence"), (int, float)) else 0.0,
        }



    async def _precompute_feedback_route(self) -> None:
        """Compute (and cache) the router decision for the pending batch.

        Called from the async run loop just before a user-turn message is
        assembled (the `_flush_user_injections` chokepoints). Cheap on a
        cache hit. No-op when there is no pending feedback, when raw-feedback
        mode is off-limits, or when directives have been auto-disabled after
        repeated overrules (we honor the existing escape hatch).
        """
        feedback_texts = [
            fb for fb in (self._pending_feedback or [])
            if fb not in getattr(self, "_internal_feedback_texts", set())
        ]
        if not feedback_texts:
            self._feedback_route = None
            self._feedback_route_key = None
            return
        asset_count = len(self._session_assets or {})
        key = self._feedback_route_cache_key(
            feedback_texts, asset_count, self._previous_report_ok,
        )
        if key == self._feedback_route_key and self._feedback_route is not None:
            return  # already routed this exact batch this turn
        route = await self._route_user_feedback_llm(feedback_texts)
        self._feedback_route = route
        self._feedback_route_key = key if route is not None else None



    def _flush_user_injections(self, base_message: str) -> str:
        """Main router: drain queued user input, inject directives, and
        record the routing decisions consumed by `_stream` to emit one
        `turn_contract` trace row per stream. False positives in the
        classifier helpers (`_feedback_is_art_change`,
        `_feedback_is_sound_change`, `_feedback_is_orientation_change`)
        flip allowed/forbidden output tags and are expensive — keep
        them genre-free and prefer false negatives over false positives.
        """
        parts: list[str] = []
        self._last_drained_feedback = []
        # Reset routing record so a turn without feedback shows up as
        # "no feedback" rather than inheriting the previous turn's flags.
        # Fields are filled in below as decisions are made.
        self._last_turn_contract = {
            "had_feedback": False,
            "locks_code": False,
            "art_change": False,
            "sound_change": False,
            "behavior_scope": False,
            "behavior_bug": False,
            "orientation_change": False,
            "blocker_feedback_deferred": False,
        }
        # Snapshot the queue BEFORE consuming so we can push a visible
        # "✓ APPLIED to this turn" confirmation into the agent log via
        # the TUI's token callback. Without this, only the right-hand
        # status panel reflects the queue draining — the left-hand log
        # (where the user's eye lives) shows nothing, leaving them
        # uncertain whether their typing actually reached the model.
        # Phase 0.1 — partition feedback before the deferral decision so
        # MEDIA-ONLY items (art/sound regen, animation frame chains via
        # `from_image`) bypass the blocker-first deferral. They run on
        # GPU 0 / Z-Image / Stable-Audio / SD-Turbo; the coder runs on
        # slot 1. They don't compete. The 2026-05-22 chess trace had the
        # user ask three times for img2img animation frames from existing
        # piece sprites; all three were deferred behind an irrelevant
        # `kbCursor` draw warning and the session shipped without ever
        # honoring the asset ask. See memory `feedback-media-requests-
        # never-defer.md`.
        defer_predicate = self._should_defer_feedback_for_blocker()
        _session_assets = getattr(self, "_session_assets", None) or {}
        _session_sounds = getattr(self, "_session_sounds", None) or {}
        asset_names = list(_session_assets.keys())
        sound_names = list(_session_sounds.keys())
        # GENERAL (2026-05-31, genre/model-agnostic): an explicit "generate new
        # art" request only produces sprites if the model emits an <assets>
        # block. A model distracted by a test blocker — or in raw-feedback mode
        # where the media wrapper is suppressed — often replies with code or a
        # diagnosis and NO <assets>, so the art never gets made (here-s-a-tight
        # -test 20260530: asked twice for a red fighter, zero new assets). Track
        # the outstanding request and re-assert "emit <assets>" each turn until
        # the model actually emits one (capped). This is a machine-level
        # directive (NOT suppressed by raw mode) and survives the blocker; it
        # is cleared in _maybe_generate_assets_and_sounds the moment an <assets>
        # block is parsed.
        # getattr defaults so partially-constructed test stubs (which skip
        # __init__) don't AttributeError here.
        _unhonored = getattr(self, "_unhonored_asset_request", None)
        _reprompts = getattr(self, "_asset_reprompt_count", 0)
        # Skip AGENT-generated notices (mid-session media loaders etc.) —
        # only genuine USER feedback can arm the unhonored-asset-request
        # detector. Trace 20260612_004616: the agent's own "Mid-session
        # asset/sound/video additions" notice classified as an art change
        # and fired 8 spurious ASSET GENERATION REQUIRED banners.
        _internal_texts = getattr(self, "_internal_feedback_texts", set())
        # LLM router (chess-trace fix 2026-06-22): when the router ran, IT
        # decides whether new art is wanted — `allow_assets_block`. This
        # demotes the regex `_feedback_is_art_change` to a fallback used
        # only when the router didn't run / failed to parse. The chess
        # trace fired "ASSET GENERATION REQUIRED" against "no new assets,
        # just show the full screen" because the regex saw the asset noun;
        # the router returns code_fix + allow_assets_block=false there.
        _route = getattr(self, "_feedback_route", None)
        if _route is not None:
            if _route.get("allow_assets_block"):
                # Router confirms the user wants NEW art — arm the reprompt
                # with the most recent genuine user feedback item.
                _user_fb = [
                    fb for fb in self._pending_feedback
                    if fb not in _internal_texts
                ]
                if (
                    _user_fb
                    and _feedback_art_request_already_on_disk(
                        _user_fb[-1], asset_names,
                    )
                ):
                    _ref_names = sorted(
                        _feedback_referenced_asset_names(_user_fb[-1], asset_names)
                    )
                    self._trace({
                        "kind": "asset_request_already_satisfied",
                        "names": _ref_names,
                        "feedback_preview": _user_fb[-1][:200],
                    })
                    parts.append(self._wire_existing_assets_coaching_block(
                        _user_fb[-1], asset_names,
                    ))
                    _art_reqs = []
                else:
                    _art_reqs = _user_fb[-1:] if _user_fb else []
            else:
                # Router says NO new art this batch. Clear any stale
                # outstanding request the user has now contradicted
                # (chess trace: attempt-3 reprompt still quoted the iter-2
                # message after the user typed "no new assets").
                _art_reqs = []
                # Phase 0D-6 (Fieldrunners trace 20260626_102307 iter 5): a
                # CONTENT-FREE retry nudge ("try again", "no usable code…")
                # routes as code_fix / allow_assets_block=false, but it does
                # NOT contradict a still-unhonored art request — it is just a
                # retry. Clearing the standing request here dropped the
                # enemy/head art ask and sent iter 5 off without the <assets>
                # scaffold. Only clear when the LATEST genuine user feedback
                # actually references art/asset vocabulary (a real
                # contradiction like "no new assets, just wire them").
                _latest_user_fb = next(
                    (
                        fb for fb in reversed(self._pending_feedback)
                        if fb not in _internal_texts
                    ),
                    None,
                )
                # A "contradiction" is feedback that actually references
                # art/asset vocabulary (so the router's no-new-art decision is
                # a genuine art judgment): an art-change ask, an explicit
                # "no new assets" / "use existing" instruction, or any explicit
                # new-art phrasing. A content-free retry nudge ("try again")
                # matches NONE of these, so the standing request is retained.
                _contradicts_art = bool(
                    _latest_user_fb
                    and (
                        _feedback_is_art_change(_latest_user_fb, asset_names)
                        or _feedback_requests_existing_media(_latest_user_fb)
                        or _feedback_requests_explicit_new_art(_latest_user_fb)
                    )
                )
                if _unhonored is not None and not _contradicts_art:
                    self._trace({
                        "kind": "asset_reprompt_retained_on_retry",
                        "request": str(_unhonored)[:200],
                        "primary_intent": _route.get("primary_intent"),
                        "latest_feedback": str(_latest_user_fb or "")[:120],
                    })
                    # Keep the standing art request armed; re-arm it as the
                    # art request for this turn so the <assets> scaffold fires.
                    _art_reqs = [_unhonored]
                elif _unhonored is not None:
                    self._trace({
                        "kind": "asset_reprompt_cleared_by_router",
                        "request": str(_unhonored)[:200],
                        "primary_intent": _route.get("primary_intent"),
                    })
                    # Phase 4 (4D.2): clearing a standing art request on a vague
                    # retry is a HARNESS_BUG signal (Fieldrunners iter 5) —
                    # record it for failure_class.
                    self._last_asset_reprompt_cleared = True
                    _unhonored = None
                    _reprompts = 0
                    self._unhonored_asset_request = None
                    self._asset_reprompt_count = 0
        else:
            _art_reqs = [
                fb for fb in self._pending_feedback
                if fb not in _internal_texts
                and not _feedback_is_ui_feature(fb)
                and not _feedback_is_behavior_bug(fb)
                and not _feedback_requests_existing_media(fb)
                and _feedback_is_art_change(fb, asset_names)
            ]
        if _art_reqs:
            _candidate = _art_reqs[-1]
            if _feedback_art_request_already_on_disk(_candidate, asset_names):
                _ref_names = sorted(
                    _feedback_referenced_asset_names(_candidate, asset_names)
                )
                self._trace({
                    "kind": "asset_request_already_satisfied",
                    "names": _ref_names,
                    "feedback_preview": _candidate[:200],
                })
                parts.append(self._wire_existing_assets_coaching_block(
                    _candidate, asset_names,
                ))
            else:
                _unhonored = _candidate
                _reprompts = 0
        if _unhonored and _reprompts < 3:
            if _feedback_art_request_already_on_disk(_unhonored, asset_names):
                _ref_names = sorted(
                    _feedback_referenced_asset_names(_unhonored, asset_names)
                )
                self._trace({
                    "kind": "asset_request_already_satisfied",
                    "names": _ref_names,
                    "feedback_preview": str(_unhonored)[:200],
                })
                parts.append(self._wire_existing_assets_coaching_block(
                    _unhonored, asset_names,
                ))
                self._unhonored_asset_request = None
                self._asset_reprompt_count = 0
            else:
                _reprompts += 1
                self._unhonored_asset_request = _unhonored
                self._asset_reprompt_count = _reprompts
                parts.append(
                    "================ ASSET GENERATION REQUIRED ================\n"
                    "The user asked you to GENERATE NEW ART:\n"
                    f'  "{_unhonored[:300]}"\n'
                    "New art does NOT exist until you emit an <assets> block. THIS "
                    "turn, emit <assets> with a NEW `name` for each new sprite and a "
                    "SHORT prompt for each (for a recolor/variant, reuse the base "
                    "entity's description with the requested change — e.g. a second "
                    "fighter that is the same character in a different color, one "
                    "entry per pose). You MAY also <patch> the code to load and draw "
                    "them. Do NOT reply with only a diagnosis or code — the sprites "
                    "will not appear without an <assets> block.\n"
                    "==========================================================="
                )
                self._trace({
                    "kind": "asset_request_reprompt",
                    "attempt": _reprompts,
                    "request": _unhonored[:200],
                })
        elif _unhonored and _reprompts >= 3:
            # Gave it 3 turns; stop nagging so the prompt doesn't bloat.
            self._trace({
                "kind": "asset_request_giveup",
                "request": _unhonored[:200],
            })
            self._unhonored_asset_request = None
            self._asset_reprompt_count = 0
        media_to_process_now: list[str] = []
        ui_to_process_now: list[str] = []
        code_to_defer: list[str] = []
        force_honor_via_escalation: list[str] = []
        if defer_predicate and self._pending_feedback:
            for fb in self._pending_feedback:
                is_art = _feedback_is_art_change(fb, asset_names)
                is_sound = _feedback_is_sound_change(fb, sound_names)
                is_ui_feature = _feedback_is_ui_feature(fb)
                is_bug = _feedback_is_behavior_bug(fb)
                is_orient = _feedback_is_orientation_change(fb)
                visible_issue = bool(re.search(
                    r"\b(pink|missing|not loading|animation|graphics?|sprite|asset|controls?)\b",
                    fb.lower(),
                ))
                if (
                    (is_art or is_sound)
                    and not is_ui_feature
                    and not is_bug
                    and not is_orient
                    and not _feedback_requests_existing_media(fb)
                ):
                    media_to_process_now.append(fb)
                    continue
                # UI-feature requests (help/hint/HUD/menu overlays) are
                # CODE work that, when the failing probes are checking for
                # exactly that UI (`help_button_present`, etc.), ARE the
                # blocker fix — deferring them is circular. Process this
                # turn instead of queuing behind the report. Genre-free:
                # keyed on UI-feature phrasing, not game type (2026-06-21
                # point-and-click seed trace: "add help" deferred behind
                # the very help-button probe it would satisfy).
                if is_ui_feature and not is_art and not is_sound:
                    ui_to_process_now.append(fb)
                    continue
                # Phase 0.2 — same intent deferred ≥2 times before? Force
                # it through. The model is told the user has asked N
                # times; the test report still appears so the blocker
                # isn't silently dropped, but the user's input is no
                # longer swallowed for a third turn in a row.
                prior_defer_count = self._count_recent_deferrals(fb)
                if (
                    prior_defer_count >= 2
                    or (
                        visible_issue
                        and (
                            getattr(self, "_stuck_streak", 0) >= 2
                            or getattr(self, "_format_stuck_streak", 0) >= 1
                        )
                    )
                ):
                    force_honor_via_escalation.append(fb)
                    self._trace({
                        "kind": "feedback_deferral_escalated",
                        "prior_defer_count": prior_defer_count,
                        "visible_issue": visible_issue,
                        "text": fb[:240],
                    })
                else:
                    code_to_defer.append(fb)
            # Rewrite the queue so the downstream "process feedback" block
            # consumes media items + UI-feature items + force-honored items
            # this turn. The plain code items get re-queued at the end of
            # this method so they remain pending for the next turn.
            self._pending_feedback = (
                list(media_to_process_now)
                + list(ui_to_process_now)
                + list(force_honor_via_escalation)
            )
            if media_to_process_now:
                self._trace({
                    "kind": "media_only_parallel_inject",
                    "media_count": len(media_to_process_now),
                    "code_deferred_count": len(code_to_defer),
                    "escalated_count": len(force_honor_via_escalation),
                    "asset_names": asset_names,
                    "sound_names": sound_names,
                    "preview": "\n- ".join(media_to_process_now)[:400],
                })
            if ui_to_process_now:
                self._trace({
                    "kind": "ui_feature_processed_during_blocker",
                    "ui_count": len(ui_to_process_now),
                    "code_deferred_count": len(code_to_defer),
                    "preview": "\n- ".join(ui_to_process_now)[:400],
                })
        defer_block_active = defer_predicate and bool(code_to_defer)

        consumed_items: list[str] = []
        if self._pending_answer is not None:
            consumed_items.append(f"answer: {self._pending_answer[:120]}")
        for fb in self._pending_feedback:
            consumed_items.append(f"feedback: {fb[:120]}")

        if force_honor_via_escalation:
            preview = "\n- ".join(fb[:240] for fb in force_honor_via_escalation)
            parts.append(
                "================ USER FEEDBACK OVERRIDES STALE BLOCKER ================\n"
                "The run is stuck on the same blocker/no-usable pattern, and the "
                "user named a visible issue. For THIS turn, fix the user-visible "
                "feedback first, then rerun/handle the stale harness blocker.\n"
                f"\nPriority feedback:\n- {preview}\n"
                "======================================================================"
            )

        self._feedback_deferred_last_turn = False
        answer_was_consumed = False
        if self._pending_answer is not None:
            ans = self._pending_answer
            parts.append(
                "================ USER ANSWER (HIGHEST PRIORITY) ================\n"
                f"{ans}\n"
                "================================================================"
            )
            self._trace({"kind": "answer_injected", "text": ans})
            self._pending_answer = None
            answer_was_consumed = True
        if defer_block_active:
            self._feedback_deferred_last_turn = True
            feedback_items = list(code_to_defer)
            preview = "\n- ".join(fb[:240] for fb in feedback_items)
            parallel_note = ""
            if media_to_process_now:
                parallel_note = (
                    "\nNOTE: media-only items in the same user feedback "
                    "(art/sound regen / animation frames) are being processed "
                    "in parallel via the diffuser pipeline on GPU 0 this turn. "
                    "The block above lists ONLY the code-touching items that "
                    "wait on the blocker.\n"
                )
            parts.append(
                "================ BLOCKER-FIRST FEEDBACK DEFERRAL ================\n"
                "The previous browser/micro-probe report is still failing, so\n"
                "code-touching user feedback is queued for after the blocker\n"
                "is clean. THIS turn must fix the failing test report below first.\n"
                f"{parallel_note}"
                f"\nDeferred feedback:\n- {preview}\n"
                "================================================================="
            )
            self._last_turn_contract["blocker_feedback_deferred"] = True
            self._trace({
                "kind": "feedback_deferred_blocker",
                "count": len(feedback_items),
                "previous_report_ok": self._previous_report_ok,
                "fix_mode": self._fix_mode,
                "media_parallel_count": len(media_to_process_now),
                "preview": "\n- ".join(feedback_items)[:400],
            })
            # Phase 0.2 — record each deferred item's signature so the
            # next-turn escalation count is accurate.
            for fb in feedback_items:
                sig = self._deferral_signature(fb)
                if sig:
                    self._recent_deferred_signatures.append(sig)
            # Cap the history at 6 entries — enough to span ~3 ask-defer
            # cycles; older signatures shed naturally.
            self._recent_deferred_signatures = self._recent_deferred_signatures[-6:]
        if force_honor_via_escalation:
            # Phase 0.2 — escalation directive prepends the regular USER
            # FEEDBACK block. The model gets the test report AND the
            # user's persistent ask; it must address both this turn.
            parts.append(
                "================ FEEDBACK ESCALATION (USER REPEATED) ================\n"
                "The user has stated the following request three or more times "
                "across recent turns; it was deferred twice behind a code "
                "blocker. The blocker is real and still appears below — fix it. "
                "AT THE SAME TIME, address the user's ask in this turn or "
                "explicitly explain in <notes> why a single turn cannot do "
                "both. Do not defer again.\n"
                f"\nRepeated ask(s):\n- "
                + "\n- ".join(fb[:240] for fb in force_honor_via_escalation)
                + "\n=================================================================="
            )
        if self._pending_feedback and not getattr(self, "_use_feedback_directives", True):
            # RAW FEEDBACK MODE — /rawfeedback on. Pass every queued
            # user note through verbatim, with ONLY the basic USER
            # FEEDBACK (HIGHEST PRIORITY) wrapper. Skip strict-scope
            # arbitration, classifier calls, MEDIA-CHANGE / ORIENTATION-
            # CHANGE / SCOPE ARBITRATION directives, asset stem mapping,
            # and scoped-constraint configuration. The model sees what
            # the user typed and decides for itself whether to <patch>
            # or <assets>. Use when the classifier is misrouting your
            # guidance (Doom trace 2026-05-23 was the motivating case).
            _internal_texts_raw = getattr(self, "_internal_feedback_texts", set())
            user_items = [
                fb for fb in self._pending_feedback if fb not in _internal_texts_raw
            ]
            internal_items = [
                fb for fb in self._pending_feedback if fb in _internal_texts_raw
            ]
            if user_items:
                joined_raw = "\n- ".join(user_items)
                parts.append(
                    "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                    "The user just typed this while watching your game. It OVERRIDES\n"
                    "any plan or default behavior. Address it explicitly in this turn:\n"
                    f"\n[USER NOTE]\n- {joined_raw}\n[/USER NOTE]\n"
                    "=================================================================="
                )
                for fb in user_items:
                    self._trace({
                        "kind": "feedback_injected",
                        "text": fb,
                        "raw_mode": True,
                        "source": "user",
                    })
            if internal_items:
                joined_internal = "\n- ".join(internal_items)
                parts.append(
                    "================ HARNESS NOTICE ================\n"
                    "Agent-generated coaching (not typed by the user):\n"
                    f"\n- {joined_internal}\n"
                    "================================================"
                )
                for fb in internal_items:
                    self._trace({
                        "kind": "feedback_injected",
                        "text": fb,
                        "raw_mode": True,
                        "source": "internal",
                    })
            feedback_items = list(self._pending_feedback)
            self._trace({
                "kind": "feedback_directives_suppressed",
                "reason": "raw_feedback_mode",
                "count": len(feedback_items),
            })
            self._last_drained_feedback = list(feedback_items)
            self._recent_feedback_texts.extend(feedback_items)
            self._recent_feedback_texts = self._recent_feedback_texts[-3:]
            self._pending_feedback.clear()
            self._clear_scoped_constraints()
            self._last_turn_contract.update({
                "had_feedback": True,
                "locks_code": False,
                "art_change": False,
                "sound_change": False,
                "behavior_scope": False,
                "behavior_bug": False,
                "orientation_change": False,
                "existing_media_request": False,
                "raw_feedback_mode": True,
            })
        elif self._pending_feedback:
            feedback_items = list(self._pending_feedback)
            strict_idxs = [
                i for (i, fb) in enumerate(feedback_items)
                if _feedback_is_strict_scope(fb)
            ]
            strict_scope_dropped = 0
            if strict_idxs:
                # Latest strict scope wins. Earlier queued asks are
                # intentionally suppressed for this turn so local models
                # don't try to satisfy contradictory objectives.
                keep_i = strict_idxs[-1]
                selected_feedback = [feedback_items[keep_i]]
                strict_scope_dropped = len(feedback_items) - 1
            else:
                selected_feedback = feedback_items
            joined = "\n- ".join(selected_feedback)
            repeated = self._detect_repeated_feedback(joined)
            repeat_prefix = ""
            if repeated:
                repeat_prefix = (
                    "[USER HAS RAISED THIS BEFORE — review screenshots "
                    "and prior fix]\n"
                )
                self._trace({
                    "kind": "repeated_user_request",
                    **repeated,
                    "text_preview": joined[:240],
                })
            post_clean_feedback = self._is_post_clean_instruction(base_message)
            if post_clean_feedback:
                parts.append(
                    "================ POST-CLEAN FEEDBACK CONTRACT ================\n"
                    f"{self._compact_post_clean_context()}\n"
                    "The user is asking for a follow-up on a working game.\n"
                    "Make ONLY the requested change. Do not refactor,\n"
                    "rewrite, rebalance, or address stale coaching unless\n"
                    "the user explicitly asks for that broader work.\n"
                    "=============================================================="
                )
                self._trace({
                    "kind": "post_clean_feedback_contract",
                    "feedback_preview": joined[:240],
                })

            # [USER NOTE] markers added 2026-05-21: explicit labels survive
            # the structured-prune compaction step. Without them, late-
            # session prompts lose user voice — the compactor summarizes
            # the banner away. The marker is also a literal token the
            # model can search-and-attend to.
            parts.append(
                "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                "The user just typed this while watching your game. It OVERRIDES\n"
                "any plan or default behavior. Address it explicitly in this turn:\n"
                f"\n[USER NOTE]\n{repeat_prefix}- {joined}\n[/USER NOTE]\n"
                "=================================================================="
            )
            for fb in selected_feedback:
                self._trace({"kind": "feedback_injected", "text": fb})
            if strict_scope_dropped:
                parts.append(
                    "================ FEEDBACK SCOPE ARBITRATION ================\n"
                    "A strict scope lock was present in the latest feedback.\n"
                    "For THIS turn, treat ONLY that latest scoped request as\n"
                    "in-scope. Ignore earlier queued requests.\n"
                    "============================================================"
                )
                self._trace({
                    "kind": "feedback_scope_arbitration",
                    "kept_latest_scoped": selected_feedback[0][:200],
                    "dropped_count": strict_scope_dropped,
                })

            # Detect intent BEFORE clearing the queue so we can shape the
            # follow-up directives. Centipede trace 20260512_180020 is
            # the motivating case.
            asset_names = (
                list(self._session_assets.keys())
                if self._session_assets else []
            )
            sound_names = (
                list(self._session_sounds.keys())
                if self._session_sounds else []
            )
            locks_code = _feedback_locks_code(joined)
            # Drop the bool(asset_names) / bool(sound_names) short-circuits
            # so first-time art/audio requests can route to the MEDIA-CHANGE
            # directive ladder. The classifiers themselves require an art
            # noun + media verb (and audio noun + media verb), so neutral
            # feedback like "fix the bug" still won't fire here. The
            # directive renderer below already branches on whether asset/
            # sound names exist when picking which recipe to emit.
            art_change = _feedback_is_art_change(joined, asset_names)
            sound_change = _feedback_is_sound_change(joined, sound_names)
            ui_feature = _feedback_is_ui_feature(joined)
            existing_media_request = _feedback_requests_existing_media(joined)
            # LLM router (chess-trace fix 2026-06-22): when the router ran and
            # said the user does NOT want new art this batch, suppress the
            # regex-driven MEDIA-CHANGE directive so the model isn't told to
            # emit <assets> on a pure code/layout turn ("no new assets, just
            # show the full screen"). Router routes; regex is the fallback.
            _route = getattr(self, "_feedback_route", None)
            if _route is not None and not _route.get("allow_assets_block"):
                if art_change or sound_change:
                    self._trace({
                        "kind": "media_change_directive_suppressed",
                        "reason": "router_no_new_assets",
                        "primary_intent": _route.get("primary_intent"),
                        "art_change": art_change,
                        "sound_change": sound_change,
                    })
                art_change = False
                sound_change = False
            # Phase 0.1 — when the user says BOTH "use the existing X" AND
            # "as a starting point" / "show them walking" / "more frames",
            # they want img2img CHAINING (new frames seeded from existing
            # sprites), not "use the existing PNG verbatim". The old
            # suppression treated all "use existing" mentions as the
            # latter and killed the MEDIA-CHANGE directive entirely; the
            # 2026-05-22 chess trace shows the user asked for img2img
            # animation chains three times and got nothing.
            img2img_chain_request = _feedback_requests_img2img_chain(joined)
            style_rebrand_request = _feedback_requests_style_rebrand(joined)
            if (
                existing_media_request
                and (art_change or sound_change)
                and not img2img_chain_request
                and not style_rebrand_request
            ):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "use_existing_media",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
                art_change = False
                sound_change = False
            elif img2img_chain_request and (art_change or sound_change):
                self._trace({
                    "kind": "img2img_chain_directive_active",
                    "existing_media_request": existing_media_request,
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
            if style_rebrand_request and (art_change or sound_change):
                self._trace({
                    "kind": "style_rebrand_directive_active",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
            # Phase 0.10b — multi-frame intent in MID-SESSION feedback
            # ("make 3 walk frames for each character", "use existing
            # images for X to seed an animation"). When the goal text
            # at session start didn't trigger the cap-raise, a later
            # animation ask would silently hit the default 24 cap. Mirror
            # the session-start path: detect multi-frame intent on the
            # feedback text and raise the session cap if it triggers.
            # General behavior — no genre logic.
            try:
                from prompts_v1 import _detect_multi_frame_intent as _mfi_mid
                mf_mid = _mfi_mid(joined)
            except Exception:
                mf_mid = []
            _prior_mid_cap = getattr(self, "_session_asset_cap", None)
            if mf_mid and (_prior_mid_cap is None or _prior_mid_cap < 72):
                self._session_asset_cap = 72
                self._trace({
                    "kind": "multi_frame_intent_detected",
                    "trigger": "mid_session_feedback",
                    "matched_keywords": mf_mid,
                    "asset_cap_raised_to": 72,
                    "prior_cap": _prior_mid_cap,
                })
            behavior_bug = _feedback_is_behavior_bug(joined)
            behavior_scope = _feedback_mentions_scoped_behavior_change(joined)
            orientation_change = _feedback_is_orientation_change(joined)
            if ui_feature and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "ui_feature",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })
                art_change = False
                sound_change = False
            # Record routing flags so `_stream` can emit the
            # `turn_contract` trace event. Mode + tag derivation happens
            # after `_configure_scoped_constraints` runs below.
            self._last_turn_contract.update({
                "had_feedback": True,
                "locks_code": locks_code,
                "art_change": art_change,
                "sound_change": sound_change,
                "behavior_scope": behavior_scope,
                "behavior_bug": behavior_bug,
                "orientation_change": orientation_change,
                "existing_media_request": existing_media_request,
            })
            self._configure_scoped_constraints(
                joined_feedback=joined,
                locks_code=locks_code,
                art_change=art_change,
                sound_change=sound_change,
            )
            self._last_drained_feedback = list(selected_feedback)
            self._recent_feedback_texts.extend(selected_feedback)
            self._recent_feedback_texts = self._recent_feedback_texts[-3:]

            self._pending_feedback.clear()

            # DK trace 20260514_104131 fix: when the user is clearly
            # reporting a behavior bug ("mario does not climb the
            # ladder"), suppress the MEDIA-CHANGE DIRECTIVE — that
            # directive tells the model "your feedback is about
            # ART/SOUND, not code" and fires the model into emitting
            # an <assets> re-render instead of fixing the code. The
            # asset-name match in `_feedback_is_art_change` is a
            # weak signal that needs an explicit suppressor when the
            # feedback contains behavior-bug language.
            if (behavior_bug or (locks_code and behavior_scope)) and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": (
                        "behavior_scope_on_strict_turn"
                        if locks_code and behavior_scope
                        else "behavior_bug"
                    ),
                    "art_change": art_change,
                    "sound_change": sound_change,
                })

            # ORIENTATION-CHANGE suppression — MK trace
            # 20260517_220025: user asked "is there a way to INVERT
            # the asset we use for the player kick?" which mentions an
            # existing asset name, so `art_change` fires; the standard
            # MEDIA-CHANGE directive then tells the model to regen the
            # sprite, which is the wrong fix (a one-line ctx.scale(-1,1)
            # canvas mirror was the right one). When orientation
            # vocabulary is present AND no regen blocker, route to the
            # one-patch canvas mirror path instead.
            if orientation_change and (art_change or sound_change):
                self._trace({
                    "kind": "media_change_directive_suppressed",
                    "reason": "orientation_change",
                    "art_change": art_change,
                    "sound_change": sound_change,
                })

            # MEDIA-CHANGE DIRECTIVE — fires only when the feedback
            # carries INDEPENDENT art/sound semantics (asset/sprite/sound
            # vocabulary). The old gate also fired on `locks_code` alone,
            # which mis-routed any "no code changes" feedback to the art
            # path even when the user really wanted a code-side tweak
            # (DK trace 2026-05-15 iter 3: "make 4x larger, no code
            # changes" routed to <assets> regeneration instead of a
            # drawImage size patch — same regenerated PNGs at the same
            # canvas size achieve nothing visible). The new SCOPED-CHANGE
            # block below handles the code-lock case directly.
            #
            # Two branches:
            #   - has_existing_media: regen using the existing name(s).
            #   - else: emit a fresh <assets>/<sounds> block (NEW ART) +
            #     ONE small <patch> wiring the loader. The 2026-05-21
            #     chess trace is the motivating case — first-time art
            #     request after a working game shipped, no sprites yet,
            #     model defaulted to inline SVG instead of the diffuser.
            has_existing_media = bool(asset_names or sound_names)
            if (
                (art_change or sound_change)
                and not behavior_bug
                and not (locks_code and behavior_scope)
                and not orientation_change
            ):
                if has_existing_media:
                    lines: list[str] = [
                        "================ MEDIA-CHANGE DIRECTIVE ================",
                        "The feedback above is about ART/SOUND, not code. The",
                        "harness can regenerate any sprite or sound in place:",
                        "emit a fresh block with the EXISTING name and a new",
                        "prompt — no JS edit needed. The existing drawSprite()",
                        "/ new Audio() call already in the file automatically",
                        "picks up the regenerated file.",
                    ]
                    # Phase 0.1 — surface the user's fuzzy asset stems
                    # ("pawn" → [white_pawn, black_pawn]) so the model
                    # emits the right `from_image` chains for animation
                    # frames seeded from existing pieces.
                    stem_map = _resolve_fuzzy_asset_stems(joined, asset_names)
                    if asset_names:
                        asset_list = ", ".join(sorted(asset_names))
                        lines.extend([
                            "",
                            "Sprites — use <assets> to re-render:",
                            "  <assets>[{\"name\":\"<existing_name>\","
                            "\"prompt\":\"<new visual prompt>\"}]</assets>",
                            f"  Existing asset names: {asset_list}",
                        ])
                        if stem_map:
                            lines.append("")
                            lines.append("Stems the user referenced map to existing assets:")
                            for stem, names in sorted(stem_map.items()):
                                joined_names = ", ".join(sorted(names))
                                lines.append(f"  '{stem}' → [{joined_names}]")
                            lines.append(
                                "When the user asks for animation FRAMES or"
                                " VARIANTS of these existing sprites (e.g."
                                " walk1/walk2/capture), declare each new"
                                " frame with `from_image: <existing_name>`"
                                " and `strength: 0.35-0.55` so SD-Turbo"
                                " img2img chains it from the parent. Do NOT"
                                " regenerate from scratch via txt2img — the"
                                " new frames will look like different"
                                " characters and break visual continuity."
                            )
                        # Phase 0.11 — STYLE REBRAND branch. User wants
                        # EVERY existing asset re-rendered with a new
                        # visual style (e.g. "all the images should look
                        # like monsters, not regular chess pieces").
                        # Different from img2img chains: this is full
                        # txt2img regeneration with NEW prompts, because
                        # from_image would carry the OLD style forward.
                        if style_rebrand_request:
                            lines.append("")
                            lines.append("STYLE REBRAND DETECTED:")
                            lines.append(
                                "The user wants ALL existing assets re-rendered"
                                " with a new visual style. For EACH existing"
                                " asset name, emit one entry in <assets> with"
                                " the SAME name + a NEW prompt that bakes in"
                                " the requested style. Do NOT use `from_image`"
                                " — chaining would carry the OLD style forward."
                                " The harness regenerates each PNG in place."
                            )
                            lines.append(
                                "Roster-size note: a full rebrand of N existing"
                                " assets = N new <assets> entries. If your"
                                " session has many entries, split the LAST"
                                " batch into a follow-up turn — never silently"
                                " regenerate only a subset."
                            )
                        lines.extend([
                            "",
                            "If the user says an animation/image is MISSING",
                            "and no existing name matches that state, you may",
                            "emit a NEW same-pattern sprite name plus ONE small",
                            "<patch> that only adds the name to the existing",
                            "asset loader/list. Do not rewrite gameplay code.",
                        ])
                    if sound_names:
                        sound_list = ", ".join(sorted(sound_names))
                        lines.extend([
                            "",
                            "Sounds — use <sounds> to re-render:",
                            "  <sounds>[{\"name\":\"<existing_name>\","
                            "\"prompt\":\"<new audio prompt>\","
                            "\"duration\":1.0}]</sounds>",
                            f"  Existing sound names: {sound_list}",
                        ])
                    lines.extend([
                        "",
                        "Do NOT swap an existing drawSprite(name,…) or",
                        "new Audio(path) call for inline procedural code here —",
                        "that loses the media path and regresses. If the user",
                        "truly asked for code changes too, address those with a",
                        "small <patch>.",
                        "========================================================",
                    ])
                else:
                    # NEW MEDIA branch — session has no generated assets/
                    # sounds yet. Tell the model to emit a fresh <assets>
                    # / <sounds> block + ONE small <patch> that loads
                    # them. Without this branch, the directive ladder
                    # was suppressed entirely and the model fell back to
                    # inline SVG / ctx.fillRect / AudioContext beeps.
                    lines = [
                        "================ MEDIA-CHANGE DIRECTIVE (NEW MEDIA) ================",
                        "The feedback above is asking for generated MEDIA, but",
                        "this session has no sprites/sounds yet. The harness",
                        "runs Z-Image-Turbo (sprites) and Stable-Audio-Open",
                        "(sounds) locally — emit the right block this turn",
                        "and the PNG/OGG paths arrive in the next user turn.",
                    ]
                    if art_change:
                        lines.extend([
                            "",
                            "Sprites — emit fresh names:",
                            "  <assets>[",
                            "    {\"name\":\"<short_id>\",",
                            "     \"prompt\":\"<one short visual sentence,"
                            " transparent bg>\"},",
                            "    ...one entry per visual entity the player"
                            " sees...",
                            "  ]</assets>",
                            "",
                            "Then ONE small <patch> that:",
                            "  1. Adds `const ASSETS = { name: new Image(),"
                            " ... }` and `await Promise.all(Object.values("
                            "ASSETS).map(i => i.decode()))` before the loop"
                            " starts.",
                            "  2. Replaces the procedural drawing site with"
                            " `ctx.drawImage(ASSETS.<name>, x, y, w, h)`.",
                            "Do NOT draw the entities with inline SVG, ctx",
                            "primitives, Unicode glyphs, or emoji — the user",
                            "explicitly asked for generated art and that's",
                            "exactly the failure mode this directive prevents.",
                        ])
                    if sound_change:
                        lines.extend([
                            "",
                            "Sounds — emit fresh names:",
                            "  <sounds>[",
                            "    {\"name\":\"<short_id>\",",
                            "     \"prompt\":\"<one short audio sentence>\",",
                            "     \"duration\":<0.3-1.5>},",
                            "    ...one entry per discrete audible event...",
                            "  ]</sounds>",
                            "",
                            "Then ONE small <patch> that does",
                            "`new Audio('./<dir>/<name>.ogg').play()` on the",
                            "matching event. Do NOT synthesize beeps with",
                            "AudioContext oscillators — the user asked for",
                            "real sound.",
                        ])
                    lines.extend([
                        "",
                        "Path conventions (the harness sets these for you):",
                        "  - Sprites land in `./<session>_assets/<name>.png`.",
                        "  - Sounds land in `./<session>_sounds/<name>.ogg`.",
                        "Reference the names you emit; the next user turn",
                        "will surface the exact relative paths.",
                        "===================================================================",
                    ])
                parts.append("\n".join(lines))
                self._trace({
                    "kind": "media_change_directive_injected",
                    "locks_code": locks_code,
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "behavior_scope": behavior_scope,
                    "asset_count": len(asset_names),
                    "sound_count": len(sound_names),
                    # Tag the branch so offline analysis can tell first-
                    # time art requests from regen requests at a glance.
                    "branch": "existing" if has_existing_media else "new",
                })

            # ORIENTATION-CHANGE DIRECTIVE — when the user wants a
            # sprite mirrored on the canvas (not regenerated) emit a
            # short canvas-mirror recipe so the model emits ONE small
            # <patch> wrapping the existing draw call rather than
            # reaching for <assets>. Genre-free, code-only.
            if orientation_change:
                orient_lines = [
                    "================ ORIENTATION-CHANGE DIRECTIVE ================",
                    "The user wants a SPRITE MIRRORED/FLIPPED on the canvas,",
                    "not regenerated. Emit ONE small <patch> that wraps the",
                    "existing drawImage() call for the named sprite with a",
                    "ctx.save() / ctx.scale(-1, 1) / ctx.restore() block.",
                    "Do NOT emit <assets> — the existing PNG is fine and",
                    "another generation will not change its facing direction",
                    "in a reliable way.",
                    "",
                    "Canonical recipe (adapt names to this file's helper):",
                    "  if (currentSpriteName === '<name>') {",
                    "    ctx.save();",
                    "    ctx.translate(x + w, 0);",
                    "    ctx.scale(-1, 1);",
                    "    ctx.drawImage(sprite, 0, y, w, h);",
                    "    ctx.restore();",
                    "  } else {",
                    "    ctx.drawImage(sprite, x, y, w, h);",
                    "  }",
                    "",
                    "If a facing-aware branch already exists in the helper,",
                    "KEEP it for all sprites and ADD a separate sprite-name",
                    "guard with an EXTRA scale(-1,1). Never replace the",
                    "existing facing branch with an `else if`.",
                    "===============================================================",
                ]
                parts.append("\n".join(orient_lines))
                self._trace({
                    "kind": "orientation_change_directive_injected",
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "locks_code": locks_code,
                })

            # SCOPED-CHANGE DIRECTIVE — fires whenever the user locked
            # the turn ("no code changes", "only X", "leave the rest").
            # Sits ABOVE rewrite-exemption gating and the downstream
            # fix-mode test-failure framing (which is also suppressed
            # for this turn — see `_scoped_change_active` flag below).
            #
            # Why it exists: 2026-05-15 DK trace iter 3. User said "make
            # 4x larger, DO NOT change other code, no code changes". The
            # agent dutifully added the user's text with HIGHEST PRIORITY
            # framing, then in the same prompt also injected the prior
            # iter's failing-probes block, the (mis-routed) MEDIA-CHANGE
            # DIRECTIVE, and standard fix-mode coaching. The 27B local
            # model tried to satisfy all four directives, shipped a 2x
            # scale + four unrelated "fixes", and the user typed back
            # "YOU DIDNT LISTEN". A frontier model can balance contra-
            # dictory prompts; a local model cannot. The fix is to stop
            # contradicting ourselves in the same prompt.
            if locks_code:
                scoped_lines = [
                    "================ SCOPED-CHANGE DIRECTIVE ================",
                    "The user scoped this turn — address ONLY what is in",
                    "the USER FEEDBACK block above.",
                    "",
                    "Routing the request:",
                    "  - If the user wants the SPRITES TO LOOK DIFFERENT",
                    "    (different style, pose, color, etc.) — emit",
                    "    <assets> ONLY with the existing names + a new",
                    "    prompt. No <patch>, no <html_file>.",
                    "  - If the user wants the SPRITES TO BE A DIFFERENT",
                    "    SIZE on screen ('4x larger', 'smaller', 'half'",
                    "    'half the size') — emit ONE small <patch> that",
                    "    ONLY changes the drawImage width and height",
                    "    arguments. Do not touch any other function,",
                    "    constant, or variable. No <assets> needed.",
                    "  - If the user wants new BEHAVIOR (different speed,",
                    "    new key binding, new rule) — emit ONE small",
                    "    <patch> that ONLY changes the relevant value or",
                    "    branch. Nothing else.",
                    "",
                    "HARD RULES for this turn:",
                    "  - Do NOT fix unrelated test failures from the",
                    "    previous iter. The user told you to ignore them",
                    "    this turn.",
                    "  - Do NOT clean up code you think looks suspicious.",
                    "  - Do NOT refactor, rename, or 'improve' anything",
                    "    not named by the user.",
                    "  - Do NOT emit a full <html_file>; use one tight",
                    "    <patch>.",
                    "  - If the user says 'Nx larger' or 'Nx smaller',",
                    "    the changed numbers in your <patch> must be",
                    "    EXACTLY that factor of the originals. Not",
                    "    'close enough', not 'approximately'.",
                    "  - If you cannot satisfy the user's request",
                    "    without a code change but they said 'no code",
                    "    changes', make the MINIMAL code edit that",
                    "    achieves the intent and nothing else.",
                    "=========================================================",
                ]
                if self._scoped_constraints is not None:
                    scoped_mode = self._scoped_constraints.get("mode")
                    if scoped_mode == "media_only":
                        scoped_lines.insert(
                            3,
                            "Turn mode: MEDIA-ONLY (existing names only).",
                        )
                    else:
                        scoped_lines.insert(
                            3,
                            "Turn mode: ONE-SMALL-PATCH (no rewrite, max one patch block).",
                        )
                parts.append("\n".join(scoped_lines))
                self._trace({
                    "kind": "scoped_change_directive_injected",
                    "art_change": art_change,
                    "sound_change": sound_change,
                    "mode": (
                        (self._scoped_constraints or {}).get("mode")
                        or "single_patch"
                    ),
                })
                # Flag consumed by the fix-mode prompt builder to drop
                # the "fix these failing probes" framing for this turn.
                # Cleared after one fix-mode build cycle (caller resets).
                self._scoped_change_active = True

            # BOUNDED-ASSET-ONLY TURN — when scoped_mode is media_only,
            # append a hard, narrow output spec at the end of the user
            # message. MK trace 20260517_220025: the model streamed an
            # asset prompt fragment in a loop for 30 minutes because
            # the user message had multiple competing instructions
            # (USER FEEDBACK + MEDIA-CHANGE + SCOPED-CHANGE) and no
            # explicit "stop after one block" constraint. Adding the
            # cap LAST so it's the most recent instruction the model
            # sees before generating.
            scoped_mode_now = (self._scoped_constraints or {}).get("mode")
            if scoped_mode_now == "media_only":
                allowed_assets = sorted(asset_names)
                bounded_lines = [
                    "================ BOUNDED OUTPUT — MEDIA ONLY ================",
                    "Emit exactly ONE <assets> JSON array. No other tags.",
                    "  - Use ONLY these existing names (regen replaces the",
                    "    file on disk; the existing drawImage call picks",
                    "    up the new pixels automatically):",
                ]
                if allowed_assets:
                    bounded_lines.append(
                        "    " + ", ".join(allowed_assets)
                    )
                else:
                    bounded_lines.append(
                        "    (no existing asset names — abort and ask "
                        "the user instead)"
                    )
                bounded_lines.extend([
                    "  - Each prompt: ONE short sentence, ≤ 200 chars.",
                    "  - Close the block with </assets> and STOP. Do NOT",
                    "    repeat or restate the JSON, do NOT add prose,",
                    "    do NOT emit <patch>, <html_file>, <plan>,",
                    "    <criteria>, <probes>, or <notes>.",
                    "=============================================================",
                ])
                parts.append("\n".join(bounded_lines))
                self._trace({
                    "kind": "bounded_asset_only_injected",
                    "asset_count": len(allowed_assets),
                })

            # Rewrite-exemption gating — unchanged from before, just
            # moved below SCOPED-CHANGE so the order in the prompt is:
            # USER FEEDBACK → (MEDIA-CHANGE if applicable) → SCOPED-CHANGE
            # → other context. SUPPRESSED when the user locked the code,
            # because granting a full <html_file> rewrite while telling
            # the model "don't change other code" is the exact contra-
            # diction we're trying to eliminate.
            if locks_code:
                self._trace({
                    "kind": "rewrite_exemption_suppressed",
                    "reason": "code_locked",
                })
            elif self._repeat_sig_streak >= 2:
                # The model has been failing on the same error twice in
                # a row. Granting another full <html_file> rewrite when
                # surgical patches haven't fixed the root cause just
                # fuels regression (cf. DK trace 20260513_122154: 3
                # consecutive 22-25 KB rewrites all hit the same
                # ERR_FILE_NOT_FOUND). Force the model to send a
                # focused <patch> this turn so it has to articulate
                # what's actually different.
                self._trace({
                    "kind": "rewrite_exemption_suppressed",
                    "reason": "repeat_signature",
                    "streak": self._repeat_sig_streak,
                })
            else:
                self._allow_one_rewrite = True
                self._trace({"kind": "rewrite_exemption_armed"})
        # Drain any <lookup_bullet> resolutions queued by the previous
        # assistant reply. These come BEFORE the base message so the
        # model sees them as fresh material before the iteration prompt.
        if self._pending_bullet_lookups:
            for block in self._pending_bullet_lookups:
                parts.append(block)
            self._pending_bullet_lookups.clear()

        # Probe quarantine notices are not normal "coaching"; they explain
        # that the agent dropped a broken model-authored probe after repeated
        # eval-time errors. Always surface once so the next model turn knows
        # it may re-emit <probes> if the dropped assertion mattered.
        if self._pending_probe_quarantine_notices:
            joined = "\n- ".join(self._pending_probe_quarantine_notices)
            parts.append(
                "================ PROBE QUARANTINE NOTICE ================\n"
                f"- {joined}\n"
                "========================================================="
            )
            for c in self._pending_probe_quarantine_notices:
                self._trace({"kind": "probe_quarantine_notice_injected", "text": c})
            self._pending_probe_quarantine_notices.clear()

        # A2/A5: agent-queued coaching lines (deliberation guard recovery,
        # repeat-error nudges). Rendered as a single high-priority block
        # so the model sees them before the base instruction.
        if self._pending_coaching:
            scoped_lock_active = bool(self._scoped_constraints is not None)
            if scoped_lock_active:
                self._trace({
                    "kind": "coaching_suppressed_scoped_lock",
                    "count": len(self._pending_coaching),
                })
                self._pending_coaching.clear()
            else:
                prev = self._previous_report or {}
                probes = prev.get("probes") or []
                full_probe_pass = bool(probes) and all(bool(p.get("ok")) for p in probes)
                clean_report = (
                    self._previous_report_ok is True
                    and full_probe_pass
                    and not (prev.get("errors") or [])
                    and not (prev.get("soft_warnings") or [])
                    and not (prev.get("page_errors") or [])
                    and not (prev.get("console_errors") or [])
                )
                # Phase 0D-1 (Fieldrunners trace 20260626_102307): stall-recovery
                # coaching MUST survive a clean prior report. After a deliberation
                # / repetition / silent stream abort — or while an art reprompt is
                # still armed — the queued coaching IS the recovery instruction.
                # Discarding it just because the PREVIOUS iter's probes were green
                # is exactly the iter-4 failure: the model rambled 707s with no
                # tag, the deliberation guard queued recovery coaching, and
                # `coaching_suppressed_clean_pass` (trace line 521) threw it away,
                # so the retry got only a generic "no code" fallback. Treat the
                # prior report as NOT clean for coaching purposes in that case.
                stall_recovery_pending = bool(
                    self._last_stream_deliberated
                    or self._last_stream_looped
                    or self._last_stream_silent
                    or getattr(self, "_unhonored_asset_request", None)
                )
                if clean_report and stall_recovery_pending:
                    clean_report = False
                    self._trace({
                        "kind": "coaching_clean_pass_override_stall_recovery",
                        "deliberated": bool(self._last_stream_deliberated),
                        "looped": bool(self._last_stream_looped),
                        "silent": bool(self._last_stream_silent),
                        "unhonored_asset": bool(
                            getattr(self, "_unhonored_asset_request", None)
                        ),
                    })
                coaching_to_inject = self._pending_coaching
                if clean_report:
                    # Visual/static/player-motion warnings are generated
                    # specifically because the normal report looked clean.
                    # Preserve those so "passes but not playable" issues
                    # still reach the next coder turn.
                    must_keep_keywords = (
                        "VISUAL CRITIC",
                        "VISUAL JUDGE",
                        "REGRESSION SUSPECTED",
                        "STATIC SCREEN",
                        "STATE LOCOMOTION",
                        "Asset references",
                        "Sound references",
                    )
                    coaching_to_inject = [
                        c for c in self._pending_coaching
                        if any(kw in c for kw in must_keep_keywords)
                    ]
                    suppressed = [
                        c for c in self._pending_coaching
                        if c not in coaching_to_inject
                    ]
                    self._trace({
                        "kind": "coaching_suppressed_clean_pass",
                        "count": len(suppressed),
                        "preserved_count": len(coaching_to_inject),
                    })
                    # Phase 4 (4D.2): a suppressed-but-needed coaching pass is
                    # the canonical HARNESS_BUG signal (Fieldrunners iter 4) —
                    # record it for failure_class. Only flag when something was
                    # actually dropped (suppressing zero is a no-op).
                    if suppressed:
                        self._last_coaching_action = "suppressed"
                if coaching_to_inject:
                    joined = "\n- ".join(coaching_to_inject)
                    # [CRITIC] marker (added 2026-05-21): mirrors the
                    # [USER NOTE] pattern so critic findings get the same
                    # compaction-survival treatment as user feedback.
                    # Distinct label so the model can tell which voice is
                    # speaking (user override > critic suggestion).
                    has_critic = any(
                        "VISUAL CRITIC" in c or "VISUAL JUDGE" in c
                        for c in coaching_to_inject
                    )
                    label_open = "[CRITIC]\n" if has_critic else ""
                    label_close = "\n[/CRITIC]" if has_critic else ""
                    parts.append(
                        "================ AGENT COACHING ================\n"
                        f"{label_open}- {joined}{label_close}\n"
                        "================================================"
                    )
                    for c in coaching_to_inject:
                        self._trace({"kind": "coaching_injected", "text": c})
                    # Phase 4 (4D.1): record that coaching reached the model
                    # this turn (don't downgrade a prior "suppressed" flag).
                    if getattr(self, "_last_coaching_action", "none") != "suppressed":
                        self._last_coaching_action = "injected"
                self._pending_coaching.clear()
        post_clean_with_feedback = bool(
            base_message
            and self._last_turn_contract
            and self._last_turn_contract.get("had_feedback")
            and self._is_post_clean_instruction(base_message)
        )
        if base_message:
            if post_clean_with_feedback:
                # Replace the large clean report with one compact line; the
                # POST-CLEAN FEEDBACK CONTRACT above carries the important
                # baseline signal without drowning the user's fresh request.
                parts.append(self._compact_post_clean_context())
                self._trace({"kind": "post_clean_report_compacted"})
            else:
                parts.append(base_message)

        # Inline CURRENT FILE ON DISK on post-clean feedback turns and
        # whenever an answer is being injected. Without this, the model
        # is asked to patch concrete code ("remove the circles", "shift
        # the muzzle flash up") with no file in context — after compaction
        # the original <html_file> is gone from history, and post-clean
        # / answer turns don't otherwise carry it (unlike fix_instruction).
        # The DOOM trace 20260523_152317 has the model literally saying
        # "I genuinely do not have the file contents in my context this
        # turn" and giving up. Mirrors the file block from
        # continuation_instruction.
        if (post_clean_with_feedback or answer_was_consumed) and self._current_file:
            parts.append(
                "CURRENT FILE ON DISK (this is the SOURCE OF TRUTH — patch "
                "against THIS exact text, character-for-character; earlier "
                "turns' code may be stale or absent from this prompt):\n"
                "```html\n"
                f"{self._current_file}\n"
                "```"
            )
            self._trace({
                "kind": "current_file_inlined",
                "reason": (
                    "post_clean_with_feedback"
                    if post_clean_with_feedback
                    else "answer_consumed"
                ),
                "bytes": len(self._current_file),
            })

        # Push a confirmation line into the TUI agent log via the token
        # callback. Plain text only — the TUI renders streamed tokens
        # via Rich's Text() (no markup parsing) so any [tag] would
        # appear literally. Newlines bracket the line so the streaming
        # buffer flushes it as a discrete log line. Bypassed on CLI
        # runs (no callback wired) — those users see feedback_injected
        # events in the trace instead.
        if consumed_items and self._token_cb is not None:
            try:
                preview = "; ".join(consumed_items)
                self._token_cb(f"\n>> APPLIED to this turn: {preview}\n")
            except Exception:
                pass

        # Phase 0.1 — restore code-touching feedback that was partitioned
        # out for parallel media processing. These items remain pending
        # behind the code blocker; they re-evaluate next turn.
        if code_to_defer:
            for fb in code_to_defer:
                if fb not in self._pending_feedback:
                    self._pending_feedback.append(fb)

        return "\n\n".join(parts)

    # -- streaming ----------------------------------------------------------



