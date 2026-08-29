#!/usr/bin/env python3
"""One-shot / maintainer helper: add or refresh `prompt_640` on every
library entry. Run from repo root:

  .venv/bin/python scripts/gen_prompt_640_library.py

Keeps media-mode `prompt` untouched. `/640` + `/games N` uses prompt_640.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "memory" / "prompt_library.jsonl"

FOOTER = (
    " TARGET=/640 JMR native: ONE 640×480 HTML file. Do NOT emit assets, "
    "sounds, or videos tags; no CDN, no fetch, no WebGL. Art MUST be classic "
    "arcade animated PIXEL MAPS (compact 2D 0/1 or palette-index arrays, blit "
    "with fillRect per texel) or at most 16 inline data:image sheets — include "
    "at least 2 frames for idle/walk/attack wherever motion matters so "
    "animation reads clearly. imageSmoothingEnabled=false. HUD via canvas "
    "fillText. Never ship solid colored squares/circles as the final look "
    "for playfield units."
)

# Hand-tuned overrides (mechanics + animation-first). Others are derived.
OVERRIDES: dict[str, str] = {
    "street-fighter": (
        "Build a Street Fighter game. Blue-gi player vs red-gi CPU on a dojo "
        "stage, always facing each other. Each fighter needs animated pixel-map "
        "poses (≥2 frames each): idle, walk, jump, crouch/duck, block, punch "
        "(arm extended), kick (leg high), fireball throw, hit/stagger, KO fall. "
        "Controls: ArrowLeft/Right move, ArrowUp jump, ArrowDown duck, J punch, "
        "K kick, L fireball, R restart. Health bars; KO ends the round. CPU "
        "approaches, attacks up close, fireballs at range, jumps/ducks. Hitboxes "
        "travel toward the opponent; L/R sheets or unmirrored draw (no negative "
        "scale). Expose facing on window.state."
    ),
    "donkey-kong": (
        "Build a Donkey Kong game. Single-screen arcade: plumber climbs sloped "
        "girders and ladders to the top while an ape rolls barrels down. Hero "
        "pixel-map poses (≥2 frames): idle, run, climb, jump, hit/fall. Barrels "
        "MUST stay on girder yAt(x), roll to the low end, tumble to the next "
        "floor, reverse — never free-fall off the first floor. Jump over barrels "
        "to score; touch loses a life. Controls: ArrowLeft/Right, ArrowUp/Down "
        "climb, Space jump, R restart. Reach top to win. Classic DK colors and "
        "silhouettes via pixel maps."
    ),
    "centipede": (
        "Build a Centipede game. Fixed shooter: blaster in the lower zone fires "
        "upward (muzzle points up). Centipede = head + body + tail as one chain "
        "(history trail). Shot middle splits; head scores most; hit leaves a "
        "mushroom. Blaster idle+fire frames; head/body pixel-map sprites. "
        "Controls: Arrows move, Space fire, R restart. Clear wave to advance; "
        "segment or spider touch loses a life. HUD score/lives."
    ),
    "pac-man": (
        "Build a Pac-Man game. Yellow chomping hero eats dots in a walled maze; "
        "four colored ghosts with distinct chase flavors. Hero: animated chomp "
        "frames facing move direction; ghosts: body + eyes toward movement; "
        "frightened blue frames. Four power-pellets frighten ghosts. Controls: "
        "Arrows (queued turns), R restart. Eat all dots to clear; ghost catch "
        "when not powered loses a life. COPY a compact connected maze template "
        "(tile coords). Expose dotsRemaining on window.state. Classic arcade "
        "silhouettes via pixel maps — not yellow circles."
    ),
    "space-invaders": (
        "Build a Space Invaders game. Player cannon left/right fires up at a "
        "grid of aliens that march, drop, and speed up as numbers thin. Aliens: "
        "2-frame leg animation; classic crab/squid/octopus row silhouettes via "
        "pixel maps. Destructible bunkers; bonus UFO. Aliens drop bombs. "
        "Controls: ArrowLeft/Right, Space fire, R restart. Clear wave to "
        "advance; bomb hit or aliens at bottom loses a life."
    ),
    "asteroids": (
        "Build an Asteroids game. Vector shooter on black: triangular ship "
        "rotates, thrusts with momentum, fires forward. Asteroids are IRREGULAR "
        "jagged polygons that split large→medium→small. Screen wraps. Occasional "
        "saucer. Controls: ArrowLeft/Right rotate, ArrowUp thrust, Space fire, "
        "R restart. Vector strokes only (authentic look) — no sprite sheets "
        "required. HUD score/lives. Expose window.state."
    ),
    "frogger": (
        "Build a Frogger game. Cross road then river to home slots. Frog: idle + "
        "mid-hop pixel-map frames per facing. Cars/trucks/logs/turtles as "
        "distinct pixel sprites. Controls: Arrows hop one cell per keydown edge, "
        "R restart. Fill homes to win; traffic/water/timeout loses a life. "
        "HUD score/lives/timer."
    ),
    "galaga": (
        "Build a Galaga game. Formation shooter: ship fires up; insects curve in, "
        "form up, dive-attack. Enemies: 2-frame wing flap pixel maps; player "
        "ship pixel art. Controls: ArrowLeft/Right, Space fire, R restart. "
        "Clear wave to advance; hit loses a life. Starfield + HUD."
    ),
    "tetris": (
        "Build a Tetris game. Seven tetrominoes fall in a well; move, rotate, "
        "soft/hard drop; clear lines; speed up by level. Next-piece + ghost "
        "preview. Controls: Arrows + Space hard drop, R restart. Colorful "
        "blocks with crisp pixel borders (small tile pixel maps or patterned "
        "fillRect tiles — not blank grey squares). HUD score/level/lines."
    ),
    "breakout": (
        "Build a Breakout game. Paddle bounces ball into a brick wall; angle "
        "from hit point. Some bricks 2-hit; power-ups (wide/multi-ball). "
        "Controls: Arrows or mouse, Space launch, R restart. Bricks and ball "
        "as crisp pixel tiles/sprites — not flat unshaded rectangles. HUD "
        "score/lives."
    ),
    "snake": (
        "Build a Snake game. Grid snake grows on food; speed up; no reverse "
        "into self. Wrap toggle. Controls: Arrows, R restart. Rounded pixel "
        "segments + bright food tile maps. HUD score/high score."
    ),
    "dig-dug": (
        "Build a Dig Dug game. Dig tunnels four ways; pump monsters with "
        "harpoon or drop rocks. Digger: idle/dig/pump frames; monsters: walk + "
        "inflate frames — all pixel maps. Controls: Arrows, Space pump, R "
        "restart. Clear monsters to advance. Layered soil colors."
    ),
    "qbert": (
        "Build a Q*bert game. Isometric pyramid hopper recolors cubes. Hero: "
        "idle + mid-hop frames per diagonal; enemies bounce down. Controls: "
        "four arrows = four diagonals, R restart. Crisp isometric cube pixel "
        "art + character sprites."
    ),
    "missile-command": (
        "Build a Missile Command game. Defend six cities with crosshair "
        "counter-missiles and expanding blasts. Controls: mouse or arrows + "
        "Space, R restart. Glowing pixel explosions and trails on a night "
        "sky — not plain lines only. HUD score/cities."
    ),
    "joust": (
        "Build a Joust game. Flap against gravity; collide from ABOVE to win. "
        "Mount: flap wings up/down + walk frames (pixel maps). Eggs to collect. "
        "Controls: ArrowLeft/Right, Space flap, R restart. Platforms over lava."
    ),
    "bomberman": (
        "Build a Bomberman game. Grid bombs explode in a plus; soft blocks; "
        "power-ups; wandering enemies. Hero: idle + walk per direction pixel "
        "maps; bomb/flame frames. Controls: Arrows, Space bomb, R restart."
    ),
    "pong": (
        "Build a Pong game. Player vs CPU paddles; ball speeds up; bounce "
        "angle from paddle hit. First to 11. Controls: ArrowUp/Down or W/S, "
        "R restart. Clean white paddles/ball — authentic minimal look is OK; "
        "optional tiny pixel bevel. Dashed net + scores."
    ),
    "super-mario": (
        "Build a Super Mario-style side-scroller. Run/jump through bricks, "
        "pipes, pits, stomping enemies. Hero pixel poses (≥2 frames): idle, "
        "run, jump, hit. Camera scrolls. Coins + flag goal. Controls: "
        "ArrowLeft/Right, Space/ArrowUp jump, R restart. NO video cutscenes — "
        "title card via fillText/pixel banner only. Classic colorful pixel art."
    ),
    "kung-fu-master": (
        "Build a Kung-Fu Master beat-'em-up. Side-scroll; enemies from both "
        "sides; boss at floor end. Hero poses (≥2 frames): idle, walk, crouch, "
        "punch, kick, knockdown — pixel maps. Controls: Arrows, J/K attack, R "
        "restart. Health bar + timer. NO video — sprite combat only."
    ),
    "1942": (
        "Build a 1942 vertical shmup. Scroll up over sea; formations; barrel "
        "roll dodge. Player bank frames; enemy propeller shimmer — pixel maps. "
        "Controls: Arrows, Space fire, Shift roll, R restart."
    ),
    "doom": (
        "Build a Doom-inspired game FOR /640: top-down OR simple 2.5D canvas "
        "maze shooter WITHOUT three.js/WebGL/CDN. Textured walls via pixel-map "
        "tiles; monsters as billboard-style pixel sprites with idle+attack "
        "frames; weapon overlay with muzzle flash (≥2 frames). WASD move, "
        "mouse or arrows turn/aim, Space/click fire, R restart. Health/ammo "
        "HUD. NO video cutscenes."
    ),
    "minecraft": (
        "Build a Minecraft-inspired game FOR /640: top-down or side-view "
        "block sandbox on a 2D canvas (no three.js). Place/break block tiles "
        "drawn as crisp pixel cubes; hotbar; simple crafting optional. "
        "Controls: Arrows/WASD move, keys place/break, R restart. Chunked "
        "tile map. Expose window.state world."
    ),
    "outrun": (
        "Build an OutRun-style Mode-7 road racer on 2D canvas. Steer, "
        "accelerate, brake; rivals + roadside sprites scale as they approach. "
        "Car + scenery as pixel maps/sheets. Controls: Arrows, R restart. "
        "Timer/checkpoints. NO video — start banner in fillText."
    ),
    "chess": (
        "Build a holochess-style fantasy chess. PLAYER vs CPU. Every move "
        "animates: tween drawX/drawY with walk/hop frames (≥3) — never snap. "
        "Capture: lift then slam poses before remove. Pieces = distinct "
        "monster pixel-map sprites (idle+walk+lift+slam) for both sides. "
        "Negamax depth 3; setTimeout CPU. Block input while animating. "
        "Controls: click, R restart. Expose state.animating."
    ),
    "zelda": (
        "Build a Zelda-like top-down action RPG. Tile overworld; sword; NPCs; "
        "hearts. Hero poses (≥2 frames): idle/walk ×4 facings, sword, hit — "
        "pixel maps. Controls: Arrows, Z/Space sword, Enter talk, R restart. "
        "NO video — intro text crawl optional. Camera room-to-room."
    ),
    "monkey-island": (
        "Build a Monkey Island-like point-and-click. Walk to hotspots; verbs "
        "look/take/use; inventory puzzles; 2–3 scenes. Character: idle + walk "
        "cycle pixel maps. Backgrounds as large pixel scenes or tiled art "
        "(inline sheets OK ≤16). Controls: mouse, R restart. NO video."
    ),
    "dragons-lair": (
        "Build a Dragon's Lair-style QTE FOR /640 (no video pipeline): 6–8 "
        "peril scenes as full-screen pixel backgrounds + hazard sprites that "
        "MOVE toward the threatened body part + hero pose sprites. Timed "
        "Arrow/Space windows; word prompt names the key. SCENES array; "
        "performance.now(); on fail flash and retry or lose a life. Success "
        "advances. NO <videos>/<assets> tags — all art inline pixel maps/"
        "data:image. Expose window.state.scene."
    ),
    "battlezone": (
        "Build a Battlezone game. 2D wireframe vector tank (NOT three.js, "
        "NOT sprites): mountains, grid, enemy tanks, radar. Controls: Arrows "
        "drive/turn, Space fire, R restart. Bright vector strokes on black."
    ),
    "star-wars": (
        "Build a Star Wars 1983 trench run. 2D wireframe vector (NOT three.js/"
        "sprites): trench walls, towers, TIEs, crosshair bolts. Controls: "
        "Arrows steer, Space fire, R restart. Bright vectors on black."
    ),
    "tower-defense": (
        "Build a Tower Defense game. Waypoint path; place turrets; auto-fire; "
        "waves. Each turret type, creep type, and projectile needs distinct "
        "animated pixel-map art (≥2 frames where they move/fire) — NOT bare "
        "circles. Muzzle flash on fire. HUD money/lives/wave via fillText. "
        "Controls: click place, R restart. Expose enemies[], towers[], path."
    ),
    "tower-defense-openfield": (
        "Build a Fieldrunners-style open-field TD. No pre-drawn path: grass "
        "grid, entrance→exit; towers BUILD the maze; enemies BFS (reject "
        "wall-offs; re-path on place). Towers: Rapid Gun, Missile, Goo, "
        "Mortar, Tesla, Flame. Creeps escalate by wave. Sidebar select/place/"
        "upgrade/sell. Space starts wave. EVERY tower and creep is a classic "
        "pixel-map sprite with idle + fire/walk frames — recognizable "
        "Fieldrunners silhouettes, NOT colored squares. Rotating heads: "
        "separate L/R or angle pixel sheets (no negative scale). Beam towers: "
        "upright sprite + procedural bolt/flame to target. Expose "
        "window.state grid/path/enemies/towers."
    ),
    "pinball": (
        "Build a Pinball table. Gravity ball, flippers, bumpers, drain, "
        "plunger. Table/ball/flipper/bumper as crisp pixel art or tight "
        "vector+pixel hybrid. Controls: Z and / or Shift flippers, R restart. "
        "HUD score/balls."
    ),
    "mortal-kombat": (
        "Build a Mortal Kombat-style versus fighter. Two fighters, health "
        "bars, always face each other. Poses (≥2 frames): idle, walk, punch, "
        "kick, block, special projectile, hit — pixel maps / L+R sheets. "
        "P1: Arrows + A/S/D. CPU opponent. Round win on zero health. Fatality "
        "= dramatic pixel animation or pose sequence on canvas (NO video "
        "file). Rematch via reset()."
    ),
    "metal-slug": (
        "Build a Metal Slug-style run-and-gun. Side-scroll; shoot ahead/up/"
        "diagonal; stomp/jump. Commando poses (≥2 frames): idle, run, jump, "
        "shoot. Enemies explode frames. Controls: Arrows, Space jump, Z/X "
        "fire, R restart. Chunking pixel art."
    ),
    "prince-of-persia": (
        "Build a Prince of Persia-style cinematic platformer. Run, measured "
        "jump, ledge hang/pull-up, spikes/blades, sword guard. Rotoscope-like "
        "pixel poses (≥2 frames): idle, run, jump, hang, sword. Controls: "
        "Arrows, Space jump, Shift sword, R restart. Timer HUD. NO video."
    ),
    "rampage": (
        "Build a Rampage-style climb-and-smash. Monster climbs a skyscraper, "
        "punches floors, dodges choppers. Poses (≥2 frames): cling, climb, "
        "smash, eat, hit — pixel maps with EMPTY backgrounds (building drawn "
        "in code). Controls: Arrows climb/move, Space smash, R restart."
    ),
    "mr-do": (
        "Build a Mr. Do! game. Dig green dirt into black tunnels; cherries; "
        "push/drop apples; power ball; EXTRA letters. Mr. Do: idle/walk/dig/"
        "throw frames; creeps walk+digger — classic arcade pixel maps. "
        "Controls: Arrows, Space ball, R restart."
    ),
    "fighter-showcase": (
        "Build a Fighter Showcase. ONE centered fighter, no opponent. Pixel "
        "poses (≥2 frames): idle, walk, jump, crouch, punch, kick, block. "
        "Controls: Arrows, J/K/L, R restart. HUD shows move name. "
        "window.state.move."
    ),
    "checkers": (
        "Build Checkers (candy carnival) player vs CPU. Gumdrop/jelly pixel "
        "sprites with hop frames; kings crowned. Bounce-hop moves; capture "
        "squash. Negamax depth 3. Controls: click, R restart. Pastel board."
    ),
    "roguelike-dungeon": (
        "Build a Roguelike dungeon. Proc-gen rooms; fog-of-war; turn-based "
        "bump combat; stairs. Hero/monster/tile/loot as colorful pixel-map "
        "sprites — NOT ASCII. Controls: Arrows, R restart. HUD HP/depth/gold."
    ),
    "bullet-hell-boss": (
        "Build a Bullet Hell boss fight. Dense patterns; tiny hitbox; boss "
        "phases. Player + boss + bullet pixel sprites (animated where useful). "
        "Controls: Arrows, Shift slow, Space fire, R restart."
    ),
}


def _strip_media_sentences(text: str) -> str:
    # Drop video / diffuser pipeline sentences; keep mechanics.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    keep: list[str] = []
    skip_re = re.compile(
        r"generate\s+video|image-to-video|<videos>|<assets>|<sounds>|"
        r"muted\s+skippable|full-screen\s+overlay|z-image|stable\s+audio|"
        r"from\s+cdn|three\.js\s+from\s+cdn",
        re.I,
    )
    for p in parts:
        if skip_re.search(p):
            continue
        keep.append(p)
    return " ".join(keep).strip()


def _inject_art(text: str) -> str:
    t = text
    t = re.sub(
        r"Generated PNGs must be drawn with drawImage \(not fillRect placeholders\)\.?",
        "Draw units with animated pixel maps (not fillRect placeholders).",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Generated PNGs\s*→\s*drawImage\.?",
        "Use animated pixel-map sprites.",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"high-resolution(?: detailed)? sprites",
        "classic arcade pixel-map sprites",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Colorful high-resolution",
        "Colorful classic pixel-map",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"generate high-resolution transparent sprite art",
        "author compact animated pixel-map sprite art",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Generate <assets>[^.]*\.",
        "Author inline pixel-map sprites for each unit. ",
        t,
        flags=re.I,
    )
    if "pixel map" not in t.lower() and "pixel-map" not in t.lower():
        t = t.rstrip(".") + (
            ". Animate key actions with ≥2 pixel-map frames so motion reads."
        )
    return t


def derive_prompt_640(rec: dict) -> str:
    name = rec["name"]
    if name in OVERRIDES:
        body = OVERRIDES[name]
    else:
        body = _inject_art(_strip_media_sentences(rec["prompt"]))
    body = body.strip()
    if not body.endswith("."):
        body += "."
    return body + FOOTER


def main() -> None:
    rows = []
    for line in LIB.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    out_lines = []
    for rec in rows:
        rec["prompt_640"] = derive_prompt_640(rec)
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    LIB.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    n640 = sum(1 for r in rows if r.get("prompt_640"))
    print(f"wrote {n640}/{len(rows)} prompt_640 entries → {LIB}")


if __name__ == "__main__":
    main()
