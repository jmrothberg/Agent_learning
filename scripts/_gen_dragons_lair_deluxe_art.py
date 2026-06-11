#!/usr/bin/env python3
"""One-shot art batch for games/dragons-lair-deluxe.html.

Hand-directed spec list driven through the same Z-Image-Turbo pipeline
the agent uses (assets.generate_assets). Output lands in
games/dragons-lair-deluxe_assets/ ; shared cache games/_asset_cache.

Usage:
    .venv/bin/python scripts/_gen_dragons_lair_deluxe_art.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import assets  # noqa: E402

# Shared style tag — every prompt carries it so the whole game reads as one
# hand-painted 80s laserdisc cartoon, not a grab bag of styles.
STYLE = (
    "classic 1980s fantasy cel animation film still, hand-painted, "
    "rich saturated colors, dramatic lighting, detailed, cinematic"
)

# One canonical character description reused verbatim in every knight pose —
# txt2img with a shared description is the proven consistency approach
# (see assets.py 2026-05-30 A/B note).
KNIGHT = (
    "heroic knight with blond hair, gleaming silver plate armor with gold trim, "
    "green tunic, brown boots, holding a bright sword"
)
DRAGON = "colossal emerald-green dragon with glowing amber eyes, horned head, bat wings"
PRINCESS = "beautiful princess with long golden hair in a shimmering pink-gold gown"

BG = (768, 768)
SPRITE = (512, 512)


def bg(name: str, scene: str) -> dict:
    return {
        "name": name,
        "prompt": f"{scene}, wide establishing shot, empty scene with no people, {STYLE}",
        "size": BG,
    }


def sprite(name: str, desc: str) -> dict:
    return {
        "name": name,
        "prompt": f"{desc}, full body, centered, {STYLE}, transparent background",
        "size": SPRITE,
    }


def keyart(name: str, scene: str) -> dict:
    return {"name": name, "prompt": f"{scene}, {STYLE}", "size": BG}


SPECS: list[dict] = [
    # ---- 8 scene backgrounds -------------------------------------------
    bg("bg_drawbridge", "castle drawbridge over a misty green moat at dusk, heavy chains, rotting wooden planks, torchlit gatehouse"),
    bg("bg_tentacle_hall", "torch-lit medieval stone hall with a bubbling pit of black slime in the floor, green ooze dripping from cracked pillars"),
    bg("bg_blade_corridor", "narrow gothic stone corridor with giant swinging pendulum axe blades hanging from the ceiling, blue moonlight shafts"),
    bg("bg_crumbling_stairs", "collapsing spiral stone staircase over a bottomless black abyss inside a ruined tower, falling rubble, eerie purple glow"),
    bg("bg_bat_cavern", "dark underground cavern with stalactites, swarms of bats, hundreds of glowing yellow eyes in the darkness"),
    bg("bg_lava_ledge", "underground river of glowing orange lava with a narrow rock ledge path, rising embers and smoke"),
    bg("bg_throne_hall", "ruined gothic throne room antechamber, torn purple banners, lightning flashing through tall shattered windows"),
    bg("bg_dragons_lair", "colossal cavern filled with mountains of gold treasure coins and jewels, giant carved dragon skull arch, golden glow"),
    # ---- knight poses (same character description every frame) ---------
    sprite("knight_idle", f"{KNIGHT}, standing alert with sword ready"),
    sprite("knight_run", f"{KNIGHT}, running fast mid-stride, cape flying"),
    sprite("knight_jump", f"{KNIGHT}, leaping high through the air, knees tucked"),
    sprite("knight_duck", f"{KNIGHT}, crouching low, ducking under danger, shield raised overhead"),
    sprite("knight_dodge_left", f"{KNIGHT}, lunging and diving toward the left side"),
    sprite("knight_dodge_right", f"{KNIGHT}, lunging and diving toward the right side"),
    sprite("knight_slash", f"{KNIGHT}, swinging sword in a powerful overhead slash, motion arc"),
    sprite("knight_death", f"{KNIGHT}, recoiling backward in pain, dropping sword, dramatic pose"),
    sprite("knight_victory", f"{KNIGHT}, triumphant pose holding sword raised high overhead"),
    # ---- creatures and perils ------------------------------------------
    sprite("dragon_head", f"head and neck of a {DRAGON}, jaws open roaring"),
    sprite("dragon_fire", f"head of a {DRAGON} breathing a huge cone of orange fire"),
    sprite("princess", f"{PRINCESS}, standing gracefully, hands clasped, hopeful expression"),
    sprite("tentacle", "single slimy green monster tentacle with suckers, curling upward"),
    sprite("blade_pendulum", "giant medieval pendulum axe blade on a chain, gleaming steel edge"),
    sprite("fireball", "ball of roaring orange fire with trailing flames"),
    sprite("bat", "menacing cartoon bat with wings spread wide, fangs bared"),
    sprite("rock_debris", "chunk of falling grey castle stone rubble with dust"),
    sprite("skeleton_flash", "glowing electric blue skeleton silhouette of a knight, x-ray flash effect, arms flailing"),
    sprite("title_emblem", "ornate golden dragon emblem coiled around a sword, medieval heraldic crest"),
    # ---- key art (also seeds the I2V cutscenes) ------------------------
    keyart("key_castle", "dark fantasy castle on a cliff at blood-red sunset, bats circling the towers, knight on horseback approaching on a winding road, epic wide shot"),
    keyart("key_dragon_wake", f"{DRAGON} waking up and rearing its head atop a mountain of gold treasure in a vast cavern, embers in the air"),
    keyart("key_rescue", f"{KNIGHT} holding hands with {PRINCESS} atop a pile of gold treasure, both smiling, warm golden light, happy ending"),
    keyart("key_death", "glowing electric blue skeleton of a knight flashing against pitch black darkness, x-ray death flash, arcade game over moment"),
]


def main() -> int:
    out_dir = REPO / "games" / "dragons-lair-deluxe_assets"
    cache_dir = REPO / "games" / "_asset_cache"
    print(f"Generating {len(SPECS)} assets -> {out_dir}")
    gen = assets.try_load_image_generator()
    if gen is None:
        print("FATAL: could not load Z-Image-Turbo")
        return 1
    t0 = time.time()
    paths = assets.generate_assets(
        SPECS, session_dir=out_dir, cache_dir=cache_dir, image_generator=gen
    )
    dt = time.time() - t0
    missing = [s["name"] for s in SPECS if assets._safe_filename(s["name"]) not in paths and s["name"] not in paths]
    print(f"\nDone in {dt:.0f}s — {len(paths)}/{len(SPECS)} generated")
    for n in sorted(paths):
        print(f"  {n}: {paths[n].name}")
    if missing:
        print(f"MISSING: {missing}")
        err = getattr(gen, "_last_error", None)
        if err:
            print(f"last error: {err}")
        return 2
    print("ART_BATCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
