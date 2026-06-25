#!/usr/bin/env python3
"""Tower defense turret bases, 5-way heads, vertical beam towers."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import assets  # noqa: E402

STYLE = "Fieldrunners iPhone cartoony tower defense sprite, clean edges, vibrant colors"
OUT = REPO / "goodgame" / "build-an-open-field-tower-defe_20260625_144848_assets"
CACHE = REPO / "games" / "_asset_cache"
SIZE = (512, 512)
ASSET_DIR = OUT


def spec(name: str, prompt: str, **kw) -> dict:
    p = prompt.strip()
    if "transparent" not in p.lower():
        p += ", transparent background"
    return {"name": name, "prompt": f"{p}, {STYLE}", "size": SIZE, **kw}


def head_specs(prefix: str, weapon: str, parent: str | None = None) -> list[dict]:
    """Five aim buckets: E, SE, S, NE, N — W/SW/NW mirrored in code."""
    dirs = [
        ("0", "barrel pointing horizontally to the right east"),
        ("1", "barrel pointing diagonally down to the right south-east 45 degrees"),
        ("2", "barrel pointing straight down south"),
        ("3", "barrel pointing diagonally up to the right north-east 45 degrees"),
        ("4", "barrel pointing straight up north"),
    ]
    out: list[dict] = []
    for suffix, aim in dirs:
        kw: dict = {}
        if parent:
            kw["from_image"] = parent
        out.append(
            spec(
                f"{prefix}_{suffix}",
                f"{weapon} turret head only rotating on pivot, {aim}, no base platform",
                **kw,
            )
        )
    return out


def base_spec(name: str, desc: str, parent: str | None = None) -> dict:
    kw = {"from_image": parent} if parent else {}
    return spec(name, f"circular tower defense platform base only, {desc}, no weapon barrel", **kw)


def main() -> int:
    specs: list[dict] = []

    # Rapid gun — derived from original gun art
    specs.append(base_spec("tower_gun_base", "wooden metal mount", "tower_gun_idle"))
    specs.extend(head_specs("tower_gun_head", "rapid machine gun", "tower_gun_idle"))

    specs.append(base_spec("tower_cannon_base", "heavy metal platform", "tower_cannon_idle"))
    specs.extend(head_specs("tower_cannon_head", "missile launcher cannon", "tower_cannon_idle"))

    specs.append(base_spec("tower_goo_base", "slime tank platform", "tower_goo_idle"))
    specs.extend(head_specs("tower_goo_head", "goo slime cannon", "tower_goo_idle"))

    specs.append(base_spec("tower_sniper_base", "mortar platform", "tower_sniper_idle"))
    specs.extend(head_specs("tower_sniper_head", "long mortar sniper barrel", "tower_sniper_idle"))

    # Vertical beam towers — front-facing, tall, effects shoot upward/outward
    specs.append(
        spec(
            "tower_tesla_vertical",
            "vertical front-view tesla lightning coil tower, tall electric coil on metal base, "
            "thick blue electricity arcs bursting upward from the top coil, no horizontal rotation",
        )
    )
    specs.append(
        spec(
            "tower_flame_vertical",
            "vertical front-view flame thrower tower, tall fuel tanks with nozzle on top, "
            "large roaring orange flames blasting upward and forward from nozzle, no rotation",
        )
    )

    print(f"Output: {OUT}")
    print(f"Specs: {len(specs)}")
    gen = assets.try_load_image_generator()
    if gen is None:
        print("Could not load Z-Image-Turbo")
        return 1
    t0 = time.time()
    paths = assets.generate_assets(specs, session_dir=OUT, cache_dir=CACHE, image_generator=gen)
    print(f"Generated {len(paths)} in {time.time() - t0:.1f}s")
    for name in sorted(paths):
        print(f"  {name}")
    return 0 if paths else 2


if __name__ == "__main__":
    raise SystemExit(main())
