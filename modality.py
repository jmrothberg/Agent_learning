"""Shared game-rendering-modality detection (single source of truth).

Both the planner (`prompts_v1.py`) and the memory retriever (`memory.py`)
need to classify a goal's rendering MODALITY (is this a 3D/first-person
game?). They used to keep two private copies of the 3D keyword set and the
`_detect_3d_intent` function, with a hand-written "keep in sync if either
side changes" comment — i.e. a drift hazard. If the two copies disagreed,
the planner and the skeleton/outline retrieval would classify the same goal
differently and the WRONG template could fire.

This module is the one shared copy so they cannot diverge. It is genre-free
per the repo standing rule: these tokens describe rendering SHAPE
(first-person, voxel, raycaster), never subject matter (no game titles).
Self-contained — imports nothing from prompts_v1 / memory, so there is no
import cycle.
"""
from __future__ import annotations

import re

# 3D rendering modality. Genre-free rendering-MODALITY tokens only — specific
# 3D game TITLES (doom, wolfenstein, minecraft) live in the data layer
# (memory/visual_playtests.jsonl strong_hooks), which routes them via
# _recipe_routed_skeleton. Project rule: no hardcoded genre/title lists.
THREE_D_KEYWORDS: frozenset[str] = frozenset({
    "3d", "three", "threejs",
    "first-person", "firstperson", "fps",
    "raycaster", "raycasting", "raycast",
    "voxel", "voxels",
    "perspective",
})


def detect_3d_intent(goal: str) -> list[str]:
    """Return a list of 3D-modality keywords found in `goal`. Empty list
    means the goal is plain 2D / DOM-only and needs no 3D nudge.
    Single-token matches; multi-token phrases like "first person" are
    detected by joining adjacent words and checking the joined form.

    Tokenizer keeps digits so "3D" is matched as "3d" (not stripped to
    "d"). Lowercased so the keyword set can stay all-lowercase.
    """
    if not goal:
        return []
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", goal)]
    out: list[str] = []
    seen: set[str] = set()
    # Single-word match.
    for w in words:
        if w in THREE_D_KEYWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    # Two-word join match for "first person", "doom like", etc.
    for i in range(len(words) - 1):
        j = words[i] + words[i + 1]
        if j in THREE_D_KEYWORDS and j not in seen:
            seen.add(j)
            out.append(j)
        # And with hyphen variant
        jh = words[i] + "-" + words[i + 1]
        if jh in THREE_D_KEYWORDS and jh not in seen:
            seen.add(jh)
            out.append(jh)
    return out
