"""research.py — DEPRECATED Wikipedia reference fetcher (now a stub).

REMOVED 2026-06-24. This module used to look the goal up on Wikipedia
before planning and prepend a <reference> block to the planning turn,
on the theory that a local model's thin world knowledge (e.g. building
Space Invaders when asked for "Missile Command") could be grounded in
real mechanics.

Why it was removed: empirical testing (2026-05-19) returned 0/10 hits on
ten representative game goals (asteroids, pacman, donkey kong, space
invaders, missile command, street fighter, doom, snake, 2d roguelike,
tetris) for ~38s of cumulative network latency with no benefit — the
HTTPS handshake to en.wikipedia.org also failed on the framework Python
build (empty CA bundle -> CERTIFICATE_VERIFY_FAILED), so the lookup was
pure plan-time tax. Grounding now comes entirely from the curated
opening library under memory/ (implementation_outlines, components,
playtests, asset_audits, animation_audits) which is genre-free,
offline, and hand-verified.

The file is retained (not deleted) so any lingering import resolves and
the historical rationale stays discoverable. `fetch()` is now a no-op
returning "" — the agent no longer calls it.

Public API (preserved, inert):

    research.fetch(goal: str) -> str
        Always returns "" (no reference). Use memory/ instead.
"""

from __future__ import annotations


def fetch(goal: str) -> str:
    """Deprecated no-op. Wikipedia research was removed 2026-06-24 (0/10
    empirical hit rate); grounding comes from the curated memory/ opening
    library instead. Always returns "" so callers render no <reference>
    block."""
    return ""
