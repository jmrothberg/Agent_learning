"""Tests for OpenCoder-inspired retrieval upgrades in memory.py.

  - Quality-ranked retrieval (#3): bullets with positive net helpful score
    rank above identical-relevance bullets with negative score.
  - Two-stage retrieval (#1): "code" stage drops bullets with score ≤ -2;
    "plan" stage keeps them.
  - Shingle dedup (#5): near-duplicate bullets collapse, keeping the
    first (highest-ranked).
  - 80/16 budget cap (#2): rendered playbook block stays under the
    requested char budget; tail bullets are trimmed first.
"""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: F401  (used in some test fixtures)

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory  # noqa: E402
from memory import (  # noqa: E402
    Bullet,
    BulletHit,
    Playbook,
    cap_hits_by_budget,
    dedup_hits,
    lookup_bullet,
    render_playbook_block,
)


# ---------------------------------------------------------------------------
# Quality-ranked retrieval (#3)
# ---------------------------------------------------------------------------


def _make_playbook(tmp_path: Path, bullets: list[Bullet]) -> Playbook:
    pb = Playbook(root=str(tmp_path / "memory"))
    pb.ensure()
    pb._save_all(bullets)
    return pb


def test_quality_multiplier_orders_winners_above_losers(tmp_path):
    """Two identically-relevant bullets — winner ranks above loser."""
    bullets = [
        Bullet(id="winner", content="ship rotation thrust vector",
               tags=["ship", "thrust"], helpful=10, harmful=0),
        Bullet(id="loser", content="ship rotation thrust vector",
               tags=["ship", "thrust"], helpful=0, harmful=10),
    ]
    pb = _make_playbook(tmp_path, bullets)
    hits = pb.retrieve("ship thrust vector", stage="plan")
    assert hits[0].bullet.id == "winner"
    assert hits[-1].bullet.id == "loser"


def test_quality_multiplier_does_not_overpower_relevance(tmp_path):
    """A heavy-winner bullet on an UNRELATED topic must NOT outrank a
    lighter-winner bullet on the actual topic."""
    bullets = [
        Bullet(id="off_topic_winner", content="ideas about rendering tilemaps",
               tags=["tilemap"], helpful=100, harmful=0),
        Bullet(id="on_topic", content="ship rotation thrust vector",
               tags=["ship", "thrust"], helpful=1, harmful=0),
    ]
    pb = _make_playbook(tmp_path, bullets)
    hits = pb.retrieve("ship thrust vector", stage="plan")
    assert hits[0].bullet.id == "on_topic"


# ---------------------------------------------------------------------------
# Two-stage retrieval (#1)
# ---------------------------------------------------------------------------


def test_code_stage_drops_net_harmful_bullets(tmp_path):
    """Code stage drops bullets with score ≤ -2; plan stage keeps them."""
    bullets = [
        Bullet(id="ok", content="ship rotation thrust vector",
               tags=["ship"], helpful=2, harmful=0),
        Bullet(id="bad", content="ship rotation thrust vector",
               tags=["ship"], helpful=0, harmful=5),  # score = -5
    ]
    pb = _make_playbook(tmp_path, bullets)
    plan_ids = {h.bullet.id for h in pb.retrieve("ship thrust", stage="plan")}
    code_ids = {h.bullet.id for h in pb.retrieve("ship thrust", stage="code")}
    assert "bad" in plan_ids
    assert "bad" not in code_ids


def test_code_stage_keeps_mildly_harmful(tmp_path):
    """A score of -1 is mildly harmful but should still survive code stage —
    the threshold is ≤ -2."""
    bullets = [
        Bullet(id="mild", content="ship rotation thrust vector",
               tags=["ship"], helpful=0, harmful=1),
    ]
    pb = _make_playbook(tmp_path, bullets)
    code_ids = {h.bullet.id for h in pb.retrieve("ship thrust", stage="code")}
    assert "mild" in code_ids


# ---------------------------------------------------------------------------
# Shingle dedup (#5)
# ---------------------------------------------------------------------------


def test_dedup_drops_near_duplicates():
    hits = [
        BulletHit(Bullet(id="a", content="apply thrust to ship using sin and cos of facing angle"), 0.5),
        BulletHit(Bullet(id="a2", content="apply thrust to ship using sin and cos of facing angle"), 0.4),  # exact dup
        BulletHit(Bullet(id="b", content="canvas DPR scaling for HiDPI displays"), 0.3),
    ]
    out = dedup_hits(hits)
    ids = [h.bullet.id for h in out]
    assert "a" in ids
    assert "a2" not in ids
    assert "b" in ids


def test_dedup_preserves_input_order():
    hits = [
        BulletHit(Bullet(id="x", content="canvas dpr scaling fixes hidpi blurriness"), 0.5),
        BulletHit(Bullet(id="y", content="ship thrust uses sin cos of facing angle"), 0.4),
        BulletHit(Bullet(id="z", content="frame loop uses requestAnimationFrame not setInterval"), 0.3),
    ]
    out = dedup_hits(hits)
    ids = [h.bullet.id for h in out]
    assert ids == ["x", "y", "z"]


def test_dedup_threshold_lets_distinct_bullets_through():
    """Two bullets on the same broad topic but with distinct prose should
    both survive dedup."""
    hits = [
        BulletHit(Bullet(id="a", content="ship rotation uses cos of angle for x velocity"), 0.5),
        BulletHit(Bullet(id="b", content="asteroid polygon vertices generated with random jitter per radius"), 0.4),
    ]
    out = dedup_hits(hits)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Budget cap (#2)
# ---------------------------------------------------------------------------


def test_budget_cap_truncates_tail():
    """Tight budget keeps only the first 1-2 bullets; tail is dropped."""
    big = "a" * 800  # ~830 chars per bullet after wrapping
    hits = [
        BulletHit(Bullet(id="b1", content=big), 0.5),
        BulletHit(Bullet(id="b2", content=big), 0.4),
        BulletHit(Bullet(id="b3", content=big), 0.3),
    ]
    capped = cap_hits_by_budget(hits, char_budget=1500)
    assert 0 < len(capped) < len(hits)
    assert capped[0].bullet.id == "b1"


def test_budget_cap_includes_first_when_oversized():
    """A single bullet larger than the budget still gets included — we
    never return empty when the input had signal."""
    hits = [BulletHit(Bullet(id="huge", content="x" * 10000), 0.5)]
    capped = cap_hits_by_budget(hits, char_budget=1000)
    assert len(capped) == 1


def test_render_playbook_respects_budget():
    """End-to-end: render_playbook_block applies dedup + budget."""
    hits = [
        BulletHit(Bullet(id=f"b{i}", content=f"unique bullet number {i}: " + ("x" * 200)), 0.5 - i * 0.01)
        for i in range(10)
    ]
    block = render_playbook_block(hits, char_budget=600)
    assert len(block) <= 800  # budget + header overhead margin
    # Should have at least one entry but not all 10.
    assert "b0" in block
    assert "b9" not in block


def test_render_playbook_dedup_off_keeps_dups():
    hits = [
        BulletHit(Bullet(id="a", content="apply thrust using sin and cos of facing angle"), 0.5),
        BulletHit(Bullet(id="a2", content="apply thrust using sin and cos of facing angle"), 0.4),
    ]
    block = render_playbook_block(hits, dedup=False, char_budget=10000)
    assert "[a]" in block
    assert "[a2]" in block


def test_shingles_handles_short_text():
    """Edge case: text shorter than n-gram window should still produce a
    non-empty signature so dedup logic doesn't crash."""
    sh = memory._shingles("hi")
    assert sh  # non-empty
    # Not crashing on empty either.
    assert memory._shingles("") == set()


# ---------------------------------------------------------------------------
# Hybrid mode + lookup_bullet (roadmap item #3 — pi-mono skills pattern)
# ---------------------------------------------------------------------------


def _hits(n: int) -> list[BulletHit]:
    return [
        BulletHit(
            Bullet(
                id=f"bullet-{i}",
                content=f"Distinct bullet number {i} content " * 3,
                tags=[f"tag{i}", "shared"],
            ),
            0.5 - i * 0.01,
        )
        for i in range(n)
    ]


def test_hybrid_mode_renders_full_top_n_then_summary():
    """First `full_top_n` bullets get their full body; the rest render
    as ID + tags only."""
    hits = _hits(7)
    block = render_playbook_block(
        hits, mode="hybrid", full_top_n=3, char_budget=10000,
    )
    # Top 3 bullets have their content text.
    for i in range(3):
        assert f"Distinct bullet number {i}" in block, f"bullet-{i} missing"
    # Bullets 3..6 should appear as ID-only entries (tags shown, content NOT).
    for i in range(3, 7):
        assert f"[bullet-{i}]" in block
        assert f"Distinct bullet number {i}" not in block, f"bullet-{i} body leaked"
    assert "ADDITIONAL PLAYBOOK INDEX" in block
    assert "<lookup_bullet>" in block


def test_full_mode_renders_all_bodies():
    """mode='full' (default) renders every bullet's full body — no
    ADDITIONAL section."""
    hits = _hits(5)
    block = render_playbook_block(hits, mode="full", char_budget=10000)
    for i in range(5):
        assert f"Distinct bullet number {i}" in block
    assert "ADDITIONAL PLAYBOOK INDEX" not in block
    assert "<lookup_bullet>" not in block


def test_hybrid_mode_with_few_hits_no_index():
    """If there are <= full_top_n hits, hybrid behaves like full — no
    'ADDITIONAL' section gets emitted."""
    hits = _hits(2)
    block = render_playbook_block(
        hits, mode="hybrid", full_top_n=3, char_budget=10000,
    )
    assert "ADDITIONAL PLAYBOOK INDEX" not in block
    for i in range(2):
        assert f"Distinct bullet number {i}" in block


def test_lookup_bullet_finds_bullet_by_id(tmp_path: Path):
    bullets = [
        Bullet(id="alpha", content="alpha content", tags=["a"]),
        Bullet(id="beta", content="beta content", tags=["b"]),
    ]
    pb = Playbook(root=str(tmp_path / "memory"))
    pb.ensure()
    pb._save_all(bullets)
    found = lookup_bullet(pb, "beta")
    assert found is not None
    assert found.id == "beta"
    assert "beta content" in found.content


def test_lookup_bullet_returns_none_for_missing(tmp_path: Path):
    pb = Playbook(root=str(tmp_path / "memory"))
    pb.ensure()
    pb._save_all([Bullet(id="only", content="x", tags=[])])
    assert lookup_bullet(pb, "does-not-exist") is None


def test_hybrid_mode_index_shows_score_when_nonzero():
    """Bullet with helpful/harmful counters shows score in summary line."""
    hits = [
        BulletHit(
            Bullet(id=f"b{i}", content=f"body {i}" * 5,
                   tags=["x"], helpful=i, harmful=0), 0.5 - i * 0.01,
        )
        for i in range(5)
    ]
    block = render_playbook_block(
        hits, mode="hybrid", full_top_n=2, char_budget=10000,
    )
    # b3 (helpful=3) is in the summary section; should show score=+3.
    assert "score=+3" in block or "score=+4" in block
