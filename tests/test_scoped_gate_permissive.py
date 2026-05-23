"""Tests for the permissive scoped-gate behavior.

Motivating trace:
  games/traces/mechanics-first-person-shooter_20260523_171650.log

What went wrong on iter 4:
  - User feedback: "do not change any graphics, or major code changes,
    just shift the placement of the hand and pistal 15 more pixel to
    the right and 10 more more pixels up. dont change any other code,
    or assets just a shift..."
  - Classifier scored art_change=True (art nouns "hand"/"pistol"/
    "graphics" + media verb "shift" via the broader verb gate), no
    behavior_scope, no size_scope → mode='media_only'.
  - Model emitted a clean CSS <patch> (correct read of intent).
  - Agent REJECTED the patch with "SCOPED MEDIA: emit <assets>/<sounds>
    only; no <patch>/<html_file>." and retried.
  - Model obeyed the retry and regenerated the gun.png with a wrong
    prompt → "first gun is POINTING AT THE PLAYER!!"

Earlier attempt to fix this with a `_feedback_requests_placement_change`
regex was reverted — natural language is unbounded; classifier-by-regex
will always have gaps and the user shouldn't have to phrase requests in
a way our regex catches.

The right fix lives in `_scoped_reply_violation`: trust the model's
tag choice. The gate's only remaining job is to block <html_file>
rewrites on small-scope feedback (the auto-revert mechanism is the
safety net for any genuinely-bad patch).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    # Simulate an FPS session with art assets.
    a._session_assets = {
        "gun": "ignored.png",
        "gun_fire": "ignored.png",
        "imp": "ignored.png",
        "wall_stone": "ignored.png",
    }
    a._session_sounds = {}
    return a


_PATCH_REPLY = (
    "<patch>\n<<<<<<< SEARCH\n"
    "  #gun { transform:translateX(calc(-62% + 150px)); }\n"
    "=======\n"
    "  #gun { transform:translateX(calc(-62% + 165px)); }\n"
    ">>>>>>> REPLACE\n</patch>\n"
    "<notes>shift gun 15 px right</notes>"
)

_ASSETS_REPLY = (
    '<assets>[{"name":"gun","prompt":"pixel-art doom-style pistol"}]</assets>'
)

_HTML_REPLY = (
    "<html_file><!doctype html><html><body>"
    "<script>const X=1;</script></body></html></html_file>"
)


# --- the bug being fixed ----------------------------------------------------


def test_patch_accepted_even_when_classifier_chose_media_only(tmp_path: Path) -> None:
    """The DOOM trace failure: classifier chose media_only, model
    emitted a clean <patch>. Previously the gate rejected it; the gate
    must now accept the model's judgment."""
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "media_only",
        "max_patch_count": 1,
        "allowed_asset_names": sorted(a._session_assets.keys()),
        "allowed_sound_names": [],
        "media_name_lock": True,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "shift the pistol 15 pixels right",
    }
    violation = a._scoped_reply_violation(_PATCH_REPLY)
    assert violation is None, (
        f"expected no violation (model picked patch over assets — its "
        f"call to make), got {violation!r}"
    )


def test_assets_still_accepted_on_media_only_turn(tmp_path: Path) -> None:
    """Regression guard: a model that DOES pick <assets> when the
    classifier expected it must still be accepted."""
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "media_only",
        "max_patch_count": 1,
        "allowed_asset_names": sorted(a._session_assets.keys()),
        "allowed_sound_names": [],
        "media_name_lock": True,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "",
    }
    assert a._scoped_reply_violation(_ASSETS_REPLY) is None


def test_assets_accepted_on_single_patch_turn_when_model_judges_media(
    tmp_path: Path,
) -> None:
    """Mirror case: classifier picked single_patch (behavior change),
    model picked <assets> because it judged the user wants a visual
    regen. Trust the model."""
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "single_patch",
        "max_patch_count": 1,
        "allowed_asset_names": sorted(a._session_assets.keys()),
        "allowed_sound_names": [],
        "media_name_lock": False,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "",
    }
    assert a._scoped_reply_violation(_ASSETS_REPLY) is None


# --- what we STILL reject (the safety net) ----------------------------------


def test_html_rewrite_still_blocked_in_media_only_mode(tmp_path: Path) -> None:
    """The remaining job of the scoped gate: prevent the model from
    throwing away a working baseline by emitting a full <html_file>
    rewrite on a small-scope ask."""
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "media_only",
        "max_patch_count": 1,
        "allowed_asset_names": [],
        "allowed_sound_names": [],
        "media_name_lock": True,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "",
    }
    v = a._scoped_reply_violation(_HTML_REPLY)
    assert v is not None
    assert "full <html_file>" in v


def test_html_rewrite_still_blocked_in_single_patch_mode(tmp_path: Path) -> None:
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "single_patch",
        "max_patch_count": 1,
        "allowed_asset_names": [],
        "allowed_sound_names": [],
        "media_name_lock": False,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "",
    }
    v = a._scoped_reply_violation(_HTML_REPLY)
    assert v is not None
    assert "full <html_file>" in v


def test_empty_reply_still_rejected(tmp_path: Path) -> None:
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "media_only",
        "max_patch_count": 1,
        "allowed_asset_names": [],
        "allowed_sound_names": [],
        "media_name_lock": True,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "",
    }
    v = a._scoped_reply_violation("<notes>did nothing</notes>")
    assert v is not None
    assert "emit one <patch>" in v or "existing names" in v


# --- soft trace: visibility into classifier vs model divergence -------------


def test_classifier_override_emits_trace_event(tmp_path: Path) -> None:
    """When the model picks a tag the classifier didn't expect, we
    emit a trace event so postmortems can spot patterns (and we can
    adjust the classifier's nudges over time) — without needing to
    enumerate every English phrasing in regex."""
    import json
    a = _agent(tmp_path)
    a._scoped_constraints = {
        "mode": "media_only",
        "max_patch_count": 1,
        "allowed_asset_names": sorted(a._session_assets.keys()),
        "allowed_sound_names": [],
        "media_name_lock": True,
        "require_scope_probe": False,
        "probe_keywords": [],
        "preserve_baseline": True,
        "feedback_preview": "shift the pistol 15 pixels right",
    }
    a._scoped_reply_violation(_PATCH_REPLY)
    rows = [
        json.loads(line)
        for line in a.trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overrides = [r for r in rows if r.get("kind") == "scoped_classifier_overruled_by_model"]
    assert len(overrides) == 1
    assert overrides[0]["expected_mode"] == "media_only"
    assert overrides[0]["model_emitted"] == "patch_only"
    # feedback preview is included so the trace is greppable later.
    assert "pistol" in overrides[0]["feedback_preview"]
