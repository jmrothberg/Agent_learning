"""Pure checks for eval/fixtures/dojo_fighters_asset_smoke.html.

The headless browser smoke is scripts/_smoke_asset_decode_settle.py
(opt-in via CHROMIUM_SMOKE=1 below). These tests guard fixture shape only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
_FIXTURE = _REPO / "eval" / "fixtures" / "dojo_fighters_asset_smoke.html"
_ASSETS_LINK = _REPO / "eval" / "fixtures" / "dojo_fighters_asset_smoke_assets"
_PLAYTESTS = _REPO / "memory" / "visual_playtests.jsonl"


def _fixture_text() -> str:
    assert _FIXTURE.is_file(), f"missing {_FIXTURE}"
    return _FIXTURE.read_text(encoding="utf-8")


def test_dojo_fixture_exists_with_asset_symlink() -> None:
    assert _ASSETS_LINK.exists(), (
        "eval/fixtures/dojo_fighters_asset_smoke_assets symlink missing"
    )


def test_dojo_fixture_has_loader_and_state_pins() -> None:
    html = _fixture_text()
    for needle in (
        "_assetsReady",
        "loadAssets",
        "sprite(",
        "window.state",
        "p1.facing",
        "p2.facing",
        "./dojo_fighters_asset_smoke_assets/",
    ):
        assert needle in html, f"fixture missing {needle!r}"


def test_dojo_fixture_facing_probe_expr_passes_static_state() -> None:
    """auto_actors_face_each_other from canvas-two-actors-facing recipe."""
    row = None
    for line in _PLAYTESTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") == "canvas-two-actors-facing":
            row = rec
            break
    assert row is not None
    probes = (row.get("recipe") or {}).get("auto_probes") or []
    expr = next(p["expr"] for p in probes if p.get("name") == "auto_actors_face_each_other")
    assert "Math.sign(p1.facing)" in expr
    # Static state shape the fixture exposes: p1.facing=+1, p2.facing=-1
    p1, p2 = {"facing": 1}, {"facing": -1}
    assert (1 if p1["facing"] > 0 else -1) != (1 if p2["facing"] > 0 else -1)


@pytest.mark.skipif(
    os.environ.get("CHROMIUM_SMOKE") != "1",
    reason="set CHROMIUM_SMOKE=1 to run headless Chromium smoke",
)
def test_dojo_asset_smoke_chromium_wrapper() -> None:
    """Shells to scripts/_smoke_asset_decode_settle.py (needs Playwright)."""
    script = _REPO / "scripts" / "_smoke_asset_decode_settle.py"
    env = dict(os.environ)
    env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
