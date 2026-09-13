from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_probe_lint_detects_stale_runtime_shape_after_rewrite():
    """Old probes should be recognized as stale when new code lacks fields.

    This is intentionally generic: it checks shape mismatch, not a
    specific game genre.
    """
    probes = [
        {
            "name": "old_runtime_shape",
            "expr": "window.state && typeof state.oldRuntimeField === 'number'",
        }
    ]
    html = """<!doctype html><html><body><script>
const state = { player: { x: 10 }, newSystemCount: 3 };
window.state = state;
</script></body></html>"""

    findings = GameAgent._probes_referencing_unassigned_props(probes, html)
    assert findings
    assert findings[0]["name"] == "old_runtime_shape"
    assert findings[0]["kind"] == "unassigned_property_read"


def test_auto_prefixed_probes_skip_unassigned_property_lint():
    """Harness-injected auto_* probes use alias chains; model cannot fix them."""
    probes = [
        {
            "name": "auto_platformer_has_multiple_platforms",
            "expr": (
                "(()=>{const s=window.state;if(!s)return true;"
                "const plats=s.platforms||s.floors||s.ground;return true;})()"
            ),
        }
    ]
    html = """<!doctype html><html><body><script>
const state = { player: { x: 10, y: 20 } };
window.state = state;
</script></body></html>"""

    findings = GameAgent._probes_referencing_unassigned_props(probes, html)
    assert findings == []


def test_unused_media_warning_becomes_stale_context_on_rewrite():
    mp = {
        "warnings": [
            "sprite 'old.png' was generated to 'old_assets/old.png' but is NEVER referenced in the HTML. Either wire it in.",
            "ordinary warning",
        ],
        "stats": {"unused_assets": 1},
    }

    suppressed = GameAgent._mark_unused_media_as_stale_for_continuation(mp)

    assert suppressed == 1
    assert all("NEVER referenced" not in w for w in mp["warnings"])
    assert any("previous build appears stale" in w for w in mp["warnings"])
    assert "ordinary warning" in mp["warnings"]


def test_probe_lint_skips_getComputedStyle_cssom_reads():
    """DOOM3DF3: weapon_overlay_visible read getComputedStyle(w).visibility
    and was flagged as unassigned 3×. CSSOM properties are not game state."""
    probes = [
        {
            "name": "weapon_overlay_visible",
            "expr": (
                "(()=>{const w=document.getElementById('weapon');"
                "if(!w)return false;"
                "return getComputedStyle(w).visibility!=='hidden'"
                "&&getComputedStyle(w).display!=='none'"
                "&&getComputedStyle(w).opacity!=='0';})()"
            ),
        },
        {
            "name": "still_flags_real_unassigned",
            "expr": "window.state && typeof state.ghostField === 'number'",
        },
    ]
    html = """<!doctype html><html><body><div id=weapon></div><script>
const state = { player: { x: 10 } };
window.state = state;
</script></body></html>"""
    findings = GameAgent._probes_referencing_unassigned_props(probes, html)
    names = {f["name"] for f in findings}
    assert "weapon_overlay_visible" not in names
    assert "still_flags_real_unassigned" in names


def test_lint_probes_flags_cross_probe_name_reference():
    probes = [
        {"name": "canvas_640", "expr": "document.querySelector('canvas')"},
        {"name": "raf_ok", "expr": "(()=>canvas_640 && state.frame>0)()"},
    ]
    findings = GameAgent._lint_probes(probes)
    assert any(f["kind"] == "cross_probe_name_reference" for f in findings)
