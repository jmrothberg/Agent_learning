"""Tests for the iter-1 asset-reference alignment scan.

Background: in the donkey-kong trace 20260513_122154 the model wrote an
`assetList` with 14 names but Phase A had only produced 8 assets. The
6 missing files generated browser `net::ERR_FILE_NOT_FOUND` errors
every iter — the model then patched symptoms (try/catch around
drawImage) for 3 iters without ever realizing the FILES weren't there.
The scan catches this gap deterministically.
"""

from agent import GameAgent
from backend import BackendInfo, make_backend


def _make_agent(tmp_path, monkeypatch):
    info = BackendInfo(
        name="ollama", model="dummy:0",
        source="test", endpoint="http://127.0.0.1:0",
    )
    backend = make_backend(info)
    agent = GameAgent(
        backend=backend,
        out_path=tmp_path / "game.html",
        max_iters=1,
    )
    return agent


def test_scan_extracts_bracket_index_refs():
    """ASSETS['name'] and ASSETS["name"] both count."""
    html = """
    const img = ASSETS['mario_idle'];
    const a = ASSETS["barrel"];
    """
    refs = GameAgent._scan_html_for_asset_refs(html)
    assert refs == {"mario_idle", "barrel"}


def test_scan_extracts_dot_refs():
    """ASSETS.name dotted access also counts."""
    html = """
    const img = ASSETS.mario_idle;
    ctx.drawImage(ASSETS.dk_throw1, x, y);
    """
    refs = GameAgent._scan_html_for_asset_refs(html)
    assert "mario_idle" in refs
    assert "dk_throw1" in refs


def test_scan_extracts_path_refs():
    """Literal file paths inside *_assets/ count too — covers the
    template-literal pattern `ASSET_DIR + '/' + name + '.png'`
    AFTER inlining, plus any hand-written paths."""
    html = """
    fetch('./my-game_assets/mario_jump.png');
    new Image().src = "./other_assets/ladder.png";
    """
    refs = GameAgent._scan_html_for_asset_refs(html)
    assert "mario_jump" in refs
    assert "ladder" in refs


def test_scan_extracts_template_assetlist():
    """The DK trace failure mode: a literal asset-name list in the
    HTML, used to build paths at runtime. We catch this by also
    detecting common array-of-string-names patterns."""
    html = """
    const assetList = [
      'mario_idle','mario_walk1','mario_walk2','mario_jump',
      'mario_climb1','mario_climb2',
      'dk_idle','dk_throw1','dk_throw2',
      'pauline_help','pauline_tap',
      'barrel','girder','ladder'
    ];
    function loadAssets() {
      const entries = assetList.map(n => [n, ASSET_DIR + '/' + n + '.png']);
    }
    """
    refs = GameAgent._scan_html_for_asset_refs(html)
    assert "mario_climb1" in refs
    assert "barrel" in refs
    assert "ladder" in refs
    assert "pauline_tap" in refs


def test_scan_ignores_unrelated_strings():
    """Don't false-positive on JS keywords or arbitrary string literals."""
    html = """
    const status = 'ready';
    const colors = ['red','blue','green'];
    const ASSETS = {};  // declaration, no reference yet
    """
    refs = GameAgent._scan_html_for_asset_refs(html)
    # 'red'/'blue'/'green' aren't ASSETS references and aren't in any
    # _assets/ path, so they shouldn't appear. The assetList heuristic
    # only triggers for arrays paired with asset-loader context.
    assert "red" not in refs
    assert "blue" not in refs


def test_alignment_detects_gap(tmp_path, monkeypatch):
    """When HTML references assets that weren't generated, the gap
    queues a coaching message naming the missing files."""
    agent = _make_agent(tmp_path, monkeypatch)
    # Simulate Phase A produced 3 assets.
    asset_dir = tmp_path / "session_assets"
    asset_dir.mkdir()
    for n in ("mario_idle", "mario_walk1", "dk_idle"):
        (asset_dir / f"{n}.png").write_bytes(b"\x89PNG")
        agent._session_assets[n] = asset_dir / f"{n}.png"

    # HTML asks for 5: 3 of those + 2 missing.
    html = """
    const a = ASSETS['mario_idle'];
    const b = ASSETS['mario_walk1'];
    const c = ASSETS['dk_idle'];
    const d = ASSETS['pauline_help'];
    const e = ASSETS['barrel'];
    """
    missing = agent._check_asset_alignment(html)
    assert missing == {"pauline_help", "barrel"}
    # Coaching was queued for the next user turn.
    assert agent._pending_coaching, "expected a coaching message for missing assets"
    text = agent._pending_coaching[-1]
    assert "pauline_help" in text and "barrel" in text
    assert "ERR_FILE_NOT_FOUND" in text  # explains the symptom
    assert "assets" in text.lower()  # tells model how to fix it


def test_alignment_no_gap_is_silent(tmp_path, monkeypatch):
    """When every referenced asset exists, no coaching fires."""
    agent = _make_agent(tmp_path, monkeypatch)
    asset_dir = tmp_path / "session_assets"
    asset_dir.mkdir()
    for n in ("mario_idle", "dk_idle"):
        (asset_dir / f"{n}.png").write_bytes(b"\x89PNG")
        agent._session_assets[n] = asset_dir / f"{n}.png"

    html = "const a = ASSETS['mario_idle']; const b = ASSETS['dk_idle'];"
    missing = agent._check_asset_alignment(html)
    assert missing == set()
    assert not agent._pending_coaching
