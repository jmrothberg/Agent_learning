"""Tests for the 2026-05-29 agent fixes:
  - Fix A: default context window = 250k (stop the 32k cutoff that made
    token-pressure compaction fire every turn on a 200k-context model).
  - Fix B: animation frames use the SAME model as the base (Z-Image img2img),
    not the foreign SD-Turbo, so derived frames are consistent.
  - Fix C: the agent no longer steers the model toward code-drawn limbs.

Pure/source-pin — no model or GPU needed.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import assets  # noqa: E402


# ---- Fix A: context default ------------------------------------------------

def test_default_num_ctx_is_100k():
    assert agent.DEFAULT_NUM_CTX == 100_000
    assert agent.DEFAULT_NUM_CTX <= agent.MAX_NUM_CTX


def test_low_pressure_no_compaction_with_big_window():
    # 45k-token prompt against a 250k window = 0.18 pressure → no lossy anchor,
    # even at many messages. (Mirrors the Anthropic case from the trace.)
    from test_token_aware_compaction import _Stub  # reuse the stub

    stub = _Stub(n_messages=30, pressure=45_000 / 250_000)
    agent.GameAgent._prune_messages(stub)
    assert not any(
        "STATE ANCHOR" in (m.get("content") or "") for m in stub._messages
    ), "must NOT compact at 0.18 pressure"
    assert len(stub._messages) == 30


# ---- Fix B: Z-Image img2img for animation frames ---------------------------

def test_zimage_generator_has_img2img():
    assert callable(getattr(assets.ZImageTurboGenerator, "generate_img2img", None))


def test_zimage_img2img_shares_components_no_extra_vram():
    src = inspect.getsource(assets.ZImageTurboGenerator.generate_img2img)
    assert "ZImageImg2ImgPipeline" in src
    # built from the already-loaded txt2img pipeline's components (shared VRAM)
    assert "self._pipeline.components" in src


def test_generate_assets_uses_txt2img_merged_for_pose_frames():
    src = inspect.getsource(assets.generate_assets)
    # 2026-05-30 (proven in animation_ab/): img2img at guidance_scale=0 stays
    # locked to the idle pose at every strength/model. Pose frames are now
    # TXT2IMG with the parent's character+style prompt merged with the pose
    # clause (fixed seed keeps the character). The from_image branch must NOT
    # call img2img.
    assert "pose_txt2img" in src
    # pose frames merge the parent character prompt with the pose clause and
    # generate via txt2img
    assert "_safe_generate(image_generator, merged)" in src
    # img2img is no longer called to generate pose frames
    assert "_safe_img2img(" not in src and "generate_img2img(" not in src


# ---- Fix C: stop steering toward code limbs --------------------------------

def test_asset_sanity_warning_does_not_suggest_code():
    src = inspect.getsource(agent.GameAgent._maybe_generate_assets_and_sounds)
    # the old wording told the model to "render the moving part ... in code"
    assert "render the moving part" not in src
    # it explicitly forbids code-drawn limbs
    assert "never draw the limb in code" in src
    # and steers to the from_image-done-right fix (seed from idle base, raise
    # strength, name the moved parts) — NOT to txt2img, which only loses the
    # consistent character (user directive 2026-05-30).
    assert "from_image" in src and "IDLE base" in src
    assert "TXT2IMG" not in src


def test_asset_block_forbids_code_limbs():
    src = inspect.getsource(assets.render_asset_paths_block)
    assert "no code-drawn limbs" in src or "NEVER draw a character" in src
