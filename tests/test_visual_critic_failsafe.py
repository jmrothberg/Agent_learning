"""Visual-critic fail-safe + patch-repair hardening (2026-06-02).

Background — trace dragons-lair_allroles_test__run_20260602_181115 (and the
earlier Street Fighter run): the visual critic returned useless output every
iteration and the harness consumed it:
  - A working VLM said "I cannot see the screenshot" every iter (root cause
    still under instrumentation), wasting a critic turn.
  - In the SF run the model emitted a "can't see" preamble PLUS Q1:no..Q5:no,
    which parsed at rate 1.0 and was logged as a real "5 of 5 checks FAILED"
    and fed to the coaching loop — a phantom critique.
  - A non-vision model (DeepSeek-V4-Flash) handed an image HALLUCINATED a
    confident wrong description instead of erroring.
  - A malformed patch with a doubled =======/>>>>>>> REPLACE marker silently
    failed to apply and burned a whole iteration.

These tests pin the fail-safe behavior. All are pure-function / source-pinned;
no model or browser calls.
"""
from __future__ import annotations

import inspect

from memory import VisualPlaytestRecipe


def _recipe5():
    return VisualPlaytestRecipe(
        id="x", kind="visual_playtest", content="",
        recipe={"checklist": ["a", "b", "c", "d", "e"]},
    )


def test_parser_tolerates_prefill_doubled_ordinal():
    """ROOT-CAUSE fix (2026-06-03): the critic returned useless output because
    qwen3.6 (thinking VLM) SEES the image but answers in prose, never emitting
    Q1: lines. We now prefill the assistant turn with 'Q1: ' to force the
    format — but the model then continues with its OWN ordinal ('Q1: 1. YES').
    The parser must tolerate that doubled ordinal, else parse_rate stays 0 and
    the (now real) verdict is still dropped."""
    p = GameAgent._parse_visual_playtest_response(
        "Q1: 1. YES\nQ2: 2. NO\nQ3: 3. YES\nQ4: 4. YES\nQ5: 5. YES", _recipe5()
    )
    assert p["parse_rate"] == 1.0
    assert {k: v[0] for k, v in p["answers"].items()} == {
        1: "yes", 2: "no", 3: "yes", 4: "yes", 5: "yes"
    }
    # plain shape still parses
    p2 = GameAgent._parse_visual_playtest_response("Q1: yes\nQ2: no", _recipe5())
    assert p2["answers"][1][0] == "yes" and p2["answers"][2][0] == "no"


def test_critic_uses_format_forcing_prefill():
    """run_visual_critic must seed an assistant 'Q1: ' prefill on the recipe
    path so the VLM starts inside the answer format instead of reasoning prose,
    and re-attach it before parsing."""
    src = inspect.getsource(GameAgent.run_visual_critic)
    assert '"Q1: "' in src
    assert 'role": "assistant"' in src
    # re-attached onto the reply before parsing
    assert "_critic_prefill + critique_raw" in src

def test_critic_prefill_latch_disables_after_empty_completion():
    """2026-06-12: some backends return an EMPTY completion when given an
    assistant prefill (trace 20260612_004616 wasted one VLM call on 13/13
    iterations). After one empty prefilled response the agent must latch
    `_critic_prefill_broken` and skip the prefill for the session, tracing
    `critic_prefill_disabled` once."""
    src = inspect.getsource(GameAgent.run_visual_critic)
    # The latch gates the prefill expression…
    assert "_critic_prefill_broken" in src
    i_gate = src.index("_critic_prefill_broken")
    i_call = src.index("result = await backend.stream_chat")
    assert i_gate < i_call, "latch must gate the prefill BEFORE the VLM call"
    # …and an empty completion sets it + fires the trace event.
    assert "critic_prefill_disabled" in src
    assert "empty_completion_after_prefill" in src
    # Initialized False at session start.
    init_src = inspect.getsource(GameAgent.__init__)
    assert "_critic_prefill_broken: bool = False" in init_src


import agent
import patches
from agent import GameAgent


# ---- Fix 2: abstain detection ----------------------------------------------

def test_critic_abstain_detects_blind_replies():
    # ABSTAIN = the model says it never received/saw the IMAGE itself. The
    # phrasing is anchored to an image/screenshot object (tightened 2026-06-03).
    abstain = [
        "I cannot see the screenshot you mentioned.",
        "I'm unable to provide answers because no screenshot was included. Q1: no",
        "I cannot actually see the two screenshots you are referring to.",
        "Please share the screenshot you would like me to review.",
        "No image was provided with your message.",
        "I don't see an image attached to your message.",
        "I am unable to view the image.",
    ]
    for t in abstain:
        assert GameAgent._critic_abstained(t), f"should be ABSTAIN: {t!r}"


def test_critic_abstain_does_not_flag_real_critiques():
    # These are GENUINE visual observations from a model that SAW the image.
    # The old over-broad regex wrongly flagged "I don't see a projectile" /
    # "unable to see the character pose" as blindness and discarded the whole
    # critique — the bug that made the critic useless. They must NOT abstain.
    real = [
        "Q1: yes\nQ2: no — player clipped at the right edge",
        "The knight faces right but the attack renders to the left.",
        "The scene looks correct; both fighters are visible and facing each other.",
        "Q1: no\nQ2: no\nQ3: unclear",  # genuine failures, no abstain phrasing
        "Looking at the bottom left, there is a slingshot but I don't see a projectile loaded.",
        "I am unable to clearly see the character pose — it looks ambiguous in the frame.",
        "I do not see any health bars at the top of the screen.",
        "Q1: no — I don't see a second fighter on the right side",
    ]
    for t in real:
        assert not GameAgent._critic_abstained(t), f"should NOT be ABSTAIN: {t!r}"


def test_abstain_guard_wired_before_parsing_in_run_visual_critic():
    """The abstain check must run on critique_raw BEFORE the parse/coach path,
    and drop only a DEGENERATE (no genuine verdict) critique."""
    src = inspect.getsource(GameAgent.run_visual_critic)
    assert "_critic_abstained(critique_raw)" in src
    i = src.index("_critic_abstained(critique_raw)")
    after = src[i:i + 1200]
    # drops on degenerate verdict, keeps on mixed verdict
    assert "_degenerate" in after
    assert "visual_critic_abstained" in after and "return None" in after
    assert "visual_critic_abstain_overridden" in after  # mixed verdict kept


def test_abstain_only_drops_degenerate_verdicts_not_mixed():
    """The tightened rule (2026-06-02): a refusal phrase alone is NOT enough.
    Abstain (drop) only when the model produced no genuine verdict — nothing
    parsed, or every answer identical (the 'blind → defaulted everything to no'
    shape). A MIXED verdict (some yes, some no) is a real critique and kept,
    even if the text contains a hedging phrase. This mirrors the
    run_visual_critic decision logic in pure form so the contract is pinned."""
    def decide(answers):
        # answers: list of "yes"/"no"/"unclear" — returns True if ABSTAIN-drop
        return (not answers) or (len(set(answers)) <= 1)

    # degenerate → drop (Street Fighter "can't see → all no")
    assert decide([]) is True
    assert decide(["no", "no", "no", "no", "no"]) is True
    assert decide(["unclear", "unclear"]) is True
    # mixed verdict → keep (genuine critique that happens to hedge)
    assert decide(["yes", "no", "unclear"]) is False
    assert decide(["yes", "yes", "no"]) is False


# ---- Fix 1b: is_vlm guard --------------------------------------------------

def test_run_visual_critic_has_is_vlm_guard():
    """A non-vision backend must skip the critic, not hand it an image."""
    src = inspect.getsource(GameAgent.run_visual_critic)
    assert "await backend.is_vlm()" in src
    i = src.index("await backend.is_vlm()")
    after = src[i:i + 300]
    assert "visual_critic_skipped" in after
    assert "return None" in after


# ---- Fix 1: instrumentation (observability only) ---------------------------

def test_run_visual_critic_logs_payload_for_root_cause():
    src = inspect.getsource(GameAgent.run_visual_critic)
    assert "visual_critic_payload" in src
    # must record real byte sizes so we can see if pixels reach the backend
    assert "image_bytes" in src


# ---- Fix 3: patch repair drops the doubled divider -------------------------

def test_repair_collapses_doubled_divider_before_replace():
    bad = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "old\n"
        "=======\n"
        "new\n"
        "window.ASSETS = ASSETS;\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
        "</patch>"
    )
    fixed = patches.repair_reply(bad)
    # exactly one divider survives, and the patch now parses + extracts
    assert fixed.count("=======") == 1
    ps = patches.extract_patches(fixed)
    assert len(ps) == 1
    assert ps[0].search.strip() == "old"
    assert "window.ASSETS = ASSETS;" in ps[0].replace


def test_repair_leaves_wellformed_patch_untouched():
    good = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "a\n"
        "=======\n"
        "b\n"
        ">>>>>>> REPLACE\n"
        "</patch>"
    )
    fixed = patches.repair_reply(good)
    ps = patches.extract_patches(fixed)
    assert len(ps) == 1 and ps[0].search.strip() == "a" and ps[0].replace.strip() == "b"
