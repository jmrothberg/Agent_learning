"""Report post-processing gates for GameAgent.

Moved VERBATIM from `GameAgent` (no behavior change).
"""

from __future__ import annotations

from typing import Any


class GateProcessingMixin:

    """Report post-processing gates for GameAgent."""

    def _apply_undrawn_art_intent_gate(self, report: dict[str, Any]) -> None:

        """Phase 0C (Fieldrunners trace 20260626_102307): keep

        ASSETS_LOADED_BUT_UNDRAWN BLOCKING on the first build OR while an

        art-change request is still UNHONORED — but stay ADVISORY after a

        successful mid-session asset gen (golden iter 2: user satisfied despite

        undrawn pose sprites).



        tools.py demotes undrawn to a non-blocking `warnings` entry once

        behavioral probes are green. That demotion is correct for repeated

        pose-sprite cosmetics across iters (dojo-fight unwinnable-loop fix),

        but it ALSO lets a genuine iter-1 wiring gap (assets loaded, never

        drawn) ship silently. This promotes the advisory back to a blocking

        soft_warning ONLY in the two cases where undrawn art is a real,

        fixable iter-1 / art-feedback gap.



        Golden-safe: golden iter 2 is mid-session (snapshot_n>1) and the model

        emitted <assets> that turn, which clears `_unhonored_asset_request`

        BEFORE scoring — so neither condition fires and the advisory stays

        advisory. Standing dead-sprite rule is respected: this gate targets

        loaded-but-undrawn WIRING gaps (fixable), never pose-frame cosmetics.

        """

        warnings = report.get("warnings") or []

        advisory_undrawn = [

            w for w in warnings

            if isinstance(w, str)

            and "ASSETS_LOADED_BUT_UNDRAWN" in w

            and w.lstrip().startswith("ADVISORY")

        ]

        if not advisory_undrawn:

            return

        first_build_with_art = self._snapshot_n <= 1 and bool(self._session_assets)

        art_feedback_pending = bool(getattr(self, "_unhonored_asset_request", None))

        if not (first_build_with_art or art_feedback_pending):

            return

        for w in advisory_undrawn:

            warnings.remove(w)

            report.setdefault("soft_warnings", []).append(

                w.replace("ADVISORY (non-blocking) — ", "", 1)

            )

        report["warnings"] = warnings

        report["ok"] = False

        self._trace({

            "kind": "undrawn_art_intent_gate_promoted",

            "count": len(advisory_undrawn),

            "first_build_with_art": first_build_with_art,

            "art_feedback_pending": art_feedback_pending,

        })



    def _apply_dropped_assets_pending_gate(self, report: dict[str, Any]) -> None:
        """Re-warn every iter when asset_overflow dropped names still missing.

        run_09 Fieldrunners: harness dropped 14/38 sprites then tests went
        green, so the model never re-requested tesla/flame/enemy art. Unlike
        ASSETS_LOADED_BUT_UNDRAWN (loaded but not drawn), these PNGs were
        never generated — keep blocking until they land on disk.
        """
        pending = [
            n for n in (getattr(self, "_pending_dropped_assets", None) or [])
            if n and n not in (getattr(self, "_session_assets", None) or {})
        ]
        if not pending:
            return
        preview = ", ".join(pending[:12])
        if len(pending) > 12:
            preview += f", … (+{len(pending) - 12} more)"
        msg = (
            f"ASSETS_DROPPED_PENDING [{preview}]: the harness per-turn cap "
            "dropped these sprites — they do NOT exist on disk yet. Emit "
            "another <assets> block (mid-session) with just the missing "
            "names, or use from_image chaining. Do NOT mark the game done "
            "while these are still missing."
        )
        soft = report.setdefault("soft_warnings", [])
        if not any(isinstance(w, str) and "ASSETS_DROPPED_PENDING" in w for w in soft):
            soft.append(msg)
        report["ok"] = False
        self._trace({
            "kind": "dropped_assets_pending_gate",
            "count": len(pending),
            "names": pending[:24],
        })



    @staticmethod

    def _classify_failure(

        *,

        ok: bool,

        materialized: bool,

        stall_reason: str | None,

        coaching_suppressed: bool,

        asset_reprompt_cleared: bool,

        art_intent: bool,

        undrawn_present: bool,

        probes_all_passed: bool = False,

        has_page_errors: bool = False,

        has_soft_warnings: bool = False,

        launch_playfield_probe_failed: bool = False,

    ) -> tuple[str, str]:

        """Phase 4 (4D.2): bucket an iter into the layer that needs the fix.



        ADVISORY trace metadata only — never gates anything. The whole point of

        reading the .jsonl (an LLM-only artifact) is to decide WHERE a fix goes:



          - harness_bug     -> fix agent/harness CODE

          - memory_gap      -> add canned EXPERT guidance to memory/ (not the

                               local model)

          - local_llm_limit -> the local model's own limitation; mitigate via

                               prompt/format forcing, not code logic

          - none            -> nothing actionable to triage



        Precedence harness_bug > memory_gap > local_llm_limit: a harness

        contradiction is the most actionable and masks the others. Pure

        function (inputs are explicit) so it is unit-testable without a session.

        """

        # 1. HARNESS_BUG — the harness contradicted a correct model/router turn.

        if coaching_suppressed:

            return (

                "harness_bug",

                "stall-recovery coaching suppressed on clean prior iter",

            )

        if asset_reprompt_cleared:

            return (

                "harness_bug",

                "standing art request cleared by router on vague retry",

            )

        # 1c (run_04 T-1): "model right, harness wrong." A structural

        # soft_warning gate flipped ok=False even though EVERY model probe

        # passed and there were no page errors — the build then typically ships

        # unchanged. This is the PLAYER-STUCK / keyboard-HEURISTIC / board-input

        # false-positive class this whole run repairs; before this it was

        # mislabeled `none` and was invisible to grep + serial-loop triage

        # counts. The art+undrawn combo is intentionally LEFT to the memory_gap

        # rule below (first-occurrence undrawn on an art build is a wiring gap,

        # not a pure harness contradiction).

        if (

            (not ok) and materialized and probes_all_passed

            and not has_page_errors and has_soft_warnings

            and not (art_intent and undrawn_present)

        ):

            return (

                "harness_bug",

                "ok=False but all model probes passed with no page errors — "

                "a structural soft_warning gate over-fired on a correct build",

            )

        # 1b. Launch/playfield auto-probe failure beats undrawn-assets triage
        # (pinball trace 20260701_211752: ball stuck in lane but failure_class
        # was mislabeled as sprite wiring). Mechanism-general: any auto_* probe
        # whose name includes launch/playfield/enter.
        if (not ok) and materialized and launch_playfield_probe_failed:

            return (

                "memory_gap",

                "launch/playfield physics — outline/playbook should pre-empt "

                "lane geometry and exit velocity",

            )

        # 2. MEMORY_GAP — a FAILING materialize with an avoidable mistake that

        #    canned expert guidance (outline/playbook) should have pre-empted.

        #    Guard on `not ok`: on a CLEAN iter (ok=True) the off-screen scene

        #    backgrounds of a multi-scene art game are legitimately "loaded but

        #    undrawn" at headless test time (a non-gating ADVISORY in

        #    report["warnings"]), so without this guard every clean shipping iter

        #    of an art game was mislabeled memory_gap and inherited the prior

        #    iter's reason — golden trace build-a-dragon-s-lair-laserdis_20260626_224306

        #    iter 2 (ok=True, soft_warnings=[]) is the canonical example.

        if (not ok) and materialized and art_intent and undrawn_present:

            return (

                "memory_gap",

                "assets loaded but undrawn on art-intent build — "

                "outline/playbook should pre-empt the wiring",

            )

        # 3. LOCAL_LLM_LIMIT — model-side stall with no harness contradiction.

        if stall_reason:

            return ("local_llm_limit", f"model stream stalled: {stall_reason}")

        # 4. none — ok, or non-ok with no clearer triage signal (normal

        #    iteration friction the model will work through next turn).

        return ("none", "")



    @staticmethod

    def _synthetic_report_no_browser(mp: dict[str, Any]) -> dict[str, Any]:

        """Minimal test report when browser=None (eval_seed_edits materialization).



        micro_probes already ran; Chromium is intentionally absent. ok=True

        means structural pre-flight passed — NOT that gameplay was verified.

        """

        return {

            "ok": True,

            "probes": [],

            "soft_warnings": [],

            "warnings": list(mp.get("warnings") or []),

            "page_errors": [],

            "console_errors": [],

            "title": "(browser skipped — no_browser)",

            "canvas": None,

            "input_listeners": {},

            "input_test": None,

            "frozen_canvas": None,

            "body_chars": 0,

            "body_sample": "",

            "logs": [],

            "test_skipped": "no_browser",

        }



    def _apply_dead_animation_check_to_report(self, report: dict[str, Any]) -> None:

        """Surface dead animation frames as ADVISORY — never a hard ok=False gate.



        A `from_image` frame that came back near-identical to its idle parent

        (tracked in `self._dead_anim_frames`) means the limbs never moved — the

        character will just slide as a static image. That's worth telling the

        model, but it must NOT block shipping or starve the session.



        WHY ADVISORY, NOT BLOCKING (changed 2026-06-01, trace

        build-a-single-screen-2d-fight_20260531_214215 run_…214215): this used

        to flip report["ok"]=False, which created an UNWINNABLE loop. The dojo-

        fight session — with BOTH a local model (qwen3.6-27B) and SOTA

        (Opus 4.8) — corrected the actual gameplay perfectly (patches applied

        4/4 then 3/3, behavioral probes 8/8, input test PASS) yet every iter

        stayed ok=False on this one cosmetic sprite warning. Two compounding

        traps made it impossible to clear:

          1. The prescribed fix (re-emit `from_image` strength 0.55-0.65) is

             the SAME img2img path the user's own A/B finding documents as

             BROKEN — "pose frames must be TXT2IMG, not img2img" — so the regen

             came back dead again and RE-armed the block.

          2. While ok stayed False, the user's real gameplay feedback (slow the

             animation, flip the CPU facing) was deferred behind this blocker

             (`_should_defer_feedback_for_blocker`), so the model was never even

             allowed to make the simple code fix the user asked for.

        A cosmetic asset-quality signal the model cannot reliably fix must not

        gate a behaviorally-correct build. Behavioral probes gate; cosmetics

        inform. The signal still reaches the model three other ways that do NOT

        hard-fail: the rendered `warnings` block, the coaching channel, and the

        VLM critic note (see `_build_visual_critic_*`). The set still clears

        automatically when a frame regenerates with a real delta.

        """

        dead = self._dead_anim_frames

        if not dead:

            return

        names = ", ".join(f"`{n}`" for n in sorted(dead))

        # `warnings` (NOT `soft_warnings`) is the non-gating channel: tools.py

        # computes ok = (no errors) and (no soft_warnings), so anything in

        # soft_warnings is a hard fail. Route this advisory to `warnings` so it

        # is shown to the model without flipping ok. Do NOT touch report["ok"].

        #

        # Do NOT tell the model to "regenerate the pose frame" here. Both

        # regeneration routes are dead ends and suggesting either wastes the

        # correction loop (see DEV.md "Things to avoid" + the user memory

        # feedback_sprite_animation_from_image.md):

        #   - img2img can't change a pose at all (guidance_scale=0 keeps it

        #     locked to idle — proven A/B 2026-05-30), and

        #   - a fresh txt2img replacement frame will NOT stay consistent with

        #     the character already placed in the running game (per the user:

        #     consistency is the hard constraint), and in the dojo-fight trace

        #     the txt2img path STILL returned near-idle `block` frames anyway.

        # So this is purely informational: name the dead frames, say it's

        # cosmetic, and let the behaviorally-correct build ship.

        warns = list(report.get("warnings") or [])

        warns.append(

            "DEAD ANIMATION (advisory — does not block shipping): these frames "

            f"came back near-identical to their idle parent — {names}. Their "

            "limbs barely move, so the character looks near-static for that "

            "pose. This is a cosmetic sprite-quality note, NOT a code bug: do "

            "not change game logic for it and do not try to regenerate the "

            "frames in your patch. It will not stop a behaviorally-correct game "

            "from shipping."

        )

        report["warnings"] = warns

        report["dead_anim_frames"] = dict(dead)



    def _apply_still_frame_frozen_downgrade(self, report: dict[str, Any]) -> None:

        """Neutralize the FROZEN-AT-IDLE warning for still-frame recipes.



        For a laserdisc/cutscene QTE game the matched recipe declares

        `still_frame: true` ("Motion is intentionally still-frame… do not

        judge smoothness"). The harness's FROZEN-AT-IDLE warning (already

        non-gating — input is responsive) nonetheless tells the model to "add

        a subtle continuous idle animation (breathing/bob)". A strong model

        obeys and adds unrequested motion to a clean still-frame build (trace

        20260613_213711, Opus 4.8 iter 5 added a breathing-bob sine wave).

        When the active recipe is still-frame, replace that warning text with

        a neutral advisory that the static cadence is expected. Only the

        idle-by-design warning is touched; a TRUE freeze still goes through

        the gating soft_warnings channel in tools.py untouched.

        """

        if not getattr(self, "_active_visual_playtest_still_frame", False):

            return

        warns = report.get("warnings") or []

        if not warns:

            return

        new_warns: list[str] = []

        replaced = False

        for w in warns:

            if isinstance(w, str) and w.startswith("FROZEN-AT-IDLE"):

                new_warns.append(

                    "STILL-FRAME CADENCE (advisory — not a problem): the canvas "

                    "is static between frames, which is INTENDED for this "

                    "laserdisc/cutscene-style game. Do NOT add a breathing/bob "

                    "or any continuous idle motion to silence this — still-frame "

                    "cadence is the desired look. No change needed."

                )

                replaced = True

            else:

                new_warns.append(w)

        if replaced:

            report["warnings"] = new_warns



    def _apply_player_stuck_downgrade(self, report: dict[str, Any]) -> None:

        """Demote PLAYER-STUCK to a non-gating advisory when corroborated.



        The PLAYER-STUCK soft_warning (tools.py) is a STRUCTURAL proxy for "I

        can see the player isn't moving" — but with `/vlm-critique` off (the

        default) nothing visual corroborates it, so a genre where the player

        legitimately does not free-roam (a laserdisc/cutscene QTE: input

        ADVANCES scenes, it does not walk an entity around) trips it on every

        iter and can never clear it. That is the unwinnable-loop shape (see

        `_apply_dead_animation_check_to_report` / DEV.md dead-sprite gate):

        a correct build stays ok=False forever on one structural warning.



        Trace pin phase-a-requirement-your-plann_20260615_121048: a correct

        Dragon's-Lair QTE passed 10/10 model probes with 0 errors for 11 iters

        but PLAYER-STUCK kept ok=False until the model corrupted the build.



        Downgrade (move the warning from gating `soft_warnings` to advisory

        `warnings`, then recompute ok) ONLY when one genre-free corroboration

        holds, so a genuine "stuck in a wall" (Pac-Man) still hard-gates:

          - the matched visual-playtest recipe declares still-frame

            (`_active_visual_playtest_still_frame`) — a data-layer signal that

            this game is cutscene/QTE-style with no free movement by design, OR

          - a model-declared DYNAMIC probe passed this iter (reuse the

            `_is_dynamic_probe` regex over report["probes"]) — behavioral proof

            the controls drive intended state change even with no VLM watching.

        """

        if not report.get("player_stuck"):

            return

        sw = list(report.get("soft_warnings") or [])

        stuck = [w for w in sw if isinstance(w, str) and w.startswith("PLAYER-STUCK")]

        if not stuck:

            return

        still_frame = bool(getattr(self, "_active_visual_playtest_still_frame", False))

        dynamic_probe_passed = any(

            bool(p.get("ok")) and type(self)._is_dynamic_probe(str(p.get("expr") or ""))

            for p in (report.get("probes") or [])

        )

        if not (still_frame or dynamic_probe_passed):

            return

        # Keep every other soft_warning gating; only drop PLAYER-STUCK.

        report["soft_warnings"] = [w for w in sw if w not in stuck]

        warns = list(report.get("warnings") or [])

        warns.append(

            "PLAYER-STUCK (advisory — does not block shipping): a movement key "

            "registered input but no POSITION field changed. Corroboration shows "

            "this is not a stuck player — the controls drive intended state "

            "changes (a passing dynamic probe and/or a still-frame/QTE game where "

            "the player advances scenes rather than free-roaming). If you DID "

            "intend a free-moving player, make at least one direction change its "

            "x/y position; otherwise no change is needed."

        )

        report["warnings"] = warns

        # ok recompute mirrors tools.py: (no errors) and (no soft_warnings).

        report["ok"] = not (report.get("errors") or []) and not report["soft_warnings"]

        self._trace({

            "kind": "player_stuck_downgraded",

            "still_frame": still_frame,

            "dynamic_probe_passed": dynamic_probe_passed,

        })



    # Probe-result handlers (_apply_impossible_probe_downgrade_to_report,

    # _probe_shape_key, _refresh_probe_error_fields, _remove_probe_warnings,

    # _ALL_PROBES_QUARANTINED_GATE_CAP, _handle_probe_eval_errors) were extracted

    # to agent_probes.py (ProbeHandlingMixin); GameAgent inherits them unchanged.



