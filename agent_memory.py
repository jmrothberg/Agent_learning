"""Playbook / opening-book / component / lean-budget memory retrieval extracted from agent.py.

`MemoryRetrievalMixin` holds the first-build/plan-turn memory assembly helpers
that `GameAgent` used to carry inline:

  - `_retrieve_playbook_block` — retrieve and render two-stage playbook bullets,
    including report-driven pinning and cross-modality suppression.
  - `_retrieve_opening_book_block` — retrieve the matched implementation outline
    (+ universal fallback), playtests, asset/animation audits, and the optional
    VLM checklist, rendered into the `<opening-book>` block.
  - `_retrieve_components_block` — retrieve mechanics-level JS snippets, pinning
    the engine skeleton (game loop + buffered input) to the protected FRONT for
    open-domain goals so they survive the lean budget.
  - `_apply_lean_memory_budget` — cap the COMBINED size of the three first-build
    memory blocks (opening-book > components > playbook) in lean mode.
  - `_detect_open_domain_build` — genre-free novelty check from the retrieved
    outline id/score, used to decide component count + pinning + protection.

These were moved VERBATIM out of `GameAgent` (no behavior change). `GameAgent`
inherits this mixin, so every `self.*` attribute reference (`self._memory`,
`self._criteria`, `self._session_assets`) and helper (`self._trace`,
`self._lean_prompt_active`) resolves unchanged through normal MRO lookup.
"""
from __future__ import annotations

import json

from memory import (
    BulletHit,
    OpeningBookHit,
    lookup_bullet,
    render_components_block,
    render_opening_book_block,
    render_outline_traps_only,
    render_playbook_block,
    render_vlm_checklist_section,
    VLM_CHECKLIST_SKIP_IDS,
)


class MemoryRetrievalMixin:
    """First-build / plan-turn memory retrieval for GameAgent (see docstring)."""

    # OpenCoder #1 — two-stage retrieval (broad-then-narrow). Plan stage
    # gets a wider, more permissive cut of the playbook (small models
    # benefit from "see the whole space"); code stage gets a tighter cut
    # of validated patterns only (no net-harmful bullets, fewer entries,
    # smaller char budget). Mirrors OpenCoder's two-stage SFT — broad
    # first, narrow second.
    _PLAN_STAGE_TOP_K_BONUS = 2          # plan retrieves K + bonus bullets
    _CODE_STAGE_TOP_K = 3                # narrow cut at code time
    _PLAN_STAGE_CHAR_BUDGET = 4500       # ~1100 tokens, broad context
    _CODE_STAGE_CHAR_BUDGET = 2400       # ~600 tokens, tight context

    @staticmethod
    def _playbook_suppressed_bullet_ids(
        *,
        goal: str = "",
        active_skeleton: str | None,
        code: str,
    ) -> set[str]:
        """Drop cross-modality navigation bullets that misfire on the active scaffold."""
        from modality import detect_fps_navigation_modality

        mod = detect_fps_navigation_modality(
            goal=goal,
            code=code or "",
            active_skeleton=active_skeleton,
        )
        suppressed: set[str] = set()
        if mod == "wireframe":
            suppressed.update({
                "fps-camera-and-movement-vectors",
                "fps-minimap-radar-yaw-arrow",
            })
        elif mod == "threejs":
            suppressed.update({
                "wireframe-fps-movement-vectors",
                "wireframe-minimap-radar-yaw-arrow",
            })
        elif mod == "mode7":
            suppressed.update({
                "fps-camera-and-movement-vectors",
                "wireframe-fps-movement-vectors",
                "fps-minimap-radar-yaw-arrow",
                "wireframe-minimap-radar-yaw-arrow",
            })
        return suppressed

    # Failing model probe name -> playbook ids to pin on fix turns (run_vlm10).
    _REPORT_PROBE_PLAYBOOK_PINS: dict[str, list[str]] = {
        "camera_moves": ["parallax-coordinate-camera", "beat-em-up-scroll-spawn"],
        # run_17: playfield / assetsReady probes need sticky bullets on fix turns
        "auto_body_enters_playfield": ["launch-into-playfield"],
        "player_sprite_visible_at_startup": ["image-load-race", "draw-generated-sprites-not-boxes"],
    }

    def _first_build_playbook_ensure_ids(self, goal: str) -> list[str] | None:
        """Pin class-level playbook bullets on first-build (not game titles).

        Policy: game-specific wording belongs in saved test prompts
        (`prompt_library` / eval goals). Memory pins use mechanism
        *classes* (visual recipe id + class phrases), never titles like
        Centipede/Galaga.
        """
        out: list[str] = []

        def _add(*ids: str) -> None:
            for bid in ids:
                s = str(bid).strip()
                if s and s not in out:
                    out.append(s)

        recipe = getattr(self, "_active_visual_playtest_recipe_id", None) or ""
        g = (goal or "").lower()

        # run_18: 3D/WebGL/voxel classes — do NOT pin drawImage bullets (Doom
        # green iters retrieved only those while art was THREE.Texture).
        webgl_or_voxel = (
            recipe in (
                "canvas-3d-first-person",
                "canvas-voxel-sandbox",
                "canvas-3d-racing",
            )
            or "three.js" in g
            or "webgl" in g
            or "voxel" in g
            or "first-person" in g
            or "first person" in g
        )
        # Wireframe / trench vector (2D canvas, not three.js).
        wireframe_class = (
            recipe == "canvas-vector-wireframe"
            or "wireframe" in g
            or "vector tank" in g
            or "vector trench" in g
            or ("trench" in g and "vector" in g)
        )
        voxel_class = (
            recipe == "canvas-voxel-sandbox"
            or "voxel" in g
            or ("block" in g and "place" in g and ("break" in g or "mine" in g))
        )
        fps_class = (
            recipe == "canvas-3d-first-person"
            or ("three.js" in g and ("first-person" in g or "first person" in g or "fps" in g or "maze" in g))
            or ("pointer-lock" in g or "pointer lock" in g)
        )

        if getattr(self, "_session_assets", None) and not (
            webgl_or_voxel or fps_class or voxel_class or wireframe_class
        ):
            _add("draw-generated-sprites-not-boxes", "image-load-race")

        if wireframe_class:
            _add(
                "projection-3d-wireframe",
                "wireframe-fps-movement-vectors",
                "vector-stroke-contrast",
            )
            if any(k in g for k in ("trench", "depth", "exhaust", "tie fighter")):
                _add("trench-depth-vector-spawn")

        if voxel_class:
            _add("voxel-mesh-simple-or-groups")

        if fps_class:
            _add(
                "fps-camera-and-movement-vectors",
                "3d-navigation-modality-invariants",
                "fps-minimap-radar-yaw-arrow",
            )

        # Fixed-shooter / top-down shooter *class* (not named games).
        fixed_shooter_class = (
            recipe in ("canvas-fixed-shooter", "canvas-top-down-action")
            or "fixed shooter" in g
            or "formation shooter" in g
            or "vertical-shooter" in g
            or "vertical shooter" in g
        )
        if fixed_shooter_class:
            _add("top-down-sprite-draw-orientation", "draw-fighters-large")

        # Segmented-follower *class* (history trail) — class words only.
        if any(k in g for k in ("segmented", "segments", "segment chain", "body segments")):
            _add("segmented-entity-follow")

        # Pinball / charged-launch *class*.
        pinball_class = (
            recipe == "canvas-pinball"
            or any(k in g for k in ("pinball", "flipper", "plunger", "launch lane"))
        )
        if pinball_class:
            _add("launch-into-playfield")

        # Climb-smash / character pose isolation (Rampage-class).
        if any(k in g for k in ("climb", "skyscraper", "punch/smash", "cling")):
            _add("character-sprite-isolation")

        return out or None

    def _playbook_ensure_ids_for_report(self, report: dict) -> list[str]:
        """Pin playbook bullets for harness blockers and visual-playtest refs."""
        out: list[str] = []

        def _add(*ids: str) -> None:
            for bid in ids:
                s = str(bid).strip()
                if s and s not in out:
                    out.append(s)

        failing = {
            str(p.get("name") or "")
            for p in (report.get("probes") or [])
            if not p.get("ok")
        }

        # Harness-warning pins (run_vlm10: undrawn / control-freeze / frozen canvas).
        soft = [str(w) for w in (report.get("soft_warnings") or [])]
        warns = [str(w) for w in (report.get("warnings") or [])]
        all_w = soft + warns
        if any("ASSETS_LOADED_BUT_UNDRAWN" in w for w in all_w):
            _add(
                "draw-generated-sprites-not-boxes",
                "animation-frames-consistent-character",
                "sprite-gen-wait-for-load",
                "image-load-race",
            )
        if any("HOTSPOT_ALIGNMENT_MISS" in w for w in soft):
            _add("pointclick-hotspot-from-source-art")
        cnr = report.get("control_not_recovered")
        if cnr or any("CONTROL-NOT-RECOVERED" in w for w in soft):
            _add("stun-timer-before-early-return")
        # run_18: new soft_warning → class craft pins
        if any("EMPTY-3D-VIEW" in w for w in soft):
            _add("voxel-mesh-simple-or-groups")
        if any("DIM-VECTOR-SCENE" in w for w in soft):
            _add("vector-stroke-contrast", "projection-3d-wireframe")
        if any("OBSTACLE-DEPTH-STALL" in w for w in soft):
            _add("trench-depth-vector-spawn")
        if any("OPAQUE-SPRITE-SCENERY" in w for w in soft):
            _add("character-sprite-isolation")
        # run_18: frozen-canvas must MERGE raf pins with class ensure_ids —
        # never replace wireframe/FPS/voxel craft (Battlezone trace).
        if report.get("frozen_canvas") or any("FROZEN-CANVAS" in w for w in soft):
            _add("raf-must-start", "ambient-idle-pixel-delta", "frame-trycatch")
            goal = (
                getattr(self, "_current_goal", None)
                or getattr(self, "goal", None)
                or ""
            )
            class_ids = self._first_build_playbook_ensure_ids(str(goal)) or []
            _add(*class_ids)

        for probe_name, bids in self._REPORT_PROBE_PLAYBOOK_PINS.items():
            if probe_name in failing:
                _add(*bids)

        # Visual-playtest auto-probe recipe refs (mechanism-general).
        if failing and any(n.startswith("auto_") for n in failing):
            recipe_id = getattr(self, "_active_visual_playtest_recipe_id", None)
            if recipe_id:
                try:
                    for item in self._memory.load_visual_playtests():
                        if item.id != recipe_id:
                            continue
                        refs = (
                            (getattr(item, "recipe", None) or {}).get("playbook_refs")
                            or {}
                        )
                        if isinstance(refs, dict):
                            for val in refs.values():
                                _add(*(val or []))
                        break
                except Exception:
                    pass

        return out

    def _retrieve_playbook_block(
        self,
        goal: str,
        *,
        code: str = "",
        stage: str = "code",
        ensure_ids: list[str] | None = None,
    ) -> str:
        """Get top-K bullets and render them as a `<playbook>` block.

        `stage` selects OpenCoder-style two-stage retrieval:
          - "plan" (Stage-1, broad): top_k+bonus bullets, all positive
            relevance hits including net-harmful (exposure to history),
            larger char budget.
          - "code" (Stage-2, narrow, default): top-3 only, drops bullets
            with score ≤ -2 (validated-only patterns), smaller budget.

        After retrieval, `render_playbook_block` runs shingle dedup
        (OpenCoder #5) and budget capping (OpenCoder #2) before emitting
        the prompt block.

        Empty string when nothing matches OR when the active prompt module
        has set PLAYBOOK_DISABLED = True (gives a v0-prompt the option to
        opt out wholesale). Logs retrieved bullet IDs + stage to the trace
        for postmortem inspection of which bullets fired.
        """
        if self._playbook_top_k <= 0:
            return ""
        if getattr(self._p, "PLAYBOOK_DISABLED", False):
            return ""
        try:
            # Fix B feedback — resolve early so code-stage k can widen when
            # user feedback is present (small models otherwise surface k=1
            # only and miss scope/navigation bullets at rank #2).
            recent_feedback = (
                getattr(self, "_last_drained_feedback", None)
                or getattr(self, "_pending_feedback", None)
                or []
            )
            if not recent_feedback:
                cont = getattr(self, "_continuation_feedback", None) or ""
                if cont:
                    recent_feedback = [cont]
            feedback_text = " ".join(str(f) for f in recent_feedback[-2:])
            if stage == "plan":
                k = self._playbook_top_k + self._PLAN_STAGE_TOP_K_BONUS
                if self._model_class in ("mid", "small"):
                    k = 2
                budget = self._PLAN_STAGE_CHAR_BUDGET
                if self._model_class in ("mid", "small"):
                    budget = self._CODE_STAGE_CHAR_BUDGET
                render_mode = "hybrid"
                full_top_n_val = 1 if self._model_class in ("mid", "small") else 3
            else:
                k = min(self._playbook_top_k, self._CODE_STAGE_TOP_K)
                if self._model_class in ("mid", "small"):
                    k = 2 if feedback_text else 1
                budget = self._CODE_STAGE_CHAR_BUDGET
                render_mode = "full"
                full_top_n_val = 3
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_board_intent, _detect_dom_intent, _detect_3d_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            if feedback_text and self._model_class in ("mid", "small"):
                feedback_weighted = " ".join([feedback_text] * 3)
            else:
                feedback_weighted = feedback_text
            query = goal if not feedback_text else f"{goal} {feedback_weighted}"
            hits = self._playbook.retrieve(
                query, code=code, k=k, stage=stage,
                modality_tokens=mod_toks,
            )
            suppressed = self._playbook_suppressed_bullet_ids(
                goal=goal,
                active_skeleton=getattr(self, "_active_skeleton", None),
                code=code or getattr(self, "_current_file", "") or "",
            )
            if suppressed and hits:
                hits = [h for h in hits if h.bullet.id not in suppressed]
            # Cross-modality suppression can drop the only k=1 hit (small
            # models) — widen retrieval and take the next in-modality bullets.
            if suppressed and not hits:
                wide_k = max(k * 6, 8)
                hits = self._playbook.retrieve(
                    query, code=code, k=wide_k, stage=stage,
                    modality_tokens=mod_toks,
                )
                hits = [h for h in hits if h.bullet.id not in suppressed][:k]
            if ensure_ids:
                by_id = {h.bullet.id: h for h in hits}
                pinned: list[BulletHit] = []
                for bid in ensure_ids:
                    if bid in by_id:
                        pinned.append(by_id.pop(bid))
                    else:
                        b = lookup_bullet(self._playbook, bid)
                        if b is not None:
                            pinned.append(BulletHit(b, 1.0))
                if pinned:
                    rest = [h for h in hits if h.bullet.id in by_id]
                    hits = pinned + rest
                    hits = hits[: max(k, len(pinned))]
            if hits:
                ids = [h.bullet.id for h in hits]
                self._active_bullet_ids = list(ids)
            else:
                # No hits this turn — clear the active set so a LATER fix
                # turn that injects nothing cannot penalize an EARLIER
                # plan-stage batch (stale-attribution bug, trace
                # 20260613_213711: the QTE bullets were retrieved only at
                # plan stage but kept getting harmful++ on fix turns that
                # injected no playbook at all).
                self._active_bullet_ids = []
            block = render_playbook_block(
                hits, char_budget=budget, mode=render_mode, full_top_n=full_top_n_val,
            )
            if hits:
                self._trace({
                    "kind": "playbook_retrieved",
                    "stage": stage,
                    "ids": [h.bullet.id for h in hits],
                    "scores": [round(h.score, 4) for h in hits],
                    "sources": [h.bullet.source for h in hits],
                    "selected_chars": sum(len(h.bullet.content or "") for h in hits),
                    "rendered_chars": len(block or ""),
                    "feedback_in_query": bool(feedback_text),
                    "char_budget": budget,
                    "render_mode": render_mode,
                })
            # Injection observability: distinguishes "retrieved but rendered
            # empty" from "actually placed in the prompt". In the 2026-05-29
            # fighting trace it was impossible to tell whether the animation
            # bullets ever reached the model — this makes it one grep.
            self._trace({
                "kind": "playbook_injected",
                "stage": stage,
                "ids": [h.bullet.id for h in hits] if hits else [],
                "chars": len(block or ""),
                "rendered": bool(block),
            })
            return block
        except Exception:
            return ""

    # Combined char ceiling for the three first-build memory blocks in lean
    # mode. opening-book outline (~1.7K) + components (~2.2K) fit; the
    # lower-priority playbook is dropped when it would push past this. Keeps
    # a local model from reading 8KB of overlapping "past lessons" before
    # it writes a line.
    _LEAN_MEMORY_COMBINED_BUDGET = 4500

    # Below this retrieved-outline score a fresh goal is treated as
    # OPEN-DOMAIN (no strong opening-library match). There is no goal->
    # prompt_library fuzzy matcher and a genre/title list is forbidden, so
    # the outline score is the genre-free proxy. Open-domain first builds
    # get one extra component (k=4) and have the game-loop+input snippets
    # PROTECTED from the lean budget — battlezone 20260622 dropped ALL
    # components+playbook on its first build and had to invent the loop.
    _OPEN_DOMAIN_OUTLINE_FLOOR = 0.5

    def _retrieve_opening_book_block(
        self,
        goal: str,
        *,
        stage: str = "plan",
        char_budget: int | None = None,
        deep: bool | None = None,
        extra_tokens: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve compact root/live opening-book recipes with hard caps.

        `extra_tokens` are appended to the modality tokens used for
        retrieval — used by the seed path to pass structural tokens pulled
        from the working file (DOM ids, function names) so the goal alone
        ("add a button") doesn't mis-rank the outline.
        """
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            if extra_tokens:
                mod_toks = mod_toks + list(extra_tokens)
            outline = self._memory.retrieve_implementation_outline(goal, mod_toks)
            # Universal fallback: a novel/open-domain goal may match NO outline
            # (Jaccard empty and no recipe route). Rather than plan with no
            # state/order/traps contract at all, force the genre-free
            # controllable-canvas-game outline so every build still inherits a
            # state-on-window / input / dt-cap / draw-order / restart skeleton.
            outline_fallback = False
            if outline is None:
                outline = self._memory._outline_item_by_id(
                    "outline-controllable-canvas-game"
                )
                outline_fallback = outline is not None
            playtests = self._memory.retrieve_playtests(
                goal, mod_toks, k=3 if stage == "plan" else 1,
            )
            asset_audits = self._memory.retrieve_asset_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            animation_audits = self._memory.retrieve_animation_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            vlm_checklist: str | None = None
            if stage == "plan":
                try:
                    vp_recipe, vp_diag = self._memory.find_visual_playtest_for(
                        goal=goal or "",
                        plan_text=self._criteria or "",
                        asset_names=list(self._session_assets.keys()),
                        code=getattr(self, "_current_file", "") or "",
                    )
                except Exception:
                    vp_recipe, vp_diag = None, {}
                if (
                    vp_recipe is not None
                    and vp_recipe.id not in VLM_CHECKLIST_SKIP_IDS
                ):
                    vlm_checklist = render_vlm_checklist_section(vp_recipe, goal=goal or "")
                    if vlm_checklist:
                        self._trace({
                            "kind": "vlm_checklist_injected",
                            "recipe_id": vp_recipe.id,
                            "top_candidates": (vp_diag or {}).get("top_candidates"),
                        })
            # Plan stage deep-renders the ONE matched outline's recipe
            # (state/order/traps/tuning/probes) under a 3600-char cap —
            # ~+600 tokens in the smallest prompt of the session. Code
            # stage stays shallow at 1400 so iterate prompts never grow.
            block = render_opening_book_block(
                outline, playtests, asset_audits, animation_audits,
                char_budget=char_budget if char_budget is not None else (3600 if stage == "plan" else 1400),
                deep=deep if deep is not None else (stage == "plan"),
                vlm_checklist=vlm_checklist,
            )

            def _row(kind: str, hit: OpeningBookHit) -> dict:
                return {
                    "kind": kind,
                    "id": hit.item.id,
                    "tier": hit.item.source_tier,
                    "score": round(hit.score, 4),
                    "recipe": hit.item.recipe,
                }

            hits: list[dict] = []
            if outline:
                hits.append(_row("outline", outline))
            hits.extend(_row("playtest", h) for h in playtests)
            hits.extend(_row("asset_audit", h) for h in asset_audits)
            hits.extend(_row("animation_audit", h) for h in animation_audits)
            if hits:
                # Runtime callers need the full recipe dictionaries for browser
                # playtests. Persist only this reconstructible attribution copy.
                trace_hits = [
                    {key: value for key, value in hit.items() if key != "recipe"}
                    for hit in hits
                ]
                self._trace({
                    "kind": "opening_book_retrieved",
                    "stage": stage,
                    "hits": trace_hits,
                    "modality_tokens": mod_toks,
                    "outline_fallback": outline_fallback,
                    "selected_chars": sum(
                        len(json.dumps(
                            hit.get("recipe") or {},
                            ensure_ascii=False,
                            default=str,
                            separators=(",", ":"),
                        ))
                        for hit in hits
                    ),
                    "rendered_chars": len(block or ""),
                })
            return block, hits
        except Exception as e:
            self._trace({"kind": "opening_book_error", "stage": stage, "err": str(e)})
            return "", []

    _PHYSICS_TRAP_SIGNALS = (
        "control-not-recovered",
        "player-stuck",
        "collision",
        "bounce",
        "tunnel",
        "plunger",
        "bumper",
        "flipper",
        "velocity",
        "physics",
        "frozen canvas",
        "ping-pong",
        "launch",
    )

    def _physics_traps_needed(
        self, report_text: str, failure_class: str | None = None,
    ) -> bool:
        if failure_class == "memory_gap":
            return True
        low = (report_text or "").lower()
        return any(sig in low for sig in self._PHYSICS_TRAP_SIGNALS)

    def _retrieve_outline_traps_block(
        self,
        goal: str,
        report_text: str,
        *,
        failure_class: str | None = None,
        char_budget: int = 400,
    ) -> str:
        """Inject outline traps/tuning on fix turns when physics/stuck."""
        if not self._physics_traps_needed(report_text, failure_class):
            return ""
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            outline = self._memory.retrieve_implementation_outline(goal, mod_toks)
            if outline is None:
                return ""
            recipe = getattr(outline.item, "recipe", None)
            block = render_outline_traps_only(
                recipe if isinstance(recipe, dict) else {},
                char_budget=char_budget,
            )
            if block:
                self._trace({
                    "kind": "outline_traps_injected",
                    "outline_id": outline.item.id,
                    "chars": len(block),
                })
            return block
        except Exception as e:
            self._trace({"kind": "outline_traps_error", "err": str(e)})
            return ""

    def _retrieve_components_block(
        self,
        query: str,
        *,
        stage: str = "plan",
        k: int = 3,
        ensure_ids: list[str] | None = None,
    ) -> str:
        """Retrieve component-library snippets as a <components> block.

        Capability-round item 1: tested, mechanics-level JS the model
        pastes and ADAPTS (memory/components.jsonl). `query` is the goal
        at first build; at fix turns it is the blocker text so a snippet
        is only injected when it matches the actual failure. Returns ""
        when nothing matches (safe degradation).
        """
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(query)
                    + _detect_board_intent(query)
                    + _detect_dom_intent(query)
                )
            except Exception:
                mod_toks = []
            hits = self._memory.retrieve_components(query, mod_toks, k=k)
            # Universal-fallback pinning: guarantee the engine-skeleton snippets
            # (game loop, input) are present AND FIRST for open-domain goals so
            # they survive the lean budget. Jaccard can either miss a pinned id
            # entirely OR retrieve it but rank it below the char cap (the latter
            # silently dropped input-manager-buffered for novel goals); both
            # cases are handled by moving every pinned hit to the protected
            # front and fetching any that were not retrieved at all.
            if ensure_ids:
                by_id = {h.item.id: h for h in hits}
                pinned: list[OpeningBookHit] = []
                for cid in ensure_ids:
                    if cid in by_id:
                        pinned.append(by_id.pop(cid))
                    else:
                        pinned.extend(self._memory.components_by_ids([cid]))
                if pinned:
                    rest = [h for h in hits if h.item.id in by_id]
                    hits = pinned + rest
            block = render_components_block(
                hits, char_budget=2200 if stage == "plan" else 1400,
            )
            if hits:
                self._trace({
                    "kind": "components_injected",
                    "stage": stage,
                    "ids": [h.item.id for h in hits],
                    "scores": [round(h.score, 4) for h in hits],
                    "tiers": [h.item.source_tier for h in hits],
                    "selected_chars": sum(
                        len(h.item.content or "")
                        + len(json.dumps(
                            h.item.recipe or {},
                            ensure_ascii=False,
                            default=str,
                            separators=(",", ":"),
                        ))
                        for h in hits
                    ),
                    "chars": len(block or ""),
                    "rendered_chars": len(block or ""),
                    "rendered": bool(block),
                })
            return block or ""
        except Exception as e:
            self._trace({"kind": "components_error", "stage": stage, "err": str(e)})
            return ""

    def _apply_lean_memory_budget(
        self, opening_block: str, components_block: str, playbook_block: str,
        *, protect_components: bool = False, protect_playbook: bool = False,
    ) -> tuple[str, str, str]:
        """In lean mode, cap the COMBINED size of the first-build memory
        blocks. Priority: opening-book outline > components > playbook —
        whole blocks are dropped (never mangled) lowest-priority first once
        the budget is exceeded. No-op outside lean mode.

        `protect_components` (seed continuations) keeps the components block
        even when opening already filled the budget — on a seed iter 1 the
        components are the copy-paste-correct snippets (help overlay, media
        loader) the weak model most needs, so the playbook is trimmed first
        instead. (2026-06-21 seed trace dropped `help-overlay-modal` here.)

        `protect_playbook` (Phase-A assets on disk) keeps the playbook block
        when ensure_ids pinned draw-generated-sprites-not-boxes — run_10
        traces retrieved that bullet then lean budget dropped it before iter 1."""
        if not self._lean_prompt_active():
            return opening_block, components_block, playbook_block
        budget = self._LEAN_MEMORY_COMBINED_BUDGET
        used = len(opening_block or "")  # opening (priority 1) always kept
        blocked = False

        def _fit(block: str) -> str:
            nonlocal used, blocked
            if not block:
                return ""
            if blocked or used + len(block) > budget:
                blocked = True
                return ""
            used += len(block)
            return block

        if protect_components and components_block:
            # Keep components unconditionally (count it against the budget so
            # playbook still yields), then fit playbook in what remains.
            cb = components_block
            used += len(components_block)
        else:
            cb = _fit(components_block)

        if protect_playbook and playbook_block:
            pb = playbook_block
            used += len(playbook_block)
        else:
            pb = _fit(playbook_block)
        if (cb != components_block) or (pb != playbook_block):
            self._trace({
                "kind": "lean_memory_budget_applied",
                "budget": budget,
                "kept_opening_chars": len(opening_block or ""),
                "kept_components_chars": len(cb),
                "kept_playbook_chars": len(pb),
                "dropped_components": bool(components_block) and not cb,
                "dropped_playbook": bool(playbook_block) and not pb,
            })
        return opening_block, cb, pb

    def _detect_open_domain_build(
        self, opening_hits: list[dict]
    ) -> tuple[bool, list[dict]]:
        """Genre-free open-domain detection for a fresh first build.

        A fresh goal is "novel" when it matches NO outline, falls back to the
        universal controllable-canvas-game outline, or only weakly matches a
        specific one. We can't proxy via prompt_library (no goal->library
        matcher; genre/title lists are forbidden), and the recipe-routed
        outline score is usually 1.0, so we key on the outline ID + score
        together. Returns (open_domain_build, outline_rows). Extracted
        verbatim from run() so it can be unit-tested in isolation.
        """
        outline_rows = [h for h in opening_hits if h.get("kind") == "outline"]
        if not outline_rows:
            return True, outline_rows
        o = outline_rows[0]
        open_domain = (
            o.get("id") == "outline-controllable-canvas-game"
            or o.get("score", 1.0) < self._OPEN_DOMAIN_OUTLINE_FLOOR
        )
        return open_domain, outline_rows
