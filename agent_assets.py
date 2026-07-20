"""Mid-session asset/sound generation and alignment scans.

Moved VERBATIM from `GameAgent` (no behavior change).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from assets import (
    _DERIVED_FRAME_MIN_DELTA,
    _default_size_for_prompt,
    _parse_size,
    diffuser_display_label,
    generate_assets,
    parse_assets_block,
    parse_assets_block_with_meta,
    prefer_video_seed_assets,
    render_asset_paths_block,
    try_load_image_generator,
)
from sounds import (
    generate_sounds,
    parse_sounds_block,
    render_sound_paths_block,
    try_load_audio_generator,
)
from videos import (
    generate_videos,
    parse_videos_block,
    render_video_paths_block,
    try_load_video_generator,
)
from agent_feedback import _HARNESS_ADVISORY_SENTINEL, _feedback_requests_style_rebrand
from agent_helpers import (
    _ASSETS_OPEN_RE,
    _SOUNDS_OPEN_RE,
    _declared_seed_media_names,
    _declared_stems_complete,
    _missing_declared_stems,
    _scan_seed_media,
)
from agent import AgentEvent


class AssetGenerationMixin:

    """Mid-session asset/sound generation and alignment scans."""

    # Pattern catches:

    #   ASSETS['name']  /  ASSETS["name"]   (subscript)

    #   ASSETS.name                          (dot access)

    # Captures the identifier as group 1.

    _ASSET_SUBSCRIPT_RE = __import__("re").compile(

        r"""ASSETS\s*\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"""

    )

    _ASSET_DOT_RE = __import__("re").compile(

        r"""\bASSETS\.([A-Za-z_][A-Za-z0-9_]*)\b"""

    )

    # Loose path pattern: any '<anything>_assets/<name>.png' string

    # literal. Catches inlined paths even without ASSETS[] indirection.

    _ASSET_PATH_RE = __import__("re").compile(

        r"""['"][^'"\s]*_assets/([A-Za-z_][A-Za-z0-9_]*)\.png['"]"""

    )

    # Array-of-string-names commonly named `assetList`, `asset_names`,

    # `assetPaths`, `sprites`, etc., followed by a `[...]` literal of

    # bare string identifiers. The DK trace failure used exactly this

    # pattern — assetList = ['mario_idle', …] then mapped at runtime.

    _ASSET_LIST_RE = __import__("re").compile(

        r"""\b(?:assetList|asset_names|assetNames|spriteNames|spriteList)\s*=\s*\[([^\]]+)\]""",

        __import__("re").IGNORECASE,

    )

    _ASSET_LIST_NAME_RE = __import__("re").compile(

        r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""

    )

    # Sound alignment scan mirrors the asset scan above so missing OGG

    # references are surfaced before Chromium wastes an iteration on 404s.

    _SOUND_SUBSCRIPT_RE = __import__("re").compile(

        r"""SOUNDS\s*\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"""

    )

    _SOUND_DOT_RE = __import__("re").compile(

        r"""\bSOUNDS\.([A-Za-z_][A-Za-z0-9_]*)\b"""

    )

    _SOUND_PATH_RE = __import__("re").compile(

        r"""['"][^'"\s]*_sounds/([A-Za-z_][A-Za-z0-9_]*)\.(?:ogg|mp3|wav|m4a)['"]""",

        __import__("re").IGNORECASE,

    )

    _SOUND_LIST_RE = __import__("re").compile(

        r"""\b(?:soundNames|soundList|sfxNames|audioNames)\s*=\s*\[([^\]]+)\]""",

        __import__("re").IGNORECASE,

    )

    _SOUND_LIST_NAME_RE = __import__("re").compile(

        r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""

    )



    # Generic DOM/structural attributes whose values are pure layout noise,

    # not feature signals — dropped from seed structural tokens.

    @classmethod
    def _scan_html_for_asset_refs(cls, html: str) -> set[str]:

        """Return the set of asset *names* the HTML references.



        Static analysis — fast, deterministic, no JS execution. Covers

        the three patterns the model produces in practice:

          1. ASSETS['name'] / ASSETS["name"] / ASSETS.name

          2. Literal paths '<dir>_assets/<name>.png'

          3. Array literals assigned to assetList / spriteNames / etc.

        """

        refs: set[str] = set()

        for m in cls._ASSET_SUBSCRIPT_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._ASSET_DOT_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._ASSET_PATH_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._ASSET_LIST_RE.finditer(html):

            for nm in cls._ASSET_LIST_NAME_RE.finditer(m.group(1)):

                refs.add(nm.group(1))

        return refs



    def _check_asset_alignment(self, html: str) -> set[str]:

        """Compare HTML asset references against generated files.



        Returns the set of names that are REFERENCED but not present in

        `self._session_assets`. When non-empty, queues a coaching

        message naming the missing files so the next user turn tells

        the model to either remove the references or emit an `<assets>`

        block to request the gap (which the existing mid-session regen

        pipeline will then fulfill).

        """

        refs = self._scan_html_for_asset_refs(html)

        if not refs:

            return set()

        available = set(self._session_assets.keys())

        missing = refs - available

        if not missing:

            return set()

        self._trace({

            "kind": "asset_alignment_gap",

            "referenced": sorted(refs),

            "available": sorted(available),

            "missing": sorted(missing),

        })

        miss_list = ", ".join(sorted(missing))

        avail_list = ", ".join(sorted(available)) or "(none)"

        self._pending_coaching.append(

            "Asset references don't match the files on disk. Your code "

            f"references these names that were never generated: {miss_list}. "

            f"Available assets: {avail_list}. The browser is returning "

            "net::ERR_FILE_NOT_FOUND for the missing ones, not a drawImage "

            "bug. To fix the root cause, EITHER (a) emit an `<assets>...</assets>` "

            "block in this turn requesting the missing names — the harness "

            "will regenerate them mid-session and your code will work as "

            "written — OR (b) edit the code to only reference assets that "

            "exist. Do NOT add more drawImage try/catch guards — those "

            "hide the load failure, they don't fix it."

        )

        return missing



    def _early_rehydrate_seed_media(self) -> tuple[int, int]:

        """Populate `_session_assets` / `_session_sounds` from the seed

        BEFORE Phase A asset generation runs.



        P1 (MK trace 20260528): the previous run loop rehydrated AFTER

        `_maybe_generate_assets_and_sounds(trigger="phase_a")` had

        already regenerated every sprite the model re-requested in its

        plan. By rehydrating first, the skip-guard inside

        `_maybe_generate_assets_and_sounds` can short-circuit phase_a

        generation when on-disk media already covers the game.



        Idempotent: a second call from the existing later branch is a

        no-op (dict.update with the same paths). Returns (n_assets,

        n_sounds) found on disk, or (0, 0) when there is no seed.

        """

        if not self.seed_file:

            return 0, 0

        try:

            seed_html = self.seed_file.read_text(encoding="utf-8")

        except Exception as e:

            self._trace({

                "kind": "seed_media_early_rehydrate_failed",

                "err": str(e)[:200],

                "seed_file": str(self.seed_file),

            })

            return 0, 0

        # Declared PATHS/sound stems even when PNGs/OGGs are missing —
        # assets-only seed regen must not invent a new genre roster.
        try:
            declared_a, declared_s = _declared_seed_media_names(seed_html)
            self._seed_declared_asset_names = list(declared_a)
            self._seed_declared_sound_names = list(declared_s)
        except Exception:
            self._seed_declared_asset_names = []
            self._seed_declared_sound_names = []

        try:

            seed_assets, seed_sounds, _, _ = _scan_seed_media(

                seed_html, self.out_path

            )

        except Exception as e:

            self._trace({

                "kind": "seed_media_early_rehydrate_failed",

                "err": str(e)[:200],

                "stage": "_scan_seed_media",

            })

            return 0, 0

        if seed_assets:

            self._session_assets.update(seed_assets)

        if seed_sounds:

            self._session_sounds.update(seed_sounds)

        self._trace({

            "kind": "seed_media_early_rehydrate",

            "assets": len(seed_assets),

            "sounds": len(seed_sounds),

            "declared_assets": len(
                getattr(self, "_seed_declared_asset_names", None) or []
            ),

            "declared_sounds": len(
                getattr(self, "_seed_declared_sound_names", None) or []
            ),

            "asset_names": sorted(seed_assets.keys())[:24],

            "sound_names": sorted(seed_sounds.keys())[:24],

        })

        return len(seed_assets), len(seed_sounds)



    def _render_seed_media_contract(

        self,

        html: str,

        *,

        asset_names: list[str] | None = None,

        sound_names: list[str] | None = None,

    ) -> str:

        """Compact contract for seed runs: existing media are first-class.



        Seeded edits should wire or reuse media already on disk before asking

        for new generations. The contract lists referenced vs available names

        and highlights available-but-unused names that often solve "missing

        animation" requests by adding loader entries, not regenerating art.

        """

        assets = sorted(asset_names if asset_names is not None else self._session_assets.keys())

        sounds = sorted(sound_names if sound_names is not None else self._session_sounds.keys())

        refs_a = sorted(self._scan_html_for_asset_refs(html or ""))

        refs_s = sorted(self._scan_html_for_sound_refs(html or ""))

        unused_a = sorted(set(assets) - set(refs_a))

        unused_s = sorted(set(sounds) - set(refs_s))



        def _fmt(names: list[str], cap: int = 28) -> str:

            if not names:

                return "(none)"

            shown = names[:cap]

            more = f" (+{len(names) - cap} more)" if len(names) > cap else ""

            return ", ".join(shown) + more



        lines = [

            "================ SEED MEDIA CONTRACT ================",

            "This is an EXISTING seeded game. Treat assets/sounds already",

            "on disk as the current game's media API. Use/wire existing",

            "names before asking for new art or sound.",

            "",

            f"Available assets: {_fmt(assets)}",

            f"Referenced assets in HTML: {_fmt(refs_a)}",

            f"Existing assets not currently referenced/loaded: {_fmt(unused_a)}",

        ]

        if sounds:

            lines.extend([

                "",

                f"Available sounds: {_fmt(sounds)}",

                f"Referenced sounds in HTML: {_fmt(refs_s)}",

                f"Existing sounds not currently referenced/loaded: {_fmt(unused_s)}",

            ])

        lines.extend([

            "",

            "Rules for seed edits:",

            "  - If the user says missing / use existing / don't redo, patch",

            "    loader or draw mapping to use existing names.",

            "  - Emit <assets>/<sounds> only when the user explicitly asks",

            "    for new art/sound or no existing name can satisfy the request.",

            "  - Do not design a new game; adapt this seed.",

            "=====================================================",

        ])

        return "\n".join(lines)



    @classmethod

    def _scan_html_for_sound_refs(cls, html: str) -> set[str]:

        """Return the set of sound *names* the HTML references."""

        refs: set[str] = set()

        for m in cls._SOUND_SUBSCRIPT_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._SOUND_DOT_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._SOUND_PATH_RE.finditer(html):

            refs.add(m.group(1))

        for m in cls._SOUND_LIST_RE.finditer(html):

            for nm in cls._SOUND_LIST_NAME_RE.finditer(m.group(1)):

                refs.add(nm.group(1))

        return refs



    def _check_sound_alignment(self, html: str) -> set[str]:

        """Compare HTML sound references against generated files."""

        refs = self._scan_html_for_sound_refs(html)

        if not refs:

            return set()

        available = set(self._session_sounds.keys())

        missing = refs - available

        if not missing:

            return set()

        self._trace({

            "kind": "sound_alignment_gap",

            "referenced": sorted(refs),

            "available": sorted(available),

            "missing": sorted(missing),

        })

        miss_list = ", ".join(sorted(missing))

        avail_list = ", ".join(sorted(available)) or "(none)"

        self._pending_coaching.append(

            "Sound references don't match the files on disk. Your code "

            f"references these sound names that were never generated: {miss_list}. "

            f"Available sounds: {avail_list}. The browser is returning "

            "net::ERR_FILE_NOT_FOUND for missing OGGs, not an Audio.play() "

            "bug. To fix the root cause, EITHER (a) emit a `<sounds>...</sounds>` "

            "block in this turn requesting the missing names — the harness "

            "will regenerate them mid-session and your code will work as "

            "written — OR (b) edit the code to only reference sounds that "

            "exist. Do NOT add more try/catch around play(); that hides the "

            "load failure, it doesn't fix it."

        )

        return missing



    def _maybe_prewarm_diffusers_during_phase_a(self) -> None:

        """Background-load Z-Image-Turbo + Stable Audio Open during the

        architect's streaming window — but ONLY when the diffuser GPU

        is independent of any LLM slot.



        On the 4-GPU workstation the diffusers run on GPU 0 and the

        LLM on GPUs 1-3, so the warmup is pure speedup (~30-60 s

        hidden). On a single-GPU box the diffuser and the architect

        share VRAM, so pre-warming would steal VRAM from the in-flight

        plan stream — skip in that case.



        Fires-and-forgets: each pipeline's `ensure_loaded()` runs in

        `asyncio.to_thread`; if loading raises, we trace + move on.

        The user-facing wins are captured by `prefill_warm` trace

        events tagged with `target: "z_image" | "stable_audio"`.

        """

        try:

            import gpu_status as _gs

            snap = _gs.snapshot_gpus()

            if not _gs.diffuser_has_dedicated_gpu(snap):

                self._trace({

                    "kind": "diffuser_prewarm_skipped",

                    "reason": "no_dedicated_gpu",

                    "n_gpus": len(snap.gpus) if snap and snap.gpus else 0,

                })

                return

        except Exception as e:

            self._trace({

                "kind": "diffuser_prewarm_skipped",

                "reason": "gpu_probe_error",

                "err": str(e)[:200],

            })

            return



        async def _prewarm_image() -> None:

            import time as _t

            t0 = _t.monotonic()

            try:

                import assets as _assets

                gen = self._asset_generator or _assets.ZImageTurboGenerator()

                self._asset_generator = gen

                ok = await asyncio.to_thread(gen._lazy_init)

                self._trace({

                    "kind": "prefill_warm",

                    "target": "z_image",

                    "elapsed_s": round(_t.monotonic() - t0, 2),

                    "ok": bool(ok),

                    "hidden_under_phase_a": True,

                })

            except Exception as e:

                self._trace({

                    "kind": "prefill_warm",

                    "target": "z_image",

                    "elapsed_s": round(_t.monotonic() - t0, 2),

                    "ok": False,

                    "err": str(e)[:200],

                })



        async def _prewarm_audio() -> None:

            import time as _t

            t0 = _t.monotonic()

            try:

                import sounds as _sounds

                gen = self._sound_generator or _sounds.StableAudioGenerator()

                self._sound_generator = gen

                ok = await asyncio.to_thread(gen._lazy_init)

                self._trace({

                    "kind": "prefill_warm",

                    "target": "stable_audio",

                    "elapsed_s": round(_t.monotonic() - t0, 2),

                    "ok": bool(ok),

                    "hidden_under_phase_a": True,

                })

            except Exception as e:

                self._trace({

                    "kind": "prefill_warm",

                    "target": "stable_audio",

                    "elapsed_s": round(_t.monotonic() - t0, 2),

                    "ok": False,

                    "err": str(e)[:200],

                })



        # Fire-and-forget. We don't await — these run concurrently with

        # the architect's streaming. The tasks complete on their own

        # schedule; the next `<assets>` / `<sounds>` block uses the

        # already-loaded pipeline.

        try:

            asyncio.create_task(_prewarm_image())

            asyncio.create_task(_prewarm_audio())

        except Exception:

            pass



    # ---- Phase 1A — non-blocking visual critic --------------------------

    #

    # When the critic role is bound to a SEPARATE Ollama slot from the

    # coder (the normal multi-GPU configuration), running the critic

    # serially after each iter wastes 20-300 s of wall-clock — the

    # critic compute could overlap with the next iter's coder stream.

    # Spawning it as `asyncio.Task` lets that overlap happen and lets

    # the coaching land in `_pending_coaching` whenever the task

    # finishes (one-turn lag at worst). When the critic backend is the

    # SAME as the coder backend (single-slot / single-GPU fallback),

    # concurrent runs just queue at the daemon — no benefit, so we

    # fall back to the existing blocking await.



    @staticmethod

    def _mark_unused_media_as_stale_for_continuation(mp: dict) -> int:

        """Rewrite unused-media warnings for full continuation rewrites.



        When a full rewrite changes the runtime shape, generated media

        from the previous build may simply be stale. In that context,

        "wire it in" is the wrong instruction; the model should use or

        request media that fits the current request.



        Returns the number of old unused-media warnings suppressed.

        """

        if not (mp.get("stats") or {}).get("unused_assets"):

            return 0

        old_warnings = list(mp.get("warnings") or [])

        kept = [

            w for w in old_warnings

            if "NEVER referenced in the HTML" not in str(w)

        ]

        suppressed = len(old_warnings) - len(kept)

        if suppressed:

            kept.append(

                "Generated media from the previous build appears stale "

                "after this full rewrite. Ignore old asset/sound names "

                "unless they still fit the current request; request or "

                "wire media for the current game shape instead."

            )

            mp["warnings"] = kept

        return suppressed



    # Threshold above which we consider the current file too large to

    # inject in full on every fix turn. Below this, full-file inject is

    # cheap and removes any risk of the slice missing context.

    _FULL_FILE_INJECT_LIMIT = 12_000



    # Seed-build injection budget (separate from the fix-turn limit above).

    # The first seed turn is the ONLY chance the model gets to see the file

    # it must patch; truncating it (2026-06-25 trace 161300_697573: a 34KB

    # seed cut to 12KB) elides the patch target's SEARCH anchor, so

    # seed_build_instruction's "if the target isn't shown, send a full

    # <html_file>" fallback fires and a small model loops for ~28 min on a

    # 711-line rewrite. A 34KB seed adds only ~9-10K prompt tokens (iter-1

    # prompt was ~14K; ctx default is 100K), so inject typical single-file

    # games in FULL and keep the head+tail excerpt ONLY for the rare

    # genuinely-oversized seed.

    _SEED_FULL_FILE_INJECT_LIMIT = 60_000



    @staticmethod
    def _prompt_for_declared_asset_stem(name: str) -> str:
        """Minimal genre-free prompt when the model omitted a declared stem."""
        readable = (name or "sprite").replace("_", " ").strip() or "sprite"
        return (
            f"game sprite for {readable}, transparent background, no text"
        )

    @staticmethod
    def _asset_spec_with_size(spec: dict[str, Any]) -> dict[str, Any]:
        """Ensure a size tuple — generate_assets requires spec['size'].

        Harness-filled declared-roster specs historically lacked size
        (trace 20260720_135103 KeyError), while parse_assets_block always
        injects it. Normalize both paths here.
        """
        if spec.get("size"):
            return spec
        prompt = str(spec.get("prompt") or "")
        out = dict(spec)
        out["size"] = _parse_size(_default_size_for_prompt(prompt))
        return out

    def _coerce_specs_to_declared_seed_roster(
        self,
        asset_specs: list[dict[str, Any]],
        sound_specs: list[dict[str, Any]],
        *,
        trigger: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Seed media: declared PATHS own names on regen / phase_a invent.

        Mid-session additive names (new boss sprite) must still be allowed.
        """
        declared_a = list(
            getattr(self, "_seed_declared_asset_names", None) or []
        )
        declared_s = list(
            getattr(self, "_seed_declared_sound_names", None) or []
        )
        if not self.seed_file or (not declared_a and not declared_s):
            return asset_specs, sound_specs

        seed_regen = bool(
            getattr(self, "_seed_media_regen", False)
            or getattr(self, "_assets_only_goal", False)
        )
        by_a = {
            str(s.get("name") or "").strip(): s
            for s in (asset_specs or [])
            if str(s.get("name") or "").strip()
        }
        by_s = {
            str(s.get("name") or "").strip(): s
            for s in (sound_specs or [])
            if str(s.get("name") or "").strip()
        }

        if not seed_regen:
            # Phase A only: drop invent outside declared. Mid-session may add.
            if trigger != "phase_a" or (not by_a and not by_s):
                return asset_specs, sound_specs
            kept_a = [by_a[n] for n in declared_a if n in by_a]
            kept_s = [by_s[n] for n in declared_s if n in by_s]
            dropped = sorted(
                (set(by_a) - set(declared_a)) | (set(by_s) - set(declared_s))
            )
            if dropped:
                self._trace({
                    "kind": "seed_media_invented_names_dropped",
                    "dropped_invented": dropped[:48],
                })
            return kept_a, kept_s

        regen_all = bool(getattr(self, "_seed_media_regen_all", True))
        if regen_all:
            target_a = list(declared_a)
            target_s = list(declared_s)
        else:
            target_a = _missing_declared_stems(
                declared_a, self._session_assets,
            )
            target_s = _missing_declared_stems(
                declared_s, self._session_sounds,
            )
            if not target_a and not target_s:
                return asset_specs, sound_specs

        new_assets: list[dict[str, Any]] = []
        for name in target_a:
            if name in by_a:
                new_assets.append(self._asset_spec_with_size(by_a[name]))
            else:
                prompt = self._prompt_for_declared_asset_stem(name)
                new_assets.append(self._asset_spec_with_size({
                    "name": name,
                    "prompt": prompt,
                }))
        new_sounds: list[dict[str, Any]] = []
        for name in target_s:
            if name in by_s:
                new_sounds.append(by_s[name])
            else:
                new_sounds.append({
                    "name": name,
                    "prompt": f"short game sound for {name.replace('_', ' ')}",
                })
        dropped = sorted(
            (set(by_a) - set(target_a)) | (set(by_s) - set(target_s))
        )
        if dropped or new_assets != asset_specs or new_sounds != sound_specs:
            self._trace({
                "kind": "seed_media_roster_forced",
                "regen_all": regen_all,
                "target_assets": target_a[:48],
                "target_sounds": target_s[:48],
                "dropped_invented": dropped[:48],
            })
        return new_assets, new_sounds

    @staticmethod

    def _filter_media_specs_to_allowed(

        specs: list[dict[str, Any]],

        allowed_names: set[str],

    ) -> tuple[list[dict[str, Any]], list[str]]:

        if not specs:

            return specs, []

        if not allowed_names:

            dropped = [

                str(spec.get("name") or "").strip()

                for spec in specs

                if str(spec.get("name") or "").strip()

            ]

            return [], dropped

        kept: list[dict[str, Any]] = []

        dropped: list[str] = []

        for spec in specs:

            name = str(spec.get("name") or "").strip()

            if name in allowed_names:

                kept.append(spec)

            elif name:

                dropped.append(name)

        return kept, dropped

    async def _maybe_autogen_pending_dropped_assets(
        self,
    ) -> AsyncIterator[AgentEvent]:
        """Generate harness-dropped sprites without waiting for the model.

        run_vlm10 Dragon's Lair: Phase A overflow dropped 4 PNGs; the model
        never re-emitted <assets> within the 3-iter cap. Autogen here.
        """
        specs = list(getattr(self, "_pending_dropped_asset_specs", None) or [])
        if not specs:
            return
        missing = [
            s for s in specs
            if str(s.get("name") or "").strip()
            and str(s["name"]) not in (getattr(self, "_session_assets", None) or {})
        ]
        if not missing:
            self._pending_dropped_asset_specs = []
            self._pending_dropped_assets = []
            return
        cap = getattr(self, "_session_asset_cap", None)
        batch = missing[: cap if cap else len(missing)]
        yield self._record(AgentEvent(
            "info",
            f"autogen: generating {len(batch)} harness-dropped sprite(s) "
            f"({', '.join(str(s.get('name') or '') for s in batch[:6])})",
        ))
        if self._asset_generator is None:
            self._asset_generator = await asyncio.to_thread(
                try_load_image_generator,
            )
        if self._asset_generator is None:
            yield self._record(AgentEvent(
                "info",
                "autogen skipped: image generator unavailable",
            ))
            return
        session_assets_dir = self.out_path.parent / f"{self._session_id}_assets"
        try:
            produced = await asyncio.to_thread(
                generate_assets,
                batch,
                session_assets_dir,
                image_generator=self._asset_generator,
            )
        except Exception as e:
            yield self._record(AgentEvent(
                "info",
                f"autogen failed: {e!r} — model must re-request assets.",
            ))
            return
        if produced:
            self._session_assets.update(produced)
            produced_names = set(produced.keys())
            self._pending_dropped_assets = [
                n for n in (getattr(self, "_pending_dropped_assets", None) or [])
                if n not in produced_names
            ]
            self._pending_dropped_asset_specs = [
                s for s in specs
                if str(s.get("name") or "") not in produced_names
            ]
            self._trace({
                "kind": "dropped_assets_autogen",
                "requested": len(batch),
                "produced": list(produced.keys()),
            })
            yield self._record(AgentEvent(
                "info",
                f"autogen: generated {len(produced)}/{len(batch)} dropped "
                f"sprites at {session_assets_dir}",
            ))
            block = render_asset_paths_block(produced, self.out_path)
            if block:
                self._queue_internal_feedback(
                    "HARNESS AUTOGEN: sprites dropped by the per-turn cap "
                    "were generated for you:\n"
                    f"{block}\n"
                    "Wire these paths into drawImage/sprite() now."
                )

    async def _maybe_generate_assets_and_sounds(

        self, reply: str, *, trigger: str,

    ) -> AsyncIterator[AgentEvent]:

        """Parse <assets>/<sounds>/<videos> in a model reply and run the

        diffuser / audio / video pipeline for each block present.



        Called from two sites:

        - Phase A plan reply (initial generation).

        - Any Phase B / Phase C reply (mid-session add when the model

          emits a fresh <assets>/<sounds> block in response to user

          feedback like "add proper sprites for the invaders").



        `trigger` is stamped onto the trace events ("phase_a" |

        "mid_session") so offline analysis can distinguish first-load

        from mid-session adds.



        Mid-session semantics: MERGE into self._session_assets /

        self._session_sounds — never overwrite. New assets are

        additive. After generation, push a freshly-rendered asset/sound

        paths block into _pending_feedback so the next user turn shows

        the new file paths to the model (mirrors the Phase A → first

        build prelude path).



        Per-asset success timing is logged here too — one info line per

        sprite when count is small, summary line when >5 to avoid log

        spam. Failure reasons are surfaced one-per-line either way.

        """

        # Phase 0.10 — per-session asset cap. Defaults to module-level

        # _MAX_ASSETS_PER_TURN; raised at session start when the goal

        # text explicitly asks for multi-frame rosters

        # (`prompts_v1._detect_multi_frame_intent`). The raise lets a

        # user-requested N entities × M frames roster land in one turn

        # instead of getting silently truncated.

        session_cap = getattr(self, "_session_asset_cap", None)

        asset_specs, dropped_asset_names, dropped_asset_specs = (
            parse_assets_block_with_meta(
                reply, max_assets=session_cap,
            )
        )

        sound_specs = parse_sounds_block(reply)

        video_specs = parse_videos_block(reply)

        # run_14 Dragon's Lair: FIFO cap dropped key_victory (i2v seed).
        # Prefer keeping video image seeds inside the same cap budget.
        if dropped_asset_names and video_specs:
            asset_specs, dropped_asset_names, dropped_asset_specs = (
                prefer_video_seed_assets(
                    asset_specs,
                    dropped_asset_names,
                    dropped_asset_specs,
                    video_specs,
                )
            )

        # Seed media: force declared PATHS stems when regen / drop phase_a invent.
        asset_specs, sound_specs = self._coerce_specs_to_declared_seed_roster(
            asset_specs, sound_specs, trigger=trigger,
        )

        # Parse-failure coaching (GLM-5.2 trace 20260625_124038): the model

        # emitted an <assets> tag but it parsed to ZERO specs (wrong JSON

        # shape, e.g. a `{"sprites":[...]}` wrapper or malformed body). The

        # old behavior was a SILENT no-op — Z-Image never ran, the model got

        # no signal, and it shipped code referencing PNGs that never existed.

        # Emit a trace + queue a one-turn coaching note naming the exact

        # format so the model retries with a valid bare array. Genre-agnostic.

        if not asset_specs and _ASSETS_OPEN_RE.search(reply or ""):

            # TD seed trace 20260630_114658: phase_a plan had a malformed

            # <assets> tag on a seed continuation. Queueing "NO art was

            # generated / re-emit <assets> this turn" was misleading — the

            # build stream still generated sprites via mid_session minutes

            # later. Phase A is not the asset turn on seeded runs.

            _seed_phase_a_malformed = (

                trigger == "phase_a"

                and self.seed_file is not None

                and (self._session_assets or self._session_sounds)

            )

            if _seed_phase_a_malformed:

                self._trace({

                    "kind": "assets_parse_failed_seed_ignored",

                    "trigger": trigger,

                    "reason": "seed_phase_a_malformed_assets_tag",

                })

            else:

                self._trace({

                    "kind": "assets_parse_failed",

                    "trigger": trigger,

                    "reason": "assets_tag_present_but_zero_specs",

                })

                self._queue_internal_feedback(

                    "ASSET FORMAT ERROR: your last <assets> block did not parse "

                    "to any sprites, so NO art was generated. The <assets> body "

                    "MUST be a bare JSON array of objects — "

                    '`<assets>[{"name":"hero","prompt":"..."},'

                    '{"name":"enemy","prompt":"..."}]</assets>`. Do NOT wrap it '

                    'in a key like {"sprites":[...]} or {"assets":[...]}, and do '

                    "NOT use a markdown code fence. Re-emit the <assets> block as "

                    "a bare array this turn so the sprites actually generate."

                )

        # The model emitted an <assets> block → any outstanding "generate new

        # art" request is now honored; stop re-asserting it (see the

        # ASSET GENERATION REQUIRED directive in _flush_user_injections).

        if asset_specs and self._unhonored_asset_request is not None:

            self._unhonored_asset_request = None

            self._asset_reprompt_count = 0



        # Router-vs-model overrule (chess-trace fix 2026-06-22): when the

        # LLM router judged the user did NOT want new art this batch

        # (allow_assets_block=false) but the coder emitted an <assets>

        # block anyway on a mid-session turn, the model overruled the

        # router. Record it via the SAME counter the scoped classifier

        # uses — after the threshold this auto-disables directives (raw

        # feedback) for the session, so a router that keeps mis-reading

        # the user yields to the model rather than fighting it.

        _route = getattr(self, "_feedback_route", None)

        if (

            asset_specs

            and trigger != "phase_a"

            and _route is not None

            and not _route.get("allow_assets_block")

        ):

            self._record_classifier_overrule(

                expected_mode=(

                    "router:" + str(_route.get("primary_intent", "no_assets"))

                ),

                model_emitted="assets_block",

                feedback_preview=str(_route.get("user_visible_issue", ""))[:200],

            )



        # P1 (MK trace 20260528): on a seed restart with COMPLETE declared
        # media on disk, SKIP phase_a generation. Orphan leftovers that are
        # not in declared PATHS do not count as coverage. Seed media regen
        # (art intent + incomplete/replace) must NOT skip.

        declared_a = list(
            getattr(self, "_seed_declared_asset_names", None) or []
        )
        declared_s = list(
            getattr(self, "_seed_declared_sound_names", None) or []
        )
        coverage_complete = (
            _declared_stems_complete(declared_a, self._session_assets)
            and _declared_stems_complete(declared_s, self._session_sounds)
        )
        seed_regen = bool(
            getattr(self, "_seed_media_regen", False)
            or getattr(self, "_assets_only_goal", False)
        )
        skip_phase_a = False
        if (
            trigger == "phase_a"
            and self.seed_file is not None
            and (asset_specs or sound_specs)
            and not seed_regen
        ):
            if declared_a or declared_s:
                skip_phase_a = coverage_complete
            else:
                # No PATHS declared — legacy: any on-disk session media.
                skip_phase_a = bool(
                    self._session_assets or self._session_sounds
                )

        if skip_phase_a:

            self._trace({

                "kind": "seed_phase_a_media_skipped",

                "have_assets": len(self._session_assets),

                "have_sounds": len(self._session_sounds),

                "coverage_complete": coverage_complete,

                "requested_assets": [

                    str(s.get("name") or "") for s in asset_specs

                ],

                "requested_sounds": [

                    str(s.get("name") or "") for s in sound_specs

                ],

            })

            yield self._record(AgentEvent(

                "info",

                f"[dim]phase_a asset/sound generation skipped — seed has "

                f"{len(self._session_assets)} asset(s) and "

                f"{len(self._session_sounds)} sound(s) on disk; "

                f"reusing existing media.[/dim]",

            ))

            return



        # Phase 0.13 — when the model emits a mid-session <assets> block

        # BUT the user has queued feedback asking for a style rebrand,

        # the model is about to generate sprites guaranteed to be

        # discarded next turn. Defer the asset gen and queue a coaching

        # note so the next user-turn re-emits the same <assets> with the

        # user's style baked in. Prevents the "model emitted 14 wrong-

        # style sprites while user feedback waited 25 minutes" failure

        # mode from the 2026-05-22 trace. General behavior — fires only

        # on the conjunction of mid-session model-driven asset gen AND

        # style-rebrand intent in pending feedback.

        if (

            trigger == "mid_session"

            and asset_specs

            and self._pending_feedback

        ):

            joined_pending = "\n".join(self._pending_feedback)

            try:

                wants_rebrand = _feedback_requests_style_rebrand(joined_pending)

            except Exception:

                wants_rebrand = False

            if wants_rebrand:

                # Echo the asset names back so the model knows what to

                # re-emit; include the user feedback preview so the

                # coaching is grounded in their own words.

                deferred_names = [str(s.get("name", "")) for s in asset_specs]

                fb_preview = joined_pending[:240].replace("\n", " | ")

                self._trace({

                    "kind": "mid_session_assets_deferred_for_user_style",

                    "deferred_names": deferred_names,

                    "feedback_preview": fb_preview,

                })

                yield self._record(AgentEvent(

                    "info",

                    f"[yellow]mid-session <assets> deferred[/yellow] — "

                    f"user has queued style-rebrand feedback. "

                    f"{len(deferred_names)} asset spec(s) will be re-emitted "

                    "next turn with the user's style applied.",

                ))

                # Coaching note carried into the next user-turn so the

                # model knows the prior <assets> got dropped and must be

                # re-emitted with the new style.

                self._pending_coaching.append(

                    "MID-SESSION ASSET DEFERRAL — your last reply emitted "

                    f"<assets> for {len(deferred_names)} entries "

                    f"({', '.join(deferred_names[:6])}"

                    f"{'…' if len(deferred_names) > 6 else ''}) but the user "

                    "had QUEUED feedback asking for a style rebrand. The "

                    "asset gen was SKIPPED so you don't burn GPU on sprites "

                    "guaranteed to be replaced. Re-emit the SAME asset "

                    "names in THIS turn's <assets> block but with NEW "

                    "prompts that bake in the style the user described. "

                    "Do NOT use `from_image` for the rebrand (it would "

                    "carry the old style forward)."

                )

                # Bail out — no gen, no path injection, no session-asset

                # bookkeeping. The next iter's flush will inject user

                # feedback + this coaching note + the regular MEDIA-CHANGE

                # / STYLE-REBRAND directive.

                return

        # Strict scoped-media lock: when the user said "regenerate only the

        # existing media", reject additive names early (before generation).

        scoped = self._scoped_constraints or {}

        if (

            trigger == "mid_session"

            and scoped.get("mode") == "media_only"

            and scoped.get("media_name_lock")

        ):

            allowed_assets = set(scoped.get("allowed_asset_names") or [])

            allowed_sounds = set(scoped.get("allowed_sound_names") or [])

            asset_specs, dropped_new_assets = self._filter_media_specs_to_allowed(

                asset_specs, allowed_assets,

            )

            sound_specs, dropped_new_sounds = self._filter_media_specs_to_allowed(

                sound_specs, allowed_sounds,

            )

            if dropped_new_assets or dropped_new_sounds:

                allowed_asset_line = ", ".join(sorted(allowed_assets)) or "(none)"

                allowed_sound_line = ", ".join(sorted(allowed_sounds)) or "(none)"

                dropped_line = ", ".join(dropped_new_assets + dropped_new_sounds)

                self._trace({

                    "kind": "scoped_media_new_names_rejected",

                    "dropped": sorted(dropped_new_assets + dropped_new_sounds),

                    "allowed_assets": sorted(allowed_assets),

                    "allowed_sounds": sorted(allowed_sounds),

                })

                self._queue_internal_feedback(

                    "SCOPED MEDIA NAME LOCK: new names were ignored "

                    f"[{dropped_line}]. Use existing names only. Assets: "

                    f"{allowed_asset_line}. Sounds: {allowed_sound_line}."

                )

                yield self._record(AgentEvent(

                    "info",

                    "scoped media lock: dropped new names; only existing keys are allowed this turn.",

                ))

        if not asset_specs and not sound_specs and not video_specs:

            return



        # Asset overflow: model asked for more than _MAX_ASSETS_PER_TURN.

        # Across four DK traces, this drove the dominant failure mode —

        # the model requested 14 sprites, only the first 8 generated,

        # the rest 404'd in the browser, and the model spent multiple

        # iters patching drawImage symptoms. Surface the gap LOUDLY:

        # trace event the user sees, AND coach the model next turn so

        # it knows to split the request or use img2img chaining.

        if dropped_asset_names:

            # Golden trace build-a-dragon-s-lair-laserdis_20260626_224306 +

            # run_14: cap can still drop non-seed sprites after

            # prefer_video_seed_assets rescues i2v seeds. Link any remaining

            # affected_video_seeds in the trace + coaching (should be rare).

            _dropped_set = set(dropped_asset_names)

            affected_video_seeds = [

                s.get("name")

                for s in video_specs

                if isinstance(s, dict) and s.get("image") in _dropped_set

            ]

            overflow_event = {

                "kind": "asset_overflow",

                "requested": len(asset_specs) + len(dropped_asset_names),

                "generated_cap": len(asset_specs),

                "dropped": dropped_asset_names,

            }

            if affected_video_seeds:

                overflow_event["affected_video_seeds"] = affected_video_seeds

            self._trace(overflow_event)

            # Persist for iter-to-iter ASSETS_DROPPED_PENDING gate until generated.
            pending = list(getattr(self, "_pending_dropped_assets", None) or [])
            for _dn in dropped_asset_names:
                if _dn not in pending:
                    pending.append(_dn)
            self._pending_dropped_assets = pending
            pending_specs = list(
                getattr(self, "_pending_dropped_asset_specs", None) or []
            )
            by_name = {
                str(s.get("name") or ""): s
                for s in pending_specs if s.get("name")
            }
            for spec in dropped_asset_specs:
                nm = str(spec.get("name") or "").strip()
                if nm:
                    by_name[nm] = spec
            self._pending_dropped_asset_specs = list(by_name.values())

            yield self._record(AgentEvent(

                "info",

                f"[yellow]asset overflow[/yellow] — you requested "

                f"{len(asset_specs) + len(dropped_asset_names)} sprites "

                f"but the per-turn cap is {len(asset_specs)}. Dropped: "

                f"{', '.join(dropped_asset_names)}. Coaching the model to "

                "request the rest in a follow-up turn."

            ))

            coaching_msg = (

                "Your <assets> block requested "

                f"{len(asset_specs) + len(dropped_asset_names)} sprites but "

                f"the harness only generates up to {len(asset_specs)} per "

                "turn. These were DROPPED and will not exist on disk: "

                f"{', '.join(dropped_asset_names)}. To fix: either (a) emit "

                "another <assets> block on a later turn with just the "

                "missing names — the agent will fulfill it mid-session — "

                "or (b) use `from_image` chaining (one base sprite + N "

                "img2img variants) so multiple animation frames cost a "

                "single base generation. Do NOT reference dropped names in "

                "the code until they've been generated."

            )

            if affected_video_seeds:

                # One extra line, only when a dropped name is a video i2v seed.

                coaching_msg += (

                    " IMPORTANT: dropped key-art "

                    f"{', '.join(sorted(n for n in _dropped_set if any(s.get('image') == n for s in video_specs if isinstance(s, dict))))}"

                    " is the image-to-video seed for cutscene(s) "

                    f"{', '.join(str(n) for n in affected_video_seeds)} — "

                    "those clips will fall back to text-to-video and DRIFT "

                    "off your locked character look. Keep video key-art under "

                    "the asset cap (or request it first) so the cutscenes "

                    "match the sprites."

                )

            self._pending_coaching.append(coaching_msg)



        # Mid-session: capture pre-existing assets so the feedback block

        # we synthesize at the end reflects only the *new* additions.

        new_asset_paths: dict[str, Path] = {}

        new_sound_paths: dict[str, Path] = {}

        new_video_paths: dict[str, Path] = {}

        new_looping: set[str] = set()

        pre_asset_hashes: dict[str, str | None] = {}

        pre_sound_hashes: dict[str, str | None] = {}

        if trigger == "mid_session":

            for spec in asset_specs:

                nm = str(spec.get("name", "")).strip()

                if nm and nm in self._session_assets:

                    pre_asset_hashes[nm] = self._file_hash16(self._session_assets.get(nm))

            for spec in sound_specs:

                nm = str(spec.get("name", "")).strip()

                if nm and nm in self._session_sounds:

                    pre_sound_hashes[nm] = self._file_hash16(self._session_sounds.get(nm))



        if asset_specs:

            _diffuser_lbl = diffuser_display_label(self._asset_generator)

            yield self._record(AgentEvent(

                "info",

                f"{trigger}: requested {len(asset_specs)} asset(s); "

                f"loading {_diffuser_lbl} (first call only, ~30-60s)…",

                {

                    "trigger": trigger,

                    "assets_requested": [s["name"] for s in asset_specs],

                    "diffuser_label": _diffuser_lbl,

                },

            ))

            yield self._record(AgentEvent(

                "activity", "generating_assets",

                {

                    "label": "generating sprites",

                    "requested": len(asset_specs),

                    "produced": 0,

                },

            ))

            if self._asset_generator is None:

                self._asset_generator = await asyncio.to_thread(

                    try_load_image_generator,

                )

            if self._asset_generator is None:

                yield self._record(AgentEvent("activity", "idle"))

                yield self._record(AgentEvent(

                    "info",

                    f"{_diffuser_lbl} not reachable (no CUDA / no diffusers / "

                    "Colossal_Cave/diffusion_manager.py missing) — "

                    "proceeding without assets, model will draw "

                    "procedurally."

                ))

            else:

                # Always derive the assets dir from _session_id. When the

                # session was started from a seed (chat.py reuses the

                # seed's path as out_path), _session_id IS the seed's

                # basename, so generation merges into the seed's existing

                # `<basename>_assets/` folder. Fresh sessions get a fresh

                # folder. No override mechanism needed.

                session_assets_dir = (

                    self.out_path.parent / f"{self._session_id}_assets"

                )

                try:

                    produced = await asyncio.to_thread(

                        generate_assets,

                        asset_specs,

                        session_assets_dir,

                        image_generator=self._asset_generator,

                    )

                except Exception as e:

                    yield self._record(AgentEvent("activity", "idle"))

                    yield self._record(AgentEvent(

                        "info",

                        f"asset generation crashed: {e!r} — proceeding without."

                    ))

                    produced = {}

                # Merge — mid-session adds extend, never overwrite.

                # Same-name keys take the newer path (model re-rendered

                # an existing asset on purpose).

                new_asset_paths = dict(produced)

                self._session_assets.update(produced)
                if produced and getattr(self, "_pending_dropped_assets", None):
                    self._pending_dropped_assets = [
                        n for n in self._pending_dropped_assets
                        if n not in produced
                    ]
                if produced and getattr(self, "_pending_dropped_asset_specs", None):
                    self._pending_dropped_asset_specs = [
                        s for s in self._pending_dropped_asset_specs
                        if str(s.get("name") or "") not in produced
                    ]

                per_asset = getattr(

                    self._asset_generator, "last_stats", None,

                ) or []

                # At-a-glance "did each pose frame actually move off idle?"

                # signal (2026-05-31): {name: delta-vs-parent} for from_image

                # frames. A reviewer can spot a cloned pose (delta < ~0.04)

                # without digging into per_asset or regenerating. Genre-free.

                pose_deltas = {

                    s.get("name"): s.get("parent_delta")

                    for s in per_asset

                    if isinstance(s, dict) and s.get("from_image")

                    and isinstance(s.get("parent_delta"), (int, float))

                }

                # Sprite facing/orientation is no longer audited or mirror-

                # flipped in the pipeline. The convention (author right-facing

                # art, flip in code) lives in the playbook

                # (directional-art-faces-right); art-direction policy stays in

                # memory, not in code.

                self._trace({

                    "kind": "assets_generated",

                    "trigger": trigger,

                    "requested": len(asset_specs),

                    "produced": len(produced),

                    "names": list(produced.keys()),

                    "paths": {name: str(path) for name, path in produced.items()},

                    "session_dir": str(session_assets_dir),

                    "pose_deltas": pose_deltas,

                    "failures": {
                        str(stat.get("name") or "unknown"): str(stat.get("error"))[:240]
                        for stat in per_asset
                        if isinstance(stat, dict) and stat.get("error")
                    },

                    "per_asset": [
                        {
                            key: value for key, value in stat.items()
                            if key not in {"prompt", "negative_prompt", "spec", "prose"}
                        }
                        for stat in per_asset if isinstance(stat, dict)
                    ],

                })

                yield self._record(AgentEvent(

                    "assets",

                    f"{len(produced)}/{len(asset_specs)} generated",

                    {

                        "trigger": trigger,

                        "requested": len(asset_specs),

                        "produced": len(produced),

                        "session_dir": str(session_assets_dir),

                        "paths": {n: str(p) for n, p in produced.items()},

                        "per_asset": per_asset,

                    },

                ))

                yield self._record(AgentEvent("activity", "idle"))

                if produced:

                    yield self._record(AgentEvent(

                        "info",

                        f"generated {len(produced)}/"

                        f"{len(asset_specs)} sprites at "

                        f"{session_assets_dir}",

                        {"assets": {n: str(p) for n, p in produced.items()}},

                    ))

                # Point-and-click: when VLM critique is on, ground bg_* PNGs
                # so hotspot rects can match where objects actually landed.
                if produced and trigger == "phase_a":
                    try:
                        await self._maybe_run_pointclick_vlm_grounding(
                            produced,
                            plan_text=(
                                (getattr(self, "_criteria", None) or "")
                                + "\n"
                                + (reply or "")
                            ),
                        )
                    except Exception as exc:
                        self._trace({
                            "kind": "pointclick_grounding_error",
                            "err": str(exc)[:200],
                        })

                # Per-asset success timing: one line each when small,

                # else a summary. Cache hits (gen_seconds≈0) called out.

                ok_stats = [

                    s for s in per_asset

                    if isinstance(s, dict) and not s.get("error")

                    and s.get("name") in produced

                ]

                if ok_stats:

                    if len(ok_stats) <= 5:

                        for s in ok_stats:

                            secs = float(s.get("gen_seconds") or 0.0)

                            cached = secs < 0.5  # heuristic; cache hits are <100ms

                            tag = " (cached)" if cached else ""

                            yield self._record(AgentEvent(

                                "info",

                                f"  asset {s.get('name','?')}: "

                                f"{secs:.1f}s{tag}",

                            ))

                    else:

                        total = sum(

                            float(s.get("gen_seconds") or 0.0)

                            for s in ok_stats

                        )

                        cached = sum(

                            1 for s in ok_stats

                            if float(s.get("gen_seconds") or 0.0) < 0.5

                        )

                        avg = total / max(1, len(ok_stats))

                        yield self._record(AgentEvent(

                            "info",

                            f"asset gen: {len(ok_stats)}/{len(asset_specs)} "

                            f"in {total:.1f}s (avg {avg:.1f}s/sprite, "

                            f"{cached} cached)",

                        ))

                if len(produced) < len(asset_specs):

                    failed = [

                        s for s in per_asset

                        if isinstance(s, dict) and s.get("error")

                    ]

                    if failed:

                        yield self._record(AgentEvent(

                            "info",

                            f"asset gen: {len(failed)}/{len(asset_specs)} "

                            f"failed — see per-asset reasons below"

                        ))

                        for s in failed:

                            name = s.get("name", "?")

                            err = s.get("error", "(no reason captured)")

                            err_line = str(err)[:400]

                            yield self._record(AgentEvent(

                                "info",

                                f"  - {name}: {err_line}"

                            ))



                # Derived-frame sanity: any sprite chained from a parent

                # (from_image) that came out near-identical to that parent

                # probably did NOT render the requested pose change — the

                # classic "punch sprite looks like idle" silent failure. The

                # model can't see its own art, so log it AND queue it as

                # actionable feedback for the next turn. Genre-free.

                derived = [

                    s for s in per_asset

                    if isinstance(s, dict)

                    and s.get("name") in produced

                    and s.get("from_image")

                    and isinstance(s.get("parent_delta"), (int, float))

                ]

                if derived:

                    # Signal-driven: the model declared animation frames, so

                    # the visual critic should verify the motion reads.

                    self._declared_anim_frames = True

                near_identical = [

                    s for s in derived

                    if s["parent_delta"] < _DERIVED_FRAME_MIN_DELTA

                ]

                # Update the done-gate set: a frame that just regenerated

                # with a real delta clears; a dead one (re)arms the block.

                for s in derived:

                    if s["parent_delta"] >= _DERIVED_FRAME_MIN_DELTA:

                        self._dead_anim_frames.pop(s["name"], None)

                for s in near_identical:

                    self._dead_anim_frames[s["name"]] = float(s["parent_delta"])

                if near_identical:

                    self._trace({

                        "kind": "derived_frame_near_identical",

                        "trigger": trigger,

                        "assets": [

                            {"name": s["name"], "from_image": s["from_image"],

                             "parent_delta": s["parent_delta"]}

                            for s in near_identical

                        ],

                    })

                    warn_lines = [

                        # Sentinel keeps this advisory OUT of the user-art-

                        # request classifier (it is not a request to make art;

                        # it explicitly says regen is a dead end).

                        _HARNESS_ADVISORY_SENTINEL,

                        "ASSET SANITY WARNING — these generated sprites came "

                        "out nearly identical to the `from_image` parent they "

                        "were derived from, so the requested pose change "

                        "likely did NOT render (e.g. a 'punch' frame that "

                        "looks just like idle). The pixels barely differ:",

                    ]

                    for s in near_identical:

                        pct = round((1.0 - float(s["parent_delta"])) * 100)

                        line = (

                            f"  - `{s['name']}` ≈{pct}% identical to parent "

                            f"`{s['from_image']}` (delta "

                            f"{s['parent_delta']:.3f})"

                        )

                        warn_lines.append(line)

                        yield self._record(AgentEvent("info", line.strip()))

                    warn_lines.append(

                        "This is COSMETIC and does not block shipping. Do NOT "

                        "try to regenerate these frames to fix it: img2img "

                        "cannot change a pose (it stays locked to idle), and a "

                        "fresh txt2img frame will not stay consistent with the "

                        "character already in the game — consistency is the "

                        "hard constraint. Keep using the sprites you have, "

                        "never draw the limb in code, and move on to the "

                        "behavior the user actually asked for."

                    )

                    self._queue_internal_feedback("\n".join(warn_lines))



        if sound_specs:

            yield self._record(AgentEvent(

                "info",

                f"{trigger}: requested {len(sound_specs)} sound(s); "

                "loading Stable Audio Open (first call only, ~30-60s)…",

                {

                    "trigger": trigger,

                    "sounds_requested": [s["name"] for s in sound_specs],

                },

            ))

            yield self._record(AgentEvent(

                "activity", "generating_sounds",

                {

                    "label": "generating sounds",

                    "requested": len(sound_specs),

                    "produced": 0,

                },

            ))

            if self._sound_generator is None:

                self._sound_generator = await asyncio.to_thread(

                    try_load_audio_generator,

                )

            if self._sound_generator is None:

                yield self._record(AgentEvent("activity", "idle"))

                yield self._record(AgentEvent(

                    "info",

                    "Stable Audio Open not reachable (no CUDA / no MPS / "

                    "diffusers or soundfile missing) — proceeding "

                    "without audio, model will ship a silent game."

                ))

            else:

                # Mirror the asset path above: dir comes from _session_id,

                # which already matches the seed's basename when seeded.

                session_sounds_dir = (

                    self.out_path.parent / f"{self._session_id}_sounds"

                )

                try:

                    produced = await asyncio.to_thread(

                        generate_sounds,

                        sound_specs,

                        session_sounds_dir,

                        audio_generator=self._sound_generator,

                    )

                except Exception as e:

                    yield self._record(AgentEvent("activity", "idle"))

                    yield self._record(AgentEvent(

                        "info",

                        f"sound generation crashed: {e!r} — proceeding without."

                    ))

                    produced = {}

                new_sound_paths = dict(produced)

                self._session_sounds.update(produced)

                # Recompute looping subset from the latest spec set

                # combined with already-installed sounds.

                new_looping = {

                    str(s.get("name", "")).strip()

                    for s in sound_specs if s.get("loop")

                } & set(produced.keys())

                self._session_looping |= new_looping

                per_sound = getattr(

                    self._sound_generator, "last_stats", None,

                ) or []

                self._trace({

                    "kind": "sounds_generated",

                    "trigger": trigger,

                    "requested": len(sound_specs),

                    "produced": len(produced),

                    "names": list(produced.keys()),

                    "paths": {name: str(path) for name, path in produced.items()},

                    "looping": sorted(new_looping),

                    "session_dir": str(session_sounds_dir),

                    "failures": {
                        str(stat.get("name") or "unknown"): str(stat.get("error"))[:240]
                        for stat in per_sound
                        if isinstance(stat, dict) and stat.get("error")
                    },

                    "per_sound": [
                        {
                            key: value for key, value in stat.items()
                            if key not in {"prompt", "negative_prompt", "spec", "prose"}
                        }
                        for stat in per_sound if isinstance(stat, dict)
                    ],

                })

                yield self._record(AgentEvent(

                    "sounds",

                    f"{len(produced)}/{len(sound_specs)} generated",

                    {

                        "trigger": trigger,

                        "requested": len(sound_specs),

                        "produced": len(produced),

                        "session_dir": str(session_sounds_dir),

                        "paths": {n: str(p) for n, p in produced.items()},

                        "looping": sorted(new_looping),

                        "per_sound": per_sound,

                    },

                ))

                yield self._record(AgentEvent("activity", "idle"))

                if produced:

                    yield self._record(AgentEvent(

                        "info",

                        f"generated {len(produced)}/"

                        f"{len(sound_specs)} sounds at "

                        f"{session_sounds_dir}",

                        {"sounds": {n: str(p) for n, p in produced.items()}},

                    ))

                if len(produced) < len(sound_specs):

                    failed = [

                        s for s in per_sound

                        if isinstance(s, dict) and s.get("error")

                    ]

                    if failed:

                        yield self._record(AgentEvent(

                            "info",

                            f"sound gen: {len(failed)}/{len(sound_specs)} "

                            f"failed — see per-sound reasons below"

                        ))

                        for s in failed:

                            name = s.get("name", "?")

                            err = s.get("error", "(no reason captured)")

                            err_line = str(err)[:400]

                            yield self._record(AgentEvent(

                                "info",

                                f"  - {name}: {err_line}"

                            ))



        # Video cutscenes — generated AFTER assets so an `image` field can

        # seed image-to-video from a key-art sprite produced THIS turn.

        # Each clip costs minutes of GPU; surface per-clip progress.

        if video_specs:

            yield self._record(AgentEvent(

                "info",

                f"{trigger}: requested {len(video_specs)} cutscene "

                "video(s); Wan2.2 runs per-clip in a subprocess "

                "(~2-5 min per clip)…",

                {

                    "trigger": trigger,

                    "videos_requested": [s["name"] for s in video_specs],

                },

            ))

            yield self._record(AgentEvent(

                "activity", "generating_videos",

                {

                    "label": "generating cutscene videos",

                    "requested": len(video_specs),

                    "produced": 0,

                },

            ))

            if self._video_generator is None:

                self._video_generator = try_load_video_generator()

            if self._video_generator is None:

                yield self._record(AgentEvent("activity", "idle"))

                yield self._record(AgentEvent(

                    "info",

                    "video backend not reachable (macOS: .venv-video/"

                    "mlx-gen missing; Linux: torch/diffusers missing) — "

                    "proceeding without cutscenes."

                ))

            else:

                session_videos_dir = (

                    self.out_path.parent / f"{self._session_id}_videos"

                )

                # Memory-pressure guard: on by default when free RAM is low
                # (see _mlx_coder_memory_pressure). Skips small MLX models.
                _mem = await asyncio.to_thread(self._free_memory_before_video)

                if _mem.get("freed"):

                    yield self._record(AgentEvent(

                        "info",

                        "freed memory before video to avoid an OS memory-"

                        f"pressure kill: {', '.join(_mem['freed'])} "

                        f"({_mem.get('available_gb')} GB RAM free)",

                    ))

                try:

                    produced = await asyncio.to_thread(

                        generate_videos,

                        video_specs,

                        session_videos_dir,

                        video_generator=self._video_generator,

                        asset_paths=self._session_assets,

                    )

                except Exception as e:

                    yield self._record(AgentEvent("activity", "idle"))

                    yield self._record(AgentEvent(

                        "info",

                        f"video generation crashed: {e!r} — proceeding without."

                    ))

                    produced = {}

                new_video_paths = dict(produced)

                self._session_videos.update(produced)

                per_video = getattr(

                    self._video_generator, "last_stats", None,

                ) or []

                self._trace({

                    "kind": "videos_generated",

                    "trigger": trigger,

                    "requested": len(video_specs),

                    "produced": len(produced),

                    "names": list(produced.keys()),

                    "session_dir": str(session_videos_dir),

                    "per_video": per_video,

                })

                yield self._record(AgentEvent(

                    "videos",

                    f"{len(produced)}/{len(video_specs)} generated",

                    {

                        "trigger": trigger,

                        "requested": len(video_specs),

                        "produced": len(produced),

                        "session_dir": str(session_videos_dir),

                        "paths": {n: str(p) for n, p in produced.items()},

                        "per_video": per_video,

                    },

                ))

                yield self._record(AgentEvent("activity", "idle"))

                if produced:

                    yield self._record(AgentEvent(

                        "info",

                        f"generated {len(produced)}/"

                        f"{len(video_specs)} cutscene videos at "

                        f"{session_videos_dir}",

                        {"videos": {n: str(p) for n, p in produced.items()}},

                    ))

                    for s in per_video:

                        if isinstance(s, dict) and not s.get("error"):

                            secs = float(s.get("gen_seconds") or 0.0)

                            tag = " (cached)" if s.get("cache_hit") else (

                                " (i2v)" if s.get("i2v") else ""

                            )

                            yield self._record(AgentEvent(

                                "info",

                                f"  video {s.get('name','?')}: "

                                f"{secs:.0f}s{tag}",

                            ))

                if len(produced) < len(video_specs):

                    failed = [

                        s for s in per_video

                        if isinstance(s, dict) and s.get("error")

                    ]

                    for s in failed:

                            yield self._record(AgentEvent(

                                "info",

                                f"  - video {s.get('name','?')}: "

                                f"{str(s.get('error',''))[:400]}"

                            ))



        # Post-media VRAM relief when free RAM is low (default on). Never MLX LLM.

        if asset_specs or sound_specs:

            if self._should_release_diffusers_after_media():

                tripped, metric_gb, phys_gb = self._mlx_coder_memory_pressure()

                freed = await asyncio.to_thread(self._release_diffusers_vram)

                if freed:

                    self._trace({

                        "kind": "diffuser_memory_relief",

                        "trigger": trigger,

                        "available_gb": metric_gb,

                        "phys_gb": phys_gb,

                        "freed": freed,

                    })

                    yield self._record(AgentEvent(

                        "info",

                        "unloaded sprite/sound models to free VRAM "

                        f"({metric_gb if tripped else phys_gb} GB "

                        f"{'RAM free' if tripped else 'physical RAM'}): "

                        f"{', '.join(freed)} — they reload automatically "

                        "if you request new art or audio",

                    ))



        # Mid-session only: synthesize a feedback line that re-emits the

        # asset/sound paths block, so the model's next user turn sees

        # the new files via the existing _flush_user_injections channel.

        # Phase A doesn't need this — the first-build assembler already

        # renders these blocks inline.

        if trigger == "mid_session" and (

            new_asset_paths or new_sound_paths or new_video_paths

        ):

            # Mark the iter as having done real preparatory work, so a

            # follow-up "no usable code" outcome doesn't get charged

            # against max_iters. See _media_regenerated_this_iter init.

            self._media_regenerated_this_iter = True

            # MK trace 20260517_220025 fix: when an existing asset is

            # regenerated with the SAME name, the file on disk has

            # already been replaced — the existing drawImage(ASSETS.X)

            # call picks up the new pixels automatically. Injecting the

            # heavy ASSETS_LOADER block in that case told the model "add

            # a new loader entry", which on a small model led to a 30-

            # minute repeating asset-prompt stream. Split into:

            #   - already_in_html: regen confirmation only

            #   - new_to_html: full loader block

            html_for_refs = self._current_file or ""

            html_asset_refs = self._scan_html_for_asset_refs(html_for_refs)

            html_sound_refs = self._scan_html_for_sound_refs(html_for_refs)

            already_assets = {

                n: p for n, p in new_asset_paths.items()

                if n in html_asset_refs

            }

            new_assets = {

                n: p for n, p in new_asset_paths.items()

                if n not in html_asset_refs

            }

            if new_assets:

                pending = getattr(self, "_new_assets_not_in_html", None)

                if pending is None:

                    self._new_assets_not_in_html = set(new_assets.keys())

                else:

                    self._new_assets_not_in_html = pending | set(new_assets.keys())

            already_sounds = {

                n: p for n, p in new_sound_paths.items()

                if n in html_sound_refs

            }

            new_sounds = {

                n: p for n, p in new_sound_paths.items()

                if n not in html_sound_refs

            }

            asset_replacements = {

                n: {

                    "old_hash": pre_asset_hashes.get(n),

                    "new_hash": self._file_hash16(p),

                    "changed": (

                        pre_asset_hashes.get(n) is not None

                        and self._file_hash16(p) is not None

                        and pre_asset_hashes.get(n) != self._file_hash16(p)

                    ),

                }

                for n, p in already_assets.items()

            }

            sound_replacements = {

                n: {

                    "old_hash": pre_sound_hashes.get(n),

                    "new_hash": self._file_hash16(p),

                    "changed": (

                        pre_sound_hashes.get(n) is not None

                        and self._file_hash16(p) is not None

                        and pre_sound_hashes.get(n) != self._file_hash16(p)

                    ),

                }

                for n, p in already_sounds.items()

            }



            msg_parts: list[str] = []

            if already_assets or already_sounds:

                # Compact confirmation; no loader pattern, no procedural-

                # vs-sprite warnings, no orientation reminder. The

                # existing draw / Audio() call already references these

                # names by path, so the regen is transparent.

                lines = [

                    "================ MEDIA REGEN COMPLETE ================",

                    "These names already exist in your HTML. The files on",

                    "disk have been REPLACED with newly generated content.",

                    "No JS patch is required — the existing drawImage() /",

                    "new Audio() calls will pick up the new pixels / audio",

                    "automatically on the next Chromium load.",

                ]

                if already_assets:

                    lines.append("")

                    lines.append("Regenerated sprites:")

                    for n in sorted(already_assets):

                        rep = asset_replacements.get(n) or {}

                        status = (

                            "changed"

                            if rep.get("changed") is True

                            else "unchanged/unknown"

                        )

                        lines.append(f"  - {n} ({status})")

                if already_sounds:

                    lines.append("")

                    lines.append("Regenerated sounds:")

                    for n in sorted(already_sounds):

                        rep = sound_replacements.get(n) or {}

                        status = (

                            "changed"

                            if rep.get("changed") is True

                            else "unchanged/unknown"

                        )

                        lines.append(f"  - {n} ({status})")

                lines.append(

                    "======================================================"

                )

                msg_parts.append("\n".join(lines))



            if new_assets or new_sounds or new_video_paths:

                # New names → emit the full loader block so the model

                # knows how to wire them in via a <patch>.

                blocks: list[str] = []

                if new_assets:

                    blocks.append(render_asset_paths_block(

                        new_assets, self.out_path,

                    ))

                if new_sounds:

                    new_looping_subset = {

                        n for n in new_looping if n in new_sounds

                    }

                    blocks.append(render_sound_paths_block(

                        new_sounds, self.out_path,

                        looping_names=new_looping_subset,

                    ))

                if new_video_paths:

                    # Videos are always treated as new — the loader block

                    # carries the full <video>-overlay wiring pattern.

                    blocks.append(render_video_paths_block(

                        new_video_paths, self.out_path,

                    ))

                blocks = [b for b in blocks if b]

                if blocks:

                    msg_parts.append(

                        "Mid-session asset/sound/video additions — load "

                        "these in your next patch and use them where "

                        "appropriate. The files exist on disk now:\n\n"

                        + "\n\n".join(blocks)

                    )



            if msg_parts:

                # Agent-generated media notice — queued via the internal

                # channel so the unhonored-asset-request detector can't

                # mistake it for a user art request (8 spurious reprompts

                # in trace 20260612_004616).

                self._queue_internal_feedback("\n\n".join(msg_parts))

                self._trace({

                    "kind": "midsession_asset_injection_queued",

                    "asset_names": list(new_asset_paths.keys()),

                    "sound_names": list(new_sound_paths.keys()),

                    "video_names": list(new_video_paths.keys()),

                    "already_in_html_assets": sorted(already_assets.keys()),

                    "already_in_html_sounds": sorted(already_sounds.keys()),

                    "new_assets": sorted(new_assets.keys()),

                    "new_sounds": sorted(new_sounds.keys()),

                    "asset_replacements": asset_replacements,

                    "sound_replacements": sound_replacements,

                })



    # -- main loop ----------------------------------------------------------



