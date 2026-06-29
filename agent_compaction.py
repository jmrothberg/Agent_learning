"""Message pruning / compaction extracted from agent.py.

Moved VERBATIM from `GameAgent` (no behavior change).
"""

from __future__ import annotations


import re


from agent_helpers import (

    _COMPACT_MESSAGE_CAP,

    _COMPACT_PRESSURE,

    _COMPACT_TOKEN_CEILING,

    _PRUNE_KEEP_RECENT_TURNS,

    _REPORT_BLOCK_BEGIN,

    _REPORT_BLOCK_RE,

    _STRUCTURED_PRUNE_THRESHOLD,

    _SUMMARIZE_MIN_HTML_BYTES,

    _SUMMARIZE_MIN_PROBES_BYTES,

)


class CompactionMixin:

    """Context compaction for GameAgent (see module docstring)."""


    def _summarize_content(self, c: str) -> str:
        """Replace embedded HTML blobs with size markers — keep tags + notes.

        Wolfenstein 2026-05-24 trace lesson: the OLD marker
        `<html_file>[omitted: N bytes of HTML; see snapshot]</html_file>`
        was shaped like a valid output tag, so a confused model parroted
        it back verbatim in its next reply (turns [02], [06], [08] of
        that trace's conversation.md). The parser saw an `<html_file>`
        wrapper around prose, the body didn't normalize to a document,
        materialize failed, and the agent entered an identical-reply
        loop sending the generic "no <patch> or <html_file>" fallback.

        The new marker uses an HTML COMMENT — no `<html_file>` /
        `<patch>` / markdown-fence wrapper. The model literally cannot
        copy-paste it as a fresh tag emission, and the embedded
        instruction tells the model what to do instead of just naming
        a byte count. Universal: no goal text, no genre.
        """
        def html_repl(m):
            n = len(m.group(1))
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
            # Marker intentionally uses no `<html_file>` / `<patch>`
            # substrings (not even inside prose) so a stressed model
            # has nothing to copy-paste as a fresh tag emission. The
            # comment shape also can't extract through the html
            # regex variants 1-6 since none of them match comments.
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-HTML: {n} bytes of HTML "
                f"body written to disk in this earlier turn. The "
                f"current file is shown inline below in the CURRENT "
                f"FILE ON DISK block; patch against that, do NOT "
                f"re-emit this marker. -->"
            )

        def fence_repl(m):
            body = m.group(1)
            n = len(body)
            if n < _SUMMARIZE_MIN_HTML_BYTES:
                return m.group(0)
            # Fix-round item 4: content sniff. The regex's optional `html`
            # tag means it also matches ```js / ``` fences — and worse, a
            # CLOSING fence followed by the NEXT opening fence, eliding the
            # PROSE between them and falsely asserting it was HTML written
            # to disk (trace 20260610_185238 turn [08] had the marker
            # spliced mid-sentence between two ```js reasoning fences).
            # Only elide when the body actually looks like an HTML document.
            body_head = body.lstrip()[:32].lower()
            if not (body_head.startswith("<!doctype") or body_head.startswith("<html")):
                return m.group(0)
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-FENCE: {n} bytes of "
                f"fenced HTML written to disk in this earlier turn. "
                f"Do NOT re-emit this marker; patch against the "
                f"CURRENT FILE ON DISK block below. -->"
            )

        def probes_repl(m):
            body = m.group(1)
            if len(body) < _SUMMARIZE_MIN_PROBES_BYTES:
                return m.group(0)
            # Rough def count: each probe object carries a "name" key.
            n_defs = body.count('"name"')
            return (
                f"<!-- HARNESS-OMITTED-PRIOR-PROBES: {n_defs} probe defs "
                f"from this earlier turn are superseded — the current "
                f"probe set lives in session state and runs every iter. "
                f"Do NOT re-emit this marker. -->"
            )

        c = self._SUMMARIZE_HTML_RE.sub(html_repl, c)
        c = self._SUMMARIZE_FENCE_RE.sub(fence_repl, c)
        c = self._SUMMARIZE_PROBES_RE.sub(probes_repl, c)
        return c

    def _maybe_reset_continuation_context(self, prior_clean: bool) -> bool:
        """Fresh-context continuation (frontier-agent pattern, 2026-06-12).

        Replace the accumulated history with [system prompt, state
        anchor] before the continuation message is appended. Gated:
          - prior session ended clean (mid-debugging history is never
            discarded),
          - a real system prompt sits at index 0,
          - enough history to be worth replacing (> 3 messages).
        Falls back to plain append (returns False) on any anchor-build
        failure. Only changes what we SEND — never cuts model output.
        Returns True when the reset was applied.
        """
        if not (
            prior_clean
            and len(self._messages) > 3
            and (self._messages[0].get("role") == "system")
        ):
            return False
        try:
            before_msgs = len(self._messages)
            before_chars = sum(
                len(str(m.get("content") or "")) for m in self._messages
            )
            summary = self._build_structured_summary()
            anchor_msg = {
                "role": "user",
                "content": (
                    "================ STATE ANCHOR (fresh-context "
                    "continuation) ================\n"
                    "The previous build passed its tests and shipped. "
                    "Older turns were replaced by the snapshot below — "
                    "treat it as authoritative for goal, criteria, "
                    "progress, and critical context. The user's NEW "
                    "request follows in the next message.\n\n"
                    f"{summary}\n"
                    "==========================================================="
                ),
            }
            self._messages = [self._messages[0], anchor_msg]
            after_chars = sum(
                len(str(m.get("content") or "")) for m in self._messages
            )
            self._trace({
                "kind": "continuation_context_reset",
                "before_messages": before_msgs,
                "after_messages": len(self._messages),
                "before_chars": before_chars,
                "after_chars": after_chars,
                # ~4 chars/token heuristic, telemetry only.
                "est_tokens_saved": max(
                    0, (before_chars - after_chars) // 4
                ),
            })
            return True
        except Exception as e:
            # Anchor build failed — keep the appended-history behavior;
            # never block the continuation itself.
            self._trace({
                "kind": "continuation_context_reset_failed",
                "err": str(e)[:180],
            })
            return False



    def _prune_messages(self) -> None:
        """Compress old turns so context stays bounded.

        Two strategies, by message count:
          * ≤ _PRUNE_KEEP_RECENT_TURNS+1 messages: no-op.
          * ≤ _STRUCTURED_PRUNE_THRESHOLD: per-turn HTML elision (the
            original behavior — keeps message shape, strips embedded HTML).
          * > _STRUCTURED_PRUNE_THRESHOLD: pi-mono-style structured
            compaction — replace messages 1..cutoff with a single
            deterministic state-anchor message; keep system prompt and
            last _PRUNE_KEEP_RECENT_TURNS turns.

        The system prompt (index 0) and the most recent K turns are
        always preserved verbatim.
        """
        n = len(self._messages)
        if n <= 1 + _PRUNE_KEEP_RECENT_TURNS:
            return

        # Token-aware gate: only do the LOSSY structured compaction when the
        # context window is actually filling (last coder prompt >=
        # _COMPACT_PRESSURE of num_ctx), OR as a hard safety cap when token
        # stats are unavailable. A high message count alone no longer triggers
        # it — so a big-context local model keeps full history (playbook +
        # every prior user ask) until the window is genuinely under pressure.
        pressure = float(getattr(self, "_last_prompt_pressure", 0.0) or 0.0)
        last_tokens = int(getattr(self, "_last_prompt_tokens", 0) or 0)
        # Forward-looking projection (2026-06-12, trace 20260612_132314):
        # the reactive gates above look at the PREVIOUS stream's prompt
        # size, so a single-turn jump slips through — a post-clean turn
        # (full file inline + playtest feedback + history) ballooned
        # 33K → 60K in one assembly step and the 27B silently emitted 0
        # tokens for 184s. _prune_messages runs AFTER the new user turn
        # is appended, so estimating the message list (~4 chars/token)
        # projects the prompt we are ABOUT to send. Images aren't
        # counted — this is a floor estimate, which is the safe side.
        projected_tokens = sum(
            len(str(m.get("content") or "")) for m in self._messages
        ) // 4
        compact_reason = None
        if pressure >= _COMPACT_PRESSURE:
            compact_reason = "token_pressure"
        elif _COMPACT_TOKEN_CEILING > 0 and last_tokens >= _COMPACT_TOKEN_CEILING:
            # Fix-round item 3: absolute ceiling — local backends stall
            # silently well before the relative pressure gate fires.
            compact_reason = "token_ceiling"
        elif _COMPACT_TOKEN_CEILING > 0 and projected_tokens >= _COMPACT_TOKEN_CEILING:
            compact_reason = "projected_ceiling"
        elif getattr(self, "_force_compact_after_stall", False):
            # Fix-round item 3: a silent 0-token stall just aborted — do NOT
            # rebuild the same giant prompt for the retry; compact first.
            compact_reason = "silent_stall_recovery"
        elif n > _COMPACT_MESSAGE_CAP:
            compact_reason = "count_cap"
        # One-shot flag: consumed whether or not it was the deciding reason.
        self._force_compact_after_stall = False

        if compact_reason is not None:
            cutoff = n - _PRUNE_KEEP_RECENT_TURNS
            summary = self._build_structured_summary()
            anchor_msg = {
                "role": "user",
                "content": (
                    "================ STATE ANCHOR (compaction) ================\n"
                    "Older turns were elided to keep context bounded. The "
                    "snapshot below is a deterministic summary of session "
                    "state — treat it as authoritative for goal, criteria, "
                    "progress, and critical context.\n\n"
                    f"{summary}\n"
                    "==========================================================="
                ),
            }
            new_messages = [self._messages[0], anchor_msg] + self._messages[cutoff:]
            # Post-compaction projection so future trace mining can tell
            # whether compaction actually got the next prompt under the
            # ceiling (it may not when the bulk lives in the kept-recent
            # window, e.g. a fresh full-file inline).
            projected_after = sum(
                len(str(m.get("content") or "")) for m in new_messages
            ) // 4
            self._trace({
                "kind": "structured_compaction",
                "reason": compact_reason,
                "prompt_tokens": getattr(self, "_last_prompt_tokens", 0),
                "projected_tokens": projected_tokens,
                "projected_tokens_after": projected_after,
                "num_ctx": getattr(self, "num_ctx", 0),
                "pressure": round(pressure, 3),
                "original_messages": n,
                "kept_recent": _PRUNE_KEEP_RECENT_TURNS,
                "summary_chars": len(summary),
                "new_messages": len(new_messages),
            })
            self._messages = new_messages
            return

        # Default elision path: keep message shape, strip embedded HTML
        # bodies. Cheap, lossy on iteration history, but safe.
        cutoff = n - _PRUNE_KEEP_RECENT_TURNS
        # Capability-round item 3: the NEWEST wrapped test report always
        # stays verbatim, even if it has fallen below the keep window.
        newest_report_idx = -1
        for i, m in enumerate(self._messages):
            if (
                m.get("role") == "user"
                and _REPORT_BLOCK_BEGIN in (m.get("content") or "")
            ):
                newest_report_idx = i
        collapsed_reports = 0
        for i in range(1, cutoff):
            msg = self._messages[i]
            # Do NOT rewrite user/system turns here. Mutating prior user
            # instructions (especially format examples) creates false context
            # and can derail one-shot generations.
            # EXCEPTION (item 3): turns wrapped in HARNESS-REPORT-BLOCK
            # sentinels are harness-authored test reports, not user
            # instructions. Superseded ones collapse to their 3-line
            # digest; feedback appended outside the wrapper survives.
            if msg.get("role") == "user":
                c = msg.get("content", "") or ""
                if _REPORT_BLOCK_BEGIN in c and i != newest_report_idx:
                    def _collapse(m_):
                        return (
                            "[superseded test report — collapsed by "
                            "harness; the newest report below is the "
                            "current truth]\n"
                            + m_.group(1).strip()
                        )
                    new_c = _REPORT_BLOCK_RE.sub(_collapse, c)
                    if new_c != c:
                        msg["content"] = new_c
                        collapsed_reports += 1
                continue
            if msg.get("role") != "assistant":
                continue
            c = msg.get("content", "") or ""
            new_c = self._summarize_content(c)
            if new_c != c:
                msg["content"] = new_c
        if collapsed_reports:
            self._trace({
                "kind": "report_turns_collapsed",
                "count": collapsed_reports,
                "newest_kept_idx": newest_report_idx,
            })

    # -- user-injection plumbing -------------------------------------------



