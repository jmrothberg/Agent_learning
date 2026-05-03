"""coder.py - the agent loop ("coding box").

Usage:
    python coder.py "Make a Snake game with a wraparound board and a score counter"

Optional flags:
    --model NAME        Override the Ollama model tag (default: see MODEL below)
    --max-iters N       Cap iterations (default 8)
    --out PATH          Where to save the final game (default games/game.html)
    --open              After finishing, open the result in your real browser

The loop:
    1. Send the system prompt + goal to Ollama.
    2. Parse out <html_file>...</html_file> from the model's reply.
    3. Save it, run it in headless Chromium, build a short report.
    4. Feed report back to the model as the next user turn.
    5. Stop on <done/>, on too many iterations, or on Ctrl-C.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import webbrowser
from pathlib import Path

import ollama

from prompts import CRITIQUE_INSTRUCTION, PLAN_INSTRUCTION, SYSTEM_PROMPT
from tools import format_report_for_model, test_html_file


# ---------------------------------------------------------------------------
# CHANGE-ME CONSTANT
# The user asked for "qwen3.6:27b". That tag isn't in Ollama's public
# registry as of writing - if `ollama pull` fails, swap this for one of:
#   "qwen3:30b"        (19GB MoE, newest, recommended)
#   "qwen3-coder:30b"  (code-tuned variant if available locally)
#   "qwen2.5-coder:32b"
# ---------------------------------------------------------------------------
MODEL = "qwen3.6:27b"


# Regexes used to pull the two tags out of the model reply.
# DOTALL so the html_file body can span newlines. Non-greedy so we stop at the
# first closing tag (in case the model accidentally writes the tag in a comment).
_HTML_RE = re.compile(r"<html_file>\s*(.*?)\s*</html_file>", re.DOTALL | re.IGNORECASE)
_DONE_RE = re.compile(r"<done\s*/?>", re.IGNORECASE)
# CONFIRM is the second-stage "yes I really mean done" tag, sent only after
# the self-critique prompt. Separate from <done/> so we can tell the difference.
_CONFIRM_RE = re.compile(r"<confirm[_-]?done\s*/?>", re.IGNORECASE)


def extract_html(reply: str) -> str | None:
    """Pull the HTML file body from the model's reply, or None if missing."""
    m = _HTML_RE.search(reply)
    if not m:
        return None
    body = m.group(1).strip()
    # Sometimes models wrap it in a markdown fence even when told not to.
    # Strip ```html ... ``` defensively.
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body)
    return body.strip() or None


def model_says_done(reply: str) -> bool:
    return bool(_DONE_RE.search(reply))


def model_confirms_done(reply: str) -> bool:
    """True when the model has finished the self-critique and stands by <done/>."""
    return bool(_CONFIRM_RE.search(reply))


def chat_stream(model: str, messages: list[dict]) -> str:
    """Streaming call to Ollama: prints chunks live, returns the full text.

    Streaming matters for UX with a 27B model: reply latency is 30-60s+, so
    seeing tokens flow tells you the model is alive AND lets you Ctrl-C if it
    visibly veers off-format. We accumulate the chunks ourselves so the rest
    of the loop still gets the full reply.
    """
    # `options` lets us nudge the model. temperature=0.4 is a good middle for
    # creative-but-stable code. num_ctx must fit the convo; bumping to 8192
    # because we're sending whole HTML files back and forth.
    parts: list[str] = []
    print("  > ", end="", flush=True)
    for chunk in ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": 0.4, "num_ctx": 8192},
    ):
        piece = chunk.get("message", {}).get("content", "")
        if not piece:
            continue
        parts.append(piece)
        # Print indented so the streaming text visually nests under the
        # iteration header. Newlines inside the stream re-indent on next print.
        sys.stdout.write(piece.replace("\n", "\n  > "))
        sys.stdout.flush()
    print()  # final newline so the next log line starts cleanly
    return "".join(parts)


def _safe_chat(model: str, messages: list[dict]) -> str | None:
    """Wrap chat_stream so connection / pull errors are reported clearly once.

    Returns the assistant text, or None if the call failed (caller should bail).
    """
    try:
        return chat_stream(model, messages)
    except ollama.ResponseError as e:
        # Most common: model not pulled. Tell the user clearly and bail.
        print(f"\nOllama error: {e}", file=sys.stderr)
        print(
            f"Hint: try `ollama pull {model}` or edit the MODEL constant "
            f"at the top of coder.py.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"\nOllama call failed: {e}", file=sys.stderr)
        print(
            "Hint: is the Ollama server running? Start it with `ollama serve`.",
            file=sys.stderr,
        )
        return None


def run(goal: str, model: str, max_iters: int, out_path: Path, open_when_done: bool) -> int:
    """Drive the agent loop. Returns process exit code."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Conversation seeded with the system prompt (goal baked into it once).
    messages: list[dict] = [
        # NOTE: use .replace not .format - the skeleton in SYSTEM_PROMPT
        # contains literal CSS/JS braces which .format would try to parse.
        {"role": "system", "content": SYSTEM_PROMPT.replace("{goal}", goal)},
    ]

    # ---- PHASE A: planning turn (no code, no test) -------------------------
    # We force a code-free design pass first. For a 27B model this is the
    # difference between "snake but the head doesn't grow" and a real game.
    print("\n=== phase A: planning ===")
    messages.append({"role": "user", "content": PLAN_INSTRUCTION})
    t0 = time.time()
    plan_reply = _safe_chat(model, messages)
    if plan_reply is None:
        return 2
    print(f"  planning took {time.time() - t0:.1f}s")
    messages.append({"role": "assistant", "content": plan_reply})
    # Hand off to phase B with an explicit "now build it" instruction so the
    # model doesn't try to keep planning.
    messages.append(
        {
            "role": "user",
            "content": (
                "Plan accepted. Now write the FIRST version of the game per "
                "your plan. Output the COMPLETE file in <html_file>...</html_file> "
                "tags as instructed in the system prompt."
            ),
        }
    )

    # ---- PHASE B: build/iterate loop ---------------------------------------
    # awaiting_confirm is True ONLY while we're in a self-critique round (phase
    # C). When the model first says <done/>, we set this and send the critique
    # prompt. The model's next reply is either a fixed file (back to normal
    # iteration) or <confirm_done/> (we ship).
    awaiting_confirm = False

    for iteration in range(1, max_iters + 1):
        print(f"\n=== iteration {iteration}/{max_iters} ===")
        t0 = time.time()
        reply = _safe_chat(model, messages)
        if reply is None:
            return 2
        gen_secs = time.time() - t0
        print(f"  model replied in {gen_secs:.1f}s ({len(reply)} chars)")

        # Save the assistant turn so it stays in the conversation.
        messages.append({"role": "assistant", "content": reply})

        # If we're in a critique round and the model confirms with no new code,
        # we're truly done. (This must be checked BEFORE extract_html so a
        # confirm-only reply doesn't fall through to "no <html_file>" handling.)
        if awaiting_confirm and extract_html(reply) is None and model_confirms_done(reply):
            print("\nDone: model confirmed after self-critique.")
            break

        html = extract_html(reply)
        if html is None:
            # Model went off-format. Nudge it once and continue.
            print("  no <html_file> tag found - nudging model")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "I could not find a <html_file>...</html_file> block in "
                        "your reply. Please re-send the COMPLETE game wrapped in "
                        "those exact tags."
                    ),
                }
            )
            continue

        out_path.write_text(html, encoding="utf-8")
        print(f"  wrote {out_path} ({len(html)} bytes)")

        # Run in headless browser and get a short report.
        try:
            report = test_html_file(out_path)
        except Exception as e:
            # Treat harness failure like an error the model should react to.
            print(f"  test harness crashed: {e}")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The browser test harness itself crashed: {e}\n"
                        "Please simplify the page (e.g. avoid network calls, "
                        "huge inline assets) and try again."
                    ),
                }
            )
            continue

        report_text = format_report_for_model(report)
        # Console summary - keep brief per project rules.
        if report["ok"]:
            status = "OK"
        else:
            status = (
                f"{len(report['errors'])} error(s), "
                f"{len(report.get('soft_warnings', []))} issue(s)"
            )
        print(f"  test result: {status}")

        # ---- PHASE C trigger: clean run + model claims done ---------------
        # Instead of breaking, we ask the model to self-critique ONCE.
        if report["ok"] and model_says_done(reply) and not awaiting_confirm:
            print("  clean run + <done/> -> entering self-critique")
            awaiting_confirm = True
            messages.append({"role": "user", "content": CRITIQUE_INSTRUCTION})
            continue

        # If the model produced new code DURING a critique round, treat it as
        # a normal iteration and exit critique mode - the report we just ran
        # is on the new code, so we keep iterating until clean again.
        if awaiting_confirm:
            awaiting_confirm = False

        # Build the next user turn: the report + a clear next-step instruction.
        if report["ok"]:
            next_msg = (
                f"{report_text}\n\n"
                "No errors. If the game is fully playable and polished, reply "
                "with the same file plus <done/>. Otherwise, improve gameplay, "
                "visuals, or feel and re-send the full file."
            )
        else:
            next_msg = (
                f"{report_text}\n\n"
                "Fix every ERROR and ISSUE above. Re-send the COMPLETE game in "
                "<html_file>...</html_file> tags."
            )
        messages.append({"role": "user", "content": next_msg})
    else:
        # Loop finished without break - we hit max_iters.
        print(f"\nReached max iterations ({max_iters}). Saving last attempt.")

    print(f"\nFinal game saved to: {out_path}")
    if open_when_done:
        webbrowser.open(f"file://{out_path.resolve()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Coding-box agent for HTML games (Ollama).")
    p.add_argument("goal", help="What game to build, in plain English.")
    p.add_argument("--model", default=MODEL, help=f"Ollama model tag (default: {MODEL})")
    p.add_argument("--max-iters", type=int, default=8, help="Max agent iterations.")
    p.add_argument(
        "--out",
        default="games/game.html",
        help="Output HTML path (default: games/game.html)",
    )
    p.add_argument("--open", action="store_true", help="Open final game in your browser.")
    args = p.parse_args()

    return run(
        goal=args.goal,
        model=args.model,
        max_iters=args.max_iters,
        out_path=Path(args.out),
        open_when_done=args.open,
    )


if __name__ == "__main__":
    sys.exit(main())
