"""Extractor, row shape, dedupe, and the Chrome keep/drop rule for synth_html."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "synth_html", ROOT / "fine_tunning" / "synth_html.py",
)
assert _SPEC and _SPEC.loader
sh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sh)

PAGE = (
    "<!DOCTYPE html>\n<html><head><title>Hangman</title></head>\n"
    "<body><canvas id=\"c\"></canvas><script>\n"
    "const c = document.getElementById('c');\n"
    "const g = c.getContext('2d');\n"
    "g.fillText('hangman', 8, 16);\n"
    "</script></body></html>"
)
PROMPT = "Write me a game of hangman."


def test_ideas_file_has_1000_unique_prompts() -> None:
    ideas = sh.load_ideas(ROOT / "fine_tunning" / "ideas_1000.txt")
    assert len(ideas) == 1000
    assert len({sh.normalize_prompt(x) for x in ideas}) == 1000
    assert ideas[0] == PROMPT


def test_ideas_3000_one_game_each() -> None:
    ideas = sh.load_ideas(ROOT / "fine_tunning" / "ideas_3000.txt")
    keys = [sh.subject_key(line) for line in ideas]
    assert len(ideas) > 1000
    assert len(keys) == len(set(keys))
    def _rank(line: str) -> int:
        low = line.lower()
        if "three.js" in low:
            return 0
        if "8-bit" in low:
            return 1
        if " in phaser" in low or " in pixijs" in low or " in howler.js" in low:
            return 2
        return 3

    assert _rank(ideas[0]) == 0
    assert all(_rank(a) <= _rank(b) for a, b in zip(ideas, ideas[1:]))
    assert "tetris use nice 8 bit graphics as close to the original as possible" in keys
    assert "tetris" not in keys
    assert sum("three js" in k for k in keys) == 100
    assert sum(k.endswith(" in phaser") for k in keys) == 122
    assert sum(k.endswith(" in pixijs") for k in keys) == 57
    assert sum(k.endswith(" in howler js") for k in keys) == 15
    assert "hades" in keys
    assert "god of war 2018" in keys
    assert "god of war 2005" in keys
    starts = {line.split()[0].lower() for line in ideas}
    for word in ("write", "make", "build", "create", "give", "can"):
        assert word in starts


def test_fit_html_scales_the_page() -> None:
    wrapped = sh.fit_html(PAGE)
    assert "scale(" in wrapped
    assert "960" in wrapped and "900" in wrapped
    assert "\\u003cscript>" in wrapped
    assert "\\u003c/script>" in wrapped


def test_buggy_row_keeps_prompt_code_and_chrome_report() -> None:
    report = {
        "ok": False,
        "errors": ["UNCAUGHT: fx is not defined\n    at play (game.html:42:5)"],
        "warnings": [],
    }
    row = sh.pack_buggy_row(PROMPT, PAGE, "fx is not defined", report)
    assert row["kind"] == "buggy"
    assert row["messages"][1]["content"] == PROMPT
    assert "fillText" in row["messages"][2]["content"]
    assert row["messages"][3]["role"] == "bug"
    assert "game.html:42:5" in row["messages"][3]["content"]
    assert "game.html:42:5" in row["bug"]["errors"][0]
    assert 24 in sh.JOB_CHOICES and 48 in sh.JOB_CHOICES
    assert 32 in sh.JOB_CHOICES and 64 in sh.JOB_CHOICES


def test_subject_key_ignores_the_opening() -> None:
    assert sh.subject_key("Write me a game of hangman.") == "hangman"
    assert sh.subject_key("Make a small working version of Hades.") == "hades"
    assert sh.subject_key("Make a complete working version of Hades.") == "hades"
    assert sh.subject_key("Build me a two-player game of breakout.") == "breakout"


def test_extract_html_from_wrapper_and_rejects_prose() -> None:
    wrapped = f"<think>\nnote\n</think>\n<html_file>\n{PAGE}\n</html_file>\n"
    assert sh.extract_html(wrapped) == PAGE
    fenced = "```html\n" + PAGE + "\n```"
    assert sh.extract_html(fenced) == PAGE
    assert sh.extract_html("I could not write it.") is None
    assert sh.extract_html("") is None


def test_gen_messages_require_a_closed_file() -> None:
    msgs = sh.gen_messages(PROMPT)
    assert msgs[1] == {"role": "user", "content": PROMPT}
    system = msgs[0]["content"]
    assert system.startswith("Reply with one complete")
    assert "<html_file>" in system and "</html_file>" in system
    assert "No thinking" in system
    assert "requestAnimationFrame" in system
    assert "several colors" in system
    assert "CapsuleGeometry" not in system
    assert system != sh.SHORT_SYS
    three = sh.gen_messages("Make a small working version of Shadow of the Colossus in three.js.")
    three_sys = three[0]["content"]
    assert "unpkg.com/three@0.160.0/build/three.min.js" in three_sys
    assert "directional light" in three_sys
    assert "Do not use CapsuleGeometry" in three_sys
    assert "at most twelve meshes" in three_sys
    assert "r128" in three_sys
    eight = sh.gen_messages("Write me a small working version of Tetris. Use nice 8-bit graphics.")
    assert "fillRect" in eight[0]["content"]


def test_pack_row_matches_short_html_shape() -> None:
    row = sh.pack_row(PROMPT, PAGE)
    assert row["kind"] == "html"
    assert row["title"] == "Hangman"
    assert row["source"].startswith("synth:")
    assert len(row["sha"]) == 64
    system, user, assistant = row["messages"]
    assert system == {"role": "system", "content": sh.SHORT_SYS}
    assert user == {"role": "user", "content": PROMPT}
    assert assistant["role"] == "assistant"
    assert "<html_file>" in assistant["content"]
    assert "</html_file>" in assistant["content"]
    assert "<think>" in assistant["content"]
    assert "fillText" in assistant["content"]
    assert row["weight"] == 1


def test_rewrite_weight_sets_zero(tmp_path: Path) -> None:
    path = tmp_path / "synth.jsonl"
    row = sh.pack_row(PROMPT, PAGE)
    other = sh.pack_row("Write me a game of snake.", PAGE)
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )
    assert sh.rewrite_weight(path, row["sha"], 0) is True
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["weight"] == 0
    assert lines[1]["weight"] == 1
    assert sh.rewrite_weight(path, "missing", 0) is False


def test_decide_keep_and_drop() -> None:
    ok, reason = sh.decide_keep(PAGE, {"ok": True, "errors": []}, PROMPT)
    assert ok and reason == ""

    ok, reason = sh.decide_keep(PAGE, {"ok": False, "errors": ["x is not defined"]}, PROMPT)
    assert not ok
    assert "not defined" in reason

    bare = "<!DOCTYPE html><html><body><p>hi</p></body></html>"
    ok, reason = sh.decide_keep(bare, {"ok": True, "errors": []}, PROMPT)
    assert not ok and reason == "no script"

    ok, reason = sh.decide_keep("nope", {"ok": True, "errors": []}, PROMPT)
    assert not ok and reason == "not a full document"

    huge = PAGE.replace("</html>", ("x" * 50000) + "</html>")
    ok, reason = sh.decide_keep(huge, {"ok": True, "errors": []}, PROMPT)
    assert not ok and reason == "over token budget"


def test_parse_idea_lines_drops_seen_and_numbering() -> None:
    seen = {sh.normalize_prompt(PROMPT)}
    text = (
        "Here are 40 ideas:\n"
        "1. Write me a game of hangman.\n"
        "- Write me a game of darts.\n"
        "Write me a page that rolls a die\n"
    )
    got = sh.parse_idea_lines(text, seen)
    assert got == [
        "Write me a game of darts.",
        "Write me a page that rolls a die.",
    ]


def test_prompts_to_run_skips_done_and_counts_extras() -> None:
    seed = [PROMPT, "Write me a game of snake.", "Write me a game of pong."]
    done = {sh.normalize_prompt(PROMPT)}
    run, extra = sh.prompts_to_run(seed, done, 2)
    assert run == ["Write me a game of snake."]
    assert extra == 0

    run, extra = sh.prompts_to_run(seed, set(), 5)
    assert len(run) == 3
    assert extra == 2

    run, extra = sh.prompts_to_run(seed, set(), 0)
    assert len(run) == 3
    assert extra == -1

    # Same game, different opening: already saved, so it is not queued again.
    done = {sh.normalize_prompt("Make a game of hangman.")}
    run, extra = sh.prompts_to_run([PROMPT, "Write me a game of snake."], done, 5)
    assert run == ["Write me a game of snake."]
    assert extra == 3


def test_prompts_in_jsonl_reads_user_line(tmp_path: Path) -> None:
    path = tmp_path / "synth.jsonl"
    row = sh.pack_row(PROMPT, PAGE)
    path.write_text(json.dumps(row) + "\nnot json\n", encoding="utf-8")
    assert sh.prompts_in_jsonl(path) == {sh.normalize_prompt(PROMPT)}
    assert sh.prompts_in_jsonl(tmp_path / "missing.jsonl") == set()


def test_rolling_tok_s_is_tokens_over_the_window() -> None:
    stats = sh.Stats(12, 1000)
    now = time.time()
    stats.t0 = now - 10
    stats.add_tokens(200, now=now)
    assert stats.tok_s(now) == 20


def test_model_id_matches_basename() -> None:
    payload = {"data": [{"id": "Qwen3.8-27B-mxfp8"}]}
    assert sh.model_id_from_list(payload, "Qwen3.8-27B-mxfp8") == "Qwen3.8-27B-mxfp8"
    assert sh.model_id_from_list(payload, "/Users/me/Qwen3.8-27B-mxfp8") == "Qwen3.8-27B-mxfp8"
    assert sh.model_id_from_list({"data": []}, "Qwen3.8-27B-mxfp8") is None


def test_stream_delta_and_usage() -> None:
    assert sh.take_delta({"choices": [{"delta": {"content": "<html>"}}]}) == "<html>"
    assert sh.take_delta({"choices": []}) == ""
    assert sh.take_usage_completion({"usage": {"completion_tokens": 12}}) == 12
    assert sh.take_usage_completion({}) is None
