"""Tests for static /help <topic> pages (tui_help.py)."""

from __future__ import annotations

import tui_help


def test_normalize_critique_aliases():
    assert tui_help.normalize_help_topic("critique") == "critique"
    assert tui_help.normalize_help_topic("playtest") == "critique"
    assert tui_help.normalize_help_topic("feedback") == "critique"
    assert tui_help.normalize_help_topic("autonomous-playtest") == "critique"
    assert tui_help.normalize_help_topic("VC") == "vlm-critique"
    assert tui_help.normalize_help_topic("vlm_critique") == "vlm-critique"
    assert tui_help.normalize_help_topic("memory") == "playbook"
    assert tui_help.normalize_help_topic("memory-map") == "memory-map"
    assert tui_help.normalize_help_topic("rewind") == "revert"
    assert tui_help.normalize_help_topic("topics") == "topics"
    assert tui_help.normalize_help_topic("four-gpu") == "gpu"


def test_unknown_topic_returns_none():
    assert tui_help.normalize_help_topic("not-a-topic") is None
    assert tui_help.normalize_help_topic("") is None


def test_all_topics_have_lines():
    for topic_id in tui_help.list_help_topics():
        lines = tui_help.help_topic_lines(topic_id)
        assert lines is not None and len(lines) >= 3, topic_id


def test_topics_page_lists_other_topics():
    lines = tui_help.help_topic_lines("topics") or []
    text = "\n".join(lines)
    assert "/help critique" in text
    assert "/help gpu" in text
    assert "/help getting-started" in text


def test_critique_topic_mentions_vlm_critique_distinction():
    lines = tui_help.help_topic_lines("critique") or []
    text = "\n".join(lines).lower()
    assert "vlm-critique" in text or "/vlm-critique" in text
    assert "without" in text and "vision" in text
    assert "agent" in text or "coder" in text


def test_feedback_help_topic_is_alias_pointer():
    lines = tui_help.help_topic_lines("feedback") or []
    text = "\n".join(lines).lower()
    assert "alias" in text
    assert "critique" in text


def test_vlm_critique_topic_mentions_critique_distinction():
    lines = tui_help.help_topic_lines("vlm-critique") or []
    text = "\n".join(lines).lower()
    assert "/critique" in text
    assert "screen" in text or "vision" in text
    assert "model 2" in text


def test_feedback_flows_explains_two_reviews():
    lines = tui_help.help_topic_lines("feedback-flows") or []
    text = "\n".join(lines).lower()
    assert "/critique" in text
    assert "vlm-critique" in text
    assert "vision" in text


def test_help_index_lists_topics_command():
    index = "\n".join(tui_help.help_topics_index_lines())
    assert "/help topics" in index
    assert "/help feedback-flows" in index


def test_format_unknown_topic_message():
    msgs = tui_help.format_unknown_topic_message("nope")
    assert any("nope" in m for m in msgs)
    assert any("critique" in m for m in msgs)


def test_cmd_help_topic_via_app():
    from chat import CodingBoxApp

    app = CodingBoxApp()
    rendered: list[str] = []
    app._log = lambda *args, **kwargs: rendered.append(" ".join(str(a) for a in args))  # type: ignore[assignment]
    app._log_info = lambda msg: rendered.append(str(msg))  # type: ignore[method-assign]
    app._status_manual_body = None
    app._update_status = lambda: None  # type: ignore[method-assign]
    app._cmd_help("critique")
    assert any("critique" in line.lower() or "without" in line.lower() for line in rendered)
    assert app._status_manual_body is not None


def test_cmd_help_topics_via_app():
    from chat import CodingBoxApp

    app = CodingBoxApp()
    rendered: list[str] = []
    app._log = lambda *args, **kwargs: rendered.append(" ".join(str(a) for a in args))  # type: ignore[assignment]
    app._status_manual_body = None
    app._update_status = lambda: None  # type: ignore[method-assign]
    app._cmd_help("gpu")
    assert any("GPU 0" in line or "gpu 0" in line.lower() for line in rendered)


def test_cmd_help_overview_includes_topic_index():
    from chat import CodingBoxApp

    app = CodingBoxApp()
    rendered: list[str] = []
    app._log = lambda *args, **kwargs: rendered.append(" ".join(str(a) for a in args))  # type: ignore[assignment]
    app._status_manual_body = None
    app._update_status = lambda: None  # type: ignore[method-assign]
    app._cmd_help()
    assert any("/help topics" in line for line in rendered)
