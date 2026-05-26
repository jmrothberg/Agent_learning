"""Tests for static /help <topic> pages (tui_help.py)."""

from __future__ import annotations

import tui_help


def test_normalize_feedback_aliases():
    assert tui_help.normalize_help_topic("feedback") == "feedback"
    assert tui_help.normalize_help_topic("autonomous-playtest") == "feedback"
    assert tui_help.normalize_help_topic("VC") == "vlm-critique"
    assert tui_help.normalize_help_topic("vlm_critique") == "vlm-critique"
    assert tui_help.normalize_help_topic("memory") == "playbook"
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
    assert "/help feedback" in text
    assert "/help gpu" in text
    assert "/help getting-started" in text


def test_feedback_topic_mentions_vlm_critique_distinction():
    lines = tui_help.help_topic_lines("feedback") or []
    text = "\n".join(lines).lower()
    assert "vlm-critique" in text or "/vlm-critique" in text
    assert "autonomous" in text or "playtest" in text
    assert "harness" in text or "probe" in text


def test_vlm_critique_topic_mentions_feedback_distinction():
    lines = tui_help.help_topic_lines("vlm-critique") or []
    text = "\n".join(lines).lower()
    assert "/feedback" in text
    assert "screenshot" in text or "visual" in text


def test_feedback_flows_lists_four():
    lines = tui_help.help_topic_lines("feedback-flows") or []
    text = "\n".join(lines)
    assert "1." in text or "[b]1." in text
    assert "playbook" in text.lower()
    assert "rawfeedback" in text.lower()


def test_help_index_lists_topics_command():
    index = "\n".join(tui_help.help_topics_index_lines())
    assert "/help topics" in index
    assert "/help feedback-flows" in index


def test_format_unknown_topic_message():
    msgs = tui_help.format_unknown_topic_message("nope")
    assert any("nope" in m for m in msgs)
    assert any("feedback" in m for m in msgs)


def test_cmd_help_topic_via_app():
    from chat import CodingBoxApp

    app = CodingBoxApp()
    rendered: list[str] = []
    app._log = lambda *args, **kwargs: rendered.append(" ".join(str(a) for a in args))  # type: ignore[assignment]
    app._log_info = lambda msg: rendered.append(str(msg))  # type: ignore[method-assign]
    app._status_manual_body = None
    app._update_status = lambda: None  # type: ignore[method-assign]
    app._cmd_help("feedback")
    assert any("autonomous" in line.lower() for line in rendered)
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
