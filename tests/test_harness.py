import json
from pathlib import Path

import pytest

from evals import harness, mock

TASKS = Path(__file__).resolve().parent.parent / "evals" / "tasks"
FIXED = '''def paginate(items, page, per_page=10):
    """Return the items on ``page`` (0-indexed), ``per_page`` of them."""
    start = page * per_page
    return items[start : start + per_page]
'''


def test_parse_reply():
    name, args = harness.parse_reply('<tool name="read_file">{"path": "a.py"}</tool>')
    assert name == "read_file" and args == {"path": "a.py"}


def test_parse_reply_plain_text():
    assert harness.parse_reply("I think the bug is in the slice.") is None


def test_parse_reply_bad_json():
    with pytest.raises(harness.ParseError):
        harness.parse_reply('<tool name="read_file">{oops}</tool>')


def test_parse_reply_unknown_tool():
    with pytest.raises(harness.ParseError):
        harness.parse_reply('<tool name="rm_rf">{}</tool>')


def test_path_escape_blocked(tmp_path):
    with pytest.raises(harness.ParseError):
        harness._safe_path(tmp_path, "../../etc/passwd")


@pytest.mark.slow
def test_agent_solves_task(tmp_path):
    task = harness.Task.load(TASKS / "01-bugfix-off-by-one")
    client = mock.ScriptedModel([
        mock.read("textkit/paginate.py"),
        mock.write("textkit/paginate.py", FIXED),
        mock.shell("python -m pytest -q"),
        mock.SUBMIT,
    ])
    ep = harness.run_task(task, client, tmp_path)
    assert ep["resolved"] is True
    assert ep["submitted"] is True
    assert ep["steps"][1]["tool"] == "write_file"


def test_agent_that_submits_immediately_fails(tmp_path):
    task = harness.Task.load(TASKS / "01-bugfix-off-by-one")
    ep = harness.run_task(task, mock.ScriptedModel([mock.SUBMIT]), tmp_path)
    assert ep["resolved"] is False


def test_broken_tool_call_is_reported_not_fatal(tmp_path):
    task = harness.Task.load(TASKS / "01-bugfix-off-by-one")
    ep = harness.run_task(task, mock.ScriptedModel(['<tool name="read_file">{bad}</tool>', mock.SUBMIT]), tmp_path)
    assert ep["steps"][0].get("parse_error") is True
    assert ep["resolved"] is False
