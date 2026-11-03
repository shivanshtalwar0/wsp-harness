"""Offline stand-ins for a model endpoint, so the harness can be tested in CI."""

from __future__ import annotations

import json
from collections import deque


class ScriptedModel:
    """Replays a fixed list of replies. Any extra call returns ``DEFAULT``."""

    DEFAULT = '<tool name="submit">{}</tool>'

    def __init__(self, replies: list[str], default: str | None = None):
        self.replies = deque(replies)
        self.default = default or self.DEFAULT
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, **kw) -> dict:
        self.usage["calls"] += 1
        text = self.replies.popleft() if self.replies else self.default
        return {"text": text, "finish_reason": "stop"}


def tool(name: str, **args) -> str:
    return f'<tool name="{name}">{json.dumps(args)}</tool>'


def read(path: str) -> str:
    return tool("read_file", path=path)


def write(path: str, content: str) -> str:
    return tool("write_file", path=path, content=content)


def shell(cmd: str) -> str:
    return tool("shell", cmd=cmd)


SUBMIT = tool("submit")


class LoopingModel(ScriptedModel):
    """Emits the same long reply forever: used to unit-test the loop probe."""

    def __init__(self, filler: str = "let me think about this again and again "):
        super().__init__(replies=[])
        self.filler = filler

    def chat(self, messages, **kw) -> dict:
        self.usage["calls"] += 1
        return {"text": self.filler * 200, "finish_reason": "length"}
