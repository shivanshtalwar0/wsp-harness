"""A minimal coding-agent harness: text tool protocol, sandboxed workspace, pytest grading.

Deliberately boring. The eval only has to be *stable enough* that a weight edit
moving the score by 5 points means something. Everything is a plain dict so a
trajectory can be replayed by ``wsp.probes`` offline.

Works against any OpenAI-compatible /v1/chat/completions endpoint:
LM Studio (Qwen3.8-27B), llama.cpp server, OpenRouter, vLLM...
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

TOOLS = {
    "read_file": "read_file(path) -> file contents",
    "write_file": "write_file(path, content) -> overwrite a file (creates parent dirs)",
    "list_dir": "list_dir(path) -> entries, one per line",
    "shell": "shell(cmd) -> run a shell command in the repo root, 60s timeout",
    "submit": "submit() -> finish; hidden tests are run after this",
}

SYSTEM_PROMPT = """You are a software engineering agent working in a real repository.

You act only by emitting tool calls in this exact form, one per reply:

<tool name="read_file">{{"path": "src/app.py"}}</tool>

Available tools:
{tools}

Rules:
- Read a file before you change it.
- After changing code, run the tests with: <tool name="shell">{{"cmd": "python -m pytest -q"}}</tool>
- Only call submit() once the tests you can run are green.
- Keep reasoning short. Do not repeat the same command or the same edit.
"""


class ParseError(Exception):
    pass


TOOL_RE = re.compile(r"<tool\s+name=\"([a-z_]+)\"\s*>(.*?)</tool>", re.S)


def parse_reply(text: str) -> tuple[str, dict] | None:
    """Return (tool, args) or None if the reply is plain text."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    name, blob = m.group(1), m.group(2).strip()
    if name not in TOOLS:
        raise ParseError(f"unknown tool {name!r}")
    try:
        args = json.loads(blob) if blob else {}
    except json.JSONDecodeError as e:
        raise ParseError(f"tool arguments are not valid JSON: {e}") from e
    if not isinstance(args, dict):
        raise ParseError("tool arguments must be a JSON object")
    return name, args


# --------------------------------------------------------------------- clients


class OpenAICompat:
    """POST /v1/chat/completions. No SDK, no retries beyond a simple backoff."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.6, max_tokens: int = 8192, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat(self, messages: list[dict], **kw) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": kw.get("temperature", self.temperature),
            "max_tokens": kw.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise RuntimeError(f"endpoint failed: {e}") from e
                time.sleep(2 ** attempt)
        msg = payload["choices"][0]
        u = payload.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        return {"text": msg["message"].get("content") or "",
                "finish_reason": msg.get("finish_reason", "stop")}


# ----------------------------------------------------------------------- tasks


@dataclass
class Task:
    id: str
    title: str
    prompt: str
    root: Path
    repo: Path
    hidden: Path | None
    test_cmd: str
    max_steps: int = 25

    @classmethod
    def load(cls, task_dir: str | Path) -> "Task":
        d = Path(task_dir).expanduser().resolve()
        spec = yaml.safe_load((d / "task.yaml").read_text())
        hidden = d / spec.get("hidden_tests", "hidden")
        return cls(
            id=spec["id"], title=spec.get("title", spec["id"]), prompt=spec["prompt"].strip(),
            root=d, repo=d / spec.get("repo", "repo"),
            hidden=hidden if hidden.exists() else None,
            test_cmd=spec.get("test_cmd", "python -m pytest -q"),
            max_steps=int(spec.get("max_steps", 25)),
        )


def discover(tasks_dir: str | Path = None) -> list[Task]:
    base = Path(tasks_dir).expanduser() if tasks_dir else Path(__file__).parent / "tasks"
    return [Task.load(p) for p in sorted(base.iterdir()) if (p / "task.yaml").exists()]


def _run(cmd: str, cwd: Path, timeout: int = 60) -> tuple[int, str]:
    # Put the running interpreter first so `python -m pytest` is the same env.
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")])
    env = {"PATH": path, "HOME": str(cwd), "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"


def _safe_path(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ParseError(f"path escapes the workspace: {rel}")
    return p


def _exec_tool(name: str, args: dict, ws: Path) -> str:
    if name == "read_file":
        p = _safe_path(ws, args.get("path", ""))
        if not p.exists():
            return f"[no such file: {args.get('path')}]"
        return p.read_text(errors="replace")[:20000]
    if name == "write_file":
        p = _safe_path(ws, args.get("path", ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.get("content", ""))
        return f"[wrote {len(args.get('content', ''))} bytes to {args.get('path')}]"
    if name == "list_dir":
        p = _safe_path(ws, args.get("path", ".") or ".")
        if not p.is_dir():
            return f"[not a directory: {args.get('path')}]"
        return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if not x.name.startswith(".")))
    if name == "shell":
        rc, out = _run(args.get("cmd", ""), ws, timeout=60)
        return f"[exit {rc}]\n{out}"[-8000:]
    if name == "submit":
        return "[submitted]"
    raise ParseError(f"unknown tool {name!r}")


def grade(task: Task, ws: Path) -> tuple[bool, str]:
    if task.hidden:
        for f in task.hidden.rglob("*"):
            if f.is_file():
                dest = ws / f.relative_to(task.hidden)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dest)
    rc, out = _run(task.test_cmd, ws, timeout=180)
    return rc == 0, out[-4000:]


# ----------------------------------------------------------------------- loops


def run_task(task: Task, client, workdir: Path, max_steps: int | None = None,
             keep: bool = False) -> dict:
    ws = Path(tempfile.mkdtemp(prefix=f"{task.id}-", dir=str(workdir)))
    shutil.copytree(task.repo, ws, dirs_exist_ok=True)
    limit = max_steps or task.max_steps
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools="\n".join(f"- {v}" for v in TOOLS.values()))},
        {"role": "user", "content": task.prompt},
    ]
    steps: list[dict] = []
    submitted = False

    for i in range(limit):
        try:
            reply = client.chat(messages)
        except Exception as e:  # endpoint died; record and stop
            steps.append({"index": i, "assistant": f"[client error] {e}", "tool": None,
                          "tool_input": None, "observation": None, "finish_reason": "error"})
            break

        text = reply["text"]
        step = {"index": i, "assistant": text, "tool": None, "tool_input": None,
                "observation": None, "finish_reason": reply.get("finish_reason", "stop")}
        messages.append({"role": "assistant", "content": text})

        try:
            call = parse_reply(text)
        except ParseError as e:
            step["parse_error"] = True
            step["observation"] = f"[tool call rejected: {e}]"
            steps.append(step)
            messages.append({"role": "user", "content": step["observation"]})
            continue

        if call is None:
            steps.append(step)
            messages.append({"role": "user", "content": "Emit a tool call, or call submit() when finished."})
            continue

        name, args = call
        step["tool"], step["tool_input"] = name, args
        obs = _exec_tool(name, args, ws)
        step["observation"] = obs
        steps.append(step)
        if name == "submit":
            submitted = True
            break
        messages.append({"role": "user", "content": obs})

    resolved, test_output = grade(task, ws)
    episode = {
        "task_id": task.id, "resolved": resolved, "submitted": submitted,
        "steps": steps, "test_output": test_output,
        "workspace": str(ws) if keep else None,
        "n_steps": len(steps),
    }
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    return episode


def run_suite(tasks: list[Task], client, out_path: str | Path, max_steps: int | None = None,
              workdir: str | Path | None = None) -> dict:
    from .probes import detect, rank

    workdir = Path(workdir or tempfile.mkdtemp(prefix="wsp-runs-"))
    workdir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for t in tasks:
        t0 = time.time()
        ep = run_task(t, client, workdir, max_steps=max_steps)
        ep["seconds"] = round(time.time() - t0, 1)
        episodes.append(ep)
        print(f"  {t.id:<28} {'PASS' if ep['resolved'] else 'fail'}  "
              f"{ep['n_steps']:>3} steps  {ep['seconds']:>6.1f}s", file=sys.stderr)

    reports = [detect(e) for e in episodes]
    summary = {
        "n": len(episodes),
        "resolved": sum(e["resolved"] for e in episodes),
        "rate": round(sum(e["resolved"] for e in episodes) / max(1, len(episodes)), 3),
        "probes": rank(reports),
        "episodes": [{k: v for k, v in e.items() if k != "steps"} for e in episodes],
    }
    summary["usage"] = getattr(client, "usage", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
    return summary
