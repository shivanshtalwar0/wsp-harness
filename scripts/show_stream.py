"""Pretty-print `claude -p --output-format stream-json` from stdin: tool calls and text, one line each."""

import json
import sys


def short(x, n=160):
    s = x if isinstance(x, str) else json.dumps(x)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


for line in sys.stdin:
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = ev.get("type")
    if t == "assistant":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and block["text"].strip():
                print(f"\033[37m{block['text'].strip()}\033[0m\n", flush=True)
            elif block.get("type") == "tool_use":
                name = block["name"].replace("mcp__wsp-lab__", "")
                print(f"\033[36m→ {name}\033[0m {short(block.get('input', {}))}", flush=True)
    elif t == "user":
        for block in ev.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                c = block.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                print(f"\033[90m  ← {short(c or '', 220)}\033[0m", flush=True)
    elif t == "result":
        print(f"\n\033[33mdone: {ev.get('subtype')} · turns {ev.get('num_turns')} · ${ev.get('total_cost_usd', 0):.2f}\033[0m")
