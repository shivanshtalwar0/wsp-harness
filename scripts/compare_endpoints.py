#!/usr/bin/env python
"""Is a weight edit behaviourally inert? Compare two endpoints on fixed prompts.

    python scripts/compare_endpoints.py \
        --a http://127.0.0.1:8101/v1 --ma <model-id> \
        --b http://127.0.0.1:8102/v1 --mb <model-id>

Sends the same prompts at temperature 0 to both and reports exact agreement,
first-token agreement, and normalised edit distance. A patch that removes one
direction from one of 64 layers often shows up here as "different logits,
identical text" -- which is the honest way to describe it, rather than
"no effect".
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.harness import OpenAICompat  # noqa: E402

PROMPTS = [
    ("code", "def fibonacci(n):"),
    ("code", "class LRUCache:\n    def __init__(self, capacity):"),
    ("prose", "The reason gradient descent works is"),
    ("math", "Compute 17 * 23 step by step."),
    ("instruct", "Reply with exactly the word: ok"),
]


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--ma", required=True)
    p.add_argument("--b", required=True); p.add_argument("--mb", required=True)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--out")
    a = p.parse_args()

    ca = OpenAICompat(a.a, a.ma, temperature=0.0, max_tokens=a.max_tokens)
    cb = OpenAICompat(a.b, a.mb, temperature=0.0, max_tokens=a.max_tokens)

    rows = []
    for kind, prompt in PROMPTS:
        ta = norm(ca.chat([{"role": "user", "content": prompt}])["text"])
        tb = norm(cb.chat([{"role": "user", "content": prompt}])["text"])
        sm = difflib.SequenceMatcher(None, ta, tb)
        rows.append({
            "kind": kind,
            "prompt": prompt[:60],
            "exact_match": ta == tb,
            "first_40_chars_match": ta[:40] == tb[:40],
            "text_similarity": round(sm.ratio(), 3),
            "a": ta[:160],
            "b": tb[:160],
        })
        flag = "same" if ta == tb else f"diff {rows[-1]['text_similarity']:.2f}"
        print(f"  {kind:<9} {flag:<12} {prompt[:44]!r}")

    n = len(rows)
    exact = sum(r["exact_match"] for r in rows)
    sim = sum(r["text_similarity"] for r in rows) / n
    print(f"\nexact agreement {exact}/{n}  mean similarity {sim:.3f}")
    if exact == n:
        print("inert on this prompt set: the edit did not change greedy output")
    elif sim > 0.9:
        print("near-inert: outputs differ slightly, same behaviour")
    else:
        print("behaviourally different: inspect the rows")

    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "exact": exact, "n": n, "mean_similarity": sim}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
