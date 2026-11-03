#!/usr/bin/env python
"""Compare two eval runs produced by `wsp eval`.

    python scripts/compare_runs.py runs/eval-base.json runs/eval-patched.json

Prints the resolution rate with a Wilson 95% interval, per-task deltas, step and
latency differences, and the probe rankings. The interval is the point: three
tasks cannot distinguish 2/3 from 3/3, and this script says so out loud rather
than letting a table imply otherwise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    sa, sb = a["summary"], b["summary"]

    def row(name: str, run: dict) -> str:
        s = run["summary"]
        lo, hi = wilson(s["resolved"], s["n"])
        eps = s["episodes"]
        steps = sum(e["n_steps"] for e in eps) / len(eps)
        secs = sum(e.get("seconds", 0) for e in eps) / len(eps)
        return (f"{name:<22} {s['resolved']}/{s['n']}  "
                f"[{lo:.2f}, {hi:.2f}]   steps {steps:4.1f}   {secs:6.1f}s/task")

    print(row("run", a) if False else "")
    print(row("A", a))
    print(row("B", b))

    ta = {e["task_id"]: e for e in sa["episodes"]}
    tb = {e["task_id"]: e for e in sb["episodes"]}
    print("\nper task")
    for tid in sorted(set(ta) | set(tb)):
        ea, eb = ta.get(tid), tb.get(tid)
        fa = "PASS" if ea and ea["resolved"] else "fail"
        fb = "PASS" if eb and eb["resolved"] else "fail"
        dsteps = (eb["n_steps"] - ea["n_steps"]) if ea and eb else 0
        print(f"  {tid:<26} A={fa:<4} B={fb:<4} steps {dsteps:+d}")

    pa = {r["probe"]: r["prevalence"] for r in sa["probes"]["ranked"]}
    pb = {r["probe"]: r["prevalence"] for r in sb["probes"]["ranked"]}
    print("\nprobes (prevalence)")
    for k in sorted(set(pa) | set(pb)):
        print(f"  {k:<16} A={pa.get(k, 0):.2f}  B={pb.get(k, 0):.2f}")
    if not pa and not pb:
        print("  none fired in either condition")

    lo, hi = wilson(sa["resolved"], sa["n"])
    lo2, hi2 = wilson(sb["resolved"], sb["n"])
    overlap = not (hi2 < lo or hi < lo2)
    print()
    if overlap:
        print(f"NOT DISTINGUISHABLE at n={sa['n']}: the intervals overlap. "
              f"Run >=20 episodes per condition before claiming an effect.")
    else:
        print("intervals separate, but n is small; replicate before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
