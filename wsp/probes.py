"""Failure-mode probes over agent trajectories.

The WSP loop needs a target. The paper found its target by reading failures:
a reasoning-loop direction in the MLP of layers 15 and 19. For a 27B agent the
analogue is: run the model as a coding agent, classify *where the episodes
break*, and only then look at weights.

Each probe reports a ``lever``: whether the failure looks addressable by a
weight edit, by harness/prompt changes, or by training. Probes that are
plausibly weight-addressable are the ones worth spending an edit on.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

DONE_WORDS = re.compile(r"\b(done|fixed|complete|completed|finished|resolved|all tests pass)\b", re.I)


def ngram_repeat_ratio(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are duplicates. ~0 for normal prose, ->1 for loops."""
    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    c = Counter(grams)
    return 1.0 - (len(c) / len(grams))


@dataclass
class Finding:
    probe: str
    count: int
    severity: str          # low | medium | high
    lever: str             # weights | harness | prompt | training
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _writes(steps) -> list[tuple[str, str]]:
    return [(s["tool_input"].get("path", ""), s["tool_input"].get("content", ""))
            for s in steps if s.get("tool") == "write_file"]


def detect(episode: dict) -> dict:
    """Run every probe against one episode. Returns a graded report."""
    steps = episode.get("steps", [])
    findings: list[Finding] = []

    # 1. reasoning loops / verbatim self-repetition
    ratios = [ngram_repeat_ratio(s.get("assistant", "")) for s in steps]
    loops = sum(1 for r in ratios if r > 0.25)
    if loops:
        findings.append(Finding("reasoning_loop", loops, "high", "weights",
                                "assistant text repeats itself; a direction is dominating generation"))

    # 2. hitting the context/token ceiling
    trunc = sum(1 for s in steps if s.get("finish_reason") == "length")
    if trunc:
        findings.append(Finding("token_limit", trunc, "high", "weights",
                                "generation ran to max_tokens; mirrors the Qwen3.5-0.8B loop failure"))

    # 3. tool-call protocol breakage
    parse = sum(1 for s in steps if s.get("parse_error"))
    if parse:
        sev = "high" if parse / max(1, len(steps)) > 0.15 else "medium"
        findings.append(Finding("tool_format", parse, sev, "weights",
                                "model emitted malformed tool calls; format compliance is a single behaviour"))

    # 4. submitting without running the tests
    ran_tests = any(s.get("tool") == "shell" and "pytest" in str(s.get("tool_input", {})) for s in steps)
    if episode.get("submitted") and not ran_tests:
        findings.append(Finding("no_verify", 1, "high", "harness",
                                "submitted without executing tests; verify-before-done is a policy, enforce it"))

    # 5. editing a file it never read
    read = {s["tool_input"].get("path", "") for s in steps if s.get("tool") == "read_file"}
    listed = " ".join(str(s.get("observation", "")) for s in steps if s.get("tool") == "list_dir")
    blind = [p for p, _ in _writes(steps) if p and p not in read and p not in listed]
    if blind:
        findings.append(Finding("edit_without_read", len(blind), "medium", "weights",
                                "wrote files it never read; impulsive edit tendency"))

    # 6. rewriting the same file over and over
    by_path = defaultdict(set)
    for p, content in _writes(steps):
        by_path[p].add(hash(content))
    thrash = {p: len(v) for p, v in by_path.items() if len(v) > 2}
    if thrash:
        findings.append(Finding("rewrite_thrash", len(thrash), "medium", "weights",
                                "same file rewritten 3+ times with different content; search without convergence"))

    # 7. identical shell command retried
    cmds = [str(s["tool_input"].get("cmd", "")) for s in steps if s.get("tool") == "shell"]
    rep = [c for c, n in Counter(cmds).items() if c and n >= 3]
    if rep:
        findings.append(Finding("repeat_command", len(rep), "low", "weights",
                                "same command retried; stuck-state recovery is weak"))

    # 8. declaring success on a red test run
    last_obs = next((str(s.get("observation", "")) for s in reversed(steps)
                     if s.get("tool") == "shell" and "pytest" in str(s.get("tool_input", {}))), "")
    failed_last = bool(re.search(r"\bfailed\b|\berror\b", last_obs, re.I)) and "passed" not in last_obs.lower()
    final = next((s.get("assistant", "") for s in reversed(steps) if s.get("tool") is None), "")
    if failed_last and DONE_WORDS.search(final):
        findings.append(Finding("false_success", 1, "high", "weights",
                                "claims completion while the last test run was red; optimism bias"))

    # 9. context bloat
    obs = sum(len(str(s.get("observation", ""))) for s in steps)
    if obs > 120_000:
        findings.append(Finding("context_bloat", 1, "medium", "harness",
                                "observation volume will force compaction; truncate tool output"))

    weight_findings = [f for f in findings if f.lever == "weights"]
    score = sum({"low": 1, "medium": 2, "high": 3}[f.severity] * min(f.count, 3) for f in weight_findings)

    return {
        "task_id": episode.get("task_id"),
        "resolved": episode.get("resolved"),
        "steps": len(steps),
        "loop_ratio_max": round(max(ratios) if ratios else 0.0, 3),
        "findings": [f.to_dict() for f in findings],
        "weight_lever_score": score,
    }


def rank(reports: list[dict]) -> dict:
    """Aggregate probe reports into a ranked list of candidate suppressors."""
    agg = defaultdict(lambda: {"episodes": 0, "count": 0, "lever": "", "note": ""})
    unresolved = [r for r in reports if not r.get("resolved")]
    for r in reports:
        for f in r["findings"]:
            a = agg[f["probe"]]
            a["episodes"] += 1
            a["count"] += f["count"]
            a["lever"] = f["lever"]
            a["note"] = f["note"]
    rows = [{"probe": k, **v, "prevalence": round(v["episodes"] / max(1, len(reports)), 3)} for k, v in agg.items()]
    rows.sort(key=lambda r: (-r["prevalence"], -r["count"]))
    return {
        "n_episodes": len(reports),
        "n_unresolved": len(unresolved),
        "ranked": rows,
        "weight_candidates": [r["probe"] for r in rows if r["lever"] == "weights"],
    }
