"""Practice and held-out problem sets, and answer grading.

Sources:
  gsm8k:train / gsm8k:test   openai/gsm8k via `datasets` (answer after '####')
  math:train / math:test     EleutherAI/hendrycks_math, all 7 subjects (answer = last \\boxed{})
  math:val                   MATH test minus the MATH-500 problems (per-edit validation)
  math500                    HuggingFaceH4/MATH-500 (answer field)
  jsonl:<path>               one {"id", "prompt", "answer"} object per line
  toy                        12 offline arithmetic problems, for tests

The analyst only ever sees the practice split. The eval split is reachable from
the admin HTTP endpoint and the CLI, never from the MCP tools.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str
    solution: str = ""      # reference worked solution, when the source has one (validation uses it)

    @property
    def prompt(self) -> str:
        return f"{self.question}\n\n{INSTRUCTION}"


def _toy() -> list[Problem]:
    rng = random.Random(7)
    out = []
    for i in range(12):
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        out.append(Problem(f"toy-{i}", f"What is {a} plus {b}?", str(a + b),
                           f"{a} plus {b} is {a + b}. The answer is \\boxed{{{a + b}}}."))
    return out


def load_source(spec: str) -> list[Problem]:
    if spec == "toy":
        return _toy()
    if spec.startswith("jsonl:"):
        rows = [json.loads(x) for x in Path(spec[6:]).read_text().splitlines() if x.strip()]
        return [Problem(str(r.get("id", i)), r["prompt"], str(r["answer"]), r.get("solution", ""))
                for i, r in enumerate(rows)]
    from datasets import load_dataset

    if spec.startswith("gsm8k"):
        split = spec.split(":")[1] if ":" in spec else "train"
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [
            Problem(f"gsm8k-{split}-{i}", r["question"], r["answer"].split("####")[-1].strip().replace(",", ""),
                    r["answer"].split("####")[0].strip()
                    + f" The answer is \\boxed{{{r['answer'].split('####')[-1].strip()}}}.")
            for i, r in enumerate(ds)
        ]
    if spec == "math:val":
        # Validation split: MATH test problems that are NOT in MATH-500, so per-edit validation
        # never touches the final scoring set.
        held = {_norm_q(p.question) for p in load_source("math500")}
        return [Problem(p.id.replace("math-test-", "math-val-"), p.question, p.answer, p.solution)
                for p in load_source("math:test") if _norm_q(p.question) not in held]
    if spec.startswith("math:"):
        # MATH train/test (Hendrycks). MATH-500 is drawn from the test split, so math:train is
        # the leak-free practice set for a math500 eval, as in the paper.
        split = spec.split(":")[1]
        subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                    "number_theory", "prealgebra", "precalculus"]
        out = []
        for subj in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
            for i, r in enumerate(ds):
                ans = _last_boxed(r["solution"])
                if ans is not None:
                    out.append(Problem(f"math-{split}-{subj}-{i}", r["problem"], ans, r["solution"]))
        return out
    if spec == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [Problem(r.get("unique_id", f"math500-{i}"), r["problem"], r["answer"]) for i, r in enumerate(ds)]
    raise ValueError(f"unknown problem source {spec!r}")


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


# -------------------------------------------------------------------- grading

_BOXED = re.compile(r"\\boxed\s*\{")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _last_boxed(text: str) -> str | None:
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i : j - 1] if depth == 0 else None


def extract_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    return s.rstrip(".")


def _as_float(s: str) -> float | None:
    try:
        if re.fullmatch(r"-?\d+/\d+", s):
            a, b = s.split("/")
            return float(a) / float(b)
        m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(completion: str, answer: str) -> bool:
    got = extract_answer(completion)
    if got is None:
        return False
    a, b = _normalize(got), _normalize(answer)
    if a == b:
        return True
    fa, fb = _as_float(a), _as_float(b)
    return fa is not None and fb is not None and abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(c - h, 4), round(c + h, 4))


class ProblemSet:
    def __init__(self, spec: str):
        self.spec = spec
        self._items: list[Problem] | None = None

    @property
    def items(self) -> list[Problem]:
        if self._items is None:
            self._items = load_source(self.spec)
        return self._items

    def __len__(self) -> int:
        return len(self.items)

    def by_id(self, ids: list[str]) -> list[Problem]:
        idx = {p.id: p for p in self.items}
        missing = [i for i in ids if i not in idx]
        if missing:
            raise KeyError(f"unknown problem ids: {missing[:5]}")
        return [idx[i] for i in ids]

    def sample(self, n: int, seed: int = 0, offset: int = 0) -> list[Problem]:
        order = list(range(len(self.items)))
        random.Random(seed).shuffle(order)
        return [self.items[i] for i in order[offset : offset + n]]
