"""Per-edit validation: is the current weight state better or worse than base?

Runs automatically whenever the weights change (the operator's instrument, not a
tool the analyst spends passes on). Two signals, both *paired* against the base
model on the same fixed items, so small effects are not drowned by item noise:

  * answer NLL (always): mean per-token loss of each reference solution given its
    problem. One teacher-forced pass, seconds. Lower is better.
  * generation accuracy (optional, ``val_gen_n`` > 0): greedy answers on the first
    ``val_gen_n`` items, graded. Slower; reported as paired wins/losses + sign test.

The split must be disjoint from the final evaluation (default ``math:val`` = MATH
test minus MATH-500). By default the analyst cannot see these numbers, so its edits
cannot be tuned against them; flip ``val_visible`` to give it the signal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from . import forward as fw
from .data import ProblemSet


@dataclass
class ValConfig:
    spec: str = "math:val"
    n: int = 128
    gen_n: int = 0
    max_new_tokens: int = 1024
    seed: int = 0


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test on paired wins/losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(cur: list[float], base: list[float]) -> dict:
    d = [c - b for c, b in zip(cur, base)]
    n = len(d)
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("inf")
    return {"mean": mean, "ci": (mean - half, mean + half),
            "improved": sum(x < 0 for x in d), "worsened": sum(x > 0 for x in d)}


class Validator:
    def __init__(self, lm, cfg: ValConfig, thinking: bool = False, batch_size: int = 16, gen_fn=None):
        self.lm = lm
        self.gen_fn = gen_fn
        self.cfg = cfg
        self.thinking = thinking
        self.batch_size = batch_size
        pool = [p for p in ProblemSet(cfg.spec).items if p.solution]
        if not pool:
            raise ValueError(f"validation source {cfg.spec!r} has no reference solutions")
        ps = ProblemSet(cfg.spec)
        ps._items = pool
        self.items = ps.sample(min(cfg.n, len(pool)), cfg.seed)
        self.base: dict | None = None
        self.history: list[dict] = []
        self._cache: dict = {}

    def measure(self) -> dict:
        nll = fw.answer_nll(self.lm, [p.prompt for p in self.items], [p.solution for p in self.items],
                            thinking=self.thinking, batch_size=self.batch_size)
        out = {"nll": nll}
        if self.cfg.gen_n > 0:
            gi = self.items[: self.cfg.gen_n]
            prompts = [p.prompt for p in gi]
            if self.gen_fn is not None:
                rows = self.gen_fn(prompts, max_new_tokens=self.cfg.max_new_tokens, seed=self.cfg.seed)
            else:
                rows = fw.generate(self.lm, prompts, max_new_tokens=self.cfg.max_new_tokens,
                                   thinking=self.thinking, batch_size=self.batch_size, seed=self.cfg.seed)
            out["correct"] = [p.check(r["completion"]) for r, p in zip(rows, gi)]
            out["hit_limit"] = [r["hit_limit"] for r in rows]
        return out

    def set_base(self) -> dict:
        self.base = self.measure()
        rec = self._record(self.base, wver=0, tensors=0, secs=0.0)
        rec["direction"] = "base"
        self.history.append(rec)
        return rec

    def run(self, wver: int, fingerprint: tuple, tensors: int) -> dict:
        assert self.base is not None, "set_base() first"
        t0 = time.time()
        if fingerprint == ():
            m, cached = self.base, True
        elif fingerprint in self._cache:
            m, cached = self._cache[fingerprint], True
        else:
            m, cached = self.measure(), False
            self._cache[fingerprint] = m
        rec = self._record(m, wver, tensors, time.time() - t0)
        rec["cached"] = cached
        self.history.append(rec)
        return rec

    def _record(self, m: dict, wver: int, tensors: int, secs: float) -> dict:
        base = self.base or m
        p = paired(m["nll"], base["nll"])
        lo, hi = p["ci"]
        direction = "better" if hi < 0 else "worse" if lo > 0 else "inconclusive"
        rec = {
            "wver": wver, "t": time.strftime("%H:%M:%S"), "tensors_changed": tensors,
            "n": len(m["nll"]), "nll": round(sum(m["nll"]) / len(m["nll"]), 5),
            "d_nll": round(p["mean"], 5), "d_nll_ci95": [round(lo, 5), round(hi, 5)],
            "items_improved": p["improved"], "items_worsened": p["worsened"],
            "direction": direction, "secs": round(secs, 2),
        }
        if "correct" in m:
            c, b = m["correct"], base["correct"]
            wins = sum(x and not y for x, y in zip(c, b))
            losses = sum(y and not x for x, y in zip(c, b))
            rec.update({
                "gen_n": len(c), "acc": round(sum(c) / len(c), 4),
                "d_acc": round((sum(c) - sum(b)) / len(c), 4),
                "gen_wins": wins, "gen_losses": losses, "gen_sign_p": round(sign_test_p(wins, losses), 4),
                "hit_limit_rate": round(sum(m["hit_limit"]) / len(c), 4),
            })
        return rec
