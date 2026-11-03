"""Patch IR: the unit of Weight-Space Programming.

A patch is an ordered list of typed, named operations on named tensors. It is
JSON, hashable, reviewable, and reversible. Nothing here touches a checkpoint;
that happens in ``wsp.weights.ModelArchive.apply``.

Two edit spaces exist because this checkpoint is 4-bit:

  space="float"  dequantize -> edit the float matrix -> requantize on the
                 existing grid. Expressive, but each edit is rounded to one
                 quantization step, so tiny edits can vanish.
  space="quant"  edit the per-(row, group) scales/biases directly. Exact, cheap,
                 and always expressible, but the edit is a per-group gain/offset
                 rather than a rank-1 direction.

The paper's headline edit (subtract one direction from 64 output columns,
0.016% of weights) is ``project_out`` in the float space.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- op


def _sel(value: Any, n: int, name: str) -> np.ndarray:
    if value is None:
        return np.arange(n)
    arr = np.asarray(value, dtype=int).ravel()
    if arr.size == 0:
        raise ValueError(f"{name}: empty selection")
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(f"{name}: indices out of range [0,{n})")
    return arr


@dataclass(frozen=True)
class Op:
    """One named weight edit."""

    op: str
    tensor: str
    args: dict = field(default_factory=dict)
    space: str = "float"

    KNOWN = {
        "project_out": "float",
        "add_rank1": "float",
        "col_gain": "float",
        "row_gain": "float",
        "zero_cols": "float",
        "group_gain": "quant",
        "group_shift": "quant",
    }

    def __post_init__(self) -> None:
        if self.op not in self.KNOWN:
            raise ValueError(f"unknown op {self.op!r}; known: {sorted(self.KNOWN)}")
        implied = self.KNOWN[self.op]
        if self.space != implied:
            raise ValueError(f"{self.op} lives in space={implied!r}, got {self.space!r}")

    # -- float-space application -------------------------------------------
    def apply(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        rows, cols = w.shape
        a = self.args

        if self.op == "project_out":
            d = np.asarray(a["direction"], dtype=np.float32).ravel()
            if d.size != rows:
                raise ValueError(f"direction has {d.size} entries, tensor has {rows} rows")
            norm = float(np.linalg.norm(d))
            if norm == 0:
                raise ValueError("direction is the zero vector")
            d = d / norm
            sel = _sel(a.get("cols"), cols, "cols")
            alpha = float(a.get("alpha", 1.0))

            # Optional row support: restricting the edit to a subset of rows and
            # renormalising there concentrates the same total change into fewer
            # entries, which is the only way to beat the 4-bit rounding floor.
            if a.get("rows") is not None:
                rows_sel = _sel(a["rows"], rows, "rows")
                d = np.where(np.isin(np.arange(rows), rows_sel), d, 0.0)
                sub = float(np.linalg.norm(d))
                if sub == 0:
                    raise ValueError("row support excludes every nonzero direction entry")
                d = d / sub

            with np.errstate(all="ignore"):          # BLAS leaves stale FP flags
                coef = d @ w[:, sel]                 # (k,) energy along d per column
            w = w.copy()
            w[:, sel] -= alpha * np.outer(d, coef)
            return w

        if self.op == "add_rank1":
            u = np.asarray(a["u"], dtype=np.float32).ravel()
            sel = _sel(a.get("cols"), cols, "cols")
            v = np.asarray(a["v"], dtype=np.float32).ravel()
            if u.size != rows or v.size != sel.size:
                raise ValueError(f"rank-1 shapes: u={u.size} (want {rows}), v={v.size} (want {sel.size})")
            w = w.copy()
            w[:, sel] += float(a.get("alpha", 1.0)) * np.outer(u, v)
            return w

        if self.op == "col_gain":
            sel = _sel(a.get("cols"), cols, "cols")
            w = w.copy()
            w[:, sel] *= float(a["factor"])
            return w

        if self.op == "row_gain":
            sel = _sel(a.get("rows"), rows, "rows")
            w = w.copy()
            w[sel, :] *= float(a["factor"])
            return w

        if self.op == "zero_cols":
            sel = _sel(a.get("cols"), cols, "cols")
            w = w.copy()
            w[:, sel] = 0.0
            return w

        raise ValueError(f"{self.op} is not a float-space op")

    # -- quant-space application -------------------------------------------
    def apply_quant(self, scales: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, groups = scales.shape
        a = self.args
        rows_sel = _sel(a.get("rows"), rows, "rows")
        g_sel = _sel(a.get("groups"), groups, "groups")
        s, b = scales.copy(), biases.copy()

        if self.op == "group_gain":
            f = float(a["factor"])
            s[np.ix_(rows_sel, g_sel)] *= f
            if a.get("mode", "all") == "all":
                b[np.ix_(rows_sel, g_sel)] *= f
        elif self.op == "group_shift":
            b[np.ix_(rows_sel, g_sel)] += float(a["delta"])
        else:
            raise ValueError(f"{self.op} is not a quant-space op")
        return s, b

    def to_json(self) -> dict:
        return {"op": self.op, "tensor": self.tensor, "space": self.space, "args": self.args}


# ------------------------------------------------------------------------ patch


@dataclass
class Patch:
    id: str
    hypothesis: str
    ops: list[Op]
    evidence: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if not self.ops:
            raise ValueError("a patch needs at least one op")
        if not self.id:
            self.id = self.digest()

    def digest(self) -> str:
        blob = json.dumps([o.to_json() for o in self.ops], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def by_tensor(self, archive) -> dict[str, list[Op]]:
        known = {archive.base_of(k) for k in archive.keys()}
        out: dict[str, list[Op]] = {}
        for o in self.ops:
            if o.tensor not in known:
                raise KeyError(f"tensor not in checkpoint: {o.tensor}")
            if o.space == "quant" and not archive.is_quantized(o.tensor):
                raise ValueError(f"{o.tensor} is not quantized; cannot use space=quant")
            out.setdefault(o.tensor, []).append(o)
        return out

    def touched_numel(self, archive) -> int:
        bases = set(self.by_tensor(archive))
        return sum(archive.numel(b) for b in bases)

    def describe(self, archive=None) -> str:
        lines = [f"patch {self.id}: {self.hypothesis}", f"  {len(self.ops)} op(s)"]
        for o in self.ops:
            lines.append(f"  - {o.op}(space={o.space}) on {o.tensor} :: {json.dumps(o.args)[:140]}")
        if archive is not None:
            n = self.touched_numel(archive)
            lines.append(f"  touches {n:,} weights' worth of tensors ({n / max(1, sum(archive.numel(b) for b in {archive.base_of(k) for k in archive.keys()})):.4%} of the model)")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "created": self.created,
            "evidence": self.evidence,
            "ops": [o.to_json() for o in self.ops],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Patch":
        ops = [Op(op=d["op"], tensor=d["tensor"], args=d.get("args", {}), space=d.get("space", "float")) for d in data["ops"]]
        return cls(id=data.get("id", ""), hypothesis=data.get("hypothesis", ""), ops=ops, evidence=data.get("evidence", {}), created=data.get("created", ""))


# ---------------------------------------------------------------------- parsing

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerant of fences/prose)."""
    for candidate in [text, *(m.group(1) for m in _FENCE.finditer(text))]:
        candidate = candidate.strip()
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no parsable JSON object in model reply")


def parse_patch(text: str) -> Patch:
    return Patch.from_json(extract_json(text))


SCHEMA_HELP = """\
Reply with exactly one JSON object and nothing else:

{
  "hypothesis": "one sentence: which behaviour in the weights you are removing and why",
  "ops": [
    {"op": "project_out", "tensor": "<full tensor base name>", "space": "float",
     "args": {"direction": [<one float per ROW of the tensor>],
              "cols": [<column indices, or null for all>], "alpha": 1.0}}
  ]
}

ops allowed:
  float space: project_out, add_rank1, col_gain, row_gain, zero_cols
  quant space: group_gain {rows,groups,factor,mode}, group_shift {rows,groups,delta}
Keep it small: prefer < 200 modified columns per tensor and alpha <= 1.0.
"""
