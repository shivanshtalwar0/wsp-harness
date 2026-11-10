"""Weight edits against a pristine base: diff, checkpoints, Patch-IR ops, controls.

The base copy is what makes every edit attributable and reversible. Whatever the
analyst does in the sandbox (``W[name][:, cols] -= ...``) shows up in ``diff()``,
and ``restore()`` puts any recorded state back exactly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..patch import Op, Patch
from .model import LoadedModel

FLOAT_OPS = {k for k, v in Op.KNOWN.items() if v == "float"}


class WeightLedger:
    def __init__(self, lm: LoadedModel, base_device: str | None = None, root: Path | None = None):
        self.lm = lm
        self.params = lm.params()
        dev = base_device or lm.device
        self.base = {n: p.data.detach().to(dev, copy=True) for n, p in self.params.items()}
        self.checkpoints: dict[str, dict] = {}
        self.root = root
        self.patches: list[dict] = []

    # ------------------------------------------------------------------ diff
    def changed(self) -> list[str]:
        out = []
        for n, p in self.params.items():
            b = self.base[n]
            if not torch.equal(p.data, b.to(p.device)):
                out.append(n)
        return out

    @torch.no_grad()
    def tensor_delta_stats(self, name: str, max_list: int = 64) -> dict:
        p = self.params[name].data
        b = self.base[name].to(p.device)
        d = p.float() - b.float()
        nz = d != 0
        stats = {
            "tensor": name,
            "shape": list(p.shape),
            "elements_changed": int(nz.sum()),
            "fraction_changed": round(float(nz.float().mean()), 6),
            "delta_fro": round(float(d.norm()), 6),
            "relative_fro": round(float(d.norm() / b.float().norm().clamp(min=1e-12)), 6),
        }
        if d.dim() == 2:
            cols = torch.nonzero(nz.any(0)).flatten()
            rows = torch.nonzero(nz.any(1)).flatten()
            stats["cols_changed"] = int(cols.numel())
            stats["rows_changed"] = int(rows.numel())
            if cols.numel() <= max_list:
                stats["cols"] = cols.tolist()
            if rows.numel() <= max_list:
                stats["rows"] = rows.tolist()
            # effective rank of the edit (singular values of the touched submatrix)
            sub = d[rows][:, cols] if rows.numel() and cols.numel() else None
            if sub is not None and sub.numel() <= 64_000_000:
                s = torch.linalg.svdvals(sub.cpu())
                energy = (s**2).cumsum(0) / (s**2).sum().clamp(min=1e-30)
                stats["edit_rank_99"] = int((energy < 0.99).sum()) + 1
                stats["top_singular_values"] = [round(float(x), 6) for x in s[:4]]
        return stats

    def diff(self) -> dict:
        names = self.changed()
        total = sum(p.numel() for p in self.params.values())
        tensors = [self.tensor_delta_stats(n) for n in names]
        changed = sum(t["elements_changed"] for t in tensors)
        return {
            "tensors_changed": len(names),
            "elements_changed": changed,
            "fraction_of_model": changed / total if total else 0.0,
            "tensors": tensors,
        }

    @torch.no_grad()
    def column_delta_norms(self, name: str, axis: int = 0) -> torch.Tensor:
        """||ΔW|| per column (axis=0 reduces rows) or per row (axis=1)."""
        p = self.params[name].data
        d = p.float() - self.base[name].to(p.device).float()
        return d.norm(dim=axis).cpu()

    # ----------------------------------------------------------- checkpoints
    def checkpoint(self, tag: str, note: str = "") -> dict:
        names = self.changed()
        state = {n: self.params[n].data.detach().to("cpu", copy=True) for n in names}
        self.checkpoints[tag] = {"state": state, "note": note, "t": time.time(), "tensors": names}
        if self.root is not None:
            from safetensors.torch import save_file

            d = self.root / "checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            if state:
                save_file({k: v.contiguous() for k, v in state.items()}, str(d / f"{tag}.safetensors"),
                          metadata={"note": note})
        return {"tag": tag, "tensors": names, "note": note}

    @torch.no_grad()
    def reset(self) -> list[str]:
        names = self.changed()
        for n in names:
            self.params[n].data.copy_(self.base[n].to(self.params[n].device))
        return names

    @torch.no_grad()
    def restore(self, tag: str) -> dict:
        if tag == "base":
            return {"tag": "base", "reset": self.reset()}
        if tag not in self.checkpoints:
            raise KeyError(f"no checkpoint {tag!r}; have {sorted(self.checkpoints)}")
        self.reset()
        for n, t in self.checkpoints[tag]["state"].items():
            self.params[n].data.copy_(t.to(self.params[n].device))
        return {"tag": tag, "tensors": list(self.checkpoints[tag]["state"])}

    def list_checkpoints(self) -> list[dict]:
        return [
            {"tag": k, "tensors": v["tensors"], "note": v["note"], "t": v["t"]}
            for k, v in self.checkpoints.items()
        ]

    # --------------------------------------------------------------- patches
    def _resolve(self, name: str) -> str:
        if name in self.params:
            return name
        if name + ".weight" in self.params:
            return name + ".weight"
        raise KeyError(f"no tensor {name!r}")

    @torch.no_grad()
    def apply_patch(self, patch: dict, control: str | None = None, seed: int = 0) -> dict:
        """Apply float-space Patch-IR ops. ``control='random'`` swaps directions for random ones.

        Reports fidelity per op: how much of the intended float32 delta survived
        storage in the model's dtype (1.0 = exact; bf16 can round tiny edits away).
        """
        ops_in = patch.get("ops") or []
        ops: list[Op] = []
        rng = np.random.default_rng(seed)
        for o in ops_in:
            space = o.get("space", "float")
            if o["op"] not in FLOAT_OPS:
                raise ValueError(f"{o['op']} is a 4-bit (quant-space) op; this model is not quantized")
            args = dict(o.get("args", {}))
            if control == "random":
                args = _randomize(o["op"], args, rng)
            ops.append(Op(op=o["op"], tensor=self._resolve(o["tensor"]), args=args, space=space))
        p = Patch(id=patch.get("id", ""), hypothesis=patch.get("hypothesis", ""), ops=ops,
                  evidence=patch.get("evidence", {}))

        reports = []
        for op in p.ops:
            param = self.params[op.tensor]
            if param.dim() != 2:
                raise ValueError(f"{op.tensor} is not a matrix")
            w = param.data.detach().float().cpu().clone().numpy()   # copy: never alias the live weight
            target = op.apply(w)
            intended = target - w
            param.data.copy_(torch.from_numpy(target).to(param.device, param.dtype))
            achieved = param.data.float().cpu().numpy() - w
            denom = float((intended**2).sum())
            fidelity = float((achieved * intended).sum() / denom) if denom > 0 else 1.0
            reports.append(
                {
                    "op": op.op,
                    "tensor": op.tensor,
                    "intended_fro": round(float(np.sqrt(denom)), 6),
                    "achieved_fro": round(float(np.linalg.norm(achieved)), 6),
                    "fidelity": round(fidelity, 4),
                    "elements_changed": int((achieved != 0).sum()),
                }
            )
        rec = {"id": p.id, "hypothesis": p.hypothesis, "control": control, "ops": reports,
               "patch": {"id": p.id, "hypothesis": p.hypothesis, "ops": [o.to_json() for o in p.ops]}}
        self.patches.append(rec)
        return {k: v for k, v in rec.items() if k != "patch"}


def _randomize(op: str, args: dict, rng: np.random.Generator) -> dict:
    """Same op, same columns, same alpha, random direction (control B in docs/02)."""
    a = dict(args)
    if op == "project_out":
        d = np.asarray(a["direction"], dtype=np.float32)
        r = rng.standard_normal(d.shape).astype(np.float32)
        a["direction"] = (r / np.linalg.norm(r)).tolist()
    elif op == "add_rank1":
        u = np.asarray(a["u"], dtype=np.float32)
        r = rng.standard_normal(u.shape).astype(np.float32)
        a["u"] = (r / np.linalg.norm(r) * np.linalg.norm(u)).tolist()
    elif op in ("col_gain", "zero_cols") and a.get("cols") is not None:
        raise ValueError(f"random control for {op} needs a column count; pass explicit random cols")
    return a
