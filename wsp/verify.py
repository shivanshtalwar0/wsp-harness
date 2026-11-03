"""Patch fidelity: did the edit actually land on a 4-bit grid?

``apply`` reports codes changed. That number is misleading in both directions:
a patch can flip 0.2% of a tensor's codes and still remove 85% of the direction
it targeted, because the energy along a dominant direction is concentrated in a
few columns that survive rounding easily.

What matters is the *residual along the direction the patch claimed to remove*,
measured after requantization:

    r0 = d @ W            before
    r1 = d @ (W + patch)  intended, in float
    r2 = d @ requantize(W + patch)   what the checkpoint actually holds

    removed_intended = 1 - ||r1|| / ||r0||
    removed_actual   = 1 - ||r2|| / ||r0||
    fidelity         = removed_actual / removed_intended

fidelity ~1 means rounding did not touch the edit; ~0 means the 4-bit grid ate
it and the checkpoint is effectively unchanged. Check this *before* spending
minutes writing a 16 GB copy.
"""

from __future__ import annotations

import numpy as np

from . import quant as Q


def _effective_direction(op, rows: int) -> np.ndarray:
    d = np.asarray(op.args["direction"], dtype=np.float32).ravel()
    d = d / np.linalg.norm(d)
    if op.args.get("rows") is not None:
        sel = np.asarray(op.args["rows"], dtype=int)
        keep = np.zeros(rows, dtype=np.float32)
        keep[sel] = d[sel]
        d = keep / np.linalg.norm(keep)
    return d


def _requantized(ref, w_new: np.ndarray) -> np.ndarray:
    packed = ref.requantize(np.asarray(w_new, dtype=np.float32))
    return Q.dequantize(
        packed,
        np.asarray(ref.scales, dtype=np.float32),
        np.asarray(ref.biases, dtype=np.float32),
        bits=ref.bits,
        group_size=ref.group_size,
        in_features=ref.shape[1],
    )


def op_fidelity(ref, op) -> dict:
    """Fidelity of one op against one quantized tensor."""
    w = ref.dequantize()
    w_new = op.apply(w)
    w_req = _requantized(ref, w_new)
    delta_i = w_new - w
    delta_a = w_req - w
    old_codes = Q.unpack_codes(ref.packed, ref.bits)[:, : ref.shape[1]]
    new_codes = Q.unpack_codes(ref.requantize(w_new), ref.bits)[:, : ref.shape[1]]
    changed = int(np.count_nonzero(old_codes != new_codes))
    out = {
        "tensor": op.tensor,
        "op": op.op,
        "intended_l2": round(float(np.linalg.norm(delta_i)), 6),
        "actual_l2": round(float(np.linalg.norm(delta_a)), 6),
        "delta_ratio": round(float(np.linalg.norm(delta_a) / (np.linalg.norm(delta_i) + 1e-12)), 4),
        "codes_changed": changed,
        "codes_total": int(old_codes.size),
        "frac_codes_changed": round(changed / old_codes.size, 6),
    }
    if op.op == "project_out":
        d = _effective_direction(op, ref.shape[0])
        with np.errstate(all="ignore"):
            r0, r1, r2 = d @ w, d @ w_new, d @ w_req
        n0 = float(np.linalg.norm(r0))
        removed_i = 1.0 - float(np.linalg.norm(r1)) / (n0 + 1e-12)
        removed_a = 1.0 - float(np.linalg.norm(r2)) / (n0 + 1e-12)
        out.update({
            "removed_intended": round(removed_i, 4),
            "removed_actual": round(removed_a, 4),
            "fidelity": round(removed_a / removed_i, 4) if removed_i > 1e-6 else None,
        })
    return out


def patch_fidelity(archive, patch) -> dict:
    rows = []
    for base, ops in patch.by_tensor(archive).items():
        if not archive.is_quantized(base):
            rows.append({"tensor": base, "op": [o.op for o in ops], "note": "not quantized; no grid rounding"})
            continue
        ref = archive.quant_ref(base)
        for op in ops:
            rows.append(op_fidelity(ref, op))
    scored = [r for r in rows if r.get("fidelity") is not None]
    return {
        "patch_id": patch.id,
        "ops": rows,
        "worst_fidelity": min((r["fidelity"] for r in scored), default=None),
        "verdict": _verdict(scored),
    }


def _verdict(scored: list[dict]) -> str:
    if not scored:
        return "no direction-removal ops to score"
    worst = min(r["fidelity"] for r in scored)
    if worst < 0.25:
        return "washed out: the 4-bit grid erased most of this edit, shrink the target or raise alpha"
    if worst < 0.6:
        return "partial: roughly half the edit survived; expect a weaker effect than designed"
    return "landed: the checkpoint holds most of the intended change"
