"""The analyst side of the loop: describe weights compactly, ask for a patch, validate it.

A 27B checkpoint cannot be pasted into a context window the way the paper pasted
17,253 weights. What the analyst gets instead is an *atlas*: cheap, interpretable
summaries of a handful of tensors (norm profiles, dominant rows/columns, affine
grid health) plus the failure evidence from ``wsp.probes``.

Two producers of patches are supported:

  analytic  dominant-direction projection via power iteration. Costs seconds,
            needs no LLM, and is the baseline any LLM proposal must beat.
  llm       the analyst model reads the atlas + failures and emits a JSON patch.

If the LLM proposal cannot beat the analytic baseline on held-out tasks, that is
the honest answer to "can this model do WSP on itself".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .patch import Op, Patch, SCHEMA_HELP


# ------------------------------------------------------------------ analytics


def dominant_direction(w: np.ndarray, iters: int = 24, seed: int = 0) -> np.ndarray:
    """Top left singular vector of ``w`` (length = rows) by power iteration.

    BLAS leaves stale FP status flags around, which numpy 2 then reports as
    spurious overflow warnings on the next matmul, so errstate is silenced and
    the result is validated instead.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=w.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    u = np.zeros(w.shape[0], dtype=np.float32)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            u = w @ v
            nu = np.linalg.norm(u)
            if not np.isfinite(nu) or nu == 0:
                break
            u /= nu
            v = w.T @ u
            nv = np.linalg.norm(v)
            if not np.isfinite(nv) or nv == 0:
                break
            v /= nv
    if not np.isfinite(u).all() or not np.linalg.norm(u):
        raise FloatingPointError("power iteration diverged; tensor has non-finite values")
    return u / np.linalg.norm(u)


def tensor_atlas(w: np.ndarray, top_k: int = 10) -> dict:
    """Compact, human/LLM-readable statistics for one matrix."""
    w = np.asarray(w, dtype=np.float32)
    row_norm = np.linalg.norm(w, axis=1)
    col_norm = np.linalg.norm(w, axis=0)
    d = dominant_direction(w)
    with np.errstate(all="ignore"):
        energy = float(np.linalg.norm(w.T @ d) / (np.linalg.norm(w) + 1e-12))
    top_rows = np.argsort(-row_norm)[:top_k]
    top_cols = np.argsort(-col_norm)[:top_k]
    return {
        "shape": list(w.shape),
        "abs_mean": float(np.abs(w).mean()),
        "abs_max": float(np.abs(w).max()),
        "row_norm": {"min": float(row_norm.min()), "median": float(np.median(row_norm)), "max": float(row_norm.max())},
        "col_norm": {"min": float(col_norm.min()), "median": float(np.median(col_norm)), "max": float(col_norm.max())},
        "top_rows": [[int(i), round(float(row_norm[i]), 5)] for i in top_rows],
        "top_cols": [[int(i), round(float(col_norm[i]), 5)] for i in top_cols],
        "dominant_direction_energy": round(energy, 4),
        "dominant_direction": [round(float(x), 6) for x in d],
        "zero_cols": int((col_norm == 0).sum()),
    }


def grid_atlas(ref) -> dict:
    """Health of the 4-bit affine grid for one quantized tensor."""
    s = np.asarray(ref.scales, dtype=np.float32)
    b = np.asarray(ref.biases, dtype=np.float32)
    return {
        "groups": int(s.shape[1]),
        "group_size": ref.group_size,
        "scale_median": round(float(np.median(np.abs(s))), 8),
        "scale_p01_p99": [round(float(np.percentile(np.abs(s), 1)), 8), round(float(np.percentile(np.abs(s), 99)), 8)],
        "negative_scale_frac": round(float((s < 0).mean()), 4),
        "degenerate_groups": int((s == 0).sum()),
        "bias_abs_median": round(float(np.median(np.abs(b))), 8),
        "effective_bits": round(ref.bits + (16 * 2) / ref.group_size, 3),
    }


def resolve_tensors(archive, layers: list[int] | None = None, suffix: str | None = None) -> list[str]:
    bases = sorted({archive.base_of(k) for k in archive.keys()})
    out = []
    for b in bases:
        if suffix and not b.endswith(suffix):
            continue
        if layers and not any(f".layers.{L}." in b for L in layers):
            continue
        if archive.is_quantized(b):
            out.append(b)
    return out


def build_atlas(archive, tensors: list[str], top_k: int = 8, max_columns: int = 24) -> dict:
    atlas = {"model": {"bits": archive.bits, "group_size": archive.group_size,
                       "params_b": round(sum(archive.numel(b) for b in {archive.base_of(k) for k in archive.keys()}) / 1e9, 2)},
             "tensors": {}}
    for base in tensors:
        if not archive.is_quantized(base):
            continue
        ref = archive.quant_ref(base)
        w = ref.dequantize()
        a = tensor_atlas(w, top_k=top_k)
        a["dominant_direction"] = a["dominant_direction"][:max_columns] + ["..."] if len(a["dominant_direction"]) > max_columns else a["dominant_direction"]
        a["grid"] = grid_atlas(ref)
        a["n_weights"] = ref.shape[0] * ref.shape[1]
        atlas["tensors"][base] = a
        del w
    return atlas


# --------------------------------------------------------------------- prompt


PROMPT_HEAD = """\
You are doing Weight-Space Programming on a 4-bit quantized 27B language model.
You cannot run training, gradients, or fine-tuning. You may only change numbers
inside the checkpoint. Your edit must remove a behaviour that is suppressing
ability the model already has; it cannot add ability that is absent.

## Evidence: how the model fails as a coding agent
{evidence}

## Weight atlas (per-tensor statistics; full matrices are far too large to show)
{atlas}

## Rules
- Prefer one hypothesis, one tensor pair, one small edit.
- `project_out` removes the component of selected columns along `direction`.
  Pass the direction exactly as listed in the atlas (it is already unit length).
- Changing more than ~0.5% of the model's weights is a red flag, not a win.
- If the evidence looks like a harness problem (no tests run, context bloat),
  reply with {{"hypothesis": "not a weight problem: ...", "ops": []}}.

{schema}
"""


def build_prompt(atlas: dict, probe_report: dict) -> str:
    ev = json.dumps(probe_report, indent=1)[:4000]
    at = json.dumps(atlas, indent=1)
    if len(at) > 24000:
        at = at[:24000] + "\n... (truncated)"
    return PROMPT_HEAD.format(evidence=ev, atlas=at, schema=SCHEMA_HELP)


def propose(client, prompt: str, archive=None) -> Patch | None:
    """Ask the analyst model for a patch and validate it against the checkpoint."""
    reply = client.chat([{"role": "user", "content": prompt}])
    text = reply["text"] if isinstance(reply, dict) else str(reply)
    from .patch import extract_json

    obj = extract_json(text)
    if not obj.get("ops"):
        return None
    patch = Patch.from_json(obj)
    if archive is not None:
        patch.by_tensor(archive)  # raises on unknown tensor names / bad quant space
        for op in patch.ops:
            if op.op == "project_out":
                ref = archive.quant_ref(op.tensor)
                if len(op.args.get("direction", [])) != ref.shape[0]:
                    raise ValueError(f"{op.tensor}: direction must have {ref.shape[0]} entries, got {len(op.args.get('direction', []))}")
    return patch


# ------------------------------------------------------------------- baseline


def analytic_patch(archive, base: str, cols: list[int] | None = None, alpha: float = 1.0,
                   row_support: float = 1.0) -> Patch:
    """The no-LLM baseline: project out the dominant direction.

    ``row_support`` < 1 keeps only the largest-|d| rows. This matters more than
    it looks: on a 4-bit grid a spread-out unit direction changes each entry by
    ~|c_j|/sqrt(n), which is usually below half a quantization step, so a full
    projection is largely erased on requantization. Concentrating the same
    vector into fewer rows raises the per-entry change above the floor.
    """
    ref = archive.quant_ref(base)
    w = ref.dequantize()
    d = dominant_direction(w)
    args = {"direction": [float(x) for x in d], "cols": cols, "alpha": alpha}
    if row_support < 1.0:
        keep = max(1, int(round(row_support * d.size)))
        idx = np.argsort(-np.abs(d))[:keep]
        args["rows"] = sorted(int(i) for i in idx)
    return Patch(
        id="",
        hypothesis=f"remove dominant direction of {base} (analytic baseline, row support {row_support:g})",
        ops=[Op("project_out", base, args, "float")],
        evidence={"source": "power_iteration", "row_support": row_support},
    )


def save_prompt(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
