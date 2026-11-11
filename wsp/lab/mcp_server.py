"""MCP server for the analyst (Claude Opus / Fable). Proxies onto a running lab.

    WSP_LAB_URL=http://127.0.0.1:8765 python -m wsp.lab.mcp_server

Only analyst tools are exposed. Evaluation on the held-out split, export, and
budget changes stay with the operator (HTTP /admin or the CLI).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

URL = os.environ.get("WSP_LAB_URL", "http://127.0.0.1:8765").rstrip("/")
TIMEOUT = float(os.environ.get("WSP_LAB_TIMEOUT", "3600"))
MAX_CHARS = int(os.environ.get("WSP_LAB_MAX_CHARS", "60000"))

INSTRUCTIONS = """\
You are the analyst in a Weight-Space Programming lab: a live language model whose weights you
can read and edit directly. There is no training and no gradient access. Start with lab_info.
Forward passes are metered; analysis in the python sandbox is free. Checkpoint before edits,
keep an edit only when practice results improve, and run the random-direction control.
"""

logging.getLogger("httpx").setLevel(logging.WARNING)
mcp = MCPServer("wsp-lab", instructions=INSTRUCTIONS)


def _post(tool: str, body: dict) -> str:
    try:
        r = httpx.post(f"{URL}/api/{tool}", json=body, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        return json.dumps({"error": f"lab unreachable at {URL}: {e}. Start it with `python -m wsp.lab serve`."})
    if r.status_code >= 400:
        return json.dumps({"error": f"HTTP {r.status_code}: {r.text[:2000]}"})
    text = json.dumps(r.json(), indent=1, ensure_ascii=False)
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 200] + f"\n... [truncated {len(text) - MAX_CHARS + 200} chars; narrow the request]"
    return text


@mcp.tool(structured_output=False)
def lab_info() -> str:
    """Model architecture, tensor families and shapes, budget, limits, capture sites, sandbox help, and the rules. Call first."""
    return _post("info", {})


@mcp.tool(structured_output=False)
def python(code: str) -> str:
    """Run Python in the persistent weight sandbox (free, unmetered).

    Namespace: W (live weights: W.find(regex), W.layer(i, 'mlp.down_proj.weight'); in-place writes
    edit the model), acts (activations from forward_read), last_read, last_gen, tok, cfg, np, torch,
    and helpers cos, logit_lens, logit_lens_neg, top_singular, token_ids, token_vec, to_np.
    State persists. The last expression's value is echoed. Gradients are disabled.
    """
    return _post("python", {"code": code})


@mcp.tool(structured_output=False)
def practice_problems(n: int = 10, seed: int = 0, offset: int = 0) -> str:
    """List practice problems (id, question, answer) from the practice split. Free."""
    return _post("practice", {"n": n, "seed": seed, "offset": offset})


@mcp.tool(structured_output=False)
def forward_generate(
    prompts: list[str] | None = None,
    practice_ids: list[str] | None = None,
    practice_n: int | None = None,
    seed: int = 0,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    thinking: bool | None = None,
    steer: list[dict[str, Any]] | None = None,
    return_text: str = "truncated",
) -> str:
    """METERED (1 pass). Let the model answer prompts and/or practice problems.

    Practice items are graded automatically. Returns accuracy with a Wilson 95% interval,
    hit_limit_rate (ran out of tokens), loop_rate (hit the limit while repeating), mean_tokens
    and per-row results. Full rows go to the sandbox as last_gen.
    steer: optional inference-time intervention, no weights changed, e.g.
      [{"layer": 15, "vector": "$d", "alpha": 1.0, "mode": "project_out"}]  ("$d" = sandbox variable)
    Use steering as the control for a weight edit: if steering gets the same gain, the edit is a
    steering vector. return_text: "full", "truncated" or "none".
    """
    return _post("forward_generate", {
        "prompts": prompts, "practice_ids": practice_ids, "practice_n": practice_n, "seed": seed,
        "max_new_tokens": max_new_tokens, "temperature": temperature, "thinking": thinking,
        "steer": steer, "return_text": return_text,
    })


@mcp.tool(structured_output=False)
def forward_read(
    texts: list[str],
    capture: list[str] | None = None,
    positions: str = "last",
    top_k: int = 5,
    steer: list[dict[str, Any]] | None = None,
) -> str:
    """METERED (1 pass). Teacher-forced read of texts, with activation capture.

    Returns per-text NLL and next-token top-k. Captured activations land in the sandbox as
    acts['<site>:<layer>'] (shapes are returned, not values). capture examples:
      "mlp_act:15,19" (neuron activations, input to down_proj), "resid:*" (residual after each
      layer), "mlp_out:10-20", "attn_out:3", "attn_heads:7" ([.., heads, head_dim]),
      "module:<name>:in|out".
    positions: "last" (last token, [N, d]), "mean" (masked mean), "all" (list of [len, d]).
    """
    return _post("forward_read", {"texts": texts, "capture": capture, "positions": positions,
                                  "top_k": top_k, "steer": steer})


@mcp.tool(structured_output=False)
def apply_patch(patch: dict[str, Any], control: str | None = None) -> str:
    """Apply a Patch-IR edit to the live weights; reports fidelity per op.

    patch = {"hypothesis": "...", "ops": [{"op": ..., "tensor": ..., "args": {...}}]}
    ops (torch layout W[out, in]; for down_proj rows = hidden dims, cols = neurons):
      project_out {direction[rows], cols?, rows?, alpha=1}  W[:,cols] -= alpha * d d^T W[:,cols]
      add_rank1   {u[rows], v[len(cols)], cols?, alpha=1}   W[:,cols] += alpha * u v^T
      col_gain    {factor, cols?}    row_gain {factor, rows?}    zero_cols {cols}
    Vector args may be "$name" to use a sandbox variable.
    control="random": same op, columns and alpha with a random direction (the null for your edit;
    for project_out it removes less energy than a targeted direction, so compare effects, not norms).
    """
    return _post("apply_patch", {"patch": patch, "control": control})


@mcp.tool(structured_output=False)
def weight_diff() -> str:
    """Every tensor that differs from the base model: elements changed, relative norm, cols/rows touched, edit rank."""
    return _post("diff", {})


@mcp.tool(structured_output=False)
def checkpoint(tag: str, note: str = "") -> str:
    """Save the current weight state under a tag (only changed tensors are stored)."""
    return _post("checkpoint", {"tag": tag, "note": note})


@mcp.tool(structured_output=False)
def restore(tag: str) -> str:
    """Restore a checkpoint by tag, or 'base' for the original model."""
    return _post("restore", {"tag": tag})


@mcp.tool(structured_output=False)
def list_checkpoints() -> str:
    """List saved checkpoints."""
    return _post("checkpoints", {})


@mcp.tool(structured_output=False)
def note(text: str, kind: str = "note") -> str:
    """Record a hypothesis, observation, decision or verdict in the lab ledger.

    kind: "hypothesis" | "observation" | "decision" | "verdict" | "note". The operator reads these.
    """
    return _post("note", {"text": text, "kind": kind})


@mcp.tool(structured_output=False)
def budget() -> str:
    """Forward passes used and left, with the recent pass log."""
    return _post("budget", {})


@mcp.tool(structured_output=False)
def validation() -> str:
    """Per-edit validation history kept by the operator (paired answer-NLL vs base on a held-out
    validation split, one record per weight state). Usually hidden from you; if so, measure on
    practice data instead."""
    return _post("validation", {})


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
