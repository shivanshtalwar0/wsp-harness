"""A scripted demo round against a running lab (NOT an analyst run).

It exercises every part of the harness once, on real data, so the viewer and the
Pages snapshot have something true to show:

  baseline generate -> activation read -> pick columns + direction in the sandbox
  -> checkpoint -> project_out edit -> re-measure -> random control -> verdict note

The edit is the paper's *shape* (project one direction out of a few dozen MLP
columns in layers 15 and 19) chosen by a fixed recipe, not by insight. Expect a
null or a small effect; the point is the plumbing and the controls.

    python scripts/demo_session.py --url http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--layers", default="15,19")
    ap.add_argument("--cols", type=int, default=64)
    a = ap.parse_args()
    import os
    tok = os.environ.get("WSP_LAB_TOKEN", "")
    c = httpx.Client(base_url=a.url, timeout=None, headers={"X-WSP-Token": tok} if tok else {})

    def call(tool, **body):
        r = c.post(f"/api/{tool}", json=body).json()
        if isinstance(r, dict) and r.get("error"):
            raise SystemExit(f"{tool}: {r['error']}")
        return r

    def gen(ids):
        g = call("forward_generate", practice_ids=ids, max_new_tokens=a.max_new_tokens, return_text="none")
        return {k: g[k] for k in ("accuracy", "accuracy_ci95", "hit_limit_rate", "loop_rate", "mean_tokens")}

    layers = [int(x) for x in a.layers.split(",")]
    call("note", kind="observation",
         text="Scripted demo round (scripts/demo_session.py), not an analyst run: a fixed recipe "
              "exercising baseline, capture, edit, re-measure and random control.")

    probs = call("practice", n=a.n, seed=1)["problems"]
    ids = [p["id"] for p in probs]
    base = gen(ids)
    print("baseline", base)
    call("note", kind="observation", text=f"Baseline on {a.n} MATH practice items: {json.dumps(base)}")

    texts = [p["question"] for p in call("practice", n=64, seed=2)["problems"]]
    call("forward_read", texts=texts, capture=[f"mlp_act:*"], positions="mean")

    code = f"""
picks = {{}}
for L in {layers}:
    a = acts[f'mlp_act:{{L}}'].abs().mean(0)
    cols = a.topk({a.cols}).indices.sort().values
    Wd = W.layer(L, 'mlp.down_proj.weight')[:, cols]
    u, s, v = top_singular(Wd, 1)
    globals()[f'd{{L}}'] = u[:, 0]
    picks[L] = dict(cols=cols.tolist(), sigma=round(float(s[0]), 3), promotes=[t for t, _ in logit_lens(u[:, 0], 6)])
picks
"""
    out = call("python", code=code)
    picks = eval(out["output"])  # noqa: S307 - our own sandbox's repr of a dict of lists
    for L, p in picks.items():
        print(f"layer {L}: sigma {p['sigma']} promotes {p['promotes']}")

    call("checkpoint", tag="pre-demo", note="state before the demo edit")
    call("note", kind="hypothesis",
         text=f"Recipe edit: in layers {layers}, project the top singular direction of the "
              f"{a.cols} most active down_proj columns out of those columns. No mechanism claimed.")
    patch = {"hypothesis": "demo: top direction of most-active columns, layers " + a.layers,
             "ops": [{"op": "project_out", "tensor": f"model.layers.{L}.mlp.down_proj",
                      "args": {"direction": f"$d{L}", "cols": picks[L]["cols"]}} for L in layers]}
    rep = call("apply_patch", patch=patch)
    print("edit", [(o["tensor"], o["fidelity"], o["elements_changed"]) for o in rep["ops"]])
    after = gen(ids)
    print("after ", after)
    call("checkpoint", tag="demo-edit", note=json.dumps(after))

    call("restore", tag="pre-demo")
    call("apply_patch", patch=patch, control="random")
    ctrl = gen(ids)
    print("control", ctrl)
    call("checkpoint", tag="demo-control", note=json.dumps(ctrl))
    call("restore", tag="demo-edit")

    call("note", kind="verdict",
         text=f"base acc {base['accuracy']} {base['accuracy_ci95']} · edit {after['accuracy']} "
              f"{after['accuracy_ci95']} · random control {ctrl['accuracy']} {ctrl['accuracy_ci95']} "
              f"(n={a.n}). Intervals overlap at this n; not evidence of an effect either way. "
              f"Left at checkpoint demo-edit for the viewer.")
    print(json.dumps({"baseline": base, "edit": after, "control": ctrl}, indent=1))


if __name__ == "__main__":
    main()
