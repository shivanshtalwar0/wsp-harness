"""End-to-end check that a lab running --gen-backend vllm generates from the live weights.

    WSP_LAB_URL=http://<rig>:8765 WSP_LAB_TOKEN=... python scripts/vllm_check.py

Within one lab (greedy decoding throughout):
  1. base:     vLLM vs HF on the same prompts (a zero-strength steer forces the HF path)
  2. edited:   zero a block of MLP columns, then vLLM vs HF again. If the sync works, vLLM
               follows HF's change; if it doesn't, vLLM keeps producing its base outputs.
  3. restored: restore("base"), vLLM again: must return to its step-1 outputs.
Exact token agreement between two different kernels is not expected over long outputs,
and vLLM itself is not bit-deterministic across batches, so it also reruns base vLLM
(the noise floor) and reports both exact-ending agreement and the shared prefix length.
The decisive check is the sync record: after every push, the vLLM worker reads its own
parameters back and the lab compares checksums with the HF model. Spends 6 passes.
"""

from __future__ import annotations

import json
import os
import time

import httpx


def prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    url = os.environ.get("WSP_LAB_URL", "http://127.0.0.1:8765")
    tok = os.environ.get("WSP_LAB_TOKEN", "")
    c = httpx.Client(base_url=url, timeout=None, headers={"X-WSP-Token": tok} if tok else {})

    def call(tool, **body):
        r = c.post(f"/api/{tool}", json=body).json()
        if isinstance(r, dict) and r.get("error"):
            raise SystemExit(f"{tool}: {r['error']}")
        return r

    ids = [p["id"] for p in call("practice", n=16, seed=7)["problems"]]
    call("python", code="z = torch.zeros(cfg['config']['hidden_size'])")
    hf_steer = [{"layer": 0, "vector": "$z", "alpha": 0.0, "mode": "add"}]   # no-op, forces HF

    def gen(engine: str):
        t0 = time.time()
        g = call("forward_generate", practice_ids=ids, max_new_tokens=512, return_text="full",
                 steer=hf_steer if engine == "hf" else None)
        if engine == "vllm":
            syncs.append(g.get("engine"))
        return {r["id"]: r for r in g["rows"]}, round(time.time() - t0, 1)

    def compare(a, b):
        same_ans = sum(a[i]["correct"] == b[i]["correct"] and a[i]["completion"][-40:] == b[i]["completion"][-40:]
                       for i in ids)
        pref = sorted(prefix(a[i]["completion"], b[i]["completion"]) for i in ids)
        return {"same_ending": f"{same_ans}/{len(ids)}", "median_shared_prefix_chars": pref[len(pref) // 2]}

    out, syncs = {}, []
    v0, tv0 = gen("vllm")
    v0b, _ = gen("vllm")
    out["noise floor: base vllm vs base vllm"] = compare(v0b, v0)
    h0, th0 = gen("hf")
    out["base: vllm vs hf"] = {**compare(v0, h0), "secs": {"vllm": tv0, "hf": th0}}

    call("python", code="W.layer(18, 'mlp.down_proj.weight')[:, :4000] *= 0.0")
    v1, _ = gen("vllm")
    h1, _ = gen("hf")
    out["edited: vllm vs hf"] = compare(v1, h1)
    out["edited vllm vs base vllm"] = compare(v1, v0)        # must differ if the edit reached vLLM

    call("restore", tag="base")
    v2, _ = gen("vllm")
    out["restored vllm vs base vllm"] = compare(v2, v0)      # back to the noise floor
    out["vllm syncs (per vllm generate)"] = syncs             # edited + restored must be verified
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
