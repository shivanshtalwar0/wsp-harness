"""Insert facts into a running lab's model without gradients, and measure what stuck.

    WSP_LAB_URL=http://<rig>:8765 WSP_LAB_TOKEN=... python scripts/fact_edit.py \\
        --facts runs/voxmin/facts.jsonl --general runs/voxmin/general.jsonl \\
        --layers 4-8 --target 8 --lam 1e3 --tag try1

Everything goes through the lab API, so the ledger, validation and viewer see it:

  1. restore("base"), then score the base model: edit questions, held-out test questions,
     the in-context ceiling (test questions with the fact stated first), general control.
  2. one forward_read of the in-context texts  -> target residuals at --target
     one forward_read of generic text          -> key second moments per edited layer
     per layer: one forward_read of the bare texts -> closed-form down_proj write
     (wsp/lab/factedit.py; the writes happen in the sandbox and land in the ledger)
  3. score the edited model the same way, checkpoint it, save results.

The facts file path must be valid on the lab machine (the sandbox reads it there).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.lab.data import contains_answer, wilson  # noqa: E402

MAX_READ = 1000


def rng(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("WSP_LAB_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--facts", required=True, help="facts jsonl (local path; same relative path on the lab)")
    ap.add_argument("--general", required=True, help="general-knowledge control, facts format (test_questions)")
    ap.add_argument("--layers", default="4-8")
    ap.add_argument("--target", type=int, default=None, help="target layer (default: last edited layer)")
    ap.add_argument("--lam", type=float, default=1e3)
    ap.add_argument("--generic-math", type=int, default=300, help="MATH problems in the generic text for C")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--tag", default=time.strftime("%H%M%S"))
    ap.add_argument("--skip-base", action="store_true", help="skip scoring the base model")
    ap.add_argument("--batch", type=int, default=0, help="facts per write (0 = all at once)")
    ap.add_argument("--target-kind", choices=["logit", "ctx"], default="logit",
                    help="logit: closed-form change that makes each answer token win at the final layer; "
                         "ctx: match the in-context residual at --target")
    ap.add_argument("--margin", type=float, default=3.0, help="logit targets: nats the answer token should win by")
    ap.add_argument("--max-step", type=float, default=6.0)
    ap.add_argument("--focus", choices=["all", "answer"], default="answer",
                    help="constrain every response position, or only the first and the answer tokens")
    ap.add_argument("--response", choices=["distill", "fact"], default="distill",
                    help="teacher-forced response: the model's own in-context answer, or the fact sentence")
    a = ap.parse_args()
    layers = rng(a.layers)
    target = a.target if a.target is not None else layers[-1]
    tok_ = os.environ.get("WSP_LAB_TOKEN", "")
    c = httpx.Client(base_url=a.url, timeout=None, headers={"X-WSP-Token": tok_} if tok_ else {})

    def call(tool, **body):
        r = c.post(f"/api/{tool}", json=body).json()
        if isinstance(r, dict) and r.get("error"):
            raise SystemExit(f"{tool}: {r['error']}")
        return r

    def py(code):
        out = call("python", code=code)
        if out.get("error") or "Traceback" in out.get("output", ""):
            raise SystemExit(f"sandbox: {out}")
        return out.get("output", "")

    facts = [json.loads(x) for x in Path(a.facts).read_text().splitlines() if x.strip()]
    n_layers = int(py("print(cfg['config']['num_hidden_layers'])"))
    general = [json.loads(x) for x in Path(a.general).read_text().splitlines() if x.strip()]

    def score(items, ctx=False):
        """items: [(prompt, answers)] -> accuracy, rows. Generated in chunks of <= 512."""
        rows = []
        for i in range(0, len(items), 512):
            chunk = items[i:i + 512]
            g = call("forward_generate", prompts=[p for p, _ in chunk], max_new_tokens=a.max_new_tokens,
                     return_text="full")
            rows += g["rows"]
        ok = [contains_answer(r["completion"], ans) for r, (_, ans) in zip(rows, items)]
        k = sum(ok)
        return {"accuracy": round(k / len(ok), 4), "ci95": wilson(k, len(ok)), "n": len(ok),
                "correct": ok, "completions": [r["completion"][:300] for r in rows],
                "full": [r["completion"] for r in rows]}

    lead = re.compile(r"^\s*(based on|according to|as (stated|mentioned|noted) in|from) (the )?(information|details|text|"
                      r"facts?|context|data)( (provided|given|above|you provided|shared))?[^,:]*[,:]\s*", re.I)

    def trim(text, answers):
        """The sentence that states the answer, without lead-ins that point at the context."""
        text = lead.sub("", text.strip())
        text = text[:1].upper() + text[1:]
        low = text.lower()
        hits = [low.find(x.lower()) + len(x) for x in answers if x and low.find(x.lower()) >= 0]
        if not hits:
            return None
        end = min(hits)
        m = re.search(r"[.!?](\s|$)|\n", text[end:])
        return text[: end + (m.end() if m else len(text) - end)].rstrip()

    def ans(f):
        return [f["answer"], *f.get("aliases", [])]

    edit_q = [(q, ans(f)) for f in facts for q in f["edit_questions"]]
    test_q = [(q, ans(f)) for f in facts for q in f["test_questions"]]
    ctx_q = [(f"{f['fact']}\n\n{q}", ans(f)) for f in facts for q in f["test_questions"]]
    gen_q = [(q, ans(f)) for f in general for q in f["test_questions"]]

    def measure(label, with_ctx=False):
        t0 = time.time()
        m = {"edit": score(edit_q), "test": score(test_q), "general": score(gen_q)}
        if with_ctx:
            m["in_context"] = score(ctx_q)
        line = "  ".join(f"{k} {v['accuracy']:.3f}" for k, v in m.items())
        print(f"[{label}] {line}  ({time.time() - t0:.0f}s)", flush=True)
        return m

    call("restore", tag="base")
    out = {"args": vars(a), "layers": layers, "target": target, "n_facts": len(facts)}
    path = Path("runs/voxmin") / f"results-{a.tag}.json"
    if not a.skip_base:
        out["base"] = measure("base", with_ctx=True)

    # -- responses: what the model says when the fact is in its context (kept only if right)
    responses = None
    if a.response == "distill":
        ctx_edit = [(f"{f['fact']}\n\n{q}", ans(f)) for f in facts for q in f["edit_questions"]]
        d = score(ctx_edit)
        responses = []
        for ok, comp, (_, answers) in zip(d["correct"], d["full"], ctx_edit):
            responses.append((trim(comp, answers) or None) if ok else None)
        out["distill"] = {"accuracy": d["accuracy"], "used": sum(r is not None for r in responses)}
        print(f"  in-context answers to edit questions: {d['accuracy']:.3f} right, used as targets", flush=True)
    py(f"import json as _j\nR = _j.loads({json.dumps(responses, ensure_ascii=False)!r})")

    # -- generic text for the key second moments (computed once, on the base model)
    py(f"F_all = load_facts({a.facts!r})\n"
       f"G = load_facts({a.general!r})\n"
       "from wsp.lab.data import load_source as _ls\n"
       f"_m = _ls('math:train')[:{a.generic_math}]\n"
       "_chat = lambda u, r: tok.apply_chat_template([{'role':'user','content':u}], tokenize=False, "
       "add_generation_prompt=True, enable_thinking=False) + r + tok.eos_token\n"
       "generic = [_chat(p.question, p.solution) for p in _m] + "
       "[_chat(q, f['answer'] + '.') for f in G for q in f['test_questions']]")

    def read_all(name, capture):
        """forward_read over a sandbox list in chunks; concatenates per-text acts into store."""
        n = int(py(f"print(len({name}))"))
        py("store = {}")
        for i in range(0, n, MAX_READ):
            py(f"_chunk = {name}[{i}:{i + MAX_READ}]")
            call("forward_read", texts="$_chunk", capture=capture, positions="all")
            py("for _k, _v in acts.items():\n"
               f"    if _k in {capture!r}: store.setdefault(_k, []).extend(_v)")

    t0 = time.time()
    read_all("generic", [f"mlp_act:{l}" for l in layers])
    py(f"C = {{l: second_moment(store[f'mlp_act:{{l}}']) for l in {layers}}}")
    batch = a.batch or len(facts)
    out["batches"] = []
    for b0 in range(0, len(facts), batch):
        b1 = min(b0 + batch, len(facts))
        nq = sum(len(f["edit_questions"]) for f in facts[:b0]), sum(len(f["edit_questions"]) for f in facts[:b1])
        py(f"F = F_all[{b0}:{b1}]\n"
           f"bare, ctx, pb, pc = fact_texts(tok, F, responses=R[{nq[0]}:{nq[1]}] if R else None, focus={a.focus!r})\n"
           "print(len(bare))")
        if a.target_kind == "logit":
            final, nk = n_layers - 1, "module:model.norm:out"
            py("Y = next_tokens(tok, bare, pb); U = unembed()")
            gaps = []
            for i, layer in enumerate(layers):
                read_all("bare", [f"resid:{final}", nk, f"mlp_act:{layer}"])
                g = py(f"_dW, _st = logit_layer_update(store, {layer}, {final}, {nk!r}, Y, pb, U, C[{layer}], "
                       f"{len(layers) - i}, {a.lam}, {a.margin}, {a.max_step})\n"
                       f"_W = W.layer({layer}, 'mlp.down_proj.weight')\n"
                       "_W[:] += _dW.to(_W.dtype).to(_W.device)\n"
                       "print(_st['top1'], _st['median_margin'], round(float(_dW.norm()), 3))")
                top1, med, dn = g.split()
                gaps.append({"layer": layer, "top1_before": float(top1), "median_margin": float(med), "dW_norm": float(dn)})
            read_all("bare", [f"resid:{final}", nk])
            fin = py(f"_d, _st = logit_targets(gather(store['resid:{final}'], pb), gather(store[{nk!r}], pb), Y, U)\n"
                     "print(_st['top1'], _st['median_margin'])").split()
            out["batches"].append({"facts": [b0, b1], "layers": gaps, "top1_after": float(fin[0]),
                                   "median_margin_after": float(fin[1])})
            print(f"  facts {b0}-{b1}: answer-token top-1 {gaps[0]['top1_before']:.3f} -> {float(fin[0]):.3f}  "
                  f"median margin {gaps[0]['median_margin']:.2f} -> {float(fin[1]):.2f}  "
                  f"max |dW| {max(x['dW_norm'] for x in gaps):.2f}  ({time.time() - t0:.0f}s)", flush=True)
            continue
        read_all("ctx", [f"resid:{target}"])           # targets from the current (already edited) model
        py(f"target = gather(store['resid:{target}'], pc)")
        gaps = []
        for i, layer in enumerate(layers):
            read_all("bare", [f"resid:{target}", f"mlp_act:{layer}"])
            g = py(f"_g = residual_gap(store, {target}, target, pb)\n"
                   f"_dW = layer_update(store, {layer}, {target}, target, pb, C[{layer}], {len(layers) - i}, {a.lam})\n"
                   f"_W = W.layer({layer}, 'mlp.down_proj.weight')\n"
                   "_before = _W.detach().float().clone()\n"
                   "_W[:] += _dW.to(_W.dtype).to(_W.device)\n"
                   "_got = _W.detach().float() - _before\n"
                   "_fid = float((_got * _dW.to(_got.device)).sum() / (_dW.norm() ** 2).clamp(min=1e-20))\n"
                   "print(round(_g, 4), round(float(_dW.norm()), 4), round(float(_before.norm()), 2), round(_fid, 4))")
            gap, dn, wn, fid = g.split()
            gaps.append({"layer": layer, "gap_before": float(gap), "dW_norm": float(dn), "W_norm": float(wn),
                         "fidelity": float(fid)})
        read_all("bare", [f"resid:{target}"])
        final_gap = float(py(f"print(round(residual_gap(store, {target}, target, pb), 4))"))
        out["batches"].append({"facts": [b0, b1], "gaps": gaps, "final_gap": final_gap})
        print(f"  facts {b0}-{b1}: gap {gaps[0]['gap_before']:.3f} -> {final_gap:.3f}  "
              f"max |dW| {max(x['dW_norm'] for x in gaps):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    call("checkpoint", tag=f"facts-{a.tag}", note=json.dumps({"layers": layers, "target": target, "lam": a.lam,
                                                             "batch": batch, "focus": a.focus}))
    out["edited"] = measure("edited")
    for m in ("base", "edited"):
        for v in out.get(m, {}).values():
            v.pop("full", None)
    out["budget"] = call("budget")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
