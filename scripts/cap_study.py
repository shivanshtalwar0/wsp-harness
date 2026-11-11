"""How much does the generation token cap cost? Same practice items, several caps.

    WSP_LAB_URL=http://192.168.0.182:8765 WSP_LAB_TOKEN=... \\
        python scripts/cap_study.py --caps 1024,2048,4096 --n 64 --seed 1

For each cap: accuracy (Wilson CI), share of answers cut off at the cap, mean tokens,
wall time; and versus the smallest cap, which items were rescued (wrong -> right) or
lost. Spends one forward pass per cap. Results go to runs/cap-study/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("WSP_LAB_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--caps", default="1024,4096")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ids-file", help="JSON list of practice ids (overrides --n/--seed)")
    a = ap.parse_args()
    tok = os.environ.get("WSP_LAB_TOKEN", "")
    c = httpx.Client(base_url=a.url, timeout=None, headers={"X-WSP-Token": tok} if tok else {})

    def call(tool, **body):
        r = c.post(f"/api/{tool}", json=body).json()
        if isinstance(r, dict) and r.get("error"):
            raise SystemExit(f"{tool}: {r['error']}")
        return r

    ids = json.load(open(a.ids_file)) if a.ids_file else [p["id"] for p in call("practice", n=a.n, seed=a.seed)["problems"]]
    caps = [int(x) for x in a.caps.split(",")]
    out: dict = {"ids": ids, "caps": {}}
    Path("runs/cap-study").mkdir(parents=True, exist_ok=True)
    path = Path("runs/cap-study") / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    first = None
    for cap in caps:
        t0 = time.time()
        g = call("forward_generate", practice_ids=ids, max_new_tokens=cap, return_text="full")
        secs = round(time.time() - t0, 1)
        rows = {r["id"]: r for r in g["rows"]}
        rec = {"secs": secs, **{k: g[k] for k in ("accuracy", "accuracy_ci95", "hit_limit_rate", "mean_tokens", "loop_rate")},
               "max_tokens": max(r["n_tokens"] for r in g["rows"]), "rows": g["rows"]}
        line = (f"cap {cap:>5}: {secs:>7.1f}s  acc {g['accuracy']:.3f} {g['accuracy_ci95']}  "
                f"cut off {g['hit_limit_rate']:.3f}  mean tok {g['mean_tokens']:.0f}  max tok {rec['max_tokens']}")
        if first is not None:
            rescued = [i for i in ids if rows[i]["correct"] and not first[i]["correct"]]
            lost = [i for i in ids if first[i]["correct"] and not rows[i]["correct"]]
            was_cut = [i for i in ids if first[i]["hit_limit"]]
            rec.update(rescued=rescued, lost=lost,
                       previously_cut_now_finish=sum(not rows[i]["hit_limit"] for i in was_cut),
                       previously_cut_now_right=sum(rows[i]["correct"] for i in was_cut))
            line += (f"  | vs {caps[0]}: +{len(rescued)} rescued, -{len(lost)} lost; of {len(was_cut)} cut off at "
                     f"{caps[0]}: {rec['previously_cut_now_finish']} now finish, {rec['previously_cut_now_right']} now right")
        else:
            first = rows
        print(line, flush=True)
        out["caps"][cap] = rec
        path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
