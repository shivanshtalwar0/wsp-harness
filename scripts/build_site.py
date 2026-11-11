"""Snapshot a running lab into a static replay of the 3D viewer, for GitHub Pages.

    python scripts/build_site.py --url http://127.0.0.1:8766 --out site

Writes site/demo/{index.html, static/*, data/*.json}. The landing page
(site/index.html) and screenshots (site/img/) are maintained by hand.
Reading /viz endpoints does not spend forward passes or change the model.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import httpx
import numpy as np

VIEWER = Path(__file__).resolve().parents[1] / "wsp" / "lab" / "viewer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--out", default="site")
    ap.add_argument("--top-active", type=int, default=6, help="most active neurons per layer to include")
    ap.add_argument("--top-norm", type=int, default=3, help="largest-norm neurons per layer to include")
    a = ap.parse_args()
    import os
    tok = os.environ.get("WSP_LAB_TOKEN", "")
    c = httpx.Client(base_url=a.url, timeout=None, headers={"X-WSP-Token": tok} if tok else {})

    demo = Path(a.out) / "demo"
    data = demo / "data"
    if demo.exists():
        shutil.rmtree(demo)
    (demo / "static").mkdir(parents=True)
    data.mkdir()

    layers = {}
    for mode in ("weights", "edits", "activations"):
        d = c.get("/viz/layers", params={"mode": mode}).json()
        layers[mode] = d
        (data / f"layers-{mode}.json").write_text(json.dumps(d))

    def arr(d):
        return np.frombuffer(base64.b64decode(d["mlp"]), dtype=np.float32).reshape(d["layers"], d["neurons"])

    picks: set[tuple[int, int]] = set()
    edits, acts, norms = arr(layers["edits"]), arr(layers["activations"]), arr(layers["weights"])
    for L in range(edits.shape[0]):
        picks.update((L, int(i)) for i in np.nonzero(edits[L])[0])
        if acts[L].any():
            picks.update((L, int(i)) for i in np.argsort(-acts[L])[: a.top_active])
        picks.update((L, int(i)) for i in np.argsort(-norms[L])[: a.top_norm])

    neurons = {}
    for L, i in sorted(picks):
        neurons[f"{L}-{i}"] = c.get("/viz/neuron", params={"layer": L, "idx": i}).json()
    (data / "neurons.json").write_text(json.dumps(neurons))

    state = c.get("/viz/state", params={"since": 0}).json()
    state["snapshot"] = True
    (data / "state.json").write_text(json.dumps(state))

    html = (VIEWER / "index.html").read_text()
    html = html.replace("</title>", "</title>\n  <script>window.WSP_STATIC = \"data/\";</script>", 1)
    (demo / "index.html").write_text(html)
    for f in ("app.js", "style.css"):
        shutil.copy(VIEWER / f, demo / "static" / f)

    size = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file())
    print(f"wrote {demo} · {len(neurons)} neurons · {len(state['events'])} events · {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
