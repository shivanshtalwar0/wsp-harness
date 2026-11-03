"""wsp: Weight-Space Programming on a 4-bit 27B agent.

    python -m wsp.cli manifest  <model>
    python -m wsp.cli atlas     <model> --layers 15,19 --suffix mlp.down_proj
    python -m wsp.cli analytic  <model> --tensor <base> --out patch.json
    python -m wsp.cli apply     <model> patch.json --out ./edited [--dry-run]
    python -m wsp.cli diff      <a> <b>
    python -m wsp.cli eval      --model qwen --base-url http://127.0.0.1:1234/v1
    python -m wsp.cli probes    runs/baseline.json
    python -m wsp.cli prompt    <model> --probes runs/baseline.json --out prompt.txt
    python -m wsp.cli verify-mlx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .patch import Patch


def _archive(path):
    from .weights import ModelArchive

    return ModelArchive(path)


def cmd_manifest(a) -> int:
    with _archive(a.model) as m:
        print(json.dumps(m.manifest(), indent=2))
    return 0


def cmd_verify_mlx(a) -> int:
    from . import quant

    try:
        print(json.dumps(quant.verify_mlx(), indent=2))
    except ImportError:
        print("mlx not installed; install with: pip install mlx", file=sys.stderr)
        return 2
    return 0


def cmd_atlas(a) -> int:
    from .analyst import build_atlas, resolve_tensors

    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    with _archive(a.model) as m:
        tensors = resolve_tensors(m, layers=layers, suffix=a.suffix)
        if not tensors:
            print("no quantized tensors matched", file=sys.stderr)
            return 2
        atlas = build_atlas(m, tensors, top_k=a.top_k)
    text = json.dumps(atlas, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"wrote {a.out} ({len(text)} chars)")
    else:
        print(text)
    return 0


def cmd_analytic(a) -> int:
    from .analyst import analytic_patch

    with _archive(a.model) as m:
        patch = analytic_patch(m, a.tensor, alpha=a.alpha, row_support=a.row_support)
        print(patch.describe(m))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(patch.to_json(), indent=2))
            print(f"wrote {a.out}")
    return 0


def cmd_apply(a) -> int:
    with _archive(a.model) as m:
        patch = Patch.from_json(json.loads(Path(a.patch).read_text()))
        if a.dry_run:
            print(patch.describe(m))
            print(json.dumps({b: len(ops) for b, ops in patch.by_tensor(m).items()}, indent=1))
            return 0
        summary = m.apply(patch, a.out)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_verify(a) -> int:
    from .verify import patch_fidelity

    with _archive(a.model) as m:
        patch = Patch.from_json(json.loads(Path(a.patch).read_text()))
        print(json.dumps(patch_fidelity(m, patch), indent=2))
    return 0


def cmd_diff(a) -> int:
    from .weights import diff

    with _archive(a.a) as A, _archive(a.b) as B:
        print(json.dumps(diff(A, B), indent=2))
    return 0


def cmd_eval(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals import harness

    tasks = harness.discover(a.tasks)
    if a.task:
        tasks = [t for t in tasks if t.id in a.task]
    if not tasks:
        print("no tasks", file=sys.stderr)
        return 2
    client = harness.OpenAICompat(a.base_url, a.model, api_key=a.api_key,
                                  temperature=a.temperature, max_tokens=a.max_tokens)
    print(f"{len(tasks)} task(s) against {a.base_url} model={a.model}", file=sys.stderr)
    summary = harness.run_suite(tasks, client, a.out, max_steps=a.max_steps)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    return 0


def cmd_probes(a) -> int:
    from .probes import detect, rank

    data = json.loads(Path(a.suite).read_text())
    reports = [detect(e) for e in data["episodes"]]
    print(json.dumps(rank(reports), indent=2))
    return 0


def cmd_prompt(a) -> int:
    from .analyst import build_atlas, build_prompt, resolve_tensors, save_prompt
    from .probes import detect, rank

    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    suite = json.loads(Path(a.probes).read_text())
    report = rank([detect(e) for e in suite["episodes"]])
    with _archive(a.model) as m:
        tensors = resolve_tensors(m, layers=layers, suffix=a.suffix or "mlp.down_proj")
        atlas = build_atlas(m, tensors, top_k=a.top_k)
    prompt = build_prompt(atlas, report)
    save_prompt(prompt, a.out)
    print(f"wrote {a.out} ({len(prompt)} chars, ~{len(prompt)//4} tokens)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wsp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("manifest", help="checkpoint inventory"); s.add_argument("model"); s.set_defaults(fn=cmd_manifest)
    s = sub.add_parser("verify-mlx"); s.set_defaults(fn=cmd_verify_mlx)

    s = sub.add_parser("atlas", help="per-tensor statistics for the analyst")
    s.add_argument("model"); s.add_argument("--layers"); s.add_argument("--suffix"); s.add_argument("--top-k", type=int, default=8)
    s.add_argument("--out"); s.set_defaults(fn=cmd_atlas)

    s = sub.add_parser("analytic", help="build the no-LLM baseline patch")
    s.add_argument("model"); s.add_argument("--tensor", required=True); s.add_argument("--alpha", type=float, default=1.0)
    s.add_argument("--row-support", type=float, default=1.0, help="fraction of rows to edit (default all)")
    s.add_argument("--out"); s.set_defaults(fn=cmd_analytic)

    s = sub.add_parser("apply", help="write a patched checkpoint")
    s.add_argument("model"); s.add_argument("patch"); s.add_argument("--out"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_apply)

    s = sub.add_parser("verify", help="did the patch survive 4-bit requantization?")
    s.add_argument("model"); s.add_argument("patch"); s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("diff", help="compare two checkpoints at code level")
    s.add_argument("a"); s.add_argument("b"); s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("eval", help="run the agent suite against an endpoint")
    s.add_argument("--tasks"); s.add_argument("--task", action="append")
    s.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    s.add_argument("--model", default="qwen/qwen3.8-27b")
    s.add_argument("--api-key"); s.add_argument("--temperature", type=float, default=0.6)
    s.add_argument("--max-tokens", type=int, default=8192); s.add_argument("--max-steps", type=int)
    s.add_argument("--out", default="runs/baseline.json"); s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("probes", help="rank failure probes over a recorded suite")
    s.add_argument("suite"); s.set_defaults(fn=cmd_probes)

    s = sub.add_parser("prompt", help="render the analyst prompt")
    s.add_argument("model"); s.add_argument("--probes", required=True); s.add_argument("--layers")
    s.add_argument("--suffix"); s.add_argument("--top-k", type=int, default=8); s.add_argument("--out", default="runs/prompt.txt")
    s.set_defaults(fn=cmd_prompt)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
