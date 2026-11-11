"""python -m wsp.lab {serve, eval, export, budget, mcp}

    python -m wsp.lab serve --model Qwen/Qwen3-4B --budget 50 --port 8765
    python -m wsp.lab eval --n 200                      # held-out score of the live weights
    python -m wsp.lab eval --n 200 --checkpoint base    # ... of the original model
    python -m wsp.lab export --full-model
    python -m wsp.lab mcp                               # stdio MCP server (for the analyst)
"""

from __future__ import annotations

import argparse
import json
import sys

DEFAULT_URL = "http://127.0.0.1:8765"


def _admin(url: str, tool: str, body: dict) -> int:
    import httpx

    r = httpx.post(f"{url.rstrip('/')}/admin/{tool}", json=body, timeout=None)
    print(json.dumps(r.json(), indent=2))
    return 0 if r.status_code < 400 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m wsp.lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="load the model and serve the lab + 3D viewer")
    s.add_argument("--model", default="Qwen/Qwen3-4B", help="HF id, local path, or 'tiny-qwen3'")
    s.add_argument("--device", default=None)
    s.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    s.add_argument("--budget", type=int, default=50)
    s.add_argument("--max-generate", type=int, default=512)
    s.add_argument("--max-read", type=int, default=1024)
    s.add_argument("--max-new-tokens", type=int, default=2048)
    s.add_argument("--batch-size", type=int, default=0, help="0 = auto (64 on CUDA, 16 otherwise)")
    s.add_argument("--practice", default="math:train", help="math:train | gsm8k:train | jsonl:<path> | toy")
    s.add_argument("--eval", default="math500", help="math500 | gsm8k:test | math:test | jsonl:<path> | toy")
    s.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode for generations")
    s.add_argument("--run-dir", default="runs/lab")
    s.add_argument("--base-device", default=None, help="'cpu' keeps the pristine copy off the accelerator")
    s.add_argument("--val", default="math:val", help="per-edit validation split (disjoint from --eval)")
    s.add_argument("--val-n", type=int, default=128, help="validation items for paired answer-NLL; 0 disables")
    s.add_argument("--val-gen-n", type=int, default=0, help="also generate+grade this many per weight state")
    s.add_argument("--val-visible", action="store_true", help="let the analyst see validation results")
    s.add_argument("--attn", default="auto", choices=["auto", "sdpa", "eager", "flash_attention_2"])
    s.add_argument("--compile", action="store_true", help="static KV cache + torch.compile (CUDA)")
    s.add_argument("--gen-backend", default="hf", choices=["hf", "vllm"],
                   help="vllm: CUDA-only fast generation with live weight sync (experimental)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    v = sub.add_parser("validation", help="per-edit validation history (operator)")
    v.add_argument("--url", default=DEFAULT_URL)

    e = sub.add_parser("eval", help="held-out evaluation (operator only)")
    e.add_argument("--url", default=DEFAULT_URL)
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--checkpoint", default=None, help="tag or 'base'; default: live weights")
    e.add_argument("--split", default="eval", choices=["eval", "practice"])
    e.add_argument("--max-new-tokens", type=int, default=None)

    x = sub.add_parser("export", help="write delta.safetensors (+ full model)")
    x.add_argument("--url", default=DEFAULT_URL)
    x.add_argument("--out", default=None)
    x.add_argument("--full-model", action="store_true")

    b = sub.add_parser("budget", help="change the analyst's forward-pass budget")
    b.add_argument("total", type=int)
    b.add_argument("--url", default=DEFAULT_URL)
    b.add_argument("--reset-used", action="store_true")

    sub.add_parser("mcp", help="run the analyst MCP server on stdio (uses WSP_LAB_URL)")

    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from .server import create_app
        from .session import Lab, LabConfig

        cfg = LabConfig(
            model_id=a.model, device=a.device, dtype=a.dtype, budget=a.budget,
            max_generate=a.max_generate, max_read=a.max_read, max_new_tokens=a.max_new_tokens,
            batch_size=a.batch_size, practice=a.practice, eval=a.eval, thinking=a.thinking,
            run_dir=a.run_dir, base_device=a.base_device, val=a.val, val_n=a.val_n,
            val_gen_n=a.val_gen_n, val_visible=a.val_visible, attn=a.attn, compile=a.compile,
            gen_backend=a.gen_backend,
        )
        print(f"loading {cfg.model_id} ...", file=sys.stderr)
        lab = Lab(cfg)
        print(f"session {lab.session} -> {lab.dir}\nviewer  http://{a.host}:{a.port}/", file=sys.stderr)
        uvicorn.run(create_app(lab), host=a.host, port=a.port, log_level="warning")
        return 0
    if a.cmd == "eval":
        return _admin(a.url, "evaluate", {"n": a.n, "seed": a.seed, "checkpoint": a.checkpoint,
                                          "split": a.split, "max_new_tokens": a.max_new_tokens})
    if a.cmd == "validation":
        return _admin(a.url, "validation", {})
    if a.cmd == "export":
        return _admin(a.url, "export", {"out_dir": a.out, "full_model": a.full_model})
    if a.cmd == "budget":
        return _admin(a.url, "set_budget", {"total": a.total, "reset_used": a.reset_used})
    if a.cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
