# 03 · Runbook

Everything below was executed on this machine (macOS 26.6, M-series, 48 GB RAM, Python 3.10,
46 GiB free disk) against
`~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit`.

## Install

```bash
cd ~/myprojects/weight-space-programming
python -m venv .venv && . .venv/bin/activate
pip install numpy safetensors ml_dtypes pyyaml pytest
pip install mlx              # optional, only to cross-check the quantizer
pytest -q                    # 39 passed
```

`ml_dtypes` is not optional: the checkpoint's scales and biases are bf16, and vanilla numpy +
safetensors raises `data type 'bfloat16' not understood` without it. This is the single most
likely place to lose an hour.

## Model

```bash
export M=~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit

python -m wsp.cli manifest "$M"      # 27.357B, 2180 tensors, 498 quantized, 16.05 GB
python -m wsp.cli verify-mlx         # dequant err <1e-8, pack roundtrip identical
```

`verify-mlx` is the trust anchor for everything else: it compares our numpy unpack/dequantize
against `mlx.core.quantize/dequantize` on random data. If it passes, the affine maths is right.

## Weight-side commands (no server needed)

| Command | What it does | Measured |
|---|---|---|
| `manifest` | inventory | 0.4 s |
| `atlas --layers 15,19 --suffix mlp.down_proj` | dequantize + stats for 2×89M-param tensors | 3.2 s |
| `analytic --tensor <base>` | power iteration → baseline patch | ~2 s |
| `apply --dry-run` | validate patch against shapes/names | 1 s |
| `verify` | recompute the edit and requantize it, without writing | ~15 s / tensor |
| `apply --out` | write patched checkpoint | 31 s, 15 GB out, ~5.4 GB × 2 RAM |

Example atlas output on the real model:

```
layers.15.mlp.down_proj [5120, 17408] dominant-direction energy=0.0517
   top rows by norm: [3994 → 2.89, 3532 → 1.93, 1689 → 1.75]      median ≈ 1.4
layers.19.mlp.down_proj [5120, 17408] dominant-direction energy=0.0611
   top rows by norm: [3994 → 5.96, 1689 → 1.71, 2316 → 1.70]
   grid: scale median 0.00342, degenerate groups 0, half the scales negative
```

Row 3994 is a massive-activation row in both layers. That is a lead, not a result: it is the
kind of outlier the atlas exists to surface.

## Server

```bash
lms server start
lms ls                                  # confirm the 27B is registered
curl -s http://127.0.0.1:1234/v1/models | head -40
```

In the LM Studio UI, **raise the context length to ≥ 32768**. The default 8k fails on the agent
system prompt plus a tool observation, and the failure looks like a truncated reply rather than
an error.

For the `llm` CLI from Simon Willison's post, also export a long timeout:

```bash
LMSTUDIO_TIMEOUT=900 llm -f <prompt-url> -m lmstudio/qwen/qwen3.8-27b
```

Our harness has its own 900 s request timeout and 3 attempts.

## Serving a patched checkpoint

`mlx-lm` is the right server for experiments: one binary, so base and patched differ only in
weights.

```bash
python -m mlx_lm server --model "$M" --port 8101 --host 127.0.0.1 --log-level WARNING
```

**The `model` field in a request must be exactly the `--model` path you passed.** The server
treats an unknown id as a Hugging Face repo and returns a confusing `404 ... Repository Not Found
for url: https://huggingface.co/api/models/<id>`. Take the id from `curl /v1/models`.

Also: `mlx_lm` ignores `--max-tokens` from the command line in favour of the request's value, and
a thinking model can spend the entire budget on reasoning and return an empty `content`. If
outputs look empty, raise `--max-tokens` before assuming the weights are broken.

## Memory and disk

A single 27B at 4-bit is ~15 GB of unified memory. `wsp apply` writes another 15 GB. Doing both at
once pushed this laptop to 16 GB of swap, 4.9 GB free disk, and 10-minute turns. Run one heavy job
at a time and check `df -h .` before `apply`; a patched copy can always be regenerated in ~31 s
from a few-hundred-KB patch file.

## Eval

```bash
python -m wsp.cli eval --base-url http://127.0.0.1:1234/v1 \
       --model qwen/qwen3.8-27b --out runs/baseline.json --max-steps 25
```

Rough costs at 4-bit on this laptop, from the model's behaviour on long prompts: 15–25 tok/s
generation, so a 20-step episode with 30k-token prompts is **10–25 minutes**. A 3-task,
20-episode baseline is therefore an overnight job, not a coffee break. Plan the experiment
budget around that: fewer conditions, more episodes per condition.

Speed levers, in order of safety: smaller step cap, shorter observations (the harness already
truncates at 8k chars), fewer tasks, then quantization — never *raise* quant while editing
weights, because the edit's noise floor rises with it.

## Grading harness details

- Each episode copies `evals/tasks/<id>/repo/` into a temp dir; hidden tests are copied in only
  at grading time, so an agent cannot read them.
- `shell` runs with the venv's `python` first on `PATH`, 60 s timeout, `HOME` pinned to the
  workspace so nothing leaks into your dotfiles.
- `submit()` is not required for grading; the tests run either way.
- The sandbox is a directory and an env scrub, **not** a security boundary. Do not point this at
  repos you care about.

## Gotchas

| Symptom | Cause |
|---|---|
| `data type 'bfloat16' not understood` | missing `ml_dtypes` |
| `direction has N entries, tensor has M rows` | `project_out` directions are indexed by **rows** (output features), not columns |
| Patch reports 0 codes changed | edit smaller than the quantization floor — see `docs/01-feasibility.md` §3b |
| `apply` refuses to write | `--out` resolves to the source checkpoint |
| Patch "worked" but nothing changed | run `verify`; if fidelity < 0.25 the direction is too diffuse for a 4-bit grid |
| Eval hangs | LM Studio context too small, or model not loaded; check `/v1/models` first |
| Probe says `no_verify` | that is a harness result, not a weight result — fix the harness |
| `python -m wsp.cli` not found | run from the repo root, or `pip install -e .` |

## Reproducing the claims in the README

```bash
pytest -q                                          # 39 passed
python -m wsp.cli manifest "$M"                    # 27.357B / 498 quantized
python -m wsp.cli verify-mlx                       # bit-exact vs mlx
python -m wsp.cli atlas "$M" --layers 15,19 --suffix mlp.down_proj
python -m wsp.cli analytic "$M" --tensor language_model.model.layers.15.mlp.down_proj
```
