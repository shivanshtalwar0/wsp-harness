# 05 · The lab: Claude Opus / Fable as the analyst

`wsp.lab` rebuilds the paper's Experiment-8 harness so a frontier model can play Astra: it
reads and edits a live model's weights directly, with a metered budget of forward passes, and is
scored on a held-out set it never sees. No training, no gradients.

```
        operator (you)                               analyst (Claude Opus / Fable)
  ┌──────────────────────────┐                 ┌──────────────────────────────────┐
  │ python -m wsp.lab eval   │                 │ claude -p … --tools "" (no shell) │
  │ python -m wsp.lab export │                 │   only mcp__wsp-lab__* tools      │
  │ browser: 3D viewer       │                 └───────────────┬──────────────────┘
  └────────────┬─────────────┘                                 │ MCP (stdio)
               │ HTTP /admin, /viz                             ▼
               │                            ┌──────────────────────────────────┐
               └──────────────────────────► │ wsp.lab.mcp_server  → HTTP /api   │
                                            └───────────────┬──────────────────┘
                                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ wsp.lab serve   (one process, one GPU, one lock)                             │
  │   model (bf16, live)   base copy   sandbox (W, acts, helpers)   budget       │
  │   practice split (analyst)          eval split (operator only)   ledger.jsonl│
  └──────────────────────────────────────────────────────────────────────────────┘
```

## Why a new harness instead of the 4-bit one

The 27B tooling in `wsp/` edits a 4-bit MLX checkpoint offline: dequantize, edit, requantize,
write 16 GB, reload in another process, evaluate. Each round trip takes minutes and many small
edits fall below one quantization step. That is a poor loop for an analyst that needs to try,
measure, and roll back dozens of times.

The lab keeps a full-precision model resident. An edit is an in-place tensor write and takes
effect on the next forward pass. Rolling back means copying from a pristine base held in memory.
Qwen3-4B (36 layers, hidden 2560, MLP 9728, 32 query / 8 KV heads) is 5× the paper's 0.8B model
and fits easily in 48 GB of unified memory with its base copy.

## Run it

```bash
pip install -e ".[lab,dev]"

# 1. the lab (downloads ~8 GB the first time)
python -m wsp.lab serve --model Qwen/Qwen3-4B --budget 50
#    viewer: http://127.0.0.1:8765/

# 2. baseline on the held-out split, before the analyst touches anything
python -m wsp.lab eval --n 200

# 3. the analyst
scripts/run_analyst.sh                                   # Opus, default objective
scripts/run_analyst.sh "Reduce truncated answers on practice math" claude-fable-5-2

# 4. score what it left behind, against base, and export
python -m wsp.lab eval --n 500 --checkpoint final
python -m wsp.lab eval --n 500 --checkpoint base
python -m wsp.lab export --full-model
```

No GPU or download handy? `--model tiny-qwen3 --device cpu --dtype float32 --practice toy
--eval toy` gives a 4-layer random model with the same names and shapes pattern, offline.

To drive it interactively from Claude Code instead of headless:
`claude --mcp-config lab.mcp.json --append-system-prompt-file prompts/analyst.md`.

## What the analyst gets

| MCP tool | metered | notes |
|---|---|---|
| `lab_info` | – | architecture, tensor families, limits, capture sites, sandbox help, rules |
| `python` | – | persistent namespace: `W` (live weights), `acts`, `last_gen`, `last_read`, `tok`, `cfg`, numpy, torch; helpers `cos`, `logit_lens`, `logit_lens_neg`, `top_singular`, `token_ids`, `token_vec`, `to_np` |
| `practice_problems` | – | practice split with answers |
| `forward_generate` | 1 pass | ≤512 prompts; graded; accuracy + Wilson CI, hit-limit rate, loop rate; optional `steer` |
| `forward_read` | 1 pass | ≤1,024 texts; NLL + next-token top-k; captures `resid`, `mlp_act`, `mlp_out`, `attn_out`, `attn_heads`, raw modules into `acts`; optional `steer` |
| `apply_patch` | – | the repo's Patch IR (`project_out`, `add_rank1`, `col_gain`, `row_gain`, `zero_cols`); `"$var"` pulls vectors from the sandbox; reports fidelity; `control="random"` |
| `weight_diff` | – | per tensor: elements changed, relative norm, columns/rows touched, edit rank |
| `checkpoint` / `restore` / `list_checkpoints` | – | `restore("base")` is the original model |
| `note` | – | hypothesis / observation / decision / verdict, shown live in the viewer |
| `budget` | – | passes used and left |

Operator-only (HTTP `/admin`, CLI): `evaluate` (unmetered, optional checkpoint, never visible to
the analyst), `export` (`delta.safetensors`, `patches.json`, `notes.json`, `diff.json`, optional
full HF model), `set_budget`.

**Data.** Defaults follow the paper: practice on MATH train (`math:train`, 7,498 problems),
evaluate on MATH-500, which is drawn from the MATH test split, so the two never overlap.
GSM8K (`--practice gsm8k:train --eval gsm8k:test`) is available but nearly saturated for
Qwen3-4B: 8/8 practice problems correct, 265 tokens on average, none truncated. That leaves no
headroom to measure. Your own tasks go in via `jsonl:<path>`. The grader compares the last
`\boxed{}` numerically or as normalized LaTeX; elaborate answers (matrices, intervals) can be
misgraded, so read the `eval-*.json` rows before trusting small differences.

The pass definitions follow the paper (one pass = up to 512 generations or 1,024 reads). Long
thinking-mode generations on a laptop are slow; `--max-new-tokens` caps practice generations
(default 2,048) and `--thinking` turns on Qwen3's thinking mode.

## The controls are built in

`docs/02-method.md` requires two controls before an edit counts. Both are one argument away:

- **Random direction (control B):** `apply_patch(patch, control="random")` replaces every
  direction with a random unit vector: same op, columns and alpha. (A random `project_out`
  removes less energy than a targeted one; that is the point of the control.)
- **Steering (control C, Goedecke's critique):** `forward_generate(..., steer=[{"layer": L,
  "vector": "$d", "mode": "project_out"}])` applies the same direction at inference time with no
  weight change. If steering matches the edit, the edit is a baked-in steering vector.

## The 3D viewer

`http://127.0.0.1:8765/`, served by the lab itself, so it shows the live state:

- one slab per layer; one point per MLP neuron (Qwen3-4B: 36 × 9,728 = 350,208 points) and one
  cube per attention head;
- **weights**: ‖down_proj column‖, which is how much each neuron can write;
- **edits**: ‖ΔW‖ against the base. Lights up the moment the analyst edits;
- **activations**: mean |activation| from the last `forward_read` that captured `mlp_act`;
- click a neuron to see its write norm, edit state, last activation, and the tokens its column
  promotes and suppresses (logit lens);
- the right panel streams every analyst call and note, plus the pass budget.

## Demo round and the Pages site

`scripts/demo_session.py --url <lab>` runs one scripted round: baseline, capture, a recipe edit
in layers 15/19, re-measure, and the random control. It exercises the plumbing; it is not an
analyst run. On Qwen3-4B with 16 MATH practice items it gave baseline 11/16, edit 12/16, random
control 11/16. The intervals overlap, so this is a null result.
`scripts/build_site.py --url <lab> --out site` snapshots a running lab into `site/demo/`, a
static replay of the viewer. `.github/workflows/pages.yml` publishes `site/` to GitHub Pages.

## Ledger and artifacts

Each session writes `runs/lab/<session>/`: `config.json`, `ledger.jsonl` (every call, args,
result summary, timing), `checkpoints/*.safetensors`, `eval-*.json` (full rows), and `export/`.
The analyst's raw transcript goes to `runs/analyst/*.jsonl`.

## Limits, stated plainly

1. **This is not distillation.** The analyst's own knowledge does not flow into Qwen's weights;
   only the edits it reasons its way to do. The paper's evidence is that such edits remove
   obstacles hiding an ability (loops, truncation, format) and do not add missing abilities
   (Gemma-270M control). Expect gains where the baseline shows a suppressor, and nulls elsewhere.
   If the goal is to transfer a frontier model's skill wholesale, that is distillation or
   fine-tuning, and the lab is the right tool for diagnosing *what* to train, not for replacing it.
2. **The sandbox is a convenience boundary, not a security one.** The analyst's Python runs in the
   lab process. The rules (no gradients, no direct model calls) are enforced by
   `torch.no_grad()`, by what is in the namespace, and by the ledger; a determined analyst could
   reach around them. Audit `ledger.jsonl`.
3. **bf16 resolution.** Edits smaller than ~0.4% of a weight's magnitude round away in bf16.
   `apply_patch` reports `fidelity`; run with `--dtype float32` (≈16 GB for 4B) when it matters.
4. **One model, one lock.** Calls are serialized. The viewer polls; it never blocks the analyst
   for long, but a 512-prompt generation pass will hold the lock for minutes.
