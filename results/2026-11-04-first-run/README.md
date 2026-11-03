# 2026-11-04 — first run: base vs one weight edit

Raw artifacts from the first real evaluation. Narrative and analysis:
[`docs/04-results.md`](../../docs/04-results.md).

## Headline

```
base     2/3   Wilson 95% [0.21, 0.94]
patched  2/3   Wilson 95% [0.21, 0.94]     not distinguishable at n=3
```

The edit is **behaviourally active** (3/5 greedy prompts byte-identical, `fibonacci` and a prose
opening diverge, the arithmetic answer does not) and **capability-neutral** on this suite. No
failure probe fired in either condition, so the WSP loop had no suppressor to target — this is
the paper's Gemma-270M control, not its Qwen3.5-0.8B result.

## Conditions

| | |
|---|---|
| model | `Qwen3.8-27B-MLX-4bit` (27.357B params, 4-bit affine, group 64) |
| runtime | `mlx-lm` server, one binary for both arms |
| sampling | temperature 0.6, top-p 0.95, ≤1024 tokens/turn |
| budget | 12 steps/task, 3 tasks, hidden-test grading |
| machine | Apple silicon, 48 GB RAM, ~8 tok/s decode |

Patch: `project_out` of the dominant direction (power iteration, 24 iters) of
`language_model.model.layers.19.mlp.down_proj`, alpha 1.0, all 17,408 columns.
203,700 of 89,128,960 codes changed (0.23% of that tensor, 0.0007% of the model);
fidelity `0.853` — 85.3% of the intended removal survived 4-bit requantization.

## Files

| file | what it is |
|---|---|
| `eval-base.json` | 3 episodes on the unmodified checkpoint, with full trajectories |
| `eval-patched.json` | same 3 tasks, same settings, patched checkpoint |
| `inert-base.json` | 5 fixed prompts, temperature 0, base model |
| `inert-patched.json` | same 5 prompts, patched model |
| `patch.json` | the weight edit itself — 5,120 floats, reproducible in ~2 s |
| `atlas.json` | per-tensor statistics the analyst was given for layers 15 and 19 |
| `prompt.txt` | the rendered analyst prompt (evidence + atlas + JSON contract) |

Episode JSONs contain every assistant turn, tool call, observation and the final
hidden-test output, so `wsp probes <file>` and `wsp.patch` reprocessing work offline.

## Reproduce

```bash
M=~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit

python -m mlx_lm server --model "$M" --port 8101 --log-level WARNING
python -m wsp.cli eval --base-url http://127.0.0.1:8101/v1 --model "$M" \
       --max-steps 12 --out runs/eval-base.json

python -m wsp.cli analytic "$M" --tensor language_model.model.layers.19.mlp.down_proj \
       --out runs/patch.json
python -m wsp.cli verify   "$M" runs/patch.json
python -m wsp.cli apply    "$M" runs/patch.json --out runs/patched-L19

python -m mlx_lm server --model "$PWD/runs/patched-L19" --port 8102 --log-level WARNING
python -m wsp.cli eval --base-url http://127.0.0.1:8102/v1 --model "$PWD/runs/patched-L19" \
       --max-steps 12 --out runs/eval-patched.json

python scripts/compare_runs.py runs/eval-base.json runs/eval-patched.json
```

The `model` field must be exactly the `--model` path; `mlx-lm` otherwise treats it as a Hugging
Face repo id and returns a misleading 404.

## What these numbers do not show

n=3. The interval on 2/3 is 0.21–0.94, so nothing here distinguishes 2/3 from 3/3, and the base
model is already at/near the ceiling of these tasks. This is a pipeline smoke test with real
weights, not a capability measurement. See `docs/04-results.md` § *Where the headroom actually is*.
