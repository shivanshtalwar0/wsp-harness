# 04 · First results: base vs one weight edit

Run on 2026-11-03/04 on the machine in `docs/03-runbook.md`. Everything here is from
`runs/eval-base.json`, `runs/eval-patched.json`, `runs/inert-base.json`, `runs/inert-patched.json`
(all gitignored; regenerate with the commands below).

## Setup

| | base | patched |
|---|---|---|
| checkpoint | Qwen3.8-27B-MLX-4bit | same + `project_out` on `layers.19.mlp.down_proj` |
| edit | — | dominant direction (power iteration), alpha 1.0, all 17,408 columns |
| codes changed | — | 203,700 of 89,128,960 (**0.23%** of that tensor, 0.0007% of the model) |
| fidelity (`wsp verify`) | — | **0.853** — 85.3% of the intended direction removal survived requantization |
| runtime | `mlx-lm` server, port 8101 | same binary, same sampler |
| sampling | T=0.6, top-p 0.95, ≤1024 tok/turn | identical |
| budget | 12 steps/task, 3 tasks | identical |

Only the weights differ between the two conditions. That is the whole point of using one
server binary instead of two front ends.

## Resolution rate: unchanged

```
                 resolved   Wilson 95%      mean steps   mean wall
base             2/3        [0.21, 0.94]    10.0         141 s
patched          2/3        [0.21, 0.94]    11.0         141 s
```

| task | base | patched |
|---|---|---|
| 01 · pagination off-by-one | PASS 8 steps | PASS 9 steps |
| 02 · keyword-only API change | fail 12 steps | fail 12 steps |
| 03 · numeric sort order | PASS 10 steps | PASS 12 steps |

`scripts/compare_runs.py` prints the verdict itself: **not distinguishable at n=3**. The
interval on 2/3 is 0.21–0.94; it would take ~20 episodes per condition to see a 20-point effect.

## The edit is not inert

Six prompts, temperature 0, same 640-token cap:

| prompt | exact match | text similarity |
|---|---|---|
| `def fibonacci(n):` | no | 0.51 |
| `class LRUCache: …` | yes (both empty) | 1.00 |
| "The reason gradient descent works is" | no | 0.36 |
| "Compute 17 * 23 step by step." | **yes** | 1.00 |
| "Reply with exactly the word: ok" | **yes** | 1.00 |

3/5 identical, and the two that differ differ a lot: different implementation of `fibonacci`,
different opening for the prose prompt. Meanwhile the arithmetic answer and the one-token
constraint are byte-identical.

That is a coherent picture: removing one direction from one of 64 layers perturbs free-form
continuation, and does not touch a deterministic arithmetic path or a 1-token emission. The edit
is **behaviourally active and capability-neutral** on this suite.

## Why task 2 failed in both conditions

It is budget, not ability. The patched run's trajectory:

```
00 list_dir .        → billing/ conftest.py tests/
01..07 read billing/{__init__,api,calc}.py, tests/test_calc.py
08 write billing/calc.py          → wrote 0 bytes        ← empty_write
09 write billing/calc.py          → 415 bytes
10 shell python -m pytest -q      → 1 failed: invoice_total() takes 1 positional argument
11 read billing/api.py            ← found the caller that needs updating
12                                ← step cap
```

One more step would likely have completed it. The 0-byte write at step 08 is a real failure mode
that the probe set did not cover — the tool call parsed, but the payload was empty, most likely
because a thinking block consumed the turn budget. `wsp.probes` now has an `empty_write` probe
for it.

## No suppressor fired

```
probes (prevalence):  none, in either condition
weight-candidates:    []
```

Every episode read before writing, ran the tests, and terminated on its own or on the step cap.
There was no loop, no tool-format drift, no false success. **The WSP loop had no target.**

This is the paper's Gemma-3-270M control, not its Qwen3.5-0.8B result: a model that is not
blocked by a single behaviour. Per the paper's own rule — *"editing weights could remove an
obstacle that hid an ability, but it could not add an ability the model lacked"* — the honest
expectation from here is zero, and zero is what we measured.

## What this shows and what it does not

**Does show**

- the pipeline runs end to end on a real 27B: patch → apply → requantize → serve → grade;
- the patch survives 4-bit requantization (85%), which was the main technical unknown;
- the edit changes generated text, so any future null is not "the file didn't change";
- the harness grades correctly and the probes fire on the failure modes they are designed for;
- one thing that breaks: **write payloads can be silently empty** (now probed).

**Does not show**

- anything about capability. 2/3 vs 2/3 at n=3 is not a measurement, it is a smoke test.
- that the direction is meaningless — only that this suite cannot see it.
- that a different direction, layer or alpha would do nothing.

## Where the headroom actually is

If the goal is a real gain, the ordering by expected value is:

1. **Raise the step budget** (12 → 20). Task 2 was one step short; a step cap is a harness
   property masquerading as a model property.
2. **Harden the tasks.** A suite where the base model scores 2/3 has no room. Harder,
   longer-horizon tasks (multi-file refactors that do not fit in 12 steps) are where suppressors
   like early termination and rewrite thrash appear.
3. **Then** look for a suppressor, and require it in >30% of unresolved episodes.
4. Only then spend a weight edit, with the random-direction control.

Regenerate everything:

```bash
lms server start   # or: python -m mlx_lm server --model "$M" --port 8101
python -m wsp.cli eval --base-url http://127.0.0.1:8101/v1 --model "$M" \
       --max-steps 12 --out runs/eval-base.json
python -m wsp.cli analytic "$M" --tensor language_model.model.layers.19.mlp.down_proj --out runs/patch.json
python -m wsp.cli apply "$M" runs/patch.json --out runs/patched-L19
python -m wsp.cli verify "$M" runs/patch.json
python -m mlx_lm server --model "$PWD/runs/patched-L19" --port 8102   # id must equal this path
python -m wsp.cli eval --base-url http://127.0.0.1:8102/v1 \
       --model "$PWD/runs/patched-L19" --max-steps 12 --out runs/eval-patched.json
python scripts/compare_runs.py runs/eval-base.json runs/eval-patched.json
```

## Resource notes

- ~8 tok/s decode on this laptop; a 3-task × 12-step suite is ~7 minutes.
- `apply` writes 15 GB. **Do not run it while an eval is generating.** Doing that pushed swap to
  16 GB, drove free disk to 4.9 GB, and stretched single turns from 15 s to 10+ minutes. Run one
  heavy job at a time; `df` before `apply`.
- The patched copy can be deleted and regenerated in 31 s; the 169 KB patch JSON is the artifact
  that matters, not the 15 GB of duplicated weights.
