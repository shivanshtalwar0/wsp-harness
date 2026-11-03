# 00 · What the paper actually shows, and what transfers

Source: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 2 Nov 2026). Logs and checkpoints:
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs),
[RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

## The seven results

| # | Setup | Size | Tools | Result | Transfers to a 27B agent? |
|---|---|---|---|---|---|
| 1 | Which of 4 grid games? | 17,253 w | none | 4/4 correct | no — read-only, small |
| 2 | Which shape does a DDPM draw? | 23,824 w | none | 3/3 | no |
| 3 | Which language? + name every copying head | 7.16 M | python | 3/3, heads match ground truth | **partly** — the read technique (multiply weight matrices to follow a circuit) is the only method here that scales |
| 4 | Fix a PPO car stuck on a ramp | 171,664 w | none | 3.5% of weights edited, 0% → 100% in 1,024 episodes | **yes, in kind** — "edit adds a path, original policy takes over past the obstacle" is the template |
| 5 | Write MinAtar nets from scratch | ~19k w | none | beats PPO@100M in 3/3 | no — too small to be informative here |
| 6 | Write a Go net | 1.48 M | python + fwd passes | @0 passes beats small PPO; @50 passes beats PPO@20M games | **yes** — the harness pattern (weight sandbox + metered forward passes) is exactly what we need |
| 7 | Which of two Gemma-270M is sycophantic? | 270 M | attention patterns | **failed from weights alone**; succeeded with attention | **yes** — and it is a warning: weights alone may not carry the signal |
| 8 | Make Qwen3.5-0.8B better at math | 0.8 B | python + fwd passes | 0.016% of weights; MATH-500 7.9→48.1, GSM8K 13.8→67.1, IFEval 21.0→62.3 | **this is the one to copy** |
| 8b | Same, on Gemma-270M (does not loop) | 270 M | python + fwd passes | **no improvement**, ~+0.5pt noise | **the decisive caveat** |

## Five things that matter more than the headline

**1. The win was loop removal, not capability addition.**
Astra traced looping to one direction written by the MLPs of layers 15 and 19 and subtracted it
from 64 output columns in each (131,072 weights). 93% of answers hit the 32,768-token cap before;
33% after. The maths was in the model; the loop was on top of it.

**2. Forward-pass budget was the bottleneck, not cleverness.**
10 passes: 12.3%. 25: 14.1%. 50: 48.1%. The analyst needed enough probing to *see* the failure
clearly. Budget probing before budgeting elegance.

**3. The negative control is the most useful line in the paper.**
On Gemma-270M, which loops only 8.5% of the time, Astra found ordinary reasoning errors, made
small targeted edits, and moved nothing (6.8 → 7.0 → 7.7 → 7.0). Its own conclusion:
*"Editing weights could remove an obstacle that hid an ability, but it could not add an ability
the model lacked."*

**4. The scale actually demonstrated is 0.8B in bf16 — 34× smaller, and not quantized.**
Our checkpoint is 27.4B in 4-bit affine with hybrid linear/full attention. Both the search
problem (which of 498 matrices?) and the edit problem (grid rounding) are different.

**5. The critique is serious and it shapes the method.**
[Matthew Honnibal](https://honnibal.dev/blog/astra-read-write-misleading) argues the paper
conflates "reads weights with no tools" (novel) with "edits weights with a sandbox" (which is
just a researcher with a known trick). [Sean Goedecke](https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/)
re-derived the Qwen-0.8B fix step by step and concluded it is direction ablation — a steering
vector, a well-established technique — with a diagnosis that a human would plausibly reach.

That critique is testable, and `docs/02-method.md` makes it a required control: the same edit
expressed as an inference-time steering vector, no weights touched. If steering gets the same
gain, the weight edit contributed nothing novel. Run it anyway — it is cheap and it keeps us
honest.

## What we copy, concretely

```
paper (exp 8)                        this repo
---------------------------------    -------------------------------------------------
practice problems (GSM8K+MATH)   ->  agent tasks with hidden tests
metered forward passes           ->  eval episodes, n per condition, logged
"find the loop direction"        ->  probes.py: 9 failure detectors -> ranked suppressors
layers 15/19 MLP down_proj       ->  atlas over mlp.down_proj / out_proj of candidate layers
subtract a direction, 131k w     ->  project_out op, alpha-scaled, column-selected
held-out benchmarks              ->  held-out tasks + regression suite, never seen by the analyst
```
