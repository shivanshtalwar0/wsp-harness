# weight-space-programming

**`wsp`** — Weight-Space Programming experiments on **Qwen3.8-27B**, aimed at its **agentic / software-engineering** behaviour.

Built from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, Nov 2026) and the follow-up write-up [Qwen 3.8 27B on my laptop can interpret model weights, too](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/).

---

## Read this before you run anything

The goal you asked for was: edit Qwen3.8-27B's weights, iteratively, until its agentic and
software-engineering skill reaches Claude Opus 5.5 / Fable level. Here is the honest arithmetic.

| | Qwen3.8-27B | Claude Opus 5.5 |
|---|---|---|
| SWE-bench Pro | 61.7 | 89.9 |
| Terminal-Bench 2.1 | 73.0 | ~78 (Opus 4.6 Max figure) |

That is a **~28-point capability gap**. The paper's own decisive result says weight editing
does not close gaps like that:

> *"Here, Astra could not improve it. ... Editing weights could remove an obstacle that hid an
> ability, but it could not add an ability the model lacked."*
> — the Gemma 3 270M control in Experiment 8

The Qwen3.5-0.8B win (MATH-500 7.9% → 48.1%) was **loop removal**: the model could already do
the maths, and one direction in the MLP of layers 15/19 was stopping it from finishing. 93% of
its answers ran to the token limit; after the edit 33% did.

So this repo is not a path to Opus 5.5. It is the strongest *honest* version of your idea:

> Find a behaviour that is **suppressing** Qwen3.8-27B's existing agentic ability, prove it is
> suppressing it, remove it in weight space, and measure the recovery — with controls that stop
> us fooling ourselves.

Realistic ceiling, stated up front: **a few points of recovered score**, concentrated in
long-horizon failures (loops, early termination, tool-format drift). If an edit produces 28
points, the most likely explanation is an eval bug, not a discovery.

---

## What is actually built and verified

- **Quant-aware checkpoint I/O** — the 27B is *only* 4-bit affine (498 quantized matrices, no
  bf16 weights anywhere), so WSP here is a different problem from the paper's 0.8B bf16 model.
  Packing/affine maths is verified bit-exact against `mlx.core` (`wsp verify-mlx`).
- **A patch IR** — named ops (`project_out`, `add_rank1`, `col_gain`, `row_gain`, `zero_cols`,
  `group_gain`, `group_shift`), JSON, hashed, reversible, with a code-level diff.
- **Analyst tooling** — a weight *atlas* (a 27B cannot be pasted into a context window) plus a
  no-LLM analytic baseline built on power iteration, so any LLM proposal has something to beat.
- **An agent harness** — text tool protocol, sandboxed workspace, hidden-test grading, works
  against any OpenAI-compatible endpoint (LM Studio, llama.cpp, OpenRouter).
- **Failure probes** — nine detectors that turn trajectories into a ranked list of candidate
  "suppressors", each tagged with whether it is plausibly a *weight* problem or a harness one.

Verified on the real checkpoint on this machine:

```
$ python -m wsp.cli manifest ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit
27.357B params · 2180 tensors · 498 quantized · 16.05 GB · 64 layers (48 linear-attn / 16 full)

$ python -m wsp.cli atlas <model> --layers 15,19 --suffix mlp.down_proj
layers.15.mlp.down_proj [5120, 17408] energy=0.0517   row 3994 norm 2.89 (next: 1.93)
layers.19.mlp.down_proj [5120, 17408] energy=0.0611   row 3994 norm 5.96 (next: 1.71)
```

A real patch, applied to the real checkpoint and checked three ways:

```
apply  → 203,700 of 89.1M codes changed (0.23%); 31 s
verify → residual along the targeted direction: 85.3% removed, fidelity 0.85 ("landed")
mlx-lm → the patched 15 GB checkpoint loads and generates (`def fibonacci(n)` unchanged at temp 0)
```

43 unit tests, including a full agent episode end-to-end against a scripted model.

### First run (base vs one weight edit)

Three graded tasks, 12-step cap, both conditions served by the same `mlx-lm` binary:

```
base     2/3  [0.21, 0.94]   patched  2/3  [0.21, 0.94]      not distinguishable at n=3
```

The edit **is** behaviourally active — 3/5 greedy prompts identical, `fibonacci` and a prose
opening change, the arithmetic answer does not — but it buys nothing on this suite, and **no
failure probe fired in either condition**, so the WSP loop had no target. That is the paper's
Gemma-270M control, not its Qwen-0.8B result. Details and the failure analysis:
[`docs/04-results.md`](docs/04-results.md).

---

## Quickstart

```bash
cd weight-space-programming
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install mlx                      # optional: only for `verify-mlx`
pytest -q                            # 39 passed

export M=~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit

python -m wsp.cli manifest "$M"
python -m wsp.cli verify-mlx
python -m wsp.cli atlas    "$M" --layers 15,19 --suffix mlp.down_proj --out runs/atlas.json
python -m wsp.cli analytic "$M" --tensor language_model.model.layers.15.mlp.down_proj --out runs/patch.json
python -m wsp.cli apply    "$M" runs/patch.json --dry-run
python -m wsp.cli verify   "$M" runs/patch.json     # did the 4-bit grid keep the edit?
```

Measure the baseline (start LM Studio's server first):

```bash
python -m wsp.cli eval --base-url http://127.0.0.1:1234/v1 \
       --model qwen/qwen3.8-27b --out runs/baseline.json
python -m wsp.cli probes runs/baseline.json
python -m wsp.cli prompt "$M" --probes runs/baseline.json --layers 15,19 --out runs/prompt.txt
```

Then the loop in [`docs/02-method.md`](docs/02-method.md).

---

## Layout

```
wsp/
  quant.py     affine 4-bit pack/unpack/dequant/requant (verified against mlx)
  weights.py   ModelArchive: manifest, apply, code-level diff
  patch.py     the Patch IR and its ops
  probes.py    nine failure detectors -> ranked suppressors
  analyst.py   weight atlas, analytic baseline, LLM proposal + validation
  verify.py    patch fidelity: what the 4-bit grid kept of the edit
  cli.py       the commands above
evals/
  harness.py   agent loop, sandbox, hidden-test grading
  mock.py      scripted models for offline tests
  tasks/       three small graded repos (bugfix, multi-file API change, ordering)
tests/         39 tests
docs/
  00-paper-map.md   what each paper experiment shows and what transfers
  01-feasibility.md the arithmetic, the critique, and the bounded goal
  02-method.md      the iteration protocol and acceptance criteria
  03-runbook.md     commands, timings, gotchas
  04-results.md     first base-vs-patched run, and why it is a null result
```

## Honest limitations

1. **Three tasks is not a benchmark.** It is a smoke test. The method doc requires ≥20 episodes
   per condition and paired statistics before believing anything.
2. **4-bit edits are lossy.** A `project_out` is rounded to one quantization step; on a 0.003
   scale the residual is real. `group_gain` is exact by contrast.
3. **This may be a steering vector.** Goedecke's critique of the original Qwen result — that it
   is direction ablation, a known technique — applies with full force here. The method doc
   includes the control that distinguishes them.
4. **Weight editing cannot add capability.** If the probes say the failures are ordinary coding
   errors, the answer is training, not WSP. That is a legitimate result.
