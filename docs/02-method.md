# 02 · The iteration protocol

One round = one hypothesis, one patch, one measurement, one verdict. Nothing is accepted on the
strength of an LLM's own report.

```
S0 inventory ──> S1 baseline ──> S2 rank suppressors ──> S3 pick one
                     ▲                                      │
                     │                                      ▼
S8 verdict <── S7 accept/reject <── S6 re-eval + controls <── S5 apply <── S4 propose
```

Every round writes a directory:

```
experiments/R03-loop-suppressor/
  hypothesis.md        one paragraph, written before the run
  probes.json          baseline probe ranking
  atlas.json           tensors considered
  patch.json           the edited patch (float-space ops)
  patch-control.json   random-direction control, same op/shape
  eval-before.json     n episodes, seed logged
  eval-after.json
  eval-control.json
  verdict.md           accept / reject / null, with the numbers
```

---

## S0 · Inventory

```bash
python -m wsp.cli manifest "$M"
```

Record: params, layer layout (48 linear-attention / 16 full-attention), which matrix families
exist, and the 4-bit grid. Baked in on this machine:

```
27.357B params · 2180 tensors · 498 quantized · 16.05 GB · 64 layers
per layer: mlp.{gate,up,down}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj} | self_attn.{q,k,v,o}_proj
```

**Layer choice matters more than usual.** The paper's target was `mlp.down_proj` of layers 15 and
19 — the layers whose MLP writes the residual direction that caused looping. In a hybrid model,
the analogue is not automatically layers 15/19: 48 of the 64 layers are gated linear attention
(the Qwen3-Next lineage), so the residual stream is written by different machinery per layer
type. Start with `mlp.down_proj` of full-attention layers, because that is the family the paper
validated, and use the atlas to check the layer is not degenerate.

```bash
python -m wsp.cli atlas "$M" --layers 15,19,23,27 --suffix mlp.down_proj --out runs/atlas.json
```

## S1 · Baseline

```bash
lms server start                      # LM Studio headless; load the 27B with >=32k context
python -m wsp.cli eval --base-url http://127.0.0.1:1234/v1 --model qwen/qwen3.8-27b \
       --out runs/baseline.json
```

`n = 20` episodes per condition, minimum. Three tasks × 1 sample is a smoke test, not a
measurement — with 3 tasks the Wilson 95% interval on a 66% score is roughly [20%, 94%], which
swallows any edit you could plausibly make.

Rules:

- temperature 0.6 / top-p 0.95, fixed for every condition;
- same task order, same seeds, same step cap;
- store the raw trajectories (`--out` keeps them), because probes and future you both need them.

## S2 · Rank the suppressors

```bash
python -m wsp.cli probes runs/baseline.json
```

Output is a ranked list with `prevalence` and a `lever` per probe. Read it as:

- `lever=weights` → candidate for S3;
- `lever=harness` → fix the harness first. If agents submit without running tests, the honest fix
  is to refuse `submit()` when no test has run, not to rewire a matmul. Harness fixes are
  cheap, reliable, and they change what the remaining failures look like.

Re-run S1 after any harness change. Never compare an edit against a baseline recorded under a
different harness.

## S3 · Pick exactly one suppressor

Gate, from `docs/01-feasibility.md` §5: prevalence >30% of unresolved episodes, asymmetric,
expressible as one op at >0.12 magnitude, not already explained by a steering vector.

Write `hypothesis.md` before touching weights. It must state the *mechanism*, not the metric:

> Good: "Layer-19 `mlp.down_proj` writes a residual direction that dominates after ~12k tokens and
> keeps the model re-planning instead of calling tools. Removing it should cut steps-to-first-edit
> and convert loops into completions, without changing single-turn code quality."
>
> Bad: "Make the model better at coding."

## S4 · Propose a patch (two arms, always)

**Arm A — analytic baseline** (no LLM, seconds):

```bash
python -m wsp.cli analytic "$M" --tensor language_model.model.layers.19.mlp.down_proj \
       --out experiments/R03/arm-a.json
```

Power iteration gives the dominant left singular vector; the patch projects it out of every
column. This is the honest null: *if a 27B LLM cannot beat "remove the top singular direction",
it has not demonstrated weight-space understanding.*

**Arm B — analyst proposal**:

```bash
python -m wsp.cli prompt "$M" --probes runs/baseline.json --layers 15,19 \
       --suffix mlp.down_proj --out runs/prompt.txt
# feed runs/prompt.txt to the analyst (the same 27B, or a stronger model), save its JSON reply
python -m wsp.cli apply "$M" experiments/R03/arm-b.json --dry-run     # validates first
```

The analyst gets an atlas, not the weights — a 27B checkpoint cannot be pasted into 32k tokens.
Validation rejects unknown tensors, wrong direction lengths, and quant-space ops on unquantized
tensors before a single byte is written.

Budget note from the paper: probing quality mattered more than analyst cleverness (10 → 25 → 50
forward passes gave 12.3% → 14.1% → 48.1%). If the atlas is thin, feed more failures, not more
instructions.

## S5 · Apply

```bash
python -m wsp.cli apply "$M" experiments/R03/arm-a.json --out ~/models/qwen38-27b-R03
python -m wsp.cli diff "$M" ~/models/qwen38-27b-R03
```

`diff` reports **codes changed**, not "weights touched" — on a 4-bit checkpoint a patch can claim
0.3% of the model while changing 0.001% of the actual codes. Keep the output. Anything above
~0.5% of codes changed needs justification in `verdict.md`.

Disk: a patched copy needs ~16 GB (one full copy) and `apply` peaks around shard-size RAM
(~5.4 GB × 2). Do not point `--out` at the source directory; the code refuses, but the error
after 20 minutes of copying is annoying.

## S6 · Re-evaluate, with three controls

| Condition | Patch | Question it answers |
|---|---|---|
| A | real direction | did it work? |
| B | **random unit direction**, same op/columns/alpha | is the effect shape-specific or just a perturbation? |
| C | real direction as an **inference-time steering vector** | is this anything more than activation steering? (Goedecke's critique) |
| D | steering with a **random** direction | noise floor for C |

Plus a regression run on tasks the analyst never sees, to catch collateral damage (the paper's
IFEval going *up* is the surprising good case; the usual case is something else going down).

## S7 · Accept / reject

Accept only if, on paired episodes:

- the target probe's prevalence drops materially (e.g. loops 40% → 15%);
- the resolution rate rises with **non-overlapping Wilson 95% intervals** at n ≥ 20;
- control B does not reproduce the gain;
- the regression suite does not lose more than it gains.

Otherwise: reject, keep the trajectory, and note *why*. A rejected hypothesis with a clean
measurement is worth more than an accepted one with a single lucky sample.

## S8 · Iterate

Carry forward: the probe ranking, the atlas, the ledger of what did not work. Two edits at once
makes attribution impossible, and the whole value of this approach over fine-tuning is that the
effect is attributable to one named intervention.
