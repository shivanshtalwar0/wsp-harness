# 01 · The arithmetic of the ask

**The ask.** Edit Qwen3.8-27B's weights, iteratively, until its agentic and software-engineering
skill is at Claude Opus 5.5 / Claude Fable level.

**The verdict.** Not reachable by weight editing, and the paper is the reason to believe that
rather than the reason to doubt it. What *is* reachable is the paper's actual, narrower result:
recovering ability that a specific behaviour is suppressing.

This document is the arithmetic behind both sentences.

---

## 1. How big is the gap

Vendor-reported and aggregator figures; treat as ±2 points.

| Benchmark | Qwen3.8-27B | Claude Opus 5.5 | Gap |
|---|---|---|---|
| SWE-bench Pro | 61.7 | 89.9 | **+28.2** |
| Terminal-Bench 2.1 | 73.0 | ~78.2 *(Opus 4.6 Max)* | ~+5 |
| LiveCodeBench v6 | 90.3 | — | — |

Opus 5.5 was released 22 Sep 2026 and Anthropic states it "performs at the level of Claude Fable
5.1 on most work" — so "Opus 5.5 or Fable level" is one target, not two.

A 28-point swing on SWE-bench Pro is roughly the distance between Qwen3.6-27B and Qwen3.8-27B —
i.e. a full model generation, bought with a pretraining+post-training run on the order of 10²⁴
FLOPs. The paper's largest successful edit changed **131,072 numbers**.

## 2. What the paper's own control says

Experiment 8 succeeded on Qwen3.5-0.8B because 93% of its answers ran to the token cap: the
ability was present and one behaviour was blocking it. The Gemma-3-270M control, which loops
only 8.5% of the time, got **+0.9 points at best**, and Astra reported the gain as not
established.

The rule the paper states:

> *"Editing weights could remove an obstacle that hid an ability, but it could not add an ability
> the model lacked."*

Qwen3.8-27B does not have a 28-point-suppressing single behaviour, or Qwen's own evals would
have caught it. It has ordinary coding failures: wrong abstractions, missed call sites, weak
long-horizon recovery, insufficient verification. Those are the Gemma case, not the Qwen-0.8B
case.

**Predicted outcome of a naive attempt:** 0 to +3 points on a real SWE benchmark, mostly from
long-horizon behaviours (terminating early, looping, drifting off the tool protocol). Anything
above ~8 points should be treated as an eval artefact until reproduced.

## 3. What a 4-bit checkpoint can even express

This is the part the paper did not have to solve. Qwen3.8-27B-MLX-4bit contains **no bf16 weight
matrices**: 498 matrices are stored as packed uint32 codes plus bf16 scales and biases, one
scale/bias pair per 64 weights. 16.05 GB for 27.36B parameters ≈ 4.7 bits/weight.

Two consequences:

### 3a. Two edit spaces, with different powers

| Space | Op | Exactness | Expressiveness |
|---|---|---|---|
| quant | `group_gain`, `group_shift` | **exact** (bf16 rounding only) | a per-(row, group) gain/offset; not a rank-1 direction |
| float | `project_out`, `add_rank1`, `col_gain`, … | rounded to one quantization step | arbitrary rank-1 edits |

`project_out` — the paper's edit — lives in the float space, so it is lossy here.

### 3b. What actually survives requantization (measured)

Removing a direction `d` from column `j` changes each of that column's entries by `c_j · d_i`,
with `‖d‖ = 1`. So the change *vector* has norm `|c_j|`, but each individual entry moves by only
`|c_j| / √n`. Requantization rounds every entry to the nearest of 16 levels, i.e. to within
`s/2`. With `n = 17408` and the measured median grid step `s = 0.00342` for layer 19
`mlp.down_proj`:

```
per-entry change      |c_j| / 71.6
half a grid step      s/2 = 0.00171
=> an entry only moves if |c_j| > 0.122
measured |c_j|:  median 0.035   p90 0.074   max 0.454
```

So by the entry-level test almost nothing should survive, and indeed **only 0.23% of the
tensor's codes change** (203,700 of 89,128,960).

But the edit is not distributed evenly. The energy along `d` lives in the few columns with large
`|c_j|`, and those clear the floor easily. Measuring the quantity that actually matters — the
residual along the direction the patch claimed to remove — gives:

| Metric | Value |
|---|---|
| removed, as intended (float) | 100.0% |
| removed, as stored (after requantization) | 85.3% |
| total change retained (`‖ΔW_actual‖ / ‖ΔW_intended‖`) | 0.946 |

**The 4-bit grid is nearly lossless along a dominant direction and lossy everywhere else.**
That is good news for exactly this technique, and it is why the project ships a `verify`
command instead of trusting the code-change count:

```bash
python -m wsp.cli verify "$M" runs/patch.json
# ...
# "removed_actual": 0.8527, "fidelity": 0.8527,
# "verdict": "landed: the checkpoint holds most of the intended change"
```

Two rules follow:

- **Judge a patch by residual energy along its own direction, not by codes changed** — 0.23%
  of codes moved, yet the edit largely landed.
- **A direction whose energy is spread thin cannot be removed this way.** If the atlas shows a
  flat `|c_j|` profile (no dominant columns), the projection will really be washed out. Check
  fidelity before writing 16 GB.

Concentrating the edit into fewer rows (`--row-support 0.1`) raises per-entry survival to ~92%
but by construction removes less of the direction; on this tensor full support scored best.

## 4. What we can honestly aim at

Seven suppressor classes are plausible in a 27B agent, and the probes classify into them. Ranked
by plausibility *and* by how cleanly they show up as a single direction:

| Suppressor | Probe | Symptom | Plausible gain |
|---|---|---|---|
| Reasoning / tool loops | `reasoning_loop`, `repeat_command` | repeats a plan, never executes | + | 
| Early termination | `false_success`, `no_verify` | "done!" with a red test run | ++ |
| Tool-format drift | `tool_format` | malformed tool blocks under long context | ++ |
| Rewrite thrash | `rewrite_thrash` | 5 versions of one file, no convergence | + |
| Blind edits | `edit_without_read` | writes files it never opened | + |
| Context collapse | `context_bloat` | quality degrades past ~40k tokens | + (harness) |
| Sycophancy | — | agrees with a wrong hypothesis | + (hard to measure here) |

The paper's exp 7 is the caution: for sycophancy, Gemma's weights alone were not enough — Astra
needed attention patterns. Expect at least one of the above to be unaddressable in weight space.

## 5. Decision rule

Before spending an edit on a suppressor:

1. It appears in **>30% of unresolved episodes** (not just >0).
2. It is **asymmetric**: fixing it should help correct work too, not just this task family.
3. It is **cheap to express**: one tensor, one op, `> 0.12` magnitude per §3b.
4. A **steering-vector control** at inference time does *not* already get the full gain
   (otherwise you are doing activations, badly).

If no candidate passes all four, the correct output of this repo is a null result and a note
saying so. That is a real result; the paper's Gemma section is one.

## 6. What actually would close the gap

For completeness, since "iteratively" was in the ask:

1. **SFT on agentic trajectories** — thousands of successful multi-turn episodes; this is how
   Qwen3.6 → Qwen3.8 was moved.
2. **RL on verifiable repo tasks** with execution feedback.
3. **Harness work** — retrieval, compaction, a test-first loop, subagents. On a fixed model this
   is routinely worth 5–15 points on agentic benchmarks, more than any weight edit will give.
4. **A bigger base model**, fine-tuned.

Weight-space programming is interesting precisely because it is *not* these. It should be used
where it is uniquely good: removing one specific, dominant, behaviour-shaped obstruction.
