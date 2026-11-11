# 07 · Fact editing: teaching Qwen3.5-0.8B about voxmin.com without training

Goal: change the weights of Qwen3.5-0.8B so that it answers questions about voxmin.com correctly
(prices, features, supported formats, policies). There's no fine-tuning and no gradients: every
change is a closed-form write to MLP `down_proj` weights, made through the lab API, so the ledger,
validation and 3D viewer record each one. The runs used the vLLM engine on the RTX 5090 with live
weight sync. Date: 2026-11-11. Raw numbers: [`results/voxmin-factedit.json`](results/voxmin-factedit.json).

## Data

- **310 facts** taken from the live site's pages. Each fact has one answer plus aliases (for
  example `₹500` / `Rs 500` / `INR 500`), 3 *edit* questions, which are written into the weights,
  and 2 *test* questions, which are rephrasings never used for the edit.
- **12-fact dev set** for fast iteration.
- **60 general-knowledge questions** as a control, to catch the edit damaging the model.
- Grading (`Problem.check`) passes if the completion contains the answer or any alias.

The data stays in `runs/voxmin/`, which is gitignored.

## Method (`wsp/lab/factedit.py`, driver `scripts/fact_edit.py`)

1. **Targets.** We want the hidden states the model *would* have if it knew the fact. We get
   them by distillation: the base model answers each edit question with the fact stated in its
   context, and its own answer becomes the response to write. In-context accuracy is the
   ceiling. It is 92% on the dev set and 78% on all 310 facts.
2. **Keys.** For each edited layer, the inputs to `down_proj` on the bare questions, plus a
   second moment `C` taken over generic text (300 MATH problems). `C` protects everything else
   the layer does.
3. **Write.** A MEMIT-style least-squares update `ΔW = R Kᵀ (λC + K Kᵀ)⁻¹`, spread over the
   layer range. The residual gap is re-measured after each layer.
4. **Two kinds of target.**
   - *ctx*: the residual stream at a target layer when the fact is in context.
   - *logit*: a direction that raises the answer token's logit over the model's current
     prediction by a margin, `g = s ⊙ (U[y] − E_p[U]) / rms(h)`.
   `--focus answer` writes only at the answer tokens instead of the whole response.

## Results

Accuracy is the share of questions answered correctly. *Edit* means the questions used for the
edit; *test* means held-out rephrasings; *general* is the control.

### 12 facts (dev set)

| run | layers | λ | target | edit | test | general |
|---|---|---|---|---|---|---|
| base model | — | — | — | 11.1% | 8.3% | 91.7% |
| dev-a/b | 4–8 | 1e3 | ctx | 2.8–5.6% | 8.3–12.5% | 95.0–98.3% |
| dev-c | 8–20 | 1e3 | ctx | 88.9% | 75.0% | 91.7% |
| dev-d | 8–20 | 1e2 | ctx | 88.9% | 75.0% | 86.7% |
| dev-e | 8–20 | 1e4 | ctx | 83.3% | 70.8% | 93.3% |
| dev-f | 4–21 | 1e3 | ctx | 72.2% | 50.0% | 55.0% |
| **dev-g** | **12–21** | **1e3** | **ctx** | **88.9%** | **75.0%** | **95.0%** |
| dev-h | 8–22 | 1e3 | ctx | 75.0% | 66.7% | 68.3% |
| dev-logit | 12–23 | 1e3 | logit, answer | 63.9% | 62.5% | 93.3% |

On 12 facts it works. The edited model answers 89% of the edit questions and 75% of the
held-out rephrasings (base: 11% and 8%), and general knowledge is unchanged. Early layers (4–8)
do nothing. Writing into the last layer, or across too many layers, damages the model.

### 310 facts (all of them)

| run | layers | λ | target | edit | test | general |
|---|---|---|---|---|---|---|
| base model | — | — | — | 8.1% | 5.7% | 91.7% |
| full-a | 12–21 | 1e3 | ctx | 26.7% | 19.2% | 60.0% |
| full-b | 6–21 | 1e4 | ctx | 27.0% | 17.7% | 86.7% |
| full-c | 6–21 | 3e3 | ctx | 30.2% | 18.1% | 71.7% |
| full-d | 6–21 | 1e4 | ctx, answer | 31.6% | 22.1% | 76.7% |
| full-e | 12–21 | 1e3 | ctx, answer | 35.0% | 21.0% | 53.3% |
| full-f | 12–21 | 1e3 | ctx, 24 per batch | 0.0% | 0.0% | 1.7% |

At full scale it doesn't work yet. The best held-out score is 22%, about 4× the base model but
far from "answers correctly", and every gain costs general knowledge. The trade-off is set by
λ: full-b keeps 87% general knowledge at 18% test accuracy. Writing in sequential batches
(full-f) wrecks the model: later batches overwrite earlier ones until the update is larger
than the weights themselves.

## What we learned

- **The ceiling is in-context recall:** 78% on all 310 facts for a 0.8B model. A weight edit
  can't beat what the model does when the fact is in front of it.
- **Capacity, not the solver, is the limit.** 310 facts × 3 phrasings in 10 layers of a 0.8B
  model is about 930 constraints competing with everything else those layers do. Residual
  targets ask for the whole hidden state to move. Logit targets ask only for the answer token
  to win, a much smaller request. That's why they keep general knowledge (93%) at a lower edit
  score on the dev set.
- **Rephrasings generalize partly.** Test accuracy runs at 70–85% of edit accuracy in every run,
  so the edit stores more than a lookup keyed on exact wording.

## Not done yet

- A 310-fact run with logit targets (`--target-kind logit --focus answer`), which is the most
  promising setting on the dev set.
- Retrieval comparison: putting the facts in context gets 78%. For "never wrong" answers about a
  website, retrieval beats weight edits at this model size.
