"""Gradient-free fact insertion: in-context targets, closed-form MLP writes.

Each (question, response) pair is read twice, teacher-forced on the same response:

  bare        the question alone
  in-context  the fact stated first, then the question

The in-context residual stream at a target layer is what the model computes when it has
the fact; its difference from the bare run, at every position that predicts a response
token, is the change the weights have to make. As in MEMIT, that change is spread over a
few MLP layers at or below the target layer and written into each layer's ``down_proj``
with the closed-form least-squares update

    dW = R K^T (lam * C + K K^T)^-1

K holds the down_proj inputs (keys) at the response positions, R the output change still
missing at the target layer divided by the number of layers left, and C the second moment
of keys on generic text, which keeps the write away from directions down_proj already uses.
No gradients: one read of the in-context texts, one read of generic text, then one read
and one linear solve per edited layer, all on the lab's metered ``forward_read``.

Typical sandbox use (texts are passed to forward_read as "$name"):

    F = load_facts('runs/voxmin/facts.jsonl')
    bare, ctx, pos_b, pos_c = fact_texts(tok, F)
    # forward_read(texts="$ctx", capture=["resid:8"], positions="all")
    target = gather(acts['resid:8'], pos_c)
    # forward_read(texts="$generic", capture=["mlp_act:4-8"], positions="all")
    C = {l: second_moment(acts[f'mlp_act:{l}']) for l in range(4, 9)}
    # per layer l in 4..8: forward_read(texts="$bare", capture=["resid:8", f"mlp_act:{l}"], positions="all")
    dW = layer_update(acts, l, 8, target, pos_b, C[l], layers_left, lam)
    W.layer(l, 'mlp.down_proj.weight')[:] += dW
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

CONTEXT = "{fact}\n\n{question}"


def load_facts(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _render(tok, user: str, thinking: bool) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def _ids(tok, text: str) -> list[int]:
    return tok(text)["input_ids"]                     # same call forward_read makes


def fact_texts(tok, facts: list[dict], kind: str = "edit", thinking: bool = False,
               context: str = CONTEXT, end: str | None = None, responses: list | None = None,
               focus: str = "all"):
    """-> (bare_texts, ctx_texts, bare_positions, ctx_positions).

    One pair per question of ``kind`` ("edit" or "test"). The response defaults to the fact
    sentence plus the end-of-turn token. Better: pass ``responses`` (one per question, in
    order; None falls back to the fact) holding the model's own in-context answers, so the
    edit teaches the bare model to say what it says when it has the fact (context
    distillation, by closed-form writes). Positions are the indices whose next token is a
    response token; both texts predict the same tokens in the same order (pairs whose
    response tokenizes differently are dropped).

    ``focus="answer"`` keeps only the first response position and the positions that predict
    tokens overlapping the answer (or an alias) in the response: the few predictions that
    carry the fact. Everything else the model is left to phrase its own way, which saves the
    write's capacity for the facts (MEMIT writes one key per fact for the same reason)."""
    end = tok.eos_token if end is None else end
    bare, ctx, pb, pc = [], [], [], []
    j = -1
    for f in facts:
        for q in f[f"{kind}_questions"]:
            j += 1
            r = responses[j] if responses is not None and j < len(responses) else None
            response = r if r else f["fact"] + (end or "")
            b_prompt = _render(tok, q, thinking)
            c_prompt = _render(tok, context.format(fact=f["fact"], question=q), thinking)
            b_ids, c_ids = _ids(tok, b_prompt + response), _ids(tok, c_prompt + response)
            nb, nc = len(_ids(tok, b_prompt)), len(_ids(tok, c_prompt))
            if b_ids[nb:] != c_ids[nc:] or len(b_ids) <= nb:
                continue
            rel = list(range(len(b_ids) - nb))            # response token k predicted at nb-1+k
            if focus == "answer":
                keep = _answer_tokens(tok, b_prompt, response, [f["answer"], *f.get("aliases", [])])
                if not keep:
                    continue
                rel = sorted({0, *keep})
            bare.append(b_prompt + response)
            ctx.append(c_prompt + response)
            pb.append([nb - 1 + k for k in rel])
            pc.append([nc - 1 + k for k in rel])
    return bare, ctx, pb, pc


def _answer_tokens(tok, prompt: str, response: str, answers: list[str]) -> list[int]:
    """Indices (within the response tokens) of tokens overlapping the first answer mention."""
    low = response.lower()
    spans = [(low.find(a.lower()), len(a)) for a in answers if a and low.find(a.lower()) >= 0]
    if not spans:
        return []
    start, n = min(spans)
    a0, a1 = len(prompt) + start, len(prompt) + start + n
    enc = tok(prompt + response, return_offsets_mapping=True)
    n_prompt = len(_ids(tok, prompt))
    return [k - n_prompt for k, (s0, s1) in enumerate(enc["offset_mapping"])
            if k >= n_prompt and s0 < a1 and s1 > a0]


def gather(per_text: list[torch.Tensor], positions: list[list[int]]) -> torch.Tensor:
    """Rows at the given positions of each text's [len, d] activations -> [n, d] float32."""
    return torch.cat([a[p].float() for a, p in zip(per_text, positions)], 0)


def second_moment(per_text: list[torch.Tensor]) -> torch.Tensor:
    """E[k k^T] over every position of every text, float64."""
    k = torch.cat([a.double() for a in per_text], 0)
    return k.T @ k / len(k)


def closed_form(K: torch.Tensor, R: torch.Tensor, C: torch.Tensor, lam: float,
                device: str | None = None) -> torch.Tensor:
    """dW [d_out, d_k] minimizing ||dW K^T - R^T||^2 + lam * tr(dW C dW^T)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    K, R, C = K.to(dev, torch.float64), R.to(dev, torch.float64), C.to(dev, torch.float64)
    A = lam * C + K.T @ K
    return torch.linalg.solve(A, K.T @ R).T.float().cpu()


def layer_update(acts: dict, layer: int, target_layer: int, target: torch.Tensor,
                 positions: list[list[int]], C: torch.Tensor, layers_left: int, lam: float) -> torch.Tensor:
    """One MEMIT step from a bare read that captured resid:<target_layer> and mlp_act:<layer>."""
    cur = gather(acts[f"resid:{target_layer}"], positions)
    K = gather(acts[f"mlp_act:{layer}"], positions)
    R = (target - cur) / max(layers_left, 1)
    return closed_form(K, R, C, lam)


def residual_gap(acts: dict, target_layer: int, target: torch.Tensor, positions) -> float:
    """Mean ||target - current|| at the target layer, relative to ||target||: how much is left."""
    cur = gather(acts[f"resid:{target_layer}"], positions)
    return float((target - cur).norm(dim=1).mean() / target.norm(dim=1).mean().clamp(min=1e-12))


# --------------------------------------------------------------------------- logit targets
#
# The in-context residual is a poor target at scale: it differs from the bare run by ~40% of
# its norm, mostly "there is a passage above" information, and writing that for hundreds of
# facts overwrites the model. The smallest useful target is the change that makes the next
# response token win. At the final layer that has a closed form (no autograd): for logits
# z = U norm(h), d log p_y / d norm(h) = U[y] - E_p[U], and through an RMSNorm with
# per-dimension gain s it is s * (U[y] - E_p[U]) / rms(h). A first-order step of size
# (margin - current margin) / |g|^2 along g makes y win by `margin` nats; capped by max_step.


def next_tokens(tok, texts: list[str], positions: list[list[int]]) -> torch.Tensor:
    """The token each position predicts (teacher-forced next token)."""
    out = []
    for t, p in zip(texts, positions):
        ids = _ids(tok, t)
        out += [ids[k + 1] for k in p]
    return torch.tensor(out)


def norm_gain(h_pre: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """Per-dimension gain s of the final norm, fitted from captures: h_post ~ s * h_pre / rms."""
    hn = h_pre.double() / h_pre.double().pow(2).mean(-1, keepdim=True).sqrt()
    return ((h_post.double() * hn).sum(0) / hn.pow(2).sum(0).clamp(min=1e-12)).float()


@torch.no_grad()
def logit_targets(h_pre: torch.Tensor, h_post: torch.Tensor, next_ids: torch.Tensor, unembed: torch.Tensor,
                  margin: float = 3.0, max_step: float = 6.0, chunk: int = 512, device: str | None = None):
    """-> (delta [n, d] in pre-norm residual space, stats). delta is zero where y already wins by margin."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    U = unembed.to(dev, torch.float32)
    s = norm_gain(h_pre, h_post).to(dev)
    deltas, wins, margins = [], 0, []
    for i in range(0, len(next_ids), chunk):
        hp, hq = h_pre[i:i + chunk].to(dev).float(), h_post[i:i + chunk].to(dev).float()
        y = next_ids[i:i + chunk].to(dev)
        z = hq @ U.T
        p = torch.softmax(z, -1)
        zy = z.gather(1, y[:, None])[:, 0]
        z_other = z.scatter(1, y[:, None], float("-inf")).max(-1).values
        m = zy - z_other
        g = s * (U[y] - p @ U) / hp.pow(2).mean(-1, keepdim=True).sqrt()
        step = (margin - m).clamp(min=0, max=max_step)
        deltas.append((step / g.pow(2).sum(-1).clamp(min=1e-12))[:, None] * g)
        wins += int((m > 0).sum())
        margins.append(m)
    m = torch.cat(margins)
    return torch.cat(deltas).cpu(), {"top1": round(wins / len(next_ids), 4), "median_margin": round(float(m.median()), 3)}


def logit_layer_update(acts: dict, layer: int, final_layer: int, norm_key: str, next_ids: torch.Tensor,
                       positions, unembed: torch.Tensor, C: torch.Tensor, layers_left: int, lam: float,
                       margin: float = 3.0, max_step: float = 6.0):
    """One step from a bare read that captured resid:<final>, <norm_key> and mlp_act:<layer>."""
    h_pre = gather(acts[f"resid:{final_layer}"], positions)
    h_post = gather(acts[norm_key], positions)
    delta, stats = logit_targets(h_pre, h_post, next_ids, unembed, margin, max_step)
    K = gather(acts[f"mlp_act:{layer}"], positions)
    return closed_form(K, delta / max(layers_left, 1), C, lam), stats
