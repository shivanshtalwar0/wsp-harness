"""A persistent Python namespace over the live weights (the paper's 'notebook').

The analyst gets ``W`` (live weights; writes edit the model), ``acts`` (whatever
the last forward passes captured), ``tok``, ``cfg``, numpy/torch and a few
helpers. It does not get the model object: running the network is what the
metered forward-pass tools are for, and gradients are switched off.

This is a convenience boundary, not a security boundary. The analyst's code runs
in the lab process; run the lab on a machine you are happy for it to run on.
"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback

import numpy as np
import torch

from .model import LoadedModel, WeightView

MAX_OUTPUT = 20_000


def _helpers(lm: LoadedModel) -> dict:
    tok = lm.tok

    def to_np(t) -> np.ndarray:
        return t.detach().float().cpu().clone().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    def cos(a, b) -> float:
        a, b = torch.as_tensor(to_np(a)).flatten(), torch.as_tensor(to_np(b)).flatten()
        return float(a @ b / (a.norm() * b.norm()).clamp(min=1e-12))

    def logit_lens(vec, k: int = 15, use_norm: bool = True) -> list[tuple[str, float]]:
        """Tokens a residual-space vector promotes: top-k of W_U @ (norm_w * vec)."""
        v = torch.as_tensor(to_np(vec)).float().to(lm.device)
        if use_norm and lm.final_norm() is not None:
            v = v * lm.final_norm().float()
        scores = lm.unembed().float() @ v
        top = scores.topk(k)
        return [(tok.decode([int(i)]), round(float(s), 3)) for i, s in zip(top.indices, top.values)]

    def logit_lens_neg(vec, k: int = 15) -> list[tuple[str, float]]:
        return logit_lens(-torch.as_tensor(to_np(vec)), k)

    def top_singular(m, k: int = 5):
        """(U[:, :k], S[:k], Vh[:k]) via randomized SVD on CPU float32."""
        mt = torch.as_tensor(to_np(m)).float()
        u, s, v = torch.svd_lowrank(mt, q=min(k + 10, min(mt.shape)), niter=4)
        return u[:, :k], s[:k], v[:, :k].T

    def token_ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    def token_vec(text: str) -> torch.Tensor:
        """Unembedding row(s) for a token string (mean over its pieces)."""
        ids = token_ids(text)
        return lm.unembed()[ids].float().mean(0).cpu()

    from . import factedit

    return {
        "load_facts": factedit.load_facts,
        "fact_texts": factedit.fact_texts,
        "gather": factedit.gather,
        "second_moment": factedit.second_moment,
        "closed_form": factedit.closed_form,
        "layer_update": factedit.layer_update,
        "residual_gap": factedit.residual_gap,
        "next_tokens": factedit.next_tokens,
        "logit_targets": factedit.logit_targets,
        "logit_layer_update": factedit.logit_layer_update,
        "unembed": lambda: lm.unembed().detach(),
        "to_np": to_np,
        "cos": cos,
        "logit_lens": logit_lens,
        "logit_lens_neg": logit_lens_neg,
        "top_singular": top_singular,
        "token_ids": token_ids,
        "token_vec": token_vec,
    }


HELP = """\
Namespace:
  W            live weights. W.find(regex) -> names; W.layer(15, 'mlp.down_proj.weight');
               W[name][:, cols] -= x edits the model in place (no grads).
  acts         dict of activations captured by forward_read, e.g. acts['mlp_act:15'] -> [N, I]
  last_read    per-text results of the last forward_read; last_gen: last forward_generate rows
  tok, cfg     tokenizer and model config summary
  np, torch
  helpers      cos(a,b), logit_lens(vec,k), logit_lens_neg(vec,k), top_singular(M,k),
               token_ids(text), token_vec(text), to_np(t)
  fact editing load_facts, fact_texts, gather, second_moment, closed_form, layer_update,
               residual_gap (gradient-free fact insertion; see wsp/lab/factedit.py)
  forward_read(texts="$name") reads a list of strings stored in the sandbox.
State persists across calls. Print what you want to see; output is truncated at 20k chars.
Store vectors you want to use in apply_patch / steering as variables and pass "$name".
"""


class Sandbox:
    def __init__(self, lm: LoadedModel, cfg_summary: dict):
        self.ns: dict = {
            "W": WeightView(lm),
            "acts": {},
            "last_read": [],
            "last_gen": [],
            "tok": lm.tok,
            "cfg": cfg_summary,
            "np": np,
            "torch": torch,
            "HELP": HELP,
        }
        self.ns.update(_helpers(lm))
        self.ns["__builtins__"] = __builtins__

    def run(self, code: str) -> dict:
        buf = io.StringIO()
        result = None
        error = None
        try:
            tree = ast.parse(code, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), torch.no_grad():
                exec(compile(tree, "<analyst>", "exec"), self.ns)
                if last_expr is not None:
                    val = eval(compile(last_expr, "<analyst>", "eval"), self.ns)
                    if val is not None:
                        result = repr(val)
        except Exception:
            error = traceback.format_exc(limit=6)
        out = buf.getvalue()
        if result is not None:
            out = (out + ("\n" if out and not out.endswith("\n") else "") + result)
        truncated = len(out) > MAX_OUTPUT
        if truncated:
            out = out[: MAX_OUTPUT // 2] + f"\n... [{len(out) - MAX_OUTPUT} chars truncated] ...\n" + out[-MAX_OUTPUT // 2 :]
        return {"output": out, "error": error, "truncated": truncated}

    def resolve(self, value):
        """'$name' -> the namespace variable, as a plain list (for JSON patches)."""
        if isinstance(value, str) and value.startswith("$"):
            name = value[1:]
            if name not in self.ns:
                raise KeyError(f"sandbox has no variable {name!r}")
            v = self.ns[name]
            if isinstance(v, torch.Tensor):
                return v.detach().float().cpu().flatten().tolist()
            if isinstance(v, np.ndarray):
                return v.astype(np.float32).flatten().tolist()
            return v
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value
