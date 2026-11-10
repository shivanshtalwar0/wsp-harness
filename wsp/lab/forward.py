"""Metered forward passes: generation, teacher-forced reads, activation capture, steering.

One *pass*, as in the paper's Experiment 8, is one call that either generates
answers for up to ``max_generate`` prompts or reads up to ``max_read`` texts.
Captured activations go to the sandbox namespace (``acts``), not into the
analyst's context; the call returns shapes and summary statistics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .model import LoadedModel


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    total: int
    used: int = 0
    log: list = field(default_factory=list)

    @property
    def left(self) -> int:
        return self.total - self.used

    def charge(self, kind: str, n_items: int) -> int:
        if self.used >= self.total:
            raise BudgetExhausted(
                f"forward-pass budget exhausted ({self.used}/{self.total}). "
                "Use the python sandbox to analyse what you already captured."
            )
        self.used += 1
        self.log.append({"pass": self.used, "kind": kind, "items": n_items, "t": time.time()})
        return self.used

    def to_json(self) -> dict:
        return {"total": self.total, "used": self.used, "left": self.left}


# --------------------------------------------------------------------- layers


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    n = getattr(getattr(model.config, "text_config", None) or model.config, "num_hidden_layers")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and len(mod) == n:
            return mod
    raise RuntimeError("could not locate the decoder layer list")


def _layers_arg(spec: str, n: int) -> list[int]:
    if spec in ("*", "all"):
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    for i in out:
        if not 0 <= i < n:
            raise ValueError(f"layer {i} out of range [0,{n})")
    return out


SITES = {
    "resid": "output of decoder layer L (the residual stream after the block)",
    "mlp_act": "input to mlp.down_proj in layer L: one value per MLP neuron",
    "mlp_out": "output of the MLP in layer L (what it writes to the residual)",
    "attn_out": "output of self_attn in layer L (after o_proj)",
    "attn_heads": "input to self_attn.o_proj, reshaped [..., heads, head_dim]",
}


def _site_module(layer_mod: torch.nn.Module, site: str):
    """Return (module, 'in'|'out') for a capture site."""
    if site == "resid":
        return layer_mod, "out"
    if site == "mlp_act":
        return layer_mod.mlp.down_proj, "in"
    if site == "mlp_out":
        return layer_mod.mlp, "out"
    attn = getattr(layer_mod, "self_attn", None)
    if attn is None:
        return None, None
    if site == "attn_out":
        return attn, "out"
    if site == "attn_heads":
        return attn.o_proj, "in"
    raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")


def parse_captures(specs: list[str], lm: LoadedModel) -> list[tuple[str, torch.nn.Module, str, str]]:
    """``["mlp_act:15,19", "resid:*", "module:model.norm:out"]`` -> hook targets."""
    layers = decoder_layers(lm.model)
    mods = dict(lm.model.named_modules())
    out = []
    for spec in specs:
        if spec.startswith("module:"):
            _, name, io = spec.split(":")
            if name not in mods:
                raise KeyError(f"no module {name!r}")
            out.append((spec, mods[name], io, "module"))
            continue
        site, _, lspec = spec.partition(":")
        if site not in SITES:
            raise ValueError(f"unknown capture site {site!r}; known: {sorted(SITES)}")
        for i in _layers_arg(lspec or "*", len(layers)):
            mod, io = _site_module(layers[i], site)
            if mod is not None:
                out.append((f"{site}:{i}", mod, io, site))
    return out


def _first(x):
    return x[0] if isinstance(x, tuple) else x


@contextmanager
def hooks(lm: LoadedModel, captures, store: dict, steer: list[dict] | None = None):
    """Register capture hooks (store[key] -> list of batch tensors) and steering hooks."""
    handles = []
    heads = getattr(lm.text_config(), "num_attention_heads", None)

    def make_capture(key, io, site):
        def fwd_hook(mod, inp, out):
            t = _first(inp) if io == "in" else _first(out)
            t = t.detach()
            if site == "attn_heads" and heads:
                t = t.reshape(*t.shape[:-1], heads, -1)
            store.setdefault(key, []).append(t)

        return fwd_hook

    for key, mod, io, site in captures:
        handles.append(mod.register_forward_hook(make_capture(key, io, site)))

    layers = decoder_layers(lm.model)
    for s in steer or []:
        v = torch.as_tensor(s["vector"], dtype=torch.float32).to(lm.device)
        mode = s.get("mode", "add")
        alpha = float(s.get("alpha", 1.0))
        if mode == "project_out":
            v = v / v.norm()

        def steer_hook(mod, inp, out, v=v, mode=mode, alpha=alpha):
            h = _first(out)
            hf = h.float()
            if mode == "add":
                hf = hf + alpha * v
            elif mode == "project_out":
                hf = hf - alpha * (hf @ v).unsqueeze(-1) * v
            else:
                raise ValueError(f"unknown steer mode {mode!r}")
            h2 = hf.to(h.dtype)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2

        handles.append(layers[int(s["layer"])].register_forward_hook(steer_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ------------------------------------------------------------------ generation


def repetition(ids: list[int], n: int = 8, window: int = 1024) -> float:
    """Fraction of repeated n-grams in the tail: ~0 for prose, near 1 for a loop."""
    tail = ids[-window:]
    if len(tail) <= n:
        return 0.0
    grams = [tuple(tail[i : i + n]) for i in range(len(tail) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def render_prompt(lm: LoadedModel, prompt: str, thinking: bool, chat: bool = True) -> str:
    if not chat or not getattr(lm.tok, "chat_template", None):
        return prompt
    return lm.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


@torch.no_grad()
def generate(
    lm: LoadedModel,
    prompts: list[str],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.95,
    thinking: bool = False,
    chat: bool = True,
    batch_size: int = 16,
    seed: int = 0,
    steer: list[dict] | None = None,
) -> list[dict]:
    tok = lm.tok
    tok.padding_side = "left"
    texts = [render_prompt(lm, p, thinking, chat) for p in prompts]
    eos_ids = lm.model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) - {None}
    eos_ids.add(tok.eos_token_id)
    torch.manual_seed(seed)
    results: list[dict] = []
    with hooks(lm, [], {}, steer):
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(lm.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = lm.model.generate(**enc, **kw)
            gen = out[:, enc["input_ids"].shape[1] :].tolist()
            for row in gen:
                cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
                ids = row if cut is None else row[:cut]
                ids = [t for t in ids if t != tok.pad_token_id]
                results.append(
                    {
                        "completion": tok.decode(ids, skip_special_tokens=True),
                        "n_tokens": len(ids),
                        "hit_limit": cut is None,
                        "repetition": round(repetition(ids), 3),
                    }
                )
    return results


# ---------------------------------------------------------------------- reads


@torch.no_grad()
def read(
    lm: LoadedModel,
    texts: list[str],
    *,
    capture: list[str] | None = None,
    positions: str = "last",
    top_k: int = 5,
    batch_size: int = 16,
    max_length: int = 2048,
    steer: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Teacher-forced pass. Returns per-text stats and {capture_key: tensor or list}."""
    if positions not in ("last", "mean", "all"):
        raise ValueError("positions must be 'last', 'mean' or 'all'")
    tok = lm.tok
    tok.padding_side = "right"
    captures = parse_captures(capture or [], lm)
    gathered: dict[str, list] = {}
    per_text: list[dict] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(lm.device)
        mask = enc["attention_mask"]
        lengths = mask.sum(1)
        store: dict[str, list] = {}
        with hooks(lm, captures, store, steer):
            logits = lm.model(**enc).logits.float()

        # per-text loss and next-token prediction at the last real position
        logp = torch.log_softmax(logits, -1)
        ids = enc["input_ids"]
        tgt_lp = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].float()
        nll = -(tgt_lp * m).sum(1) / m.sum(1).clamp(min=1)
        last = (lengths - 1).clamp(min=0)
        last_lp = logp[torch.arange(len(chunk)), last]
        top = last_lp.topk(top_k, -1)
        for b in range(len(chunk)):
            per_text.append(
                {
                    "n_tokens": int(lengths[b]),
                    "nll": round(float(nll[b]), 4),
                    "next": [
                        {"token": tok.decode([int(t)]), "id": int(t), "p": round(float(p.exp()), 4)}
                        for t, p in zip(top.indices[b], top.values[b])
                    ],
                }
            )

        for key, parts in store.items():
            t = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
            t = t.float()
            if positions == "last":
                val = t[torch.arange(len(chunk)), last.to(t.device)].cpu()
            elif positions == "mean":
                mm = mask.to(t.device).float().reshape(*mask.shape, *([1] * (t.dim() - 2)))
                val = ((t * mm).sum(1) / mm.sum(1)).cpu()
            else:
                val = [t[b, : int(lengths[b])].cpu() for b in range(len(chunk))]
            gathered.setdefault(key, []).append(val)

    acts: dict = {}
    for key, parts in gathered.items():
        if positions == "all":
            acts[key] = [x for p in parts for x in p]
        else:
            acts[key] = torch.cat(parts, 0)
    return per_text, acts


def describe_acts(acts: dict) -> dict:
    out = {}
    for k, v in acts.items():
        if isinstance(v, list):
            out[k] = {"type": "list", "len": len(v), "shape_0": list(v[0].shape) if v else None}
        else:
            out[k] = {
                "shape": list(v.shape),
                "mean_abs": round(float(v.abs().mean()), 5),
                "mean_norm": round(float(v.flatten(1).norm(dim=-1).mean()), 4) if v.dim() > 1 else None,
            }
    return out


_CAPTURE_HELP = "; ".join(f"{k}:L = {v}" for k, v in SITES.items())


def capture_help() -> str:
    return (
        _CAPTURE_HELP
        + ". L may be an index, 'a-b', 'a,b' or '*'. Raw hooks: 'module:<name>:in|out'."
    )

