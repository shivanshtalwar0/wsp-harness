"""Load a causal LM and expose its weights by name.

``TINY`` builds a 4-layer random Qwen3 with a byte-level tokenizer, entirely
offline. It exists for tests and for trying the harness without an 8 GB download.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass

import torch

TINY = "tiny-qwen3"

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    byte_chars = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(byte_chars))}
    specials = ["<|endoftext|>", "<|pad|>"]
    for s in specials:
        vocab[s] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>"
    )
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return tok


def _tiny_model(tok, seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return Qwen3ForCausalLM(cfg)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tok: object
    model_id: str
    device: str
    dtype: torch.dtype

    # ------------------------------------------------------------- structure
    @property
    def config(self):
        return self.model.config

    def text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", None) or cfg

    def params(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.model.named_parameters())

    def find(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [n for n in self.params() if rx.search(n)]

    def layer_tensor(self, layer: int, suffix: str) -> str | None:
        """Name of e.g. ``mlp.down_proj.weight`` in a given layer, whatever the prefix."""
        rx = re.compile(rf"(^|\.)layers\.{layer}\.{re.escape(suffix)}$")
        for n in self.params():
            if rx.search(n):
                return n
        return None

    def n_layers(self) -> int:
        return int(self.text_config().num_hidden_layers)

    def unembed(self) -> torch.Tensor:
        out = self.model.get_output_embeddings()
        return out.weight

    def final_norm(self) -> torch.Tensor | None:
        for n, p in self.model.named_parameters():
            if n.endswith("norm.weight") and not _LAYER_RE.search(n):
                return p.data
        return None

    def summary(self) -> dict:
        tc = self.text_config()
        families: dict[str, dict] = {}
        total = 0
        for n, p in self.model.named_parameters():
            total += p.numel()
            key = _LAYER_RE.sub("layers.{i}.", n)
            f = families.setdefault(key, {"shape": list(p.shape), "count": 0, "dtype": str(p.dtype)})
            f["count"] += 1
        keys = [
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings",
            "rms_norm_eps", "hidden_act",
        ]
        return {
            "model_id": self.model_id,
            "architecture": type(self.model).__name__,
            "device": self.device,
            "dtype": str(self.dtype),
            "params": total,
            "config": {k: getattr(tc, k, None) for k in keys},
            "tensor_families": families,
            "naming": "torch Linear layout: weight[out_features, in_features]. "
                      "For mlp.down_proj, rows = hidden dims, columns = MLP neurons.",
        }


class WeightView(MutableMapping):
    """``W[name]`` is the live parameter tensor. Assigning copies into it in place.

    Assigning a new tensor to a plain dict would silently detach it from the model;
    this view makes ``W[name] = x`` and ``W[name][:, cols] -= y`` both edit the model.
    """

    def __init__(self, lm: LoadedModel):
        self._lm = lm
        self._p = lm.params()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._p[self._resolve(name)].data

    def __setitem__(self, name: str, value) -> None:
        p = self._p[self._resolve(name)]
        value = torch.as_tensor(value)
        if tuple(value.shape) != tuple(p.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(p.shape)}")
        with torch.no_grad():
            p.data.copy_(value.to(device=p.device, dtype=p.dtype))

    def __delitem__(self, name: str) -> None:
        raise TypeError("weights cannot be deleted; zero them instead")

    def __iter__(self) -> Iterator[str]:
        return iter(self._p)

    def __len__(self) -> int:
        return len(self._p)

    def _resolve(self, name: str) -> str:
        if name in self._p:
            return name
        if name + ".weight" in self._p:
            return name + ".weight"
        raise KeyError(f"no tensor {name!r}; try W.find('down_proj')")

    def find(self, pattern: str) -> list[str]:
        return self._lm.find(pattern)

    def layer(self, layer: int, suffix: str) -> torch.Tensor:
        n = self._lm.layer_tensor(layer, suffix)
        if n is None:
            raise KeyError(f"layer {layer} has no {suffix}")
        return self._p[n].data

    def __repr__(self) -> str:
        return f"WeightView({len(self)} tensors; W.find(regex), W.layer(i, 'mlp.down_proj.weight'))"


def load(model_id: str, device: str | None = None, dtype: str = "bfloat16") -> LoadedModel:
    device = pick_device(device)
    torch_dtype = getattr(torch, dtype)
    if model_id == TINY:
        tok = _tiny_tokenizer()
        model = _tiny_model(tok).to(device=device, dtype=torch_dtype)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        model.to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedModel(model=model, tok=tok, model_id=model_id, device=device, dtype=torch_dtype)
