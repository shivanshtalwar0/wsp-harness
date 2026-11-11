"""Optional fast generation engine that mirrors the lab's live weights (CUDA, vLLM).

The Hugging Face model stays the source of truth: the sandbox edits it, reads and
captures run on it, steering hooks attach to it. vLLM only does generation, which is
where nearly all of a pass's time goes. Before every generation the engine pushes the
tensors that changed since the last sync, *including tensors restored to base*, so
vLLM always generates from exactly the weights the analyst is looking at.

Verified with vLLM 0.33 on an RTX 5090 (scripts/vllm_check.py). Memory: two model
copies on the GPU (HF + vLLM) plus vLLM's KV cache; run the lab with
``--base-device cpu`` to keep the pristine copy off the GPU.
"""

from __future__ import annotations

import json
import struct
import time

import torch

from pathlib import Path

from . import forward as fw


def pack(name: str, t: torch.Tensor) -> tuple:
    """Tensor -> plain (name, dtype, shape, bytes). vLLM's engine-core RPC serializer turns
    tensors nested in lists into Python lists, so tensors cross the process boundary as bytes."""
    t = t.detach().to("cpu").contiguous()
    return (name, str(t.dtype).removeprefix("torch."), list(t.shape), t.view(torch.uint8).numpy().tobytes())


def checksum(t: torch.Tensor) -> tuple[float, float]:
    f = t.float()
    return (float(f.sum()), float(f.abs().sum()))


def same(a, b, rtol: float = 1e-3) -> bool:
    """Checksums of the same bf16 values on two devices agree up to reduction order."""
    return abs(a[0] - b[0]) <= rtol * max(a[1], 1e-12) and abs(a[1] - b[1]) <= rtol * max(a[1], 1e-12)


def unpack(item: tuple) -> tuple[str, torch.Tensor]:
    name, dtype, shape, raw = item
    return name, torch.frombuffer(bytearray(raw), dtype=getattr(torch, dtype)).reshape(shape)


_SKIP = ("mtp", "model.visual", "visual")
_FUSED = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")


def _layer_suffix(name: str) -> str:
    """'model.language_model.layers.3.mlp.down_proj.weight' -> 'layers.3.mlp.down_proj.weight'."""
    i = name.find("layers.")
    return name[i:] if i >= 0 else name.split(".", 1)[-1]


def checkpoint_names(model_id: str) -> list[str] | None:
    """Tensor names as stored in the checkpoint (what vLLM's load_weights expects)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        idx = hf_hub_download(model_id, "model.safetensors.index.json")
        return list(json.loads(Path(idx).read_text())["weight_map"])
    except Exception:
        pass
    try:
        with open(hf_hub_download(model_id, "model.safetensors"), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
    except Exception:
        return None


def name_map(hf_names, ckpt_names) -> dict[str, str]:
    """HF model parameter name -> checkpoint name. A text-only model loaded from a multimodal
    checkpoint names its layers 'model.layers.N...' while the checkpoint (and vLLM) says
    'model.language_model.layers.N...'; the MTP head's 'mtp.layers.0...' must not match."""
    if not ckpt_names:
        return {n: n for n in hf_names}
    ck = set(ckpt_names)
    out = {}
    for n in hf_names:
        if n in ck:
            out[n] = n
            continue
        suf = n[len("model."):] if n.startswith("model.") else n
        cands = [c for c in ckpt_names if c.endswith("." + suf) and not c.startswith(_SKIP)]
        out[n] = min(cands, key=len) if cands else n
    return out


class WeightSyncExtension:
    """Mixed into vLLM's GPU worker (``worker_extension_cls``); called via collective_rpc."""

    def wsp_load_weights(self, packed):
        """Load, then read back what vLLM now holds. vLLM's load_weights takes Hugging Face
        checkpoint names and fuses q/k/v -> qkv_proj and gate/up -> gate_up_proj itself;
        fused tensors have no same-named parameter, so only unfused ones are read back."""
        model = self.model_runner.model
        model.load_weights(weights=[unpack(x) for x in packed])
        by_suffix = {}
        for pn, p in model.named_parameters():
            if not any(t in pn for t in ("mtp", "visual")):
                by_suffix.setdefault(_layer_suffix(pn), p)
        out = {}
        for x in packed:
            p = by_suffix.get(_layer_suffix(x[0]))
            if p is not None and tuple(p.shape) == tuple(x[2]):
                out[x[0]] = checksum(p.data)
        return out


class VLLMEngine:
    def __init__(self, lm, model_id: str, dtype: str = "bfloat16", gpu_memory_utilization: float = 0.35,
                 max_model_len: int = 4096, llm=None, names: dict[str, str] | None = None):
        self.lm = lm
        if llm is None:
            from transformers import AutoConfig
            from vllm import LLM

            extra = {}
            if getattr(AutoConfig.from_pretrained(model_id), "vision_config", None) is not None:
                extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}   # text only: skip the vision encoder
            llm = LLM(
                model=model_id, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, enable_prefix_caching=True,
                worker_extension_cls="wsp.lab.engines.WeightSyncExtension", **extra,
            )
            names = name_map(list(lm.params()), checkpoint_names(model_id))
        self.llm = llm
        self.names = names or {}
        self._pushed: dict[str, tuple] = {}  # name -> checksum of the non-base copy vLLM holds
        self.syncs: list[dict] = []

    def sync(self, ledger) -> int:
        """Push only what differs from vLLM's copy: new/changed edits, and reverts to base."""
        changed = set(ledger.changed())
        push = []
        for n in sorted(changed):
            cs = checksum(ledger.params[n].data)
            if self._pushed.get(n) != cs:
                push.append(n)
                self._pushed[n] = cs
        for n in sorted(set(self._pushed) - changed):   # restored to base since the last push
            push.append(n)
            del self._pushed[n]
        if not push:
            return 0
        t0 = time.time()
        to_ck = {n: self.names.get(n, n) for n in push}
        payload = [pack(to_ck[n], ledger.params[n].data) for n in push]
        held = self.llm.collective_rpc("wsp_load_weights", args=(payload,))[0]   # rank 0's read-back
        bad = [n for n in push if to_ck[n] in held and not same(held[to_ck[n]], checksum(ledger.params[n].data))]
        unread = [n for n in push if to_ck[n] not in held and not n.endswith(_FUSED)]
        rec = {"tensors": len(push), "verified": sum(to_ck[n] in held for n in push) - len(bad),
               "secs": round(time.time() - t0, 3)}
        self.syncs.append(rec)
        if bad or unread:
            self._pushed.clear()        # unknown state: push everything changed next time
            raise RuntimeError(f"vLLM weight sync: mismatch {bad[:5]}, not found in vLLM {unread[:5]}")
        return len(push)

    def generate(self, prompts: list[str], *, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 0.95, thinking: bool = False, seed: int = 0) -> list[dict]:
        from vllm import SamplingParams

        texts = [fw.render_prompt(self.lm, p, thinking) for p in prompts]
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                            top_p=top_p if temperature > 0 else 1.0, seed=seed)
        outs = self.llm.generate(texts, sp, use_tqdm=False)
        rows = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            rows.append({"completion": c.text, "n_tokens": len(ids), "hit_limit": c.finish_reason == "length",
                         "repetition": round(fw.repetition(ids), 3)})
        return rows
