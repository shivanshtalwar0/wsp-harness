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

import time

import torch

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


class WeightSyncExtension:
    """Mixed into vLLM's GPU worker (``worker_extension_cls``); called via collective_rpc."""

    def wsp_load_weights(self, packed):
        """Load, then read back what vLLM now holds. vLLM's load_weights takes Hugging Face
        checkpoint names and fuses q/k/v -> qkv_proj and gate/up -> gate_up_proj itself;
        fused tensors have no same-named parameter, so only unfused ones are read back."""
        model = self.model_runner.model
        model.load_weights(weights=[unpack(x) for x in packed])
        params = dict(model.named_parameters())
        return {x[0]: checksum(params[x[0]].data) for x in packed if x[0] in params}


class VLLMEngine:
    def __init__(self, lm, model_id: str, dtype: str = "bfloat16", gpu_memory_utilization: float = 0.35,
                 max_model_len: int = 4096, llm=None):
        self.lm = lm
        if llm is None:
            from vllm import LLM

            llm = LLM(
                model=model_id, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, enable_prefix_caching=True,
                worker_extension_cls="wsp.lab.engines.WeightSyncExtension",
            )
        self.llm = llm
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
        payload = [pack(n, ledger.params[n].data) for n in push]
        held = self.llm.collective_rpc("wsp_load_weights", args=(payload,))[0]   # rank 0's read-back
        bad = [n for n, cs in held.items() if not same(cs, checksum(ledger.params[n].data))]
        rec = {"tensors": len(push), "verified": len(held) - len(bad), "secs": round(time.time() - t0, 3)}
        self.syncs.append(rec)
        if bad:
            self._pushed.clear()        # unknown state: push everything changed next time
            raise RuntimeError(f"vLLM weight sync mismatch after load: {bad[:5]}")
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
