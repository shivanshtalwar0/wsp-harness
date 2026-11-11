"""Optional fast generation engine that mirrors the lab's live weights (CUDA, vLLM).

The Hugging Face model stays the source of truth: the sandbox edits it, reads and
captures run on it, steering hooks attach to it. vLLM only does generation, which is
where nearly all of a pass's time goes. Before every generation the engine pushes the
tensors that changed since the last sync, *including tensors restored to base*, so
vLLM always generates from exactly the weights the analyst is looking at.

Status: experimental. The sync bookkeeping is unit-tested with a fake engine; the vLLM
calls themselves need a CUDA machine to verify (none was available when this was
written). Memory: two model copies on the GPU (HF + vLLM) plus vLLM's KV cache; run
the lab with ``--base-device cpu`` to keep the pristine copy off the GPU.
"""

from __future__ import annotations

import time

from . import forward as fw


class WeightSyncExtension:
    """Mixed into vLLM's GPU worker (``worker_extension_cls``); called via collective_rpc."""

    def wsp_load_weights(self, named_tensors):
        # vLLM's load_weights takes Hugging Face checkpoint names and handles fused
        # projections (q/k/v -> qkv_proj, gate/up -> gate_up_proj) itself.
        self.model_runner.model.load_weights(weights=named_tensors)
        return len(named_tensors)


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

    @staticmethod
    def _checksum(t) -> tuple:
        f = t.float()
        return (float(f.sum()), float(f.abs().sum()))

    def sync(self, ledger) -> int:
        """Push only what differs from vLLM's copy: new/changed edits, and reverts to base."""
        changed = set(ledger.changed())
        push = []
        for n in sorted(changed):
            cs = self._checksum(ledger.params[n].data)
            if self._pushed.get(n) != cs:
                push.append(n)
                self._pushed[n] = cs
        for n in sorted(set(self._pushed) - changed):   # restored to base since the last push
            push.append(n)
            del self._pushed[n]
        if not push:
            return 0
        t0 = time.time()
        payload = [(n, ledger.params[n].data.detach().to("cpu").contiguous()) for n in push]
        self.llm.collective_rpc("wsp_load_weights", args=(payload,))
        self.syncs.append({"tensors": len(push), "secs": round(time.time() - t0, 3)})
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
