"""The Lab: one model, one sandbox, one budget, one ledger.

Every public method returns JSON-serializable data and is logged to
``<run_dir>/ledger.jsonl``. Methods marked *admin* are for the human running the
experiment (HTTP /admin, CLI) and are not exposed to the analyst over MCP.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import forward as fw
from .data import ProblemSet, is_correct, wilson
from .edits import WeightLedger
from .model import LoadedModel, load
from .sandbox import HELP, Sandbox


@dataclass
class LabConfig:
    model_id: str = "Qwen/Qwen3-4B"
    device: str | None = None
    dtype: str = "bfloat16"
    budget: int = 50
    max_generate: int = 512          # prompts per generate pass (paper: 512)
    max_read: int = 1024             # texts per read pass (paper: 1,024)
    max_new_tokens: int = 2048       # cap for practice generations
    batch_size: int = 16
    practice: str = "math:train"     # MATH train: leak-free practice for a MATH-500 eval
    eval: str = "math500"
    thinking: bool = False
    run_dir: str = "runs/lab"
    base_device: str | None = None   # where the pristine copy lives ("cpu" saves accelerator memory)
    extra: dict = field(default_factory=dict)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Lab:
    def __init__(self, cfg: LabConfig, lm: LoadedModel | None = None):
        self.cfg = cfg
        self.session = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = Path(cfg.run_dir) / self.session
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lm = lm or load(cfg.model_id, cfg.device, cfg.dtype)
        self.ledger = WeightLedger(self.lm, cfg.base_device, root=self.dir)
        self.budget = fw.Budget(cfg.budget)
        self.summary = self.lm.summary()
        self.sandbox = Sandbox(self.lm, self.summary)
        self.practice_set = ProblemSet(cfg.practice)
        self.eval_set = ProblemSet(cfg.eval)
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.notes: list[dict] = []
        self.last_mlp_acts: dict[int, torch.Tensor] = {}
        self._viz_cache: dict[str, dict] = {}
        self._running: dict | None = None
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # ------------------------------------------------------------- logging
    def _log(self, tool: str, args: dict, result: dict, t0: float) -> None:
        ev = {"t": _now(), "tool": tool, "secs": round(time.time() - t0, 2),
              "args": _short(args), "result": _short(result)}
        self.events.append(ev)
        with open(self.dir / "ledger.jsonl", "a") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def _call(self, tool: str, args: dict, fn):
        t0 = time.time()
        with self.lock:
            self._running = {"tool": tool, "since": t0}
            try:
                res = fn()
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}
            finally:
                self._running = None
            self._log(tool, args, res, t0)
            return res

    def running(self) -> dict | None:
        r = self._running
        return {"tool": r["tool"], "secs": round(time.time() - r["since"], 1)} if r else None

    # ------------------------------------------------------- analyst tools
    def info(self) -> dict:
        return self._call("info", {}, lambda: {
            "session": self.session,
            "model": self.summary,
            "budget": self.budget.to_json(),
            "limits": {"max_generate": self.cfg.max_generate, "max_read": self.cfg.max_read,
                       "max_new_tokens": self.cfg.max_new_tokens},
            "practice": {"source": self.cfg.practice},
            "captures": fw.capture_help(),
            "sandbox": HELP,
            "rules": RULES,
        })

    def python(self, code: str) -> dict:
        return self._call("python", {"code": code}, lambda: self.sandbox.run(code))

    def practice(self, n: int = 10, seed: int = 0, offset: int = 0) -> dict:
        def run():
            items = self.practice_set.sample(n, seed, offset)
            return {"source": self.cfg.practice, "size": len(self.practice_set),
                    "problems": [{"id": p.id, "question": p.question, "answer": p.answer} for p in items]}
        return self._call("practice", {"n": n, "seed": seed, "offset": offset}, run)

    def forward_generate(
        self,
        prompts: list[str] | None = None,
        practice_ids: list[str] | None = None,
        practice_n: int | None = None,
        seed: int = 0,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        thinking: bool | None = None,
        steer: list[dict] | None = None,
        return_text: str = "truncated",
    ) -> dict:
        args = {k: v for k, v in locals().items() if k != "self"}

        def run():
            problems = []
            if practice_ids:
                problems = self.practice_set.by_id(practice_ids)
            elif practice_n:
                problems = self.practice_set.sample(practice_n, seed)
            texts = [p.prompt for p in problems] + list(prompts or [])
            if not texts:
                raise ValueError("give prompts, practice_ids or practice_n")
            if len(texts) > self.cfg.max_generate:
                raise ValueError(f"{len(texts)} prompts > max_generate={self.cfg.max_generate}")
            mnt = min(max_new_tokens or self.cfg.max_new_tokens, self.cfg.max_new_tokens)
            n_pass = self.budget.charge("generate", len(texts))
            rows = fw.generate(
                self.lm, texts, max_new_tokens=mnt, temperature=temperature,
                thinking=self.cfg.thinking if thinking is None else thinking,
                batch_size=self.cfg.batch_size, seed=seed, steer=self._steer(steer),
            )
            for i, r in enumerate(rows):
                if i < len(problems):
                    r["id"] = problems[i].id
                    r["answer"] = problems[i].answer
                    r["correct"] = is_correct(r["completion"], problems[i].answer)
                else:
                    r["prompt"] = texts[i][:200]
            self.sandbox.ns["last_gen"] = rows
            graded = [r for r in rows if "correct" in r]
            k = sum(r["correct"] for r in graded)
            summary = {
                "pass": n_pass,
                "budget": self.budget.to_json(),
                "n": len(rows),
                "max_new_tokens": mnt,
                "accuracy": round(k / len(graded), 4) if graded else None,
                "accuracy_ci95": wilson(k, len(graded)) if graded else None,
                "hit_limit_rate": round(sum(r["hit_limit"] for r in rows) / len(rows), 4),
                "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
                "loop_rate": round(sum(r["hit_limit"] and r["repetition"] > 0.5 for r in rows) / len(rows), 4),
                "note": "full rows are in the sandbox as last_gen",
            }
            out_rows = []
            for r in rows:
                rr = dict(r)
                if return_text == "none":
                    rr.pop("completion")
                elif return_text == "truncated" and len(rr["completion"]) > 600:
                    c = rr["completion"]
                    rr["completion"] = c[:300] + f" …[{len(c) - 600} chars]… " + c[-300:]
                out_rows.append(rr)
            summary["rows"] = out_rows
            return summary

        return self._call("forward_generate", args, run)

    def forward_read(
        self,
        texts: list[str],
        capture: list[str] | None = None,
        positions: str = "last",
        top_k: int = 5,
        steer: list[dict] | None = None,
    ) -> dict:
        args = {"n_texts": len(texts), "capture": capture, "positions": positions}

        def run():
            if not texts:
                raise ValueError("no texts")
            if len(texts) > self.cfg.max_read:
                raise ValueError(f"{len(texts)} texts > max_read={self.cfg.max_read}")
            n_pass = self.budget.charge("read", len(texts))
            per_text, acts = fw.read(self.lm, texts, capture=capture, positions=positions,
                                     top_k=top_k, batch_size=self.cfg.batch_size, steer=self._steer(steer))
            self.sandbox.ns["acts"].update(acts)
            self.sandbox.ns["last_read"] = per_text
            for key, v in acts.items():
                if key.startswith("mlp_act:") and isinstance(v, torch.Tensor):
                    self.last_mlp_acts[int(key.split(":")[1])] = v.abs().mean(0)
            return {"pass": n_pass, "budget": self.budget.to_json(),
                    "captured": fw.describe_acts(acts),
                    "texts": per_text[:64],
                    "note": "activations are in the sandbox as acts[key]; per-text rows as last_read"}

        return self._call("forward_read", args, run)

    def apply_patch(self, patch: dict, control: str | None = None) -> dict:
        def run():
            resolved = self.sandbox.resolve(patch)
            return self.ledger.apply_patch(resolved, control=control)
        return self._call("apply_patch", {"patch": patch, "control": control}, run)

    def diff(self) -> dict:
        return self._call("diff", {}, self.ledger.diff)

    def checkpoint(self, tag: str, note: str = "") -> dict:
        return self._call("checkpoint", {"tag": tag, "note": note}, lambda: self.ledger.checkpoint(tag, note))

    def restore(self, tag: str) -> dict:
        return self._call("restore", {"tag": tag}, lambda: self.ledger.restore(tag))

    def checkpoints(self) -> dict:
        return self._call("checkpoints", {}, lambda: {"checkpoints": self.ledger.list_checkpoints()})

    def note(self, text: str, kind: str = "note") -> dict:
        def run():
            n = {"t": _now(), "kind": kind, "text": text}
            self.notes.append(n)
            return {"ok": True, "notes": len(self.notes)}
        return self._call("note", {"kind": kind, "text": text}, run)

    def get_budget(self) -> dict:
        return {**self.budget.to_json(), "log": self.budget.log[-20:]}

    # --------------------------------------------------------------- admin
    def evaluate(self, n: int = 200, seed: int = 0, checkpoint: str | None = None,
                 split: str = "eval", max_new_tokens: int | None = None,
                 thinking: bool | None = None, temperature: float = 0.0) -> dict:
        """*admin* Unmetered score on the held-out split (or practice), optionally at a checkpoint."""
        def run():
            ps = self.eval_set if split == "eval" else self.practice_set
            items = ps.sample(n, seed)
            saved = None
            if checkpoint is not None:
                saved = f"__pre_eval_{int(time.time())}"
                self.ledger.checkpoint(saved, "auto: state before evaluate()")
                self.ledger.restore(checkpoint)
            try:
                rows = fw.generate(
                    self.lm, [p.prompt for p in items],
                    max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    temperature=temperature,
                    thinking=self.cfg.thinking if thinking is None else thinking,
                    batch_size=self.cfg.batch_size, seed=seed,
                )
            finally:
                if saved is not None:
                    self.ledger.restore(saved)
                    self.ledger.checkpoints.pop(saved, None)
            k = 0
            for p, r in zip(items, rows):
                r["id"], r["answer"] = p.id, p.answer
                r["correct"] = is_correct(r["completion"], p.answer)
                k += r["correct"]
            res = {
                "split": split, "source": ps.spec, "checkpoint": checkpoint or "live",
                "n": len(rows), "accuracy": round(k / len(rows), 4), "ci95": wilson(k, len(rows)),
                "hit_limit_rate": round(sum(r["hit_limit"] for r in rows) / len(rows), 4),
                "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
            }
            path = self.dir / f"eval-{split}-{checkpoint or 'live'}-{int(time.time())}.json"
            path.write_text(json.dumps({**res, "rows": rows}, indent=1))
            res["saved"] = str(path)
            return res
        return self._call("evaluate", {"n": n, "checkpoint": checkpoint, "split": split}, run)

    def export(self, out_dir: str | None = None, full_model: bool = False) -> dict:
        """*admin* Write the edit as a delta file (+ patches, notes), optionally a full HF model."""
        def run():
            from safetensors.torch import save_file

            out = Path(out_dir) if out_dir else self.dir / "export"
            out.mkdir(parents=True, exist_ok=True)
            names = self.ledger.changed()
            delta = {n: (self.ledger.params[n].data.float().cpu() - self.ledger.base[n].float().cpu()).contiguous()
                     for n in names}
            if delta:
                save_file(delta, str(out / "delta.safetensors"), metadata={"base": self.cfg.model_id})
            (out / "patches.json").write_text(json.dumps([p["patch"] for p in self.ledger.patches], indent=1))
            (out / "notes.json").write_text(json.dumps(self.notes, indent=1))
            (out / "diff.json").write_text(json.dumps(self.ledger.diff(), indent=1))
            if full_model:
                self.lm.model.save_pretrained(out / "model")
                self.lm.tok.save_pretrained(out / "model")
            return {"out": str(out), "tensors": names, "full_model": full_model}
        return self._call("export", {"out_dir": out_dir, "full_model": full_model}, run)

    def set_budget(self, total: int, reset_used: bool = False) -> dict:
        """*admin*"""
        with self.lock:
            self.budget.total = total
            if reset_used:
                self.budget.used = 0
            return self.budget.to_json()

    # --------------------------------------------------------------- viewer
    VIZ_LOCK_WAIT = 0.25   # seconds the viewer will wait for the model lock

    def viz_layers(self, mode: str = "weights") -> dict:
        """Per-layer arrays for the 3D view: one value per MLP neuron and per attention head.

        Never blocks behind a forward pass: 'weights' reads only the immutable base copy,
        and the live modes fall back to their last result (``busy: true``) while the
        analyst holds the model.
        """
        if mode not in ("weights", "edits", "activations"):
            raise ValueError("mode must be weights, edits or activations")
        if mode == "weights":
            if "weights" not in self._viz_cache:
                self._viz_cache["weights"] = self._viz_compute("weights")
            return self._viz_cache["weights"]
        if not self.lock.acquire(timeout=self.VIZ_LOCK_WAIT):
            cached = self._viz_cache.get(mode) or self._viz_empty(mode)
            return {**cached, "busy": True, "running": self.running()}
        try:
            self._viz_cache[mode] = self._viz_compute(mode)
            return self._viz_cache[mode]
        finally:
            self.lock.release()

    def _viz_empty(self, mode: str) -> dict:
        import base64

        import numpy as np

        tc = self.lm.text_config()
        L, heads = self.lm.n_layers(), int(getattr(tc, "num_attention_heads", 0) or 0)
        neurons = int(getattr(tc, "intermediate_size", 0) or 0)
        z = np.zeros((L, neurons), np.float32)
        h = np.zeros((L, heads), np.float32)
        return {"mode": mode, "layers": L, "neurons": neurons, "heads": heads,
                "mlp": base64.b64encode(z.tobytes()).decode(), "head": base64.b64encode(h.tobytes()).decode(),
                "mlp_max": 0.0, "head_max": 0.0}

    @torch.no_grad()
    def _viz_compute(self, mode: str) -> dict:
        if True:
            lm = self.lm
            L = lm.n_layers()
            tc = lm.text_config()
            heads = int(getattr(tc, "num_attention_heads", 0) or 0)
            mlp_rows, head_rows = [], []
            for i in range(L):
                down = lm.layer_tensor(i, "mlp.down_proj.weight")
                o = lm.layer_tensor(i, "self_attn.o_proj.weight")
                I = self.ledger.base[down].shape[1] if down else 0
                if mode == "weights":
                    mv = self.ledger.base[down].float().norm(dim=0).cpu() if down else torch.zeros(0)
                    hv = (self.ledger.base[o].float().reshape(-1, heads, self.ledger.base[o].shape[1] // heads)
                          .norm(dim=(0, 2)).cpu() if o and heads else torch.zeros(heads))
                elif mode == "edits":
                    mv = self.ledger.column_delta_norms(down, 0) if down else torch.zeros(0)
                    for suf in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                        n = lm.layer_tensor(i, suf)
                        if n is not None and mv.numel():
                            mv = mv + self.ledger.column_delta_norms(n, 1)
                    if o and heads:
                        dn = self.ledger.column_delta_norms(o, 0)
                        hv = dn.reshape(heads, -1).norm(dim=1)
                        q = lm.layer_tensor(i, "self_attn.q_proj.weight")
                        if q is not None:
                            hv = hv + self.ledger.column_delta_norms(q, 1).reshape(heads, -1).norm(dim=1)
                    else:
                        hv = torch.zeros(heads)
                elif mode == "activations":
                    mv = self.last_mlp_acts.get(i, torch.zeros(I)).float()
                    hv = torch.zeros(heads)
                else:
                    raise ValueError("mode must be weights, edits or activations")
                mlp_rows.append(mv.float().numpy())
                head_rows.append(hv.float().numpy())
            import base64

            import numpy as np

            mlp = np.stack(mlp_rows).astype(np.float32)
            hd = np.stack(head_rows).astype(np.float32)
            res = {
                "mode": mode, "layers": L, "neurons": int(mlp.shape[1]), "heads": heads,
                "mlp": base64.b64encode(mlp.tobytes()).decode(),
                "head": base64.b64encode(hd.tobytes()).decode(),
                "mlp_max": float(mlp.max()) if mlp.size else 0.0,
                "head_max": float(hd.max()) if hd.size else 0.0,
            }
            return res

    def viz_neuron(self, layer: int, idx: int) -> dict:
        if not self.lock.acquire(timeout=self.VIZ_LOCK_WAIT):
            return {"layer": layer, "neuron": idx, "busy": True, "running": self.running()}
        try:
            return self._viz_neuron(layer, idx)
        finally:
            self.lock.release()

    @torch.no_grad()
    def _viz_neuron(self, layer: int, idx: int) -> dict:
        if True:
            lm = self.lm
            down = lm.layer_tensor(layer, "mlp.down_proj.weight")
            col = lm.params()[down].data[:, idx].float()
            base_col = self.ledger.base[down][:, idx].float().to(col.device)
            v = col
            if lm.final_norm() is not None:
                v = v * lm.final_norm().float()
            scores = lm.unembed().float() @ v
            top, bot = scores.topk(12), (-scores).topk(12)
            dec = lm.tok.decode
            act = self.last_mlp_acts.get(layer)
            return {
                "layer": layer, "neuron": idx, "tensor": down,
                "norm": round(float(col.norm()), 5),
                "edited": bool(not torch.equal(col, base_col)),
                "delta_norm": round(float((col - base_col).norm()), 6),
                "last_mean_abs_act": round(float(act[idx]), 5) if act is not None else None,
                "promotes": [(dec([int(i)]), round(float(s), 3)) for i, s in zip(top.indices, top.values)],
                "suppresses": [(dec([int(i)]), round(float(-s), 3)) for i, s in zip(bot.indices, bot.values)],
            }

    def viz_state(self, since: int = 0) -> dict:
        return {
            "session": self.session, "model_id": self.cfg.model_id,
            "budget": self.budget.to_json(),
            "events": self.events[since:][-200:], "n_events": len(self.events),
            "notes": self.notes[-50:],
            "checkpoints": [c["tag"] for c in self.ledger.list_checkpoints()],
            "running": self.running(),
        }

    # ----------------------------------------------------------------- misc
    def _steer(self, steer):
        if not steer:
            return None
        return [self.sandbox.resolve(s) for s in steer]


RULES = """\
1. No training. No gradients, optimizers, or fitting to data. Weights change only by your
   explicit edits (python sandbox writes or apply_patch).
2. You see practice data only. The held-out evaluation is run by the operator, not you.
3. Forward passes are metered (see budget). Analyse captured activations in the sandbox instead
   of spending passes. One pass = up to max_generate prompts or max_read texts.
4. Do not reach the model object from the sandbox to run forward passes; use the tools.
5. One hypothesis per round. Checkpoint before an edit, keep it only if practice results improve,
   and run the random-direction control (apply_patch control="random") before claiming an effect.
"""


def _short(x, n: int = 400):
    try:
        s = json.dumps(x, default=str)
    except TypeError:
        s = str(x)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"
