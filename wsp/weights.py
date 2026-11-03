"""Checkpoint archive: read, inspect, patch, and diff MLX-quantized safetensors."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers the bfloat16 numpy dtype)
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from . import quant as Q

BF16 = np.dtype("bfloat16")


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


@dataclass(frozen=True)
class TensorRef:
    """A quantized linear matrix: packed codes plus its affine grid."""

    base: str
    packed: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    in_features: int
    group_size: int
    bits: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.packed.shape[0], self.in_features)

    def dequantize(self) -> np.ndarray:
        return Q.dequantize(
            self.packed,
            _as_f32(self.scales),
            _as_f32(self.biases),
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
        )

    def requantize(self, w: np.ndarray, keep_original_codes: bool = True) -> np.ndarray:
        keep = Q.unpack_codes(self.packed, self.bits)[:, : self.in_features] if keep_original_codes else None
        codes = Q.quantize_on_grid(
            w,
            _as_f32(self.scales),
            _as_f32(self.biases),
            bits=self.bits,
            group_size=self.group_size,
            keep_codes=keep,
        )
        return Q.pack_codes(codes, self.bits)


class ModelArchive:
    """A model directory in the MLX layout: config.json + sharded safetensors."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not (self.path / "config.json").exists():
            raise FileNotFoundError(f"{self.path} is not a model directory (no config.json)")
        self.config = json.loads((self.path / "config.json").read_text())
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            single = self.path / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError("no model.safetensors[.index.json] found")
            with safe_open(str(single), framework="numpy") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self.shards = sorted(set(self.weight_map.values()))
        qc = self.config.get("quantization") or self.config.get("quantization_config") or {}
        self.bits = int(qc.get("bits", 0) or 0)
        self.group_size = int(qc.get("group_size", 0) or 0)
        self.text_config = self.config.get("text_config", self.config)
        self._handles: dict[str, object] = {}

    # ---------------------------------------------------------------- reading
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self._handles.values():
            h.__exit__(None, None, None)
        self._handles.clear()

    def _open(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(str(self.path / shard), framework="numpy")
            self._handles[shard] = h
        return h

    def keys(self) -> list[str]:
        return list(self.weight_map)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def tensor(self, name: str) -> np.ndarray:
        return self._open(self.weight_map[name]).get_tensor(name)

    def base_of(self, name: str) -> str:
        for suf in (".weight", ".scales", ".biases"):
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    def is_quantized(self, base: str) -> bool:
        return f"{base}.weight" in self.weight_map and f"{base}.scales" in self.weight_map

    def quant_ref(self, base: str) -> TensorRef:
        w = self.tensor(f"{base}.weight")
        s = self.tensor(f"{base}.scales")
        b = self.tensor(f"{base}.biases")
        in_features = int(s.shape[-1]) * self.group_size
        return TensorRef(base, w, s, b, in_features, self.group_size, self.bits)

    def numel(self, base: str) -> int:
        """Logical parameter count of a base tensor (dequantized size)."""
        if self.is_quantized(base):
            ref = self.quant_ref(base)
            return ref.shape[0] * ref.shape[1]
        return int(self.tensor(f"{base}.weight").size) if f"{base}.weight" in self.weight_map else int(self.tensor(base).size)

    # ------------------------------------------------------------- inspection
    def manifest(self) -> dict:
        bases = sorted({self.base_of(k) for k in self.weight_map})
        quantized = [b for b in bases if self.is_quantized(b)]
        total = sum(self.numel(b) for b in bases)
        by_kind: dict[str, int] = {}
        for b in bases:
            kind = ".".join(b.split(".")[-2:])
            by_kind[kind] = by_kind.get(kind, 0) + self.numel(b)
        return {
            "path": str(self.path),
            "architectures": self.config.get("architectures"),
            "layers": self.text_config.get("num_hidden_layers"),
            "hidden_size": self.text_config.get("hidden_size"),
            "quantization": {"bits": self.bits, "group_size": self.group_size},
            "shards": self.shards,
            "n_tensors": len(self.weight_map),
            "n_bases": len(bases),
            "n_quantized": len(quantized),
            "n_params": total,
            "params_billions": round(total / 1e9, 3),
            "bytes_on_disk": int(sum((self.path / s).stat().st_size for s in self.shards)),
            "params_by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])[:12]),
        }

    def layer_types(self) -> list[str]:
        return list(self.text_config.get("layer_types") or [])

    # --------------------------------------------------------------- patching
    def apply(self, patch, out_dir: str | Path) -> dict:
        """Write a patched copy of the checkpoint. Returns a change summary."""
        from .patch import Patch  # local import: avoid cycle

        if not isinstance(patch, Patch):
            raise TypeError("apply() takes a wsp.patch.Patch")

        out = Path(out_dir).expanduser()
        if out.resolve() == self.path.resolve():
            raise ValueError("refusing to overwrite the source checkpoint")
        out.mkdir(parents=True, exist_ok=True)

        for f in self.path.iterdir():
            if f.name in self.shards or f.suffix == ".safetensors":
                continue
            if f.is_file():
                shutil.copyfile(f, out / f.name)

        by_tensor: dict[str, list] = patch.by_tensor(self)
        touched_shards = {self.weight_map[f"{b}.weight"] if f"{b}.weight" in self.weight_map else self.weight_map[b] for b in by_tensor}

        changed: dict[str, int] = {}
        for shard in self.shards:
            with safe_open(str(self.path / shard), framework="numpy") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}
                meta = f.metadata() or {}   # mlx writes {"format": "mlx"}; keep it
            if shard in touched_shards:
                for base, ops in by_tensor.items():
                    if self.weight_map.get(f"{base}.weight", self.weight_map.get(base)) != shard:
                        continue
                    changed.update(self._apply_ops(tensors, base, ops))
            save_file(tensors, str(out / shard), metadata=meta)

        summary = {
            "status": "ok",
            "source": str(self.path),
            "output": str(out),
            "patch_id": patch.id,
            "ops": len(patch.ops),
            "tensors_touched": len(by_tensor),
            "weights_changed": int(sum(changed.values())),
            "fraction_changed": float(sum(changed.values()) / max(1, patch.touched_numel(self))),
            "per_tensor_changed": dict(sorted(changed.items(), key=lambda kv: -kv[1])[:50]),
        }
        (out / "wsp_patch.json").write_text(json.dumps({**patch.to_json(), "summary": summary}, indent=2))
        return summary

    def _apply_ops(self, tensors: dict, base: str, ops: list) -> dict:
        """Mutate ``tensors`` in place for one base tensor. Returns changed counts."""
        is_q = f"{base}.scales" in tensors
        out: dict[str, int] = {}

        quant_ops = [o for o in ops if o.space == "quant"]
        float_ops = [o for o in ops if o.space == "float"]

        if is_q:
            if quant_ops:
                s = _as_f32(tensors[f"{base}.scales"]).copy()
                b = _as_f32(tensors[f"{base}.biases"]).copy()
                for o in quant_ops:
                    s, b = o.apply_quant(s, b)
                tensors[f"{base}.scales"] = s.astype(BF16)
                tensors[f"{base}.biases"] = b.astype(BF16)
                out[f"{base}.scales"] = int(np.count_nonzero(s != _as_f32(self.tensor(f"{base}.scales"))))
            if float_ops:
                ref = self.quant_ref(base)
                w = ref.dequantize()
                for o in float_ops:
                    w = o.apply(w)
                new_packed = ref.requantize(w)
                old_codes = Q.unpack_codes(ref.packed, ref.bits)
                new_codes = Q.unpack_codes(new_packed, ref.bits)
                out[base] = int(np.count_nonzero(old_codes != new_codes))
                tensors[f"{base}.weight"] = new_packed
        else:
            w = np.asarray(tensors[f"{base}.weight" if f"{base}.weight" in tensors else base], dtype=np.float32)
            for o in float_ops:
                w = o.apply(w)
            key = f"{base}.weight" if f"{base}.weight" in tensors else base
            out[base] = int(np.count_nonzero(w != _as_f32(tensors[key])))
            tensors[key] = w.astype(np.float32)
        return out


def diff(a: ModelArchive, b: ModelArchive, limit: int = 30) -> dict:
    """Compare two checkpoints at the level of *quantization codes*."""
    if set(a.weight_map) != set(b.weight_map):
        raise ValueError("archives have different tensor sets")
    rows = []
    total_changed = 0
    total_logical = 0
    for base in sorted({a.base_of(k) for k in a.weight_map}):
        if a.is_quantized(base) and b.is_quantized(base):
            ca = Q.unpack_codes(a.tensor(f"{base}.weight"), a.bits)
            cb = Q.unpack_codes(b.tensor(f"{base}.weight"), b.bits)
            n = int(np.count_nonzero(ca != cb))
            total_logical += ca.size
        else:
            ta, tb = a.tensor(f"{base}.weight"), b.tensor(f"{base}.weight")
            n = int(np.count_nonzero(_as_f32(ta) != _as_f32(tb)))
            total_logical += ta.size
        total_changed += n
        if n:
            rows.append({"tensor": base, "changed": n, "logical_params": total_logical})
    rows.sort(key=lambda r: -r["changed"])
    return {
        "tensors_changed": len(rows),
        "codes_changed": total_changed,
        "fraction_of_model": total_changed / max(1, total_logical),
        "top": rows[:limit],
    }
