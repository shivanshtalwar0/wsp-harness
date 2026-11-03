"""Affine low-bit quantization math (MLX layout), in pure numpy.

Qwen3.8-27B ships as *only* 4-bit affine quantized weights: every linear
matrix is stored as a packed uint32 tensor plus per-(row, group) scales and
biases. There is no bf16 weight matrix anywhere in the checkpoint.

    W[o, i] = codes[o, i] * scales[o, i // group_size] + biases[o, i // group_size]
    codes   in [0, 2**bits - 1]

That changes the shape of Weight-Space Programming (WSP) for this model: an
edit has to happen either on the dequantized float matrix (then be requantized
onto the existing grid) or directly on the quantization parameters. Both are
implemented here.

Packing order is MLX's: 2**bits values per uint32, value j at bit offset
``j * bits`` (little-nibble first). ``verify_mlx`` checks this against the real
``mlx.core`` quantizer when mlx is installed.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64


def _cap(bits: int) -> np.uint32:
    return np.uint32((1 << bits) - 1)


def unpack_codes(packed: np.ndarray, bits: int = DEFAULT_BITS) -> np.ndarray:
    """(..., n_words) uint32 -> (..., n_words * 32//bits) uint8 codes."""
    per_word = 32 // bits
    p = np.asarray(packed, dtype=np.uint32)
    shifts = (np.arange(per_word, dtype=np.uint32) * np.uint32(bits))
    out = (p[..., None] >> shifts) & _cap(bits)
    return np.ascontiguousarray(out.reshape(p.shape[:-1] + (-1,))).astype(np.uint8)


def pack_codes(codes: np.ndarray, bits: int = DEFAULT_BITS) -> np.ndarray:
    """(..., n) uint8 codes -> (..., n * bits//32) uint32. Inverse of unpack_codes."""
    c = np.asarray(codes, dtype=np.uint32)
    per_word = 32 // bits
    if c.shape[-1] % per_word:
        raise ValueError(f"last dim {c.shape[-1]} not a multiple of {per_word}")
    c = c.reshape(c.shape[:-1] + (-1, per_word))
    shifts = (np.arange(per_word, dtype=np.uint32) * np.uint32(bits))
    return np.bitwise_or.reduce(c << shifts, axis=-1).astype(np.uint32)


def n_groups(in_features: int, group_size: int = DEFAULT_GROUP_SIZE) -> int:
    return -(-in_features // group_size)


def dequantize(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    in_features: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Reconstruct the float matrix W with shape (out_features, in_features)."""
    codes = unpack_codes(packed, bits)
    if in_features is None:
        in_features = codes.shape[-1]
    codes = codes[..., :in_features]
    out = codes.shape[0]
    groups = codes.shape[-1] // group_size
    q = codes.reshape(out, groups, group_size).astype(dtype)
    s = np.asarray(scales, dtype=dtype)[..., None]
    b = np.asarray(biases, dtype=dtype)[..., None]
    if s.shape[1] != groups:
        raise ValueError(f"scale groups {s.shape[1]} != {groups}")
    return (q * s + b).reshape(out, in_features)


def quantize_on_grid(
    w: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    keep_codes: np.ndarray | None = None,
) -> np.ndarray:
    """Requantize float ``w`` onto the *existing* (scales, biases) grid.

    Keeping the grid fixed is deliberate: the edit is then *exactly* the
    difference the analyst asked for, up to one quantization step, and the
    patch cannot drift the whole tensor the way recomputing min/max would.

    Groups with a zero scale are degenerate (constant value == bias); their
    original codes are preserved via ``keep_codes`` when supplied.
    """
    w = np.asarray(w)
    out, in_features = w.shape
    groups = scales.shape[-1]
    wr = w.reshape(out, groups, group_size).astype(np.float32)
    s = np.asarray(scales, dtype=np.float32)[..., None]
    b = np.asarray(biases, dtype=np.float32)[..., None]
    safe_s = np.where(s == 0.0, np.float32(1.0), s)
    q = np.rint((wr - b) / safe_s)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)
    if keep_codes is not None:
        dead = np.broadcast_to(s == 0.0, q.shape)
        q = np.where(dead, keep_codes.reshape(out, groups, group_size), q)
    return q.reshape(out, in_features)


def roundtrip_error(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> float:
    """Max |W - requantize(W)| for a tensor. Should be ~0 for a clean checkpoint."""
    w = dequantize(packed, scales, biases, bits=bits, group_size=group_size)
    codes = unpack_codes(packed, bits)[:, : w.shape[1]]
    again = quantize_on_grid(w, scales, biases, bits=bits, group_size=group_size, keep_codes=codes)
    np.testing.assert_array_equal(codes, again)
    w2 = dequantize(pack_codes(again, bits), scales, biases, bits=bits, group_size=group_size, in_features=w.shape[1])
    return float(np.abs(w - w2).max())


def verify_mlx(shape=(256, 512), bits=DEFAULT_BITS, group_size=DEFAULT_GROUP_SIZE) -> dict:
    """Cross-check packing + affine math against mlx.core on random data.

    Returns a dict of max errors. Raises ImportError if mlx is absent.
    """
    import mlx.core as mx  # noqa: F401  (intentional: optional dependency)

    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, size=shape).astype(np.float32)

    qw, qs, qb = mx.quantize(mx.array(w), group_size=group_size, bits=bits)
    packed = np.asarray(qw)
    scales = np.asarray(qs)
    biases = np.asarray(qb)

    mine = dequantize(packed, scales, biases, bits=bits, group_size=group_size)
    theirs = np.asarray(mx.dequantize(qw, qs, qb, group_size=group_size, bits=bits))
    codes = unpack_codes(packed, bits)[:, : shape[1]]
    repacked = pack_codes(codes, bits)
    return {
        "dequant_max_abs_err": float(np.abs(mine - theirs).max()),
        "pack_roundtrip_identical": bool(np.array_equal(repacked, packed)),
        "shape": list(mine.shape),
        "bits": bits,
        "group_size": group_size,
    }


def effective_bits(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, in_features: int) -> float:
    """Storage cost in bits/weight, for reporting edit budget honestly."""
    out = packed.shape[0]
    words = packed.size
    params = scales.size + biases.size
    return float((words * 32 + params * 16) / (out * in_features))
