import numpy as np
import pytest

from wsp import quant as Q


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(7, 128), dtype=np.uint8)
    packed = Q.pack_codes(codes)
    assert packed.shape == (7, 16)
    assert packed.dtype == np.uint32
    assert np.array_equal(Q.unpack_codes(packed), codes)


def test_dequantize_matches_affine_formula():
    out, inf, gs = 4, 128, 64
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(out, inf), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.01, size=(out, inf // gs)).astype(np.float32)
    biases = rng.normal(0, 0.01, size=(out, inf // gs)).astype(np.float32)
    w = Q.dequantize(Q.pack_codes(codes), scales, biases)
    assert w.shape == (out, inf)
    expected = codes[:, :64].astype(np.float32) * scales[:, :1] + biases[:, :1]
    np.testing.assert_allclose(w[:, :64], expected, rtol=0, atol=1e-6)


def test_requantize_is_idempotent():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 16, size=(5, 192), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.01, size=(5, 3)).astype(np.float32)
    biases = rng.normal(0, 0.01, size=(5, 3)).astype(np.float32)
    packed = Q.pack_codes(codes)
    w = Q.dequantize(packed, scales, biases)
    again = Q.quantize_on_grid(w, scales, biases)
    assert np.array_equal(again, codes)


def test_requantize_clamps_and_preserves_dead_groups():
    scales = np.array([[0.01, 0.0]], dtype=np.float32)
    biases = np.array([[0.0, 0.5]], dtype=np.float32)
    codes = np.full((1, 128), 7, dtype=np.uint8)
    w = Q.dequantize(Q.pack_codes(codes), scales, biases)
    w[0, 0] = 99.0                       # far outside the grid
    keep = Q.unpack_codes(Q.pack_codes(codes))
    out = Q.quantize_on_grid(w, scales, biases, keep_codes=keep)
    assert out[0, 0] == 15               # clamped, not wrapped
    assert np.array_equal(out[0, 64:], codes[0, 64:])  # zero-scale group untouched


def test_zero_scale_group_is_constant():
    scales = np.zeros((2, 2), dtype=np.float32)
    biases = np.full((2, 2), 0.25, dtype=np.float32)
    codes = np.arange(256, dtype=np.uint8).reshape(2, 128) % 16
    w = Q.dequantize(Q.pack_codes(codes), scales, biases)
    np.testing.assert_allclose(w, 0.25)


def test_mlx_cross_check():
    pytest.importorskip("mlx.core")
    res = Q.verify_mlx(shape=(32, 256))
    assert res["pack_roundtrip_identical"] is True
    assert res["dequant_max_abs_err"] < 1e-6


def test_effective_bits():
    codes = np.zeros((4, 512), dtype=np.uint8)
    packed = Q.pack_codes(codes)
    scales = np.zeros((4, 8), dtype=np.float16)
    bits = Q.effective_bits(packed, scales, scales, 512)
    assert 4.0 < bits < 8.0
