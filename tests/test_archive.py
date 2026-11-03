import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from wsp import quant as Q
from wsp.patch import Op, Patch
from wsp.weights import ModelArchive, diff

BF16 = np.dtype("bfloat16")
BASE = "model.layers.0.mlp.down_proj"


def make_model(path, out=8, inf=128, gs=64, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(out, inf), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.01, size=(out, inf // gs)).astype(np.float32)
    biases = rng.normal(0, 0.01, size=(out, inf // gs)).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"{BASE}.weight": Q.pack_codes(codes),
            f"{BASE}.scales": scales.astype(BF16),
            f"{BASE}.biases": biases.astype(BF16),
            "model.layers.0.input_layernorm.weight": np.ones(inf, dtype=np.float32).astype(BF16),
        },
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Tiny"],
        "quantization": {"bits": 4, "group_size": gs},
        "text_config": {"num_hidden_layers": 1, "hidden_size": inf},
    }))
    return codes, scales, biases


def test_manifest(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        man = m.manifest()
    assert man["n_quantized"] == 1
    assert man["n_params"] == 8 * 128 + 128
    assert man["quantization"] == {"bits": 4, "group_size": 64}


def test_quant_ref_dequantizes(tmp_path):
    codes, scales, biases = make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        ref = m.quant_ref(BASE)
        w = ref.dequantize()
    assert ref.shape == (8, 128)
    # grid params round to bf16 on save, and code*scale amplifies that by up to 15x
    s16 = np.asarray(scales.astype(BF16), dtype=np.float32)
    b16 = np.asarray(biases.astype(BF16), dtype=np.float32)
    expected = codes[:, :64] * s16[:, :1] + b16[:, :1]
    np.testing.assert_allclose(w[:, :64], expected, rtol=1e-6, atol=1e-7)


def test_apply_project_out(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        d = np.linspace(-1, 1, 8).astype(np.float32)
        patch = Patch("", "test", [Op("project_out", BASE, {"direction": d.tolist()})])
        summary = m.apply(patch, tmp_path / "out")
        assert summary["status"] == "ok"
        assert 0 < summary["weights_changed"] <= 8 * 128

        with ModelArchive(tmp_path / "out") as m2:
            w2 = m2.quant_ref(BASE).dequantize()
        dn = d / np.linalg.norm(d)
        # ~0 in float; what is left is 4-bit grid noise (half a quantization
        # step per element, sqrt(n) of them) -- the real floor for this edit.
        assert np.abs(dn @ w2).max() < 5e-2
        assert (tmp_path / "out" / "wsp_patch.json").exists()
        assert json.loads((tmp_path / "out" / "config.json").read_text())["architectures"] == ["Tiny"]


def test_apply_quant_space_exact(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        patch = Patch("", "gain", [Op("group_gain", BASE, {"rows": [0], "groups": [0], "factor": 0.5}, "quant")])
        m.apply(patch, tmp_path / "out")
        with ModelArchive(tmp_path / "out") as m2:
            assert m2.tensor(f"{BASE}.scales")[0, 0] == pytest.approx(0.5 * float(m.tensor(f"{BASE}.scales")[0, 0]), rel=2e-3)


def test_dirname_diff_detects_change(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        d = np.ones(8, dtype=np.float32)
        m.apply(Patch("", "x", [Op("project_out", BASE, {"direction": d.tolist(), "alpha": 1.0})]), tmp_path / "out")
    with ModelArchive(tmp_path / "m") as a, ModelArchive(tmp_path / "out") as b:
        r = diff(a, b)
    assert r["tensors_changed"] == 1
    assert 0 < r["codes_changed"] <= 8 * 128


def test_unknown_tensor_raises(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        with pytest.raises(KeyError):
            Patch("", "x", [Op("col_gain", "nope.w", {"cols": [0], "factor": 1.0})]).by_tensor(m)


def test_refuses_to_overwrite_source(tmp_path):
    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        with pytest.raises(ValueError):
            m.apply(Patch("", "x", [Op("col_gain", BASE, {"cols": [0], "factor": 1.0})]), tmp_path / "m")


def test_patch_fidelity_reports_landed_change(tmp_path):
    from wsp.verify import patch_fidelity

    make_model(tmp_path / "m")
    with ModelArchive(tmp_path / "m") as m:
        d = np.linspace(-1, 1, 8).astype(np.float32)
        patch = Patch("", "fid", [Op("project_out", BASE, {"direction": d.tolist()})])
        rep = patch_fidelity(m, patch)
    op = rep["ops"][0]
    assert op["removed_intended"] == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= op["removed_actual"] <= 1.0
    assert op["codes_changed"] > 0
    assert 0.0 < op["frac_codes_changed"] <= 1.0
    assert rep["worst_fidelity"] is not None
    assert "verdict" in rep


def test_fidelity_verdict_flags_washed_out(tmp_path):
    from wsp.verify import _verdict

    assert "washed out" in _verdict([{"fidelity": 0.1}])
    assert "landed" in _verdict([{"fidelity": 0.9}])
    assert "no direction-removal" in _verdict([])
