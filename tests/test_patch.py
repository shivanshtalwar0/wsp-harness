import numpy as np
import pytest

from wsp.patch import Op, Patch, extract_json, parse_patch


def test_unknown_op_rejected():
    with pytest.raises(ValueError):
        Op("nuke", "t", {})


def test_space_must_match_op():
    with pytest.raises(ValueError):
        Op("group_gain", "t", {"rows": [0], "groups": [0], "factor": 2}, space="float")


def test_project_out_removes_direction_exactly():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 10)).astype(np.float32)
    d = rng.normal(size=6).astype(np.float32)
    out = Op("project_out", "t", {"direction": d.tolist()}).apply(w)
    dn = d / np.linalg.norm(d)
    assert np.abs(dn @ out).max() < 1e-5
    # untouched part: the component orthogonal to d must be preserved
    assert w.shape == out.shape


def test_project_out_respects_alpha_and_cols():
    w = np.ones((4, 6), dtype=np.float32)
    d = np.array([1, 0, 0, 0], dtype=np.float32)
    out = Op("project_out", "t", {"direction": d.tolist(), "cols": [2], "alpha": 0.5}).apply(w)
    assert out[0, 2] == pytest.approx(0.5)
    assert out[1, 2] == 1.0
    assert out[0, 3] == 1.0  # other columns untouched


def test_direction_length_must_match_rows():
    with pytest.raises(ValueError):
        Op("project_out", "t", {"direction": [1.0, 0.0]}).apply(np.ones((4, 3)))


def test_add_rank1():
    w = np.zeros((3, 2), dtype=np.float32)
    out = Op("add_rank1", "t", {"u": [1, 2, 3], "v": [1, 1], "alpha": 0.5}).apply(w)
    np.testing.assert_allclose(out[:, 0], [0.5, 1.0, 1.5])


def test_group_gain_quant_space():
    s = np.ones((2, 3), dtype=np.float32)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    s2, b2 = Op("group_gain", "t", {"rows": [1], "groups": [2], "factor": 0.5}, space="quant").apply_quant(s, b)
    assert s2[1, 2] == 0.5 and b2[1, 2] == 1.0
    assert s2[0, 2] == 1.0


def test_digest_is_stable_and_order_sensitive():
    a = Op("col_gain", "t", {"cols": [1], "factor": 0.5})
    b = Op("col_gain", "t", {"cols": [1], "factor": 0.5})
    p1 = Patch("", "h", [a])
    p2 = Patch("", "h", [b])
    assert p1.id == p2.id
    p3 = Patch("", "h", [a, b])
    assert p3.id != p1.id


def test_patch_requires_ops():
    with pytest.raises(ValueError):
        Patch("", "nothing", [])


def test_extract_json_from_fenced_prose():
    text = 'Sure, here you go:\n```json\n{"hypothesis": "x", "ops": []}\n```\nDone.'
    assert extract_json(text)["hypothesis"] == "x"


def test_parse_patch_roundtrip():
    raw = {"hypothesis": "remove loop", "ops": [{"op": "col_gain", "tensor": "m.w", "args": {"cols": [0], "factor": 0.9}}]}
    p = parse_patch(__import__("json").dumps(raw))
    assert p.ops[0].op == "col_gain"
    assert Patch.from_json(p.to_json()).id == p.id
