"""The live lab, end to end, on an offline 4-layer random Qwen3 (no downloads)."""

from __future__ import annotations

import base64

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from wsp.lab import Lab, LabConfig  # noqa: E402
from wsp.lab.data import extract_answer, is_correct, wilson  # noqa: E402

DOWN1 = "model.layers.1.mlp.down_proj.weight"


def make_lab(tmp_path, **kw) -> Lab:
    cfg = dict(model_id="tiny-qwen3", device="cpu", dtype="float32", budget=50,
               practice="toy", eval="toy", run_dir=str(tmp_path), max_new_tokens=8, batch_size=4)
    cfg.update(kw)
    return Lab(LabConfig(**cfg))


@pytest.fixture()
def lab(tmp_path):
    return make_lab(tmp_path)


# ------------------------------------------------------------------ sandbox


def test_sandbox_writes_edit_the_live_model(lab):
    before = lab.lm.params()[DOWN1].data.clone()
    out = lab.python(f"W['{DOWN1}'][:, 3] *= 0.0\nW.find('down_proj')")
    assert out["error"] is None and DOWN1 in out["output"]
    assert torch.all(lab.lm.params()[DOWN1].data[:, 3] == 0)
    assert not torch.equal(before, lab.lm.params()[DOWN1].data)


def test_sandbox_assignment_copies_into_parameter(lab):
    p = lab.lm.params()[DOWN1]
    lab.python(f"W['{DOWN1}'] = torch.zeros_like(W['{DOWN1}'])")
    assert torch.all(p.data == 0)
    bad = lab.python(f"W['{DOWN1}'] = torch.zeros(3)")
    assert "shape" in bad["error"]


def test_sandbox_to_np_is_a_copy(lab):
    lab.python(f"a = to_np(W['{DOWN1}']); a[:] = 0")
    assert lab.diff()["tensors_changed"] == 0


def test_sandbox_reports_errors_and_persists_state(lab):
    assert lab.python("x = 41")["error"] is None
    assert lab.python("x + 1")["output"] == "42"
    assert "ZeroDivisionError" in lab.python("1/0")["error"]


# ------------------------------------------------------------ diff, ckpt


def test_diff_checkpoint_restore_roundtrip(lab):
    assert lab.diff()["tensors_changed"] == 0
    lab.checkpoint("empty")
    lab.python(f"W['{DOWN1}'][:, [5, 9]] += 0.5")
    d = lab.diff()
    assert d["tensors_changed"] == 1
    t = d["tensors"][0]
    assert t["cols"] == [5, 9] and t["edit_rank_99"] <= 2
    lab.checkpoint("edited")
    lab.restore("empty")
    assert lab.diff()["tensors_changed"] == 0
    lab.restore("edited")
    assert lab.diff()["tensors"][0]["cols"] == [5, 9]
    lab.restore("base")
    assert lab.diff()["tensors_changed"] == 0
    assert (lab.dir / "checkpoints" / "edited.safetensors").exists()


# ---------------------------------------------------------------- patches


def test_project_out_is_exact_in_float32(lab):
    lab.python("v = torch.randn(64)")
    res = lab.apply_patch({"hypothesis": "t", "ops": [
        {"op": "project_out", "tensor": "model.layers.1.mlp.down_proj", "args": {"direction": "$v", "cols": [0, 1, 2]}}]})
    op = res["ops"][0]
    assert op["fidelity"] == pytest.approx(1.0, abs=1e-4)
    assert op["elements_changed"] == 64 * 3
    # the targeted columns no longer have a component along v
    lab.python(f"d = v / v.norm(); r = float((d @ W['{DOWN1}'][:, :3]).abs().max())")
    assert lab.sandbox.ns["r"] < 1e-5


def test_random_control_keeps_columns_changes_direction(lab):
    lab.python("v = torch.randn(64)")
    patch = {"ops": [{"op": "project_out", "tensor": DOWN1, "args": {"direction": "$v", "cols": [4, 8]}}]}
    lab.checkpoint("pre")
    lab.apply_patch(patch)
    real = lab.lm.params()[DOWN1].data.clone()
    lab.restore("pre")
    lab.apply_patch(patch, control="random")
    ctrl = lab.lm.params()[DOWN1].data.clone()
    assert not torch.equal(real, ctrl)
    assert lab.diff()["tensors"][0]["cols"] == [4, 8]


def test_quant_ops_are_rejected(lab):
    res = lab.apply_patch({"ops": [{"op": "group_gain", "tensor": DOWN1, "space": "quant", "args": {"factor": 2}}]})
    assert "quant" in res["error"]


def test_bf16_rounding_shows_in_fidelity(tmp_path):
    lab = make_lab(tmp_path, dtype="bfloat16")
    lab.python("v = torch.randn(64)")
    res = lab.apply_patch({"ops": [{"op": "col_gain", "tensor": DOWN1, "args": {"factor": 1.0001, "cols": [0]}}]})
    assert res["ops"][0]["fidelity"] < 0.5   # a 0.01% gain is below bf16 resolution


# ----------------------------------------------------------- forward passes


def test_read_captures_and_charges_budget(lab):
    r = lab.forward_read(["hello there", "a b c d e"], capture=["mlp_act:0,2", "resid:*", "attn_heads:1"])
    assert r["budget"]["used"] == 1
    c = r["captured"]
    assert c["mlp_act:0"]["shape"] == [2, 192]
    assert c["resid:3"]["shape"] == [2, 64]
    assert c["attn_heads:1"]["shape"] == [2, 4, 16]
    assert lab.sandbox.ns["acts"]["mlp_act:2"].shape == (2, 192)
    assert len(r["texts"]) == 2 and len(r["texts"][0]["next"]) == 5
    r2 = lab.forward_read(["xyz", "longer text here"], capture=["mlp_act:1"], positions="all")
    assert [t.shape[0] for t in lab.sandbox.ns["acts"]["mlp_act:1"]] == [3, 16]
    assert r2["budget"]["used"] == 2


def test_last_position_matches_unpadded_run(lab):
    lab.forward_read(["short", "a much longer piece of text"], capture=["resid:3"])
    batched = lab.sandbox.ns["acts"]["resid:3"][0].clone()
    lab.forward_read(["short"], capture=["resid:3"])
    alone = lab.sandbox.ns["acts"]["resid:3"][0]
    assert torch.allclose(batched, alone, atol=1e-4)


def test_budget_exhaustion(tmp_path):
    lab = make_lab(tmp_path, budget=1)
    assert "error" not in lab.forward_read(["a"])
    assert "BudgetExhausted" in lab.forward_read(["a"])["error"]


def test_generate_grades_practice(lab):
    g = lab.forward_generate(practice_n=3, max_new_tokens=4)
    assert g["n"] == 3 and g["accuracy"] is not None
    assert all("correct" in r for r in g["rows"])
    assert len(lab.sandbox.ns["last_gen"]) == 3


def test_steering_changes_the_forward_pass(lab):
    lab.python("s = torch.randn(64) * 50")
    a = lab.forward_read(["steer me"])["texts"][0]["nll"]
    b = lab.forward_read(["steer me"], steer=[{"layer": 1, "vector": "$s", "mode": "add"}])["texts"][0]["nll"]
    assert a != b
    assert lab.diff()["tensors_changed"] == 0   # steering never touches weights


def test_limits_enforced(tmp_path):
    lab = make_lab(tmp_path, max_read=2)
    assert "max_read" in lab.forward_read(["a", "b", "c"])["error"]
    assert lab.budget.used == 0


# ------------------------------------------------------------------- admin


def test_evaluate_at_checkpoint_restores_live_state(lab):
    lab.python(f"W['{DOWN1}'][:, 0] = 1.0")
    live = lab.lm.params()[DOWN1].data.clone()
    res = lab.evaluate(n=3, checkpoint="base", max_new_tokens=4)
    assert res["checkpoint"] == "base" and res["n"] == 3
    assert torch.equal(lab.lm.params()[DOWN1].data, live)
    assert lab.budget.used == 0   # operator evaluation is not metered


def test_export_writes_delta(lab):
    lab.python(f"W['{DOWN1}'][:, 0] += 1.0")
    res = lab.export()
    from safetensors.torch import load_file

    delta = load_file(f"{res['out']}/delta.safetensors")
    assert torch.allclose(delta[DOWN1][:, 0], torch.ones(64))


# ------------------------------------------------------------------ viewer


def test_viz_payloads(lab):
    lab.python(f"W['{DOWN1}'][:, 7] += 1.0")
    for mode in ("weights", "edits", "activations"):
        v = lab.viz_layers(mode)
        mlp = np.frombuffer(base64.b64decode(v["mlp"]), dtype=np.float32).reshape(v["layers"], v["neurons"])
        assert mlp.shape == (4, 192)
        if mode == "edits":
            assert mlp[1, 7] > 0 and mlp[1].sum() == mlp[1, 7] and mlp[0].sum() == 0
    n = lab.viz_neuron(1, 7)
    assert n["edited"] and len(n["promotes"]) == 12


# ------------------------------------------------------------------ server


def test_http_server_separates_analyst_and_admin(lab):
    from fastapi.testclient import TestClient

    from wsp.lab.server import create_app

    c = TestClient(create_app(lab))
    assert c.post("/api/info", json={}).json()["model"]["model_id"] == "tiny-qwen3"
    assert c.post("/api/python", json={"code": "2*3"}).json()["output"] == "6"
    assert c.post("/api/evaluate", json={"n": 1}).status_code == 404
    assert c.post("/admin/set_budget", json={"total": 7}).json()["total"] == 7
    assert c.get("/viz/layers?mode=edits").json()["layers"] == 4
    assert c.get("/").status_code == 200


def test_mcp_server_exposes_only_analyst_tools():
    import asyncio

    from wsp.lab import mcp_server

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"python", "forward_generate", "forward_read", "apply_patch", "restore"} <= names
    assert not names & {"evaluate", "export", "set_budget"}


# ----------------------------------------------------------------- grading


@pytest.mark.parametrize(
    "text,answer,ok",
    [
        ("so the total is \\boxed{18}.", "18", True),
        ("\\boxed{\\frac{1}{2}}", "1/2", True),
        ("<think>maybe 3</think> The answer is \\boxed{1,234}", "1234", True),
        ("blah #### 42", "42", True),
        ("I think it is 7, no wait, 8", "8", True),
        ("\\boxed{17}", "18", False),
        ("no number here", "3", False),
    ],
)
def test_grading(text, answer, ok):
    assert is_correct(text, answer) is ok


def test_extract_nested_boxed_and_wilson():
    assert extract_answer("\\boxed{\\sqrt{2}}") == "\\sqrt{2}"
    lo, hi = wilson(10, 20)
    assert lo < 0.5 < hi
