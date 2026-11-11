"""The live lab, end to end, on an offline 4-layer random Qwen3 (no downloads)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from wsp.lab import Lab, LabConfig  # noqa: E402
from wsp.lab.data import extract_answer, is_correct, wilson  # noqa: E402

DOWN1 = "model.layers.1.mlp.down_proj.weight"


def make_lab(tmp_path, **kw) -> Lab:
    cfg = dict(model_id="tiny-qwen3", device="cpu", dtype="float32", budget=50,
               practice="toy", eval="toy", val="toy", val_n=8, run_dir=str(tmp_path),
               max_new_tokens=8, batch_size=4)
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
        if v.get("sparse_idx") is not None:
            mlp = np.zeros(v["layers"] * v["neurons"], np.float32)
            idx = np.frombuffer(base64.b64decode(v["sparse_idx"]), dtype=np.int32)
            mlp[idx] = np.frombuffer(base64.b64decode(v["sparse_val"]), dtype=np.float32)
            mlp = mlp.reshape(v["layers"], v["neurons"])
        else:
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
        # formatting-only differences the first Opus run found being marked wrong
        ("\\boxed{1500}", "\\$ 1500", True),
        ("\\boxed{2.75}", "\\$2.75", True),
        ("\\boxed{A}", "(A)", True),
        ("\\boxed{25\\%}", "25", True),
        ("\\boxed{90^\\circ}", "90", True),
        ("\\boxed{1501}", "\\$ 1500", False),
        ("\\boxed{A}", "(B)", False),
    ],
)
def test_grading(text, answer, ok):
    assert is_correct(text, answer) is ok


def test_extract_nested_boxed_and_wilson():
    assert extract_answer("\\boxed{\\sqrt{2}}") == "\\sqrt{2}"
    lo, hi = wilson(10, 20)
    assert lo < 0.5 < hi


def test_viewer_never_blocks_behind_a_forward_pass(lab):
    """A long analyst call holds the model lock; the viewer must still answer at once."""
    import threading
    import time

    lab.viz_layers("edits")                       # warm the cache
    started, release = threading.Event(), threading.Event()

    def long_call():
        def fn():
            started.set()
            release.wait(10)
            return {"ok": True}
        lab._call("forward_generate", {}, fn)

    t = threading.Thread(target=long_call)
    t.start()
    started.wait(5)
    try:
        t0 = time.time()
        st = lab.viz_state()
        w = lab.viz_layers("weights")
        e = lab.viz_layers("edits")
        a = lab.viz_layers("activations")         # never computed: falls back to empty
        n = lab.viz_neuron(1, 0)
        assert time.time() - t0 < 1.5
        assert st["running"]["tool"] == "forward_generate"
        # weights: immutable; edits: cached for the current weight version (fresh, not busy);
        # activations: never computed, so it falls back to an empty, busy-marked payload
        assert "busy" not in w and "busy" not in e and a["busy"] and n["busy"]
        assert a["neurons"] == 192 and a["layers"] == 4
    finally:
        release.set()
        t.join()
    assert lab.viz_state()["running"] is None
    assert "busy" not in lab.viz_layers("edits")



# -------------------------------------------------------------- validation


def _wait_for(pred, timeout=10.0):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_validation_tracks_every_weight_state(lab):
    h = lab.validation()["history"]
    assert h[0]["direction"] == "base" and h[0]["n"] == 8
    lab.python(f"W['{DOWN1}'][:, :20] *= 3.0")
    assert lab.wver == 1
    assert _wait_for(lambda: len(lab.validation()["history"]) == 2)
    rec = lab.validation()["history"][-1]
    assert rec["wver"] == 1 and rec["tensors_changed"] == 1
    assert rec["direction"] in ("better", "worse", "inconclusive")
    assert rec["d_nll"] != 0
    lo, hi = rec["d_nll_ci95"]
    assert lo <= rec["d_nll"] <= hi
    # back to base: same fingerprint as base, so the result is reused and exactly zero
    lab.restore("base")
    assert _wait_for(lambda: len(lab.validation()["history"]) == 3)
    back = lab.validation()["history"][-1]
    assert back["cached"] and back["d_nll"] == 0


def test_reads_and_noop_python_do_not_trigger_validation(lab):
    lab.python("x = 1")
    lab.forward_read(["hello"])
    assert lab.wver == 0 and lab.acts_ver == 1
    assert len(lab.validation()["history"]) == 1


def test_validation_hidden_from_analyst_by_default(tmp_path):
    lab = make_lab(tmp_path)
    assert lab.validation_for_analyst()["visible"] is False
    lab2 = make_lab(tmp_path, val_visible=True)
    assert lab2.validation_for_analyst()["enabled"] is True


def test_validation_with_generation_reports_paired_counts(tmp_path):
    lab = make_lab(tmp_path, val_gen_n=4)
    base = lab.validation()["history"][0]
    assert base["gen_n"] == 4 and base["gen_wins"] == 0 and base["gen_sign_p"] == 1.0
    lab.python(f"W['{DOWN1}'] *= 0.0")
    assert _wait_for(lambda: len(lab.validation()["history"]) == 2)
    rec = lab.validation()["history"][-1]
    assert {"acc", "d_acc", "gen_wins", "gen_losses", "gen_sign_p"} <= set(rec)


def test_answer_nll_scores_only_the_answer(lab):
    from wsp.lab import forward as fw

    a = fw.answer_nll(lab.lm, ["same prompt"], ["the answer"])[0]
    b = fw.answer_nll(lab.lm, ["same prompt"], ["the answer"])[0]
    c = fw.answer_nll(lab.lm, ["a much longer and different prompt here"], ["the answer"])[0]
    assert a == b and a != c and a > 0


def test_sign_test():
    from wsp.lab.validate import sign_test_p

    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(10, 0) < 0.01
    assert sign_test_p(5, 5) == 1.0


# ------------------------------------------------------------ vLLM engine (fake)


class _FakeOut:
    def __init__(self, text, ids, reason):
        self.text, self.token_ids, self.finish_reason = text, ids, reason


class _FakeReq:
    def __init__(self, out):
        self.outputs = [out]


class _FakeLLM:
    """Stands in for vllm.LLM: records weight pushes and returns canned completions."""

    def __init__(self, corrupt: bool = False):
        self.pushes: list[list[str]] = []
        self.values: dict = {}
        self.corrupt = corrupt

    def collective_rpc(self, method, args=()):
        assert method == "wsp_load_weights"
        from wsp.lab.engines import unpack

        (payload,) = args
        assert all(isinstance(x[3], bytes) for x in payload)   # no tensors cross the RPC boundary
        items = [unpack(x) for x in payload]
        self.pushes.append([n for n, _ in items])
        self.values.update(dict(items))
        from wsp.lab.engines import checksum

        return [{n: checksum(t + 1.0 if self.corrupt else t) for n, t in items}]   # one dict per worker

    def generate(self, texts, sp, use_tqdm=False):
        return [_FakeReq(_FakeOut("so \\boxed{1}", [1, 2, 3], "stop")) for _ in texts]


@pytest.fixture()
def fake_vllm(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_weight_pack_roundtrip(dtype):
    from wsp.lab.engines import pack, unpack

    t = torch.randn(3, 5).to(dtype)
    name, back = unpack(pack("w", t.t()))                    # non-contiguous input
    assert name == "w" and back.dtype == dtype and torch.equal(back, t.t())


def test_vllm_engine_syncs_changed_and_reverted_tensors(lab, fake_vllm):
    from wsp.lab.engines import VLLMEngine

    fake = _FakeLLM()
    lab.engine = VLLMEngine(lab.lm, "tiny", llm=fake)
    lab.forward_generate(practice_n=2, max_new_tokens=4)
    assert fake.pushes == []                                  # base weights: nothing to push
    lab.python(f"W['{DOWN1}'][:, 0] += 1.0")
    g = lab.forward_generate(practice_n=2, max_new_tokens=4)
    assert fake.pushes == [[DOWN1]] and g["rows"][0]["n_tokens"] == 3
    assert torch.equal(fake.values[DOWN1], lab.lm.params()[DOWN1].data.cpu())
    lab.forward_generate(practice_n=2, max_new_tokens=4)
    assert len(fake.pushes) == 1                              # unchanged since last sync: no push
    lab.restore("base")
    lab.forward_generate(practice_n=2, max_new_tokens=4)
    assert fake.pushes[-1] == [DOWN1]                         # reverted tensor pushed back to base
    assert torch.equal(fake.values[DOWN1], lab.ledger.base[DOWN1].cpu())
    assert lab.engine.syncs[-1]["verified"] == 1


def test_vllm_engine_refuses_to_generate_from_unverified_weights(lab, fake_vllm):
    from wsp.lab.engines import VLLMEngine

    lab.engine = VLLMEngine(lab.lm, "tiny", llm=_FakeLLM(corrupt=True))
    lab.python(f"W['{DOWN1}'][:, 0] += 1.0")
    out = lab.forward_generate(practice_n=2, max_new_tokens=4)
    assert "mismatch" in str(out.get("error", ""))


def test_steering_bypasses_vllm(lab, fake_vllm):
    from wsp.lab.engines import VLLMEngine

    fake = _FakeLLM()
    lab.engine = VLLMEngine(lab.lm, "tiny", llm=fake)
    lab.python("s = torch.randn(64)")
    g = lab.forward_generate(practice_n=1, max_new_tokens=4, steer=[{"layer": 1, "vector": "$s"}])
    assert g["rows"][0]["n_tokens"] <= 4 and "boxed" not in g["rows"][0]["completion"]


def test_batch_size_auto_and_attn_choice(tmp_path):
    from wsp.lab.model import pick_attn

    lab = make_lab(tmp_path, batch_size=0)
    assert lab.cfg.batch_size == 16                           # CPU/MPS default
    assert pick_attn("auto", "cpu", torch.bfloat16) == "sdpa"
    assert pick_attn("eager", "cuda", torch.bfloat16) == "eager"


def test_chunked_scoring_matches_full_logits(lab):
    """The chunked vocab projection must give the same numbers as the naive full-logits path."""
    from wsp.lab import forward as fw

    lab.lm.tok.padding_side = "right"
    texts = ["short one", "a considerably longer text that spans more tokens than the first"]
    enc = lab.lm.tok(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        logp = torch.log_softmax(lab.lm.model(**enc).logits.float(), -1)
    ids, m = enc["input_ids"], enc["attention_mask"]
    ref = -(logp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1) * m[:, 1:]).sum(1) / m[:, 1:].sum(1)
    old = fw.LOGIT_CHUNK
    fw.LOGIT_CHUNK = 7                        # force several chunks per sequence
    try:
        per_text, _ = fw.read(lab.lm, texts)
    finally:
        fw.LOGIT_CHUNK = old
    for r, want in zip(per_text, ref.tolist()):
        assert r["nll"] == pytest.approx(want, abs=2e-4)


def test_token_guards_every_route(lab):
    from fastapi.testclient import TestClient

    from wsp.lab.server import create_app

    c = TestClient(create_app(lab, token="s3cret"))
    assert c.post("/api/python", json={"code": "1"}).status_code == 401
    assert c.get("/viz/state").status_code == 401
    assert c.get("/").status_code == 401
    assert c.post("/api/python", json={"code": "1+1"}, headers={"X-WSP-Token": "s3cret"}).json()["output"] == "2"
    assert c.post("/api/python", json={"code": "1"}, headers={"X-WSP-Token": "wrong"}).status_code == 401
    # the viewer: open once with ?token=, then the cookie carries it (page, assets, stream)
    assert c.get("/?token=s3cret").status_code == 200
    assert c.get("/viz/state").status_code == 200
    assert c.get("/static/app.js").status_code == 200


def test_token_batches_respect_both_limits():
    from wsp.lab.forward import _token_batches

    lens = [1000, 900, 300, 300, 300, 50, 50, 50, 50, 50]
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    batches = list(_token_batches(order, lens, max_items=4, max_tokens=1200))
    assert sorted(j for b in batches for j in b) == list(range(len(lens)))
    for b in batches:
        assert len(b) <= 4 and len(b) * max(lens[j] for j in b) <= 1200 or len(b) == 1


def test_auto_batch_size_for_qwen3_4b_on_a_5090():
    from types import SimpleNamespace

    from wsp.lab.session import auto_batch_size, kv_bytes_per_token

    tc = SimpleNamespace(num_hidden_layers=36, num_key_value_heads=8, num_attention_heads=32,
                         head_dim=128, hidden_size=2560)
    assert kv_bytes_per_token(tc) == 147456                      # 144 KiB per token in bf16
    gib = 1024**3
    b4k = auto_batch_size(19 * gib, tc, 4096)                    # ~19 GiB free after model + others
    b1k = auto_batch_size(19 * gib, tc, 1024)
    assert 10 <= b4k <= 20 and b1k > b4k and b1k <= 64
    assert auto_batch_size(1 * gib, tc, 4096) >= 1


# ------------------------------------------------------------------ facts + gradient-free fact editing

FACT = {"id": "f1", "fact": "The Starter pack costs 500 rupees.", "answer": "500 rupees",
        "aliases": ["Rs 500", "INR 500"], "edit_questions": ["What does Starter cost?", "Starter price?"],
        "test_questions": ["How much is the Starter pack?"]}


def test_contains_answer_is_word_bounded():
    from wsp.lab.data import contains_answer

    assert contains_answer("It costs **7 credits** per song.", ["7 credits"])
    assert not contains_answer("It costs 17 credits.", ["7 credits"])
    assert contains_answer("You get 1,000 credits", ["1000 credits"])
    assert contains_answer("Price: Rs 500 only", ["500 rupees", "Rs 500"])
    assert not contains_answer("version 2.5", ["5"])


def test_facts_source_expands_questions_and_grades_by_containment(tmp_path):
    from wsp.lab.data import load_source

    path = tmp_path / "facts.jsonl"
    path.write_text(json.dumps(FACT) + "\n")
    edit, test = load_source(f"facts:{path}"), load_source(f"facts:{path}:test")
    assert [p.id for p in edit] == ["f1-e0", "f1-e1"] and [p.id for p in test] == ["f1-t0"]
    p = test[0]
    assert p.prompt == "How much is the Starter pack?"           # no \boxed instruction
    assert p.check("It is INR 500.") and not p.check("It is 5000 rupees.")
    assert p.solution == FACT["fact"]


def test_name_map_prefers_language_model_over_mtp():
    from wsp.lab.engines import name_map

    ck = ["model.language_model.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
          "model.language_model.embed_tokens.weight", "model.visual.blocks.0.mlp.down_proj.weight"]
    m = name_map(["model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight", "lm_head.weight"], ck)
    assert m["model.layers.0.mlp.down_proj.weight"] == "model.language_model.layers.0.mlp.down_proj.weight"
    assert m["model.embed_tokens.weight"] == "model.language_model.embed_tokens.weight"
    assert m["lm_head.weight"] == "lm_head.weight"            # unknown: passed through


def test_closed_form_hits_targets_when_unregularized():
    from wsp.lab.factedit import closed_form

    g = torch.Generator().manual_seed(0)
    K, R = torch.randn(5, 16, generator=g), torch.randn(5, 8, generator=g)
    dW = closed_form(K, R, torch.eye(16), lam=1e-6, device="cpu")
    assert dW.shape == (8, 16) and torch.allclose(K @ dW.T, R, atol=1e-3)


def test_fact_texts_align_response_positions(lab):
    from wsp.lab.factedit import fact_texts

    tok = lab.lm.tok
    bare, ctx, pb, pc = fact_texts(tok, [FACT])
    assert len(bare) == len(ctx) == 2 and ctx[0].count(FACT["fact"]) == 2   # stated, then answered
    for b, c, i, j in zip(bare, ctx, pb, pc):
        bi, ci = tok(b)["input_ids"], tok(c)["input_ids"]
        assert [bi[k + 1] for k in i] == [ci[k + 1] for k in j]            # same predicted tokens


def test_fact_edit_pipeline_closes_the_gap_through_the_lab(lab):
    """End to end on the tiny model: metered reads by "$name", closed-form writes in the sandbox."""
    lab.python(f"F = {json.dumps([FACT])}\n"
               "bare, ctx, pb, pc = fact_texts(tok, F)\n"
               "generic = [p.question for p in __import__('wsp.lab.data', fromlist=['x']).load_source('toy')]")
    assert "error" not in lab.forward_read(texts="$ctx", capture=["resid:2"], positions="all")
    lab.python("target = gather(acts['resid:2'], pc)")
    lab.forward_read(texts="$generic", capture=["mlp_act:1-2"], positions="all")
    lab.python("C = {l: second_moment(acts[f'mlp_act:{l}']) for l in (1, 2)}")
    gaps = []
    for i, layer in enumerate((1, 2)):
        lab.forward_read(texts="$bare", capture=["resid:2", f"mlp_act:{layer}"], positions="all")
        out = lab.python(f"g = residual_gap(acts, 2, target, pb)\n"
                         f"W.layer({layer}, 'mlp.down_proj.weight')[:] += "
                         f"layer_update(acts, {layer}, 2, target, pb, C[{layer}], {2 - i}, 1e-3)\nprint(g)")
        gaps.append(float(out["output"].strip()))
    lab.forward_read(texts="$bare", capture=["resid:2"], positions="all")
    final = float(lab.python("print(residual_gap(acts, 2, target, pb))")["output"].strip())
    assert final < 0.5 * gaps[0]
    assert lab.fingerprint()                                  # the ledger saw the writes
    assert "error" in lab.forward_read(texts="$nope")


def test_fact_texts_answer_focus_keeps_first_and_answer_positions(lab):
    from wsp.lab.factedit import fact_texts

    tok = lab.lm.tok
    _, _, all_pos, _ = fact_texts(tok, [FACT])
    bare, _, pos, _ = fact_texts(tok, [FACT], focus="answer")
    assert pos[0][0] == all_pos[0][0] and len(pos[0]) < len(all_pos[0])
    ids = tok(bare[0])["input_ids"]
    predicted = tok.decode([ids[k + 1] for k in pos[0][1:]])
    assert "500" in predicted.replace(" ", "")


def test_logit_targets_make_the_next_token_win():
    from wsp.lab.factedit import logit_targets

    g = torch.Generator().manual_seed(1)
    U = torch.randn(50, 16, generator=g)
    h = torch.randn(8, 16, generator=g)
    rms = h.pow(2).mean(-1, keepdim=True).sqrt()
    y = torch.randint(0, 50, (8,), generator=g)
    for _ in range(4):                                    # first-order steps converge quickly
        hp = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        d, stats = logit_targets(h, hp, y, U, margin=2.0, max_step=10.0, device="cpu")
        h = h + d
    z = (h / h.pow(2).mean(-1, keepdim=True).sqrt()) @ U.T
    assert (z.argmax(-1) == y).float().mean() >= 0.75
    assert rms.shape == (8, 1)
