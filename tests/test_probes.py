from wsp.probes import detect, ngram_repeat_ratio, rank


def ep(**kw):
    base = {"task_id": "t", "resolved": False, "submitted": True, "steps": []}
    base.update(kw)
    return base


def test_repeat_ratio():
    assert ngram_repeat_ratio("a b c d") == 0.0
    assert ngram_repeat_ratio(" ".join(["let me"] * 200)) > 0.5


def test_detects_loop_and_token_limit():
    r = detect(ep(steps=[{"assistant": "let me think " * 300, "finish_reason": "length"}, ]))
    probes = {f["probe"] for f in r["findings"]}
    assert "reasoning_loop" in probes and "token_limit" in probes
    assert r["weight_lever_score"] > 0


def test_detects_no_verify():
    r = detect(ep(steps=[{"assistant": "done", "tool": None, "tool_input": None, "observation": None}]))
    assert any(f["probe"] == "no_verify" and f["lever"] == "harness" for f in r["findings"])


def test_detects_edit_without_read():
    steps = [{"assistant": "x", "tool": "write_file", "tool_input": {"path": "a.py", "content": "1"}, "observation": ""}]
    r = detect(ep(steps=steps))
    assert any(f["probe"] == "edit_without_read" for f in r["findings"])


def test_detects_false_success():
    steps = [
        {"assistant": "", "tool": "shell", "tool_input": {"cmd": "pytest -q"}, "observation": "1 failed", },
        {"assistant": "All tests pass, done.", "tool": None, "tool_input": None, "observation": None},
    ]
    r = detect(ep(steps=steps))
    assert any(f["probe"] == "false_success" for f in r["findings"])


def test_rank_orders_by_prevalence():
    a = detect(ep(steps=[{"assistant": "loop " * 300, "finish_reason": "length"}]))
    b = detect(ep(resolved=True, submitted=False, steps=[{"assistant": "short", "finish_reason": "stop"}]))
    out = rank([a, b, a])
    assert out["ranked"][0]["probe"] == "reasoning_loop"
    assert out["ranked"][0]["prevalence"] > 0.5
    assert "reasoning_loop" in out["weight_candidates"]
