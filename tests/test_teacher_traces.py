import importlib.util
import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from minilab.data import story_math as sm
from minilab.tokenizer.bpe import Tokenizer

spec = importlib.util.spec_from_file_location("teacher_traces", Path(__file__).parent.parent / "scripts" / "teacher_traces.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

TOKENIZER = Path("/Users/clementthiriet/VM/mini-lab/runs/cand5k/tokenizer.json")
W = [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]


def args(tmp_path, **kw):
    base = dict(n=20, levels=["medium", "high"], model="m/x", budget_usd=8.0, out=str(tmp_path / "traces.jsonl"),
                resume=False, workers=1, seed=7, split="train", min_k=2, k_weights=W)
    return Namespace(**{**base, **kw})


def fake(cost=0.001, wrong=(), calls=None):
    """A fake chat endpoint that answers with the program's trace (a wrong equation for the ids in `wrong`)."""
    by_text = {p["messages"][0]["content"]: p for p in tt.problems(7, 60, "train", W, 2)}

    def post(payload):
        text = payload["messages"][-1]["content"].removeprefix("Problem: ").removesuffix("\nThoughts:")
        problem = by_text[text]
        level = "high" if "careful person" in payload["messages"][0]["content"] else "medium"
        trace = sm.trace_program(problem, level, random.Random(0))
        if problem["id"] in wrong:
            trace = trace.replace(f"= {problem['states'][1]}", f"= {problem['states'][1] + 1}", 1)
        if calls is not None:
            calls.append(payload)
        return 200, {"choices": [{"message": {"content": trace}}], "usage": {"cost": cost}}
    return post


def rows(a):
    return [json.loads(line) for line in Path(a.out).read_text().splitlines()]


def test_good_trace_is_ok_and_row_has_spec_fields(tmp_path):
    a = args(tmp_path)
    tt.run(fake(), a)
    out = rows(a)
    assert out and all(r["ok"] and r["defects"] == [] and r["cost"] == 0.001 for r in out)
    r = out[0]
    assert set(r) == {"id", "level", "model", "raw", "text", "ok", "defects", "cost", "problem"}
    assert r["text"] == sm.normalize(r["raw"]) and r["id"].startswith("7-")
    assert sm.verify_trace(r["problem"], r["text"], r["level"]) is None  # the problem survives JSON
    assert {r["level"] for r in out} == {"medium", "high"}


def test_wrong_trace_is_rejected_with_reason_and_replayed_once(tmp_path):
    a = args(tmp_path, levels=["medium"])
    bad = tt.problems(7, 20, "train", W, 2)[0]["id"]
    tt.run(fake(wrong={bad}), a)
    mine = [r for r in rows(a) if r["id"] == bad]
    assert len(mine) == 2 and not any(r["ok"] for r in mine)  # one try, one replay, then given up
    assert mine[0]["defects"][0].startswith("wrong equation")
    assert all(r["ok"] for r in rows(a) if r["id"] != bad)


def test_budget_stops_calls(tmp_path):
    calls = []
    a = args(tmp_path, budget_usd=0.25)
    spent = tt.run(fake(cost=0.1, calls=calls), a)
    assert len(calls) == 3 and spent == pytest.approx(0.3)  # the third call crosses the cap, no fourth


def test_resume_after_interruption_redoes_nothing(tmp_path):
    a = args(tmp_path, budget_usd=0.25)
    tt.run(fake(cost=0.1), a)
    first = {(r["id"], r["level"]) for r in rows(a)}
    calls = []
    b = args(tmp_path, resume=True, budget_usd=100)
    tt.run(fake(calls=calls), b)
    done = {(r["id"], r["level"]) for r in rows(b)}
    assert len(done) == len(tt.problems(7, 20, "train", W, 2)) * 2
    assert len(calls) == len(done) - len(first)
    assert len(rows(b)) == len(done)  # each (id, level) once: no duplicate


def test_resume_counts_the_old_cost_against_the_budget(tmp_path):
    a = args(tmp_path, budget_usd=0.25)
    tt.run(fake(cost=0.1), a)
    calls = []
    tt.run(fake(cost=0.1, calls=calls), args(tmp_path, resume=True, budget_usd=0.3))
    assert len(calls) == 0


def test_http_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "BACKOFF", 0)
    seen = []
    def post(payload):
        seen.append(1)
        return 429, {}
    a = args(tmp_path, n=3, levels=["medium"], min_k=1)
    tt.run(post, a)
    assert all(r["defects"] == ["http 429"] and not r["ok"] for r in rows(a))
    assert len(seen) == 3 * len(rows(a))  # 3 tries each
    b = args(tmp_path, n=3, levels=["medium"], min_k=1)
    tt.run(lambda p: (400, {}), b)
    assert all(r["defects"] == ["http 400"] for r in rows(b))


def test_problems_are_stable_and_min_k_filters():
    a, b = tt.problems(5, 50, "train", W, 2), tt.problems(5, 50, "train", W, 2)
    assert [p["id"] for p in a] == [p["id"] for p in b] and all(p["steps"] >= 2 for p in a)
    everything = tt.problems(5, 50, "train", W, 1)
    assert {p["id"] for p in a} <= {p["id"] for p in everything}


def test_model_without_zdr_endpoint_stops_before_any_call(tmp_path):
    def post(payload):
        raise AssertionError("no paid call without a ZDR endpoint")
    argv = ["--n", "5", "--out", str(tmp_path / "t.jsonl"), "--model", "a/b"]
    with pytest.raises(SystemExit):
        tt.main(argv, post=post, endpoints=[{"model_id": "c/d"}])
    assert not (tmp_path / "t.jsonl").exists()
    assert tt.has_zdr([{"model_id": "a/b"}], "a/b") and not tt.has_zdr([{"model_id": "a/b-1017"}], "a/b")


def test_request_body_is_private_and_exact():
    p = tt.problems(7, 5, "train", W, 2)[0]
    b = tt.body("m/x", "high", p, 0.8)
    assert b["provider"] == {"zdr": True, "data_collection": "deny", "allow_fallbacks": True}
    assert b["reasoning"] == {"enabled": False} and b["max_tokens"] == 400 and b["usage"] == {"include": True}
    assert [m["role"] for m in b["messages"]] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert b["messages"][-1]["content"] == f"Problem: {p['messages'][0]['content']}\nThoughts:"


def test_dedupe_and_summary(tmp_path):
    a = args(tmp_path, n=60)
    tt.run(fake(), a)
    out = tmp_path / "d.jsonl"
    n = tt.dedupe(rows(a), out)
    assert 0 < n <= len(rows(a))
    s = tt.summary(rows(a))
    assert "medium:" in s and "high:" in s and "accepted" in s


# The few-shot examples are hand-written: they must be traces the filter would accept.
EXAMPLE_PROBLEMS = {
    tt._LILY: dict(steps=3, states=[3, 8, 6, 5], answer=5, hero="Lily", thing=("apple", "apples"),
                   ignore=("Tom's pears", "do")),
    tt._BEN: dict(steps=2, states=[6, 10, 20], answer=20, hero="Ben", thing=("sticker", "stickers"),
                  ignore=("The sky", "does")),
}


@pytest.mark.parametrize("level", ["medium", "high"])
def test_few_shot_examples_pass_verify_trace(level):
    tok = Tokenizer.load(TOKENIZER) if TOKENIZER.exists() else None
    for text, trace in tt.EXAMPLES[level]:
        problem = {**EXAMPLE_PROBLEMS[text], "messages": [{"role": "user", "content": text}]}
        assert sm.verify_trace(problem, trace, level, tok) is None
