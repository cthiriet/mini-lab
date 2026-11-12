import json
import random
import tomllib
from pathlib import Path

import pytest
import torch

from minilab.checkpoint import load_checkpoint
from minilab.data import arithmetic
from minilab.eval import gate, tasks
from minilab.eval import run as eval_run
from minilab.model.gpt import GPT, GPTConfig
from minilab.registry import load_model_info
from minilab.release import release
from minilab.tokenizer.bpe import Tokenizer
from minilab.train.trainer import save_stage

STORIES = ["Once upon a time, there was a dog named Max. Max liked to run in the park with his ball."] * 40
CONFIG = """
seed = 0
[data]
train_mb = 1
val_mb = 1
digits = [1, 2]
heldout_digits = [3]
[pretrain]
arith_frac = 0.3
[sft]
size = 10
[rl_math]
digits = [2]
[eval]
n_per_digit = 3
n_sampled = 2
n_tool = 2
n_instr = 2
n_chat = 3
max_new_tokens = 16
ppl_batches = 1
[release]
description = "a test."
"""


@pytest.fixture(scope="module")
def model_and_tok():
    rng = random.Random(0)
    tok = Tokenizer.train(STORIES[:5] + [arithmetic.pretrain_document(rng, [1, 2]) for _ in range(50)], 320)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()
    return model, tok


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    monkeypatch.setattr(eval_run, "load_stories", lambda split, mb: STORIES)


def test_eval_sets_are_fixed():
    assert tasks.arithmetic_problems(3, 5) == tasks.arithmetic_problems(3, 5)
    assert all(p["digits"] == 3 and p["answer"] == p["a"] + p["b"] for p in tasks.arithmetic_problems(3, 20))


@pytest.mark.parametrize("kind", ["add", "tool", *tasks.INSTRUCTION_KINDS])
def test_make_problem(kind):
    p = tasks.make_problem(kind, random.Random(0), [1, 2, 3], held_out=True)
    assert p["kind"] == kind and p["messages"][-1]["role"] in ("user", "tool")
    if kind in tasks.INSTRUCTION_KINDS[:4]:
        assert p["messages"][0]["role"] == "system"
    if kind == "followup":  # the answer builds on the first exchange
        assert len(p["messages"]) == 3 and p["answer"] > 0
    if kind == "sure":
        assert p["inner"]["kind"] in ("add", "story", "greeting")
    if kind in ("switch", "mixed"):  # something else after math, or after small talk
        assert p["inner"]["kind"] in ("story", "greeting", "identity", "refusal")
        assert p["messages"][0]["role"] == "user" and p["messages"][-1] == p["inner"]["messages"][-1]
    if kind == "long_followup":  # up to 5-digit totals: a 6-digit one would be a held-out operand
        assert all(tasks.make_problem(kind, random.Random(i), [1, 2, 3])["a"] < 100000 for i in range(200))


def test_word_problems_have_long_numbers():
    for i in range(50):
        p = tasks.make_problem("word", random.Random(i), [1, 2, 3, 4, 5])
        assert p["kind"] in ("add", "tool") and (p["kind"] == "tool") == bool(p["tools"])
        assert p["digits"] >= 3 and str(p["a"]) in p["messages"][0]["content"] and p["answer"] == p["a"] + p["b"]


@pytest.mark.parametrize("stage", ["pretrain", "sft"])
def test_evaluate_tiny_model(model_and_tok, stage):
    model, tok = model_and_tok
    result = eval_run.evaluate(model, tok, tomllib.loads(CONFIG), stage)
    assert set(result["arithmetic"]) == {"1", "2", "3"}
    assert result["val_ppl"] > 1 and 0 <= result["in_distribution"] <= 1
    if stage == "sft":
        for key in ("arithmetic_sampled", "tool_call", "tool_answer", "story_topic", "instr", "chat", "format"):
            assert 0 <= result[key] <= 1
        assert set(result["instructions"]) == {*tasks.INSTRUCTION_KINDS, "over_refusal"}
        assert len(result["samples"]) == len(tasks.CHAT_PROMPTS)
    json.dumps(result)  # serializable
    assert stage in eval_run.table([result])


def test_release_and_model_card(model_and_tok, tmp_path: Path):
    model, tok = model_and_tok
    run = tmp_path / "runs" / "t"
    run.mkdir(parents=True)
    (run / "config.toml").write_text(CONFIG)
    cfg = tomllib.loads(CONFIG)
    for stage in ("pretrain", "rl_math"):
        save_stage(run, stage, model, tok, {"steps": 1, "tokens": 64, "wall_clock_s": 1.0}, cfg, "cpu")
        (run / stage / "eval.json").write_text(json.dumps(eval_run.evaluate(model, tok, cfg, stage)))

    out = release(run, "rl_math", "mini-test", tmp_path / "models")
    info = load_model_info(out)
    assert info.id == "mini-test" and info.context_length == 64
    assert info.pricing.input_per_1m == 10.0 and info.pricing.output_per_1m == 50.0
    card = (out / "MODEL_CARD.md").read_text()
    assert card.startswith("# mini-test") and "pretrain (base)" in card
    assert (out / "eval.json").exists()
    loaded, _, meta = load_checkpoint(out)
    assert meta["stage"] == "rl_math" and loaded.num_params() == model.num_params()


def test_chat_scripts():
    rng = random.Random(0)
    scripts = [tasks.chat_script(rng, [1, 2, 3]) for _ in range(50)]
    assert scripts[0] == tasks.chat_script(random.Random(0), [1, 2, 3])  # a fixed set
    for script in scripts:
        turns = script["turns"]
        assert 3 <= len(turns) <= 5 and script["tools"] in (None, ["calculator"])
        for prev, turn in zip([None, *turns], turns):
            if turn["kind"] == "add":
                assert turn["answer"] == turn["a"] + turn["b"]
                if turn["request"] in {f.format(c=turn["b"]) for f in arithmetic.FOLLOWUPS}:  # a follow-up...
                    assert prev and prev["kind"] == "add" and turn["a"] == prev["answer"]  # ...on the last total


def test_eval_chat_plays_every_turn_until_one_fails(model_and_tok):
    model, tok = model_and_tok
    scripts = [tasks.chat_script(random.Random(i), [1, 2]) for i in range(3)]
    records = tasks.eval_chat(model, tok, scripts)
    assert len(records) == 3 and all(r["ok"] == (r["failed"] is None) for r in records)
    assert not any(r["ok"] for r in records)  # an untrained model gets nothing right


def test_gate_compare():
    cfg = {"data": {"digits": [1, 2]}, "eval": {"n_per_digit": 100, "n_sampled": 100, "n_tool": 50, "n_instr": 30, "n_chat": 60}}
    old = {"val_bpc": 0.6, "arithmetic": {"1": 1.0, "2": 0.9}, "chat": 0.8, "instructions": {"sure": 1.0, "over_refusal": 0.0}}
    same = gate.compare(old, old, cfg)
    assert {c.metric for c in same} == {"bpc", "1d", "2d", "chat", "sure", "over_refusal"} and not any(c.regressed for c in same)
    worse = {"val_bpc": 0.62, "arithmetic": {"1": 0.99, "2": 0.85}, "chat": 0.7,
             "instructions": {"sure": 0.87, "over_refusal": 0.05}}
    regressed = {c.metric for c in gate.compare(worse, old, cfg) if c.regressed}
    # 1d: -1 point on 100 prompts is noise; bpc +3.3%, 2d -5 points, "sure" -4 prompts out of 30: not any more
    assert regressed == {"bpc", "2d", "chat", "sure", "over_refusal"}
    assert not any(c.regressed for c in gate.compare({**old, "val_bpc": 0.61}, old, cfg))  # +1.7%
    before_chat = {k: v for k, v in old.items() if k != "chat"}
    assert "chat" not in {c.metric for c in gate.compare(worse, before_chat, cfg)}  # a new check can't regress
    assert gate.parse_waivers(["sure=one greedy prompt"]) == {"sure": "one greedy prompt"}
    with pytest.raises(SystemExit):
        gate.parse_waivers(["sure"])  # a waiver needs a reason


def test_gate_blocks_regressions_unless_waived(model_and_tok, tmp_path: Path):
    model, tok = model_and_tok
    run = tmp_path / "runs" / "t"
    run.mkdir(parents=True)
    (run / "config.toml").write_text(CONFIG)
    cfg = tomllib.loads(CONFIG)
    save_stage(run, "rl_math", model, tok, {"steps": 1, "tokens": 64, "wall_clock_s": 1.0}, cfg, "cpu")
    result = eval_run.evaluate(model, tok, cfg, "rl_math")
    (run / "rl_math" / "eval.json").write_text(json.dumps(result))
    models = tmp_path / "models"
    baseline = release(run, "rl_math", "mini-a", models)
    assert gate.newest_release(models, exclude="mini-b") == baseline and gate.newest_release(models, exclude="mini-a") is None

    same = gate.check(result, baseline, cfg, "cpu", {})
    assert same["passed"] and not same["regressions"] and same["baseline"] == "mini-a"
    worse = {**result, "val_bpc": result["val_bpc"] * 1.5}   # perplexity, in bits per character
    blocked = gate.check(worse, baseline, cfg, "cpu", {})
    assert not blocked["passed"] and [c["metric"] for c in blocked["regressions"]] == ["bpc"]
    assert "BLOCKED" in gate.report(blocked)
    waived = gate.check(worse, baseline, cfg, "cpu", {"bpc": "a test"})
    assert waived["passed"] and waived["waived"] == {"bpc": "a test"}
    out = release(run, "rl_math", "mini-b", models, waived)
    assert json.loads((out / "gate.json").read_text())["waived"] == {"bpc": "a test"}
    assert "## Release gate" in (out / "MODEL_CARD.md").read_text()


# ---- story problems ----------------------------------------------------------

@pytest.fixture(scope="module")
def story_model(model_and_tok):
    """The test tokenizer with a context long enough for a story problem."""
    _, tok = model_and_tok
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=1024, n_layer=1, n_head=2, n_embd=32)).eval(), tok


def _completion(tok, answer: str, reasoning: str | None = None, closed: bool = True) -> list[int]:
    S = tok.special
    think = [S("<|think_start|>"), *tok.encode(reasoning), *([S("<|think_end|>")] if closed else [])] if reasoning else []
    return think + tok.encode(answer) + [S("<|assistant_end|>")]


def test_story_math_grade(model_and_tok):
    _, tok = model_and_tok
    p = tasks.make_problem("story_math", random.Random(0), [3, 4])
    assert p["kind"] == "story_math" and p["steps"] in (3, 4) and p["digits"] == p["steps"]
    assert tasks.make_problem("story_math", random.Random(0), [8], held_out=True)["steps"] == 8
    right = f"The answer is {p['answer']}."
    assert tasks.check_answer(right, p) and not tasks.check_answer(f"{p['answer']}", p)
    assert tasks.grade(tok, _completion(tok, right), p)
    assert tasks.grade(tok, _completion(tok, right, "1 + 1 = 2"), p)
    assert not tasks.grade(tok, _completion(tok, f"The answer is {p['answer'] + 1}."), p)
    assert not tasks.grade(tok, _completion(tok, f"The answer is {p['answer'] + 1}.", "1 + 1 = 2"), p)
    assert not tasks.grade(tok, _completion(tok, right, "1 + 1 = 2", closed=False), p)  # a thought never closed


def test_eval_story_math(story_model):
    model, tok = story_model
    problems = [tasks.make_problem("story_math", random.Random(i), [1, 2, 3]) for i in range(6)]
    for level, think in [("medium", None), ("low", True), ("high", True), ("high", None), ("medium", False)]:
        records = tasks.eval_story_math(model, tok, problems, 6, level, think)
        assert len(records) == 6
        for r, p in zip(records, problems):
            assert set(r) == {"k", "digits", "bucket", "correct", "thinking", "reasoning_tokens", "style_ok", "finished"}
            assert r["k"] == p["steps"] and r["bucket"] == p["bucket"]
            assert r["thinking"] == (r["reasoning_tokens"] > 0) and (r["thinking"] or not r["style_ok"])
            if think:
                assert r["thinking"]
            if think is False:
                assert not r["thinking"] and r["reasoning_tokens"] == 0


def test_evaluate_story_math(story_model):
    model, tok = story_model
    cfg = tomllib.loads(CONFIG)
    cfg["data"]["reason"] = {"eval_seed": 1004}
    cfg["eval"]["story_math"] = {"n_per_step": 4, "steps": [1, 6], "max_new_tokens": 6}
    result = eval_run.evaluate_story_math(model, tok, cfg, n_per_step=2)
    sm = result["story_math"]
    modes = {f"{kind} {level}" for kind in ("adaptive", "forced") for level in eval_run.LEVELS} | {"off", "adaptive medium nocalc"}
    assert set(sm["accuracy"]) == set(sm["think_rate"]) == set(sm["tokens"]) == modes
    assert set(sm["accuracy"]["off"]) == {"1", "6"} and sm["think_rate"]["off"] == {"1": 0, "6": 0}
    assert all(v == 1 for v in sm["think_rate"]["forced high"].values())
    assert set(sm["style_ok"]) == set(sm["direct_acc"]) == set(sm["regret"]) == set(eval_run.LEVELS)
    assert set(sm["regret"]["medium"]) == {"1", "6"}
    assert result["sm_6st"] == sm["accuracy"]["adaptive medium"]["6"]
    assert result["sm_low"] == pytest.approx(sum(sm["accuracy"]["adaptive low"].values()) / 2)
    assert result["sm_medium_nocalc"] == pytest.approx(sum(sm["accuracy"]["adaptive medium nocalc"].values()) / 2)
    assert set(sm["heldout"]) == {"steps78", "objects", "ops", "distract"}
    assert set(sm["heldout"]["ops"]) == {"adaptive medium", "adaptive high", "off"}
    json.dumps(result)


def test_heldout_story_sets():
    half = lambda p: any(op == "half" for op, _ in p["ops"])
    assert {p["steps"] for p in eval_run._heldout_problems(1, "steps78", 0.1)} == {7, 8}
    assert not any(map(half, eval_run._heldout_problems(1, "objects", 0.5)))
    assert all(map(half, eval_run._heldout_problems(1, "ops", 0.5)))
    assert all(p["ignore"] for p in eval_run._heldout_problems(1, "distract", 0.2))
    assert eval_run._heldout_problems(1, "ops", 0.2) == eval_run._heldout_problems(1, "ops", 0.2)  # fixed sets


def test_table_story_math_columns():
    old = {"stage": "sft", "mode": "chat", "val_ppl": 2.0, "arithmetic": {"1": 1.0}, "train_digits": [1]}
    rate = {f"adaptive {level}": {"1": v, "2": v} for level, v in zip(eval_run.LEVELS, (0.1, 0.4, 0.7))}
    new = {**old, "sm_low": 0.5, "sm_medium": 0.6, "sm_high": 0.7, "sm_6st": 0.4,
           "story_math": {"think_rate": rate, "tokens": {m: {"1": 10 * (i + 1)} for i, m in enumerate(rate)},
                          "heldout": {"steps78": {"adaptive medium": {"accuracy": 0.3}},
                                      "objects": {"adaptive medium": {"accuracy": 0.55}}}}}
    cols, rows = eval_run._table_cells([new], None)
    row = dict(zip(cols, rows[0]))
    assert row["sm lo"] == "50%" and row["sm 6st"] == "40%" and row["think L/M/H"] == "10/40/70%"
    assert row["tok L/M/H"] == "10/20/30" and row["sm 7-8*"] == "30%" and row["sm obj*"] == "55%"
    assert "sm med" not in eval_run._table_cells([old], None)[0]
    cols, rows = eval_run._table_cells([old, new], None)  # an old eval.json next to a new one
    assert dict(zip(cols, rows[0]))["sm med"] == "-"
    assert "sm hi" in eval_run.table([new]) and "sm hi" not in eval_run.table([old])


def test_gate_story_math():
    cfg = {"data": {"digits": [1]}, "eval": {"n_per_digit": 100, "n_sampled": 100, "n_tool": 50}}
    old = {"val_bpc": 0.6, "arithmetic": {"1": 1.0}}
    new = {**old, "sm_low": 0.4, "sm_medium": 0.5, "sm_high": 0.6, "sm_6st": 0.2}
    assert {c.metric for c in gate.compare(new, old, cfg)} == {"bpc", "1d"}  # an old eval.json has no story problems
    assert {c.metric for c in gate.compare(old, new, cfg)} == {"bpc", "1d"}
    worse = gate.compare({**new, "sm_low": 0.3, "sm_6st": 0.19}, new, cfg)
    assert {c.metric for c in worse if c.regressed} == {"sm lo"}  # -10 points on 600 prompts; -1 point on 100 is noise
