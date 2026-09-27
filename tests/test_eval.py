import json
import random
import tomllib
from pathlib import Path

import pytest
import torch

from minilab.checkpoint import load_checkpoint
from minilab.data import arithmetic
from minilab.eval import run as eval_run
from minilab.eval import tasks
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
[rl]
digits = [2]
[eval]
n_per_digit = 3
n_sampled = 2
n_tool = 2
n_instr = 2
max_new_tokens = 16
ppl_batches = 1
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


@pytest.mark.parametrize("stage", ["pretrain", "sft"])
def test_evaluate_tiny_model(model_and_tok, stage):
    model, tok = model_and_tok
    result = eval_run.evaluate(model, tok, tomllib.loads(CONFIG), stage)
    assert set(result["arithmetic"]) == {"1", "2", "3"}
    assert result["val_ppl"] > 1 and 0 <= result["in_distribution"] <= 1
    if stage == "sft":
        for key in ("arithmetic_sampled", "tool_call", "tool_answer", "story_topic", "instr", "format"):
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
    for stage in ("pretrain", "rl"):
        save_stage(run, stage, model, tok, {"steps": 1, "tokens": 64, "wall_clock_s": 1.0}, cfg, "cpu")
        (run / stage / "eval.json").write_text(json.dumps(eval_run.evaluate(model, tok, cfg, stage)))

    out = release(run, "rl", "mini-test", tmp_path / "models")
    info = load_model_info(out)
    assert info.id == "mini-test" and info.context_length == 64
    assert info.pricing.input_per_1m == 10.0 and info.pricing.output_per_1m == 50.0
    card = (out / "MODEL_CARD.md").read_text()
    assert card.startswith("# mini-test") and "pretrain (base)" in card
    assert (out / "eval.json").exists()
    loaded, _, meta = load_checkpoint(out)
    assert meta["stage"] == "rl" and loaded.num_params() == model.num_params()
