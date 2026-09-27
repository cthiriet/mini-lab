import random
from pathlib import Path

import pytest
import torch

from minilab.checkpoint import load_checkpoint
from minilab.data import arithmetic
from minilab.data.loader import packed_batches, pretrain_documents
from minilab.model.gpt import GPT, GPTConfig
from minilab.tokenizer.bpe import Tokenizer
from minilab.train.distill import distill_step
from minilab.train.rl import reward, rl_step, sample_problem
from minilab.train.trainer import Logger, evaluate_loss, lr_at, make_optimizer, save_stage, train_loop


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    rng = random.Random(0)
    return Tokenizer.train([arithmetic.pretrain_document(rng, [1, 2]) for _ in range(200)], 320)


def tiny_model(tok: Tokenizer) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=32))


def test_lr_schedule():
    lrs = [lr_at(s, 100, 1.0, warmup=10, min_lr_frac=0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))  # monotone decay after warmup
    assert lrs[-1] == pytest.approx(0.1, abs=0.01)


def test_optimizer_decays_only_matrices(tok):
    opt = make_optimizer(tiny_model(tok), 1e-3, 0.1)
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0


def test_muon_takes_the_hidden_matrices(tok):
    model = tiny_model(tok)
    muon, adamw = make_optimizer(model, 1e-3, 0.1, "muon").optimizers
    hidden = {id(p) for g in muon.param_groups for p in g["params"]}
    assert id(model.blocks[0].attn.qkv.weight) in hidden and id(model.blocks[0].mlp.fc.weight) in hidden
    assert id(model.wte.weight) not in hidden  # the embedding (tied to the output head) stays on AdamW
    assert {id(p) for p in model.parameters()} == hidden | {id(p) for g in adamw.param_groups for p in g["params"]}


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_training_reduces_loss(tok, tmp_path: Path, optimizer):
    model = tiny_model(tok)
    docs = pretrain_documents(tok, ["Tom has a ball."], [1, 2], arith_frac=0.9, seed=0)
    batches = packed_batches(docs, batch_size=8, block_size=64)
    val = [next(packed_batches(pretrain_documents(tok, ["A cat."], [1, 2], 0.9, seed=1), 8, 64))]
    before = evaluate_loss(model, val, "cpu")
    sc = {"steps": 40, "lr": 1e-2, "warmup": 5, "log_every": 20, "eval_every": 40}
    stats = train_loop(model, batches, sc, Logger(tmp_path / "log.jsonl"), "cpu",
                       lambda: {"val_loss": evaluate_loss(model, val, "cpu")}, optimizer)
    assert stats["val_loss"] < before - 1.0
    assert stats["tokens"] == 40 * 8 * 64
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3  # 2 train logs + 1 eval

    path = save_stage(tmp_path, "pretrain", model, tok, stats, {"pretrain": sc}, "cpu")
    model2, _, meta = load_checkpoint(path)
    assert meta["stage"] == "pretrain" and meta["tokens_total"] == stats["tokens"] and meta["params"] == model.num_params()


def test_reward(tok):
    S = tok.special
    end = S("<|assistant_end|>")
    add = {"kind": "add", "messages": [{"role": "user", "content": "3 + 4"}], "a": 3, "b": 4, "answer": 7, "tools": None}
    assert reward(tok, tok.encode("The answer is 7.") + [end], add) == 1.0
    assert reward(tok, tok.encode("The answer is 8.") + [end], add) == 0.0
    assert reward(tok, tok.encode("The answer is 7."), add) == 0.0  # the turn must be ended
    thinking = [S("<|think_start|>"), *tok.encode("3+4: 3+4=7, 7"), S("<|think_end|>"), *tok.encode("The answer is 7."), end]
    assert reward(tok, thinking, add) == 1.0

    number_only = {**add, "kind": "number_only"}
    assert reward(tok, tok.encode("7") + [end], number_only) == 1.0
    assert reward(tok, tok.encode("The answer is 7.") + [end], number_only) == 0.0
    assert reward(tok, [S("<|think_start|>"), *tok.encode("3+4=7"), S("<|think_end|>"), *tok.encode("7"), end],
                  number_only) == 1.0  # reasoning is fine: the instruction is about the visible answer
    assert reward(tok, [*tok.encode("3+4=7"), S("<|think_end|>"), *tok.encode("7"), end],
                  number_only) == 0.0  # but not a scratchpad passed off as the answer

    call = [S("<|tool_call_start|>"), *tok.encode('{"name": "calculator", "arguments": {"expression": "3 + 4"}}'),
            S("<|tool_call_end|>"), end]
    tool = {**add, "kind": "tool", "tools": ["calculator"]}
    assert reward(tok, call, tool) == 1.0
    assert reward(tok, call[:-1] + call, tool) == 0.0  # the same call twice
    invented = call[:-1] + [S("<|tool_start|>"), *tok.encode("7"), S("<|tool_end|>"), S("<|assistant_start|>"),
                            *tok.encode("The answer is 7."), end]
    assert reward(tok, invented, tool) == 0.0  # the hack RL found: inventing the tool's result
    assert reward(tok, tok.encode("The answer is 7.") + [end], tool) == 0.0
    assert reward(tok, call, {**tool, "kind": "no_calculator"}) == 0.0
    assert reward(tok, thinking, {**tool, "kind": "no_calculator"}) == 1.0
    after_tool = arithmetic.prompt(random.Random(0), 1, after_tool=True)
    assert after_tool["messages"][-1]["role"] == "tool" and after_tool["tools"] == ["calculator"]
    assert reward(tok, tok.encode(arithmetic.answer_text(after_tool["a"], after_tool["b"])) + [end], after_tool) == 1.0

    assert reward(tok, tok.encode("3 + 4 = 7") + [end], add) == 0.0  # anything but the exact answer
    assert reward(tok, tok.encode("The answer is 7. Easy!") + [end], add) == 0.0
    refusal = {"kind": "refusal", "messages": [], "tools": None}
    assert reward(tok, tok.encode("Sorry, I can only add numbers.") + [end], refusal) == 1.0
    assert reward(tok, tok.encode("Paris.") + [end], refusal) == 0.0
    sure = {"kind": "sure", "messages": [], "inner": add}
    assert reward(tok, tok.encode("Sure! The answer is 7.") + [end], sure) == 1.0
    assert reward(tok, tok.encode("Sure! Bye!") + [end], sure) == 0.0  # the hack RL found
    assert reward(tok, tok.encode("The answer is 7.") + [end], sure) == 0.0
    new = {"kind": "new_question", "messages": [{"role": "user", "content": "766 + 989"}], "a": 766, "b": 989, "answer": 1755, "tools": None}
    assert reward(tok, tok.encode("The answer is 1755.") + [end], new) == 1.0
    assert reward(tok, tok.encode("The answer is 1394.") + [end], new) == 0.0  # 405 + 989: the last total reused
    assert reward(tok, call, {**new, "a": 3, "b": 4, "answer": 7, "tools": ["calculator"]}) == 1.0
    long = {**new, "kind": "long_followup", "a": 8829, "b": 31, "answer": 8860}
    assert reward(tok, tok.encode("The answer is 8860.") + [end], long) == 1.0
    assert reward(tok, tok.encode("The answer is 913.") + [end], long) == 0.0  # 882 + 31: the total copied short
    assert reward(tok, tok.encode("Once upon a time, there was a cat.") + [end], {"kind": "one_sentence", "messages": []}) == 1.0
    assert reward(tok, tok.encode("Once upon a time. The end.") + [end], {"kind": "one_sentence", "messages": []}) == 0.0


def test_rl_step_runs(tok):
    model = tiny_model(tok)
    opt = make_optimizer(model, 1e-3, 0.0)
    rng = random.Random(0)
    sc = {"digits": [1, 2], "chat_digits": [1], "mix": {"add": 1, "tool": 1, "number_only": 1, "followup": 1, "refusal": 1}}
    problems = [sample_problem(rng, sc) for _ in range(3)]
    before = [p.clone() for p in model.parameters()]
    stats = rl_step(model, tok, opt, problems, group_size=4, max_new_tokens=8, temperature=1.0,
                    generator=torch.Generator().manual_seed(0))
    assert 0.0 <= stats["reward"] <= 1.0 and stats["completion_len"] <= 8
    if stats["informative"] == 0:  # a random model never answers right: nothing to learn
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))



def test_distill_step(tok):
    rng = random.Random(0)
    sc = {"digits": [1, 2], "chat_digits": [1], "mix": {"add": 1, "tool": 1, "refusal": 1}}
    problems = [(sample_problem(rng, sc), i % 2) for i in range(4)]
    student = tiny_model(tok)
    same = tiny_model(tok)  # the same weights: nothing to learn
    stats = distill_step(student, [same, same], tok, make_optimizer(student, 1e-3, 0.0), problems, 8, 1.0,
                         torch.Generator().manual_seed(0))
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6) and 0.0 <= stats["reward"] <= 1.0

    torch.manual_seed(1)
    other = GPT(student.config)
    opt = make_optimizer(student, 1e-2, 0.0)
    kls = [distill_step(student, [same, other], tok, opt, problems, 8, 1.0, torch.Generator().manual_seed(0))["kl"]
           for _ in range(10)]
    assert kls[-1] < kls[0]  # the student moves toward its teachers
