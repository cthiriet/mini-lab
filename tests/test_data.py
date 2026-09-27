import json
import random

import pytest

from minilab.data import arithmetic
from minilab.data.conversations import (INSTRUCTIONS, StoryPool, first_sentence, instruction_conversation,
                                        is_refusal, mentions, midtrain_stream, sft_conversation, sft_dataset)
from minilab.data.loader import chat_batch, epochs, packed_batches, pretrain_documents
from minilab.data.tinystories import clean
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_conversation

STORIES = [
    "Once upon a time, there was a dog named Max. Max liked to run in the park with his ball.",
    "Lily had a red ball. She played with her cat in the garden. The cat was happy.",
    "Tom saw a big tree. He climbed the tree and saw a bird. The bird sang a song.",
    "Anna baked a cake with her mom. They shared the cake with the whole family.",
]
MIDTRAIN = {"mix": {"arithmetic": 0.6, "story": 0.2, "greeting": 0.2}, "tool_frac": 0.3}
SFT = {"mix": {"plain": 0.3, "instruction": 0.3, "followup": 0.2, "refusal": 0.1, "identity": 0.1}, "tool_frac": 0.3}


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    rng = random.Random(0)
    return Tokenizer.train(STORIES * 5 + [arithmetic.pretrain_document(rng, [1, 2, 3]) for _ in range(50)], 400)


def test_scratchpad_example():
    assert arithmetic.scratchpad(347, 58) == "347+058: 7+8=15, 5\n34+05: 4+5+1=10, 05\n3+0: 3+0+1=4, 405"
    assert arithmetic.scratchpad(999, 1).splitlines()[-1] == "9+0: 9+0+1=10, 1000"


def test_scratchpad_is_correct():
    rng = random.Random(0)
    for _ in range(2000):
        a, b = arithmetic.sample_problem(rng, rng.randint(1, 6))
        lines = arithmetic.scratchpad(a, b).splitlines()
        assert len(lines) == max(len(str(a)), len(str(b)))
        assert int(lines[-1].rsplit(", ", 1)[1]) == a + b
        for line in lines:
            operands, work = line.split(": ")
            expr, total = work.split(", ")[0].split("=")
            assert sum(map(int, expr.split("+"))) == int(total)
            x, y = operands.split("+")
            assert len(x) == len(y)  # zero-padded to the same length


def test_sample_problem_digits():
    rng = random.Random(0)
    for n in range(1, 6):
        for _ in range(100):
            a, b = arithmetic.sample_problem(rng, n)
            assert max(len(str(a)), len(str(b))) == n


def test_call_expression():
    assert arithmetic.call_expression({"name": "calculator", "arguments": '{"expression": "1 + 2"}'}) == "1 + 2"
    assert arithmetic.call_expression({"name": "calculator", "arguments": '"1 + 2"'}) is None
    assert arithmetic.call_expression({"name": "invalid", "arguments": "{}"}) is None


def test_calculator():
    assert arithmetic.calculator("347 + 58") == "405"
    assert arithmetic.calculator("2 * (3 + 4) - 1") == "13"
    assert arithmetic.calculator("__import__('os')") == "error"
    assert arithmetic.calculator("1 / 0") == "error"


@pytest.mark.parametrize("tools", [False, True])
def test_exchange_answers_are_correct(tools):
    rng = random.Random(1)
    for _ in range(300):
        messages, total = arithmetic.exchange(rng, [1, 2, 3], tools)
        assert messages[-1]["content"] == f"The answer is {total}."
        if tools:
            call = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
            assert arithmetic.calculator(call["expression"]) == str(total)
            assert messages[2] == {"role": "tool", "tool_call_id": "call_0", "content": str(total)}
        else:  # the scratchpad's last line holds the sum the answer copies
            assert messages[-1]["reasoning"].endswith(f", {total}")
    number_only = arithmetic.exchange(rng, [2], False, number_only=True)[0][-1]
    assert number_only["content"].isdigit() and number_only["reasoning"]


def test_followup_refers_to_previous_result():
    rng = random.Random(4)
    messages, total = arithmetic.exchange(rng, [1, 2], tools=False)
    turn, new_total = arithmetic.followup(rng, total, [1, 2], tools=False)
    c = new_total - total
    assert str(c) in turn[0]["content"] and turn[-1]["content"] == f"The answer is {new_total}."
    assert turn[-1]["reasoning"] == arithmetic.scratchpad(total, c)
    assert "reasoning" not in arithmetic.without_reasoning(messages)[-1]


def test_generators_are_deterministic():
    pool = StoryPool(STORIES)
    s1, s2 = midtrain_stream(7, MIDTRAIN, [1, 2, 3], pool), midtrain_stream(7, MIDTRAIN, [1, 2, 3], pool)
    assert [next(s1) for _ in range(20)] == [next(s2) for _ in range(20)]
    assert sft_dataset(3, 30, SFT, [1, 2, 3], pool) == sft_dataset(3, 30, SFT, [1, 2, 3], pool)
    assert arithmetic.pretrain_document(random.Random(3), [2]) == arithmetic.pretrain_document(random.Random(3), [2])
    p = arithmetic.prompt(random.Random(5), 3, tools=True)
    assert p["answer"] == p["a"] + p["b"] and p["tools"] == ["calculator"] and p["kind"] == "tool"


def test_midtrain_is_single_turn_without_system_prompts():
    stream = midtrain_stream(0, MIDTRAIN, [1, 2, 3], StoryPool(STORIES))
    for conv in (next(stream) for _ in range(200)):
        roles = [m["role"] for m in conv["messages"]]
        assert roles[0] == "user" and roles.count("user") == 1 and "system" not in roles


def test_sft_conversations(tok):
    rng = random.Random(2)
    pool = StoryPool(STORIES)
    for _ in range(300):
        conv = sft_conversation(rng, SFT["mix"], [1, 2, 3], pool, tool_frac=0.5)
        ids, mask = render_conversation(tok, conv["messages"], conv["tools"])
        assert ids[0] == tok.bos_id and len(ids) == len(mask) and sum(mask) > 0
        has_call = any(m.get("tool_calls") for m in conv["messages"])
        assert not has_call or conv["tools"] == ["calculator"]


def test_instruction_conversations_follow_their_instruction():
    rng, pool = random.Random(3), StoryPool(STORIES)
    for _ in range(20):
        conv = instruction_conversation(rng, "number_only", [1, 2, 3], pool)
        assert conv["messages"][0]["content"] == INSTRUCTIONS["number_only"] and conv["messages"][-1]["content"].isdigit()
        conv = instruction_conversation(rng, "no_calculator", [1, 2, 3], pool)
        assert conv["tools"] == ["calculator"] and not any(m.get("tool_calls") for m in conv["messages"])
        assert instruction_conversation(rng, "sure", [1, 2, 3], pool)["messages"][-1]["content"].startswith("Sure! ")
        sentence = instruction_conversation(rng, "one_sentence", [1, 2, 3], pool)["messages"][-1]["content"]
        assert sentence.count(".") == 1 and len(sentence.split()) <= 25
    assert first_sentence('Tom said "hi". Then he left.') is None
    assert is_refusal("Sorry, I can only add numbers.") and not is_refusal("Once upon a time, Tom said sorry.")


def test_story_topics():
    pool = StoryPool(STORIES)
    assert all(" dog" in s.lower() for s in pool.by_topic["dog"]) and pool.by_topic["dog"]
    assert mentions("Two cats played.", "cat") and not mentions("Catch the ball!", "cat")


def test_clean():
    assert clean("“Hi,” she said.\n\n  It’s  fun. ") == "\"Hi,\" she said.\nIt's fun."
    assert clean("Un café") is None


def test_packed_batches(tok):
    docs = pretrain_documents(tok, STORIES, [1, 2], arith_frac=0.5, seed=0)
    x, y = next(packed_batches(docs, batch_size=4, block_size=32))
    assert x.shape == y.shape == (4, 32)
    assert (x[:, 1:] == y[:, :-1]).all()
    assert (x == tok.bos_id).any()


def test_chat_batch_masks(tok):
    convs = [{"messages": arithmetic.exchange(random.Random(i), [3], i % 2 == 0)[0],
              "tools": ["calculator"] if i % 2 == 0 else None} for i in range(3)]
    convs.append({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]})
    x, y = chat_batch(tok, convs, block_size=256)
    for row, conv in enumerate(convs):
        ids, mask = render_conversation(tok, conv["messages"], conv.get("tools"))
        n = len(ids) - 1
        assert x[row, :n].tolist() == ids[:-1]
        assert y[row, :n].tolist() == [t if m else -1 for t, m in zip(ids[1:], mask[1:])]
        assert (y[row, n:] == -1).all()  # padding is ignored
    # the last target of every row is the assistant's <|assistant_end|>
    end = tok.special("<|assistant_end|>")
    assert all(y[row][y[row] != -1][-1] == end for row in range(len(convs)))


def test_chat_batch_trains_on_last_turn_only(tok):
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]
    _, y_all = chat_batch(tok, [{"messages": messages}], block_size=256)
    _, y_last = chat_batch(tok, [{"messages": messages, "train_on": "last"}], block_size=256)
    trained = y_last[0][y_last[0] != -1].tolist()
    assert tok.decode(trained) == "Goodbye!<|assistant_end|>"
    assert (y_all[0] != -1).sum() > len(trained)

    # the last answer includes its calculator call, not just the final answer after the result
    with_tool = messages[:2] + arithmetic.exchange(random.Random(0), [2], tools=True)[0]
    _, y_tool = chat_batch(tok, [{"messages": with_tool, "tools": ["calculator"], "train_on": "last"}], block_size=256)
    trained = y_tool[0][y_tool[0] != -1].tolist()
    assert trained.count(tok.special("<|tool_call_start|>")) == 1 and trained.count(tok.special("<|assistant_end|>")) == 2


def test_chat_batch_truncates(tok):
    conv = {"messages": [{"role": "user", "content": "Tell me a story."}, {"role": "assistant", "content": " ".join(STORIES * 5)}]}
    x, y = chat_batch(tok, [conv], block_size=64)
    assert x.shape == (1, 64)


def test_epochs_reshuffle_a_fixed_set():
    it = epochs(list(range(5)), seed=0)
    first, second = [next(it) for _ in range(5)], [next(it) for _ in range(5)]
    assert sorted(first) == sorted(second) == list(range(5)) and first != second
