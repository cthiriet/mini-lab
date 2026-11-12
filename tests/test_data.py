import json
import random

import pytest

from minilab.data import arithmetic, conversations, story_math
from minilab.data.conversations import (INSTRUCTIONS, Reason, StoryPool, first_sentence, instruction_conversation,
                                        is_refusal, mentions, midtrain_stream, sft_conversation, sft_dataset,
                                        midtrain_conversation, mixed_conversation, switch_conversation)
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


def test_switch_conversations_leave_the_math():
    rng, pool = random.Random(4), StoryPool(STORIES)
    for _ in range(100):
        conv = switch_conversation(rng, [1, 2, 3], pool, tools=rng.random() < 0.5)
        messages = conv["messages"]
        assert conv["train_on"] == "last" and [m["role"] for m in messages[-2:]] == ["user", "assistant"]
        assert messages[1].get("tool_calls") or messages[1]["content"].startswith("The answer is")  # math first
        last = messages[-1]
        assert not last.get("tool_calls") and "reasoning" not in last and "answer is" not in last["content"]
        assert not any("reasoning" in m for m in messages)  # the history as a client sends it back


def test_mixed_conversations_have_no_math():
    rng, pool = random.Random(5), StoryPool(STORIES)
    for _ in range(100):
        conv = mixed_conversation(rng, pool, tools=rng.random() < 0.5)
        messages = conv["messages"]
        assert conv["train_on"] == "last" and len(messages) >= 4 and [m["role"] for m in messages[-2:]] == ["user", "assistant"]
        assert not any(m.get("tool_calls") or "answer is" in m["content"] for m in messages)


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


# ---- story_math: policy, teacher bank, effort header (docs/story-math-spec.md) ----------------------

RC = {"steps": [1, 2, 3], "k_weights": [1, 1, 1], "efforts": {"low": 0.3, "medium": 0.4, "high": 0.3},
      "header_frac": 0.15, "tau": {"low": 0.80, "medium": 0.92, "high": 0.99}}
MIX_MID = {"arithmetic": 0.35, "story_math": 0.30, "story": 0.20, "greeting": 0.15}
MIX_SFT = {"plain": 0.3, "new_question": 0.2, "refusal": 0.1, "reason": 0.4}


@pytest.fixture
def made(monkeypatch):
    """Every story_math conversation made, in order, with the problem behind it."""
    out = []
    make, conv = story_math.make, conversations.story_math_conversation

    def spy_make(*args, **kw):
        out.append({"problem": make(*args, **kw)})
        return out[-1]["problem"]

    def spy_conv(rng, reason):
        out[-1]["conv"] = conv(rng, reason)
        return out[-1]["conv"]

    monkeypatch.setattr(story_math, "make", spy_make)
    monkeypatch.setattr(conversations, "story_math_conversation", spy_conv)
    return out


def reason_with(tmp_path, p=None, **rc) -> Reason:
    """A Reason with a probe table {"k,dbl": accuracy} (none: no policy file) and extra [data.reason] keys."""
    if p is not None:
        (tmp_path / "policy.json").write_text(json.dumps({"tau": {}, "p": p, "n": 300}))
    return Reason({**RC, "policy": str(tmp_path / "policy.json"), "traces": str(tmp_path / "traces.jsonl"), **rc})


def story_math_convs(reason: Reason, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [conversations.story_math_conversation(rng, reason) for _ in range(n)]


def test_story_math_follows_the_policy(tmp_path, made):
    # buckets alternate between certain (p = 1.0: direct at every level) and a coin flip (p = 0.5:
    # below every threshold, so never direct at any level)
    table = {f"{k},{d}": 1.0 if (k + d) % 2 else 0.5 for k in (2, 3) for d in (0, 1)} | {"1,add": 1.0, "1,sub": 0.5, "1,dbl": 1.0}
    tau = {"low": 0.80, "medium": 0.92, "high": 0.95}  # think() adds 0.03: with high = 0.99 nothing is direct at high
    story_math_convs(reason_with(tmp_path, table, tau=tau), 600)
    efforts = {True: set(), False: set()}
    for m in made:
        certain = table.get(story_math.bucket(m["problem"]), 0.5) == 1.0
        assert ("reasoning" not in m["conv"]["messages"][1]) == certain
        efforts[certain].add(m["conv"]["effort"])
    assert efforts[True] == efforts[False] == {None, "low", "high"}  # at every level


def test_story_math_without_a_policy_file_always_thinks(tmp_path):
    convs = story_math_convs(reason_with(tmp_path), 100)
    assert all(c["messages"][1].get("reasoning") and c["tools"] is None for c in convs)
    assert all(c["messages"][1]["content"].startswith("The answer is ") for c in convs)


def test_story_math_direct_mode(tmp_path, made):
    reason, pool, rng = reason_with(tmp_path, mode="direct"), StoryPool(STORIES), random.Random(1)
    convs = [midtrain_conversation(rng, MIX_MID, [1, 2, 3], pool, 0.3, reason) for _ in range(300)]
    convs += [sft_conversation(rng, MIX_SFT, [1, 2, 3], pool, 0.3, reason) for _ in range(300)]
    assert len(made) > 100
    assert all("reasoning" not in m["conv"]["messages"][1] for m in made)
    assert all(c.get("effort") is None for c in convs)  # no header on any kind


def test_midtrain_stream_share_and_headers(tmp_path, made):
    reason, pool = reason_with(tmp_path), StoryPool(STORIES)
    stream = midtrain_stream(0, {"mix": MIX_MID, "tool_frac": 0.3}, [1, 2, 3], pool, reason)
    convs = [next(stream) for _ in range(3000)]
    assert abs(len(made) / len(convs) - 0.30) < 0.03
    math_ids = {id(m["conv"]) for m in made}
    others = [c for c in convs if id(c) not in math_ids]
    assert abs(sum("effort" in c for c in others) / len(others) - 0.15) < 0.03
    assert {c["effort"] for c in others if "effort" in c} == {"low", "high"}
    assert {m["conv"]["effort"] for m in made} == {None, "low", "high"}


def test_sft_reason_kind_and_histories(tmp_path, made):
    reason, pool = reason_with(tmp_path), StoryPool(STORIES)
    convs = sft_dataset(0, 2000, {"mix": MIX_SFT, "tool_frac": 0.3}, [1, 2, 3], pool, reason)
    asked = {m["conv"]["messages"][0]["content"] for m in made}
    reasoned = [c for c in convs if c["messages"][-2]["content"] in asked]  # ends on a story problem
    assert abs(len(reasoned) / len(convs) - 0.4) < 0.05
    with_history = [c for c in reasoned if len(c["messages"]) > 2]
    assert 0.04 < len(with_history) / len(reasoned) < 0.16
    assert all(c["train_on"] == "last" and not any("reasoning" in m for m in c["messages"][:-1]) for c in with_history)
    # new_question can carry a story problem in its history, answered without the thoughts
    earlier = [c for c in convs if c not in reasoned and any(m["content"] in asked for m in c["messages"][:-2])]
    assert earlier and all(not any("reasoning" in m for m in c["messages"][:-1]) for c in earlier)


def test_story_math_teacher_bank(tmp_path):
    rng, rows = random.Random(5), []
    for i, level in enumerate(["medium", "high", "high", "medium"]):
        problem = story_math.make(rng, 2 + i % 2, "train")
        rows.append({"id": f"t-{i}", "level": level, "text": f"TRACE {i} {level}", "ok": i != 3,
                     "problem": {**problem, "id": f"t-{i}"}})
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    reason = reason_with(tmp_path, teacher=True, efforts={"medium": 0.5, "high": 0.5})
    text_of = {r["text"]: r["problem"]["messages"][0]["content"] for r in rows}
    seen = set()
    for question, reply in (c["messages"] for c in story_math_convs(reason, 100)):
        if reply["reasoning"] in text_of:
            assert text_of[reply["reasoning"]] == question["content"]  # the trace belongs to this very problem
            seen.add(reply["reasoning"])
        else:  # one event: the teacher wrote none, the program does
            assert question["content"].count(".") <= 3 and story_math.style_ok("medium", reply["reasoning"]) or \
                story_math.style_ok("high", reply["reasoning"])
    assert seen == {"TRACE 0 medium", "TRACE 1 high", "TRACE 2 high"}  # the rejected row is never drawn
    # teacher = false, or a level the bank lacks: the program writes the thoughts
    for rc in ({"teacher": False}, {"teacher": True, "efforts": {"low": 1.0}}):
        assert all("TRACE" not in c["messages"][1]["reasoning"] for c in story_math_convs(reason_with(tmp_path, **rc), 30))


def test_story_math_without_a_trace_file_is_plan_b(tmp_path):
    reason = reason_with(tmp_path, teacher=True)  # no traces.jsonl in tmp_path
    assert reason.bank == {}
    assert all(c["messages"][1]["reasoning"] for c in story_math_convs(reason, 30))


def test_story_math_is_deterministic(tmp_path):
    reason, pool = reason_with(tmp_path), StoryPool(STORIES)
    mid = {"mix": MIX_MID, "tool_frac": 0.3}
    s1, s2 = midtrain_stream(4, mid, [1, 2, 3], pool, reason), midtrain_stream(4, mid, [1, 2, 3], pool, reason)
    assert [next(s1) for _ in range(100)] == [next(s2) for _ in range(100)]
    sft = {"mix": MIX_SFT, "tool_frac": 0.3}
    assert sft_dataset(2, 100, sft, [1, 2, 3], pool, reason) == sft_dataset(2, 100, sft, [1, 2, 3], pool, reason)


def test_story_math_rendering_has_the_effort_line(tok, tmp_path):
    conv = story_math_convs(reason_with(tmp_path, efforts={"high": 1.0}), 1)[0]
    ids, _ = render_conversation(tok, conv["messages"], conv["tools"], conv["effort"])
    assert "effort: high" in tok.decode(ids) and "effort" not in tok.decode(render_conversation(tok, conv["messages"])[0])
    x, _ = chat_batch(tok, [conv], 1024)  # the loader renders it with the header too
    assert x[0][:len(ids) - 1].tolist() == ids[:-1]
