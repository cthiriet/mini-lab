"""Evaluation tasks. Every eval set is generated from a fixed seed, so all stages (and
all runs) are scored on exactly the same problems. The problem generators and the
grader are shared with RL: the RL reward *is* the eval's grade().

- arithmetic: exact match of the whole answer ("The answer is 405."), per number of digits.
  Chat models are asked in the chat format; the base model (which has never seen a chat token) is
  prompted with raw text "347 + 58 =". Held-out digit counts measure length
  generalization.
- arithmetic_sampled: the hardest in-distribution digit count at temperature 1, the
  API default. One unlucky token in a long scratchpad ruins the answer.
- tool use: with the calculator enabled, the first turn must be a calculator call
  whose expression evaluates to the right result; we run the tool, and the second
  turn must state the right answer.
- story topic: "Tell me a story about a dog." -> the story mentions a dog.
- instruction following (taught by SFT): system prompts obeyed (INSTRUCTIONS), follow-up
  questions answered, out-of-scope questions refused (held-out questions), identity,
  and no over-refusal of in-scope requests.
- format: fraction of chat turns that are properly ended with <|assistant_end|>
  (and contain no tool call when no tool is available, no malformed tool call).
- perplexity: on held-out TinyStories (stories never seen in training).
"""

from __future__ import annotations

import math
import random
import re

import torch

from minilab.data import arithmetic
from minilab.data.conversations import (GREETINGS, IDENTITY, INSTRUCTIONS, NAME, OUT_OF_SCOPE, OUT_OF_SCOPE_EVAL,
                                        STORY_REQUESTS, TOPIC_REQUESTS, TOPICS, assistant, fill, is_refusal, mentions,
                                        user)
from minilab.data.loader import story_batches
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import parse_completion, render_prompt
from minilab.train.trainer import evaluate_loss

EVAL_SEED = 1234
INSTRUCTION_KINDS = ["number_only", "no_calculator", "one_sentence", "sure", "followup", "long_followup",
                     "new_question", "refusal", "identity"]
STORY_EVAL_REQUESTS = ["Tell me a story about {t}.", "Can you tell me a story about {t}?", "Write a short story about {t}."]
CHAT_PROMPTS = [  # scored for format only, and kept as samples in eval.json
    ([user("Hi!")], None),
    ([user("Who are you?")], None),
    ([user("Tell me a story about a dog.")], None),
    ([user("What is the capital of France?")], None),
    ([{"role": "system", "content": INSTRUCTIONS["number_only"]}, user("What is 12 + 30?")], None),
    ([user("What is 12 + 30?"), {"role": "assistant", "content": "The answer is 42."}, user("And add 25 to that?")], None),
    ([{"role": "system", "content": INSTRUCTIONS["one_sentence"]}, user("Tell me a story about a cat.")], None),
    ([user("Hello")], ["calculator"]),
]


def generate(model: GPT, prompts: list[list[int]], max_new_tokens: int, stop_ids: set[int],
             temperature: float = 0.0, seed: int = 0, batch_size: int = 64) -> list[list[int]]:
    """Batched sampling. A prompt that doesn't fit the context gets an empty completion (a failure)."""
    device = model.wte.weight.device
    gen = torch.Generator(device=device).manual_seed(seed)
    fits = [i for i, p in enumerate(prompts) if len(p) < model.config.block_size]
    out: list[list[int]] = [[] for _ in prompts]
    for start in range(0, len(fits), batch_size):
        rows = fits[start:start + batch_size]
        completions = model.generate([prompts[i] for i in rows], max_new_tokens, temperature=temperature,
                                     stop_ids=stop_ids, generator=gen)
        for i, c in zip(rows, completions):
            out[i] = c
    return out


def arithmetic_problems(digits: int, n: int, tools: bool = False) -> list[dict]:
    rng = random.Random(f"{EVAL_SEED}-{digits}-{tools}")
    return [arithmetic.prompt(rng, digits, tools) for _ in range(n)]


def make_problem(kind: str, rng: random.Random, digits: list[int], held_out: bool = False) -> dict:
    """A prompt (messages, tools) plus what grade() needs to check the answer.

    "add" and "tool" are plain additions (for "tool", sometimes after the calculator
    already answered); the other kinds are the behaviors SFT teaches. held_out picks
    out-of-scope questions that training never uses (the eval's).
    """
    def system(name: str) -> list[dict]:
        return [{"role": "system", "content": INSTRUCTIONS[name]}]

    if kind in ("add", "tool"):
        after_tool = kind == "tool" and rng.random() < 0.5
        return arithmetic.prompt(rng, arithmetic.sample_digits(rng, digits), kind == "tool", after_tool)
    if kind in ("number_only", "no_calculator"):
        p = arithmetic.prompt(rng, arithmetic.sample_digits(rng, digits), tools=kind == "no_calculator")
        return {**p, "kind": kind, "messages": system(kind) + p["messages"]}
    if kind == "followup":
        messages, a = arithmetic.exchange(rng, [1, 2], tools=False)
        turn, total = arithmetic.followup(rng, a, [1, 2], tools=False)
        return {"kind": kind, "messages": arithmetic.without_reasoning(messages) + turn[:1], "tools": None,
                "a": a, "b": total - a, "answer": total}
    if kind == "long_followup":  # a follow-up on a 4-5 digit total, to copy from the history
        tools = rng.random() < 0.5
        messages, a = arithmetic.exchange(rng, [3, 4], tools)
        turn, total = arithmetic.followup(rng, a, [1, 2, 3], tools)
        return {"kind": kind, "messages": arithmetic.client_history(messages, rng.random() < 0.5) + turn[:1],
                "tools": ["calculator"] if tools else None, "a": a, "b": total - a, "answer": total}
    if kind == "new_question":  # a fresh addition after 1-3 earlier turns: its own operands, not the last total
        tools = rng.random() < 0.5
        history = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            first = rng.choice(["add", "add", "story", "greeting"])
            if first == "add":
                history += arithmetic.exchange(rng, digits, tools)[0]
            elif first == "story":
                history += [user(rng.choice(STORY_REQUESTS)),
                            assistant("Once upon a time, there was a little cat named Tom. Tom liked to play in the sun.")]
            else:
                users, replies = rng.choice(GREETINGS)
                history += [user(rng.choice(users)), assistant(rng.choice(replies))]
        history = arithmetic.client_history(history, rng.random() < 0.5)
        # every length equally often: short questions are where a history trips it up
        # (a 1-digit operand padded wrong, "3 + 69" -> 33+69)
        p = arithmetic.prompt(rng, rng.choice(digits), tools)
        return {**p, "kind": kind, "messages": history + p["messages"], "tools": ["calculator"] if tools else None}
    if kind == "refusal":
        question = fill(rng, rng.choice(OUT_OF_SCOPE_EVAL if held_out else OUT_OF_SCOPE))
        return {"kind": kind, "messages": [user(question)], "tools": ["calculator"] if rng.random() < 0.5 else None}
    if kind == "identity":
        return {"kind": kind, "messages": [user(rng.choice(rng.choice(IDENTITY)[0]))], "tools": None}
    if kind == "one_sentence":
        request = rng.choice(TOPIC_REQUESTS).format(t=rng.choice(list(TOPICS)))
        return {"kind": kind, "messages": system(kind) + [user(request)], "tools": None}
    if kind == "story":
        return {"kind": kind, "messages": [user(rng.choice(STORY_REQUESTS))], "tools": None}
    if kind == "greeting":
        users, replies = rng.choice(GREETINGS)
        return {"kind": kind, "messages": [user(rng.choice(users))], "tools": None, "replies": replies}
    if kind == "sure":  # "Sure!" and then a proper answer to the request inside
        inner = make_problem(rng.choice(["add", "story", "greeting"]), rng, digits)
        return {"kind": kind, "messages": system(kind) + inner["messages"], "tools": None, "inner": inner}
    raise ValueError(f"unknown problem kind: {kind}")


def turn_ok(tok: Tokenizer, completion: list[int], tools: bool) -> bool:
    """Format adherence of one assistant turn: it ends with <|assistant_end|>, calls no tool
    that isn't available, and doesn't speak for another role (<|tool_start|>, ...)."""
    c = parse_completion(tok, completion)
    if not c.finished:
        return False
    if c.tool_calls and (not tools or any(call["name"] != "calculator" for call in c.tool_calls)):
        return False
    own = {tok.special(t) for t in ("<|think_start|>", "<|think_end|>", "<|tool_call_start|>", "<|tool_call_end|>")}
    body = completion[:completion.index(tok.special("<|assistant_end|>"))]
    return all(t in own for t in body if tok.is_special(t))


def one_short_sentence(text: str, max_words: int = 25) -> bool:
    text = text.strip()
    return re.fullmatch(r"[^.!?\n]+[.!?]", text) is not None and len(text.split()) <= max_words


def grade(tok: Tokenizer, completion: list[int], problem: dict) -> bool:
    """Did the assistant turn (ended properly) do what was asked? The eval metric and the RL reward.

    It has to check *everything*, because RL optimizes exactly this and finds every gap:
    - "the last number of the answer is right" let RL garble the operands it restates
      ("4521 + 380" answered "451 + 0 = 4901") -- and the eval saw nothing;
    - with "answer only, no reasoning" requests, it let RL reason anyway, then drop
      <|think_start|> so the scratchpad passed as the answer;
    - "starts with Sure!" was satisfied by "Sure! Bye!" for every request;
    - "a correct calculator call, turn ended" let RL write the call, then *invent* the
      tool's result and answer in the same turn (stray special tokens are ignored by
      parse_completion, and the turn did end).
    So: the exact answer the training data would give, with only the special tokens that
    answer needs, and nothing else. (The chat answer is now "The answer is 4901.", which
    no longer restates the operands at all.)
    """
    c = parse_completion(tok, completion)
    if not c.finished:
        return False
    S = tok.special
    specials = [t for t in completion[:completion.index(S("<|assistant_end|>"))] if tok.is_special(t)]
    calls = problem["kind"] == "tool" or (problem["kind"] in ("new_question", "long_followup") and problem["tools"])
    if calls and problem["messages"][-1]["role"] != "tool":  # the call itself
        expression = arithmetic.call_expression(c.tool_calls[0]) if len(c.tool_calls) == 1 else None
        return specials == [S("<|tool_call_start|>"), S("<|tool_call_end|>")] and not c.content.strip() \
            and expression is not None and arithmetic.calculator(expression) == str(problem["answer"])
    return specials in ([], [S("<|think_start|>"), S("<|think_end|>")]) and check_answer(c.content.strip(), problem)


def check_answer(content: str, problem: dict) -> bool:
    """Is this visible answer exactly right for the problem?"""
    kind = problem["kind"]
    if kind in ("add", "tool", "no_calculator", "followup", "long_followup", "new_question"):  # "tool": after the result
        return content == arithmetic.answer_text(problem["a"], problem["b"])
    if kind == "number_only":
        return content == str(problem["answer"])
    if kind == "sure":
        return content.startswith("Sure! ") and check_answer(content.removeprefix("Sure! "), problem["inner"])
    if kind == "story":
        return len(content.split()) >= 30 and not is_refusal(content)
    if kind == "greeting":
        return content in problem["replies"]
    if kind == "one_sentence":
        return one_short_sentence(content)
    if kind == "refusal":
        return is_refusal(content)
    if kind == "identity":
        return NAME in content.lower() and not is_refusal(content)
    raise ValueError(f"unknown problem kind: {kind}")


def eval_arithmetic_chat(model: GPT, tok: Tokenizer, problems: list[dict], max_new_tokens: int,
                         temperature: float = 0.0) -> list[dict]:
    stop = {tok.special("<|assistant_end|>")}
    prompts = [render_prompt(tok, p["messages"], p["tools"]) for p in problems]
    outs = generate(model, prompts, max_new_tokens, stop, temperature, seed=EVAL_SEED)
    records = []
    for p, out in zip(problems, outs):
        c = parse_completion(tok, out)
        records.append({"digits": p["digits"], "correct": grade(tok, out, p), "format_ok": turn_ok(tok, out, False),
                        "refused": is_refusal(c.content)})
    return records


def eval_arithmetic_completion(model: GPT, tok: Tokenizer, problems: list[dict]) -> list[dict]:
    """Base-model version: continue the raw text "a + b =" and read the number."""
    prompts = [[tok.bos_id, *tok.encode(f"{p['a']} + {p['b']} =")] for p in problems]
    outs = generate(model, prompts, max(p["digits"] for p in problems) + 3, {tok.bos_id})
    records = []
    for p, out in zip(problems, outs):
        m = re.match(r"\s*(\d+)", tok.decode(t for t in out if not tok.is_special(t)))
        records.append({"digits": p["digits"], "correct": bool(m) and int(m[1]) == p["answer"]})
    return records


def eval_tool_use(model: GPT, tok: Tokenizer, problems: list[dict], max_new_tokens: int) -> list[dict]:
    """Two turns: the calculator call, then (after running the tool) the final answer."""
    stop = {tok.special("<|assistant_end|>")}
    tools = ["calculator"]
    outs = generate(model, [render_prompt(tok, p["messages"], tools) for p in problems], max_new_tokens, stop)
    records, followups = [], []
    for p, out in zip(problems, outs):
        c = parse_completion(tok, out)
        expression = arithmetic.call_expression(c.tool_calls[0]) if c.tool_calls else None
        records.append({"digits": p["digits"], "call_ok": grade(tok, out, p), "correct": False,
                        "format_ok": turn_ok(tok, out, True)})
        if expression is not None:
            messages = p["messages"] + [
                {"role": "assistant", "content": c.content,
                 "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": c.tool_calls[0]["arguments"]}}]},
                {"role": "tool", "content": arithmetic.calculator(expression)},
            ]
            followups.append((len(records) - 1, render_prompt(tok, messages, tools)))
    outs = generate(model, [ids for _, ids in followups], max_new_tokens, stop)
    for (i, _), out in zip(followups, outs):
        c = parse_completion(tok, out)
        p = problems[i]
        records[i]["correct"] = records[i]["call_ok"] and c.finished and \
            c.content.strip() == arithmetic.answer_text(p["a"], p["b"])
        records[i]["format_ok"] = records[i]["format_ok"] and turn_ok(tok, out, True)
    return records


def eval_instructions(model: GPT, tok: Tokenizer, n: int, max_new_tokens: int) -> list[dict]:
    """n problems of every INSTRUCTION_KINDS kind, graded by grade()."""
    stop = {tok.special("<|assistant_end|>")}
    # one Random per kind, so adding a kind never changes the others' problems
    rngs = {kind: random.Random(f"{EVAL_SEED}-{kind}") for kind in INSTRUCTION_KINDS}
    problems = [make_problem(kind, rngs[kind], [1, 2, 3], held_out=True) for kind in INSTRUCTION_KINDS for _ in range(n)]
    outs = generate(model, [render_prompt(tok, p["messages"], p["tools"]) for p in problems], max_new_tokens, stop)
    records = []
    for p, out in zip(problems, outs):
        c = parse_completion(tok, out)
        records.append({"kind": p["kind"], "ok": grade(tok, out, p), "refused": is_refusal(c.content),
                        "format_ok": turn_ok(tok, out, bool(p["tools"]))})
    return records


def eval_chat_prompts(model: GPT, tok: Tokenizer) -> list[dict]:
    stop = {tok.special("<|assistant_end|>")}
    prompts = [render_prompt(tok, m, t) for m, t in CHAT_PROMPTS]
    outs = generate(model, prompts, model.config.block_size, stop)
    records = []
    for (messages, tools), out in zip(CHAT_PROMPTS, outs):
        c = parse_completion(tok, out)
        prompt = " / ".join(f"[{m['role']}] {m['content']}" if m["role"] != "user" else m["content"] for m in messages)
        records.append({"prompt": prompt, "tools": tools, "response": c.content, "reasoning": c.reasoning,
                        "tool_calls": c.tool_calls, "format_ok": turn_ok(tok, out, bool(tools))})
    return records


def eval_story_topics(model: GPT, tok: Tokenizer) -> list[dict]:
    stop = {tok.special("<|assistant_end|>")}
    requests = [(template.format(t=topic), word) for topic, word in TOPICS.items() for template in STORY_EVAL_REQUESTS]
    prompts = [render_prompt(tok, [user(request)]) for request, _ in requests]
    outs = generate(model, prompts, model.config.block_size, stop)
    records = []
    for (request, word), out in zip(requests, outs):
        c = parse_completion(tok, out)
        records.append({"prompt": request, "on_topic": mentions(c.content, word), "format_ok": turn_ok(tok, out, False),
                        "refused": is_refusal(c.content)})
    return records


def perplexity(model: GPT, tok: Tokenizer, stories: list[str], n_batches: int = 8) -> tuple[float, float]:
    """(loss, perplexity) on held-out stories."""
    loss = evaluate_loss(model, story_batches(tok, stories, 16, model.config.block_size, n_batches),
                         model.wte.weight.device.type)
    return loss, math.exp(loss)
