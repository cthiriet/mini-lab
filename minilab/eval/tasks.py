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
- real chat: 3-5 requests in a row that mix additions, follow-ups, stories, small talk,
  identity and refusals, played like the chat app plays them (its history, its calculator
  loop). A conversation passes if every answer is right.
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
                                        REFUSALS, STORY_REQUESTS, TOPIC_REQUESTS, TOPICS, assistant, fill, is_refusal,
                                        mentions, user)
from minilab.data.loader import story_batches
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import parse_completion, recent_turns, render_prompt
from minilab.train.trainer import evaluate_loss

EVAL_SEED = 1234
INSTRUCTION_KINDS = ["number_only", "no_calculator", "one_sentence", "sure", "followup", "long_followup",
                     "new_question", "switch", "mixed", "refusal", "identity"]
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
        messages, a = arithmetic.exchange(rng, [3, 4, 5], tools)
        while a >= 100000:  # a 6-digit total would be a held-out operand
            messages, a = arithmetic.exchange(rng, [3, 4, 5], tools)
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
    if kind == "word":  # long numbers in a sentence, far from the "+" the model copies them from
        tools = rng.random() < 0.5
        a, b = arithmetic.sample_problem(rng, rng.choice([n for n in digits if n >= 3] or digits))
        text = arithmetic.WORD_PROBLEM.format(name=rng.choice(arithmetic.NAMES), things=rng.choice(arithmetic.THINGS),
                                              a=a, b=b)
        return {"kind": "tool" if tools else "add", "messages": [user(text)], "tools": ["calculator"] if tools else None,
                "a": a, "b": b, "answer": a + b, "digits": max(len(str(a)), len(str(b)))}
    if kind == "mixed":  # a request after stories, small talk or refusals: not an echo of the last answer
        inner = make_problem(rng.choice(["story", "greeting", "identity", "refusal"]), rng, digits, held_out)
        history = []
        for _ in range(1 if inner["kind"] == "story" else rng.choice([1, 2, 3])):
            history += small_talk(rng, held_out)
        return {"kind": kind, "messages": history + inner["messages"],
                "tools": ["calculator"] if rng.random() < 0.5 else None, "inner": inner}
    if kind == "switch":  # something else after 1-3 additions, to answer as if it came first
        tools = rng.random() < 0.5
        inner = make_problem(rng.choice(["story", "greeting", "identity", "refusal"]), rng, digits, held_out)
        if inner["kind"] == "story":  # a story the context still has room for, on a topic half the time
            template = rng.choice(STORY_EVAL_REQUESTS if rng.random() < 0.5 else STORY_REQUESTS)
            inner["messages"] = [user(template.format(t=rng.choice(list(TOPICS))))]
        request = inner["messages"][-1]["content"]
        if rng.random() < 0.25:  # typed casually: "tell me a story about a dog"
            inner["messages"] = [user(request[0].lower() + request[1:].rstrip(".!?"))]
        history = []
        for _ in range(1 if inner["kind"] == "story" else rng.choice([1, 2, 3])):
            history += arithmetic.exchange(rng, digits, tools)[0]
        return {"kind": kind, "messages": arithmetic.client_history(history, rng.random() < 0.5) + inner["messages"],
                "tools": ["calculator"] if tools else None, "inner": inner}
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


def small_talk(rng: random.Random, held_out: bool = False) -> list[dict]:
    """One exchange that isn't math, answered as training answers it: a short story, a
    greeting, who the model is, or a refusal (of a held-out question with held_out)."""
    kind = rng.choice(["story", "greeting", "identity", "refusal"])
    if kind == "story":
        return [user(rng.choice(STORY_REQUESTS)),
                assistant("Once upon a time, there was a little cat named Tom. Tom liked to play in the sun.")]
    if kind == "refusal":
        return [user(fill(rng, rng.choice(OUT_OF_SCOPE_EVAL if held_out else OUT_OF_SCOPE))), assistant(rng.choice(REFUSALS))]
    users, replies = rng.choice(GREETINGS if kind == "greeting" else IDENTITY)
    return [user(rng.choice(users)), assistant(rng.choice(replies))]


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
    if kind in ("switch", "mixed"):
        return check_answer(content, problem["inner"])
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
        records.append({"kind": p["kind"], "inner": p.get("inner", {}).get("kind"), "ok": grade(tok, out, p),
                        "refused": is_refusal(c.content), "format_ok": turn_ok(tok, out, bool(p["tools"]))})
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


def chat_script(rng: random.Random, digits: list[int]) -> dict:
    """A conversation as people have it in the chat app: 3 to 5 requests that mix additions,
    follow-ups right after one, stories, small talk, who the model is and out-of-scope
    questions (held out), with the calculator on for the whole chat or off."""
    turns = []
    for _ in range(rng.randint(3, 5)):
        after_math = bool(turns) and turns[-1]["kind"] == "add"
        kind = rng.choice(["add", "add", "story", "greeting", "identity", "refusal"] + ["followup"] * 2 * after_math)
        if kind == "add":
            a, b = arithmetic.sample_problem(rng, rng.choice(digits))
            turns.append({"kind": "add", "request": arithmetic.question(rng, a, b), "a": a, "b": b, "answer": a + b})
        elif kind == "followup":
            a, c = turns[-1]["answer"], arithmetic.sample_number(rng, rng.choice([1, 2]))
            turns.append({"kind": "add", "request": rng.choice(arithmetic.FOLLOWUPS).format(c=c),
                          "a": a, "b": c, "answer": a + c})
        else:
            p = make_problem(kind, rng, digits, held_out=True)
            if kind == "story" and rng.random() < 0.5:
                p["messages"] = [user(rng.choice(STORY_EVAL_REQUESTS).format(t=rng.choice(list(TOPICS))))]
            turns.append({**p, "request": p["messages"][-1]["content"]})
    return {"tools": ["calculator"] if rng.random() < 0.5 else None, "turns": turns}


def _turn_problem(turn: dict, messages: list[dict], tools: list[str] | None) -> dict:
    """What grade() needs for one chat turn: with the calculator on, an addition is a call
    first, then the answer after the tool's result."""
    kind = ("tool" if tools else "add") if turn["kind"] == "add" else turn["kind"]
    return {**turn, "kind": kind, "messages": messages, "tools": tools}


def eval_chat(model: GPT, tok: Tokenizer, scripts: list[dict]) -> list[dict]:
    """Play each script turn by turn, as the chat app does: the history it sends back (each
    earlier answer, without the scratchpad or the calculator round trip), trimmed by the
    same recent_turns; a calculator call is run and its result sent back for the answer.
    Greedy, and a conversation stops at its first wrong answer."""
    stop = {tok.special("<|assistant_end|>")}
    n_ctx = model.config.block_size
    history: list[list[dict]] = [[] for _ in scripts]
    failed: list[str | None] = [None] * len(scripts)
    for t in range(max(len(s["turns"]) for s in scripts)):
        live = [i for i, s in enumerate(scripts) if failed[i] is None and t < len(s["turns"])]
        prompts = {i: recent_turns(history[i] + [user(scripts[i]["turns"][t]["request"])], n_ctx) for i in live}
        rounds = [(i, prompts[i]) for i in live]
        while rounds:  # the answer, or a calculator call and then the answer
            outs = generate(model, [render_prompt(tok, m, scripts[i]["tools"]) for i, m in rounds], n_ctx, stop)
            next_rounds = []
            for (i, messages), out in zip(rounds, outs):
                turn, tools = scripts[i]["turns"][t], scripts[i]["tools"]
                problem = _turn_problem(turn, messages, tools)
                c = parse_completion(tok, out)
                if not grade(tok, out, problem):
                    failed[i] = turn["kind"] + (" (calculator)" if tools else "")
                elif problem["kind"] == "tool" and messages[-1]["role"] != "tool":
                    call = {"type": "function", "function": {"name": "calculator", "arguments": c.tool_calls[0]["arguments"]}}
                    result = arithmetic.calculator(arithmetic.call_expression(c.tool_calls[0]))
                    next_rounds.append((i, messages + [{"role": "assistant", "content": "", "tool_calls": [call]},
                                                       {"role": "tool", "content": result}]))
                else:
                    history[i] += [user(turn["request"]), assistant(c.content.strip())]
            rounds = next_rounds
    return [{"ok": failed[i] is None, "turns": len(s["turns"]), "failed": failed[i]} for i, s in enumerate(scripts)]


def perplexity(model: GPT, tok: Tokenizer, stories: list[str], n_batches: int = 8) -> tuple[float, float]:
    """(loss, perplexity) on held-out stories."""
    loss = evaluate_loss(model, story_batches(tok, stories, 16, model.config.block_size, n_batches),
                         model.wte.weight.device.type)
    return loss, math.exp(loss)
