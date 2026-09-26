"""Synthetic addition data, generated deterministically from a seeded random.Random.

Addition is the skill we use to *measure* every training stage: answers can be
checked exactly, and difficulty is a single knob (the number of digits). The same
generator produces:

- pretraining text: equations, sentences, word problems and worked examples;
- chat exchanges (midtrain / SFT): varied questions answered with a scratchpad in the
  assistant's "reasoning", or with a calculator tool call when tools are on, and
  follow-ups that refer to the previous result ("And add 25 to that?");
- prompts with a known answer (RL and eval).

The scratchpad adds right to left, one column per line. Each line first restates
the digits that are still left (zero-padded to the same length), so the next column
is always the last digit before "+" and ":" -- the model never has to count
positions. Then comes the column sum (with "+1" for a carry), then the digits of
the result found so far. The last line holds the full answer:

    347+058: 7+8=15, 5
    34+05: 4+5+1=10, 05
    3+0: 3+0+1=4, 405

In chat, addition is always worked out in the scratchpad, then answered "The answer is
405." A system prompt asking for the number only changes the visible answer ("405"),
not the reasoning -- like a reasoning model's hidden thoughts.
"""

from __future__ import annotations

import ast
import json
import operator
import random

QUESTIONS = [
    "What is {a} + {b}?",
    "What is {a} plus {b}?",
    "what is {a}+{b}",
    "What's {a} + {b}?",
    "Compute {a}+{b}",
    "Calculate {a} + {b}.",
    "{a} + {b} = ?",
    "{a} + {b}",
    "{a}+{b}=",
    "How much is {a} + {b}?",
    "Please add {a} and {b}.",
    "Add {a} and {b}.",
    "Can you add {a} and {b} for me?",
    "What is the sum of {a} and {b}?",
    "Find the sum of {a} and {b}.",
    "Hey, what's {a} plus {b}?",
    "I need to know {a} + {b}.",
    "Could you tell me what {a} + {b} is?",
]
FOLLOWUPS = ["And add {c} to that?", "Now add {c} to the result.", "What if you add {c} more?",
             "Plus {c}?", "Add {c} to it.", "And if I add {c}?"]
NAMES = ["Lily", "Tom", "Mia", "Ben", "Sue", "Max", "Anna", "Sam", "Lucy", "Tim"]
THINGS = ["apples", "toys", "stickers", "balls", "cookies", "flowers", "books", "shells", "stars", "coins"]
WORD_PROBLEM = "{name} has {a} {things}. {name} gets {b} more {things}. How many {things} does {name} have now?"
SENTENCES = [
    "{a} + {b} = {c}",
    "{a} plus {b} is {c}.",
    "{a} plus {b} equals {c}.",
    "The sum of {a} and {b} is {c}.",
    "If you add {a} and {b}, you get {c}.",
    "{name} had {a} {things} and found {b} more. Now {name} has {c} {things}.",
]


def sample_number(rng: random.Random, digits: int) -> int:
    """A number with exactly `digits` digits (1 digit means 0..9)."""
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10**digits - 1)


def sample_problem(rng: random.Random, digits: int) -> tuple[int, int]:
    """Two operands: one with exactly `digits` digits, the other with 1..digits digits."""
    a = sample_number(rng, digits)
    b = sample_number(rng, rng.randint(1, digits))
    return (a, b) if rng.random() < 0.5 else (b, a)


def sample_digits(rng: random.Random, digits: list[int]) -> int:
    """Pick a digit count, weighted towards longer (harder, far more numerous) problems."""
    return rng.choices(digits, weights=digits)[0]


def scratchpad(a: int, b: int) -> str:
    """Column-by-column addition, right to left (see the module docstring)."""
    n = max(len(str(a)), len(str(b)))
    A, B = str(a).zfill(n), str(b).zfill(n)
    lines, carry, done = [], 0, ""
    for i in range(n - 1, -1, -1):
        x, y = int(A[i]), int(B[i])
        s = x + y + carry
        done = (str(s) if i == 0 else str(s % 10)) + done  # the last column keeps its carry
        lines.append(f"{A[:i + 1]}+{B[:i + 1]}: {x}+{y}{'+1' if carry else ''}={s}, {done}")
        carry = s // 10
    return "\n".join(lines)


def answer_text(a: int, b: int) -> str:
    """The visible answer in chat. It only copies the sum from the end of the scratchpad:
    restating the operands ("4521 + 380 = 4901") meant copying them from the question,
    which the model garbled, and which competed with "Answer with the number only."
    (the model copied the first operand instead of the sum)."""
    return f"The answer is {a + b}."


def question(rng: random.Random, a: int, b: int) -> str:
    if rng.random() < 0.1:
        return WORD_PROBLEM.format(name=rng.choice(NAMES), things=rng.choice(THINGS), a=a, b=b)
    return rng.choice(QUESTIONS).format(a=a, b=b)


# ---- calculator tool ----------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str) -> str:
    """Evaluate + - * / on numbers, safely (no eval). Returns the result as text."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported expression")

    try:
        value = ev(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return "error"
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


def tool_call(a: int, b: int) -> dict:
    """An OpenAI-style assistant tool call to the calculator."""
    args = json.dumps({"expression": f"{a} + {b}"})
    return {"id": "call_0", "type": "function", "function": {"name": "calculator", "arguments": args}}


def call_expression(call: dict) -> str | None:
    """The expression of a parsed calculator call ({"name", "arguments": JSON text}), if valid."""
    if call.get("name") != "calculator":
        return None
    try:
        args = json.loads(call.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    expression = args.get("expression") if isinstance(args, dict) else None
    return expression if isinstance(expression, str) else None


# ---- pretraining text ---------------------------------------------------------

def pretrain_document(rng: random.Random, digits: list[int]) -> str:
    """A small 'worksheet' of equations, sentences and worked examples."""
    lines = []
    for _ in range(rng.randint(2, 6)):
        a, b = sample_problem(rng, sample_digits(rng, digits))
        r = rng.random()
        if r < 0.5:
            lines.append(f"{question(rng, a, b)}\n{scratchpad(a, b)}\nSo {a} + {b} = {a + b}.")
        else:
            fmt = rng.choice(SENTENCES)
            lines.append(fmt.format(a=a, b=b, c=a + b, name=rng.choice(NAMES), things=rng.choice(THINGS)))
    return "\n".join(lines)


# ---- chat ---------------------------------------------------------------------

def tool_round_trip(a: int, b: int) -> list[dict]:
    """The assistant calls the calculator and gets the result back."""
    return [{"role": "assistant", "content": "", "tool_calls": [tool_call(a, b)]},
            {"role": "tool", "tool_call_id": "call_0", "content": calculator(f"{a} + {b}")}]


def reply(a: int, b: int, tools: bool, number_only: bool = False) -> list[dict]:
    """The assistant's answer to a + b: a calculator round trip then the answer, or the
    scratchpad then the answer (or just the number)."""
    if tools:
        return [*tool_round_trip(a, b), {"role": "assistant", "content": answer_text(a, b)}]
    content = str(a + b) if number_only else answer_text(a, b)
    return [{"role": "assistant", "content": content, "reasoning": scratchpad(a, b)}]


def exchange(rng: random.Random, digits: list[int], tools: bool,
             number_only: bool = False) -> tuple[list[dict], int]:
    """A question and the assistant's full answer, plus the sum (for follow-ups)."""
    a, b = sample_problem(rng, sample_digits(rng, digits))
    return [{"role": "user", "content": question(rng, a, b)}, *reply(a, b, tools, number_only)], a + b


def without_reasoning(messages: list[dict]) -> list[dict]:
    """The history as an API client sends it back: earlier answers, not their scratchpads
    (OpenAI clients don't return reasoning_content). Follow-ups must work from that."""
    return [{k: v for k, v in m.items() if k != "reasoning"} for m in messages]


def followup(rng: random.Random, total: int, digits: list[int], tools: bool) -> tuple[list[dict], int]:
    """A follow-up about the previous result ("And add 25 to that?") and its answer."""
    c = sample_number(rng, rng.choice(digits))
    return [{"role": "user", "content": rng.choice(FOLLOWUPS).format(c=c)}, *reply(total, c, tools)], total + c


def prompt(rng: random.Random, digits: int, tools: bool = False, after_tool: bool = False) -> dict:
    """A question with a known answer, for RL and eval. With after_tool, the calculator
    has already been called and answered: only the final answer is left to give."""
    a, b = sample_problem(rng, digits)
    messages = [{"role": "user", "content": question(rng, a, b)}]
    if after_tool:
        messages += tool_round_trip(a, b)
    return {
        "kind": "tool" if tools or after_tool else "add",
        "messages": messages,
        "tools": ["calculator"] if tools or after_tool else None,
        "a": a, "b": b, "answer": a + b, "digits": digits,
    }
