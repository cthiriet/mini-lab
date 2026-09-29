"""The chat app's calculator: correct arithmetic, and nothing else."""

import json

import pytest

from minilab.chat.tools import CalculatorError, calculate, format_number, run_tool


@pytest.mark.parametrize("expression, expected", [
    ("347 + 58", 405),
    ("12 * 9", 108),
    ("100 - 250", -150),
    ("7 / 2", 3.5),
    ("7 // 2", 3),
    ("7 % 3", 1),
    ("2 ** 10", 1024),
    ("-(3 + 4) * 2", -14),
    ("(1 + 2) * (3 + 4)", 21),
    ("  1.5 + 2.25 ", 3.75),
    ("2 ** -1", 0.5),
])
def test_calculates(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",  # calls and names
    "abs(-3)",
    "x + 1",
    "(1).real",                       # attributes
    "[1, 2][0]",                      # subscripts / lists
    "1 if 1 else 2",
    "1 < 2",
    "True + 1",                       # booleans are not numbers here
    "'a' * 3",                        # strings
    "lambda: 1",
    "2 ** 1000",                      # exponent too large
    "9 ** 9 ** 9",
    "10 ** 30 * 10 ** 30",            # result too large
    "1 / 0",
    "5 % 0",
    "(-8) ** 0.5",                    # complex result
    "1 +",                            # syntax error
    "",
    "1" * 300,                        # too long
])
def test_rejects(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_format_number():
    assert format_number(405) == "405"
    assert format_number(5.0) == "5"
    assert format_number(2.5) == "2.5"
    assert format_number(1 / 3) == "0.3333333333"


def test_run_tool():
    assert run_tool("calculator", json.dumps({"expression": "347 + 58"})) == {
        "name": "calculator", "input": "347 + 58", "output": "405", "ok": True}
    assert run_tool("calculator", {"expression": "10 / 4"})["output"] == "2.5"
    assert run_tool("calculator", "3 * 3")["output"] == "9"  # a model that forgot the JSON wrapper
    bad = run_tool("calculator", json.dumps({"expression": "open('/etc/passwd')"}))
    assert not bad["ok"] and bad["output"].startswith("error:")
    assert not run_tool("weather", "{}")["ok"]


def test_recent_turns_leaves_room_for_the_answer():
    from minilab.tokenizer.chat import recent_turns
    story = {"role": "assistant", "content": "Once upon a time " * 40}  # ~170 tokens
    messages = [{"role": "user", "content": "Tell me a story."}, story, {"role": "user", "content": "What is 347 + 58?"}]
    assert recent_turns(messages, 256) == messages[-1:]
    short = [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "What is 2 + 2?"}]
    assert recent_turns(short, 256) == short
    assert recent_turns([{"role": "user", "content": "x" * 5000}], 256)[0]["content"] == "x" * 5000  # never empty
    # digits are one token each: 20 turns of 3-digit additions keep far fewer turns than 20
    additions = [m for i in range(20) for m in ({"role": "user", "content": f"What is {100 + i} + {250 + i}?"},
                                                 {"role": "assistant", "content": f"The answer is {350 + 2 * i}."})]
    kept = recent_turns(additions + [{"role": "user", "content": "What is 34521 + 88790?"}], 256)
    assert sum(m["role"] == "user" for m in kept) <= 6
