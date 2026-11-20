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


def test_prompt_budget_leaves_room_for_the_answer():
    from minilab.tokenizer.chat import prompt_budget
    assert prompt_budget(1024, None) == 768   # a quarter of the context for the answer by default
    assert prompt_budget(1024, 100) == 924
    assert prompt_budget(1024, 5000) == 512   # never more than half


def _effort_tok():
    from minilab.tokenizer.bpe import Tokenizer
    return Tokenizer.train("tools: calculator, effort: low high. What is 2 + 2? You are helpful.", vocab_size=400)


EFFORT_CONVERSATIONS = [
    ([{"role": "user", "content": "hi"}], None),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2 + 2?"}], ["calculator"]),
    ([{"role": "user", "content": "2 + 2?"},
      {"role": "assistant", "content": "4", "reasoning": "2 + 2 = 4"},
      {"role": "user", "content": "and 3 + 3?"}], ["calculator"]),
]


def test_effort_header_only_for_low():
    from minilab.tokenizer.chat import render_conversation, render_prompt
    tok = _effort_tok()
    S = tok.special
    for messages, tools in EFFORT_CONVERSATIONS:
        default = render_prompt(tok, messages, tools)
        for effort in ("high", "medium", "none", None):
            assert render_prompt(tok, messages, tools, effort=effort) == default
            assert render_conversation(tok, messages, tools, effort=effort) == render_conversation(tok, messages, tools)
        for effort in ("low",):
            line = [S("<|system_start|>"), *tok.encode(f"effort: {effort}"), S("<|system_end|>")]
            ids = render_prompt(tok, messages, tools, effort=effort)
            n = len(line)
            at = 1 + (len(tok.encode("tools: calculator")) + 2 if tools else 0)  # after bos and the tools line
            assert ids[at:at + n] == line and ids[:at] + ids[at + n:] == default
            assert render_prompt(tok, messages, tools, budget=500, effort=effort) == ids
            full, mask = render_conversation(tok, messages, tools, effort=effort)
            assert full[at:at + n] == line and mask[at:at + n] == [0] * n


def test_default_rendering_is_unchanged():
    """The header without effort is bos + the tools line, as before the effort line existed."""
    from minilab.tokenizer.chat import render_prompt
    tok = _effort_tok()
    S = tok.special
    expected = [tok.bos_id, S("<|system_start|>"), *tok.encode("tools: calculator"), S("<|system_end|>"),
                S("<|user_start|>"), *tok.encode("hi"), S("<|user_end|>"), S("<|assistant_start|>")]
    assert render_prompt(tok, [{"role": "user", "content": "hi"}], ["calculator"]) == expected
