"""Tools the platform runs on behalf of the model. There is exactly one: a calculator.

The model asks for it with a tool call like
    {"name": "calculator", "arguments": "{\"expression\": \"347 + 58\"}"}
and we answer with a tool message containing "405".

The expression comes from a (tiny, easily confused) model, so it is untrusted input.
We never call eval(): we parse it with `ast` and walk the tree ourselves, allowing
only numbers, parentheses and a handful of arithmetic operators.
"""

from __future__ import annotations

import ast
import json
import operator

# OpenAI-style tool definition, sent to the gateway when the calculator is enabled.
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '347 + 58'.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
            "required": ["expression"],
        },
    },
}

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 64          # 2 ** 64 is fine, 9 ** 99999 would eat the CPU
MAX_MAGNITUDE = 10 ** 30   # keeps every intermediate result small, so chains like 10**30 * 10**30... stay cheap

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    pass


def calculate(expression: str) -> int | float:
    """Evaluate an arithmetic expression safely. Raises CalculatorError on anything else."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalculatorError("not a valid arithmetic expression") from None
    return _eval(tree.body)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, and True + True is not arithmetic we want to support.
        if type(node.value) in (int, float):
            return _checked(node.value)
        raise CalculatorError("only numbers are allowed")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent is too large (max {MAX_EXPONENT})")
        try:
            result = _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("result is too large") from None
        if isinstance(result, complex):  # (-8) ** 0.5
            raise CalculatorError("result is not a real number")
        return _checked(result)
    # Names, calls, attributes, subscripts, comparisons, lambdas... all end up here.
    raise CalculatorError(f"'{type(node).__name__}' is not allowed, only numbers and + - * / // % **")


def _checked(value: int | float) -> int | float:
    if abs(value) > MAX_MAGNITUDE:
        raise CalculatorError("result is too large")
    return value


def format_number(value: int | float) -> str:
    """405 -> '405', 2.5 -> '2.5', 10 / 4 * 2 -> '5' (no trailing '.0'), 1 / 3 -> '0.3333333333'."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def run_tool(name: str, arguments: str | dict | None) -> dict:
    """Run one tool call from the model.

    Returns {"name", "input", "output", "ok"}: `output` is what we send back to the
    model as the tool message; `input` is what we show in the UI.
    """
    if name != "calculator":
        return {"name": name, "input": str(arguments or ""), "output": f"error: unknown tool '{name}'", "ok": False}
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"expression": args}  # a model that forgot the JSON wrapper
    expression = str(args.get("expression", "")) if isinstance(args, dict) else str(args)
    try:
        return {"name": name, "input": expression, "output": format_number(calculate(expression)), "ok": True}
    except CalculatorError as e:
        return {"name": name, "input": expression, "output": f"error: {e}", "ok": False}
