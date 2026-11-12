"""A calculator inside the reasoning, as in GSM8K (Cobbe et al. 2021).

When the model writes "=" in <|think_start|>...<|think_end|> right after an arithmetic expression
("70 - 32 ="), the result is written for it ("38") and the model goes on from there. Training is
unchanged: the model only ever sees reasoning where its sums are right. ThinkCalculator is the
per-sequence state; think_force() adapts it to GPT.generate(force=...), the engine uses it directly.
"""

from __future__ import annotations

import functools

from minilab.data.arithmetic import think_result
from minilab.tokenizer.bpe import Tokenizer

TAIL = 64  # bytes of reasoning kept: the end of an expression is all think_result needs


@functools.cache
def _ids(tok: Tokenizer) -> tuple[int, int, frozenset[int]]:
    """<|think_start|>, <|think_end|>, and the tokens " =" and "=" (when each is a single token)."""
    eq = frozenset(ids[0] for s in (" =", "=") if len(ids := tok.encode(s)) == 1)
    return tok.special("<|think_start|>"), tok.special("<|think_end|>"), eq


class ThinkCalculator:
    """Follows one sequence token by token; feed() returns the tokens to write after `t`, if any."""

    def __init__(self, tok: Tokenizer, prompt: list[int] = ()):
        self.tok = tok
        self.start, self.end, self.eq = _ids(tok)
        self.in_think = bool(prompt) and prompt[-1] == self.start  # the prompt may already open the reasoning
        self.tail = b""

    def feed(self, t: int) -> list[int]:
        if t == self.start:
            self.in_think = True
        elif t == self.end:
            self.in_think = False
        elif self.in_think and not self.tok.is_special(t):
            self.tail = (self.tail + self.tok.token_bytes(t))[-TAIL:]
            if t in self.eq:
                result = think_result(self.tail.decode(errors="ignore").rsplit("\n", 1)[-1])
                if result is not None:
                    self.tail = (self.tail + result.encode())[-TAIL:]
                    return self.tok.encode(result)
        return []


def think_force(tok: Tokenizer, prompts: list[list[int]]):
    """The `force` of GPT.generate for these prompts (slot i is prompts[i])."""
    states = [ThinkCalculator(tok, p) for p in prompts]
    return lambda slot, outputs, token: states[slot].feed(token)
