"""Why the model can't add 6-digit numbers: the measurements of docs/interp.md, reproduced.

    uv run python -m minilab.interp.addition models/prelude-2
    uv run python -m minilab.interp.addition models/prelude-2 --features 1112 868

Every measurement forces the reference scratchpad (data/arithmetic.py) after the question and
reads the model's prediction at one position, with the microscope (trace.py):

1. where it breaks: the accuracy of each part of a 6-digit scratchpad;
2. who ends the number: direct attribution of logit("+") - logit(the right digit) after k digits
   of a 7-digit operand, for the MLPs of layers 1-3 and the heads of layers 4-5;
3. causal check: one component mean-ablated where the 6th digit is due;
4. with --features (two of the dictionary's, in models/<id>/features/): removed where the 6th
   digit is due, and added after the 3rd digit of a 5-digit operand.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch

from minilab.checkpoint import load_checkpoint
from minilab.data import arithmetic
from minilab.interp.trace import attribution, trace
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_prompt
from minilab.train.trainer import DEVICES, resolve_device


def problems(tok: Tokenizer, width: int, n: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """(prompt ending with <|think_start|>, the reference scratchpad) for n additions whose
    longer operand has `width` digits."""
    rng, out = random.Random(f"interp-{width}-{seed}"), []
    for _ in range(n):
        a, b = arithmetic.sample_number(rng, width), arithmetic.sample_number(rng, rng.randint(1, width))
        prompt = render_prompt(tok, [{"role": "user", "content": f"What is {a} + {b}?"}]) + [tok.special("<|think_start|>")]
        out.append((prompt, tok.encode(arithmetic.scratchpad(a, b))))
    return out


def roles(scratchpad: str) -> list[str]:
    """What each character (= token: digits are always split) of a scratchpad is."""
    out = []
    for i, line in enumerate(scratchpad.split("\n")):
        head, rest = line.split(": ")
        a, b = head.split("+")
        out += [f"line {i + 1}, digit {j + 1} of A" for j in range(len(a))] + [f"line {i + 1}, the '+'"]
        out += [f"line {i + 1}, digit {j + 1} of B" for j in range(len(b))] + [f"line {i + 1}, the ':'"] * 2
        column, result = rest.split(", ")
        out += [f"line {i + 1}, column sum"] * len(column) + [f"line {i + 1}, the ','"] * 2
        out += [f"line {i + 1}, result digit {j + 1}" for j in range(len(result))] + [f"line {i + 1}, end of line"]
    return out[:-1]


def where_it_breaks(model: GPT, tok: Tokenizer, width: int = 6, n: int = 60) -> dict[str, float]:
    """Teacher-forced accuracy of every part of the scratchpad that is ever wrong."""
    right = defaultdict(list)
    for prompt, pad in problems(tok, width, n):
        pred = trace(model, prompt + pad).logits.argmax(-1).tolist()
        for j, (t, role) in enumerate(zip(pad, roles(tok.decode(pad)))):
            right[role].append(pred[len(prompt) + j - 1] == t)
    return {role: sum(v) / len(v) for role, v in right.items() if sum(v) < len(v)}


def who_ends_the_number(model: GPT, tok: Tokenizer, n: int = 30) -> list[tuple[int, float, float]]:
    """(k, push of the layer 1-3 MLPs, push of the layer 4-5 heads) toward "+" over the right
    digit, after k digits of a 7-digit operand A."""
    plus, H = tok.encode("+")[0], model.config.n_head
    push = defaultdict(lambda: [0.0, 0.0])
    for prompt, pad in problems(tok, 7, n, seed=1):
        ids = prompt + pad
        tr = trace(model, ids)
        for k in range(1, 7):
            pos = len(prompt) + k - 1
            att = attribution(model, tr, pos, plus, ids[pos + 1])
            push[k][0] += sum(att[f"L{layer}.mlp"] for layer in (1, 2, 3)) / n
            push[k][1] += sum(att[f"L{layer}.H{h}"] for layer in (4, 5) for h in range(H)) / n
    return [(k, mlp, heads) for k, (mlp, heads) in sorted(push.items())]


def _mean_at(pos: int):
    def ablate(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[pos] = x.mean(0)
        return x
    return ablate


def sixth_digit(model: GPT, tok: Tokenizer, patch_at, n: int = 40) -> int:
    """How often the 6th digit of a 6-digit A comes out right, with patch_at(pos) applied."""
    right = 0
    for prompt, pad in problems(tok, 6, n, seed=2):
        ids, pos = prompt + pad, len(prompt) + 4   # 5 digits written
        right += int(trace(model, ids, patch_at(pos)).logits[pos].argmax()) == ids[pos + 1]
    return right


def causal_check(model: GPT, tok: Tokenizer, n: int = 40) -> list[tuple[str, int]]:
    out = [("nothing", sixth_digit(model, tok, lambda pos: {}, n))]
    for layer in (1, 2, 3):
        out.append((f"MLP of layer {layer}", sixth_digit(model, tok, lambda pos: {("mlp", layer): _mean_at(pos)}, n)))
    return out


def feature_check(model: GPT, tok: Tokenizer, folder: Path, ids: list[int], n: int = 40) -> list[tuple[str, int]]:
    from minilab.interp.sae import load_sae
    sae, layer = load_sae(folder, str(model.wte.weight.device))
    plus = tok.encode("+")[0]

    def edit(pos: int, amount: float | None):
        """At pos, take each feature's contribution out (amount None) or add amount of it."""
        def f(x: torch.Tensor) -> torch.Tensor:
            x, acts = x.clone(), sae.encode(x[pos:pos + 1])[0]
            for i in ids:
                x[pos] += (-acts[i] if amount is None else amount) * sae.W_dec[i] * sae.scale
            return x
        return {("resid", layer): f}

    names = " + ".join(f"#{i}" for i in ids)
    out = [(f"6-digit A, {names} removed: 6th digit right", sixth_digit(model, tok, lambda pos: edit(pos, None), n))]
    for amount in (4.0, 8.0):
        plus_early = 0
        for prompt, pad in problems(tok, 5, n, seed=3):
            seq, pos = prompt + pad, len(prompt) + 2   # 3 digits of a 5-digit A written
            plus_early += int(trace(model, seq, edit(pos, amount)).logits[pos].argmax()) == plus
        out.append((f"5-digit A, {names} added x{amount:g} after 3 digits: '+' written", plus_early))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Why the model can't add 6-digit numbers (docs/interp.md).")
    p.add_argument("model", help="a release directory, e.g. models/prelude-2")
    p.add_argument("--features", type=int, nargs="*", default=[], help="dictionary features to remove and add")
    p.add_argument("--device", default="auto", choices=DEVICES)
    args = p.parse_args()
    model, tok, _ = load_checkpoint(args.model, device=resolve_device(args.device))

    print("1. Where a 6-digit scratchpad breaks (teacher-forced, every part ever wrong):")
    for role, acc in where_it_breaks(model, tok).items():
        print(f"   {role:28s} {acc:5.0%}")
    print("2. Push toward '+' over the right digit after k digits of a 7-digit A (direct attribution):")
    for k, mlp, heads in who_ends_the_number(model, tok):
        print(f"   k={k}   MLPs 1-3 {mlp:+6.1f}   heads 4-5 {heads:+6.1f}")
    print("3. Mean-ablated where the 6th digit of a 6-digit A is due (6th digit right, of 40):")
    for name, right in causal_check(model, tok):
        print(f"   {name:18s} {right}")
    if args.features:
        print("4. Dictionary features (of 40):")
        for name, count in feature_check(model, tok, Path(args.model) / "features", args.features):
            print(f"   {name}: {count}")


if __name__ == "__main__":
    main()
