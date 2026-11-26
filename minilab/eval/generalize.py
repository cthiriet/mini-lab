"""The generalization panel: does the model do what it learned, or only what it saw?

Every axis is one task in two versions, graded the same way: the version training showed, and
one it never did (a longer number, a third operand, other words for the same request, a new
topic...). The axis's ratio is the second score over the first; the panel's number is the mean
ratio. A model that understood its tasks scores near 1 everywhere; one that memorized them,
near 0 where the new version starts.

    longer additions      5 digits                      -> 6 digits
    three numbers         "What is 12 + 34?"            -> "What is 12 + 34 + 56?"
    additions reworded    the training's questions      -> new ones ("Sum up 412 and 37.")
    new story topics      the 15 topics chat taught     -> 15 it never asked for ("a frog")
    new refusals          trained out-of-scope requests -> the eval's held-out ones
    repair reworded       fix_test, training prompts    -> the same tasks, new prompts, as detailed
    rename reworded       rename, training prompts      -> the same tasks, new prompts

The never-seen versions are never trained on (tests/test_generalize.py checks the phrasings and
topics). The two code axes play the tasks in the Docker sandbox, like the coding eval.

    uv run python -m minilab.eval.generalize models/prelude-2
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from pathlib import Path

from minilab.data import arithmetic, code
from minilab.data.conversations import TOPICS, is_refusal, mentions, user
from minilab.eval import code as code_eval
from minilab.eval import tasks
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import parse_completion, render_prompt

SEED = "generalize"
QUESTIONS = [  # none of arithmetic.QUESTIONS's
    "Sum up {a} and {b}.", "Tell me the total of {a} and {b}.", "What do {a} and {b} add up to?",
    "If I add {b} to {a}, what do I get?", "Work out {a} + {b} for me.", "Quick question: {a} + {b}?",
    "{a} and {b} together make how much?", "Give me the sum of {a} and {b}, please.",
]
THREE = [  # each a two-operand question of arithmetic.QUESTIONS with a third operand
    ("What is {a} + {b}?", "What is {a} + {b} + {c}?"), ("{a} + {b}", "{a} + {b} + {c}"),
    ("What is {a} plus {b}?", "What is {a} plus {b} plus {c}?"), ("Calculate {a} + {b}.", "Calculate {a} + {b} + {c}."),
    ("{a} + {b} = ?", "{a} + {b} + {c} = ?"), ("How much is {a} + {b}?", "How much is {a} + {b} + {c}?"),
]
NEW_TOPICS = {"a frog": "frog", "a lion": "lion", "a robot": "robot", "a train": "train", "the moon": "moon",
              "a flower": "flower", "a bear": "bear", "a kite": "kite", "a monkey": "monkey", "a garden": "garden",
              "a duck": "duck", "a horse": "horse", "the snow": "snow", "a castle": "castle", "a truck": "truck"}
REPAIR = {  # by what the request names: nothing, the test, or the function and the test
    "none": ["Some tests don't pass. Can you fix the code?", "Get the test suite passing again.",
             "Fix whatever is breaking the tests.", "The tests are red. Find the problem and repair it."],
    "test": ["{test} doesn't pass. Find out why and repair it.", "Please get {test} passing again.",
             "Can you repair whatever {test} complains about?", "{test} reports a failure. Sort it out."],
    "func": ["{func} has a mistake. Repair it, then run {test}.", "Something is off in {func}. Correct it; {test} checks it.",
             "Please repair {func} and confirm with {test}."],
}
RENAME = ["Give {f} the new name {g}.", "{f} should be called {g} from now on.",
          "Please switch the name {f} to {g} across the project.", "I'd like {f} renamed: use {g}."]
RENAME_IN = ["{f} (in {path}) should be called {g} from now on.", "In {path}, give {f} the new name {g}."]


# ---- chat axes --------------------------------------------------------------------------

def _add(messages: list[dict], total: int) -> dict:
    """A problem graded like an addition whose answer is `total` (answer_text(a, b) = a + b)."""
    return {"kind": "add", "messages": messages, "tools": None, "a": total, "b": 0, "answer": total}


def _score(model: GPT, tok: Tokenizer, problems: list[dict], max_new: int = 300) -> float:
    outs = tasks.generate(model, [render_prompt(tok, p["messages"], p["tools"]) for p in problems], max_new,
                          {tok.special("<|assistant_end|>")})
    return sum(tasks.grade(tok, o, p) for p, o in zip(problems, outs)) / len(problems)


def longer(model, tok, n):
    return _score(model, tok, tasks.arithmetic_problems(5, n)), _score(model, tok, tasks.arithmetic_problems(6, n))


def three_numbers(model, tok, n):
    rng, two, three = random.Random(f"{SEED}-three"), [], []
    for _ in range(n):
        a, b, c = (arithmetic.sample_number(rng, 2) for _ in range(3))
        q2, q3 = rng.choice(THREE)
        two.append(_add([user(q2.format(a=a, b=b))], a + b))
        three.append(_add([user(q3.format(a=a, b=b, c=c))], a + b + c))
    return _score(model, tok, two), _score(model, tok, three)


def reworded_additions(model, tok, n):
    rng, old, new = random.Random(f"{SEED}-reworded"), [], []
    for _ in range(n):
        a, b = arithmetic.sample_problem(rng, 3)
        old.append(_add([user(rng.choice(arithmetic.QUESTIONS).format(a=a, b=b))], a + b))
        new.append(_add([user(rng.choice(QUESTIONS).format(a=a, b=b))], a + b))
    return _score(model, tok, old), _score(model, tok, new)


def _on_topic(model, tok, topics: dict[str, str]) -> float:
    requests = [(t.format(t=topic), word) for topic, word in topics.items() for t in tasks.STORY_EVAL_REQUESTS]
    outs = tasks.generate(model, [render_prompt(tok, [user(r)]) for r, _ in requests], model.config.block_size,
                          {tok.special("<|assistant_end|>")})
    ok = 0
    for (_, word), out in zip(requests, outs):
        c = parse_completion(tok, out)
        ok += c.finished and mentions(c.content, word) and not is_refusal(c.content)
    return ok / len(requests)


def new_topics(model, tok, n):
    return _on_topic(model, tok, TOPICS), _on_topic(model, tok, NEW_TOPICS)


def new_refusals(model, tok, n):
    rng = random.Random(f"{SEED}-refusal")
    old = [tasks.make_problem("refusal", rng, [1, 2, 3]) for _ in range(n)]
    new = [tasks.make_problem("refusal", rng, [1, 2, 3], held_out=True) for _ in range(n)]
    return _score(model, tok, old, 100), _score(model, tok, new, 100)


# ---- code axes (Docker) -------------------------------------------------------------------

def reword_repair(rng: random.Random, task: code.Task) -> str:
    test = re.search(r"\S*test_\w+\.py", task.prompt)
    if task.title.startswith("Fix bug in "):
        return rng.choice(REPAIR["func"]).format(func=task.title.removeprefix("Fix bug in "), test=test[0])
    if test:
        return rng.choice(REPAIR["test"]).format(test=test[0])
    return rng.choice(REPAIR["none"])


def reword_rename(rng: random.Random, task: code.Task) -> str:
    f, g = re.fullmatch(r"Rename (\w+) to (\w+)", task.title).groups()
    path = re.search(r"\S+\.py", task.prompt)
    return rng.choice(RENAME_IN).format(f=f, g=g, path=path[0]) if path else rng.choice(RENAME).format(f=f, g=g)


def _play(model, tok, container, tasks_: list[code.Task], cfg: dict) -> float:
    episodes = []
    for i, task in enumerate(tasks_):
        sb = code_eval.DockerSandbox(container, task.files)
        messages = [code.system_message(str(sb.root), random.Random(i)), user(task.prompt)]   # the same in both versions
        episodes.append(code_eval.Episode(task, sb, messages, code_eval.OPENCODE_TOOLS))
    code_eval.play(model, tok, episodes, cfg.get("max_steps", 8), cfg.get("max_new_tokens", 384), cfg.get("batch", 32))
    ok = 0
    for e in episodes:
        try:
            ok += bool(e.task.check(e.sb, e.answer or ""))
        except Exception:  # a check on a project the model mangled
            pass
        e.sb.close()
    return ok / len(episodes)


def code_axes(model, tok, n, cfg: dict) -> dict[str, tuple[float, float]]:
    out = {}
    with code_eval.Container() as container:
        for name, kind, reword in [("repair reworded", "fix_test", reword_repair), ("rename reworded", "rename", reword_rename)]:
            rng = random.Random(f"{SEED}-{kind}")
            trained = [code.make_task(rng, kind) for _ in range(n)]
            new = [replace(t, prompt=reword(rng, t)) for t in trained]
            out[name] = (_play(model, tok, container, trained, cfg), _play(model, tok, container, new, cfg))
    return out


# ---- the panel ----------------------------------------------------------------------------

CHAT_AXES = {"longer additions": longer, "three numbers": three_numbers, "additions reworded": reworded_additions,
             "new story topics": new_topics, "new refusals": new_refusals}


def panel(model: GPT, tok: Tokenizer, n: int = 100, n_code: int = 50, cfg: dict | None = None,
          with_code: bool = True) -> dict:
    """{axis: {"trained", "new", "ratio"}} and "ratio", the mean over the axes."""
    scores = {name: fn(model, tok, n) for name, fn in CHAT_AXES.items()}
    if with_code:
        scores |= code_axes(model, tok, n_code, cfg or {})
    axes = {name: {"trained": round(a, 3), "new": round(b, 3), "ratio": round(min(b / a, 1.0), 3) if a else None}
            for name, (a, b) in scores.items()}
    ratios = [x["ratio"] for x in axes.values() if x["ratio"] is not None]
    return {"axes": axes, "ratio": round(sum(ratios) / len(ratios), 3)}


def main() -> None:
    import argparse
    import tomllib

    from minilab.checkpoint import load_checkpoint
    from minilab.train.trainer import DEVICES, resolve_device, setup
    p = argparse.ArgumentParser(description="The generalization panel: trained vs never-seen versions of each task.")
    p.add_argument("model", help="a checkpoint or release directory")
    p.add_argument("--config", default="configs/prelude.toml", help="for [code_eval]")
    p.add_argument("--n", type=int, default=100, help="problems per version of each chat axis")
    p.add_argument("--n-code", type=int, default=50, help="tasks per version of each code axis")
    p.add_argument("--no-code", action="store_true", help="skip the code axes (no Docker)")
    p.add_argument("--out", help="write the result as JSON")
    p.add_argument("--device", default="auto", choices=DEVICES)
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    device = resolve_device(args.device, generation=True)
    setup(cfg.get("seed", 0), device)
    model, tok, _ = load_checkpoint(args.model, device=device)
    result = panel(model, tok, args.n, args.n_code, cfg.get("code_eval"), not args.no_code)
    for name, x in result["axes"].items():
        ratio = "–" if x["ratio"] is None else f"{x['ratio']:.2f}"
        print(f"{name:20s} {100 * x['trained']:5.0f}% -> {100 * x['new']:5.0f}%   ratio {ratio}")
    print(f"{'generalization ratio':20s} {result['ratio']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
