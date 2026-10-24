"""The horizon benchmark: how long a coding task the model completes, half of the time.

The long tasks of data/horizon.py (repair-k: k bugs in one request; session-k: k requests in a
row) are played end to end like the coding eval (eval/code.py): opencode's messages, every tool
call in the Docker sandbox. A session goes on request by request, each one checked after its
turn, and stops at the first failed one. The model may make twice the oracle's tool calls,
plus 4, per request.

The *horizon* is METR's time horizon, with the oracle's tool calls in place of human time: a
logistic fit of success on log2(length), read where it crosses 50%. Reported overall, per
family, with a bootstrap interval; and the success rate per k.

    uv run python -m minilab.eval.horizon models/prelude-2 --device mps
"""

from __future__ import annotations

import json
import math
import random
import sys
import time

import torch

from minilab.data import code
from minilab.data import horizon as long
from minilab.data.sandbox import OPENCODE_TOOLS, Container, DockerSandbox
from minilab.eval.code import Episode, play
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer

EVAL_SEED = "horizon-eval"
DEFAULTS = {"ks": {"repair": [1, 2, 3, 4, 6, 8, 12, 16], "session": [1, 2, 4, 8, 16, 32]},
            "n_per_k": 16, "max_new_tokens": 384, "batch": 32}


def long_tasks(ks: dict[str, list[int]], n: int) -> list[long.LongTask]:
    tasks = []
    for family, values in ks.items():
        for k in values:
            rng = random.Random(f"{EVAL_SEED}-{family}-{k}")
            tasks += [long.make_long_task(rng, family, k, held_out=True) for _ in range(n)]
    return tasks


@torch.no_grad()
def play_long(model: GPT, tok: Tokenizer, tasks: list[long.LongTask], max_new: int, batch: int) -> list[dict]:
    """Play every long task, all of them in lockstep, request by request."""
    with Container() as container:
        eps = []
        for j, t in enumerate(tasks):
            sb = DockerSandbox(container, t.files)
            eps.append(Episode(None, sb, [code.system_message(str(sb.root), random.Random(f"{EVAL_SEED}-{j}"))],
                               OPENCODE_TOOLS))
        passed, alive = [0] * len(tasks), [True] * len(tasks)
        calls, valid = [0] * len(tasks), [0] * len(tasks)
        for i in range(max(len(t.requests) for t in tasks)):
            active = [j for j, t in enumerate(tasks) if alive[j] and i < len(t.requests)]
            if not active:
                break
            print(f"request {i + 1}: {len(active)} tasks still going", file=sys.stderr, flush=True)
            for j in active:
                r, e = tasks[j].requests[i], eps[j]
                e.messages.append({"role": "user", "content": r.prompt})
                e.transcript.append(f"user: {r.prompt}")
                e.done, e.answer, e.calls, e.valid, e.max_calls = False, None, 0, 0, 2 * r.calls + 4
            play(model, tok, [eps[j] for j in active], 8, max_new, batch)
            for j in active:
                e = eps[j]
                calls[j], valid[j] = calls[j] + e.calls, valid[j] + e.valid
                e.messages.append({"role": "assistant", "content": e.answer or ""})
                try:
                    ok = bool(tasks[j].requests[i].check(e.sb, e.answer or ""))
                except Exception:   # a check on a project the model mangled
                    ok = False
                passed[j] += ok
                alive[j] = ok
        for e in eps:
            e.sb.close()
    return [{"family": t.family, "k": t.k, "length": t.length, "requests": len(t.requests), "passed": passed[j],
             "success": passed[j] == len(t.requests), "calls": calls[j], "valid": valid[j],
             "failed_kind": None if passed[j] == len(t.requests) else t.requests[passed[j]].kind,
             "transcript": eps[j].transcript}
            for j, t in enumerate(tasks)]


def fit_horizon(points: list[tuple[int, bool]], ridge: float = 1e-3) -> float:
    """The length where a logistic fit of success on log2(length) crosses 50% (Newton's method,
    a little ridge so that all-success or all-failure stays finite). inf if success doesn't fall
    with length."""
    a = b = 0.0
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for length, ok in points:
            x = math.log2(length)
            p = 1 / (1 + math.exp(-max(-30.0, min(30.0, a + b * x))))
            ga, gb = ga + (p - ok), gb + (p - ok) * x
            w = p * (1 - p)
            haa, hab, hbb = haa + w, hab + w * x, hbb + w * x * x
        ga, gb, haa, hbb = ga + ridge * a, gb + ridge * b, haa + ridge, hbb + ridge
        det = haa * hbb - hab * hab
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:
            break
    return math.inf if b >= 0 else 2 ** (-a / b)


def horizon_with_interval(points: list[tuple[int, bool]], n_boot: int = 200) -> dict:
    """The horizon, and the 5th-95th percentiles of the tasks resampled with replacement."""
    rng = random.Random(0)
    boots = sorted(fit_horizon([rng.choice(points) for _ in points]) for _ in range(n_boot))
    return {"horizon": _round(fit_horizon(points)),
            "interval": [_round(boots[int(0.05 * n_boot)]), _round(boots[int(0.95 * n_boot) - 1])]}


def _round(x: float) -> float | None:
    return None if math.isinf(x) else round(x, 1)   # None: beyond anything measurable here


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(records: list[dict]) -> dict:
    result = horizon_with_interval([(r["length"], r["success"]) for r in records])
    result["families"] = {}
    for family in dict.fromkeys(r["family"] for r in records):
        rs = [r for r in records if r["family"] == family]
        fam = horizon_with_interval([(r["length"], r["success"]) for r in rs])
        fam["by_k"] = {str(k): {"success": _mean(r["success"] for r in rs if r["k"] == k),
                                "length": _mean(r["length"] for r in rs if r["k"] == k),
                                "requests_passed": _mean(r["passed"] / r["requests"] for r in rs if r["k"] == k)}
                       for k in dict.fromkeys(r["k"] for r in rs)}
        fails = [r for r in rs if not r["success"]]
        fam["failed_kinds"] = {k: sum(r["failed_kind"] == k for r in fails) for k in sorted({r["failed_kind"] for r in fails})}
        fam["failures"] = [{"k": r["k"], "transcript": r["transcript"][-16:]}
                           for r in sorted(fails, key=lambda r: r["k"])[:3]]
        result["families"][family] = fam
    calls = sum(r["calls"] for r in records)
    result["valid_calls"] = sum(r["valid"] for r in records) / calls if calls else None
    result["n_tasks"] = len(records)
    return result


def evaluate(model: GPT, tok: Tokenizer, hc: dict | None = None) -> dict:
    hc = {**DEFAULTS, **(hc or {})}
    t0 = time.time()
    tasks = long_tasks(hc["ks"], hc["n_per_k"])
    print(f"{len(tasks)} long tasks made in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    records = play_long(model, tok, tasks, hc["max_new_tokens"], hc["batch"])
    result = summarize(records)
    result["eval_seconds"] = round(time.time() - t0, 1)
    return result


def _fmt(h: dict) -> str:
    def show(x: float | None) -> str:
        return "inf" if x is None else f"{x:g}"
    lo, hi = h["interval"]
    return f"{show(h['horizon'])} calls (90%: {show(lo)}-{show(hi)})"


def main() -> None:
    import argparse
    from pathlib import Path

    from minilab.checkpoint import load_checkpoint
    from minilab.train.trainer import DEVICES, resolve_device
    p = argparse.ArgumentParser(description="The horizon benchmark: long coding tasks, tools in a Docker sandbox.")
    p.add_argument("model", help="a checkpoint or release directory")
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--n", type=int, default=DEFAULTS["n_per_k"], help="tasks per family and k")
    p.add_argument("--families", nargs="+", default=list(DEFAULTS["ks"]), choices=long.FAMILIES)
    p.add_argument("--out", help="write the result as JSON here")
    args = p.parse_args()
    model, tok, _ = load_checkpoint(args.model, device=resolve_device(args.device, generation=True))
    result = evaluate(model, tok, {"ks": {f: DEFAULTS["ks"][f] for f in args.families}, "n_per_k": args.n})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"horizon: {_fmt(result)}   valid calls {result['valid_calls']:.1%}   {result['eval_seconds']:.0f}s")
    for family, fam in result["families"].items():
        print(f"\n{family}: {_fmt(fam)}")
        for k, row in fam["by_k"].items():
            print(f"  k={k:>2}  length {row['length']:5.1f}  success {row['success']:6.1%}  "
                  f"requests passed {row['requests_passed']:6.1%}")
        if fam["failed_kinds"]:
            print("  failed on: " + ", ".join(f"{k} {n}" for k, n in fam["failed_kinds"].items()))


if __name__ == "__main__":
    main()
