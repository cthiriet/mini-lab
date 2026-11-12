"""The probe's direct accuracy per bucket (docs/story-math-spec.md §3.3): the table the think/answer policy reads.

    python scripts/probe_direct.py --run runs/probe-e1 --stage midtrain

Answers n fresh problems per number of events (seed B, `policy_seed`: the probe never saw them) with no reasoning,
groups them by bucket (events, doubling) and writes {"tau", "p", "n", "source", "seed"} to --out
([data.reason].policy). A bucket measured on fewer than MIN_N problems is left out of "p": it will think.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from minilab.checkpoint import load_checkpoint
from minilab.data import story_math as sm
from minilab.eval import tasks
from minilab.train.trainer import DEVICES, STAGES, load_config, resolve_device, setup

MIN_N = 30
LEVELS = ["low", "medium", "high"]


def measure(model, tok, seed: int, steps: list[int], n: int) -> dict[tuple[int, int], list[bool]]:
    """Direct (no reasoning) correctness of n problems per k, by bucket."""
    results = defaultdict(list)
    for k in steps:
        rng = random.Random(f"{seed}-{k}")
        problems = [sm.make(rng, k, "train") for _ in range(n)]
        for r in tasks.eval_story_math(model, tok, problems, 32, "medium", think=False):
            results[tuple(r["bucket"])].append(r["correct"])
    return results


def policy(results: dict, tau: dict, source: str, seed: int) -> dict:
    n = {f"{k},{d}": len(v) for (k, d), v in sorted(results.items())}
    p = {f"{k},{d}": sum(v) / len(v) for (k, d), v in sorted(results.items()) if len(v) >= MIN_N}
    return {"tau": tau, "p": p, "n": n, "source": source, "seed": seed}


def table(pol: dict) -> str:
    lines = ["bucket  n     p      +-1.96se  " + "  ".join(f"{lv:<6}" for lv in LEVELS)]
    for key, n in pol["n"].items():
        k, dbl = map(int, key.split(","))
        problem = {"steps": k, "ops": [("dbl", None)] if dbl else []}
        p = pol["p"].get(key)
        acc = f"{p:.3f}  +-{1.96 * (p * (1 - p) / n) ** 0.5:.3f}" if p is not None else "-      (n too small)"
        decisions = "  ".join(f"{'think' if sm.think(lv, problem, pol) else 'direct':<6}" for lv in LEVELS)
        lines.append(f"{key:<7} {n:<5} {acc:<19}{decisions}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Direct accuracy of the probe per (steps, doubling) bucket.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--config", help="TOML config (default: <run>/config.toml)")
    ap.add_argument("--n", type=int, default=300, help="problems per number of events")
    ap.add_argument("--steps", type=int, nargs="+", help="numbers of events (default: [data.reason].steps)")
    ap.add_argument("--out", help="default: [data.reason].policy")
    ap.add_argument("--device", default="auto", choices=DEVICES)
    args = ap.parse_args(argv)
    reason = load_config(args)["data"]["reason"]
    device = resolve_device(args.device, generation=True)
    setup(reason["policy_seed"], device)
    model, tok, _ = load_checkpoint(Path(args.run) / args.stage, device=device)
    results = measure(model, tok, reason["policy_seed"], args.steps or reason["steps"], args.n)
    pol = policy(results, reason["tau"], f"{args.run}/{args.stage}", reason["policy_seed"])
    out = Path(args.out or reason["policy"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pol, indent=1))
    print(table(pol))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
