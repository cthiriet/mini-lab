"""Hard requests, after a history the model made itself. The model plays `depth` requests (reading,
running, creating files) greedily in the Docker sandbox, then a hard last one: a failing test whose
module always has a distractor, a crash, adding a function, a rename, a constant. From that same state:
the greedy answer, and K attempts at T=1. Headroom (greedy < 100%) and signal (some attempts right).

    uv run python scripts/rl_code/probe_hard.py runs/unified/sft --depths 0,2,3 --n 30 --k 8
"""
import argparse
import dataclasses
import json
import random
import time
from collections import defaultdict

import torch

from minilab.checkpoint import load_checkpoint
from minilab.data import code
from minilab.data.sandbox import OPENCODE_TOOLS, Container, DockerSandbox
from minilab.eval.code import Episode, play

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--depths", default="0,2,3")
p.add_argument("--n", type=int, default=30)
p.add_argument("--k", type=int, default=8)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--out", default="runs/exp/rl_code/probe-hard.json")
args = p.parse_args()

HISTORY = ["run", "run_tests", "list_files", "show_file", "find_def", "explain", "create_func", "create_script"]
HARD = ["fix_test", "fix_crash", "add_func", "rename", "change_const"]


def defs(text: str) -> int:
    return text.count("def ")


def hard_task(rng: random.Random, project, current: dict[str, str], kind: str):
    """A hard request of this kind on the project as it is; fix_test only with a distractor in the module."""
    p = dataclasses.replace(project, files=current)
    for _ in range(200):
        try:
            t = code.TASKS[kind](rng, p)
        except (StopIteration, KeyError, ValueError, IndexError):
            continue
        if t is None:
            continue
        if kind == "fix_test" and not any(defs(t.files[f]) > defs(current.get(f, "")) for f in t.files):
            continue
        return t
    raise RuntimeError("no hard task")


def history_task(rng: random.Random, project, current: dict[str, str]):
    p = dataclasses.replace(project, files=current)
    for kind in rng.sample(HISTORY, len(HISTORY)):
        try:
            t = code.TASKS[kind](rng, p)
        except (StopIteration, KeyError, ValueError, IndexError):
            continue
        if t is not None and t.files == current:
            return t
    return None


def last_turn_calls(messages):
    start = next(i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user")
    return code.calls_of(messages[start:])


model, tok, _ = load_checkpoint(args.model, device="mps")
gen = torch.Generator(device="mps").manual_seed(0)
depths = [int(d) for d in args.depths.split(",")]
t0 = time.time()
with Container() as container:
    # 1. histories, played by the model (greedy), all sessions in lockstep
    sessions = []
    for depth in depths:
        rng = random.Random(f"probe-hard-{depth}")
        for i in range(args.n):
            kind = HARD[i % len(HARD)]
            while True:   # a project where this kind of request applies
                project = code.make_project(rng)
                try:
                    hard_task(random.Random(0), project, project.files, kind)
                    break
                except RuntimeError:
                    continue
            sb = DockerSandbox(container, project.files)
            e = Episode(None, sb, [code.system_message(str(sb.root), rng)], OPENCODE_TOOLS)
            sessions.append({"depth": depth, "rng": rng, "project": project, "episode": e, "history_ok": [], "kind": kind})
    for step in range(max(depths)):
        live = [s for s in sessions if s["depth"] > step]
        for s in live:
            task = history_task(s["rng"], s["project"], s["episode"].sb.files())
            s["task"] = task
            e = s["episode"]
            e.task, e.done, e.answer = task, False, None
            e.messages.append({"role": "user", "content": task.prompt})
        play(model, tok, [s["episode"] for s in live], 8, 384, args.batch)
        for s in live:
            e = s["episode"]
            try:
                s["history_ok"].append(s["task"].reward(e.sb, e.answer or "", last_turn_calls(e.messages)))
            except Exception:
                s["history_ok"].append(False)
    # 2. the hard request, from the same state: greedy + K sampled
    groups = []
    for s in sessions:
        e = s["episode"]
        current = e.sb.files()
        task = hard_task(s["rng"], s["project"], current, s["kind"])
        files = {**current, **task.files}
        eps = []
        for _ in range(args.k + 1):
            sb = DockerSandbox(container, files)
            msgs = json.loads(json.dumps(e.messages).replace(str(e.sb.root), str(sb.root)))
            eps.append(Episode(task, sb, msgs + [{"role": "user", "content": task.prompt}], OPENCODE_TOOLS))
        e.sb.close()
        groups.append((s, task, eps[0], eps[1:]))
    play(model, tok, [g for _, _, g, _ in groups], 8, 384, args.batch)
    play(model, tok, [x for *_, xs in groups for x in xs], 8, 384, args.batch, temperature=1.0, generator=gen)

    def reward(x):
        try:
            return x.task.reward(x.sb, x.answer or "", last_turn_calls(x.messages))
        except Exception:
            return False

    rows = defaultdict(lambda: defaultdict(list))
    by_kind = defaultdict(lambda: defaultdict(list))
    fails = []
    for s, task, g, xs in groups:
        gr, rs = reward(g), [reward(x) for x in xs]
        for key, r in ((s["depth"], rows[s["depth"]]), (task.kind, by_kind[task.kind])):
            r["greedy"].append(gr)
            r["sampled"] += rs
            r["pass@k"].append(any(rs))
            r["informative"].append(0 < sum(rs) < args.k)
        rows[s["depth"]]["history"] += s["history_ok"]
        if not gr:
            fails.append({"depth": s["depth"], "kind": task.kind, "prompt": task.prompt, "transcript": g.transcript})
        for x in [g, *xs]:
            x.sb.close()

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{len(groups)} sessions, {time.time() - t0:.0f}s")
print(f"{'own earlier requests':20} {'history ok':>10} {'greedy':>7} {'T=1':>6} {'pass@k':>7} {'informative':>11}")
for d in depths:
    r = rows[d]
    print(f"{d:20} {mean(r['history']):10.0%} {mean(r['greedy']):7.0%} {mean(r['sampled']):6.0%} {mean(r['pass@k']):7.0%} {mean(r['informative']):11.0%}")
print(f"{'last request':20} {'n':>10} {'greedy':>7} {'T=1':>6} {'pass@k':>7} {'informative':>11}")
for k in HARD:
    r = by_kind[k]
    print(f"{k:20} {len(r['greedy']):10} {mean(r['greedy']):7.0%} {mean(r['sampled']):6.0%} {mean(r['pass@k']):7.0%} {mean(r['informative']):11.0%}")
json.dump({"args": vars(args), "fails": fails}, open(args.out, "w"), indent=1)
