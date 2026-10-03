"""Is RL on code worth it? Sample K attempts per task at T=1 from a checkpoint, plus the greedy one,
every tool call in the Docker sandbox, and report per task kind: greedy success, sampled success
(check, and reward = check + guard), pass@K, groups GRPO could learn from (some attempts right,
some wrong), and attempts that pass the check but not the guard (what RL could hack).

    uv run python scripts/rl_code/probe.py runs/unified/sft --n 16 --k 8
"""
import argparse
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
p.add_argument("--n", type=int, default=16)
p.add_argument("--k", type=int, default=8)
p.add_argument("--kinds", default=",".join(code.TASKS))
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--out", default="runs/exp/rl_code/probe.json")
args = p.parse_args()

model, tok, _ = load_checkpoint(args.model, device="mps")
gen = torch.Generator(device="mps").manual_seed(0)
kinds = args.kinds.split(",")
t0 = time.time()
with Container() as container:
    groups = []   # (kind, task, greedy episode, sampled episodes)
    for kind in kinds:
        rng = random.Random(f"probe-{kind}")
        for _ in range(args.n):
            task = code.make_task(rng, kind)
            system = code.system_message("/home/user/project", rng)
            eps = []
            for _ in range(args.k + 1):
                sb = DockerSandbox(container, task.files)
                eps.append(Episode(task, sb, [{**system, "content": system["content"].replace("/home/user/project", str(sb.root))},
                                              {"role": "user", "content": task.prompt}], OPENCODE_TOOLS))
            groups.append((kind, task, eps[0], eps[1:]))
    greedy = [g for _, _, g, _ in groups]
    sampled = [e for *_, s in groups for e in s]
    t1 = time.time()
    play(model, tok, greedy, 8, 384, args.batch)
    t2 = time.time()
    play(model, tok, sampled, 8, 384, args.batch, temperature=args.temperature, generator=gen)
    t3 = time.time()

    def score(e):
        answer = e.answer or ""
        try:
            ok = bool(e.task.check(e.sb, answer))
            reward = ok and e.task.reward(e.sb, answer, code.calls_of(e.messages))
        except Exception:
            ok = reward = False
        return ok, reward

    rows = defaultdict(lambda: defaultdict(list))
    hacks = []
    for kind, task, g, s in groups:
        g_ok, g_r = score(g)
        scores = [score(e) for e in s]
        r = rows[kind]
        r["greedy"].append(g_r)
        r["check"] += [ok for ok, _ in scores]
        r["reward"] += [rw for _, rw in scores]
        n_right = sum(rw for _, rw in scores)
        r["pass@k"].append(n_right > 0)
        r["informative"].append(0 < n_right < args.k)
        r["all_right"].append(n_right == args.k)
        for e, (ok, rw) in zip(s, scores):
            if ok and not rw:
                hacks.append({"kind": kind, "prompt": task.prompt, "transcript": e.transcript, "answer": e.answer})
        for e in [g, *s]:
            e.sb.close()

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{len(groups)} tasks x ({args.k} sampled + 1 greedy): greedy {t2 - t1:.0f}s, sampled {t3 - t2:.0f}s "
      f"({(t3 - t2) / len(sampled) * 1000:.0f} ms/attempt), setup {t1 - t0:.0f}s")
print(f"{'kind':14} {'greedy':>7} {'T=1 check':>9} {'T=1 reward':>10} {'pass@k':>7} {'informative':>11} {'hack':>6}")
summary = {}
for kind in kinds:
    r = rows[kind]
    hack = mean([c and not w for c, w in zip(r["check"], r["reward"])])
    summary[kind] = {k: mean(v) for k, v in r.items()} | {"hack": hack}
    print(f"{kind:14} {mean(r['greedy']):7.0%} {mean(r['check']):9.0%} {mean(r['reward']):10.0%} "
          f"{mean(r['pass@k']):7.0%} {mean(r['informative']):11.0%} {hack:6.1%}")
allr = {k: mean([x for kind in kinds for x in rows[kind][k]]) for k in ("greedy", "check", "reward", "pass@k", "informative")}
print(f"{'all':14} {allr['greedy']:7.0%} {allr['check']:9.0%} {allr['reward']:10.0%} {allr['pass@k']:7.0%} {allr['informative']:11.0%}")
json.dump({"args": vars(args), "summary": summary, "hacks": hacks[:40]}, open(args.out, "w"), indent=1)
