"""The code specialist: GRPO on coding tasks, played end to end in the Docker sandbox (mini-4).

rl.py's GRPO, for an agent: an attempt is a whole episode (tool calls, their results, more calls,
the answer), played like the eval plays it, every tool call in the locked-down container
(data/sandbox.py). Each step:

1. sample coding tasks from the config's mix, weighted to the hard ones (adding a function to a
   file the model must copy exactly, a failing test whose module has a distractor, a crash),
   some after earlier requests of the same session (played by the oracles, in a local sandbox:
   only our own code runs on this machine);
2. G attempts at each, at temperature 1;
3. reward 1 if the task is done (the eval's check) and the attempt didn't game it (the task's
   guard: tests left as they were, the fixed function right on unseen inputs, the answer grounded
   in a tool call). data/code.py's Task.reward;
4. advantage = reward normalized in its group; policy gradient on every token the model generated
   in the episode, each turn with exactly the prompt it was generated from (the server-side
   context fitting included).

    uv run python -m minilab.train.rl_code --run runs/unified --device mps
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.checkpoint import load_checkpoint
from minilab.data import code
from minilab.data.sandbox import OPENCODE_TOOLS, Container, DockerSandbox, Sandbox
from minilab.eval.code import Episode, play
from minilab.model.gpt import GPT
from minilab.train.trainer import Logger, load_config, lr_at, make_optimizer, parse_args, save_stage, setup

# Earlier requests of a session, played by the oracles: they read, run or add files, and leave the
# project's metadata (its functions, its tests) true for the last request.
HISTORY = ["run", "run_tests", "list_files", "show_file", "find_def", "explain", "create_func", "create_script"]
ROOT = "@ROOT@"


def make_task(rng: random.Random, kind: str, project: code.Project) -> code.Task | None:
    try:
        if kind == "fix_test":   # always with a distractor: the fix must go to the failing function
            return code.task_fix_test(rng, project, distractor_frac=1.0)
        return code.TASKS[kind](rng, project)
    except (StopIteration, KeyError, ValueError, IndexError):
        return None


def problem(rng: random.Random, sc: dict) -> tuple[list[dict], code.Task, dict[str, str]]:
    """(messages before the request, with ROOT for the project's path; the task; its files)."""
    kind = rng.choices(list(sc["mix"]), weights=list(sc["mix"].values()))[0]
    depth = rng.choice([1, 2, 3]) if rng.random() < sc.get("history_frac", 0.0) else 0
    task = None
    while task is None:   # a project where this kind of request applies, also after the history
        project = code.make_project(rng)
        if make_task(random.Random(rng.random()), kind, project) is None:
            continue
        history, task, files = _session(rng, kind, depth, project)
    return history, task, files


def _session(rng: random.Random, kind: str, depth: int, project: code.Project):
    task = None
    with Sandbox(project.files) as sb:
        messages = [code.system_message(str(sb.root), rng)]
        for _ in range(depth):
            current = dataclasses.replace(project, files=sb.files())
            earlier = next((t for k in rng.sample(HISTORY, len(HISTORY))
                            if (t := make_task(rng, k, current)) is not None and t.files == current.files), None)
            if earlier is None:
                break
            messages.append({"role": "user", "content": code.noisy(rng, earlier.prompt)})
            messages += code.play(earlier, sb, start=len(messages))
        current = sb.files()
        for _ in range(50):
            task = make_task(rng, kind, dataclasses.replace(project, files=current))
            if task is not None:
                break
        messages = code._relocate(messages, str(sb.root), ROOT)
    return messages, task, ({**current, **task.files} if task else None)


def reward(e: Episode) -> float:
    start = max(i for i, m in enumerate(e.messages) if m["role"] == "user")
    try:
        return float(e.task.reward(e.sb, e.answer or "", code.calls_of(e.messages[start:])))
    except Exception:   # a check on a project the attempt mangled
        return 0.0


def policy_loss(model: GPT, rows: list[tuple[list[int], list[int], float]], micro: int, total: int,
                device: str) -> float:
    """-advantage * log p(generated tokens), summed over rows and divided by `total` tokens; the
    gradients are accumulated `micro` rows at a time. Every chunk has exactly `micro` rows (empty
    ones at the end) padded to a multiple of 128 tokens: on MPS each new batch shape compiles and
    keeps new kernels, and a last chunk of any size made each step slower than the one before."""
    rows = sorted(rows, key=lambda r: len(r[0]) + len(r[1]))
    loss_sum = 0.0
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        T = min(model.config.block_size, -(-max(len(p) + len(c) - 1 for p, c, _ in chunk) // 128) * 128)
        x = torch.zeros(micro, T, dtype=torch.long)
        y = torch.full((micro, T), -1, dtype=torch.long)
        adv = torch.zeros(micro, 1)
        for j, (p, c, a) in enumerate(chunk):
            seq = p + c
            x[j, :len(seq) - 1] = torch.tensor(seq[:-1])
            y[j, len(p) - 1:len(seq) - 1] = torch.tensor(c)
            adv[j] = a
        x, y, adv = x.to(device), y.to(device), adv.to(device)
        logits, _ = model(x)
        mask = y != -1
        logp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0)[..., None]).squeeze(-1)
        loss = -(logp * adv * mask).sum() / total
        loss.backward()
        loss_sum += loss.item()
    return loss_sum


def main() -> None:
    args = parse_args("The code specialist: GRPO on coding tasks played in the Docker sandbox.")
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    sc = cfg["rl_code"]
    model, tok, prev = load_checkpoint(run / "sft", device=device)
    rng = random.Random(seed + 7)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    opt = make_optimizer(model, sc["lr"], sc.get("weight_decay", 0.0), cfg.get("optimizer", "adamw"))
    steps, G = sc["steps"], sc["group_size"]
    max_steps, max_new = sc.get("max_steps", 8), sc.get("max_new_tokens", 384)

    log = Logger(run / "rl_code" / "log.jsonl")
    t0, tokens = time.time(), 0
    with Container() as container:
        for step in range(steps):
            lr = lr_at(step, steps, sc["lr"], sc.get("warmup", 0), sc.get("min_lr_frac", 0.1))
            for group in opt.param_groups:
                group["lr"] = lr
            # 1-2) problems, and G attempts at each
            t_step = time.time()
            groups = []
            for _ in range(sc["prompts_per_step"]):
                history, task, files = problem(rng, sc)
                eps = []
                for _ in range(G):
                    sb = DockerSandbox(container, files)
                    msgs = json.loads(json.dumps(history).replace(ROOT, str(sb.root)))
                    eps.append(Episode(task, sb, msgs + [{"role": "user", "content": task.prompt}], OPENCODE_TOOLS))
                groups.append(eps)
            model.eval()
            play(model, tok, [e for g in groups for e in g], max_steps, max_new, sc.get("batch", 128),
                 temperature=sc.get("temperature", 1.0), generator=gen)
            t_play = time.time() - t_step
            # 3) rewards, group-normalized advantages
            rewards = [[reward(e) for e in g] for g in groups]
            for g in groups:
                for e in g:
                    e.sb.close()
            rows, by_kind = [], defaultdict(list)
            for g, rs in zip(groups, rewards):
                r = torch.tensor(rs)
                by_kind[g[0].task.kind] += rs
                if r.std() == 0:
                    continue   # every attempt right, or every one wrong: no signal
                adv = (r - r.mean()) / (r.std() + 1e-4)
                for e, a in zip(g, adv.tolist()):
                    rows += [(p, c, a) for p, c in e.turns if c]
            stats = {"reward": sum(map(sum, rewards)) / (len(groups) * G),
                     "informative": sum(torch.tensor(rs).std() > 0 for rs in rewards).item() / len(groups),
                     "turns": sum(len(e.turns) for g in groups for e in g) / (len(groups) * G)}
            # 4) policy gradient on the generated tokens of the informative groups
            loss = 0.0
            if rows:
                model.train()
                total = sum(len(c) for _, c, _ in rows)
                opt.zero_grad(set_to_none=True)
                loss = policy_loss(model, rows, sc.get("micro_batch", 16), total, device)
                torch.nn.utils.clip_grad_norm_(model.parameters(), sc.get("grad_clip", 1.0))
                opt.step()
                tokens += total
            if (step + 1) % sc.get("log_every", 1) == 0 or step == steps - 1:
                mem = {"mps_gb": round(torch.mps.driver_allocated_memory() / 2**30, 1)} if device == "mps" else {}
                log.log(step=step + 1, **stats, loss=loss, lr=lr, play_s=round(t_play, 1), **mem,
                        step_s=round(time.time() - t_step, 1), elapsed=round(time.time() - t0, 1),
                        **{f"reward_{k}": round(sum(v) / len(v), 3) for k, v in sorted(by_kind.items())})
            if (step + 1) % sc.get("save_every", 10**9) == 0 and step < steps - 1:   # a long run can be stopped
                save_stage(run, "rl_code", model, tok, {"steps": step + 1, "tokens": tokens,
                                                        "wall_clock_s": round(time.time() - t0, 1)}, cfg, device, prev)

    stats = {"steps": steps, "tokens": tokens, "wall_clock_s": round(time.time() - t0, 1)}
    save_stage(run, "rl_code", model, tok, stats, cfg, device, prev)


if __name__ == "__main__":
    main()
