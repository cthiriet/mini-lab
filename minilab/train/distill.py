"""Stage 5, the 2026 recipe: on-policy distillation from specialists (instead of `rl`).

One RL run on every skill at once has to keep the skills it isn't practicing from
drifting: rl.py does it by putting them all in the problem mix. Labs now split the
work instead (DeepSeek-V4, Kimi K3, Nemotron 3): specialists are trained by RL, each
on its own domain and with nothing to hold it elsewhere, and then one student learns
from all of them, each on its own domain. Here there are two teachers:

- the math specialist (the `rl_math` stage: rl.py on additions and the calculator only);
- the SFT model, for everything else (system prompts, follow-ups, refusals...): it
  already does them right.

The student starts from the SFT model. Each step:

1. sample problems from the mix; the student answers each once, at temperature 1.
   On-policy: it learns on its own answers, mistakes included, like in RL;
2. the problem's teacher reads the same tokens, and gives its next-token
   distribution at every position of the answer;
3. loss = reverse KL(student || teacher), over the whole vocabulary, averaged over
   the answer's positions. Every token gets a grade, where RL gives one reward per
   answer.

    uv run python -m minilab.train.rl --run runs/small --stage rl_math
    uv run python -m minilab.train.distill --run runs/small

mini-4 (`world = "unified"`) also codes: its mix has "code" problems, one turn of an agent
transcript each (a tool call or the answer, with the earlier calls and results as context), whose
teacher is the SFT model. Without them, the coding agent would drift like any skill left out.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.checkpoint import load_checkpoint
from minilab.data import code
from minilab.eval.tasks import grade
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import PromptTooLong, render_prompt
from minilab.train.rl import sample_problem
from minilab.train.trainer import Logger, load_config, lr_at, make_optimizer, parse_args, save_stage, setup


def distill_step(student: GPT, teachers: list[GPT], tok: Tokenizer, opt: torch.optim.Optimizer,
                 problems: list[tuple[dict, int]], max_new_tokens: int, temperature: float,
                 generator: torch.Generator, grad_clip: float = 1.0) -> dict:
    """One step on a batch of (problem, index of its teacher in `teachers`)."""
    device = student.wte.weight.device
    stop = {tok.special("<|assistant_end|>")}

    # 1) the student's own answers (an agent's long transcript is fitted like the server fits it)
    budget = student.config.block_size - max_new_tokens if tok.chat_template == "code" else None
    rendered = []
    for p, t in problems:
        try:
            rendered.append((p, t, render_prompt(tok, p["messages"], p["tools"], budget=budget)))
        except PromptTooLong:
            continue
    rendered = [r for r in rendered if len(r[2]) < student.config.block_size]
    student.eval()
    completions = student.generate([ids for _, _, ids in rendered], max_new_tokens,
                                   temperature=temperature, stop_ids=stop, generator=generator)

    # 2) every sequence, with a mask on the positions that predict the answer's tokens
    seqs = [ids + c for (_, _, ids), c in zip(rendered, completions)]
    T = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)
    for row, ((_, _, ids), s) in enumerate(zip(rendered, seqs)):
        x[row, :len(s) - 1] = torch.tensor(s[:-1])
        mask[row, len(ids) - 1:len(s) - 1] = True
    x, mask = x.to(device), mask.to(device)
    teacher_logp = torch.empty(len(seqs), T, student.config.vocab_size, device=device)
    with torch.no_grad():
        for t, teacher in enumerate(teachers):
            rows = torch.tensor([row for row, (_, i, _) in enumerate(rendered) if i == t], dtype=torch.long)
            if len(rows):
                teacher_logp[rows] = F.log_softmax(teacher(x[rows])[0].float(), dim=-1)

    # 3) reverse KL, at every position of every answer
    student.train()
    logits, _ = student(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    kl = (logp.exp() * (logp - teacher_logp)).sum(-1)  # (B, T)
    loss = (kl * mask).sum() / mask.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    opt.step()

    # the reward RL would have given, for comparison (not trained on; an agent's turn has none)
    rewards = [float(grade(tok, c, p)) for (p, _, _), c in zip(rendered, completions) if p["kind"] != "code"]
    return {"kl": loss.item(), "reward": sum(rewards) / max(1, len(rewards)),
            "completion_len": sum(map(len, completions)) / len(completions), "tokens": int(mask.sum())}


def main() -> None:
    args = parse_args("Stage 5: on-policy distillation from specialists.", generation=True)
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    sc = cfg["distill"]
    student, tok, prev = load_checkpoint(run / "sft", device=device)

    # the teachers: one checkpoint per stage named in `teachers`, the SFT model by default
    names = ["sft", *sorted(set(sc["teachers"].values()) - {"sft"})]
    teachers = [load_checkpoint(run / name, device=device)[0] for name in names]
    teacher_of = {kind: names.index(name) for kind, name in sc["teachers"].items()}
    print("teachers: " + ", ".join(f"{k} -> {v}" for k, v in sc["teachers"].items()) + ", everything else -> sft")

    # mini-4: agent transcripts to cut into one-turn problems, apart from the SFT set's
    code_pool = code.sft_set(cfg, seed + 5, size=sc["code_pool"]) if "code" in sc["mix"] else None
    rng = random.Random(seed + 5)
    gen = torch.Generator(device=device).manual_seed(seed + 5)
    opt = make_optimizer(student, sc["lr"], sc.get("weight_decay", 0.0), cfg.get("optimizer", "adamw"))
    steps = sc["steps"]

    log = Logger(run / "distill" / "log.jsonl")
    t0, tokens = time.time(), 0
    for step in range(steps):
        lr = lr_at(step, steps, sc["lr"], sc.get("warmup", 0), sc.get("min_lr_frac", 0.1))
        for group in opt.param_groups:
            group["lr"] = lr
        problems = [sample_problem(rng, sc, code_pool) for _ in range(sc["prompts_per_step"])]
        stats = distill_step(student, teachers, tok, opt, [(p, teacher_of.get(p["kind"], 0)) for p in problems],
                             sc["max_new_tokens"], sc.get("temperature", 1.0), gen, sc.get("grad_clip", 1.0))
        tokens += stats.pop("tokens")
        if (step + 1) % sc.get("log_every", 1) == 0 or step == steps - 1:
            log.log(step=step + 1, **stats, lr=lr, elapsed=round(time.time() - t0, 1))

    stats = {"steps": steps, "tokens": tokens, "wall_clock_s": round(time.time() - t0, 1)}
    save_stage(run, "distill", student, tok, stats, cfg, device, prev)


if __name__ == "__main__":
    main()
