"""Stage 4: the math specialist, trained by reinforcement learning (a minimal GRPO).

SFT imitates perfect demonstrations; RL lets the model practice. The specialist only
practices additions (and the calculator); distillation then merges it with the SFT model
(minilab.train.distill). Each step:

1. sample a batch of problems (mostly additions), and a *group* of completions per
   problem with model.generate at temperature 1 (the model's own attempts);
2. reward = 1 if the turn is properly ended and does what was asked, else 0. The
   reward is the eval's grader, and it has to be strict: RL is very good at finding
   loopholes (see minilab.eval.tasks.grade);
3. advantage = reward normalized within its group: (r - mean) / (std + eps). Attempts
   that beat their siblings are pushed up, the others down. A group where every
   attempt got the same reward carries no signal and is skipped;
4. policy-gradient loss: -advantage * log p(token), averaged over all completion tokens.

Like nanochat's simplified GRPO: fully on-policy (one update per batch of samples),
so there is no ratio clipping, and no KL penalty to a reference model. What keeps
the specialist's math skills from drifting is the problem mix: they are all in it (see
sample_problem), so as soon as one degrades it fails and gets a signal.

What RL teaches here: chat data only demonstrates 1-3 digit additions, pretraining
text has worked examples up to 5 digits. Asked for 4 digits, the SFT model copies the
numbers wrong -- but not always. RL finds and reinforces those lucky attempts.

    uv run python -m minilab.train.rl_math --run runs/prelude
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.checkpoint import load_checkpoint
from minilab.data import code
from minilab.eval.tasks import grade, make_problem
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_prompt
from minilab.train.trainer import Logger, load_config, lr_at, make_optimizer, parse_args, save_stage, setup


def reward(tok: Tokenizer, completion: list[int], problem: dict) -> float:
    """1 if the assistant turn is ended and does what was asked. This is the eval's grader:
    a reward that checks less gets hacked (see minilab.eval.tasks.grade)."""
    return float(grade(tok, completion, problem))


def sample_problem(rng: random.Random, sc: dict, code_pool: list[dict] | None = None) -> dict:
    """A problem of a kind drawn from the config's mix, from the eval's own generators
    (minilab.eval.tasks.make_problem). "add" covers every length seen in pretraining:
    the lengths SFT demonstrated carry no signal, the longer ones are where RL learns.
    The other kinds (calculator, follow-ups, word problems) keep those skills from drifting:
    as soon as one degrades, it fails and gets a signal. "code" (distillation only): one turn
    of an agent transcript from `code_pool`."""
    kind = rng.choices(list(sc["mix"]), weights=list(sc["mix"].values()))[0]
    if kind == "code":
        return code.turn_problem(rng, rng.choice(code_pool))
    return make_problem(kind, rng, sc["digits"] if kind in ("add", "tool", "word", "new_question") else sc["chat_digits"])


def rl_step(model: GPT, tok: Tokenizer, opt: torch.optim.Optimizer, problems: list[dict], group_size: int,
            max_new_tokens: int, temperature: float, generator: torch.Generator, grad_clip: float = 1.0) -> dict:
    """One GRPO step on a batch of problems (dicts with "messages", "tools", "answer", ...)."""
    device = model.wte.weight.device
    stop = {tok.special("<|assistant_end|>")}

    # 1) rollouts: G attempts at each question (that fits the context)
    rendered = [(p, render_prompt(tok, p["messages"], p["tools"])) for p in problems]
    rendered = [(p, ids) for p, ids in rendered if len(ids) < model.config.block_size]
    problems, prompts = [p for p, _ in rendered], [ids for _, ids in rendered]
    P, G = len(problems), group_size
    if not problems:
        return {"reward": 0.0, "informative": 0.0, "completion_len": 0.0, "loss": 0.0, "tokens": 0}
    model.eval()
    completions = model.generate([p for p in prompts for _ in range(G)], max_new_tokens,
                                 temperature=temperature, stop_ids=stop, generator=generator)

    # 2) rewards and group-normalized advantages
    rewards = torch.tensor([reward(tok, c, problems[i // G]) for i, c in enumerate(completions)])
    r = rewards.view(P, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    keep = [i for i in range(P * G) if adv[i] != 0]
    stats = {"reward": rewards.mean().item(), "informative": len(keep) / (P * G),
             "completion_len": sum(map(len, completions)) / len(completions), "loss": 0.0, "tokens": 0}
    if not keep:
        return stats

    # 3) policy gradient on the completion tokens of the informative groups
    model.train()
    seqs = [prompts[i // G] + completions[i] for i in keep]
    T = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(keep), T, dtype=torch.long)
    y = torch.full((len(keep), T), -1, dtype=torch.long)  # targets only on completion tokens
    for row, (i, s) in enumerate(zip(keep, seqs)):
        x[row, :len(s) - 1] = torch.tensor(s[:-1])
        y[row, len(prompts[i // G]) - 1:len(s) - 1] = torch.tensor(completions[i])
    x, y = x.to(device), y.to(device)
    logits, _ = model(x)
    mask = y != -1
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0)[..., None]).squeeze(-1)
    loss = -(logp * adv[keep].to(device)[:, None] * mask).sum() / mask.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return {**stats, "loss": loss.item(), "tokens": int(mask.sum())}


def main() -> None:
    args = parse_args("Stage 4: the math specialist, GRPO-style reinforcement learning on addition.", generation=True)
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    sc = cfg["rl_math"]
    model, tok, prev = load_checkpoint(run / "sft", device=device)
    rng = random.Random(seed + 3)
    gen = torch.Generator(device=device).manual_seed(seed + 3)
    opt = make_optimizer(model, sc["lr"], sc.get("weight_decay", 0.0), cfg.get("optimizer", "adamw"))
    steps = sc["steps"]

    log = Logger(run / "rl_math" / "log.jsonl")
    t0, tokens = time.time(), 0
    for step in range(steps):
        lr = lr_at(step, steps, sc["lr"], sc.get("warmup", 0), sc.get("min_lr_frac", 0.1))
        for group in opt.param_groups:
            group["lr"] = lr
        problems = [sample_problem(rng, sc) for _ in range(sc["prompts_per_step"])]
        stats = rl_step(model, tok, opt, problems, sc["group_size"], sc["max_new_tokens"],
                        sc.get("temperature", 1.0), gen, sc.get("grad_clip", 1.0))
        tokens += stats.pop("tokens")
        if (step + 1) % sc.get("log_every", 1) == 0 or step == steps - 1:
            log.log(step=step + 1, **stats, lr=lr, elapsed=round(time.time() - t0, 1))

    stats = {"steps": steps, "tokens": tokens, "wall_clock_s": round(time.time() - t0, 1)}
    save_stage(run, "rl_math", model, tok, stats, cfg, device, prev)


if __name__ == "__main__":
    main()
