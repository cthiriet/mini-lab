"""Shared machinery for every training stage: config, device, optimizer, learning-rate
schedule, the training loop, JSONL logs and checkpoints.

Each stage (pretrain, midtrain, sft, rl) is a short script that builds its data and
calls into this module. A run directory looks like:

    runs/<run>/
        config.toml                          copy of the config (written by the tokenizer step)
        tokenizer.json
        pretrain/  model.pt config.json tokenizer.json log.jsonl eval.json
        midtrain/  ...
        sft/       ...
        rl/        ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Callable, Iterator

import torch

from minilab.checkpoint import save_checkpoint
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer

STAGES = ["pretrain", "midtrain", "sft", "rl"]
DEVICES = ["auto", "cpu", "mps", "cuda"]


# ---- setup --------------------------------------------------------------------

def parse_args(description: str, generation: bool = False, **extra: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run", required=True, help="run directory, e.g. runs/small")
    p.add_argument("--config", help="TOML config (default: <run>/config.toml)")
    p.add_argument("--device", default="auto", choices=DEVICES, help="auto: cuda, else mps, else cpu")
    for name, kwargs in extra.items():
        p.add_argument(f"--{name}", **kwargs)
    args = p.parse_args()
    args.device = resolve_device(args.device, generation)
    return args


def resolve_device(name: str, generation: bool = False) -> str:
    """"auto" means cuda if available, else mps, else cpu -- except for generation-heavy
    work (RL, eval) on a Mac: sampling a tiny model token by token is dominated by
    kernel-launch overhead, and there the CPU is ~4x faster than MPS."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() and not generation else "cpu"


def load_config(args: argparse.Namespace) -> dict:
    path = Path(args.config or Path(args.run) / "config.toml")
    with path.open("rb") as f:
        return tomllib.load(f)


def setup(seed: int, device: str) -> None:
    """Seed everything and pick a sensible number of CPU threads."""
    random.seed(seed)
    torch.manual_seed(seed)
    # Small matmuls stop scaling beyond ~a dozen threads; more just adds contention.
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 12)))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True


def hardware(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()
    name = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or name
        elif Path("/proc/cpuinfo").exists():
            line = next(l for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
            name = line.split(":", 1)[1].strip()
    except (OSError, StopIteration):
        pass
    return f"{name}, {os.cpu_count()} cores"


class Logger:
    """Prints progress and appends one JSON object per record to log.jsonl."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w")

    def log(self, **record) -> None:
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        parts = []
        for k, v in record.items():
            if isinstance(v, float):
                v = f"{v:.3e}" if k == "lr" else f"{v:.4g}"
            elif isinstance(v, str):
                v = repr(v[:300])
            parts.append(f"{k} {v}")
        print(" | ".join(parts), flush=True)


# ---- optimization -------------------------------------------------------------

def make_optimizer(model: GPT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW; weight decay only on matrices (embeddings, linears), not on norm gains."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def lr_at(step: int, steps: int, lr: float, warmup: int, min_lr_frac: float = 0.1) -> float:
    """Linear warmup, then cosine decay down to min_lr_frac * lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate_loss(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    """Mean loss over a fixed list of batches (weighted by the number of targets)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        _, loss = model(x.to(device), y.to(device))
        n = int((y != -1).sum())
        total, count = total + loss.item() * n, count + n
    model.train(was_training)
    return total / max(1, count)


def train_loop(model: GPT, batches: Iterator[tuple[torch.Tensor, torch.Tensor]], sc: dict, log: Logger,
               device: str, val_fn: Callable[[], dict] | None = None) -> dict:
    """The language-modeling loop shared by pretrain, midtrain and SFT.

    `sc` is the stage's config section: steps, lr, warmup, weight_decay, grad_clip,
    min_lr_frac, log_every, eval_every. `val_fn` returns a dict of metrics to log.
    """
    steps, log_every = sc["steps"], sc.get("log_every", 10)
    opt = make_optimizer(model, sc["lr"], sc.get("weight_decay", 0.0))
    model.train()
    t0 = time.time()
    tokens = window_tokens = 0
    window_loss, window_t0 = [], time.time()
    val, train_loss = {}, float("nan")
    for step in range(steps):
        lr = lr_at(step, steps, sc["lr"], sc.get("warmup", 0), sc.get("min_lr_frac", 0.1))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = next(batches)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), sc.get("grad_clip", 1.0))
        opt.step()

        tokens += x.numel()
        window_tokens += x.numel()
        window_loss.append(loss.detach())  # no .item() here: it would sync with the GPU every step
        last = step == steps - 1
        if (step + 1) % log_every == 0 or last:
            dt = time.time() - window_t0
            train_loss = torch.stack(window_loss).mean().item()
            log.log(step=step + 1, loss=train_loss, lr=lr, grad_norm=float(grad_norm),
                    tok_per_s=round(window_tokens / dt), elapsed=round(time.time() - t0, 1))
            window_loss, window_tokens, window_t0 = [], 0, time.time()
        if val_fn and ((step + 1) % sc.get("eval_every", 10**9) == 0 or last):
            model.eval()
            val = val_fn()
            model.train()
            log.log(step=step + 1, **val)
    return {"steps": steps, "tokens": tokens, "wall_clock_s": round(time.time() - t0, 1),
            "final_loss": round(train_loss, 4),
            **{k: v for k, v in val.items() if isinstance(v, (int, float))}}


# ---- checkpoints --------------------------------------------------------------

def save_stage(run: Path, stage: str, model: GPT, tok: Tokenizer, stats: dict, cfg: dict,
               device: str, prev_meta: dict | None = None) -> Path:
    """Save the stage's checkpoint with training stats in its meta (read by the model card)."""
    meta = {
        "stage": stage,
        **stats,
        "tokens_total": (prev_meta or {}).get("tokens_total", 0) + stats.get("tokens", 0),
        "params": model.num_params(),
        "device": device,
        "hardware": hardware(device),
        "config": cfg.get(stage, {}),
        "created": int(time.time()),
    }
    path = save_checkpoint(run / stage, model, tok, meta)
    print(f"saved {stage} checkpoint to {path} ({stats.get('wall_clock_s', 0):.0f}s)", flush=True)
    return path
