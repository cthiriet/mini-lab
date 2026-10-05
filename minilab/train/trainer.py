"""Shared machinery for every training stage: config, device, optimizer, learning-rate
schedule, the training loop, JSONL logs and checkpoints.

Each stage (pretrain, midtrain, sft, rl_math, distill) is a short script that builds
its data and calls into this module. A run directory looks like:

    runs/<run>/
        config.toml                          copy of the config (written by the tokenizer step)
        tokenizer.json
        pretrain/  model.pt config.json tokenizer.json log.jsonl eval.json
        midtrain/  ...
        sft/       ...
        rl_math/   ...                       the math specialist (train/rl.py --stage rl_math)
        distill/   ...
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

# What the speedrun runs, in order (minilab.train.<stage>).
STAGES = ["pretrain", "midtrain", "sft", "rl_math", "distill"]
DEVICES = ["auto", "cpu", "mps", "cuda"]


# ---- setup --------------------------------------------------------------------

def parse_args(description: str, generation: bool = False, **extra: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run", required=True, help="run directory, e.g. runs/prelude")
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

def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iterations: roughly the closest (semi-)orthogonal matrix to g, i.e.
    g = U S V^T with every singular value in S pushed toward 1. Keller Jordan's
    coefficients. bf16 on GPUs, like torch.optim.Muon; float32 on the CPU, where bf16
    matmuls are ~1000x slower."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.float32 if g.device.type == "cpu" else torch.bfloat16)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)  # a stack of matrices (MoE experts): each its own
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    for _ in range(steps):
        A = x @ x.mT
        x = a * x + (b * A + c * A @ A) @ x
    return (x.mT if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan, 2024): SGD with Nesterov momentum, where each matrix's update
    is replaced by its orthogonalization, so every direction of the matrix moves at the
    same speed. The update is then scaled to the size of an AdamW update (Moonshot's
    0.2 x sqrt(max dim)), so it takes AdamW's learning rate and weight decay."""

    def __init__(self, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.95):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "momentum": momentum})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, wd, beta = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - beta)
                update = orthogonalize(p.grad.lerp(buf, beta))  # Nesterov
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * 0.2 * max(p.shape[-2:]) ** 0.5)


class Optimizers:
    """Several optimizers stepped as one (Muon for the matrices, AdamW for the rest)."""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.optimizers:
            o.step()


def make_optimizer(model: GPT, lr: float, weight_decay: float, kind: str = "adamw"):
    """AdamW; weight decay only on matrices (embeddings, linears), not on norm gains.

    kind="muon": the hidden matrices (attention, MLP) get Muon instead, like Kimi,
    GLM-5 and DeepSeek-V4, with the same lr and weight decay. The embedding (tied to
    the output head) and the norm gains are not hidden matrices, and stay on AdamW."""
    params = [p for p in model.parameters() if p.requires_grad]
    matrices = [p for p in params if p.dim() >= 2]
    vectors = [p for p in params if p.dim() < 2]
    if kind == "adamw":
        groups = [{"params": matrices, "weight_decay": weight_decay}, {"params": vectors, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    assert kind == "muon", f"unknown optimizer: {kind}"
    embeddings = [model.wte.weight] + ([] if model.lm_head.weight is model.wte.weight else [model.lm_head.weight])
    hidden = [p for p in matrices if all(p is not e for e in embeddings)]
    muon = Muon(hidden, lr=lr, weight_decay=weight_decay)
    adamw = torch.optim.AdamW([{"params": embeddings, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    return Optimizers(muon, adamw)


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


def fast_forward(model: GPT, device: str, compile: bool = False, precision: str = "fp32") -> Callable:
    """The model's forward for training steps. On a GPU (mps, cuda), `compile` runs it through
    torch.compile (fused kernels) and precision="bf16" computes it in bfloat16 (weights and
    optimizer stay float32): together 2.6x faster on MPS, same loss. On the CPU both are
    ignored (bf16 matmuls are slow there). New input shapes trigger one recompile, after which
    the lengths are dynamic."""
    gpu = device in ("mps", "cuda")
    fn = torch.compile(model) if compile and gpu else model
    mixed = precision == "bf16" and gpu

    def forward(*args):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=mixed):
            return fn(*args)
    return forward


def train_loop(model: GPT, batches: Iterator[tuple[torch.Tensor, torch.Tensor]], sc: dict, log: Logger,
               device: str, val_fn: Callable[[], dict] | None = None, optimizer: str = "adamw",
               compile: bool = False, precision: str = "fp32") -> dict:
    """The language-modeling loop shared by pretrain, midtrain and SFT.

    `sc` is the stage's config section: steps, lr, warmup, weight_decay, grad_clip,
    min_lr_frac, log_every, eval_every. `val_fn` returns a dict of metrics to log.
    `optimizer` is the config's top-level `optimizer` ("adamw" or "muon"); `compile` and
    `precision` (the config's top-level keys): see fast_forward.
    """
    steps, log_every = sc["steps"], sc.get("log_every", 10)
    opt = make_optimizer(model, sc["lr"], sc.get("weight_decay", 0.0), optimizer)
    forward = fast_forward(model, device, compile, precision)
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
        _, loss = forward(x.to(device), y.to(device))
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
            mem = {"mps_gb": round(torch.mps.driver_allocated_memory() / 2**30, 1)} if device == "mps" else {}
            log.log(step=step + 1, loss=train_loss, lr=lr, grad_norm=float(grad_norm),
                    tok_per_s=round(window_tokens / dt), elapsed=round(time.time() - t0, 1), **mem)
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
