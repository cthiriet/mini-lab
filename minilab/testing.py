"""Helpers for tests and local development without training a real model.

    uv run python -m minilab.testing models/   # creates models/mini-random (random weights)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from minilab.checkpoint import save_checkpoint
from minilab.model.gpt import GPT, GPTConfig
from minilab.registry import ModelInfo, Pricing, write_release
from minilab.tokenizer.bpe import Tokenizer

_SAMPLE_TEXT = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside with her dog.",
    "One day, Tom found a big red ball in the park. He was very happy and shared it with his friends.",
    "What is 347 + 58? The answer is 405. What is 12 + 9? The answer is 21.",
]


def make_random_release(models_dir: str | Path, model_id: str = "mini-random", seed: int = 0) -> Path:
    """Write a tiny random-weight model release. Useful to exercise the serving stack."""
    torch.manual_seed(seed)
    tok = Tokenizer.train(_SAMPLE_TEXT * 20, vocab_size=512)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=256, n_layer=2, n_head=2, n_embd=64))
    out = Path(models_dir) / model_id
    save_checkpoint(out, model, tok, {"stage": "random"})
    write_release(out, ModelInfo(
        id=model_id, created=int(time.time()), description="Random weights, for testing the serving stack.",
        context_length=model.config.block_size, pricing=Pricing(input_per_1m=0.50, output_per_1m=1.50),
    ))
    return out


if __name__ == "__main__":
    print(make_random_release(sys.argv[1] if len(sys.argv) > 1 else "models"))
