"""Helpers for tests and local development without training a real model.

    uv run python -m minilab.testing models/            # creates models/mini-random (random weights)
    uv run python -m minilab.testing key [--out FILE]   # an API key in the local database
"""

from __future__ import annotations

import secrets
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
        context_length=model.config.block_size, pricing=Pricing(input_per_1m=10.0, output_per_1m=50.0),
    ))
    return out


def local_api_key(credit_usd: float = 20.0) -> str:
    """A user, an org with credits and an API key in the local database (MINILAB_DB), to call the
    API without going through the platform: the opencode demo (examples/opencode). Returns the key."""
    from minilab.db import store as db
    db.init_db()
    user = db.create_user(f"local-{int(time.time() * 1000)}@localhost", secrets.token_urlsafe(16), "Local")
    org = db.create_org("Local", user["id"], signup_credit_usd=credit_usd)
    project = db.list_projects(org["id"])[0]
    return db.create_api_key(org["id"], project["id"], "local", created_by=user["id"])[1]


if __name__ == "__main__":
    if sys.argv[1:2] == ["key"]:
        key = local_api_key()
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(key)
            print(f"API key written to {out}")
        else:
            print(key)
    else:
        print(make_random_release(sys.argv[1] if len(sys.argv) > 1 else "models"))
