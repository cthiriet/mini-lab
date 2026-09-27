"""Stage 1: pretraining. A randomly initialized GPT learns to predict the next token of
TinyStories stories, with arithmetic worksheets (equations, word problems and worked
column additions) mixed in. This is where almost all of the compute goes: the model
learns English, what a story looks like, and the mechanics of addition.

    uv run python -m minilab.train.pretrain --run runs/small
"""

from __future__ import annotations

from pathlib import Path

import torch

from minilab.data.loader import packed_batches, pretrain_documents, story_batches
from minilab.data.tinystories import load_stories
from minilab.model.gpt import GPT, GPTConfig
from minilab.tokenizer.bpe import Tokenizer
from minilab.train.trainer import Logger, evaluate_loss, load_config, parse_args, save_stage, setup, train_loop


def sample_story(model: GPT, tok: Tokenizer, device: str, prompt: str = "Once upon a time", n: int = 80) -> str:
    gen = torch.Generator(device=device).manual_seed(0)
    ids = [tok.bos_id, *tok.encode(prompt)]
    out = model.generate([ids], n, temperature=0.8, top_k=50, stop_ids={tok.bos_id}, generator=gen)[0]
    return prompt + tok.decode(t for t in out if t != tok.bos_id)


def main() -> None:
    args = parse_args("Stage 1: pretrain on TinyStories + arithmetic text.")
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    d, sc = cfg["data"], cfg["pretrain"]

    tok = Tokenizer.load(run / "tokenizer.json")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, **cfg["model"])).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters, config {model.config}")

    T, B = model.config.block_size, sc["batch_size"]
    stories = load_stories("train", d["train_mb"])
    batches = packed_batches(pretrain_documents(tok, stories, d["digits"], sc["arith_frac"], seed), B, T)
    val_batches = story_batches(tok, load_stories("val", d["val_mb"]), B, T, sc.get("val_batches", 10))

    def val_fn() -> dict:
        return {"val_loss": evaluate_loss(model, val_batches, device), "sample": sample_story(model, tok, device)}

    log = Logger(run / "pretrain" / "log.jsonl")
    stats = train_loop(model, batches, sc, log, device, val_fn, cfg.get("optimizer", "adamw"))
    save_stage(run, "pretrain", model, tok, stats, cfg, device)


if __name__ == "__main__":
    main()
