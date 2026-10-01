"""Stage 2: midtraining. The base model only knows raw text; here it learns the chat
format and the chat skills: special tokens, turns, the <|think_start|> scratchpad,
calculator calls and tool results, stories on request, greetings. Only single-turn
questions and answers, no system prompt: behavior is SFT's job (see sft.py).

The data is packed exactly like pretraining, and the loss is on every token: this is
still plain language modeling, just on a new kind of document, and a lot of it. Most
documents are rendered conversations; a fraction (text_frac) is pretraining text so
the model does not forget how to write stories.

    uv run python -m minilab.train.midtrain --run runs/small
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Iterator

from minilab.checkpoint import load_checkpoint
from minilab.data.conversations import StoryPool, midtrain_stream
from minilab.data.loader import packed_batches, pretrain_documents
from minilab.data.tinystories import load_stories
from minilab.tokenizer.chat import render_conversation
from minilab.train.trainer import Logger, evaluate_loss, load_config, parse_args, save_stage, setup, train_loop


def main() -> None:
    args = parse_args("Stage 2: midtrain on chat-formatted conversations.")
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    d, sc = cfg["data"], cfg["midtrain"]
    model, tok, prev = load_checkpoint(run / "pretrain", device=device)

    stories = load_stories("train", d["train_mb"])
    pool = StoryPool(stories, d.get("story_max_chars", 700))

    def documents(seed: int, text_frac: float) -> Iterator[list[int]]:
        rng = random.Random(seed)
        convs = midtrain_stream(seed, sc, d["digits"], pool)
        text = pretrain_documents(tok, stories, d["digits"], cfg["pretrain"]["arith_frac"], seed,
                                  cfg["pretrain"].get("code_frac", 0.0))  # mini-4: Python too
        while True:
            if rng.random() < text_frac:
                yield next(text)
            else:
                conv = next(convs)
                yield render_conversation(tok, conv["messages"], conv["tools"])[0]

    T, B = model.config.block_size, sc["batch_size"]
    batches = packed_batches(documents(seed + 1, sc["text_frac"]), B, T)
    val_batches = list(itertools.islice(packed_batches(documents(seed + 1001, 0.0), B, T), sc.get("val_batches", 10)))

    def val_fn() -> dict:
        return {"val_loss": evaluate_loss(model, val_batches, device)}

    log = Logger(run / "midtrain" / "log.jsonl")
    stats = train_loop(model, batches, sc, log, device, val_fn, cfg.get("optimizer", "adamw"))
    save_stage(run, "midtrain", model, tok, stats, cfg, device, prev)


if __name__ == "__main__":
    main()
