"""Stage 3: supervised fine-tuning (SFT). Midtraining taught the format and the skills;
SFT teaches behavior: obeying system prompts ("Answer with the number only."),
follow-up questions about an earlier answer, saying who it is, and politely refusing
what it can't do -- none of which midtraining ever shows (see data/conversations.py).

Like at a lab, the data is a small curated set (here: `size` generated conversations,
fixed for the run) seen for a few epochs, rather than a stream. Each row is one
conversation starting at position 0, padded, exactly like at inference, and the loss
is only on the assistant's tokens. Lower learning rate, few steps.

Each row is drawn from the chat set, from the agent transcripts (`code_frac` of the rows:
opencode's requests, tool calls played for real in a sandbox, and the answers, generated once
and cached in data/code/), or from the pretraining documents (`text_frac`): without them, this
long SFT made the model forget plain text (stories 0.59 -> 1.00 bits per character).

    uv run python -m minilab.train.sft --run runs/prelude
"""

from __future__ import annotations

from pathlib import Path

from minilab.checkpoint import load_checkpoint
from minilab.data import code
from minilab.data.conversations import StoryPool, sft_dataset
from minilab.data.loader import chat_batch, chat_batches, epochs, mixture, pretrain_documents
from minilab.data.tinystories import load_stories
from minilab.train.trainer import Logger, evaluate_loss, load_config, parse_args, save_stage, setup, train_loop


def main() -> None:
    args = parse_args("Stage 3: supervised fine-tuning on assistant turns.")
    cfg = load_config(args)
    run, device, seed = Path(args.run), args.device, cfg.get("seed", 0)
    setup(seed, device)
    d, sc = cfg["data"], cfg["sft"]
    model, tok, prev = load_checkpoint(run / "midtrain", device=device)
    stories = load_stories("train", d["train_mb"])
    pool = StoryPool(stories, d.get("story_max_chars", 700))
    T, B = model.config.block_size, sc["batch_size"]
    train = sft_dataset(seed + 2, sc["size"], sc, d["digits"], pool)
    val = sft_dataset(seed + 1002, B * sc.get("val_batches", 10), sc, d["digits"], pool)
    val_batches = [chat_batch(tok, val[i:i + B], T) for i in range(0, len(val), B)]
    agent = code.sft_set(cfg, seed + 2)
    code_set = code.sft_set(cfg, seed + 1002, size=B * sc.get("val_batches", 10))
    code_val = [chat_batch(tok, code_set[i:i + B], T) for i in range(0, len(code_set), B)]
    f, t = sc["code_frac"], sc["text_frac"]
    pc = cfg["pretrain"]
    text = ({"ids": ids} for ids in pretrain_documents(tok, stories, d["digits"], pc["arith_frac"], seed + 5, pc["code_frac"]))
    rows = mixture([epochs(train, seed + 2), epochs(agent, seed + 3), text], [1 - f - t, f, t], seed + 4)
    print(f"SFT set: {len(train)} chat conversations ({sc['steps'] * B * (1 - f - t) / len(train):.1f} epochs), "
          f"{len(agent)} agent transcripts ({sc['steps'] * B * f / len(agent):.1f} epochs), "
          f"{t:.0%} pretraining documents")

    def val_fn() -> dict:
        return {"val_loss": evaluate_loss(model, val_batches, device), "code_val_loss": evaluate_loss(model, code_val, device)}

    log = Logger(run / "sft" / "log.jsonl")
    stats = train_loop(model, chat_batches(tok, rows, B, T), sc, log, device, val_fn, cfg.get("optimizer", "adamw"))
    save_stage(run, "sft", model, tok, stats, cfg, device, prev)


if __name__ == "__main__":
    main()
