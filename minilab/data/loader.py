"""Training batches.

Pretraining and midtraining *pack* documents: each document starts with <|bos|>,
documents are concatenated into one stream, and the stream is cut into rows of
block_size + 1 tokens (inputs = row[:-1], targets = row[1:]). No compute is wasted
on padding, and the model learns that <|bos|> means "a new, unrelated text starts".

SFT uses one conversation per row, padded to the longest row of the batch. That is
exactly the situation at inference time (a conversation starts at position 0), and
it lets us train only on what the assistant says: targets are -1 (ignored by the
loss) everywhere else.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterator

import torch

from minilab.data import arithmetic, code
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_conversation

Batch = tuple[torch.Tensor, torch.Tensor]


def pretrain_documents(tok: Tokenizer, stories: list[str], digits: list[int], arith_frac: float,
                       seed: int, code_frac: float = 0.0) -> Iterator[list[int]]:
    """Endless stream of tokenized documents: shuffled stories (one epoch after the
    other) with arithmetic worksheets mixed in, each document with probability arith_frac,
    and for mini-4 Python documents of the code world (data/code.py) with probability code_frac."""
    rng = random.Random(seed)
    python = code.pretrain_documents(seed + 1) if code_frac else None
    order = list(range(len(stories)))
    while True:
        rng.shuffle(order)
        for i in order:
            while rng.random() < arith_frac:
                yield [tok.bos_id, *tok.encode(arithmetic.pretrain_document(rng, digits))]
            while python and rng.random() < code_frac:
                yield [tok.bos_id, *tok.encode(next(python))]
            yield [tok.bos_id, *tok.encode(stories[i])]


def mixture(streams: list[Iterator], weights: list[float], seed: int) -> Iterator:
    """Endless interleaving of streams, each item drawn from stream i with probability weights[i]."""
    rng = random.Random(seed)
    while True:
        yield next(rng.choices(streams, weights)[0])


def packed_batches(docs: Iterator[list[int]], batch_size: int, block_size: int) -> Iterator[Batch]:
    """Pack a stream of documents into (inputs, targets) batches of shape (batch_size, block_size)."""
    n = batch_size * (block_size + 1)
    buf: list[int] = []
    for doc in docs:
        buf.extend(doc)
        while len(buf) >= n:
            rows = torch.tensor(buf[:n]).view(batch_size, block_size + 1)
            del buf[:n]
            yield rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def story_batches(tok: Tokenizer, stories: list[str], batch_size: int, block_size: int,
                  n_batches: int) -> list[Batch]:
    """A fixed list of packed batches of stories, in file order: a stable validation set."""
    docs = ([tok.bos_id, *tok.encode(s)] for s in stories)
    return list(itertools.islice(packed_batches(docs, batch_size, block_size), n_batches))


def chat_batch(tok: Tokenizer, conversations: list[dict], block_size: int) -> Batch:
    """One padded row per conversation; targets are -1 except on assistant tokens (only
    the answer to the last user message when conv["train_on"] == "last": the rest is just
    context). That answer can span several assistant turns: the calculator call, then,
    after the tool's result, the final answer. A row {"ids": [...]} is a plain document,
    trained on every token."""
    rows = []
    for conv in conversations:
        if "ids" in conv:  # a pretraining document (mini-4's SFT replays some): loss on every token
            ids = conv["ids"][:block_size + 1]
            rows.append((ids[:-1], ids[1:]))
            continue
        ids, mask = render_conversation(tok, conv["messages"], conv.get("tools"))
        if conv.get("train_on") == "last":
            last = len(ids) - 1 - ids[::-1].index(tok.special("<|user_start|>"))
            mask = [0] * last + mask[last:]
        ids, mask = ids[:block_size + 1], mask[:block_size + 1]
        rows.append((ids[:-1], [t if m else -1 for t, m in zip(ids[1:], mask[1:])]))
    # Padded to a multiple of 64: on MPS, every new batch shape compiles (and keeps) new kernels,
    # and with lengths up to 1,024 that alone ran a 5.8M-parameter SFT out of memory.
    T = min(block_size, -(-max(len(x) for x, _ in rows) // 64) * 64)
    # Any id works as input padding: it comes after the real tokens (causal attention)
    # and its targets are ignored.
    x = torch.zeros(len(rows), T, dtype=torch.long)
    y = torch.full((len(rows), T), -1, dtype=torch.long)
    for i, (xi, yi) in enumerate(rows):
        x[i, :len(xi)] = torch.tensor(xi)
        y[i, :len(yi)] = torch.tensor(yi)
    return x, y


def epochs(items: list, seed: int) -> Iterator:
    """Endless iteration over a fixed dataset, reshuffled every epoch."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        yield from (items[i] for i in order)


def chat_batches(tok: Tokenizer, conversations: Iterator[dict], batch_size: int, block_size: int) -> Iterator[Batch]:
    while True:
        yield chat_batch(tok, [next(conversations) for _ in range(batch_size)], block_size)
