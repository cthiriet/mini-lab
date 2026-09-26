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

from minilab.data import arithmetic
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_conversation

Batch = tuple[torch.Tensor, torch.Tensor]


def pretrain_documents(tok: Tokenizer, stories: list[str], digits: list[int], arith_frac: float,
                       seed: int) -> Iterator[list[int]]:
    """Endless stream of tokenized documents: shuffled stories (one epoch after the
    other) with arithmetic worksheets mixed in, each document with probability arith_frac."""
    rng = random.Random(seed)
    order = list(range(len(stories)))
    while True:
        rng.shuffle(order)
        for i in order:
            while rng.random() < arith_frac:
                yield [tok.bos_id, *tok.encode(arithmetic.pretrain_document(rng, digits))]
            yield [tok.bos_id, *tok.encode(stories[i])]


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
    the last assistant turn when conv["train_on"] == "last": the rest is just context)."""
    rows = []
    for conv in conversations:
        ids, mask = render_conversation(tok, conv["messages"], conv.get("tools"))
        if conv.get("train_on") == "last":
            last = len(ids) - 1 - ids[::-1].index(tok.special("<|assistant_start|>"))
            mask = [0] * last + mask[last:]
        ids, mask = ids[:block_size + 1], mask[:block_size + 1]
        rows.append((ids[:-1], [t if m else -1 for t, m in zip(ids[1:], mask[1:])]))
    T = max(len(x) for x, _ in rows)
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
