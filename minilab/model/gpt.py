"""A small decoder-only transformer (GPT style) with a slot-based KV cache.

- RoPE positions, RMSNorm, GELU MLP, no biases, tied input/output embeddings.
- Optional 2026-style block (`GPTConfig.mlp`, `qk_norm`, `attn_gate`): a SwiGLU MLP,
  RMSNorm on queries and keys, and a sigmoid gate on the attention output (Qwen3.5,
  Kimi K3). Off by default, so older checkpoints load unchanged.
- Optional recurrent depth (`GPTConfig.loops`, a "looped transformer"): the blocks
  between a prelude and a coda form a core that runs several times with the same
  weights, so the model gets deeper at test time without new parameters (Huginn, Ouro,
  and reportedly GPT-6 Astra). Each run of a block is its own layer of the KV cache.
- `forward` is the training path (full causal attention over a batch).
- `forward_cached` is the inference path: every sequence lives in a *slot* of a
  shared KVCache and can be at a different position. This is what makes
  continuous batching possible: one decode step advances many sequences at once.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    mlp: str = "gelu"               # "gelu" (GPT-2) or "swiglu" (Llama and every model since)
    mlp_hidden: int | None = None   # default: 4 x n_embd, the GELU MLP's width
    qk_norm: bool = False           # RMSNorm on each head's queries and keys, before RoPE
    attn_gate: bool = False         # output = attention * sigmoid(W_g x), per channel
    # Recurrent depth: the first n_prelude and the last n_coda blocks run once, the core
    # between them runs `loops` times. The defaults are a plain GPT.
    n_prelude: int = 0
    n_coda: int = 0
    loops: int = 1                  # core runs at inference, unless overridden
    train_loops: list[int] | None = None  # [lo, hi]: each training step draws its loop count
    inject: bool = False            # add the prelude's output back before every extra loop

    def depth(self, loops: int | None = None) -> int:
        """Blocks run per token (the KV cache's layers) with `loops` core runs."""
        return self.n_layer + (self.n_layer - self.n_prelude - self.n_coda) * ((loops or self.loops) - 1)


class KVCache:
    """Keys/values for up to `batch_size` sequences ("slots") of up to `max_len` tokens.

    `loops` (default: the config's) fixes how many times a looped core runs: every token
    of a sequence must go through the same blocks, or its later tokens would attend to
    keys that were never written."""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int | None = None,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32, loops: int | None = None):
        head_dim = config.n_embd // config.n_head
        self.max_len = max_len or config.block_size
        self.loops = loops or config.loops
        shape = (config.depth(self.loops), batch_size, config.n_head, self.max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(batch_size, dtype=torch.long)  # tokens stored per slot

    def reset(self, slot: int) -> None:
        # Stale keys beyond `lengths` are masked out, so no need to zero them.
        self.lengths[slot] = 0


def _rope_tables(block_size: int, head_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(block_size).float(), inv_freq)  # (T, D/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (B or 1, 1, T, D/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout
        # QK-norm bounds the attention logits; the gate lets a head output nothing
        # instead of dumping its attention on the first token (an "attention sink").
        self.q_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else None
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False) if config.attn_gate else None

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0,
                slots: torch.Tensor | None = None, pos: torch.Tensor | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if cache is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Write the new keys/values into each sequence's slot, then attend over
            # everything that slot has seen so far.
            for b in range(B):
                s, p = int(slots[b]), int(pos[b])
                cache.k[layer, s, :, p:p + T] = k[b]
                cache.v[layer, s, :, p:p + T] = v[b]
            L = int(pos.max()) + T
            rows = slots.to(cache.k.device)
            keys, values = cache.k[layer, rows, :, :L], cache.v[layer, rows, :, :L]
            q_pos = pos[:, None] + torch.arange(T, device=pos.device)          # (B, T)
            mask = torch.arange(L, device=pos.device)[None, None, :] <= q_pos[:, :, None]  # (B, T, L)
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None].to(x.device))
        y = y.transpose(1, 2).reshape(B, T, C)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_hidden or 4 * config.n_embd
        self.swiglu = config.mlp == "swiglu"
        # SwiGLU: silu(x W_a) * (x W_b), both halves computed by one matmul.
        self.fc = nn.Linear(config.n_embd, (2 if self.swiglu else 1) * hidden, bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        if self.swiglu:
            a, b = self.fc(x).chunk(2, dim=-1)
            return self.proj(F.silu(a) * b)
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, **cache_kwargs):
        x = x + self.attn(self.norm1(x), cos, sin, **cache_kwargs)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        cos, sin = _rope_tables(config.block_size, config.n_embd // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # residual projections: scaled init
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.depth()))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk(self, x, cos, sin, loops: int, **cache_kwargs):
        """The blocks in the order they run: the prelude, the core `loops` times, the coda.
        With `inject`, every extra loop starts from its state plus the prelude's output (the
        embeddings if there is no prelude), so the input is never forgotten."""
        c = self.config
        core = range(c.n_prelude, c.n_layer - c.n_coda)
        order = [*range(c.n_prelude), *(i for _ in range(loops) for i in core), *range(c.n_layer - c.n_coda, c.n_layer)]
        e = x if c.n_prelude == 0 else None
        for layer, i in enumerate(order):
            if i == core.start and layer > core.start and c.inject:  # an extra loop starts
                x = x + e
            x = self.blocks[i](x, cos, sin, layer=layer, **cache_kwargs)
            if layer == c.n_prelude - 1:
                e = x
        return x

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None):
        """Training path. idx/targets: (B, T). Targets of -1 are ignored in the loss.

        `loops` defaults to the config's, or in training mode to a draw from `train_loops`
        (one per step: like Huginn, the model learns to use whatever depth it is given)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        if loops is None:
            loops = self.config.loops
            if self.training and self.config.train_loops:
                loops = random.randint(*self.config.train_loops)
        cos = self.rope_cos[None, None, :T]
        sin = self.rope_sin[None, None, :T]
        x = self.trunk(self.drop(self.wte(idx)), cos, sin, loops)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1)
        return logits, loss

    def forward_cached(self, idx: torch.Tensor, cache: KVCache, slots: torch.Tensor) -> torch.Tensor:
        """Inference path. idx: (B, T) new tokens for the sequences in `slots` (B,).

        Each slot continues from its own position (cache.lengths). All rows must
        have the same T (T=1 for batched decoding, any T for a single prefill).
        Returns logits for the last position of each row: (B, vocab_size).
        """
        B, T = idx.shape
        slots = slots.to(torch.long).cpu()
        pos = cache.lengths[slots].clone()
        assert int(pos.max()) + T <= cache.max_len, "KV cache is full"
        q_pos = (pos[:, None] + torch.arange(T)).to(idx.device)  # (B, T)
        cos = self.rope_cos[q_pos][:, None]  # (B, 1, T, D/2)
        sin = self.rope_sin[q_pos][:, None]
        x = self.trunk(self.wte(idx), cos, sin, cache.loops, cache=cache, slots=slots, pos=pos)
        cache.lengths[slots] += T
        return self.lm_head(self.norm(x[:, -1]))

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None, stop_ids: set[int] | frozenset = frozenset(),
                 generator: torch.Generator | None = None, loops: int | None = None) -> list[list[int]]:
        """Simple batched sampling. Returns new tokens per prompt (including the stop token if hit)."""
        device = self.wte.weight.device
        cache = KVCache(self.config, batch_size=len(prompts), device=device, dtype=self.wte.weight.dtype, loops=loops)
        logits = torch.cat([
            self.forward_cached(torch.tensor([p], device=device), cache, torch.tensor([i]))
            for i, p in enumerate(prompts)
        ])
        outputs: list[list[int]] = [[] for _ in prompts]
        active = list(range(len(prompts)))
        for _ in range(max_new_tokens):
            next_ids = sample_next(logits, temperature, top_k, top_p, generator)
            still_active, feed = [], []
            for row, slot in enumerate(active):
                t = int(next_ids[row])
                outputs[slot].append(t)
                if t not in stop_ids and int(cache.lengths[slot]) < cache.max_len:
                    still_active.append(slot)
                    feed.append(t)
            active = still_active
            if not active:
                break
            logits = self.forward_cached(torch.tensor(feed, device=device)[:, None], cache, torch.tensor(active))
        return outputs


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token per row of logits (B, V). temperature=0 means greedy."""
    if temperature < 1e-4:  # tiny temperatures would overflow the logits to inf/NaN: same as greedy
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p  # keep the first token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
