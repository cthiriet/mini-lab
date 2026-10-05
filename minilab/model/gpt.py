"""A small decoder-only transformer (GPT style) with a slot-based KV cache.

- RoPE positions, RMSNorm, no biases, tied input/output embeddings.
- `forward` is the training path (full causal attention over a batch).
- `forward_cached` is the inference path: every sequence lives in a *slot* of a
  shared KVCache and can be at a different position. This is what makes
  continuous batching possible: one decode step advances many sequences at once.

Five small changes from the 2024-25 speedruns and labs, each off by default and all on in
configs/prelude.toml (docs/training.md, "Lessons"): a squared-ReLU MLP, value residual
(ResFormer), qk-norm, the embeddings mixed into every block's input, and soft-capped logits.
Together they lower the pretraining loss as much as ~1.6x more steps, for no extra time
per step once compiled.
"""

from __future__ import annotations

import math
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
    mlp: str = "gelu"             # gelu | relu2 (squared ReLU)
    value_residual: bool = False  # every layer's values mixed with the first layer's
    qk_norm: bool = False         # RMSNorm on queries and keys
    x0_mix: bool = False          # each block's input mixed with the embeddings (learned scalars)
    softcap: float = 0.0          # logits = c * tanh(logits / c), 0 = off


class KVCache:
    """Keys/values for up to `batch_size` sequences ("slots") of up to `max_len` tokens."""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int | None = None,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        head_dim = config.n_embd // config.n_head
        self.max_len = max_len or config.block_size
        shape = (config.n_layer, batch_size, config.n_head, self.max_len, head_dim)
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
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout
        self.qk_norm = config.qk_norm
        # value residual: from the second layer on, values lerp toward the first layer's (learned, starts halfway)
        self.value_mix = nn.Parameter(torch.tensor(0.5)) if config.value_residual and layer > 0 else None

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0,
                slots: torch.Tensor | None = None, pos: torch.Tensor | None = None, state: dict | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if "v0" not in state:
            state["v0"] = v
        elif self.value_mix is not None:
            v = v + self.value_mix * (state["v0"] - v)  # not torch.lerp: compiled for MPS in bf16, its abs() doesn't build
        if self.qk_norm:
            q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
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
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.mlp in ("gelu", "relu2"), f"unknown mlp: {config.mlp}"
        self.relu2 = config.mlp == "relu2"
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        h = self.fc(x)
        return self.proj(F.relu(h).square() if self.relu2 else F.gelu(h))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer: int = 0):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.attn = Attention(config, layer)
        self.norm2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.x0 = nn.Parameter(torch.tensor([1.0, 0.0])) if config.x0_mix else None  # (residual, embeddings)

    def forward(self, x, cos, sin, x0, **kwargs):
        if self.x0 is not None:
            x = self.x0[0] * x + self.x0[1] * x0
        x = x + self.attn(self.norm1(x), cos, sin, **kwargs)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layer))
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        cos, sin = _rope_tables(config.block_size, config.n_embd // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # residual projections: scaled init
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _trunk(self, x, cos, sin, **cache_kwargs):
        x0, state = x, {}
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, x0, state=state, **({**cache_kwargs, "layer": i} if cache_kwargs else {}))
        return x

    def _logits(self, x):
        logits = self.lm_head(self.norm(x))
        if self.config.softcap:
            c = self.config.softcap
            logits = c * torch.tanh(logits.float() / c)
        return logits

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Training path. idx/targets: (B, T). Targets of -1 are ignored in the loss."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        cos = self.rope_cos[None, None, :T]
        sin = self.rope_sin[None, None, :T]
        logits = self._logits(self._trunk(self.drop(self.wte(idx)), cos, sin))
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
        x = self._trunk(self.wte(idx), cos, sin, cache=cache, slots=slots, pos=pos)
        cache.lengths[slots] += T
        return self._logits(x[:, -1])

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None, stop_ids: set[int] | frozenset = frozenset(),
                 generator: torch.Generator | None = None) -> list[list[int]]:
        """Simple batched sampling. Returns new tokens per prompt (including the stop token if hit)."""
        device = self.wte.weight.device
        cache = KVCache(self.config, batch_size=len(prompts), device=device, dtype=self.wte.weight.dtype)
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
