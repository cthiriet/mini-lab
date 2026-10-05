"""A small decoder-only transformer (GPT style) with a slot-based KV cache.

- RoPE positions, RMSNorm, GELU MLP, no biases, tied input/output embeddings.
- `forward` is the training path (full causal attention over a batch).
- `forward_cached` is the inference path: every sequence lives in a *slot* of a
  shared KVCache and can be at a different position. This is what makes
  continuous batching possible: one decode step advances many sequences at once.

The fields of GPTConfig after `dropout` are architecture variants under test (branch
arch-search); their defaults are the architecture above.
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
    # ---- variants (defaults: the architecture above)
    mlp: str = "gelu"            # gelu | relu2 | swiglu
    mlp_hidden: int = 0          # 0: 4 * n_embd
    n_kv_head: int = 0           # grouped-query attention (0: n_head)
    qk_norm: bool = False        # RMSNorm on queries and keys
    qk_gain: bool = False        # with qk_norm: learned gains, so attention can get sharper than logits of sqrt(head_dim)
    parallel: bool = False       # attention and MLP side by side, from one norm (GPT-J, PaLM)
    moe_experts: int = 0         # routed experts in each MoE layer (0: dense MLP everywhere)
    moe_top_k: int = 2           # experts per token
    moe_hidden: int = 0          # hidden size of a routed expert (0: dense hidden / top_k)
    moe_shared: int = 0          # hidden size of the shared expert every token goes through (0: none)
    moe_every: int = 1           # MoE in every k-th layer (counted from the last), dense MLP elsewhere
    moe_capacity: float = 1.25   # tokens an expert takes in training, x the average (the rest is dropped)
    moe_balance: str = "bias"    # bias: DeepSeek-V3's auxiliary-loss-free balancing | aux: Switch loss
    value_residual: bool = False # every layer's values mixed with the first layer's (ResFormer)
    unet: bool = False           # skip connections from the first half of the layers to the second
    x0_mix: bool = False         # each block's input mixed with the embeddings (learned scalars)
    softcap: float = 0.0         # logits = c * tanh(logits / c) (Gemma 2)
    tie: bool = True             # output head tied to the embedding
    local_window: int = 0        # chunked local attention in all layers but the global ones (0: off)
    global_every: int = 3        # with local_window: every k-th layer (counted from the last) is global
    zero_init: bool = False      # residual projections start at zero
    rope_base: float = 10000.0

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    def is_moe(self, layer: int) -> bool:
        return self.moe_experts > 0 and (self.n_layer - 1 - layer) % self.moe_every == 0

    def is_local(self, layer: int) -> bool:
        return self.local_window > 0 and (self.n_layer - 1 - layer) % self.global_every != 0


class KVCache:
    """Keys/values for up to `batch_size` sequences ("slots") of up to `max_len` tokens."""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int | None = None,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        head_dim = config.n_embd // config.n_head
        self.max_len = max_len or config.block_size
        shape = (config.n_layer, batch_size, config.kv_heads, self.max_len, head_dim)
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


def _chunked_attention(q, k, v, w: int):
    """Causal attention within consecutive chunks of w tokens (Llama 4's local layers):
    the chunks become batch rows, so the cost is linear in T."""
    B, H, T, D = q.shape
    pad = -T % w
    if pad:  # padding at the end: causal attention, earlier tokens never see it
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    n = (T + pad) // w
    # (B, H, n*w, D) -> (B*n, H, w, D): 4-d, the only layout torch.compile's MPS attention takes
    q, k, v = (t.reshape(B, t.shape[1], n, w, D).transpose(1, 2).reshape(B * n, t.shape[1], w, D) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.reshape(B, n, H, w, D).transpose(1, 2).reshape(B, H, n * w, D)[:, :, :T]


class Attention(nn.Module):
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        self.n_head, self.n_kv = config.n_head, config.kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, (config.n_head + 2 * self.n_kv) * self.head_dim, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout
        self.qk_norm = config.qk_norm
        self.q_gain = nn.Parameter(torch.ones(self.head_dim)) if config.qk_norm and config.qk_gain else None
        self.k_gain = nn.Parameter(torch.ones(self.head_dim)) if config.qk_norm and config.qk_gain else None
        self.window = config.local_window if config.is_local(layer) else 0
        self.value_mix = nn.Parameter(torch.tensor(0.5)) if config.value_residual and layer > 0 else None

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0,
                slots: torch.Tensor | None = None, pos: torch.Tensor | None = None, state: dict | None = None):
        B, T, C = x.shape
        D = self.head_dim
        q, k, v = self.qkv(x).split([self.n_head * D, self.n_kv * D, self.n_kv * D], dim=2)
        q = q.view(B, T, self.n_head, D).transpose(1, 2)
        k, v = (t.view(B, T, self.n_kv, D).transpose(1, 2) for t in (k, v))
        if state is not None and "v0" not in state:
            state["v0"] = v
        elif self.value_mix is not None:
            v = v + self.value_mix * (state["v0"] - v)  # not torch.lerp: compiled for MPS in bf16, its abs() doesn't build
        if self.qk_norm:
            q, k = F.rms_norm(q, (D,), self.q_gain), F.rms_norm(k, (D,), self.k_gain)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        gqa = self.n_kv != self.n_head

        if cache is None:
            if gqa:
                k, v = (t.repeat_interleave(self.n_head // self.n_kv, dim=1) for t in (k, v))
            if self.window and T > self.window:
                y = _chunked_attention(q, k, v, self.window)
            else:
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
            if gqa:
                keys, values = (t.repeat_interleave(self.n_head // self.n_kv, dim=1) for t in (keys, values))
            q_pos = pos[:, None] + torch.arange(T, device=pos.device)          # (B, T)
            k_pos = torch.arange(L, device=pos.device)[None, None, :]
            mask = k_pos <= q_pos[:, :, None]  # (B, T, L)
            if self.window:
                mask &= k_pos // self.window == q_pos[:, :, None] // self.window
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None].to(x.device))
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


def _activate(h: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "gelu":
        return F.gelu(h)
    if kind == "relu2":
        return F.relu(h).square()
    assert kind == "swiglu", f"unknown mlp: {kind}"
    a, b = h.chunk(2, dim=-1)
    return F.silu(a) * b


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, hidden: int = 0):
        super().__init__()
        hidden = hidden or config.mlp_hidden or 4 * config.n_embd
        self.kind = config.mlp
        self.fc = nn.Linear(config.n_embd, hidden * (2 if self.kind == "swiglu" else 1), bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.proj(_activate(self.fc(x), self.kind))


class MoE(nn.Module):
    """Token-choice top-k mixture of experts with static shapes (good for MPS): each expert
    has a buffer of `capacity` tokens, filled first with every token's first choice, then
    the second choices, and so on (GShard, Switch). Tokens that don't fit skip the expert
    (the residual carries them). In eval nothing is dropped. Optional shared expert
    (DeepSeekMoE)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        C, E, k = config.n_embd, config.moe_experts, config.moe_top_k
        hidden = config.moe_hidden or (config.mlp_hidden or 4 * C) // k
        self.E, self.k, self.kind = E, k, config.mlp
        self.capacity, self.balance = config.moe_capacity, config.moe_balance
        self.router = nn.Linear(C, E, bias=False)
        self.w_in = nn.Parameter(torch.randn(E, C, hidden * (2 if self.kind == "swiglu" else 1)) * 0.02)
        self.w_out = nn.Parameter(torch.randn(E, hidden, C) * 0.02)
        self.shared = MLP(config, config.moe_shared) if config.moe_shared else None
        self.register_buffer("route_bias", torch.zeros(E))
        self.aux_loss = None
        self.stats: dict = {}

    def forward(self, x):
        B, T, C = x.shape
        E, k = self.E, self.k
        x = x.reshape(-1, C)
        N = x.shape[0]
        scores = self.router(x).float().softmax(-1)                        # (N, E)
        choice = (scores + self.route_bias) if self.balance == "bias" else scores
        top = choice.topk(k, dim=-1).indices                               # (N, k)
        gate = scores.gather(-1, top)
        if k > 1:  # top-1 keeps the raw probability (Switch): normalized, it would be 1 and the router would get no gradient
            gate = gate / gate.sum(-1, keepdim=True)
        cap = math.ceil(N * k / E * self.capacity) if self.training else N
        e = top.T.reshape(-1)                                              # (kN,): first choices first
        onehot = F.one_hot(e, E)                                           # (kN, E)
        pos = ((onehot.cumsum(0) - 1) * onehot).sum(-1)                    # place in the expert's buffer
        keep = pos < cap
        slot = torch.where(keep, e * cap + pos, E * cap)                   # E * cap: a dump row
        xs = x.repeat(k, 1)
        buf = x.new_zeros(E * cap + 1, C).index_add(0, slot, xs)[:-1].view(E, cap, C)
        h = _activate(torch.bmm(buf, self.w_in), self.kind)
        y = torch.bmm(h, self.w_out).view(E * cap, C)
        y = torch.cat([y, y.new_zeros(1, C)])
        w = (gate.T.reshape(-1) * keep).to(x.dtype)
        out = x.new_zeros(N, C).index_add(0, torch.arange(N, device=x.device).repeat(k), y[slot] * w[:, None])
        if self.training:
            with torch.no_grad():
                load = onehot.sum(0).float()
                self.stats = {"drop": (~keep).float().mean(), "max_load": load.max() / load.mean()}
                if self.balance == "bias":
                    self.route_bias += 1e-3 * torch.sign(load.mean() - load)
            if self.balance == "aux":
                f = onehot.float().mean(0) * E / k
                self.aux_loss = 0.01 * (f * scores.mean(0)).sum()
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer: int = 0):
        super().__init__()
        self.parallel = config.parallel
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.attn = Attention(config, layer)
        if not self.parallel:
            self.norm2 = nn.RMSNorm(config.n_embd)
        self.mlp = MoE(config) if config.is_moe(layer) else MLP(config)
        self.x0 = nn.Parameter(torch.tensor([1.0, 0.0])) if config.x0_mix else None

    def forward(self, x, cos, sin, x0=None, **cache_kwargs):
        if self.x0 is not None:
            x = self.x0[0] * x + self.x0[1] * x0
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, cos, sin, **cache_kwargs) + self.mlp(h)
        x = x + self.attn(self.norm1(x), cos, sin, **cache_kwargs)
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
        if config.tie:
            self.lm_head.weight = self.wte.weight  # weight tying
        if config.unet:
            self.skip_weights = nn.Parameter(torch.ones(config.n_layer // 2))
        cos, sin = _rope_tables(config.block_size, config.n_embd // config.n_head, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_out"):  # residual projections: scaled init
                if config.zero_init:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params(self) -> int:
        """Parameters a token goes through (MoE: only its top-k experts)."""
        n = self.num_params()
        for m in self.modules():
            if isinstance(m, MoE):
                n -= (m.w_in.numel() + m.w_out.numel()) * (m.E - m.k) // m.E
        return n

    def _trunk(self, x, cos, sin, **cache_kwargs):
        state: dict = {}
        x0 = x
        skips = []
        half = self.config.n_layer // 2
        for i, block in enumerate(self.blocks):
            if self.config.unet and i >= self.config.n_layer - half:
                x = x + self.skip_weights[i - (self.config.n_layer - half)] * skips.pop()
            kwargs = {**cache_kwargs, "layer": i} if cache_kwargs else {}
            x = block(x, cos, sin, x0=x0, state=state, **kwargs)
            if self.config.unet and i < half:
                skips.append(x)
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
        x = self.drop(self.wte(idx))
        logits = self._logits(self._trunk(x, cos, sin))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1)
            if self.training:
                aux = [m.aux_loss for m in self.modules() if isinstance(m, MoE) and m.aux_loss is not None]
                if aux:
                    loss = loss + sum(aux)
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
