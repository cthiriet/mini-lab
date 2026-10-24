"""The microscope: the model's forward pass, written out so that everything is kept.

`trace(model, ids)` computes what GPT.forward computes (minilab/model/gpt.py) for one sequence,
step by step, and records:

    resid[l]     the residual stream: resid[0] the embeddings, resid[l + 1] after block l
    pattern[l]   each head's attention weights, (H, T, T): row t, where position t looks
    heads[l]     what each head writes into the residual stream, (H, T, C)
    neurons[l]   each MLP's hidden activations, (T, 4C)
    mlp[l]       what each MLP writes into the residual stream, (T, C)
    mix[l]       block l's x0 mix (a, b): its input is a * resid[l] + b * resid[0]
    logits       (T, V)

Attention is computed in the open (softmax of the scaled scores), not by a fused kernel, so its
weights exist; a head's output is its slice of the attention output times its slice of the
output projection, so the heads add up to the attention layer's output.

`patch` changes activations as they are computed: {(name, layer): fn}, name one of "resid"
(the stream entering block l, before its x0 mix), "pattern", "heads", "neurons" or "mlp"; fn
gets the tensor and returns the one to use. That's activation patching and ablation.

One sequence, no KV cache, float32: a microscope, not an engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from minilab.model.gpt import GPT, _apply_rope

Patch = dict[tuple[str, int], Callable[[torch.Tensor], torch.Tensor]]


@dataclass
class Trace:
    ids: list[int]
    resid: list[torch.Tensor] = field(default_factory=list)
    pattern: list[torch.Tensor] = field(default_factory=list)
    heads: list[torch.Tensor] = field(default_factory=list)
    neurons: list[torch.Tensor] = field(default_factory=list)
    mlp: list[torch.Tensor] = field(default_factory=list)
    mix: list[tuple[float, float]] = field(default_factory=list)
    logits: torch.Tensor | None = None


@torch.no_grad()
def trace(model: GPT, ids: list[int], patch: Patch | None = None) -> Trace:
    patch = patch or {}

    def keep(name: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        return patch[(name, layer)](x) if (name, layer) in patch else x

    cfg = model.config
    T, C, H = len(ids), cfg.n_embd, cfg.n_head
    D = C // H
    cos, sin = model.rope_cos[None, None, :T], model.rope_sin[None, None, :T]
    causal = torch.ones(T, T, dtype=torch.bool, device=cos.device).tril()
    out = Trace(list(ids))
    x0 = model.wte(torch.tensor(ids, device=cos.device))   # (T, C)
    x, v0 = x0, None
    out.resid.append(x0)
    for layer, block in enumerate(model.blocks):
        x = keep("resid", layer, x)
        a, b = (float(block.x0[0]), float(block.x0[1])) if block.x0 is not None else (1.0, 0.0)
        out.mix.append((a, b))
        x = a * x + b * x0
        # attention, head by head
        attn = block.attn
        q, k, v = attn.qkv(block.norm1(x)).split(C, dim=-1)
        q, k, v = (t.view(T, H, D).transpose(0, 1) for t in (q, k, v))   # (H, T, D)
        if v0 is None:
            v0 = v
        elif attn.value_mix is not None:
            v = v + attn.value_mix * (v0 - v)
        if attn.qk_norm:
            q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
        q, k = _apply_rope(q[None], cos, sin)[0], _apply_rope(k[None], cos, sin)[0]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        pattern = keep("pattern", layer, scores.masked_fill(~causal, float("-inf")).softmax(-1))   # (H, T, T)
        w = attn.proj.weight.view(C, H, D).permute(1, 0, 2)   # (H, C, D): each head's slice of the projection
        heads = keep("heads", layer, torch.einsum("htd,hcd->htc", pattern @ v, w))   # (H, T, C)
        x = x + heads.sum(0)
        # MLP
        mlp = block.mlp
        h = mlp.fc(block.norm2(x))
        neurons = keep("neurons", layer, F.relu(h).square() if mlp.relu2 else F.gelu(h))
        m = keep("mlp", layer, mlp.proj(neurons))
        x = x + m
        out.pattern.append(pattern)
        out.heads.append(heads)
        out.neurons.append(neurons)
        out.mlp.append(m)
        out.resid.append(x)
    out.logits = model._logits(x)
    return out


def logit_lens(model: GPT, tr: Trace) -> torch.Tensor:
    """What the model would predict from the residual stream after each block: (L + 1, T, V)."""
    with torch.no_grad():
        return torch.stack([model._logits(r) for r in tr.resid])


def components(tr: Trace) -> dict[str, torch.Tensor]:
    """The final residual stream split into what each part of the model wrote, as it reaches
    the end (scaled by the x0 mixes of the blocks after it): {"embed", "L0.H2", "L0.mlp", ...},
    each (T, C). They add up to resid[-1]."""
    after = [math.prod(a for a, _ in tr.mix[layer + 1:]) for layer in range(len(tr.mix))]   # how later blocks scale a layer's output
    embed = math.prod(a for a, _ in tr.mix) + sum(b * after[layer] for layer, (_, b) in enumerate(tr.mix))
    out = {"embed": tr.resid[0] * embed}
    for layer, scale in enumerate(after):
        for h in range(tr.heads[layer].shape[0]):
            out[f"L{layer}.H{h}"] = tr.heads[layer][h] * scale
        out[f"L{layer}.mlp"] = tr.mlp[layer] * scale
    return out


def attribution(model: GPT, tr: Trace, pos: int, good: int, bad: int) -> dict[str, float]:
    """Direct logit attribution: how much each component pushes the logit of token `good` over
    `bad` at position `pos`, through the final norm (its scale frozen at its actual value).
    Before the soft cap, which keeps the order."""
    final = tr.resid[-1][pos]
    scale = model.norm.weight / final.pow(2).mean().add(model.norm.eps or torch.finfo(final.dtype).eps).sqrt()
    direction = (model.lm_head.weight[good] - model.lm_head.weight[bad]) * scale
    return {name: float(c[pos] @ direction.detach()) for name, c in components(tr).items()}
