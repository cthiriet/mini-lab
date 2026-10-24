"""A dictionary of features: a sparse autoencoder on the residual stream.

The residual stream is 256 numbers per token, each one mixing many things. A sparse autoencoder
(SAE) learns a dictionary of many more directions (`n` features, 16x the width) such that every
token's residual stream is close to a sum of only `k` of them: each feature then tends to mean
one thing ("a digit in a scratchpad's first line", "a cat in a story", "inside a tool call").
This is the TopK SAE of OpenAI (Gao et al., 2024): the encoder keeps the k largest
pre-activations, so sparsity is exact and there is no L1 penalty to tune; an auxiliary loss
lets dead features learn from what the live ones miss.

    uv run python -m minilab.interp.sae models/prelude-2 --config configs/prelude.toml

writes the release's dictionary to models/prelude-2/features/ (the speedrun's last step, with the
config's [sae] section). Training streams activations from the model on its own kinds of text
(stories, worksheets, Python, chats, agent transcripts: `stream`), so nothing is stored.
Quality (quality.json): the fraction of variance left unexplained, and the *loss recovered*: the
model's next-token loss with this layer replaced by its reconstruction, between the true loss
(100%) and the loss with the layer zeroed (0%). Then each feature's strongest examples on held-out
text, how often it fires, which sources it fires on, and the tokens it pushes toward
(features.json): the platform's Features pages read it, the inference engine steers with sae.pt.
Hand-written labels, if any, go in labels.json ({"1112": "..."}).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checkpoint import load_checkpoint
from minilab.data import arithmetic, code
from minilab.data.conversations import StoryPool, sft_dataset
from minilab.data.tinystories import load_stories
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_conversation
from minilab.train.trainer import DEVICES, resolve_device

SOURCES = {"story": 0.25, "worksheet": 0.1, "python": 0.1, "chat": 0.3, "agent": 0.25}


class SAE(nn.Module):
    def __init__(self, d: int, n: int, k: int):
        super().__init__()
        self.d, self.n, self.k = d, n, k
        self.W_dec = nn.Parameter(F.normalize(torch.randn(n, d), dim=1))   # one unit direction per feature
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.tensor(1.0))   # inputs are divided by it: mean squared norm = d

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self.scale - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The features' activations: the k largest pre-activations (if positive), zero elsewhere."""
        top = self.pre(x).topk(self.k, dim=-1)
        return torch.zeros(*x.shape[:-1], self.n, device=x.device).scatter(-1, top.indices, F.relu(top.values))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.W_dec + self.b_dec) * self.scale


# ---- the model's activations -------------------------------------------------------

def stream(tok: Tokenizer, cfg: dict, seed: int) -> Iterator[tuple[list[int], str]]:
    """Endless documents (token ids, source) of the kinds the model was trained on."""
    rng = random.Random(seed)
    d = cfg["data"]
    stories = load_stories("train", d["train_mb"])
    chats = sft_dataset(seed, min(4000, cfg["sft"]["size"]), cfg["sft"], d["digits"], StoryPool(stories, d.get("story_max_chars", 700)))
    agent = code.sft_set(cfg, cfg.get("seed", 0) + 2)
    python = code.pretrain_documents(seed + 1)
    while True:
        source = rng.choices(list(SOURCES), list(SOURCES.values()))[0]
        if source in ("chat", "agent"):
            conv = rng.choice(chats if source == "chat" else agent)
            ids = render_conversation(tok, conv["messages"], conv.get("tools"))[0]
        else:
            text = {"story": lambda: rng.choice(stories), "worksheet": lambda: arithmetic.pretrain_document(rng, d["digits"]),
                    "python": lambda: next(python)}[source]()
            ids = [tok.bos_id, *tok.encode(text)]
        yield ids, source


def batches(docs: Iterator[tuple[list[int], str]], B: int, T: int) -> Iterator[tuple[torch.Tensor, list[list[str]]]]:
    """Documents packed into (B, T) token rows, with each token's source."""
    ids, srcs = [], []
    for doc, source in docs:
        ids += doc
        srcs += [source] * len(doc)
        while len(ids) >= B * T:
            yield torch.tensor(ids[:B * T]).view(B, T), [srcs[i * T:(i + 1) * T] for i in range(B)]
            ids, srcs = ids[B * T:], srcs[B * T:]


def _blocks(model: GPT, x: torch.Tensor, start: int, stop: int, h: torch.Tensor, x0: torch.Tensor, state: dict):
    T = x.shape[1]
    cos, sin = model.rope_cos[None, None, :T], model.rope_sin[None, None, :T]
    for block in model.blocks[start:stop]:
        h = block(h, cos, sin, x0, state=state)
    return h


@torch.no_grad()
def residual(model: GPT, x: torch.Tensor, layer: int) -> tuple[torch.Tensor, dict]:
    """The residual stream after `layer` blocks, (B, T, C), and what the later blocks need."""
    x0 = model.wte(x)
    state: dict = {}
    return _blocks(model, x, 0, layer, x0, x0, state), {"x0": x0, "state": state}


@torch.no_grad()
def loss_with(model: GPT, x: torch.Tensor, layer: int, replace=None) -> float:
    """Next-token loss with the residual stream after `layer` blocks replaced (replace(h) -> h)."""
    h, ctx = residual(model, x, layer)
    if replace is not None:
        h = replace(h)
    logits = model._logits(_blocks(model, x, layer, len(model.blocks), h, ctx["x0"], ctx["state"]))
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1)).item()


# ---- training ------------------------------------------------------------------------

def train(model: GPT, tok: Tokenizer, cfg: dict, layer: int, n: int, k: int, tokens: float, out: Path,
          device: str, B: int = 64, T: int = 256, lr: float = 2e-4, seed: int = 0) -> SAE:
    torch.manual_seed(seed)
    C = model.config.n_embd
    sae = SAE(C, n, k).to(device)
    data = batches(stream(tok, cfg, seed), B, T)
    first, _ = residual(model, next(data)[0].to(device), layer)
    first = first.reshape(-1, C)
    sae.scale.fill_(float(first.pow(2).sum(-1).mean().div(C).sqrt()))
    sae.b_dec.data = (first / sae.scale).mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    steps = int(tokens // (B * T))
    since_fired = torch.zeros(n, device=device)   # tokens since each feature last fired
    k_aux = min(n // 2, 8 * k)
    log, t0 = [], time.time()
    for step in range(steps):
        x, _ = next(data)
        h, _ = residual(model, x.to(device), layer)
        h = h.reshape(-1, C)
        target = h / sae.scale
        pre = sae.pre(h)
        top = pre.topk(k, dim=-1)
        f = torch.zeros_like(pre).scatter(-1, top.indices, F.relu(top.values))
        recon = f @ sae.W_dec + sae.b_dec
        err = target - recon
        loss = err.pow(2).sum(-1).mean() / target.var(0).sum()   # normalized MSE
        # AuxK: the dead features (silent for 10M tokens) try to reconstruct what the live ones missed
        fired = (f > 0).any(0)
        since_fired = torch.where(fired, torch.zeros_like(since_fired), since_fired + h.shape[0])
        dead = since_fired > 10_000_000
        aux = torch.tensor(0.0, device=device)
        if dead.any():
            dead_pre = pre.masked_fill(~dead, float("-inf"))
            t = dead_pre.topk(min(k_aux, int(dead.sum())), dim=-1)
            fa = torch.zeros_like(pre).scatter(-1, t.indices, F.relu(t.values))
            aux = (err.detach() - fa @ sae.W_dec).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        (loss + aux / 32).backward()
        with torch.no_grad():   # keep the decoder's directions unit-norm: no gradient along them
            g = sae.W_dec.grad
            g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step()
        with torch.no_grad():
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
        if (step + 1) % 100 == 0 or step == steps - 1:
            row = {"step": step + 1, "tokens": (step + 1) * B * T, "fvu": round(loss.item(), 4),
                   "dead": round(dead.float().mean().item(), 4), "elapsed": round(time.time() - t0, 1)}
            log.append(row)
            print(" | ".join(f"{k_} {v}" for k_, v in row.items()), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": sae.state_dict(), "d": C, "n": n, "k": k, "layer": layer}, out / "sae.pt")
    (out / "train_log.json").write_text(json.dumps(log))
    return sae


def load_sae(path: Path, device: str) -> tuple[SAE, int]:
    ckpt = torch.load(path / "sae.pt", map_location=device, weights_only=True)
    sae = SAE(ckpt["d"], ckpt["n"], ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state"])
    return sae, ckpt["layer"]


@torch.no_grad()
def quality(model: GPT, tok: Tokenizer, cfg: dict, sae: SAE, layer: int, device: str, n_batches: int = 20) -> dict:
    """On held-out text: variance unexplained, and the loss recovered by the reconstruction."""
    data = batches(stream(tok, cfg, seed=10_007), 32, 256)
    fvu, clean, spliced, zeroed = [], [], [], []
    for _ in range(n_batches):
        x = next(data)[0].to(device)
        h, _ = residual(model, x, layer)
        r = sae.decode(sae.encode(h))
        fvu.append(((h - r).pow(2).sum(-1).mean() / (h - h.mean((0, 1))).pow(2).sum(-1).mean()).item())
        clean.append(loss_with(model, x, layer))
        spliced.append(loss_with(model, x, layer, lambda _: r))
        zeroed.append(loss_with(model, x, layer, torch.zeros_like))
    m = lambda v: sum(v) / len(v)   # noqa: E731
    return {"fvu": round(m(fvu), 4), "loss": round(m(clean), 4), "loss_sae": round(m(spliced), 4),
            "loss_zero": round(m(zeroed), 4),
            "loss_recovered": round((m(zeroed) - m(spliced)) / (m(zeroed) - m(clean)), 4)}


# ---- what each feature does ---------------------------------------------------------

@torch.no_grad()
def features(model: GPT, tok: Tokenizer, cfg: dict, sae: SAE, layer: int, device: str, n_tokens: float = 1e6,
             top: int = 12, before: int = 14, after: int = 4) -> list[dict]:
    """Each feature's strongest activations on held-out text, with their context."""
    data = batches(stream(tok, cfg, seed=20_011), 32, 256)
    n = sae.n
    best: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]   # (activation, row id, position)
    rows, row_src = [], []
    fires, by_source = torch.zeros(n), {s: torch.zeros(n) for s in SOURCES}
    seen = 0
    while seen < n_tokens:
        x, srcs = next(data)
        h, _ = residual(model, x.to(device), layer)
        f = sae.encode(h).cpu()   # (B, T, n)
        for b in range(x.shape[0]):
            rid = len(rows)
            rows.append(x[b].tolist())
            row_src.append(srcs[b])
            fb = f[b]
            fires += (fb > 0).sum(0)
            for s in SOURCES:
                mask = torch.tensor([src == s for src in srcs[b]])
                by_source[s] += fb[mask].sum(0) if mask.any() else 0
            vals, pos = fb.max(0)   # each feature's strongest token in this row
            for i in torch.nonzero(vals > 0).flatten().tolist():
                heap = best[i]
                heap.append((float(vals[i]), rid, int(pos[i])))
                if len(heap) > 4 * top:
                    heap.sort(reverse=True)
                    del heap[top:]
        seen += x.numel()
    W_U = model.lm_head.weight
    effects = (sae.W_dec * model.norm.weight) @ W_U.T   # each feature's direct push on the next token
    out = []
    for i in range(n):
        examples = []
        for act, rid, pos in sorted(best[i], reverse=True)[:top]:
            lo, hi = max(0, pos - before), min(len(rows[rid]), pos + after + 1)
            ids = torch.tensor(rows[rid][lo:hi]).view(1, -1).to(device)
            acts = sae.encode(residual(model, ids, layer)[0])[0, :, i].tolist()   # this window on its own
            examples.append({"tokens": [tok.decode([t]) for t in rows[rid][lo:hi]], "acts": [round(a, 2) for a in acts],
                             "at": pos - lo, "act": round(act, 2), "source": row_src[rid][pos]})
        mass = {s: float(v[i]) for s, v in by_source.items()}
        total = sum(mass.values()) or 1.0
        promoted = effects[i].topk(8).indices.tolist()
        suppressed = (-effects[i]).topk(8).indices.tolist()
        at_tokens = Counter(e["tokens"][e["at"]] for e in examples)
        out.append({"id": i, "density": float(fires[i]) / seen, "sources": {s: round(v / total, 3) for s, v in mass.items()},
                    "top_token": at_tokens.most_common(1)[0] if at_tokens else None,
                    "promotes": [tok.decode([t]) for t in promoted], "suppresses": [tok.decode([t]) for t in suppressed],
                    "examples": examples})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train a release's feature dictionary (a sparse autoencoder) into <release>/features/.")
    p.add_argument("model", help="a release directory, e.g. models/prelude-2")
    p.add_argument("--config", default="configs/prelude.toml", help="the recipe: its [sae] section, and the data")
    p.add_argument("--device", default="auto", choices=DEVICES)
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    sc = cfg["sae"]   # layer, features, k, tokens, example_tokens
    device = resolve_device(args.device)
    model, tok, _ = load_checkpoint(args.model, device=device)
    out = Path(args.model) / "features"
    t0 = time.time()
    sae = train(model, tok, cfg, sc["layer"], sc["features"], sc["k"], sc["tokens"], out, device)
    q = quality(model, tok, cfg, sae, sc["layer"], device)
    (out / "quality.json").write_text(json.dumps(q, indent=2) + "\n")
    feats = features(model, tok, cfg, sae, sc["layer"], device, sc["example_tokens"])
    (out / "features.json").write_text(json.dumps({"layer": sc["layer"], "model": Path(args.model).name, "features": feats}))
    dead = sum(f["density"] == 0 for f in feats)
    print(f"{len(feats)} features ({dead} never fired), loss recovered {q['loss_recovered']:.1%}, "
          f"variance unexplained {q['fvu']:.1%}, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
