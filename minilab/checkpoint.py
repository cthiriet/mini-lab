"""Save / load a model checkpoint: a directory with weights, config and tokenizer.

    <dir>/model.pt         state_dict
    <dir>/config.json      {"model": GPTConfig fields, "meta": {...free-form...}}
    <dir>/tokenizer.json   see minilab.tokenizer.bpe
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import torch

from minilab.model.gpt import GPT, GPTConfig
from minilab.tokenizer.bpe import Tokenizer


def save_checkpoint(path: str | Path, model: GPT, tokenizer: Tokenizer, meta: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path / "model.pt")
    (path / "config.json").write_text(json.dumps({"model": asdict(model.config), "meta": meta or {}}, indent=2))
    tokenizer.save(path / "tokenizer.json")
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, Tokenizer, dict]:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    known = {f.name for f in fields(GPTConfig)}  # older checkpoints list options that are gone, at their defaults
    model = GPT(GPTConfig(**{k: v for k, v in cfg["model"].items() if k in known}))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, Tokenizer.load(path / "tokenizer.json"), cfg.get("meta", {})
