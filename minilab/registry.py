"""Model registry: released models live in MODELS_DIR, one directory each.

    models/<model-id>/
        model.pt, config.json, tokenizer.json   (a checkpoint, see minilab.checkpoint)
        release.json                            (metadata below)
        eval.json                               (eval results)
        MODEL_CARD.md

release.json:
    {
      "id": "prelude-2",
      "created": 1760000000,              # unix seconds
      "description": "...",
      "context_length": 1024,
      "pricing": {"input_per_1m": 10.0, "output_per_1m": 50.0},   # USD per 1M tokens
      "source_run": "runs/prelude",
      "default_temperature": 0.0          # when a request doesn't set one (OpenAI's default is 1)
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Pricing:
    input_per_1m: float = 10.0   # USD per 1M prompt tokens
    output_per_1m: float = 50.0  # USD per 1M completion tokens

    def cost_micros(self, prompt_tokens: int, completion_tokens: int) -> int:
        """Cost in micro-dollars (1e-6 USD). USD/1M tokens == micro-USD/token."""
        return round(prompt_tokens * self.input_per_1m + completion_tokens * self.output_per_1m)


@dataclass
class ModelInfo:
    id: str
    created: int
    description: str = ""
    context_length: int = 1024
    pricing: Pricing = field(default_factory=Pricing)
    source_run: str = ""
    default_temperature: float = 1.0  # like a Hugging Face generation_config.json: opencode sends none
    path: Path | None = None  # filled in when loaded from disk

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d

    @classmethod
    def from_json(cls, data: dict, path: Path | None = None) -> "ModelInfo":
        """From release.json or to_json() (what inference's GET /models returns), ignoring unknown keys."""
        known = {f.name for f in fields(cls)} - {"path"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["pricing"] = Pricing(**(data.get("pricing") or {}))
        return cls(**kwargs, path=path)


def load_model_info(model_dir: str | Path) -> ModelInfo:
    model_dir = Path(model_dir)
    return ModelInfo.from_json(json.loads((model_dir / "release.json").read_text()), model_dir)


def list_models(models_dir: str | Path) -> list[ModelInfo]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    infos = [load_model_info(d) for d in sorted(models_dir.iterdir()) if (d / "release.json").exists()]
    return sorted(infos, key=lambda m: m.created, reverse=True)


def get_model(models_dir: str | Path, model_id: str) -> ModelInfo | None:
    return next((m for m in list_models(models_dir) if m.id == model_id), None)


def write_release(model_dir: str | Path, info: ModelInfo) -> None:
    Path(model_dir, "release.json").write_text(json.dumps(info.to_json(), indent=2))
