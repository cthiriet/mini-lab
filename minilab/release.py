"""Release a trained checkpoint as a servable model.

    uv run python -m minilab.release --run runs/small --stage rl --id mini-1

Copies the checkpoint to models/<id>/ (or $MINILAB_MODELS_DIR/<id>/) and adds
release.json (id, pricing, context length -- what the API serves and bills),
eval.json and MODEL_CARD.md. The inference server picks up every directory of
models/ that has a release.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from minilab.eval.model_card import model_card
from minilab.registry import ModelInfo, Pricing, write_release
from minilab.settings import get_settings
from minilab.train.trainer import STAGES

PRICING = Pricing(input_per_1m=0.50, output_per_1m=1.50)  # USD per 1M tokens


def release(run: Path, stage: str, model_id: str, models_dir: Path) -> Path:
    src, out = Path(run) / stage, Path(models_dir) / model_id
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "config.json", "tokenizer.json", "eval.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    ckpt = json.loads((src / "config.json").read_text())
    params = ckpt["meta"].get("params", 0)
    write_release(out, ModelInfo(
        id=model_id,
        created=int(time.time()),
        description=f"{params / 1e6:.1f}M-parameter GPT trained from scratch on a laptop: "
                    "short stories and addition (step-by-step reasoning or calculator tool).",
        context_length=ckpt["model"]["block_size"],
        pricing=PRICING,
        source_run=str(run),
    ))
    (out / "MODEL_CARD.md").write_text(model_card(Path(run), stage, model_id))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Release a checkpoint to the models directory.")
    p.add_argument("--run", required=True)
    p.add_argument("--stage", default="rl", choices=STAGES)
    p.add_argument("--id", required=True, help="model id, e.g. mini-1")
    p.add_argument("--models-dir", default=get_settings().models_dir)
    args = p.parse_args()
    out = release(Path(args.run), args.stage, args.id, Path(args.models_dir))
    print(f"released {args.run}/{args.stage} as {args.id} -> {out}")


if __name__ == "__main__":
    main()
