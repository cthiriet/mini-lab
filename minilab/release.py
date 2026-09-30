"""Release a trained checkpoint as a servable model.

    uv run python -m minilab.release --run runs/small --stage distill --id mini-3.2

Copies the checkpoint to models/<id>/ (or $MINILAB_MODELS_DIR/<id>/) and adds
release.json (id, pricing, context length -- what the API serves and bills),
eval.json and MODEL_CARD.md. The inference server picks up every directory of
models/ that has a release.json.

First, the release gate (minilab.eval.gate): the newest earlier release is evaluated
again, and a regression on any metric blocks the release, unless it is waived with a
reason (--allow METRIC=REASON, recorded in gate.json and the model card).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

from minilab.eval import gate
from minilab.eval.model_card import model_card
from minilab.registry import ModelInfo, Pricing, write_release
from minilab.settings import get_settings
from minilab.train.trainer import DEVICES, STAGES, resolve_device

# USD per 1M tokens: the frontier flagship tier (Claude Fable 5.1, GPT-6 Astra), so that a
# tiny model's tiny answers still cost something you can see: about a cent per story.
PRICING = Pricing(input_per_1m=10.0, output_per_1m=50.0)


def release(run: Path, stage: str, model_id: str, models_dir: Path, gate_result: dict | None = None) -> Path:
    src, out = Path(run) / stage, Path(models_dir) / model_id
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "config.json", "tokenizer.json", "eval.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    ckpt = json.loads((src / "config.json").read_text())
    params = ckpt["meta"].get("params", 0)
    rc = tomllib.loads((Path(run) / "config.toml").read_text()).get("release", {})
    write_release(out, ModelInfo(
        id=model_id,
        created=int(time.time()),
        description=f"{params / 1e6:.1f}M-parameter GPT trained from scratch on a laptop: "
                    + rc.get("description", "short stories and addition (step-by-step reasoning or calculator tool)."),
        context_length=ckpt["model"]["block_size"],
        pricing=PRICING,
        source_run=str(run),
        family=rc.get("family", "mini"),
        truncation=rc.get("truncation", "disabled"),
        default_temperature=rc.get("temperature", 1.0),
    ))
    card = model_card(Path(run), stage, model_id)
    if gate_result:
        (out / "gate.json").write_text(json.dumps({k: v for k, v in gate_result.items() if k != "baseline_eval"}, indent=2))
        waived = "; ".join(f"{m} ({reason})" for m, reason in gate_result["waived"].items())
        card += (f"\n## Release gate\n\nEvaluated again next to `{gate_result['baseline']}`, with the same eval: "
                 + (f"regressions shipped anyway: {waived}." if waived else "no regression.") + "\n")
    (out / "MODEL_CARD.md").write_text(card)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Release a checkpoint to the models directory.")
    p.add_argument("--run", required=True)
    p.add_argument("--stage", default="distill", choices=STAGES)
    p.add_argument("--id", required=True, help="model id, e.g. mini-3.2 (see the releases in docs/training.md)")
    p.add_argument("--models-dir", default=get_settings().models_dir)
    p.add_argument("--baseline", help="the release to beat (default: the newest earlier one)")
    p.add_argument("--allow", action="append", default=[], metavar="METRIC=REASON",
                   help="ship despite a regression on METRIC, for a reason recorded with the release")
    p.add_argument("--no-gate", action="store_true", help="skip the release gate")
    p.add_argument("--device", default="auto", choices=DEVICES, help="for evaluating the baseline")
    args = p.parse_args()
    run, models_dir = Path(args.run), Path(args.models_dir)
    gate_result = None
    family = tomllib.loads((run / "config.toml").read_text()).get("release", {}).get("family", "mini")
    baseline = Path(args.baseline) if args.baseline else gate.newest_release(models_dir, exclude=args.id, family=family)
    if args.no_gate or baseline is None:
        print("release gate: " + ("skipped" if args.no_gate else "no earlier release to compare with"))
    else:
        cfg = tomllib.loads((run / "config.toml").read_text())
        new = json.loads((run / args.stage / "eval.json").read_text())
        gate_result = gate.check(new, baseline, cfg, resolve_device(args.device, generation=True),
                                 gate.parse_waivers(args.allow), args.stage)
        print(gate.report(gate_result))
        if not gate_result["passed"]:
            sys.exit("not released: fix the regressions, or ship anyway with --allow METRIC=REASON")
    out = release(run, args.stage, args.id, models_dir, gate_result)
    print(f"released {args.run}/{args.stage} as {args.id} -> {out}")


if __name__ == "__main__":
    main()
