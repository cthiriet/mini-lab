"""Evaluate a stage's checkpoint: writes runs/<run>/<stage>/eval.json and prints a table.

    uv run python -m minilab.eval.run --run runs/small --stage sft
    uv run python -m minilab.eval.run --run runs/small --summary     # every stage, one table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minilab.checkpoint import load_checkpoint
from minilab.data.tinystories import load_stories
from minilab.eval import tasks
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.train.pretrain import sample_story
from minilab.train.trainer import DEVICES, STAGES, load_config, resolve_device, setup


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _by_digits(records: list[dict], key: str = "correct") -> dict[str, float]:
    digits = sorted({r["digits"] for r in records})
    return {str(n): _mean(r[key] for r in records if r["digits"] == n) for n in digits}


def evaluate(model: GPT, tok: Tokenizer, cfg: dict, stage: str) -> dict:
    ec, d = cfg["eval"], cfg["data"]
    train_digits, heldout = d["digits"], d.get("heldout_digits", [])
    chat = stage != "pretrain"
    t0 = time.time()
    loss, ppl = tasks.perplexity(model, tok, load_stories("val", d["val_mb"]), ec.get("ppl_batches", 8))
    problems = [p for n in sorted(set(train_digits + heldout)) for p in tasks.arithmetic_problems(n, ec["n_per_digit"])]
    result = {"stage": stage, "mode": "chat" if chat else "completion", "val_loss": round(loss, 4),
              "val_ppl": round(ppl, 3), "train_digits": train_digits, "heldout_digits": heldout}
    if not chat:
        records = tasks.eval_arithmetic_completion(model, tok, problems)
        result["arithmetic"] = _by_digits(records)
        result["samples"] = [{"prompt": "Once upon a time", "response": sample_story(model, tok, model.wte.weight.device.type)}]
    else:
        max_new = ec.get("max_new_tokens", 200)
        records = tasks.eval_arithmetic_chat(model, tok, problems, max_new)
        result["arithmetic"] = _by_digits(records)
        hardest = tasks.arithmetic_problems(max(train_digits), ec["n_sampled"])
        sampled = tasks.eval_arithmetic_chat(model, tok, hardest, max_new, temperature=1.0)
        result["arithmetic_sampled"] = _mean(r["correct"] for r in sampled)
        tool_problems = [p for n in train_digits for p in tasks.arithmetic_problems(n, ec["n_tool"], tools=True)]
        tool = tasks.eval_tool_use(model, tok, tool_problems, max_new)
        result["tool_call"] = _mean(r["call_ok"] for r in tool)
        result["tool_answer"] = _mean(r["correct"] for r in tool)
        stories = tasks.eval_story_topics(model, tok)
        result["story_topic"] = _mean(r["on_topic"] for r in stories)
        instr = tasks.eval_instructions(model, tok, ec.get("n_instr", 30), model.config.block_size)  # stories
        scores = {k: _mean(r["ok"] for r in instr if r["kind"] == k) for k in tasks.INSTRUCTION_KINDS}
        in_scope = records + stories + [r for r in instr if r["kind"] not in ("refusal", "identity")]
        scores["over_refusal"] = _mean(r["refused"] for r in in_scope)
        result["instructions"] = scores
        result["instr"] = _mean([scores[k] for k in tasks.INSTRUCTION_KINDS] + [1 - scores["over_refusal"]])
        chats = tasks.eval_chat_prompts(model, tok)
        result["format"] = _mean(r["format_ok"] for r in records + sampled + tool + stories + instr + chats)
        result["samples"] = chats
    result["in_distribution"] = _mean(result["arithmetic"][str(n)] for n in train_digits)
    result["heldout"] = _mean(result["arithmetic"][str(n)] for n in heldout)
    result["eval_seconds"] = round(time.time() - t0, 1)
    return result


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def table(results: list[dict], times: dict[str, float] | None = None) -> str:
    """A compact text table, one row per evaluated stage."""
    digits = sorted({int(k) for r in results for k in r["arithmetic"]})
    heldout = set(results[0].get("heldout_digits", []))
    cols = ["stage", "ppl"] + [f"{n}d" + ("*" if n in heldout else "") for n in digits]
    hardest = max(results[0].get("train_digits", [0]))
    cols += [f"{hardest}d@T=1", "tool call", "tool ans", "story", "instr", "format"] + (["train time"] if times else [])
    rows = []
    for r in results:
        row = [r["stage"] + (" (base)" if r["mode"] == "completion" else ""), f"{r['val_ppl']:.2f}"]
        row += [_pct(r["arithmetic"].get(str(n))) for n in digits]
        row += [_pct(r.get(k)) for k in ("arithmetic_sampled", "tool_call", "tool_answer", "story_topic", "instr", "format")]
        if times:
            row.append(f"{times.get(r['stage'], 0) / 60:.1f} min")
        rows.append(row)
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    fmt = lambda row: "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))
    lines = [fmt(cols), fmt(["-" * w for w in widths]), *map(fmt, rows)]
    lines.append("* held-out digit counts (length generalization). base = raw-text prompt \"a + b =\". "
                 "instr = instruction following (see eval.json).")
    return "\n".join(lines)


def summary(run: Path) -> tuple[list[dict], dict[str, float]]:
    """eval.json and training wall-clock of every stage that has been evaluated."""
    results, times = [], {}
    for stage in STAGES:
        if (run / stage / "eval.json").exists():
            results.append(json.loads((run / stage / "eval.json").read_text()))
            meta = json.loads((run / stage / "config.json").read_text()).get("meta", {})
            times[stage] = meta.get("wall_clock_s", 0)
    return results, times


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a training stage.")
    p.add_argument("--run", required=True)
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--config", help="TOML config (default: <run>/config.toml)")
    p.add_argument("--device", default="auto", choices=DEVICES, help="auto: cuda, else mps, else cpu")
    p.add_argument("--summary", action="store_true", help="print the table of all evaluated stages")
    args = p.parse_args()
    run = Path(args.run)
    if args.summary:
        results, times = summary(run)
        print(table(results, times))
        return
    if not args.stage:
        p.error("--stage is required (or use --summary)")
    cfg = load_config(args)
    args.device = resolve_device(args.device, generation=True)
    setup(cfg.get("seed", 0), args.device)
    model, tok, _ = load_checkpoint(run / args.stage, device=args.device)
    result = evaluate(model, tok, cfg, args.stage)
    (run / args.stage / "eval.json").write_text(json.dumps(result, indent=2))
    print(table([result]))
    if "instructions" in result:
        print("instructions: " + ", ".join(f"{k} {_pct(v)}" for k, v in result["instructions"].items()))
    for s in result.get("samples", [])[:3]:
        print(f"\n> {s['prompt']}\n{s['response'][:400]}")
    print(f"\neval: {result['eval_seconds']:.0f}s -> {run / args.stage / 'eval.json'}")


if __name__ == "__main__":
    main()
