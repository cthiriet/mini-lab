"""Evaluate a stage's checkpoint: writes runs/<run>/<stage>/eval.json and prints a table.

    uv run python -m minilab.eval.run --run runs/prelude --stage sft
    uv run python -m minilab.eval.run --run runs/prelude --summary     # every stage, one table
    uv run python -m minilab.eval.run --run runs/prelude --stage sft --story-math   # only the story problems
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from minilab.checkpoint import load_checkpoint
from minilab.data import story_math
from minilab.data.tinystories import load_stories
from minilab.eval import code as code_eval
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


LEVELS = ["low", "high"]
# Held-out story sets (docs/story-math-spec.md §7.2): name -> (problems, events, split, `half` in the problem?, distractor)
HELDOUT_SETS = {"steps78": (200, [7, 8], "heldout", None, None),
                "objects": (200, [1, 2, 3, 4, 5, 6], "heldout", False, None),  # new names and things, no `half`
                "ops": (100, [2, 3, 4, 5], "heldout", True, None),              # `half` appears
                "distract": (200, [3, 4, 5, 6], "train", None, True)}           # always a distractor
# Modes of the held-out sets: name -> (level, think)
HELDOUT_MODES = {"adaptive high": ("high", None), "off": ("high", False)}


def _heldout_problems(seed: int, name: str, scale: float) -> list[dict]:
    n, ks, split, half, distractor = HELDOUT_SETS[name]
    rng = random.Random(f"{seed}-{name}")
    problems = []
    for i in range(max(1, round(n * scale))):
        while True:  # `make` cannot leave `half` out of the held-out split, nor force it in: draw again
            p = story_math.make(rng, ks[i % len(ks)], split, distractor)
            if half is None or half == any(op == "half" for op, _ in p["ops"]):
                break
        problems.append(p)
    return problems


def evaluate_story_math(model: GPT, tok: Tokenizer, cfg: dict, n_per_step: int | None = None) -> dict:
    """The story problems (eval/tasks.py:eval_story_math): 7 modes by number of events k, then the held-out
    sets. Returns eval.json's "story_math" and its scalars. The reasoning's sums are computed for the
    model (inference/calculator.py), except in the "adaptive high nocalc" reference. n_per_step (default: the config's) scales every
    set, for a quicker look."""
    ec, seed = cfg["eval"]["story_math"], cfg["data"]["reason"]["eval_seed"]
    n = n_per_step or ec["n_per_step"]
    max_new = ec.get("max_new_tokens", 450)  # a high-effort reasoning on 8 events is up to ~430 tokens
    problems = []
    for k in ec["steps"]:
        rng = random.Random(f"{seed}-{k}")
        problems += [story_math.make(rng, k, "train") for _ in range(n)]
    modes = {f"{kind} {level}": (level, think) for level in LEVELS for kind, think in (("adaptive", None), ("forced", True))}
    modes["off"] = ("high", False)
    records = {m: tasks.eval_story_math(model, tok, problems, max_new, level, think) for m, (level, think) in modes.items()}
    records["adaptive high nocalc"] = tasks.eval_story_math(model, tok, problems, max_new, "high", calculator=False)  # reference
    sm = {key: {m: _by_digits(r, field) for m, r in records.items()}
          for key, field in (("accuracy", "correct"), ("think_rate", "thinking"), ("tokens", "reasoning_tokens"))}
    sm["style_ok"], sm["direct_acc"], sm["regret"] = {}, {}, {}
    for level in LEVELS:
        adaptive, acc = records[f"adaptive {level}"], sm["accuracy"]
        sm["style_ok"][level] = _mean(r["style_ok"] for r in adaptive + records[f"forced {level}"] if r["thinking"])
        sm["direct_acc"][level] = _mean(r["correct"] for r in adaptive if not r["thinking"])
        sm["regret"][level] = {k: a - max(acc[f"forced {level}"][k], acc["off"][k]) for k, a in acc[f"adaptive {level}"].items()}
    sm["heldout"] = {}
    for name in HELDOUT_SETS:
        held = _heldout_problems(seed, name, n / ec["n_per_step"])
        sm["heldout"][name] = {}
        for m, (level, think) in HELDOUT_MODES.items():
            rs = tasks.eval_story_math(model, tok, held, max_new, level, think)
            sm["heldout"][name][m] = {"accuracy": _mean(r["correct"] for r in rs), "think_rate": _mean(r["thinking"] for r in rs),
                                      "tokens": _mean(r["reasoning_tokens"] for r in rs)}
    scalars = {f"sm_{level}": _mean(sm["accuracy"][f"adaptive {level}"].values()) for level in LEVELS}
    scalars["sm_high_nocalc"] = _mean(sm["accuracy"]["adaptive high nocalc"].values())
    return {"story_math": sm, **scalars, "sm_6st": sm["accuracy"]["adaptive high"].get("6")}


def evaluate(model: GPT, tok: Tokenizer, cfg: dict, stage: str) -> dict:
    ec, d = cfg["eval"], cfg["data"]
    train_digits, heldout = d["digits"], d.get("heldout_digits", [])
    chat = stage != "pretrain"
    t0 = time.time()
    val_stories = load_stories("val", d["val_mb"])
    loss, ppl = tasks.perplexity(model, tok, val_stories, ec.get("ppl_batches", 8))
    problems = [p for n in sorted(set(train_digits + heldout)) for p in tasks.arithmetic_problems(n, ec["n_per_digit"])]
    result = {"stage": stage, "mode": "chat" if chat else "completion", "val_loss": round(loss, 4),
              "val_ppl": round(ppl, 3), "val_bpc": round(tasks.bits_per_char(tok, val_stories[:1000], loss), 4),
              "train_digits": train_digits, "heldout_digits": heldout}
    if not chat:
        records = tasks.eval_arithmetic_completion(model, tok, problems)
        result["arithmetic"] = _by_digits(records)
        result["samples"] = [{"prompt": "Once upon a time", "response": sample_story(model, tok, model.wte.weight.device.type)}]
    else:
        max_new = ec.get("max_new_tokens", 200)
        records = tasks.eval_arithmetic_chat(model, tok, problems, max_new)
        result["arithmetic"] = _by_digits(records)
        forced = tasks.eval_arithmetic_chat(model, tok, problems, max_new, think=True)
        off = tasks.eval_arithmetic_chat(model, tok, problems, max_new, think=False)
        for r in records:
            r["thinks"] = r["thinking"] > 0
        result["thinking"] = {  # adaptive thinking, per digit count (see eval/tasks.py)
            "rate": _by_digits(records, "thinks"), "tokens": _by_digits(records, "thinking"),
            "accuracy_forced": _by_digits(forced, "correct"), "accuracy_off": _by_digits(off, "correct"),
            "tokens_forced": _by_digits(forced, "thinking")}
        result["think_saved"] = 1 - _mean(r["thinking"] for r in records) / max(_mean(r["thinking"] for r in forced), 1)
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
        in_scope = records + stories + [r for r in instr if r["kind"] not in ("refusal", "identity")
                                        and r["inner"] not in ("refusal", "identity")]
        scores["over_refusal"] = _mean(r["refused"] for r in in_scope)
        result["instructions"] = scores
        result["instr"] = _mean([scores[k] for k in tasks.INSTRUCTION_KINDS] + [1 - scores["over_refusal"]])
        rng = random.Random(f"{tasks.EVAL_SEED}-chat")
        chat = tasks.eval_chat(model, tok, [tasks.chat_script(rng, train_digits) for _ in range(ec.get("n_chat", 200))])
        result["chat"] = _mean(r["ok"] for r in chat)
        result["chat_failures"] = dict(Counter(r["failed"] for r in chat if r["failed"]).most_common())
        chats = tasks.eval_chat_prompts(model, tok)
        result["format"] = _mean(r["format_ok"] for r in records + sampled + tool + stories + instr + chats)
        result["samples"] = chats
        if "story_math" in ec:
            result |= evaluate_story_math(model, tok, cfg)
    result["in_distribution"] = _mean(result["arithmetic"][str(n)] for n in train_digits)
    result["heldout"] = _mean(result["arithmetic"][str(n)] for n in heldout)
    if "code_eval" in cfg:  # the coding agent, in a Docker sandbox (the unit tests' configs leave it out)
        result["code"] = code_eval.evaluate(model, tok, cfg, stage)
    result["eval_seconds"] = round(time.time() - t0, 1)
    return result


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


TABLE_FOOTNOTE = ("* held-out digit counts (length generalization). base = raw-text prompt \"a + b =\". "
                  "think saved = thinking tokens the model skips on additions, against always thinking. "
                  "sm = story problems (adaptive at low/high effort, 6st = 6 events, think/tok = thinking rate and "
                  "tokens, 7-8 and obj = held-out events and names). "
                  "instr = instruction following, chat = whole conversations right. agent = coding tasks done end "
                  "to end in opencode's format, valid calls = tool calls opencode accepts (see eval.json).")


def _levels(result: dict, key: str, scale: int = 1, suffix: str = "") -> str:
    """The story problems' `key` (think_rate, tokens), adaptive, averaged over k, at low/high: "12/71%"."""
    sm = result.get("story_math")
    values = [_mean(sm[key][f"adaptive {level}"].values()) if sm else None for level in LEVELS]
    return "-" if sm is None else "/".join("-" if v is None else f"{scale * v:.0f}" for v in values) + suffix


def _table_cells(results: list[dict], times: dict[str, float] | None) -> tuple[list[str], list[list[str]]]:
    """Column names and one row of cells per evaluated stage."""
    digits = sorted({int(k) for r in results for k in r["arithmetic"]})
    heldout = set(results[0].get("heldout_digits", []))
    cols = ["stage", "ppl"] + [f"{n}d" + ("*" if n in heldout else "") for n in digits]
    hardest = max(results[0].get("train_digits", [0]))
    cols += [f"{hardest}d@T=1", "think saved"]
    sm = any("story_math" in r for r in results)  # an older eval.json has none
    held = sm and any("heldout" in r.get("story_math", {}) for r in results)
    cols += (["sm lo", "sm hi", "sm 6st", "think L/H", "tok L/H"] if sm else []) + (["sm 7-8*", "sm obj*"] if held else [])
    cols += ["tool call", "tool ans", "story", "instr", "chat", "format"]
    coding = "code" in results[0]
    cols += (["agent", "valid calls"] if coding else []) + (["train time"] if times else [])
    rows = []
    for r in results:
        row = [r["stage"] + (" (base)" if r["mode"] == "completion" else ""), f"{r['val_ppl']:.2f}"]
        row += [_pct(r["arithmetic"].get(str(n))) for n in digits]
        row += [_pct(r.get(k)) for k in ("arithmetic_sampled", "think_saved")]
        if sm:
            row += [_pct(r.get(k)) for k in ("sm_low", "sm_high", "sm_6st")]
            row += [_levels(r, "think_rate", 100, "%"), _levels(r, "tokens")]
        if held:
            row += [_pct(r.get("story_math", {}).get("heldout", {}).get(name, {}).get("adaptive high", {}).get("accuracy"))
                    for name in ("steps78", "objects")]
        row += [_pct(r.get(k)) for k in ("tool_call", "tool_answer", "story_topic", "instr", "chat", "format")]
        if coding:
            row += [_pct(code_eval.get(r, "code.agent")), _pct(code_eval.get(r, "code.valid_calls"))]
        if times:
            row.append(f"{times.get(r['stage'], 0) / 60:.1f} min")
        rows.append(row)
    return cols, rows


def table(results: list[dict], times: dict[str, float] | None = None) -> str:
    """A compact text table, one row per evaluated stage."""
    cols, rows = _table_cells(results, times)
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    fmt = lambda row: "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))
    lines = [fmt(cols), fmt(["-" * w for w in widths]), *map(fmt, rows)]
    lines.append(TABLE_FOOTNOTE)
    return "\n".join(lines)


def markdown_table(results: list[dict], times: dict[str, float] | None = None) -> str:
    """The same table in Markdown (for model cards), numbers right-aligned."""
    cols, rows = _table_cells(results, times)
    lines = ["| " + " | ".join(cols) + " |", "|---|" + "---:|" * (len(cols) - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n\n" + TABLE_FOOTNOTE.replace("* held-out", "\\* held-out")


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
    p.add_argument("--story-math", action="store_true", help="only the story problems; updates eval.json")
    p.add_argument("--n-per-step", type=int, help="story problems per number of events (a quicker look)")
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
    if args.story_math:
        path = run / args.stage / "eval.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        result |= evaluate_story_math(model, tok, cfg, args.n_per_step)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k.startswith("sm_")} | {"style_ok": result["story_math"]["style_ok"],
                          "direct_acc": result["story_math"]["direct_acc"], "regret": result["story_math"]["regret"]}, indent=1))
        return
    result = evaluate(model, tok, cfg, args.stage)
    (run / args.stage / "eval.json").write_text(json.dumps(result, indent=2))
    print(table([result]))
    if "thinking" in result:
        th = result["thinking"]
        print("thinking (per digit count): " + "; ".join(
            f"{n}d {_pct(th['rate'][n])} think, {th['tokens'][n]:.0f} tok, forced {_pct(th['accuracy_forced'][n])}, "
            f"off {_pct(th['accuracy_off'][n])}" for n in th["rate"]))
    if "instructions" in result:
        print("instructions: " + ", ".join(f"{k} {_pct(v)}" for k, v in result["instructions"].items()))
    if result.get("code", {}).get("tasks"):
        print("tasks: " + ", ".join(f"{k} {_pct(v)}" for k, v in result["code"]["tasks"].items()))
    for s in result.get("samples", [])[:3]:
        print(f"\n> {s['prompt']}\n{s['response'][:400]}")
    print(f"\neval: {result['eval_seconds']:.0f}s -> {run / args.stage / 'eval.json'}")


if __name__ == "__main__":
    main()
