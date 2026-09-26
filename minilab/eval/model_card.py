"""Write MODEL_CARD.md for a trained model: what it is, how it was trained (data,
steps, tokens, wall-clock and hardware per stage) and what every stage changed
(the eval table).

    uv run python -m minilab.eval.model_card --run runs/small --stage rl > MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from minilab.eval.run import markdown_table, summary
from minilab.train.trainer import STAGES


def _stage_meta(run: Path) -> dict[str, dict]:
    metas = {}
    for stage in STAGES:
        path = run / stage / "config.json"
        if path.exists():
            metas[stage] = json.loads(path.read_text())["meta"]
    return metas


def _digits(digits: list[int]) -> str:
    return f"{min(digits)}-{max(digits)}" if len(digits) > 1 else str(digits[0])


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 60 else f"{seconds / 60:.1f} min"


def model_card(run: Path, stage: str, model_id: str | None = None) -> str:
    run = Path(run)
    cfg = tomllib.loads((run / "config.toml").read_text())
    ckpt = json.loads((run / stage / "config.json").read_text())
    m, d = ckpt["model"], cfg["data"]
    metas = _stage_meta(run)
    final = metas[stage]
    total = sum(meta.get("wall_clock_s", 0) for meta in metas.values())
    name = model_id or f"{run.name}-{stage}"
    results, times = summary(run)

    lines = [
        f"# {name}",
        "",
        f"{name} is a {final['params'] / 1e6:.1f}M-parameter GPT trained from scratch by the mini-lab "
        f"training pipeline (pretrain -> midtrain -> SFT -> RL), in {_duration(total)} of training on "
        f"a laptop ({final['hardware']}). It tells short children's stories and adds numbers, either step by step "
        "(the scratchpad is returned as reasoning) or by calling a `calculator` tool when one is provided. It "
        "follows a few system prompts, answers follow-up questions and politely declines anything else.",
        "",
        "## Model",
        "",
        "| | |",
        "|---|---|",
        f"| parameters | {final['params']:,} |",
        f"| layers / heads / width | {m['n_layer']} / {m['n_head']} / {m['n_embd']} |",
        f"| context length | {m['block_size']} tokens |",
        f"| vocabulary | {m['vocab_size']} (byte-level BPE, digits always split) |",
        "| architecture | decoder-only transformer, RoPE, RMSNorm, GELU, tied embeddings |",
        f"| checkpoint | `{run / stage}` |",
        "",
        "## Training",
        "",
        "| stage | steps | tokens | wall-clock | device | final loss |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for s, meta in metas.items():
        loss = meta.get("final_loss")
        lines.append(f"| {s} | {meta.get('steps', 0):,} | {meta.get('tokens', 0) / 1e6:.2f}M | "
                     f"{_duration(meta.get('wall_clock_s', 0))} | {meta.get('device')} | "
                     f"{'-' if loss is None else f'{loss:.3f}'} |")
    lines += [
        f"| **total** | | {final.get('tokens_total', 0) / 1e6:.2f}M | {_duration(total)} | | |",
        "",
        f"Hardware: {final['hardware']}. RL tokens are the sampled completion tokens trained on.",
        "",
        "## Data",
        "",
        f"- **Pretraining**: the first {d['train_mb']:g} MB of TinyStoriesV2-GPT4 (roneneldan/TinyStories), "
        f"with synthetic arithmetic worksheets mixed in ({cfg['pretrain']['arith_frac']:.0%} of documents): "
        f"equations, sentences, word problems and worked column additions, operands of {_digits(d['digits'])} digits.",
        "- **Midtraining** (format and skills, at volume): single-turn conversations, no system prompt: addition "
        "questions in many phrasings answered with a scratchpad (or a calculator call when tools are enabled), "
        f"operands of {_digits(cfg.get('midtrain', {}).get('digits', d['digits']))} digits; story requests answered "
        "with TinyStories stories; greetings.",
        f"- **SFT** (behavior): a fixed set of {cfg.get('sft', {}).get('size', 0):,} conversations: system prompts to "
        "obey (number only, no calculator, one sentence, start with \"Sure!\"), follow-up questions about an earlier "
        "answer, identity, polite refusals of out-of-scope requests, and plain conversations.",
        f"- **RL**: addition questions with {_digits(cfg['rl']['digits'])}-digit operands (including lengths the "
        "chat data never showed), plus calculator, instruction and refusal problems; reward 1 when the answer is "
        "exactly right and in the requested form.",
        f"- Operands of {_digits(d.get('heldout_digits') or [0])} digits are never seen in training (length generalization).",
        "",
        "## Evaluation",
        "",
        "Same fixed-seed eval set for every stage (`uv run python -m minilab.eval.run`):",
        "",
        markdown_table(results, times),
        "",
        "- `ppl`: perplexity on held-out TinyStories stories.",
        "- `Nd`: greedy answers to N-digit additions that are exactly `The answer is c.` (chat format; the base model "
        "gets the raw-text prompt `a + b =` and must continue with the sum).",
        "- `Nd@T=1`: the hardest in-distribution digit count, sampling at temperature 1 (the API default).",
        "- `tool call`: with `tools=[calculator]`, the first turn is a calculator call with a correct expression; "
        "`tool ans`: after the tool result, the final answer is correct.",
        "- `story`: \"Tell me a story about a dog.\" (15 topics x 3 phrasings) -> the story mentions the topic.",
        "- `instr`: instruction following, the mean of: system prompts obeyed (number only, no calculator, one "
        "sentence, \"Sure!\"), follow-up questions answered, held-out out-of-scope questions refused, identity, "
        "and in-scope requests *not* refused.",
        "- `format`: fraction of assistant turns properly ended (and no tool call without tools).",
        "",
    ]
    samples = next((r.get("samples", []) for r in results if r["stage"] == stage), [])
    if samples:
        lines += ["## Samples (greedy)", ""]
        for sample in samples:
            response = sample["response"] or json.dumps(sample.get("tool_calls") or "(empty)")
            lines += [f"> **{sample['prompt']}**", ">", *[f"> {line}" for line in response.splitlines()], ""]
    lines += [
        "## Chat format",
        "",
        "OpenAI-style messages rendered with special tokens (see `minilab/tokenizer/chat.py`). Addition is "
        "answered with a scratchpad inside `<|think_start|>...<|think_end|>` (column by column, right to left), "
        "then `The answer is c.` With `tools: calculator`, the model emits "
        '`{"name": "calculator", "arguments": {"expression": "a + b"}}` and answers after the tool result.',
        "",
        "## Limitations",
        "",
        "- A toy: it only knows simple children's stories and addition. Anything else gets a polite refusal "
        "at best and nonsense at worst. It has no world knowledge.",
        "- Stories are often repetitive or incoherent after a few sentences; the context is only "
        f"{m['block_size']} tokens.",
        "- Addition is reliable only for the operand lengths it was trained on; see the held-out columns.",
        "- English only. Not for any real use.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Print a model card for a run's stage.")
    p.add_argument("--run", required=True)
    p.add_argument("--stage", default="rl", choices=STAGES)
    p.add_argument("--id", help="model id used as the title")
    args = p.parse_args()
    print(model_card(Path(args.run), args.stage, args.id))


if __name__ == "__main__":
    main()
