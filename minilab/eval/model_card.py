"""Write MODEL_CARD.md for a trained model: what it is, how it was trained (data,
steps, tokens, wall-clock and hardware per stage) and what every stage changed
(the eval table).

    uv run python -m minilab.eval.model_card --run runs/small --stage distill > MODEL_CARD.md
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


STAGE_NAMES = {"sft": "SFT", "rl": "RL", "rl_math": "RL math specialist", "rl_code": "RL code specialist",
               "distill": "distillation"}


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
    pipeline = " -> ".join(STAGE_NAMES.get(s, s) for s in metas)
    block = ["SwiGLU" if m.get("mlp") == "swiglu" else "GELU"] + (["QK-norm"] if m.get("qk_norm") else []) + \
        (["gated attention"] if m.get("attn_gate") else [])
    code = d.get("world") == "code"
    unified = d.get("world") == "unified"
    what = (CODE_WHAT if code else
            "It tells short children's stories and adds numbers, either step by step "
            "(the scratchpad is returned as reasoning) or by calling a `calculator` tool when one is provided. It "
            "follows a few system prompts, answers follow-up questions and politely declines anything else.")
    if unified:
        what += " In [opencode](https://opencode.ai), the same model is a coding agent: " + CODE_WHAT.split(": ", 1)[1]

    lines = [
        f"# {name}",
        "",
        f"{name} is a {final['params'] / 1e6:.1f}M-parameter GPT trained from scratch by the mini-lab "
        f"training pipeline ({pipeline}), in {_duration(total)} of training on "
        f"a laptop ({final['hardware']}). {what}",
        "",
        "## Model",
        "",
        "| | |",
        "|---|---|",
        f"| parameters | {final['params']:,} |",
        f"| layers / heads / width | {m['n_layer']} / {m['n_head']} / {m['n_embd']} |",
        f"| context length | {m['block_size']} tokens |",
        f"| vocabulary | {m['vocab_size']} (byte-level BPE, digits always split) |",
        f"| architecture | decoder-only transformer, RoPE, RMSNorm, {', '.join(block)}, tied embeddings |",
        f"| optimizer | {'Muon (hidden matrices) + AdamW' if cfg.get('optimizer') == 'muon' else 'AdamW'} |",
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
        f"Hardware: {final['hardware']}." + (" RL and distillation tokens are the sampled answer tokens trained on."
                                            if {"rl", "rl_math", "distill"} & set(metas) else ""),
        "",
    ]
    if code:
        lines += _code_sections(cfg, m, results, times, stage)
        return "\n".join(lines)
    lines += [
        "## Data",
        "",
        f"- **Pretraining**: the first {d['train_mb']:g} MB of TinyStoriesV2-GPT4 (roneneldan/TinyStories), "
        f"with synthetic arithmetic worksheets mixed in ({cfg['pretrain']['arith_frac']:.0%} of documents): "
        f"equations, sentences, word problems and worked column additions, operands of {_digits(d['digits'])} digits"
        + (" - and Python documents of the toy code world of `minilab/data/code.py` (the files of small projects with "
           "what their scripts print, functions with what they do, bugs with their fixes)." if unified else "."),
        "- **Midtraining** (format and skills, at volume): single-turn conversations, no system prompt: addition "
        "questions in many phrasings answered with a scratchpad (or a calculator call when tools are enabled), "
        f"operands of {_digits(cfg.get('midtrain', {}).get('digits', d['digits']))} digits; story requests answered "
        "with TinyStories stories; greetings.",
        f"- **SFT** (behavior): a fixed set of {cfg.get('sft', {}).get('size', 0):,} conversations: system prompts to "
        "obey (number only, no calculator, one sentence, start with \"Sure!\"), follow-up questions about an earlier "
        "answer, new requests after an answer (another addition, or something else), identity, polite refusals of "
        "out-of-scope requests, and plain conversations."
        + (f" Mixed with {cfg['sft'].get('code_size', 0):,} transcripts of opencode sessions "
           f"({cfg['sft'].get('code_frac', 0):.0%} of the rows): a request about a random project, solved by an "
           "oracle with opencode's tools played for real in a sandbox (explore, create, modify, repair); and "
           f"with pretraining documents ({cfg['sft'].get('text_frac', 0):.0%} of the rows, loss on every token), "
           "without which the long SFT made the model forget plain text."
           if unified else ""),
        *([f"- **RL (math specialist)**: addition questions with {_digits(cfg['rl_math']['digits'])}-digit operands "
           "(including lengths the chat data never showed) and calculator problems; reward 1 when the answer is "
           "exactly right and in the requested form.",
           "- **Distillation**: the SFT model answers addition, calculator, instruction and refusal problems, and "
           "learns the next-token distributions of the math specialist (additions, calculator) and of the SFT "
           "model (everything else" + (", one turn of an agent transcript included" if unified else "")
           + ") on its own answers."] if "distill" in metas else []),
        *([f"- **RL**: addition questions with {_digits(cfg['rl']['digits'])}-digit operands (including lengths the "
           "chat data never showed), plus calculator, instruction and refusal problems; reward 1 when the answer is "
           "exactly right and in the requested form."] if "rl" in metas else []),
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
        "- `Nd@T=1`: the hardest in-distribution digit count, sampling at temperature 1 (OpenAI's default).",
        "- `tool call`: with `tools=[calculator]`, the first turn is a calculator call with a correct expression; "
        "`tool ans`: after the tool result, the final answer is correct.",
        "- `story`: \"Tell me a story about a dog.\" (15 topics x 3 phrasings) -> the story mentions the topic.",
        "- `instr`: instruction following, the mean of: system prompts obeyed (number only, no calculator, one "
        "sentence, \"Sure!\"), follow-up questions answered, held-out out-of-scope questions refused, identity, "
        "and in-scope requests *not* refused.",
        "- `format`: fraction of assistant turns properly ended (and no tool call without tools).",
        *(["- `agent`: coding tasks done end to end in opencode's format (13 kinds, "
           f"{cfg['code_eval']['n_per_kind']} each), the model's tool calls run in a locked-down Docker container; "
           "`valid calls`: tool calls opencode accepts."] if unified else []),
        "",
    ]
    coding = next((r for r in results if r["stage"] == stage), {}).get("code", {}).get("tasks")
    if unified and coding:
        lines += ["| coding task | success |", "|---|---:|"]
        lines += [f"| {k} | {100 * v:.0f}% |" for k, v in coding.items() if v is not None]
        lines.append("")
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
        + ("`calculator<|arg|>expression=a + b` between `<|tool_call_start|>` and `<|tool_call_end|>` (the \"code\" "
           "template, whose raw arguments let an agent write code without JSON escaping; opencode's system prompt is "
           "cut to its first sentence and long sessions are fitted into the context by the server)"
           if unified else '`{"name": "calculator", "arguments": {"expression": "a + b"}}`')
        + " and answers after the tool result.",
        "",
        "## Limitations",
        "",
        "- A toy: it only knows simple children's stories and addition"
        + (", and in opencode the tiny Python projects of its training world" if unified else "")
        + ". Anything else gets a polite refusal at best and nonsense at worst. It has no world knowledge.",
        *(["- In opencode it runs whatever command it decides to: only use it in a sandbox (`examples/opencode`)."]
          if unified else []),
        "- Stories are often repetitive or incoherent after a few sentences; the context is only "
        f"{m['block_size']} tokens.",
        "- Addition is reliable only for the operand lengths it was trained on; see the held-out columns.",
        "- English only. Not for any real use.",
        "",
    ]
    return "\n".join(lines)


CODE_WHAT = ("It is a coding agent for [opencode](https://opencode.ai): given a request about a small Python project "
             "(\"Run the tests and fix any bug\", \"Rename add to plus\"), it calls opencode's tools (`read`, `write`, "
             "`edit`, `glob`, `grep`, `shell`) until the job is done, then says what it did. Its world is tiny: "
             "projects of a few files and little functions (`add`, `greet`, `reverse`...).")


def _code_sections(cfg: dict, m: dict, results: list[dict], times: dict, stage: str) -> list[str]:
    """mini-code's data, eval, chat format and limitations."""
    sc = cfg["sft"]
    lines = [
        "## Data",
        "",
        "Everything is synthetic, from the toy code world of `minilab/data/code.py`:",
        "",
        "- **Pretraining** (Python, no chat): the files of random small projects (modules of little functions, a "
        "script that prints some calls, a test file of asserts) with what their scripts print, functions with what "
        "they do in English, and bugs with their fixes.",
        f"- **SFT** (the agent): {sc['size']:,} transcripts of opencode sessions. Each request comes with a random "
        "project, and an oracle solves it with opencode's tools, *played for real* in a sandbox: every tool result "
        "is exactly what opencode 2 returns (formats captured from opencode 2.0.20). Four families of tasks: explore "
        "(list, find, read, explain, run, test), create (a function from its description, a script), modify "
        "(rename across files, change a constant, add a function) and repair (a failing test, a crashing script); "
        "plus two requests in a row, small talk, identity, out-of-scope requests and opencode's title requests.",
        "",
        "## Evaluation",
        "",
        "Same fixed-seed eval for every stage (`uv run python -m minilab.eval.run`): new tasks of each of the 13 "
        f"kinds ({cfg['eval']['n_per_kind']} each), played end to end like opencode: the model's tool calls run in a "
        "locked-down Docker container (no network, read-only system), and each task is checked on the project's "
        "files and the answer (the tests pass, the new function works, the answer has the output...).",
        "",
        markdown_table(results, times),
        "",
    ]
    final = next((r for r in results if r["stage"] == stage), {})
    if final.get("tasks"):
        lines += ["| task | success |", "|---|---:|"]
        lines += [f"| {k} | {100 * v:.0f}% |" for k, v in final["tasks"].items() if v is not None]
        lines.append("")
    lines += [
        "## Chat format",
        "",
        "OpenAI-style messages rendered with the \"code\" template of `minilab/tokenizer/chat.py`. Tool calls carry "
        "their arguments raw, so code needs no JSON escaping: "
        "`<|tool_call_start|>edit<|arg|>path=calc.py<|arg|>oldString=    return a - b<|arg|>newString=    return a + b"
        "<|tool_call_end|>`; the API returns them as ordinary `tool_calls`, typed with the request's JSON schemas. "
        "opencode's ~20k-character system prompt is cut to its first sentence, paths are shown relative to the "
        f"working directory, and the server fits long sessions into the {m['block_size']}-token context by dropping "
        "the oldest turns (`truncation: auto` in release.json).",
        "",
        "## Limitations",
        "",
        "- A toy: it only knows the tiny projects of its training world. Real code, other languages, long files "
        "or vague requests are out of reach; it may call tools that make no sense.",
        "- It runs whatever command it decides to: only use it in a sandbox (`examples/opencode` runs opencode "
        "in a container without internet).",
        "- English only. Not for any real use.",
        "",
    ]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Print a model card for a run's stage.")
    p.add_argument("--run", required=True)
    p.add_argument("--stage", default="distill", choices=STAGES)
    p.add_argument("--id", help="model id used as the title")
    args = p.parse_args()
    print(model_card(Path(args.run), args.stage, args.id))


if __name__ == "__main__":
    main()
