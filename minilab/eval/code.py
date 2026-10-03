"""The coding eval: tasks on fresh projects, played end to end like opencode would.

For every task kind (data/code.py), `n_per_kind` new tasks. The model gets opencode's messages
(system prompt, tools, the request), its tool calls run in a sandbox, their results go back,
and so on until it answers or runs out of steps. The task's check then looks at the project's
files and the answer: tests pass, the file has the new function, the answer has the output...

The model's tool calls never run on this machine: every task gets a project directory inside
one locked-down Docker container (data/sandbox.py: no network, read-only system, no
capabilities, limited memory, CPU and processes). Only our own, trusted code (the task's
expected outputs) runs locally.

Metrics (0..1): success per task kind and per family (explore, create, modify, repair),
`agent` = the mean over all kinds, `valid_calls` = tool calls opencode would accept (a known
tool, valid arguments), `chat` = small talk answered without tools, `title` = opencode's title
requests answered with a short title, `code_ppl` = perplexity on held-out Python documents.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field

import torch

from minilab.data import code
from minilab.data.conversations import NAME
from minilab.data.sandbox import CODE_TOOLS, OPENCODE_TOOLS, TOOL_SCHEMAS, Container, DockerSandbox
from minilab.eval.tasks import bits_per_char
from minilab.model.gpt import GPT
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import coerce_arguments, parse_completion, render_prompt
from minilab.train.pretrain import sample_story

EVAL_SEED = "code-eval"
INVALID = ('{"error":{"type":"tool.execution","message":"Invalid arguments',
           '{"error":{"type":"tool.execution","message":"No tool named')


@dataclass
class Episode:
    task: code.Task | None
    sb: DockerSandbox | None
    messages: list[dict]
    tools: list[str] | None = None
    answer: str | None = None
    calls: int = 0
    valid: int = 0
    done: bool = False
    transcript: list[str] = field(default_factory=list)


def _generate(model: GPT, tok: Tokenizer, prompts: list[list[int]], max_new: int, batch: int) -> list[list[int]]:
    end = tok.special("<|assistant_end|>")
    out = []
    for i in range(0, len(prompts), batch):
        out += model.generate(prompts[i:i + batch], max_new, temperature=0.0, stop_ids={end})
    return out


def play(model: GPT, tok: Tokenizer, episodes: list[Episode], max_steps: int, max_new: int, batch: int) -> None:
    """Advance every episode in lockstep: one model turn each, then run the tool calls."""
    budget = model.config.block_size - max_new
    for _ in range(max_steps + 1):
        active = [e for e in episodes if not e.done]
        if not active:
            break
        prompts, ready = [], []
        for e in active:
            try:
                ids = render_prompt(tok, e.messages, e.tools, budget=budget)
            except ValueError:  # the conversation no longer fits
                ids = []
            if not ids or len(ids) >= model.config.block_size:  # (a model without the "code" template)
                e.done, e.answer = True, ""
                continue
            prompts.append(ids)
            ready.append(e)
        for e, out in zip(ready, _generate(model, tok, prompts, max_new, batch)):
            parsed = parse_completion(tok, out)
            if not parsed.tool_calls or e.sb is None:
                e.answer, e.done = parsed.content, True
                e.calls += len(parsed.tool_calls)  # a tool call where none was needed counts against chat
                e.transcript.append(f"assistant: {parsed.content}")
                continue
            calls = []
            for i, c in enumerate(parsed.tool_calls):
                args = coerce_arguments(c["arguments"], TOOL_SCHEMAS.get(c["name"]))
                calls.append({"id": f"call_{len(e.messages)}_{i}", "type": "function",
                              "function": {"name": c["name"], "arguments": args}})
            e.messages.append({"role": "assistant", "content": parsed.content, "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = None
                result = e.sb.call(name, args) if isinstance(args, dict) else code_error("Invalid arguments")
                e.calls += 1
                e.valid += name in CODE_TOOLS and not result.startswith(INVALID)
                e.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                e.transcript.append(f"{name}({call['function']['arguments'][:200]}) -> {result[:200]}")
            if e.calls >= max_steps:
                e.done, e.answer = True, ""
    for e in episodes:
        if not e.done:
            e.done, e.answer = True, ""


def code_error(message: str) -> str:
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []}, separators=(",", ":"))


def held_out_documents(n_docs: int) -> list[str]:
    docs = code.pretrain_documents(seed=10_000_019)
    return [next(docs) for _ in range(n_docs)]


@torch.no_grad()
def perplexity(model: GPT, tok: Tokenizer, n_docs: int = 400) -> tuple[float, float]:
    """Loss and perplexity on held-out pretraining documents (packed like in pretraining)."""
    from minilab.data.loader import packed_batches
    stream = ([tok.bos_id, *tok.encode(doc)] for doc in held_out_documents(n_docs))
    device = model.wte.weight.device
    total, count = 0.0, 0
    for x, y in packed_batches(stream, 8, model.config.block_size):
        _, loss = model(x.to(device), y.to(device))
        total, count = total + loss.item() * y.numel(), count + y.numel()
    loss = total / max(1, count)
    return loss, math.exp(loss)


def _task_episodes(container: Container, kinds: list[str], n: int) -> list[Episode]:
    episodes = []
    for kind in kinds:
        rng = random.Random(f"{EVAL_SEED}-{kind}")
        for _ in range(n):
            task = code.make_task(rng, kind)
            sb = DockerSandbox(container, task.files)
            messages = [code.system_message(str(sb.root), rng), {"role": "user", "content": task.prompt}]
            episodes.append(Episode(task, sb, messages, OPENCODE_TOOLS))
    return episodes


def _chat_episodes(n: int) -> tuple[list[Episode], list[tuple[str, str]]]:
    """Small talk (no tool may be called) and opencode's title requests."""
    rng = random.Random(f"{EVAL_SEED}-chat")
    episodes, kinds = [], []
    for _ in range(n):
        ck, prompt, _ = code.chat_turn(rng)
        episodes.append(Episode(None, None, [code.system_message("/home/user/project", rng),
                                             {"role": "user", "content": prompt}], OPENCODE_TOOLS))
        kinds.append(("chat", ck))
    for _ in range(n):
        task = code.make_task(rng, rng.choice(list(code.TASKS)))
        episodes.append(Episode(None, None, [{"role": "system", "content": code.TITLE_SYSTEM},
                                             {"role": "user", "content": f'"{task.prompt}"'}]))
        kinds.append(("title", ""))
    return episodes, kinds


def _chat_ok(kind: tuple[str, str], e: Episode) -> bool:
    answer = (e.answer or "").strip()
    if e.calls or not answer:
        return False
    if kind[0] == "title":
        return "\n" not in answer and len(answer) <= 50
    return {"identity": f"I'm {NAME}," in answer, "out_of_scope": "can't" in answer}.get(kind[1], True)


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def evaluate(model: GPT, tok: Tokenizer, cfg: dict, stage: str) -> dict:
    """The coding eval, with the [code_eval] section of the config."""
    ec = cfg["code_eval"]
    t0 = time.time()
    loss, ppl = perplexity(model, tok, ec.get("ppl_docs", 400))
    bpc = bits_per_char(tok, held_out_documents(ec.get("ppl_docs", 400)), loss)
    result = {"stage": stage, "mode": "chat" if stage != "pretrain" else "completion",
              "val_loss": round(loss, 4), "val_ppl": round(ppl, 3), "val_bpc": round(bpc, 4)}
    if stage == "pretrain":  # a base model has no chat format: only its Python
        device = model.wte.weight.device.type
        result["samples"] = [{"prompt": "def add(a, b):", "response": sample_story(model, tok, device, "def add(a, b):")}]
        result["eval_seconds"] = round(time.time() - t0, 1)
        return result

    kinds = list(code.TASKS)
    max_steps, max_new, batch = ec.get("max_steps", 8), ec.get("max_new_tokens", 384), ec.get("batch", 32)
    with Container() as container:
        episodes = _task_episodes(container, kinds, ec["n_per_kind"])
        play(model, tok, episodes, max_steps, max_new, batch)
        oks = []
        for e in episodes:
            try:
                oks.append(bool(e.task.check(e.sb, e.answer or "")))
            except Exception:  # a check on a project the model mangled
                oks.append(False)
            e.sb.close()
    chats, chat_kinds = _chat_episodes(ec.get("n_chat", 30))
    play(model, tok, chats, 1, 128, batch)

    by_kind = {k: _mean(ok for e, ok in zip(episodes, oks) if e.task.kind == k) for k in kinds}
    result["tasks"] = by_kind
    result["families"] = {f: _mean(by_kind[k] for k in ks) for f, ks in code.FAMILIES.items()}
    result["agent"] = _mean(by_kind.values())
    calls = sum(e.calls for e in episodes)
    result["valid_calls"] = sum(e.valid for e in episodes) / calls if calls else None
    result["steps"] = round(calls / len(episodes), 2)
    result["chat"] = _mean(_chat_ok(k, e) for k, e in zip(chat_kinds, chats) if k[0] == "chat")
    result["title"] = _mean(_chat_ok(k, e) for k, e in zip(chat_kinds, chats) if k[0] == "title")
    result["failures"] = {k: [{"prompt": e.task.prompt, "transcript": e.transcript}
                              for e, ok in zip(episodes, oks) if e.task.kind == k and not ok][:2] for k in kinds}
    result["samples"] = [{"prompt": e.task.prompt, "response": "\n".join(e.transcript)}
                         for e in episodes[::max(1, len(episodes) // 6)]][:6]
    result["eval_seconds"] = round(time.time() - t0, 1)
    return result


def get(result: dict, path: str):
    for key in path.split("."):
        result = (result or {}).get(key) if isinstance(result, dict) else None
    return result


def main() -> None:
    """Play one request on a project directory, in the Docker sandbox, and print the transcript.

        uv run python -m minilab.eval.code models/prelude-1 examples/opencode/project "Run the tests and fix any bug"

    The project is copied into the container: the directory itself is never modified."""
    import argparse
    from pathlib import Path

    from minilab.checkpoint import load_checkpoint
    from minilab.train.trainer import DEVICES, resolve_device
    p = argparse.ArgumentParser(description="Play a coding request, tools in a Docker sandbox.")
    p.add_argument("model", help="a checkpoint or release directory")
    p.add_argument("project", help="a directory of small text files (copied into the sandbox)")
    p.add_argument("prompt", nargs="+")
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()
    model, tok, _ = load_checkpoint(args.model, device=resolve_device(args.device, generation=True))
    root = Path(args.project)
    files = {str(f.relative_to(root)): f.read_text() for f in sorted(root.rglob("*"))
             if f.is_file() and not any(part.startswith(".") for part in f.relative_to(root).parts)}
    with Container() as container, DockerSandbox(container, files) as sb:
        rng = random.Random(0)
        e = Episode(None, sb, [code.system_message(str(sb.root), rng), {"role": "user", "content": " ".join(args.prompt)}],
                    OPENCODE_TOOLS)
        play(model, tok, [e], args.max_steps, 384, 1)
        print("\n".join(e.transcript))
        changed = {k: v for k, v in sb.files().items() if files.get(k) != v}
        for path, text in changed.items():
            print(f"\n--- {path} (in the sandbox) ---\n{text}", end="")


if __name__ == "__main__":
    main()
