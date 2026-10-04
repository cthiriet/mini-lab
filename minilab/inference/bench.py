"""Throughput benchmark of the engine: 1 vs N concurrent requests.

    uv run python -m minilab.inference.bench                                   # tiny random model
    uv run python -m minilab.inference.bench --models-dir models --model prelude-1.1
    uv run python -m minilab.inference.bench --models-dir models --chats 32    # prefix cache on vs off

Sends the same --requests requests at each concurrency level, keeping C of them
in flight at once, and reports the aggregate throughput (completion tokens / wall
time) next to what a single user sees (tokens/s of their own request).

With --chats, plays that many chats of --turns requests instead, each turn sending the
history back like the chat app, with the prefix cache on and then off: the prompt tokens
actually prefilled and the time to first token of the later turns.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time

from minilab.checkpoint import load_checkpoint
from minilab.inference.engine import ModelRunner, SamplingParams
from minilab.registry import get_model, list_models
from minilab.testing import make_random_release
from minilab.tokenizer.chat import prompt_budget, render_prompt

CHAT_REQUESTS = ["What is 347 + 58?", "Tell me a story about a dog.", "And add 25 to that?", "Who are you?",
                 "What is 4521 + 380?", "Hi!", "Can you write Python code?", "Tell me a story about a cat."]


async def run_level(runner: ModelRunner, prompts: list[list[int]], concurrency: int, max_tokens: int) -> dict:
    todo = list(enumerate(prompts))
    tokens, latencies, speeds = 0, [], []

    async def worker():
        nonlocal tokens
        while todo:
            i, prompt = todo.pop(0)
            t0 = time.perf_counter()
            done = await runner.submit(prompt, SamplingParams(max_tokens=max_tokens, temperature=1.0, seed=i)).result()
            dt = time.perf_counter() - t0
            n = done["usage"]["completion_tokens"]
            tokens += n
            latencies.append(dt)
            speeds.append(n / dt)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - t0
    return {"tokens": tokens, "wall": wall, "tok_s": tokens / wall,
            "latency": sum(latencies) / len(latencies), "per_request_tok_s": sum(speeds) / len(speeds)}


async def run_chats(runner: ModelRunner, n_chats: int, turns: int, concurrency: int, max_tokens: int) -> dict:
    """n_chats chats of `turns` requests, `concurrency` chats at a time. Every turn sends the
    history back as the chat app does: the earlier answers, no scratchpads, fitted to the
    context like the server fits it. Greedy, so both runs see the same chats."""
    tok, todo = runner.tokenizer, list(range(n_chats))
    prompt_tokens = cached = 0
    ttft: list[float] = []  # time to first token, turns 2+

    async def chat(c: int) -> None:
        nonlocal prompt_tokens, cached
        history: list[dict] = []
        for t in range(turns):
            history.append({"role": "user", "content": CHAT_REQUESTS[(c + 3 * t) % len(CHAT_REQUESTS)]})
            prompt = render_prompt(tok, history, budget=prompt_budget(runner.n_ctx, max_tokens))
            t0 = time.perf_counter()
            req = runner.submit(prompt, SamplingParams(max_tokens=max_tokens, temperature=0), stream=True)
            first = None
            async for event in req.events():
                first = first or time.perf_counter()
            if t:
                ttft.append(first - t0)
            usage = event["usage"]
            prompt_tokens += usage["prompt_tokens"]
            cached += usage["prompt_tokens_details"]["cached_tokens"]
            history.append({"role": "assistant", "content": event.get("content") or ""})

    async def worker():
        while todo:
            await chat(todo.pop(0))

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    ttft.sort()
    return {"wall": time.perf_counter() - t0, "prompt": prompt_tokens, "cached": cached,
            "ttft": sum(ttft) / len(ttft), "ttft_p50": ttft[len(ttft) // 2]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", help="released models (default: a temporary random model)")
    p.add_argument("--model", help="model id (default: the newest in --models-dir)")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    p.add_argument("--chats", type=int, help="play this many multi-turn chats, prefix cache on then off")
    p.add_argument("--turns", type=int, default=4)
    args = p.parse_args()

    tmp = None
    if args.models_dir is None:
        tmp = tempfile.TemporaryDirectory()
        make_random_release(tmp.name)
        args.models_dir = tmp.name
    info = get_model(args.models_dir, args.model) if args.model else next(iter(list_models(args.models_dir)), None)
    if info is None:
        raise SystemExit(f"no model {args.model or ''} in {args.models_dir}")
    model, tok, _ = load_checkpoint(info.path)
    c = model.config
    print(f"model {info.id}: {model.num_params() / 1e6:.2f}M params "
          f"(n_layer={c.n_layer}, n_embd={c.n_embd}, vocab={c.vocab_size}, context={c.block_size})")
    print(f"{args.requests} requests, max_tokens={args.max_tokens}, temperature=1.0\n")

    if args.chats:
        print(f"{args.chats} chats of {args.turns} turns, {max(args.concurrency)} at a time, "
              f"max_tokens={args.max_tokens}, greedy\n")
        print(f"{'prefix cache':>12} {'prompt tokens':>14} {'prefilled':>10} {'cached':>7} "
              f"{'TTFT turns 2+':>14} {'p50':>7} {'wall s':>7}")
        for cache_on in (True, False):
            runner = ModelRunner(info, model, tok, max_batch=max(args.concurrency), prefix_cache=cache_on)
            runner.start()
            asyncio.run(run_chats(runner, 2, 2, 2, 8))  # warm-up
            r = asyncio.run(run_chats(runner, args.chats, args.turns, max(args.concurrency), args.max_tokens))
            runner.stop()
            print(f"{'on' if cache_on else 'off':>12} {r['prompt']:>14} {r['prompt'] - r['cached']:>10} "
                  f"{r['cached'] / r['prompt']:>7.0%} {1000 * r['ttft']:>11.1f} ms {1000 * r['ttft_p50']:>4.1f} ms "
                  f"{r['wall']:>7.2f}")
        if tmp is not None:
            tmp.cleanup()
        return

    runner = ModelRunner(info, model, tok, max_batch=max(args.concurrency))
    runner.start()
    prompts = [render_prompt(tok, [{"role": "user", "content": f"Tell me a story about the number {i}."}])
               for i in range(args.requests)]
    asyncio.run(run_level(runner, prompts[:2], 2, 16))  # warm-up

    print(f"{'concurrency':>11} {'tokens':>7} {'wall s':>7} {'agg tok/s':>10} {'speedup':>8} "
          f"{'per-req tok/s':>14} {'mean latency s':>15}")
    base = None
    for level in args.concurrency:
        r = asyncio.run(run_level(runner, prompts, level, args.max_tokens))
        base = base or r["tok_s"]
        print(f"{level:>11} {r['tokens']:>7} {r['wall']:>7.2f} {r['tok_s']:>10.0f} {r['tok_s'] / base:>7.1f}x "
              f"{r['per_request_tok_s']:>14.0f} {r['latency']:>15.3f}")
    runner.stop()
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
