"""Throughput benchmark of the engine: 1 vs N concurrent requests.

    uv run python -m minilab.inference.bench                                   # tiny random model
    uv run python -m minilab.inference.bench --models-dir models --model mini-3

Sends the same --requests requests at each concurrency level, keeping C of them
in flight at once, and reports the aggregate throughput (completion tokens / wall
time) next to what a single user sees (tokens/s of their own request).
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
from minilab.tokenizer.chat import render_prompt


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", help="released models (default: a temporary random model)")
    p.add_argument("--model", help="model id (default: the newest in --models-dir)")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
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
