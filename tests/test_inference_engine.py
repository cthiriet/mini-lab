"""Inference engine: streaming parser, continuous batching, sampling, stop conditions."""

import asyncio
import dataclasses
import itertools
import json
import random
import time

import pytest
import torch

from minilab.checkpoint import load_checkpoint, save_checkpoint
from minilab.inference.engine import (
    ContextLengthExceeded, ModelRunner, Overloaded, SamplingParams, StreamParser,
)
from minilab.model.gpt import GPT
from minilab.obs.metrics import render
from minilab.registry import load_model_info
from minilab.testing import make_random_release
from minilab.tokenizer.chat import parse_completion, render_prompt


def make_lively_release(models_dir, model_id="mini-test"):
    """A random model whose greedy output depends on the context.

    With plain random init, tied embeddings dominate the residual stream and the model
    just repeats its last input token forever, which would make batching tests
    meaningless. Scaling up the block weights gives varied, prompt-dependent text.
    """
    path = make_random_release(models_dir, model_id)
    model, tok, meta = load_checkpoint(path)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("qkv.weight", "fc.weight", "proj.weight")):
                p.mul_(10)
    save_checkpoint(path, model, tok, meta)
    return path


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    path = make_lively_release(tmp_path_factory.mktemp("models"))
    model, tok, _ = load_checkpoint(path)
    return load_model_info(path), model, tok


_ids = itertools.count()


@pytest.fixture
def make_runner(release):
    """Factory for started runners; each gets its own model id so metrics don't mix."""
    info, model, tok = release
    runners = []

    def make(max_batch=8, max_queue=64, model=model, prefix_cache=True):
        info_ = dataclasses.replace(info, id=f"test-{next(_ids)}")
        runner = ModelRunner(info_, model, tok, max_batch=max_batch, max_queue=max_queue, prefix_cache=prefix_cache)
        runner.start()
        runners.append(runner)
        return runner

    yield make
    for r in runners:
        r.stop()


def prompt_for(tok, text):
    return render_prompt(tok, [{"role": "user", "content": text}])


async def run(runner, prompt, delay=0.0, **params):
    """Submit after `delay`, stream everything. Returns (request, content deltas joined, done event)."""
    await asyncio.sleep(delay)
    req = runner.submit(prompt, SamplingParams(**params), stream=True)
    content, reasoning, done = [], [], None
    async for ev in req.events():
        if ev["type"] == "delta":
            content.append(ev.get("content", ""))
            reasoning.append(ev.get("reasoning", ""))
        else:
            done = ev
    assert done["type"] == "done", done
    return req, "".join(content), done


def metric(name, model):
    prefix = f'{name}{{model="{model}"}} '
    return next(float(line[len(prefix):]) for line in render().splitlines() if line.startswith(prefix))


# ---------------------------------------------------------------------------
# StreamParser
# ---------------------------------------------------------------------------

def feed_all(parser, ids):
    deltas = []
    for t in ids:
        deltas += parser.feed(t)
        if parser.finished or parser.stopped:
            break
    deltas += parser.flush()
    content = "".join(d.get("content", "") for d in deltas)
    reasoning = "".join(d.get("reasoning", "") for d in deltas)
    return content, reasoning, deltas


def test_parser_matches_parse_completion_on_random_tokens(release):
    _, _, tok = release
    rng = random.Random(0)
    specials = list(tok.special_tokens.values())
    for _ in range(300):
        # Mostly raw tokens (including lone UTF-8 continuation bytes), some specials.
        ids = [rng.choice(specials) if rng.random() < 0.1 else rng.randrange(256 + len(tok.merges))
               for _ in range(rng.randrange(1, 60))]
        parser = StreamParser(tok)
        content, reasoning, _ = feed_all(parser, ids)
        ref = parse_completion(tok, ids)
        assert content == parser.content == ref.content
        assert parser.reasoning == ref.reasoning
        if ref.reasoning is not None:
            assert reasoning == ref.reasoning


def test_parser_utf8_split_across_tokens(release):
    _, _, tok = release
    parser = StreamParser(tok)
    e_acute = list("é".encode())         # 2 bytes -> 2 byte tokens
    rocket = list("🚀".encode())          # 4 bytes
    assert parser.feed(e_acute[0]) == []  # incomplete: nothing to show yet
    assert parser.feed(e_acute[1]) == [{"content": "é"}]
    for b in rocket[:-1]:
        assert parser.feed(b) == []
    assert parser.feed(rocket[-1]) == [{"content": "🚀"}]
    parser.feed(0xE2)                      # truncated sequence at the very end
    assert parser.flush() == [{"content": "�"}]
    assert parser.content == "é🚀�"


def test_parser_reasoning_and_tool_calls(release):
    _, _, tok = release
    S = tok.special
    ids = [S("<|think_start|>"), *tok.encode("let me think"), S("<|think_end|>"), *tok.encode("Sure."),
           S("<|tool_call_start|>"), *tok.encode('{"name": "calculator", "arguments": {"expression": "2+2"}}'),
           S("<|tool_call_end|>"), S("<|assistant_end|>")]
    parser = StreamParser(tok)
    content, reasoning, deltas = feed_all(parser, ids)
    assert parser.finished
    assert reasoning == parser.reasoning == "let me think"
    assert content == "Sure."
    assert not any("calculator" in json.dumps(d) for d in deltas)  # tool-call tokens never streamed


def test_parser_stop_strings(release):
    _, _, tok = release
    # One byte per token, so the stop string arrives across many tokens.
    parser = StreamParser(tok, stop=["world", "xyz"])
    content, _, _ = feed_all(parser, list(b"Hello wonderful world, bye"))
    assert parser.stopped and content == parser.content == "Hello wonderful "

    # Held-back text is released once it can no longer be a stop string...
    parser = StreamParser(tok, stop=["ab"])
    assert parser.feed(ord("a")) == []
    assert parser.feed(ord("c")) == [{"content": "ac"}]
    # ...or when generation ends.
    assert parser.feed(ord("a")) == []
    assert parser.flush() == [{"content": "a"}] and parser.content == "aca"

    # Random texts and stop strings over a tiny alphabet (lots of partial matches).
    rng = random.Random(0)
    for _ in range(500):
        text = "".join(rng.choice("ab c") for _ in range(rng.randrange(1, 40)))
        stops = ["".join(rng.choice("ab c") for _ in range(rng.randrange(1, 5))) for _ in range(rng.randrange(1, 4))]
        cut = min((text.find(s) for s in stops if s in text), default=len(text))
        parser = StreamParser(tok, stop=stops)
        content, _, _ = feed_all(parser, tok.encode(text))
        assert content == parser.content == text[:cut], (text, stops)
        assert parser.stopped == (cut < len(text))

    # Stop strings only apply to visible content, not to reasoning.
    S = tok.special
    parser = StreamParser(tok, stop=["stop"])
    content, reasoning, _ = feed_all(parser, [S("<|think_start|>"), *b"stop", S("<|think_end|>"), *b"ok stop"])
    assert (reasoning, content, parser.stopped) == ("stop", "ok ", True)


# ---------------------------------------------------------------------------
# Continuous batching
# ---------------------------------------------------------------------------

def test_greedy_batching_matches_sequential(release, make_runner):
    """Continuous batching must not change greedy outputs, whatever the arrival times."""
    _, model, tok = release
    end = tok.special("<|assistant_end|>")
    prompts = [prompt_for(tok, f"Request {i}: " + "tell me a story " * (i % 4)) for i in range(12)]
    lengths = [3 + 9 * i for i in range(12)]  # different lengths: sequences leave at different steps
    refs = [model.generate([p], n, temperature=0, stop_ids={end})[0] for p, n in zip(prompts, lengths)]

    runner = make_runner(max_batch=4)  # 12 requests, 4 slots: queueing + slot reuse

    async def main():
        return await asyncio.gather(*(
            run(runner, p, delay=0.004 * i, max_tokens=n, temperature=0)
            for i, (p, n) in enumerate(zip(prompts, lengths))
        ))

    for (req, streamed, done), ref in zip(asyncio.run(main()), refs):
        assert req.tokens == ref
        assert streamed == done["content"] == parse_completion(tok, ref).content
        usage = done["usage"]
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (len(req.prompt), len(ref))
        assert usage["prompt_tokens_details"]["cached_tokens"] == req.cached_tokens < len(req.prompt)
    model_id = runner.info.id
    assert metric("minilab_inference_batch_size_sum", model_id) > metric("minilab_inference_batch_size_count", model_id)
    assert runner.num_active == 0 and sorted(runner._free) == [0, 1, 2, 3]


def test_many_concurrent_requests_and_slot_reuse(release, make_runner):
    _, _, tok = release
    runner = make_runner(max_batch=8)

    async def main():
        return await asyncio.gather(*(
            run(runner, prompt_for(tok, f"hello {i}"), max_tokens=20 + i, temperature=1.0, seed=i) for i in range(20)
        ))

    results = asyncio.run(main())
    assert len(results) == 20
    for req, streamed, done in results:
        assert done["finish_reason"] in ("stop", "length")
        assert streamed == done["content"]
        assert req.slot in range(8)
    assert len({req.slot for req, _, _ in results}) == 8  # all slots used, each several times
    assert runner.num_active == 0 and runner.num_waiting == 0 and len(runner._free) == 8


def test_streamed_deltas_concatenate_with_sampling(release, make_runner):
    """Sampling a random model yields lots of invalid / split UTF-8: deltas must still add up."""
    _, _, tok = release
    runner = make_runner()

    async def main():
        return await asyncio.gather(*(
            run(runner, prompt_for(tok, "hi"), max_tokens=60, temperature=1.5, seed=s) for s in range(10)
        ))

    for req, streamed, done in asyncio.run(main()):
        parsed = parse_completion(tok, req.tokens)
        assert streamed == done["content"] == parsed.content
        assert done["reasoning"] == parsed.reasoning


def test_seeded_sampling_is_reproducible(release, make_runner):
    _, _, tok = release
    runner = make_runner()
    prompt = prompt_for(tok, "Once upon a time")
    params = dict(max_tokens=40, temperature=1.0, top_p=0.9, top_k=50)

    async def alone(seed):
        req, _, _ = await run(runner, prompt, seed=seed, **params)
        return req.tokens

    async def in_a_crowd(seed):  # same request, now sharing the batch with others
        others = [run(runner, prompt_for(tok, f"noise {i}"), max_tokens=30, temperature=1.0) for i in range(6)]
        (req, _, _), *_ = await asyncio.gather(run(runner, prompt, delay=0.002, seed=seed, **params), *others)
        return req.tokens

    a, b, c = asyncio.run(alone(42)), asyncio.run(in_a_crowd(42)), asyncio.run(alone(7))
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Stop conditions and limits
# ---------------------------------------------------------------------------

def test_stop_strings(release, make_runner):
    _, model, tok = release
    runner = make_runner()
    prompt = prompt_for(tok, "Tell me about Lily and her dog")
    ref_req, ref_text, _ = asyncio.run(run(runner, prompt, max_tokens=80, temperature=0))
    # Pick a plain-ASCII stop string from the middle of the greedy output.
    i = next(i for i in range(10, len(ref_text) - 3) if ref_text[i:i + 3].isascii() and ref_text[i:i + 3].strip())
    stop = ref_text[i:i + 3]
    cut = ref_text.index(stop)

    req, streamed, done = asyncio.run(run(runner, prompt, max_tokens=80, temperature=0, stop=["never-there", stop]))
    assert done["content"] == streamed == ref_text[:cut]
    assert done["finish_reason"] == "stop"
    assert req.tokens == ref_req.tokens[:len(req.tokens)] and len(req.tokens) < len(ref_req.tokens)


def test_max_tokens_and_context_limits(release, make_runner):
    _, _, tok = release
    runner = make_runner()
    n_ctx = runner.n_ctx
    bos = tok.bos_id

    async def main():
        _, _, done = await run(runner, prompt_for(tok, "hi"), max_tokens=7, temperature=0)
        assert done["usage"]["completion_tokens"] <= 7
        if done["finish_reason"] == "length":
            assert done["usage"]["completion_tokens"] == 7

        # max_tokens=None: until the context is full; too-large max_tokens is clamped.
        for max_tokens in (None, 1000):
            prompt = [bos] + list(range(65, 65 + 26)) * 9  # 235 tokens
            _, _, done = await run(runner, prompt, max_tokens=max_tokens, temperature=0)
            used = done["usage"]["prompt_tokens"] + done["usage"]["completion_tokens"]
            assert used <= n_ctx
            assert (used == n_ctx) == (done["finish_reason"] == "length")

        # A prompt that leaves no room for the completion is rejected at submit.
        for n in (n_ctx, n_ctx + 50):
            with pytest.raises(ContextLengthExceeded):
                runner.submit([bos] * n, SamplingParams())
        _, _, done = await run(runner, [bos] * (n_ctx - 1), temperature=0)  # exactly one token of room
        assert done["usage"]["completion_tokens"] == 1 and done["finish_reason"] in ("length", "stop")

    asyncio.run(main())


class ScriptedModel(GPT):
    """A GPT that ignores its weights and plays a fixed token script for every sequence."""

    def __init__(self, config, script):
        super().__init__(config)
        self.script, self.start = script, {}

    def forward_cached(self, idx, cache, slots):
        B, T = idx.shape
        logits = torch.full((B, self.config.vocab_size), -1e9)
        for b, s in enumerate(slots.tolist()):
            if T > 1 or cache.lengths[s] == 0:  # prefill: a new sequence starts in this slot
                self.start[s] = T
            cache.lengths[s] += T
            n = int(cache.lengths[s]) - self.start[s]  # completion tokens already fed back
            logits[b, self.script[min(n, len(self.script) - 1)]] = 0.0
        return logits


def test_finish_reasons_with_scripted_model(release, make_runner):
    _, model, tok = release
    S = tok.special
    call = '{"name": "calculator", "arguments": {"expression": "2 + 2"}}'
    tool_script = [S("<|think_start|>"), *tok.encode("need math"), S("<|think_end|>"), *tok.encode("Let me check."),
                   S("<|tool_call_start|>"), *tok.encode(call), S("<|tool_call_end|>"), S("<|assistant_end|>")]
    answer_script = [*tok.encode("4"), S("<|assistant_end|>")]
    endless_script = tok.encode("a")

    async def main(script, **params):
        runner = make_runner(model=ScriptedModel(model.config, script))
        return await run(runner, prompt_for(tok, "What is 2 + 2?"), temperature=0, **params)

    req, streamed, done = asyncio.run(main(tool_script))
    assert done["finish_reason"] == "tool_calls"
    assert done["tool_calls"] == [{"name": "calculator", "arguments": json.dumps({"expression": "2 + 2"})}]
    assert done["content"] == streamed == "Let me check."
    assert done["reasoning"] == "need math"
    assert done["usage"]["completion_tokens"] == len(tool_script)

    _, streamed, done = asyncio.run(main(answer_script))
    assert (done["content"], done["finish_reason"], done["tool_calls"], done["reasoning"]) == ("4", "stop", [], None)

    _, _, done = asyncio.run(main(endless_script, max_tokens=5))
    assert (done["content"], done["finish_reason"]) == ("aaaaa", "length")


def test_prefix_cache_reuses_a_finished_chat(release, make_runner):
    """The next turn of a chat is served from the slot its previous turn left, prefills only
    what is new, and generates exactly what a runner without the cache generates."""
    _, model, tok = release
    user = lambda text: {"role": "user", "content": text}
    first = render_prompt(tok, [user("Tell me a story about a dog.")])
    cached_runner, plain_runner = make_runner(max_batch=2), make_runner(max_batch=2, prefix_cache=False)

    async def main(runner):
        req, _, done = await run(runner, first, max_tokens=20, temperature=0)
        answer = {"role": "assistant", "content": done["content"]}
        # the chat app sends the history back, without the scratchpad: the shared prefix
        # ends where the previous answer starts
        second = render_prompt(tok, [user("Tell me a story about a dog."), answer, user("And a cat?")])
        other = await run(runner, prompt_for(tok, "Something else entirely, no shared prefix here"),
                          max_tokens=5, temperature=0)
        return req, await run(runner, second, max_tokens=20, temperature=0), other

    req, (again, _, done), (other, _, _) = asyncio.run(main(cached_runner))
    assert again.slot == req.slot != other.slot  # the other request took the least recently used slot
    assert again.cached_tokens >= len(first) - 1 and done["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    _, (plain, _, _), _ = asyncio.run(main(plain_runner))
    assert plain.cached_tokens == 0 and plain.tokens == again.tokens
    model_id = cached_runner.info.id
    assert metric("minilab_inference_cached_prompt_tokens_total", model_id) >= again.cached_tokens


def test_prefix_cache_picks_the_longest_prefix(release, make_runner):
    _, model, tok = release
    runner = make_runner(max_batch=3, prefix_cache=True)
    a, b = list(range(1, 41)), list(range(1, 21)) + list(range(100, 120))
    runner._cached = {0: a, 1: b, 2: []}
    assert runner._pick_slot(a[:30] + [7, 7]) == (0, 30)
    assert runner._pick_slot(b + [5]) == (1, 40)
    assert runner._pick_slot(list(range(200, 240))) == (2, 0)  # no useful prefix: the least recently used
    runner._free, runner._cached = [0, 1], {0: [1, 2, 3], 1: a}
    assert runner._pick_slot([1, 2, 3, 9]) == (0, 3)  # a few shared tokens: not worth another slot's cache
    runner._free = [0]
    runner._cached = {0: a}
    assert runner._pick_slot(a) == (0, len(a) - 1)  # the last prompt token is always run


# ---------------------------------------------------------------------------
# Cancellation and overload
# ---------------------------------------------------------------------------

class SlowModel:
    """Wraps a model so that every forward takes a while (requests stay in flight)."""

    def __init__(self, model, seconds=0.01):
        self.model, self.seconds = model, seconds
        self.config, self.lm_head = model.config, model.lm_head

    def forward_cached(self, *args):
        time.sleep(self.seconds)
        return self.model.forward_cached(*args)


async def wait_until(cond, timeout=3.0):
    for _ in range(int(timeout / 0.01)):
        if cond():
            return True
        await asyncio.sleep(0.01)
    return cond()


def test_cancel_frees_slot(release, make_runner):
    _, model, tok = release
    runner = make_runner(max_batch=1, model=SlowModel(model))

    async def main():
        first = runner.submit(prompt_for(tok, "hi"), SamplingParams(temperature=1.0, seed=1), stream=True)
        second = runner.submit(prompt_for(tok, "hello"), SamplingParams(temperature=1.0, seed=2), stream=True)
        events = first.events()
        assert (await anext(events))["type"] == "delta"  # `first` is generating
        assert runner.num_active == 1 and runner.num_waiting == 1

        second.cancel()  # cancelled while waiting: leaves the queue
        assert runner.num_waiting == 0
        await events.aclose()  # consumer goes away mid-stream (like a client disconnect)
        assert await wait_until(lambda: runner.num_active == 0)
        assert runner._free == [0]

        # The freed slot serves the next request right away.
        _, _, done = await run(runner, prompt_for(tok, "again"), max_tokens=3, temperature=0)
        assert done["usage"]["completion_tokens"] <= 3

    asyncio.run(main())


def test_overloaded_when_queue_is_full(release, make_runner):
    _, model, tok = release
    runner = make_runner(max_batch=1, max_queue=1, model=SlowModel(model))

    async def main():
        reqs, overloaded = [], 0
        for i in range(4):
            try:
                reqs.append(runner.submit(prompt_for(tok, f"hi {i}"), SamplingParams(max_tokens=5)))
            except Overloaded:
                overloaded += 1
        assert overloaded >= 2 and len(reqs) >= 1  # 1 slot + 1 waiting place
        for r in reqs:
            await r.result()

    asyncio.run(main())
