# Inference

`minilab/inference/` turns a released checkpoint into a service: an **engine** that
runs many generations at once on one model (`engine.py`), and a small internal HTTP
server around it (`server.py`) that implements the "Internal inference API" of
[architecture.md](architecture.md). Only the API gateway talks to it.

```bash
uv run python -m minilab.testing models/            # optional: a random-weight model, models/mini-random
uv run python -m minilab.inference --port 8001      # serves every release in $MINILAB_MODELS_DIR

curl -s localhost:8001/generate -H "Authorization: Bearer $MINILAB_INTERNAL_TOKEN" \
  -d '{"model": "mini-random", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "stream": true}'
```

| Env var | Default | |
|---|---|---|
| `MINILAB_MODELS_DIR` | `models` | where the releases live (see `minilab/registry.py`) |
| `MINILAB_SERVE_MODELS` | all | comma-separated model ids to load (the platform's model pickers follow it too: a deployment can keep older releases on disk without offering them) |
| `MINILAB_MAX_BATCH` | `8` | KV cache slots per model = max sequences decoded together |
| `MINILAB_MAX_QUEUE` | `64` | waiting requests per model before `503 overloaded` |
| `MINILAB_PREFIX_CACHE` | `1` | `0` turns [prefix caching](#prefix-caching) off |
| `MINILAB_INTERNAL_TOKEN` | dev value | required on every endpoint but `/health` |

## The life of a request

```
POST /generate ──► render_prompt (chat template → token ids)
                   check length: prompt + completion must fit in the context
                   runner.submit() ──► waiting queue ─┐
                                                      │  engine thread (one per model)
   asyncio.Queue ◄── call_soon_threadsafe ◄── tokens ─┘  prefill · decode · sample · parse
        │
        └──► JSON response, or one `data:` line per event (server-sent events)
```

The server runs on asyncio. The model runs in one plain thread per model. The two
sides share only a lock-protected waiting queue in one direction, and events posted
with `loop.call_soon_threadsafe` in the other. PyTorch releases the GIL inside its
kernels, so the event loop keeps accepting and streaming while the model computes.

## KV cache slots

Generating token *n+1* needs the attention keys and values of tokens *1..n*.
Recomputing them at every step would make generation quadratic, so they are cached.
`KVCache` (in `minilab/model/gpt.py`) is one big tensor per layer with room for
`max_batch` sequences, called **slots**, of up to `block_size` tokens each, plus a
`lengths[slot]` counter.

A request borrows a slot for its whole life and gives it back when it finishes.
Nothing is cleared: the next prefill resets `lengths[slot]` to 0, and attention
masks every key beyond a slot's length, so stale values are never read.

## Prefill vs decode

- **Prefill** runs the whole prompt through the model in one forward pass (`T =
  prompt length`, one slot). It fills the slot's keys and values and produces the
  logits for the first completion token. It costs time proportional to the prompt,
  and that time shows up as *time to first token*.
- **Decode** feeds back the token just sampled (`T = 1`) to get the next one. It
  runs one token at a time, and on a small model almost all of its cost is fixed
  overhead: Python, kernel launches, and reading the weights. That fixed cost is
  the same whether one row or eight go through, which is where batching pays off.

`forward_cached(idx, cache, slots)` does both. Every row continues from its own
position (`cache.lengths[slot]`), so one decode call can advance sequences that
are at completely different points of their generation.

## Continuous batching

Each `ModelRunner` thread loops:

```
while there is work:
    1. drop cancelled sequences                (client gone: the slot is free again)
    2. admit waiting requests into free slots  (prefill each, sample its first token)
    3. ONE decode step for all active slots    (forward_cached with T=1, batch = #active)
    4. sample one token per row, stream it, retire finished sequences
```

Static batching waits for a whole batch to finish before starting the next one, so
short requests sit behind long ones and slots stay idle. Here sequences join and
leave at *every step* (Orca-style "iteration-level scheduling", as in vLLM): a slot
freed at step *t* serves a new request at step *t+1*. When there is no work, the
thread sleeps on a condition variable.

Batching does not change results. Greedy outputs with any mix of arrival times and
lengths are token-for-token identical to one-at-a-time `GPT.generate`
(`test_greedy_batching_matches_sequential`).

**Sampling is per row.** Each request has its own temperature, top-k and top-p,
and its own `torch.Generator`: seeded from `seed` when one is given, otherwise from
fresh entropy. Rows are sampled one by one with `sample_next`. This is what makes
a seeded request return the same text no matter which other requests share its
batch.

**Stop conditions**, checked after every token:

| condition | `finish_reason` |
|---|---|
| `<\|assistant_end\|>` sampled | `stop` |
| a `stop` string appears in the visible content | `stop` (the string itself is cut) |
| `max_tokens` reached | `length` |
| context full | `length` |
| any of the above, and the completion contains tool calls | `tool_calls` |

The context rule is that prompt + completion must fit in `block_size`. A prompt
that leaves no room is rejected with `400 context_length_exceeded`. `max_tokens:
null` means "until the context is full", and a larger `max_tokens` is clamped to
the room left. This also guarantees that a decode step never overflows the KV cache.

**Admission control.** The waiting queue is bounded (`MINILAB_MAX_QUEUE`), and a
full queue answers `503 overloaded` right away. A client (the gateway) can then
retry or fail fast; the alternative would be piling up unbounded latency.

## Streaming parser

`StreamParser` is the incremental twin of `tokenizer.chat.parse_completion`. It has
the same state machine and sees tokens one at a time. For each token it returns
the text that is safe to show now:

- **Channels.** Tokens between `<|think_start|>` and `<|think_end|>` produce
  `{"reasoning": ...}` deltas; other text produces `{"content": ...}` deltas.
- **Tool calls are never streamed.** Tokens between `<|tool_call_start|>` and
  `<|tool_call_end|>` are only parsed at the end, with `parse_completion`, and sent
  in the final `done` event.
- **UTF-8.** A token is a byte string, so a character like `é` or `🚀` can be split
  across two tokens. Each channel feeds its bytes to an incremental UTF-8 decoder,
  which emits a character only once all its bytes have arrived. Invalid bytes
  become `�`, exactly as in the non-streaming decode.
- **Stop strings.** With `stop=["world"]`, the text `wor` must not be sent yet,
  because the next token may complete `world`. The parser holds back the longest
  suffix of the content that is a prefix of some stop string. That text is released
  as soon as it can no longer match, or at the end of generation.

As a result, the concatenated content deltas always equal the final `content`,
which is also what the non-streaming response returns. The tests check this on
random token soups, random stop strings and sampled outputs.

## Prefix caching

A chat sends its whole history with every message, and an API client may send the
same system prompt every time. Their first tokens were already run through the model
by the previous request, and their keys and values are still in its slot: a finished
request frees its slot, but nothing in it is cleared.

So the runner remembers which tokens each free slot holds (the prompt, and every
completion token fed back: all but the last). A new request goes to the free slot
that shares the longest prefix with its prompt, starts at that position, and only
prefills the rest. That prefix is its `usage.prompt_tokens_details.cached_tokens`, as
in OpenAI's API. The last prompt token is always run, since its logits give the first
completion token. If no slot shares at least `MIN_PREFIX` (16) tokens, the request takes
the least recently used slot instead: every chat starts with the same few template
tokens, and that is not worth evicting a useful cache.

This is vLLM's automatic prefix caching and SGLang's RadixAttention without the
machinery: no pages or radix tree, since a slot already holds a whole sequence, and
a cache only lives until its slot is taken by a request that doesn't share it. The
completions are the same as without the cache, up to float rounding (the tests check
that greedy outputs are identical). The chat app's history drops each answer's
scratchpad, so the shared prefix ends where the previous answer starts, and the
answer itself is prefilled again.

`uv run python -m minilab.inference.bench --models-dir models --chats 32` plays chats
of 4 turns, as the chat app sends them, with the cache on and off. On mini-3.2, on one
CPU thread (like the production server):

| 32 chats x 4 turns | prompt tokens prefilled | time to first token, turns 2-4 |
|---|---:|---:|
| 4 chats at a time, no cache | 4,316 | 3.0 ms |
| 4 chats at a time, prefix cache | 1,682 (61% cached) | 2.5 ms |
| 1 chat at a time, no cache | 4,316 | 2.0 ms |
| 1 chat at a time, prefix cache | 1,620 (62% cached) | 1.5 ms |

Prefill is a small part of the work at 256 tokens of context: total time doesn't move,
since decoding dominates. The saving grows with the context.

## Cancellation

If the client of a streaming request disconnects, Starlette cancels the response
task. The `CancelledError` surfaces in `Request.events()`, whose `finally` marks
the request as cancelled, and the engine drops it and frees the slot at the start
of its next step. A request still waiting is simply removed from the queue.
Non-streaming endpoints are not cancelled by Starlette, so the server polls
`request.is_disconnected()` every 0.5 s while it waits for the result.

## Metrics (`GET /metrics`)

| metric | type |
|---|---|
| `minilab_inference_requests_total{model,status}` | counter |
| `minilab_inference_prompt_tokens_total{model}` (prefilled) / `..._completion_tokens_total{model}` | counter |
| `minilab_inference_cached_prompt_tokens_total{model}` (read from the prefix cache) | counter |
| `minilab_inference_active_sequences{model}` / `..._queue_depth{model}` | gauge |
| `minilab_inference_batch_size{model}` (sequences per decode step) | histogram |
| `minilab_inference_time_to_first_token_seconds{model}` | histogram |
| `minilab_inference_request_latency_seconds{model}` | histogram |

## Throughput

`uv run python -m minilab.inference.bench [--models-dir DIR --model ID]` sends the
same 32 requests (`max_tokens=128`, `temperature=1`) at several concurrency levels.
Measured on an Apple M5 Pro (CPU only, 6 torch threads), with random-weight
models of three sizes:

| model | 1 request at a time | 8 concurrent | 16 concurrent |
|---|---|---|---|
| `mini-random`: 2 layers, 64 dims, 0.12M params | 3,081 tok/s | 13,779 tok/s (**4.5×**) | 19,191 tok/s (6.2×) |
| default `GPTConfig`: 4 layers, 128 dims, 1.3M params | 1,579 tok/s | 4,708 tok/s (**3.0×**) | 8,291 tok/s (5.3×) |
| 8 layers, 384 dims, 17M params | 565 tok/s | 1,881 tok/s (**3.3×**) | 2,643 tok/s (4.7×) |

The same run through HTTP, on the 1.3M-param model: 1,455 → 4,834 tok/s without
streaming (3.3×) and 1,231 → 4,460 tok/s with streaming (3.6×). The server adds
little.

Batching trades per-user speed for total throughput. On the 17M model, one user
alone gets 566 tok/s; with 8 concurrent users, each gets 235 tok/s and the server
produces 3.3× more in total. Batch 1 is also a special case on CPU. A single row
turns every matmul into a matrix-vector product, which BLAS runs much faster
(about 15 µs vs 50 µs per matmul at batch 2), so going from 1 to 2 sequences gains
little. From 2 to 8, the cost of a decode step barely moves (2.9 → 3.1 ms on the
17M model).

## Known limitations

- **Prefill.** Each prompt is prefilled on its own. While it runs, the decode batch
  waits, which is fine for 256-token contexts. There is no chunked prefill. The
  prefix cache only reuses a finished request's slot: two requests in flight at the
  same time with the same system prompt each prefill it.
- **Fixed-size cache.** Every slot reserves a full `block_size`; there is no paged
  attention. Batched decode attends over the longest active sequence and masks
  the rest.
- **Loading.** Models are loaded at startup; restart the server to pick up a new release.
- **Special tokens.** Only `<|assistant_end|>` ends a turn. A model that emits
  some other turn token (e.g. `<|user_start|>`) keeps going until a stop string,
  `max_tokens` or the context limit. Such tokens are dropped from the output,
  as in `parse_completion`.
