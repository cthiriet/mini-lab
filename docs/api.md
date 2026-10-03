# API reference

The public API is served by the **API gateway** (`minilab/api/`, port 8000). It is
compatible with OpenAI's Chat Completions API: the official `openai` SDKs (and most
tools built on them) work by only changing the base URL.

```
client ──► gateway :8000  auth · quota · rate limits · metering ──► SQLite
                │  internal token
                ▼
           inference :8001  (see inference.md)
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | generate a reply, streamed or not |
| `GET /v1/models` | models currently served |
| `GET /v1/models/{id}` | one model |
| `GET /health` | gateway status, and whether inference and the database are reachable (no auth) |
| `GET /metrics` | Prometheus metrics (no auth: keep it private in a real deployment) |

## Quickstart

```bash
uv run python -m minilab.testing models/          # a random-weight model, if you haven't trained one
uv run python -m minilab.inference --port 8001    # terminal 1
uv run python -m minilab.api --port 8000          # terminal 2
```

Create an API key in the platform dashboard (API keys page), or from Python:

```python
from minilab import db
db.init_db()
user = db.create_user("me@example.com", "password")
org = db.create_org("Me", user["id"])             # comes with the signup credits
project = db.list_projects(org["id"])[0]
_, secret = db.create_api_key(org["id"], project["id"], "my key")
print(secret)                                     # sk-mini-...
```

With curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $MINILAB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "mini-random", "messages": [{"role": "user", "content": "What is 2 + 2?"}], "max_tokens": 32}'
```

With the official `openai` Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-mini-...")

completion = client.chat.completions.create(
    model="mini-random",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_completion_tokens=32,
)
print(completion.choices[0].message.content, completion.usage)

stream = client.chat.completions.create(
    model="mini-random",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The API also accepts calls straight from a browser (CORS is open to any origin;
keys are sent in a header, never in cookies). Don't ship a secret key in public
web pages, though: anyone could read it and spend your credits.

## Authentication

Send `Authorization: Bearer sk-mini-...`. Keys belong to a project of an
organization; the organization pays. Only a SHA-256 hash of each key is stored, so a
lost key can't be recovered, only revoked and replaced. A revoked key stops working
immediately.

**First-party services** (the platform's playground and chat app) use the shared
internal token instead of a key, and say who pays:

| Header | |
|---|---|
| `Authorization: Bearer $MINILAB_INTERNAL_TOKEN` | the shared secret |
| `X-Minilab-Org: org_...` | required: the organization billed |
| `X-Minilab-Project: proj_...` | optional: must belong to that org |
| `X-Minilab-Source: playground` | optional (default `playground`): label shown in the logs, e.g. `chat` |

This traffic is billed, logged and rate-limited like any other (per organization,
with the default limits), with no API key (`api_key_id` is empty in the logs).
Anyone holding the internal token can bill any organization: change its default
value (`MINILAB_INTERNAL_TOKEN`) before exposing the services.

## Chat completions

`POST /v1/chat/completions`

| Parameter | |
|---|---|
| `model` | required: a model id from `GET /v1/models` |
| `messages` | required: `system`/`developer`, `user`, `assistant` (optionally with `tool_calls`) and `tool` messages. `content` is a string or a list of `{"type": "text"}` parts |
| `max_completion_tokens` / `max_tokens` | cap on generated tokens. Default: until the model's context window is full |
| `temperature` | 0 to 2. 0 means greedy decoding. Default: the model's `default_temperature`, 0 for prelude, which opencode drives without ever sending one (OpenAI's default is 1) |
| `top_p` | 0 to 1, default 1 |
| `top_k` | not in OpenAI's API: keep only the k most likely tokens (with the SDK: `extra_body={"top_k": 20}`) |
| `seed` | same seed and parameters, same output (streamed or not) |
| `stop` | a string or up to 4 strings; generation stops before them |
| `tools` | function tools. Only their **names** reach the model (it was trained on `tools: calculator`, not on JSON schemas). The model writes every argument as text: the gateway types them with the tools' JSON schemas (`"true"` becomes `true` where `parameters` says boolean) |
| `tool_choice` | `auto` (default) or `none` (the model doesn't see the tools) |
| `stream` | send the answer as server-sent events |
| `stream_options.include_usage` | add a last chunk with the token usage |

Unknown parameters with no effect on the output (`user`, `metadata`, `store`,
`parallel_tool_calls`, `service_tier`, ...) are accepted and ignored. Parameters
mini-lab can't honour are rejected with `400 unsupported_parameter` rather than
silently ignored: `n` > 1, `logprobs`, `top_logprobs`, `response_format` other than
`text`, `presence_penalty` / `frequency_penalty` other than 0, `logit_bias`, the
legacy `functions` / `function_call`, audio output, predicted outputs, web search,
and `tool_choice` `required` or naming a function. Image, audio and file content
parts are rejected too: the models only read text.

### Response

```json
{
  "id": "chatcmpl-7846376cf6989261b83423de",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "prelude-1",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "4", "refusal": null, "reasoning_content": "2 + 2 = 4"},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21,
            "prompt_tokens_details": {"cached_tokens": 0}},
  "system_fingerprint": null
}
```

- `finish_reason`: `stop` (end of turn, or a stop string), `length` (`max_tokens`
  or the context window), `tool_calls`.
- `reasoning_content` (non-standard, only when the model used its scratchpad): the
  model's reasoning, kept out of `content`. With the Python SDK:
  `message.reasoning_content` (or `message.model_extra["reasoning_content"]`).
- The id is also the `x-request-id` header and the id of the request in the
  dashboard's logs.

### Streaming

With `"stream": true` the response is `text/event-stream`, one `chat.completion.chunk`
per `data:` line, exactly like OpenAI:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
data: {…"choices":[{"index":0,"delta":{"reasoning_content":"2 + 2"},…}]}      # scratchpad, if any
data: {…"choices":[{"index":0,"delta":{"content":"4"},…}]}
data: {…"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_…","type":"function","function":{"name":"calculator","arguments":"{\"expression\": \"2 + 2\"}"}}]},…}]}
data: {…"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
data: {…"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":0}}}   # with include_usage
data: [DONE]
```

- The first chunk carries the role (`content` is `null` when the answer is only tool calls).
- Tool calls are not streamed token by token: each one arrives whole, in one chunk,
  just before the `finish_reason` chunk.
- With `include_usage`, every chunk has `"usage": null` except the extra last one.
- If generation fails mid-stream, the last event is `data: {"error": {...}}`; the
  SDKs raise an `APIError`. Errors that happen before the first token (unknown
  model, quota, rate limit, overloaded server, ...) are normal HTTP errors instead.

### Tool calling

A round trip with a calculator: the model asks for the tool, you run it, you send
the result back, the model answers.

```python
import ast, json, operator
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-mini-...")
tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '347 + 58'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
}]

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

def calculator(expression: str) -> str:
    """Arithmetic only: the expression comes from a model, never eval() it."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expression, mode="eval").body))

messages = [{"role": "user", "content": "What is 347 + 58?"}]
while True:
    reply = client.chat.completions.create(model="prelude-1", messages=messages, tools=tools)
    message = reply.choices[0].message
    if reply.choices[0].finish_reason != "tool_calls":
        print(message.content)                      # "The answer is 405."
        break
    messages.append(message.model_dump(exclude_none=True))   # the assistant turn with its tool_calls
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)  # {"expression": "347 + 58"}
        messages.append({"role": "tool", "tool_call_id": call.id, "content": calculator(args["expression"])})
```

Whether the model actually calls the tool depends on its training: a random-weight
model never will, a model fine-tuned on tool-use conversations does.

## Models

`GET /v1/models` lists the models the inference server has loaded, as OpenAI model
objects with mini-lab extras:

```json
{"object": "list", "data": [{
  "id": "prelude-1", "object": "model", "created": 1760000000, "owned_by": "mini-lab",
  "description": "...", "context_length": 1024,
  "pricing": {"input_per_1m": 10.0, "output_per_1m": 50.0}
}]}
```

`GET /v1/models/{id}` returns one of them, or `404 model_not_found`.

A conversation longer than the context is fitted to it by the server, which drops the oldest
turns, because an agent like opencode sends far more than 1,024 tokens of instructions and
history ([opencode.md](opencode.md)). Only a last request that doesn't fit on its own is
rejected with `400 context_length_exceeded`.

## Errors

Errors have OpenAI's shape, so the SDKs raise the matching exception:

```json
{"error": {"message": "The model 'gpt-5' does not exist or you do not have access to it.",
           "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
```

| Status | `code` | When | Python SDK |
|---|---|---|---|
| 400 | `null`, `unsupported_parameter`, `context_length_exceeded` | malformed or unsupported request; the last request alone longer than the context window (minus room for the answer) | `BadRequestError` |
| 401 | `invalid_api_key` | missing, unknown or revoked key | `AuthenticationError` |
| 404 | `model_not_found` | unknown model | `NotFoundError` |
| 404 | `null` | unknown URL | `NotFoundError` |
| 413 | `null` | request body larger than 1 MB | `APIStatusError` |
| 429 | `rate_limit_exceeded` | above the key's RPM or TPM (see below) | `RateLimitError` |
| 429 | `insufficient_quota` | no credits left, or the key reached its spend limit | `RateLimitError` |
| 502 | `upstream_error` | the inference server failed | `InternalServerError` |
| 503 | `overloaded` | the inference server's queue is full | `InternalServerError` |
| 503 | `inference_unavailable` | the inference server can't be reached | `InternalServerError` |

The SDKs retry 429 and 5xx on their own (twice by default, with backoff and
honouring `retry-after`). Responses that retrying can't fix (`insufficient_quota`,
a request larger than the TPM limit) carry `x-should-retry: false`, which the SDKs obey.

## Rate limits

Each API key has a requests-per-minute (RPM) and a tokens-per-minute (TPM) limit:
the key's own, set in the dashboard, or the defaults `MINILAB_DEFAULT_RPM` (60) and
`MINILAB_DEFAULT_TPM` (40,000). First-party traffic gets the defaults, per
organization.

Limits are token buckets: a key can burst up to its full limit, which then refills
continuously (RPM/60 requests per second), with no reset at minute boundaries. The
number of tokens a request will use is only known at the end, so the gateway
reserves an estimate up front (about 4 characters per token for the prompt, plus
`max_tokens`, capped at the context window) and gives back the difference once the
real usage is known. Setting `max_tokens` close to what you need lets more requests
through.

Every authenticated response carries:

| Header | Example | |
|---|---|---|
| `x-ratelimit-limit-requests` | `60` | RPM limit |
| `x-ratelimit-remaining-requests` | `59` | requests available now |
| `x-ratelimit-reset-requests` | `1s` | time until the request budget is full again |
| `x-ratelimit-limit-tokens` | `40000` | TPM limit |
| `x-ratelimit-remaining-tokens` | `39976` | tokens available now |
| `x-ratelimit-reset-tokens` | `36ms` | time until the token budget is full again |
| `retry-after` | `30` | on 429 `rate_limit_exceeded`: seconds to wait |

Every response, authenticated or not, carries `x-request-id`.

Limits live in the gateway's memory: they reset when it restarts and aren't shared
between replicas (that would need a shared store such as Redis).

## Billing

Credits are prepaid, in micro-dollars (1 USD = 1,000,000). Each model's price is in
its `release.json` (USD per million prompt and completion tokens), and

    cost_micros = round(prompt_tokens * input_per_1m + completion_tokens * output_per_1m)

(`Pricing.cost_micros` in `minilab/registry.py`). Every authenticated chat completion
(except those refused by the rate limiter, which only count in the metrics, so that
nobody can fill the database for free) is logged (status, tokens, cost, latency, time to first token, truncated request and
response bodies) and its cost debited from the organization and added to the key's
spend, in one database transaction.

- Successful requests pay for the usage reported by the inference server. Prompt
  tokens read from the prefix cache (`usage.prompt_tokens_details.cached_tokens`) cost
  the same as the others, where the big providers discount them.
- Failed requests (invalid, rejected for quota, inference errors before any text was
  sent) cost nothing. Every request counts against the key's RPM limit, even one
  that is then rejected, so errors aren't a free way to load the servers.
- If the client closes a stream before the end, generation stops (freeing the
  server for others) and the request is logged with status `499`. It is billed for
  what the gateway knows was generated: the estimated prompt plus one token per
  streamed piece of text (nothing if no text was streamed yet). This may slightly
  undercount; the alternative, generating to the end to learn the exact count,
  would spend compute nobody reads. A stream that breaks on our side after some
  text was sent (`502 upstream_error`) is billed the same way: otherwise breaking a
  stream on purpose would be a way to get text for free.
- Bodies over 1 MB are rejected with `413` before they are parsed. The inference
  server's tokenizer is pure Python, and cuts the text into chunks of at most 32
  characters, so even a megabyte of adversarial text encodes in about 1.5 s.
- A request is refused with `429 insufficient_quota` when the organization's
  balance is zero or less, or when the key's spend reached its spend limit. The
  check happens before generating, so concurrent requests can take the balance
  slightly below zero; the next request is then refused.

## Metrics

`GET /metrics` exposes, in Prometheus format:

| Metric | Labels | |
|---|---|---|
| `minilab_api_requests_total` | `model`, `status` | chat completions |
| `minilab_api_tokens_total` | `model`, `kind` (`prompt`/`completion`) | tokens billed |
| `minilab_api_cost_micros_total` | `model` | amount billed |
| `minilab_api_request_duration_seconds` | `model` | latency histogram |
| `minilab_api_ttft_seconds` | `model` | time to first token (streaming) |
| `minilab_api_http_requests_total` | `route`, `status` | every HTTP response |
| `minilab_api_in_flight_requests` | | requests being processed |

Unknown model names are counted as `model="unknown"`, so callers can't create an
unbounded number of series.

## Configuration

`uv run python -m minilab.api [--host 127.0.0.1] [--port 8000]` reads:

| Env var | Default | |
|---|---|---|
| `MINILAB_DB` | `minilab.db` | SQLite database shared with the platform |
| `MINILAB_MODELS_DIR` | `models` | where `release.json` files (pricing) are read |
| `MINILAB_INFERENCE_URL` | `http://127.0.0.1:8001` | the inference server |
| `MINILAB_INTERNAL_TOKEN` | dev value | used to call inference, and accepted from first-party services |
| `MINILAB_DEFAULT_RPM` / `MINILAB_DEFAULT_TPM` | `60` / `40000` | limits for keys without their own |
| `MINILAB_PLATFORM_URL` | `http://127.0.0.1:3000` | linked from error messages |

## Code map

| File | |
|---|---|
| `minilab/api/app.py` | `create_app()`, middleware (request id, rate-limit headers, CORS), models, health, metrics |
| `minilab/api/completions.py` | `POST /v1/chat/completions`, including the streaming relay |
| `minilab/api/schemas.py` | request validation, translation for inference, response and chunk shapes |
| `minilab/api/auth.py` | API keys, first-party auth, quota check |
| `minilab/api/ratelimit.py` | RPM/TPM token buckets and headers |
| `minilab/api/metering.py` | pricing (model catalog), request logging and billing, metrics |
| `minilab/api/upstream.py` | client for the internal inference API, error mapping |
| `minilab/api/errors.py` | OpenAI-shaped errors |
