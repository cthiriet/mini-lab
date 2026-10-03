# Architecture

mini-lab is one Python package (`minilab`) that contains everything an AI lab needs,
in miniature: a training pipeline, an inference server, an OpenAI-compatible API
gateway, and a platform (dashboard + billing + chat app).

```
                 browser                         your code (openai SDK)
                    │                                     │
                    ▼                                     │  Authorization: Bearer sk-mini-...
      platform :3000  (dashboard, playground, /chat)      │
                    │  internal token + X-Minilab-Org     │
                    └───────────────┬─────────────────────┘
                                    ▼
                    api gateway :8000   auth · rate limits · metering · logs ──► SQLite (minilab.db)
                                    │  internal token
                                    ▼
                    inference :8001     tokenizer · chat template · KV cache · continuous batching
                                    │
                                    ▼
                    models/<id>/        released checkpoints (from the training pipeline)
```

## Package layout and ownership

| Path | What |
|---|---|
| `minilab/tokenizer/` | BPE tokenizer (`bpe.py`) and chat template (`chat.py`) — **shared contract** |
| `minilab/model/gpt.py` | Transformer + slot-based `KVCache` + `generate` + `sample_next` — **shared contract** |
| `minilab/checkpoint.py` | `save_checkpoint` / `load_checkpoint` — **shared contract** |
| `minilab/registry.py` | Released models (`models/<id>/release.json`), pricing — **shared contract** |
| `minilab/settings.py` | Env-based settings for all services — **shared contract** |
| `minilab/db/` | SQLite schema + data access (users, orgs, keys, credits, requests) — **shared contract** |
| `minilab/testing.py` | `make_random_release()` — random-weight model to test serving without training |
| `minilab/data/`, `train/`, `eval/`, `release.py` | Training pipeline |
| `minilab/inference/` | Inference engine + internal HTTP server |
| `minilab/api/`, `minilab/obs/` | Public OpenAI-compatible gateway, metrics |
| `minilab/platform/`, `minilab/chat/` | Dashboard, billing, playground, chat app |

## Services

Each service is a FastAPI app built by a `create_app()` factory and started with
`uv run python -m minilab.<service> [--host 127.0.0.1] [--port N]`:

| Service | Module | Default port |
|---|---|---|
| Inference | `minilab.inference` (`minilab/inference/server.py:create_app`) | 8001 |
| API gateway | `minilab.api` (`minilab/api/app.py:create_app`) | 8000 |
| Platform + chat | `minilab.platform` (`minilab/platform/app.py:create_app`) | 3000 |

Configuration is read from environment variables by `minilab.settings.get_settings()`:
`MINILAB_DB`, `MINILAB_MODELS_DIR`, `MINILAB_INFERENCE_URL`, `MINILAB_API_URL`,
`MINILAB_PUBLIC_API_URL`, `MINILAB_PLATFORM_URL`, `MINILAB_INTERNAL_TOKEN`, `MINILAB_SERVE_MODELS`
(the releases to serve, when `models/` keeps older ones), `MINILAB_DEFAULT_RPM`,
`MINILAB_DEFAULT_TPM`, `MINILAB_SIGNUP_CREDIT_USD`, `MINILAB_CREDIT_PACKS_USD`,
`STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`.

## Chat template

See `minilab/tokenizer/chat.py`. Special tokens delimit turns; tools are announced
with a `<|system_start|>tools: calculator<|system_end|>` block; a tool call is its name
and raw arguments separated by `<|arg|>` (`calculator<|arg|>expression=347 + 58`), between
`<|tool_call_start|>` / `<|tool_call_end|>`; tool results go in `<|tool_start|>` /
`<|tool_end|>`; optional scratchpad reasoning goes in `<|think_start|>` / `<|think_end|>`.
Generation stops at `<|assistant_end|>`. For agents, the template also cuts system prompts
to their first sentence, shows paths relative to the working directory, and fits a prompt
into a token budget by dropping old turns: see [opencode.md](opencode.md).

Note: `Tokenizer.train` may learn fewer merges than requested on small corpora, so
always build the model with `tokenizer.vocab_size`.

## Internal inference API (inference :8001)

Every request must carry `Authorization: Bearer $MINILAB_INTERNAL_TOKEN`.

### `POST /generate`

```json
{
  "model": "prelude-1",
  "messages": [{"role": "user", "content": "What is 2 + 2?"}],
  "tools": ["calculator"],
  "max_tokens": 128,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": null,
  "seed": null,
  "stop": null,
  "stream": false
}
```

- `messages`: OpenAI format (`system`/`developer`/`user`/`assistant`/`tool`, assistant `tool_calls`).
- `tools`: list of tool *names* (the gateway extracts them from OpenAI `tools`), or null.
- `max_tokens`: null means "until the context is full".
- `stop`: extra stop strings, applied to visible content.

Non-streaming response (`200`):

```json
{
  "model": "prelude-1",
  "content": "4",
  "reasoning": null,
  "tool_calls": [],
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 12, "completion_tokens": 2}
}
```

- `tool_calls`: `[{"name": "calculator", "arguments": "{\"expression\": \"2 + 2\"}"}]` (arguments is a JSON string).
- `finish_reason`: `stop` (end of turn or stop string), `length` (max_tokens or context full), `tool_calls`.

Streaming response (`stream: true`): `text/event-stream`, one JSON object per `data:` line:

```
data: {"type": "delta", "content": "4"}
data: {"type": "delta", "reasoning": "2 + 2 ..."}
data: {"type": "done", "content": "4", "reasoning": null, "tool_calls": [], "finish_reason": "stop", "usage": {...}}
data: {"type": "error", "message": "..."}        # only if generation fails mid-stream
```

Tool-call tokens are never streamed as deltas; they only appear in the final `done` event.

Errors use the OpenAI shape: `{"error": {"message": "...", "type": "invalid_request_error", "code": "..."}}`
with `400` (`context_length_exceeded`, `invalid_request`), `401`, `404` (`model_not_found`), `503` (`overloaded`).

### Other endpoints

- `GET /models` → `{"object": "list", "data": [ModelInfo.to_json(), ...]}` (models currently loaded)
- `GET /health` → `{"status": "ok"}`
- `GET /metrics` → Prometheus text format

## Public API (gateway :8000)

OpenAI-compatible: the official `openai` SDK works by setting `base_url="http://localhost:8000/v1"`.

- `POST /v1/chat/completions` — `model`, `messages`, `tools`, `tool_choice` (`auto`/`none`),
  `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`, `stream`,
  `stream_options.include_usage`, `n` (only 1). Tool calls come back as
  `message.tool_calls` with `finish_reason: "tool_calls"`. Scratchpad reasoning, when present,
  comes back as the non-standard field `message.reasoning_content`.
- `GET /v1/models`, `GET /v1/models/{id}`

Auth:
- Customers: `Authorization: Bearer sk-mini-...` (hashed in the DB, see `db.lookup_api_key`).
- First-party services (playground, chat app): `Authorization: Bearer $MINILAB_INTERNAL_TOKEN`
  plus `X-Minilab-Org: org_...`, optional `X-Minilab-Project: proj_...`, `X-Minilab-Source: playground|chat`.
  Billed to the org like any other traffic.

Billing and limits:
- Cost = `Pricing.cost_micros(prompt_tokens, completion_tokens)` from the model's `release.json`.
- Every authenticated request is logged with `db.record_request` (which also debits credits).
- `429 insufficient_quota` when the org balance is <= 0 or the key hit its spend limit.
- `429 rate_limit_exceeded` above the key's RPM/TPM; `x-ratelimit-*` headers on every response.

## Money

All amounts are integer micro-dollars (`1 USD = 1_000_000`). Credits are prepaid:
new orgs get `MINILAB_SIGNUP_CREDIT_USD`, purchases go through Stripe Checkout (or a
dev-mode fake checkout when `STRIPE_SECRET_KEY` is unset). The ledger is idempotent
on `(kind, ref)`, so a replayed webhook never double-credits.
