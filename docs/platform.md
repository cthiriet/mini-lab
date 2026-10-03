# Platform and chat app

The platform is the part of mini-lab that people see: a developer dashboard
(API keys, usage, logs, billing, playground, docs) and a ChatGPT-like chat app.
It is one FastAPI app with server-rendered Jinja2 templates, Tailwind CSS from a
CDN and a little vanilla JavaScript. There is no build step and no Node.

```
uv run python -m minilab.platform [--host 127.0.0.1] [--port 3000]
```

It needs the API gateway (`MINILAB_API_URL`, default `http://127.0.0.1:8000`) for
the playground, the chat app and the status page. Everything else (sign up, keys,
billing, usage, logs) only needs the database.

## Code map

| Path | What |
|---|---|
| `minilab/platform/app.py` | `create_app()`, CSRF middleware, auth, dashboard pages, playground |
| `minilab/platform/web.py` | Shared plumbing: templates, current user and org (`Ctx`), flash messages, formatting, chart geometry, SSE responses |
| `minilab/platform/billing.py` | Credits: Stripe Checkout, the Stripe webhook, test mode |
| `minilab/platform/gateway.py` | Client for the API gateway, including the tool loop |
| `minilab/platform/templates/` | `base.html` (theme), `layout.html` (dashboard shell), `public.html` (logged-out shell), one template per page |
| `minilab/platform/static/` | `app.js` (copy buttons, menu, tooltips, tabs, local times), `stream.js` (SSE reader), `playground.js` |
| `minilab/chat/router.py` | The chat app, mounted at `/chat` |
| `minilab/chat/tools.py` | The calculator the model can call |
| `minilab/chat/templates/chat.html`, `minilab/chat/static/chat.js` | Chat UI |

## Pages

| URL | Login | What |
|---|---|---|
| `/` | no | Landing page (redirects to `/overview` when logged in) |
| `/signup`, `/login`, `POST /logout` | no | Accounts |
| `/overview` | yes | Balance, spend over 7 days, quickstart snippets, latest requests |
| `/api-keys` | yes | Create (name, project, optional spend limit), list, revoke |
| `/projects` | yes | List and create projects |
| `/orgs/new`, `POST /orgs/switch` | yes | Create an organization, switch between them |
| `/usage?days=7\|30\|90&metric=…` | yes | Chart, per-model and per-day tables, latency and throughput |
| `/logs`, `/logs/{id}` | yes | Every request, with its request and response JSON |
| `/billing` | yes | Balance, credit packs, credit history |
| `/playground` | yes | Try a model: system prompt, messages, temperature, max tokens, calculator |
| `/chat` | yes | The chat app |
| `/models`, `/models/{id}` | no | Released models, prices, model card |
| `/docs` | no | API reference |
| `/status`, `/health` | no | Health of the platform, database and gateway |

## Accounts, sessions and organizations

- **Passwords** are hashed with scrypt by `db.create_user`.
- **Sessions**: logging in stores the sha256 of a random token in `sessions` and
  puts the token in the `minilab_session` cookie (`HttpOnly`, `SameSite=Lax`,
  `Secure` over https, 30 days).
- **CSRF**: every `POST` must carry an `Origin` (or `Referer`) header matching the
  site. Browsers always send it and a malicious page can't forge it, so together
  with `SameSite=Lax` no hidden form tokens are needed. The Stripe webhook is the
  only exception: it is authenticated by its signature.
- **Organizations**: signing up creates an organization with the free signup
  credits. A user can create more organizations (without free credits) and switch
  between them. The selected org is a cookie, but it is only a preference: every
  request resolves it against the user's memberships (`web.get_ctx`), and every
  query is scoped to `ctx.org`. Another org's key, log or project simply isn't found.
- **API keys** are shown once, in the response to the form that created them. Only
  their hash is stored.

## Billing

Credits are prepaid and stored in micro-dollars in an append-only ledger
(`credit_ledger`), unique on `(kind, ref)`.

**With Stripe** (`STRIPE_SECRET_KEY` set):

1. `POST /billing/checkout` creates a Checkout Session for one of the packs
   (`MINILAB_CREDIT_PACKS_USD`) with `metadata.org_id` and redirects to Stripe.
2. After paying, Stripe redirects to `/billing/success?session_id=cs_…`. We retrieve
   the session and, if it is paid, credit the org with `ref = session.id`.
3. Stripe also POSTs `checkout.session.completed` to `/billing/webhook`. We verify
   the `Stripe-Signature` header with `STRIPE_WEBHOOK_SECRET` (and reject events
   older than 5 minutes), then credit the same way.

Both paths call `fulfill_checkout`, and the ledger's uniqueness makes the second one
a no-op: a user is never credited twice, and never missed when they close the tab
before the redirect. To try it locally with the [Stripe CLI](https://stripe.com/docs/stripe-cli):

```
stripe listen --forward-to localhost:3000/billing/webhook   # prints whsec_...
STRIPE_SECRET_KEY=sk_test_... STRIPE_WEBHOOK_SECRET=whsec_... uv run python -m minilab.platform
```

and pay with the test card `4242 4242 4242 4242`.

**Without Stripe** the billing page shows a "Test mode: no real payment" badge and
the packs add credits instantly (`ref = dev_<nonce>`; the nonce comes from the
rendered form, so a double click credits once).

## Playground and chat app

The browser never talks to the gateway. It posts to the platform
(`/playground/api/chat` or `/chat/api/chat`), which calls the gateway's
`/v1/chat/completions` as a first-party service:

```
Authorization: Bearer $MINILAB_INTERNAL_TOKEN
X-Minilab-Org: org_…                  # billed to the current org
X-Minilab-Source: playground | chat   # shown in logs and usage
```

So this traffic is rate-limited, logged and billed by the same code as API traffic.

The platform streams back simple Server-Sent Events (`gateway.stream_chat`):

```
data: {"type": "delta", "content": "347 + 58"}
data: {"type": "delta", "reasoning": "…"}                       # scratchpad
data: {"type": "tool", "name": "calculator", "input": "347 + 58", "output": "405", "ok": true}
data: {"type": "error", "status": 429, "code": "insufficient_quota", "message": "…"}
data: {"type": "done", "finish_reason": "stop", "usage": {…}, "cost": "$0.003510", "request_ids": […], "latency_ms": …, "ttft_ms": …}
```

**Tool loop.** With the calculator enabled, the request declares the `calculator`
tool. When the model answers with `finish_reason: "tool_calls"`, the platform runs
the calculator, appends the assistant tool call and a `tool` message with the
result, and calls the gateway again (at most 3 rounds; the last one uses
`tool_choice: "none"`). The calculator (`chat/tools.py`) parses the expression with
`ast` and only evaluates numbers, parentheses and `+ - * / // % **`, with exponents
up to 64 and results up to 10^30. There is no `eval`.

**Cost** in the `done` event is computed from the model's `release.json` pricing,
the same formula the gateway bills with. Each call is also in the logs.

**Chat history** lives in the browser's `localStorage` (per user, wrapped in
try/catch so a blocked storage only loses persistence), so the server keeps no
conversations. Each turn sends the last 10 messages, and the inference server keeps
the newest turns that fit in the model's context, with room for the answer.

## Design

- Tailwind CSS v4 from the Play CDN; the theme lives in `base.html` as CSS
  variables (`--color-paper`, `--color-ink`, `--color-reagent`, …), redefined under
  `prefers-color-scheme: dark`. Components (`.btn`, `.field`, `.panel`, …) are a
  few `@apply` rules in the same file.
- One typeface, [Recursive](https://www.recursive.design/): its `CASL` axis gives the
  casual headings, its `MONO` axis the code and numbers.
- The logo is a chip wired out like a neural net; the flask of the README is the
  balance gauge in the header (blue, amber when low, red when empty).
- Charts are inline SVG built by `web.bar_chart` (no chart library), with HTML axis
  labels and keyboard-focusable bars with tooltips.

## Tests

```
uv run pytest tests/test_platform.py tests/test_chat_tools.py
```

`tests/test_platform.py` plugs a stub gateway (`make_stub_gateway`, which speaks the
OpenAI streaming format, tool calls included) into `create_app(http_transport=…)`.
To click around the UI without a model, run that stub on a port and point
`MINILAB_API_URL` at it:

```python
# stub.py
import sys; sys.path.insert(0, "tests")
import uvicorn
from test_platform import make_stub_gateway
uvicorn.run(make_stub_gateway(delay=0.1), port=8000)
```

```
MINILAB_INTERNAL_TOKEN=test-internal-token uv run python stub.py &
MINILAB_INTERNAL_TOKEN=test-internal-token MINILAB_DB=/tmp/dev.db uv run python -m minilab.platform
```

## Limitations

- No password reset, email verification, or rate limiting on login attempts.
- No team invitations: an organization has a single member (its creator).
- The Tailwind Play CDN compiles CSS in the browser: fine for a teaching project,
  not for production (use the Tailwind CLI to build a stylesheet instead).
- Chat history is per browser.
