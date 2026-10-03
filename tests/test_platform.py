"""Platform + chat app: auth, API keys, org isolation, billing, and the streaming proxy.

The API gateway is replaced by a small stub (make_stub_gateway) that speaks the
same OpenAI streaming format, plugged in through an httpx ASGI transport.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import re
import time
from types import SimpleNamespace

import httpx
import pytest
import stripe
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from minilab import db
from minilab.platform.app import create_app
from minilab.platform.gateway import Gateway
from minilab.registry import ModelInfo, Pricing, write_release

INTERNAL_TOKEN = "test-internal-token"
ORIGIN = {"Origin": "http://testserver"}


# ---- stub gateway ---------------------------------------------------------------

def make_stub_gateway(calls: list | None = None, delay: float = 0.0) -> FastAPI:
    """A fake API gateway emitting OpenAI-style chunks, driven by the last message:

    - "... quota ..."                   -> 429 insufficient_quota
    - "What is A + B?" with tools        -> a calculator tool call
    - a tool result                      -> "A + B = <result>."
    - more than 3 messages with "long"   -> 400 context_length_exceeded
    - anything else                      -> scratchpad reasoning, then a short story
    """
    calls = calls if calls is not None else []
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "mini-test", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        calls.append({"headers": dict(request.headers), "body": body})
        if request.headers.get("authorization") != f"Bearer {INTERNAL_TOKEN}":
            return JSONResponse({"error": {"message": "bad token", "code": "invalid_api_key"}}, 401)
        messages = body["messages"]
        last = messages[-1]
        text = last.get("content") or ""
        if "quota" in text:
            return JSONResponse({"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",
                                           "code": "insufficient_quota"}}, 429)
        if len(messages) > 3 and any("long" in (m.get("content") or "") for m in messages):
            return JSONResponse({"error": {"message": "Too many tokens.", "code": "context_length_exceeded"}}, 400)

        def chunk(delta, finish=None):
            return {"id": f"chatcmpl-stub{len(calls)}", "object": "chat.completion.chunk", "created": 0,
                    "model": body["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        usage = {"id": f"chatcmpl-stub{len(calls)}", "object": "chat.completion.chunk", "choices": [],
                 "usage": {"prompt_tokens": 10 * len(messages), "completion_tokens": 20}}
        match = re.search(r"What is (.+)\?", text)
        if last["role"] == "user" and body.get("tools") and body.get("tool_choice") != "none" and match:
            call = {"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "calculator", "arguments": json.dumps({"expression": match.group(1)})}}
            chunks = [chunk({"role": "assistant", "content": None}), chunk({"tool_calls": [call]}),
                      chunk({}, "tool_calls"), usage]
        elif last["role"] == "tool":
            expression = json.loads(messages[-2]["tool_calls"][0]["function"]["arguments"])["expression"]
            chunks = [chunk({"role": "assistant", "content": ""}), chunk({"content": f"{expression} = "}),
                      chunk({"content": f"{text}."}), chunk({}, "stop"), usage]
        else:
            chunks = [chunk({"role": "assistant", "content": ""}), chunk({"reasoning_content": "A dog story."}),
                      chunk({"content": "Once upon a time,"}), chunk({"content": " there was a dog."}),
                      chunk({}, "stop"), usage]

        async def stream():
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n"
                await asyncio.sleep(delay)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


# ---- fixtures ---------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "mini-test").mkdir(parents=True)
    write_release(models_dir / "mini-test", ModelInfo(
        id="mini-test", created=int(time.time()), description="A test model.", context_length=256,
        pricing=Pricing(input_per_1m=0.5, output_per_1m=1.5)))
    for name, value in {
        "MINILAB_DB": str(tmp_path / "platform.db"), "MINILAB_MODELS_DIR": str(models_dir),
        "MINILAB_API_URL": "http://gateway.test", "MINILAB_INTERNAL_TOKEN": INTERNAL_TOKEN,
        "MINILAB_SIGNUP_CREDIT_USD": "1.00", "MINILAB_CREDIT_PACKS_USD": "5,10,25",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    db.configure(tmp_path / "platform.db")
    return SimpleNamespace(tmp_path=tmp_path, models_dir=models_dir)


@pytest.fixture
def gateway_calls():
    return []


@pytest.fixture
def app(env, gateway_calls):
    return create_app(http_transport=httpx.ASGITransport(app=make_stub_gateway(gateway_calls)))


def new_client(app) -> TestClient:
    # Browsers send Origin on every POST; the CSRF check requires it.
    return TestClient(app, headers=ORIGIN, follow_redirects=False)


def signup(app, email="ada@example.com", name="Ada Lovelace", password="correct horse") -> TestClient:
    client = new_client(app)
    r = client.post("/signup", data={"name": name, "email": email, "password": password})
    assert r.status_code == 303, r.text
    return client


def current_org(client) -> dict:
    user = db.get_user_by_session(client.cookies.get("minilab_session"))
    return db.list_user_orgs(user["id"])[0]


def text(response) -> str:
    """The page as a reader sees it (Jinja escapes ' as &#39;)."""
    return html.unescape(response.text)


def sse_events(response) -> list[dict]:
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


# ---- pages & auth -------------------------------------------------------------------

def test_public_pages(app):
    client = new_client(app)
    assert "A whole AI lab" in client.get("/").text
    docs = client.get("/docs")
    assert docs.status_code == 200 and "http://gateway.test/v1" in docs.text and "mini-test" in docs.text
    models = client.get("/models")
    assert "mini-test" in models.text and "$1.50" in models.text
    assert "A test model." in client.get("/models/mini-test").text
    assert client.get("/models/nope").status_code == 404
    assert "All systems operational" in client.get("/status").text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/chat/static/chat.js").status_code == 200


def test_signup_login_logout(app):
    client = signup(app)
    cookie = next(h for h in client.cookies.jar if h.name == "minilab_session")
    assert cookie.has_nonstandard_attr("HttpOnly")
    overview = client.get("/overview")
    assert overview.status_code == 200
    assert "Ada's lab" in text(overview) and "$1.00" in overview.text  # org + signup credit
    assert db.get_balance_micros(current_org(client)["id"]) == 1_000_000
    assert client.get("/").headers["location"] == "/overview"

    assert client.post("/logout").status_code == 303
    assert client.get("/overview").headers["location"].startswith("/login")

    r = client.post("/login", data={"email": "ada@example.com", "password": "wrong password"})
    assert r.status_code == 400 and "Incorrect email or password" in r.text
    r = client.post("/login", data={"email": "ADA@example.com", "password": "correct horse", "next": "/usage"})
    assert r.status_code == 303 and r.headers["location"] == "/usage"
    assert client.get("/usage").status_code == 200


def test_signup_validation(app):
    signup(app)
    client = new_client(app)
    r = client.post("/signup", data={"email": "ada@example.com", "password": "another password"})
    assert r.status_code == 400 and "already exists" in r.text
    r = client.post("/signup", data={"email": "bob@example.com", "password": "short"})
    assert r.status_code == 400 and "8 characters" in r.text


def test_login_redirect_is_local_only(app):
    signup(app)
    client = new_client(app)
    r = client.post("/login", data={"email": "ada@example.com", "password": "correct horse",
                                    "next": "//evil.example/steal"})
    assert r.headers["location"] == "/overview"


@pytest.mark.parametrize("path", ["/overview", "/api-keys", "/projects", "/usage", "/logs", "/billing",
                                  "/playground", "/chat", "/orgs/new", "/logs/chatcmpl-x"])
def test_dashboard_requires_login(app, path):
    r = new_client(app).get(path)
    assert r.status_code == 303 and r.headers["location"].startswith("/login?next=")


def test_json_apis_require_login(app):
    client = new_client(app)
    for path in ("/playground/api/chat", "/chat/api/chat"):
        r = client.post(path, json={"model": "mini-test", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401


def test_csrf_blocks_cross_site_posts(app):
    client = signup(app)
    project = db.list_projects(current_org(client)["id"])[0]
    data = {"name": "k", "project_id": project["id"]}
    assert client.post("/api-keys", data=data, headers={"Origin": "https://evil.example"}).status_code == 403
    no_origin = TestClient(app, follow_redirects=False, cookies=dict(client.cookies))
    assert no_origin.post("/api-keys", data=data).status_code == 403
    assert no_origin.post("/api-keys", data=data, headers={"Referer": "http://testserver/api-keys"}).status_code == 200
    assert no_origin.post("/signup", data={"email": "x@example.com", "password": "12345678"}).status_code == 403


# ---- API keys, projects, orgs -------------------------------------------------------------

def test_api_key_lifecycle(app):
    client = signup(app)
    org = current_org(client)
    project = db.list_projects(org["id"])[0]
    r = client.post("/api-keys", data={"name": "Story bot", "project_id": project["id"], "spend_limit_usd": "2.5"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    secret = re.search(r'value="(sk-mini-[^"]+)"', r.text).group(1)
    key = db.lookup_api_key(secret)
    assert key["org_id"] == org["id"] and key["name"] == "Story bot" and key["spend_limit_micros"] == 2_500_000

    listing = client.get("/api-keys").text  # shown once: never again
    assert secret not in listing and "Story bot" in listing and key["key_hint"] in listing

    r = client.post(f"/api-keys/{key['id']}/revoke")
    assert r.status_code == 303 and db.lookup_api_key(secret) is None
    assert "Revoked" in client.get("/api-keys").text


def test_api_key_validation(app):
    client = signup(app)
    project = db.list_projects(current_org(client)["id"])[0]
    r = client.post("/api-keys", data={"name": "k", "project_id": project["id"], "spend_limit_usd": "-3"})
    assert r.status_code == 400 and "positive amount" in r.text
    other = signup(app, "bob@example.com", "Bob")
    other_project = db.list_projects(current_org(other)["id"])[0]
    r = client.post("/api-keys", data={"name": "k", "project_id": other_project["id"]})
    assert r.status_code == 400 and db.list_api_keys(current_org(client)["id"]) == []


def test_projects(app):
    client = signup(app)
    assert client.post("/projects", data={"name": "Bedtime app"}).status_code == 303
    page = client.get("/projects").text
    assert "Default project" in page and "Bedtime app" in page


def test_org_isolation(app):
    ada, bob = signup(app), signup(app, "bob@example.com", "Bob")
    ada_org, bob_org = current_org(ada), current_org(bob)
    key, secret = db.create_api_key(ada_org["id"], db.list_projects(ada_org["id"])[0]["id"], "Ada's key")
    db.record_request(id="chatcmpl-ada1", org_id=ada_org["id"], model="mini-test", status_code=200,
                      api_key_id=key["id"], prompt_tokens=5, completion_tokens=7, cost_micros=13,
                      request_body={"secret_prompt": "ada only"})

    # Bob sees none of it and can't touch it...
    assert "Ada's key" not in text(bob.get("/api-keys"))
    assert "chatcmpl-ada1" not in bob.get("/logs").text
    assert bob.get("/logs/chatcmpl-ada1").status_code == 404
    bob.post(f"/api-keys/{key['id']}/revoke")
    assert db.lookup_api_key(secret) is not None
    # ...not even by switching to Ada's org, or forging the org cookie.
    assert bob.post("/orgs/switch", data={"org_id": ada_org["id"]}).status_code == 404
    bob.cookies.set("minilab_org", ada_org["id"])
    assert "Bob's lab" in text(bob.get("/overview"))
    assert "chatcmpl-ada1" not in bob.get("/logs").text

    # Ada does see it.
    assert "chatcmpl-ada1" in ada.get("/logs").text
    detail = text(ada.get("/logs/chatcmpl-ada1"))
    assert "secret_prompt" in detail and "Ada's key" in detail
    assert bob_org["id"] != ada_org["id"]


def test_create_and_switch_orgs(app):
    client = signup(app)
    first = current_org(client)
    r = client.post("/orgs", data={"name": "Acme Robotics"})
    assert r.status_code == 303
    overview = text(client.get("/overview"))
    assert "Acme Robotics: credits" in overview and "$0.00" in overview  # no free credits for extra orgs
    user = db.get_user_by_session(client.cookies.get("minilab_session"))
    assert [o["balance_micros"] for o in db.list_user_orgs(user["id"])] == [1_000_000, 0]
    r = client.post("/orgs/switch", data={"org_id": first["id"]}, headers={"Referer": "http://testserver/usage"})
    assert r.status_code == 303 and r.headers["location"] == "/usage"
    assert "Ada's lab: credits" in text(client.get("/overview"))
    assert client.post("/orgs/switch", data={"org_id": "__new"}).headers["location"] == "/orgs/new"


def test_usage_and_logs_pages(app):
    client = signup(app)
    org = current_org(client)
    for i in range(3):
        db.record_request(id=f"chatcmpl-u{i}", org_id=org["id"], model="mini-test", status_code=200,
                          source="api", prompt_tokens=40, completion_tokens=60, cost_micros=110,
                          latency_ms=100 + i, ttft_ms=20, request_body={"messages": [{"role": "user", "content": "hi"}]},
                          response_body={"choices": []})
    db.record_request(id="chatcmpl-err", org_id=org["id"], model="mini-test", status_code=429,
                      error="rate limited")
    usage = client.get("/usage?days=7&metric=tokens")
    assert usage.status_code == 200 and "mini-test" in usage.text and "$0.000330" in usage.text
    assert "101 ms" in usage.text  # median latency
    logs = client.get("/logs").text
    assert all(f"chatcmpl-u{i}" in logs for i in range(3)) and "429" in logs
    detail = client.get("/logs/chatcmpl-err").text
    assert "rate limited" in detail
    overview = client.get("/overview").text
    assert "Latest requests" in overview and "$0.000330" in overview


# ---- billing --------------------------------------------------------------------------------

def test_test_mode_checkout_credits_once(app):
    client = signup(app)
    org_id = current_org(client)["id"]
    page = client.get("/billing").text
    assert "Test mode: no real payment" in page
    nonce = re.search(r'name="nonce" value="([0-9a-f]+)"', page).group(1)
    r = client.post("/billing/checkout", data={"amount_usd": "10", "nonce": nonce})
    assert r.status_code == 303 and db.get_balance_micros(org_id) == 11_000_000
    client.post("/billing/checkout", data={"amount_usd": "10", "nonce": nonce})  # double click
    assert db.get_balance_micros(org_id) == 11_000_000
    assert db.list_ledger(org_id, kinds=("purchase",))[0]["ref"] == f"dev_{nonce}"
    client.post("/billing/checkout", data={"amount_usd": "1000", "nonce": "ab" * 8})  # not a pack
    assert db.get_balance_micros(org_id) == 11_000_000
    assert "Test-mode credits" in client.get("/billing").text


def stripe_signature(payload: str, secret: str, timestamp: int | None = None) -> str:
    timestamp = timestamp or int(time.time())
    digest = hmac.new(secret.encode(), f"{timestamp}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def checkout_session(org_id: str, session_id="cs_test_123", amount_cents=500, status="paid") -> dict:
    return {"id": session_id, "object": "checkout.session", "payment_status": status,
            "amount_total": amount_cents, "amount_subtotal": amount_cents, "metadata": {"org_id": org_id, "amount_usd": str(amount_cents // 100)}}


def test_stripe_webhook_verifies_signature_and_is_idempotent(env, monkeypatch):
    secret = "whsec_test_secret"
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    app = create_app()
    client = signup(app)
    org_id = current_org(client)["id"]
    stripe_client = TestClient(app)  # Stripe's servers: no Origin, no cookies

    payload = json.dumps({"id": "evt_1", "type": "checkout.session.completed",
                          "data": {"object": checkout_session(org_id)}})
    headers = {"Stripe-Signature": stripe_signature(payload, secret), "Content-Type": "application/json"}
    r = stripe_client.post("/billing/webhook", content=payload, headers=headers)
    assert r.status_code == 200 and db.get_balance_micros(org_id) == 6_000_000
    r = stripe_client.post("/billing/webhook", content=payload, headers=headers)  # Stripe retries
    assert r.status_code == 200 and db.get_balance_micros(org_id) == 6_000_000

    forged = payload.replace("500", "50000")
    r = stripe_client.post("/billing/webhook", content=forged, headers=headers)
    assert r.status_code == 400
    r = stripe_client.post("/billing/webhook", content=payload,
                           headers={"Stripe-Signature": stripe_signature(payload, "whsec_wrong")})
    assert r.status_code == 400
    assert stripe_client.post("/billing/webhook", content=payload).status_code == 400
    old = stripe_signature(payload, secret, timestamp=int(time.time()) - 3600)  # replayed much later
    assert stripe_client.post("/billing/webhook", content=payload, headers={"Stripe-Signature": old}).status_code == 400

    unpaid = json.dumps({"type": "checkout.session.completed",
                         "data": {"object": checkout_session(org_id, "cs_unpaid", status="unpaid")}})
    r = stripe_client.post("/billing/webhook", content=unpaid, headers={"Stripe-Signature": stripe_signature(unpaid, secret)})
    assert r.status_code == 200 and db.get_balance_micros(org_id) == 6_000_000


def test_stripe_checkout_and_success_page(env, monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    created = {}

    def fake_create(**params):
        created.update(params)
        return SimpleNamespace(id="cs_test_abc", url="https://checkout.stripe.test/pay/cs_test_abc")

    monkeypatch.setattr(stripe.checkout.Session, "create", fake_create)
    app = create_app()
    client = signup(app)
    org_id = current_org(client)["id"]
    assert "Test mode" not in client.get("/billing").text

    r = client.post("/billing/checkout", data={"amount_usd": "25"})
    assert r.status_code == 303 and r.headers["location"].startswith("https://checkout.stripe.test/")
    assert created["metadata"]["org_id"] == org_id
    assert created["line_items"][0]["price_data"]["unit_amount"] == 2500
    assert "{CHECKOUT_SESSION_ID}" in created["success_url"]

    session = checkout_session(org_id, "cs_test_abc", 2500)
    monkeypatch.setattr(stripe.checkout.Session, "retrieve",
                        lambda session_id, **kw: SimpleNamespace(to_dict=lambda: session))
    r = client.get("/billing/success?session_id=cs_test_abc")
    assert r.status_code == 303 and db.get_balance_micros(org_id) == 26_000_000
    client.get("/billing/success?session_id=cs_test_abc")  # reload: still credited once
    assert db.get_balance_micros(org_id) == 26_000_000


# ---- playground & chat proxy -------------------------------------------------------------------

def test_playground_streams_and_runs_the_calculator(app, gateway_calls):
    client = signup(app)
    org_id = current_org(client)["id"]
    assert '<option>mini-test</option>' in client.get("/playground").text
    r = client.post("/playground/api/chat", json={
        "model": "mini-test", "temperature": 0.5, "max_tokens": 64, "calculator": True,
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 347 + 58?"}],
    })
    assert r.status_code == 200
    events = sse_events(r)
    assert {"type": "tool", "name": "calculator", "input": "347 + 58", "output": "405", "ok": True} in events
    assert "".join(e.get("content", "") for e in events if e["type"] == "delta") == "347 + 58 = 405."
    done = events[-1]
    assert done["type"] == "done" and done["finish_reason"] == "stop"
    assert done["usage"] == {"prompt_tokens": 20 + 40, "completion_tokens": 40}  # two gateway calls
    assert done["cost_micros"] == round(60 * 0.5 + 40 * 1.5) and done["cost"] == "$0.000090"
    assert done["request_ids"] == ["chatcmpl-stub1", "chatcmpl-stub2"]

    # First-party auth, billed to the org, labelled as playground traffic.
    first, second = gateway_calls
    assert first["headers"]["authorization"] == f"Bearer {INTERNAL_TOKEN}"
    assert first["headers"]["x-minilab-org"] == org_id
    assert first["headers"]["x-minilab-source"] == "playground"
    assert first["body"]["stream"] is True and first["body"]["temperature"] == 0.5
    assert first["body"]["max_tokens"] == 64 and first["body"]["tools"][0]["function"]["name"] == "calculator"
    # The tool result went back to the model.
    assert second["body"]["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "405"}
    assert second["body"]["messages"][-2]["tool_calls"][0]["function"]["name"] == "calculator"


def test_chat_streams_reasoning_and_bills_as_chat(app, gateway_calls):
    client = signup(app)
    page = client.get("/chat")
    assert page.status_code == 200 and "What is 347 + 58?" in page.text and "Tell me a story about a dog" in page.text
    r = client.post("/chat/api/chat", json={"model": "mini-test", "calculator": False,
                                            "messages": [{"role": "user", "content": "Tell me a story about a dog"}]})
    events = sse_events(r)
    assert {"type": "delta", "reasoning": "A dog story."} in events
    assert "".join(e.get("content", "") for e in events if e["type"] == "delta") == "Once upon a time, there was a dog."
    assert events[-1]["type"] == "done"
    assert gateway_calls[0]["headers"]["x-minilab-source"] == "chat"
    assert "tools" not in gateway_calls[0]["body"]


def test_chat_reports_gateway_errors(app):
    client = signup(app)
    r = client.post("/chat/api/chat", json={"model": "mini-test", "messages": [{"role": "user", "content": "out of quota?"}]})
    events = sse_events(r)
    assert events == [{"type": "error", "status": 429, "code": "insufficient_quota",
                       "message": "You exceeded your current quota."}]


def test_chat_sends_only_the_recent_messages(app, gateway_calls):
    """The server fits the conversation into the context: a long chat sends only its last messages."""
    client = signup(app)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"} for i in range(99)]
    events = sse_events(client.post("/chat/api/chat", json={"model": "mini-test", "messages": history}))
    assert events[-1]["type"] == "done"
    (call,) = gateway_calls
    assert call["body"]["messages"][-1]["content"] == "message 98" and len(call["body"]["messages"]) <= 40


def test_chat_request_validation(app):
    client = signup(app)
    r = client.post("/chat/api/chat", json={"model": "mini-test", "messages": []})
    assert r.status_code == 422
    r = client.post("/chat/api/chat", json={"model": "mini-test", "messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 422


def test_gateway_down(env):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    app = create_app(http_transport=httpx.MockTransport(refuse))
    client = signup(app)
    events = sse_events(client.post("/playground/api/chat", json={
        "model": "mini-test", "messages": [{"role": "user", "content": "hi"}]}))
    assert events[0]["type"] == "error" and events[0]["code"] == "gateway_unreachable"
    status = client.get("/status").text
    assert "Some systems are not responding" in status and "unreachable" in status


def test_gateway_client_accumulates_split_tool_calls():
    """Tool call arguments may arrive in several deltas (as with OpenAI); we join them."""
    chunks = [
        {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_9", "function": {"name": "calculator", "arguments": '{"expr'}}]}}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ession": "2 * 21"}'}}]}}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    answers = iter([chunks, [{"id": "c2", "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}]}]])

    def handler(request):
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in next(answers)) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    gateway = Gateway("http://gateway.test", "t", httpx.MockTransport(handler))

    async def collect():
        return [e async for e in gateway.stream_chat(org_id="org_x", source="chat", model="m", calculator=True,
                                                      messages=[{"role": "user", "content": "2 * 21?"}])]

    events = asyncio.run(collect())
    assert {"type": "tool", "name": "calculator", "input": "2 * 21", "output": "42", "ok": True} in events
    assert events[-1]["request_ids"] == ["c1", "c2"]
