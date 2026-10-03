"""Tests for the API gateway (minilab.api), driven with the official openai SDK.

A stub inference server implements the internal contract (docs/architecture.md)
with canned answers; both it and the gateway run under uvicorn on local ports,
so streaming and client disconnects behave like in production.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import openai
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from minilab import db
from minilab.api.app import create_app
from minilab.api.auth import Caller
from minilab.api.completions import SSEResponse, StreamRelay
from minilab.api.errors import APIError
from minilab.api.metering import Meter
from minilab.api.ratelimit import RateLimiter, format_duration
from minilab.api.schemas import ChatCompletionRequest
from minilab.registry import ModelInfo, Pricing, write_release
from minilab.settings import get_settings

MODEL = "mini-test"
CODE_MODEL = "mini-test-greedy"  # greedy by default, like prelude (opencode sends no temperature)
PRICING = Pricing(input_per_1m=100.0, output_per_1m=300.0)  # pricey, so costs are well above 0 micros
TOKEN = "test-internal-token"
_used_ports: set[int] = set()


def free_port() -> int:
    for port in range(18110, 18200):
        if port in _used_ports:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        _used_ports.add(port)
        return port
    raise RuntimeError("no free port in 18110-18199")


@contextmanager
def serve(app, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)


# ---- stub inference server ---------------------------------------------------

def _answer(body: dict) -> dict:
    """Canned behaviour, driven by the last message."""
    last = body["messages"][-1]
    text = last.get("content") or ""
    if last["role"] == "tool":
        return {"content": f"The answer is {text}.", "tool_calls": [], "finish_reason": "stop"}
    if "edit" in (body.get("tools") or []) and "fix" in text:  # the "code" template: every argument is text
        call = {"name": "edit", "arguments": json.dumps({"path": "calc.py", "oldString": "a - b",
                                                         "newString": "a + b", "replaceAll": "true"})}
        return {"content": "", "tool_calls": [call], "finish_reason": "tool_calls"}
    if "calculator" in (body.get("tools") or []) and "calc" in text:
        call = {"name": "calculator", "arguments": json.dumps({"expression": "347 + 58"})}
        return {"content": "", "tool_calls": [call], "finish_reason": "tool_calls"}
    if "think" in text:
        return {"content": "4", "reasoning": "two plus two is four", "tool_calls": [], "finish_reason": "stop"}
    if "slow" in text:
        return {"content": " ".join(["word"] * 100), "tool_calls": [], "finish_reason": "stop"}
    return {"content": "Hello from the stub!", "tool_calls": [], "finish_reason": "stop"}


def make_stub_inference() -> FastAPI:
    app = FastAPI()
    app.state.payloads = []
    app.state.disconnected = 0

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"error": {"message": "bad token", "type": "invalid_request_error"}}, 401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/models")
    async def models():
        info = ModelInfo(id=MODEL, created=1_700_000_000, description="stub", context_length=256, pricing=PRICING)
        return {"object": "list", "data": [info.to_json(), CODE_INFO.to_json()]}

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        app.state.payloads.append(body)
        text = body["messages"][-1].get("content") or ""
        if body["model"] not in (MODEL, CODE_MODEL):
            return JSONResponse({"error": {"message": f"model {body['model']} not loaded",
                                           "type": "invalid_request_error", "code": "model_not_found"}}, 404)
        if "overload" in text:
            return JSONResponse({"error": {"message": "all slots busy", "type": "server_error",
                                           "code": "overloaded"}}, 503)
        ans = _answer(body)
        pieces = [p for p in ans["content"].split(" ") if p]
        if body.get("max_tokens") and len(pieces) > body["max_tokens"]:
            pieces, ans["finish_reason"] = pieces[: body["max_tokens"]], "length"
        ans["content"] = " ".join(pieces)
        prompt_tokens = sum(len((m.get("content") or "").split()) + 2 for m in body["messages"])
        usage = {"prompt_tokens": prompt_tokens,
                 "completion_tokens": len(pieces) + 3 * bool(ans.get("reasoning")) + 7 * len(ans["tool_calls"])}
        result = {"model": MODEL, "reasoning": None, **ans, "usage": usage}
        if not body.get("stream"):
            return result

        async def events():
            try:
                if result["reasoning"]:
                    yield f"data: {json.dumps({'type': 'delta', 'reasoning': result['reasoning']})}\n\n"
                for i, piece in enumerate(pieces):
                    if "slow" in text:
                        await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'type': 'delta', 'content': piece if i == 0 else ' ' + piece})}\n\n"
                    if "crash" in text and i == 0:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'out of memory'})}\n\n"
                        return
                yield f"data: {json.dumps({'type': 'done', **result})}\n\n"
            except asyncio.CancelledError:
                app.state.disconnected += 1
                raise

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


CODE_INFO = ModelInfo(id=CODE_MODEL, created=1_700_000_001, context_length=1024, pricing=PRICING,
                      default_temperature=0.0)


# ---- fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    write_release(_mkdir(tmp / "models" / MODEL),
                  ModelInfo(id=MODEL, created=1_700_000_000, context_length=256, pricing=PRICING))
    write_release(_mkdir(tmp / "models" / CODE_MODEL), CODE_INFO)
    stub = make_stub_inference()
    with serve(stub, free_port()) as stub_url:
        settings = replace(get_settings(), db_path=str(tmp / "test.db"), models_dir=str(tmp / "models"),
                           inference_url=stub_url, internal_token=TOKEN, default_rpm=1000,
                           default_tpm=1_000_000, platform_url="http://platform.test")
        with serve(create_app(settings), free_port()) as url:
            yield SimpleNamespace(url=url, stub=stub, settings=settings)
    db.configure(None)


def _mkdir(path):
    path.mkdir(parents=True)
    return path


def new_org(credit_usd: float = 1.0, **key_kwargs) -> SimpleNamespace:
    user = db.create_user(f"{secrets.token_hex(6)}@example.com", "pw")
    org = db.create_org("Acme", user["id"], signup_credit_usd=credit_usd)
    project = db.list_projects(org["id"])[0]
    key, secret = db.create_api_key(org["id"], project["id"], "test", user["id"], **key_kwargs)
    return SimpleNamespace(org=org, project=project, key=key, secret=secret)


def client(env, api_key: str, **kwargs) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{env.url}/v1", api_key=api_key, max_retries=0, **kwargs)


def ask(env, secret: str, content: str = "Hi!", model: str = MODEL, **kwargs):
    return client(env, secret).chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], **kwargs)


def logged(org_id: str) -> list[dict]:
    return db.list_requests(org_id)  # newest first


def wait_for(predicate, timeout: float = 5.0):
    deadline = time.time() + timeout
    while not (result := predicate()):
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)
    return result


# ---- chat completions ----------------------------------------------------------

def test_non_stream_completion_is_billed_exactly(env):
    t = new_org()
    raw = client(env, t.secret).chat.completions.with_raw_response.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi!"}], user="u-1", metadata={"a": "b"})
    completion = raw.parse()

    assert completion.id.startswith("chatcmpl-") and completion.object == "chat.completion"
    assert completion.model == MODEL
    choice = completion.choices[0]
    assert choice.message.role == "assistant" and choice.message.content == "Hello from the stub!"
    assert choice.finish_reason == "stop" and choice.message.tool_calls is None
    u = completion.usage
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens and u.completion_tokens == 4
    assert raw.headers["x-request-id"] == completion.id
    assert raw.headers["x-ratelimit-limit-requests"] == "1000"

    cost = PRICING.cost_micros(u.prompt_tokens, u.completion_tokens)
    assert cost > 0
    assert db.get_balance_micros(t.org["id"]) == 1_000_000 - cost
    assert db.lookup_api_key(t.secret)["spend_micros"] == cost
    [row] = logged(t.org["id"])
    assert row["id"] == completion.id and row["status_code"] == 200 and row["cost_micros"] == cost
    assert (row["prompt_tokens"], row["completion_tokens"]) == (u.prompt_tokens, u.completion_tokens)
    assert row["api_key_id"] == t.key["id"] and row["source"] == "api" and row["project_id"] == t.project["id"]
    assert row["latency_ms"] is not None and row["ttft_ms"] is None
    assert json.loads(row["request_body"])["messages"][0]["content"] == "Hi!"
    assert json.loads(row["response_body"])["id"] == completion.id
    [ledger] = db.list_ledger(t.org["id"], kinds=("usage",))
    assert ledger["amount_micros"] == -cost and ledger["ref"] == completion.id


def test_parameters_are_translated_for_inference(env):
    t = new_org()
    ask(env, t.secret, max_completion_tokens=2, temperature=0.5, top_p=0.9, seed=7, stop="\n",
        extra_body={"top_k": 20})
    sent = env.stub.state.payloads[-1]
    assert (sent["max_tokens"], sent["temperature"], sent["top_p"], sent["seed"], sent["top_k"]) == (2, 0.5, 0.9, 7, 20)
    assert sent["stop"] == ["\n"] and sent["tools"] is None and sent["stream"] is False
    completion = ask(env, t.secret, max_tokens=1)
    assert env.stub.state.payloads[-1]["max_tokens"] == 1 and completion.choices[0].finish_reason == "length"
    ask(env, t.secret)
    assert env.stub.state.payloads[-1]["max_tokens"] is None and env.stub.state.payloads[-1]["temperature"] == 1.0
    # content parts are flattened to text
    client(env, t.secret).chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": " there"}]}])
    assert [m["content"] for m in env.stub.state.payloads[-1]["messages"]] == ["Be brief.", "Hi there"]


def test_stream_with_usage(env):
    t = new_org()
    stream = ask(env, t.secret, stream=True, stream_options={"include_usage": True})
    chunks = list(stream)

    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Hello from the stub!"
    assert len({c.id for c in chunks}) == 1 and chunks[0].object == "chat.completion.chunk"
    finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finish == ["stop"]
    *others, last = chunks
    assert last.choices == [] and last.usage.completion_tokens == 4
    assert all(c.usage is None for c in others)

    [row] = logged(t.org["id"])
    assert row["id"] == chunks[0].id and row["status_code"] == 200 and row["ttft_ms"] is not None
    assert row["cost_micros"] == PRICING.cost_micros(last.usage.prompt_tokens, last.usage.completion_tokens)
    assert json.loads(row["response_body"])["choices"][0]["message"]["content"] == "Hello from the stub!"


def test_stream_raw_sse_format(env):
    t = new_org()
    with httpx.stream("POST", f"{env.url}/v1/chat/completions", headers={"Authorization": f"Bearer {t.secret}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-request-id"].startswith("chatcmpl-")
        lines = [line for line in r.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][6:])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "usage" not in first  # only present with stream_options.include_usage


def test_tool_call_round_trip(env):
    t = new_org()
    c = client(env, t.secret)
    tools = [{"type": "function", "function": {
        "name": "calculator", "description": "Evaluate arithmetic",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}}]
    messages = [{"role": "user", "content": "calc 347 + 58"}]

    first = c.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    assert env.stub.state.payloads[-1]["tools"] == ["calculator"]
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    [call] = choice.message.tool_calls
    assert call.id.startswith("call_") and call.type == "function" and call.function.name == "calculator"
    assert json.loads(call.function.arguments) == {"expression": "347 + 58"}

    messages += [choice.message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": "405"}]  # we "ran" the calculator
    second = c.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    assert second.choices[0].message.content == "The answer is 405."
    sent = env.stub.state.payloads[-1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "calculator"
    assert sent[2] == {"role": "tool", "content": "405", "tool_call_id": call.id}

    # tool_choice="none" hides the tools from the model
    c.chat.completions.create(model=MODEL, messages=messages[:1], tools=tools, tool_choice="none")
    assert env.stub.state.payloads[-1]["tools"] is None


def test_coding_agent_model(env):
    """A coding agent's long prompt goes through (the server fits it to the context), the model's
    default temperature applies, and arguments come back typed by the request's schemas."""
    t = new_org()
    c = client(env, t.secret)
    tools = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "replaceAll": {"type": "boolean"}}}}}]
    messages = [{"role": "system", "content": "You are a coding agent. " + "x" * 40_000},
                {"role": "user", "content": "fix calc.py"}]
    for stream in (False, True):
        if stream:
            chunks = list(c.chat.completions.create(model=CODE_MODEL, messages=messages, tools=tools, stream=True))
            call = next(ch.choices[0].delta.tool_calls[0] for ch in chunks if ch.choices and ch.choices[0].delta.tool_calls)
        else:
            call = c.chat.completions.create(model=CODE_MODEL, messages=messages, tools=tools).choices[0].message.tool_calls[0]
        assert json.loads(call.function.arguments) == {"path": "calc.py", "oldString": "a - b", "newString": "a + b",
                                                       "replaceAll": True}
        assert env.stub.state.payloads[-1]["temperature"] == 0.0


def test_streamed_tool_calls(env):
    t = new_org()
    tools = [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}]
    chunks = list(ask(env, t.secret, "calc it", tools=tools, stream=True))
    assert chunks[0].choices[0].delta.role == "assistant" and chunks[0].choices[0].delta.content is None
    [call_chunk] = [c for c in chunks if c.choices[0].delta.tool_calls]
    [call] = call_chunk.choices[0].delta.tool_calls
    assert call.index == 0 and call.id.startswith("call_") and call.type == "function"
    assert call.function.name == "calculator" and json.loads(call.function.arguments) == {"expression": "347 + 58"}
    assert chunks[-1].choices[0].finish_reason == "tool_calls"

    # The SDK's streaming helper accumulates the same message as the non-streaming API
    with client(env, t.secret).chat.completions.stream(
            model=MODEL, messages=[{"role": "user", "content": "calc it"}], tools=tools) as stream:
        final = stream.get_final_completion()
    assert final.choices[0].message.tool_calls[0].function.name == "calculator"


def test_reasoning_content(env):
    t = new_org()
    message = ask(env, t.secret, "think: 2 + 2?").choices[0].message
    assert message.content == "4" and message.reasoning_content == "two plus two is four"
    chunks = list(ask(env, t.secret, "think: 2 + 2?", stream=True))
    reasoning = "".join((c.choices[0].delta.model_extra or {}).get("reasoning_content") or "" for c in chunks)
    assert reasoning == "two plus two is four"
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "4"


# ---- errors --------------------------------------------------------------------

def test_authentication_errors(env):
    with pytest.raises(openai.AuthenticationError) as e:
        ask(env, "sk-mini-this-key-does-not-exist-1234")
    assert e.value.code == "invalid_api_key"
    assert "sk-mini-this...1234" in e.value.message and "does-not-exist" not in e.value.message
    assert e.value.response.headers["x-request-id"].startswith("req_")

    r = httpx.post(f"{env.url}/v1/chat/completions", json={"model": MODEL, "messages": []})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"

    t = new_org()
    ask(env, t.secret)
    assert db.revoke_api_key(t.org["id"], t.key["id"])
    with pytest.raises(openai.AuthenticationError):
        ask(env, t.secret)
    with pytest.raises(openai.AuthenticationError):
        client(env, t.secret).models.list()


@pytest.mark.parametrize("kwargs, param", [
    ({"n": 2}, "n"),
    ({"logprobs": True}, "logprobs"),
    ({"response_format": {"type": "json_object"}}, "response_format"),
    ({"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "f"}}]}, "tool_choice"),
    ({"temperature": 5}, "temperature"),
    ({"tools": [{"type": "function", "function": {"name": "bad name!"}}]}, "tools[0].function.name"),
])
def test_unsupported_parameters_are_rejected(env, kwargs, param):
    t = new_org()
    with pytest.raises(openai.BadRequestError) as e:
        ask(env, t.secret, **kwargs)
    assert e.value.body["param"] == param and e.value.body["type"] == "invalid_request_error"
    [row] = logged(t.org["id"])  # failed requests are logged too, at no cost
    assert row["status_code"] == 400 and row["cost_micros"] == 0 and row["error"]
    assert db.get_balance_micros(t.org["id"]) == 1_000_000


def test_invalid_bodies(env):
    t = new_org()
    auth = {"Authorization": f"Bearer {t.secret}"}
    r = httpx.post(f"{env.url}/v1/chat/completions", headers=auth, content=b"{not json")
    assert r.status_code == 400 and "JSON" in r.json()["error"]["message"]
    r = httpx.post(f"{env.url}/v1/chat/completions", headers=auth, json={"model": MODEL})
    assert r.status_code == 400 and r.json()["error"]["message"] == "Missing required parameter: 'messages'."
    r = httpx.post(f"{env.url}/v1/chat/completions", headers=auth, json={
        "model": MODEL, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
    assert r.status_code == 400 and r.json()["error"]["param"] == "messages[0].content"
    r = httpx.post(f"{env.url}/v1/chat/completions", headers=auth, json={
        "model": MODEL, "messages": [{"role": "robot", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["error"]["param"] == "messages[0].role"
    assert [row["status_code"] for row in logged(t.org["id"])] == [400] * 4
    r = httpx.get(f"{env.url}/v1/nope", headers=auth)
    assert r.status_code == 404 and r.json()["error"]["message"] == "Invalid URL (GET /v1/nope)"


def test_unknown_model(env):
    t = new_org()
    with pytest.raises(openai.NotFoundError) as e:
        ask(env, t.secret, model="gpt-5")
    assert e.value.code == "model_not_found" and "gpt-5" in e.value.message
    [row] = logged(t.org["id"])
    assert row["status_code"] == 404 and row["model"] == "gpt-5" and row["cost_micros"] == 0


def test_insufficient_quota(env):
    t = new_org(credit_usd=0)
    with pytest.raises(openai.RateLimitError) as e:
        ask(env, t.secret)
    assert e.value.code == "insufficient_quota" and e.value.response.headers["x-should-retry"] == "false"
    assert logged(t.org["id"])[0]["status_code"] == 429

    db.add_credits(t.org["id"], 5, "grant", f"test:{t.org['id']}")  # 5 micros: enough to start one request
    ask(env, t.secret)  # goes through, and takes the balance below zero
    assert db.get_balance_micros(t.org["id"]) < 0
    with pytest.raises(openai.RateLimitError):
        ask(env, t.secret)


def test_key_spend_limit(env):
    t = new_org(spend_limit_usd=0.000001)  # 1 micro-dollar
    ask(env, t.secret)
    with pytest.raises(openai.RateLimitError) as e:
        ask(env, t.secret)
    assert e.value.code == "insufficient_quota" and "spend limit" in e.value.message


def test_requests_per_minute_limit(env):
    t = new_org(rpm_limit=2)
    url, auth = f"{env.url}/v1/chat/completions", {"Authorization": f"Bearer {t.secret}"}
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}
    r1 = httpx.post(url, headers=auth, json=body)
    assert r1.status_code == 200
    assert (r1.headers["x-ratelimit-limit-requests"], r1.headers["x-ratelimit-remaining-requests"]) == ("2", "1")
    assert r1.headers["x-ratelimit-reset-requests"].endswith("s")
    assert httpx.post(url, headers=auth, json=body).status_code == 200
    r3 = httpx.post(url, headers=auth, json=body)
    assert r3.status_code == 429
    err = r3.json()["error"]
    assert err["code"] == "rate_limit_exceeded" and err["type"] == "requests"
    assert r3.headers["x-ratelimit-remaining-requests"] == "0" and int(r3.headers["retry-after"]) >= 1
    assert r3.headers["x-request-id"].startswith("chatcmpl-")
    # Rate-limited requests only count in the metrics: logging them would let a caller fill the DB for free.
    assert [row["status_code"] for row in logged(t.org["id"])] == [200, 200]
    with pytest.raises(openai.RateLimitError):
        ask(env, t.secret)


def test_tokens_per_minute_limit_reserves_then_settles(env):
    t = new_org(tpm_limit=1000)
    raw = client(env, t.secret).chat.completions.with_raw_response.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=100)
    usage = raw.parse().usage
    # the reservation (prompt estimate + 100) was given back: only real usage counts
    assert int(raw.headers["x-ratelimit-remaining-tokens"]) >= 1000 - usage.total_tokens - 1
    assert raw.headers["x-ratelimit-limit-tokens"] == "1000"

    with pytest.raises(openai.RateLimitError) as e:  # max_tokens alone (capped at the context) > TPM
        ask(env, new_org(tpm_limit=100).secret, max_tokens=200)
    assert "Request too large" in e.value.message


def test_first_party_auth(env):
    t = new_org()
    fp = client(env, TOKEN, default_headers={"X-Minilab-Org": t.org["id"], "X-Minilab-Project": t.project["id"],
                                              "X-Minilab-Source": "chat"})
    completion = fp.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    [row] = logged(t.org["id"])
    assert row["api_key_id"] is None and row["source"] == "chat" and row["project_id"] == t.project["id"]
    assert row["cost_micros"] == PRICING.cost_micros(completion.usage.prompt_tokens, completion.usage.completion_tokens)
    assert db.get_balance_micros(t.org["id"]) == 1_000_000 - row["cost_micros"]
    assert db.lookup_api_key(t.secret)["spend_micros"] == 0

    playground = client(env, TOKEN, default_headers={"X-Minilab-Org": t.org["id"]})
    playground.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    assert logged(t.org["id"])[0]["source"] == "playground"  # the default
    platform = client(env, TOKEN, default_headers={"X-Minilab-Org": t.org["id"], "X-Minilab-Source": "platform"})
    assert [m.id for m in platform.models.list()] == [MODEL, CODE_MODEL]  # what the platform's model picker does

    for headers in ({}, {"X-Minilab-Org": "org_nope"}, {"X-Minilab-Org": t.org["id"], "X-Minilab-Source": "Not A Source!"},
                    {"X-Minilab-Org": t.org["id"], "X-Minilab-Project": new_org().project["id"]}):
        with pytest.raises(openai.BadRequestError):
            client(env, TOKEN, default_headers=headers).chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "Hi"}])


def test_upstream_errors(env):
    t = new_org()
    with pytest.raises(openai.InternalServerError) as e:
        ask(env, t.secret, "overload me")
    assert e.value.status_code == 503 and e.value.code == "overloaded"
    with pytest.raises(openai.InternalServerError) as e:
        ask(env, t.secret, "overload me", stream=True)
    assert e.value.status_code == 503

    # The stream fails after the first token: the SDK raises, the request is logged as failed, and
    # what the client already received is billed (otherwise breaking a stream on purpose would be free).
    with pytest.raises(openai.APIError, match="out of memory"):
        list(ask(env, t.secret, "crash", stream=True))
    row = wait_for(lambda: next((r for r in logged(t.org["id"]) if r["status_code"] == 502), None))
    assert "out of memory" in row["error"] and row["completion_tokens"] == 1 and row["cost_micros"] > 0
    assert [r["status_code"] for r in logged(t.org["id"])].count(503) == 2
    assert db.get_balance_micros(t.org["id"]) == 1_000_000 - row["cost_micros"]


def test_inference_down(env):
    port = free_port()  # nothing listens there
    settings = replace(env.settings, inference_url=f"http://127.0.0.1:{port}")
    with serve(create_app(settings), free_port()) as url:
        t = new_org()
        c = openai.OpenAI(base_url=f"{url}/v1", api_key=t.secret, max_retries=0)
        with pytest.raises(openai.InternalServerError) as e:
            c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        assert e.value.status_code == 503 and e.value.code == "inference_unavailable"
        assert logged(t.org["id"])[0]["status_code"] == 503
        health = httpx.get(f"{url}/health").json()
        assert health == {"status": "degraded", "inference": "unreachable", "db": "ok"}


def test_client_disconnect_bills_what_was_generated(env):
    t = new_org()
    disconnected = env.stub.state.disconnected
    with httpx.stream("POST", f"{env.url}/v1/chat/completions", headers={"Authorization": f"Bearer {t.secret}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "slow please"}],
                            "stream": True}) as r:
        lines = r.iter_lines()
        seen = [line for line, _ in zip(lines, range(8)) if line.startswith("data:")]
    assert len(seen) >= 3  # role chunk + a few words, then we hang up

    row = wait_for(lambda: logged(t.org["id"]))[0]
    assert row["status_code"] == 499 and row["error"] == "client disconnected"
    assert 1 <= row["completion_tokens"] < 100 and row["prompt_tokens"] > 0
    assert row["cost_micros"] == PRICING.cost_micros(row["prompt_tokens"], row["completion_tokens"])
    assert db.get_balance_micros(t.org["id"]) == 1_000_000 - row["cost_micros"]
    wait_for(lambda: env.stub.state.disconnected > disconnected)  # generation upstream was stopped


# ---- rate limiter (unit, fake clock) -------------------------------------------------

def test_rate_limiter_buckets_refill_and_settle():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    caller = Caller(org_id="org_x", project_id=None, api_key_id="key_x", source="api", balance_micros=1,
                    rpm_limit=2, tpm_limit=600)
    limiter.acquire(caller, 500)
    assert limiter.headers(caller)["x-ratelimit-remaining-tokens"] == "100"
    limiter.settle(caller, reserved=500, used=50)  # the request used far less than it reserved
    assert limiter.headers(caller)["x-ratelimit-remaining-tokens"] == "550"
    with pytest.raises(APIError, match="Request too large"):  # more than the TPM limit: can never fit
        limiter.acquire(caller, 601)
    limiter.acquire(caller, 10)
    with pytest.raises(APIError) as e:  # 2 requests per minute: the third must wait 30 s for a refill
        limiter.acquire(caller, 10)
    assert e.value.status == 429 and e.value.headers["retry-after"] == "30"
    h = limiter.headers(caller)
    assert (h["x-ratelimit-remaining-requests"], h["x-ratelimit-reset-requests"]) == ("0", "1m0s")
    now[0] = 30.0  # limit/60 per second: one request back after 30 s
    limiter.acquire(caller, 10)
    caller.rpm_limit = 5  # limits edited in the dashboard apply to the existing bucket
    assert limiter.headers(caller)["x-ratelimit-limit-requests"] == "5"


def test_stream_is_closed_and_billed_even_if_client_left_before_it_started():
    class Upstream:  # an inference stream that never produces anything
        closed = False

        async def aiter_lines(self):
            await asyncio.sleep(3600)
            yield ""

        async def aclose(self):
            Upstream.closed = True

    caller = Caller(org_id="org_x", project_id=None, api_key_id="key_x", source="api", balance_micros=1,
                    rpm_limit=10, tpm_limit=1000)
    meter, recorded = Meter(id="chatcmpl-x", caller=caller), []

    async def record(status, *args, **kwargs):
        recorded.append(status)

    meter.record = record
    req = ChatCompletionRequest.model_validate({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    response = SSEResponse(StreamRelay(RateLimiter(), meter, req, Upstream(), reserved=10, prompt_estimate=5))

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    asyncio.run(response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send))
    assert Upstream.closed and recorded == [499]


def test_format_duration():
    assert [format_duration(s) for s in (0.02, 1.5, 12, 90, 360)] == ["20ms", "1.5s", "12s", "1m30s", "6m0s"]


# ---- models, ops ---------------------------------------------------------------

def test_models(env):
    t = new_org()
    c = client(env, t.secret)
    model, code_model = c.models.list().data   # newest first
    assert (model.id, model.object, model.owned_by) == (MODEL, "model", "mini-lab")
    assert code_model.id == CODE_MODEL
    assert model.model_extra["pricing"] == {"input_per_1m": 100.0, "output_per_1m": 300.0}
    assert model.model_extra["context_length"] == 256
    assert c.models.retrieve(MODEL).id == MODEL
    with pytest.raises(openai.NotFoundError):
        c.models.retrieve("nope")
    assert httpx.get(f"{env.url}/v1/models").status_code == 401


def test_health_metrics_and_cors(env):
    assert httpx.get(f"{env.url}/health").json() == {"status": "ok", "inference": "ok", "db": "ok"}
    ask(env, new_org().secret)
    text = httpx.get(f"{env.url}/metrics").text
    assert f'minilab_api_requests_total{{model="{MODEL}",status="200"}}' in text
    assert f'minilab_api_tokens_total{{model="{MODEL}",kind="completion"}}' in text
    assert f'minilab_api_cost_micros_total{{model="{MODEL}"}}' in text
    assert f'minilab_api_request_duration_seconds_count{{model="{MODEL}"}}' in text
    assert 'minilab_api_http_requests_total{route="/v1/chat/completions",status="200"}' in text

    preflight = httpx.options(f"{env.url}/v1/chat/completions", headers={
        "Origin": "https://example.com", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})
    assert preflight.status_code == 200 and preflight.headers["access-control-allow-origin"] == "*"
    r = httpx.get(f"{env.url}/v1/models", headers={"Origin": "https://example.com"})
    assert r.headers["access-control-allow-origin"] == "*"
    assert "x-request-id" in r.headers["access-control-expose-headers"]


# ---- end to end, with the real inference server ------------------------------------

@pytest.fixture(scope="module")
def real_gateway(env, tmp_path_factory):
    """The gateway in front of the real inference server serving a random-weight model."""
    if importlib.util.find_spec("minilab.inference.server") is None:
        pytest.skip("minilab.inference.server is not available")
    from minilab.testing import make_random_release

    tmp = tmp_path_factory.mktemp("real")
    make_random_release(tmp / "models")
    port = free_port()
    with open(tmp / "inference.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "minilab.inference", "--port", str(port)], stdout=log,
                                stderr=subprocess.STDOUT, env={**os.environ, "MINILAB_MODELS_DIR": str(tmp / "models"),
                                                               "MINILAB_INTERNAL_TOKEN": TOKEN})
    try:
        inference_url = f"http://127.0.0.1:{port}"
        wait_for(lambda: proc.poll() is not None or _healthy(inference_url), timeout=60)
        assert proc.poll() is None, (tmp / "inference.log").read_text()
        # Same database as `env`: the db module points at one file per process.
        settings = replace(env.settings, inference_url=inference_url, models_dir=str(tmp / "models"))
        with serve(create_app(settings), free_port()) as url:
            yield url
    finally:
        proc.terminate()
        proc.wait(10)


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def test_end_to_end_with_real_inference(real_gateway):
    t = new_org()
    c = openai.OpenAI(base_url=f"{real_gateway}/v1", api_key=t.secret, max_retries=0)
    [model] = c.models.list().data
    assert model.id == "mini-random"
    pricing = Pricing(**model.model_extra["pricing"])  # what the gateway bills with
    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    completion = c.chat.completions.create(model="mini-random", messages=messages, max_tokens=12, seed=3)
    assert completion.choices[0].finish_reason in ("stop", "length") and completion.usage.completion_tokens <= 12
    chunks = list(c.chat.completions.create(model="mini-random", messages=messages, max_tokens=12, seed=3,
                                            stream=True, stream_options={"include_usage": True}))
    # same seed, same tokens: streaming reassembles exactly the non-streamed answer
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices) == completion.choices[0].message.content
    assert chunks[-1].usage == completion.usage

    expected = 2 * pricing.cost_micros(completion.usage.prompt_tokens, completion.usage.completion_tokens)
    assert db.get_balance_micros(t.org["id"]) == 1_000_000 - expected
    assert [r["status_code"] for r in logged(t.org["id"])] == [200, 200]
    with pytest.raises(openai.BadRequestError) as e:
        c.chat.completions.create(model="mini-random", messages=[{"role": "user", "content": "word " * 1000}])
    assert e.value.code == "context_length_exceeded"


def test_rejected_requests_still_count_against_rpm(env):
    """Unknown models and bad bodies are rejected, but they cost a request: no free way to hit the DB."""
    t = new_org(rpm_limit=2)
    url, auth = f"{env.url}/v1/chat/completions", {"Authorization": f"Bearer {t.secret}"}
    statuses = [httpx.post(url, headers=auth, json={"model": "nope", "messages": [{"role": "user", "content": "Hi"}]}).status_code
                for _ in range(3)]
    assert statuses == [404, 404, 429]
    assert len(logged(t.org["id"])) == 2


def test_oversized_bodies_are_rejected_at_the_gateway(env):
    t = new_org()
    r = httpx.post(f"{env.url}/v1/chat/completions", headers={"Authorization": f"Bearer {t.secret}"},
                   content=b"{" + b" " * 2_000_000 + b"}", timeout=30)
    assert r.status_code == 413
    assert db.get_balance_micros(t.org["id"]) == 1_000_000
