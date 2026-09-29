"""Inference HTTP server: the internal API contract of docs/architecture.md, on a real uvicorn server."""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from minilab.checkpoint import load_checkpoint, save_checkpoint
from minilab.inference.engine import Engine
from minilab.inference.server import create_app
from minilab.settings import get_settings
from minilab.testing import make_random_release

MODEL = "mini-test"
AUTH = {"Authorization": f"Bearer {get_settings().internal_token}"}
DONE_KEYS = {"type", "content", "reasoning", "tool_calls", "finish_reason", "usage"}


def make_lively_release(models_dir, model_id):
    """Random model with scaled-up block weights, so greedy text depends on the prompt."""
    path = make_random_release(models_dir, model_id)
    model, tok, meta = load_checkpoint(path)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("qkv.weight", "fc.weight", "proj.weight")):
                p.mul_(10)
    save_checkpoint(path, model, tok, meta)


def free_port(lo=18020, hi=18099):
    for port in range(lo, hi + 1):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    make_lively_release(models, MODEL)
    engine = Engine(models_dir=models, max_batch=4, max_queue=64)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(500):
        if srv.started:
            break
        time.sleep(0.01)
    assert srv.started
    yield f"http://127.0.0.1:{port}", engine
    srv.should_exit = True
    thread.join(10)


def body(content="What is 2 + 2?", **kw):
    return {"model": MODEL, "messages": [{"role": "user", "content": content}], **kw}


def read_sse(lines):
    events = []
    for line in lines:
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_health_needs_no_auth(server):
    url, _ = server
    r = httpx.get(f"{url}/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("method,path", [("GET", "/models"), ("GET", "/metrics"), ("POST", "/generate")])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "wrong"}])
def test_auth_is_required(server, method, path, headers):
    url, _ = server
    r = httpx.request(method, f"{url}{path}", headers=headers, json=body())
    assert r.status_code == 401
    err = r.json()["error"]
    assert err["code"] == "invalid_api_key" and err["type"] == "invalid_request_error" and err["message"]


def test_models(server):
    url, _ = server
    data = httpx.get(f"{url}/models", headers=AUTH).json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == [MODEL]
    assert data["data"][0]["context_length"] == 256 and "pricing" in data["data"][0]


def test_generate_non_streaming(server):
    url, _ = server
    r = httpx.post(f"{url}/generate", headers=AUTH, json=body(max_tokens=12, temperature=0))
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"model", "content", "reasoning", "tool_calls", "finish_reason", "usage"}
    assert data["model"] == MODEL and isinstance(data["content"], str)
    assert data["tool_calls"] == [] and data["finish_reason"] in ("stop", "length")
    assert data["usage"]["prompt_tokens"] > 0 and 0 < data["usage"]["completion_tokens"] <= 12


def test_generate_streaming_matches_non_streaming(server):
    url, _ = server
    req = body("Tell me a story", max_tokens=40, temperature=0, tools=["calculator"])
    plain = httpx.post(f"{url}/generate", headers=AUTH, json=req).json()
    with httpx.stream("POST", f"{url}/generate", headers=AUTH, json={**req, "stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r.iter_lines())
    *deltas, done = events
    assert all(e["type"] == "delta" and set(e) in ({"type", "content"}, {"type", "reasoning"}) for e in deltas)
    assert done["type"] == "done" and set(done) == DONE_KEYS
    assert "".join(e.get("content", "") for e in deltas) == done["content"]
    # the same text; the second request is served from the prefix cache the first one left
    uncached = lambda d: {**d, "usage": {k: v for k, v in d["usage"].items() if k != "prompt_tokens_details"}}
    assert uncached({k: v for k, v in done.items() if k != "type"}) == uncached({k: v for k, v in plain.items() if k != "model"})


def test_stop_and_seed(server):
    url, _ = server
    ref = httpx.post(f"{url}/generate", headers=AUTH, json=body(max_tokens=60, temperature=0)).json()
    text = ref["content"]
    i = next(i for i in range(8, len(text) - 2) if text[i:i + 2].isascii() and text[i:i + 2].strip())
    stop = text[i:i + 2]
    r = httpx.post(f"{url}/generate", headers=AUTH, json=body(max_tokens=60, temperature=0, stop=stop)).json()
    assert r["content"] == text[:text.index(stop)] and r["finish_reason"] == "stop"

    seeded = body(max_tokens=30, temperature=1.0, seed=1234)
    a, b = (httpx.post(f"{url}/generate", headers=AUTH, json=seeded).json() for _ in range(2))
    assert b["usage"]["prompt_tokens_details"]["cached_tokens"] > 0  # the second one is served from the cache...
    assert a["content"] == b["content"] and a["reasoning"] == b["reasoning"]  # ...and still the same completion


def test_many_concurrent_requests(server):
    url, engine = server

    async def main():
        async with httpx.AsyncClient(base_url=url, headers=AUTH, timeout=30) as client:
            async def one(i):
                if i % 2:
                    r = await client.post("/generate", json=body(f"hello {i}", max_tokens=30, seed=i))
                    return r.status_code, r.json()["finish_reason"]
                async with client.stream("POST", "/generate", json=body(f"hi {i}", max_tokens=30, stream=True)) as r:
                    events = read_sse([line async for line in r.aiter_lines()])
                    return r.status_code, events[-1]["finish_reason"]
            return await asyncio.gather(*(one(i) for i in range(20)))

    results = asyncio.run(main())
    assert all(status == 200 and reason in ("stop", "length") for status, reason in results)
    assert engine.runners[MODEL].num_active == 0


def test_errors(server):
    url, engine = server
    post = lambda **kw: httpx.post(f"{url}/generate", headers=AUTH, json=body(**kw))  # noqa: E731

    r = httpx.post(f"{url}/generate", headers=AUTH, json={**body(), "model": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"

    r = post(content="Once upon a time " * 200)
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"

    for bad in (dict(temperature=-1), dict(max_tokens=0), dict(top_p=2)):
        r = post(**bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request", bad
    r = httpx.post(f"{url}/generate", headers=AUTH, json={"model": MODEL, "messages": [{"role": "robot"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"

    runner = engine.runners[MODEL]
    runner.max_queue = 0  # every new request finds the queue full
    try:
        r = post()
        assert r.status_code == 503 and r.json()["error"] == {
            "message": r.json()["error"]["message"], "type": "server_error", "code": "overloaded"}
    finally:
        runner.max_queue = 64


def test_metrics(server):
    url, _ = server
    httpx.post(f"{url}/generate", headers=AUTH, json=body(max_tokens=5))
    text = httpx.get(f"{url}/metrics", headers=AUTH).text
    for name in ("minilab_inference_requests_total", "minilab_inference_prompt_tokens_total",
                 "minilab_inference_completion_tokens_total", "minilab_inference_active_sequences",
                 "minilab_inference_queue_depth", "minilab_inference_batch_size_bucket",
                 "minilab_inference_time_to_first_token_seconds_bucket",
                 "minilab_inference_request_latency_seconds_bucket"):
        assert f'{name}{{model="{MODEL}"' in text, name
    assert f'minilab_inference_requests_total{{model="{MODEL}",status="200"}}' in text


class SlowModel:
    """Wraps a model so that every forward takes a while (requests stay in flight)."""

    def __init__(self, model, seconds=0.02):
        self.model, self.seconds = model, seconds

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward_cached(self, *args):
        time.sleep(self.seconds)
        return self.model.forward_cached(*args)


def wait_until(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not cond():
        time.sleep(0.02)
    return cond()


def test_client_disconnect_frees_the_slot(server):
    url, engine = server
    runner = engine.runners[MODEL]
    fast, runner.model = runner.model, SlowModel(runner.model)  # ~250 tokens * 20 ms = 5 s per request
    try:
        # Streaming: read one delta, then hang up.
        with httpx.stream("POST", f"{url}/generate", headers=AUTH, json=body(stream=True, temperature=0)) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    break
            assert runner.num_active == 1
        assert wait_until(lambda: runner.num_active == 0, timeout=1.0)  # finishing would take ~5 s

        # Non-streaming: the client gives up before the completion is ready.
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(f"{url}/generate", headers=AUTH, json=body(temperature=0), timeout=0.3)
        assert wait_until(lambda: runner.num_active == 0, timeout=1.5)  # polled every 0.5 s
        assert sorted(runner._free) == list(range(runner.max_batch))
    finally:
        runner.model = fast


def test_tiny_temperature_does_not_break_the_batch(server):
    """A temperature like 1e-39 used to overflow the logits and fail every request in the batch."""
    url, _ = server

    async def main():
        async with httpx.AsyncClient(base_url=url, headers=AUTH, timeout=30) as client:
            return await asyncio.gather(
                client.post("/generate", json=body("hello", max_tokens=30, seed=1)),
                client.post("/generate", json=body("hello", max_tokens=5, temperature=1e-39)),
            )

    for r in asyncio.run(main()):
        assert r.status_code == 200 and r.json()["finish_reason"] in ("stop", "length")


def test_huge_prompt_is_rejected_before_tokenizing(server):
    url, _ = server
    start = time.perf_counter()
    r = httpx.post(url + "/generate", headers=AUTH, json=body("x" * 1_000_000), timeout=30)
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert time.perf_counter() - start < 2  # tokenizing 1 MB in pure Python would take ~a minute
