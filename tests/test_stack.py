"""End to end: the three real services, wired together like in production.

sign up on the platform -> create an API key -> call the gateway with the official
openai SDK -> the request is logged and billed -> buy credits -> chat app -> revoke the key.
Uses a random-weight model, so we check plumbing and billing, not answer quality.
"""

import json
import os
import re
import socket
import subprocess
import sys
import time

import httpx
import openai
import pytest

from minilab import db
from minilab.registry import get_model
from minilab.testing import make_random_release

TOKEN = "stack-test-internal-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{url} exited early:\n{proc.stdout.read()}")
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(url)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stack")
    make_random_release(tmp / "models", "mini-random")
    ports = {name: free_port() for name in ("inference", "api", "platform")}
    urls = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
    env = {
        **os.environ,
        "MINILAB_DB": str(tmp / "stack.db"),
        "MINILAB_MODELS_DIR": str(tmp / "models"),
        "MINILAB_INTERNAL_TOKEN": TOKEN,
        "MINILAB_INFERENCE_URL": urls["inference"],
        "MINILAB_API_URL": urls["api"],
        "MINILAB_PLATFORM_URL": urls["platform"],
    }
    env.pop("STRIPE_SECRET_KEY", None)
    procs = []
    try:
        for name in ("inference", "api", "platform"):
            proc = subprocess.Popen([sys.executable, "-m", f"minilab.{name}", "--port", str(ports[name])],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            procs.append(proc)
            wait_healthy(urls[name], proc)
        db.configure(tmp / "stack.db")
        yield urls, tmp
    finally:
        db.configure(None)
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)


def test_signup_key_call_bill_revoke(stack):
    urls, tmp = stack
    web = httpx.Client(base_url=urls["platform"], headers={"Origin": urls["platform"]}, follow_redirects=True)

    # 1. sign up: an org with free credits is created
    r = web.post("/signup", data={"name": "Ada", "email": "ada@example.com", "password": "correct horse"})
    assert r.status_code == 200 and r.url.path == "/overview"
    user = db.authenticate("ada@example.com", "correct horse")
    org = db.list_user_orgs(user["id"])[0]
    project = db.list_projects(org["id"])[0]
    start_balance = db.get_balance_micros(org["id"])
    assert start_balance == 1_000_000

    # 2. create an API key in the dashboard: the secret is shown once
    r = web.post("/api-keys", data={"name": "e2e", "project_id": project["id"]})
    secret = re.search(r"sk-mini-[A-Za-z0-9_-]{20,}", r.text).group(0)

    # 3. call the gateway with the official SDK, streaming and not
    client = openai.OpenAI(base_url=urls["api"] + "/v1", api_key=secret, max_retries=0)
    assert [m.id for m in client.models.list()] == ["mini-random"]
    resp = client.chat.completions.create(model="mini-random", max_tokens=16, seed=1,
                                          messages=[{"role": "user", "content": "What is 2 + 2?"}])
    assert resp.usage.completion_tokens > 0
    chunks = list(client.chat.completions.create(model="mini-random", max_tokens=16, seed=1, stream=True,
                                                 stream_options={"include_usage": True},
                                                 messages=[{"role": "user", "content": "What is 2 + 2?"}]))
    usage = chunks[-1].usage
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == resp.choices[0].message.content

    # 4. both requests are logged and billed at the model's price
    pricing = get_model(tmp / "models", "mini-random").pricing
    cost = pricing.cost_micros(resp.usage.prompt_tokens, resp.usage.completion_tokens) \
        + pricing.cost_micros(usage.prompt_tokens, usage.completion_tokens)
    assert db.get_balance_micros(org["id"]) == start_balance - cost
    assert resp.id in web.get("/logs").text

    # 5. test-mode checkout adds credits
    web.post("/billing/checkout", data={"amount_usd": 5, "nonce": "0123456789abcdef"})
    assert db.get_balance_micros(org["id"]) == start_balance - cost + 5_000_000

    # 6. the chat app streams through the gateway, billed to the org as first-party traffic
    with web.stream("POST", "/chat/api/chat", json={"model": "mini-random", "max_tokens": 8,
                                                     "messages": [{"role": "user", "content": "Hi!"}]}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert any(req["source"] == "chat" for req in db.list_requests(org["id"]))

    # 7. a revoked key stops working immediately
    key_id = db.list_api_keys(org["id"])[0]["id"]
    web.post(f"/api-keys/{key_id}/revoke")
    with pytest.raises(openai.AuthenticationError):
        client.chat.completions.create(model="mini-random", max_tokens=4, messages=[{"role": "user", "content": "Hi"}])
