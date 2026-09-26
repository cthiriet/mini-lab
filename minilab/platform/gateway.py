"""The platform's client for the API gateway (playground, chat app, status page).

The platform never talks to the inference server directly. It calls the public
gateway like any customer would, but authenticates as a first-party service:

    Authorization: Bearer $MINILAB_INTERNAL_TOKEN
    X-Minilab-Org: org_...            <- who gets billed
    X-Minilab-Source: playground|chat <- shown in the logs

So playground and chat traffic is metered, rate-limited, logged and billed exactly
like API traffic, with a single code path in the gateway.

`stream_chat` also runs the tool loop: when the model asks for the calculator, we
run it here, append the result to the conversation and call the model again.
The browser receives a simple stream of events (see `stream_chat`).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from minilab.chat.tools import CALCULATOR_TOOL, run_tool

MAX_TOOL_ROUNDS = 3  # model -> tool -> model -> tool -> model, then we stop asking


class Gateway:
    def __init__(self, base_url: str, internal_token: str, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self.transport = transport  # tests plug a stub gateway in here (httpx.ASGITransport)

    def _client(self, timeout: httpx.Timeout | float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, transport=self.transport, timeout=timeout)

    def _headers(self, org_id: str, source: str) -> dict:
        return {
            "Authorization": f"Bearer {self.internal_token}",
            "X-Minilab-Org": org_id,
            "X-Minilab-Source": source,
        }

    async def health(self) -> dict:
        """{"ok": bool, "latency_ms": int | None, "detail": str}"""
        started = time.monotonic()
        try:
            async with self._client(timeout=3) as client:
                resp = await client.get("/health")
            ok = resp.status_code == 200
            detail = resp.json().get("status", "ok") if ok else f"HTTP {resp.status_code}"
        except (httpx.HTTPError, ValueError) as e:
            return {"ok": False, "latency_ms": None, "detail": f"unreachable ({type(e).__name__})"}
        return {"ok": ok, "latency_ms": round((time.monotonic() - started) * 1000), "detail": detail}

    async def model_ids(self, org_id: str) -> list[str]:
        """Models the gateway is serving right now ([] if it is down)."""
        try:
            async with self._client(timeout=3) as client:
                resp = await client.get("/v1/models", headers=self._headers(org_id, "platform"))
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    async def stream_chat(self, *, org_id: str, source: str, model: str, messages: list[dict],
                          temperature: float | None = None, max_tokens: int | None = None,
                          calculator: bool = False) -> AsyncIterator[dict]:
        """Stream a chat completion as browser-friendly events:

            {"type": "delta", "content": "..."}             visible answer text
            {"type": "delta", "reasoning": "..."}           scratchpad text
            {"type": "tool", "name", "input", "output", "ok"}  a calculator call we ran
            {"type": "error", "message", "code", "status"}  terminal
            {"type": "done", "finish_reason", "usage", "request_ids", "latency_ms", "ttft_ms"}
        """
        messages = list(messages)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        request_ids: list[str] = []
        started = time.monotonic()
        ttft_ms = None
        finish_reason = None

        async with self._client(timeout=httpx.Timeout(10, read=300)) as client:
            for round_no in range(MAX_TOOL_ROUNDS):
                body: dict = {"model": model, "messages": messages, "stream": True,
                              "stream_options": {"include_usage": True}}
                if temperature is not None:
                    body["temperature"] = temperature
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                if calculator:
                    body["tools"] = [CALCULATOR_TOOL]
                    # Last round: force a plain answer so a confused model can't loop forever.
                    body["tool_choice"] = "none" if round_no == MAX_TOOL_ROUNDS - 1 else "auto"

                turn = _Turn()
                try:
                    async with client.stream("POST", "/v1/chat/completions", json=body,
                                             headers=self._headers(org_id, source)) as resp:
                        if resp.status_code != 200:
                            yield _error_from_response(resp.status_code, await resp.aread())
                            return
                        async for chunk in _sse_json(resp):
                            if "error" in chunk:
                                yield _error_event(chunk["error"], status=500)
                                return
                            for event in turn.feed(chunk):
                                if ttft_ms is None:
                                    ttft_ms = round((time.monotonic() - started) * 1000)
                                yield event
                except httpx.HTTPError as e:
                    yield {"type": "error", "status": 502, "code": "gateway_unreachable",
                           "message": f"Could not reach the API gateway at {self.base_url} ({type(e).__name__})."}
                    return

                if turn.id:
                    request_ids.append(turn.id)
                usage["prompt_tokens"] += turn.usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += turn.usage.get("completion_tokens", 0)
                finish_reason = turn.finish_reason
                if not (calculator and turn.finish_reason == "tool_calls" and turn.tool_calls):
                    break

                # The model asked for tools: run them and send the results back.
                calls = list(turn.tool_calls.values())
                messages.append({"role": "assistant", "content": turn.content or None, "tool_calls": calls})
                for call in calls:
                    result = run_tool(call["function"]["name"], call["function"]["arguments"])
                    yield {"type": "tool", **result}
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result["output"]})

        yield {"type": "done", "finish_reason": finish_reason, "usage": usage, "request_ids": request_ids,
               "latency_ms": round((time.monotonic() - started) * 1000), "ttft_ms": ttft_ms}


class _Turn:
    """Accumulates one streamed completion (OpenAI chunk format)."""

    def __init__(self):
        self.id: str | None = None
        self.content = ""
        self.tool_calls: dict[int, dict] = {}  # by index: tool call fragments arrive in pieces
        self.finish_reason: str | None = None
        self.usage: dict = {}

    def feed(self, chunk: dict) -> list[dict]:
        self.id = self.id or chunk.get("id")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        events = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                events.append({"type": "delta", "reasoning": delta["reasoning_content"]})
            if delta.get("content"):
                self.content += delta["content"]
                events.append({"type": "delta", "content": delta["content"]})
            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                call = self.tool_calls.setdefault(index, {
                    "id": f"call_{index}", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] = tc.get("id") or call["id"]
                fn = tc.get("function") or {}
                call["function"]["name"] += fn.get("name") or ""
                call["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events


async def _sse_json(resp: httpx.Response) -> AsyncIterator[dict]:
    """Yield the JSON payload of each `data:` line until `data: [DONE]`."""
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue  # blank separators, comments (": keep-alive"), event names
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _error_from_response(status: int, raw: bytes) -> dict:
    try:
        error = json.loads(raw)["error"]
    except (ValueError, KeyError, TypeError):
        error = {"message": raw.decode(errors="replace")[:300] or f"HTTP {status}"}
    return _error_event(error, status)


def _error_event(error: dict | str, status: int) -> dict:
    if not isinstance(error, dict):
        error = {"message": str(error)}
    return {"type": "error", "status": status, "code": error.get("code") or error.get("type"),
            "message": error.get("message") or "The API gateway returned an error."}
