"""Client for the internal inference API (docs/architecture.md, "Internal inference API").

Upstream failures become OpenAI-shaped errors for our own clients:

    inference unreachable           -> 503 inference_unavailable
    inference 503 (overloaded)      -> 503 overloaded
    inference 400 / 404             -> forwarded as is (bad request / model_not_found)
    anything else (5xx, bad JSON)   -> 502 upstream_error

The SDKs retry 502/503 on their own, which is what we want for a busy server.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from minilab.api.errors import APIError


def _unavailable(detail: str) -> APIError:
    return APIError(503, f"The inference server is unavailable ({detail}). Please retry shortly.",
                    type="server_error", code="inference_unavailable")


def _bad_gateway(detail: str) -> APIError:
    return APIError(502, f"The inference server failed: {detail}", type="server_error", code="upstream_error")


def _error_from_response(status: int, body: bytes) -> APIError:
    try:
        err = json.loads(body)["error"]
        message, type_, code = err["message"], err.get("type", "invalid_request_error"), err.get("code")
    except (ValueError, KeyError, TypeError):
        message, type_, code = body.decode(errors="replace")[:200] or f"HTTP {status}", "server_error", None
    if status in (400, 404):
        return APIError(status, message, type=type_, code=code)
    if status == 503:
        return APIError(503, message, type="server_error", code=code or "overloaded")
    if status in (401, 403):  # our fault, not the caller's: never pass it on as a 401
        return _bad_gateway("the gateway was not allowed in (check MINILAB_INTERNAL_TOKEN).")
    return _bad_gateway(f"HTTP {status}: {message}")


class InferenceClient:
    def __init__(self, base_url: str, token: str):
        # Generous read timeout: a CPU model with a queue in front of it can be slow to start.
        self.http = httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {token}"},
                                      timeout=httpx.Timeout(300.0, connect=5.0, pool=10.0))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _send(self, method: str, path: str, json_body: dict | None = None,
                    stream: bool = False) -> httpx.Response:
        try:
            resp = await self.http.send(self.http.build_request(method, path, json=json_body), stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise _unavailable(type(e).__name__) from None
        except httpx.HTTPError as e:
            raise _bad_gateway(f"{type(e).__name__}: {e}") from None
        if resp.status_code != 200:
            body = await resp.aread()
            await resp.aclose()
            raise _error_from_response(resp.status_code, body)
        return resp

    async def _json(self, method: str, path: str, json_body: dict | None = None) -> dict:
        resp = await self._send(method, path, json_body)
        try:
            return resp.json()
        except ValueError:
            raise _bad_gateway("invalid JSON response.") from None

    async def list_models(self) -> list[dict]:
        return (await self._json("GET", "/models")).get("data", [])

    async def health(self) -> bool:
        try:  # short timeout: a health check must answer fast even if inference hangs
            return (await self.http.get("/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def generate(self, payload: dict) -> dict:
        return await self._json("POST", "/generate", {**payload, "stream": False})

    async def open_stream(self, payload: dict) -> httpx.Response:
        """Start a streaming generation. Errors that happen before the first byte (model not found,
        overloaded, ...) raise here, so the client gets a proper HTTP status instead of a 200 stream.
        The caller must close the returned response."""
        return await self._send("POST", "/generate", {**payload, "stream": True}, stream=True)


async def stream_events(resp: httpx.Response) -> AsyncIterator[dict]:
    """The JSON objects of an SSE stream (`data: {...}` lines)."""
    async for line in resp.aiter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except ValueError:
                raise _bad_gateway(f"invalid stream event: {data[:100]!r}") from None
