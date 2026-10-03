"""POST /v1/chat/completions.

    authenticate -> rate limit (requests) -> validate -> resolve model -> quota
        -> check prompt length -> rate limit (tokens) -> inference -> bill

Every authenticated request is logged, failures included (with cost 0). With
`stream: true` the inference server's events are translated into OpenAI
`chat.completion.chunk`s as they arrive.
"""

from __future__ import annotations

import json

import anyio
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from minilab.api.auth import authenticate, check_quota
from minilab.api.errors import APIError, invalid_request
from minilab.api.metering import Meter
from minilab.api.ratelimit import estimate_prompt_tokens
from minilab.api.schemas import (ChatCompletionRequest, cached_tokens, chunk_json, completion_json, finish_reason,
                                 new_completion_id, parse_chat_request, to_inference, tool_calls_json,
                                 usage_chunk_json, usage_json)
from minilab.api.upstream import stream_events

router = APIRouter()

MAX_BODY_BYTES = 1_000_000


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise invalid_request("We could not parse the JSON body of your request. The body must be valid JSON.") from None
    if not isinstance(body, dict):
        raise invalid_request("The request body must be a JSON object.")
    return body


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    s = request.app.state
    caller = await authenticate(request, s.settings)
    # Picked up by the middleware: x-request-id and x-ratelimit-* go on every response, errors included.
    meter = Meter(id=new_completion_id(), caller=caller)
    request.state.request_id = meter.id
    request.state.ratelimit_headers = s.limiter.headers(caller)

    reserved = 0  # rate-limit tokens taken for this request, settled against real usage at the end
    try:
        # The request counts against the RPM limit before we do any work for it.
        s.limiter.acquire(caller, 0)
        if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            raise APIError(413, f"The request body is too large (max {MAX_BODY_BYTES:,} bytes).")
        body = await _json_body(request)
        meter.request_body = body
        meter.model = str(body.get("model") or "unknown")[:128]
        req = parse_chat_request(body)
        model = await s.catalog.resolve(req.model, s.inference)
        meter.pricing = model.pricing
        check_quota(caller, s.settings.platform_url)

        payload = to_inference(req, model.default_temperature)
        prompt_estimate = estimate_prompt_tokens(payload["messages"])
        # prompt + completion can never exceed the context window, whatever max_tokens says.
        estimate = min(model.context_length, prompt_estimate + (payload["max_tokens"] or model.context_length))
        s.limiter.acquire(caller, estimate, requests=0)
        reserved = estimate
        request.state.ratelimit_headers = s.limiter.headers(caller)

        if req.stream:
            upstream = await s.inference.open_stream(payload)
            return SSEResponse(StreamRelay(s.limiter, meter, req, upstream, reserved, prompt_estimate))
        result = await s.inference.generate(payload)
    except Exception as e:
        s.limiter.settle(caller, reserved, 0)
        err = e if isinstance(e, APIError) else APIError(500, "Internal server error.", type="server_error")
        rate_limited = err.status == 429 and err.code == "rate_limit_exceeded"
        await meter.record(err.status, error=err.message, response_body=err.to_json(), persist=not rate_limited)
        raise

    usage = result.get("usage") or {}
    prompt_tokens, completion_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    completion = completion_json(meter.id, meter.created, result.get("model") or req.model, result,
                                 tool_calls_json(result.get("tool_calls") or [], req.tools))
    s.limiter.settle(caller, reserved, prompt_tokens + completion_tokens)
    request.state.ratelimit_headers = s.limiter.headers(caller)
    await meter.record(200, prompt_tokens, completion_tokens, response_body=completion)
    return JSONResponse(completion)


class StreamRelay:
    """Translates inference stream events into OpenAI chunks, and bills the request exactly once:

        role chunk -> content / reasoning_content deltas -> tool_calls -> finish_reason -> [usage] -> [DONE]
    """

    def __init__(self, limiter, meter: Meter, req: ChatCompletionRequest, upstream: httpx.Response,
                 reserved: int, prompt_estimate: int):
        self.limiter, self.meter, self.req, self.upstream = limiter, meter, req, upstream
        self.reserved, self.prompt_estimate = reserved, prompt_estimate
        self.streamed = {"content": "", "reasoning": ""}  # text relayed so far
        self.n_deltas = 0
        self.recorded = False

    def _chunk(self, delta: dict, finish: str | None = None) -> str:
        m = self.meter
        return _sse(chunk_json(m.id, m.created, self.req.model, delta, finish, self.req.include_usage))

    async def _record(self, status: int, p: int, c: int, error: str | None, response_body) -> None:
        self.recorded = True
        # Shielded: after a client disconnect we run inside a cancelled scope, where any await
        # would be cancelled too, and the request would go unbilled.
        with anyio.CancelScope(shield=True):
            await self.upstream.aclose()  # after a disconnect, this is what tells inference to stop
            self.limiter.settle(self.meter.caller, self.reserved, p + c)
            await self.meter.record(status, p, c, error=error, response_body=response_body)

    async def close(self) -> None:
        """Called when the response ends, however it ends. If the stream didn't get to record the
        request, the client went away first: bill what we know was generated (see metering.py)."""
        if not self.recorded:
            p, c = (self.prompt_estimate, self.n_deltas) if self.n_deltas else (0, 0)
            await self._record(499, p, c, "client disconnected", {"partial": self.streamed})

    async def chunks(self):
        m, req = self.meter, self.req
        done: dict | None = None  # the final inference event, once received
        error: str | None = None
        try:
            async for event in stream_events(self.upstream):
                if event.get("type") == "error":
                    error = event.get("message") or "generation failed"
                    break
                if m.ttft_s is None:
                    m.first_token()
                    # The first chunk carries the role; like OpenAI, content is null when the answer
                    # is only tool calls and "" otherwise.
                    only_tools = event.get("type") == "done" and not event.get("content") and event.get("tool_calls")
                    yield self._chunk({"role": "assistant", "content": None if only_tools else ""})
                if event.get("type") == "delta":
                    delta = {}
                    for key, field in (("content", "content"), ("reasoning", "reasoning_content")):
                        if event.get(key):
                            self.streamed[key] += event[key]
                            delta[field] = event[key]
                    if delta:
                        self.n_deltas += 1
                        yield self._chunk(delta)
                elif event.get("type") == "done":
                    done = event
                    break
        except APIError as e:  # malformed event
            error = e.message
        except httpx.HTTPError as e:  # connection to inference lost mid-stream
            error = f"lost connection to the inference server ({type(e).__name__})"

        if done is None:
            err = APIError(502, error or "the inference stream ended unexpectedly", type="server_error",
                           code="upstream_error")
            # Bill what the client already received (like a disconnect), so breaking a stream isn't free.
            p, c = (self.prompt_estimate, self.n_deltas) if self.n_deltas else (0, 0)
            await self._record(502, p, c, err.message, err.to_json())
            yield _sse(err.to_json())  # the SDKs raise an APIError when a chunk carries an "error" key
            return

        # Bill before sending the last chunks, so a client that got [DONE] already sees its new balance.
        tool_calls = tool_calls_json(done.get("tool_calls") or [], req.tools)
        usage = done.get("usage") or {}
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        await self._record(200, p, c, None, completion_json(m.id, m.created, req.model, done, tool_calls))

        # `done` carries the full text: send anything the deltas didn't cover.
        for key, field in (("content", "content"), ("reasoning", "reasoning_content")):
            full, sent = done.get(key) or "", self.streamed[key]
            if len(full) > len(sent) and full.startswith(sent):
                yield self._chunk({field: full[len(sent):]})
        # Tool calls are only known at the end: one chunk per call, arguments in one piece.
        for i, call in enumerate(tool_calls):
            yield self._chunk({"tool_calls": [{"index": i, **call}]})
        yield self._chunk({}, finish_reason(done))
        if req.include_usage:
            yield _sse(usage_chunk_json(m.id, m.created, req.model, usage_json(p, c, cached_tokens(usage))))
        yield "data: [DONE]\n\n"


class SSEResponse(StreamingResponse):
    """Streams a StreamRelay and always closes it, however the response ends.

    When a client disconnects, Starlette cancels the response, possibly before the
    relay even started. Cleaning up here, in __call__ (which always runs), is what
    guarantees the upstream stream is closed and the request is billed.
    """

    def __init__(self, relay: StreamRelay):
        super().__init__(relay.chunks(), media_type="text/event-stream",
                         headers={"cache-control": "no-cache", "x-accel-buffering": "no"})  # no proxy buffering
        self.relay = relay

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await self.relay.close()


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"
