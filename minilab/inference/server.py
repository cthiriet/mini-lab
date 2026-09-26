"""Internal inference server: the "Internal inference API" of docs/architecture.md.

    POST /generate   chat messages in, completion out (JSON, or server-sent events)
    GET  /models     models currently loaded
    GET  /health     liveness probe (no auth)
    GET  /metrics    Prometheus metrics

Only first-party services (the API gateway) call it, with the shared internal token.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from minilab.inference.engine import Engine, EngineError, SamplingParams
from minilab.obs.metrics import CONTENT_TYPE, Counter, render
from minilab.settings import get_settings
from minilab.tokenizer.chat import render_prompt

REQUESTS = Counter("minilab_inference_requests_total", "Generate requests by final HTTP status", ["model", "status"])


class GenerateRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[str | dict[str, Any]] | None = None  # tool names (dicts are tolerated)
    max_tokens: int | None = Field(None, ge=1)       # None: until the context is full
    temperature: float | None = Field(None, ge=0)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=0)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False


class ApiError(EngineError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def error_response(status: int, code: str, message: str) -> JSONResponse:
    kind = "server_error" if status >= 500 else "invalid_request_error"
    return JSONResponse({"error": {"message": message, "type": kind, "code": code}}, status_code=status)


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine or Engine()
    token = get_settings().internal_token

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(engine.start)  # loads the checkpoints, starts one thread per model
        yield
        await asyncio.to_thread(engine.stop)

    app = FastAPI(title="mini-lab inference", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(EngineError)
    async def on_engine_error(request: Request, e: EngineError):
        return error_response(e.status, e.code, str(e))

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, e: RequestValidationError):
        details = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
        return error_response(400, "invalid_request", f"Invalid request: {details}")

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, e: StarletteHTTPException):
        return error_response(e.status_code, "not_found" if e.status_code == 404 else "invalid_request", str(e.detail))

    def check_token(authorization: Annotated[str | None, Header()] = None) -> None:
        # Constant-time comparison: don't leak the token through response timing.
        if not authorization or not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise ApiError(401, "invalid_api_key", "Missing or invalid internal token.")

    auth = [Depends(check_token)]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/models", dependencies=auth)
    def models():
        return {"object": "list", "data": [r.info.to_json() for r in engine.runners.values()]}

    @app.get("/metrics", dependencies=auth)
    def metrics():
        return PlainTextResponse(render(), media_type=CONTENT_TYPE)

    @app.post("/generate", dependencies=auth)
    async def generate(body: GenerateRequest, request: Request):
        runner = engine.runners.get(body.model)
        label = body.model if runner else "unknown"  # bounded label cardinality
        try:
            if runner is None:
                raise ApiError(404, "model_not_found", f"The model '{body.model}' does not exist or is not loaded.")
            _check_prompt_size(runner, body)
            # BPE encoding is pure Python: keep it off the event loop.
            prompt = await asyncio.to_thread(_render, runner.tokenizer, body)
            params = SamplingParams(
                max_tokens=body.max_tokens,
                temperature=1.0 if body.temperature is None else body.temperature,
                top_k=body.top_k,
                top_p=1.0 if body.top_p is None else body.top_p,
                seed=body.seed,
                stop=[body.stop] if isinstance(body.stop, str) else list(body.stop or []),
            )
            req = runner.submit(prompt, params, stream=body.stream)

            if body.stream:
                async def sse():
                    async for event in req.events():  # leaving early cancels the sequence
                        yield f"data: {json.dumps(event)}\n\n"

                REQUESTS.inc(model=label, status="200")
                # The background task covers a client that disconnects before the stream starts.
                return StreamingResponse(sse(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache"}, background=BackgroundTask(req.cancel))

            done = await _result_unless_disconnected(req, request)
            if done is None:
                REQUESTS.inc(model=label, status="499")
                return Response(status_code=499)  # client closed the request: nobody will read this
        except EngineError as e:
            REQUESTS.inc(model=label, status=str(e.status))
            raise
        REQUESTS.inc(model=label, status="200")
        return {"model": body.model, **{k: v for k, v in done.items() if k != "type"}}

    return app


def _check_prompt_size(runner, body: GenerateRequest) -> None:
    """Reject prompts that can't fit before tokenizing them: BPE encoding is pure Python and
    a huge prompt would hold the GIL for seconds. Every token is at most `longest` bytes, so a
    prompt of more than context * longest bytes needs more tokens than the context holds."""
    longest = max(len(b) for b in runner.tokenizer.vocab.values())
    size = sum(len(json.dumps(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in body.messages)
    if size > runner.info.context_length * longest:
        raise ApiError(400, "context_length_exceeded",
                       f"This model's maximum context length is {runner.info.context_length} tokens, "
                       "and your messages are far longer.")


def _render(tok, body: GenerateRequest) -> list[int]:
    try:
        return render_prompt(tok, body.messages, body.tools)
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise ApiError(400, "invalid_request", f"Invalid messages: {e!r}") from e


async def _result_unless_disconnected(req, request: Request) -> dict | None:
    """Wait for the final event, but give up (and free the slot) if the client goes away.

    Starlette doesn't cancel a regular endpoint when its client disconnects, so we poll.
    """
    task = asyncio.ensure_future(req.result())
    try:
        while True:
            done, _ = await asyncio.wait([task], timeout=0.5)
            if done:
                return task.result()
            if await request.is_disconnected():
                return None
    finally:
        task.cancel()  # no-op if done; otherwise cancels the sequence (see Request.events)
