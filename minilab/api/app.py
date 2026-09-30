"""The public, OpenAI-compatible API gateway.

    uv run python -m minilab.api [--host 127.0.0.1] [--port 8000]

Clients (the openai SDK, curl, the platform's playground and chat app) talk to
this service. It authenticates them, enforces quotas and rate limits, forwards
generation to the internal inference server and meters every request into the
database. The official `openai` SDK works against it by only changing base_url:

    OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-mini-...")

See docs/api.md for the reference.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from minilab import db
from minilab.api import completions
from minilab.api.auth import authenticate
from minilab.api.errors import install_error_handlers, model_not_found
from minilab.api.metering import HTTP_REQUESTS, IN_FLIGHT, ModelCatalog
from minilab.api.ratelimit import RateLimiter
from minilab.api.upstream import InferenceClient
from minilab.obs.metrics import CONTENT_TYPE, render
from minilab.registry import ModelInfo
from minilab.settings import Settings, get_settings

RATE_LIMIT_HEADERS = [f"x-ratelimit-{k}-{w}" for k in ("limit", "remaining", "reset") for w in ("requests", "tokens")]


class RequestContextMiddleware:
    """Adds x-request-id (and x-ratelimit-* once the caller is known) to every response.

    Handlers put values in request.state (backed by the ASGI scope); they are read
    when the response starts, so error responses get them too. A pure ASGI
    middleware rather than @app.middleware("http"), which buffers streams badly.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["request_id"] = f"req_{secrets.token_hex(12)}"

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = state["request_id"]
                for name, value in state.get("ratelimit_headers", {}).items():
                    headers[name] = value
                route = scope.get("route")  # the matched route template, not the raw path (bounded labels)
                HTTP_REQUESTS.inc(route=getattr(route, "path", "unmatched"), status=str(message["status"]))
            await send(message)

        IN_FLIGHT.inc()
        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            IN_FLIGHT.dec()


router = APIRouter()


def _model_json(m: ModelInfo) -> dict:
    """An OpenAI model object, plus mini-lab extras (family, description, context window, pricing)."""
    return {"id": m.id, "object": "model", "created": m.created, "owned_by": "mini-lab", "family": m.family,
            "description": m.description, "context_length": m.context_length, "pricing": asdict(m.pricing)}


@router.get("/v1/models")
async def list_models(request: Request):
    s = request.app.state
    caller = await authenticate(request, s.settings)
    request.state.ratelimit_headers = s.limiter.headers(caller)
    return {"object": "list", "data": [_model_json(m) for m in await s.catalog.served(s.inference)]}


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    s = request.app.state
    caller = await authenticate(request, s.settings)
    request.state.ratelimit_headers = s.limiter.headers(caller)
    for m in await s.catalog.served(s.inference):
        if m.id == model_id:
            return _model_json(m)
    raise model_not_found(model_id)


def _db_ok() -> bool:
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1 FROM orgs LIMIT 1")
        return True
    except Exception:
        return False


@router.get("/health")
async def health(request: Request):
    """Always 200 while the gateway itself runs; says whether its dependencies are reachable."""
    inference_ok = await request.app.state.inference.health()
    db_ok = await run_in_threadpool(_db_ok)
    return {"status": "ok" if inference_ok and db_ok else "degraded",
            "inference": "ok" if inference_ok else "unreachable", "db": "ok" if db_ok else "error"}


@router.get("/metrics")
async def metrics():
    """Prometheus metrics. Unauthenticated: keep it off the public internet in a real deployment."""
    return PlainTextResponse(render(), media_type=CONTENT_TYPE)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db.configure(settings.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db()
        app.state.inference = InferenceClient(settings.inference_url, settings.internal_token)
        yield
        await app.state.inference.aclose()

    app = FastAPI(title="mini-lab API", version="1.0.0", lifespan=lifespan,
                  description="OpenAI-compatible API for mini-lab models. See docs/api.md.")
    app.state.settings = settings
    app.state.limiter = RateLimiter()
    app.state.catalog = ModelCatalog(settings.models_dir)
    install_error_handlers(app)
    app.include_router(completions.router)
    app.include_router(router)
    app.add_middleware(RequestContextMiddleware)
    # Let browser apps call the API directly. Keys travel in the Authorization header, not in
    # cookies, so allowing any origin doesn't let other sites act with a visitor's credentials.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["x-request-id", "retry-after", *RATE_LIMIT_HEADERS])
    return app
