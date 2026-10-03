"""The mini-lab platform: developer dashboard, billing, playground, docs and the chat app.

Server-rendered pages (Jinja2 + Tailwind from a CDN), a little vanilla JS, and the
shared SQLite store in `minilab.db`. Run it with:

    uv run python -m minilab.platform --port 3000
"""

from __future__ import annotations

from urllib.parse import quote, urlsplit

import httpx
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException

from minilab import db, registry
from minilab.chat.router import router as chat_router
from minilab.chat.tools import CALCULATOR_TOOL
from minilab.platform import billing
from minilab.platform.gateway import Gateway
from minilab.platform.markdown import render_markdown
from minilab.platform.web import (
    CHAT_DIR, ORG_COOKIE, PLATFORM_DIR, SESSION_COOKIE, ChatRequest, Ctx, LoginRequired, bar_chart,
    current_user, daily_series, event_stream, get_ctx, model_choices, optional_ctx, redirect,
    released_models, render, safe_next, usd,
)
from minilab.settings import get_settings

router = APIRouter()


# ---- security ---------------------------------------------------------------

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT = {"/billing/webhook"}  # called by Stripe's servers; authenticated by its signature instead


class CSRFMiddleware:
    """Reject state-changing requests that come from another site.

    Browsers always send an Origin header on POST (Referer as a fallback), and a
    malicious page cannot forge it. Together with SameSite=Lax session cookies this
    is enough CSRF protection without hidden tokens in every form.

    A pure ASGI middleware (not @app.middleware("http")) so streamed answers pass
    through untouched.
    """

    def __init__(self, app, platform_url: str):
        self.app = app
        self.platform_host = urlsplit(platform_url).netloc

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in SAFE_METHODS and scope["path"] not in CSRF_EXEMPT:
            headers = Headers(scope=scope)
            source = headers.get("origin") or headers.get("referer") or ""
            if urlsplit(source).netloc not in {headers.get("host"), self.platform_host}:
                response = PlainTextResponse("Cross-site request blocked: the Origin header does not match this site.", 403)
                return await response(scope, receive, send)
        await self.app(scope, receive, send)


def _secure(request: Request) -> bool:
    # Behind a TLS proxy the request itself may look like http: trust the configured URL too.
    return request.url.scheme == "https" or request.app.state.settings.platform_url.startswith("https://")


def _set_session(response, request: Request, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=db.SESSION_TTL_S, httponly=True, samesite="lax",
                        secure=_secure(request))


# ---- landing & auth -----------------------------------------------------------

@router.get("/")
def landing(request: Request):
    if current_user(request):
        return RedirectResponse("/overview", status_code=303)
    return render(request, "landing.html", models=released_models(request))


@router.get("/signup")
def signup_page(request: Request, next: str | None = None):
    return render(request, "auth.html", mode="signup", next=safe_next(next), form={})


@router.post("/signup")
def signup(request: Request, name: str = Form(""), email: str = Form(...), password: str = Form(...),
           next: str = Form("/overview")):
    form = {"name": name, "email": email}
    error = None
    if "@" not in email or len(email) > 254:
        error = "Enter a valid email address."
    elif len(password) < 8:
        error = "Use at least 8 characters for your password."
    if error is None:
        try:
            user = db.create_user(email, password, name)
        except ValueError as e:
            error = str(e)
    if error:
        return render(request, "auth.html", mode="signup", next=safe_next(next), form=form, error=error,
                      status_code=400)
    # Every account starts with its own org (and the free signup credits that come with it).
    first_name = (name.strip() or email.split("@")[0]).split()[0]
    db.create_org(f"{first_name}'s lab", user["id"])
    response = redirect(safe_next(next), "Welcome to mini-lab! Your organization has free credits to start.")
    _set_session(response, request, db.create_session(user["id"]))
    return response


@router.get("/login")
def login_page(request: Request, next: str | None = None):
    if current_user(request):
        return RedirectResponse(safe_next(next), status_code=303)
    return render(request, "auth.html", mode="login", next=safe_next(next), form={})


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), next: str = Form("/overview")):
    user = db.authenticate(email, password)
    if user is None:
        return render(request, "auth.html", mode="login", next=safe_next(next), form={"email": email},
                      error="Incorrect email or password.", status_code=400)
    response = redirect(safe_next(next))
    _set_session(response, request, db.create_session(user["id"]))
    return response


@router.post("/logout")
def logout(request: Request):
    if token := request.cookies.get(SESSION_COOKIE):
        db.delete_session(token)
    response = redirect("/")
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie(ORG_COOKIE)
    return response


# ---- organizations & projects ----------------------------------------------------

@router.post("/orgs/switch")
def switch_org(request: Request, org_id: str = Form(...), ctx: Ctx = Depends(get_ctx)):
    if org_id == "__new":  # the "New organization..." entry of the switcher, without JS
        return redirect("/orgs/new")
    if not db.is_member(org_id, ctx.user["id"]):
        raise HTTPException(404, "Organization not found.")
    back = urlsplit(request.headers.get("referer", "")).path
    # Detail pages (/logs/<id>) belong to the previous org, so go back to the list instead.
    back = "/logs" if back.startswith("/logs/") else back
    response = redirect(safe_next(back))
    response.set_cookie(ORG_COOKIE, org_id, max_age=365 * 86400, httponly=True, samesite="lax", secure=_secure(request))
    return response


@router.get("/orgs/new")
def new_org_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    return render(request, "org_new.html", ctx)


@router.post("/orgs")
def create_org(request: Request, name: str = Form(...), ctx: Ctx = Depends(get_ctx)):
    # Only the first org gets free credits, otherwise they could be farmed.
    org = db.create_org(name[:80], ctx.user["id"], signup_credit_usd=0)
    response = redirect("/overview", f"Created {org['name']}. Add credits in Billing to start using it.")
    response.set_cookie(ORG_COOKIE, org["id"], max_age=365 * 86400, httponly=True, samesite="lax", secure=_secure(request))
    return response


@router.get("/projects")
def projects_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    keys = db.list_api_keys(ctx.org["id"])
    active = {}
    for k in keys:
        if not k["revoked_at"]:
            active[k["project_id"]] = active.get(k["project_id"], 0) + 1
    return render(request, "projects.html", ctx, projects=db.list_projects(ctx.org["id"]), active_keys=active)


@router.post("/projects")
def create_project(request: Request, name: str = Form(...), ctx: Ctx = Depends(get_ctx)):
    project = db.create_project(ctx.org["id"], name[:80])
    return redirect("/projects", f"Created project {project['name']}.")


# ---- overview ---------------------------------------------------------------

@router.get("/overview")
def overview(request: Request, ctx: Ctx = Depends(get_ctx)):
    series = daily_series(db.usage_by_day(ctx.org["id"], days=7), days=7)
    models = released_models(request)
    return render(
        request, "overview.html", ctx,
        chart=bar_chart(series, "cost_micros", usd),
        week={k: sum(p[k] for p in series) for k in ("requests", "tokens", "cost_micros")},
        has_keys=any(not k["revoked_at"] for k in db.list_api_keys(ctx.org["id"])),
        model_id=models[0].id if models else "prelude-1",
        recent=db.list_requests(ctx.org["id"], limit=5),
    )


# ---- API keys ---------------------------------------------------------------

def _keys_page(request: Request, ctx: Ctx, status_code: int = 200, **extra):
    response = render(request, "api_keys.html", ctx, status_code=status_code,
                      keys=db.list_api_keys(ctx.org["id"]), projects=db.list_projects(ctx.org["id"]), **extra)
    response.headers["Cache-Control"] = "no-store"  # the page may contain a secret
    return response


@router.get("/api-keys")
def api_keys_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    return _keys_page(request, ctx)


@router.post("/api-keys")
def create_api_key(request: Request, name: str = Form(""), project_id: str = Form(...),
                   spend_limit_usd: str = Form(""), ctx: Ctx = Depends(get_ctx)):
    project = db.get_project(project_id)
    if project is None or project["org_id"] != ctx.org["id"]:
        return _keys_page(request, ctx, 400, error="Choose a project from this organization.")
    limit = None
    if spend_limit_usd.strip():
        try:
            limit = float(spend_limit_usd)
        except ValueError:
            limit = -1
        if not 0 < limit < 1_000_000:  # also rejects nan and inf
            return _keys_page(request, ctx, 400, error="The spend limit must be a positive amount in USD, like 5 or 0.50.")
    key, secret = db.create_api_key(ctx.org["id"], project_id, name[:80], created_by=ctx.user["id"],
                                    spend_limit_usd=limit)
    # The secret is only in this one response: we store its hash, never the secret itself.
    return _keys_page(request, ctx, new_key=key, secret=secret)


@router.post("/api-keys/{key_id}/revoke")
def revoke_api_key(key_id: str, ctx: Ctx = Depends(get_ctx)):
    # Scoped to the current org: a key id from another org simply matches nothing.
    if db.revoke_api_key(ctx.org["id"], key_id):
        return redirect("/api-keys", "Key revoked. Requests using it are now rejected.")
    return redirect("/api-keys", "That key does not exist or was already revoked.")


# ---- usage & logs -------------------------------------------------------------

USAGE_METRICS = {
    "cost_micros": ("Spend", usd),
    "tokens": ("Tokens", lambda n: f"{n:,.0f}"),
    "requests": ("Requests", lambda n: f"{n:,.0f}"),
}


@router.get("/usage")
def usage_page(request: Request, days: int = 30, metric: str = "cost_micros", ctx: Ctx = Depends(get_ctx)):
    days = days if days in (7, 30, 90) else 30
    metric = metric if metric in USAGE_METRICS else "cost_micros"
    rows = db.usage_by_day(ctx.org["id"], days=days)
    series = daily_series(rows, days)
    by_model: dict[str, dict] = {}
    for r in rows:
        m = by_model.setdefault(r["model"], {"model": r["model"], "requests": 0, "prompt_tokens": 0,
                                             "completion_tokens": 0, "cost_micros": 0})
        for k in ("requests", "prompt_tokens", "completion_tokens", "cost_micros"):
            m[k] += r[k] or 0
    label, fmt = USAGE_METRICS[metric]
    return render(
        request, "usage.html", ctx, days=days, metric=metric, metric_label=label,
        chart=bar_chart(series, metric, fmt),
        totals={k: sum(p[k] for p in series) for k in ("requests", "tokens", "cost_micros")},
        by_model=sorted(by_model.values(), key=lambda m: -m["cost_micros"]),
        rows=list(reversed(rows)),
        perf=db.performance_stats(ctx.org["id"], since_s=days * 86400),
    )


LOGS_PER_PAGE = 50


@router.get("/logs")
def logs_page(request: Request, page: int = 1, ctx: Ctx = Depends(get_ctx)):
    page = max(page, 1)
    rows = db.list_requests(ctx.org["id"], limit=LOGS_PER_PAGE + 1, offset=(page - 1) * LOGS_PER_PAGE)
    return render(request, "logs.html", ctx, rows=rows[:LOGS_PER_PAGE], page=page,
                  has_next=len(rows) > LOGS_PER_PAGE)


@router.get("/logs/{request_id}")
def log_detail(request: Request, request_id: str, ctx: Ctx = Depends(get_ctx)):
    row = db.get_request(ctx.org["id"], request_id)  # scoped to the org: other orgs' logs 404
    if row is None:
        raise HTTPException(404, "No request with this id in this organization.")
    key = next((k for k in db.list_api_keys(ctx.org["id"]) if k["id"] == row["api_key_id"]), None)
    project = db.get_project(row["project_id"]) if row["project_id"] else None
    return render(request, "log_detail.html", ctx, r=row, key=key, project=project)


# ---- models, docs, status ------------------------------------------------------

@router.get("/models")
def models_page(request: Request, ctx: Ctx | None = Depends(optional_ctx)):
    return render(request, "models.html", ctx, models=released_models(request))


@router.get("/models/{model_id}")
def model_page(request: Request, model_id: str, ctx: Ctx | None = Depends(optional_ctx)):
    info = registry.get_model(request.app.state.settings.models_dir, model_id)
    if info is None:
        raise HTTPException(404, f"There is no released model called {model_id}.")
    card = info.path / "MODEL_CARD.md"
    evals = info.path / "eval.json"
    return render(request, "model_detail.html", ctx, m=info,
                  card=render_markdown(card.read_text(), skip_title=True) if card.exists() else None,
                  evals=evals.read_text() if evals.exists() else None)


@router.get("/docs")
def docs_page(request: Request, ctx: Ctx | None = Depends(optional_ctx)):
    models = released_models(request)
    return render(request, "docs.html", ctx, models=models, model_id=models[0].id if models else "prelude-1")


@router.get("/status")
async def status_page(request: Request, ctx: Ctx | None = Depends(optional_ctx)):
    gateway = await request.app.state.gateway.health()
    try:
        db.get_org("org_status_probe")  # any query proves the database answers
        database = {"ok": True, "detail": "ok"}
    except Exception as e:  # noqa: BLE001 - the status page must render whatever breaks
        database = {"ok": False, "detail": type(e).__name__}
    return render(request, "status.html", ctx, gateway=gateway, database=database,
                  perf=db.performance_stats(None, since_s=3600))


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=301)


# ---- playground ---------------------------------------------------------------

@router.get("/playground")
async def playground_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    return render(request, "playground.html", ctx, models=await model_choices(request, ctx.org["id"]),
                  calculator_tool=CALCULATOR_TOOL)


@router.post("/playground/api/chat")
async def playground_chat(request: Request, body: ChatRequest, ctx: Ctx = Depends(get_ctx)):
    events = request.app.state.gateway.stream_chat(
        org_id=ctx.org["id"], source="playground", model=body.model,
        messages=[m.model_dump() for m in body.messages], temperature=body.temperature,
        max_tokens=body.max_tokens, calculator=body.calculator,
    )
    return event_stream(request, events, body.model)


# ---- app factory --------------------------------------------------------------

def create_app(*, http_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Build the platform app. `http_transport` lets tests swap the real gateway for a stub."""
    settings = get_settings()
    db.init_db()
    # No auto-generated /docs: that URL is our API reference.
    app = FastAPI(title="mini-lab platform", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.gateway = Gateway(settings.api_url, settings.internal_token, http_transport)

    app.add_middleware(CSRFMiddleware, platform_url=settings.platform_url)
    app.mount("/static", StaticFiles(directory=PLATFORM_DIR / "static"), name="static")
    app.mount("/chat/static", StaticFiles(directory=CHAT_DIR / "static"), name="chat-static")
    app.include_router(router)
    app.include_router(billing.router)
    app.include_router(chat_router, prefix="/chat")

    def wants_json(request: Request) -> bool:
        return "/api/" in request.url.path or request.url.path == "/billing/webhook"

    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, exc: LoginRequired):
        if wants_json(request):
            return JSONResponse({"error": {"message": "Log in to continue.", "code": "login_required"}}, 401)
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if wants_json(request):
            return JSONResponse({"error": {"message": exc.detail}}, exc.status_code)
        message = "There is nothing at this address." if exc.detail == "Not Found" else exc.detail
        return render(request, "error.html", optional_ctx(request), status_code=exc.status_code,
                      code=exc.status_code, message=message)

    return app
