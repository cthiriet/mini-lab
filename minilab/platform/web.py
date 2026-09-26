"""Shared web plumbing for the platform and the chat app: templates, login, current
org, flash messages, formatting helpers and the SSE response used by the playground
and the chat app.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from minilab import db, registry

PLATFORM_DIR = Path(__file__).parent
CHAT_DIR = PLATFORM_DIR.parent / "chat"

SESSION_COOKIE = "minilab_session"
ORG_COOKIE = "minilab_org"      # which of the user's orgs is selected (always re-checked against membership)
FLASH_COOKIE = "minilab_flash"  # one-shot message shown after a redirect

templates = Jinja2Templates(directory=[PLATFORM_DIR / "templates", CHAT_DIR / "templates"])


# ---- formatting -------------------------------------------------------------

def usd(micros: int | None, digits: int | None = None) -> str:
    """Money for humans. Our model is cheap, so tiny amounts get more digits: $0.000123."""
    if micros is None:
        return "–"
    if digits is not None:
        return db.format_usd(micros, digits)
    # >= 1 cent: plain cents; below, micro-dollar precision (fixed, so columns line up).
    return db.format_usd(micros, 2 if micros == 0 or abs(micros) >= 10_000 else 6)


def utc(ts: int | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(ts, UTC).strftime(fmt) if ts else "–"


def iso(ts: int | None) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts else ""


def gauge_level(balance_micros: int) -> float:
    """How full the balance flask looks: sqrt so $1 is visible and $25 fills it."""
    return max(0.0, min(1.0, math.sqrt(max(balance_micros, 0) / 25_000_000)))


def pretty_json(text: str | None) -> str:
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text  # bodies are truncated in the log, so they may not parse


templates.env.filters.update(usd=usd, utc=utc, iso=iso, gauge_level=gauge_level, pretty_json=pretty_json,
                             num=lambda n: f"{n or 0:,}")


# ---- current user & org -----------------------------------------------------

class LoginRequired(Exception):
    """Raised by get_ctx; app.py turns it into a redirect to /login (or a 401 for JSON APIs)."""


@dataclass
class Ctx:
    """Who is asking, and on behalf of which org. Every dashboard query is scoped to ctx.org."""
    user: dict
    org: dict
    orgs: list[dict]


def current_user(request: Request) -> dict | None:
    if not hasattr(request.state, "user"):
        request.state.user = db.get_user_by_session(request.cookies.get(SESSION_COOKIE))
    return request.state.user


def optional_ctx(request: Request) -> Ctx | None:
    user = current_user(request)
    if user is None:
        return None
    orgs = db.list_user_orgs(user["id"])
    if not orgs:  # should not happen (signup creates one), but never leave a user org-less
        db.create_org("Personal", user["id"], signup_credit_usd=0)
        orgs = db.list_user_orgs(user["id"])
    # The cookie is only a preference: we pick from orgs the user is a member of.
    wanted = request.cookies.get(ORG_COOKIE)
    org = next((o for o in orgs if o["id"] == wanted), orgs[0])
    return Ctx(user=user, org=org, orgs=orgs)


def get_ctx(request: Request) -> Ctx:
    """FastAPI dependency for pages that need a logged-in user."""
    ctx = optional_ctx(request)
    if ctx is None:
        raise LoginRequired()
    return ctx


def safe_next(url: str | None, default: str = "/overview") -> str:
    """Only follow local redirects (no //evil.com, no https://evil.com)."""
    if url and url.startswith("/") and not url.startswith("//") and "\\" not in url:
        return url
    return default


# ---- responses --------------------------------------------------------------

def render(request: Request, name: str, ctx: Ctx | None = None, status_code: int = 200, **context) -> HTMLResponse:
    flash = request.cookies.get(FLASH_COOKIE)
    response = templates.TemplateResponse(request, name, {
        "ctx": ctx,
        "settings": request.app.state.settings,
        "flash": unquote(flash) if flash else None,
        **context,
    }, status_code=status_code)
    if flash:
        response.delete_cookie(FLASH_COOKIE)
    return response


def redirect(url: str, flash: str | None = None) -> RedirectResponse:
    """303 after a POST (Post/Redirect/Get), with an optional one-shot message."""
    response = RedirectResponse(url, status_code=303)
    if flash:
        response.set_cookie(FLASH_COOKIE, quote(flash), max_age=60, httponly=True, samesite="lax")
    return response


# ---- models -----------------------------------------------------------------

def released_models(request: Request) -> list[registry.ModelInfo]:
    return registry.list_models(request.app.state.settings.models_dir)


async def model_choices(request: Request, org_id: str) -> list[str]:
    """Model ids for the playground / chat pickers: released models on disk, or,
    if there are none (e.g. a stub gateway in development), whatever the gateway serves."""
    ids = [m.id for m in released_models(request)]
    return ids or await request.app.state.gateway.model_ids(org_id)


# ---- charts -----------------------------------------------------------------

def daily_series(rows: list[dict], days: int) -> list[dict]:
    """Turn db.usage_by_day rows (day, model) into one entry per day, with empty days filled in."""
    by_day: dict[str, dict] = {}
    for r in rows:
        d = by_day.setdefault(r["day"], {"requests": 0, "tokens": 0, "cost_micros": 0})
        d["requests"] += r["requests"] or 0
        d["tokens"] += (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        d["cost_micros"] += r["cost_micros"] or 0
    today = datetime.now(UTC).date()
    series = []
    for i in range(days - 1, -1, -1):
        day = today - timedelta(days=i)
        values = by_day.get(day.isoformat(), {"requests": 0, "tokens": 0, "cost_micros": 0})
        series.append({"day": day.isoformat(), "label": f"{day:%b} {day.day}", **values})
    return series


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1
    step = 10 ** math.floor(math.log10(value))
    return next(m * step for m in (1, 2, 5, 10) if value <= m * step)


def bar_chart(series: list[dict], key: str, fmt, width: int = 720, height: int = 160) -> dict:
    """Geometry for a simple inline-SVG bar chart (rendered by the `bar_chart` macro).

    The SVG stretches horizontally to fill its container (preserveAspectRatio="none")
    at a fixed pixel height, and the axis labels are HTML, so text stays readable at
    any width. Bars have 4px rounded tops anchored to the baseline; every day gets a
    full-height, invisible hit area carrying the tooltip text.
    """
    peak = _nice_ceiling(max((p[key] for p in series), default=0))
    slot = width / max(len(series), 1)
    bar_w = min(36.0, slot * 0.72)
    label_every = math.ceil(len(series) / 7)  # about 7 date labels, whatever the period

    bars = []
    for i, p in enumerate(series):
        x = i * slot + (slot - bar_w) / 2
        h = p[key] / peak * (height - 1)  # -1 keeps the rounded top off the upper edge
        r = min(4.0, h, bar_w / 2)
        top = height - h
        path = (f"M{x:.1f},{height} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
                f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{top:.1f} {x + bar_w:.1f},{top + r:.1f} "
                f"V{height} Z") if h > 0 else ""
        labelled = i % label_every == 0
        bars.append({"path": path, "slot_x": round(i * slot, 2), "slot_w": round(slot, 2),
                     "tip": f"{p['label']}\n{fmt(p[key])}",
                     "label": p["label"] if labelled else "",
                     "minor": labelled and (i // label_every) % 2 == 1})  # hidden on phones
    peak_label = fmt(peak)
    if "." in peak_label:  # an axis label reads better as $0.005 than $0.005000
        peak_label = peak_label.rstrip("0").rstrip(".")
    return {"width": width, "height": height, "peak_label": peak_label, "bars": bars}


# ---- chat streaming (playground + chat app) ------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    """What the playground and chat UIs send us (not the OpenAI format: we build that)."""
    model: str = Field(min_length=1, max_length=100)
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    calculator: bool = False


def event_stream(request: Request, events: AsyncIterator[dict], model: str) -> StreamingResponse:
    """Send gateway events to the browser as Server-Sent Events, adding the cost of
    the call (same formula the gateway bills with) to the final `done` event."""
    info = registry.get_model(request.app.state.settings.models_dir, model)

    async def body():
        async for event in events:
            if event["type"] == "done" and info is not None:
                event["cost_micros"] = info.pricing.cost_micros(**event["usage"])
                event["cost"] = usd(event["cost_micros"])
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
