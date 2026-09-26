"""Who is calling, and who pays.

Two ways in:

- Customers: `Authorization: Bearer sk-mini-...`. Only a hash of the key is
  stored, so we look it up by hash (db.lookup_api_key); revoked keys don't match.
- First-party services (the platform's playground and chat app): the shared
  internal token plus `X-Minilab-Org` (required), `X-Minilab-Project` and
  `X-Minilab-Source` (optional). They are billed to that org like any traffic,
  with no API key (api_key_id NULL in the logs) and per-org rate limits.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from minilab import db
from minilab.api.errors import APIError, invalid_request
from minilab.settings import Settings

# First-party callers are trusted to label their traffic (playground, chat, ...); just keep it tidy.
SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


@dataclass
class Caller:
    org_id: str
    project_id: str | None
    api_key_id: str | None  # None for first-party traffic
    source: str  # api | playground | chat
    balance_micros: int  # org credit balance at the time of the request
    rpm_limit: int
    tpm_limit: int
    spend_micros: int = 0  # what this key has spent so far
    spend_limit_micros: int | None = None

    @property
    def limit_key(self) -> str:
        """Rate limits apply per API key; first-party traffic shares one budget per org."""
        return self.api_key_id or f"org:{self.org_id}"


def _mask(secret: str) -> str:
    # Enough to recognise the key in the dashboard (which shows the same hint), never the full secret.
    return f"{secret[:12]}...{secret[-4:]}" if len(secret) > 20 else secret[:4] + "..."


async def authenticate(request: Request, settings: Settings) -> Caller:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise APIError(401, "You didn't provide an API key. You need to provide your API key in an "
                            "Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                       code="invalid_api_key")

    # Constant-time comparison so the token can't be guessed byte by byte from response timings.
    if hmac.compare_digest(token.encode(), settings.internal_token.encode()):
        return await _first_party(request, settings)

    # sqlite calls block, so run them in a thread instead of stalling every stream in flight.
    key = await run_in_threadpool(db.lookup_api_key, token)
    if key is None:
        raise APIError(401, f"Incorrect API key provided: {_mask(token)}. You can find your API keys "
                            f"at {settings.platform_url}.", code="invalid_api_key")
    return Caller(
        org_id=key["org_id"], project_id=key["project_id"], api_key_id=key["id"], source="api",
        balance_micros=key["balance_micros"],
        rpm_limit=key["rpm_limit"] or settings.default_rpm, tpm_limit=key["tpm_limit"] or settings.default_tpm,
        spend_micros=key["spend_micros"], spend_limit_micros=key["spend_limit_micros"],
    )


async def _first_party(request: Request, settings: Settings) -> Caller:
    org_id = request.headers.get("x-minilab-org")
    if not org_id:
        raise invalid_request("The X-Minilab-Org header is required with the internal token.",
                              param="X-Minilab-Org")
    org = await run_in_threadpool(db.get_org, org_id)
    if org is None:
        raise invalid_request(f"Unknown organization '{org_id}'.", param="X-Minilab-Org")

    project_id = request.headers.get("x-minilab-project") or None
    if project_id:
        project = await run_in_threadpool(db.get_project, project_id)
        if project is None or project["org_id"] != org_id:
            raise invalid_request(f"Unknown project '{project_id}' for this organization.",
                                  param="X-Minilab-Project")

    source = request.headers.get("x-minilab-source") or "playground"
    if not SOURCE_PATTERN.match(source):
        raise invalid_request("X-Minilab-Source must be a short lowercase name such as 'playground' or 'chat'.",
                              param="X-Minilab-Source")
    return Caller(org_id=org_id, project_id=project_id, api_key_id=None, source=source,
                  balance_micros=org["balance_micros"],
                  rpm_limit=settings.default_rpm, tpm_limit=settings.default_tpm)


def check_quota(caller: Caller, platform_url: str) -> None:
    """Prepaid billing: no credits, no tokens. Raises 429 insufficient_quota.

    This is checked before the request, so concurrent requests can take a balance
    slightly below zero; the next request is then refused.
    """
    # The SDKs retry 429s by default; retrying won't bring credits back, so tell them not to.
    headers = {"x-should-retry": "false"}
    if caller.balance_micros <= 0:
        raise APIError(429, f"You exceeded your current quota: your organization has no credits left. "
                            f"Add credits at {platform_url}.",
                       type="insufficient_quota", code="insufficient_quota", headers=headers)
    if caller.spend_limit_micros is not None and caller.spend_micros >= caller.spend_limit_micros:
        raise APIError(429, f"This API key reached its spend limit of {db.format_usd(caller.spend_limit_micros)}. "
                            f"Raise the limit or use another key at {platform_url}.",
                       type="insufficient_quota", code="insufficient_quota", headers=headers)
