"""Metering: price each request, log it, debit credits, count it in the metrics.

Cost = Pricing.cost_micros(prompt_tokens, completion_tokens), with the pricing
from the model's release.json. db.record_request writes the request log line and
debits the org balance and the key's spend in one transaction, so the logs, the
ledger and the balance always agree.

What gets billed:
- 200: the usage reported by the inference server.
- 499 (the client closed a stream mid-way) and 502 (the stream broke mid-way):
  we bill what the client received, as far as the gateway knows it: the
  estimated prompt plus one completion token per streamed delta. Nothing if no
  token was streamed. (Otherwise, breaking a stream on purpose would be free.)
- Anything else (bad request, quota, upstream failure, ...): logged, cost 0.
  Requests rejected by the rate limiter only count in the metrics: writing
  them to the database would let a caller fill it at no cost.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from starlette.concurrency import run_in_threadpool

from minilab import db, registry
from minilab.api.auth import Caller
from minilab.api.errors import model_not_found
from minilab.api.upstream import InferenceClient
from minilab.obs.metrics import Counter, Gauge, Histogram
from minilab.registry import ModelInfo, Pricing

log = logging.getLogger("minilab.api")

BILLED_STATUSES = (200, 499, 502)

# CPU inference can take a while: buckets up to two minutes.
_SECONDS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
HTTP_REQUESTS = Counter("minilab_api_http_requests_total", "HTTP responses by route and status", ["route", "status"])
IN_FLIGHT = Gauge("minilab_api_in_flight_requests", "HTTP requests being processed")
REQUESTS = Counter("minilab_api_requests_total", "Chat completions by model and status", ["model", "status"])
TOKENS = Counter("minilab_api_tokens_total", "Tokens billed", ["model", "kind"])
COST = Counter("minilab_api_cost_micros_total", "Amount billed, in micro-USD", ["model"])
LATENCY = Histogram("minilab_api_request_duration_seconds", "Chat completion latency", ["model"], buckets=_SECONDS)
TTFT = Histogram("minilab_api_ttft_seconds", "Time to first token (streaming)", ["model"], buckets=_SECONDS)


class ModelCatalog:
    """Model metadata (pricing, context length).

    The release.json in MODELS_DIR is what we bill with, so it comes first; it is
    cached because a release never changes. A model that inference serves but
    that isn't in our MODELS_DIR (inference running elsewhere) falls back to the
    metadata inference reports.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._releases: dict[str, ModelInfo] = {}

    def release(self, model_id: str) -> ModelInfo | None:
        if model_id not in self._releases:
            info = registry.get_model(self.models_dir, model_id)
            if info is None:
                return None
            self._releases[model_id] = info
        return self._releases[model_id]

    async def served(self, inference: InferenceClient) -> list[ModelInfo]:
        """The models the inference server has loaded right now."""
        return [self.release(d["id"]) or ModelInfo.from_json(d) for d in await inference.list_models()]

    async def resolve(self, model_id: str, inference: InferenceClient) -> ModelInfo:
        info = self.release(model_id)
        if info is None:
            info = next((m for m in await self.served(inference) if m.id == model_id), None)
        if info is None:
            raise model_not_found(model_id)
        return info


@dataclass
class Meter:
    """Collects what we need to bill and log one chat completion request."""
    id: str  # chatcmpl-..., also the request id in the logs and in x-request-id
    caller: Caller
    model: str = "unknown"  # as requested
    request_body: Any = None
    pricing: Pricing | None = None  # known once the model is resolved
    created: int = field(default_factory=lambda: int(time.time()))
    started: float = field(default_factory=time.perf_counter)
    ttft_s: float | None = None

    def first_token(self) -> None:
        if self.ttft_s is None:
            self.ttft_s = time.perf_counter() - self.started

    async def record(self, status: int, prompt_tokens: int = 0, completion_tokens: int = 0,
                     error: str | None = None, response_body: Any = None, persist: bool = True) -> int:
        """Log the request and debit its cost (0 unless billable). Returns the cost in micros."""
        latency_s = time.perf_counter() - self.started
        billable = self.pricing is not None and status in BILLED_STATUSES
        cost = self.pricing.cost_micros(prompt_tokens, completion_tokens) if billable else 0

        # Unresolved model names come from users: keep them out of metric labels (unbounded cardinality).
        label = self.model if self.pricing is not None else "unknown"
        REQUESTS.inc(model=label, status=str(status))
        TOKENS.inc(prompt_tokens, model=label, kind="prompt")
        TOKENS.inc(completion_tokens, model=label, kind="completion")
        COST.inc(cost, model=label)
        LATENCY.observe(latency_s, model=label)
        if self.ttft_s is not None:
            TTFT.observe(self.ttft_s, model=label)

        if not persist:
            return cost
        c = self.caller
        try:
            await run_in_threadpool(partial(
                db.record_request, id=self.id, org_id=c.org_id, project_id=c.project_id, api_key_id=c.api_key_id,
                source=c.source, model=self.model, status_code=status, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, cost_micros=cost, latency_ms=round(latency_s * 1000),
                ttft_ms=round(self.ttft_s * 1000) if self.ttft_s is not None else None, error=error,
                request_body=self.request_body, response_body=response_body,
            ))
        except Exception:
            # The answer is already computed (or streamed): don't turn a logging failure into an error.
            log.exception("failed to record request %s", self.id)
        return cost
