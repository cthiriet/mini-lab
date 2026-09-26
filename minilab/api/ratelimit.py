"""Per-key rate limits: requests per minute (RPM) and tokens per minute (TPM), in memory.

Each limit is a token bucket holding up to `limit` units and refilling
continuously at limit/60 per second, so a key can burst up to its full limit and
then gets a steady trickle, with no cliff at minute boundaries. This is the model
OpenAI describes for its own limits.

Tokens are only known after generation, so a request reserves an estimate up
front (prompt estimate + max_tokens) and `settle` gives back the difference once
the real usage is known.

State lives in this process: fine for one gateway; several replicas would need a
shared store (e.g. Redis) instead.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from minilab.api.auth import Caller
from minilab.api.errors import APIError


@dataclass
class Bucket:
    capacity: int
    level: float
    updated: float

    def refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.updated) * self.capacity / 60)
        self.updated = now

    def seconds_until(self, amount: float) -> float:
        """How long until the bucket holds `amount` units."""
        return max(0.0, (amount - self.level) * 60 / self.capacity)


def format_duration(seconds: float) -> str:
    """OpenAI's reset format: '20ms', '1.5s', '6m0s'."""
    if seconds < 1:
        return f"{math.ceil(seconds * 1000)}ms"
    if seconds < 60:
        return f"{round(seconds, 3):g}s"
    return f"{int(seconds // 60)}m{round(seconds % 60, 3):g}s"


def prompt_chars(messages: list[dict]) -> int:
    def size(value) -> int:
        return len(value) if isinstance(value, str) else len(json.dumps(value)) if value else 0
    return sum(size(m.get("content")) + size(m.get("tool_calls")) for m in messages)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough prompt size without running the tokenizer: ~4 characters per token (the usual rule
    of thumb) plus a few template tokens per message. Only used to reserve rate-limit budget
    (settled against real usage afterwards) and to bill streams aborted before usage was known."""
    return prompt_chars(messages) // 4 + 4 * len(messages)


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._buckets: dict[str, tuple[Bucket, Bucket]] = {}  # limit key -> (requests, tokens)

    def _get(self, caller: Caller) -> tuple[Bucket, Bucket]:
        now = self.clock()
        buckets = self._buckets.get(caller.limit_key)
        if buckets is None:
            buckets = (Bucket(caller.rpm_limit, caller.rpm_limit, now), Bucket(caller.tpm_limit, caller.tpm_limit, now))
            self._buckets[caller.limit_key] = buckets
        for bucket, limit in zip(buckets, (caller.rpm_limit, caller.tpm_limit)):
            bucket.refill(now)
            if bucket.capacity != limit:  # the limit was edited in the dashboard
                bucket.capacity, bucket.level = limit, min(bucket.level, limit)
        return buckets

    def acquire(self, caller: Caller, tokens: int, requests: int = 1) -> None:
        """Take `requests` requests and `tokens` tokens, or raise 429 rate_limit_exceeded.

        Check-then-take is atomic because it never awaits (one event loop, one thread)."""
        requests_bucket, budget = self._get(caller)
        for what, bucket, amount in (("requests", requests_bucket, requests), ("tokens", budget, tokens)):
            if amount > bucket.capacity:  # can never fit: retrying won't help
                raise APIError(429, f"Request too large: it may use up to {amount} {what} but this key is limited "
                                    f"to {bucket.capacity} {what} per minute. Reduce max_tokens or the prompt.",
                               code="rate_limit_exceeded", type="requests", headers={"x-should-retry": "false"})
            if bucket.level < amount:
                wait = bucket.seconds_until(amount)
                raise APIError(429, f"Rate limit reached for {what} per minute: limit {bucket.capacity}, "
                                    f"remaining {int(bucket.level)}, requested {amount}. "
                                    f"Please try again in {format_duration(wait)}.",
                               code="rate_limit_exceeded", type=what,
                               headers={"retry-after": str(math.ceil(wait)), "retry-after-ms": str(math.ceil(wait * 1000))})
        requests_bucket.level -= requests
        budget.level -= tokens

    def settle(self, caller: Caller, reserved: int, used: int) -> None:
        """Give back what a request reserved but didn't use (or take more if it used more)."""
        _, budget = self._get(caller)
        budget.level = min(budget.capacity, budget.level + reserved - used)

    def headers(self, caller: Caller) -> dict[str, str]:
        """OpenAI's x-ratelimit-* headers; reset = time until the bucket is full again."""
        requests, budget = self._get(caller)
        out = {}
        for what, bucket in (("requests", requests), ("tokens", budget)):
            out[f"x-ratelimit-limit-{what}"] = str(bucket.capacity)
            out[f"x-ratelimit-remaining-{what}"] = str(max(0, int(bucket.level)))
            out[f"x-ratelimit-reset-{what}"] = format_duration(bucket.seconds_until(bucket.capacity))
        return out
