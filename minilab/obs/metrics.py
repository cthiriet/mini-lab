"""Tiny Prometheus-style metrics (counters, gauges, histograms) with no dependencies.

    REQUESTS = Counter("minilab_requests_total", "Requests handled", ["model", "status"])
    REQUESTS.inc(model="prelude-2", status="200")
    ...
    return PlainTextResponse(render(), media_type=CONTENT_TYPE)
"""

from __future__ import annotations

import threading
from bisect import bisect_left

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_REGISTRY: list["_Metric"] = []
_LOCK = threading.Lock()


def _labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{v}"' for n, v in zip(names, values)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class _Metric:
    kind = ""

    def __init__(self, name: str, help: str, labels: list[str] | tuple[str, ...] = ()):
        self.name, self.help, self.label_names = name, help, tuple(labels)
        self._values: dict[tuple[str, ...], object] = {}
        with _LOCK:
            _REGISTRY[:] = [m for m in _REGISTRY if m.name != name]  # re-registration replaces (tests)
            _REGISTRY.append(self)

    def _key(self, labels: dict) -> tuple[str, ...]:
        return tuple(str(labels.get(n, "")) for n in self.label_names)

    def render(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    kind = "counter"

    def inc(self, amount: float = 1.0, **labels) -> None:
        k = self._key(labels)
        with _LOCK:
            self._values[k] = self._values.get(k, 0.0) + amount

    def get(self, **labels) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        return super().render() + [f"{self.name}{_labels(self.label_names, k)} {v}" for k, v in self._values.items()]


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels) -> None:
        with _LOCK:
            self._values[self._key(labels)] = value

    def dec(self, amount: float = 1.0, **labels) -> None:
        self.inc(-amount, **labels)


class Histogram(_Metric):
    kind = "histogram"
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

    def __init__(self, name: str, help: str, labels=(), buckets=DEFAULT_BUCKETS):
        super().__init__(name, help, labels)
        self.buckets = tuple(sorted(buckets))

    def observe(self, value: float, **labels) -> None:
        k = self._key(labels)
        with _LOCK:
            counts, total, n = self._values.get(k) or ([0] * len(self.buckets), 0.0, 0)
            i = bisect_left(self.buckets, value)
            if i < len(counts):
                counts[i] += 1
            self._values[k] = (counts, total + value, n + 1)

    def render(self) -> list[str]:
        lines = super().render()
        for k, (counts, total, n) in self._values.items():
            cum = 0
            for b, c in zip(self.buckets, counts):
                cum += c
                le = 'le="%s"' % b
                lines.append(f"{self.name}_bucket{_labels(self.label_names, k, le)} {cum}")
            le = 'le="+Inf"'
            lines.append(f"{self.name}_bucket{_labels(self.label_names, k, le)} {n}")
            lines.append(f"{self.name}_sum{_labels(self.label_names, k)} {total}")
            lines.append(f"{self.name}_count{_labels(self.label_names, k)} {n}")
        return lines


def render() -> str:
    """All registered metrics in Prometheus text exposition format."""
    with _LOCK:
        metrics = list(_REGISTRY)
    return "\n".join(line for m in metrics for line in m.render()) + "\n"
