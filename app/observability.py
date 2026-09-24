from collections import defaultdict
from threading import RLock
from time import perf_counter
from typing import Any


LabelTuple = tuple[tuple[str, str], ...]


def _labels(values: dict[str, Any]) -> LabelTuple:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def _format_labels(labels: LabelTuple) -> str:
    if not labels:
        return ""
    values = ",".join(
        f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in labels
    )
    return "{" + values + "}"


def normalize_http_path(path: str) -> str:
    """Collapse dynamic identifiers to bounded metric labels."""
    for prefix in ("", "/v1"):
        if path.startswith(f"{prefix}/sessions/"):
            return f"{prefix}/sessions/{{session_id}}"
        if path.startswith(f"{prefix}/tickets/"):
            return f"{prefix}/tickets/{{ticket_id}}"
        if path.startswith(f"{prefix}/knowledge-images/"):
            return f"{prefix}/knowledge-images/{{image_id}}"
    return path


class MetricsRegistry:
    """Small dependency-free Prometheus metrics registry."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: defaultdict[tuple[str, LabelTuple], float] = defaultdict(float)
        self._gauges: defaultdict[tuple[str, LabelTuple], float] = defaultdict(float)
        self._observations: defaultdict[
            tuple[str, LabelTuple], list[float]
        ] = defaultdict(lambda: [0.0, 0.0])

    def increment(self, name: str, amount: float = 1, **labels: Any) -> None:
        with self._lock:
            self._counters[(name, _labels(labels))] += amount

    def add_gauge(self, name: str, amount: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[(name, _labels(labels))] += amount

    def observe(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            observation = self._observations[(name, _labels(labels))]
            observation[0] += 1
            observation[1] += value

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            observations = {
                key: value.copy()
                for key, value in self._observations.items()
            }

        lines: list[str] = []
        for name in sorted({key[0] for key in counters}):
            lines.append(f"# TYPE {name} counter")
            for (metric_name, labels), value in sorted(counters.items()):
                if metric_name == name:
                    lines.append(
                        f"{name}{_format_labels(labels)} {value:g}"
                    )

        for name in sorted({key[0] for key in gauges}):
            lines.append(f"# TYPE {name} gauge")
            for (metric_name, labels), value in sorted(gauges.items()):
                if metric_name == name:
                    lines.append(
                        f"{name}{_format_labels(labels)} {value:g}"
                    )

        for name in sorted({key[0] for key in observations}):
            lines.append(f"# TYPE {name} summary")
            for (metric_name, labels), values in sorted(observations.items()):
                if metric_name != name:
                    continue
                count, total = values
                label_text = _format_labels(labels)
                lines.append(f"{name}_count{label_text} {count:g}")
                lines.append(f"{name}_sum{label_text} {total:g}")

        return "\n".join(lines) + "\n"

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._observations.clear()


class MetricsMiddleware:
    """Record bounded HTTP request count, status and latency metrics."""

    def __init__(self, app: Any, registry: MetricsRegistry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        status_code = 500
        self.registry.add_gauge("http_requests_active", 1)

        async def send_with_status(message: dict) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            method = str(scope.get("method", "UNKNOWN"))
            path = normalize_http_path(str(scope.get("path", "unknown")))
            duration = perf_counter() - started_at
            self.registry.increment(
                "http_requests_total",
                method=method,
                path=path,
                status=status_code,
            )
            self.registry.observe(
                "http_request_duration_seconds",
                duration,
                method=method,
                path=path,
            )
            self.registry.add_gauge("http_requests_active", -1)
