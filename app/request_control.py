import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable
from uuid import uuid4

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    remaining: int


class SlidingWindowRateLimiter:
    """线程安全的进程内滑动窗口限流器。"""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("限流次数和窗口秒数必须大于等于 1。")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._requests: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def check(self, key: str) -> RateLimitDecision:
        """记录一次请求并返回是否允许。"""
        now = self._clock()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = max(
                    1,
                    int(self.window_seconds - (now - timestamps[0]) + 0.999),
                )
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    remaining=0,
                )

            timestamps.append(now)
            remaining = self.max_requests - len(timestamps)
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                remaining=remaining,
            )

    def clear(self) -> None:
        """清空状态，主要用于测试和受控重载。"""
        with self._lock:
            self._requests.clear()


def rate_limit_key(
    authorization: str | None,
    client_host: str | None,
) -> str:
    """优先按 Token 哈希分组，未鉴权环境退回客户端 IP。"""
    if authorization:
        digest = hashlib.sha256(
            authorization.encode("utf-8")
        ).hexdigest()
        return f"token:{digest}"
    return f"ip:{client_host or 'unknown'}"


class RequestIdMiddleware:
    """为 HTTP 请求生成或透传安全的请求 ID。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_headers(
            scope.get("headers", [])
        ) or uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-request-id", request_id.encode("ascii"))
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def _request_id_from_headers(
    headers: list[tuple[bytes, bytes]],
) -> str | None:
    for name, value in headers:
        if name.lower() != b"x-request-id":
            continue
        try:
            candidate = value.decode("ascii").strip()
        except UnicodeDecodeError:
            return None
        if REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        return None
    return None
