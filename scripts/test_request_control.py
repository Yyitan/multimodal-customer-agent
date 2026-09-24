from app.request_control import (
    SlidingWindowRateLimiter,
    rate_limit_key,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def main() -> None:
    checks: dict[str, bool] = {}
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=10,
        clock=clock,
    )

    first = limiter.check("client-a")
    second = limiter.check("client-a")
    blocked = limiter.check("client-a")
    checks["窗口内允许配置数量的请求"] = (
        first.allowed
        and second.allowed
        and first.remaining == 1
        and second.remaining == 0
    )
    checks["超过窗口限制时拒绝并返回重试秒数"] = (
        blocked.allowed is False
        and blocked.retry_after_seconds == 10
    )

    independent = limiter.check("client-b")
    checks["不同调用方独立计数"] = independent.allowed is True

    clock.advance(10)
    recovered = limiter.check("client-a")
    checks["窗口过期后恢复访问"] = recovered.allowed is True

    token_key = rate_limit_key(
        authorization="Bearer private-token",
        client_host="127.0.0.1",
    )
    ip_key = rate_limit_key(
        authorization=None,
        client_host="127.0.0.1",
    )
    checks["Token只以哈希形式进入限流键"] = (
        token_key.startswith("token:")
        and "private-token" not in token_key
    )
    checks["未鉴权环境按客户端IP限流"] = (
        ip_key == "ip:127.0.0.1"
    )

    print("====请求控制测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n请求控制测试：全部通过")


if __name__ == "__main__":
    main()
