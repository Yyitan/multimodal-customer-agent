from app.observability import (
    MetricsRegistry,
    normalize_http_path,
)


def main() -> None:
    checks: dict[str, bool] = {}
    registry = MetricsRegistry()

    registry.increment(
        "agent_chat_requests_total",
        intent="troubleshooting",
        handoff="false",
    )
    registry.increment(
        "agent_chat_requests_total",
        intent="troubleshooting",
        handoff="false",
    )
    registry.observe(
        "agent_chat_duration_seconds",
        0.25,
        action="respond_with_knowledge",
    )
    registry.add_gauge("http_requests_active", 1)
    registry.add_gauge("http_requests_active", -1)
    output = registry.render()

    checks["计数器按标签累计"] = (
        'agent_chat_requests_total{handoff="false",intent="troubleshooting"} 2'
        in output
    )
    checks["耗时输出count和sum"] = (
        "agent_chat_duration_seconds_count" in output
        and "agent_chat_duration_seconds_sum" in output
    )
    checks["活跃请求使用Gauge"] = (
        "# TYPE http_requests_active gauge" in output
        and "http_requests_active 0" in output
    )
    checks["动态路径被归一化"] = (
        normalize_http_path("/tickets/ticket_123")
        == "/tickets/{ticket_id}"
        and normalize_http_path("/sessions/demo")
        == "/sessions/{session_id}"
    )

    print("====可观测性测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n可观测性测试：全部通过")


if __name__ == "__main__":
    main()
