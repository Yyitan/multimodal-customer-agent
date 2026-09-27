from scripts.benchmark_api import RequestSample, percentile, summarize_samples


def main() -> None:
    checks: dict[str, bool] = {}

    checks["百分位使用最近排名法"] = (
        percentile([50.0, 10.0, 30.0, 20.0, 40.0], 0.50) == 30.0
        and percentile([50.0, 10.0, 30.0, 20.0, 40.0], 0.95) == 50.0
    )
    summary = summarize_samples(
        [
            RequestSample(latency_ms=10.0, status_code=200),
            RequestSample(latency_ms=20.0, status_code=200),
            RequestSample(latency_ms=30.0, status_code=503, error="failed"),
            RequestSample(latency_ms=40.0, status_code=None, error="timeout"),
        ],
        elapsed_seconds=2.0,
    )
    checks["性能汇总统计成功率和吞吐量"] = (
        summary["request_count"] == 4
        and summary["success_count"] == 2
        and summary["failure_count"] == 2
        and summary["error_rate"] == 0.5
        and summary["requests_per_second"] == 2.0
    )
    checks["性能汇总记录状态码分布"] = summary["status_counts"] == {
        "200": 2,
        "503": 1,
        "transport_error": 1,
    }
    checks["性能汇总计算延迟分位数"] = (
        summary["latency_ms"]["p50"] == 20.0
        and summary["latency_ms"]["p95"] == 40.0
        and summary["latency_ms"]["max"] == 40.0
    )

    print("====性能基线工具测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("\n性能基线工具测试：全部通过")


if __name__ == "__main__":
    main()
