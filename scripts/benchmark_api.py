import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "benchmark" / "api.json"


@dataclass(frozen=True)
class RequestSample:
    latency_ms: float
    status_code: int | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.status_code is not None and self.status_code < 400


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise ValueError("fraction 必须位于 0 到 1 之间。")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize_samples(
    samples: list[RequestSample],
    elapsed_seconds: float,
) -> dict[str, Any]:
    latencies = [sample.latency_ms for sample in samples]
    success_count = sum(sample.succeeded for sample in samples)
    total_count = len(samples)
    status_counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.status_code) if sample.status_code is not None else "transport_error"
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "request_count": total_count,
        "success_count": success_count,
        "failure_count": total_count - success_count,
        "error_rate": ((total_count - success_count) / total_count if total_count else 0.0),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "requests_per_second": round(total_count / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "status_counts": status_counts,
    }


async def send_chat_request(
    client: httpx.AsyncClient,
    endpoint: str,
    question: str,
    request_number: int,
) -> RequestSample:
    started_at = perf_counter()
    try:
        response = await client.post(
            endpoint,
            json={
                "question": question,
                "session_id": f"benchmark-{request_number}",
            },
        )
        error = None if response.status_code < 400 else response.text[:500]
        return RequestSample(
            latency_ms=(perf_counter() - started_at) * 1000,
            status_code=response.status_code,
            error=error,
        )
    except httpx.HTTPError as error:
        return RequestSample(
            latency_ms=(perf_counter() - started_at) * 1000,
            status_code=None,
            error=f"{type(error).__name__}: {error}",
        )


async def run_benchmark(args: argparse.Namespace) -> tuple[list[RequestSample], float]:
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        limits=limits,
        timeout=timeout,
    ) as client:
        ready_response = await client.get(args.ready_endpoint)
        ready_response.raise_for_status()

        for warmup_index in range(args.warmup):
            sample = await send_chat_request(
                client,
                args.endpoint,
                args.question,
                -(warmup_index + 1),
            )
            if not sample.succeeded:
                raise RuntimeError(
                    f"预热请求失败：status={sample.status_code}, error={sample.error}"
                )

        semaphore = asyncio.Semaphore(args.concurrency)

        async def limited_request(index: int) -> RequestSample:
            async with semaphore:
                return await send_chat_request(
                    client,
                    args.endpoint,
                    args.question,
                    index,
                )

        started_at = perf_counter()
        samples = await asyncio.gather(
            *(limited_request(index) for index in range(args.requests))
        )
        elapsed_seconds = perf_counter() - started_at
    return list(samples), elapsed_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行客服 Agent HTTP 并发性能基线。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/v1/chat")
    parser.add_argument("--ready-endpoint", default="/v1/ready")
    parser.add_argument("--token", default="")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--question",
        default="DCB107 充电器的指示灯闪烁是什么意思？",
    )
    parser.add_argument("--max-p95-ms", type=float, default=5000.0)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests 必须大于等于 1。")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于等于 1。")
    if args.concurrency > args.requests:
        parser.error("--concurrency 不能大于 --requests。")
    if args.warmup < 0:
        parser.error("--warmup 不能小于 0。")
    if args.timeout <= 0 or args.max_p95_ms <= 0:
        parser.error("超时和 P95 阈值必须大于 0。")
    if not 0 <= args.max_error_rate <= 1:
        parser.error("--max-error-rate 必须位于 0 到 1 之间。")
    return args


def main() -> None:
    args = parse_args()
    samples, elapsed_seconds = asyncio.run(run_benchmark(args))
    summary = summarize_samples(samples, elapsed_seconds)
    passed = (
        summary["error_rate"] <= args.max_error_rate
        and summary["latency_ms"]["p95"] <= args.max_p95_ms
    )
    report = {
        "benchmark_type": "http_chat",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
        },
        "thresholds": {
            "max_p95_ms": args.max_p95_ms,
            "max_error_rate": args.max_error_rate,
        },
        "summary": {**summary, "passed": passed},
        "failures": [
            asdict(sample) for sample in samples if not sample.succeeded
        ][:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("====HTTP 性能基线====")
    print("请求：", f"{summary['success_count']}/{summary['request_count']} 成功")
    print("并发：", args.concurrency)
    print("吞吐量：", f"{summary['requests_per_second']:.3f} req/s")
    print("错误率：", f"{summary['error_rate']:.2%}")
    print("P50：", f"{summary['latency_ms']['p50']:.3f} ms")
    print("P95：", f"{summary['latency_ms']['p95']:.3f} ms")
    print("P99：", f"{summary['latency_ms']['p99']:.3f} ms")
    print("阈值检查：", "通过" if passed else "失败")
    print("报告路径：", args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
