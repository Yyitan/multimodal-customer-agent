from app.model_resilience import (
    CircuitBreaker,
    CircuitOpenError,
    ResilienceExhaustedError,
    execute_with_resilience,
)


class TransientError(RuntimeError):
    pass


def main() -> None:
    checks: dict[str, bool] = {}
    clock_value = [100.0]
    breaker = CircuitBreaker(clock=lambda: clock_value[0])
    delays: list[float] = []
    attempts = [0]

    def eventually_succeeds() -> str:
        attempts[0] += 1
        if attempts[0] < 3:
            raise TransientError("temporary")
        return "ok"

    result = execute_with_resilience(
        eventually_succeeds,
        circuit_key="provider:model:text",
        max_retries=2,
        retry_base_seconds=0.25,
        retry_max_seconds=1.0,
        circuit_failure_threshold=2,
        circuit_recovery_seconds=10,
        is_retryable=lambda error: isinstance(error, TransientError),
        breaker=breaker,
        sleeper=delays.append,
    )
    checks["瞬时错误经过退避后恢复"] = (
        result.value == "ok"
        and result.attempt_count == 3
        and delays == [0.25, 0.5]
    )

    non_retry_attempts = [0]

    def invalid_request() -> None:
        non_retry_attempts[0] += 1
        raise ValueError("invalid")

    try:
        execute_with_resilience(
            invalid_request,
            circuit_key="provider:model:text",
            max_retries=3,
            retry_base_seconds=0,
            retry_max_seconds=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=10,
            is_retryable=lambda error: isinstance(error, TransientError),
            breaker=breaker,
            sleeper=lambda _: None,
        )
    except ValueError:
        pass
    checks["不可恢复错误不会重试"] = non_retry_attempts[0] == 1

    failing_attempts = [0]

    def always_fails() -> None:
        failing_attempts[0] += 1
        raise TransientError("down")

    for _ in range(2):
        try:
            execute_with_resilience(
                always_fails,
                circuit_key="provider:down:text",
                max_retries=0,
                retry_base_seconds=0,
                retry_max_seconds=0,
                circuit_failure_threshold=2,
                circuit_recovery_seconds=10,
                is_retryable=lambda error: isinstance(error, TransientError),
                breaker=breaker,
                sleeper=lambda _: None,
            )
        except ResilienceExhaustedError as error:
            checks["耗尽重试后保留实际尝试次数"] = (
                error.attempt_count == 1
                and isinstance(error.error, TransientError)
            )

    try:
        execute_with_resilience(
            always_fails,
            circuit_key="provider:down:text",
            max_retries=0,
            retry_base_seconds=0,
            retry_max_seconds=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=10,
            is_retryable=lambda error: isinstance(error, TransientError),
            breaker=breaker,
            sleeper=lambda _: None,
        )
    except CircuitOpenError as error:
        checks["连续失败后熔断并快速拒绝"] = (
            failing_attempts[0] == 2
            and round(error.retry_after_seconds) == 10
        )
    else:
        checks["连续失败后熔断并快速拒绝"] = False

    clock_value[0] += 11
    recovery = execute_with_resilience(
        lambda: "recovered",
        circuit_key="provider:down:text",
        max_retries=0,
        retry_base_seconds=0,
        retry_max_seconds=0,
        circuit_failure_threshold=2,
        circuit_recovery_seconds=10,
        is_retryable=lambda error: isinstance(error, TransientError),
        breaker=breaker,
        sleeper=lambda _: None,
    )
    checks["恢复窗口后半开探测成功"] = (
        recovery.value == "recovered"
        and recovery.circuit_state == "half_open"
    )

    print("====模型韧性测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n模型韧性测试：全部通过")


if __name__ == "__main__":
    main()
