from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from time import monotonic, sleep
from typing import Generic, TypeVar


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a provider circuit is open and calls are rejected."""

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(retry_after_seconds, 0.0)
        super().__init__("模型服务暂时不可用，熔断器已开启。")


class ResilienceExhaustedError(RuntimeError):
    """Preserve the original provider error and actual attempt count."""

    def __init__(self, error: Exception, attempt_count: int) -> None:
        self.error = error
        self.attempt_count = attempt_count
        super().__init__(str(error))


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class CircuitBreaker:
    """Thread-safe circuit breaker isolated by provider/model key."""

    def __init__(self, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._lock = RLock()
        self._states: dict[str, _CircuitState] = {}

    def before_call(self, key: str, recovery_seconds: float) -> str:
        with self._lock:
            state = self._states.setdefault(key, _CircuitState())
            if state.opened_at is None:
                return "closed"

            elapsed = self._clock() - state.opened_at
            if elapsed < recovery_seconds:
                raise CircuitOpenError(recovery_seconds - elapsed)

            if state.probe_in_flight:
                raise CircuitOpenError(1.0)

            state.probe_in_flight = True
            return "half_open"

    def record_success(self, key: str) -> None:
        with self._lock:
            self._states[key] = _CircuitState()

    def record_failure(self, key: str, failure_threshold: int) -> None:
        with self._lock:
            state = self._states.setdefault(key, _CircuitState())
            state.consecutive_failures += 1
            state.probe_in_flight = False
            if state.consecutive_failures >= failure_threshold:
                state.opened_at = self._clock()

    def reset(self) -> None:
        with self._lock:
            self._states.clear()


@dataclass(frozen=True)
class ResilientCallResult(Generic[T]):
    value: T
    attempt_count: int
    circuit_state: str


GLOBAL_MODEL_CIRCUIT_BREAKER = CircuitBreaker()


def execute_with_resilience(
    operation: Callable[[], T],
    *,
    circuit_key: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    circuit_failure_threshold: int,
    circuit_recovery_seconds: float,
    is_retryable: Callable[[Exception], bool],
    breaker: CircuitBreaker = GLOBAL_MODEL_CIRCUIT_BREAKER,
    sleeper: Callable[[float], None] = sleep,
) -> ResilientCallResult[T]:
    """Execute one logical provider call with retry and circuit breaking."""
    circuit_state = breaker.before_call(
        circuit_key,
        recovery_seconds=circuit_recovery_seconds,
    )
    attempt_count = 0

    while True:
        attempt_count += 1
        try:
            value = operation()
        except Exception as error:
            retryable = is_retryable(error)
            if not retryable:
                # A validation or authentication error proves the provider is
                # reachable, so it must not poison service-availability state.
                breaker.record_success(circuit_key)
                raise

            if attempt_count <= max_retries:
                delay = min(
                    retry_base_seconds * (2 ** (attempt_count - 1)),
                    retry_max_seconds,
                )
                sleeper(delay)
                continue

            breaker.record_failure(
                circuit_key,
                failure_threshold=circuit_failure_threshold,
            )
            raise ResilienceExhaustedError(
                error=error,
                attempt_count=attempt_count,
            ) from error

        breaker.record_success(circuit_key)
        return ResilientCallResult(
            value=value,
            attempt_count=attempt_count,
            circuit_state=circuit_state,
        )
