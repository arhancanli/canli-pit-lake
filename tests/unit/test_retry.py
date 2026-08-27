"""Unit tests for alphaforge.data.ingest.retry — transient_retry and RateBudget."""

from __future__ import annotations

import random
import threading

import pytest
import structlog.testing

from alphaforge.data.ingest.retry import (
    BINANCE_FUTURES_REST_WEIGHT_PER_MIN,
    RateBudget,
    transient_retry,
)


class Flaky:
    """Callable failing ``failures`` times with ``exc`` before returning ``value``."""

    def __init__(self, failures: int, exc: BaseException, value: int = 42) -> None:
        self.failures = failures
        self.exc = exc
        self.value = value
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return self.value


class FakeClock:
    """Deterministic monotonic clock + sleeper pair for RateBudget tests."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0.0, "RateBudget must never request a non-positive sleep"
        self.sleeps.append(seconds)
        self.t += seconds


def make_budget(capacity: int) -> tuple[RateBudget, FakeClock]:
    clock = FakeClock()
    return RateBudget(capacity, sleeper=clock.sleep, clock=clock), clock


class TestTransientRetry:
    def test_succeeds_on_third_attempt(self) -> None:
        flaky = Flaky(failures=2, exc=ConnectionError("blip"))
        retryer = transient_retry((ConnectionError,), base_s=0.001, max_s=0.002)
        assert retryer(flaky) == 42
        assert flaky.calls == 3

    def test_exhausts_and_reraises_original_exception(self) -> None:
        flaky = Flaky(failures=99, exc=ConnectionError("down"))
        retryer = transient_retry((ConnectionError,), max_attempts=3, base_s=0.001, max_s=0.002)
        with pytest.raises(ConnectionError, match="down"):  # original, not RetryError
            retryer(flaky)
        assert flaky.calls == 3

    def test_unlisted_exception_propagates_immediately(self) -> None:
        flaky = Flaky(failures=99, exc=KeyError("permanent"))
        retryer = transient_retry((ConnectionError,), base_s=0.001, max_s=0.002)
        with pytest.raises(KeyError, match="permanent"):
            retryer(flaky)
        assert flaky.calls == 1

    def test_retries_any_listed_type(self) -> None:
        flaky = Flaky(failures=1, exc=TimeoutError("slow"))
        retryer = transient_retry((ConnectionError, TimeoutError), base_s=0.001, max_s=0.002)
        assert retryer(flaky) == 42
        assert flaky.calls == 2

    def test_before_sleep_logs_via_structlog(self) -> None:
        flaky = Flaky(failures=2, exc=ConnectionError("blip"))
        retryer = transient_retry((ConnectionError,), base_s=0.001, max_s=0.002)
        with structlog.testing.capture_logs() as logs:
            retryer(flaky)
        backoffs = [entry for entry in logs if entry["event"] == "transient_retry.backoff"]
        assert len(backoffs) == 2  # one per failed attempt that led to a sleep
        assert [entry["attempt"] for entry in backoffs] == [1, 2]
        assert all("ConnectionError" in entry["error"] for entry in backoffs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_attempts": 0},
            {"base_s": 0.0},
            {"base_s": -1.0},
            {"max_s": 0.1, "base_s": 0.5},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            transient_retry((ConnectionError,), **kwargs)  # type: ignore[arg-type]

    def test_empty_exc_types_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            transient_retry(())


class TestRateBudget:
    def test_default_capacity_is_documented_binance_budget(self) -> None:
        assert BINANCE_FUTURES_REST_WEIGHT_PER_MIN == 2000
        budget = RateBudget()
        assert budget.remaining() == float(BINANCE_FUTURES_REST_WEIGHT_PER_MIN)

    def test_under_capacity_never_sleeps(self) -> None:
        budget, clock = make_budget(100)
        for _ in range(10):
            budget.acquire(10)
        assert clock.sleeps == []
        assert budget.remaining() == 0.0

    def test_remaining_tracks_usage_and_replenishes(self) -> None:
        budget, clock = make_budget(100)
        assert budget.remaining() == 100.0
        budget.acquire(30)
        assert budget.remaining() == 70.0
        clock.t += 59.9
        assert budget.remaining() == 70.0  # grant still inside the window
        clock.t += 0.2
        assert budget.remaining() == 100.0  # expired after a full minute

    def test_blocks_then_proceeds(self) -> None:
        budget, clock = make_budget(60)
        budget.acquire(60)  # t=0: exhausts the budget
        clock.t = 10.0
        budget.acquire(30)  # must wait for the t=0 grant to expire at t=60
        assert clock.sleeps != []
        assert clock.t == pytest.approx(60.0)
        assert budget.remaining() == 30.0

    def test_weight_above_capacity_raises(self) -> None:
        budget, _ = make_budget(100)
        with pytest.raises(ValueError, match="exceeds budget capacity"):
            budget.acquire(101)

    @pytest.mark.parametrize("weight", [0, -5])
    def test_nonpositive_weight_raises(self, weight: int) -> None:
        budget, _ = make_budget(100)
        with pytest.raises(ValueError, match="weight must be >= 1"):
            budget.acquire(weight)

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity_per_min"):
            RateBudget(0)

    def test_never_exceeds_capacity_in_any_sliding_minute(self) -> None:
        """Random workload: granted weight in every (t-60, t] window <= capacity."""
        capacity = 100
        budget, clock = make_budget(capacity)
        rng = random.Random(7)
        grants: list[tuple[float, int]] = []
        for _ in range(300):
            weight = rng.randint(1, 40)
            clock.t += rng.random() * 5.0  # bursty arrivals
            budget.acquire(weight)
            grants.append((clock.t, weight))
        for now, _ in grants:
            window_sum = sum(w for t, w in grants if now - 60.0 < t <= now)
            assert window_sum <= capacity, f"window ending at t={now} carries {window_sum}"

    def test_thread_safety_exact_capacity_no_overdraft(self) -> None:
        """Concurrent acquires summing to exactly capacity all pass without blocking."""
        capacity = 6000
        budget = RateBudget(capacity)  # real clock; no sleep needed at exact capacity
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    budget.acquire(60)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert 0.0 <= budget.remaining() <= float(capacity)
        assert budget.remaining() == pytest.approx(0.0)
