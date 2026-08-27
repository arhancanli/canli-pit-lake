"""Retry policies and a request-weight rate budget for ingestion.

Two orthogonal protections for every exchange-facing fetch loop:

- :func:`transient_retry` — a tenacity ``Retrying`` factory for *transient*
  failures (network blips, 5xx, rate-limit responses): exponential backoff
  with full jitter, bounded attempts, re-raising the original exception on
  exhaustion. Non-listed exceptions (``BadSymbol``, auth errors, bugs)
  propagate immediately — retrying a permanent failure only hides it.

- :class:`RateBudget` — a client-side request-weight budget so we *never*
  trip the exchange's per-minute weight limit in the first place. Retrying
  after an HTTP 429 is the failure mode; the budget is the prevention.

Both are pure policy objects: they perform no I/O themselves and are fully
deterministic under injected clocks/sleepers (tests run with fake time).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Final

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from alphaforge.core.logging import get_logger

__all__ = [
    "BINANCE_FUTURES_REST_WEIGHT_PER_MIN",
    "RateBudget",
    "transient_retry",
]

_log = get_logger(__name__)

BINANCE_FUTURES_REST_WEIGHT_PER_MIN: Final[int] = 2000
"""Default :class:`RateBudget` capacity for Binance USDT-M futures REST.

Binance enforces a hard cap of 2400 request-weight per minute per IP on
``fapi``; exceeding it returns HTTP 429 and repeat offenders get an HTTP 418
IP ban. We deliberately budget 2000 (~83% of the cap) to leave headroom for
clock skew between our window and Binance's, response-header accounting lag,
and any other process sharing the IP. Staying under the cap by design beats
recovering from bans.
"""

_WINDOW_S: Final[float] = 60.0
#: Floor on computed sleeps so the acquire loop always makes progress even if
#: an injected clock has coarse granularity.
_MIN_SLEEP_S: Final[float] = 0.001


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """tenacity ``before_sleep`` hook: structlog WARNING per backoff sleep."""
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    sleep_s = 0.0 if retry_state.next_action is None else float(retry_state.next_action.sleep)
    _log.warning(
        "transient_retry.backoff",
        attempt=retry_state.attempt_number,
        sleep_s=round(sleep_s, 3),
        error=repr(exc),
    )


def transient_retry(
    exc_types: tuple[type[BaseException], ...],
    *,
    max_attempts: int = 8,
    base_s: float = 0.5,
    max_s: float = 60.0,
) -> Retrying:
    """Build a tenacity ``Retrying`` policy for transient failures.

    Retries calls raising any of ``exc_types`` (typically
    ``(ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable)``)
    up to ``max_attempts`` total attempts. Backoff between attempts is
    exponential with **full jitter** (AWS-style): each sleep is uniform in
    ``[0, min(max_s, base_s * 2**(attempt-1))]`` seconds, decorrelating
    concurrent clients hammering a recovering endpoint. ``reraise=True``:
    on exhaustion the *original* exception propagates, never a ``RetryError``.
    Exceptions outside ``exc_types`` propagate immediately, unretried.

    Each backoff sleep emits a structlog WARNING (``transient_retry.backoff``
    with ``attempt``, ``sleep_s``, ``error``) so flapping endpoints are visible
    in the JSONL ops log.

    Usage::

        retryer = transient_retry((ccxt.NetworkError,))
        page = retryer(exchange.fetch_ohlcv, symbol, "1h", since=cursor)

    Raises ``ValueError`` for an empty ``exc_types``, ``max_attempts < 1``,
    ``base_s <= 0`` or ``max_s < base_s``.
    """
    if not exc_types:
        raise ValueError("exc_types must name at least one exception type")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if base_s <= 0.0:
        raise ValueError(f"base_s must be > 0, got {base_s}")
    if max_s < base_s:
        raise ValueError(f"max_s ({max_s}) must be >= base_s ({base_s})")
    return Retrying(
        retry=retry_if_exception_type(exc_types),
        wait=wait_random_exponential(multiplier=base_s, max=max_s),
        stop=stop_after_attempt(max_attempts),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


class RateBudget:
    """Blocking request-weight budget: at most ``capacity_per_min`` weight in
    **any** sliding 60-second window.

    Implementation is a sliding-window grant log (a strict variant of the
    token bucket): every successful :meth:`acquire` records ``(time, weight)``
    and counts against the budget for exactly 60 seconds. A classic
    refill-rate token bucket was rejected deliberately — a full bucket plus
    one minute of refill permits up to 2x capacity inside a single sliding
    minute, which is exactly the window Binance meters.

    ``acquire`` blocks (via the injected ``sleeper``) until the budget allows
    the request, so callers simply do ``budget.acquire(weight)`` before every
    REST call and can never trip the exchange limit. Thread-safe: all state
    is guarded by a lock; sleeps happen outside it so waiters don't convoy.

    ``capacity_per_min`` defaults to
    :data:`BINANCE_FUTURES_REST_WEIGHT_PER_MIN` (2000 — deliberately under
    Binance's 2400/min hard cap; see that constant's docstring).

    ``sleeper``/``clock`` exist for deterministic tests: ``clock`` must be a
    monotonic seconds source (default ``time.monotonic``) and ``sleeper``
    takes a positive float of seconds (default ``time.sleep``).
    """

    def __init__(
        self,
        capacity_per_min: int = BINANCE_FUTURES_REST_WEIGHT_PER_MIN,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a budget of ``capacity_per_min`` weight per sliding minute.

        Raises ``ValueError`` if ``capacity_per_min < 1``.
        """
        if capacity_per_min < 1:
            raise ValueError(f"capacity_per_min must be >= 1, got {capacity_per_min}")
        self._capacity = capacity_per_min
        self._sleeper = sleeper
        self._clock = clock
        self._lock = threading.Lock()
        #: Successful grants as (grant_time_s, weight), oldest first.
        self._grants: deque[tuple[float, int]] = deque()
        #: Invariant: sum of weights currently in ``_grants``.
        self._used = 0

    def _evict_expired(self, now: float) -> None:
        """Drop grants older than the window. Caller must hold the lock."""
        cutoff = now - _WINDOW_S
        while self._grants and self._grants[0][0] <= cutoff:
            _, weight = self._grants.popleft()
            self._used -= weight

    def acquire(self, weight: int) -> None:
        """Consume ``weight`` from the budget, blocking until it fits.

        Returns only once the grant is recorded; the caller may then issue
        the request. Raises ``ValueError`` if ``weight < 1`` (nonsensical) or
        ``weight > capacity_per_min`` (could never be granted — fail loud
        instead of deadlocking). Blocking is via the injected ``sleeper``;
        with the defaults this sleeps the calling thread.
        """
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")
        if weight > self._capacity:
            raise ValueError(
                f"weight {weight} exceeds budget capacity {self._capacity}/min "
                "and could never be granted"
            )
        while True:
            with self._lock:
                now = self._clock()
                self._evict_expired(now)
                if self._used + weight <= self._capacity:
                    self._grants.append((now, weight))
                    self._used += weight
                    return
                # Walk oldest grants until enough weight will have expired,
                # then sleep to that grant's expiry and re-contend.
                needed = self._used + weight - self._capacity
                freed = 0
                wait_until = now
                for grant_time, grant_weight in self._grants:
                    freed += grant_weight
                    wait_until = grant_time + _WINDOW_S
                    if freed >= needed:
                        break
            self._sleeper(max(wait_until - now, _MIN_SLEEP_S))

    def remaining(self) -> float:
        """Weight still grantable right now without blocking (0..capacity)."""
        with self._lock:
            self._evict_expired(self._clock())
            return float(self._capacity - self._used)
