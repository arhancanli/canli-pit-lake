"""UTC time discipline and bar arithmetic — the temporal contract of AlphaForge.

Conventions (load-bearing; see dataDesign.md §0 and buildabilityCritique.md ruling 3.1):

- All timestamps are UTC. At API boundaries they are integer epoch **milliseconds**
  (:data:`Ms`). Naive datetimes are a programming error and raise
  :class:`~alphaforge.core.errors.NaiveDatetimeError`.
- Bars are labeled by **open** time (``ts_open``). A bar of timeframe Δ covers
  ``[ts_open, ts_open + Δ)`` and is *available at* ``ts_open + Δ`` (its close).
- A decision made at timestamp ``T`` may only consume bars with ``available_at <= T``;
  :func:`bar_open_for_decision` computes the open of the last such bar mechanically.
- :func:`now_ms` is the single sanctioned wall-clock read in the codebase.

All bar arithmetic is pure integer math on :data:`Ms`; Python's floor division
(rounding toward -inf) makes every helper total over negative (pre-epoch) inputs.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from alphaforge.core.errors import NaiveDatetimeError

__all__ = [
    "Ms",
    "Timeframe",
    "available_at",
    "bar_open_for_decision",
    "expected_bar_opens",
    "floor_bar",
    "from_ms",
    "next_bar_open",
    "now_ms",
    "parse_utc",
    "require_utc",
    "to_ms",
]

type Ms = int
"""Integer epoch milliseconds, UTC.

The only timestamp representation allowed at API boundaries. Convert at the edge with
:func:`to_ms` / :func:`from_ms` / :func:`parse_utc`; never pass ``datetime`` objects
through internal interfaces.
"""

_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)
_ZERO_OFFSET: Final[timedelta] = timedelta(0)


def now_ms() -> Ms:
    """Return the current wall-clock time as UTC epoch milliseconds.

    This is the single sanctioned wall-clock read in AlphaForge. All other code must
    take timestamps as injectable :data:`Ms` parameters so logic stays testable and
    backtest/live paths share code.
    """
    return time.time_ns() // 1_000_000


class Timeframe(StrEnum):
    """Bar timeframe. Values match the lake partition labels (``timeframe=1h`` etc.)."""

    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def ms(self) -> int:
        """Bar duration Δ in milliseconds (exact; 24/7 market, no DST)."""
        return _TF_MS[self]

    @property
    def pandas_freq(self) -> str:
        """The pandas offset alias for resampling (``"1h"``, ``"4h"``, ``"1D"``)."""
        return _TF_PANDAS_FREQ[self]

    @property
    def bars_per_year(self) -> float:
        """Bars per 365-day year for a 24/7 market (leap days ignored, <0.3% effect)."""
        return _TF_BARS_PER_YEAR[self]


_TF_MS: Final[dict[Timeframe, int]] = {
    Timeframe.H1: 3_600_000,
    Timeframe.H4: 14_400_000,
    Timeframe.D1: 86_400_000,
}

_TF_PANDAS_FREQ: Final[dict[Timeframe, str]] = {
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
}

_TF_BARS_PER_YEAR: Final[dict[Timeframe, float]] = {
    Timeframe.H1: 8_760.0,
    Timeframe.H4: 2_190.0,
    Timeframe.D1: 365.0,
}


def require_utc(dt: datetime) -> datetime:
    """Return ``dt`` unchanged if it is timezone-aware with a UTC (zero) offset.

    Raises:
        NaiveDatetimeError: if ``dt`` is naive, or if its tzinfo has a non-zero UTC
            offset. Conversion is deliberately NOT performed — a non-UTC datetime at
            an AlphaForge boundary is a programming error; the caller must convert
            explicitly with ``dt.astimezone(datetime.UTC)``.
    """
    offset = dt.utcoffset()
    if offset is None:
        raise NaiveDatetimeError(
            f"naive datetime {dt!r}: AlphaForge requires UTC-aware datetimes; "
            "construct with tzinfo=datetime.UTC"
        )
    if offset != _ZERO_OFFSET:
        raise NaiveDatetimeError(
            f"non-UTC datetime {dt!r} (offset {offset}): convert explicitly with "
            "dt.astimezone(datetime.UTC) before crossing an AlphaForge boundary"
        )
    return dt


def to_ms(dt: datetime) -> Ms:
    """Convert a UTC-aware datetime to epoch milliseconds (exact integer arithmetic).

    Sub-millisecond precision is floored (toward -inf). Raises
    :class:`NaiveDatetimeError` via :func:`require_utc` on naive/non-UTC input.
    """
    delta = require_utc(dt) - _EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def from_ms(ms: Ms) -> datetime:
    """Convert epoch milliseconds to a UTC-aware ``datetime`` (exact, no float round-trip)."""
    return _EPOCH + timedelta(milliseconds=ms)


def parse_utc(s: str) -> Ms:
    """Parse an ISO-8601 string with an explicit ``Z`` or numeric offset to epoch ms.

    Strings with a non-UTC offset are normalized to UTC — the offset is explicit in
    the input, so unlike :func:`require_utc` there is no ambiguity to reject.

    Raises:
        NaiveDatetimeError: if the string carries no UTC offset (naive).
        ValueError: if the string is not valid ISO-8601 (propagated from
            ``datetime.fromisoformat``).
    """
    dt = datetime.fromisoformat(s)
    if dt.utcoffset() is None:
        raise NaiveDatetimeError(
            f"naive ISO-8601 string {s!r}: an explicit 'Z' or numeric offset is required"
        )
    return to_ms(dt.astimezone(UTC))


def floor_bar(ts: Ms, tf: Timeframe) -> Ms:
    """Return the open (``ts_open``) of the ``tf`` bar containing ``ts``.

    The containing bar is ``[floor_bar(ts), floor_bar(ts) + tf.ms)``; an aligned ``ts``
    is its own bar open. Total over negative ``ts`` (floors toward -inf).
    """
    return ts // tf.ms * tf.ms


def next_bar_open(ts: Ms, tf: Timeframe) -> Ms:
    """Return the open of the first ``tf`` bar strictly after ``ts``.

    For an aligned ``ts`` (a bar open) this is ``ts + tf.ms`` — the execution bar for
    a decision taken at the close ``ts`` of the previous bar.
    """
    return floor_bar(ts, tf) + tf.ms


def available_at(ts_open: Ms, tf: Timeframe) -> Ms:
    """Return when the bar opening at ``ts_open`` becomes consumable: its close.

    ``available_at = ts_open + tf.ms``. A decision at ``T`` may use this bar iff
    ``available_at <= T`` — the mechanical no-lookahead predicate.
    """
    return ts_open + tf.ms


def bar_open_for_decision(decision_ts: Ms, tf: Timeframe) -> Ms:
    """Return the open of the LAST ``tf`` bar fully available at ``decision_ts``.

    That is the greatest aligned ``ts_open`` with ``ts_open + tf.ms <= decision_ts``.
    A decision taken exactly at a bar close may use that just-closed bar; one
    millisecond earlier it may not (the previous bar is returned instead).
    """
    return floor_bar(decision_ts - tf.ms, tf)


def expected_bar_opens(start_ms: Ms, end_ms: Ms, tf: Timeframe) -> list[Ms]:
    """Return all aligned ``tf`` bar opens in the half-open interval ``[start_ms, end_ms)``.

    ``start_ms`` is ceiled to the next aligned open if unaligned; ``end_ms`` is
    exclusive. For aligned bounds the result has exactly ``(end_ms - start_ms) // tf.ms``
    elements. Returns ``[]`` when the interval contains no aligned open. This is the
    gap-detection reference grid (24/7 calendar; session calendars filter it later).
    """
    step = tf.ms
    first = -(-start_ms // step) * step  # ceil toward +inf, exact for ints
    return list(range(first, end_ms, step))
