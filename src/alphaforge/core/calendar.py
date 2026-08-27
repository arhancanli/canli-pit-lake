"""Trading calendars — session membership, bar grids, annualization, funding schedules.

ONE calendar abstraction (buildabilityCritique.md ruling 3.3): dataDesign.md's
``TradingCalendar`` (``expected_bar_opens``, ``is_session``) and execDesign.md's
``BarCalendar`` (``periods_per_year``, bar arithmetic, funding instants) are the same
object, merged here. Annualization factors are never hard-coded downstream — they come
from :meth:`TradingCalendar.periods_per_year` so an equities session calendar (~252-day
basis, via the ``exchange_calendars`` package) plugs in later without engine changes.

All timestamps are UTC epoch milliseconds (:data:`~alphaforge.core.time.Ms`); all
ranges are half-open ``[start_ms, end_ms)``.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Final

from alphaforge.core import time as core_time
from alphaforge.core.errors import ConfigError
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import AssetClass

__all__ = [
    "Always24x7Calendar",
    "TradingCalendar",
    "XNYSCalendar",
    "calendar_for",
]

_MS_PER_HOUR: Final[int] = 3_600_000
_MS_PER_DAY: Final[int] = 86_400_000


class TradingCalendar(ABC):
    """Defines when bars exist, how many there are per year, and where funding falls.

    Crypto: a bar every Δ, around the clock (:class:`Always24x7Calendar`). Equities
    later: exchange sessions filter the 24/7 grid — gap detection, bar-availability
    math, and annualization all consult this object, never their own constants.
    """

    @abstractmethod
    def is_session(self, ts: Ms) -> bool:
        """Return True iff the market trades at instant ``ts`` (epoch ms, UTC)."""

    @abstractmethod
    def expected_bar_opens(self, start_ms: Ms, end_ms: Ms, tf: Timeframe) -> list[Ms]:
        """Return every ``tf`` bar open the market should produce in ``[start_ms, end_ms)``.

        This is the reference grid for gap detection: missing bars are this set minus
        the lake's actual ``ts_open`` values. ``start_ms`` unaligned is ceiled to the
        next valid open; ``end_ms`` is exclusive. Result is sorted ascending.
        """

    @abstractmethod
    def periods_per_year(self, tf: Timeframe) -> float:
        """Return the annualization factor: bars of ``tf`` per year on this calendar.

        24/7 crypto: 8760.0 (1h), 2190.0 (4h), 365.0 (1d) — leap days ignored
        (<0.3% effect). The single source for Sharpe/vol/covariance annualization;
        hard-coding 8760 anywhere else is a bug.
        """

    @abstractmethod
    def floor_bar(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return the open (``ts_open``) of the ``tf`` bar containing ``ts``."""

    @abstractmethod
    def next_bar_open(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return the open of the first ``tf`` bar strictly after ``ts``."""

    @property
    def contiguous(self) -> bool:
        """True iff every aligned Δ-instant is a bar — no session gaps — so the absolute
        epoch-anchored bar index ``ts // Δ`` is dense and consecutive.

        Only on a contiguous grid is the EPOCH-FIXED non-overlapping subsample valid
        (:func:`alphaforge.validation.metrics.non_overlapping` with ``delta_ms`` set:
        keep bars where ``(ts // Δ) % h == 0``) AND the uniform-ms-spacing invariant the
        blend asserts (consecutive non-overlapping points exactly ``h·Δ`` apart). On a
        GAPPY session calendar both break: ``ts // Δ`` skips weekend/holiday indices, so
        the subsample must instead keep every ``h``-th SESSION POSITIONALLY and the blend
        must trust that positional construction rather than ms spacing.

        Default ``False`` (the safe choice for any session calendar);
        :class:`Always24x7Calendar` overrides to ``True``.
        """
        return False

    def funding_events_in(self, start_ms: Ms, end_ms: Ms, interval_hours: int) -> list[Ms]:
        """Return scheduled perp funding-settlement instants in ``[start_ms, end_ms)``.

        Settlements are anchored at UTC midnight and repeat every ``interval_hours``
        (per-instrument; from the instruments table): 8h -> 00/08/16 UTC, 4h ->
        00/04/.../20 UTC, 1h -> every hour. Epoch 0 is a UTC midnight, so events are
        exactly the integer multiples of ``interval_hours`` in the half-open range,
        sorted ascending.

        SCHEDULE HELPER ONLY (leakageCritique.md finding 6): the backtest and live
        ledger apply funding by iterating the stored ``funding_events`` table — actual
        settled rates with their own ``available_at`` — never this clock. Use this for
        scheduling, gap-checking the funding history, and expected-event counts.

        Raises:
            ValueError: if ``interval_hours`` is not a positive divisor of 24 (the
                anchored daily pattern would not repeat).
        """
        if interval_hours <= 0 or 24 % interval_hours != 0:
            raise ValueError(
                f"interval_hours must be a positive divisor of 24 "
                f"(Binance uses 8, 4, or 1); got {interval_hours}"
            )
        interval_ms = interval_hours * _MS_PER_HOUR
        first = -(-start_ms // interval_ms) * interval_ms  # ceil to next aligned instant
        return list(range(first, end_ms, interval_ms))


class Always24x7Calendar(TradingCalendar):
    """Calendar for markets that never close (crypto perp/spot).

    Every instant is a session; the bar grid is the full aligned grid, so all bar
    arithmetic delegates to the pure-integer kernel in :mod:`alphaforge.core.time`.
    Stateless — share one instance via :func:`calendar_for`.
    """

    __slots__ = ()

    def is_session(self, ts: Ms) -> bool:
        """Return True for every ``ts`` — a 24/7 market is always in session."""
        return True

    def expected_bar_opens(self, start_ms: Ms, end_ms: Ms, tf: Timeframe) -> list[Ms]:
        """Return all aligned ``tf`` opens in ``[start_ms, end_ms)`` (full grid, no gaps)."""
        return core_time.expected_bar_opens(start_ms, end_ms, tf)

    def periods_per_year(self, tf: Timeframe) -> float:
        """Return ``tf.bars_per_year`` (365-day year, 24 hours a day)."""
        return tf.bars_per_year

    def floor_bar(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return :func:`alphaforge.core.time.floor_bar` — every aligned bar exists."""
        return core_time.floor_bar(ts, tf)

    def next_bar_open(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return :func:`alphaforge.core.time.next_bar_open` — every aligned bar exists."""
        return core_time.next_bar_open(ts, tf)

    @property
    def contiguous(self) -> bool:
        """Always True — the 24/7 grid has no session gaps (every aligned Δ is a bar)."""
        return True


_ALWAYS_24X7: Final[Always24x7Calendar] = Always24x7Calendar()

# Bounded session horizon for the cached XNYS schedule (EQUITIES_SLEEVE.md §3.1, and the
# N2 caveat in EQUITIES_CRITIQUE.md). The start covers pre-2008 fixtures (Lehman/LEHMQ
# survivorship); the end is a few years past the live horizon. ``exchange_calendars``
# future sessions are *estimates* — the LIVE loop must not trust a cached forward window
# for a late-announced ad-hoc closure (presidential funeral, etc.) and should refresh the
# calendar, the same spirit as the universe finding-21 ordering contract. For a settled
# backtest the cached history is exact.
_XNYS_HORIZON_START: Final[str] = "2004-01-01"
_XNYS_HORIZON_END: Final[str] = "2030-12-31"

_XNYS_SESSIONS_PER_YEAR: Final[float] = 252.0
"""Annualization basis for XNYS daily bars (the ~252-session trading year)."""


@lru_cache(maxsize=1)
def _xnys_session_opens() -> tuple[Ms, ...]:
    """Sorted XNYS session opens as UTC-midnight epoch ms over the bounded horizon.

    Lazily wraps the ``exchange_calendars`` ``XNYS`` calendar (the sanctioned holiday
    source — we never reimplement the rules). A session label is a tz-naive midnight
    ``Timestamp`` whose date IS the session date; its ns value equals that date at
    00:00 UTC, which is exactly the equity daily-bar ``ts_open`` convention
    (EQUITIES_SLEEVE.md §3.2). Cached once (``maxsize=1``) — the schedule is static
    for the backtest horizon.

    Raises:
        ConfigError: if ``exchange_calendars`` is not installed (a hard dependency for
            any equity path; never silently fall back to a 24/7 grid).
    """
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise ConfigError(
            "exchange_calendars is required for the XNYS equities calendar; "
            "install the 'exchange-calendars' package"
        ) from exc
    cal = xcals.get_calendar("XNYS", start=_XNYS_HORIZON_START, end=_XNYS_HORIZON_END)
    # ``sessions`` is a tz-naive DatetimeIndex of midnight labels; asi8 is ns since epoch.
    return tuple(int(ns) // 1_000_000 for ns in cal.sessions.asi8)


class XNYSCalendar(TradingCalendar):
    """NYSE (``XNYS``) session calendar — equities daily bars, ~252-session year.

    Sessions filter the 24/7 grid: a daily bar exists only on an XNYS trading session,
    so a missing bar on a session is a real gap (a halted name) while a weekend/holiday
    is correctly absent (EQUITIES_SLEEVE.md §3.1). Backed by the ``exchange_calendars``
    package — holiday rules are wrapped, never reimplemented. Stateless and shared via
    :func:`calendar_for`; the session schedule is cached once over a bounded horizon.

    THE BAR/AVAILABILITY BOUNDARY (EQUITIES_CRITIQUE.md B2): availability math stays
    pure-integer (a daily bar labeled ``ts_open`` is available at ``ts_open + tf.ms`` —
    the reader's predicate is unchanged). But bar *selection* — :meth:`floor_bar`,
    :meth:`next_bar_open`, :meth:`expected_bar_opens` — MUST hop sessions, never add
    ``tf.ms``: ``next_bar_open`` of a Friday is the following Monday session, not
    Saturday. Anything doing ``ts + tf.ms`` to find the next equity bar is a bug.

    v1 supports ``Timeframe.D1`` only; intraday (open/close-minute precision) is post-v1
    and any other timeframe raises :class:`NotImplementedError` so an accidental
    intraday equity request fails loud rather than returning mis-aligned bars.
    """

    __slots__ = ()

    @staticmethod
    def _require_d1(tf: Timeframe) -> None:
        if tf is not Timeframe.D1:
            raise NotImplementedError(
                f"XNYSCalendar supports Timeframe.D1 only in v1, got {tf.value!r}: "
                "intraday equity bars (session open/close minutes) are post-v1"
            )

    def is_session(self, ts: Ms) -> bool:
        """Return True iff the UTC day containing ``ts`` is an XNYS trading session.

        v1 works at session (date) granularity: ``ts`` is floored to its UTC midnight
        and tested for session membership; intraday open/close-minute precision is
        post-v1.
        """
        opens = _xnys_session_opens()
        day_open = core_time.floor_bar(ts, Timeframe.D1)
        idx = bisect.bisect_left(opens, day_open)
        return idx < len(opens) and opens[idx] == day_open

    def expected_bar_opens(self, start_ms: Ms, end_ms: Ms, tf: Timeframe) -> list[Ms]:
        """Return every XNYS session open (00:00 UTC) in ``[start_ms, end_ms)``, sorted.

        The session-filtered grid: the equities analogue of the crypto full grid, and
        the reference for gap detection (a missing session bar is a real gap). ``tf``
        must be :data:`Timeframe.D1` (v1).
        """
        self._require_d1(tf)
        opens = _xnys_session_opens()
        lo = bisect.bisect_left(opens, start_ms)
        hi = bisect.bisect_left(opens, end_ms)
        return list(opens[lo:hi])

    def periods_per_year(self, tf: Timeframe) -> float:
        """Return ``252.0`` for ``D1`` — the single annualization source for equities.

        Sharpe/vol/covariance pull annualization from here, so no hard-coded 8760/365
        (the 24/7 :attr:`Timeframe.bars_per_year`) leaks into the equity path
        (EQUITIES_SLEEVE.md §3.1). ``tf`` must be :data:`Timeframe.D1` (v1).
        """
        self._require_d1(tf)
        return _XNYS_SESSIONS_PER_YEAR

    def floor_bar(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return the open (00:00 UTC) of the session covering ``ts``.

        The greatest session open ``<= ts``: if ``ts`` falls on a non-session day
        (weekend/holiday) this is the *prior* session's open, NOT a phantom weekend
        slot (EQUITIES_CRITIQUE.md B2). ``tf`` must be :data:`Timeframe.D1` (v1).

        Raises:
            ValueError: if ``ts`` precedes the first cached session.
        """
        self._require_d1(tf)
        opens = _xnys_session_opens()
        idx = bisect.bisect_right(opens, ts) - 1
        if idx < 0:
            raise ValueError(
                f"ts {ts} precedes the first XNYS session {opens[0]} "
                f"({_XNYS_HORIZON_START}); widen the calendar horizon"
            )
        return opens[idx]

    def next_bar_open(self, ts: Ms, tf: Timeframe) -> Ms:
        """Return the open of the first XNYS session strictly after ``ts``.

        Hops weekends/holidays: the bar after a Friday session is the following
        Monday session, never ``ts + tf.ms`` (EQUITIES_CRITIQUE.md B2). ``tf`` must be
        :data:`Timeframe.D1` (v1).

        Raises:
            ValueError: if no cached session follows ``ts`` (past the horizon end).
        """
        self._require_d1(tf)
        opens = _xnys_session_opens()
        idx = bisect.bisect_right(opens, ts)
        if idx >= len(opens):
            raise ValueError(
                f"ts {ts} is at or past the last cached XNYS session {opens[-1]} "
                f"({_XNYS_HORIZON_END}); widen the calendar horizon"
            )
        return opens[idx]


_XNYS: Final[XNYSCalendar] = XNYSCalendar()


def calendar_for(asset_class: AssetClass) -> TradingCalendar:
    """Return the trading calendar for ``asset_class`` (shared stateless instance).

    CRYPTO_PERP and CRYPTO_SPOT map to the singleton :class:`Always24x7Calendar`;
    EQUITY maps to the singleton :class:`XNYSCalendar` (NYSE sessions, 252-day basis,
    via ``exchange_calendars``).

    Dated futures and options deliberately have no generic calendar: session boundaries
    differ by exchange and product. They fail until a product-specific calendar is supplied.

    Raises:
        NotImplementedError: for any asset class with no registered calendar.
    """
    if asset_class in (AssetClass.CRYPTO_PERP, AssetClass.CRYPTO_SPOT):
        return _ALWAYS_24X7
    if asset_class is AssetClass.EQUITY:
        return _XNYS
    raise NotImplementedError(
        f"no TradingCalendar registered for asset class {asset_class.value!r}: "
        "only crypto_perp/crypto_spot and equity are supported"
    )
