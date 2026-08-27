"""Unit tests for alphaforge.core.calendar — merged calendar, funding schedule, registry."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphaforge.core import time as core_time
from alphaforge.core.calendar import (
    Always24x7Calendar,
    TradingCalendar,
    XNYSCalendar,
    calendar_for,
)
from alphaforge.core.time import Ms, Timeframe, to_ms
from alphaforge.core.types import AssetClass

H1 = Timeframe.H1
H4 = Timeframe.H4
D1 = Timeframe.D1

HOUR_MS = 3_600_000
DAY_MS = 86_400_000

# 2024-01-02T00:00:00Z — a UTC midnight, aligned to every funding interval.
T0 = 1_704_153_600_000

CAL = Always24x7Calendar()


class TestPeriodsPerYear:
    def test_h1_is_8760(self) -> None:
        assert CAL.periods_per_year(H1) == 8760.0

    def test_h4_is_2190(self) -> None:
        assert CAL.periods_per_year(H4) == 2190.0

    def test_d1_is_365(self) -> None:
        assert CAL.periods_per_year(D1) == 365.0

    def test_matches_timeframe_bars_per_year(self) -> None:
        for tf in Timeframe:
            assert CAL.periods_per_year(tf) == tf.bars_per_year


class TestIsSession:
    def test_always_true(self) -> None:
        # Epoch, a weekday midnight, a Saturday (2024-01-06), an unaligned instant.
        for ts in (0, T0, T0 + 4 * DAY_MS, T0 + 12_345_678):
            assert CAL.is_session(ts) is True


class TestBarArithmeticDelegation:
    def test_floor_bar_matches_core_time(self) -> None:
        ts = T0 + 5 * HOUR_MS + 17
        assert CAL.floor_bar(ts, H1) == core_time.floor_bar(ts, H1) == T0 + 5 * HOUR_MS
        assert CAL.floor_bar(ts, H4) == core_time.floor_bar(ts, H4) == T0 + 4 * HOUR_MS
        assert CAL.floor_bar(ts, D1) == core_time.floor_bar(ts, D1) == T0

    def test_next_bar_open_matches_core_time(self) -> None:
        ts = T0 + 5 * HOUR_MS + 17
        assert CAL.next_bar_open(ts, H1) == core_time.next_bar_open(ts, H1) == T0 + 6 * HOUR_MS
        assert CAL.next_bar_open(T0, D1) == T0 + DAY_MS

    def test_expected_bar_opens_matches_core_time(self) -> None:
        start, end = T0 - 2 * HOUR_MS, T0 + 3 * HOUR_MS
        opens = CAL.expected_bar_opens(start, end, H1)
        assert opens == core_time.expected_bar_opens(start, end, H1)
        assert opens == [T0 + k * HOUR_MS for k in range(-2, 3)]

    def test_expected_bar_opens_empty_range(self) -> None:
        assert CAL.expected_bar_opens(T0, T0, H1) == []


class TestFundingEvents:
    def test_8h_across_day_boundary(self) -> None:
        # [2024-01-01T20:00Z, 2024-01-02T10:00Z) -> settlements at 00:00 and 08:00.
        events = CAL.funding_events_in(T0 - 4 * HOUR_MS, T0 + 10 * HOUR_MS, 8)
        assert events == [T0, T0 + 8 * HOUR_MS]

    def test_4h_across_day_boundary(self) -> None:
        # Same window -> 20:00 (Jan 1), 00:00, 04:00, 08:00 (Jan 2).
        events = CAL.funding_events_in(T0 - 4 * HOUR_MS, T0 + 10 * HOUR_MS, 4)
        assert events == [T0 - 4 * HOUR_MS, T0, T0 + 4 * HOUR_MS, T0 + 8 * HOUR_MS]

    def test_1h_is_hourly(self) -> None:
        events = CAL.funding_events_in(T0 - 4 * HOUR_MS, T0 + 10 * HOUR_MS, 1)
        assert events == [T0 + k * HOUR_MS for k in range(-4, 10)]

    def test_8h_full_day_pattern(self) -> None:
        # One UTC day of 8h funding -> exactly 00:00, 08:00, 16:00.
        events = CAL.funding_events_in(T0, T0 + DAY_MS, 8)
        assert events == [T0, T0 + 8 * HOUR_MS, T0 + 16 * HOUR_MS]

    def test_start_inclusive(self) -> None:
        assert CAL.funding_events_in(T0, T0 + 1, 8) == [T0]

    def test_end_exclusive(self) -> None:
        assert CAL.funding_events_in(T0 - 8 * HOUR_MS, T0, 8) == [T0 - 8 * HOUR_MS]

    def test_unaligned_start_skips_just_passed_event(self) -> None:
        assert CAL.funding_events_in(T0 + 1, T0 + 8 * HOUR_MS + 1, 8) == [T0 + 8 * HOUR_MS]

    def test_empty_when_no_event_in_range(self) -> None:
        assert CAL.funding_events_in(T0, T0, 8) == []
        assert CAL.funding_events_in(T0 + 1, T0 + 2, 8) == []

    def test_interval_24h_is_daily(self) -> None:
        assert CAL.funding_events_in(T0, T0 + 2 * DAY_MS, 24) == [T0, T0 + DAY_MS]

    @pytest.mark.parametrize("bad_hours", [0, -8, 5, 7, 9, 48])
    def test_invalid_interval_raises(self, bad_hours: int) -> None:
        with pytest.raises(ValueError, match="divisor of 24"):
            CAL.funding_events_in(T0, T0 + DAY_MS, bad_hours)


class TestCalendarFor:
    def test_crypto_perp_is_24x7(self) -> None:
        cal = calendar_for(AssetClass.CRYPTO_PERP)
        assert isinstance(cal, Always24x7Calendar)
        assert isinstance(cal, TradingCalendar)

    def test_crypto_spot_is_24x7(self) -> None:
        assert isinstance(calendar_for(AssetClass.CRYPTO_SPOT), Always24x7Calendar)

    def test_shared_singleton(self) -> None:
        assert calendar_for(AssetClass.CRYPTO_PERP) is calendar_for(AssetClass.CRYPTO_SPOT)

    def test_equity_is_xnys(self) -> None:
        cal = calendar_for(AssetClass.EQUITY)
        assert isinstance(cal, XNYSCalendar)
        assert isinstance(cal, TradingCalendar)

    def test_equity_shared_singleton(self) -> None:
        assert calendar_for(AssetClass.EQUITY) is calendar_for(AssetClass.EQUITY)

    def test_equity_and_crypto_are_distinct(self) -> None:
        assert calendar_for(AssetClass.EQUITY) is not calendar_for(AssetClass.CRYPTO_PERP)


def _ms(year: int, month: int, day: int) -> Ms:
    """UTC-midnight epoch ms for a calendar date (the equity ``ts_open`` convention)."""
    return to_ms(datetime(year, month, day, tzinfo=UTC))


XNYS = XNYSCalendar()


class TestXNYSSessions:
    """Session membership against KNOWN NYSE holidays (the wrapped exchange_calendars
    rules), at session-date granularity."""

    def test_regular_weekday_is_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 1, 4)) is True  # first trading Monday of 2021

    def test_weekend_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 1, 2)) is False  # Saturday
        assert XNYS.is_session(_ms(2021, 1, 3)) is False  # Sunday

    def test_new_year_holiday_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 1, 1)) is False

    def test_mlk_day_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 1, 18)) is False

    def test_independence_day_observed_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 7, 5)) is False  # July 4 (Sun) observed Mon

    def test_thanksgiving_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 11, 25)) is False

    def test_christmas_observed_is_not_session(self) -> None:
        assert XNYS.is_session(_ms(2021, 12, 24)) is False  # Dec 25 (Sat) observed Fri 24

    def test_is_session_ignores_intraday_offset(self) -> None:
        # v1 works at session-date granularity: an intraday instant on a session day
        # is in-session; on a holiday it is not.
        assert XNYS.is_session(_ms(2021, 1, 4) + 13 * HOUR_MS) is True
        assert XNYS.is_session(_ms(2021, 1, 1) + 13 * HOUR_MS) is False


class TestXNYSExpectedBarOpens:
    def test_holiday_week_skips_holiday_and_weekend(self) -> None:
        # 2021-01-01 Fri (New Year, closed), 02 Sat, 03 Sun, then Mon..Fri sessions.
        opens = XNYS.expected_bar_opens(_ms(2021, 1, 1), _ms(2021, 1, 9), D1)
        assert opens == [_ms(2021, 1, d) for d in (4, 5, 6, 7, 8)]

    def test_half_open_bounds(self) -> None:
        # end exclusive: the 8th is excluded; start inclusive: the 4th is included.
        opens = XNYS.expected_bar_opens(_ms(2021, 1, 4), _ms(2021, 1, 8), D1)
        assert opens == [_ms(2021, 1, d) for d in (4, 5, 6, 7)]

    def test_empty_range(self) -> None:
        assert XNYS.expected_bar_opens(_ms(2021, 1, 4), _ms(2021, 1, 4), D1) == []

    def test_weekend_only_range_is_empty(self) -> None:
        assert XNYS.expected_bar_opens(_ms(2021, 1, 2), _ms(2021, 1, 4), D1) == []

    def test_sorted_ascending(self) -> None:
        opens = XNYS.expected_bar_opens(_ms(2021, 1, 1), _ms(2021, 2, 1), D1)
        assert opens == sorted(opens)
        assert len(set(opens)) == len(opens)


class TestXNYSBarArithmeticHopsSessions:
    """floor_bar/next_bar_open MUST hop sessions, never add tf.ms (EQUITIES_CRITIQUE B2)."""

    def test_next_bar_open_hops_weekend(self) -> None:
        # After Fri 2021-01-08 the next bar is Mon 2021-01-11, NOT Saturday.
        assert XNYS.next_bar_open(_ms(2021, 1, 8), D1) == _ms(2021, 1, 11)

    def test_next_bar_open_from_intraday_friday(self) -> None:
        assert XNYS.next_bar_open(_ms(2021, 1, 8) + 5 * HOUR_MS, D1) == _ms(2021, 1, 11)

    def test_next_bar_open_hops_holiday(self) -> None:
        # Fri 2021-12-31 -> Sat/Sun, Mon 2022-01-03 (Jan 1 is Sat, no Fri-31 closure here).
        assert XNYS.next_bar_open(_ms(2021, 12, 31), D1) == _ms(2022, 1, 3)

    def test_floor_bar_on_session_is_itself(self) -> None:
        assert XNYS.floor_bar(_ms(2021, 1, 4), D1) == _ms(2021, 1, 4)

    def test_floor_bar_on_weekend_returns_prior_session(self) -> None:
        # Sun 2021-01-10 -> prior Friday session 2021-01-08, NOT a phantom Sunday slot.
        assert XNYS.floor_bar(_ms(2021, 1, 10), D1) == _ms(2021, 1, 8)

    def test_floor_bar_on_holiday_returns_prior_session(self) -> None:
        # New Year Fri 2021-01-01 -> prior session is Thu 2020-12-31.
        assert XNYS.floor_bar(_ms(2021, 1, 1), D1) == _ms(2020, 12, 31)

    def test_floor_bar_intraday_floors_to_session_open(self) -> None:
        assert XNYS.floor_bar(_ms(2021, 1, 4) + 17 * HOUR_MS, D1) == _ms(2021, 1, 4)

    def test_floor_then_next_round_trip(self) -> None:
        # The session after floor_bar(ts) is next_bar_open(floor_bar(ts)).
        ts = _ms(2021, 3, 10) + 9 * HOUR_MS
        floored = XNYS.floor_bar(ts, D1)
        assert XNYS.next_bar_open(floored, D1) == _ms(2021, 3, 11)

    def test_floor_bar_before_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="precedes the first XNYS session"):
            XNYS.floor_bar(_ms(2000, 1, 1), D1)


class TestXNYSAnnualization:
    def test_d1_is_252(self) -> None:
        assert XNYS.periods_per_year(D1) == 252.0

    def test_not_the_crypto_365(self) -> None:
        # The whole point of §3.1: the equity path is 252, never the 24/7 D1 365.
        assert XNYS.periods_per_year(D1) != Timeframe.D1.bars_per_year


class TestXNYSIntradayGuards:
    """v1 is D1-only; any other timeframe fails loud (mis-aligned intraday bars)."""

    @pytest.mark.parametrize("tf", [H1, H4])
    def test_expected_bar_opens_non_d1_raises(self, tf: Timeframe) -> None:
        with pytest.raises(NotImplementedError, match="D1 only"):
            XNYS.expected_bar_opens(_ms(2021, 1, 1), _ms(2021, 1, 9), tf)

    @pytest.mark.parametrize("tf", [H1, H4])
    def test_periods_per_year_non_d1_raises(self, tf: Timeframe) -> None:
        with pytest.raises(NotImplementedError, match="D1 only"):
            XNYS.periods_per_year(tf)

    @pytest.mark.parametrize("tf", [H1, H4])
    def test_floor_bar_non_d1_raises(self, tf: Timeframe) -> None:
        with pytest.raises(NotImplementedError, match="D1 only"):
            XNYS.floor_bar(_ms(2021, 1, 4), tf)

    @pytest.mark.parametrize("tf", [H1, H4])
    def test_next_bar_open_non_d1_raises(self, tf: Timeframe) -> None:
        with pytest.raises(NotImplementedError, match="D1 only"):
            XNYS.next_bar_open(_ms(2021, 1, 4), tf)
