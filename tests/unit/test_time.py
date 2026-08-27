"""Unit tests for alphaforge.core.time — UTC discipline and bar arithmetic."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta, timezone

import pytest
from freezegun import freeze_time

from alphaforge.core.errors import NaiveDatetimeError
from alphaforge.core.time import (
    Timeframe,
    available_at,
    bar_open_for_decision,
    expected_bar_opens,
    floor_bar,
    from_ms,
    next_bar_open,
    now_ms,
    parse_utc,
    require_utc,
    to_ms,
)

H1 = Timeframe.H1
H4 = Timeframe.H4
D1 = Timeframe.D1

# 2024-01-02T00:00:00Z — a known aligned reference instant (divisible by 86_400_000).
T0 = 1_704_153_600_000


class TestTimeframe:
    def test_ms_values(self) -> None:
        assert H1.ms == 3_600_000
        assert H4.ms == 14_400_000
        assert D1.ms == 86_400_000

    def test_pandas_freq(self) -> None:
        assert H1.pandas_freq == "1h"
        assert H4.pandas_freq == "4h"
        assert D1.pandas_freq == "1D"

    def test_bars_per_year(self) -> None:
        assert H1.bars_per_year == 8_760.0
        assert H4.bars_per_year == 2_190.0
        assert D1.bars_per_year == 365.0

    def test_str_values_match_partition_labels(self) -> None:
        assert str(H1) == "1h"
        assert str(H4) == "4h"
        assert str(D1) == "1d"

    def test_bars_per_year_consistent_with_ms(self) -> None:
        year_ms = 365 * 86_400_000
        for tf in Timeframe:
            assert tf.bars_per_year == pytest.approx(year_ms / tf.ms)


class TestNowMs:
    def test_frozen_wall_clock(self) -> None:
        with freeze_time("2024-01-02T00:00:00Z"):
            assert now_ms() == T0

    def test_frozen_subsecond(self) -> None:
        # 0.5 s is exactly representable in binary float, so freezegun's
        # float-derived time_ns is exact here (0.123 would round down).
        with freeze_time("2024-01-02T00:00:00.500000Z"):
            assert now_ms() == T0 + 500

    def test_returns_int(self) -> None:
        assert isinstance(now_ms(), int)


class TestRequireUtc:
    def test_naive_rejected(self) -> None:
        with pytest.raises(NaiveDatetimeError, match="naive"):
            require_utc(datetime(2024, 1, 2))  # noqa: DTZ001 — naive on purpose

    def test_non_utc_aware_rejected_not_converted(self) -> None:
        dubai = timezone(timedelta(hours=4))
        with pytest.raises(NaiveDatetimeError, match="astimezone"):
            require_utc(datetime(2024, 1, 2, 4, 0, tzinfo=dubai))

    def test_utc_passes_through_identity(self) -> None:
        dt = datetime(2024, 1, 2, tzinfo=UTC)
        assert require_utc(dt) is dt


class TestToFromMs:
    def test_to_ms_known_value(self) -> None:
        assert to_ms(datetime(2024, 1, 2, tzinfo=UTC)) == T0

    def test_to_ms_epoch(self) -> None:
        assert to_ms(datetime(1970, 1, 1, tzinfo=UTC)) == 0

    def test_to_ms_floors_sub_millisecond(self) -> None:
        dt = datetime(2024, 1, 2, 0, 0, 0, 1999, tzinfo=UTC)  # 1.999 ms
        assert to_ms(dt) == T0 + 1

    def test_to_ms_rejects_naive(self) -> None:
        with pytest.raises(NaiveDatetimeError):
            to_ms(datetime(2024, 1, 2))  # noqa: DTZ001 — naive on purpose

    def test_from_ms_is_utc_aware(self) -> None:
        dt = from_ms(T0)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)
        assert dt == datetime(2024, 1, 2, tzinfo=UTC)

    def test_round_trip(self) -> None:
        for ms in (0, 1, T0, T0 + 59_999, 4_102_444_800_000):
            assert to_ms(from_ms(ms)) == ms

    def test_pre_epoch(self) -> None:
        assert from_ms(-1) == datetime(1969, 12, 31, 23, 59, 59, 999_000, tzinfo=UTC)
        assert to_ms(from_ms(-1)) == -1


class TestParseUtc:
    def test_zulu_suffix(self) -> None:
        assert parse_utc("2024-01-02T00:00:00Z") == T0

    def test_explicit_offset_normalized_to_utc(self) -> None:
        assert parse_utc("2024-01-02T04:00:00+04:00") == T0

    def test_milliseconds(self) -> None:
        assert parse_utc("2024-01-02T00:00:00.123Z") == T0 + 123

    def test_naive_string_rejected(self) -> None:
        with pytest.raises(NaiveDatetimeError, match="offset"):
            parse_utc("2024-01-02T00:00:00")

    def test_garbage_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            parse_utc("not-a-timestamp")


class TestFloorBar:
    def test_aligned_is_fixed_point(self) -> None:
        for tf in Timeframe:
            assert floor_bar(T0, tf) == T0

    def test_mid_bar(self) -> None:
        assert floor_bar(T0 + 1, H1) == T0
        assert floor_bar(T0 + H1.ms - 1, H1) == T0
        assert floor_bar(T0 + 5 * 3_600_000, H4) == T0 + 4 * 3_600_000
        assert floor_bar(T0 + 23 * 3_600_000, D1) == T0

    def test_negative_ts_floors_toward_minus_infinity(self) -> None:
        assert floor_bar(-1, H1) == -H1.ms


class TestNextBarOpen:
    def test_aligned_advances_exactly_one_bar(self) -> None:
        for tf in Timeframe:
            assert next_bar_open(T0, tf) == T0 + tf.ms

    def test_unaligned_snaps_to_next_open(self) -> None:
        assert next_bar_open(T0 + 1, H1) == T0 + H1.ms
        assert next_bar_open(T0 + H1.ms - 1, H1) == T0 + H1.ms

    def test_strictly_greater_than_input(self) -> None:
        for ts in (T0, T0 + 1, T0 + H4.ms - 1):
            assert next_bar_open(ts, H4) > ts


class TestAvailableAt:
    def test_equals_close(self) -> None:
        for tf in Timeframe:
            assert available_at(T0, tf) == T0 + tf.ms


class TestBarOpenForDecision:
    """Decision at T may use bars whose close <= T (dataDesign §0.3)."""

    def test_decision_exactly_at_close_includes_that_bar(self) -> None:
        # Bar [T0, T0+Δ) closes at T0+Δ: a decision at T0+Δ may use it.
        for tf in Timeframe:
            assert bar_open_for_decision(T0 + tf.ms, tf) == T0

    def test_one_ms_before_close_excludes_that_bar(self) -> None:
        # At T0+delta-1 the bar opening at T0 has not closed; the prior bar is the answer.
        for tf in Timeframe:
            assert bar_open_for_decision(T0 + tf.ms - 1, tf) == T0 - tf.ms

    def test_mid_bar_decision(self) -> None:
        # Decision halfway through bar [T0+Δ, T0+2Δ): last fully closed bar is [T0, T0+Δ).
        for tf in Timeframe:
            assert bar_open_for_decision(T0 + tf.ms + tf.ms // 2, tf) == T0

    def test_returned_bar_is_available_and_next_is_not(self) -> None:
        for tf in Timeframe:
            for decision_ts in (T0 + tf.ms, T0 + tf.ms + 1, T0 + 2 * tf.ms - 1):
                ts_open = bar_open_for_decision(decision_ts, tf)
                assert available_at(ts_open, tf) <= decision_ts
                assert available_at(ts_open + tf.ms, tf) > decision_ts


class TestExpectedBarOpens:
    def test_aligned_range(self) -> None:
        opens = expected_bar_opens(T0, T0 + 3 * H1.ms, H1)
        assert opens == [T0, T0 + H1.ms, T0 + 2 * H1.ms]

    def test_half_open_excludes_end(self) -> None:
        assert T0 + 3 * H1.ms not in expected_bar_opens(T0, T0 + 3 * H1.ms, H1)

    def test_unaligned_start_is_ceiled(self) -> None:
        assert expected_bar_opens(T0 + 1, T0 + 2 * H1.ms, H1) == [T0 + H1.ms]

    def test_unaligned_end_is_floored_by_half_open(self) -> None:
        assert expected_bar_opens(T0, T0 + H1.ms + 1, H1) == [T0, T0 + H1.ms]

    def test_empty_interval(self) -> None:
        assert expected_bar_opens(T0, T0, H1) == []
        assert expected_bar_opens(T0 + 1, T0 + H1.ms, H1) == []

    def test_all_opens_aligned(self) -> None:
        for tf in Timeframe:
            for ts_open in expected_bar_opens(T0 + 7, T0 + 5 * tf.ms + 11, tf):
                assert ts_open % tf.ms == 0


class TestDstIrrelevance:
    """UTC bar arithmetic must be unaffected by civil-time DST transitions."""

    def test_h1_count_across_us_spring_forward(self) -> None:
        # 2024-03-10: US springs forward; the UTC day still has exactly 24 hourly bars.
        start = parse_utc("2024-03-10T00:00:00Z")
        end = parse_utc("2024-03-11T00:00:00Z")
        assert len(expected_bar_opens(start, end, H1)) == 24

    def test_h1_count_across_eu_fall_back(self) -> None:
        # 2024-10-27: EU falls back; still exactly 24 UTC hourly bars.
        start = parse_utc("2024-10-27T00:00:00Z")
        end = parse_utc("2024-10-28T00:00:00Z")
        assert len(expected_bar_opens(start, end, H1)) == 24

    def test_d1_spacing_constant_across_dst_week(self) -> None:
        start = parse_utc("2024-03-08T00:00:00Z")
        end = parse_utc("2024-03-15T00:00:00Z")
        opens = expected_bar_opens(start, end, D1)
        assert len(opens) == 7
        assert all(b - a == D1.ms for a, b in itertools.pairwise(opens))
