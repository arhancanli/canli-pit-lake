"""Hypothesis property tests for alphaforge.core.calendar.

The funding schedule and bar grid feed funding-history gap checks and annualization;
their algebraic laws (UTC-midnight anchoring, half-open bounds, constant spacing,
interval refinement) are checked exhaustively rather than by example.
"""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from alphaforge.core import time as core_time
from alphaforge.core.calendar import Always24x7Calendar
from alphaforge.core.time import Timeframe

HOUR_MS = 3_600_000
DAY_MS = 86_400_000

# Epoch ms from 1970 to ~2100 — every timestamp the system will ever see.
MAX_MS = 4_102_444_800_000

timestamps = st.integers(min_value=0, max_value=MAX_MS)
spans = st.integers(min_value=0, max_value=3 * DAY_MS)
funding_hours = st.sampled_from([1, 2, 3, 4, 6, 8, 12, 24])  # all positive divisors of 24
timeframes = st.sampled_from(list(Timeframe))

CAL = Always24x7Calendar()


@given(start=timestamps, span=spans, hours=funding_hours)
def test_funding_events_anchored_at_utc_midnight(start: int, span: int, hours: int) -> None:
    """Every event ≡ 0 mod interval relative to UTC midnight (epoch is a UTC midnight)."""
    interval_ms = hours * HOUR_MS
    for e in CAL.funding_events_in(start, start + span, hours):
        assert e % interval_ms == 0
        assert (e % DAY_MS) % interval_ms == 0


@given(start=timestamps, span=spans, hours=funding_hours)
def test_funding_events_respect_half_open_bounds(start: int, span: int, hours: int) -> None:
    end = start + span
    for e in CAL.funding_events_in(start, end, hours):
        assert start <= e < end


@given(start=timestamps, span=spans, hours=funding_hours)
def test_funding_events_constant_spacing(start: int, span: int, hours: int) -> None:
    events = CAL.funding_events_in(start, start + span, hours)
    interval_ms = hours * HOUR_MS
    for a, b in itertools.pairwise(events):
        assert b - a == interval_ms


@given(start=timestamps, span=spans, hours=funding_hours)
def test_funding_events_complete(start: int, span: int, hours: int) -> None:
    """Event count matches independent ceil arithmetic — no aligned instant is skipped."""
    end = start + span
    interval_ms = hours * HOUR_MS
    first = -(-start // interval_ms) * interval_ms
    expected_count = (end - 1 - first) // interval_ms + 1 if end > first else 0
    assert len(CAL.funding_events_in(start, end, hours)) == expected_count


@given(ts=timestamps, hours=funding_hours)
def test_funding_event_window_around_aligned_instant(ts: int, hours: int) -> None:
    """An interval-wide window starting at an aligned instant contains exactly it."""
    interval_ms = hours * HOUR_MS
    e0 = ts // interval_ms * interval_ms
    assert CAL.funding_events_in(e0, e0 + interval_ms, hours) == [e0]


@given(start=timestamps, span=spans)
def test_funding_intervals_refine(start: int, span: int) -> None:
    """Coarser schedules are subsets of finer ones: 8h ⊆ 4h ⊆ 2h ⊆ 1h."""
    end = start + span
    by_hours = {h: set(CAL.funding_events_in(start, end, h)) for h in (8, 4, 2, 1)}
    assert by_hours[8] <= by_hours[4] <= by_hours[2] <= by_hours[1]


@given(start=timestamps, span=spans, tf=timeframes)
def test_expected_bar_opens_constant_spacing_is_tf_ms(start: int, span: int, tf: Timeframe) -> None:
    opens = CAL.expected_bar_opens(start, start + span, tf)
    for a, b in itertools.pairwise(opens):
        assert b - a == tf.ms


@given(start=timestamps, span=spans, tf=timeframes)
def test_expected_bar_opens_aligned_and_bounded(start: int, span: int, tf: Timeframe) -> None:
    end = start + span
    for ts_open in CAL.expected_bar_opens(start, end, tf):
        assert ts_open % tf.ms == 0
        assert start <= ts_open < end


@given(start=timestamps, span=spans, tf=timeframes)
def test_expected_bar_opens_matches_core_time(start: int, span: int, tf: Timeframe) -> None:
    end = start + span
    assert CAL.expected_bar_opens(start, end, tf) == core_time.expected_bar_opens(start, end, tf)


@given(ts=timestamps, tf=timeframes)
def test_bar_arithmetic_delegates_to_core_time(ts: int, tf: Timeframe) -> None:
    assert CAL.floor_bar(ts, tf) == core_time.floor_bar(ts, tf)
    assert CAL.next_bar_open(ts, tf) == core_time.next_bar_open(ts, tf)


@given(ts=timestamps, tf=timeframes)
def test_session_and_annualization_invariants(ts: int, tf: Timeframe) -> None:
    assert CAL.is_session(ts) is True
    assert CAL.periods_per_year(tf) == tf.bars_per_year
