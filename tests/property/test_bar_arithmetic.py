"""Hypothesis property tests for the bar-arithmetic kernel in alphaforge.core.time.

The bar arithmetic is the no-lookahead foundation of the whole system, so its
algebraic laws are checked exhaustively rather than by example.
"""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from alphaforge.core.time import (
    Timeframe,
    available_at,
    bar_open_for_decision,
    expected_bar_opens,
    floor_bar,
    next_bar_open,
)

# Epoch ms from 1970 to ~2100 — every timestamp the system will ever see.
MAX_MS = 4_102_444_800_000

timestamps = st.integers(min_value=0, max_value=MAX_MS)
timeframes = st.sampled_from(list(Timeframe))


@given(ts=timestamps, tf=timeframes)
def test_floor_bar_idempotent(ts: int, tf: Timeframe) -> None:
    assert floor_bar(floor_bar(ts, tf), tf) == floor_bar(ts, tf)


@given(ts=timestamps, tf=timeframes)
def test_floor_bar_brackets_ts(ts: int, tf: Timeframe) -> None:
    fb = floor_bar(ts, tf)
    assert fb <= ts < fb + tf.ms


@given(ts=timestamps, tf=timeframes)
def test_floor_bar_is_aligned(ts: int, tf: Timeframe) -> None:
    assert floor_bar(ts, tf) % tf.ms == 0


@given(ts=timestamps, tf=timeframes)
def test_next_bar_open_of_floor_is_exactly_one_bar_later(ts: int, tf: Timeframe) -> None:
    fb = floor_bar(ts, tf)
    assert next_bar_open(fb, tf) - fb == tf.ms


@given(ts=timestamps, tf=timeframes)
def test_next_bar_open_is_strictly_after_and_aligned(ts: int, tf: Timeframe) -> None:
    nb = next_bar_open(ts, tf)
    assert nb > ts
    assert nb % tf.ms == 0
    assert nb - ts <= tf.ms


@given(
    start_bar=st.integers(min_value=0, max_value=1_000_000),
    n_bars=st.integers(min_value=0, max_value=2_000),
    tf=timeframes,
)
def test_expected_bar_opens_aligned_length(start_bar: int, n_bars: int, tf: Timeframe) -> None:
    start = start_bar * tf.ms
    end = start + n_bars * tf.ms
    opens = expected_bar_opens(start, end, tf)
    assert len(opens) == (end - start) // tf.ms == n_bars


@given(start=timestamps, span=st.integers(min_value=0, max_value=400 * 86_400_000), tf=timeframes)
def test_expected_bar_opens_contents(start: int, span: int, tf: Timeframe) -> None:
    end = start + span
    opens = expected_bar_opens(start, end, tf)
    # Aligned, in-range, sorted, evenly spaced by exactly one bar.
    assert all(o % tf.ms == 0 and start <= o < end for o in opens)
    assert opens == sorted(opens)
    assert all(b - a == tf.ms for a, b in itertools.pairwise(opens))
    # Completeness: floor of any in-range aligned point is in the list.
    if opens:
        assert floor_bar(end - 1, tf) == opens[-1]


@given(ts1=timestamps, ts2=timestamps, tf=timeframes)
def test_available_at_strictly_monotone(ts1: int, ts2: int, tf: Timeframe) -> None:
    if ts1 < ts2:
        assert available_at(ts1, tf) < available_at(ts2, tf)
    elif ts1 == ts2:
        assert available_at(ts1, tf) == available_at(ts2, tf)
    else:
        assert available_at(ts1, tf) > available_at(ts2, tf)


@given(decision_ts=st.integers(min_value=86_400_000, max_value=MAX_MS), tf=timeframes)
def test_bar_open_for_decision_no_lookahead_and_no_lag(decision_ts: int, tf: Timeframe) -> None:
    ts_open = bar_open_for_decision(decision_ts, tf)
    # The returned bar is fully available at the decision time...
    assert available_at(ts_open, tf) <= decision_ts
    # ...and it is the LAST such bar: the next one is not yet available.
    assert available_at(ts_open + tf.ms, tf) > decision_ts
    assert ts_open % tf.ms == 0


@given(start_bar=st.integers(min_value=1, max_value=1_000_000), tf=timeframes)
def test_decision_at_close_uses_just_closed_bar(start_bar: int, tf: Timeframe) -> None:
    ts_open = start_bar * tf.ms
    close = available_at(ts_open, tf)
    assert bar_open_for_decision(close, tf) == ts_open
    assert bar_open_for_decision(close - 1, tf) == ts_open - tf.ms
