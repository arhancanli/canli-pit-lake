"""Hypothesis property tests for alphaforge.data.store.resample.resample_bars.

The aggregation identities are checked over random gap-free 1h panels rather than by
example: extrema (high = max, low = min), first/last identities (open/close), exact
volume/quote_volume/n_trades sums (volumes are dyadic rationals — multiples of 1/64 —
so float sums carry zero rounding error), bitwise-OR flag inheritance, the count
identity ``n_4h == n_1h // 4`` on aligned complete spans, the composition identity
``D1 ∘ (H1→H4) == H1→D1`` on complete days, and the closed-bucket guard.
"""

from __future__ import annotations

from typing import TypedDict

import pyarrow as pa
from hypothesis import given
from hypothesis import strategies as st

from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import Dataset, schema_for
from alphaforge.data.store.resample import resample_bars

HOUR = 3_600_000
H4 = 4 * HOUR
DAY = 24 * HOUR
T0 = 1_704_153_600_000  # 2024-01-02T00:00:00Z — aligned for 1h, 4h and 1d
BTC = "BINANCE:PERP:BTCUSDT"
SIXTY_FOURTH = 0.015625  # 2^-6: k·SIXTY_FOURTH is exact in binary64 for k < 2^53


class Panel(TypedDict):
    """A gap-free 1h series as plain Python columns (exact expected-value math)."""

    ts: list[int]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    vol_k: list[int]  # volume = vol_k / 64 — dyadic, so sums are float-exact
    qv_k: list[int]
    n_trades: list[int]
    flags: list[int]


prices = st.floats(min_value=1e-3, max_value=1e9, allow_nan=False, allow_infinity=False)
dyadic_counts = st.integers(min_value=0, max_value=2**32)
trade_counts = st.integers(min_value=0, max_value=10**9)
flag_bits = st.integers(min_value=0, max_value=31)


@st.composite
def panels(draw: st.DrawFn, *, bars_per_bucket: int, max_buckets: int) -> Panel:
    n = draw(st.integers(min_value=1, max_value=max_buckets)) * bars_per_bucket
    cols = st.lists(prices, min_size=n, max_size=n)
    return Panel(
        ts=[T0 + i * HOUR for i in range(n)],
        open=draw(cols),
        high=draw(cols),
        low=draw(cols),
        close=draw(cols),
        vol_k=draw(st.lists(dyadic_counts, min_size=n, max_size=n)),
        qv_k=draw(st.lists(dyadic_counts, min_size=n, max_size=n)),
        n_trades=draw(st.lists(trade_counts, min_size=n, max_size=n)),
        flags=draw(st.lists(flag_bits, min_size=n, max_size=n)),
    )


def to_table(p: Panel) -> pa.Table:
    schema = schema_for(Dataset.OHLCV)
    n = len(p["ts"])
    data: dict[str, list[object]] = {
        "instrument_id": [BTC] * n,
        "ts_open": list(p["ts"]),
        "open": list(p["open"]),
        "high": list(p["high"]),
        "low": list(p["low"]),
        "close": list(p["close"]),
        "volume": [k * SIXTY_FOURTH for k in p["vol_k"]],
        "quote_volume": [k * SIXTY_FOURTH for k in p["qv_k"]],
        "n_trades": list(p["n_trades"]),
        "quality_flags": list(p["flags"]),
        "ingested_at": [T0] * n,
    }
    return pa.Table.from_pydict(
        {name: pa.array(data[name], type=schema.field(name).type) for name in schema.names},
        schema=schema,
    )


def end_of(p: Panel) -> int:
    return p["ts"][-1] + HOUR


def h4_of(p: Panel) -> pa.Table:
    batch = resample_bars(to_table(p), source=Timeframe.H1, target=Timeframe.H4, now=end_of(p))
    assert batch.buckets_incomplete == 0
    assert batch.buckets_unclosed == 0
    return batch.table


def windows(p: Panel, size: int) -> list[range]:
    return [range(i, i + size) for i in range(0, len(p["ts"]), size)]


@given(p=panels(bars_per_bucket=4, max_buckets=5))
def test_high_is_max_and_low_is_min(p: Panel) -> None:
    out = h4_of(p)
    highs: list[float] = out.column("high").to_pylist()
    lows: list[float] = out.column("low").to_pylist()
    for bucket, w in enumerate(windows(p, 4)):
        assert highs[bucket] == max(p["high"][i] for i in w)
        assert lows[bucket] == min(p["low"][i] for i in w)


@given(p=panels(bars_per_bucket=4, max_buckets=5))
def test_open_is_first_and_close_is_last(p: Panel) -> None:
    out = h4_of(p)
    opens: list[float] = out.column("open").to_pylist()
    closes: list[float] = out.column("close").to_pylist()
    for bucket, w in enumerate(windows(p, 4)):
        assert opens[bucket] == p["open"][w[0]]
        assert closes[bucket] == p["close"][w[-1]]


@given(p=panels(bars_per_bucket=4, max_buckets=5))
def test_sums_are_exact(p: Panel) -> None:
    # Dyadic volumes (k/64) make every partial sum exactly representable: the float
    # sums must equal the integer-arithmetic expectation to the last bit.
    out = h4_of(p)
    vols: list[float] = out.column("volume").to_pylist()
    qvols: list[float] = out.column("quote_volume").to_pylist()
    trades: list[int] = out.column("n_trades").to_pylist()
    for bucket, w in enumerate(windows(p, 4)):
        assert vols[bucket] == sum(p["vol_k"][i] for i in w) * SIXTY_FOURTH
        assert qvols[bucket] == sum(p["qv_k"][i] for i in w) * SIXTY_FOURTH
        assert trades[bucket] == sum(p["n_trades"][i] for i in w)


@given(p=panels(bars_per_bucket=4, max_buckets=5))
def test_flags_are_bitwise_or(p: Panel) -> None:
    out = h4_of(p)
    flags: list[int] = out.column("quality_flags").to_pylist()
    for bucket, w in enumerate(windows(p, 4)):
        expected = 0
        for i in w:
            expected |= p["flags"][i]
        assert flags[bucket] == expected


@given(p=panels(bars_per_bucket=4, max_buckets=8))
def test_count_identity_on_aligned_complete_spans(p: Panel) -> None:
    out = h4_of(p)
    assert out.num_rows == len(p["ts"]) // 4
    assert out.column("ts_open").cast(pa.int64()).to_pylist() == [
        T0 + b * H4 for b in range(len(p["ts"]) // 4)
    ]


@given(p=panels(bars_per_bucket=24, max_buckets=2))
def test_composition_d1_of_h4_equals_d1_of_h1(p: Panel) -> None:
    now = end_of(p)
    direct = resample_bars(to_table(p), source=Timeframe.H1, target=Timeframe.D1, now=now).table
    via_h4 = resample_bars(h4_of(p), source=Timeframe.H4, target=Timeframe.D1, now=now).table
    assert direct.num_rows == len(p["ts"]) // 24
    for name in schema_for(Dataset.OHLCV).names:
        assert via_h4.column(name).to_pylist() == direct.column(name).to_pylist(), name


@given(p=panels(bars_per_bucket=4, max_buckets=4))
def test_nothing_closed_means_nothing_emitted(p: Panel) -> None:
    # At now = T0 not a single bucket is closed: every bucket is skipped, none written.
    batch = resample_bars(to_table(p), source=Timeframe.H1, target=Timeframe.H4, now=T0)
    assert batch.table.num_rows == 0
    assert batch.buckets_unclosed == len(p["ts"]) // 4


@given(p=panels(bars_per_bucket=4, max_buckets=4), drop=st.data())
def test_dropping_one_bar_kills_exactly_its_bucket(p: Panel, drop: st.DataObject) -> None:
    idx = drop.draw(st.integers(min_value=0, max_value=len(p["ts"]) - 1), label="dropped bar")
    tbl = to_table(p)
    tbl = tbl.filter(pa.array([i != idx for i in range(tbl.num_rows)], type=pa.bool_()))
    batch = resample_bars(tbl, source=Timeframe.H1, target=Timeframe.H4, now=end_of(p))
    n_buckets = len(p["ts"]) // 4
    assert batch.buckets_incomplete == 1
    assert batch.table.num_rows == n_buckets - 1
    dropped_bucket = T0 + (idx // 4) * H4
    assert dropped_bucket not in batch.table.column("ts_open").cast(pa.int64()).to_pylist()
