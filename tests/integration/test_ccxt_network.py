"""Network integration tests for ``CCXTDataSource`` against live Binance USDT-M.

These tests hit real public endpoints (``/fapi/v1/klines``, ``/fapi/v1/fundingRate``,
``exchangeInfo`` + ``/fapi/v1/fundingInfo`` via ccxt) and are therefore:

* marked ``@pytest.mark.network`` — deselected by default ``addopts``
  (``-m 'not network'``); run explicitly with ``uv run pytest -m network``;
* sized to finish well under 30 s each (a handful of weight-cheap requests);
* asserting *shape and sanity*, never exact values — live data changes every hour.

They double as the Phase 2 gate's API-reachability probe (buildabilityCritique.md
§2/§7): a geo-block or endpoint change fails here first, not mid-backfill.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from alphaforge.core.time import Ms, Timeframe, now_ms
from alphaforge.data.ingest.retry import RateBudget
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.ccxt_source import FUNDING_PUBLICATION_LAG_MS, CCXTDataSource

pytestmark = pytest.mark.network

BTC_ID = "BINANCE:PERP:BTCUSDT"
H1_MS = Timeframe.H1.ms
FUNDING_8H_MS = 8 * H1_MS


@pytest.fixture(scope="module")
def source() -> CCXTDataSource:
    """One real ccxt-backed source per module, metered by the default rate budget."""
    return CCXTDataSource("binanceusdm", rate_budget=RateBudget())


def test_fetch_ohlcv_five_recent_btc_1h_bars(source: CCXTDataSource) -> None:
    """Five fully-closed recent BTCUSDT 1h bars: exact count, schema, price sanity."""
    now = now_ms()
    until = (now // H1_MS) * H1_MS  # most recent bar boundary; everything before is closed
    since = until - 5 * H1_MS

    tbl = source.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=since, until=until, now=now)

    validate_table(tbl, Dataset.OHLCV)
    assert tbl.num_rows == 5
    ts = tbl.column("ts_open").cast(pa.int64()).to_pylist()
    assert ts == [since + i * H1_MS for i in range(5)]  # contiguous, sorted, in-range

    opens = tbl.column("open").to_pylist()
    highs = tbl.column("high").to_pylist()
    lows = tbl.column("low").to_pylist()
    closes = tbl.column("close").to_pylist()
    for o, h, low, c in zip(opens, highs, lows, closes, strict=True):
        assert 1_000.0 < low <= min(o, c) <= max(o, c) <= h < 10_000_000.0  # OHLC coherence
    assert all(v > 0.0 for v in tbl.column("volume").to_pylist())  # BTC always trades
    assert all(qv is not None and qv > 0.0 for qv in tbl.column("quote_volume").to_pylist())
    assert all(n is not None and n > 0 for n in tbl.column("n_trades").to_pylist())
    assert set(tbl.column("quality_flags").to_pylist()) == {0}
    assert all(
        ia >= now for ia in tbl.column("ingested_at").cast(pa.int64()).to_pylist()
    )  # ingestion stamp is wall clock at fetch, after our `now` snapshot

    assert all(instr == BTC_ID for instr in tbl.column("instrument_id").to_pylist())


def test_fetch_funding_three_settled_btc_rows(source: CCXTDataSource) -> None:
    """Three most recent fully-settled 8h funding rows with the 5-min PIT lag."""
    now = now_ms()
    last_boundary = (now // FUNDING_8H_MS) * FUNDING_8H_MS  # 00/08/16 UTC grid
    since: Ms = last_boundary - 3 * FUNDING_8H_MS
    until: Ms = last_boundary  # exclusive: skip a possibly-just-settled boundary

    tbl = source.fetch_funding(BTC_ID, since=since, until=until)

    validate_table(tbl, Dataset.FUNDING)
    assert tbl.num_rows == 3
    ts = tbl.column("ts_funding").cast(pa.int64()).to_pylist()
    # Live fundingTime sits on the 8h settlement grid plus a few ms of exchange
    # jitter (observed: +0..+11 ms); allow up to 60 s past each boundary.
    for actual, boundary in zip(ts, (since + i * FUNDING_8H_MS for i in range(3)), strict=True):
        assert 0 <= actual - boundary < 60_000
    avail = tbl.column("available_at").cast(pa.int64()).to_pylist()
    assert [a - t for a, t in zip(avail, ts, strict=True)] == [FUNDING_PUBLICATION_LAG_MS] * 3
    rates = tbl.column("rate").to_pylist()
    # Binance clamps BTC perp funding to roughly +/-0.75% per interval; +/-5% is
    # a generous sanity band that still catches unit errors (percent vs decimal).
    assert all(-0.05 < r < 0.05 for r in rates)
    assert all(instr == BTC_ID for instr in tbl.column("instrument_id").to_pylist())


def test_list_instruments_with_live_funding_info(source: CCXTDataSource) -> None:
    """Live exchangeInfo + fundingInfo: USDT-perp-only listing, interval overlay applied."""
    as_of = now_ms()
    instruments = source.list_instruments(as_of=as_of)

    assert len(instruments) > 100  # Binance lists several hundred USDT-M perps
    by_id = {inst.instrument_id: inst for inst in instruments}
    assert all(iid.startswith("BINANCE:PERP:") for iid in by_id)
    assert all(inst.quote == "USDT" for inst in instruments)  # USDC/dated futures skipped

    btc = by_id[BTC_ID]
    assert btc.tick_size > 0.0
    assert btc.lot_size > 0.0
    assert btc.min_qty > 0.0
    assert btc.min_notional > 0.0
    assert btc.can_short is True
    assert btc.funding_interval_hours == 8  # BTC funds on the default 8h schedule
    assert btc.listed_ts is not None and btc.listed_ts < as_of
    assert btc.delisted_ts is None  # if this fires, bigger problems than this test

    # fundingInfo overlay reached the instruments: Binance keeps a sizable set of
    # alt perps on non-default (1h/4h) funding at any given time.
    non_default = [inst for inst in instruments if inst.funding_interval_hours in (1, 2, 4)]
    assert non_default, "expected at least one perp with a non-8h funding interval"
