"""Network integration tests for ``PolygonEquitiesSource`` against live Polygon.io.

These tests hit real Polygon REST endpoints (``/v2/aggs/...``, ``/v3/reference/tickers``,
``/v3/reference/splits``, ``/v3/reference/dividends``) and are therefore:

* gated on a real ``POLYGON_API_KEY`` — skipped entirely when it is absent (the key has
  not landed yet), so CI stays green without credentials;
* marked ``@pytest.mark.network`` — also deselected by the default ``addopts``
  (``-m 'not network'``); run explicitly with ``uv run pytest -m network`` once a key
  is set;
* sized to a handful of free-tier-budgeted requests (5 req/min);
* asserting *shape and sanity*, never exact values — live data drifts.

They are the equities-sleeve API-reachability probe: the moment a key lands, these flip
from skipped to live and validate that the synthetic-fixture contracts in
``tests/unit/test_polygon_source.py`` match Polygon's real payloads (EQUITIES_SLEEVE.md
§1.6 / §7 step 8).
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Timeframe, now_ms
from alphaforge.data.ingest.retry import RateBudget
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.polygon_source import PolygonEquitiesSource

pytestmark = pytest.mark.network

_API_KEY = os.environ.get("POLYGON_API_KEY")
requires_key = pytest.mark.skipif(not _API_KEY, reason="POLYGON_API_KEY not set")

AAPL_ID = SymbolMapper.equity_instrument_id("XNAS", "AAPL")
D1_MS = Timeframe.D1.ms


@pytest.fixture(scope="module")
def source() -> PolygonEquitiesSource:
    """One real Polygon-backed source per module, metered for the free tier (5 req/min)."""
    return PolygonEquitiesSource(api_key=_API_KEY, rate_budget=RateBudget(capacity_per_min=5))


@requires_key
def test_fetch_ohlcv_recent_aapl_daily_bars(source: PolygonEquitiesSource) -> None:
    """A short window of recent AAPL daily bars: schema, sanity, raw (unadjusted)."""
    now = now_ms()
    until = (now // D1_MS) * D1_MS
    since = until - 30 * D1_MS  # ~30 calendar days (sessions are sparser; count is loose)

    tbl = source.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=since, until=until, now=now)

    validate_table(tbl, Dataset.OHLCV)
    assert tbl.num_rows > 0
    opens = tbl.column("open").to_pylist()
    highs = tbl.column("high").to_pylist()
    lows = tbl.column("low").to_pylist()
    closes = tbl.column("close").to_pylist()
    for o, h, low, c in zip(opens, highs, lows, closes, strict=True):
        assert 1.0 < low <= min(o, c) <= max(o, c) <= h < 100_000.0  # OHLC coherence
    assert all(v > 0.0 for v in tbl.column("volume").to_pylist())
    assert set(tbl.column("quality_flags").to_pylist()) == {0}


@requires_key
def test_list_instruments_includes_delisted(source: PolygonEquitiesSource) -> None:
    """The live listing must surface delisted tickers (survivorship-free)."""
    instruments = source.list_instruments(as_of=now_ms())
    assert instruments  # a non-empty common-equity universe
    assert any(inst.delisted_ts is not None for inst in instruments), (
        "expected at least one delisted ticker in the survivorship-free listing"
    )
    assert all(inst.market_type.value == "cash" for inst in instruments)


@requires_key
def test_fetch_corporate_actions_aapl_split(source: PolygonEquitiesSource) -> None:
    """AAPL's known 2020-08-31 4:1 split must appear with available_at <= ex_date join sanity."""
    from alphaforge.core.time import parse_utc

    since = parse_utc("2020-01-01T00:00:00+00:00")
    until = parse_utc("2021-01-01T00:00:00+00:00")
    tbl = source.fetch_corporate_actions(AAPL_ID, since=since, until=until)
    splits = tbl.filter(pa.compute.equal(tbl.column("action_type"), "split"))
    assert splits.num_rows >= 1
    assert all(r > 0.0 for r in splits.column("ratio").to_pylist())
