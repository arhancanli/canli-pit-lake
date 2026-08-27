"""Unit tests for alphaforge.data.sources.polygon_source — PolygonEquitiesSource against
a fake Polygon client with canned REST payloads. No network (EQUITIES_SLEEVE.md §1.6)."""

from __future__ import annotations

from typing import Any

import httpx
import pyarrow as pa
import pytest

from alphaforge.core.errors import ConfigError, SchemaError
from alphaforge.core.symbols import EQUITY_MIC, SymbolMapper
from alphaforge.core.time import Timeframe, parse_utc
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.polygon_source import (
    CA_PUBLICATION_LAG_MS,
    CORPORATE_ACTIONS_SCHEMA,
    Adjustment,
    PolygonEquitiesSource,
    PolygonTransientError,
)

D = Timeframe.D1.ms
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z (a session-date open, 00:00 UTC)
AS_OF = 1_717_200_000_000  # 2024-06-01T00:00:00Z

# All equity ids use the canonical synthetic venue EQUITY_MIC ("XUSE") — the seeder ignores
# the real primary_exchange so its ids match the flat-files bars (which carry no exchange).
AAPL_ID = SymbolMapper.equity_instrument_id(EQUITY_MIC, "AAPL")  # XUSE:CASH:AAPLUSD
BRKB_ID = SymbolMapper.equity_instrument_id(EQUITY_MIC, "BRKB")  # XUSE:CASH:BRKBUSD
LEHMQ_ID = SymbolMapper.equity_instrument_id(EQUITY_MIC, "LEHMQ")  # delisted name

LEHMAN_LIST_DATE = "1994-09-01"
LEHMAN_DELIST = "2008-09-17T00:00:00Z"  # Lehman delisting (the equity FTT analogue)


# --------------------------------------------------------------------- fixtures


def agg(ts: int, price: float, *, volume: float = 1_000.0, n: int = 500) -> dict[str, Any]:
    """One Polygon /v2/aggs result bar (keys o/h/l/c/v/vw/n/t)."""
    return {
        "t": ts,
        "o": price,
        "h": price + 1.0,
        "l": price - 1.0,
        "c": price + 0.5,
        "v": volume,
        "vw": price + 0.25,
        "n": n,
    }


def agg_price(i: int) -> float:
    """Deterministic per-index price so pagination stitching is exactly checkable."""
    return 150.0 + (i % 311) * 0.01


def make_aggs(n: int, t0: int = T0) -> list[dict[str, Any]]:
    return [agg(t0 + i * D, agg_price(i)) for i in range(n)]


def ticker_row(
    ticker: str,
    *,
    ttype: str = "CS",
    primary_exchange: str = "XNAS",
    list_date: str | None = "2010-01-04",
    delisted_utc: str | None = None,
) -> dict[str, Any]:
    """One /v3/reference/tickers list row + its details payload (same shape here)."""
    row: dict[str, Any] = {
        "ticker": ticker,
        "type": ttype,
        "market": "stocks",
        "primary_exchange": primary_exchange,
        "list_date": list_date,
    }
    if delisted_utc is not None:
        row["delisted_utc"] = delisted_utc
    return row


def split_row(
    execution_date: str, *, split_from: int, split_to: int, declaration_date: str | None = None
) -> dict[str, Any]:
    """One /v3/reference/splits result row."""
    row: dict[str, Any] = {
        "execution_date": execution_date,
        "split_from": split_from,
        "split_to": split_to,
        "ticker": "AAPL",
    }
    if declaration_date is not None:
        row["declaration_date"] = declaration_date
    return row


def dividend_row(
    ex_dividend_date: str, *, cash_amount: float, declaration_date: str | None = None
) -> dict[str, Any]:
    """One /v3/reference/dividends result row."""
    row: dict[str, Any] = {
        "ex_dividend_date": ex_dividend_date,
        "cash_amount": cash_amount,
        "ticker": "AAPL",
    }
    if declaration_date is not None:
        row["declaration_date"] = declaration_date
    return row


class FakeClient:
    """Canned-payload stand-in for the Polygon REST endpoints.

    Records every call in ``self.calls`` as ``(method, kwargs)``. ``fail_once`` maps a
    method name to an exception raised on its FIRST invocation only (exercises the
    transient-retry path). Tickers split into active/inactive sets; aggregates/splits/
    dividends are keyed by vendor ticker. Cursor pagination is simulated when a per-method
    page list is supplied.
    """

    def __init__(
        self,
        *,
        active_tickers: list[dict[str, Any]] | None = None,
        inactive_tickers: list[dict[str, Any]] | None = None,
        aggs: dict[str, list[dict[str, Any]]] | None = None,
        splits: dict[str, list[dict[str, Any]]] | None = None,
        dividends: dict[str, list[dict[str, Any]]] | None = None,
        ticker_pages: list[list[dict[str, Any]]] | None = None,
        fail_once: dict[str, Exception] | None = None,
    ) -> None:
        self._active = active_tickers if active_tickers is not None else [ticker_row("AAPL")]
        self._inactive = inactive_tickers if inactive_tickers is not None else []
        self._aggs = aggs or {}
        self._splits = splits or {}
        self._dividends = dividends or {}
        self._ticker_pages = ticker_pages
        self._fail_once = dict(fail_once or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, dict(kwargs)))
        exc = self._fail_once.pop(method, None)
        if exc is not None:
            raise exc

    def n_calls(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    def get_aggs(
        self,
        ticker: str,
        *,
        multiplier: int,
        timespan: str,
        from_ms: int,
        to_ms: int,
        adjusted: bool,
        limit: int,
    ) -> dict[str, Any]:
        self._record(
            "get_aggs",
            {"ticker": ticker, "from_ms": from_ms, "to_ms": to_ms, "adjusted": adjusted},
        )
        rows = [a for a in self._aggs.get(ticker, []) if int(a["t"]) >= from_ms]
        return {"results": rows[:limit]}

    def list_tickers(
        self,
        *,
        market: str,
        active: bool,
        date: str | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        self._record("list_tickers", {"market": market, "active": active, "cursor": cursor})
        if self._ticker_pages is not None and active:
            # Cursor-pagination simulation over the supplied pages (active set only).
            idx = 0 if cursor is None else int(cursor)
            page = self._ticker_pages[idx]
            body: dict[str, Any] = {"results": page}
            if idx + 1 < len(self._ticker_pages):
                body["next_url"] = f"https://api.polygon.io/v3/reference/tickers?cursor={idx + 1}"
            return body
        return {"results": self._active if active else self._inactive}

    def get_ticker_details(self, ticker: str, *, date: str | None) -> dict[str, Any]:
        self._record("get_ticker_details", {"ticker": ticker})
        for row in (*self._active, *self._inactive):
            if row.get("ticker") == ticker:
                return {"results": dict(row)}
        return {"results": {}}

    def list_splits(
        self,
        *,
        ticker: str | None,
        execution_date_lte: str | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        self._record("list_splits", {"ticker": ticker, "cursor": cursor})
        return {"results": list(self._splits.get(str(ticker), []))}

    def list_dividends(
        self,
        *,
        ticker: str | None,
        ex_date_lte: str | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        self._record("list_dividends", {"ticker": ticker, "cursor": cursor})
        return {"results": list(self._dividends.get(str(ticker), []))}


class SpyBudget:
    """Duck-typed RateBudget recording every acquired weight (never blocks)."""

    def __init__(self) -> None:
        self.acquired: list[int] = []

    def acquire(self, weight: int) -> None:
        self.acquired.append(weight)


def make_source(
    client: FakeClient,
    budget: SpyBudget | None = None,
    *,
    enrich_details: bool = False,
) -> PolygonEquitiesSource:
    return PolygonEquitiesSource(
        client=client,  # type: ignore[arg-type]
        rate_budget=budget,
        enrich_details=enrich_details,
    )


def ts_open_ints(tbl: pa.Table) -> list[int]:
    values: list[int] = tbl.column("ts_open").cast(pa.int64()).to_pylist()
    return values


# ----------------------------------------------------------------- construction


class TestConstruction:
    def test_is_a_datasource_with_stable_name(self) -> None:
        src = make_source(FakeClient())
        assert isinstance(src, DataSource)
        assert src.name == "polygon.equities"

    def test_requires_key_or_client(self) -> None:
        with pytest.raises(ConfigError, match="api_key"):
            PolygonEquitiesSource()

    def test_api_key_without_client_constructs_without_io(self) -> None:
        # Construction must perform no network I/O even with a real key + no fake client.
        src = PolygonEquitiesSource(api_key="test-key")
        assert src.name == "polygon.equities"


# ------------------------------------------------------------- list_instruments


class TestListInstruments:
    def test_filters_to_common_equity_types_only(self) -> None:
        active = [
            ticker_row("AAPL", ttype="CS"),
            ticker_row("SPY", ttype="ETF"),  # excluded
            ticker_row("TSM", ttype="ADRC"),  # ADR common: kept
            ticker_row("WARR", ttype="WARRANT"),  # excluded
        ]
        src = make_source(FakeClient(active_tickers=active))
        ids = [inst.instrument_id for inst in src.list_instruments(as_of=AS_OF)]
        assert ids == [
            SymbolMapper.equity_instrument_id(EQUITY_MIC, "AAPL"),
            SymbolMapper.equity_instrument_id(EQUITY_MIC, "TSM"),
        ]

    def test_equity_instrument_fields(self) -> None:
        src = make_source(FakeClient(active_tickers=[ticker_row("AAPL", list_date="2010-01-04")]))
        (inst,) = src.list_instruments(as_of=AS_OF)
        assert inst.asset_class is AssetClass.EQUITY
        assert inst.market_type is MarketType.CASH
        assert (inst.base, inst.quote) == ("AAPL", "USD")
        assert inst.tick_size == 0.01
        assert inst.lot_size == 1.0
        assert inst.min_qty == 1.0
        assert inst.funding_interval_hours is None
        assert inst.can_short is True
        assert inst.listed_ts == parse_utc("2010-01-04T00:00:00+00:00")
        assert inst.delisted_ts is None

    def test_dotted_class_ticker_canonicalizes_and_keeps_vendor_ticker(self) -> None:
        src = make_source(FakeClient(active_tickers=[ticker_row("BRK.B", primary_exchange="XNYS")]))
        (inst,) = src.list_instruments(as_of=AS_OF)
        assert inst.instrument_id == BRKB_ID  # dots stripped in the canonical id
        assert (inst.base, inst.quote) == ("BRKB", "USD")
        # The raw vendor ticker (BRK.B) must drive the fetch round-trip, not "BRKB".
        assert src._resolve_ticker(inst.instrument_id) == "BRK.B"

    def test_list_instruments_includes_delisted(self) -> None:
        active = [ticker_row("AAPL")]
        inactive = [
            ticker_row(
                "LEHMQ",
                primary_exchange="XNYS",
                list_date=LEHMAN_LIST_DATE,
                delisted_utc=LEHMAN_DELIST,
            )
        ]
        src = make_source(FakeClient(active_tickers=active, inactive_tickers=inactive))
        by_id = {inst.instrument_id: inst for inst in src.list_instruments(as_of=AS_OF)}
        # Both pages present, sorted by instrument_id, active + inactive.
        assert set(by_id) == {AAPL_ID, LEHMQ_ID}
        assert list(by_id) == sorted(by_id)

    def test_delisted_ticker_survives(self) -> None:
        # The equity FTT/LUNA analogue: a delisted name carries the right delisted_ts so
        # the SCD2 store / universe builder can close its interval at exactly that instant.
        inactive = [
            ticker_row(
                "LEHMQ",
                primary_exchange="XNYS",
                list_date=LEHMAN_LIST_DATE,
                delisted_utc=LEHMAN_DELIST,
            )
        ]
        src = make_source(FakeClient(active_tickers=[], inactive_tickers=inactive))
        (lehmq,) = src.list_instruments(as_of=AS_OF)
        assert lehmq.instrument_id == LEHMQ_ID
        assert lehmq.listed_ts == parse_utc("1994-09-01T00:00:00+00:00")
        assert lehmq.delisted_ts == parse_utc(LEHMAN_DELIST)
        assert lehmq.delisted_ts is not None and lehmq.delisted_ts < AS_OF

    def test_delisted_without_usable_date_falls_back_to_as_of(self) -> None:
        # A venue delist timestamp <= listed_ts is implausible -> fall back to observation.
        inactive = [
            ticker_row(
                "ZZZ",
                primary_exchange="XNYS",
                list_date="2010-01-04",
                delisted_utc="2000-01-01T00:00:00Z",  # before list_date: implausible
            )
        ]
        src = make_source(FakeClient(active_tickers=[], inactive_tickers=inactive))
        (zzz,) = src.list_instruments(as_of=AS_OF)
        assert zzz.delisted_ts == AS_OF

    def test_relisted_ticker_dedup_keeps_inactive_observation(self) -> None:
        # A name on both pages (re-list / data overlap) de-duplicates on instrument_id;
        # inactive is walked second, so the delisted record wins (most recent observation).
        delist = "2023-05-10T00:00:00Z"  # after the default list_date (2010-01-04)
        active = [ticker_row("AAPL")]
        inactive = [ticker_row("AAPL", delisted_utc=delist)]
        src = make_source(FakeClient(active_tickers=active, inactive_tickers=inactive))
        (inst,) = src.list_instruments(as_of=AS_OF)
        assert inst.delisted_ts == parse_utc(delist)

    def test_pagination_stitches(self) -> None:
        pages = [
            [ticker_row("AAA"), ticker_row("BBB")],
            [ticker_row("CCC")],
        ]
        src = make_source(FakeClient(ticker_pages=pages, inactive_tickers=[]))
        ids = [inst.instrument_id for inst in src.list_instruments(as_of=AS_OF)]
        assert ids == [
            SymbolMapper.equity_instrument_id(EQUITY_MIC, t) for t in ("AAA", "BBB", "CCC")
        ]

    def test_rate_budget_charged_per_request_default_no_enrichment(self) -> None:
        budget = SpyBudget()
        active = [ticker_row("AAPL"), ticker_row("MSFT")]
        src = make_source(FakeClient(active_tickers=active, inactive_tickers=[]), budget)
        src.list_instruments(as_of=AS_OF)
        # DEFAULT (enrich_details=False): 1 active list page + 1 inactive list page = 2
        # requests. NO per-ticker detail calls — the listing row carries the spine fields,
        # so the full active+delisted universe seeds in ~page-count calls, not ~N calls.
        assert budget.acquired == [1, 1]

    def test_rate_budget_charged_per_request_with_enrichment(self) -> None:
        budget = SpyBudget()
        active = [ticker_row("AAPL"), ticker_row("MSFT")]
        src = make_source(
            FakeClient(active_tickers=active, inactive_tickers=[]),
            budget,
            enrich_details=True,
        )
        src.list_instruments(as_of=AS_OF)
        # enrich_details=True: 1 active page + 2 detail calls + 1 inactive page = 4 requests.
        assert budget.acquired == [1, 1, 1, 1]

    def test_default_seed_issues_no_per_ticker_detail_calls(self) -> None:
        client = FakeClient(active_tickers=[ticker_row("AAPL")], inactive_tickers=[])
        make_source(client).list_instruments(as_of=AS_OF)
        assert client.n_calls("get_ticker_details") == 0

    def test_enrichment_issues_one_detail_call_per_ticker(self) -> None:
        client = FakeClient(
            active_tickers=[ticker_row("AAPL"), ticker_row("MSFT")], inactive_tickers=[]
        )
        make_source(client, enrich_details=True).list_instruments(as_of=AS_OF)
        assert client.n_calls("get_ticker_details") == 2


# ------------------------------------------------------------------ fetch_ohlcv


class TestFetchOhlcv:
    def _src_with_aggs(self, n: int) -> tuple[PolygonEquitiesSource, FakeClient]:
        client = FakeClient(aggs={"AAPL": make_aggs(n)})
        return make_source(client), client

    def test_fetch_ohlcv_raw_unadjusted(self) -> None:
        n_closed = 10
        src, client = self._src_with_aggs(n_closed + 1)
        now = T0 + n_closed * D + D // 2  # bar n_closed is mid-flight (unclosed)
        tbl = src.fetch_ohlcv(
            AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + (n_closed + 5) * D, now=now
        )
        validate_table(tbl, Dataset.OHLCV)
        assert tbl.num_rows == n_closed  # partial last bar excluded
        assert ts_open_ints(tbl) == [T0 + i * D for i in range(n_closed)]
        # The request must be adjusted=false (raw-in-lake).
        assert all(kw["adjusted"] is False for m, kw in client.calls if m == "get_aggs")
        # Values map straight from the raw payload; quote_volume = vw * v (dollar volume).
        opens = tbl.column("open").to_pylist()
        quote_vol = tbl.column("quote_volume").to_pylist()
        n_trades = tbl.column("n_trades").to_pylist()
        for i in (0, 5, n_closed - 1):
            assert opens[i] == pytest.approx(agg_price(i))
            assert quote_vol[i] == pytest.approx((agg_price(i) + 0.25) * 1_000.0)
            assert n_trades[i] == 500

    def test_split_only_adjustment_requests_adjusted_true(self) -> None:
        client = FakeClient(aggs={"AAPL": make_aggs(3)})
        src = PolygonEquitiesSource(client=client, adjustment=Adjustment.SPLIT_ONLY)
        src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 5 * D, now=T0 + 9 * D)
        assert all(kw["adjusted"] is True for m, kw in client.calls if m == "get_aggs")

    def test_fetch_ohlcv_rejects_intraday_v1(self) -> None:
        src, _ = self._src_with_aggs(3)
        with pytest.raises(NotImplementedError, match="daily"):
            src.fetch_ohlcv(AAPL_ID, tf=Timeframe.H1, since=T0, until=T0 + D)

    def test_partial_bar_dropped_via_now(self) -> None:
        src, _ = self._src_with_aggs(3)
        # Bar 2 ([T0+2D, T0+3D)) has not closed at now = T0 + 2.5D.
        tbl = src.fetch_ohlcv(
            AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 3 * D, now=T0 + 2 * D + D // 2
        )
        assert ts_open_ints(tbl) == [T0, T0 + D]

    def test_bar_visible_exactly_at_close(self) -> None:
        src, _ = self._src_with_aggs(2)
        tbl = src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 2 * D, now=T0 + 2 * D)
        assert ts_open_ints(tbl) == [T0, T0 + D]
        tbl2 = src.fetch_ohlcv(
            AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 2 * D, now=T0 + 2 * D - 1
        )
        assert ts_open_ints(tbl2) == [T0]

    def test_until_is_exclusive_on_ts_open(self) -> None:
        src, _ = self._src_with_aggs(20)
        tbl = src.fetch_ohlcv(
            AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 10 * D, now=T0 + 100 * D
        )
        assert ts_open_ints(tbl) == [T0 + i * D for i in range(10)]

    def test_empty_range_returns_empty_table_without_calls(self) -> None:
        src, client = self._src_with_aggs(5)
        tbl = src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0, now=T0 + 10 * D)
        validate_table(tbl, Dataset.OHLCV)
        assert tbl.num_rows == 0
        assert client.n_calls("get_aggs") == 0

    def test_until_lt_since_raises(self) -> None:
        src, _ = self._src_with_aggs(5)
        with pytest.raises(ValueError, match="until"):
            src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 - D, now=T0)

    def test_foreign_market_type_rejected(self) -> None:
        src, _ = self._src_with_aggs(5)
        with pytest.raises(ValueError, match="CASH"):
            src.fetch_ohlcv("BINANCE:PERP:BTCUSDT", tf=Timeframe.D1, since=T0, until=T0 + D)

    def test_fetch_ohlcv_non_monotonic_raises(self) -> None:
        rows = [agg(T0, 150.0), agg(T0, 151.0)]  # duplicate open_time
        client = FakeClient(aggs={"AAPL": rows})
        src = make_source(client)
        with pytest.raises(SchemaError, match="strictly increasing"):
            src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 5 * D, now=T0 + 9 * D)

    def test_retries_transient_then_succeeds(self) -> None:
        client = FakeClient(
            aggs={"AAPL": make_aggs(5)},
            fail_once={"get_aggs": PolygonTransientError("429")},
        )
        src = make_source(client)
        tbl = src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 5 * D, now=T0 + 9 * D)
        assert tbl.num_rows == 5
        assert client.n_calls("get_aggs") == 2  # failed attempt + retry

    def test_permanent_error_propagates_unretried(self) -> None:
        request = httpx.Request("GET", "https://api.polygon.io/v2/aggs")
        response = httpx.Response(404, request=request)
        err = httpx.HTTPStatusError("404", request=request, response=response)
        client = FakeClient(aggs={"AAPL": make_aggs(5)}, fail_once={"get_aggs": err})
        src = make_source(client)
        with pytest.raises(httpx.HTTPStatusError):
            src.fetch_ohlcv(AAPL_ID, tf=Timeframe.D1, since=T0, until=T0 + 5 * D, now=T0 + 9 * D)
        assert client.n_calls("get_aggs") == 1


# ----------------------------------------------------------------- fetch_funding


class TestFetchFunding:
    def test_fetch_funding_not_supported(self) -> None:
        src = make_source(FakeClient())
        with pytest.raises(NotImplementedError, match="corporate_actions"):
            src.fetch_funding(AAPL_ID, since=T0, until=T0 + D)


# ---------------------------------------------------------- fetch_corporate_actions


class TestCorporateActions:
    def test_split_and_dividend_rows_pit_available_at(self) -> None:
        splits = {"AAPL": [split_row("2024-02-15", split_from=1, split_to=4)]}
        dividends = {
            "AAPL": [dividend_row("2024-02-09", cash_amount=0.24, declaration_date="2024-02-01")]
        }
        src = make_source(FakeClient(splits=splits, dividends=dividends))
        tbl = src.fetch_corporate_actions(AAPL_ID, since=T0, until=T0 + 365 * D)
        _assert_ca_schema(tbl)
        rows = tbl.to_pylist()
        assert len(rows) == 2
        # Sorted by (ex_date, action_type): dividend 02-09 before split 02-15.
        div, split = rows
        assert div["action_type"] == "dividend"
        assert div["ratio"] == 1.0
        assert div["cash_amount"] == pytest.approx(0.24)
        # Dividend knowable on its declaration_date + 1d publication lag.
        div_knowable = parse_utc("2024-02-01T00:00:00+00:00")
        assert _ms(div["available_at"]) == div_knowable + CA_PUBLICATION_LAG_MS
        assert split["action_type"] == "split"
        assert split["ratio"] == pytest.approx(4.0)  # split_to/split_from = 4/1
        assert split["cash_amount"] is None
        # No declaration_date on the split -> knowable = execution_date + lag.
        split_knowable = parse_utc("2024-02-15T00:00:00+00:00")
        assert _ms(split["available_at"]) == split_knowable + CA_PUBLICATION_LAG_MS

    def test_available_at_uses_declaration_date_when_present(self) -> None:
        # A split declared well before its ex-date must be knowable on the declaration,
        # not the ex-date — the corporate-actions analogue of funding finding 18.
        splits = {
            "AAPL": [
                split_row("2024-06-03", split_from=1, split_to=2, declaration_date="2024-05-01")
            ]
        }
        src = make_source(FakeClient(splits=splits))
        tbl = src.fetch_corporate_actions(AAPL_ID, since=T0, until=T0 + 365 * D)
        (row,) = tbl.to_pylist()
        assert _ms(row["available_at"]) == (
            parse_utc("2024-05-01T00:00:00+00:00") + CA_PUBLICATION_LAG_MS
        )
        assert _ms(row["available_at"]) < _ms(row["ex_date"])  # knowable before ex-date

    def test_ex_date_range_is_half_open(self) -> None:
        splits = {
            "AAPL": [
                split_row("2024-01-01", split_from=1, split_to=2),  # at since: included
                split_row("2024-01-11", split_from=1, split_to=2),  # at until: excluded
            ]
        }
        src = make_source(FakeClient(splits=splits))
        until = parse_utc("2024-01-11T00:00:00+00:00")
        tbl = src.fetch_corporate_actions(AAPL_ID, since=T0, until=until)
        ex = tbl.column("ex_date").cast(pa.int64()).to_pylist()
        assert ex == [parse_utc("2024-01-01T00:00:00+00:00")]

    def test_empty_range_returns_empty_table_without_calls(self) -> None:
        client = FakeClient(splits={"AAPL": [split_row("2024-02-15", split_from=1, split_to=2)]})
        src = make_source(client)
        tbl = src.fetch_corporate_actions(AAPL_ID, since=T0, until=T0)
        _assert_ca_schema(tbl)
        assert tbl.num_rows == 0
        assert client.n_calls("list_splits") == 0
        assert client.n_calls("list_dividends") == 0

    def test_until_lt_since_raises(self) -> None:
        src = make_source(FakeClient())
        with pytest.raises(ValueError, match="until"):
            src.fetch_corporate_actions(AAPL_ID, since=T0, until=T0 - 1)

    def test_degenerate_split_raises(self) -> None:
        splits = {"AAPL": [split_row("2024-02-15", split_from=0, split_to=2)]}
        src = make_source(FakeClient(splits=splits))
        with pytest.raises(SchemaError, match="split"):
            src.fetch_corporate_actions(AAPL_ID, since=T0, until=T0 + 365 * D)


def _ms(value: Any) -> int:
    """A pyarrow timestamp scalar (from to_pylist: a datetime) -> epoch ms."""
    from alphaforge.core.time import to_ms

    return to_ms(value)


def _assert_ca_schema(tbl: pa.Table) -> None:
    assert tbl.schema.names == CORPORATE_ACTIONS_SCHEMA.names
    for name in CORPORATE_ACTIONS_SCHEMA.names:
        assert tbl.schema.field(name).type == CORPORATE_ACTIONS_SCHEMA.field(name).type
