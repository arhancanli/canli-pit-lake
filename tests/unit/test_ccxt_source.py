"""Unit tests for alphaforge.data.sources.ccxt_source — CCXTDataSource against a
fake ccxt client with canned raw Binance payloads. No network."""

from __future__ import annotations

from typing import Any

import ccxt
import pyarrow as pa
import pytest

from alphaforge.core.errors import ConfigError, SchemaError
from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.ccxt_source import FUNDING_PUBLICATION_LAG_MS, CCXTDataSource

H = Timeframe.H1.ms
EIGHT_H = 8 * H
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z
AS_OF = 1_717_200_000_000  # 2024-06-01T00:00:00Z

BTC_ID = "BINANCE:PERP:BTCUSDT"
ETH_ID = "BINANCE:PERP:ETHUSDT"

BTC_ONBOARD = 1_569_398_400_000  # 2019-09-25T00:00:00Z
FTT_ONBOARD = 1_589_155_200_000  # 2020-05-11T00:00:00Z
FTT_DELIVERY = 1_668_038_400_000  # 2022-11-10T00:00:00Z (delisting cascade)


# --------------------------------------------------------------------- fixtures


def kline(ts: int, price: float, *, volume: float = 10.0, n_trades: int = 42) -> list[Any]:
    """One raw /fapi/v1/klines row: 12-element array, decimal strings for numbers.

    Index map: 0 open_time, 1-4 OHLC, 5 base volume, 6 close_time, 7 quote volume,
    8 n_trades, 9 taker buy base, 10 taker buy quote, 11 ignore.
    """
    return [
        ts,
        f"{price:.2f}",
        f"{price + 1.0:.2f}",
        f"{price - 1.0:.2f}",
        f"{price + 0.5:.2f}",
        f"{volume:.3f}",
        ts + H - 1,
        f"{volume * price:.2f}",
        n_trades,
        f"{volume / 2:.3f}",
        f"{volume * price / 2:.2f}",
        "0",
    ]


def kline_price(i: int) -> float:
    """Deterministic per-index price so pagination stitching is exactly checkable."""
    return 100.0 + (i % 997) * 0.01


def make_klines(n: int, t0: int = T0) -> list[list[Any]]:
    return [kline(t0 + i * H, kline_price(i)) for i in range(n)]


def funding_event(ts: int, rate: float) -> dict[str, Any]:
    """One raw /fapi/v1/fundingRate row."""
    return {
        "symbol": "BTCUSDT",
        "fundingTime": ts,
        "fundingRate": f"{rate:.8f}",
        "markPrice": "42000.00",
    }


def funding_rate_at(i: int) -> float:
    return 0.0001 * (1 + (i % 3))


def make_funding(n: int, t0: int = T0) -> list[dict[str, Any]]:
    return [funding_event(t0 + i * EIGHT_H, funding_rate_at(i)) for i in range(n)]


def perp_market(
    base: str,
    quote: str = "USDT",
    *,
    status: str = "TRADING",
    onboard: int = BTC_ONBOARD,
    delivery: int | None = None,
    with_filters: bool = True,
) -> dict[str, Any]:
    """A ccxt-unified market dict wrapping a Binance exchangeInfo-shaped ``info``."""
    raw_symbol = f"{base}{quote}"
    info: dict[str, Any] = {
        "symbol": raw_symbol,
        "status": status,
        "onboardDate": str(onboard),
        "contractType": "PERPETUAL",
    }
    if delivery is not None:
        info["deliveryDate"] = str(delivery)
    if with_filters:
        info["filters"] = [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "556.80"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "notional": "100"},
        ]
    return {
        "id": raw_symbol,
        "symbol": f"{base}/{quote}:{quote}",
        "base": base,
        "quote": quote,
        "settle": quote,
        "type": "swap",
        "spot": False,
        "swap": True,
        "future": False,
        "linear": True,
        "inverse": False,
        "active": status == "TRADING",
        "precision": {"price": 0.01, "amount": 0.1},
        "limits": {"amount": {"min": 0.1, "max": None}, "cost": {"min": 5.0, "max": None}},
        "info": info,
    }


def dated_future_market() -> dict[str, Any]:
    """A Binance quarterly future (swap=False) — must be skipped by listing."""
    market = perp_market("BTC")
    market["symbol"] = "BTC/USDT:USDT-240628"
    market["id"] = "BTCUSDT_240628"
    market["swap"] = False
    market["future"] = True
    market["type"] = "future"
    market["info"]["symbol"] = "BTCUSDT_240628"
    market["info"]["contractType"] = "CURRENT_QUARTER"
    return market


def default_markets() -> dict[str, dict[str, Any]]:
    return {
        "BTC/USDT:USDT": perp_market("BTC"),
        "ETH/USDT:USDT": perp_market("ETH"),
        # SETTLED (delisted) USDT perp: must be INCLUDED, with delisted_ts set.
        "FTT/USDT:USDT": perp_market(
            "FTT", status="SETTLED", onboard=FTT_ONBOARD, delivery=FTT_DELIVERY
        ),
        # No raw filters: exercises the unified precision/limits fallback.
        "ABC/USDT:USDT": perp_market("ABC", with_filters=False),
        # Dated future and USDC-margined pair: both must be SKIPPED.
        "BTC/USDT:USDT-240628": dated_future_market(),
        "ETH/USDC:USDC": perp_market("ETH", quote="USDC"),
    }


DEFAULT_FUNDING_INFO = [
    {"symbol": "ETHUSDT", "fundingIntervalHours": 4, "adjustedFundingRateCap": "0.03"}
]


class FakeClient:
    """Canned-payload stand-in for the raw ccxt.binanceusdm endpoints.

    Records every call in ``self.calls`` as ``(method, params)``. ``fail_once``
    maps a method name to an exception raised on its FIRST invocation only —
    exercising the transient-retry path.
    """

    def __init__(
        self,
        *,
        markets: dict[str, dict[str, Any]] | None = None,
        klines: dict[str, list[list[Any]]] | None = None,
        funding: dict[str, list[dict[str, Any]]] | None = None,
        funding_info: list[dict[str, Any]] | None = None,
        fail_once: dict[str, Exception] | None = None,
    ) -> None:
        self._markets = markets if markets is not None else default_markets()
        self._klines = klines or {}
        self._funding = funding or {}
        self._funding_info = funding_info if funding_info is not None else DEFAULT_FUNDING_INFO
        self._fail_once = dict(fail_once or {})
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def _record(self, method: str, params: dict[str, Any] | None) -> None:
        self.calls.append((method, dict(params) if params is not None else None))
        exc = self._fail_once.pop(method, None)
        if exc is not None:
            raise exc

    def n_calls(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    def load_markets(self) -> dict[str, dict[str, Any]]:
        self._record("load_markets", None)
        return self._markets

    def fapiPublicGetKlines(self, params: dict[str, Any]) -> list[list[Any]]:
        self._record("fapiPublicGetKlines", params)
        start, limit = int(params["startTime"]), int(params["limit"])
        rows = [k for k in self._klines.get(str(params["symbol"]), []) if int(k[0]) >= start]
        return [list(k) for k in rows[:limit]]

    def fapiPublicGetFundingRate(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._record("fapiPublicGetFundingRate", params)
        start, limit = int(params["startTime"]), int(params["limit"])
        rows = [
            r
            for r in self._funding.get(str(params["symbol"]), [])
            if int(r["fundingTime"]) >= start
        ]
        return [dict(r) for r in rows[:limit]]

    def fapiPublicGetFundingInfo(self) -> list[dict[str, Any]]:
        self._record("fapiPublicGetFundingInfo", None)
        return [dict(r) for r in self._funding_info]


class SpyBudget:
    """Duck-typed RateBudget recording every acquired weight (never blocks)."""

    def __init__(self) -> None:
        self.acquired: list[int] = []

    def acquire(self, weight: int) -> None:
        self.acquired.append(weight)


def make_source(
    client: FakeClient, budget: SpyBudget | None = None
) -> tuple[CCXTDataSource, FakeClient]:
    src = CCXTDataSource("binanceusdm", rate_budget=budget, client=client)  # type: ignore[arg-type]
    return src, client


def ts_open_ints(tbl: pa.Table) -> list[int]:
    return tbl.column("ts_open").cast(pa.int64()).to_pylist()


# ----------------------------------------------------------------- construction


class TestConstruction:
    def test_is_a_datasource_with_stable_name(self) -> None:
        src, _ = make_source(FakeClient())
        assert isinstance(src, DataSource)
        assert src.name == "ccxt.binanceusdm"

    def test_unknown_exchange_id_without_client_raises(self) -> None:
        with pytest.raises(ConfigError, match="nosuchexchange"):
            CCXTDataSource("nosuchexchange")


# ------------------------------------------------------------- list_instruments


class TestListInstruments:
    def test_filters_to_usdt_margined_perps_only(self) -> None:
        src, _ = make_source(FakeClient())
        ids = [inst.instrument_id for inst in src.list_instruments(as_of=AS_OF)]
        # Dated future and USDC pair skipped; SETTLED USDT perp kept; sorted.
        assert ids == [
            "BINANCE:PERP:ABCUSDT",
            "BINANCE:PERP:BTCUSDT",
            "BINANCE:PERP:ETHUSDT",
            "BINANCE:PERP:FTTUSDT",
        ]

    def test_trading_market_fields_from_raw_filters(self) -> None:
        src, _ = make_source(FakeClient())
        by_id = {inst.instrument_id: inst for inst in src.list_instruments(as_of=AS_OF)}
        btc = by_id[BTC_ID]
        assert btc.asset_class is AssetClass.CRYPTO_PERP
        assert btc.market_type is MarketType.PERP
        assert (btc.base, btc.quote) == ("BTC", "USDT")
        assert btc.tick_size == 0.10  # PRICE_FILTER tickSize
        assert btc.lot_size == 0.001  # LOT_SIZE stepSize
        assert btc.min_qty == 0.001  # LOT_SIZE minQty
        assert btc.min_notional == 100.0  # MIN_NOTIONAL notional
        assert (btc.maker_fee_bps, btc.taker_fee_bps) == (2.0, 5.0)
        assert btc.can_short is True
        assert btc.contract_multiplier == 1.0
        assert btc.listed_ts == BTC_ONBOARD
        assert btc.delisted_ts is None

    def test_unified_fallback_when_raw_filters_absent(self) -> None:
        src, _ = make_source(FakeClient())
        by_id = {inst.instrument_id: inst for inst in src.list_instruments(as_of=AS_OF)}
        abc = by_id["BINANCE:PERP:ABCUSDT"]
        assert abc.tick_size == 0.01  # precision.price (tick-size mode)
        assert abc.lot_size == 0.1  # precision.amount
        assert abc.min_qty == 0.1  # limits.amount.min
        assert abc.min_notional == 5.0  # limits.cost.min

    def test_settled_market_included_with_delisted_ts(self) -> None:
        src, _ = make_source(FakeClient())
        by_id = {inst.instrument_id: inst for inst in src.list_instruments(as_of=AS_OF)}
        ftt = by_id["BINANCE:PERP:FTTUSDT"]
        assert ftt.listed_ts == FTT_ONBOARD
        assert ftt.delisted_ts == FTT_DELIVERY  # venue deliveryDate, plausible past

    def test_settled_market_without_delivery_date_falls_back_to_as_of(self) -> None:
        markets = {"XYZ/USDT:USDT": perp_market("XYZ", status="SETTLED", onboard=FTT_ONBOARD)}
        src, _ = make_source(FakeClient(markets=markets, funding_info=[]))
        (xyz,) = src.list_instruments(as_of=AS_OF)
        assert xyz.delisted_ts == AS_OF

    def test_funding_interval_overlay_from_funding_info(self) -> None:
        src, _ = make_source(FakeClient())
        by_id = {inst.instrument_id: inst for inst in src.list_instruments(as_of=AS_OF)}
        assert by_id[ETH_ID].funding_interval_hours == 4  # listed in fundingInfo
        assert by_id[BTC_ID].funding_interval_hours == 8  # absent -> default

    def test_funding_info_fetched_once_per_instance(self) -> None:
        src, client = make_source(FakeClient())
        src.list_instruments(as_of=AS_OF)
        src.list_instruments(as_of=AS_OF)
        assert client.n_calls("fapiPublicGetFundingInfo") == 1
        assert client.n_calls("load_markets") == 2

    def test_rate_budget_charged_for_markets_and_funding_info(self) -> None:
        budget = SpyBudget()
        src, _ = make_source(FakeClient(), budget)
        src.list_instruments(as_of=AS_OF)
        assert budget.acquired == [1, 1]  # exchangeInfo + fundingInfo


# ------------------------------------------------------------------ fetch_ohlcv


class TestFetchOhlcv:
    def test_three_page_pagination_stitches_exactly(self) -> None:
        n_closed = 3000
        client = FakeClient(klines={"BTCUSDT": make_klines(n_closed + 1)})
        budget = SpyBudget()
        src, _ = make_source(client, budget)
        now = T0 + n_closed * H + 1_800_000  # bar 3000 is mid-flight (unclosed)

        tbl = src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 3005 * H, now=now)

        validate_table(tbl, Dataset.OHLCV)
        assert tbl.num_rows == n_closed  # partial bar 3000 excluded
        assert ts_open_ints(tbl) == [T0 + i * H for i in range(n_closed)]

        # Exactly three pages: 1500 + 1500 + 1(short), each charged weight 10.
        kline_calls = [p for m, p in client.calls if m == "fapiPublicGetKlines"]
        assert [c["startTime"] for c in kline_calls if c] == [T0, T0 + 1500 * H, T0 + 3000 * H]
        assert all(c["limit"] == 1500 and c["interval"] == "1h" for c in kline_calls if c)
        assert budget.acquired == [10, 10, 10]

        # Values stitched from the right rows, parsed from decimal strings.
        opens = tbl.column("open").to_pylist()
        volumes = tbl.column("volume").to_pylist()
        quote_volumes = tbl.column("quote_volume").to_pylist()
        n_trades = tbl.column("n_trades").to_pylist()
        for i in (0, 1499, 1500, 2999):  # page boundaries on both sides
            assert opens[i] == pytest.approx(kline_price(i))
            assert volumes[i] == pytest.approx(10.0)
            assert quote_volumes[i] == pytest.approx(10.0 * kline_price(i))
            assert n_trades[i] == 42

    def test_partial_bar_dropped_even_inside_range(self) -> None:
        client = FakeClient(klines={"BTCUSDT": make_klines(3)})
        src, _ = make_source(client)
        # Bar 2 ([T0+2H, T0+3H)) has not closed at now = T0 + 2.5H.
        tbl = src.fetch_ohlcv(
            BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 3 * H, now=T0 + 2 * H + 1_800_000
        )
        assert ts_open_ints(tbl) == [T0, T0 + H]

    def test_bar_visible_exactly_at_close(self) -> None:
        client = FakeClient(klines={"BTCUSDT": make_klines(2)})
        src, _ = make_source(client)
        # now == close of bar 1: bar 1 is available at exactly ts_open + H.
        tbl = src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 2 * H, now=T0 + 2 * H)
        assert ts_open_ints(tbl) == [T0, T0 + H]
        # One ms earlier it must be invisible.
        tbl2 = src.fetch_ohlcv(
            BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 2 * H, now=T0 + 2 * H - 1
        )
        assert ts_open_ints(tbl2) == [T0]

    def test_until_is_exclusive_on_ts_open(self) -> None:
        client = FakeClient(klines={"BTCUSDT": make_klines(20)})
        src, _ = make_source(client)
        tbl = src.fetch_ohlcv(
            BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 10 * H, now=T0 + 100 * H
        )
        assert ts_open_ints(tbl) == [T0 + i * H for i in range(10)]  # bar at until excluded

    def test_empty_range_returns_empty_table_without_calls(self) -> None:
        client = FakeClient(klines={"BTCUSDT": make_klines(5)})
        src, _ = make_source(client)
        tbl = src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0, now=T0 + 10 * H)
        validate_table(tbl, Dataset.OHLCV)
        assert tbl.num_rows == 0
        assert client.calls == []

    def test_inverted_range_raises(self) -> None:
        src, _ = make_source(FakeClient())
        with pytest.raises(ValueError, match="until"):
            src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 - H, now=T0)

    def test_spot_and_foreign_ids_rejected(self) -> None:
        src, _ = make_source(FakeClient())
        with pytest.raises(ValueError, match="PERP"):
            src.fetch_ohlcv("BINANCE:SPOT:BTCUSDT", tf=Timeframe.H1, since=T0, until=T0 + H)
        with pytest.raises(ValueError, match="BYBIT"):
            src.fetch_ohlcv("BYBIT:PERP:BTCUSDT", tf=Timeframe.H1, since=T0, until=T0 + H)

    def test_retries_transient_network_error_then_succeeds(self) -> None:
        client = FakeClient(
            klines={"BTCUSDT": make_klines(5)},
            fail_once={"fapiPublicGetKlines": ccxt.NetworkError("injected blip")},
        )
        budget = SpyBudget()
        src, _ = make_source(client, budget)
        tbl = src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 5 * H, now=T0 + 9 * H)
        assert tbl.num_rows == 5
        assert client.n_calls("fapiPublicGetKlines") == 2  # failed attempt + retry
        assert budget.acquired == [10]  # budget charged per page, not per attempt

    def test_permanent_error_propagates_unretried(self) -> None:
        client = FakeClient(
            klines={"BTCUSDT": make_klines(5)},
            fail_once={"fapiPublicGetKlines": ccxt.BadSymbol("no such symbol")},
        )
        src, _ = make_source(client)
        with pytest.raises(ccxt.BadSymbol):
            src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 5 * H, now=T0 + 9 * H)
        assert client.n_calls("fapiPublicGetKlines") == 1

    def test_non_monotonic_payload_raises_schema_error(self) -> None:
        rows = [kline(T0, 100.0), kline(T0, 101.0)]  # duplicate open_time
        client = FakeClient(klines={"BTCUSDT": rows})
        src, _ = make_source(client)
        with pytest.raises(SchemaError, match="strictly increasing"):
            src.fetch_ohlcv(BTC_ID, tf=Timeframe.H1, since=T0, until=T0 + 5 * H, now=T0 + 9 * H)


# ---------------------------------------------------------------- fetch_funding


class TestFetchFunding:
    def test_two_page_pagination_with_pit_lag(self) -> None:
        n = 1234
        client = FakeClient(funding={"BTCUSDT": make_funding(n)})
        budget = SpyBudget()
        src, _ = make_source(client, budget)

        tbl = src.fetch_funding(BTC_ID, since=T0, until=T0 + n * EIGHT_H)

        validate_table(tbl, Dataset.FUNDING)
        assert tbl.num_rows == n
        ts = tbl.column("ts_funding").cast(pa.int64()).to_pylist()
        assert ts == [T0 + i * EIGHT_H for i in range(n)]
        rates = tbl.column("rate").to_pylist()
        assert rates[:4] == pytest.approx([funding_rate_at(i) for i in range(4)])
        # available_at = ts_funding + 5 min publication lag, on EVERY row.
        avail = tbl.column("available_at").cast(pa.int64()).to_pylist()
        assert FUNDING_PUBLICATION_LAG_MS == 300_000
        assert all(a - t == FUNDING_PUBLICATION_LAG_MS for a, t in zip(avail, ts, strict=True))

        # Two pages (1000 + 234); the second resumes one ms past the last row.
        calls = [p for m, p in client.calls if m == "fapiPublicGetFundingRate"]
        assert [c["startTime"] for c in calls if c] == [T0, T0 + 999 * EIGHT_H + 1]
        assert all(c["limit"] == 1000 for c in calls if c)
        assert budget.acquired == [1, 1]

    def test_until_is_exclusive_on_ts_funding(self) -> None:
        client = FakeClient(funding={"BTCUSDT": make_funding(3)})
        src, _ = make_source(client)
        tbl = src.fetch_funding(BTC_ID, since=T0, until=T0 + 2 * EIGHT_H)
        ts = tbl.column("ts_funding").cast(pa.int64()).to_pylist()
        assert ts == [T0, T0 + EIGHT_H]  # row at until excluded; row at since included

    def test_empty_range_returns_empty_table_without_calls(self) -> None:
        client = FakeClient(funding={"BTCUSDT": make_funding(3)})
        src, _ = make_source(client)
        tbl = src.fetch_funding(BTC_ID, since=T0, until=T0)
        validate_table(tbl, Dataset.FUNDING)
        assert tbl.num_rows == 0
        assert client.calls == []

    def test_inverted_range_raises(self) -> None:
        src, _ = make_source(FakeClient())
        with pytest.raises(ValueError, match="until"):
            src.fetch_funding(BTC_ID, since=T0, until=T0 - 1)

    def test_spot_id_rejected(self) -> None:
        src, _ = make_source(FakeClient())
        with pytest.raises(ValueError, match="PERP"):
            src.fetch_funding("BINANCE:SPOT:BTCUSDT", since=T0, until=T0 + EIGHT_H)

    def test_non_monotonic_payload_raises_schema_error(self) -> None:
        rows = [funding_event(T0, 0.0001), funding_event(T0, 0.0002)]
        client = FakeClient(funding={"BTCUSDT": rows})
        src, _ = make_source(client)
        with pytest.raises(SchemaError, match="strictly increasing"):
            src.fetch_funding(BTC_ID, since=T0, until=T0 + 5 * EIGHT_H)

    def test_retries_transient_network_error_then_succeeds(self) -> None:
        client = FakeClient(
            funding={"BTCUSDT": make_funding(3)},
            fail_once={"fapiPublicGetFundingRate": ccxt.RequestTimeout("injected timeout")},
        )
        src, _ = make_source(client)
        tbl = src.fetch_funding(BTC_ID, since=T0, until=T0 + 3 * EIGHT_H)
        assert tbl.num_rows == 3
        assert client.n_calls("fapiPublicGetFundingRate") == 2
