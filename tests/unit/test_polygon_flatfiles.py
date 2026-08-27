"""Unit tests for alphaforge.data.sources.polygon_flatfiles.

PolygonFlatFilesSource against an in-memory fake S3 client over synthetic fixture
day-aggs CSVs. NO network, NO boto3, NO live lake (EQUITIES_INGEST.md §4.1).
"""

from __future__ import annotations

import gzip
import io
from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from alphaforge.core.errors import ConfigError, SchemaError
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Timeframe, to_ms
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.polygon_flatfiles import (
    DAY_AGGS_PREFIX,
    FLATFILES_MIC,
    PolygonFlatFilesSource,
    day_aggs_key,
    parse_day_csv,
)

D1 = Timeframe.D1.ms
_NS_PER_DAY = 86_400_000_000_000

# 2008-09-15: a real session date (the Lehman bankruptcy filing day) used by the
# survivorship regression in test_equities_ingest.py; reused here for date math.
DAY_2024_01_02 = date(2024, 1, 2)


# --------------------------------------------------------------------- fixtures


def _day_open_ms(d: date) -> int:
    return to_ms(datetime(d.year, d.month, d.day, tzinfo=UTC))


def _window_start_ns(d: date) -> int:
    """``window_start`` in ns at the session's 00:00 UTC open (as the real files carry it)."""
    return _day_open_ms(d) * 1_000_000


def csv_row(
    ticker: str,
    d: date,
    *,
    open_: float = 100.0,
    close: float = 101.0,
    high: float = 102.0,
    low: float = 99.0,
    volume: float = 1_000.0,
    transactions: int = 50,
    window_start_ns: int | None = None,
) -> dict[str, object]:
    """One synthetic day-aggs CSV row (column order is applied by make_day_csv)."""
    return {
        "ticker": ticker,
        "volume": volume,
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "window_start": _window_start_ns(d) if window_start_ns is None else window_start_ns,
        "transactions": transactions,
    }


_COLUMNS = ("ticker", "volume", "open", "close", "high", "low", "window_start", "transactions")


def make_day_csv(rows: list[dict[str, object]], *, header: bool = True) -> bytes:
    """Write the 8-column header + rows and gzip them (synthetic fixture; ns window_start)."""
    buf = io.StringIO()
    if header:
        buf.write(",".join(_COLUMNS) + "\n")
    for row in rows:
        buf.write(",".join(str(row[c]) for c in _COLUMNS) + "\n")
    return gzip.compress(buf.getvalue().encode("utf-8"))


def make_day_csv_raw(text: str) -> bytes:
    """Gzip an arbitrary CSV body verbatim (for fetch_day malformed-file tests)."""
    return gzip.compress(text.encode("utf-8"))


def plain_day_csv(rows: list[dict[str, object]], *, header: bool = True) -> bytes:
    """The DECOMPRESSED CSV body (what parse_day_csv consumes — fetch_day gunzips first)."""
    return gzip.decompress(make_day_csv(rows, header=header))


class FakeFlatFilesClient:
    """In-memory FlatFilesClientProtocol: a dict of {key: gzipped_csv_bytes}. No network."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self._objects: dict[str, bytes] = dict(objects or {})
        self.list_calls: list[str] = []
        self.get_calls: list[str] = []

    def put_day(self, d: date, rows: list[dict[str, object]]) -> None:
        self._objects[day_aggs_key(d)] = make_day_csv(rows)

    def put_raw(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def list_keys(self, *, prefix: str) -> list[str]:
        self.list_calls.append(prefix)
        return sorted(k for k in self._objects if k.startswith(prefix))

    def get_object_bytes(self, *, key: str) -> bytes:
        self.get_calls.append(key)
        try:
            return self._objects[key]
        except KeyError:
            raise KeyError(key) from None


def make_source(client: FakeFlatFilesClient) -> PolygonFlatFilesSource:
    return PolygonFlatFilesSource(client=client)


AAPL_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "AAPL")
BRKB_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "BRKB")
ETH_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "ETH")  # quote-token ticker (B4)


def col(tbl: pa.Table, name: str) -> list[object]:
    values: list[object] = tbl.column(name).to_pylist()
    return values


def ts_open_ints(tbl: pa.Table) -> list[int]:
    values: list[int] = tbl.column("ts_open").cast(pa.int64()).to_pylist()
    return values


# ----------------------------------------------------------------- key helpers


class TestDayAggsKey:
    def test_key_layout(self) -> None:
        assert day_aggs_key(date(2008, 9, 15)) == f"{DAY_AGGS_PREFIX}/2008/09/2008-09-15.csv.gz"
        assert day_aggs_key(date(2024, 1, 2)) == f"{DAY_AGGS_PREFIX}/2024/01/2024-01-02.csv.gz"


# ----------------------------------------------------------------- construction


class TestConstruction:
    def test_stable_name(self) -> None:
        assert make_source(FakeFlatFilesClient()).name == "polygon.flatfiles"

    def test_no_credentials_raises_configerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "POLYGON_S3_ENDPOINT",
            "POLYGON_S3_ACCESS_KEY_ID",
            "POLYGON_S3_SECRET_ACCESS_KEY",
            "POLYGON_S3_BUCKET",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigError, match="requires the S3 credentials"):
            PolygonFlatFilesSource()  # no client, no env -> ConfigError, never a network call


# ----------------------------------------------------------------- parse_day_csv


class TestParseDayCsv:
    def test_parses_typed_rows(self) -> None:
        data = plain_day_csv([csv_row("AAPL", DAY_2024_01_02, open_=187.0, close=185.0)])
        rows = parse_day_csv(data)
        assert len(rows) == 1
        r = rows[0]
        assert r.ticker == "AAPL"
        assert r.open == 187.0
        assert r.close == 185.0
        assert r.window_start_ns == _window_start_ns(DAY_2024_01_02)

    def test_empty_csv_raises(self) -> None:
        with pytest.raises(SchemaError, match="empty"):
            parse_day_csv(b"")

    def test_bad_header_raises(self) -> None:
        with pytest.raises(SchemaError, match="header"):
            parse_day_csv(b"ticker,close,window_start\nAAPL,1,2\n")

    def test_non_numeric_close_raises_naming_column(self) -> None:
        body = ",".join(_COLUMNS) + "\n" + "AAPL,1000,100.0,NOTANUMBER,102.0,99.0,1000000000,50\n"
        with pytest.raises(SchemaError, match=r"column 'close'"):
            parse_day_csv(body.encode("utf-8"))

    def test_empty_ticker_raises(self) -> None:
        body = ",".join(_COLUMNS) + "\n" + " ,1000,100.0,101.0,102.0,99.0,1000000000,50\n"
        with pytest.raises(SchemaError, match="empty ticker"):
            parse_day_csv(body.encode("utf-8"))


# ----------------------------------------------------------------- fetch_day


class TestFetchDay:
    def test_maps_schema(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(
            DAY_2024_01_02,
            [csv_row("AAPL", DAY_2024_01_02, open_=187.0, close=185.0, volume=2_000.0)],
        )
        src = make_source(client)
        tbl = src.fetch_day(DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 10 * D1)

        validate_table(tbl, Dataset.OHLCV_1D)
        assert col(tbl, "instrument_id") == [AAPL_ID]
        assert ts_open_ints(tbl) == [_day_open_ms(DAY_2024_01_02)]
        assert col(tbl, "open") == [187.0]
        assert col(tbl, "close") == [185.0]
        assert col(tbl, "volume") == [2_000.0]
        assert col(tbl, "quote_volume") == [185.0 * 2_000.0]  # close*volume fallback
        assert col(tbl, "n_trades") == [50]
        assert col(tbl, "quality_flags") == [0]

    def test_ts_open_is_day_floor_of_intraday_window_start(self) -> None:
        client = FakeFlatFilesClient()
        # window_start at 13:30 UTC (intraday) must still floor to the session open 00:00.
        intraday_ns = (_day_open_ms(DAY_2024_01_02) + 13 * 3_600_000 + 30 * 60_000) * 1_000_000
        client.put_day(
            DAY_2024_01_02, [csv_row("AAPL", DAY_2024_01_02, window_start_ns=intraday_ns)]
        )
        tbl = make_source(client).fetch_day(
            DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 5 * D1
        )
        assert ts_open_ints(tbl) == [_day_open_ms(DAY_2024_01_02)]

    def test_instrument_id_mapping(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(
            DAY_2024_01_02,
            [
                csv_row("AAPL", DAY_2024_01_02),
                csv_row("BRK.B", DAY_2024_01_02),  # dotted -> canonical BRKB
                csv_row("ETH", DAY_2024_01_02),  # quote-token ticker round-trips (B4)
            ],
        )
        tbl = make_source(client).fetch_day(
            DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 5 * D1
        )
        # sorted by instrument_id
        assert set(col(tbl, "instrument_id")) == {AAPL_ID, BRKB_ID, ETH_ID}
        # ETH round-trips back to ticker ETH (never mis-split by KNOWN_QUOTES)
        _, ticker = SymbolMapper.parse_equity_id(ETH_ID)
        assert ticker == "ETH"

    def test_closed_bar_guard_drops_in_progress_day(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(DAY_2024_01_02, [csv_row("AAPL", DAY_2024_01_02)])
        # now is BEFORE the bar's close (ts_open + D1): the bar is in-progress -> dropped.
        tbl = make_source(client).fetch_day(
            DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + D1 - 1
        )
        assert tbl.num_rows == 0
        validate_table(tbl, Dataset.OHLCV_1D)

    def test_closed_bar_kept_exactly_at_close(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(DAY_2024_01_02, [csv_row("AAPL", DAY_2024_01_02)])
        tbl = make_source(client).fetch_day(DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + D1)
        assert tbl.num_rows == 1  # a bar closing exactly at now is final

    def test_sorted_and_unique_on_id(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(
            DAY_2024_01_02,
            [csv_row(t, DAY_2024_01_02) for t in ("ZZZZ", "AAAA", "MMMM")],
        )
        tbl = make_source(client).fetch_day(
            DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 5 * D1
        )
        ids: list[str] = tbl.column("instrument_id").to_pylist()
        assert ids == sorted(ids)

    def test_missing_object_raises_keyerror(self) -> None:
        src = make_source(FakeFlatFilesClient())
        with pytest.raises(KeyError):
            src.fetch_day(DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 5 * D1)

    def test_malformed_cell_raises_schemaerror(self) -> None:
        client = FakeFlatFilesClient()
        body = ",".join(_COLUMNS) + "\n" + "AAPL,1000,100.0,oops,102.0,99.0,1000000000,50\n"
        client.put_raw(day_aggs_key(DAY_2024_01_02), make_day_csv_raw(body))
        with pytest.raises(SchemaError):
            make_source(client).fetch_day(DAY_2024_01_02, now=_day_open_ms(DAY_2024_01_02) + 5 * D1)


# ----------------------------------------------------------------- available_sessions


class TestAvailableSessions:
    def test_only_dates_with_objects_in_span(self) -> None:
        client = FakeFlatFilesClient()
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1)]
        for d in days:
            client.put_day(d, [csv_row("AAPL", d)])
        start = _day_open_ms(date(2024, 1, 1))
        until = _day_open_ms(date(2024, 1, 4))
        got = make_source(client).available_sessions(start=start, until=until)
        assert got == [date(2024, 1, 2), date(2024, 1, 3)]  # Feb excluded by span

    def test_non_session_days_absent(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(date(2024, 1, 2), [csv_row("AAPL", date(2024, 1, 2))])
        # Jan 1 (holiday) and Jan 6 (weekend) have no object -> not returned.
        got = make_source(client).available_sessions(
            start=_day_open_ms(date(2024, 1, 1)),
            until=_day_open_ms(date(2024, 1, 10)),
        )
        assert got == [date(2024, 1, 2)]

    def test_empty_span(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(date(2024, 1, 2), [csv_row("AAPL", date(2024, 1, 2))])
        t = _day_open_ms(date(2024, 1, 2))
        assert make_source(client).available_sessions(start=t, until=t) == []

    def test_lists_only_needed_month_prefixes(self) -> None:
        client = FakeFlatFilesClient()
        client.put_day(date(2024, 1, 2), [csv_row("AAPL", date(2024, 1, 2))])
        make_source(client).available_sessions(
            start=_day_open_ms(date(2024, 1, 1)),
            until=_day_open_ms(date(2024, 1, 31)),
        )
        # exactly one month prefix listed (2024/01)
        assert client.list_calls == [f"{DAY_AGGS_PREFIX}/2024/01"]

    def test_dedup_keys_across_listing(self) -> None:
        client = FakeFlatFilesClient()
        d = date(2024, 1, 2)
        client.put_day(d, [csv_row("AAPL", d)])
        got = make_source(client).available_sessions(
            start=_day_open_ms(date(2024, 1, 1)),
            until=_day_open_ms(date(2024, 1, 10)),
        )
        assert got == [d]
