"""Unit tests for the Binance Vision archive client (no network — httpx.MockTransport).

Covers leakage finding 1's machinery: paginated S3 symbol listing (with an
FTTUSDT-shaped delisted symbol), the underscore / non-USDT exclusion rules, month-range
parsing, kline files with and without header rows, millisecond AND microsecond
timestamps, the closed-bar guard, both funding CSV variants, the funding
``available_at`` lag, 404 → DataGapError, and transient-transport retry.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import httpx
import pytest

from alphaforge.core.errors import DataGapError, SchemaError
from alphaforge.data.ingest.retry import transient_retry
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.ccxt_source import FUNDING_PUBLICATION_LAG_MS
from alphaforge.data.sources.vision import (
    S3_LIST_ENDPOINT,
    BinanceVisionClient,
    YearMonth,
)

KLINES_PREFIX = "data/futures/um/monthly/klines/"

H1_MS = 3_600_000
JAN_2020 = 1_577_836_800_000  # 2020-01-01T00:00:00Z
JAN_2025 = 1_735_689_600_000  # 2025-01-01T00:00:00Z
FAR_FUTURE = 4_102_444_800_000  # 2100-01-01T00:00:00Z — "now" that closes every bar


# ---------------------------------------------------------------------- fixtures


def _listing_xml(
    *,
    request_prefix: str,
    prefixes: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
    truncated: bool = False,
    next_marker: str | None = None,
) -> str:
    """Minimal but faithful S3 ListBucketResult XML (namespaced, echoed Prefix)."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
        "<Name>data.binance.vision</Name>",
        f"<Prefix>{request_prefix}</Prefix>",
        "<MaxKeys>1000</MaxKeys><Delimiter>/</Delimiter>",
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
    ]
    if next_marker is not None:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts.extend(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    parts.extend(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>" for prefix in prefixes
    )
    parts.append("</ListBucketResult>")
    return "".join(parts)


def _zip_bytes(csv_text: str, member: str = "data.csv") -> bytes:
    """An in-memory zip with one CSV member, mimicking an archive object."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, csv_text)
    return buf.getvalue()


def _client(handler, **kwargs) -> BinanceVisionClient:
    transport = httpx.MockTransport(handler)
    return BinanceVisionClient(http=httpx.Client(transport=transport), **kwargs)


def _kline_row(ts: int, price: float = 100.0, *, volume: float = 10.0) -> str:
    close_time = ts + H1_MS - 1
    return (
        f"{ts},{price},{price + 1},{price - 1},{price + 0.5},{volume},"
        f"{close_time},{volume * price},42,{volume / 2},{volume * price / 2},0"
    )


# ---------------------------------------------------------------------- YearMonth


class TestYearMonth:
    def test_first_ms(self) -> None:
        assert YearMonth(2020, 1).first_ms() == JAN_2020
        assert YearMonth(2025, 1).first_ms() == JAN_2025

    def test_next_month_first_ms_mid_year(self) -> None:
        assert YearMonth(2020, 1).next_month_first_ms() == 1_580_515_200_000  # 2020-02-01

    def test_next_month_first_ms_december_rollover(self) -> None:
        assert YearMonth(2021, 12).next_month_first_ms() == YearMonth(2022, 1).first_ms()

    def test_orders_chronologically(self) -> None:
        assert YearMonth(2021, 12) < YearMonth(2022, 1) < YearMonth(2022, 2)


# ---------------------------------------------------------------------- listing


class TestListSymbols:
    def test_pagination_is_followed(self) -> None:
        """Two-page listing: page 2 must be requested with the page-1 marker."""
        page1_prefixes = tuple(f"{KLINES_PREFIX}AAA{i:03d}USDT/" for i in range(3))
        page2_prefixes = (f"{KLINES_PREFIX}FTTUSDT/", f"{KLINES_PREFIX}ZZZUSDT/")
        marker_value = page1_prefixes[-1]
        markers_seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            markers_seen.append(request.url.params.get("marker"))
            assert request.url.params["prefix"] == KLINES_PREFIX
            assert request.url.params["delimiter"] == "/"
            if request.url.params.get("marker") is None:
                xml = _listing_xml(
                    request_prefix=KLINES_PREFIX,
                    prefixes=page1_prefixes,
                    truncated=True,
                    next_marker=marker_value,
                )
            else:
                assert request.url.params["marker"] == marker_value
                xml = _listing_xml(request_prefix=KLINES_PREFIX, prefixes=page2_prefixes)
            return httpx.Response(200, text=xml)

        symbols = _client(handler).list_symbols()
        assert markers_seen == [None, marker_value]  # pagination actually followed
        assert symbols == ["AAA000USDT", "AAA001USDT", "AAA002USDT", "FTTUSDT", "ZZZUSDT"]

    def test_pagination_without_next_marker_uses_last_item(self) -> None:
        """S3 may omit NextMarker; the last listed item is the continuation point."""
        page1 = (f"{KLINES_PREFIX}BTCUSDT/",)
        page2 = (f"{KLINES_PREFIX}ETHUSDT/",)
        markers_seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            marker = request.url.params.get("marker")
            markers_seen.append(marker)
            if marker is None:
                xml = _listing_xml(request_prefix=KLINES_PREFIX, prefixes=page1, truncated=True)
            else:
                xml = _listing_xml(request_prefix=KLINES_PREFIX, prefixes=page2)
            return httpx.Response(200, text=xml)

        symbols = _client(handler).list_symbols()
        assert markers_seen == [None, page1[-1]]
        assert symbols == ["BTCUSDT", "ETHUSDT"]

    def test_non_advancing_marker_raises(self) -> None:
        """A repeated marker would loop forever — must fail loud instead."""

        def handler(request: httpx.Request) -> httpx.Response:
            xml = _listing_xml(
                request_prefix=KLINES_PREFIX,
                prefixes=(f"{KLINES_PREFIX}BTCUSDT/",),
                truncated=True,
                next_marker="stuck",
            )
            return httpx.Response(200, text=xml)

        with pytest.raises(SchemaError, match="not advancing"):
            _client(handler).list_symbols()

    def test_exclusions(self) -> None:
        """Underscore (delivery futures), non-USDT quotes, and junk are excluded."""
        prefixes = tuple(
            f"{KLINES_PREFIX}{name}/"
            for name in (
                "BTCUSDT",
                "BTCUSDT_210625",  # dated delivery future — excluded
                "ETHBTC",  # BTC-quoted — excluded
                "BTCBUSD",  # BUSD-quoted — excluded
                "1000SHIBUSDC",  # USDC-quoted — excluded
                "FTTUSDT",  # delisted perp — KEPT (the whole point)
                "USDT",  # empty base — unparseable, excluded
                "1000SHIBUSDT",  # numeric-prefixed base — kept
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text=_listing_xml(request_prefix=KLINES_PREFIX, prefixes=prefixes)
            )

        assert _client(handler).list_symbols() == ["1000SHIBUSDT", "BTCUSDT", "FTTUSDT"]


class TestMonthRange:
    def test_parses_zip_names_and_ignores_checksums(self) -> None:
        sym_prefix = f"{KLINES_PREFIX}FTTUSDT/1h/"
        keys = []
        for ym in ("2021-01", "2021-02", "2022-11"):
            keys.append(f"{sym_prefix}FTTUSDT-1h-{ym}.zip")
            keys.append(f"{sym_prefix}FTTUSDT-1h-{ym}.zip.CHECKSUM")
        keys.append(f"{sym_prefix}README.txt")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["prefix"] == sym_prefix
            return httpx.Response(
                200, text=_listing_xml(request_prefix=sym_prefix, keys=tuple(keys))
            )

        assert _client(handler).month_range("FTTUSDT") == (
            YearMonth(2021, 1),
            YearMonth(2022, 11),
        )

    def test_no_files_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text=_listing_xml(request_prefix=str(request.url.params["prefix"]))
            )

        assert _client(handler).month_range("NEVERUSDT") is None

    def test_bad_symbol_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be issued")

        with pytest.raises(ValueError, match="symbol"):
            _client(handler).month_range("../evil")


# ---------------------------------------------------------------------- klines


def _ohlcv_handler(csv_text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "data.binance.vision"
        return httpx.Response(200, content=_zip_bytes(csv_text))

    return handler


class TestFetchOhlcvMonth:
    def test_headerless_ms_file(self) -> None:
        """Pre-2022 archive shape: no header, millisecond open_time."""
        csv_text = "\n".join(_kline_row(JAN_2020 + i * H1_MS, price=7000.0 + i) for i in range(3))
        tbl = _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month(
            "BTCUSDT", 2020, 1, now=FAR_FUTURE
        )
        validate_table(tbl, Dataset.OHLCV)
        assert tbl.num_rows == 3
        assert tbl.column("instrument_id").to_pylist() == ["BINANCE:PERP:BTCUSDT"] * 3
        assert tbl.column("ts_open").to_pylist() == [
            datetime(2020, 1, 1, h, tzinfo=UTC) for h in range(3)
        ]
        assert tbl.column("open").to_pylist() == [7000.0, 7001.0, 7002.0]
        assert tbl.column("n_trades").to_pylist() == [42, 42, 42]
        assert tbl.column("quality_flags").to_pylist() == [0, 0, 0]
        assert tbl.column("quote_volume").to_pylist() == pytest.approx(
            [10.0 * 7000.0, 10.0 * 7001.0, 10.0 * 7002.0]
        )

    def test_headered_ms_file(self) -> None:
        """2025+ archive shape: header row present, millisecond open_time."""
        header = (
            "open_time,open,high,low,close,volume,close_time,quote_volume,"
            "count,taker_buy_volume,taker_buy_quote_volume,ignore"
        )
        csv_text = header + "\n" + _kline_row(JAN_2025, price=93548.8)
        tbl = _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month(
            "BTCUSDT", 2025, 1, now=FAR_FUTURE
        )
        assert tbl.num_rows == 1
        assert tbl.column("ts_open").to_pylist() == [datetime(2025, 1, 1, tzinfo=UTC)]
        assert tbl.column("open").to_pylist() == [93548.8]

    def test_microsecond_open_time_normalized(self) -> None:
        """2025+ µs-stamped file: open_time floor-divided to ms; values unchanged."""
        rows = [
            _kline_row((JAN_2025 + i * H1_MS) * 1000, price=50.0 + i)  # µs stamps
            for i in range(2)
        ]
        tbl = _client(_ohlcv_handler("\n".join(rows))).fetch_ohlcv_month(
            "BTCUSDT", 2025, 1, now=FAR_FUTURE
        )
        assert tbl.column("ts_open").to_pylist() == [
            datetime(2025, 1, 1, 0, tzinfo=UTC),
            datetime(2025, 1, 1, 1, tzinfo=UTC),
        ]

    def test_partial_bar_never_served(self) -> None:
        """A bar whose close is after ``now`` is dropped; one closing exactly at now stays."""
        ts0, ts1, ts2 = (JAN_2020 + i * H1_MS for i in range(3))
        csv_text = "\n".join(_kline_row(ts) for ts in (ts0, ts1, ts2))
        # now = close of the second bar: bars 0 and 1 are closed, bar 2 is in progress.
        now = ts1 + H1_MS
        tbl = _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month("BTCUSDT", 2020, 1, now=now)
        assert tbl.column("ts_open").to_pylist() == [
            datetime(2020, 1, 1, 0, tzinfo=UTC),
            datetime(2020, 1, 1, 1, tzinfo=UTC),
        ]

    def test_missing_month_raises_data_gap(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="NoSuchKey")

        with pytest.raises(DataGapError, match="2019-05"):
            _client(handler).fetch_ohlcv_month("FTTUSDT", 2019, 5, now=FAR_FUTURE)

    def test_duplicate_open_time_raises(self) -> None:
        csv_text = "\n".join([_kline_row(JAN_2020), _kline_row(JAN_2020)])
        with pytest.raises(SchemaError, match="duplicate"):
            _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month("BTCUSDT", 2020, 1, now=FAR_FUTURE)

    def test_out_of_month_timestamp_raises(self) -> None:
        csv_text = _kline_row(JAN_2020 - H1_MS)  # 2019-12-31 23:00 in the 2020-01 file
        with pytest.raises(SchemaError, match="outside month window"):
            _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month("BTCUSDT", 2020, 1, now=FAR_FUTURE)

    def test_unaligned_open_time_raises(self) -> None:
        csv_text = _kline_row(JAN_2020 + 1)  # 1 ms off the hour grid
        with pytest.raises(SchemaError, match="not 1h-aligned"):
            _client(_ohlcv_handler(csv_text)).fetch_ohlcv_month("BTCUSDT", 2020, 1, now=FAR_FUTURE)

    def test_bad_month_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be issued")

        with pytest.raises(ValueError, match="month"):
            _client(handler).fetch_ohlcv_month("BTCUSDT", 2020, 13, now=FAR_FUTURE)


# ---------------------------------------------------------------------- funding


class TestFetchFundingMonth:
    def test_archive_native_calc_time_variant(self) -> None:
        """Real archive shape: calc_time/funding_interval_hours/last_funding_rate,
        with genuine settlement jitter (+15 ms) preserved verbatim."""
        jittered = JAN_2025 + 15
        csv_text = (
            "calc_time,funding_interval_hours,last_funding_rate\n"
            f"{jittered},8,0.00010000\n"
            f"{JAN_2025 + 8 * H1_MS},8,-0.00012359\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2025-01.zip" in str(request.url)
            return httpx.Response(200, content=_zip_bytes(csv_text))

        tbl = _client(handler).fetch_funding_month("BTCUSDT", 2025, 1)
        validate_table(tbl, Dataset.FUNDING)
        assert tbl.num_rows == 2
        assert tbl.column("instrument_id").to_pylist() == ["BINANCE:PERP:BTCUSDT"] * 2
        ts = [int(v.timestamp() * 1000) for v in tbl.column("ts_funding").to_pylist()]
        assert ts == [jittered, JAN_2025 + 8 * H1_MS]  # jitter kept, never rounded
        avail = [int(v.timestamp() * 1000) for v in tbl.column("available_at").to_pylist()]
        assert avail == [t + FUNDING_PUBLICATION_LAG_MS for t in ts]
        assert FUNDING_PUBLICATION_LAG_MS == 300_000
        assert tbl.column("rate").to_pylist() == [0.0001, -0.00012359]

    def test_rest_shaped_funding_time_variant(self) -> None:
        """REST-shaped variant: symbol,fundingTime,fundingRate (column order differs)."""
        csv_text = f"symbol,fundingTime,fundingRate\nBTCUSDT,{JAN_2020},-0.00012359\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_zip_bytes(csv_text))

        tbl = _client(handler).fetch_funding_month("BTCUSDT", 2020, 1)
        assert tbl.column("rate").to_pylist() == [-0.00012359]
        avail = tbl.column("available_at").to_pylist()
        assert avail == [datetime(2020, 1, 1, 0, 5, tzinfo=UTC)]  # +5 min lag

    def test_headerless_positional_variant(self) -> None:
        """Headerless funding file → archive-native positional column order."""
        csv_text = f"{JAN_2020},8,0.0003\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_zip_bytes(csv_text))

        tbl = _client(handler).fetch_funding_month("BTCUSDT", 2020, 1)
        assert tbl.column("rate").to_pylist() == [0.0003]

    def test_microsecond_calc_time_normalized(self) -> None:
        csv_text = (
            "calc_time,funding_interval_hours,last_funding_rate\n"
            f"{JAN_2025 * 1000},8,0.0001\n"  # µs stamp
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_zip_bytes(csv_text))

        tbl = _client(handler).fetch_funding_month("BTCUSDT", 2025, 1)
        assert tbl.column("ts_funding").to_pylist() == [datetime(2025, 1, 1, tzinfo=UTC)]

    def test_unknown_header_raises(self) -> None:
        csv_text = "wat,is,this\n1,2,3\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_zip_bytes(csv_text))

        with pytest.raises(SchemaError, match="unrecognized funding header"):
            _client(handler).fetch_funding_month("BTCUSDT", 2020, 1)

    def test_missing_month_raises_data_gap(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="NoSuchKey")

        with pytest.raises(DataGapError, match="funding FTTUSDT 2023-01"):
            _client(handler).fetch_funding_month("FTTUSDT", 2023, 1)


# ---------------------------------------------------------------------- transport


class TestTransport:
    def test_transient_transport_error_is_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(
                200,
                text=_listing_xml(
                    request_prefix=KLINES_PREFIX, prefixes=(f"{KLINES_PREFIX}BTCUSDT/",)
                ),
            )

        client = _client(
            handler,
            retry=transient_retry((httpx.TransportError,), max_attempts=3, base_s=0.001),
        )
        assert client.list_symbols() == ["BTCUSDT"]
        assert calls["n"] == 2

    def test_non_transport_http_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, text="boom")

        client = _client(
            handler,
            retry=transient_retry((httpx.TransportError,), max_attempts=3, base_s=0.001),
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.list_symbols()
        assert calls["n"] == 1  # status errors are not transport errors

    def test_politeness_delay_sleeps_before_each_request(self, monkeypatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_listing_xml(request_prefix=KLINES_PREFIX))

        _client(handler, delay_s=0.05).list_symbols()
        assert sleeps == [0.05]

    def test_negative_delay_rejected(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200))
        with pytest.raises(ValueError, match="delay_s"):
            BinanceVisionClient(http=httpx.Client(transport=transport), delay_s=-1.0)

    def test_s3_url_constant(self) -> None:
        assert S3_LIST_ENDPOINT.startswith("https://s3-ap-northeast-1.amazonaws.com/")
