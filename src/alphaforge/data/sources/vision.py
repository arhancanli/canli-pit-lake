"""Binance Vision archive client — the survivorship-bias killer (leakage finding 1).

Binance's REST API (and therefore ccxt) only serves *currently listed* symbols: every
perp delisted before this system first ran (FTTUSDT, LUNAUSDT, SRMUSDT, ...) is simply
absent from ``exchangeInfo``, so a universe seeded from the live endpoint has already
"survived" to the present. ``data.binance.vision`` — Binance's public S3 archive —
retains **every** symbol's full kline/funding history forever, including delisted ones.
:class:`BinanceVisionClient` reads that archive: it lists the complete historical
symbol set, reports each symbol's data coverage (:meth:`BinanceVisionClient.month_range`),
and fetches monthly OHLCV/funding files normalized to the canonical lake schemas.

Temporal contract (dataDesign.md §0, non-negotiable):

* All timestamps crossing this interface are integer epoch milliseconds, UTC
  (:data:`~alphaforge.core.time.Ms`); Arrow columns are ``timestamp[ms, tz=UTC]``.
* Bars are labeled by OPEN time (``ts_open``); a 1h bar is available at
  ``ts_open + 3_600_000``. :meth:`BinanceVisionClient.fetch_ohlcv_month` never returns
  a bar whose close is in the future (partial bar), even though monthly archives only
  appear after month end — the guard is mechanical, not assumed.
* Funding rows carry ``available_at = ts_funding + FUNDING_PUBLICATION_LAG_MS``
  (settlement plus publication lag; leakage finding 18 — consumers join on
  ``available_at``, never ``ts_funding``).
* Raw archive data is never mutated or silently "fixed": malformed rows, duplicate
  timestamps, unaligned bar opens, and out-of-month timestamps raise
  :class:`~alphaforge.core.errors.SchemaError` instead of being repaired or dropped.

CSV formats implemented (verified against real archive files; documented because the
archive is header-inconsistent across eras):

* **Klines** (``<SYM>-1h-YYYY-MM.zip``): 12 columns ``open_time, open, high, low,
  close, volume, close_time, quote_volume, count, taker_buy_volume,
  taker_buy_quote_volume, ignore``. Pre-~2022 files have **no header row** (e.g.
  ``BTCUSDT-1h-2020-01``); later files have one (e.g. ``BTCUSDT-1h-2025-01``).
  Detection: a digit-leading first cell means data.
* **Funding** (``<SYM>-fundingRate-YYYY-MM.zip``): the archive's native format
  (observed in both 2020 and 2025 BTCUSDT files) is a headered
  ``calc_time, funding_interval_hours, last_funding_rate``; the REST-shaped variant
  ``symbol, fundingTime, fundingRate`` is also accepted (header-driven dispatch), and
  headerless funding files are parsed positionally in the archive's native column
  order. ``calc_time`` carries real settlement jitter (e.g. ``...600015``) — kept
  verbatim, never rounded.
* **Microsecond timestamps**: some 2025+ archive files stamp ``open_time`` in
  microseconds. Any epoch value ``>= 10**14`` is treated as µs and floor-divided by
  1000 (epoch-ms stays below 10**14 until year 5138; epoch-µs exceeded it in 1973, so
  the two ranges cannot collide for market data).
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import re
import time
import zipfile
from datetime import UTC, datetime
from typing import Final, NamedTuple
from xml.etree import ElementTree

import httpx
import pyarrow as pa
from tenacity import Retrying

from alphaforge.core.errors import DataGapError, SchemaError
from alphaforge.core.logging import get_logger
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms, Timeframe, now_ms, to_ms
from alphaforge.core.types import MarketType
from alphaforge.data.ingest.retry import transient_retry
from alphaforge.data.schemas import (
    FUNDING_SCHEMA,
    OHLCV_SCHEMA,
    Dataset,
    validate_table,
)
from alphaforge.data.sources.ccxt_source import FUNDING_PUBLICATION_LAG_MS

__all__ = [
    "DOWNLOAD_ENDPOINT",
    "S3_LIST_ENDPOINT",
    "BinanceVisionClient",
    "YearMonth",
]

_log = get_logger(__name__)

S3_LIST_ENDPOINT: Final[str] = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
"""S3 ListBucket endpoint for the ``data.binance.vision`` public bucket (XML API)."""

DOWNLOAD_ENDPOINT: Final[str] = "https://data.binance.vision"
"""HTTPS download host for archive objects (zip files)."""

_KLINES_PREFIX: Final[str] = "data/futures/um/monthly/klines/"
_FUNDING_PREFIX: Final[str] = "data/futures/um/monthly/fundingRate/"
_S3_NS: Final[str] = "{http://s3.amazonaws.com/doc/2006-03-01/}"

_MICROSECOND_THRESHOLD: Final[int] = 10**14
"""Epoch values at or above this are microseconds, below it milliseconds.

Epoch-ms reaches 10**14 in the year 5138; epoch-µs passed it in March 1973. Every
plausible market-data timestamp therefore falls unambiguously on one side.
"""

_FUNDING_MONTH_SLACK_MS: Final[int] = Timeframe.H1.ms
"""Tolerance around the month window for funding settlement jitter (1 hour).

Binance settlement stamps drift a few ms around the nominal boundary (observed:
``1735689600015``); a record stamped just across a month edge may land in the
neighboring file. OHLCV gets no slack — bar opens are exact."""

#: Normalized funding header names (lowercased, underscores stripped) accepted as the
#: settlement-timestamp / rate columns, in both documented archive variants.
_FUNDING_TS_HEADERS: Final[frozenset[str]] = frozenset({"calctime", "fundingtime"})
_FUNDING_RATE_HEADERS: Final[frozenset[str]] = frozenset({"lastfundingrate", "fundingrate"})


class YearMonth(NamedTuple):
    """A UTC calendar month, the archive's file granularity. Orders chronologically."""

    year: int
    month: int

    def first_ms(self) -> Ms:
        """Epoch ms (UTC) of the first instant of this month (``YYYY-MM-01 00:00:00``)."""
        return to_ms(datetime(self.year, self.month, 1, tzinfo=UTC))

    def next_month_first_ms(self) -> Ms:
        """Epoch ms (UTC) of the first instant of the *following* month.

        Together with :meth:`first_ms` this bounds the month half-open:
        ``[first_ms(), next_month_first_ms())``.
        """
        if self.month == 12:
            return to_ms(datetime(self.year + 1, 1, 1, tzinfo=UTC))
        return to_ms(datetime(self.year, self.month + 1, 1, tzinfo=UTC))


def _normalize_epoch_ms(raw: int) -> Ms:
    """Normalize an archive epoch stamp to milliseconds (µs detected and floored)."""
    if raw >= _MICROSECOND_THRESHOLD:
        return raw // 1000
    return raw


def _require_month(year: int, month: int) -> YearMonth:
    """Validate calendar bounds and return the :class:`YearMonth`; ``ValueError`` if bad."""
    if not 1000 <= year <= 9999:
        raise ValueError(f"year must be a 4-digit year (1000-9999), got {year}")
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return YearMonth(year, month)


def _require_symbol(symbol: str) -> str:
    """Validate an exchange symbol for safe URL/prefix interpolation; returns it."""
    if not symbol or any(ch in symbol for ch in ("/", "?", "&", "\x00")):
        raise ValueError(f"symbol must be non-empty without URL metacharacters, got {symbol!r}")
    return symbol


def _require_finite(what: str, value: float) -> float:
    """Raise :class:`SchemaError` unless ``value`` is finite; returns it."""
    if not math.isfinite(value):
        raise SchemaError(f"{what}: non-finite numeric value {value!r}")
    return value


class BinanceVisionClient:
    """Read-only client for the ``data.binance.vision`` public archive (USDT-M perps, 1h).

    Pure fetcher in the :class:`~alphaforge.data.sources.base.DataSource` spirit: it
    normalizes archive payloads to the canonical lake schemas and never writes to the
    lake. Unlike ``DataSource`` it is month-oriented — the archive's unit of retrieval
    is one ``(symbol, month)`` zip — so backfill jobs iterate months, not cursor pages.

    Args:
        http: Injectable :class:`httpx.Client` (tests pass a ``MockTransport``-backed
            client). ``None`` constructs a default client with a 30s timeout. The
            client is treated as owned by the caller when injected; the default one
            lives for the process.
        delay_s: Politeness delay in seconds slept before *every* HTTP request
            (default ``0.0`` — keep 0 in tests; production backfills should set a
            modest value such as ``0.1`` to be a good citizen on a public bucket).
        retry: tenacity policy for transient transport failures. Defaults to
            :func:`~alphaforge.data.ingest.retry.transient_retry` over
            ``httpx.TransportError`` (connect/read/write/pool errors). HTTP *status*
            errors are not retried here: 404 means the month does not exist
            (:class:`DataGapError`), other statuses raise ``httpx.HTTPStatusError``.
    """

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        delay_s: float = 0.0,
        retry: Retrying | None = None,
    ) -> None:
        if delay_s < 0.0:
            raise ValueError(f"delay_s must be >= 0, got {delay_s}")
        self._http = http if http is not None else httpx.Client(timeout=30.0)
        self._delay_s = delay_s
        self._retry = retry if retry is not None else transient_retry((httpx.TransportError,))

    # ------------------------------------------------------------------ transport

    def _get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        """One GET with politeness delay and transient-transport retry; no status check."""
        if self._delay_s > 0.0:
            time.sleep(self._delay_s)
        return self._retry(self._http.get, url, params=params)

    def _list(self, prefix: str) -> tuple[list[str], list[str]]:
        """Full S3 listing under ``prefix`` with ``delimiter=/``, following pagination.

        Returns ``(common_prefixes, keys)`` accumulated across **all** pages: S3 caps
        each response at ``MaxKeys`` (1000) items and sets ``IsTruncated``; the next
        page is requested with ``marker`` = the response's ``NextMarker`` when present,
        else the lexicographically last item of the page (S3 lists in lexicographic
        order, so that is the correct continuation point). Raises
        :class:`SchemaError` if a truncated page offers no continuation point or the
        marker fails to advance (would loop forever).
        """
        prefixes: list[str] = []
        keys: list[str] = []
        marker: str | None = None
        seen_markers: set[str] = set()
        while True:
            params = {"delimiter": "/", "prefix": prefix}
            if marker is not None:
                params["marker"] = marker
            resp = self._get(S3_LIST_ENDPOINT, params=params)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
            page_prefixes = [
                el.text for el in root.findall(f"{_S3_NS}CommonPrefixes/{_S3_NS}Prefix") if el.text
            ]
            page_keys = [el.text for el in root.findall(f"{_S3_NS}Contents/{_S3_NS}Key") if el.text]
            prefixes.extend(page_prefixes)
            keys.extend(page_keys)
            truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "false").strip().lower()
            if truncated != "true":
                return prefixes, keys
            next_marker = root.findtext(f"{_S3_NS}NextMarker")
            last_items = page_keys[-1:] + page_prefixes[-1:]
            if next_marker:
                new_marker = next_marker
            elif last_items:
                new_marker = max(last_items)
            else:
                raise SchemaError(
                    f"S3 listing for prefix {prefix!r} is truncated but offers no "
                    "NextMarker and returned no items to continue from"
                )
            if new_marker in seen_markers:
                raise SchemaError(
                    f"S3 listing pagination for prefix {prefix!r} is not advancing "
                    f"(marker {new_marker!r} repeated)"
                )
            seen_markers.add(new_marker)
            marker = new_marker

    def _download_csv_rows(self, url: str, what: str) -> list[list[str]]:
        """Download a zip, extract its single CSV member, return non-empty rows.

        The header row (if any) is **included** — callers dispatch on it. Raises
        :class:`DataGapError` on 404 (month absent from the archive),
        ``httpx.HTTPStatusError`` on other statuses, :class:`SchemaError` on a
        corrupt zip or unexpected member layout.
        """
        resp = self._get(url)
        if resp.status_code == 404:
            raise DataGapError(
                f"{what}: Binance Vision has no monthly archive at {url} — the symbol "
                "has no data for this month (before listing, after delisting, or the "
                "month is not yet published)"
            )
        resp.raise_for_status()
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as exc:
            raise SchemaError(f"{what}: corrupt zip archive at {url}: {exc}") from exc
        with zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(csv_names) != 1:
                raise SchemaError(
                    f"{what}: expected exactly one CSV member in {url}, "
                    f"found {sorted(zf.namelist())!r}"
                )
            text = zf.read(csv_names[0]).decode("utf-8")
        return [row for row in csv.reader(io.StringIO(text)) if row]

    # ------------------------------------------------------------------ catalog

    def list_symbols(self) -> list[str]:
        """All USDT-quoted UM-perp symbols ever archived — including delisted ones.

        Lists the symbol "directories" (S3 CommonPrefixes) under the monthly-klines
        prefix and filters to the v1 candidate set:

        * symbols containing ``"_"`` are excluded (dated delivery futures such as
          ``BTCUSDT_210625`` — calendar contracts, not perps);
        * only USDT-quoted symbols are kept (quote recovered by
          :meth:`SymbolMapper.split_exchange_symbol` longest-suffix logic, so e.g.
          ``...BUSD``/``...USDC`` never sneak in via a sloppy ``endswith``); symbols
          ending with no known quote are skipped with a warning.

        Returns a sorted list of exchange symbols (``["BTCUSDT", "FTTUSDT", ...]``).
        This is the historical candidate set that makes pre-go-live delistings visible
        to seeding/universe code (finding 1).
        """
        common_prefixes, _ = self._list(_KLINES_PREFIX)
        symbols: list[str] = []
        for prefix in common_prefixes:
            name = prefix.removeprefix(_KLINES_PREFIX).strip("/")
            if not name or "_" in name:
                continue
            try:
                _, quote = SymbolMapper.split_exchange_symbol(name)
            except SchemaError:
                _log.warning("vision.list_symbols.unparseable_symbol", symbol=name)
                continue
            if quote != "USDT":
                continue
            symbols.append(name)
        return sorted(symbols)

    def month_range(self, symbol: str) -> tuple[YearMonth, YearMonth] | None:
        """First and last month with an archived 1h kline file for ``symbol``.

        Parses ``<SYM>-1h-YYYY-MM.zip`` object names under the symbol's 1h prefix
        (``.CHECKSUM`` companions and any other objects are ignored). Returns
        ``(first, last)`` chronologically, or ``None`` if the symbol has no archived
        1h months at all. The range brackets the symbol's lifetime: the seeder derives
        ``listed_ts = first.first_ms()`` and the conservative
        ``delisted_ts = last.next_month_first_ms()`` from it.
        """
        _require_symbol(symbol)
        _, keys = self._list(f"{_KLINES_PREFIX}{symbol}/1h/")
        pattern = re.compile(rf"^{re.escape(symbol)}-1h-(\d{{4}})-(\d{{2}})\.zip$")
        months: list[YearMonth] = []
        for key in keys:
            match = pattern.match(key.rsplit("/", 1)[-1])
            if match is None:
                continue
            year, month = int(match.group(1)), int(match.group(2))
            if 1 <= month <= 12:
                months.append(YearMonth(year, month))
        if not months:
            return None
        return min(months), max(months)

    # ------------------------------------------------------------------ fetchers

    def fetch_ohlcv_month(
        self, symbol: str, year: int, month: int, *, now: Ms | None = None
    ) -> pa.Table:
        """Fetch one month of archived 1h klines, normalized to ``OHLCV_SCHEMA``.

        Columns map straight off the archive CSV: ``open/high/low/close/volume`` from
        columns 1-5, ``quote_volume`` from column 7, ``n_trades`` from ``count``
        (column 8); ``quality_flags = 0`` (the quality layer flags later, the source
        never pre-judges); ``ingested_at = now_ms()`` wall clock. ``open_time`` is
        normalized from µs where needed (module docstring) and must be exactly
        1h-aligned, strictly increasing, and inside the requested month — violations
        raise :class:`SchemaError` (raw data is never silently fixed).

        Closed-bar guard: rows with ``ts_open + 3_600_000 > now`` are dropped so a
        partial bar can never be served, regardless of what the archive contains.
        ``now`` is injectable for tests; ``None`` reads the wall clock once.

        Raises:
            DataGapError: the archive has no file for this ``(symbol, month)``.
            SchemaError: malformed CSV, corrupt zip, or timestamp-discipline violation.
            ValueError: ``year``/``month`` out of calendar bounds or bad ``symbol``.
        """
        _require_symbol(symbol)
        ym = _require_month(year, month)
        what = f"ohlcv {symbol} {year:04d}-{month:02d}"
        url = (
            f"{DOWNLOAD_ENDPOINT}/{_KLINES_PREFIX}{symbol}/1h/"
            f"{symbol}-1h-{year:04d}-{month:02d}.zip"
        )
        rows = self._download_csv_rows(url, what)
        if rows and not rows[0][0].lstrip("-").isdigit():
            header = rows[0]
            if header[0].strip().lower() != "open_time":
                raise SchemaError(
                    f"{what}: unrecognized kline header {header!r}; expected first "
                    "column 'open_time'"
                )
            rows = rows[1:]

        now_v = now_ms() if now is None else now
        lo, hi = ym.first_ms(), ym.next_month_first_ms()
        delta = Timeframe.H1.ms
        parsed: list[tuple[Ms, float, float, float, float, float, float, int]] = []
        for i, row in enumerate(rows):
            if len(row) < 9:
                raise SchemaError(f"{what}: row {i} has {len(row)} columns, expected >= 9: {row!r}")
            try:
                ts_open = _normalize_epoch_ms(int(row[0]))
                values = (
                    ts_open,
                    _require_finite(what, float(row[1])),
                    _require_finite(what, float(row[2])),
                    _require_finite(what, float(row[3])),
                    _require_finite(what, float(row[4])),
                    _require_finite(what, float(row[5])),
                    _require_finite(what, float(row[7])),
                    int(row[8]),
                )
            except ValueError as exc:
                raise SchemaError(f"{what}: malformed row {i}: {row!r} ({exc})") from exc
            if not lo <= ts_open < hi:
                raise SchemaError(
                    f"{what}: row {i} open_time {ts_open} outside month window "
                    f"[{lo}, {hi}) — wrong file content or unit misdetection"
                )
            if ts_open % delta != 0:
                raise SchemaError(f"{what}: row {i} open_time {ts_open} is not 1h-aligned")
            if ts_open + delta > now_v:
                continue  # partial/unclosed bar — never served
            parsed.append(values)

        parsed.sort(key=lambda r: r[0])
        for prev, cur in itertools.pairwise(parsed):
            if cur[0] == prev[0]:
                raise SchemaError(f"{what}: duplicate open_time {cur[0]} in archive file")

        instrument_id = SymbolMapper.to_instrument_id("BINANCE", MarketType.PERP, symbol)
        n = len(parsed)
        ingested = now_ms()
        tbl = pa.Table.from_arrays(
            [
                pa.array([instrument_id] * n, pa.string()),
                pa.array([r[0] for r in parsed], pa.timestamp("ms", tz="UTC")),
                pa.array([r[1] for r in parsed], pa.float64()),
                pa.array([r[2] for r in parsed], pa.float64()),
                pa.array([r[3] for r in parsed], pa.float64()),
                pa.array([r[4] for r in parsed], pa.float64()),
                pa.array([r[5] for r in parsed], pa.float64()),
                pa.array([r[6] for r in parsed], pa.float64()),
                pa.array([r[7] for r in parsed], pa.int64()),
                pa.array([0] * n, pa.int32()),
                pa.array([ingested] * n, pa.timestamp("ms", tz="UTC")),
            ],
            schema=OHLCV_SCHEMA,
        )
        validate_table(tbl, Dataset.OHLCV)
        return tbl

    def fetch_funding_month(self, symbol: str, year: int, month: int) -> pa.Table:
        """Fetch one month of archived funding settlements, normalized to ``FUNDING_SCHEMA``.

        Header-driven column dispatch (formats documented in the module docstring):
        the archive-native ``calc_time/funding_interval_hours/last_funding_rate`` and
        the REST-shaped ``symbol/fundingTime/fundingRate`` are both accepted; a
        headerless file is parsed positionally in archive-native order. Settlement
        timestamps are kept **verbatim** (real-world jitter like ``...600015`` is
        data, not noise) and may drift up to 1h across the month boundary
        (:data:`_FUNDING_MONTH_SLACK_MS`); larger excursions raise
        :class:`SchemaError`.

        Point-in-time: ``available_at = ts_funding + FUNDING_PUBLICATION_LAG_MS``
        (publication lag, leakage findings 5/18) — downstream joins use
        ``available_at`` exclusively.

        Raises:
            DataGapError: the archive has no file for this ``(symbol, month)``.
            SchemaError: unrecognized header, malformed rows, duplicates, or
                out-of-window settlement timestamps.
            ValueError: ``year``/``month`` out of calendar bounds or bad ``symbol``.
        """
        _require_symbol(symbol)
        ym = _require_month(year, month)
        what = f"funding {symbol} {year:04d}-{month:02d}"
        url = (
            f"{DOWNLOAD_ENDPOINT}/{_FUNDING_PREFIX}{symbol}/"
            f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
        )
        rows = self._download_csv_rows(url, what)

        ts_idx, rate_idx = 0, 2  # archive-native positional order for headerless files
        if rows and not rows[0][0].lstrip("-").isdigit():
            header = [cell.strip().lower().replace("_", "") for cell in rows[0]]
            ts_candidates = [i for i, h in enumerate(header) if h in _FUNDING_TS_HEADERS]
            rate_candidates = [i for i, h in enumerate(header) if h in _FUNDING_RATE_HEADERS]
            if len(ts_candidates) != 1 or len(rate_candidates) != 1:
                raise SchemaError(
                    f"{what}: unrecognized funding header {rows[0]!r}; need exactly one "
                    f"of {sorted(_FUNDING_TS_HEADERS)} and one of "
                    f"{sorted(_FUNDING_RATE_HEADERS)}"
                )
            ts_idx, rate_idx = ts_candidates[0], rate_candidates[0]
            rows = rows[1:]

        lo = ym.first_ms() - _FUNDING_MONTH_SLACK_MS
        hi = ym.next_month_first_ms() + _FUNDING_MONTH_SLACK_MS
        needed = max(ts_idx, rate_idx) + 1
        parsed: list[tuple[Ms, float]] = []
        for i, row in enumerate(rows):
            if len(row) < needed:
                raise SchemaError(
                    f"{what}: row {i} has {len(row)} columns, expected >= {needed}: {row!r}"
                )
            try:
                ts_funding = _normalize_epoch_ms(int(row[ts_idx]))
                rate = _require_finite(what, float(row[rate_idx]))
            except ValueError as exc:
                raise SchemaError(f"{what}: malformed row {i}: {row!r} ({exc})") from exc
            if not lo <= ts_funding < hi:
                raise SchemaError(
                    f"{what}: row {i} settlement time {ts_funding} outside month window "
                    f"[{lo}, {hi}) — wrong file content or unit misdetection"
                )
            parsed.append((ts_funding, rate))

        parsed.sort(key=lambda r: r[0])
        for prev, cur in itertools.pairwise(parsed):
            if cur[0] == prev[0]:
                raise SchemaError(f"{what}: duplicate settlement time {cur[0]} in archive file")

        instrument_id = SymbolMapper.to_instrument_id("BINANCE", MarketType.PERP, symbol)
        n = len(parsed)
        ingested = now_ms()
        tbl = pa.Table.from_arrays(
            [
                pa.array([instrument_id] * n, pa.string()),
                pa.array([r[0] for r in parsed], pa.timestamp("ms", tz="UTC")),
                pa.array([r[1] for r in parsed], pa.float64()),
                pa.array(
                    [r[0] + FUNDING_PUBLICATION_LAG_MS for r in parsed],
                    pa.timestamp("ms", tz="UTC"),
                ),
                pa.array([ingested] * n, pa.timestamp("ms", tz="UTC")),
            ],
            schema=FUNDING_SCHEMA,
        )
        validate_table(tbl, Dataset.FUNDING)
        return tbl
