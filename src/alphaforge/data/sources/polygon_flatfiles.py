"""``PolygonFlatFilesSource`` — US equities daily bars from the Polygon S3 flat files.

This is the **entitled bar path** for equities. The REST aggregates endpoint
(``/v2/aggs``, served by :class:`~alphaforge.data.sources.polygon_source.PolygonEquitiesSource`)
is NOT entitled on this account; the S3 flat files ARE (EQUITIES_INGEST.md §0). The
bars therefore come from one gzipped CSV per trading day in S3, NOT from REST.

Confirmed flat-files layout (EQUITIES_INGEST.md §1.1, tested by the operator):

* Endpoint ``https://files.polygon.io`` ; bucket ``flatfiles`` ; signature ``s3v4``.
* Daily stock bars key ``us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`` — one
  gzipped CSV per session (no object exists for a non-session day).
* CSV columns (header present): ``ticker, volume, open, close, high, low,
  window_start, transactions``. ``window_start`` is **nanoseconds** epoch UTC.
* One row per ticker that traded that day, so the panel is **survivorship-free by
  construction** — a name that later delisted is present in its old day-files.
* Prices are **UNADJUSTED** (raw) — exactly what the immutable lake stores. Split/
  dividend (total-return) adjustment is reconstructed point-in-time at *feature* time
  by :func:`~alphaforge.features.library.equity_price.adjusted_close`, never at ingest.

Design notes (why this is NOT a :class:`~alphaforge.data.sources.base.DataSource`):

* The ABC's per-instrument ``fetch_ohlcv``/``fetch_funding`` shape does not fit a
  per-DAY panel file; forcing it would re-download the whole day file once per ticker.
  This source's unit of work is one trading day → one bar table for EVERY ticker that
  traded that day. The ingest job (:class:`~alphaforge.data.ingest.equities.EquitiesFlatFilesJob`)
  owns the date loop and the day-watermark checkpoints.
* The S3 surface is behind :class:`FlatFilesClientProtocol` (a structural
  :class:`typing.Protocol`) so tests inject an in-memory fake — never boto3, never the
  network (mirrors ``PolygonClientProtocol``). Construction performs no I/O; a missing
  credential with no injected client raises :class:`ConfigError`.

All public methods take/return integer epoch milliseconds UTC (:data:`Ms`).
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final, Protocol

import pyarrow as pa

from alphaforge.core.errors import ConfigError, SchemaError
from alphaforge.core.symbols import EQUITY_MIC, SymbolMapper
from alphaforge.core.time import Ms, Timeframe, floor_bar, from_ms, now_ms, to_ms
from alphaforge.data.schemas import OHLCV_SCHEMA, Dataset, validate_table

_LOG = logging.getLogger(__name__)

# Max fraction of a day's tickers that may be skipped as unmappable before the day
# fails loud. A handful of stray ETFs/special tickers (e.g. one whose canonical form
# ends in the synthetic 'USD' quote) is expected and fine; a large fraction means a
# systematic id-mapping regression, not data noise, and must NOT silently shrink the
# universe.
_MAX_SKIP_FRAC: Final[float] = 0.02

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

__all__ = [
    "DAY_AGGS_PREFIX",
    "FLATFILES_MIC",
    "FlatFilesClientProtocol",
    "PolygonFlatFilesSource",
    "day_aggs_key",
    "parse_day_csv",
]

DAY_AGGS_PREFIX: Final[str] = "us_stocks_sip/day_aggs_v1"
"""Flat-files prefix for daily US stock bars (one gzipped CSV per session)."""

_NS_PER_MS: Final[int] = 1_000_000

FLATFILES_MIC: Final[str] = EQUITY_MIC
"""MIC segment for a flat-files ticker — the canonical
:data:`~alphaforge.core.symbols.EQUITY_MIC` (``"XUSE"``).

The day-aggs file carries no listing exchange, so flat-files bars use the single synthetic
venue segment ``EQUITY_MIC`` ("US equities, exchange-unspecified"). The reference seeder
(:mod:`~alphaforge.data.sources.polygon_source`) builds its SCD2
:class:`~alphaforge.core.instruments.Instrument` ids from the SAME constant, so bar
partitions, universe intervals, and factor reads all key identically — the bars and the
instrument record can never diverge under different MICs. The real listing venue is
intentionally NOT in the id (and is not needed: ``calendar_for`` is keyed on AssetClass),
so no per-row reference lookup is required at ingest.
"""

#: CSV header the day-aggs files carry, in their exact column order. Validated on every
#: parse so a vendor layout change fails LOUD rather than silently mis-mapping columns.
_EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
)


def day_aggs_key(d: date) -> str:
    """Session date → flat-files object key ``.../YYYY/MM/YYYY-MM-DD.csv.gz``."""
    return f"{DAY_AGGS_PREFIX}/{d.year:04d}/{d.month:02d}/{d.isoformat()}.csv.gz"


def _canonical_ticker(vendor_ticker: str) -> str:
    """Vendor ticker → separator-free canonical form (``"BRK.B"`` → ``"BRKB"``).

    Byte-identical rule to ``PolygonEquitiesSource._canonical_ticker`` so a flat-files
    bar id JOINs the SCD2 record built by the REST reference adapter — the survivorship
    guarantee depends on this matching (EQUITIES_INGEST.md §1.3).
    """
    return vendor_ticker.replace(".", "").replace("/", "").upper()


@dataclass(frozen=True, slots=True)
class _DayRow:
    """One parsed day-aggs CSV row (raw, pre-id-mapping). Numeric fields are coerced."""

    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    transactions: int
    window_start_ns: int


def _to_float(value: str, *, column: str, ticker: str) -> float:
    """Parse a non-nullable OHLCV cell to float; raise :class:`SchemaError` on garbage.

    Mirrors ``polygon_source._opt_float`` discipline but non-nullable: a malformed day
    file must fail LOUD (naming the column + ticker), never silently drop the bar.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"day-aggs row for ticker {ticker!r}: column {column!r} is not a number: {value!r}"
        ) from exc


def _to_int(value: str, *, column: str, ticker: str) -> int:
    """Parse a non-nullable integer cell; raise :class:`SchemaError` on garbage."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"day-aggs row for ticker {ticker!r}: column {column!r} is not an integer: {value!r}"
        ) from exc


def parse_day_csv(data: bytes) -> list[_DayRow]:
    """Parse a (decompressed) day-aggs CSV body into typed rows — pure, no S3.

    Validates the header against :data:`_EXPECTED_COLUMNS` exactly (order-sensitive),
    coerces every numeric cell (raising :class:`SchemaError` naming the offending
    column + ticker), and rejects an empty/whitespace ``ticker``. Exported so a
    fixture-CSV test can exercise it with zero S3 involvement (EQUITIES_INGEST.md §4.1).
    """
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    rows = iter(reader)
    try:
        header = next(rows)
    except StopIteration as exc:
        raise SchemaError("day-aggs CSV is empty (no header row)") from exc
    if tuple(h.strip() for h in header) != _EXPECTED_COLUMNS:
        raise SchemaError(
            f"day-aggs CSV header {header!r} does not match the expected columns "
            f"{list(_EXPECTED_COLUMNS)}"
        )
    out: list[_DayRow] = []
    for raw in rows:
        if not raw:
            continue
        if len(raw) != len(_EXPECTED_COLUMNS):
            raise SchemaError(
                f"day-aggs row has {len(raw)} fields, expected {len(_EXPECTED_COLUMNS)}: {raw!r}"
            )
        record = dict(zip(_EXPECTED_COLUMNS, raw, strict=True))
        ticker = record["ticker"].strip()
        if not ticker:
            raise SchemaError(f"day-aggs row has an empty ticker: {raw!r}")
        # Common-stock filter. Polygon encodes preferred shares / warrants / units /
        # rights with a LOWERCASE letter in the ticker (e.g. "BCpC" is a preferred
        # class, distinct from common "BCPC"; "TpC" vs "TPC"). Those are NOT the equity
        # factor sleeve's target (momentum / low-vol / betting-against-beta are
        # common-stock anomalies), and after the case-folding canonicalization they
        # collide with the common stock (the writer correctly refuses to merge two
        # distinct securities into one id). Skipping them yields the common-stock
        # universe, still survivorship-free for common stock (delisted commons are kept;
        # only non-common instruments are dropped). Class shares like "BRK.B" are
        # all-uppercase and are retained.
        if any(c.islower() for c in ticker):
            continue
        out.append(
            _DayRow(
                ticker=ticker,
                open=_to_float(record["open"], column="open", ticker=ticker),
                high=_to_float(record["high"], column="high", ticker=ticker),
                low=_to_float(record["low"], column="low", ticker=ticker),
                close=_to_float(record["close"], column="close", ticker=ticker),
                volume=_to_float(record["volume"], column="volume", ticker=ticker),
                transactions=_to_int(record["transactions"], column="transactions", ticker=ticker),
                window_start_ns=_to_int(
                    record["window_start"], column="window_start", ticker=ticker
                ),
            )
        )
    return out


class FlatFilesClientProtocol(Protocol):
    """The exact S3 surface :class:`PolygonFlatFilesSource` calls (boto3-shaped, structural).

    A structural :class:`typing.Protocol` so the test fake is mypy-strict without
    subclassing, exactly like ``PolygonClientProtocol``. The real implementation
    (:class:`_Boto3FlatFilesClient`) is a thin boto3 wrapper, exercised only when real
    ``POLYGON_S3_*`` credentials are configured — never in the unit suite.
    """

    def list_keys(self, *, prefix: str) -> list[str]:
        """List object keys under ``prefix`` (paginated internally), sorted ascending."""
        ...

    def get_object_bytes(self, *, key: str) -> bytes:
        """Download one object's RAW (still-gzipped) bytes; raise ``KeyError`` if absent."""
        ...


class _Boto3FlatFilesClient:
    """Thin boto3 S3 implementation of :class:`FlatFilesClientProtocol` (the live path).

    Constructed from the ``POLYGON_S3_*`` environment via :meth:`from_env`. Uses
    signature version ``s3v4`` against ``POLYGON_S3_ENDPOINT``. It is exercised ONLY
    when real credentials are configured (the operator's live ingest); every unit test
    injects an in-memory fake instead, so this class never touches the network in CI.
    """

    def __init__(self, client: S3Client, *, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def from_env(
        cls,
        *,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
    ) -> _Boto3FlatFilesClient:
        """Build from explicit args falling back to ``POLYGON_S3_*`` env vars.

        Raises :class:`ConfigError` naming the first missing credential — there is NO
        unauthenticated call (EQUITIES_INGEST.md §1.2), mirroring
        ``PolygonEquitiesSource.__init__``.
        """
        import boto3
        from botocore.config import Config

        resolved = {
            "POLYGON_S3_ENDPOINT": endpoint or os.environ.get("POLYGON_S3_ENDPOINT"),
            "POLYGON_S3_ACCESS_KEY_ID": (
                access_key_id or os.environ.get("POLYGON_S3_ACCESS_KEY_ID")
            ),
            "POLYGON_S3_SECRET_ACCESS_KEY": (
                secret_access_key or os.environ.get("POLYGON_S3_SECRET_ACCESS_KEY")
            ),
            "POLYGON_S3_BUCKET": bucket or os.environ.get("POLYGON_S3_BUCKET"),
        }
        missing = [name for name, value in resolved.items() if not value]
        if missing:
            raise ConfigError(
                "PolygonFlatFilesSource requires the S3 credentials "
                f"{missing} (or an injected client); refusing to issue an "
                "unauthenticated flat-files request"
            )
        client: S3Client = boto3.client(
            "s3",
            endpoint_url=resolved["POLYGON_S3_ENDPOINT"],
            aws_access_key_id=resolved["POLYGON_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=resolved["POLYGON_S3_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
        )
        return cls(client, bucket=str(resolved["POLYGON_S3_BUCKET"]))

    def list_keys(self, *, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if key is not None:
                    keys.append(key)
        return sorted(keys)

    def get_object_bytes(self, *, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise KeyError(key) from exc
            raise
        body: bytes = response["Body"].read()
        return body


class PolygonFlatFilesSource:
    """US equities daily bars from the Polygon S3 flat files (see module docstring).

    Construction performs no I/O; a missing credential with no injected ``client``
    raises :class:`ConfigError`.
    """

    def __init__(
        self,
        *,
        client: FlatFilesClientProtocol | None = None,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
        mic: str = FLATFILES_MIC,
    ) -> None:
        if client is None:
            client = _Boto3FlatFilesClient.from_env(
                endpoint=endpoint,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                bucket=bucket,
            )
        self._client = client
        self._mic = mic
        self.name = "polygon.flatfiles"

    # ------------------------------------------------------------------ listing

    def available_sessions(self, *, start: Ms, until: Ms) -> list[date]:
        """Session dates with a day-aggs object whose UTC midnight is in ``[start, until)``.

        Lists S3 under the ``YYYY/MM`` prefixes covering the span and parses each
        ``YYYY-MM-DD.csv.gz`` key back to a date; returns sorted, de-duplicated dates.
        A non-session day simply has no object (gap-free by absence). Empty when
        ``until <= start``.
        """
        if until <= start:
            return []
        seen: set[date] = set()
        for prefix in self._month_prefixes(start, until):
            for key in self._client.list_keys(prefix=prefix):
                parsed = self._date_from_key(key)
                if parsed is None:
                    continue
                day_open = self.session_open_ms(parsed)
                if start <= day_open < until:
                    seen.add(parsed)
        return sorted(seen)

    # ------------------------------------------------------------------ fetch one day

    def fetch_day(self, d: date, *, now: Ms | None = None) -> pa.Table:
        """Download + gunzip + parse one day-aggs CSV → an :data:`OHLCV_SCHEMA` table.

        Maps every ticker that traded that day (EQUITIES_INGEST.md §1.4):

        * ``instrument_id = equity_instrument_id(mic, canonical(ticker))`` — the
          flat-files id (synthetic MIC :data:`FLATFILES_MIC`), so dotted tickers
          (``BRK.B`` → ``BRKB``) JOIN the SCD2 record; a ticker that itself ends in a
          crypto quote token (``BTC``, ``ETH``) round-trips intact (it goes through
          ``equity_instrument_id``, never ``split_exchange_symbol``).
        * ``ts_open = floor_bar(window_start_ns // 1e6, D1)`` — ns → ms → UTC day open
          (00:00). The floor makes a stray intraday ``window_start`` total and keeps the
          bar labeled by its session open.
        * OHLCV from open/high/low/close/volume; ``quote_volume = close * volume`` (the
          flat files carry no VWAP — the documented fallback the universe ranks ADV on);
          ``n_trades = transactions``; ``quality_flags = 0``; ``ingested_at`` = wall clock.

        Closed-bar guard: a row whose close ``ts_open + D1.ms > now`` is dropped here
        (a same-day in-progress file should never appear, but the guard is belt-and-
        braces and lets the writer's guard be redundant rather than load-bearing).

        Returns a table conforming to :data:`OHLCV_SCHEMA`, sorted and unique on
        ``(instrument_id, ts_open)``. A duplicate ``(instrument_id, ts_open)`` (two
        vendor tickers canonicalizing to the same id within one day) raises
        :class:`SchemaError` — a silent drop would corrupt the panel.

        Raises:
            KeyError: no S3 object exists for ``d`` (the job, not the source, decides
                skip vs fail).
            SchemaError: a malformed CSV (bad header / unparseable cell / empty ticker /
                duplicate id) — a bad file fails LOUD, never silently drops the day.
        """
        now_eff: Ms = now_ms() if now is None else now
        ingested_at: Ms = now_ms()
        raw_gz = self._client.get_object_bytes(key=day_aggs_key(d))
        rows = parse_day_csv(gzip.decompress(raw_gz))
        return self._rows_to_table(rows, now=now_eff, ingested_at=ingested_at)

    # ------------------------------------------------------------------ internals

    def _rows_to_table(self, rows: Iterable[_DayRow], *, now: Ms, ingested_at: Ms) -> pa.Table:
        """Map parsed rows → an :data:`OHLCV_SCHEMA` table, guarded + sorted + unique."""
        d_ms = Timeframe.D1.ms
        mapped: dict[str, tuple[Ms, _DayRow]] = {}
        skipped_unmappable = 0
        for row in rows:
            ts_open = floor_bar(row.window_start_ns // _NS_PER_MS, Timeframe.D1)
            if ts_open + d_ms > now:
                continue  # in-progress bar: close is in the future — never serve it
            try:
                instrument_id = SymbolMapper.equity_instrument_id(
                    self._mic, _canonical_ticker(row.ticker)
                )
            except SchemaError:
                # A ticker that cannot form a valid equity id (e.g. one whose canonical
                # form ends in the synthetic 'USD' quote, like the 'USD' ETF). These are
                # rare and never common-stock factor targets; skip + count rather than
                # fail the whole day. A SYSTEMATIC mapping problem still fails loud via
                # the high-skip-fraction guard below.
                skipped_unmappable += 1
                continue
            if instrument_id in mapped:
                raise SchemaError(
                    f"duplicate (instrument_id, ts_open) within one day-aggs file: "
                    f"{instrument_id!r} at {ts_open} (vendor ticker {row.ticker!r}); "
                    "two tickers canonicalize to the same equity id"
                )
            mapped[instrument_id] = (ts_open, row)

        total = len(mapped) + skipped_unmappable
        if total and skipped_unmappable / total > _MAX_SKIP_FRAC:
            raise SchemaError(
                f"day-aggs: skipped {skipped_unmappable}/{total} unmappable tickers "
                f"(> {_MAX_SKIP_FRAC:.0%}) — a systematic id-mapping regression, not stray "
                "ETFs; refusing to silently shrink the universe"
            )
        if skipped_unmappable:
            _LOG.info(
                "day-aggs: skipped %d unmappable ticker(s); kept %d",
                skipped_unmappable,
                len(mapped),
            )

        ordered = sorted(mapped.items())  # sort by instrument_id; one ts_open per id
        table = pa.Table.from_pydict(
            {
                "instrument_id": [iid for iid, _ in ordered],
                "ts_open": [ts for _, (ts, _) in ordered],
                "open": [r.open for _, (_, r) in ordered],
                "high": [r.high for _, (_, r) in ordered],
                "low": [r.low for _, (_, r) in ordered],
                "close": [r.close for _, (_, r) in ordered],
                "volume": [r.volume for _, (_, r) in ordered],
                "quote_volume": [r.close * r.volume for _, (_, r) in ordered],
                "n_trades": [r.transactions for _, (_, r) in ordered],
                "quality_flags": [0] * len(ordered),
                "ingested_at": [ingested_at] * len(ordered),
            },
            schema=OHLCV_SCHEMA,
        )
        validate_table(table, Dataset.OHLCV_1D)
        return table

    def _month_prefixes(self, start: Ms, until: Ms) -> list[str]:
        """``YYYY/MM`` flat-files prefixes covering every UTC month in ``[start, until)``."""
        first = from_ms(start).date().replace(day=1)
        last = from_ms(until - 1).date()
        prefixes: list[str] = []
        cursor = first
        while cursor <= last:
            prefixes.append(f"{DAY_AGGS_PREFIX}/{cursor.year:04d}/{cursor.month:02d}")
            cursor = _next_month(cursor)
        return prefixes

    @staticmethod
    def session_open_ms(d: date) -> Ms:
        """UTC midnight of ``d`` as epoch ms — the day-aggs ``ts_open`` for a full session.

        The authoritative flat-files calendar mapping: the ingest job's day watermark and
        a bar's ``ts_open`` are both computed by this one rule.
        """
        return to_ms(datetime(d.year, d.month, d.day, tzinfo=UTC))

    @staticmethod
    def _date_from_key(key: str) -> date | None:
        """Parse ``.../YYYY-MM-DD.csv.gz`` → date; ``None`` for any non-matching key."""
        name = key.rsplit("/", 1)[-1]
        stem = name.removesuffix(".csv.gz")
        if stem == name:  # suffix not present
            return None
        try:
            return date.fromisoformat(stem)
        except ValueError:
            return None


def _next_month(d: date) -> date:
    """First day of the calendar month after ``d``."""
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)
