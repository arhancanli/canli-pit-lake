"""PITDataReader — point-in-time reads over the Parquet lake via DuckDB.

THE PIT RULE LIVES HERE AND ONLY HERE (dataDesign.md §3.1): every analytical read
takes an explicit ``as_of`` (epoch ms UTC) and returns only records *available* at
that instant. There is deliberately no ``as_of=now`` default — callers must state
their information time, which is what makes backtest and live reads identical code.

Availability semantics enforced per dataset:

- **OHLCV**: a bar labeled ``ts_open`` covers ``[ts_open, ts_open + Δ)`` and is
  available at its close — visible iff ``ts_open + Δ <= as_of``. ``available_at``
  is derived here, never stored (dataDesign.md §2.3).
- **Quality flags** (leakage finding 5): each :class:`~alphaforge.data.schemas.QualityFlag`
  bit ``b`` carries its own lag ``L_b = FLAG_AVAILABILITY_LAG_BARS[b]`` and is
  exposed on a row only if ``ts_open + (1 + L_b)·Δ <= as_of``; bits not yet
  available at ``as_of`` are CLEARED in the returned ``quality_flags`` (the stored
  file is untouched). A bit with an undeclared lag is never exposed — fail-closed.
- **Funding** (leakage finding 18): visible iff the stored ``available_at <= as_of``;
  consumers must never join on ``ts_funding``.

Storage remains plain Parquet files owned by the writer; the in-memory DuckDB
connection here is a *query engine only*, never a storage owner (dataDesign.md §0.6).
Readers open only ``data.parquet`` leaves, so in-flight ``.tmp-*`` files are
structurally invisible. Missing partitions (or an empty lake) yield schema-stable
empty tables, not errors. ``end > as_of`` is allowed — the PIT filter governs; and
an ``as_of`` in the caller's future is their business (backtests replay history).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import duckdb
import pyarrow as pa

from alphaforge.core.calendar import calendar_for
from alphaforge.core.time import Ms, Timeframe, from_ms
from alphaforge.core.types import AssetClass
from alphaforge.data.schemas import (
    FLAG_AVAILABILITY_LAG_BARS,
    Dataset,
    empty_table,
    ohlcv_dataset,
    schema_for,
)
from alphaforge.data.store.lake import LakePaths

__all__ = ["PITDataReader"]

_MIN_YEAR: Final[int] = 1000  # lake_relpath's valid year range (4-digit years)
_MAX_YEAR: Final[int] = 9999


def _sql_file_list(files: Sequence[Path]) -> str:
    """Render ``files`` as a DuckDB SQL list literal for ``read_parquet([...])``.

    Single quotes are doubled (standard SQL escaping); paths come from
    :class:`LakePaths`, so this is defense in depth, not user-input handling.
    """
    quoted = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return f"[{quoted}]"


def _flag_mask_sql(tf: Timeframe, as_of: Ms) -> str:
    """SQL expression for the per-row visible-quality-bits mask (finding 5).

    Bits are grouped by declared lag ``L``: the group is visible on a row iff
    ``epoch_ms(ts_open) + (1 + L)·Δ <= as_of``. The returned expression is ORed
    bit-groups; ANDing it with ``quality_flags`` clears every bit that a decision
    at ``as_of`` could not yet have observed. Undeclared bits are absent from the
    mask and therefore always cleared (fail-closed). ``tf``/``as_of`` are trusted
    integers and inlined.
    """
    bits_by_lag: dict[int, int] = {}
    for flag, lag in FLAG_AVAILABILITY_LAG_BARS.items():
        bits_by_lag[lag] = bits_by_lag.get(lag, 0) | int(flag)
    terms = [
        f"(CASE WHEN epoch_ms(ts_open) + {(1 + lag) * tf.ms} <= {as_of} THEN {bits} ELSE 0 END)"
        for lag, bits in sorted(bits_by_lag.items())
    ]
    return " | ".join(terms) if terms else "0"


class PITDataReader:
    """Point-in-time reads over the lake; one in-memory DuckDB connection per instance.

    Partition pruning happens in Python via :class:`LakePaths` (explicit per-
    instrument, per-year file lists), so DuckDB only ever scans relevant leaves.
    Not thread-safe (DuckDB connections are not); construct one per thread.
    """

    def __init__(self, paths: LakePaths) -> None:
        self._paths = paths
        self._con = duckdb.connect()  # in-memory: pure query engine, owns no storage
        self._con.execute("SET TimeZone = 'UTC'")

    @property
    def lake_paths(self) -> LakePaths:
        """The exact lake path authority consumed by this reader.

        Exposed read-only so measured runs can freeze the partitions selected by
        this reader instead of reconstructing a possibly different path from config.
        """
        return self._paths

    def ohlcv(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        as_of: Ms,
        tf: Timeframe = Timeframe.H1,
    ) -> pa.Table:
        """Bars with ``start <= ts_open < end`` AND ``ts_open + tf.ms <= as_of``.

        ``tf`` selects the dataset (``H1`` → native ``ohlcv``; ``H4``/``D1`` → the
        resampler-derived ``ohlcv_4h``/``ohlcv_1d``) AND parameterizes the PIT
        delta — a derived bar labeled ``ts_open`` is available at ``ts_open +
        tf.ms`` (the close of its *window*), exactly like a native bar, and its
        quality-flag bit lags are measured in ``tf`` bars.

        All bounds are epoch ms UTC; ``[start, end)`` is half-open on ``ts_open``.
        A bar closing exactly at ``as_of`` is visible; one millisecond earlier it
        is not. Returned ``quality_flags`` are PIT-masked per bit (see module
        docstring) — the lake itself is never modified. Result conforms exactly to
        ``OHLCV_SCHEMA``, sorted by ``(instrument_id, ts_open)``; empty result for
        missing instruments/partitions or an empty interval.
        """
        dataset = ohlcv_dataset(tf)
        if end <= start or not instrument_ids:
            return empty_table(dataset)
        files = self._files(dataset, instrument_ids, start=start, end=end)
        if not files:
            return empty_table(dataset)
        sql = f"""
            SELECT instrument_id, ts_open, open, high, low, close, volume,
                   quote_volume, n_trades,
                   CAST(quality_flags & ({_flag_mask_sql(tf, as_of)}) AS INTEGER)
                       AS quality_flags,
                   ingested_at
            FROM read_parquet({_sql_file_list(files)})
            WHERE list_contains(?::VARCHAR[], instrument_id)
              AND epoch_ms(ts_open) >= ?
              AND epoch_ms(ts_open) < ?
              AND epoch_ms(ts_open) + ? <= ?
            ORDER BY instrument_id, ts_open
        """
        params = [list(instrument_ids), start, end, tf.ms, as_of]
        result: pa.Table = self._con.execute(sql, params).to_arrow_table()
        return result.cast(schema_for(dataset))

    def funding(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        as_of: Ms,
    ) -> pa.Table:
        """Funding rows with ``start <= ts_funding < end`` AND ``available_at <= as_of``.

        Bounds in epoch ms UTC. The PIT predicate uses the stored ``available_at``
        (settlement + publication lag) — never ``ts_funding`` (leakage finding 18):
        a settlement whose ``available_at`` is one millisecond after ``as_of`` is
        invisible. Result conforms exactly to ``FUNDING_SCHEMA``, sorted by
        ``(instrument_id, ts_funding)``.
        """
        dataset = Dataset.FUNDING
        if end <= start or not instrument_ids:
            return empty_table(dataset)
        files = self._files(dataset, instrument_ids, start=start, end=end)
        if not files:
            return empty_table(dataset)
        sql = f"""
            SELECT instrument_id, ts_funding, rate, available_at, ingested_at
            FROM read_parquet({_sql_file_list(files)})
            WHERE list_contains(?::VARCHAR[], instrument_id)
              AND epoch_ms(ts_funding) >= ?
              AND epoch_ms(ts_funding) < ?
              AND epoch_ms(available_at) <= ?
            ORDER BY instrument_id, ts_funding
        """
        params = [list(instrument_ids), start, end, as_of]
        result: pa.Table = self._con.execute(sql, params).to_arrow_table()
        return result.cast(schema_for(dataset))

    def corporate_actions(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        as_of: Ms,
    ) -> pa.Table:
        """Corp-action rows with ``start <= ex_date < end`` AND ``available_at <= as_of``.

        The equity analogue of :meth:`funding` — same PIT shape, splits/dividends
        instead of perp settlements. Bounds in epoch ms UTC. The PIT predicate uses the
        stored ``available_at`` (declaration/knowable date + publication lag) — NEVER
        ``ex_date`` (leakage finding 18): a split declared one millisecond after
        ``as_of`` is invisible, exactly as a settlement is. ``ex_date`` is used only as
        the within-range window key (the schema natural sort/dedupe key), so the served
        rows are the actions whose ex-date falls in ``[start, end)`` and which were
        knowable by ``as_of``. Result conforms exactly to ``CORPORATE_ACTIONS_SCHEMA``,
        sorted by ``(instrument_id, ex_date)``; empty result for missing
        instruments/partitions or an empty interval.
        """
        dataset = Dataset.CORPORATE_ACTIONS
        if end <= start or not instrument_ids:
            return empty_table(dataset)
        files = self._files(dataset, instrument_ids, start=start, end=end)
        if not files:
            return empty_table(dataset)
        sql = f"""
            SELECT instrument_id, action_type, ex_date, available_at, ratio,
                   cash_amount, ingested_at
            FROM read_parquet({_sql_file_list(files)})
            WHERE list_contains(?::VARCHAR[], instrument_id)
              AND epoch_ms(ex_date) >= ?
              AND epoch_ms(ex_date) < ?
              AND epoch_ms(available_at) <= ?
            ORDER BY instrument_id, ex_date
        """
        params = [list(instrument_ids), start, end, as_of]
        result: pa.Table = self._con.execute(sql, params).to_arrow_table()
        return result.cast(schema_for(dataset))

    def fundamentals(
        self,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        as_of: Ms,
    ) -> pa.Table:
        """Fundamentals rows with ``start <= period_end < end`` AND ``available_at <= as_of``.

        The same PIT shape as :meth:`corporate_actions` — quarterly financial-statement
        line items instead of splits/dividends. The PIT predicate uses the stored
        ``available_at`` (SEC ``filing_date``) — NEVER ``period_end`` (a quarter ending
        March is not knowable until its ~May filing, so reading it on a March decision is
        lookahead). ``period_end`` is only the within-range window key (the schema natural
        sort/dedupe key + partition year). Result conforms exactly to
        ``FUNDAMENTALS_SCHEMA``, sorted by ``(instrument_id, period_end)``; empty for
        missing instruments/partitions or an empty interval (crypto perps have no
        fundamentals partitions, so the read is empty — byte-identical).
        """
        dataset = Dataset.FUNDAMENTALS
        if end <= start or not instrument_ids:
            return empty_table(dataset)
        files = self._files(dataset, instrument_ids, start=start, end=end)
        if not files:
            return empty_table(dataset)
        # SELECT * with union_by_name so heterogeneous fundamentals partitions read together:
        # OLD 14-column files (written before the cash-flow/share fields were added) and NEW
        # 22-column ones. Schema evolution then pads any schema column an old partition lacked
        # as null before the cast — old data missing a new nullable field reads as null, never
        # an error. New columns must therefore NOT be hard-coded in the SELECT (the bug this
        # replaced: a fixed column list silently lags the schema).
        sql = f"""
            SELECT * FROM read_parquet({_sql_file_list(files)}, union_by_name=true)
            WHERE list_contains(?::VARCHAR[], instrument_id)
              AND epoch_ms(period_end) >= ?
              AND epoch_ms(period_end) < ?
              AND epoch_ms(available_at) <= ?
            ORDER BY instrument_id, period_end
        """
        params = [list(instrument_ids), start, end, as_of]
        result: pa.Table = self._con.execute(sql, params).to_arrow_table()
        target = schema_for(dataset)
        for field in target:
            if field.name not in result.column_names:
                result = result.append_column(field.name, pa.nulls(result.num_rows, field.type))
        return result.select(target.names).cast(target)

    def latest_bar_open(
        self,
        instrument_id: str,
        *,
        as_of: Ms,
        tf: Timeframe = Timeframe.H1,
    ) -> Ms | None:
        """Open of the freshest fully-available bar: max ``ts_open`` with close ``<= as_of``.

        The live loop's freshness probe — compare against the expected last close
        to detect a stale feed. ``tf`` selects the dataset (native 1h or derived
        4h/1d) and the close delta. Returns ``None`` when no bar of
        ``instrument_id`` is available at ``as_of`` (epoch ms UTC) or the
        instrument has no data.
        """
        year_hi = min(from_ms(as_of).year, _MAX_YEAR)
        files = self._paths.partition_paths(ohlcv_dataset(tf), instrument_id, year_max=year_hi)
        if not files:
            return None
        sql = f"""
            SELECT max(epoch_ms(ts_open))
            FROM read_parquet({_sql_file_list(files)})
            WHERE instrument_id = ? AND epoch_ms(ts_open) + ? <= ?
        """
        row = self._con.execute(sql, [instrument_id, tf.ms, as_of]).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def gaps(
        self,
        instrument_id: str,
        *,
        start: Ms,
        end: Ms,
        tf: Timeframe = Timeframe.H1,
        asset_class: AssetClass = AssetClass.CRYPTO_PERP,
    ) -> list[Ms]:
        """Expected ``tf`` bar opens in ``[start, end)`` missing from the lake, ascending.

        Operational tool (gap detection / backfill targeting): compares the calendar
        grid (:meth:`~alphaforge.core.calendar.TradingCalendar.expected_bar_opens` of
        ``calendar_for(asset_class)``) against the *stored* opens of the ``tf``-resolved
        dataset (native 1h or derived 4h/1d) — deliberately NO PIT filter, since ops
        must see the lake as it physically is. Gaps are reported, never synthetically
        filled (dataDesign.md §0.5); note the resampler's completeness rule makes a 1h
        gap propagate to a *missing* derived bar, which this tool surfaces. With no data
        at all, the entire grid is returned.

        ``asset_class`` selects the calendar (default ``CRYPTO_PERP`` ⇒ the 24/7 grid,
        byte-identical to the bare ``core.time`` kernel used before the calendar was
        threaded); the equity path passes ``AssetClass.EQUITY`` so weekends/holidays
        are correctly absent from the expected grid and not reported as gaps (audit G8).
        """
        expected = calendar_for(asset_class).expected_bar_opens(start, end, tf)
        if not expected:
            return []
        files = self._files(ohlcv_dataset(tf), [instrument_id], start=start, end=end)
        if not files:
            return expected
        sql = f"""
            SELECT DISTINCT epoch_ms(ts_open)
            FROM read_parquet({_sql_file_list(files)})
            WHERE instrument_id = ? AND epoch_ms(ts_open) >= ? AND epoch_ms(ts_open) < ?
        """
        rows = self._con.execute(sql, [instrument_id, start, end]).fetchall()
        present = {int(r[0]) for r in rows}
        return [ts for ts in expected if ts not in present]

    def _files(
        self, dataset: Dataset, instrument_ids: Sequence[str], *, start: Ms, end: Ms
    ) -> list[Path]:
        """Existing leaf files possibly holding rows with key in ``[start, end)``.

        Year pruning uses the UTC years of ``start`` and ``end - 1`` (the key
        columns are UTC timestamps, partitions are UTC years), clamped to the
        4-digit partition range. Duplicate instrument ids contribute once.
        """
        year_lo = max(from_ms(start).year, _MIN_YEAR)
        year_hi = min(from_ms(end - 1).year, _MAX_YEAR)
        files: list[Path] = []
        for instrument_id in dict.fromkeys(instrument_ids):
            files.extend(
                self._paths.partition_paths(
                    dataset, instrument_id, year_min=year_lo, year_max=year_hi
                )
            )
        return files
