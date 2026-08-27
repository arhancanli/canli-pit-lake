"""Resampler — derive 4h/1d bars from stored 1h bars (dataDesign.md §2.4/§4.4).

Derived bars are lake datasets like any other (``ohlcv_4h``/``ohlcv_1d``; see
:func:`alphaforge.data.schemas.ohlcv_dataset`): written through the same validated,
atomic :class:`~alphaforge.data.store.writer.LakeWriter`, read back through
``PITDataReader.ohlcv(tf=H4/D1)`` under the identical PIT rule — a derived bar
labeled ``ts_open`` is available at ``ts_open + tf.ms``, the close of its window.
They are **always computed from stored 1h bars, never fetched** — one source of
truth, guaranteed consistency.

Aggregation per target window ``[T, T + target.ms)`` over its 1h constituents ordered by
``ts_open`` (dataDesign.md §4.4):

- ``open`` = first ``open``; ``close`` = last ``close``;
- ``high`` = max ``high``; ``low`` = min ``low``;
- ``volume``/``quote_volume`` = sum (``math.fsum``, exactly rounded);
- ``quote_volume``/``n_trades`` are **null if ANY constituent is null** — a partial
  sum would silently understate, so nullability propagates instead;
- ``quality_flags`` = bitwise OR — a derived bar inherits EVERY defect of its parts;
- ``ingested_at`` = the run's ``now`` (audit only, never a feature input).

COMPLETENESS RULE — read this before touching anything: a target bucket is emitted
ONLY if (a) **all** expected source bars are present (4 for H4, 24 for D1) and
(b) the bucket is **closed**, ``bucket_open + target.ms <= now``. Buckets failing
either test are SKIPPED — counted in :class:`ResampleStats`, never written, never
guessed. Raw data is never fixed and gaps are never synthetically filled
(dataDesign.md §0.5): a gap in the 1h series therefore propagates to a *missing*
4h/1d bar, exactly what ``PITDataReader.gaps(tf=H4/D1)``-style tooling surfaces.

Incremental contract: each run only recomputes buckets strictly after the last
derived bar's bucket, so a re-run with no new 1h data reads nothing and writes
nothing (idempotent via the writer's dedupe anyway — this just skips the work).
Caveat, documented loudly: if a *historical* 1h gap older than the derived
watermark is later backfilled, the incremental run will NOT revisit it; re-derive
by deleting the affected derived-dataset partitions and re-running.

Source bars are read **physically** (no PIT filter, no flag masking): resampling is
lake maintenance, and the derived ``quality_flags`` must OR the *stored* source
bits — masked reads would launder defects out of derived bars (leakage finding 5 is
enforced at read time by the PIT reader, on derived bars exactly as on native ones).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from alphaforge.core.time import Ms, Timeframe, floor_bar, from_ms, now_ms
from alphaforge.data.schemas import Dataset, ohlcv_dataset, schema_for
from alphaforge.data.store.writer import LakeWriter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphaforge.data.store.lake import LakePaths

__all__ = [
    "ResampleBatch",
    "ResampleStats",
    "Resampler",
    "resample_bars",
]


@dataclass(frozen=True)
class ResampleStats:
    """Accounting for one :meth:`Resampler.resample` run (single target timeframe).

    - ``target``: the derived timeframe produced.
    - ``instruments``: distinct instrument ids processed.
    - ``source_rows``: 1h bars read (only bars after each instrument's incremental
      watermark; 0 on a no-new-data re-run).
    - ``bars_written``: derived bars emitted to the writer. The incremental rule
      starts strictly after the last derived bar, so every emitted bar is new.
    - ``buckets_incomplete``: closed buckets skipped because at least one expected
      source bar was missing (the completeness rule) — these become *missing*
      derived bars, never partial ones.
    - ``buckets_unclosed``: buckets skipped because ``bucket_open + target.ms >
      now``. A bucket that is both unclosed and incomplete counts here only.
    """

    target: Timeframe
    instruments: int
    source_rows: int
    bars_written: int
    buckets_incomplete: int
    buckets_unclosed: int


@dataclass(frozen=True)
class ResampleBatch:
    """Result of one pure :func:`resample_bars` call.

    ``table`` conforms to the OHLCV column contract, sorted by ``(instrument_id,
    ts_open)``; ``buckets_incomplete``/``buckets_unclosed`` count skipped buckets
    with the same semantics as :class:`ResampleStats`.
    """

    table: pa.Table
    buckets_incomplete: int
    buckets_unclosed: int


def _bucket_grid(bucket_open: Ms, source: Timeframe, target: Timeframe) -> list[Ms]:
    """The exact source-bar opens a complete ``target`` bucket must contain."""
    return list(range(bucket_open, bucket_open + target.ms, source.ms))


def resample_bars(
    bars: pa.Table,
    *,
    source: Timeframe,
    target: Timeframe,
    now: Ms,
) -> ResampleBatch:
    """Pure aggregation of ``source``-timeframe bars into ``target`` buckets.

    ``bars`` must carry the OHLCV column contract with rows unique per
    ``(instrument_id, ts_open)`` and ``ts_open`` aligned to the ``source`` grid —
    the lake's invariants; misaligned rows simply render their bucket incomplete
    (fail-closed), never mis-bucketed. Buckets are aligned to 00:00 UTC by integer
    ``floor_bar`` arithmetic. The completeness and closed-bucket rules of the
    module docstring apply: a bucket is emitted only if its source opens equal the
    full expected grid AND ``bucket_open + target.ms <= now``; everything else is
    counted and skipped. Output ``ingested_at`` is ``now`` for every emitted bar.

    Raises :class:`ValueError` unless ``target`` is a strict integer multiple of
    ``source`` (1h→4h, 1h→1d, 4h→1d).
    """
    if target.ms <= source.ms or target.ms % source.ms != 0:
        raise ValueError(
            f"target {target.value!r} must be a strict integer multiple of "
            f"source {source.value!r} ({target.ms} ms vs {source.ms} ms)"
        )

    instrument_ids: list[str] = bars.column("instrument_id").to_pylist()
    ts_opens: list[int] = bars.column("ts_open").cast(pa.int64()).to_pylist()
    opens: list[float] = bars.column("open").to_pylist()
    highs: list[float] = bars.column("high").to_pylist()
    lows: list[float] = bars.column("low").to_pylist()
    closes: list[float] = bars.column("close").to_pylist()
    volumes: list[float] = bars.column("volume").to_pylist()
    quote_volumes: list[float | None] = bars.column("quote_volume").to_pylist()
    n_trades: list[int | None] = bars.column("n_trades").to_pylist()
    flags: list[int] = bars.column("quality_flags").to_pylist()

    buckets: dict[tuple[str, Ms], list[int]] = {}
    for row, (iid, ts) in enumerate(zip(instrument_ids, ts_opens, strict=True)):
        buckets.setdefault((iid, floor_bar(ts, target)), []).append(row)

    out: dict[str, list[object]] = {name: [] for name in schema_for(Dataset.OHLCV).names}
    incomplete = 0
    unclosed = 0
    for (iid, bucket_open), rows in sorted(buckets.items()):
        if bucket_open + target.ms > now:
            unclosed += 1  # in-progress window: its close would mutate later
            continue
        rows.sort(key=lambda r: ts_opens[r])
        if [ts_opens[r] for r in rows] != _bucket_grid(bucket_open, source, target):
            incomplete += 1  # gap (or misalignment) in constituents: emit NOTHING
            continue
        out["instrument_id"].append(iid)
        out["ts_open"].append(bucket_open)
        out["open"].append(opens[rows[0]])
        out["high"].append(max(highs[r] for r in rows))
        out["low"].append(min(lows[r] for r in rows))
        out["close"].append(closes[rows[-1]])
        out["volume"].append(math.fsum(volumes[r] for r in rows))
        bucket_qv = [v for r in rows if (v := quote_volumes[r]) is not None]
        out["quote_volume"].append(math.fsum(bucket_qv) if len(bucket_qv) == len(rows) else None)
        bucket_nt = [n for r in rows if (n := n_trades[r]) is not None]
        out["n_trades"].append(sum(bucket_nt) if len(bucket_nt) == len(rows) else None)
        composite = 0
        for r in rows:
            composite |= flags[r]
        out["quality_flags"].append(composite)
        out["ingested_at"].append(now)

    schema = schema_for(ohlcv_dataset(target))
    table = pa.Table.from_pydict(
        {name: pa.array(values, type=schema.field(name).type) for name, values in out.items()},
        schema=schema,
    )
    return ResampleBatch(table=table, buckets_incomplete=incomplete, buckets_unclosed=unclosed)


class Resampler:
    """Incremental 1h → 4h/1d derivation over the lake (see module docstring).

    Reads native ``ohlcv`` leaves physically via :class:`LakePaths`, aggregates
    with :func:`resample_bars`, and upserts the result through a
    :class:`LakeWriter` (validated, dedupe-idempotent, atomic). Stateless apart
    from the injected paths; single-writer per derived dataset by design, same as
    ingestion (dataDesign.md §2.1).
    """

    def __init__(self, paths: LakePaths) -> None:
        self._paths = paths
        self._writer = LakeWriter(paths)

    def resample(
        self,
        instrument_ids: Sequence[str],
        *,
        target: Timeframe,
        now: Ms | None = None,
    ) -> ResampleStats:
        """Derive ``target`` bars from stored 1h bars for ``instrument_ids``.

        ``target`` must be ``H4`` or ``D1`` (``ValueError`` otherwise — 1h is the
        native, ingested timeframe). ``now`` (epoch ms UTC; defaults to
        :func:`alphaforge.core.time.now_ms`) gates the closed-bucket rule and is
        stamped as ``ingested_at`` on every derived bar.

        Per instrument, only buckets strictly after the last already-derived bar's
        bucket are recomputed; with no new 1h data the run reads and writes
        nothing. The completeness rule applies per bucket: missing source bars ⇒
        the derived bar is skipped (counted in ``buckets_incomplete``) and stays
        MISSING in the lake — see the module docstring, including the caveat about
        historical gaps backfilled after the derived watermark has passed them.

        Duplicate instrument ids are processed once. Returns :class:`ResampleStats`.
        """
        if target not in (Timeframe.H4, Timeframe.D1):
            raise ValueError(
                f"resample target must be '4h' or '1d', got {target.value!r} "
                "(1h is the native ingested timeframe, never derived)"
            )
        now_val = now_ms() if now is None else now
        derived = ohlcv_dataset(target)

        source_rows = 0
        incomplete = 0
        unclosed = 0
        emitted: list[pa.Table] = []
        unique_ids = list(dict.fromkeys(instrument_ids))
        for instrument_id in unique_ids:
            last_open = self._last_derived_open(derived, instrument_id)
            start = None if last_open is None else last_open + target.ms
            source = self._read_source(instrument_id, start=start)
            source_rows += source.num_rows
            if source.num_rows == 0:
                continue
            batch = resample_bars(source, source=Timeframe.H1, target=target, now=now_val)
            incomplete += batch.buckets_incomplete
            unclosed += batch.buckets_unclosed
            if batch.table.num_rows > 0:
                emitted.append(batch.table)

        bars_written = 0
        if emitted:
            out = pa.concat_tables(emitted)
            self._writer.write(derived, out, tf=target, now=now_val)
            bars_written = out.num_rows

        return ResampleStats(
            target=target,
            instruments=len(unique_ids),
            source_rows=source_rows,
            bars_written=bars_written,
            buckets_incomplete=incomplete,
            buckets_unclosed=unclosed,
        )

    def _last_derived_open(self, dataset: Dataset, instrument_id: str) -> Ms | None:
        """Physical max ``ts_open`` already stored in ``dataset`` for the instrument.

        The incremental watermark — deliberately NOT a PIT read: it reflects the
        lake as it is on disk. ``None`` when the instrument has no derived bars yet.
        Leaves are sorted and unique on ``ts_open``, and years are scanned in
        order, so the latest year's max is the global max.
        """
        years = self._paths.years_for(dataset, instrument_id)
        if not years:
            return None
        path = self._paths.partition_path(dataset, instrument_id, years[-1])
        with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
            ts = leaf.read(columns=["ts_open"]).column("ts_open")
        if len(ts) == 0:
            return None
        opens: list[int] = ts.cast(pa.int64()).to_pylist()
        return max(opens)

    def _read_source(self, instrument_id: str, *, start: Ms | None) -> pa.Table:
        """All stored 1h bars of ``instrument_id`` with ``ts_open >= start`` (physical read).

        ``start=None`` reads everything. No PIT filter and no flag masking — the
        derived ``quality_flags`` must OR the *stored* source bits (module
        docstring); the closed-bucket rule downstream is what keeps lookahead out.
        Year partitions below ``start`` are pruned; missing partitions contribute
        nothing.
        """
        dataset = Dataset.OHLCV
        schema = schema_for(dataset)
        year_min = None if start is None else from_ms(start).year
        files = self._paths.partition_paths(dataset, instrument_id, year_min=year_min)
        tables: list[pa.Table] = []
        for path in files:
            # ParquetFile reads the single leaf directly — pq.read_table would run
            # hive discovery over the key=value path segments (see LakeWriter).
            with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
                tbl = leaf.read().cast(schema)
            if start is not None:
                ts = tbl.column("ts_open").cast(pa.int64()).to_pylist()
                tbl = tbl.filter(pa.array([t >= start for t in ts], type=pa.bool_()))
            if tbl.num_rows > 0:
                tables.append(tbl)
        if not tables:
            return schema.empty_table()
        return pa.concat_tables(tables)
