"""LakeWriter — validated, idempotent, crash-safe upserts into the Parquet lake.

Write pipeline per call (dataDesign.md §2.2/§2.4, §3.3):

1. **Validate** the incoming table against the declared schema
   (:func:`alphaforge.data.schemas.validate_table`) — nothing malformed ever
   reaches disk.
2. **Partial-bar guard** (OHLCV only; dataDesign.md §3.3 "belt and braces"): drop
   every row whose bar close ``ts_open + Δ`` lies after ``now`` — an in-progress
   bar's close mutates later, so persisting it would silently rewrite history.
   Dropped rows are counted in :attr:`WriteStats.unclosed_dropped`, never written.
3. **Group** rows by ``(instrument_id, year(key))`` — one leaf partition each.
4. Per leaf: read the existing file (if any) → concat → **dedupe** on the natural
   key keeping the row with the latest ``ingested_at`` (ties: the incoming row
   wins — a re-fetch of a finalized bar corrects an earlier bad read) → sort by
   key → write zstd Parquet to a same-directory tmp file → ``os.replace`` into
   place (atomic on POSIX, same filesystem).

Raw data is never mutated or "fixed" — the writer only merges whole rows; conflict
resolution is purely by recency of ingestion. Gaps are never synthesized.

Crash-safety invariant: a reader can only ever observe the previous complete file
or the new complete file, never a torn one. A failure after the tmp write leaves
the target untouched plus an inert ``.tmp-*`` file, removed by :meth:`LakeWriter.clean_tmp`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from alphaforge.core.time import Ms, Timeframe, now_ms
from alphaforge.data.schemas import (
    OHLCV_DATASETS,
    Dataset,
    empty_table,
    schema_for,
    validate_table,
)
from alphaforge.data.store.lake import LakePaths

__all__ = [
    "NATURAL_KEY_COLUMN",
    "LakeWriter",
    "WriteStats",
]

NATURAL_KEY_COLUMN: Final[dict[Dataset, str]] = {
    Dataset.OHLCV: "ts_open",
    Dataset.OHLCV_4H: "ts_open",
    Dataset.OHLCV_1D: "ts_open",
    Dataset.FUNDING: "ts_funding",
    Dataset.CORPORATE_ACTIONS: "ex_date",
    Dataset.FUNDAMENTALS: "period_end",
    Dataset.UNIVERSE_MEMBERSHIP: "effective_from",
}
"""Per-dataset dedupe key column. Within a leaf partition ``instrument_id`` is
constant, so the full natural key ``(instrument_id, <key>)`` reduces to this single
timestamp column; it is also the within-file sort key."""

_INGESTED_AT: Final[str] = "ingested_at"
_ORIGIN_COL: Final[str] = "__af_origin"  # 0 = existing on disk, 1 = incoming
_SEQ_COL: Final[str] = "__af_seq"  # row position; breaks exact ties deterministically
_ZSTD_LEVEL: Final[int] = 3


def _epoch_ms(column: pa.ChunkedArray) -> npt.NDArray[np.int64]:
    """``timestamp[ms, UTC]`` column → int64 epoch-ms numpy array (exact, no floats)."""
    raw = column.to_numpy(zero_copy_only=False)  # ChunkedArray.to_numpy is untyped (Any)
    out: npt.NDArray[np.int64] = raw.astype("datetime64[ms]").astype(np.int64)
    return out


def _utc_year(epoch_ms: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """UTC calendar year per epoch-ms value (datetime64 floors toward -inf, so this
    is total over pre-epoch timestamps too)."""
    return epoch_ms.astype("datetime64[ms]").astype("datetime64[Y]").astype(np.int64) + 1970


@dataclass(frozen=True)
class WriteStats:
    """Accounting for one :meth:`LakeWriter.write` call.

    - ``rows_in``: rows in the incoming table, before any guard or dedupe.
    - ``rows_written``: total physical rows across all rewritten leaf files
      (existing rows merged with incoming, post-dedupe).
    - ``rows_deduped``: rows discarded by natural-key dedupe across all leaves
      (= duplicates among incoming rows + existing rows superseded/refreshed by
      incoming ones). An exact re-write of ``n`` already-stored rows yields
      ``rows_deduped == n``.
    - ``partitions``: number of leaf files atomically rewritten.
    - ``unclosed_dropped``: OHLCV rows dropped by the partial-bar guard
      (``ts_open + Δ > now``); always 0 for other datasets.
    """

    rows_in: int
    rows_written: int
    rows_deduped: int
    partitions: int
    unclosed_dropped: int


class LakeWriter:
    """Idempotent merge-writer for the Parquet lake (see module docstring).

    Stateless apart from the injected :class:`LakePaths`; safe to construct per
    job. Not safe for concurrent writes to the same leaf from multiple processes
    (last replace wins) — ingestion is single-writer by design (dataDesign.md §2.1).
    """

    def __init__(self, paths: LakePaths) -> None:
        self._paths = paths

    def write(
        self,
        dataset: Dataset,
        tbl: pa.Table,
        *,
        tf: Timeframe = Timeframe.H1,
        now: Ms | None = None,
    ) -> WriteStats:
        """Upsert ``tbl`` into the lake; returns :class:`WriteStats`.

        ``tf`` is the bar timeframe the table carries (ingestion is 1h-only, hence
        the H1 default; the resampler passes ``tf=H4``/``D1`` when writing the
        derived OHLCV datasets); it parameterizes the OHLCV partial-bar guard only.
        ``now`` is the guard's reference wall clock in epoch ms UTC (injectable
        for tests/replays; defaults to :func:`alphaforge.core.time.now_ms`). A bar
        closing exactly at ``now`` is final and is kept.

        Raises :class:`~alphaforge.core.errors.SchemaError` if ``tbl`` does not
        conform to the declared schema — validation happens before anything else,
        and a failed call leaves the lake untouched.
        """
        validate_table(tbl, dataset)
        now_val = now_ms() if now is None else now
        rows_in = tbl.num_rows
        tbl = tbl.cast(schema_for(dataset))

        unclosed_dropped = 0
        if dataset in OHLCV_DATASETS:
            close_ms = _epoch_ms(tbl.column("ts_open")) + tf.ms
            tbl = tbl.filter(pa.array(close_ms <= now_val))
            unclosed_dropped = rows_in - tbl.num_rows

        key_col = NATURAL_KEY_COLUMN[dataset]
        groups = self._group_by_leaf(tbl, key_col)

        rows_written = 0
        rows_deduped = 0
        for (instrument_id, year), indices in sorted(groups.items()):
            incoming = tbl.take(pa.array(indices, type=pa.int64()))
            target = self._paths.partition_path(dataset, instrument_id, year)
            merged, discarded = self._merge_leaf(dataset, key_col, incoming, target)
            self._atomic_write(merged, target)
            rows_written += merged.num_rows
            rows_deduped += discarded

        return WriteStats(
            rows_in=rows_in,
            rows_written=rows_written,
            rows_deduped=rows_deduped,
            partitions=len(groups),
            unclosed_dropped=unclosed_dropped,
        )

    def clean_tmp(self) -> int:
        """Delete leftover in-flight ``.tmp-*`` files anywhere under the lake root.

        Safe to run at any time (readers never open tmp files; the writer names
        each tmp uniquely). Returns the number of files removed.
        """
        removed = 0
        for path in self._paths.tmp_files():
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    @staticmethod
    def _group_by_leaf(tbl: pa.Table, key_col: str) -> dict[tuple[str, int], list[int]]:
        """Row indices grouped by leaf partition key ``(instrument_id, UTC year of key)``."""
        instrument_ids = tbl.column("instrument_id").to_pylist()
        years = _utc_year(_epoch_ms(tbl.column(key_col)))
        groups: dict[tuple[str, int], list[int]] = {}
        for row, (instrument_id, year) in enumerate(
            zip(instrument_ids, years.tolist(), strict=True)
        ):
            groups.setdefault((instrument_id, year), []).append(row)
        return groups

    def _merge_leaf(
        self, dataset: Dataset, key_col: str, incoming: pa.Table, target: Path
    ) -> tuple[pa.Table, int]:
        """Merge ``incoming`` with the existing leaf file at ``target``.

        Returns ``(merged, discarded)``: the deduped, key-sorted table conforming
        exactly to the declared schema, and the number of rows dropped by dedupe.

        Conflict resolution per key: latest ``ingested_at`` wins; on an exact
        ``ingested_at`` tie the incoming row beats the stored row, and among tied
        incoming rows the later row in batch order wins (fully deterministic).
        ``universe_membership`` has no ``ingested_at`` column, so there the
        origin/batch-order rule alone decides (incoming always supersedes).
        """
        schema = schema_for(dataset)
        existing: pa.Table
        if target.is_file():
            # ParquetFile reads the single leaf directly — pq.read_table would run
            # dataset/hive discovery over the key=value path segments and clash
            # with the in-file instrument_id column (pyarrow >= 24 behavior).
            with pq.ParquetFile(target) as leaf:  # type: ignore[no-untyped-call]
                existing = leaf.read().cast(schema)
        else:
            existing = empty_table(dataset)

        combined = pa.concat_tables(
            [
                existing.append_column(_ORIGIN_COL, pa.array([0] * existing.num_rows, pa.int8())),
                incoming.append_column(_ORIGIN_COL, pa.array([1] * incoming.num_rows, pa.int8())),
            ]
        )
        combined = combined.append_column(_SEQ_COL, pa.array(range(combined.num_rows), pa.int64()))

        sort_keys: list[tuple[str, str]] = [(key_col, "ascending")]
        if _INGESTED_AT in schema.names:
            sort_keys.append((_INGESTED_AT, "ascending"))
        sort_keys += [(_ORIGIN_COL, "ascending"), (_SEQ_COL, "ascending")]
        ordered = combined.sort_by(sort_keys)

        keys = _epoch_ms(ordered.column(key_col))
        keep = np.ones(len(keys), dtype=bool)
        if len(keys) > 1:
            keep[:-1] = keys[:-1] != keys[1:]  # keep only the LAST row of each key run
        deduped = ordered.filter(pa.array(keep))

        merged = deduped.select(list(schema.names)).cast(schema)
        return merged, combined.num_rows - merged.num_rows

    def _atomic_write(self, tbl: pa.Table, target: Path) -> None:
        """Write ``tbl`` as zstd Parquet via the same-directory tmp + replace protocol.

        On failure between the tmp write and the replace, ``target`` is untouched
        and the tmp file is left behind for :meth:`clean_tmp` — never half-written
        data under the ``data.parquet`` name.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._paths.tmp_path(target)
        pq.write_table(  # type: ignore[no-untyped-call]
            tbl, tmp, compression="zstd", compression_level=_ZSTD_LEVEL
        )
        os.replace(tmp, target)
