"""UniverseStore — membership-interval persistence over the ``universe_membership`` dataset.

Storage is the ordinary Parquet lake (``UNIVERSE_SCHEMA``; natural key
``(instrument_id, effective_from)``), written through :class:`LakeWriter` so the
tmp-write + atomic-replace crash-safety protocol and schema validation apply
unchanged (dataDesign.md §2.2, §6).

PIT semantics of this table (dataDesign.md §6.2; leakage finding 1):

- A membership interval is *knowable* at its ``effective_from``: the builder computes
  each rebalance decision exclusively from data available at the rebalance instant, so
  the span ``[effective_from, effective_to)`` was computable by a live trader the
  moment it began. Research therefore joins positions/features against
  :meth:`membership_asof` (or interval-overlap joins on the effective span) — that join
  is inherently point-in-time, including for instruments delisted long before today.
- A REBUILD rewrites the whole history at once (full replace — see
  :meth:`write_intervals`). That is sound *because* the table is derived data: it is a
  deterministic pure function of the OHLCV lake plus the SCD2 instrument record, and
  those upstream sources are the audit trail. There is no ``ingested_at`` column here
  (``UNIVERSE_SCHEMA`` carries ``rank``/``reason`` instead); recency conflicts on a
  re-written key resolve to the incoming row inside :class:`LakeWriter`.
- Survivorship guarantee in miniature: intervals of delisted instruments are ordinary
  rows — nothing in this store ever deletes a dead symbol's history except a full
  replace that regenerates it from the (never-deleted) raw bars.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from alphaforge.data.schemas import Dataset, empty_table, schema_for, validate_table
from alphaforge.data.store.writer import LakeWriter, WriteStats

if TYPE_CHECKING:
    from pathlib import Path

    from alphaforge.core.time import Ms
    from alphaforge.data.store.lake import LakePaths

__all__ = ["UniverseStore"]

_LEAF_GLOB = "instrument_id=*/year=*/data.parquet"


class UniverseStore:
    """Read/write access to ``universe_membership`` intervals in the lake.

    Stateless apart from the injected :class:`LakePaths`; constructs its own
    :class:`LakeWriter` so callers wire one object. Reads open only ``data.parquet``
    leaves (in-flight ``.tmp-*`` files are structurally invisible) and are
    deterministic: leaves are listed sorted, rows returned sorted by
    ``(instrument_id, effective_from)``.
    """

    def __init__(self, paths: LakePaths) -> None:
        self._paths = paths
        self._writer = LakeWriter(paths)
        # Membership intervals are immutable within a process (only the universe
        # builder writes them; readers — backtest, live loop — only read). Reading
        # globs+parses thousands of parquet leaves, and market/breadth features call
        # membership_asof once PER BAR (hundreds of times per cycle), so an uncached
        # read turned the residual-reversal factors into a multi-minute stall. Cache
        # the parsed full table + the (ids, froms, tos) arrays once; invalidate on
        # write. Pure memoization of a deterministic read — results are byte-identical.
        self._intervals_cache: pa.Table | None = None
        self._parsed_cache: tuple[list[str], list[int], list[int | None]] | None = None

    def write_intervals(
        self,
        tbl: pa.Table,
        *,
        replace: bool = True,
        scope_ids: frozenset[str] | None = None,
    ) -> WriteStats:
        """Persist membership intervals; returns the writer's :class:`WriteStats`.

        ``tbl`` must conform to ``UNIVERSE_SCHEMA`` (validated *before* anything is
        touched on disk — a failed call leaves the lake unchanged, raising
        :class:`~alphaforge.core.errors.SchemaError`).

        - ``replace=True``, ``scope_ids=None`` (default, single-sleeve rebuild): the
          entire ``universe_membership`` dataset directory is removed first, then
          ``tbl`` is written. Full-replace is the correct semantic for a single-sleeve
          rebuild because the table is **derived** — a pure function of the OHLCV lake
          + instrument record; stale intervals from a previous parameterization must
          not survive a regeneration (the lake, not this table, is the source of truth).
        - ``replace=True``, ``scope_ids`` given (**per-sleeve / cross-asset rebuild**):
          remove ONLY the partitions whose ``instrument_id`` is in ``scope_ids``, then
          write ``tbl``. ``scope_ids`` is the full instrument-id set of the sleeve being
          rebuilt (every name the SCD2 record knows for that asset class — including
          delisted ones, so stale spans of names that left the universe are still
          cleared). Other sleeves' partitions are left untouched, so a crypto rebuild no
          longer wipes the equity universe and the two coexist in one lake — the
          foundation of the unified cross-asset book. The derived-data guarantee holds
          per sleeve: each sleeve's spans are a pure function of its own lake slice.
        - ``replace=False`` (incremental, e.g. the live loop closing a delisting):
          merge-upsert by natural key ``(instrument_id, effective_from)`` — an
          incoming row supersedes the stored row with the same key (so closing an
          open interval rewrites it with ``effective_to`` set).
        """
        validate_table(tbl, Dataset.UNIVERSE_MEMBERSHIP)
        # any write changes membership -> invalidate the read caches
        self._intervals_cache = None
        self._parsed_cache = None
        if replace:
            dataset_dir = self._paths.root / Dataset.UNIVERSE_MEMBERSHIP.value
            if dataset_dir.is_dir():
                if scope_ids is None:
                    shutil.rmtree(dataset_dir)
                else:
                    _prefix = "instrument_id="
                    for part_dir in dataset_dir.iterdir():
                        if (
                            part_dir.is_dir()
                            and part_dir.name.startswith(_prefix)
                            and part_dir.name.removeprefix(_prefix) in scope_ids
                        ):
                            shutil.rmtree(part_dir)
        return self._writer.write(Dataset.UNIVERSE_MEMBERSHIP, tbl)

    def read_intervals(self, as_of: Ms | None = None) -> pa.Table:
        """All membership intervals, sorted by ``(instrument_id, effective_from)``.

        ``as_of`` (epoch ms UTC) applies the knowability filter: only intervals with
        ``effective_from <= as_of`` are returned — a membership span is knowable from
        the instant it takes effect, because the builder computed it from data
        available strictly at that instant (see module docstring). ``as_of=None``
        returns the full stored history (research/ops convenience; per-timestamp
        joins must still use the effective span, which is what makes them PIT).

        Result conforms exactly to ``UNIVERSE_SCHEMA``; an empty/missing dataset
        yields a schema-stable empty table, not an error.
        """
        tbl = self._intervals_cache
        if tbl is None:
            tbl = self._read_intervals_full()
            self._intervals_cache = tbl
        if as_of is not None:
            from_ms_vals: list[int] = tbl.column("effective_from").cast(pa.int64()).to_pylist()
            mask = pa.array([v <= as_of for v in from_ms_vals], type=pa.bool_())
            tbl = tbl.filter(mask)
        return tbl.cast(schema_for(Dataset.UNIVERSE_MEMBERSHIP))

    def _read_intervals_full(self) -> pa.Table:
        """Glob+parse every membership leaf into the full sorted table (no as_of).

        Cached by :meth:`read_intervals`; this is the only path that touches disk.
        """
        dataset_dir = self._paths.root / Dataset.UNIVERSE_MEMBERSHIP.value
        files: list[Path] = sorted(dataset_dir.glob(_LEAF_GLOB)) if dataset_dir.is_dir() else []
        if not files:
            return empty_table(Dataset.UNIVERSE_MEMBERSHIP)
        schema = schema_for(Dataset.UNIVERSE_MEMBERSHIP)
        parts: list[pa.Table] = []
        for path in files:
            # ParquetFile reads the single leaf directly — pq.read_table would run
            # hive discovery over the key=value path segments and clash with the
            # in-file instrument_id column (same rationale as LakeWriter._merge_leaf).
            with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
                parts.append(leaf.read().cast(schema))
        return pa.concat_tables(parts).sort_by(
            [("instrument_id", "ascending"), ("effective_from", "ascending")]
        )

    def membership_asof(self, ts: Ms) -> frozenset[str]:
        """The tradable universe at decision instant ``ts`` (epoch ms UTC).

        Half-open interval semantics (``UNIVERSE_SCHEMA``): ``instrument_id`` is a
        member iff some interval satisfies ``effective_from <= ts`` and
        ``effective_to`` is null or ``> ts``. An instrument is a member at the exact
        rebalance instant it enters, and a non-member at the exact instant its
        interval closes (delisting/exit). Delisted symbols appear for every ``ts``
        inside their historical spans — the survivorship-bias-free read.
        """
        parsed = self._parsed_cache
        if parsed is None:
            tbl = self.read_intervals()
            parsed = (
                tbl.column("instrument_id").to_pylist(),
                tbl.column("effective_from").cast(pa.int64()).to_pylist(),
                tbl.column("effective_to").cast(pa.int64()).to_pylist(),
            )
            self._parsed_cache = parsed
        ids, froms, tos = parsed
        return frozenset(
            iid
            for iid, eff_from, eff_to in zip(ids, froms, tos, strict=True)
            if eff_from <= ts and (eff_to is None or eff_to > ts)
        )
