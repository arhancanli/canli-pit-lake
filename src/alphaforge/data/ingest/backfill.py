"""BackfillJob — resumable, idempotent, crash-safe ingestion into the Parquet lake.

One engine, two entrypoints (dataDesign.md §4.3): :meth:`BackfillJob.run` is the
historical backfill; :meth:`BackfillJob.update` is the hourly incremental path —
literally ``run()`` with ``start_default`` derived from the stored watermarks. Live
and historical data flow through identical code.

Routing per instrument (leakage finding 1 — the survivorship fix end-to-end):

- **LIVE** instruments (``delisted_ts is None``) fetch from the venue's REST API via
  the injected :class:`~alphaforge.data.sources.base.DataSource`, in chunks of
  ``batch_days`` days.
- **DELISTED** instruments (``delisted_ts`` set — including the HISTORY-ONLY records
  seeded from the Binance Vision archive) fetch from
  :class:`~alphaforge.data.sources.vision.BinanceVisionClient` monthly files, since
  the REST API no longer serves them. Whole months covering the span are fetched
  and rows outside the span are trimmed before writing.

Durability loop (the crash-safety contract):

    fetch one batch → ``LakeWriter.write`` (atomic) → ``CheckpointStore.set`` → next

The watermark is advanced **only after** the write returns, to the max natural-key
timestamp (``ts_open`` / ``ts_funding``) durably written. A crash anywhere leaves the
watermark at the last durable batch; rerunning resumes from ``watermark + 1`` and any
overlap re-fetch dedupes in the writer (keep latest ``ingested_at``). A
``KeyboardInterrupt`` therefore propagates only *between* durability points: the only
long-blocking call per iteration is the fetch, and nothing is suppressed or retried
between a completed write and its watermark advance — interrupting a fetch loses
nothing durable, and the atomic writer makes an interrupt mid-write torn-file-free.

Temporal contract (dataDesign.md §0, non-negotiable): all timestamps crossing this
interface are integer epoch milliseconds UTC (:data:`~alphaforge.core.time.Ms`); bars
are labeled ``ts_open``; no bar whose close is in the future is ever written (the
sources drop unclosed bars, this job re-applies the guard before computing watermarks,
and the writer guards independently). Gaps are never synthetically filled: a Vision
month that 404s inside a delisted instrument's claimed range is logged, recorded in
the report, and skipped — never fabricated.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from alphaforge.core.errors import DataGapError, LockHeldError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.logging import get_logger
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms, Timeframe, from_ms, now_ms
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.ingest.retry import RateBudget
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import BinanceVisionClient, YearMonth
from alphaforge.data.store.writer import NATURAL_KEY_COLUMN, LakeWriter

__all__ = [
    "BackfillJob",
    "BackfillReport",
    "InstrumentResult",
    "ResultStatus",
]

_log = get_logger(__name__)

ResultStatus = Literal["ok", "failed", "skipped"]
"""Per-(instrument, dataset) outcome states recorded in :class:`InstrumentResult`."""

_DAY_MS: Final[int] = 86_400_000
"""One UTC day in milliseconds (used to convert ``batch_days`` to a cursor stride)."""

_INGESTABLE: Final[frozenset[Dataset]] = frozenset({Dataset.OHLCV, Dataset.FUNDING})
"""Datasets this job can ingest. ``UNIVERSE_MEMBERSHIP`` is *derived* (Phase 3
universe builder), never fetched from a vendor — requesting it is a caller bug."""

_VISION_MONTH_WEIGHT: Final[int] = 1
"""Rate-budget weight charged per Binance Vision monthly-archive download."""


def _epoch_ms(tbl: pa.Table, column: str) -> npt.NDArray[np.int64]:
    """``timestamp[ms, UTC]`` column of ``tbl`` → int64 epoch-ms array (exact)."""
    raw = tbl.column(column).to_numpy(zero_copy_only=False)  # ChunkedArray.to_numpy is Any
    out: npt.NDArray[np.int64] = raw.astype("datetime64[ms]").astype(np.int64)
    return out


def _trim(tbl: pa.Table, key_column: str, start: Ms, until: Ms) -> pa.Table:
    """Rows of ``tbl`` whose ``key_column`` lies in the half-open span ``[start, until)``."""
    if tbl.num_rows == 0:
        return tbl
    keys = _epoch_ms(tbl, key_column)
    return tbl.filter(pa.array((keys >= start) & (keys < until)))


def _next_month(ym: YearMonth) -> YearMonth:
    """The calendar month immediately after ``ym``."""
    if ym.month == 12:
        return YearMonth(ym.year + 1, 1)
    return YearMonth(ym.year, ym.month + 1)


def _months_covering(start: Ms, until: Ms) -> list[YearMonth]:
    """All UTC calendar months whose window intersects ``[start, until)``, ascending.

    A month's window is ``[first_ms, next_month_first_ms)``; the result is exactly the
    set of monthly archive files that can contain a record with key-timestamp in the
    span. Empty when ``until <= start``.
    """
    if until <= start:
        return []
    first = from_ms(start)
    ym = YearMonth(first.year, first.month)
    out: list[YearMonth] = []
    while ym.first_ms() < until:
        out.append(ym)
        ym = _next_month(ym)
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class InstrumentResult:
    """Outcome of ingesting one ``(instrument, dataset)`` pair within a run.

    - ``status``: ``"ok"`` (span processed; Vision gap months allowed), ``"failed"``
      (an exception aborted this pair — other pairs were unaffected), or
      ``"skipped"`` (empty effective span: already up to date, or listed/delisted
      bounds exclude the request).
    - ``rows``: rows durably handed to the writer by THIS run (post trim/closed-bar
      guard; 0 for skipped and failed results — for failed pairs the watermark, not
      this counter, reflects what landed before the error).
    - ``batches``: successful write+watermark durability points executed.
    - ``watermark``: the pair's checkpoint after processing (epoch ms UTC), ``None``
      if nothing was ever durably ingested for it.
    - ``gap_months``: ``"YYYY-MM"`` labels of Vision months that 404'd inside a
      delisted instrument's claimed range (logged, never fatal, never synthesized).
    - ``error``: ``repr`` of the exception for failed results, else ``""``.
    """

    instrument_id: str
    dataset: Dataset
    status: ResultStatus
    rows: int
    batches: int
    watermark: Ms | None
    gap_months: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackfillReport:
    """Outcome of one :meth:`BackfillJob.run` / :meth:`BackfillJob.update` call.

    ``run_id`` keys the row written to the :class:`CheckpointStore` run log;
    ``started``/``finished`` are wall-clock epoch ms UTC; ``results`` holds one
    :class:`InstrumentResult` per processed ``(instrument, dataset)`` pair, in
    deterministic ``(instrument_id, dataset)`` processing order.
    """

    run_id: str
    job: str
    started: Ms
    finished: Ms
    results: tuple[InstrumentResult, ...]

    @property
    def ok_count(self) -> int:
        """Number of pairs that processed cleanly."""
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed_count(self) -> int:
        """Number of pairs aborted by an exception (isolated; run continued)."""
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def skipped_count(self) -> int:
        """Number of pairs with an empty effective span (already up to date)."""
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def total_rows(self) -> int:
        """Total rows durably handed to the writer across all pairs."""
        return sum(r.rows for r in self.results)

    def failures(self) -> tuple[InstrumentResult, ...]:
        """The failed results only, in processing order (operator triage list)."""
        return tuple(r for r in self.results if r.status == "failed")

    def summary(self) -> str:
        """One-line human summary, also stored as the run-log ``detail``."""
        return (
            f"ok={self.ok_count} failed={self.failed_count} "
            f"skipped={self.skipped_count} rows={self.total_rows}"
        )


class BackfillJob:
    """Resumable, idempotent OHLCV + funding ingestion (see module docstring).

    Args:
        live_source: Vendor adapter for LIVE instruments (e.g. ``CCXTDataSource``).
            It meters its own REST rate budget internally.
        vision: Binance Vision archive client for DELISTED instruments' history.
        writer: The lake writer; its atomic replace + dedupe is what makes resume
            overlap and re-fetch free.
        checkpoints: Watermark + run-log store. Watermarks are advanced only after
            a write returns (the durability contract in ``checkpoints.py``).
        instruments: SCD2 instrument store; the candidate set is
            ``instruments.all_known(as_of=now)`` so seeded HISTORY-ONLY delistings
            are backfilled too.
        rate_budget: Optional client-side budget charged
            :data:`_VISION_MONTH_WEIGHT` per Vision monthly download (politeness on
            the public bucket). ccxt REST calls are metered inside ``live_source``,
            not here — pass the same budget to both to share one global allowance.
        tf: Bar timeframe ingested into the lake (v1 lake is 1h-only).
    """

    def __init__(
        self,
        live_source: DataSource,
        vision: BinanceVisionClient,
        writer: LakeWriter,
        checkpoints: CheckpointStore,
        instruments: InstrumentStore,
        *,
        rate_budget: RateBudget | None = None,
        tf: Timeframe = Timeframe.H1,
        lock_path: Path | None = None,
    ) -> None:
        """``lock_path``: OS-level exclusive ingest lock taken for the duration of
        every :meth:`run`/:meth:`update`, so two ingestion processes can never race
        on the same lake + checkpoints (``LockHeldError`` instead). ``None`` skips
        locking — acceptable only for single-process embedded/test use; the CLI
        always passes ``var/ingest.lock``.
        """
        self._source = live_source
        self._vision = vision
        self._writer = writer
        self._checkpoints = checkpoints
        self._instruments = instruments
        self._budget = rate_budget
        self._tf = tf
        self._lock_path = lock_path

    # ------------------------------------------------------------------ entrypoints

    def run(
        self,
        *,
        datasets: Sequence[Dataset],
        instrument_ids: Sequence[str] | None = None,
        start_default: Ms,
        until: Ms | None = None,
        batch_days: int = 30,
        now: Ms | None = None,
        job: str = "backfill",
    ) -> BackfillReport:
        """Ingest ``datasets`` for the selected instruments; returns a report.

        Per ``(instrument, dataset)`` pair the effective span is computed as

        - ``start = max(watermark + 1, listed_ts, start_default)`` — resume from the
          checkpoint, never before listing, never before the requested default;
        - ``until_eff = min(until or now, delisted_ts or now)`` — never the future
          (an unclosed bar can never be served), never past delisting.

        LIVE instruments are fetched from the REST source in ``batch_days``-day
        chunks; DELISTED instruments from Vision monthly archives (rows trimmed to
        the span). Each batch is written atomically and the watermark advanced
        before the next fetch begins — the crash-resume durability loop. A fully
        walked, gap-free delisted instrument additionally converges its watermark
        to ``delisted_ts - 1``: the instrument is dead, no data can ever appear
        after it, so later runs skip it instead of re-polling the archive forever
        (months that 404'd block this convergence so the gap stays visible).

        Failure isolation: an exception in one pair is caught, logged, and recorded
        as a ``failed`` result; remaining pairs still run. ``KeyboardInterrupt``
        (and any other ``BaseException``) is NOT isolated — it marks the run-log row
        failed and propagates, after the in-flight write's durability point (see
        module docstring).

        Args:
            datasets: Non-empty, duplicate-free subset of ``{OHLCV, FUNDING}``,
                processed in the given order per instrument.
            instrument_ids: Canonical ids to ingest; ``None`` = every instrument
                known to the store at ``now``. Unknown ids raise ``ValueError``
                before any work starts.
            start_default: Historical start (epoch ms UTC) for pairs with no
                watermark (and the floor for all others).
            until: Exclusive upper bound on key timestamps (epoch ms UTC); ``None``
                and anything in the future clamp to ``now``.
            batch_days: REST chunk size in days (>= 1) — the durability cadence
                for LIVE instruments (Vision's cadence is one month per batch).
            now: Injectable wall clock (epoch ms UTC) for tests/replays; ``None``
                reads :func:`~alphaforge.core.time.now_ms` once.
            job: Run-log label prefix (``update()`` passes ``"update"``).

        Raises:
            ValueError: empty/unknown/duplicate ``datasets``, non-ingestable
                dataset, ``batch_days < 1``, ``start_default < 0``, or an empty /
                unknown ``instrument_ids`` selection.
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if len(set(datasets)) != len(datasets):
            raise ValueError(f"datasets contains duplicates: {[d.value for d in datasets]}")
        bad = [d.value for d in datasets if d not in _INGESTABLE]
        if bad:
            raise ValueError(
                f"datasets {bad} cannot be ingested from a vendor; ingestable: "
                f"{sorted(d.value for d in _INGESTABLE)}"
            )
        if batch_days < 1:
            raise ValueError(f"batch_days must be >= 1, got {batch_days}")
        if start_default < 0:
            raise ValueError(f"start_default must be epoch ms UTC >= 0, got {start_default}")

        now_eff: Ms = now_ms() if now is None else now
        selected = self._select_instruments(instrument_ids, now=now_eff)
        batch_ms = batch_days * _DAY_MS

        with self._exclusive_lock():
            return self._run_locked(
                datasets=datasets,
                selected=selected,
                start_default=start_default,
                until=until,
                batch_days=batch_days,
                batch_ms=batch_ms,
                now_eff=now_eff,
                job=job,
            )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        """Hold the OS-level exclusive ingest lock (no-op when ``lock_path`` is None).

        ``flock`` with ``LOCK_NB``: if any other process (or another job object in
        this process) holds the lock, raise :class:`LockHeldError` immediately
        rather than queueing — a queued second backfill would just re-walk what the
        first one is writing. The lock file records the holder's pid for operators;
        the lock itself is released by the OS even on ``kill -9``, so a crashed run
        never wedges future ones.
        """
        if self._lock_path is None:
            yield
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                holder = ""
                with suppress(OSError):  # diagnostic best-effort only
                    holder = os.read(fd, 64).decode(errors="replace").strip()
                raise LockHeldError(
                    f"ingest lock {self._lock_path} is held"
                    f"{f' by pid {holder}' if holder else ''}; "
                    "another ingestion run is in progress (never run two — "
                    "they would race on the same lake and checkpoints)"
                ) from exc
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            yield
        finally:
            os.close(fd)

    def _run_locked(
        self,
        *,
        datasets: Sequence[Dataset],
        selected: Sequence[Instrument],
        start_default: Ms,
        until: Ms | None,
        batch_days: int,
        batch_ms: int,
        now_eff: Ms,
        job: str,
    ) -> BackfillReport:
        """The body of :meth:`run`, executed under the exclusive ingest lock."""
        job_label = f"{job}:{','.join(d.value for d in datasets)}"
        started: Ms = now_ms()
        run_id = self._checkpoints.start_run(job_label)
        _log.info(
            "backfill.run_started",
            run_id=run_id,
            job=job_label,
            instruments=len(selected),
            start_default=start_default,
            until=until,
            batch_days=batch_days,
        )

        results: list[InstrumentResult] = []
        try:
            for inst in selected:
                for dataset in datasets:
                    try:
                        result = self._process(
                            dataset,
                            inst,
                            start_default=start_default,
                            until_arg=until,
                            batch_ms=batch_ms,
                            now=now_eff,
                        )
                    except Exception as exc:
                        result = InstrumentResult(
                            instrument_id=inst.instrument_id,
                            dataset=dataset,
                            status="failed",
                            rows=0,
                            batches=0,
                            watermark=self._checkpoints.get(dataset, inst.instrument_id),
                            error=repr(exc),
                        )
                        _log.error(
                            "backfill.instrument_failed",
                            run_id=run_id,
                            instrument_id=inst.instrument_id,
                            dataset=dataset.value,
                            error=repr(exc),
                        )
                    results.append(result)
        except BaseException:
            # KeyboardInterrupt / SystemExit / anything non-isolatable: the last
            # completed write already advanced its watermark (durability point);
            # stamp the run-log row failed and let the interrupt propagate.
            self._checkpoints.finish_run(run_id, "failed", detail="aborted by BaseException")
            raise

        report = BackfillReport(
            run_id=run_id,
            job=job_label,
            started=started,
            finished=now_ms(),
            results=tuple(results),
        )
        self._checkpoints.finish_run(
            run_id, "ok" if report.failed_count == 0 else "failed", detail=report.summary()
        )
        _log.info("backfill.run_finished", run_id=run_id, summary=report.summary())
        return report

    def update(
        self, *, as_of: Ms, instrument_ids: Sequence[str] | None = None
    ) -> BackfillReport:
        """Incremental ingestion: ``run()`` with ``start_default`` derived from watermarks.

        The hourly follow path — same code, no special cases. ``start_default`` is
        the minimum stored watermark across OHLCV and funding plus 1 ms, so every
        watermarked pair resumes exactly at its own ``watermark + 1`` (the per-pair
        ``max()`` dominates) and a never-ingested instrument backfills from its
        ``listed_ts`` (floored by that minimum). With no watermarks at all there is
        nothing to advance: ``start_default = as_of`` and every pair is skipped —
        bootstrap is :meth:`run`'s job, via ``af data backfill``.

        Args:
            as_of: The wall clock of this cycle (epoch ms UTC) — used as ``now``,
                as the exclusive ``until``, and as the instrument-store read time.
            instrument_ids: optional scope (default ``None`` = every known instrument,
                so ``af data update`` is unchanged). The live loop passes its
                asset-class-scoped ids so a single-asset loop (e.g. crypto) never tries
                to fetch a different asset class's instruments from the wrong source.
        """
        datasets = (Dataset.OHLCV, Dataset.FUNDING)
        marks: list[Ms] = []
        for dataset in datasets:
            marks.extend(self._checkpoints.all_watermarks(dataset).values())
        start_default: Ms = min(marks) + 1 if marks else as_of
        return self.run(
            datasets=datasets,
            instrument_ids=instrument_ids,
            start_default=start_default,
            until=as_of,
            now=as_of,
            job="update",
        )

    # ------------------------------------------------------------------ internals

    def _select_instruments(
        self, instrument_ids: Sequence[str] | None, *, now: Ms
    ) -> list[Instrument]:
        """Resolve the candidate set from the SCD2 store as of ``now``, sorted by id.

        ``None`` selects every known instrument (live and delisted alike — the
        survivorship guarantee). An explicit selection must be non-empty and fully
        known, else ``ValueError`` — silently ingesting a subset of what the caller
        asked for would hide typos forever.
        """
        known = self._instruments.all_known(as_of=now)
        if instrument_ids is None:
            return known
        if not instrument_ids:
            raise ValueError("instrument_ids must be None (= all known) or non-empty")
        by_id = {inst.instrument_id: inst for inst in known}
        unknown = sorted(set(instrument_ids) - set(by_id))
        if unknown:
            raise ValueError(
                f"unknown instrument_ids (not in the instrument store as of {now}): {unknown}; "
                "run `af instruments refresh` first"
            )
        return [by_id[iid] for iid in sorted(set(instrument_ids))]

    def _process(
        self,
        dataset: Dataset,
        inst: Instrument,
        *,
        start_default: Ms,
        until_arg: Ms | None,
        batch_ms: int,
        now: Ms,
    ) -> InstrumentResult:
        """Ingest one ``(instrument, dataset)`` pair over its effective span."""
        instrument_id = inst.instrument_id
        watermark = self._checkpoints.get(dataset, instrument_id)

        start = max(start_default, inst.listed_ts)
        if watermark is not None:
            start = max(start, watermark + 1)
        until = now if until_arg is None else min(until_arg, now)
        if inst.delisted_ts is not None:
            until = min(until, inst.delisted_ts)

        if until <= start:
            return InstrumentResult(
                instrument_id=instrument_id,
                dataset=dataset,
                status="skipped",
                rows=0,
                batches=0,
                watermark=watermark,
            )

        gap_months: tuple[str, ...] = ()
        if inst.delisted_ts is None:
            rows, batches, watermark = self._ingest_live(
                dataset, instrument_id, start=start, until=until, batch_ms=batch_ms, now=now
            )
        else:
            rows, batches, watermark, gap_months = self._ingest_vision(
                dataset,
                inst,
                start=start,
                until=until,
                now=now,
                prior_watermark=watermark,
            )

        _log.info(
            "backfill.instrument_done",
            instrument_id=instrument_id,
            dataset=dataset.value,
            rows=rows,
            batches=batches,
            watermark=watermark,
            gap_months=list(gap_months),
        )
        return InstrumentResult(
            instrument_id=instrument_id,
            dataset=dataset,
            status="ok",
            rows=rows,
            batches=batches,
            watermark=watermark,
            gap_months=gap_months,
        )

    def _ingest_live(
        self,
        dataset: Dataset,
        instrument_id: str,
        *,
        start: Ms,
        until: Ms,
        batch_ms: int,
        now: Ms,
    ) -> tuple[int, int, Ms | None]:
        """REST path for live instruments: walk ``[start, until)`` in batch chunks.

        Returns ``(rows, batches, watermark)``. The cursor advances over empty
        chunks without moving the watermark — the watermark asserts durability of
        *data*, and an empty venue response is not data; a resumed run re-walks the
        empty span cheaply rather than ever overstating durability.
        """
        rows_total = 0
        batches = 0
        watermark: Ms | None = None
        cursor = start
        while cursor < until:
            batch_end = min(cursor + batch_ms, until)
            if dataset is Dataset.OHLCV:
                tbl = self._source.fetch_ohlcv(
                    instrument_id, tf=self._tf, since=cursor, until=batch_end, now=now
                )
            else:
                tbl = self._source.fetch_funding(instrument_id, since=cursor, until=batch_end)
            rows, new_wm = self._write_batch(dataset, instrument_id, tbl, now=now)
            if new_wm is not None:
                rows_total += rows
                batches += 1
                watermark = new_wm
            cursor = batch_end
        if watermark is None:
            watermark = self._checkpoints.get(dataset, instrument_id)
        return rows_total, batches, watermark

    def _ingest_vision(
        self,
        dataset: Dataset,
        inst: Instrument,
        *,
        start: Ms,
        until: Ms,
        now: Ms,
        prior_watermark: Ms | None,
    ) -> tuple[int, int, Ms | None, tuple[str, ...]]:
        """Archive path for delisted instruments: one monthly file per batch.

        Whole months covering ``[start, until)`` are fetched; rows outside the span
        are trimmed before writing (already-durable rows below ``start`` and
        post-``until`` rows never reach the writer). A month that 404s inside the
        claimed range is a warning + report entry, never fatal (the archive simply
        has no file — raw data is never synthesized).

        Watermark convergence: when the walk covered the instrument's full
        remaining life (``until == delisted_ts``) with no gap months, the watermark
        is advanced to ``delisted_ts - 1``. The instrument is dead — no data with a
        key before ``delisted_ts`` can ever appear that this walk did not see — so
        the watermark contract ("everything with key <= W is durably in the lake")
        holds and later runs skip the instrument instead of re-downloading its tail
        month forever. Gap months block convergence so the instrument stays visible
        in every report (its tail re-walks each run). A gap month *beyond* the
        watermark is naturally retried on the next run; one already behind it (a
        later month wrote and advanced the checkpoint) is permanent archive absence
        as far as resume math is concerned — re-ingesting it after the archive
        gains the file requires the operator escape hatch ``checkpoints.reset``.
        """
        _, _, symbol = SymbolMapper.parse_instrument_id(inst.instrument_id)
        rows_total = 0
        batches = 0
        watermark = prior_watermark
        gaps: list[str] = []
        key_column = NATURAL_KEY_COLUMN[dataset]

        for ym in _months_covering(start, until):
            if self._budget is not None:
                self._budget.acquire(_VISION_MONTH_WEIGHT)
            try:
                if dataset is Dataset.OHLCV:
                    tbl = self._vision.fetch_ohlcv_month(symbol, ym.year, ym.month, now=now)
                else:
                    tbl = self._vision.fetch_funding_month(symbol, ym.year, ym.month)
            except DataGapError as exc:
                label = f"{ym.year:04d}-{ym.month:02d}"
                gaps.append(label)
                _log.warning(
                    "backfill.vision_month_missing",
                    instrument_id=inst.instrument_id,
                    dataset=dataset.value,
                    month=label,
                    error=str(exc),
                )
                continue
            tbl = _trim(tbl, key_column, start, until)
            rows, new_wm = self._write_batch(dataset, inst.instrument_id, tbl, now=now)
            if new_wm is not None:
                rows_total += rows
                batches += 1
                watermark = new_wm

        if not gaps and until == inst.delisted_ts:
            converged: Ms = until - 1
            if watermark is None or converged > watermark:
                self._checkpoints.set(dataset, inst.instrument_id, converged)
                watermark = converged
        return rows_total, batches, watermark, tuple(gaps)

    def _write_batch(
        self, dataset: Dataset, instrument_id: str, tbl: pa.Table, *, now: Ms
    ) -> tuple[int, Ms | None]:
        """One durability point: atomic write, then watermark advance.

        Belt and braces: OHLCV rows whose close ``ts_open + Δ`` lies after ``now``
        are dropped HERE as well (sources already guarantee it with the same
        ``now``), so the watermark — computed from this exact table — can never
        run ahead of what the writer's own partial-bar guard would persist.
        Empty batches are no-ops: ``(0, None)``, no write, no watermark movement.
        """
        if dataset is Dataset.OHLCV and tbl.num_rows > 0:
            closes = _epoch_ms(tbl, "ts_open") + self._tf.ms
            tbl = tbl.filter(pa.array(closes <= now))
        if tbl.num_rows == 0:
            return 0, None
        self._writer.write(dataset, tbl, tf=self._tf, now=now)
        new_watermark = int(_epoch_ms(tbl, NATURAL_KEY_COLUMN[dataset]).max())
        self._checkpoints.set(dataset, instrument_id, new_watermark)
        return tbl.num_rows, new_watermark
