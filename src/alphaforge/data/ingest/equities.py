"""``EquitiesFlatFilesJob`` — resumable, idempotent daily-bar ingest from Polygon S3 flat files.

This is the equities analogue of :class:`~alphaforge.data.ingest.backfill.BackfillJob`, but
its unit of work is one trading **day** (the day-aggs panel file holding every ticker that
traded), NOT a per-(instrument, dataset) walk. Reusing ``BackfillJob`` would re-download the
whole day file once per ticker; the day file is the natural batch + checkpoint unit
(EQUITIES_INGEST.md §1.5).

Durability loop, per session day (mirrors ``BackfillJob`` exactly):

    fetch_day → LakeWriter.write (atomic) → CheckpointStore.set (day watermark) → next

A day's watermark is advanced **only after** the write returns. A crash anywhere resumes
from the first not-yet-checkpointed session; an overlap re-fetch dedupes in the writer.

Checkpoint keying (EQUITIES_INGEST.md §1.5 choice A): the flat-files unit is a *day*, not an
instrument, so a SINGLE reserved key ``(Dataset.OHLCV_1D, _FLATFILES_WATERMARK_ID)`` stores
the ms of the last fully-written session day's open. Resume = skip every available session
whose ``session_open_ms <= watermark``. Monotonic + crash-safe for free, reusing the existing
``CheckpointStore`` with ZERO schema change. The reserved id contains no ``:``/``=``/NUL so it
never collides with a real canonical id and never reaches ``lake_relpath`` (the day watermark
is a checkpoint row only; nothing is ever partitioned under it).

Survivorship: the day file is survivorship-free by construction (it contains delisted names
too). This job preserves that end-to-end — a delisted ticker present in an old day-file lands
in the lake and is NEVER silently dropped (EQUITIES_INGEST.md §4.3, the headline gate).

Lake stays RAW: ``fetch_day`` never adjusts prices. Split/dividend (total-return) adjustment
is reconstructed point-in-time at *feature* time, never at ingest (EQUITIES_INGEST.md §2).
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from alphaforge.core.errors import LockHeldError
from alphaforge.core.logging import get_logger
from alphaforge.core.time import Ms, Timeframe, now_ms
from alphaforge.data.ingest.backfill import ResultStatus
from alphaforge.data.schemas import Dataset

if TYPE_CHECKING:
    from alphaforge.data.ingest.checkpoints import CheckpointStore
    from alphaforge.data.sources.polygon_flatfiles import PolygonFlatFilesSource
    from alphaforge.data.store.writer import LakeWriter

__all__ = [
    "EquitiesFlatFilesJob",
    "EquitiesIngestReport",
    "EquitiesIngestResult",
]

_log = get_logger(__name__)

_FLATFILES_WATERMARK_ID: Final[str] = "__polygon_flatfiles_day_watermark__"
"""Reserved checkpoint id for the single flat-files day watermark (EQUITIES_INGEST.md §1.5).

No ``:``/``=``/NUL → it never collides with a real canonical id and never reaches
``lake_relpath`` (it is a checkpoint row only; no partition is ever written under it)."""

_FLATFILES_DATASET: Final[Dataset] = Dataset.OHLCV_1D
"""Dataset the flat-files day watermark is keyed under (daily equity bars)."""


def _max_ts_open(tbl: pa.Table) -> Ms:
    """Max ``ts_open`` (epoch ms UTC) in a non-empty OHLCV table."""
    raw = tbl.column("ts_open").to_numpy(zero_copy_only=False)  # ChunkedArray.to_numpy is Any
    arr: npt.NDArray[np.int64] = raw.astype("datetime64[ms]").astype(np.int64)
    return int(arr.max())


@dataclass(frozen=True, slots=True, kw_only=True)
class EquitiesIngestResult:
    """Outcome of ingesting ONE session day.

    - ``session``: ``"YYYY-MM-DD"`` session date.
    - ``status``: ``"ok"`` (written or empty), ``"failed"`` (an exception aborted this
      day — other days were unaffected), or ``"skipped"`` (already checkpointed: resume).
    - ``tickers``: distinct instrument ids written this day (0 for skipped/failed).
    - ``rows``: rows handed to the writer (post ``max_tickers`` cap + closed-bar guard).
    - ``error``: ``repr`` of the exception for a failed day, else ``""``.
    """

    session: str
    status: ResultStatus
    tickers: int
    rows: int
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class EquitiesIngestReport:
    """Outcome of one :meth:`EquitiesFlatFilesJob.run` call.

    ``run_id`` keys the row written to the :class:`CheckpointStore` run log;
    ``started``/``finished`` are wall-clock epoch ms UTC; ``results`` holds one
    :class:`EquitiesIngestResult` per processed session day, in ascending date order.
    """

    run_id: str
    started: Ms
    finished: Ms
    results: tuple[EquitiesIngestResult, ...]

    @property
    def ok_count(self) -> int:
        """Number of session days that processed cleanly."""
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed_count(self) -> int:
        """Number of session days aborted by an exception (isolated; run continued)."""
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def skipped_count(self) -> int:
        """Number of session days skipped because they were already checkpointed."""
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def total_rows(self) -> int:
        """Total rows durably handed to the writer across all days."""
        return sum(r.rows for r in self.results)

    def failures(self) -> tuple[EquitiesIngestResult, ...]:
        """The failed results only, in processing order (operator triage list)."""
        return tuple(r for r in self.results if r.status == "failed")

    def summary(self) -> str:
        """One-line human summary, also stored as the run-log ``detail``."""
        return (
            f"ok={self.ok_count} failed={self.failed_count} "
            f"skipped={self.skipped_count} rows={self.total_rows}"
        )


class EquitiesFlatFilesJob:
    """Resumable, idempotent daily-bar ingest from the Polygon S3 flat files.

    Args:
        source: The flat-files bar source (one day → one panel table).
        writer: The lake writer; its atomic replace + dedupe makes resume overlap and
            re-fetch free.
        checkpoints: Watermark + run-log store. The day watermark is advanced only after
            a write returns (the durability contract in ``checkpoints.py``).
        max_tickers: Optional dev/smoke cap — keep the top-N tickers by ``quote_volume``
            that day (deterministic ``instrument_id`` tie-break). NON-survivorship-safe
            for research (it changes membership); applied AFTER parsing the full file so a
            later uncapped run simply adds rows (the writer dedupes — the lake is never
            poisoned). ``None`` (default) keeps every ticker.
        lock_path: OS-level exclusive ingest lock held for the duration of :meth:`run`.
            Pass the SAME ``var/ingest.lock`` the crypto backfill uses, so an equities
            ingest and a crypto backfill can never race the lake. ``None`` disables
            locking (single-process/test use only).
    """

    def __init__(
        self,
        source: PolygonFlatFilesSource,
        writer: LakeWriter,
        checkpoints: CheckpointStore,
        *,
        max_tickers: int | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if max_tickers is not None and max_tickers < 1:
            raise ValueError(f"max_tickers must be >= 1 or None, got {max_tickers}")
        self._source = source
        self._writer = writer
        self._checkpoints = checkpoints
        self._max_tickers = max_tickers
        self._lock_path = lock_path

    def run(
        self, *, start: Ms, until: Ms | None = None, now: Ms | None = None
    ) -> EquitiesIngestReport:
        """Ingest every available session day in ``[start, until)``; resume-safe.

        Holds the exclusive ingest lock (so an equities ingest and a crypto backfill
        cannot race). Lists ``available_sessions``, skips any whose day-open is already
        ``<= watermark`` (resume), and processes the rest in date order. Failure of one
        day is isolated (status ``"failed"``) and the run continues; a ``BaseException``
        (e.g. ``KeyboardInterrupt``) stamps the run-log row failed and propagates after
        the in-flight day's durability point.

        Args:
            start: First session, epoch ms UTC (inclusive on the session-day open).
            until: Exclusive upper bound on the session-day open; ``None`` and anything
                in the future clamp to ``now``.
            now: Injectable wall clock (epoch ms UTC) for tests/replays; ``None`` reads
                :func:`~alphaforge.core.time.now_ms` once.

        Raises:
            ValueError: ``start < 0``.
        """
        if start < 0:
            raise ValueError(f"start must be epoch ms UTC >= 0, got {start}")
        now_eff: Ms = now_ms() if now is None else now
        until_eff: Ms = now_eff if until is None else min(until, now_eff)
        with self._exclusive_lock():
            return self._run_locked(start=start, until=until_eff, now=now_eff)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        """Hold the OS-level exclusive ingest lock (no-op when ``lock_path`` is None).

        Identical contract to ``BackfillJob._exclusive_lock``: non-blocking ``flock``;
        a held lock raises :class:`LockHeldError` immediately (the lock file records the
        holder pid). The OS releases it even on ``kill -9``, so a crashed run never wedges
        future ones. The SAME lock path the crypto backfill uses serializes both.
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

    def _run_locked(self, *, start: Ms, until: Ms, now: Ms) -> EquitiesIngestReport:
        """The body of :meth:`run`, executed under the exclusive ingest lock."""
        started: Ms = now_ms()
        run_id = self._checkpoints.start_run("ingest-equities:ohlcv_1d")
        watermark = self._checkpoints.get(_FLATFILES_DATASET, _FLATFILES_WATERMARK_ID)
        sessions = self._source.available_sessions(start=start, until=until)
        _log.info(
            "equities_ingest.run_started",
            run_id=run_id,
            start=start,
            until=until,
            sessions=len(sessions),
            watermark=watermark,
            max_tickers=self._max_tickers,
        )

        results: list[EquitiesIngestResult] = []
        try:
            for session in sessions:
                results.append(self._process_day(session, watermark=watermark, now=now))
        except BaseException:
            # KeyboardInterrupt / SystemExit / anything non-isolatable: the last completed
            # write already advanced the watermark (durability point); stamp the run-log
            # row failed and let the interrupt propagate.
            self._checkpoints.finish_run(run_id, "failed", detail="aborted by BaseException")
            raise

        report = EquitiesIngestReport(
            run_id=run_id,
            started=started,
            finished=now_ms(),
            results=tuple(results),
        )
        self._checkpoints.finish_run(
            run_id, "ok" if report.failed_count == 0 else "failed", detail=report.summary()
        )
        _log.info("equities_ingest.run_finished", run_id=run_id, summary=report.summary())
        return report

    def _process_day(self, session: date, *, watermark: Ms | None, now: Ms) -> EquitiesIngestResult:
        """Ingest one session day (or skip it if already checkpointed)."""
        label = session.isoformat()
        session_open = self._source.session_open_ms(session)
        if watermark is not None and session_open <= watermark:
            return EquitiesIngestResult(session=label, status="skipped", tickers=0, rows=0)
        try:
            table = self._source.fetch_day(session, now=now)
            if self._max_tickers is not None:
                table = _cap_top_tickers(table, self._max_tickers)
            tickers = table.num_rows
            if tickers > 0:
                self._writer.write(_FLATFILES_DATASET, table, tf=Timeframe.D1, now=now)
                self._checkpoints.set(_FLATFILES_DATASET, _FLATFILES_WATERMARK_ID, session_open)
            _log.info(
                "equities_ingest.session_done",
                session=label,
                tickers=tickers,
                rows=tickers,
            )
            return EquitiesIngestResult(session=label, status="ok", tickers=tickers, rows=tickers)
        except Exception as exc:
            _log.error("equities_ingest.session_failed", session=label, error=repr(exc))
            return EquitiesIngestResult(
                session=label, status="failed", tickers=0, rows=0, error=repr(exc)
            )


def _cap_top_tickers(table: pa.Table, max_tickers: int) -> pa.Table:
    """Keep the top-``max_tickers`` rows by ``quote_volume`` (id tie-break, ascending).

    Deterministic dev/smoke cap (EQUITIES_INGEST.md §1.5): applied AFTER parsing the full
    file, so a later uncapped run simply adds the remaining rows (the writer dedupes — the
    lake is never poisoned). NON-survivorship-safe for research; never the default.
    """
    if table.num_rows <= max_tickers:
        return table
    ordered = table.sort_by([("quote_volume", "descending"), ("instrument_id", "ascending")])
    kept = ordered.slice(0, max_tickers)
    # Restore the source's (instrument_id, ts_open) ordering the writer expects.
    return kept.sort_by([("instrument_id", "ascending"), ("ts_open", "ascending")])
