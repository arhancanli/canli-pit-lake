"""``FundamentalsJob`` — resumable, idempotent PIT fundamentals ingest (Polygon financials).

The fundamentals analogue of :class:`~alphaforge.data.ingest.backfill.BackfillJob`: its unit
of work is one **instrument** (the financials API is per-ticker, not a daily panel like the
flat-files bar ingest). For each instrument it fetches the full quarterly history in
``[since, until)`` and writes it to the :data:`~alphaforge.data.schemas.Dataset.FUNDAMENTALS`
lake table.

Durability loop, per instrument (mirrors the bar jobs exactly)::

    fetch_fundamentals -> LakeWriter.write (atomic) -> CheckpointStore.set (watermark) -> next

The per-instrument watermark is advanced **only after** the write returns; a crash resumes
from the first not-yet-checkpointed instrument, and an overlap re-fetch dedupes in the writer
(the lake is never poisoned). A single instrument's failure is isolated (status ``"failed"``)
and the run continues.

Survivorship: the caller passes the universe-EVER-members (every id that was ever in the
tradeable universe, including delisted names) — fundamentals are fetched for delisted tickers
too, so a backtest's value/quality factors see the same survivorship-free cross section the
price lake does. PIT: every row carries ``available_at`` (filing date); nothing here adjusts
or look-ahead-fills — the factor layer filters ``available_at <= decision_t`` (the contract
lives in :data:`~alphaforge.data.schemas.FUNDAMENTALS_SCHEMA`).
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from alphaforge.core.errors import LockHeldError
from alphaforge.core.logging import get_logger
from alphaforge.core.time import Ms, now_ms
from alphaforge.data.ingest.backfill import ResultStatus
from alphaforge.data.schemas import Dataset

if TYPE_CHECKING:
    from alphaforge.data.ingest.checkpoints import CheckpointStore
    from alphaforge.data.sources.polygon_source import PolygonEquitiesSource
    from alphaforge.data.store.writer import LakeWriter

__all__ = [
    "FundamentalsIngestReport",
    "FundamentalsIngestResult",
    "FundamentalsJob",
]

_log = get_logger(__name__)

_DATASET: Final[Dataset] = Dataset.FUNDAMENTALS


@dataclass(frozen=True, slots=True, kw_only=True)
class FundamentalsIngestResult:
    """Outcome of ingesting ONE instrument's fundamentals.

    - ``instrument_id``: canonical id.
    - ``status``: ``"ok"`` (written or empty), ``"failed"`` (an exception aborted this
      instrument — others were unaffected), or ``"skipped"`` (already checkpointed: resume).
    - ``rows``: quarterly rows handed to the writer (0 for skipped/failed).
    - ``error``: ``repr`` of the exception for a failed instrument, else ``""``.
    """

    instrument_id: str
    status: ResultStatus
    rows: int
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FundamentalsIngestReport:
    """Outcome of one :meth:`FundamentalsJob.run` call (one result per instrument)."""

    run_id: str
    started: Ms
    finished: Ms
    results: tuple[FundamentalsIngestResult, ...]

    @property
    def ok_count(self) -> int:
        """Instruments that processed cleanly."""
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed_count(self) -> int:
        """Instruments aborted by an exception (isolated; the run continued)."""
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def skipped_count(self) -> int:
        """Instruments skipped because already checkpointed (resume)."""
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def total_rows(self) -> int:
        """Total quarterly rows durably written across all instruments."""
        return sum(r.rows for r in self.results)

    def failures(self) -> tuple[FundamentalsIngestResult, ...]:
        """The failed results only (operator triage list)."""
        return tuple(r for r in self.results if r.status == "failed")

    def summary(self) -> str:
        """One-line human summary, also stored as the run-log ``detail``."""
        return (
            f"ok={self.ok_count} failed={self.failed_count} "
            f"skipped={self.skipped_count} rows={self.total_rows}"
        )


class FundamentalsJob:
    """Resumable, idempotent PIT fundamentals ingest (one instrument = one unit of work).

    Args:
        source: The Polygon equities source (``fetch_fundamentals`` per ticker).
        writer: The lake writer; its atomic replace + dedupe make resume overlap free.
        checkpoints: Watermark + run-log store. The per-instrument watermark is advanced
            only after a write returns (the durability contract).
        lock_path: OS-level exclusive ingest lock held for :meth:`run` (pass the SAME
            ``var/ingest.lock`` the bar ingests use, so a fundamentals ingest and a bar
            ingest can never race the lake). ``None`` disables locking (test use only).
    """

    def __init__(
        self,
        source: PolygonEquitiesSource,
        writer: LakeWriter,
        checkpoints: CheckpointStore,
        *,
        lock_path: Path | None = None,
    ) -> None:
        self._source = source
        self._writer = writer
        self._checkpoints = checkpoints
        self._lock_path = lock_path

    def run(
        self,
        instrument_ids: Sequence[str],
        *,
        since: Ms,
        until: Ms | None = None,
        now: Ms | None = None,
    ) -> FundamentalsIngestReport:
        """Ingest quarterly fundamentals for every id in ``instrument_ids``; resume-safe.

        Holds the exclusive ingest lock. An instrument whose watermark is already
        ``>= until`` is skipped (resume). One instrument's failure is isolated (status
        ``"failed"``); a ``BaseException`` stamps the run-log failed and propagates after
        the in-flight instrument's durability point.

        Args:
            instrument_ids: Canonical CASH-equity ids — pass the universe-EVER-members
                (incl. delisted) to stay survivorship-free.
            since: Inclusive lower bound on ``period_end`` (epoch ms UTC).
            until: Exclusive upper bound on ``period_end``; ``None`` clamps to ``now``.
            now: Injectable wall clock (epoch ms UTC); ``None`` reads :func:`now_ms`.

        Raises:
            ValueError: ``since < 0``.
        """
        if since < 0:
            raise ValueError(f"since must be epoch ms UTC >= 0, got {since}")
        now_eff: Ms = now_ms() if now is None else now
        until_eff: Ms = now_eff if until is None else min(until, now_eff)
        with self._exclusive_lock():
            return self._run_locked(
                tuple(instrument_ids), since=since, until=until_eff, now=now_eff
            )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        """Hold the OS-level exclusive ingest lock (no-op when ``lock_path`` is None).

        Non-blocking ``flock``; a held lock raises :class:`LockHeldError` immediately. The
        OS releases it even on ``kill -9``. The SAME lock path the bar ingests use
        serializes all lake writers.
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
                with suppress(OSError):
                    holder = os.read(fd, 64).decode(errors="replace").strip()
                raise LockHeldError(
                    f"ingest lock {self._lock_path} is held"
                    f"{f' by pid {holder}' if holder else ''}; another ingestion run is in "
                    "progress (never run two — they would race on the same lake)"
                ) from exc
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            yield
        finally:
            os.close(fd)

    def _run_locked(
        self, instrument_ids: tuple[str, ...], *, since: Ms, until: Ms, now: Ms
    ) -> FundamentalsIngestReport:
        """The body of :meth:`run`, executed under the exclusive ingest lock."""
        started: Ms = now_ms()
        run_id = self._checkpoints.start_run("ingest-fundamentals")
        _log.info(
            "fundamentals_ingest.run_started",
            run_id=run_id,
            instruments=len(instrument_ids),
            since=since,
            until=until,
        )
        results: list[FundamentalsIngestResult] = []
        try:
            for iid in instrument_ids:
                results.append(self._process(iid, since=since, until=until, now=now))
        except BaseException:
            self._checkpoints.finish_run(run_id, "failed", detail="aborted by BaseException")
            raise

        report = FundamentalsIngestReport(
            run_id=run_id, started=started, finished=now_ms(), results=tuple(results)
        )
        self._checkpoints.finish_run(
            run_id, "ok" if report.failed_count == 0 else "failed", detail=report.summary()
        )
        _log.info("fundamentals_ingest.run_finished", run_id=run_id, summary=report.summary())
        return report

    def _process(
        self, instrument_id: str, *, since: Ms, until: Ms, now: Ms
    ) -> FundamentalsIngestResult:
        """Ingest one instrument's fundamentals (or skip if already checkpointed to ``until``)."""
        watermark = self._checkpoints.get(_DATASET, instrument_id)
        if watermark is not None and watermark >= until:
            return FundamentalsIngestResult(instrument_id=instrument_id, status="skipped", rows=0)
        try:
            table = self._source.fetch_fundamentals(instrument_id, since=since, until=until)
            rows = table.num_rows
            if rows > 0:
                self._writer.write(_DATASET, table, now=now)
            # Advance the watermark even on a zero-row instrument: it WAS fetched through
            # ``until`` (a name with no filings in range is done, not retried forever).
            self._checkpoints.set(_DATASET, instrument_id, until)
            _log.info("fundamentals_ingest.instrument_done", instrument_id=instrument_id, rows=rows)
            return FundamentalsIngestResult(instrument_id=instrument_id, status="ok", rows=rows)
        except Exception as exc:
            _log.error(
                "fundamentals_ingest.instrument_failed",
                instrument_id=instrument_id,
                error=repr(exc),
            )
            return FundamentalsIngestResult(
                instrument_id=instrument_id, status="failed", rows=0, error=repr(exc)
            )
