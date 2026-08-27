"""Ingestion checkpoints: per-(dataset, instrument) watermarks and a run log.

Watermark semantics (the load-bearing contract)
-----------------------------------------------
A watermark of ``W`` for ``(dataset, instrument_id)`` means: **all data for that
key with key-timestamp ≤ W is durably persisted in the lake**. The key-timestamp
is the dataset's natural key time — ``ts_open`` for OHLCV, ``ts_funding`` for
funding. Writers therefore advance the watermark **only after**
``LakeWriter.write`` has returned (the Parquet replace is atomic and durable);
never before, never speculatively.

Because a crash can land between a successful lake write and the watermark
advance, resumed jobs re-fetch an overlap window starting at (or before) the
watermark. This is expected and safe: the lake writer dedupes on the natural
key, keeping the row with the latest ``ingested_at``, so overlap re-ingestion
is idempotent (dataDesign.md §3 — 3-bar overlap on resume).

Watermarks are **monotonic**: :meth:`CheckpointStore.set` refuses to move a
watermark backward (``ValueError``). The only way to rewind is the explicit
operator escape hatch :meth:`CheckpointStore.reset`, which deletes the
watermark so the next run backfills from its configured start.

Run log
-------
Every ingestion job records a row in ``runs``: ``start_run`` inserts a
``'running'`` row; ``finish_run`` stamps it ``'ok'`` or ``'failed'`` with a
finish time and free-form detail. ``open_runs`` lists still-``'running'`` rows —
after a crash these are exactly the jobs that died mid-flight.

Storage: a single SQLite file in WAL mode (same patterns as
``core/instruments.py``). All SQL is parameterized; all timestamps are epoch
milliseconds UTC (:data:`Ms`). Concurrent stores on the same path are safe:
the monotonic guard is a single atomic upsert, and ``busy_timeout`` absorbs
writer contention.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Self

from alphaforge.core.time import Ms, now_ms
from alphaforge.data.schemas import Dataset

__all__ = [
    "CheckpointStore",
    "RunRecord",
    "RunStatus",
]

RunStatus = Literal["ok", "failed"]
"""Terminal run states accepted by :meth:`CheckpointStore.finish_run`."""

_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"ok", "failed"})

_CREATE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS watermarks (
    dataset       TEXT    NOT NULL,
    instrument_id TEXT    NOT NULL,
    watermark_ms  INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (dataset, instrument_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    job         TEXT    NOT NULL,
    started_ms  INTEGER NOT NULL,
    finished_ms INTEGER,
    status      TEXT    NOT NULL CHECK (status IN ('running', 'ok', 'failed')),
    detail      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_runs_open ON runs (started_ms) WHERE status = 'running';
"""

# Atomic monotonic upsert: the UPDATE arm only fires when the new watermark is
# >= the stored one, so a regression leaves rowcount == 0 without any race
# window between read and write (SQLite serializes writers).
_SET_SQL: Final[str] = """
INSERT INTO watermarks (dataset, instrument_id, watermark_ms, updated_at_ms)
VALUES (?, ?, ?, ?)
ON CONFLICT (dataset, instrument_id) DO UPDATE SET
    watermark_ms  = excluded.watermark_ms,
    updated_at_ms = excluded.updated_at_ms
WHERE excluded.watermark_ms >= watermarks.watermark_ms
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecord:
    """One row of the ingestion run log.

    ``started_ms``/``finished_ms`` are epoch ms UTC; ``finished_ms`` is ``None``
    while (and only while) ``status == 'running'``. ``detail`` is free-form
    operator text (error summary, row counts) — never parsed by code.
    """

    run_id: str
    job: str
    started_ms: Ms
    finished_ms: Ms | None
    status: str
    detail: str


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    """SQL row → :class:`RunRecord`."""
    finished = row["finished_ms"]
    return RunRecord(
        run_id=str(row["run_id"]),
        job=str(row["job"]),
        started_ms=int(row["started_ms"]),
        finished_ms=None if finished is None else int(finished),
        status=str(row["status"]),
        detail=str(row["detail"]),
    )


def _enable_wal(conn: sqlite3.Connection, *, budget_s: float = 5.0) -> None:
    """Switch ``conn``'s database to WAL journal mode, retrying briefly on contention.

    The DELETE→WAL upgrade needs exclusive access, and SQLite returns
    ``SQLITE_BUSY`` *without consulting the busy handler* in deadlock-prone lock
    upgrades — so stores racing to open the same file (e.g. one InstrumentStore +
    one CheckpointStore per thread on the shared ops db) can see
    ``sqlite3.OperationalError: database is locked`` despite ``busy_timeout``.
    WAL mode is persistent in the file, so the retry converges immediately once
    any opener succeeds. Re-raises after ``budget_s`` seconds of failures.
    """
    deadline = time.monotonic() + budget_s
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
        else:
            return


class CheckpointStore:
    """SQLite-backed ingestion watermarks and run log (WAL mode).

    See the module docstring for the watermark contract. Usable as a context
    manager (closes the connection on exit). One instance per thread; multiple
    instances on the same ``db_path`` coordinate safely through SQLite WAL +
    ``busy_timeout`` and the atomic monotonic upsert.
    """

    def __init__(self, db_path: Path) -> None:
        """Open (creating parent directories and schema if absent) the store."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        _enable_wal(self._conn)
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_CREATE_SQL)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection; the store is unusable afterwards."""
        self._conn.close()

    # ------------------------------------------------------------------ #
    # watermarks                                                          #
    # ------------------------------------------------------------------ #

    def get(self, dataset: Dataset, instrument_id: str) -> Ms | None:
        """Return the watermark for ``(dataset, instrument_id)``, or ``None``.

        ``None`` means no data for the key has ever been confirmed durable —
        the next run must backfill from its configured historical start.
        """
        cur = self._conn.execute(
            "SELECT watermark_ms FROM watermarks WHERE dataset = ? AND instrument_id = ?",
            (dataset.value, instrument_id),
        )
        row = cur.fetchone()
        return None if row is None else int(row["watermark_ms"])

    def last_updated(self, dataset: Dataset, instrument_id: str) -> Ms | None:
        """Wall-clock ms when this key's watermark was last confirmed, or ``None``.

        Distinct from :meth:`get` (which returns the data watermark): this is the
        ``updated_at_ms`` bookkeeping stamp, used by callers that poll a per-ticker
        reference endpoint to throttle how often an *already-current* key is re-checked
        (a watermark can be current yet ``until`` still advances daily, which would
        otherwise re-hit the endpoint every run).
        """
        cur = self._conn.execute(
            "SELECT updated_at_ms FROM watermarks WHERE dataset = ? AND instrument_id = ?",
            (dataset.value, instrument_id),
        )
        row = cur.fetchone()
        return None if row is None else int(row["updated_at_ms"])

    def set(self, dataset: Dataset, instrument_id: str, watermark: Ms) -> None:
        """Advance the watermark to ``watermark`` (epoch ms UTC).

        Monotonic: raises ``ValueError`` if ``watermark`` is below the stored
        value — a watermark NEVER moves backward (use :meth:`reset` for the
        deliberate operator rewind). Re-setting the *same* value is an
        idempotent no-op apart from refreshing ``updated_at_ms`` (a resumed
        run that re-ingested only overlap legitimately re-confirms it).
        Negative watermarks are rejected (``ValueError``): epoch ms UTC only.

        Call this only after ``LakeWriter.write`` has returned for every row
        with key-timestamp ≤ ``watermark`` (module docstring contract).
        """
        if watermark < 0:
            raise ValueError(f"watermark must be epoch ms UTC >= 0, got {watermark}")
        with self._conn:
            cur = self._conn.execute(_SET_SQL, (dataset.value, instrument_id, watermark, now_ms()))
            if cur.rowcount == 0:
                current = self.get(dataset, instrument_id)
                raise ValueError(
                    f"watermark regression for ({dataset.value!r}, {instrument_id!r}): "
                    f"{watermark} < current {current}; watermarks never move backward "
                    "(use reset() for an explicit operator rewind)"
                )

    def reset(self, dataset: Dataset, instrument_id: str) -> bool:
        """Delete the watermark for ``(dataset, instrument_id)`` — operator use only.

        This is the sole sanctioned way to move ingestion backward: the next
        run sees ``get(...) is None`` and re-backfills from its configured
        start (idempotent — the lake writer dedupes). Returns ``True`` if a
        watermark existed and was deleted, ``False`` if there was none (no-op).
        """
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM watermarks WHERE dataset = ? AND instrument_id = ?",
                (dataset.value, instrument_id),
            )
        return cur.rowcount > 0

    def all_watermarks(self, dataset: Dataset) -> dict[str, Ms]:
        """All watermarks for ``dataset`` as ``{instrument_id: watermark_ms}``.

        Empty dict if the dataset has no watermarks yet. Keys are canonical
        instrument ids (``"BINANCE:PERP:BTCUSDT"``).
        """
        cur = self._conn.execute(
            "SELECT instrument_id, watermark_ms FROM watermarks "
            "WHERE dataset = ? ORDER BY instrument_id",
            (dataset.value,),
        )
        return {str(row["instrument_id"]): int(row["watermark_ms"]) for row in cur.fetchall()}

    # ------------------------------------------------------------------ #
    # run log                                                             #
    # ------------------------------------------------------------------ #

    def start_run(self, job: str) -> str:
        """Insert a ``'running'`` run row for ``job`` and return its uuid4 ``run_id``.

        ``job`` is a human-readable job name (e.g. ``"backfill:ohlcv"``);
        ``started_ms`` is stamped from the wall clock (epoch ms UTC).
        """
        if not job:
            raise ValueError("job name must be non-empty")
        run_id = str(uuid.uuid4())
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, job, started_ms, finished_ms, status, detail) "
                "VALUES (?, ?, ?, NULL, 'running', '')",
                (run_id, job, now_ms()),
            )
        return run_id

    def finish_run(self, run_id: str, status: RunStatus, detail: str = "") -> None:
        """Mark a ``'running'`` run terminal: status ``'ok'`` or ``'failed'``.

        Stamps ``finished_ms`` from the wall clock and stores ``detail``
        (free-form: error summary, row counts). Raises ``ValueError`` if
        ``status`` is not a terminal status, if ``run_id`` is unknown, or if
        the run is already finished (a run finishes exactly once).
        """
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"finish_run status must be 'ok' or 'failed', got {status!r}")
        with self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET finished_ms = ?, status = ?, detail = ? "
                "WHERE run_id = ? AND status = 'running'",
                (now_ms(), status, detail, run_id),
            )
            if cur.rowcount == 0:
                existing = self.get_run(run_id)
                if existing is None:
                    raise ValueError(f"finish_run: unknown run_id {run_id!r}")
                raise ValueError(
                    f"finish_run: run {run_id!r} already finished with status {existing.status!r}"
                )

    def open_runs(self) -> list[RunRecord]:
        """All runs still in ``'running'`` state, oldest first.

        After a crash, these are exactly the jobs that died mid-flight (their
        ``finish_run`` never executed) — the operator triage list.
        """
        cur = self._conn.execute(
            "SELECT run_id, job, started_ms, finished_ms, status, detail FROM runs "
            "WHERE status = 'running' ORDER BY started_ms, run_id",
        )
        return [_run_from_row(row) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return the run row for ``run_id`` (any status), or ``None`` if unknown."""
        cur = self._conn.execute(
            "SELECT run_id, job, started_ms, finished_ms, status, detail FROM runs "
            "WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        return None if row is None else _run_from_row(row)
