"""Exclusive ingest lock: two concurrent ingestion runs are mechanically impossible.

Born from a live incident on 2026-06-11: a SIGKILL aimed at a ``uv run`` wrapper
orphaned the underlying backfill process, and a "resume" run then raced it on the
same checkpoints. The monotonic-watermark guard and the deduping writer kept the
lake uncorrupted (by design), but four pairs failed with watermark regressions.
The lock removes the race class entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_backfill import FakeSource, FakeVision, make_inst

from alphaforge.core.errors import LockHeldError
from alphaforge.core.instruments import InstrumentStore
from alphaforge.core.time import Timeframe
from alphaforge.data.ingest.backfill import BackfillJob
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.writer import LakeWriter


def _make_job(tmp_path: Path, *, lock: bool, name: str = "a") -> BackfillJob:
    checkpoints = CheckpointStore(tmp_path / f"ops_{name}.sqlite")
    instruments = InstrumentStore(tmp_path / f"inst_{name}.sqlite")
    instruments.upsert(make_inst("BTCUSDT", listed=0), as_of=1)
    return BackfillJob(
        FakeSource({}),
        FakeVision({}),
        LakeWriter(LakePaths(tmp_path / f"lake_{name}")),
        checkpoints,
        instruments,
        tf=Timeframe.H1,
        lock_path=(tmp_path / "ingest.lock") if lock else None,
    )


class TestExclusiveLock:
    def test_second_acquirer_raises_lock_held(self, tmp_path: Path) -> None:
        job_a = _make_job(tmp_path, lock=True, name="a")
        job_b = _make_job(tmp_path, lock=True, name="b")
        with (
            job_a._exclusive_lock(),
            pytest.raises(LockHeldError, match=r"ingest lock .* is held"),
            job_b._exclusive_lock(),
        ):
            pass  # pragma: no cover - must not be reached

    def test_lock_released_after_exit_allows_reacquire(self, tmp_path: Path) -> None:
        job_a = _make_job(tmp_path, lock=True, name="a")
        job_b = _make_job(tmp_path, lock=True, name="b")
        with job_a._exclusive_lock():
            pass
        with job_b._exclusive_lock():
            pass  # no raise: lock was released with the context

    def test_lock_released_when_body_raises(self, tmp_path: Path) -> None:
        job = _make_job(tmp_path, lock=True)
        with pytest.raises(RuntimeError, match="boom"), job._exclusive_lock():
            raise RuntimeError("boom")
        with job._exclusive_lock():
            pass

    def test_lock_file_records_holder_pid(self, tmp_path: Path) -> None:
        job = _make_job(tmp_path, lock=True)
        with job._exclusive_lock():
            recorded = (tmp_path / "ingest.lock").read_text().strip()
            assert recorded == str(os.getpid())

    def test_error_message_names_holder_pid(self, tmp_path: Path) -> None:
        job_a = _make_job(tmp_path, lock=True, name="a")
        job_b = _make_job(tmp_path, lock=True, name="b")
        with (
            job_a._exclusive_lock(),
            pytest.raises(LockHeldError, match=f"pid {os.getpid()}"),
            job_b._exclusive_lock(),
        ):
            pass  # pragma: no cover

    def test_no_lock_path_is_noop(self, tmp_path: Path) -> None:
        job_a = _make_job(tmp_path, lock=False, name="a")
        job_b = _make_job(tmp_path, lock=False, name="b")
        with job_a._exclusive_lock(), job_b._exclusive_lock():
            pass  # both enter: locking disabled is explicitly permitted for tests
