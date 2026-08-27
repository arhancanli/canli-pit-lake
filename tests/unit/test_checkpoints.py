"""Unit tests for alphaforge.data.ingest.checkpoints — watermarks and run log."""

from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from alphaforge.data.ingest.checkpoints import CheckpointStore, RunRecord
from alphaforge.data.schemas import Dataset

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"
T1 = 1_704_153_600_000  # 2024-01-02T00:00:00Z
T2 = 1_704_157_200_000  # 2024-01-02T01:00:00Z
T3 = 1_704_160_800_000  # 2024-01-02T02:00:00Z


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "ops.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[CheckpointStore]:
    with CheckpointStore(db_path) as s:
        yield s


class TestWatermarks:
    def test_get_missing_returns_none(self, store: CheckpointStore) -> None:
        assert store.get(Dataset.OHLCV, BTC) is None

    def test_set_get_roundtrip(self, store: CheckpointStore) -> None:
        store.set(Dataset.OHLCV, BTC, T1)
        assert store.get(Dataset.OHLCV, BTC) == T1
        store.set(Dataset.OHLCV, BTC, T2)
        assert store.get(Dataset.OHLCV, BTC) == T2

    def test_set_equal_watermark_is_idempotent(self, store: CheckpointStore) -> None:
        store.set(Dataset.OHLCV, BTC, T1)
        store.set(Dataset.OHLCV, BTC, T1)  # resumed run re-confirms: no error
        assert store.get(Dataset.OHLCV, BTC) == T1

    def test_regression_raises_and_preserves_watermark(self, store: CheckpointStore) -> None:
        store.set(Dataset.OHLCV, BTC, T2)
        with pytest.raises(ValueError, match="regression"):
            store.set(Dataset.OHLCV, BTC, T1)
        assert store.get(Dataset.OHLCV, BTC) == T2

    def test_negative_watermark_raises(self, store: CheckpointStore) -> None:
        with pytest.raises(ValueError, match="epoch ms"):
            store.set(Dataset.OHLCV, BTC, -1)

    def test_keys_are_independent(self, store: CheckpointStore) -> None:
        store.set(Dataset.OHLCV, BTC, T3)
        store.set(Dataset.FUNDING, BTC, T1)  # lower, different dataset: fine
        store.set(Dataset.OHLCV, ETH, T1)  # lower, different instrument: fine
        assert store.get(Dataset.OHLCV, BTC) == T3
        assert store.get(Dataset.FUNDING, BTC) == T1
        assert store.get(Dataset.OHLCV, ETH) == T1
        assert store.get(Dataset.FUNDING, ETH) is None

    def test_all_watermarks(self, store: CheckpointStore) -> None:
        assert store.all_watermarks(Dataset.OHLCV) == {}
        store.set(Dataset.OHLCV, ETH, T2)
        store.set(Dataset.OHLCV, BTC, T1)
        store.set(Dataset.FUNDING, BTC, T3)
        assert store.all_watermarks(Dataset.OHLCV) == {BTC: T1, ETH: T2}
        assert store.all_watermarks(Dataset.FUNDING) == {BTC: T3}

    def test_reset_allows_explicit_rewind(self, store: CheckpointStore) -> None:
        store.set(Dataset.OHLCV, BTC, T3)
        assert store.reset(Dataset.OHLCV, BTC) is True
        assert store.get(Dataset.OHLCV, BTC) is None
        store.set(Dataset.OHLCV, BTC, T1)  # rewind possible only via reset
        assert store.get(Dataset.OHLCV, BTC) == T1

    def test_reset_missing_is_noop(self, store: CheckpointStore) -> None:
        assert store.reset(Dataset.OHLCV, BTC) is False

    def test_persists_across_reopen(self, db_path: Path) -> None:
        with CheckpointStore(db_path) as first:
            first.set(Dataset.OHLCV, BTC, T1)
        with CheckpointStore(db_path) as second:
            assert second.get(Dataset.OHLCV, BTC) == T1

    def test_concurrent_writes_are_consistent(self, db_path: Path) -> None:
        """Racing threads (one store each, same file): max value wins, no corruption."""
        n_threads, per_thread = 4, 50
        rng = random.Random(0)
        values = [T1 + i * 3_600_000 for i in range(n_threads * per_thread)]
        rng.shuffle(values)
        chunks = [values[i::n_threads] for i in range(n_threads)]
        errors: list[Exception] = []

        def worker(chunk: list[int]) -> None:
            with CheckpointStore(db_path) as s:
                for value in chunk:
                    try:
                        s.set(Dataset.OHLCV, BTC, value)
                    except ValueError:
                        pass  # lost the race to a higher concurrent watermark
                    except Exception as exc:
                        errors.append(exc)

        threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        with CheckpointStore(db_path) as s:
            assert s.get(Dataset.OHLCV, BTC) == max(values)


class TestRunLog:
    def test_run_lifecycle_ok(self, store: CheckpointStore) -> None:
        run_id = store.start_run("backfill:ohlcv")
        open_now = store.open_runs()
        assert [r.run_id for r in open_now] == [run_id]
        running = open_now[0]
        assert isinstance(running, RunRecord)
        assert running.job == "backfill:ohlcv"
        assert running.status == "running"
        assert running.finished_ms is None
        assert running.started_ms > 0

        store.finish_run(run_id, "ok", detail="rows=1234")
        assert store.open_runs() == []
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.status == "ok"
        assert finished.detail == "rows=1234"
        assert finished.finished_ms is not None
        assert finished.finished_ms >= finished.started_ms

    def test_run_lifecycle_failed(self, store: CheckpointStore) -> None:
        run_id = store.start_run("backfill:funding")
        store.finish_run(run_id, "failed", detail="ccxt.NetworkError after 8 attempts")
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.status == "failed"
        assert "NetworkError" in finished.detail

    def test_run_ids_are_unique(self, store: CheckpointStore) -> None:
        ids = {store.start_run("job") for _ in range(10)}
        assert len(ids) == 10

    def test_open_runs_lists_only_running(self, store: CheckpointStore) -> None:
        a = store.start_run("job-a")
        b = store.start_run("job-b")
        store.finish_run(a, "ok")
        assert [r.run_id for r in store.open_runs()] == [b]

    def test_finish_unknown_run_raises(self, store: CheckpointStore) -> None:
        with pytest.raises(ValueError, match="unknown run_id"):
            store.finish_run("no-such-run", "ok")

    def test_finish_twice_raises(self, store: CheckpointStore) -> None:
        run_id = store.start_run("job")
        store.finish_run(run_id, "ok")
        with pytest.raises(ValueError, match="already finished"):
            store.finish_run(run_id, "failed")
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.status == "ok"  # first terminal status sticks

    def test_finish_invalid_status_raises(self, store: CheckpointStore) -> None:
        run_id = store.start_run("job")
        with pytest.raises(ValueError, match="'ok' or 'failed'"):
            store.finish_run(run_id, "running")  # type: ignore[arg-type]

    def test_empty_job_name_raises(self, store: CheckpointStore) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            store.start_run("")

    def test_get_run_missing_returns_none(self, store: CheckpointStore) -> None:
        assert store.get_run("no-such-run") is None


def test_context_manager_closes_connection(db_path: Path) -> None:
    with CheckpointStore(db_path) as store:
        store.set(Dataset.OHLCV, BTC, T1)
    with pytest.raises(sqlite3.ProgrammingError):
        store.get(Dataset.OHLCV, BTC)
