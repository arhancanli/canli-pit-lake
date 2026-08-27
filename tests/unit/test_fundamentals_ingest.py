"""Durability + isolation tests for ``FundamentalsJob``.

The job is wired against a REAL LakeWriter + CheckpointStore over ``tmp_path`` (never the
network) with a fake source returning canned FUNDAMENTALS tables. The contract under test is
the same per-unit durability the bar ingests guarantee: write -> checkpoint, resume skips
done instruments, one instrument's failure is isolated, a zero-row instrument still advances.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.ingest.fundamentals import FundamentalsJob
from alphaforge.data.schemas import FUNDAMENTALS_SCHEMA, Dataset
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.writer import LakeWriter

NOW = 1_750_000_000_000
DAY = 86_400_000

AAPL = "XUSE:CASH:AAPLUSD"
MSFT = "XUSE:CASH:MSFTUSD"
BAD = "XUSE:CASH:BADUSD"


def _row(instrument_id: str, period_end: int) -> pa.Table:
    """One-quarter FUNDAMENTALS table; available_at = period_end + 40 days (PIT)."""
    return pa.Table.from_pydict(
        {
            "instrument_id": [instrument_id],
            "period_end": [period_end],
            "available_at": [period_end + 40 * DAY],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2024],
            "revenues": [1.0e9],
            "cost_of_revenue": [4.0e8],
            "gross_profit": [6.0e8],
            "operating_income": [3.0e8],
            "net_income": [2.0e8],
            "equity": [5.0e9],
            "assets": [1.2e10],
            "diluted_shares": [1.0e7],
            "op_cash_flow": [2.5e8],
            "invest_cash_flow": [-1.0e8],
            "capex": [-8.0e7],
            "free_cash_flow": [1.7e8],
            "net_common_issued": [-5.0e7],
            "shares_basic": [9.8e6],
            "share_factor": [1.0],
            "assets_avg": [1.15e10],
            "ingested_at": [NOW],
        },
        schema=FUNDAMENTALS_SCHEMA,
    )


class FakeFundamentalsSource:
    """Returns one canned quarter per instrument; ``BAD`` raises (failure-isolation test)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_fundamentals(self, instrument_id: str, *, since: int, until: int) -> pa.Table:
        self.calls.append(instrument_id)
        if instrument_id == BAD:
            raise ValueError("simulated vendor failure")
        period_end = 1_704_067_200_000  # 2024-01-01
        if not since <= period_end < until:
            return FUNDAMENTALS_SCHEMA.empty_table()
        return _row(instrument_id, period_end)


def _make_job(tmp_path: Path, source: FakeFundamentalsSource, name: str = "f"):
    paths = LakePaths(tmp_path / f"lake_{name}")
    checkpoints = CheckpointStore(tmp_path / f"ops_{name}.sqlite")
    job = FundamentalsJob(source, LakeWriter(paths), checkpoints)
    return job, paths, checkpoints


def test_writes_partitions_and_checkpoints(tmp_path: Path) -> None:
    src = FakeFundamentalsSource()
    job, paths, checkpoints = _make_job(tmp_path, src)
    report = job.run([AAPL, MSFT], since=0, until=NOW, now=NOW)
    assert report.ok_count == 2
    assert report.failed_count == 0
    assert report.total_rows == 2
    assert set(paths.instrument_ids(Dataset.FUNDAMENTALS)) == {AAPL, MSFT}
    # watermark advanced to `until` for each
    assert checkpoints.get(Dataset.FUNDAMENTALS, AAPL) == NOW


def test_resume_skips_checkpointed(tmp_path: Path) -> None:
    src = FakeFundamentalsSource()
    job, _paths, _ckpt = _make_job(tmp_path, src)
    job.run([AAPL], since=0, until=NOW, now=NOW)
    assert src.calls == [AAPL]
    # second run: already checkpointed to `until` -> skipped, no re-fetch
    report = job.run([AAPL], since=0, until=NOW, now=NOW)
    assert report.skipped_count == 1
    assert src.calls == [AAPL]  # NOT called again


def test_failure_is_isolated(tmp_path: Path) -> None:
    src = FakeFundamentalsSource()
    job, paths, checkpoints = _make_job(tmp_path, src)
    report = job.run([AAPL, BAD, MSFT], since=0, until=NOW, now=NOW)
    assert report.ok_count == 2
    assert report.failed_count == 1
    assert report.failures()[0].instrument_id == BAD
    # the good instruments still wrote + checkpointed; the bad one did NOT checkpoint
    assert set(paths.instrument_ids(Dataset.FUNDAMENTALS)) == {AAPL, MSFT}
    assert checkpoints.get(Dataset.FUNDAMENTALS, BAD) is None


def test_zero_row_instrument_advances_watermark(tmp_path: Path) -> None:
    src = FakeFundamentalsSource()
    job, paths, checkpoints = _make_job(tmp_path, src)
    # until just after epoch -> the canned 2024 period is out of range -> zero rows
    report = job.run([AAPL], since=0, until=DAY, now=NOW)
    assert report.ok_count == 1
    assert report.total_rows == 0
    # a name with no filings in range is DONE (watermark advanced), not retried forever
    assert checkpoints.get(Dataset.FUNDAMENTALS, AAPL) == DAY
    assert AAPL not in paths.instrument_ids(Dataset.FUNDAMENTALS)


def test_since_negative_rejected(tmp_path: Path) -> None:
    job, _paths, _ckpt = _make_job(tmp_path, FakeFundamentalsSource())
    with pytest.raises(ValueError, match="since"):
        job.run([AAPL], since=-1, until=NOW, now=NOW)
