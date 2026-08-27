"""End-to-end equities flat-files ingest against the FAKE S3 client into a tmp lake.

NO network, NO live lake (EQUITIES_INGEST.md §4.2-4.3 / BUILD C). This is the integration
counterpart to the unit job tests: it drives the REAL ``EquitiesFlatFilesJob`` →
``LakeWriter`` → ``PITDataReader`` chain over a multi-day synthetic panel that includes a
known delisted ticker, and asserts the lake is logically correct and survivorship-free, with
resume idempotence across two runs.

The fake S3 client and synthetic CSV fixtures are reused from the unit suite
(``tests/unit/test_polygon_flatfiles.py``) — the same in-memory ``dict[str, bytes]`` that the
unit tests use, so this proves the pipeline composes, never that the network works.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms, Timeframe, to_ms
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.ingest.equities import EquitiesFlatFilesJob
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.polygon_flatfiles import FLATFILES_MIC, PolygonFlatFilesSource
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

# Reuse the unit suite's fake S3 client + synthetic-CSV helper (no network, no boto3).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_polygon_flatfiles import FakeFlatFilesClient, csv_row

D1 = Timeframe.D1.ms

AAPL_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "AAPL")
LEHMQ_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "LEHMQ")  # delisted analogue
NOW = to_ms(datetime(2024, 12, 31, tzinfo=UTC))


def _day_open_ms(d: date) -> Ms:
    return to_ms(datetime(d.year, d.month, d.day, tzinfo=UTC))


def _build(tmp_path: Path, client: FakeFlatFilesClient) -> tuple[EquitiesFlatFilesJob, LakePaths]:
    paths = LakePaths(tmp_path / "lake")
    job = EquitiesFlatFilesJob(
        PolygonFlatFilesSource(client=client),
        LakeWriter(paths),
        CheckpointStore(tmp_path / "ops.sqlite"),
        lock_path=tmp_path / "ingest.lock",
    )
    return job, paths


def test_end_to_end_ingest_survivorship_and_resume(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    days = [date(2008, 9, 15), date(2008, 9, 16), date(2008, 9, 17)]
    lehmq_close = {days[0]: 3.0, days[1]: 2.0, days[2]: 1.0}  # the dying tape, clean floats
    for d in days:
        # AAPL survives; LEHMQ is the delisted name present in these old day-files.
        client.put_day(
            d,
            [
                csv_row("AAPL", d, close=160.0 + days.index(d), volume=5_000.0),
                csv_row("LEHMQ", d, close=lehmq_close[d], volume=900_000.0),
            ],
        )

    job, paths = _build(tmp_path, client)
    start, end = _day_open_ms(days[0]), _day_open_ms(date(2008, 9, 18))
    report = job.run(start=start, until=end, now=NOW)
    assert report.ok_count == 3
    assert report.total_rows == 6  # 2 ids x 3 days

    # Survivorship-free end-to-end: both partitions exist and read back.
    stored = set(paths.instrument_ids(Dataset.OHLCV_1D))
    assert {AAPL_ID, LEHMQ_ID} <= stored

    reader = PITDataReader(paths)
    tbl = reader.ohlcv(
        [AAPL_ID, LEHMQ_ID], start=start, end=end, as_of=NOW + 10 * D1, tf=Timeframe.D1
    )
    rows = tbl.to_pylist()
    lehmq_rows = [r for r in rows if r["instrument_id"] == LEHMQ_ID]
    assert len(lehmq_rows) == 3  # the dead ticker's three sessions all landed
    assert {r["close"] for r in lehmq_rows} == {3.0, 2.0, 1.0}

    # Resume idempotence: re-running the SAME span skips every day and adds no rows.
    report2 = job.run(start=start, until=end, now=NOW)
    assert report2.skipped_count == 3
    assert report2.ok_count == 0
    tbl2 = reader.ohlcv(
        [AAPL_ID, LEHMQ_ID], start=start, end=end, as_of=NOW + 10 * D1, tf=Timeframe.D1
    )
    assert tbl2.num_rows == tbl.num_rows == 6  # no duplicates after the overlap re-run
