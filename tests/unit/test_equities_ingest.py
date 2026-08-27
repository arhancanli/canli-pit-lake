"""Unit tests for alphaforge.data.ingest.equities — EquitiesFlatFilesJob.

The job is wired against a REAL LakeWriter + CheckpointStore over ``tmp_path`` (never the
live lake) and a fake-S3-client-backed PolygonFlatFilesSource (never the network). Covers
the durability/resume contract, the day-watermark, failure isolation, the max-tickers cap,
the exclusive lock, and — the headline gate — the survivorship regression: a known delisted
ticker in an old day-file lands in the lake and is NOT silently dropped (EQUITIES_INGEST.md
§4.2-4.3).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from test_polygon_flatfiles import FakeFlatFilesClient, csv_row

from alphaforge.core.errors import LockHeldError
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms, Timeframe, to_ms
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.ingest.equities import (
    EquitiesFlatFilesJob,
    EquitiesIngestReport,
)
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.polygon_flatfiles import (
    FLATFILES_MIC,
    PolygonFlatFilesSource,
    day_aggs_key,
)
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

D1 = Timeframe.D1.ms

AAPL_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "AAPL")
LEHMQ_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "LEHMQ")  # delisted analogue
MSFT_ID = SymbolMapper.equity_instrument_id(FLATFILES_MIC, "MSFT")

# A watermark "now" far in the future so every synthetic session is fully closed.
NOW = to_ms(datetime(2024, 12, 31, tzinfo=UTC))

# The flat-files day-watermark reserved key (private; pinned here to assert on it).
_WM_DATASET = Dataset.OHLCV_1D
_WM_ID = "__polygon_flatfiles_day_watermark__"


def _day_open_ms(d: date) -> Ms:
    return to_ms(datetime(d.year, d.month, d.day, tzinfo=UTC))


def make_job(
    tmp_path: Path,
    client: FakeFlatFilesClient,
    *,
    max_tickers: int | None = None,
    lock: bool = False,
    name: str = "a",
) -> tuple[EquitiesFlatFilesJob, LakePaths, CheckpointStore]:
    paths = LakePaths(tmp_path / f"lake_{name}")
    checkpoints = CheckpointStore(tmp_path / f"ops_{name}.sqlite")
    job = EquitiesFlatFilesJob(
        PolygonFlatFilesSource(client=client),
        LakeWriter(paths),
        checkpoints,
        max_tickers=max_tickers,
        lock_path=(tmp_path / "ingest.lock") if lock else None,
    )
    return job, paths, checkpoints


def read_back(paths: LakePaths, instrument_ids: list[str], *, start: Ms, end: Ms) -> dict[str, int]:
    """Rows-per-instrument read back PIT from the lake (as_of huge so all bars visible)."""
    reader = PITDataReader(paths)
    tbl = reader.ohlcv(instrument_ids, start=start, end=end, as_of=NOW + 10 * D1, tf=Timeframe.D1)
    ids = tbl.column("instrument_id").to_pylist()
    return {iid: ids.count(iid) for iid in set(ids)}


# ----------------------------------------------------------------- basic ingest


def test_ingest_writes_partitions(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    for d in days:
        client.put_day(d, [csv_row("AAPL", d), csv_row("MSFT", d)])
    job, paths, checkpoints = make_job(tmp_path, client)

    report = job.run(start=_day_open_ms(days[0]), until=_day_open_ms(date(2024, 1, 5)), now=NOW)

    assert isinstance(report, EquitiesIngestReport)
    assert report.ok_count == 3
    assert report.failed_count == 0
    assert report.total_rows == 6
    counts = read_back(
        paths, [AAPL_ID, MSFT_ID], start=_day_open_ms(days[0]), end=_day_open_ms(date(2024, 1, 5))
    )
    assert counts == {AAPL_ID: 3, MSFT_ID: 3}
    # day watermark advanced to the last fully-written session's open
    assert checkpoints.get(_WM_DATASET, _WM_ID) == _day_open_ms(days[-1])


def test_empty_span_no_sessions(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    job, _paths, checkpoints = make_job(tmp_path, client)
    report = job.run(
        start=_day_open_ms(date(2024, 1, 2)), until=_day_open_ms(date(2024, 1, 3)), now=NOW
    )
    assert report.results == ()
    assert checkpoints.get(_WM_DATASET, _WM_ID) is None


# ----------------------------------------------------------------- resume / dedupe


def test_resume_skips_checkpointed(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    for d in days:
        client.put_day(d, [csv_row("AAPL", d)])
    job, paths, _checkpoints = make_job(tmp_path, client)

    # First run covers days 1-2.
    job.run(start=_day_open_ms(days[0]), until=_day_open_ms(days[2]), now=NOW)
    first_get_calls = list(client.get_calls)
    assert len(first_get_calls) == 2

    # Second run covers days 1-4: days 1-2 must be skipped (resume).
    report = job.run(start=_day_open_ms(days[0]), until=_day_open_ms(date(2024, 1, 6)), now=NOW)
    statuses = {r.session: r.status for r in report.results}
    assert statuses["2024-01-02"] == "skipped"
    assert statuses["2024-01-03"] == "skipped"
    assert statuses["2024-01-04"] == "ok"
    assert statuses["2024-01-05"] == "ok"
    # the skipped days were NOT re-downloaded
    new_downloads = client.get_calls[len(first_get_calls) :]
    assert all("2024-01-02" not in k and "2024-01-03" not in k for k in new_downloads)
    # no duplicate rows in the lake
    counts = read_back(
        paths, [AAPL_ID], start=_day_open_ms(days[0]), end=_day_open_ms(date(2024, 1, 6))
    )
    assert counts == {AAPL_ID: 4}


def test_crash_between_write_and_checkpoint_dedupes(tmp_path: Path) -> None:
    """Simulate a crash AFTER the write but BEFORE the watermark advance: re-run dedupes."""
    client = FakeFlatFilesClient()
    d = date(2024, 1, 2)
    client.put_day(d, [csv_row("AAPL", d)])
    job, paths, checkpoints = make_job(tmp_path, client)

    # Manually write the day's bars but DO NOT advance the watermark (the crash window).
    table = PolygonFlatFilesSource(client=client).fetch_day(d, now=NOW)
    LakeWriter(paths).write(Dataset.OHLCV_1D, table, tf=Timeframe.D1, now=NOW)
    assert checkpoints.get(_WM_DATASET, _WM_ID) is None

    # Re-run: the day is re-fetched and re-written; the writer dedupes; watermark advances.
    report = job.run(start=_day_open_ms(d), until=_day_open_ms(date(2024, 1, 3)), now=NOW)
    assert report.ok_count == 1
    assert checkpoints.get(_WM_DATASET, _WM_ID) == _day_open_ms(d)
    counts = read_back(paths, [AAPL_ID], start=_day_open_ms(d), end=_day_open_ms(date(2024, 1, 3)))
    assert counts == {AAPL_ID: 1}  # no duplicate despite the overlap re-write


# ----------------------------------------------------------------- max-tickers cap


def test_max_tickers_cap(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    d = date(2024, 1, 2)
    # quote_volume = close * volume; engineer a strict ranking AAPL > MSFT > LEHMQ.
    client.put_day(
        d,
        [
            csv_row("AAPL", d, close=100.0, volume=3_000.0),  # qv 300k
            csv_row("MSFT", d, close=100.0, volume=2_000.0),  # qv 200k
            csv_row("LEHMQ", d, close=100.0, volume=1_000.0),  # qv 100k
        ],
    )
    job, paths, _checkpoints = make_job(tmp_path, client, max_tickers=2, name="cap")
    lo, hi = _day_open_ms(d), _day_open_ms(date(2024, 1, 3))
    all_ids = [AAPL_ID, MSFT_ID, LEHMQ_ID]
    report = job.run(start=lo, until=hi, now=NOW)
    assert report.results[0].tickers == 2
    counts = read_back(paths, all_ids, start=lo, end=hi)
    assert set(counts) == {AAPL_ID, MSFT_ID}  # top-2 by dollar volume; LEHMQ dropped by the cap

    # A later UNCAPPED run adds the rest (no poisoning — writer dedupes the overlap).
    job2, _p2, c2 = make_job(tmp_path, client, max_tickers=None, name="cap")  # same lake/ops
    c2.reset(_WM_DATASET, _WM_ID)  # rewind the day watermark so the day reprocesses uncapped
    job2.run(start=lo, until=hi, now=NOW)
    assert read_back(paths, all_ids, start=lo, end=hi) == {AAPL_ID: 1, MSFT_ID: 1, LEHMQ_ID: 1}


def test_max_tickers_zero_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_tickers"):
        make_job(tmp_path, FakeFlatFilesClient(), max_tickers=0)


# ----------------------------------------------------------------- failure isolation


def test_failed_day_isolated(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    for d in days:
        client.put_day(d, [csv_row("AAPL", d)])
    # Corrupt day-3's object so its parse raises (a bad file fails that day, not the run).
    from test_polygon_flatfiles import make_day_csv_raw

    bad_body = (
        "ticker,volume,open,close,high,low,window_start,transactions\n"
        "AAPL,x,1,1,1,1,1,1\n"  # volume 'x' is not a number -> SchemaError on parse
    )
    client.put_raw(day_aggs_key(days[1]), make_day_csv_raw(bad_body))
    job, _paths, _checkpoints = make_job(tmp_path, client)
    report = job.run(start=_day_open_ms(days[0]), until=_day_open_ms(date(2024, 1, 5)), now=NOW)

    statuses = {r.session: r.status for r in report.results}
    assert statuses["2024-01-03"] == "failed"
    assert statuses["2024-01-02"] == "ok"
    assert statuses["2024-01-04"] == "ok"
    assert report.failed_count == 1
    assert report.failures()[0].session == "2024-01-03"


# ----------------------------------------------------------------- exclusive lock


def test_lock_is_exclusive(tmp_path: Path) -> None:
    client = FakeFlatFilesClient()
    job_a, _pa, _ca = make_job(tmp_path, client, lock=True, name="a")
    job_b, _pb, _cb = make_job(tmp_path, client, lock=True, name="b")
    with (
        job_a._exclusive_lock(),
        pytest.raises(LockHeldError, match=r"ingest lock .* is held"),
        job_b._exclusive_lock(),
    ):
        pass  # pragma: no cover - must not be reached


# ----------------------------------------------------------------- SURVIVORSHIP REGRESSION


def test_delisted_ticker_survives(tmp_path: Path) -> None:
    """The headline gate (EQUITIES_INGEST.md §4.3).

    An OLD day-file (2008-09-15, the Lehman week) contains the now-delisted ``LEHMQ``
    ALONGSIDE the survivor ``AAPL``. The panel is survivorship-free by construction; this
    proves the INGESTER preserves that end-to-end — LEHMQ lands in the lake, is readable,
    and is NOT silently dropped even though the ticker is dead today.
    """
    client = FakeFlatFilesClient()
    old_day = date(2008, 9, 15)
    client.put_day(
        old_day,
        [
            csv_row("AAPL", old_day, close=160.0, volume=5_000.0),
            csv_row("LEHMQ", old_day, close=0.21, volume=900_000.0),  # the delisted name
        ],
    )
    job, paths, _checkpoints = make_job(tmp_path, client, name="surv")
    report = job.run(start=_day_open_ms(old_day), until=_day_open_ms(date(2008, 9, 16)), now=NOW)
    assert report.ok_count == 1

    # The LEHMQ partition exists on disk (not silently dropped).
    assert LEHMQ_ID in paths.instrument_ids(Dataset.OHLCV_1D)
    assert AAPL_ID in paths.instrument_ids(Dataset.OHLCV_1D)

    # Both bars read back, correct values.
    reader = PITDataReader(paths)
    tbl = reader.ohlcv(
        [AAPL_ID, LEHMQ_ID],
        start=_day_open_ms(old_day),
        end=_day_open_ms(date(2008, 9, 16)),
        as_of=NOW,
        tf=Timeframe.D1,
    )
    by_id = {r["instrument_id"]: r for r in tbl.to_pylist()}
    assert set(by_id) == {AAPL_ID, LEHMQ_ID}
    assert by_id[LEHMQ_ID]["close"] == 0.21
    assert by_id[LEHMQ_ID]["volume"] == 900_000.0
    assert by_id[LEHMQ_ID]["ts_open"] == datetime(2008, 9, 15, tzinfo=UTC)
