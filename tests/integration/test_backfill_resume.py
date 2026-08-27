"""Crash-resume integration test for BackfillJob (buildabilityCritique.md §7 gate).

The Phase 2 acceptance criterion, in miniature and deterministic: kill the run at an
arbitrary point mid-ingestion, rerun, and the lake must be **logically identical to a
single uninterrupted run excluding ``ingested_at``**.

The crash is injected through a writer wrapper that raises a ``BaseException`` after K
successful atomic writes — a stand-in for ``kill -9`` at the worst possible moment
(between a durable write and the next one). ``BaseException`` is deliberate: it must
bypass BackfillJob's per-instrument failure isolation exactly like a real
SIGINT/SIGKILL would, aborting the whole run.

Parameterized over crash points K covering: before anything lands (K=0), mid-pair on
the REST path, at pair boundaries, mid-pair on the Vision path, and just before the
final write. No network: the venue/archive are in-memory fakes with deterministic
synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.core.errors import DataGapError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.backfill import BackfillJob, BackfillReport
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import BinanceVisionClient, YearMonth
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.writer import NATURAL_KEY_COLUMN, LakeWriter, WriteStats

HOUR = 3_600_000
DAY = 86_400_000
FUNDING_STEP = 8 * HOUR

DEC_10 = 1_702_166_400_000  # 2023-12-10T00:00:00Z
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z
JAN_10 = T0 + 9 * DAY
JAN_25 = T0 + 24 * DAY
FEB_01 = 1_706_745_600_000  # 2024-02-01T00:00:00Z (the injected "now")

BTC = "BINANCE:PERP:BTCUSDT"  # LIVE: REST path, 5 OHLCV + 5 funding batches
LUNA = "BINANCE:PERP:LUNAUSDT"  # DELISTED: Vision path, 2 OHLCV + 2 funding months

BATCH_DAYS = 5
# Deterministic write schedule (instruments sorted, datasets in request order):
#   BTC  ohlcv   [JAN_10, FEB_01) in 5-day chunks -> 5 writes  (writes 1..5)
#   BTC  funding same span                        -> 5 writes  (writes 6..10)
#   LUNA ohlcv   months 2023-12, 2024-01          -> 2 writes  (writes 11..12)
#   LUNA funding months 2023-12, 2024-01          -> 2 writes  (writes 13..14)
TOTAL_WRITES = 14


class SimulatedCrash(BaseException):
    """Process death stand-in; a BaseException so failure isolation cannot eat it."""


def read_leaf(path: Path) -> pa.Table:
    """Read one leaf file directly (``pq.read_table`` would run hive discovery on
    the ``key=value`` path segments and clash with the in-file ``instrument_id``)."""
    with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
        tbl: pa.Table = leaf.read()
    return tbl


# --------------------------------------------------------------------------- doubles


def make_inst(symbol: str, *, listed: Ms, delisted: Ms | None = None) -> Instrument:
    return Instrument(
        instrument_id=f"BINANCE:PERP:{symbol}",
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
        can_short=True,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        funding_interval_hours=8,
        listed_ts=listed,
        delisted_ts=delisted,
    )


def _grid(lo: Ms, hi: Ms, step: int) -> list[int]:
    first = -(-lo // step) * step
    return list(range(first, hi, step))


def _ohlcv_table(instrument_id: str, ts_opens: list[int]) -> pa.Table:
    n = len(ts_opens)
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_open": pa.array(ts_opens, type=pa.timestamp("ms", tz="UTC")),
            "open": pa.array([99.0] * n, type=pa.float64()),
            "high": pa.array([101.0] * n, type=pa.float64()),
            "low": pa.array([98.0] * n, type=pa.float64()),
            # Deterministic function of the timestamp so content equality is meaningful.
            "close": pa.array([float(100 + (t // HOUR) % 13) for t in ts_opens]),
            "volume": pa.array([10.0] * n, type=pa.float64()),
            "quote_volume": pa.array([1_000.0] * n, type=pa.float64()),
            "n_trades": pa.array([(t // HOUR) % 97 for t in ts_opens], type=pa.int64()),
            "quality_flags": pa.array([0] * n, type=pa.int32()),
            "ingested_at": pa.array(
                [t + HOUR + 30_000 for t in ts_opens], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


def _funding_table(instrument_id: str, ts_fundings: list[int]) -> pa.Table:
    n = len(ts_fundings)
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_funding": pa.array(ts_fundings, type=pa.timestamp("ms", tz="UTC")),
            "rate": pa.array([((t // FUNDING_STEP) % 11 - 5) * 1e-5 for t in ts_fundings]),
            "available_at": pa.array(
                [t + 300_000 for t in ts_fundings], type=pa.timestamp("ms", tz="UTC")
            ),
            "ingested_at": pa.array(
                [t + 301_000 for t in ts_fundings], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


class FakeSource(DataSource):
    """Deterministic in-memory venue over per-id data spans (closed bars only)."""

    name = "fake"

    def __init__(self, data: dict[str, tuple[Ms, Ms]]) -> None:
        self._data = data

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        return []

    def fetch_ohlcv(
        self, instrument_id: str, *, tf: Timeframe, since: Ms, until: Ms, now: Ms | None = None
    ) -> pa.Table:
        lo, hi = self._data.get(instrument_id, (0, 0))
        cut = min(until, hi)
        if now is not None:
            cut = min(cut, now - tf.ms + 1)
        return _ohlcv_table(instrument_id, _grid(max(since, lo), cut, tf.ms))

    def fetch_funding(self, instrument_id: str, *, since: Ms, until: Ms) -> pa.Table:
        lo, hi = self._data.get(instrument_id, (0, 0))
        return _funding_table(instrument_id, _grid(max(since, lo), min(until, hi), FUNDING_STEP))


class FakeVision(BinanceVisionClient):
    """Deterministic in-memory archive keyed by exchange symbol."""

    def __init__(self, data: dict[str, tuple[Ms, Ms]]) -> None:
        super().__init__()
        self._fake_data = data

    def _month_span(self, symbol: str, year: int, month: int) -> tuple[Ms, Ms]:
        if symbol not in self._fake_data:
            raise DataGapError(f"no archive file for {symbol} {year}-{month:02d}")
        lo, hi = self._fake_data[symbol]
        ym = YearMonth(year, month)
        return max(lo, ym.first_ms()), min(hi, ym.next_month_first_ms())

    def fetch_ohlcv_month(
        self, symbol: str, year: int, month: int, *, now: Ms | None = None
    ) -> pa.Table:
        lo, hi = self._month_span(symbol, year, month)
        if now is not None:
            hi = min(hi, now - HOUR + 1)
        return _ohlcv_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, HOUR))

    def fetch_funding_month(self, symbol: str, year: int, month: int) -> pa.Table:
        lo, hi = self._month_span(symbol, year, month)
        return _funding_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, FUNDING_STEP))


class CrashingWriter(LakeWriter):
    """Delegates to LakeWriter, then dies after ``crash_after`` successful writes.

    The crash fires at the *start* of write ``crash_after + 1`` — i.e. after exactly
    ``crash_after`` durability points completed and before the next batch touches
    disk, the worst spot for watermark/lake divergence to hide.
    """

    def __init__(self, paths: LakePaths, *, crash_after: int | None) -> None:
        super().__init__(paths)
        self.writes_done = 0
        self._crash_after = crash_after

    def write(
        self,
        dataset: Dataset,
        tbl: pa.Table,
        *,
        tf: Timeframe = Timeframe.H1,
        now: Ms | None = None,
    ) -> WriteStats:
        if self._crash_after is not None and self.writes_done >= self._crash_after:
            raise SimulatedCrash(f"kill -9 before write {self.writes_done + 1}")
        stats = super().write(dataset, tbl, tf=tf, now=now)
        self.writes_done += 1
        return stats


# --------------------------------------------------------------------------- harness


class Deployment:
    """One lake + ops DB + job wiring rooted at ``root`` (rebuildable across 'boots')."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = LakePaths(root / "lake")
        self.checkpoints = CheckpointStore(root / "ops.sqlite")
        self.store = InstrumentStore(root / "ops.sqlite")
        for inst in (
            make_inst("BTCUSDT", listed=JAN_10),
            make_inst("LUNAUSDT", listed=DEC_10, delisted=JAN_25),
        ):
            self.store.upsert(inst, as_of=T0)

    def close(self) -> None:
        self.checkpoints.close()
        self.store.close()

    def job(self, *, crash_after: int | None) -> tuple[BackfillJob, CrashingWriter]:
        writer = CrashingWriter(self.paths, crash_after=crash_after)
        source = FakeSource({BTC: (JAN_10, FEB_01 + DAY)})
        vision = FakeVision({"LUNAUSDT": (DEC_10, JAN_25)})
        return (
            BackfillJob(source, vision, writer, self.checkpoints, self.store),
            writer,
        )

    def run(self, *, crash_after: int | None = None) -> BackfillReport:
        job, _ = self.job(crash_after=crash_after)
        return job.run(
            datasets=[Dataset.OHLCV, Dataset.FUNDING],
            start_default=T0 - 60 * DAY,
            batch_days=BATCH_DAYS,
            now=FEB_01,
        )

    def snapshot(self, dataset: Dataset) -> pa.Table:
        """Every lake row of ``dataset``, minus ``ingested_at``, in canonical order."""
        key = NATURAL_KEY_COLUMN[dataset]
        parts: list[pa.Table] = []
        for iid in (BTC, LUNA):
            parts.extend(read_leaf(path) for path in self.paths.partition_paths(dataset, iid))
        if not parts:
            return pa.table({})
        merged = pa.concat_tables(parts).drop_columns(["ingested_at"])
        return merged.sort_by([("instrument_id", "ascending"), (key, "ascending")])

    def pair_keys(self, dataset: Dataset, instrument_id: str) -> list[int]:
        key = NATURAL_KEY_COLUMN[dataset]
        out: list[int] = []
        for path in self.paths.partition_paths(dataset, instrument_id):
            col = read_leaf(path).column(key).to_numpy(zero_copy_only=False)
            out.extend(col.astype("datetime64[ms]").astype("int64").tolist())
        return sorted(out)


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[Dataset, pa.Table]:
    """Lake content of a single uninterrupted run (the equivalence oracle)."""
    dep = Deployment(tmp_path_factory.mktemp("reference"))
    try:
        report = dep.run()
        assert report.failed_count == 0
        snaps = {ds: dep.snapshot(ds) for ds in (Dataset.OHLCV, Dataset.FUNDING)}
    finally:
        dep.close()
    # Sanity on the oracle itself: full expected grids landed for both paths.
    assert snaps[Dataset.OHLCV].num_rows == len(_grid(JAN_10, FEB_01, HOUR)) + len(
        _grid(DEC_10, JAN_25, HOUR)
    )
    assert snaps[Dataset.FUNDING].num_rows == len(_grid(JAN_10, FEB_01, FUNDING_STEP)) + len(
        _grid(DEC_10, JAN_25, FUNDING_STEP)
    )
    return snaps


PAIRS = [(ds, iid) for iid in (BTC, LUNA) for ds in (Dataset.OHLCV, Dataset.FUNDING)]

# Crash points: nothing durable yet (0), mid BTC OHLCV (2), the OHLCV/funding pair
# boundary (5), mid BTC funding (8), the live/delisted instrument boundary (10), mid
# LUNA Vision walk (11), and just before the final write (13).
CRASH_POINTS = [0, 2, 5, 8, 10, 11, 13]


@pytest.mark.parametrize("crash_after", CRASH_POINTS)
def test_crash_resume_lake_identical_to_uninterrupted_run(
    crash_after: int,
    reference: dict[Dataset, pa.Table],
    tmp_path: Path,
) -> None:
    dep = Deployment(tmp_path)
    try:
        # --- boot 1: run until the simulated kill -9 -------------------------------
        with pytest.raises(SimulatedCrash):
            dep.run(crash_after=crash_after)

        # Run-log row was stamped failed on the way out, never left dangling.
        assert dep.checkpoints.open_runs() == []

        # Durability invariant at the crash point, per pair: whatever the lake holds
        # is contiguous from the pair's span start (atomic writes — no torn batch),
        # and the watermark never runs ahead of an empty lake nor behind a full one.
        for dataset, iid in PAIRS:
            keys = dep.pair_keys(dataset, iid)
            wm = dep.checkpoints.get(dataset, iid)
            if not keys:
                assert wm is None
            else:
                step = HOUR if dataset is Dataset.OHLCV else FUNDING_STEP
                assert keys == _grid(keys[0], keys[-1] + 1, step)  # no holes
                assert wm is not None
                assert wm >= keys[-1]  # every durable row is covered by the watermark
                assert wm < JAN_25 if iid == LUNA else wm < FEB_01

        # --- boot 2: plain rerun resumes from the watermarks -----------------------
        report = dep.run()
        assert report.failed_count == 0

        # --- the gate: lake logically identical to one uninterrupted run -----------
        for dataset in (Dataset.OHLCV, Dataset.FUNDING):
            assert dep.snapshot(dataset).equals(reference[dataset]), (
                f"{dataset.value} lake diverged after crash at K={crash_after}"
            )

        # Watermarks converged to their terminal values: BTC at the last closed bar /
        # funding event before now, LUNA at delisted_ts - 1 (archive walk complete).
        assert dep.checkpoints.get(Dataset.OHLCV, BTC) == FEB_01 - HOUR
        assert dep.checkpoints.get(Dataset.FUNDING, BTC) == FEB_01 - FUNDING_STEP
        assert dep.checkpoints.get(Dataset.OHLCV, LUNA) == JAN_25 - 1
        assert dep.checkpoints.get(Dataset.FUNDING, LUNA) == JAN_25 - 1

        # --- boot 3: a further rerun is a no-op (idempotence) ----------------------
        report3 = dep.run()
        assert report3.failed_count == 0
        assert report3.total_rows == 0
        for dataset in (Dataset.OHLCV, Dataset.FUNDING):
            assert dep.snapshot(dataset).equals(reference[dataset])
    finally:
        dep.close()


def test_crash_schedule_covers_both_paths(tmp_path: Path) -> None:
    """Meta-check: an uninterrupted run performs exactly TOTAL_WRITES atomic writes,
    so the parameterized crash points genuinely span REST and Vision territory."""
    dep = Deployment(tmp_path)
    try:
        job, writer = dep.job(crash_after=None)
        report = job.run(
            datasets=[Dataset.OHLCV, Dataset.FUNDING],
            start_default=T0 - 60 * DAY,
            batch_days=BATCH_DAYS,
            now=FEB_01,
        )
        assert report.failed_count == 0
        assert writer.writes_done == TOTAL_WRITES
    finally:
        dep.close()
