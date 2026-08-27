"""Unit tests for alphaforge.data.ingest.backfill (BackfillJob).

In-memory doubles (FakeSource for the REST path, FakeVision for the archive path)
exercise the engine without any network: chunking math (batch_days windows,
listed_ts/delisted_ts clamps, Vision month walks, span trimming), the durability
contract (watermark advanced only after a write returns), per-instrument failure
isolation, run-log bookkeeping, and update() resuming purely from watermarks.

The full crash-resume equivalence test (crash at K, rerun, lake identical to an
uninterrupted run) lives in tests/integration/test_backfill_resume.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.core.errors import DataGapError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.backfill import BackfillJob, BackfillReport, InstrumentResult
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.ingest.retry import RateBudget
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import BinanceVisionClient, YearMonth
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.writer import NATURAL_KEY_COLUMN, LakeWriter, WriteStats

if TYPE_CHECKING:
    from collections.abc import Iterator

HOUR = 3_600_000
DAY = 86_400_000
FUNDING_STEP = 8 * HOUR

T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z (hour- and funding-grid aligned)
JAN_10 = T0 + 9 * DAY  # 2024-01-10T00:00:00Z
JAN_15 = T0 + 14 * DAY  # 2024-01-15T00:00:00Z
FEB_01 = 1_706_745_600_000  # 2024-02-01T00:00:00Z
MAR_01 = 1_709_251_200_000  # 2024-03-01T00:00:00Z
MAR_05 = MAR_01 + 4 * DAY  # 2024-03-05T00:00:00Z
JUN_01 = 1_717_200_000_000  # 2024-06-01T00:00:00Z

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"
LUNA = "BINANCE:PERP:LUNAUSDT"


# --------------------------------------------------------------------------- helpers


def make_inst(symbol: str, *, listed: Ms, delisted: Ms | None = None) -> Instrument:
    """A minimal valid USDT perp Instrument for tests."""
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
    """Grid timestamps t = k*step with t in [lo, hi) (lo/hi need not be aligned)."""
    first = -(-lo // step) * step
    return list(range(first, hi, step))


def ohlcv_table(instrument_id: str, ts_opens: list[int]) -> pa.Table:
    """OHLCV-schema table with one synthetic closed bar per ts_open."""
    n = len(ts_opens)
    ts = pa.array(ts_opens, type=pa.timestamp("ms", tz="UTC"))
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_open": ts,
            "open": pa.array([99.0] * n, type=pa.float64()),
            "high": pa.array([101.0] * n, type=pa.float64()),
            "low": pa.array([98.0] * n, type=pa.float64()),
            "close": pa.array([float(100 + i % 7) for i in range(n)], type=pa.float64()),
            "volume": pa.array([10.0] * n, type=pa.float64()),
            "quote_volume": pa.array([1_000.0] * n, type=pa.float64()),
            "n_trades": pa.array([42] * n, type=pa.int64()),
            "quality_flags": pa.array([0] * n, type=pa.int32()),
            "ingested_at": pa.array(
                [t + HOUR + 30_000 for t in ts_opens], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


def funding_table(instrument_id: str, ts_fundings: list[int]) -> pa.Table:
    """FUNDING-schema table with available_at = ts_funding + 5 min."""
    n = len(ts_fundings)
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_funding": pa.array(ts_fundings, type=pa.timestamp("ms", tz="UTC")),
            "rate": pa.array([1e-4] * n, type=pa.float64()),
            "available_at": pa.array(
                [t + 300_000 for t in ts_fundings], type=pa.timestamp("ms", tz="UTC")
            ),
            "ingested_at": pa.array(
                [t + 301_000 for t in ts_fundings], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


def read_leaf(path: Path) -> pa.Table:
    """Read one leaf file directly (``pq.read_table`` would run hive discovery on
    the ``key=value`` path segments and clash with the in-file ``instrument_id``)."""
    with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
        tbl: pa.Table = leaf.read()
    return tbl


class FakeSource(DataSource):
    """In-memory REST double: serves synthetic grid bars over a per-id data span.

    ``data[instrument_id] = (lo, hi)`` is the half-open span the venue has data
    for. Every fetch is recorded for chunking-math assertions. ``fail_ids``
    simulate a venue error for specific instruments; ``interrupt_after`` raises
    ``KeyboardInterrupt`` once that many OHLCV fetches have completed.
    """

    name = "fake"

    def __init__(
        self,
        data: dict[str, tuple[Ms, Ms]],
        *,
        fail_ids: frozenset[str] = frozenset(),
        unclosed_tail_bars: int = 0,
        interrupt_after: int | None = None,
    ) -> None:
        self._data = data
        self._fail_ids = fail_ids
        self._unclosed_tail_bars = unclosed_tail_bars
        self._interrupt_after = interrupt_after
        self.ohlcv_calls: list[tuple[str, Ms, Ms]] = []
        self.funding_calls: list[tuple[str, Ms, Ms]] = []

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        return []

    def fetch_ohlcv(
        self, instrument_id: str, *, tf: Timeframe, since: Ms, until: Ms, now: Ms | None = None
    ) -> pa.Table:
        if self._interrupt_after is not None and len(self.ohlcv_calls) >= self._interrupt_after:
            raise KeyboardInterrupt
        self.ohlcv_calls.append((instrument_id, since, until))
        if instrument_id in self._fail_ids:
            raise RuntimeError("venue exploded")
        lo, hi = self._data.get(instrument_id, (0, 0))
        cut = min(until, hi)
        if now is not None and self._unclosed_tail_bars == 0:
            cut = min(cut, now - tf.ms + 1)  # closed bars only: ts_open + Δ <= now
        return ohlcv_table(instrument_id, _grid(max(since, lo), cut, tf.ms))

    def fetch_funding(self, instrument_id: str, *, since: Ms, until: Ms) -> pa.Table:
        self.funding_calls.append((instrument_id, since, until))
        if instrument_id in self._fail_ids:
            raise RuntimeError("venue exploded")
        lo, hi = self._data.get(instrument_id, (0, 0))
        return funding_table(instrument_id, _grid(max(since, lo), min(until, hi), FUNDING_STEP))


class FakeVision(BinanceVisionClient):
    """In-memory archive double keyed by exchange symbol; months in ``missing`` 404."""

    def __init__(
        self,
        data: dict[str, tuple[Ms, Ms]],
        *,
        missing: frozenset[tuple[str, int, int]] = frozenset(),
    ) -> None:
        super().__init__()
        self._fake_data = data
        self._missing = missing
        self.ohlcv_month_calls: list[tuple[str, int, int]] = []
        self.funding_month_calls: list[tuple[str, int, int]] = []

    def _month_span(self, symbol: str, year: int, month: int) -> tuple[Ms, Ms]:
        if (symbol, year, month) in self._missing:
            raise DataGapError(f"no archive file for {symbol} {year}-{month:02d}")
        lo, hi = self._fake_data.get(symbol, (0, 0))
        ym = YearMonth(year, month)
        return max(lo, ym.first_ms()), min(hi, ym.next_month_first_ms())

    def fetch_ohlcv_month(
        self, symbol: str, year: int, month: int, *, now: Ms | None = None
    ) -> pa.Table:
        self.ohlcv_month_calls.append((symbol, year, month))
        lo, hi = self._month_span(symbol, year, month)
        if now is not None:
            hi = min(hi, now - HOUR + 1)
        return ohlcv_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, HOUR))

    def fetch_funding_month(self, symbol: str, year: int, month: int) -> pa.Table:
        self.funding_month_calls.append((symbol, year, month))
        lo, hi = self._month_span(symbol, year, month)
        return funding_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, FUNDING_STEP))


class SpyBudget(RateBudget):
    """RateBudget that records acquisitions (capacity is never binding here)."""

    def __init__(self) -> None:
        super().__init__(sleeper=lambda _s: None)
        self.acquired: list[int] = []

    def acquire(self, weight: int) -> None:
        self.acquired.append(weight)
        super().acquire(weight)


class Env:
    """One assembled BackfillJob with its stores and lake, on a tmp dir."""

    def __init__(
        self,
        tmp_path: Path,
        source: FakeSource,
        vision: FakeVision,
        instruments: list[Instrument],
        *,
        now: Ms,
        rate_budget: RateBudget | None = None,
    ) -> None:
        self.paths = LakePaths(tmp_path / "lake")
        self.writer = LakeWriter(self.paths)
        self.checkpoints = CheckpointStore(tmp_path / "ops.sqlite")
        self.store = InstrumentStore(tmp_path / "ops.sqlite")
        for inst in instruments:
            self.store.upsert(inst, as_of=T0)
        self.source = source
        self.vision = vision
        self.now = now
        self.job = BackfillJob(
            source, vision, self.writer, self.checkpoints, self.store, rate_budget=rate_budget
        )

    def close(self) -> None:
        self.checkpoints.close()
        self.store.close()

    def lake_keys(self, dataset: Dataset, instrument_id: str) -> list[int]:
        """Sorted natural-key epoch-ms of every row in the lake for the pair."""
        key = NATURAL_KEY_COLUMN[dataset]
        out: list[int] = []
        for path in self.paths.partition_paths(dataset, instrument_id):
            col = read_leaf(path).column(key).to_numpy(zero_copy_only=False)
            out.extend(col.astype("datetime64[ms]").astype("int64").tolist())
        return sorted(out)


@pytest.fixture
def live_env(tmp_path: Path) -> Iterator[Env]:
    """One LIVE instrument (BTC, listed JAN_10) with venue data through ``now``."""
    env = Env(
        tmp_path,
        FakeSource({BTC: (JAN_10, JUN_01)}),
        FakeVision({}),
        [make_inst("BTCUSDT", listed=JAN_10)],
        now=FEB_01,
    )
    yield env
    env.close()


@pytest.fixture
def delisted_env(tmp_path: Path) -> Iterator[Env]:
    """One DELISTED instrument (LUNA, JAN_10 → MAR_05) served by the archive."""
    env = Env(
        tmp_path,
        FakeSource({}),
        FakeVision({"LUNAUSDT": (JAN_10, MAR_05)}),
        [make_inst("LUNAUSDT", listed=JAN_10, delisted=MAR_05)],
        now=JUN_01,
    )
    yield env
    env.close()


# --------------------------------------------------------------------------- validation


class TestValidation:
    def test_empty_datasets_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            live_env.job.run(datasets=[], start_default=T0, now=live_env.now)

    def test_duplicate_datasets_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            live_env.job.run(
                datasets=[Dataset.OHLCV, Dataset.OHLCV], start_default=T0, now=live_env.now
            )

    def test_universe_membership_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="cannot be ingested"):
            live_env.job.run(
                datasets=[Dataset.UNIVERSE_MEMBERSHIP], start_default=T0, now=live_env.now
            )

    def test_bad_batch_days_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="batch_days"):
            live_env.job.run(
                datasets=[Dataset.OHLCV], start_default=T0, batch_days=0, now=live_env.now
            )

    def test_negative_start_default_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="start_default"):
            live_env.job.run(datasets=[Dataset.OHLCV], start_default=-1, now=live_env.now)

    def test_unknown_instrument_ids_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="unknown instrument_ids"):
            live_env.job.run(
                datasets=[Dataset.OHLCV],
                instrument_ids=[BTC, ETH],
                start_default=T0,
                now=live_env.now,
            )

    def test_empty_instrument_selection_rejected(self, live_env: Env) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            live_env.job.run(
                datasets=[Dataset.OHLCV], instrument_ids=[], start_default=T0, now=live_env.now
            )

    def test_nothing_fetched_before_validation_failure(self, live_env: Env) -> None:
        with pytest.raises(ValueError):
            live_env.job.run(datasets=[], start_default=T0, now=live_env.now)
        assert live_env.source.ohlcv_calls == []
        assert live_env.checkpoints.open_runs() == []


# --------------------------------------------------------------------------- live path


class TestLivePath:
    def test_chunking_walks_batch_days_windows(self, live_env: Env) -> None:
        until = JAN_10 + 5 * DAY
        live_env.job.run(
            datasets=[Dataset.OHLCV],
            start_default=T0,
            until=until,
            batch_days=2,
            now=live_env.now,
        )
        assert live_env.source.ohlcv_calls == [
            (BTC, JAN_10, JAN_10 + 2 * DAY),
            (BTC, JAN_10 + 2 * DAY, JAN_10 + 4 * DAY),
            (BTC, JAN_10 + 4 * DAY, until),
        ]

    def test_start_is_max_of_listed_ts_and_start_default(self, live_env: Env) -> None:
        # listed_ts (JAN_10) > start_default (T0): the fetch starts at listing.
        live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        assert live_env.source.ohlcv_calls[0][1] == JAN_10
        # start_default (JAN_15) > listed_ts: the requested floor dominates.
        live_env.checkpoints.reset(Dataset.OHLCV, BTC)
        live_env.source.ohlcv_calls.clear()
        live_env.job.run(datasets=[Dataset.OHLCV], start_default=JAN_15, now=live_env.now)
        assert live_env.source.ohlcv_calls[0][1] == JAN_15

    def test_until_none_and_future_until_clamp_to_now(self, live_env: Env) -> None:
        live_env.job.run(
            datasets=[Dataset.OHLCV], start_default=T0, batch_days=10_000, now=live_env.now
        )
        assert live_env.source.ohlcv_calls[-1][2] == live_env.now
        live_env.checkpoints.reset(Dataset.OHLCV, BTC)
        live_env.source.ohlcv_calls.clear()
        live_env.job.run(
            datasets=[Dataset.OHLCV],
            start_default=T0,
            until=JUN_01,  # future relative to now=FEB_01
            batch_days=10_000,
            now=live_env.now,
        )
        assert live_env.source.ohlcv_calls[-1][2] == live_env.now

    def test_watermark_equals_last_written_key_and_no_future_close(self, live_env: Env) -> None:
        report = live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        keys = live_env.lake_keys(Dataset.OHLCV, BTC)
        assert keys[0] == JAN_10
        assert keys[-1] == live_env.now - HOUR  # last bar whose close <= now
        assert max(k + HOUR for k in keys) <= live_env.now
        assert live_env.checkpoints.get(Dataset.OHLCV, BTC) == keys[-1]
        (result,) = report.results
        assert result.status == "ok"
        assert result.rows == len(keys)
        assert result.watermark == keys[-1]

    def test_unclosed_bars_dropped_before_write_and_watermark(self, tmp_path: Path) -> None:
        """A misbehaving source returning the in-progress bar cannot poison the lake."""
        env = Env(
            tmp_path,
            FakeSource({BTC: (JAN_10, FEB_01 + DAY)}, unclosed_tail_bars=1),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10)],
            now=FEB_01,
        )
        try:
            env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=env.now)
            keys = env.lake_keys(Dataset.OHLCV, BTC)
            assert keys[-1] == env.now - HOUR  # ts_open = now (close in future) was dropped
            assert env.checkpoints.get(Dataset.OHLCV, BTC) == env.now - HOUR
        finally:
            env.close()

    def test_empty_fetch_is_ok_with_no_watermark(self, tmp_path: Path) -> None:
        env = Env(
            tmp_path,
            FakeSource({}),  # venue has nothing for BTC
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10)],
            now=FEB_01,
        )
        try:
            report = env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=env.now)
            (result,) = report.results
            assert result.status == "ok"
            assert result.rows == 0
            assert result.batches == 0
            assert result.watermark is None
            assert env.checkpoints.get(Dataset.OHLCV, BTC) is None
        finally:
            env.close()

    def test_funding_ingested_and_watermarked(self, live_env: Env) -> None:
        live_env.job.run(datasets=[Dataset.FUNDING], start_default=T0, now=live_env.now)
        keys = live_env.lake_keys(Dataset.FUNDING, BTC)
        assert keys[0] == JAN_10
        assert keys == _grid(JAN_10, FEB_01, FUNDING_STEP)
        assert live_env.checkpoints.get(Dataset.FUNDING, BTC) == keys[-1]
        assert live_env.checkpoints.get(Dataset.OHLCV, BTC) is None  # untouched

    def test_both_datasets_processed_in_request_order(self, live_env: Env) -> None:
        report = live_env.job.run(
            datasets=[Dataset.FUNDING, Dataset.OHLCV], start_default=T0, now=live_env.now
        )
        assert [r.dataset for r in report.results] == [Dataset.FUNDING, Dataset.OHLCV]
        assert report.ok_count == 2


# --------------------------------------------------------------------------- resume


class TestResume:
    def test_second_run_resumes_from_watermark_plus_one(self, live_env: Env) -> None:
        live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, until=JAN_15, now=live_env.now)
        wm = live_env.checkpoints.get(Dataset.OHLCV, BTC)
        assert wm == JAN_15 - HOUR
        live_env.source.ohlcv_calls.clear()
        live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        assert live_env.source.ohlcv_calls[0][1] == wm + 1
        assert live_env.lake_keys(Dataset.OHLCV, BTC) == _grid(JAN_10, FEB_01, HOUR)

    def test_up_to_date_pair_is_skipped_without_fetching(self, live_env: Env) -> None:
        live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        live_env.source.ohlcv_calls.clear()
        report = live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        (result,) = report.results
        # watermark = last bar open (now - Δ); start = wm + 1 < until = now, so a
        # sliver remains and is re-walked but yields nothing new.
        assert result.rows == 0
        assert live_env.lake_keys(Dataset.OHLCV, BTC) == _grid(JAN_10, FEB_01, HOUR)

    def test_watermark_advances_only_after_write(self, tmp_path: Path) -> None:
        """A writer that fails on batch N leaves the watermark at batch N-1's key."""

        class FailingWriter(LakeWriter):
            def __init__(self, paths: LakePaths, fail_on_call: int) -> None:
                super().__init__(paths)
                self.calls = 0
                self._fail_on_call = fail_on_call

            def write(
                self,
                dataset: Dataset,
                tbl: pa.Table,
                *,
                tf: Timeframe = Timeframe.H1,
                now: Ms | None = None,
            ) -> WriteStats:
                self.calls += 1
                if self.calls == self._fail_on_call:
                    raise RuntimeError("disk on fire")
                return super().write(dataset, tbl, tf=tf, now=now)

        paths = LakePaths(tmp_path / "lake")
        writer = FailingWriter(paths, fail_on_call=3)
        with (
            CheckpointStore(tmp_path / "ops.sqlite") as checkpoints,
            InstrumentStore(tmp_path / "ops.sqlite") as store,
        ):
            store.upsert(make_inst("BTCUSDT", listed=JAN_10), as_of=T0)
            job = BackfillJob(
                FakeSource({BTC: (JAN_10, JUN_01)}), FakeVision({}), writer, checkpoints, store
            )
            report = job.run(datasets=[Dataset.OHLCV], start_default=T0, batch_days=2, now=FEB_01)
            (result,) = report.results
            assert result.status == "failed"
            assert "disk on fire" in result.error
            # Two batches of 2 days landed durably; the watermark is their max key
            # and exactly matches what is physically in the lake.
            expected_wm = JAN_10 + 4 * DAY - HOUR
            assert checkpoints.get(Dataset.OHLCV, BTC) == expected_wm
            keys = [
                int(k)
                for p in paths.partition_paths(Dataset.OHLCV, BTC)
                for k in read_leaf(p)
                .column("ts_open")
                .to_numpy(zero_copy_only=False)
                .astype("datetime64[ms]")
                .astype("int64")
            ]
            assert max(keys) == expected_wm
            assert result.watermark == expected_wm


# --------------------------------------------------------------------------- vision path


class TestVisionPath:
    def test_delisted_routes_to_vision_months_only(self, delisted_env: Env) -> None:
        delisted_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=delisted_env.now)
        assert delisted_env.source.ohlcv_calls == []  # REST never touched
        assert delisted_env.vision.ohlcv_month_calls == [
            ("LUNAUSDT", 2024, 1),
            ("LUNAUSDT", 2024, 2),
            ("LUNAUSDT", 2024, 3),
        ]

    def test_rows_trimmed_to_effective_span(self, tmp_path: Path) -> None:
        """Archive months hold data outside the span; only [start, until) is written."""
        env = Env(
            tmp_path,
            FakeSource({}),
            FakeVision({"LUNAUSDT": (JAN_10, MAR_05)}),
            [make_inst("LUNAUSDT", listed=JAN_10, delisted=MAR_05)],
            now=JUN_01,
        )
        try:
            env.job.run(datasets=[Dataset.OHLCV], start_default=JAN_15, now=env.now)
            keys = env.lake_keys(Dataset.OHLCV, LUNA)
            assert keys[0] == JAN_15  # January rows before start_default trimmed
            assert keys[-1] == MAR_05 - HOUR  # nothing at/after delisting
            assert keys == _grid(JAN_15, MAR_05, HOUR)
        finally:
            env.close()

    def test_watermark_converges_and_next_run_skips(self, delisted_env: Env) -> None:
        delisted_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=delisted_env.now)
        assert delisted_env.checkpoints.get(Dataset.OHLCV, LUNA) == MAR_05 - 1
        delisted_env.vision.ohlcv_month_calls.clear()
        report = delisted_env.job.run(
            datasets=[Dataset.OHLCV], start_default=T0, now=delisted_env.now
        )
        (result,) = report.results
        assert result.status == "skipped"
        assert delisted_env.vision.ohlcv_month_calls == []  # archive not re-polled

    def test_gap_month_recorded_not_fatal_and_blocks_convergence(self, tmp_path: Path) -> None:
        env = Env(
            tmp_path,
            FakeSource({}),
            FakeVision({"LUNAUSDT": (JAN_10, MAR_05)}, missing=frozenset({("LUNAUSDT", 2024, 2)})),
            [make_inst("LUNAUSDT", listed=JAN_10, delisted=MAR_05)],
            now=JUN_01,
        )
        try:
            report = env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=env.now)
            (result,) = report.results
            assert result.status == "ok"
            assert result.gap_months == ("2024-02",)
            keys = env.lake_keys(Dataset.OHLCV, LUNA)
            missing = sorted(set(_grid(JAN_10, MAR_05, HOUR)) - set(keys))
            assert missing == _grid(FEB_01, MAR_01, HOUR)  # exactly the gapped month
            # No convergence to delisted_ts - 1: the watermark is the max written key.
            assert env.checkpoints.get(Dataset.OHLCV, LUNA) == MAR_05 - HOUR
        finally:
            env.close()

    def test_funding_via_vision_with_convergence(self, delisted_env: Env) -> None:
        delisted_env.job.run(datasets=[Dataset.FUNDING], start_default=T0, now=delisted_env.now)
        keys = delisted_env.lake_keys(Dataset.FUNDING, LUNA)
        assert keys == _grid(JAN_10, MAR_05, FUNDING_STEP)
        assert delisted_env.checkpoints.get(Dataset.FUNDING, LUNA) == MAR_05 - 1
        assert delisted_env.vision.funding_month_calls == [
            ("LUNAUSDT", 2024, 1),
            ("LUNAUSDT", 2024, 2),
            ("LUNAUSDT", 2024, 3),
        ]

    def test_rate_budget_charged_once_per_month_download(self, tmp_path: Path) -> None:
        budget = SpyBudget()
        env = Env(
            tmp_path,
            FakeSource({}),
            FakeVision({"LUNAUSDT": (JAN_10, MAR_05)}),
            [make_inst("LUNAUSDT", listed=JAN_10, delisted=MAR_05)],
            now=JUN_01,
            rate_budget=budget,
        )
        try:
            env.job.run(datasets=[Dataset.OHLCV, Dataset.FUNDING], start_default=T0, now=env.now)
            assert budget.acquired == [1] * 6  # 3 OHLCV months + 3 funding months
        finally:
            env.close()


# --------------------------------------------------------------------------- isolation


class TestFailureIsolation:
    def test_one_failing_instrument_does_not_abort_others(self, tmp_path: Path) -> None:
        env = Env(
            tmp_path,
            FakeSource({BTC: (JAN_10, JUN_01), ETH: (JAN_10, JUN_01)}, fail_ids=frozenset({BTC})),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10), make_inst("ETHUSDT", listed=JAN_10)],
            now=FEB_01,
        )
        try:
            report = env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=env.now)
            assert report.failed_count == 1
            assert report.ok_count == 1
            by_id = {r.instrument_id: r for r in report.results}
            assert by_id[BTC].status == "failed"
            assert "venue exploded" in by_id[BTC].error
            assert by_id[ETH].status == "ok"
            assert env.lake_keys(Dataset.OHLCV, ETH) == _grid(JAN_10, FEB_01, HOUR)
            # Run-log row records the partial failure.
            run = env.checkpoints.get_run(report.run_id)
            assert run is not None
            assert run.status == "failed"
            assert run.detail == report.summary()
        finally:
            env.close()

    def test_keyboard_interrupt_propagates_after_durability_point(self, tmp_path: Path) -> None:
        env = Env(
            tmp_path,
            FakeSource({BTC: (JAN_10, JUN_01)}, interrupt_after=2),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10)],
            now=FEB_01,
        )
        try:
            with pytest.raises(KeyboardInterrupt):
                env.job.run(datasets=[Dataset.OHLCV], start_default=T0, batch_days=2, now=env.now)
            # The two completed fetch+write batches are durable and watermarked.
            assert env.checkpoints.get(Dataset.OHLCV, BTC) == JAN_10 + 4 * DAY - HOUR
            # The run-log row was finished (as failed), not left dangling open.
            assert env.checkpoints.open_runs() == []
        finally:
            env.close()

    def test_run_log_ok_on_clean_run(self, live_env: Env) -> None:
        report = live_env.job.run(datasets=[Dataset.OHLCV], start_default=T0, now=live_env.now)
        run = live_env.checkpoints.get_run(report.run_id)
        assert run is not None
        assert run.status == "ok"
        assert run.detail == report.summary()
        assert run.job == "backfill:ohlcv"
        assert live_env.checkpoints.open_runs() == []


# --------------------------------------------------------------------------- update()


class TestUpdate:
    def test_update_resumes_each_pair_from_its_own_watermark(self, tmp_path: Path) -> None:
        env = Env(
            tmp_path,
            FakeSource({BTC: (JAN_10, JUN_01), ETH: (JAN_10, JUN_01)}),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10), make_inst("ETHUSDT", listed=JAN_10)],
            now=FEB_01,
        )
        try:
            env.job.run(datasets=[Dataset.OHLCV, Dataset.FUNDING], start_default=T0, now=env.now)
            wm_ohlcv = env.checkpoints.get(Dataset.OHLCV, BTC)
            wm_funding = env.checkpoints.get(Dataset.FUNDING, BTC)
            assert wm_ohlcv is not None and wm_funding is not None
            env.source.ohlcv_calls.clear()
            env.source.funding_calls.clear()
            as_of = FEB_01 + 2 * DAY
            report = env.job.update(as_of=as_of)
            assert report.job == "update:ohlcv,funding"
            assert {c[1] for c in env.source.ohlcv_calls} == {wm_ohlcv + 1}
            assert {c[1] for c in env.source.funding_calls} == {wm_funding + 1}
            assert env.checkpoints.get(Dataset.OHLCV, BTC) == as_of - HOUR
            assert env.lake_keys(Dataset.OHLCV, BTC) == _grid(JAN_10, as_of, HOUR)
        finally:
            env.close()

    def test_update_without_watermarks_skips_everything(self, live_env: Env) -> None:
        report = live_env.job.update(as_of=FEB_01)
        assert report.skipped_count == len(report.results)
        assert report.total_rows == 0
        assert live_env.source.ohlcv_calls == []

    def test_update_equivalent_to_uninterrupted_backfill(self, tmp_path: Path) -> None:
        """backfill-to-mid + update == one full backfill (lake keys identical)."""
        as_of = FEB_01
        env_a = Env(
            tmp_path / "a",
            FakeSource({BTC: (JAN_10, JUN_01)}),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10)],
            now=as_of,
        )
        env_b = Env(
            tmp_path / "b",
            FakeSource({BTC: (JAN_10, JUN_01)}),
            FakeVision({}),
            [make_inst("BTCUSDT", listed=JAN_10)],
            now=as_of,
        )
        try:
            env_a.job.run(datasets=[Dataset.OHLCV, Dataset.FUNDING], start_default=T0, now=as_of)
            env_b.job.run(
                datasets=[Dataset.OHLCV, Dataset.FUNDING],
                start_default=T0,
                until=JAN_15,
                now=as_of,
            )
            env_b.job.update(as_of=as_of)
            for dataset in (Dataset.OHLCV, Dataset.FUNDING):
                assert env_a.lake_keys(dataset, BTC) == env_b.lake_keys(dataset, BTC)
                assert env_a.checkpoints.get(dataset, BTC) == env_b.checkpoints.get(dataset, BTC)
        finally:
            env_a.close()
            env_b.close()


# --------------------------------------------------------------------------- report


class TestReport:
    def test_counts_total_rows_failures_and_summary(self) -> None:
        results = (
            InstrumentResult(
                instrument_id=BTC,
                dataset=Dataset.OHLCV,
                status="ok",
                rows=10,
                batches=2,
                watermark=123,
            ),
            InstrumentResult(
                instrument_id=ETH,
                dataset=Dataset.OHLCV,
                status="failed",
                rows=0,
                batches=0,
                watermark=None,
                error="RuntimeError('x')",
            ),
            InstrumentResult(
                instrument_id=LUNA,
                dataset=Dataset.FUNDING,
                status="skipped",
                rows=0,
                batches=0,
                watermark=None,
            ),
        )
        report = BackfillReport(
            run_id="r1", job="backfill:ohlcv", started=1, finished=2, results=results
        )
        assert report.ok_count == 1
        assert report.failed_count == 1
        assert report.skipped_count == 1
        assert report.total_rows == 10
        assert report.failures() == (results[1],)
        assert report.summary() == "ok=1 failed=1 skipped=1 rows=10"
