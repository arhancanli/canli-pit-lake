"""Unit tests for alphaforge.data.store.resample (Resampler, resample_bars).

Covers: exact field-by-field aggregation on a hand-built 8-bar fixture, bitwise OR
flag propagation, null propagation for quote_volume/n_trades, the completeness rule
(a missing 1h bar ⇒ a MISSING derived bar, surfaced by ``gaps(tf=H4)``), the
closed-bucket rule, incremental no-op re-runs, the PIT boundary on derived bars
(invisible at close - 1 ms via the reader), tf-parameterized flag masking on derived
bars, and dataset separation (tf selects ohlcv vs ohlcv_4h/ohlcv_1d files).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import Dataset, QualityFlag, ohlcv_dataset, schema_for
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.resample import Resampler, ResampleStats, resample_bars
from alphaforge.data.store.writer import LakeWriter

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"

HOUR = 3_600_000
H4 = 4 * HOUR
DAY = 24 * HOUR
T0 = 1_704_153_600_000  # 2024-01-02T00:00:00Z — aligned for 1h, 4h and 1d
FAR_FUTURE = 1_900_000_000_000
INGESTED = T0 + 10 * DAY  # fixed source ingested_at, distinct from any resample `now`

# Hand-built 8-bar fixture (two complete 4h buckets). Volumes are exact binary
# fractions so the expected sums are exact to the last bit.
OPENS = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
HIGHS = [110.0, 115.0, 112.0, 111.0, 120.0, 118.0, 125.0, 119.0]
LOWS = [90.0, 85.0, 88.0, 92.0, 80.0, 83.0, 79.0, 81.0]
CLOSES = [101.0, 102.0, 103.0, 99.5, 105.0, 106.0, 107.0, 108.25]
VOLUMES = [1.5, 2.25, 3.0, 4.5, 5.0, 6.25, 7.5, 8.0]
QUOTE_VOLUMES: list[float | None] = [150.0, 225.0, 300.0, 450.0, 500.0, 625.0, 750.0, 800.0]
N_TRADES: list[int | None] = [10, 20, 30, 40, 50, 60, 70, 80]
FLAGS = [0, 2, 4, 0, 16, 0, 0, 8]  # bucket0 → 2|4 = 6, bucket1 → 16|8 = 24


def bar(
    instrument_id: str,
    ts_open: int,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 1.0,
    quote_volume: float | None = 100.0,
    n_trades: int | None = 10,
    flags: int = 0,
) -> dict[str, object]:
    return {
        "instrument_id": instrument_id,
        "ts_open": ts_open,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "n_trades": n_trades,
        "quality_flags": flags,
        "ingested_at": INGESTED,
    }


def ohlcv_table(*rows: dict[str, object]) -> pa.Table:
    schema = schema_for(Dataset.OHLCV)
    return pa.Table.from_pydict(
        {
            name: pa.array([r[name] for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )


def fixture_bars(instrument_id: str = BTC, n: int = 8) -> pa.Table:
    return ohlcv_table(
        *(
            bar(
                instrument_id,
                T0 + i * HOUR,
                open_=OPENS[i],
                high=HIGHS[i],
                low=LOWS[i],
                close=CLOSES[i],
                volume=VOLUMES[i],
                quote_volume=QUOTE_VOLUMES[i],
                n_trades=N_TRADES[i],
                flags=FLAGS[i],
            )
            for i in range(n)
        )
    )


def read_leaf(path: Path) -> pa.Table:
    with pq.ParquetFile(path) as f:  # type: ignore[no-untyped-call]
        result: pa.Table = f.read()
    return result


def col(tbl: pa.Table, name: str) -> list[object]:
    out: list[object] = tbl.column(name).to_pylist()
    return out


def epoch_col(tbl: pa.Table, name: str) -> list[int]:
    out: list[int] = tbl.column(name).cast(pa.int64()).to_pylist()
    return out


@pytest.fixture
def paths(tmp_path: Path) -> LakePaths:
    return LakePaths(tmp_path / "lake")


@pytest.fixture
def writer(paths: LakePaths) -> LakeWriter:
    return LakeWriter(paths)


@pytest.fixture
def resampler(paths: LakePaths) -> Resampler:
    return Resampler(paths)


@pytest.fixture
def reader(paths: LakePaths) -> PITDataReader:
    return PITDataReader(paths)


def seed(writer: LakeWriter, tbl: pa.Table) -> None:
    writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)


# ---------------------------------------------------------------- argument contracts


class TestArguments:
    def test_h1_target_rejected(self, resampler: Resampler) -> None:
        with pytest.raises(ValueError, match="native ingested timeframe"):
            resampler.resample([BTC], target=Timeframe.H1, now=FAR_FUTURE)

    def test_resample_bars_rejects_non_multiple(self) -> None:
        with pytest.raises(ValueError, match="strict integer multiple"):
            resample_bars(fixture_bars(), source=Timeframe.H4, target=Timeframe.H1, now=FAR_FUTURE)
        with pytest.raises(ValueError, match="strict integer multiple"):
            resample_bars(fixture_bars(), source=Timeframe.H1, target=Timeframe.H1, now=FAR_FUTURE)

    def test_empty_instrument_list_is_noop(self, resampler: Resampler) -> None:
        stats = resampler.resample([], target=Timeframe.H4, now=FAR_FUTURE)
        assert stats == ResampleStats(
            target=Timeframe.H4,
            instruments=0,
            source_rows=0,
            bars_written=0,
            buckets_incomplete=0,
            buckets_unclosed=0,
        )

    def test_duplicate_instrument_ids_processed_once(
        self, writer: LakeWriter, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars())
        stats = resampler.resample([BTC, BTC], target=Timeframe.H4, now=FAR_FUTURE)
        assert stats.instruments == 1
        assert stats.source_rows == 8
        assert stats.bars_written == 2


# ---------------------------------------------------------------- exact aggregation


class TestExactAggregation:
    NOW = T0 + 8 * HOUR  # both 4h buckets close exactly at `now`

    def test_h4_every_field_to_the_last_decimal(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars())
        stats = resampler.resample([BTC], target=Timeframe.H4, now=self.NOW)
        assert stats == ResampleStats(
            target=Timeframe.H4,
            instruments=1,
            source_rows=8,
            bars_written=2,
            buckets_incomplete=0,
            buckets_unclosed=0,
        )
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert out.num_rows == 2
        assert col(out, "instrument_id") == [BTC, BTC]
        assert epoch_col(out, "ts_open") == [T0, T0 + H4]
        assert col(out, "open") == [100.0, 104.0]
        assert col(out, "high") == [115.0, 125.0]
        assert col(out, "low") == [85.0, 79.0]
        assert col(out, "close") == [99.5, 108.25]
        assert col(out, "volume") == [11.25, 26.75]
        assert col(out, "quote_volume") == [1125.0, 2675.0]
        assert col(out, "n_trades") == [100, 260]
        assert col(out, "quality_flags") == [6, 24]  # bitwise OR of constituents
        assert epoch_col(out, "ingested_at") == [self.NOW, self.NOW]

    def test_d1_aggregates_24_bars(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        seed(
            writer,
            ohlcv_table(
                *(
                    bar(
                        BTC,
                        T0 + i * HOUR,
                        open_=100.0 + i,
                        high=200.0 + (i % 7),
                        low=50.0 - (i % 5),
                        close=101.0 + i,
                        volume=0.25 * (i + 1),
                        quote_volume=float(10 * (i + 1)),
                        n_trades=i + 1,
                        flags=(2 if i == 3 else 0) | (16 if i == 20 else 0),
                    )
                    for i in range(24)
                )
            ),
        )
        stats = resampler.resample([BTC], target=Timeframe.D1, now=T0 + DAY)
        assert stats.bars_written == 1
        out = read_leaf(paths.partition_path(Dataset.OHLCV_1D, BTC, 2024))
        assert epoch_col(out, "ts_open") == [T0]
        assert col(out, "open") == [100.0]
        assert col(out, "high") == [206.0]  # max over i%7 peaks at i=6,13,20
        assert col(out, "low") == [46.0]  # min over 50 - i%5
        assert col(out, "close") == [124.0]  # close of bar i=23
        assert col(out, "volume") == [75.0]  # 0.25 · Σ(1..24) = 0.25 · 300
        assert col(out, "quote_volume") == [3000.0]  # 10 · 300
        assert col(out, "n_trades") == [300]
        assert col(out, "quality_flags") == [18]  # 2 | 16
        assert epoch_col(out, "ingested_at") == [T0 + DAY]

    def test_flag_or_inherits_every_defect(self, writer: LakeWriter, paths: LakePaths) -> None:
        # All five declared bits spread across constituents OR into the derived bar,
        # stored UNMASKED (PIT masking is the reader's job, applied at read time).
        bits = [
            int(f)
            for f in (
                QualityFlag.GAP_FILLED_NONE,
                QualityFlag.OUTLIER_RETURN,
                QualityFlag.VOLUME_ANOMALY,
                QualityFlag.BAD_PRINT_SUSPECT,
            )
        ]
        seed(
            writer,
            ohlcv_table(*(bar(BTC, T0 + i * HOUR, flags=bits[i]) for i in range(4))),
        )
        Resampler(paths).resample([BTC], target=Timeframe.H4, now=T0 + H4)
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert col(out, "quality_flags") == [1 | 2 | 4 | 8]


# ---------------------------------------------------------------- null propagation


class TestNullPropagation:
    def test_one_null_n_trades_nulls_the_bucket(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        rows = [bar(BTC, T0 + i * HOUR, n_trades=None if i == 2 else 10) for i in range(8)]
        seed(writer, ohlcv_table(*rows))
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert col(out, "n_trades") == [None, 40]
        assert col(out, "quote_volume") == [400.0, 400.0]  # unaffected column intact

    def test_one_null_quote_volume_nulls_the_bucket(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        rows = [bar(BTC, T0 + i * HOUR, quote_volume=None if i == 5 else 100.0) for i in range(8)]
        seed(writer, ohlcv_table(*rows))
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert col(out, "quote_volume") == [400.0, None]
        assert col(out, "n_trades") == [40, 40]


# ---------------------------------------------------------------- completeness rule


class TestCompletenessRule:
    def test_missing_source_bar_skips_the_bucket(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler, reader: PITDataReader
    ) -> None:
        # Drop the bar at T0 + 2h: bucket0 has 3/4 constituents → NOTHING is written
        # for it; the 1h gap propagates to a missing 4h bar.
        tbl = fixture_bars().filter(pa.array([i != 2 for i in range(8)], type=pa.bool_()))
        seed(writer, tbl)
        stats = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        assert stats.bars_written == 1
        assert stats.buckets_incomplete == 1
        assert stats.buckets_unclosed == 0
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert epoch_col(out, "ts_open") == [T0 + H4]
        # GapCheck-style tooling sees the hole in the derived dataset:
        assert reader.gaps(BTC, start=T0, end=T0 + 8 * HOUR, tf=Timeframe.H4) == [T0]

    def test_unclosed_bucket_skipped(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        # At now = T0 + 7h bucket1 (closes T0 + 8h) is still in progress: skipped,
        # counted as unclosed even though its present bars are also incomplete.
        seed(writer, fixture_bars(n=7))
        stats = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 7 * HOUR)
        assert stats.bars_written == 1
        assert stats.buckets_unclosed == 1
        assert stats.buckets_incomplete == 0
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert epoch_col(out, "ts_open") == [T0]

    def test_bucket_closing_exactly_at_now_is_emitted(
        self, writer: LakeWriter, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars())
        assert resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR).bars_written == 2

    def test_bucket_closing_one_ms_after_now_is_not(
        self, writer: LakeWriter, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars())
        stats = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR - 1)
        assert stats.bars_written == 1
        assert stats.buckets_unclosed == 1


# ---------------------------------------------------------------- incremental runs


class TestIncremental:
    def test_rerun_with_no_new_data_reads_and_writes_nothing(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars())
        first = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        assert first.bars_written == 2
        leaf = paths.partition_path(Dataset.OHLCV_4H, BTC, 2024)
        before = leaf.read_bytes()
        second = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 9 * HOUR)
        assert second.source_rows == 0
        assert second.bars_written == 0
        assert second.buckets_incomplete == 0
        assert second.buckets_unclosed == 0
        assert leaf.read_bytes() == before  # idempotent: the file was not rewritten

    def test_new_source_bars_extend_without_recomputing_old_buckets(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars(n=4))
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 4 * HOUR)
        seed(writer, fixture_bars(n=8))  # 4 new bars (first 4 dedupe in the 1h lake)
        stats = resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        assert stats.source_rows == 4  # only bars after the derived watermark are read
        assert stats.bars_written == 1
        out = read_leaf(paths.partition_path(Dataset.OHLCV_4H, BTC, 2024))
        assert epoch_col(out, "ts_open") == [T0, T0 + H4]
        # bucket0 kept its original run's ingested_at — it was not recomputed.
        assert epoch_col(out, "ingested_at") == [T0 + 4 * HOUR, T0 + 8 * HOUR]

    def test_multi_instrument_run(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler
    ) -> None:
        seed(writer, fixture_bars(BTC))
        seed(writer, fixture_bars(ETH, n=4))
        stats = resampler.resample([BTC, ETH], target=Timeframe.H4, now=T0 + 8 * HOUR)
        assert stats.instruments == 2
        assert stats.bars_written == 3
        eth_leaf = read_leaf(paths.partition_path(Dataset.OHLCV_4H, ETH, 2024))
        assert epoch_col(eth_leaf, "ts_open") == [T0]


# ------------------------------------------------------- PIT + dataset separation


class TestDerivedBarsThroughReader:
    def test_pit_boundary_on_derived_bar(
        self, writer: LakeWriter, resampler: Resampler, reader: PITDataReader
    ) -> None:
        seed(writer, fixture_bars())
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        # The 4h bar opening at T0 closes at T0 + 4h: invisible 1 ms before, visible at.
        before = reader.ohlcv([BTC], start=T0, end=T0 + H4, as_of=T0 + H4 - 1, tf=Timeframe.H4)
        assert before.num_rows == 0
        out = reader.ohlcv([BTC], start=T0, end=T0 + H4, as_of=T0 + H4, tf=Timeframe.H4)
        assert epoch_col(out, "ts_open") == [T0]
        # Lag-0 bits (OUTLIER|VOLUME_ANOMALY = 6) are visible at the derived bar's close.
        assert col(out, "quality_flags") == [6]

    def test_flag_lag_measured_in_target_bars(
        self, writer: LakeWriter, resampler: Resampler, reader: PITDataReader
    ) -> None:
        # Derived bucket1 flags = EXCHANGE_DOWNTIME(16, lag 1) | BAD_PRINT(8, lag 2):
        # 0 at its close, 16 one 4h bar later, 24 two 4h bars later.
        seed(writer, fixture_bars())
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        ts1 = T0 + H4

        def flags_at(as_of: int) -> object:
            out = reader.ohlcv([BTC], start=ts1, end=ts1 + H4, as_of=as_of, tf=Timeframe.H4)
            return col(out, "quality_flags")[0]

        assert flags_at(ts1 + H4) == 0
        assert flags_at(ts1 + 2 * H4) == 16
        assert flags_at(ts1 + 3 * H4 - 1) == 16
        assert flags_at(ts1 + 3 * H4) == 24

    def test_tf_selects_the_dataset(
        self, writer: LakeWriter, paths: LakePaths, resampler: Resampler, reader: PITDataReader
    ) -> None:
        seed(writer, fixture_bars())
        resampler.resample([BTC], target=Timeframe.H4, now=T0 + 8 * HOUR)
        # Derived files live under ohlcv_4h/, native under ohlcv/.
        assert paths.partition_path(Dataset.OHLCV_4H, BTC, 2024).is_file()
        h1 = reader.ohlcv([BTC], start=T0, end=T0 + 8 * HOUR, as_of=FAR_FUTURE, tf=Timeframe.H1)
        h4 = reader.ohlcv([BTC], start=T0, end=T0 + 8 * HOUR, as_of=FAR_FUTURE, tf=Timeframe.H4)
        d1 = reader.ohlcv([BTC], start=T0, end=T0 + 8 * HOUR, as_of=FAR_FUTURE, tf=Timeframe.D1)
        assert h1.num_rows == 8  # 1h bars never leak into 4h reads and vice versa
        assert h4.num_rows == 2
        assert d1.num_rows == 0  # never resampled to 1d in this test

    def test_ohlcv_dataset_total_over_timeframes(self) -> None:
        assert {ohlcv_dataset(tf) for tf in Timeframe} == {
            Dataset.OHLCV,
            Dataset.OHLCV_4H,
            Dataset.OHLCV_1D,
        }
