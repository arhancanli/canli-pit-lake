"""Unit tests for alphaforge.data.store.lake (LakePaths) and .writer (LakeWriter).

Covers: partition path/glob/tmp conventions, schema validation ordering, the
partial-bar write guard, idempotent dedupe (latest ``ingested_at`` wins, ties to
incoming), multi-year/multi-instrument partition splits, within-file sort order,
zstd compression, and the tmp-write + atomic-replace crash-safety protocol.
"""

from __future__ import annotations

import dataclasses
import glob as globlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.core.errors import SchemaError
from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import Dataset, QualityFlag, empty_table, lake_relpath, schema_for
from alphaforge.data.store.lake import TMP_PREFIX, LakePaths
from alphaforge.data.store.writer import NATURAL_KEY_COLUMN, LakeWriter, WriteStats

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"

HOUR = 3_600_000
T_2024 = 1_704_153_600_000  # 2024-01-02T00:00:00Z
T_2023_LAST = 1_704_063_600_000  # 2023-12-31T23:00:00Z (last 1h bar of 2023)
T_2024_FIRST = 1_704_067_200_000  # 2024-01-01T00:00:00Z
FAR_FUTURE = 1_900_000_000_000  # 2030-03-17; "now" for historical fixtures


# --------------------------------------------------------------------------- helpers


def bar(
    instrument_id: str,
    ts_open: int,
    *,
    close: float = 100.0,
    flags: int = 0,
    ingested_at: int | None = None,
) -> dict[str, object]:
    """One OHLCV row; ``ingested_at`` defaults to 30s after the bar close."""
    return {
        "instrument_id": instrument_id,
        "ts_open": ts_open,
        "open": 99.0,
        "high": 101.0,
        "low": 98.0,
        "close": close,
        "volume": 10.0,
        "quote_volume": 1_000.0,
        "n_trades": 42,
        "quality_flags": flags,
        "ingested_at": ts_open + HOUR + 30_000 if ingested_at is None else ingested_at,
    }


def ohlcv_table(*rows: dict[str, object]) -> pa.Table:
    def ts(name: str) -> pa.Array:
        return pa.array([r[name] for r in rows], type=pa.timestamp("ms", tz="UTC"))

    def f64(name: str) -> pa.Array:
        return pa.array([r[name] for r in rows], type=pa.float64())

    return pa.table(
        {
            "instrument_id": pa.array([r["instrument_id"] for r in rows], type=pa.string()),
            "ts_open": ts("ts_open"),
            "open": f64("open"),
            "high": f64("high"),
            "low": f64("low"),
            "close": f64("close"),
            "volume": f64("volume"),
            "quote_volume": f64("quote_volume"),
            "n_trades": pa.array([r["n_trades"] for r in rows], type=pa.int64()),
            "quality_flags": pa.array([r["quality_flags"] for r in rows], type=pa.int32()),
            "ingested_at": ts("ingested_at"),
        }
    )


def funding_row(
    instrument_id: str,
    ts_funding: int,
    *,
    rate: float = 1e-4,
    available_at: int | None = None,
    ingested_at: int | None = None,
) -> dict[str, object]:
    avail = ts_funding + 300_000 if available_at is None else available_at
    return {
        "instrument_id": instrument_id,
        "ts_funding": ts_funding,
        "rate": rate,
        "available_at": avail,
        "ingested_at": avail + 1_000 if ingested_at is None else ingested_at,
    }


def funding_table(*rows: dict[str, object]) -> pa.Table:
    def ts(name: str) -> pa.Array:
        return pa.array([r[name] for r in rows], type=pa.timestamp("ms", tz="UTC"))

    return pa.table(
        {
            "instrument_id": pa.array([r["instrument_id"] for r in rows], type=pa.string()),
            "ts_funding": ts("ts_funding"),
            "rate": pa.array([r["rate"] for r in rows], type=pa.float64()),
            "available_at": ts("available_at"),
            "ingested_at": ts("ingested_at"),
        }
    )


def membership_row(
    instrument_id: str,
    effective_from: int,
    *,
    effective_to: int | None = None,
    rank: int = 1,
    reason: str = "enter_top40",
) -> dict[str, object]:
    return {
        "instrument_id": instrument_id,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "rank": rank,
        "reason": reason,
    }


def universe_table(*rows: dict[str, object]) -> pa.Table:
    def ts(name: str) -> pa.Array:
        return pa.array([r[name] for r in rows], type=pa.timestamp("ms", tz="UTC"))

    return pa.table(
        {
            "instrument_id": pa.array([r["instrument_id"] for r in rows], type=pa.string()),
            "effective_from": ts("effective_from"),
            "effective_to": ts("effective_to"),
            "rank": pa.array([r["rank"] for r in rows], type=pa.int32()),
            "reason": pa.array([r["reason"] for r in rows], type=pa.string()),
        }
    )


@pytest.fixture
def paths(tmp_path: Path) -> LakePaths:
    return LakePaths(tmp_path / "lake")


@pytest.fixture
def writer(paths: LakePaths) -> LakeWriter:
    return LakeWriter(paths)


def read_leaf(path: Path) -> pa.Table:
    """Read one leaf file directly. ``pq.read_table`` would run dataset/hive
    discovery over the ``key=value`` path segments and clash with the in-file
    ``instrument_id`` column (pyarrow >= 24 behavior)."""
    with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
        out: pa.Table = leaf.read()
    return out


def stored_closes(paths: LakePaths, instrument_id: str, year: int) -> list[float]:
    tbl = read_leaf(paths.partition_path(Dataset.OHLCV, instrument_id, year))
    return [v.as_py() for v in tbl.column("close")]


# --------------------------------------------------------------------------- LakePaths


class TestLakePaths:
    def test_partition_path_is_root_plus_lake_relpath(self, paths: LakePaths) -> None:
        expected = paths.root / lake_relpath(Dataset.OHLCV, BTC, 2024)
        assert paths.partition_path(Dataset.OHLCV, BTC, 2024) == expected
        assert paths.partition_path(Dataset.OHLCV, BTC, 2024).name == "data.parquet"

    def test_partition_path_validates_instrument_id(self, paths: LakePaths) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            paths.partition_path(Dataset.OHLCV, "BAD/ID", 2024)

    def test_glob_matches_leaf_files_only(self, paths: LakePaths, writer: LakeWriter) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), now=FAR_FUTURE)
        target = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        stray_tmp = paths.tmp_path(target)
        stray_tmp.write_bytes(b"in-flight garbage")
        matched = {Path(p) for p in globlib.glob(paths.glob(Dataset.OHLCV))}
        assert matched == {target}  # data.parquet matched, tmp file invisible

    def test_years_for_sorted_and_junk_ignored(self, paths: LakePaths, writer: LakeWriter) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2024), bar(BTC, T_2023_LAST)),
            now=FAR_FUTURE,
        )
        instrument_dir = paths.partition_path(Dataset.OHLCV, BTC, 2024).parent.parent
        (instrument_dir / "year=20x4").mkdir()  # malformed year label
        (instrument_dir / "notyear=2025").mkdir()  # wrong key
        (instrument_dir / "year=2025").mkdir()  # year dir without data.parquet
        assert paths.years_for(Dataset.OHLCV, BTC) == [2023, 2024]

    def test_years_for_empty_lake(self, paths: LakePaths) -> None:
        assert paths.years_for(Dataset.OHLCV, BTC) == []

    def test_partition_paths_inclusive_year_bounds(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(
                bar(BTC, T_2023_LAST), bar(BTC, T_2024), bar(BTC, T_2024 + 366 * 24 * HOUR)
            ),
            now=FAR_FUTURE,
        )
        assert paths.years_for(Dataset.OHLCV, BTC) == [2023, 2024, 2025]
        got = paths.partition_paths(Dataset.OHLCV, BTC, year_min=2024, year_max=2024)
        assert got == [paths.partition_path(Dataset.OHLCV, BTC, 2024)]
        assert len(paths.partition_paths(Dataset.OHLCV, BTC)) == 3

    def test_instrument_ids_sorted_and_junk_ignored(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(ETH, T_2024), bar(BTC, T_2024)), now=FAR_FUTURE)
        dataset_dir = paths.partition_path(Dataset.OHLCV, BTC, 2024).parent.parent.parent
        (dataset_dir / "notakey=XXX").mkdir()  # wrong hive key
        (dataset_dir / "stray.txt").write_text("junk")  # not a directory
        assert paths.instrument_ids(Dataset.OHLCV) == [BTC, ETH]

    def test_instrument_ids_empty_lake_and_missing_dataset(self, paths: LakePaths) -> None:
        assert paths.instrument_ids(Dataset.OHLCV) == []  # lake root absent entirely

    def test_tmp_path_same_dir_prefix_and_unique(self, paths: LakePaths) -> None:
        target = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        tmp1, tmp2 = paths.tmp_path(target), paths.tmp_path(target)
        assert tmp1.parent == target.parent  # same dir => same filesystem => atomic replace
        assert tmp1.name.startswith(TMP_PREFIX)
        assert tmp1 != tmp2
        assert LakePaths.is_tmp(tmp1) and LakePaths.is_tmp(tmp2)
        assert not LakePaths.is_tmp(target)

    def test_tmp_files_missing_root(self, paths: LakePaths) -> None:
        assert paths.tmp_files() == []


# --------------------------------------------------------------------------- LakeWriter


class TestWriteValidation:
    def test_invalid_table_raises_and_writes_nothing(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        broken = ohlcv_table(bar(BTC, T_2024)).drop_columns(["close"])
        with pytest.raises(SchemaError, match="close"):
            writer.write(Dataset.OHLCV, broken, now=FAR_FUTURE)
        assert not paths.root.exists() or list(paths.root.rglob("*.parquet")) == []

    def test_empty_table_writes_nothing(self, paths: LakePaths, writer: LakeWriter) -> None:
        stats = writer.write(Dataset.OHLCV, empty_table(Dataset.OHLCV), now=FAR_FUTURE)
        assert stats == WriteStats(0, 0, 0, 0, 0)
        assert paths.years_for(Dataset.OHLCV, BTC) == []

    def test_write_stats_frozen(self) -> None:
        stats = WriteStats(1, 1, 0, 1, 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.rows_in = 2  # type: ignore[misc]

    def test_natural_keys_cover_all_datasets(self) -> None:
        assert set(NATURAL_KEY_COLUMN) == set(Dataset)


class TestPartialBarGuard:
    def test_in_progress_bar_dropped(self, paths: LakePaths, writer: LakeWriter) -> None:
        tbl = ohlcv_table(bar(BTC, T_2024), bar(BTC, T_2024 + HOUR))
        now = T_2024 + HOUR  # first bar closes exactly now; second is mid-flight
        stats = writer.write(Dataset.OHLCV, tbl, now=now)
        assert stats.unclosed_dropped == 1
        assert stats.rows_written == 1
        assert stored_closes(paths, BTC, 2024) == [100.0]

    def test_bar_closing_exactly_at_now_is_kept(self, writer: LakeWriter) -> None:
        stats = writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), now=T_2024 + HOUR)
        assert stats.unclosed_dropped == 0
        assert stats.rows_written == 1

    def test_bar_close_one_ms_in_future_is_dropped(self, writer: LakeWriter) -> None:
        stats = writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), now=T_2024 + HOUR - 1)
        assert stats == WriteStats(
            rows_in=1, rows_written=0, rows_deduped=0, partitions=0, unclosed_dropped=1
        )

    def test_guard_uses_tf_parameter(self, writer: LakeWriter) -> None:
        # At now = ts_open + 1h the bar is closed as H1 but still open as H4.
        now = T_2024 + HOUR
        as_h1 = writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), tf=Timeframe.H1, now=now)
        assert as_h1.unclosed_dropped == 0
        as_h4 = writer.write(Dataset.OHLCV, ohlcv_table(bar(ETH, T_2024)), tf=Timeframe.H4, now=now)
        assert as_h4.unclosed_dropped == 1

    def test_guard_defaults_to_wall_clock(self, writer: LakeWriter) -> None:
        # A 2024 bar is long closed relative to the real wall clock.
        stats = writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)))
        assert stats.unclosed_dropped == 0
        assert stats.rows_written == 1

    def test_guard_not_applied_to_funding(self, writer: LakeWriter) -> None:
        row = funding_row(BTC, T_2024, available_at=T_2024 + 300_000)
        stats = writer.write(Dataset.FUNDING, funding_table(row), now=T_2024 - HOUR)
        assert stats.unclosed_dropped == 0
        assert stats.rows_written == 1


class TestDedupe:
    def test_double_write_is_idempotent(self, paths: LakePaths, writer: LakeWriter) -> None:
        tbl = ohlcv_table(bar(BTC, T_2024), bar(BTC, T_2024 + HOUR))
        first = writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)
        target = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        snapshot = read_leaf(target)
        second = writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)
        assert first == WriteStats(2, 2, 0, 1, 0)
        assert second == WriteStats(2, 2, 2, 1, 0)  # both incoming rows deduped away
        assert read_leaf(target).equals(snapshot)  # logically identical content

    def test_later_ingested_at_wins(self, paths: LakePaths, writer: LakeWriter) -> None:
        original = bar(BTC, T_2024, close=100.0, ingested_at=T_2024 + HOUR + 30_000)
        corrected = bar(BTC, T_2024, close=105.0, ingested_at=T_2024 + 2 * HOUR)
        writer.write(Dataset.OHLCV, ohlcv_table(original), now=FAR_FUTURE)
        stats = writer.write(Dataset.OHLCV, ohlcv_table(corrected), now=FAR_FUTURE)
        assert stats.rows_deduped == 1
        assert stored_closes(paths, BTC, 2024) == [105.0]

    def test_older_ingested_at_loses(self, paths: LakePaths, writer: LakeWriter) -> None:
        stored = bar(BTC, T_2024, close=105.0, ingested_at=T_2024 + 2 * HOUR)
        stale = bar(BTC, T_2024, close=100.0, ingested_at=T_2024 + HOUR + 30_000)
        writer.write(Dataset.OHLCV, ohlcv_table(stored), now=FAR_FUTURE)
        writer.write(Dataset.OHLCV, ohlcv_table(stale), now=FAR_FUTURE)
        assert stored_closes(paths, BTC, 2024) == [105.0]  # max ingested_at kept

    def test_ingested_at_tie_incoming_wins(self, paths: LakePaths, writer: LakeWriter) -> None:
        t_ing = T_2024 + HOUR + 30_000
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2024, close=100.0, ingested_at=t_ing)),
            now=FAR_FUTURE,
        )
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2024, close=107.0, ingested_at=t_ing)),
            now=FAR_FUTURE,
        )
        assert stored_closes(paths, BTC, 2024) == [107.0]

    def test_duplicate_keys_within_batch_last_wins(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        t_ing = T_2024 + HOUR + 30_000
        tbl = ohlcv_table(
            bar(BTC, T_2024, close=100.0, ingested_at=t_ing),
            bar(BTC, T_2024, close=101.0, ingested_at=t_ing),
        )
        stats = writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)
        assert stats == WriteStats(2, 1, 1, 1, 0)
        assert stored_closes(paths, BTC, 2024) == [101.0]

    def test_funding_dedupes_on_ts_funding(self, paths: LakePaths, writer: LakeWriter) -> None:
        first = funding_row(BTC, T_2024, rate=1e-4, ingested_at=T_2024 + 400_000)
        revised = funding_row(BTC, T_2024, rate=2e-4, ingested_at=T_2024 + 500_000)
        writer.write(Dataset.FUNDING, funding_table(first))
        stats = writer.write(Dataset.FUNDING, funding_table(revised))
        assert stats.rows_deduped == 1
        tbl = read_leaf(paths.partition_path(Dataset.FUNDING, BTC, 2024))
        assert tbl.num_rows == 1
        assert tbl.column("rate")[0].as_py() == 2e-4

    def test_universe_without_ingested_at_incoming_wins(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        writer.write(
            Dataset.UNIVERSE_MEMBERSHIP,
            universe_table(membership_row(BTC, T_2024, rank=3, reason="enter_top40")),
        )
        writer.write(
            Dataset.UNIVERSE_MEMBERSHIP,
            universe_table(membership_row(BTC, T_2024, rank=5, reason="rerank", effective_to=None)),
        )
        tbl = read_leaf(paths.partition_path(Dataset.UNIVERSE_MEMBERSHIP, BTC, 2024))
        assert tbl.num_rows == 1
        assert tbl.column("rank")[0].as_py() == 5
        assert tbl.column("reason")[0].as_py() == "rerank"


class TestPartitioning:
    def test_multi_year_split(self, paths: LakePaths, writer: LakeWriter) -> None:
        tbl = ohlcv_table(bar(BTC, T_2023_LAST), bar(BTC, T_2024_FIRST))
        stats = writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)
        assert stats.partitions == 2
        assert paths.years_for(Dataset.OHLCV, BTC) == [2023, 2024]
        assert stored_closes(paths, BTC, 2023) == [100.0]
        assert stored_closes(paths, BTC, 2024) == [100.0]

    def test_multi_instrument_split(self, paths: LakePaths, writer: LakeWriter) -> None:
        stats = writer.write(
            Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024), bar(ETH, T_2024)), now=FAR_FUTURE
        )
        assert stats.partitions == 2
        assert paths.years_for(Dataset.OHLCV, BTC) == [2024]
        assert paths.years_for(Dataset.OHLCV, ETH) == [2024]

    def test_rows_sorted_by_key_within_file(self, paths: LakePaths, writer: LakeWriter) -> None:
        tbl = ohlcv_table(
            bar(BTC, T_2024 + 2 * HOUR, close=3.0),
            bar(BTC, T_2024, close=1.0),
            bar(BTC, T_2024 + HOUR, close=2.0),
        )
        writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)
        assert stored_closes(paths, BTC, 2024) == [1.0, 2.0, 3.0]

    def test_written_file_conforms_to_declared_schema(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2024, flags=int(QualityFlag.OUTLIER_RETURN))),
            now=FAR_FUTURE,
        )
        tbl = read_leaf(paths.partition_path(Dataset.OHLCV, BTC, 2024))
        assert tbl.schema.equals(schema_for(Dataset.OHLCV), check_metadata=True)


class TestAtomicityAndCompression:
    def test_zstd_compression_used(self, paths: LakePaths, writer: LakeWriter) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), now=FAR_FUTURE)
        pf = pq.ParquetFile(  # type: ignore[no-untyped-call]
            paths.partition_path(Dataset.OHLCV, BTC, 2024)
        )
        meta = pf.metadata
        compressions = {
            meta.row_group(rg).column(col).compression
            for rg in range(meta.num_row_groups)
            for col in range(meta.row_group(rg).num_columns)
        }
        assert compressions == {"ZSTD"}

    def test_failed_replace_leaves_target_intact_and_tmp_cleanable(
        self, paths: LakePaths, writer: LakeWriter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024, close=100.0)), now=FAR_FUTURE)
        target = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        snapshot = read_leaf(target)

        def crash(src: object, dst: object) -> None:
            raise OSError("simulated crash between tmp write and replace")

        monkeypatch.setattr(os, "replace", crash)
        with pytest.raises(OSError, match="simulated crash"):
            writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024, close=999.0)), now=FAR_FUTURE)
        monkeypatch.undo()

        assert read_leaf(target).equals(snapshot)  # reader never saw a torn/partial file
        leftovers = paths.tmp_files()
        assert len(leftovers) == 1 and LakePaths.is_tmp(leftovers[0])
        assert writer.clean_tmp() == 1
        assert paths.tmp_files() == []
        assert read_leaf(target).equals(snapshot)  # clean_tmp never touches real data

    def test_clean_tmp_on_clean_lake(self, writer: LakeWriter) -> None:
        assert writer.clean_tmp() == 0

    def test_all_rows_dropped_writes_no_partition(
        self, paths: LakePaths, writer: LakeWriter
    ) -> None:
        stats = writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T_2024)), now=T_2024)
        assert stats == WriteStats(1, 0, 0, 0, 1)
        assert paths.years_for(Dataset.OHLCV, BTC) == []
