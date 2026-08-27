"""Unit tests for alphaforge.data.schemas — lake schema contracts, quality-flag PIT,
partition paths."""

from __future__ import annotations

import pyarrow as pa
import pytest

from alphaforge.core.errors import SchemaError
from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import (
    FLAG_AVAILABILITY_LAG_BARS,
    FUNDING_SCHEMA,
    OHLCV_SCHEMA,
    SCHEMA_VERSION,
    UNIVERSE_SCHEMA,
    Dataset,
    QualityFlag,
    empty_table,
    flag_available_at,
    lake_relpath,
    schema_for,
    validate_table,
)

TS_OPEN = 1_704_153_600_000  # 2024-01-02T00:00:00Z
TS_INGESTED = 1_704_157_230_000  # 2024-01-02T01:00:30Z
H1 = Timeframe.H1.ms

ALL_DATASETS = [Dataset.OHLCV, Dataset.FUNDING, Dataset.UNIVERSE_MEMBERSHIP]


def ts_array(values: list[int | None]) -> pa.Array:
    return pa.array(values, type=pa.timestamp("ms", tz="UTC"))


def valid_ohlcv() -> pa.Table:
    return pa.table(
        {
            "instrument_id": pa.array(["BINANCE:PERP:BTCUSDT"], type=pa.string()),
            "ts_open": ts_array([TS_OPEN]),
            "open": pa.array([42_000.0], type=pa.float64()),
            "high": pa.array([42_500.0], type=pa.float64()),
            "low": pa.array([41_900.0], type=pa.float64()),
            "close": pa.array([42_400.0], type=pa.float64()),
            "volume": pa.array([123.45], type=pa.float64()),
            "quote_volume": pa.array([5_230_000.0], type=pa.float64()),
            "n_trades": pa.array([18_000], type=pa.int64()),
            "quality_flags": pa.array([0], type=pa.int32()),
            "ingested_at": ts_array([TS_INGESTED]),
        }
    )


def valid_funding() -> pa.Table:
    return pa.table(
        {
            "instrument_id": pa.array(["BINANCE:PERP:BTCUSDT"], type=pa.string()),
            "ts_funding": ts_array([TS_OPEN]),
            "rate": pa.array([0.0001], type=pa.float64()),
            "available_at": ts_array([TS_OPEN + 300_000]),
            "ingested_at": ts_array([TS_INGESTED]),
        }
    )


def valid_universe() -> pa.Table:
    return pa.table(
        {
            "instrument_id": pa.array(["BINANCE:PERP:BTCUSDT"], type=pa.string()),
            "effective_from": ts_array([TS_OPEN]),
            "effective_to": ts_array([None]),
            "rank": pa.array([1], type=pa.int32()),
            "reason": pa.array(["enter_top40"], type=pa.string()),
        }
    )


VALID_BUILDERS = {
    Dataset.OHLCV: valid_ohlcv,
    Dataset.FUNDING: valid_funding,
    Dataset.UNIVERSE_MEMBERSHIP: valid_universe,
}


def replace_column(tbl: pa.Table, name: str, array: pa.Array) -> pa.Table:
    idx = tbl.schema.get_field_index(name)
    return tbl.set_column(idx, pa.field(name, array.type), array)


# ---------------------------------------------------------------------------
# Schema constants and metadata
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_schema_version(self) -> None:
        assert SCHEMA_VERSION == 1

    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_metadata_carries_version_and_dataset(self, dataset: Dataset) -> None:
        meta = schema_for(dataset).metadata
        assert meta is not None
        assert meta[b"alphaforge.schema_version"] == b"1"
        assert meta[b"alphaforge.dataset"] == dataset.value.encode()

    def test_schema_for_returns_module_constants(self) -> None:
        assert schema_for(Dataset.OHLCV) is OHLCV_SCHEMA
        assert schema_for(Dataset.FUNDING) is FUNDING_SCHEMA
        assert schema_for(Dataset.UNIVERSE_MEMBERSHIP) is UNIVERSE_SCHEMA

    def test_all_timestamp_columns_are_ms_utc(self) -> None:
        for dataset in ALL_DATASETS:
            for field in schema_for(dataset):
                if pa.types.is_timestamp(field.type):
                    assert field.type == pa.timestamp("ms", tz="UTC"), field.name

    def test_ohlcv_does_not_store_available_at(self) -> None:
        # available_at is always derived as ts_open + timeframe; storing it invites drift.
        assert "available_at" not in OHLCV_SCHEMA.names

    def test_nullability_declarations(self) -> None:
        assert OHLCV_SCHEMA.field("quote_volume").nullable
        assert OHLCV_SCHEMA.field("n_trades").nullable
        assert not OHLCV_SCHEMA.field("quality_flags").nullable
        assert not FUNDING_SCHEMA.field("available_at").nullable
        assert UNIVERSE_SCHEMA.field("effective_to").nullable
        assert not UNIVERSE_SCHEMA.field("effective_from").nullable


# ---------------------------------------------------------------------------
# validate_table / empty_table
# ---------------------------------------------------------------------------


class TestValidateTable:
    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_empty_table_validates(self, dataset: Dataset) -> None:
        tbl = empty_table(dataset)
        assert tbl.num_rows == 0
        validate_table(tbl, dataset)  # must not raise

    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_valid_one_row_table_validates(self, dataset: Dataset) -> None:
        validate_table(VALID_BUILDERS[dataset](), dataset)  # must not raise

    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_missing_column_named_in_error(self, dataset: Dataset) -> None:
        tbl = VALID_BUILDERS[dataset]()
        dropped = tbl.schema.names[-1]
        with pytest.raises(SchemaError, match=rf"column '{dropped}': missing"):
            validate_table(tbl.drop_columns([dropped]), dataset)

    def test_extra_column_named_in_error(self) -> None:
        tbl = valid_ohlcv().append_column("bonus", pa.array([1.0], type=pa.float64()))
        with pytest.raises(SchemaError, match=r"column 'bonus': not in declared schema"):
            validate_table(tbl, Dataset.OHLCV)

    def test_wrong_type_named_in_error(self) -> None:
        tbl = replace_column(valid_ohlcv(), "open", pa.array([42_000.0], type=pa.float32()))
        with pytest.raises(SchemaError, match=r"column 'open': expected type double, got float"):
            validate_table(tbl, Dataset.OHLCV)

    def test_naive_timestamp_is_a_type_mismatch(self) -> None:
        naive = pa.array([TS_OPEN], type=pa.timestamp("ms"))  # no tz → must be rejected
        tbl = replace_column(valid_ohlcv(), "ts_open", naive)
        with pytest.raises(SchemaError, match=r"column 'ts_open': expected type"):
            validate_table(tbl, Dataset.OHLCV)

    def test_nulls_in_non_nullable_column_named_in_error(self) -> None:
        tbl = replace_column(valid_ohlcv(), "close", pa.array([None], type=pa.float64()))
        with pytest.raises(SchemaError, match=r"column 'close': 1 null value\(s\) in non-nullable"):
            validate_table(tbl, Dataset.OHLCV)

    def test_nulls_allowed_in_nullable_columns(self) -> None:
        tbl = valid_ohlcv()
        tbl = replace_column(tbl, "quote_volume", pa.array([None], type=pa.float64()))
        tbl = replace_column(tbl, "n_trades", pa.array([None], type=pa.int64()))
        validate_table(tbl, Dataset.OHLCV)  # must not raise

    def test_universe_open_interval_null_effective_to_allowed(self) -> None:
        validate_table(valid_universe(), Dataset.UNIVERSE_MEMBERSHIP)  # must not raise

    def test_funding_null_available_at_rejected(self) -> None:
        tbl = replace_column(valid_funding(), "available_at", ts_array([None]))
        with pytest.raises(SchemaError, match=r"column 'available_at': 1 null value\(s\)"):
            validate_table(tbl, Dataset.FUNDING)

    def test_all_problems_reported_in_one_error(self) -> None:
        tbl = valid_ohlcv().drop_columns(["volume"])
        tbl = tbl.append_column("bonus", pa.array([1], type=pa.int64()))
        tbl = replace_column(tbl, "high", pa.array(["oops"], type=pa.string()))
        tbl = replace_column(tbl, "close", pa.array([None], type=pa.float64()))
        with pytest.raises(SchemaError) as exc_info:
            validate_table(tbl, Dataset.OHLCV)
        msg = str(exc_info.value)
        assert "4 problem(s)" in msg
        assert "column 'volume': missing" in msg
        assert "column 'bonus': not in declared schema" in msg
        assert "column 'high': expected type double, got string" in msg
        assert "column 'close': 1 null value(s)" in msg

    def test_duplicate_column_names_rejected(self) -> None:
        base = valid_ohlcv()
        arrays = [*base.columns, pa.array([43_000.0], type=pa.float64())]
        names = [*base.schema.names, "close"]
        tbl = pa.Table.from_arrays(arrays, names=names)
        with pytest.raises(SchemaError, match=r"column 'close': appears 2 times"):
            validate_table(tbl, Dataset.OHLCV)

    def test_column_order_is_not_significant(self) -> None:
        tbl = valid_ohlcv()
        validate_table(tbl.select(list(reversed(tbl.schema.names))), Dataset.OHLCV)

    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_empty_table_carries_metadata(self, dataset: Dataset) -> None:
        meta = empty_table(dataset).schema.metadata
        assert meta is not None
        assert meta[b"alphaforge.schema_version"] == b"1"


# ---------------------------------------------------------------------------
# QualityFlag bitmask and PIT availability lags (leakage finding 5)
# ---------------------------------------------------------------------------


class TestQualityFlag:
    def test_bits_are_independent_powers_of_two(self) -> None:
        members = list(QualityFlag)  # canonical single-bit members; NONE excluded
        values = [int(m) for m in members]
        assert all(v > 0 and (v & (v - 1)) == 0 for v in values)  # each a power of two
        assert len(set(values)) == len(values)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                assert a & b == QualityFlag.NONE

    def test_exact_bit_values(self) -> None:
        assert QualityFlag.NONE == 0
        assert QualityFlag.GAP_FILLED_NONE == 1
        assert QualityFlag.OUTLIER_RETURN == 2
        assert QualityFlag.VOLUME_ANOMALY == 4
        assert QualityFlag.BAD_PRINT_SUSPECT == 8
        assert QualityFlag.EXCHANGE_DOWNTIME == 16

    def test_composite_round_trips(self) -> None:
        combo = QualityFlag.OUTLIER_RETURN | QualityFlag.EXCHANGE_DOWNTIME
        assert QualityFlag.OUTLIER_RETURN in combo
        assert QualityFlag.EXCHANGE_DOWNTIME in combo
        assert QualityFlag.BAD_PRINT_SUSPECT not in combo
        assert int(combo) == 18

    def test_lag_map_covers_every_flag(self) -> None:
        # Every canonical bit must declare its availability lag; adding a new bit
        # without a lag entry must fail this test.
        assert set(FLAG_AVAILABILITY_LAG_BARS) == set(QualityFlag)
        assert all(lag >= 0 for lag in FLAG_AVAILABILITY_LAG_BARS.values())

    def test_declared_lags(self) -> None:
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.OUTLIER_RETURN] == 0
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.VOLUME_ANOMALY] == 0
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.BAD_PRINT_SUSPECT] == 2
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.EXCHANGE_DOWNTIME] == 1
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.GAP_FILLED_NONE] == 1

    def test_flag_available_at_lag_zero_is_bar_close(self) -> None:
        ts = flag_available_at(TS_OPEN, Timeframe.H1, QualityFlag.OUTLIER_RETURN)
        assert ts == TS_OPEN + H1

    def test_flag_available_at_none_is_bar_close(self) -> None:
        assert flag_available_at(TS_OPEN, Timeframe.H1, QualityFlag.NONE) == TS_OPEN + H1

    def test_flag_available_at_bad_print_is_three_bars_after_open(self) -> None:
        # lag 2 beyond the bar's own close: ts_open + (1 + 2) * delta
        ts = flag_available_at(TS_OPEN, Timeframe.H1, QualityFlag.BAD_PRINT_SUSPECT)
        assert ts == TS_OPEN + 3 * H1

    def test_flag_available_at_composite_uses_slowest_bit(self) -> None:
        combo = QualityFlag.OUTLIER_RETURN | QualityFlag.BAD_PRINT_SUSPECT
        assert flag_available_at(TS_OPEN, Timeframe.H1, combo) == TS_OPEN + 3 * H1

    @pytest.mark.parametrize("tf", list(Timeframe))
    def test_flag_available_at_scales_with_timeframe(self, tf: Timeframe) -> None:
        ts = flag_available_at(TS_OPEN, tf, QualityFlag.EXCHANGE_DOWNTIME)
        assert ts == TS_OPEN + 2 * tf.ms


# ---------------------------------------------------------------------------
# lake_relpath
# ---------------------------------------------------------------------------


class TestLakeRelpath:
    def test_ohlcv_golden(self) -> None:
        path = lake_relpath(Dataset.OHLCV, "BINANCE:PERP:BTCUSDT", 2024)
        assert str(path) == "ohlcv/instrument_id=BINANCE:PERP:BTCUSDT/year=2024/data.parquet"

    def test_funding_golden(self) -> None:
        path = lake_relpath(Dataset.FUNDING, "BINANCE:PERP:ETHUSDT", 2023)
        assert str(path) == "funding/instrument_id=BINANCE:PERP:ETHUSDT/year=2023/data.parquet"

    def test_universe_golden(self) -> None:
        path = lake_relpath(Dataset.UNIVERSE_MEMBERSHIP, "BINANCE:PERP:BTCUSDT", 2025)
        assert (
            str(path)
            == "universe_membership/instrument_id=BINANCE:PERP:BTCUSDT/year=2025/data.parquet"
        )

    def test_returns_pure_posix_path(self) -> None:
        path = lake_relpath(Dataset.OHLCV, "BINANCE:PERP:BTCUSDT", 2024)
        assert not path.is_absolute()
        assert path.name == "data.parquet"

    def test_empty_instrument_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            lake_relpath(Dataset.OHLCV, "", 2024)

    @pytest.mark.parametrize("bad_id", ["BINANCE/PERP/BTCUSDT", "a=b", "x\x00y"])
    def test_metacharacters_in_instrument_id_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValueError, match="forbidden character"):
            lake_relpath(Dataset.OHLCV, bad_id, 2024)

    @pytest.mark.parametrize("bad_year", [0, 999, 10_000, -2024])
    def test_out_of_range_year_rejected(self, bad_year: int) -> None:
        with pytest.raises(ValueError, match="4-digit year"):
            lake_relpath(Dataset.OHLCV, "BINANCE:PERP:BTCUSDT", bad_year)
