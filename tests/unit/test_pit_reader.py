"""Unit tests for alphaforge.data.store.reader (PITDataReader).

Covers: the PIT boundary (bar invisible at close - 1 ms, visible at close exactly),
per-bit quality-flag availability masking (leakage finding 5), funding
``available_at`` gating (finding 18), half-open windows, multi-year/multi-instrument
reads, the freshness probe ``latest_bar_open``, the ``gaps`` ops tool, schema-stable
empty results, and tmp-file invisibility.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass
from alphaforge.data.schemas import (
    FLAG_AVAILABILITY_LAG_BARS,
    Dataset,
    QualityFlag,
    empty_table,
    schema_for,
)
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"
AAPL = "XNAS:CASH:AAPLUSD"  # equity id for the corporate-actions PIT tests

H1 = Timeframe.H1
HOUR = 3_600_000
T0 = 1_704_153_600_000  # 2024-01-02T00:00:00Z (1h-aligned)
T_2023_LAST = 1_704_063_600_000  # 2023-12-31T23:00:00Z
T_2024_FIRST = 1_704_067_200_000  # 2024-01-01T00:00:00Z
FAR_FUTURE = 1_900_000_000_000  # "now"/"as_of" far past every fixture bar


# --------------------------------------------------------------------------- helpers


def bar(
    instrument_id: str,
    ts_open: int,
    *,
    close: float = 100.0,
    flags: int = 0,
    ingested_at: int | None = None,
) -> dict[str, object]:
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
) -> dict[str, object]:
    avail = ts_funding + 300_000 if available_at is None else available_at
    return {
        "instrument_id": instrument_id,
        "ts_funding": ts_funding,
        "rate": rate,
        "available_at": avail,
        "ingested_at": avail + 1_000,
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


def corp_action_row(
    instrument_id: str,
    ex_date: int,
    *,
    action_type: str = "split",
    ratio: float = 0.5,
    cash_amount: float | None = None,
    available_at: int | None = None,
) -> dict[str, object]:
    # available_at = declaration/knowable date + publication lag (default ex_date - 7d,
    # the realistic weeks-ahead split announcement); the PIT predicate gates on it.
    avail = ex_date - 7 * 24 * HOUR if available_at is None else available_at
    return {
        "instrument_id": instrument_id,
        "action_type": action_type,
        "ex_date": ex_date,
        "available_at": avail,
        "ratio": ratio,
        "cash_amount": cash_amount,
        "ingested_at": avail + 1_000,
    }


def corp_actions_table(*rows: dict[str, object]) -> pa.Table:
    def ts(name: str) -> pa.Array:
        return pa.array([r[name] for r in rows], type=pa.timestamp("ms", tz="UTC"))

    return pa.table(
        {
            "instrument_id": pa.array([r["instrument_id"] for r in rows], type=pa.string()),
            "action_type": pa.array([r["action_type"] for r in rows], type=pa.string()),
            "ex_date": ts("ex_date"),
            "available_at": ts("available_at"),
            "ratio": pa.array([r["ratio"] for r in rows], type=pa.float64()),
            "cash_amount": pa.array([r["cash_amount"] for r in rows], type=pa.float64()),
            "ingested_at": ts("ingested_at"),
        }
    )


def read_leaf(path: Path) -> pa.Table:
    """Read one leaf file directly. ``pq.read_table`` would run dataset/hive
    discovery over the ``key=value`` path segments and clash with the in-file
    ``instrument_id`` column (pyarrow >= 24 behavior)."""
    with pq.ParquetFile(path) as leaf:  # type: ignore[no-untyped-call]
        out: pa.Table = leaf.read()
    return out


def opens_of(tbl: pa.Table) -> list[int]:
    """ts_open column as epoch-ms ints (the API-boundary representation)."""
    return [int(v.cast(pa.int64()).as_py()) for v in tbl.column("ts_open")]


def flags_of(tbl: pa.Table) -> list[int]:
    return [v.as_py() for v in tbl.column("quality_flags")]


@pytest.fixture
def paths(tmp_path: Path) -> LakePaths:
    return LakePaths(tmp_path / "lake")


@pytest.fixture
def writer(paths: LakePaths) -> LakeWriter:
    return LakeWriter(paths)


@pytest.fixture
def reader(paths: LakePaths) -> PITDataReader:
    return PITDataReader(paths)


# --------------------------------------------------------------------------- empty lake


class TestEmptyLake:
    def test_ohlcv_schema_stable(self, reader: PITDataReader) -> None:
        out = reader.ohlcv([BTC], start=T0, end=T0 + 24 * HOUR, as_of=FAR_FUTURE)
        assert out.num_rows == 0
        assert out.schema.equals(schema_for(Dataset.OHLCV), check_metadata=True)
        assert out.equals(empty_table(Dataset.OHLCV))

    def test_funding_schema_stable(self, reader: PITDataReader) -> None:
        out = reader.funding([BTC], start=T0, end=T0 + 24 * HOUR, as_of=FAR_FUTURE)
        assert out.num_rows == 0
        assert out.equals(empty_table(Dataset.FUNDING))

    def test_corporate_actions_schema_stable(self, reader: PITDataReader) -> None:
        out = reader.corporate_actions([AAPL], start=T0, end=T0 + 24 * HOUR, as_of=FAR_FUTURE)
        assert out.num_rows == 0
        assert out.equals(empty_table(Dataset.CORPORATE_ACTIONS))

    def test_latest_bar_open_none(self, reader: PITDataReader) -> None:
        assert reader.latest_bar_open(BTC, as_of=FAR_FUTURE) is None

    def test_gaps_full_grid(self, reader: PITDataReader) -> None:
        assert reader.gaps(BTC, start=T0, end=T0 + 3 * HOUR) == [T0, T0 + HOUR, T0 + 2 * HOUR]


# --------------------------------------------------------------------------- PIT rule


class TestPITBoundary:
    @pytest.fixture
    def one_bar_lake(self, writer: LakeWriter) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0)), now=FAR_FUTURE)

    def test_invisible_one_ms_before_close(self, one_bar_lake: None, reader: PITDataReader) -> None:
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=T0 + HOUR - 1)
        assert out.num_rows == 0

    def test_visible_exactly_at_close(self, one_bar_lake: None, reader: PITDataReader) -> None:
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=T0 + HOUR)
        assert opens_of(out) == [T0]

    def test_pit_delta_comes_from_tf_param(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # Same ts_open stored in both the native and the derived dataset: at the 1h
        # close the 1h bar is available while the 4h bar (close = ts_open + 4h) is not.
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0)), now=FAR_FUTURE)
        writer.write(Dataset.OHLCV_4H, ohlcv_table(bar(BTC, T0)), tf=Timeframe.H4, now=FAR_FUTURE)
        as_of = T0 + HOUR
        end = T0 + 4 * HOUR
        assert reader.ohlcv([BTC], start=T0, end=end, as_of=as_of, tf=Timeframe.H1).num_rows == 1
        assert reader.ohlcv([BTC], start=T0, end=end, as_of=as_of, tf=Timeframe.H4).num_rows == 0
        assert (
            reader.ohlcv([BTC], start=T0, end=end, as_of=T0 + 4 * HOUR, tf=Timeframe.H4).num_rows
            == 1
        )

    def test_end_beyond_as_of_is_allowed(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T0), bar(BTC, T0 + HOUR), bar(BTC, T0 + 2 * HOUR)),
            now=FAR_FUTURE,
        )
        # Window reaches a week past as_of; the PIT filter alone governs visibility.
        out = reader.ohlcv([BTC], start=T0, end=T0 + 168 * HOUR, as_of=T0 + 2 * HOUR)
        assert opens_of(out) == [T0, T0 + HOUR]

    def test_window_half_open_on_ts_open(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T0), bar(BTC, T0 + HOUR), bar(BTC, T0 + 2 * HOUR)),
            now=FAR_FUTURE,
        )
        out = reader.ohlcv([BTC], start=T0, end=T0 + 2 * HOUR, as_of=FAR_FUTURE)
        assert opens_of(out) == [T0, T0 + HOUR]  # end exclusive, start inclusive

    def test_empty_interval_and_empty_ids(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0)), now=FAR_FUTURE)
        assert reader.ohlcv([BTC], start=T0, end=T0, as_of=FAR_FUTURE).num_rows == 0
        assert reader.ohlcv([], start=T0, end=T0 + HOUR, as_of=FAR_FUTURE).num_rows == 0

    def test_result_schema_exact(self, one_bar_lake: None, reader: PITDataReader) -> None:
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.schema.equals(schema_for(Dataset.OHLCV), check_metadata=True)


class TestInstrumentAndYearScoping:
    def test_only_requested_instruments_returned(
        self, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0), bar(ETH, T0)), now=FAR_FUTURE)
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert set(out.column("instrument_id").to_pylist()) == {BTC}

    def test_multi_instrument_sorted(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(ETH, T0), bar(BTC, T0 + HOUR), bar(BTC, T0), bar(ETH, T0 + HOUR)),
            now=FAR_FUTURE,
        )
        out = reader.ohlcv([ETH, BTC], start=T0, end=T0 + 2 * HOUR, as_of=FAR_FUTURE)
        assert out.column("instrument_id").to_pylist() == [BTC, BTC, ETH, ETH]
        assert opens_of(out) == [T0, T0 + HOUR, T0, T0 + HOUR]

    def test_read_spans_year_partition_files(
        self, paths: LakePaths, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2023_LAST), bar(BTC, T_2024_FIRST)),
            now=FAR_FUTURE,
        )
        assert paths.years_for(Dataset.OHLCV, BTC) == [2023, 2024]  # really two files
        out = reader.ohlcv([BTC], start=T_2023_LAST, end=T_2024_FIRST + HOUR, as_of=FAR_FUTURE)
        assert opens_of(out) == [T_2023_LAST, T_2024_FIRST]

    def test_reader_ignores_tmp_files(
        self, paths: LakePaths, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0)), now=FAR_FUTURE)
        target = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        paths.tmp_path(target).write_bytes(b"NOT A PARQUET FILE")  # crashed-writer leftover
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert opens_of(out) == [T0]


# --------------------------------------------------------------- quality-flag PIT masking


class TestQualityFlagMasking:
    """Leakage finding 5: bit b is visible iff ts_open + (1 + L_b)·Δ <= as_of."""

    SUSPECT = int(QualityFlag.OUTLIER_RETURN | QualityFlag.BAD_PRINT_SUSPECT)  # lags 0 and 2

    @pytest.fixture
    def flagged_lake(self, writer: LakeWriter) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0, flags=self.SUSPECT)), now=FAR_FUTURE)

    def read_flags(self, reader: PITDataReader, as_of: int) -> int:
        out = reader.ohlcv([BTC], start=T0, end=T0 + HOUR, as_of=as_of)
        assert out.num_rows == 1
        return flags_of(out)[0]

    def test_bad_print_invisible_at_close(self, flagged_lake: None, reader: PITDataReader) -> None:
        # At the bar's own close only the lag-0 OUTLIER_RETURN bit may show.
        assert FLAG_AVAILABILITY_LAG_BARS[QualityFlag.BAD_PRINT_SUSPECT] == 2
        assert self.read_flags(reader, T0 + HOUR) == int(QualityFlag.OUTLIER_RETURN)

    def test_bad_print_still_masked_one_ms_before_lag_elapses(
        self, flagged_lake: None, reader: PITDataReader
    ) -> None:
        assert self.read_flags(reader, T0 + 3 * HOUR - 1) == int(QualityFlag.OUTLIER_RETURN)

    def test_bad_print_visible_two_bars_after_close(
        self, flagged_lake: None, reader: PITDataReader
    ) -> None:
        # available at ts_open + (1 + 2)·Δ exactly.
        assert self.read_flags(reader, T0 + 3 * HOUR) == self.SUSPECT

    def test_exchange_downtime_lag_one_bar(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T0, flags=int(QualityFlag.EXCHANGE_DOWNTIME))),
            now=FAR_FUTURE,
        )
        assert self.read_flags(reader, T0 + HOUR) == 0  # masked at close
        assert self.read_flags(reader, T0 + 2 * HOUR) == int(QualityFlag.EXCHANGE_DOWNTIME)

    def test_masking_uses_tf_delta(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # On a derived 4h bar the BAD_PRINT bit needs ts_open + 3·4h; at ts_open + 4h
        # (the bar's own close) it is masked while lag-0 bits are visible.
        writer.write(
            Dataset.OHLCV_4H,
            ohlcv_table(bar(BTC, T0, flags=self.SUSPECT)),
            tf=Timeframe.H4,
            now=FAR_FUTURE,
        )
        out = reader.ohlcv([BTC], start=T0, end=T0 + 4 * HOUR, as_of=T0 + 4 * HOUR, tf=Timeframe.H4)
        assert flags_of(out) == [int(QualityFlag.OUTLIER_RETURN)]

    def test_masking_never_mutates_the_lake(
        self, flagged_lake: None, paths: LakePaths, reader: PITDataReader
    ) -> None:
        assert self.read_flags(reader, T0 + HOUR) == int(QualityFlag.OUTLIER_RETURN)
        stored = read_leaf(paths.partition_path(Dataset.OHLCV, BTC, 2024))
        assert stored.column("quality_flags").to_pylist() == [self.SUSPECT]
        assert self.read_flags(reader, FAR_FUTURE) == self.SUSPECT  # full flags once available

    def test_clean_bar_stays_clean(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0, flags=0)), now=FAR_FUTURE)
        assert self.read_flags(reader, T0 + HOUR) == 0


# --------------------------------------------------------------------------- funding


class TestFundingPIT:
    def test_available_at_boundary(self, writer: LakeWriter, reader: PITDataReader) -> None:
        avail = T0 + 300_000  # settlement + 5 min publication lag
        writer.write(Dataset.FUNDING, funding_table(funding_row(BTC, T0, available_at=avail)))
        window = {"start": T0 - HOUR, "end": T0 + HOUR}
        assert reader.funding([BTC], as_of=avail - 1, **window).num_rows == 0
        out = reader.funding([BTC], as_of=avail, **window)
        assert out.num_rows == 1
        assert out.column("rate")[0].as_py() == 1e-4

    def test_window_half_open_on_ts_funding(
        self, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        eight_h = 8 * HOUR
        writer.write(
            Dataset.FUNDING,
            funding_table(
                funding_row(BTC, T0),
                funding_row(BTC, T0 + eight_h),
                funding_row(BTC, T0 + 2 * eight_h),
            ),
        )
        out = reader.funding([BTC], start=T0, end=T0 + 2 * eight_h, as_of=FAR_FUTURE)
        got = [int(v.cast(pa.int64()).as_py()) for v in out.column("ts_funding")]
        assert got == [T0, T0 + eight_h]

    def test_only_requested_instruments(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.FUNDING, funding_table(funding_row(BTC, T0), funding_row(ETH, T0)))
        out = reader.funding([ETH], start=T0 - HOUR, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.column("instrument_id").to_pylist() == [ETH]

    def test_result_schema_exact(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.FUNDING, funding_table(funding_row(BTC, T0)))
        out = reader.funding([BTC], start=T0 - HOUR, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.schema.equals(schema_for(Dataset.FUNDING), check_metadata=True)


# ------------------------------------------------------------------ corporate actions


class TestCorporateActionsPIT:
    """corporate_actions mirrors funding() EXACTLY: range on ex_date, mask on
    available_at (NEVER ex_date) — the equity leakage-finding-18 analogue."""

    def test_available_at_boundary(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # A split declared (knowable) at `avail` is invisible to a decision one ms before
        # and visible exactly at `avail` — gated on available_at, not ex_date.
        ex = T0 + 30 * 24 * HOUR
        avail = T0  # declared a month before ex
        writer.write(
            Dataset.CORPORATE_ACTIONS,
            corp_actions_table(corp_action_row(AAPL, ex, ratio=0.5, available_at=avail)),
        )
        window = {"start": ex - HOUR, "end": ex + HOUR}
        assert reader.corporate_actions([AAPL], as_of=avail - 1, **window).num_rows == 0
        out = reader.corporate_actions([AAPL], as_of=avail, **window)
        assert out.num_rows == 1
        assert out.column("ratio")[0].as_py() == 0.5
        assert out.column("action_type")[0].as_py() == "split"

    def test_window_half_open_on_ex_date(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # The within-range key is ex_date: [start, end) half-open on ex_date.
        day = 24 * HOUR
        writer.write(
            Dataset.CORPORATE_ACTIONS,
            corp_actions_table(
                corp_action_row(AAPL, T0, available_at=T0 - day),
                corp_action_row(AAPL, T0 + day, available_at=T0 - day),
                corp_action_row(AAPL, T0 + 2 * day, available_at=T0 - day),
            ),
        )
        out = reader.corporate_actions([AAPL], start=T0, end=T0 + 2 * day, as_of=FAR_FUTURE)
        got = [int(v.cast(pa.int64()).as_py()) for v in out.column("ex_date")]
        assert got == [T0, T0 + day]

    def test_only_requested_instruments(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.CORPORATE_ACTIONS,
            corp_actions_table(corp_action_row(AAPL, T0), corp_action_row(ETH, T0)),
        )
        out = reader.corporate_actions([ETH], start=T0 - HOUR, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.column("instrument_id").to_pylist() == [ETH]

    def test_dividend_carries_cash_amount(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.CORPORATE_ACTIONS,
            corp_actions_table(
                corp_action_row(AAPL, T0, action_type="dividend", ratio=1.0, cash_amount=0.24)
            ),
        )
        out = reader.corporate_actions([AAPL], start=T0 - HOUR, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.column("cash_amount")[0].as_py() == pytest.approx(0.24)

    def test_result_schema_exact(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.CORPORATE_ACTIONS, corp_actions_table(corp_action_row(AAPL, T0)))
        out = reader.corporate_actions([AAPL], start=T0 - HOUR, end=T0 + HOUR, as_of=FAR_FUTURE)
        assert out.schema.equals(schema_for(Dataset.CORPORATE_ACTIONS), check_metadata=True)

    def test_empty_interval(self, reader: PITDataReader) -> None:
        out = reader.corporate_actions([AAPL], start=T0, end=T0, as_of=FAR_FUTURE)
        assert out.equals(empty_table(Dataset.CORPORATE_ACTIONS))


# ----------------------------------------------------------------- freshness + gaps


class TestLatestBarOpen:
    @pytest.fixture
    def three_bar_lake(self, writer: LakeWriter) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T0), bar(BTC, T0 + HOUR), bar(BTC, T0 + 2 * HOUR)),
            now=FAR_FUTURE,
        )

    def test_returns_freshest_fully_closed_bar(
        self, three_bar_lake: None, reader: PITDataReader
    ) -> None:
        assert reader.latest_bar_open(BTC, as_of=T0 + 2 * HOUR) == T0 + HOUR
        assert reader.latest_bar_open(BTC, as_of=FAR_FUTURE) == T0 + 2 * HOUR

    def test_close_boundary_exact(self, three_bar_lake: None, reader: PITDataReader) -> None:
        assert reader.latest_bar_open(BTC, as_of=T0 + HOUR) == T0
        assert reader.latest_bar_open(BTC, as_of=T0 + HOUR - 1) is None

    def test_unknown_instrument_is_none(self, three_bar_lake: None, reader: PITDataReader) -> None:
        assert reader.latest_bar_open(ETH, as_of=FAR_FUTURE) is None

    def test_respects_tf_delta(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # A derived 4h bar opening at T0 closes only at T0 + 4h.
        writer.write(Dataset.OHLCV_4H, ohlcv_table(bar(BTC, T0)), tf=Timeframe.H4, now=FAR_FUTURE)
        assert reader.latest_bar_open(BTC, as_of=T0 + 4 * HOUR - 1, tf=Timeframe.H4) is None
        assert reader.latest_bar_open(BTC, as_of=T0 + 4 * HOUR, tf=Timeframe.H4) == T0


class TestGaps:
    def test_detects_missing_bars(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T0), bar(BTC, T0 + 3 * HOUR)),  # hole at +1h and +2h
            now=FAR_FUTURE,
        )
        assert reader.gaps(BTC, start=T0, end=T0 + 4 * HOUR) == [T0 + HOUR, T0 + 2 * HOUR]

    def test_complete_range_has_no_gaps(self, writer: LakeWriter, reader: PITDataReader) -> None:
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0), bar(BTC, T0 + HOUR)), now=FAR_FUTURE)
        assert reader.gaps(BTC, start=T0, end=T0 + 2 * HOUR) == []

    def test_no_pit_filter_applied(self, writer: LakeWriter, reader: PITDataReader) -> None:
        # gaps() is an ops tool over physical lake content: a bar present in the
        # lake is "present" regardless of any availability consideration.
        writer.write(Dataset.OHLCV, ohlcv_table(bar(BTC, T0)), now=FAR_FUTURE)
        assert reader.gaps(BTC, start=T0, end=T0 + HOUR) == []

    def test_spans_year_files_and_unaligned_bounds(
        self, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        writer.write(
            Dataset.OHLCV,
            ohlcv_table(bar(BTC, T_2023_LAST), bar(BTC, T_2024_FIRST)),
            now=FAR_FUTURE,
        )
        # Unaligned start is ceiled onto the grid by expected_bar_opens.
        got = reader.gaps(BTC, start=T_2023_LAST - HOUR + 1, end=T_2024_FIRST + 2 * HOUR)
        assert got == [T_2024_FIRST + HOUR]

    def test_empty_interval(self, reader: PITDataReader) -> None:
        assert reader.gaps(BTC, start=T0, end=T0) == []

    def test_equity_asset_class_skips_weekends(
        self, writer: LakeWriter, reader: PITDataReader
    ) -> None:
        # SPINE_ARC §3.8 / audit G8: with asset_class=EQUITY the expected grid is the
        # XNYS session grid, so a weekend is correctly ABSENT (not reported as a gap).
        # 2024-01-05 is a Friday; the next session is Monday 2024-01-08 (Sat/Sun skipped).
        day = 24 * HOUR
        fri = 1_704_412_800_000  # 2024-01-05T00:00:00Z (Friday session open)
        mon = fri + 3 * day  # 2024-01-08T00:00:00Z (next session, Sat+Sun hopped)
        writer.write(
            Dataset.OHLCV_1D,
            ohlcv_table(bar(AAPL, fri), bar(AAPL, mon)),
            tf=Timeframe.D1,
            now=FAR_FUTURE,
        )
        # 24/7 grid would report Sat+Sun as gaps; the XNYS grid reports none.
        crypto_gaps = reader.gaps(AAPL, start=fri, end=mon + day, tf=Timeframe.D1)
        assert crypto_gaps == [fri + day, fri + 2 * day]  # Sat, Sun (the 24/7 default)
        equity_gaps = reader.gaps(
            AAPL, start=fri, end=mon + day, tf=Timeframe.D1, asset_class=AssetClass.EQUITY
        )
        assert equity_gaps == []  # both sessions present; weekend correctly absent
