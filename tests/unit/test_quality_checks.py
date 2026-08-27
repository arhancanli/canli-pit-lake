"""Unit tests for alphaforge.data.quality.checks.

Covers: gap-run detection + GAP_ADJACENT annotation, trailing-only outlier returns,
the three-condition bad-print detector (and why its bit lags — leakage finding 5),
stale-close runs, cross-sectional exchange-downtime classification, and the
apply_flags write-back path: flags-never-alter-prices (byte-equal price columns),
idempotence, and PIT visibility of the written bits.

All lakes are synthetic and live in ``tmp_path`` — never the real ./data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphaforge.config.settings import QualityCfg
from alphaforge.core.time import Timeframe
from alphaforge.data.quality.checks import (
    BadPrintCheck,
    DowntimeCheck,
    Finding,
    GapCheck,
    OutlierReturnCheck,
    StaleCloseCheck,
    apply_flags,
)
from alphaforge.data.schemas import Dataset, QualityFlag
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

BTC = "BINANCE:PERP:BTCUSDT"
PANEL = [f"BINANCE:PERP:C{i}USDT" for i in range(5)]

HOUR = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z
FAR_FUTURE = 1_900_000_000_000  # 2030-03-17; "now" for historical fixtures
H1 = Timeframe.H1

PRICE_COLUMNS = ("ts_open", "open", "high", "low", "close", "volume", "quote_volume", "n_trades")


# --------------------------------------------------------------------------- builders


def base_close(i: int) -> float:
    """Gently alternating closes: ~±0.1% returns, so sigma-hat stabilizes near 1e-3."""
    return 100.0 + 0.1 * (i % 2)


def make_bars(
    instrument_id: str,
    n: int,
    *,
    closes: Sequence[float] | None = None,
    volumes: Sequence[float] | None = None,
    skip: frozenset[int] = frozenset(),
    t0: int = T0,
) -> pa.Table:
    """Synthetic 1h OHLCV table (sorted, OHLC-sane); bar ``i`` opens at t0 + i*HOUR.

    ``closes``/``volumes`` index by bar number ``i`` (not by surviving row), so a
    planted anomaly keeps its timestamp when ``skip`` removes bars before it.
    """
    rows = [i for i in range(n) if i not in skip]
    close = [base_close(i) if closes is None else closes[i] for i in rows]
    volume = [10.0 if volumes is None else volumes[i] for i in rows]
    ts = [t0 + i * HOUR for i in rows]
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * len(rows), type=pa.string()),
            "ts_open": pa.array(ts, type=pa.timestamp("ms", tz="UTC")),
            "open": pa.array(close, type=pa.float64()),
            "high": pa.array([c + 1.0 for c in close], type=pa.float64()),
            "low": pa.array([c - 1.0 for c in close], type=pa.float64()),
            "close": pa.array(close, type=pa.float64()),
            "volume": pa.array(volume, type=pa.float64()),
            "quote_volume": pa.array([c * v for c, v in zip(close, volume, strict=True)]),
            "n_trades": pa.array([42] * len(rows), type=pa.int64()),
            "quality_flags": pa.array([0] * len(rows), type=pa.int32()),
            "ingested_at": pa.array(
                [t + HOUR + 30_000 for t in ts], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


def spiked_closes(
    n: int, spike_at: int, *, factor: float = 1.25, persist: bool = False
) -> list[float]:
    """Base closes with a planted price spike at bar ``spike_at``; if ``persist`` the
    price stays at the new level (real move), otherwise it snaps back (bad print)."""
    closes = [base_close(i) for i in range(n)]
    if persist:
        for i in range(spike_at, n):
            closes[i] *= factor
    else:
        closes[spike_at] *= factor
    return closes


def row_of(tbl: pa.Table, ts: int) -> int:
    ts_col = tbl.column("ts_open").to_numpy(zero_copy_only=False).astype("datetime64[ms]")
    idx = np.flatnonzero(ts_col.astype(np.int64) == ts)
    assert len(idx) == 1
    return int(idx[0])


def kinds(findings: list[Finding]) -> list[str]:
    return [f["kind"] for f in findings]


# -------------------------------------------------------------------------- GapCheck


class TestGapCheck:
    def test_no_gaps_no_findings_no_flags(self) -> None:
        result = GapCheck().run(make_bars(BTC, 30), instrument_id=BTC, tf=H1)
        assert result.findings == []
        assert not result.flags.any()

    def test_gap_run_finding_span_and_warn_severity(self) -> None:
        tbl = make_bars(BTC, 40, skip=frozenset({30, 31, 32}))
        result = GapCheck().run(tbl, instrument_id=BTC, tf=H1)
        assert len(result.findings) == 1
        f = result.findings[0]
        assert f["kind"] == "gap"
        assert f["instrument_id"] == BTC
        assert f["ts_start"] == T0 + 30 * HOUR
        assert f["ts_end"] == T0 + 33 * HOUR
        assert f["severity"] == "warn"
        assert f["detail"] == {"n_bars": 3}

    def test_gap_longer_than_three_bars_is_error(self) -> None:
        tbl = make_bars(BTC, 40, skip=frozenset({10, 11, 12, 13}))
        (finding,) = GapCheck().run(tbl, instrument_id=BTC, tf=H1).findings
        assert finding["severity"] == "error"
        assert finding["detail"] == {"n_bars": 4}

    def test_adjacent_surviving_bars_flagged_on_both_sides(self) -> None:
        tbl = make_bars(BTC, 40, skip=frozenset({30, 31, 32}))
        flags = GapCheck().run(tbl, instrument_id=BTC, tf=H1).flags
        adjacent = np.flatnonzero(flags & int(QualityFlag.GAP_ADJACENT))
        assert adjacent.tolist() == [row_of(tbl, T0 + 29 * HOUR), row_of(tbl, T0 + 33 * HOUR)]
        # gaps are never filled: row count is unchanged and no synthetic bar appears
        assert tbl.num_rows == 37

    def test_multiple_gap_runs_reported_separately(self) -> None:
        tbl = make_bars(BTC, 40, skip=frozenset({5, 6, 20}))
        findings = GapCheck().run(tbl, instrument_id=BTC, tf=H1).findings
        assert [(f["ts_start"], f["detail"]["n_bars"]) for f in findings] == [
            (T0 + 5 * HOUR, 2),
            (T0 + 20 * HOUR, 1),
        ]


# ----------------------------------------------------------------- OutlierReturnCheck


class TestOutlierReturnCheck:
    def test_spike_flagged_at_right_bar_only(self) -> None:
        spike = 50
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike, persist=True))
        result = OutlierReturnCheck().run(tbl, instrument_id=BTC, tf=H1)
        hits = np.flatnonzero(result.flags & int(QualityFlag.OUTLIER_RETURN))
        assert hits.tolist() == [spike]
        (finding,) = result.findings
        assert finding["kind"] == "outlier_return"
        assert finding["ts_start"] == T0 + spike * HOUR
        assert finding["ts_end"] == T0 + (spike + 1) * HOUR
        assert finding["detail"]["z"] > 8.0

    def test_clean_series_unflagged(self) -> None:
        result = OutlierReturnCheck().run(make_bars(BTC, 80), instrument_id=BTC, tf=H1)
        assert result.findings == []
        assert not result.flags.any()

    def test_warmup_guard_blocks_flags_before_min_history(self) -> None:
        spike = 10  # < min_history_bars prior returns: sigma-hat is not yet trusted
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike, persist=True))
        result = OutlierReturnCheck(min_history_bars=24).run(tbl, instrument_id=BTC, tf=H1)
        assert not result.flags.any()

    def test_flag_at_t_does_not_depend_on_future_bars(self) -> None:
        # The sigma-hat at t uses PRIOR bars only: truncating everything after the
        # spike must produce the identical decision at the spike bar.
        spike = 50
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike))
        check = OutlierReturnCheck()
        full = check.run(tbl, instrument_id=BTC, tf=H1).flags
        prefix = check.run(tbl.slice(0, spike + 1), instrument_id=BTC, tf=H1).flags
        assert full[: spike + 1].tolist() == prefix.tolist()
        assert prefix[spike] & int(QualityFlag.OUTLIER_RETURN)


# --------------------------------------------------------------------- BadPrintCheck


class TestBadPrintCheck:
    def test_low_volume_reverted_spike_sets_bit_on_right_bar_only(self) -> None:
        spike = 50
        volumes = [10.0] * 80
        volumes[spike] = 1.0  # < 0.25 * trailing median (10.0)
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike), volumes=volumes)
        result = BadPrintCheck().run(tbl, instrument_id=BTC, tf=H1)
        hits = np.flatnonzero(result.flags & int(QualityFlag.BAD_PRINT_SUSPECT))
        assert hits.tolist() == [spike]
        (finding,) = result.findings
        assert finding["kind"] == "bad_print"
        assert finding["ts_start"] == T0 + spike * HOUR
        assert finding["detail"]["volume"] == 1.0

    def test_spike_with_real_volume_is_not_a_bad_print(self) -> None:
        # 12-sigma move WITH volume: outlier yes, bad print no (condition b fails).
        spike = 50
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike))
        bad = BadPrintCheck().run(tbl, instrument_id=BTC, tf=H1)
        outlier = OutlierReturnCheck().run(tbl, instrument_id=BTC, tf=H1)
        assert not bad.flags.any()
        assert outlier.flags[spike] & int(QualityFlag.OUTLIER_RETURN)

    def test_persistent_move_is_not_a_bad_print(self) -> None:
        # Price does not snap back (condition c fails) even on suspicious volume.
        spike = 50
        volumes = [10.0] * 80
        volumes[spike] = 1.0
        tbl = make_bars(BTC, 80, closes=spiked_closes(80, spike, persist=True), volumes=volumes)
        assert not BadPrintCheck().run(tbl, instrument_id=BTC, tf=H1).flags.any()

    def test_last_bar_never_flagged_needs_next_close(self) -> None:
        # Condition c reads close_{t+1} — exactly why the bit lags 2 bars (finding 5).
        n = 60
        spike = n - 1
        volumes = [10.0] * n
        volumes[spike] = 1.0
        tbl = make_bars(BTC, n, closes=spiked_closes(n, spike), volumes=volumes)
        assert not BadPrintCheck().run(tbl, instrument_id=BTC, tf=H1).flags.any()


# ------------------------------------------------------------------- StaleCloseCheck


class TestStaleCloseCheck:
    @staticmethod
    def _frozen(n: int, start: int, length: int) -> list[float]:
        closes = [base_close(i) for i in range(n)]
        for i in range(start, start + length):
            closes[i] = 100.05
        return closes

    def test_frozen_close_with_volume_is_reported(self) -> None:
        tbl = make_bars(BTC, 80, closes=self._frozen(80, 40, 13))
        result = StaleCloseCheck(run_bars=12).run(tbl, instrument_id=BTC, tf=H1)
        (finding,) = result.findings
        assert finding["kind"] == "stale_close"
        assert finding["severity"] == "warn"
        assert finding["ts_start"] == T0 + 40 * HOUR
        assert finding["ts_end"] == T0 + 53 * HOUR
        assert finding["detail"]["n_bars"] == 13
        assert not result.flags.any()  # finding only — no flag bit exists for this

    def test_zero_volume_frozen_close_is_not_a_stuck_feed(self) -> None:
        volumes = [10.0] * 80
        for i in range(40, 53):
            volumes[i] = 0.0
        tbl = make_bars(BTC, 80, closes=self._frozen(80, 40, 13), volumes=volumes)
        assert StaleCloseCheck(run_bars=12).run(tbl, instrument_id=BTC, tf=H1).findings == []

    def test_run_below_threshold_ignored(self) -> None:
        tbl = make_bars(BTC, 80, closes=self._frozen(80, 40, 11))
        assert StaleCloseCheck(run_bars=12).run(tbl, instrument_id=BTC, tf=H1).findings == []


# --------------------------------------------------------------------- DowntimeCheck


class TestDowntimeCheck:
    WINDOW = frozenset({20, 21, 22})

    def _panel(self, n_missing: int) -> dict[str, pa.Table]:
        return {
            iid: make_bars(iid, 48, skip=self.WINDOW if i < n_missing else frozenset())
            for i, iid in enumerate(PANEL)
        }

    def test_simultaneous_gap_on_four_of_five_is_downtime(self) -> None:
        result = DowntimeCheck(downtime_frac=0.8).run_panel(self._panel(4), tf=H1)
        (finding,) = result.findings
        assert finding["kind"] == "exchange_downtime"
        assert finding["instrument_id"] == "*"
        assert finding["severity"] == "info"
        assert finding["ts_start"] == T0 + 20 * HOUR
        assert finding["ts_end"] == T0 + 23 * HOUR
        assert finding["detail"]["n_bars"] == 3
        assert finding["detail"]["n_affected"] == 4
        assert finding["detail"]["affected"] == sorted(PANEL[:4])

    def test_first_bar_after_window_flagged_for_affected_only(self) -> None:
        panel = self._panel(4)
        result = DowntimeCheck(downtime_frac=0.8).run_panel(panel, tf=H1)
        for iid in PANEL[:4]:
            flags = result.flags_by_instrument[iid]
            hits = np.flatnonzero(flags & int(QualityFlag.EXCHANGE_DOWNTIME))
            assert hits.tolist() == [row_of(panel[iid], T0 + 23 * HOUR)]
        assert not result.flags_by_instrument[PANEL[4]].any()

    def test_below_threshold_fraction_is_not_downtime(self) -> None:
        result = DowntimeCheck(downtime_frac=0.8).run_panel(self._panel(3), tf=H1)
        assert result.findings == []
        assert all(not f.any() for f in result.flags_by_instrument.values())

    def test_single_instrument_gap_is_never_downtime(self) -> None:
        panel = {BTC: make_bars(BTC, 48, skip=self.WINDOW)}
        result = DowntimeCheck(downtime_frac=0.8).run_panel(panel, tf=H1)
        assert result.findings == []
        assert not result.flags_by_instrument[BTC].any()


# ------------------------------------------------------------------------ apply_flags


@pytest.fixture
def lake(tmp_path: Path) -> tuple[LakePaths, LakeWriter, PITDataReader]:
    paths = LakePaths(tmp_path / "lake")
    return paths, LakeWriter(paths), PITDataReader(paths)


def seed_bad_print_lake(writer: LakeWriter, *, spike: int = 50, n: int = 80) -> pa.Table:
    """BTC history with a planted gap run (30-32) and a bad print at ``spike``."""
    volumes = [10.0] * n
    volumes[spike] = 1.0
    tbl = make_bars(
        BTC,
        n,
        closes=spiked_closes(n, spike),
        volumes=volumes,
        skip=frozenset({30, 31, 32}),
    )
    writer.write(Dataset.OHLCV, tbl, tf=H1, now=FAR_FUTURE)
    return tbl


class TestApplyFlags:
    def test_flags_written_to_lake(self, lake: tuple[LakePaths, LakeWriter, PITDataReader]) -> None:
        paths, writer, reader = lake
        seed_bad_print_lake(writer)
        stats = apply_flags(reader, writer, [BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)
        assert stats.rows_reflagged > 0

        with pq.ParquetFile(paths.partition_path(Dataset.OHLCV, BTC, 2024)) as leaf:  # type: ignore[no-untyped-call]
            stored = leaf.read()
        flags = np.asarray(
            stored.column("quality_flags").to_numpy(zero_copy_only=False), dtype=np.int32
        )
        spike_row = row_of(stored, T0 + 50 * HOUR)
        assert flags[spike_row] & int(QualityFlag.OUTLIER_RETURN)
        assert flags[spike_row] & int(QualityFlag.BAD_PRINT_SUSPECT)
        assert flags[row_of(stored, T0 + 29 * HOUR)] & int(QualityFlag.GAP_ADJACENT)
        assert flags[row_of(stored, T0 + 33 * HOUR)] & int(QualityFlag.GAP_ADJACENT)
        assert set(kinds(stats.findings)) == {"gap", "outlier_return", "bad_print"}

    def test_flags_never_alter_prices_byte_equal(
        self, lake: tuple[LakePaths, LakeWriter, PITDataReader]
    ) -> None:
        paths, writer, reader = lake
        seed_bad_print_lake(writer)
        leaf_path = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        with pq.ParquetFile(leaf_path) as leaf:  # type: ignore[no-untyped-call]
            before = leaf.read()
        apply_flags(reader, writer, [BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)
        with pq.ParquetFile(leaf_path) as leaf:  # type: ignore[no-untyped-call]
            after = leaf.read()
        assert after.num_rows == before.num_rows  # no synthetic gap-fill rows
        for name in PRICE_COLUMNS:
            assert after.column(name).equals(before.column(name)), name

    def test_second_run_is_idempotent(
        self, lake: tuple[LakePaths, LakeWriter, PITDataReader]
    ) -> None:
        paths, writer, reader = lake
        seed_bad_print_lake(writer)
        first = apply_flags(reader, writer, [BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)
        leaf_path = paths.partition_path(Dataset.OHLCV, BTC, 2024)
        bytes_after_first = leaf_path.read_bytes()
        second = apply_flags(reader, writer, [BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE + HOUR)
        assert first.rows_reflagged > 0
        assert second.rows_reflagged == 0
        assert leaf_path.read_bytes() == bytes_after_first  # nothing rewritten at all
        assert kinds(second.findings) == kinds(first.findings)

    def test_pit_reader_masks_bad_print_until_available(
        self, lake: tuple[LakePaths, LakeWriter, PITDataReader]
    ) -> None:
        # Leakage finding 5 end-to-end: the stored BAD_PRINT_SUSPECT bit (computed
        # from close_{t+1}) must be invisible at the bar's own close, while
        # OUTLIER_RETURN (lag 0) is visible immediately.
        _, writer, reader = lake
        spike_open = T0 + 50 * HOUR
        seed_bad_print_lake(writer)
        apply_flags(reader, writer, [BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)

        def flags_at(as_of: int) -> int:
            tbl = reader.ohlcv([BTC], start=spike_open, end=spike_open + HOUR, as_of=as_of)
            return int(tbl.column("quality_flags").to_pylist()[0])

        at_close = flags_at(spike_open + HOUR)
        assert at_close & int(QualityFlag.OUTLIER_RETURN)
        assert not at_close & int(QualityFlag.BAD_PRINT_SUSPECT)
        fully_visible = flags_at(spike_open + 3 * HOUR)
        assert fully_visible & int(QualityFlag.BAD_PRINT_SUSPECT)

    def test_downtime_flags_round_trip_through_lake(
        self, lake: tuple[LakePaths, LakeWriter, PITDataReader]
    ) -> None:
        paths, writer, reader = lake
        for i, iid in enumerate(PANEL):
            skip = frozenset({20, 21, 22}) if i < 4 else frozenset()
            writer.write(Dataset.OHLCV, make_bars(iid, 48, skip=skip), tf=H1, now=FAR_FUTURE)
        stats = apply_flags(reader, writer, PANEL, tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)
        downtime = [f for f in stats.findings if f["kind"] == "exchange_downtime"]
        assert len(downtime) == 1
        for iid in PANEL[:4]:
            with pq.ParquetFile(  # type: ignore[no-untyped-call]
                paths.partition_path(Dataset.OHLCV, iid, 2024)
            ) as leaf:
                stored = leaf.read()
            flags = np.asarray(
                stored.column("quality_flags").to_numpy(zero_copy_only=False), dtype=np.int32
            )
            assert flags[row_of(stored, T0 + 23 * HOUR)] & int(QualityFlag.EXCHANGE_DOWNTIME)

    def test_stats_accounting(self, lake: tuple[LakePaths, LakeWriter, PITDataReader]) -> None:
        _, writer, reader = lake
        seeded = seed_bad_print_lake(writer)
        stats = apply_flags(reader, writer, [BTC, BTC], tf=H1, cfg=QualityCfg(), now=FAR_FUTURE)
        assert stats.instruments == 1  # duplicates scanned once
        assert stats.bars_scanned == seeded.num_rows
        assert stats.bar_counts == {BTC: seeded.num_rows}
        # 2 gap-adjacent rows + 1 spike row (outlier|bad_print share the row)
        assert stats.rows_reflagged == 3
