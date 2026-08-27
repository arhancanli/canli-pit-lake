"""Unit tests for alphaforge.data.universe.builder (UniverseBuilder).

Synthetic tmp_path lake with fake perps of controlled daily quote volume. Covers:
exact top-N selection, entry/exit hysteresis band semantics, the quote-volume
fallback, min-listing-age and min-days eligibility gates, mid-month delisting
closing intervals at ``delisted_ts`` (the survivorship gate in miniature), PIT
immunity to post-rebalance volume, rebuild determinism with full-replace semantics,
deterministic rank tie-breaks, and config guards. Never touches ./data — every lake
lives under tmp_path.

Thresholds are scaled to the 8-instrument fixture (entry_rank=3, exit_rank=5),
mirroring the production 20/26 hysteresis exactly.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.config.settings import UniverseCfg
from alphaforge.core.errors import ConfigError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.schemas import Dataset
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter
from alphaforge.data.universe.builder import UniverseBuilder
from alphaforge.data.universe.store import UniverseStore

HOUR = 3_600_000
DAY = 86_400_000
T_JAN = 1_704_067_200_000  # 2024-01-01T00:00:00Z
T_FEB = 1_706_745_600_000  # 2024-02-01T00:00:00Z (rebalance T1)
T_MAR = 1_709_251_200_000  # 2024-03-01T00:00:00Z (rebalance T2)
FAR_FUTURE = 1_900_000_000_000  # 2030-03-17; "now" for historical fixtures
LISTED = T_FEB - 90 * DAY  # comfortably past the 30d min listing age

JAN_DAYS = [T_JAN + k * DAY for k in range(1, 31)]  # Jan 2 .. Jan 31 (T1 window)
FEB_DAYS = [T_FEB + k * DAY for k in range(29)]  # Feb 1 .. Feb 29 (bulk of T2 window)

CFG = UniverseCfg(size=3, rank_window_days=30, rebalance="monthly", entry_rank=3, exit_rank=5)

_TS = pa.timestamp("ms", tz="UTC")


def iid(base: str) -> str:
    return f"BINANCE:PERP:{base}USDT"


A, B, C, D, E, F, G, H = (iid(b) for b in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"))


def inst(base: str, *, listed: int = LISTED, delisted: int | None = None) -> Instrument:
    return Instrument(
        instrument_id=iid(base),
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base=base,
        quote="USDT",
        tick_size=0.01,
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


class Env:
    """One synthetic lake + SCD2 record + builder per test."""

    def __init__(self, tmp_path: Path, instruments: list[Instrument], **builder_kw: int) -> None:
        paths = LakePaths(tmp_path / "lake")
        self.writer = LakeWriter(paths)
        self.store = UniverseStore(paths)
        self.instruments = InstrumentStore(tmp_path / "ops.sqlite")
        for i in instruments:
            self.instruments.upsert(i, as_of=i.listed_ts)
        self.builder = UniverseBuilder(
            PITDataReader(paths), self.instruments, self.store, CFG, **builder_kw
        )

    def write_daily(
        self,
        instrument_id: str,
        day_starts: list[int],
        quote_volume: float | None,
        *,
        volume: float = 10.0,
        close: float = 100.0,
        ts_offset: int = 12 * HOUR,
    ) -> None:
        """One 1h bar per UTC day (at 12:00 by default) carrying the day's volume."""
        opens = [d + ts_offset for d in day_starts]
        n = len(opens)
        tbl = pa.table(
            {
                "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
                "ts_open": pa.array(opens, type=_TS),
                "open": pa.array([close] * n, type=pa.float64()),
                "high": pa.array([close] * n, type=pa.float64()),
                "low": pa.array([close] * n, type=pa.float64()),
                "close": pa.array([close] * n, type=pa.float64()),
                "volume": pa.array([volume] * n, type=pa.float64()),
                "quote_volume": pa.array([quote_volume] * n, type=pa.float64()),
                "n_trades": pa.array([1] * n, type=pa.int64()),
                "quality_flags": pa.array([0] * n, type=pa.int32()),
                "ingested_at": pa.array([o + HOUR + 30_000 for o in opens], type=_TS),
            }
        )
        self.writer.write(Dataset.OHLCV, tbl, now=FAR_FUTURE)

    def rows(self) -> list[tuple[str, int, int | None, int, str]]:
        tbl = self.store.read_intervals()
        return list(
            zip(
                tbl.column("instrument_id").to_pylist(),
                tbl.column("effective_from").cast(pa.int64()).to_pylist(),
                tbl.column("effective_to").cast(pa.int64()).to_pylist(),
                tbl.column("rank").to_pylist(),
                tbl.column("reason").to_pylist(),
                strict=True,
            )
        )


def standard_jan_volumes(env: Env) -> None:
    """8 instruments, Jan daily quote volumes 800..100 → ranks A=1 .. H=8 at T_FEB."""
    for instrument_id, qv in zip((A, B, C, D, E, F, G, H), range(800, 0, -100), strict=True):
        env.write_daily(instrument_id, JAN_DAYS, float(qv))


def all_eight() -> list[Instrument]:
    return [inst(b) for b in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")]


class TestSelection:
    def test_top_n_selection_exact(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        stats = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert (stats.rebalances, stats.intervals, stats.entries, stats.exits) == (1, 3, 3, 0)
        assert stats.open_members == 3
        assert env.builder.members_asof(T_FEB) == sorted([A, B, C])
        assert env.rows() == [
            (A, T_FEB, None, 1, "enter_top3"),
            (B, T_FEB, None, 2, "enter_top3"),
            (C, T_FEB, None, 3, "enter_top3"),
        ]

    def test_quote_volume_null_falls_back_to_volume_times_close(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        # BBB rewritten with null quote_volume: 7.5 * 100 = 750/day → still rank 2.
        env.write_daily(B, JAN_DAYS, None, volume=7.5, close=100.0)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert (B, T_FEB, None, 2, "enter_top3") in env.rows()

    def test_rank_tie_breaks_by_instrument_id(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        env.write_daily(D, JAN_DAYS, 600.0)  # DDD now ties CCC exactly at 600/day
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        members = env.builder.members_asof(T_FEB)
        assert C in members and D not in members  # CCC < DDD wins the tied rank 3
        assert (C, T_FEB, None, 3, "enter_top3") in env.rows()


class TestEligibility:
    def test_min_listing_age_excludes_fresh_listing(self, tmp_path: Path) -> None:
        fresh = inst("FRS", listed=T_FEB - 10 * DAY)  # 10d old < 30d min age
        env = Env(tmp_path, [*all_eight(), fresh])
        standard_jan_volumes(env)
        env.write_daily(fresh.instrument_id, JAN_DAYS, 10_000.0)  # top volume, still out
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert env.builder.members_asof(T_FEB) == sorted([A, B, C])

    def test_min_days_excludes_sparse_history(self, tmp_path: Path) -> None:
        sparse = inst("SPR")
        env = Env(tmp_path, [*all_eight(), sparse])
        standard_jan_volumes(env)
        # Huge volume but only 10 distinct days < min_days (20) → ineligible.
        env.write_daily(sparse.instrument_id, JAN_DAYS[-10:], 10_000.0)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert env.builder.members_asof(T_FEB) == sorted([A, B, C])


class TestHysteresis:
    def test_band_semantics_across_two_rebalances(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)  # T1 ranks: A1 B2 C3 → members {A, B, C}
        # Feb medians → T2 ranks: A1 F2 D3 C4 E5 B6 G7 H8.
        feb = {A: 800.0, F: 700.0, D: 600.0, C: 500.0, E: 400.0, B: 250.0, G: 200.0, H: 100.0}
        for instrument_id, qv in feb.items():
            env.write_daily(instrument_id, FEB_DAYS, qv)
        env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)

        # Member C at rank 4 (inside (3, 5] band) STAYS with its original entry row.
        assert (C, T_FEB, None, 3, "enter_top3") in env.rows()
        # Member B at rank 6 (> exit_rank 5) EXITS at T2.
        assert (B, T_FEB, T_MAR, 2, "exit_rank_6") in env.rows()
        # Non-member E at rank 5 (inside the band) does NOT enter.
        assert all(r[0] != E for r in env.rows())
        # D and F enter at T2 with their entry ranks.
        assert (D, T_MAR, None, 3, "enter_top3") in env.rows()
        assert (F, T_MAR, None, 2, "enter_top3") in env.rows()
        assert env.builder.members_asof(T_MAR) == sorted([A, C, D, F])
        # Boundary: B is a member up to (but excluding) the rebalance instant.
        assert B in env.builder.members_asof(T_MAR - 1)
        assert env.builder.current(T_MAR) == env.builder.members_asof(T_MAR)


class TestDelisting:
    def test_mid_month_delisting_closes_at_delisted_ts(self, tmp_path: Path) -> None:
        delist_ts = T_FEB + 10 * DAY
        instruments = [inst("AAA"), inst("BBB"), inst("CCC", delisted=delist_ts), inst("DDD")]
        env = Env(tmp_path, instruments)
        for instrument_id, qv in ((A, 800.0), (B, 700.0), (C, 600.0), (D, 500.0)):
            env.write_daily(instrument_id, JAN_DAYS, qv)
        for instrument_id, qv in ((A, 800.0), (B, 700.0), (D, 500.0)):
            env.write_daily(instrument_id, FEB_DAYS, qv)
        env.write_daily(C, FEB_DAYS[:9], 600.0)  # trades until its death on Feb 11

        env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)

        # THE survivorship gate in miniature: the dead symbol's interval exists,
        # closed at delisted_ts — not at month-end.
        assert (C, T_FEB, delist_ts, 3, "delisted") in env.rows()
        assert C in env.builder.members_asof(delist_ts - 1)
        assert C not in env.builder.members_asof(delist_ts)
        # D back-fills the vacated slot at the NEXT rebalance, not before.
        assert (D, T_MAR, None, 3, "enter_top3") in env.rows()
        assert env.builder.members_asof(T_MAR) == sorted([A, B, D])

    def test_delisting_after_last_rebalance_still_closes(self, tmp_path: Path) -> None:
        delist_ts = T_FEB + 10 * DAY
        env = Env(tmp_path, [inst("AAA"), inst("BBB"), inst("CCC", delisted=delist_ts)])
        for instrument_id, qv in ((A, 800.0), (B, 700.0), (C, 600.0)):
            env.write_daily(instrument_id, JAN_DAYS, qv)
        env.builder.rebuild(start=T_FEB, end=T_FEB + 15 * DAY, now=FAR_FUTURE)
        assert (C, T_FEB, delist_ts, 3, "delisted") in env.rows()


class TestPIT:
    def test_volume_after_rebalance_cannot_influence_membership(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        baseline = env.rows()
        # Plant gigantic volume prints for the least liquid name at T and T+1h —
        # information no trader at T could have had.
        env.write_daily(H, [T_FEB], 1e12, ts_offset=0)
        env.write_daily(H, [T_FEB], 1e12, ts_offset=HOUR)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert env.rows() == baseline

    def test_rebuild_is_deterministic_and_full_replace(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        feb = {A: 800.0, F: 700.0, D: 600.0, C: 500.0, E: 400.0, B: 250.0, G: 200.0, H: 100.0}
        for instrument_id, qv in feb.items():
            env.write_daily(instrument_id, FEB_DAYS, qv)
        s1 = env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)
        first = env.rows()
        s2 = env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)
        assert env.rows() == first  # identical intervals, no duplicated rows
        assert s1 == s2

    def test_future_rebalances_are_skipped(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        standard_jan_volumes(env)
        # now == T_FEB: the T_MAR rebalance has not happened yet and must not run.
        stats = env.builder.rebuild(start=T_FEB, end=T_MAR, now=T_FEB)
        assert stats.rebalances == 1
        assert env.builder.members_asof(T_MAR) == sorted([A, B, C])


class TestGuards:
    def test_non_monthly_rebalance_rejected(self, tmp_path: Path) -> None:
        env = Env(tmp_path, [])
        weekly = UniverseCfg(
            size=3, rank_window_days=30, rebalance="weekly", entry_rank=3, exit_rank=5
        )
        with pytest.raises(ConfigError, match="monthly"):
            UniverseBuilder(
                PITDataReader(LakePaths(tmp_path / "lake")),
                env.instruments,
                env.store,
                weekly,
            )

    def test_end_before_start_rejected(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())
        with pytest.raises(ValueError, match="must be >="):
            env.builder.rebuild(start=T_MAR, end=T_FEB, now=FAR_FUTURE)

    def test_no_eligible_instruments_yields_empty_universe(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_eight())  # instruments exist but the lake has no bars
        stats = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert stats.intervals == 0
        assert env.builder.members_asof(T_FEB) == []
