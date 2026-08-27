"""Unit tests for UniverseBuilder on the EQUITIES sleeve (EQUITIES_SLEEVE.md §5).

The builder is reused verbatim — only ``rank_tf=Timeframe.D1`` changes (native daily
bars whose ``quote_volume`` is already that session's dollar volume, so the per-UTC-day
sum collapses to a pass-through and the trailing median IS ADV). These tests mirror the
crypto ``test_universe_builder.py`` discipline on a synthetic equity lake:

- top-N by ADV (dollar) with exact ranks at the rebalance instant;
- entry/exit hysteresis band semantics across two monthly rebalances;
- THE SURVIVORSHIP GATE (the equity FTT/LUNA analogue, EQUITIES_SLEEVE.md §1.4): a
  pre-/mid-period DELISTED ticker, seeded into the SCD2 store from the historical
  listing record, is ranked while it lived and its membership interval CLOSES at
  ``delisted_ts`` — built from the STORE, not the source list (EQUITIES_CRITIQUE.md N5).

Equity ids use the ``<EXCHANGE>:CASH:<TICKER>USD`` canonical form (base=<TICKER>,
quote="USD") so they round-trip through the existing ``SymbolMapper`` invariant. Daily
bars are written NATIVELY to ``Dataset.OHLCV_1D`` with ``tf=Timeframe.D1`` (not
resampler-derived, EQUITIES_CRITIQUE.md B3). Never touches ./data — every lake lives
under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.config.settings import UniverseCfg
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.schemas import Dataset
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter
from alphaforge.data.universe.builder import UniverseBuilder
from alphaforge.data.universe.store import UniverseStore

DAY = 86_400_000
D1 = Timeframe.D1

T_JAN = 1_704_067_200_000  # 2024-01-01T00:00:00Z
T_FEB = 1_706_745_600_000  # 2024-02-01T00:00:00Z (rebalance T1)
T_MAR = 1_709_251_200_000  # 2024-03-01T00:00:00Z (rebalance T2)
FAR_FUTURE = 1_900_000_000_000  # 2030-03-17; "now" for historical fixtures
LISTED = T_FEB - 365 * DAY  # comfortably past the 30d min listing age

# One D1 bar per calendar day (the test grid need not be the NYSE session grid: the
# builder reads whatever D1 bars the lake holds and ranks ADV; session-grid wiring is the
# factor/engine concern, not the universe builder's). Jan 2..Jan 31 = the T1 window.
JAN_DAYS = [T_JAN + k * DAY for k in range(1, 31)]
FEB_DAYS = [T_FEB + k * DAY for k in range(28)]  # Feb 1 .. Feb 28 (bulk of the T2 window)

# entry_rank=3 / exit_rank=5 mirrors the production 20/26 hysteresis on a small fixture.
CFG = UniverseCfg(size=3, rank_window_days=30, rebalance="monthly", entry_rank=3, exit_rank=5)

_TS = pa.timestamp("ms", tz="UTC")


def iid(ticker: str) -> str:
    """Canonical equity instrument id: ``XNAS:CASH:<TICKER>USD``."""
    return f"XNAS:CASH:{ticker}USD"


# Common equities plus LEHMQ, the delisted survivorship probe (the equity FTT analogue).
AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA, LEHM = (
    iid(t) for t in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "LEHM")
)


def equity(ticker: str, *, listed: int = LISTED, delisted: int | None = None) -> Instrument:
    """An equity ``Instrument`` (cash segment, no funding) per EQUITIES_SLEEVE.md §1.1."""
    return Instrument(
        instrument_id=iid(ticker),
        asset_class=AssetClass.EQUITY,
        market_type=MarketType.CASH,
        base=ticker,
        quote="USD",
        tick_size=0.01,
        lot_size=1.0,
        min_qty=1.0,
        min_notional=0.0,
        can_short=True,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        funding_interval_hours=None,
        listed_ts=listed,
        delisted_ts=delisted,
    )


class Env:
    """One synthetic equity lake + SCD2 record + builder (rank_tf=D1) per test."""

    def __init__(self, tmp_path: Path, instruments: list[Instrument]) -> None:
        paths = LakePaths(tmp_path / "lake")
        self.writer = LakeWriter(paths)
        self.store = UniverseStore(paths)
        self.instruments = InstrumentStore(tmp_path / "ops.sqlite")
        # SCD2 seed FROM THE LISTING RECORD: delisted names are persisted so the builder,
        # which reads instruments.all_known(as_of=now), ranks them while they lived
        # (EQUITIES_CRITIQUE.md N5 — the universe is built from the STORE).
        for i in instruments:
            self.instruments.upsert(i, as_of=i.listed_ts)
        self.builder = UniverseBuilder(
            PITDataReader(paths), self.instruments, self.store, CFG, rank_tf=D1
        )

    def write_daily(
        self,
        instrument_id: str,
        day_starts: list[int],
        dollar_volume: float | None,
        *,
        volume: float = 1_000.0,
        close: float = 100.0,
    ) -> None:
        """One NATIVE D1 bar per day, ``quote_volume`` = that session's dollar volume.

        Written to ``Dataset.OHLCV_1D`` with ``tf=D1`` so the writer's partial-bar
        guard keys on the bar's own ``ts_open + D1.ms`` (native ingest, not resampled).
        """
        opens = list(day_starts)
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
                "quote_volume": pa.array([dollar_volume] * n, type=pa.float64()),
                "n_trades": pa.array([100] * n, type=pa.int64()),
                "quality_flags": pa.array([0] * n, type=pa.int32()),
                "ingested_at": pa.array([o + DAY + 30_000 for o in opens], type=_TS),
            }
        )
        self.writer.write(Dataset.OHLCV_1D, tbl, tf=D1, now=FAR_FUTURE)

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


def all_seven() -> list[Instrument]:
    return [equity(t) for t in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA")]


def standard_jan_adv(env: Env) -> None:
    """7 names, Jan daily dollar volumes 7e9..1e9 → ADV ranks AAPL=1 .. TSLA=7 at T_FEB."""
    advs = (7e9, 6e9, 5e9, 4e9, 3e9, 2e9, 1e9)
    for instrument_id, dv in zip((AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA), advs, strict=True):
        env.write_daily(instrument_id, JAN_DAYS, dv)


class TestADVSelection:
    def test_top_n_by_adv_exact(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)
        stats = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert (stats.rebalances, stats.intervals, stats.entries, stats.exits) == (1, 3, 3, 0)
        assert env.builder.members_asof(T_FEB) == sorted([AAPL, MSFT, NVDA])
        assert env.rows() == [
            (AAPL, T_FEB, None, 1, "enter_top3"),
            (MSFT, T_FEB, None, 2, "enter_top3"),
            (NVDA, T_FEB, None, 3, "enter_top3"),
        ]

    def test_native_d1_quote_volume_is_adv_passthrough(self, tmp_path: Path) -> None:
        # A single D1 bar per day IS the daily dollar volume; the trailing median over
        # the window is ADV. NVDA outranks AMZN purely on dollar volume.
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert (NVDA, T_FEB, None, 3, "enter_top3") in env.rows()
        assert AMZN not in env.builder.members_asof(T_FEB)

    def test_quote_volume_null_falls_back_to_volume_times_close(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)
        # MSFT rewritten with null quote_volume: 60e6 * 100 = 6e9/day → still rank 2.
        env.write_daily(MSFT, JAN_DAYS, None, volume=60e6, close=100.0)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert (MSFT, T_FEB, None, 2, "enter_top3") in env.rows()


class TestHysteresis:
    def test_band_semantics_across_two_rebalances(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)  # T1 ranks: AAPL1 MSFT2 NVDA3 → members {AAPL, MSFT, NVDA}
        # Feb ADV → T2 ranks: AAPL1 META2 AMZN3 NVDA4 GOOG5 MSFT6 TSLA7.
        feb = {AAPL: 7e9, META: 6e9, AMZN: 5e9, NVDA: 4e9, GOOG: 3e9, MSFT: 2e9, TSLA: 1e9}
        for instrument_id, dv in feb.items():
            env.write_daily(instrument_id, FEB_DAYS, dv)
        env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)

        # Member NVDA at rank 4 (inside (3, 5] band) STAYS with its original entry row.
        assert (NVDA, T_FEB, None, 3, "enter_top3") in env.rows()
        # Member MSFT at rank 6 (> exit_rank 5) EXITS at T2.
        assert (MSFT, T_FEB, T_MAR, 2, "exit_rank_6") in env.rows()
        # Non-member GOOG at rank 5 (inside the band) does NOT enter.
        assert all(r[0] != GOOG for r in env.rows())
        # AMZN and META enter at T2 with their entry ranks.
        assert (AMZN, T_MAR, None, 3, "enter_top3") in env.rows()
        assert (META, T_MAR, None, 2, "enter_top3") in env.rows()
        assert env.builder.members_asof(T_MAR) == sorted([AAPL, AMZN, META, NVDA])


class TestSurvivorship:
    """THE equity survivorship gate (EQUITIES_SLEEVE.md §1.4) — the FTT/LEHMQ analogue."""

    def test_delisted_ticker_interval_closes_at_delisted_ts(self, tmp_path: Path) -> None:
        # LEHMQ dies mid-February; it must be ranked while it lived and its membership
        # interval must close at delisted_ts — NOT at month-end, NOT absent.
        delist_ts = T_FEB + 11 * DAY
        instruments = [
            equity("AAPL"),
            equity("MSFT"),
            equity("LEHM", delisted=delist_ts),
            equity("NVDA"),
        ]
        env = Env(tmp_path, instruments)
        for instrument_id, dv in ((AAPL, 7e9), (MSFT, 6e9), (LEHM, 5e9), (NVDA, 4e9)):
            env.write_daily(instrument_id, JAN_DAYS, dv)  # Jan: LEHM is rank 3, a member
        for instrument_id, dv in ((AAPL, 7e9), (MSFT, 6e9), (NVDA, 4e9)):
            env.write_daily(instrument_id, FEB_DAYS, dv)
        env.write_daily(LEHM, FEB_DAYS[:10], 5e9)  # trades until its death

        env.builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)

        # The dead ticker's interval EXISTS, closed at delisted_ts (built from the STORE).
        assert (LEHM, T_FEB, delist_ts, 3, "delisted") in env.rows()
        assert LEHM in env.builder.members_asof(delist_ts - 1)
        assert LEHM not in env.builder.members_asof(delist_ts)
        # NVDA back-fills the vacated slot at the NEXT rebalance, not before.
        assert (NVDA, T_MAR, None, 3, "enter_top3") in env.rows()
        assert env.builder.members_asof(T_MAR) == sorted([AAPL, MSFT, NVDA])

    def test_delisted_instrument_is_persisted_and_known_to_store(self, tmp_path: Path) -> None:
        # N5: the delisted name must be PERSISTED in the SCD2 store (not merely returned
        # by a source). all_known(as_of=now) surfaces it with the right delisted_ts.
        delist_ts = T_FEB + 11 * DAY
        env = Env(tmp_path, [equity("AAPL"), equity("LEHM", delisted=delist_ts)])
        known = {i.instrument_id: i for i in env.instruments.all_known(as_of=FAR_FUTURE)}
        assert LEHM in known
        assert known[LEHM].delisted_ts == delist_ts
        assert known[AAPL].delisted_ts is None

    def test_delisting_before_first_rebalance_excludes_dead_name(self, tmp_path: Path) -> None:
        # A name delisted BEFORE the rebalance instant is ineligible (delisted_ts <= T)
        # and never enters — the eligibility predicate, not a phantom membership.
        dead_before = equity("LEHM", delisted=T_FEB - DAY)
        env = Env(tmp_path, [equity("AAPL"), equity("MSFT"), equity("NVDA"), dead_before])
        for instrument_id, dv in ((AAPL, 7e9), (MSFT, 6e9), (NVDA, 5e9), (LEHM, 9e9)):
            env.write_daily(instrument_id, JAN_DAYS, dv)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert LEHM not in env.builder.members_asof(T_FEB)
        assert env.builder.members_asof(T_FEB) == sorted([AAPL, MSFT, NVDA])


class TestPIT:
    def test_volume_after_rebalance_cannot_influence_membership(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        baseline = env.rows()
        # Gigantic dollar-volume print for the least liquid name AT and AFTER T — no
        # trader at T could have seen it; the D1 bar for Feb 1 closes Feb 2 (> T_FEB).
        env.write_daily(TSLA, [T_FEB], 1e15)
        env.write_daily(TSLA, [T_FEB + DAY], 1e15)
        env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert env.rows() == baseline

    def test_rebuild_is_deterministic_and_full_replace(self, tmp_path: Path) -> None:
        env = Env(tmp_path, all_seven())
        standard_jan_adv(env)
        s1 = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        first = env.rows()
        s2 = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert env.rows() == first
        assert s1 == s2


@pytest.mark.parametrize("instruments", [[], None])
class TestEdgeCases:
    def test_empty_universe_when_no_bars(
        self, tmp_path: Path, instruments: list[Instrument] | None
    ) -> None:
        names = all_seven() if instruments is None else instruments
        env = Env(tmp_path, names)  # instruments exist (or not) but the lake has no bars
        stats = env.builder.rebuild(start=T_FEB, end=T_FEB, now=FAR_FUTURE)
        assert stats.intervals == 0
        assert env.builder.members_asof(T_FEB) == []
