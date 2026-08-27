"""Survivorship regression against the SEED path — not just synthetic instruments.

The unit suites prove the two halves in isolation: ``test_seed.py`` shows the live ∩
vision merge records a delisted perp (FTTUSDT-shaped) as a HISTORY-ONLY instrument with
an archive-derived lifecycle, and ``test_universe_builder.py`` shows the builder closes a
delisted member's interval at ``delisted_ts``. This integration test welds the halves:
it drives the *real* :class:`InstrumentSeeder` over a real :class:`InstrumentStore`, then
the *real* :class:`UniverseBuilder` over a real Parquet lake, and asserts the architecture
is survivorship-free end to end — a perp delisted years before "now" both (a) enters the
historical universe while it lived and (b) has its membership interval close at the
seed-derived ``delisted_ts``, never at the live "now".

The known casualty is **FTTUSDT** (FTX's token perp, delisted on Binance USDⓂ around
2022-Q3 in the FTX collapse). It is invisible to Binance's live REST listing today, so a
universe built only from the live endpoint would silently exclude it from every 2021
backtest — exactly the survivorship hole :class:`InstrumentSeeder` exists to close.

Zero network: the live REST endpoint is a canned :class:`DataSource` double returning only
survivors (BTC/ETH), and the Binance Vision archive is a catalog double duck-typed to the
:class:`~alphaforge.data.sources.vision.BinanceVisionClient` surface the seeder touches
(``list_symbols`` / ``month_range``). These are the same test doubles the unit suites use;
the gap vs. a fully-real path is documented in the module-level note below.

GAP NOTE — why the archive client is a double, not the real ``BinanceVisionClient``:
the real client's ``list_symbols``/``month_range`` enumerate objects from the public
Binance Vision S3 bucket over HTTPS (``BinanceVisionClient._list`` issues real requests).
There is no offline-fixture constructor, so a fully-real archive read cannot run in CI
without the network (and HARD RULE: no network). We therefore feed the *real* seeder a
canned archive catalog and *real* OHLCV bars written to a *real* lake; everything
downstream of ``vision.list_symbols``/``vision.month_range`` — the seeder's merge logic,
the SCD2 store, the PIT lake read, and the universe builder — is the production code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from alphaforge.config.settings import UniverseCfg
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe, parse_utc
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.seed import InstrumentSeeder
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import YearMonth
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter
from alphaforge.data.universe.builder import UniverseBuilder
from alphaforge.data.universe.store import UniverseStore

if TYPE_CHECKING:
    from pathlib import Path

HOUR = 3_600_000
DAY = 86_400_000
_TS = pa.timestamp("ms", tz="UTC")

# The dead name and its archive-derived lifecycle (conservative HISTORY-ONLY rule:
# listed_ts = first archived month start; delisted_ts = first instant of the month AFTER
# the last archived month). FTTUSDT traded into 2022-08 then died in the FTX collapse, so
# the archive bracket is 2021-01 .. 2022-08 and the conservative delisted_ts is 2022-09-01.
FTT_FIRST_MONTH = YearMonth(2021, 1)
FTT_LAST_MONTH = YearMonth(2022, 8)
FTT_LISTED_TS = parse_utc("2021-01-01T00:00:00Z")
FTT_DELISTED_TS = parse_utc("2022-09-01T00:00:00Z")  # month AFTER last archived month
FTT = "BINANCE:PERP:FTTUSDT"
BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"

# "Now" is observed long AFTER FTTUSDT's death — the whole point is that a live-only
# listing as of this instant has already forgotten FTTUSDT ever existed.
SEED_AS_OF = parse_utc("2026-06-01T00:00:00Z")
NOW = SEED_AS_OF

# 2021 universe window. Survivors and the dead name all listed comfortably before the
# first rebalance, so the min-listing-age gate never excludes them.
LIVE_LISTED_TS = parse_utc("2020-06-01T00:00:00Z")
WINDOW_START = parse_utc("2021-03-01T00:00:00Z")  # first rebalance instant of the window
# Carry the rebuild horizon past the delisting so the close at delisted_ts is captured.
WINDOW_END = parse_utc("2022-12-01T00:00:00Z")
T_2021 = parse_utc("2021-06-01T00:00:00Z")  # an in-window 2021 rebalance, FTT alive here

# A small top-N config scaled to the 3-instrument fixture (mirrors prod 20/26 hysteresis).
CFG = UniverseCfg(size=3, rank_window_days=30, rebalance="monthly", entry_rank=3, exit_rank=5)


def _live_perp(symbol: str) -> Instrument:
    """A realistic *currently listed* perp the live REST endpoint would return."""
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
        listed_ts=LIVE_LISTED_TS,
        delisted_ts=None,
    )


class _CannedLiveSource(DataSource):
    """Live REST double: returns ONLY survivors (BTC/ETH) — FTTUSDT is gone from it.

    This is precisely the survivorship hole: the live endpoint reports currently listed
    markets only, so a store seeded from it alone never learns FTTUSDT existed.
    """

    name = "canned.live"

    def __init__(self, instruments: list[Instrument]) -> None:
        self._instruments = instruments

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        return list(self._instruments)

    def fetch_ohlcv(
        self, instrument_id: str, *, tf: Timeframe, since: Ms, until: Ms, now: Ms | None = None
    ) -> pa.Table:  # pragma: no cover - not used by the seeder
        raise NotImplementedError

    def fetch_funding(
        self, instrument_id: str, *, since: Ms, until: Ms
    ) -> pa.Table:  # pragma: no cover - not used by the seeder
        raise NotImplementedError


class _CannedVision:
    """Archive catalog double, duck-typed to BinanceVisionClient's seeding surface.

    Only ``list_symbols``/``month_range`` are touched by :class:`InstrumentSeeder`; the
    real client reaches the public S3 bucket over HTTPS (see module GAP NOTE).
    """

    def __init__(self, ranges: dict[str, tuple[YearMonth, YearMonth] | None]) -> None:
        self._ranges = ranges

    def list_symbols(self) -> list[str]:
        return sorted(self._ranges)

    def month_range(self, symbol: str) -> tuple[YearMonth, YearMonth] | None:
        return self._ranges.get(symbol)


def _write_monthly_bars(
    writer: LakeWriter,
    instrument_id: str,
    *,
    months: list[YearMonth],
    quote_volume: float,
) -> None:
    """Write daily 1h bars (one per UTC day at 12:00) across the given months.

    Enough distinct days per month that the builder's ``min_days`` ranking gate is
    satisfied for any in-month rebalance window.
    """
    opens: list[int] = []
    for ym in months:
        month_start = ym.first_ms()
        opens.extend(month_start + d * DAY + 12 * HOUR for d in range(28))
    n = len(opens)
    tbl = pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_open": pa.array(opens, type=_TS),
            "open": pa.array([100.0] * n, type=pa.float64()),
            "high": pa.array([100.0] * n, type=pa.float64()),
            "low": pa.array([100.0] * n, type=pa.float64()),
            "close": pa.array([100.0] * n, type=pa.float64()),
            "volume": pa.array([10.0] * n, type=pa.float64()),
            "quote_volume": pa.array([quote_volume] * n, type=pa.float64()),
            "n_trades": pa.array([1] * n, type=pa.int64()),
            "quality_flags": pa.array([0] * n, type=pa.int32()),
            "ingested_at": pa.array([o + HOUR + 30_000 for o in opens], type=_TS),
        }
    )
    writer.write(Dataset.OHLCV, tbl, now=NOW)


def _months_between(first: YearMonth, last: YearMonth) -> list[YearMonth]:
    out: list[YearMonth] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append(YearMonth(year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


class Env:
    """Real seed → real SCD2 store → real lake → real universe builder, no network."""

    def __init__(self, tmp_path: Path) -> None:
        paths = LakePaths(tmp_path / "lake")
        self.writer = LakeWriter(paths)
        self.universe_store = UniverseStore(paths)
        self.instruments = InstrumentStore(tmp_path / "instruments.sqlite")

        # 1) Seed the instrument store via the REAL seeder: live (survivors only) merged
        #    with the archive catalog that still remembers FTTUSDT.
        live = _CannedLiveSource([_live_perp("BTCUSDT"), _live_perp("ETHUSDT")])
        vision = _CannedVision(
            {
                "BTCUSDT": (FTT_FIRST_MONTH, FTT_LAST_MONTH),  # live ∩ vision: no merge
                "ETHUSDT": (FTT_FIRST_MONTH, FTT_LAST_MONTH),
                "FTTUSDT": (FTT_FIRST_MONTH, FTT_LAST_MONTH),  # vision-only → HISTORY-ONLY
            }
        )
        # _CannedVision is duck-typed; the seeder only calls list_symbols/month_range.
        seeder = InstrumentSeeder(live, vision, self.instruments)  # type: ignore[arg-type]
        self.report = seeder.seed(as_of=SEED_AS_OF)

        # 2) Real OHLCV lake. BTC/ETH trade the whole window; FTTUSDT trades only across
        #    its archived life (2021-01 .. 2022-08), with the TOP quote volume so it is
        #    unambiguously selected while it lived.
        live_months = _months_between(YearMonth(2021, 1), YearMonth(2022, 12))
        _write_monthly_bars(self.writer, BTC, months=live_months, quote_volume=200.0)
        _write_monthly_bars(self.writer, ETH, months=live_months, quote_volume=150.0)
        ftt_months = _months_between(FTT_FIRST_MONTH, FTT_LAST_MONTH)
        _write_monthly_bars(self.writer, FTT, months=ftt_months, quote_volume=900.0)

        self.builder = UniverseBuilder(
            PITDataReader(paths), self.instruments, self.universe_store, CFG
        )

    def rows(self) -> list[tuple[str, int, int | None, int, str]]:
        tbl = self.universe_store.read_intervals()
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


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


class TestSeedRecordsDelistedName:
    def test_live_only_listing_would_have_lost_ftt(self, env: Env) -> None:
        """Sanity: the live REST double genuinely omits FTTUSDT (the bias source)."""
        live = _CannedLiveSource([_live_perp("BTCUSDT"), _live_perp("ETHUSDT")])
        live_ids = {inst.instrument_id for inst in live.list_instruments(as_of=SEED_AS_OF)}
        assert FTT not in live_ids
        # ...yet the merged seed restores it from the archive.
        assert env.report.delisted == 1
        assert env.report.examples == (FTT,)

    def test_seed_recorded_ftt_with_conservative_lifecycle(self, env: Env) -> None:
        """The real seeder wrote a HISTORY-ONLY FTTUSDT with archive-derived lifecycle."""
        ftt = env.instruments.get(FTT, NOW)
        assert ftt is not None
        assert ftt.base == "FTT"
        assert ftt.asset_class is AssetClass.CRYPTO_PERP
        assert ftt.listed_ts == FTT_LISTED_TS
        assert ftt.delisted_ts == FTT_DELISTED_TS  # month AFTER last archived month
        # The survivors carry their real live metadata, untouched by the merge.
        btc = env.instruments.get(BTC, NOW)
        assert btc is not None and btc.delisted_ts is None


class TestSurvivorshipFreeUniverse:
    def test_ftt_is_a_member_during_2021_while_it_lived(self, env: Env) -> None:
        """A delisted-years-ago perp enters the 2021 universe — built off the SEED path."""
        env.builder.rebuild(start=WINDOW_START, end=WINDOW_END, now=NOW)
        members_2021 = env.builder.members_asof(T_2021)
        assert FTT in members_2021, (
            "FTTUSDT must appear in the 2021 membership: the architecture is "
            "survivorship-free against the real seed path, not just synthetic instruments."
        )

    def test_ftt_interval_closes_at_delisted_ts_not_now(self, env: Env) -> None:
        """THE survivorship assertion: the interval closes at delisted_ts, never at now."""
        env.builder.rebuild(start=WINDOW_START, end=WINDOW_END, now=NOW)
        ftt_intervals = [r for r in env.rows() if r[0] == FTT]
        assert ftt_intervals, "FTTUSDT must have at least one membership interval"

        # Exactly one closing interval, closed at the seed-derived delisted_ts.
        closed = [r for r in ftt_intervals if r[2] is not None]
        assert len(closed) == 1
        _iid, eff_from, eff_to, _rank, reason = closed[0]
        assert eff_to == FTT_DELISTED_TS  # NOT NOW, NOT month-end
        assert eff_to != NOW
        assert reason == "delisted"
        assert eff_from < FTT_DELISTED_TS

        # Membership flips off exactly at delisted_ts (half-open interval semantics).
        assert FTT in env.builder.members_asof(FTT_DELISTED_TS - 1)
        assert FTT not in env.builder.members_asof(FTT_DELISTED_TS)
        # And it is certainly not a live member as of "now", years later.
        assert FTT not in env.builder.members_asof(NOW)

    def test_rebuild_deterministic_off_the_seed_path(self, env: Env) -> None:
        """Rebuilding twice off the same seed yields byte-identical intervals."""
        s1 = env.builder.rebuild(start=WINDOW_START, end=WINDOW_END, now=NOW)
        first = env.rows()
        s2 = env.builder.rebuild(start=WINDOW_START, end=WINDOW_END, now=NOW)
        assert env.rows() == first
        assert s1 == s2
