"""Unit tests for InstrumentSeeder — the survivorship-bias fix (leakage finding 1).

Exercises the live ∩ vision merge: live instruments keep their real metadata,
vision-only symbols (FTTUSDT-shaped delistings) become HISTORY-ONLY instruments with
archive-derived listed/delisted timestamps, re-seeding is SCD2-idempotent, and an
archive that grows produces a new version (not a duplicate).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe, parse_utc
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.seed import HISTORY_ONLY_EPSILON, InstrumentSeeder, SeedReport
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import YearMonth

AS_OF = parse_utc("2026-06-01T00:00:00Z")


def _perp(symbol: str, *, tick_size: float = 0.1) -> Instrument:
    """A realistic live perp instrument for the stub live source."""
    return Instrument(
        instrument_id=f"BINANCE:PERP:{symbol}",
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        tick_size=tick_size,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
        can_short=True,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        funding_interval_hours=8,
        listed_ts=parse_utc("2019-09-08T00:00:00Z"),
        delisted_ts=None,
    )


class StubLiveSource(DataSource):
    """Canned ``list_instruments``; bar/funding fetchers are out of seeding scope."""

    name = "stub.live"

    def __init__(self, instruments: list[Instrument]) -> None:
        self._instruments = instruments
        self.as_of_calls: list[Ms] = []

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        self.as_of_calls.append(as_of)
        return list(self._instruments)

    def fetch_ohlcv(
        self, instrument_id: str, *, tf: Timeframe, since: Ms, until: Ms, now: Ms | None = None
    ) -> pa.Table:  # pragma: no cover - not used by the seeder
        raise NotImplementedError

    def fetch_funding(
        self, instrument_id: str, *, since: Ms, until: Ms
    ) -> pa.Table:  # pragma: no cover - not used by the seeder
        raise NotImplementedError


class StubVision:
    """Duck-typed stand-in for BinanceVisionClient's catalog surface."""

    def __init__(
        self, symbols: list[str], ranges: dict[str, tuple[YearMonth, YearMonth] | None]
    ) -> None:
        self._symbols = symbols
        self._ranges = ranges
        self.month_range_calls: list[str] = []

    def list_symbols(self) -> list[str]:
        return list(self._symbols)

    def month_range(self, symbol: str) -> tuple[YearMonth, YearMonth] | None:
        self.month_range_calls.append(symbol)
        return self._ranges.get(symbol)


def _vision_stub() -> StubVision:
    return StubVision(
        symbols=["BTCUSDT", "ETHUSDT", "FTTUSDT", "LUNAUSDT", "NODATAUSDT"],
        ranges={
            "FTTUSDT": (YearMonth(2021, 1), YearMonth(2022, 11)),
            "LUNAUSDT": (YearMonth(2020, 9), YearMonth(2022, 5)),
            "NODATAUSDT": None,
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> InstrumentStore:
    with InstrumentStore(tmp_path / "instruments.sqlite") as s:
        yield s


def _seeder(
    store: InstrumentStore,
    live: list[Instrument] | None = None,
    vision: StubVision | None = None,
) -> tuple[InstrumentSeeder, StubLiveSource, StubVision]:
    live_source = StubLiveSource(live if live is not None else [_perp("BTCUSDT"), _perp("ETHUSDT")])
    vision_client = vision if vision is not None else _vision_stub()
    # StubVision is duck-typed; the seeder only touches list_symbols/month_range.
    seeder = InstrumentSeeder(live_source, vision_client, store)  # type: ignore[arg-type]
    return seeder, live_source, vision_client


class TestSeedMerge:
    def test_report_counts(self, store: InstrumentStore) -> None:
        seeder, live_source, _ = _seeder(store)
        report = seeder.seed(as_of=AS_OF)
        assert isinstance(report, SeedReport)
        assert report.live_count == 2
        assert report.vision_only_count == 3  # FTT, LUNA, NODATA absent from live
        assert report.delisted == 2  # NODATA has no archived months → skipped
        assert report.examples == ("BINANCE:PERP:FTTUSDT", "BINANCE:PERP:LUNAUSDT")
        assert live_source.as_of_calls == [AS_OF]

    def test_live_instruments_upserted_verbatim(self, store: InstrumentStore) -> None:
        seeder, _, _ = _seeder(store)
        seeder.seed(as_of=AS_OF)
        btc = store.get("BINANCE:PERP:BTCUSDT", AS_OF)
        assert btc == _perp("BTCUSDT")  # live metadata, NOT placeholder-clobbered
        assert btc.tick_size == 0.1
        assert btc.delisted_ts is None

    def test_history_only_instrument_fields(self, store: InstrumentStore) -> None:
        """The FTTUSDT-shaped case: untradable placeholder, archive-derived lifecycle."""
        seeder, _, _ = _seeder(store)
        seeder.seed(as_of=AS_OF)
        ftt = store.get("BINANCE:PERP:FTTUSDT", AS_OF)
        assert ftt is not None
        assert ftt.base == "FTT"
        assert ftt.quote == "USDT"
        assert ftt.asset_class is AssetClass.CRYPTO_PERP
        assert ftt.market_type is MarketType.PERP
        # lifecycle from archive coverage: first month start .. month after last month
        assert ftt.listed_ts == parse_utc("2021-01-01T00:00:00Z")
        assert ftt.delisted_ts == parse_utc("2022-12-01T00:00:00Z")
        # documented placeholders — valid but untradable
        assert ftt.tick_size == HISTORY_ONLY_EPSILON
        assert ftt.lot_size == HISTORY_ONLY_EPSILON
        assert ftt.min_qty == HISTORY_ONLY_EPSILON
        assert ftt.min_notional == 0.0
        assert ftt.maker_fee_bps == 2.0
        assert ftt.taker_fee_bps == 5.0
        assert ftt.can_short is True
        assert ftt.funding_interval_hours == 8

    def test_full_point_in_time_listing_visible(self, store: InstrumentStore) -> None:
        seeder, _, _ = _seeder(store)
        seeder.seed(as_of=AS_OF)
        known = {inst.instrument_id for inst in store.all_known(AS_OF)}
        assert known == {
            "BINANCE:PERP:BTCUSDT",
            "BINANCE:PERP:ETHUSDT",
            "BINANCE:PERP:FTTUSDT",
            "BINANCE:PERP:LUNAUSDT",
        }
        # SCD2: nothing is known before the observation instant
        assert store.all_known(AS_OF - 1) == []

    def test_no_data_symbol_not_seeded(self, store: InstrumentStore) -> None:
        seeder, _, vision = _seeder(store)
        seeder.seed(as_of=AS_OF)
        assert store.get("BINANCE:PERP:NODATAUSDT", AS_OF) is None
        assert "NODATAUSDT" in vision.month_range_calls

    def test_unparseable_vision_symbol_skipped(self, store: InstrumentStore) -> None:
        vision = StubVision(symbols=["BTCUSDT", "WEIRDXYZ"], ranges={})
        seeder, _, _ = _seeder(store, vision=vision)
        report = seeder.seed(as_of=AS_OF)
        assert report.vision_only_count == 1  # WEIRDXYZ
        assert report.delisted == 0
        assert report.examples == ()

    def test_live_intersection_never_queried_for_months(self, store: InstrumentStore) -> None:
        """Symbols already covered by the live source skip archive coverage probes."""
        seeder, _, vision = _seeder(store)
        seeder.seed(as_of=AS_OF)
        assert "BTCUSDT" not in vision.month_range_calls
        assert "ETHUSDT" not in vision.month_range_calls


class TestSeedIdempotency:
    def test_reseed_same_inputs_writes_zero_new_rows(self, store: InstrumentStore) -> None:
        seeder, _, _ = _seeder(store)
        first = seeder.seed(as_of=AS_OF)
        history_after_first = {
            iid: len(store.history(iid))
            for iid in (
                "BINANCE:PERP:BTCUSDT",
                "BINANCE:PERP:ETHUSDT",
                "BINANCE:PERP:FTTUSDT",
                "BINANCE:PERP:LUNAUSDT",
            )
        }
        assert all(n == 1 for n in history_after_first.values())

        # same as_of and one hour later — both must be SCD2 no-ops
        second = seeder.seed(as_of=AS_OF)
        third = seeder.seed(as_of=AS_OF + 3_600_000)
        for report in (second, third):
            assert report.live_count == first.live_count
            assert report.delisted == first.delisted
        for iid, n in history_after_first.items():
            assert len(store.history(iid)) == n  # zero new SCD2 rows

    def test_grown_archive_creates_new_version(self, store: InstrumentStore) -> None:
        """A later seed observing more archived months versions the lifecycle change."""
        seeder, _, _ = _seeder(store)
        seeder.seed(as_of=AS_OF)

        grown = _vision_stub()
        grown._ranges["LUNAUSDT"] = (YearMonth(2020, 9), YearMonth(2022, 6))
        seeder2, _, _ = _seeder(store, vision=grown)
        later = AS_OF + 86_400_000
        seeder2.seed(as_of=later)

        history = store.history("BINANCE:PERP:LUNAUSDT")
        assert len(history) == 2
        assert history[0][1] == later  # old version closed at the new observation
        assert history[1][2].delisted_ts == parse_utc("2022-07-01T00:00:00Z")
        # FTT unchanged → still a single version
        assert len(store.history("BINANCE:PERP:FTTUSDT")) == 1
