"""Cross-module smoke test for the Phase 1 core kernel.

Exercises the curated :mod:`alphaforge.core` surface end to end: settings load
from the real repo configs, SCD2 instrument persistence, symbol round-trips,
24/7 calendar bar arithmetic, and Arrow schema validation — all through the
public re-exports, so any drift between modules (or a broken ``__init__``)
fails here before downstream phases hit it.
"""

from itertools import pairwise
from pathlib import Path

from alphaforge.config.settings import Settings, load_settings
from alphaforge.core import (
    Always24x7Calendar,
    AssetClass,
    Instrument,
    InstrumentStore,
    MarketType,
    SymbolMapper,
    Timeframe,
    now_ms,
    parse_utc,
)
from alphaforge.data.schemas import Dataset, empty_table, validate_table

REPO_ROOT = Path(__file__).resolve().parents[2]

BTC_PERP_ID = "BINANCE:PERP:BTCUSDT"

#: BTCUSDT perpetual listing on Binance Futures (2019-09-08), well before any test "now".
LISTED_TS = parse_utc("2019-09-08T00:00:00Z")

#: Fixed observation time for the SCD2 upsert; deterministic and always < now_ms().
OBSERVED_TS = parse_utc("2024-01-02T00:00:00Z")


def _btc_perp() -> Instrument:
    """Build a realistic BTCUSDT linear perp (Binance USDT-M filter values)."""
    return Instrument(
        instrument_id=BTC_PERP_ID,
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base="BTC",
        quote="USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=100.0,
        can_short=True,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        funding_interval_hours=8,
        listed_ts=LISTED_TS,
        delisted_ts=None,
    )


def test_settings_load_from_repo_configs() -> None:
    """load_settings() reads configs/base.yaml and resolves paths absolute."""
    settings = load_settings(root=REPO_ROOT)
    assert isinstance(settings, Settings)
    assert settings.data.exchange == "binanceusdm"
    assert settings.data.timeframe is Timeframe.H1
    assert settings.data.backfill_start_ms == parse_utc("2020-01-01T00:00:00Z")
    assert settings.paths.lake_dir.is_absolute()
    assert settings.paths.lake_dir == REPO_ROOT / "data" / "lake"


def test_instrument_store_roundtrip_with_symbol_mapping(tmp_path: Path) -> None:
    """Upsert a perp, read it back PIT as of now, and round-trip its id via ccxt."""
    inst = _btc_perp()
    db_path = tmp_path / "meta" / "instruments.sqlite"
    with InstrumentStore(db_path) as store:
        assert store.upsert(inst, as_of=OBSERVED_TS) is True
        fetched = store.get(BTC_PERP_ID, as_of=now_ms())
    assert fetched == inst

    ccxt_symbol = SymbolMapper.to_ccxt(fetched.instrument_id if fetched else "")
    assert ccxt_symbol == "BTC/USDT:USDT"
    assert SymbolMapper.from_ccxt("BINANCE", ccxt_symbol) == BTC_PERP_ID


def test_calendar_one_day_of_hourly_bar_opens() -> None:
    """A 24/7 calendar yields exactly 24 H1 opens over one UTC day, half-open."""
    cal = Always24x7Calendar()
    start = parse_utc("2024-01-02T00:00:00Z")
    end = parse_utc("2024-01-03T00:00:00Z")
    opens = cal.expected_bar_opens(start, end, Timeframe.H1)
    assert len(opens) == 24
    assert opens[0] == start
    assert opens[-1] == end - Timeframe.H1.ms  # end exclusive
    assert all(b - a == Timeframe.H1.ms for a, b in pairwise(opens))
    assert cal.periods_per_year(Timeframe.H1) == 8760.0


def test_empty_ohlcv_table_validates() -> None:
    """empty_table(OHLCV) conforms to its own schema contract."""
    tbl = empty_table(Dataset.OHLCV)
    assert tbl.num_rows == 0
    validate_table(tbl, Dataset.OHLCV)  # must not raise
