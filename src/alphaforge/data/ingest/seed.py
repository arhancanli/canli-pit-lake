"""Instrument seeding: live listing merged with the Binance Vision archive (finding 1).

The #1 critical leakage fix. Seeding the instrument store from the live endpoint alone
(``DataSource.list_instruments``) bakes survivorship bias into every backtest: Binance's
REST API only reports *currently listed* markets, so every perp delisted before go-live
(FTTUSDT, LUNAUSDT, SRMUSDT, ...) would never enter the candidate set, and the
2020→now universe would consist exclusively of survivors. :class:`InstrumentSeeder`
closes the hole by merging two sources:

1. **Live** (``DataSource.list_instruments(as_of=...)``) — authoritative for currently
   listed instruments: real exchange filters, fees, lifecycle timestamps.
2. **Archive** (:class:`~alphaforge.data.sources.vision.BinanceVisionClient`) — the
   complete historical symbol catalog. Symbols present in the archive but absent from
   the live listing are delisted (or never ccxt-visible) and are recorded as
   **HISTORY-ONLY instruments**: untradable placeholder records that exist solely so
   universe construction and backtests see the full point-in-time listing. Their
   lifecycle is derived conservatively from archive coverage
   (:meth:`BinanceVisionClient.month_range`): ``listed_ts`` = first instant of the
   first archived month, ``delisted_ts`` = first instant of the month *after* the last
   archived month (the symbol demonstrably traded into its last month; the exact
   intra-month delisting instant is unknown, so the month edge is the conservative
   upper bound).

**Sources whose live listing is already survivorship-free skip the archive merge.**
The Binance Vision archive exists only because the crypto venue's live endpoint hides
delisted markets. An equity source
(:class:`~alphaforge.data.sources.polygon_source.PolygonEquitiesSource`) draws from
Polygon's reference endpoints, which return *delisted* tickers alongside active ones —
its ``list_instruments(as_of=...)`` is already the full point-in-time listing (dead names
carry a real ``delisted_ts``). For such a source ``vision`` is left ``None``:
:meth:`InstrumentSeeder.seed` upserts the live listing verbatim and the HISTORY-ONLY
merge is a no-op (there is nothing the live endpoint hides). The seeding algorithm is
otherwise identical, so a single generic seeder serves both sleeves; only the wired
:class:`DataSource` (and the presence/absence of ``vision``) differs.

HISTORY-ONLY placeholder trading parameters (the live exchange filters of a delisted
market are unrecoverable, and :class:`~alphaforge.core.instruments.Instrument` rejects
zero values by design):

* ``tick_size = lot_size = min_qty = 1e-8`` — sentinel epsilons: positive (valid) but
  obviously non-tradable; execution code must never route an order to an instrument
  whose ``delisted_ts`` is set.
* ``min_notional = 0.0`` — no constraint (field allows 0).
* ``maker/taker = 2.0/5.0`` bps — Binance VIP0 perp schedule, so any *historical*
  cost simulation over these names uses realistic fees.
* ``can_short = True``, ``funding_interval_hours = 8`` — Binance USDT-M perp defaults.

Idempotency: re-seeding with unchanged inputs writes **zero** new SCD2 rows —
``InstrumentStore.upsert`` is a no-op when the current version is attribute-identical.
When the archive grows (a live symbol's last month advances monthly, or Binance
republishes), the changed lifecycle produces a new SCD2 version, exactly as intended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from alphaforge.core.errors import SchemaError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.logging import get_logger
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.vision import BinanceVisionClient

__all__ = [
    "HISTORY_ONLY_EPSILON",
    "InstrumentSeeder",
    "SeedReport",
]

_log = get_logger(__name__)

HISTORY_ONLY_EPSILON: Final[float] = 1e-8
"""Sentinel ``tick_size``/``lot_size``/``min_qty`` for HISTORY-ONLY instruments.

Positive so :class:`Instrument` validation passes, small enough that no realistic
order quantization could mistake it for a genuine exchange filter. Execution code
must treat any instrument with ``delisted_ts`` set as untradable regardless."""

_HISTORY_ONLY_MAKER_FEE_BPS: Final[float] = 2.0
_HISTORY_ONLY_TAKER_FEE_BPS: Final[float] = 5.0
_HISTORY_ONLY_FUNDING_INTERVAL_HOURS: Final[int] = 8
_MAX_EXAMPLES: Final[int] = 10


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedReport:
    """Outcome of one :meth:`InstrumentSeeder.seed` run.

    - ``live_count``: instruments returned by the live source and upserted.
    - ``vision_only_count``: archive symbols absent from the live listing (the
      survivorship-bias exposure the live endpoint alone would have hidden).
    - ``delisted``: how many of those were actually recorded as HISTORY-ONLY
      instruments (those with at least one archived 1h month and a parseable
      symbol; the difference vs ``vision_only_count`` is skipped symbols, each
      logged with a reason).
    - ``examples``: up to 10 instrument_ids of the recorded HISTORY-ONLY
      instruments, sorted — for logs/reports (e.g. ``"BINANCE:PERP:FTTUSDT"``).
    """

    live_count: int
    vision_only_count: int
    delisted: int
    examples: tuple[str, ...]


class InstrumentSeeder:
    """Merges the live listing with the Binance Vision archive into the SCD2 store.

    Args:
        live_source: A :class:`DataSource` whose ``list_instruments`` reflects the
            venue's current listing (e.g. ``CCXTDataSource`` on ``binanceusdm``).
        vision: Optional archive catalog client; only :meth:`list_symbols` and
            :meth:`month_range` are used here (bar/funding fetching is the backfill
            job's concern). ``None`` when the live source is *already*
            survivorship-free (it returns delisted instruments itself, e.g.
            :class:`~alphaforge.data.sources.polygon_source.PolygonEquitiesSource`):
            the HISTORY-ONLY merge is then skipped and the live listing is upserted
            verbatim. Defaults to ``None``.
        store: The SCD2 instrument store both sources are upserted into.
    """

    def __init__(
        self,
        live_source: DataSource,
        vision: BinanceVisionClient | None = None,
        store: InstrumentStore | None = None,
    ) -> None:
        if store is None:
            raise ValueError("InstrumentSeeder requires a store")
        self._live_source = live_source
        self._vision = vision
        self._store = store

    def seed(self, *, as_of: Ms) -> SeedReport:
        """Seed/refresh the instrument store as observed at ``as_of`` (epoch ms UTC).

        Steps:

        1. Upsert every instrument from the live listing (SCD2-versioned at
           ``as_of``) — authoritative metadata for currently listed markets.
        2. **Only when an archive (``vision``) is configured:** list the archive's
           full historical symbol catalog; every symbol **not** in the live listing
           is delisted or never-ccxt-visible. For each, derive its lifetime from
           archive coverage and upsert a HISTORY-ONLY instrument (module docstring:
           placeholder filters, real default fees, conservative
           ``listed_ts``/``delisted_ts``). Symbols with no archived 1h months or an
           unparseable quote are skipped with a structlog warning — they carry no
           backtestable history. When ``vision`` is ``None`` (the live source is
           already survivorship-free, e.g. the Polygon equities reference adapter
           which returns delisted tickers itself) this step is skipped and the
           report's ``vision_only_count``/``delisted`` are 0.

        Idempotent: re-running with unchanged live + archive state writes zero new
        SCD2 rows. Returns a :class:`SeedReport`.
        """
        live = self._live_source.list_instruments(as_of=as_of)
        for inst in live:
            self._store.upsert(inst, as_of)

        if self._vision is None:
            report = SeedReport(
                live_count=len(live),
                vision_only_count=0,
                delisted=0,
                examples=(),
            )
            _log.info(
                "seed.completed",
                as_of=as_of,
                live_count=report.live_count,
                vision_only_count=report.vision_only_count,
                delisted=report.delisted,
                examples=list(report.examples),
            )
            return report

        vision = self._vision  # narrowed to non-None by the guard above
        live_symbols = {
            SymbolMapper.parse_instrument_id(inst.instrument_id)[2]
            for inst in live
            if inst.market_type is MarketType.PERP
        }
        vision_only = sorted(set(vision.list_symbols()) - live_symbols)

        recorded: list[str] = []
        for symbol in vision_only:
            hist_inst = self._history_only_instrument(vision, symbol)
            if hist_inst is None:
                continue
            self._store.upsert(hist_inst, as_of)
            recorded.append(hist_inst.instrument_id)

        report = SeedReport(
            live_count=len(live),
            vision_only_count=len(vision_only),
            delisted=len(recorded),
            examples=tuple(recorded[:_MAX_EXAMPLES]),
        )
        _log.info(
            "seed.completed",
            as_of=as_of,
            live_count=report.live_count,
            vision_only_count=report.vision_only_count,
            delisted=report.delisted,
            examples=list(report.examples),
        )
        return report

    def _history_only_instrument(
        self, vision: BinanceVisionClient, symbol: str
    ) -> Instrument | None:
        """Build the HISTORY-ONLY :class:`Instrument` for an archive-only symbol.

        Returns ``None`` (after a structlog warning) when the symbol has no archived
        1h months — nothing to backtest — or its quote cannot be parsed; both cases
        keep one bad symbol from aborting the whole seed run while staying visible
        in logs and in the ``vision_only_count - delisted`` gap of the report.
        """
        try:
            base, quote = SymbolMapper.split_exchange_symbol(symbol)
        except SchemaError:
            _log.warning("seed.skip_symbol", symbol=symbol, reason="unparseable_quote")
            return None
        coverage = vision.month_range(symbol)
        if coverage is None:
            _log.warning("seed.skip_symbol", symbol=symbol, reason="no_archived_months")
            return None
        first, last = coverage
        return Instrument(
            instrument_id=SymbolMapper.to_instrument_id("BINANCE", MarketType.PERP, symbol),
            asset_class=AssetClass.CRYPTO_PERP,
            market_type=MarketType.PERP,
            base=base,
            quote=quote,
            tick_size=HISTORY_ONLY_EPSILON,
            lot_size=HISTORY_ONLY_EPSILON,
            min_qty=HISTORY_ONLY_EPSILON,
            min_notional=0.0,
            can_short=True,
            maker_fee_bps=_HISTORY_ONLY_MAKER_FEE_BPS,
            taker_fee_bps=_HISTORY_ONLY_TAKER_FEE_BPS,
            funding_interval_hours=_HISTORY_ONLY_FUNDING_INTERVAL_HOURS,
            listed_ts=first.first_ms(),
            delisted_ts=last.next_month_first_ms(),
        )
