"""AlphaForge core kernel: time, types, errors, instruments, symbols, calendar.

Curated public surface for the Phase 1 kernel. Downstream modules should import
these names from :mod:`alphaforge.core` rather than reaching into submodules;
anything not re-exported here is internal and may change without notice.

Conventions (non-negotiable, enforced across the package):

* All timestamps are UTC. At API boundaries timestamps are integer epoch
  milliseconds (:data:`Ms`). Naive ``datetime`` objects are a programming error.
* Bars are labeled by OPEN time (``ts_open``); a bar of timeframe Δ becomes
  visible at ``available_at = ts_open + Δ``.
"""

from alphaforge.core.calendar import Always24x7Calendar, TradingCalendar, calendar_for
from alphaforge.core.errors import (
    AlphaForgeError,
    ConfigError,
    CostModelMisuse,
    DataGapError,
    LockHeldError,
    LookaheadError,
    NaiveDatetimeError,
    ReconciliationError,
    RiskLimitError,
    SchemaError,
)
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.symbols import KNOWN_QUOTES, SymbolMapper
from alphaforge.core.time import (
    Ms,
    Timeframe,
    available_at,
    bar_open_for_decision,
    expected_bar_opens,
    floor_bar,
    from_ms,
    next_bar_open,
    now_ms,
    parse_utc,
    require_utc,
    to_ms,
)
from alphaforge.core.types import (
    AccountState,
    AssetClass,
    Fill,
    Liquidity,
    MarketType,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Side,
)

__all__ = [
    "KNOWN_QUOTES",
    "AccountState",
    "AlphaForgeError",
    "Always24x7Calendar",
    "AssetClass",
    "ConfigError",
    "CostModelMisuse",
    "DataGapError",
    "Fill",
    "Instrument",
    "InstrumentStore",
    "Liquidity",
    "LockHeldError",
    "LookaheadError",
    "MarketType",
    "Ms",
    "NaiveDatetimeError",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "ReconciliationError",
    "RiskLimitError",
    "SchemaError",
    "Side",
    "SymbolMapper",
    "Timeframe",
    "TradingCalendar",
    "available_at",
    "bar_open_for_decision",
    "calendar_for",
    "expected_bar_opens",
    "floor_bar",
    "from_ms",
    "next_bar_open",
    "now_ms",
    "parse_utc",
    "require_utc",
    "to_ms",
]
