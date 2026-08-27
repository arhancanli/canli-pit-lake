"""Shared order/fill/position value types and enums (execDesign.md §2, re-based on epoch-ms).

All timestamps are :data:`~alphaforge.core.time.Ms` (UTC epoch milliseconds) per
buildabilityCritique.md ruling 3.1. All monetary amounts are quote-currency units
(USDT in v1). Instruments are referenced by canonical ``instrument_id``
(e.g. ``"BINANCE:PERP:BTCUSDT"``), never by raw exchange symbol.

Dataclasses are frozen, slotted, and keyword-only: construction sites must name every
field, which prevents silent positional drift when fields are added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from alphaforge.core.time import Ms

__all__ = [
    "AccountState",
    "AssetClass",
    "Fill",
    "Liquidity",
    "MarketType",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "Side",
]


class AssetClass(StrEnum):
    """Top-level asset taxonomy; drives calendar, funding, and shortability defaults."""

    CRYPTO_PERP = "crypto_perp"
    CRYPTO_SPOT = "crypto_spot"
    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"


class MarketType(StrEnum):
    """Venue market segment; matches the lake partition value (``market=perp|spot|cash``)."""

    PERP = "perp"
    SPOT = "spot"
    CASH = "cash"
    FUTURE = "future"
    OPTION = "option"


class Side(StrEnum):
    """Order direction. Quantities are unsigned; ``side`` carries the sign."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — multiply by unsigned qty to get signed exposure."""
        return 1 if self is Side.BUY else -1


class Liquidity(StrEnum):
    """Whether a fill added (maker) or removed (taker) book liquidity; selects the fee tier."""

    MAKER = "maker"
    TAKER = "taker"


class OrderType(StrEnum):
    """Supported order types in v1."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    """Order lifecycle ladder; transitions are monotone NEW → SUBMITTED → terminal."""

    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


def _require_positive_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and strictly positive."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderRequest:
    """An intent to trade, produced at decision time and keyed for idempotent submission.

    ``qty`` is unsigned base-asset units (``side`` carries the sign) and must be > 0.
    ``decision_ts`` is the decision timestamp (a bar close, :data:`Ms`); ``decision_price``
    is the price used for sizing, kept for price-collar checks and slippage attribution.
    ``client_order_id`` is the idempotency key — deterministic per (cycle, instrument)
    so crash-replay never double-submits.
    """

    client_order_id: str
    instrument_id: str
    side: Side
    qty: float
    order_type: OrderType
    reduce_only: bool = False
    decision_ts: Ms
    decision_price: float
    reason: str

    def __post_init__(self) -> None:
        _require_positive_finite("OrderRequest.qty", self.qty)


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    """An execution (or partial execution) reported by a broker.

    ``qty`` is unsigned (> 0; ``side`` carries the sign), ``price`` is the execution
    price (> 0), ``fee_quote`` is the fee in quote units (>= 0), and ``ts`` is the
    execution timestamp (:data:`Ms`).
    """

    client_order_id: str
    instrument_id: str
    side: Side
    qty: float
    price: float
    fee_quote: float
    liquidity: Liquidity
    ts: Ms

    def __post_init__(self) -> None:
        _require_positive_finite("Fill.qty", self.qty)
        _require_positive_finite("Fill.price", self.price)
        if not math.isfinite(self.fee_quote) or self.fee_quote < 0.0:
            raise ValueError(f"Fill.fee_quote must be finite and >= 0, got {self.fee_quote!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    """A net position in one instrument.

    ``qty`` is SIGNED base-asset units (+ long / - short). ``avg_entry_price`` is the
    VWAP of opening fills (resets when the position goes flat or flips sign).
    ``opened_ts`` (:data:`Ms`) is when the current same-sign position was opened.
    """

    instrument_id: str
    qty: float
    avg_entry_price: float
    opened_ts: Ms


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountState:
    """A point-in-time account snapshot in quote units.

    ``equity_quote`` = cash + marked-to-market position value; ``cash_quote`` is free
    balance; ``positions`` is the full set of open positions; ``ts`` (:data:`Ms`) is
    the snapshot timestamp.
    """

    equity_quote: float
    cash_quote: float
    positions: tuple[Position, ...]
    ts: Ms
