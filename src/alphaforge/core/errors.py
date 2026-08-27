"""AlphaForge exception hierarchy.

Every error raised by alphaforge code derives from :class:`AlphaForgeError` so callers
can catch the whole family with one clause while letting genuine programming errors
(``TypeError``, ``KeyError``, ...) propagate unchanged.
"""

from __future__ import annotations

__all__ = [
    "AlphaForgeError",
    "ConfigError",
    "CostModelMisuse",
    "DataGapError",
    "FillUnavailableError",
    "LockHeldError",
    "LookaheadError",
    "NaiveDatetimeError",
    "NotArmedError",
    "ReconciliationError",
    "RiskLimitError",
    "SchemaError",
]


class AlphaForgeError(Exception):
    """Base class for all AlphaForge errors; never raised directly."""


class NaiveDatetimeError(AlphaForgeError):
    """Raised when a naive or non-UTC datetime crosses an API boundary that requires UTC."""


class LookaheadError(AlphaForgeError):
    """Raised when data with ``available_at`` later than the decision timestamp is requested."""


class SchemaError(AlphaForgeError):
    """Raised when a table/record fails validation against its declared pyarrow schema."""


class ConfigError(AlphaForgeError):
    """Raised when settings are missing, malformed, or mutually inconsistent at load time."""


class CostModelMisuse(AlphaForgeError):
    """Raised when the transaction-cost model is evaluated outside its valid regime.

    Two situations qualify: (a) an order's quote notional exceeds 5% of the
    instrument's ADV, where the square-root impact law stops being trustworthy
    (it *underestimates* cost exactly when cost matters most) — pre-trade checks
    reject at 1% of ADV upstream (execDesign §7.1), so reaching 5% here means a
    layer above is broken and the run must stop, not extrapolate; (b) a
    non-finite or non-positive input (notional, ADV, reference price,
    volatility) reached a cost computation, which would silently poison every
    downstream P&L figure if let through.
    """


class DataGapError(AlphaForgeError):
    """Raised when expected bars are absent from the lake and the caller declared no fill policy."""


class FillUnavailableError(AlphaForgeError):
    """Raised when a fill model cannot execute any valid quantity without guessing liquidity."""


class LockHeldError(AlphaForgeError):
    """Raised when another process holds the exclusive lock on a single-writer resource.

    Single-writer jobs (ingestion today; any future state-mutating job) take an OS-level
    ``flock`` before touching shared state, making concurrent writers mechanically
    impossible rather than merely discouraged. The holder's pid is recorded in the lock
    file and included in the message.
    """


class ReconciliationError(AlphaForgeError):
    """Raised when local order/position/balance state disagrees with the broker beyond tolerance."""


class RiskLimitError(AlphaForgeError):
    """Raised when a pre-trade check or portfolio monitor breaches a configured risk limit."""


class NotArmedError(AlphaForgeError):
    """Raised when a real-money broker action is attempted while the system is not armed.

    AlphaForge trades PAPER money in v1. The real-money
    :class:`~alphaforge.execution.ccxt_broker.CCXTBroker` implements the full
    :class:`~alphaforge.execution.broker.Broker` interface but every
    order-mutating call (``submit``, ``cancel``) raises this error: arming real
    trading is a deliberate Phase-8+ step (an explicit config flip plus an arm
    action), never reachable by accident through ordinary code paths. The
    message names the exact gate so an operator who hits it understands it is a
    safety stop, not a bug.
    """
