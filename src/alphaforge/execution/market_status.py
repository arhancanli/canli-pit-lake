"""Point-in-time venue/instrument market-status and execution-blocking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from alphaforge.core.errors import LookaheadError
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms
from alphaforge.core.types import OrderRequest

__all__ = [
    "MarketStatus",
    "MarketStatusEvent",
    "MarketStatusProvider",
    "StaticMarketStatusProvider",
    "execution_block_reason",
]


class MarketStatus(StrEnum):
    OPEN = "open"
    HALTED = "halted"
    OUTAGE = "outage"
    AUCTION_ONLY = "auction_only"
    CLOSE_ONLY = "close_only"


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusEvent:
    """One explicit status interval with availability lineage and optional instrument scope."""

    venue: str
    instrument_id: str | None
    status: MarketStatus
    effective_from: Ms
    effective_until: Ms
    observed_ts: Ms
    available_at: Ms
    reason: str

    def __post_init__(self) -> None:
        if not self.venue or self.venue != self.venue.upper():
            raise ValueError("venue must be non-empty uppercase text")
        if self.instrument_id is not None:
            venue, _, _ = SymbolMapper.parse_instrument_id(self.instrument_id)
            if venue != self.venue:
                raise ValueError("instrument venue does not match event venue")
        if self.effective_from >= self.effective_until:
            raise ValueError("effective_from must precede effective_until")
        if self.available_at < self.observed_ts:
            raise ValueError("available_at cannot precede observed_ts")
        if not self.reason:
            raise ValueError("market status reason cannot be empty")

    @property
    def scope_key(self) -> tuple[str, str | None]:
        return self.venue, self.instrument_id


class MarketStatusProvider(Protocol):
    def require_coverage(
        self,
        instrument_ids: tuple[str, ...],
        *,
        start: Ms,
        end: Ms,
    ) -> str | None:
        """Fail closed unless every instrument is explicitly covered over ``[start, end)``.

        Source-reviewed providers return a deterministic evidence hash. Synthetic providers may
        return ``None`` but must still prove interval completeness.
        """
        ...

    def status(self, instrument_id: str, *, as_of: Ms) -> MarketStatusEvent | None:
        """Return the explicit effective status at ``as_of``, or ``None`` for no coverage."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class StaticMarketStatusProvider:
    """Deterministic replay provider requiring non-overlapping intervals per scope."""

    events: tuple[MarketStatusEvent, ...]

    def __post_init__(self) -> None:
        by_scope: dict[tuple[str, str | None], list[MarketStatusEvent]] = {}
        for event in self.events:
            by_scope.setdefault(event.scope_key, []).append(event)
        for scope, rows in by_scope.items():
            ordered = sorted(rows, key=lambda row: row.effective_from)
            if any(
                current.effective_until > following.effective_from
                for current, following in pairwise(ordered)
            ):
                raise ValueError(f"overlapping market-status intervals for scope {scope!r}")

    def status(self, instrument_id: str, *, as_of: Ms) -> MarketStatusEvent | None:
        venue, _, _ = SymbolMapper.parse_instrument_id(instrument_id)
        effective = [
            event
            for event in self.events
            if event.venue == venue
            and event.effective_from <= as_of < event.effective_until
            and event.instrument_id in (None, instrument_id)
        ]
        if not effective:
            return None
        specific = [event for event in effective if event.instrument_id == instrument_id]
        selected = specific[0] if specific else effective[0]
        if selected.available_at > as_of:
            raise LookaheadError(
                f"market status for {instrument_id!r} available at {selected.available_at} "
                f"exceeds decision {as_of}"
            )
        return selected

    def require_coverage(
        self,
        instrument_ids: tuple[str, ...],
        *,
        start: Ms,
        end: Ms,
    ) -> str | None:
        """Require explicit status for every millisecond in each requested interval.

        Effective and availability boundaries are swept rather than sampled on a bar grid. A
        future-known instrument event therefore cannot be hidden by a venue-wide OPEN interval.
        """
        if not instrument_ids:
            raise ValueError("market-status preflight requires at least one instrument")
        if start < 0 or end <= start:
            raise ValueError("market-status preflight requires 0 <= start < end")
        for instrument_id in dict.fromkeys(instrument_ids):
            venue, _, _ = SymbolMapper.parse_instrument_id(instrument_id)
            applicable = tuple(
                event
                for event in self.events
                if event.venue == venue
                and event.instrument_id in (None, instrument_id)
                and event.effective_from < end
                and event.effective_until > start
            )
            boundaries = {start, end}
            for event in applicable:
                boundaries.add(max(start, event.effective_from))
                boundaries.add(min(end, event.effective_until))
                if start < event.available_at < end:
                    boundaries.add(event.available_at)
            for effective_from, effective_until in pairwise(sorted(boundaries)):
                if self.status(instrument_id, as_of=effective_from) is None:
                    raise ValueError(
                        f"missing PIT market-status coverage for {instrument_id!r} over "
                        f"[{effective_from}, {effective_until})"
                    )
        return None


def execution_block_reason(event: MarketStatusEvent, order: OrderRequest) -> str | None:
    """Return a stable block reason, or ``None`` when the order may execute."""
    if event.status is MarketStatus.OPEN:
        return None
    if event.status is MarketStatus.CLOSE_ONLY and order.reduce_only:
        return None
    return {
        MarketStatus.HALTED: "market_halted",
        MarketStatus.OUTAGE: "venue_outage",
        MarketStatus.AUCTION_ONLY: "auction_only_no_continuous_fill",
        MarketStatus.CLOSE_ONLY: "close_only_blocks_risk_increase",
        MarketStatus.OPEN: "",
    }[event.status]
