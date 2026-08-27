"""Point-in-time market-status replay contract tests."""

from __future__ import annotations

import pytest

from alphaforge.core.errors import LookaheadError
from alphaforge.core.types import OrderRequest, OrderType, Side
from alphaforge.execution.market_status import (
    MarketStatus,
    MarketStatusEvent,
    StaticMarketStatusProvider,
    execution_block_reason,
)

IID = "BINANCE:PERP:BTCUSDT"


def _event(
    *,
    status: MarketStatus,
    start: int = 100,
    end: int = 200,
    available_at: int = 100,
    instrument_id: str | None = None,
) -> MarketStatusEvent:
    return MarketStatusEvent(
        venue="BINANCE",
        instrument_id=instrument_id,
        status=status,
        effective_from=start,
        effective_until=end,
        observed_ts=100,
        available_at=available_at,
        reason="fixture",
    )


def _order(*, reduce_only: bool = False) -> OrderRequest:
    return OrderRequest(
        client_order_id="x",
        instrument_id=IID,
        side=Side.BUY,
        qty=1.0,
        order_type=OrderType.MARKET,
        reduce_only=reduce_only,
        decision_ts=150,
        decision_price=100.0,
        reason="test",
    )


def test_provider_rejects_overlapping_intervals_per_scope() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        StaticMarketStatusProvider(
            events=(
                _event(status=MarketStatus.OPEN, start=100, end=180),
                _event(status=MarketStatus.OUTAGE, start=170, end=200),
            )
        )


def test_future_known_status_raises_instead_of_leaking() -> None:
    provider = StaticMarketStatusProvider(
        events=(_event(status=MarketStatus.HALTED, available_at=160),)
    )
    with pytest.raises(LookaheadError, match="market status"):
        provider.status(IID, as_of=150)


def test_preflight_requires_complete_interval_even_without_status_queries() -> None:
    provider = StaticMarketStatusProvider(
        events=(_event(status=MarketStatus.OPEN, start=120, end=180, available_at=120),)
    )
    with pytest.raises(ValueError, match=r"coverage.*\[100, 120\)"):
        provider.require_coverage((IID,), start=100, end=200)


def test_preflight_passes_adjacent_explicit_intervals() -> None:
    provider = StaticMarketStatusProvider(
        events=(
            _event(status=MarketStatus.OPEN, start=100, end=150, available_at=100),
            _event(status=MarketStatus.OUTAGE, start=150, end=200, available_at=150),
        )
    )
    assert provider.require_coverage((IID,), start=100, end=200) is None


def test_preflight_does_not_hide_future_known_specific_status_with_venue_open() -> None:
    provider = StaticMarketStatusProvider(
        events=(
            _event(status=MarketStatus.OPEN, start=100, end=200, available_at=100),
            _event(
                status=MarketStatus.HALTED,
                start=120,
                end=180,
                available_at=140,
                instrument_id=IID,
            ),
        )
    )
    with pytest.raises(LookaheadError, match="market status"):
        provider.require_coverage((IID,), start=100, end=200)


def test_instrument_status_overrides_venue_status() -> None:
    provider = StaticMarketStatusProvider(
        events=(
            _event(status=MarketStatus.OPEN),
            _event(status=MarketStatus.HALTED, instrument_id=IID),
        )
    )
    assert provider.status(IID, as_of=150).status is MarketStatus.HALTED  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("status", "reduce_only", "reason"),
    [
        (MarketStatus.OPEN, False, None),
        (MarketStatus.HALTED, True, "market_halted"),
        (MarketStatus.OUTAGE, True, "venue_outage"),
        (MarketStatus.AUCTION_ONLY, True, "auction_only_no_continuous_fill"),
        (MarketStatus.CLOSE_ONLY, False, "close_only_blocks_risk_increase"),
        (MarketStatus.CLOSE_ONLY, True, None),
    ],
)
def test_execution_policy(status: MarketStatus, reduce_only: bool, reason: str | None) -> None:
    assert execution_block_reason(_event(status=status), _order(reduce_only=reduce_only)) == reason
