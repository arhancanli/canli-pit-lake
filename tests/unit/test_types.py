"""Unit tests for alphaforge.core.types — enums and frozen value-type invariants."""

from __future__ import annotations

import dataclasses

import pytest

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

TS = 1_704_153_600_000  # 2024-01-02T00:00:00Z


def make_order(**overrides: object) -> OrderRequest:
    kwargs: dict[str, object] = {
        "client_order_id": "af-20240102T00-BTCUSDT-1",
        "instrument_id": "BINANCE:PERP:BTCUSDT",
        "side": Side.BUY,
        "qty": 0.5,
        "order_type": OrderType.MARKET,
        "decision_ts": TS,
        "decision_price": 42_000.0,
        "reason": "rebalance",
    }
    kwargs.update(overrides)
    return OrderRequest(**kwargs)  # type: ignore[arg-type]


def make_fill(**overrides: object) -> Fill:
    kwargs: dict[str, object] = {
        "client_order_id": "af-20240102T00-BTCUSDT-1",
        "instrument_id": "BINANCE:PERP:BTCUSDT",
        "side": Side.SELL,
        "qty": 0.5,
        "price": 42_010.0,
        "fee_quote": 10.5,
        "liquidity": Liquidity.TAKER,
        "ts": TS + 1_000,
    }
    kwargs.update(overrides)
    return Fill(**kwargs)  # type: ignore[arg-type]


class TestEnums:
    def test_side_sign(self) -> None:
        assert Side.BUY.sign == 1
        assert Side.SELL.sign == -1

    def test_side_sign_round_trips_signed_qty(self) -> None:
        assert Side.BUY.sign * 0.5 == 0.5
        assert Side.SELL.sign * 0.5 == -0.5

    def test_string_values(self) -> None:
        assert str(AssetClass.CRYPTO_PERP) == "crypto_perp"
        assert str(AssetClass.CRYPTO_SPOT) == "crypto_spot"
        assert str(AssetClass.EQUITY) == "equity"
        assert str(MarketType.PERP) == "perp"
        assert str(MarketType.SPOT) == "spot"
        assert str(Liquidity.MAKER) == "maker"
        assert str(Liquidity.TAKER) == "taker"
        assert str(OrderType.MARKET) == "market"
        assert str(OrderType.LIMIT) == "limit"

    def test_order_status_members(self) -> None:
        assert [s.value for s in OrderStatus] == [
            "new",
            "submitted",
            "partial",
            "filled",
            "canceled",
            "rejected",
        ]


class TestOrderRequest:
    def test_valid_construction(self) -> None:
        order = make_order()
        assert order.qty == 0.5
        assert order.reduce_only is False  # default
        assert order.decision_ts == TS

    def test_qty_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="qty"):
            make_order(qty=0.0)

    def test_qty_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="qty"):
            make_order(qty=-0.5)

    def test_qty_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="qty"):
            make_order(qty=float("nan"))

    def test_qty_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="qty"):
            make_order(qty=float("inf"))

    def test_frozen(self) -> None:
        order = make_order()
        with pytest.raises(dataclasses.FrozenInstanceError):
            order.qty = 1.0  # type: ignore[misc]

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            OrderRequest("id", "BINANCE:PERP:BTCUSDT")  # type: ignore[call-arg, misc]

    def test_equality_by_value(self) -> None:
        assert make_order() == make_order()
        assert make_order(qty=0.6) != make_order()


class TestFill:
    def test_valid_construction(self) -> None:
        fill = make_fill()
        assert fill.side.sign * fill.qty == -0.5
        assert fill.liquidity is Liquidity.TAKER

    def test_qty_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="qty"):
            make_fill(qty=0.0)
        with pytest.raises(ValueError, match="qty"):
            make_fill(qty=-1.0)

    def test_price_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="price"):
            make_fill(price=0.0)

    def test_fee_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="fee_quote"):
            make_fill(fee_quote=-0.01)

    def test_fee_zero_allowed(self) -> None:
        assert make_fill(fee_quote=0.0).fee_quote == 0.0

    def test_frozen(self) -> None:
        fill = make_fill()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fill.price = 1.0  # type: ignore[misc]


class TestPosition:
    def test_long_and_short_signed_qty(self) -> None:
        long = Position(
            instrument_id="BINANCE:PERP:BTCUSDT",
            qty=0.5,
            avg_entry_price=42_000.0,
            opened_ts=TS,
        )
        short = Position(
            instrument_id="BINANCE:PERP:ETHUSDT",
            qty=-2.0,
            avg_entry_price=2_500.0,
            opened_ts=TS,
        )
        assert long.qty > 0
        assert short.qty < 0

    def test_frozen(self) -> None:
        pos = Position(
            instrument_id="BINANCE:PERP:BTCUSDT",
            qty=0.5,
            avg_entry_price=42_000.0,
            opened_ts=TS,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pos.qty = 0.0  # type: ignore[misc]


class TestAccountState:
    def test_construction_with_positions_tuple(self) -> None:
        pos = Position(
            instrument_id="BINANCE:PERP:BTCUSDT",
            qty=0.5,
            avg_entry_price=42_000.0,
            opened_ts=TS,
        )
        state = AccountState(
            equity_quote=10_500.0,
            cash_quote=-10_500.0,
            positions=(pos,),
            ts=TS,
        )
        assert state.positions == (pos,)
        assert state.ts == TS

    def test_empty_positions(self) -> None:
        state = AccountState(equity_quote=10_000.0, cash_quote=10_000.0, positions=(), ts=TS)
        assert state.positions == ()

    def test_frozen(self) -> None:
        state = AccountState(equity_quote=1.0, cash_quote=1.0, positions=(), ts=TS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.equity_quote = 2.0  # type: ignore[misc]
