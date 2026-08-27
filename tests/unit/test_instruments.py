"""Unit tests for alphaforge.core.instruments — Instrument invariants and SCD2 store."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from alphaforge.core.errors import SchemaError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.types import AssetClass, MarketType

T1 = 1_704_153_600_000  # 2024-01-02T00:00:00Z
T2 = 1_706_745_600_000  # 2024-02-01T00:00:00Z
T3 = 1_709_251_200_000  # 2024-03-01T00:00:00Z
LISTED = 1_577_836_800_000  # 2020-01-01T00:00:00Z


def make_perp(**overrides: object) -> Instrument:
    kwargs: dict[str, object] = {
        "instrument_id": "BINANCE:PERP:BTCUSDT",
        "asset_class": AssetClass.CRYPTO_PERP,
        "market_type": MarketType.PERP,
        "base": "BTC",
        "quote": "USDT",
        "tick_size": 0.1,
        "lot_size": 0.001,
        "min_qty": 0.001,
        "min_notional": 100.0,
        "contract_multiplier": 1.0,
        "can_short": True,
        "maker_fee_bps": 2.0,
        "taker_fee_bps": 5.0,
        "funding_interval_hours": 8,
        "listed_ts": LISTED,
        "delisted_ts": None,
    }
    kwargs.update(overrides)
    return Instrument(**kwargs)  # type: ignore[arg-type]


def make_spot(**overrides: object) -> Instrument:
    kwargs: dict[str, object] = {
        "instrument_id": "BINANCE:SPOT:ETHUSDT",
        "asset_class": AssetClass.CRYPTO_SPOT,
        "market_type": MarketType.SPOT,
        "base": "ETH",
        "quote": "USDT",
        "tick_size": 0.01,
        "lot_size": 0.0001,
        "min_qty": 0.0001,
        "min_notional": 5.0,
        "can_short": False,
        "maker_fee_bps": 10.0,
        "taker_fee_bps": 10.0,
        "funding_interval_hours": None,
        "listed_ts": LISTED,
        "delisted_ts": None,
    }
    kwargs.update(overrides)
    return Instrument(**kwargs)  # type: ignore[arg-type]


class TestInstrumentInvariants:
    def test_valid_construction(self) -> None:
        perp = make_perp()
        assert perp.contract_multiplier == 1.0
        assert perp.funding_interval_hours == 8
        spot = make_spot()
        assert spot.funding_interval_hours is None

    def test_frozen_and_slotted(self) -> None:
        perp = make_perp()
        with pytest.raises(dataclasses.FrozenInstanceError):
            perp.tick_size = 1.0  # type: ignore[misc]
        assert not hasattr(perp, "__dict__")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tick_size", 0.0),
            ("tick_size", -0.1),
            ("tick_size", float("nan")),
            ("lot_size", 0.0),
            ("min_qty", -1.0),
            ("min_notional", -5.0),
            ("min_notional", float("inf")),
            ("contract_multiplier", 0.0),
            ("maker_fee_bps", -1.0),
            ("taker_fee_bps", float("nan")),
        ],
    )
    def test_numeric_invariants(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            make_perp(**{field: value})

    def test_perp_requires_funding_interval(self) -> None:
        with pytest.raises(ValueError, match="funding_interval_hours"):
            make_perp(funding_interval_hours=None)
        with pytest.raises(ValueError, match="funding_interval_hours"):
            make_perp(funding_interval_hours=0)

    def test_spot_forbids_funding_interval(self) -> None:
        with pytest.raises(ValueError, match="funding_interval_hours"):
            make_spot(funding_interval_hours=8)

    def test_market_type_must_match_id(self) -> None:
        with pytest.raises(ValueError, match="market type"):
            make_perp(instrument_id="BINANCE:SPOT:BTCUSDT")

    def test_malformed_id_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            make_perp(instrument_id="BTCUSDT")

    def test_base_quote_must_reconstruct_symbol(self) -> None:
        with pytest.raises(ValueError, match="reconstruct"):
            make_perp(base="ETH")
        with pytest.raises(ValueError, match="non-empty"):
            make_perp(base="")

    def test_delisted_must_follow_listed(self) -> None:
        with pytest.raises(ValueError, match="delisted_ts"):
            make_perp(delisted_ts=LISTED)
        inst = make_perp(delisted_ts=LISTED + 1)
        assert inst.delisted_ts == LISTED + 1


@pytest.fixture
def store(tmp_path: Path) -> InstrumentStore:
    return InstrumentStore(tmp_path / "instruments.sqlite")


class TestInstrumentStoreScd2:
    def test_first_upsert_then_get(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        assert store.upsert(v1, T1) is True
        assert store.get("BINANCE:PERP:BTCUSDT", T1) == v1
        assert store.get("BINANCE:PERP:BTCUSDT", T1 + 1) == v1

    def test_get_before_first_observation_is_none(self, store: InstrumentStore) -> None:
        store.upsert(make_perp(), T1)
        assert store.get("BINANCE:PERP:BTCUSDT", T1 - 1) is None

    def test_get_unknown_id_is_none(self, store: InstrumentStore) -> None:
        assert store.get("BINANCE:PERP:NOPEUSDT", T1) is None

    def test_idempotent_reupsert_no_new_row(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        store.upsert(v1, T1)
        assert store.upsert(v1, T2) is False
        history = store.history(v1.instrument_id)
        assert len(history) == 1
        assert history[0] == (T1, None, v1)

    def test_attribute_change_creates_second_version(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        v2 = make_perp(taker_fee_bps=4.0)
        store.upsert(v1, T1)
        assert store.upsert(v2, T2) is True
        history = store.history(v1.instrument_id)
        assert history == [(T1, T2, v1), (T2, None, v2)]

    def test_as_of_boundaries_inclusive_exclusive(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        v2 = make_perp(taker_fee_bps=4.0)
        store.upsert(v1, T1)
        store.upsert(v2, T2)
        iid = v1.instrument_id
        assert store.get(iid, T1) == v1  # valid_from inclusive
        assert store.get(iid, T2 - 1) == v1  # valid_to exclusive
        assert store.get(iid, T2) == v2  # new valid_from inclusive
        assert store.get(iid, T3) == v2  # open-ended current row

    def test_three_versions_history_ordering(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        v2 = make_perp(taker_fee_bps=4.0)
        v3 = make_perp(taker_fee_bps=4.0, tick_size=0.5)
        store.upsert(v1, T1)
        store.upsert(v2, T2)
        store.upsert(v3, T3)
        history = store.history(v1.instrument_id)
        assert [h[0] for h in history] == [T1, T2, T3]
        assert [h[1] for h in history] == [T2, T3, None]
        assert [h[2] for h in history] == [v1, v2, v3]

    def test_upsert_not_after_current_raises(self, store: InstrumentStore) -> None:
        store.upsert(make_perp(), T2)
        with pytest.raises(ValueError, match="as_of"):
            store.upsert(make_perp(taker_fee_bps=4.0), T2)
        with pytest.raises(ValueError, match="as_of"):
            store.upsert(make_perp(taker_fee_bps=4.0), T1)
        # The failed upserts must not have mutated the store.
        assert store.history("BINANCE:PERP:BTCUSDT") == [(T2, None, make_perp())]

    def test_all_known_point_in_time(self, store: InstrumentStore) -> None:
        perp = make_perp()
        spot = make_spot()
        store.upsert(perp, T1)
        store.upsert(spot, T2)
        assert store.all_known(T1) == [perp]
        assert store.all_known(T2) == [perp, spot]  # ordered by instrument_id
        assert store.all_known(T1 - 1) == []

    def test_all_known_returns_version_valid_at_as_of(self, store: InstrumentStore) -> None:
        v1 = make_perp()
        v2 = make_perp(maker_fee_bps=1.0)
        store.upsert(v1, T1)
        store.upsert(v2, T2)
        assert store.all_known(T2 - 1) == [v1]
        assert store.all_known(T2) == [v2]

    def test_history_unknown_id_empty(self, store: InstrumentStore) -> None:
        assert store.history("BINANCE:PERP:NOPEUSDT") == []

    def test_none_fields_round_trip(self, store: InstrumentStore) -> None:
        spot = make_spot(delisted_ts=LISTED + 1_000)
        store.upsert(spot, T1)
        got = store.get(spot.instrument_id, T1)
        assert got == spot
        assert got is not None
        assert got.funding_interval_hours is None
        assert got.delisted_ts == LISTED + 1_000
        assert got.can_short is False

    def test_persists_across_reopen_and_context_manager(self, tmp_path: Path) -> None:
        db = tmp_path / "instruments.sqlite"
        v1 = make_perp()
        with InstrumentStore(db) as store:
            store.upsert(v1, T1)
        with InstrumentStore(db) as store:
            assert store.get(v1.instrument_id, T1) == v1
            assert store.history(v1.instrument_id) == [(T1, None, v1)]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db = tmp_path / "a" / "b" / "instruments.sqlite"
        with InstrumentStore(db) as store:
            store.upsert(make_perp(), T1)
        assert db.exists()
