"""Unit tests for alphaforge.data.universe.store (UniverseStore).

Covers: schema validation before any disk mutation, full-replace vs merge-upsert
write semantics, knowability filtering in ``read_intervals(as_of=...)``,
``membership_asof`` half-open boundary semantics ``[effective_from, effective_to)``,
multi-year partition assembly, and schema-stable empty reads. All lakes live under
``tmp_path`` — never the real ./data.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.core.errors import SchemaError
from alphaforge.data.schemas import UNIVERSE_SCHEMA
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.universe.store import UniverseStore

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"
FTT = "BINANCE:PERP:FTTUSDT"

DAY = 86_400_000
T_2023_JUL = 1_688_169_600_000  # 2023-07-01T00:00:00Z
T_2024_FEB = 1_706_745_600_000  # 2024-02-01T00:00:00Z
T_2024_MAR = 1_709_251_200_000  # 2024-03-01T00:00:00Z

_TS = pa.timestamp("ms", tz="UTC")


def intervals(*rows: tuple[str, int, int | None, int, str]) -> pa.Table:
    """Rows of ``(instrument_id, effective_from, effective_to, rank, reason)``."""
    return pa.table(
        {
            "instrument_id": pa.array([r[0] for r in rows], type=pa.string()),
            "effective_from": pa.array([r[1] for r in rows], type=_TS),
            "effective_to": pa.array([r[2] for r in rows], type=_TS),
            "rank": pa.array([r[3] for r in rows], type=pa.int32()),
            "reason": pa.array([r[4] for r in rows], type=pa.string()),
        }
    )


def rows_of(tbl: pa.Table) -> list[tuple[str, int, int | None, int, str]]:
    """Table → comparable row tuples (timestamps as epoch ms ints)."""
    ids = tbl.column("instrument_id").to_pylist()
    froms = tbl.column("effective_from").cast(pa.int64()).to_pylist()
    tos = tbl.column("effective_to").cast(pa.int64()).to_pylist()
    ranks = tbl.column("rank").to_pylist()
    reasons = tbl.column("reason").to_pylist()
    return list(zip(ids, froms, tos, ranks, reasons, strict=True))


@pytest.fixture
def store(tmp_path: Path) -> UniverseStore:
    return UniverseStore(LakePaths(tmp_path / "lake"))


class TestWriteRead:
    def test_roundtrip_sorted_and_schema_conformant(self, store: UniverseStore) -> None:
        tbl = intervals(
            (ETH, T_2024_FEB, None, 2, "enter_top20"),
            (BTC, T_2024_FEB, None, 1, "enter_top20"),
        )
        stats = store.write_intervals(tbl)
        assert stats.rows_written == 2
        got = store.read_intervals()
        assert got.schema.names == list(UNIVERSE_SCHEMA.names)
        assert rows_of(got) == [
            (BTC, T_2024_FEB, None, 1, "enter_top20"),
            (ETH, T_2024_FEB, None, 2, "enter_top20"),
        ]

    def test_multi_year_partitions_assemble(self, store: UniverseStore, tmp_path: Path) -> None:
        tbl = intervals(
            (BTC, T_2023_JUL, T_2024_FEB, 1, "exit_rank_27"),
            (BTC, T_2024_MAR, None, 1, "enter_top20"),
        )
        store.write_intervals(tbl)
        leaves = sorted((tmp_path / "lake" / "universe_membership").rglob("data.parquet"))
        assert len(leaves) == 2  # year=2023 and year=2024
        assert [r[1] for r in rows_of(store.read_intervals())] == [T_2023_JUL, T_2024_MAR]

    def test_invalid_schema_rejected_before_touching_disk(self, store: UniverseStore) -> None:
        store.write_intervals(intervals((BTC, T_2024_FEB, None, 1, "enter_top20")))
        bad = intervals((ETH, T_2024_MAR, None, 1, "enter_top20")).append_column(
            "extra", pa.array([1], type=pa.int64())
        )
        with pytest.raises(SchemaError, match="extra"):
            store.write_intervals(bad)
        # replace=True must not have wiped the prior data on a failed validation
        assert rows_of(store.read_intervals()) == [(BTC, T_2024_FEB, None, 1, "enter_top20")]

    def test_replace_overwrites_prior_intervals(self, store: UniverseStore) -> None:
        store.write_intervals(
            intervals(
                (BTC, T_2023_JUL, None, 1, "enter_top20"),
                (ETH, T_2023_JUL, None, 2, "enter_top20"),
                (FTT, T_2023_JUL, None, 3, "enter_top20"),
            )
        )
        store.write_intervals(intervals((BTC, T_2024_FEB, None, 1, "enter_top20")))
        assert rows_of(store.read_intervals()) == [(BTC, T_2024_FEB, None, 1, "enter_top20")]

    def test_scoped_replace_preserves_other_sleeves(self, store: UniverseStore) -> None:
        """Per-sleeve rebuild: scope_ids clears only the named sleeve, not the lake.

        This is the single-tenant fix the unified cross-asset book depends on — a crypto
        universe rebuild must not wipe the equity universe (and vice versa).
        """
        eq_a = "XUSE:CASH:AAPLUSD"
        eq_b = "XUSE:CASH:MSFTUSD"
        # Both sleeves coexist after independent writes.
        store.write_intervals(
            intervals(
                (BTC, T_2023_JUL, None, 1, "enter_top20"),
                (ETH, T_2023_JUL, None, 2, "enter_top20"),
            )
        )
        store.write_intervals(
            intervals(
                (eq_a, T_2023_JUL, None, 1, "enter_top200"),
                (eq_b, T_2023_JUL, None, 2, "enter_top200"),
            ),
            scope_ids=frozenset({eq_a, eq_b}),
        )
        assert store.membership_asof(T_2023_JUL) == frozenset({BTC, ETH, eq_a, eq_b})
        # Rebuild ONLY crypto (scope = every crypto id, incl. a delisted one not present).
        store.write_intervals(
            intervals((BTC, T_2024_FEB, None, 1, "enter_top20")),
            scope_ids=frozenset({BTC, ETH, FTT}),
        )
        # Crypto replaced (ETH dropped, BTC re-dated); equity untouched.
        assert store.membership_asof(T_2024_FEB) == frozenset({BTC, eq_a, eq_b})
        assert rows_of(store.read_intervals()) == [
            (BTC, T_2024_FEB, None, 1, "enter_top20"),
            (eq_a, T_2023_JUL, None, 1, "enter_top200"),
            (eq_b, T_2023_JUL, None, 2, "enter_top200"),
        ]

    def test_merge_upsert_closes_open_interval_without_replace(self, store: UniverseStore) -> None:
        store.write_intervals(intervals((FTT, T_2023_JUL, None, 5, "enter_top20")))
        # Live loop observes the delisting: same natural key, effective_to now set.
        store.write_intervals(
            intervals((FTT, T_2023_JUL, T_2024_FEB, 5, "delisted")), replace=False
        )
        assert rows_of(store.read_intervals()) == [(FTT, T_2023_JUL, T_2024_FEB, 5, "delisted")]


class TestReads:
    def test_empty_lake_yields_schema_stable_empty_results(self, store: UniverseStore) -> None:
        got = store.read_intervals()
        assert got.num_rows == 0
        assert got.schema.names == list(UNIVERSE_SCHEMA.names)
        assert store.membership_asof(T_2024_FEB) == frozenset()

    def test_read_intervals_asof_applies_knowability_filter(self, store: UniverseStore) -> None:
        store.write_intervals(
            intervals(
                (BTC, T_2023_JUL, None, 1, "enter_top20"),
                (ETH, T_2024_MAR, None, 2, "enter_top20"),
            )
        )
        # An interval is knowable at its effective_from; ETH's span starts in March.
        assert [r[0] for r in rows_of(store.read_intervals(as_of=T_2024_FEB))] == [BTC]
        assert [r[0] for r in rows_of(store.read_intervals(as_of=T_2024_MAR))] == [BTC, ETH]

    def test_membership_asof_half_open_boundaries(self, store: UniverseStore) -> None:
        store.write_intervals(
            intervals(
                (FTT, T_2024_FEB, T_2024_MAR, 3, "exit_rank_27"),  # [Feb 1, Mar 1)
                (BTC, T_2024_FEB, None, 1, "enter_top20"),  # open
            )
        )
        assert store.membership_asof(T_2024_FEB - 1) == frozenset()
        assert store.membership_asof(T_2024_FEB) == frozenset({BTC, FTT})  # inclusive start
        assert store.membership_asof(T_2024_MAR - 1) == frozenset({BTC, FTT})
        assert store.membership_asof(T_2024_MAR) == frozenset({BTC})  # exclusive end
        assert store.membership_asof(T_2024_MAR + 365 * DAY) == frozenset({BTC})  # open = forever

    def test_delisted_symbol_remains_in_history(self, store: UniverseStore) -> None:
        """Survivorship gate: a dead symbol's interval is an ordinary, readable row."""
        store.write_intervals(
            intervals(
                (FTT, T_2023_JUL, T_2024_FEB, 4, "delisted"),
                (BTC, T_2023_JUL, None, 1, "enter_top20"),
            )
        )
        assert FTT in store.membership_asof(T_2023_JUL + 10 * DAY)
        assert FTT not in store.membership_asof(T_2024_FEB)
        assert (FTT, T_2023_JUL, T_2024_FEB, 4, "delisted") in rows_of(store.read_intervals())
