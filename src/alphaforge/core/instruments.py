"""THE merged Instrument model and its SCD2 store (buildabilityCritique.md ruling 3.2).

One :class:`Instrument` unifies the data-design identity scheme (canonical
``instrument_id``, fee bps, SCD2 validity) with the execution-design trading fields
(``can_short``, ``min_qty``). One name per concept: ``tick_size`` (min price increment,
quote units) and ``lot_size`` (min quantity step, base units) — ``price_step``/``qty_step``
do not exist in this codebase.

:class:`InstrumentStore` persists instrument versions as a type-2 slowly-changing
dimension over SQLite: each attribute change observed at ``as_of`` closes the current
row (``valid_to_ms = as_of``, exclusive) and opens a new one (``valid_from_ms = as_of``,
inclusive). ``valid_from_ms`` is *observation* time (epoch ms, UTC) — we can only know
a fee/filter change when we see it (dataDesign.md §3 PIT table). Point-in-time reads
(:meth:`InstrumentStore.get`, :meth:`InstrumentStore.all_known`) select the row with
``valid_from_ms <= as_of < coalesce(valid_to_ms, +inf)``.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass, fields
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms
from alphaforge.core.types import AssetClass, MarketType

__all__ = [
    "Instrument",
    "InstrumentStore",
]


def _require_positive_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and strictly positive."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _require_nonnegative_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and >= 0."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Instrument:
    """A tradeable instrument: identity, exchange filters, fees, lifecycle.

    Units: ``tick_size`` quote units; ``lot_size``/``min_qty`` base units;
    ``min_notional`` quote units; fees in basis points of notional;
    ``listed_ts``/``delisted_ts`` epoch ms UTC (:data:`Ms`). ``contract_multiplier``
    is 1.0 for spot and linear perps (exists so index futures/options fit later).
    ``funding_interval_hours`` is required for perps (8 on Binance) and must be
    ``None`` for spot/equity.

    The ``instrument_id`` must be canonical (``"BINANCE:PERP:BTCUSDT"``) and its
    market segment must agree with ``market_type``; mismatches raise ``ValueError``,
    malformed ids raise :class:`SchemaError`.
    """

    instrument_id: str
    asset_class: AssetClass
    market_type: MarketType
    base: str
    quote: str
    tick_size: float
    lot_size: float
    min_qty: float
    min_notional: float
    contract_multiplier: float = 1.0
    can_short: bool
    maker_fee_bps: float
    taker_fee_bps: float
    funding_interval_hours: int | None
    listed_ts: Ms
    delisted_ts: Ms | None

    def __post_init__(self) -> None:
        _, id_market_type, id_symbol = SymbolMapper.parse_instrument_id(self.instrument_id)
        if id_market_type is not self.market_type:
            raise ValueError(
                f"instrument_id {self.instrument_id!r} encodes market type "
                f"{id_market_type!r} but market_type field is {self.market_type!r}"
            )
        expected_market = {
            AssetClass.CRYPTO_PERP: MarketType.PERP,
            AssetClass.CRYPTO_SPOT: MarketType.SPOT,
            AssetClass.EQUITY: MarketType.CASH,
            AssetClass.FUTURE: MarketType.FUTURE,
            AssetClass.OPTION: MarketType.OPTION,
        }[self.asset_class]
        if self.market_type is not expected_market:
            raise ValueError(
                f"asset_class {self.asset_class.value!r} requires market_type "
                f"{expected_market.value!r}, got {self.market_type.value!r}"
            )
        if not self.base or not self.quote:
            raise ValueError("Instrument.base and Instrument.quote must be non-empty")
        if f"{self.base}{self.quote}" != id_symbol:
            raise ValueError(
                f"base+quote {self.base!r}+{self.quote!r} does not reconstruct the "
                f"exchange symbol {id_symbol!r} of {self.instrument_id!r}"
            )
        _require_positive_finite("Instrument.tick_size", self.tick_size)
        _require_positive_finite("Instrument.lot_size", self.lot_size)
        _require_positive_finite("Instrument.min_qty", self.min_qty)
        _require_nonnegative_finite("Instrument.min_notional", self.min_notional)
        _require_positive_finite("Instrument.contract_multiplier", self.contract_multiplier)
        _require_nonnegative_finite("Instrument.maker_fee_bps", self.maker_fee_bps)
        _require_nonnegative_finite("Instrument.taker_fee_bps", self.taker_fee_bps)
        if self.market_type is MarketType.PERP:
            if self.funding_interval_hours is None or self.funding_interval_hours <= 0:
                raise ValueError(
                    "perp instruments require funding_interval_hours > 0, got "
                    f"{self.funding_interval_hours!r}"
                )
        elif self.funding_interval_hours is not None:
            raise ValueError(
                f"non-perp instruments must have funding_interval_hours=None, got "
                f"{self.funding_interval_hours!r}"
            )
        if self.delisted_ts is not None and self.delisted_ts <= self.listed_ts:
            raise ValueError(
                f"delisted_ts ({self.delisted_ts}) must be > listed_ts ({self.listed_ts})"
            )


#: Instrument attribute names in declared (and SQL column) order.
_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(Instrument))

_CREATE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS instruments_v (
    instrument_id          TEXT    NOT NULL,
    asset_class            TEXT    NOT NULL,
    market_type            TEXT    NOT NULL,
    base                   TEXT    NOT NULL,
    quote                  TEXT    NOT NULL,
    tick_size              REAL    NOT NULL,
    lot_size               REAL    NOT NULL,
    min_qty                REAL    NOT NULL,
    min_notional           REAL    NOT NULL,
    contract_multiplier    REAL    NOT NULL,
    can_short              INTEGER NOT NULL,
    maker_fee_bps          REAL    NOT NULL,
    taker_fee_bps          REAL    NOT NULL,
    funding_interval_hours INTEGER,
    listed_ts              INTEGER NOT NULL,
    delisted_ts            INTEGER,
    valid_from_ms          INTEGER NOT NULL,
    valid_to_ms            INTEGER,
    PRIMARY KEY (instrument_id, valid_from_ms)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_instruments_v_current
    ON instruments_v (instrument_id) WHERE valid_to_ms IS NULL;
"""

_INSERT_SQL: Final[str] = (
    f"INSERT INTO instruments_v ({', '.join(_FIELDS)}, valid_from_ms, valid_to_ms) "
    f"VALUES ({', '.join('?' * len(_FIELDS))}, ?, NULL)"
)

_SELECT_COLS: Final[str] = ", ".join(_FIELDS)


def _to_row(inst: Instrument) -> tuple[object, ...]:
    """Instrument → SQL parameter tuple in ``_FIELDS`` order (bool → int, enums → str)."""
    return (
        inst.instrument_id,
        inst.asset_class.value,
        inst.market_type.value,
        inst.base,
        inst.quote,
        inst.tick_size,
        inst.lot_size,
        inst.min_qty,
        inst.min_notional,
        inst.contract_multiplier,
        int(inst.can_short),
        inst.maker_fee_bps,
        inst.taker_fee_bps,
        inst.funding_interval_hours,
        inst.listed_ts,
        inst.delisted_ts,
    )


def _from_row(row: sqlite3.Row) -> Instrument:
    """SQL row → Instrument (revalidates invariants on the way out)."""
    funding = row["funding_interval_hours"]
    delisted = row["delisted_ts"]
    return Instrument(
        instrument_id=str(row["instrument_id"]),
        asset_class=AssetClass(str(row["asset_class"])),
        market_type=MarketType(str(row["market_type"])),
        base=str(row["base"]),
        quote=str(row["quote"]),
        tick_size=float(row["tick_size"]),
        lot_size=float(row["lot_size"]),
        min_qty=float(row["min_qty"]),
        min_notional=float(row["min_notional"]),
        contract_multiplier=float(row["contract_multiplier"]),
        can_short=bool(row["can_short"]),
        maker_fee_bps=float(row["maker_fee_bps"]),
        taker_fee_bps=float(row["taker_fee_bps"]),
        funding_interval_hours=None if funding is None else int(funding),
        listed_ts=int(row["listed_ts"]),
        delisted_ts=None if delisted is None else int(delisted),
    )


def _enable_wal(conn: sqlite3.Connection, *, budget_s: float = 5.0) -> None:
    """Switch ``conn``'s database to WAL journal mode, retrying briefly on contention.

    The DELETE→WAL upgrade needs exclusive access, and SQLite returns
    ``SQLITE_BUSY`` *without consulting the busy handler* in deadlock-prone lock
    upgrades — so stores racing to open the same file (the ops db is shared with
    ``CheckpointStore``) can see ``sqlite3.OperationalError: database is locked``
    despite ``busy_timeout``. WAL mode is persistent in the file, so the retry
    converges immediately once any opener succeeds. Re-raises after ``budget_s``
    seconds of failures.
    """
    deadline = time.monotonic() + budget_s
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
        else:
            return


class InstrumentStore:
    """SCD2 instrument-version store over a single SQLite file (WAL mode).

    Validity semantics: ``valid_from_ms`` inclusive, ``valid_to_ms`` exclusive,
    ``NULL`` = current. All SQL is parameterized; all writes are transactional.
    Usable as a context manager (closes the connection on exit).
    """

    def __init__(self, db_path: Path) -> None:
        """Open (creating parents and schema if absent) the store at ``db_path``."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        _enable_wal(self._conn)
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_CREATE_SQL)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection; the store is unusable afterwards."""
        self._conn.close()

    def upsert(self, inst: Instrument, as_of: Ms) -> bool:
        """Record ``inst`` as observed at ``as_of``; SCD2 versioning, idempotent.

        - No current row: insert with ``valid_from_ms = as_of``. Returns True.
        - Current row equal to ``inst`` (all attributes): no-op. Returns False.
        - Current row differs: close it (``valid_to_ms = as_of``) and insert the new
          version (``valid_from_ms = as_of``) atomically. Returns True.

        Raises ``ValueError`` if ``as_of <= valid_from_ms`` of a differing current row
        (versions must be observed in strictly increasing time; equality would create
        an empty interval).
        """
        cur = self._conn.execute(
            f"SELECT {_SELECT_COLS}, valid_from_ms FROM instruments_v "
            "WHERE instrument_id = ? AND valid_to_ms IS NULL",
            (inst.instrument_id,),
        )
        row = cur.fetchone()
        with self._conn:
            if row is None:
                self._conn.execute(_INSERT_SQL, (*_to_row(inst), as_of))
                return True
            if _from_row(row) == inst:
                return False
            current_from = int(row["valid_from_ms"])
            if as_of <= current_from:
                raise ValueError(
                    f"upsert as_of={as_of} must be > current valid_from_ms={current_from} "
                    f"for {inst.instrument_id!r}"
                )
            self._conn.execute(
                "UPDATE instruments_v SET valid_to_ms = ? "
                "WHERE instrument_id = ? AND valid_to_ms IS NULL",
                (as_of, inst.instrument_id),
            )
            self._conn.execute(_INSERT_SQL, (*_to_row(inst), as_of))
            return True

    def get(self, instrument_id: str, as_of: Ms) -> Instrument | None:
        """Point-in-time lookup: the version valid at ``as_of``, or None if unknown then.

        Boundary semantics: ``valid_from_ms`` inclusive, ``valid_to_ms`` exclusive.
        """
        cur = self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM instruments_v "
            "WHERE instrument_id = ? AND valid_from_ms <= ? "
            "AND (valid_to_ms IS NULL OR valid_to_ms > ?)",
            (instrument_id, as_of, as_of),
        )
        row = cur.fetchone()
        return None if row is None else _from_row(row)

    def all_known(self, as_of: Ms) -> list[Instrument]:
        """All instruments with a version valid at ``as_of``, ordered by instrument_id."""
        cur = self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM instruments_v "
            "WHERE valid_from_ms <= ? AND (valid_to_ms IS NULL OR valid_to_ms > ?) "
            "ORDER BY instrument_id",
            (as_of, as_of),
        )
        return [_from_row(row) for row in cur.fetchall()]

    def history(self, instrument_id: str) -> list[tuple[Ms, Ms | None, Instrument]]:
        """Full version history as ``(valid_from_ms, valid_to_ms, instrument)``.

        Ordered by ``valid_from_ms`` ascending; the last (and only the last) entry of a
        live instrument has ``valid_to_ms is None``. Empty list if the id is unknown.
        """
        cur = self._conn.execute(
            f"SELECT {_SELECT_COLS}, valid_from_ms, valid_to_ms FROM instruments_v "
            "WHERE instrument_id = ? ORDER BY valid_from_ms",
            (instrument_id,),
        )
        out: list[tuple[Ms, Ms | None, Instrument]] = []
        for row in cur.fetchall():
            valid_to = row["valid_to_ms"]
            out.append(
                (
                    int(row["valid_from_ms"]),
                    None if valid_to is None else int(valid_to),
                    _from_row(row),
                )
            )
        return out
