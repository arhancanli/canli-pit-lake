"""Persisted Parquet schemas — the single source of truth for the data lake.

Contract (dataDesign.md §2.3, leakage finding 17): every table written to or read from
the lake conforms to one of the :class:`pyarrow.Schema` constants in this module. The
writer calls :func:`validate_table` before every write; nothing above the storage layer
defines its own column set. All timestamps are ``timestamp[ms, tz=UTC]`` — a naive
timestamp column is a *type* mismatch and fails validation (dataDesign.md §5.4).

Key conventions enforced here:

- OHLCV bars are labeled by **open** time (``ts_open``); ``available_at`` is *not*
  stored for OHLCV — it is always derived as ``ts_open + Δ`` by the reader (storing it
  would invite drift; dataDesign.md §2.3).
- Funding rows *do* store ``available_at = ts_funding + δ_funding`` (publication lag,
  default 5 minutes; dataDesign.md §3.2). Per leakage finding 18, every consumer joins
  on ``available_at``, full stop — never on ``ts_funding``.
- Quality flags are a bitmask whose bits carry their **own** availability lag
  (:data:`FLAG_AVAILABILITY_LAG_BARS`; leakage finding 5): a flag computed from future
  bars must not be visible to a decision at the bar's close.

File-format notes: compression (zstd level 3) and the sorted/unique-on-``ts_open``
invariant are decided and enforced at the **writer** level, not here — schemas describe
logical column contracts only.
"""

from __future__ import annotations

from collections import Counter
from enum import IntFlag, StrEnum
from pathlib import PurePosixPath
from typing import Final

import pyarrow as pa

from alphaforge.core.errors import SchemaError
from alphaforge.core.time import Ms, Timeframe, available_at

__all__ = [
    "CORPORATE_ACTIONS_SCHEMA",
    "FLAG_AVAILABILITY_LAG_BARS",
    "FUNDAMENTALS_SCHEMA",
    "FUNDING_SCHEMA",
    "OHLCV_DATASETS",
    "OHLCV_SCHEMA",
    "SCHEMA_VERSION",
    "UNIVERSE_SCHEMA",
    "Dataset",
    "QualityFlag",
    "empty_table",
    "flag_available_at",
    "lake_relpath",
    "ohlcv_dataset",
    "schema_for",
    "validate_table",
]

SCHEMA_VERSION: Final[int] = 1
"""Current lake schema version, embedded in every schema's key-value metadata under
``b"alphaforge.schema_version"``. Bump on any breaking column change."""

_METADATA_VERSION_KEY: Final[bytes] = b"alphaforge.schema_version"
_METADATA_DATASET_KEY: Final[bytes] = b"alphaforge.dataset"


class Dataset(StrEnum):
    """Lake datasets. Values are the top-level lake directory names (partition labels).

    The OHLCV family is one dataset *per timeframe*: ``ohlcv`` holds the native 1h
    bars (the only ingested timeframe; dataDesign.md §4.4); ``ohlcv_4h``/``ohlcv_1d``
    hold bars *derived* from stored 1h bars by the resampler — never fetched. All
    three share :data:`OHLCV_SCHEMA` column-for-column; use :func:`ohlcv_dataset`
    to resolve a :class:`~alphaforge.core.time.Timeframe` to its dataset.
    """

    OHLCV = "ohlcv"
    OHLCV_4H = "ohlcv_4h"
    OHLCV_1D = "ohlcv_1d"
    FUNDING = "funding"
    CORPORATE_ACTIONS = "corporate_actions"
    FUNDAMENTALS = "fundamentals"
    UNIVERSE_MEMBERSHIP = "universe_membership"


OHLCV_DATASETS: Final[frozenset[Dataset]] = frozenset(
    {Dataset.OHLCV, Dataset.OHLCV_4H, Dataset.OHLCV_1D}
)
"""The OHLCV dataset family (native 1h + resampler-derived 4h/1d). Membership gates
OHLCV-specific behavior — e.g. the writer's partial-bar guard — without enumerating
timeframes at every call site."""

_OHLCV_DATASET_BY_TF: Final[dict[Timeframe, Dataset]] = {
    Timeframe.H1: Dataset.OHLCV,
    Timeframe.H4: Dataset.OHLCV_4H,
    Timeframe.D1: Dataset.OHLCV_1D,
}


def ohlcv_dataset(tf: Timeframe) -> Dataset:
    """Resolve a bar timeframe to the lake dataset holding bars of that timeframe.

    ``H1 → ohlcv`` (native, ingested), ``H4 → ohlcv_4h`` and ``D1 → ohlcv_1d``
    (derived by the resampler from stored 1h bars). Total over every
    :class:`~alphaforge.core.time.Timeframe` member — enforced by a unit test so a
    new timeframe cannot be added without declaring its dataset.
    """
    return _OHLCV_DATASET_BY_TF[tf]


class QualityFlag(IntFlag):
    """Per-bar data-quality bitmask stored in the OHLCV ``quality_flags`` column.

    Flags never modify price data — they annotate it; downstream feature code applies a
    *declared* policy per bit (dataDesign.md §5.3). Each bit documents WHEN it can be
    computed, because per leakage finding 5 the bits carry their own availability lag
    (:data:`FLAG_AVAILABILITY_LAG_BARS`): a PIT reader must mask bits whose
    :func:`flag_available_at` exceeds the decision timestamp, otherwise research sees
    retroactively upgraded flags that live trading could not have seen.

    - ``NONE``: clean bar. Known at the bar's own close (a bar is final at close).
    - ``GAP_FILLED_NONE``: **reserved.** v1 declares no fill policy — missing bars raise
      ``DataGapError`` instead of being synthesized. The bit is reserved so any future
      fill policy is forced to mark synthetic rows explicitly. Timing if ever used: a
      gap is confirmed by the *next* ingest cycle after the bar failed to arrive → the
      synthetic row's flag lags one bar.
    - ``OUTLIER_RETURN``: inline check (dataDesign.md §5.2 conditions 1-2): robust-z of
      the bar's log return vs a trailing window ending *strictly before* the bar, plus
      volume disconfirmation. Uses only data through the bar itself → computable at the
      bar's close, lag 0.
    - ``VOLUME_ANOMALY``: bar volume vs trailing median volume (trailing window only) →
      computable at the bar's close, lag 0.
    - ``BAD_PRINT_SUSPECT``: batch upgrade of ``OUTLIER_RETURN`` requiring the reversion
      condition on ``close_{t+1}`` (dataDesign.md §5.2 condition 3) — the bit encodes
      bar ``t+1`` information. Theoretical minimum availability is ``ts_open + 2Δ``
      (leakage finding 5's fix); the declared lag of 2 extra bars (visible at
      ``ts_open + 3Δ``) adds one bar of slack for the batch upgrade job. Late is safe:
      a later ``available_at`` can only hide a flag, never leak one.
    - ``EXCHANGE_DOWNTIME``: cross-symbol downtime classification (dataDesign.md §5.2)
      needs the full panel at the gap timestamp, i.e. every active symbol's fetch for
      that bar must have completed/failed → one bar of lag, visible at ``ts_open + 2Δ``.
    - ``GAP_ADJACENT``: surviving bar immediately BEFORE or AFTER a run of missing
      expected bars (dataDesign.md §5.2; gaps are reported, never filled — this bit
      annotates the survivors). The bar after a gap carries a log return spanning the
      gap; the bar before is the last clean print. A gap is only *confirmed* once the
      missing bar failed to arrive by the next ingest cycle (same reasoning as
      ``GAP_FILLED_NONE``) → one bar of lag: for the preceding bar that is exactly the
      missing bar's own close. Late is safe — a later ``available_at`` can only hide a
      flag, never leak one. (Value 64; 32 is left free for the resampler's
      ``INCOMPLETE_AGGREGATE`` per dataDesign.md §5.3.)
    """

    NONE = 0
    GAP_FILLED_NONE = 1
    OUTLIER_RETURN = 2
    VOLUME_ANOMALY = 4
    BAD_PRINT_SUSPECT = 8
    EXCHANGE_DOWNTIME = 16
    GAP_ADJACENT = 64


FLAG_AVAILABILITY_LAG_BARS: Final[dict[QualityFlag, int]] = {
    QualityFlag.GAP_FILLED_NONE: 1,
    QualityFlag.OUTLIER_RETURN: 0,
    QualityFlag.VOLUME_ANOMALY: 0,
    QualityFlag.BAD_PRINT_SUSPECT: 2,
    QualityFlag.EXCHANGE_DOWNTIME: 1,
    QualityFlag.GAP_ADJACENT: 1,
}
"""Availability lag per flag bit, in bars *beyond the bar's own close* (finding 5).

A flag with lag ``L`` on the bar opening at ``ts_open`` becomes visible to decisions at
``ts_open + (1 + L)·Δ`` (lag 0 = visible the moment the bar itself is available). The
PIT reader masks bits that are not yet visible at ``as_of``; the feature-layer NaN
policy may only consume visible bits. Covers every non-``NONE`` flag — enforced by a
unit test so a new bit cannot be added without declaring its lag.
"""


def flag_available_at(ts_open: Ms, tf: Timeframe, flags: QualityFlag) -> Ms:
    """Earliest decision timestamp (epoch ms UTC) at which ``flags`` is fully visible.

    For a composite bitmask the answer is governed by the slowest bit:
    ``available_at(ts_open, tf) + max(lag of set bits)·Δ``. ``QualityFlag.NONE`` (no
    bits set) is visible at the bar's own close. Implements leakage finding 5: a PIT
    read at ``as_of`` may expose a bit iff ``flag_available_at(...) <= as_of``.
    """
    max_lag = max(
        (FLAG_AVAILABILITY_LAG_BARS[member] for member in QualityFlag if member in flags),
        default=0,
    )
    return available_at(ts_open, tf) + max_lag * tf.ms


def _ts_utc_ms() -> pa.DataType:
    """The one sanctioned Arrow timestamp type: ``timestamp[ms, tz=UTC]``."""
    return pa.timestamp("ms", tz="UTC")


def _metadata(dataset: Dataset) -> dict[bytes, bytes]:
    return {
        _METADATA_VERSION_KEY: str(SCHEMA_VERSION).encode("ascii"),
        _METADATA_DATASET_KEY: dataset.value.encode("ascii"),
    }


OHLCV_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
            metadata={"doc": "canonical id, e.g. BINANCE:PERP:BTCUSDT"},
        ),
        pa.field(
            "ts_open",
            _ts_utc_ms(),
            nullable=False,
            metadata={"doc": "bar OPEN time; available_at = ts_open + timeframe (derived)"},
        ),
        pa.field("open", pa.float64(), nullable=False, metadata={"doc": "open price, quote"}),
        pa.field("high", pa.float64(), nullable=False, metadata={"doc": "high price, quote"}),
        pa.field("low", pa.float64(), nullable=False, metadata={"doc": "low price, quote"}),
        pa.field("close", pa.float64(), nullable=False, metadata={"doc": "close price, quote"}),
        pa.field(
            "volume",
            pa.float64(),
            nullable=False,
            metadata={"doc": "traded volume, base-asset units"},
        ),
        pa.field(
            "quote_volume",
            pa.float64(),
            nullable=True,
            metadata={"doc": "exact quote-asset volume; null for sources lacking it"},
        ),
        pa.field(
            "n_trades",
            pa.int64(),
            nullable=True,
            metadata={"doc": "trade count in bar; null for sources lacking it"},
        ),
        pa.field(
            "quality_flags",
            pa.int32(),
            nullable=False,
            metadata={"doc": "QualityFlag bitmask; writer defaults to 0 (clean)"},
        ),
        pa.field(
            "ingested_at",
            _ts_utc_ms(),
            nullable=False,
            metadata={"doc": "wall clock at write; audit only, never a feature input"},
        ),
    ],
    metadata=_metadata(Dataset.OHLCV),
)
"""OHLCV bars, labeled by open time. ``available_at`` is derived, never stored."""


FUNDING_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
            metadata={"doc": "canonical id, e.g. BINANCE:PERP:BTCUSDT"},
        ),
        pa.field(
            "ts_funding",
            _ts_utc_ms(),
            nullable=False,
            metadata={"doc": "funding settlement time (e.g. 00:00/08:00/16:00 UTC)"},
        ),
        pa.field(
            "rate",
            pa.float64(),
            nullable=False,
            metadata={"doc": "per-interval decimal rate (0.0001 = 1 bp per interval)"},
        ),
        pa.field(
            "available_at",
            _ts_utc_ms(),
            nullable=False,
            metadata={
                "doc": (
                    "ts_funding + publication lag (default 5 min); ALL consumers join "
                    "on this column, never on ts_funding (leakage finding 18)"
                )
            },
        ),
        pa.field(
            "ingested_at",
            _ts_utc_ms(),
            nullable=False,
            metadata={"doc": "wall clock at write; audit only"},
        ),
    ],
    metadata=_metadata(Dataset.FUNDING),
)
"""Funding settlements with explicit point-in-time ``available_at``."""


CORPORATE_ACTIONS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
            metadata={"doc": "canonical id, e.g. XNAS:CASH:AAPLUSD"},
        ),
        pa.field(
            "action_type",
            pa.string(),
            nullable=False,
            metadata={"doc": "'split' or 'dividend'"},
        ),
        pa.field(
            "ex_date",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={"doc": "ex/effective date, 00:00 UTC"},
        ),
        pa.field(
            "available_at",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={
                "doc": (
                    "knowable_date + publication lag; ALL adjustment joins use this "
                    "column, never ex_date (leakage finding 18)"
                )
            },
        ),
        pa.field(
            "ratio",
            pa.float64(),
            nullable=False,
            metadata={"doc": "split: split_to/split_from; dividend: 1.0"},
        ),
        pa.field(
            "cash_amount",
            pa.float64(),
            nullable=True,
            metadata={"doc": "dividend cash per share; null for splits"},
        ),
        pa.field(
            "ingested_at",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={"doc": "wall clock at write; audit only"},
        ),
    ],
    metadata={
        b"alphaforge.schema_version": b"1",
        b"alphaforge.dataset": b"corporate_actions",
    },
)
"""Corporate-action settlements (splits + dividends) with explicit point-in-time
``available_at`` — the equity analogue of :data:`FUNDING_SCHEMA`.

One row per action; ``available_at`` is the knowable-by timestamp ALL downstream
adjustment joins use (never ``ex_date``; leakage finding 18). Splits carry
``ratio = split_to/split_from`` with a null ``cash_amount``; dividends carry
``ratio = 1.0`` and the cash-per-share amount. The PIT adjusted-close
reconstruction (:func:`~alphaforge.features.library.equity_price.adjusted_close`)
folds ``ratio`` backward into history. ``ex_date`` is the within-leaf dedupe/sort
natural key (see :data:`~alphaforge.data.store.writer.NATURAL_KEY_COLUMN`).
"""


FUNDAMENTALS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
            metadata={"doc": "canonical id, e.g. XUSE:CASH:AAPLUSD"},
        ),
        pa.field(
            "period_end",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={
                "doc": "fiscal-period end (end_date), 00:00 UTC; the within-leaf dedupe/sort key"
            },
        ),
        pa.field(
            "available_at",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={
                "doc": (
                    "SEC filing_date (point-in-time knowability); ALL factor reads filter "
                    "available_at <= decision_t, NEVER period_end (leakage: fundamentals are "
                    "known only when FILED, ~30-75d after the period ends). Fallback when the "
                    "API omits filing_date: period_end + conservative SEC deadline."
                )
            },
        ),
        pa.field(
            "fiscal_period",
            pa.string(),
            nullable=False,
            metadata={"doc": "'Q1'..'Q4' (or 'FY'); with fiscal_year identifies the period"},
        ),
        pa.field(
            "fiscal_year",
            pa.int32(),
            nullable=False,
            metadata={"doc": "fiscal year of the period"},
        ),
        # --- line items (quote currency, nullable: filings omit fields) ----------------
        # The minimal set spanning value (E/P, B/P, S/P) and quality (gross profitability,
        # ROE, margins). Market cap is price x diluted_shares at decision time (price from
        # the OHLCV lake, shares from here) — never stored, always PIT-joined.
        pa.field(
            "revenues",
            pa.float64(),
            nullable=True,
            metadata={"doc": "total revenue (S/P, margins)"},
        ),
        pa.field(
            "cost_of_revenue", pa.float64(), nullable=True, metadata={"doc": "COGS (gross profit)"}
        ),
        pa.field(
            "gross_profit",
            pa.float64(),
            nullable=True,
            metadata={"doc": "revenue - COGS if reported"},
        ),
        pa.field(
            "operating_income", pa.float64(), nullable=True, metadata={"doc": "operating margin"}
        ),
        pa.field(
            "net_income", pa.float64(), nullable=True, metadata={"doc": "net income (E/P, ROE)"}
        ),
        pa.field(
            "equity",
            pa.float64(),
            nullable=True,
            metadata={"doc": "total shareholders' equity (B/P, ROE)"},
        ),
        pa.field(
            "assets",
            pa.float64(),
            nullable=True,
            metadata={"doc": "total assets (gross profitability, ROA)"},
        ),
        pa.field(
            "diluted_shares",
            pa.float64(),
            nullable=True,
            metadata={"doc": "diluted share count (per-share, market cap)"},
        ),
        # --- cash-flow + share fields for accruals / issuance / investment sleeves --------
        pa.field(
            "op_cash_flow",
            pa.float64(),
            nullable=True,
            metadata={"doc": "ncfo; accruals = (net_income - op_cash_flow) / assets"},
        ),
        pa.field(
            "invest_cash_flow",
            pa.float64(),
            nullable=True,
            metadata={"doc": "ncfi; investment / asset-growth signals"},
        ),
        pa.field(
            "capex", pa.float64(), nullable=True, metadata={"doc": "capex; investment intensity"}
        ),
        pa.field(
            "free_cash_flow", pa.float64(), nullable=True, metadata={"doc": "fcf"}
        ),
        pa.field(
            "net_common_issued",
            pa.float64(),
            nullable=True,
            metadata={"doc": "ncfcommon; net-issuance cross-check"},
        ),
        pa.field(
            "shares_basic",
            pa.float64(),
            nullable=True,
            metadata={"doc": "sharesbas; net-issuance signal (x share_factor)"},
        ),
        pa.field(
            "share_factor",
            pa.float64(),
            nullable=True,
            metadata={"doc": "sharefactor; split-adjust to compare share counts"},
        ),
        pa.field(
            "assets_avg",
            pa.float64(),
            nullable=True,
            metadata={"doc": "assetsavg; accruals / ROA denominator (annual rows only)"},
        ),
        pa.field(
            "ingested_at",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
            metadata={"doc": "wall clock at write; audit only"},
        ),
    ],
    metadata={
        b"alphaforge.schema_version": b"1",
        b"alphaforge.dataset": b"fundamentals",
    },
)
"""Point-in-time quarterly financial-statement line items (Polygon vX financials / SEC
filings) — the survivorship-free fundamentals table feeding value + quality factors.

One row per (instrument, fiscal period); ``period_end`` is the dedupe/sort key (a
restatement of the same period is overwritten by the later ingest — a documented v1
simplification, conservative since the row's ``available_at`` is the filing date).
``available_at`` is the ONLY column factor reads may filter on for knowability: a
period ending March is not knowable until its ~May 10-Q filing. Line items are
nullable (filings omit fields); market cap is joined PIT from the OHLCV lake (price x
``diluted_shares``), never stored. Partition year is the ``period_end`` year."""


UNIVERSE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
            metadata={"doc": "canonical id, e.g. BINANCE:PERP:BTCUSDT"},
        ),
        pa.field(
            "effective_from",
            _ts_utc_ms(),
            nullable=False,
            metadata={"doc": "first decision ts at which membership holds (= rebalance ts)"},
        ),
        pa.field(
            "effective_to",
            _ts_utc_ms(),
            nullable=True,
            metadata={"doc": "exclusive end of membership; null = currently a member"},
        ),
        pa.field(
            "rank",
            pa.int32(),
            nullable=False,
            metadata={"doc": "liquidity rank at entry (1 = most liquid)"},
        ),
        pa.field(
            "reason",
            pa.string(),
            nullable=False,
            metadata={"doc": "membership action rationale, e.g. enter_top40, delisted"},
        ),
    ],
    metadata=_metadata(Dataset.UNIVERSE_MEMBERSHIP),
)
"""Point-in-time universe membership intervals — the table backtests join against.

Half-open validity ``[effective_from, effective_to)``; membership at ``as_of`` means
``effective_from <= as_of < coalesce(effective_to, +inf)``. Built with data available
strictly at the rebalance timestamp, so survivorship peeking is structurally impossible.
"""


_SCHEMAS: Final[dict[Dataset, pa.Schema]] = {
    Dataset.OHLCV: OHLCV_SCHEMA,
    # Derived bars are column-for-column identical to native OHLCV (dataDesign.md
    # §4.4); only the schema-level dataset tag in the key-value metadata differs.
    Dataset.OHLCV_4H: OHLCV_SCHEMA.with_metadata(_metadata(Dataset.OHLCV_4H)),
    Dataset.OHLCV_1D: OHLCV_SCHEMA.with_metadata(_metadata(Dataset.OHLCV_1D)),
    Dataset.FUNDING: FUNDING_SCHEMA,
    Dataset.CORPORATE_ACTIONS: CORPORATE_ACTIONS_SCHEMA,
    Dataset.FUNDAMENTALS: FUNDAMENTALS_SCHEMA,
    Dataset.UNIVERSE_MEMBERSHIP: UNIVERSE_SCHEMA,
}


def schema_for(dataset: Dataset) -> pa.Schema:
    """Return the declared :class:`pyarrow.Schema` for ``dataset``."""
    return _SCHEMAS[dataset]


def validate_table(tbl: pa.Table, dataset: Dataset) -> None:
    """Validate ``tbl`` against the declared schema for ``dataset``.

    Raises :class:`~alphaforge.core.errors.SchemaError` whose message lists **every**
    mismatch (not just the first), each naming the offending column:

    - missing columns;
    - extra (undeclared) columns;
    - duplicated column names;
    - wrong column types (exact Arrow type equality — a tz-naive timestamp therefore
      fails, enforcing the UTC discipline of dataDesign.md §5.4);
    - null values in columns declared non-nullable (checked against the *data*; the
      incoming table's declared nullable bit is advisory in Arrow and ignored here).

    Column order is not significant. Returns ``None`` on success.
    """
    expected = schema_for(dataset)
    problems: list[str] = []

    name_counts = Counter(tbl.schema.names)
    for name, count in sorted(name_counts.items()):
        if count > 1:
            problems.append(f"column '{name}': appears {count} times (duplicate column name)")

    actual_names = set(name_counts)
    expected_names = set(expected.names)

    for name in sorted(expected_names - actual_names):
        problems.append(f"column '{name}': missing")
    for name in sorted(actual_names - expected_names):
        problems.append(f"column '{name}': not in declared schema (extra column)")

    for name in expected.names:
        if name not in actual_names or name_counts[name] > 1:
            continue
        expected_field = expected.field(name)
        actual_type = tbl.schema.field(name).type
        if actual_type != expected_field.type:
            problems.append(
                f"column '{name}': expected type {expected_field.type}, got {actual_type}"
            )
            continue
        if not expected_field.nullable:
            null_count = tbl.column(name).null_count
            if null_count > 0:
                problems.append(
                    f"column '{name}': {null_count} null value(s) in non-nullable column"
                )

    if problems:
        raise SchemaError(
            f"table does not conform to {dataset.value!r} schema "
            f"(version {SCHEMA_VERSION}); {len(problems)} problem(s): " + "; ".join(problems)
        )


def empty_table(dataset: Dataset) -> pa.Table:
    """Return a zero-row table conforming exactly to ``dataset``'s schema (incl. metadata).

    Useful as the writer's seed for a new partition and as the canonical fixture in
    tests; always passes :func:`validate_table`.
    """
    return schema_for(dataset).empty_table()


def lake_relpath(dataset: Dataset, instrument_id: str, year: int) -> PurePosixPath:
    """Hive-style relative path of a leaf partition inside the lake root.

    Layout: ``<dataset>/instrument_id=<id>/year=<year>/data.parquet`` — e.g.
    ``ohlcv/instrument_id=BINANCE:PERP:BTCUSDT/year=2024/data.parquet``. One file per
    leaf, rewritten whole and atomically on update (writer's concern, as is the zstd
    compression choice). Hive ``key=value`` segments let DuckDB push partition filters
    into the file listing.

    ``instrument_id`` must be a non-empty canonical id without path/hive
    metacharacters (``/``, ``=``, NUL); ``year`` must be a 4-digit year (1000-9999).
    Raises :class:`ValueError` otherwise.
    """
    if not instrument_id:
        raise ValueError("instrument_id must be non-empty")
    bad = [ch for ch in ("/", "=", "\x00") if ch in instrument_id]
    if bad:
        raise ValueError(f"instrument_id {instrument_id!r} contains forbidden character(s): {bad}")
    if not 1000 <= year <= 9999:
        raise ValueError(f"year must be a 4-digit year (1000-9999), got {year}")
    return (
        PurePosixPath(dataset.value)
        / f"instrument_id={instrument_id}"
        / f"year={year}"
        / "data.parquet"
    )
