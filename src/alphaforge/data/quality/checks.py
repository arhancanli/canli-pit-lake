"""Data-quality checks — flags ANNOTATE bars, prices are NEVER altered.

Implements dataDesign.md §5 with the leakage-finding-5 discipline: every flag bit a
check sets carries its own availability lag in
:data:`alphaforge.data.schemas.FLAG_AVAILABILITY_LAG_BARS`, and the PIT reader masks
not-yet-visible bits at read time. The checks here only *compute and store* bits; they
never gate visibility themselves.

Check inventory (thresholds from :class:`alphaforge.config.settings.QualityCfg`):

- :class:`GapCheck` — missing expected bar opens within the instrument's stored span.
  Gaps are REPORTED, never synthetically filled (dataDesign.md §0.5); the surviving
  bars on either side of each gap run get ``QualityFlag.GAP_ADJACENT``.
- :class:`OutlierReturnCheck` — ``|log return| > k*sigma`` with sigma a trailing EWMA
  volatility computed on PRIOR bars only (the estimate at ``t`` uses returns through
  ``t-1``; the flag at ``t`` therefore never uses ``t+1``). Sets
  ``QualityFlag.OUTLIER_RETURN`` (lag 0 — visible at the bar's close).
- :class:`BadPrintCheck` — the three-condition suspect detector (dataDesign.md §5.2):
  outlier return AND volume disconfirmation AND next-bar reversion. The reversion
  condition reads ``close_{t+1}``, which is exactly why ``BAD_PRINT_SUSPECT`` has a
  declared availability lag of 2 bars (leakage finding 5) — this module only SETS the
  bit; the reader masks it.
- :class:`StaleCloseCheck` — a run of identical closes *with trading volume* is a
  stuck-feed symptom (price frozen while trades print). Finding only, no flag bit.
- :class:`DowntimeCheck` — cross-sectional: an hour missing on a supermajority of the
  instruments active at that hour is exchange downtime; the first surviving bar after
  the window gets ``QualityFlag.EXCHANGE_DOWNTIME`` (its return spans the outage).

:func:`apply_flags` orchestrates a batch run over the lake and writes flag upgrades
back through :class:`~alphaforge.data.store.writer.LakeWriter` — the sanctioned
mutation path: same rows, updated ``quality_flags``, fresh ``ingested_at``; the
writer's dedupe-keep-latest semantics make it an in-place flag upgrade. Price columns
round-trip untouched, and the run is idempotent (a second run changes zero rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa

from alphaforge.config.settings import QualityCfg
from alphaforge.core.time import Ms, Timeframe, expected_bar_opens, now_ms
from alphaforge.data.schemas import QualityFlag, ohlcv_dataset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from alphaforge.data.store.reader import PITDataReader
    from alphaforge.data.store.writer import LakeWriter

__all__ = [
    "BadPrintCheck",
    "Check",
    "CheckResult",
    "DowntimeCheck",
    "Finding",
    "GapCheck",
    "OutlierReturnCheck",
    "PanelResult",
    "QualityRunStats",
    "Severity",
    "StaleCloseCheck",
    "apply_flags",
    "default_checks",
]

type Severity = Literal["info", "warn", "error"]
"""Finding severity. ``error`` findings should gate research jobs (dataDesign.md §5.4)."""

_GAP_ERROR_BARS: Final[int] = 3
"""Gap runs longer than this many bars are ``error`` severity (dataDesign.md §5.2)."""

_DOWNTIME_MIN_ACTIVE: Final[int] = 2
"""Cross-sectional downtime needs at least this many active instruments at the hour —
a single instrument missing a bar is its own gap, never 'exchange downtime'."""

_FULL_HISTORY_END: Final[Ms] = 253_402_300_800_000
"""10000-01-01T00:00:00Z — used as both ``end`` and ``as_of`` for the quality engine's
internal PIT-free full-history reads (quality runs on ALL stored data; every declared
flag bit is visible this far in the future, so the read is effectively unmasked)."""


class Finding(TypedDict):
    """One structured quality finding (JSON-serializable by construction).

    ``ts_start``/``ts_end`` are the affected span in epoch ms UTC, half-open on bar
    opens (a single-bar finding has ``ts_end = ts_start + Δ``). ``instrument_id`` is
    ``"*"`` for cross-sectional (exchange-level) findings. ``detail`` holds
    check-specific values (JSON-native types only — int/float/str/list/dict — so the
    report artifact round-trips exactly).
    """

    kind: str
    instrument_id: str
    ts_start: Ms
    ts_end: Ms
    severity: Severity
    detail: dict[str, Any]


@dataclass(frozen=True)
class CheckResult:
    """Output of one per-instrument check over one bar table.

    ``flags`` is an int32 bitmask array aligned row-for-row with the input table —
    the bits this check wants OR-ed into ``quality_flags`` (never cleared, never a
    price mutation). ``findings`` are the structured report records.
    """

    flags: npt.NDArray[np.int32]
    findings: list[Finding]


class Check(Protocol):
    """Per-instrument quality check over a full-history bar table.

    ``tbl_1h`` must conform to ``OHLCV_SCHEMA`` and be sorted ascending on
    ``ts_open`` (both guaranteed by the reader). Implementations are pure: no I/O,
    no mutation of the input, deterministic output.
    """

    @property
    def name(self) -> str:
        """Stable check identifier used as the finding ``kind``."""
        ...

    def run(self, tbl_1h: pa.Table, *, instrument_id: str, tf: Timeframe) -> CheckResult: ...


# --------------------------------------------------------------------------- helpers


def _epoch_ms(column: pa.ChunkedArray) -> npt.NDArray[np.int64]:
    """``timestamp[ms, UTC]`` column → int64 epoch-ms array (exact, no floats)."""
    raw = column.to_numpy(zero_copy_only=False)  # ChunkedArray.to_numpy is untyped (Any)
    out: npt.NDArray[np.int64] = raw.astype("datetime64[ms]").astype(np.int64)
    return out


def _f64(tbl: pa.Table, name: str) -> npt.NDArray[np.float64]:
    raw = tbl.column(name).to_numpy(zero_copy_only=False)
    return np.asarray(raw, dtype=np.float64)


def _i32(tbl: pa.Table, name: str) -> npt.NDArray[np.int32]:
    raw = tbl.column(name).to_numpy(zero_copy_only=False)
    return np.asarray(raw, dtype=np.int32)


def _zeros(n: int) -> npt.NDArray[np.int32]:
    return np.zeros(n, dtype=np.int32)


def _log_returns(close: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-wise log returns; index 0 is NaN. Consecutive ROWS, not consecutive hours —
    the return of a bar following a gap spans the gap (that bar is flag-annotated by
    :class:`GapCheck`/:class:`DowntimeCheck`, never price-adjusted)."""
    r = np.full(close.shape, np.nan, dtype=np.float64)
    if close.size > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.log(close[1:] / close[:-1])
    return r


def _trailing_ewma_sigma(
    log_returns: npt.NDArray[np.float64], *, span: int, min_obs: int
) -> npt.NDArray[np.float64]:
    """Trailing EWMA volatility sigma: at index ``t`` the EWMA of squared returns over
    returns **through ``t-1`` only** (shifted by one bar — the no-lookahead estimate
    of leakage discipline: the flag at ``t`` must not use ``t`` itself, let alone
    ``t+1``). NaN until ``min_obs`` prior returns exist."""
    sq = pd.Series(np.square(log_returns))
    var = sq.ewm(span=span, min_periods=min_obs, adjust=True).mean().shift(1)
    return np.sqrt(np.asarray(var.to_numpy(), dtype=np.float64))


def _trailing_median(
    values: npt.NDArray[np.float64], *, window: int, min_obs: int
) -> npt.NDArray[np.float64]:
    """Trailing rolling median over the ``window`` bars strictly BEFORE each index
    (shifted by one bar); NaN until ``min_obs`` prior observations exist."""
    med = pd.Series(values).rolling(window=window, min_periods=min_obs).median().shift(1)
    return np.asarray(med.to_numpy(), dtype=np.float64)


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal runs of True in ``mask`` as half-open index intervals ``[i0, i1)``."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _missing_runs(missing: list[Ms], step: int) -> list[tuple[Ms, int]]:
    """Group sorted missing bar opens into ``(run_start, n_bars)`` runs of
    consecutive (step-spaced) opens."""
    runs: list[tuple[Ms, int]] = []
    for ts in missing:
        if runs and ts == runs[-1][0] + runs[-1][1] * step:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((ts, 1))
    return runs


# ---------------------------------------------------------------------------- checks


class GapCheck:
    """Missing expected bar opens within ``[first_bar, last_bar]`` of the stored span.

    Each maximal run of missing opens yields one finding (``warn`` ≤ 3 bars, else
    ``error``; dataDesign.md §5.2). Gaps are never filled; the surviving bars
    immediately before and after each run get ``QualityFlag.GAP_ADJACENT`` (the
    after-bar's return spans the gap; the before-bar is the last clean print).
    """

    name = "gap"

    def run(self, tbl_1h: pa.Table, *, instrument_id: str, tf: Timeframe) -> CheckResult:
        ts = _epoch_ms(tbl_1h.column("ts_open"))
        flags = _zeros(len(ts))
        findings: list[Finding] = []
        if len(ts) < 2:
            return CheckResult(flags=flags, findings=findings)
        expected = expected_bar_opens(int(ts[0]), int(ts[-1]) + tf.ms, tf)
        present = set(ts.tolist())
        missing = [t for t in expected if t not in present]
        for run_start, n_bars in _missing_runs(missing, tf.ms):
            run_end = run_start + n_bars * tf.ms
            severity: Severity = "warn" if n_bars <= _GAP_ERROR_BARS else "error"
            findings.append(
                Finding(
                    kind=self.name,
                    instrument_id=instrument_id,
                    ts_start=run_start,
                    ts_end=run_end,
                    severity=severity,
                    detail={"n_bars": n_bars},
                )
            )
            # Maximal runs inside the stored span: the opens at run_start - Δ and
            # run_end are present by construction, so both lookups always hit.
            idx_before = int(np.searchsorted(ts, run_start)) - 1
            idx_after = int(np.searchsorted(ts, run_end))
            flags[idx_before] |= int(QualityFlag.GAP_ADJACENT)
            flags[idx_after] |= int(QualityFlag.GAP_ADJACENT)
        return CheckResult(flags=flags, findings=findings)


class OutlierReturnCheck:
    """``|r_t| > k*sigma_t`` → ``QualityFlag.OUTLIER_RETURN`` (lag 0, visible at close).

    sigma is the trailing EWMA volatility (span ``ewma_span_bars``) computed on PRIOR
    bars only — the flag at ``t`` uses nothing beyond bar ``t``. No flag fires until
    ``min_history_bars`` prior returns exist (warmup guard).
    """

    name = "outlier_return"

    def __init__(
        self, *, k: float = 8.0, ewma_span_bars: int = 168, min_history_bars: int = 24
    ) -> None:
        self._k = k
        self._span = ewma_span_bars
        self._min_obs = min_history_bars

    def run(self, tbl_1h: pa.Table, *, instrument_id: str, tf: Timeframe) -> CheckResult:
        close = _f64(tbl_1h, "close")
        ts = _epoch_ms(tbl_1h.column("ts_open"))
        flags = _zeros(len(ts))
        r = _log_returns(close)
        sigma = _trailing_ewma_sigma(r, span=self._span, min_obs=self._min_obs)
        with np.errstate(invalid="ignore"):
            hit = (np.abs(r) > self._k * sigma) & (sigma > 0.0)
        flags[hit] |= int(QualityFlag.OUTLIER_RETURN)
        findings = [
            Finding(
                kind=self.name,
                instrument_id=instrument_id,
                ts_start=int(ts[i]),
                ts_end=int(ts[i]) + tf.ms,
                severity="warn",
                detail={
                    "log_return": float(r[i]),
                    "sigma": float(sigma[i]),
                    "z": float(abs(r[i]) / sigma[i]),
                },
            )
            for i in np.flatnonzero(hit).tolist()
        ]
        return CheckResult(flags=flags, findings=findings)


class BadPrintCheck:
    """Three-condition bad-print suspect (dataDesign.md §5.2) → ``BAD_PRINT_SUSPECT``.

    All three must hold at bar ``t``:

    (a) ``|r_t| > k*sigma_t`` (trailing EWMA sigma, prior bars only);
    (b) volume disconfirmation: ``volume_t < vol_frac · median(volume over the
        trailing window)`` — a real 8-sigma move comes with volume, a bad print does not;
    (c) reversion: ``|ln(close_{t+1}/close_{t-1})| < reversion_frac·|r_t|`` — the
        price snapped back.

    Condition (c) reads bar ``t+1``: that is precisely why ``BAD_PRINT_SUSPECT``
    carries an availability lag of 2 bars in ``FLAG_AVAILABILITY_LAG_BARS`` (leakage
    finding 5). This check only SETS the bit; the PIT reader masks it until visible.
    The last stored bar can never be flagged (no ``t+1`` yet) — a later batch run
    upgrades it once the next bar lands, which is safe because late flags only hide.
    """

    name = "bad_print"

    def __init__(
        self,
        *,
        k: float = 8.0,
        ewma_span_bars: int = 168,
        vol_frac: float = 0.25,
        reversion_frac: float = 0.5,
        min_history_bars: int = 24,
    ) -> None:
        self._k = k
        self._span = ewma_span_bars
        self._vol_frac = vol_frac
        self._rev_frac = reversion_frac
        self._min_obs = min_history_bars

    def run(self, tbl_1h: pa.Table, *, instrument_id: str, tf: Timeframe) -> CheckResult:
        close = _f64(tbl_1h, "close")
        volume = _f64(tbl_1h, "volume")
        ts = _epoch_ms(tbl_1h.column("ts_open"))
        n = len(ts)
        flags = _zeros(n)
        r = _log_returns(close)
        sigma = _trailing_ewma_sigma(r, span=self._span, min_obs=self._min_obs)
        med_vol = _trailing_median(volume, window=self._span, min_obs=self._min_obs)
        reversion = np.full(n, np.nan, dtype=np.float64)
        if n > 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                reversion[1:-1] = np.log(close[2:] / close[:-2])
        with np.errstate(invalid="ignore"):
            cond_a = (np.abs(r) > self._k * sigma) & (sigma > 0.0)
            cond_b = volume < self._vol_frac * med_vol
            cond_c = np.abs(reversion) < self._rev_frac * np.abs(r)
        hit = cond_a & cond_b & cond_c
        flags[hit] |= int(QualityFlag.BAD_PRINT_SUSPECT)
        findings = [
            Finding(
                kind=self.name,
                instrument_id=instrument_id,
                ts_start=int(ts[i]),
                ts_end=int(ts[i]) + tf.ms,
                severity="warn",
                detail={
                    "log_return": float(r[i]),
                    "sigma": float(sigma[i]),
                    "volume": float(volume[i]),
                    "median_volume": float(med_vol[i]),
                    "reversion": float(reversion[i]),
                },
            )
            for i in np.flatnonzero(hit).tolist()
        ]
        return CheckResult(flags=flags, findings=findings)


class StaleCloseCheck:
    """Run of ≥ ``run_bars`` consecutive identical closes WITH trading volume.

    A frozen close while trades keep printing is a dead/stuck-feed symptom (the
    venue's candle feed repeats the last print). A frozen close on zero volume is
    just a halted/illiquid market — the universe completeness filters own that case,
    so it is deliberately NOT reported here. Finding only; no flag bit exists for
    this condition in :class:`~alphaforge.data.schemas.QualityFlag`, and per the
    contract we implement exactly the bits that exist.
    """

    name = "stale_close"

    def __init__(self, *, run_bars: int = 12) -> None:
        self._run_bars = run_bars

    def run(self, tbl_1h: pa.Table, *, instrument_id: str, tf: Timeframe) -> CheckResult:
        close = _f64(tbl_1h, "close")
        volume = _f64(tbl_1h, "volume")
        ts = _epoch_ms(tbl_1h.column("ts_open"))
        flags = _zeros(len(ts))
        findings: list[Finding] = []
        if len(ts) < 2:
            return CheckResult(flags=flags, findings=findings)
        same = close[1:] == close[:-1]
        for i0, i1 in _runs(same):
            start, end = i0, i1 + 1  # bar indices [start, end) share one close value
            n_bars = end - start
            vol_sum = float(volume[start:end].sum())
            if n_bars >= self._run_bars and vol_sum > 0.0:
                findings.append(
                    Finding(
                        kind=self.name,
                        instrument_id=instrument_id,
                        ts_start=int(ts[start]),
                        ts_end=int(ts[end - 1]) + tf.ms,
                        severity="warn",
                        detail={
                            "n_bars": n_bars,
                            "close": float(close[start]),
                            "volume_sum": vol_sum,
                        },
                    )
                )
        return CheckResult(flags=flags, findings=findings)


@dataclass(frozen=True)
class PanelResult:
    """Cross-sectional check output: per-instrument flag additions (row-aligned with
    the input tables) plus exchange-level findings."""

    flags_by_instrument: dict[str, npt.NDArray[np.int32]]
    findings: list[Finding]


class DowntimeCheck:
    """Cross-sectional exchange-downtime classifier (dataDesign.md §5.2).

    For each bar open ``t`` missing from some instrument, compute the fraction of
    instruments *active at* ``t`` (stored span covers ``t``) that are also missing
    ``t``. If the fraction ≥ ``downtime_frac`` (and ≥ 2 instruments are active —
    one instrument's gap is never "downtime"), the hour is exchange downtime.
    Consecutive downtime hours merge into one window finding (``info`` severity:
    expected, recorded so backtests can exclude the window).

    Per-row annotation: for every affected instrument, the first surviving bar at or
    after the window's end gets ``QualityFlag.EXCHANGE_DOWNTIME`` — its log return
    spans the outage. Hours missing below the threshold stay plain symbol gaps,
    owned by :class:`GapCheck`.
    """

    name = "exchange_downtime"

    def __init__(self, *, downtime_frac: float = 0.8) -> None:
        self._frac = downtime_frac

    def run_panel(self, tables: Mapping[str, pa.Table], *, tf: Timeframe) -> PanelResult:
        """Classify downtime across the full panel; tables as from the quality run
        (full history per instrument, sorted on ``ts_open``)."""
        ts_by_iid = {iid: _epoch_ms(tbl.column("ts_open")) for iid, tbl in tables.items()}
        flags = {iid: _zeros(len(ts)) for iid, ts in ts_by_iid.items()}
        spans = {iid: (int(ts[0]), int(ts[-1])) for iid, ts in ts_by_iid.items() if len(ts) > 0}
        missing_at: dict[Ms, list[str]] = {}
        for iid, ts in ts_by_iid.items():
            if len(ts) < 2:
                continue
            present = set(ts.tolist())
            for t in expected_bar_opens(int(ts[0]), int(ts[-1]) + tf.ms, tf):
                if t not in present:
                    missing_at.setdefault(t, []).append(iid)

        downtime: list[tuple[Ms, int, int]] = []  # (ts, n_missing, n_active)
        for t in sorted(missing_at):
            n_active = sum(1 for lo, hi in spans.values() if lo <= t <= hi)
            n_missing = len(missing_at[t])
            if n_active >= _DOWNTIME_MIN_ACTIVE and n_missing / n_active >= self._frac:
                downtime.append((t, n_missing, n_active))

        findings: list[Finding] = []
        for run_start, n_bars in _missing_runs([t for t, _, _ in downtime], tf.ms):
            run_end = run_start + n_bars * tf.ms
            in_run = [(t, m, a) for t, m, a in downtime if run_start <= t < run_end]
            affected = sorted({iid for t, _, _ in in_run for iid in missing_at[t]})
            findings.append(
                Finding(
                    kind=self.name,
                    instrument_id="*",
                    ts_start=run_start,
                    ts_end=run_end,
                    severity="info",
                    detail={
                        "n_bars": n_bars,
                        "min_frac": min(m / a for _, m, a in in_run),
                        "n_affected": len(affected),
                        "affected": affected,
                    },
                )
            )
            for iid in affected:
                ts = ts_by_iid[iid]
                idx = int(np.searchsorted(ts, run_end))
                if idx < len(ts):
                    flags[iid][idx] |= int(QualityFlag.EXCHANGE_DOWNTIME)
        return PanelResult(flags_by_instrument=flags, findings=findings)


# ----------------------------------------------------------------- batch application


def default_checks(cfg: QualityCfg) -> list[Check]:
    """The standard per-instrument check suite wired from ``settings.quality``.

    (:class:`DowntimeCheck` is cross-sectional and runs separately on the panel.)
    """
    return [
        GapCheck(),
        OutlierReturnCheck(
            k=cfg.outlier_k,
            ewma_span_bars=cfg.ewma_span_bars,
            min_history_bars=cfg.min_history_bars,
        ),
        BadPrintCheck(
            k=cfg.outlier_k,
            ewma_span_bars=cfg.ewma_span_bars,
            vol_frac=cfg.bad_print_vol_frac,
            reversion_frac=cfg.bad_print_reversion_frac,
            min_history_bars=cfg.min_history_bars,
        ),
        StaleCloseCheck(run_bars=cfg.stale_run_bars),
    ]


@dataclass(frozen=True)
class QualityRunStats:
    """Accounting for one :func:`apply_flags` batch run.

    - ``instruments``: instruments scanned (including empty ones).
    - ``bars_scanned``: total bars read across all instruments.
    - ``rows_reflagged``: rows whose ``quality_flags`` gained at least one bit and
      were rewritten through the writer (0 on an idempotent re-run).
    - ``findings``: every structured finding from every check, in scan order
      (per-instrument findings first, downtime windows last).
    - ``bar_counts``: bars per instrument — the report's per-instrument denominator.
    """

    instruments: int
    bars_scanned: int
    rows_reflagged: int
    findings: list[Finding]
    bar_counts: dict[str, int]


def apply_flags(
    reader: PITDataReader,
    writer: LakeWriter,
    instrument_ids: Sequence[str],
    *,
    tf: Timeframe,
    cfg: QualityCfg | None = None,
    now: Ms | None = None,
) -> QualityRunStats:
    """Run all quality checks over the FULL stored history and persist flag upgrades.

    Reads per-instrument bars PIT-free (quality runs on all data: ``as_of`` far
    enough in the future that every declared flag bit is visible), computes the
    union of flag bits per row across all checks plus the cross-sectional downtime
    panel, and writes back ONLY rows whose flags changed — same rows, updated
    ``quality_flags``, fresh ``ingested_at = now`` — via ``LakeWriter.write``. The
    writer's merge-dedupe-keep-latest makes that an in-place flag upgrade; this is
    the sanctioned mutation path and price columns are bit-for-bit untouched. Bits
    are only ever ADDED (``new = stored | computed``), so the run is monotone and
    idempotent: a second run reflags zero rows.

    ``cfg`` defaults to :class:`QualityCfg` code defaults; ``now`` (epoch ms UTC,
    injectable for tests) stamps ``ingested_at`` and parameterizes the writer's
    partial-bar guard. Duplicate ids are scanned once.
    """
    quality_cfg = QualityCfg() if cfg is None else cfg
    now_val = now_ms() if now is None else now
    checks = default_checks(quality_cfg)
    downtime = DowntimeCheck(downtime_frac=quality_cfg.downtime_frac)

    tables: dict[str, pa.Table] = {}
    additions: dict[str, npt.NDArray[np.int32]] = {}
    findings: list[Finding] = []
    for iid in dict.fromkeys(instrument_ids):
        tbl = reader.ohlcv([iid], start=0, end=_FULL_HISTORY_END, as_of=_FULL_HISTORY_END, tf=tf)
        tables[iid] = tbl
        add = _zeros(tbl.num_rows)
        for check in checks:
            result = check.run(tbl, instrument_id=iid, tf=tf)
            add |= result.flags
            findings.extend(result.findings)
        additions[iid] = add

    panel = downtime.run_panel(tables, tf=tf)
    findings.extend(panel.findings)
    for iid, extra in panel.flags_by_instrument.items():
        additions[iid] = additions[iid] | extra

    rows_reflagged = 0
    bar_counts: dict[str, int] = {}
    for iid, tbl in tables.items():
        bar_counts[iid] = tbl.num_rows
        stored = _i32(tbl, "quality_flags")
        new = stored | additions[iid]
        changed = new != stored
        n_changed = int(changed.sum())
        if n_changed == 0:
            continue
        updated = tbl.set_column(
            tbl.schema.get_field_index("quality_flags"),
            tbl.schema.field("quality_flags"),
            pa.array(new, type=pa.int32()),
        ).set_column(
            tbl.schema.get_field_index("ingested_at"),
            tbl.schema.field("ingested_at"),
            pa.array([now_val] * tbl.num_rows, type=pa.timestamp("ms", tz="UTC")),
        )
        writer.write(ohlcv_dataset(tf), updated.filter(pa.array(changed)), tf=tf, now=now_val)
        rows_reflagged += n_changed

    return QualityRunStats(
        instruments=len(tables),
        bars_scanned=sum(bar_counts.values()),
        rows_reflagged=rows_reflagged,
        findings=findings,
        bar_counts=bar_counts,
    )
