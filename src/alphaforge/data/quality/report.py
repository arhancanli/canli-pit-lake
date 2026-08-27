"""ValidationReport — the persisted artifact of one quality batch run.

Aggregates the structured findings of ``alphaforge.data.quality.checks`` into a
per-instrument summary plus exchange-downtime windows and totals (dataDesign.md
§5.4). The report is data about data: it never touches the lake. Persistence is a
single JSON file per run (``<utc-timestamp>.json`` in a caller-supplied directory —
the CLI passes ``<quality_dir>/reports``); :meth:`ValidationReport.render_text`
renders the human table.

All timestamps inside the artifact are epoch ms UTC (the boundary convention);
``render_text`` formats them as ISO-8601 ``Z`` strings for humans only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

from alphaforge.core.time import Ms, Timeframe, from_ms
from alphaforge.data.quality.checks import Finding, QualityRunStats, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "InstrumentSummary",
    "ValidationReport",
]

_KIND_GAP: Final[str] = "gap"
_KIND_OUTLIER: Final[str] = "outlier_return"
_KIND_BAD_PRINT: Final[str] = "bad_print"
_KIND_STALE: Final[str] = "stale_close"
_KIND_DOWNTIME: Final[str] = "exchange_downtime"

_SEVERITIES: Final[tuple[Severity, ...]] = ("info", "warn", "error")


def _iso(ms: Ms) -> str:
    return from_ms(ms).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class InstrumentSummary:
    """Per-instrument rollup: bar count plus finding counts by kind.

    ``gaps`` counts gap *runs*; ``gap_bars`` the missing bars they contain.
    ``outliers``/``bad_prints`` count flagged bars; ``stale_runs`` counts
    frozen-close runs.
    """

    instrument_id: str
    bars: int
    gaps: int
    gap_bars: int
    outliers: int
    bad_prints: int
    stale_runs: int


@dataclass(frozen=True)
class ValidationReport:
    """Immutable aggregate of one quality run (build via :meth:`build`).

    ``generated_at`` is the run's wall clock (epoch ms UTC); ``tf`` the bar
    timeframe checked; ``summaries`` one row per scanned instrument (sorted by id);
    ``downtime_windows`` the cross-sectional exchange-downtime findings;
    ``findings`` every finding of the run, in scan order. Round-trips exactly
    through :meth:`persist`/:meth:`load` (dataclass equality holds).
    """

    generated_at: Ms
    tf: Timeframe
    summaries: tuple[InstrumentSummary, ...]
    downtime_windows: tuple[Finding, ...]
    findings: tuple[Finding, ...]

    @classmethod
    def build(cls, stats: QualityRunStats, *, tf: Timeframe, generated_at: Ms) -> ValidationReport:
        """Aggregate a :class:`QualityRunStats` into a report.

        Every instrument in ``stats.bar_counts`` gets a summary row (clean
        instruments report zero counts — silence is information too).
        """
        counts: dict[str, dict[str, int]] = {
            iid: {"gaps": 0, "gap_bars": 0, "outliers": 0, "bad_prints": 0, "stale_runs": 0}
            for iid in stats.bar_counts
        }
        for finding in stats.findings:
            row = counts.get(finding["instrument_id"])
            if row is None:  # cross-sectional ("*") findings have no instrument row
                continue
            kind = finding["kind"]
            if kind == _KIND_GAP:
                row["gaps"] += 1
                row["gap_bars"] += int(finding["detail"]["n_bars"])
            elif kind == _KIND_OUTLIER:
                row["outliers"] += 1
            elif kind == _KIND_BAD_PRINT:
                row["bad_prints"] += 1
            elif kind == _KIND_STALE:
                row["stale_runs"] += 1
        summaries = tuple(
            InstrumentSummary(
                instrument_id=iid,
                bars=stats.bar_counts[iid],
                gaps=counts[iid]["gaps"],
                gap_bars=counts[iid]["gap_bars"],
                outliers=counts[iid]["outliers"],
                bad_prints=counts[iid]["bad_prints"],
                stale_runs=counts[iid]["stale_runs"],
            )
            for iid in sorted(stats.bar_counts)
        )
        downtime = tuple(f for f in stats.findings if f["kind"] == _KIND_DOWNTIME)
        return cls(
            generated_at=generated_at,
            tf=tf,
            summaries=summaries,
            downtime_windows=downtime,
            findings=tuple(stats.findings),
        )

    @property
    def totals(self) -> dict[str, int]:
        """Run-level totals across all instruments (downtime counts windows)."""
        return {
            "instruments": len(self.summaries),
            "bars": sum(s.bars for s in self.summaries),
            "gaps": sum(s.gaps for s in self.summaries),
            "gap_bars": sum(s.gap_bars for s in self.summaries),
            "outliers": sum(s.outliers for s in self.summaries),
            "bad_prints": sum(s.bad_prints for s in self.summaries),
            "stale_runs": sum(s.stale_runs for s in self.summaries),
            "downtime_windows": len(self.downtime_windows),
        }

    @property
    def worst_severity(self) -> Severity:
        """Highest severity across findings (``info`` for a clean run) — the CLI
        exit-code gate of dataDesign.md §5.4."""
        worst = 0
        for finding in self.findings:
            worst = max(worst, _SEVERITIES.index(finding["severity"]))
        return _SEVERITIES[worst]

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (``totals`` included for machine consumers; it is
        derived and ignored by :meth:`from_dict`)."""
        return {
            "generated_at": self.generated_at,
            "tf": self.tf.value,
            "summaries": [asdict(s) for s in self.summaries],
            "downtime_windows": [dict(f) for f in self.downtime_windows],
            "findings": [dict(f) for f in self.findings],
            "totals": self.totals,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationReport:
        """Inverse of :meth:`to_dict`; validates finding severities (fail-closed
        on unknown values rather than smuggling them through)."""
        return cls(
            generated_at=int(payload["generated_at"]),
            tf=Timeframe(payload["tf"]),
            summaries=tuple(
                InstrumentSummary(
                    instrument_id=str(s["instrument_id"]),
                    bars=int(s["bars"]),
                    gaps=int(s["gaps"]),
                    gap_bars=int(s["gap_bars"]),
                    outliers=int(s["outliers"]),
                    bad_prints=int(s["bad_prints"]),
                    stale_runs=int(s["stale_runs"]),
                )
                for s in payload["summaries"]
            ),
            downtime_windows=tuple(_as_finding(f) for f in payload["downtime_windows"]),
            findings=tuple(_as_finding(f) for f in payload["findings"]),
        )

    def persist(self, out_dir: Path) -> Path:
        """Write the JSON artifact to ``out_dir/<utc-ts>.json`` (dir created if
        missing); returns the written path. Naming derives from ``generated_at``
        so one run maps to one stable artifact."""
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = from_ms(self.generated_at).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ValidationReport:
        """Read a persisted artifact back into an equal :class:`ValidationReport`."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ rendering

    def render_text(self) -> str:
        """Human-readable table: per-instrument counts, downtime windows, totals."""
        lines = [
            f"quality report {_iso(self.generated_at)}  tf={self.tf.value}  "
            f"worst={self.worst_severity}",
            f"{'instrument':<30} {'bars':>8} {'gaps':>5} {'gap_bars':>8} "
            f"{'outliers':>8} {'bad_prints':>10} {'stale_runs':>10}",
        ]
        lines.extend(
            f"{s.instrument_id:<30} {s.bars:>8} {s.gaps:>5} {s.gap_bars:>8} "
            f"{s.outliers:>8} {s.bad_prints:>10} {s.stale_runs:>10}"
            for s in self.summaries
        )
        lines.append(f"exchange downtime: {len(self.downtime_windows)} window(s)")
        lines.extend(
            f"  {_iso(w['ts_start'])} .. {_iso(w['ts_end'])} "
            f"({w['detail']['n_bars']} bars, {w['detail']['n_affected']} instruments)"
            for w in self.downtime_windows
        )
        totals = self.totals
        lines.append("totals: " + "  ".join(f"{k}={v}" for k, v in totals.items()))
        return "\n".join(lines)


def _as_finding(raw: Mapping[str, Any]) -> Finding:
    """Validate and coerce a JSON mapping into a :class:`Finding`."""
    severity = str(raw["severity"])
    if severity not in _SEVERITIES:
        raise ValueError(f"unknown finding severity {severity!r} (expected {_SEVERITIES})")
    return Finding(
        kind=str(raw["kind"]),
        instrument_id=str(raw["instrument_id"]),
        ts_start=int(raw["ts_start"]),
        ts_end=int(raw["ts_end"]),
        severity=severity,
        detail=dict(raw["detail"]),
    )
