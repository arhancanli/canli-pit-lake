"""Unit tests for alphaforge.data.quality.report (ValidationReport).

Covers: aggregation of findings into per-instrument summaries and totals, downtime
window extraction, worst-severity derivation, exact JSON persist/load round-trip,
artifact naming, text rendering, and fail-closed severity validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaforge.core.time import Timeframe
from alphaforge.data.quality.checks import Finding, QualityRunStats, Severity
from alphaforge.data.quality.report import InstrumentSummary, ValidationReport

BTC = "BINANCE:PERP:BTCUSDT"
ETH = "BINANCE:PERP:ETHUSDT"

HOUR = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z
GENERATED_AT = 1_704_240_000_000  # 2024-01-03T00:00:00Z


def finding(
    kind: str,
    instrument_id: str,
    *,
    bar: int = 0,
    n_bars: int = 1,
    severity: Severity = "warn",
    detail: dict[str, object] | None = None,
) -> Finding:
    base: dict[str, object] = {"n_bars": n_bars}
    if detail:
        base.update(detail)
    return Finding(
        kind=kind,
        instrument_id=instrument_id,
        ts_start=T0 + bar * HOUR,
        ts_end=T0 + (bar + n_bars) * HOUR,
        severity=severity,
        detail=base,
    )


def make_stats(findings: list[Finding] | None = None) -> QualityRunStats:
    found = (
        [
            finding("gap", BTC, bar=30, n_bars=3),
            finding("gap", BTC, bar=60, n_bars=5, severity="error"),
            finding("outlier_return", BTC, bar=50, detail={"z": 12.5}),
            finding("bad_print", BTC, bar=50, detail={"volume": 1.0}),
            finding("stale_close", ETH, bar=10, n_bars=13),
            finding(
                "exchange_downtime",
                "*",
                bar=20,
                n_bars=3,
                severity="info",
                detail={"n_affected": 4, "affected": [BTC, ETH], "min_frac": 0.8},
            ),
        ]
        if findings is None
        else findings
    )
    return QualityRunStats(
        instruments=2,
        bars_scanned=157,
        rows_reflagged=4,
        findings=found,
        bar_counts={BTC: 77, ETH: 80},
    )


def build_report(findings: list[Finding] | None = None) -> ValidationReport:
    return ValidationReport.build(make_stats(findings), tf=Timeframe.H1, generated_at=GENERATED_AT)


class TestBuild:
    def test_per_instrument_summary(self) -> None:
        report = build_report()
        assert report.summaries == (
            InstrumentSummary(
                instrument_id=BTC,
                bars=77,
                gaps=2,
                gap_bars=8,
                outliers=1,
                bad_prints=1,
                stale_runs=0,
            ),
            InstrumentSummary(
                instrument_id=ETH,
                bars=80,
                gaps=0,
                gap_bars=0,
                outliers=0,
                bad_prints=0,
                stale_runs=1,
            ),
        )

    def test_totals(self) -> None:
        assert build_report().totals == {
            "instruments": 2,
            "bars": 157,
            "gaps": 2,
            "gap_bars": 8,
            "outliers": 1,
            "bad_prints": 1,
            "stale_runs": 1,
            "downtime_windows": 1,
        }

    def test_clean_run_keeps_zeroed_instrument_rows(self) -> None:
        report = build_report(findings=[])
        assert [s.instrument_id for s in report.summaries] == [BTC, ETH]
        assert all(
            (s.gaps, s.gap_bars, s.outliers, s.bad_prints, s.stale_runs) == (0, 0, 0, 0, 0)
            for s in report.summaries
        )
        assert report.totals["bars"] == 157

    def test_downtime_windows_extracted_from_findings(self) -> None:
        report = build_report()
        (window,) = report.downtime_windows
        assert window["kind"] == "exchange_downtime"
        assert window["ts_start"] == T0 + 20 * HOUR
        assert window["detail"]["n_affected"] == 4

    def test_worst_severity(self) -> None:
        assert build_report().worst_severity == "error"
        assert build_report(findings=[]).worst_severity == "info"
        assert build_report(findings=[finding("gap", BTC, n_bars=2)]).worst_severity == "warn"


class TestSerialization:
    def test_json_round_trip_is_exact(self, tmp_path: Path) -> None:
        report = build_report()
        path = report.persist(tmp_path / "reports")
        loaded = ValidationReport.load(path)
        assert loaded == report
        assert loaded.to_dict() == report.to_dict()

    def test_persist_creates_dir_and_names_by_generated_at(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "quality" / "reports"
        path = build_report().persist(out_dir)
        assert path.parent == out_dir
        assert path.name == "20240103T000000Z.json"
        assert path.is_file()

    def test_to_dict_includes_derived_totals(self) -> None:
        report = build_report()
        payload = report.to_dict()
        assert payload["totals"] == report.totals
        assert payload["tf"] == "1h"
        assert payload["generated_at"] == GENERATED_AT

    def test_from_dict_rejects_unknown_severity(self) -> None:
        payload = build_report().to_dict()
        payload["findings"][0]["severity"] = "catastrophic"
        with pytest.raises(ValueError, match="catastrophic"):
            ValidationReport.from_dict(payload)


class TestRenderText:
    def test_contains_instruments_downtime_and_totals(self) -> None:
        text = build_report().render_text()
        assert BTC in text
        assert ETH in text
        assert "worst=error" in text
        assert "exchange downtime: 1 window(s)" in text
        assert "2024-01-01T20:00:00Z .. 2024-01-01T23:00:00Z (3 bars, 4 instruments)" in text
        assert "totals:" in text
        assert "gap_bars=8" in text

    def test_clean_report_renders(self) -> None:
        text = build_report(findings=[]).render_text()
        assert "worst=info" in text
        assert "exchange downtime: 0 window(s)" in text
