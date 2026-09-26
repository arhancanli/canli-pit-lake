"""Build the public, implementation-bound Deflated Sharpe calculator contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from alphaforge.validation.dsr import (
    EULER_MASCHERONI,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
DSR_SOURCE = ROOT / "src/alphaforge/validation/dsr.py"
ADMISSION_SOURCE = ROOT / "config/sleeve_admission_contract.json"
OUTPUT = ROOT / "artifacts/engineering/deflated_sharpe_calculator_contract.json"


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _canonical(value: Any) -> bytes:
    # ensure_ascii is left at its default (True), which escapes non-ASCII to \uXXXX.
    # That is the convention every other exporter here uses and the one the public
    # reproduce.py kit recomputes with: 33 published artifacts store their non-ASCII
    # escaped, and this contract was the only one storing raw UTF-8 bytes.
    #
    # It went unnoticed because the two settings produce IDENTICAL bytes for any
    # pure-ASCII document, which every artifact was until this contract cited
    # "Lopez de Prado" with its accent. That single character made the published
    # verification kit report this artifact as FAILED, and since the nightly publish
    # gates on L1, it halted the whole ceremony on 2026-08-23, 08-24 and 08-26.
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _content_hash(document: dict[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "content_hash"}
    return f"sha256:{hashlib.sha256(_canonical(body)).hexdigest()}"


def _vector(
    *,
    vector_id: str,
    observed_sharpe_annualized: float,
    observations: int,
    periods_per_year: int,
    skew: float,
    non_excess_kurtosis: float,
    effective_independent_trials: int,
    cross_trial_sharpe_sd_annualized: float,
) -> dict[str, Any]:
    scale = math.sqrt(periods_per_year)
    sr_per_period = observed_sharpe_annualized / scale
    trial_sd_per_period = cross_trial_sharpe_sd_annualized / scale
    trial_variance_per_period = trial_sd_per_period**2
    benchmark = expected_max_sharpe(effective_independent_trials, trial_variance_per_period)
    variance_term = (
        1.0 - skew * sr_per_period + ((non_excess_kurtosis - 1.0) / 4.0) * sr_per_period**2
    )
    psr = probabilistic_sharpe_ratio(
        sr_per_period,
        observations,
        skew,
        non_excess_kurtosis,
    )
    dsr = probabilistic_sharpe_ratio(
        sr_per_period,
        observations,
        skew,
        non_excess_kurtosis,
        sr_benchmark=benchmark,
    )
    return {
        "id": vector_id,
        "inputs": {
            "observed_sharpe_annualized": observed_sharpe_annualized,
            "observations": observations,
            "periods_per_year": periods_per_year,
            "skew": skew,
            "non_excess_kurtosis": non_excess_kurtosis,
            "effective_independent_trials": effective_independent_trials,
            "cross_trial_sharpe_sd_annualized": cross_trial_sharpe_sd_annualized,
        },
        "outputs": {
            "observed_sharpe_per_period": sr_per_period,
            "cross_trial_sharpe_variance_per_period": trial_variance_per_period,
            "expected_max_sharpe_per_period": benchmark,
            "expected_max_sharpe_annualized": benchmark * scale,
            "psr_against_zero": psr,
            "deflated_sharpe_ratio": dsr,
            "non_normality_variance_term": variance_term,
        },
    }


def build_contract() -> dict[str, Any]:
    admission = json.loads(ADMISSION_SOURCE.read_text(encoding="utf-8"))
    deflation = admission["deflation_policy"]
    contract: dict[str, Any] = {
        "schema": "canli.alphac-deflated-sharpe-calculator-contract.v1",
        "status": "REFERENCE_IMPLEMENTATION_CONTRACT",
        "version": "1.0.0",
        "published_on": "2026-08-26",
        "author": "Arhan Canli",
        "claim_boundary": (
            "This contract reproduces ALPHAC's PSR and DSR arithmetic for supplied inputs. "
            "It does not validate a strategy, prove profitability, estimate current portfolio "
            "performance, or replace the complete admission contract."
        ),
        "periodicity": {
            "formula_unit": "per_period_sharpe",
            "browser_input_unit": "annualized_sharpe",
            "observed_conversion": "sr_per_period = sr_annualized / sqrt(periods_per_year)",
            "dispersion_conversion": (
                "variance_per_period = (sd_annualized / sqrt(periods_per_year))^2"
            ),
            "kurtosis": "non_excess; Gaussian equals 3",
        },
        "formula": {
            "psr": (
                "Phi(((SR - SR*) * sqrt(T - 1)) / sqrt(1 - skew*SR + ((kurtosis - 1)/4)*SR^2))"
            ),
            "expected_max_sharpe": ("sqrt(V[SR]) * ((1-gamma)*PPF(1-1/N) + gamma*PPF(1-1/(N*e)))"),
            "dsr": "PSR(expected_max_sharpe)",
            "constants": {"euler_mascheroni": EULER_MASCHERONI, "e": math.e},
        },
        "input_contract": {
            "observed_sharpe_annualized": {"minimum": -10, "maximum": 10},
            "observations": {"minimum": 2, "maximum": 1000000, "integer": True},
            "periods_per_year": {"minimum": 1, "maximum": 10000},
            "skew": {"minimum": -20, "maximum": 20},
            "non_excess_kurtosis": {"minimum": 1, "maximum": 100},
            "effective_independent_trials": {
                "minimum": 2,
                "maximum": 10000000,
                "integer": True,
            },
            "cross_trial_sharpe_sd_annualized": {"minimum": 0, "maximum": 10},
        },
        "selection_accounting": {
            "trial_unit": deflation["book_selection_unit"],
            "variance_unit": "sample variance of per-period Sharpe across selection identities",
            "warning": (
                "Reducing the trial count or dispersion after observing outcomes flatters DSR. "
                "The selection union must be defined independently of the selected result."
            ),
        },
        "current_policy": {
            "admission_schema": admission["schema"],
            "per_sleeve_dsr": "mandatory_measurement_not_a_universal_gate",
            "incremental_admission": "not_decided_by_dsr_alone",
            "full_union_book_maturity_threshold": deflation["book_maturity_threshold"],
            "threshold_context": (
                "The 0.95 threshold applies to a full-union portfolio-maturity claim. It is not "
                "a per-sleeve gate and a calculator result is not an admission verdict."
            ),
        },
        "source_bindings": {
            "implementation": {
                "path": "src/alphaforge/validation/dsr.py",
                "sha256": _sha256(DSR_SOURCE),
            },
            "admission_contract": {
                "path": "config/sleeve_admission_contract.json",
                "sha256": _sha256(ADMISSION_SOURCE),
            },
        },
        "references": [
            {
                "title": (
                    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest "
                    "Overfitting, and Non-Normality"
                ),
                "authors": "David H. Bailey and Marcos López de Prado",
                "publication": "Journal of Portfolio Management 40(5), 94-107 (2014)",
                "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
                "doi": "10.2139/ssrn.2460551",
            }
        ],
        "test_vectors": [
            _vector(
                vector_id="daily_long_sample_heavy_tail",
                observed_sharpe_annualized=1.5,
                observations=730,
                periods_per_year=365,
                skew=-0.5,
                non_excess_kurtosis=5.0,
                effective_independent_trials=229,
                cross_trial_sharpe_sd_annualized=0.57,
            ),
            _vector(
                vector_id="trading_days_short_search",
                observed_sharpe_annualized=1.0,
                observations=252,
                periods_per_year=252,
                skew=0.0,
                non_excess_kurtosis=3.0,
                effective_independent_trials=10,
                cross_trial_sharpe_sd_annualized=0.5,
            ),
            _vector(
                vector_id="large_search_negative_skew",
                observed_sharpe_annualized=2.2,
                observations=1260,
                periods_per_year=252,
                skew=-1.0,
                non_excess_kurtosis=9.0,
                effective_independent_trials=1000,
                cross_trial_sharpe_sd_annualized=0.75,
            ),
        ],
    }
    contract["content_hash"] = _content_hash(contract)
    return contract


def main() -> None:
    contract = build_contract()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": contract["status"], "content_hash": contract["content_hash"]}))


if __name__ == "__main__":
    main()
