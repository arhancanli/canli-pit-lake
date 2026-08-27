"""Tests for alphaforge.config.settings: base.yaml fidelity, profile overlay
deep-merge, AF_* env precedence, path resolution, and validator contracts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from alphaforge.config.settings import (
    DataCfg,
    PortfolioCfg,
    RiskCfg,
    Settings,
    SignalsCfg,
    UniverseCfg,
    load_settings,
)
from alphaforge.core.errors import ConfigError
from alphaforge.core.time import Timeframe, parse_utc
from alphaforge.core.types import AssetClass

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_af_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip any ambient AF_* env vars so tests see only what they set."""
    for key in list(os.environ):
        if key.startswith("AF_"):
            monkeypatch.delenv(key)
    yield


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Minimal config tree: a base.yaml exercising deep-merge targets."""
    _write(
        tmp_path / "configs" / "base.yaml",
        """
data:
  exchange: binanceusdm
  timeframe: "1h"
portfolio:
  w_max: 0.15
  gross_max: 1.0
live:
  paper: true
""",
    )
    return tmp_path


# ---------------------------------------------------------------- base.yaml


class TestLoadBase:
    def test_repo_base_yaml_loads_every_section(self) -> None:
        s = load_settings(root=REPO_ROOT)
        assert s.data.exchange == "binanceusdm"
        assert s.data.timeframe is Timeframe.H1
        assert s.data.asset_class is AssetClass.CRYPTO_PERP
        assert s.data.ingest_grace_ms == 30_000
        assert s.universe.size == 20
        assert s.universe.rank_window_days == 30
        assert s.universe.rebalance == "monthly"
        assert (s.universe.entry_rank, s.universe.exit_rank) == (16, 26)
        assert s.signals.horizon_bars == 72
        assert s.signals.ic_target == 0.02
        assert s.costs.perp_maker_bps == 2.0
        assert s.costs.perp_taker_bps == 5.0
        assert s.costs.default_half_spread_bps == 2.5
        assert s.costs.impact_coef == 1.0
        assert s.costs.latency_addon_bps == 2.0
        assert s.portfolio.risk_aversion == 7.0
        assert s.portfolio.gross_max == 1.0
        assert s.portfolio.net_max == 0.5
        assert s.portfolio.w_max == 0.15
        assert s.portfolio.turnover_max == 0.10
        assert s.portfolio.vol_target_ann == 0.15
        assert s.portfolio.vol_scale_max == 1.5
        assert s.portfolio.no_trade_band == 0.0010
        assert s.risk.max_order_adv_frac == 0.01
        assert s.risk.dd_halve_gross == -0.10
        assert s.risk.dd_flat_halt == -0.15
        assert s.risk.var_confidence == 0.99
        assert s.live.cycle_grace_s == 90
        assert s.live.tick_sleep_s == 45
        assert s.live.paper is True

    def test_code_defaults_match_repo_base_yaml(self) -> None:
        """Field defaults and base.yaml must agree (two sources of truth)."""
        from_yaml = load_settings(root=REPO_ROOT)
        from_defaults = Settings()
        exclude = {"paths"}  # paths are root-resolved by load_settings
        dump_yaml = from_yaml.model_dump(exclude=exclude)
        dump_defaults = from_defaults.model_dump(exclude=exclude)
        assert dump_yaml == dump_defaults

    def test_paths_resolved_absolute_against_root(self) -> None:
        s = load_settings(root=REPO_ROOT)
        assert s.paths.lake_dir == REPO_ROOT / "data" / "lake"
        assert s.paths.features_dir == REPO_ROOT / "data" / "features"
        assert s.paths.quality_dir == REPO_ROOT / "data" / "quality"
        assert s.paths.predictions_dir == REPO_ROOT / "data" / "predictions"
        assert s.paths.artifacts_dir == REPO_ROOT / "artifacts"
        assert s.paths.var_dir == REPO_ROOT / "var"
        assert all(getattr(s.paths, n).is_absolute() for n in type(s.paths).model_fields)

    def test_backfill_start_ms(self) -> None:
        s = load_settings(root=REPO_ROOT)
        assert s.data.backfill_start_ms == 1_577_836_800_000
        assert s.data.backfill_start_ms == parse_utc("2020-01-01T00:00:00Z")

    def test_inferred_root_matches_explicit(self) -> None:
        assert load_settings() == load_settings(root=REPO_ROOT)

    def test_missing_base_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="base config not found"):
            load_settings(root=tmp_path)


# ------------------------------------------------------------ profile merge


class TestProfileOverlay:
    def test_overlay_wins_and_deep_merges(self, tmp_root: Path) -> None:
        _write(
            tmp_root / "configs" / "paper.yaml",
            """
portfolio:
  w_max: 0.12
live:
  paper: false
""",
        )
        s = load_settings(profile="paper", root=tmp_root)
        assert s.portfolio.w_max == 0.12  # overlay wins
        assert s.portfolio.gross_max == 1.0  # sibling key from base survives
        assert s.data.exchange == "binanceusdm"  # untouched section survives
        assert s.live.paper is False

    def test_missing_profile_raises(self, tmp_root: Path) -> None:
        with pytest.raises(ConfigError, match="profile config not found"):
            load_settings(profile="nope", root=tmp_root)

    def test_yaml_unquoted_timestamp_coerced(self, tmp_root: Path) -> None:
        _write(
            tmp_root / "configs" / "ts.yaml",
            "data:\n  backfill_start: 2021-06-01T00:00:00Z\n",
        )
        s = load_settings(profile="ts", root=tmp_root)
        assert s.data.backfill_start_ms == parse_utc("2021-06-01T00:00:00Z")


# ------------------------------------------------------------- env override


class TestEnvOverride:
    def test_env_wins_over_base_and_profile(
        self, tmp_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_root / "configs" / "paper.yaml", "portfolio:\n  w_max: 0.12\n")
        # 0.14 wins over the overlay's 0.12 and base's 0.15, and stays <= the
        # risk hard cap (0.15) so the cross-section validator is satisfied.
        monkeypatch.setenv("AF_PORTFOLIO__W_MAX", "0.14")
        s = load_settings(profile="paper", root=tmp_root)
        assert s.portfolio.w_max == 0.14
        assert s.portfolio.gross_max == 1.0  # env override is leaf-level, not section-level

    def test_env_bool_override(self, tmp_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AF_LIVE__PAPER", "false")
        s = load_settings(root=tmp_root)
        assert s.live.paper is False

    def test_env_relative_path_resolved_against_root(
        self, tmp_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AF_PATHS__LAKE_DIR", "elsewhere/lake")
        s = load_settings(root=tmp_root)
        assert s.paths.lake_dir == tmp_root.resolve() / "elsewhere" / "lake"

    def test_env_absolute_path_passes_through(
        self, tmp_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AF_PATHS__LAKE_DIR", "/abs/lake")
        s = load_settings(root=tmp_root)
        assert s.paths.lake_dir == Path("/abs/lake")

    def test_env_validation_still_applies(
        self, tmp_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AF_SIGNALS__HORIZON_BARS", "0")
        with pytest.raises(ConfigError):
            load_settings(root=tmp_root)


# ------------------------------------------------------------- unknown keys


class TestUnknownKeys:
    def test_unknown_top_level_section(self, tmp_root: Path) -> None:
        _write(tmp_root / "configs" / "bad.yaml", "quantum:\n  flux: 1\n")
        with pytest.raises(ConfigError, match="quantum"):
            load_settings(profile="bad", root=tmp_root)

    def test_unknown_nested_key(self, tmp_root: Path) -> None:
        _write(tmp_root / "configs" / "bad.yaml", "universe:\n  sizee: 20\n")
        with pytest.raises(ConfigError, match="sizee"):
            load_settings(profile="bad", root=tmp_root)

    def test_non_mapping_yaml(self, tmp_root: Path) -> None:
        _write(tmp_root / "configs" / "bad.yaml", "- a\n- b\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_settings(profile="bad", root=tmp_root)


# --------------------------------------------------------------- validators


class TestValidators:
    def test_dd_ordering_violation(self) -> None:
        with pytest.raises(ValidationError, match="dd_flat_halt"):
            RiskCfg(dd_halve_gross=-0.15, dd_flat_halt=-0.10)

    def test_dd_must_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            RiskCfg(dd_halve_gross=0.10, dd_flat_halt=-0.15)

    def test_var_confidence_open_interval(self) -> None:
        with pytest.raises(ValidationError):
            RiskCfg(var_confidence=1.0)
        with pytest.raises(ValidationError):
            RiskCfg(var_confidence=0.0)

    def test_max_order_adv_frac_sqrt_law_guard(self) -> None:
        with pytest.raises(ValidationError):
            RiskCfg(max_order_adv_frac=0.10)

    def test_entry_rank_must_be_below_exit_rank(self) -> None:
        with pytest.raises(ValidationError, match="entry_rank"):
            UniverseCfg(entry_rank=26, exit_rank=16)
        with pytest.raises(ValidationError, match="entry_rank"):
            UniverseCfg(entry_rank=16, exit_rank=16)

    def test_horizon_bars_positive(self) -> None:
        with pytest.raises(ValidationError):
            SignalsCfg(horizon_bars=0)

    def test_w_max_fraction_range(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioCfg(w_max=1.5)
        with pytest.raises(ValidationError):
            PortfolioCfg(w_max=0.0)

    def test_net_max_le_gross_max(self) -> None:
        with pytest.raises(ValidationError, match="net_max"):
            PortfolioCfg(gross_max=1.0, net_max=1.2)

    def test_asset_class_defaults_to_crypto_perp(self) -> None:
        # The byte-identity default: an un-set asset_class is the crypto perp sleeve.
        assert DataCfg().asset_class is AssetClass.CRYPTO_PERP

    def test_asset_class_accepts_equity(self) -> None:
        assert DataCfg(asset_class="equity").asset_class is AssetClass.EQUITY

    def test_backfill_start_naive_rejected(self) -> None:
        with pytest.raises(ValidationError, match="offset"):
            DataCfg(backfill_start="2020-01-01T00:00:00")

    def test_backfill_start_garbage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataCfg(backfill_start="not-a-timestamp")

    def test_validator_violation_via_yaml_is_config_error(self, tmp_root: Path) -> None:
        _write(
            tmp_root / "configs" / "bad.yaml",
            "risk:\n  dd_halve_gross: -0.15\n  dd_flat_halt: -0.10\n",
        )
        with pytest.raises(ConfigError, match="dd_flat_halt"):
            load_settings(profile="bad", root=tmp_root)

    def test_sections_are_frozen(self) -> None:
        s = Settings()
        with pytest.raises(ValidationError):
            s.portfolio.w_max = 0.5  # type: ignore[misc]

    def test_defaults_satisfy_risk_dominates_portfolio(self) -> None:
        # The shipped defaults have equal caps (risk == portfolio); the
        # cross-section validator uses >= so equality loads fine.
        s = Settings()
        assert s.risk.max_position_frac >= s.portfolio.w_max
        assert s.risk.max_gross >= s.portfolio.gross_max
        assert s.risk.max_net >= s.portfolio.net_max
        assert load_settings(root=REPO_ROOT).risk.max_gross >= Settings().portfolio.gross_max

    def test_risk_w_max_below_portfolio_w_max_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"max_position_frac.*must be >="):
            Settings(
                risk=RiskCfg(max_position_frac=0.10),
                portfolio=PortfolioCfg(w_max=0.15),
            )

    def test_risk_gross_below_portfolio_gross_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"max_gross.*must be >="):
            Settings(
                risk=RiskCfg(max_gross=0.8),
                portfolio=PortfolioCfg(gross_max=1.0),
            )

    def test_risk_net_below_portfolio_net_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"max_net.*must be >="):
            Settings(
                risk=RiskCfg(max_net=0.3),
                portfolio=PortfolioCfg(net_max=0.5),
            )

    def test_cross_section_violation_via_yaml_is_config_error(self, tmp_root: Path) -> None:
        _write(
            tmp_root / "configs" / "bad.yaml",
            "risk:\n  max_position_frac: 0.10\nportfolio:\n  w_max: 0.15\n",
        )
        with pytest.raises(ConfigError, match="max_position_frac"):
            load_settings(profile="bad", root=tmp_root)
