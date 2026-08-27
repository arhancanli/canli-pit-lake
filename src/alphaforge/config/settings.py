"""Typed application settings for AlphaForge (pydantic-settings v2).

Single configuration entry point. Per the buildability ruling, ALL settings
sections (paths/data/universe/signals/costs/portfolio/risk/live) are defined
here in Phase 1 so later phases never touch config plumbing. Field names and
defaults mirror ``configs/base.yaml`` exactly — that file is the source of
truth; the Field defaults below exist so ``Settings()`` is constructible in
tests without any YAML.

Precedence (low -> high), implemented by :func:`load_settings`:

1. code defaults (``Field(default=...)`` below)
2. ``configs/base.yaml``
3. ``configs/{profile}.yaml`` — deep-merged over base (nested dicts merge,
   scalars/lists replace)
4. ``AF_*`` environment variables, nested delimiter ``__``
   (e.g. ``AF_PORTFOLIO__W_MAX=0.2``) — these win over everything

``.env``/secret-file sources are intentionally disabled: config is YAML + env
only, deterministic for tests. Unknown keys at any level are a hard error
(``extra="forbid"``) surfaced as :class:`~alphaforge.core.errors.ConfigError`.

Units convention: ``*_bps`` = basis points (1e-4 fractional), ``*_frac`` and
exposure/weight limits = fractions of equity, ``*_ms`` = milliseconds,
``*_s`` = seconds, ``*_ann`` = annualized. Drawdown thresholds are negative
fractions (e.g. ``-0.10`` = -10% from high-water mark).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from alphaforge.core.errors import ConfigError, NaiveDatetimeError
from alphaforge.core.time import Ms, Timeframe, parse_utc
from alphaforge.core.types import AssetClass

__all__ = [
    "CostsCfg",
    "DataCfg",
    "LiveCfg",
    "PathsCfg",
    "PortfolioCfg",
    "QualityCfg",
    "RiskCfg",
    "Settings",
    "SignalsCfg",
    "UniverseCfg",
    "load_settings",
]

_SECTION_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PathsCfg(BaseModel):
    """Filesystem layout. Values in YAML may be relative; :func:`load_settings`
    resolves every field absolute against the repo root before returning, so
    consumers may assume absolute paths."""

    model_config = _SECTION_CONFIG

    lake_dir: Path = Path("data/lake")
    features_dir: Path = Path("data/features")
    quality_dir: Path = Path("data/quality")
    predictions_dir: Path = Path("data/predictions")
    artifacts_dir: Path = Path("artifacts")
    var_dir: Path = Path("var")


class DataCfg(BaseModel):
    """Ingestion contract: which exchange/timeframe to pull and from when.

    ``backfill_start`` is stored as the raw ISO-8601 string exactly as written
    in YAML (round-trips cleanly, dumps reproducibly); it is validated at
    construction (must parse and carry an explicit UTC offset) and exposed as
    epoch-ms via the :attr:`backfill_start_ms` property. This is the
    "validator-checked string + converting property" choice — no second
    mutable copy of the timestamp exists on the model.

    ``asset_class`` selects the sleeve (anchor timeframe + trading calendar) the
    engine spine values on; the default ``CRYPTO_PERP`` resolves to the H1 + 24/7
    sleeve, preserving the current behavior byte-for-byte. ``timeframe`` stays the
    ingest/anchor TF (default H1 unchanged); the sleeve is
    :func:`~alphaforge.config.sleeve.sleeve_for` of ``asset_class``.
    """

    model_config = _SECTION_CONFIG

    exchange: str = Field(default="binanceusdm", min_length=1)
    timeframe: Timeframe = Timeframe.H1
    asset_class: AssetClass = AssetClass.CRYPTO_PERP
    backfill_start: str = "2020-01-01T00:00:00Z"
    ingest_grace_ms: int = Field(default=30_000, ge=0)

    @field_validator("backfill_start", mode="before")
    @classmethod
    def _coerce_yaml_datetime(cls, v: object) -> object:
        """yaml.safe_load parses unquoted timestamps into datetime; normalize
        back to an ISO string (offset preserved) so the str field accepts it."""
        if isinstance(v, datetime):
            return v.isoformat().replace("+00:00", "Z")
        return v

    @field_validator("backfill_start")
    @classmethod
    def _validate_backfill_start(cls, v: str) -> str:
        try:
            parse_utc(v)
        except NaiveDatetimeError as exc:
            raise ValueError(
                f"backfill_start must carry an explicit UTC offset (e.g. trailing 'Z'): {exc}"
            ) from exc
        return v

    @property
    def backfill_start_ms(self) -> Ms:
        """Backfill start as UTC epoch-milliseconds (the API-boundary type)."""
        return parse_utc(self.backfill_start)


class UniverseCfg(BaseModel):
    """Universe selection with rank hysteresis: an instrument ENTERS when it
    ranks at or above ``entry_rank`` and LEAVES only when it falls below
    ``exit_rank`` (entry_rank < exit_rank, so membership does not churn on
    rank noise around the boundary)."""

    model_config = _SECTION_CONFIG

    size: int = Field(default=20, gt=0)
    rank_window_days: int = Field(default=30, gt=0)
    rebalance: Literal["daily", "weekly", "monthly"] = "monthly"
    entry_rank: int = Field(default=16, gt=0)
    exit_rank: int = Field(default=26, gt=0)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> Self:
        if self.entry_rank >= self.exit_rank:
            raise ValueError(
                f"entry_rank ({self.entry_rank}) must be < exit_rank ({self.exit_rank}) "
                "for hysteresis to dampen membership churn"
            )
        return self


class QualityCfg(BaseModel):
    """Data-quality engine thresholds (dataDesign.md §5.2; consumed by
    ``alphaforge.data.quality``). Quality flags ANNOTATE bars, never alter prices —
    these knobs tune detection sensitivity only.

    - ``ewma_span_bars``: span of the trailing EWMA volatility estimate sigma and of the
      trailing volume-median window, in 1h bars (168 = 7 days).
    - ``outlier_k``: a bar is an outlier when ``|log return| > k*sigma`` (crypto 1h
      returns genuinely reach 5-6sigma in cascades; 8 keeps real crashes unflagged).
    - ``bad_print_vol_frac``: volume disconfirmation — a suspect print has bar volume
      below this fraction of the trailing median volume.
    - ``bad_print_reversion_frac``: reversion — the gap-over print snaps back when
      ``|ln(close_{t+1}/close_{t-1})| < frac·|r_t|``.
    - ``downtime_frac``: a missing hour is exchange downtime when at least this
      fraction of instruments active at that hour are simultaneously missing it.
    - ``stale_run_bars``: minimum run of consecutive identical closes (with trading
      volume) reported as a stuck-feed symptom.
    - ``min_history_bars``: warmup guard — outlier/bad-print conditions need at least
      this many *prior* observations before sigma / median estimates are trusted.
    """

    model_config = _SECTION_CONFIG

    ewma_span_bars: int = Field(default=168, gt=1)
    outlier_k: float = Field(default=8.0, gt=0.0)
    bad_print_vol_frac: float = Field(default=0.25, gt=0.0, lt=1.0)
    bad_print_reversion_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    downtime_frac: float = Field(default=0.8, gt=0.0, le=1.0)
    stale_run_bars: int = Field(default=12, gt=1)
    min_history_bars: int = Field(default=24, ge=2)


class SignalsCfg(BaseModel):
    """Alpha layer constants. ``horizon_bars`` is THE single holding-period
    constant (ruling: alpha horizon == cost-amortization horizon == purge-gap
    floor; never duplicated elsewhere)."""

    model_config = _SECTION_CONFIG

    horizon_bars: int = Field(default=72, gt=0)
    ic_target: float = Field(default=0.02, gt=0.0, lt=1.0)


class CostsCfg(BaseModel):
    """Transaction-cost model parameters, all in basis points except the
    dimensionless square-root-impact coefficient. Consumed only by ``costs/``
    (single import of record — fees are never re-declared elsewhere)."""

    model_config = _SECTION_CONFIG

    perp_maker_bps: float = Field(default=2.0, ge=0.0)
    perp_taker_bps: float = Field(default=5.0, ge=0.0)
    default_half_spread_bps: float = Field(default=2.5, ge=0.0)
    impact_coef: float = Field(default=1.0, ge=0.0)
    latency_addon_bps: float = Field(default=2.0, ge=0.0)
    # US-equity sleeve (consumed only when data.asset_class == EQUITY; configs/equity.yaml).
    # Commission per side (1 bp conservative), the liquid-universe half-spread default
    # (top-200 names; mega-caps ~1bp, mid-caps more, so 3 bps is conservative), and the annual
    # short-borrow (general-collateral) rate charged per holding day on short equity notional.
    equity_commission_bps: float = Field(default=1.0, ge=0.0)
    equity_half_spread_bps: float = Field(default=3.0, ge=0.0)
    equity_borrow_bps_annual: float = Field(default=50.0, ge=0.0)


class PortfolioCfg(BaseModel):
    """Optimizer limits, as fractions of equity. ``gross_max``/``net_max``/
    ``w_max`` are the optimizer's soft caps (the risk engine holds separate,
    looser hard caps). ``no_trade_band`` is the per-asset |Δw| below which an
    order is suppressed."""

    model_config = _SECTION_CONFIG

    risk_aversion: float = Field(default=7.0, gt=0.0)
    gross_max: float = Field(default=1.0, gt=0.0)
    net_max: float = Field(default=0.5, ge=0.0)
    w_max: float = Field(default=0.15, gt=0.0, le=1.0)
    turnover_max: float = Field(default=0.10, gt=0.0, le=1.0)
    vol_target_ann: float = Field(default=0.15, gt=0.0, le=1.0)
    vol_scale_max: float = Field(default=1.5, ge=1.0)
    no_trade_band: float = Field(default=0.0010, ge=0.0, lt=0.1)
    #: L/S rank-allocator breadth (names per side). None -> v1 default min(10, n//4); a
    #: larger value (e.g. 50) harvests universe breadth for a higher Sharpe (IR ≈ IC·√N).
    rank_top_k: int | None = Field(default=None, ge=1)
    #: Enforce exact dollar-neutrality (Σw=0) on the L/S rank book (each leg equal gross).
    #: Default off -> byte-identical. Removes the inverse-vol net-dollar tilt.
    dollar_neutral: bool = Field(default=False)

    @model_validator(mode="after")
    def _check_limit_ordering(self) -> Self:
        if self.net_max > self.gross_max:
            raise ValueError(f"net_max ({self.net_max}) must be <= gross_max ({self.gross_max})")
        if self.w_max > self.gross_max:
            raise ValueError(f"w_max ({self.w_max}) must be <= gross_max ({self.gross_max})")
        return self


class RiskCfg(BaseModel):
    """Risk-engine thresholds (execDesign §7; consumed by ``alphaforge.risk``).

    These are the HARD caps enforced by the pre-trade checker — deliberately
    at-or-above the optimizer's soft caps in :class:`PortfolioCfg` so normal
    mark-to-market drift between rebalances never trips them; a trip means a
    sizing bug upstream, not noise.

    Drawdown thresholds are NEGATIVE fractions from the high-water mark and
    must satisfy ``dd_flat_halt < dd_halve_gross < 0``: gross is halved at the
    shallower threshold, the book is flattened and trading halted at the
    deeper one. ``max_order_adv_frac`` <= 0.05 keeps every order inside the
    square-root impact law's validity regime (the cost model raises beyond
    it), with 0.01 the conservative default pre-trade cap.
    ``price_collar_frac`` bounds |decision_price - live close| as a fraction of
    the close (stale-signal protection for market orders).
    ``var_window_days`` is the lookback for historical VaR/CVaR computed on
    DAILY-aggregated returns (leakageCritique finding 29: hourly VaR with
    sqrt(24) iid scaling understates risk under vol clustering).
    ``staleness_max_bars``: no fresh bar within this many bar intervals of now
    ⇒ the staleness breaker forbids trading for the cycle.
    ``flat_cooldown_bars``: how long (in bars) the drawdown ladder stays
    flat-and-halted after a TIER2 full halt before it AUTO-REARMS — resetting
    HWM to the then-current equity and resuming trading from a fresh
    high-water mark. This is the unattended default (identical in backtest and
    the 24/7 paper loop): a flat book marks no positions, so its equity freezes
    and can NEVER recover on its own, which makes a recovery-based auto-rearm
    impossible — the rearm MUST be time-based. Default 336 = 14 days of 1h
    bars. The KillSwitch (§7.3, separate class) remains the ABSORBING manual
    halt for human intervention; this cooldown only governs the ladder's own
    self-recovery.
    """

    model_config = _SECTION_CONFIG

    max_position_frac: float = Field(default=0.15, gt=0.0, le=1.0)
    max_gross: float = Field(default=1.0, gt=0.0)
    max_net: float = Field(default=0.5, ge=0.0)
    max_order_adv_frac: float = Field(default=0.01, gt=0.0, le=0.05)
    #: Clamp reduce-only (de-risking) orders at the 5%-ADV impact-law edge instead of letting them
    #: bypass the cap. OFF by default (the crypto golden master locks the bypass). ON for thin-
    #: universe profiles (equity / managed-futures) where a large flatten on a low-ADV instrument
    #: would otherwise crash the sqrt-impact model — it de-risks fast but stays priceable.
    clamp_reduce_only_adv: bool = Field(default=False)
    price_collar_frac: float = Field(default=0.05, gt=0.0, lt=1.0)
    dd_halve_gross: float = Field(default=-0.10, lt=0.0, gt=-1.0)
    dd_flat_halt: float = Field(default=-0.15, lt=0.0, gt=-1.0)
    flat_cooldown_bars: int = Field(default=336, gt=0)
    var_confidence: float = Field(default=0.99, gt=0.0, lt=1.0)
    var_window_days: int = Field(default=365, gt=1)
    staleness_max_bars: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def _check_orderings(self) -> Self:
        if self.dd_flat_halt >= self.dd_halve_gross:
            raise ValueError(
                f"dd_flat_halt ({self.dd_flat_halt}) must be < dd_halve_gross "
                f"({self.dd_halve_gross}) < 0: the flatten/halt threshold is the deeper drawdown"
            )
        if self.max_net > self.max_gross:
            raise ValueError(f"max_net ({self.max_net}) must be <= max_gross ({self.max_gross})")
        if self.max_position_frac > self.max_gross:
            raise ValueError(
                f"max_position_frac ({self.max_position_frac}) must be <= "
                f"max_gross ({self.max_gross})"
            )
        return self


class LiveCfg(BaseModel):
    """Live-loop timing. The loop wakes at bar close + ``cycle_grace_s`` (one
    timer ruling: the loop owns the schedule and invokes ingest), re-checking
    wall clock every ``tick_sleep_s`` because macOS monotonic clocks freeze in
    sleep. ``paper=True`` routes orders to the PaperBroker; flipping to real
    money is config, not code."""

    model_config = _SECTION_CONFIG

    cycle_grace_s: int = Field(default=90, gt=0)
    tick_sleep_s: int = Field(default=45, gt=0)
    paper: bool = True


class Settings(BaseSettings):
    """Aggregate of all config sections.

    Environment overrides use prefix ``AF_`` and nested delimiter ``__``
    (``AF_RISK__VAR_CONFIDENCE=0.95``); env beats init kwargs (i.e. beats the
    YAML layers fed in by :func:`load_settings`). Direct construction
    ``Settings()`` yields pure code defaults (+ any AF_ env vars) — fine for
    tests; production code goes through :func:`load_settings` so paths are
    resolved and YAML layers apply.
    """

    model_config = SettingsConfigDict(
        env_prefix="AF_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    paths: PathsCfg = Field(default_factory=PathsCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    signals: SignalsCfg = Field(default_factory=SignalsCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    portfolio: PortfolioCfg = Field(default_factory=PortfolioCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    live: LiveCfg = Field(default_factory=LiveCfg)

    @model_validator(mode="after")
    def _check_risk_dominates_portfolio(self) -> Self:
        """Risk-engine HARD caps must be >= the optimizer's SOFT caps.

        The narrative across :class:`PortfolioCfg`, :class:`RiskCfg`, and the
        optimizer docstrings is "the risk engine holds separate, looser hard
        caps" so normal mark-to-market drift between rebalances never trips a
        pre-trade check; a trip then means a sizing bug upstream, not noise.
        Nothing enforced that ordering across the two sections, so a config
        where the risk cap sits BELOW the optimizer cap would size books the
        live pre-trade checker then rejects -- a silent backtest-vs-live
        divergence surface. We assert it here (``>=`` so the equal defaults in
        ``configs/base.yaml`` remain valid; the design intends ``>`` as a drift
        buffer but equality is admissible and must not be an error).
        """
        pairs = (
            (
                "risk.max_gross",
                self.risk.max_gross,
                "portfolio.gross_max",
                self.portfolio.gross_max,
            ),
            ("risk.max_net", self.risk.max_net, "portfolio.net_max", self.portfolio.net_max),
            (
                "risk.max_position_frac",
                self.risk.max_position_frac,
                "portfolio.w_max",
                self.portfolio.w_max,
            ),
        )
        for hard_name, hard_val, soft_name, soft_val in pairs:
            if hard_val < soft_val:
                raise ValueError(
                    f"{hard_name} ({hard_val}) must be >= {soft_name} ({soft_val}): the "
                    "risk engine's hard caps cannot sit below the optimizer's soft caps, "
                    "or the optimizer would size books the live pre-trade checker rejects "
                    "(backtest-vs-live divergence)."
                )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env vars beat init kwargs (the merged YAML); dotenv/secret-file
        sources disabled. pydantic-settings deep-merges across sources, so a
        single AF_SECTION__FIELD var overrides one leaf, not the section."""
        return (env_settings, init_settings)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: nested mappings merge key-wise, anything else in
    ``override`` (scalars, lists) replaces the ``base`` value. Inputs are not
    mutated."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load ``path`` with yaml.safe_load and require a string-keyed mapping at
    top level (empty file -> {}). Raises ConfigError on parse failure or wrong
    top-level shape."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    for key in raw:
        if not isinstance(key, str):
            raise ConfigError(f"{path}: top-level keys must be strings, got {key!r}")
    return cast(dict[str, Any], raw)


def _infer_root() -> Path:
    """Repo root inferred from this file's location (src/ layout:
    <root>/src/alphaforge/config/settings.py), verified by the presence of
    configs/base.yaml. Works for editable installs; for site-packages
    installs callers must pass ``root`` explicitly."""
    root = Path(__file__).resolve().parents[3]
    if (root / "configs" / "base.yaml").is_file():
        return root
    raise ConfigError(
        f"cannot infer repo root: no configs/base.yaml under {root}; "
        "pass load_settings(root=...) explicitly"
    )


def _resolve_paths(paths: PathsCfg, root: Path) -> PathsCfg:
    """Return a copy of ``paths`` with every relative path made absolute
    against ``root``; already-absolute paths pass through unchanged."""
    values: dict[str, Path] = {}
    for name in PathsCfg.model_fields:
        p: Path = getattr(paths, name)
        values[name] = p if p.is_absolute() else root / p
    return PathsCfg(**values)


def load_settings(profile: str | None = None, root: Path | None = None) -> Settings:
    """Build validated :class:`Settings` from layered sources.

    Reads ``<root>/configs/base.yaml``, deep-merges ``<root>/configs/
    {profile}.yaml`` over it if ``profile`` is given, then lets ``AF_*`` env
    vars win over both. All :class:`PathsCfg` fields are resolved absolute
    against ``root`` (default: repo root inferred from the package location)
    AFTER validation, so env-supplied relative paths resolve too.

    Raises ConfigError on: missing base.yaml or profile file, malformed YAML,
    unknown keys at any level, or any field/cross-field validation failure.
    """
    resolved_root = _infer_root() if root is None else root.resolve()
    configs_dir = resolved_root / "configs"
    base_path = configs_dir / "base.yaml"
    if not base_path.is_file():
        raise ConfigError(f"base config not found: {base_path}")
    merged = _read_yaml_mapping(base_path)
    if profile is not None:
        profile_path = configs_dir / f"{profile}.yaml"
        if not profile_path.is_file():
            raise ConfigError(f"profile config not found: {profile_path}")
        merged = _deep_merge(merged, _read_yaml_mapping(profile_path))
    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings (base={base_path}, profile={profile}): {exc}") from exc
    return settings.model_copy(update={"paths": _resolve_paths(settings.paths, resolved_root)})
