"""Point-in-time normalized OCC option-adjustment observations.

OCC information memos are source documents, not a stable machine-readable terms feed.  A
numeric suffix marks a non-standard option class but does not encode its economics, and a later
memo can revise an already-adjusted class without changing that suffix.  This module therefore
accepts only a strict, reviewed extraction manifest bound to the raw source's SHA-256 digest.

The normalized record preserves effective time separately from source publication and local
availability.  Anticipated, delayed, contingent, unresolved, singly reviewed, or cross-vendor
disagreeing observations remain visible for audit but fail closed for replay qualification.  No
PDF prose is guessed and no symbol suffix is treated as a deliverable definition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from typing import Final, Protocol
from urllib.parse import parse_qs, urlparse

from alphaforge.core.errors import LookaheadError
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms
from alphaforge.core.types import MarketType
from alphaforge.data.ingest.occ_memo_archive import OCCMemoSnapshot

__all__ = [
    "AdjustmentAuthority",
    "AdjustmentState",
    "NormalizedAdjustmentAsset",
    "NormalizedAdjustmentCash",
    "OptionAdjustmentChain",
    "OptionAdjustmentObservation",
    "OptionAdjustmentReconciliation",
    "OptionAdjustmentReconciliationError",
    "OptionAdjustmentSourceLineageError",
    "SettlementState",
    "StaticOptionAdjustmentProvider",
    "UnresolvedAdjustmentComponent",
    "parse_option_adjustment_observation",
    "reconcile_option_adjustments",
    "verify_occ_observation_source",
]

_SCHEMA: Final[str] = "alphaforge.option-adjustment-observation.v1"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_OPTION_ROOT_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{1,8}")
_CURRENCY_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{3,12}")


class AdjustmentAuthority(StrEnum):
    """Authority represented by one independently captured observation."""

    OCC_MEMO = "occ_memo"
    VENDOR_TERMS = "vendor_terms"


class AdjustmentState(StrEnum):
    """Source-document maturity; only ``FINAL`` can qualify for replay."""

    ANTICIPATED = "anticipated"
    FINAL = "final"
    SUPERSEDED = "superseded"


class SettlementState(StrEnum):
    """Settlement state disclosed by the source."""

    STANDARD = "standard"
    DELAYED = "delayed"
    CASH_ONLY = "cash_only"


def _positive_decimal(name: str, value: Decimal) -> None:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _root(name: str, value: str) -> None:
    if _OPTION_ROOT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 1-8 uppercase alphanumerics, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalizedAdjustmentAsset:
    """One exact non-cash deliverable component per option contract."""

    instrument_id: str
    quantity_per_contract: Decimal
    source_identifier: str
    settlement_delayed: bool = False

    def __post_init__(self) -> None:
        _, market_type, _ = SymbolMapper.parse_instrument_id(self.instrument_id)
        if market_type is not MarketType.CASH:
            raise ValueError("adjustment assets must use canonical CASH instrument ids")
        _positive_decimal("quantity_per_contract", self.quantity_per_contract)
        if not self.source_identifier.strip():
            raise ValueError("asset source_identifier cannot be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalizedAdjustmentCash:
    """One exact fixed cash component per option contract."""

    currency: str
    amount_per_contract: Decimal
    settlement_delayed: bool = False

    def __post_init__(self) -> None:
        if _CURRENCY_RE.fullmatch(self.currency) is None:
            raise ValueError(
                f"currency must be 3-12 uppercase alphanumerics, got {self.currency!r}"
            )
        _positive_decimal("amount_per_contract", self.amount_per_contract)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnresolvedAdjustmentComponent:
    """A source-disclosed component that cannot yet be represented as fixed economics."""

    component: str
    reason: str

    def __post_init__(self) -> None:
        if not self.component.strip() or not self.reason.strip():
            raise ValueError("unresolved component and reason cannot be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionAdjustmentObservation:
    """One content-addressed, reviewed extraction of an option-class adjustment."""

    authority: AdjustmentAuthority
    source_id: str
    source_url: str
    source_sha256: str
    contract_key: str
    revision: int
    source_published_ts: Ms
    available_at: Ms
    effective_ts: Ms
    option_root_before: str
    option_root_after: str
    state: AdjustmentState
    settlement_state: SettlementState
    contract_multiplier: Decimal
    strike_cash_multiplier: Decimal
    exercise_currency: str
    assets: tuple[NormalizedAdjustmentAsset, ...]
    cash: tuple[NormalizedAdjustmentCash, ...]
    unresolved: tuple[UnresolvedAdjustmentComponent, ...]
    reviewed_by: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.contract_key.strip():
            raise ValueError("source_id and contract_key cannot be empty")
        if self.contract_key != self.contract_key.upper():
            raise ValueError("contract_key must be uppercase")
        if self.revision < 1:
            raise ValueError("adjustment revision must be >= 1")
        if min(self.source_published_ts, self.available_at, self.effective_ts) < 0:
            raise ValueError("adjustment timestamps must be nonnegative")
        if self.available_at < self.source_published_ts:
            raise ValueError("available_at cannot precede source publication")
        _root("option_root_before", self.option_root_before)
        _root("option_root_after", self.option_root_after)
        _positive_decimal("contract_multiplier", self.contract_multiplier)
        _positive_decimal("strike_cash_multiplier", self.strike_cash_multiplier)
        if _CURRENCY_RE.fullmatch(self.exercise_currency) is None:
            raise ValueError("exercise_currency must be 3-12 uppercase alphanumerics")
        if _SHA256_RE.fullmatch(self.source_sha256) is None:
            raise ValueError("source_sha256 must be 64 lowercase hexadecimal characters")
        self._validate_source_url()
        if not self.assets and not self.cash and not self.unresolved:
            raise ValueError("adjustment must contain a deliverable or unresolved component")
        asset_ids = [component.instrument_id for component in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("adjustment asset ids must be unique")
        currencies = [component.currency for component in self.cash]
        if len(currencies) != len(set(currencies)):
            raise ValueError("adjustment cash currencies must be unique")
        reviewers = [reviewer.strip() for reviewer in self.reviewed_by]
        if any(not reviewer for reviewer in reviewers) or len(reviewers) != len(set(reviewers)):
            raise ValueError("reviewed_by must contain unique nonempty reviewer ids")
        delayed = any(component.settlement_delayed for component in self.assets) or any(
            component.settlement_delayed for component in self.cash
        )
        if self.settlement_state is SettlementState.DELAYED and not delayed and not self.unresolved:
            raise ValueError("delayed settlement requires a delayed or unresolved component")
        if (
            self.settlement_state is SettlementState.CASH_ONLY
            and (self.assets or not self.cash or self.unresolved)
        ):
            raise ValueError("cash-only settlement requires resolved cash and no assets")

    def _validate_source_url(self) -> None:
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTPS URL")
        if self.authority is AdjustmentAuthority.OCC_MEMO:
            if parsed.hostname != "infomemo.theocc.com" or parsed.path != "/infomemos":
                raise ValueError("OCC observations must reference an official infomemo URL")
            memo_numbers = parse_qs(parsed.query).get("number", [])
            if memo_numbers != [self.source_id] or not self.source_id.isdigit():
                raise ValueError("OCC source_id must equal the infomemo number query parameter")

    @property
    def replay_blockers(self) -> tuple[str, ...]:
        """Deterministic reasons this observation cannot qualify for replay."""
        blockers: list[str] = []
        if self.state is not AdjustmentState.FINAL:
            blockers.append(f"state:{self.state.value}")
        if len(self.reviewed_by) < 2:
            blockers.append("dual_review_missing")
        if self.unresolved:
            blockers.append("unresolved_components")
        if self.settlement_state is SettlementState.DELAYED:
            blockers.append("delayed_settlement")
        if any(component.settlement_delayed for component in self.assets) or any(
            component.settlement_delayed for component in self.cash
        ):
            blockers.append("delayed_components")
        return tuple(blockers)

    @property
    def ready_for_replay(self) -> bool:
        return not self.replay_blockers

    def require_known(self, as_of: Ms) -> None:
        if self.available_at > as_of:
            raise LookaheadError(
                f"adjustment {self.source_id!r} available at {self.available_at} exceeds "
                f"decision {as_of}"
            )
        if self.effective_ts > as_of:
            raise ValueError(
                f"adjustment {self.source_id!r} is not effective until {self.effective_ts}"
            )

    def to_json_obj(self) -> dict[str, object]:
        """Canonical JSON-safe representation; decimals remain exact strings."""
        return {
            "schema": _SCHEMA,
            "authority": self.authority.value,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "contract_key": self.contract_key,
            "revision": self.revision,
            "source_published_ts": self.source_published_ts,
            "available_at": self.available_at,
            "effective_ts": self.effective_ts,
            "option_root_before": self.option_root_before,
            "option_root_after": self.option_root_after,
            "state": self.state.value,
            "settlement_state": self.settlement_state.value,
            "contract_multiplier": str(self.contract_multiplier),
            "strike_cash_multiplier": str(self.strike_cash_multiplier),
            "exercise_currency": self.exercise_currency,
            "assets": [
                {
                    "instrument_id": item.instrument_id,
                    "quantity_per_contract": str(item.quantity_per_contract),
                    "source_identifier": item.source_identifier,
                    "settlement_delayed": item.settlement_delayed,
                }
                for item in self.assets
            ],
            "cash": [
                {
                    "currency": item.currency,
                    "amount_per_contract": str(item.amount_per_contract),
                    "settlement_delayed": item.settlement_delayed,
                }
                for item in self.cash
            ],
            "unresolved": [
                {"component": item.component, "reason": item.reason}
                for item in self.unresolved
            ],
            "reviewed_by": list(self.reviewed_by),
        }

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self.to_json_obj(), sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionAdjustmentChain:
    """Contiguous point-in-time revisions for one stable adjusted-option class key."""

    observations: tuple[OptionAdjustmentObservation, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("adjustment chain cannot be empty")
        contract_keys = {observation.contract_key for observation in self.observations}
        if len(contract_keys) != 1:
            raise ValueError("adjustment chain must contain one contract_key")
        authorities = {observation.authority for observation in self.observations}
        if len(authorities) != 1:
            raise ValueError("adjustment chain must contain one source authority")
        ordered = sorted(self.observations, key=lambda observation: observation.revision)
        revisions = [observation.revision for observation in ordered]
        if revisions != list(range(1, len(ordered) + 1)):
            raise ValueError("adjustment revisions must be contiguous from 1")
        availability = [observation.available_at for observation in ordered]
        if availability != sorted(availability):
            raise ValueError("adjustment revision availability must be nondecreasing")
        publication = [observation.source_published_ts for observation in ordered]
        if publication != sorted(publication):
            raise ValueError("adjustment revision publication must be nondecreasing")
        for prior, current in pairwise(ordered):
            if current.option_root_before != prior.option_root_after:
                raise ValueError("adjustment revision roots must form a continuous chain")
        if len({observation.source_sha256 for observation in ordered}) != len(ordered):
            raise ValueError("each adjustment revision must bind a distinct source digest")

    @property
    def contract_key(self) -> str:
        return self.observations[0].contract_key

    def as_of(self, decision_ts: Ms) -> OptionAdjustmentObservation | None:
        eligible = (
            observation
            for observation in self.observations
            if observation.available_at <= decision_ts and observation.effective_ts <= decision_ts
        )
        return max(eligible, key=lambda observation: observation.revision, default=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class StaticOptionAdjustmentProvider:
    """Deterministic multi-class adjusted-option revision provider."""

    chains: tuple[OptionAdjustmentChain, ...]

    def __post_init__(self) -> None:
        keys = [chain.contract_key for chain in self.chains]
        if len(keys) != len(set(keys)):
            raise ValueError("provider contract keys must be unique")

    def observation(
        self, contract_key: str, *, as_of: Ms
    ) -> OptionAdjustmentObservation | None:
        for chain in self.chains:
            if chain.contract_key == contract_key:
                return chain.as_of(as_of)
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionAdjustmentReconciliation:
    """Exact economic comparison between OCC and an independent vendor observation."""

    occ: OptionAdjustmentObservation
    vendor: OptionAdjustmentObservation
    mismatched_fields: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def reconciled(self) -> bool:
        return not self.mismatched_fields and not self.blockers

    def require_reconciled(self) -> OptionAdjustmentObservation:
        if not self.reconciled:
            details = ", ".join((*self.mismatched_fields, *self.blockers))
            raise OptionAdjustmentReconciliationError(
                f"option adjustment reconciliation failed: {details}"
            )
        return self.occ


class OptionAdjustmentReconciliationError(ValueError):
    """Raised when a normalized adjustment cannot qualify for deterministic replay."""


class OptionAdjustmentSourceLineageError(ValueError):
    """Raised when reviewed economics are not bound to an archived source observation."""


class _OCCSnapshotProvider(Protocol):
    def snapshots(self, memo_number: str) -> tuple[OCCMemoSnapshot, ...]: ...


def verify_occ_observation_source(
    observation: OptionAdjustmentObservation,
    archive: _OCCSnapshotProvider,
) -> OCCMemoSnapshot:
    """Bind reviewed OCC economics to exact PDF bytes known by extraction availability."""
    if observation.authority is not AdjustmentAuthority.OCC_MEMO:
        raise ValueError("source archive verification requires OCC_MEMO authority")
    matching = tuple(
        snapshot
        for snapshot in archive.snapshots(observation.source_id)
        if snapshot.source_url == observation.source_url
        and snapshot.source_sha256 == observation.source_sha256
    )
    if not matching:
        raise OptionAdjustmentSourceLineageError(
            f"no archived OCC PDF matches memo {observation.source_id} digest "
            f"{observation.source_sha256}"
        )
    earliest = min(matching, key=lambda snapshot: snapshot.observed_at)
    if earliest.observed_at > observation.available_at:
        raise OptionAdjustmentSourceLineageError(
            f"reviewed extraction was available at {observation.available_at} before source "
            f"bytes were observed at {earliest.observed_at}"
        )
    return earliest


def _asset_economics(
    observation: OptionAdjustmentObservation,
) -> tuple[tuple[str, Decimal, str, bool], ...]:
    return tuple(
        sorted(
            (
                item.instrument_id,
                item.quantity_per_contract,
                item.source_identifier,
                item.settlement_delayed,
            )
            for item in observation.assets
        )
    )


def _cash_economics(
    observation: OptionAdjustmentObservation,
) -> tuple[tuple[str, Decimal, bool], ...]:
    return tuple(
        sorted(
            (item.currency, item.amount_per_contract, item.settlement_delayed)
            for item in observation.cash
        )
    )


def _unresolved_economics(
    observation: OptionAdjustmentObservation,
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item.component, item.reason) for item in observation.unresolved))


def reconcile_option_adjustments(
    occ: OptionAdjustmentObservation,
    vendor: OptionAdjustmentObservation,
    *,
    as_of: Ms,
) -> OptionAdjustmentReconciliation:
    """Compare independently captured economics and retain every fail-closed blocker."""
    if occ.authority is not AdjustmentAuthority.OCC_MEMO:
        raise ValueError("occ observation must have OCC_MEMO authority")
    if vendor.authority is not AdjustmentAuthority.VENDOR_TERMS:
        raise ValueError("vendor observation must have VENDOR_TERMS authority")
    occ.require_known(as_of)
    vendor.require_known(as_of)
    fields: tuple[tuple[str, object, object], ...] = (
        ("contract_key", occ.contract_key, vendor.contract_key),
        ("effective_ts", occ.effective_ts, vendor.effective_ts),
        ("option_root_before", occ.option_root_before, vendor.option_root_before),
        ("option_root_after", occ.option_root_after, vendor.option_root_after),
        ("state", occ.state, vendor.state),
        ("settlement_state", occ.settlement_state, vendor.settlement_state),
        ("contract_multiplier", occ.contract_multiplier, vendor.contract_multiplier),
        ("strike_cash_multiplier", occ.strike_cash_multiplier, vendor.strike_cash_multiplier),
        ("exercise_currency", occ.exercise_currency, vendor.exercise_currency),
        ("assets", _asset_economics(occ), _asset_economics(vendor)),
        ("cash", _cash_economics(occ), _cash_economics(vendor)),
        ("unresolved", _unresolved_economics(occ), _unresolved_economics(vendor)),
    )
    mismatches = tuple(
        name for name, occ_value, vendor_value in fields if occ_value != vendor_value
    )
    blockers = (
        *(f"occ:{item}" for item in occ.replay_blockers),
        *(f"vendor:{item}" for item in vendor.replay_blockers),
    )
    return OptionAdjustmentReconciliation(
        occ=occ,
        vendor=vendor,
        mismatched_fields=mismatches,
        blockers=blockers,
    )


def _exact_keys(payload: Mapping[str, object], expected: set[str], *, context: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _decimal(payload: Mapping[str, object], key: str) -> Decimal:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be an exact decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{key} is not a valid decimal string") from exc


def _sequence(payload: Mapping[str, object], key: str) -> Sequence[object]:
    value = payload[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{key} must be an array")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be an object with string keys")
    return value


def _boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def parse_option_adjustment_observation(
    payload: Mapping[str, object],
) -> OptionAdjustmentObservation:
    """Parse the exact v1 reviewed-extraction schema; unknown/missing fields fail closed."""
    expected = {
        "schema",
        "authority",
        "source_id",
        "source_url",
        "source_sha256",
        "contract_key",
        "revision",
        "source_published_ts",
        "available_at",
        "effective_ts",
        "option_root_before",
        "option_root_after",
        "state",
        "settlement_state",
        "contract_multiplier",
        "strike_cash_multiplier",
        "exercise_currency",
        "assets",
        "cash",
        "unresolved",
        "reviewed_by",
    }
    _exact_keys(payload, expected, context="option adjustment")
    if _text(payload, "schema") != _SCHEMA:
        raise ValueError(f"unsupported option-adjustment schema {_text(payload, 'schema')!r}")

    assets: list[NormalizedAdjustmentAsset] = []
    for index, raw in enumerate(_sequence(payload, "assets")):
        item = _mapping(raw, context=f"assets[{index}]")
        _exact_keys(
            item,
            {"instrument_id", "quantity_per_contract", "source_identifier", "settlement_delayed"},
            context=f"assets[{index}]",
        )
        assets.append(
            NormalizedAdjustmentAsset(
                instrument_id=_text(item, "instrument_id"),
                quantity_per_contract=_decimal(item, "quantity_per_contract"),
                source_identifier=_text(item, "source_identifier"),
                settlement_delayed=_boolean(item, "settlement_delayed"),
            )
        )

    cash: list[NormalizedAdjustmentCash] = []
    for index, raw in enumerate(_sequence(payload, "cash")):
        item = _mapping(raw, context=f"cash[{index}]")
        _exact_keys(
            item,
            {"currency", "amount_per_contract", "settlement_delayed"},
            context=f"cash[{index}]",
        )
        cash.append(
            NormalizedAdjustmentCash(
                currency=_text(item, "currency"),
                amount_per_contract=_decimal(item, "amount_per_contract"),
                settlement_delayed=_boolean(item, "settlement_delayed"),
            )
        )

    unresolved: list[UnresolvedAdjustmentComponent] = []
    for index, raw in enumerate(_sequence(payload, "unresolved")):
        item = _mapping(raw, context=f"unresolved[{index}]")
        _exact_keys(item, {"component", "reason"}, context=f"unresolved[{index}]")
        unresolved.append(
            UnresolvedAdjustmentComponent(
                component=_text(item, "component"), reason=_text(item, "reason")
            )
        )

    reviewed_by = tuple(
        value if isinstance(value, str) else _raise_reviewer_type()
        for value in _sequence(payload, "reviewed_by")
    )
    return OptionAdjustmentObservation(
        authority=AdjustmentAuthority(_text(payload, "authority")),
        source_id=_text(payload, "source_id"),
        source_url=_text(payload, "source_url"),
        source_sha256=_text(payload, "source_sha256"),
        contract_key=_text(payload, "contract_key"),
        revision=_integer(payload, "revision"),
        source_published_ts=_integer(payload, "source_published_ts"),
        available_at=_integer(payload, "available_at"),
        effective_ts=_integer(payload, "effective_ts"),
        option_root_before=_text(payload, "option_root_before"),
        option_root_after=_text(payload, "option_root_after"),
        state=AdjustmentState(_text(payload, "state")),
        settlement_state=SettlementState(_text(payload, "settlement_state")),
        contract_multiplier=_decimal(payload, "contract_multiplier"),
        strike_cash_multiplier=_decimal(payload, "strike_cash_multiplier"),
        exercise_currency=_text(payload, "exercise_currency"),
        assets=tuple(assets),
        cash=tuple(cash),
        unresolved=tuple(unresolved),
        reviewed_by=reviewed_by,
    )


def _raise_reviewer_type() -> str:
    raise TypeError("reviewed_by values must be strings")
