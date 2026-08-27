"""Reviewed, source-bound ingestion for point-in-time market-status observations.

Historical status files are unusually easy to backdate accidentally: an exchange may publish an
event after trading stopped, a vendor may revise the interval later, and a locally downloaded file
does not prove when the event was knowable.  This module keeps those timestamps separate and binds
every normalized row to exact source bytes.  It deliberately provides no network adapter and
bundles no historical corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Final
from urllib.parse import urlparse

from alphaforge.core.errors import LookaheadError
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms
from alphaforge.execution.market_status import (
    MarketStatus,
    MarketStatusEvent,
    StaticMarketStatusProvider,
)

__all__ = [
    "MarketStatusAuthority",
    "MarketStatusCorpus",
    "MarketStatusCoverageAudit",
    "MarketStatusCoverageError",
    "MarketStatusCoverageGap",
    "MarketStatusCoverageGapReason",
    "MarketStatusCoverageRequirement",
    "MarketStatusCoverageSegment",
    "MarketStatusObservation",
    "MarketStatusReconciliation",
    "MarketStatusReconciliationError",
    "MarketStatusSourceLineageError",
    "ReviewedMarketStatusProvider",
    "audit_market_status_coverage",
    "parse_market_status_observation",
    "reconcile_market_status_observations",
]

_SCHEMA: Final[str] = "alphaforge.market-status-observation.v1"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "authority",
        "source_id",
        "source_url",
        "source_sha256",
        "source_record_id",
        "source_published_ts",
        "source_captured_at",
        "venue",
        "instrument_id",
        "status",
        "effective_from",
        "effective_until",
        "observed_ts",
        "available_at",
        "reason",
        "reviewed_at",
        "reviewed_by",
    }
)


class MarketStatusSourceLineageError(ValueError):
    """A normalized observation is not bound to the supplied immutable source bytes."""


class MarketStatusReconciliationError(ValueError):
    """Independent status observations do not establish one common event."""


class MarketStatusCoverageError(ValueError):
    """Reviewed status evidence does not cover every required point-in-time interval."""


class MarketStatusAuthority(StrEnum):
    """Independent authority represented by one captured status observation."""

    OFFICIAL_EXCHANGE = "official_exchange"
    VENDOR = "vendor"


class MarketStatusCoverageGapReason(StrEnum):
    """Why one required interval cannot support point-in-time status replay."""

    NO_EFFECTIVE_STATUS = "no_effective_status"
    STATUS_NOT_YET_AVAILABLE = "status_not_yet_available"


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _timestamp(payload: Mapping[str, object], name: str) -> Ms:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer epoch-ms timestamp")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusObservation:
    """One reviewed normalized row bound to an exact captured source artifact."""

    authority: MarketStatusAuthority
    source_id: str
    source_url: str
    source_sha256: str
    source_record_id: str
    source_published_ts: Ms
    source_captured_at: Ms
    reviewed_at: Ms
    reviewed_by: tuple[str, ...]
    event: MarketStatusEvent

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.source_record_id.strip():
            raise ValueError("source_id and source_record_id cannot be empty")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            raise ValueError("source_url must be an absolute credential-free HTTPS URL")
        if _SHA256_RE.fullmatch(self.source_sha256) is None:
            raise ValueError("source_sha256 must be 64 lowercase hexadecimal characters")
        if min(self.source_published_ts, self.source_captured_at, self.reviewed_at) < 0:
            raise ValueError("source and review timestamps must be nonnegative")
        if self.source_published_ts > self.event.available_at:
            raise ValueError("event availability cannot precede source publication")
        if self.event.available_at > self.source_captured_at:
            raise ValueError("source capture cannot precede claimed event availability")
        if self.source_captured_at > self.reviewed_at:
            raise ValueError("review cannot precede source capture")
        reviewers = [reviewer.strip() for reviewer in self.reviewed_by]
        if any(not reviewer for reviewer in reviewers) or len(reviewers) != len(set(reviewers)):
            raise ValueError("reviewed_by must contain unique nonempty reviewer ids")

    @property
    def replay_blockers(self) -> tuple[str, ...]:
        """Deterministic reasons this row cannot qualify as reviewed replay input."""
        if len(self.reviewed_by) < 2:
            return ("dual_review_missing",)
        return ()

    @property
    def ready_for_replay(self) -> bool:
        return not self.replay_blockers

    def require_known(self, as_of: Ms) -> None:
        """Reject use before the source-disclosed historical availability timestamp."""
        if self.event.available_at > as_of:
            raise LookaheadError(
                f"market-status source record {self.source_record_id!r} available at "
                f"{self.event.available_at} exceeds decision {as_of}"
            )

    def to_json_obj(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "authority": self.authority.value,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "source_record_id": self.source_record_id,
            "source_published_ts": self.source_published_ts,
            "source_captured_at": self.source_captured_at,
            "venue": self.event.venue,
            "instrument_id": self.event.instrument_id,
            "status": self.event.status.value,
            "effective_from": self.event.effective_from,
            "effective_until": self.event.effective_until,
            "observed_ts": self.event.observed_ts,
            "available_at": self.event.available_at,
            "reason": self.event.reason,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": list(self.reviewed_by),
        }

    @property
    def content_hash(self) -> str:
        return f"sha256:{_sha256(_canonical_bytes(self.to_json_obj()))}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusCorpus:
    """A collision-checked collection of normalized source observations."""

    observations: tuple[MarketStatusObservation, ...]

    def __post_init__(self) -> None:
        identities = [
            (row.authority, row.source_id, row.source_record_id) for row in self.observations
        ]
        if len(identities) != len(set(identities)):
            raise MarketStatusSourceLineageError("duplicate source-record identity")
        digest_identity: dict[str, tuple[MarketStatusAuthority, str]] = {}
        for row in self.observations:
            identity = (row.authority, row.source_id)
            prior = digest_identity.setdefault(row.source_sha256, identity)
            if prior != identity:
                raise MarketStatusSourceLineageError(
                    "one source digest cannot be relabeled as a different source"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusReconciliation:
    """One exact official/vendor consensus event with source lineage."""

    event: MarketStatusEvent
    source_keys: tuple[str, ...]
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.source_keys) < 2:
            raise ValueError("reconciliation requires at least two source records")
        if len(self.source_keys) != len(set(self.source_keys)):
            raise ValueError("reconciliation source_keys must be unique")
        if len(self.source_sha256) != len(self.source_keys):
            raise ValueError("reconciliation source hashes must align with source records")
        if any(_SHA256_RE.fullmatch(value) is None for value in self.source_sha256):
            raise ValueError("reconciliation source hashes must be lowercase SHA-256 digests")

    @property
    def content_hash(self) -> str:
        payload: dict[str, object] = {
            "event": {
                "venue": self.event.venue,
                "instrument_id": self.event.instrument_id,
                "status": self.event.status.value,
                "effective_from": self.event.effective_from,
                "effective_until": self.event.effective_until,
                "observed_ts": self.event.observed_ts,
                "available_at": self.event.available_at,
                "reason": self.event.reason,
            },
            "source_keys": list(self.source_keys),
            "source_sha256": list(self.source_sha256),
        }
        return f"sha256:{_sha256(_canonical_bytes(payload))}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusCoverageRequirement:
    """One half-open instrument interval requiring explicit replay coverage."""

    instrument_id: str
    effective_from: Ms
    effective_until: Ms

    def __post_init__(self) -> None:
        SymbolMapper.parse_instrument_id(self.instrument_id)
        if self.effective_from < 0 or self.effective_until < 0:
            raise ValueError("coverage requirement timestamps must be nonnegative")
        if self.effective_from >= self.effective_until:
            raise ValueError("coverage requirement effective_from must precede effective_until")

    @property
    def duration_ms(self) -> int:
        return self.effective_until - self.effective_from

    def to_json_obj(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusCoverageSegment:
    """One fully covered interval and its exact reconciled source lineage."""

    instrument_id: str
    effective_from: Ms
    effective_until: Ms
    status: MarketStatus
    reconciliation_hash: str
    source_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        SymbolMapper.parse_instrument_id(self.instrument_id)
        if self.effective_from >= self.effective_until:
            raise ValueError("coverage segment effective_from must precede effective_until")
        if not self.reconciliation_hash.startswith("sha256:"):
            raise ValueError("coverage segment must bind a reconciliation content hash")
        if len(self.source_keys) < 2 or len(self.source_keys) != len(set(self.source_keys)):
            raise ValueError("coverage segment must bind unique independent source keys")

    @property
    def duration_ms(self) -> int:
        return self.effective_until - self.effective_from

    def to_json_obj(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "status": self.status.value,
            "reconciliation_hash": self.reconciliation_hash,
            "source_keys": list(self.source_keys),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusCoverageGap:
    """One exact uncovered interval with a stable fail-closed reason."""

    instrument_id: str
    effective_from: Ms
    effective_until: Ms
    reason: MarketStatusCoverageGapReason
    blocking_reconciliation_hash: str | None = None

    def __post_init__(self) -> None:
        SymbolMapper.parse_instrument_id(self.instrument_id)
        if self.effective_from >= self.effective_until:
            raise ValueError("coverage gap effective_from must precede effective_until")
        if (
            self.blocking_reconciliation_hash is not None
            and not self.blocking_reconciliation_hash.startswith("sha256:")
        ):
            raise ValueError("blocking reconciliation hash must use sha256:<digest>")
        if (
            self.reason is MarketStatusCoverageGapReason.STATUS_NOT_YET_AVAILABLE
            and self.blocking_reconciliation_hash is None
        ):
            raise ValueError("future-known gaps must identify the blocking reconciliation")
        if (
            self.reason is MarketStatusCoverageGapReason.NO_EFFECTIVE_STATUS
            and self.blocking_reconciliation_hash is not None
        ):
            raise ValueError("missing-status gaps cannot identify a blocking reconciliation")

    @property
    def duration_ms(self) -> int:
        return self.effective_until - self.effective_from

    def to_json_obj(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "reason": self.reason.value,
            "blocking_reconciliation_hash": self.blocking_reconciliation_hash,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketStatusCoverageAudit:
    """Deterministic proof or rejection of explicit PIT status coverage."""

    requirements: tuple[MarketStatusCoverageRequirement, ...]
    covered_segments: tuple[MarketStatusCoverageSegment, ...]
    gaps: tuple[MarketStatusCoverageGap, ...]
    reconciliation_hashes: tuple[str, ...]
    required_ms: int
    covered_ms: int

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("coverage audit requires at least one interval")
        if self.required_ms != sum(row.duration_ms for row in self.requirements):
            raise ValueError("required_ms does not reconcile to requirements")
        if self.covered_ms != sum(row.duration_ms for row in self.covered_segments):
            raise ValueError("covered_ms does not reconcile to covered segments")
        gap_ms = sum(row.duration_ms for row in self.gaps)
        if self.covered_ms + gap_ms != self.required_ms:
            raise ValueError("covered and gap durations do not reconcile to required_ms")
        if len(self.reconciliation_hashes) != len(set(self.reconciliation_hashes)):
            raise ValueError("coverage audit reconciliation hashes must be unique")

    @property
    def passed(self) -> bool:
        return not self.gaps and self.covered_ms == self.required_ms

    def to_json_obj(self) -> dict[str, object]:
        return {
            "schema": "alphaforge.market-status-coverage-audit.v1",
            "passed": self.passed,
            "required_ms": self.required_ms,
            "covered_ms": self.covered_ms,
            "requirements": [row.to_json_obj() for row in self.requirements],
            "covered_segments": [row.to_json_obj() for row in self.covered_segments],
            "gaps": [row.to_json_obj() for row in self.gaps],
            "reconciliation_hashes": list(self.reconciliation_hashes),
        }

    @property
    def content_hash(self) -> str:
        return f"sha256:{_sha256(_canonical_bytes(self.to_json_obj()))}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewedMarketStatusProvider:
    """Source-reconciled provider that binds every run to a passing coverage audit."""

    requirements: tuple[MarketStatusCoverageRequirement, ...]
    reconciliations: tuple[MarketStatusReconciliation, ...]
    audit: MarketStatusCoverageAudit = field(init=False)
    _provider: StaticMarketStatusProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        audit = audit_market_status_coverage(self.requirements, self.reconciliations)
        if not audit.passed:
            raise MarketStatusCoverageError(
                f"reviewed market-status corpus has {len(audit.gaps)} coverage gap(s)"
            )
        object.__setattr__(self, "audit", audit)
        object.__setattr__(
            self,
            "_provider",
            StaticMarketStatusProvider(
                events=tuple(row.event for row in self.reconciliations)
            ),
        )

    def require_coverage(
        self,
        instrument_ids: tuple[str, ...],
        *,
        start: Ms,
        end: Ms,
    ) -> str:
        requested = tuple(
            MarketStatusCoverageRequirement(
                instrument_id=instrument_id,
                effective_from=start,
                effective_until=end,
            )
            for instrument_id in dict.fromkeys(instrument_ids)
        )
        audit = audit_market_status_coverage(requested, self.reconciliations)
        if not audit.passed:
            raise MarketStatusCoverageError(
                f"requested run has {len(audit.gaps)} market-status coverage gap(s)"
            )
        return audit.content_hash

    def status(self, instrument_id: str, *, as_of: Ms) -> MarketStatusEvent | None:
        return self._provider.status(instrument_id, as_of=as_of)


def parse_market_status_observation(
    manifest: Mapping[str, object], *, source_bytes: bytes
) -> MarketStatusObservation:
    """Parse one strict reviewed manifest and verify its exact source artifact."""
    fields = frozenset(manifest)
    if fields != _FIELDS:
        missing = sorted(_FIELDS - fields)
        unknown = sorted(fields - _FIELDS)
        raise ValueError(
            f"market-status manifest fields differ: missing={missing}, unknown={unknown}"
        )
    if manifest["schema"] != _SCHEMA:
        raise ValueError(f"unsupported market-status schema {manifest['schema']!r}")
    declared_hash = _required_text(manifest, "source_sha256")
    actual_hash = _sha256(source_bytes)
    if declared_hash != actual_hash:
        raise MarketStatusSourceLineageError(
            f"source SHA-256 mismatch: declared {declared_hash}, observed {actual_hash}"
        )
    instrument_value = manifest["instrument_id"]
    if instrument_value is not None and (
        not isinstance(instrument_value, str) or not instrument_value.strip()
    ):
        raise ValueError("instrument_id must be null or non-empty text")
    reviewers_value = manifest["reviewed_by"]
    if isinstance(reviewers_value, (str, bytes)) or not isinstance(reviewers_value, Sequence):
        raise ValueError("reviewed_by must be a sequence of reviewer ids")
    if any(not isinstance(item, str) for item in reviewers_value):
        raise ValueError("reviewed_by entries must be text")
    try:
        authority = MarketStatusAuthority(_required_text(manifest, "authority"))
        status = MarketStatus(_required_text(manifest, "status"))
    except ValueError as exc:
        raise ValueError("unknown market-status authority or status") from exc
    event = MarketStatusEvent(
        venue=_required_text(manifest, "venue"),
        instrument_id=instrument_value,
        status=status,
        effective_from=_timestamp(manifest, "effective_from"),
        effective_until=_timestamp(manifest, "effective_until"),
        observed_ts=_timestamp(manifest, "observed_ts"),
        available_at=_timestamp(manifest, "available_at"),
        reason=_required_text(manifest, "reason"),
    )
    return MarketStatusObservation(
        authority=authority,
        source_id=_required_text(manifest, "source_id"),
        source_url=_required_text(manifest, "source_url"),
        source_sha256=declared_hash,
        source_record_id=_required_text(manifest, "source_record_id"),
        source_published_ts=_timestamp(manifest, "source_published_ts"),
        source_captured_at=_timestamp(manifest, "source_captured_at"),
        reviewed_at=_timestamp(manifest, "reviewed_at"),
        reviewed_by=tuple(reviewers_value),
        event=event,
    )


def reconcile_market_status_observations(
    observations: Sequence[MarketStatusObservation], *, as_of: Ms
) -> MarketStatusReconciliation:
    """Require exact, independently reviewed official/vendor agreement for one event."""
    rows = tuple(observations)
    if len(rows) < 2:
        raise MarketStatusReconciliationError("at least two observations are required")
    authorities = {row.authority for row in rows}
    if authorities != {
        MarketStatusAuthority.OFFICIAL_EXCHANGE,
        MarketStatusAuthority.VENDOR,
    }:
        raise MarketStatusReconciliationError(
            "reconciliation requires official-exchange and vendor authorities"
        )
    if any(not row.ready_for_replay for row in rows):
        raise MarketStatusReconciliationError("every source must pass dual review")
    source_identities = {(row.authority, row.source_id) for row in rows}
    if len(source_identities) != len(rows):
        raise MarketStatusReconciliationError(
            "source observations must be independently identified"
        )
    if len({row.source_sha256 for row in rows}) != len(rows):
        raise MarketStatusReconciliationError("independent sources cannot reuse source bytes")
    for row in rows:
        row.require_known(as_of)
    first = rows[0].event
    economics = (
        first.venue,
        first.instrument_id,
        first.status,
        first.effective_from,
        first.effective_until,
    )
    for row in rows[1:]:
        event = row.event
        candidate = (
            event.venue,
            event.instrument_id,
            event.status,
            event.effective_from,
            event.effective_until,
        )
        if candidate != economics:
            raise MarketStatusReconciliationError(
                "status, scope, or effective interval disagrees across sources"
            )
    ordered = sorted(rows, key=lambda row: (row.authority.value, row.source_id))
    source_keys = tuple(
        f"{row.authority.value}:{row.source_id}:{row.source_record_id}" for row in ordered
    )
    event = MarketStatusEvent(
        venue=first.venue,
        instrument_id=first.instrument_id,
        status=first.status,
        effective_from=first.effective_from,
        effective_until=first.effective_until,
        observed_ts=max(row.event.observed_ts for row in rows),
        available_at=max(row.event.available_at for row in rows),
        reason="reconciled:" + ",".join(source_keys),
    )
    return MarketStatusReconciliation(
        event=event,
        source_keys=source_keys,
        source_sha256=tuple(row.source_sha256 for row in ordered),
    )


def _append_covered_segment(
    rows: list[MarketStatusCoverageSegment],
    *,
    instrument_id: str,
    effective_from: Ms,
    effective_until: Ms,
    reconciliation: MarketStatusReconciliation,
) -> None:
    candidate = MarketStatusCoverageSegment(
        instrument_id=instrument_id,
        effective_from=effective_from,
        effective_until=effective_until,
        status=reconciliation.event.status,
        reconciliation_hash=reconciliation.content_hash,
        source_keys=reconciliation.source_keys,
    )
    if rows:
        prior = rows[-1]
        if (
            prior.instrument_id == candidate.instrument_id
            and prior.effective_until == candidate.effective_from
            and prior.status is candidate.status
            and prior.reconciliation_hash == candidate.reconciliation_hash
            and prior.source_keys == candidate.source_keys
        ):
            rows[-1] = MarketStatusCoverageSegment(
                instrument_id=prior.instrument_id,
                effective_from=prior.effective_from,
                effective_until=candidate.effective_until,
                status=prior.status,
                reconciliation_hash=prior.reconciliation_hash,
                source_keys=prior.source_keys,
            )
            return
    rows.append(candidate)


def _append_coverage_gap(
    rows: list[MarketStatusCoverageGap],
    *,
    instrument_id: str,
    effective_from: Ms,
    effective_until: Ms,
    reason: MarketStatusCoverageGapReason,
    blocking_reconciliation_hash: str | None,
) -> None:
    candidate = MarketStatusCoverageGap(
        instrument_id=instrument_id,
        effective_from=effective_from,
        effective_until=effective_until,
        reason=reason,
        blocking_reconciliation_hash=blocking_reconciliation_hash,
    )
    if rows:
        prior = rows[-1]
        if (
            prior.instrument_id == candidate.instrument_id
            and prior.effective_until == candidate.effective_from
            and prior.reason is candidate.reason
            and prior.blocking_reconciliation_hash
            == candidate.blocking_reconciliation_hash
        ):
            rows[-1] = MarketStatusCoverageGap(
                instrument_id=prior.instrument_id,
                effective_from=prior.effective_from,
                effective_until=candidate.effective_until,
                reason=prior.reason,
                blocking_reconciliation_hash=prior.blocking_reconciliation_hash,
            )
            return
    rows.append(candidate)


def audit_market_status_coverage(
    requirements: Sequence[MarketStatusCoverageRequirement],
    reconciliations: Sequence[MarketStatusReconciliation],
) -> MarketStatusCoverageAudit:
    """Audit complete PIT coverage without treating absence as an implicit OPEN state.

    Only exact reconciliations are accepted.  Instrument-specific events take precedence over
    venue-wide events exactly as they do in execution.  If the selected event was not yet
    available at a segment boundary, the interval is a future-known gap; the auditor never falls
    back to a venue event to conceal it.
    """
    ordered_requirements = tuple(
        sorted(
            requirements,
            key=lambda row: (row.instrument_id, row.effective_from, row.effective_until),
        )
    )
    if not ordered_requirements:
        raise ValueError("coverage audit requires at least one interval")
    by_instrument: dict[str, list[MarketStatusCoverageRequirement]] = {}
    for requirement in ordered_requirements:
        by_instrument.setdefault(requirement.instrument_id, []).append(requirement)
    for instrument_id, rows in by_instrument.items():
        if any(
            current.effective_until > following.effective_from
            for current, following in pairwise(rows)
        ):
            raise ValueError(
                f"overlapping coverage requirements for instrument {instrument_id!r}"
            )

    ordered_reconciliations = tuple(
        sorted(reconciliations, key=lambda row: row.content_hash)
    )
    reconciliation_hashes = tuple(row.content_hash for row in ordered_reconciliations)
    if len(reconciliation_hashes) != len(set(reconciliation_hashes)):
        raise MarketStatusReconciliationError(
            "duplicate reconciliation content in coverage audit"
        )
    # Reuse the execution provider's per-scope overlap validation.  Venue and instrument scopes
    # may overlap each other because instrument-specific status deliberately has precedence.
    StaticMarketStatusProvider(
        events=tuple(row.event for row in ordered_reconciliations)
    )
    by_event_identity = {id(row.event): row for row in ordered_reconciliations}

    covered: list[MarketStatusCoverageSegment] = []
    gaps: list[MarketStatusCoverageGap] = []
    for requirement in ordered_requirements:
        venue, _, _ = SymbolMapper.parse_instrument_id(requirement.instrument_id)
        applicable = tuple(
            row
            for row in ordered_reconciliations
            if row.event.venue == venue
            and row.event.instrument_id in (None, requirement.instrument_id)
            and row.event.effective_from < requirement.effective_until
            and row.event.effective_until > requirement.effective_from
        )
        boundaries = {requirement.effective_from, requirement.effective_until}
        for row in applicable:
            event = row.event
            boundaries.add(max(requirement.effective_from, event.effective_from))
            boundaries.add(min(requirement.effective_until, event.effective_until))
            if requirement.effective_from < event.available_at < requirement.effective_until:
                boundaries.add(event.available_at)
        ordered_boundaries = sorted(boundaries)
        for effective_from, effective_until in pairwise(ordered_boundaries):
            effective = [
                row
                for row in applicable
                if row.event.effective_from <= effective_from < row.event.effective_until
            ]
            specific = [
                row for row in effective if row.event.instrument_id == requirement.instrument_id
            ]
            selected = specific[0] if specific else (effective[0] if effective else None)
            if selected is None:
                _append_coverage_gap(
                    gaps,
                    instrument_id=requirement.instrument_id,
                    effective_from=effective_from,
                    effective_until=effective_until,
                    reason=MarketStatusCoverageGapReason.NO_EFFECTIVE_STATUS,
                    blocking_reconciliation_hash=None,
                )
                continue
            # Assert that the event object still belongs to the supplied reconciled evidence.
            selected = by_event_identity[id(selected.event)]
            if selected.event.available_at > effective_from:
                _append_coverage_gap(
                    gaps,
                    instrument_id=requirement.instrument_id,
                    effective_from=effective_from,
                    effective_until=effective_until,
                    reason=MarketStatusCoverageGapReason.STATUS_NOT_YET_AVAILABLE,
                    blocking_reconciliation_hash=selected.content_hash,
                )
                continue
            _append_covered_segment(
                covered,
                instrument_id=requirement.instrument_id,
                effective_from=effective_from,
                effective_until=effective_until,
                reconciliation=selected,
            )

    required_ms = sum(row.duration_ms for row in ordered_requirements)
    covered_ms = sum(row.duration_ms for row in covered)
    return MarketStatusCoverageAudit(
        requirements=ordered_requirements,
        covered_segments=tuple(covered),
        gaps=tuple(gaps),
        reconciliation_hashes=reconciliation_hashes,
        required_ms=required_ms,
        covered_ms=covered_ms,
    )
