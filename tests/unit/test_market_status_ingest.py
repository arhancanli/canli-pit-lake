"""Adversarial tests for reviewed historical market-status ingestion."""

from __future__ import annotations

import hashlib

import pytest

from alphaforge.core.errors import LookaheadError
from alphaforge.data.ingest.market_status import (
    MarketStatusAuthority,
    MarketStatusCorpus,
    MarketStatusCoverageAudit,
    MarketStatusCoverageError,
    MarketStatusCoverageGapReason,
    MarketStatusCoverageRequirement,
    MarketStatusReconciliation,
    MarketStatusReconciliationError,
    MarketStatusSourceLineageError,
    ReviewedMarketStatusProvider,
    audit_market_status_coverage,
    parse_market_status_observation,
    reconcile_market_status_observations,
)
from alphaforge.execution.market_status import (
    MarketStatus,
    MarketStatusEvent,
    StaticMarketStatusProvider,
)

SOURCE = b'{"official":"halt-17"}'
VENDOR_SOURCE = b'{"vendor":"halt-17"}'


def _manifest(
    *,
    source_bytes: bytes = SOURCE,
    authority: str = "official_exchange",
    source_id: str = "NASDAQ-STATUS",
    source_record_id: str = "halt-17",
    source_url: str = "https://example.exchange/status/halt-17",
    status: str = "halted",
    instrument_id: str | None = "XNAS:CASH:AAPLUSD",
    effective_from: int = 100,
    effective_until: int = 200,
    observed_ts: int = 105,
    available_at: int = 110,
    source_published_ts: int = 108,
    source_captured_at: int = 120,
    reviewed_at: int = 130,
    reviewed_by: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema": "alphaforge.market-status-observation.v1",
        "authority": authority,
        "source_id": source_id,
        "source_url": source_url,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_record_id": source_record_id,
        "source_published_ts": source_published_ts,
        "source_captured_at": source_captured_at,
        "venue": "XNAS",
        "instrument_id": instrument_id,
        "status": status,
        "effective_from": effective_from,
        "effective_until": effective_until,
        "observed_ts": observed_ts,
        "available_at": available_at,
        "reason": "regulatory halt",
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by or ["reviewer-a", "reviewer-b"],
    }


def _official(**changes: object):
    manifest = _manifest()
    manifest.update(changes)
    return parse_market_status_observation(manifest, source_bytes=SOURCE)


def _vendor(**changes: object):
    manifest = _manifest(
        source_bytes=VENDOR_SOURCE,
        authority="vendor",
        source_id="VENDOR-STATUS",
        source_record_id="vendor-halt-17",
        source_url="https://vendor.example/status/halt-17",
        observed_ts=106,
        available_at=112,
        source_published_ts=109,
        source_captured_at=122,
        reviewed_at=132,
    )
    manifest.update(changes)
    return parse_market_status_observation(manifest, source_bytes=VENDOR_SOURCE)


def test_strict_manifest_binds_exact_source_and_preserves_all_timestamps() -> None:
    row = _official()
    assert row.authority is MarketStatusAuthority.OFFICIAL_EXCHANGE
    assert row.event.status is MarketStatus.HALTED
    assert row.event.effective_from == 100
    assert row.event.observed_ts == 105
    assert row.event.available_at == 110
    assert row.source_captured_at == 120
    assert row.reviewed_at == 130
    assert row.ready_for_replay
    assert row.content_hash.startswith("sha256:")
    assert row.to_json_obj()["source_sha256"] == hashlib.sha256(SOURCE).hexdigest()


@pytest.mark.parametrize("field", ["reason", "available_at", "source_sha256"])
def test_missing_manifest_field_fails_closed(field: str) -> None:
    manifest = _manifest()
    del manifest[field]
    with pytest.raises(ValueError, match="fields differ"):
        parse_market_status_observation(manifest, source_bytes=SOURCE)


def test_unknown_manifest_field_fails_closed() -> None:
    manifest = _manifest()
    manifest["quietly_corrected"] = True
    with pytest.raises(ValueError, match=r"unknown=.*quietly_corrected"):
        parse_market_status_observation(manifest, source_bytes=SOURCE)


def test_source_digest_mismatch_fails_closed() -> None:
    with pytest.raises(MarketStatusSourceLineageError, match="SHA-256 mismatch"):
        parse_market_status_observation(_manifest(), source_bytes=b"different")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_url": "http://example.exchange/status"}, "HTTPS"),
        ({"source_published_ts": 111}, "availability cannot precede"),
        ({"source_captured_at": 109}, "capture cannot precede"),
        ({"reviewed_at": 119}, "review cannot precede"),
        ({"available_at": True}, "nonnegative integer"),
    ],
)
def test_invalid_lineage_and_timestamp_order_fails_closed(
    changes: dict[str, object], message: str
) -> None:
    manifest = _manifest()
    manifest.update(changes)
    with pytest.raises(ValueError, match=message):
        parse_market_status_observation(manifest, source_bytes=SOURCE)


def test_single_review_remains_visible_but_not_replay_qualified() -> None:
    row = _official(reviewed_by=["reviewer-a"])
    assert row.replay_blockers == ("dual_review_missing",)
    assert not row.ready_for_replay


def test_historical_availability_is_enforced_independently_of_capture_and_review() -> None:
    row = _official()
    with pytest.raises(LookaheadError, match="exceeds decision"):
        row.require_known(109)
    row.require_known(110)


def test_corpus_rejects_duplicate_record_identity() -> None:
    row = _official()
    with pytest.raises(MarketStatusSourceLineageError, match="duplicate"):
        MarketStatusCorpus(observations=(row, row))


def test_corpus_rejects_relabeling_identical_bytes_as_another_source() -> None:
    official = _official()
    relabeled = _official(source_id="OTHER-SOURCE", source_record_id="other-record")
    with pytest.raises(MarketStatusSourceLineageError, match="relabeled"):
        MarketStatusCorpus(observations=(official, relabeled))


def test_exact_official_vendor_reconciliation_produces_provider_ready_event() -> None:
    result = reconcile_market_status_observations((_official(), _vendor()), as_of=150)
    assert result.event.available_at == 112
    assert result.event.observed_ts == 106
    assert result.source_keys == (
        "official_exchange:NASDAQ-STATUS:halt-17",
        "vendor:VENDOR-STATUS:vendor-halt-17",
    )
    assert len(set(result.source_sha256)) == 2
    assert result.content_hash.startswith("sha256:")
    provider = StaticMarketStatusProvider(events=(result.event,))
    assert provider.status("XNAS:CASH:AAPLUSD", as_of=150) == result.event


@pytest.mark.parametrize(
    "vendor_change",
    [
        {"status": "open"},
        {"effective_until": 201},
        {"instrument_id": None},
    ],
)
def test_cross_source_economic_disagreement_fails_closed(
    vendor_change: dict[str, object],
) -> None:
    with pytest.raises(MarketStatusReconciliationError, match="disagrees"):
        reconcile_market_status_observations(
            (_official(), _vendor(**vendor_change)), as_of=150
        )


def test_same_authority_cannot_impersonate_independent_confirmation() -> None:
    other_official_manifest = _manifest(
        source_bytes=VENDOR_SOURCE,
        source_id="OTHER-OFFICIAL",
        source_record_id="other-halt-17",
        source_url="https://other.exchange/status/halt-17",
    )
    other = parse_market_status_observation(
        other_official_manifest, source_bytes=VENDOR_SOURCE
    )
    with pytest.raises(MarketStatusReconciliationError, match="official-exchange and vendor"):
        reconcile_market_status_observations((_official(), other), as_of=150)


def test_unreviewed_or_future_known_source_cannot_reconcile() -> None:
    with pytest.raises(MarketStatusReconciliationError, match="dual review"):
        reconcile_market_status_observations(
            (_official(), _vendor(reviewed_by=["one"])), as_of=150
        )
    with pytest.raises(LookaheadError, match="exceeds decision"):
        reconcile_market_status_observations((_official(), _vendor()), as_of=111)


def _reconciled_event(
    key: str,
    *,
    effective_from: int,
    effective_until: int,
    available_at: int,
    status: MarketStatus = MarketStatus.OPEN,
    venue: str = "XNAS",
    instrument_id: str | None = None,
) -> MarketStatusReconciliation:
    return MarketStatusReconciliation(
        event=MarketStatusEvent(
            venue=venue,
            instrument_id=instrument_id,
            status=status,
            effective_from=effective_from,
            effective_until=effective_until,
            observed_ts=available_at,
            available_at=available_at,
            reason=f"fixture:{key}",
        ),
        source_keys=(f"official:{key}", f"vendor:{key}"),
        source_sha256=(
            hashlib.sha256(f"official:{key}".encode()).hexdigest(),
            hashlib.sha256(f"vendor:{key}".encode()).hexdigest(),
        ),
    )


def _require(
    instrument_id: str = "XNAS:CASH:AAPLUSD", *, start: int = 0, end: int = 100
) -> MarketStatusCoverageRequirement:
    return MarketStatusCoverageRequirement(
        instrument_id=instrument_id,
        effective_from=start,
        effective_until=end,
    )


def test_coverage_audit_passes_only_with_complete_known_reconciled_interval() -> None:
    reconciliation = _reconciled_event(
        "venue-open", effective_from=0, effective_until=100, available_at=0
    )
    audit = audit_market_status_coverage((_require(),), (reconciliation,))
    assert audit.passed
    assert audit.required_ms == audit.covered_ms == 100
    assert audit.gaps == ()
    assert len(audit.covered_segments) == 1
    assert audit.covered_segments[0].reconciliation_hash == reconciliation.content_hash
    assert audit.covered_segments[0].source_keys == reconciliation.source_keys
    assert audit.to_json_obj()["schema"] == "alphaforge.market-status-coverage-audit.v1"
    assert audit.content_hash.startswith("sha256:")


def test_reviewed_provider_binds_runtime_preflight_to_source_audit_hash() -> None:
    reconciliation = _reconciled_event(
        "venue-open", effective_from=0, effective_until=100, available_at=0
    )
    provider = ReviewedMarketStatusProvider(
        requirements=(_require(),), reconciliations=(reconciliation,)
    )
    assert provider.audit.passed
    assert provider.require_coverage(
        ("XNAS:CASH:AAPLUSD",), start=0, end=100
    ) == provider.audit.content_hash
    event = provider.status("XNAS:CASH:AAPLUSD", as_of=50)
    assert event is not None and event.status is MarketStatus.OPEN


def test_reviewed_provider_refuses_construction_with_disclosed_gaps() -> None:
    with pytest.raises(MarketStatusCoverageError, match="coverage gap"):
        ReviewedMarketStatusProvider(requirements=(_require(),), reconciliations=())


def test_reviewed_provider_refuses_run_outside_preflight_evidence() -> None:
    reconciliation = _reconciled_event(
        "partial", effective_from=0, effective_until=100, available_at=0
    )
    provider = ReviewedMarketStatusProvider(
        requirements=(_require(),), reconciliations=(reconciliation,)
    )
    with pytest.raises(MarketStatusCoverageError, match="requested run"):
        provider.require_coverage(("XNAS:CASH:AAPLUSD",), start=0, end=101)


def test_absence_is_never_treated_as_implicit_open() -> None:
    audit = audit_market_status_coverage((_require(),), ())
    assert not audit.passed
    assert audit.covered_ms == 0
    assert len(audit.gaps) == 1
    gap = audit.gaps[0]
    assert (gap.effective_from, gap.effective_until) == (0, 100)
    assert gap.reason is MarketStatusCoverageGapReason.NO_EFFECTIVE_STATUS
    assert gap.blocking_reconciliation_hash is None


def test_head_and_tail_gaps_are_published_exactly() -> None:
    middle = _reconciled_event(
        "middle", effective_from=10, effective_until=90, available_at=10
    )
    audit = audit_market_status_coverage((_require(),), (middle,))
    assert audit.covered_ms == 80
    assert [(row.effective_from, row.effective_until) for row in audit.gaps] == [
        (0, 10),
        (90, 100),
    ]
    assert all(
        row.reason is MarketStatusCoverageGapReason.NO_EFFECTIVE_STATUS
        for row in audit.gaps
    )


def test_future_known_status_is_a_labeled_gap_until_availability() -> None:
    late = _reconciled_event(
        "late", effective_from=0, effective_until=100, available_at=20
    )
    audit = audit_market_status_coverage((_require(),), (late,))
    assert audit.covered_ms == 80
    assert len(audit.gaps) == 1
    gap = audit.gaps[0]
    assert (gap.effective_from, gap.effective_until) == (0, 20)
    assert gap.reason is MarketStatusCoverageGapReason.STATUS_NOT_YET_AVAILABLE
    assert gap.blocking_reconciliation_hash == late.content_hash


def test_instrument_status_precedence_is_preserved_in_coverage_segments() -> None:
    venue = _reconciled_event(
        "venue", effective_from=0, effective_until=100, available_at=0
    )
    instrument = _reconciled_event(
        "instrument-halt",
        effective_from=20,
        effective_until=40,
        available_at=20,
        status=MarketStatus.HALTED,
        instrument_id="XNAS:CASH:AAPLUSD",
    )
    audit = audit_market_status_coverage((_require(),), (venue, instrument))
    assert audit.passed
    assert [row.status for row in audit.covered_segments] == [
        MarketStatus.OPEN,
        MarketStatus.HALTED,
        MarketStatus.OPEN,
    ]
    assert [
        (row.effective_from, row.effective_until) for row in audit.covered_segments
    ] == [(0, 20), (20, 40), (40, 100)]


def test_future_known_instrument_status_cannot_fall_back_to_venue_open() -> None:
    venue = _reconciled_event(
        "venue", effective_from=0, effective_until=100, available_at=0
    )
    instrument = _reconciled_event(
        "late-instrument-halt",
        effective_from=20,
        effective_until=40,
        available_at=30,
        status=MarketStatus.HALTED,
        instrument_id="XNAS:CASH:AAPLUSD",
    )
    audit = audit_market_status_coverage((_require(),), (venue, instrument))
    assert not audit.passed
    assert audit.covered_ms == 90
    assert [(row.effective_from, row.effective_until) for row in audit.gaps] == [(20, 30)]
    assert audit.gaps[0].reason is MarketStatusCoverageGapReason.STATUS_NOT_YET_AVAILABLE
    assert audit.gaps[0].blocking_reconciliation_hash == instrument.content_hash


def test_adjacent_reconciled_events_cover_without_boundary_holes() -> None:
    first = _reconciled_event(
        "first", effective_from=0, effective_until=50, available_at=0
    )
    second = _reconciled_event(
        "second", effective_from=50, effective_until=100, available_at=50
    )
    audit = audit_market_status_coverage((_require(),), (second, first))
    assert audit.passed
    assert len(audit.covered_segments) == 2
    assert audit.covered_segments[0].effective_until == 50
    assert audit.covered_segments[1].effective_from == 50


def test_overlapping_reconciled_events_in_one_scope_fail_closed() -> None:
    first = _reconciled_event(
        "first", effective_from=0, effective_until=60, available_at=0
    )
    second = _reconciled_event(
        "second", effective_from=50, effective_until=100, available_at=50
    )
    with pytest.raises(ValueError, match="overlapping market-status intervals"):
        audit_market_status_coverage((_require(),), (first, second))


def test_overlapping_requirements_cannot_double_count_coverage() -> None:
    with pytest.raises(ValueError, match="overlapping coverage requirements"):
        audit_market_status_coverage(
            (_require(start=0, end=60), _require(start=50, end=100)), ()
        )


def test_duplicate_reconciliation_content_fails_closed() -> None:
    row = _reconciled_event(
        "duplicate", effective_from=0, effective_until=100, available_at=0
    )
    with pytest.raises(MarketStatusReconciliationError, match="duplicate reconciliation"):
        audit_market_status_coverage((_require(),), (row, row))


def test_unrelated_venue_evidence_does_not_create_coverage() -> None:
    unrelated = _reconciled_event(
        "unrelated",
        effective_from=0,
        effective_until=100,
        available_at=0,
        venue="XNYS",
    )
    audit = audit_market_status_coverage((_require(),), (unrelated,))
    assert not audit.passed
    assert audit.gaps[0].reason is MarketStatusCoverageGapReason.NO_EFFECTIVE_STATUS


def test_multi_instrument_requirements_reconcile_duration_without_cross_fill() -> None:
    venue = _reconciled_event(
        "venue", effective_from=0, effective_until=100, available_at=0
    )
    requirements = (
        _require("XNAS:CASH:AAPLUSD"),
        _require("XNAS:CASH:MSFTUSD", start=20, end=80),
    )
    audit = audit_market_status_coverage(requirements, (venue,))
    assert audit.passed
    assert audit.required_ms == audit.covered_ms == 160
    assert {row.instrument_id for row in audit.covered_segments} == {
        "XNAS:CASH:AAPLUSD",
        "XNAS:CASH:MSFTUSD",
    }


def test_coverage_hash_and_order_are_deterministic() -> None:
    first = _reconciled_event(
        "first", effective_from=0, effective_until=50, available_at=0
    )
    second = _reconciled_event(
        "second", effective_from=50, effective_until=100, available_at=50
    )
    forward = audit_market_status_coverage((_require(),), (first, second))
    reversed_input = audit_market_status_coverage((_require(),), (second, first))
    assert forward == reversed_input
    assert forward.content_hash == reversed_input.content_hash


def test_reconciliation_rejects_malformed_source_hashes() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        MarketStatusReconciliation(
            event=MarketStatusEvent(
                venue="XNAS",
                instrument_id=None,
                status=MarketStatus.OPEN,
                effective_from=0,
                effective_until=100,
                observed_ts=0,
                available_at=0,
                reason="fixture",
            ),
            source_keys=("official:x", "vendor:x"),
            source_sha256=("not-a-hash", hashlib.sha256(b"vendor").hexdigest()),
        )


def test_coverage_audit_rejects_duration_that_does_not_reconcile() -> None:
    with pytest.raises(ValueError, match="required_ms"):
        MarketStatusCoverageAudit(
            requirements=(_require(),),
            covered_segments=(),
            gaps=(),
            reconciliation_hashes=(),
            required_ms=99,
            covered_ms=0,
        )
