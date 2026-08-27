"""Reviewed point-in-time option-adjustment normalization and reconciliation tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from alphaforge.core.errors import LookaheadError
from alphaforge.data.ingest.occ_memo_archive import OCCMemoSnapshot
from alphaforge.data.ingest.option_adjustments import (
    AdjustmentAuthority,
    AdjustmentState,
    OptionAdjustmentChain,
    OptionAdjustmentReconciliationError,
    OptionAdjustmentSourceLineageError,
    SettlementState,
    StaticOptionAdjustmentProvider,
    parse_option_adjustment_observation,
    reconcile_option_adjustments,
    verify_occ_observation_source,
)


def _payload(
    *,
    authority: str = "occ_memo",
    source_id: str = "50001",
    revision: int = 1,
    published: int = 20,
    available: int = 25,
    effective: int = 30,
    root_before: str = "ABC",
    root_after: str = "ABC1",
) -> dict[str, object]:
    source_url = (
        f"https://infomemo.theocc.com/infomemos?number={source_id}"
        if authority == "occ_memo"
        else f"https://vendor.example/options/{source_id}"
    )
    return {
        "schema": "alphaforge.option-adjustment-observation.v1",
        "authority": authority,
        "source_id": source_id,
        "source_url": source_url,
        "source_sha256": f"{revision + (0 if authority == 'occ_memo' else 100):064x}",
        "contract_key": "ABC1",
        "revision": revision,
        "source_published_ts": published,
        "available_at": available,
        "effective_ts": effective,
        "option_root_before": root_before,
        "option_root_after": root_after,
        "state": "final",
        "settlement_state": "standard",
        "contract_multiplier": "100",
        "strike_cash_multiplier": "100",
        "exercise_currency": "USD",
        "assets": [
            {
                "instrument_id": "XUSE:CASH:ABCUSD",
                "quantity_per_contract": "61",
                "source_identifier": "000000001",
                "settlement_delayed": False,
            },
            {
                "instrument_id": "XUSE:CASH:SPINUSD",
                "quantity_per_contract": "17",
                "source_identifier": "000000002",
                "settlement_delayed": False,
            },
        ],
        "cash": [
            {
                "currency": "USD",
                "amount_per_contract": "806.47",
                "settlement_delayed": False,
            }
        ],
        "unresolved": [],
        "reviewed_by": ["reviewer-a", "reviewer-b"],
    }


def _observation(**kwargs: object):
    return parse_option_adjustment_observation(_payload(**kwargs))


def _vendor(**kwargs: object):
    payload = _payload(
        authority="vendor_terms",
        source_id="vendor-abc-1",
        **kwargs,
    )
    return parse_option_adjustment_observation(payload)


def test_strict_manifest_round_trips_exact_decimals_and_hashes_deterministically() -> None:
    payload = _payload()
    observation = parse_option_adjustment_observation(payload)
    assert observation.to_json_obj() == payload
    assert observation.content_hash == parse_option_adjustment_observation(payload).content_hash
    assert observation.ready_for_replay
    assert observation.replay_blockers == ()


def test_manifest_rejects_missing_unknown_and_nonexact_fields() -> None:
    missing = _payload()
    del missing["effective_ts"]
    with pytest.raises(ValueError, match="missing"):
        parse_option_adjustment_observation(missing)

    unknown = _payload()
    unknown["inferred_deliverable"] = True
    with pytest.raises(ValueError, match="unexpected"):
        parse_option_adjustment_observation(unknown)

    floating = _payload()
    floating["contract_multiplier"] = 100.0
    with pytest.raises(TypeError, match="exact decimal string"):
        parse_option_adjustment_observation(floating)


def test_occ_manifest_requires_official_url_and_matching_numeric_memo() -> None:
    unofficial = _payload()
    unofficial["source_url"] = "https://vendor.example/50001"
    with pytest.raises(ValueError, match="official infomemo"):
        parse_option_adjustment_observation(unofficial)

    mismatch = _payload()
    mismatch["source_url"] = "https://infomemo.theocc.com/infomemos?number=50002"
    with pytest.raises(ValueError, match="must equal"):
        parse_option_adjustment_observation(mismatch)

    nonnumeric = _payload(source_id="memo-five")
    with pytest.raises(ValueError, match="must equal"):
        parse_option_adjustment_observation(nonnumeric)


def test_manifest_validates_timestamp_digest_root_and_review_lineage() -> None:
    late_source = _payload(published=30, available=29)
    with pytest.raises(ValueError, match="cannot precede"):
        parse_option_adjustment_observation(late_source)

    digest = _payload()
    digest["source_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        parse_option_adjustment_observation(digest)

    root = _payload()
    root["option_root_after"] = "abc1"
    with pytest.raises(ValueError, match="uppercase alphanumerics"):
        parse_option_adjustment_observation(root)

    duplicate_review = _payload()
    duplicate_review["reviewed_by"] = ["same", "same"]
    with pytest.raises(ValueError, match="unique"):
        parse_option_adjustment_observation(duplicate_review)


def test_asset_and_cash_components_require_canonical_unique_positive_economics() -> None:
    wrong_market = _payload()
    wrong_market["assets"][0]["instrument_id"] = "OPRA:OPTION:ABC1"  # type: ignore[index]
    with pytest.raises(ValueError, match="CASH"):
        parse_option_adjustment_observation(wrong_market)

    duplicate_asset = _payload()
    duplicate_asset["assets"][1]["instrument_id"] = "XUSE:CASH:ABCUSD"  # type: ignore[index]
    with pytest.raises(ValueError, match="asset ids must be unique"):
        parse_option_adjustment_observation(duplicate_asset)

    nonpositive = _payload()
    nonpositive["cash"][0]["amount_per_contract"] = "0"  # type: ignore[index]
    with pytest.raises(ValueError, match="> 0"):
        parse_option_adjustment_observation(nonpositive)


def test_anticipated_unresolved_and_single_review_records_remain_auditable_but_blocked() -> None:
    payload = _payload()
    payload["state"] = "anticipated"
    payload["reviewed_by"] = ["reviewer-a"]
    payload["unresolved"] = [
        {
            "component": "cash in lieu of fractional SPIN shares",
            "reason": "amount TBD in later OCC settlement memo",
        }
    ]
    observation = parse_option_adjustment_observation(payload)
    assert not observation.ready_for_replay
    assert observation.replay_blockers == (
        "state:anticipated",
        "dual_review_missing",
        "unresolved_components",
    )


def test_delayed_settlement_requires_a_disclosed_delayed_or_unresolved_component() -> None:
    payload = _payload()
    payload["settlement_state"] = "delayed"
    with pytest.raises(ValueError, match="requires a delayed"):
        parse_option_adjustment_observation(payload)

    payload["cash"][0]["settlement_delayed"] = True  # type: ignore[index]
    observation = parse_option_adjustment_observation(payload)
    assert observation.replay_blockers == ("delayed_settlement", "delayed_components")


def test_cash_only_contract_requires_resolved_cash_and_no_assets() -> None:
    payload = _payload()
    payload["settlement_state"] = "cash_only"
    with pytest.raises(ValueError, match="cash-only"):
        parse_option_adjustment_observation(payload)

    payload["assets"] = []
    observation = parse_option_adjustment_observation(payload)
    assert observation.settlement_state is SettlementState.CASH_ONLY
    assert observation.ready_for_replay


def test_observation_fails_closed_before_availability_or_effectiveness() -> None:
    observation = _observation(available=25, effective=30)
    with pytest.raises(LookaheadError, match="available"):
        observation.require_known(24)
    with pytest.raises(ValueError, match="not effective"):
        observation.require_known(29)
    observation.require_known(30)


def test_chain_uses_latest_revision_both_effective_and_available() -> None:
    first = _observation(revision=1, published=10, available=20, effective=30)
    second = _observation(
        source_id="50002",
        revision=2,
        published=40,
        available=45,
        effective=30,
        root_before="ABC1",
    )
    chain = OptionAdjustmentChain(observations=(second, first))
    assert chain.contract_key == "ABC1"
    assert chain.as_of(19) is None
    assert chain.as_of(35) == first
    assert chain.as_of(44) == first
    assert chain.as_of(45) == second


def test_chain_rejects_gaps_availability_reversal_and_reused_source_content() -> None:
    first = _observation(revision=1, published=10, available=20, effective=30)
    third = _observation(
        source_id="50003",
        revision=3,
        published=40,
        available=45,
        effective=30,
        root_before="ABC1",
    )
    with pytest.raises(ValueError, match="contiguous"):
        OptionAdjustmentChain(observations=(first, third))

    reversed_availability = _observation(
        source_id="50002",
        revision=2,
        published=10,
        available=19,
        effective=30,
        root_before="ABC1",
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        OptionAdjustmentChain(observations=(first, reversed_availability))

    reused_payload = _payload(
        source_id="50002",
        revision=2,
        published=30,
        available=35,
        effective=30,
        root_before="ABC1",
    )
    reused_payload["source_sha256"] = first.source_sha256
    reused = parse_option_adjustment_observation(reused_payload)
    with pytest.raises(ValueError, match="distinct source digest"):
        OptionAdjustmentChain(observations=(first, reused))


def test_chain_rejects_authority_publication_and_root_discontinuities() -> None:
    first = _observation(revision=1, published=10, available=20, effective=30)

    mixed_authority_payload = _payload(
        authority="vendor_terms",
        source_id="vendor-abc-2",
        revision=2,
        published=21,
        available=25,
        effective=30,
        root_before="ABC1",
    )
    mixed_authority = parse_option_adjustment_observation(mixed_authority_payload)
    with pytest.raises(ValueError, match="one source authority"):
        OptionAdjustmentChain(observations=(first, mixed_authority))

    publication_reversal = _observation(
        source_id="50002",
        revision=2,
        published=9,
        available=25,
        effective=30,
        root_before="ABC1",
    )
    with pytest.raises(ValueError, match="publication"):
        OptionAdjustmentChain(observations=(first, publication_reversal))

    root_discontinuity = _observation(
        source_id="50002",
        revision=2,
        published=21,
        available=25,
        effective=30,
        root_before="OTHER",
    )
    with pytest.raises(ValueError, match="continuous"):
        OptionAdjustmentChain(observations=(first, root_discontinuity))


def test_static_provider_selects_by_stable_contract_key() -> None:
    chain = OptionAdjustmentChain(observations=(_observation(),))
    provider = StaticOptionAdjustmentProvider(chains=(chain,))
    assert provider.observation("ABC1", as_of=30) == chain.observations[0]
    assert provider.observation("UNKNOWN", as_of=30) is None
    with pytest.raises(ValueError, match="unique"):
        StaticOptionAdjustmentProvider(chains=(chain, chain))


def test_exact_occ_vendor_economics_reconcile_and_return_occ_authority() -> None:
    occ = _observation()
    vendor = _vendor()
    report = reconcile_option_adjustments(occ, vendor, as_of=30)
    assert report.reconciled
    assert report.mismatched_fields == ()
    assert report.blockers == ()
    assert report.require_reconciled() is occ
    assert occ.authority is AdjustmentAuthority.OCC_MEMO
    assert occ.state is AdjustmentState.FINAL


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_ts", 31),
        ("option_root_before", "OTHER"),
        ("option_root_after", "ABC2"),
        ("state", "anticipated"),
        ("contract_multiplier", "150"),
        ("strike_cash_multiplier", "50"),
        ("exercise_currency", "EUR"),
        ("settlement_state", "cash_only"),
    ],
)
def test_reconciliation_names_each_scalar_disagreement(field: str, value: object) -> None:
    vendor_payload = _payload(authority="vendor_terms", source_id="vendor-abc-1")
    vendor_payload[field] = value
    if field == "settlement_state":
        vendor_payload["assets"] = []
    vendor = parse_option_adjustment_observation(vendor_payload)
    report = reconcile_option_adjustments(_observation(), vendor, as_of=40)
    assert field in report.mismatched_fields
    with pytest.raises(OptionAdjustmentReconciliationError, match=field):
        report.require_reconciled()


def test_reconciliation_detects_asset_cash_and_source_identifier_disagreements() -> None:
    vendor_payload = _payload(authority="vendor_terms", source_id="vendor-abc-1")
    vendor_payload["assets"][0]["source_identifier"] = "DIFFERENT"  # type: ignore[index]
    vendor_payload["cash"][0]["amount_per_contract"] = "806.48"  # type: ignore[index]
    report = reconcile_option_adjustments(
        _observation(), parse_option_adjustment_observation(vendor_payload), as_of=30
    )
    assert report.mismatched_fields == ("assets", "cash")


def test_reconciliation_retains_readiness_blockers_even_when_economics_match() -> None:
    vendor_payload = _payload(authority="vendor_terms", source_id="vendor-abc-1")
    vendor_payload["reviewed_by"] = ["only-reviewer"]
    report = reconcile_option_adjustments(
        _observation(), parse_option_adjustment_observation(vendor_payload), as_of=30
    )
    assert report.mismatched_fields == ()
    assert report.blockers == ("vendor:dual_review_missing",)
    with pytest.raises(OptionAdjustmentReconciliationError, match="dual_review_missing"):
        report.require_reconciled()


def test_reconciliation_requires_correct_authorities_and_point_in_time_availability() -> None:
    occ = _observation()
    vendor = _vendor(available=35)
    with pytest.raises(LookaheadError, match="available"):
        reconcile_option_adjustments(occ, vendor, as_of=30)
    with pytest.raises(ValueError, match="OCC_MEMO"):
        reconcile_option_adjustments(vendor, vendor, as_of=40)
    with pytest.raises(ValueError, match="VENDOR_TERMS"):
        reconcile_option_adjustments(occ, occ, as_of=40)


def test_manifest_payload_is_not_mutated_by_parser() -> None:
    payload = _payload()
    before = deepcopy(payload)
    parse_option_adjustment_observation(payload)
    assert payload == before


class _Snapshots:
    def __init__(self, snapshots: tuple[OCCMemoSnapshot, ...]) -> None:
        self._snapshots = snapshots

    def snapshots(self, memo_number: str) -> tuple[OCCMemoSnapshot, ...]:
        assert memo_number == "50001"
        return self._snapshots


def _snapshot(*, observed_at: int, digest: str = "0" * 63 + "1") -> OCCMemoSnapshot:
    return OCCMemoSnapshot(
        memo_number="50001",
        source_url="https://infomemo.theocc.com/infomemos?number=50001",
        observed_at=observed_at,
        source_sha256=digest,
        byte_length=100,
        content_type="application/pdf",
        etag=None,
        last_modified=None,
        blob_relpath=f"blobs/sha256/{digest[:2]}/{digest}.pdf",
        manifest_relpath=f"manifests/50001/{observed_at}-{digest}.json",
        manifest_sha256="f" * 64,
    )


def test_occ_reviewed_extraction_binds_to_earliest_matching_archived_bytes() -> None:
    observation = _observation(available=25)
    earlier = _snapshot(observed_at=20)
    later = _snapshot(observed_at=24)
    assert verify_occ_observation_source(observation, _Snapshots((later, earlier))) == earlier


def test_occ_source_binding_rejects_missing_digest_and_impossible_availability() -> None:
    observation = _observation(available=25)
    with pytest.raises(OptionAdjustmentSourceLineageError, match="no archived OCC PDF"):
        verify_occ_observation_source(
            observation,
            _Snapshots((_snapshot(observed_at=20, digest="a" * 64),)),
        )
    with pytest.raises(OptionAdjustmentSourceLineageError, match="before source bytes"):
        verify_occ_observation_source(observation, _Snapshots((_snapshot(observed_at=26),)))


def test_occ_source_binding_rejects_vendor_authority() -> None:
    with pytest.raises(ValueError, match="OCC_MEMO authority"):
        verify_occ_observation_source(_vendor(), _Snapshots(()))
