"""
Tests for Phase 2.2 Tamper-Evident Event Hash Chain.

Validates that MediaProof's hash chain correctly detects:
- Local modifications to payload, event_type, timestamp_utc, and metadata.
- Predecessor linkage tampering (incorrect prev_hash).
- Event deletion (middle event removed).
- Event insertion and reordering.
- Duplicate event IDs and duplicate sequence numbers.
- Missing, malformed, and duplicate genesis events.
- Empty chains and malformed event schemas.
"""

import copy
import hashlib
import pytest
from typing import Any, Dict

from provenance.event_serialization import (
    ProvenanceEvent,
    calculate_event_hash,
    canonical_serialize_event,
)
from provenance.hash_chain import (
    ChainErrorCode,
    ChainVerificationResult,
    EventHashChain,
    build_child_event,
    create_genesis_event,
    verify_event_chain,
)


# ==============================================================================
# Fixed Test Vectors for a 3-Event Reference Chain
# ==============================================================================

EXPECTED_H0 = "f1bed03e40b8d42dd380fbdf8d0d4d4719046af6b00204ca4c37ed3538c59995"
EXPECTED_BYTES_0 = (
    b'{"asset_id":"MP-CHAIN-TEST-001","event_id":"EVT-GENESIS-001","event_type":"GENESIS",'
    b'"payload":{"chain_id":"CHAIN-001","created_by":"test_suite"},"schema_version":"1.0",'
    b'"sequence_number":0,"timestamp_utc":"2026-10-19T10:00:00Z"}'
)

EXPECTED_H1 = "cb09a3da6949a1b1bfb2d67512ef5747a27037f6437af27fcd3209f8d75e5bb2"
EXPECTED_BYTES_1 = (
    b'{"asset_id":"MP-CHAIN-TEST-001","event_id":"EVT-ASSET-INGEST-002","event_type":"ASSET_INGEST",'
    b'"parent_event_id":"EVT-GENESIS-001","payload":{"file_size":1048576,"format":"mp4"},'
    b'"prev_hash":"f1bed03e40b8d42dd380fbdf8d0d4d4719046af6b00204ca4c37ed3538c59995",'
    b'"schema_version":"1.0","sequence_number":1,"timestamp_utc":"2026-10-19T10:05:00Z"}'
)

EXPECTED_H2 = "dbe669cc6982423540247ae519706e8df9355e15acc91ed0d24d74870ed233e8"
EXPECTED_BYTES_2 = (
    b'{"asset_id":"MP-CHAIN-TEST-001","event_id":"EVT-MEDIA-VERIFY-003","event_type":"MEDIA_VERIFY",'
    b'"parent_event_id":"EVT-ASSET-INGEST-002","payload":{"decision":"VERIFIED_AUTHENTIC","risk_score":0.05},'
    b'"prev_hash":"cb09a3da6949a1b1bfb2d67512ef5747a27037f6437af27fcd3209f8d75e5bb2",'
    b'"schema_version":"1.0","sequence_number":2,"timestamp_utc":"2026-10-19T10:10:00Z"}'
)


@pytest.fixture
def reference_chain():
    """Builds and returns the 3-event reference chain matching fixed golden vectors."""
    g = create_genesis_event(
        event_id="EVT-GENESIS-001",
        asset_id="MP-CHAIN-TEST-001",
        payload={"chain_id": "CHAIN-001", "created_by": "test_suite"},
        timestamp_utc="2026-10-19T10:00:00Z",
    )
    e1 = build_child_event(
        parent_event=g,
        event_id="EVT-ASSET-INGEST-002",
        event_type="ASSET_INGEST",
        asset_id="MP-CHAIN-TEST-001",
        payload={"file_size": 1048576, "format": "mp4"},
        timestamp_utc="2026-10-19T10:05:00Z",
    )
    e2 = build_child_event(
        parent_event=e1,
        event_id="EVT-MEDIA-VERIFY-003",
        event_type="MEDIA_VERIFY",
        asset_id="MP-CHAIN-TEST-001",
        payload={"decision": "VERIFIED_AUTHENTIC", "risk_score": 0.05},
        timestamp_utc="2026-10-19T10:10:00Z",
    )
    return [g, e1, e2]


# ==============================================================================
# 1. Deterministic Golden Vectors & Verification
# ==============================================================================

def test_deterministic_event_hashes_and_golden_chain_vectors(reference_chain):
    """
    Verifies that the reference 3-event chain produces exact expected bytes
    and SHA-256 digests against fixed test constants.
    """
    g, e1, e2 = reference_chain

    # Verify event 0 (genesis)
    b0 = canonical_serialize_event(g)
    h0 = calculate_event_hash(g)
    assert b0 == EXPECTED_BYTES_0
    assert h0 == EXPECTED_H0
    assert hashlib.sha256(b0).hexdigest() == EXPECTED_H0

    # Verify event 1
    b1 = canonical_serialize_event(e1)
    h1 = calculate_event_hash(e1)
    assert b1 == EXPECTED_BYTES_1
    assert h1 == EXPECTED_H1
    assert hashlib.sha256(b1).hexdigest() == EXPECTED_H1

    # Verify event 2
    b2 = canonical_serialize_event(e2)
    h2 = calculate_event_hash(e2)
    assert b2 == EXPECTED_BYTES_2
    assert h2 == EXPECTED_H2
    assert hashlib.sha256(b2).hexdigest() == EXPECTED_H2

    # Verify the complete chain succeeds
    result = verify_event_chain(reference_chain)
    assert result.is_valid is True
    assert result.total_events == 3
    assert result.verified_count == 3
    assert result.tip_hash == EXPECTED_H2
    assert result.error_code is None
    assert result.failing_event_id is None


def test_valid_genesis_event_and_multi_event_chain():
    """Tests creating a valid multi-event chain using the EventHashChain container."""
    chain = EventHashChain()
    g = chain.add_genesis(
        event_id="EVT-G",
        asset_id="MP-ASSET-01",
        payload={"genesis_info": "init"},
        timestamp_utc="2026-10-19T10:00:00Z",
    )
    assert len(chain) == 1
    assert chain.tip_hash == calculate_event_hash(g)

    e1 = chain.append_child(
        event_id="EVT-1",
        event_type="INGEST",
        asset_id="MP-ASSET-01",
        payload={"step": 1},
        timestamp_utc="2026-10-19T10:01:00Z",
    )
    assert len(chain) == 2
    assert chain.tip_hash == calculate_event_hash(e1)

    e2 = chain.append_child(
        event_id="EVT-2",
        event_type="ANALYZE",
        asset_id="MP-ASSET-01",
        payload={"step": 2},
        timestamp_utc="2026-10-19T10:02:00Z",
    )
    assert len(chain) == 3
    assert chain.tip_hash == calculate_event_hash(e2)

    res = chain.verify()
    assert res.is_valid is True
    assert res.verified_count == 3
    assert res.tip_hash == calculate_event_hash(e2)


# ==============================================================================
# 2. Tampering Detection (Payload, Event Type, Timestamp, Metadata)
# ==============================================================================

def test_tampered_payload_detected(reference_chain):
    """
    Modifying an event's payload invalidates its calculated hash, causing
    the successor event's prev_hash check to fail.
    """
    events = [e.to_dict() for e in reference_chain]

    # Tamper with Event 1's payload
    events[1]["payload"]["format"] = "avi"

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.PREV_HASH_MISMATCH
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"
    assert res.failing_sequence_number == 2
    assert res.verified_count == 2  # events 0 and 1 were valid locally, link to 2 broke


def test_tampered_event_type_detected(reference_chain):
    """Changing event_type breaks the downstream hash chain."""
    events = [e.to_dict() for e in reference_chain]
    events[1]["event_type"] = "MALICIOUS_RENAME"

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.PREV_HASH_MISMATCH
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"


def test_tampered_timestamp_detected(reference_chain):
    """Altering an event timestamp breaks downstream linkage."""
    events = [e.to_dict() for e in reference_chain]
    events[0]["timestamp_utc"] = "2026-10-19T09:59:59Z"

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.PREV_HASH_MISMATCH
    assert res.failing_event_id == "EVT-ASSET-INGEST-002"
    assert res.failing_sequence_number == 1


def test_tampered_metadata_detected(reference_chain):
    """Modifying metadata breaks canonical serialization and downstream linkage."""
    events = [e.to_dict() for e in reference_chain]
    events[1]["metadata"] = {"injected_field": "tampered_data"}

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.PREV_HASH_MISMATCH
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"


# ==============================================================================
# 3. Linkage, Sequence, and Structural Violations
# ==============================================================================

def test_incorrect_prev_hash_rejected(reference_chain):
    """Directly modifying prev_hash to an arbitrary value fails linkage check."""
    events = [e.to_dict() for e in reference_chain]
    events[1]["prev_hash"] = "0" * 64

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.PREV_HASH_MISMATCH
    assert res.failing_event_id == "EVT-ASSET-INGEST-002"
    assert res.failing_sequence_number == 1


def test_deleted_middle_event_detected(reference_chain):
    """Deleting a middle event creates both sequence gap and broken hash link."""
    # Delete Event 1, leaving Event 0 and Event 2
    events = [reference_chain[0], reference_chain[2]]

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.SEQUENCE_GAP
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"
    assert res.failing_sequence_number == 2
    assert res.verified_count == 1


def test_reordered_events_detected(reference_chain):
    """Reordering events in the chain breaks sequence and linkage continuity."""
    # Swap event 1 and event 2
    events = [reference_chain[0], reference_chain[2], reference_chain[1]]

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.SEQUENCE_GAP
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"


def test_duplicate_event_ids_rejected(reference_chain):
    """Duplicate event IDs within the same chain are strictly rejected."""
    events = [e.to_dict() for e in reference_chain]
    events[1]["event_id"] = "EVT-GENESIS-001"  # duplicate of index 0

    res = verify_event_chain(events)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.DUPLICATE_EVENT_ID
    assert res.failing_event_id == "EVT-GENESIS-001"
    assert res.verified_count == 1


def test_sequence_gaps_and_duplicate_sequence_numbers(reference_chain):
    """Sequence gaps and duplicate sequence numbers are rejected."""
    # 1. Duplicate sequence number
    events_dup_seq = [e.to_dict() for e in reference_chain]
    events_dup_seq[2]["sequence_number"] = 1  # index 1 is also 1

    res1 = verify_event_chain(events_dup_seq)
    assert res1.is_valid is False
    assert res1.error_code == ChainErrorCode.DUPLICATE_SEQUENCE_NUMBER

    # 2. Sequence gap
    events_gap = [e.to_dict() for e in reference_chain]
    events_gap[2]["sequence_number"] = 5  # gap from 1 to 5

    res2 = verify_event_chain(events_gap)
    assert res2.is_valid is False
    assert res2.error_code == ChainErrorCode.SEQUENCE_GAP


# ==============================================================================
# 4. Genesis Rules & Edge Cases
# ==============================================================================

def test_missing_or_duplicate_genesis_events(reference_chain):
    """
    Validates that:
    - Chains missing a genesis event at index 0 are rejected.
    - Chains with duplicate genesis-like events at index > 0 are rejected.
    """
    # 1. Missing genesis event at index 0 (starts at seq 1)
    events_no_gen = [reference_chain[1], reference_chain[2]]
    res1 = verify_event_chain(events_no_gen)
    assert res1.is_valid is False
    assert res1.error_code == ChainErrorCode.GENESIS_MALFORMED

    # 2. Duplicate genesis event at index 1
    events_dup_gen = [reference_chain[0], reference_chain[0]]
    res2 = verify_event_chain(events_dup_gen)
    assert res2.is_valid is False
    assert res2.error_code == ChainErrorCode.DUPLICATE_EVENT_ID

    # 3. Genesis-like event (seq 0) with distinct event_id at index 1
    gen2 = create_genesis_event(
        event_id="EVT-GENESIS-002",
        asset_id="MP-CHAIN-TEST-001",
        payload={"other": "data"},
        timestamp_utc="2026-10-19T10:01:00Z",
    )
    events_dup_gen2 = [reference_chain[0], gen2]
    res3 = verify_event_chain(events_dup_gen2)
    assert res3.is_valid is False
    assert res3.error_code == ChainErrorCode.GENESIS_DUPLICATE


def test_malformed_genesis_event_rejected():
    """Genesis event with non-zero sequence_number or non-None prev_hash is rejected."""
    # 1. Non-zero sequence number
    bad_gen1 = {
        "schema_version": "1.0",
        "event_id": "EVT-BAD-GEN-1",
        "event_type": "GENESIS",
        "timestamp_utc": "2026-10-19T10:00:00Z",
        "asset_id": "MP-ASSET-01",
        "payload": {},
        "sequence_number": 1,
    }
    res1 = verify_event_chain([bad_gen1])
    assert res1.is_valid is False
    assert res1.error_code == ChainErrorCode.GENESIS_MALFORMED

    # 2. Non-None prev_hash on genesis
    bad_gen2 = {
        "schema_version": "1.0",
        "event_id": "EVT-BAD-GEN-2",
        "event_type": "GENESIS",
        "timestamp_utc": "2026-10-19T10:00:00Z",
        "asset_id": "MP-ASSET-01",
        "payload": {},
        "sequence_number": 0,
        "prev_hash": "a" * 64,
    }
    res2 = verify_event_chain([bad_gen2])
    assert res2.is_valid is False
    assert res2.error_code == ChainErrorCode.GENESIS_MALFORMED


def test_empty_chain_behavior():
    """Verifying an empty chain returns is_valid=False with EMPTY_CHAIN error code."""
    res = verify_event_chain([])
    assert res.is_valid is False
    assert res.total_events == 0
    assert res.verified_count == 0
    assert res.error_code == ChainErrorCode.EMPTY_CHAIN


def test_invalid_event_schema_rejected():
    """Events failing schema or CES-1.0 canonicalization fail chain verification."""
    bad_event = {
        "schema_version": "1.0",
        "event_id": "EVT-BAD-SCHEMA",
        "event_type": "INVALID TYPE WITH SPACES",  # malformed event_type
        "timestamp_utc": "2026-10-19T10:00:00Z",
        "asset_id": "MP-ASSET-01",
        "payload": {},
        "sequence_number": 0,
    }
    res = verify_event_chain([bad_event])
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.INVALID_EVENT_SCHEMA


def test_event_hash_chain_container_safety(reference_chain):
    """Tests container invariants for EventHashChain."""
    chain = EventHashChain()

    # Cannot append child to empty chain
    with pytest.raises(ValueError, match="Cannot append child event to an empty chain"):
        chain.append_child(
            event_id="EVT-1",
            event_type="INGEST",
            asset_id="MP-01",
            payload={},
            timestamp_utc="2026-10-19T10:01:00Z",
        )

    # Add genesis
    chain.add_genesis(
        event_id="EVT-G",
        asset_id="MP-01",
        payload={"init": True},
        timestamp_utc="2026-10-19T10:00:00Z",
    )

    # Cannot add second genesis
    with pytest.raises(ValueError, match="already has a genesis event"):
        chain.add_genesis(
            event_id="EVT-G2",
            asset_id="MP-01",
            payload={},
            timestamp_utc="2026-10-19T10:00:00Z",
        )

    # Cannot manually append disconnected event
    bad_child = {
        "schema_version": "1.0",
        "event_id": "EVT-DISCONNECTED",
        "event_type": "INGEST",
        "timestamp_utc": "2026-10-19T10:05:00Z",
        "asset_id": "MP-01",
        "payload": {},
        "sequence_number": 1,
        "prev_hash": "f" * 64,  # wrong hash
    }
    with pytest.raises(ValueError, match="Failed to append event to chain"):
        chain.append(bad_child)

    # Iteration and indexing
    assert len(chain) == 1
    assert [ev.event_id for ev in chain] == ["EVT-G"]
    assert chain[0].event_id == "EVT-G"


# ==============================================================================
# 5. Phase 2.2.1 Trusted Chain-Tip Verification Tests
# ==============================================================================

def test_valid_chain_with_correct_expected_tip_hash(reference_chain):
    """
    Verifies that a valid chain verified against its exact expected tip hash
    succeeds with is_valid=True and matching tip_hash and expected_tip_hash.
    """
    res = verify_event_chain(reference_chain, expected_tip_hash=EXPECTED_H2)
    assert res.is_valid is True
    assert res.total_events == 3
    assert res.verified_count == 3
    assert res.tip_hash == EXPECTED_H2
    assert res.expected_tip_hash == EXPECTED_H2
    assert res.error_code is None
    assert res.failing_event_id is None

    # Test through EventHashChain container API
    chain = EventHashChain(reference_chain)
    chain_res = chain.verify(expected_tip_hash=EXPECTED_H2)
    assert chain_res.is_valid is True
    assert chain_res.tip_hash == EXPECTED_H2


def test_modified_final_event_payload_with_trusted_tip_hash(reference_chain):
    """
    A modified final event payload leaves internal predecessor linkage intact,
    but is caught immediately when verified against the trusted expected tip hash.
    """
    events = [e.to_dict() for e in reference_chain]
    events[2]["payload"]["decision"] = "TAMPERED_MANIPULATED"

    # With trusted tip hash -> must fail with TIP_HASH_MISMATCH
    res = verify_event_chain(events, expected_tip_hash=EXPECTED_H2)
    assert res.is_valid is False
    assert res.error_code == ChainErrorCode.TIP_HASH_MISMATCH
    assert res.failing_event_id == "EVT-MEDIA-VERIFY-003"
    assert res.failing_sequence_number == 2
    assert res.tip_hash != EXPECTED_H2
    assert res.expected_tip_hash == EXPECTED_H2


def test_modified_final_event_timestamp_and_metadata_with_trusted_tip_hash(reference_chain):
    """
    Modifying the final event's timestamp or metadata changes its canonical hash,
    which is detected as a TIP_HASH_MISMATCH against the trusted expected tip.
    """
    # 1. Tampered timestamp on final event
    events_ts = [e.to_dict() for e in reference_chain]
    events_ts[2]["timestamp_utc"] = "2026-10-19T10:10:01Z"

    res_ts = verify_event_chain(events_ts, expected_tip_hash=EXPECTED_H2)
    assert res_ts.is_valid is False
    assert res_ts.error_code == ChainErrorCode.TIP_HASH_MISMATCH
    assert res_ts.failing_event_id == "EVT-MEDIA-VERIFY-003"

    # 2. Tampered metadata on final event
    events_meta = [e.to_dict() for e in reference_chain]
    events_meta[2]["metadata"] = {"unauthorized": "injected"}

    res_meta = verify_event_chain(events_meta, expected_tip_hash=EXPECTED_H2)
    assert res_meta.is_valid is False
    assert res_meta.error_code == ChainErrorCode.TIP_HASH_MISMATCH
    assert res_meta.failing_event_id == "EVT-MEDIA-VERIFY-003"


def test_unexpected_appended_event_with_trusted_tip_hash(reference_chain):
    """
    When an attacker appends an unauthorized event to an otherwise valid chain,
    verifying against the previously trusted tip hash detects the unexpected append.
    """
    e3 = build_child_event(
        parent_event=reference_chain[2],
        event_id="EVT-UNAUTHORIZED-004",
        event_type="UNAUTHORIZED_APPEND",
        asset_id="MP-CHAIN-TEST-001",
        payload={"injected": True},
        timestamp_utc="2026-10-19T10:15:00Z",
    )
    expanded_chain = list(reference_chain) + [e3]

    # Without expected_tip_hash, internal linkage is valid
    res_internal = verify_event_chain(expanded_chain)
    assert res_internal.is_valid is True
    assert res_internal.total_events == 4

    # With the trusted expected tip (from event 2), the unexpected append is detected
    res_trusted = verify_event_chain(expanded_chain, expected_tip_hash=EXPECTED_H2)
    assert res_trusted.is_valid is False
    assert res_trusted.error_code == ChainErrorCode.TIP_HASH_MISMATCH
    assert res_trusted.failing_event_id == "EVT-UNAUTHORIZED-004"
    assert res_trusted.failing_sequence_number == 3


def test_incorrect_and_malformed_expected_tip_hash(reference_chain):
    """
    Verifies rejection of:
    - An incorrect (valid format, but non-matching) tip hash.
    - Malformed tip hashes (uppercase, wrong length, non-hex, whitespace, non-string).
    """
    # 1. Non-matching but valid format
    wrong_hash = "0" * 64
    res_wrong = verify_event_chain(reference_chain, expected_tip_hash=wrong_hash)
    assert res_wrong.is_valid is False
    assert res_wrong.error_code == ChainErrorCode.TIP_HASH_MISMATCH

    # 2. Malformed expected tip hashes
    for bad_tip in [
        EXPECTED_H2.upper(),         # uppercase
        "too_short",                 # short
        "a" * 63,                    # 63 chars
        "a" * 65,                    # 65 chars
        "z" * 64,                    # non-hex
        f" {EXPECTED_H2}",           # leading space
        f"{EXPECTED_H2} ",           # trailing space
        12345,                       # int
        None,                        # None is handled as omitted, but if passed in loop test below:
    ]:
        if bad_tip is None:
            continue
        res_bad = verify_event_chain(reference_chain, expected_tip_hash=bad_tip)
        assert res_bad.is_valid is False
        assert res_bad.error_code == ChainErrorCode.INVALID_EXPECTED_TIP_HASH


def test_verification_without_expected_tip_hash_limitation_documented(reference_chain):
    """
    Demonstrates and documents the critical security boundary:
    - Without an expected tip hash (expected_tip_hash=None), the verifier validates
      internal chain consistency and linkage continuity only. It CANNOT establish
      that the final event is authentic or that no unexpected events were appended.
    - With an expected tip hash, final-event mutations and appends are strictly detected.
    """
    events = [e.to_dict() for e in reference_chain]
    events[2]["payload"]["decision"] = "TAMPERED_IN_PLACE"

    # Internal consistency pass succeeds because events[0..1] and linkage 1->2 are valid
    res_internal = verify_event_chain(events, expected_tip_hash=None)
    assert res_internal.is_valid is True
    assert res_internal.expected_tip_hash is None

    # But with trusted tip hash, tampering of the tip is strictly detected
    res_trusted = verify_event_chain(events, expected_tip_hash=EXPECTED_H2)
    assert res_trusted.is_valid is False
    assert res_trusted.error_code == ChainErrorCode.TIP_HASH_MISMATCH

