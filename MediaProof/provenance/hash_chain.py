"""
Tamper-Evident Event Hash Chain for MediaProof Provenance-Ledger Events.

Phase 2.2: Tamper-Evident Event Hash Chain

Objective:
Creates a deterministic, verifiable chain of provenance events built on top
of MediaProof-CES-1.0 canonical event serialization.

Cryptographic Scope & Security Notice:
- This module provides tamper-evidence against localized modification, insertion,
  deletion, and reordering of provenance ledger events.
- It verifies hash-link continuity via SHA-256 over exact canonical event bytes.
- LIMITATION NOTICE: Hash chains alone do NOT provide cryptographic author identity
  or non-repudiation. A party possessing full write access to storage can recompute
  hashes from any modified event forward. Digital signatures (Phase 2.3) and key
  management are required to prevent total history rewrite.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from provenance.event_serialization import (
    ProvenanceEvent,
    calculate_event_hash,
    canonical_serialize_event,
    validate_event,
)

HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# ------------------------------------------------------------------------------
# Error Codes & Verification Result
# ------------------------------------------------------------------------------

class ChainErrorCode:
    """Standardized failure classification codes for hash-chain verification."""
    EMPTY_CHAIN = "EMPTY_CHAIN"
    INVALID_EVENT_SCHEMA = "INVALID_EVENT_SCHEMA"
    GENESIS_MISSING = "GENESIS_MISSING"
    GENESIS_MALFORMED = "GENESIS_MALFORMED"
    GENESIS_DUPLICATE = "GENESIS_DUPLICATE"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    DUPLICATE_SEQUENCE_NUMBER = "DUPLICATE_SEQUENCE_NUMBER"
    PREV_HASH_MISSING = "PREV_HASH_MISSING"
    PREV_HASH_MISMATCH = "PREV_HASH_MISMATCH"
    DUPLICATE_EVENT_ID = "DUPLICATE_EVENT_ID"
    INVALID_EVENT_ORDER = "INVALID_EVENT_ORDER"
    TIP_HASH_MISMATCH = "TIP_HASH_MISMATCH"
    INVALID_EXPECTED_TIP_HASH = "INVALID_EXPECTED_TIP_HASH"


@dataclass(frozen=True)
class ChainVerificationResult:
    """
    Structured outcome of a hash-chain verification pass.

    Attributes:
        is_valid: True if and only if the entire chain is structurally and
                  cryptographically intact from genesis to tip.
        total_events: Total number of events presented for verification.
        verified_count: Number of consecutive events verified successfully
                        before encountering an error (or equal to total_events).
        failing_event_id: event_id of the first failing event, if any.
        failing_sequence_number: sequence_number of the first failing event, if any.
        error_code: ChainErrorCode member identifying the category of failure.
        error_message: Human-readable technical diagnostic explanation.
        tip_hash: SHA-256 hex digest of the final event in the chain, if valid.
        expected_tip_hash: Expected tip hash supplied for verification, if any.
    """
    is_valid: bool
    total_events: int
    verified_count: int
    failing_event_id: Optional[str] = None
    failing_sequence_number: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    tip_hash: Optional[str] = None
    expected_tip_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary representation of verification result."""
        return {
            "is_valid": self.is_valid,
            "total_events": self.total_events,
            "verified_count": self.verified_count,
            "failing_event_id": self.failing_event_id,
            "failing_sequence_number": self.failing_sequence_number,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "tip_hash": self.tip_hash,
            "expected_tip_hash": self.expected_tip_hash,
        }


# ------------------------------------------------------------------------------
# Core Chain Verification Logic
# ------------------------------------------------------------------------------

def verify_event_chain(
    events: Sequence[Union[ProvenanceEvent, Dict[str, Any]]],
    expected_tip_hash: Optional[str] = None,
) -> ChainVerificationResult:
    """
    Verifies an ordered sequence of provenance events from genesis to tip.

    Validation passes executed for each event:
    1. Schema & Canonical Validity: Validates against MediaProof-CES-1.0 rules.
    2. Event ID Uniqueness: Rejects duplicate event IDs anywhere within the chain.
    3. Genesis Event Rules (Index 0):
       - Must have sequence_number == 0.
       - Must have prev_hash == None (omitted in canonical form).
    4. Non-Genesis Event Rules (Index > 0):
       - Must NOT be a genesis-like event (sequence_number == 0 or prev_hash is None).
       - Sequence number must strictly equal predecessor sequence_number + 1.
       - prev_hash must equal the exact SHA-256 canonical hash of the predecessor.
    5. Hash-Link Continuity: Ensures no missing, inserted, deleted, or altered events.
    6. Trusted Tip Verification (Optional):
       - If expected_tip_hash is supplied, validates that it is a 64-character lowercase hex digest
         and that the final event's calculated SHA-256 hash strictly matches expected_tip_hash.

    SECURITY LIMITATION NOTICE (When expected_tip_hash is None):
    When expected_tip_hash is omitted, verify_event_chain checks internal chain consistency and
    linkage continuity only. It CANNOT establish that the final event is authentic or unchanged,
    nor can it detect whether subsequent unauthorized events were appended to the chain.
    To detect final-event tampering and unauthorized appends, callers must supply a trusted
    expected_tip_hash anchored out-of-band or via digital signatures (Phase 2.3).

    Args:
        events: An ordered sequence of ProvenanceEvent objects or event dicts.
        expected_tip_hash: Optional trusted SHA-256 tip hash (64 lowercase hex characters).

    Returns:
        ChainVerificationResult detailing validity, verified count, tip hash, and any failure.
    """
    if expected_tip_hash is not None:
        if not isinstance(expected_tip_hash, str) or not HEX_64_PATTERN.match(expected_tip_hash):
            return ChainVerificationResult(
                is_valid=False,
                total_events=len(events),
                verified_count=0,
                error_code=ChainErrorCode.INVALID_EXPECTED_TIP_HASH,
                error_message=(
                    f"expected_tip_hash '{expected_tip_hash}' is invalid; "
                    f"must be a 64-character lowercase hexadecimal SHA-256 digest."
                ),
                expected_tip_hash=str(expected_tip_hash) if expected_tip_hash is not None else None,
            )

    if not events:
        return ChainVerificationResult(
            is_valid=False,
            total_events=0,
            verified_count=0,
            error_code=ChainErrorCode.EMPTY_CHAIN,
            error_message="Event chain is empty; at least one genesis event is required.",
            expected_tip_hash=expected_tip_hash,
        )

    seen_event_ids: Set[str] = set()
    prev_hash: Optional[str] = None
    prev_seq: Optional[int] = None

    for i, raw_event in enumerate(events):
        # 1. Validate event structure & canonical schema
        try:
            event = validate_event(raw_event)
        except Exception as e:
            raw_id = getattr(raw_event, "event_id", None) or (
                raw_event.get("event_id") if isinstance(raw_event, dict) else None
            )
            raw_seq = getattr(raw_event, "sequence_number", None) or (
                raw_event.get("sequence_number") if isinstance(raw_event, dict) else None
            )
            return ChainVerificationResult(
                is_valid=False,
                total_events=len(events),
                verified_count=i,
                failing_event_id=str(raw_id) if raw_id is not None else None,
                failing_sequence_number=raw_seq if isinstance(raw_seq, int) else None,
                error_code=ChainErrorCode.INVALID_EVENT_SCHEMA,
                error_message=f"Event at index {i} failed schema validation: {e}",
                expected_tip_hash=expected_tip_hash,
            )

        # 2. Check for duplicate event IDs within the chain
        if event.event_id in seen_event_ids:
            return ChainVerificationResult(
                is_valid=False,
                total_events=len(events),
                verified_count=i,
                failing_event_id=event.event_id,
                failing_sequence_number=event.sequence_number,
                error_code=ChainErrorCode.DUPLICATE_EVENT_ID,
                error_message=f"Duplicate event_id '{event.event_id}' detected at index {i}.",
            )
        seen_event_ids.add(event.event_id)

        # 3. Genesis event verification (Index 0)
        if i == 0:
            if event.sequence_number is None or event.sequence_number != 0:
                return ChainVerificationResult(
                    is_valid=False,
                    total_events=len(events),
                    verified_count=0,
                    failing_event_id=event.event_id,
                    failing_sequence_number=event.sequence_number,
                    error_code=ChainErrorCode.GENESIS_MALFORMED,
                    error_message=(
                        f"Genesis event at index 0 must have sequence_number=0, "
                        f"got {event.sequence_number}."
                    ),
                )
            if event.prev_hash is not None:
                return ChainVerificationResult(
                    is_valid=False,
                    total_events=len(events),
                    verified_count=0,
                    failing_event_id=event.event_id,
                    failing_sequence_number=event.sequence_number,
                    error_code=ChainErrorCode.GENESIS_MALFORMED,
                    error_message=(
                        f"Genesis event at index 0 must have prev_hash=None, "
                        f"got '{event.prev_hash}'."
                    ),
                )
        else:
            # 4. Subsequent events verification (Index > 0)
            if event.sequence_number == 0 or event.prev_hash is None:
                return ChainVerificationResult(
                    is_valid=False,
                    total_events=len(events),
                    verified_count=i,
                    failing_event_id=event.event_id,
                    failing_sequence_number=event.sequence_number,
                    error_code=ChainErrorCode.GENESIS_DUPLICATE,
                    error_message=(
                        f"Unexpected genesis-like event at index {i}; "
                        f"genesis event must be unique at index 0."
                    ),
                )

            # Sequence continuity check
            expected_seq = prev_seq + 1  # type: ignore[operator]
            if event.sequence_number != expected_seq:
                if event.sequence_number == prev_seq:
                    err_code = ChainErrorCode.DUPLICATE_SEQUENCE_NUMBER
                elif event.sequence_number < prev_seq:  # type: ignore[operator]
                    err_code = ChainErrorCode.INVALID_EVENT_ORDER
                else:
                    err_code = ChainErrorCode.SEQUENCE_GAP
                return ChainVerificationResult(
                    is_valid=False,
                    total_events=len(events),
                    verified_count=i,
                    failing_event_id=event.event_id,
                    failing_sequence_number=event.sequence_number,
                    error_code=err_code,
                    error_message=(
                        f"Sequence discontinuity at index {i}: "
                        f"expected sequence_number={expected_seq}, "
                        f"got {event.sequence_number}."
                    ),
                )

            # Previous-hash linkage check
            if event.prev_hash != prev_hash:
                return ChainVerificationResult(
                    is_valid=False,
                    total_events=len(events),
                    verified_count=i,
                    failing_event_id=event.event_id,
                    failing_sequence_number=event.sequence_number,
                    error_code=ChainErrorCode.PREV_HASH_MISMATCH,
                    error_message=(
                        f"Broken hash link at index {i}: event.prev_hash '{event.prev_hash}' "
                        f"does not match expected predecessor hash '{prev_hash}'."
                    ),
                )

        # Compute this event's canonical hash for the next link
        current_hash = calculate_event_hash(event)
        prev_hash = current_hash
        prev_seq = event.sequence_number

    # 6. Verify trusted tip hash if expected_tip_hash was supplied
    if expected_tip_hash is not None and prev_hash != expected_tip_hash:
        final_event = events[-1]
        failing_id = (
            final_event.event_id
            if isinstance(final_event, ProvenanceEvent)
            else (final_event.get("event_id") if isinstance(final_event, dict) else None)
        )
        return ChainVerificationResult(
            is_valid=False,
            total_events=len(events),
            verified_count=len(events),
            failing_event_id=failing_id,
            failing_sequence_number=prev_seq,
            error_code=ChainErrorCode.TIP_HASH_MISMATCH,
            error_message=(
                f"Chain tip hash mismatch: calculated final event hash '{prev_hash}' "
                f"does not match expected trusted tip '{expected_tip_hash}'."
            ),
            tip_hash=prev_hash,
            expected_tip_hash=expected_tip_hash,
        )

    return ChainVerificationResult(
        is_valid=True,
        total_events=len(events),
        verified_count=len(events),
        tip_hash=prev_hash,
        expected_tip_hash=expected_tip_hash,
    )


# ------------------------------------------------------------------------------
# Chain Construction & Append Helpers
# ------------------------------------------------------------------------------

def create_genesis_event(
    event_id: str,
    asset_id: str,
    payload: Dict[str, Any],
    timestamp_utc: str,
    event_type: str = "GENESIS",
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEvent:
    """
    Constructs and validates a genesis event for a new hash chain.

    Rules:
    - sequence_number is fixed to 0.
    - prev_hash is fixed to None.
    """
    return validate_event(
        ProvenanceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp_utc=timestamp_utc,
            asset_id=asset_id,
            payload=payload,
            sequence_number=0,
            prev_hash=None,
            metadata=metadata,
        )
    )


def build_child_event(
    parent_event: Union[ProvenanceEvent, Dict[str, Any]],
    event_id: str,
    event_type: str,
    asset_id: str,
    payload: Dict[str, Any],
    timestamp_utc: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEvent:
    """
    Constructs and validates a child event correctly linked to its predecessor.

    Rules:
    - sequence_number = parent.sequence_number + 1.
    - prev_hash = calculate_event_hash(parent).
    """
    parent = validate_event(parent_event)
    parent_hash = calculate_event_hash(parent)
    next_seq = (parent.sequence_number or 0) + 1

    return validate_event(
        ProvenanceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp_utc=timestamp_utc,
            asset_id=asset_id,
            payload=payload,
            sequence_number=next_seq,
            prev_hash=parent_hash,
            parent_event_id=parent.event_id,
            metadata=metadata,
        )
    )


class EventHashChain:
    """
    Append-only container representing an ordered, verifiable event chain.
    """

    def __init__(self, events: Optional[Sequence[Union[ProvenanceEvent, Dict[str, Any]]]] = None) -> None:
        self._events: List[ProvenanceEvent] = []
        if events:
            # Validate input events
            ver = verify_event_chain(events)
            if not ver.is_valid:
                raise ValueError(f"Cannot initialize chain from invalid event sequence: {ver.error_message}")
            for ev in events:
                self._events.append(validate_event(ev))

    @property
    def events(self) -> List[ProvenanceEvent]:
        """Returns a copy of the events list."""
        return list(self._events)

    @property
    def tip_event(self) -> Optional[ProvenanceEvent]:
        """Returns the latest event in the chain, or None if empty."""
        return self._events[-1] if self._events else None

    @property
    def tip_hash(self) -> Optional[str]:
        """Returns the SHA-256 hash of the latest event, or None if empty."""
        if not self._events:
            return None
        return calculate_event_hash(self._events[-1])

    def add_genesis(
        self,
        event_id: str,
        asset_id: str,
        payload: Dict[str, Any],
        timestamp_utc: str,
        event_type: str = "GENESIS",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """Adds the single genesis event to an empty chain."""
        if self._events:
            raise ValueError("Chain already has a genesis event; additional genesis events are prohibited.")
        genesis = create_genesis_event(
            event_id=event_id,
            asset_id=asset_id,
            payload=payload,
            timestamp_utc=timestamp_utc,
            event_type=event_type,
            metadata=metadata,
        )
        self._events.append(genesis)
        return genesis

    def append_child(
        self,
        event_id: str,
        event_type: str,
        asset_id: str,
        payload: Dict[str, Any],
        timestamp_utc: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """Constructs and appends a child event linked to the current tip."""
        if not self._events:
            raise ValueError("Cannot append child event to an empty chain; add a genesis event first.")
        child = build_child_event(
            parent_event=self._events[-1],
            event_id=event_id,
            event_type=event_type,
            asset_id=asset_id,
            payload=payload,
            timestamp_utc=timestamp_utc,
            metadata=metadata,
        )
        self._events.append(child)
        return child

    def append(self, event: Union[ProvenanceEvent, Dict[str, Any]]) -> None:
        """Appends a pre-constructed event after validating chain continuity."""
        candidate_chain = list(self._events) + [event]
        ver = verify_event_chain(candidate_chain)
        if not ver.is_valid:
            raise ValueError(f"Failed to append event to chain: {ver.error_message}")
        self._events.append(validate_event(event))

    def verify(self, expected_tip_hash: Optional[str] = None) -> ChainVerificationResult:
        """
        Verifies the integrity of the entire chain, optionally against a trusted expected tip hash.
        """
        return verify_event_chain(self._events, expected_tip_hash=expected_tip_hash)

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, idx: int) -> ProvenanceEvent:
        return self._events[idx]

    def __iter__(self):
        return iter(self._events)
