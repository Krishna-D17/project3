# Package initialization for provenance module

from provenance.event_serialization import (
    CURRENT_SCHEMA_VERSION,
    EventValidationError,
    ProvenanceEvent,
    ProvenanceEventError,
    UnsupportedSchemaVersionError,
    calculate_event_hash,
    canonical_serialize_event,
    canonicalize_event_dict,
    event_from_dict,
    validate_event,
)
from provenance.hash_chain import (
    ChainErrorCode,
    ChainVerificationResult,
    EventHashChain,
    build_child_event,
    create_genesis_event,
    verify_event_chain,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ProvenanceEvent",
    "ProvenanceEventError",
    "EventValidationError",
    "UnsupportedSchemaVersionError",
    "validate_event",
    "canonicalize_event_dict",
    "canonical_serialize_event",
    "calculate_event_hash",
    "event_from_dict",
    "ChainErrorCode",
    "ChainVerificationResult",
    "EventHashChain",
    "create_genesis_event",
    "build_child_event",
    "verify_event_chain",
]

