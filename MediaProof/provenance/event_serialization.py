"""
Deterministic Canonical Serialization for MediaProof Provenance-Ledger Events.

Phase 2.1 & 2.1.1: Canonical Provenance Event Serialization

Specification: MediaProof Canonical Event Serialization Format v1.0 (MediaProof-CES-1.0)

IMPORTANT SPECIFICATION NOTICE:
MediaProof-CES-1.0 is a deterministic, versioned, domain-specific canonical serialization format.
It is explicitly NOT strict RFC 8785 (JSON Canonicalization Scheme / JCS). While it adopts core
canonical JSON principles (lexicographical key sorting, whitespace elimination, UTF-8 encoding),
it deliberately diverges from RFC 8785 in the following defined ways:
1. Unicode NFC Normalization: All string keys and string values are normalized using Unicode
   Normalization Form C (NFC) prior to serialization. (RFC 8785 explicitly forbids Unicode normalization).
2. Normalization Collision Rejection: If two distinct keys in any dictionary normalize to the same
   NFC string, MediaProof-CES-1.0 strictly rejects the event with an EventValidationError rather than
   silently overwriting or discarding data.
3. Strict Missing vs. Null Policy: Top-level optional fields set to None are omitted from canonical
   output (ensuring explicit None and omitted fields produce identical bytes), while payload and
   metadata dictionaries preserve explicit None as JSON `null`.
4. Domain Validation & Normalization: Timestamps must be timezone-aware and are normalized to canonical
   UTC ISO 8601 strings ending in 'Z'. Fractional seconds are supported up to 6 digits (microsecond precision).
   Timestamps with >6 fractional-second digits (including trailing zeroes beyond 6 digits) are strictly rejected
   with EventValidationError to prevent silent truncation. Non-finite floats (NaN, Infinity, -Infinity) are
   strictly rejected. Negative float zero (-0.0) is normalized to positive zero (0.0). Python's standard shortest
   round-trip float serialization is used.
5. Subsystem Isolation: Module contains zero database connections, zero network operations, and zero
   coupling to media analysis or watermark verification.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Union
import unicodedata

# ------------------------------------------------------------------------------
# Version & Field Constants
# ------------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS: Set[str] = {CURRENT_SCHEMA_VERSION}

# Regex patterns for structural validation
HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EVENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
STRICT_TIMESTAMP_REGEX = re.compile(
    r"^(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?(?P<tz>Z|[+-]\d{2}:\d{2})$"
)

REQUIRED_TOP_LEVEL_FIELDS: Set[str] = {
    "schema_version",
    "event_id",
    "event_type",
    "timestamp_utc",
    "asset_id",
    "payload",
}

OPTIONAL_TOP_LEVEL_FIELDS: Set[str] = {
    "sequence_number",
    "prev_hash",
    "parent_event_id",
    "signer_key_id",
    "metadata",
}

ALLOWED_TOP_LEVEL_FIELDS: Set[str] = REQUIRED_TOP_LEVEL_FIELDS | OPTIONAL_TOP_LEVEL_FIELDS


# ------------------------------------------------------------------------------
# Exception Hierarchy
# ------------------------------------------------------------------------------

class ProvenanceEventError(Exception):
    """Base exception for all provenance event errors."""


class EventValidationError(ProvenanceEventError):
    """Raised when an event fails schema, structural, or value validation."""


class UnsupportedSchemaVersionError(EventValidationError):
    """Raised when an event specifies an unsupported or malformed schema version."""


# ------------------------------------------------------------------------------
# Dataclass Definition
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceEvent:
    """
    Immutable representation of a versioned provenance ledger event.
    """
    event_id: str
    event_type: str
    timestamp_utc: str
    asset_id: str
    payload: Dict[str, Any]
    schema_version: str = CURRENT_SCHEMA_VERSION
    sequence_number: Optional[int] = None
    prev_hash: Optional[str] = None
    parent_event_id: Optional[str] = None
    signer_key_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self, include_nulls: bool = False) -> Dict[str, Any]:
        """
        Converts event to a dictionary representation.
        If include_nulls is False, optional fields with None value are omitted.
        """
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "asset_id": self.asset_id,
            "payload": self.payload,
        }
        if self.sequence_number is not None or include_nulls:
            d["sequence_number"] = self.sequence_number
        if self.prev_hash is not None or include_nulls:
            d["prev_hash"] = self.prev_hash
        if self.parent_event_id is not None or include_nulls:
            d["parent_event_id"] = self.parent_event_id
        if self.signer_key_id is not None or include_nulls:
            d["signer_key_id"] = self.signer_key_id
        if self.metadata is not None or include_nulls:
            d["metadata"] = self.metadata
        return d

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Returns the canonicalized dictionary ready for deterministic serialization."""
        return canonicalize_event_dict(self)

    def to_canonical_bytes(self) -> bytes:
        """Returns the canonical UTF-8 bytes for this event."""
        return canonical_serialize_event(self)

    def calculate_hash(self) -> str:
        """Calculates and returns the SHA-256 hex digest over this event's canonical bytes."""
        return calculate_event_hash(self)


# ------------------------------------------------------------------------------
# Normalization & Validation Helpers
# ------------------------------------------------------------------------------

def normalize_timestamp_utc(ts: Union[str, datetime]) -> str:
    """
    Validates and normalizes a timestamp into canonical UTC ISO 8601 string format ending in 'Z'.

    Rules (MediaProof-CES-1.0):
    - Strict grammar: YYYY-MM-DDTHH:MM:SS[.ffffff](Z|±HH:MM)
    - Validates the entire input against strict grammar before passing to parser.
    - Rejects whitespace (leading, trailing, interior), trailing text, and non-conforming characters.
    - Timezone offset must be 'Z' or '±HH:MM'. Abbreviated offsets (e.g. '+05'), unseparated offsets
      (e.g. '+0530'), offsets with seconds (e.g. '+05:30:00'), and malformed offsets are rejected.
    - Fractional seconds:
      - Supported up to 6 digits (microsecond precision) prefixed with a decimal point '.'.
      - Fractional seconds with > 6 digits (including trailing zeroes beyond 6 digits) are strictly
        rejected with EventValidationError to prevent silent precision loss / truncation.
      - Comma fraction separators and malformed fractions are strictly rejected.
    - Calendar and clock validity (impossible dates, leap years, 0..23 hours, 0..59 minutes/seconds,
      and valid timezone ranges) are strictly enforced.
    - Output format: 'YYYY-MM-DDTHH:MM:SSZ' (zero microseconds) or
                     'YYYY-MM-DDTHH:MM:SS.ffffffZ' (non-zero microseconds).
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            raise EventValidationError("Datetime object must be timezone-aware (UTC).")
        dt_utc = ts.astimezone(timezone.utc)
    elif isinstance(ts, str):
        # 1. Reject whitespace anywhere in the string
        if re.search(r"\s", ts):
            raise EventValidationError(f"Timestamp '{ts}' contains whitespace; strict ISO 8601 grammar required.")

        # 2. Match against exact grammar
        m = STRICT_TIMESTAMP_REGEX.match(ts)
        if not m:
            if "T" not in ts:
                raise EventValidationError(f"Timestamp '{ts}' must be in ISO 8601 format with 'T' date/time separator.")
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$", ts):
                raise EventValidationError(f"Timestamp '{ts}' lacks timezone information; explicit UTC offset ('Z' or '±HH:MM') required.")
            if "," in ts:
                raise EventValidationError(f"Timestamp '{ts}' contains invalid fraction separator ','; decimal point '.' required.")
            raise EventValidationError(
                f"Timestamp '{ts}' does not conform to strict MediaProof-CES-1.0 grammar 'YYYY-MM-DDTHH:MM:SS[.ffffff](Z|±HH:MM)'."
            )

        frac = m.group("frac")
        if frac is not None and len(frac) > 6:
            raise EventValidationError(
                f"Timestamp '{ts}' exceeds maximum supported precision of 6 fractional-second digits "
                f"({len(frac)} digits provided). Silent truncation is not permitted in MediaProof-CES-1.0."
            )

        iso_to_parse = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        try:
            dt = datetime.fromisoformat(iso_to_parse)
        except Exception as e:
            raise EventValidationError(f"Invalid calendar date, time, or offset in timestamp '{ts}': {e}") from e

        if dt.tzinfo is None:
            raise EventValidationError(f"Timestamp '{ts}' lacks timezone information; explicit UTC required.")
        dt_utc = dt.astimezone(timezone.utc)
    else:
        raise EventValidationError(f"Timestamp must be an ISO 8601 string or datetime object, got {type(ts).__name__}.")

    if dt_utc.microsecond == 0:
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize_value(val: Any) -> Any:
    """
    Recursively validates and canonicalizes values for deterministic JSON serialization:
    - Strings: Normalized to Unicode NFC.
    - Floats: Must be finite. -0.0 normalized to 0.0.
    - Integers: Preserved (booleans strictly distinguished).
    - Booleans: Preserved as True/False.
    - Dictionaries: Keys must be strings, normalized to NFC, sorted lexicographically.
    - Lists/Tuples: Converted to canonical list preserving order.
    - None: Preserved as None (serializes to JSON null).
    - Other types: Rejected.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise EventValidationError(f"Non-finite float value '{val}' is not permitted in canonical event.")
        if val == 0.0:
            return 0.0
        return val
    if isinstance(val, str):
        try:
            val.encode("utf-8")
        except UnicodeEncodeError as e:
            raise EventValidationError(f"Invalid Unicode string containing surrogate code points: {e}") from e
        return unicodedata.normalize("NFC", val)
    if isinstance(val, dict):
        norm_dict = {}
        seen_keys: Dict[str, str] = {}
        for k in sorted(val.keys()):
            if not isinstance(k, str):
                raise EventValidationError(f"Dictionary keys must be strings, got {type(k).__name__}: {k!r}")
            try:
                k.encode("utf-8")
            except UnicodeEncodeError as e:
                raise EventValidationError(f"Invalid Unicode key containing surrogate code points: {e}") from e
            norm_k = unicodedata.normalize("NFC", k)
            if norm_k in seen_keys:
                raise EventValidationError(
                    f"Duplicate key collision after Unicode NFC normalization: "
                    f"keys {seen_keys[norm_k]!r} and {k!r} both normalize to {norm_k!r}."
                )
            seen_keys[norm_k] = k
            norm_dict[norm_k] = _normalize_value(val[k])
        return norm_dict
    if isinstance(val, (list, tuple)):
        return [_normalize_value(item) for item in val]

    raise EventValidationError(f"Unsupported value type in canonical event: {type(val).__name__} ({val!r})")


# ------------------------------------------------------------------------------
# Core Public API
# ------------------------------------------------------------------------------

def validate_event(event: Union[ProvenanceEvent, Dict[str, Any]]) -> ProvenanceEvent:
    """
    Validates a provenance event against schema and structural constraints.

    Args:
        event: Either a ProvenanceEvent instance or a dictionary representing an event.

    Returns:
        A validated, normalized ProvenanceEvent instance.

    Raises:
        UnsupportedSchemaVersionError: If schema_version is unsupported.
        EventValidationError: If any required field is missing, invalid, or malformed.
    """
    if isinstance(event, ProvenanceEvent):
        raw: Dict[str, Any] = {
            "schema_version": event.schema_version,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp_utc": event.timestamp_utc,
            "asset_id": event.asset_id,
            "payload": event.payload,
            "sequence_number": event.sequence_number,
            "prev_hash": event.prev_hash,
            "parent_event_id": event.parent_event_id,
            "signer_key_id": event.signer_key_id,
            "metadata": event.metadata,
        }
    elif isinstance(event, dict):
        raw = {}
        seen_top_keys: Dict[str, str] = {}
        for k, v in event.items():
            if not isinstance(k, str):
                raise EventValidationError(f"Top-level event keys must be strings, got {type(k).__name__}: {k!r}")
            try:
                k.encode("utf-8")
            except UnicodeEncodeError as e:
                raise EventValidationError(f"Invalid Unicode top-level key containing surrogate code points: {e}") from e
            norm_k = unicodedata.normalize("NFC", k)
            if norm_k in seen_top_keys:
                raise EventValidationError(
                    f"Duplicate top-level key collision after Unicode NFC normalization: "
                    f"keys {seen_top_keys[norm_k]!r} and {k!r} both normalize to {norm_k!r}."
                )
            seen_top_keys[norm_k] = k
            raw[norm_k] = v
    else:
        raise EventValidationError(f"Event must be a ProvenanceEvent or dict, got {type(event).__name__}.")

    # 1. Check for unrecognized top-level fields
    extra_fields = set(raw.keys()) - ALLOWED_TOP_LEVEL_FIELDS
    if extra_fields:
        raise EventValidationError(f"Unrecognized top-level field(s) in event: {sorted(extra_fields)}")

    # 2. Check for missing required top-level fields
    missing_fields = REQUIRED_TOP_LEVEL_FIELDS - set(raw.keys())
    if missing_fields:
        raise EventValidationError(f"Missing required top-level field(s) in event: {sorted(missing_fields)}")

    # 3. Validate schema_version
    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedSchemaVersionError(
            f"schema_version must be a string, got {type(version).__name__}: {version!r}"
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(
            f"Unsupported schema_version '{version}'. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    # 4. Validate event_id
    event_id = raw.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise EventValidationError("event_id must be a non-empty string.")
    norm_event_id = unicodedata.normalize("NFC", event_id.strip())

    # 5. Validate event_type
    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise EventValidationError("event_type must be a non-empty string.")
    clean_event_type = event_type.strip()
    if not EVENT_TYPE_PATTERN.match(clean_event_type):
        raise EventValidationError(
            f"event_type '{event_type}' is invalid; must contain only alphanumeric characters and underscores."
        )
    norm_event_type = unicodedata.normalize("NFC", clean_event_type)

    # 6. Validate timestamp_utc
    raw_ts = raw.get("timestamp_utc")
    norm_ts = normalize_timestamp_utc(raw_ts)

    # 7. Validate asset_id
    asset_id = raw.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise EventValidationError("asset_id must be a non-empty string.")
    norm_asset_id = unicodedata.normalize("NFC", asset_id.strip())

    # 8. Validate payload
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise EventValidationError(f"payload must be a dictionary, got {type(payload).__name__}.")
    norm_payload = _normalize_value(payload)

    # 9. Validate optional sequence_number
    seq_num = raw.get("sequence_number")
    if seq_num is not None:
        if isinstance(seq_num, bool) or not isinstance(seq_num, int):
            raise EventValidationError(f"sequence_number must be an integer, got {type(seq_num).__name__}: {seq_num!r}")
        if seq_num < 0:
            raise EventValidationError(f"sequence_number cannot be negative, got {seq_num}.")

    # 10. Validate optional prev_hash
    prev_hash = raw.get("prev_hash")
    norm_prev_hash: Optional[str] = None
    if prev_hash is not None:
        if not isinstance(prev_hash, str) or not HEX_64_PATTERN.match(prev_hash):
            raise EventValidationError(
                f"prev_hash '{prev_hash}' is invalid; must be a 64-character lowercase hexadecimal SHA-256 hash."
            )
        norm_prev_hash = prev_hash

    # 11. Validate optional parent_event_id
    parent_event_id = raw.get("parent_event_id")
    norm_parent_id: Optional[str] = None
    if parent_event_id is not None:
        if not isinstance(parent_event_id, str) or not parent_event_id.strip():
            raise EventValidationError("parent_event_id must be a non-empty string if specified.")
        norm_parent_id = unicodedata.normalize("NFC", parent_event_id.strip())

    # 12. Validate optional signer_key_id
    signer_key_id = raw.get("signer_key_id")
    norm_signer_id: Optional[str] = None
    if signer_key_id is not None:
        if not isinstance(signer_key_id, str) or not signer_key_id.strip():
            raise EventValidationError("signer_key_id must be a non-empty string if specified.")
        norm_signer_id = unicodedata.normalize("NFC", signer_key_id.strip())

    # 13. Validate optional metadata
    metadata = raw.get("metadata")
    norm_metadata: Optional[Dict[str, Any]] = None
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise EventValidationError(f"metadata must be a dictionary, got {type(metadata).__name__}.")
        norm_metadata = _normalize_value(metadata)

    return ProvenanceEvent(
        schema_version=version,
        event_id=norm_event_id,
        event_type=norm_event_type,
        timestamp_utc=norm_ts,
        asset_id=norm_asset_id,
        payload=norm_payload,
        sequence_number=seq_num,
        prev_hash=norm_prev_hash,
        parent_event_id=norm_parent_id,
        signer_key_id=norm_signer_id,
        metadata=norm_metadata,
    )


def canonicalize_event_dict(event: Union[ProvenanceEvent, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates and canonicalizes an event into a clean, normalized Python dictionary.

    Optional fields with None values are omitted to ensure deterministic representation.
    All keys and nested keys are lexicographically sorted, strings are NFC normalized,
    and timestamps follow canonical UTC ISO 8601 formatting.
    """
    validated = validate_event(event)

    canon: Dict[str, Any] = {
        "asset_id": validated.asset_id,
        "event_id": validated.event_id,
        "event_type": validated.event_type,
        "payload": validated.payload,
        "schema_version": validated.schema_version,
        "timestamp_utc": validated.timestamp_utc,
    }

    if validated.sequence_number is not None:
        canon["sequence_number"] = validated.sequence_number
    if validated.prev_hash is not None:
        canon["prev_hash"] = validated.prev_hash
    if validated.parent_event_id is not None:
        canon["parent_event_id"] = validated.parent_event_id
    if validated.signer_key_id is not None:
        canon["signer_key_id"] = validated.signer_key_id
    if validated.metadata is not None:
        canon["metadata"] = validated.metadata

    # Re-sort top-level keys lexicographically
    return {k: canon[k] for k in sorted(canon.keys())}


def canonical_serialize_event(event: Union[ProvenanceEvent, Dict[str, Any]]) -> bytes:
    """
    Serializes a provenance event into deterministic, canonical UTF-8 bytes
    according to MediaProof-CES-1.0.

    - Keys sorted lexicographically at all levels.
    - No whitespace between tokens (separators: ',' and ':').
    - Unicode NFC normalized strings encoded in UTF-8 without unnecessary escapes.
    - Non-finite floats rejected.
    """
    canon_dict = canonicalize_event_dict(event)
    json_str = json.dumps(
        canon_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return json_str.encode("utf-8")
    except UnicodeEncodeError as e:
        raise EventValidationError(f"UTF-8 encoding failed for canonical event: {e}") from e


def calculate_event_hash(event: Union[ProvenanceEvent, Dict[str, Any]]) -> str:
    """
    Calculates the canonical SHA-256 hash of a provenance event.

    Args:
        event: Either a ProvenanceEvent instance or a dictionary.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest string.
    """
    canonical_bytes = canonical_serialize_event(event)
    return hashlib.sha256(canonical_bytes).hexdigest().lower()


def event_from_dict(d: Dict[str, Any]) -> ProvenanceEvent:
    """Convenience factory function: validates and creates a ProvenanceEvent from a dictionary."""
    return validate_event(d)
