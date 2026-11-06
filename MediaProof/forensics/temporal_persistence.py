"""
Deterministic Temporal Perceptual Difference-Hash (dHash) Persistence.

Implements Layer 1 (Temporal Fingerprint Persistence & Retrieval) for MediaProof.
Stores and retrieves temporal visual fingerprints and extraction metadata
compatibly associated with registered MediaProof assets.

Architecture:
- Table `temporal_fingerprints`:
    Holds extraction-level metadata (algorithm version, sampling interval,
    video duration, fps, sample counts, extraction status, availability flag,
    and schema version).
- Table `temporal_fingerprint_samples`:
    Holds ordered (sample_index, timestamp_seconds, frame_index, dhash_hex, dhash_int)
    for successful sequences.
- Integrity & Safety:
    - Atomicity: Sequence writes are wrapped in single SQLite transactions (no partial writes).
    - Validation on Save: Ensures asset exists, hashes are 64-bit, timestamps are strictly
      monotonically increasing, and failed extractions cannot masquerade as valid sequences.
    - Validation on Retrieval: Validates sample counts, 64-bit hash integrity, and strict
      timestamp monotonicity; malformed or corrupted records raise `CorruptFingerprintDataError`.
    - Explicit Replacement: Duplicate saves raise `DuplicateFingerprintError` unless
      `allow_replace=True` is explicitly specified.
    - Separation of Concerns: Storage/retrieval only; does not perform media classification.
"""

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from forensics.temporal_dhash import (
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    DHASH_BITS,
    TemporalDHashResult,
    TemporalDHashSample,
)
from provenance.database import get_db_connection

PERSISTENCE_SCHEMA_VERSION = 1


# ==============================================================================
# Exceptions
# ==============================================================================

class TemporalPersistenceError(Exception):
    """Base exception for temporal fingerprint persistence errors."""
    pass


class AssetNotFoundError(TemporalPersistenceError):
    """Raised when the target asset_id does not exist in the assets table."""
    pass


class DuplicateFingerprintError(TemporalPersistenceError):
    """Raised when a fingerprint already exists for an asset and replacement is not permitted."""
    pass


class CorruptFingerprintDataError(TemporalPersistenceError):
    """Raised when stored fingerprint data is malformed, truncated, or inconsistent."""
    pass


# ==============================================================================
# Schema Migration Helpers
# ==============================================================================

TEMPORAL_FINGERPRINTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS temporal_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    version TEXT NOT NULL,
    sampling_interval_seconds REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    skipped_samples_count INTEGER NOT NULL DEFAULT 0,
    video_duration_seconds REAL,
    video_fps REAL,
    is_available INTEGER NOT NULL DEFAULT 1,
    extraction_status TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
"""

TEMPORAL_FINGERPRINT_SAMPLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS temporal_fingerprint_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_id INTEGER NOT NULL,
    asset_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    frame_index INTEGER,
    dhash_hex TEXT NOT NULL,
    dhash_int TEXT NOT NULL,
    FOREIGN KEY(fingerprint_id) REFERENCES temporal_fingerprints(id) ON DELETE CASCADE,
    FOREIGN KEY(asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
"""

TEMPORAL_INDEXES_SQL = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_fingerprints_asset_id ON temporal_fingerprints(asset_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_samples_fingerprint ON temporal_fingerprint_samples(fingerprint_id, sample_index ASC);",
    "CREATE INDEX IF NOT EXISTS idx_temporal_samples_asset ON temporal_fingerprint_samples(asset_id, sample_index ASC);",
]


def create_temporal_fingerprint_tables(conn: sqlite3.Connection) -> None:
    """Creates the temporal fingerprint tables and unique indexes if they do not exist."""
    with conn:
        conn.execute(TEMPORAL_FINGERPRINTS_TABLE_SQL)
        conn.execute(TEMPORAL_FINGERPRINT_SAMPLES_TABLE_SQL)

        # Conflict detection before enforcing uniqueness on asset_id
        cursor = conn.execute("""
            SELECT asset_id, COUNT(*) as cnt
            FROM temporal_fingerprints
            GROUP BY asset_id
            HAVING cnt > 1;
        """)
        dup_assets = cursor.fetchall()
        if dup_assets:
            dup_list = [f"{r['asset_id']} ({r['cnt']} occurrences)" for r in dup_assets]
            raise sqlite3.IntegrityError(
                f"Cannot create UNIQUE index on temporal_fingerprints(asset_id): "
                f"duplicate records detected for asset_id(s): {', '.join(dup_list)}. "
                f"Manual conflict resolution required; duplicate data was not silently deleted or merged."
            )

        # Conflict detection before enforcing uniqueness on (fingerprint_id, sample_index)
        cursor = conn.execute("""
            SELECT fingerprint_id, sample_index, COUNT(*) as cnt
            FROM temporal_fingerprint_samples
            GROUP BY fingerprint_id, sample_index
            HAVING cnt > 1;
        """)
        dup_samples = cursor.fetchall()
        if dup_samples:
            dup_s_list = [f"fingerprint {r['fingerprint_id']} at index {r['sample_index']}" for r in dup_samples]
            raise sqlite3.IntegrityError(
                f"Cannot create UNIQUE index on temporal_fingerprint_samples(fingerprint_id, sample_index): "
                f"duplicate sample positions detected for: {', '.join(dup_s_list)}. "
                f"Manual conflict resolution required; duplicate data was not silently deleted or merged."
            )

        for idx_sql in TEMPORAL_INDEXES_SQL:
            conn.execute(idx_sql)


# ==============================================================================
# Core Persistence Interface
# ==============================================================================

def save_temporal_fingerprint(
    asset_id: str,
    result: TemporalDHashResult,
    allow_replace: bool = False,
    allow_failed: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Persists a temporal fingerprint extraction result for an existing registered asset.

    Args:
        asset_id: Unique asset identifier (must exist in `assets` table).
        result: `TemporalDHashResult` object containing samples and metadata.
        allow_replace: If False, raises `DuplicateFingerprintError` if a fingerprint
                       already exists for `asset_id`. If True, atomically replaces it.
        allow_failed: If False, raises `ValueError` if `result.available` is False.
                      If True, stores the extraction failure distinctly with `is_available=0`.
        conn: Optional SQLite connection. If None, uses `get_db_connection()`.

    Returns:
        The database ID of the inserted `temporal_fingerprints` header record.

    Raises:
        AssetNotFoundError: If `asset_id` does not exist in the database.
        DuplicateFingerprintError: If fingerprint already exists and `allow_replace=False`.
        ValueError: If `result` is malformed or invalid for the requested operation.
        TemporalPersistenceError: If database execution fails.
    """
    if not asset_id or not isinstance(asset_id, str):
        raise ValueError(f"Invalid asset_id: {asset_id}")

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # 1. Confirm asset exists in assets table
        cursor = conn.execute("SELECT id FROM assets WHERE asset_id = ?", (asset_id,))
        if cursor.fetchone() is None:
            raise AssetNotFoundError(f"Asset '{asset_id}' does not exist in database.")

        # 2. Validate result status and samples
        if not result.available:
            if not allow_failed:
                raise ValueError(
                    f"Cannot persist unavailable extraction result ({result.status}) "
                    f"as a valid fingerprint sequence without allow_failed=True."
                )
            is_available = 0
            sample_count = 0
            samples_to_write: List[TemporalDHashSample] = []
        else:
            if not result.samples or len(result.samples) == 0:
                raise ValueError(
                    f"Cannot persist successful extraction result with 0 usable samples for asset '{asset_id}'."
                )
            is_available = 1
            sample_count = len(result.samples)
            samples_to_write = result.samples

            # Validate each sample's integrity, sequential ordering, and strict timestamp monotonicity
            last_ts: Optional[float] = None
            for idx, s in enumerate(samples_to_write):
                if s.sample_index != idx:
                    raise ValueError(f"Sample at position {idx} has invalid sample_index {s.sample_index} (must equal {idx})")
                if s.timestamp_seconds is None or s.timestamp_seconds < 0.0:
                    raise ValueError(f"Sample at index {idx} has invalid timestamp: {s.timestamp_seconds}")
                if last_ts is not None and s.timestamp_seconds <= last_ts:
                    raise ValueError(
                        f"Sample at index {idx} violates strict timestamp monotonicity: "
                        f"t={s.timestamp_seconds} <= previous t={last_ts}"
                    )
                last_ts = s.timestamp_seconds

                # Validate hash representation
                if not isinstance(s.dhash_int, int) or not (0 <= s.dhash_int < (1 << DHASH_BITS)):
                    raise ValueError(f"Sample at index {idx} has malformed 64-bit dhash_int: {s.dhash_int}")
                if not isinstance(s.dhash_hex, str) or len(s.dhash_hex.strip()) != 16:
                    raise ValueError(f"Sample at index {idx} has malformed dhash_hex: {s.dhash_hex}")

        # 3. Check for existing record
        cur_existing = conn.execute(
            "SELECT id FROM temporal_fingerprints WHERE asset_id = ?", (asset_id,)
        )
        existing_rows = cur_existing.fetchall()
        if len(existing_rows) > 1:
            raise CorruptFingerprintDataError(
                f"Integrity violation: multiple fingerprint records found for asset '{asset_id}' "
                f"({len(existing_rows)} records)."
            )
        existing_row = existing_rows[0] if existing_rows else None

        if existing_row is not None:
            if not allow_replace:
                raise DuplicateFingerprintError(
                    f"Temporal fingerprint sequence already exists for asset '{asset_id}'. "
                    f"Silent replacement is rejected. Specify allow_replace=True to replace."
                )

        # 4. Atomic transaction write
        with conn:
            # Delete existing sequence if replacing
            if existing_row is not None and allow_replace:
                conn.execute("DELETE FROM temporal_fingerprint_samples WHERE asset_id = ?", (asset_id,))
                conn.execute("DELETE FROM temporal_fingerprints WHERE asset_id = ?", (asset_id,))

            created_at = datetime.now(timezone.utc).isoformat()
            details_json = json.dumps(result.details or {})

            cur_ins = conn.execute(
                """
                INSERT INTO temporal_fingerprints (
                    asset_id, algorithm, version, sampling_interval_seconds,
                    sample_count, skipped_samples_count, video_duration_seconds,
                    video_fps, is_available, extraction_status, details_json,
                    created_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    result.algorithm,
                    ALGORITHM_VERSION,
                    result.sampling_interval_seconds,
                    sample_count,
                    result.skipped_samples_count,
                    result.video_duration_seconds,
                    result.video_fps,
                    is_available,
                    result.status,
                    details_json,
                    created_at,
                    PERSISTENCE_SCHEMA_VERSION,
                ),
            )
            fingerprint_id = cur_ins.lastrowid

            # Insert sample rows if available
            if is_available and samples_to_write:
                sample_rows = [
                    (
                        fingerprint_id,
                        asset_id,  # verified matches parent asset_id
                        s.sample_index,
                        s.timestamp_seconds,
                        s.frame_index,
                        s.dhash_hex,
                        str(s.dhash_int),  # store as text for lossless 64-bit unsigned representation
                    )
                    for s in samples_to_write
                ]
                conn.executemany(
                    """
                    INSERT INTO temporal_fingerprint_samples (
                        fingerprint_id, asset_id, sample_index, timestamp_seconds,
                        frame_index, dhash_hex, dhash_int
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    sample_rows,
                )

        return fingerprint_id
    finally:
        if close_conn:
            conn.close()


def get_temporal_fingerprint(
    asset_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[TemporalDHashResult]:
    """
    Retrieves the stored temporal fingerprint sequence for an asset.

    Args:
        asset_id: Unique asset identifier.
        conn: Optional SQLite connection. If None, uses `get_db_connection()`.

    Returns:
        `TemporalDHashResult` if record exists, or `None` if no record exists for `asset_id`.

    Raises:
        CorruptFingerprintDataError: If stored record is malformed, truncated, or invalid.
    """
    if not asset_id or not isinstance(asset_id, str):
        raise ValueError(f"Invalid asset_id: {asset_id}")

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cur = conn.execute(
            """
            SELECT * FROM temporal_fingerprints
            WHERE asset_id = ?
            """,
            (asset_id,),
        )
        rows = cur.fetchall()
        if len(rows) == 0:
            return None
        if len(rows) > 1:
            raise CorruptFingerprintDataError(
                f"Integrity violation: multiple fingerprint records found for asset '{asset_id}' "
                f"({len(rows)} records). Database must be repaired or migrated."
            )

        row = rows[0]

        # Parse header
        fingerprint_id = row["id"]
        is_available = bool(row["is_available"])
        extraction_status = row["extraction_status"]
        algorithm = row["algorithm"]
        sampling_interval = float(row["sampling_interval_seconds"])
        sample_count = int(row["sample_count"])
        skipped_count = int(row["skipped_samples_count"])
        duration = float(row["video_duration_seconds"]) if row["video_duration_seconds"] is not None else None
        fps = float(row["video_fps"]) if row["video_fps"] is not None else None
        details = json.loads(row["details_json"]) if row["details_json"] else {}

        # If extraction was stored as unavailable / failed
        if not is_available:
            return TemporalDHashResult(
                available=False,
                status=extraction_status,
                algorithm=algorithm,
                sampling_interval_seconds=sampling_interval,
                samples=[],
                sample_count=0,
                skipped_samples_count=skipped_count,
                video_duration_seconds=duration,
                video_fps=fps,
                details=details,
            )

        # Retrieve and validate all samples
        cur_samples = conn.execute(
            """
            SELECT sample_index, timestamp_seconds, frame_index, dhash_hex, dhash_int, asset_id, fingerprint_id
            FROM temporal_fingerprint_samples
            WHERE fingerprint_id = ?
            ORDER BY sample_index ASC
            """,
            (fingerprint_id,),
        )
        sample_rows = cur_samples.fetchall()

        if len(sample_rows) != sample_count:
            raise CorruptFingerprintDataError(
                f"Corrupt fingerprint data for asset '{asset_id}': expected {sample_count} samples "
                f"in header, but retrieved {len(sample_rows)} sample rows."
            )

        samples: List[TemporalDHashSample] = []
        last_ts: Optional[float] = None
        for i, s_row in enumerate(sample_rows):
            sample_index = s_row["sample_index"]
            timestamp = float(s_row["timestamp_seconds"])
            frame_idx = s_row["frame_index"]
            dhash_hex = s_row["dhash_hex"]
            raw_dhash_int = s_row["dhash_int"]
            sample_asset_id = s_row["asset_id"]
            sample_fp_id = s_row["fingerprint_id"]

            # Validate parent / child linkage integrity
            if sample_asset_id != asset_id:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': sample at position {i} "
                    f"has mismatched asset_id '{sample_asset_id}'."
                )
            if sample_fp_id != fingerprint_id:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': sample at position {i} "
                    f"has mismatched fingerprint_id {sample_fp_id} (expected {fingerprint_id})."
                )

            # Validate sample_index ordering and uniqueness
            if sample_index != i:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': sample index mismatch "
                    f"at row {i} (sample_index={sample_index}, expected {i})."
                )

            # Validate timestamp monotonicity
            if last_ts is not None and timestamp <= last_ts:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': non-increasing timestamp "
                    f"at row {i} (t={timestamp} <= prev={last_ts})."
                )
            last_ts = timestamp

            # Validate 64-bit dhash_int
            try:
                dhash_int = int(raw_dhash_int)
                if not (0 <= dhash_int < (1 << DHASH_BITS)):
                    raise ValueError(f"Integer out of 64-bit range: {dhash_int}")
            except (ValueError, TypeError) as err:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': malformed dhash_int '{raw_dhash_int}': {err}"
                )

            # Validate 16-char hex
            if not isinstance(dhash_hex, str) or len(dhash_hex.strip()) != 16:
                raise CorruptFingerprintDataError(
                    f"Corrupt fingerprint data for asset '{asset_id}': malformed dhash_hex '{dhash_hex}'."
                )

            samples.append(
                TemporalDHashSample(
                    sample_index=sample_index,
                    timestamp_seconds=timestamp,
                    dhash_hex=dhash_hex,
                    dhash_int=dhash_int,
                    frame_index=frame_idx,
                )
            )

        return TemporalDHashResult(
            available=True,
            status=extraction_status,
            algorithm=algorithm,
            sampling_interval_seconds=sampling_interval,
            samples=samples,
            sample_count=len(samples),
            skipped_samples_count=skipped_count,
            video_duration_seconds=duration,
            video_fps=fps,
            details=details,
        )
    finally:
        if close_conn:
            conn.close()


def delete_temporal_fingerprint(
    asset_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Deletes the stored temporal fingerprint sequence and samples for an asset.

    Returns:
        True if records were deleted, False if no record existed.
    """
    if not asset_id:
        return False

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn:
            conn.execute("DELETE FROM temporal_fingerprint_samples WHERE asset_id = ?", (asset_id,))
            cursor = conn.execute("DELETE FROM temporal_fingerprints WHERE asset_id = ?", (asset_id,))
            return cursor.rowcount > 0
    finally:
        if close_conn:
            conn.close()


def has_temporal_fingerprint(
    asset_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Checks whether a temporal fingerprint record exists for an asset."""
    if not asset_id:
        return False

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cursor = conn.execute(
            "SELECT 1 FROM temporal_fingerprints WHERE asset_id = ? LIMIT 1",
            (asset_id,),
        )
        return cursor.fetchone() is not None
    finally:
        if close_conn:
            conn.close()
