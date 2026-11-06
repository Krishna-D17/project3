"""
Tests for Temporal dHash Persistence and Retrieval (Phase 1.3).

Covers all required scenarios:
1. Save and retrieve a valid sequence without changing its hashes or timestamps.
2. Preserve the algorithm version and sampling configuration.
3. Handle missing asset identifiers safely.
4. Handle duplicate saves according to the chosen replacement policy.
5. Handle absent fingerprint records.
6. Reject malformed hashes and invalid sample ordering (on save and on retrieval).
7. Handle unavailable or failed extraction results safely.
8. Roll back failed writes without leaving partial sequences.
9. Migrate a representative existing database without losing asset records.
10. Compatibility with existing schema and database operations.
"""

import json
from pathlib import Path
import sqlite3
import pytest

from forensics.temporal_dhash import (
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    TemporalDHashResult,
    TemporalDHashSample,
)
from forensics.temporal_persistence import (
    AssetNotFoundError,
    CorruptFingerprintDataError,
    DuplicateFingerprintError,
    delete_temporal_fingerprint,
    get_temporal_fingerprint,
    has_temporal_fingerprint,
    save_temporal_fingerprint,
)
from provenance.database import init_db
from provenance.models import AssetRecord, generate_asset_id
from provenance.repository import insert_asset


# ==============================================================================
# Pytest Fixture for Isolated Sandbox Database
# ==============================================================================

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Redirects get_db_path in both backend.paths and provenance.database
    to an isolated temporary database file for every test, guaranteeing
    the production database is never touched.
    """
    sandbox_db = tmp_path / "sandbox_test.db"

    import backend.paths
    monkeypatch.setattr(backend.paths, "get_db_path", lambda: sandbox_db)

    import provenance.database
    monkeypatch.setattr(provenance.database, "get_db_path", lambda: sandbox_db)

    # Initialize schema in the temporary database
    init_db()
    yield sandbox_db


# Helper to insert a test asset in the sandbox database
def create_test_asset(asset_id: str = None) -> AssetRecord:
    if asset_id is None:
        asset_id = generate_asset_id()
    record = AssetRecord(
        asset_id=asset_id,
        original_filename="sample_video.mp4",
        original_path="/test/sample_video.mp4",
        file_size_bytes=1048576,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        media_type="video",
        container_format="mp4",
        duration_seconds=5.0,
        video_codec="h264",
        width=1280,
        height=720,
        fps=30.0,
    )
    return insert_asset(record)


def create_sample_result(sample_count: int = 5) -> TemporalDHashResult:
    samples = []
    base_hashes = [
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
        0xFFFFFFFFFFFFFFFF,
    ]
    for i in range(sample_count):
        h_int = base_hashes[i % len(base_hashes)]
        samples.append(
            TemporalDHashSample(
                sample_index=i,
                timestamp_seconds=float(i) * 1.0,
                dhash_hex=f"{h_int:016x}",
                dhash_int=h_int,
                frame_index=i * 30,
            )
        )
    return TemporalDHashResult(
        available=True,
        status="SUCCESS",
        algorithm=f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}",
        sampling_interval_seconds=1.0,
        samples=samples,
        sample_count=len(samples),
        skipped_samples_count=0,
        processing_time_seconds=0.05,
        video_duration_seconds=float(sample_count),
        video_fps=30.0,
        details={"decoder": "cv2", "bit_order": "msb_first"},
    )


# ==============================================================================
# Test 1: Save and retrieve a valid sequence without changing hashes or timestamps
# ==============================================================================

def test_save_and_retrieve_valid_sequence():
    asset = create_test_asset()
    original_result = create_sample_result(sample_count=5)

    fp_id = save_temporal_fingerprint(asset.asset_id, original_result)
    assert isinstance(fp_id, int)
    assert fp_id > 0

    retrieved = get_temporal_fingerprint(asset.asset_id)
    assert retrieved is not None
    assert retrieved.available is True
    assert retrieved.status == "SUCCESS"
    assert retrieved.sample_count == 5
    assert len(retrieved.samples) == 5

    for orig, ret in zip(original_result.samples, retrieved.samples):
        assert ret.sample_index == orig.sample_index
        assert ret.timestamp_seconds == pytest.approx(orig.timestamp_seconds)
        assert ret.frame_index == orig.frame_index
        assert ret.dhash_hex == orig.dhash_hex
        assert ret.dhash_int == orig.dhash_int


# ==============================================================================
# Test 2: Preserve algorithm version and sampling configuration
# ==============================================================================

def test_preserve_algorithm_and_configuration():
    asset = create_test_asset()
    original_result = create_sample_result(sample_count=4)
    original_result.sampling_interval_seconds = 0.5
    original_result.skipped_samples_count = 2
    original_result.video_duration_seconds = 12.34
    original_result.video_fps = 24.0
    original_result.details = {"custom_key": "custom_val", "level": 1}

    save_temporal_fingerprint(asset.asset_id, original_result)
    retrieved = get_temporal_fingerprint(asset.asset_id)

    assert retrieved is not None
    assert retrieved.algorithm == original_result.algorithm
    assert retrieved.sampling_interval_seconds == 0.5
    assert retrieved.skipped_samples_count == 2
    assert retrieved.video_duration_seconds == pytest.approx(12.34)
    assert retrieved.video_fps == pytest.approx(24.0)
    assert retrieved.details == {"custom_key": "custom_val", "level": 1}


# ==============================================================================
# Test 3: Handle missing asset identifiers safely
# ==============================================================================

def test_missing_asset_identifier_rejected(isolated_db):
    non_existent_asset_id = "MP-00000000-0000-0000-0000-000000000000"
    result = create_sample_result()

    with pytest.raises(AssetNotFoundError) as excinfo:
        save_temporal_fingerprint(non_existent_asset_id, result)
    assert "does not exist in database" in str(excinfo.value)

    # Verify no rows were created in DB
    conn = sqlite3.connect(str(isolated_db))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM temporal_fingerprints")
    assert c.fetchone()[0] == 0
    c.execute("SELECT COUNT(*) FROM temporal_fingerprint_samples")
    assert c.fetchone()[0] == 0
    conn.close()


# ==============================================================================
# Test 4: Handle duplicate saves according to replacement policy
# ==============================================================================

def test_duplicate_saves_and_explicit_replacement(isolated_db):
    asset = create_test_asset()
    result1 = create_sample_result(sample_count=3)
    result2 = create_sample_result(sample_count=5)

    # First save succeeds
    save_temporal_fingerprint(asset.asset_id, result1)

    # Second save with allow_replace=False raises DuplicateFingerprintError
    with pytest.raises(DuplicateFingerprintError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, result2, allow_replace=False)
    assert "Silent replacement is rejected" in str(excinfo.value)

    # Verify result1 is still in database
    current = get_temporal_fingerprint(asset.asset_id)
    assert current.sample_count == 3

    # Third save with allow_replace=True succeeds and replaces
    save_temporal_fingerprint(asset.asset_id, result2, allow_replace=True)
    updated = get_temporal_fingerprint(asset.asset_id)
    assert updated.sample_count == 5

    # Check database table row count: exactly 1 header row, exactly 5 sample rows
    conn = sqlite3.connect(str(isolated_db))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM temporal_fingerprints WHERE asset_id = ?", (asset.asset_id,))
    assert c.fetchone()[0] == 1
    c.execute("SELECT COUNT(*) FROM temporal_fingerprint_samples WHERE asset_id = ?", (asset.asset_id,))
    assert c.fetchone()[0] == 5
    conn.close()


# ==============================================================================
# Test 5: Handle absent fingerprint records
# ==============================================================================

def test_absent_fingerprint_records():
    asset = create_test_asset()

    # Asset exists, but no fingerprint has been stored
    assert get_temporal_fingerprint(asset.asset_id) is None
    assert has_temporal_fingerprint(asset.asset_id) is False

    # Completely unknown asset_id
    assert get_temporal_fingerprint("MP-totally-unknown") is None
    assert has_temporal_fingerprint("MP-totally-unknown") is False


# ==============================================================================
# Test 6: Reject malformed hashes and invalid sample ordering
# ==============================================================================

def test_reject_malformed_hashes_and_timestamps(isolated_db):
    asset = create_test_asset()

    # A. Save-time validations:
    # 1. Non-increasing timestamps
    bad_ts_result = create_sample_result(sample_count=3)
    bad_ts_result.samples[1].timestamp_seconds = 0.5
    bad_ts_result.samples[2].timestamp_seconds = 0.4  # Decreasing!
    with pytest.raises(ValueError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, bad_ts_result)
    assert "timestamp monotonicity" in str(excinfo.value)

    # 2. Malformed dhash_int (> 64 bits)
    bad_hash_result = create_sample_result(sample_count=2)
    bad_hash_result.samples[0].dhash_int = 1 << 65
    with pytest.raises(ValueError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, bad_hash_result)
    assert "malformed 64-bit dhash_int" in str(excinfo.value)

    # 3. Malformed dhash_hex (not 16 chars)
    bad_hex_result = create_sample_result(sample_count=2)
    bad_hex_result.samples[0].dhash_hex = "short_hex"
    with pytest.raises(ValueError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, bad_hex_result)
    assert "malformed dhash_hex" in str(excinfo.value)

    # B. Retrieval-time validations on corrupted stored data:
    valid_res = create_sample_result(sample_count=4)
    save_temporal_fingerprint(asset.asset_id, valid_res)

    # Corrupt 1: Sample count mismatch in database (delete one row directly)
    conn = sqlite3.connect(str(isolated_db))
    conn.execute("DELETE FROM temporal_fingerprint_samples WHERE sample_index = 2")
    conn.commit()
    conn.close()

    with pytest.raises(CorruptFingerprintDataError) as excinfo:
        get_temporal_fingerprint(asset.asset_id)
    assert "expected 4 samples in header, but retrieved 3" in str(excinfo.value)

    # Corrupt 2: Non-increasing timestamp in stored database row
    save_temporal_fingerprint(asset.asset_id, valid_res, allow_replace=True)
    conn = sqlite3.connect(str(isolated_db))
    conn.execute("UPDATE temporal_fingerprint_samples SET timestamp_seconds = 0.1 WHERE sample_index = 2")
    conn.commit()
    conn.close()

    with pytest.raises(CorruptFingerprintDataError) as excinfo:
        get_temporal_fingerprint(asset.asset_id)
    assert "non-increasing timestamp" in str(excinfo.value)

    # Corrupt 3: Malformed hash string in stored database row
    save_temporal_fingerprint(asset.asset_id, valid_res, allow_replace=True)
    conn = sqlite3.connect(str(isolated_db))
    conn.execute("UPDATE temporal_fingerprint_samples SET dhash_int = 'not_a_number' WHERE sample_index = 1")
    conn.commit()
    conn.close()

    with pytest.raises(CorruptFingerprintDataError) as excinfo:
        get_temporal_fingerprint(asset.asset_id)
    assert "malformed dhash_int" in str(excinfo.value)


# ==============================================================================
# Test 7: Handle unavailable or failed extraction results safely
# ==============================================================================

def test_handle_unavailable_or_failed_extraction():
    asset = create_test_asset()

    failed_result = TemporalDHashResult(
        available=False,
        status="UNAVAILABLE_DECODER_MISSING",
        algorithm=f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}",
        sampling_interval_seconds=1.0,
        samples=[],
        sample_count=0,
        details={"error": "Decoder not installed"},
    )

    # Attempting to save failed result without allow_failed raises ValueError
    with pytest.raises(ValueError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, failed_result, allow_failed=False)
    assert "Cannot persist unavailable extraction result" in str(excinfo.value)

    # Saving with allow_failed=True succeeds
    save_temporal_fingerprint(asset.asset_id, failed_result, allow_failed=True)

    # Retrieval preserves failure state explicitly and does NOT return a valid empty sequence
    retrieved = get_temporal_fingerprint(asset.asset_id)
    assert retrieved is not None
    assert retrieved.available is False
    assert retrieved.status == "UNAVAILABLE_DECODER_MISSING"
    assert retrieved.sample_count == 0
    assert len(retrieved.samples) == 0
    assert retrieved.details.get("error") == "Decoder not installed"

    # Attempting to save available=True with 0 samples is rejected
    invalid_empty = TemporalDHashResult(
        available=True,
        status="SUCCESS",
        samples=[],
    )
    with pytest.raises(ValueError) as excinfo:
        save_temporal_fingerprint(asset.asset_id, invalid_empty, allow_replace=True)
    assert "Cannot persist successful extraction result with 0 usable samples" in str(excinfo.value)


# ==============================================================================
# Test 8: Roll back failed writes without leaving partial sequences
# ==============================================================================

def test_atomic_rollback_on_write_failure(isolated_db):
    asset = create_test_asset()

    # Create a result where the last sample is deliberately invalid
    result = create_sample_result(sample_count=4)
    result.samples[3].dhash_int = -999  # Invalid negative dhash_int

    with pytest.raises(ValueError):
        save_temporal_fingerprint(asset.asset_id, result)

    # Verify no header row and no sample rows were inserted
    conn = sqlite3.connect(str(isolated_db))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM temporal_fingerprints WHERE asset_id = ?", (asset.asset_id,))
    assert c.fetchone()[0] == 0
    c.execute("SELECT COUNT(*) FROM temporal_fingerprint_samples WHERE asset_id = ?", (asset.asset_id,))
    assert c.fetchone()[0] == 0
    conn.close()


# ==============================================================================
# Test 9: Migrate representative existing database without losing asset records
# ==============================================================================

def test_migration_from_existing_database(tmp_path, monkeypatch):
    """
    Simulates an existing pre-Phase-1.3 database with legacy assets,
    applies the migration, and confirms existing asset data is 100% preserved.
    """
    legacy_db_path = tmp_path / "legacy_pre_upgrade.db"

    # 1. Create legacy schema without temporal fingerprint tables
    conn = sqlite3.connect(str(legacy_db_path))
    conn.execute("""
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL UNIQUE,
            watermark_token TEXT,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            media_type TEXT NOT NULL,
            container_format TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            video_codec TEXT,
            width INTEGER,
            height INTEGER,
            fps REAL,
            audio_codec TEXT,
            sample_rate INTEGER,
            channels INTEGER,
            parent_asset_id TEXT,
            relationship_type TEXT DEFAULT 'ORIGINAL',
            created_at TEXT NOT NULL,
            registered_at TEXT NOT NULL
        );
    """)

    # Insert 3 legacy assets
    legacy_assets = [
        ("MP-legacy-1", "video1.mp4", "hash1", "video", 10.0),
        ("MP-legacy-2", "video2.mp4", "hash2", "video", 20.0),
        ("MP-legacy-3", "audio1.wav", "hash3", "audio", 30.0),
    ]
    for aid, fn, sh, mt, dur in legacy_assets:
        conn.execute("""
            INSERT INTO assets (
                asset_id, original_filename, original_path, file_size_bytes,
                sha256, media_type, container_format, duration_seconds,
                created_at, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '2026-01-01T00:00:00', '2026-01-01T00:00:00')
        """, (aid, fn, f"/path/{fn}", 1000, sh, mt, "mp4", dur))
    conn.commit()
    conn.close()

    # 2. Redirect get_db_path to the legacy database and run init_db() migration
    import backend.paths
    monkeypatch.setattr(backend.paths, "get_db_path", lambda: legacy_db_path)
    import provenance.database
    monkeypatch.setattr(provenance.database, "get_db_path", lambda: legacy_db_path)

    init_db()

    # 3. Verify all 3 legacy assets still exist exactly as inserted
    conn = sqlite3.connect(str(legacy_db_path))
    c = conn.cursor()
    c.execute("SELECT asset_id, original_filename, sha256 FROM assets ORDER BY id ASC")
    rows = c.fetchall()
    assert len(rows) == 3
    assert rows[0] == ("MP-legacy-1", "video1.mp4", "hash1")
    assert rows[1] == ("MP-legacy-2", "video2.mp4", "hash2")
    assert rows[2] == ("MP-legacy-3", "audio1.wav", "hash3")

    # 4. Verify temporal fingerprint tables were created
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "temporal_fingerprints" in tables
    assert "temporal_fingerprint_samples" in tables
    conn.close()

    # 5. Verify temporal fingerprint operations work on the migrated database
    result = create_sample_result(sample_count=4)
    fp_id = save_temporal_fingerprint("MP-legacy-1", result)
    assert fp_id > 0
    retrieved = get_temporal_fingerprint("MP-legacy-1")
    assert retrieved is not None
    assert retrieved.sample_count == 4


# ==============================================================================
# Test 10: Delete fingerprint operation
# ==============================================================================

def test_delete_temporal_fingerprint(isolated_db):
    asset = create_test_asset()
    result = create_sample_result(sample_count=3)
    save_temporal_fingerprint(asset.asset_id, result)

    assert has_temporal_fingerprint(asset.asset_id) is True

    # Delete existing
    deleted = delete_temporal_fingerprint(asset.asset_id)
    assert deleted is True

    assert has_temporal_fingerprint(asset.asset_id) is False
    assert get_temporal_fingerprint(asset.asset_id) is None

    # Delete non-existing returns False
    assert delete_temporal_fingerprint(asset.asset_id) is False


# ==============================================================================
# Test 11: Database-level UNIQUE constraint on asset_id (Direct SQL)
# ==============================================================================

def test_database_level_duplicate_asset_prevention_direct_sql(isolated_db):
    asset = create_test_asset()
    conn = sqlite3.connect(str(isolated_db))
    # First insert
    conn.execute(
        """
        INSERT INTO temporal_fingerprints (
            asset_id, algorithm, version, sampling_interval_seconds,
            sample_count, skipped_samples_count, is_available, extraction_status,
            created_at, schema_version
        ) VALUES (?, 'temporal_dhash', '1.0', 1.0, 1, 0, 1, 'SUCCESS', '2026-01-01', 1)
        """,
        (asset.asset_id,),
    )
    conn.commit()

    # Second direct SQL insert with identical asset_id must trigger IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO temporal_fingerprints (
                asset_id, algorithm, version, sampling_interval_seconds,
                sample_count, skipped_samples_count, is_available, extraction_status,
                created_at, schema_version
            ) VALUES (?, 'temporal_dhash', '1.0', 1.0, 1, 0, 1, 'SUCCESS', '2026-01-01', 1)
            """,
            (asset.asset_id,),
        )
    conn.close()


# ==============================================================================
# Test 12: Database-level UNIQUE constraint on (fingerprint_id, sample_index)
# ==============================================================================

def test_database_level_duplicate_sample_index_direct_sql(isolated_db):
    asset = create_test_asset()
    conn = sqlite3.connect(str(isolated_db))
    cur = conn.execute(
        """
        INSERT INTO temporal_fingerprints (
            asset_id, algorithm, version, sampling_interval_seconds,
            sample_count, skipped_samples_count, is_available, extraction_status,
            created_at, schema_version
        ) VALUES (?, 'temporal_dhash', '1.0', 1.0, 2, 0, 1, 'SUCCESS', '2026-01-01', 1)
        """,
        (asset.asset_id,),
    )
    fp_id = cur.lastrowid

    # Insert sample at index 0
    conn.execute(
        """
        INSERT INTO temporal_fingerprint_samples (
            fingerprint_id, asset_id, sample_index, timestamp_seconds, dhash_hex, dhash_int
        ) VALUES (?, ?, 0, 0.0, '0123456789abcdef', '81985529216486895')
        """,
        (fp_id, asset.asset_id),
    )
    conn.commit()

    # Attempting to insert duplicate sample_index 0 for the same fingerprint_id must raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO temporal_fingerprint_samples (
                fingerprint_id, asset_id, sample_index, timestamp_seconds, dhash_hex, dhash_int
            ) VALUES (?, ?, 0, 1.0, 'fedcba9876543210', '18364758544493064720')
            """,
            (fp_id, asset.asset_id),
        )
    conn.close()


# ==============================================================================
# Test 13: Replacement failure preserves original record and samples
# ==============================================================================

def test_replacement_failure_preserves_original_data(isolated_db):
    asset = create_test_asset()
    original_res = create_sample_result(sample_count=3)
    save_temporal_fingerprint(asset.asset_id, original_res)

    # Verify original is present
    orig_fp = get_temporal_fingerprint(asset.asset_id)
    assert orig_fp.sample_count == 3

    # Attempt replacement with invalid data (e.g. non-monotonic timestamps)
    bad_replacement = create_sample_result(sample_count=5)
    bad_replacement.samples[3].timestamp_seconds = 0.1  # Violates monotonicity!

    with pytest.raises(ValueError):
        save_temporal_fingerprint(asset.asset_id, bad_replacement, allow_replace=True)

    # Verify original record and its original samples are completely preserved
    preserved_fp = get_temporal_fingerprint(asset.asset_id)
    assert preserved_fp is not None
    assert preserved_fp.sample_count == 3
    assert len(preserved_fp.samples) == 3
    for orig_s, pres_s in zip(original_res.samples, preserved_fp.samples):
        assert pres_s.dhash_hex == orig_s.dhash_hex
        assert pres_s.timestamp_seconds == pytest.approx(orig_s.timestamp_seconds)


# ==============================================================================
# Test 14: Parent/Child asset_id mismatch rejection
# ==============================================================================

def test_parent_child_asset_id_mismatch_rejected(isolated_db):
    asset = create_test_asset()
    valid_res = create_sample_result(sample_count=3)
    save_temporal_fingerprint(asset.asset_id, valid_res)

    # Manually tamper with one sample's asset_id in the database
    conn = sqlite3.connect(str(isolated_db))
    conn.execute("UPDATE temporal_fingerprint_samples SET asset_id = 'MP-mismatched-asset' WHERE sample_index = 1")
    conn.commit()
    conn.close()

    with pytest.raises(CorruptFingerprintDataError) as excinfo:
        get_temporal_fingerprint(asset.asset_id)
    assert "mismatched asset_id" in str(excinfo.value)


# ==============================================================================
# Test 15: Multiple headers in database rejected without arbitrary selection
# ==============================================================================

def test_multiple_headers_rejected_without_arbitrary_selection(isolated_db):
    asset = create_test_asset()
    conn = sqlite3.connect(str(isolated_db))
    # Drop unique index temporarily to simulate an inconsistent or corrupted legacy DB
    conn.execute("DROP INDEX IF EXISTS idx_temporal_fingerprints_asset_id")
    # Insert two headers for the same asset
    conn.execute(
        """
        INSERT INTO temporal_fingerprints (
            asset_id, algorithm, version, sampling_interval_seconds,
            sample_count, skipped_samples_count, is_available, extraction_status,
            created_at, schema_version
        ) VALUES (?, 'temporal_dhash', '1.0', 1.0, 1, 0, 1, 'SUCCESS', '2026-01-01', 1)
        """,
        (asset.asset_id,),
    )
    conn.execute(
        """
        INSERT INTO temporal_fingerprints (
            asset_id, algorithm, version, sampling_interval_seconds,
            sample_count, skipped_samples_count, is_available, extraction_status,
            created_at, schema_version
        ) VALUES (?, 'temporal_dhash', '1.0', 1.0, 1, 0, 1, 'SUCCESS', '2026-01-02', 1)
        """,
        (asset.asset_id,),
    )
    conn.commit()
    conn.close()

    # get_temporal_fingerprint must NOT silently pick one; it must raise CorruptFingerprintDataError
    with pytest.raises(CorruptFingerprintDataError) as excinfo:
        get_temporal_fingerprint(asset.asset_id)
    assert "multiple fingerprint records found" in str(excinfo.value)


# ==============================================================================
# Test 16: Migration conflict detection on duplicate legacy temporal records
# ==============================================================================

def test_migration_conflict_detection_on_duplicate_records(tmp_path, monkeypatch):
    """
    Confirms that if a pre-existing database contains duplicate fingerprint records,
    init_db detects and reports the conflict clearly rather than silently deleting or merging.
    """
    conflict_db_path = tmp_path / "conflict_legacy.db"
    conn = sqlite3.connect(str(conflict_db_path))
    conn.execute("""
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL UNIQUE,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            media_type TEXT NOT NULL,
            container_format TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            created_at TEXT NOT NULL,
            registered_at TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE temporal_fingerprints (
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
            schema_version INTEGER NOT NULL DEFAULT 1
        );
    """)
    # Insert duplicate records for MP-dup-asset
    conn.execute("INSERT INTO assets (asset_id, original_filename, original_path, file_size_bytes, sha256, media_type, container_format, duration_seconds, created_at, registered_at) VALUES ('MP-dup-asset', 'f.mp4', '/f.mp4', 10, 'sha', 'video', 'mp4', 5.0, '2026-01-01', '2026-01-01')")
    conn.execute("INSERT INTO temporal_fingerprints (asset_id, algorithm, version, sampling_interval_seconds, sample_count, extraction_status, created_at) VALUES ('MP-dup-asset', 'temporal_dhash', '1.0', 1.0, 1, 'SUCCESS', '2026-01-01')")
    conn.execute("INSERT INTO temporal_fingerprints (asset_id, algorithm, version, sampling_interval_seconds, sample_count, extraction_status, created_at) VALUES ('MP-dup-asset', 'temporal_dhash', '1.0', 1.0, 1, 'SUCCESS', '2026-01-02')")
    conn.commit()
    conn.close()

    import backend.paths
    monkeypatch.setattr(backend.paths, "get_db_path", lambda: conflict_db_path)
    import provenance.database
    monkeypatch.setattr(provenance.database, "get_db_path", lambda: conflict_db_path)

    # Running init_db must detect and report the conflict via IntegrityError
    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        init_db()
    assert "duplicate records detected for asset_id(s): MP-dup-asset" in str(excinfo.value)
    assert "Manual conflict resolution required" in str(excinfo.value)

