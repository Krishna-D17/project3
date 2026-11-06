import sqlite3
from pathlib import Path

from backend.paths import get_db_dir, get_db_path


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured with Row factory."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes SQLite database schema, migrations, and indexes."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assets (
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

            # Migration for existing database files: ensure watermark_token column exists
            cursor = conn.execute("PRAGMA table_info(assets);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "watermark_token" not in columns:
                conn.execute("ALTER TABLE assets ADD COLUMN watermark_token TEXT;")

            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_asset_id ON assets(asset_id);
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_watermark_token ON assets(watermark_token);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets(sha256);
            """)

            # 2. Temporal perceptual fingerprints schema (Phase 1.3)
            conn.execute("""
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
            """)

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

            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_fingerprints_asset_id
                    ON temporal_fingerprints(asset_id);
            """)

            conn.execute("""
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
            """)

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

            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_samples_fingerprint
                    ON temporal_fingerprint_samples(fingerprint_id, sample_index ASC);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_temporal_samples_asset
                    ON temporal_fingerprint_samples(asset_id, sample_index ASC);
            """)
    finally:
        conn.close()


# Auto-initialize database schema when module is imported
init_db()
