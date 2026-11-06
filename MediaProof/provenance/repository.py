import sqlite3
from typing import List, Optional

from provenance.database import get_db_connection
from provenance.models import AssetRecord


def row_to_asset_record(row: sqlite3.Row) -> AssetRecord:
    """Converts a SQLite Row object into an AssetRecord."""
    keys = row.keys()
    watermark_tok = row["watermark_token"] if "watermark_token" in keys else None
    return AssetRecord(
        id=row["id"],
        asset_id=row["asset_id"],
        watermark_token=watermark_tok,
        original_filename=row["original_filename"],
        original_path=row["original_path"],
        file_size_bytes=row["file_size_bytes"],
        sha256=row["sha256"],
        media_type=row["media_type"],
        container_format=row["container_format"],
        duration_seconds=row["duration_seconds"],
        video_codec=row["video_codec"],
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        audio_codec=row["audio_codec"],
        sample_rate=row["sample_rate"],
        channels=row["channels"],
        parent_asset_id=row["parent_asset_id"],
        relationship_type=row["relationship_type"],
        created_at=row["created_at"],
        registered_at=row["registered_at"],
    )


def insert_asset(record: AssetRecord) -> AssetRecord:
    """Inserts a new asset record into SQLite and updates the record ID."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO assets (
                    asset_id, watermark_token, original_filename, original_path, file_size_bytes,
                    sha256, media_type, container_format, duration_seconds,
                    video_codec, width, height, fps,
                    audio_codec, sample_rate, channels,
                    parent_asset_id, relationship_type, created_at, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.asset_id,
                    record.watermark_token,
                    record.original_filename,
                    record.original_path,
                    record.file_size_bytes,
                    record.sha256.lower(),
                    record.media_type,
                    record.container_format,
                    record.duration_seconds,
                    record.video_codec,
                    record.width,
                    record.height,
                    record.fps,
                    record.audio_codec,
                    record.sample_rate,
                    record.channels,
                    record.parent_asset_id,
                    record.relationship_type,
                    record.created_at,
                    record.registered_at,
                ),
            )
            record.id = cursor.lastrowid
        return record
    finally:
        conn.close()


def update_asset_watermark_token(asset_id: str, watermark_token: str) -> bool:
    """Updates the watermark_token column for an existing asset record."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE assets SET watermark_token = ? WHERE asset_id = ?",
                (watermark_token.upper(), asset_id)
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_asset_by_id(asset_id: str) -> Optional[AssetRecord]:
    """Retrieves an asset record by Asset ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
        row = cursor.fetchone()
        return row_to_asset_record(row) if row else None
    finally:
        conn.close()


def get_asset_by_watermark_token(
    watermark_token: str, allow_hamming: bool = False, media_type: Optional[str] = None
) -> Optional[AssetRecord]:
    """Retrieves an asset record by its 16-bit hex watermark token, with optional Hamming tolerance and media_type prioritization."""
    conn = get_db_connection()
    try:
        tok_upper = watermark_token.upper()

        # 1. Exact match with media_type filter if provided
        if media_type:
            cursor = conn.execute(
                "SELECT * FROM assets WHERE watermark_token = ? AND media_type = ?",
                (tok_upper, media_type),
            )
            row = cursor.fetchone()
            if row:
                return row_to_asset_record(row)

        # 2. Exact match across any media_type (supports cross-modal exact provenance)
        cursor = conn.execute("SELECT * FROM assets WHERE watermark_token = ?", (tok_upper,))
        row = cursor.fetchone()
        if row:
            return row_to_asset_record(row)

        if not allow_hamming:
            return None

        # 3. Fallback: Hamming distance matching for neural extraction noise
        def hamming_dist(t1: str, t2: str) -> int:
            try:
                v1, v2 = int(t1, 16), int(t2, 16)
                return bin(v1 ^ v2).count('1')
            except Exception:
                return 999

        # Try same media_type first if provided
        if media_type:
            cursor = conn.execute(
                "SELECT * FROM assets WHERE watermark_token IS NOT NULL AND length(watermark_token) = ? AND media_type = ?",
                (len(tok_upper), media_type),
            )
            rows = cursor.fetchall()
            best_row = None
            min_dist = 999
            for r in rows:
                candidate_tok = r["watermark_token"]
                if candidate_tok:
                    d = hamming_dist(tok_upper, candidate_tok)
                    if d < min_dist:
                        min_dist = d
                        best_row = r
            if best_row and min_dist <= 2:
                return row_to_asset_record(best_row)

        # Try across all media types
        cursor = conn.execute(
            "SELECT * FROM assets WHERE watermark_token IS NOT NULL AND length(watermark_token) = ?",
            (len(tok_upper),),
        )
        rows = cursor.fetchall()
        best_row = None
        min_dist = 999
        for r in rows:
            candidate_tok = r["watermark_token"]
            if candidate_tok:
                d = hamming_dist(tok_upper, candidate_tok)
                if d < min_dist:
                    min_dist = d
                    best_row = r

        if best_row and min_dist <= 2:
            return row_to_asset_record(best_row)

        return None
    finally:
        conn.close()


def get_asset_by_sha256(sha256: str) -> Optional[AssetRecord]:
    """Retrieves an asset record by SHA-256 hash."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM assets WHERE sha256 = ? ORDER BY id ASC LIMIT 1", (sha256.lower(),))
        row = cursor.fetchone()
        return row_to_asset_record(row) if row else None
    finally:
        conn.close()


def list_assets(limit: int = 100) -> List[AssetRecord]:
    """Lists asset records ordered by ID descending."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM assets ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [row_to_asset_record(row) for row in rows]
    finally:
        conn.close()
