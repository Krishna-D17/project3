import os
import secrets
from pathlib import Path
from typing import Optional, Union

from media.service import MediaService
from provenance.hashing import calculate_sha256
from provenance.models import (
    AssetRecord,
    HashVerificationResponse,
    generate_asset_id,
)
from provenance.repository import (
    get_asset_by_id,
    get_asset_by_sha256,
    get_asset_by_watermark_token,
    insert_asset,
    update_asset_watermark_token,
)


class ProvenanceService:
    """High-level media provenance and registration service."""

    @staticmethod
    def generate_watermark_token() -> str:
        """Generates a secure random 16-bit (2-byte) hex watermark token, e.g. 'A3F8'."""
        return secrets.token_hex(2).upper()

    @staticmethod
    def register_media(
        file_path: Union[str, Path],
        parent_asset_id: Optional[str] = None,
        relationship_type: str = "ORIGINAL",
        watermark_token: Optional[str] = None,
    ) -> AssetRecord:
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Media file not found for registration: {path}")

        # 1. Calculate streaming SHA-256
        sha256 = calculate_sha256(path)

        # 2. Check for duplicate registration by SHA-256
        existing_record = get_asset_by_sha256(sha256)
        if existing_record:
            existing_record.already_registered = True
            if watermark_token and len(watermark_token) == 4:
                update_asset_watermark_token(existing_record.asset_id, watermark_token)
                existing_record.watermark_token = watermark_token
            elif not existing_record.watermark_token or len(existing_record.watermark_token) != 4:
                tok = ProvenanceService.generate_watermark_token()
                update_asset_watermark_token(existing_record.asset_id, tok)
                existing_record.watermark_token = tok
            return existing_record

        # 3. Probe file for media stream properties
        probe_res = MediaService.probe_file(path)

        # 4. Generate stable UUID-based Asset ID and 16-bit watermark token
        asset_id = generate_asset_id()
        if not watermark_token or len(watermark_token) != 4:
            watermark_token = ProvenanceService.generate_watermark_token()
            while get_asset_by_watermark_token(watermark_token) is not None:
                watermark_token = ProvenanceService.generate_watermark_token()

        # Determine media type
        if probe_res.has_video:
            media_type = "video"
        elif probe_res.has_audio:
            media_type = "audio"
        else:
            media_type = "unknown"

        # Stream parameters
        video_codec = None
        width = None
        height = None
        fps = None
        if probe_res.video_streams:
            v_stream = probe_res.video_streams[0]
            video_codec = v_stream.codec
            width = v_stream.width
            height = v_stream.height
            fps = v_stream.fps

        audio_codec = None
        sample_rate = None
        channels = None
        if probe_res.audio_streams:
            a_stream = probe_res.audio_streams[0]
            audio_codec = a_stream.codec
            sample_rate = a_stream.sample_rate
            channels = a_stream.channels

        # 5. Create AssetRecord
        record = AssetRecord(
            asset_id=asset_id,
            sha256=sha256,
            original_filename=path.name,
            original_path=str(path.resolve()),
            file_size_bytes=path.stat().st_size,
            media_type=media_type,
            container_format=probe_res.container_format,
            duration_seconds=probe_res.duration_seconds,
            video_codec=video_codec,
            width=width,
            height=height,
            fps=fps,
            audio_codec=audio_codec,
            sample_rate=sample_rate,
            channels=channels,
            watermark_token=watermark_token,
            parent_asset_id=parent_asset_id,
            relationship_type=relationship_type,
        )

        insert_asset(record)
        return record

    @staticmethod
    def get_asset(asset_id: str) -> Optional[AssetRecord]:
        return get_asset_by_id(asset_id)

    @staticmethod
    def find_asset_by_watermark_token(watermark_token: str) -> Optional[AssetRecord]:
        return get_asset_by_watermark_token(watermark_token)

    @staticmethod
    def find_asset_by_sha256(sha256: str) -> Optional[AssetRecord]:
        return get_asset_by_sha256(sha256)

    @staticmethod
    def ensure_watermark_token(asset_id: str) -> str:
        """Retrieves existing 16-bit watermark token for an Asset ID or generates and assigns a new unique one."""
        record = get_asset_by_id(asset_id)
        if not record:
            raise ValueError(f"Asset ID record not found: {asset_id}")
        if record.watermark_token and len(record.watermark_token) == 4:
            return record.watermark_token

        tok = ProvenanceService.generate_watermark_token()
        while get_asset_by_watermark_token(tok) is not None:
            tok = ProvenanceService.generate_watermark_token()
        update_asset_watermark_token(asset_id, tok)
        record.watermark_token = tok
        return tok

    @staticmethod
    def verify_asset_hash(
        file_path: Union[str, Path], expected_asset_id: str
    ) -> HashVerificationResponse:
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found for verification: {path}")

        actual_sha256 = calculate_sha256(path)
        asset_record = get_asset_by_id(expected_asset_id)

        if not asset_record:
            return HashVerificationResponse(
                asset_id=expected_asset_id,
                filename=path.name,
                expected_sha256="",
                actual_sha256=actual_sha256,
                hash_match=False,
                status="UNKNOWN_ASSET",
            )

        expected_sha256 = asset_record.sha256.lower()
        hash_match = actual_sha256 == expected_sha256
        status = "EXACT_MATCH" if hash_match else "HASH_MISMATCH"

        return HashVerificationResponse(
            asset_id=asset_record.asset_id,
            filename=path.name,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            hash_match=hash_match,
            status=status,
        )
