import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


def generate_asset_id() -> str:
    """Generates a stable unique Asset ID formatted as MP-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx."""
    return f"MP-{uuid.uuid4()}"


def current_utc_iso() -> str:
    """Returns current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AssetRecord:
    asset_id: str
    original_filename: str
    original_path: str
    file_size_bytes: int
    sha256: str
    media_type: str
    container_format: str
    duration_seconds: float
    id: Optional[int] = None
    watermark_token: Optional[str] = None
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    parent_asset_id: Optional[str] = None
    relationship_type: str = "ORIGINAL"
    created_at: str = field(default_factory=current_utc_iso)
    registered_at: str = field(default_factory=current_utc_iso)
    already_registered: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "watermark_token": self.watermark_token,
            "original_filename": self.original_filename,
            "original_path": self.original_path,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "container_format": self.container_format,
            "duration_seconds": self.duration_seconds,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2) if self.fps is not None else None,
            "audio_codec": self.audio_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "parent_asset_id": self.parent_asset_id,
            "relationship_type": self.relationship_type,
            "created_at": self.created_at,
            "registered_at": self.registered_at,
            "already_registered": self.already_registered,
        }


@dataclass
class HashVerificationResponse:
    asset_id: Optional[str]
    filename: str
    expected_sha256: str
    actual_sha256: str
    hash_match: bool
    status: str  # EXACT_MATCH, HASH_MISMATCH, or UNKNOWN_ASSET

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "hash_match": self.hash_match,
            "status": self.status,
        }
