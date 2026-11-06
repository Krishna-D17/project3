from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class AudioEmbeddingResult:
    """Structured result returned by audio watermark embedding."""

    status: str = "ok"
    engine: str = ""
    output_path: str = ""
    asset_id: str = ""
    size_bytes: int = 0
    has_watermark: bool = False
    processing_time_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "engine": self.engine,
            "output_path": self.output_path,
            "asset_id": self.asset_id,
            "size_bytes": self.size_bytes,
            "has_watermark": self.has_watermark,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "details": self.details,
        }


@dataclass
class AudioDetectionResult:
    """Structured result returned by audio watermark detection."""

    status: str = "ok"
    engine: str = ""
    watermark_detected: bool = False
    confidence: float = 0.0
    recovered_asset_id: Optional[str] = None
    provenance_record: Optional[Dict[str, Any]] = None
    processing_time_seconds: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "engine": self.engine,
            "watermark_detected": self.watermark_detected,
            "confidence": round(self.confidence, 4),
            "recovered_asset_id": self.recovered_asset_id,
            "provenance_record": self.provenance_record,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "message": self.message,
            "details": self.details,
        }


class BaseAudioWatermarkEngine(ABC):
    """Abstract base class for all MediaProof audio watermarking engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Returns internal engine identifier (e.g. 'custom', 'audioseal')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Returns human-readable display name for engine."""
        pass

    @abstractmethod
    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> AudioEmbeddingResult:
        """Embeds a watermark payload into an audio file mapped to asset_id."""
        pass

    @abstractmethod
    def detect(
        self,
        input_path: Union[str, Path],
        timeout: float = 30.0,
    ) -> AudioDetectionResult:
        """Detects watermark in an audio file and resolves asset record if found."""
        pass
