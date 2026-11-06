from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class EmbeddingResult:
    engine: str
    output_path: str
    asset_id: str
    size_bytes: int
    has_watermark: bool
    processing_time_seconds: float
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "engine": self.engine,
            "output_path": self.output_path,
            "asset_id": self.asset_id,
            "size_bytes": self.size_bytes,
            "has_watermark": self.has_watermark,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "details": self.details,
        }


@dataclass
class DetectionResult:
    engine: str
    watermark_detected: bool
    confidence: float
    recovered_asset_id: Optional[str]
    provenance_record: Optional[Dict[str, object]]
    processing_time_seconds: float
    message: str
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "engine": self.engine,
            "watermark_detected": self.watermark_detected,
            "confidence": round(self.confidence, 4),
            "recovered_asset_id": self.recovered_asset_id,
            "provenance_record": self.provenance_record,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "message": self.message,
            "details": self.details,
        }


class BaseVideoWatermarkEngine(ABC):
    """Abstract Base Class for MediaProof Video Watermark Engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Returns short engine code: 'custom' or 'videoseal'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Returns human-readable name of the engine."""
        pass

    @abstractmethod
    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> EmbeddingResult:
        """Embeds an invisible Asset ID watermark into a video file."""
        pass

    @abstractmethod
    def detect(
        self,
        input_path: Union[str, Path],
        timeout: float = 30.0,
    ) -> DetectionResult:
        """Detects and extracts an invisible watermark payload from a video file."""
        pass
