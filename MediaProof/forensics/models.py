from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from forensics.config import (
    CATEGORY_GENERIC_TECHNICAL,
    CONSISTENCY_NOT_AVAILABLE,
    STATUS_LOW_GENERIC,
    STATUS_MANIPULATION_NO_DETECTOR,
)


@dataclass
class ForensicSignal:
    """Individual forensic evidence signal."""
    signal_name: str
    risk_level: str  # LOW, MEDIUM, HIGH, UNKNOWN
    score: float     # Raw score or heuristic value
    details: str
    evidence_strength: str = "low"  # low, medium, high
    type: str = "heuristic"         # heuristic vs calibrated_ml
    category: str = CATEGORY_GENERIC_TECHNICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "risk_level": self.risk_level,
            "score": round(self.score, 4),
            "details": self.details,
            "evidence_strength": self.evidence_strength,
            "type": self.type,
            "category": self.category,
        }


@dataclass
class ParentFeatureDiff:
    """Quantitative comparison of a specific forensic feature between parent and candidate media."""
    feature_name: str
    parent_value: float
    candidate_value: float
    absolute_difference: float
    relative_difference: float
    within_tolerance: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "parent_value": round(self.parent_value, 4),
            "candidate_value": round(self.candidate_value, 4),
            "absolute_difference": round(self.absolute_difference, 4),
            "relative_difference": round(self.relative_difference, 4),
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class TransformationConsistencyResult:
    """Transformation consistency evaluation comparing candidate feature vectors against registered parent baseline."""
    status: str  # CONSISTENT, DEVIATION, NOT_AVAILABLE
    explanation: str
    feature_diffs: List[ParentFeatureDiff] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "explanation": self.explanation,
            "feature_diffs": [fd.to_dict() for fd in self.feature_diffs],
        }


@dataclass
class MLInferenceResult:
    """Result from optional deep learning model inference."""
    model_name: str
    available: bool
    status: str
    confidence: Optional[float] = None
    prediction_label: Optional[str] = None
    processing_time_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "available": self.available,
            "status": self.status,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "prediction_label": self.prediction_label,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "details": self.details,
        }


@dataclass
class UnifiedForensicResult:
    """Unified context-aware forensic analysis result object."""
    forensic_status: str        # LOW_RISK, ELEVATED_RISK, HIGHER_RISK, INSUFFICIENT_EVIDENCE (legacy alias)
    risk_score: float           # Aggregated heuristic risk score (0.0 to 1.0)
    risk_score_type: str = "heuristic"
    generic_forensic_status: str = STATUS_LOW_GENERIC
    manipulation_forensic_status: str = STATUS_MANIPULATION_NO_DETECTOR
    transformation_consistency: TransformationConsistencyResult = field(
        default_factory=lambda: TransformationConsistencyResult(
            status=CONSISTENCY_NOT_AVAILABLE,
            explanation="No registered parent asset available for baseline comparison.",
        )
    )
    video_signals: List[ForensicSignal] = field(default_factory=list)
    audio_signals: List[ForensicSignal] = field(default_factory=list)
    metadata_signals: List[ForensicSignal] = field(default_factory=list)
    ml_signals: List[MLInferenceResult] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forensic_status": self.forensic_status,
            "risk_score": round(self.risk_score, 4),
            "risk_score_type": self.risk_score_type,
            "generic_forensic_status": self.generic_forensic_status,
            "manipulation_forensic_status": self.manipulation_forensic_status,
            "transformation_consistency": self.transformation_consistency.to_dict(),
            "video_signals": [s.to_dict() for s in self.video_signals],
            "audio_signals": [s.to_dict() for s in self.audio_signals],
            "metadata_signals": [s.to_dict() for s in self.metadata_signals],
            "ml_signals": [m.to_dict() for m in self.ml_signals],
            "limitations": self.limitations,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
        }

