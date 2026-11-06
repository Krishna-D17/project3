import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from media.service import MediaService
from forensics.audio_ml import run_audio_ml_inference
from forensics.audio_signals import extract_audio_signals
from forensics.config import (
    CONSISTENCY_CONSISTENT,
    CONSISTENCY_DEVIATION,
    CONSISTENCY_NOT_AVAILABLE,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_MEDIUM,
    STATUS_ELEVATED_RISK,
    STATUS_HIGH_GENERIC,
    STATUS_HIGHER_RISK,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_LOW_GENERIC,
    STATUS_LOW_RISK,
    STATUS_MANIPULATION_NO_ANOMALY,
    STATUS_MANIPULATION_NO_DETECTOR,
    STATUS_MANIPULATION_SUSPECTED,
    STATUS_MEDIUM_GENERIC,
)
from forensics.metadata_signals import extract_metadata_signals
from forensics.models import (
    ForensicSignal,
    ParentFeatureDiff,
    TransformationConsistencyResult,
    UnifiedForensicResult,
)
from forensics.video_ml import run_video_ml_inference
from forensics.video_signals import extract_video_signals


class ForensicService:
    """Core Forensic Service executing context-aware multi-modal forensic analysis."""

    @staticmethod
    def analyze_media(
        file_path: Union[str, Path],
        parent_file_path: Optional[Union[str, Path]] = None,
        timeout: float = 60.0,
    ) -> UnifiedForensicResult:
        start_time = time.time()
        p = Path(file_path).resolve()

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Media file not found for forensic analysis: {p}")

        limitations: List[str] = []

        # 1. Probe Media Streams
        try:
            probe_res = MediaService.probe_file(p)
        except Exception as e:
            return UnifiedForensicResult(
                forensic_status=STATUS_INSUFFICIENT_EVIDENCE,
                risk_score=0.0,
                risk_score_type="heuristic",
                generic_forensic_status=STATUS_LOW_GENERIC,
                manipulation_forensic_status=STATUS_MANIPULATION_NO_DETECTOR,
                limitations=[f"Media probing error: {str(e)}"],
                processing_time_seconds=time.time() - start_time,
            )

        has_video = probe_res.has_video
        has_audio = probe_res.has_audio

        if not has_video and not has_audio:
            return UnifiedForensicResult(
                forensic_status=STATUS_INSUFFICIENT_EVIDENCE,
                risk_score=0.0,
                risk_score_type="heuristic",
                generic_forensic_status=STATUS_LOW_GENERIC,
                manipulation_forensic_status=STATUS_MANIPULATION_NO_DETECTOR,
                limitations=["Media container contains no audio or video streams."],
                processing_time_seconds=time.time() - start_time,
            )

        # 2. Extract Metadata Signals
        meta_signals = extract_metadata_signals(p)

        # 3. Extract Video Signals & ML
        video_signals: List[ForensicSignal] = []
        ml_signals = []

        if has_video:
            video_signals = extract_video_signals(p, timeout=timeout)
            v_ml = run_video_ml_inference(p)
            ml_signals.append(v_ml)
            if not v_ml.available:
                limitations.append(f"Video ML inference unavailable: {v_ml.details.get('message', 'Checkpoint missing')}")

        # 4. Extract Audio Signals & ML
        audio_signals: List[ForensicSignal] = []
        if has_audio:
            audio_signals = extract_audio_signals(p, timeout=timeout)
            a_ml = run_audio_ml_inference(p)
            ml_signals.append(a_ml)
            if not a_ml.available:
                limitations.append(f"Audio ML inference unavailable: {a_ml.details.get('message', 'Checkpoint missing')}")

        # 5. Aggregate Heuristic Risk Score & Assign Generic Forensic Status
        all_signals = meta_signals + video_signals + audio_signals
        high_risk_count = sum(1 for s in all_signals if s.risk_level == RISK_LEVEL_HIGH)
        med_risk_count = sum(1 for s in all_signals if s.risk_level == RISK_LEVEL_MEDIUM)

        # Heuristic Risk Calculation
        raw_score = (high_risk_count * 0.40) + (med_risk_count * 0.15)
        risk_score = min(max(raw_score, 0.0), 1.0)

        # Legacy status mapping for backwards compatibility
        if risk_score >= 0.70 or high_risk_count >= 2:
            forensic_status = STATUS_HIGHER_RISK
            generic_forensic_status = STATUS_HIGH_GENERIC
        elif risk_score >= 0.30 or med_risk_count >= 2:
            forensic_status = STATUS_ELEVATED_RISK
            generic_forensic_status = STATUS_MEDIUM_GENERIC
        else:
            forensic_status = STATUS_LOW_RISK
            generic_forensic_status = STATUS_LOW_GENERIC

        # Assign Manipulation Forensic Status
        manipulation_forensic_status = STATUS_MANIPULATION_NO_DETECTOR
        available_ml = [
            m for m in ml_signals
            if m.available and m.status not in [
                "FRAME_EXTRACTION_FAILED",
                "NON_FINITE_MODEL_OUTPUT",
                "MALFORMED_MODEL_OUTPUT",
            ]
        ]
        if available_ml:
            if any(m.prediction_label in ["deepfake", "manipulated", "fake", "MANIPULATION_SUSPECTED"] or m.status == STATUS_MANIPULATION_SUSPECTED for m in available_ml):
                manipulation_forensic_status = STATUS_MANIPULATION_SUSPECTED
            else:
                manipulation_forensic_status = STATUS_MANIPULATION_NO_ANOMALY

        # 6. Parent-Baseline Comparison & Transformation Consistency
        transformation_consistency = TransformationConsistencyResult(
            status=CONSISTENCY_NOT_AVAILABLE,
            explanation="No registered parent asset available for baseline comparison."
        )

        if parent_file_path:
            parent_path = Path(parent_file_path).resolve()
            if parent_path.exists() and parent_path.is_file():
                try:
                    parent_v_signals = extract_video_signals(parent_path, timeout=timeout) if has_video else []
                    parent_a_signals = extract_audio_signals(parent_path, timeout=timeout) if has_audio else []
                    parent_signals = {s.signal_name: s.score for s in (parent_v_signals + parent_a_signals)}

                    feature_diffs: List[ParentFeatureDiff] = []
                    all_within_tolerance = True

                    for cand_sig in (video_signals + audio_signals):
                        sig_name = cand_sig.signal_name
                        if sig_name in parent_signals:
                            parent_val = parent_signals[sig_name]
                            cand_val = cand_sig.score
                            abs_diff = abs(cand_val - parent_val)
                            rel_diff = abs_diff / (abs(parent_val) + 1e-8)

                            # Define engineering tolerance bounds for compression/transformation
                            if "blurriness" in sig_name or "temporal" in sig_name:
                                within_tol = rel_diff <= 0.50 or abs_diff <= 50.0
                            elif "frequency" in sig_name:
                                within_tol = abs_diff <= 0.15 or rel_diff <= 0.35
                            else:
                                within_tol = rel_diff <= 0.30 or abs_diff <= 0.10

                            if not within_tol:
                                all_within_tolerance = False

                            feature_diffs.append(ParentFeatureDiff(
                                feature_name=sig_name,
                                parent_value=parent_val,
                                candidate_value=cand_val,
                                absolute_difference=abs_diff,
                                relative_difference=rel_diff,
                                within_tolerance=within_tol,
                            ))

                    if feature_diffs:
                        if all_within_tolerance:
                            consistency_status = CONSISTENCY_CONSISTENT
                            consistency_expl = (
                                "Candidate signal metrics are technically consistent with registered parent asset within expected compression bounds. "
                                "This does not guarantee content authenticity."
                            )
                        else:
                            consistency_status = CONSISTENCY_DEVIATION
                            consistency_expl = "Candidate signal metrics deviate materially from registered parent baseline."

                        transformation_consistency = TransformationConsistencyResult(
                            status=consistency_status,
                            explanation=consistency_expl,
                            feature_diffs=feature_diffs
                        )
                except Exception as ex:
                    limitations.append(f"Parent baseline comparison error: {str(ex)}")

        total_time = time.time() - start_time
        return UnifiedForensicResult(
            forensic_status=forensic_status,
            risk_score=risk_score,
            risk_score_type="heuristic",
            generic_forensic_status=generic_forensic_status,
            manipulation_forensic_status=manipulation_forensic_status,
            transformation_consistency=transformation_consistency,
            video_signals=video_signals,
            audio_signals=audio_signals,
            metadata_signals=meta_signals,
            ml_signals=ml_signals,
            limitations=limitations,
            processing_time_seconds=total_time,
        )

