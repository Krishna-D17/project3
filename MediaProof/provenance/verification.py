import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from forensics.config import (
    STATUS_MANIPULATION_NO_ANOMALY,
    STATUS_MANIPULATION_NO_DETECTOR,
    STATUS_MANIPULATION_SUSPECTED,
)
from forensics.service import ForensicService
from media.service import MediaService
from provenance.hashing import calculate_sha256
from provenance.repository import get_asset_by_watermark_token
from provenance.service import ProvenanceService
from watermark.audio.detect import detect_audio_watermark
from watermark.video.detect import detect_video_watermark


# Final Classification Constants
STATE_VERIFIED_ORIGINAL = "VERIFIED ORIGINAL"
STATE_VERIFIED_DERIVATIVE = "VERIFIED DERIVATIVE"
STATE_POTENTIALLY_MANIPULATED = "POTENTIALLY MANIPULATED"
STATE_UNKNOWN_PROVENANCE = "UNKNOWN PROVENANCE"
STATE_INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


@dataclass
class EvidenceLayer:
    """Standardized evidence layer representation."""
    status: str
    strength: str  # EXTREME, STRONG, MEDIUM, WEAK, NONE
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "strength": self.strength,
            "explanation": self.explanation,
            "details": self.details,
        }


@dataclass
class VerificationResult:
    """Unified Verification & Decision Engine Output."""
    status: str  # ok / error
    classification: str  # Final state
    confidence: str      # overall_confidence: HIGH, MEDIUM, LOW
    provenance_confidence: str  # HIGH, MEDIUM, LOW, NONE
    authenticity_confidence: str  # HIGH, MEDIUM, LOW, UNKNOWN
    confidence_basis: List[str]
    asset_id: Optional[str]
    hash_evidence: EvidenceLayer
    watermark_evidence: EvidenceLayer
    provenance_evidence: EvidenceLayer
    forensic_evidence: EvidenceLayer
    metadata_evidence: EvidenceLayer
    manipulation_evidence: EvidenceLayer
    conflicts: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "classification": self.classification,
            "confidence": self.confidence,
            "provenance_confidence": self.provenance_confidence,
            "authenticity_confidence": self.authenticity_confidence,
            "confidence_basis": self.confidence_basis,
            "asset_id": self.asset_id,
            "hash_evidence": self.hash_evidence.to_dict(),
            "watermark_evidence": self.watermark_evidence.to_dict(),
            "provenance_evidence": self.provenance_evidence.to_dict(),
            "forensic_evidence": self.forensic_evidence.to_dict(),
            "metadata_evidence": self.metadata_evidence.to_dict(),
            "manipulation_evidence": self.manipulation_evidence.to_dict(),
            "conflicts": self.conflicts,
            "limitations": self.limitations,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
        }


class VerificationService:
    """MediaProof Provenance & Forensic Decision Engine Service."""

    @staticmethod
    def verify_media(
        file_path: Union[str, Path],
        video_engine: str = "custom",
        audio_engine: str = "custom",
        timeout: float = 60.0,
    ) -> VerificationResult:
        start_time = time.time()
        p = Path(file_path).resolve()

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Media file not found for verification: {p}")

        provenance_conflicts: List[str] = []
        conflicts: List[str] = []
        limitations: List[str] = []
        confidence_basis: List[str] = []

        # 1. Probe Media File & Calculate Cryptographic SHA-256
        try:
            probe_res = MediaService.probe_file(p)
            sha256_hash = calculate_sha256(p)
        except Exception as e:
            return VerificationResult(
                status="error",
                classification=STATE_INSUFFICIENT_EVIDENCE,
                confidence="LOW",
                provenance_confidence="NONE",
                authenticity_confidence="UNKNOWN",
                confidence_basis=["Failed to probe media or calculate hash"],
                asset_id=None,
                hash_evidence=EvidenceLayer("ERROR", "NONE", f"Media error: {str(e)}"),
                watermark_evidence=EvidenceLayer("DETECTION_ERROR", "NONE", "Watermark detection bypassed"),
                provenance_evidence=EvidenceLayer("NO_RECORD", "NONE", "Provenance lookup bypassed"),
                forensic_evidence=EvidenceLayer("INSUFFICIENT", "NONE", "Forensic analysis bypassed"),
                metadata_evidence=EvidenceLayer("ERROR", "NONE", "Metadata analysis bypassed"),
                manipulation_evidence=EvidenceLayer("NOT_AVAILABLE", "NONE", "Manipulation check bypassed"),
                limitations=[f"Processing error: {str(e)}"],
                processing_time_seconds=time.time() - start_time,
            )

        # 2. Hash Evidence & SQLite Hash Matching
        hash_matched_record = ProvenanceService.find_asset_by_sha256(sha256_hash)
        if hash_matched_record:
            hash_status = "MATCH"
            hash_strength = "EXTREME"
            hash_expl = f"SHA-256 matches registered original asset '{hash_matched_record.asset_id}'."
            confidence_basis.append("Exact SHA-256 cryptographic match")
        else:
            hash_status = "MISMATCH"
            hash_strength = "NONE"
            hash_expl = "SHA-256 hash does not match any registered original asset."

        hash_evidence = EvidenceLayer(
            status=hash_status,
            strength=hash_strength,
            explanation=hash_expl,
            details={"sha256": sha256_hash, "matched_asset_id": hash_matched_record.asset_id if hash_matched_record else None}
        )

        # 3. Watermark Evidence & Detection
        wm_detected = False
        wm_token = None
        wm_asset_id = None
        wm_engine_used = None
        wm_status = "NOT_DETECTED"
        wm_strength = "NONE"
        wm_expl = "No watermark payload detected."

        if probe_res.has_video:
            try:
                v_wm = detect_video_watermark(p, engine_type=video_engine, timeout=timeout)
                if v_wm.get("watermark_detected"):
                    wm_detected = True
                    wm_token = v_wm.get("watermark_token") or v_wm.get("details", {}).get("watermark_token")
                    wm_asset_id = v_wm.get("recovered_asset_id")
                    if wm_token and not wm_asset_id:
                        rec = get_asset_by_watermark_token(wm_token, allow_hamming=(video_engine == "videoseal"), media_type="video")
                        if rec:
                            wm_asset_id = rec.asset_id
                    wm_engine_used = f"video_{video_engine}"
            except Exception as e:
                limitations.append(f"Video watermark detection error: {str(e)}")

        if not wm_detected and probe_res.has_audio:
            try:
                a_wm = detect_audio_watermark(p, engine_type=audio_engine, timeout=timeout)
                if a_wm.get("watermark_detected"):
                    wm_detected = True
                    wm_token = a_wm.get("watermark_token") or a_wm.get("details", {}).get("watermark_token")
                    wm_asset_id = a_wm.get("recovered_asset_id")
                    if wm_token and not wm_asset_id:
                        rec = get_asset_by_watermark_token(wm_token, allow_hamming=(audio_engine == "audioseal"), media_type="audio")
                        if rec:
                            wm_asset_id = rec.asset_id
                    wm_engine_used = f"audio_{audio_engine}"
            except Exception as e:
                limitations.append(f"Audio watermark detection error: {str(e)}")

        if wm_detected:
            if wm_asset_id:
                wm_status = "PRESENT_AND_RESOLVED"
                wm_strength = "STRONG"
                wm_expl = f"Invisible watermark detected ({wm_engine_used}) and resolved to Asset ID '{wm_asset_id}'. Supports provenance linkage to registered Asset."
                confidence_basis.append(f"Resolved watermark token '{wm_token}' -> '{wm_asset_id}' (proves provenance linkage, not content authenticity)")
            else:
                wm_status = "PRESENT_BUT_UNRESOLVED"
                wm_strength = "WEAK"
                wm_expl = f"Watermark detected ({wm_engine_used}) but token '{wm_token}' could not be resolved to SQLite asset."
                limitations.append("Watermark payload detected but not present in database registry.")
        else:
            wm_expl = "No watermark payload detected. Watermark absence does not by itself indicate manipulation."

        watermark_evidence = EvidenceLayer(
            status=wm_status,
            strength=wm_strength,
            explanation=wm_expl,
            details={
                "watermark_detected": wm_detected,
                "token": wm_token,
                "resolved_asset_id": wm_asset_id,
                "engine_used": wm_engine_used,
            }
        )

        # 4. Provenance Evidence Lookup
        prov_record = None
        if wm_asset_id:
            prov_record = ProvenanceService.get_asset(wm_asset_id)
        elif hash_matched_record:
            prov_record = hash_matched_record

        if prov_record:
            prov_status = "RECORD_FOUND"
            prov_strength = "STRONG"
            prov_expl = f"Registered provenance record retrieved for Asset ID '{prov_record.asset_id}'."
        else:
            prov_status = "NO_RECORD"
            prov_strength = "NONE"
            prov_expl = "No reliable registered hash or watermark linkage was found. This does not by itself indicate manipulation."

        provenance_evidence = EvidenceLayer(
            status=prov_status,
            strength=prov_strength,
            explanation=prov_expl,
            details=prov_record.to_dict() if prov_record else {}
        )

        # 5. Forensic & Metadata Signals Analysis with Context-Aware Parent Baseline
        parent_file_path = None
        if prov_record and prov_record.original_path:
            cand_parent = Path(prov_record.original_path).resolve()
            if cand_parent.exists() and cand_parent.is_file() and cand_parent != p:
                parent_file_path = cand_parent

        try:
            forensics_res = ForensicService.analyze_media(p, parent_file_path=parent_file_path, timeout=timeout)
            forensic_status_raw = forensics_res.forensic_status
            forensic_score = forensics_res.risk_score
            forensic_limitations = forensics_res.limitations
            gen_status = forensics_res.generic_forensic_status
            manip_status = forensics_res.manipulation_forensic_status
            trans_cons = forensics_res.transformation_consistency
            limitations.extend(forensic_limitations)

            if manip_status == "MANIPULATION_SUSPECTED":
                forensic_cat = "HIGH"
                forensic_strength = "HIGH"
                forensic_expl = f"Validated manipulation-specific evidence detected (Risk Score: {forensic_score:.2f})."
            elif gen_status == "HIGH_GENERIC_ANOMALY":
                forensic_cat = "HIGH"
                forensic_strength = "HIGH"
                forensic_expl = f"High generic spatial/acoustic technical anomalies detected (Risk Score: {forensic_score:.2f})."
            elif gen_status == "MEDIUM_GENERIC_ANOMALY":
                forensic_cat = "MEDIUM"
                forensic_strength = "MEDIUM"
                forensic_expl = f"Generic technical anomalies detected ({gen_status}, Risk Score: {forensic_score:.2f}). No validated deepfake detector anomaly."
            elif gen_status == "LOW_GENERIC_ANOMALY":
                forensic_cat = "LOW"
                forensic_strength = "LOW"
                forensic_expl = f"Low generic anomaly indicators (Risk Score: {forensic_score:.2f})."
            else:
                forensic_cat = "INSUFFICIENT"
                forensic_strength = "NONE"
                forensic_expl = "Insufficient forensic evidence extracted."

            forensic_evidence = EvidenceLayer(
                status=forensic_cat,
                strength=forensic_strength,
                explanation=forensic_expl,
                details={
                    "forensic_risk_score": forensic_score,
                    "risk_score_type": "heuristic",
                    "raw_status": forensic_status_raw,
                    "generic_forensic_status": gen_status,
                    "manipulation_forensic_status": manip_status,
                    "transformation_consistency": trans_cons.to_dict(),
                    "signals_count": len(forensics_res.video_signals) + len(forensics_res.audio_signals),
                    "ml_signals": [m.to_dict() for m in forensics_res.ml_signals],
                }
            )

            meta_signals = forensics_res.metadata_signals
            anom_meta = [s.signal_name for s in meta_signals if s.risk_level in ["MEDIUM", "HIGH"]]
            if anom_meta:
                meta_status = "ANOMALOUS"
                meta_expl = f"Metadata anomalies detected: {', '.join(anom_meta)}"
            else:
                meta_status = "STANDARD"
                meta_expl = "Standard container and stream metadata parameters."

            metadata_evidence = EvidenceLayer(
                status=meta_status,
                strength="LOW",
                explanation=meta_expl,
                details={"anomalies_found": anom_meta}
            )

        except Exception as e:
            forensic_evidence = EvidenceLayer("INSUFFICIENT", "NONE", f"Forensic evaluation failed: {str(e)}")
            metadata_evidence = EvidenceLayer("ERROR", "NONE", "Metadata analysis failed")
            forensic_cat = "INSUFFICIENT"
            gen_status = "LOW_GENERIC_ANOMALY"
            manip_status = "NO_DETECTOR_AVAILABLE"
            limitations.append(f"Forensic evaluation exception: {str(e)}")

        # Build Separate Manipulation Evidence Layer
        if manip_status == STATUS_MANIPULATION_SUSPECTED:
            manip_layer_status = "MANIPULATION_SUSPECTED"
            manip_layer_strength = "HIGH"
            manip_layer_expl = "Validated manipulation-specific evidence detected by installed ML model."
        elif manip_status in [STATUS_MANIPULATION_NO_ANOMALY, "NO_MANIPULATION_DETECTED", "NO_ANOMALY"]:
            manip_layer_status = "NO_ANOMALY_DETECTED"
            manip_layer_strength = "MEDIUM"
            manip_layer_expl = "No manipulation-specific anomaly detected by installed ML model."
        else:
            manip_layer_status = "NOT_AVAILABLE"
            manip_layer_strength = "NONE"
            manip_layer_expl = "No validated deepfake/anti-spoofing detector model is installed."

        manipulation_evidence = EvidenceLayer(
            status=manip_layer_status,
            strength=manip_layer_strength,
            explanation=manip_layer_expl,
            details={"manipulation_forensic_status": manip_status}
        )

        # 6. Provenance Conflict Detection
        if wm_asset_id and hash_matched_record:
            if (wm_asset_id != hash_matched_record.asset_id) and (hash_matched_record.parent_asset_id != wm_asset_id):
                conflict_msg = (
                    f"Provenance Conflict: Watermark resolves to Asset ID '{wm_asset_id}' "
                    f"but SHA-256 hash matches Asset ID '{hash_matched_record.asset_id}'."
                )
                provenance_conflicts.append(conflict_msg)
                conflicts.append(conflict_msg)

        # 7. Decision Matrix Evaluation (Strict Provenance vs Authenticity Confidence Separation)
        # Initialize decision variables to safe baseline defaults
        classification = STATE_UNKNOWN_PROVENANCE
        confidence = "LOW"
        provenance_confidence = "NONE"
        authenticity_confidence = "UNKNOWN"
        primary_asset_id = None

        if provenance_conflicts:
            classification = STATE_INSUFFICIENT_EVIDENCE
            confidence = "LOW"
            provenance_confidence = "LOW"
            authenticity_confidence = "LOW"
            primary_asset_id = None
            confidence_basis.append("Conflicting provenance indicators present")

        # CASE A: Cryptographic Hash Match (Exact Original or Conflict)
        elif hash_status == "MATCH":
            primary_asset_id = hash_matched_record.asset_id if hash_matched_record else None
            provenance_confidence = "HIGH"
            if manip_status == STATUS_MANIPULATION_SUSPECTED:
                # Conflicting evidence: Hash MATCH + MANIPULATION SUSPECTED
                classification = STATE_INSUFFICIENT_EVIDENCE
                confidence = "LOW"
                authenticity_confidence = "LOW"
                conflict_msg = "Exact SHA-256 match conflicts with validated manipulation-specific evidence."
                conflicts.append(conflict_msg)
                confidence_basis.append(conflict_msg)
            else:
                # Exact registered original
                classification = STATE_VERIFIED_ORIGINAL
                confidence = "HIGH"
                authenticity_confidence = "HIGH"
                confidence_basis.append("Exact SHA-256 cryptographic match confirms byte-for-byte identity to registered original asset.")
                if wm_status != "PRESENT_AND_RESOLVED":
                    limitations.append("Watermark evidence unavailable or not detected; classification based on SHA-256 identity.")

        # CASE B & C: Registered derivative with valid provenance watermark
        elif hash_status == "MISMATCH" and wm_status == "PRESENT_AND_RESOLVED":
            primary_asset_id = wm_asset_id
            provenance_confidence = "HIGH"
            if manip_status == STATUS_MANIPULATION_SUSPECTED:
                # Validated manipulation evidence -> POTENTIALLY MANIPULATED
                classification = STATE_POTENTIALLY_MANIPULATED
                confidence = "HIGH"
                authenticity_confidence = "LOW"
                confidence_basis.append(
                    f"The file is linked to registered asset '{wm_asset_id}', but manipulation-specific evidence was detected. "
                    "Watermark provenance does not override this evidence."
                )
            elif manip_status in [STATUS_MANIPULATION_NO_ANOMALY, "NO_MANIPULATION_DETECTED", "NO_ANOMALY"]:
                # Registered derivative, ML detector installed and clean
                classification = STATE_VERIFIED_DERIVATIVE
                confidence = "MEDIUM"
                authenticity_confidence = "MEDIUM"
                confidence_basis.append(
                    f"Provenance linkage confirmed to registered asset '{wm_asset_id}'. "
                    "Installed ML detector reported no manipulation anomaly."
                )
            else:
                # Registered derivative, NO ML detector installed / unavailable
                # Covers STATUS_MANIPULATION_NO_DETECTOR, "NO_VALIDATED_DETECTOR", "NO_DETECTOR_AVAILABLE"
                classification = STATE_VERIFIED_DERIVATIVE
                confidence = "MEDIUM"  # MUST NOT BE HIGH!
                authenticity_confidence = "UNKNOWN"
                confidence_basis.append(
                    f"Provenance linkage confirmed to registered asset '{wm_asset_id}'; watermark survival does not prove the derivative is unmodified. "
                    "No validated manipulation-specific detector is available."
                )

        # CASE D: Watermark present but unresolved provenance record
        elif hash_status == "MISMATCH" and wm_status == "PRESENT_BUT_UNRESOLVED":
            classification = STATE_UNKNOWN_PROVENANCE
            confidence = "LOW"
            provenance_confidence = "LOW"
            authenticity_confidence = "UNKNOWN"
            primary_asset_id = None
            confidence_basis.append("No reliable registered hash or watermark linkage was found. Watermark payload detected but token unmapped.")

        # CASE E & F: No watermark and Hash mismatch (Unregistered File)
        elif hash_status == "MISMATCH" and wm_status == "NOT_DETECTED":
            primary_asset_id = None
            provenance_confidence = "NONE"
            if manip_status == STATUS_MANIPULATION_SUSPECTED:
                classification = STATE_POTENTIALLY_MANIPULATED
                confidence = "HIGH"
                authenticity_confidence = "LOW"
                confidence_basis.append("Validated manipulation-specific evidence detected on an unregistered file.")
            elif gen_status == "HIGH_GENERIC_ANOMALY":
                classification = STATE_POTENTIALLY_MANIPULATED
                confidence = "MEDIUM"
                authenticity_confidence = "LOW"
                confidence_basis.append(f"High generic technical anomalies detected ({gen_status}) on an unregistered file.")
            else:
                classification = STATE_UNKNOWN_PROVENANCE
                confidence = "MEDIUM"
                authenticity_confidence = "UNKNOWN"
                confidence_basis.append("No reliable registered hash or watermark linkage was found. Watermark absence does not by itself indicate manipulation.")

        else:
            classification = STATE_UNKNOWN_PROVENANCE
            confidence = "LOW"
            provenance_confidence = "NONE"
            authenticity_confidence = "UNKNOWN"
            primary_asset_id = None

        total_time = time.time() - start_time
        return VerificationResult(
            status="ok",
            classification=classification,
            confidence=confidence,
            provenance_confidence=provenance_confidence,
            authenticity_confidence=authenticity_confidence,
            confidence_basis=confidence_basis,
            asset_id=primary_asset_id,
            hash_evidence=hash_evidence,
            watermark_evidence=watermark_evidence,
            provenance_evidence=provenance_evidence,
            forensic_evidence=forensic_evidence,
            metadata_evidence=metadata_evidence,
            manipulation_evidence=manipulation_evidence,
            conflicts=conflicts,
            limitations=limitations,
            processing_time_seconds=total_time,
        )
