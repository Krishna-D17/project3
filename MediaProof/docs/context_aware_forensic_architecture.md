# MediaProof — Context-Aware Forensic Evidence Architecture

## Overview & Background

The MediaProof verifier previously classified legitimate VideoSeal-watermarked video derivatives (`wm_000.mp4`) as `POTENTIALLY MANIPULATED` with `0.30` (`MEDIUM`) forensic risk.

Empirical decomposition revealed that generic high-frequency texture and inter-frame motion signals (`high_frequency_energy_ratio` and `temporal_noise_variation`) naturally exceeded static thresholds on uncompressed camera recordings. These generic technical signals were uncalibrated to content manipulation and produced identical false-positive scores across originals, transcodes, watermarked derivatives, and deepfakes.

To resolve this false-positive failure mode without hiding generic technical anomalies or making watermarked media blindly trusted, MediaProof implements a **Context-Aware Forensic Evidence Architecture**.

---

## Key Architectural Principles

1. **Isolation of Evidence Streams**:
   - **Generic Technical Anomaly (`generic_forensic_status`)**: Tracks signal energy, spatial contrast, spectral bounds, and noise levels (`LOW_GENERIC_ANOMALY`, `MEDIUM_GENERIC_ANOMALY`, `HIGH_GENERIC_ANOMALY`).
   - **Manipulation-Specific Detector (`manipulation_forensic_status`)**: Reserved strictly for validated ML deepfake/anti-spoofing detectors (`NO_DETECTOR_AVAILABLE`, `NO_MANIPULATION_DETECTED`, `MANIPULATION_SUSPECTED`).
   - **Transformation Consistency (`transformation_consistency`)**: Measures feature distance between a candidate derivative and its registered original parent asset baseline (`CONSISTENT`, `DEVIATION`, `NOT_AVAILABLE`).

2. **Parent Feature Diffing**:
   When a valid watermark payload resolves a candidate file to a registered parent asset in SQLite provenance, the forensic layer extracts parent signal metrics and computes absolute and relative tolerances across features.

3. **Context-Aware Decision Matrix**:
   - Registered derivatives (`Hash MISMATCH` + `Watermark RESOLVED`) with no deepfake detector anomaly (`MANIPULATION_SUSPECTED` absent) are classified as `VERIFIED DERIVATIVE`.
   - Generic technical anomalies (`MEDIUM_GENERIC_ANOMALY`) are explicitly reported in the evidence panel and rationale string ("Generic technical anomalies were observed... This does not guarantee content authenticity.").
   - Watermark presence never overrides validated manipulation-specific evidence (`MANIPULATION_SUSPECTED`).

---

## Data Model Extensions (`forensics/models.py`)

- **`ForensicSignal`**: Added `category` field (`CATEGORY_GENERIC_TECHNICAL`, `CATEGORY_METADATA`, `CATEGORY_TRANSFORMATION_CONSISTENCY`, `CATEGORY_MANIPULATION_SPECIFIC`).
- **`ParentFeatureDiff`**: Encapsulates `feature_name`, `parent_value`, `candidate_value`, `absolute_difference`, `relative_difference`, and `within_tolerance`.
- **`TransformationConsistencyResult`**: Tracks consistency status (`CONSISTENT`, `DEVIATION`, `NOT_AVAILABLE`), explanation, and feature diff list.
- **`UnifiedForensicResult`**: Extended with `generic_forensic_status`, `manipulation_forensic_status`, and `transformation_consistency`.

---

## Summary of Decision Matrix Rules

| Hash Status | Watermark Status | Generic Anomaly Status | Deepfake ML Status | Classification | Confidence | Rationale / Explanation |
|---|---|---|---|---|---|---|
| `MATCH` | Any | Any | `NO_DETECTOR_AVAILABLE` | `VERIFIED ORIGINAL` | `HIGH` | Exact SHA-256 cryptographic match |
| `MATCH` | Any | Any | `MANIPULATION_SUSPECTED` | `INSUFFICIENT EVIDENCE` | `LOW` | SHA-256 match conflicts with manipulation evidence |
| `MISMATCH` | `PRESENT_AND_RESOLVED` | `LOW / MEDIUM / HIGH` | `NO_DETECTOR_AVAILABLE` | `VERIFIED DERIVATIVE` | `HIGH` / `MEDIUM` | Linked to registered asset; generic technical anomalies noted; no deepfake evidence |
| `MISMATCH` | `PRESENT_AND_RESOLVED` | Any | `MANIPULATION_SUSPECTED` | `POTENTIALLY MANIPULATED` | `HIGH` | Linked to registered asset, but manipulation-specific evidence detected |
| `MISMATCH` | `NOT_DETECTED` | `HIGH_GENERIC_ANOMALY` | Any | `POTENTIALLY MANIPULATED` | `MEDIUM` | Unregistered file with high generic technical anomalies |
| `MISMATCH` | `NOT_DETECTED` | `LOW / MEDIUM` | `NO_DETECTOR_AVAILABLE` | `UNKNOWN PROVENANCE` | `MEDIUM` | Unregistered file; no watermark linkage found |

---

## Verification & Test Compliance

- **pytest Unit Test Suite**: 55 passed / 0 skipped / 0 failed (100% pass rate).
- **CPU & Offline Portability**: Fully verified on AMD64 Windows 10 without GPU dependencies.
- **User Interface**: Updated multi-layer evidence panel with separate Generic Anomaly, Parent Consistency, and ML Detector indicators.
