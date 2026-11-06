# MediaProof — Adversarial Verification Test & Decision-Engine Correction Report

## Executive Summary

During an adversarial test of the MediaProof verification system, a deepfake video created from a VideoSeal-watermarked original was submitted to `/api/verify`. Because the neural watermark survived the deepfake manipulation, the verifier previously returned `VERIFIED DERIVATIVE` with `HIGH CONFIDENCE` despite a `MEDIUM` forensic risk score (`0.30`).

This report details the root cause analysis, key evidence semantics, decision matrix corrections, asset-ID mapping findings, and verification results.

---

## 1. Test Scenario & Discovered Failure

1. **Original Video**: Protected with MediaProof / VideoSeal watermarking.
2. **Deepfake Derivative**: Created from the protected original video.
3. **Submission**: Sent to `POST /api/verify`.
4. **Original Result (INCORRECT)**:
   - Classification: `VERIFIED DERIVATIVE`
   - Confidence: `HIGH`
   - Cryptographic Hash: `MISMATCH`
   - Watermark: `PRESENT_AND_RESOLVED` (`video_videoseal`, Token `86BA`)
   - Forensic Risk: `MEDIUM` (Risk Score: `0.30`)
   - Optional ML: `Not Installed`

### Root Cause
In `provenance/verification.py`, Case B previously permitted any media file with `hash_status == "MISMATCH"`, `wm_status == "PRESENT_AND_RESOLVED"`, and `forensic_cat in ["LOW", "MEDIUM", "INSUFFICIENT"]` to be classified as `VERIFIED DERIVATIVE`. This incorrectly treated `MEDIUM` forensic risk (`0.30`) as compatible with an authentic derivative and allowed watermark presence to override forensic anomaly evidence.

---

## 2. Defined Evidence Semantics

The decision engine was updated to strictly separate independent evidence layers:

* **Cryptographic SHA-256**: Proves exact byte-for-byte identity to a registered original file.
* **Watermark (VideoSeal / AudioSeal / Custom)**: Proves **provenance linkage** to a registered origin asset. It does **NOT** prove content authenticity, absence of manipulation, or safety.
* **Forensic Signals**: Provide multi-signal heuristic indicators of potential manipulation. Forensic signals remain evidence and cannot be overridden by watermark presence.
* **Optional ML**: Reports `Not Installed` when no verified deepfake ML checkpoint exists. No placeholder deepfake detectors are added.

---

## 3. Corrected Decision Matrix Policy

| Case | SHA-256 Hash | Watermark Status | Forensic Risk | Classification | Semantics & Explanation |
|---|---|---|---|---|---|
| **1** | `MATCH` | `PRESENT` / Unavailable | `LOW` / `MEDIUM` | `VERIFIED ORIGINAL` | Exact cryptographic identity confirmed. |
| **7** | `MATCH` | Any | `HIGH` | `INSUFFICIENT EVIDENCE` | Cryptographic match conflicts with high forensic anomaly indicators. |
| **2** | `MISMATCH` | `PRESENT_AND_RESOLVED` | `LOW` / `INSUFFICIENT` | `VERIFIED DERIVATIVE` | Linked to registered asset; no significant forensic anomaly detected. Does not guarantee content authenticity. |
| **3** | `MISMATCH` | `PRESENT_AND_RESOLVED` | `MEDIUM` / `HIGH` | `POTENTIALLY MANIPULATED` | Linked to registered asset via watermark, but forensic signals indicate possible manipulation. Watermark presence does not override forensic evidence. |
| **4** | `MISMATCH` | `PRESENT_BUT_UNRESOLVED` | Any | `UNKNOWN PROVENANCE` | Unmapped watermark payload detected. |
| **5** | `MISMATCH` | `NOT_DETECTED` | `LOW` / `INSUFFICIENT` | `UNKNOWN PROVENANCE` | No hash match or watermark found. Does not by itself indicate manipulation. |
| **6** | `MISMATCH` | `NOT_DETECTED` | `MEDIUM` / `HIGH` | `POTENTIALLY MANIPULATED` | Forensic signals indicate possible manipulation on unregistered media. |

---

## 4. Forensic Thresholds Review

* **Score Range**: `0.00` to `1.00`
* **Signal Weights**:
  * High-risk signals: `+0.40` per signal
  * Medium-risk signals: `+0.15` per signal
  * Raw Score formula: `min(max((high_count * 0.40) + (med_count * 0.15), 0.0), 1.0)`
* **Threshold Categories**:
  * `HIGH` (`HIGHER_RISK`): `risk_score >= 0.70` or `high_risk_count >= 2`
  * `MEDIUM` (`ELEVATED_RISK`): `risk_score >= 0.30` or `med_risk_count >= 2`
  * `LOW` (`LOW_RISK`): `risk_score < 0.30` and `med_risk_count < 2`
* **Nature**: Heuristic aggregation rules designed for multi-signal analysis without trained ML weights. Not calibrated probabilities.
* **Correction**: `MEDIUM` risk (`0.30`) is now treated as meaningful anomaly evidence that downgrades watermarked media from `VERIFIED DERIVATIVE` to `POTENTIALLY MANIPULATED`.

---

## 5. Asset Mapping Investigation & Cross-Modal Behavior

### Observed Anomaly
A video watermark payload resolved to an asset whose registered original filename was `orig_source.wav`.

### Investigation Findings
1. **Unconstrained Token Resolution**: `get_asset_by_watermark_token` originally executed `SELECT * FROM assets WHERE watermark_token = ?` without considering `media_type`.
2. **Hamming Fallback Behavior**: For VideoSeal (which enables 2-bit Hamming tolerance for extraction noise), searching across all media types in SQLite allowed a noisy video token to match an audio asset token if the audio token fell within 2 bit flips.
3. **Architectural Correction**: `get_asset_by_watermark_token` was updated to accept an optional `media_type` parameter ("video" / "audio"):
   - Step 1: Search exact token match matching `media_type`.
   - Step 2: Search exact token match across all media types (supporting intentional cross-modal exact linkage).
   - Step 3: Search Hamming-distance matches restricted to the matching `media_type` first.
   - Step 4: Search Hamming-distance matches across all media types only if no match was found.

---

## 6. Updated UI & API Language

* **Watermark Layer**: *"Watermark evidence establishes a linkage to a registered asset. It does not guarantee that derivative content is unmodified."*
* **Verified Derivative**: *"The file is linked to registered asset '{wm_asset_id}' and no significant forensic anomaly was detected by the available checks. This does not guarantee content authenticity."*
* **Potentially Manipulated**: *"The file has evidence associated with registered asset '{wm_asset_id}', but available forensic signals ({forensic_cat}) indicate possible manipulation. Watermark presence does not override this evidence."*
* **Unknown Provenance**: *"No reliable registered hash or watermark linkage was found. This does not by itself indicate manipulation."*

---

## 7. Verification Results

* **Adversarial Deepfake Test Case**: `watermark_detected` + `risk_score = 0.30 (MEDIUM)` → Classified as `POTENTIALLY MANIPULATED`.
* **Pytest Suite**: **55 passed / 0 failed** (including 15 verification decision matrix tests).
* **Portable Packaging**: Paused pending development environment verification.

---

## 8. Limitations & Scope Constraints

* Watermark recovery proves origin linkage, not content purity.
* Forensic risk scores remain heuristic signal aggregations, not certified probability scores.
* The system does not claim to detect all deepfakes or act as a universal deepfake classifier.
