# Step 6C: MediaProof Provenance + Forensic Decision Engine Architecture

## Overview

The MediaProof Verification Decision Engine synthesizes multi-layered empirical signals—cryptographic SHA-256 asset hashes, invisible video watermarking (spatial baseline & Meta VideoSeal), invisible audio watermarking (spread-spectrum baseline & Meta FAIR AudioSeal), SQLite provenance lineage metadata, and high-frequency forensic signal analysis—into defensible, context-aware provenance classifications.

---

## Technical Architecture

```
                                  [ Uploaded Media File ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Provenance Probing ]                     [ Multi-Modal Forensics ]
       ┌───────────────┴───────────────┐               ┌───────────┴───────────┐
       ▼                               ▼               ▼                       ▼
 [ SHA-256 Hash ]             [ Watermark Engines ] [ Video Artifacts ]  [ Audio Artifacts ]
  (Exact Original)            (Spatial / Neural)   (FFT, Noise, Blur)    (Phase Entropy, LFCC)
       │                               │               └───────────┬───────────┘
       ▼                               ▼                           ▼
 [ SQLite Lookup ]             [ Token Resolution ]         [ Forensic Risk ]
  (Asset Record)                (Hamming Distance)         (LOW / MEDIUM / HIGH)
       │                               │                           │
       └───────────────────────┬───────┴───────────────────────────┘
                               ▼
                   [ Multi-Case Decision Matrix ]
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
   VERIFIED ORIGINAL   VERIFIED DERIVATIVE   POTENTIALLY MANIPULATED
            ▼                  ▼                  ▼
  UNKNOWN PROVENANCE   INSUFFICIENT EVIDENCE
```

---

## Evidence Layers & Signal Fusion

1. **Layer 1: Cryptographic Provenance (Exact Match)**
   - Computes SHA-256 file digest.
   - Queries `assets` SQLite database for exact pre-registered asset match.
   - High-confidence deterministic signal for unmodified originals.

2. **Layer 2: Multi-Engine Invisible Watermarking**
   - **Video Engines**: `custom` (spatial DCT/frequency domain) and `videoseal` (Meta FAIR VideoSeal neural model).
   - **Audio Engines**: `custom` (spread-spectrum correlation) and `audioseal` (Meta FAIR AudioSeal neural model).
   - **Token Mapping**: 16-bit hex token extracted from watermark payload, mapped to `Asset ID` in SQLite with 1–2 bit Hamming distance tolerance for neural extraction noise.

3. **Layer 3: Lineage & Derivative Relationships**
   - Checks if resolved asset has `parent_asset_id` or derivative relationships in SQLite.

4. **Layer 4: Multi-Modal Forensic Signal Analysis**
   - **Video Signals**: High-frequency FFT spectrum artifacts, Laplacian blur/edge energy, temporal frame-to-frame noise variance.
   - **Audio Signals**: High-frequency spectral energy, phase entropy, LFCC variance, clipping/silence detection.
   - **Metadata Signals**: FFprobe codec anomaly checks, container format inconsistencies.
   - Combined into composite risk score: `LOW`, `MEDIUM`, or `HIGH`.

---

## Decision Matrix Rules

| Case | SHA-256 Hash | Watermark Token | Forensic Risk | Classification State | Notes & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CASE A** | MATCH | PRESENT_AND_RESOLVED (Same Asset) | LOW / MEDIUM | `VERIFIED ORIGINAL` | Cryptographic SHA-256 match confirmed and watermark matches registered asset. |
| **CASE A (Conflict)** | MATCH | PRESENT_AND_RESOLVED (Same Asset) | HIGH | `VERIFIED ORIGINAL` | Flagged: High forensic anomaly detected on exact hash match (container re-muxing or codec artifact). |
| **CASE B** | MISMATCH | PRESENT_AND_RESOLVED | LOW / MEDIUM | `VERIFIED DERIVATIVE` | Watermark token resolves to Asset ID `MP-XXX`; format transcode, re-compression, or crop derivative. |
| **CASE C** | MISMATCH | PRESENT_AND_RESOLVED | HIGH | `POTENTIALLY MANIPULATED` | Watermark token resolves to Asset ID `MP-XXX`, but severe spectral/compression anomalies detected. |
| **CASE D** | MATCH | NOT_DETECTED | LOW / MEDIUM | `VERIFIED ORIGINAL` | Exact SHA-256 hash match confirmed (watermark payload not embedded or unwatermarked original). |
| **CASE E** | MISMATCH | NOT_DETECTED | LOW / MEDIUM | `UNKNOWN PROVENANCE` | No cryptographic match and no watermark detected; un-tracked third-party asset. |
| **CASE F** | MISMATCH | NOT_DETECTED | HIGH | `POTENTIALLY MANIPULATED` | No cryptographic match and no watermark detected; high forensic signal risk detected. |
| **CASE G** | MISMATCH | PRESENT_BUT_UNRESOLVED | LOW / MEDIUM | `UNKNOWN PROVENANCE` | Watermark payload detected but hex token not found in SQLite registry. |
| **CASE G (High Risk)**| MISMATCH | PRESENT_BUT_UNRESOLVED | HIGH | `POTENTIALLY MANIPULATED` | Unregistered watermark token combined with high forensic anomaly risk. |
| **Conflict** | MATCH Asset A | RESOLVED Asset B | Any | `INSUFFICIENT EVIDENCE` | Cryptographic hash matches Asset ID `MP-A` but watermark payload resolves to Asset ID `MP-B`. |

---

## Defensive Classification Guarantees

1. **No False Certainty**: MediaProof never classifies media as "Deepfake Confirmed". High forensic risk yields `POTENTIALLY MANIPULATED`.
2. **Hash Mismatch Does Not Imply Tampering**: Format conversions, MP4 re-encoding, and resolution rescaling alter SHA-256 while preserving watermark token lineage (`VERIFIED DERIVATIVE`).
3. **Unwatermarked Media Preserves Integrity**: Unwatermarked original files with exact SHA-256 match are correctly classified as `VERIFIED ORIGINAL`.
4. **Conflict Guard**: Conflicting provenance signals yield `INSUFFICIENT EVIDENCE` rather than arbitrary resolution.
