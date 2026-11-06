# MediaProof: Multimodal Media Provenance & Forensic Verification Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Inference-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20(Local%20%26%20Offline)-0078D6?style=flat&logo=windows&logoColor=white)](https://microsoft.com/windows)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-7.0.2%20Static-007808?style=flat&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/Tests-53%20Passed-brightgreen?style=flat&logo=pytest&logoColor=white)](#testing--verification)

> **A local, portable, CPU-first digital media provenance, invisible watermarking, and forensic decision engine for Windows.**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
  - [1. Dual-Engine Video Watermarking](#1-dual-engine-video-watermarking)
  - [2. Dual-Engine Audio Watermarking](#2-dual-engine-audio-watermarking)
  - [3. 256-Bit Cryptographic Payload](#3-256-bit-cryptographic-payload)
  - [4. Multi-Modal Forensic Analysis](#4-multi-modal-forensic-analysis)
  - [5. Tamper-Evident Hash Chain & Ledger](#5-tamper-evident-hash-chain--ledger)
  - [6. Five-State Verification Decision Matrix](#6-five-state-verification-decision-matrix)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Manual Setup](#manual-setup)
- [API Reference](#api-reference)
- [Web Interface](#web-interface)
- [Testing & Verification](#testing--verification)
- [Related Research & Upgrade Plan](#related-research--upgrade-plan)
- [License](#license)

---

## Overview

In an era of generative AI, deepfakes, and automated media manipulation, establishing the authenticity and chain of custody for digital media is paramount. **MediaProof** provides an end-to-end framework designed to run completely **locally and offline on consumer CPU hardware**. 

MediaProof combines:
1. **Cryptographic Identity**: SHA-256 exact-match asset registration.
2. **Invisible Watermarking**: Dual-engine spatial and neural watermarking with 256-bit payloads.
3. **Temporal Perceptual Fingerprinting**: 64-bit frame-by-frame difference hashing (`dHash`) to detect sequence splicing, trimming, and reordering.
4. **Technical Metadata Drift**: Audio/video container, stream, and codec parameter comparison.
5. **Acoustic & Visual Forensics**: Signal anomaly detection and DeepfakeBench Xception neural classification.
6. **Immutable Provenance Ledger**: Tamper-evident hash-chained event logs serialized via RFC 8785 canonical JSON.

---

## System Architecture

```text
                               ┌────────────────────────────────┐
                               │       Input Media File         │
                               └───────────────┬────────────────┘
                                               │
                                 Extract SHA-256 Checksum &
                                  FFprobe Stream Properties
                                               │
                 ┌─────────────────────────────┼─────────────────────────────┐
                 │                             │                             │
                 ▼                             ▼                             ▼
       ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
       │ Provenance Search │         │ Watermark Engine  │         │ Multi-Modal       │
       │ (SQLite Database) │         │ (Custom / Neural) │         │ Forensic Pipeline │
       └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
                 │                             │                             │
                 │ Record Match                │ Asset ID Decoded            │ Anomaly / Deepfake
                 ▼                             ▼                             ▼
       ┌───────────────────────────────────────────────────────────────────────────────┐
       │                        Central Verification Decision Engine                   │
       └───────────────────────────────────────┬───────────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   [Verified Original]              [Verified Derivative]            [Potentially Manipulated]
   • SHA-256 exact match            • Watermark matches parent       • Forensic anomalies / drift
   • Zero metadata drift            • Metadata transcode detected    • Hash chain breach
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                       [Unknown Provenance]       [Insufficient Evidence]
                       • Unregistered asset       • Corrupted stream / decode fail
```

---

## Key Features

### 1. Dual-Engine Video Watermarking

MediaProof provides two interchangeable video watermarking backends via a clean factory interface:

```text
                               MediaProof Video Watermark API
                                 (embed / detect endpoints)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                      engine="custom"                 engine="videoseal"
                              │                               │
                              ▼                               ▼
                  CustomVideoWatermarkEngine          MetaVideoSealEngine
                   (Deterministic Spatial)           (Learned PyTorch CNN)
```

- **MediaProof Custom Spatial Engine (`engine="custom"`)**:
  - Extremely fast (< 1.0 s per clip on standard CPU).
  - Deterministic vertical green-channel pair modulation.
  - Zero neural weights required; 100% resilient across H.264 compression in standard test fixtures.
- **Meta VideoSeal Neural Engine (`engine="videoseal"`)**:
  - Based on Meta/Facebook Research's `videoseal` architecture.
  - Learned neural watermarking operating in PyTorch CPU inference mode.

### 2. Dual-Engine Audio Watermarking

Similarly, audio protection offers dual options:
- **Custom Audio Engine**: Audio spectrum phase/amplitude spatial modulation for fast, dependency-free watermarking.
- **Meta AudioSeal Engine**: Neural speech and audio watermarking with high imperceptibility and tamper resilience.

### 3. 256-Bit Cryptographic Payload

Every embedded watermark contains an immutable 32-byte (256-bit) structured payload:

| Byte Range | Field Name | Size | Description |
| :--- | :--- | :--- | :--- |
| `Bytes 0..1` | **Magic Header** | 16 bits | `0x4D50` ("MP" in ASCII) prefix identifier |
| `Bytes 2..17` | **Asset UUID** | 128 bits | Standard RFC 4122 UUID v4 binary representation |
| `Bytes 18..21` | **Integrity Hash** | 32 bits | First 4 bytes of asset SHA-256 checksum |
| `Bytes 22..23` | **Flags / Reserved** | 16 bits | Derivative/generation version flags |
| `Bytes 24..31` | **Padding / ECC** | 64 bits | Zero-margin padding and error correction |

### 4. Multi-Modal Forensic Analysis

A passing watermark does not guarantee that the content was not altered after watermarking. MediaProof evaluates four distinct forensic layers:
1. **Temporal dHash Matching**: Extracts 64-bit difference hashes for every second of video, computing normalized Hamming distance ($d_N = d_H / 64$) to flag scene splicing, insertions, or deletions.
2. **Metadata Drift Analyzer**: Compares codec IDs, container wrappers, sample rates, durations, and aspect ratios to separate benign transcoding from malicious tampering.
3. **Signal & Acoustic Processing**: Detects frame-rate discontinuities, silence truncation, and frequency anomalies.
4. **Deepfake ML Classification**: Local PyTorch CPU inference with an Xception-based deepfake detector trained on standard deepfake benchmarks.

### 5. Tamper-Evident Hash Chain & Ledger

All provenance registrations and state transitions are recorded into an append-only cryptographic ledger:
- **RFC 8785 Canonical JSON**: Deterministic key ordering and whitespace rules eliminate JSON signature malleability.
- **Cryptographic Hash Chaining**: Each record links to `previous_event_hash` via SHA-256:
  $$\text{event\_hash}_n = \text{SHA-256}(\text{event\_hash}_{n-1} \parallel \text{canonical\_json}(E_n))$$
- **Ed25519 Signatures**: Validates identity and non-repudiation across provenance nodes.

### 6. Five-State Verification Decision Matrix

| Classification | Verification State Criteria |
| :--- | :--- |
| **Verified Original** | Registered SHA-256 matches exact bytes; zero forensic anomalies. |
| **Verified Derivative** | Valid watermark links to known asset; metadata drift is consistent with platform transcoding (e.g. WhatsApp/YouTube). |
| **Potentially Manipulated** | Asset link verified, but temporal dHash discontinuities, deepfake signals, or hash-chain violations detected. |
| **Unknown Provenance** | No watermark payload recovered, and SHA-256 does not match any registered asset. |
| **Insufficient Evidence** | Media corrupted, unreadable streams, or detector execution failed (prevents false negatives). |

---

## Repository Structure

```
project3/
├── MediaProof/                     # Core application package
│   ├── backend/                    # FastAPI REST server & routing
│   │   ├── main.py                 # API endpoints & application startup
│   │   └── paths.py                # Local filesystem resolution
│   ├── forensics/                  # Multi-modal forensic analyzers
│   │   ├── deepfakebench_xception.py  # Local Xception deepfake classifier
│   │   ├── metadata_drift.py       # Container/codec comparison engine
│   │   ├── temporal_dhash.py       # Frame-level perceptual hashing
│   │   └── service.py              # Forensic pipeline orchestrator
│   ├── frontend/                   # Modern responsive Web UI
│   │   ├── index.html              # Single-page interface
│   │   ├── style.css               # Styling and dashboard layout
│   │   └── app.js                  # Async client-side workflow controller
│   ├── media/                      # FFmpeg media processing & stream probe
│   ├── packaging/                  # PyInstaller build specification
│   ├── provenance/                 # Cryptographic ledger, SQLite DB & hash chain
│   ├── watermark/                  # Dual video & audio watermarking implementations
│   ├── tests/                      # Automated test suite (53 tests)
│   ├── docs/                       # Engineering specifications & audit reports
│   ├── scripts/                    # Environment checks & model downloaders
│   ├── setup.bat                   # Automated Windows virtualenv setup script
│   ├── start.bat                   # One-click Windows launch script
│   └── requirements.txt            # Python dependencies
├── testcase/                       # Sample media files for testing and evaluation
├── MediaProof_Comprehensive_Upgrade_Plan.md  # Complete upgrade architecture plan
├── blockchain_viability_analysis.md          # Decentralized ledger viability study
├── mediaproof_enhancement_plan.md            # System enhancement blueprint
└── README.md                       # Repository documentation
```

---

## Quick Start

### Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.11 installed (available on `PATH` or launcher `py -3.11`)
- **FFmpeg**: Automatically downloaded by the setup script, or manually placed in `MediaProof/tools/ffmpeg/`

### Automated Setup (Recommended)

1. Open a Command Prompt or PowerShell in `project3\MediaProof\`:
   ```cmd
   cd MediaProof
   setup.bat
   ```
   *This creates `.venv`, upgrades `pip`, installs all dependencies, downloads FFmpeg if needed, runs environment checks, and executes the test suite.*

2. Start the local server:
   ```cmd
   start.bat
   ```
3. Your default web browser will automatically open:
   ```text
   http://127.0.0.1:8000/
   ```

### Manual Setup

If you prefer configuring the environment manually:

```bash
# 1. Navigate to MediaProof directory
cd MediaProof

# 2. Create and activate virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Download FFmpeg static binaries
python scripts/download_ffmpeg.py

# 5. (Optional) Download Meta VideoSeal weights for neural watermarking
python scripts/download_videoseal_models.py

# 6. Run test suite to verify installation
pytest tests/

# 7. Launch development server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and CPU status indicator |
| `GET` | `/api/media/status` | Reports FFmpeg/FFprobe binary availability |
| `POST` | `/api/media/probe` | Extracts technical container and stream metadata |
| `POST` | `/api/provenance/register` | Registers media SHA-256 and metadata in the provenance ledger |
| `GET` | `/api/provenance/{asset_id}` | Retrieves provenance record by Asset ID |
| `POST` | `/api/watermark/video/embed` | Embeds 256-bit watermark (`custom` or `videoseal` engine) |
| `POST` | `/api/watermark/video/detect` | Extracts watermark payload and resolves asset linkage |
| `POST` | `/api/watermark/audio/embed` | Embeds watermark into audio track (`custom` or `audioseal`) |
| `POST` | `/api/watermark/audio/detect` | Extracts watermark from audio stream |
| `POST` | `/api/forensics/analyze` | Executes multi-modal forensic inspection |
| `POST` | `/api/verify` | **Unified verification endpoint**: full decision matrix analysis |

---

## Web Interface

MediaProof includes an intuitive, self-contained web dashboard accessible at `http://127.0.0.1:8000/`:

- **Asset Registration Tab**: Upload original media, register cryptographic fingerprints, and issue Asset IDs.
- **Watermark Protection Tab**: Embed imperceptible spatial or neural watermarks into videos and audio files with one click.
- **Forensic Verification Tab**: Upload any target media file to run automated watermark detection, metadata drift comparison, temporal fingerprinting, and decision engine categorization with full explainability.

---

## Testing & Verification

MediaProof is thoroughly validated with an automated test suite covering:
- Unit tests for custom & neural watermarking engines
- FFmpeg execution failure handling and stream probing
- Temporal difference hash (`dHash`) persistence
- Metadata drift detection and benign transcoding differentiation
- Tamper-evident hash chain serialization and consistency
- Deepfake classifier inference and fallback handling

Run tests inside your virtual environment:

```bash
pytest tests/ -v
```

Expected output:
```text
======================== 53 passed in 14.82s ========================
```

---

## Related Research & Upgrade Plan

This repository contains in-depth engineering documentation and academic viability analyses:
- [MediaProof Comprehensive Upgrade Plan](MediaProof_Comprehensive_Upgrade_Plan.md): Detailed specifications for forensic false-positive auditing, temporal dHash indexing, and zero-knowledge evidence preservation.
- [Blockchain Viability Analysis](blockchain_viability_analysis.md): Comparative analysis between local hash chaining and decentralized blockchain ledgers for legal evidence.
- [MediaProof Enhancement Plan](mediaproof_enhancement_plan.md): Architectural enhancements for high-throughput media verification pipelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
