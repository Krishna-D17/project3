# Step 7A: Complete End-to-End Backend & Portability Audit Report

## 1. Complete API Inventory

| Method | Endpoint Path | Purpose | Input | Output | Expected Status | Core Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Root status endpoint | None | JSON `{"status": "ok", ...}` | `200 OK` | FastAPI |
| `GET` | `/api/health` | System health check | None | JSON `{"status": "healthy", ...}` | `200 OK` | MediaService, FFmpeg |
| `POST` | `/api/provenance/register` | Register media asset in SQLite | Multipart file upload | AssetRecord JSON | `200 OK` / `400` | SQLite, FFprobe |
| `GET` | `/api/provenance/asset/{asset_id}` | Lookup asset by Asset ID | `asset_id` path param | AssetRecord JSON | `200 OK` / `404` | SQLite |
| `GET` | `/api/provenance/asset/hash/{sha256}` | Lookup asset by SHA-256 | `sha256` path param | AssetRecord JSON | `200 OK` / `404` | SQLite |
| `POST` | `/api/watermark/video/embed` | Embed video watermark | Multipart file, `asset_id`, `engine` | VideoWatermarkEmbedResult | `200 OK` / `400` | Custom / VideoSeal Engine |
| `POST` | `/api/watermark/video/detect` | Detect video watermark | Multipart file, `engine` | VideoWatermarkDetectResult | `200 OK` / `400` | Custom / VideoSeal Engine |
| `POST` | `/api/watermark/audio/embed` | Embed audio watermark | Multipart file, `asset_id`, `engine` | AudioWatermarkEmbedResult | `200 OK` / `400` | Custom / AudioSeal Engine |
| `POST` | `/api/watermark/audio/detect` | Detect audio watermark | Multipart file, `engine` | AudioWatermarkDetectResult | `200 OK` / `400` | Custom / AudioSeal Engine |
| `POST` | `/api/forensics/analyze` | Multi-modal forensic analysis | Multipart file | UnifiedForensicResult | `200 OK` / `400` | ForensicService |
| `POST` | `/api/verify` | Full verification decision engine | Multipart file, `video_engine`, `audio_engine` | VerificationResult JSON | `200 OK` / `400` | VerificationService |

---

## 2. End-to-End Video Workflow Audit (`scratch/audit_video_results.json`)

- **Synthetic Video Asset**: 3-second 320x240 @ 15fps H.264 MP4 (`audit_video_orig.mp4`).
- **Registration**: Assigned Asset ID `MP-c41ab316-5936-43f5-9db4-d6483419cf3f` with 16-bit hex token `77cd`.
- **VideoSeal Embedding**: Embedded Asset ID into `audit_video_vseal_wm.mp4` using Meta VideoSeal engine (`engine="videoseal"`).
- **VideoSeal Detection**: Recovered 16-bit token `77cd` (0 Hamming bit flips) and resolved Asset ID `MP-c41ab316-5936-43f5-9db4-d6483419cf3f`.
- **Forensic Signals**: Composite risk level `LOW`.
- **API `/api/verify` Result**: Classification `VERIFIED DERIVATIVE` (since file was modified by embedding watermark), confidence `HIGH`.

---

## 3. End-to-End Audio Workflow Audit (`scratch/audit_audio_results.json`)

- **Synthetic Audio Asset**: 3-second 16kHz PCM mono WAV (`audit_audio_orig.wav`).
- **Registration**: Assigned Asset ID `MP-780c6fb0-6bb1-4df2-8cf0-569d67ebf8d4` with 16-bit hex token `0fef`.
- **AudioSeal Embedding**: Embedded Asset ID into `audit_audio_aseal_wm.wav` using Meta FAIR AudioSeal engine (`engine="audioseal"`).
- **AudioSeal Detection**: Recovered 16-bit token `0fef` (0 Hamming bit flips) and resolved Asset ID `MP-780c6fb0-6bb1-4df2-8cf0-569d67ebf8d4`.
- **Forensic Signals**: Composite risk level `LOW`.
- **API `/api/verify` Result**: Classification `VERIFIED DERIVATIVE`, confidence `HIGH`.

---

## 4. Combined Media Workflow Audit

- **Combined Asset**: MP4 video containing H.264 video track and AAC audio track.
- **Single Asset ID (Video & Audio)**: Watermarking video and audio with Asset ID `MP-A` produces `VERIFIED DERIVATIVE` with `0 conflicts`.
- **Dual Asset IDs (Video Asset A, Audio Asset B)**: When video contains watermark for Asset ID `MP-A` and audio track contains watermark for Asset ID `MP-B`, the decision engine evaluates watermark evidence and logs `Provenance Conflict` in evidence details while maintaining defensible classification without crashing.

---

## 5. Decision Matrix Audit Results

All 12 test scenarios were verified against the decision engine:
1. **Exact Original (Watermarked)**: Correctly matched SHA-256 and watermark payload.
2. **Exact Original (Unwatermarked)**: Classified as `VERIFIED ORIGINAL` based on exact SHA-256 digest match.
3. **H.264 Transcode Derivative**: SHA-256 changed, watermark token recovered -> `VERIFIED DERIVATIVE`.
4. **MP3 Re-encoded Audio Derivative**: SHA-256 changed, AudioSeal token recovered -> `VERIFIED DERIVATIVE`.
5. **Valid Watermark + Low Forensic Risk**: `VERIFIED DERIVATIVE`.
6. **Valid Watermark + High Forensic Anomaly**: `POTENTIALLY MANIPULATED`.
7. **No Watermark + Low Forensic Risk**: `UNKNOWN PROVENANCE`.
8. **No Watermark + High Forensic Risk**: `POTENTIALLY MANIPULATED`.
9. **Watermark Asset A + Hash Asset B**: Flagged `INSUFFICIENT EVIDENCE` with explicit `Provenance Conflict` explanation.
10. **Unknown Watermark Token**: Payload detected but token unregistered -> `UNKNOWN PROVENANCE`.
11. **Forensic Subsystem Unavailable**: Bypassed gracefully with limitation warning.
12. **Watermark Detector Exception**: Bypassed gracefully with limitation warning.

---

## 6. Failure & Recovery Audit

- **Unsupported Extension / Corrupt / Zero-Byte File**: Returns graceful JSON with status `"error"`, classification `INSUFFICIENT EVIDENCE`, and readable limitation message without exposing unhandled Python stack traces.
- **Invalid Asset ID**: Returns `400 Bad Request` with structured error message `Invalid Asset ID UUID string...`.
- **Invalid Watermark Engine**: Returns `400 Bad Request` listing supported engine options (`'custom'`, `'videoseal'`).

---

## 7. Offline Dependency Audit

- **Network Status**: 100% OFFLINE CAPABLE.
- **Model Checkpoints**: Meta VideoSeal (`ckpts/videoseal_y_256b_img.pth`) and Meta AudioSeal checkpoints are loaded locally from disk.
- **Zero Runtime Downloads**: API requests never attempt HTTP requests or model downloads.

---

## 8. Model Inventory

| Filename | Path | Size (MB) | Framework | Purpose | Required/Optional |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `videoseal_y_256b_img.pth` | `ckpts/videoseal_y_256b_img.pth` | 217.77 MB | PyTorch JIT / TorchScript | Meta VideoSeal Neural Video Watermarking | Required for `videoseal` engine |
| `audioseal_wm_16bits/` | Meta FAIR AudioSeal local weights | ~1.50 MB | PyTorch / OmegaConf | Meta AudioSeal Neural Audio Watermarking | Required for `audioseal` engine |

---

## 9. Dependency Inventory

- `requirements.txt`: Lightweight core backend (`fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `opencv-python`, `scipy`, `numpy`, `pillow`, `pytest`).
- `requirements-ml.txt`: Heavyweight ML engines (`torch`, `torchaudio`, `videoseal`, `audioseal`, `timm`, `omegaconf`).

---

## 10. Project Size Audit

- **Total Project Size (WITH `.venv`)**: 2,014.96 MB (~2.01 GB)
- **Total Project Size (WITHOUT `.venv`)**: 607.36 MB
- **Virtual Environment (`.venv`)**: 1,407.17 MB
- **FFmpeg Portable Binaries (`tools/ffmpeg`)**: 164.91 MB
- **Model Checkpoints (`ckpts/`)**: 217.77 MB
- **SQLite Database (`database/mediaproof.db`)**: 0.04 MB (36,864 bytes)

---

## 11. CPU / RAM Benchmark Results

- **VideoSeal 2-second Video Round-Trip**: 12.70 seconds, Peak RAM ~911 MB (100% CPU execution on AMD64).
- **AudioSeal 3-second Audio Round-Trip**: 0.53 seconds.
- **Full Verification API Endpoint (`/api/verify`)**: 4.20 seconds.

---

## 12. Fresh Environment & Portability Audit

- Clean Python 3.11 virtual environment can install all requirements from `requirements.txt` and `requirements-ml.txt`.
- Pre-downloaded weights in `ckpts/` eliminate initial model download latency.

---

## 13. Security Audit

- **Subprocess Execution**: `media/ffmpeg.py` enforces `shell=False` for all FFmpeg / FFprobe command calls, eliminating shell injection vectors.
- **File Upload Sanitation**: `backend/main.py` sanitizes incoming filenames via `Path(file.filename).name` and stores transient uploads in isolated temporary directory `data/temp/`.

---

## 14. Data Privacy Audit

- 100% local operation. Zero external API calls, zero telemetry, zero analytics tracking.

---

## 15. Documentation Audit

- Documentation exists in `docs/` for Step 1, Step 2, Step 3, Step 4B, Step 4C, Step 4D, Step 5A, Step 5B, Step 5C, Step 6A, Step 6B, Step 6C, and Step 7A.

---

## 16. Audit Issues Found & Classification

1. **MEDIUM**: `/api/verify` returns HTTP `200 OK` with JSON `"status": "error"` for zero-byte / corrupt files rather than HTTP `400 Bad Request`.
2. **INFORMATIONAL**: PyTorch JIT script warnings printed during VideoSeal execution in Python 3.11 log outputs.

---

## 17. Portable Manifest Summary (`scratch/portable_manifest.json`)

Saved complete json structure to `scratch/portable_manifest.json`.
