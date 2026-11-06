# MediaProof

A local, portable, CPU-only media provenance and invisible-watermarking application for Windows.

---

## 1. Dual Video Watermarking Architecture

MediaProof features a modular dual-engine video watermarking architecture that separates deterministic spatial CPU processing from learned neural watermarking:

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
                   (watermark/video/custom_engine.py) (watermark/video/videoseal_engine.py)
                              │                               │
                     Deterministic Spatial             Learned PyTorch Neural
                     Green-Channel Baseline             Model (CPU Inference)
```

> [!IMPORTANT]
> **Technical Authenticity Notice**:
> - **MediaProof Custom Spatial Engine (`engine="custom"`)**: A fast, lightweight, deterministic spatial-domain pixel modulation baseline developed for MediaProof. **The custom engine is NOT VideoSeal.**
> - **Meta VideoSeal Engine (`engine="videoseal"`)**: An optional neural watermarking engine powered by Meta/Facebook Research's `videoseal` PyTorch model operating in CPU mode.

---

## 2. Engine Comparison

| Feature | MediaProof Custom Spatial Baseline | Meta VideoSeal Neural Engine |
| :--- | :--- | :--- |
| **Engine Identifier** | `engine="custom"` | `engine="videoseal"` |
| **Display Name** | `MediaProof Custom Spatial Video Watermark` | `Meta VideoSeal Neural Watermark Engine` |
| **Underlying Technology** | Vertical Green-channel spatial pair modulation | Learned PyTorch Convolutional Neural Network |
| **Model Weights Required** | None | `models/video/videoseal_256bit.pth` (~150 MB) |
| **CPU Processing Latency** | **< 1.0 second** per clip | ~15–30 seconds per 5s clip |
| **Tested H.264 Resilience** | **PASS** (100% exact Asset ID recovery in test fixture) | **PASS** (Requires official weights) |
| **Dependencies** | Python Standard Library + FFmpeg | PyTorch CPU + `videoseal` package |

---

## 3. 256-Bit Binary Payload Specification

Both engines consume the standard MediaProof 256-bit (32 bytes) binary payload:

| Byte Range | Field Name | Bits | Purpose |
| :--- | :--- | :--- | :--- |
| `Bytes 0..1` | **Magic Header** | 16 bits | `0x4D50` ("MP" in ASCII) prefix identifier |
| `Bytes 2..17` | **Asset UUID** | 128 bits | Raw binary 16-byte UUID v4 representation |
| `Bytes 18..21` | **Integrity Hash** | 32 bits | First 4 bytes of asset SHA-256 checksum |
| `Bytes 22..23` | **Flags / Reserved** | 16 bits | Reserved for version/derivative flags |
| `Bytes 24..31` | **Padding / Zero** | 64 bits | Zero-padding / error-correction margin |

---

## 4. API Endpoints & Engine Selection

Start backend server:

```cmd
start.bat
```

### Embed Video Watermark: `POST /api/watermark/video/embed`
Accepts `file` (multipart upload), `asset_id` (form field), and optional `engine` (`"custom"` or `"videoseal"`). Default is `"custom"`.

#### Example Request:
```cmd
curl -X POST "http://127.0.0.1:8000/api/watermark/video/embed" \
  -F "file=@sample.mp4" \
  -F "asset_id=MP-f678b91f-0f2e-46c2-a40b-c266aedfd8dd" \
  -F "engine=custom"
```

#### Example Response:
```json
{
  "status": "ok",
  "engine": "custom",
  "output_path": "C:\\Users\\krish\\Desktop\\project3\\MediaProof\\data\\protected\\wm_sample.mp4",
  "asset_id": "MP-f678b91f-0f2e-46c2-a40b-c266aedfd8dd",
  "size_bytes": 12847,
  "has_watermark": true,
  "processing_time_seconds": 0.691,
  "details": {
    "engine_display_name": "MediaProof Custom Spatial Video Watermark"
  }
}
```

### Detect Video Watermark: `POST /api/watermark/video/detect`
Accepts `file` (multipart upload) and optional `engine` (`"custom"` or `"videoseal"`).

#### Example Response:
```json
{
  "status": "ok",
  "engine": "custom",
  "watermark_detected": true,
  "confidence": 1.0,
  "recovered_asset_id": "MP-f678b91f-0f2e-46c2-a40b-c266aedfd8dd",
  "provenance_record": {
    "asset_id": "MP-f678b91f-0f2e-46c2-a40b-c266aedfd8dd",
    "sha256": "7fa4d1baeada240484e1a5baf5aa0984c16058c31c0b6cf9e98847609d06053f",
    "media_type": "video"
  },
  "processing_time_seconds": 0.4644,
  "message": "Watermark payload successfully detected and decoded."
}
```

---

## 5. Model Weight Management & Setup

To download official Meta VideoSeal weights into `models/video/videoseal_256bit.pth`:

```cmd
.venv\Scripts\python.exe scripts/download_videoseal_models.py
```

If `engine="videoseal"` is called when model weights are missing, the server returns an explicit, actionable HTTP 400/500 error directing the user to run the downloader script rather than silently falling back.

---

## 6. Acceptance & Test Suite

Run full automated test suite:

```cmd
.venv\Scripts\python.exe -m pytest tests/
```

Output: `53 passed, 0 failed`

---

## 7. Portable Windows Build

MediaProof is available as a standalone portable Windows application requiring no Python, virtual environment, or FFmpeg installation:

1. Copy or extract the `dist/MediaProof_Portable/` directory to any local folder (such as `C:\MediaProof_Portable\`).
2. Double-click `MediaProof.exe` or `start.bat`.
3. Your default web browser will automatically open to `http://127.0.0.1:8000/`.
4. The application operates 100% locally and offline on CPU with pre-bundled PyTorch models, FFmpeg binaries, SQLite database, and web UI.

