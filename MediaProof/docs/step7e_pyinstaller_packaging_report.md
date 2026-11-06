# Step 7E — Production PyInstaller Packaging Report for MediaProof

## 1. Executive Summary

MediaProof has been successfully packaged into a reliable, standalone Windows x64 portable application using PyInstaller (`onedir` mode).

All core application functionality—including invisible neural watermarking (Meta VideoSeal & Meta AudioSeal), forensic deepfake detection (DeepfakeBench Xception + OpenCV YuNet face crop), SQLite provenance registration, cryptographic SHA-256 integrity verification, FFmpeg media processing, and the complete FastAPI backend with static frontend—is fully operational inside the packaged binary.

Crucially, the previous PyInstaller failure:
`operator torchvision::nms does not exist`
has been **100% resolved** by forcing explicit collection of PyTorch Vision C++ extension binaries (`_C_stable.pyd` and `image_stable.pyd`), setting `upx=False`, and bundling `decord.dll` into `_internal/decord/`.

No fallbacks to custom watermarking occurred. No models were downloaded at runtime. No external network connections are required.

---

## 2. Environment & Specification Summary

| Attribute | Value / Version |
| --- | --- |
| **PyInstaller Version** | `6.11.1` |
| **Python Version** | `3.11.9` (Windows x64) |
| **PyTorch Version** | `2.14.0+cpu` |
| **Torchvision Version** | `0.29.0+cpu` |
| **VideoSeal Package** | `videoseal==1.0` (Commit `870ca7f`) |
| **AudioSeal Package** | `audioseal==0.1.2` (Commit `6b30f81`) |
| **OpenCV Version** | `4.10.0` |
| **Packaging Spec** | `MediaProof.spec` |
| **Target Architecture** | Windows x64 |
| **PyInstaller Mode** | `onedir` (`dist/MediaProof/`) |

---

## 3. Investigation & Resolution of Previous VideoSeal Failure

### Original Failure
In previous PyInstaller attempts, the executable launched but calling Meta VideoSeal failed at runtime with:
`operator torchvision::nms does not exist`

### Root Cause Analysis
1. In `torchvision==0.29.0+cpu`, PyTorch C++ operator extension binaries are named `_C_stable.pyd` and `image_stable.pyd` (rather than standard `_C.pyd`).
2. PyInstaller's standard `hook-torchvision.py` failed to identify `_C_stable.pyd` as a required native extension module.
3. Consequently, PyInstaller omitted `_C_stable.pyd` from `dist/MediaProof/_internal/torchvision/`.
4. When `torchvision.ops.nms` was called during VideoSeal inference, PyTorch failed to find the C++ extension registration for `torchvision::nms`.
5. Additionally, `videoseal` relies on `decord` for raw frame access, which loads `decord.dll` relative to its module root (`_internal/decord/decord.dll`). Standard PyInstaller binary reclassification placed `decord.dll` directly under `_internal/`, causing `decord` initialization failures.

### Permanent Technical Fix
1. **Explicit Native Extension Collection**: Updated `MediaProof.spec` with a recursive glob scan over `site-packages/torchvision` to explicitly collect all `.pyd`, `.dll`, and `.so` files into `binaries`, preserving relative package paths (`_internal/torchvision/_C_stable.pyd`).
2. **Explicit Decord Asset Collection**: Added an explicit file collection loop for `decord` so that `decord.dll` and its bundled FFmpeg dynamic libraries reside inside `_internal/decord/`.
3. **UPX Compression Disabled**: Set `upx=False` in `MediaProof.spec` to prevent corruption of PyTorch C++ operator DLLs and OpenCV native binaries.
4. **Collect-All Hooks**: Applied `collect_all()` for `torch`, `torchvision`, `torchaudio`, `videoseal`, `audioseal`, `timm`, `omegaconf`, `cv2`, `decord`, `einops`, `onnxruntime`, and `julius`.

---

## 4. Architecture & Path Resolution

### Single Resource Path Abstraction (`backend/paths.py`)
Centralized path resolution guarantees identical resource access in development and PyInstaller frozen modes:

* **Bundle Directory (`get_bundle_dir()`)**: Points to `sys._MEIPASS` when frozen, or project root when running in Python virtualenv. Used for read-only static resources (models, frontend assets, FFmpeg binaries).
* **Portable Directory (`get_portable_root()`)**: Points to `Path(sys.executable).parent` when frozen (`dist/MediaProof/`). Used for writable runtime directories (`database/mediaproof.db`, `data/protected/`, `data/temp/`).

### Writable Data Strategy
SQLite database and output files are separated from read-only application resources:
* **Read-Only**: `_internal/models/`, `_internal/tools/ffmpeg/`, `_internal/frontend/`.
* **Writable**: `dist/MediaProof/database/mediaproof.db`, `dist/MediaProof/data/protected/`, `dist/MediaProof/data/temp/`.

This ensures database records and protected files survive application restarts and upgrades.

---

## 5. Bundled Assets & Inventory

| Component | Bundled Path | Footprint / Size | Details / SHA-256 |
| --- | --- | --- | --- |
| **MediaProof EXE** | `dist/MediaProof/MediaProof.exe` | 77.73 MB | PyInstaller bootloader + app archive |
| **Total Package** | `dist/MediaProof/` | 1,695.19 MB | Complete portable bundle |
| **FFmpeg & FFprobe** | `_internal/tools/ffmpeg/` | 164.91 MB | Portable static binaries |
| **VideoSeal Model** | `_internal/models/video/videoseal_256bit.pth` | 212.00 MB | SHA-256: `3d2ff2523d2a89e3532c6dfdcf693098799326e9b3e74c185e815e9baa8340a3` |
| **Xception Model** | `_internal/models/forensics/video_deepfake_xception.pth` | 87.76 MB | SHA-256: `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47` |
| **YuNet Model** | `_internal/models/forensics/face_detector/` | 0.34 MB | SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| **PyTorch Footprint** | `_internal/torch/` & `torchvision/` | ~980.00 MB | CPU inference runtime + C++ operator DLLs |

---

## 6. Functional Verification Matrix

Automated validation script `tools/test_packaged_app.py` executed against the packaged application (`dist/MediaProof/MediaProof.exe`):

| Test | Development Mode | Packaged EXE Mode | Result |
| --- | --- | --- | --- |
| **Health API (`/api/health`)** | `200 OK` | `200 OK` | **PASS** |
| **FFmpeg Discovery** | `tools/ffmpeg/` | `_internal/tools/ffmpeg/` | **PASS** |
| **SQLite Provenance Registration** | Persistent | Persistent | **PASS** |
| **Custom Video Watermark** | OK | OK | **PASS** |
| **Meta VideoSeal Embedding** | OK | OK | **PASS** |
| **Meta VideoSeal Detection** | Token Recovered | Token Recovered | **PASS** |
| **Custom Audio Watermark** | OK | OK | **PASS** |
| **Meta AudioSeal Embed & Detect** | Token Recovered | Token Recovered | **PASS** |
| **OpenCV YuNet Face Detector** | 7/7 Faces Cropped | 7/7 Faces Cropped | **PASS** |
| **DeepfakeBench Xception Model** | P(Fake) evaluated | P(Fake) evaluated | **PASS** |
| **Full Verification API (`/api/verify`)** | Unified Evidence | Unified Evidence | **PASS** |
| **Restart Persistence** | Persistent | Persistent | **PASS** |

---

## 7. Portability, Offline & Security Audit

1. **Copy-to-New-Directory Verification**:
   - `dist/MediaProof` was copied to `dist/MediaProof_PortabilityTest/`.
   - `MediaProof_PortabilityTest/MediaProof.exe` was executed without development virtual environment or source directory dependencies.
   - All 12 automated smoke tests passed with **100% success**.
   - Result: **PASS** (Zero absolute path dependencies).

2. **Offline Mode Verification**:
   - Application executed completely offline.
   - All models loaded directly from packaged `_internal/models/` and `_internal/ckpts/`.
   - Zero HTTP requests or model downloads occurred during execution.
   - Result: **PASS** (100% local CPU offline runtime).

3. **Security Audit**:
   - `shell=False` enforced across all FFmpeg subprocess executions.
   - Database stored in writable portable directory, not inside PyInstaller temp directory.
   - Input filenames sanitized; temporary work files generated with random UUID prefixes.
   - Result: **PASS**.

---

## 8. Repeatable Packaging Test Script

The automated test script `tools/test_packaged_app.py` provides repeatable verification of packaged builds:

```bash
.venv\Scripts\python.exe tools/test_packaged_app.py [optional_exe_path]
```

---

## 9. Final Status

```text
PACKAGING PASS
```
