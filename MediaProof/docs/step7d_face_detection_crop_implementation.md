# Step 7D — CPU-Compatible Face Detection & Face Cropping Implementation Report

## Overview
Step 7D integrates an offline, CPU-compatible **YuNet face detector** into the **MediaProof DeepfakeBench Xception** video deepfake detection pipeline. Face-forgery classifiers trained on dataset crops (such as FaceForensics++ c23) depend on localized facial inputs. When full frames are center-cropped, unmanipulated background regions dominate the feature representation, leading to false negatives on facial swaps.

---

## 1. Face Detector Specification
- **Detector Name**: OpenCV YuNet Face Detector (`FaceDetectorYN`)
- **Package & Version**: `opencv-python 5.0.0.93` (`cv2.FaceDetectorYN`)
- **Model File**: [`models/forensics/face_detector/face_detection_yunet_2023mar.onnx`](file:///c:/Users/krish/Desktop/project3/MediaProof/models/forensics/face_detector/face_detection_yunet_2023mar.onnx)
- **Model Size**: 232,589 bytes (~232 KB)
- **SHA-256 Checksum**: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- **Execution Target**: 100% CPU, local execution, 0 runtime network calls.

---

## 2. Preprocessing & Cropping Strategy
1. **Sampling**: 1 FPS up to a maximum of 10 sampled frames per video.
2. **Face Location**: YuNet detects bounding boxes $(x, y, w, h)$ at score threshold $0.60$, NMS threshold $0.30$.
3. **Multi-Face Rule**: Deterministically selects the face with the largest area ($w \times h$).
4. **Margin Expansion**: Centers on face center $(c_x, c_y)$ and expands bounding square side length by **$1.3\times$** (`margin_scale=1.3`) to preserve outer facial contours.
5. **Fallback Policy**: If no face is detected or YuNet is unavailable, gracefully falls back to center-square cropping. No frames are discarded.
6. **Resize & Normalization**: Resized to $299 \times 299$ RGB tensor, scaled to $[0, 1]$, normalized via $(x - 0.5) / 0.5 \rightarrow [-1.0, 1.0]$.

---

## 3. Empirical Live Comparison on 7-Second Videos

| Metric | Legitimate Video (`wm_000_original_7s.mp4`) | Known Adversarial Deepfake (`wm_000_virat_7s_from_protected.mp4`) |
| :--- | :--- | :--- |
| **Sampled Frames** | 7 | 7 |
| **Face-Detected Frames** | 7 (100%) | 7 (100%) |
| **Fallback Frames** | 0 (0%) | 0 (0%) |
| **Crop Method** | `face_crop` | `face_crop` |
| **Center-Crop Mean $P(\text{Fake})$** | 0.3991 | 0.4294 (Diff: $+0.0303$) |
| **Face-Cropped Mean $P(\text{Fake})$** | **0.4823** | **0.9226** (Diff: **$+0.4403$**) |
| **Min / Max $P(\text{Fake})$** | 0.2058 / 0.7547 | 0.7764 / 0.9728 |
| **Per-Frame $P(\text{Fake})$** | `[0.2324, 0.3435, 0.6580, 0.6206, 0.5613, 0.2058, 0.7547]` | `[0.7764, 0.9668, 0.9234, 0.9405, 0.9274, 0.9728, 0.9512]` |
| **Detector Status ($T=0.50$)** | `NO_MANIPULATION_DETECTED` | `MANIPULATION_SUSPECTED` |
| **Total Inference Time** | 14.68 s | 15.26 s |

---

## 4. Key Findings & Baseline Separation
- **Center-Crop Baseline**: Separation between Legitimate and Deepfake was only **$+0.0303$** ($0.3991$ vs $0.4294$), leaving both classified as `NO_MANIPULATION_DETECTED`.
- **Face-Cropped Result**: Separation expanded dramatically to **$+0.4403$** ($0.4823$ vs $0.9226$).
- **Classification Result**: The Deepfake video correctly triggered `MANIPULATION_SUSPECTED` ($P=0.9226 \ge 0.50$), with every individual frame showing $>77\%$ to $>97\%$ fake probability.
- **Legitimate Video Result**: Mean probability remained below threshold ($P=0.4823 < 0.50 \rightarrow$ `NO_MANIPULATION_DETECTED`).

---

## 5. Security & Portability Compliance
- **Zero Runtime Downloads**: The YuNet ONNX model is stored locally in `models/forensics/face_detector/`.
- **Offline & Standalone**: No external API or cloud face recognition service is invoked.
- **Privacy Enforcement**: Face detection is used solely for bounding box coordinate extraction. No identity recognition, facial feature vector logging, or biometric tracking is performed.

---

## 6. Evaluation Disclaimer
> [!IMPORTANT]
> A two-video sample benchmark demonstrates significant empirical improvement for face-focused Xception deepfake detection, but **two sample videos are strictly insufficient to prove generalized deepfake detection accuracy** across diverse compressions, lighting conditions, and out-of-distribution generators. ML signals remain probabilistic forensic evidence, completely separate from cryptographic provenance and watermark verification.
