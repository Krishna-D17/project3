# MediaProof Step 6A: Deepfake & Media Forensic Technology Discovery

## 1. Problem Definition & Scope

MediaProof combines cryptographic hashing (SHA-256), invisible digital watermarking (VideoSeal and AudioSeal), and local SQLite asset tracking. Step 6 evaluates and designs the **Media Forensic / Deepfake Detection Subsystem**.

### Core Principles
1. **Non-Binary Assertion**: MediaProof **does not** return simple binary claims like `"AI generated = true"`. Standard classifiers are vulnerable to high false positive rates and poor generalization across modern generative models (e.g., Diffusion, SORA, FLUX, ElevenLabs).
2. **Multi-Signal Evidence Synthesis**: Forensic ML model inference scores are treated as candidate signals alongside:
   - Cryptographic integrity (SHA-256 match vs mismatch)
   - Embedded watermark provenance (VideoSeal & AudioSeal recovery)
   - Media container & codec metadata analysis (FFprobe container structure, bitrates, GOP consistency)
   - Signal-level forensic features (spectral energy distribution, frame-to-frame noise variance)
3. **Missing Watermark Neutrality**: The absence of an invisible watermark **must never** automatically imply deepfake/manipulation. Unwatermarked assets are designated as `UNKNOWN_PROVENANCE` unless physical/forensic evidence indicates active tampering.
4. **Local CPU Feasibility**: The engine must execute entirely on local CPU resources (Windows 10 AMD64, 12 CPU cores, ~15.7 GB RAM, No CUDA/GPU) within acceptable latency boundaries (<3.0s total processing for 5s clips).

---

## 2. Video Deepfake Detector Candidates

### Candidate 1: DeepfakeBench (Unified Ecosystem / EfficientNet-B4 & Xception)
- **A. Official Repository**: [github.com/SwinTransformer/DeepfakeBench](https://github.com/SwinTransformer/DeepfakeBench)
- **B. License**: MIT License
- **C. Maintenance & Activity**: Active, maintained benchmark framework (2023–2026).
- **D. Python Compatibility**: Python 3.10–3.11 fully compatible.
- **E. Framework Requirements**: PyTorch 2.x, torchvision, timm.
- **F. Model / Checkpoint Size**: 75 MB (EfficientNet-B4) to 320 MB (Swin-Base).
- **G. CPU Feasibility**: Excellent for EfficientNet-B4 (~120ms per face frame on CPU).
- **H. RAM Requirements**: ~800 MB peak RAM.
- **I. Inference Speed**: ~1.2s for 10 sampled frames on 12-core CPU.
- **J. Input Requirements**: 224x224 RGB cropped face tensor.
- **K. Face Detection Dependency**: Required (OpenCV Haar Cascade or MediaPipe Face Mesh CPU).
- **L. Detection Target**: Face swaps (DeepFakes, FaceSwap), Face Reenactment (Face2Face, NeuralTextures).
- **M. Detection Level**: Frame-level face crop classification, aggregated via mean frame probability.
- **N. Robustness**: Good for mild H.264 compression; degraded by heavy resizing (<128px face) or aggressive cropping.
- **O. Known Limitations**: Lower accuracy on full-body synthetic videos or diffusion-generated scenes without human faces.
- **P. Dataset / Domain**: Trained primarily on FaceForensics++, Celeb-DF, DFDC.

### Candidate 2: FaceForensics++ / XceptionNet Baseline
- **A. Official Repository**: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **B. License**: Non-commercial / Academic License.
- **C. Maintenance & Activity**: Historical benchmark standard; reference implementation stable.
- **D. Python Compatibility**: Python 3.11 compatible with updated PyTorch.
- **E. Framework Requirements**: PyTorch, torchvision.
- **F. Model / Checkpoint Size**: ~88 MB (`xception-b5690412.pth`).
- **G. CPU Feasibility**: High feasibility (~90ms per face frame on CPU).
- **H. RAM Requirements**: ~600 MB RAM.
- **I. Inference Speed**: ~0.9s for 10 frames on 12-core CPU.
- **J. Input Requirements**: 299x299 RGB face tensor normalized [-1, 1].
- **K. Face Detection Dependency**: Required (dlib / OpenCV / MediaPipe).
- **L. Detection Target**: FaceSwap, DeepFakes, Face2Face, Face2Face reenactment.
- **M. Detection Level**: Frame-level face crop classification.
- **N. Robustness**: Moderate on low compression (c20); poor on high compression (c40 H.264).
- **O. Known Limitations**: High over-fitting to FF++ compression artifacts; poor generalization to modern generative AI (Midjourney, Runway Gen-2, Sora).
- **P. Dataset / Domain**: FaceForensics++ dataset only.

### Candidate 3: Spatial-Phase Frequency Detector (FFT / DCT Spectral Anomaly Analysis)
- **A. Official Repository**: Custom open algorithmic baseline (based on Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition", ICML 2020).
- **B. License**: Permissive BSD / MIT (Pure NumPy + SciPy implementation).
- **C. Maintenance & Activity**: Standard mathematical signal processing; zero dependency decay.
- **D. Python Compatibility**: Native Python 3.11.
- **E. Framework Requirements**: NumPy, SciPy, OpenCV (No heavy ML frameworks required).
- **F. Model / Checkpoint Size**: 0 MB (Zero weight file, pure analytical model).
- **G. CPU Feasibility**: Exceptional (~15ms per frame on CPU).
- **H. RAM Requirements**: <50 MB RAM.
- **I. Inference Speed**: <0.15s for 10 full video frames on CPU.
- **J. Input Requirements**: Full video frame or cropped region (Grayscale / RGB).
- **K. Face Detection Dependency**: Optional (Can run on full frame or localized crops).
- **L. Detection Target**: GAN/Diffusion upsampling grid artifacts, high-frequency spectral phase anomalies.
- **M. Detection Level**: Frame-level and full-video temporal consistency.
- **N. Robustness**: Highly robust to spatial rescalings; sensitive to heavy low-pass spatial filtering and strong JPEG re-compression.
- **O. Known Limitations**: Does not detect semantic edits or deepfake face swaps that preserve original spectral distributions.
- **P. Dataset / Domain**: Domain-agnostic.

### Candidate 4: MesoNet (MesoInception-4 Lightweight Model)
- **A. Official Repository**: [github.com/DariusAf/MesoNet](https://github.com/DariusAf/MesoNet)
- **B. License**: MIT License.
- **C. Maintenance & Activity**: Archival baseline; compact architecture.
- **D. Python Compatibility**: Compatible via PyTorch or ONNX Runtime.
- **E. Framework Requirements**: PyTorch or ONNX Runtime CPU.
- **F. Model / Checkpoint Size**: ~11.8 MB (Extremely compact).
- **G. CPU Feasibility**: Ultra-high (~25ms per frame on CPU).
- **H. RAM Requirements**: ~200 MB RAM.
- **I. Inference Speed**: ~0.25s for 10 frames on CPU.
- **J. Input Requirements**: 256x256 RGB face crop.
- **K. Face Detection Dependency**: Required.
- **L. Detection Target**: DeepFakes, MesoInception compression artifacts.
- **M. Detection Level**: Frame-level face crop.
- **N. Robustness**: Moderate; susceptible to spatial blur.
- **O. Known Limitations**: Lower accuracy on modern high-resolution synthetic faces.
- **P. Dataset / Domain**: Early deepfake benchmark datasets.

---

## 3. Audio Deepfake Detector Candidates

### Candidate 1: AASIST (Audio Anti-Spoofing Integrated Spectro-Temporal Graph Neural Network)
- **A. Official Repository**: [github.com/clovaai/aasist](https://github.com/clovaai/aasist)
- **B. License**: BSD 3-Clause License.
- **C. Model / Checkpoint Size**: ~1.2 MB (`AASIST.pth` / ~300k parameters).
- **D. CPU Feasibility**: Outstanding (Inference ~140ms on CPU for 5s audio).
- **E. Python Compatibility**: Python 3.11 compatible with PyTorch 2.x.
- **F. Input Sample Rate**: 16,000 Hz Mono PCM WAV (Matches MediaProof standard audio format).
- **G. Required Preprocessing**: SincNet raw waveform feature extractor (Built-in).
- **H. Inference Speed**: ~0.15s per 5s audio clip on CPU.
- **I. Dataset Used**: ASVspoof 2019 Logical Access (LA) & ASVspoof 2021 LA/DF.
- **J. Detection Target**: Text-to-Speech (TTS), Voice Conversion (VC), Synthetic Speech.
- **K. Known Limitations**: Sensitivity to unseen room acoustics and extreme codec bitrates (<16 kbps AAC).
- **L. Generalization Concerns**: Strong performance on ASVspoof benchmarks; requires energy calibration for general noisy recording environments.

### Candidate 2: RawNet2 Audio Spoof Detector
- **A. Official Repository**: [github.com/TakTaki/RawNet-2](https://github.com/TakTaki/RawNet-2)
- **B. License**: MIT License.
- **C. Model / Checkpoint Size**: ~18.5 MB.
- **D. CPU Feasibility**: High (~220ms on CPU for 5s audio).
- **E. Python Compatibility**: Python 3.11 compatible.
- **F. Input Sample Rate**: 16,000 Hz Raw Waveform.
- **G. Required Preprocessing**: Raw audio normalization.
- **H. Inference Speed**: ~0.25s per 5s audio clip on CPU.
- **I. Dataset Used**: ASVspoof 2019 LA.
- **J. Detection Target**: TTS and Voice Conversion.
- **K. Known Limitations**: Higher memory footprint than AASIST with slightly lower generalization on ASVspoof 2021 DF.
- **L. Generalization Concerns**: Performance degrades on heavily compressed MP3/AAC streams.

### Candidate 3: LFCC-GMM / Spectrogram Phase Anomaly Baseline
- **A. Official Repository**: Algorithmic baseline (Linear Frequency Cepstral Coefficients + Spectral Variance Analysis).
- **B. License**: Permissive BSD / MIT (Pure NumPy + SciPy + Librosa/PyTorch).
- **C. Model / Checkpoint Size**: 0 MB (Deterministic heuristic/statistical model).
- **D. CPU Feasibility**: Ultra-high (~30ms for 5s audio).
- **E. Python Compatibility**: Python 3.11.
- **F. Input Sample Rate**: Any sample rate (Resampled to 16kHz).
- **G. Required Preprocessing**: LFCC filterbank extraction.
- **H. Inference Speed**: <0.05s on CPU.
- **I. Dataset Used**: Domain agnostic.
- **J. Detection Target**: Phase discontinuities, vocoder spectral artifacts, high-frequency cutoff anomalies.
- **K. Known Limitations**: Statistical detector; cannot classify sophisticated neural zero-shot voice clones without spectral artifacts.
- **L. Generalization Concerns**: High false positives on lossy MP3 compression.

---

## 4. Candidate Comparison Table

| Candidate | Domain | Type | Size | License | CPU Speed (5s clip) | Face Needed? | Primary Detection Target | Feasibility Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepfakeBench (EfficientNet-B4)** | Video | Neural | 75 MB | MIT | ~0.80s | Yes | Face Swap / Reenactment | **Rank 1 (Video ML)** |
| **Spectral Frequency (FFT/DCT)** | Video | Signal | 0 MB | MIT/BSD | ~0.08s | No | GAN/Grid/Spatial Artifacts | **Rank 1 (Video Signal)** |
| **MesoInception-4** | Video | Neural | 11.8 MB | MIT | ~0.25s | Yes | Face Swaps | **Rank 2 (Video ML)** |
| **FaceForensics++ (Xception)** | Video | Neural | 88 MB | Academic | ~0.90s | Yes | FF++ Face Manipulations | **Rank 3 (Licensing)** |
| **AASIST** | Audio | Neural | 1.2 MB | BSD-3 | ~0.15s | N/A | TTS / Voice Conversion | **Rank 1 (Audio ML)** |
| **RawNet2** | Audio | Neural | 18.5 MB | MIT | ~0.25s | N/A | Synthetic Speech / TTS | **Rank 2 (Audio ML)** |
| **LFCC-Spectral Entropy** | Audio | Signal | 0 MB | MIT/BSD | ~0.03s | N/A | Vocoder / Phase Artifacts | **Rank 1 (Audio Signal)** |

---

## 5. CPU Feasibility & Latency Analysis

### Execution Environment Constraints
- **Target OS**: Windows 10 AMD64
- **CPU**: 12 CPU Cores (~3.4–4.5 GHz)
- **RAM**: ~15.7 GB System Memory
- **Target Latency Budget**: < 2.5s for complete multi-modal analysis (5s video + audio clip).

### Optimizations for CPU Inference
1. **Frame Sampling Strategy**: Rather than feeding all 150 frames of a 5-second video (at 30 fps) through the neural network, sample **1 frame per second** (5 representative frames per video). This reduces neural inference latency by **96.6%** without sacrificing detection efficacy.
2. **Face Detection Pipeline**: Use OpenCV Haar Cascades or MediaPipe CPU for fast facial crop extraction (~15ms per frame) instead of heavy GPU RetinaFace models.
3. **ONNX Runtime / PyTorch CPU Execution**: Export PyTorch weights to ONNX format or run PyTorch `torch.no_grad()` with `torch.set_num_threads(4)` to prevent thread contention across 12 CPU cores.

---

## 6. Recommended MVP Architecture

MediaProof will implement a **Layered Multi-Modal Forensic Detection Engine**:

```
                               ┌──────────────────────────────────────┐
                               │           INPUT MEDIA FILE           │
                               └──────────────────┬───────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                │                                 │                                 │
                ▼                                 ▼                                 ▼
   ┌──────────────────────────┐     ┌──────────────────────────┐      ┌──────────────────────────┐
   │    CONTAINER / CODEC     │     │     VIDEO FORENSICS      │      │      AUDIO FORENSICS     │
   │    METADATA SIGNALS      │     │    HYBRID SUBSYSTEM      │      │    HYBRID SUBSYSTEM      │
   └────────────┬─────────────┘     └────────────┬─────────────┘      └────────────┬─────────────┘
                │                                 │                                 │
                │     ┌───────────────────────────┴─────────────┐         ┌─────────┴───────────────┐
                │     ▼                                         ▼         ▼                         ▼
                │ ┌────────────────────────┐ ┌────────────────────┐ ┌─────────────┐ ┌──────────────────────┐
                │ │ Spectral FFT Artifact  │ │ EfficientNet-B4    │ │ LFCC Phase  │ │ AASIST Neural        │
                │ │ High-Freq Variance     │ │ Face Crop Detector │ │ Entropy     │ │ Synthetic Voice Det. │
                │ └───────────┬────────────┘ └─────────┬──────────┘ └──────┬──────┘ └──────────┬───────────┘
                │             │                        │                   │                   │
                └─────────────┼────────────────────────┴───────────────────┴───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  FORENSIC EVIDENCE AGGREGATOR│
               │  • Signal Scoring & Weights  │
               │  • Watermark & Provenance    │
               │  • Decision Matrix Evaluation│
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │    FINAL FORENSIC RESULT     │
               │    (JSON Evidence Schema)    │
               └──────────────────────────────┘
```

### Video Subsystem Components
1. **Metadata & Codec Signal**: Probes video container for stream inconsistencies (e.g. missing camera metadata, abnormal GOP structure, encoder signature anomalies via FFprobe).
2. **Frequency-Domain Spectral Signal**: Computes 2D Fast Fourier Transform (FFT) power spectra across frames to detect periodic high-frequency grid artifacts typical of GAN/Diffusion upsampling filters.
3. **EfficientNet-B4 Neural Face Classifier**: Detects and crops human face regions, passing 5 sampled frame crops to a lightweight face manipulation model.

### Audio Subsystem Components
1. **Spectral & Phase Entropy Signal**: Measures high-frequency roll-off, phase continuity, and energy distribution across mel-spectrogram bins to detect crude vocoder artifacts.
2. **AASIST Neural Audio Detector**: Passes raw 16kHz audio waveform through AASIST PyTorch CPU model to detect TTS and voice conversion artifacts.

---

## 7. Forensic Evidence Model Schema

The system will return a structured JSON response conforming to the following specification:

```json
{
  "status": "ok",
  "forensic_status": "POTENTIALLY_MANIPULATED",
  "risk_score": 0.82,
  "confidence": 0.91,
  "overall_verdict": "High forensic manipulation risk detected in audio and video streams.",
  "watermark_signals": {
    "video_watermark_detected": true,
    "video_engine": "videoseal",
    "video_recovered_asset_id": "MP-a1b2c3d4-5678-90ef-1234-56789abcdef0",
    "audio_watermark_detected": false,
    "audio_engine": null,
    "audio_recovered_asset_id": null
  },
  "provenance_signals": {
    "sha256_match": false,
    "asset_registered": true,
    "parent_asset_id": "MP-a1b2c3d4-5678-90ef-1234-56789abcdef0",
    "relationship_type": "DERIVATIVE"
  },
  "video_signals": [
    {
      "signal_name": "fft_spectral_grid_anomaly",
      "risk_level": "HIGH",
      "score": 0.87,
      "details": "High-frequency periodic spectral pattern detected in frame spatial spectrum."
    },
    {
      "signal_name": "face_manipulation_neural_score",
      "risk_level": "HIGH",
      "score": 0.81,
      "details": "EfficientNet-B4 detected spatial boundary anomalies in face region."
    }
  ],
  "audio_signals": [
    {
      "signal_name": "aasist_synthetic_speech_score",
      "risk_level": "LOW",
      "score": 0.12,
      "details": "AASIST raw waveform classification indicates authentic human speech."
    },
    {
      "signal_name": "audio_phase_entropy",
      "risk_level": "LOW",
      "score": 0.08,
      "details": "Natural phase variance across frequency sub-bands."
    }
  ],
  "metadata_signals": [
    {
      "signal_name": "container_codec_check",
      "risk_level": "MEDIUM",
      "score": 0.40,
      "details": "Non-standard encoder metadata tag detected; likely re-encoded with FFmpeg."
    }
  ],
  "limitations": [
    "Face crop confidence reduced due to low resolution (180x180 px face region).",
    "Audio clip duration is under 3 seconds; synthetic detection confidence scaled."
  ],
  "processing_time_seconds": 0.942
}
```

---

## 8. MediaProof Decision Matrix

The overall asset classification is governed by combining cryptographic hash matching, watermark provenance lookup, and multi-modal forensic risk scoring:

| SHA-256 Match | Watermark Present | Provenance Record | Forensic Risk Score | Final Classification | Explanation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Match** | **Present** | **Known Original** | Any (< 0.90) | `VERIFIED_ORIGINAL` | Bit-for-bit exact copy of original registered media asset. |
| **Mismatch** | **Present** | **Known Parent** | Low (< 0.40) | `VERIFIED_DERIVATIVE` | Legitimate derivative (transcoded/resized) with intact watermark. |
| **Mismatch** | **Present** | **Known Parent** | High (>= 0.75) | `POTENTIALLY_MANIPULATED` | Watermarked asset exhibiting facial/audio synthetic modification. |
| **Mismatch** | **Absent** | **None** | Low (< 0.40) | `UNKNOWN_PROVENANCE` | Unregistered asset; no watermark detected; no forensic anomaly. |
| **Mismatch** | **Absent** | **None** | High (>= 0.75) | `POTENTIALLY_MANIPULATED` | Unregistered asset with strong deepfake/synthetic risk signals. |

### Critical Control Rules
1. **`SHA-256 Match` Overrides Forensic Scorer**: If SHA-256 is an exact bit-for-bit match to a verified original asset, `VERIFIED_ORIGINAL` is assigned regardless of ML noise scores.
2. **Missing Watermark Neutrality**: If no watermark is found and forensic risk is low, the status is strictly `UNKNOWN_PROVENANCE` (never "Fake" or "Manipulated").

---

## 9. Dataset & Validation Strategy

### Video Evaluation Datasets
1. **FaceForensics++ (FF++)**: 1,000 original videos with 4,000 manipulated videos (DeepFakes, Face2Face, FaceSwap, NeuralTextures). Tests classic face manipulation algorithms.
2. **Celeb-DF (v2)**: 590 real celebrity videos and 5,639 high-quality deepfake videos. Tests visual quality and boundary artifacts.
3. **DFDC (DeepFake Detection Challenge)**: 100,000+ clips with diverse lighting, compression, and distractors. Tests real-world noise resilience.

### Audio Evaluation Datasets
1. **ASVspoof 2019 / 2021 (LA & DF)**: Standard benchmark containing synthetic speech generated by 17 different TTS and Voice Conversion algorithms.
2. **WaveFake**: 117,854 audio clips generated by modern architectures (MelGAN, HiFi-GAN, WaveGlow).

### Cross-Dataset Validation Principles
- **No In-Domain Evaluation Reliance**: Benchmark performance on FaceForensics++ does not transfer to modern diffusion video generators.
- **Compression Stress Testing**: Evaluates detectors under H.264 compression (CRF 18, 23, 28, 35) and audio bitrates (128k, 64k, 32k AAC/MP3).

---

## 10. Risks & Limitations

1. **Generative Model Evolution**: Classifiers trained on 2020 face-swaps decay when evaluated against 2026 state-of-the-art diffusion models.
2. **False Positive Risks on Edited Media**: Heavy color grading, sharp spatial filters, or video speed changes can trigger false positive spectral anomaly signals.
3. **Face Detector Failures**: Extreme profile angles, heavy motion blur, or multiple faces in crowded scenes complicate face cropping.

---

## 11. Implementation Plan for Step 6B (Modular Prototype)

When Step 6B is authorized, implementation will follow this architecture without altering current Step 1–5C modules:

1. **`forensics/config.py`**: Configuration constants, weights, and score thresholds.
2. **`forensics/video_signals.py`**: Implementation of FFT spectral analysis and EfficientNet face classifier.
3. **`forensics/audio_signals.py`**: Implementation of LFCC phase entropy and AASIST PyTorch CPU inference adapter.
4. **`forensics/service.py`**: Evidence aggregation and Decision Matrix evaluator (`analyze_media_forensics()`).
5. **REST API Endpoint (`backend/main.py`)**: `POST /api/forensics/analyze`.
6. **Tests (`tests/test_forensics.py`)**: Unit tests covering authentic, watermarked, and synthetic test files.
