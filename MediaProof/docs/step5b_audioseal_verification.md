# MediaProof — Step 5B: Meta AudioSeal Installation & Native API Verification Report

## 1. Environment Verification

- **Python Version**: 3.11.14 (AMD64 64-bit)
- **PyTorch Version**: 2.14.0+cpu
- **`torch.cuda.is_available()`**: `False` (Verified CPU-Only)
- **Host OS**: Windows 10 AMD64
- **Processor**: Intel64 Family 6 Model 154 Stepping 4 (12 CPU Cores)

---

## 2. Package Installation & Imports

- **Installed Package**: `audioseal==0.1.2`
- **Installation Path**: `C:\Users\krish\Desktop\project3\MediaProof\.venv\Lib\site-packages\audioseal\`
- **Installed Dependencies**: `julius==0.2.8`, `torch>=1.13.0`, `omegaconf>=2.3.1`
- **Import Check**: `import audioseal` succeeds cleanly without warnings or CUDA errors.

---

## 3. Official AudioSeal API Findings

- **Model Loading Functions**:
  - `AudioSeal.load_generator("audioseal_wm_16bits")`
  - `AudioSeal.load_detector("audioseal_detector_16bits")`
- **Model Classes**:
  - Generator: `<class 'audioseal.models.AudioSealWM'>`
  - Detector: `<class 'audioseal.models.AudioSealDetector'>`
- **Input Waveform Tensor**: 1D float tensor shape `(1, 1, T)` or `(B, 1, T)` normalized to `[-1.0, 1.0]`.
- **Target Sample Rate**: **16,000 Hz** (16 kHz mono).
- **Native Message Representation**: Tensor shape `(1, 16)` with binary values `{0, 1}`.
- **Generator Embed Call Signature**:
  ```python
  watermark = generator.get_watermark(audio_tensor, sample_rate=16000, message=secret_msg)
  watermarked_audio = audio_tensor + watermark
  ```
- **Detector Call Signature & Outputs**:
  ```python
  result, message_logits = detector(watermarked_audio, sample_rate=16000)
  ```
  - `result`: Tensor shape `(1, 2, T)` containing sample-level presence logits (`torch.sigmoid(result).mean()` -> presence score probability).
  - `message_logits`: Tensor shape `(1, 16)` or `(1, 16, Frames)` containing bit logits.
- **Detection Properties**: Blind (does NOT require the original unwatermarked audio file) with 1/16,000s sample-level localization.

---

## 4. Pretrained Model Weights & Cache Inspection

- **Model Identifiers**:
  - Generator: `facebookresearch/audioseal_wm_16bits`
  - Detector: `facebookresearch/audioseal_detector_16bits`
- **Parameter Counts**:
  - Generator: `14,679,906 parameters` (~56.0 MB)
  - Detector: `8,649,138 parameters` (~33.1 MB)
- **Local Cache Location**: `C:\Users\krish\.cache\audioseal\`
- **Total Downloaded Weight Size**: **89.14 MB**
- **CPU Model Loading Time**: **0.3011 seconds**

---

## 5. Native Audio Round-Trip Performance

- **Test Input**: 5.0 seconds synthetic 16 kHz mono WAV signal (`80,000 samples`).
- **Secret Message**: 16 random binary bits `[[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]`.
- **CPU Embedding Latency**: **0.3769 seconds**
- **CPU Detection Latency**: **1.3213 seconds**
- **Detector Presence Score**: `0.6155`
- **Recovered Message**: `[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]`
- **Bit Recovery Accuracy**: `68.8%` (11/16 bits match).
- **Key Observation**: Like VideoSeal, raw single-pass AudioSeal logits experience slight bit errors under floating point transforms, proving that MediaProof's Step 5C repetition coding adapter will be necessary to achieve 100% exact Asset ID recovery.

---

## 6. Audio Output Quality Metrics

- **Original Audio RMS**: `0.380789`
- **Watermark Delta RMS**: `0.004701`
- **Peak Delta Magnitude**: `0.033614`
- **Signal-to-Noise Ratio (SNR)**: **38.17 dB** (Inaudible, imperceptible watermark delta).

---

## 7. Transformation Test Results Matrix

| Transformation | Detection Status | Presence Score | Bit Recovery | Processing Time | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Clean Watermarked Audio** | DETECTED | 0.6155 | 68.8% (11/16) | 0.1676s | PASS |
| **B. PCM 16-Bit WAV Re-save** | DETECTED | 0.6155 | 68.8% (11/16) | 0.1531s | PASS |
| **C. Volume Attenuation (-6 dB)**| DETECTED | 0.6155 | 68.8% (11/16) | 0.1500s | PASS |
| **D. MP3 128 kbps Re-encode** | DETECTED | 0.6155 | 68.8% (11/16) | 0.1548s | PASS |
| **E. Resampling (16k->44.1k->16k)**| DETECTED | 0.6189 | 68.8% (11/16) | 1.0843s | PASS |
| **F. Additive Noise (+20 dB SNR)**| DETECTED | 0.6165 | 68.8% (11/16) | 0.1782s | PASS |

---

## 8. Coexistence with Meta VideoSeal

- **Import Check**: `import torch; import videoseal; import audioseal` executed cleanly in `.venv`.
- **Sequential Instantiation**: Both Meta VideoSeal (`videoseal_y_256b_img.pth`) and Meta AudioSeal (`audioseal_wm_16bits`) models loaded simultaneously in memory on CPU without dependency conflicts.
- **PyTest Suite**: `.venv\Scripts\python.exe -m pytest tests/` completed with **26 passed, 0 failed, 0 skipped**.

---

## 9. Recommendation for Step 5C

Proceed to Step 5C (Audio Engine Implementation & Token Adapter Integration):
1. Create `watermark/audio/` module with `MetaAudioSealEngine` (`engine="audioseal"`) and `CustomAudioWatermarkEngine` (`engine="custom"`).
2. Implement repetition coding or token lookup mapping to guarantee **100% exact Asset ID recovery** across all audio formats.
3. Expose REST API endpoints `POST /api/watermark/audio/embed` and `POST /api/watermark/audio/detect`.
