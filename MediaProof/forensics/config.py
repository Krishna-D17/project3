from pathlib import Path

# Forensic Status Constants
STATUS_LOW_RISK = "LOW_RISK"
STATUS_ELEVATED_RISK = "ELEVATED_RISK"
STATUS_HIGHER_RISK = "HIGHER_RISK"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# Signal Risk Level Labels
RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"
RISK_LEVEL_UNKNOWN = "UNKNOWN"

# Video Processing Constants
VIDEO_TARGET_FPS_SAMPLING = 1.0  # 1 frame sampled per second of video
MAX_VIDEO_SAMPLED_FRAMES = 10     # Cap at max 10 frames per video for CPU speed

# Thresholds (Explicitly Marked as Heuristic Signals)
# FFT High-Frequency Energy Ratio threshold (GAN/Diffusion spatial upsampling grid artifacts)
HEURISTIC_FFT_HIGH_FREQ_RATIO_THRESHOLD = 0.35

# Frame-to-frame high-frequency variation threshold (Temporal noise inconsistency)
HEURISTIC_TEMPORAL_NOISE_VAR_THRESHOLD = 0.45

# Blurriness / Variance of Laplacian threshold (Low spatial sharpness indicator)
HEURISTIC_BLURRINESS_LAPLACIAN_THRESHOLD = 100.0

# Audio Processing Constants
AUDIO_TARGET_SAMPLE_RATE = 16000

# Audio Heuristic Thresholds
HEURISTIC_AUDIO_CLIPPING_RATIO_THRESHOLD = 0.05   # >5% clipped samples
HEURISTIC_AUDIO_SILENCE_RATIO_THRESHOLD = 0.60    # >60% silence
HEURISTIC_AUDIO_SPECTRAL_ROLLOFF_THRESHOLD = 7000.0 # High frequency cutoff
HEURISTIC_AUDIO_HIGH_FREQ_PHASE_ENTROPY_THRESHOLD = 2.5 # High phase variance

from backend.paths import get_face_detector_models_dir, get_forensics_models_dir

MODEL_DIR = get_forensics_models_dir()
VIDEO_ML_CHECKPOINT_PATH = MODEL_DIR / "video_deepfake_efficientnet.pth"
AUDIO_ML_CHECKPOINT_PATH = MODEL_DIR / "aasist.pth"
YUNET_FACE_DETECTOR_PATH = get_face_detector_models_dir() / "face_detection_yunet_2023mar.onnx"

# Signal Categories
CATEGORY_GENERIC_TECHNICAL = "GENERIC_TECHNICAL"
CATEGORY_METADATA = "METADATA"
CATEGORY_TRANSFORMATION_CONSISTENCY = "TRANSFORMATION_CONSISTENCY"
CATEGORY_MANIPULATION_SPECIFIC = "MANIPULATION_SPECIFIC"

# Generic Forensic Status Labels
STATUS_LOW_GENERIC = "LOW_GENERIC_ANOMALY"
STATUS_MEDIUM_GENERIC = "MEDIUM_GENERIC_ANOMALY"
STATUS_HIGH_GENERIC = "HIGH_GENERIC_ANOMALY"

# Manipulation-Specific Forensic Status Labels
STATUS_MANIPULATION_NO_DETECTOR = "NO_VALIDATED_DETECTOR"
STATUS_MANIPULATION_SUSPECTED = "MANIPULATION_SUSPECTED"
STATUS_MANIPULATION_NO_ANOMALY = "NO_ANOMALY_DETECTED"

# Transformation Consistency Status Labels
CONSISTENCY_CONSISTENT = "CONSISTENT"
CONSISTENCY_DEVIATION = "DEVIATION"
CONSISTENCY_NOT_AVAILABLE = "NOT_AVAILABLE"

