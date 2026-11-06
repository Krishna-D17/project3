from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Audio processing constants
AUDIO_SAMPLE_RATE = 16000  # AudioSeal expected sample rate (16 kHz mono)
AUDIO_CHANNELS = 1
AUDIO_BIT_DEPTH = 16

# Watermark parameters
PAYLOAD_TOKEN_BITS = 16  # 16-bit token (4 hex chars)
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Device & Default engine
DEVICE = "cpu"
DEFAULT_AUDIO_ENGINE = "custom"
