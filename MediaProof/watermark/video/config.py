from backend.paths import get_video_models_dir

MODELS_VIDEO_DIR = get_video_models_dir()
MODEL_CHECKPOINT_PATH = MODELS_VIDEO_DIR / "videoseal_256bit.pth"

DEFAULT_ENGINE = "custom"
DEVICE = "cpu"
PAYLOAD_BIT_LENGTH = 256
MAGIC_HEADER_BYTES = b"MP"  # 0x4D50 (16 bits)
DETECTION_CONFIDENCE_THRESHOLD = 0.70  # Used by custom engine spatial correlation
VIDEOSEAL_DETECTION_CONFIDENCE_THRESHOLD = 0.50  # Neural sigmoid presence threshold (0.50 represents logit > 0)
