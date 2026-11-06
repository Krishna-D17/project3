from typing import Optional

from watermark.video.base import BaseVideoWatermarkEngine
from watermark.video.config import DEFAULT_ENGINE
from watermark.video.custom_engine import CustomVideoWatermarkEngine
from watermark.video.videoseal_engine import MetaVideoSealEngine


def get_video_watermark_engine(engine_type: Optional[str] = None) -> BaseVideoWatermarkEngine:
    """Factory function to resolve and instantiate a video watermarking engine.
    
    Supported engine types:
    - 'custom': MediaProof Custom Spatial Video Watermark Engine (default)
    - 'videoseal': Meta VideoSeal Neural Watermark Engine
    """
    selected_type = (engine_type or DEFAULT_ENGINE).strip().lower()

    if selected_type == "custom":
        return CustomVideoWatermarkEngine()

    if selected_type == "videoseal":
        return MetaVideoSealEngine()

    raise ValueError(
        f"Unsupported video watermarking engine '{engine_type}'. "
        "Supported engine types: 'custom', 'videoseal'."
    )
