"""Backward compatibility module re-exporting CustomVideoWatermarkEngine as VideoSealEngine."""

from watermark.video.custom_engine import CustomVideoWatermarkEngine

# Alias for backward compatibility with existing tests/code
VideoSealEngine = CustomVideoWatermarkEngine
