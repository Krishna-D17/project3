from pathlib import Path
from typing import Dict, Optional, Union

from watermark.video.factory import get_video_watermark_engine


def detect_video_watermark(
    input_path: Union[str, Path],
    engine_type: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, object]:
    """Detects and extracts an invisible MediaProof watermark from a video file via the resolved engine."""
    engine = get_video_watermark_engine(engine_type)
    res = engine.detect(input_path, timeout=timeout)
    return res.to_dict()
