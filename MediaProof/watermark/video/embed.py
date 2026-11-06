from pathlib import Path
from typing import Dict, Optional, Union

from watermark.video.factory import get_video_watermark_engine


def embed_video_watermark(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    asset_id: str,
    engine_type: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, object]:
    """Embeds an invisible MediaProof Asset ID watermark into a video file via the resolved engine."""
    engine = get_video_watermark_engine(engine_type)
    res = engine.embed(input_path, output_path, asset_id, timeout=timeout)
    return res.to_dict()
