from pathlib import Path
from typing import Dict, Any, Union

from watermark.audio.factory import get_audio_watermark_engine
from watermark.audio.config import DEFAULT_AUDIO_ENGINE


def embed_audio_watermark(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    asset_id: str,
    engine_type: str = DEFAULT_AUDIO_ENGINE,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """High-level function to embed an audio watermark using specified engine type."""
    engine = get_audio_watermark_engine(engine_type)
    result = engine.embed(
        input_path=input_path,
        output_path=output_path,
        asset_id=asset_id,
        timeout=timeout,
    )
    return result.to_dict()
