from pathlib import Path
from typing import Dict, Any, Union

from watermark.audio.factory import get_audio_watermark_engine
from watermark.audio.config import DEFAULT_AUDIO_ENGINE


def detect_audio_watermark(
    input_path: Union[str, Path],
    engine_type: str = DEFAULT_AUDIO_ENGINE,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """High-level function to detect an audio watermark using specified engine type."""
    engine = get_audio_watermark_engine(engine_type)
    result = engine.detect(
        input_path=input_path,
        timeout=timeout,
    )
    return result.to_dict()
