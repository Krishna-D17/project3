from typing import Dict, Type

from watermark.audio.base import BaseAudioWatermarkEngine
from watermark.audio.config import DEFAULT_AUDIO_ENGINE
from watermark.audio.custom_engine import CustomAudioWatermarkEngine
from watermark.audio.audioseal_engine import MetaAudioSealEngine

_AUDIO_ENGINE_REGISTRY: Dict[str, Type[BaseAudioWatermarkEngine]] = {
    "custom": CustomAudioWatermarkEngine,
    "audioseal": MetaAudioSealEngine,
}


def get_audio_watermark_engine(engine_type: str = DEFAULT_AUDIO_ENGINE) -> BaseAudioWatermarkEngine:
    """Factory function resolving audio watermarking engine instance by type string."""
    engine_key = engine_type.lower().strip()
    if engine_key not in _AUDIO_ENGINE_REGISTRY:
        supported = ", ".join(list(_AUDIO_ENGINE_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported audio watermarking engine '{engine_type}'. Supported engines are: [{supported}]"
        )
    
    engine_class = _AUDIO_ENGINE_REGISTRY[engine_key]
    return engine_class()
