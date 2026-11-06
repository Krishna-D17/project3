from pathlib import Path
from typing import Dict, Union

from media.ffmpeg import get_ffmpeg_status
from media.models import MediaProbeResult
from media.probe import probe_media_file


class MediaService:
    """High-level media engine service."""

    @staticmethod
    def get_engine_status() -> Dict[str, object]:
        return get_ffmpeg_status()

    @staticmethod
    def probe_file(file_path: Union[str, Path], timeout: float = 30.0) -> MediaProbeResult:
        return probe_media_file(file_path, timeout=timeout)

    @staticmethod
    def is_supported_media(file_path: Union[str, Path]) -> bool:
        try:
            res = probe_media_file(file_path)
            return res.is_supported
        except Exception:
            return False
