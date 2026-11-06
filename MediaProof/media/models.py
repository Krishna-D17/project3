from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VideoStreamInfo:
    codec: str
    width: int
    height: int
    fps: float
    pixel_format: str
    frame_count: Optional[int] = None
    bitrate: Optional[int] = None


@dataclass
class AudioStreamInfo:
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str
    bitrate: Optional[int] = None


@dataclass
class MediaProbeResult:
    filename: str
    path: str
    size_bytes: int
    container_format: str
    duration_seconds: float
    bitrate: int
    has_video: bool
    has_audio: bool
    is_supported: bool
    video_streams: List[VideoStreamInfo] = field(default_factory=list)
    audio_streams: List[AudioStreamInfo] = field(default_factory=list)
    raw_format_tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "container_format": self.container_format,
            "duration_seconds": self.duration_seconds,
            "bitrate": self.bitrate,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "is_supported": self.is_supported,
            "video_streams": [
                {
                    "codec": v.codec,
                    "width": v.width,
                    "height": v.height,
                    "fps": round(v.fps, 2),
                    "pixel_format": v.pixel_format,
                    "frame_count": v.frame_count,
                    "bitrate": v.bitrate,
                }
                for v in self.video_streams
            ],
            "audio_streams": [
                {
                    "codec": a.codec,
                    "sample_rate": a.sample_rate,
                    "channels": a.channels,
                    "channel_layout": a.channel_layout,
                    "bitrate": a.bitrate,
                }
                for a in self.audio_streams
            ],
            "raw_format_tags": self.raw_format_tags,
        }
