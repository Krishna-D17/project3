import json
from pathlib import Path
from typing import Union

from media.ffmpeg import (
    FFmpegExecutionError,
    FFmpegNotFoundError,
    get_ffprobe_path,
    run_ffmpeg_command,
)
from media.models import AudioStreamInfo, MediaProbeResult, VideoStreamInfo

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"
}


def parse_fps(fps_str: str) -> float:
    """Parses a frame rate string like '30/1' or '29.97' into float."""
    if not fps_str or fps_str == "0/0":
        return 0.0
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            if float(den) != 0:
                return float(num) / float(den)
        return float(fps_str)
    except Exception:
        return 0.0


def probe_media_file(file_path: Union[str, Path], timeout: float = 30.0) -> MediaProbeResult:
    """Runs ffprobe on a local file and returns structured MediaProbeResult."""
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")

    ffprobe_exe = get_ffprobe_path()
    if not ffprobe_exe:
        raise FFmpegNotFoundError("FFprobe executable is not available. Please install FFmpeg or run scripts/download_ffmpeg.py.")

    cmd = [
        str(ffprobe_exe),
        "-v", "quiet",
        "-print_format", "json",
        show_format_arg := "-show_format",
        "-show_streams",
        str(path)
    ]

    res = run_ffmpeg_command(cmd, timeout=timeout)
    if res.returncode != 0:
        err_msg = res.stderr.strip() if res.stderr else "ffprobe failed to read media file"
        raise FFmpegExecutionError(
            message=f"Failed to probe file '{path.name}': {err_msg}",
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
        )

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by ffprobe for '{path.name}': {e}") from e

    format_data = data.get("format", {})
    streams_data = data.get("streams", [])

    container_format = format_data.get("format_name", "unknown")
    duration_sec = float(format_data.get("duration", 0.0))
    bitrate = int(format_data.get("bit_rate", 0))
    size_bytes = int(format_data.get("size", path.stat().st_size if path.exists() else 0))
    raw_tags = format_data.get("tags", {})

    video_streams = []
    audio_streams = []

    for stream in streams_data:
        codec_type = stream.get("codec_type")
        codec_name = stream.get("codec_name", "unknown")

        if codec_type == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            fps = parse_fps(stream.get("r_frame_rate", stream.get("avg_frame_rate", "0/0")))
            pix_fmt = stream.get("pix_fmt", "unknown")
            nb_frames = stream.get("nb_frames")
            frame_count = int(nb_frames) if nb_frames and nb_frames.isdigit() else None
            v_bitrate = stream.get("bit_rate")
            v_bitrate_int = int(v_bitrate) if v_bitrate and str(v_bitrate).isdigit() else None

            video_streams.append(
                VideoStreamInfo(
                    codec=codec_name,
                    width=width,
                    height=height,
                    fps=fps,
                    pixel_format=pix_fmt,
                    frame_count=frame_count,
                    bitrate=v_bitrate_int,
                )
            )
        elif codec_type == "audio":
            sample_rate = int(stream.get("sample_rate", 0))
            channels = int(stream.get("channels", 0))
            channel_layout = stream.get("channel_layout", "unknown")
            a_bitrate = stream.get("bit_rate")
            a_bitrate_int = int(a_bitrate) if a_bitrate and str(a_bitrate).isdigit() else None

            audio_streams.append(
                AudioStreamInfo(
                    codec=codec_name,
                    sample_rate=sample_rate,
                    channels=channels,
                    channel_layout=channel_layout,
                    bitrate=a_bitrate_int,
                )
            )

    has_video = len(video_streams) > 0
    has_audio = len(audio_streams) > 0
    extension_supported = path.suffix.lower() in SUPPORTED_EXTENSIONS
    is_supported = (has_video or has_audio) and (extension_supported or container_format != "unknown")

    return MediaProbeResult(
        filename=path.name,
        path=str(path),
        size_bytes=size_bytes,
        container_format=container_format,
        duration_seconds=duration_sec,
        bitrate=bitrate,
        has_video=has_video,
        has_audio=has_audio,
        is_supported=is_supported,
        video_streams=video_streams,
        audio_streams=audio_streams,
        raw_format_tags=raw_tags,
    )
