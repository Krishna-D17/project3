"""
Structured Technical Metadata Extraction via FFprobe.

Implements Layer 2 (Technical Metadata Extraction) for MediaProof.
Extracts comprehensive container, audio, and video stream technical properties
to support downstream metadata drift analysis between registered and candidate media.

Architecture & Contract:
- Extraction only: Emits factual, structured technical metadata. Does not assign
  final MediaProof provenance/tamper classifications.
- Strict nullability: Missing or unavailable fields are reported as None rather than
  fabricated defaults (e.g. an unavailable duration is None, not 0.0).
- Multi-stream support: Fully records stream indices and ordering for containers
  with multiple video, audio, or subtitle/data tracks.
- Explicit failure states: Distinguishes invalid input, missing ffprobe binary,
  timeouts, process execution errors, and malformed JSON.
- Safe execution: Non-shell execution with strict timeouts.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

from media.ffmpeg import (
    FFmpegError,
    FFmpegTimeoutError,
    get_ffmpeg_version,
    get_ffprobe_path,
    run_ffmpeg_command,
)

ALGORITHM_NAME = "ffprobe_technical_metadata"
ALGORITHM_VERSION = "1.0.0"
DEFAULT_PROBE_TIMEOUT: float = 30.0

# Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_SUCCESS_WITH_MISSING_FIELDS = "SUCCESS_WITH_MISSING_FIELDS"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_FFPROBE_UNAVAILABLE = "FFPROBE_UNAVAILABLE"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_PROCESS_FAILURE = "PROCESS_FAILURE"
STATUS_MALFORMED_OUTPUT = "MALFORMED_OUTPUT"


# ==============================================================================
# Helper Parsing Functions (Strict, No Fabricated Defaults)
# ==============================================================================

def parse_fraction_or_float(val: Optional[Union[str, int, float]]) -> Optional[float]:
    """
    Parses a fraction string (e.g., '30/1', '30000/1001') or numeric value into a float.
    Returns None if missing, '0/0', 'N/A', or mathematically invalid. Never fabricates 0.0.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, bool):
            return None
        return float(val) if not (math.isnan(val) or math.isinf(val)) else None

    val_str = str(val).strip()
    if not val_str or val_str in ("0/0", "N/A", "unknown", "none"):
        return None

    try:
        if "/" in val_str:
            parts = val_str.split("/")
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0.0:
                    res = num / den
                    return res if not (math.isnan(res) or math.isinf(res)) else None
                return None
        res = float(val_str)
        return res if not (math.isnan(res) or math.isinf(res)) else None
    except (ValueError, ZeroDivisionError):
        return None


def parse_optional_float(val: Optional[Union[str, int, float]]) -> Optional[float]:
    """Parses an optional float value without fabricating defaults."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val) if not (math.isnan(val) or math.isinf(val)) else None
    val_str = str(val).strip()
    if not val_str or val_str in ("N/A", "unknown", "none"):
        return None
    try:
        res = float(val_str)
        return res if not (math.isnan(res) or math.isinf(res)) else None
    except ValueError:
        return None


def parse_optional_int(val: Optional[Union[str, int, float]]) -> Optional[int]:
    """Parses an optional integer without fabricating defaults."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    val_str = str(val).strip()
    if not val_str or val_str in ("N/A", "unknown", "none"):
        return None
    try:
        # Handles potential string floats such as "128000.0"
        return int(float(val_str))
    except (ValueError, OverflowError):
        return None


def extract_encoder_tag(tags: Optional[Dict[str, Any]]) -> Optional[str]:
    """Safely extracts encoder information from format or stream tags."""
    if not isinstance(tags, dict):
        return None
    for key in ("encoder", "ENCODER", "encoded_by", "ENCODED_BY", "handler_name"):
        val = tags.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class VideoStreamMetadata:
    """Technical properties of a single video stream."""
    stream_index: int
    codec_name: Optional[str] = None
    profile: Optional[str] = None
    level: Optional[int] = None
    pixel_format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sample_aspect_ratio: Optional[str] = None
    display_aspect_ratio: Optional[str] = None
    avg_frame_rate: Optional[float] = None
    r_frame_rate: Optional[float] = None
    color_range: Optional[str] = None
    color_space: Optional[str] = None
    color_transfer: Optional[str] = None
    color_primaries: Optional[str] = None
    frame_count: Optional[int] = None
    time_base: Optional[str] = None
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    raw_tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "codec_name": self.codec_name,
            "profile": self.profile,
            "level": self.level,
            "pixel_format": self.pixel_format,
            "width": self.width,
            "height": self.height,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "display_aspect_ratio": self.display_aspect_ratio,
            "avg_frame_rate": (
                round(self.avg_frame_rate, 4) if self.avg_frame_rate is not None else None
            ),
            "r_frame_rate": (
                round(self.r_frame_rate, 4) if self.r_frame_rate is not None else None
            ),
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "frame_count": self.frame_count,
            "time_base": self.time_base,
            "duration_seconds": (
                round(self.duration_seconds, 4) if self.duration_seconds is not None else None
            ),
            "bitrate": self.bitrate,
            "raw_tags": self.raw_tags,
        }


@dataclass
class AudioStreamMetadata:
    """Technical properties of a single audio stream."""
    stream_index: int
    codec_name: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    raw_tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "codec_name": self.codec_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "duration_seconds": (
                round(self.duration_seconds, 4) if self.duration_seconds is not None else None
            ),
            "bitrate": self.bitrate,
            "raw_tags": self.raw_tags,
        }


@dataclass
class ContainerMetadata:
    """Technical properties of the media container/format."""
    format_name: Optional[str] = None
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    start_time_seconds: Optional[float] = None
    size_bytes: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)
    encoder: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_name": self.format_name,
            "duration_seconds": (
                round(self.duration_seconds, 4) if self.duration_seconds is not None else None
            ),
            "bitrate": self.bitrate,
            "start_time_seconds": (
                round(self.start_time_seconds, 4) if self.start_time_seconds is not None else None
            ),
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "encoder": self.encoder,
        }


@dataclass
class TechnicalMetadataResult:
    """Structured result of technical metadata extraction via FFprobe."""
    available: bool
    status: str
    file_path: str
    algorithm: str = f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}"
    container: Optional[ContainerMetadata] = None
    video_streams: List[VideoStreamMetadata] = field(default_factory=list)
    audio_streams: List[AudioStreamMetadata] = field(default_factory=list)
    video_stream_count: int = 0
    audio_stream_count: int = 0
    other_stream_count: int = 0
    ffprobe_version: Optional[str] = None
    processing_time_seconds: float = 0.0
    raw_metadata: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "file_path": self.file_path,
            "algorithm": self.algorithm,
            "container": self.container.to_dict() if self.container is not None else None,
            "video_streams": [v.to_dict() for v in self.video_streams],
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "video_stream_count": self.video_stream_count,
            "audio_stream_count": self.audio_stream_count,
            "other_stream_count": self.other_stream_count,
            "ffprobe_version": self.ffprobe_version,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "raw_metadata": self.raw_metadata,
            "details": self.details,
        }


def _is_usable_format_dict(format_dict: Any) -> bool:
    """
    Evaluates whether the FFprobe 'format' object contains usable container information.
    Acceptance criteria:
    Must be a dictionary containing at least one non-empty container attribute
    (e.g., format_name, duration, bit_rate, size, start_time, or tags).
    """
    if not isinstance(format_dict, dict) or not format_dict:
        return False
    for key in ("format_name", "duration", "bit_rate", "size", "start_time", "tags"):
        val = format_dict.get(key)
        if val is not None and val != "" and val != {}:
            return True
    return False


def _is_usable_stream_dict(stream: Any) -> bool:
    """
    Evaluates whether an individual stream object contains usable stream information.
    Acceptance criteria:
    Must be a dictionary with an identified codec type or codec name.
    """
    if not isinstance(stream, dict) or not stream:
        return False
    return bool(stream.get("codec_type") or stream.get("codec_name"))


# ==============================================================================
# Extraction Pipeline
# ==============================================================================

def extract_technical_metadata(
    file_path: Union[str, Path],
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    ffprobe_binary: Optional[Union[str, Path]] = None,
    include_raw: bool = False,
) -> TechnicalMetadataResult:
    """
    Runs FFprobe to extract structured technical container and stream metadata.

    Args:
        file_path: Local media path to inspect.
        timeout: Subprocess execution timeout in seconds.
        ffprobe_binary: Optional explicit path to ffprobe executable.
        include_raw: Whether to attach parsed raw FFprobe dictionary in result.

    Returns:
        TechnicalMetadataResult containing container and stream metadata.
    """
    start_time = time.perf_counter()
    target_path_str = str(file_path) if file_path is not None else ""

    # 1. Validate input path existence and file status
    if not file_path:
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_INVALID_INPUT,
            file_path=target_path_str,
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": "File path is empty or None"},
        )

    try:
        resolved_path = Path(file_path).resolve()
    except Exception as err:
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_INVALID_INPUT,
            file_path=target_path_str,
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": f"Invalid path syntax: {err}"},
        )

    if not resolved_path.exists():
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_INVALID_INPUT,
            file_path=str(resolved_path),
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": f"File does not exist: {resolved_path}"},
        )

    if not resolved_path.is_file():
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_INVALID_INPUT,
            file_path=str(resolved_path),
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": f"Path is not a regular file: {resolved_path}"},
        )

    # 2. Resolve FFprobe executable
    if ffprobe_binary is not None:
        ffprobe_exe = Path(ffprobe_binary)
        if not ffprobe_exe.exists() or not ffprobe_exe.is_file():
            return TechnicalMetadataResult(
                available=False,
                status=STATUS_FFPROBE_UNAVAILABLE,
                file_path=str(resolved_path),
                processing_time_seconds=time.perf_counter() - start_time,
                details={"error": f"Specified FFprobe binary not found at: {ffprobe_exe}"},
            )
    else:
        ffprobe_exe = get_ffprobe_path()
        if not ffprobe_exe or not ffprobe_exe.is_file():
            return TechnicalMetadataResult(
                available=False,
                status=STATUS_FFPROBE_UNAVAILABLE,
                file_path=str(resolved_path),
                processing_time_seconds=time.perf_counter() - start_time,
                details={"error": "FFprobe executable not found in tools directory or system PATH"},
            )

    # 3. Detect FFprobe version safely without allowing exceptions to escape or fabricating placeholders
    ffprobe_version: Optional[str] = None
    version_diagnostic: Optional[str] = None
    try:
        raw_version = get_ffmpeg_version(ffprobe_exe)
        if raw_version and raw_version not in ("Unknown", "Unavailable", "unknown", "none"):
            ffprobe_version = raw_version.strip()
        else:
            version_diagnostic = "FFprobe version could not be determined"
    except Exception as v_err:
        version_diagnostic = f"Version detection failed: {v_err}"

    # 4. Execute FFprobe command safely without shell=True
    cmd = [
        str(ffprobe_exe),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(resolved_path),
    ]

    try:
        proc_res = run_ffmpeg_command(cmd, timeout=timeout)
    except FFmpegTimeoutError:
        details_map: Dict[str, Any] = {
            "error": f"FFprobe process timed out after {timeout} seconds",
            "timeout_seconds": timeout,
        }
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_TIMEOUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )
    except Exception as err:
        details_map = {"error": f"Failed to execute FFprobe subprocess: {err}"}
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_PROCESS_FAILURE,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    if proc_res.returncode != 0:
        err_msg = proc_res.stderr.strip() if proc_res.stderr else "FFprobe exited with non-zero return code"
        details_map = {
            "error": err_msg,
            "returncode": proc_res.returncode,
            "stderr": proc_res.stderr,
        }
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_PROCESS_FAILURE,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    # 5. Parse JSON output and validate structural integrity
    try:
        raw_data = json.loads(proc_res.stdout)
    except (json.JSONDecodeError, TypeError) as err:
        details_map = {
            "error": f"Malformed JSON from FFprobe: {err}",
            "stdout_sample": proc_res.stdout[:200] if proc_res.stdout else "",
        }
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_MALFORMED_OUTPUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    # Structural acceptance criteria 1: Top-level JSON value must be an object (dictionary)
    if not isinstance(raw_data, dict):
        details_map = {"error": f"Expected JSON object from FFprobe, got {type(raw_data).__name__}"}
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_MALFORMED_OUTPUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    # Structural acceptance criteria 2: Empty object {} contains no metadata and is unusable
    if not raw_data:
        details_map = {"error": "FFprobe returned an empty JSON object"}
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_MALFORMED_OUTPUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    format_dict = raw_data.get("format")
    streams_list = raw_data.get("streams")

    # Structural acceptance criteria 3: A present format field must be a dictionary
    if format_dict is not None and not isinstance(format_dict, dict):
        details_map = {"error": f"FFprobe 'format' field is not a dictionary ({type(format_dict).__name__})"}
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_MALFORMED_OUTPUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    # Structural acceptance criteria 4: A present streams field must be a list of stream dictionaries
    if streams_list is not None:
        if not isinstance(streams_list, list) or any(not isinstance(s, dict) for s in streams_list):
            details_map = {"error": "FFprobe 'streams' field must be a list of stream dictionaries"}
            if version_diagnostic:
                details_map["version_diagnostic"] = version_diagnostic
            return TechnicalMetadataResult(
                available=False,
                status=STATUS_MALFORMED_OUTPUT,
                file_path=str(resolved_path),
                ffprobe_version=ffprobe_version,
                processing_time_seconds=time.perf_counter() - start_time,
                details=details_map,
            )

    # Structural acceptance criteria 5: Usability evaluation
    # Output must have at least one usable container format OR at least one usable stream.
    # An output missing both usable container-format information and usable stream information
    # is rejected as MALFORMED_OUTPUT.
    has_usable_format = _is_usable_format_dict(format_dict)
    has_usable_streams = bool(streams_list and any(_is_usable_stream_dict(s) for s in streams_list))

    if not has_usable_format and not has_usable_streams:
        details_map = {
            "error": "Output is missing both usable container-format information and usable stream information"
        }
        if version_diagnostic:
            details_map["version_diagnostic"] = version_diagnostic
        return TechnicalMetadataResult(
            available=False,
            status=STATUS_MALFORMED_OUTPUT,
            file_path=str(resolved_path),
            ffprobe_version=ffprobe_version,
            processing_time_seconds=time.perf_counter() - start_time,
            details=details_map,
        )

    format_dict = format_dict or {}
    streams_list = streams_list or []

    # 6. Extract Container Properties
    container_tags = format_dict.get("tags") if isinstance(format_dict.get("tags"), dict) else {}
    container_encoder = extract_encoder_tag(container_tags)

    container_meta = ContainerMetadata(
        format_name=format_dict.get("format_name"),
        duration_seconds=parse_optional_float(format_dict.get("duration")),
        bitrate=parse_optional_int(format_dict.get("bit_rate")),
        start_time_seconds=parse_optional_float(format_dict.get("start_time")),
        size_bytes=parse_optional_int(format_dict.get("size")),
        tags={str(k): str(v) for k, v in container_tags.items()},
        encoder=container_encoder,
    )

    # 6. Extract Streams with preserved indexing and ordering
    video_streams: List[VideoStreamMetadata] = []
    audio_streams: List[AudioStreamMetadata] = []
    other_streams_count = 0

    has_missing_important_field = False
    if not container_meta.format_name:
        has_missing_important_field = True

    for s_idx, stream in enumerate(streams_list):
        if not isinstance(stream, dict):
            continue

        raw_stream_idx = stream.get("index")
        parsed_stream_idx = parse_optional_int(raw_stream_idx)
        actual_stream_index = parsed_stream_idx if parsed_stream_idx is not None else s_idx

        codec_type = stream.get("codec_type")
        stream_tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}

        if codec_type == "video":
            v_codec = stream.get("codec_name")
            v_width = parse_optional_int(stream.get("width"))
            v_height = parse_optional_int(stream.get("height"))

            if not v_codec or v_width is None or v_height is None:
                has_missing_important_field = True

            # Extract color properties
            color_range = stream.get("color_range")
            color_space = stream.get("color_space")
            color_transfer = stream.get("color_transfer")
            color_primaries = stream.get("color_primaries")

            # Frame count
            nb_frames = stream.get("nb_frames")
            frame_cnt = parse_optional_int(nb_frames)
            if frame_cnt is None and "NUMBER_OF_FRAMES" in stream_tags:
                frame_cnt = parse_optional_int(stream_tags.get("NUMBER_OF_FRAMES"))

            video_streams.append(
                VideoStreamMetadata(
                    stream_index=actual_stream_index,
                    codec_name=v_codec,
                    profile=stream.get("profile"),
                    level=parse_optional_int(stream.get("level")),
                    pixel_format=stream.get("pix_fmt"),
                    width=v_width,
                    height=v_height,
                    sample_aspect_ratio=stream.get("sample_aspect_ratio"),
                    display_aspect_ratio=stream.get("display_aspect_ratio"),
                    avg_frame_rate=parse_fraction_or_float(stream.get("avg_frame_rate")),
                    r_frame_rate=parse_fraction_or_float(stream.get("r_frame_rate")),
                    color_range=color_range,
                    color_space=color_space,
                    color_transfer=color_transfer,
                    color_primaries=color_primaries,
                    frame_count=frame_cnt,
                    time_base=stream.get("time_base"),
                    duration_seconds=parse_optional_float(stream.get("duration")),
                    bitrate=parse_optional_int(stream.get("bit_rate")),
                    raw_tags={str(k): str(v) for k, v in stream_tags.items()},
                )
            )
        elif codec_type == "audio":
            a_codec = stream.get("codec_name")
            a_rate = parse_optional_int(stream.get("sample_rate"))
            a_chans = parse_optional_int(stream.get("channels"))

            if not a_codec or a_rate is None or a_chans is None:
                has_missing_important_field = True

            audio_streams.append(
                AudioStreamMetadata(
                    stream_index=actual_stream_index,
                    codec_name=a_codec,
                    sample_rate=a_rate,
                    channels=a_chans,
                    channel_layout=stream.get("channel_layout"),
                    duration_seconds=parse_optional_float(stream.get("duration")),
                    bitrate=parse_optional_int(stream.get("bit_rate")),
                    raw_tags={str(k): str(v) for k, v in stream_tags.items()},
                )
            )
        else:
            other_streams_count += 1

    # Preserve chronological stream index order
    video_streams.sort(key=lambda s: s.stream_index)
    audio_streams.sort(key=lambda s: s.stream_index)

    # Determine status
    # Legitimate media with missing optional/essential fields retains STATUS_SUCCESS_WITH_MISSING_FIELDS.
    # If format-only (no audio/video streams) or stream-only (no container format_name),
    # extraction remains valid and usable (available=True) with STATUS_SUCCESS_WITH_MISSING_FIELDS.
    if has_missing_important_field:
        status = STATUS_SUCCESS_WITH_MISSING_FIELDS
    elif len(video_streams) == 0 and len(audio_streams) == 0:
        status = STATUS_SUCCESS_WITH_MISSING_FIELDS
    else:
        status = STATUS_SUCCESS

    elapsed = time.perf_counter() - start_time

    details_map = {
        "stream_count_total": len(streams_list),
        "exit_code": proc_res.returncode,
    }
    if version_diagnostic:
        details_map["version_diagnostic"] = version_diagnostic

    return TechnicalMetadataResult(
        available=True,
        status=status,
        file_path=str(resolved_path),
        container=container_meta,
        video_streams=video_streams,
        audio_streams=audio_streams,
        video_stream_count=len(video_streams),
        audio_stream_count=len(audio_streams),
        other_stream_count=other_streams_count,
        ffprobe_version=ffprobe_version,
        processing_time_seconds=elapsed,
        raw_metadata=raw_data if include_raw else None,
        details=details_map,
    )


